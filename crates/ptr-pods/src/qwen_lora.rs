//! Unmerged LoRA and activated-LoRA (aLoRA) adapters for the reference Qwen2 stack
//! (experiment R009 substrate).
//!
//! A projection `W` with an adapter pair `(A, B)` computes `y = W x + s * B (A x)`; the base
//! weights are never modified, so the base KV rows stay valid for every request that does not
//! carry the adapter. `A` is `rank x in`, `B` is `out x rank`, both drawn from a splitmix64
//! stream seeded by `(weight_seed, adapter identity, generation)`. `B` is drawn like `A`
//! (not zero-initialised): a zero `B` would make the adapter a no-op and every experiment
//! built on it vacuous, so a construction with an all-zero `B` is refused.
//!
//! Kinds:
//!
//! * [`Qwen2AdapterKind::Qkv`]: Q, K and V projections of every layer `>= first_layer`;
//! * [`Qwen2AdapterKind::QOnly`]: the Q projection only;
//! * [`Qwen2AdapterKind::MlpOnly`]: a residual-space correction of the MLP block output,
//!   `s * B (A h)` with `h` the post-attention normalised hidden state (`A: rank x hidden`,
//!   `B: hidden x rank`), for dense and MoE layers alike;
//! * [`Qwen2AdapterKind::Activated`]: aLoRA, QKV at every layer from `first_layer`, active
//!   only for token positions `>= invocation`. Earlier positions run the base path, so their
//!   KV rows are bitwise the base rows.
//!
//! Chain rules live in [`crate::kv_chain`]: K/V deltas bind into `chain(l)` of the adapted
//! layer, Q and MLP deltas only into `chain(l + 1)`.

use crate::qwen_stack::{domain_seed, Qwen2StackConfig, SeededStream};
use crate::KvBackendError;

const LORA_DOMAIN: &str = "ptr.qwen2.lora";

/// R009 pin: LoRA rank.
pub const QWEN2_ADAPTER_PINNED_RANK: usize = 4;
/// R009 pin: LoRA scale `s`.
pub const QWEN2_ADAPTER_PINNED_SCALE: f32 = 2.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen2AdapterKind {
    Qkv,
    QOnly,
    MlpOnly,
    /// aLoRA: QKV, active for positions `>= invocation` only.
    Activated {
        invocation: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen2AdapterError {
    BlankIdentity,
    ZeroRank,
    /// The scale is zero, negative or not finite.
    InvalidScale,
    FirstLayerOutOfRange {
        first_layer: usize,
        num_layers: usize,
    },
    /// The seeded `B` came out all zero (the adapter would be a no-op).
    ZeroDelta,
    /// A pair shape does not match the stack.
    Shape,
    NonFinite,
}

impl std::fmt::Display for Qwen2AdapterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Qwen2AdapterError {}

impl From<KvBackendError> for Qwen2AdapterError {
    fn from(_: KvBackendError) -> Self {
        Self::Shape
    }
}

/// The projection an adapter pair acts on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen2AdapterProjection {
    Q,
    K,
    V,
    Mlp,
}

/// What to build.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2AdapterSpec {
    pub identity: String,
    pub generation: u64,
    pub kind: Qwen2AdapterKind,
    pub first_layer: usize,
    pub rank: usize,
    pub scale: f32,
}

impl Qwen2AdapterSpec {
    /// R009 pin: rank 4, scale 2.0.
    pub fn pinned(
        identity: impl Into<String>,
        generation: u64,
        kind: Qwen2AdapterKind,
        first_layer: usize,
    ) -> Self {
        Self {
            identity: identity.into(),
            generation,
            kind,
            first_layer,
            rank: QWEN2_ADAPTER_PINNED_RANK,
            scale: QWEN2_ADAPTER_PINNED_SCALE,
        }
    }
}

/// One `(A, B)` pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2LoraPair {
    pub rank: usize,
    pub input_size: usize,
    pub output_size: usize,
    /// `rank x input_size`, row-major.
    pub a: Vec<f32>,
    /// `output_size x rank`, row-major.
    pub b: Vec<f32>,
}

impl Qwen2LoraPair {
    fn seeded(stream: &mut SeededStream, rank: usize, input: usize, output: usize) -> Self {
        Self {
            rank,
            input_size: input,
            output_size: output,
            a: stream.matrix(rank * input, (input as f32).sqrt().recip()),
            b: stream.matrix(output * rank, (rank as f32).sqrt().recip()),
        }
    }

    fn validate(&self, input: usize, output: usize, rank: usize) -> Result<(), Qwen2AdapterError> {
        if self.rank != rank
            || self.input_size != input
            || self.output_size != output
            || self.a.len() != rank * input
            || self.b.len() != output * rank
        {
            return Err(Qwen2AdapterError::Shape);
        }
        if !self.a.iter().chain(&self.b).all(|value| value.is_finite()) {
            return Err(Qwen2AdapterError::NonFinite);
        }
        Ok(())
    }

    /// `scale * B (A x)` with a fixed reduction order.
    pub(crate) fn delta(&self, input: &[f32], scale: f32) -> Result<Vec<f32>, KvBackendError> {
        if input.len() != self.input_size {
            return Err(KvBackendError::ShapeMismatch);
        }
        let projected: Vec<f32> = self
            .a
            .chunks_exact(self.input_size)
            .map(|row| row.iter().zip(input).map(|(a, x)| a * x).sum::<f32>())
            .collect();
        Ok(self
            .b
            .chunks_exact(self.rank)
            .map(|row| scale * row.iter().zip(&projected).map(|(b, p)| b * p).sum::<f32>())
            .collect())
    }
}

/// The pairs of one decoder layer; `None` for a projection the kind does not adapt.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2AdapterLayer {
    pub q: Option<Qwen2LoraPair>,
    pub k: Option<Qwen2LoraPair>,
    pub v: Option<Qwen2LoraPair>,
    pub mlp: Option<Qwen2LoraPair>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2Adapter {
    identity: String,
    generation: u64,
    kind: Qwen2AdapterKind,
    first_layer: usize,
    rank: usize,
    scale: f32,
    /// One entry per decoder layer; `None` below `first_layer`.
    layers: Vec<Option<Qwen2AdapterLayer>>,
}

impl Qwen2Adapter {
    /// Seed the adapter from `(weight_seed, identity, generation)` and attach it at every
    /// layer `>= spec.first_layer`.
    pub fn seeded(
        config: &Qwen2StackConfig,
        spec: &Qwen2AdapterSpec,
        weight_seed: u64,
    ) -> Result<Self, Qwen2AdapterError> {
        config.validate().map_err(|_| Qwen2AdapterError::Shape)?;
        if spec.identity.trim().is_empty() {
            return Err(Qwen2AdapterError::BlankIdentity);
        }
        if spec.rank == 0 {
            return Err(Qwen2AdapterError::ZeroRank);
        }
        if !spec.scale.is_finite() || spec.scale <= 0.0 {
            return Err(Qwen2AdapterError::InvalidScale);
        }
        if spec.first_layer >= config.num_layers {
            return Err(Qwen2AdapterError::FirstLayerOutOfRange {
                first_layer: spec.first_layer,
                num_layers: config.num_layers,
            });
        }
        let hidden = config.layer.hidden_size;
        let kv_width =
            config.layer.num_key_value_heads * (hidden / config.layer.num_attention_heads);
        let mut stream = SeededStream::new(domain_seed(
            LORA_DOMAIN,
            &[
                &weight_seed.to_le_bytes(),
                spec.identity.as_bytes(),
                &spec.generation.to_le_bytes(),
            ],
        ));
        let rank = spec.rank;
        let layers = (0..config.num_layers)
            .map(|layer| {
                (layer >= spec.first_layer).then(|| match spec.kind {
                    Qwen2AdapterKind::Qkv | Qwen2AdapterKind::Activated { .. } => {
                        Qwen2AdapterLayer {
                            q: Some(Qwen2LoraPair::seeded(&mut stream, rank, hidden, hidden)),
                            k: Some(Qwen2LoraPair::seeded(&mut stream, rank, hidden, kv_width)),
                            v: Some(Qwen2LoraPair::seeded(&mut stream, rank, hidden, kv_width)),
                            mlp: None,
                        }
                    }
                    Qwen2AdapterKind::QOnly => Qwen2AdapterLayer {
                        q: Some(Qwen2LoraPair::seeded(&mut stream, rank, hidden, hidden)),
                        k: None,
                        v: None,
                        mlp: None,
                    },
                    Qwen2AdapterKind::MlpOnly => Qwen2AdapterLayer {
                        q: None,
                        k: None,
                        v: None,
                        mlp: Some(Qwen2LoraPair::seeded(&mut stream, rank, hidden, hidden)),
                    },
                })
            })
            .collect();
        let adapter = Self {
            identity: spec.identity.clone(),
            generation: spec.generation,
            kind: spec.kind,
            first_layer: spec.first_layer,
            rank,
            scale: spec.scale,
            layers,
        };
        adapter.validate_for(config)?;
        Ok(adapter)
    }

    /// Check shapes, finiteness and that no pair has an all-zero `B`.
    pub(crate) fn validate_for(&self, config: &Qwen2StackConfig) -> Result<(), Qwen2AdapterError> {
        if self.identity.trim().is_empty() {
            return Err(Qwen2AdapterError::BlankIdentity);
        }
        if self.rank == 0 {
            return Err(Qwen2AdapterError::ZeroRank);
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(Qwen2AdapterError::InvalidScale);
        }
        if self.layers.len() != config.num_layers || self.first_layer >= config.num_layers {
            return Err(Qwen2AdapterError::Shape);
        }
        let hidden = config.layer.hidden_size;
        let kv_width =
            config.layer.num_key_value_heads * (hidden / config.layer.num_attention_heads);
        for (index, layer) in self.layers.iter().enumerate() {
            match (layer, index >= self.first_layer) {
                (None, false) => continue,
                (Some(_), true) => {}
                _ => return Err(Qwen2AdapterError::Shape),
            }
            let layer = layer.as_ref().expect("checked above");
            let expected = match self.kind {
                Qwen2AdapterKind::Qkv | Qwen2AdapterKind::Activated { .. } => [
                    (true, hidden),
                    (true, kv_width),
                    (true, kv_width),
                    (false, 0),
                ],
                Qwen2AdapterKind::QOnly => [(true, hidden), (false, 0), (false, 0), (false, 0)],
                Qwen2AdapterKind::MlpOnly => [(false, 0), (false, 0), (false, 0), (true, hidden)],
            };
            for (pair, (present, output)) in [&layer.q, &layer.k, &layer.v, &layer.mlp]
                .into_iter()
                .zip(expected)
            {
                match (pair, present) {
                    (Some(pair), true) => {
                        pair.validate(hidden, output, self.rank)?;
                        if pair.b.iter().all(|value| *value == 0.0) {
                            return Err(Qwen2AdapterError::ZeroDelta);
                        }
                    }
                    (None, false) => {}
                    _ => return Err(Qwen2AdapterError::Shape),
                }
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn kind(&self) -> Qwen2AdapterKind {
        self.kind
    }

    pub fn first_layer(&self) -> usize {
        self.first_layer
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn layer(&self, layer: usize) -> Option<&Qwen2AdapterLayer> {
        self.layers.get(layer).and_then(Option::as_ref)
    }

    /// Whether the adapter changes a stored K/V row (QKV and aLoRA do; Q-only and MLP-only
    /// do not).
    pub fn adapts_kv(&self) -> bool {
        matches!(
            self.kind,
            Qwen2AdapterKind::Qkv | Qwen2AdapterKind::Activated { .. }
        )
    }

    /// The first position the adapter is active at: `0` except for aLoRA.
    pub fn activation_position(&self) -> usize {
        match self.kind {
            Qwen2AdapterKind::Activated { invocation } => invocation,
            _ => 0,
        }
    }

    /// Whether the adapter acts on the token at `position`.
    pub fn active_at(&self, position: usize) -> bool {
        position >= self.activation_position()
    }

    /// A copy whose pair for `projection` at `layer` has one `B` entry shifted by `delta`,
    /// for tests that need identical shapes with different bits. `None` when the adapter has
    /// no such pair.
    pub fn with_perturbed_b(
        mut self,
        layer: usize,
        projection: Qwen2AdapterProjection,
        delta: f32,
    ) -> Option<Self> {
        let layer = self.layers.get_mut(layer)?.as_mut()?;
        let pair = match projection {
            Qwen2AdapterProjection::Q => layer.q.as_mut(),
            Qwen2AdapterProjection::K => layer.k.as_mut(),
            Qwen2AdapterProjection::V => layer.v.as_mut(),
            Qwen2AdapterProjection::Mlp => layer.mlp.as_mut(),
        }?;
        pair.b[0] += delta;
        Some(self)
    }

    /// A copy under another identity and generation with the same weight bits.
    pub fn relabelled(mut self, identity: impl Into<String>, generation: u64) -> Self {
        self.identity = identity.into();
        self.generation = generation;
        self
    }
}
