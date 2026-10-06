//! N-layer Qwen2-style reference stack over a paged KV backend.
//!
//! This is the reference model for experiment R005 (doc 36, P3): token embedding, `N`
//! decoder layers, a final norm and an LM head producing logits. Keys are stored
//! **pre-RoPE** when the cache schema declares [`KvKeyEncoding::PreRope`] and are rotated at
//! attention time with the absolute position, so a layer-0 key row does not depend on where
//! the sequence starts. With [`KvKeyEncoding::PostRope`] they are rotated before storing
//! (the legacy behaviour). Both encodings run the same rotation on the same values, so
//! their logits are bitwise equal at canonical positions.
//!
//! All math is CPU `f32` with a fixed reduction order, and every token is processed one at a
//! time through the same path, so a prefill, a decode and a full prefill of the same tokens
//! are bitwise reproducible.

use crate::qwen::{
    add, apply_rope, grouped_causal_attention, rms_norm, silu, Qwen2DecoderWeights,
    Qwen2LayerConfig, Qwen2LinearWeights, Qwen2RopeRegime, Qwen2RopeScaling,
};
use crate::qwen_lora::{Qwen2Adapter, Qwen2AdapterError};
use crate::qwen_moe::{ExpertLoads, Qwen2Moe, Qwen2MoeError, RoutingCounts, RoutingDeterminism};
use crate::{
    KvBackendError, KvContinuationExecutor, KvKeyEncoding, KvTensorDType, KvTensorSchema,
    PagedKvTensorBackend, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, ModelVersion};
use sha2::{Digest as _, Sha256};

/// Optional per-head RMS normalisation of queries and keys, applied before RoPE.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen2QkNorm {
    Disabled,
    Enabled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2StackConfig {
    pub layer: Qwen2LayerConfig,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub key_encoding: KvKeyEncoding,
    pub qk_norm: Qwen2QkNorm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen2StackError {
    ZeroLayers,
    ZeroVocab,
    HeadsNotDivisibleByKvHeads {
        heads: usize,
        kv_heads: usize,
    },
    /// The layer configuration or a weight shape is invalid.
    InvalidLayer(KvBackendError),
    InvalidWeights(&'static str),
    EmptyInput,
    /// A decode call must carry exactly one token.
    DecodeNeedsOneToken {
        tokens: usize,
    },
    TokenOutOfVocabulary {
        token: u32,
        vocab_size: usize,
    },
    /// The cache stores keys under a different encoding than this stack uses.
    KeyEncodingMismatch {
        expected: KvKeyEncoding,
        found: KvKeyEncoding,
    },
    /// The cache schema does not match this stack's shape.
    CacheSchemaMismatch,
    ContextExceeded {
        required: usize,
        maximum: usize,
    },
    /// The layer scales RoPE by sequence length, so every run needs the regime fixed at
    /// admission ([`Qwen2PagedStack::admit_rope_regime`]); the regime-less trait entry
    /// points cannot choose one.
    RopeRegimeRequired,
    /// The regime is not the one this stack's configuration yields for its admitted length.
    RopeRegimeMismatch,
    /// The run would reach a position beyond the length the regime was admitted for. A
    /// different effective regime needs a new admission and a recompute.
    RopeRegimeExceeded {
        required: usize,
        admitted: usize,
    },
    Kv(KvBackendError),
    /// The bound adapter does not fit this stack.
    Adapter(Qwen2AdapterError),
    /// The MoE does not fit this stack, or routing found no live expert.
    Moe(Qwen2MoeError),
    /// A co-batched run needs one input per cache and an `order` that is a permutation of
    /// the request indices.
    InvalidBatch,
}

impl From<Qwen2AdapterError> for Qwen2StackError {
    fn from(error: Qwen2AdapterError) -> Self {
        Self::Adapter(error)
    }
}

impl From<Qwen2MoeError> for Qwen2StackError {
    fn from(error: Qwen2MoeError) -> Self {
        Self::Moe(error)
    }
}

impl From<KvBackendError> for Qwen2StackError {
    fn from(error: KvBackendError) -> Self {
        Self::Kv(error)
    }
}

impl std::fmt::Display for Qwen2StackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Qwen2StackError {}

impl Qwen2StackConfig {
    pub fn head_dim(&self) -> Result<usize, Qwen2StackError> {
        self.validate()?;
        Ok(self.layer.hidden_size / self.layer.num_attention_heads)
    }

    pub fn validate(&self) -> Result<(), Qwen2StackError> {
        if self.num_layers == 0 {
            return Err(Qwen2StackError::ZeroLayers);
        }
        if self.vocab_size == 0 {
            return Err(Qwen2StackError::ZeroVocab);
        }
        let heads = self.layer.num_attention_heads;
        let kv_heads = self.layer.num_key_value_heads;
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(Qwen2StackError::HeadsNotDivisibleByKvHeads { heads, kv_heads });
        }
        self.layer.validate().map_err(Qwen2StackError::InvalidLayer)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2StackLayerWeights {
    pub decoder: Qwen2DecoderWeights,
    /// Per-head query norm weights (`head_dim` long); required iff QK-norm is enabled.
    pub query_norm: Option<Vec<f32>>,
    /// Per-head key norm weights (`head_dim` long); required iff QK-norm is enabled.
    pub key_norm: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2StackWeights {
    /// `vocab_size * hidden_size`, row per token.
    pub embedding: Vec<f32>,
    pub layers: Vec<Qwen2StackLayerWeights>,
    pub final_norm: Vec<f32>,
    /// `vocab_size x hidden_size`.
    pub lm_head: Qwen2LinearWeights,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen2StackInput {
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2StackOutput {
    /// One logits vector (`vocab_size` long) per input token.
    pub logits: Vec<Vec<f32>>,
    pub first_position: usize,
    pub next_position: usize,
}

/// What a request binds on top of the base stack: an optional adapter. The per-layer MLP
/// kind (Dense or MoE) is a property of the stack itself ([`Qwen2PagedStack::with_moe`]).
/// The default binding is the plain base model and takes exactly the unbound code path.
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen2Binding<'a> {
    pub adapter: Option<&'a Qwen2Adapter>,
}

/// Per-run state threaded through the forward pass.
struct ForwardContext<'a, 'b> {
    adapter: Option<&'a Qwen2Adapter>,
    loads: Option<&'b mut ExpertLoads>,
    trace: Option<&'b mut RoutingCounts>,
}

/// The logits of the requests of a co-batched run, in request-index order.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2CobatchedOutput {
    pub outputs: Vec<Qwen2StackOutput>,
    /// Selections dropped by the capacity rule, over the whole batch.
    pub routing: RoutingCounts,
}

#[derive(Clone, Debug)]
pub struct Qwen2PagedStack {
    config: Qwen2StackConfig,
    weights: Qwen2StackWeights,
    moe: Option<Qwen2Moe>,
}

impl Qwen2PagedStack {
    pub fn new(
        config: Qwen2StackConfig,
        weights: Qwen2StackWeights,
    ) -> Result<Self, Qwen2StackError> {
        config.validate()?;
        let hidden = config.layer.hidden_size;
        let head_dim = hidden / config.layer.num_attention_heads;
        let embedding_len = config
            .vocab_size
            .checked_mul(hidden)
            .ok_or(Qwen2StackError::InvalidWeights("embedding size overflow"))?;
        if weights.embedding.len() != embedding_len {
            return Err(Qwen2StackError::InvalidWeights("embedding shape"));
        }
        if weights.layers.len() != config.num_layers {
            return Err(Qwen2StackError::InvalidWeights("layer count"));
        }
        if weights.final_norm.len() != hidden {
            return Err(Qwen2StackError::InvalidWeights("final norm shape"));
        }
        weights
            .lm_head
            .validate()
            .map_err(Qwen2StackError::InvalidLayer)?;
        if (weights.lm_head.output_size, weights.lm_head.input_size) != (config.vocab_size, hidden)
        {
            return Err(Qwen2StackError::InvalidWeights("lm head shape"));
        }
        for layer in &weights.layers {
            layer
                .decoder
                .validate(&config.layer)
                .map_err(Qwen2StackError::InvalidLayer)?;
            let norms_ok = match config.qk_norm {
                Qwen2QkNorm::Disabled => layer.query_norm.is_none() && layer.key_norm.is_none(),
                Qwen2QkNorm::Enabled => {
                    layer
                        .query_norm
                        .as_ref()
                        .is_some_and(|w| w.len() == head_dim)
                        && layer.key_norm.as_ref().is_some_and(|w| w.len() == head_dim)
                }
            };
            if !norms_ok {
                return Err(Qwen2StackError::InvalidWeights("qk norm weights"));
            }
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        let linear_finite = |linear: &Qwen2LinearWeights| {
            finite(&linear.weight) && linear.bias.as_deref().is_none_or(finite)
        };
        let layers_finite = weights.layers.iter().all(|layer| {
            let decoder = &layer.decoder;
            finite(&decoder.input_layernorm)
                && finite(&decoder.post_attention_layernorm)
                && [
                    &decoder.query,
                    &decoder.key,
                    &decoder.value,
                    &decoder.output,
                    &decoder.gate,
                    &decoder.up,
                    &decoder.down,
                ]
                .into_iter()
                .all(linear_finite)
                && layer.query_norm.as_deref().is_none_or(finite)
                && layer.key_norm.as_deref().is_none_or(finite)
        });
        if !(finite(&weights.embedding)
            && finite(&weights.final_norm)
            && linear_finite(&weights.lm_head)
            && layers_finite)
        {
            return Err(Qwen2StackError::InvalidWeights("non-finite weight"));
        }
        Ok(Self {
            config,
            weights,
            moe: None,
        })
    }

    /// Replace the dense MLP of the layers the MoE names by routed experts. The dense
    /// weights stay in the stack unchanged (unused by those layers).
    pub fn with_moe(mut self, moe: Qwen2Moe) -> Result<Self, Qwen2StackError> {
        moe.validate_for(&self.config)?;
        self.moe = Some(moe);
        Ok(self)
    }

    pub fn moe(&self) -> Option<&Qwen2Moe> {
        self.moe.as_ref()
    }

    /// Whether outputs can depend on what else runs in the same batch.
    pub fn routing_determinism(&self) -> RoutingDeterminism {
        self.moe
            .as_ref()
            .map_or(RoutingDeterminism::NotRouted, Qwen2Moe::routing_determinism)
    }

    /// The weights, read-only (for canonical digests such as the KV chain).
    pub fn weights(&self) -> &Qwen2StackWeights {
        &self.weights
    }

    /// Deterministic seeded reference weights (splitmix64 stream, fixed tensor order), the
    /// model R005 pins by `weight_seed`. Not trained weights.
    pub fn reference(config: Qwen2StackConfig, seed: u64) -> Result<Self, Qwen2StackError> {
        config.validate()?;
        let mut stream = SeededStream::new(seed);
        let layer = &config.layer;
        let hidden = layer.hidden_size;
        let head_dim = hidden / layer.num_attention_heads;
        let kv_width = layer.num_key_value_heads * head_dim;
        let embedding = stream.matrix(config.vocab_size * hidden, 1.0);
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let decoder = Qwen2DecoderWeights {
                input_layernorm: stream.norm(hidden),
                query: stream.linear(hidden, hidden, true),
                key: stream.linear(kv_width, hidden, true),
                value: stream.linear(kv_width, hidden, true),
                output: stream.linear(hidden, hidden, false),
                post_attention_layernorm: stream.norm(hidden),
                gate: stream.linear(layer.intermediate_size, hidden, false),
                up: stream.linear(layer.intermediate_size, hidden, false),
                down: stream.linear(hidden, layer.intermediate_size, false),
            };
            let (query_norm, key_norm) = match config.qk_norm {
                Qwen2QkNorm::Disabled => (None, None),
                Qwen2QkNorm::Enabled => (Some(stream.norm(head_dim)), Some(stream.norm(head_dim))),
            };
            layers.push(Qwen2StackLayerWeights {
                decoder,
                query_norm,
                key_norm,
            });
        }
        let final_norm = stream.norm(hidden);
        let lm_head = stream.linear(config.vocab_size, hidden, false);
        Self::new(
            config,
            Qwen2StackWeights {
                embedding,
                layers,
                final_norm,
                lm_head,
            },
        )
    }

    pub fn config(&self) -> &Qwen2StackConfig {
        &self.config
    }

    /// The cache schema this stack requires, declaring the stack's key encoding.
    pub fn kv_schema(
        &self,
        model: ModelVersion,
        adapter: AdapterVersion,
        device: DeviceId,
    ) -> Result<KvTensorSchema, Qwen2StackError> {
        Ok(KvTensorSchema {
            model,
            adapter,
            device,
            layer_count: self.config.num_layers,
            attention_heads: self.config.layer.num_attention_heads,
            key_value_heads: self.config.layer.num_key_value_heads,
            head_dim: self.config.head_dim()?,
            batch_size: 1,
            dtype: KvTensorDType::F32,
        }
        .with_key_encoding(self.config.key_encoding))
    }

    /// Fix the effective RoPE regime of a request admitted with `prompt_tokens` prompt
    /// tokens and at most `max_new_tokens` generated tokens (doc 36 section D). Compute it
    /// once per request and pass the same value to every
    /// [`Self::prefill_in_regime`] / [`Self::decode_in_regime`] of that request.
    pub fn admit_rope_regime(
        &self,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<Qwen2RopeRegime, Qwen2StackError> {
        let admitted =
            prompt_tokens
                .checked_add(max_new_tokens)
                .ok_or(Qwen2StackError::ContextExceeded {
                    required: usize::MAX,
                    maximum: self.config.layer.max_position_embeddings,
                })?;
        self.config
            .layer
            .rope_regime(admitted)
            .map_err(Qwen2StackError::InvalidLayer)
    }

    /// The regime of a stack without length-dependent scaling.
    fn static_regime(&self) -> Result<Qwen2RopeRegime, Qwen2StackError> {
        match self.config.layer.rope_scaling {
            Qwen2RopeScaling::None => Ok(self.config.layer.static_rope_regime()),
            _ => Err(Qwen2StackError::RopeRegimeRequired),
        }
    }

    /// Prefill under a regime fixed at admission. Never recomputes the regime.
    pub fn prefill_in_regime<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2StackInput,
        regime: &Qwen2RopeRegime,
    ) -> Result<Qwen2StackOutput, Qwen2StackError> {
        self.run(
            backend,
            cache,
            input,
            false,
            regime,
            &Qwen2Binding::default(),
            None,
        )
    }

    /// Decode one token under the regime fixed at admission. Never recomputes the regime.
    pub fn decode_in_regime<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2StackInput,
        regime: &Qwen2RopeRegime,
    ) -> Result<Qwen2StackOutput, Qwen2StackError> {
        self.run(
            backend,
            cache,
            input,
            true,
            regime,
            &Qwen2Binding::default(),
            None,
        )
    }

    /// Validate one request against its cache and the admitted regime; returns the position
    /// of its first token.
    fn admit_request<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &B::Cache,
        input: &Qwen2StackInput,
        require_single_token: bool,
        regime: &Qwen2RopeRegime,
        binding: &Qwen2Binding<'_>,
    ) -> Result<usize, Qwen2StackError> {
        if let Some(adapter) = binding.adapter {
            adapter.validate_for(&self.config)?;
        }
        if !self.config.layer.accepts_regime(regime) {
            return Err(Qwen2StackError::RopeRegimeMismatch);
        }
        if input.tokens.is_empty() {
            return Err(Qwen2StackError::EmptyInput);
        }
        if require_single_token && input.tokens.len() != 1 {
            return Err(Qwen2StackError::DecodeNeedsOneToken {
                tokens: input.tokens.len(),
            });
        }
        if let Some(token) = input
            .tokens
            .iter()
            .find(|token| **token as usize >= self.config.vocab_size)
        {
            return Err(Qwen2StackError::TokenOutOfVocabulary {
                token: *token,
                vocab_size: self.config.vocab_size,
            });
        }
        let before = backend.snapshot(cache)?;
        before.schema.validate_key_encoding()?;
        let found = before.schema.key_encoding();
        if found != self.config.key_encoding {
            return Err(Qwen2StackError::KeyEncodingMismatch {
                expected: self.config.key_encoding,
                found,
            });
        }
        let layer = &self.config.layer;
        if before.schema.layer_count != self.config.num_layers
            || before.schema.attention_heads != layer.num_attention_heads
            || before.schema.key_value_heads != layer.num_key_value_heads
            || before.schema.head_dim != self.config.head_dim()?
            || before.schema.batch_size != 1
            || before.schema.dtype != KvTensorDType::F32
        {
            return Err(Qwen2StackError::CacheSchemaMismatch);
        }
        let required = before.sequence_length + input.tokens.len();
        if required > layer.max_position_embeddings {
            return Err(Qwen2StackError::ContextExceeded {
                required,
                maximum: layer.max_position_embeddings,
            });
        }

        let first_position = before.position_offset + before.sequence_length;
        let end_position = first_position + input.tokens.len();
        if end_position > regime.admitted_sequence_length() {
            return Err(Qwen2StackError::RopeRegimeExceeded {
                required: end_position,
                admitted: regime.admitted_sequence_length(),
            });
        }
        Ok(first_position)
    }

    fn run<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2StackInput,
        require_single_token: bool,
        regime: &Qwen2RopeRegime,
        binding: &Qwen2Binding<'_>,
        trace: Option<&mut RoutingCounts>,
    ) -> Result<Qwen2StackOutput, Qwen2StackError> {
        let first_position = self.admit_request(
            backend,
            cache,
            &input,
            require_single_token,
            regime,
            binding,
        )?;
        let mut context = ForwardContext {
            adapter: binding.adapter,
            loads: None,
            trace,
        };
        let mut logits = Vec::with_capacity(input.tokens.len());
        for token in input.tokens {
            logits.push(self.forward_token(
                backend,
                cache,
                token,
                regime.effective_base(),
                &mut context,
            )?);
        }
        Ok(Qwen2StackOutput {
            next_position: first_position + logits.len(),
            first_position,
            logits,
        })
    }

    /// Prefill under a regime and a binding (adapter). `Qwen2Binding::default()` is the
    /// plain base path.
    pub fn prefill_bound_in_regime<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2StackInput,
        regime: &Qwen2RopeRegime,
        binding: &Qwen2Binding<'_>,
    ) -> Result<Qwen2StackOutput, Qwen2StackError> {
        self.run(backend, cache, input, false, regime, binding, None)
    }

    /// Decode one token under a regime and a binding.
    pub fn decode_bound_in_regime<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2StackInput,
        regime: &Qwen2RopeRegime,
        binding: &Qwen2Binding<'_>,
    ) -> Result<Qwen2StackOutput, Qwen2StackError> {
        self.run(backend, cache, input, true, regime, binding, None)
    }

    /// Prefill and report how often each expert was selected on this input, per layer.
    /// Use it to pick the most-routed expert of a layer on a probe prompt.
    pub fn prefill_traced_in_regime<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2StackInput,
        regime: &Qwen2RopeRegime,
        binding: &Qwen2Binding<'_>,
    ) -> Result<(Qwen2StackOutput, RoutingCounts), Qwen2StackError> {
        let mut counts = self
            .moe
            .as_ref()
            .map(RoutingCounts::new)
            .unwrap_or_default();
        let output = self.run(
            backend,
            cache,
            input,
            false,
            regime,
            binding,
            Some(&mut counts),
        )?;
        Ok((output, counts))
    }

    /// Prefill several requests **co-batched**: step `t` runs token `t` of every request
    /// that still has one, in the order given by `order` (a permutation of the request
    /// indices), and the experts' per-step loads are shared by the whole step. With
    /// [`crate::Qwen2RoutingMode::Dropless`] the result is independent of `order`; with
    /// [`crate::Qwen2RoutingMode::BatchVariantCapacity`] it is not.
    pub fn prefill_cobatched_in_regime<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        caches: &mut [B::Cache],
        inputs: Vec<Qwen2StackInput>,
        order: &[usize],
        regime: &Qwen2RopeRegime,
        binding: &Qwen2Binding<'_>,
    ) -> Result<Qwen2CobatchedOutput, Qwen2StackError> {
        let count = caches.len();
        let mut seen = vec![false; count];
        if inputs.len() != count
            || order.len() != count
            || order
                .iter()
                .any(|index| *index >= count || std::mem::replace(&mut seen[*index], true))
        {
            return Err(Qwen2StackError::InvalidBatch);
        }
        let mut firsts = Vec::with_capacity(count);
        for (cache, input) in caches.iter().zip(&inputs) {
            firsts.push(self.admit_request(backend, cache, input, false, regime, binding)?);
        }
        let mut counts = self
            .moe
            .as_ref()
            .map(RoutingCounts::new)
            .unwrap_or_default();
        let mut logits: Vec<Vec<Vec<f32>>> = vec![Vec::new(); count];
        let steps = inputs
            .iter()
            .map(|input| input.tokens.len())
            .max()
            .unwrap_or(0);
        for step in 0..steps {
            let mut loads = self.moe.as_ref().map(ExpertLoads::new);
            for request in order {
                let Some(token) = inputs[*request].tokens.get(step) else {
                    continue;
                };
                let mut context = ForwardContext {
                    adapter: binding.adapter,
                    loads: loads.as_mut(),
                    trace: Some(&mut counts),
                };
                let row = self.forward_token(
                    backend,
                    &mut caches[*request],
                    *token,
                    regime.effective_base(),
                    &mut context,
                )?;
                logits[*request].push(row);
            }
        }
        let outputs = logits
            .into_iter()
            .zip(firsts)
            .map(|(logits, first_position)| Qwen2StackOutput {
                next_position: first_position + logits.len(),
                first_position,
                logits,
            })
            .collect();
        Ok(Qwen2CobatchedOutput {
            outputs,
            routing: counts,
        })
    }

    fn forward_token<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        token: u32,
        theta: f32,
        context: &mut ForwardContext<'_, '_>,
    ) -> Result<Vec<f32>, Qwen2StackError> {
        let layer_config = &self.config.layer;
        let heads = layer_config.num_attention_heads;
        let kv_heads = layer_config.num_key_value_heads;
        let hidden = layer_config.hidden_size;
        let head_dim = hidden / heads;
        let kv_width = kv_heads * head_dim;
        let eps = layer_config.rms_norm_eps;
        let encoding = self.config.key_encoding;

        let state = backend.snapshot(cache)?;
        let cached = state.sequence_length;
        let position = state.position_offset + cached;
        let start = token as usize * hidden;
        let mut residual = self.weights.embedding[start..start + hidden].to_vec();
        let mut stored_keys = Vec::with_capacity(self.config.num_layers);
        let mut stored_values = Vec::with_capacity(self.config.num_layers);

        for (index, weights) in self.weights.layers.iter().enumerate() {
            let decoder = &weights.decoder;
            let normalized = rms_norm(&residual, &decoder.input_layernorm, eps)?;
            let mut query = decoder.query.apply(&normalized)?;
            let mut key = decoder.key.apply(&normalized)?;
            let mut value = decoder.value.apply(&normalized)?;
            // Unmerged adapter deltas, y = W x + s B (A x); an aLoRA acts from its
            // invocation position on, earlier positions take the base path untouched.
            let adapter = context
                .adapter
                .filter(|adapter| adapter.active_at(position));
            let adapter_layer = adapter.and_then(|adapter| adapter.layer(index));
            let adapter_scale = adapter.map_or(0.0, Qwen2Adapter::scale);
            if let Some(pair) = adapter_layer.and_then(|layer| layer.q.as_ref()) {
                query = add(&query, &pair.delta(&normalized, adapter_scale)?)?;
            }
            if let Some(pair) = adapter_layer.and_then(|layer| layer.k.as_ref()) {
                key = add(&key, &pair.delta(&normalized, adapter_scale)?)?;
            }
            if let Some(pair) = adapter_layer.and_then(|layer| layer.v.as_ref()) {
                value = add(&value, &pair.delta(&normalized, adapter_scale)?)?;
            }
            if let (Some(query_norm), Some(key_norm)) = (&weights.query_norm, &weights.key_norm) {
                per_head_norm(&mut query, head_dim, query_norm, eps)?;
                per_head_norm(&mut key, head_dim, key_norm, eps)?;
            }
            apply_rope(&mut query, heads, head_dim, position, theta)?;
            let mut rotated_key = key.clone();
            apply_rope(&mut rotated_key, kv_heads, head_dim, position, theta)?;

            // Keys and values of the cached tokens plus the current one, in token order.
            let mut keys = Vec::with_capacity((cached + 1) * kv_width);
            let mut values = Vec::with_capacity((cached + 1) * kv_width);
            if cached > 0 {
                let snapshot_layer = state
                    .layers
                    .get(index)
                    .ok_or(KvBackendError::SnapshotSchemaMismatch)?;
                if snapshot_layer.keys.values.len() != cached * kv_width
                    || snapshot_layer.values.values.len() != cached * kv_width
                {
                    return Err(KvBackendError::ShapeMismatch.into());
                }
                match encoding {
                    KvKeyEncoding::PostRope => keys.extend_from_slice(&snapshot_layer.keys.values),
                    KvKeyEncoding::PreRope => {
                        for (row, chunk) in snapshot_layer
                            .keys
                            .values
                            .chunks_exact(kv_width)
                            .enumerate()
                        {
                            let mut rotated = chunk.to_vec();
                            apply_rope(
                                &mut rotated,
                                kv_heads,
                                head_dim,
                                state.position_offset + row,
                                theta,
                            )?;
                            keys.extend_from_slice(&rotated);
                        }
                    }
                }
                values.extend_from_slice(&snapshot_layer.values.values);
            }
            keys.extend_from_slice(&rotated_key);
            values.extend_from_slice(&value);

            let attended = grouped_causal_attention(
                &query,
                &keys,
                &values,
                cached + 1,
                heads,
                kv_heads,
                head_dim,
                layer_config.sliding_window,
            );
            let attention_output = decoder.output.apply(&attended)?;
            let after_attention = add(&residual, &attention_output)?;
            let post_attention =
                rms_norm(&after_attention, &decoder.post_attention_layernorm, eps)?;
            let mut mlp = match self.moe.as_ref().filter(|moe| moe.is_moe_layer(index)) {
                Some(moe) => moe.apply(
                    index,
                    &post_attention,
                    context.loads.as_deref_mut(),
                    context.trace.as_deref_mut(),
                )?,
                None => {
                    let gate = decoder.gate.apply(&post_attention)?;
                    let up = decoder.up.apply(&post_attention)?;
                    let activated: Vec<f32> = gate
                        .into_iter()
                        .zip(up)
                        .map(|(gate, up)| silu(gate) * up)
                        .collect();
                    decoder.down.apply(&activated)?
                }
            };
            if let Some(pair) = adapter_layer.and_then(|layer| layer.mlp.as_ref()) {
                mlp = add(&mlp, &pair.delta(&post_attention, adapter_scale)?)?;
            }
            residual = add(&after_attention, &mlp)?;

            stored_keys.push(TensorRef {
                layer: index,
                shape: vec![1, kv_width],
                values: match encoding {
                    KvKeyEncoding::PostRope => rotated_key,
                    KvKeyEncoding::PreRope => key,
                },
            });
            stored_values.push(TensorRef {
                layer: index,
                shape: vec![1, kv_width],
                values: value,
            });
        }
        backend.append(cache, &stored_keys, &stored_values)?;

        let final_hidden = rms_norm(&residual, &self.weights.final_norm, eps)?;
        Ok(self.weights.lm_head.apply(&final_hidden)?)
    }
}

impl<B: PagedKvTensorBackend> KvContinuationExecutor<B> for Qwen2PagedStack {
    type Input = Qwen2StackInput;
    type Output = Qwen2StackOutput;
    type Error = Qwen2StackError;

    fn prefill(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Self::Input,
    ) -> Result<Self::Output, Self::Error> {
        let regime = self.static_regime()?;
        self.run(
            backend,
            cache,
            input,
            false,
            &regime,
            &Qwen2Binding::default(),
            None,
        )
    }

    fn decode(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Self::Input,
    ) -> Result<Self::Output, Self::Error> {
        let regime = self.static_regime()?;
        self.run(
            backend,
            cache,
            input,
            true,
            &regime,
            &Qwen2Binding::default(),
            None,
        )
    }
}

fn per_head_norm(
    values: &mut [f32],
    head_dim: usize,
    weights: &[f32],
    epsilon: f32,
) -> Result<(), KvBackendError> {
    for head in values.chunks_exact_mut(head_dim) {
        let normalized = rms_norm(head, weights, epsilon)?;
        head.copy_from_slice(&normalized);
    }
    Ok(())
}

/// A stream seed for a named domain: SHA-256 over the domain and the length-prefixed parts,
/// first 8 bytes. Keeps independent seeded tensors (adapters, experts) apart from the dense
/// reference stream.
pub(crate) fn domain_seed(domain: &str, parts: &[&[u8]]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain.as_bytes());
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("a digest has 32 bytes"))
}

/// splitmix64 stream mapped to `f32` in `[-1, 1)`; the only randomness in the reference.
pub(crate) struct SeededStream {
    state: u64,
}

impl SeededStream {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 24 bits give an exact f32 in [0, 1).
        ((z >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }

    pub(crate) fn matrix(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_unit() * scale).collect()
    }

    fn norm(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| 1.0 + self.next_unit() * 0.1).collect()
    }

    pub(crate) fn linear(&mut self, output: usize, input: usize, bias: bool) -> Qwen2LinearWeights {
        let scale = (input as f32).sqrt().recip();
        let weight = self.matrix(output * input, scale);
        let bias = bias.then(|| self.matrix(output, 0.1));
        Qwen2LinearWeights {
            output_size: output,
            input_size: input,
            weight,
            bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Qwen2Activation, Qwen2RopeScaling};

    fn config(encoding: KvKeyEncoding) -> Qwen2StackConfig {
        Qwen2StackConfig {
            layer: Qwen2LayerConfig {
                hidden_size: 8,
                intermediate_size: 16,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                max_position_embeddings: 32,
                rope_theta: 10_000.0,
                rope_scaling: Qwen2RopeScaling::None,
                rms_norm_eps: 1e-6,
                hidden_act: Qwen2Activation::Silu,
                sliding_window: None,
            },
            num_layers: 2,
            vocab_size: 11,
            key_encoding: encoding,
            qk_norm: Qwen2QkNorm::Disabled,
        }
    }

    #[test]
    fn seeded_reference_is_deterministic_and_seed_sensitive() {
        let a = Qwen2PagedStack::reference(config(KvKeyEncoding::PreRope), 7).unwrap();
        let b = Qwen2PagedStack::reference(config(KvKeyEncoding::PreRope), 7).unwrap();
        let c = Qwen2PagedStack::reference(config(KvKeyEncoding::PreRope), 8).unwrap();
        assert_eq!(a.weights, b.weights);
        assert_ne!(a.weights, c.weights);
    }

    #[test]
    fn config_rejects_invalid_shapes_with_typed_errors() {
        let mut bad = config(KvKeyEncoding::PreRope);
        bad.num_layers = 0;
        assert_eq!(bad.validate(), Err(Qwen2StackError::ZeroLayers));
        let mut bad = config(KvKeyEncoding::PreRope);
        bad.vocab_size = 0;
        assert_eq!(bad.validate(), Err(Qwen2StackError::ZeroVocab));
        let mut bad = config(KvKeyEncoding::PreRope);
        bad.layer.num_key_value_heads = 3;
        assert!(matches!(
            bad.validate(),
            Err(Qwen2StackError::HeadsNotDivisibleByKvHeads { .. })
        ));
    }

    #[test]
    fn weights_with_wrong_shape_are_rejected() {
        let stack = Qwen2PagedStack::reference(config(KvKeyEncoding::PostRope), 1).unwrap();
        let mut weights = stack.weights.clone();
        weights.embedding.pop();
        assert_eq!(
            Qwen2PagedStack::new(stack.config.clone(), weights).unwrap_err(),
            Qwen2StackError::InvalidWeights("embedding shape")
        );
        let mut weights = stack.weights.clone();
        weights.layers[0].query_norm = Some(vec![1.0; 4]);
        assert_eq!(
            Qwen2PagedStack::new(stack.config.clone(), weights).unwrap_err(),
            Qwen2StackError::InvalidWeights("qk norm weights")
        );
        let mut weights = stack.weights.clone();
        weights.layers[1].decoder.gate.weight[0] = f32::NAN;
        assert_eq!(
            Qwen2PagedStack::new(stack.config.clone(), weights).unwrap_err(),
            Qwen2StackError::InvalidWeights("non-finite weight")
        );
    }
}
