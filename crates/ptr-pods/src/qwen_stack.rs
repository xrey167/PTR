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
    Qwen2LayerConfig, Qwen2LinearWeights,
};
use crate::{
    KvBackendError, KvContinuationExecutor, KvKeyEncoding, KvTensorDType, KvTensorSchema,
    PagedKvTensorBackend, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, ModelVersion};

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
    Kv(KvBackendError),
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

#[derive(Clone, Debug)]
pub struct Qwen2PagedStack {
    config: Qwen2StackConfig,
    weights: Qwen2StackWeights,
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
        Ok(Self { config, weights })
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

    fn run<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2StackInput,
        require_single_token: bool,
    ) -> Result<Qwen2StackOutput, Qwen2StackError> {
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
        let mut logits = Vec::with_capacity(input.tokens.len());
        for token in input.tokens {
            logits.push(self.forward_token(backend, cache, token)?);
        }
        Ok(Qwen2StackOutput {
            next_position: first_position + logits.len(),
            first_position,
            logits,
        })
    }

    fn forward_token<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        token: u32,
    ) -> Result<Vec<f32>, Qwen2StackError> {
        let layer_config = &self.config.layer;
        let heads = layer_config.num_attention_heads;
        let kv_heads = layer_config.num_key_value_heads;
        let hidden = layer_config.hidden_size;
        let head_dim = hidden / heads;
        let kv_width = kv_heads * head_dim;
        let eps = layer_config.rms_norm_eps;
        let theta = layer_config.rope_theta;
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
            let value = decoder.value.apply(&normalized)?;
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
            let gate = decoder.gate.apply(&post_attention)?;
            let up = decoder.up.apply(&post_attention)?;
            let activated: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| silu(gate) * up)
                .collect();
            let mlp = decoder.down.apply(&activated)?;
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
        self.run(backend, cache, input, false)
    }

    fn decode(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Self::Input,
    ) -> Result<Self::Output, Self::Error> {
        self.run(backend, cache, input, true)
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

/// splitmix64 stream mapped to `f32` in `[-1, 1)`; the only randomness in the reference.
struct SeededStream {
    state: u64,
}

impl SeededStream {
    fn new(seed: u64) -> Self {
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

    fn matrix(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_unit() * scale).collect()
    }

    fn norm(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| 1.0 + self.next_unit() * 0.1).collect()
    }

    fn linear(&mut self, output: usize, input: usize, bias: bool) -> Qwen2LinearWeights {
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
    }
}
