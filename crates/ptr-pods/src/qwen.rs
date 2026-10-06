use crate::{KvBackendError, PagedKvTensorBackend, TensorRef};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen2Activation {
    Silu,
    Unsupported(String),
}

/// RoPE context-extension scheme of a Qwen2 layer.
#[derive(Clone, Debug, PartialEq)]
pub enum Qwen2RopeScaling {
    None,
    /// Dynamic NTK-aware scaling, the Hugging Face `rope_type = "dynamic"` scheme.
    ///
    /// For a sequence length `seq_len` longer than `original_max_position_embeddings`
    /// (`omp`) the rotary base becomes
    ///
    /// ```text
    /// base' = theta * ((factor * seq_len / omp) - (factor - 1)) ^ (head_dim / (head_dim - 2))
    /// ```
    ///
    /// and for `seq_len <= omp` it stays `theta`. Source: Hugging Face transformers,
    /// `modeling_rope_utils._compute_dynamic_ntk_parameters` (with a full rotary
    /// dimension, so `dim == head_dim`).
    ///
    /// The effective base depends on the sequence length, so a request must fix it once
    /// at admission: see [`Qwen2RopeRegime`].
    DynamicNtk {
        factor: f32,
        original_max_position_embeddings: usize,
    },
    Unsupported(String),
}

/// The effective RoPE regime of one request, fixed at admission (doc 36 section D).
///
/// The regime is the **effective rotary base** computed once, from the admitted maximum
/// sequence length (`prompt_tokens + max_new_tokens`), by [`Qwen2LayerConfig::rope_regime`].
/// It is represented by the exact bits of that `f32` base ([`Self::effective_base_bits`]):
/// two admitted lengths that give the same base (every length `<= omp` gives `theta`) are
/// the same regime and may share cache entries, because every rotation they produce is
/// bitwise identical. The base is computed in `f64` and rounded once to `f32`.
///
/// Prefill and decode of the request both use this one value and never recompute it, so a
/// decode that crosses `omp` stays bitwise equal to a full prefill under the same regime.
/// The regime also records the admitted length: a run that would go beyond it is refused
/// (an actual regime transition needs a fresh admission and a recompute).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen2RopeRegime {
    effective_base: f32,
    admitted_sequence_length: usize,
}

impl Qwen2RopeRegime {
    /// The rotary base every rotation of the request uses.
    pub fn effective_base(&self) -> f32 {
        self.effective_base
    }

    /// Exact bits of [`Self::effective_base`]; what the KV chain binds.
    pub fn effective_base_bits(&self) -> u32 {
        self.effective_base.to_bits()
    }

    /// The maximum sequence length the regime was admitted for (`usize::MAX` when the
    /// layer has no scaling and so no length dependence).
    pub fn admitted_sequence_length(&self) -> usize {
        self.admitted_sequence_length
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2LayerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rope_scaling: Qwen2RopeScaling,
    pub rms_norm_eps: f32,
    pub hidden_act: Qwen2Activation,
    pub sliding_window: Option<usize>,
}

impl Qwen2LayerConfig {
    pub fn head_dim(&self) -> Result<usize, KvBackendError> {
        self.validate()?;
        Ok(self.hidden_size / self.num_attention_heads)
    }

    pub fn validate(&self) -> Result<(), KvBackendError> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.max_position_embeddings == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !(self.rope_theta.is_finite() && self.rope_theta > 0.0)
            || !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0)
            || self.sliding_window == Some(0)
            || !matches!(&self.hidden_act, Qwen2Activation::Silu)
        {
            return Err(KvBackendError::InvalidSchema(
                "invalid Qwen2 layer configuration".into(),
            ));
        }
        let head_dim = self.hidden_size / self.num_attention_heads;
        if !head_dim.is_multiple_of(2) {
            return Err(KvBackendError::InvalidSchema(
                "Qwen2 RoPE requires an even head dimension".into(),
            ));
        }
        match &self.rope_scaling {
            Qwen2RopeScaling::None => {}
            Qwen2RopeScaling::DynamicNtk {
                factor,
                original_max_position_embeddings,
            } => {
                // `head_dim / (head_dim - 2)` needs head_dim > 2.
                if !(factor.is_finite() && *factor >= 1.0)
                    || *original_max_position_embeddings == 0
                    || head_dim <= 2
                {
                    return Err(KvBackendError::InvalidSchema(
                        "invalid Qwen2 dynamic NTK scaling".into(),
                    ));
                }
            }
            Qwen2RopeScaling::Unsupported(_) => {
                return Err(KvBackendError::InvalidSchema(
                    "invalid Qwen2 layer configuration".into(),
                ));
            }
        }
        Ok(())
    }

    /// Fix the effective RoPE regime for a request admitted with at most
    /// `admitted_sequence_length` positions (`prompt_tokens + max_new_tokens`).
    pub fn rope_regime(
        &self,
        admitted_sequence_length: usize,
    ) -> Result<Qwen2RopeRegime, KvBackendError> {
        self.validate()?;
        if admitted_sequence_length == 0 {
            return Err(KvBackendError::InvalidSchema(
                "admitted sequence length must be positive".into(),
            ));
        }
        match &self.rope_scaling {
            Qwen2RopeScaling::None => Ok(self.static_rope_regime()),
            Qwen2RopeScaling::DynamicNtk {
                factor,
                original_max_position_embeddings: omp,
            } => {
                let effective_base = if admitted_sequence_length <= *omp {
                    self.rope_theta
                } else {
                    let head_dim = (self.hidden_size / self.num_attention_heads) as f64;
                    let factor = f64::from(*factor);
                    let scale =
                        factor * admitted_sequence_length as f64 / *omp as f64 - (factor - 1.0);
                    let base = f64::from(self.rope_theta) * scale.powf(head_dim / (head_dim - 2.0));
                    let rounded = base as f32;
                    if !(base.is_finite() && rounded.is_finite() && rounded > 0.0) {
                        return Err(KvBackendError::InvalidSchema(
                            "dynamic NTK base is not finite".into(),
                        ));
                    }
                    rounded
                };
                Ok(Qwen2RopeRegime {
                    effective_base,
                    admitted_sequence_length,
                })
            }
            Qwen2RopeScaling::Unsupported(_) => Err(KvBackendError::InvalidSchema(
                "invalid Qwen2 layer configuration".into(),
            )),
        }
    }

    /// The regime of a layer with no length-dependent scaling: the plain `rope_theta`.
    /// Only meaningful when `rope_scaling` is [`Qwen2RopeScaling::None`].
    pub(crate) fn static_rope_regime(&self) -> Qwen2RopeRegime {
        Qwen2RopeRegime {
            effective_base: self.rope_theta,
            admitted_sequence_length: usize::MAX,
        }
    }

    /// Whether `regime` is exactly what [`Self::rope_regime`] yields for its admitted
    /// length under this configuration.
    pub(crate) fn accepts_regime(&self, regime: &Qwen2RopeRegime) -> bool {
        let expected = match self.rope_scaling {
            Qwen2RopeScaling::None => Ok(self.static_rope_regime()),
            _ => self.rope_regime(regime.admitted_sequence_length),
        };
        expected.is_ok_and(|expected| {
            expected.effective_base_bits() == regime.effective_base_bits()
                && expected.admitted_sequence_length == regime.admitted_sequence_length
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2LinearWeights {
    pub output_size: usize,
    pub input_size: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl Qwen2LinearWeights {
    pub fn validate(&self) -> Result<(), KvBackendError> {
        let expected = self
            .output_size
            .checked_mul(self.input_size)
            .ok_or(KvBackendError::OutOfMemory)?;
        if self.output_size == 0
            || self.input_size == 0
            || self.weight.len() != expected
            || self
                .bias
                .as_ref()
                .is_some_and(|bias| bias.len() != self.output_size)
        {
            return Err(KvBackendError::ShapeMismatch);
        }
        Ok(())
    }

    pub(crate) fn apply(&self, input: &[f32]) -> Result<Vec<f32>, KvBackendError> {
        self.validate()?;
        if input.len() != self.input_size {
            return Err(KvBackendError::ShapeMismatch);
        }
        Ok(self
            .weight
            .chunks_exact(self.input_size)
            .enumerate()
            .map(|(row, weights)| {
                let projected = weights
                    .iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>();
                projected + self.bias.as_ref().map_or(0.0, |bias| bias[row])
            })
            .collect())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2DecoderWeights {
    pub input_layernorm: Vec<f32>,
    pub query: Qwen2LinearWeights,
    pub key: Qwen2LinearWeights,
    pub value: Qwen2LinearWeights,
    pub output: Qwen2LinearWeights,
    pub post_attention_layernorm: Vec<f32>,
    pub gate: Qwen2LinearWeights,
    pub up: Qwen2LinearWeights,
    pub down: Qwen2LinearWeights,
}

impl Qwen2DecoderWeights {
    pub fn validate(&self, config: &Qwen2LayerConfig) -> Result<(), KvBackendError> {
        config.validate()?;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_width = config.num_key_value_heads * head_dim;
        if self.input_layernorm.len() != config.hidden_size
            || self.post_attention_layernorm.len() != config.hidden_size
            || (self.query.output_size, self.query.input_size)
                != (config.hidden_size, config.hidden_size)
            || (self.key.output_size, self.key.input_size) != (kv_width, config.hidden_size)
            || (self.value.output_size, self.value.input_size) != (kv_width, config.hidden_size)
            || (self.output.output_size, self.output.input_size)
                != (config.hidden_size, config.hidden_size)
            || (self.gate.output_size, self.gate.input_size)
                != (config.intermediate_size, config.hidden_size)
            || (self.up.output_size, self.up.input_size)
                != (config.intermediate_size, config.hidden_size)
            || (self.down.output_size, self.down.input_size)
                != (config.hidden_size, config.intermediate_size)
        {
            return Err(KvBackendError::ShapeMismatch);
        }
        for weights in [
            &self.query,
            &self.key,
            &self.value,
            &self.output,
            &self.gate,
            &self.up,
            &self.down,
        ] {
            weights.validate()?;
        }
        Ok(())
    }

    #[cfg(feature = "candle-cuda")]
    pub fn from_safetensors<P: AsRef<std::path::Path>>(
        path: P,
        layer: usize,
        config: &Qwen2LayerConfig,
    ) -> Result<Self, KvBackendError> {
        use candle_core::{DType, Device, Tensor};

        config.validate()?;
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(|error| KvBackendError::Backend(error.to_string()))?;
        let prefix = format!("model.layers.{layer}");
        let contains = |name: &str| {
            tensors
                .tensors()
                .iter()
                .any(|(candidate, _)| candidate == name)
        };
        let load_vector = |name: &str, length: usize| -> Result<Vec<f32>, KvBackendError> {
            let tensor = tensors
                .load(name, &Device::Cpu)
                .map_err(|error| KvBackendError::Backend(error.to_string()))?;
            if tensor.dtype() != DType::F32 || tensor.dims() != [length] {
                return Err(KvBackendError::ShapeMismatch);
            }
            tensor
                .to_vec1::<f32>()
                .map_err(|error| KvBackendError::Backend(error.to_string()))
        };
        let load_linear = |stem: &str,
                           output_size: usize,
                           input_size: usize,
                           bias_expected: bool|
         -> Result<Qwen2LinearWeights, KvBackendError> {
            let weight_name = format!("{prefix}.{stem}.weight");
            let weight: Tensor = tensors
                .load(&weight_name, &Device::Cpu)
                .map_err(|error| KvBackendError::Backend(error.to_string()))?;
            if weight.dtype() != DType::F32 || weight.dims() != [output_size, input_size] {
                return Err(KvBackendError::ShapeMismatch);
            }
            let bias_name = format!("{prefix}.{stem}.bias");
            let bias = if contains(&bias_name) {
                Some(load_vector(&bias_name, output_size)?)
            } else if bias_expected {
                return Err(KvBackendError::InvalidSchema(format!(
                    "missing required Qwen2 tensor {bias_name}"
                )));
            } else {
                None
            };
            Ok(Qwen2LinearWeights {
                output_size,
                input_size,
                weight: weight
                    .to_vec2::<f32>()
                    .map_err(|error| KvBackendError::Backend(error.to_string()))?
                    .into_iter()
                    .flatten()
                    .collect(),
                bias,
            })
        };

        let hidden = config.hidden_size;
        let kv_width = config.num_key_value_heads * (hidden / config.num_attention_heads);
        let intermediate = config.intermediate_size;
        let weights = Self {
            input_layernorm: load_vector(&format!("{prefix}.input_layernorm.weight"), hidden)?,
            query: load_linear("self_attn.q_proj", hidden, hidden, true)?,
            key: load_linear("self_attn.k_proj", kv_width, hidden, true)?,
            value: load_linear("self_attn.v_proj", kv_width, hidden, true)?,
            output: load_linear("self_attn.o_proj", hidden, hidden, false)?,
            post_attention_layernorm: load_vector(
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
            )?,
            gate: load_linear("mlp.gate_proj", intermediate, hidden, false)?,
            up: load_linear("mlp.up_proj", intermediate, hidden, false)?,
            down: load_linear("mlp.down_proj", hidden, intermediate, false)?,
        };
        weights.validate(config)?;
        Ok(weights)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2DecoderInput {
    pub hidden_states: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2DecoderOutput {
    pub hidden_states: Vec<Vec<f32>>,
    pub first_position: usize,
    pub next_position: usize,
}

#[derive(Clone, Debug)]
pub struct Qwen2PagedDecoder {
    config: Qwen2LayerConfig,
    weights: Qwen2DecoderWeights,
}

impl Qwen2PagedDecoder {
    pub fn new(
        config: Qwen2LayerConfig,
        weights: Qwen2DecoderWeights,
    ) -> Result<Self, KvBackendError> {
        weights.validate(&config)?;
        // This decoder applies `rope_theta` as is and has no admission-time regime, so it
        // must not silently ignore a scaling scheme.
        if config.rope_scaling != Qwen2RopeScaling::None {
            return Err(KvBackendError::InvalidSchema(
                "Qwen2PagedDecoder supports no RoPE scaling".into(),
            ));
        }
        Ok(Self { config, weights })
    }

    pub fn config(&self) -> &Qwen2LayerConfig {
        &self.config
    }

    fn run<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Qwen2DecoderInput,
        require_single_token: bool,
    ) -> Result<Qwen2DecoderOutput, KvBackendError> {
        if input.hidden_states.is_empty()
            || (require_single_token && input.hidden_states.len() != 1)
            || input
                .hidden_states
                .iter()
                .any(|token| token.len() != self.config.hidden_size)
        {
            return Err(KvBackendError::ShapeMismatch);
        }
        let before = backend.snapshot(cache)?;
        before.schema.validate_key_encoding()?;
        let head_dim = self.config.head_dim()?;
        // This decoder rotates keys before storing them, so it only serves a cache that
        // declares post-RoPE keys; a pre-RoPE cache would be silently mislabelled.
        if before.schema.key_encoding() != crate::KvKeyEncoding::PostRope
            || before.schema.layer_count != 1
            || before.schema.attention_heads != self.config.num_attention_heads
            || before.schema.key_value_heads != self.config.num_key_value_heads
            || before.schema.head_dim != head_dim
            || before.schema.batch_size != 1
            || before.sequence_length + input.hidden_states.len()
                > self.config.max_position_embeddings
        {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }

        let first_position = before.position_offset + before.sequence_length;
        let mut outputs = Vec::with_capacity(input.hidden_states.len());
        for hidden in input.hidden_states {
            let current = backend.snapshot(cache)?;
            let position = current.position_offset + current.sequence_length;
            let normalized = rms_norm(
                &hidden,
                &self.weights.input_layernorm,
                self.config.rms_norm_eps,
            )?;
            let mut query = self.weights.query.apply(&normalized)?;
            let mut key = self.weights.key.apply(&normalized)?;
            let value = self.weights.value.apply(&normalized)?;
            apply_rope(
                &mut query,
                self.config.num_attention_heads,
                head_dim,
                position,
                self.config.rope_theta,
            )?;
            apply_rope(
                &mut key,
                self.config.num_key_value_heads,
                head_dim,
                position,
                self.config.rope_theta,
            )?;
            let key_value_width = self.config.num_key_value_heads * head_dim;
            backend.append(
                cache,
                &[TensorRef {
                    layer: 0,
                    shape: vec![1, key_value_width],
                    values: key,
                }],
                &[TensorRef {
                    layer: 0,
                    shape: vec![1, key_value_width],
                    values: value,
                }],
            )?;
            let state = backend.snapshot(cache)?;
            let attended = grouped_causal_attention(
                &query,
                &state.layers[0].keys.values,
                &state.layers[0].values.values,
                state.sequence_length,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                head_dim,
                self.config.sliding_window,
            );
            let attention_output = self.weights.output.apply(&attended)?;
            let after_attention = add(&hidden, &attention_output)?;
            let post_attention = rms_norm(
                &after_attention,
                &self.weights.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            let gate = self.weights.gate.apply(&post_attention)?;
            let up = self.weights.up.apply(&post_attention)?;
            let activated: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| silu(gate) * up)
                .collect();
            let mlp = self.weights.down.apply(&activated)?;
            outputs.push(add(&after_attention, &mlp)?);
        }

        Ok(Qwen2DecoderOutput {
            next_position: first_position + outputs.len(),
            first_position,
            hidden_states: outputs,
        })
    }
}

impl<B: PagedKvTensorBackend> crate::KvContinuationExecutor<B> for Qwen2PagedDecoder {
    type Input = Qwen2DecoderInput;
    type Output = Qwen2DecoderOutput;
    type Error = KvBackendError;

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

pub(crate) fn rms_norm(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, KvBackendError> {
    if input.len() != weights.len() || input.is_empty() {
        return Err(KvBackendError::ShapeMismatch);
    }
    let variance = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (variance + epsilon).sqrt().recip();
    Ok(input
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

pub(crate) fn apply_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<(), KvBackendError> {
    if values.len() != heads * head_dim || !head_dim.is_multiple_of(2) {
        return Err(KvBackendError::ShapeMismatch);
    }
    let half = head_dim / 2;
    for head in 0..heads {
        let start = head * head_dim;
        let original = values[start..start + head_dim].to_vec();
        for offset in 0..half {
            let frequency = theta.powf(-((2 * offset) as f32) / head_dim as f32);
            let angle = position as f32 * frequency;
            let cosine = angle.cos();
            let sine = angle.sin();
            let first = original[offset];
            let second = original[half + offset];
            values[start + offset] = first * cosine - second * sine;
            values[start + half + offset] = second * cosine + first * sine;
        }
    }
    Ok(())
}

pub(crate) fn grouped_causal_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    sequence_length: usize,
    heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
) -> Vec<f32> {
    let query_width = heads * head_dim;
    let key_value_width = key_value_heads * head_dim;
    let heads_per_key_value = heads / key_value_heads;
    let first_token = sliding_window
        .map(|window| sequence_length.saturating_sub(window))
        .unwrap_or(0);
    let scale = (head_dim as f32).sqrt();
    let mut output = vec![0.0; query_width];
    for head in 0..heads {
        let head_start = head * head_dim;
        let key_value_head_start = (head / heads_per_key_value) * head_dim;
        let mut scores = Vec::with_capacity(sequence_length - first_token);
        for token in first_token..sequence_length {
            let key_start = token * key_value_width + key_value_head_start;
            scores.push(
                query[head_start..head_start + head_dim]
                    .iter()
                    .zip(&keys[key_start..key_start + head_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    / scale,
            );
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - maximum).exp();
                *score
            })
            .sum::<f32>();
        for (token, score) in (first_token..sequence_length).zip(scores) {
            let value_start = token * key_value_width + key_value_head_start;
            for offset in 0..head_dim {
                output[head_start + offset] += (score / denominator) * values[value_start + offset];
            }
        }
    }
    output
}

pub(crate) fn add(left: &[f32], right: &[f32]) -> Result<Vec<f32>, KvBackendError> {
    if left.len() != right.len() {
        return Err(KvBackendError::ShapeMismatch);
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect())
}

pub(crate) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}
