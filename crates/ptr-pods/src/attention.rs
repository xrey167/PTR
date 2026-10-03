use crate::{KvBackendError, PagedKvTensorBackend, TensorRef};

#[derive(Clone, Debug, PartialEq)]
pub struct CausalAttentionInput {
    pub tokens: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalAttentionOutput {
    pub hidden_states: Vec<Vec<f32>>,
    pub first_position: usize,
    pub next_position: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CausalAttentionWeights {
    pub model_dim: usize,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub output: Vec<f32>,
}

impl CausalAttentionWeights {
    pub fn validate(&self) -> Result<(), KvBackendError> {
        let matrix = self
            .model_dim
            .checked_mul(self.model_dim)
            .ok_or(KvBackendError::OutOfMemory)?;
        if self.model_dim == 0
            || self.query.len() != matrix
            || self.key.len() != matrix
            || self.value.len() != matrix
            || self.output.len() != matrix
        {
            return Err(KvBackendError::InvalidSchema(
                "causal-attention weights must be four square F32 matrices".into(),
            ));
        }
        Ok(())
    }

    pub fn identity(model_dim: usize) -> Result<Self, KvBackendError> {
        if model_dim == 0 {
            return Err(KvBackendError::InvalidSchema("zero model dimension".into()));
        }
        let mut matrix = vec![0.0; model_dim * model_dim];
        for index in 0..model_dim {
            matrix[index * model_dim + index] = 1.0;
        }
        Ok(Self {
            model_dim,
            query: matrix.clone(),
            key: matrix.clone(),
            value: matrix.clone(),
            output: matrix,
        })
    }

    #[cfg(feature = "candle-cuda")]
    pub fn from_safetensors<P: AsRef<std::path::Path>>(
        path: P,
        query_name: &str,
        key_name: &str,
        value_name: &str,
        output_name: &str,
    ) -> Result<Self, KvBackendError> {
        use candle_core::{DType, Device};

        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(|error| KvBackendError::Backend(error.to_string()))?;
        let load = |name: &str| -> Result<(usize, Vec<f32>), KvBackendError> {
            let tensor = tensors
                .load(name, &Device::Cpu)
                .map_err(|error| KvBackendError::Backend(error.to_string()))?;
            if tensor.dtype() != DType::F32 || tensor.dims().len() != 2 {
                return Err(KvBackendError::DtypeMismatch);
            }
            let dims = tensor.dims();
            if dims[0] == 0 || dims[0] != dims[1] {
                return Err(KvBackendError::ShapeMismatch);
            }
            let values = tensor
                .to_vec2::<f32>()
                .map_err(|error| KvBackendError::Backend(error.to_string()))?
                .into_iter()
                .flatten()
                .collect();
            Ok((dims[0], values))
        };
        let (model_dim, query) = load(query_name)?;
        let (key_dim, key) = load(key_name)?;
        let (value_dim, value) = load(value_name)?;
        let (output_dim, output) = load(output_name)?;
        if [key_dim, value_dim, output_dim]
            .into_iter()
            .any(|dimension| dimension != model_dim)
        {
            return Err(KvBackendError::ShapeMismatch);
        }
        let weights = Self {
            model_dim,
            query,
            key,
            value,
            output,
        };
        weights.validate()?;
        Ok(weights)
    }
}

pub trait KvContinuationExecutor<B: PagedKvTensorBackend> {
    type Input;
    type Output;
    type Error;

    fn prefill(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Self::Input,
    ) -> Result<Self::Output, Self::Error>;

    fn decode(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: Self::Input,
    ) -> Result<Self::Output, Self::Error>;
}

#[derive(Clone, Debug)]
pub struct ReferenceCausalAttentionExecutor {
    weights: CausalAttentionWeights,
    heads: usize,
}

impl ReferenceCausalAttentionExecutor {
    pub fn new(weights: CausalAttentionWeights, heads: usize) -> Result<Self, KvBackendError> {
        weights.validate()?;
        if heads == 0 || !weights.model_dim.is_multiple_of(heads) {
            return Err(KvBackendError::InvalidSchema(
                "attention heads must divide model dimension".into(),
            ));
        }
        Ok(Self { weights, heads })
    }

    fn run<B: PagedKvTensorBackend>(
        &self,
        backend: &B,
        cache: &mut B::Cache,
        input: CausalAttentionInput,
        require_single_token: bool,
    ) -> Result<CausalAttentionOutput, KvBackendError> {
        if input.tokens.is_empty()
            || (require_single_token && input.tokens.len() != 1)
            || input
                .tokens
                .iter()
                .any(|token| token.len() != self.weights.model_dim)
        {
            return Err(KvBackendError::ShapeMismatch);
        }
        let before = backend.snapshot(cache)?;
        if before.schema.layer_count != 1
            || before.schema.attention_heads != self.heads
            || before.schema.key_value_heads != self.heads
            || before.schema.attention_heads * before.schema.head_dim != self.weights.model_dim
        {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }
        let first_position = before.position_offset + before.sequence_length;
        let mut outputs = Vec::with_capacity(input.tokens.len());
        for token in input.tokens {
            let query = project(&self.weights.query, self.weights.model_dim, &token);
            let key = project(&self.weights.key, self.weights.model_dim, &token);
            let value = project(&self.weights.value, self.weights.model_dim, &token);
            backend.append(
                cache,
                &[TensorRef {
                    layer: 0,
                    shape: vec![1, self.weights.model_dim],
                    values: key,
                }],
                &[TensorRef {
                    layer: 0,
                    shape: vec![1, self.weights.model_dim],
                    values: value,
                }],
            )?;
            let state = backend.snapshot(cache)?;
            let keys = &state.layers[0].keys.values;
            let values = &state.layers[0].values.values;
            let attended = causal_attention(
                &query,
                keys,
                values,
                state.sequence_length,
                self.heads,
                self.weights.model_dim / self.heads,
            );
            outputs.push(project(
                &self.weights.output,
                self.weights.model_dim,
                &attended,
            ));
        }
        Ok(CausalAttentionOutput {
            first_position,
            next_position: first_position + outputs.len(),
            hidden_states: outputs,
        })
    }
}

impl<B: PagedKvTensorBackend> KvContinuationExecutor<B> for ReferenceCausalAttentionExecutor {
    type Input = CausalAttentionInput;
    type Output = CausalAttentionOutput;
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

fn project(matrix: &[f32], width: usize, input: &[f32]) -> Vec<f32> {
    matrix
        .chunks_exact(width)
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum()
        })
        .collect()
}

fn causal_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    sequence_length: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let width = heads * head_dim;
    let scale = (head_dim as f32).sqrt();
    let mut output = vec![0.0; width];
    for head in 0..heads {
        let head_start = head * head_dim;
        let mut scores = Vec::with_capacity(sequence_length);
        for token in 0..sequence_length {
            let key_start = token * width + head_start;
            let score = query[head_start..head_start + head_dim]
                .iter()
                .zip(&keys[key_start..key_start + head_dim])
                .map(|(left, right)| left * right)
                .sum::<f32>()
                / scale;
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0;
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        for (token, score) in scores.into_iter().enumerate() {
            let value_start = token * width + head_start;
            for offset in 0..head_dim {
                output[head_start + offset] += (score / denominator) * values[value_start + offset];
            }
        }
    }
    output
}
