//! Loading a Qwen2 checkpoint from `model.safetensors`.
//!
//! Matrices stay bf16 (half the memory traffic of `f32` and exactly the checkpoint's values);
//! norm weights and biases are small and become `f32`.

use safetensors::{tensor::TensorView, Dtype, SafeTensors};

use super::config::QwenModelConfig;
use super::generator::QwenCpuError;
use super::math::{bf16_to_f32, Bf16Matrix};

pub(super) struct LayerWeights {
    pub(super) input_norm: Vec<f32>,
    pub(super) q: Bf16Matrix,
    pub(super) q_bias: Vec<f32>,
    pub(super) k: Bf16Matrix,
    pub(super) k_bias: Vec<f32>,
    pub(super) v: Bf16Matrix,
    pub(super) v_bias: Vec<f32>,
    pub(super) o: Bf16Matrix,
    pub(super) post_norm: Vec<f32>,
    pub(super) gate: Bf16Matrix,
    pub(super) up: Bf16Matrix,
    pub(super) down: Bf16Matrix,
}

pub(super) struct Weights {
    pub(super) embed: Bf16Matrix,
    pub(super) layers: Vec<LayerWeights>,
    pub(super) final_norm: Vec<f32>,
    /// `None` when the output head is tied to the embedding matrix.
    pub(super) lm_head: Option<Bf16Matrix>,
}

impl Weights {
    /// Parse a safetensors buffer against `config`.
    pub(super) fn from_bytes(bytes: &[u8], config: &QwenModelConfig) -> Result<Self, QwenCpuError> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|error| QwenCpuError::Weights(format!("safetensors: {error}")))?;
        let h = config.hidden_size;
        let embed = matrix(&tensors, "model.embed_tokens.weight", config.vocab_size, h)?;
        // `num_layers` is capped by the config validation; no capacity is reserved from it.
        let mut layers = Vec::new();
        for layer in 0..config.num_layers {
            let p = format!("model.layers.{layer}");
            let kv = config.kv_width();
            layers.push(LayerWeights {
                input_norm: vector(&tensors, &format!("{p}.input_layernorm.weight"), h)?,
                q: matrix(&tensors, &format!("{p}.self_attn.q_proj.weight"), h, h)?,
                q_bias: vector(&tensors, &format!("{p}.self_attn.q_proj.bias"), h)?,
                k: matrix(&tensors, &format!("{p}.self_attn.k_proj.weight"), kv, h)?,
                k_bias: vector(&tensors, &format!("{p}.self_attn.k_proj.bias"), kv)?,
                v: matrix(&tensors, &format!("{p}.self_attn.v_proj.weight"), kv, h)?,
                v_bias: vector(&tensors, &format!("{p}.self_attn.v_proj.bias"), kv)?,
                o: matrix(&tensors, &format!("{p}.self_attn.o_proj.weight"), h, h)?,
                post_norm: vector(&tensors, &format!("{p}.post_attention_layernorm.weight"), h)?,
                gate: matrix(
                    &tensors,
                    &format!("{p}.mlp.gate_proj.weight"),
                    config.intermediate_size,
                    h,
                )?,
                up: matrix(
                    &tensors,
                    &format!("{p}.mlp.up_proj.weight"),
                    config.intermediate_size,
                    h,
                )?,
                down: matrix(
                    &tensors,
                    &format!("{p}.mlp.down_proj.weight"),
                    h,
                    config.intermediate_size,
                )?,
            });
        }
        let final_norm = vector(&tensors, "model.norm.weight", h)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(matrix(&tensors, "lm_head.weight", config.vocab_size, h)?)
        };
        Ok(Self {
            embed,
            layers,
            final_norm,
            lm_head,
        })
    }
}

fn view<'a>(tensors: &'a SafeTensors<'a>, name: &str) -> Result<TensorView<'a>, QwenCpuError> {
    tensors
        .tensor(name)
        .map_err(|_| QwenCpuError::Weights(format!("missing tensor {name}")))
}

fn matrix(
    tensors: &SafeTensors<'_>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Bf16Matrix, QwenCpuError> {
    let tensor = view(tensors, name)?;
    if tensor.shape() != [rows, cols] {
        return Err(QwenCpuError::Weights(format!(
            "tensor {name} has shape {:?}, expected [{rows}, {cols}]",
            tensor.shape()
        )));
    }
    if tensor.dtype() != Dtype::BF16 {
        return Err(QwenCpuError::Weights(format!(
            "tensor {name} is {:?}; only bf16 matrices are supported",
            tensor.dtype()
        )));
    }
    let data = tensor
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    Ok(Bf16Matrix { rows, cols, data })
}

fn vector(tensors: &SafeTensors<'_>, name: &str, len: usize) -> Result<Vec<f32>, QwenCpuError> {
    let tensor = view(tensors, name)?;
    if tensor.shape() != [len] {
        return Err(QwenCpuError::Weights(format!(
            "tensor {name} has shape {:?}, expected [{len}]",
            tensor.shape()
        )));
    }
    match tensor.dtype() {
        Dtype::BF16 => Ok(tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16_to_f32(u16::from_le_bytes(*pair)))
            .collect()),
        Dtype::F32 => Ok(tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|quad| f32::from_le_bytes(*quad))
            .collect()),
        other => Err(QwenCpuError::Weights(format!(
            "tensor {name} is {other:?}; expected bf16 or f32"
        ))),
    }
}
