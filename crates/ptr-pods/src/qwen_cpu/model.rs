//! The Qwen2 forward pass with a per-layer `f32` KV cache.

use super::config::QwenModelConfig;
use super::generator::QwenCpuError;
use super::math::{
    apply_rope, bf16_to_f32, dot_f32, rms_norm, rope_angles, silu, softmax, Bf16Matrix,
};
use super::weights::Weights;

/// Keys (post-RoPE) and values per layer, `len` positions each of `kv_width` values.
pub(super) struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    len: usize,
}

impl KvCache {
    fn new(layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); layers],
            values: vec![Vec::new(); layers],
            len: 0,
        }
    }

    /// Drop every position at or after `len`.
    pub(super) fn truncate(&mut self, len: usize, kv_width: usize) {
        if len < self.len {
            for layer in 0..self.keys.len() {
                self.keys[layer].truncate(len * kv_width);
                self.values[layer].truncate(len * kv_width);
            }
            self.len = len;
        }
    }
}

pub(super) struct Model {
    config: QwenModelConfig,
    weights: Weights,
    threads: usize,
    pub(super) cache: KvCache,
}

/// `t`-major projection: `out[i * rows + r]`, from the row-major scratch of `matmul`.
fn project(
    matrix: &Bf16Matrix,
    xs: &[f32],
    t: usize,
    bias: Option<&[f32]>,
    threads: usize,
) -> Vec<f32> {
    let mut rows_major = vec![0.0_f32; matrix.rows * t];
    matrix.matmul(xs, t, bias, threads, &mut rows_major);
    if t == 1 {
        return rows_major;
    }
    let mut out = vec![0.0_f32; matrix.rows * t];
    for r in 0..matrix.rows {
        for i in 0..t {
            out[i * matrix.rows + r] = rows_major[r * t + i];
        }
    }
    out
}

impl Model {
    pub(super) fn new(config: QwenModelConfig, weights: Weights, threads: usize) -> Self {
        let cache = KvCache::new(config.num_layers);
        Self {
            config,
            weights,
            threads: threads.max(1),
            cache,
        }
    }

    pub(super) fn config(&self) -> &QwenModelConfig {
        &self.config
    }

    pub(super) fn reset_to(&mut self, len: usize) {
        let kv_width = self.config.kv_width();
        self.cache.truncate(len, kv_width);
    }

    /// Feed `tokens` at positions `cache.len() ..`, append their keys and values, and return the
    /// logits of the last token when `want_logits`.
    pub(super) fn forward(
        &mut self,
        tokens: &[u32],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, QwenCpuError> {
        let t = tokens.len();
        if t == 0 {
            return Err(QwenCpuError::Input("no tokens to feed".into()));
        }
        let start = self.cache.len;
        if start + t > self.config.max_position_embeddings {
            return Err(QwenCpuError::Input(
                "sequence exceeds the model context".into(),
            ));
        }
        let h = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let kv_width = self.config.kv_width();
        let heads = self.config.num_heads;
        let group = heads / self.config.num_kv_heads;
        let eps = self.config.rms_norm_eps;
        let threads = self.threads;

        let mut x = vec![0.0_f32; t * h];
        for (i, token) in tokens.iter().enumerate() {
            let id = usize::try_from(*token).unwrap_or(usize::MAX);
            if id >= self.config.vocab_size {
                return Err(QwenCpuError::Input(format!(
                    "token id {token} is outside the vocabulary"
                )));
            }
            let row = &self.weights.embed.data[id * h..(id + 1) * h];
            for (dst, src) in x[i * h..(i + 1) * h].iter_mut().zip(row) {
                *dst = bf16_to_f32(*src);
            }
        }
        let angles: Vec<(Vec<f32>, Vec<f32>)> = (0..t)
            .map(|i| rope_angles(start + i, head_dim, self.config.rope_theta))
            .collect();

        let mut normed = vec![0.0_f32; t * h];
        let scale = 1.0 / (head_dim as f32).sqrt();
        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            for i in 0..t {
                rms_norm(
                    &x[i * h..(i + 1) * h],
                    &layer.input_norm,
                    eps,
                    &mut normed[i * h..(i + 1) * h],
                );
            }
            let mut q = project(&layer.q, &normed, t, Some(&layer.q_bias), threads);
            let mut k = project(&layer.k, &normed, t, Some(&layer.k_bias), threads);
            let v = project(&layer.v, &normed, t, Some(&layer.v_bias), threads);
            for i in 0..t {
                let (cos, sin) = &angles[i];
                apply_rope(&mut q[i * h..(i + 1) * h], head_dim, cos, sin);
                apply_rope(&mut k[i * kv_width..(i + 1) * kv_width], head_dim, cos, sin);
            }
            self.cache.keys[layer_index].extend_from_slice(&k);
            self.cache.values[layer_index].extend_from_slice(&v);

            let keys = &self.cache.keys[layer_index];
            let values = &self.cache.values[layer_index];
            let mut attended = vec![0.0_f32; t * h];
            for i in 0..t {
                let visible = start + i + 1;
                let mut scores = vec![0.0_f32; visible];
                for head in 0..heads {
                    let kv_head = head / group;
                    let query = &q[i * h + head * head_dim..i * h + (head + 1) * head_dim];
                    for (position, score) in scores.iter_mut().enumerate() {
                        let offset = position * kv_width + kv_head * head_dim;
                        *score = dot_f32(query, &keys[offset..offset + head_dim]) * scale;
                    }
                    softmax(&mut scores);
                    let out = &mut attended[i * h + head * head_dim..i * h + (head + 1) * head_dim];
                    for (position, weight) in scores.iter().enumerate() {
                        let offset = position * kv_width + kv_head * head_dim;
                        for (dst, value) in out.iter_mut().zip(&values[offset..offset + head_dim]) {
                            *dst += weight * value;
                        }
                    }
                }
            }
            let projected = project(&layer.o, &attended, t, None, threads);
            for (dst, add) in x.iter_mut().zip(&projected) {
                *dst += add;
            }

            for i in 0..t {
                rms_norm(
                    &x[i * h..(i + 1) * h],
                    &layer.post_norm,
                    eps,
                    &mut normed[i * h..(i + 1) * h],
                );
            }
            let gate = project(&layer.gate, &normed, t, None, threads);
            let up = project(&layer.up, &normed, t, None, threads);
            let activated: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = project(&layer.down, &activated, t, None, threads);
            for (dst, add) in x.iter_mut().zip(&down) {
                *dst += add;
            }
        }
        self.cache.len = start + t;

        if !want_logits {
            return Ok(None);
        }
        let mut last = vec![0.0_f32; h];
        rms_norm(&x[(t - 1) * h..], &self.weights.final_norm, eps, &mut last);
        let head = self.weights.lm_head.as_ref().unwrap_or(&self.weights.embed);
        let mut logits = vec![0.0_f32; head.rows];
        head.matmul(&last, 1, None, threads, &mut logits);
        Ok(Some(logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_cpu::math::f32_to_bf16;
    use safetensors::{serialize, tensor::TensorView, Dtype};

    fn tiny_config(tie: bool) -> QwenModelConfig {
        QwenModelConfig::from_json(&format!(
            r#"{{"model_type":"qwen2","hidden_act":"silu","hidden_size":16,
            "intermediate_size":24,"num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":2,"vocab_size":32,"max_position_embeddings":64,
            "rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":{tie}}}"#
        ))
        .unwrap()
    }

    /// Deterministic pseudo-random values in `[-scale, scale]`.
    fn values(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn tiny_weights(config: &QwenModelConfig) -> Weights {
        let mut named: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut seed = 0;
        let mut push = |name: String, shape: Vec<usize>, scale: f32, norm: bool| {
            seed += 1;
            let n: usize = shape.iter().product();
            let raw = if norm {
                values(seed, n, 0.2).iter().map(|v| v + 1.0).collect()
            } else {
                values(seed, n, scale)
            };
            let bytes: Vec<u8> = raw
                .iter()
                .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
                .collect();
            named.push((name, shape, bytes));
        };
        let (h, kv, i) = (
            config.hidden_size,
            config.kv_width(),
            config.intermediate_size,
        );
        push(
            "model.embed_tokens.weight".into(),
            vec![config.vocab_size, h],
            0.5,
            false,
        );
        for l in 0..config.num_layers {
            let p = format!("model.layers.{l}");
            push(format!("{p}.input_layernorm.weight"), vec![h], 0.0, true);
            push(
                format!("{p}.self_attn.q_proj.weight"),
                vec![h, h],
                0.4,
                false,
            );
            push(format!("{p}.self_attn.q_proj.bias"), vec![h], 0.1, false);
            push(
                format!("{p}.self_attn.k_proj.weight"),
                vec![kv, h],
                0.4,
                false,
            );
            push(format!("{p}.self_attn.k_proj.bias"), vec![kv], 0.1, false);
            push(
                format!("{p}.self_attn.v_proj.weight"),
                vec![kv, h],
                0.4,
                false,
            );
            push(format!("{p}.self_attn.v_proj.bias"), vec![kv], 0.1, false);
            push(
                format!("{p}.self_attn.o_proj.weight"),
                vec![h, h],
                0.4,
                false,
            );
            push(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                0.0,
                true,
            );
            push(format!("{p}.mlp.gate_proj.weight"), vec![i, h], 0.4, false);
            push(format!("{p}.mlp.up_proj.weight"), vec![i, h], 0.4, false);
            push(format!("{p}.mlp.down_proj.weight"), vec![h, i], 0.4, false);
        }
        push("model.norm.weight".into(), vec![h], 0.0, true);
        if !config.tie_word_embeddings {
            push(
                "lm_head.weight".into(),
                vec![config.vocab_size, h],
                0.5,
                false,
            );
        }
        let views: Vec<(String, TensorView<'_>)> = named
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let file = serialize(views, None).unwrap();
        Weights::from_bytes(&file, config).unwrap()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn batched_prefill_equals_token_by_token_and_is_thread_independent() {
        for tie in [true, false] {
            let config = tiny_config(tie);
            let tokens: Vec<u32> = vec![3, 17, 5, 29, 0, 11, 8];
            let mut batched = Model::new(config.clone(), tiny_weights(&config), 1);
            let whole = batched.forward(&tokens, true).unwrap().unwrap();
            assert_eq!(whole.len(), config.vocab_size);
            assert!(whole.iter().all(|v| v.is_finite()));

            let mut stepwise = Model::new(config.clone(), tiny_weights(&config), 4);
            let mut last = Vec::new();
            for token in &tokens {
                last = stepwise.forward(&[*token], true).unwrap().unwrap();
            }
            assert!(
                max_diff(&whole, &last) < 1e-3,
                "tie={tie}: {}",
                max_diff(&whole, &last)
            );

            // Truncating the cache and re-feeding the tail reproduces the same logits.
            stepwise.reset_to(4);
            let again = stepwise.forward(&tokens[4..], true).unwrap().unwrap();
            assert!(max_diff(&whole, &again) < 1e-3);
        }
    }

    #[test]
    fn causal_attention_does_not_look_ahead() {
        let config = tiny_config(true);
        let mut a = Model::new(config.clone(), tiny_weights(&config), 1);
        let mut b = Model::new(config.clone(), tiny_weights(&config), 1);
        let one = a.forward(&[3, 17, 5], true).unwrap().unwrap();
        // Logits after a 3-token prefix do not depend on tokens fed later.
        b.forward(&[3, 17, 5], false).unwrap();
        b.forward(&[9, 9], false).unwrap();
        b.reset_to(3);
        let two = b.forward(&[1], true).unwrap().unwrap();
        assert!(
            max_diff(&one, &two) > 1e-4,
            "different next tokens must differ"
        );
        let mut c = Model::new(config.clone(), tiny_weights(&config), 1);
        c.forward(&[3, 17], false).unwrap();
        let three = c.forward(&[5], true).unwrap().unwrap();
        assert!(max_diff(&one, &three) < 1e-3);
    }

    #[test]
    fn out_of_vocabulary_and_overlong_input_are_refused() {
        let config = tiny_config(true);
        let mut model = Model::new(config.clone(), tiny_weights(&config), 1);
        assert!(model.forward(&[32], true).is_err());
        assert!(model.forward(&[], true).is_err());
        assert!(model.forward(&vec![1; 65], true).is_err());
    }
}
