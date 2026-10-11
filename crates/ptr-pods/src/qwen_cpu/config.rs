//! `config.json` of a Qwen2 checkpoint.

use super::generator::QwenCpuError;

/// Largest accepted value of each size field of an untrusted `config.json`. They are far above
/// any dense Qwen2 checkpoint (72B: 80 layers, hidden 8192) and keep every size derived from the
/// config small enough that nothing is allocated from an unvalidated number.
pub mod limits {
    pub const MAX_LAYERS: usize = 128;
    pub const MAX_HIDDEN_SIZE: usize = 16_384;
    pub const MAX_INTERMEDIATE_SIZE: usize = 65_536;
    pub const MAX_HEADS: usize = 256;
    pub const MAX_VOCAB_SIZE: usize = 1 << 20;
    pub const MAX_POSITIONS: usize = 1 << 20;
}

/// The architecture parameters the CPU model needs, read from `config.json`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct QwenModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
}

impl QwenModelConfig {
    /// Parse and validate `config.json`.
    ///
    /// # Errors
    /// [`QwenCpuError::Config`] for a missing or inconsistent field, a model type other than
    /// `qwen2`, an activation other than `silu`, or an unsupported feature (sliding window,
    /// rope scaling).
    pub fn from_json(text: &str) -> Result<Self, QwenCpuError> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|error| QwenCpuError::Config(format!("config.json is not JSON: {error}")))?;
        let uint = |key: &str| -> Result<usize, QwenCpuError> {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n > 0)
                .ok_or_else(|| QwenCpuError::Config(format!("missing or invalid `{key}`")))
        };
        let float = |key: &str| -> Result<f64, QwenCpuError> {
            value
                .get(key)
                .and_then(serde_json::Value::as_f64)
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or_else(|| QwenCpuError::Config(format!("missing or invalid `{key}`")))
        };
        if value.get("model_type").and_then(serde_json::Value::as_str) != Some("qwen2") {
            return Err(QwenCpuError::Config("model_type must be qwen2".into()));
        }
        if value.get("hidden_act").and_then(serde_json::Value::as_str) != Some("silu") {
            return Err(QwenCpuError::Config("hidden_act must be silu".into()));
        }
        if value
            .get("use_sliding_window")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Err(QwenCpuError::Config(
                "sliding-window attention is not supported".into(),
            ));
        }
        if value
            .get("rope_scaling")
            .is_some_and(|scaling| !scaling.is_null())
        {
            return Err(QwenCpuError::Config("rope scaling is not supported".into()));
        }
        let eps = float("rms_norm_eps")?;
        #[allow(clippy::cast_possible_truncation)]
        let rms_norm_eps = eps as f32;
        let config = Self {
            hidden_size: uint("hidden_size")?,
            intermediate_size: uint("intermediate_size")?,
            num_layers: uint("num_hidden_layers")?,
            num_heads: uint("num_attention_heads")?,
            num_kv_heads: uint("num_key_value_heads")?,
            vocab_size: uint("vocab_size")?,
            max_position_embeddings: uint("max_position_embeddings")?,
            rope_theta: float("rope_theta")?,
            rms_norm_eps,
            tie_word_embeddings: value
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        };
        config.validate()?;
        Ok(config)
    }

    /// Per-head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Width of the key and value projections.
    #[must_use]
    pub fn kv_width(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    fn validate(&self) -> Result<(), QwenCpuError> {
        let bad = |why: &str| Err(QwenCpuError::Config(why.into()));
        for (name, value, max) in [
            ("num_hidden_layers", self.num_layers, limits::MAX_LAYERS),
            ("hidden_size", self.hidden_size, limits::MAX_HIDDEN_SIZE),
            (
                "intermediate_size",
                self.intermediate_size,
                limits::MAX_INTERMEDIATE_SIZE,
            ),
            ("num_attention_heads", self.num_heads, limits::MAX_HEADS),
            ("num_key_value_heads", self.num_kv_heads, limits::MAX_HEADS),
            ("vocab_size", self.vocab_size, limits::MAX_VOCAB_SIZE),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                limits::MAX_POSITIONS,
            ),
        ] {
            if value > max {
                return bad(&format!("`{name}` {value} is above the supported {max}"));
            }
        }
        // The f32 cast of a tiny or huge f64 can flush to zero or overflow to infinity.
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0) {
            return bad("`rms_norm_eps` is not a positive finite f32");
        }
        if self.num_kv_heads > self.num_heads {
            return bad("num_key_value_heads exceeds num_attention_heads");
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return bad("hidden_size must be a multiple of num_attention_heads");
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return bad("num_attention_heads must be a multiple of num_key_value_heads");
        }
        if !self.head_dim().is_multiple_of(2) {
            return bad("head dimension must be even");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{"model_type":"qwen2","hidden_act":"silu","hidden_size":896,
        "intermediate_size":4864,"num_hidden_layers":24,"num_attention_heads":14,
        "num_key_value_heads":2,"vocab_size":151936,"max_position_embeddings":32768,
        "rope_theta":1000000.0,"rms_norm_eps":1e-06,"tie_word_embeddings":true,
        "use_sliding_window":false}"#;

    #[test]
    fn parses_the_shape_of_a_small_qwen2() {
        let config = QwenModelConfig::from_json(GOOD).unwrap();
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.kv_width(), 128);
        assert!(config.tie_word_embeddings);
        assert_eq!(config.num_layers, 24);
    }

    #[test]
    fn refuses_what_it_does_not_implement() {
        for (from, to) in [
            ("\"qwen2\"", "\"llama\""),
            ("\"silu\"", "\"gelu\""),
            (
                "\"use_sliding_window\":false",
                "\"use_sliding_window\":true",
            ),
            ("\"num_attention_heads\":14", "\"num_attention_heads\":15"),
            (
                "\"rope_theta\"",
                "\"rope_scaling\":{\"type\":\"yarn\"},\"rope_theta\"",
            ),
        ] {
            let text = GOOD.replace(from, to);
            assert!(QwenModelConfig::from_json(&text).is_err(), "{to}");
        }
        assert!(QwenModelConfig::from_json("not json").is_err());
    }

    #[test]
    fn a_hostile_config_is_refused_before_anything_is_sized_from_it() {
        for (from, to) in [
            (
                "\"num_hidden_layers\":24",
                "\"num_hidden_layers\":1000000000000000000",
            ),
            (
                "\"num_hidden_layers\":24",
                "\"num_hidden_layers\":1000000000",
            ),
            ("\"num_hidden_layers\":24", "\"num_hidden_layers\":129"),
            ("\"hidden_size\":896", "\"hidden_size\":1000000000000"),
            (
                "\"intermediate_size\":4864",
                "\"intermediate_size\":4000000000",
            ),
            (
                "\"vocab_size\":151936",
                "\"vocab_size\":18446744073709551615",
            ),
            (
                "\"max_position_embeddings\":32768",
                "\"max_position_embeddings\":4294967296",
            ),
            (
                "\"num_attention_heads\":14",
                "\"num_attention_heads\":100000",
            ),
            ("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":1e-60"),
            ("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":1e300"),
            ("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":0"),
            ("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":-1e-6"),
            ("\"num_key_value_heads\":2", "\"num_key_value_heads\":28"),
            ("\"num_hidden_layers\":24", "\"num_hidden_layers\":-3"),
            ("\"num_hidden_layers\":24", "\"num_hidden_layers\":2.5"),
            ("\"num_hidden_layers\":24", "\"num_hidden_layers\":\"24\""),
        ] {
            let text = GOOD.replace(from, to);
            assert_ne!(text, GOOD, "{to}");
            assert!(QwenModelConfig::from_json(&text).is_err(), "{to}");
        }
        // The edge of the accepted range still parses.
        let at_cap = GOOD.replace("\"num_hidden_layers\":24", "\"num_hidden_layers\":128");
        assert_eq!(QwenModelConfig::from_json(&at_cap).unwrap().num_layers, 128);
    }
}
