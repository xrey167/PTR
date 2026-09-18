#[derive(Clone, Debug)]
pub struct TypedAttentionConfig {
    pub model_width: usize,
    pub heads: usize,
    pub type_bias: bool,
    pub epistemic_bias: bool,
    pub validity_mask: bool,
}

impl Default for TypedAttentionConfig {
    fn default() -> Self {
        Self {
            model_width: 512,
            heads: 8,
            type_bias: true,
            epistemic_bias: true,
            validity_mask: true,
        }
    }
}
