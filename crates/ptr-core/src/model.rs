use crate::{ptr_ar::PtrArConfig, ptr_diff::PtrDiffConfig, typed_attention::TypedAttentionConfig};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreFamily { Autoregressive, Diffusion }

#[derive(Clone, Debug)]
pub struct PtrCoreConfig {
    pub family: CoreFamily,
    pub semantic_slots: usize,
    pub typed_attention: TypedAttentionConfig,
    pub ar: PtrArConfig,
    pub diffusion: PtrDiffConfig,
}

impl Default for PtrCoreConfig {
    fn default() -> Self {
        Self { family: CoreFamily::Autoregressive, semantic_slots: 32, typed_attention: Default::default(), ar: Default::default(), diffusion: Default::default() }
    }
}
