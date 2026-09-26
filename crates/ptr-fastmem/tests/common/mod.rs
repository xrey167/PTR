use ptr_fastmem::{Decay, FastMemoryConfig, IdentifierCodebook, SourceRef, WriteRequest};
use ptr_types::Generation;

pub fn config(checkpoint_interval: u32) -> FastMemoryConfig {
    FastMemoryConfig {
        heads: 2,
        key_dim: 8,
        value_dim: 6,
        checkpoint_interval,
        max_writes: 4096,
    }
}

/// The identifier codebook a test memory of `config` is bound to.
#[allow(dead_code)]
pub fn codebook(config: &FastMemoryConfig) -> IdentifierCodebook {
    IdentifierCodebook::new(7, config.value_len()).unwrap()
}

/// A deterministic, dense write about `source` at generation 1. Values depend on
/// `salt` so different writes are not collinear.
pub fn write_about(source: &str, salt: u32) -> WriteRequest {
    let config = config(1);
    let key = (0..config.key_len())
        .map(|i| ((i as u32 * 7 + salt * 13) % 11) as f32 - 5.0)
        .map(|x| if x == 0.0 { 0.5 } else { x })
        .collect();
    let value = (0..config.value_len())
        .map(|i| ((i as u32 * 5 + salt * 3) % 9) as f32 * 0.25 - 1.0)
        .collect();
    WriteRequest {
        source: SourceRef {
            key: source.to_owned(),
            generation: Generation(1),
            input_digest: [salt as u8; 32],
        },
        key,
        value,
        beta: 0.75,
        decay: Decay::Scalar(0.97),
    }
}
