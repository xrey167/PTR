//! A self-contained CPU text generator for a Qwen2-architecture checkpoint (feature `qwen-cpu`).
//!
//! ```text
//!  model dir: config.json  tokenizer.json  model.safetensors (bf16)
//!        |
//!   QwenModelConfig::from_json  -->  Weights::load (bf16 kept, norms/biases f32)
//!        |
//!   chat_prompt(system, user) -> tokenizer -> ids -> prefill (batched) -> decode (greedy)
//!        |                                                   |
//!        +------ prompt-prefix KV reuse ---------------------+--> stop at end-of-turn / bound
//! ```
//!
//! It is deliberately independent of the paged-KV executors in this crate: those are verified
//! contracts for admission, leases and taint, over f32 layers; this module is the smallest
//! complete path from a model directory to text. It implements [`crate::TextGenerator`], the
//! only type other crates see.
//!
//! `generation_config.json` is ignored: decoding is greedy argmax with the stop tokens and the
//! token bound of the request, whatever sampling or repetition settings the checkpoint ships.
//! Untrusted inputs: `config.json` sizes are capped, every file is size-checked before it is
//! read and may be pinned by SHA-256, and a marker string inside prompt text is plain text.
//!
//! Scope: dense Qwen2 (grouped-query attention, rotary embeddings without scaling, SwiGLU MLP,
//! RMSNorm, tied or separate output head), bf16 weights, greedy decoding, `f32` arithmetic.
//! Not built: sampling, quantization, sliding-window attention, rope scaling, MoE.

mod config;
mod generator;
mod math;
mod model;
mod tokenizer;
mod weights;

pub use config::QwenModelConfig;
pub use generator::{
    QwenCpu, QwenCpuError, QwenCpuOptions, QwenCpuStats, QwenDigests, DEFAULT_MAX_WEIGHTS_BYTES,
};
pub use tokenizer::chat_prompt;
