//! Machinery shared by the R009 (layer-shared LoRA KV) and R010 (MoE expert KV revocation)
//! harnesses: the corpus, the model pins, the matrix of invalidation cells, the emitter and
//! the ledger-anchored liveness oracle. It imports nothing from the frozen `r005` harness;
//! what was needed there is copied here.

pub mod corpus;
pub mod emit;
pub mod experts;
pub mod lab;
pub mod matrix;
pub mod pins;

/// `corpus_generator_seed` of both experiments (each experiment's drift test compares it
/// with its `config.toml`).
pub const SEQUENCE_SEED: u64 = 20_240_607;

/// How big the run is. Only [`Scale::Pinned`] is the experiment; [`Scale::Reduced`] runs a
/// subset of the arms for the smoke test and says so in its output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the smoke test only.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
        }
    }
}
