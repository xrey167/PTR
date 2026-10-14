//! Error type of the reference forward pass.

use crate::config::ConfigError;
use crate::tensor::ShapeError;
use std::fmt;

/// Why a forward pass, head or denoising step was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum CoreError {
    /// A tensor shape violation.
    Shape(ShapeError),
    /// An invalid configuration.
    Config(ConfigError),
    /// No slots were supplied.
    NoSlots,
    /// More slots than `max_slots`.
    TooManySlots { got: usize, max: usize },
    /// The validity mask admits no slot, so attention has nothing to read.
    NoAdmittedSlots,
    /// A caller-supplied mask does not match the slot count.
    MaskLength { expected: usize, found: usize },
    /// A slot's latent vector is not `d_model` wide.
    SlotWidth {
        slot: u32,
        expected: usize,
        found: usize,
    },
    /// A non-finite value entered or left a computation.
    NonFinite { what: &'static str },
    /// The halting epsilon is outside `[0, 1)`.
    EpsilonOutOfRange,
    /// A codebook lookup failed (stable diagnostic code).
    Codebook { code: &'static str },
    /// A pooled vector is not `d_model` wide.
    PooledWidth { expected: usize, found: usize },
    /// No action candidates were supplied.
    NoCandidates,
    /// More action candidates than the bound.
    TooManyCandidates { got: usize, max: usize },
    /// The deterministic check rejected every candidate.
    AllCandidatesRejected,
    /// A PTR-Diff token id is outside the vocabulary.
    TokenOutOfVocab {
        position: usize,
        token: u32,
        vocab: usize,
    },
    /// The PTR-Diff canvas is empty or longer than `max_slots`.
    CanvasLength { len: usize, max: usize },
    /// The PTR-Diff step count is outside `1..=MAX_DENOISE_STEPS`.
    DenoiseSteps { steps: u32 },
    /// A training set is empty or an option is zero.
    TrainingInput { what: &'static str },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CoreError {}

impl From<ShapeError> for CoreError {
    fn from(e: ShapeError) -> Self {
        Self::Shape(e)
    }
}

impl From<ConfigError> for CoreError {
    fn from(e: ConfigError) -> Self {
        Self::Config(e)
    }
}
