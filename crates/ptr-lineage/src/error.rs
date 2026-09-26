use std::fmt;

/// Refusals from the adapter lineage, interference, replay and merge logic.
#[derive(Clone, Debug, PartialEq)]
pub enum LineageError {
    /// Two matrices that must be multiplied or compared have incompatible shapes.
    ShapeMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A matrix, vector or scalar input holds NaN or infinity, or an update
    /// would store one.
    NonFinite { field: &'static str },
    /// A matrix has no rows or no columns, or a set that needs a member has
    /// none.
    Empty { field: &'static str },
    /// An adapter id is already registered.
    DuplicateAdapter { id: String },
    /// An adapter names a parent that is not registered.
    UnknownAdapter { id: String },
    /// A child adapter was trained on a different base model than its parent.
    BaseMismatch { expected: String, actual: String },
    /// A lifecycle transition that the current status does not allow.
    InvalidTransition {
        id: String,
        from: &'static str,
        to: &'static str,
    },
    /// A gate report did not pass; nothing is promoted on a failing report.
    GateFailed { id: String },
    /// A held-out sample was offered to the replay pool.
    HeldOutSample { id: String },
    /// A replay sample id is already in the pool.
    DuplicateSample { id: String },
    /// A replay sample id is not in the pool.
    UnknownSample { id: String },
    /// A parameter is outside its valid range.
    InvalidParameter {
        field: &'static str,
        message: &'static str,
    },
}

impl LineageError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::ShapeMismatch { .. } => "PTR_LINEAGE_SHAPE_MISMATCH",
            Self::NonFinite { .. } => "PTR_LINEAGE_NON_FINITE",
            Self::Empty { .. } => "PTR_LINEAGE_EMPTY",
            Self::DuplicateAdapter { .. } => "PTR_LINEAGE_DUPLICATE",
            Self::UnknownAdapter { .. } => "PTR_LINEAGE_UNKNOWN",
            Self::BaseMismatch { .. } => "PTR_LINEAGE_BASE_MISMATCH",
            Self::InvalidTransition { .. } => "PTR_LINEAGE_INVALID_TRANSITION",
            Self::GateFailed { .. } => "PTR_LINEAGE_GATE_FAILED",
            Self::HeldOutSample { .. } => "PTR_LINEAGE_HELD_OUT_SAMPLE",
            Self::DuplicateSample { .. } => "PTR_LINEAGE_DUPLICATE_SAMPLE",
            Self::UnknownSample { .. } => "PTR_LINEAGE_UNKNOWN_SAMPLE",
            Self::InvalidParameter { .. } => "PTR_LINEAGE_INVALID_PARAMETER",
        }
    }
}

impl fmt::Display for LineageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(formatter, "{field}: expected {expected}, got {actual}"),
            Self::NonFinite { field } => write!(formatter, "{field} holds a non-finite value"),
            Self::Empty { field } => write!(formatter, "{field} is empty"),
            Self::DuplicateAdapter { id } => write!(formatter, "adapter {id:?} already registered"),
            Self::UnknownAdapter { id } => write!(formatter, "adapter {id:?} is not registered"),
            Self::BaseMismatch { expected, actual } => write!(
                formatter,
                "base model {actual:?} does not match the lineage base {expected:?}"
            ),
            Self::InvalidTransition { id, from, to } => {
                write!(formatter, "adapter {id:?} cannot move from {from} to {to}")
            }
            Self::GateFailed { id } => write!(formatter, "adapter {id:?} failed its gate"),
            Self::HeldOutSample { id } => {
                write!(
                    formatter,
                    "sample {id:?} is held out and may never be replayed"
                )
            }
            Self::DuplicateSample { id } => write!(formatter, "sample {id:?} already pooled"),
            Self::UnknownSample { id } => write!(formatter, "sample {id:?} is not pooled"),
            Self::InvalidParameter { field, message } => write!(formatter, "{field}: {message}"),
        }
    }
}

impl std::error::Error for LineageError {}
