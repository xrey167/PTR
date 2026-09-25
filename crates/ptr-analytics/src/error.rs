use std::fmt;

/// Refusals from the statistics kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum StatsError {
    /// There is nothing to compute over.
    Empty { field: &'static str },
    /// Two inputs that must align have different lengths.
    LengthMismatch { expected: usize, actual: usize },
    /// A probability vector is not a distribution, or a truth index is outside it.
    InvalidDistribution { item: usize },
    /// A parameter is outside its valid range.
    InvalidParameter {
        field: &'static str,
        message: &'static str,
    },
    /// More successes than trials.
    InvalidCount { successes: u64, trials: u64 },
}

impl StatsError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Empty { .. } => "PTR_STATS_EMPTY",
            Self::LengthMismatch { .. } => "PTR_STATS_LENGTH_MISMATCH",
            Self::InvalidDistribution { .. } => "PTR_STATS_INVALID_DISTRIBUTION",
            Self::InvalidParameter { .. } => "PTR_STATS_INVALID_PARAMETER",
            Self::InvalidCount { .. } => "PTR_STATS_INVALID_COUNT",
        }
    }
}

impl fmt::Display for StatsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { field } => write!(formatter, "{field} is empty"),
            Self::LengthMismatch { expected, actual } => {
                write!(formatter, "length {actual} does not match {expected}")
            }
            Self::InvalidDistribution { item } => {
                write!(
                    formatter,
                    "item {item} has no valid probability distribution"
                )
            }
            Self::InvalidParameter { field, message } => write!(formatter, "{field}: {message}"),
            Self::InvalidCount { successes, trials } => {
                write!(formatter, "{successes} successes exceed {trials} trials")
            }
        }
    }
}

impl std::error::Error for StatsError {}
