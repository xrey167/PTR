use std::fmt;

/// Every way a fast-memory operation can refuse. Each variant names the value it
/// saw and, where one exists, the value it expected, so a refusal can be read
/// without re-running the computation that produced it.
#[derive(Clone, Debug, PartialEq)]
pub enum FastMemoryError {
    /// A configuration field is outside its supported range.
    InvalidConfig {
        field: &'static str,
        value: u64,
        message: &'static str,
    },
    /// A vector's length does not match the configured shape.
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The write strength must lie in `(0, 1]`; outside it the update is no
    /// longer a contraction along the key and the state can grow without bound.
    InvalidBeta { value: f32 },
    /// A decay factor must lie in `(0, 1]`.
    InvalidDecay { index: usize, value: f32 },
    /// A head's key has zero length and cannot be normalised.
    ZeroKey { head: usize },
    /// A value is NaN or infinite.
    NonFinite { field: &'static str, index: usize },
    /// A written value cell exceeds [`crate::MAX_VALUE_MAGNITUDE`] in
    /// magnitude; beyond it a fold of admitted writes could overflow `f32`.
    ValueOutOfRange { index: usize, value: f32 },
    /// A decode threshold is NaN, infinite or negative; a comparison against
    /// it could not refuse an ambiguous or weak readout.
    InvalidThreshold { field: &'static str, value: f32 },
    /// A write sequence number did not advance the journal by exactly one.
    OutOfOrderWrite { expected: u64, actual: u64 },
    /// The journal holds its configured maximum of writes.
    JournalFull { limit: u32 },
    /// A read was refused because the state depends on inputs that are no
    /// longer admissible; they must be excluded (refolded away) first.
    Denied { sources: usize },
    /// Serialized state is malformed; `reason` names the first check that failed.
    CorruptState { reason: &'static str },
    /// Serialized state parsed but its digest does not match its bytes.
    DigestMismatch,
}

impl FastMemoryError {
    /// A stable diagnostic code per variant.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidConfig { .. } => "PTR_FASTMEM_INVALID_CONFIG",
            Self::DimensionMismatch { .. } => "PTR_FASTMEM_DIMENSION_MISMATCH",
            Self::InvalidBeta { .. } => "PTR_FASTMEM_INVALID_BETA",
            Self::InvalidDecay { .. } => "PTR_FASTMEM_INVALID_DECAY",
            Self::ZeroKey { .. } => "PTR_FASTMEM_ZERO_KEY",
            Self::NonFinite { .. } => "PTR_FASTMEM_NON_FINITE",
            Self::ValueOutOfRange { .. } => "PTR_FASTMEM_VALUE_OUT_OF_RANGE",
            Self::InvalidThreshold { .. } => "PTR_FASTMEM_INVALID_THRESHOLD",
            Self::OutOfOrderWrite { .. } => "PTR_FASTMEM_OUT_OF_ORDER_WRITE",
            Self::JournalFull { .. } => "PTR_FASTMEM_JOURNAL_FULL",
            Self::Denied { .. } => "PTR_FASTMEM_DENIED",
            Self::CorruptState { .. } => "PTR_FASTMEM_CORRUPT_STATE",
            Self::DigestMismatch => "PTR_FASTMEM_DIGEST_MISMATCH",
        }
    }
}

impl fmt::Display for FastMemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig {
                field,
                value,
                message,
            } => write!(formatter, "{field}={value}: {message}"),
            Self::DimensionMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has length {actual}, expected {expected}"
            ),
            Self::InvalidBeta { value } => {
                write!(formatter, "beta={value} is outside (0, 1]")
            }
            Self::InvalidDecay { index, value } => {
                write!(formatter, "decay[{index}]={value} is outside (0, 1]")
            }
            Self::ZeroKey { head } => write!(formatter, "key for head {head} has zero length"),
            Self::NonFinite { field, index } => {
                write!(formatter, "{field}[{index}] is not finite")
            }
            Self::ValueOutOfRange { index, value } => write!(
                formatter,
                "value[{index}]={value} exceeds the magnitude bound {}",
                crate::MAX_VALUE_MAGNITUDE
            ),
            Self::InvalidThreshold { field, value } => write!(
                formatter,
                "{field}={value} is not a finite, non-negative threshold"
            ),
            Self::OutOfOrderWrite { expected, actual } => write!(
                formatter,
                "write sequence {actual} does not follow the journal, expected {expected}"
            ),
            Self::JournalFull { limit } => {
                write!(formatter, "journal holds its limit of {limit} writes")
            }
            Self::Denied { sources } => write!(
                formatter,
                "state depends on {sources} inputs that are no longer admissible"
            ),
            Self::CorruptState { reason } => {
                write!(formatter, "corrupt fast-memory state: {reason}")
            }
            Self::DigestMismatch => write!(formatter, "fast-memory state digest mismatch"),
        }
    }
}

impl std::error::Error for FastMemoryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_order_display_names_expected_and_actual() {
        let message = FastMemoryError::OutOfOrderWrite {
            expected: 4,
            actual: 9,
        }
        .to_string();
        assert!(message.contains('4'));
        assert!(message.contains('9'));
    }

    #[test]
    fn every_variant_has_a_distinct_code() {
        let variants = [
            FastMemoryError::InvalidConfig {
                field: "heads",
                value: 0,
                message: "",
            },
            FastMemoryError::DimensionMismatch {
                field: "key",
                expected: 1,
                actual: 2,
            },
            FastMemoryError::InvalidBeta { value: 2.0 },
            FastMemoryError::InvalidDecay {
                index: 0,
                value: 0.0,
            },
            FastMemoryError::ZeroKey { head: 0 },
            FastMemoryError::NonFinite {
                field: "value",
                index: 0,
            },
            FastMemoryError::ValueOutOfRange {
                index: 0,
                value: f32::MAX,
            },
            FastMemoryError::InvalidThreshold {
                field: "min_margin",
                value: f32::NAN,
            },
            FastMemoryError::OutOfOrderWrite {
                expected: 1,
                actual: 3,
            },
            FastMemoryError::JournalFull { limit: 1 },
            FastMemoryError::Denied { sources: 1 },
            FastMemoryError::CorruptState { reason: "" },
            FastMemoryError::DigestMismatch,
        ];
        let mut codes: Vec<_> = variants.iter().map(FastMemoryError::code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), variants.len());
    }
}
