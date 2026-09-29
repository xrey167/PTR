use std::fmt;

use ptr_types::{CapsuleId, Generation};

use crate::projection::IdentifierCodebook;

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
    /// A vector's length, or a query's head count or head width, does not
    /// match the configured shape.
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
    /// A head's key is all zero: it has no direction to normalise. Any other
    /// finite head is normalised, whatever its scale.
    ZeroKey { head: usize },
    /// A value is NaN or infinite.
    NonFinite { field: &'static str, index: usize },
    /// A written value cell exceeds [`crate::MAX_VALUE_MAGNITUDE`] in
    /// magnitude; beyond it a fold of admitted writes could overflow `f32`.
    ValueOutOfRange { index: usize, value: f32 },
    /// A decode threshold is NaN, infinite or negative; a comparison against
    /// it could not refuse an ambiguous or weak readout.
    InvalidThreshold { field: &'static str, value: f32 },
    /// A restored journal's sequence number was not above the one before it
    /// (or was zero): `FastMemory::restore` requires sequence numbers to
    /// strictly increase. Gaps are accepted, because revoked writes leave
    /// them, so a lost journal row is not reported here. `expected` is the
    /// smallest number that would have been accepted.
    OutOfOrderWrite { expected: u64, actual: u64 },
    /// A write would be journaled at `u64::MAX`, the one sequence number
    /// without a successor: no write could be numbered after it, so a journal
    /// may not end there. Likewise a write at or above the lower limit
    /// `FastMemory::with_sequence_limit` set, a limit set below a number the
    /// memory already took (`seq` is that number), or a high-water mark
    /// `FastMemory::with_sequence_high_water` was given at or above the
    /// memory's limit (`seq` is the mark).
    SequenceExhausted { seq: u64 },
    /// The journal holds its configured maximum of writes.
    JournalFull { limit: u32 },
    /// A read was refused because the state depends on inputs that are no
    /// longer admissible; they must be excluded (refolded away) first.
    Denied { sources: usize },
    /// A fact code handed to decoding is from another identifier codebook
    /// than the readout's memory (another seed or length): its scores would
    /// be crosstalk, plausible but meaningless. `index` is the fact's
    /// position among the candidates.
    CodebookMismatch {
        index: usize,
        expected: IdentifierCodebook,
        actual: IdentifierCodebook,
    },
    /// A fact candidate names a lifecycle target in a namespace reserved for
    /// targets that are not capsules (`constraint:<key>`, `procedure:<id>`).
    /// A decoded hit names a capsule, and the runtime accepts no capsule id in
    /// either namespace, so such a candidate is neither built nor decoded.
    ReservedTarget { target: String },
    /// Two fact candidates handed to decoding name the same capsule at the
    /// same generation but differ: under one codebook a fact has one code, so
    /// at most one of them is that fact's, and neither is chosen silently.
    /// `first` and `index` are their positions among the candidates. A
    /// candidate equal to an earlier one is not refused; it is scored once.
    ConflictingFact {
        capsule: CapsuleId,
        generation: Generation,
        first: usize,
        index: usize,
    },
    /// A query is read by a memory bound to a key projection (its digest,
    /// `expected`) but was projected by another one, or states none
    /// (`actual`): its scores against keys written under that projection
    /// would be crosstalk, plausible but meaningless.
    ProjectionMismatch {
        expected: [u8; 32],
        actual: Option<[u8; 32]>,
    },
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
            Self::SequenceExhausted { .. } => "PTR_FASTMEM_SEQUENCE_EXHAUSTED",
            Self::JournalFull { .. } => "PTR_FASTMEM_JOURNAL_FULL",
            Self::Denied { .. } => "PTR_FASTMEM_DENIED",
            Self::CodebookMismatch { .. } => "PTR_FASTMEM_CODEBOOK_MISMATCH",
            Self::ReservedTarget { .. } => "PTR_FASTMEM_RESERVED_TARGET",
            Self::ConflictingFact { .. } => "PTR_FASTMEM_CONFLICTING_FACT",
            Self::ProjectionMismatch { .. } => "PTR_FASTMEM_PROJECTION_MISMATCH",
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
            Self::ZeroKey { head } => write!(formatter, "key for head {head} is all zero"),
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
                "write sequence {actual} is not above the journal's previous one, expected at \
                 least {expected}"
            ),
            Self::SequenceExhausted { seq } => write!(
                formatter,
                "write sequence {seq} is not below the memory's sequence limit, so no write can \
                 be numbered after it"
            ),
            Self::JournalFull { limit } => {
                write!(formatter, "journal holds its limit of {limit} writes")
            }
            Self::Denied { sources } => write!(
                formatter,
                "state depends on {sources} inputs that are no longer admissible"
            ),
            Self::CodebookMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "fact code {index} is from codebook seed {} length {}, the readout's memory \
                 uses seed {} length {}",
                actual.seed(),
                actual.len(),
                expected.seed(),
                expected.len()
            ),
            Self::ReservedTarget { target } => write!(
                formatter,
                "{target:?} is a constraint or procedure target, not a capsule: it is no fact \
                 candidate"
            ),
            Self::ConflictingFact {
                capsule,
                generation,
                first,
                index,
            } => write!(
                formatter,
                "fact codes {first} and {index} both name {:?} at generation {} with different \
                 codes: a fact has one code under a codebook",
                capsule.0, generation.0
            ),
            Self::ProjectionMismatch { expected, actual } => {
                write!(formatter, "the query states key projection ")?;
                match actual {
                    Some(actual) => write_hex(formatter, actual)?,
                    None => write!(formatter, "none")?,
                }
                write!(formatter, ", the memory's keys were written under ")?;
                write_hex(formatter, expected)
            }
            Self::CorruptState { reason } => {
                write!(formatter, "corrupt fast-memory state: {reason}")
            }
            Self::DigestMismatch => write!(formatter, "fast-memory state digest mismatch"),
        }
    }
}

impl std::error::Error for FastMemoryError {}

fn write_hex(formatter: &mut fmt::Formatter<'_>, digest: &[u8; 32]) -> fmt::Result {
    digest
        .iter()
        .try_for_each(|byte| write!(formatter, "{byte:02x}"))
}

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
    fn a_projection_mismatch_names_both_digests_in_full() {
        let message = FastMemoryError::ProjectionMismatch {
            expected: [0xab; 32],
            actual: Some([0x01; 32]),
        }
        .to_string();
        assert!(message.contains(&"ab".repeat(32)), "{message}");
        assert!(message.contains(&"01".repeat(32)), "{message}");
        let unstated = FastMemoryError::ProjectionMismatch {
            expected: [0xab; 32],
            actual: None,
        }
        .to_string();
        assert!(unstated.contains("projection none"), "{unstated}");
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
            FastMemoryError::SequenceExhausted { seq: u64::MAX },
            FastMemoryError::JournalFull { limit: 1 },
            FastMemoryError::Denied { sources: 1 },
            FastMemoryError::CodebookMismatch {
                index: 0,
                expected: IdentifierCodebook::new(1, 2).unwrap(),
                actual: IdentifierCodebook::new(2, 2).unwrap(),
            },
            FastMemoryError::ReservedTarget {
                target: "constraint:budget".into(),
            },
            FastMemoryError::ConflictingFact {
                capsule: CapsuleId::from("a"),
                generation: Generation(1),
                first: 0,
                index: 1,
            },
            FastMemoryError::ProjectionMismatch {
                expected: [1; 32],
                actual: None,
            },
            FastMemoryError::CorruptState { reason: "" },
            FastMemoryError::DigestMismatch,
        ];
        let mut codes: Vec<_> = variants.iter().map(FastMemoryError::code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), variants.len());
    }
}
