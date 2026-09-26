use std::collections::BTreeSet;
use std::fmt;

use ptr_types::{Generation, Revision};

/// Refusals from building, sealing, rebuilding or certifying a branch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BranchError {
    /// `Put` and `Remove` overwrite a value, so the branch must have read the
    /// value it overwrites; otherwise a concurrent change would be lost silently.
    UnreadTarget {
        key: String,
    },
    /// Raw request text and Pod outputs are written by ingress only: a branch
    /// that could overwrite them could erase the source an interpretation was
    /// derived from.
    ReservedNamespace {
        key: String,
    },
    /// A value cannot be encoded canonically (for example, an oversized key).
    InvalidValue {
        key: String,
    },
    /// A counter or set operation found a value that is not a counter or set.
    NotACounter {
        key: String,
    },
    NotASet {
        key: String,
    },
    /// A set member may not be empty.
    InvalidMember {
        key: String,
    },
    /// A counter addition overflowed `i64`.
    CounterOverflow {
        key: String,
    },
    /// Certification was asked against a snapshot older than the branch base.
    SnapshotBehindBase {
        base: Revision,
        snapshot: Revision,
    },
    /// A value read, a prefix scanned, or the input set of a touched key
    /// changed between the branch base and the target. Keys under a changed
    /// prefix are reported as `prefix*`.
    Conflict {
        keys: BTreeSet<String>,
    },
    /// A lifecycle target the branch relied on is no longer live at the
    /// generation it relied on (superseded, revoked or unknown).
    LifecycleChanged {
        targets: BTreeSet<String>,
    },
    /// The branch already relied on another generation of this target. At
    /// most one generation of a target is live at a time, so a branch whose
    /// conclusions rest on two could never be certified.
    ConflictingReliance {
        target: String,
        relied: Generation,
        declared: Generation,
    },
    /// The parts of a sealed branch are not what sealing an open branch
    /// records: a key an operation touches has no recorded base value or
    /// input set, a base value or input set is recorded for a key no
    /// operation touches, or the recorded base value of a touched key differs
    /// from the value the branch read for it (both digest the same base
    /// value). `reason` says which.
    MalformedSeal {
        key: String,
        reason: &'static str,
    },
}

impl BranchError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnreadTarget { .. } => "PTR_BRANCH_UNREAD_TARGET",
            Self::ReservedNamespace { .. } => "PTR_BRANCH_RESERVED_NAMESPACE",
            Self::InvalidValue { .. } => "PTR_BRANCH_INVALID_VALUE",
            Self::NotACounter { .. } => "PTR_BRANCH_NOT_A_COUNTER",
            Self::NotASet { .. } => "PTR_BRANCH_NOT_A_SET",
            Self::InvalidMember { .. } => "PTR_BRANCH_INVALID_MEMBER",
            Self::CounterOverflow { .. } => "PTR_BRANCH_COUNTER_OVERFLOW",
            Self::SnapshotBehindBase { .. } => "PTR_BRANCH_SNAPSHOT_BEHIND_BASE",
            Self::Conflict { .. } => "PTR_BRANCH_CONFLICT",
            Self::LifecycleChanged { .. } => "PTR_BRANCH_LIFECYCLE_CHANGED",
            Self::ConflictingReliance { .. } => "PTR_BRANCH_CONFLICTING_RELIANCE",
            Self::MalformedSeal { .. } => "PTR_BRANCH_MALFORMED_SEAL",
        }
    }
}

impl fmt::Display for BranchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnreadTarget { key } => {
                write!(formatter, "{key:?} is overwritten without having been read")
            }
            Self::ReservedNamespace { key } => {
                write!(
                    formatter,
                    "{key:?} is in a namespace only ingress may write"
                )
            }
            Self::InvalidValue { key } => {
                write!(formatter, "value of {key:?} has no canonical encoding")
            }
            Self::NotACounter { key } => write!(formatter, "{key:?} does not hold a counter"),
            Self::NotASet { key } => write!(formatter, "{key:?} does not hold a set"),
            Self::InvalidMember { key } => write!(formatter, "empty set member for {key:?}"),
            Self::CounterOverflow { key } => write!(formatter, "counter {key:?} overflowed"),
            Self::SnapshotBehindBase { base, snapshot } => write!(
                formatter,
                "snapshot revision {} is behind branch base {}",
                snapshot.0, base.0
            ),
            Self::Conflict { keys } => {
                write!(formatter, "declared dependencies changed at {keys:?}")
            }
            Self::LifecycleChanged { targets } => {
                write!(
                    formatter,
                    "relied-on generations no longer live: {targets:?}"
                )
            }
            Self::ConflictingReliance {
                target,
                relied,
                declared,
            } => write!(
                formatter,
                "{target:?} was relied on at generation {} and then at {}",
                relied.0, declared.0
            ),
            Self::MalformedSeal { key, reason } => {
                write!(formatter, "sealed branch is malformed at {key:?}: {reason}")
            }
        }
    }
}

impl std::error::Error for BranchError {}

/// Refusals from calibrating or evaluating a triage policy.
#[derive(Clone, Debug, PartialEq)]
pub enum ArbiterError {
    /// A risk or confidence level outside `(0, 1)`.
    InvalidRisk { field: &'static str, value: f64 },
    /// No calibration samples were supplied.
    EmptyCalibration,
    /// A calibration rate outside `[0, 1)`.
    InvalidExploration { rate: f64 },
    /// An auto-propose threshold that is not a finite score in `[0, 1]`. A
    /// NaN would admit nothing and a negative one everything eligible, so
    /// either would silently make a policy "never" or "always" auto-propose.
    InvalidThreshold { value: f32 },
    /// A calibration draw that is not a finite number in `[0, 1)`.
    InvalidDraw { draw: f64 },
    /// A logged propensity outside `[0, 1]`, a logged action the logging
    /// policy gave zero probability, or one it gave a probability so small
    /// that the record's importance weight is not finite.
    InvalidPropensity { index: usize, value: f64 },
    /// A logged reward that is not a finite number: any estimate that
    /// averaged it would be NaN or infinite rather than a refusal.
    InvalidReward { index: usize, value: f64 },
    /// A logged score that is not a finite number in `[0, 1]`, the range of
    /// every score a triage sees. A target policy's probability of each
    /// action is computed from the score, so a NaN would silently read as
    /// "below every threshold" and one outside the range as a score no
    /// triage ever had.
    InvalidScore { index: usize, value: f32 },
    /// The evaluated policy takes an action the logging policy never took for
    /// this record, so no reweighting can estimate its value.
    PositivityViolation { index: usize },
    /// An off-policy estimate (`ips`, `snips`, `effective_sample_size` or
    /// `doubly_robust`) of a log that passed every other check does not come
    /// out as a finite number, so it is refused rather than returned.
    NonFiniteEstimate { estimate: &'static str },
    /// No logged decisions were supplied.
    EmptyLog,
    /// A recorded policy needs a version to be cited by.
    EmptyVersion,
    /// A threshold rule that does not fit how the policy was made.
    InvalidRule {
        rule: &'static str,
        message: &'static str,
    },
    /// A calibration set names one branch twice; the unit is the branch.
    DuplicateCalibrationBranch { branch: String },
    /// A triage cited as a policy's is not one that policy can produce, for
    /// any verification report, score and calibration draw: its decision,
    /// slice flag or auto-propose propensity contradicts the policy's
    /// threshold and calibration rate for its eligibility and score, or its
    /// score is not a probability. Calibration and off-policy evaluation
    /// would attribute it to a logging policy that never made it.
    UnexplainedTriage { reason: &'static str },
}

impl ArbiterError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRisk { .. } => "PTR_ARBITER_INVALID_RISK",
            Self::EmptyCalibration => "PTR_ARBITER_EMPTY_CALIBRATION",
            Self::InvalidExploration { .. } => "PTR_ARBITER_INVALID_EXPLORATION",
            Self::InvalidThreshold { .. } => "PTR_ARBITER_INVALID_THRESHOLD",
            Self::InvalidDraw { .. } => "PTR_ARBITER_INVALID_DRAW",
            Self::InvalidPropensity { .. } => "PTR_ARBITER_INVALID_PROPENSITY",
            Self::InvalidReward { .. } => "PTR_ARBITER_INVALID_REWARD",
            Self::InvalidScore { .. } => "PTR_ARBITER_INVALID_SCORE",
            Self::PositivityViolation { .. } => "PTR_ARBITER_POSITIVITY_VIOLATION",
            Self::NonFiniteEstimate { .. } => "PTR_ARBITER_NONFINITE_ESTIMATE",
            Self::EmptyLog => "PTR_ARBITER_EMPTY_LOG",
            Self::EmptyVersion => "PTR_ARBITER_EMPTY_VERSION",
            Self::InvalidRule { .. } => "PTR_ARBITER_INVALID_RULE",
            Self::DuplicateCalibrationBranch { .. } => "PTR_ARBITER_DUPLICATE_CALIBRATION_BRANCH",
            Self::UnexplainedTriage { .. } => "PTR_ARBITER_UNEXPLAINED_TRIAGE",
        }
    }
}

impl fmt::Display for ArbiterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRisk { field, value } => {
                write!(formatter, "{field} = {value} is outside (0, 1)")
            }
            Self::EmptyCalibration => write!(formatter, "no calibration samples"),
            Self::InvalidExploration { rate } => {
                write!(formatter, "calibration rate {rate} is outside [0, 1)")
            }
            Self::InvalidThreshold { value } => {
                write!(
                    formatter,
                    "auto-propose threshold {value} is outside [0, 1]"
                )
            }
            Self::InvalidDraw { draw } => {
                write!(formatter, "calibration draw {draw} is outside [0, 1)")
            }
            Self::InvalidPropensity { index, value } => {
                write!(formatter, "record {index} has invalid propensity {value}")
            }
            Self::InvalidReward { index, value } => {
                write!(formatter, "record {index} has nonfinite reward {value}")
            }
            Self::InvalidScore { index, value } => {
                write!(formatter, "record {index} has score {value} outside [0, 1]")
            }
            Self::PositivityViolation { index } => write!(
                formatter,
                "record {index}: the evaluated policy takes an action the logging policy never took"
            ),
            Self::NonFiniteEstimate { estimate } => {
                write!(
                    formatter,
                    "the {estimate} estimate of the log is not finite"
                )
            }
            Self::EmptyLog => write!(formatter, "no logged decisions"),
            Self::EmptyVersion => write!(formatter, "a recorded policy has no version"),
            Self::InvalidRule { rule, message } => write!(formatter, "rule {rule}: {message}"),
            Self::DuplicateCalibrationBranch { branch } => write!(
                formatter,
                "branch {branch:?} appears twice in one calibration set"
            ),
            Self::UnexplainedTriage { reason } => {
                write!(
                    formatter,
                    "the policy cannot have produced this triage: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for ArbiterError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_behind_base_names_both_revisions() {
        let message = BranchError::SnapshotBehindBase {
            base: Revision(9),
            snapshot: Revision(4),
        }
        .to_string();
        assert!(message.contains('9') && message.contains('4'));
    }
}
