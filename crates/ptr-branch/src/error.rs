use std::collections::BTreeSet;
use std::fmt;

use ptr_types::Revision;

/// Refusals from building, sealing or certifying a branch.
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
    /// A value, or a prefix scanned, changed between the branch base and the
    /// target. Keys under a changed prefix are reported as `prefix*`.
    Conflict {
        keys: BTreeSet<String>,
    },
    /// A lifecycle target the branch relied on is no longer live at the
    /// generation it relied on (superseded, revoked or unknown).
    LifecycleChanged {
        targets: BTreeSet<String>,
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
            Self::Conflict { keys } => write!(formatter, "read set changed at {keys:?}"),
            Self::LifecycleChanged { targets } => {
                write!(
                    formatter,
                    "relied-on generations no longer live: {targets:?}"
                )
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
    /// A calibration draw that is not a finite number in `[0, 1)`.
    InvalidDraw { draw: f64 },
    /// A logged propensity outside `[0, 1]`, or a logged action the logging
    /// policy gave zero probability.
    InvalidPropensity { index: usize, value: f64 },
    /// The evaluated policy takes an action the logging policy never took for
    /// this record, so no reweighting can estimate its value.
    PositivityViolation { index: usize },
    /// No logged decisions were supplied.
    EmptyLog,
}

impl ArbiterError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRisk { .. } => "PTR_ARBITER_INVALID_RISK",
            Self::EmptyCalibration => "PTR_ARBITER_EMPTY_CALIBRATION",
            Self::InvalidExploration { .. } => "PTR_ARBITER_INVALID_EXPLORATION",
            Self::InvalidDraw { .. } => "PTR_ARBITER_INVALID_DRAW",
            Self::InvalidPropensity { .. } => "PTR_ARBITER_INVALID_PROPENSITY",
            Self::PositivityViolation { .. } => "PTR_ARBITER_POSITIVITY_VIOLATION",
            Self::EmptyLog => "PTR_ARBITER_EMPTY_LOG",
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
            Self::InvalidDraw { draw } => {
                write!(formatter, "calibration draw {draw} is outside [0, 1)")
            }
            Self::InvalidPropensity { index, value } => {
                write!(formatter, "record {index} has invalid propensity {value}")
            }
            Self::PositivityViolation { index } => write!(
                formatter,
                "record {index}: the evaluated policy takes an action the logging policy never took"
            ),
            Self::EmptyLog => write!(formatter, "no logged decisions"),
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
