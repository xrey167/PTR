//! Speculative agent branches over immutable semantic snapshots, certified
//! against declared dependencies and triaged by a verifier-bounded, calibrated
//! policy.
//!
//! A branch has no authority. It is a private overlay on one snapshot that
//! declares what its conclusions depend on — value digests of the keys it
//! read, range digests of the prefixes it scanned, the input set of every key
//! it touches, and the lifecycle generations it relied on — and the
//! operations it would apply.
//! Certification against a newer snapshot either refuses (a dependency
//! changed) or yields a [`MergePlan`]: one ordinary semantic delta plus the
//! revision it was certified against, committed only through the runtime's
//! verified-delta path. Triage decides whether a plan is proposed
//! automatically, sent to a person or dropped; it can move a branch towards
//! more human review but never past verification.

mod arbiter;
mod branch;
mod certify;
mod digest;
mod error;
mod ops;

pub use arbiter::{
    calibrate_threshold, calibration_draw, certify_threshold, doubly_robust, evaluate_off_policy,
    threshold_grid, AutoThreshold, CalibrationSample, LoggedTriage, OffPolicyEstimate,
    PolicyRecord, ThresholdRule, TriageDecision, TriageOutcome, TriagePolicy, THRESHOLD_GRID_STEPS,
};
pub use branch::{Branch, BranchId, SealedBranch, SealedBranchParts, RESERVED_PREFIXES};
pub use certify::{certify, Certification, MergePlan};
pub use digest::{InputsDigest, RangeDigest, ValueDigest};
pub use error::{ArbiterError, BranchError};
pub use ops::{
    counter_value, read_counter_value, read_set_value, set_value, BranchOp, COUNTER_TYPE,
    OP_SOURCE, SET_TYPE,
};
