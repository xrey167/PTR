//! Foundational domain types shared by PTR. This crate intentionally has no external dependencies.

mod codebook;
mod confidence;
mod validity_mask;
pub use codebook::{CodeFamily, Codebook, CodebookError, CodebookVersion, CognitiveType, TypeCode};
pub use confidence::{ConfidenceEstimate, ConfidenceTarget, ConfidenceTargetMismatch};
pub use validity_mask::{MaskError, ValidityMask};

use std::fmt;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Revision(pub u64);

impl Revision {
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Generation(pub u64);

impl Generation {
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CommitIndex(pub u64);

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub String);
        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

string_id!(ProjectId);
string_id!(CapsuleId);
string_id!(ArtifactId);
string_id!(CapabilityId);
string_id!(TypeId);
string_id!(PodId);
string_id!(CandidateId);
string_id!(RequestId);
string_id!(NodeId);
string_id!(EvidenceId);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Probability(f32);

impl Probability {
    pub fn new(value: f32) -> Option<Self> {
        (value.is_finite() && (0.0..=1.0).contains(&value)).then_some(Self(value))
    }
    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for Probability {
    fn default() -> Self {
        Self(0.5)
    }
}

/// Semantic role of a typed value independent of whether it is known,
/// hypothetical, observed, or uncertain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SemanticRole {
    Goal,
    Constraint,
    Claim,
    Evidence,
    Resource,
    Capability,
    Relation,
    Procedure,
    Action,
}

/// Epistemic state is orthogonal to semantic role and lifecycle validity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EpistemicState {
    Unknown,
    Assumed,
    Hypothesis,
    Observed,
    Inferred,
    Verified,
}

/// Shape of uncertainty carried by a semantic value. A distribution is a
/// representation of uncertainty, not a semantic role or authority level.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UncertaintyKind {
    Point,
    Interval,
    Distribution,
}

/// Cognitive/reasoning operator requested by the model or router.
///
/// This taxonomy is shared across ptr-core, ptr-model-api, routing and
/// training so operator identity never degrades to a provider-specific string.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReasoningOperator {
    Semantic,
    Deductive,
    Probabilistic,
    Statistical,
    Temporal,
    Causal,
    Search,
    Optimization,
    Simulation,
    Symbolic,
    ExternalPod,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Effect {
    Pure,
    Read,
    Mutation,
    External,
    Irreversible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Validity {
    Live,
    Superseded,
    Revoked,
    Disputed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationLevel {
    Unverified,
    LatentAgreement,
    SampleVerified,
    FullSemantic,
    Deterministic,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceRef {
    pub source: EvidenceId,
    pub note: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Epistemic<T> {
    Unknown,
    Assumed(T),
    Hypothesis { value: T, confidence: Probability },
    Distribution(Vec<(T, Probability)>),
    Observed(T),
    Inferred(T),
    Known(T),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypedValue<T> {
    pub value: T,
    pub generation: Generation,
    pub validity: Validity,
    pub provenance: Vec<ProvenanceRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticIssue {
    pub code: String,
    pub message: String,
    pub hard: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_is_bounded() {
        assert!(Probability::new(0.5).is_some());
        assert!(Probability::new(-0.1).is_none());
        assert!(Probability::new(1.1).is_none());
    }

    #[test]
    fn lifecycle_versions_are_distinct_concepts() {
        assert_ne!(Revision(7).0, Generation(8).0);
    }

    #[test]
    fn semantic_role_and_epistemic_state_are_independent_axes() {
        let role = SemanticRole::Claim;
        let state = EpistemicState::Hypothesis;
        let uncertainty = UncertaintyKind::Distribution;

        assert_eq!(role, SemanticRole::Claim);
        assert_eq!(state, EpistemicState::Hypothesis);
        assert_eq!(uncertainty, UncertaintyKind::Distribution);
    }

    #[test]
    fn reasoning_operator_is_a_typed_cross_component_contract() {
        let operator = ReasoningOperator::Probabilistic;
        assert_eq!(operator, ReasoningOperator::Probabilistic);
    }
}
