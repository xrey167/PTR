//! Foundational domain types shared by PTR. This crate intentionally has no external dependencies.

mod checkpoint;
mod codebook;
mod confidence;
mod slot_encoding;
mod validity_mask;
pub use checkpoint::{CheckpointError, CheckpointHeader, TableSize, FORMAT_V2};
pub use codebook::{
    exception_width, CodeFamily, Codebook, CodebookError, CodebookException, CodebookVersion,
    CognitiveType, TypeCode, EXCEPTIONS,
};
pub use confidence::{ConfidenceEstimate, ConfidenceTarget, ConfidenceTargetMismatch};
pub use slot_encoding::{
    EncodingError, EncodingVersion, SlotEncoding, SlotVector, MAX_PAYLOAD_BYTES, MAX_SLOT_WIDTH,
};
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
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
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
string_id!(
    /// The name work is attributed to. Agents are principals: a branch's author
    /// (`ptr_branch::Branch::open`) and a fast memory's owner
    /// (`ptr_pg::FastMemoryRecord::principal`) are a `PrincipalId`.
    ///
    /// It is a bare name and checks nothing: `From<&str>` and the public field
    /// take any string, the empty one and one with surrounding whitespace
    /// included, and nothing binds it to an execution session. Effect attempts
    /// do not use this type: `EffectAttempted.principal` is a string. An
    /// attempt made through an admitted execution session records that
    /// session's principal, which the runtime validated when it created the
    /// session (not empty, no surrounding whitespace, no control characters).
    /// `PtrRuntime::commit` and replay check an `EffectAttempted`'s key but
    /// not its principal, so an attempt written through them records any
    /// string, the empty one included, and names no admitted session. That a
    /// `PrincipalId` names the principal the caller's execution session
    /// admitted, the same string that session's effect attempts record, is
    /// the caller's obligation until branches and fast memories are opened
    /// only through an entry point that takes the principal from the session.
    PrincipalId
);

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
    fn a_principal_id_is_recorded_as_given_whatever_name_it_holds() {
        // Nothing here validates or admits a principal; the runtime's session
        // rules (non-empty, no surrounding whitespace) are not this type's.
        for name in ["agent-7", "agent-7 ", "", "\tagent-7"] {
            assert_eq!(PrincipalId::from(name).0, name);
            assert_eq!(PrincipalId::from(name).to_string(), name);
        }
        assert_ne!(PrincipalId::from("agent-7"), PrincipalId::from("agent-7 "));
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
