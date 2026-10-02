//! Foundational domain types shared by PTR. This crate intentionally has no external dependencies.

mod checkpoint;
mod codebook;
mod confidence;
mod mesh;
mod pod_address;
mod slot_encoding;
mod validity_mask;
pub use checkpoint::{CheckpointError, CheckpointHeader, TableSize, FORMAT_V2, FORMAT_V3};
pub use codebook::{
    exception_width, CodeFamily, Codebook, CodebookError, CodebookException, CodebookVersion,
    CognitiveType, TypeCode, EXCEPTIONS,
};
pub use confidence::{ConfidenceEstimate, ConfidenceTarget, ConfidenceTargetMismatch};
pub use mesh::{MeshTunnelEventKind, MeshTunnelLifecycleEvent};
pub use pod_address::{
    FrameKind, MeshEndpointBinding, MeshRouteKind, PodAddress, PodEndpoint, PodHeader,
    PodRevisionAddress, PodRoute, PodTransport, ResolveError, RouteConstraints,
};
pub use slot_encoding::{
    EncodingError, EncodingVersion, SlotEncoding, SlotVector, MAX_PAYLOAD_BYTES, MAX_SLOT_WIDTH,
};
pub use validity_mask::{MaskError, ValidityMask};

pub type Digest = [u8; 32];

use sha2::{Digest as ShaDigest, Sha256};
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

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FencingToken(pub u128);

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
string_id!(PodIdentity);
string_id!(CandidateId);
string_id!(RequestId);
string_id!(NodeId);
string_id!(EvidenceId);
string_id!(KnowledgeObjectId);
string_id!(NamespaceId);
string_id!(RawEventId);
string_id!(StateId);
string_id!(ScopeId);
string_id!(SessionId);
string_id!(ModelVersion);
string_id!(AdapterVersion);
string_id!(PolicyVersion);
string_id!(EntityId);
string_id!(RetrievalKey);
string_id!(DeviceId);
string_id!(ExecutorIdentity);
string_id!(KeyId);
string_id!(RegionId);
string_id!(ZoneId);
string_id!(PeerId);
string_id!(TraceId);
string_id!(NetworkId);
string_id!(InvitationId);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AuthenticationLevel {
    Anonymous,
    Password,
    MultiFactor,
    HardwareBound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdentityContext {
    pub subject: String,
    pub issuer: String,
    pub groups: Vec<String>,
    pub authentication_level: AuthenticationLevel,
    pub session_id: SessionId,
    pub issued_at: Timestamp,
    pub expires_at: Timestamp,
    pub identity_digest: Digest,
}

impl IdentityContext {
    pub fn canonical_digest(&self) -> Digest {
        let mut bytes = Vec::new();
        put_text(&mut bytes, &self.subject);
        put_text(&mut bytes, &self.issuer);
        bytes.extend_from_slice(&(self.groups.len() as u64).to_le_bytes());
        for group in &self.groups {
            put_text(&mut bytes, group);
        }
        bytes.push(self.authentication_level as u8);
        put_text(&mut bytes, &self.session_id.0);
        bytes.extend_from_slice(&self.issued_at.0.to_le_bytes());
        bytes.extend_from_slice(&self.expires_at.0.to_le_bytes());
        Sha256::digest(bytes).into()
    }

    pub fn has_valid_digest(&self) -> bool {
        self.identity_digest == self.canonical_digest()
    }

    pub fn is_valid_at(&self, now: Timestamp) -> bool {
        self.has_valid_digest() && self.issued_at.0 <= now.0 && now.0 < self.expires_at.0
    }
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionRequest {
    pub identity: IdentityContext,
    pub project: ProjectId,
    pub capability: CapabilityId,
    pub input_type: TypeId,
    pub pod: Option<PodId>,
    pub session: SessionId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionDenial {
    ExpiredIdentity,
    SessionMismatch,
    ProjectDenied,
    CapabilityDenied,
    InputTypeDenied,
    PolicyRevisionStale,
    SessionRevoked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionDecision {
    Allowed {
        project: ProjectId,
        capabilities: Vec<CapabilityId>,
        expires_at: Timestamp,
        policy_revision: Revision,
    },
    Denied {
        reason: AdmissionDenial,
    },
}

pub trait IdentityProvider: Send + Sync {
    type Error;

    fn authenticate(&self, credential: &[u8]) -> Result<IdentityContext, Self::Error>;
}

pub trait AdmissionPolicy: Send + Sync {
    type Error;

    fn decide(&self, request: &AdmissionRequest) -> Result<AdmissionDecision, Self::Error>;
}

/// Transport-neutral description of a stateful request whose remote outcome
/// cannot be established. The transport reports it; the runtime decides how
/// to fence and recover the associated scope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UncertainRequest {
    pub request_id: u64,
    pub scope_id: ScopeId,
}

pub trait StatefulRequestRecovery {
    type Error;

    fn recover_uncertain(&mut self, request: UncertainRequest) -> Result<(), Self::Error>;
}

/// Durable lifecycle transitions for execution scopes. This type lives in the
/// shared domain crate so the ledger can encode/replay it without depending
/// on ptr-runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScopeLifecycleKind {
    Created,
    Admitted,
    Started,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    Revoked,
    Released,
}

/// The stateful bindings carried by a scope lifecycle record. Optional fields
/// are absent for ordinary session/turn scopes and present for stateful Pod
/// calls.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ScopeLeaseBinding {
    pub state_id: Option<StateId>,
    pub pod_id: Option<PodId>,
    pub placement_epoch: Option<u64>,
    pub fencing_token: Option<u128>,
    pub generation: Option<Generation>,
    pub resource_lease_id: Option<String>,
}

/// Append-only, replayable scope lifecycle record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScopeLifecycleEvent {
    pub scope_id: ScopeId,
    pub parent_id: Option<ScopeId>,
    pub session_id: SessionId,
    pub project_id: ProjectId,
    pub created_at: Timestamp,
    pub deadline: Option<Timestamp>,
    pub kind: ScopeLifecycleKind,
    pub previous_kind: Option<ScopeLifecycleKind>,
    pub lease: ScopeLeaseBinding,
    pub revision: Revision,
    pub reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Timestamp(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EventRole {
    System,
    User,
    Assistant,
    Tool,
}

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

/// The typed action contract shared by model, routing, runtime and security.
/// `ptr-core::action_head` re-exports this type for compatibility.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionIr {
    pub operation: String,
    pub target: String,
    pub capability: CapabilityId,
    pub effect: Effect,
    pub input_type: TypeId,
    pub generation: Generation,
    pub revision: Revision,
    pub payload: Vec<u8>,
}

/// Exact semantic payload contents exposed to a model backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticPayload {
    pub type_id: TypeId,
    pub source: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticValue {
    Text(String),
    Payload(SemanticPayload),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticEntry {
    pub key: String,
    pub value: SemanticValue,
}

/// Owned, deterministic snapshot contents supplied to a model backend.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SemanticContext {
    pub revision: Revision,
    pub entries: Vec<SemanticEntry>,
}

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
