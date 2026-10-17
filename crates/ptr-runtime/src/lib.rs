pub mod admission;
mod carrier_select;
pub mod compacted;
pub mod directory;
pub mod egress;
pub mod execution;
pub mod expert;
mod handle_admission;
mod host_quota;
pub mod identity;
pub mod kv;
mod kv_derivation;
pub mod kv_infer;
pub mod kv_infer_layered;
pub mod kv_infer_session;
pub mod kv_prefix;
mod kv_secondary;
pub mod kv_taint;
pub mod manifest;
pub mod memory;
pub mod merge;
pub mod migration;
mod moved_tensor;
pub mod neural;
pub mod persistence;
pub mod placement;
pub mod pod_head;
pub mod pod_representation;
pub mod policy;
pub mod protected_state;
pub mod reflex;
pub mod routing;
pub mod sandbox;
pub mod scopes;
pub mod semantic;
pub mod split_stage;
pub mod split_stage_v2;
pub mod stepped;
pub mod tensor_kv;
pub mod tier;
pub mod tier_grant;
pub mod tier_prefetch;
pub mod vcw;
pub mod vcw_host;

pub use admission::{AdmissionError, AdmissionGrant, IdentityAdmissionController};
pub use carrier_select::{
    select_carrier, Calibration, CarrierRefusal, CarrierSelection, CarrierTable, CarrierTableError,
    EndpointCapabilities, EndpointConfinement, HostProcess, PayloadOwnership, PeerProbe,
    SelectionReason, TensorLocation, MAX_DEVICES_PER_NODE,
};
pub use directory::{InMemoryPodDirectory, PodDirectory, PodHeadSource};
pub use handle_admission::{
    revalidate_operation, AdmittedHandle, ConsumeRequest, ConsumedHandle, HandleAdmissionRequest,
    HandlePolicy, HandlePolicyViolation, HandleRejection, HandleSource, HandleTable,
    OperationCheck, OperationFault, OperationLiveness, ReclaimNotice, ReclaimReason,
    RegistrationRefused, Teardown, WorkerSession,
};
pub use host_quota::{
    HostQuotaLedger, HostQuotaPolicy, HostQuotaPolicyViolation, HostQuotaUsage, QuotaCharge,
    QuotaRefusal, ReleaseFault, SharedHostQuota, TrackedPair, VerifiedTeardown,
    DEFAULT_CHARGE_GRANULARITY_BYTES,
};
#[cfg(feature = "oidc-http")]
pub use identity::ReqwestJwksRefresher;
pub use identity::{
    AutheliaForwardAuthAdapter, ForwardAuthHeaders, HttpJwksRefresher, JwksDocument, JwksFetchFn,
    JwksRefresher, JwksStore, OidcAlgorithm, OidcError, OidcIdentityAdapter,
};
pub use kv::{KvUseError, KvUseRequest, KvUseTicket, RuntimeKvAuthority};
pub use kv_derivation::{AdmittedSegments, DerivedRun, DerivedRunRequest};
pub use kv_infer::{
    IndexUpdate, PositionalControl, PositionalOptions, PositionalReport, ReferenceKvError,
    ReferenceKvFaults, ReferenceKvLimits, ReferenceKvLiveness, ReferenceKvOutcome,
    ReferenceKvProbe, ReferenceKvProbeMode, ReferenceKvReuseBackend, ReferenceKvScope,
    ReferenceKvStackShape, REFERENCE_KV_WEIGHT_SEED,
};
pub use kv_infer_layered::{
    BindingError, BindingRecord, BindingSuccessors, BoundExpert, ExpertSuccessor, LayerAssembly,
    LayeredReuseReport, MoeBinding, StackBinding,
};
pub use kv_infer_session::{
    KvCancelReason, KvCancellation, KvSession, KvSessionFinish, KvSessionPhase, KvSessionStats,
    KvSessionStep,
};
pub use kv_prefix::{
    chunk_keys, expert_operand_identities, CatalogLimits, KvCapacityResource, KvCatalogError,
    KvCatalogHit, KvCatalogStats, KvCatalogWiringError, KvChainId, KvChunkKey, KvDependency,
    KvDependencyKind, KvForkError, KvInsert, KvLayerEntry, KvLayerHit, KvLayerQuery,
    KvLayeredInsert, KvLayeredLookup, KvLivenessOracle, KvLookup, KvPageCharge, KvPagedInsert,
    KvPin, KvPrefixCatalog, KvPrefixRetirer, KvRequestPositions, KvRestoreReport, KvSalt,
    KvSnapshotError, SemanticSegmentError, KV_CHUNK_TOKENS, KV_SNAPSHOT_MAX_BYTES,
};
pub use kv_secondary::{
    SecondaryConfig, SecondaryError, SecondaryIndex, SecondaryInsert, SecondaryInsertOutcome,
    SecondaryLimits, SecondaryLineage, SecondaryLookup, SecondaryLookupOutcome, SecondaryPin,
    SecondaryPinState, SecondaryRefusal, SecondaryRefusalCounts, SecondaryRefusalReason,
    SecondaryResource, SecondaryRows, SecondaryRunId, SecondarySectionError, SecondarySplice,
    SecondaryStats, DEFAULT_WINDOW_TOKENS, SECONDARY_SECTION_MAX_BYTES,
};
pub use manifest::{
    validate_and_build, validate_authority, ExecutionManifestRegistry, KnowledgeManifestValidity,
    ManifestAuthorityRegistry, ManifestBindingAuthority, ManifestValidity,
    RuntimeExecutionManifestResolver, RuntimeManifestError, ValidatedExecutionManifest,
    MAX_DEPENDENCY_CLOSURE,
};
pub use merge::{
    ChangeOrigin, HoldReason, MergeAuthority, MergeHold, MergeOutcome, MergePreview, MergeReceipt,
    SemanticChange, SemanticGrant, SemanticGrantInfo, SemanticRefusal, SemanticVerdict,
};
pub use migration::{KvMigrationController, MigrationError, MigrationRecord, MigrationState};
pub use moved_tensor::{MovedTensor, TensorMoveRefused};
pub use placement::{
    DeviceHealth, DeviceRecord, FencedStateLease, FencingToken, NodeHealth, NodeRecord, Placement,
    PlacementEpoch, PlacementError, PodPlacementController, ProjectPlacementRequest,
};
pub use pod_head::{pod_lifecycle_target, AdvancePodHead, PodHeadError, PodHeadStatus};
pub use pod_representation::{
    AdmittedRepresentations, BindPodRepresentation, PodRepresentationError, RepresentationDenial,
    RepresentationRefusal,
};
#[cfg(feature = "oidc-http")]
pub use policy::ReqwestPolicyBundleLoader;
pub use policy::{
    CapabilityPolicy, HttpPolicyBundleLoader, PodPolicyBinding, PolicyBundleLoader,
    PolicyBundlePayload, PolicyError, PolicyFetchFn, PolicyTrustStore, ProjectPolicy,
    SignedPolicyBundle,
};
pub use protected_state::ProtectedStateCoordinator;
pub use reflex::{FallbackConfigError, SetFallbackPlan, MAX_FALLBACK_PLANS};
pub use routing::{EmbeddingSampler, EmbeddingSamplerError, PodSampler, QueryEmbedder, RouteQuery};
pub use sandbox::{
    ExternalSandboxExecutor, NativeSandboxExecutor, NetworkMode, SandboxBackend, SandboxError,
    SandboxExecutor, SandboxLease, SandboxProfile, SandboxRequest, SandboxResponse,
};
pub use scopes::{
    CleanupError, ExecutionScope, ScopeCleanupCoordinator, ScopeError, ScopeEvent, ScopeEventKind,
    ScopeRegistry, ScopeState, MAX_SCOPES,
};
pub use split_stage::{
    admit_frame, admit_frame_in, prefix_key, revalidate_chain, revalidate_chain_in, AdmittedFrame,
    ChainFault, RuntimeStageLiveness, SplitStageCache, StageLiveness, StoreError, UseFault,
    UseTicket, OPEN_USES_PER_ENTRY,
};
pub use split_stage_v2::{
    admit_frame_v2, admit_frame_v2_in, input_target, prefix_key_v2, AdmissionRequestV2,
    AdmittedFrameV2, DownstreamKey, FrameFault, InputLiveness, PersistFault, PersistOp,
    RuntimeInputLiveness, ServedUseV2, SplitStageCacheV2, StoreErrorV2, Tainted, UseFaultV2,
    OPEN_USES_PER_ENTRY_V2,
};
pub use stepped::{CancelPolicy, CancelledRun, RunStep, StepResources, SteppedRun};
pub use tensor_kv::{
    ManagedKvHandle, ManagedKvMetadata, ManagedKvPageContext, ManagedKvRegistry,
    ManagedOnlineKvSnapshot, TensorKvError,
};
pub use tier::{
    backend_lifecycle_digest, replica_lifecycle_digest, ActiveWriteBinding, AdmittedTierObject,
    BackendLifecycleState, BudgetedReplicaPolicy, DeterministicLruPolicy, ExplicitTierPolicy,
    JournaledBackend, JournaledReplica, LookaheadPrefetchPolicy, PrefetchOutcome, ReplicaState,
    ResidencyRecord, TierJournalProjection, TierPin, TierPlacementPolicy, TierPlan,
    TierPlanningContext, TierPolicyError, TierProjectionError, TierReplica,
    TierResidencyController, TierRuntimeError, TierTransferAuthorization, TierTransferReceipt,
};
pub use tier_grant::{
    ManualTierClock, SystemTierClock, TierClock, TierGrant, TierGrantLimits, TOMBSTONE_ENTRY_BYTES,
};
pub use tier_prefetch::{
    plan_prefetch, EligibleTargets, PlannedPrefetch, PodEligibilityGate, PrefetchDrop,
    PrefetchDropReason, PrefetchEligibility, PrefetchIndexError, PrefetchObjectIndex, PrefetchPlan,
    PrefetchPlanError, PrefetchPlanRequest, PrefetchSignal, PrefetchSignalError,
    PrefetchSignalSource, PrefetchTarget, MAX_PREFETCH_INDEX_ENTRIES, MAX_PREFETCH_TARGETS,
};

/// A complete, runtime-bound request to admit one typed output produced by a
/// Pod. The payload itself remains owned by the Pod boundary; this request
/// carries the bindings the runtime must verify before any promotion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodOutputAdmissionRequest {
    pub request_id: RequestId,
    pub manifest_digest: ptr_types::Digest,
    pub manifest: ptr_pods::ExecutionManifest,
    pub pod_id: ptr_types::PodId,
    pub output: ptr_pods::PodOutput,
    pub expected_type: ptr_types::TypeId,
    pub session_id: ptr_types::SessionId,
    pub scope_id: ScopeId,
    pub placement_epoch: PlacementEpoch,
    pub fencing_token: FencingToken,
    /// What the caller knows of the output's KV-reuse provenance. The runtime joins it with
    /// its own registry (the request id, the output identity, its payload and its
    /// dependencies), so an Exact here never un-taints what the runtime recorded.
    /// `StateDelta`, `VerifiedResult` and `ActionProposal` outputs are refused when the join is
    /// Approximate; the other kinds are stored as tainted candidates or hypotheses.
    pub kv_taint: ptr_model_api::KvReuseTaint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionAdmission {
    pub request_id: RequestId,
    pub session_id: ptr_types::SessionId,
    pub scope_id: ScopeId,
    pub project: ProjectId,
    pub pod_id: ptr_types::PodId,
    pub output_digest: ptr_types::Digest,
    pub action_digest: ptr_types::Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PodOutputAdmission {
    ObservationCandidate {
        revision: Revision,
        semantic_key: String,
    },
    Hypothesis {
        branch_id: String,
    },
    StateDelta {
        revision: Revision,
    },
    ActionProposal {
        action: Box<ActionIr>,
        admission: ActionAdmission,
    },
    VerifiedResult {
        revision: Revision,
    },
}

/// Whether the semantic delta `encoded_delta` writes `payload` under `key`, with
/// the same type and the same bytes. A delta that cannot be decoded, does not
/// touch the key or writes a text value carries nothing of the kind.
fn delta_carries_payload(encoded_delta: &[u8], key: &str, payload: &TypedPayload) -> bool {
    ptr_semdb::SemanticDelta::decode(encoded_delta)
        .ok()
        .and_then(|delta| delta.upserts.get(key).cloned())
        .is_some_and(|value| match value {
            SemanticValue::Payload(stored) => {
                stored.type_id == payload.type_id && stored.bytes == payload.bytes
            }
            SemanticValue::Text(_) => false,
        })
}

// The unit tests of vcw_host share the host-side set-up of tests/expert_support, which names the
// crate by its public path.
#[cfg(test)]
extern crate self as ptr_runtime;

#[cfg(test)]
mod delta_payload_tests {
    use super::*;

    fn delta_with(key: &str, type_id: &str, bytes: &[u8]) -> Vec<u8> {
        let mut delta = ptr_semdb::SemanticDelta::default();
        delta.upserts.insert(
            key.to_owned(),
            SemanticPayload {
                type_id: ptr_types::TypeId::from(type_id),
                source: "pod".into(),
                bytes: bytes.to_vec(),
            }
            .into(),
        );
        delta.encode().unwrap()
    }

    fn payload(type_id: &str, bytes: &[u8]) -> TypedPayload {
        TypedPayload {
            type_id: ptr_types::TypeId::from(type_id),
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn a_delta_carries_a_payload_only_under_its_key_with_its_type_and_bytes() {
        let encoded = delta_with("key", "text", b"admitted");
        assert!(delta_carries_payload(
            &encoded,
            "key",
            &payload("text", b"admitted")
        ));
        assert!(!delta_carries_payload(
            &encoded,
            "key",
            &payload("text", b"other")
        ));
        assert!(!delta_carries_payload(
            &encoded,
            "key",
            &payload("json", b"admitted")
        ));
        assert!(!delta_carries_payload(
            &encoded,
            "elsewhere",
            &payload("text", b"admitted")
        ));
        assert!(!delta_carries_payload(
            b"not a delta",
            "key",
            &payload("text", b"admitted")
        ));
    }

    #[test]
    fn a_text_value_is_not_a_payload() {
        let mut delta = ptr_semdb::SemanticDelta::default();
        delta.upserts.insert("key".to_owned(), "admitted".into());
        let encoded = delta.encode().unwrap();
        assert!(!delta_carries_payload(
            &encoded,
            "key",
            &payload("text", b"admitted")
        ));
    }
}

/// The branch a Pod hypothesis is committed under: derived from the request, the
/// Pod and the output digest, so a retry of one admission names the same branch.
fn pod_branch_id(
    request_id: &RequestId,
    pod_id: &ptr_types::PodId,
    output_digest: &ptr_types::Digest,
) -> String {
    format!(
        "pod-branch:{}:{}:{}",
        request_id,
        pod_id,
        digest_hex(output_digest)
    )
}

fn digest_hex(digest: &ptr_types::Digest) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Most runtime events [`PtrRuntime::events`] retains (invariant 8). The ledger is the record of
/// what happened; this log is observability.
pub const MAX_RETAINED_EVENTS: usize = 65_536;

const ACTION_PAYLOAD_VERSION: u8 = 1;
const MAX_ACTION_FIELD_BYTES: usize = 64 * 1024;
const MAX_ACTION_BODY_BYTES: usize = 1024 * 1024;

fn decode_action_payload(bytes: &[u8]) -> Result<ActionIr, String> {
    let mut cursor = 0usize;
    let take = |cursor: &mut usize, count: usize| -> Result<&[u8], String> {
        let end = cursor
            .checked_add(count)
            .ok_or_else(|| "action payload length overflow".to_owned())?;
        let slice = bytes
            .get(*cursor..end)
            .ok_or_else(|| "truncated action payload".to_owned())?;
        *cursor = end;
        Ok(slice)
    };
    let byte = |cursor: &mut usize| -> Result<u8, String> { Ok(take(cursor, 1)?[0]) };
    let u64_value = |cursor: &mut usize| -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            take(cursor, 8)?.try_into().expect("fixed width"),
        ))
    };
    let string = |cursor: &mut usize| -> Result<String, String> {
        let length = u64_value(cursor)? as usize;
        if length > MAX_ACTION_FIELD_BYTES {
            return Err("action field exceeds limit".into());
        }
        String::from_utf8(take(cursor, length)?.to_vec())
            .map_err(|_| "action field is not utf-8".into())
    };
    let body = |cursor: &mut usize| -> Result<Vec<u8>, String> {
        let length = u64_value(cursor)? as usize;
        if length > MAX_ACTION_BODY_BYTES {
            return Err("action body exceeds limit".into());
        }
        Ok(take(cursor, length)?.to_vec())
    };
    if byte(&mut cursor)? != ACTION_PAYLOAD_VERSION {
        return Err("unsupported action payload version".into());
    }
    let operation = string(&mut cursor)?;
    let target = string(&mut cursor)?;
    let capability = ptr_types::CapabilityId(string(&mut cursor)?);
    let input_type = ptr_types::TypeId(string(&mut cursor)?);
    let effect = match byte(&mut cursor)? {
        1 => ptr_types::Effect::Pure,
        2 => ptr_types::Effect::Read,
        3 => ptr_types::Effect::Mutation,
        4 => ptr_types::Effect::External,
        5 => ptr_types::Effect::Irreversible,
        other => return Err(format!("unknown action effect {other}")),
    };
    let generation = Generation(u64_value(&mut cursor)?);
    let revision = Revision(u64_value(&mut cursor)?);
    let payload = body(&mut cursor)?;
    if cursor != bytes.len()
        || operation.is_empty()
        || target.is_empty()
        || capability.0.is_empty()
        || input_type.0.is_empty()
        || generation.0 == 0
    {
        return Err("invalid or trailing action payload".into());
    }
    Ok(ActionIr {
        operation,
        target,
        capability,
        effect,
        input_type,
        generation,
        revision,
        payload,
    })
}

/// Typed bridge owned by the runtime side of the PodWire boundary. It keeps
/// transport crates independent from scope, lease, and journal internals.
pub struct RuntimeRecoveryAdapter<'a, C: ScopeCleanupCoordinator> {
    runtime: &'a mut PtrRuntime,
    scope_id: ScopeId,
    coordinator: &'a mut C,
    lease: ScopeLeaseBinding,
}

impl<'a, C: ScopeCleanupCoordinator> RuntimeRecoveryAdapter<'a, C> {
    pub fn new(
        runtime: &'a mut PtrRuntime,
        scope_id: ScopeId,
        coordinator: &'a mut C,
        lease: ScopeLeaseBinding,
    ) -> Self {
        Self {
            runtime,
            scope_id,
            coordinator,
            lease,
        }
    }
}

impl<C: ScopeCleanupCoordinator> StatefulRequestRecovery for RuntimeRecoveryAdapter<'_, C> {
    type Error = RuntimeError;

    fn recover_uncertain(&mut self, request: UncertainRequest) -> Result<(), Self::Error> {
        if request.scope_id != self.scope_id {
            return Err(RuntimeError::Scope(ScopeError::InvalidLeaseBinding));
        }
        self.runtime.recover_uncertain_request(
            &self.scope_id,
            request.request_id,
            self.coordinator,
            self.lease.clone(),
        )
    }
}

use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_events::{EventEnvelope, RuntimeEvent};
use ptr_ledger::{
    integrity, Attestation, CommittedEvent, FileLedger, InMemoryLedger, Ledger, LedgerEvent,
    MAX_RETAINED_RESPONSE,
};
use ptr_model_api::{
    InferenceBackend, KvReuseTaint, ModelEvent, ModelRequest, ResumableInferenceBackend,
};
use ptr_pods::{EvidenceVerifier, PodEvidenceBundle, PodHypothesis, PodOutputKind, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_security::{
    ActionAuthorization, AuthorizationDecision, AuthorizationDenial, PermissionSet,
};
use ptr_semdb::{
    PreparedDelta, SemanticError, SemanticHost, SemanticPayload, SemanticSnapshot, SemanticValue,
};
use ptr_state::MaterializedState;
use ptr_types::{
    CommitIndex, EvidenceId, Generation, IdentityContext, PrincipalId, Probability, ProjectId,
    ProvenanceRef, RequestId, Revision, ScopeId, ScopeLeaseBinding, StatefulRequestRecovery,
    Timestamp, UncertainRequest, Validity,
};
use ptr_verifier::{VerificationStatus, Verifier};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    Snapshot(persistence::SnapshotError),
    Compacted(compacted::CompactedError),
    Neural(neural::NeuralError),
    Expert(expert::ExpertAdmissionError),
    Semantic(SemanticError),
    InvalidConfig(String),
    Ledger(String),
    Scope(ScopeError),
    ScopeCleanup(CleanupError),
    ScopeRevisionMismatch {
        expected: Revision,
        actual: Revision,
    },
    StaleRevision {
        action: Revision,
        current: Revision,
    },
    /// A stepped run was ended because the semantic revision moved for a reason other than the
    /// run's own writes (`bound` is the revision the run was bound to, `current` the store's),
    /// before it did any more work. Only [`CancelledRun::into_error`] produces it. It is a saving
    /// of work, not a safety check: the effect boundary refuses a stale action
    /// ([`RuntimeError::StaleRevision`]) whatever this check did.
    RunCancelled {
        bound: Revision,
        current: Revision,
    },
    /// A step was requested for a stepped run that already finished, was cancelled or failed.
    SteppedRunEnded {
        request: String,
    },
    UnknownGeneration {
        target: String,
    },
    StaleGeneration {
        target: String,
        action: Generation,
        current: Option<Generation>,
    },
    ReplayIndexMismatch {
        expected: CommitIndex,
        actual: CommitIndex,
    },
    Model(String),
    /// An action derived from a model run that used Approximate KV reuse reached the
    /// hard effect boundary (doc 36, P2a). `report_digest` names the reuse report.
    ApproximateStateAtEffectBoundary {
        report_digest: ptr_types::Digest,
    },
    /// The bounded registry of live taint subjects is full; the run is refused before anything
    /// is appended.
    KvTaintRegistryFull {
        limit: usize,
    },
    /// The taint registry refused a change, a replayed record or a restored state (fail closed).
    KvTaint(kv_taint::KvTaintError),
    /// A pod head move or a stored head was refused (see [`pod_head::PodHeadError`]).
    PodHead(Box<pod_head::PodHeadError>),
    /// A learned pod representation could not be bound to its pod.
    PodRepresentation(pod_representation::PodRepresentationError),
    PodUnavailable {
        capability: String,
        input_type: String,
    },
    /// A fallback plan was refused (see [`reflex::FallbackConfigError`]).
    FallbackConfig(reflex::FallbackConfigError),
    /// The primary path missed and the configured fallback was refused. Names the miss and why
    /// the default pod could not serve; never a pod.
    FailoverRefused {
        miss: ptr_router::MissReason,
        reason: ptr_router::UncoveredReason,
    },
    /// The routing boundary failed while candidates may exist. The reason is a fixed string that
    /// never names a Pod.
    RouteFailed {
        reason: &'static str,
    },
    PodEffectRequiresActionBoundary {
        pod: String,
    },
    Pod(String),
    PodVerificationFailed {
        pod: String,
    },
    PodEvidence(ptr_pods::EvidenceError),
    PodOutputTypeMismatch {
        pod: String,
        expected: String,
        actual: String,
    },
    PodOutputAdmission(String),
    /// A host write was refused before append: the installed grant's
    /// verifiers did not pass at the required level, or reported a hard
    /// finding, named by code.
    SemanticVerificationRejected(merge::SemanticRefusal),
    /// A semantic write that needs a grant, on a runtime with none installed.
    NoSemanticGrant,
    /// A second grant; the first stays installed.
    SemanticGrantInstalled,
    /// A grant that cannot be installed, and why.
    InvalidSemanticGrant {
        reason: &'static str,
    },
    /// A host write under a grant that does not allow host writes.
    HostWritesNotGranted,
    /// A merge under [`MergeAuthority::Triage`] with a grant that has no
    /// merge policy.
    NoMergePolicy,
    /// A merge under [`MergeAuthority::Reviewed`] by a reviewer the grant
    /// does not list.
    UnknownReviewer {
        reviewer: String,
    },
    /// A merge under [`MergeAuthority::Reviewed`] by the principal who authored
    /// the branch. An approval is a second pair of eyes, so a reviewer cannot
    /// give it for their own branch.
    SelfApproval {
        reviewer: String,
    },
    /// Provenance text a record would carry (a host write's principal, a
    /// merged branch's id or author) is not an identifier of at most
    /// [`merge::MAX_PROVENANCE_TEXT`] bytes; `field` names it.
    InvalidProvenanceText {
        field: &'static str,
    },
    /// A branch that is already merged, and where.
    BranchAlreadyMerged {
        branch: String,
        at: CommitIndex,
    },
    /// A branch that does not certify against the runtime's current state,
    /// or whose sealed form is refused, as `ptr_branch` refuses it.
    Certification(ptr_branch::BranchError),
    /// A reviewed merge whose plan is no longer the one approved: the
    /// approved digest, and the digest of the plan certification yields now.
    MergePlanChanged {
        approved: [u8; 32],
        current: [u8; 32],
    },
    /// The grant's merge policy refused to triage, named by the arbiter's
    /// code.
    InvalidMergePolicy {
        code: &'static str,
    },
    /// A merge whose record would be larger than the ledger frames; nothing
    /// was appended.
    MergeRecordTooLarge,
    /// A host write or merge that writes, removes or derives a key only
    /// ingress writes.
    ReservedSemanticNamespace {
        key: String,
    },
    /// A verifier reported a finding code that is not an identifier of at
    /// most [`merge::MAX_FINDING_CODE`] bytes, or soft findings past what a
    /// record carries; the change fails closed.
    InvalidVerificationReport {
        verifier: String,
    },
    /// `commit` was handed a semantic record: those are written only by the
    /// semantic write paths, which build and check them.
    SemanticRecordOutsideSemanticPath,
    /// An expert structural record ([`LedgerEvent::ExpertStructuralCommitted`]) was handed to
    /// the public `commit`. Such a record may only be written by
    /// [`PtrRuntime::admit_expert_proposal`], which checks the proposer, the gate evidence and the
    /// compare-and-set and derives the event's ids from the record. Replay does not use this
    /// error: it re-checks the record and reports [`RuntimeError::Expert`]. An expert fence
    /// record ([`LedgerEvent::ExpertFenceCommitted`]) is refused the same way: it may only be
    /// written by [`PtrRuntime::fence_expert`] and [`PtrRuntime::readmit_expert`]; so is an
    /// expert authority change record ([`LedgerEvent::ExpertAuthorityChanged`]), which only
    /// [`PtrRuntime::change_expert_authority`] may write.
    StructuralRecordOutsideAdmissionPath,
    /// A semantic record written before origins existed, replayed after a
    /// record with an attributed origin (rule R1).
    LegacySemanticRecord {
        index: CommitIndex,
    },
    /// Committing this record would take the execution section of a compacted snapshot past
    /// its bounds ([`compacted::MAX_SECTION_BYTES`] bytes, or more than the section's item bound
    /// in one of its lists). The section holds the unsettled attempts (each reserving room for
    /// its settlement), the settled keys and the scope block, and nothing removes a settled key
    /// or a scope, so it is held to one budget at every commit that grows it. Nothing was
    /// committed; the snapshot this runtime exports stays within the section.
    ExecutionSectionFull {
        /// The bytes the section would take.
        bytes: usize,
        /// The longest of its lists, in items.
        items: usize,
    },
    /// The execution section of this runtime's committed history (a history replayed, opened
    /// or restored from a log written before the execution-section budget) is larger than a
    /// compacted snapshot carries, so no snapshot can be exported; see
    /// [`PtrRuntime::execution_section_overflow`](crate::PtrRuntime::execution_section_overflow).
    /// The log stays authoritative and complete; nothing was written.
    CompactionBlocked {
        /// The bytes the section would take.
        bytes: usize,
        /// The longest of its lists, in items.
        items: usize,
    },
    /// The ledger holds an admission of this request and Pod recorded under the version-1
    /// output digest (before issue #141), its follow-up (hypothesis or promotion) is missing,
    /// and the retry's output has that same version-1 digest. Version 1 is not injective, so
    /// the retry cannot be proven to be the admitted output and the follow-up is not completed
    /// from it: an operator has to recover the admission. Nothing was appended.
    LegacyAdmissionIncomplete {
        /// The request id of the admission.
        request_id: RequestId,
        /// The Pod the admission was recorded for.
        pod_id: ptr_types::PodId,
    },
    ModelResumeLimit {
        max_rounds: usize,
    },
    ModelNoProgress,
    MultiplePodRequests {
        count: usize,
    },
    MultipleControlEvents {
        count: usize,
    },
    MixedControlEvents,
    PermissionDenied,
    ExecutionFenced,
    InvalidLifecycleTransition {
        target: String,
        current: Option<Generation>,
        requested: Generation,
    },
    CapsuleProjectMismatch {
        capsule: String,
        current: String,
        requested: String,
    },
    /// A `pod:<project>/<namespace>/<pod>` capsule was committed under another project than the one
    /// it names.
    PodCapsuleProjectMismatch {
        capsule: String,
        embedded: String,
        project: String,
    },
    /// A `pod:<project>/<namespace>/<pod>` capsule was committed at generation 0, which no pod
    /// revision has (`PodSemanticManifest::validate` refuses it), so it can never become live.
    PodCapsuleGenerationZero {
        capsule: String,
    },
    ReservedTargetNamespace {
        target: String,
    },
    SnapshotBeyondHistory {
        covers: CommitIndex,
        last_applied: CommitIndex,
    },
    /// An at-most-once key is not a well-formed identifier, or, in an attempt
    /// being committed, is longer than [`execution::MAX_KEY_BYTES`], the most a
    /// compacted snapshot carries. An attempt a log already holds is replayed
    /// with a key of any length, as [`execution::MAX_KEY_BYTES`] explains.
    InvalidEffectKey {
        key: String,
    },
    /// The target or the operation of an attempt being committed is empty or longer than
    /// [`execution::MAX_ATTEMPT_FIELD_BYTES`]: a compacted snapshot carries each as a string of 1
    /// to that many bytes, so while such an attempt were unsettled no snapshot could be exported
    /// (issue #153). Nothing was committed. Replay accepts a record a log already holds.
    InvalidEffectAttemptField {
        /// Which field is out of bounds.
        field: execution::EffectAttemptField,
        /// Its length in bytes.
        bytes: usize,
    },
    /// An effect settlement or reconciliation ([`LedgerEvent::EffectSettled`],
    /// [`LedgerEvent::EffectReconciled`]) was handed to the public `commit`. Only the execution
    /// path writes one (a settlement after dispatch, `settle_detached`, `reconcile`), which keeps
    /// a retained response within the compacted snapshot's execution-section budget (issue
    /// #153). Nothing was committed.
    SettlementOutsideExecutionPath,
    /// A second attempt under a key whose first attempt is still unsettled. Two
    /// live attempts would make the key meaningless, and a settlement could not
    /// say which of them it ends.
    EffectKeyInFlight {
        key: String,
    },
    /// A settlement or reconciliation names an attempt that is not awaiting one.
    UnknownEffectAttempt {
        attempt: CommitIndex,
    },
    /// A retained response is larger than this format allows, or does not match
    /// the digest committed beside it.
    InconsistentEffectResponse {
        attempt: CommitIndex,
    },
    /// A semantic record's origin breaks a rule every semantic record is
    /// replayed under (and every writer checks before it appends): an
    /// ingress record of another shape, a host write or merge that touches
    /// an ingress key or carries an attestation that does not hold, or a
    /// merge whose plan digest does not match its delta or whose branch is
    /// already merged. `index` is where the record was committed, or `None`
    /// for one about to be written; `reason` names the rule.
    InvalidSemanticOrigin {
        index: Option<CommitIndex>,
        reason: &'static str,
    },
    /// A segment presented as `Raw` is not the request's own ingress text, or cannot be
    /// checked against it (fail closed). `segment` is its index in the request.
    MisclassifiedRawSegment {
        segment: usize,
        reason: &'static str,
    },
    /// A `Derived` segment names no derivation this runtime minted (a forged, foreign or, after
    /// a restart, clean-and-forgotten id), or carries other content than was minted under it.
    UnknownDerivedSegment {
        segment: usize,
    },
    /// A derived run cannot be turned into a derived segment, so nothing was recorded.
    DerivedRunUnavailable {
        reason: &'static str,
    },
    /// A Pod output of a promoting or acting kind (`StateDelta`, `VerifiedResult`,
    /// `ActionProposal`) whose KV-reuse taint is Approximate was refused before the verifier ran
    /// and before anything was appended.
    ApproximatePodOutput {
        kind: PodOutputKind,
        report_digest: ptr_types::Digest,
    },
    /// A SemDB write (`boundary`) was refused because its request or content is tainted by
    /// Approximate KV reuse; nothing was appended.
    ApproximateSemanticContent {
        boundary: kv_taint::SemanticTaintBoundary,
        report_digest: ptr_types::Digest,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResumableRun {
    pub model_events: Vec<ModelEvent>,
    pub observations: Vec<TypedPayload>,
    pub action: Option<ActionIr>,
    /// Join of the KV-reuse taint of the model output this run ingested; Exact unless the
    /// run came through [`PtrRuntime::ingest_model_run`] with Approximate reuse.
    pub kv_taint: KvReuseTaint,
}

enum RuntimeLedger {
    Memory(InMemoryLedger),
    File(FileLedger),
}

impl RuntimeLedger {
    fn append(&mut self, event: LedgerEvent) -> Result<CommitIndex, RuntimeError> {
        match self {
            Self::Memory(ledger) => ledger
                .append(event)
                .map_err(|error| RuntimeError::Ledger(error.to_string())),
            Self::File(ledger) => ledger
                .append_durable(event)
                .map_err(|error| RuntimeError::Ledger(error.to_string())),
        }
    }

    fn events(&self) -> &[CommittedEvent] {
        match self {
            Self::Memory(ledger) => ledger.events(),
            Self::File(ledger) => ledger.events(),
        }
    }
}

pub struct PtrRuntime {
    execution: execution::ExecutionState,
    pub config: PtrConfig,
    semdb: SemanticHost,
    ledger: RuntimeLedger,
    state: MaterializedState,
    permissions: PermissionSet,
    live_generations: BTreeMap<String, Generation>,
    capsule_projects: BTreeMap<String, String>,
    revoked_generations: BTreeSet<(String, Generation)>,
    events: Vec<EventEnvelope>,
    dropped_events: u64,
    next_event_sequence: u64,
    /// The host's policy for semantic writes that are not ingress; none until
    /// the host installs one. Not history: a runtime rebuilt from its log
    /// has none.
    semantic_grant: Option<merge::SemanticGrant>,
    scopes: ScopeRegistry,
    execution_manifests: Arc<RwLock<ExecutionManifestRegistry>>,
    manifest_authority: Arc<RwLock<ManifestAuthorityRegistry>>,
    policy_trust: policy::PolicyTrustStore,
    /// The attestor keys the host trusts; host configuration, not history.
    attestors: expert::ExpertAttestorRegistry,
    policy_authority_ready: bool,
    revoked_sessions: BTreeSet<ptr_types::SessionId>,
    egress_refusals: egress::EgressRefusalWindows,
    hypotheses: BTreeMap<String, PodHypothesis>,
    tier_journal: TierJournalProjection,
    experts: expert::ExpertRegistryProjection,
    /// Exact KV prefix catalog; none until the host calls `enable_kv_prefix_catalog`.
    kv_catalog: Option<kv_prefix::KvPrefixCatalog>,
    /// The revision at which each semantic key was last affected by a committed delta; the
    /// use-time liveness check of a semantic KV dependency reads it.
    semantic_edits: kv_prefix::SemanticEdits,
    /// Live approximate-KV taint subjects, projected from the committed `KvTaintCommitted`
    /// records; actions in it are refused by `authorize_action`.
    taint: kv_taint::TaintRegistry,
    /// Head of each stable pod identity, projected from the committed `PodHeadAdvanced`
    /// records.
    pod_heads: pod_head::PodHeads,
    /// Fallback plans and reflex counters: host configuration and telemetry, not history.
    reflex: reflex::ReflexState,
}

impl PtrRuntime {
    pub fn new(config: PtrConfig) -> Result<Self, RuntimeError> {
        Self::with_ledger(config, RuntimeLedger::Memory(InMemoryLedger::default()))
    }

    /// [`PtrRuntime::new`] trusting the attestor keys `attestors`, which the host builds from its
    /// own configuration. The trusted set is fixed here: there is no call that changes it later.
    ///
    /// # Errors
    /// As [`PtrRuntime::new`].
    pub fn new_with_attestors(
        config: PtrConfig,
        attestors: expert::ExpertAttestorRegistry,
    ) -> Result<Self, RuntimeError> {
        let mut runtime = Self::new(config)?;
        runtime.attestors = attestors;
        Ok(runtime)
    }

    pub fn open_durable(config: PtrConfig, path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        Self::open_durable_with_attestors(config, path, expert::ExpertAttestorRegistry::new())
    }

    /// [`PtrRuntime::open_durable`] replaying against the attestor keys the host trusts: an
    /// attested expert record replays only when its signature verifies under the key `attestors`
    /// holds for its attestor and epoch. With [`PtrRuntime::open_durable`] the registry is empty
    /// and a ledger holding an attested record does not open.
    ///
    /// # Errors
    /// As [`PtrRuntime::open_durable`], and the attestation refusals of a replayed record.
    pub fn open_durable_with_attestors(
        config: PtrConfig,
        path: impl AsRef<Path>,
        attestors: expert::ExpertAttestorRegistry,
    ) -> Result<Self, RuntimeError> {
        let ledger =
            FileLedger::open(path).map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        Self::from_durable_ledger(config, ledger, attestors)
    }

    fn from_durable_ledger(
        config: PtrConfig,
        ledger: FileLedger,
        attestors: expert::ExpertAttestorRegistry,
    ) -> Result<Self, RuntimeError> {
        let persisted = ledger.events().to_vec();
        let mut runtime = Self::with_ledger(config, RuntimeLedger::File(ledger))?;
        runtime.attestors = attestors;
        for committed in &persisted {
            runtime.validate_lifecycle_event(&committed.event)?;
            let semantic =
                runtime.prepare_semantic_event(Some(committed.index), &committed.event)?;
            runtime.apply_committed(committed, semantic)?;
        }
        runtime.scopes.mark_recovery_required();
        Ok(runtime)
    }

    fn with_ledger(config: PtrConfig, ledger: RuntimeLedger) -> Result<Self, RuntimeError> {
        config.validate().map_err(RuntimeError::InvalidConfig)?;
        Ok(Self {
            execution: execution::ExecutionState::default(),
            config,
            semdb: SemanticHost::default(),
            ledger,
            state: MaterializedState::default(),
            permissions: PermissionSet::default(),
            live_generations: BTreeMap::new(),
            capsule_projects: BTreeMap::new(),
            revoked_generations: BTreeSet::new(),
            events: Vec::new(),
            dropped_events: 0,
            next_event_sequence: 1,
            semantic_grant: None,
            scopes: ScopeRegistry::default(),
            execution_manifests: Arc::new(RwLock::new(ExecutionManifestRegistry::default())),
            manifest_authority: Arc::new(RwLock::new(ManifestAuthorityRegistry::default())),
            policy_trust: policy::PolicyTrustStore::default(),
            attestors: expert::ExpertAttestorRegistry::default(),
            policy_authority_ready: false,
            revoked_sessions: BTreeSet::new(),
            egress_refusals: egress::EgressRefusalWindows::default(),
            hypotheses: BTreeMap::new(),
            tier_journal: TierJournalProjection::default(),
            experts: expert::ExpertRegistryProjection::default(),
            kv_catalog: None,
            semantic_edits: kv_prefix::SemanticEdits::default(),
            taint: kv_taint::TaintRegistry::default(),
            pod_heads: pod_head::PodHeads::default(),
            reflex: reflex::ReflexState::default(),
        })
    }

    pub fn replay(config: PtrConfig, events: &[CommittedEvent]) -> Result<Self, RuntimeError> {
        Self::replay_with_attestors(config, events, expert::ExpertAttestorRegistry::new())
    }

    /// [`PtrRuntime::replay`] against the attestor keys the host trusts; see
    /// [`PtrRuntime::open_durable_with_attestors`].
    ///
    /// # Errors
    /// As [`PtrRuntime::replay`], and the attestation refusals of a replayed record.
    pub fn replay_with_attestors(
        config: PtrConfig,
        events: &[CommittedEvent],
        attestors: expert::ExpertAttestorRegistry,
    ) -> Result<Self, RuntimeError> {
        let mut runtime = Self::new(config)?;
        runtime.attestors = attestors;
        for expected in events {
            runtime.validate_lifecycle_event(&expected.event)?;
            let semantic = runtime.prepare_semantic_event(Some(expected.index), &expected.event)?;
            let actual = runtime.ledger.append(expected.event.clone())?;
            if actual != expected.index {
                return Err(RuntimeError::ReplayIndexMismatch {
                    expected: expected.index,
                    actual,
                });
            }
            let committed = runtime
                .ledger
                .events()
                .last()
                .expect("replay append created committed event")
                .clone();
            runtime.apply_committed(&committed, semantic)?;
        }
        runtime.scopes.mark_recovery_required();
        Ok(runtime)
    }

    pub fn revision(&self) -> Revision {
        self.semdb.revision()
    }

    pub fn snapshot(&self) -> SemanticSnapshot {
        self.semdb.snapshot()
    }

    pub fn permissions_mut(&mut self) -> &mut PermissionSet {
        self.execution.invalidate_pending();
        &mut self.permissions
    }

    /// The most recent runtime events, oldest first: at most [`MAX_RETAINED_EVENTS`]; older
    /// ones are dropped (see [`PtrRuntime::dropped_events`]) and the sequences show the gap.
    pub fn events(&self) -> &[EventEnvelope] {
        &self.events
    }

    /// Runtime events dropped from the front of [`PtrRuntime::events`] because the log was full.
    pub fn dropped_events(&self) -> u64 {
        self.dropped_events
    }

    pub fn committed_events(&self) -> &[CommittedEvent] {
        self.ledger.events()
    }

    pub fn scopes(&self) -> &ScopeRegistry {
        &self.scopes
    }

    pub fn tier_journal(&self) -> &TierJournalProjection {
        &self.tier_journal
    }

    pub fn hypothesis(&self, branch_id: &str) -> Option<&PodHypothesis> {
        self.hypotheses.get(branch_id)
    }

    /// Rebuild a persisted Pod hypothesis as a normal sealed semantic branch
    /// and route it through the existing, grant-controlled `merge_branch`
    /// authority. The hypothesis record itself never publishes semantics.
    pub fn merge_pod_hypothesis(
        &mut self,
        branch_id: &str,
        authority: MergeAuthority,
    ) -> Result<MergeOutcome, RuntimeError> {
        let hypothesis =
            self.hypotheses.get(branch_id).cloned().ok_or_else(|| {
                RuntimeError::PodOutputAdmission("unknown hypothesis branch".into())
            })?;
        let mut branch = ptr_branch::Branch::open(
            ptr_branch::BranchId::from(branch_id),
            PrincipalId::from(hypothesis.pod_id.0.as_str()),
            self.snapshot(),
        );
        let key = format!("pod-hypothesis:{branch_id}");
        branch
            .read(&key)
            .map_err(|error| RuntimeError::PodOutputAdmission(format!("{error:?}")))?;
        branch
            .put(
                &key,
                SemanticValue::Payload(SemanticPayload {
                    type_id: hypothesis.output.type_id.clone(),
                    source: hypothesis.pod_id.to_string(),
                    bytes: hypothesis.output.bytes.clone(),
                }),
            )
            .map_err(|error| RuntimeError::PodOutputAdmission(format!("{error:?}")))?;
        let sealed = branch
            .seal()
            .map_err(|error| RuntimeError::PodOutputAdmission(format!("{error:?}")))?;
        self.merge_branch(&sealed, authority)
    }

    /// Admit an execution manifest only after its Knowledge and Artifact
    /// lineages have been resolved against the supplied authoritative stores.
    /// The resulting immutable record is then held by the runtime registry;
    /// callers cannot replace it with a different record under the same digest.
    pub fn admit_execution_manifest(
        &mut self,
        manifest: ptr_pods::ExecutionManifest,
        knowledge: &ptr_memory::KnowledgeStore,
        artifacts: &dyn ptr_pods::ArtifactCatalog,
        now: Timestamp,
    ) -> Result<manifest::ValidatedExecutionManifest, manifest::RuntimeManifestError> {
        let validated =
            manifest::validate_and_build(manifest, knowledge, artifacts, self.revision(), now)?;
        let manifest_bytes = validated
            .manifest
            .encode_canonical()
            .map_err(|_| manifest::RuntimeManifestError::InvalidManifest)?;
        let mut authority_projection = self
            .manifest_authority
            .read()
            .expect("manifest authority lock poisoned")
            .clone();
        authority_projection.observe_manifest(&validated.manifest)?;
        self.commit(LedgerEvent::ExecutionManifestAdmitted {
            manifest_digest: validated.manifest.manifest_digest,
            generation: validated.manifest.generation,
            revision: validated.runtime_revision,
            policy_revision: validated.manifest.policy_revision,
            manifest: manifest_bytes,
        })
        .map_err(|_| manifest::RuntimeManifestError::InvalidManifest)?;
        self.manifest_authority
            .write()
            .expect("manifest authority lock poisoned")
            .clone_from(&authority_projection);
        Ok(validated)
    }

    /// Strict manifest admission with runtime-owned principal, policy and
    /// snapshot indexes. The legacy structural method remains available for
    /// callers that are still assembling those indexes; production admission
    /// should use this authority-bound entry point.
    pub fn admit_execution_manifest_with_authority<A: manifest::ManifestBindingAuthority>(
        &mut self,
        manifest: ptr_pods::ExecutionManifest,
        knowledge: &ptr_memory::KnowledgeStore,
        artifacts: &dyn ptr_pods::ArtifactCatalog,
        authority: &A,
        now: Timestamp,
    ) -> Result<manifest::ValidatedExecutionManifest, manifest::RuntimeManifestError> {
        let validated =
            manifest::validate_and_build(manifest, knowledge, artifacts, self.revision(), now)?;
        manifest::validate_authority(&validated.manifest, authority)?;
        let manifest_bytes = validated
            .manifest
            .encode_canonical()
            .map_err(|_| manifest::RuntimeManifestError::InvalidManifest)?;
        let mut authority_projection = self
            .manifest_authority
            .read()
            .expect("manifest authority lock poisoned")
            .clone();
        authority_projection.observe_manifest(&validated.manifest)?;
        self.commit(LedgerEvent::ExecutionManifestAdmitted {
            manifest_digest: validated.manifest.manifest_digest,
            generation: validated.manifest.generation,
            revision: validated.runtime_revision,
            policy_revision: validated.manifest.policy_revision,
            manifest: manifest_bytes,
        })
        .map_err(|_| manifest::RuntimeManifestError::InvalidManifest)?;
        self.manifest_authority
            .write()
            .expect("manifest authority lock poisoned")
            .clone_from(&authority_projection);
        Ok(validated)
    }

    pub fn resolve_execution_manifest(
        &self,
        digest: &ptr_types::Digest,
    ) -> Result<manifest::ValidatedExecutionManifest, manifest::RuntimeManifestError> {
        self.execution_manifests
            .read()
            .expect("manifest registry lock poisoned")
            .resolve(digest)
    }

    /// Resolve an admitted manifest for use at `now`: the digest must be admitted and not
    /// revoked, and every knowledge object it binds must still exist, not be invalidated and
    /// hold at `now`. [`Self::resolve_execution_manifest`] is the lookup without that check,
    /// for callers that only inspect the record.
    pub fn resolve_execution_manifest_at(
        &self,
        digest: &ptr_types::Digest,
        knowledge: &ptr_memory::KnowledgeStore,
        now: Timestamp,
    ) -> Result<manifest::ValidatedExecutionManifest, manifest::RuntimeManifestError> {
        let resolved = self.resolve_execution_manifest(digest)?;
        resolved.ensure_knowledge_valid_at(knowledge, now)?;
        Ok(resolved)
    }

    pub fn execution_manifest_registry(&self) -> Arc<RwLock<ExecutionManifestRegistry>> {
        Arc::clone(&self.execution_manifests)
    }

    pub fn manifest_authority_registry(&self) -> Arc<RwLock<ManifestAuthorityRegistry>> {
        Arc::clone(&self.manifest_authority)
    }

    pub fn install_policy_key(
        &mut self,
        key_id: impl Into<String>,
        key: ed25519_dalek::VerifyingKey,
    ) {
        self.policy_trust.insert(key_id, key);
    }

    pub fn revoke_policy_key(&mut self, key_id: impl Into<String>) {
        self.policy_trust.revoke(key_id);
    }

    pub fn policy_authority_ready(&self) -> bool {
        self.policy_authority_ready
    }

    /// Persist a session tombstone before materializing it locally. A
    /// repeated revoke is idempotent and does not append a second mutation.
    pub fn revoke_identity_session(
        &mut self,
        session: ptr_types::SessionId,
        reason: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        if session.0.is_empty() {
            return Err(RuntimeError::InvalidConfig("empty session id".into()));
        }
        if self.revoked_sessions.contains(&session) {
            return Ok(());
        }
        let reason = reason.into();
        if reason.is_empty() {
            return Err(RuntimeError::InvalidConfig(
                "empty session revoke reason".into(),
            ));
        }
        self.commit(LedgerEvent::SessionRevoked {
            session_id: session.clone(),
            reason,
        })?;
        self.revoked_sessions.insert(session);
        Ok(())
    }

    pub fn is_identity_session_revoked(&self, session: &ptr_types::SessionId) -> bool {
        self.revoked_sessions.contains(session)
    }

    pub fn admit_identity_context(
        &mut self,
        identity: &IdentityContext,
        now: Timestamp,
    ) -> Result<PrincipalId, RuntimeError> {
        if self.revoked_sessions.contains(&identity.session_id) {
            return Err(RuntimeError::InvalidConfig(
                "identity session revoked".into(),
            ));
        }
        self.manifest_authority
            .write()
            .expect("manifest authority lock poisoned")
            .admit_identity(identity, now)
            .map_err(|error| {
                RuntimeError::InvalidConfig(format!("identity admission failed: {error:?}"))
            })
    }

    pub fn refresh_snapshot_authority(
        &mut self,
    ) -> Result<(Revision, ptr_types::Digest), RuntimeError> {
        let snapshot = self.snapshot();
        let revision = snapshot.revision;
        let digest = snapshot.canonical_digest();
        self.manifest_authority
            .write()
            .expect("manifest authority lock poisoned")
            .register_snapshot(revision, digest)
            .map_err(|error| {
                RuntimeError::InvalidConfig(format!("snapshot authority failed: {error:?}"))
            })?;
        Ok((revision, digest))
    }

    /// Add a digest already verified by `ProtectedStateCoordinator` to the
    /// runtime authority projection. The protected store remains the owner of
    /// ciphertext and key validation; this method only binds its result to the
    /// current SemDB snapshot.
    pub fn register_protected_snapshot_authority(
        &mut self,
        revision: Revision,
        digest: ptr_types::Digest,
    ) -> Result<(), manifest::RuntimeManifestError> {
        let mut projection = self
            .manifest_authority
            .read()
            .expect("manifest authority lock poisoned")
            .clone();
        projection.register_protected_snapshot(revision, digest)?;
        *self
            .manifest_authority
            .write()
            .expect("manifest authority lock poisoned") = projection;
        Ok(())
    }

    pub fn admit_execution_manifest_from_runtime(
        &mut self,
        manifest: ptr_pods::ExecutionManifest,
        knowledge: &ptr_memory::KnowledgeStore,
        artifacts: &dyn ptr_pods::ArtifactCatalog,
        identity: &IdentityContext,
        now: Timestamp,
    ) -> Result<manifest::ValidatedExecutionManifest, manifest::RuntimeManifestError> {
        let principal = PrincipalId(format!("{}:{}", identity.issuer, identity.subject));
        if !identity.is_valid_at(now) || manifest.principal != principal {
            return Err(manifest::RuntimeManifestError::PrincipalNotAdmitted(
                principal,
            ));
        }
        if !self.policy_authority_ready {
            return Err(manifest::RuntimeManifestError::PolicyRevisionMissing);
        }
        self.admit_identity_context(identity, now)
            .map_err(|_| manifest::RuntimeManifestError::PrincipalNotAdmitted(principal.clone()))?;
        let (live_revision, live_digest) = self.refresh_snapshot_authority().map_err(|_| {
            manifest::RuntimeManifestError::UnknownSnapshotRevision(manifest.snapshot_revision)
        })?;
        if manifest.snapshot_revision != live_revision {
            return Err(manifest::RuntimeManifestError::SnapshotRevisionMismatch {
                expected: live_revision,
                actual: manifest.snapshot_revision,
            });
        }
        let authority = self
            .manifest_authority
            .read()
            .expect("manifest authority lock poisoned")
            .clone();
        authority.validate_protected_snapshot(manifest.snapshot_revision, live_digest)?;
        self.admit_execution_manifest_with_authority(
            manifest, knowledge, artifacts, &authority, now,
        )
    }

    pub fn activate_policy_bundle(
        &mut self,
        bundle: policy::SignedPolicyBundle,
        now: Timestamp,
    ) -> Result<(), RuntimeError> {
        bundle.verify(&self.policy_trust, now).map_err(|error| {
            RuntimeError::InvalidConfig(format!("policy verification failed: {error:?}"))
        })?;
        let digest = bundle.payload_digest;
        let mut projection = self
            .manifest_authority
            .read()
            .expect("manifest authority lock poisoned")
            .clone();
        projection
            .activate_policy(bundle.revision, digest)
            .map_err(|error| {
                RuntimeError::InvalidConfig(format!("policy activation failed: {error:?}"))
            })?;
        let key_id = bundle.key_id.clone();
        let encoded = bundle.encode_for_ledger();
        self.commit(LedgerEvent::PolicyBundleActivated {
            revision: bundle.revision,
            key_id,
            bundle_digest: digest,
            bundle: encoded,
        })?;
        self.manifest_authority
            .write()
            .expect("manifest authority lock poisoned")
            .clone_from(&projection);
        self.policy_authority_ready = true;
        Ok(())
    }

    pub fn revoke_policy_revision(
        &mut self,
        revision: Revision,
        reason: impl Into<String>,
    ) -> Result<(), RuntimeError> {
        let current_projection = self
            .manifest_authority
            .read()
            .expect("manifest authority lock poisoned")
            .clone();
        if current_projection.is_policy_revoked(revision) {
            return Ok(());
        }
        let digest = current_projection
            .policy_digest(revision)
            .ok_or_else(|| RuntimeError::InvalidConfig("unknown policy revision".into()))?;
        let mut projection = self
            .manifest_authority
            .read()
            .expect("manifest authority lock poisoned")
            .clone();
        projection.revoke_policy(revision).map_err(|error| {
            RuntimeError::InvalidConfig(format!("policy revocation failed: {error:?}"))
        })?;
        self.commit(LedgerEvent::PolicyBundleRevoked {
            revision,
            bundle_digest: digest,
            reason: reason.into(),
        })?;
        self.manifest_authority
            .write()
            .expect("manifest authority lock poisoned")
            .clone_from(&projection);
        Ok(())
    }

    pub fn revoke_execution_manifest(
        &mut self,
        digest: ptr_types::Digest,
    ) -> Result<(), manifest::RuntimeManifestError> {
        if self
            .execution_manifests
            .read()
            .expect("manifest registry lock poisoned")
            .is_revoked(&digest)
        {
            return Ok(());
        }
        let current = self.resolve_execution_manifest(&digest)?;
        self.commit(LedgerEvent::ExecutionManifestRevoked {
            manifest_digest: digest,
            generation: current.manifest.generation,
            revision: self.revision(),
        })
        .map(|_| ())
        .map_err(|_| manifest::RuntimeManifestError::UnknownManifestDigest(digest))
    }

    pub fn create_scope(
        &mut self,
        scope: ExecutionScope,
        lease: ScopeLeaseBinding,
    ) -> Result<CommitIndex, RuntimeError> {
        ScopeRegistry::validate_lease_shape(&lease).map_err(RuntimeError::Scope)?;
        if self.scopes.get(&scope.id).is_ok() {
            return Err(RuntimeError::Scope(ScopeError::Duplicate(scope.id)));
        }
        if let Some(parent_id) = &scope.parent {
            let parent = self.scopes.get(parent_id).map_err(RuntimeError::Scope)?;
            if parent.session != scope.session || parent.project != scope.project {
                return Err(RuntimeError::Scope(ScopeError::ParentScopeMismatch));
            }
        }
        let event = scopes::lifecycle_event(
            &scope,
            None,
            ScopeState::Created,
            self.revision(),
            lease,
            None,
        );
        self.commit(LedgerEvent::ScopeLifecycle(event))
    }

    pub fn transition_scope(
        &mut self,
        id: &ScopeId,
        next: ScopeState,
        lease: ScopeLeaseBinding,
        reason: Option<String>,
    ) -> Result<CommitIndex, RuntimeError> {
        let current = self.scopes.get(id).map_err(RuntimeError::Scope)?.clone();
        ScopeRegistry::validate_lease_shape(&lease).map_err(RuntimeError::Scope)?;
        if self.scopes.lease(id).map_err(RuntimeError::Scope)? != &lease {
            return Err(RuntimeError::Scope(ScopeError::InvalidLeaseBinding));
        }
        if self.scopes.recovery_required(id)
            && !matches!(
                next,
                ScopeState::Cancelled | ScopeState::Revoked | ScopeState::Released
            )
        {
            return Err(RuntimeError::Scope(ScopeError::RecoveryRequired(
                id.clone(),
            )));
        }
        let event = scopes::lifecycle_event(
            &current,
            Some(current.state),
            next,
            self.revision(),
            lease,
            reason,
        );
        self.commit(LedgerEvent::ScopeLifecycle(event))
    }

    /// Explicitly recover a scope left open by a restart or session loss. The
    /// normal cleanup coordinator remains the only path that releases its
    /// external leases and session resources.
    pub fn recover_scope<C: ScopeCleanupCoordinator>(
        &mut self,
        id: &ScopeId,
        coordinator: &mut C,
        lease: ScopeLeaseBinding,
    ) -> Result<(), RuntimeError> {
        if !self.scopes.recovery_required(id) {
            return Err(RuntimeError::Scope(ScopeError::InvalidCommittedEvent));
        }
        self.cleanup_scope(id, coordinator, lease)
    }

    /// Recover a scope after a stateful PodWire request returned an outcome
    /// that cannot be established. The transport layer reports the request
    /// id; this runtime method fences the scope, journalizes revocation, and
    /// runs the same idempotent cleanup path used for restart recovery. A
    /// caller must establish a new session and re-admit/recompute before
    /// issuing another stateful request.
    pub fn recover_uncertain_request<C: ScopeCleanupCoordinator>(
        &mut self,
        id: &ScopeId,
        request_id: u64,
        coordinator: &mut C,
        lease: ScopeLeaseBinding,
    ) -> Result<(), RuntimeError> {
        self.scopes
            .mark_scope_recovery_required(id)
            .map_err(RuntimeError::Scope)?;
        self.transition_scope(
            id,
            ScopeState::Revoked,
            lease.clone(),
            Some(format!("request-uncertain:{request_id}")),
        )?;
        self.cleanup_scope(id, coordinator, lease)
    }

    pub fn cleanup_scope<C: ScopeCleanupCoordinator>(
        &mut self,
        id: &ScopeId,
        coordinator: &mut C,
        lease: ScopeLeaseBinding,
    ) -> Result<(), RuntimeError> {
        if self.scopes.lease(id).map_err(RuntimeError::Scope)? != &lease {
            return Err(RuntimeError::Scope(ScopeError::InvalidLeaseBinding));
        }
        let scope = self.scopes.get(id).map_err(RuntimeError::Scope)?.clone();
        coordinator
            .stop_intake(&scope)
            .map_err(RuntimeError::ScopeCleanup)?;
        coordinator
            .cancel_children(&scope)
            .map_err(RuntimeError::ScopeCleanup)?;
        for child in self.scopes.children(id) {
            if self.scopes.get(&child).map_err(RuntimeError::Scope)?.state != ScopeState::Released {
                let child_lease = self
                    .scopes
                    .lease(&child)
                    .map_err(RuntimeError::Scope)?
                    .clone();
                self.cleanup_scope(&child, coordinator, child_lease)?;
            }
        }
        if !scope.state.is_terminal() {
            self.transition_scope(
                id,
                ScopeState::Cancelled,
                lease.clone(),
                Some("cleanup".to_owned()),
            )?;
        }
        let refreshed = self.scopes.get(id).map_err(RuntimeError::Scope)?.clone();
        coordinator
            .drain_queues(&refreshed)
            .map_err(RuntimeError::ScopeCleanup)?;
        coordinator
            .release_pod_lease(&refreshed)
            .map_err(RuntimeError::ScopeCleanup)?;
        coordinator
            .release_resource_lease(&refreshed)
            .map_err(RuntimeError::ScopeCleanup)?;
        coordinator
            .close_session(&refreshed)
            .map_err(RuntimeError::ScopeCleanup)?;
        if refreshed.state != ScopeState::Released {
            self.transition_scope(id, ScopeState::Released, lease, Some("cleanup".to_owned()))?;
        }
        Ok(())
    }

    pub fn materialized_state(&self) -> &MaterializedState {
        &self.state
    }

    fn set_live_generation(&mut self, target: impl Into<String>, generation: Generation) {
        self.live_generations.insert(target.into(), generation);
    }

    pub fn live_generation(&self, target: &str) -> Option<Generation> {
        self.live_generations.get(target).copied()
    }

    /// The project a capsule was committed under, if the lifecycle authority knows the capsule.
    #[must_use]
    pub fn capsule_project(&self, capsule: &str) -> Option<&str> {
        self.capsule_projects.get(capsule).map(String::as_str)
    }

    /// Every lifecycle target and its current generation, ordered by target.
    /// Revocation tombstones are exposed separately by [`Self::revoked_generations`].
    pub fn live_generations(&self) -> impl Iterator<Item = (&str, Generation)> {
        self.live_generations
            .iter()
            .map(|(target, generation)| (target.as_str(), *generation))
    }

    /// Every revoked target/generation pair, including targets without a live generation.
    pub fn revoked_generations(&self) -> impl Iterator<Item = (&str, Generation)> {
        self.revoked_generations
            .iter()
            .map(|(target, generation)| (target.as_str(), *generation))
    }

    /// Whether `target` at `generation` may be used now, as the lifecycle
    /// authority sees it.
    ///
    /// [`Self::live_generation`] alone cannot answer this: a revocation adds a
    /// tombstone and leaves the live generation in place, so a revoked
    /// generation still reads as live there. This combines the tombstone set
    /// with generation equality, the way neural-state admission does, and is
    /// the check every derived hit (search, fast memory, projection row) must
    /// pass before it is used. `None` means the authority knows no such
    /// generation: the target is unknown or the generation is ahead of it.
    pub fn generation_validity(&self, target: &str, generation: Generation) -> Option<Validity> {
        if self
            .revoked_generations
            .contains(&(target.to_owned(), generation))
        {
            return Some(Validity::Revoked);
        }
        match self.live_generations.get(target) {
            Some(live) if *live == generation => Some(Validity::Live),
            Some(live) if *live > generation => Some(Validity::Superseded),
            Some(_) | None => None,
        }
    }

    /// Run one request whose model output comes as a [`ptr_model_api::ModelRun`] (a KV-reusing
    /// backend), continuing the resumable flow through [`PtrRuntime::ingest_model_run`].
    ///
    /// The request text is ingested first, exactly as in the plain resumable flow, and
    /// `infer` receives the resulting semantic context. The run is never turned into a
    /// [`ResumableRun`] except by `ingest_model_run`, so an Approximate run leaves its
    /// action recorded as tainted and `authorize_action` refuses it. This path does not
    /// resume Pods: a run that asks for one is refused with [`RuntimeError::Model`], and a
    /// run that neither finishes nor yields an action is [`RuntimeError::ModelNoProgress`].
    ///
    /// If the request text is byte-identical to a live tainted content subject (tainted model
    /// output re-presented as request text), the run carries that Approximate taint although
    /// its own reuse was Exact: its action is recorded and refused at `authorize_action`. Only
    /// the exact digest of the whole text is matched; a paraphrase, substring or whitespace
    /// edit is not detected, and honest identical text is tainted too (fail closed).
    pub fn run_model_run_resumable<F>(
        &mut self,
        request_id: RequestId,
        raw_text: impl Into<String>,
        infer: F,
    ) -> Result<ResumableRun, RuntimeError>
    where
        F: FnOnce(
            ptr_types::SemanticContext,
        ) -> Result<ptr_model_api::ModelRun, ptr_model_api::ModelError>,
    {
        let raw_text = raw_text.into();
        // Raw text that is exactly a live tainted content subject (tainted output re-presented
        // as request text) keeps its taint: the run's action is recorded as Approximate. So does
        // a request id that is already a live Request subject (a reused id of an Approximate run).
        let raw_taint = self.input_kv_taint(&request_id, &raw_text);
        let _revision = self.ingest_text(request_id.clone(), raw_text)?;
        let model_run = infer(self.snapshot().semantic_context())
            .map_err(|error| RuntimeError::Model(error.0))?;
        // The only place a ModelRun becomes a ResumableRun (KV taint is recorded here).
        let run = self.ingest_model_run_with_input(&request_id, raw_taint, &model_run)?;
        if model_run
            .events
            .iter()
            .any(|event| matches!(event, ModelEvent::PodRequested { .. }))
        {
            return Err(RuntimeError::Model(
                "a segmented (KV-reusing) run cannot resume a Pod".into(),
            ));
        }
        let finished = run
            .model_events
            .iter()
            .any(|event| matches!(event, ModelEvent::Finished));
        if run.action.is_some() {
            return Ok(run);
        }
        if !finished {
            return Err(RuntimeError::ModelNoProgress);
        }
        self.emit(RuntimeEvent::RequestFinished(request_id));
        Ok(run)
    }

    /// Run one request through a plain backend and return its events.
    ///
    /// Input taint (the raw text equals a live tainted content subject, or the request id is a
    /// live Request subject) is computed before the backend runs, as in
    /// [`PtrRuntime::run_model_run_resumable`]. If it is Approximate, the Request subject, the
    /// generated text and every action of the events are recorded taint-first before they are
    /// returned, so `authorize_action` refuses those actions. A backend cannot report KV reuse
    /// here, so input taint is the only taint this path sees. Exact input appends nothing.
    ///
    /// # Errors
    /// [`RuntimeError::KvTaintRegistryFull`] when recording would exceed the taint bound (the
    /// events are not returned), or any ingress, backend or ledger error.
    pub fn run_model_once<B: InferenceBackend + ?Sized>(
        &mut self,
        request_id: RequestId,
        raw_text: impl Into<String>,
        backend: &B,
    ) -> Result<Vec<ModelEvent>, RuntimeError> {
        let raw_text = raw_text.into();
        // Same input taint as `run_model_run_resumable`, computed before the backend runs.
        let input_taint = self.input_kv_taint(&request_id, &raw_text);
        let _revision = self.ingest_text(request_id.clone(), raw_text.clone())?;
        let events = backend
            .infer(&ModelRequest {
                request_id: request_id.clone(),
                semantic_context: self.snapshot().semantic_context(),
                raw_text,
            })
            .map_err(|error| RuntimeError::Model(error.0))?;
        // Approximate input: the request, the generated text and every action are recorded
        // taint-first before the events are returned (and so before any Pod output of the
        // request can be promoted). Exact input appends nothing.
        self.record_input_taint(&request_id, input_taint, &events)?;
        if events
            .iter()
            .any(|event| matches!(event, ModelEvent::Finished))
        {
            self.emit(RuntimeEvent::RequestFinished(request_id));
        }
        Ok(events)
    }

    /// [`PtrRuntime::run_model_once`], then invoke and promote each requested Pod. Taint of the
    /// input is recorded by `run_model_once` before any Pod output is promoted, so a request
    /// with Approximate input has every Pod output refused with
    /// [`RuntimeError::ApproximateSemanticContent`].
    pub fn run_model_with_pods<B, V>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        pods: &PodRegistry,
        verifier: &V,
    ) -> Result<Vec<TypedPayload>, RuntimeError>
    where
        B: InferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        let model_events = self.run_model_once(request_id.clone(), raw_text, backend)?;
        let mut outputs = Vec::new();

        for event in model_events {
            let ModelEvent::PodRequested {
                capability,
                input_type,
                payload,
            } = event
            else {
                continue;
            };

            // A Pod in another project is unavailable in exactly the same
            // words as a Pod that does not exist. Saying which it was would
            // answer, across the boundary, whether that Pod exists. The input
            // taint was recorded by `run_model_once`.
            let dispatched = self.dispatch_pod(
                &routing::PodSelection::Plain,
                &reflex::PodRequest {
                    pods,
                    project,
                    request_id: &request_id,
                    capability: &capability,
                    input_type: &input_type,
                    payload: &payload,
                },
                None,
                verifier,
            )?;
            let output = dispatched.output;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Convenience wrapper that selects Pods with the plain, unsampled [`ptr_router::PodRouter::select`].
    /// It records no draw, propensity, nonce or key identity; the production path is
    /// [`PtrRuntime::run_resumable_with_pods_using_router`] with a [`PodSampler`].
    pub fn run_resumable_with_pods<B, V>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        pods: &PodRegistry,
        verifier: &V,
        max_rounds: usize,
    ) -> Result<ResumableRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        self.run_resumable_inner(
            request_id, project, raw_text, backend, pods, None, verifier, max_rounds,
        )
    }

    /// Like [`PtrRuntime::run_resumable_with_pods`], but every Pod is chosen through the
    /// trusted checked routing path of `router`. Each selection emits
    /// [`RuntimeEvent::PodRouted`] (Pod, propensity, nonce, draw and key identity) before
    /// [`RuntimeEvent::PodInvoked`]; `Ok(None)` is the usual `PodUnavailable` and a routing
    /// error is `RouteFailed` with a Pod-free reason.
    #[allow(clippy::too_many_arguments)]
    pub fn run_resumable_with_pods_using_router<B, V>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        pods: &PodRegistry,
        router: &dyn PodSampler,
        verifier: &V,
        max_rounds: usize,
    ) -> Result<ResumableRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        self.run_resumable_inner(
            request_id,
            project,
            raw_text,
            backend,
            pods,
            Some(router),
            verifier,
            max_rounds,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_resumable_inner<B, V>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        pods: &PodRegistry,
        router: Option<&dyn PodSampler>,
        verifier: &V,
        max_rounds: usize,
    ) -> Result<ResumableRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        self.run_resumable_blocking(
            request_id, project, raw_text, backend, pods, router, verifier, max_rounds,
        )
    }

    pub fn authorization_decision(&self, action: &ActionIr) -> AuthorizationDecision {
        let current_generation = self.live_generations.get(&action.target).copied();
        self.permissions.authorize(ActionAuthorization {
            target: action.target.clone(),
            capability: action.capability.clone(),
            effect: action.effect,
            action_revision: action.revision,
            current_revision: self.semdb.revision(),
            action_generation: action.generation,
            current_generation,
            generation_revoked: self
                .revoked_generations
                .contains(&(action.target.clone(), action.generation)),
            require_capability: self.config.action_boundary.require_capability,
            require_current_revision: self.config.action_boundary.require_current_revision,
            require_live_generation: self.config.action_boundary.require_live_generation,
        })
    }

    pub fn authorize_action(&self, action: &ActionIr) -> Result<(), RuntimeError> {
        self.check_kv_taint(action, KvReuseTaint::Exact)?;
        match self.authorization_decision(action) {
            AuthorizationDecision::Allow(_) => Ok(()),
            AuthorizationDecision::Deny(AuthorizationDenial::StaleRevision { action, current }) => {
                Err(RuntimeError::StaleRevision { action, current })
            }
            AuthorizationDecision::Deny(AuthorizationDenial::UnknownGeneration { target }) => {
                Err(RuntimeError::UnknownGeneration { target })
            }
            AuthorizationDecision::Deny(AuthorizationDenial::StaleGeneration {
                target,
                action,
                current,
            }) => Err(RuntimeError::StaleGeneration {
                target,
                action,
                current,
            }),
            AuthorizationDecision::Deny(
                AuthorizationDenial::MissingCapability { .. }
                | AuthorizationDenial::EffectNotPermitted { .. },
            ) => Err(RuntimeError::PermissionDenied),
        }
    }

    pub fn commit(&mut self, event: LedgerEvent) -> Result<CommitIndex, RuntimeError> {
        // Validate before the first durable byte: rejected transitions must never
        // poison committed history or become authoritative on a later restart.
        // A settlement is refused by name before the fence check: it is the one record kind
        // that is written while the runtime is fenced, and only by the execution path, which
        // holds its retained response to the execution-section budget (issue #153).
        if matches!(
            event,
            LedgerEvent::EffectSettled { .. } | LedgerEvent::EffectReconciled { .. }
        ) {
            return Err(RuntimeError::SettlementOutsideExecutionPath);
        }
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        validate_new_record(&event)?;
        // A new scope must still fit a compacted snapshot (invariant 8). Only new records are
        // held to it; replay accepts a history written before the bound.
        if let LedgerEvent::ScopeLifecycle(scope_event) = &event {
            self.scopes
                .check_capacity(scope_event)
                .map_err(RuntimeError::Scope)?;
        }
        // One budget for the whole execution section of a compacted snapshot (invariant 8): a
        // new attempt (with room for its settlement) or a new scope is refused before it is
        // committed if the section would no longer fit.
        self.check_execution_section_growth(&event)?;
        self.validate_lifecycle_event(&event)?;
        let semantic = self.prepare_semantic_event(None, &event)?;
        self.append_prepared(event, semantic)
    }

    pub fn commit_tier_backend_lifecycle(
        &mut self,
        backend: ptr_storage::TierBackendId,
        tier: ptr_storage::StorageTier,
        state: BackendLifecycleState,
        revision: Revision,
    ) -> Result<CommitIndex, RuntimeError> {
        let event_digest = backend_lifecycle_digest(&backend, tier, state, revision);
        self.commit(LedgerEvent::TierBackendLifecycle {
            backend_id: backend.0,
            tier: tier.code(),
            state: state.code(),
            revision,
            event_digest,
        })
    }

    pub fn commit_tier_object(
        &mut self,
        object: &ptr_storage::TierObjectManifest,
    ) -> Result<CommitIndex, RuntimeError> {
        let manifest = object.encode_canonical().map_err(|error| {
            RuntimeError::InvalidConfig(format!("invalid tier object: {error:?}"))
        })?;
        self.commit(LedgerEvent::TierObjectCommitted {
            root_digest: object.root_digest,
            generation: object.generation,
            revision: object.revision,
            manifest,
        })
    }

    pub fn commit_tier_replica_lifecycle(
        &mut self,
        object: ptr_types::Digest,
        backend: ptr_storage::TierBackendId,
        tier: ptr_storage::StorageTier,
        state: ReplicaState,
        generation: Generation,
        revision: Revision,
    ) -> Result<CommitIndex, RuntimeError> {
        let event_digest =
            replica_lifecycle_digest(object, &backend, tier, state, generation, revision);
        self.commit(LedgerEvent::TierReplicaLifecycle {
            root_digest: object,
            backend_id: backend.0,
            tier: tier.code(),
            state: state.code(),
            generation,
            revision,
            event_digest,
        })
    }

    /// Attach one physical tier backend while recording every durable
    /// lifecycle transition in the authoritative runtime ledger.
    pub async fn attach_tier_backend_journaled(
        &mut self,
        controller: &TierResidencyController,
        backend: Arc<dyn ptr_storage::TierBackend>,
        first_revision: Revision,
    ) -> Result<(), RuntimeError> {
        let id = backend.identity();
        let tier = backend.capabilities().tier;
        let health_revision = Revision(
            first_revision
                .0
                .checked_add(1)
                .ok_or_else(|| RuntimeError::InvalidConfig("tier revision exhausted".into()))?,
        );
        let available_revision = Revision(
            first_revision
                .0
                .checked_add(2)
                .ok_or_else(|| RuntimeError::InvalidConfig("tier revision exhausted".into()))?,
        );
        self.commit_tier_backend_lifecycle(
            id.clone(),
            tier,
            BackendLifecycleState::Configured,
            first_revision,
        )?;
        if let Err(error) = controller.attach_backend(backend).await {
            self.commit_tier_backend_lifecycle(
                id,
                tier,
                BackendLifecycleState::Revoked,
                health_revision,
            )?;
            return Err(RuntimeError::InvalidConfig(format!(
                "tier backend attach failed: {error:?}"
            )));
        }
        self.commit_tier_backend_lifecycle(
            id.clone(),
            tier,
            BackendLifecycleState::HealthChecked,
            health_revision,
        )?;
        self.commit_tier_backend_lifecycle(
            id,
            tier,
            BackendLifecycleState::Available,
            available_revision,
        )?;
        Ok(())
    }

    /// Register an already materialized source replica journal-first. A crash
    /// after `Preparing` leaves an explicit recovery record instead of a
    /// silently authoritative replica.
    pub async fn register_tier_object_journaled(
        &mut self,
        controller: &TierResidencyController,
        object: ptr_storage::TierObjectManifest,
        backend: ptr_storage::TierBackendId,
        preparing_revision: Revision,
    ) -> Result<(), RuntimeError> {
        let backend_projection = self
            .tier_journal
            .backend(&backend)
            .ok_or_else(|| RuntimeError::InvalidConfig("tier backend is not journaled".into()))?;
        if backend_projection.state != BackendLifecycleState::Available {
            return Err(RuntimeError::InvalidConfig(
                "tier backend is not durably available".into(),
            ));
        }
        let tier = controller.backend_tier(&backend).map_err(|error| {
            RuntimeError::InvalidConfig(format!("tier backend unavailable: {error:?}"))
        })?;
        if backend_projection.tier != tier {
            return Err(RuntimeError::InvalidConfig(
                "tier backend projection mismatch".into(),
            ));
        }
        let available_revision = Revision(
            preparing_revision
                .0
                .checked_add(1)
                .ok_or_else(|| RuntimeError::InvalidConfig("tier revision exhausted".into()))?,
        );
        self.commit_tier_object(&object)?;
        self.commit_tier_replica_lifecycle(
            object.root_digest,
            backend.clone(),
            tier,
            ReplicaState::Preparing,
            object.generation,
            preparing_revision,
        )?;
        controller
            .register_object(object.clone(), backend.clone())
            .await
            .map_err(|error| {
                RuntimeError::InvalidConfig(format!("tier object registration failed: {error:?}"))
            })?;
        self.commit_tier_replica_lifecycle(
            object.root_digest,
            backend,
            tier,
            ReplicaState::Available,
            object.generation,
            available_revision,
        )?;
        Ok(())
    }

    /// Transfer a replica through the physical controller and publish it only
    /// through matching `Preparing -> Available` ledger records.
    pub async fn transfer_tier_replica_journaled(
        &mut self,
        controller: &TierResidencyController,
        object: ptr_types::Digest,
        source: &ptr_storage::TierBackendId,
        destination: &ptr_storage::TierBackendId,
        authorization: Option<TierTransferAuthorization>,
        preparing_revision: Revision,
    ) -> Result<TierTransferReceipt, RuntimeError> {
        let record = controller.residency(&object).map_err(|error| {
            RuntimeError::InvalidConfig(format!("tier object unavailable: {error:?}"))
        })?;
        let tier = controller.backend_tier(destination).map_err(|error| {
            RuntimeError::InvalidConfig(format!("tier destination unavailable: {error:?}"))
        })?;
        let terminal_revision = Revision(
            preparing_revision
                .0
                .checked_add(1)
                .ok_or_else(|| RuntimeError::InvalidConfig("tier revision exhausted".into()))?,
        );
        self.commit_tier_replica_lifecycle(
            object,
            destination.clone(),
            tier,
            ReplicaState::Preparing,
            record.object.generation,
            preparing_revision,
        )?;
        match controller
            .transfer(object, source, destination, authorization)
            .await
        {
            Ok(receipt) => {
                self.commit_tier_replica_lifecycle(
                    object,
                    destination.clone(),
                    tier,
                    ReplicaState::Available,
                    record.object.generation,
                    terminal_revision,
                )?;
                Ok(receipt)
            }
            Err(error) => {
                self.commit_tier_replica_lifecycle(
                    object,
                    destination.clone(),
                    tier,
                    ReplicaState::Corrupt,
                    record.object.generation,
                    terminal_revision,
                )?;
                Err(RuntimeError::InvalidConfig(format!(
                    "tier transfer failed: {error:?}"
                )))
            }
        }
    }

    /// Admit one complete typed Pod output before routing it to any durable
    /// semantic, branch, or effect boundary. The admission record is written
    /// first; a later semantic promotion is a separate authoritative event.
    pub fn admit_pod_output<V>(
        &mut self,
        request: PodOutputAdmissionRequest,
        verifier: &V,
    ) -> Result<PodOutputAdmission, RuntimeError>
    where
        V: Verifier<TypedPayload> + ?Sized,
    {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        self.scopes
            .ensure_available(&request.scope_id)
            .map_err(RuntimeError::Scope)?;
        let scope = self
            .scopes
            .get(&request.scope_id)
            .map_err(RuntimeError::Scope)?;
        if scope.session != request.session_id {
            return Err(RuntimeError::PodOutputAdmission(
                "scope and request session differ".into(),
            ));
        }
        let scope_lease = self
            .scopes
            .lease(&request.scope_id)
            .map_err(RuntimeError::Scope)?;
        if scope_lease
            .placement_epoch
            .is_some_and(|epoch| epoch != request.placement_epoch.0)
            || scope_lease
                .fencing_token
                .is_some_and(|token| token != request.fencing_token.0)
        {
            return Err(RuntimeError::PodOutputAdmission(
                "scope lease fencing binding is stale".into(),
            ));
        }
        if request.placement_epoch.0 == 0 || request.fencing_token.0 == 0 {
            return Err(RuntimeError::PodOutputAdmission(
                "missing placement or fencing binding".into(),
            ));
        }
        request
            .manifest
            .validate()
            .map_err(|error| RuntimeError::PodOutputAdmission(format!("{error:?}")))?;
        if request.manifest_digest != request.manifest.manifest_digest {
            return Err(RuntimeError::PodOutputAdmission(
                "manifest digest does not match manifest".into(),
            ));
        }
        if request.output.manifest_digest != request.manifest_digest {
            return Err(RuntimeError::PodOutputAdmission(
                "output manifest binding mismatch".into(),
            ));
        }
        if request.output.generation != request.manifest.generation {
            return Err(RuntimeError::PodOutputAdmission(
                "output generation does not match manifest".into(),
            ));
        }
        if !request
            .manifest
            .artifacts
            .iter()
            .any(|artifact| artifact.digest == request.output.artifact_digest)
        {
            return Err(RuntimeError::PodOutputAdmission(
                "output artifact is not in the execution manifest".into(),
            ));
        }
        if request
            .output
            .provenance
            .iter()
            .any(|item| item.source.0.is_empty())
        {
            return Err(RuntimeError::PodOutputAdmission(
                "output provenance contains an empty source".into(),
            ));
        }
        if request.output.dependencies.iter().any(|digest| {
            *digest == [0; 32]
                || !request
                    .manifest
                    .knowledge
                    .iter()
                    .chain(request.manifest.artifacts.iter())
                    .chain(request.manifest.reader.iter())
                    .any(|binding| binding.digest == *digest)
        }) {
            return Err(RuntimeError::PodOutputAdmission(
                "output dependency is unknown or invalidated".into(),
            ));
        }
        let allow_promotion = matches!(
            request.output.kind,
            PodOutputKind::StateDelta | PodOutputKind::VerifiedResult
        );
        request
            .output
            .validate(&request.expected_type, allow_promotion)
            .map_err(|error| {
                RuntimeError::PodOutputAdmission(format!("invalid output: {error:?}"))
            })?;
        // A hypothesis is recorded unverified: its record carries no verified bit, and replay
        // and the commit check rebuild its output digest with `verified: false`. A hypothesis
        // marked verified would be admitted and then fail its own record (an admission no retry
        // could complete), so it is refused here, before anything is appended.
        if request.output.kind == PodOutputKind::Hypothesis && request.output.verified {
            return Err(RuntimeError::PodOutputAdmission(
                "a hypothesis output cannot be marked verified".into(),
            ));
        }
        if request.output.revision < request.manifest.snapshot_revision {
            return Err(RuntimeError::StaleRevision {
                action: request.output.revision,
                current: request.manifest.snapshot_revision,
            });
        }
        // Approximate-KV taint is decided before the verifier runs: a tainted output of a
        // promoting or acting kind must not even be judged, and nothing is appended.
        let output_digest = request.output.digest();
        let legacy_digest = request.output.legacy_v1_digest();
        let bindings: Vec<&ptr_pods::LineageBinding> = request
            .manifest
            .knowledge
            .iter()
            .chain(request.manifest.artifacts.iter())
            .chain(request.manifest.reader.iter())
            .chain(request.manifest.adapter.iter())
            .collect();
        let kv_taint = self.pod_output_kv_taint(
            &request.request_id,
            request.kv_taint,
            &output_digest,
            &request.output.dependencies,
            &request.output.payload.bytes,
            &bindings,
        );
        if let ptr_model_api::KvReuseTaint::Approximate { report_digest } = kv_taint {
            match request.output.kind {
                PodOutputKind::StateDelta
                | PodOutputKind::VerifiedResult
                | PodOutputKind::ActionProposal => {
                    return Err(RuntimeError::ApproximatePodOutput {
                        kind: request.output.kind,
                        report_digest,
                    });
                }
                PodOutputKind::Observation
                | PodOutputKind::Candidate
                | PodOutputKind::ToolResult
                | PodOutputKind::EnvironmentObservation
                | PodOutputKind::Hypothesis => {}
            }
        }
        let report = verifier.verify(&request.output.payload);
        let passed = report.status == VerificationStatus::Pass
            && !report.findings.iter().any(|finding| finding.hard);
        self.emit(RuntimeEvent::VerifierResult {
            verifier: "pod-output".into(),
            passed,
        });
        if !passed {
            return Err(RuntimeError::PodVerificationFailed {
                pod: request.pod_id.to_string(),
            });
        }
        let action = if request.output.kind == PodOutputKind::ActionProposal {
            let action = decode_action_payload(&request.output.payload.bytes)
                .map_err(RuntimeError::PodOutputAdmission)?;
            if action.generation != request.output.generation
                || action.revision != request.output.revision
            {
                return Err(RuntimeError::PodOutputAdmission(
                    "action generation/revision does not match output".into(),
                ));
            }
            self.authorize_action(&action)?;
            Some(action)
        } else {
            None
        };

        // An admission recorded before version 2 under this output's version-1 digest (issue
        // #141). Version 1 is not injective, so only its recorded result is returned, when its
        // follow-up is in the ledger, and a missing follow-up is refused, never completed from
        // a version-1 match. A hypothesis follow-up records the whole output, so it is compared
        // exactly, and one that records another output is a reused request id. Each refusal is
        // decided before anything is appended and leaves the ledger as it was. A retry that is
        // answered is taint-contained like a version-2 retry (issue #153): its taint is recorded
        // for the recorded output (by its version-1 digest, which names its candidate key or
        // branch id) and for this output, before the recorded result is returned.
        let mut legacy_answer = None;
        if legacy_digest != output_digest {
            for (position, committed) in self.committed_events().iter().enumerate() {
                let LedgerEvent::PodOutputAdmitted {
                    request_id,
                    session_id,
                    scope_id,
                    pod_id,
                    output_digest: existing_digest,
                    output_kind,
                    revision,
                    ..
                } = &committed.event
                else {
                    continue;
                };
                if request_id != &request.request_id
                    || pod_id != &request.pod_id
                    || existing_digest != &legacy_digest
                {
                    continue;
                }
                if scope_id != &request.scope_id || session_id != &request.session_id {
                    return Err(RuntimeError::PodOutputAdmission(
                        "request id was admitted for a different scope or session".into(),
                    ));
                }
                if *output_kind == 2
                    && self.hypothesis_followup_differs(&request, &legacy_digest, position)
                {
                    return Err(RuntimeError::PodOutputAdmission(
                        "request id was reused with a different output".into(),
                    ));
                }
                if !self.admission_followup_committed(
                    *output_kind,
                    &request,
                    &legacy_digest,
                    position,
                ) {
                    return Err(RuntimeError::LegacyAdmissionIncomplete {
                        request_id: request.request_id.clone(),
                        pod_id: request.pod_id.clone(),
                    });
                }
                legacy_answer = Some((*output_kind, *revision));
                break;
            }
        }
        if let Some((output_kind, revision)) = legacy_answer {
            if let ptr_model_api::KvReuseTaint::Approximate { report_digest } = kv_taint {
                self.record_legacy_pod_output_taint(
                    &legacy_digest,
                    &output_digest,
                    &request.output.payload.bytes,
                    report_digest,
                )?;
            }
            return self.admission_result_from_code(output_kind, revision, &request, legacy_digest);
        }
        // Taint first: the registry record precedes the admission, the candidate and the
        // hypothesis record, so a crash in between leaves a stray taint record (which only
        // refuses more), never an untainted candidate. It also precedes the retry lookup below,
        // so a retry of an earlier admission is contained too. A full registry refuses here,
        // before anything of this admission is appended.
        if let ptr_model_api::KvReuseTaint::Approximate { report_digest } = kv_taint {
            self.record_pod_output_taint(
                &output_digest,
                &request.output.payload.bytes,
                report_digest,
            )?;
        }
        // A retry finds the admission record of an earlier attempt. The record
        // says the output was admitted, not that what follows it was applied:
        // promotion is a separate commit that can fail after it. So the earlier
        // answer is replayed only when that follow-up is in the ledger as well;
        // otherwise the admission record stays as it is and the missing step is
        // carried out now instead of reporting a success that has no state behind it.
        let mut already_admitted = false;
        for (position, committed) in self.committed_events().iter().enumerate() {
            if let LedgerEvent::PodOutputAdmitted {
                request_id,
                session_id,
                scope_id,
                pod_id,
                output_digest: existing_digest,
                output_kind,
                revision,
                ..
            } = &committed.event
            {
                if request_id == &request.request_id && pod_id == &request.pod_id {
                    if existing_digest != &output_digest {
                        return Err(RuntimeError::PodOutputAdmission(
                            "request id was reused with a different output".into(),
                        ));
                    }
                    // The follow-up a retry writes belongs to the admission it
                    // completes, so it must come from the scope and session that
                    // admission was made for.
                    if scope_id != &request.scope_id || session_id != &request.session_id {
                        return Err(RuntimeError::PodOutputAdmission(
                            "request id was admitted for a different scope or session".into(),
                        ));
                    }
                    if self.admission_followup_committed(
                        *output_kind,
                        &request,
                        &output_digest,
                        position,
                    ) {
                        return self.admission_result_from_code(
                            *output_kind,
                            *revision,
                            &request,
                            output_digest,
                        );
                    }
                    already_admitted = true;
                }
            }
        }

        let attestation = Attestation {
            required: ptr_types::VerificationLevel::FullSemantic,
            level: report.level,
            verifiers: vec!["pod-output".into()],
            findings: report
                .findings
                .iter()
                .filter(|finding| !finding.hard)
                .map(|finding| format!("pod-output/{}", finding.code))
                .collect(),
        };
        let admitted = LedgerEvent::PodOutputAdmitted {
            request_id: request.request_id.clone(),
            session_id: request.session_id.clone(),
            scope_id: request.scope_id.clone(),
            pod_id: request.pod_id.clone(),
            manifest_digest: request.manifest_digest,
            artifact_digest: request.output.artifact_digest,
            generation: request.output.generation,
            revision: request.output.revision,
            output_kind: request.output.kind_code(),
            output_type: request.output.payload.type_id.clone(),
            output_digest,
            verification: attestation.clone(),
        };
        // A hypothesis is two records. Both are built and checked before the first
        // is committed: refusing the second one afterwards would leave an
        // admission whose retry reports success for a hypothesis that was never
        // created.
        let hypothesis = (request.output.kind == PodOutputKind::Hypothesis).then(|| {
            let branch_id = pod_branch_id(&request.request_id, &request.pod_id, &output_digest);
            let event = LedgerEvent::PodHypothesisCommitted {
                request_id: request.request_id.clone(),
                session_id: request.session_id.clone(),
                scope_id: request.scope_id.clone(),
                branch_id: branch_id.clone(),
                pod_id: request.pod_id.clone(),
                manifest_digest: request.output.manifest_digest,
                artifact_digest: request.output.artifact_digest,
                generation: request.output.generation,
                revision: request.output.revision,
                output_type: request.output.payload.type_id.clone(),
                output_digest,
                payload: request.output.payload.bytes.clone(),
                provenance: request
                    .output
                    .provenance
                    .iter()
                    .map(|item| (item.source.0.clone(), item.note.clone()))
                    .collect(),
                dependencies: request.output.dependencies.clone(),
                confidence_bits: 1.0f32.to_bits(),
                latency_millis: 0,
                verification: attestation,
            };
            (branch_id, event)
        });
        for event in std::iter::once(&admitted).chain(hypothesis.as_ref().map(|(_, event)| event)) {
            ptr_ledger::check_encodable(event).map_err(|error| {
                RuntimeError::PodOutputAdmission(format!("output cannot be recorded: {error}"))
            })?;
        }
        if !already_admitted {
            self.commit(admitted)?;
        }

        match request.output.kind {
            PodOutputKind::Hypothesis => {
                let (branch_id, event) =
                    hypothesis.expect("a hypothesis event was built for a hypothesis output");
                self.commit(event)?;
                Ok(PodOutputAdmission::Hypothesis { branch_id })
            }
            PodOutputKind::ActionProposal => {
                let action = action.expect("action proposal was decoded before admission");
                Ok(PodOutputAdmission::ActionProposal {
                    action: Box::new(action.clone()),

                    admission: ActionAdmission {
                        request_id: request.request_id,
                        session_id: request.session_id,
                        scope_id: request.scope_id.clone(),
                        project: self
                            .scopes
                            .get(&request.scope_id)
                            .map_err(RuntimeError::Scope)?
                            .project
                            .clone(),
                        pod_id: request.pod_id,
                        action_digest: execution::action_digest(&action),
                        output_digest,
                    },
                })
            }
            PodOutputKind::StateDelta => self
                .promote_pod_output(
                    &request.request_id,
                    &request.pod_id,
                    &request.output.payload,
                    report.level,
                )
                .map(|_| PodOutputAdmission::StateDelta {
                    revision: request.output.revision,
                }),
            PodOutputKind::VerifiedResult => self
                .promote_pod_output(
                    &request.request_id,
                    &request.pod_id,
                    &request.output.payload,
                    report.level,
                )
                .map(|_| PodOutputAdmission::VerifiedResult {
                    revision: request.output.revision,
                }),
            PodOutputKind::Observation
            | PodOutputKind::Candidate
            | PodOutputKind::ToolResult
            | PodOutputKind::EnvironmentObservation => self
                .promote_pod_candidate(
                    &request.request_id,
                    &request.pod_id,
                    &request.output.payload,
                    output_digest,
                    report.level,
                )
                .map(|commit| PodOutputAdmission::ObservationCandidate {
                    revision: commit.revision,
                    semantic_key: semantic::pod_candidate_key(
                        &request.request_id,
                        &request.pod_id,
                        &output_digest,
                    ),
                }),
        }
    }

    /// Whether the step that follows the `PodOutputAdmitted` record of `kind`
    /// is in the ledger: the hypothesis record, or the semantic write the
    /// promotion makes. An action proposal has no follow-up state.
    fn admission_followup_committed(
        &self,
        kind: u8,
        request: &PodOutputAdmissionRequest,
        output_digest: &ptr_types::Digest,
        admitted_at: usize,
    ) -> bool {
        if kind == 6 {
            return true;
        }
        // Only a record written after the admission can be its follow-up: an
        // origin of a promotion carries no output digest for the state-changing
        // kinds, so an earlier record with the same request and Pod must not
        // count as this admission's.
        self.committed_events()
            .iter()
            .skip(admitted_at + 1)
            .any(|committed| match (&committed.event, kind) {
                (
                    LedgerEvent::PodHypothesisCommitted {
                        request_id,
                        session_id,
                        scope_id,
                        branch_id,
                        pod_id,
                        output_digest: digest,
                        ..
                    },
                    2,
                ) => {
                    // The output digest does not cover the session, the scope or
                    // the branch id, so each is compared with what this admission
                    // derives; a record for another one is not its follow-up.
                    request_id == &request.request_id
                        && pod_id == &request.pod_id
                        && digest == output_digest
                        && session_id == &request.session_id
                        && scope_id == &request.scope_id
                        && branch_id == &pod_branch_id(&request.request_id, &request.pod_id, digest)
                        // The record holds the whole output, so it is compared exactly (by the
                        // injective version-2 digest of what it records), not only by the
                        // digest it carries, which may be the non-injective version 1.
                        && hypothesis_output(&committed.event)
                            .is_some_and(|recorded| recorded.digest() == request.output.digest())
                }
                (
                    LedgerEvent::SemanticDeltaCommitted {
                        origin:
                            ptr_ledger::SemanticOrigin::PodCandidate {
                                request: origin_request,
                                pod,
                                output_digest: digest,
                                ..
                            },
                        encoded_delta,
                        ..
                    },
                    0 | 1 | 3 | 4,
                ) => {
                    // The digest in the origin is not recomputed from what the
                    // delta writes, so the stored payload is compared as well.
                    origin_request == &request.request_id.0
                        && pod == &request.pod_id.0
                        && digest == output_digest
                        && delta_carries_payload(
                            encoded_delta,
                            &semantic::pod_candidate_key(
                                &request.request_id,
                                &request.pod_id,
                                output_digest,
                            ),
                            &request.output.payload,
                        )
                }
                (
                    LedgerEvent::SemanticDeltaCommitted {
                        origin:
                            ptr_ledger::SemanticOrigin::PodOutput {
                                request: origin_request,
                                pod,
                                ..
                            },
                        encoded_delta,
                        ..
                    },
                    5 | 7,
                ) => {
                    // The origin carries no output digest, so the promoted value
                    // itself is compared with what this admission admitted.
                    origin_request == &request.request_id.0
                        && pod == &request.pod_id.0
                        && delta_carries_payload(
                            encoded_delta,
                            &semantic::pod_output_key(&request.request_id, &request.pod_id),
                            &request.output.payload,
                        )
                }
                _ => false,
            })
    }

    /// Whether a hypothesis record written after the admission at `admitted_at` for this
    /// request, Pod, session, scope and `output_digest` (its branch id included) records an
    /// output other than the request's: the same request id reused for another output that
    /// shares the recorded digest (only possible under the non-injective version-1 digest).
    fn hypothesis_followup_differs(
        &self,
        request: &PodOutputAdmissionRequest,
        output_digest: &ptr_types::Digest,
        admitted_at: usize,
    ) -> bool {
        let wanted = request.output.digest();
        self.committed_events()
            .iter()
            .skip(admitted_at + 1)
            .any(|committed| match &committed.event {
                LedgerEvent::PodHypothesisCommitted {
                    request_id,
                    session_id,
                    scope_id,
                    branch_id,
                    pod_id,
                    output_digest: digest,
                    ..
                } => {
                    request_id == &request.request_id
                        && pod_id == &request.pod_id
                        && digest == output_digest
                        && session_id == &request.session_id
                        && scope_id == &request.scope_id
                        && branch_id == &pod_branch_id(&request.request_id, &request.pod_id, digest)
                        && hypothesis_output(&committed.event)
                            .is_none_or(|recorded| recorded.digest() != wanted)
                }
                _ => false,
            })
    }

    fn admission_result_from_code(
        &self,
        kind: u8,
        revision: Revision,
        request: &PodOutputAdmissionRequest,
        output_digest: ptr_types::Digest,
    ) -> Result<PodOutputAdmission, RuntimeError> {
        match kind {
            0..=1 | 3..=4 => Ok(PodOutputAdmission::ObservationCandidate {
                // The durable PodOutputAdmitted record carries the producer's
                // revision. Candidate promotion has its own SemDB revision;
                // on idempotent replay return the currently materialized one.
                revision: self.revision(),
                semantic_key: semantic::pod_candidate_key(
                    &request.request_id,
                    &request.pod_id,
                    &output_digest,
                ),
            }),
            2 => Ok(PodOutputAdmission::Hypothesis {
                branch_id: format!(
                    "pod-branch:{}:{}:{}",
                    request.request_id,
                    request.pod_id,
                    digest_hex(&output_digest)
                ),
            }),
            5 => Ok(PodOutputAdmission::StateDelta { revision }),
            6 => {
                let action = decode_action_payload(&request.output.payload.bytes)
                    .map_err(RuntimeError::PodOutputAdmission)?;
                if action.generation != request.output.generation
                    || action.revision != request.output.revision
                {
                    return Err(RuntimeError::PodOutputAdmission(
                        "action generation/revision does not match output".into(),
                    ));
                }
                self.authorize_action(&action)?;
                Ok(PodOutputAdmission::ActionProposal {
                    action: Box::new(action.clone()),
                    admission: ActionAdmission {
                        request_id: request.request_id.clone(),
                        session_id: request.session_id.clone(),
                        scope_id: request.scope_id.clone(),
                        project: self
                            .scopes
                            .get(&request.scope_id)
                            .map_err(RuntimeError::Scope)?
                            .project
                            .clone(),
                        pod_id: request.pod_id.clone(),
                        action_digest: execution::action_digest(&action),
                        output_digest,
                    },
                })
            }
            7 => Ok(PodOutputAdmission::VerifiedResult { revision }),
            _ => Err(RuntimeError::PodOutputAdmission(
                "invalid persisted output kind".into(),
            )),
        }
    }

    /// Journal a mesh/tunnel lifecycle transition through the same durable
    /// authority as semantic and scope events. Callers must still perform the
    /// platform tunnel operation separately; this method only commits the
    /// validated control-plane fact.
    pub fn commit_mesh_event(
        &mut self,
        event: ptr_types::MeshTunnelLifecycleEvent,
    ) -> Result<CommitIndex, RuntimeError> {
        self.commit(LedgerEvent::MeshTunnelLifecycle(event))
    }

    /// Verify and journal one complete Pod evidence bundle. The Pod schema and
    /// replay semantics remain in ptr-pods; ptr-runtime owns the admission
    /// boundary and writes the exact canonical bytes to the authoritative ledger.
    pub fn commit_pod_evidence(
        &mut self,
        bundle: &PodEvidenceBundle,
    ) -> Result<CommitIndex, RuntimeError> {
        bundle.verify().map_err(RuntimeError::PodEvidence)?;
        self.append_pod_evidence(bundle)
    }

    /// Signed variant for admissions that require an attested evidence origin.
    /// Signature verification is policy-selected so local integrity-only and
    /// remote Ed25519/TPM/HSM-backed evidence can share this runtime boundary.
    pub fn commit_signed_pod_evidence<V: EvidenceVerifier>(
        &mut self,
        bundle: &PodEvidenceBundle,
        verifier: &V,
    ) -> Result<CommitIndex, RuntimeError> {
        bundle
            .verify_with(verifier)
            .map_err(RuntimeError::PodEvidence)?;
        self.append_pod_evidence(bundle)
    }

    fn append_pod_evidence(
        &mut self,
        bundle: &PodEvidenceBundle,
    ) -> Result<CommitIndex, RuntimeError> {
        let encoded = bundle
            .encode_canonical()
            .map_err(RuntimeError::PodEvidence)?;
        self.commit(LedgerEvent::PodEvidenceCommitted {
            session_id: bundle.session_id.clone(),
            trace_id: bundle.trace_id.clone(),
            manifest_digest: bundle.manifest_digest,
            artifact_digest: bundle.artifact_digest,
            generation: bundle.generation,
            revision: bundle.revision,
            bundle_digest: bundle.bundle_digest,
            bundle: encoded,
        })
    }

    /// Settling or reconciling an attempt is the act that ends a fence, so it
    /// cannot be gated on the fence it ends. Ambiguity about this runtime's own
    /// last append still blocks it: appending onto a history whose last outcome
    /// is unknown would build on a position the runtime cannot describe.
    pub(crate) fn commit_settlement(
        &mut self,
        event: LedgerEvent,
    ) -> Result<CommitIndex, RuntimeError> {
        if self.execution.is_commit_uncertain() {
            return Err(RuntimeError::ExecutionFenced);
        }
        self.validate_lifecycle_event(&event)?;
        // The execution path drops a response that would not fit (`record_settlement`), so this
        // holds for every settlement it writes; it is checked here so that no settlement, by
        // any writer, can take the section past its budget.
        self.check_execution_section_growth(&event)?;
        self.append_checked(event, None)
    }

    /// Append a runtime-written record that is not an effect settlement (the egress refusal
    /// flush): unlike [`Self::commit_settlement`] it respects the execution fence, so an
    /// unsettled effect or an uncertain last append holds it back, and it skips the admission
    /// check of [`Self::commit`] because the runtime itself is the writer.
    pub(crate) fn commit_gated(&mut self, event: LedgerEvent) -> Result<CommitIndex, RuntimeError> {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        self.validate_lifecycle_event(&event)?;
        self.append_prepared(event, None)
    }

    fn append_prepared(
        &mut self,
        event: LedgerEvent,
        semantic: Option<PreparedDelta>,
    ) -> Result<CommitIndex, RuntimeError> {
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        self.append_checked(event, semantic)
    }

    fn append_checked(
        &mut self,
        event: LedgerEvent,
        semantic: Option<PreparedDelta>,
    ) -> Result<CommitIndex, RuntimeError> {
        // A record the log format cannot hold is refused before the commit is marked uncertain:
        // the durable ledger would refuse it without writing a byte, and the uncertainty mark
        // would then fence this runtime for good; the in-memory ledger would keep a record
        // that no export or durable log can hold.
        ptr_ledger::check_encodable(&event)
            .map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        self.execution.begin_commit();
        let index = self.ledger.append(event)?;
        let committed = self
            .ledger
            .events()
            .last()
            .expect("append created committed event")
            .clone();
        self.apply_committed(&committed, semantic)?;
        self.execution.complete_commit();
        Ok(index)
    }

    fn validate_activation(&self, target: &str, requested: Generation) -> Result<(), RuntimeError> {
        let current = self.live_generation(target);
        if current.is_some_and(|generation| requested < generation)
            || self
                .revoked_generations
                .contains(&(target.to_owned(), requested))
        {
            return Err(RuntimeError::InvalidLifecycleTransition {
                target: target.to_owned(),
                current,
                requested,
            });
        }
        Ok(())
    }

    fn validate_lifecycle_event(&self, event: &LedgerEvent) -> Result<(), RuntimeError> {
        match event {
            LedgerEvent::CapsuleCommitted {
                project,
                capsule,
                generation,
            } => {
                let target = capsule.to_string();
                // Until target keys are a cross-component tagged union, prevent
                // capsule IDs from impersonating the reserved namespaces. `expert:` belongs to
                // the expert registry, which alone answers for an expert's lifecycle.
                if target.starts_with("constraint:")
                    || target.starts_with("procedure:")
                    || target.starts_with(ptr_state::EXPERT_PREFIX)
                {
                    return Err(RuntimeError::ReservedTargetNamespace { target });
                }
                // A pod lifecycle target names its project; only that project may commit it, or
                // one project could make another project's pod head or representation live.
                if let Some(embedded) = pod_head::pod_capsule_project(&target) {
                    if embedded != project.to_string() {
                        return Err(RuntimeError::PodCapsuleProjectMismatch {
                            capsule: target.clone(),
                            embedded: embedded.to_owned(),
                            project: project.to_string(),
                        });
                    }
                    if generation.0 == 0 {
                        return Err(RuntimeError::PodCapsuleGenerationZero { capsule: target });
                    }
                }
                if let Some(current) = self.capsule_projects.get(&target) {
                    if current != &project.to_string() {
                        return Err(RuntimeError::CapsuleProjectMismatch {
                            capsule: target,
                            current: current.clone(),
                            requested: project.to_string(),
                        });
                    }
                }
                self.validate_activation(&target, *generation)
            }
            LedgerEvent::CapsuleSuperseded { capsule, old, new } => {
                let target = capsule.to_string();
                let current = self.live_generation(&target);
                if !self.capsule_projects.contains_key(&target)
                    || current != Some(*old)
                    || new <= old
                {
                    return Err(RuntimeError::InvalidLifecycleTransition {
                        target,
                        current,
                        requested: *new,
                    });
                }
                self.validate_activation(&target, *new)
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => {
                self.validate_activation(&format!("constraint:{key}"), *generation)
            }
            LedgerEvent::ProcedurePromoted { id, generation } => {
                self.validate_activation(&format!("procedure:{id}"), *generation)
            }
            LedgerEvent::SnapshotCommitted { covers, .. } if covers.0 > self.state.last_applied => {
                Err(RuntimeError::SnapshotBeyondHistory {
                    covers: *covers,
                    last_applied: CommitIndex(self.state.last_applied),
                })
            }
            LedgerEvent::EffectAttempted { key, .. } => {
                let Some(key) = key else { return Ok(()) };
                // Every build has held a key to this, so a log never holds one
                // that is not an identifier. What a snapshot can carry is a
                // bound on new records only (`validate_new_record`).
                if !execution::valid_identifier(key) {
                    return Err(RuntimeError::InvalidEffectKey { key: key.clone() });
                }
                if self.execution.key_in_flight(key) {
                    return Err(RuntimeError::EffectKeyInFlight { key: key.clone() });
                }
                Ok(())
            }
            LedgerEvent::EffectSettled {
                attempt,
                response,
                response_digest,
            } => {
                if !self.execution.is_unsettled(*attempt) {
                    return Err(RuntimeError::UnknownEffectAttempt { attempt: *attempt });
                }
                // Checked before anything is allocated from it, and checked
                // against its own digest: a retained response that disagrees with
                // the digest beside it is two answers, which is worse than none.
                if let Some(response) = response {
                    if response.len() > MAX_RETAINED_RESPONSE
                        || integrity::sha256(response) != *response_digest
                    {
                        return Err(RuntimeError::InconsistentEffectResponse { attempt: *attempt });
                    }
                }
                Ok(())
            }
            LedgerEvent::EffectReconciled { attempt, .. } => {
                if !self.execution.is_unsettled(*attempt) {
                    return Err(RuntimeError::UnknownEffectAttempt { attempt: *attempt });
                }
                Ok(())
            }
            LedgerEvent::ScopeLifecycle(event) => {
                if event.revision != self.revision() {
                    return Err(RuntimeError::ScopeRevisionMismatch {
                        expected: self.revision(),
                        actual: event.revision,
                    });
                }
                self.scopes
                    .validate_committed(event)
                    .map_err(RuntimeError::Scope)
            }
            LedgerEvent::ProtectedStateCommitted {
                domain,
                logical_id,
                revision,
                plaintext_digest,
                ciphertext_digest,
                anchor_digest,
                ..
            } => {
                if *domain > 4
                    || logical_id.is_empty()
                    || revision.0 == 0
                    || *plaintext_digest == [0; 32]
                    || *ciphertext_digest == [0; 32]
                    || *anchor_digest == [0; 32]
                {
                    return Err(RuntimeError::InvalidConfig(
                        "invalid protected-state ledger reference".into(),
                    ));
                }
                Ok(())
            }
            LedgerEvent::MeshTunnelLifecycle(event) => event.validate().map_err(|reason| {
                RuntimeError::InvalidConfig(format!("invalid mesh event: {reason}"))
            }),
            LedgerEvent::ExecutionManifestAdmitted {
                manifest_digest,
                generation,
                revision,
                policy_revision,
                manifest,
            } => {
                let decoded =
                    ptr_pods::ExecutionManifest::decode_canonical(manifest).map_err(|_| {
                        RuntimeError::InvalidConfig("invalid execution manifest".into())
                    })?;
                if decoded.manifest_digest != *manifest_digest
                    || decoded.generation != *generation
                    || decoded.snapshot_revision != *revision
                    || decoded.policy_revision != *policy_revision
                {
                    return Err(RuntimeError::InvalidConfig(
                        "execution manifest event binding mismatch".into(),
                    ));
                }
                Ok(())
            }
            LedgerEvent::ExecutionManifestRevoked {
                manifest_digest,
                generation,
                revision,
            } => {
                if *manifest_digest == [0; 32] || generation.0 == 0 {
                    return Err(RuntimeError::InvalidConfig(
                        "invalid execution manifest revocation".into(),
                    ));
                }
                if *revision != self.revision() {
                    return Err(RuntimeError::ScopeRevisionMismatch {
                        expected: self.revision(),
                        actual: *revision,
                    });
                }
                Ok(())
            }
            LedgerEvent::PodEvidenceCommitted {
                session_id,
                trace_id,
                manifest_digest,
                artifact_digest,
                generation,
                revision,
                bundle_digest,
                bundle,
            } => {
                let decoded = ptr_pods::PodEvidenceBundle::decode_canonical(bundle)
                    .map_err(RuntimeError::PodEvidence)?;
                if decoded.session_id != *session_id
                    || decoded.trace_id != *trace_id
                    || decoded.manifest_digest != *manifest_digest
                    || decoded.artifact_digest != *artifact_digest
                    || decoded.generation != *generation
                    || decoded.revision != *revision
                    || decoded.bundle_digest != *bundle_digest
                {
                    return Err(RuntimeError::PodEvidence(
                        ptr_pods::EvidenceError::DigestMismatch,
                    ));
                }
                Ok(())
            }
            LedgerEvent::PodOutputAdmitted {
                request_id,
                session_id,
                scope_id,
                pod_id,
                manifest_digest,
                artifact_digest,
                generation,
                revision: _,
                output_kind,
                output_type,
                output_digest,
                verification,
            } => {
                if request_id.0.is_empty()
                    || session_id.0.is_empty()
                    || scope_id.0.is_empty()
                    || pod_id.0.is_empty()
                    || output_type.0.is_empty()
                    || generation.0 == 0
                    || *output_kind > 7
                    || *manifest_digest == [0; 32]
                    || *artifact_digest == [0; 32]
                    || *output_digest == [0; 32]
                    || verification.verifiers.is_empty()
                {
                    return Err(RuntimeError::PodOutputAdmission(
                        "invalid Pod output admission event".into(),
                    ));
                }
                Ok(())
            }
            LedgerEvent::PodHypothesisCommitted {
                request_id,
                session_id,
                scope_id,
                branch_id,
                pod_id,
                manifest_digest,
                artifact_digest,
                generation,
                output_type,
                output_digest,
                provenance,
                dependencies,
                confidence_bits,
                verification,
                ..
            } => {
                if request_id.0.is_empty()
                    || session_id.0.is_empty()
                    || scope_id.0.is_empty()
                    || branch_id.is_empty()
                    || pod_id.0.is_empty()
                    || output_type.0.is_empty()
                    || generation.0 == 0
                    || *manifest_digest == [0; 32]
                    || *artifact_digest == [0; 32]
                    || *output_digest == [0; 32]
                    || provenance.iter().any(|(source, _)| source.is_empty())
                    || dependencies.contains(&[0; 32])
                    || !f32::from_bits(*confidence_bits).is_finite()
                    || !(0.0..=1.0).contains(&f32::from_bits(*confidence_bits))
                    || verification.verifiers.is_empty()
                {
                    return Err(RuntimeError::PodOutputAdmission(
                        "invalid Pod hypothesis event".into(),
                    ));
                }
                let output = hypothesis_output(event).expect("a hypothesis event");
                // Admission records the version-2 digest. A record committed before version 2
                // (issue #141) carries the version-1 digest, which replay still accepts: the
                // two versions are domain-separated, so a stored digest equals at most one. A
                // new record must carry version 2 (`validate_new_record`).
                if output.digest() != *output_digest && output.legacy_v1_digest() != *output_digest
                {
                    return Err(RuntimeError::PodOutputAdmission(
                        "hypothesis output digest mismatch".into(),
                    ));
                }
                Ok(())
            }
            LedgerEvent::PolicyBundleActivated {
                revision,
                key_id,
                bundle_digest,
                bundle,
            } => {
                if revision.0 == 0 || key_id.is_empty() || *bundle_digest == [0; 32] {
                    return Err(RuntimeError::InvalidConfig(
                        "invalid policy activation".into(),
                    ));
                }
                let decoded =
                    policy::SignedPolicyBundle::decode_for_ledger(bundle).map_err(|error| {
                        RuntimeError::InvalidConfig(format!("invalid policy bundle: {error:?}"))
                    })?;
                if decoded.revision != *revision
                    || decoded.key_id != *key_id
                    || decoded.payload_digest != *bundle_digest
                {
                    return Err(RuntimeError::InvalidConfig(
                        "policy activation binding mismatch".into(),
                    ));
                }
                let mut projection = self
                    .manifest_authority
                    .read()
                    .expect("manifest authority lock poisoned")
                    .clone();
                projection
                    .activate_policy(*revision, *bundle_digest)
                    .map_err(|error| {
                        RuntimeError::InvalidConfig(format!("policy revision conflict: {error:?}"))
                    })
            }
            LedgerEvent::PolicyBundleRevoked {
                revision,
                bundle_digest,
                reason,
            } => {
                if revision.0 == 0 || *bundle_digest == [0; 32] || reason.is_empty() {
                    return Err(RuntimeError::InvalidConfig(
                        "invalid policy revocation".into(),
                    ));
                }
                let projection = self
                    .manifest_authority
                    .read()
                    .expect("manifest authority lock poisoned")
                    .clone();
                if projection.policy_digest(*revision) != Some(*bundle_digest) {
                    return Err(RuntimeError::InvalidConfig(
                        "policy revocation binding mismatch".into(),
                    ));
                }
                Ok(())
            }
            LedgerEvent::SessionRevoked { session_id, reason } => {
                if session_id.0.is_empty() || reason.is_empty() {
                    return Err(RuntimeError::InvalidConfig(
                        "invalid session revocation".into(),
                    ));
                }
                if self.revoked_sessions.contains(session_id) {
                    return Ok(());
                }
                Ok(())
            }
            LedgerEvent::TierBackendLifecycle { .. }
            | LedgerEvent::TierObjectCommitted { .. }
            | LedgerEvent::TierReplicaLifecycle { .. } => {
                let mut projection = self.tier_journal.clone();
                projection.apply(event).map_err(|error| {
                    RuntimeError::InvalidConfig(format!("invalid tier journal event: {error:?}"))
                })
            }
            // Replay re-derives the ids from the record and repeats the compare-and-set against
            // the registry as of this point of the history (see the `expert` module).
            LedgerEvent::ExpertStructuralCommitted { .. } => self.validate_expert_event(event),
            // Replay repeats the fence's compare-and-set against the registry as of this point.
            LedgerEvent::ExpertFenceCommitted { .. } => self.validate_expert_fence_event(event),
            // Revocation tombstones are monotone and may precede activation.
            // Verifier/snapshot records do not confer permissions or load state.
            LedgerEvent::SemanticDeltaCommitted { .. }
            | LedgerEvent::Revoked { .. }
            | LedgerEvent::ProcedureRevoked { .. }
            | LedgerEvent::VerifierAttested { .. }
            | LedgerEvent::SnapshotCommitted { .. } => Ok(()),
            // A Record must stay within the live bound, a Clear must name live subjects of the
            // same recipe; otherwise the log is refused (fail closed). A forged Record applies
            // (taint only refuses more). Only a directly committed record is refused earlier
            // ([`validate_new_record`]).
            LedgerEvent::KvTaintCommitted { .. } => self.validate_kv_taint_event(event),
            // Replay re-derives the fence set at this point of the history and compares its count
            // and digest with the event's (see the `expert` module).
            LedgerEvent::InputInvalidated { .. } => self.validate_input_invalidated_event(event),
            // Replay re-derives the fence set of an authority change at this point of the history
            // and compares its count and digest with the event's, and holds the epoch to the
            // current standing (see the `expert::authority` module).
            LedgerEvent::ExpertAuthorityChanged { .. } => {
                self.validate_authority_changed_event(event)
            }
            // A refusal confers nothing and loads nothing; its shape is the codec's to hold.
            LedgerEvent::EgressRefused { .. } => ptr_ledger::check_encodable(event)
                .map_err(|error| RuntimeError::Ledger(error.to_string())),
            // A compare-and-swap on the head of one pod identity, checked again on replay.
            LedgerEvent::PodHeadAdvanced {
                project,
                identity,
                old,
                new,
            } => self.validate_pod_head_advance(project, identity, old.as_ref(), new),
        }
    }

    fn apply_committed(
        &mut self,
        committed: &CommittedEvent,
        semantic: Option<PreparedDelta>,
    ) -> Result<(), RuntimeError> {
        self.invalidate_kv_catalog(&committed.event, semantic.as_ref());
        match &committed.event {
            LedgerEvent::SemanticDeltaCommitted { .. } => {
                let delta =
                    semantic.ok_or(RuntimeError::Semantic(SemanticError::InvalidEncoding))?;
                let affected = delta.affected().clone();
                self.semdb
                    .apply_prepared(delta)
                    .map_err(RuntimeError::Semantic)?;
                let snapshot = self.semdb.snapshot();
                self.semantic_edits.record(
                    &affected,
                    |key| snapshot.value(key).is_some(),
                    self.semdb.revision(),
                );
            }
            LedgerEvent::CapsuleCommitted {
                project,
                capsule,
                generation,
            } => {
                self.capsule_projects
                    .insert(capsule.to_string(), project.to_string());
                self.set_live_generation(capsule.to_string(), *generation);
            }
            LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                self.set_live_generation(capsule.to_string(), *new);
            }
            LedgerEvent::Revoked {
                subject,
                generation,
            } => {
                self.revoked_generations
                    .insert((subject.clone(), *generation));
                // A revocation of a semantic key is a floor over every citation of it.
                let held = self.semdb.snapshot().value(subject).is_some();
                self.semantic_edits
                    .record_revocation(subject, held, self.semdb.revision());
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => {
                self.set_live_generation(format!("constraint:{key}"), *generation);
            }
            LedgerEvent::ProcedurePromoted { id, generation } => {
                self.set_live_generation(format!("procedure:{id}"), *generation);
            }
            LedgerEvent::ProcedureRevoked { id, generation } => {
                self.revoked_generations
                    .insert((format!("procedure:{id}"), *generation));
            }
            LedgerEvent::VerifierAttested { .. } | LedgerEvent::SnapshotCommitted { .. } => {}
            LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                effect,
                generation,
                revision,
                action_digest,
                ..
            } => {
                self.execution.record_attempt(execution::UnsettledEffect {
                    attempt: committed.index,
                    key: key.clone(),
                    target: target.clone(),
                    operation: operation.clone(),
                    effect: *effect,
                    identity: execution::ActionIdentity {
                        project: project.clone(),
                        principal: principal.clone(),
                        revision: *revision,
                        generation: *generation,
                        action_digest: *action_digest,
                    },
                });
            }
            LedgerEvent::EffectSettled {
                attempt, response, ..
            } => {
                self.execution
                    .settle(*attempt, execution::Settlement::Applied(response.clone()));
            }
            LedgerEvent::EffectReconciled {
                attempt, applied, ..
            } => {
                // Reconciliation establishes whether the effect applied, never a
                // response: the runtime that could have reproduced one would not
                // have needed reconciling.
                let settlement = if *applied {
                    execution::Settlement::Applied(None)
                } else {
                    execution::Settlement::NotApplied
                };
                self.execution.settle(*attempt, settlement);
            }
            LedgerEvent::ScopeLifecycle(event) => {
                self.scopes
                    .apply_committed(event.clone())
                    .map_err(RuntimeError::Scope)?;
                self.emit(RuntimeEvent::ScopeLifecycle {
                    scope: event.scope_id.clone(),
                    kind: event.kind,
                });
            }
            LedgerEvent::MeshTunnelLifecycle(_)
            | LedgerEvent::PodEvidenceCommitted { .. }
            | LedgerEvent::PodOutputAdmitted { .. } => {}
            LedgerEvent::KvTaintCommitted { .. } => self.apply_kv_taint_event(&committed.event),
            LedgerEvent::ExpertStructuralCommitted { .. } => {
                self.apply_expert_event(&committed.event)?;
            }
            LedgerEvent::ExpertFenceCommitted { .. } => {
                self.apply_expert_fence_event(&committed.event)?;
            }
            LedgerEvent::InputInvalidated { .. } => {
                self.apply_input_invalidated_event(&committed.event)?;
            }
            LedgerEvent::ExpertAuthorityChanged { .. } => {
                self.apply_authority_changed_event(&committed.event)?;
            }
            // Audit only: the refusal changes no runtime state. It stays in the ledger and in
            // the `ptr-state` projection.
            LedgerEvent::EgressRefused { .. } => {}
            LedgerEvent::TierBackendLifecycle { .. }
            | LedgerEvent::TierObjectCommitted { .. }
            | LedgerEvent::TierReplicaLifecycle { .. } => {
                self.tier_journal.apply(&committed.event).map_err(|error| {
                    RuntimeError::InvalidConfig(format!("invalid committed tier event: {error:?}"))
                })?;
            }
            LedgerEvent::PodHypothesisCommitted {
                branch_id,
                pod_id,
                generation,
                revision,
                output_type,
                payload,
                provenance,
                confidence_bits,
                latency_millis,
                verification,
                ..
            } => {
                let hypothesis = PodHypothesis {
                    branch_id: branch_id.clone(),
                    pod_id: pod_id.clone(),
                    generation: *generation,
                    evidence: provenance
                        .iter()
                        .map(|(source, note)| ProvenanceRef {
                            source: EvidenceId(source.clone()),
                            note: note.clone(),
                        })
                        .collect(),
                    output: TypedPayload {
                        type_id: output_type.clone(),
                        bytes: payload.clone(),
                    },
                    confidence: Probability::new(f32::from_bits(*confidence_bits)).ok_or_else(
                        || RuntimeError::PodOutputAdmission("invalid hypothesis confidence".into()),
                    )?,
                    latency: Duration::from_millis(*latency_millis),
                    verification: ptr_verifier::VerificationReport {
                        status: VerificationStatus::Pass,
                        level: verification.level,
                        score: Probability::new(1.0).expect("one is a probability"),
                        findings: verification
                            .findings
                            .iter()
                            .map(|code| ptr_verifier::Finding {
                                code: code.clone(),
                                message: String::new(),
                                hard: false,
                            })
                            .collect(),
                    },
                };
                if self
                    .hypotheses
                    .insert(branch_id.clone(), hypothesis)
                    .is_some()
                {
                    return Err(RuntimeError::PodOutputAdmission(
                        "duplicate hypothesis branch".into(),
                    ));
                }
                let _ = revision;
            }
            LedgerEvent::ExecutionManifestAdmitted { manifest, .. } => {
                let decoded =
                    ptr_pods::ExecutionManifest::decode_canonical(manifest).map_err(|_| {
                        RuntimeError::InvalidConfig("invalid execution manifest".into())
                    })?;
                self.execution_manifests
                    .write()
                    .expect("manifest registry lock poisoned")
                    .admit_replayed(decoded, self.revision())
                    .map_err(|_| {
                        RuntimeError::InvalidConfig("manifest registry conflict".into())
                    })?;
                let decoded =
                    ptr_pods::ExecutionManifest::decode_canonical(manifest).map_err(|_| {
                        RuntimeError::InvalidConfig("invalid execution manifest".into())
                    })?;
                self.manifest_authority
                    .write()
                    .expect("manifest authority lock poisoned")
                    .observe_manifest(&decoded)
                    .map_err(|_| {
                        RuntimeError::InvalidConfig("manifest authority conflict".into())
                    })?;
            }
            LedgerEvent::ExecutionManifestRevoked {
                manifest_digest, ..
            } => {
                self.execution_manifests
                    .write()
                    .expect("manifest registry lock poisoned")
                    .revoke(*manifest_digest)
                    .map_err(|_| {
                        RuntimeError::InvalidConfig("unknown manifest revocation".into())
                    })?;
            }
            LedgerEvent::PolicyBundleActivated {
                revision,
                bundle_digest,
                ..
            } => {
                self.manifest_authority
                    .write()
                    .expect("manifest authority lock poisoned")
                    .activate_policy(*revision, *bundle_digest)
                    .map_err(|error| {
                        RuntimeError::InvalidConfig(format!(
                            "policy activation conflict: {error:?}"
                        ))
                    })?;
                self.policy_authority_ready = true;
            }
            LedgerEvent::PolicyBundleRevoked { revision, .. } => {
                self.manifest_authority
                    .write()
                    .expect("manifest authority lock poisoned")
                    .revoke_policy(*revision)
                    .map_err(|error| {
                        RuntimeError::InvalidConfig(format!(
                            "policy revocation conflict: {error:?}"
                        ))
                    })?;
                self.policy_authority_ready = self
                    .manifest_authority
                    .read()
                    .expect("manifest authority lock poisoned")
                    .current_policy_revision()
                    .0
                    != 0;
            }
            LedgerEvent::SessionRevoked { session_id, .. } => {
                self.revoked_sessions.insert(session_id.clone());
            }
            LedgerEvent::ProtectedStateCommitted {
                domain: 3,
                revision,
                plaintext_digest,
                ..
            } => {
                self.manifest_authority
                    .write()
                    .expect("manifest authority lock poisoned")
                    .register_protected_snapshot(*revision, *plaintext_digest)
                    .map_err(|error| {
                        RuntimeError::InvalidConfig(format!(
                            "protected snapshot authority conflict: {error:?}"
                        ))
                    })?;
            }
            LedgerEvent::ProtectedStateCommitted { .. } => {}
            LedgerEvent::PodHeadAdvanced {
                project,
                identity,
                new,
                ..
            } => {
                self.pod_heads
                    .set(project.clone(), identity.clone(), new.clone());
            }
        }

        self.state.apply(committed);
        self.emit(RuntimeEvent::CommitApplied(committed.index));
        Ok(())
    }

    /// Test hook: leave the execution state as a commit whose outcome is unknown does, so the
    /// next append is refused.
    #[cfg(test)]
    pub(crate) fn mark_commit_uncertain_for_test(&mut self) {
        self.execution.begin_commit();
    }

    fn emit(&mut self, event: RuntimeEvent) {
        // Invariant 8: the in-process event log is bounded. When it is full the older half is
        // dropped (amortized O(1) per event); sequences keep counting, so a reader sees the
        // gap, and the ledger, not this log, is the record of what happened.
        if self.events.len() >= MAX_RETAINED_EVENTS {
            let dropped = self.events.len() - MAX_RETAINED_EVENTS / 2;
            self.events.drain(..dropped);
            self.dropped_events = self.dropped_events.saturating_add(dropped as u64);
        }
        self.events.push(EventEnvelope {
            sequence: self.next_event_sequence,
            event,
        });
        self.next_event_sequence = self.next_event_sequence.saturating_add(1);
    }
}

/// What a record must meet to be committed now, beyond the validation every
/// record is replayed with: a keyed attempt names a key a compacted snapshot can
/// carry.
///
/// A snapshot carries every settled key, whatever its outcome, as a string of 1
/// to [`execution::MAX_KEY_BYTES`] bytes, so an attempt under a longer key would
/// leave every export failing once it settled. It is refused before anything is
/// appended. The project and principal an applied key carries are written at
/// any length, so they are recorded as given.
///
/// Replay does not apply this. Earlier builds held a key only to being an
/// identifier, of any length, and a log holding a longer one opened and failed
/// only at export. Refusing it at replay would turn that into a runtime that
/// cannot open after an upgrade.
///
/// A semantic record is refused outright
/// ([`RuntimeError::SemanticRecordOutsideSemanticPath`]): only the semantic
/// write paths build one, with the origin they checked.
///
/// An expert structural record is refused the same way
/// ([`RuntimeError::StructuralRecordOutsideAdmissionPath`]): only the expert admission path may
/// build one. So is an expert fence record: only `fence_expert` and `readmit_expert` may, and an
/// input invalidation record ([`LedgerEvent::InputInvalidated`]): only
/// `invalidate_input`, which derives its fence set, may, and so is an authority change record
/// ([`LedgerEvent::ExpertAuthorityChanged`]): only `change_expert_authority` may. A taint
/// record ([`LedgerEvent::KvTaintCommitted`]) is refused the same way: only the runtime's taint
/// path may write one.
/// The Pod output a `PodHypothesisCommitted` record describes, for checking its output digest.
fn hypothesis_output(event: &LedgerEvent) -> Option<ptr_pods::PodOutput> {
    let LedgerEvent::PodHypothesisCommitted {
        manifest_digest,
        artifact_digest,
        generation,
        revision,
        output_type,
        payload,
        provenance,
        dependencies,
        ..
    } = event
    else {
        return None;
    };
    Some(ptr_pods::PodOutput {
        kind: PodOutputKind::Hypothesis,
        payload: TypedPayload {
            type_id: output_type.clone(),
            bytes: payload.clone(),
        },
        generation: *generation,
        manifest_digest: *manifest_digest,
        artifact_digest: *artifact_digest,
        provenance: provenance
            .iter()
            .map(|(source, note)| ProvenanceRef {
                source: EvidenceId(source.clone()),
                note: note.clone(),
            })
            .collect(),
        dependencies: dependencies.clone(),
        revision: *revision,
        verified: false,
    })
}

fn validate_new_record(event: &LedgerEvent) -> Result<(), RuntimeError> {
    // A new hypothesis record carries the version-2 output digest only; the version-1 digest
    // (ambiguous, issue #141) is accepted for records already in a log, on replay and restore.
    if let (LedgerEvent::PodHypothesisCommitted { output_digest, .. }, Some(output)) =
        (event, hypothesis_output(event))
    {
        if output.digest() != *output_digest {
            return Err(RuntimeError::PodOutputAdmission(
                "hypothesis output digest mismatch".into(),
            ));
        }
    }
    if matches!(
        event,
        LedgerEvent::ExpertStructuralCommitted { .. }
            | LedgerEvent::ExpertFenceCommitted { .. }
            | LedgerEvent::InputInvalidated { .. }
            | LedgerEvent::ExpertAuthorityChanged { .. }
            | LedgerEvent::KvTaintCommitted { .. }
            | LedgerEvent::EgressRefused { .. }
    ) {
        return Err(RuntimeError::StructuralRecordOutsideAdmissionPath);
    }
    if matches!(event, LedgerEvent::SemanticDeltaCommitted { .. }) {
        return Err(RuntimeError::SemanticRecordOutsideSemanticPath);
    }
    if let LedgerEvent::EffectAttempted { key: Some(key), .. } = event {
        if !execution::valid_key(key) {
            return Err(RuntimeError::InvalidEffectKey { key: key.clone() });
        }
    }
    if let LedgerEvent::EffectAttempted {
        target, operation, ..
    } = event
    {
        for (field, value) in [
            (execution::EffectAttemptField::Target, target),
            (execution::EffectAttemptField::Operation, operation),
        ] {
            if !execution::valid_attempt_field(value) {
                return Err(RuntimeError::InvalidEffectAttemptField {
                    field,
                    bytes: value.len(),
                });
            }
        }
    }
    Ok(())
}
