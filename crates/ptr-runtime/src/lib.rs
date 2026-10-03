pub mod admission;
pub mod compacted;
pub mod directory;
pub mod execution;
pub mod identity;
pub mod kv;
pub mod manifest;
pub mod memory;
pub mod merge;
pub mod migration;
pub mod neural;
pub mod persistence;
pub mod placement;
pub mod policy;
pub mod protected_state;
pub mod sandbox;
pub mod scopes;
pub mod semantic;
pub mod tensor_kv;
pub mod tier;

pub use admission::{AdmissionError, AdmissionGrant, IdentityAdmissionController};
pub use directory::{InMemoryPodDirectory, PodDirectory};
#[cfg(feature = "oidc-http")]
pub use identity::ReqwestJwksRefresher;
pub use identity::{
    AutheliaForwardAuthAdapter, ForwardAuthHeaders, HttpJwksRefresher, JwksDocument, JwksFetchFn,
    JwksRefresher, JwksStore, OidcAlgorithm, OidcError, OidcIdentityAdapter,
};
pub use kv::{KvUseError, KvUseRequest, KvUseTicket, RuntimeKvAuthority};
pub use manifest::{
    validate_and_build, validate_authority, ExecutionManifestRegistry, ManifestAuthorityRegistry,
    ManifestBindingAuthority, RuntimeExecutionManifestResolver, RuntimeManifestError,
    ValidatedExecutionManifest,
};
pub use merge::{
    ChangeOrigin, HoldReason, MergeAuthority, MergeHold, MergeOutcome, MergePreview, MergeReceipt,
    SemanticChange, SemanticGrant, SemanticGrantInfo, SemanticRefusal, SemanticVerdict,
};
pub use migration::{KvMigrationController, MigrationError, MigrationRecord, MigrationState};
pub use placement::{
    DeviceHealth, DeviceRecord, FencedStateLease, FencingToken, NodeHealth, NodeRecord, Placement,
    PlacementEpoch, PlacementError, PodPlacementController,
};
#[cfg(feature = "oidc-http")]
pub use policy::ReqwestPolicyBundleLoader;
pub use policy::{
    CapabilityPolicy, HttpPolicyBundleLoader, PodPolicyBinding, PolicyBundleLoader,
    PolicyBundlePayload, PolicyError, PolicyFetchFn, PolicyTrustStore, ProjectPolicy,
    SignedPolicyBundle,
};
pub use protected_state::ProtectedStateCoordinator;
pub use sandbox::{
    ExternalSandboxExecutor, NativeSandboxExecutor, NetworkMode, SandboxBackend, SandboxError,
    SandboxExecutor, SandboxLease, SandboxProfile, SandboxRequest, SandboxResponse,
};
pub use scopes::{
    CleanupError, ExecutionScope, ScopeCleanupCoordinator, ScopeError, ScopeEvent, ScopeEventKind,
    ScopeRegistry, ScopeState,
};
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

fn digest_hex(digest: &ptr_types::Digest) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

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
    InferenceBackend, ModelEvent, ModelObservation, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{EvidenceVerifier, PodEvidenceBundle, PodHypothesis, PodOutputKind, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::PodRouter;
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
    PodUnavailable {
        capability: String,
        input_type: String,
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
    /// A semantic record written before origins existed, replayed after a
    /// record with an attributed origin (rule R1).
    LegacySemanticRecord {
        index: CommitIndex,
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
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResumableRun {
    pub model_events: Vec<ModelEvent>,
    pub observations: Vec<TypedPayload>,
    pub action: Option<ActionIr>,
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
    next_event_sequence: u64,
    /// The host's policy for semantic writes that are not ingress; none until
    /// the host installs one. Not history: a runtime rebuilt from its log
    /// has none.
    semantic_grant: Option<merge::SemanticGrant>,
    scopes: ScopeRegistry,
    execution_manifests: Arc<RwLock<ExecutionManifestRegistry>>,
    manifest_authority: Arc<RwLock<ManifestAuthorityRegistry>>,
    policy_trust: policy::PolicyTrustStore,
    policy_authority_ready: bool,
    revoked_sessions: BTreeSet<ptr_types::SessionId>,
    hypotheses: BTreeMap<String, PodHypothesis>,
    tier_journal: TierJournalProjection,
}

impl PtrRuntime {
    pub fn new(config: PtrConfig) -> Result<Self, RuntimeError> {
        Self::with_ledger(config, RuntimeLedger::Memory(InMemoryLedger::default()))
    }

    pub fn open_durable(config: PtrConfig, path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let ledger =
            FileLedger::open(path).map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        Self::from_durable_ledger(config, ledger)
    }

    fn from_durable_ledger(config: PtrConfig, ledger: FileLedger) -> Result<Self, RuntimeError> {
        let persisted = ledger.events().to_vec();
        let mut runtime = Self::with_ledger(config, RuntimeLedger::File(ledger))?;
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
            next_event_sequence: 1,
            semantic_grant: None,
            scopes: ScopeRegistry::default(),
            execution_manifests: Arc::new(RwLock::new(ExecutionManifestRegistry::default())),
            manifest_authority: Arc::new(RwLock::new(ManifestAuthorityRegistry::default())),
            policy_trust: policy::PolicyTrustStore::default(),
            policy_authority_ready: false,
            revoked_sessions: BTreeSet::new(),
            hypotheses: BTreeMap::new(),
            tier_journal: TierJournalProjection::default(),
        })
    }

    pub fn replay(config: PtrConfig, events: &[CommittedEvent]) -> Result<Self, RuntimeError> {
        let mut runtime = Self::new(config)?;
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

    pub fn events(&self) -> &[EventEnvelope] {
        &self.events
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
    ) -> Result<manifest::ValidatedExecutionManifest, manifest::RuntimeManifestError> {
        let validated =
            manifest::validate_and_build(manifest, knowledge, artifacts, self.revision())?;
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
    ) -> Result<manifest::ValidatedExecutionManifest, manifest::RuntimeManifestError> {
        let validated =
            manifest::validate_and_build(manifest, knowledge, artifacts, self.revision())?;
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
        self.admit_execution_manifest_with_authority(manifest, knowledge, artifacts, &authority)
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

    pub fn run_model_once<B: InferenceBackend + ?Sized>(
        &mut self,
        request_id: RequestId,
        raw_text: impl Into<String>,
        backend: &B,
    ) -> Result<Vec<ModelEvent>, RuntimeError> {
        let raw_text = raw_text.into();
        let _revision = self.ingest_text(request_id.clone(), raw_text.clone())?;
        let events = backend
            .infer(&ModelRequest {
                request_id: request_id.clone(),
                semantic_context: self.snapshot().semantic_context(),
                raw_text,
            })
            .map_err(|error| RuntimeError::Model(error.0))?;
        if events
            .iter()
            .any(|event| matches!(event, ModelEvent::Finished))
        {
            self.emit(RuntimeEvent::RequestFinished(request_id));
        }
        Ok(events)
    }

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
            // answer, across the boundary, whether that Pod exists.
            let pod = PodRouter
                .select(pods, project, &capability, &input_type)
                .and_then(|route| pods.get(project, &route.pod_id))
                .ok_or_else(|| RuntimeError::PodUnavailable {
                    capability: capability.to_string(),
                    input_type: input_type.to_string(),
                })?;

            if pod
                .manifest()
                .effects
                .iter()
                .any(|effect| !matches!(effect, ptr_types::Effect::Pure | ptr_types::Effect::Read))
            {
                return Err(RuntimeError::PodEffectRequiresActionBoundary {
                    pod: pod.manifest().id.to_string(),
                });
            }

            self.emit(RuntimeEvent::PodInvoked(pod.manifest().id.to_string()));
            let output = pod
                .invoke(TypedPayload {
                    type_id: input_type,
                    bytes: payload,
                })
                .map_err(RuntimeError::Pod)?;

            self.promote_verified_pod_output(&request_id, pod.manifest(), &output, verifier)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

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
        self.run_resumable_with_pods_using_router(
            request_id, project, raw_text, backend, pods, &PodRouter, verifier, max_rounds,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_resumable_with_pods_using_router<B, V>(
        &mut self,
        request_id: RequestId,
        project: &ProjectId,
        raw_text: impl Into<String>,
        backend: &B,
        pods: &PodRegistry,
        router: &PodRouter,
        verifier: &V,
        max_rounds: usize,
    ) -> Result<ResumableRun, RuntimeError>
    where
        B: ResumableInferenceBackend + ?Sized,
        V: Verifier<TypedPayload> + ?Sized,
    {
        let raw_text = raw_text.into();
        let _revision = self.ingest_text(request_id.clone(), raw_text.clone())?;
        let mut request = ModelRequest {
            request_id: request_id.clone(),
            semantic_context: self.snapshot().semantic_context(),
            raw_text,
        };
        let mut pending = backend
            .infer(&request)
            .map_err(|error| RuntimeError::Model(error.0))?;

        let mut run = ResumableRun::default();

        for round in 0..=max_rounds {
            let finished = pending
                .iter()
                .any(|event| matches!(event, ModelEvent::Finished));
            let pod_requests = pending
                .iter()
                .filter_map(|event| match event {
                    ModelEvent::PodRequested {
                        capability,
                        input_type,
                        payload,
                    } => Some((capability.clone(), input_type.clone(), payload.clone())),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let actions = pending
                .iter()
                .filter_map(|event| match event {
                    ModelEvent::ActionReady(action) => Some(action.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            run.model_events.extend(pending);

            let control_count = pod_requests.len() + actions.len();
            if control_count > 1 {
                if !pod_requests.is_empty() && !actions.is_empty() {
                    return Err(RuntimeError::MixedControlEvents);
                }
                return Err(RuntimeError::MultipleControlEvents {
                    count: control_count,
                });
            }

            if finished && control_count == 0 {
                self.emit(RuntimeEvent::RequestFinished(request_id));
                return Ok(run);
            }

            if let Some(action) = actions.into_iter().next() {
                run.action = Some(action);
                return Ok(run);
            }

            let Some((capability, input_type, payload)) = pod_requests.into_iter().next() else {
                return Err(RuntimeError::ModelNoProgress);
            };

            if round == max_rounds {
                return Err(RuntimeError::ModelResumeLimit { max_rounds });
            }

            // A Pod in another project is unavailable in exactly the same
            // words as a Pod that does not exist. Saying which it was would
            // answer, across the boundary, whether that Pod exists.
            let pod = router
                .select(pods, project, &capability, &input_type)
                .and_then(|route| pods.get(project, &route.pod_id))
                .ok_or_else(|| RuntimeError::PodUnavailable {
                    capability: capability.to_string(),
                    input_type: input_type.to_string(),
                })?;

            if pod
                .manifest()
                .effects
                .iter()
                .any(|effect| !matches!(effect, ptr_types::Effect::Pure | ptr_types::Effect::Read))
            {
                return Err(RuntimeError::PodEffectRequiresActionBoundary {
                    pod: pod.manifest().id.to_string(),
                });
            }

            self.emit(RuntimeEvent::PodInvoked(pod.manifest().id.to_string()));
            let output = pod
                .invoke(TypedPayload {
                    type_id: input_type,
                    bytes: payload,
                })
                .map_err(RuntimeError::Pod)?;

            let revision =
                self.promote_verified_pod_output(&request_id, pod.manifest(), &output, verifier)?;

            let observation = ModelObservation {
                revision,
                source: pod.manifest().id.to_string(),
                type_id: output.type_id.clone(),
                payload: output.bytes.clone(),
            };
            run.observations.push(output);

            request.semantic_context = self.snapshot().semantic_context();
            pending = backend
                .resume(&ModelResumeRequest {
                    request_id: request_id.clone(),
                    semantic_context: request.semantic_context.clone(),
                    round: round as u32 + 1,
                    observation,
                })
                .map_err(|error| RuntimeError::Model(error.0))?;
        }

        Err(RuntimeError::ModelResumeLimit { max_rounds })
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
        if self.execution.is_fenced() {
            return Err(RuntimeError::ExecutionFenced);
        }
        validate_new_record(&event)?;
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
        if request.output.revision < request.manifest.snapshot_revision {
            return Err(RuntimeError::StaleRevision {
                action: request.output.revision,
                current: request.manifest.snapshot_revision,
            });
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
        let output_digest = request.output.digest();
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
            let branch_id = format!(
                "pod-branch:{}:{}:{}",
                request.request_id,
                request.pod_id,
                digest_hex(&output_digest)
            );
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
                        pod_id,
                        output_digest: digest,
                        ..
                    },
                    2,
                ) => {
                    request_id == &request.request_id
                        && pod_id == &request.pod_id
                        && digest == output_digest
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
                        ..
                    },
                    0 | 1 | 3 | 4,
                ) => {
                    origin_request == &request.request_id.0
                        && pod == &request.pod_id.0
                        && digest == output_digest
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
                        && ptr_semdb::SemanticDelta::decode(encoded_delta)
                            .ok()
                            .and_then(|delta| {
                                delta
                                    .upserts
                                    .get(&semantic::pod_output_key(
                                        &request.request_id,
                                        &request.pod_id,
                                    ))
                                    .cloned()
                            })
                            .is_some_and(|value| match value {
                                SemanticValue::Payload(payload) => {
                                    payload.type_id == request.output.payload.type_id
                                        && payload.bytes == request.output.payload.bytes
                                }
                                SemanticValue::Text(_) => false,
                            })
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
        self.append_checked(event, None)
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
                // capsule IDs from impersonating the two reserved namespaces.
                if target.starts_with("constraint:") || target.starts_with("procedure:") {
                    return Err(RuntimeError::ReservedTargetNamespace { target });
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
                revision,
                output_type,
                output_digest,
                payload,
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
                let output = ptr_pods::PodOutput {
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
                };
                if output.digest() != *output_digest {
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
            // Revocation tombstones are monotone and may precede activation.
            // Verifier/snapshot records do not confer permissions or load state.
            LedgerEvent::SemanticDeltaCommitted { .. }
            | LedgerEvent::Revoked { .. }
            | LedgerEvent::ProcedureRevoked { .. }
            | LedgerEvent::VerifierAttested { .. }
            | LedgerEvent::SnapshotCommitted { .. } => Ok(()),
        }
    }

    fn apply_committed(
        &mut self,
        committed: &CommittedEvent,
        semantic: Option<PreparedDelta>,
    ) -> Result<(), RuntimeError> {
        match &committed.event {
            LedgerEvent::SemanticDeltaCommitted { .. } => {
                self.semdb
                    .apply_prepared(
                        semantic.ok_or(RuntimeError::Semantic(SemanticError::InvalidEncoding))?,
                    )
                    .map_err(RuntimeError::Semantic)?;
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
        }

        self.state.apply(committed);
        self.emit(RuntimeEvent::CommitApplied(committed.index));
        Ok(())
    }

    fn emit(&mut self, event: RuntimeEvent) {
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
fn validate_new_record(event: &LedgerEvent) -> Result<(), RuntimeError> {
    if matches!(event, LedgerEvent::SemanticDeltaCommitted { .. }) {
        return Err(RuntimeError::SemanticRecordOutsideSemanticPath);
    }
    if let LedgerEvent::EffectAttempted { key: Some(key), .. } = event {
        if !execution::valid_key(key) {
            return Err(RuntimeError::InvalidEffectKey { key: key.clone() });
        }
    }
    Ok(())
}
