pub mod admission;
pub mod compacted;
pub mod directory;
pub mod execution;
pub mod kv;
pub mod memory;
pub mod merge;
pub mod migration;
pub mod neural;
pub mod persistence;
pub mod placement;
pub mod protected_state;
pub mod sandbox;
pub mod scopes;
pub mod semantic;
pub mod tensor_kv;

pub use admission::{AdmissionError, AdmissionGrant, IdentityAdmissionController};
pub use directory::{InMemoryPodDirectory, PodDirectory};
pub use kv::{KvUseError, KvUseRequest, KvUseTicket, RuntimeKvAuthority};
pub use merge::{
    ChangeOrigin, HoldReason, MergeAuthority, MergeHold, MergeOutcome, MergePreview, MergeReceipt,
    SemanticChange, SemanticGrant, SemanticGrantInfo, SemanticRefusal, SemanticVerdict,
};
pub use migration::{KvMigrationController, MigrationError, MigrationRecord, MigrationState};
pub use placement::{
    DeviceHealth, DeviceRecord, FencedStateLease, FencingToken, NodeHealth, NodeRecord, Placement,
    PlacementEpoch, PlacementError, PodPlacementController,
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
pub use tensor_kv::{ManagedKvHandle, ManagedKvMetadata, ManagedKvRegistry, TensorKvError};

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
    integrity, CommittedEvent, FileLedger, InMemoryLedger, Ledger, LedgerEvent,
    MAX_RETAINED_RESPONSE,
};
use ptr_model_api::{
    InferenceBackend, ModelEvent, ModelObservation, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
use ptr_router::PodRouter;
use ptr_security::{
    ActionAuthorization, AuthorizationDecision, AuthorizationDenial, PermissionSet,
};
use ptr_semdb::{PreparedDelta, SemanticError, SemanticHost, SemanticSnapshot};
use ptr_state::MaterializedState;
use ptr_types::{
    CommitIndex, Generation, ProjectId, RequestId, Revision, ScopeId, ScopeLeaseBinding,
    StatefulRequestRecovery, UncertainRequest, Validity,
};
use ptr_verifier::Verifier;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

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
    PodOutputTypeMismatch {
        pod: String,
        expected: String,
        actual: String,
    },
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
                domain, logical_id, ..
            } => {
                if *domain > 4 || logical_id.is_empty() {
                    return Err(RuntimeError::InvalidConfig(
                        "invalid protected-state ledger reference".into(),
                    ));
                }
                Ok(())
            }
            LedgerEvent::MeshTunnelLifecycle(event) => event.validate().map_err(|reason| {
                RuntimeError::InvalidConfig(format!("invalid mesh event: {reason}"))
            }),
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
            LedgerEvent::ProtectedStateCommitted { .. } | LedgerEvent::MeshTunnelLifecycle(_) => {}
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
