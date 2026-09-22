pub mod compacted;
pub mod execution;
pub mod neural;
pub mod persistence;
pub mod semantic;

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
use ptr_security::{
    ActionAuthorization, AuthorizationDecision, AuthorizationDenial, PermissionSet,
};
use ptr_semdb::{PreparedDelta, SemanticError, SemanticHost, SemanticSnapshot};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, Generation, ProjectId, RequestId, Revision};
use ptr_verifier::{VerificationStatus, Verifier};
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
    ModelResumeLimit {
        max_rounds: usize,
    },
    ModelNoProgress,
    MultiplePodRequests {
        count: usize,
    },
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
    /// An at-most-once key is not a well-formed identifier.
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
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResumableRun {
    pub model_events: Vec<ModelEvent>,
    pub observations: Vec<TypedPayload>,
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
            let semantic = runtime.prepare_semantic_event(&committed.event)?;
            runtime.apply_committed(committed, semantic)?;
        }
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
        })
    }

    pub fn replay(config: PtrConfig, events: &[CommittedEvent]) -> Result<Self, RuntimeError> {
        let mut runtime = Self::new(config)?;
        for expected in events {
            runtime.validate_lifecycle_event(&expected.event)?;
            let semantic = runtime.prepare_semantic_event(&expected.event)?;
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

    pub fn materialized_state(&self) -> &MaterializedState {
        &self.state
    }

    fn set_live_generation(&mut self, target: impl Into<String>, generation: Generation) {
        self.live_generations.insert(target.into(), generation);
    }

    pub fn live_generation(&self, target: &str) -> Option<Generation> {
        self.live_generations.get(target).copied()
    }

    pub fn run_model_once<B: InferenceBackend>(
        &mut self,
        request_id: RequestId,
        raw_text: impl Into<String>,
        backend: &B,
    ) -> Result<Vec<ModelEvent>, RuntimeError> {
        let raw_text = raw_text.into();
        let revision = self.ingest_text(request_id.clone(), raw_text.clone())?;
        let events = backend
            .infer(&ModelRequest {
                request_id: request_id.clone(),
                revision,
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
        B: InferenceBackend,
        V: Verifier<TypedPayload>,
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
            let pod = pods
                .resolve(project, &capability, &input_type)
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

            let report = verifier.verify(&output);
            let passed = report.status == VerificationStatus::Pass;
            self.emit(RuntimeEvent::VerifierResult {
                verifier: "pod-output".into(),
                passed,
            });
            if !passed {
                return Err(RuntimeError::PodVerificationFailed {
                    pod: pod.manifest().id.to_string(),
                });
            }

            self.promote_pod_output(&request_id, &pod.manifest().id, &output)?;
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
        B: ResumableInferenceBackend,
        V: Verifier<TypedPayload>,
    {
        let raw_text = raw_text.into();
        let revision = self.ingest_text(request_id.clone(), raw_text.clone())?;
        let mut request = ModelRequest {
            request_id: request_id.clone(),
            revision,
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

            run.model_events.extend(pending);

            if finished && pod_requests.is_empty() {
                self.emit(RuntimeEvent::RequestFinished(request_id));
                return Ok(run);
            }

            if pod_requests.len() > 1 {
                return Err(RuntimeError::MultiplePodRequests {
                    count: pod_requests.len(),
                });
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
            let pod = pods
                .resolve(project, &capability, &input_type)
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

            let report = verifier.verify(&output);
            let passed = report.status == VerificationStatus::Pass;
            self.emit(RuntimeEvent::VerifierResult {
                verifier: "pod-output".into(),
                passed,
            });
            if !passed {
                return Err(RuntimeError::PodVerificationFailed {
                    pod: pod.manifest().id.to_string(),
                });
            }

            let revision = self.promote_pod_output(&request_id, &pod.manifest().id, &output)?;

            let observation = ModelObservation {
                revision,
                source: pod.manifest().id.to_string(),
                type_id: output.type_id.clone(),
                payload: output.bytes.clone(),
            };
            run.observations.push(output);

            request.revision = revision;
            pending = backend
                .resume(&ModelResumeRequest {
                    request_id: request_id.clone(),
                    revision,
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
        self.validate_lifecycle_event(&event)?;
        let semantic = self.prepare_semantic_event(&event)?;
        self.append_prepared(event, semantic)
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
                target,
                operation,
                effect,
                ..
            } => {
                self.execution.record_attempt(execution::UnsettledEffect {
                    attempt: committed.index,
                    key: key.clone(),
                    target: target.clone(),
                    operation: operation.clone(),
                    effect: *effect,
                });
            }
            LedgerEvent::EffectSettled {
                attempt, response, ..
            } => {
                let outcome = match response {
                    Some(response) => execution::SettledOutcome::Applied {
                        response: response.clone(),
                    },
                    None => execution::SettledOutcome::AppliedWithoutResponse,
                };
                self.execution.settle(*attempt, outcome);
            }
            LedgerEvent::EffectReconciled {
                attempt, applied, ..
            } => {
                // Reconciliation establishes whether the effect applied, never a
                // response: the runtime that could have reproduced one would not
                // have needed reconciling.
                let outcome = if *applied {
                    execution::SettledOutcome::AppliedWithoutResponse
                } else {
                    execution::SettledOutcome::NotApplied
                };
                self.execution.settle(*attempt, outcome);
            }
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
