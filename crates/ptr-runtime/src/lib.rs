use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_events::{EventEnvelope, RuntimeEvent};
use ptr_ledger::{CommittedEvent, FileLedger, InMemoryLedger, Ledger, LedgerEvent};
use ptr_model_api::{
    InferenceBackend, ModelEvent, ModelObservation, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::PodRegistry;
use ptr_protocol::TypedPayload;
use ptr_security::PermissionSet;
use ptr_semdb::{SemanticDelta, SemanticHost, SemanticSnapshot};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, Generation, RequestId, Revision};
use ptr_verifier::{VerificationStatus, Verifier};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
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
            Self::Memory(ledger) => Ok(ledger.append(event)),
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
    pub config: PtrConfig,
    semdb: SemanticHost,
    ledger: RuntimeLedger,
    state: MaterializedState,
    permissions: PermissionSet,
    live_generations: BTreeMap<String, Generation>,
    revoked_generations: BTreeSet<(String, Generation)>,
    events: Vec<EventEnvelope>,
    next_event_sequence: u64,
}

impl PtrRuntime {
    pub fn new(config: PtrConfig) -> Result<Self, RuntimeError> {
        Self::with_ledger(config, RuntimeLedger::Memory(InMemoryLedger::default()))
    }

    pub fn open_durable(
        config: PtrConfig,
        path: impl AsRef<Path>,
    ) -> Result<Self, RuntimeError> {
        let ledger =\n            FileLedger::open(path).map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        let persisted = ledger.events().to_vec();
        let mut runtime = Self::with_ledger(config, RuntimeLedger::File(ledger))?;
        for committed in &persisted {
            runtime.apply_committed(committed);
        }
        Ok(runtime)
    }

    fn with_ledger(config: PtrConfig, ledger: RuntimeLedger) -> Result<Self, RuntimeError> {
        config.validate().map_err(RuntimeError::InvalidConfig)?;
        Ok(Self {
            config,
            semdb: SemanticHost::default(),
            ledger,
            state: MaterializedState::default(),
            permissions: PermissionSet::default(),
            live_generations: BTreeMap::new(),
            revoked_generations: BTreeSet::new(),
            events: Vec::new(),
            next_event_sequence: 1,
        })
    }

    pub fn replay(config: PtrConfig, events: &[CommittedEvent]) -> Result<Self, RuntimeError> {
        let mut runtime = Self::new(config)?;
        for expected in events {
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
            runtime.apply_committed(&committed);
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

    pub fn set_live_generation(&mut self, target: impl Into<String>, generation: Generation) {
        self.live_generations.insert(target.into(), generation);
    }

    pub fn live_generation(&self, target: &str) -> Option<Generation> {
        self.live_generations.get(target).copied()
    }

    pub fn ingest_text(&mut self, request: RequestId, text: impl Into<String>) -> Revision {
        self.emit(RuntimeEvent::RequestStarted(request.clone()));
        let mut delta = SemanticDelta::default();
        delta
            .upserts
            .insert(format!("request:{request}:raw"), text.into());
        let (revision, _) = self.semdb.apply_delta(delta);
        self.emit(RuntimeEvent::SnapshotOpened(revision));
        revision
    }

    pub fn run_model_once<B: InferenceBackend>(
        &mut self,
        request_id: RequestId,
        raw_text: impl Into<String>,
        backend: &B,
    ) -> Result<Vec<ModelEvent>, RuntimeError> {
        let raw_text = raw_text.into();
        let revision = self.ingest_text(request_id.clone(), raw_text.clone());
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

            let pod = pods.resolve(&capability, &input_type).ok_or_else(|| {
                RuntimeError::PodUnavailable {
                    capability: capability.to_string(),
                    input_type: input_type.to_string(),
                }
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

            let mut delta = SemanticDelta::default();
            delta.upserts.insert(
                format!("request:{request_id}:pod:{}:output_type", pod.manifest().id),
                output.type_id.to_string(),
            );
            self.semdb.apply_delta(delta);
            outputs.push(output);
        }

        Ok(outputs)
    }

    pub fn run_resumable_with_pods<B, V>(
        &mut self,
        request_id: RequestId,
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
        let revision = self.ingest_text(request_id.clone(), raw_text.clone());
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

            let pod = pods.resolve(&capability, &input_type).ok_or_else(|| {
                RuntimeError::PodUnavailable {
                    capability: capability.to_string(),
                    input_type: input_type.to_string(),
                }
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

            let mut delta = SemanticDelta::default();
            delta.upserts.insert(
                format!("request:{request_id}:pod:{}:output_type", pod.manifest().id),
                output.type_id.to_string(),
            );
            let (revision, _) = self.semdb.apply_delta(delta);

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

    pub fn authorize_action(&self, action: &ActionIr) -> Result<(), RuntimeError> {
        if self.config.action_boundary.require_current_revision
            && action.revision != self.semdb.revision()
        {
            return Err(RuntimeError::StaleRevision {
                action: action.revision,
                current: self.semdb.revision(),
            });
        }

        if self.config.action_boundary.require_live_generation {
            let current = self.live_generations.get(&action.target).copied();
            if self
                .revoked_generations
                .contains(&(action.target.clone(), action.generation))
                || current.is_some_and(|generation| generation != action.generation)
            {
                return Err(RuntimeError::StaleGeneration {
                    target: action.target.clone(),
                    action: action.generation,
                    current,
                });
            }
            if current.is_none() {
                return Err(RuntimeError::UnknownGeneration {
                    target: action.target.clone(),
                });
            }
        }

        if self.config.action_boundary.require_capability
            && !self.permissions.allows(&action.capability, action.effect)
        {
            return Err(RuntimeError::PermissionDenied);
        }
        Ok(())
    }

    pub fn commit(&mut self, event: LedgerEvent) -> Result<CommitIndex, RuntimeError> {
        let index = self.ledger.append(event)?;
        let committed = self
            .ledger
            .events()
            .last()
            .expect("append created committed event")
            .clone();
        self.apply_committed(&committed);
        Ok(index)
    }

    fn apply_committed(&mut self, committed: &CommittedEvent) {
        match &committed.event {
            LedgerEvent::CapsuleCommitted {
                capsule,
                generation,
                ..
            } => {
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
        }

        self.state.apply(committed);
        self.emit(RuntimeEvent::CommitApplied(committed.index));
    }

    fn emit(&mut self, event: RuntimeEvent) {
        self.events.push(EventEnvelope {
            sequence: self.next_event_sequence,
            event,
        });
        self.next_event_sequence = self.next_event_sequence.saturating_add(1);
    }
}
