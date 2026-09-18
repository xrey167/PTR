use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_events::{EventEnvelope, RuntimeEvent};
use ptr_ledger::{CommittedEvent, InMemoryLedger, Ledger, LedgerEvent};
use ptr_model_api::{InferenceBackend, ModelEvent, ModelRequest};
use ptr_security::PermissionSet;
use ptr_semdb::{SemanticDelta, SemanticHost, SemanticSnapshot};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, Generation, RequestId, Revision};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidConfig(String),
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
    PermissionDenied,
}

pub struct PtrRuntime {
    pub config: PtrConfig,
    semdb: SemanticHost,
    ledger: InMemoryLedger,
    state: MaterializedState,
    permissions: PermissionSet,
    live_generations: BTreeMap<String, Generation>,
    revoked_generations: BTreeSet<(String, Generation)>,
    events: Vec<EventEnvelope>,
    next_event_sequence: u64,
}

impl PtrRuntime {
    pub fn new(config: PtrConfig) -> Result<Self, RuntimeError> {
        config.validate().map_err(RuntimeError::InvalidConfig)?;
        Ok(Self {
            config,
            semdb: SemanticHost::default(),
            ledger: InMemoryLedger::default(),
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
            let actual = runtime.ledger.append(expected.event.clone());
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
        if events.iter().any(|event| matches!(event, ModelEvent::Finished)) {
            self.emit(RuntimeEvent::RequestFinished(request_id));
        }
        Ok(events)
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

    pub fn commit(&mut self, event: LedgerEvent) -> CommitIndex {
        let index = self.ledger.append(event);
        let committed = self
            .ledger
            .events()
            .last()
            .expect("append created committed event")
            .clone();
        self.apply_committed(&committed);
        index
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