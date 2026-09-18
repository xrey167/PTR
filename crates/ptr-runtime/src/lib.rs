use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_events::{EventEnvelope, RuntimeEvent};
use ptr_ledger::{CommittedEvent, InMemoryLedger, Ledger, LedgerEvent};
use ptr_security::PermissionSet;
use ptr_semdb::{SemanticDelta, SemanticHost, SemanticSnapshot};
use ptr_state::MaterializedState;
use ptr_types::{CommitIndex, RequestId, Revision};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidConfig(String),
    StaleRevision { action: Revision, current: Revision },
    PermissionDenied,
}

pub struct PtrRuntime {
    pub config: PtrConfig,
    semdb: SemanticHost,
    ledger: InMemoryLedger,
    state: MaterializedState,
    permissions: PermissionSet,
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
            events: Vec::new(),
            next_event_sequence: 1,
        })
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

    pub fn materialized_state(&self) -> &MaterializedState {
        &self.state
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

    pub fn authorize_action(&self, action: &ActionIr) -> Result<(), RuntimeError> {
        if self.config.action_boundary.require_current_revision
            && action.revision != self.semdb.revision()
        {
            return Err(RuntimeError::StaleRevision {
                action: action.revision,
                current: self.semdb.revision(),
            });
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
        let committed: CommittedEvent = self
            .ledger
            .events()
            .last()
            .expect("append created committed event")
            .clone();
        self.state.apply(&committed);
        self.emit(RuntimeEvent::CommitApplied(index));
        index
    }

    fn emit(&mut self, event: RuntimeEvent) {
        self.events.push(EventEnvelope {
            sequence: self.next_event_sequence,
            event,
        });
        self.next_event_sequence = self.next_event_sequence.saturating_add(1);
    }
}
