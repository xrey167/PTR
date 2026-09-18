use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerEvent {
    CapsuleCommitted { project: ProjectId, capsule: CapsuleId, generation: Generation },
    CapsuleSuperseded { capsule: CapsuleId, old: Generation, new: Generation },
    Revoked { subject: String, generation: Generation },
    HardConstraintCommitted { key: String, generation: Generation },
    VerifierAttested { subject: String, passed: bool },
    ProcedurePromoted { id: String, generation: Generation },
    ProcedureRevoked { id: String, generation: Generation },
    SnapshotCommitted { revision: u64, covers: CommitIndex },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedEvent { pub index: CommitIndex, pub event: LedgerEvent }

pub trait Ledger { fn append(&mut self, event: LedgerEvent) -> CommitIndex; fn events(&self) -> &[CommittedEvent]; }

#[derive(Clone, Debug, Default)]
pub struct InMemoryLedger { events: Vec<CommittedEvent> }
impl Ledger for InMemoryLedger {
    fn append(&mut self, event: LedgerEvent) -> CommitIndex {
        let index = CommitIndex(self.events.len() as u64 + 1);
        self.events.push(CommittedEvent { index, event }); index
    }
    fn events(&self) -> &[CommittedEvent] { &self.events }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionBarrier { pub snapshot_covers: CommitIndex, pub all_consumers_caught_up: bool, pub unresolved_revocations: usize }
impl CompactionBarrier { pub fn safe(&self) -> bool { self.all_consumers_caught_up && self.unresolved_revocations == 0 } }
