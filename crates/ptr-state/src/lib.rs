use ptr_ledger::{CommittedEvent, LedgerEvent};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyOutcome {
    Applied,
    Duplicate,
    OutOfOrder,
    Gap,
}

#[derive(Clone, Debug, Default)]
pub struct MaterializedState {
    pub values: BTreeMap<String, String>,
    pub last_applied: u64,
}

impl MaterializedState {
    pub fn try_apply(&mut self, committed: &CommittedEvent) -> ApplyOutcome {
        let index = committed.index.0;
        if index == self.last_applied {
            return ApplyOutcome::Duplicate;
        }
        if index < self.last_applied {
            return ApplyOutcome::OutOfOrder;
        }
        if index != self.last_applied.saturating_add(1) {
            return ApplyOutcome::Gap;
        }

        match &committed.event {
            LedgerEvent::Revoked {
                subject,
                generation,
            } => {
                self.values
                    .insert(format!("revoked:{subject}"), generation.0.to_string());
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => {
                self.values
                    .insert(format!("constraint:{key}"), generation.0.to_string());
            }
            _ => {}
        }
        self.last_applied = index;
        ApplyOutcome::Applied
    }

    pub fn apply(&mut self, committed: &CommittedEvent) {
        let _ = self.try_apply(committed);
    }
}
