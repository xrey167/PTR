use ptr_ledger::{CommittedEvent, LedgerEvent};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct MaterializedState {
    pub values: BTreeMap<String, String>,
    pub last_applied: u64,
}
impl MaterializedState {
    pub fn apply(&mut self, committed: &CommittedEvent) {
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
        self.last_applied = committed.index.0;
    }
}
