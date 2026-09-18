use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_state::{ApplyOutcome, MaterializedState};
use ptr_types::{CommitIndex, Generation};

fn event(index: u64, key: &str) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(index),
        event: LedgerEvent::HardConstraintCommitted {
            key: key.into(),
            generation: Generation(index),
        },
    }
}

#[test]
fn duplicate_and_out_of_order_events_do_not_rewind_state() {
    let mut state = MaterializedState::default();
    assert_eq!(state.try_apply(&event(1, "a")), ApplyOutcome::Applied);
    assert_eq!(state.try_apply(&event(1, "a")), ApplyOutcome::Duplicate);
    assert_eq!(state.try_apply(&event(3, "c")), ApplyOutcome::Gap);
    assert_eq!(state.last_applied, 1);
    assert_eq!(state.try_apply(&event(2, "b")), ApplyOutcome::Applied);
    assert_eq!(state.try_apply(&event(1, "a")), ApplyOutcome::OutOfOrder);
    assert_eq!(state.last_applied, 2);
}
