use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_state::{classify_next, projection_entries, ApplyOutcome, MaterializedState};
use ptr_types::{CommitIndex, Generation, Revision};

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

#[test]
fn commit_classification_does_not_wrap_at_the_u64_boundary() {
    assert_eq!(classify_next(u64::MAX - 1, u64::MAX), None);
    assert_eq!(
        classify_next(u64::MAX - 2, u64::MAX),
        Some(ApplyOutcome::Gap)
    );
    assert_eq!(classify_next(u64::MAX, 0), Some(ApplyOutcome::OutOfOrder));
    assert_eq!(classify_next(0, 0), Some(ApplyOutcome::Duplicate));
}

#[test]
fn rejected_events_do_not_leak_projected_values_and_can_be_retried_in_order() {
    let mut state = MaterializedState::default();
    state.try_apply(&event(1, "first"));
    let before = state.values.clone();
    for (index, outcome) in [
        (1, ApplyOutcome::Duplicate),
        (0, ApplyOutcome::OutOfOrder),
        (3, ApplyOutcome::Gap),
    ] {
        assert_eq!(state.try_apply(&event(index, "must-not-appear")), outcome);
        assert_eq!(state.values, before);
        assert_eq!(state.last_applied, 1);
    }
    assert_eq!(state.try_apply(&event(2, "next")), ApplyOutcome::Applied);
    assert_eq!(
        state.try_apply(&event(3, "must-not-appear")),
        ApplyOutcome::Applied
    );
    assert_eq!(
        state
            .values
            .get("constraint:must-not-appear")
            .map(String::as_str),
        Some("3")
    );
}

#[test]
fn semantic_projection_contains_only_the_revision_and_never_payload_bytes() {
    let committed = CommittedEvent {
        index: CommitIndex(1),
        event: LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(40),
            revision: Revision(41),
            encoded_delta: b"private payload".to_vec(),
        },
    };
    assert_eq!(
        projection_entries(&committed),
        vec![("semdb:revision".into(), "41".into())]
    );
    let mut state = MaterializedState::default();
    assert_eq!(state.try_apply(&committed), ApplyOutcome::Applied);
    assert_eq!(state.values.len(), 1);
    assert_eq!(
        state.values.get("semdb:revision").map(String::as_str),
        Some("41")
    );
}
