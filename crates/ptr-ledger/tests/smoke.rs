use ptr_ledger::{InMemoryLedger, Ledger, LedgerEvent};
use ptr_types::{CommitIndex, Generation};

fn revocation(subject: &str) -> LedgerEvent {
    LedgerEvent::Revoked {
        subject: subject.into(),
        generation: Generation(1),
    }
}

#[test]
fn append_assigns_monotonic_index() {
    let mut l = InMemoryLedger::default();
    let a = l.append(revocation("x")).unwrap();
    let b = l.append(revocation("y")).unwrap();
    assert!(b.0 > a.0);
}

#[test]
fn an_exhausted_commit_index_is_refused_rather_than_repeated() {
    // A floor at the ceiling is only reachable from a trusted compacted anchor,
    // but the failure it would produce is the one this layer must never have:
    // two records at the same index. The append is refused and nothing is stored.
    let mut ledger = InMemoryLedger::resuming_above(CommitIndex(u64::MAX));
    let error = ledger.append(revocation("x")).unwrap_err();
    assert_eq!(error.to_string(), "PTR_LEDGER_INDEX_EXHAUSTED");
    assert!(ledger.events().is_empty());

    // One below the ceiling still commits exactly once, and the next append is
    // refused instead of repeating that index.
    let mut ledger = InMemoryLedger::resuming_above(CommitIndex(u64::MAX - 1));
    assert_eq!(
        ledger.append(revocation("x")).unwrap(),
        CommitIndex(u64::MAX)
    );
    assert!(ledger.append(revocation("y")).is_err());
    assert_eq!(ledger.events().len(), 1);
}
