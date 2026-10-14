//! Fault-injection tests for `DurableMemoryStore` (a child module so it can reach the journal).

use super::*;
use crate::journal::test_support::TempPath;
use crate::journal::FaultInjection;

fn new_record(id: &str) -> NewMemoryRecord {
    NewMemoryRecord {
        id: MemoryRecordId::from(id),
        class: MemoryClass::Episodic,
        body: format!("body of {id}"),
        sensitive: false,
        pinned: false,
        supports: vec![],
        derived_from: vec![],
    }
}

fn quiet() -> MemoryStoreOptions {
    MemoryStoreOptions {
        compact_after_records: None,
        ..MemoryStoreOptions::default()
    }
}

#[test]
fn sync_failure_is_invisible_and_a_retry_applies_exactly_once() {
    let path = TempPath::new("ms-syncfail");
    let mut store = DurableMemoryStore::open(&path.0, quiet()).unwrap();
    store.insert(new_record("a")).unwrap();
    let bytes = std::fs::read(&path.0).unwrap();
    let clock = store.clock();

    store.journal.inject_fault(FaultInjection {
        fail_sync: true,
        fail_truncate: false,
    });
    assert!(matches!(
        store.insert(new_record("b")),
        Err(MemoryStoreError::Journal(JournalError::Io { .. }))
    ));
    assert_eq!(std::fs::read(&path.0).unwrap(), bytes);
    assert!(store.get(&MemoryRecordId::from("b")).is_none());
    assert_eq!(store.clock(), clock);

    store.journal.inject_fault(FaultInjection::default());
    store.insert(new_record("b")).unwrap();
    drop(store);
    let reopened = DurableMemoryStore::open(&path.0, quiet()).unwrap();
    assert_eq!(reopened.records().count(), 2);
    assert!(reopened.get(&MemoryRecordId::from("b")).is_some());
}

#[test]
fn failed_rollback_poisons_and_stops_appends() {
    let path = TempPath::new("ms-truncfail");
    let mut store = DurableMemoryStore::open(&path.0, quiet()).unwrap();
    store.journal.inject_fault(FaultInjection {
        fail_sync: true,
        fail_truncate: true,
    });
    assert!(store.insert(new_record("b")).is_err());
    assert!(store.get(&MemoryRecordId::from("b")).is_none());
    store.journal.inject_fault(FaultInjection::default());
    assert!(matches!(
        store.insert(new_record("c")),
        Err(MemoryStoreError::Journal(JournalError::Poisoned))
    ));
}
