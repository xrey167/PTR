//! Fault-injection tests for `DurableKnowledgeStore` (a child module so it can reach the journal).

use super::*;
use crate::journal::test_support::TempPath;
use crate::journal::FaultInjection;
use ptr_types::{EventRole, RawEventId, SessionId, Timestamp};

fn event(id: &str) -> RawEvent {
    RawEvent::new(
        RawEventId::from(id),
        SessionId::from("s"),
        EventRole::User,
        id.as_bytes().to_vec(),
        Timestamp(1),
    )
}

#[test]
fn sync_failure_is_invisible_and_a_retry_applies_exactly_once() {
    let path = TempPath::new("dk-syncfail");
    let options = DurableOptions {
        compact_after_records: None,
        ..DurableOptions::default()
    };
    let mut durable = DurableKnowledgeStore::open(&path.0, options).unwrap();
    durable.append_raw(event("a")).unwrap();
    let digest = durable.state_digest();
    let bytes = std::fs::read(&path.0).unwrap();

    durable.journal.inject_fault(FaultInjection {
        fail_sync: true,
        fail_truncate: false,
        ..FaultInjection::default()
    });
    assert!(matches!(
        durable.append_raw(event("b")),
        Err(DurableKnowledgeError::Journal(JournalError::Io { .. }))
    ));
    assert_eq!(std::fs::read(&path.0).unwrap(), bytes);
    assert!(durable.store().raw(&RawEventId::from("b")).is_none());
    assert_eq!(durable.state_digest(), digest);

    durable.journal.inject_fault(FaultInjection::default());
    assert!(matches!(durable.append_raw(event("b")), Ok(true)));
    let digest = durable.state_digest();
    drop(durable);
    let reopened = DurableKnowledgeStore::open(&path.0, options).unwrap();
    assert_eq!(reopened.state_digest(), digest);
    assert!(reopened.store().raw(&RawEventId::from("b")).is_some());
}

#[test]
fn failed_rollback_poisons_the_wrapper_and_stops_appends() {
    let path = TempPath::new("dk-truncfail");
    let mut durable = DurableKnowledgeStore::open(&path.0, DurableOptions::default()).unwrap();
    durable.journal.inject_fault(FaultInjection {
        fail_sync: true,
        fail_truncate: true,
        ..FaultInjection::default()
    });
    assert!(durable.append_raw(event("b")).is_err());
    durable.journal.inject_fault(FaultInjection::default());
    assert!(matches!(
        durable.append_raw(event("c")),
        Err(DurableKnowledgeError::Poisoned)
    ));
}
