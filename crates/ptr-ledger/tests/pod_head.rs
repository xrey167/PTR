use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_types::{
    Generation, NamespaceId, PodAddress, PodId, PodIdentity, PodRevisionAddress, ProjectId,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn address(generation: u64) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(
            ProjectId::from("project"),
            NamespaceId::from("ns"),
            PodId::from("reader"),
        )
        .unwrap(),
        semantic_revision: [4; 32],
        generation: Generation(generation),
    }
}

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "ptr-{name}-{}.ledger",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn pod_head_moves_round_trip_through_the_durable_ledger() {
    let path = temp_path("pod-head");
    let first = LedgerEvent::PodHeadAdvanced {
        identity: PodIdentity::from("pod:stable"),
        old: None,
        new: address(1),
    };
    let second = LedgerEvent::PodHeadAdvanced {
        identity: PodIdentity::from("pod:stable"),
        old: Some(address(1)),
        new: address(2),
    };
    {
        let mut ledger = FileLedger::open(&path).unwrap();
        ledger.append_durable(first.clone()).unwrap();
        ledger.append_durable(second.clone()).unwrap();
    }
    let ledger = FileLedger::open(&path).unwrap();
    let events: Vec<_> = ledger.events().iter().map(|e| e.event.clone()).collect();
    assert_eq!(events, vec![first, second]);
    drop(ledger);
    let _ = std::fs::remove_file(path);
}

#[test]
fn a_non_rising_head_move_is_refused_before_any_byte_is_appended() {
    let path = temp_path("pod-head-refused");
    let stale = LedgerEvent::PodHeadAdvanced {
        identity: PodIdentity::from("pod:stable"),
        old: Some(address(2)),
        new: address(2),
    };
    let mut ledger = FileLedger::open(&path).unwrap();
    assert!(ledger.append_durable(stale).is_err());
    assert!(ledger.events().is_empty());
    drop(ledger);
    let _ = std::fs::remove_file(path);
}
