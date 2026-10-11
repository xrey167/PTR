//! The raft-engine failpoints the L001-v2 harness drives, reachable through the
//! `raft-engine-failpoints` feature. Instrumentation only: this checks that the
//! points are hit, in the order a synchronized write runs them, and that a
//! callback at each one does not disturb the append.
use fail::FailScenario;
use ptr_ledger::{LedgerEvent, RaftEngineLedger};
use ptr_types::Generation;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

const POINTS: [&str; 3] = [
    "file_pipe_log::append",
    "log_file::write::err",
    "log_fd::sync::err",
];

#[test]
fn a_synchronized_append_hits_append_then_write_then_sync() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ptr-engine-fp-{nonce}"));
    let scenario = FailScenario::setup();
    let mut ledger = RaftEngineLedger::open(&dir).unwrap();
    let hits = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    for point in POINTS {
        let hits = Arc::clone(&hits);
        fail::cfg_callback(point, move || hits.lock().unwrap().push(point)).unwrap();
    }
    let index = ledger
        .append_durable(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    assert_eq!(index.0, 1);
    let seen = hits.lock().unwrap().clone();
    let first = |name: &str| seen.iter().position(|hit| *hit == name);
    let (append, write, sync) = (
        first(POINTS[0]).expect("append point hit"),
        first(POINTS[1]).expect("write point hit"),
        first(POINTS[2]).expect("sync point hit"),
    );
    assert!(append < write && write < sync, "{seen:?}");
    scenario.teardown();
    drop(ledger);
    let reopened = RaftEngineLedger::open(&dir).unwrap();
    assert_eq!(reopened.events().len(), 1);
    drop(reopened);
    std::fs::remove_dir_all(dir).unwrap();
}
