use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_types::{Generation, Revision, SessionId, TraceId};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn pod_evidence_round_trips_through_the_durable_ledger() {
    let path = std::env::temp_dir().join(format!(
        "ptr-pod-evidence-{}.ledger",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let event = LedgerEvent::PodEvidenceCommitted {
        session_id: SessionId::from("session"),
        trace_id: TraceId::from("trace"),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        generation: Generation(3),
        revision: Revision(4),
        bundle_digest: [5; 32],
        bundle: b"canonical-evidence".to_vec(),
    };
    {
        let mut ledger = FileLedger::open(&path).unwrap();
        ledger.append_durable(event.clone()).unwrap();
    }
    let ledger = FileLedger::open(&path).unwrap();
    assert_eq!(ledger.events().len(), 1);
    assert_eq!(ledger.events()[0].event, event);
    drop(ledger);
    let _ = std::fs::remove_file(path);
}
