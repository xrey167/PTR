use ptr_ledger::{Attestation, FileLedger, LedgerEvent};
use ptr_types::{
    Generation, PodId, RequestId, Revision, ScopeId, SessionId, TypeId, VerificationLevel,
};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn pod_output_admission_round_trips_through_the_durable_ledger() {
    let path = std::env::temp_dir().join(format!(
        "ptr-pod-output-{}.ledger",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let event = LedgerEvent::PodOutputAdmitted {
        request_id: RequestId::from("request"),
        session_id: SessionId::from("session"),
        scope_id: ScopeId::from("scope"),
        pod_id: PodId::from("pod"),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        generation: Generation(3),
        revision: Revision(4),
        output_kind: 1,
        output_type: TypeId::from("text"),
        output_digest: [5; 32],
        verification: Attestation {
            required: VerificationLevel::FullSemantic,
            level: VerificationLevel::Deterministic,
            verifiers: vec!["pod-output".into()],
            findings: vec![],
        },
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

#[test]
fn pod_hypothesis_round_trips_with_payload_and_lineage() {
    let path = std::env::temp_dir().join(format!(
        "ptr-pod-hypothesis-{}.ledger",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let event = LedgerEvent::PodHypothesisCommitted {
        request_id: RequestId::from("request"),
        session_id: SessionId::from("session"),
        scope_id: ScopeId::from("scope"),
        branch_id: "branch-1".into(),
        pod_id: PodId::from("pod"),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        generation: Generation(3),
        revision: Revision(4),
        output_type: TypeId::from("text"),
        output_digest: [5; 32],
        payload: b"hypothesis".to_vec(),
        provenance: vec![("evidence".into(), Some("note".into()))],
        dependencies: vec![[6; 32]],
        confidence_bits: 1.0f32.to_bits(),
        latency_millis: 7,
        verification: Attestation {
            required: VerificationLevel::FullSemantic,
            level: VerificationLevel::Deterministic,
            verifiers: vec!["pod-output".into()],
            findings: vec![],
        },
    };
    {
        let mut ledger = FileLedger::open(&path).unwrap();
        ledger.append_durable(event.clone()).unwrap();
    }
    let ledger = FileLedger::open(&path).unwrap();
    assert_eq!(ledger.events()[0].event, event);
    drop(ledger);
    let _ = std::fs::remove_file(path);
}
