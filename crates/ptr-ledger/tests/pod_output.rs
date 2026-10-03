use ptr_ledger::{Attestation, FileLedger, LedgerEvent};
use ptr_types::{
    Generation, PodId, RequestId, Revision, ScopeId, SessionId, TypeId, VerificationLevel,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn pod_events(
    verification: Attestation,
    provenance: usize,
    dependencies: usize,
) -> [LedgerEvent; 2] {
    [
        LedgerEvent::PodOutputAdmitted {
            request_id: RequestId::from("request"),
            session_id: SessionId::from("session"),
            scope_id: ScopeId::from("scope"),
            pod_id: PodId::from("pod"),
            manifest_digest: [1; 32],
            artifact_digest: [2; 32],
            generation: Generation(1),
            revision: Revision(1),
            output_kind: 1,
            output_type: TypeId::from("text"),
            output_digest: [3; 32],
            verification: verification.clone(),
        },
        LedgerEvent::PodHypothesisCommitted {
            request_id: RequestId::from("request"),
            session_id: SessionId::from("session"),
            scope_id: ScopeId::from("scope"),
            branch_id: "branch".into(),
            pod_id: PodId::from("pod"),
            manifest_digest: [1; 32],
            artifact_digest: [2; 32],
            generation: Generation(1),
            revision: Revision(1),
            output_type: TypeId::from("text"),
            output_digest: [3; 32],
            payload: b"hypothesis".to_vec(),
            provenance: (0..provenance)
                .map(|n| (format!("evidence-{n}"), Some("note".into())))
                .collect(),
            dependencies: vec![[4; 32]; dependencies],
            confidence_bits: 1.0_f32.to_bits(),
            latency_millis: 7,
            verification,
        },
    ]
}

fn valid_attestation() -> Attestation {
    Attestation {
        required: VerificationLevel::Deterministic,
        level: VerificationLevel::Deterministic,
        verifiers: vec!["verifier".into()],
        findings: vec![],
    }
}

#[test]
fn both_pod_event_kinds_reject_each_invalid_attestation_boundary() {
    use ptr_ledger::{check_encodable, MAX_ATTESTATION_FINDINGS, MAX_ATTESTATION_VERIFIERS};
    for (verification, code) in [
        (
            Attestation {
                verifiers: vec![],
                ..valid_attestation()
            },
            "PTR_LEDGER_ATTESTATION_LIMIT",
        ),
        (
            Attestation {
                verifiers: vec!["v".into(); MAX_ATTESTATION_VERIFIERS + 1],
                ..valid_attestation()
            },
            "PTR_LEDGER_ATTESTATION_LIMIT",
        ),
        (
            Attestation {
                findings: vec!["f".into(); MAX_ATTESTATION_FINDINGS + 1],
                ..valid_attestation()
            },
            "PTR_LEDGER_ATTESTATION_LIMIT",
        ),
        (
            Attestation {
                required: VerificationLevel::Unverified,
                ..valid_attestation()
            },
            "PTR_LEDGER_ATTESTATION_REQUIREMENT",
        ),
    ] {
        for event in pod_events(verification, 0, 0) {
            let error = check_encodable(&event).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert_eq!(error.to_string(), code);
        }
    }
}

#[test]
fn rejected_pod_counts_leave_the_durable_log_unchanged_and_boundary_events_reopen() {
    use ptr_ledger::{
        MAX_ATTESTATION_FINDINGS, MAX_ATTESTATION_VERIFIERS, MAX_HYPOTHESIS_DEPENDENCIES,
        MAX_HYPOTHESIS_PROVENANCE,
    };
    let path = std::env::temp_dir().join(format!(
        "ptr-pod-bounds-{}-{}.ledger",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut ledger = FileLedger::open(&path).unwrap();
    let at_limit = Attestation {
        verifiers: vec!["v".into(); MAX_ATTESTATION_VERIFIERS],
        findings: vec!["f".into(); MAX_ATTESTATION_FINDINGS],
        ..valid_attestation()
    };
    let expected = pod_events(
        at_limit,
        MAX_HYPOTHESIS_PROVENANCE,
        MAX_HYPOTHESIS_DEPENDENCIES,
    );
    ledger.append_durable(expected[0].clone()).unwrap();
    // The open ledger holds a byte-range lock that Windows enforces against
    // reads, so the length is compared here; the contents are read back after
    // the ledger is dropped.
    let before = std::fs::metadata(&path).unwrap().len();
    for (provenance, dependencies) in [
        (MAX_HYPOTHESIS_PROVENANCE + 1, 0),
        (0, MAX_HYPOTHESIS_DEPENDENCIES + 1),
    ] {
        let event = pod_events(valid_attestation(), provenance, dependencies)[1].clone();
        let error = ledger.append_durable(event).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "PTR_LEDGER_HYPOTHESIS_LIMIT");
        assert_eq!(ledger.events().len(), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
    }
    assert_eq!(
        ledger.append_durable(expected[1].clone()).unwrap(),
        ptr_types::CommitIndex(2)
    );
    drop(ledger);
    let reopened = FileLedger::open(&path).unwrap();
    assert_eq!(
        reopened
            .events()
            .iter()
            .map(|entry| entry.event.clone())
            .collect::<Vec<_>>(),
        expected
    );
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

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
