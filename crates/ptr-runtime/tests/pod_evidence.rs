use ptr_config::PtrConfig;
use ptr_pods::{
    Ed25519EvidenceSigner, Ed25519EvidenceVerifier, EvidenceSigner, PodEvidenceBundle,
    PodTurnEvent, PodTurnKind,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{
    Generation, PodAddress, PodRevisionAddress, ProjectId, RequestId, Revision, ScopeId, SessionId,
    StateId, TraceId, TypeId,
};

fn event() -> PodTurnEvent {
    PodTurnEvent {
        kind: PodTurnKind::Observation,
        sequence: 1,
        session_id: SessionId::from("session"),
        scope_id: ScopeId::from("scope"),
        turn_id: 1,
        request_id: RequestId::from("request"),
        pod: PodRevisionAddress {
            address: PodAddress::new(ProjectId::from("project"), "ns".into(), "pod".into())
                .unwrap(),
            semantic_revision: [3; 32],
            generation: Generation(1),
        },
        generation: Generation(1),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        input_type: Some(TypeId::from("Input")),
        output_type: Some(TypeId::from("Output")),
        state_before: Some(StateId::from("before")),
        state_after: Some(StateId::from("after")),
        revision: Revision(4),
    }
}

fn sealed_bundle() -> PodEvidenceBundle {
    let mut bundle = PodEvidenceBundle::new(
        SessionId::from("session"),
        TraceId::from("trace"),
        [1; 32],
        [2; 32],
        Generation(1),
        Revision(4),
    )
    .unwrap();
    bundle.append_event(event()).unwrap();
    bundle.seal().unwrap();
    bundle
}

#[test]
fn runtime_commits_exactly_the_verified_evidence_bundle() {
    let bundle = sealed_bundle();
    let encoded = bundle.encode_canonical().unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();

    runtime.commit_pod_evidence(&bundle).unwrap();

    let committed = runtime.committed_events().last().unwrap();
    match &committed.event {
        ptr_ledger::LedgerEvent::PodEvidenceCommitted {
            session_id,
            trace_id,
            bundle_digest,
            bundle: stored,
            ..
        } => {
            assert_eq!(session_id, &SessionId::from("session"));
            assert_eq!(trace_id, &TraceId::from("trace"));
            assert_eq!(bundle_digest, &bundle.bundle_digest);
            assert_eq!(stored, &encoded);
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn runtime_refuses_tampered_evidence_before_append() {
    let mut bundle = sealed_bundle();
    bundle.events[0].sequence = 2;
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();

    assert!(matches!(
        runtime.commit_pod_evidence(&bundle),
        Err(ptr_runtime::RuntimeError::PodEvidence(_))
    ));
    assert!(runtime.committed_events().is_empty());
}

#[test]
fn signed_runtime_path_requires_and_accepts_a_valid_signature() {
    let mut bundle = sealed_bundle();
    let signer = Ed25519EvidenceSigner::from_bytes(&[11; 32]);
    bundle.sign_with(&signer).unwrap();
    let verifier = Ed25519EvidenceVerifier::from_bytes(&signer.public_key()).unwrap();
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();

    runtime
        .commit_signed_pod_evidence(&bundle, &verifier)
        .unwrap();
    assert_eq!(runtime.committed_events().len(), 1);

    let wrong_verifier = Ed25519EvidenceVerifier::from_bytes(&[12; 32]).unwrap();
    let mut another_runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(another_runtime
        .commit_signed_pod_evidence(&bundle, &wrong_verifier)
        .is_err());
    assert!(another_runtime.committed_events().is_empty());
}

#[test]
fn raw_ledger_evidence_is_validated_before_it_can_be_committed() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let result = runtime.commit(ptr_ledger::LedgerEvent::PodEvidenceCommitted {
        session_id: SessionId::from("session"),
        trace_id: TraceId::from("trace"),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        generation: Generation(1),
        revision: Revision(4),
        bundle_digest: [9; 32],
        bundle: b"not-a-canonical-evidence-bundle".to_vec(),
    });
    assert!(matches!(
        result,
        Err(ptr_runtime::RuntimeError::PodEvidence(_))
    ));
    assert!(runtime.committed_events().is_empty());
}
