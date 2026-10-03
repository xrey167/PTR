use ed25519_dalek::SigningKey;
use ptr_pods::{
    Ed25519EvidenceSigner, Ed25519EvidenceVerifier, EvidenceError, EvidenceSigner,
    PodEvidenceBundle, PodOutput, PodOutputKind, PodTurnEvent,
};
use ptr_protocol::TypedPayload;
use ptr_types::{
    Generation, PodAddress, PodRevisionAddress, ProjectId, RequestId, Revision, ScopeId, SessionId,
    StateId, TraceId, TypeId,
};

fn event(sequence: u64) -> PodTurnEvent {
    PodTurnEvent {
        kind: ptr_pods::PodTurnKind::Observation,
        sequence,
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

fn bundle() -> PodEvidenceBundle {
    PodEvidenceBundle::new(
        SessionId::from("session"),
        TraceId::from("trace"),
        [1; 32],
        [2; 32],
        Generation(1),
        Revision(4),
    )
    .unwrap()
}

fn output() -> PodOutput {
    PodOutput {
        kind: PodOutputKind::Observation,
        payload: TypedPayload {
            type_id: TypeId::from("Output"),
            bytes: b"ok".to_vec(),
        },
        generation: Generation(1),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        provenance: vec![],
        dependencies: vec![],
        revision: Revision(4),
        verified: false,
    }
}

#[test]
fn evidence_bundle_seals_and_replays_deterministically() {
    let mut left = bundle();
    left.append_event(event(1)).unwrap();
    left.append_output(output()).unwrap();
    let digest = left.seal().unwrap();
    assert_eq!(left.replay().unwrap().events.len(), 1);

    let mut right = bundle();
    right.append_event(event(1)).unwrap();
    right.append_output(left.outputs[0].clone()).unwrap();
    assert_eq!(right.seal().unwrap(), digest);
}

#[test]
fn canonical_round_trip_preserves_interleaved_chain_order() {
    let mut original = bundle();
    original.append_event(event(1)).unwrap();
    original.append_output(output()).unwrap();
    original.append_event(event(2)).unwrap();
    original.seal().unwrap();

    let encoded = original.encode_canonical().unwrap();
    let decoded = PodEvidenceBundle::decode_canonical(&encoded).unwrap();
    assert_eq!(decoded, original);
    assert_eq!(decoded.encode_canonical().unwrap(), encoded);
}

#[test]
fn tampering_is_detected_and_sequence_is_strict() {
    let mut bundle = bundle();
    assert_eq!(
        bundle.append_event(event(2)),
        Err(EvidenceError::InvalidSequence)
    );
    bundle.append_event(event(1)).unwrap();
    bundle.seal().unwrap();
    bundle.events[0].revision = Revision(5);
    assert_eq!(bundle.verify(), Err(EvidenceError::DigestMismatch));
}

#[test]
fn ed25519_signature_is_bound_to_the_verified_bundle_digest() {
    let mut bundle = bundle();
    bundle.append_event(event(1)).unwrap();
    let key = SigningKey::from_bytes(&[7; 32]);
    bundle.sign(&key).unwrap();
    assert!(bundle.verify_signature().is_ok());

    bundle.events[0].revision = Revision(5);
    assert_eq!(
        bundle.verify_signature(),
        Err(EvidenceError::DigestMismatch)
    );
}

#[test]
fn provider_neutral_signer_and_verifier_round_trip() {
    let mut bundle = bundle();
    bundle.append_event(event(1)).unwrap();
    let signer = Ed25519EvidenceSigner::from_bytes(&[9; 32]);
    bundle.sign_with(&signer).unwrap();
    let verifier = Ed25519EvidenceVerifier::from_bytes(&signer.public_key()).unwrap();
    assert!(bundle.verify_with(&verifier).is_ok());
}
