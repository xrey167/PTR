use ptr_pods::{PodOutput, PodOutputKind};
use ptr_protocol::TypedPayload;
use ptr_types::{Generation, ProvenanceRef, Revision, TypeId};

fn output(kind: PodOutputKind, bytes: &[u8]) -> PodOutput {
    PodOutput {
        kind,
        payload: TypedPayload {
            type_id: TypeId::from("text"),
            bytes: bytes.to_vec(),
        },
        generation: Generation(1),
        manifest_digest: [1; 32],
        artifact_digest: [2; 32],
        provenance: vec![ProvenanceRef {
            source: ptr_types::EvidenceId::from("source"),
            note: Some("note".into()),
        }],
        dependencies: vec![[3; 32], [4; 32]],
        revision: Revision(2),
        verified: false,
    }
}

#[test]
fn output_digest_is_semantic_and_order_independent_for_sets() {
    let first = output(PodOutputKind::Observation, b"same");
    let mut second = first.clone();
    second.dependencies.reverse();
    second.provenance.reverse();
    assert_eq!(first.digest(), second.digest());

    let mut changed = first.clone();
    changed.payload.bytes = b"changed".to_vec();
    assert_ne!(first.digest(), changed.digest());
}

#[test]
fn output_digest_changes_for_lineage_and_kind() {
    let first = output(PodOutputKind::Observation, b"same");
    let mut changed_dependency = first.clone();
    changed_dependency.dependencies[0] = [9; 32];
    assert_ne!(first.digest(), changed_dependency.digest());

    let mut changed_kind = first;
    changed_kind.kind = PodOutputKind::Hypothesis;
    assert_ne!(
        changed_kind.digest(),
        output(PodOutputKind::Observation, b"same").digest()
    );
}

#[test]
fn output_validation_keeps_promotion_fail_closed() {
    let mut output = output(PodOutputKind::StateDelta, b"state");
    assert!(output.validate(&TypeId::from("text"), true).is_err());
    output.verified = true;
    assert!(output.validate(&TypeId::from("text"), true).is_ok());
    assert!(output.validate(&TypeId::from("other"), true).is_err());
}
