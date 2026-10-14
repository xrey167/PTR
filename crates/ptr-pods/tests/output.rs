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

/// Issue #141: version 1 carried no count for the dependencies or the provenance entries, so a
/// 32-byte dependency `u64le(23) || 23 bytes || 0x00` read as a provenance entry with source
/// those 23 bytes and no note. Both outputs below encode to the same version-1 bytes.
#[test]
fn a_dependency_shaped_like_a_provenance_entry_does_not_alias() {
    let source = "abcdefghijklmnopqrstuvw";
    assert_eq!(source.len(), 23);
    let mut forged = [0u8; 32];
    forged[..8].copy_from_slice(&23u64.to_le_bytes());
    forged[8..31].copy_from_slice(source.as_bytes());
    forged[31] = 0;

    let base = output(PodOutputKind::Observation, b"same");
    let mut as_dependency = base.clone();
    as_dependency.dependencies = vec![forged];
    as_dependency.provenance = Vec::new();
    let mut as_provenance = base;
    as_provenance.dependencies = Vec::new();
    as_provenance.provenance = vec![ProvenanceRef {
        source: ptr_types::EvidenceId::from(source),
        note: None,
    }];

    assert_ne!(as_dependency, as_provenance);
    // The ambiguity is real under version 1 ...
    assert_eq!(
        as_dependency.legacy_v1_digest(),
        as_provenance.legacy_v1_digest()
    );
    // ... and gone under version 2.
    assert_ne!(as_dependency.digest(), as_provenance.digest());

    // With a provenance entry on both sides (as `validate` requires) the same shift aliases.
    let kept = ProvenanceRef {
        source: ptr_types::EvidenceId::from("zzz"),
        note: None,
    };
    as_dependency.provenance = vec![kept.clone()];
    as_provenance.provenance.push(kept);
    assert_eq!(
        as_dependency.legacy_v1_digest(),
        as_provenance.legacy_v1_digest()
    );
    assert_ne!(as_dependency.digest(), as_provenance.digest());
}

#[test]
fn the_versioned_digest_differs_from_the_legacy_one() {
    let first = output(PodOutputKind::Observation, b"same");
    assert_ne!(first.digest(), first.legacy_v1_digest());
}
