//! Compacted materialized snapshots: does state at a floor plus the retained
//! journal reconstruct exactly what a full replay produces, and does a revocation
//! below the floor still deny?
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_ledger::LedgerEvent;
use ptr_runtime::compacted::CompactedAnchor;
use ptr_runtime::PtrRuntime;
use ptr_security::{AuthorizationDecision, AuthorizationDenial};
use ptr_semdb::{SemanticDelta, SemanticPayload};
use ptr_types::{CapabilityId, CommitIndex, Effect, Generation, Revision, TypeId};

/// A runtime exercising semantic values with dependencies, payload bytes, capsule
/// lifecycle, supersession, constraints, procedures and a revocation.
fn fixture() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let mut delta = SemanticDelta::default();
    delta
        .upserts
        .insert("source".into(), "München\n東京\0".into());
    delta.upserts.insert("derived".into(), "cached".into());
    delta.upserts.insert(
        "payload".into(),
        SemanticPayload {
            type_id: "Bytes".into(),
            source: "pod:a".into(),
            bytes: vec![0, 255, 128, 3],
        }
        .into(),
    );
    delta
        .dependencies
        .insert("derived".into(), ["source".into()].into());
    runtime.apply_semantic_delta(Revision(0), delta).unwrap();

    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:kept".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "capsule:moved".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: "capsule:moved".into(),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::HardConstraintCommitted {
            key: "budget".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::ProcedurePromoted {
            id: "deploy".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:kept".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
}

/// Records committed after a snapshot's floor, which the log retains.
fn append_above_floor(runtime: &mut PtrRuntime) {
    let mut delta = SemanticDelta::default();
    delta.removals.insert("source".into());
    runtime
        .apply_semantic_delta(runtime.revision(), delta)
        .unwrap();
    runtime
        .commit(LedgerEvent::ProcedureRevoked {
            id: "deploy".into(),
            generation: Generation(1),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::VerifierAttested {
            subject: "capsule:moved".into(),
            passed: true,
        })
        .unwrap();
}

/// Everything two runtimes must agree on for reconstruction to be exact.
fn assert_equivalent(reference: &PtrRuntime, restored: &PtrRuntime) {
    assert_eq!(reference.revision(), restored.revision());
    assert_eq!(
        reference.materialized_state().last_applied,
        restored.materialized_state().last_applied
    );
    assert_eq!(
        reference.materialized_state().values,
        restored.materialized_state().values
    );

    // Semantic ground and dependency sets, over the union of both key sets so a
    // key present in only one of them cannot hide.
    let left = reference.snapshot();
    let right = restored.snapshot();
    let mut keys: Vec<&str> = left.keys().chain(right.keys()).collect();
    keys.sort_unstable();
    keys.dedup();
    assert!(!keys.is_empty());
    for key in keys {
        assert_eq!(left.value(key), right.value(key), "value {key}");
        assert_eq!(
            left.inputs(key).collect::<Vec<_>>(),
            right.inputs(key).collect::<Vec<_>>(),
            "inputs {key}"
        );
    }

    for target in [
        "capsule:kept",
        "capsule:moved",
        "constraint:budget",
        "procedure:deploy",
        "capsule:absent",
    ] {
        assert_eq!(
            reference.live_generation(target),
            restored.live_generation(target),
            "live generation {target}"
        );
    }
}

fn action_on(target: &str, generation: Generation, revision: Revision) -> ActionIr {
    ActionIr {
        operation: "write".into(),
        target: target.into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation,
        revision,
        payload: b"payload".to_vec(),
    }
}

/// Re-seal mutated bytes so validation cannot pass merely because the digest
/// still matches: this is the attacker who can recompute it.
fn reseal(bytes: &mut [u8], floor: LogAnchor, revision: Revision) -> CompactedAnchor {
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    CompactedAnchor {
        revision,
        floor,
        digest,
    }
}

#[test]
fn compacted_reconstruction_equals_a_full_replay() {
    let mut original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let floor = snapshot.anchor().floor;
    assert_eq!(snapshot.covers(), floor.index);

    append_above_floor(&mut original);
    let all_events = original.committed_events().to_vec();
    let above_floor = &all_events[floor.index.0 as usize..];
    assert_eq!(above_floor.len(), 3);

    let reference = PtrRuntime::replay(PtrConfig::default(), &all_events).unwrap();
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        above_floor,
    )
    .unwrap();

    assert_equivalent(&reference, &restored);
    // The retained records keep the indices they were committed at rather than
    // being renumbered from 1.
    assert_eq!(restored.committed_events(), above_floor);
    assert_eq!(
        restored.materialized_state().last_applied,
        all_events.len() as u64
    );
}

#[test]
fn restoring_with_no_retained_journal_reproduces_the_floor_exactly() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_equivalent(&original, &restored);
    assert!(restored.committed_events().is_empty());
}

#[test]
fn a_revocation_below_the_floor_still_denies_after_compaction() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();

    // The record that revoked this generation is below the floor and would be
    // discarded by a cutover. Without it surviving in the snapshot the live
    // generation would still read as 1 and this action would be allowed, which is
    // exactly the resurrection global invariant 3 forbids.
    let revoked = action_on("capsule:kept", Generation(1), restored.revision());
    assert!(matches!(
        restored.authorization_decision(&revoked),
        AuthorizationDecision::Deny(AuthorizationDenial::StaleGeneration { .. })
    ));
    assert!(restored.authorize_action(&revoked).is_err());

    // A superseded capsule's old generation is refused for the same reason, and
    // its current generation survived the floor as well.
    assert_eq!(
        restored.live_generation("capsule:moved"),
        Some(Generation(2))
    );
    let superseded = action_on("capsule:moved", Generation(1), restored.revision());
    assert!(restored.authorize_action(&superseded).is_err());
}

#[test]
fn every_single_bit_mutation_of_a_compacted_snapshot_is_rejected() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let trusted = snapshot.anchor();
    let reference = snapshot.bytes().to_vec();

    for offset in 0..reference.len() {
        for bit in 0..8 {
            let mut bytes = reference.clone();
            bytes[offset] ^= 1 << bit;
            assert!(
                PtrRuntime::restore_compacted(PtrConfig::default(), &bytes, trusted, &[]).is_err(),
                "offset {offset} bit {bit} accepted"
            );
        }
    }
    assert!(PtrRuntime::restore_compacted(PtrConfig::default(), &reference, trusted, &[]).is_ok());
}

#[test]
fn framing_length_and_trusted_identity_are_all_required() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let trusted = snapshot.anchor();
    let reference = snapshot.bytes().to_vec();
    let code = |bytes: &[u8], anchor: CompactedAnchor| {
        PtrRuntime::restore_compacted(PtrConfig::default(), bytes, anchor, &[])
            .err()
            .map(|error| format!("{error:?}"))
            .expect("rejection")
    };

    // Truncation and extension.
    for length in [0usize, 79, reference.len() - 1] {
        assert!(code(&reference[..length], trusted).contains("Compacted"));
    }
    let mut extended = reference.clone();
    extended.push(0);
    assert!(code(&extended, trusted).contains("Compacted"));

    // A resealed artifact with a reserved bit set is still refused, so the
    // reserved field is a real constraint and not just digest-protected padding.
    let mut reserved = reference.clone();
    reserved[72] = 1;
    let anchor = reseal(&mut reserved, trusted.floor, trusted.revision);
    assert!(code(&reserved, anchor).contains("ReservedField"));

    // Section lengths that do not add up to the body, resealed.
    let mut lengths = reference.clone();
    lengths[56] = lengths[56].wrapping_add(1);
    let anchor = reseal(&mut lengths, trusted.floor, trusted.revision);
    assert!(code(&lengths, anchor).contains("LengthMismatch"));

    // Wrong magic.
    let mut magic = reference.clone();
    magic[..8].copy_from_slice(b"PTRCS002");
    let anchor = reseal(&mut magic, trusted.floor, trusted.revision);
    assert!(code(&magic, anchor).contains("UnsupportedVersion"));

    // Each trusted field must match: revision, floor index, floor digest, digest.
    for wrong in [
        CompactedAnchor {
            revision: Revision(trusted.revision.0 + 1),
            ..trusted
        },
        CompactedAnchor {
            floor: LogAnchor {
                index: CommitIndex(trusted.floor.index.0 + 1),
                digest: trusted.floor.digest,
            },
            ..trusted
        },
        CompactedAnchor {
            floor: LogAnchor {
                index: trusted.floor.index,
                digest: [0xab; 32],
            },
            ..trusted
        },
        CompactedAnchor {
            digest: [0xcd; 32],
            ..trusted
        },
    ] {
        assert!(code(&reference, wrong).contains("AnchorMismatch"));
    }
}

#[test]
fn a_noncanonical_lifecycle_section_is_rejected_even_when_resealed() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let trusted = snapshot.anchor();
    let reference = snapshot.bytes().to_vec();
    let semantic_len = u64::from_le_bytes(reference[56..64].try_into().unwrap()) as usize;
    let lifecycle_start = 80 + semantic_len;

    // The first materialized section entry begins after the 8-byte lifecycle magic
    // and the 4-byte count. Rewriting a key so the section is no longer strictly
    // ascending must fail on ordering, not on the digest.
    let first_key_len_at = lifecycle_start + 12;
    let key_len = u32::from_le_bytes(
        reference[first_key_len_at..first_key_len_at + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    let mut reordered = reference.clone();
    let key_at = first_key_len_at + 4;
    // "zzz…" sorts after every real key, so the remaining entries break ordering.
    for byte in &mut reordered[key_at..key_at + key_len] {
        *byte = b'z';
    }
    let anchor = reseal(&mut reordered, trusted.floor, trusted.revision);
    let error = PtrRuntime::restore_compacted(PtrConfig::default(), &reordered, anchor, &[])
        .err()
        .map(|error| format!("{error:?}"))
        .expect("rejection");
    assert!(error.contains("NoncanonicalSection"), "{error}");

    // A wrong lifecycle magic is a version failure, also independent of the digest.
    let mut magic = reference.clone();
    magic[lifecycle_start..lifecycle_start + 8].copy_from_slice(b"PTRLC002");
    let anchor = reseal(&mut magic, trusted.floor, trusted.revision);
    let error = PtrRuntime::restore_compacted(PtrConfig::default(), &magic, anchor, &[])
        .err()
        .map(|error| format!("{error:?}"))
        .expect("rejection");
    assert!(error.contains("UnsupportedVersion"), "{error}");
}

#[test]
fn retained_records_must_continue_above_the_floor() {
    let mut original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let floor = snapshot.anchor().floor;
    append_above_floor(&mut original);
    let all_events = original.committed_events().to_vec();

    // A record at or below the floor is already accounted for by the snapshot;
    // replaying it would apply the same commit twice.
    for start in [0usize, 1, floor.index.0 as usize - 1] {
        let error = PtrRuntime::restore_compacted(
            PtrConfig::default(),
            snapshot.bytes(),
            snapshot.anchor(),
            &all_events[start..],
        )
        .err()
        .map(|error| format!("{error:?}"))
        .expect("rejection");
        assert!(error.contains("JournalNotAboveFloor"), "{error}");
    }

    // A gap above the floor is refused by the ordinary replay index check.
    let error = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &all_events[floor.index.0 as usize + 1..],
    )
    .err()
    .map(|error| format!("{error:?}"))
    .expect("rejection");
    assert!(error.contains("ReplayIndexMismatch"), "{error}");
}

#[test]
fn a_restored_runtime_rejects_semantic_work_against_a_stale_base_revision() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let mut restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();

    // Restoring installs an exact revision, so revision isolation still applies:
    // a delta based on an earlier revision is refused.
    let mut delta = SemanticDelta::default();
    delta.upserts.insert("fresh".into(), "value".into());
    assert!(restored
        .apply_semantic_delta(Revision(0), delta.clone())
        .is_err());
    assert!(restored
        .apply_semantic_delta(restored.revision(), delta)
        .is_ok());

    // A derived value still requires its declared input to be supplied.
    let mut stale = SemanticDelta::default();
    stale.upserts.insert("derived".into(), "recomputed".into());
    let before = restored.revision();
    assert!(restored.apply_semantic_delta(before, stale).is_ok());
    assert!(restored.revision().0 > before.0);
}

#[test]
fn an_exported_snapshot_carries_its_own_coverage() {
    let mut original = fixture();
    let first = original.export_compacted_snapshot().unwrap();
    let covered = original.committed_events().len() as u64;
    assert_eq!(first.covers(), CommitIndex(covered));
    assert_eq!(first.anchor().revision, original.revision());

    append_above_floor(&mut original);
    let second = original.export_compacted_snapshot().unwrap();
    assert!(second.covers().0 > first.covers().0);
    assert_ne!(second.anchor().digest, first.anchor().digest);
    // Each snapshot only ever reports the position it actually holds.
    assert_eq!(
        second.covers(),
        CommitIndex(original.committed_events().len() as u64)
    );
}
