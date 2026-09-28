//! Compacted materialized snapshots: does state at a floor plus the retained
//! journal reconstruct exactly what a full replay produces, and does a revocation
//! below the floor still deny?
use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::integrity::{self, LogAnchor};
use ptr_ledger::LedgerEvent;
use ptr_runtime::compacted::{CompactedAnchor, CompactedError};
use ptr_runtime::{PtrRuntime, RuntimeError};
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
fn an_empty_runtime_round_trips_at_the_empty_floor() {
    let original = PtrRuntime::new(PtrConfig::default()).unwrap();
    let snapshot = original.export_compacted_snapshot().unwrap();
    assert_eq!(snapshot.covers(), CommitIndex(0));
    assert_eq!(snapshot.anchor().floor, LogAnchor::empty());
    assert_eq!(snapshot.anchor().revision, Revision(0));

    let restored = PtrRuntime::restore_compacted(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    assert_eq!(restored.revision(), Revision(0));
    assert_eq!(restored.materialized_state().last_applied, 0);
    assert!(restored.materialized_state().values.is_empty());
    assert!(restored.snapshot().keys().next().is_none());
    assert!(restored.committed_events().is_empty());
}

#[test]
fn a_runtime_restored_at_the_index_ceiling_refuses_commits_atomically() {
    let original = PtrRuntime::new(PtrConfig::default()).unwrap();
    let snapshot = original.export_compacted_snapshot().unwrap();
    let mut bytes = snapshot.bytes().to_vec();
    bytes[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let floor = LogAnchor {
        index: CommitIndex(u64::MAX),
        digest: snapshot.anchor().floor.digest,
    };
    let anchor = reseal(&mut bytes, floor, Revision(0));
    let mut restored =
        PtrRuntime::restore_compacted(PtrConfig::default(), &bytes, anchor, &[]).unwrap();

    let event = LedgerEvent::VerifierAttested {
        subject: "capsule:a".into(),
        passed: true,
    };
    assert_eq!(
        restored.commit(event.clone()),
        Err(RuntimeError::Ledger(
            "PTR_LEDGER_INDEX_EXHAUSTED".to_owned()
        ))
    );
    assert!(restored.committed_events().is_empty());
    assert_eq!(restored.materialized_state().last_applied, u64::MAX);

    // A failed append happens after the runtime begins a commit, so the outcome
    // is conservatively fenced. Retrying cannot invent a record or an index.
    assert_eq!(restored.commit(event), Err(RuntimeError::ExecutionFenced));
    assert!(restored.committed_events().is_empty());
    assert_eq!(restored.materialized_state().last_applied, u64::MAX);
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

    // Bytes 72..80 were a reserved field in PTRCS001 and are the execution
    // section's length from PTRCS002 on. Claiming a longer section than the body holds
    // is refused after resealing, so the field is a real constraint rather than
    // digest-protected padding — the same property the reserved check used to
    // give, now carried by a field that means something.
    let mut execution_len = reference.clone();
    execution_len[72] = execution_len[72].wrapping_add(1);
    let anchor = reseal(&mut execution_len, trusted.floor, trusted.revision);
    assert!(code(&execution_len, anchor).contains("LengthMismatch"));

    // Section lengths that do not add up to the body, resealed.
    let mut lengths = reference.clone();
    lengths[56] = lengths[56].wrapping_add(1);
    let anchor = reseal(&mut lengths, trusted.floor, trusted.revision);
    assert!(code(&lengths, anchor).contains("LengthMismatch"));

    // Wrong magic. PTRCS004 is this build's format and PTRCS003 the earlier one
    // it still reads, so the unknown version has to be a later layout this
    // build has no rules for.
    let mut magic = reference.clone();
    magic[..8].copy_from_slice(b"PTRCS005");
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
    magic[lifecycle_start..lifecycle_start + 8].copy_from_slice(b"PTRLC003");
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

/// Where the lifecycle section of a snapshot starts: after the 80-byte header
/// and the semantic section.
fn lifecycle_start(bytes: &[u8]) -> usize {
    80 + u64::from_le_bytes(bytes[56..64].try_into().unwrap()) as usize
}

/// A copy of `bytes` whose outer and lifecycle magics are `outer` and
/// `lifecycle`, resealed.
fn with_magics(
    bytes: &[u8],
    trusted: CompactedAnchor,
    outer: &[u8; 8],
    lifecycle: &[u8; 8],
) -> (Vec<u8>, CompactedAnchor) {
    let mut bytes = bytes.to_vec();
    let start = lifecycle_start(&bytes);
    bytes[..8].copy_from_slice(outer);
    bytes[start..start + 8].copy_from_slice(lifecycle);
    let anchor = reseal(&mut bytes, trusted.floor, trusted.revision);
    (bytes, anchor)
}

/// A copy of `bytes` whose lifecycle section also holds `key = value` in
/// the materialized map, in key order, with the header's lifecycle length
/// and the seal updated.
fn with_materialized(
    bytes: &[u8],
    trusted: CompactedAnchor,
    key: &str,
    value: &str,
) -> (Vec<u8>, CompactedAnchor) {
    let start = lifecycle_start(bytes);
    let lifecycle_len = u64::from_le_bytes(bytes[64..72].try_into().unwrap()) as usize;
    let section = &bytes[start..start + lifecycle_len];
    let count = u32::from_le_bytes(section[8..12].try_into().unwrap()) as usize;
    let mut offset = 12;
    let mut entries = Vec::new();
    for _ in 0..count {
        let mut pair = Vec::new();
        for _ in 0..2 {
            let length =
                u32::from_le_bytes(section[offset..offset + 4].try_into().unwrap()) as usize;
            pair.push(section[offset + 4..offset + 4 + length].to_vec());
            offset += 4 + length;
        }
        entries.push((pair[0].clone(), pair[1].clone()));
    }
    entries.push((key.as_bytes().to_vec(), value.as_bytes().to_vec()));
    entries.sort();
    let mut rebuilt = section[..8].to_vec();
    rebuilt.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, value) in &entries {
        for text in [key, value] {
            rebuilt.extend_from_slice(&(text.len() as u32).to_le_bytes());
            rebuilt.extend_from_slice(text);
        }
    }
    rebuilt.extend_from_slice(&section[offset..]);
    let mut out = bytes[..start].to_vec();
    out[64..72].copy_from_slice(&(rebuilt.len() as u64).to_le_bytes());
    out.extend_from_slice(&rebuilt);
    out.extend_from_slice(&bytes[start + lifecycle_len..]);
    let anchor = reseal(&mut out, trusted.floor, trusted.revision);
    (out, anchor)
}

fn restore_error(bytes: &[u8], anchor: CompactedAnchor) -> Option<RuntimeError> {
    PtrRuntime::restore_compacted(PtrConfig::default(), bytes, anchor, &[]).err()
}

#[test]
fn a_snapshot_of_the_earlier_layout_restores_as_unattested() {
    // PTRCS003 with PTRLC001, as the build before attributed records wrote
    // it: the same sections under the earlier magics.
    let mut original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    assert_eq!(&snapshot.bytes()[..8], b"PTRCS004");
    let start = lifecycle_start(snapshot.bytes());
    assert_eq!(&snapshot.bytes()[start..start + 8], b"PTRLC002");
    let (earlier, anchor) = with_magics(
        snapshot.bytes(),
        snapshot.anchor(),
        b"PTRCS003",
        b"PTRLC001",
    );

    append_above_floor(&mut original);
    let all_events = original.committed_events().to_vec();
    let above_floor = &all_events[snapshot.anchor().floor.index.0 as usize..];
    let reference = PtrRuntime::replay(PtrConfig::default(), &all_events).unwrap();
    let restored =
        PtrRuntime::restore_compacted(PtrConfig::default(), &earlier, anchor, above_floor).unwrap();
    assert_equivalent(&reference, &restored);
    let values = &restored.materialized_state().values;
    assert!(!values.contains_key(ptr_state::ATTESTED_MARKER));
    assert!(!values
        .keys()
        .any(|key| key.starts_with(ptr_state::MERGED_BRANCH_PREFIX)));
}

#[test]
fn an_earlier_lifecycle_section_carrying_attestation_keys_is_refused() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    for (key, value) in [
        (ptr_state::ATTESTED_MARKER, "1"),
        ("branch-merge:2:b1", "7"),
    ] {
        assert!(
            key == ptr_state::ATTESTED_MARKER || key.starts_with(ptr_state::MERGED_BRANCH_PREFIX)
        );
        let (current, anchor) = with_materialized(snapshot.bytes(), snapshot.anchor(), key, value);
        // In this build's layout the key is state like any other.
        let restored =
            PtrRuntime::restore_compacted(PtrConfig::default(), &current, anchor, &[]).unwrap();
        assert_eq!(
            restored
                .materialized_state()
                .values
                .get(key)
                .map(String::as_str),
            Some(value)
        );
        // In the layout from before attributed records no history could have
        // set it, so the snapshot is refused by version.
        let (earlier, anchor) = with_magics(&current, anchor, b"PTRCS003", b"PTRLC001");
        assert_eq!(
            restore_error(&earlier, anchor),
            Some(RuntimeError::Compacted(CompactedError::UnsupportedVersion)),
            "{key}"
        );
    }
    // A key that only shares the marker's namespace is not refused.
    let (earlier, anchor) = with_magics(
        snapshot.bytes(),
        snapshot.anchor(),
        b"PTRCS003",
        b"PTRLC001",
    );
    let (earlier, anchor) = with_materialized(&earlier, anchor, "semdb:attested-not", "1");
    assert_eq!(restore_error(&earlier, anchor), None);
}

#[test]
fn mismatched_outer_and_lifecycle_versions_are_refused() {
    let original = fixture();
    let snapshot = original.export_compacted_snapshot().unwrap();
    for (outer, lifecycle) in [(b"PTRCS004", b"PTRLC001"), (b"PTRCS003", b"PTRLC002")] {
        let (bytes, anchor) = with_magics(snapshot.bytes(), snapshot.anchor(), outer, lifecycle);
        assert_eq!(
            restore_error(&bytes, anchor),
            Some(RuntimeError::Compacted(CompactedError::UnsupportedVersion))
        );
    }
    // The two pairs this build reads both restore.
    for (outer, lifecycle) in [(b"PTRCS004", b"PTRLC002"), (b"PTRCS003", b"PTRLC001")] {
        let (bytes, anchor) = with_magics(snapshot.bytes(), snapshot.anchor(), outer, lifecycle);
        assert_eq!(restore_error(&bytes, anchor), None);
    }
}
