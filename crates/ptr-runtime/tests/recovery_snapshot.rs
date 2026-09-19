use ptr_config::PtrConfig;
use ptr_ledger::{
    integrity::{self, LogAnchor},
    LedgerEvent,
};
use ptr_runtime::{
    persistence::{SnapshotAnchor, MAX_SNAPSHOT_BYTES},
    PtrRuntime,
};
use ptr_semdb::{SemanticDelta, SemanticPayload, SemanticValue};
use ptr_types::{Generation, Revision};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-snapshot-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> PtrRuntime {
    let mut r = PtrRuntime::new(PtrConfig::default()).unwrap();
    let mut d = SemanticDelta::default();
    d.upserts.insert("source".into(), "München\n東京\0".into());
    d.upserts.insert("derived".into(), "cached".into());
    d.upserts.insert(
        "unrelated".into(),
        SemanticPayload {
            type_id: "Bytes".into(),
            source: "pod:a".into(),
            bytes: vec![0, 255, 128, 3],
        }
        .into(),
    );
    d.dependencies
        .insert("derived".into(), ["source".into()].into());
    r.apply_semantic_delta(Revision(0), d).unwrap();
    r.commit(LedgerEvent::CapsuleCommitted {
        project: "p".into(),
        capsule: "a".into(),
        generation: Generation(1),
    })
    .unwrap();
    r.commit(LedgerEvent::Revoked {
        subject: "a".into(),
        generation: Generation(1),
    })
    .unwrap();
    let mut d = SemanticDelta::default();
    d.removals.insert("source".into());
    r.apply_semantic_delta(r.revision(), d).unwrap();
    r
}
fn compare(left: &PtrRuntime, right: &PtrRuntime) {
    assert_eq!(left.revision(), right.revision());
    assert_eq!(left.committed_events(), right.committed_events());
    assert_eq!(
        left.materialized_state().values,
        right.materialized_state().values
    );
    assert_eq!(
        left.materialized_state().last_applied,
        right.materialized_state().last_applied
    );
    let a = left.snapshot();
    let b = right.snapshot();
    for key in ["source", "derived", "unrelated"] {
        assert_eq!(a.value(key), b.value(key));
        assert_eq!(
            a.inputs(key).collect::<Vec<_>>(),
            b.inputs(key).collect::<Vec<_>>()
        );
    }
    assert!(!right.snapshot_is_current(&a));
}

#[test]
fn snapshot_restores_exact_state_dependencies_tombstones_and_append_position() {
    let original = fixture();
    let snapshot = original.export_recovery_snapshot().unwrap();
    let mut restored = PtrRuntime::restore_recovery_snapshot(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
    )
    .unwrap();
    compare(&original, &restored);
    assert_eq!(
        restored.snapshot().inputs("derived").collect::<Vec<_>>(),
        vec!["source"]
    );
    let mut d = SemanticDelta::default();
    d.upserts.insert("derived".into(), "stale".into());
    assert!(restored
        .apply_semantic_delta(restored.revision(), d)
        .is_err());
    assert!(restored
        .commit(LedgerEvent::CapsuleCommitted {
            project: "p".into(),
            capsule: "a".into(),
            generation: Generation(1)
        })
        .is_err());
    let tmp = Temp::new();
    let mut disk = PtrRuntime::restore_durable_snapshot(
        PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        tmp.path("log"),
    )
    .unwrap();
    compare(&original, &disk);
    disk.ingest_text("new".into(), "next").unwrap();
    let next_anchor = disk.journal_anchor().unwrap();
    assert_eq!(next_anchor.index.0, snapshot.anchor().log.index.0 + 1);
    drop(disk);
    let reopened =
        PtrRuntime::open_durable_at(PtrConfig::default(), tmp.path("log"), next_anchor).unwrap();
    assert_eq!(reopened.revision(), Revision(3));
    assert_eq!(reopened.snapshot().get("request:new:raw"), Some("next"));
    drop(reopened);
    let error =
        PtrRuntime::open_durable_at(PtrConfig::default(), tmp.path("log"), snapshot.anchor().log)
            .err()
            .expect("an outdated anchor must fail after the lock is released");
    assert!(
        matches!(error, ptr_runtime::RuntimeError::Ledger(ref message)
        if message == "PTR_LOG_ANCHOR_MISMATCH")
    );
}

#[test]
fn all_snapshot_bit_mutations_and_truncations_fail_without_creating_destination() {
    let snapshot = fixture().export_recovery_snapshot().unwrap();
    for pos in 0..snapshot.bytes().len() {
        for bit in 0..8 {
            let mut bytes = snapshot.bytes().to_vec();
            bytes[pos] ^= 1 << bit;
            assert!(
                PtrRuntime::restore_recovery_snapshot(
                    PtrConfig::default(),
                    &bytes,
                    snapshot.anchor()
                )
                .is_err(),
                "{pos}/{bit}"
            );
        }
        assert!(PtrRuntime::restore_recovery_snapshot(
            PtrConfig::default(),
            &snapshot.bytes()[..pos],
            snapshot.anchor()
        )
        .is_err());
    }
    let tmp = Temp::new();
    let mut bytes = snapshot.bytes().to_vec();
    bytes[72] ^= 1;
    assert!(PtrRuntime::restore_durable_snapshot(
        PtrConfig::default(),
        &bytes,
        snapshot.anchor(),
        tmp.path("must-not-exist")
    )
    .is_err());
    assert!(!tmp.path("must-not-exist").exists());
}

#[test]
fn wrong_trusted_coordinates_and_rollback_are_rejected() {
    let mut runtime = fixture();
    let old = runtime.export_recovery_snapshot().unwrap();
    runtime.ingest_text("new".into(), "changed").unwrap();
    let current = runtime.export_recovery_snapshot().unwrap();
    assert!(PtrRuntime::restore_recovery_snapshot(
        PtrConfig::default(),
        old.bytes(),
        current.anchor()
    )
    .is_err());
    for which in 0..4 {
        let mut expected = current.anchor();
        match which {
            0 => expected.revision.0 += 1,
            1 => expected.log.index.0 += 1,
            2 => expected.log.digest[0] ^= 1,
            _ => expected.digest[0] ^= 1,
        }
        assert!(PtrRuntime::restore_recovery_snapshot(
            PtrConfig::default(),
            current.bytes(),
            expected
        )
        .is_err());
    }
}

#[test]
fn internally_valid_checksums_do_not_bypass_semantic_revision_validation() {
    let s = fixture().export_recovery_snapshot().unwrap();
    let mut bytes = s.bytes().to_vec();
    bytes[8..16].copy_from_slice(&99_u64.to_le_bytes());
    let end = bytes.len() - 32;
    let digest = integrity::sha256(&bytes[..end]);
    bytes[end..].copy_from_slice(&digest);
    let trusted = SnapshotAnchor {
        revision: Revision(99),
        digest,
        ..s.anchor()
    };
    assert!(PtrRuntime::restore_recovery_snapshot(PtrConfig::default(), &bytes, trusted).is_err());
}

#[test]
fn immutable_snapshot_files_and_existing_live_logs_are_never_overwritten() {
    let tmp = Temp::new();
    let s = fixture().export_recovery_snapshot().unwrap();
    s.write_new(tmp.path("snapshot")).unwrap();
    assert!(s.write_new(tmp.path("snapshot")).is_err());
    assert_eq!(std::fs::read(tmp.path("snapshot")).unwrap(), s.bytes());
    let restored =
        PtrRuntime::read_recovery_snapshot(PtrConfig::default(), tmp.path("snapshot"), s.anchor())
            .unwrap();
    compare(&fixture(), &restored);
    std::fs::write(tmp.path("live"), b"do not replace").unwrap();
    assert!(PtrRuntime::restore_durable_snapshot(
        PtrConfig::default(),
        s.bytes(),
        s.anchor(),
        tmp.path("live")
    )
    .is_err());
    assert_eq!(std::fs::read(tmp.path("live")).unwrap(), b"do not replace");
    let file = std::fs::File::create(tmp.path("huge")).unwrap();
    file.set_len(MAX_SNAPSHOT_BYTES as u64 + 1).unwrap();
    drop(file);
    assert!(
        PtrRuntime::read_recovery_snapshot(PtrConfig::default(), tmp.path("huge"), s.anchor())
            .is_err()
    );
}

#[test]
fn explicit_legacy_migration_validates_before_publication_and_keeps_source() {
    let tmp = Temp::new();
    let legacy = [7, 0, 0, 0, 4, 1, 0, 0, 0, b'x', 1];
    std::fs::write(tmp.path("v1"), legacy).unwrap();
    assert!(PtrRuntime::open_durable(PtrConfig::default(), tmp.path("v1")).is_err());
    let r = PtrRuntime::migrate_legacy_log(PtrConfig::default(), tmp.path("v1"), tmp.path("v2"))
        .unwrap();
    assert_eq!(r.committed_events().len(), 1);
    let anchor = r.journal_anchor().unwrap();
    drop(r);
    assert_eq!(std::fs::read(tmp.path("v1")).unwrap(), legacy);
    assert!(PtrRuntime::open_durable_at(PtrConfig::default(), tmp.path("v2"), anchor).is_ok());
    // A valid legacy codec with an impossible future SnapshotCommitted event.
    let mut invalid = 17_u32.to_le_bytes().to_vec();
    invalid.push(7);
    invalid.extend_from_slice(&0_u64.to_le_bytes());
    invalid.extend_from_slice(&99_u64.to_le_bytes());
    std::fs::write(tmp.path("invalid-v1"), &invalid).unwrap();
    assert!(PtrRuntime::migrate_legacy_log(
        PtrConfig::default(),
        tmp.path("invalid-v1"),
        tmp.path("no-target")
    )
    .is_err());
    assert!(!tmp.path("no-target").exists());
    assert_eq!(std::fs::read(tmp.path("invalid-v1")).unwrap(), invalid);
}

#[test]
fn empty_state_snapshot_roundtrip_has_a_real_genesis_anchor() {
    let r = PtrRuntime::new(PtrConfig::default()).unwrap();
    let s = r.export_recovery_snapshot().unwrap();
    assert_eq!(s.anchor().log, LogAnchor::empty());
    let restored =
        PtrRuntime::restore_recovery_snapshot(PtrConfig::default(), s.bytes(), s.anchor()).unwrap();
    assert_eq!(restored.revision(), Revision(0));
    assert!(restored.committed_events().is_empty());
    assert_eq!(restored.snapshot().value("missing"), None::<&SemanticValue>);
}
