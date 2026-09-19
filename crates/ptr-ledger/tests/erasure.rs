//! What deletion removes and what still holds it.
//!
//! The first test is the gate's own sentence — "logical current-value deletion
//! does not erase history" — turned into an assertion, because a claim like that
//! is only worth anything if the system is checked against it rather than
//! documented as complying with it.
use ptr_ledger::acknowledged::{AcknowledgedLedger, TailPolicy};
use ptr_ledger::anchor::AnchorKey;
use ptr_ledger::compaction::{CompactionDecision, LogPaths, RetentionPolicy};
use ptr_ledger::integrity::{decode_log_from, encode_log};
use ptr_ledger::retention::{retains, OutOfReach, Retainer};
use ptr_ledger::{CommittedEvent, CompactionBarrier, LedgerEvent};
use ptr_types::{CommitIndex, Generation, Revision};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A value that will not occur by chance in framing, digests or indices.
const SECRET: &[u8] = b"patient-4711-diagnosis-confidential";
const KEPT: &[u8] = b"unrelated-retained-value";

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-erasure-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn paths(&self) -> LogPaths {
        LogPaths::new(&self.0, "journal")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn key() -> AnchorKey {
    AnchorKey::from_bytes([0x6e; 32])
}

/// A semantic transaction carrying `value` as opaque committed bytes, which is how
/// a real payload reaches the journal.
fn carrying(base: u64, value: &[u8]) -> LedgerEvent {
    LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(base),
        revision: Revision(base + 1),
        encoded_delta: value.to_vec(),
    }
}

fn noise(n: u64) -> LedgerEvent {
    LedgerEvent::Revoked {
        subject: format!("subject:{n}"),
        generation: Generation(n),
    }
}

fn clear_barrier(covers: u64) -> CompactionBarrier {
    CompactionBarrier {
        snapshot_covers: CommitIndex(covers),
        all_consumers_caught_up: true,
        unresolved_revocations: 0,
    }
}

/// A ledger whose record 1 carries the secret and whose record 2 supersedes it.
fn ledger_with_superseded_secret(tmp: &Temp) -> AcknowledgedLedger {
    let mut ledger = AcknowledgedLedger::create(&tmp.paths(), key()).unwrap();
    ledger.append_acknowledged(carrying(0, SECRET)).unwrap();
    // The logical deletion: a later transaction replaces the value.
    ledger.append_acknowledged(carrying(1, KEPT)).unwrap();
    ledger.append_acknowledged(noise(3)).unwrap();
    ledger.append_acknowledged(noise(4)).unwrap();
    ledger
}

#[test]
fn logical_deletion_does_not_erase_history() {
    let tmp = Temp::new();
    let ledger = ledger_with_superseded_secret(&tmp);
    let base = ledger.anchor().base.index;

    // The value is superseded: the newest semantic transaction carries the
    // replacement, so nothing current refers to the old value any more.
    let current = ledger
        .events()
        .iter()
        .rev()
        .find_map(|committed| match &committed.event {
            LedgerEvent::SemanticDeltaCommitted { encoded_delta, .. } => Some(encoded_delta),
            _ => None,
        })
        .expect("a semantic transaction was committed");
    assert_eq!(current.as_slice(), KEPT);
    assert!(!retains(current, SECRET));
    drop(ledger);

    // And yet it is still there, in the live log, in plaintext.
    let audit = tmp.paths().audit_erasure(SECRET, base).unwrap();
    assert!(!audit.erased_where_reachable());
    assert_eq!(
        audit.retainers(),
        [Retainer::Log(tmp.paths().log_path(base))]
    );
    assert_eq!(audit.scanned_logs(), 1);

    // An audit never claims anything about what it cannot reach.
    assert_eq!(
        audit.out_of_reach(),
        [
            OutOfReach::HostRetainedArtifacts,
            OutOfReach::StorageResidue,
            OutOfReach::ModelDerivedState,
        ]
    );
}

#[test]
fn erasure_needs_the_floor_past_the_record_and_the_orphan_reclaimed() {
    let tmp = Temp::new();
    let mut ledger = ledger_with_superseded_secret(&tmp);

    // Raise the floor past record 1, which is the only record carrying the secret.
    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(3), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    assert_eq!(plan.base.index, CommitIndex(1));
    let outcome = ledger.compact(plan).unwrap();
    let base = ledger.anchor().base.index;

    // The live log no longer holds it — but the superseded file still does, so
    // compaction alone is not erasure.
    let audit = tmp.paths().audit_erasure(SECRET, base).unwrap();
    assert!(!audit.erased_where_reachable());
    assert_eq!(
        audit.retainers(),
        [Retainer::Log(outcome.superseded_log.clone())]
    );
    assert_eq!(audit.scanned_logs(), 2);
    assert!(!retains(&std::fs::read(&outcome.live_log).unwrap(), SECRET));

    // Reclaiming the orphan completes it for everything reachable.
    assert_eq!(
        ledger.reclaim_orphans().unwrap(),
        std::slice::from_ref(&outcome.superseded_log)
    );
    let audit = tmp.paths().audit_erasure(SECRET, base).unwrap();
    assert!(audit.erased_where_reachable());
    assert!(audit.retainers().is_empty());
    assert_eq!(audit.scanned_logs(), 1);
    assert_eq!(audit.live_base(), base);

    // Unrelated committed data is untouched: this is targeted retention, not a
    // wipe. And the surviving log still verifies against its floor.
    let audit = tmp.paths().audit_erasure(KEPT, base).unwrap();
    assert!(!audit.erased_where_reachable());
    let bytes = std::fs::read(&outcome.live_log).unwrap();
    let verified = decode_log_from(&bytes, ledger.anchor().base).unwrap();
    assert_eq!(verified.events().len(), 3);
    assert_eq!(verified.anchor(), ledger.anchor().log);
}

#[test]
fn an_unreclaimed_orphan_keeps_retaining_after_a_later_cutover() {
    let tmp = Temp::new();
    let mut ledger = ledger_with_superseded_secret(&tmp);
    let CompactionDecision::Compact(first) = ledger
        .plan_compaction(RetentionPolicy::keeping(3), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    let first = ledger.compact(first).unwrap();
    // Deliberately skip reclamation, as an interrupted operator would.
    let CompactionDecision::Compact(second) = ledger
        .plan_compaction(RetentionPolicy::keeping(1), clear_barrier(4))
        .unwrap()
    else {
        panic!("a second floor was expected");
    };
    let second = ledger.compact(second).unwrap();
    let base = ledger.anchor().base.index;

    // Two orphans now exist and the oldest still carries the secret.
    let audit = tmp.paths().audit_erasure(SECRET, base).unwrap();
    assert_eq!(audit.scanned_logs(), 3);
    assert_eq!(
        audit.retainers(),
        [Retainer::Log(first.superseded_log.clone())]
    );
    assert!(!audit.erased_where_reachable());

    let reclaimed = ledger.reclaim_orphans().unwrap();
    assert_eq!(reclaimed.len(), 2);
    assert!(reclaimed.contains(&first.superseded_log));
    assert!(reclaimed.contains(&second.superseded_log));
    assert!(tmp
        .paths()
        .audit_erasure(SECRET, base)
        .unwrap()
        .erased_where_reachable());
}

#[test]
fn a_snapshot_the_host_retains_defeats_erasure_and_must_be_declared() {
    let tmp = Temp::new();
    let mut ledger = ledger_with_superseded_secret(&tmp);

    // Stand in for a recovery snapshot: the whole journal, copied elsewhere before
    // the floor moved. ptr-runtime's RecoverySnapshot embeds exactly this.
    let retained_elsewhere = encode_log(ledger.events()).unwrap();
    assert!(retains(&retained_elsewhere, SECRET));

    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(2), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    ledger.compact(plan).unwrap();
    ledger.reclaim_orphans().unwrap();
    let base = ledger.anchor().base.index;

    // Reachable erasure is complete.
    let audit = tmp.paths().audit_erasure(SECRET, base).unwrap();
    assert!(audit.erased_where_reachable());

    // Folding in the host's own artifact shows the truth: still retained. An audit
    // cannot discover this, which is why the boundary is a permanent part of its
    // report rather than a caveat in prose.
    let audit = audit.with_retained_elsewhere("recovery-snapshot", &retained_elsewhere);
    assert!(!audit.erased_where_reachable());
    assert_eq!(
        audit.retainers(),
        [Retainer::Elsewhere("recovery-snapshot".into())]
    );

    // An artifact that does not contain it is not reported.
    let unrelated = encode_log(&[CommittedEvent {
        index: CommitIndex(1),
        event: noise(9),
    }])
    .unwrap();
    let audit = tmp
        .paths()
        .audit_erasure(SECRET, base)
        .unwrap()
        .with_retained_elsewhere("unrelated", &unrelated);
    assert!(audit.erased_where_reachable());
}

#[test]
fn destroying_the_anchor_key_is_not_erasure() {
    let tmp = Temp::new();
    let ledger = ledger_with_superseded_secret(&tmp);
    let base = ledger.anchor().base;
    drop(ledger);

    // Lose the key and the anchor with it: verification is gone.
    std::fs::remove_file(tmp.paths().anchor_path()).unwrap();
    assert!(AcknowledgedLedger::open(&tmp.paths(), key(), 0, TailPolicy::Acknowledge).is_err());

    // The plaintext is completely unaffected, and the records still decode. A key
    // protects integrity, never confidentiality; treating its destruction as
    // erasure would be a category error.
    let audit = tmp.paths().audit_erasure(SECRET, base.index).unwrap();
    assert!(!audit.erased_where_reachable());
    let bytes = std::fs::read(tmp.paths().log_path(base.index)).unwrap();
    let verified = decode_log_from(&bytes, base).unwrap();
    assert_eq!(verified.events().len(), 4);
}

#[test]
fn the_audit_reports_presence_not_interpretation() {
    // A byte search is deliberate: a record scan that failed to recognize an
    // encoding would report erasure that did not happen, and an audit must never
    // err in that direction.
    assert!(retains(b"framing..SECRET..framing", b"SECRET"));
    assert!(retains(b"SECRET", b"SECRET"));
    assert!(!retains(b"SECRE", b"SECRET"));
    assert!(!retains(b"", b"SECRET"));
    // An empty needle is not a match: "everything retains nothing" is not a useful
    // answer, and returning true would make every audit fail.
    assert!(!retains(b"anything", b""));
}

#[test]
fn erasure_boundaries_carry_stable_codes() {
    assert_eq!(
        OutOfReach::ALL.map(|boundary| boundary.code()),
        [
            "PTR_ERASURE_HOST_RETAINED_ARTIFACTS",
            "PTR_ERASURE_STORAGE_RESIDUE",
            "PTR_ERASURE_MODEL_DERIVED_STATE",
        ]
    );
}
