//! Retention planning, the cutover commit point, and what an interruption on
//! either side of it leaves behind.
use ptr_ledger::acknowledged::{AcknowledgedLedger, Split, TailPolicy};
use ptr_ledger::anchor::{AnchorKey, AnchorStore};
use ptr_ledger::compaction::{CompactionDecision, CompactionPlan, LogPaths, RetentionPolicy};
use ptr_ledger::integrity::{chain_anchors, decode_log, decode_log_from, LogAnchor};
use ptr_ledger::{CommittedEvent, CompactionBarrier, FileLedger, LedgerEvent};
use ptr_types::{CommitIndex, Generation};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-compaction-{}-{}",
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
    AnchorKey::from_bytes([0x33; 32])
}
fn revocation(n: u64) -> LedgerEvent {
    LedgerEvent::Revoked {
        subject: format!("subject:{n}"),
        generation: Generation(n),
    }
}
fn subjects(events: &[CommittedEvent]) -> Vec<String> {
    events
        .iter()
        .map(|committed| match &committed.event {
            LedgerEvent::Revoked { subject, .. } => subject.clone(),
            other => panic!("unexpected event {other:?}"),
        })
        .collect()
}
fn indices(events: &[CommittedEvent]) -> Vec<u64> {
    events.iter().map(|event| event.index.0).collect()
}
fn scratch_of(anchor: &Path) -> PathBuf {
    let mut name = anchor.file_name().unwrap().to_os_string();
    name.push(".publishing");
    anchor.with_file_name(name)
}

fn ledger(tmp: &Temp, records: u64) -> AcknowledgedLedger {
    let mut ledger = AcknowledgedLedger::create(&tmp.paths(), key()).unwrap();
    for n in 1..=records {
        ledger.append_acknowledged(revocation(n)).unwrap();
    }
    ledger
}

/// A barrier that permits compaction up to `covers`.
fn clear_barrier(covers: u64) -> CompactionBarrier {
    CompactionBarrier {
        snapshot_covers: CommitIndex(covers),
        all_consumers_caught_up: true,
        unresolved_revocations: 0,
    }
}

/// Verifies that retention holds until enough history sits above the floor.
#[test]
fn retention_holds_until_enough_history_sits_above_the_floor() {
    let tmp = Temp::new();
    let ledger = ledger(&tmp, 3);

    // Keeping five of three records leaves nothing to discard.
    assert_eq!(
        ledger
            .plan_compaction(RetentionPolicy::keeping(5), clear_barrier(3))
            .unwrap(),
        CompactionDecision::Retain
    );
    // Exactly at the retention count is still nothing to discard.
    assert_eq!(
        ledger
            .plan_compaction(RetentionPolicy::keeping(3), clear_barrier(3))
            .unwrap(),
        CompactionDecision::Retain
    );

    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(2), clear_barrier(3))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    assert_eq!(plan.base.index, CommitIndex(1));
    assert_eq!(plan.discarded_records, 1);
    assert_eq!(plan.retained_records, 2);
    assert_eq!(plan.tail, ledger.anchor().log);
}

/// Verifies that an unsafe barrier blocks regardless of retention.
#[test]
fn an_unsafe_barrier_blocks_regardless_of_retention() {
    let tmp = Temp::new();
    let ledger = ledger(&tmp, 6);
    let policy = RetentionPolicy::keeping(1);

    for barrier in [
        CompactionBarrier {
            snapshot_covers: CommitIndex(5),
            all_consumers_caught_up: false,
            unresolved_revocations: 0,
        },
        CompactionBarrier {
            snapshot_covers: CommitIndex(5),
            all_consumers_caught_up: true,
            unresolved_revocations: 2,
        },
    ] {
        assert!(!barrier.safe());
        assert_eq!(
            ledger.plan_compaction(policy, barrier).unwrap(),
            CompactionDecision::Blocked(barrier)
        );
    }
}

/// Verifies that the floor never rises above snapshot coverage.
#[test]
fn the_floor_never_rises_above_snapshot_coverage() {
    let tmp = Temp::new();
    let ledger = ledger(&tmp, 5);

    // Retention alone would propose floor 4; coverage caps it at 2.
    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(1), clear_barrier(2))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    assert_eq!(plan.base.index, CommitIndex(2));
    assert_eq!(plan.discarded_records, 2);
    assert_eq!(plan.retained_records, 3);

    // Coverage at or below the current floor leaves nothing safe to discard.
    let barrier = clear_barrier(0);
    assert_eq!(
        ledger
            .plan_compaction(RetentionPolicy::keeping(1), barrier)
            .unwrap(),
        CompactionDecision::Blocked(barrier)
    );
}

/// Verifies that a cutover reconstructs exactly the retained history.
#[test]
fn a_cutover_reconstructs_exactly_the_retained_history() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 6);
    let before = ledger.anchor();
    let old_log = tmp.paths().log_path(CommitIndex(0));

    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(3), clear_barrier(6))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    let outcome = ledger.compact(plan).unwrap();
    assert_eq!(outcome.live_log, tmp.paths().log_path(CommitIndex(3)));
    assert_eq!(outcome.superseded_log, old_log);

    // In memory: only the retained suffix, at its original indices.
    assert_eq!(
        subjects(ledger.events()),
        ["subject:4", "subject:5", "subject:6"]
    );
    assert_eq!(indices(ledger.events()), [4, 5, 6]);
    // The tail is untouched and the floor has moved.
    assert_eq!(ledger.anchor().log, before.log);
    assert_eq!(ledger.anchor().base, plan.base);
    assert!(ledger.anchor().is_compacted());
    assert!(ledger.epoch() > before.epoch);
    // Nothing was overwritten: the superseded log is still there.
    assert!(old_log.exists());

    // Appending continues the same chain above the floor.
    ledger.append_acknowledged(revocation(7)).unwrap();
    assert_eq!(indices(ledger.events()), [4, 5, 6, 7]);
    let epoch = ledger.epoch();
    drop(ledger);

    // A reopen resolves the live file from the anchor and reconstructs exactly.
    let (reopened, split) =
        AcknowledgedLedger::open(&tmp.paths(), key(), epoch, TailPolicy::Acknowledge).unwrap();
    assert_eq!(split, Split::Aligned);
    assert_eq!(indices(reopened.events()), [4, 5, 6, 7]);
    assert_eq!(
        subjects(reopened.events()),
        ["subject:4", "subject:5", "subject:6", "subject:7"]
    );
    assert_eq!(reopened.anchor().base, plan.base);
}

/// Verifies that a compacted log cannot be read without its trusted floor.
#[test]
fn a_compacted_log_cannot_be_read_without_its_trusted_floor() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 4);
    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(2), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    let outcome = ledger.compact(plan).unwrap();
    let base = ledger.anchor().base;
    drop(ledger);

    let bytes = std::fs::read(&outcome.live_log).unwrap();
    // Reading it as though it began at index 1 fails closed rather than guessing.
    assert!(decode_log(&bytes).is_err());
    assert!(FileLedger::open(&outcome.live_log).is_err());
    // With the floor from protected anchor storage it verifies exactly.
    let verified = decode_log_from(&bytes, base).unwrap();
    assert_eq!(indices(verified.events()), [3, 4]);
    assert_eq!(verified.anchor(), plan.tail);
}

/// Verifies that an interruption before the commit point keeps the previous log live.
#[test]
fn an_interruption_before_the_commit_point_keeps_the_previous_log_live() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 4);
    let before = ledger.anchor();
    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(2), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };

    // Block anchor publication, so the cutover fails exactly at its commit point
    // with the new artifact already built.
    let scratch = scratch_of(&tmp.paths().anchor_path());
    std::fs::create_dir(&scratch).unwrap();
    assert!(ledger.compact(plan).is_err());
    std::fs::remove_dir(&scratch).unwrap();

    // The anchor still names the old floor, and nothing was lost.
    assert_eq!(ledger.anchor(), before);
    assert_eq!(indices(ledger.events()), [1, 2, 3, 4]);
    let half_built = tmp.paths().log_path(plan.base.index);
    assert!(half_built.exists());
    assert_eq!(
        ledger.paths().orphans(CommitIndex(0)).unwrap(),
        [half_built]
    );

    // Reclaiming clears it, and the retry then succeeds.
    assert_eq!(ledger.reclaim_orphans().unwrap().len(), 1);
    let outcome = ledger.compact(plan).unwrap();
    assert_eq!(outcome.live_log, tmp.paths().log_path(plan.base.index));
    assert_eq!(indices(ledger.events()), [3, 4]);
}

/// Verifies that an interruption after the commit point keeps the new log live.
#[test]
fn an_interruption_after_the_commit_point_keeps_the_new_log_live() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 4);
    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(2), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    let outcome = ledger.compact(plan).unwrap();
    let epoch = ledger.epoch();
    // Crash immediately after the commit point: no reclamation ran.
    drop(ledger);
    assert!(outcome.superseded_log.exists());
    assert!(outcome.live_log.exists());

    let (reopened, split) =
        AcknowledgedLedger::open(&tmp.paths(), key(), epoch, TailPolicy::Acknowledge).unwrap();
    assert_eq!(split, Split::Aligned);
    assert_eq!(indices(reopened.events()), [3, 4]);
    assert_eq!(
        reopened.paths().orphans(plan.base.index).unwrap(),
        std::slice::from_ref(&outcome.superseded_log)
    );
    assert_eq!(
        reopened.reclaim_orphans().unwrap(),
        std::slice::from_ref(&outcome.superseded_log)
    );
    assert!(!outcome.superseded_log.exists());
    assert!(outcome.live_log.exists());
}

/// Verifies that a stale or off boundary plan is refused without touching anything.
#[test]
fn a_stale_or_off_boundary_plan_is_refused_without_touching_anything() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 4);
    let anchor = ledger.anchor();
    let anchors = chain_anchors(ledger.events(), anchor.base).unwrap();
    let good = CompactionPlan {
        base: anchors[1],
        tail: anchor.log,
        discarded_records: 2,
        retained_records: 2,
    };

    let stale_tail = CompactionPlan {
        tail: LogAnchor {
            index: CommitIndex(4),
            digest: [7; 32],
        },
        ..good
    };
    assert_eq!(
        ledger.compact(stale_tail).err().map(|error| error.code()),
        Some("PTR_COMPACT_STALE_PLAN")
    );

    let not_advancing = CompactionPlan {
        base: LogAnchor::empty(),
        ..good
    };
    assert_eq!(
        ledger
            .compact(not_advancing)
            .err()
            .map(|error| error.code()),
        Some("PTR_COMPACT_FLOOR_NOT_ADVANCING")
    );

    let above_tail = CompactionPlan {
        base: LogAnchor {
            index: CommitIndex(9),
            digest: anchors[1].digest,
        },
        ..good
    };
    assert_eq!(
        ledger.compact(above_tail).err().map(|error| error.code()),
        Some("PTR_COMPACT_FLOOR_ABOVE_TAIL")
    );

    // A real index carrying a digest this log never produced.
    let wrong_digest = CompactionPlan {
        base: LogAnchor {
            index: CommitIndex(2),
            digest: [0xcd; 32],
        },
        ..good
    };
    assert_eq!(
        ledger.compact(wrong_digest).err().map(|error| error.code()),
        Some("PTR_COMPACT_FLOOR_NOT_ON_BOUNDARY")
    );

    // Every refusal left the ledger exactly as it was.
    assert_eq!(ledger.anchor(), anchor);
    assert_eq!(indices(ledger.events()), [1, 2, 3, 4]);
    assert!(ledger.paths().orphans(CommitIndex(0)).unwrap().is_empty());
    // The plan that was valid all along still works.
    assert!(ledger.compact(good).is_ok());
}

/// Verifies that an existing destination is never overwritten.
#[test]
fn an_existing_destination_is_never_overwritten() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 4);
    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(2), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    let destination = tmp.paths().log_path(plan.base.index);
    std::fs::write(&destination, b"not a log").unwrap();

    assert_eq!(
        ledger.compact(plan).err().map(|error| error.code()),
        Some("PTR_COMPACT_DESTINATION_EXISTS")
    );
    assert_eq!(std::fs::read(&destination).unwrap(), b"not a log");
    assert_eq!(indices(ledger.events()), [1, 2, 3, 4]);
}

/// Verifies that the origin survives discarding the record that defined it.
#[test]
fn the_origin_survives_discarding_the_record_that_defined_it() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 4);
    let original_origin = ledger.anchor().origin;
    assert!(original_origin.is_set());

    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(2), clear_barrier(4))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    ledger.compact(plan).unwrap();

    // Index 1 is gone, so the log can no longer derive its own identity; the
    // anchor carries it instead of letting a substituted log name itself.
    assert_eq!(ledger.log().derived_origin(), None);
    assert_eq!(ledger.anchor().origin, original_origin);
    let epoch = ledger.epoch();
    ledger.append_acknowledged(revocation(5)).unwrap();
    assert_eq!(ledger.anchor().origin, original_origin);
    assert!(ledger.epoch() > epoch);
}

/// Verifies that split detection still applies above a compacted floor.
#[test]
fn split_detection_still_applies_above_a_compacted_floor() {
    let tmp = Temp::new();
    let mut ledger = ledger(&tmp, 5);
    let CompactionDecision::Compact(plan) = ledger
        .plan_compaction(RetentionPolicy::keeping(3), clear_barrier(5))
        .unwrap()
    else {
        panic!("a floor was expected");
    };
    let outcome = ledger.compact(plan).unwrap();
    let epoch = ledger.epoch();
    drop(ledger);

    // An unacknowledged durable record above the floor still recovers.
    let mut direct = FileLedger::lock_for_recovery(&outcome.live_log, plan.base)
        .unwrap()
        .repair(
            AnchorStore::open_expecting(tmp.paths().anchor_path(), key(), epoch)
                .unwrap()
                .current(),
            TailPolicy::Acknowledge,
        )
        .unwrap()
        .0;
    direct.append_durable(revocation(6)).unwrap();
    drop(direct);

    let (split, _) = AcknowledgedLedger::inspect(&tmp.paths(), key(), epoch).unwrap();
    assert!(matches!(
        split,
        Split::Unacknowledged {
            complete_records: 1,
            incomplete_frame: false,
            ..
        }
    ));
    let (recovered, _) =
        AcknowledgedLedger::open(&tmp.paths(), key(), epoch, TailPolicy::Acknowledge).unwrap();
    assert_eq!(indices(recovered.events()), [3, 4, 5, 6]);
    let recovered_epoch = recovered.epoch();
    drop(recovered);

    // Losing a committed record above the floor is still unrecoverable.
    let truncated = {
        let bytes = std::fs::read(&outcome.live_log).unwrap();
        let verified = decode_log_from(&bytes, plan.base).unwrap();
        let keep = &verified.events()[..verified.events().len() - 1];
        ptr_ledger::integrity::encode_log_from(keep, plan.base).unwrap()
    };
    std::fs::write(&outcome.live_log, &truncated).unwrap();
    let (split, _) = AcknowledgedLedger::inspect(&tmp.paths(), key(), recovered_epoch).unwrap();
    assert_eq!(
        split,
        Split::LostSuffix {
            acknowledged: CommitIndex(6),
            present: CommitIndex(5),
        }
    );
    assert!(!split.is_recoverable());
}

#[test]
fn a_neighbouring_log_set_is_never_reported_as_this_set_s_orphan() {
    // `journal-` is also the start of `journal-backup-...`, and reclaim_orphans
    // deletes what orphans() returns — so a prefix match alone would delete
    // another set's live log.
    let tmp = Temp::new();
    let ours = tmp.paths();
    let neighbour = LogPaths::new(ours.directory(), "journal-backup");

    let live = ours.log_path(CommitIndex(0));
    let stale = ours.log_path(CommitIndex(3));
    let theirs = neighbour.log_path(CommitIndex(0));
    for path in [&live, &stale, &theirs] {
        std::fs::write(path, b"").unwrap();
    }
    // Names this set could not have written: a short field, a non-numeric one,
    // a value past u64, and a non-canonical spelling of a number.
    for name in [
        "journal-1.log",
        "journal-000000000000000000ff.log",
        "journal-99999999999999999999.log",
        "journal-+0000000000000000001.log",
    ] {
        std::fs::write(ours.directory().join(name), b"").unwrap();
    }

    assert_eq!(
        ours.orphans(CommitIndex(0)).unwrap(),
        std::slice::from_ref(&stale)
    );
    // Symmetrically, the neighbour does not own ours.
    assert_eq!(
        neighbour.orphans(CommitIndex(0)).unwrap(),
        Vec::<PathBuf>::new()
    );
    assert!(theirs.exists());

    // The live log is excluded by identity, not by name shape.
    assert_eq!(
        ours.orphans(CommitIndex(3)).unwrap(),
        [live],
        "the other floor becomes the orphan once the anchor names this one"
    );
    assert!(stale.exists());
}
