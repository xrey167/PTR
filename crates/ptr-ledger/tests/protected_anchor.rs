//! Protected anchor storage and the log/anchor acknowledgment protocol.
//!
//! Every split outcome from `ptr_ledger::acknowledged` is exercised here, in both
//! directions: a recoverable state must recover to an exact expected history, and
//! an unrecoverable state must refuse *and* leave every byte in place.
use ptr_ledger::acknowledged::{AcknowledgedError, AcknowledgedLedger, Split, TailPolicy};
use ptr_ledger::anchor::{AnchorError, AnchorKey, AnchorStore, ChainOrigin, ProtectedAnchor};
use ptr_ledger::compaction::LogPaths;
use ptr_ledger::integrity::{decode_log, encode_log, LogAnchor};
use ptr_ledger::{CommittedEvent, FileLedger, LedgerEvent};
use ptr_types::{CommitIndex, Generation};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-anchor-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn paths(&self) -> LogPaths {
        LogPaths::new(&self.0, "journal")
    }
    /// The uncompacted log, whose floor is index 0.
    fn log(&self) -> PathBuf {
        self.paths().log_path(CommitIndex(0))
    }
    fn anchor(&self) -> PathBuf {
        self.paths().anchor_path()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn key() -> AnchorKey {
    AnchorKey::from_bytes([0x5a; 32])
}
fn other_key() -> AnchorKey {
    AnchorKey::from_bytes([0xa5; 32])
}
fn revocation(n: u64) -> LedgerEvent {
    LedgerEvent::Revoked {
        subject: format!("subject:{n}"),
        generation: Generation(n),
    }
}
fn committed(n: u64) -> CommittedEvent {
    CommittedEvent {
        index: CommitIndex(n),
        event: revocation(n),
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
/// Read a file's bytes.
///
/// Windows advisory locks are mandatory for I/O through independent handles, so a
/// log must never be read while a ledger still holds it. Every call below either
/// targets the anchor, which is not locked, or follows a `drop`.
fn snapshot(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()
}
/// Scratch name an interrupted anchor publication leaves behind.
fn scratch_of(anchor: &Path) -> PathBuf {
    let mut name = anchor.file_name().unwrap().to_os_string();
    name.push(".publishing");
    anchor.with_file_name(name)
}

/// Append durably without advancing the anchor: exactly the state a crash between
/// the two synchronized writes leaves behind.
fn append_without_acknowledging(log: &Path, event: LedgerEvent) {
    let mut ledger = FileLedger::open(log).unwrap();
    ledger.append_durable(event).unwrap();
}

fn make_ledger(tmp: &Temp, records: u64) -> AcknowledgedLedger {
    let mut ledger = AcknowledgedLedger::create(&tmp.paths(), key()).unwrap();
    for n in 1..=records {
        assert_eq!(
            ledger.append_acknowledged(revocation(n)).unwrap(),
            CommitIndex(n)
        );
    }
    ledger
}

#[test]
fn initialization_is_single_shot_and_round_trips_through_authentication() {
    let tmp = Temp::new();
    let store = AnchorStore::initialize(tmp.anchor(), key(), ProtectedAnchor::initial()).unwrap();
    let published = store.current();
    assert_eq!(published, ProtectedAnchor::initial());
    assert_eq!(published.epoch, 1);
    assert!(!published.is_compacted());
    drop(store);

    let reopened = AnchorStore::open(tmp.anchor(), key()).unwrap();
    assert_eq!(reopened.current(), published);

    // A second initialization would silently replace a history's commitment.
    assert_eq!(
        AnchorStore::initialize(tmp.anchor(), key(), ProtectedAnchor::initial()).err(),
        Some(AnchorError::Exists)
    );
}

#[test]
fn absent_forged_and_mutated_anchors_are_all_refused() {
    let tmp = Temp::new();
    assert_eq!(
        AnchorStore::open(tmp.anchor(), key()).err(),
        Some(AnchorError::Absent)
    );

    let mut store =
        AnchorStore::initialize(tmp.anchor(), key(), ProtectedAnchor::initial()).unwrap();
    store
        .acknowledge(
            LogAnchor {
                index: CommitIndex(4),
                digest: [3; 32],
            },
            ChainOrigin::from_first_record([9; 32]),
        )
        .unwrap();
    drop(store);

    // Authentic under one key is meaningless under another.
    assert_eq!(
        AnchorStore::open(tmp.anchor(), other_key()).err(),
        Some(AnchorError::Unauthenticated)
    );

    let original = snapshot(&tmp.anchor());
    for offset in 0..original.len() {
        for bit in 0..8 {
            let mut bytes = original.clone();
            bytes[offset] ^= 1 << bit;
            std::fs::write(tmp.anchor(), &bytes).unwrap();
            assert!(
                AnchorStore::open(tmp.anchor(), key()).is_err(),
                "offset {offset} bit {bit} accepted"
            );
        }
    }

    // Truncation and extension change the length, which is checked before parsing.
    for length in [0usize, 1, original.len() - 1] {
        std::fs::write(tmp.anchor(), &original[..length]).unwrap();
        assert_eq!(
            AnchorStore::open(tmp.anchor(), key()).err(),
            Some(AnchorError::Format)
        );
    }
    let mut extended = original.clone();
    extended.push(0);
    std::fs::write(tmp.anchor(), &extended).unwrap();
    assert_eq!(
        AnchorStore::open(tmp.anchor(), key()).err(),
        Some(AnchorError::Format)
    );

    std::fs::write(tmp.anchor(), &original).unwrap();
    assert!(AnchorStore::open(tmp.anchor(), key()).is_ok());
}

#[test]
fn stale_witness_rollback_is_detected_only_by_the_retained_epoch() {
    let tmp = Temp::new();
    let mut ledger = make_ledger(&tmp, 2);
    let stale = snapshot(&tmp.anchor());
    let stale_epoch = ledger.epoch();
    ledger.append_acknowledged(revocation(3)).unwrap();
    let fresh_epoch = ledger.epoch();
    assert!(fresh_epoch > stale_epoch);
    drop(ledger);

    // Reinstating the older record restores a genuinely authentic commitment.
    std::fs::write(tmp.anchor(), &stale).unwrap();
    let unwitnessed = AnchorStore::open(tmp.anchor(), key()).unwrap();
    assert_eq!(unwitnessed.epoch(), stale_epoch);
    drop(unwitnessed);

    // Only the retained counter exposes it, which is the whole point of holding
    // the epoch outside the file being checked.
    assert_eq!(
        AnchorStore::open_expecting(tmp.anchor(), key(), fresh_epoch).err(),
        Some(AnchorError::StaleEpoch {
            stored: stale_epoch,
            expected: fresh_epoch,
        })
    );

    // The rolled-back anchor now describes less history than the log holds, so
    // the acknowledgment layer reports a recoverable tail rather than health.
    let (split, _) = AcknowledgedLedger::inspect(&tmp.paths(), key(), 0).unwrap();
    assert!(matches!(
        split,
        Split::Unacknowledged {
            complete_records: 1,
            incomplete_frame: false,
            ..
        }
    ));
}

#[test]
fn interrupted_publication_leaves_the_previous_anchor_authoritative() {
    let tmp = Temp::new();
    let ledger = make_ledger(&tmp, 1);
    let before = ledger.anchor();
    drop(ledger);

    // A scratch file is what a crash between write and rename leaves behind.
    let scratch = scratch_of(&tmp.anchor());
    std::fs::write(&scratch, b"interrupted publication garbage").unwrap();
    let reopened = AnchorStore::open(tmp.anchor(), key()).unwrap();
    assert_eq!(reopened.current(), before);
    drop(reopened);

    // The next publication replaces the scratch file instead of tripping over it.
    let (mut ledger, split) =
        AcknowledgedLedger::open(&tmp.paths(), key(), before.epoch, TailPolicy::Acknowledge)
            .unwrap();
    assert_eq!(split, Split::Aligned);
    ledger.append_acknowledged(revocation(2)).unwrap();
    assert_eq!(ledger.anchor().log.index, CommitIndex(2));
    // The rename consumes the scratch name, so no stale file accumulates and no
    // cleanup step has to be remembered after a crash.
    assert!(!scratch.exists());
    assert_eq!(
        snapshot(&tmp.anchor()).len(),
        ptr_ledger::anchor::ANCHOR_BYTES
    );
}

#[test]
fn acknowledged_appends_keep_the_pair_aligned_across_reopen() {
    let tmp = Temp::new();
    let ledger = make_ledger(&tmp, 3);
    let anchor = ledger.anchor();
    // One epoch for initialization plus one per acknowledged append.
    assert_eq!(ledger.epoch(), 4);
    assert_eq!(anchor.log.index, CommitIndex(3));
    assert!(anchor.origin.is_set());
    assert_eq!(subjects(ledger.events()).len(), 3);
    drop(ledger);

    let (reopened, split) =
        AcknowledgedLedger::open(&tmp.paths(), key(), anchor.epoch, TailPolicy::Acknowledge)
            .unwrap();
    assert_eq!(split, Split::Aligned);
    assert_eq!(reopened.anchor(), anchor);
    assert_eq!(
        subjects(reopened.events()),
        ["subject:1", "subject:2", "subject:3"]
    );
}

#[test]
fn durable_unacknowledged_records_are_kept_by_default() {
    let tmp = Temp::new();
    let ledger = make_ledger(&tmp, 2);
    let before = ledger.anchor();
    drop(ledger);
    append_without_acknowledging(&tmp.log(), revocation(3));

    let (split, _) = AcknowledgedLedger::inspect(&tmp.paths(), key(), 0).unwrap();
    assert!(matches!(
        split,
        Split::Unacknowledged {
            complete_records: 1,
            incomplete_frame: false,
            ..
        }
    ));
    assert!(split.is_recoverable());

    let (recovered, observed) =
        AcknowledgedLedger::open(&tmp.paths(), key(), before.epoch, TailPolicy::Acknowledge)
            .unwrap();
    assert_eq!(observed, split);
    // A revocation that reached the log stays in force after recovery.
    assert_eq!(
        subjects(recovered.events()),
        ["subject:1", "subject:2", "subject:3"]
    );
    assert_eq!(recovered.anchor().log.index, CommitIndex(3));
    assert!(recovered.epoch() > before.epoch);
    assert_eq!(recovered.anchor().origin, before.origin);
}

#[test]
fn discarding_durable_records_requires_an_explicit_policy() {
    let tmp = Temp::new();
    let ledger = make_ledger(&tmp, 2);
    let before = ledger.anchor();
    drop(ledger);
    append_without_acknowledging(&tmp.log(), revocation(3));
    let with_tail = snapshot(&tmp.log());

    // Refusing changes nothing at all.
    let refused = AcknowledgedLedger::open(&tmp.paths(), key(), before.epoch, TailPolicy::Refuse);
    assert_eq!(
        refused.err().map(|error| error.code()),
        Some("PTR_ACK_TAIL_REFUSED")
    );
    assert_eq!(snapshot(&tmp.log()), with_tail);

    let (trimmed, _) =
        AcknowledgedLedger::open(&tmp.paths(), key(), before.epoch, TailPolicy::Discard).unwrap();
    assert_eq!(subjects(trimmed.events()), ["subject:1", "subject:2"]);
    assert_eq!(trimmed.anchor().log, before.log);
    // Discard does not advance the anchor, because nothing new was acknowledged.
    assert_eq!(trimmed.epoch(), before.epoch);
    drop(trimmed);
    assert_eq!(
        snapshot(&tmp.log()),
        encode_log(&[committed(1), committed(2)]).unwrap()
    );
}

#[test]
fn an_incomplete_frame_is_never_a_record_under_any_policy() {
    for policy in [
        TailPolicy::Acknowledge,
        TailPolicy::Discard,
        TailPolicy::Refuse,
    ] {
        let tmp = Temp::new();
        let ledger = make_ledger(&tmp, 2);
        let before = ledger.anchor();
        drop(ledger);
        let complete = snapshot(&tmp.log());

        // A header-sized fragment of the next frame, as a torn write leaves it.
        let mut torn = complete.clone();
        torn.extend_from_slice(b"PTRFR002");
        std::fs::write(tmp.log(), &torn).unwrap();

        let (split, _) = AcknowledgedLedger::inspect(&tmp.paths(), key(), 0).unwrap();
        assert_eq!(
            split,
            Split::Unacknowledged {
                complete_records: 0,
                incomplete_frame: true,
                tail: before.log,
            }
        );

        if policy == TailPolicy::Refuse {
            assert!(AcknowledgedLedger::open(&tmp.paths(), key(), before.epoch, policy).is_err());
            assert_eq!(snapshot(&tmp.log()), torn);
            continue;
        }

        let (repaired, _) =
            AcknowledgedLedger::open(&tmp.paths(), key(), before.epoch, policy).unwrap();
        assert_eq!(subjects(repaired.events()), ["subject:1", "subject:2"]);
        assert_eq!(repaired.anchor(), before);
        drop(repaired);
        assert_eq!(snapshot(&tmp.log()), complete);
    }
}

#[test]
fn a_log_behind_its_anchor_is_refused_and_left_untouched() {
    let tmp = Temp::new();
    let ledger = make_ledger(&tmp, 3);
    let anchor = ledger.anchor();
    drop(ledger);

    // Lose the last committed record: a valid shorter chain, not corruption.
    let shortened = encode_log(&[committed(1), committed(2)]).unwrap();
    std::fs::write(tmp.log(), &shortened).unwrap();

    let (split, _) = AcknowledgedLedger::inspect(&tmp.paths(), key(), 0).unwrap();
    assert_eq!(
        split,
        Split::LostSuffix {
            acknowledged: CommitIndex(3),
            present: CommitIndex(2),
        }
    );
    assert!(!split.is_recoverable());

    for policy in [
        TailPolicy::Acknowledge,
        TailPolicy::Discard,
        TailPolicy::Refuse,
    ] {
        let error = AcknowledgedLedger::open(&tmp.paths(), key(), anchor.epoch, policy)
            .err()
            .unwrap();
        assert_eq!(error.code(), "PTR_ACK_LOST_SUFFIX");
        assert_eq!(
            error,
            AcknowledgedError::Unrecoverable(Split::LostSuffix {
                acknowledged: CommitIndex(3),
                present: CommitIndex(2),
            })
        );
        assert_eq!(snapshot(&tmp.log()), shortened);
    }
}

#[test]
fn a_rehashed_alternative_suffix_is_refused_although_its_chain_is_valid() {
    let tmp = Temp::new();
    let ledger = make_ledger(&tmp, 2);
    let anchor = ledger.anchor();
    drop(ledger);

    // Same origin, same length, different second record, chain rebuilt end to end.
    let substitute = CommittedEvent {
        index: CommitIndex(2),
        event: LedgerEvent::Revoked {
            subject: "attacker".into(),
            generation: Generation(2),
        },
    };
    let alternative = encode_log(&[committed(1), substitute]).unwrap();
    assert!(decode_log(&alternative).is_ok());
    std::fs::write(tmp.log(), &alternative).unwrap();

    let (split, _) = AcknowledgedLedger::inspect(&tmp.paths(), key(), 0).unwrap();
    assert_eq!(split, Split::Diverged);
    let error =
        AcknowledgedLedger::open(&tmp.paths(), key(), anchor.epoch, TailPolicy::Acknowledge)
            .err()
            .unwrap();
    assert_eq!(error.code(), "PTR_ACK_DIVERGED");
    assert_eq!(snapshot(&tmp.log()), alternative);
}

#[test]
fn an_anchor_from_another_log_is_refused_by_origin() {
    let ours = Temp::new();
    let theirs = Temp::new();
    let mine = make_ledger(&ours, 2);
    let mine_anchor = mine.anchor();
    drop(mine);

    let mut other = AcknowledgedLedger::create(&theirs.paths(), key()).unwrap();
    other.append_acknowledged(revocation(100)).unwrap();
    other.append_acknowledged(revocation(200)).unwrap();
    let other_anchor = other.anchor();
    drop(other);
    assert_ne!(mine_anchor.origin, other_anchor.origin);

    // Present our anchor beside their log.
    std::fs::copy(ours.anchor(), theirs.anchor()).unwrap();
    let (split, _) = AcknowledgedLedger::inspect(&theirs.paths(), key(), 0).unwrap();
    assert_eq!(split, Split::OriginMismatch);
    let error = AcknowledgedLedger::open(&theirs.paths(), key(), 0, TailPolicy::Acknowledge)
        .err()
        .unwrap();
    assert_eq!(error.code(), "PTR_ACK_ORIGIN_MISMATCH");
}

#[test]
fn a_failed_acknowledgment_fences_the_writer_and_leaves_a_recoverable_tail() {
    let tmp = Temp::new();
    let mut ledger = make_ledger(&tmp, 1);
    let before = ledger.anchor();

    // Occupying the scratch name with a directory makes publication fail without
    // touching the log, which is the ordering this protocol is built around.
    std::fs::create_dir(scratch_of(&tmp.anchor())).unwrap();
    let failure = ledger.append_acknowledged(revocation(2)).err().unwrap();
    assert_eq!(failure, AcknowledgedError::Anchor(AnchorError::Storage));

    // The record is durable; only its acknowledgment is missing.
    assert_eq!(
        ledger.append_acknowledged(revocation(3)).err(),
        Some(AcknowledgedError::Fenced)
    );
    assert_eq!(ledger.anchor(), before);
    drop(ledger);

    std::fs::remove_dir(scratch_of(&tmp.anchor())).unwrap();
    let (recovered, split) =
        AcknowledgedLedger::open(&tmp.paths(), key(), before.epoch, TailPolicy::Acknowledge)
            .unwrap();
    assert!(matches!(
        split,
        Split::Unacknowledged {
            complete_records: 1,
            ..
        }
    ));
    assert_eq!(subjects(recovered.events()), ["subject:1", "subject:2"]);
    assert_eq!(recovered.anchor().log.index, CommitIndex(2));
}

#[test]
fn anchor_advancement_is_monotonic_in_index_epoch_and_digest() {
    let tmp = Temp::new();
    let mut store =
        AnchorStore::initialize(tmp.anchor(), key(), ProtectedAnchor::initial()).unwrap();
    let origin = ChainOrigin::from_first_record([1; 32]);
    let at = |index: u64, digest: u8| LogAnchor {
        index: CommitIndex(index),
        digest: [digest; 32],
    };

    store.acknowledge(at(2, 0xaa), origin).unwrap();
    assert_eq!(store.epoch(), 2);
    assert_eq!(store.current().origin, origin);

    // Restating the same prefix is idempotent, but only at the same digest.
    store.acknowledge(at(2, 0xaa), origin).unwrap();
    assert_eq!(store.epoch(), 3);
    assert_eq!(
        store.acknowledge(at(2, 0xbb), origin).err(),
        Some(AnchorError::NonMonotonic)
    );
    // Going backwards is refused even with a correct digest for that index.
    assert_eq!(
        store.acknowledge(at(1, 0xaa), origin).err(),
        Some(AnchorError::NonMonotonic)
    );
    // An established origin cannot be rebound.
    let foreign = ChainOrigin::from_first_record([2; 32]);
    store.acknowledge(at(3, 0xcc), foreign).unwrap();
    assert_eq!(store.current().origin, origin);
    assert_eq!(store.current().log, at(3, 0xcc));
}

#[test]
fn epoch_exhaustion_refuses_all_advances_without_republishing() {
    let tmp = Temp::new();
    let exhausted = ProtectedAnchor {
        epoch: u64::MAX,
        ..ProtectedAnchor::initial()
    };
    let mut store = AnchorStore::initialize(tmp.anchor(), key(), exhausted).unwrap();
    let before = snapshot(&tmp.anchor());

    assert_eq!(
        store.acknowledge(
            LogAnchor {
                index: CommitIndex(1),
                digest: [0xaa; 32],
            },
            ChainOrigin::from_first_record([0xaa; 32]),
        ),
        Err(AnchorError::NonMonotonic)
    );
    assert_eq!(
        store.advance_compacted(
            LogAnchor {
                index: CommitIndex(1),
                digest: [0xaa; 32],
            },
            LogAnchor {
                index: CommitIndex(2),
                digest: [0xbb; 32],
            },
        ),
        Err(AnchorError::NonMonotonic)
    );

    assert_eq!(store.current(), exhausted);
    assert_eq!(snapshot(&tmp.anchor()), before);
}

#[test]
fn creating_a_pair_never_adopts_existing_files() {
    let tmp = Temp::new();
    let ledger = make_ledger(&tmp, 1);
    drop(ledger);

    // Both files present.
    assert!(AcknowledgedLedger::create(&tmp.paths(), key()).is_err());
    // Anchor present, log absent.
    std::fs::remove_file(tmp.log()).unwrap();
    assert_eq!(
        AcknowledgedLedger::create(&tmp.paths(), key())
            .err()
            .map(|error| error.code()),
        Some("PTR_ANCHOR_EXISTS")
    );
    assert!(!tmp.log().exists());
    // Log present, anchor absent: the log is not adopted either.
    std::fs::remove_file(tmp.anchor()).unwrap();
    FileLedger::create_new(tmp.log()).unwrap();
    assert!(AcknowledgedLedger::create(&tmp.paths(), key()).is_err());
    assert!(!tmp.anchor().exists());
}
