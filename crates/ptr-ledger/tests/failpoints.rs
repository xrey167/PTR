use fail::FailScenario;
use ptr_ledger::acknowledged::{AcknowledgedLedger, Split, TailPolicy};
use ptr_ledger::anchor::{AnchorKey, AnchorStore};
use ptr_ledger::compaction::LogPaths;
use ptr_ledger::{FileLedger, LedgerEvent};
use ptr_types::Generation;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ptr-ledger-failpoint-{nonce}.log"))
}

#[test]
fn panic_after_length_prefix_recovers_without_phantom_event() {
    let path = temp_path();
    let scenario = FailScenario::setup();

    let mut ledger = FileLedger::open(&path).unwrap();
    fail::cfg("ledger.after_length_before_payload", "panic").unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        ledger
            .append_durable(LedgerEvent::Revoked {
                subject: "capsule:a".into(),
                generation: Generation(7),
            })
            .unwrap();
    }));
    assert!(result.is_err());

    drop(ledger);
    scenario.teardown();

    assert!(FileLedger::open(&path).is_err());
    let recovered =
        FileLedger::recover_unacknowledged_tail(&path, ptr_ledger::integrity::LogAnchor::empty())
            .unwrap();
    assert!(recovered.events().is_empty());
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        ptr_ledger::integrity::LOG_MAGIC.len() as u64
    );

    drop(recovered);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn failed_append_poison_survives_unwind_until_reopen() {
    for (point, durable_count) in [
        ("ledger.before_record_write", 0),
        ("ledger.after_length_before_payload", 0),
        ("ledger.after_payload_before_sync", 1),
        ("ledger.after_sync_before_memory", 1),
    ] {
        let path = temp_path();
        let scenario = FailScenario::setup();
        let event = LedgerEvent::Revoked {
            subject: "capsule:poison".into(),
            generation: Generation(9),
        };
        let mut ledger = FileLedger::open(&path).unwrap();
        fail::cfg(point, "panic").unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| ledger.append_durable(event.clone()))).is_err());
        fail::remove(point);
        let length = std::fs::metadata(&path).unwrap().len();
        assert!(
            ledger.append_durable(event.clone()).is_err(),
            "{point} must poison the handle"
        );
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            length,
            "poisoned writes must not touch disk"
        );
        assert!(ledger.events().is_empty());
        drop(ledger);
        scenario.teardown();
        let mut recovered = if point == "ledger.after_length_before_payload" {
            assert!(FileLedger::open(&path).is_err());
            FileLedger::recover_unacknowledged_tail(
                &path,
                ptr_ledger::integrity::LogAnchor::empty(),
            )
            .unwrap()
        } else {
            FileLedger::open(&path).unwrap()
        };
        // These are unwind tests, not power-loss tests: a fully written record
        // is visible to reopen even when the original call never acknowledged it.
        assert_eq!(recovered.events().len(), durable_count);
        let index = recovered.append_durable(event).unwrap();
        assert_eq!(index.0, durable_count as u64 + 1);
        drop(recovered);
        std::fs::remove_file(path).unwrap();
    }
}

/// The four points of one anchor publication, in the order the steps run.
const ANCHOR_POINTS: [&str; 4] = [
    "anchor.after_scratch_write_before_sync",
    "anchor.after_scratch_sync_before_rename",
    "anchor.after_rename_before_dir_sync",
    "anchor.after_dir_sync",
];

fn temp_dir(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ptr-ledger-anchor-fp-{label}-{nonce}"));
    std::fs::create_dir(&dir).unwrap();
    dir
}

fn key() -> AnchorKey {
    AnchorKey::from_bytes([0x3c; 32])
}

fn revocation(n: u64) -> LedgerEvent {
    LedgerEvent::Revoked {
        subject: format!("capsule:{n}"),
        generation: Generation(n),
    }
}

#[test]
fn one_anchor_publication_hits_each_of_its_four_points_once_in_order() {
    let dir = temp_dir("order");
    let paths = LogPaths::new(&dir, "journal");
    let scenario = FailScenario::setup();
    let mut ledger = AcknowledgedLedger::create(&paths, key()).unwrap();
    let hits = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    for point in ANCHOR_POINTS {
        let hits = Arc::clone(&hits);
        fail::cfg_callback(point, move || hits.lock().unwrap().push(point)).unwrap();
    }
    ledger.append_acknowledged(revocation(1)).unwrap();
    assert_eq!(*hits.lock().unwrap(), ANCHOR_POINTS);
    scenario.teardown();
    drop(ledger);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_panic_at_each_anchor_point_leaves_the_old_or_the_new_anchor_never_a_blend() {
    // Before the rename the old record is authoritative; from the rename on, the
    // new one is. Either way the record the log already holds is kept on reopen
    // and nothing acknowledged is lost.
    for (point, anchor_has_new) in ANCHOR_POINTS.into_iter().zip([false, false, true, true]) {
        let dir = temp_dir("panic");
        let paths = LogPaths::new(&dir, "journal");
        let scenario = FailScenario::setup();
        let mut ledger = AcknowledgedLedger::create(&paths, key()).unwrap();
        ledger.append_acknowledged(revocation(1)).unwrap();
        fail::cfg(point, "panic").unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            ledger.append_acknowledged(revocation(2)).unwrap();
        }));
        assert!(result.is_err(), "{point} must interrupt the append");
        fail::remove(point);
        drop(ledger);
        scenario.teardown();

        let (split, anchor) = AcknowledgedLedger::inspect(&paths, key(), 0).unwrap();
        assert_eq!(
            anchor.log.index.0,
            if anchor_has_new { 2 } else { 1 },
            "{point}"
        );
        if anchor_has_new {
            assert_eq!(split, Split::Aligned, "{point}");
        } else {
            assert!(
                matches!(
                    split,
                    Split::Unacknowledged {
                        complete_records: 1,
                        ..
                    }
                ),
                "{point}: {split:?}"
            );
        }
        let (reopened, _) =
            AcknowledgedLedger::open(&paths, key(), 0, TailPolicy::Acknowledge).unwrap();
        assert_eq!(
            reopened.events().len(),
            2,
            "{point}: the synced record survives"
        );
        assert_eq!(reopened.anchor().log.index.0, 2, "{point}");
        drop(reopened);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn the_scratch_file_of_an_interrupted_publication_is_never_authoritative() {
    let dir = temp_dir("scratch");
    let paths = LogPaths::new(&dir, "journal");
    let scenario = FailScenario::setup();
    let mut ledger = AcknowledgedLedger::create(&paths, key()).unwrap();
    fail::cfg(ANCHOR_POINTS[0], "panic").unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        ledger.append_acknowledged(revocation(1)).unwrap();
    }))
    .is_err());
    fail::remove(ANCHOR_POINTS[0]);
    drop(ledger);
    scenario.teardown();

    let leftovers = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".publishing"))
        .count();
    assert_eq!(
        leftovers, 1,
        "the interrupted publication left its scratch file"
    );
    let store = AnchorStore::open(paths.anchor_path(), key()).unwrap();
    assert_eq!(
        store.current().log.index.0,
        0,
        "the anchor on record is the old one"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_directory_is_synchronized_at_creation_and_at_every_anchor_publication() {
    // Creating the log syncs its directory once, the first anchor once more, and each
    // acknowledged append publishes one anchor.
    let dir = temp_dir("syncparent");
    let paths = LogPaths::new(&dir, "journal");
    let scenario = FailScenario::setup();
    let hits = Arc::new(Mutex::new(0usize));
    {
        // Only this thread's calls: the other tests of this binary run beside it.
        let hits = Arc::clone(&hits);
        let owner = std::thread::current().id();
        fail::cfg_callback("ledger.sync_parent", move || {
            if std::thread::current().id() == owner {
                *hits.lock().unwrap() += 1;
            }
        })
        .unwrap();
    }
    let mut ledger = AcknowledgedLedger::create(&paths, key()).unwrap();
    assert_eq!(*hits.lock().unwrap(), 2);
    ledger.append_acknowledged(revocation(1)).unwrap();
    ledger.append_acknowledged(revocation(2)).unwrap();
    assert_eq!(*hits.lock().unwrap(), 4);
    scenario.teardown();
    drop(ledger);
    std::fs::remove_dir_all(dir).unwrap();
}

/// A crash between the truncation of a zero tail and its sync leaves a log that the next
/// recovery trims to the same state (the trim is idempotent).
#[test]
fn a_crash_during_the_trim_of_a_zero_tail_recovers_to_the_same_state() {
    let dir = std::env::temp_dir().join(format!(
        "ptr-ledger-trim-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    // Set up first: the failpoint registry is global and other tests configure it.
    let scenario = FailScenario::setup();
    let paths = LogPaths::new(&dir, "journal");
    let key = || AnchorKey::from_bytes([3; 32]);
    let mut ledger = AcknowledgedLedger::create(&paths, key()).unwrap();
    for n in 1..=2u64 {
        ledger
            .append_acknowledged(LedgerEvent::Revoked {
                subject: format!("subject:{n}"),
                generation: Generation(n),
            })
            .unwrap();
    }
    let epoch = ledger.anchor().epoch;
    drop(ledger);
    let log = paths.log_path(ptr_types::CommitIndex(0));
    let complete = std::fs::read(&log).unwrap();
    let mut torn = complete.clone();
    torn.extend_from_slice(&[0u8; 400]);
    std::fs::write(&log, &torn).unwrap();

    fail::cfg("ledger.trim_after_truncate_before_sync", "panic").unwrap();
    let crashed = catch_unwind(AssertUnwindSafe(|| {
        let _ = AcknowledgedLedger::open(&paths, key(), epoch, TailPolicy::Acknowledge);
    }));
    assert!(crashed.is_err());
    scenario.teardown();
    // In-process the truncation stays visible in the page cache, so the file is the complete
    // log; the unwind must not have corrupted anything.
    assert_eq!(std::fs::read(&log).unwrap(), complete);
    let (recovered, _) =
        AcknowledgedLedger::open(&paths, key(), epoch, TailPolicy::Acknowledge).unwrap();
    assert_eq!(recovered.events().len(), 2);
    drop(recovered);
    assert_eq!(std::fs::read(&log).unwrap(), complete);
    // A power loss would also have lost the truncation: write the torn bytes back over the log
    // (the state the crash can leave on disk) and recover from that. The end state is the same.
    std::fs::write(&log, &torn).unwrap();
    let (replayed, _) =
        AcknowledgedLedger::open(&paths, key(), epoch, TailPolicy::Acknowledge).unwrap();
    assert_eq!(replayed.events().len(), 2);
    assert_eq!(replayed.recovery().unwrap().trimmed_zero_tail_bytes, 400);
    drop(replayed);
    assert_eq!(std::fs::read(&log).unwrap(), complete);
    std::fs::remove_dir_all(&dir).unwrap();
}
