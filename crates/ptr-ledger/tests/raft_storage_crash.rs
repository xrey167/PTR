//! The durable state of a Raft member across a stop in the middle of a write: a torn tail of
//! the log, a snapshot installed or recorded and the process gone before the next file moved.
use ptr_ledger::raft_node::RaftNode;
use ptr_ledger::raft_storage::FileRaftStorage;
use ptr_ledger::LedgerEvent;
use ptr_types::{CapsuleId, Generation, ProjectId};
use raft::prelude::{Snapshot, SnapshotMetadata};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Failpoints are process-wide, so the tests of this file take turns.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-raft-crash-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn dir(&self) -> PathBuf {
        self.0.join("state")
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A single-voter member that led, committed `n` events and stopped; returns the last raft
/// index (the leader's own first entry is index 1).
fn with_entries(temp: &Temp, n: u64) -> u64 {
    let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
    node.campaign().unwrap();
    assert!(node.is_leader());
    for generation in 1..=n {
        node.propose(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(generation),
        })
        .unwrap();
    }
    assert_eq!(node.committed_events().len() as u64, n);
    drop(node);
    FileRaftStorage::open(&temp.dir(), &[1], &[])
        .unwrap()
        .wl()
        .last_index()
}

fn snapshot_at(index: u64) -> Snapshot {
    Snapshot {
        data: b"state".to_vec(),
        metadata: Some(SnapshotMetadata {
            index,
            term: 1,
            conf_state: None,
        }),
    }
}

/// A single-voter storage holding `n` appended entries with commit at `commit`; the file holds
/// `n` equally sized records.
fn storage_with_records(temp: &Temp, n: u64, commit: u64) -> u64 {
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    let mut core = storage.wl();
    core.set_hard_state(&raft::prelude::HardState {
        term: 1,
        vote: 1,
        commit,
    })
    .unwrap();
    let entries: Vec<raft::prelude::Entry> = (1..=n)
        .map(|index| raft::prelude::Entry {
            index,
            term: 1,
            data: vec![index as u8; 16],
            ..Default::default()
        })
        .collect();
    core.append(&entries).unwrap();
    let length = std::fs::metadata(temp.dir().join("log")).unwrap().len();
    (length - 8) / n
}

fn snapshot_files(temp: &Temp) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(temp.dir())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("snapshot"))
        .collect();
    names.sort();
    names
}

#[test]
fn a_torn_unsynced_tail_is_trimmed_reported_and_the_committed_prefix_is_kept() {
    let _turn = serial();
    for tail in [
        b"PTRRFR01".to_vec(),
        vec![0u8; 700],
        b"PTRRFR01\x07\0\0\0\0\0\0\0garbage that is not a record".to_vec(),
    ] {
        let temp = Temp::new("torn");
        let last = with_entries(&temp, 4);
        let log = temp.dir().join("log");
        let whole = std::fs::metadata(&log).unwrap().len();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&log)
            .unwrap()
            .write_all(&tail)
            .unwrap();
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        assert_eq!(storage.wl().last_index(), last);
        assert_eq!(
            std::fs::metadata(&log).unwrap().len(),
            whole,
            "the torn tail is cut off"
        );
        assert_eq!(storage.wl().trimmed_tail_bytes(), tail.len() as u64);
    }
}

#[test]
fn a_clean_log_reports_nothing_trimmed() {
    let _turn = serial();
    let temp = Temp::new("clean");
    with_entries(&temp, 3);
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(storage.wl().trimmed_tail_bytes(), 0);
}

#[test]
fn a_damaged_record_with_valid_records_after_it_is_refused_above_the_commit_too() {
    let _turn = serial();
    // Commit is at 1; records 2..4 were appended (and may have been acknowledged) above it.
    // Damage in record 2 with valid records 3 and 4 after it is not a torn tail.
    let temp = Temp::new("damage-then-valid");
    let size = storage_with_records(&temp, 4, 1);
    let log = temp.dir().join("log");
    let mut bytes = std::fs::read(&log).unwrap();
    bytes[(8 + size + 10) as usize] ^= 0xff;
    std::fs::write(&log, &bytes).unwrap();
    let refused = FileRaftStorage::open(&temp.dir(), &[1], &[]);
    assert!(
        refused.is_err(),
        "records that may have been acknowledged are not cut"
    );
    assert_eq!(
        std::fs::read(&log).unwrap(),
        bytes,
        "a refused log is left as it was found"
    );
}

#[test]
fn damage_in_the_last_record_alone_is_a_torn_tail_and_is_cut_and_reported() {
    let _turn = serial();
    let temp = Temp::new("damage-last");
    let size = storage_with_records(&temp, 4, 1);
    let log = temp.dir().join("log");
    let mut bytes = std::fs::read(&log).unwrap();
    bytes[(8 + 3 * size + 10) as usize] ^= 0xff;
    std::fs::write(&log, &bytes).unwrap();
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(storage.wl().last_index(), 3);
    assert_eq!(storage.wl().trimmed_tail_bytes(), size);
}

#[test]
fn damage_below_the_committed_index_is_refused_not_trimmed() {
    let _turn = serial();
    let temp = Temp::new("damaged");
    with_entries(&temp, 4);
    let log = temp.dir().join("log");
    let mut bytes = std::fs::read(&log).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&log, bytes).unwrap();
    assert!(
        FileRaftStorage::open(&temp.dir(), &[1], &[]).is_err(),
        "committed entries that cannot be read are not silently dropped"
    );
}

#[test]
fn a_log_cut_below_the_commit_of_a_pending_snapshot_is_refused() {
    let _turn = serial();
    // While a snapshot waits to be accounted for the member can hold committed entries above
    // it; losing one of them is refused as it is otherwise.
    let temp = Temp::new("pending-commit");
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        let mut core = storage.wl();
        core.apply_snapshot(snapshot_with_voter(10)).unwrap();
        core.append(&[raft::prelude::Entry {
            index: 11,
            term: 1,
            data: vec![7u8; 16],
            ..Default::default()
        }])
        .unwrap();
        core.set_commit(11).unwrap();
    }
    let log = temp.dir().join("log");
    let bytes = std::fs::read(&log).unwrap();
    std::fs::write(&log, &bytes[..bytes.len() - 5]).unwrap();
    assert!(FileRaftStorage::open(&temp.dir(), &[1], &[]).is_err());
}

fn snapshot_with_voter(index: u64) -> Snapshot {
    Snapshot {
        data: b"state".to_vec(),
        metadata: Some(SnapshotMetadata {
            index,
            term: 1,
            conf_state: Some(raft::prelude::ConfState {
                voters: vec![1],
                ..Default::default()
            }),
        }),
    }
}

#[test]
fn a_snapshot_payload_that_is_not_the_one_the_state_names_is_refused() {
    let _turn = serial();
    let temp = Temp::new("tampered-payload");
    with_entries(&temp, 3);
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage.wl().record_snapshot(2, b"state".to_vec()).unwrap();
    }
    let files = snapshot_files(&temp);
    assert_eq!(
        files.len(),
        1,
        "one payload, named by its digest: {files:?}"
    );
    std::fs::write(temp.dir().join(&files[0]), b"other bytes").unwrap();
    assert!(FileRaftStorage::open(&temp.dir(), &[1], &[]).is_err());
}

#[test]
fn a_count_of_u64_max_is_a_count_and_not_a_pending_snapshot() {
    let _turn = serial();
    let temp = Temp::new("max-count");
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage.wl().set_applied_events(u64::MAX).unwrap();
        assert!(!storage.wl().snapshot_pending());
    }
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert!(!storage.wl().snapshot_pending());
}

#[test]
fn a_snapshot_installed_over_an_old_log_leaves_a_state_a_restart_can_finish() {
    let _turn = serial();
    // The state that names the snapshot is written first; a stop before the log was reset
    // leaves the old log under it.
    let temp = Temp::new("install");
    with_entries(&temp, 3);
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage.wl().apply_snapshot(snapshot_at(10)).unwrap();
        assert!(storage.wl().snapshot_pending());
    }
    // Put the old log back, as if the process had stopped before it was reset.
    let with_old = Temp::new("install-old");
    with_entries(&with_old, 3);
    std::fs::copy(with_old.dir().join("log"), temp.dir().join("log")).unwrap();
    let node = RaftNode::open(&temp.dir(), 1, &[1]).expect("the member opens");
    assert!(node.committed_events().is_empty());
    let mut node = node;
    let installed = node.take_installed_snapshot().expect("offered again");
    assert_eq!(installed.index, 10);
    assert_eq!(installed.state, b"state");
}

#[test]
fn a_snapshot_recorded_before_the_log_was_compacted_leaves_a_state_a_restart_can_read() {
    let _turn = serial();
    let temp = Temp::new("record");
    let last = with_entries(&temp, 5);
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage.wl().record_snapshot(3, b"state".to_vec()).unwrap();
    }
    // The state names the snapshot at 3; put the whole old log back under it.
    let old = Temp::new("record-old");
    with_entries(&old, 5);
    std::fs::copy(old.dir().join("log"), temp.dir().join("log")).unwrap();
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(
        storage.wl().first_index(),
        4,
        "the covered entries are skipped"
    );
    assert_eq!(storage.wl().last_index(), last);
}

#[test]
fn a_node_reopened_over_a_log_ending_below_its_commit_is_refused() {
    let _turn = serial();
    let temp = Temp::new("lost-committed");
    with_entries(&temp, 4);
    let log = temp.dir().join("log");
    let bytes = std::fs::read(&log).unwrap();
    // Cut the file inside the last record: it was committed, and is gone.
    std::fs::write(&log, &bytes[..bytes.len() - 20]).unwrap();
    assert!(RaftNode::open(&temp.dir(), 1, &[1]).is_err());
}

#[cfg(feature = "failpoints")]
mod failpoints {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn crash_at(point: &str, run: impl FnOnce()) {
        let scenario = fail::FailScenario::setup();
        fail::cfg(point, "panic").unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(run));
        drop(scenario);
        assert!(outcome.is_err(), "{point} was reached");
    }

    use raft::Storage as _;

    /// The snapshot a member would serve, as (index, payload).
    fn served(storage: &FileRaftStorage) -> (u64, Vec<u8>) {
        let snapshot = storage.snapshot(0, 1).expect("a snapshot is served");
        (snapshot.get_metadata().index, snapshot.data.to_vec())
    }

    #[test]
    fn a_stop_at_each_step_of_installing_a_snapshot_leaves_a_member_that_opens_with_a_matching_pair(
    ) {
        let _turn = super::serial();
        // (point, whether the state naming the snapshot had landed)
        for (point, installed) in [
            ("raft.atomic.after_write_before_sync", false),
            ("raft.install.after_payload", false),
            ("raft.atomic.after_sync_before_rename", false),
            ("raft.state.after_rename_before_dir_sync", true),
            ("raft.install.after_log_sync_before_rename", true),
            ("raft.install.after_log_reset", true),
        ] {
            let temp = Temp::new("install-crash");
            with_entries(&temp, 3);
            let dir = temp.dir();
            crash_at(point, || {
                let storage = FileRaftStorage::open(&dir, &[1], &[]).unwrap();
                let _ = storage.wl().apply_snapshot(snapshot_with_voter(10));
            });
            let storage = FileRaftStorage::open(&temp.dir(), &[1], &[])
                .unwrap_or_else(|error| panic!("{point}: {error}"));
            let (index, payload) = served(&storage);
            if installed {
                assert_eq!((index, payload.as_slice()), (10, &b"state"[..]), "{point}");
                assert!(storage.wl().snapshot_pending(), "{point}");
                assert_eq!(storage.wl().first_index(), 11, "{point}");
                assert_eq!(snapshot_files(&temp).len(), 1, "{point}");
            } else {
                assert_eq!((index, payload.len()), (0, 0), "{point}: the old pair");
                assert!(!storage.wl().snapshot_pending(), "{point}");
                assert_eq!(storage.wl().first_index(), 1, "{point}");
                assert!(
                    snapshot_files(&temp).is_empty(),
                    "{point}: {:?}",
                    snapshot_files(&temp)
                );
            }
            drop(storage);
            let mut node = RaftNode::open(&temp.dir(), 1, &[1])
                .unwrap_or_else(|error| panic!("{point}: {error}"));
            let offered = node.take_installed_snapshot();
            assert_eq!(offered.is_some(), installed, "{point}");
            if installed {
                assert!(node.committed_events().is_empty(), "{point}");
            } else {
                assert_eq!(
                    node.committed_events().len(),
                    3,
                    "{point}: the old log is intact"
                );
            }
        }
    }

    #[test]
    fn a_stop_at_each_step_of_recording_a_snapshot_leaves_a_member_that_opens_and_counts() {
        let _turn = super::serial();
        // (point, whether the state naming the snapshot had landed)
        for (point, recorded) in [
            ("raft.atomic.after_write_before_sync", false),
            ("raft.snapshot.after_payload", false),
            ("raft.state.after_rename_before_dir_sync", true),
            ("raft.snapshot.after_state", true),
            ("raft.compact.after_sync_before_rename", true),
            ("raft.compact.after_rename_before_dir_sync", true),
            ("raft.snapshot.after_log_rewrite", true),
        ] {
            let temp = Temp::new("record-crash");
            with_entries(&temp, 5);
            let dir = temp.dir();
            crash_at(point, || {
                let storage = FileRaftStorage::open(&dir, &[1], &[]).unwrap();
                let _ = storage.wl().record_snapshot(3, b"state".to_vec());
            });
            let storage = FileRaftStorage::open(&temp.dir(), &[1], &[])
                .unwrap_or_else(|error| panic!("{point}: {error}"));
            let (index, payload) = served(&storage);
            if recorded {
                assert_eq!((index, payload.as_slice()), (3, &b"state"[..]), "{point}");
                assert_eq!(storage.wl().first_index(), 4, "{point}");
            } else {
                assert_eq!((index, payload.len()), (0, 0), "{point}: the old pair");
                assert_eq!(storage.wl().first_index(), 1, "{point}");
            }
            drop(storage);
            let node = RaftNode::open(&temp.dir(), 1, &[1])
                .unwrap_or_else(|error| panic!("{point}: {error}"));
            // Five events either way; the snapshot covers the two at raft indexes 2 and 3.
            assert_eq!(node.applied_events(), 5, "{point}");
            let held = if recorded { 3 } else { 5 };
            assert_eq!(node.committed_events().len(), held, "{point}");
        }
    }

    #[test]
    fn opening_a_log_syncs_what_it_read() {
        let _turn = super::serial();
        let temp = Temp::new("open-sync");
        with_entries(&temp, 3);
        let scenario = fail::FailScenario::setup();
        let hits = std::sync::Arc::new(AtomicU64::new(0));
        let counter = std::sync::Arc::clone(&hits);
        fail::cfg_callback("raft.open.log_sync", move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
        drop(FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap());
        drop(scenario);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "the log was synced once, at open"
        );
    }

    #[test]
    fn a_failed_snapshot_write_leaves_the_memory_as_the_disk_is() {
        let _turn = super::serial();
        for installing in [false, true] {
            let temp = Temp::new("failed-snapshot-write");
            with_entries(&temp, 5);
            let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
            let scenario = fail::FailScenario::setup();
            fail::cfg("raft.state.before_write", "1*return").unwrap();
            let outcome = if installing {
                storage.wl().apply_snapshot(snapshot_with_voter(10))
            } else {
                storage.wl().record_snapshot(3, b"state".to_vec())
            };
            drop(scenario);
            assert!(outcome.is_err(), "installing {installing}");
            assert_eq!(served(&storage), (0, Vec::new()), "installing {installing}");
            assert_eq!(storage.wl().first_index(), 1, "installing {installing}");
            assert!(!storage.wl().snapshot_pending(), "installing {installing}");
            let hard = storage.wl().hard_state().clone();
            drop(storage);
            let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
            assert_eq!(reopened.wl().hard_state(), &hard, "installing {installing}");
            assert_eq!(
                served(&reopened),
                (0, Vec::new()),
                "installing {installing}"
            );
        }
    }

    #[test]
    fn an_io_error_in_a_write_is_an_error_and_not_a_silent_success() {
        let _turn = super::serial();
        let temp = Temp::new("io-error");
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        let scenario = fail::FailScenario::setup();
        fail::cfg("raft.append.before_write", "1*return").unwrap();
        let failed = storage.wl().append(&[raft::prelude::Entry {
            index: 1,
            term: 1,
            ..Default::default()
        }]);
        fail::cfg("raft.state.before_write", "1*return").unwrap();
        let state_failed = storage.wl().set_commit(0);
        drop(scenario);
        assert!(failed.is_err());
        assert!(state_failed.is_err());
        assert_eq!(storage.wl().last_index(), 0, "nothing was appended");
    }
}
