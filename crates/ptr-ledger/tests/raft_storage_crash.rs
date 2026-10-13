//! The durable state of a Raft member across a stop in the middle of a write: a torn tail of
//! the log, a snapshot installed or recorded and the process gone before the next file moved.
use ptr_ledger::raft_node::{NoSnapshotPending, NodeFailed, RaftNode};
use ptr_ledger::raft_storage::{FileRaftStorage, MAX_SNAPSHOT_BYTES};
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

#[test]
fn a_log_cut_at_a_record_boundary_below_the_commit_is_refused_and_does_not_panic() {
    let _turn = serial();
    // A lost suffix or an older copy of the file ends cleanly: no torn record to notice.
    let temp = Temp::new("boundary-cut");
    let size = storage_with_records(&temp, 4, 4);
    let log = temp.dir().join("log");
    let bytes = std::fs::read(&log).unwrap();
    std::fs::write(&log, &bytes[..(8 + 3 * size) as usize]).unwrap();
    let refused = FileRaftStorage::open(&temp.dir(), &[1], &[]).expect_err("committed entry lost");
    assert!(
        refused.to_string().contains("below the committed index"),
        "{refused}"
    );
    assert!(RaftNode::open(&temp.dir(), 1, &[1]).is_err());
    // The same cut at or above the commit index is a shorter log, not a lost one.
    let temp = Temp::new("boundary-cut-above");
    let size = storage_with_records(&temp, 4, 2);
    let log = temp.dir().join("log");
    let bytes = std::fs::read(&log).unwrap();
    std::fs::write(&log, &bytes[..(8 + 3 * size) as usize]).unwrap();
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(storage.wl().last_index(), 3);
}

#[test]
fn files_past_their_size_bound_are_refused_before_they_are_read() {
    let _turn = serial();
    // The state file.
    let temp = Temp::new("big-state");
    with_entries(&temp, 2);
    let state = std::fs::OpenOptions::new()
        .write(true)
        .open(temp.dir().join("state"))
        .unwrap();
    state.set_len(2 * 1024 * 1024).unwrap();
    drop(state);
    let refused = FileRaftStorage::open(&temp.dir(), &[1], &[]).expect_err("oversized state");
    assert!(refused.to_string().contains("size bound"), "{refused}");

    // The retained snapshot payload (sparse: no disk is used).
    let temp = Temp::new("big-payload");
    with_entries(&temp, 3);
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage.wl().record_snapshot(2, b"state".to_vec()).unwrap();
    }
    let files = snapshot_files(&temp);
    let payload = std::fs::OpenOptions::new()
        .write(true)
        .open(temp.dir().join(&files[0]))
        .unwrap();
    payload.set_len(MAX_SNAPSHOT_BYTES as u64 + 1).unwrap();
    drop(payload);
    let refused = FileRaftStorage::open(&temp.dir(), &[1], &[]).expect_err("oversized payload");
    assert!(refused.to_string().contains("size bound"), "{refused}");

    // The log (sparse).
    let temp = Temp::new("big-log");
    with_entries(&temp, 2);
    let log = std::fs::OpenOptions::new()
        .write(true)
        .open(temp.dir().join("log"))
        .unwrap();
    log.set_len(ptr_ledger::raft_storage::MAX_LOG_BYTES + 1)
        .unwrap();
    drop(log);
    let refused = FileRaftStorage::open(&temp.dir(), &[1], &[]).expect_err("oversized log");
    assert!(refused.to_string().contains("size bound"), "{refused}");
}

#[test]
fn an_append_that_would_pass_the_log_bound_is_refused_before_a_byte_moves() {
    let _turn = serial();
    let temp = Temp::new("log-bound");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    let before = std::fs::read(temp.dir().join("log")).unwrap();
    // Entries of the largest allowed size, until the file would pass its bound.
    let big = |index: u64| raft::prelude::Entry {
        index,
        term: 1,
        data: vec![0u8; ptr_ledger::raft_storage::MAX_ENTRY_BYTES],
        ..Default::default()
    };
    let count = ptr_ledger::raft_storage::MAX_LOG_BYTES
        / ptr_ledger::raft_storage::MAX_ENTRY_BYTES as u64
        + 1;
    let entries: Vec<_> = (1..=count).map(big).collect();
    let refused = storage.wl().append(&entries).expect_err("past the bound");
    assert!(refused.to_string().contains("size bound"), "{refused}");
    assert_eq!(std::fs::read(temp.dir().join("log")).unwrap(), before);
    assert!(
        !storage.wl().is_poisoned(),
        "a refusal before any write is not a poisoning"
    );
}

#[test]
fn leftover_scratch_files_are_removed_at_open_and_foreign_files_are_not() {
    let _turn = serial();
    let temp = Temp::new("leftovers");
    with_entries(&temp, 2);
    let dir = temp.dir();
    for name in ["state.tmp", "log.rewrite", "snapshot-notes.txt"] {
        std::fs::write(dir.join(name), b"left over").unwrap();
    }
    let orphan = format!("snapshot-{}", "ab".repeat(32));
    std::fs::write(dir.join(&orphan), b"orphan payload").unwrap();
    std::fs::write(dir.join(format!("{orphan}.tmp")), b"half payload").unwrap();
    drop(FileRaftStorage::open(&dir, &[1], &[]).unwrap());
    assert!(!dir.join("state.tmp").exists());
    assert!(!dir.join("log.rewrite").exists());
    assert!(!dir.join(&orphan).exists());
    assert!(!dir.join(format!("{orphan}.tmp")).exists());
    assert!(
        dir.join("snapshot-notes.txt").exists(),
        "a file this module did not write is not its to remove"
    );
}

#[test]
fn an_applied_count_at_the_top_of_the_range_is_refused_and_does_not_overflow() {
    let _turn = serial();
    // The count is host input. Counting one more event past u64::MAX is a refusal.
    let temp = Temp::new("overflow");
    FileRaftStorage::open(&temp.dir(), &[1], &[])
        .unwrap()
        .wl()
        .apply_snapshot(snapshot_with_voter(10))
        .unwrap();
    let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
    assert!(node.take_installed_snapshot().is_some());
    node.resume_with_applied(u64::MAX).unwrap();
    node.campaign().unwrap();
    let refused = node
        .propose(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .expect_err("one event past u64::MAX");
    assert!(refused.contains("exhausted"), "{refused}");

    // The storage counts the events a snapshot covers the same way.
    let temp = Temp::new("overflow-storage");
    storage_with_records(&temp, 3, 3);
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage.wl().set_applied_events(u64::MAX).unwrap();
    assert!(storage.wl().record_snapshot(2, b"state".to_vec()).is_err());
    assert_eq!(storage.wl().first_index(), 1, "nothing was compacted");
}

#[test]
fn a_log_with_records_and_no_state_file_is_refused_and_left_as_it_was() {
    let _turn = serial();
    let temp = Temp::new("state-missing");
    with_entries(&temp, 2);
    std::fs::remove_file(temp.dir().join("state")).unwrap();
    let before = std::fs::read(temp.dir().join("log")).unwrap();
    let refused = FileRaftStorage::open(&temp.dir(), &[1], &[]).expect_err("records without state");
    assert!(
        refused.to_string().contains("state file is missing"),
        "{refused}"
    );
    assert_eq!(std::fs::read(temp.dir().join("log")).unwrap(), before);
    assert!(!temp.dir().join("state").exists());

    // A log holding only its header (a stop during the first open) is started over.
    let temp = Temp::new("state-missing-header-only");
    drop(FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap());
    std::fs::remove_file(temp.dir().join("state")).unwrap();
    FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
}

#[test]
fn resume_with_applied_is_refused_when_no_snapshot_is_pending_and_the_count_stays() {
    let _turn = serial();
    let temp = Temp::new("resume-nothing-pending");
    with_entries(&temp, 3);
    let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
    assert_eq!(node.applied_events(), 3);
    let refused = node.resume_with_applied(9).expect_err("nothing is pending");
    assert!(refused.contains(NoSnapshotPending::CODE), "{refused}");
    assert_eq!(node.applied_events(), 3, "the count did not move");
    assert!(!node.is_failed(), "a refused call is not a failure");
    drop(node);
    let reopened = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
    assert_eq!(
        reopened.applied_events(),
        3,
        "the durable count did not move"
    );

    // A retry after a success is refused the same way and renumbers nothing.
    let temp = Temp::new("resume-twice");
    FileRaftStorage::open(&temp.dir(), &[1], &[])
        .unwrap()
        .wl()
        .apply_snapshot(snapshot_with_voter(10))
        .unwrap();
    let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
    node.resume_with_applied(5).unwrap();
    assert!(node.resume_with_applied(7).is_err());
    assert_eq!(node.applied_events(), 5);
}

#[test]
fn losing_held_entries_while_resuming_fails_the_member() {
    let _turn = serial();
    let temp = Temp::new("resume-partial");
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        let mut core = storage.wl();
        core.apply_snapshot(snapshot_with_voter(10)).unwrap();
        // An entry above the snapshot that is committed and does not decode as an event.
        core.append(&[raft::prelude::Entry {
            index: 11,
            term: 1,
            data: vec![0xff; 8],
            ..Default::default()
        }])
        .unwrap();
        core.set_commit(11).unwrap();
    }
    let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
    assert!(node.take_installed_snapshot().is_some());
    let failed = node
        .resume_with_applied(5)
        .expect_err("the held entry does not decode");
    assert!(
        !failed.contains(NodeFailed::CODE),
        "the first error is the cause: {failed}"
    );
    assert!(node.is_failed());
    assert!(
        !node.is_poisoned(),
        "the storage is sound; the member is not"
    );
    for refused in [
        node.tick().map(drop),
        node.campaign().map(drop),
        node.resume_with_applied(5),
    ] {
        let error = refused.expect_err("every state-changing call is refused");
        assert!(error.contains(NodeFailed::CODE), "{error}");
    }
}

#[cfg(feature = "failpoints")]
mod failpoints {
    use super::*;
    use ptr_ledger::raft_storage::StoragePoisoned;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Whether an I/O error is the typed refusal of a poisoned storage.
    fn is_poisoned_refusal(error: &std::io::Error) -> bool {
        error
            .get_ref()
            .is_some_and(|inner| inner.is::<StoragePoisoned>())
    }

    fn entry_at(index: u64) -> raft::prelude::Entry {
        raft::prelude::Entry {
            index,
            term: 1,
            data: vec![index as u8; 16],
            ..Default::default()
        }
    }

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

    /// Run `change` with the failpoint configured, then drop the scenario.
    fn with_failpoint<T>(point: &str, action: &str, change: impl FnOnce() -> T) -> T {
        let scenario = fail::FailScenario::setup();
        fail::cfg(point, action).unwrap();
        let outcome = change();
        drop(scenario);
        outcome
    }

    #[test]
    fn an_error_past_the_point_of_no_return_poisons_the_storage_and_no_append_is_acknowledged() {
        let _turn = super::serial();
        // Each point is past a rename, a truncation or a partial write; the storage must refuse
        // every later write, and a restart must hold every entry acknowledged before.
        for point in [
            "raft.compact.after_sync_before_rename",
            "raft.compact.after_rename_before_dir_sync",
            "raft.state.after_rename_before_dir_sync",
        ] {
            let temp = Temp::new("poison-compact");
            let last = with_entries(&temp, 5);
            let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
            let failed = with_failpoint(point, "return", || {
                storage.wl().record_snapshot(3, b"state".to_vec())
            });
            assert!(failed.is_err(), "{point}");
            assert!(storage.wl().is_poisoned(), "{point}");
            // The handle may be on an unlinked file or the offsets may belong to the other
            // file: an append here would be acknowledged into the wrong place.
            let refused = storage.wl().append(&[entry_at(last + 1)]);
            assert!(
                refused.as_ref().is_err_and(is_poisoned_refusal),
                "{point}: {refused:?}"
            );
            assert!(storage
                .wl()
                .set_hard_state(&raft::prelude::HardState {
                    term: 9,
                    vote: 1,
                    commit: 1
                })
                .as_ref()
                .is_err_and(is_poisoned_refusal));
            assert!(storage
                .wl()
                .record_snapshot(4, Vec::new())
                .as_ref()
                .is_err_and(is_poisoned_refusal));
            drop(storage);
            // Nothing acknowledged is lost: the member reopens with all five events.
            let node = RaftNode::open(&temp.dir(), 1, &[1])
                .unwrap_or_else(|error| panic!("{point}: {error}"));
            assert_eq!(node.applied_events(), 5, "{point}");
            let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
            assert_eq!(reopened.wl().last_index(), last, "{point}");
        }
    }

    #[test]
    fn a_failed_state_rename_sync_poisons_and_a_restart_finds_one_consistent_state() {
        let _turn = super::serial();
        let temp = Temp::new("poison-state");
        let last = with_entries(&temp, 3);
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        let failed = with_failpoint("raft.state.after_rename_before_dir_sync", "return", || {
            storage.wl().set_commit(last)
        });
        assert!(failed.is_err());
        assert!(storage.wl().is_poisoned());
        assert!(storage
            .wl()
            .append(&[entry_at(last + 1)])
            .as_ref()
            .is_err_and(is_poisoned_refusal));
        drop(storage);
        let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        assert_eq!(reopened.wl().last_index(), last);
        assert!(!reopened.wl().is_poisoned(), "a restart starts sound");
    }

    #[test]
    fn a_write_that_fails_part_way_poisons_and_loses_nothing_acknowledged() {
        let _turn = super::serial();
        // (point, whether the new entries conflict with held ones, so that the file was
        // shortened first)
        for (point, conflicting) in [
            ("raft.append.partial_write", false),
            ("raft.append.partial_write", true),
            ("raft.append.after_write_before_sync", false),
            ("raft.append.after_write_before_sync", true),
        ] {
            let temp = Temp::new("poison-append");
            storage_with_records(&temp, 4, 2);
            let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
            // Entries 3 and 4 are above the commit index (2); a new leader may replace them.
            let first = if conflicting { 3 } else { 5 };
            let new: Vec<_> = (first..first + 2).map(entry_at).collect();
            let failed = with_failpoint(point, "return", || storage.wl().append(&new));
            assert!(failed.is_err(), "{point} {conflicting}");
            assert!(storage.wl().is_poisoned(), "{point} {conflicting}");
            let refused = storage.wl().append(&[entry_at(first + 2)]);
            assert!(
                refused.as_ref().is_err_and(is_poisoned_refusal),
                "{point} {conflicting}: {refused:?}"
            );
            drop(storage);
            let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[])
                .unwrap_or_else(|error| panic!("{point} {conflicting}: {error}"));
            let core = reopened.wl();
            assert!(
                core.last_index() >= 2,
                "{point} {conflicting}: the committed prefix is intact"
            );
            assert!(core.committed_entries().len() == 2, "{point} {conflicting}");
            if point == "raft.append.partial_write" {
                // The torn half-record is cut off at open and reported.
                assert!(core.trimmed_tail_bytes() > 0, "{point}");
            }
        }
    }

    #[test]
    fn an_error_installing_a_snapshot_past_the_state_write_poisons_and_a_restart_finishes() {
        let _turn = super::serial();
        for point in [
            "raft.install.after_log_sync_before_rename",
            "raft.install.after_log_reset",
        ] {
            let temp = Temp::new("poison-install");
            with_entries(&temp, 3);
            let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
            let failed = with_failpoint(point, "return", || {
                storage.wl().apply_snapshot(snapshot_with_voter(10))
            });
            assert!(failed.is_err(), "{point}");
            assert!(storage.wl().is_poisoned(), "{point}");
            assert!(storage
                .wl()
                .append(&[entry_at(11)])
                .as_ref()
                .is_err_and(is_poisoned_refusal));
            drop(storage);
            let mut node = RaftNode::open(&temp.dir(), 1, &[1])
                .unwrap_or_else(|error| panic!("{point}: {error}"));
            assert_eq!(node.take_installed_snapshot().map(|s| s.index), Some(10));
        }
    }

    #[test]
    fn a_poisoned_member_answers_nothing_and_a_restart_holds_what_it_acknowledged() {
        let _turn = super::serial();
        let temp = Temp::new("poison-node");
        let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
        node.campaign().unwrap();
        let mut acknowledged = 0;
        for generation in 1..=3 {
            node.propose(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from("capsule:a"),
                generation: Generation(generation),
            })
            .unwrap();
            acknowledged += 1;
        }
        assert!(!node.is_poisoned());
        let outcome = with_failpoint("raft.state.after_rename_before_dir_sync", "return", || {
            node.propose(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from("capsule:a"),
                generation: Generation(4),
            })
        });
        assert!(outcome.is_err(), "the proposal is not acknowledged");
        assert!(node.is_poisoned());
        for refused in [
            node.tick().map(drop),
            node.campaign().map(drop),
            node.propose(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from("capsule:a"),
                generation: Generation(5),
            })
            .map(drop),
            node.record_snapshot(2, Vec::new()),
            node.resume_with_applied(1),
        ] {
            let error = refused.expect_err("every call is refused");
            assert!(error.contains(StoragePoisoned::CODE), "{error}");
        }
        assert_eq!(node.committed_events().len(), acknowledged);
        drop(node);
        let reopened = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
        assert!(!reopened.is_poisoned());
        assert!(
            reopened.committed_events().len() >= acknowledged,
            "every acknowledged event is still there"
        );
    }

    #[test]
    fn an_error_after_a_ready_that_does_not_poison_still_fails_the_member() {
        let _turn = super::serial();
        // (failpoint, calls to let through first): an append refused before a byte moved and a
        // hard-state write refused before its rename.
        for (point, skip) in [
            ("raft.append.before_write", 0),
            ("raft.state.before_write", 0),
        ] {
            let temp = Temp::new("node-failed");
            let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
            node.campaign().unwrap();
            let event = |generation| LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from("capsule:a"),
                generation: Generation(generation),
            };
            node.propose(event(1)).unwrap();
            let acknowledged = node.committed_events().len();
            let outcome = with_failpoint(point, &format!("{skip}*off->1*return"), || {
                node.propose(event(2))
            });
            assert!(outcome.is_err(), "{point} {skip}: not acknowledged");
            assert!(
                !node.is_poisoned(),
                "{point} {skip}: these errors do not poison"
            );
            assert!(node.is_failed(), "{point} {skip}");
            for refused in [
                node.tick().map(drop),
                node.campaign().map(drop),
                node.propose(event(3)).map(drop),
                node.record_snapshot(2, Vec::new()),
                node.report_unreachable(2).map(drop),
            ] {
                let error = refused.expect_err("every state-changing call is refused");
                assert!(error.contains(NodeFailed::CODE), "{point} {skip}: {error}");
            }
            assert_eq!(
                node.committed_events().len(),
                acknowledged,
                "{point} {skip}"
            );
            drop(node);
            let reopened = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
            assert!(!reopened.is_failed(), "a restart starts sound");
            assert!(
                reopened.committed_events().len() >= acknowledged,
                "{point} {skip}"
            );
        }
    }

    #[test]
    fn a_failed_resume_leaves_the_snapshot_offered_and_the_count_as_it_was() {
        let _turn = super::serial();
        let temp = Temp::new("resume-rollback");
        with_entries(&temp, 3);
        {
            let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
            storage
                .wl()
                .apply_snapshot(snapshot_with_voter(10))
                .unwrap();
        }
        let mut node = RaftNode::open(&temp.dir(), 1, &[1]).unwrap();
        assert!(node.take_installed_snapshot().is_some());
        let failed = with_failpoint("raft.state.before_write", "1*return", || {
            node.resume_with_applied(5)
        });
        assert!(failed.is_err());
        assert!(
            node.take_installed_snapshot().is_some(),
            "the vetted snapshot is still offered"
        );
        assert_eq!(node.applied_events(), 0, "the count did not move");
        // The call can be repeated.
        node.resume_with_applied(5).unwrap();
        assert!(node.take_installed_snapshot().is_none());
        assert_eq!(node.applied_events(), 5);
    }

    #[test]
    fn an_error_syncing_the_log_at_open_is_an_error_and_not_a_member_answering_for_it() {
        let _turn = super::serial();
        let temp = Temp::new("open-sync-error");
        with_entries(&temp, 3);
        let refused = with_failpoint("raft.open.log_sync", "return", || {
            FileRaftStorage::open(&temp.dir(), &[1], &[])
        });
        assert!(refused.is_err(), "an unsynced log is not answered for");
    }
}
