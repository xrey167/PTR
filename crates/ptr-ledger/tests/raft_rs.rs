//! Raft state that a restart must not lose, and a log that must never splice two
//! histories together.
//!
//! The single-node group proves nothing about consensus — that is Gate 2's
//! remaining work. What is testable here is the half of Raft's safety argument
//! that lives on disk: a node that forgets its term can join a stale one, a node
//! that forgets its vote can vote twice in one term, and a node that forgets
//! committed entries can acknowledge a history it no longer has.

use ptr_ledger::raft_storage::FileRaftStorage;
use ptr_ledger::{LedgerEvent, SingleNodeRaftConsensus};
use ptr_types::{CapsuleId, Generation, ProjectId};
use raft::prelude::{Entry, HardState};
use raft::Storage;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-raft-{label}-{}-{}",
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

fn capsule(generation: u64) -> LedgerEvent {
    LedgerEvent::CapsuleCommitted {
        project: ProjectId::from("p"),
        capsule: CapsuleId::from("capsule:a"),
        generation: Generation(generation),
    }
}

fn entry(index: u64, term: u64, data: &[u8]) -> Entry {
    Entry {
        index,
        term,
        data: data.to_vec(),
        ..Entry::default()
    }
}

#[test]
fn single_node_raft_commits_ptr_events_in_order() {
    let temp = Temp::new("order");
    let mut consensus = SingleNodeRaftConsensus::open(&temp.dir(), 1).unwrap();
    assert!(consensus.is_leader());

    let first = consensus.propose(capsule(1)).unwrap();
    let second = consensus
        .propose(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();

    assert_eq!(first.commit_index.0, 1);
    assert_eq!(second.commit_index.0, 2);
    assert!(second.raft_index > first.raft_index);
    assert_eq!(consensus.committed_events().len(), 2);
}

#[test]
fn a_restarted_node_restores_exactly_the_committed_state() {
    let temp = Temp::new("restart");
    let expected = {
        let mut consensus = SingleNodeRaftConsensus::open(&temp.dir(), 1).unwrap();
        consensus.propose(capsule(1)).unwrap();
        consensus.propose(capsule(2)).unwrap();
        consensus.committed_events().to_vec()
    };
    assert_eq!(expected.len(), 2);

    let mut reopened = SingleNodeRaftConsensus::open(&temp.dir(), 1).unwrap();
    assert_eq!(
        reopened.committed_events(),
        expected.as_slice(),
        "a reopened node must hold exactly the history it committed, index by index"
    );

    // And it continues the same sequence rather than starting over.
    let third = reopened.propose(capsule(3)).unwrap();
    assert_eq!(third.commit_index.0, 3);
}

#[test]
fn the_term_and_the_vote_a_node_has_seen_survive_the_process() {
    let temp = Temp::new("term");
    let before = {
        let consensus = SingleNodeRaftConsensus::open(&temp.dir(), 7).unwrap();
        consensus.term()
    };
    assert!(before > 0, "a node that campaigned has seen a term");

    // Read the durable state directly: this is the file a restart depends on.
    let storage = FileRaftStorage::open(&temp.dir(), &[7], &[]).unwrap();
    let state = storage.initial_state().unwrap();
    assert_eq!(state.hard_state.term, before);
    assert_eq!(
        state.hard_state.vote, 7,
        "a node that campaigned voted for itself, and that vote is what must not be forgotten"
    );
    assert_eq!(state.conf_state.voters, vec![7]);
    drop(storage);

    let after = SingleNodeRaftConsensus::open(&temp.dir(), 7).unwrap();
    assert!(
        after.term() > before,
        "a new election is a later term, never an earlier one: {} then {}",
        before,
        after.term()
    );
}

#[test]
fn an_existing_node_keeps_its_recorded_configuration_rather_than_a_caller_s() {
    // Replacing a node's recorded membership with whatever the caller passes is
    // how a node rejoins a group it was removed from.
    let temp = Temp::new("conf");
    FileRaftStorage::open(&temp.dir(), &[1, 2, 3], &[4]).unwrap();
    let reopened = FileRaftStorage::open(&temp.dir(), &[9], &[]).unwrap();
    let state = reopened.initial_state().unwrap();
    assert_eq!(state.conf_state.voters, vec![1, 2, 3]);
    assert_eq!(state.conf_state.learners, vec![4]);
}

#[test]
fn a_conflicting_append_truncates_rather_than_splicing_two_histories() {
    let temp = Temp::new("truncate");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage
        .wl()
        .append(&[
            entry(1, 1, b"one"),
            entry(2, 1, b"two"),
            entry(3, 1, b"three"),
        ])
        .unwrap();
    assert_eq!(storage.last_index().unwrap(), 3);

    // A new leader's entry at 2 replaces 2 and everything after it.
    storage.wl().append(&[entry(2, 2, b"other")]).unwrap();
    assert_eq!(storage.last_index().unwrap(), 2);
    assert_eq!(storage.term(2).unwrap(), 2);
    drop(storage);

    // On disk, not merely in memory: the reopened log holds one history.
    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(reopened.last_index().unwrap(), 2);
    assert_eq!(reopened.term(1).unwrap(), 1);
    assert_eq!(reopened.term(2).unwrap(), 2);
    let entries = reopened
        .entries(1, 3, None, raft::GetEntriesContext::empty(false))
        .unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].data, b"other".to_vec());
}

#[test]
fn a_gap_and_an_overwrite_of_compacted_history_are_both_refused() {
    let temp = Temp::new("gap");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage.wl().append(&[entry(1, 1, b"one")]).unwrap();

    let gap = storage
        .wl()
        .append(&[entry(3, 1, b"three")])
        .expect_err("an index past the end leaves a gap");
    assert!(gap.to_string().contains("gap"), "{gap}");

    let ragged = storage
        .wl()
        .append(&[entry(2, 1, b"two"), entry(4, 1, b"four")])
        .expect_err("entries must be consecutive");
    assert!(ragged.to_string().contains("consecutive"), "{ragged}");

    // Nothing partial was written by either refusal.
    assert_eq!(storage.last_index().unwrap(), 1);
}

#[test]
fn a_record_whose_digest_does_not_match_is_refused_on_open() {
    let temp = Temp::new("digest");
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage
            .wl()
            .append(&[entry(1, 1, b"one"), entry(2, 1, b"two")])
            .unwrap();
    }
    let log = temp.dir().join("log");
    let original = std::fs::read(&log).unwrap();

    // A bit flipped in a payload: every field still parses, and only the digest
    // says the record is not what was written.
    let mut payload_damaged = original.clone();
    let inside_payload = original.len() - 33;
    payload_damaged[inside_payload] ^= 0x01;
    std::fs::write(&log, &payload_damaged).unwrap();
    let error = FileRaftStorage::open(&temp.dir(), &[1], &[])
        .expect_err("a damaged record is not readable state");
    assert!(error.to_string().contains("digest"), "{error}");

    // A bit flipped in a length field fails earlier, while reading: a different
    // fault with a different refusal, rather than one message for everything.
    let mut length_damaged = original.clone();
    let inside_length = original.len() - 40;
    length_damaged[inside_length] ^= 0x01;
    std::fs::write(&log, &length_damaged).unwrap();
    let error = FileRaftStorage::open(&temp.dir(), &[1], &[])
        .expect_err("a length that runs past the file is not readable state");
    assert!(error.to_string().contains("ends inside"), "{error}");

    std::fs::write(&log, &original).unwrap();
    FileRaftStorage::open(&temp.dir(), &[1], &[]).expect("the intact log still opens");
}

#[test]
fn a_record_removed_from_the_middle_is_refused_even_though_each_record_is_intact() {
    // Records are not hash-chained, because a Raft log legitimately loses its tail
    // to truncation. Continuity is what protects the middle, so it has to be
    // checked rather than assumed.
    let temp = Temp::new("removed");
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage
            .wl()
            .append(&[
                entry(1, 1, b"aaa"),
                entry(2, 1, b"bbb"),
                entry(3, 1, b"ccc"),
            ])
            .unwrap();
    }
    let log = temp.dir().join("log");
    let bytes = std::fs::read(&log).unwrap();
    // Equal payload sizes make the three records equal in length, so the middle
    // one can be cut out without re-encoding anything.
    let header = 8;
    let record = (bytes.len() - header) / 3;
    assert_eq!(bytes.len(), header + record * 3);
    let mut spliced = bytes[..header + record].to_vec();
    spliced.extend_from_slice(&bytes[header + record * 2..]);
    std::fs::write(&log, &spliced).unwrap();

    let error = FileRaftStorage::open(&temp.dir(), &[1], &[])
        .expect_err("a log that skips an index is not a log");
    assert!(error.to_string().contains("jumps from 2 to 3"), "{error}");
}

#[test]
fn a_truncated_log_and_a_damaged_state_file_are_both_refused() {
    let temp = Temp::new("torn");
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        storage.wl().append(&[entry(1, 1, b"one")]).unwrap();
    }
    let log = temp.dir().join("log");
    let bytes = std::fs::read(&log).unwrap();
    for length in [9, 20, bytes.len() - 1] {
        std::fs::write(&log, &bytes[..length]).unwrap();
        let error = FileRaftStorage::open(&temp.dir(), &[1], &[])
            .expect_err("a torn record is not readable state");
        let message = error.to_string();
        assert!(
            message.contains("ends inside") || message.contains("magic"),
            "{length}: {message}"
        );
    }
    std::fs::write(&log, &bytes).unwrap();
    FileRaftStorage::open(&temp.dir(), &[1], &[]).expect("the intact log still opens");

    let state = temp.dir().join("state");
    let mut state_bytes = std::fs::read(&state).unwrap();
    state_bytes[10] ^= 0x40;
    std::fs::write(&state, &state_bytes).unwrap();
    let error = FileRaftStorage::open(&temp.dir(), &[1], &[])
        .expect_err("a state file that does not match its digest names a term nobody held");
    assert!(error.to_string().contains("digest"), "{error}");
}

#[test]
fn a_snapshot_compacts_the_log_and_the_entries_it_covers_leave_the_file() {
    let temp = Temp::new("snapshot");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage
        .wl()
        .append(&[
            entry(1, 1, b"one"),
            entry(2, 1, b"two"),
            entry(3, 1, b"three"),
        ])
        .unwrap();
    storage
        .wl()
        .set_hard_state(&HardState {
            term: 1,
            vote: 1,
            commit: 2,
        })
        .unwrap();
    storage
        .wl()
        .record_snapshot(2, b"state-at-2".to_vec())
        .unwrap();

    assert_eq!(storage.first_index().unwrap(), 3);
    assert_eq!(storage.last_index().unwrap(), 3);
    // The term at the snapshot's own index is still answerable, which is what
    // matching a follower's log against it needs.
    assert_eq!(storage.term(2).unwrap(), 1);
    assert_eq!(
        storage.term(1).expect_err("index 1 is compacted away"),
        raft::Error::Store(raft::StorageError::Compacted)
    );
    drop(storage);

    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(reopened.first_index().unwrap(), 3);
    assert_eq!(reopened.last_index().unwrap(), 3);
    let snapshot = reopened.snapshot(2, 1).unwrap();
    assert_eq!(snapshot.get_metadata().index, 2);
    assert_eq!(snapshot.get_metadata().term, 1);
    assert_eq!(snapshot.data, b"state-at-2".to_vec());
    assert_eq!(snapshot.get_metadata().get_conf_state().voters, vec![1]);
}

#[test]
fn a_snapshot_older_than_requested_is_refused_rather_than_fabricated() {
    // MemStorage answers this by inventing a snapshot at the requested index,
    // because it *is* the state machine. A durable store is not, and a payload
    // that does not match its claimed position would be sent to a follower as
    // truth.
    let temp = Temp::new("unavailable");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage.wl().append(&[entry(1, 1, b"one")]).unwrap();
    assert_eq!(
        storage
            .snapshot(5, 1)
            .expect_err("there is no snapshot at 5"),
        raft::Error::Store(raft::StorageError::SnapshotTemporarilyUnavailable)
    );
}

#[test]
fn a_snapshot_ahead_of_the_committed_index_is_refused() {
    let temp = Temp::new("ahead");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage
        .wl()
        .append(&[entry(1, 1, b"one"), entry(2, 1, b"two")])
        .unwrap();
    let error = storage
        .wl()
        .record_snapshot(2, b"too-early".to_vec())
        .expect_err("nothing is committed yet, so nothing can be snapshotted");
    assert!(
        error.to_string().contains("ahead of the committed"),
        "{error}"
    );
}

#[test]
fn prost_codec_preserves_entry_wire_bytes_and_rejects_malformed_payloads() {
    use raft::codec::Message as _;
    let mut entry = raft::eraftpb::Entry::default();
    entry.set_term(7);
    entry.set_index(9);
    entry.set_data(vec![1, 2, 3]);
    let golden = vec![0x10, 7, 0x18, 9, 0x22, 3, 1, 2, 3];
    assert_eq!(entry.write_to_bytes().unwrap(), golden);
    assert_eq!(entry.compute_size() as usize, golden.len());
    let mut decoded = raft::eraftpb::Entry::default();
    decoded.merge_from_bytes(&golden).unwrap();
    assert_eq!(decoded, entry);
    for malformed in [vec![0x22, 0xff], vec![0x10, 0x80], vec![0xff; 32]] {
        let mut decoded = raft::eraftpb::Entry::default();
        assert!(decoded.merge_from_bytes(&malformed).is_err());
    }
}
