//! Raft state that a restart must not lose, and a log that must never splice two
//! histories together.
//!
//! The single-node group proves nothing about consensus — that is Gate 2's
//! remaining work. What is testable here is the half of Raft's safety argument
//! that lives on disk: a node that forgets its term can join a stale one, a node
//! that forgets its vote can vote twice in one term, and a node that forgets
//! committed entries can acknowledge a history it no longer has.

use ptr_ledger::integrity::MAX_RECORD_BYTES;
use ptr_ledger::raft_storage::FileRaftStorage;
use ptr_ledger::{
    Attestation, LedgerEvent, RetentionFloor, SemanticOrigin, SingleNodeRaftConsensus,
};
use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId, Revision, VerificationLevel};
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
        .record_snapshot(
            2,
            b"state-at-2".to_vec(),
            RetentionFloor::covering(CommitIndex(2)),
        )
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
        .record_snapshot(
            2,
            b"too-early".to_vec(),
            RetentionFloor::covering(CommitIndex(2)),
        )
        .expect_err("nothing is committed yet, so nothing can be snapshotted");
    assert!(
        error.to_string().contains("ahead of the committed"),
        "{error}"
    );
}

/// Three committed entries, each carrying an event, and nothing compacted yet.
fn three_committed_events(temp: &Temp) -> FileRaftStorage {
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
            commit: 3,
        })
        .unwrap();
    storage
}

/// The storage's files, for asserting a refusal left nothing behind.
fn files(temp: &Temp) -> (Vec<u8>, Vec<u8>, bool) {
    let read = |name: &str| std::fs::read(temp.dir().join(name)).unwrap_or_default();
    (
        read("state"),
        read("log"),
        temp.dir().join("snapshot").exists(),
    )
}

fn floor(events: u64) -> RetentionFloor {
    RetentionFloor::covering(CommitIndex(events))
}

#[test]
fn a_compaction_past_the_retention_floor_is_refused_and_moves_nothing() {
    // The retained snapshot covers two events. Compacting to index 3 would discard
    // the third, which nothing else retains: a follower that needs it could be
    // neither replayed to nor sent a snapshot holding it.
    let temp = Temp::new("past-floor");
    let storage = three_committed_events(&temp);
    let before = files(&temp);
    let refused = storage
        .wl()
        .record_snapshot(3, b"state-at-2".to_vec(), floor(2))
        .expect_err("index 3 is past what the retained snapshot covers");
    assert!(
        refused
            .to_string()
            .starts_with("PTR_RAFT_COMPACTION_ABOVE_RETENTION_FLOOR"),
        "{refused}"
    );
    assert_eq!(
        files(&temp),
        before,
        "neither file moves and no payload lands"
    );
    assert_eq!(storage.first_index().unwrap(), 1);
    assert_eq!(storage.wl().applied_events(), 0);
    drop(storage);

    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(reopened.first_index().unwrap(), 1);
    assert_eq!(
        reopened
            .entries(1, 4, None, raft::GetEntriesContext::empty(false))
            .unwrap()
            .len(),
        3,
        "every entry is still there to replay"
    );
}

#[test]
fn a_compaction_up_to_the_retention_floor_discards_exactly_what_it_covers() {
    // The control for the refusal above: the same log, a floor that covers the
    // prefix, and the entries below it are gone while the one above stays.
    let temp = Temp::new("to-floor");
    let storage = three_committed_events(&temp);
    storage
        .wl()
        .record_snapshot(2, b"state-at-2".to_vec(), floor(2))
        .expect("the retained snapshot covers both discarded events");
    assert_eq!(storage.first_index().unwrap(), 3);
    for gone in [1, 2] {
        assert_eq!(
            storage
                .entries(gone, 4, None, raft::GetEntriesContext::empty(false))
                .expect_err("below the floor"),
            raft::Error::Store(raft::StorageError::Compacted)
        );
    }
    assert_eq!(
        storage
            .entries(3, 4, None, raft::GetEntriesContext::empty(false))
            .unwrap(),
        vec![entry(3, 1, b"three")]
    );
    assert_eq!(storage.wl().applied_events(), 2);
    drop(storage);

    // The coverage is recorded with the position, so a reopened member numbers
    // the next event from it rather than from zero.
    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(reopened.first_index().unwrap(), 3);
    assert_eq!(reopened.wl().applied_events(), 2);
}

#[test]
fn a_snapshot_claiming_events_its_position_had_not_produced_is_refused() {
    // A member that installs this payload holds it *at* index 2 and is replayed
    // from 3. A payload that already contains the third event would have it
    // applied twice.
    let temp = Temp::new("claims-more");
    let storage = three_committed_events(&temp);
    let before = files(&temp);
    let refused = storage
        .wl()
        .record_snapshot(2, b"state-at-3".to_vec(), floor(3))
        .expect_err("the payload covers more than index 2 produced");
    assert!(
        refused
            .to_string()
            .starts_with("PTR_RAFT_SNAPSHOT_FLOOR_MISMATCH"),
        "{refused}"
    );
    assert_eq!(files(&temp), before);
    assert_eq!(storage.first_index().unwrap(), 1);
}

#[test]
fn the_retention_floor_is_counted_in_ledger_events_not_raft_indexes() {
    // A leader's first entry in a term is empty and a configuration change is not
    // an event, so raft index 4 here is ledger event 2. A host that passed the
    // raft index as the floor would be claiming two events nobody committed.
    let temp = Temp::new("events");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    let mut conf_change = entry(2, 1, b"not an event");
    conf_change.set_entry_type(raft::prelude::EntryType::EntryConfChange);
    storage
        .wl()
        .append(&[
            entry(1, 1, b""),
            conf_change,
            entry(3, 1, b"first event"),
            entry(4, 1, b"second event"),
        ])
        .unwrap();
    storage
        .wl()
        .set_hard_state(&HardState {
            term: 1,
            vote: 1,
            commit: 4,
        })
        .unwrap();

    let refused = storage
        .wl()
        .record_snapshot(4, b"state".to_vec(), floor(4))
        .expect_err("the raft index is not the ledger index");
    assert!(
        refused
            .to_string()
            .starts_with("PTR_RAFT_SNAPSHOT_FLOOR_MISMATCH"),
        "{refused}"
    );
    let refused = storage
        .wl()
        .record_snapshot(4, b"state".to_vec(), floor(1))
        .expect_err("one event short");
    assert!(
        refused
            .to_string()
            .starts_with("PTR_RAFT_COMPACTION_ABOVE_RETENTION_FLOOR"),
        "{refused}"
    );
    // Everything before the first event carried none, so a snapshot covering no
    // events may discard it.
    storage
        .wl()
        .record_snapshot(2, b"empty state".to_vec(), floor(0))
        .expect("indexes 1 and 2 produced no event");
    storage
        .wl()
        .record_snapshot(4, b"state".to_vec(), floor(2))
        .expect("two events through index 4, counted from the earlier snapshot");
    assert_eq!(storage.first_index().unwrap(), 5);
    assert_eq!(storage.wl().applied_events(), 2);
}

/// A snapshot a leader sent, describing its state at index 5.
fn sent_at_five() -> raft::prelude::Snapshot {
    raft::prelude::Snapshot {
        data: b"a leader's state at 5".to_vec(),
        metadata: Some(raft::prelude::SnapshotMetadata {
            index: 5,
            term: 1,
            conf_state: Some(raft::prelude::ConfState {
                voters: vec![1],
                ..Default::default()
            }),
        }),
    }
}

#[test]
fn an_installed_snapshot_blocks_compaction_until_the_application_accounts_for_it() {
    // An installed payload covers a number of events only the application can
    // read. Until it says how many, the count below the snapshot is the old
    // position's, and any floor checked against it would be checked against the
    // wrong base.
    let temp = Temp::new("unaccounted");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage.wl().apply_snapshot(sent_at_five()).unwrap();
    storage
        .wl()
        .append(&[entry(6, 1, b"six"), entry(7, 1, b"seven")])
        .unwrap();
    storage.wl().set_commit(7).unwrap();

    let before = files(&temp);
    let refused = storage
        .wl()
        .record_snapshot(7, b"state".to_vec(), floor(2))
        .expect_err("no base to count from");
    assert!(
        refused
            .to_string()
            .starts_with("PTR_RAFT_SNAPSHOT_UNACCOUNTED"),
        "{refused}"
    );
    assert_eq!(files(&temp), before);

    // The application reads the payload: it covered ten events.
    storage.wl().set_applied_events(10).unwrap();
    let refused = storage
        .wl()
        .record_snapshot(7, b"state".to_vec(), floor(11))
        .expect_err("indexes 6 and 7 add two events to ten");
    assert!(
        refused
            .to_string()
            .starts_with("PTR_RAFT_COMPACTION_ABOVE_RETENTION_FLOOR"),
        "{refused}"
    );
    storage
        .wl()
        .record_snapshot(7, b"state".to_vec(), floor(12))
        .expect("counted from the accounted base");
    assert_eq!(storage.first_index().unwrap(), 8);
}

#[test]
fn an_unaccounted_snapshot_is_still_unaccounted_after_a_restart() {
    // Held in memory alone, a restart between installing and accounting would come
    // back with the previous position's count and nothing to say it was stale.
    let temp = Temp::new("unaccounted-restart");
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage.wl().apply_snapshot(sent_at_five()).unwrap();
    storage.wl().append(&[entry(6, 1, b"six")]).unwrap();
    storage.wl().set_commit(6).unwrap();
    drop(storage);

    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert!(reopened.wl().events_unaccounted());
    let refused = reopened
        .wl()
        .record_snapshot(6, b"state".to_vec(), floor(1))
        .expect_err("the count of the position before the snapshot is not a base");
    assert!(
        refused
            .to_string()
            .starts_with("PTR_RAFT_SNAPSHOT_UNACCOUNTED"),
        "{refused}"
    );

    // Accounting clears it durably, and only then.
    reopened.wl().set_applied_events(3).unwrap();
    drop(reopened);
    let accounted = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert!(!accounted.wl().events_unaccounted());
    assert_eq!(accounted.wl().applied_events(), 3);
    accounted
        .wl()
        .record_snapshot(6, b"state".to_vec(), floor(4))
        .expect("three covered by the installed payload and one at index 6");
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

fn host_write(required: VerificationLevel, verifiers: &[&str]) -> LedgerEvent {
    LedgerEvent::SemanticDeltaCommitted {
        base_revision: Revision(0),
        revision: Revision(1),
        encoded_delta: vec![1, 2, 3],
        origin: SemanticOrigin::Host {
            principal: "operator".into(),
            verification: Attestation {
                required,
                level: VerificationLevel::FullSemantic,
                verifiers: verifiers.iter().map(|name| (*name).to_owned()).collect(),
                findings: Vec::new(),
            },
        },
    }
}

#[test]
fn an_origin_the_decoder_refuses_is_never_proposed() {
    // Proposed, it would be a committed entry no member can apply, on every
    // member and after every restart.
    let temp = Temp::new("origin-bound");
    let accepted = host_write(VerificationLevel::FullSemantic, &["schema"]);
    let refused = [
        (
            host_write(VerificationLevel::FullSemantic, &[]),
            "PTR_LEDGER_ATTESTATION_LIMIT",
        ),
        (
            host_write(VerificationLevel::SampleVerified, &["schema"]),
            "PTR_LEDGER_ATTESTATION_REQUIREMENT",
        ),
        (
            // The tag, both revisions and the delta's length take 21 bytes.
            LedgerEvent::SemanticDeltaCommitted {
                base_revision: Revision(0),
                revision: Revision(1),
                encoded_delta: vec![0; MAX_RECORD_BYTES - 20],
                origin: SemanticOrigin::Legacy,
            },
            "PTR_LOG_PAYLOAD_LIMIT",
        ),
    ];
    {
        let mut consensus = SingleNodeRaftConsensus::open(&temp.dir(), 1).unwrap();
        for (event, code) in refused {
            assert_eq!(consensus.propose(event).unwrap_err(), code);
            assert!(consensus.committed_events().is_empty());
        }
        let receipt = consensus.propose(accepted.clone()).unwrap();
        assert_eq!(receipt.commit_index.0, 1);
    }
    let reopened = SingleNodeRaftConsensus::open(&temp.dir(), 1).unwrap();
    assert_eq!(reopened.committed_events().len(), 1);
    assert_eq!(reopened.committed_events()[0].event, accepted);
}
