//! Raft state that a restart must not lose, and a log that must never splice two
//! histories together.
//!
//! The single-node group proves nothing about consensus — that is Gate 2's
//! remaining work. What is testable here is the half of Raft's safety argument
//! that lives on disk: a node that forgets its term can join a stale one, a node
//! that forgets its vote can vote twice in one term, and a node that forgets
//! committed entries can acknowledge a history it no longer has.

use ptr_ledger::integrity::MAX_RECORD_BYTES;
use ptr_ledger::raft_storage::{FileRaftStorage, SnapshotFloorRefusal};
use ptr_ledger::{
    Attestation, LedgerEvent, RetentionFloor, SemanticOrigin, SingleNodeRaftConsensus,
};
use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId, Revision, VerificationLevel};
use raft::prelude::{ConfState, Entry, EntryType, HardState, Snapshot, SnapshotMetadata};
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
        // Committed, so that a record that cannot be read is missing committed history and not
        // an unacknowledged tail (which open cuts off and reports).
        storage.wl().set_commit(2).unwrap();
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
    assert!(error.to_string().contains("committed"), "{error}");

    // A bit flipped in a length field fails earlier, while reading; the refusal is the same
    // because either way a committed record is gone.
    let mut length_damaged = original.clone();
    let inside_length = original.len() - 40;
    length_damaged[inside_length] ^= 0x01;
    std::fs::write(&log, &length_damaged).unwrap();
    let error = FileRaftStorage::open(&temp.dir(), &[1], &[])
        .expect_err("a length that runs past the file is not readable state");
    assert!(error.to_string().contains("committed"), "{error}");

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
        storage.wl().set_commit(1).unwrap();
    }
    let log = temp.dir().join("log");
    let bytes = std::fs::read(&log).unwrap();
    for length in [9, 20, bytes.len() - 1] {
        std::fs::write(&log, &bytes[..length]).unwrap();
        let error = FileRaftStorage::open(&temp.dir(), &[1], &[])
            .expect_err("a torn record is not readable state");
        let message = error.to_string();
        assert!(
            message.contains("committed") || message.contains("magic"),
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

// Raft's log floor against the ledger's retention floor.

fn typed_entry(index: u64, kind: EntryType, data: &[u8]) -> Entry {
    Entry {
        entry_type: kind as i32,
        ..entry(index, 1, data)
    }
}

/// A committed log whose raft indexes and ledger events disagree: a leader's empty entry
/// (1), two events (2, 3), a configuration change (4) and a third event (5). Through raft
/// index 4 the prefix produced two events; through 5, three.
fn log_with_non_event_entries(temp: &Temp) -> FileRaftStorage {
    let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    storage
        .wl()
        .append(&[
            typed_entry(1, EntryType::EntryNormal, b""),
            typed_entry(2, EntryType::EntryNormal, b"two"),
            typed_entry(3, EntryType::EntryNormal, b"three"),
            typed_entry(4, EntryType::EntryConfChange, b"conf"),
            typed_entry(5, EntryType::EntryNormal, b"five"),
        ])
        .unwrap();
    storage
        .wl()
        .set_hard_state(&HardState {
            term: 1,
            vote: 1,
            commit: 5,
        })
        .unwrap();
    storage
}

fn floor(events: u64) -> RetentionFloor {
    RetentionFloor::covering(CommitIndex(events))
}

fn refusal_of(error: &std::io::Error) -> SnapshotFloorRefusal {
    *error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<SnapshotFloorRefusal>())
        .unwrap_or_else(|| panic!("not a floor refusal: {error}"))
}

#[test]
fn compacting_short_of_the_retained_floor_is_refused_and_leaves_the_log_whole() {
    let temp = Temp::new("floor-above");
    let storage = log_with_non_event_entries(&temp);
    let error = storage
        .wl()
        .record_snapshot_within(4, b"state".to_vec(), floor(1))
        .expect_err("two events would leave raft with only one of them retained");
    assert_eq!(
        refusal_of(&error),
        SnapshotFloorRefusal::AboveRetentionFloor {
            index: 4,
            events: CommitIndex(2),
            floor: CommitIndex(1),
        }
    );
    assert!(
        error
            .to_string()
            .starts_with("PTR_RAFT_COMPACTION_ABOVE_RETENTION_FLOOR"),
        "{error}"
    );
    assert_eq!(storage.first_index().unwrap(), 1, "nothing was discarded");
    assert!(!storage.wl().is_poisoned(), "a refusal is not a fault");
    drop(storage);

    // Nothing reached the disk either: the reopened log still holds every entry and no
    // snapshot is on offer.
    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(reopened.first_index().unwrap(), 1);
    assert_eq!(reopened.snapshot(0, 1).unwrap().get_metadata().index, 0);
}

#[test]
fn a_payload_claiming_more_than_the_prefix_produced_is_refused() {
    let temp = Temp::new("floor-mismatch");
    let storage = log_with_non_event_entries(&temp);
    let error = storage
        .wl()
        .record_snapshot_within(4, b"state".to_vec(), floor(3))
        .expect_err("the third event is above index 4, so a member would apply it twice");
    assert_eq!(
        refusal_of(&error),
        SnapshotFloorRefusal::FloorMismatch {
            index: 4,
            events: CommitIndex(2),
            floor: CommitIndex(3),
        }
    );
    assert!(
        error
            .to_string()
            .starts_with("PTR_RAFT_SNAPSHOT_FLOOR_MISMATCH"),
        "{error}"
    );
    assert_eq!(storage.first_index().unwrap(), 1);
}

#[test]
fn the_floor_is_counted_in_ledger_events_not_raft_indexes() {
    let temp = Temp::new("floor-exact");
    let storage = log_with_non_event_entries(&temp);
    // The raft index is the wrong number for the same position, in both directions.
    for wrong in [4, 5] {
        storage
            .wl()
            .record_snapshot_within(4, b"state".to_vec(), floor(wrong))
            .expect_err("a raft index is not a ledger position");
    }
    storage
        .wl()
        .record_snapshot_within(4, b"state-at-4".to_vec(), floor(2))
        .expect("exactly the events the prefix produced");
    assert_eq!(storage.first_index().unwrap(), 5);
    assert_eq!(storage.last_index().unwrap(), 5);
    drop(storage);

    // The coverage is persisted with the snapshot: the member reopens numbered from it,
    // so the next snapshot's floor is two plus the one event above, not one.
    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    assert_eq!(reopened.first_index().unwrap(), 5);
    reopened
        .wl()
        .record_snapshot_within(5, b"state-at-5".to_vec(), floor(1))
        .expect_err("a count restarted from zero is the bug this refuses");
    reopened
        .wl()
        .record_snapshot_within(5, b"state-at-5".to_vec(), floor(3))
        .unwrap();
    assert_eq!(
        reopened.snapshot(0, 1).unwrap().data,
        b"state-at-5".to_vec()
    );
}

#[test]
fn a_floor_of_zero_covers_a_prefix_with_no_events() {
    let temp = Temp::new("floor-zero");
    let storage = log_with_non_event_entries(&temp);
    storage
        .wl()
        .record_snapshot_within(1, Vec::new(), floor(0))
        .expect("the leader's empty entry produced no event");
    assert_eq!(storage.first_index().unwrap(), 2);
}

#[test]
fn the_self_attested_record_snapshot_still_counts_by_the_same_rule() {
    // `record_snapshot` takes its floor from the log, so it cannot be refused for a floor;
    // what it must do is persist the same count `record_snapshot_within` would.
    let temp = Temp::new("floor-self");
    let storage = log_with_non_event_entries(&temp);
    storage.wl().record_snapshot(4, b"state".to_vec()).unwrap();
    storage
        .wl()
        .record_snapshot_within(5, b"state".to_vec(), floor(3))
        .expect("two events through 4 and one more through 5");
}

#[test]
fn an_unaccounted_installed_snapshot_blocks_compaction_across_a_restart() {
    let temp = Temp::new("floor-unaccounted");
    {
        let storage = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
        let mut core = storage.wl();
        core.apply_snapshot(Snapshot {
            data: b"installed".to_vec(),
            metadata: Some(SnapshotMetadata {
                index: 10,
                term: 1,
                conf_state: Some(ConfState {
                    voters: vec![1],
                    ..Default::default()
                }),
            }),
        })
        .unwrap();
        core.append(&[entry(11, 1, b"after")]).unwrap();
        core.set_commit(11).unwrap();
        let error = core
            .record_snapshot_within(11, b"mine".to_vec(), floor(1))
            .expect_err("there is no base to count from");
        assert_eq!(
            refusal_of(&error),
            SnapshotFloorRefusal::Unaccounted { snapshot_index: 10 }
        );
        assert!(
            error
                .to_string()
                .starts_with("PTR_RAFT_SNAPSHOT_UNACCOUNTED"),
            "{error}"
        );
        core.record_snapshot(11, b"mine".to_vec())
            .expect_err("the self-attested form has no base either");
        assert_eq!(core.first_index(), 11);
    }
    let reopened = FileRaftStorage::open(&temp.dir(), &[1], &[]).unwrap();
    let error = reopened
        .wl()
        .record_snapshot_within(10, b"mine".to_vec(), floor(0))
        .expect_err("the marker is durable: still blocked after the restart");
    assert!(matches!(
        refusal_of(&error),
        SnapshotFloorRefusal::Unaccounted { snapshot_index: 10 }
    ));
    // The control: accounting for the snapshot lifts the block.
    reopened.wl().set_applied_events(4).unwrap();
    reopened
        .wl()
        .record_snapshot_within(10, b"mine".to_vec(), floor(4))
        .unwrap();
}
