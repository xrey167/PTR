//! A three-member group, driven message by message.
//!
//! Nothing here waits. Ticks are a count and delivery is explicit, so a partition
//! is "these messages do not arrive" rather than "sleep and hope" — which is the
//! difference between a test that fails when the code is wrong and a test that
//! fails when the machine is busy.
//!
//! The harness is deliberately dumb: one FIFO queue, one node per id, and a set of
//! ids whose messages are dropped in both directions. A clever harness would be a
//! second implementation of the thing under test.

use ptr_ledger::raft_node::{
    decode_message, encode_message, MembershipChange, RaftNode, RetainedSnapshotAnchor,
    SnapshotAnchorMismatch, MAX_MESSAGE_BYTES,
};
use ptr_ledger::{CommittedEvent, LedgerEvent};
use ptr_types::{CapsuleId, Generation, ProjectId};
use raft::prelude::Message;
use raft::StateRole;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);

impl Temp {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ptr-cluster-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn member(&self, id: u64) -> PathBuf {
        self.0.join(format!("node-{id}"))
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const VOTERS: [u64; 3] = [1, 2, 3];

fn event(generation: u64) -> LedgerEvent {
    LedgerEvent::CapsuleCommitted {
        project: ProjectId::from("p"),
        capsule: CapsuleId::from("capsule:a"),
        generation: Generation(generation),
    }
}

struct Cluster {
    nodes: Vec<RaftNode>,
    unreachable: HashSet<u64>,
}

impl Cluster {
    fn open(temp: &Temp) -> Self {
        let nodes = VOTERS
            .iter()
            .map(|id| RaftNode::open(&temp.member(*id), *id, &VOTERS).unwrap())
            .collect();
        Self {
            nodes,
            unreachable: HashSet::new(),
        }
    }

    fn node(&mut self, id: u64) -> &mut RaftNode {
        self.nodes
            .iter_mut()
            .find(|node| node.id() == id)
            .expect("member exists")
    }

    fn get(&self, id: u64) -> &RaftNode {
        self.nodes
            .iter()
            .find(|node| node.id() == id)
            .expect("member exists")
    }

    /// Deliver until nothing is in flight, dropping anything to or from an
    /// unreachable member.
    fn settle(&mut self, initial: Vec<Message>) {
        let mut queue: VecDeque<Message> = initial.into();
        // A bound rather than `loop`: a group that never goes quiet is a bug, and
        // a test that hangs reports it as a timeout instead of as a failure.
        for _ in 0..10_000 {
            let Some(message) = queue.pop_front() else {
                return;
            };
            if self.unreachable.contains(&message.from) || self.unreachable.contains(&message.to) {
                continue;
            }
            let produced = self.node(message.to).step(message).unwrap();
            queue.extend(produced);
        }
        panic!("the group did not go quiet");
    }

    fn leader(&self) -> Option<u64> {
        let leaders: Vec<u64> = self
            .nodes
            .iter()
            .filter(|node| node.is_leader())
            .map(RaftNode::id)
            .collect();
        match leaders.as_slice() {
            [single] => Some(*single),
            _ => None,
        }
    }

    fn elect(&mut self, id: u64) {
        let messages = self.node(id).campaign().unwrap();
        self.settle(messages);
    }

    /// Tick until the member emits something, then deliver it.
    ///
    /// One tick is not a heartbeat: the interval is `heartbeat_tick`, so a test
    /// that ticks once and expects contact is testing its own arithmetic.
    fn heartbeat(&mut self, id: u64) {
        for _ in 0..8 {
            let messages = self.node(id).tick().unwrap();
            if !messages.is_empty() {
                self.settle(messages);
                return;
            }
        }
        panic!("member {id} produced no heartbeat within the interval");
    }

    fn propose(&mut self, id: u64, generation: u64) -> Result<(), String> {
        let messages = self.node(id).propose(event(generation))?;
        self.settle(messages);
        Ok(())
    }

    fn events(&self, id: u64) -> Vec<CommittedEvent> {
        self.get(id).committed_events().to_vec()
    }
}

#[test]
fn a_three_member_group_elects_exactly_one_leader_without_waiting() {
    let temp = Temp::new("elect");
    let mut cluster = Cluster::open(&temp);
    assert_eq!(cluster.leader(), None, "nobody leads before an election");

    cluster.elect(1);

    assert_eq!(cluster.leader(), Some(1));
    let term = cluster.get(1).term();
    for id in [2, 3] {
        assert_eq!(cluster.get(id).role(), StateRole::Follower);
        assert_eq!(
            cluster.get(id).term(),
            term,
            "a follower that voted is in the leader's term"
        );
    }
}

#[test]
fn a_majority_commits_and_a_minority_neither_commits_nor_buffers() {
    let temp = Temp::new("majority");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);

    // Node 3 hears nothing from here on.
    cluster.unreachable.insert(3);
    cluster.propose(1, 1).expect("two of three is a majority");

    assert_eq!(cluster.events(1).len(), 1, "the leader committed");
    assert_eq!(
        cluster.events(2).len(),
        1,
        "the follower it reached committed"
    );
    assert!(
        cluster.events(3).is_empty(),
        "a member outside the majority has decided nothing"
    );

    // A follower does not refuse outright — raft forwards the proposal to the
    // leader it last knew. What matters is that the write is not ordered and is not
    // held either: the message is addressed to a member it cannot reach, and
    // nothing replays it later. A buffered proposal that resurfaces is a write the
    // group never ordered.
    cluster
        .propose(3, 2)
        .expect("a follower forwards rather than refusing");
    assert!(cluster.events(3).is_empty());

    // Heal the partition and let the leader talk: the forwarded proposal is gone,
    // and member 3 receives exactly the one entry the majority decided.
    cluster.unreachable.remove(&3);
    cluster.heartbeat(1);
    assert_eq!(cluster.events(3), cluster.events(1));
    assert_eq!(
        cluster.events(3).len(),
        1,
        "the dropped proposal did not resurface"
    );
}

#[test]
fn a_member_that_knows_no_leader_refuses_a_write_outright() {
    // Before any election there is nobody to forward to, and raft says so rather
    // than holding the proposal until a leadership that may never come.
    let temp = Temp::new("leaderless");
    let mut cluster = Cluster::open(&temp);
    let refused = cluster
        .propose(2, 1)
        .expect_err("a member with no leader cannot order a write");
    assert!(
        refused.to_lowercase().contains("dropped"),
        "expected a dropped proposal, got {refused}"
    );
    assert!(cluster.events(2).is_empty());
}

#[test]
fn a_deposed_leader_cannot_commit_and_what_it_wrote_alone_does_not_survive() {
    let temp = Temp::new("deposed");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    cluster.propose(1, 1).unwrap();
    let agreed = cluster.events(1);
    assert_eq!(agreed.len(), 1);

    // Partition the leader away and let it write on its own.
    cluster.unreachable.insert(1);
    cluster
        .propose(1, 99)
        .expect("a partitioned leader still believes it leads");
    assert_eq!(
        cluster.events(1),
        agreed,
        "it appended, and appending is not committing"
    );

    // The remaining two elect a new leader in a higher term and commit.
    cluster.elect(2);
    assert_eq!(cluster.get(2).role(), StateRole::Leader);
    assert!(cluster.get(2).term() > cluster.get(1).term());
    cluster.propose(2, 2).unwrap();
    let after = cluster.events(2);
    assert_eq!(after.len(), 2);

    // Heal the partition. The old leader learns the higher term, steps down, and
    // the entry only it held is overwritten rather than resurrected.
    cluster.unreachable.remove(&1);
    cluster.heartbeat(2);

    assert_eq!(cluster.get(1).role(), StateRole::Follower);
    assert_eq!(
        cluster.events(1),
        after,
        "one history, and it is the group's"
    );
    for committed in cluster.events(1) {
        assert_ne!(
            committed.event,
            event(99),
            "an entry only the deposed leader held must never be committed"
        );
    }
}

#[test]
fn a_leader_change_leaves_one_total_order_with_no_index_assigned_twice() {
    let temp = Temp::new("order");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    cluster.propose(1, 1).unwrap();
    cluster.propose(1, 2).unwrap();

    cluster.unreachable.insert(1);
    cluster.elect(3);
    cluster.propose(3, 3).unwrap();
    cluster.unreachable.remove(&1);
    cluster.heartbeat(3);

    let reference = cluster.events(3);
    assert_eq!(reference.len(), 3);
    // Indexes are exactly 1..n, in order, with nothing repeated.
    let indexes: Vec<u64> = reference.iter().map(|entry| entry.index.0).collect();
    assert_eq!(indexes, vec![1, 2, 3]);

    for id in VOTERS {
        let events = cluster.events(id);
        assert!(
            reference.starts_with(&events) || events.starts_with(&reference),
            "member {id} holds a different order: {events:?} against {reference:?}"
        );
    }
}

#[test]
fn duplicated_and_stale_messages_change_nothing() {
    let temp = Temp::new("duplicates");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);

    // Keep the messages the leader produces for a proposal, then deliver them,
    // then deliver the very same batch again.
    let first = cluster.node(1).propose(event(1)).unwrap();
    assert!(!first.is_empty(), "a proposal in a group produces traffic");
    cluster.settle(first.clone());
    let settled: Vec<Vec<CommittedEvent>> = VOTERS.iter().map(|id| cluster.events(*id)).collect();

    cluster.settle(first.clone());
    cluster.settle(first);
    for (position, id) in VOTERS.iter().enumerate() {
        assert_eq!(
            cluster.events(*id),
            settled[position],
            "member {id} changed on a replayed batch"
        );
    }

    // A stale append from an earlier term, delivered late.
    let mut stale = Message::default();
    stale.set_msg_type(raft::prelude::MessageType::MsgAppend);
    stale.to = 2;
    stale.from = 1;
    stale.term = 1;
    stale.index = 1;
    let reply = cluster.node(2).step(stale).unwrap();
    cluster.settle(reply);
    for (position, id) in VOTERS.iter().enumerate() {
        assert_eq!(
            cluster.events(*id),
            settled[position],
            "member {id} changed on a stale append"
        );
    }
}

#[test]
fn a_message_round_trips_and_a_damaged_one_is_refused_before_it_is_stepped() {
    let temp = Temp::new("codec");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    let outbound = cluster.node(1).propose(event(1)).unwrap();
    let original = outbound
        .first()
        .expect("a proposal produces traffic")
        .clone();

    // The positive control: the wire format is the message.
    let encoded = encode_message(&original).unwrap();
    assert_eq!(decode_message(&encoded).unwrap(), original);

    let before: Vec<Vec<CommittedEvent>> = VOTERS.iter().map(|id| cluster.events(*id)).collect();

    // Damage that cannot decode is refused, and nothing is stepped. A message is a
    // claim about a term and a log position; half of one claims nothing.
    let mut refusals = 0;
    for position in 0..encoded.len() {
        let mut damaged = encoded.clone();
        damaged[position] ^= 0xff;
        if decode_message(&damaged).is_err() {
            refusals += 1;
        }
    }
    assert!(
        refusals > 0,
        "at least one single-byte change must be rejected outright"
    );
    for truncated in 1..encoded.len() {
        // A prefix either decodes to something else or is refused; either way it is
        // not the original message, which is the property that matters.
        if let Ok(message) = decode_message(&encoded[..truncated]) {
            assert_ne!(
                message, original,
                "a truncated frame decoded as the original"
            );
        }
    }
    assert!(decode_message(&vec![0; MAX_MESSAGE_BYTES + 1]).is_err());

    for (position, id) in VOTERS.iter().enumerate() {
        assert_eq!(
            cluster.events(*id),
            before[position],
            "refusing a frame must not touch state"
        );
    }
}

#[test]
fn a_restarted_member_restores_exactly_the_leader_s_committed_state() {
    let temp = Temp::new("restart");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    cluster.propose(1, 1).unwrap();
    cluster.propose(1, 2).unwrap();
    let leader = cluster.events(1);
    assert_eq!(leader.len(), 2);
    assert_eq!(
        cluster.events(3),
        leader,
        "the follower agreed before it died"
    );

    // Reopen member 3 from its own files alone.
    let reopened = RaftNode::open(&temp.member(3), 3, &VOTERS).unwrap();
    assert_eq!(
        reopened.committed_events(),
        leader.as_slice(),
        "a restarted member must restore exactly what the leader committed"
    );
    assert_eq!(reopened.role(), StateRole::Follower);
    assert!(
        reopened.term() >= cluster.get(1).term(),
        "the term it recorded is not behind the one it acknowledged"
    );
}

/// The test's own state machine: the generations it has seen committed, in order.
///
/// A snapshot payload is the *application's* state, and this is this test's
/// application. Encoding it here rather than in the library is the point: the raft
/// layer carries opaque bytes and has no opinion about them, which is what lets a
/// PTR deployment put a PTRCS002 compacted snapshot there instead of a second
/// artifact invented for raft.
fn encode_state(events: &[CommittedEvent]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(events.len() * 8);
    for committed in events {
        match &committed.event {
            LedgerEvent::CapsuleCommitted { generation, .. } => {
                bytes.extend_from_slice(&generation.0.to_le_bytes());
            }
            other => panic!("this test's state machine does not model {other:?}"),
        }
    }
    bytes
}

fn decode_state(bytes: &[u8]) -> Vec<u64> {
    assert_eq!(bytes.len() % 8, 0, "a state payload is whole generations");
    bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect()
}

#[test]
fn a_member_compacted_past_is_caught_up_by_a_snapshot_and_equals_a_full_replay() {
    let temp = Temp::new("snapshot");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    for generation in 1..=3 {
        cluster.propose(1, generation).unwrap();
    }
    assert_eq!(cluster.events(3).len(), 3, "everyone is up to date first");

    // Member 3 stops hearing anything, and the group moves on without it.
    cluster.unreachable.insert(3);
    for generation in 4..=5 {
        cluster.propose(1, generation).unwrap();
    }
    assert_eq!(cluster.events(3).len(), 3, "it is behind by two");

    // The leader records its state and discards the log those entries are in, so
    // catching member 3 up by replay is no longer possible.
    let leader_state = encode_state(&cluster.events(1));
    let covered = cluster.get(1).raft_committed();
    cluster
        .node(1)
        .record_snapshot(covered, leader_state.clone())
        .unwrap();

    // Heal it. The leader has nothing to replay, so it sends what it does have.
    cluster.unreachable.remove(&3);
    cluster.heartbeat(1);

    let installed = cluster
        .node(3)
        .take_installed_snapshot()
        .expect("a member the leader cannot replay to is sent a snapshot");
    assert_eq!(installed.index, covered);
    assert_eq!(
        installed.state, leader_state,
        "the payload arrives byte for byte"
    );
    assert!(
        cluster.events(3).is_empty(),
        "what the snapshot describes stops being the member's own to report"
    );

    // The application accounts for it, and the group continues.
    let restored = decode_state(&installed.state);
    assert_eq!(restored, vec![1, 2, 3, 4, 5]);
    cluster
        .node(3)
        .resume_with_applied(restored.len() as u64)
        .unwrap();
    cluster.propose(1, 6).unwrap();

    // Restored by snapshot equals restored by full replay: member 2 replayed every
    // entry, member 3 was handed a payload and then one entry, and what they hold is
    // the same sequence with the same indexes.
    let by_replay = cluster.events(2);
    let mut by_snapshot = restored;
    by_snapshot.extend(
        cluster
            .events(3)
            .iter()
            .map(|committed| match &committed.event {
                LedgerEvent::CapsuleCommitted { generation, .. } => generation.0,
                other => panic!("unexpected {other:?}"),
            }),
    );
    assert_eq!(
        by_snapshot,
        decode_state(&encode_state(&by_replay)),
        "a member restored by snapshot holds the same history as one restored by replay"
    );
    let tail = cluster.events(3);
    assert_eq!(tail.len(), 1);
    assert_eq!(
        tail[0].index.0, 6,
        "the event after a snapshot continues the ledger's numbering rather than restarting it"
    );
    assert_eq!(cluster.get(3).applied_events(), 6);
}

#[test]
fn a_member_refuses_to_apply_anything_until_a_snapshot_is_accounted_for() {
    // Carrying on as though the payload were empty would number the next event 1 and
    // disagree with every other member about what that index means.
    let temp = Temp::new("unaccounted");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    for generation in 1..=3 {
        cluster.propose(1, generation).unwrap();
    }
    cluster.unreachable.insert(3);
    for generation in 4..=5 {
        cluster.propose(1, generation).unwrap();
    }
    let covered = cluster.get(1).raft_committed();
    let state = encode_state(&cluster.events(1));
    cluster.node(1).record_snapshot(covered, state).unwrap();
    cluster.unreachable.remove(&3);
    cluster.heartbeat(1);
    assert!(
        cluster.node(3).take_installed_snapshot().is_some(),
        "the member was compacted past, so it is sent a snapshot"
    );

    // Another entry arrives before the application has said what the payload covers.
    // Appending it is fine; *applying* it is what must be refused, and that happens a
    // message later when the commit index moves, so keep delivering until one of them
    // is refused.
    let messages = cluster.node(1).propose(event(6)).unwrap();
    let mut queue: VecDeque<Message> = messages.into();
    let mut refused = None;
    while let Some(message) = queue.pop_front() {
        match cluster.node(message.to).step(message) {
            Ok(produced) => queue.extend(produced),
            Err(error) => {
                refused = Some(error);
                break;
            }
        }
    }
    let refused = refused.expect("applying an entry above an unaccounted snapshot");
    assert!(
        refused.contains("resume_with_applied"),
        "the refusal must name what is missing: {refused}"
    );
}

#[test]
fn a_leader_with_no_recorded_snapshot_cannot_invent_one_for_a_member_it_compacted_past() {
    // The storage refuses to fabricate a snapshot at the requested index, so a
    // leader that never recorded one simply cannot catch the member up. Being stuck
    // is the honest outcome; a payload that did not match its claimed position would
    // be sent as truth.
    let temp = Temp::new("nosnapshot");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    cluster.propose(1, 1).unwrap();
    cluster.unreachable.insert(3);
    cluster.propose(1, 2).unwrap();

    // Compact the leader's log without recording a payload for it: record_snapshot
    // is the only way to compact, so the closest thing is a snapshot the leader
    // then cannot serve — which is what a fresh member asking for an old index sees.
    let ahead = cluster.get(1).raft_committed() + 5;
    let refused = cluster
        .node(1)
        .record_snapshot(ahead, Vec::new())
        .expect_err("a snapshot ahead of the committed index is refused");
    assert!(
        refused.contains("ahead of the committed"),
        "the refusal must say why: {refused}"
    );

    // And member 3, still behind, has not been handed anything.
    cluster.unreachable.remove(&3);
    cluster.heartbeat(1);
    assert!(
        cluster.node(3).take_installed_snapshot().is_none(),
        "no snapshot was recorded, so none can arrive"
    );
    // It catches up by replay instead, because the log was never discarded.
    assert_eq!(cluster.events(3), cluster.events(1));
}

#[test]
fn a_restarted_deposed_leader_cannot_resurrect_the_tail_it_wrote_alone() {
    // Durable fencing, not protocol fencing: the entry the old leader wrote while
    // partitioned is on its disk, and a restart is exactly the moment it could come
    // back. The group's history must win, and the stale tail must leave the file.
    let temp = Temp::new("fenced");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    cluster.propose(1, 1).unwrap();
    let agreed = cluster.events(1);

    cluster.unreachable.insert(1);
    cluster.propose(1, 99).expect("it still believes it leads");
    cluster.elect(2);
    cluster.propose(2, 2).unwrap();
    let group = cluster.events(2);
    assert_eq!(group.len(), 2);
    assert_eq!(agreed.len(), 1);

    // Restart the deposed leader from its own files, with its uncommitted tail still
    // on disk, and put it back in the group.
    let reopened = RaftNode::open(&temp.member(1), 1, &VOTERS).unwrap();
    assert_eq!(
        reopened.committed_events(),
        agreed.as_slice(),
        "it recovers what was committed, not what it wrote alone"
    );
    let position = cluster
        .nodes
        .iter()
        .position(|node| node.id() == 1)
        .expect("member 1");
    cluster.nodes[position] = reopened;
    cluster.unreachable.remove(&1);
    cluster.heartbeat(2);

    assert_eq!(cluster.get(1).role(), StateRole::Follower);
    assert_eq!(
        cluster.events(1),
        group,
        "a restarted deposed leader ends with the group's history"
    );
    for committed in cluster.events(1) {
        assert_ne!(
            committed.event,
            event(99),
            "the tail it wrote alone must not survive a restart either"
        );
    }
}

/// The anchor a host retained out of band, for the snapshot member 3 is about to
/// be sent. Built from the leader's own state so the positive case is the honest
/// one; the negatives perturb it rather than the payload.
fn retained_for(state: &[u8], index: u64) -> RetainedSnapshotAnchor {
    RetainedSnapshotAnchor {
        index,
        digest: ptr_ledger::integrity::sha256(state),
    }
}

/// Set up a cluster where member 3 has a snapshot installed, and return the
/// leader's payload and the position it covers.
fn cluster_with_installed_snapshot(temp: &Temp) -> (Cluster, Vec<u8>, u64) {
    let mut cluster = Cluster::open(temp);
    cluster.elect(1);
    for generation in 1..=3 {
        cluster.propose(1, generation).unwrap();
    }
    cluster.unreachable.insert(3);
    for generation in 4..=5 {
        cluster.propose(1, generation).unwrap();
    }
    let leader_state = encode_state(&cluster.events(1));
    let covered = cluster.get(1).raft_committed();
    cluster
        .node(1)
        .record_snapshot(covered, leader_state.clone())
        .unwrap();
    cluster.unreachable.remove(&3);
    cluster.heartbeat(1);
    assert!(
        cluster.node(3).take_installed_snapshot().is_some(),
        "the scenario requires a snapshot to have landed"
    );
    (cluster, leader_state, covered)
}

#[test]
fn a_snapshot_matching_an_independently_retained_anchor_is_taken() {
    let temp = Temp::new("anchor-ok");
    let (mut cluster, state, covered) = cluster_with_installed_snapshot(&temp);

    let taken = cluster
        .node(3)
        .take_installed_snapshot_matching(retained_for(&state, covered))
        .expect("the payload is the one the retained anchor describes");
    assert_eq!(taken.index, covered);
    assert_eq!(taken.state, state, "the payload arrives byte for byte");
    assert_eq!(
        taken.digest,
        ptr_ledger::integrity::sha256(&state),
        "the digest is computed from the bytes that arrived, not read out of them"
    );
}

#[test]
fn a_payload_the_retained_anchor_does_not_describe_is_refused_and_stays_installed() {
    let temp = Temp::new("anchor-digest");
    let (mut cluster, state, covered) = cluster_with_installed_snapshot(&temp);

    // Same position, different state: a leader sending a well-formed snapshot for
    // the right index whose contents are not what the host expects. Raft cannot
    // tell the difference — the payload is opaque to it — which is exactly why the
    // anchor has to come from somewhere the sender does not control.
    let mut elsewhere = state.clone();
    elsewhere.extend_from_slice(b"not what was retained");

    assert_eq!(
        cluster
            .node(3)
            .take_installed_snapshot_matching(retained_for(&elsewhere, covered)),
        Err(SnapshotAnchorMismatch::Digest)
    );

    // The refusal must not look like "no snapshot arrived". A member that carried
    // on would renumber every event after the snapshot.
    assert!(
        cluster.node(3).take_installed_snapshot().is_some(),
        "a refused snapshot stays installed, so the member stays blocked"
    );
}

#[test]
fn a_payload_describing_another_position_is_refused_by_index() {
    let temp = Temp::new("anchor-index");
    let (mut cluster, state, covered) = cluster_with_installed_snapshot(&temp);

    let error = cluster
        .node(3)
        .take_installed_snapshot_matching(retained_for(&state, covered + 1))
        .expect_err("a snapshot for another position is not this one");
    assert_eq!(
        error,
        SnapshotAnchorMismatch::Index {
            installed: covered,
            retained: covered + 1
        },
        "the refusal names both positions so an operator can tell which is wrong"
    );
    assert_eq!(error.code(), "PTR_SNAPSHOT_ANCHOR_INDEX");
}

/// The control.
///
/// Both refusals above would pass against a check that refused everything, so the
/// same member, in the same state, must still accept the genuine pair — and a
/// member with nothing installed must say *that* rather than a mismatch.
#[test]
fn the_anchor_check_accepts_the_genuine_pair_and_distinguishes_having_nothing() {
    let temp = Temp::new("anchor-control");
    let (mut cluster, state, covered) = cluster_with_installed_snapshot(&temp);

    assert!(
        cluster
            .node(3)
            .take_installed_snapshot_matching(retained_for(&state, covered))
            .is_ok(),
        "the genuine pair is accepted by the same check that refused the others"
    );

    // Member 2 never fell behind, so nothing was installed on it. "Nothing to
    // match" is a different answer from "what arrived is wrong".
    assert_eq!(
        cluster
            .node(2)
            .take_installed_snapshot_matching(retained_for(&state, covered)),
        Err(SnapshotAnchorMismatch::NothingInstalled)
    );
}

#[test]
fn a_leader_removes_a_voter_and_the_new_configuration_survives_a_restart() {
    let temp = Temp::new("member-remove");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    cluster.propose(1, 1).unwrap();

    assert_eq!(
        cluster.get(1).voters(),
        vec![1, 2, 3],
        "the group starts as the three it was opened with"
    );

    // Member 3 is already gone, which is the case a removal is usually for. It
    // also keeps the test honest: a removed member that is still being stepped
    // would refuse the message, since raft will not step a peer that is no longer
    // in its configuration.
    cluster.unreachable.insert(3);
    let messages = cluster
        .node(1)
        .propose_membership(MembershipChange::RemoveVoter(3))
        .expect("a leader may propose");
    cluster.settle(messages);

    assert_eq!(
        cluster.get(1).voters(),
        vec![1, 2],
        "the leader applied the change it committed"
    );
    assert_eq!(
        cluster.get(2).voters(),
        vec![1, 2],
        "and so did the follower that replicated it"
    );

    // The point of the gate: a configuration that exists only in memory is a
    // membership change no restart can honour. Reopen member 2 from its own disk
    // and ask it again.
    drop(cluster);
    let reopened = RaftNode::open(&temp.member(2), 2, &VOTERS).unwrap();
    assert_eq!(
        reopened.voters(),
        vec![1, 2],
        "the recorded configuration is the authority on reopen, not the voters \
         passed to open"
    );
}

#[test]
fn a_leader_adds_a_voter_and_the_new_member_joins_the_group() {
    let temp = Temp::new("member-add");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    cluster.propose(1, 1).unwrap();

    // The joiner is opened knowing only itself; what makes it a voter is the
    // committed change, not the list it was opened with.
    cluster
        .nodes
        .push(RaftNode::open(&temp.member(4), 4, &[4]).unwrap());

    let messages = cluster
        .node(1)
        .propose_membership(MembershipChange::AddVoter(4))
        .expect("a leader may propose");
    cluster.settle(messages);

    assert_eq!(
        cluster.get(1).voters(),
        vec![1, 2, 3, 4],
        "the group now counts four voters"
    );
    let heartbeat = cluster.node(1).tick().unwrap();
    cluster.settle(heartbeat);
    assert_eq!(
        cluster.events(4),
        cluster.events(1),
        "and the joiner is caught up to exactly what the leader committed"
    );
}

#[test]
fn a_follower_cannot_propose_a_membership_change() {
    let temp = Temp::new("member-follower");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);

    let error = cluster
        .node(2)
        .propose_membership(MembershipChange::RemoveVoter(3))
        .expect_err("a follower asking peers to accept a configuration nobody agreed");
    assert!(error.contains("only a leader"), "{error}");
}

/// The control.
///
/// The assertions above would hold just as well if `voters()` reported a
/// constant, so the observable has to be shown to move only when a change
/// actually commits: a refused proposal must leave it exactly as it was.
#[test]
fn a_refused_proposal_leaves_the_configuration_untouched() {
    let temp = Temp::new("member-control");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);

    let before = cluster.get(2).voters();
    assert!(cluster
        .node(2)
        .propose_membership(MembershipChange::RemoveVoter(3))
        .is_err());
    assert_eq!(
        cluster.get(2).voters(),
        before,
        "nothing was proposed, so nothing changed"
    );

    // And the same member reports the change once a leader does commit one, so
    // the equality above is not simply an inert reading.
    cluster.unreachable.insert(3);
    let messages = cluster
        .node(1)
        .propose_membership(MembershipChange::RemoveVoter(3))
        .unwrap();
    cluster.settle(messages);
    assert_ne!(
        cluster.get(2).voters(),
        before,
        "the same observable moves when a change really commits"
    );
}

#[test]
fn reopening_after_a_snapshot_continues_the_numbering_it_established() {
    let temp = Temp::new("applied-events");
    let mut cluster = Cluster::open(&temp);
    cluster.elect(1);
    for generation in 1..=4 {
        cluster.propose(1, generation).unwrap();
    }
    let covered = cluster.get(1).raft_committed();
    let state = encode_state(&cluster.events(1));
    cluster.node(1).record_snapshot(covered, state).unwrap();
    // The application accounts for the four events the payload describes, so the
    // records below the floor stop being this member's to report.
    cluster.node(1).resume_with_applied(4).unwrap();
    cluster.propose(1, 5).unwrap();

    let fifth = cluster
        .events(1)
        .last()
        .expect("the event after the snapshot")
        .index
        .0;
    assert_eq!(fifth, 5, "the fifth event is the fifth event");
    drop(cluster);

    // The events a snapshot covers are described by a payload, not by records
    // this member still holds. A restart that counted from zero would hand the
    // next event an index the snapshot already covers, and two different events
    // would claim it — which is what happened before the count was made durable.
    let reopened = RaftNode::open(&temp.member(1), 1, &VOTERS).unwrap();
    let recovered: Vec<u64> = reopened
        .committed_events()
        .iter()
        .map(|event| event.index.0)
        .collect();
    assert_eq!(
        recovered,
        vec![5],
        "the one record above the floor keeps the index it was committed at"
    );
}
