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

use ptr_ledger::raft_node::{decode_message, encode_message, RaftNode, MAX_MESSAGE_BYTES};
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
