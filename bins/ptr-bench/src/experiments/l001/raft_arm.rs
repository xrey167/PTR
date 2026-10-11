//! Arm D: three in-process `RaftNode`s on a deterministic router, no sockets and no timers.
//!
//! Four scenario kinds, each a seeded schedule: a **minority leader** (the leader is isolated
//! and proposes a revocation that cannot commit, the majority elects another and commits the
//! workload, the partition heals), a **term change** (a revocation is committed, the leader is
//! deposed, the new leader goes on, the old one rejoins), **restarts** (members and leaders are
//! closed and reopened between proposals) and **message faults** (the router drops, duplicates
//! and reorders messages while the leader proposes, then the faults stop).
//!
//! The harness decides what is committed from the schedule and not from the members: an event
//! proposed while its proposer can reach a majority is *committable*; one proposed by an
//! isolated leader is not, and must never appear in any member's committed events. Every
//! member's committed events are replayed into `PtrRuntime` and compared with the oracle's
//! answer for the events the schedule committed.

use super::check::{add, bump, judge_history, Expectation};
use super::file_arm::RunDir;
use super::model::{Ev, Workload};
use super::router::{Faults, Router, Stats};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;
use ptr_ledger::{CommittedEvent, LedgerEvent, RaftNode};
use std::path::PathBuf;
use std::sync::Arc;

const ARM: &str = "raft";
const VOTERS: [u64; 3] = [1, 2, 3];

/// How many times the leader speaks before a healed group is judged as it stands.
const CONVERGENCE_ROUNDS: usize = 6;

/// The subject the minority leader revokes: committed first, so a leak is visible.
const ISOLATED: &str = "capsule:l001:iso";

struct Cluster {
    dir: PathBuf,
    nodes: Vec<RaftNode>,
    router: Router,
    last_leader: Option<u64>,
    leader_changes: i64,
    minority_proposals: i64,
    deposed_restarts: i64,
    restarts: i64,
    stalled: i64,
    /// A member that could not be opened again from its own directory.
    open_failures: i64,
    /// A proposal the schedule says a leader must be able to make, and was refused.
    propose_failures: i64,
}

impl Cluster {
    fn open(dir: PathBuf, router_seed: u64) -> Result<Self, String> {
        let nodes = VOTERS
            .iter()
            .map(|id| RaftNode::open(&dir.join(format!("node-{id}")), *id, &VOTERS))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            dir,
            nodes,
            router: Router::new(router_seed),
            last_leader: None,
            leader_changes: 0,
            minority_proposals: 0,
            deposed_restarts: 0,
            restarts: 0,
            stalled: 0,
            open_failures: 0,
            propose_failures: 0,
        })
    }

    fn node(&mut self, id: u64) -> &mut RaftNode {
        self.nodes
            .iter_mut()
            .find(|node| node.id() == id)
            .expect("member exists")
    }

    fn lists(&self) -> Vec<(u64, Vec<CommittedEvent>)> {
        self.nodes
            .iter()
            .map(|node| (node.id(), node.committed_events().to_vec()))
            .collect()
    }

    fn settle(&mut self, initial: Vec<ptr_ledger::RaftMessage>) -> Result<(), String> {
        self.router.settle(&mut self.nodes, initial)?;
        self.note_leader();
        Ok(())
    }

    /// The leader of the highest term, if any member believes it leads.
    fn leader(&self) -> Option<u64> {
        self.nodes
            .iter()
            .filter(|node| node.is_leader())
            .max_by_key(|node| node.term())
            .map(RaftNode::id)
    }

    fn note_leader(&mut self) {
        let leader = self.leader();
        if leader.is_some() && leader != self.last_leader {
            if self.last_leader.is_some() {
                self.leader_changes += 1;
            }
            self.last_leader = leader;
        }
    }

    fn elect(&mut self, id: u64) -> Result<(), String> {
        let messages = self.node(id).campaign()?;
        self.settle(messages)?;
        if !self.node(id).is_leader() {
            self.stalled += 1;
        }
        Ok(())
    }

    /// `id` proposes `event`; a refusal by the node is reported to the caller.
    fn propose(&mut self, id: u64, event: &LedgerEvent) -> Result<(), String> {
        let isolated = self.router.isolated.contains(&id);
        let messages = self.node(id).propose(event.clone())?;
        if isolated && self.node(id).is_leader() {
            self.minority_proposals += 1;
        }
        self.settle(messages)
    }

    /// A proposal the schedule requires to be accepted (the proposer leads, whether or not a
    /// majority can hear it): a refusal is counted against the product and the schedule goes on.
    fn must_propose(&mut self, id: u64, event: &LedgerEvent) -> Result<(), String> {
        if self.propose(id, event).is_err() {
            self.propose_failures += 1;
        }
        Ok(())
    }

    /// Tick `id` until it speaks, then deliver what it says.
    fn heartbeat(&mut self, id: u64) -> Result<(), String> {
        for _ in 0..8 {
            let messages = self.node(id).tick()?;
            if !messages.is_empty() {
                return self.settle(messages);
            }
        }
        self.stalled += 1;
        Ok(())
    }

    /// After the faults and the partitions are over: `leader` speaks until every member holds
    /// as many committed events as it does, for at most [`CONVERGENCE_ROUNDS`] rounds. A group
    /// of three on a reliable router converges in two; a member that has not by the bound is
    /// judged as it stands.
    fn converge(&mut self, leader: u64) -> Result<(), String> {
        for _ in 0..CONVERGENCE_ROUNDS {
            self.heartbeat(leader)?;
            let target = self.node(leader).committed_events().len();
            if self
                .nodes
                .iter()
                .all(|node| node.committed_events().len() >= target)
            {
                break;
            }
        }
        Ok(())
    }

    /// Close member `id` and open it again from its directory. Returns how many events it had
    /// committed before.
    fn restart(&mut self, id: u64) -> Result<usize, String> {
        let before = self.node(id).committed_events().len();
        let was_leader = self.node(id).is_leader();
        let own_term = self.node(id).term();
        let others_lead = self
            .nodes
            .iter()
            .any(|node| node.id() != id && node.is_leader() && node.term() > own_term);
        if was_leader && others_lead {
            self.deposed_restarts += 1;
        }
        self.restarts += 1;
        // A member that cannot read its own directory again is a recovery the product failed;
        // it is counted, and the old handle stays so that the schedule can go on.
        let Ok(reopened) = RaftNode::open(&self.dir.join(format!("node-{id}")), id, &VOTERS) else {
            self.open_failures += 1;
            return Ok(before);
        };
        let slot = self
            .nodes
            .iter()
            .position(|node| node.id() == id)
            .expect("member exists");
        self.nodes[slot] = reopened;
        Ok(before)
    }

    fn finish(self, tally: &mut Tally) {
        add(tally, ARM, "leader_changes", self.leader_changes);
        add(tally, ARM, "minority_proposals", self.minority_proposals);
        add(tally, ARM, "deposed_leader_restarts", self.deposed_restarts);
        add(tally, ARM, "restarts", self.restarts);
        add(tally, ARM, "unjustified_refusals", self.open_failures);
        add(tally, ARM, "reference_mismatches", self.propose_failures);
        add(tally, ARM, "raft_stalled", self.stalled);
        let stats: &Stats = &self.router.stats;
        add(tally, ARM, "messages_dropped", stats.dropped());
        add(tally, ARM, "messages_duplicated", stats.duplicated);
        add(tally, ARM, "messages_reordered", stats.reordered);
        add(tally, ARM, "messages_delivered", stats.delivered);
    }
}

/// Judge every member's committed events: nothing in `forbidden` may appear, each list must be
/// a prefix of the committed order, and with `floor` set it must hold at least `acked` events.
fn check_members(
    tally: &mut Tally,
    cluster: &Cluster,
    truth: &Arc<Workload>,
    forbidden: &[LedgerEvent],
    acked: usize,
    floor: bool,
) {
    for (_, list) in cluster.lists() {
        if list
            .iter()
            .any(|committed| forbidden.contains(&committed.event))
        {
            bump(tally, ARM, "raft_revocation_visible_before_commit");
        }
        let expect = Expectation {
            acked: if floor { acked } else { 0 },
            attempted: truth.events.len(),
            must_refuse: false,
            damage: None,
            class: None,
            witness: 0,
        };
        bump(tally, ARM, "members_replayed");
        judge_history(tally, ARM, truth, &expect, &list);
    }
}

/// The index of the first revocation of the workload.
fn first_revocation(workload: &Workload) -> usize {
    workload
        .events
        .iter()
        .position(Ev::is_revocation)
        .expect("every workload revokes")
}

fn with_isolated_subject(workload: &Workload) -> Arc<Workload> {
    let mut events = vec![Ev::Commit {
        subject: ISOLATED.into(),
        generation: 1,
    }];
    events.extend(workload.events.iter().cloned());
    Arc::new(Workload::from_events(events))
}

fn minority_leader(
    tally: &mut Tally,
    dir: PathBuf,
    seed: u64,
    workload: &Workload,
    rng: &mut Rng,
) -> Result<(), String> {
    let truth = with_isolated_subject(workload);
    let revoke = Ev::Revoke {
        subject: ISOLATED.into(),
        generation: 1,
    }
    .to_ledger();
    let n = truth.events.len();
    // At least the first revocation is committed before the partition.
    let first = first_revocation(workload) + 2;
    let split = rng.range(first as u64, (n - 1) as u64) as usize;
    let mut cluster = Cluster::open(dir, seed)?;
    cluster.elect(1)?;
    for event in &truth.events[..split] {
        cluster.must_propose(1, &event.to_ledger())?;
    }
    check_members(
        tally,
        &cluster,
        &truth,
        std::slice::from_ref(&revoke),
        0,
        false,
    );
    cluster.router.isolated.insert(1);
    cluster.must_propose(1, &revoke)?;
    check_members(
        tally,
        &cluster,
        &truth,
        std::slice::from_ref(&revoke),
        0,
        false,
    );
    cluster.elect(2)?;
    for event in &truth.events[split..] {
        cluster.must_propose(2, &event.to_ledger())?;
    }
    let acked = cluster.node(2).committed_events().len();
    check_members(
        tally,
        &cluster,
        &truth,
        std::slice::from_ref(&revoke),
        0,
        false,
    );
    // The isolated leader still believes it leads when it restarts, with the revocation it
    // wrote alone on its disk and nowhere else.
    let before = cluster.restart(1)?;
    check_members(
        tally,
        &cluster,
        &truth,
        std::slice::from_ref(&revoke),
        0,
        false,
    );
    let _ = before;
    cluster.router.isolated.clear();
    cluster.converge(2)?;
    check_members(
        tally,
        &cluster,
        &truth,
        std::slice::from_ref(&revoke),
        acked,
        true,
    );
    // The deposed leader restarts: what it wrote alone must not come back.
    let before = cluster.restart(1)?;
    check_members(
        tally,
        &cluster,
        &truth,
        std::slice::from_ref(&revoke),
        before.min(acked),
        true,
    );
    cluster.finish(tally);
    Ok(())
}

fn term_change(
    tally: &mut Tally,
    dir: PathBuf,
    seed: u64,
    workload: &Workload,
    rng: &mut Rng,
) -> Result<(), String> {
    let truth = Arc::new(workload.clone());
    let n = truth.events.len();
    let first = first_revocation(workload) + 1;
    let split = rng.range(first as u64, (n - 1) as u64) as usize;
    let mut cluster = Cluster::open(dir, seed)?;
    cluster.elect(1)?;
    for event in &truth.events[..split] {
        cluster.must_propose(1, &event.to_ledger())?;
    }
    let acked_before = cluster.node(1).committed_events().len();
    cluster.router.isolated.insert(1);
    cluster.elect(2)?;
    for event in &truth.events[split..] {
        cluster.must_propose(2, &event.to_ledger())?;
    }
    cluster.router.isolated.clear();
    cluster.converge(2)?;
    let acked = cluster.node(2).committed_events().len().max(acked_before);
    check_members(tally, &cluster, &truth, &[], acked, true);
    cluster.finish(tally);
    Ok(())
}

fn restarts(
    tally: &mut Tally,
    dir: PathBuf,
    seed: u64,
    workload: &Workload,
    rng: &mut Rng,
) -> Result<(), String> {
    let mut cluster = Cluster::open(dir, seed)?;
    cluster.elect(1)?;
    let mut leader = 1u64;
    let mut proposed: Vec<Ev> = Vec::new();
    for event in &workload.events {
        if cluster.node(leader).is_leader() && cluster.propose(leader, &event.to_ledger()).is_ok() {
            proposed.push(event.clone());
        } else {
            cluster.stalled += 1;
        }
        if rng.chance(0.4) {
            let victim = VOTERS[rng.index(VOTERS.len())];
            let before = cluster.restart(victim)?;
            // A member that restarts still holds everything it had committed.
            let list = cluster.node(victim).committed_events().to_vec();
            let truth = Arc::new(Workload::from_events(proposed.clone()));
            let expect = Expectation {
                acked: before,
                attempted: truth.events.len(),
                must_refuse: false,
                damage: None,
                class: None,
                witness: 0,
            };
            bump(tally, ARM, "members_replayed");
            judge_history(tally, ARM, &truth, &expect, &list);
            if victim == leader {
                cluster.elect(victim)?;
                if !cluster.node(victim).is_leader() {
                    leader = cluster.leader().unwrap_or(leader);
                }
            }
        }
    }
    cluster.converge(leader)?;
    let acked = cluster.node(leader).committed_events().len();
    if !proposed.is_empty() {
        let truth = Arc::new(Workload::from_events(proposed));
        check_members(tally, &cluster, &truth, &[], acked, true);
    }
    cluster.finish(tally);
    Ok(())
}

fn message_faults(
    tally: &mut Tally,
    dir: PathBuf,
    seed: u64,
    workload: &Workload,
    rng: &mut Rng,
) -> Result<(), String> {
    let mut cluster = Cluster::open(dir, seed)?;
    cluster.router.faults = Faults {
        drop: 120,
        duplicate: 150,
        reorder: 200,
    };
    for _ in 0..3 {
        if cluster.node(1).is_leader() {
            break;
        }
        cluster.elect(1)?;
    }
    let mut proposed: Vec<Ev> = Vec::new();
    for event in &workload.events {
        if cluster.node(1).is_leader() && cluster.propose(1, &event.to_ledger()).is_ok() {
            proposed.push(event.clone());
        }
        if rng.chance(0.5) {
            cluster.heartbeat(1)?;
        }
    }
    let acked = cluster.node(1).committed_events().len();
    cluster.router.faults = Faults::default();
    cluster.converge(1)?;
    if proposed.is_empty() {
        cluster.stalled += 1;
    } else {
        let truth = Arc::new(Workload::from_events(proposed));
        check_members(tally, &cluster, &truth, &[], acked, true);
        let longest = cluster
            .lists()
            .iter()
            .map(|(_, list)| list.len())
            .max()
            .unwrap_or(0);
        for (_, list) in cluster.lists() {
            if list.len() < longest {
                bump(tally, ARM, "raft_unconverged_members");
            }
        }
    }
    cluster.finish(tally);
    Ok(())
}

/// Run `per_kind` scenarios of each of the four kinds.
pub fn run(
    tally: &mut Tally,
    runs: &mut RunDir,
    rng: &mut Rng,
    per_kind: usize,
) -> Result<(), String> {
    type Scenario = fn(&mut Tally, PathBuf, u64, &Workload, &mut Rng) -> Result<(), String>;
    let kinds: [(&str, Scenario); 4] = [
        ("minority-leader", minority_leader),
        ("term-change", term_change),
        ("restarts", restarts),
        ("message-faults", message_faults),
    ];
    for (_, scenario) in kinds {
        for _ in 0..per_kind {
            let mut stream = rng.fork(0x4a);
            let workload = super::model::generate(
                &mut stream,
                super::params::EVENTS_MIN,
                super::params::EVENTS_MAX,
            );
            let seed = stream.next_u64();
            let dir = runs.fresh();
            let checks_before = tally.get("arms.raft.members_replayed");
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scenario(tally, dir.clone(), seed, &workload, &mut stream)
            }));
            match outcome {
                Err(_) => bump(tally, ARM, "product_panics"),
                Ok(Err(_)) => {
                    // A node refused a step of a valid schedule (a campaign, a tick, a
                    // delivery that never went quiet, a first opening): the product's.
                    bump(tally, ARM, "raft_scenario_errors");
                    bump(tally, ARM, "reference_mismatches");
                }
                Ok(Ok(())) => bump(tally, ARM, "scenarios"),
            }
            if tally.get("arms.raft.members_replayed") > checks_before {
                bump(tally, ARM, "scenarios_checked");
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}
