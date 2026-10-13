//! Arms S and G: three Raft members over `FileRaftStorage`, crashed inside the transitions of
//! their durable state that L001-v2 never reached (snapshot recording, log compaction, snapshot
//! installation, the rewrite of the state file, the append of entries), and held to a
//! single-node reference after partitions, restarts and message faults.
//!
//! A scenario is a seeded schedule (`Spec`) run on a deterministic router: elect member 1, propose
//! the first events, isolate member 3, propose more, let members 1 and 2 record a snapshot (which
//! compacts their logs), heal the partition so that the leader catches member 3 up by sending
//! it the snapshot, propose the rest and converge. The host's part is played by the harness: it
//! vets an installed snapshot against the anchor it retained out of band
//! (`take_installed_snapshot_matching`), accounts for it (`resume_with_applied`), and builds the
//! application's state from the snapshot's payload and the events above it.
//!
//! The same schedule is run once without a fault (the observing run, which is also baseline B0
//! and counts the hits of every failpoint) and then with one: a panic at the n-th hit of a
//! storage failpoint (the process-abort class, taken from the observing run's count), or an I/O
//! error returned in place of a write. A member that crashed or failed is opened again from its
//! directory and the schedule goes on. Four properties are checked, at every call or at the end:
//!
//! 1. **Persist before send**: every vote a member grants and every append it acknowledges in a
//!    message a call returns is already in the member's files at that moment (the storage is
//!    opened separately and read), and a call whose persistence failed returns an error and no
//!    message.
//! 2. **No forgetting**: a member opened again holds at least the term, the commit index and the
//!    applied events it had when its last call returned, and the history it presents is a prefix
//!    of the workload with the indices 1 to n.
//! 3. **Convergence**: after the faults, the leader's heartbeats bring every member to the same
//!    history within a pinned number of rounds, and that history is the workload (the single-node
//!    reference).
//! 4. **No unvetted state**: a snapshot whose payload omits a revocation is refused by the
//!    anchor match, the member stays blocked on it across a restart, and the unanchored take of
//!    the same snapshot (baseline B1) would have made a revoked generation authorizable.

use super::super::l001::check::{add, bump, InjectedPanic};
use super::super::l001::image::{check_in_model, damage_shape, power_loss, Policy};
use super::super::l001::model::{generate, Ev, Workload};
use super::super::l001::probe;
use super::cutover::{explain, label, RunDir};
use super::view::{decode_payload, encode_payload, judge_logical, logical};
use super::{metrics, params};
use crate::experiments::layered_kv::emit::Tally;
use crate::experiments::rng::Rng;
use fail::FailScenario;
use ptr_ledger::integrity::sha256;
use ptr_ledger::raft_node::{RetainedSnapshotAnchor, SnapshotAnchorMismatch};
use ptr_ledger::raft_storage::FileRaftStorage;
use ptr_ledger::{CommittedEvent, RaftMessage, RaftNode};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub const ARM: &str = "member";
const VOTERS: [u64; 3] = [1, 2, 3];
/// The text an injected I/O error carries (`raft_storage::injected_io_error`).
const INJECTED: &str = "PTR_RAFT_INJECTED_IO_ERROR";

// The numbers of raft's message types (the generated enum of `raft-proto`); a test pins them.
const MSG_APPEND_RESPONSE: i32 = 4;
const MSG_REQUEST_VOTE_RESPONSE: i32 = 6;
const MSG_SNAPSHOT: i32 = 7;

/// How a scenario is run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Inject {
    /// No fault: the observing run, baseline B0.
    None,
    /// Panic at the `target`-th hit of `point`.
    Crash { point: &'static str, target: usize },
    /// Return an I/O error at the `target`-th hit of `point`.
    IoError { point: &'static str, target: usize },
}

/// A seeded schedule.
#[derive(Clone)]
pub struct Spec {
    pub seed: u64,
    pub workload: Arc<Workload>,
    /// Events proposed before member 3 is isolated, and before the snapshots.
    pub a: usize,
    pub b: usize,
    pub lag: bool,
    pub snapshots: bool,
    /// A member that restarts after the snapshots were recorded (1, 2 or 0 for none).
    pub restart_mid: u64,
    /// Member 3 restarts between the installation of a snapshot and the host accounting for it.
    pub restart_before_resume: bool,
    /// Message faults while the last events are proposed.
    pub faults: bool,
    /// Member 1 records a snapshot whose payload omits the revocations.
    pub tamper: bool,
    /// The leader is isolated after the first convergence and another member is elected.
    pub depose: bool,
}

/// A schedule whose snapshot can be forged so as to resurrect a revoked generation: the history
/// has a revocation of a generation that is still the live one at the end of the snapshot's
/// prefix. Returns the length of that prefix.
fn forgeable_prefix(workload: &Workload) -> Option<usize> {
    let mut live: BTreeMap<&str, u64> = BTreeMap::new();
    let mut found = None;
    for (i, event) in workload.events.iter().enumerate() {
        match event {
            Ev::Commit {
                subject,
                generation,
            } => {
                live.insert(subject, *generation);
            }
            Ev::Supersede { subject, new, .. } => {
                live.insert(subject, *new);
            }
            Ev::Revoke {
                subject,
                generation,
            } => {
                if live.get(subject.as_str()) == Some(generation) {
                    found = Some(i + 1);
                }
            }
        }
    }
    found.filter(|prefix| *prefix + 2 <= workload.events.len())
}

/// Draw a schedule. `kind` decides what it must contain.
pub fn spec_for(kind: &str, rng: &mut Rng) -> Spec {
    loop {
        let workload = Arc::new(generate(rng, params::EVENTS_MIN, params::EVENTS_MAX));
        let n = workload.events.len();
        let tamper = kind == "tamper";
        let forge_at = forgeable_prefix(&workload);
        if tamper && forge_at.is_none() {
            continue;
        }
        let (a, b) = if let Some(prefix) = forge_at.filter(|_| tamper) {
            (prefix.saturating_sub(2).max(2), prefix)
        } else {
            let a = 2 + rng.index(n / 4);
            let b = (a + 2 + rng.index((n - a) / 2).max(1)).min(n - 2);
            (a, b)
        };
        let plain = kind == "heal";
        return Spec {
            seed: rng.next_u64(),
            workload,
            a,
            b,
            lag: !plain || rng.chance(0.6),
            snapshots: !plain || rng.chance(0.6),
            restart_mid: if rng.chance(0.5) {
                // The member that records a forged snapshot is not restarted: what it presents
                // afterwards is the forgery, which nobody judges.
                if tamper {
                    2
                } else {
                    1 + rng.index(2) as u64
                }
            } else {
                0
            },
            restart_before_resume: rng.chance(0.5),
            faults: plain && rng.chance(0.7),
            tamper,
            depose: plain,
        };
    }
}

/// Failpoint bookkeeping shared with the callbacks.
#[derive(Default)]
struct Injector {
    hits: Mutex<BTreeMap<&'static str, usize>>,
    /// Set once the members exist: the creation of their durable state is the harness's setup,
    /// not a step of the schedule, so a failpoint neither counts nor fires before it.
    armed: AtomicBool,
    fired: AtomicBool,
    /// The member whose call is running.
    current: AtomicU64,
    /// Captures of the member directories at `raft.append.after_write_before_sync`.
    captures: Mutex<Vec<Capture>>,
    capturing: AtomicBool,
    /// The length of each member's log after its last completed call.
    log_len: Mutex<BTreeMap<u64, usize>>,
    /// The number of events each member held when its last call returned.
    view_len: Mutex<BTreeMap<u64, usize>>,
}

/// A capture as the run reports it: member, its files, durable log length, acknowledged view.
pub type CaptureRow = (u64, BTreeMap<String, Vec<u8>>, usize, usize);

/// A member's files at the moment between the write of an append and its sync.
#[derive(Clone)]
struct Capture {
    node: u64,
    files: BTreeMap<String, Vec<u8>>,
    durable_log: usize,
    acked_view: usize,
}

/// What a scenario produced besides its counters.
pub struct Run {
    pub counters: Vec<(String, i64)>,
    pub hits: BTreeMap<&'static str, usize>,
    pub captures: Vec<CaptureRow>,
    pub fired: bool,
}

/// What a member showed the host at its last completed call, and what it had durable.
#[derive(Clone, Copy, Debug, Default)]
struct Facts {
    term: u64,
    commit: u64,
    applied: u64,
    pending: bool,
}

impl Facts {
    fn of(node: &mut RaftNode) -> Self {
        Self {
            term: node.term(),
            commit: node.raft_committed(),
            applied: node.applied_events(),
            pending: node.take_installed_snapshot().is_some(),
        }
    }
}

/// The router's faults, in per mille of messages.
#[derive(Clone, Copy, Default)]
struct Faults {
    drop: u64,
    duplicate: u64,
    reorder: u64,
}

enum Called<T> {
    Done(T),
    Failed(String),
    Crashed,
}

struct Group {
    dir: PathBuf,
    nodes: Vec<Option<RaftNode>>,
    facts: BTreeMap<u64, Facts>,
    isolated: BTreeSet<u64>,
    faults: Faults,
    rng: Rng,
    inj: Arc<Injector>,
    spec: Spec,
    /// The anchors the host retained out of band: snapshot index to (index, digest) of the
    /// genuine payload.
    retained: BTreeMap<u64, RetainedSnapshotAnchor>,
    /// The number of events the genuine payload of each retained snapshot holds.
    retained_len: BTreeMap<u64, usize>,
    /// Snapshot indices a member vetted or produced itself.
    last_leader: Option<u64>,
    /// The first pending snapshot of member 3 was already restarted over.
    restarted_over_pending: bool,
    /// The host is not looking at member 3's pending snapshot: later appends reach it first.
    holding: bool,
    forger: Option<u64>,
    stalled: bool,
    lost_node: bool,
    tamper_held: bool,
    /// Snapshots the host refused: (member, index).
    rejected: BTreeSet<(u64, u64)>,
    counters: Tally,
}

/// The reference history: what a single node fed the workload in order holds. The workload is
/// its own reference (an oracle shares no code with the product), and the unfaulted run must
/// reproduce it exactly.
fn reference(spec: &Spec) -> &[Ev] {
    &spec.workload.events
}

impl Group {
    /// A group with no member yet; the caller opens them.
    fn new(dir: PathBuf, spec: Spec, inj: Arc<Injector>, counters: Tally) -> Self {
        Self {
            dir,
            nodes: Vec::new(),
            facts: BTreeMap::new(),
            isolated: BTreeSet::new(),
            faults: Faults::default(),
            rng: Rng::new(spec.seed),
            inj,
            spec,
            retained: BTreeMap::new(),
            retained_len: BTreeMap::new(),
            last_leader: None,
            restarted_over_pending: false,
            holding: false,
            forger: None,
            stalled: false,
            lost_node: false,
            tamper_held: false,
            rejected: BTreeSet::new(),
            counters,
        }
    }

    fn node_dir(&self, id: u64) -> PathBuf {
        self.dir.join(format!("node-{id}"))
    }

    fn slot(&self, id: u64) -> usize {
        VOTERS.iter().position(|v| *v == id).expect("a voter")
    }

    fn alive(&self, id: u64) -> bool {
        self.nodes[self.slot(id)].is_some()
    }

    fn open(dir: &Path, id: u64) -> Result<RaftNode, OpenFailure> {
        match catch_unwind(AssertUnwindSafe(|| RaftNode::open(dir, id, &VOTERS))) {
            Err(payload) => {
                explain(
                    "the product panicked in open",
                    &super::cutover::panic_text(payload.as_ref()),
                );
                Err(OpenFailure::Panicked)
            }
            Ok(Err(error)) => Err(OpenFailure::Refused(error)),
            Ok(Ok(node)) => Ok(node),
        }
    }

    /// Run one call on member `id`: a panic at a failpoint is a crash, an error is a failed
    /// persistence, and both end with the member opened again from its directory.
    fn call<T>(
        &mut self,
        id: u64,
        f: impl FnOnce(&mut RaftNode) -> Result<T, String>,
    ) -> Called<T> {
        let slot = self.slot(id);
        let Some(node) = self.nodes[slot].as_mut() else {
            return Called::Failed("the member is down".into());
        };
        self.inj.current.store(id, Ordering::SeqCst);
        let before = Facts::of(node);
        let outcome = catch_unwind(AssertUnwindSafe(|| f(node)));
        match outcome {
            Ok(Ok(value)) => {
                self.after_call(id);
                Called::Done(value)
            }
            Ok(Err(error)) => {
                if error.contains(INJECTED) {
                    add(&mut self.counters, ARM, "injected_io_errors", 1);
                }
                self.restart(id, Some(before), "failed");
                Called::Failed(error)
            }
            Err(payload) => {
                if payload.downcast_ref::<InjectedPanic>().is_some() {
                    add(&mut self.counters, ARM, "unwind_panics_observed", 1);
                } else {
                    explain(
                        "the product panicked in a call",
                        &super::cutover::panic_text(payload.as_ref()),
                    );
                    bump(&mut self.counters, ARM, "product_panics");
                }
                self.restart(id, Some(before), "crashed");
                Called::Crashed
            }
        }
    }

    /// What every completed call is checked for.
    fn after_call(&mut self, id: u64) {
        let slot = self.slot(id);
        if let Some(node) = self.nodes[slot].as_mut() {
            let facts = Facts::of(node);
            self.facts.insert(id, facts);
        }
        self.check_files(id);
        if let Some(view) = self.view_of(id) {
            let mut lens = self.inj.view_len.lock().expect("lens");
            let held = lens.entry(id).or_insert(0);
            *held = (*held).max(view.len());
        }
        let len = std::fs::metadata(self.node_dir(id).join("log"))
            .map(|m| m.len() as usize)
            .unwrap_or(0);
        self.inj.log_len.lock().expect("lens").insert(id, len);
    }

    /// The files of a member that are synced by a completed call: the log, the state and every
    /// snapshot payload (kept under the name of its own digest).
    fn member_files(&self, id: u64) -> Vec<String> {
        let mut names = vec!["log".to_string(), "state".to_string()];
        if let Ok(listing) = std::fs::read_dir(self.node_dir(id)) {
            for entry in listing.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.starts_with("snapshot-") && !name.ends_with(".tmp") {
                    names.push(name);
                }
            }
        }
        names
    }

    /// The durability probe over a member's files after a completed call: the log is synced
    /// by the append, the state and the snapshot by their atomic write.
    fn check_files(&mut self, id: u64) {
        if !self.probing() {
            return;
        }
        for name in self.member_files(id) {
            let path = self.node_dir(id).join(&name);
            if !path.exists() {
                continue;
            }
            match probe::has_unsynced_pages(&path) {
                Some(false) => self.counters.add("probe_clean_checked", 1),
                Some(true) => {
                    explain(
                        "a file holds unsynced pages after a completed call",
                        &format!("member {id}: {name}"),
                    );
                    bump(&mut self.counters, ARM, "unsynced_acks");
                }
                None => {}
            }
        }
    }

    fn probing(&self) -> bool {
        self.counters.get("probe_enabled") == 1
    }

    /// Close member `id` and open it again from its directory; compare with what it showed.
    fn restart(&mut self, id: u64, before: Option<Facts>, why: &str) {
        let slot = self.slot(id);
        self.nodes[slot] = None;
        add(&mut self.counters, ARM, "restarts", 1);
        let opened = Self::open(&self.node_dir(id), id);
        let mut node = match opened {
            Ok(node) => node,
            Err(OpenFailure::Panicked) => {
                bump(&mut self.counters, ARM, "product_panics");
                self.lost_node = true;
                return;
            }
            Err(OpenFailure::Refused(error)) => {
                explain(
                    &format!("member {id} could not be opened after it {why}"),
                    &error,
                );
                bump(&mut self.counters, ARM, "unjustified_refusals");
                self.lost_node = true;
                return;
            }
        };
        bump(&mut self.counters, ARM, "recoveries_member");
        self.counters.add("recoveries", 1);
        self.counters.add("recovered", 1);
        let now = Facts::of(&mut node);
        let held = self.inj.view_len.lock().expect("lens").get(&id).copied();
        if let Some(before) = before {
            let regressed = now.term < before.term
                || now.commit < before.commit
                || (!before.pending && !now.pending && now.applied < before.applied);
            if regressed {
                explain(
                    &format!("member {id} forgot after it {why}"),
                    &format!("before {before:?}, now {now:?}"),
                );
                bump(&mut self.counters, ARM, "member_regressed");
            }
        }
        self.facts.insert(id, now);
        self.nodes[slot] = Some(node);
        if let Some(view) = self.view_of(id).filter(|_| self.forger != Some(id)) {
            let acked = held.unwrap_or(0);
            if view.len() < acked {
                explain(
                    &format!("member {id} presents fewer events after it {why}"),
                    &format!("{} < {acked}", view.len()),
                );
                bump(&mut self.counters, ARM, "member_regressed");
            }
            let attempted = self.spec.workload.events.len();
            judge_logical(
                &mut self.counters,
                ARM,
                &self.spec.workload,
                acked.min(view.len()),
                attempted,
                &view,
            );
        }
        self.inj.log_len.lock().expect("lens").insert(
            id,
            std::fs::metadata(self.node_dir(id).join("log"))
                .map(|m| m.len() as usize)
                .unwrap_or(0),
        );
    }

    /// The history member `id` presents to the application: the events its snapshot's payload
    /// holds, then the events it holds above them. `None` while a snapshot is waiting to be
    /// vetted (the member presents nothing until the host has accounted for it).
    fn view_of(&mut self, id: u64) -> Option<Vec<CommittedEvent>> {
        let slot = self.slot(id);
        let node = self.nodes[slot].as_mut()?;
        if node.take_installed_snapshot().is_some() {
            return None;
        }
        let payload = FileRaftStorage::open(&self.dir.join(format!("node-{id}")), &VOTERS, &[])
            .ok()
            .map(|storage| storage.wl().snapshot().data.to_vec())
            .unwrap_or_default();
        let prefix = decode_payload(&payload).unwrap_or_default();
        let held = prefix.len() as u64;
        // The numbering a member presents: each commit index once, in order, the first no lower
        // than 1 and no higher than the one just above its snapshot.
        let mut previous: Option<u64> = None;
        let mut misnumbered = false;
        for event in node.committed_events() {
            let expected = previous.map(|last| last + 1);
            misnumbered |= expected.is_some_and(|next| event.index.0 != next)
                || (expected.is_none() && (event.index.0 < 1 || event.index.0 > held + 1));
            previous = Some(event.index.0);
        }
        let suffix: Vec<CommittedEvent> = node
            .committed_events()
            .iter()
            .filter(|event| event.index.0 > held)
            .cloned()
            .collect();
        if misnumbered {
            explain(
                "a member presents commit indices that are not consecutive from its snapshot",
                &format!("member {id}, {held} events in its snapshot"),
            );
            bump(&mut self.counters, ARM, "reorder_or_duplicate_index");
        }
        Some(logical(&prefix, &suffix))
    }

    fn deliver(&mut self, initial: Vec<RaftMessage>) {
        let mut queue: VecDeque<RaftMessage> = initial.into();
        for _ in 0..20_000 {
            let Some(front) = (!queue.is_empty()).then(|| {
                let position = if queue.len() > 1 && self.rng.below(1000) < self.faults.reorder {
                    self.counters.add("messages_reordered", 1);
                    self.rng.index(queue.len())
                } else {
                    0
                };
                queue.remove(position).expect("in range")
            }) else {
                return;
            };
            let message = front;
            if self.isolated.contains(&message.from) || self.isolated.contains(&message.to) {
                self.counters.add("messages_dropped", 1);
                self.snapshot_lost(&message);
                continue;
            }
            if self.rng.below(1000) < self.faults.drop {
                self.counters.add("messages_dropped", 1);
                self.snapshot_lost(&message);
                continue;
            }
            if self.rng.below(1000) < self.faults.duplicate {
                self.counters.add("messages_duplicated", 1);
                queue.push_back(message.clone());
            }
            let to = message.to;
            if !VOTERS.contains(&to) || !self.alive(to) {
                self.counters.add("messages_dropped", 1);
                self.snapshot_lost(&message);
                continue;
            }
            self.counters.add("messages_delivered", 1);
            let from = to;
            let sent = message.clone();
            match self.call(to, |node| node.step(message)) {
                Called::Done(produced) => {
                    self.release(from, &produced);
                    queue.extend(produced);
                    // The host looks at once for a snapshot the step installed.
                    self.vet_all();
                }
                // The member crashed or its write failed while it took the message.
                Called::Failed(_) | Called::Crashed => self.snapshot_lost(&sent),
            }
        }
        // The group did not go quiet.
        bump(&mut self.counters, ARM, "reference_mismatches");
    }

    /// The transport's part: a snapshot that did not arrive is reported to the sender, which
    /// would otherwise wait for it for ever.
    fn snapshot_lost(&mut self, message: &RaftMessage) {
        if message.msg_type != MSG_SNAPSHOT || !self.alive(message.from) {
            return;
        }
        let peer = message.to;
        if let Called::Done(messages) = self.call(message.from, |node| {
            node.report_snapshot(peer, ptr_ledger::SnapshotDelivery::Failed)
        }) {
            self.release(message.from, &messages);
        }
    }

    /// The promises a call's messages make are already in the member's files.
    fn release(&mut self, from: u64, messages: &[RaftMessage]) {
        let mut vote: Option<(u64, u64)> = None;
        let mut appended = 0u64;
        for message in messages {
            if message.from != from || message.reject {
                continue;
            }
            match message.msg_type {
                MSG_REQUEST_VOTE_RESPONSE => vote = Some((message.term, message.to)),
                MSG_APPEND_RESPONSE => appended = appended.max(message.index),
                _ => {}
            }
        }
        if vote.is_none() && appended == 0 {
            return;
        }
        let Ok(storage) = FileRaftStorage::open(&self.node_dir(from), &VOTERS, &[]) else {
            bump(&mut self.counters, ARM, "promise_not_durable");
            return;
        };
        let (term, voted, last) = {
            let core = storage.wl();
            (
                core.hard_state().term,
                core.hard_state().vote,
                core.last_index(),
            )
        };
        if let Some((promised_term, candidate)) = vote {
            add(&mut self.counters, ARM, "promises_checked", 1);
            add(&mut self.counters, ARM, "promises_vote", 1);
            if term < promised_term || voted != candidate {
                explain(
                    "a vote was granted that is not in the member's files",
                    &format!("member {from} term {term} vote {voted}, promised term {promised_term} to {candidate}"),
                );
                bump(&mut self.counters, ARM, "promise_not_durable");
            }
        }
        if appended > 0 {
            add(&mut self.counters, ARM, "promises_checked", 1);
            if last < appended {
                explain(
                    "an append was acknowledged that is not in the member's files",
                    &format!("member {from} last index {last}, acknowledged {appended}"),
                );
                bump(&mut self.counters, ARM, "promise_not_durable");
            }
        }
    }

    /// The leader of the highest term among the members that can speak.
    fn leader(&self) -> Option<u64> {
        VOTERS
            .iter()
            .filter(|id| !self.isolated.contains(id))
            .filter_map(|id| self.nodes[self.slot(*id)].as_ref())
            .filter(|node| node.is_leader())
            .max_by_key(|node| node.term())
            .map(RaftNode::id)
    }

    fn note_leader(&mut self) {
        let leader = self.leader();
        if leader.is_some() && leader != self.last_leader {
            if self.last_leader.is_some() {
                add(&mut self.counters, ARM, "leader_changes", 1);
            }
            self.last_leader = leader;
        }
    }

    /// The schedule's leader change: the leader is cut off and another member is elected by the
    /// two that can hear each other. Candidates are tried in turn, up to `ELECTION_ATTEMPTS`
    /// times under the router's faults and once more with them off, so the change does not
    /// depend on which votes the faults happened to drop. A group that still has no other
    /// leader then has not done what the plan asks of a baseline (`reference_mismatches`).
    fn depose_leader(&mut self) {
        let old = self.leader().unwrap_or(VOTERS[0]);
        self.isolated.insert(old);
        let candidates: Vec<u64> = VOTERS
            .iter()
            .copied()
            .filter(|id| *id != old && self.alive(*id))
            .collect();
        let armed = self.faults;
        let mut deposed = false;
        let mut used = 0;
        if !candidates.is_empty() {
            for attempt in 0..=params::ELECTION_ATTEMPTS {
                if attempt == params::ELECTION_ATTEMPTS {
                    self.faults = Faults::default();
                }
                used = attempt + 1;
                self.elect(candidates[attempt % candidates.len()]);
                if self.leader().is_some_and(|leader| leader != old) {
                    deposed = true;
                    break;
                }
            }
        }
        self.faults = armed;
        // Descriptive, in no rule: how many attempts the depositions used, the latest attempt
        // on which a leader changed, and the depositions that needed the faults-off attempt.
        self.counters.add("deposition_attempts_total", used as i64);
        if deposed {
            let latest = self
                .counters
                .get("deposition_attempts_max_run")
                .max(used as i64);
            self.counters.set("deposition_attempts_max_run", latest);
            if used > params::ELECTION_ATTEMPTS {
                self.counters.add("deposition_clean_attempt_needed", 1);
            }
        }
        if !deposed {
            explain(
                "the leader could not be deposed",
                &format!(
                    "member {old} still leads after {} attempts",
                    params::ELECTION_ATTEMPTS + 1
                ),
            );
            bump(&mut self.counters, ARM, "reference_mismatches");
        }
    }

    fn elect(&mut self, id: u64) {
        if let Called::Done(messages) = self.call(id, RaftNode::campaign) {
            self.release(id, &messages);
            self.deliver(messages);
        }
        self.note_leader();
    }

    /// A leader that can speak, electing one if there is none; `None` when the group cannot.
    fn ensure_leader(&mut self) -> Option<u64> {
        if let Some(leader) = self.leader() {
            return Some(leader);
        }
        for candidate in VOTERS {
            if self.isolated.contains(&candidate) || !self.alive(candidate) {
                continue;
            }
            self.elect(candidate);
            if let Some(leader) = self.leader() {
                self.resync();
                return Some(leader);
            }
        }
        self.stalled = true;
        self.counters.add("raft_stalled", 1);
        None
    }

    /// Two rounds of the leader's heartbeats, so that a new leader commits what it inherited.
    fn resync(&mut self) {
        for _ in 0..2 {
            if let Some(leader) = self.leader() {
                self.heartbeat(leader);
            }
        }
    }

    fn heartbeat(&mut self, id: u64) {
        for _ in 0..8 {
            let Called::Done(messages) = self.call(id, RaftNode::tick) else {
                return;
            };
            if !messages.is_empty() {
                self.release(id, &messages);
                self.deliver(messages);
                self.note_leader();
                return;
            }
        }
    }

    /// The number of events the leader has committed, as the application sees them.
    fn leader_len(&mut self, leader: u64) -> usize {
        self.view_of(leader).map_or(0, |view| view.len())
    }

    /// Propose the events from `from` up to `to` (exclusive) through the leader; the position
    /// the group has actually reached is read back from the leader afterwards, so an event
    /// lost in a crash is proposed again and one that survived is not.
    fn propose_until(&mut self, to: usize) {
        let mut attempts = 0;
        loop {
            let Some(leader) = self.ensure_leader() else {
                return;
            };
            let done = self.leader_len(leader);
            if done >= to || attempts > 3 * to {
                return;
            }
            attempts += 1;
            let event = self.spec.workload.events[done].to_ledger();
            match self.call(leader, |node| node.propose(event)) {
                Called::Done(messages) => {
                    self.release(leader, &messages);
                    self.deliver(messages);
                    self.note_leader();
                }
                Called::Failed(error) if !error.contains(INJECTED) => {
                    explain("a proposal a leader must accept was refused", &error);
                    bump(&mut self.counters, ARM, "reference_mismatches");
                    return;
                }
                Called::Failed(_) | Called::Crashed => {
                    self.resync();
                }
            }
            if self.leader().is_none() {
                self.note_leader();
            }
            // Let the followers learn the commit index of what was proposed. A message the
            // router dropped is sent again by the next heartbeat, so the leader is given a few
            // rounds to commit what it accepted before the same event is proposed a second time
            // (a proposal accepted and not yet committed is not a lost one).
            for _ in 0..params::PROPOSAL_SETTLE_ROUNDS {
                let Some(leader) = self.leader() else {
                    break;
                };
                self.heartbeat(leader);
                if self.leader_len(leader) > done {
                    break;
                }
            }
        }
    }

    /// Leader heartbeats until every member that is up holds the leader's history; judged
    /// against the pinned bound. Returns whether the group converged.
    fn converge(&mut self) -> bool {
        self.converge_rounds(true)
    }

    /// The same heartbeats while the router's faults may still be armed: the rounds are not
    /// judged and not counted (the bound is derived for a router that loses nothing, and the
    /// claim is about the heartbeats after the faults), the group is only brought as far as it
    /// goes before the next step.
    fn settle(&mut self) {
        self.converge_rounds(false);
    }

    fn converge_rounds(&mut self, judged: bool) -> bool {
        let Some(leader) = self.ensure_leader() else {
            return false;
        };
        for round in 1..=params::CONVERGENCE_ROUNDS {
            self.heartbeat(leader);
            if self.vet_all() {
                continue;
            }
            let target = self.leader_len(leader);
            let mut all = true;
            for id in VOTERS {
                if !self.alive(id) || (id == 3 && (self.tamper_held || self.holding)) {
                    continue;
                }
                if self.view_of(id).map_or(0, |view| view.len()) < target {
                    all = false;
                }
            }
            if all {
                if judged {
                    add(
                        &mut self.counters,
                        ARM,
                        "rounds_to_converge_total",
                        round as i64,
                    );
                    let max = self.counters.get("rounds_max_run").max(round as i64);
                    self.counters.set("rounds_max_run", max);
                }
                return true;
            }
        }
        if judged {
            explain(
                "the group did not converge within the bound",
                &format!("{} rounds", params::CONVERGENCE_ROUNDS),
            );
            bump(&mut self.counters, ARM, "convergence_bound_exceeded");
        }
        false
    }

    /// The host's part: take a snapshot a member was sent, vet it against the anchor the host
    /// retained, and account for the events it holds. Returns whether any member had one.
    fn vet_all(&mut self) -> bool {
        let mut any = false;
        for id in VOTERS {
            if !self.alive(id) {
                continue;
            }
            let slot = self.slot(id);
            let pending = self.nodes[slot]
                .as_mut()
                .and_then(RaftNode::take_installed_snapshot);
            let Some(installed) = pending else {
                continue;
            };
            if self.rejected.contains(&(id, installed.index)) {
                continue;
            }
            if id == 3 && self.holding {
                // The host puts the snapshot off (see `play`): the member is sent the entries
                // above it first and restarted over them.
                continue;
            }
            any = true;
            if id == 3 && self.spec.restart_before_resume && !self.restarted_over_pending {
                self.restarted_over_pending = true;
                add(
                    &mut self.counters,
                    ARM,
                    "restarts_between_install_and_resume",
                    1,
                );
                let before = self.facts.get(&id).copied();
                self.restart(
                    id,
                    before,
                    "restarted before the host accounted for its snapshot",
                );
                continue;
            }
            self.vet(id, installed.index, &installed.state, installed.digest);
        }
        any
    }

    fn vet(&mut self, id: u64, index: u64, state: &[u8], digest: [u8; 32]) {
        let Some(retained) = self.retained.get(&index).copied() else {
            explain(
                "a snapshot arrived that the host retained no anchor for",
                &format!("index {index}"),
            );
            bump(&mut self.counters, ARM, "reference_mismatches");
            return;
        };
        let genuine = retained.digest == digest;
        if self.spec.tamper && !genuine {
            // Baseline B1: the unanchored take of the same snapshot.
            add(&mut self.counters, ARM, "b1_recoveries", 1);
            // What the unanchored take would have made of the forgery: a runtime replayed from
            // the payload, asked about the history the genuine snapshot stands for.
            let events = decode_payload(state).unwrap_or_default();
            let genuine = self.retained_len.get(&index).copied().unwrap_or(0);
            let mut scratch = Tally::new(metrics::LIST);
            if let Ok(mut runtime) = ptr_runtime::PtrRuntime::replay(
                ptr_config::PtrConfig::default(),
                &logical(&events, &[]),
            ) {
                super::view::compare_runtime(
                    &mut scratch,
                    "b1",
                    &self.spec.workload,
                    &mut runtime,
                    genuine,
                    genuine,
                );
            }
            explain(
                "the unanchored take of the forgery",
                &format!(
                    "genuine {genuine}, payload {}, stale {}, mismatches {}",
                    events.len(),
                    scratch.get("stale_accepts"),
                    scratch.get("authority_mismatches")
                ),
            );
            if scratch.get("stale_accepts") > 0 {
                add(
                    &mut self.counters,
                    ARM,
                    "b1_stale_accepts",
                    scratch.get("stale_accepts"),
                );
                add(&mut self.counters, ARM, "b1_lie_images_accepted_stale", 1);
            }
        }
        let outcome = self.call(id, |node| {
            Ok(node.take_installed_snapshot_matching(retained))
        });
        match outcome {
            Called::Done(Ok(taken)) => {
                if !genuine {
                    explain(
                        "a snapshot with another digest than the retained anchor was taken",
                        &format!("index {index}"),
                    );
                    bump(&mut self.counters, ARM, "tampered_snapshot_accepted");
                }
                let Some(events) = decode_payload(&taken.state) else {
                    explain(
                        "a vetted snapshot whose payload does not decode",
                        &format!(
                            "member {id}, index {}, {} bytes, genuine {genuine}",
                            taken.index,
                            taken.state.len()
                        ),
                    );
                    bump(&mut self.counters, ARM, "recovered_not_prefix");
                    return;
                };
                let applied = events.len() as u64;
                match self.call(id, |node| node.resume_with_applied(applied)) {
                    Called::Done(()) => add(&mut self.counters, ARM, "snapshots_installed", 1),
                    Called::Failed(error) => {
                        explain("the host could not account for a vetted snapshot", &error);
                    }
                    Called::Crashed => {}
                }
            }
            Called::Done(Err(mismatch)) => {
                let _: SnapshotAnchorMismatch = mismatch;
                if genuine {
                    explain(
                        "a genuine snapshot was refused by its anchor",
                        &format!("{mismatch:?}"),
                    );
                    bump(&mut self.counters, ARM, "unjustified_refusals");
                } else {
                    self.tamper_held = true;
                    self.rejected.insert((id, index));
                    add(&mut self.counters, ARM, "tamper_controls", 1);
                    self.counters.add("snapshot_blocked", 1);
                }
            }
            Called::Failed(_) | Called::Crashed => {}
        }
    }

    /// Record a snapshot at the member's commit index; its payload is the application's state.
    fn record_snapshot(&mut self, id: u64, forged: bool) {
        let slot = self.slot(id);
        let Some(view) = self.view_of(id) else {
            return;
        };
        let Some(index) = self.nodes[slot].as_ref().map(RaftNode::raft_committed) else {
            return;
        };
        let events: Vec<Ev> = self.spec.workload.events[..view.len()].to_vec();
        let genuine = encode_payload(&events);
        let payload = if forged {
            let without: Vec<Ev> = events
                .iter()
                .filter(|e| !e.is_revocation())
                .cloned()
                .collect();
            encode_payload(&without)
        } else {
            genuine.clone()
        };
        self.retained_len.insert(index, events.len());
        // The host retains the anchor of the genuine payload out of band, by index.
        self.retained.insert(
            index,
            RetainedSnapshotAnchor {
                index,
                digest: sha256(&genuine),
            },
        );
        if let Called::Done(()) = self.call(id, |node| node.record_snapshot(index, payload)) {
            add(&mut self.counters, ARM, "snapshots_recorded", 1);
        }
    }

    /// The end of a scenario: the judgement of every member's history.
    fn finish(&mut self, converged: bool) {
        let n = self.spec.workload.events.len();
        let reference = reference(&self.spec).to_vec();
        for id in VOTERS {
            if !self.alive(id) {
                continue;
            }
            if self.forger == Some(id) {
                continue;
            }
            let Some(view) = self.view_of(id) else {
                if !self.tamper_held {
                    bump(&mut self.counters, ARM, "reference_divergence");
                }
                continue;
            };
            let acked = self
                .inj
                .view_len
                .lock()
                .expect("lens")
                .get(&id)
                .copied()
                .unwrap_or(0);
            bump(&mut self.counters, ARM, "members_replayed");
            judge_logical(
                &mut self.counters,
                ARM,
                &self.spec.workload,
                acked.min(view.len()),
                n,
                &view,
            );
            if converged && !self.stalled && view.len() != reference.len() {
                explain(
                    "a member does not hold the reference history",
                    &format!("member {id}: {} of {}", view.len(), reference.len()),
                );
                bump(&mut self.counters, ARM, "reference_divergence");
            }
        }
    }
}

enum OpenFailure {
    Panicked,
    Refused(String),
}

/// Run one schedule under one injection. Everything, nodes included, lives in the calling
/// thread, so that the whole run can be put under a watchdog.
pub fn run_life(dir: PathBuf, spec: Spec, inject: Inject, probing: bool, capture: bool) -> Run {
    label(format!(
        "{inject:?} seed {} a {} b {} lag {} snap {} mid {} brr {} faults {} tamper {}",
        spec.seed,
        spec.a,
        spec.b,
        spec.lag,
        spec.snapshots,
        spec.restart_mid,
        spec.restart_before_resume,
        spec.faults,
        spec.tamper
    ));
    let scenario = FailScenario::setup();
    let inj = Arc::new(Injector::default());
    inj.capturing.store(capture, Ordering::SeqCst);
    let owner = std::thread::current().id();
    for point in params::MEMBER_POINTS.iter().chain(&params::ERROR_POINTS) {
        let point: &'static str = point;
        let inj = Arc::clone(&inj);
        let dir = dir.clone();
        let crash = match &inject {
            Inject::Crash { point: p, target } if *p == point => Some(*target),
            _ => None,
        };
        // An I/O error is injected by the failpoint's own `return` action, which cannot share
        // the point with a callback: the run that injects it does not count that point's hits.
        if matches!(&inject, Inject::IoError { point: p, .. } if *p == point) {
            continue;
        }
        fail::cfg_callback(point, move || {
            if std::thread::current().id() != owner || !inj.armed.load(Ordering::SeqCst) {
                return;
            }
            let nth = {
                let mut hits = inj.hits.lock().expect("hits");
                let slot = hits.entry(point).or_insert(0);
                *slot += 1;
                *slot - 1
            };
            if point == "raft.append.after_write_before_sync"
                && inj.capturing.load(Ordering::SeqCst)
            {
                let node = inj.current.load(Ordering::SeqCst);
                let node_dir = dir.join(format!("node-{node}"));
                let files = read_files(&node_dir);
                let durable_log = inj
                    .log_len
                    .lock()
                    .expect("lens")
                    .get(&node)
                    .copied()
                    .unwrap_or(0);
                let acked_view = inj
                    .view_len
                    .lock()
                    .expect("lens")
                    .get(&node)
                    .copied()
                    .unwrap_or(0);
                inj.captures.lock().expect("captures").push(Capture {
                    node,
                    files,
                    durable_log,
                    acked_view,
                });
            }
            if crash == Some(nth) {
                inj.fired.store(true, Ordering::SeqCst);
                std::panic::panic_any(InjectedPanic);
            }
        })
        .expect("register a failpoint callback");
    }
    let mut counters = Tally::new(metrics::LIST);
    counters.set("probe_enabled", i64::from(probing));
    let mut group = Group::new(dir.clone(), spec.clone(), Arc::clone(&inj), counters);
    let opened: Result<Vec<RaftNode>, String> = VOTERS
        .iter()
        .map(|id| RaftNode::open(&dir.join(format!("node-{id}")), *id, &VOTERS))
        .collect();
    match opened {
        Err(error) => {
            explain("the first opening of a member was refused", &error);
            bump(&mut group.counters, ARM, "reference_mismatches");
        }
        Ok(nodes) => {
            group.nodes = nodes.into_iter().map(Some).collect();
            inj.armed.store(true, Ordering::SeqCst);
            if let Inject::IoError { point, target } = &inject {
                fail::cfg(*point, &format!("{target}*off->1*return")).expect("configure the error");
            }
            play(&mut group, &spec);
        }
    }
    let converged_counters = group.counters.counters();
    let hits = inj.hits.lock().expect("hits").clone();
    let captures: Vec<CaptureRow> = inj
        .captures
        .lock()
        .expect("captures")
        .iter()
        .map(|c| (c.node, c.files.clone(), c.durable_log, c.acked_view))
        .collect();
    let fired = inj.fired.load(Ordering::SeqCst);
    drop(group);
    drop(scenario);
    Run {
        counters: converged_counters,
        hits,
        captures,
        fired,
    }
}

/// The schedule itself.
fn play(group: &mut Group, spec: &Spec) {
    let n = spec.workload.events.len();
    group.elect(1);
    if group.leader().is_none() {
        group.stalled = true;
        return;
    }
    group.propose_until(spec.a);
    if spec.lag {
        group.isolated.insert(3);
    }
    group.propose_until(spec.b);
    if spec.snapshots {
        group.resync();
        for id in [1u64, 2] {
            if group.alive(id) {
                let forged = spec.tamper && id == 1;
                if forged {
                    group.forger = Some(1);
                }
                group.record_snapshot(id, forged);
            }
        }
    }
    if spec.restart_mid != 0 {
        let id = spec.restart_mid;
        let before = group.facts.get(&id).copied();
        group.restart(id, before, "restarted by the schedule");
    }
    group.isolated.clear();
    // A member that is to be restarted over its pending snapshot first gets the entries above it:
    // what it keeps committed there is what the restart must be able to open again.
    group.holding = spec.restart_before_resume && !spec.tamper;
    let mut converged = group.converge();
    if spec.faults {
        group.faults = Faults {
            drop: params::FAULT_DROP_PERMILLE,
            duplicate: params::FAULT_DUPLICATE_PERMILLE,
            reorder: params::FAULT_REORDER_PERMILLE,
        };
    }
    if spec.depose {
        // The leader is cut off, another member is elected by the two that can hear each
        // other, and the group goes on under it; the old leader rejoins below.
        group.depose_leader();
        group.propose_until((spec.b + n) / 2);
        group.isolated.clear();
        group.settle();
    }
    if !spec.tamper {
        group.propose_until(n);
    }
    group.faults = Faults::default();
    if group.holding {
        group.holding = false;
        let slot = group.slot(3);
        let pending = group.nodes[slot]
            .as_mut()
            .and_then(RaftNode::take_installed_snapshot)
            .is_some();
        if pending {
            group.restarted_over_pending = true;
            add(
                &mut group.counters,
                ARM,
                "restarts_between_install_and_resume",
                1,
            );
            let before = group.facts.get(&3).copied();
            group.restart(
                3,
                before,
                "restarted before the host accounted for its snapshot",
            );
        }
    }
    if spec.tamper {
        // The blocked member restarts: it must still be blocked.
        let slot = group.slot(3);
        if group.tamper_held && group.nodes[slot].is_some() {
            let before = group.facts.get(&3).copied();
            group.restart(3, before, "restarted while its snapshot was refused");
            let pending = group.nodes[slot]
                .as_mut()
                .and_then(RaftNode::take_installed_snapshot)
                .is_some();
            if !pending {
                let adopted = FileRaftStorage::open(&group.node_dir(3), &VOTERS, &[])
                    .ok()
                    .map(|storage| storage.wl().snapshot().data.to_vec())
                    .and_then(|data| decode_payload(&data))
                    .unwrap_or_default();
                if adopted.iter().all(|event| !event.is_revocation()) && !adopted.is_empty() {
                    explain(
                        "a refused snapshot is the member's state after a restart",
                        "no pending snapshot",
                    );
                    bump(&mut group.counters, ARM, "tampered_snapshot_accepted");
                }
            }
        }
        group.finish(false);
        return;
    }
    converged = group.converge() && converged;
    group.finish(converged);
    group.counters.add("scenarios_done", 1);
}

fn read_files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter_map(|path| {
                    let name = path.file_name()?.to_string_lossy().into_owned();
                    Some((name, std::fs::read(&path).ok()?))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The power-loss images of the captured appends: the member's log as it stands between the
/// write and the sync of an append, with the bytes written after the synced length kept or
/// lost by each policy. The member is opened again over the image and must present at least
/// what it had acknowledged.
pub fn image_cases(
    tally: &mut Tally,
    runs: &mut RunDir,
    spec: &Spec,
    captures: &[CaptureRow],
    rng: &mut Rng,
) -> Result<(), String> {
    let mut picked: Vec<usize> = Vec::new();
    while picked.len() < params::IMAGE_CAPTURES_PER_SCENARIO.min(captures.len()) {
        let at = rng.index(captures.len());
        if !picked.contains(&at) {
            picked.push(at);
        }
    }
    picked.sort_unstable();
    for at in picked {
        let (node, files, durable_log, acked_view) = &captures[at];
        let Some(written) = files.get("log") else {
            continue;
        };
        for policy in Policy::HONEST {
            let image = power_loss(written, *durable_log, policy, rng);
            check_in_model(&image, written, *durable_log)?;
            if image.lost_bytes {
                add(
                    tally,
                    ARM,
                    "images_lost_bytes",
                    (written.len() - image.bytes.len()).max(1) as i64,
                );
            }
            if image.damaged {
                add(tally, ARM, "images_damaged", 1);
            }
            let _ = damage_shape(&image, written, *durable_log);
            let dir = runs.fresh();
            let node_dir = dir.join(format!("node-{node}"));
            std::fs::create_dir_all(&node_dir).map_err(|e| e.to_string())?;
            for (name, bytes) in files {
                let bytes = if name == "log" { &image.bytes } else { bytes };
                std::fs::write(node_dir.join(name), bytes).map_err(|e| e.to_string())?;
            }
            let opened = Group::open(&node_dir, *node);
            bump(tally, ARM, "recoveries_member");
            tally.add("recoveries", 1);
            match opened {
                Err(OpenFailure::Panicked) => bump(tally, ARM, "product_panics"),
                Err(OpenFailure::Refused(error)) => {
                    explain(
                        &format!(
                            "a power-loss image of member {node}'s log was refused ({})",
                            policy.name()
                        ),
                        &error,
                    );
                    tally.add("refusals", 1);
                    bump(tally, ARM, "unjustified_refusals");
                }
                Ok(member) => {
                    tally.add("recovered", 1);
                    let payload = FileRaftStorage::open(&node_dir, &VOTERS, &[])
                        .ok()
                        .map(|storage| storage.wl().snapshot().data.to_vec())
                        .unwrap_or_default();
                    let prefix = decode_payload(&payload).unwrap_or_default();
                    let held = prefix.len() as u64;
                    let suffix: Vec<CommittedEvent> = member
                        .committed_events()
                        .iter()
                        .filter(|event| event.index.0 > held)
                        .cloned()
                        .collect();
                    let view = logical(&prefix, &suffix);
                    if view.len() < *acked_view {
                        bump(tally, ARM, "member_regressed");
                    }
                    judge_logical(
                        tally,
                        ARM,
                        &spec.workload,
                        (*acked_view).min(view.len()),
                        spec.workload.events.len(),
                        &view,
                    );
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}

/// The number of hits to crash at, drawn from the observing run's count, or `None` when the
/// schedule never reaches the point.
pub fn target_for(
    hits: &BTreeMap<&'static str, usize>,
    point: &str,
    rng: &mut Rng,
) -> Option<usize> {
    let total = hits.get(point).copied().unwrap_or(0);
    (total > 0).then(|| rng.index(total))
}

/// Keep clippy quiet about the counter helper being used by the tests only.
#[allow(dead_code)]
fn _unused(_: &AtomicUsize) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_message_type_numbers_are_the_generated_enums() {
        // Only the heartbeat constructor is public; the others follow its numbering (the
        // generated enum of raft-proto counts from MsgHup = 0).
        assert_eq!(ptr_ledger::raft_node::message_type_heartbeat() as i32, 8);
        assert_eq!(MSG_APPEND_RESPONSE, 4);
        assert_eq!(MSG_REQUEST_VOTE_RESPONSE, 6);
    }

    #[test]
    fn a_forgeable_schedule_revokes_a_generation_that_is_still_live() {
        let mut rng = Rng::new(7);
        let spec = spec_for("tamper", &mut rng);
        assert!(spec.tamper);
        let prefix = forgeable_prefix(&spec.workload).expect("forgeable");
        assert_eq!(spec.b, prefix);
    }

    /// A group of three fresh members in a scratch directory, member 1 elected.
    fn group_led_by_one(label: &str) -> (Group, PathBuf) {
        let dir = std::env::temp_dir().join(format!("l002-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut rng = Rng::new(1001);
        let spec = spec_for("heal", &mut rng);
        let mut group = Group::new(
            dir.clone(),
            spec,
            Arc::new(Injector::default()),
            Tally::new(metrics::LIST),
        );
        group.nodes = VOTERS
            .iter()
            .map(|id| {
                Some(RaftNode::open(&dir.join(format!("node-{id}")), *id, &VOTERS).expect("open"))
            })
            .collect();
        group.elect(1);
        assert_eq!(group.leader(), Some(1));
        (group, dir)
    }

    #[test]
    fn a_deposed_leader_is_replaced_even_when_the_faults_drop_every_message() {
        // Not a fault the plan draws (12 per cent): the worst the router can do, to show that
        // the leader change does not depend on what the faults happened to drop.
        let _scenario = FailScenario::setup();
        let (mut group, dir) = group_led_by_one("depose-all-dropped");
        group.faults = Faults {
            drop: 1000,
            duplicate: 0,
            reorder: 0,
        };
        group.depose_leader();
        assert!(group.leader().is_some_and(|leader| leader != 1));
        assert_eq!(group.counters.get("leader_changes"), 1);
        assert_eq!(group.counters.get("reference_mismatches"), 0);
        assert_eq!(group.faults.drop, 1000, "the armed faults are put back");
        // Every attempt under the faults failed, so the one with them off is what deposed it,
        // and the record says so.
        let attempts = params::ELECTION_ATTEMPTS as i64 + 1;
        assert_eq!(group.counters.get("deposition_attempts_total"), attempts);
        assert_eq!(group.counters.get("deposition_attempts_max_run"), attempts);
        assert_eq!(group.counters.get("deposition_clean_attempt_needed"), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_deposition_on_a_healthy_network_is_counted_on_its_first_attempt() {
        let _scenario = FailScenario::setup();
        let (mut group, dir) = group_led_by_one("depose-healthy");
        group.depose_leader();
        assert!(group.leader().is_some_and(|leader| leader != 1));
        assert_eq!(group.counters.get("deposition_attempts_total"), 1);
        assert_eq!(group.counters.get("deposition_attempts_max_run"), 1);
        assert_eq!(group.counters.get("deposition_clean_attempt_needed"), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_leader_that_cannot_be_deposed_is_a_failed_baseline_not_a_quiet_miss() {
        let _scenario = FailScenario::setup();
        let (mut group, dir) = group_led_by_one("depose-impossible");
        group.nodes[1] = None;
        group.nodes[2] = None;
        group.depose_leader();
        assert_eq!(group.counters.get("reference_mismatches"), 1);
        assert_eq!(group.counters.get("deposition_clean_attempt_needed"), 0);
        assert_eq!(
            group.counters.get("deposition_attempts_max_run"),
            0,
            "none deposed"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rounds_under_the_faults_are_neither_judged_nor_counted() {
        let _scenario = FailScenario::setup();
        let (mut group, dir) = group_led_by_one("settle-unjudged");
        // A member that cannot be brought to the leader's history in the rounds: nothing is
        // reported for a phase that is not judged, and the judged phase reports it.
        group.nodes[2] = None;
        group.isolated.insert(2);
        let leader = group.ensure_leader().expect("a leader");
        let _ = group.leader_len(leader);
        group.settle();
        assert_eq!(group.counters.get("convergence_bound_exceeded"), 0);
        assert_eq!(group.counters.get("rounds_to_converge_total"), 0);
        group.converge();
        assert!(group.counters.get("rounds_to_converge_total") > 0);
        let _ = std::fs::remove_dir_all(dir);
    }
}
