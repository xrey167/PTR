//! One Raft node, with its messages handed back instead of dropped.
//!
//! The single-node harness could throw outbound messages away, because a
//! one-member group has nobody to send them to. Everything a group does — voting,
//! replicating, deposing a leader — is in those messages, so a node that composes
//! into a cluster has to surrender them, and something else has to carry them.
//!
//! This type deliberately does **not** know about transport. Messages come out and
//! go in; whether that happens through a socket, an ALPN on `ptr-net`, or a test
//! that delivers them by hand is not its business. That separation is what makes a
//! partition testable without a timer: nothing here is driven by wall-clock time,
//! so a test decides exactly which message arrives and which does not.
use crate::raft_storage::{carries_ledger_event, AppliedEvents, FileRaftStorage, StoragePoisoned};
use crate::{decode_event, encode_event, integrity, CommittedEvent, LedgerEvent, RetentionFloor};
use ptr_types::CommitIndex;
use raft::codec::Message as _;
use raft::prelude::{
    ConfChange, ConfChangeType, Config as RaftConfig, Entry, EntryType, Message, RawNode,
    SnapshotStatus,
};
use raft::StateRole;
use std::path::Path;

/// Raft's own message type, re-exported so a transport can name it without
/// depending on the raft crate itself.
///
/// One pin in one place: a second crate declaring its own `raft` dependency could
/// drift to a different version, and the two would then disagree about the wire
/// format of the very messages they exchange.
pub use raft::prelude::Message as RaftMessage;

/// Bound on one encoded raft message. A frame past this is refused rather than
/// allocated for: the sender is not trusted to bound what the receiver reads.
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// State a leader sent to a member it had compacted past.
///
/// The payload is opaque here on purpose: a snapshot is the *application's* state,
/// and this layer carrying it does not entitle it to an opinion about what is
/// inside. A PTR deployment puts a PTRCS004 compacted snapshot here rather than a
/// second artifact invented for raft.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstalledSnapshot {
    /// The committed position the payload describes.
    pub index: u64,
    /// The term at that position.
    pub term: u64,
    /// The application's own encoding of its state at that position.
    pub state: Vec<u8>,
    /// Digest of `state`, computed **here** from the bytes that arrived rather
    /// than read out of them.
    ///
    /// A digest carried inside the payload would say only that the payload agrees
    /// with itself. This one exists so a host that retained an anchor out of band
    /// can compare against something the sender did not get to choose.
    pub digest: [u8; 32],
}

/// What a host retained out of band about a snapshot it is willing to accept.
///
/// Raft is not Byzantine fault tolerant and this does not make it so: a member
/// that accepts a snapshot from the leader its protocol elected is the design,
/// not a gap. What this allows is the deployment that *does* retain anchors
/// elsewhere — an operator, a control plane, a second channel — to say which
/// state it expects, so a member does not restore from whatever arrived simply
/// because it arrived from the current leader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainedSnapshotAnchor {
    /// The committed position the payload must describe.
    pub index: u64,
    /// The digest the payload's bytes must have.
    pub digest: [u8; 32],
}

/// Why a member refused to take an installed snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotAnchorMismatch {
    /// No snapshot has been installed, so there is nothing to match.
    NothingInstalled,
    /// The payload describes a different committed position than was retained.
    Index { installed: u64, retained: u64 },
    /// The payload's bytes are not the ones the retained anchor describes.
    Digest,
}

impl SnapshotAnchorMismatch {
    /// Stable diagnostic code for this refusal.
    pub fn code(self) -> &'static str {
        match self {
            Self::NothingInstalled => "PTR_SNAPSHOT_NONE_INSTALLED",
            Self::Index { .. } => "PTR_SNAPSHOT_ANCHOR_INDEX",
            Self::Digest => "PTR_SNAPSHOT_ANCHOR_DIGEST",
        }
    }
}

/// A change to who votes in this group.
///
/// One voter at a time, deliberately. Raft's single-step membership change is
/// safe only because consecutive configurations overlap in a majority; changing
/// two at once can split the group into two disjoint majorities that each
/// believe they can commit. Offering a batch API would make that mistake
/// available, so it is not offered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MembershipChange {
    /// Add a voter. The new member must already exist and be reachable, or the
    /// group loses the ability to commit until it is.
    AddVoter(u64),
    /// Remove a voter.
    RemoveVoter(u64),
}

/// How the snapshot a leader sent to a peer ended, as only the transport knows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotDelivery {
    /// The peer took it.
    Delivered,
    /// The message was dropped, the peer was down or refused it.
    Failed,
}

/// The refusal every state-changing call gets once the member has failed (see
/// [`RaftNode::is_failed`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeFailed;

impl NodeFailed {
    /// Stable diagnostic code, also the start of the error text.
    pub const CODE: &'static str = "PTR_RAFT_NODE_FAILED";
}

impl std::fmt::Display for NodeFailed {
    /// Name the code and what to do about it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: an error after raft handed this member a ready state left the raft state ahead \
             of what the host applied; restart the member so it replays its log",
            Self::CODE
        )
    }
}

impl std::error::Error for NodeFailed {}

/// The refusal [`RaftNode::resume_with_applied`] gives when no snapshot is waiting to be
/// accounted for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NoSnapshotPending;

impl NoSnapshotPending {
    /// Stable diagnostic code, also the start of the error text.
    pub const CODE: &'static str = "PTR_RAFT_NO_SNAPSHOT_PENDING";
}

impl std::fmt::Display for NoSnapshotPending {
    /// Name the code and why the call is refused.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: no installed snapshot is waiting to be accounted for; a count set now would \
             renumber every later event",
            Self::CODE
        )
    }
}

impl std::error::Error for NoSnapshotPending {}

/// A Raft member over durable state.
pub struct RaftNode {
    node: RawNode<FileRaftStorage>,
    committed: Vec<CommittedEvent>,
    /// How many events this member has applied in total, including the ones a
    /// snapshot accounts for. The ledger index of the next event continues from
    /// here rather than from the length of what is still held.
    applied_events: u64,
    /// A snapshot that landed and has not been accounted for yet. While this is
    /// set, applying anything further is refused: the events below it are described
    /// by a payload only the application can read, and continuing as though it were
    /// empty would renumber every event that follows.
    installed: Option<InstalledSnapshot>,
    /// Committed entries held back while `installed` is set, in order (raft counts them as
    /// applied, so it will not hand them over again): applied by
    /// [`RaftNode::resume_with_applied`].
    deferred: Vec<Entry>,
    /// Set by any error after raft handed the member a ready state (or after the member took
    /// its deferred entries): raft has consumed committed entries and a light ready that the
    /// host did not apply, so carrying on would renumber every later event. Only a restart
    /// (which replays the log) clears it.
    failed: bool,
    id: u64,
}

impl std::fmt::Debug for RaftNode {
    /// Describe the node's position without printing its log.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaftNode")
            .field("id", &self.id)
            .field("role", &self.node.raft.state)
            .field("term", &self.node.raft.term)
            .field("committed", &self.committed.len())
            .finish()
    }
}

impl RaftNode {
    /// Open a member's durable state and recover what it had committed.
    ///
    /// `voters` is only used the first time: an existing node's recorded
    /// configuration is the authority, for the reason
    /// [`FileRaftStorage::open`] gives.
    pub fn open(dir: &Path, id: u64, voters: &[u64]) -> Result<Self, String> {
        let storage = FileRaftStorage::open(dir, voters, &[]).map_err(|error| error.to_string())?;
        let recovered = storage.wl().committed_entries();
        let applied = storage.wl().hard_state().commit;
        // A snapshot installed and not yet accounted for (the process stopped in between) is
        // offered to the host again: it has not been vetted, and the member applies nothing
        // above it until it is.
        let (pending, recovered_applied) = {
            let core = storage.wl();
            match (core.applied(), core.pending_snapshot()) {
                (AppliedEvents::Counted(count), _) => (None, count),
                (AppliedEvents::SnapshotPending, snapshot) => (
                    snapshot.map(|snapshot| InstalledSnapshot {
                        index: snapshot.index,
                        term: snapshot.term,
                        digest: integrity::sha256(&snapshot.payload),
                        state: snapshot.payload,
                    }),
                    0,
                ),
            }
        };

        let config = RaftConfig {
            id,
            election_tick: 10,
            heartbeat_tick: 3,
            max_size_per_msg: 1024 * 1024,
            max_inflight_msgs: 256,
            applied,
            ..Default::default()
        };
        config.validate().map_err(|error| error.to_string())?;

        let logger = slog::Logger::root(slog::Discard, slog::o!());
        let node = RawNode::new(&config, storage, &logger).map_err(|error| error.to_string())?;
        let recovered_count = recovered.len() as u64;
        let mut member = Self {
            node,
            committed: Vec::new(),
            // Recovered from the state file, so replay continues the numbering a
            // snapshot established rather than restarting it.
            applied_events: recovered_applied,
            installed: pending,
            deferred: Vec::new(),
            failed: false,
            id,
        };
        let _ = recovered_count;
        if member.installed.is_some() {
            // Entries above a snapshot that is waiting to be accounted for were committed and
            // are kept; they are applied, in order, once the host has accounted for it.
            member.deferred = recovered;
        } else {
            member.apply_committed(recovered)?;
        }
        Ok(member)
    }

    /// Whether the member's storage hit an I/O error past a point of no return and refuses
    /// every write from now on ([`StoragePoisoned`]): the member answers nothing, and is
    /// restarted from what its disk holds.
    ///
    /// Limit: a poisoned member still answers the reads of committed state (`is_leader`,
    /// `role`, `term`, `voters`, `committed_events`, `raft_committed`); it can say it leads
    /// while every write is refused, and raft's in-memory term and commit may be ahead of the
    /// rolled-back storage memory. Callers check this before relying on them.
    pub fn is_poisoned(&self) -> bool {
        self.node.raft.raft_log.store.wl().is_poisoned()
    }

    /// Whether an error after raft handed the member a ready state (a refused or failed append,
    /// hard-state or commit write, an entry that would not apply, a deferred entry lost) left
    /// raft ahead of what the host applied. The storage need not be poisoned for this: raft hands
    /// committed entries over once, so the member refuses every state-changing call
    /// ([`NodeFailed`]) and is restarted, which replays its log.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// The refusal every call that could change the member's state gets once it is poisoned or
    /// failed.
    fn ensure_sound(&self) -> Result<(), String> {
        if self.is_poisoned() {
            return Err(StoragePoisoned.to_string());
        }
        if self.failed {
            return Err(NodeFailed.to_string());
        }
        Ok(())
    }

    /// This member's id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Leader, follower, candidate or pre-candidate.
    pub fn role(&self) -> StateRole {
        self.node.raft.state
    }

    /// Whether this member currently believes it leads.
    ///
    /// *Believes*: a leader that has been partitioned away still says yes until it
    /// learns otherwise, which is exactly why believing it is not enough to commit.
    pub fn is_leader(&self) -> bool {
        self.node.raft.state == StateRole::Leader
    }

    /// The term this member is in.
    pub fn term(&self) -> u64 {
        self.node.raft.term
    }

    /// The raft log position this member considers committed.
    pub fn raft_committed(&self) -> u64 {
        self.node.raft.raft_log.committed
    }

    /// The ledger events this member holds, in commit order.
    ///
    /// After a snapshot is installed this holds only what came *after* it: the
    /// events below are described by the snapshot's payload, which this layer cannot
    /// read.
    pub fn committed_events(&self) -> &[CommittedEvent] {
        &self.committed
    }

    /// How many events this member has applied in total, snapshot included.
    pub fn applied_events(&self) -> u64 {
        self.applied_events
    }

    /// Record this member's state at a committed index and discard the log below it.
    ///
    /// The payload is the application's; this layer neither builds nor inspects it.
    /// Recording a snapshot is what makes `Storage::snapshot` answerable, which is
    /// what lets a leader catch up a member it has compacted past — a leader with no
    /// recorded snapshot has nothing to send, and says so rather than inventing one.
    ///
    /// The floor is taken from this member's own log, so nothing here checks that a retained
    /// snapshot backs the payload; a host that has one uses
    /// [`RaftNode::record_snapshot_within`].
    pub fn record_snapshot(&mut self, index: u64, state: Vec<u8>) -> Result<(), String> {
        self.ensure_sound()?;
        let store = self.node.raft.raft_log.store.clone();
        let outcome = store.wl().record_snapshot(index, state);
        outcome.map_err(|error| error.to_string())
    }

    /// [`RaftNode::record_snapshot`] bounded by the ledger's retention floor.
    ///
    /// `floor` is what the application's durable snapshot covers, in ledger events (for PTR,
    /// `CompactedSnapshot::covers`). The log is discarded only through the raft index whose
    /// prefix produced exactly that many events: a floor short of it is refused with
    /// `PTR_RAFT_COMPACTION_ABOVE_RETENTION_FLOOR` (the entries between would be retained
    /// nowhere), one past it with `PTR_RAFT_SNAPSHOT_FLOOR_MISMATCH` (a member installing the
    /// payload would apply those events again), and a member holding an installed snapshot it
    /// has not accounted for with `PTR_RAFT_SNAPSHOT_UNACCOUNTED`. The log makes the
    /// conversion between the two numberings, since only it knows which raft indexes carried
    /// no event. Every refusal leaves the files untouched. Whether `floor` is backed by a
    /// retained artifact stays the host's claim.
    pub fn record_snapshot_within(
        &mut self,
        index: u64,
        state: Vec<u8>,
        floor: RetentionFloor,
    ) -> Result<(), String> {
        self.ensure_sound()?;
        let store = self.node.raft.raft_log.store.clone();
        let outcome = store.wl().record_snapshot_within(index, state, floor);
        outcome.map_err(|error| error.to_string())
    }

    /// Take the snapshot a leader installed, if one arrived.
    ///
    /// The caller must follow it with [`RaftNode::resume_with_applied`], because only
    /// the caller knows how many events the payload accounts for.
    pub fn take_installed_snapshot(&mut self) -> Option<InstalledSnapshot> {
        self.installed.clone()
    }

    /// Take an installed snapshot only if it is the one a host retained an anchor
    /// for, out of band.
    ///
    /// [`RaftNode::take_installed_snapshot`] returns whatever the elected leader
    /// sent, which rules out a *stranger* — the sender is the authenticated
    /// connection — but not the leader itself. This is the variant for a
    /// deployment that knows, by some other route, which state it expects.
    ///
    /// A mismatch refuses and **leaves the snapshot installed**: the member stays
    /// blocked from applying anything further rather than silently continuing as
    /// though no snapshot had arrived, which is the outcome that would renumber
    /// every event after it.
    pub fn take_installed_snapshot_matching(
        &mut self,
        retained: RetainedSnapshotAnchor,
    ) -> Result<InstalledSnapshot, SnapshotAnchorMismatch> {
        let installed = self
            .installed
            .as_ref()
            .ok_or(SnapshotAnchorMismatch::NothingInstalled)?;
        if installed.index != retained.index {
            return Err(SnapshotAnchorMismatch::Index {
                installed: installed.index,
                retained: retained.index,
            });
        }
        // Compared in constant time is not the point here — this is an integrity
        // check against a retained value, not a secret — but it is compared
        // against the digest computed from the arriving bytes, never one read out
        // of them.
        if installed.digest != retained.digest {
            return Err(SnapshotAnchorMismatch::Digest);
        }
        Ok(installed.clone())
    }

    /// Account for an installed snapshot: how many events its payload covers.
    ///
    /// Until this is called, applying further entries is refused. A member that
    /// carried on as though the snapshot were empty would number the next event 1
    /// and disagree with every other member about what that index means.
    ///
    /// The count is made durable first and the member's own fields change only after that: a
    /// failed write leaves the snapshot installed and offered, the count as it was and the held
    /// entries held, so the call can be repeated. With no snapshot pending the call is refused
    /// ([`NoSnapshotPending`]). If applying the held entries then fails part-way, the rest are
    /// lost to the member (raft will not hand them over again): it has failed ([`NodeFailed`]).
    pub fn resume_with_applied(&mut self, applied_events: u64) -> Result<(), String> {
        self.ensure_sound()?;
        // Refused when nothing is pending: the count would overwrite a durable one and renumber
        // every later event (a retry after a success is such a call).
        if self.installed.is_none() {
            return Err(NoSnapshotPending.to_string());
        }
        // Durable, because the events this count covers are described by a
        // payload rather than by records this member still holds. A restart that
        // counted from zero would hand the next event an index the snapshot
        // already covers, and two different events would claim it.
        let store = self.node.raft.raft_log.store.clone();
        let outcome = store.wl().set_applied_events(applied_events);
        outcome.map_err(|error| error.to_string())?;
        self.applied_events = applied_events;
        self.installed = None;
        let deferred = std::mem::take(&mut self.deferred);
        // Raft counts these entries as applied and will not hand them over again, so an error
        // part-way loses the rest: the member has failed.
        self.apply_committed(deferred)
            .inspect_err(|_| self.failed = true)
    }

    /// Tell the group how the snapshot it sent to `peer` ended: delivered when the peer
    /// took it, otherwise (the message was dropped, the peer was down or refused it) a failure.
    ///
    /// Only the transport knows. A leader that has sent a member a snapshot stops sending it
    /// anything else until it hears how that ended, and a member that never received it, or
    /// lost it in a crash before it was durable, never says; without this report the leader
    /// waits for ever and the member is never caught up. A failure makes the leader try again
    /// after its next heartbeat interval.
    pub fn report_snapshot(
        &mut self,
        peer: u64,
        outcome: SnapshotDelivery,
    ) -> Result<Vec<Message>, String> {
        self.ensure_sound()?;
        self.node.report_snapshot(
            peer,
            match outcome {
                SnapshotDelivery::Delivered => SnapshotStatus::Finish,
                SnapshotDelivery::Failed => SnapshotStatus::Failure,
            },
        );
        self.drain()
    }

    /// Tell the group that the last message sent to `peer` did not arrive, so that a leader
    /// that was streaming entries to it falls back to probing instead of filling a window nobody
    /// is reading.
    pub fn report_unreachable(&mut self, peer: u64) -> Result<Vec<Message>, String> {
        self.ensure_sound()?;
        self.node.report_unreachable(peer);
        self.drain()
    }

    /// Propose a change to the group's membership.
    ///
    /// Only a leader may: a follower proposing one would be asking peers to
    /// accept a configuration nobody agreed, so this refuses rather than
    /// forwarding. The change is not in effect when this returns — it is an
    /// entry like any other, and it takes hold when it commits and
    /// `RaftNode::apply_committed` applies it, which is what makes a
    /// membership change survive a restart.
    pub fn propose_membership(&mut self, change: MembershipChange) -> Result<Vec<Message>, String> {
        self.ensure_sound()?;
        if !self.is_leader() {
            return Err("only a leader may propose a membership change".into());
        }
        let mut cc = ConfChange::default();
        match change {
            MembershipChange::AddVoter(id) => {
                cc.set_change_type(ConfChangeType::AddNode);
                cc.node_id = id;
            }
            MembershipChange::RemoveVoter(id) => {
                cc.set_change_type(ConfChangeType::RemoveNode);
                cc.node_id = id;
            }
        }
        self.node
            .propose_conf_change(Vec::new(), cc)
            .map_err(|error| error.to_string())?;
        self.drain()
    }

    /// The voters this member currently believes the group has, in ascending
    /// order.
    ///
    /// *Believes*: a member that has not yet applied a committed change still
    /// reports the old set, for the same reason `is_leader` is a belief.
    ///
    /// Sorted because raft's own `ConfState` keeps voters in whatever order the
    /// changes arrived — `[3, 1, 4, 2]` for a group of four — and a membership
    /// answer that depends on arrival order is one callers would compare wrongly.
    pub fn voters(&self) -> Vec<u64> {
        let store = self.node.raft.raft_log.store.clone();
        let mut voters = store.wl().conf_state().voters.clone();
        voters.sort_unstable();
        voters
    }

    /// Stand for election, returning the messages that asks for votes.
    pub fn campaign(&mut self) -> Result<Vec<Message>, String> {
        self.ensure_sound()?;
        self.node.campaign().map_err(|error| error.to_string())?;
        self.drain()
    }

    /// Advance this member's logical clock by one tick.
    ///
    /// Ticks are a count, not a duration: a test decides how many happen and when,
    /// so an election timeout is reached deterministically rather than by waiting.
    pub fn tick(&mut self) -> Result<Vec<Message>, String> {
        self.ensure_sound()?;
        self.node.tick();
        self.drain()
    }

    /// Propose an event.
    ///
    /// A member that does not lead refuses — raft drops the proposal — rather than
    /// buffering it for a leadership it may never regain. A buffered proposal that
    /// is replayed later is a write the group never ordered.
    pub fn propose(&mut self, event: LedgerEvent) -> Result<Vec<Message>, String> {
        self.ensure_sound()?;
        crate::check_encodable(&event).map_err(|error| error.to_string())?;
        self.node
            .propose(Vec::new(), encode_event(&event))
            .map_err(|error| error.to_string())?;
        self.drain()
    }

    /// Take one message from another member.
    pub fn step(&mut self, message: Message) -> Result<Vec<Message>, String> {
        self.ensure_sound()?;
        self.node.step(message).map_err(|error| error.to_string())?;
        self.drain()
    }

    /// Persist, apply and collect outbound messages for one ready state.
    ///
    /// The order is raft's requirement, not a preference: `messages` may go out
    /// before the write, `persisted_messages` only after it. Sending a vote or an
    /// append that the sender has not yet flushed is how a crash turns into a
    /// promise nobody kept.
    ///
    /// An error here, after `ready()` has handed over committed entries and a light ready that
    /// raft will not hand over again, marks the member failed whether or not the storage was
    /// poisoned: it would otherwise keep running ahead of what the host applied.
    fn drain(&mut self) -> Result<Vec<Message>, String> {
        self.drain_ready().inspect_err(|_| self.failed = true)
    }

    fn drain_ready(&mut self) -> Result<Vec<Message>, String> {
        let mut outbound = Vec::new();
        while self.node.has_ready() {
            let store = self.node.raft.raft_log.store.clone();
            let mut ready = self.node.ready();

            outbound.extend(ready.take_messages());

            if !ready.snapshot().is_empty() {
                let snapshot = ready.snapshot().clone();
                let metadata = snapshot.get_metadata().clone();
                store
                    .wl()
                    .apply_snapshot(snapshot.clone())
                    .map_err(|error| error.to_string())?;
                // The events this member held below the snapshot are now described
                // by the payload rather than by its own log, so they stop being
                // this member's to report. What replaces them is the caller's to
                // decide, which is why nothing further applies until it has.
                self.committed.clear();
                let state = snapshot.data.to_vec();
                self.installed = Some(InstalledSnapshot {
                    index: metadata.index,
                    term: metadata.term,
                    digest: integrity::sha256(&state),
                    state,
                });
            }

            let committed = ready.take_committed_entries();

            if !ready.entries().is_empty() {
                store
                    .wl()
                    .append(ready.entries())
                    .map_err(|error| error.to_string())?;
            }

            if let Some(hard_state) = ready.hs() {
                store
                    .wl()
                    .set_hard_state(hard_state)
                    .map_err(|error| error.to_string())?;
            }

            outbound.extend(ready.take_persisted_messages());

            self.apply_committed(committed)?;

            let mut light = self.node.advance(ready);
            if let Some(commit) = light.commit_index() {
                store
                    .wl()
                    .set_commit(commit)
                    .map_err(|error| error.to_string())?;
            }
            outbound.extend(light.take_messages());
            self.apply_committed(light.take_committed_entries())?;
            self.node.advance_apply();
        }
        Ok(outbound)
    }

    /// Apply decided entries, skipping the ones that carry no event.
    ///
    /// A leader's first entry in a new term is empty, and configuration changes are
    /// not ledger events, so the commit index and the ledger's own index are
    /// different sequences on purpose.
    fn apply_committed(&mut self, entries: Vec<Entry>) -> Result<(), String> {
        let mut entries = entries.into_iter();
        while let Some(entry) = entries.next() {
            // A configuration change is not a ledger event, but it is not
            // nothing either: until it is applied here raft's own view of the
            // group is unchanged, so an entry that committed would have no
            // effect and the recorded configuration would disagree with the
            // log that produced it.
            if entry.get_entry_type() == EntryType::EntryConfChange {
                let mut cc = ConfChange::default();
                cc.merge_from_bytes(entry.data.as_ref())
                    .map_err(|error| error.to_string())?;
                let state = self
                    .node
                    .apply_conf_change(&cc)
                    .map_err(|error| error.to_string())?;
                let store = self.node.raft.raft_log.store.clone();
                store
                    .wl()
                    .set_conf_state(state)
                    .map_err(|error| error.to_string())?;
                continue;
            }
            // The predicate snapshot recording counts by, so the ledger index assigned here and
            // the retention floor checked there agree.
            if !carries_ledger_event(&entry) {
                continue;
            }
            if self.installed.is_some() {
                // Not applied now, kept for later: raft does not hand these entries over again,
                // so the entry and what follows it are applied by `resume_with_applied`. This is
                // not an error: the call that brought them also has messages to hand back, and
                // an error would drop them. The host sees the wait through
                // `take_installed_snapshot`.
                self.deferred.push(entry);
                self.deferred.extend(entries);
                return Ok(());
            }
            let event = decode_event(entry.data.as_ref()).map_err(|error| error.to_string())?;
            self.applied_events = self
                .applied_events
                .checked_add(1)
                .ok_or_else(|| "the count of applied events is exhausted".to_string())?;
            let index = CommitIndex(self.applied_events);
            self.committed.push(CommittedEvent { index, event });
        }
        Ok(())
    }
}

/// The heartbeat message type, for callers that build a message without naming
/// the raft crate.
///
/// A transport test needs *some* well-formed message to send, and this keeps it
/// from declaring its own `raft` dependency just to name a constant.
pub fn message_type_heartbeat() -> raft::prelude::MessageType {
    raft::prelude::MessageType::MsgHeartbeat
}

/// Encode a raft message for a transport.
pub fn encode_message(message: &Message) -> Result<Vec<u8>, String> {
    let bytes = message
        .write_to_bytes()
        .map_err(|error| error.to_string())?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err("raft message is over the bound".into());
    }
    Ok(bytes)
}

/// Decode a raft message from a transport, or refuse it.
///
/// A frame that does not decode is dropped, never partially applied: a message is
/// a claim about a term and a log position, and half of one claims nothing.
pub fn decode_message(bytes: &[u8]) -> Result<Message, String> {
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err("raft message is over the bound".into());
    }
    let mut message = Message::default();
    message
        .merge_from_bytes(bytes)
        .map_err(|error| error.to_string())?;
    Ok(message)
}
