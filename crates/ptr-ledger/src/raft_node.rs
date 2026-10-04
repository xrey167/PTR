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
use crate::raft_storage::FileRaftStorage;
use crate::{
    decode_event, encode_event, event_codec_version, integrity, CommittedEvent, LedgerEvent,
    BASE_CODEC_VERSION, LEDGER_CODEC_VERSION,
};
use ptr_types::CommitIndex;
use raft::codec::Message as _;
use raft::prelude::{
    ConfChange, ConfChangeType, Config as RaftConfig, Entry, EntryType, Message, RawNode,
};
use raft::StateRole;
use std::collections::BTreeMap;
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
/// inside. A PTR deployment puts a PTRCS005 compacted snapshot here rather than a
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

/// Which codec version each member reports it reads, by raft id.
///
/// Supplied by the host, which asks each member (`LEDGER_CODEC_VERSION` of the
/// build it runs). A report is the operator's word about a member, not a proof:
/// what it buys is that raising the ceiling is a decision taken against a stated
/// view of every voter, and refused when that view has a gap.
pub type CodecReports = BTreeMap<u64, u16>;

/// Marks a raft snapshot payload that carries the group's codec ceiling ahead of
/// the application's own state.
///
/// Written only once the ceiling is above the base version, so a group that has
/// never raised one sends exactly the application's bytes — which is what a
/// member on an older build expects during a rolling upgrade, when no ceiling can
/// have been raised yet.
const SNAPSHOT_CEILING_MAGIC: &[u8; 8] = b"PTRRSC01";

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
    /// A ceiling this leader proposed and the raft log position it was proposed
    /// at, until an entry at or past that position is applied.
    ///
    /// While it is set a voter may not be added: the joiner's report was checked
    /// against the ceiling in force, and a raise already on its way would hand it
    /// records it cannot read.
    pending_ceiling: Option<(u16, u64)>,
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
        let recovered_applied = storage.wl().applied_events();

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
            installed: None,
            pending_ceiling: None,
            id,
        };
        let _ = recovered_count;
        member.apply_committed(recovered)?;
        Ok(member)
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
    /// Recording a snapshot is what makes [`Storage::snapshot`] answerable, which is
    /// what lets a leader catch up a member it has compacted past — a leader with no
    /// recorded snapshot has nothing to send, and says so rather than inventing one.
    ///
    /// Once the group's codec ceiling is above the base version, the payload raft
    /// carries is prefixed with it ([`SNAPSHOT_CEILING_MAGIC`]): the record that
    /// raised it is below the snapshot, so a member caught up by this snapshot
    /// would otherwise never learn it.
    pub fn record_snapshot(&mut self, index: u64, state: Vec<u8>) -> Result<(), String> {
        let ceiling = self.codec_ceiling();
        let payload = if ceiling > BASE_CODEC_VERSION {
            let mut wrapped = Vec::with_capacity(SNAPSHOT_CEILING_MAGIC.len() + 2 + state.len());
            wrapped.extend_from_slice(SNAPSHOT_CEILING_MAGIC);
            wrapped.extend_from_slice(&ceiling.to_le_bytes());
            wrapped.extend_from_slice(&state);
            wrapped
        } else {
            state
        };
        let store = self.node.raft.raft_log.store.clone();
        let outcome = store.wl().record_snapshot(index, payload);
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
    pub fn resume_with_applied(&mut self, applied_events: u64) -> Result<(), String> {
        self.applied_events = applied_events;
        self.installed = None;
        // Durable, because the events this count covers are described by a
        // payload rather than by records this member still holds. A restart that
        // counted from zero would hand the next event an index the snapshot
        // already covers, and two different events would claim it.
        let store = self.node.raft.raft_log.store.clone();
        let outcome = store.wl().set_applied_events(applied_events);
        outcome.map_err(|error| error.to_string())
    }

    /// Propose a change to the group's membership.
    ///
    /// Only a leader may: a follower proposing one would be asking peers to
    /// accept a configuration nobody agreed, so this refuses rather than
    /// forwarding. The change is not in effect when this returns — it is an
    /// entry like any other, and it takes hold when it commits and
    /// [`RaftNode::apply_committed`] applies it, which is what makes a
    /// membership change survive a restart.
    ///
    /// A voter is added only if `reports` says it reads the committed codec
    /// ceiling — otherwise it would join a group whose log it cannot apply — and
    /// not while a ceiling this leader proposed is still on its way.
    pub fn propose_membership(
        &mut self,
        change: MembershipChange,
        reports: &CodecReports,
    ) -> Result<Vec<Message>, String> {
        if !self.is_leader() {
            return Err("only a leader may propose a membership change".into());
        }
        if let MembershipChange::AddVoter(id) = change {
            if let Some((pending, _)) = self.pending_ceiling {
                return Err(format!(
                    "PTR_LEDGER_CODEC_CEILING: a ceiling of {pending} is pending; add the \
                     voter once it has been applied"
                ));
            }
            let ceiling = self.codec_ceiling();
            match reports.get(&id) {
                Some(reported) if *reported >= ceiling => {}
                Some(reported) => {
                    return Err(format!(
                        "PTR_LEDGER_CODEC_CEILING: voter {id} reads codec version {reported}, \
                         below the committed {ceiling}"
                    ))
                }
                None => {
                    return Err(format!(
                        "PTR_LEDGER_CODEC_CEILING: no codec report for joining voter {id}"
                    ))
                }
            }
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
        self.node.campaign().map_err(|error| error.to_string())?;
        self.drain()
    }

    /// Advance this member's logical clock by one tick.
    ///
    /// Ticks are a count, not a duration: a test decides how many happen and when,
    /// so an election timeout is reached deterministically rather than by waiting.
    pub fn tick(&mut self) -> Result<Vec<Message>, String> {
        self.node.tick();
        self.drain()
    }

    /// Propose an event.
    ///
    /// A member that does not lead refuses — raft drops the proposal — rather than
    /// buffering it for a leadership it may never regain. A buffered proposal that
    /// is replayed later is a write the group never ordered.
    ///
    /// A record whose tag a later codec version introduced is refused until the
    /// group has committed a ceiling at least that high
    /// ([`RaftNode::propose_codec_ceiling`]): every member applies every record,
    /// and a member on an older build stops at a tag it does not know.
    pub fn propose(&mut self, event: LedgerEvent) -> Result<Vec<Message>, String> {
        if matches!(event, LedgerEvent::CodecCeilingRaised { .. }) {
            return Err(
                "PTR_LEDGER_CODEC_CEILING: a ceiling is raised with propose_codec_ceiling, \
                 against every voter's report"
                    .into(),
            );
        }
        let needs = event_codec_version(&event);
        let ceiling = self.codec_ceiling();
        if needs > ceiling {
            return Err(format!(
                "PTR_LEDGER_CODEC_CEILING: this record needs codec version {needs} and the \
                 group has committed {ceiling}"
            ));
        }
        crate::check_encodable(&event).map_err(|error| error.to_string())?;
        self.node
            .propose(Vec::new(), encode_event(&event))
            .map_err(|error| error.to_string())?;
        self.drain()
    }

    /// The codec version every voter has been reported to read, as committed.
    pub fn codec_ceiling(&self) -> u16 {
        let store = self.node.raft.raft_log.store.clone();
        let ceiling = store.wl().codec_ceiling();
        ceiling
    }

    /// Propose raising the group's codec ceiling to `version`.
    ///
    /// Leader only, and only when `reports` names **every** current voter at
    /// `version` or above — this member included, at its own build's version.
    /// Refused while a membership change is pending, because the voter set the
    /// reports were checked against is then about to change. The ceiling takes
    /// hold when the record commits and is applied, durably, like any other.
    pub fn propose_codec_ceiling(
        &mut self,
        version: u16,
        reports: &CodecReports,
    ) -> Result<Vec<Message>, String> {
        if !self.is_leader() {
            return Err("only a leader may propose a codec ceiling".into());
        }
        let ceiling = self.codec_ceiling();
        if version <= ceiling || version > LEDGER_CODEC_VERSION {
            return Err(format!(
                "PTR_LEDGER_CODEC_CEILING: {version} is not above the committed {ceiling} \
                 and at most this build's {LEDGER_CODEC_VERSION}"
            ));
        }
        if self.node.raft.has_pending_conf() {
            return Err(
                "PTR_LEDGER_CODEC_CEILING: a membership change is pending; the voters the \
                 reports describe are about to change"
                    .into(),
            );
        }
        for voter in self.voters() {
            let reported = if voter == self.id {
                Some(LEDGER_CODEC_VERSION)
            } else {
                reports.get(&voter).copied()
            };
            match reported {
                Some(reported) if reported >= version => {}
                Some(reported) => {
                    return Err(format!(
                        "PTR_LEDGER_CODEC_CEILING: voter {voter} reads codec version \
                         {reported}, below {version}"
                    ))
                }
                None => {
                    return Err(format!(
                        "PTR_LEDGER_CODEC_CEILING: no codec report for voter {voter}"
                    ))
                }
            }
        }
        let event = LedgerEvent::CodecCeilingRaised { version };
        crate::check_encodable(&event).map_err(|error| error.to_string())?;
        let at = self.node.raft.raft_log.last_index() + 1;
        self.node
            .propose(Vec::new(), encode_event(&event))
            .map_err(|error| error.to_string())?;
        self.pending_ceiling = Some((version, at));
        self.drain()
    }

    /// Take one message from another member.
    pub fn step(&mut self, message: Message) -> Result<Vec<Message>, String> {
        self.node.step(message).map_err(|error| error.to_string())?;
        self.drain()
    }

    /// Persist, apply and collect outbound messages for one ready state.
    ///
    /// The order is raft's requirement, not a preference: `messages` may go out
    /// before the write, `persisted_messages` only after it. Sending a vote or an
    /// append that the sender has not yet flushed is how a crash turns into a
    /// promise nobody kept.
    fn drain(&mut self) -> Result<Vec<Message>, String> {
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
                let (ceiling, state) = split_snapshot_ceiling(snapshot.data.as_ref())?;
                if let Some(ceiling) = ceiling {
                    store
                        .wl()
                        .raise_codec_ceiling(ceiling)
                        .map_err(|error| error.to_string())?;
                }
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
        for entry in entries {
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
            if let Some((_, at)) = self.pending_ceiling {
                if entry.index >= at {
                    self.pending_ceiling = None;
                }
            }
            if entry.data.is_empty() || entry.get_entry_type() != EntryType::EntryNormal {
                continue;
            }
            if self.installed.is_some() {
                return Err(
                    "a snapshot is waiting to be accounted for; call resume_with_applied first"
                        .into(),
                );
            }
            let event = decode_event(entry.data.as_ref()).map_err(|error| error.to_string())?;
            if let LedgerEvent::CodecCeilingRaised { version } = event {
                let store = self.node.raft.raft_log.store.clone();
                store
                    .wl()
                    .raise_codec_ceiling(version)
                    .map_err(|error| error.to_string())?;
            }
            self.applied_events += 1;
            let index = CommitIndex(self.applied_events);
            self.committed.push(CommittedEvent { index, event });
        }
        Ok(())
    }
}

/// Separate a raft snapshot payload into the codec ceiling it carries, if any, and
/// the application's own bytes.
///
/// A ceiling above this build's version is refused: the member would be joining a
/// group that has agreed to records it cannot read.
fn split_snapshot_ceiling(data: &[u8]) -> Result<(Option<u16>, Vec<u8>), String> {
    let Some(rest) = data.strip_prefix(SNAPSHOT_CEILING_MAGIC.as_slice()) else {
        return Ok((None, data.to_vec()));
    };
    let (ceiling, state) = rest
        .split_first_chunk::<2>()
        .ok_or("raft snapshot ends inside its codec ceiling")?;
    let ceiling = u16::from_le_bytes(*ceiling);
    if ceiling <= BASE_CODEC_VERSION || ceiling > LEDGER_CODEC_VERSION {
        return Err(format!(
            "PTR_LEDGER_CODEC_CEILING: the snapshot carries codec version {ceiling}, this \
             build reads up to {LEDGER_CODEC_VERSION}"
        ));
    }
    Ok((Some(ceiling), state.to_vec()))
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
