//! The global workspace contract: a bounded, verifier-gated, broadcast blackboard for Pods.
//!
//! This is a contract, not a model. Pods submit typed [`Proposal`]s; each cycle the workspace
//! admits at most [`WorkspaceConfig::slots`] of them, and broadcasts one immutable
//! [`WorkspaceSnapshot`] to every subscriber. Nothing here learns, and nothing a proposal carries
//! can change how it is judged.
//!
//! # Cycle
//!
//! [`Workspace::run_cycle`] starts at revision `r`, takes the host's [`CycleHost`] (verifier,
//! its identity, and the author registry) and produces a snapshot at revision `r + 1`. It is
//! two-phase: everything is computed into a plan without touching the workspace, and the plan is
//! committed afterwards with no fallible step. A verifier that panics is caught and judged
//! `Unknown` with the hard finding `verifier-panic` (counted); a panic anywhere else leaves the
//! workspace exactly as it was.
//!
//! 1. Subscriptions whose Pod generation the registry reports revoked are dropped.
//! 2. Incumbents are retired, each recorded: [`RetireReason::AuthorRevoked`] (registry),
//!    [`RetireReason::Aged`] ([`SlotDecay`]), [`RetireReason::Explicit`]
//!    ([`Workspace::request_retire`]), and [`RetireReason::Contradicted`] when
//!    [`Revalidation`] re-runs the verifier on the incumbent's original proposal and the gate no
//!    longer passes.
//! 3. Every pending proposal is judged in proposal-id order: stale (see below), revoked author,
//!    then the host verifier through [`check_verification_gate`]: status `Pass`, no hard finding,
//!    level at least [`WorkspaceConfig::required_level`], score never read (invariant 11). The
//!    rule is that of `Candidate::passes` in `ptr-feedback`. `Fail`, `Unknown` and `Disputed`
//!    never occupy a slot.
//! 4. Passing proposals compete. The order is the effective bid (the requested bid clamped by
//!    [`WorkspaceConfig::max_bid`] and the registry's per-author ceiling), then SHA-256 over
//!    (seed, new revision, proposal id), then proposal id. Content already held, or won earlier
//!    in the cycle, is a duplicate and keeps its first author; an author at
//!    [`WorkspaceConfig::max_slots_per_author`] is refused; a proposal that finds no free slot
//!    may displace an incumbent only under an opt-in [`Displacement::Policy`] (below). Otherwise
//!    it is outbid.
//! 5. One snapshot is built and offered to every subscriber's bounded mailbox. A full mailbox
//!    refuses the new snapshot for that subscriber (counted, reported in the [`CycleReport`]); it
//!    never blocks the cycle and never evicts what the subscriber has not yet read.
//!
//! Every proposal that does not enter a slot gets a [`Refusal`] in the snapshot with its reason
//! and the verifier findings raised against it, so findings are broadcast. Pending proposals
//! never survive a cycle: a loser resubmits (a proposal is valid again as long as it is not
//! stale).
//!
//! # Staleness
//!
//! A proposal computed against revision `p` is judged at revision `r >= p`. With
//! [`WorkspaceConfig::max_staleness`] `= 0` (the default) only `p == r` is current, like the
//! effect boundary's revision check. With `max_staleness = k`, a slow Pod's proposal is still
//! current when `r - p <= k` **and no slot changed (was admitted, retired or displaced) after
//! revision `p`**: its view of the slots is then exactly the present one.
//!
//! # Redaction
//!
//! Verifier messages are free text that may echo private evidence, so by default snapshots carry
//! finding **codes and hard flags only** ([`FindingVisibility::HostOnly`]); the bounded messages
//! go to the caller of `run_cycle` in [`CycleReport::host_findings`]. A host that opts into
//! [`FindingVisibility::Public`] publishes untrusted text to every subscriber, including the
//! finding code and message, which may carry text derived from proposals.
//!
//! # Determinism
//!
//! For one pending set, one workspace state, one pure verifier and one registry the result is a
//! function of those alone: submission order does not matter once the same set is pending.
//! `submit` itself is order dependent where a bound is hit ([`SubmitError::PendingFull`],
//! [`SubmitError::AuthorPendingFull`]), which is why a per-author quota exists. The verifier must
//! be a deterministic function of the [`Proposal`] and its [`VerifierIdentity`] is recorded in
//! every snapshot.
//!
//! # Authority
//!
//! Slot content is workspace state derived from verified proposals; it is
//! [`Authority::Provisional`] (invariant 4) and promotes nothing. [`Proposal`] and
//! [`ProposalInit`] have no field for a gate, a threshold, a verifier, a level or a report, and
//! [`WorkspaceConfig`] has no setter, so no submitted value can change how it is judged. The
//! compile-fail example pins the first half and its twin shows the same value compiles without
//! the extra field:
//!
//! ```compile_fail,E0560
//! use ptr_pods::{Bid, PayloadRef, ProposalInit, Workspace};
//! use ptr_types::{Generation, PodId, TypeId, VerificationLevel};
//! fn init(ws: &Workspace) -> ProposalInit {
//!     ProposalInit {
//!         author: PodId::from("p"),
//!         author_generation: Generation(1),
//!         content: [1; 32],
//!         payload: PayloadRef::new(TypeId::from("t"), [2; 32]),
//!         computed_against: ws.revision(),
//!         bid: Bid::new(1),
//!         required_level: VerificationLevel::Unverified,
//!     }
//! }
//! ```
//!
//! ```
//! use ptr_pods::{Bid, PayloadRef, ProposalInit, Workspace};
//! use ptr_types::{Generation, PodId, TypeId};
//! fn init(ws: &Workspace) -> ProposalInit {
//!     ProposalInit {
//!         author: PodId::from("p"),
//!         author_generation: Generation(1),
//!         content: [1; 32],
//!         payload: PayloadRef::new(TypeId::from("t"), [2; 32]),
//!         computed_against: ws.revision(),
//!         bid: Bid::new(1),
//!     }
//! }
//! # let _ = init;
//! ```
//!
//! # Displacement
//!
//! Off by default. Under [`Displacement::Policy`], a challenger that passed the gate and finds no
//! free slot may take an incumbent's slot only if all hold: the incumbent survived this cycle's
//! revalidation; it is not pinned ([`Workspace::pin`]); it has been resident at least
//! `min_residency` cycles; the challenger's verification level is at least the incumbent's; the
//! challenger's effective bid exceeds the incumbent's by at least `margin` (a bid alone never
//! displaces, it only ranks proposals that passed the gate at a sufficient level); and the
//! per-cycle and per-author caps are not exhausted. The victim is the incumbent with the lowest
//! (bid, level, admission revision, slot index). The displaced content is recorded with
//! [`RetireReason::Displaced`].

mod config;
mod cycle;
mod gate;
mod proposal;
mod snapshot;

pub use config::{
    Displacement, DisplacementPolicy, FindingVisibility, Revalidation, SlotDecay,
    UnboundedResidencyRisk, WorkspaceConfig, WorkspaceConfigError, WorkspaceConfigInit,
    MAX_PENDING_PROPOSALS, MAX_RETIRED_LOG, MAX_SUBSCRIBER_MAILBOX, MAX_WORKSPACE_SLOTS,
    MAX_WORKSPACE_SUBSCRIBERS,
};
pub use cycle::{
    AuthorRegistry, CycleHost, CycleReport, Standing, UnrestrictedAuthors,
    MAX_HOST_FINDINGS_PER_CYCLE, VERIFIER_PANIC_CODE,
};
pub use gate::{check_verification_gate, GateRefusal};
pub use proposal::{
    Bid, PayloadRef, Proposal, ProposalInit, SlotIndex, SubscriberId, WorkspaceProposalError,
    WorkspaceProposalId, WorkspaceRevision,
};
pub use snapshot::{
    Authority, FindingRecord, HostFinding, Refusal, RefusalKind, RefusalReason, RetireReason,
    RetiredRecord, SlotContent, VerifierIdentity, WorkspaceSnapshot, MAX_FINDINGS_PER_REFUSAL,
    MAX_FINDING_CODE_BYTES, MAX_FINDING_MESSAGE_BYTES, MAX_SNAPSHOT_FINDINGS,
    MAX_VERIFIER_NAME_BYTES,
};

use ptr_types::{Generation, PodId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

/// Monotone counters; nothing the workspace refuses or drops goes uncounted.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorkspaceCounters {
    pub(super) cycles: u64,
    pub(super) submitted: u64,
    pub(super) submit_pending_full: u64,
    pub(super) submit_author_full: u64,
    pub(super) submit_rejected: u64,
    pub(super) admitted: u64,
    pub(super) retired: u64,
    pub(super) displaced: u64,
    pub(super) contradicted: u64,
    pub(super) author_revoked_retired: u64,
    pub(super) refused: [u64; 8],
    pub(super) delivered: u64,
    pub(super) mailbox_overflow: u64,
    pub(super) retired_log_evicted: u64,
    pub(super) verifier_panics: u64,
    pub(super) subscriptions_revoked: u64,
}

impl WorkspaceCounters {
    pub fn cycles(&self) -> u64 {
        self.cycles
    }

    /// Proposals accepted into the pending area.
    pub fn submitted(&self) -> u64 {
        self.submitted
    }

    /// Submissions refused because the pending area was full.
    pub fn submit_pending_full(&self) -> u64 {
        self.submit_pending_full
    }

    /// Submissions refused because the author's pending quota was full.
    pub fn submit_author_full(&self) -> u64 {
        self.submit_author_full
    }

    /// Submissions refused for a future revision or as a duplicate.
    pub fn submit_rejected(&self) -> u64 {
        self.submit_rejected
    }

    pub fn admitted(&self) -> u64 {
        self.admitted
    }

    /// Content retired for any reason.
    pub fn retired(&self) -> u64 {
        self.retired
    }

    pub fn displaced(&self) -> u64 {
        self.displaced
    }

    pub fn contradicted(&self) -> u64 {
        self.contradicted
    }

    pub fn author_revoked_retired(&self) -> u64 {
        self.author_revoked_retired
    }

    pub fn refused(&self, kind: RefusalKind) -> u64 {
        self.refused[kind.index()]
    }

    pub fn refused_total(&self) -> u64 {
        self.refused.iter().sum()
    }

    /// Snapshot deliveries into a mailbox.
    pub fn delivered(&self) -> u64 {
        self.delivered
    }

    /// Snapshot deliveries refused by a full mailbox.
    pub fn mailbox_overflow(&self) -> u64 {
        self.mailbox_overflow
    }

    /// Retired records pushed out of the bounded retired log.
    pub fn retired_log_evicted(&self) -> u64 {
        self.retired_log_evicted
    }

    /// Verifier calls that panicked and were judged `Unknown`.
    pub fn verifier_panics(&self) -> u64 {
        self.verifier_panics
    }

    /// Subscriptions dropped because their Pod generation was revoked.
    pub fn subscriptions_revoked(&self) -> u64 {
        self.subscriptions_revoked
    }
}

/// A refused [`Workspace::submit`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SubmitError {
    /// The pending area is at [`WorkspaceConfig::max_pending`].
    PendingFull,
    /// The author holds [`WorkspaceConfig::max_pending_per_author`] pending proposals.
    AuthorPendingFull,
    /// The proposal claims a revision the workspace has not reached.
    FutureRevision {
        computed_against: WorkspaceRevision,
        current: WorkspaceRevision,
    },
    /// An identical proposal is already pending.
    Duplicate,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PendingFull => f.write_str("pending area is full"),
            Self::AuthorPendingFull => f.write_str("author pending quota is full"),
            Self::FutureRevision {
                computed_against,
                current,
            } => write!(
                f,
                "proposal computed against future revision {} (current {})",
                computed_against.0, current.0
            ),
            Self::Duplicate => f.write_str("an identical proposal is already pending"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// A refused control operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkspaceError {
    SlotOutOfRange(SlotIndex),
    SlotEmpty(SlotIndex),
    SubscribersFull,
    DuplicateSubscriber,
    UnknownSubscriber(SubscriberId),
    RevisionExhausted,
    /// An internal bound was found violated by [`Workspace::check_invariants`].
    InvariantViolated(&'static str),
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlotOutOfRange(s) => write!(f, "slot {} out of range", s.get()),
            Self::SlotEmpty(s) => write!(f, "slot {} is empty", s.get()),
            Self::SubscribersFull => f.write_str("subscriber limit reached"),
            Self::DuplicateSubscriber => f.write_str("pod generation already subscribed"),
            Self::UnknownSubscriber(id) => write!(f, "unknown subscriber {}", id.get()),
            Self::RevisionExhausted => f.write_str("workspace revision exhausted"),
            Self::InvariantViolated(what) => write!(f, "workspace invariant violated: {what}"),
        }
    }
}

impl std::error::Error for WorkspaceError {}

/// Request to retire the content of a slot at the next cycle.
#[derive(Clone, Copy, Debug)]
pub struct RetireRequest {
    pub slot: SlotIndex,
}

/// Request to pin or unpin a slot. Pinned content is never displaced (it may still age, be
/// revoked or be contradicted).
#[derive(Clone, Copy, Debug)]
pub struct PinRequest {
    pub slot: SlotIndex,
}

/// Request to subscribe a Pod generation to broadcasts.
#[derive(Clone, Debug)]
pub struct SubscribeRequest {
    pub pod: PodId,
    pub generation: Generation,
}

/// Per-subscriber delivery statistics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubscriberStats {
    pub queued: usize,
    pub delivered: u64,
    pub overflowed: u64,
}

pub(super) struct Subscriber {
    pub(super) pod: PodId,
    pub(super) generation: Generation,
    pub(super) mailbox: VecDeque<Arc<WorkspaceSnapshot>>,
    pub(super) delivered: u64,
    pub(super) overflowed: u64,
}

/// A bounded global workspace; see the module documentation for the cycle.
pub struct Workspace {
    pub(super) config: WorkspaceConfig,
    pub(super) revision: WorkspaceRevision,
    pub(super) slots: Vec<Option<SlotContent>>,
    /// Revision of the last change (admission, retirement) of each slot.
    pub(super) last_change: Vec<u64>,
    pub(super) pinned: Vec<bool>,
    pub(super) pending: BTreeMap<WorkspaceProposalId, Proposal>,
    pub(super) retire_requests: BTreeSet<SlotIndex>,
    pub(super) subscribers: BTreeMap<SubscriberId, Subscriber>,
    pub(super) next_subscriber: u64,
    pub(super) retired_log: VecDeque<RetiredRecord>,
    pub(super) counters: WorkspaceCounters,
}

impl Workspace {
    pub fn new(config: WorkspaceConfig) -> Self {
        let slots = config.slots();
        Self {
            config,
            revision: WorkspaceRevision(0),
            slots: vec![None; slots],
            last_change: vec![0; slots],
            pinned: vec![false; slots],
            pending: BTreeMap::new(),
            retire_requests: BTreeSet::new(),
            subscribers: BTreeMap::new(),
            next_subscriber: 0,
            retired_log: VecDeque::new(),
            counters: WorkspaceCounters::default(),
        }
    }

    pub fn config(&self) -> &WorkspaceConfig {
        &self.config
    }

    /// The revision proposals are computed against; proposals are judged by the next cycle.
    pub fn revision(&self) -> WorkspaceRevision {
        self.revision
    }

    pub fn counters(&self) -> &WorkspaceCounters {
        &self.counters
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn occupied(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// The content currently in `slot`, or `None` when free or out of range.
    pub fn slot(&self, slot: SlotIndex) -> Option<&SlotContent> {
        self.slots.get(slot.get()).and_then(Option::as_ref)
    }

    pub fn is_pinned(&self, slot: SlotIndex) -> bool {
        self.pinned.get(slot.get()).copied().unwrap_or(false)
    }

    /// Recent retirements, oldest first, bounded by [`WorkspaceConfig::retired_log`].
    pub fn retired_log(&self) -> impl Iterator<Item = &RetiredRecord> {
        self.retired_log.iter()
    }

    /// Place `proposal` in the pending area. It is not authoritative and occupies no slot.
    pub fn submit(&mut self, proposal: Proposal) -> Result<WorkspaceProposalId, SubmitError> {
        if proposal.computed_against > self.revision {
            self.counters.submit_rejected += 1;
            return Err(SubmitError::FutureRevision {
                computed_against: proposal.computed_against,
                current: self.revision,
            });
        }
        let id = proposal.id;
        if self.pending.contains_key(&id) {
            self.counters.submit_rejected += 1;
            return Err(SubmitError::Duplicate);
        }
        let by_author = self
            .pending
            .values()
            .filter(|p| p.author == proposal.author)
            .count();
        if by_author >= self.config.max_pending_per_author() {
            self.counters.submit_author_full += 1;
            return Err(SubmitError::AuthorPendingFull);
        }
        if self.pending.len() >= self.config.max_pending() {
            self.counters.submit_pending_full += 1;
            return Err(SubmitError::PendingFull);
        }
        self.pending.insert(id, proposal);
        self.counters.submitted += 1;
        Ok(id)
    }

    fn check_occupied(&self, slot: SlotIndex) -> Result<(), WorkspaceError> {
        if slot.get() >= self.slots.len() {
            return Err(WorkspaceError::SlotOutOfRange(slot));
        }
        if self.slots[slot.get()].is_none() {
            return Err(WorkspaceError::SlotEmpty(slot));
        }
        Ok(())
    }

    /// Ask for the content of `request.slot` to be retired by the next cycle.
    pub fn request_retire(&mut self, request: RetireRequest) -> Result<(), WorkspaceError> {
        self.check_occupied(request.slot)?;
        self.retire_requests.insert(request.slot);
        Ok(())
    }

    /// Protect the content of a slot from displacement.
    pub fn pin(&mut self, request: PinRequest) -> Result<(), WorkspaceError> {
        self.check_occupied(request.slot)?;
        self.pinned[request.slot.get()] = true;
        Ok(())
    }

    pub fn unpin(&mut self, request: PinRequest) -> Result<(), WorkspaceError> {
        self.check_occupied(request.slot)?;
        self.pinned[request.slot.get()] = false;
        Ok(())
    }

    /// Subscribe a Pod generation. Each (pod, generation) subscribes at most once.
    pub fn subscribe(&mut self, request: SubscribeRequest) -> Result<SubscriberId, WorkspaceError> {
        if self.subscribers.len() >= self.config.max_subscribers() {
            return Err(WorkspaceError::SubscribersFull);
        }
        if self
            .subscribers
            .values()
            .any(|s| s.pod == request.pod && s.generation == request.generation)
        {
            return Err(WorkspaceError::DuplicateSubscriber);
        }
        let id = SubscriberId(self.next_subscriber);
        self.next_subscriber += 1;
        self.subscribers.insert(
            id,
            Subscriber {
                pod: request.pod,
                generation: request.generation,
                mailbox: VecDeque::new(),
                delivered: 0,
                overflowed: 0,
            },
        );
        Ok(id)
    }

    pub fn unsubscribe(&mut self, id: SubscriberId) -> Result<(), WorkspaceError> {
        self.subscribers
            .remove(&id)
            .map(|_| ())
            .ok_or(WorkspaceError::UnknownSubscriber(id))
    }

    /// The Pod generation behind a subscription.
    pub fn subscriber_identity(&self, id: SubscriberId) -> Option<(&PodId, Generation)> {
        self.subscribers.get(&id).map(|s| (&s.pod, s.generation))
    }

    /// Take the oldest snapshot waiting in `id`'s mailbox.
    pub fn receive(
        &mut self,
        id: SubscriberId,
    ) -> Result<Option<Arc<WorkspaceSnapshot>>, WorkspaceError> {
        let subscriber = self
            .subscribers
            .get_mut(&id)
            .ok_or(WorkspaceError::UnknownSubscriber(id))?;
        Ok(subscriber.mailbox.pop_front())
    }

    pub fn subscriber_stats(&self, id: SubscriberId) -> Option<SubscriberStats> {
        self.subscribers.get(&id).map(|s| SubscriberStats {
            queued: s.mailbox.len(),
            delivered: s.delivered,
            overflowed: s.overflowed,
        })
    }

    /// Verify internal bounds: slot count, pending, subscriber, mailbox and log limits.
    pub fn check_invariants(&self) -> Result<(), WorkspaceError> {
        if self.slots.len() != self.config.slots()
            || self.last_change.len() != self.slots.len()
            || self.pinned.len() != self.slots.len()
        {
            return Err(WorkspaceError::InvariantViolated("slot count changed"));
        }
        if self.pending.len() > self.config.max_pending() {
            return Err(WorkspaceError::InvariantViolated("pending area over bound"));
        }
        if self.subscribers.len() > self.config.max_subscribers() {
            return Err(WorkspaceError::InvariantViolated("subscribers over bound"));
        }
        if self
            .subscribers
            .values()
            .any(|s| s.mailbox.len() > self.config.mailbox_depth())
        {
            return Err(WorkspaceError::InvariantViolated("mailbox over bound"));
        }
        if self.retired_log.len() > self.config.retired_log() {
            return Err(WorkspaceError::InvariantViolated("retired log over bound"));
        }
        Ok(())
    }
}
