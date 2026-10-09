//! Validated, immutable workspace parameters.

use super::Bid;
use ptr_types::VerificationLevel;
use std::fmt;
use std::num::NonZeroU64;

/// Largest accepted slot count.
pub const MAX_WORKSPACE_SLOTS: usize = 64;
/// Largest accepted pending-proposal bound.
pub const MAX_PENDING_PROPOSALS: usize = 1024;
/// Largest accepted subscriber count.
pub const MAX_WORKSPACE_SUBSCRIBERS: usize = 64;
/// Largest accepted per-subscriber mailbox depth.
pub const MAX_SUBSCRIBER_MAILBOX: usize = 64;
/// Largest accepted retired-content log.
pub const MAX_RETIRED_LOG: usize = 4096;

/// Proof that the caller read the risk of [`SlotDecay::NeverUnsafe`]: content with no residency
/// limit outlives the verifier, contract and generation it was admitted under unless
/// [`Revalidation`] re-checks it. [`WorkspaceConfig::new`] refuses it without revalidation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnboundedResidencyRisk(());

impl UnboundedResidencyRisk {
    /// Acknowledge that slot content may stay forever unless revalidation or an explicit retire
    /// removes it. Not for long-running use.
    pub fn acknowledge() -> Self {
        Self(())
    }
}

/// When slot content leaves on its own.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotDecay {
    /// Content is retired in the cycle that produces a revision this many revisions after the one
    /// that admitted it.
    AfterCycles(NonZeroU64),
    /// No residency limit. Unsafe for long runs: verifier-governed content would be permanent.
    /// Requires an acknowledgement and a [`Revalidation`] other than `Never`.
    NeverUnsafe(UnboundedResidencyRisk),
}

/// Whether, and how often, the host verifier is re-run on incumbents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Revalidation {
    /// Incumbents are verified only when admitted.
    Never,
    /// Every incumbent is re-verified every cycle.
    EveryCycle,
    /// An incumbent is re-verified when its residency is a multiple of this many cycles.
    EveryCycles(NonZeroU64),
}

/// Whether finding messages (free text from verifiers, possibly echoing private evidence) are
/// part of the broadcast snapshot. Invariant 13: the default is `HostOnly`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FindingVisibility {
    /// Snapshots carry finding codes and hard flags only; messages go to the host in
    /// the cycle report's `host_findings`.
    #[default]
    HostOnly,
    /// Bounded messages are part of the snapshot. They are untrusted data to every reader.
    Public,
}

/// Opt-in displacement of incumbents; see the module documentation of the cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DisplacementPolicy {
    /// A challenger's effective bid must exceed the incumbent's by at least this much.
    pub margin: u32,
    /// Cycles an incumbent is protected after admission (at least 1).
    pub min_residency: NonZeroU64,
    /// Displacements allowed in one cycle (1..=slots).
    pub max_per_cycle: usize,
    /// Displacements one author may cause in one cycle (1..=`max_per_cycle`).
    pub max_per_author: usize,
}

/// Displacement mode. `Off` (the default of [`WorkspaceConfigInit::conservative`]) keeps the
/// workspace a bounded cache; `Policy` makes it a competition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Displacement {
    Off,
    Policy(DisplacementPolicy),
}

/// Unvalidated workspace parameters; see [`WorkspaceConfig::new`].
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceConfigInit {
    pub slots: usize,
    pub max_pending: usize,
    /// Pending proposals one author pod may hold.
    pub max_pending_per_author: usize,
    /// Slots one author pod may occupy.
    pub max_slots_per_author: usize,
    pub max_subscribers: usize,
    pub mailbox_depth: usize,
    pub retired_log: usize,
    pub required_level: VerificationLevel,
    pub decay: SlotDecay,
    pub revalidation: Revalidation,
    pub displacement: Displacement,
    pub finding_visibility: FindingVisibility,
    /// Upper bound of every effective bid, whatever the proposal says.
    pub max_bid: Bid,
    /// Revisions a proposal may lag, provided no slot changed since (0 = strictly current).
    pub max_staleness: u64,
    pub tie_break_seed: u64,
}

impl WorkspaceConfigInit {
    /// Conservative defaults: strict staleness, 64-cycle residency, revalidation every cycle, no
    /// displacement, host-only finding messages, one slot per author is not forced.
    pub fn conservative(slots: usize) -> Self {
        Self {
            slots,
            max_pending: 64,
            max_pending_per_author: 16,
            max_slots_per_author: slots.max(1),
            max_subscribers: 8,
            mailbox_depth: 4,
            retired_log: 64,
            required_level: VerificationLevel::SampleVerified,
            decay: SlotDecay::AfterCycles(NonZeroU64::new(64).expect("non-zero")),
            revalidation: Revalidation::EveryCycle,
            displacement: Displacement::Off,
            finding_visibility: FindingVisibility::HostOnly,
            max_bid: Bid::new(u32::MAX),
            max_staleness: 0,
            tie_break_seed: 0,
        }
    }
}

/// Validated, immutable workspace parameters.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceConfig {
    pub(super) init: WorkspaceConfigInit,
}

/// A rejected [`WorkspaceConfigInit`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkspaceConfigError {
    /// `slots` must be in `1..=MAX_WORKSPACE_SLOTS`.
    Slots(usize),
    /// `max_pending` must be in `1..=MAX_PENDING_PROPOSALS`.
    MaxPending(usize),
    /// `max_pending_per_author` must be in `1..=max_pending`.
    MaxPendingPerAuthor(usize),
    /// `max_slots_per_author` must be in `1..=slots`.
    MaxSlotsPerAuthor(usize),
    /// `max_subscribers` must be in `1..=MAX_WORKSPACE_SUBSCRIBERS`.
    MaxSubscribers(usize),
    /// `mailbox_depth` must be in `1..=MAX_SUBSCRIBER_MAILBOX`.
    MailboxDepth(usize),
    /// `retired_log` must be in `1..=MAX_RETIRED_LOG`.
    RetiredLog(usize),
    /// `SlotDecay::NeverUnsafe` needs a revalidation other than `Never`.
    UnboundedResidencyWithoutRevalidation,
    /// A displacement field is out of range.
    Displacement(&'static str),
}

impl fmt::Display for WorkspaceConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Slots(n) => write!(f, "slots {n} outside 1..={MAX_WORKSPACE_SLOTS}"),
            Self::MaxPending(n) => write!(f, "max_pending {n} outside 1..={MAX_PENDING_PROPOSALS}"),
            Self::MaxPendingPerAuthor(n) => {
                write!(f, "max_pending_per_author {n} outside 1..=max_pending")
            }
            Self::MaxSlotsPerAuthor(n) => write!(f, "max_slots_per_author {n} outside 1..=slots"),
            Self::MaxSubscribers(n) => {
                write!(
                    f,
                    "max_subscribers {n} outside 1..={MAX_WORKSPACE_SUBSCRIBERS}"
                )
            }
            Self::MailboxDepth(n) => {
                write!(f, "mailbox_depth {n} outside 1..={MAX_SUBSCRIBER_MAILBOX}")
            }
            Self::RetiredLog(n) => write!(f, "retired_log {n} outside 1..={MAX_RETIRED_LOG}"),
            Self::UnboundedResidencyWithoutRevalidation => {
                f.write_str("unbounded residency requires revalidation")
            }
            Self::Displacement(what) => write!(f, "displacement policy invalid: {what}"),
        }
    }
}

impl std::error::Error for WorkspaceConfigError {}

fn in_bound(value: usize, max: usize) -> bool {
    (1..=max).contains(&value)
}

impl WorkspaceConfig {
    /// Validate `init`: every capacity is positive and bounded (invariant 8).
    pub fn new(init: WorkspaceConfigInit) -> Result<Self, WorkspaceConfigError> {
        use WorkspaceConfigError as E;
        if !in_bound(init.slots, MAX_WORKSPACE_SLOTS) {
            return Err(E::Slots(init.slots));
        }
        if !in_bound(init.max_pending, MAX_PENDING_PROPOSALS) {
            return Err(E::MaxPending(init.max_pending));
        }
        if !in_bound(init.max_pending_per_author, init.max_pending) {
            return Err(E::MaxPendingPerAuthor(init.max_pending_per_author));
        }
        if !in_bound(init.max_slots_per_author, init.slots) {
            return Err(E::MaxSlotsPerAuthor(init.max_slots_per_author));
        }
        if !in_bound(init.max_subscribers, MAX_WORKSPACE_SUBSCRIBERS) {
            return Err(E::MaxSubscribers(init.max_subscribers));
        }
        if !in_bound(init.mailbox_depth, MAX_SUBSCRIBER_MAILBOX) {
            return Err(E::MailboxDepth(init.mailbox_depth));
        }
        if !in_bound(init.retired_log, MAX_RETIRED_LOG) {
            return Err(E::RetiredLog(init.retired_log));
        }
        if matches!(init.decay, SlotDecay::NeverUnsafe(_))
            && init.revalidation == Revalidation::Never
        {
            return Err(E::UnboundedResidencyWithoutRevalidation);
        }
        if let Displacement::Policy(p) = init.displacement {
            if p.margin == 0 {
                return Err(E::Displacement("margin must be at least 1"));
            }
            if !in_bound(p.max_per_cycle, init.slots) {
                return Err(E::Displacement("max_per_cycle outside 1..=slots"));
            }
            if !in_bound(p.max_per_author, p.max_per_cycle) {
                return Err(E::Displacement("max_per_author outside 1..=max_per_cycle"));
            }
        }
        Ok(Self { init })
    }

    pub fn slots(&self) -> usize {
        self.init.slots
    }

    pub fn max_pending(&self) -> usize {
        self.init.max_pending
    }

    pub fn max_pending_per_author(&self) -> usize {
        self.init.max_pending_per_author
    }

    pub fn max_slots_per_author(&self) -> usize {
        self.init.max_slots_per_author
    }

    pub fn max_subscribers(&self) -> usize {
        self.init.max_subscribers
    }

    pub fn mailbox_depth(&self) -> usize {
        self.init.mailbox_depth
    }

    pub fn retired_log(&self) -> usize {
        self.init.retired_log
    }

    pub fn required_level(&self) -> VerificationLevel {
        self.init.required_level
    }

    pub fn decay(&self) -> SlotDecay {
        self.init.decay
    }

    pub fn revalidation(&self) -> Revalidation {
        self.init.revalidation
    }

    pub fn displacement(&self) -> Displacement {
        self.init.displacement
    }

    pub fn finding_visibility(&self) -> FindingVisibility {
        self.init.finding_visibility
    }

    pub fn max_bid(&self) -> Bid {
        self.init.max_bid
    }

    pub fn max_staleness(&self) -> u64 {
        self.init.max_staleness
    }

    pub fn tie_break_seed(&self) -> u64 {
        self.init.tie_break_seed
    }
}
