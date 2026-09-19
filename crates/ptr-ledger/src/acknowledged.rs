//! The acknowledgment protocol that binds a durable log to protected anchor
//! storage, and the recovery rules for every way the two can disagree.
//!
//! A record becomes durable when [`FileLedger::append_durable`] returns, and it
//! becomes *acknowledged* when the anchor covering it is installed. Those are two
//! synchronized writes to two files, so a crash can land between them and the
//! pair can be found in any of a small number of states. Each one has exactly one
//! correct reading, and this module names them rather than guessing:
//!
//! | State | Meaning | Recovery |
//! |---|---|---|
//! | [`Split::Aligned`] | the anchor covers the log's tail | open |
//! | [`Split::Unacknowledged`] | durable records, or a partial frame, sit above the anchor | [`TailPolicy`] |
//! | [`Split::LostSuffix`] | the anchor covers more than the log holds | refuse |
//! | [`Split::Diverged`] | same length, different history | refuse |
//! | [`Split::OriginMismatch`] | the anchor belongs to another log | refuse |
//! | [`Split::BaseMismatch`] | the compaction floors disagree | refuse |
//!
//! The asymmetry between the last four and the second is deliberate. A log that
//! is *ahead* of its anchor has lost nothing, so recovery can proceed. A log that
//! is *behind* its anchor has lost committed history, and no local evidence can
//! reconstruct it — presenting such a log as healthy is precisely the rollback
//! that global invariant 3 forbids.
use crate::anchor::{AnchorError, AnchorKey, AnchorStore, ChainOrigin, ProtectedAnchor};
use crate::compaction::{
    CompactionDecision, CompactionFault, CompactionOutcome, CompactionPlan, LogPaths,
    RetentionPolicy,
};
use crate::integrity::{self, LogAnchor};
use crate::{CommittedEvent, CompactionBarrier, FileLedger, LedgerEvent};
use ptr_types::CommitIndex;
use std::fmt;
use std::io;
use std::path::PathBuf;

/// How a log relates to its protected anchor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Split {
    /// The anchor covers exactly the log's complete tail and no partial frame
    /// follows it.
    Aligned,
    /// Durable, chain-valid records and/or one partial frame follow the anchor.
    /// The records were committed by a previous run whose anchor advance did not
    /// complete; the partial frame was never a record at all.
    Unacknowledged {
        complete_records: usize,
        incomplete_frame: bool,
        tail: LogAnchor,
    },
    /// The anchor acknowledges a longer history than the log contains.
    LostSuffix {
        acknowledged: CommitIndex,
        present: CommitIndex,
    },
    /// The anchor's covered index exists in the log at a different digest, so the
    /// log is a different history of the same length or shorter.
    Diverged,
    /// The log's first record disagrees with the anchor's recorded origin.
    OriginMismatch,
    /// The anchor's compaction floor is not the floor this log was read against.
    BaseMismatch,
}

impl Split {
    /// True when opening may proceed, possibly after tail repair.
    pub fn is_recoverable(self) -> bool {
        matches!(self, Self::Aligned | Self::Unacknowledged { .. })
    }
}

/// What to do with durable records that sit above the anchor.
///
/// An incomplete trailing frame is removed under every policy; it is a partial
/// write, not a record. This choice only governs *complete* records.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TailPolicy {
    /// Acknowledge them. Nothing durable is discarded, which is the safe
    /// direction: an appended-but-unacknowledged revocation stays in force.
    #[default]
    Acknowledge,
    /// Remove them. This destroys durably committed records and is correct only
    /// for a host that can prove nothing ever observed them.
    Discard,
    /// Refuse to open and leave every byte in place, for an operator who wants to
    /// decide out of band.
    Refuse,
}

/// What a repair actually did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TailRecovery {
    /// Durable records above the previous anchor that were kept.
    pub acknowledged_records: usize,
    /// Durable records above the previous anchor that were removed.
    pub discarded_records: usize,
    /// Whether a partial trailing frame was truncated.
    pub incomplete_frame_removed: bool,
    /// The log's tail after repair.
    pub tail: LogAnchor,
    /// The origin derivable from the repaired log, when it still holds index 1.
    pub origin: Option<ChainOrigin>,
}

/// Failures of the acknowledged-ledger layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AcknowledgedError {
    /// Protected anchor storage refused.
    Anchor(AnchorError),
    /// The log itself could not be read or framed. Carries the integrity or I/O
    /// diagnostic verbatim.
    Log(String),
    /// A split that must not be repaired locally.
    Unrecoverable(Split),
    /// [`TailPolicy::Refuse`] was selected and a tail was present.
    TailRefused {
        complete_records: usize,
        incomplete_frame: bool,
    },
    /// An append left the log and anchor in an unknown relationship. Reopen and
    /// classify before writing again.
    Fenced,
    /// A cutover refused before touching anything.
    Compaction(CompactionFault),
}

impl AcknowledgedError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Anchor(error) => error.code(),
            Self::Log(_) => "PTR_ACK_LOG",
            Self::Unrecoverable(Split::LostSuffix { .. }) => "PTR_ACK_LOST_SUFFIX",
            Self::Unrecoverable(Split::Diverged) => "PTR_ACK_DIVERGED",
            Self::Unrecoverable(Split::OriginMismatch) => "PTR_ACK_ORIGIN_MISMATCH",
            Self::Unrecoverable(Split::BaseMismatch) => "PTR_ACK_BASE_MISMATCH",
            // Aligned and Unacknowledged are recoverable and never wrapped here.
            Self::Unrecoverable(_) => "PTR_ACK_UNRECOVERABLE",
            Self::TailRefused { .. } => "PTR_ACK_TAIL_REFUSED",
            Self::Fenced => "PTR_ACK_FENCED",
            Self::Compaction(fault) => fault.code(),
        }
    }
}

impl fmt::Display for AcknowledgedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Log(diagnostic) => write!(f, "{}: {diagnostic}", self.code()),
            _ => f.write_str(self.code()),
        }
    }
}

impl std::error::Error for AcknowledgedError {}

impl From<AnchorError> for AcknowledgedError {
    fn from(error: AnchorError) -> Self {
        Self::Anchor(error)
    }
}

impl From<io::Error> for AcknowledgedError {
    fn from(error: io::Error) -> Self {
        Self::Log(error.to_string())
    }
}

/// A durable log paired with the protected anchor that says how far it is
/// acknowledged.
///
/// Appends are ordered log-first: a record is synchronized before the anchor that
/// covers it advances. The reverse order would let a crash produce
/// [`Split::LostSuffix`], an unrecoverable state, in exchange for nothing.
#[derive(Debug)]
pub struct AcknowledgedLedger {
    log: FileLedger,
    anchors: AnchorStore,
    paths: LogPaths,
    fenced: bool,
}

impl AcknowledgedLedger {
    /// Create a new log and its first anchor. Neither file may already exist, so
    /// this can never adopt a history it did not create.
    pub fn create(paths: &LogPaths, key: AnchorKey) -> Result<Self, AcknowledgedError> {
        if paths.anchor_path().exists() {
            return Err(AnchorError::Exists.into());
        }
        let log = FileLedger::create_new(paths.log_path(CommitIndex(0)))?;
        let anchors =
            AnchorStore::initialize(paths.anchor_path(), key, ProtectedAnchor::initial())?;
        Ok(Self {
            log,
            anchors,
            paths: paths.clone(),
            fenced: false,
        })
    }

    /// Classify an existing set without modifying anything.
    ///
    /// `minimum_epoch` is the rollback witness described on
    /// [`AnchorStore::open_expecting`]; pass 0 only when anchor-file rollback is
    /// genuinely out of scope.
    pub fn inspect(
        paths: &LogPaths,
        key: AnchorKey,
        minimum_epoch: u64,
    ) -> Result<(Split, ProtectedAnchor), AcknowledgedError> {
        let anchors = AnchorStore::open_expecting(paths.anchor_path(), key, minimum_epoch)?;
        let anchor = anchors.current();
        let locked = FileLedger::lock_for_recovery(paths.log_path(anchor.base.index), anchor.base)?;
        Ok((locked.classify(anchor), anchor))
    }

    /// Open an existing set, repairing only an unacknowledged tail.
    ///
    /// The anchor names the live log, so a cutover interrupted on either side of
    /// its commit point resolves here without a guess. Returns the split that was
    /// found, so a caller can record that recovery happened rather than infer it.
    pub fn open(
        paths: &LogPaths,
        key: AnchorKey,
        minimum_epoch: u64,
        policy: TailPolicy,
    ) -> Result<(Self, Split), AcknowledgedError> {
        let mut anchors = AnchorStore::open_expecting(paths.anchor_path(), key, minimum_epoch)?;
        let anchor = anchors.current();
        let locked = FileLedger::lock_for_recovery(paths.log_path(anchor.base.index), anchor.base)?;
        let split = locked.classify(anchor);
        match split {
            Split::Aligned => {}
            Split::Unacknowledged {
                complete_records,
                incomplete_frame,
                ..
            } => {
                if policy == TailPolicy::Refuse {
                    return Err(AcknowledgedError::TailRefused {
                        complete_records,
                        incomplete_frame,
                    });
                }
            }
            fault => return Err(AcknowledgedError::Unrecoverable(fault)),
        }
        let (log, recovery) = locked.repair(anchor, policy)?;
        if recovery.tail != anchor.log {
            anchors.acknowledge(recovery.tail, recovery.origin.unwrap_or(ChainOrigin::UNSET))?;
        }
        Ok((
            Self {
                log,
                anchors,
                paths: paths.clone(),
                fenced: false,
            },
            split,
        ))
    }

    /// Append a record and then acknowledge it.
    ///
    /// On success the record is both durable and acknowledged. If the anchor
    /// advance fails, the record remains durable and this ledger fences itself:
    /// the pair is in [`Split::Unacknowledged`], which a reopen resolves, and
    /// continuing to append would stack more unacknowledged records on an anchor
    /// that is already behind.
    pub fn append_acknowledged(
        &mut self,
        event: LedgerEvent,
    ) -> Result<CommitIndex, AcknowledgedError> {
        if self.fenced {
            return Err(AcknowledgedError::Fenced);
        }
        let index = self.log.append_durable(event)?;
        let tail = self.log.anchor()?;
        let origin = self.log.derived_origin();
        self.fenced = true;
        self.anchors
            .acknowledge(tail, origin.unwrap_or(ChainOrigin::UNSET))?;
        self.fenced = false;
        Ok(index)
    }

    pub fn events(&self) -> &[CommittedEvent] {
        self.log.events()
    }

    pub fn anchor(&self) -> ProtectedAnchor {
        self.anchors.current()
    }

    /// The epoch to retain as the next [`AnchorStore::open_expecting`] witness.
    pub fn epoch(&self) -> u64 {
        self.anchors.epoch()
    }

    pub fn log(&self) -> &FileLedger {
        &self.log
    }

    pub fn paths(&self) -> &LogPaths {
        &self.paths
    }

    /// Ask a retention policy whether any prefix may be discarded.
    ///
    /// The proposal is capped by [`CompactionBarrier::snapshot_covers`], because
    /// discarding records that no snapshot covers destroys the only description of
    /// the state they produced. An unsafe barrier blocks outright: a consumer that
    /// has not caught up still needs the records, and an unresolved revocation
    /// must not have its evidence removed while it is still in force.
    pub fn plan_compaction(
        &self,
        policy: RetentionPolicy,
        barrier: CompactionBarrier,
    ) -> Result<CompactionDecision, AcknowledgedError> {
        let anchor = self.anchors.current();
        let floor = anchor.base.index.0;
        let tail = anchor.log;
        if tail.index.0.saturating_sub(floor) <= policy.keep_records {
            return Ok(CompactionDecision::Retain);
        }
        if !barrier.safe() {
            return Ok(CompactionDecision::Blocked(barrier));
        }
        let proposed = (tail.index.0 - policy.keep_records).min(barrier.snapshot_covers.0);
        if proposed <= floor {
            return Ok(CompactionDecision::Blocked(barrier));
        }
        let anchors = integrity::chain_anchors(self.log.events(), anchor.base)?;
        let offset = (proposed - floor) as usize;
        let base = *anchors
            .get(offset - 1)
            .ok_or(CompactionFault::FloorNotOnBoundary)?;
        Ok(CompactionDecision::Compact(CompactionPlan {
            base,
            tail,
            discarded_records: proposed - floor,
            retained_records: tail.index.0 - proposed,
        }))
    }

    /// Execute a cutover: build the compacted log, then advance the anchor.
    ///
    /// The anchor advance is the commit point, and it is the only step that
    /// changes which file is live. Everything before it is validated and
    /// create-new, so a failure leaves the previous log authoritative and the
    /// half-built artifact an orphan. Nothing is overwritten and nothing is
    /// deleted; see [`Self::reclaim_orphans`].
    pub fn compact(
        &mut self,
        plan: CompactionPlan,
    ) -> Result<CompactionOutcome, AcknowledgedError> {
        if self.fenced {
            return Err(AcknowledgedError::Fenced);
        }
        let anchor = self.anchors.current();
        if plan.tail != anchor.log {
            return Err(CompactionFault::StalePlan.into());
        }
        if plan.base.index.0 <= anchor.base.index.0 {
            return Err(CompactionFault::FloorNotAdvancing.into());
        }
        if plan.base.index.0 > anchor.log.index.0 {
            return Err(CompactionFault::FloorAboveTail.into());
        }
        let offset = (plan.base.index.0 - anchor.base.index.0) as usize;
        let anchors = integrity::chain_anchors(self.log.events(), anchor.base)?;
        if anchors.get(offset - 1) != Some(&plan.base) {
            return Err(CompactionFault::FloorNotOnBoundary.into());
        }
        let retained = self.log.events()[offset..].to_vec();
        let bytes = integrity::encode_log_from(&retained, plan.base)?;
        let live_log = self.paths.log_path(plan.base.index);
        let superseded_log = self.paths.log_path(anchor.base.index);
        if live_log.exists() {
            return Err(CompactionFault::DestinationExists.into());
        }
        let compacted = FileLedger::create_from_log_above(&live_log, &bytes, plan.base, plan.tail)?;
        self.anchors.advance_compacted(plan.base, plan.tail)?;
        // Past the commit point the anchor names the new file, so installing it
        // here cannot disagree with what a reopen would resolve.
        self.log = compacted;
        Ok(CompactionOutcome {
            plan,
            live_log,
            superseded_log,
        })
    }

    /// Delete log files of this set that the anchor does not name.
    ///
    /// Only call this on a ledger opened with a witnessed epoch. Under a
    /// rolled-back anchor the live file is the one that looks like an orphan, and
    /// this would delete exactly the history that was being protected.
    pub fn reclaim_orphans(&self) -> Result<Vec<PathBuf>, AcknowledgedError> {
        let live = self.anchors.current().base.index;
        let orphans = self.paths.orphans(live)?;
        for path in &orphans {
            std::fs::remove_file(path)?;
        }
        Ok(orphans)
    }
}
