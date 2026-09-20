//! Retention, compaction planning and the cutover that raises a log's floor.
//!
//! Compaction discards a prefix of committed history, so the question that
//! decides the whole design is what a crash mid-cutover leaves behind. Renaming a
//! newly built log over the live one cannot answer it: afterwards the file either
//! starts at index 1 or above the floor, and with a single fixed path both states
//! produce the same chain error and neither can be told from corruption.
//!
//! So the floor goes in the file name and the protected anchor is the only thing
//! that says which file is live. A cutover writes a new file, then advances the
//! anchor. A crash before that advance leaves the old file live and the new one an
//! orphan; a crash after it leaves the new file live and the old one an orphan.
//! Both are unambiguous, nothing is overwritten, and reclaiming an orphan is an
//! explicit separate step rather than automatic deletion next to a crash.
use crate::acknowledged::AcknowledgedError;
use crate::integrity::LogAnchor;
use crate::CompactionBarrier;
use ptr_types::CommitIndex;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const LOG_SUFFIX: &str = ".log";
const ANCHOR_SUFFIX: &str = ".anchor";

/// File naming for an anchored log whose compaction floor is part of its name.
///
/// The anchor sits at a fixed path because there is exactly one; logs are named
/// per floor because several can exist at once during a cutover, and only the
/// anchor decides which of them is authoritative.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogPaths {
    directory: PathBuf,
    stem: String,
}

impl LogPaths {
    /// Name one independently anchored log set within a directory.
    pub fn new(directory: impl Into<PathBuf>, stem: impl Into<String>) -> Self {
        Self {
            directory: directory.into(),
            stem: stem.into(),
        }
    }

    /// Directory containing this set's anchor and log generations.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Path of this set's protected anchor.
    pub fn anchor_path(&self) -> PathBuf {
        self.directory.join(format!("{}{ANCHOR_SUFFIX}", self.stem))
    }

    /// Path of the log whose floor is `base`. Zero padding keeps directory
    /// listings in commit order, which matters when an operator is reading them
    /// during recovery.
    pub fn log_path(&self, base: CommitIndex) -> PathBuf {
        self.directory
            .join(format!("{}-{:020}{LOG_SUFFIX}", self.stem, base.0))
    }

    /// Log files of this set other than the live one.
    ///
    /// A crash during a cutover leaves exactly one of these, on either side of
    /// the anchor advance. They are never deleted implicitly.
    ///
    /// A file belongs to this set only if it is a name this set could have
    /// produced: [`LogPaths::log_path`] of the floor its name encodes must be
    /// exactly this path. Matching the prefix alone is not enough, because the
    /// prefix of one stem can be the prefix of another — `journal-` also starts
    /// `journal-backup-00000000000000000000.log` — and
    /// [`AcknowledgedLedger::reclaim_orphans`](crate::AcknowledgedLedger::reclaim_orphans)
    /// deletes what this returns. Reporting a neighbouring set's *live* log as
    /// this set's orphan would destroy committed history that nothing here is
    /// even responsible for.
    pub fn orphans(&self, live_base: CommitIndex) -> io::Result<Vec<PathBuf>> {
        let live = self.log_path(live_base);
        let prefix = format!("{}-", self.stem);
        let mut found = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(floor) = name
                .strip_prefix(&prefix)
                .and_then(|rest| rest.strip_suffix(LOG_SUFFIX))
            else {
                continue;
            };
            // Round-tripping the parsed floor rejects everything we could not
            // have written: another stem's suffix, a non-numeric field, a value
            // past `u64`, and a non-canonical spelling of a number.
            let Ok(base) = floor.parse::<u64>() else {
                continue;
            };
            let canonical = self.log_path(CommitIndex(base));
            if canonical == path && path != live {
                found.push(path);
            }
        }
        found.sort();
        Ok(found)
    }
}

/// How much committed history to keep above the floor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionPolicy {
    /// Records to retain above the new floor. A floor is only proposed when more
    /// than this many records sit above the current one.
    pub keep_records: u64,
}

impl RetentionPolicy {
    /// Retain at least `keep_records` commits above a proposed floor.
    pub fn keeping(keep_records: u64) -> Self {
        Self { keep_records }
    }
}

/// A validated floor advance, produced only by
/// [`AcknowledgedLedger::plan_compaction`](crate::AcknowledgedLedger::plan_compaction).
///
/// Holding one is not permission to discard anything on its own: the records it
/// removes must already be reconstructible from a snapshot covering the floor,
/// which is why [`CompactionBarrier::snapshot_covers`] caps it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionPlan {
    /// The new floor. The compacted log begins at the next index and chains from
    /// this digest.
    pub base: LogAnchor,
    /// The tail, which compaction never moves.
    pub tail: LogAnchor,
    pub discarded_records: u64,
    pub retained_records: u64,
}

/// The outcome of consulting a retention policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactionDecision {
    /// Not enough history above the current floor to act on.
    Retain,
    /// A consumer is behind, a revocation is unresolved, or no snapshot covers
    /// any discardable prefix. Carries the barrier that refused.
    Blocked(CompactionBarrier),
    Compact(CompactionPlan),
}

/// What a completed cutover produced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactionOutcome {
    pub plan: CompactionPlan,
    /// The now-live compacted log.
    pub live_log: PathBuf,
    /// The previous log, retained until explicitly reclaimed.
    pub superseded_log: PathBuf,
}

/// Reasons a cutover refuses before touching anything.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactionFault {
    /// The plan's floor is at or below the current one.
    FloorNotAdvancing,
    /// The plan's floor is above the acknowledged tail.
    FloorAboveTail,
    /// The plan's floor is not a record boundary of this log, or its digest does
    /// not match the one this log produces at that index.
    FloorNotOnBoundary,
    /// The plan's tail is not this log's acknowledged tail, so the plan was made
    /// against a different state.
    StalePlan,
    /// A log file for the proposed floor already exists. Never overwritten.
    DestinationExists,
}

impl CompactionFault {
    /// Stable diagnostic code for this refusal.
    pub fn code(self) -> &'static str {
        match self {
            Self::FloorNotAdvancing => "PTR_COMPACT_FLOOR_NOT_ADVANCING",
            Self::FloorAboveTail => "PTR_COMPACT_FLOOR_ABOVE_TAIL",
            Self::FloorNotOnBoundary => "PTR_COMPACT_FLOOR_NOT_ON_BOUNDARY",
            Self::StalePlan => "PTR_COMPACT_STALE_PLAN",
            Self::DestinationExists => "PTR_COMPACT_DESTINATION_EXISTS",
        }
    }
}

impl From<CompactionFault> for AcknowledgedError {
    /// Preserve a compaction refusal at the acknowledged-ledger boundary.
    fn from(fault: CompactionFault) -> Self {
        Self::Compaction(fault)
    }
}
