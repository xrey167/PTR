//! Durable Raft state: term, vote, commit, configuration and the log itself.
//!
//! Raft's safety argument rests on state a node remembers across a restart. A
//! node that forgets the term it has seen can join a stale one; a node that
//! forgets its vote can vote twice in the same term and elect two leaders; a node
//! that forgets committed entries can acknowledge a history it no longer has.
//! `MemStorage` holds all three in memory, so a process that dies has agreed to
//! things it can no longer account for.
//!
//! So this is the same state on disk, with two ordering rules that the file format
//! cannot enforce on its own and the caller must respect:
//!
//! 1. Entries and hard state are written **and flushed** before any message that
//!    depends on them leaves the node. [`Core::append`] and
//!    [`Core::set_hard_state`] flush before returning, so the ordering is the
//!    order the calls are made in.
//! 2. A log record is only replaced by truncation, never edited in place. A
//!    conflicting append from a new leader shortens the file and then writes,
//!    so a torn write can lose the tail and never corrupt the prefix.
//!
//! What this is not: it is not consensus, and it is not a cluster. It is the
//! durable state a real group needs underneath it, and a single node using it is
//! still a single node.
use crate::compaction::RetentionFloor;
use crate::integrity::sha256;
use ptr_types::CommitIndex;
use raft::prelude::{ConfState, Entry, EntryType, HardState, Snapshot, SnapshotMetadata};
use raft::util::limit_size;
use raft::{Error as RaftError, GetEntriesContext, RaftState, Storage, StorageError};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// Marks the state file. A foreign file is refused rather than parsed.
const STATE_MAGIC: &[u8; 8] = b"PTRRST03";
/// Marks each log record.
const RECORD_MAGIC: &[u8; 8] = b"PTRRFR01";
/// Marks the log file, once, at offset zero.
const LOG_MAGIC: &[u8; 8] = b"PTRRLG01";
/// Bound on one entry's payload. An entry past this is refused on the way in, so
/// the file can never hold one that cannot be read back.
pub const MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;
/// How many ledger events a member has accounted for.
///
/// A closed state, not a number with a magic value: from the moment a snapshot is installed
/// until the host has accounted for the events it covers (`set_applied_events`) the count is
/// not known. A restart that finds [`AppliedEvents::SnapshotPending`] knows the installation was
/// interrupted or not yet accounted for: the log is reset and the snapshot is offered to the host
/// again, rather than adopted as though it had been vetted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AppliedEvents {
    /// The events applied in total, snapshot included.
    Counted(u64),
    /// A snapshot was installed whose events the host has not accounted for.
    SnapshotPending,
}

/// The snapshot a restart finds installed and not yet accounted for.
pub(crate) struct PendingSnapshot {
    pub(crate) index: u64,
    pub(crate) term: u64,
    pub(crate) payload: Vec<u8>,
}

/// Everything the state file holds.
struct StateImage {
    hard_state: HardState,
    conf_state: ConfState,
    snapshot: SnapshotMetadata,
    snapshot_digest: [u8; 32],
    fence: u64,
    applied: AppliedEvents,
}

/// What reading the log found.
struct LogImage {
    entries: Vec<Entry>,
    offsets: Vec<u64>,
    /// Records the snapshot covers were still in the file.
    skipped: bool,
    /// Bytes cut off the end because they were not a record that could have been acknowledged.
    trimmed: u64,
}

/// Bound on the retained snapshot payload.
pub const MAX_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;
/// Bound on the log file: it is read whole at open, so a file past this is refused before it is
/// read, and an append that would take the file past it is refused before a byte moves.
pub const MAX_LOG_BYTES: u64 = 1024 * 1024 * 1024;
/// Bound on the state file (a few hundred bytes plus the configuration's id lists).
const MAX_STATE_BYTES: u64 = 1024 * 1024;

/// Whether the storage can still be trusted to describe its files.
///
/// An I/O error past the point where a transition could be undone (a rename done and its
/// directory entry not known to be durable, a write that stopped part-way) leaves memory, the
/// file handle and the files disagreeing, and anything acknowledged from there could land in a
/// file that is no longer the log. From then on every write is refused with
/// [`StoragePoisoned`]; the member is restarted, and what it reopens is what the disk holds.
///
/// Reads keep answering from memory when poisoned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Health {
    Sound,
    Poisoned,
}

/// The refusal every write gets once the storage is poisoned (see [`Core::is_poisoned`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoragePoisoned;

impl StoragePoisoned {
    /// Stable diagnostic code, also the start of the error text.
    pub const CODE: &'static str = "PTR_RAFT_STORAGE_POISONED";
}

impl std::fmt::Display for StoragePoisoned {
    /// Name the code and what to do about it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: an I/O error past a point of no return left the storage unsure of its files; \
             restart the member",
            Self::CODE
        )
    }
}

impl std::error::Error for StoragePoisoned {}

/// Why [`Core::record_snapshot_within`] refused to compact the log or record a snapshot.
///
/// Carried inside the [`io::Error`] the storage returns (kind `InvalidData`), so a caller can
/// recover the variant with `error.get_ref()` and `downcast_ref`; the text starts with the
/// stable code. Every refusal is made before a byte is written.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotFloorRefusal {
    /// The entries through `index` produced `events` ledger events and the retained snapshot
    /// covers only `floor` of them, so discarding the log would leave the rest retained
    /// nowhere.
    AboveRetentionFloor {
        index: u64,
        events: CommitIndex,
        floor: CommitIndex,
    },
    /// The payload claims to cover `floor` events but the entries through `index` produced only
    /// `events`: a member restored from it would apply the difference a second time when raft
    /// replays from `index + 1`.
    FloorMismatch {
        index: u64,
        events: CommitIndex,
        floor: CommitIndex,
    },
    /// A snapshot installed at `snapshot_index` has not been accounted for, so there is no base
    /// to count ledger events from.
    Unaccounted { snapshot_index: u64 },
}

impl SnapshotFloorRefusal {
    /// Stable diagnostic code of this refusal, also the start of its text.
    pub const fn code(self) -> &'static str {
        match self {
            Self::AboveRetentionFloor { .. } => "PTR_RAFT_COMPACTION_ABOVE_RETENTION_FLOOR",
            Self::FloorMismatch { .. } => "PTR_RAFT_SNAPSHOT_FLOOR_MISMATCH",
            Self::Unaccounted { .. } => "PTR_RAFT_SNAPSHOT_UNACCOUNTED",
        }
    }
}

impl std::fmt::Display for SnapshotFloorRefusal {
    /// Name the code and the two numbers that disagree.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let code = self.code();
        match *self {
            Self::AboveRetentionFloor {
                index,
                events,
                floor,
            } => write!(
                f,
                "{code}: compacting to raft index {index} discards ledger events through {}, \
                 and the retained snapshot covers only through {}",
                events.0, floor.0
            ),
            Self::FloorMismatch {
                index,
                events,
                floor,
            } => write!(
                f,
                "{code}: raft index {index} produced ledger events through {}, and the payload \
                 claims to cover through {}",
                events.0, floor.0
            ),
            Self::Unaccounted { snapshot_index } => write!(
                f,
                "{code}: the snapshot installed at {snapshot_index} has not been accounted for, \
                 so no ledger position can be derived from a raft index"
            ),
        }
    }
}

impl std::error::Error for SnapshotFloorRefusal {}

impl From<SnapshotFloorRefusal> for io::Error {
    /// Every floor refusal is invalid input, never a fault of the disk.
    fn from(refusal: SnapshotFloorRefusal) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, refusal)
    }
}

/// Whether applying `entry` produces a ledger event.
///
/// One predicate, used where entries are applied ([`RaftNode`](crate::RaftNode)) and where the
/// log is counted for a snapshot, because a retention floor is counted in ledger events and
/// compaction is made in raft indexes: if the two places disagreed about which entries are
/// events, the conversion between them would be wrong by exactly the entries they disagree
/// about. A leader's empty entry and a configuration change are not events.
pub(crate) fn carries_ledger_event(entry: &Entry) -> bool {
    entry.get_entry_type() == EntryType::EntryNormal && !entry.data.is_empty()
}

/// Entry-type wire codes.
///
/// Explicit, never `as` casts on the protobuf enum: a code written into a file
/// outlives the enum's declaration order, and a renumbering would make every
/// record already on disk decode as a different kind of entry than it was
/// written as.
fn entry_type_code(kind: EntryType) -> u8 {
    match kind {
        EntryType::EntryNormal => 0,
        EntryType::EntryConfChange => 1,
        EntryType::EntryConfChangeV2 => 2,
    }
}

/// The entry type a code denotes, or a refusal.
fn entry_type_from_code(code: u8) -> io::Result<EntryType> {
    match code {
        0 => Ok(EntryType::EntryNormal),
        1 => Ok(EntryType::EntryConfChange),
        2 => Ok(EntryType::EntryConfChangeV2),
        other => Err(invalid(format!("unknown raft entry type code {other}"))),
    }
}

/// Construct a stable invalid-data refusal.
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Durable [`Storage`] for a Raft node, backed by two files in one directory.
///
/// Cloning shares the same state, the way `MemStorage` does, because `RawNode`
/// takes the storage by value and the application still needs to write to it.
#[derive(Clone)]
pub struct FileRaftStorage {
    core: Arc<Mutex<Core>>,
}

impl std::fmt::Debug for FileRaftStorage {
    /// Describe the state without printing file handles.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.core.lock() {
            Ok(core) => f
                .debug_struct("FileRaftStorage")
                .field("dir", &core.dir)
                .field("term", &core.hard_state.term)
                .field("vote", &core.hard_state.vote)
                .field("commit", &core.hard_state.commit)
                .field("first_index", &core.first_index())
                .field("last_index", &core.last_index())
                .finish(),
            Err(_) => f.write_str("FileRaftStorage(poisoned)"),
        }
    }
}

impl FileRaftStorage {
    /// Create durable state for a new node, or open what is already there.
    ///
    /// The configuration is only used when there is nothing on disk: an existing
    /// node's recorded configuration is the authority, and replacing it with a
    /// caller's idea of the membership is how a node rejoins a group it was
    /// removed from.
    pub fn open(dir: &Path, voters: &[u64], learners: &[u64]) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let state_path = dir.join("state");
        let log_path = dir.join("log");
        // Scratch files a stop left behind (a state or log being replaced) name nothing the
        // state refers to; they are removed so that they do not accumulate.
        remove_leftover(&atomic_scratch(&state_path));
        remove_leftover(&log_path.with_extension("rewrite"));

        let existing = state_path.is_file();
        let mut log = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&log_path)?;

        let mut core = if existing {
            let image = read_state(&state_path)?;
            let snapshot_data = load_payload(dir, &image.snapshot_digest)?;
            remove_other_payloads(dir, &image.snapshot_digest);
            // What the log holds when it is read may have been written by a process that
            // stopped before it synced: it is synced now, before the member answers for it.
            sync_opened_log(&log)?;
            // While a snapshot waits to be accounted for the member may already hold entries
            // above it (appended and acknowledged), which are kept. A log that still starts at
            // or below the snapshot is the one from before the installation, whose reset was
            // interrupted: all of it goes. The committed prefix is checked in either state.
            let pending = image.applied == AppliedEvents::SnapshotPending;
            let read = read_log(&mut log, image.snapshot.index, image.hard_state.commit)?;
            let (mut entries, mut offsets) = (read.entries, read.offsets);
            if pending && read.skipped {
                let (reset, kept) = rewrite_log(&log_path, &[])?;
                log = reset;
                offsets = kept;
                entries.clear();
            } else if read.skipped {
                // Records the snapshot covers are still in the file under the state that names
                // it: the file is rewritten without them, so that it holds what it is read as.
                let (rewritten, kept) = rewrite_log(&log_path, &entries)?;
                log = rewritten;
                offsets = kept;
            }
            Core {
                dir: dir.to_path_buf(),
                state_path,
                log_path,
                log,
                hard_state: image.hard_state,
                conf_state: image.conf_state,
                snapshot_metadata: image.snapshot,
                snapshot_digest: image.snapshot_digest,
                snapshot_data,
                entries,
                offsets,
                fence: image.fence,
                applied: image.applied,
                trimmed_tail_bytes: read.trimmed,
                health: Health::Sound,
            }
        } else {
            // No state file but a log holding records: the state was deleted or never written
            // for a log somebody else made. Starting fresh would overwrite the header and leave
            // the old records to be mixed with new appends, so it is refused (an operator
            // decision). A log that is empty or holds only the header is what a stop during
            // the first open leaves, and is started over.
            if log.metadata()?.len() > LOG_MAGIC.len() as u64 {
                return Err(invalid(
                    "a log with records exists but its state file is missing; refusing to start \
                     over it",
                ));
            }
            write_log_header(&mut log)?;
            let conf_state = ConfState {
                voters: voters.to_vec(),
                learners: learners.to_vec(),
                ..ConfState::default()
            };
            Core {
                dir: dir.to_path_buf(),
                state_path,
                log_path,
                log,
                hard_state: HardState::default(),
                conf_state,
                snapshot_metadata: SnapshotMetadata::default(),
                snapshot_digest: sha256(&[]),
                snapshot_data: Vec::new(),
                entries: Vec::new(),
                offsets: Vec::new(),
                fence: 0,
                applied: AppliedEvents::Counted(0),
                trimmed_tail_bytes: 0,
                health: Health::Sound,
            }
        };
        if !existing {
            core.persist_state()?;
        }
        Ok(Self {
            core: Arc::new(Mutex::new(core)),
        })
    }

    /// The write side, mirroring `MemStorage::wl`.
    ///
    /// Panics if a previous writer panicked while holding the lock: state that may
    /// have been half-written is not state to keep using.
    pub fn wl(&self) -> MutexGuard<'_, Core> {
        self.core.lock().expect("raft storage lock is not poisoned")
    }
}

/// The state itself. Reachable through [`FileRaftStorage::wl`].
pub struct Core {
    dir: PathBuf,
    state_path: PathBuf,
    log_path: PathBuf,
    log: File,
    hard_state: HardState,
    conf_state: ConfState,
    snapshot_metadata: SnapshotMetadata,
    /// Digest of `snapshot_data`, recorded in the state file: the payload and the position it
    /// is served under are one pair, and a payload whose digest the state does not name is not
    /// served.
    snapshot_digest: [u8; 32],
    snapshot_data: Vec<u8>,
    entries: Vec<Entry>,
    /// File offset of `entries[i]`, so a conflicting append can shorten the file
    /// instead of rewriting it.
    offsets: Vec<u64>,
    /// How many ledger events this member has applied in total, snapshot
    /// included.
    ///
    /// Durable because a snapshot covers events whose records are gone: a member
    /// that reopened counting from zero would hand the next event an index the
    /// snapshot already describes, and two different events would claim it.
    applied: AppliedEvents,
    /// Bytes cut off the end of the log when it was opened (zero for a clean log).
    trimmed_tail_bytes: u64,
    /// The highest term this storage has ever accepted a write under.
    ///
    /// This is the fencing token. It is kept in memory only as a cache of what is
    /// on disk — every write re-reads the file, because the whole point is to
    /// notice a *different process* that has moved it on.
    fence: u64,
    /// See [`Health`].
    health: Health,
}

impl Core {
    /// Index of the first entry the log can still serve.
    pub fn first_index(&self) -> u64 {
        match self.entries.first() {
            Some(entry) => entry.index,
            None => self.snapshot_metadata.index.saturating_add(1),
        }
    }

    /// Index of the last entry in the log.
    pub fn last_index(&self) -> u64 {
        match self.entries.last() {
            Some(entry) => entry.index,
            None => self.snapshot_metadata.index,
        }
    }

    /// The recorded hard state.
    pub fn hard_state(&self) -> &HardState {
        &self.hard_state
    }

    /// The recorded configuration.
    pub fn conf_state(&self) -> &ConfState {
        &self.conf_state
    }

    /// Whether an I/O error past a point of no return has made this storage refuse every
    /// write ([`StoragePoisoned`]). Reads keep answering from memory; a poisoned storage is
    /// restarted, and what it reopens is what the disk holds.
    pub fn is_poisoned(&self) -> bool {
        self.health == Health::Poisoned
    }

    fn poison(&mut self) {
        self.health = Health::Poisoned;
    }

    fn refuse_if_poisoned(&self) -> io::Result<()> {
        match self.health {
            Health::Sound => Ok(()),
            Health::Poisoned => Err(io::Error::other(StoragePoisoned)),
        }
    }

    /// Record term, vote and commit, flushed before returning. On a failure the hard state in
    /// memory is left as it was.
    pub fn set_hard_state(&mut self, hard_state: &HardState) -> io::Result<()> {
        let previous = std::mem::replace(&mut self.hard_state, hard_state.clone());
        self.persist_state()
            .inspect_err(|_| self.hard_state = previous)
    }

    /// Record a new commit index, flushed before returning. On a failure the commit index in
    /// memory is left as it was.
    pub fn set_commit(&mut self, commit: u64) -> io::Result<()> {
        let previous = std::mem::replace(&mut self.hard_state.commit, commit);
        self.persist_state()
            .inspect_err(|_| self.hard_state.commit = previous)
    }

    /// How many ledger events this member has accounted for.
    pub(crate) fn applied(&self) -> AppliedEvents {
        self.applied
    }

    /// Whether a snapshot was installed whose events the host has not yet accounted for.
    pub fn snapshot_pending(&self) -> bool {
        self.applied == AppliedEvents::SnapshotPending
    }

    /// The snapshot waiting to be accounted for, if one is.
    pub(crate) fn pending_snapshot(&self) -> Option<PendingSnapshot> {
        self.snapshot_pending().then(|| PendingSnapshot {
            index: self.snapshot_metadata.index,
            term: self.snapshot_metadata.term,
            payload: self.snapshot_data.clone(),
        })
    }

    /// Bytes cut off the end of the log when it was opened: zero for a clean log. A non-zero
    /// value is what a stop in the middle of an unacknowledged append leaves, and is reported
    /// rather than repaired silently.
    pub fn trimmed_tail_bytes(&self) -> u64 {
        self.trimmed_tail_bytes
    }

    /// Record how many events the member has applied, flushed before returning. On a failure
    /// the count in memory is left as it was.
    pub fn set_applied_events(&mut self, applied_events: u64) -> io::Result<()> {
        let previous = self.applied;
        self.applied = AppliedEvents::Counted(applied_events);
        self.persist_state()
            .inspect_err(|_| self.applied = previous)
    }

    /// Record a new configuration, flushed before returning. On a failure the configuration in
    /// memory is left as it was.
    pub fn set_conf_state(&mut self, conf_state: ConfState) -> io::Result<()> {
        let previous = std::mem::replace(&mut self.conf_state, conf_state);
        self.persist_state()
            .inspect_err(|_| self.conf_state = previous)
    }

    /// Append entries, replacing any that conflict.
    ///
    /// Returns an error rather than panicking where `MemStorage` panics: a gap or
    /// an overwrite of compacted history is a refusal this layer can report, and
    /// a process that aborts inside a write is the situation durability exists to
    /// survive.
    ///
    /// A refusal before the file is touched leaves everything as it was. Once the file has been
    /// shortened or written to, an error leaves it holding something memory does not describe
    /// (a shortened log, a partial record), so the storage is poisoned: no later write is
    /// acknowledged into a file whose offsets nobody can vouch for. Memory is changed only after
    /// the records are durable.
    ///
    /// A multi-entry append whose later records reach the disk while an earlier one is torn is
    /// never acknowledged; a restart finds valid records after the damage and refuses the log
    /// (fail-closed, an operator decision), rather than cutting what may have been acknowledged.
    pub fn append(&mut self, entries: &[Entry]) -> io::Result<()> {
        self.refuse_if_poisoned()?;
        let Some(first) = entries.first() else {
            return Ok(());
        };
        // Fenced before a byte of the log moves. An append is a write to the same
        // shared storage the state file lives on, and a stale writer appending
        // entries is exactly the case a fence exists to stop.
        self.fenced_to()?;
        if first.index < self.first_index() {
            return Err(invalid(format!(
                "append at {} would overwrite compacted history below {}",
                first.index,
                self.first_index()
            )));
        }
        let next = self
            .last_index()
            .checked_add(1)
            .ok_or_else(|| invalid("the raft log index is exhausted"))?;
        if first.index > next {
            return Err(invalid(format!(
                "append at {} leaves a gap after {}",
                first.index,
                self.last_index()
            )));
        }
        let mut total = 0u64;
        for (position, entry) in entries.iter().enumerate() {
            let expected = first
                .index
                .checked_add(position as u64)
                .ok_or_else(|| invalid("the raft log index is exhausted"))?;
            if entry.index != expected {
                return Err(invalid("appended raft entries are not consecutive"));
            }
            if entry.data.len() > MAX_ENTRY_BYTES || entry.context.len() > MAX_ENTRY_BYTES {
                return Err(invalid("raft entry payload is over the bound"));
            }
            total += record_len(entry);
        }
        let keep = (first.index - self.first_index()) as usize;
        let start = match self.offsets.get(keep) {
            Some(offset) => *offset,
            None => self.log.metadata()?.len(),
        };
        // Checked before anything is encoded or written.
        if start.saturating_add(total) > MAX_LOG_BYTES {
            return Err(invalid("the raft log would pass its size bound"));
        }
        let mut buffer = Vec::with_capacity(total as usize);
        let mut lengths = Vec::with_capacity(entries.len());
        for entry in entries {
            let record = encode_record(entry);
            debug_assert_eq!(record.len() as u64, record_len(entry));
            lengths.push(record.len() as u64);
            buffer.extend_from_slice(&record);
        }

        // Test-only: an I/O error here must leave the log untouched and reach the caller,
        // which must then release no message that depends on the entries.
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.append.before_write", |_| Err(injected_io_error()));

        match self.write_entries(keep, start, entries, &lengths, &buffer) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.poison();
                Err(error)
            }
        }
    }

    /// The file part of an append: shorten to `start` if entries conflict, write, sync, and only
    /// then move memory. An error here is past the point of no return (the caller poisons).
    fn write_entries(
        &mut self,
        keep: usize,
        start: u64,
        entries: &[Entry],
        lengths: &[u64],
        buffer: &[u8],
    ) -> io::Result<()> {
        // Truncate first. The file is shortened to the offset the new entries
        // start at, so what is on disk is always a prefix of one history rather
        // than two spliced together.
        if keep < self.entries.len() {
            self.log.set_len(start)?;
        }
        self.log.seek(SeekFrom::Start(start))?;
        // Test-only: a write that puts part of the records down and then fails.
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.append.partial_write", |_| {
            let torn = lengths.first().map_or(0, |length| *length as usize / 2);
            let _ = self.log.write_all(&buffer[..torn]);
            Err(injected_io_error())
        });
        self.log.write_all(buffer)?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.append.after_write_before_sync", |_| Err(
            injected_io_error()
        ));
        self.log.sync_data()?;

        // Durable: memory follows.
        self.entries.truncate(keep);
        self.offsets.truncate(keep);
        let mut position = start;
        for (entry, length) in entries.iter().zip(lengths) {
            self.entries.push(entry.clone());
            self.offsets.push(position);
            position += length;
        }
        Ok(())
    }

    /// Replace the log with a snapshot's position and payload.
    ///
    /// Once the state naming the snapshot is durable, an error while the log is replaced poisons
    /// the storage: the state and the log no longer describe one history, and a restart (which
    /// finishes the installation) is the only way forward.
    pub fn apply_snapshot(&mut self, snapshot: Snapshot) -> io::Result<()> {
        self.refuse_if_poisoned()?;
        let metadata = snapshot.get_metadata().clone();
        if self.first_index() > metadata.index {
            return Err(invalid(format!(
                "snapshot at {} is behind the log's first index {}",
                metadata.index,
                self.first_index()
            )));
        }
        if snapshot.data.len() > MAX_SNAPSHOT_BYTES {
            return Err(invalid("raft snapshot payload is over the bound"));
        }
        // Fenced before anything is written, because this path resets the log after it
        // persists state: a fenced writer must not get as far as that.
        self.fenced_to()?;

        // The payload lands under a name that is its own digest, and the state names that
        // digest: a stop at any point leaves the old state with the old payload or the new
        // state with the new one, never one under the other.
        let digest = sha256(&snapshot.data);
        write_payload(&self.dir, &digest, &snapshot.data)?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.install.after_payload");

        // The state that names the snapshot is written before the log is reset, and marks the
        // events it covers as not yet accounted for. A crash between the two leaves a state
        // that a restart recognises (`AppliedEvents::SnapshotPending`) and finishes, where the
        // other order left an empty log under a commit index far above it, which raft refuses
        // by panicking.
        let previous_term = self.hard_state.term;
        let previous_commit = self.hard_state.commit;
        let previous_conf = self.conf_state.clone();
        let previous_metadata = std::mem::replace(&mut self.snapshot_metadata, metadata.clone());
        let previous_digest = std::mem::replace(&mut self.snapshot_digest, digest);
        let previous_applied = self.applied;
        self.hard_state.term = previous_term.max(metadata.term);
        self.hard_state.commit = metadata.index;
        self.conf_state = metadata.get_conf_state().clone();
        self.applied = AppliedEvents::SnapshotPending;
        if let Err(error) = self.persist_state() {
            // Nothing in memory runs ahead of the disk it could not reach.
            self.hard_state.term = previous_term;
            self.hard_state.commit = previous_commit;
            self.conf_state = previous_conf;
            self.snapshot_metadata = previous_metadata;
            self.snapshot_digest = previous_digest;
            self.applied = previous_applied;
            return Err(error);
        }
        self.snapshot_data = snapshot.data.to_vec();
        remove_other_payloads(&self.dir, &self.snapshot_digest);

        self.reset_log().inspect_err(|_| self.poison())
    }

    /// The log is replaced by an empty one beside it and renamed over the old, so that no stop
    /// leaves a log that is not a log. Memory follows the file only once the rename is durable.
    fn reset_log(&mut self) -> io::Result<()> {
        let scratch = write_scratch_log(&self.log_path, &[])?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.install.after_log_sync_before_rename", |_| Err(
            injected_io_error()
        ));
        let (reset, offsets) = install_scratch_log(&self.log_path, scratch)?;
        self.log = reset;
        self.offsets = offsets;
        self.entries.clear();
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.install.after_log_reset", |_| Err(injected_io_error()));
        Ok(())
    }

    /// The retained snapshot, as raft asks for it.
    pub fn snapshot(&self) -> Snapshot {
        let mut metadata = self.snapshot_metadata.clone();
        metadata.set_conf_state(self.conf_state.clone());
        Snapshot {
            data: self.snapshot_data.clone(),
            metadata: Some(metadata),
        }
    }

    /// Whether the payload held is the one the state names.
    fn payload_matches_state(&self) -> bool {
        sha256(&self.snapshot_data) == self.snapshot_digest
    }

    /// How many ledger events the entries at or below `index` produce, counted from the events
    /// the snapshot position already covers upward and by the predicate the node applies
    /// entries by.
    ///
    /// Refused while an installed snapshot is unaccounted for: the stored count then belongs to
    /// the position before it.
    fn events_through(&self, index: u64) -> io::Result<u64> {
        let AppliedEvents::Counted(base) = self.applied else {
            return Err(SnapshotFloorRefusal::Unaccounted {
                snapshot_index: self.snapshot_metadata.index,
            }
            .into());
        };
        let above = self
            .entries
            .iter()
            .filter(|entry| entry.index <= index)
            .filter(|entry| carries_ledger_event(entry))
            .count() as u64;
        base.checked_add(above)
            .ok_or_else(|| invalid("the count of applied events is exhausted"))
    }

    /// Record a snapshot this node produced itself, trusting that it covers exactly the events
    /// the log through `index` produced.
    ///
    /// This is [`Core::record_snapshot_within`] with the floor taken from the log itself, so the
    /// floor checks pass by construction: it exists for hosts that keep no retained snapshot of
    /// their own to name, and it gives no protection against discarding an event nothing else
    /// retains. A host that has a durable snapshot must call
    /// [`Core::record_snapshot_within`] with that snapshot's floor.
    pub fn record_snapshot(&mut self, index: u64, data: Vec<u8>) -> io::Result<()> {
        let events = self.events_through(index)?;
        self.record_snapshot_within(index, data, RetentionFloor::covering(CommitIndex(events)))
    }

    /// Record a snapshot this node produced itself, discarding the log it covers, bounded by
    /// the ledger's retention floor.
    ///
    /// This is both halves of what `MemStorage` splits between `compact` and a
    /// fabricated snapshot: a durable store cannot invent a snapshot it does not
    /// hold, so the application hands one in.
    ///
    /// `floor` is the ledger position the application's durable snapshot covers, and it must
    /// be **exactly** the number of events the entries at or below `index` produce:
    ///
    /// - Short of it is refused with [`SnapshotFloorRefusal::AboveRetentionFloor`]: the log
    ///   would lose events no retained snapshot describes.
    /// - Past it is refused with [`SnapshotFloorRefusal::FloorMismatch`]: `data` is what a
    ///   member that installs it holds *at* `index`, and raft replays from `index + 1`, so a
    ///   payload covering events the prefix has not produced would apply them twice.
    /// - While an installed snapshot is unaccounted for it is refused with
    ///   [`SnapshotFloorRefusal::Unaccounted`].
    ///
    /// All three are refused before anything is written. The count is taken here, from the log,
    /// because only the log knows which raft indexes carried no event; it is persisted with the
    /// snapshot position.
    ///
    /// Once the state naming the snapshot is durable, an error while the log is compacted
    /// poisons the storage (the state and the log no longer describe one history).
    pub fn record_snapshot_within(
        &mut self,
        index: u64,
        data: Vec<u8>,
        floor: RetentionFloor,
    ) -> io::Result<()> {
        self.refuse_if_poisoned()?;
        if index > self.hard_state.commit {
            return Err(invalid(format!(
                "snapshot at {index} is ahead of the committed index {}",
                self.hard_state.commit
            )));
        }
        if index < self.snapshot_metadata.index {
            return Err(invalid("snapshot moves the covered position backwards"));
        }
        if data.len() > MAX_SNAPSHOT_BYTES {
            return Err(invalid("raft snapshot payload is over the bound"));
        }
        let term = self.term_of(index)?;
        let events = self.events_through(index)?;
        let retained = floor.index();
        match events.cmp(&retained.0) {
            std::cmp::Ordering::Greater => {
                return Err(SnapshotFloorRefusal::AboveRetentionFloor {
                    index,
                    events: CommitIndex(events),
                    floor: retained,
                }
                .into())
            }
            std::cmp::Ordering::Less => {
                return Err(SnapshotFloorRefusal::FloorMismatch {
                    index,
                    events: CommitIndex(events),
                    floor: retained,
                }
                .into())
            }
            std::cmp::Ordering::Equal => {}
        }
        // Fenced before the payload lands and before the log is rewritten, for the
        // same reason `apply_snapshot` fences first: `compact_to` discards the
        // prefix, so a fenced writer reaching it has already destroyed shared
        // history it was supposed to be excluded from touching.
        self.fenced_to()?;
        let snapshot_metadata = SnapshotMetadata {
            index,
            term,
            conf_state: Some(self.conf_state.clone()),
        };

        // `events_through` succeeded, so the count is known: those the previous snapshot
        // covered and every event entry from there up to `index`.
        let AppliedEvents::Counted(already) = self.applied else {
            return Err(SnapshotFloorRefusal::Unaccounted {
                snapshot_index: self.snapshot_metadata.index,
            }
            .into());
        };
        let covered = events;

        let digest = sha256(&data);
        write_payload(&self.dir, &digest, &data)?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.snapshot.after_payload");
        // The state names the payload by digest and position together, and is written before
        // anything in memory moves; the log is compacted after.
        let previous_metadata = std::mem::replace(&mut self.snapshot_metadata, snapshot_metadata);
        let previous_digest = std::mem::replace(&mut self.snapshot_digest, digest);
        self.applied = AppliedEvents::Counted(covered);
        if let Err(error) = self.persist_state() {
            self.snapshot_metadata = previous_metadata;
            self.snapshot_digest = previous_digest;
            self.applied = AppliedEvents::Counted(already);
            return Err(error);
        }
        self.snapshot_data = data;
        remove_other_payloads(&self.dir, &self.snapshot_digest);
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.snapshot.after_state");
        self.compact_to(index).inspect_err(|_| self.poison())?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.snapshot.after_log_rewrite");
        Ok(())
    }

    /// Discard log entries at or below `index`, rewriting the file without them.
    ///
    /// A prefix cannot be removed from a file in place, so the remainder is
    /// written beside the old log and renamed over it. The cost is proportional to
    /// what is kept, which is the opposite of what compaction is for — acceptable
    /// because compaction is rare, and stated rather than hidden.
    ///
    /// Memory follows the file at the rename: the kept entries, their offsets and the handle on
    /// the new file are assigned right after it, so that nothing is ever appended through the
    /// handle of a file that is no longer the log. An error before the rename leaves memory
    /// untouched, yet the caller poisons on any error, conservatively: the state already names
    /// the snapshot, and a restart rewrites the log.
    fn compact_to(&mut self, index: u64) -> io::Result<()> {
        if index < self.first_index() {
            return Ok(());
        }
        if index > self.last_index() {
            return Err(invalid(format!(
                "compaction to {index} is past the last index {}",
                self.last_index()
            )));
        }
        let drop = (index - self.first_index()) as usize + 1;
        let kept: Vec<Entry> = self.entries[drop..].to_vec();

        let scratch = write_scratch_log(&self.log_path, &kept)?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.compact.after_sync_before_rename", |_| Err(
            injected_io_error()
        ));
        fs::rename(self.log_path.with_extension("rewrite"), &self.log_path)?;
        self.log = scratch.file;
        self.offsets = scratch.offsets;
        self.entries = kept;
        sync_after_rename(&self.dir)
    }

    /// Committed data entries, in order, for a node rebuilding its state after a
    /// restart.
    ///
    /// Entries above the recorded commit index are deliberately excluded: they are
    /// replicated, not decided, and treating them as history is how a node
    /// acknowledges something the group never agreed.
    pub fn committed_entries(&self) -> Vec<Entry> {
        self.entries
            .iter()
            .filter(|entry| entry.index <= self.hard_state.commit)
            .cloned()
            .collect()
    }

    /// Term of an index the log or the snapshot still accounts for.
    fn term_of(&self, index: u64) -> io::Result<u64> {
        if index == self.snapshot_metadata.index {
            return Ok(self.snapshot_metadata.term);
        }
        let first = self.first_index();
        if index < first || index > self.last_index() {
            return Err(invalid(format!("index {index} is outside the log")));
        }
        Ok(self.entries[(index - first) as usize].term)
    }

    /// Write the state file atomically.
    /// Refuse this write if another writer has moved the fence past our term.
    ///
    /// Read from **disk**, not from `self.fence`. A stale writer's in-memory copy
    /// is its own stale copy, so consulting it would fence nothing: the case this
    /// exists for is a second process that shares the storage and does not run the
    /// protocol — one partitioned from its peers but not from their disk, or one
    /// resumed from an old image. It is the term on disk that says the group has
    /// moved on.
    ///
    /// The cost is a read before every write. That is what a fence costs, and it
    /// is stated in `31-cluster-integrity.md` rather than hidden.
    fn fenced_to(&mut self) -> io::Result<u64> {
        let on_disk = match read_state(&self.state_path) {
            Ok(image) => image.fence,
            // No state file yet: nothing has claimed this storage, so there is
            // nothing to be fenced by. Any other error is a real failure and is
            // not treated as "unfenced".
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error),
        };
        if on_disk > self.hard_state.term {
            return Err(invalid(format!(
                "PTR_RAFT_FENCED: storage was claimed at term {on_disk}, this writer is at term {}",
                self.hard_state.term
            )));
        }
        self.fence = on_disk.max(self.hard_state.term);
        Ok(self.fence)
    }

    /// Write the state file. The state-file fields (hard state, commit, configuration, snapshot
    /// position and digest, applied count) are changed in memory by the caller *before* this
    /// runs and rolled back by it if this fails, so for them memory does not wait for the I/O
    /// (only the log's entries, offsets and handle do). An error before the rename leaves the
    /// old file and the caller restores memory; an error after it (the directory entry may or may not be durable)
    /// poisons the storage, because neither the old nor the new state can be assumed.
    fn persist_state(&mut self) -> io::Result<()> {
        self.refuse_if_poisoned()?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.state.before_write", |_| Err(injected_io_error()));
        let fence = self.fenced_to()?;
        let bytes = encode_state(&StateImage {
            hard_state: self.hard_state.clone(),
            conf_state: self.conf_state.clone(),
            snapshot: self.snapshot_metadata.clone(),
            snapshot_digest: self.snapshot_digest,
            fence,
            applied: self.applied,
        });
        atomic_write(&self.state_path, &bytes)?;
        sync_state_dir(&self.dir).inspect_err(|_| self.poison())
    }
}

impl Storage for FileRaftStorage {
    /// The state a restarted node starts from.
    fn initial_state(&self) -> raft::Result<RaftState> {
        let core = self.wl();
        Ok(RaftState::new(
            core.hard_state.clone(),
            core.conf_state.clone(),
        ))
    }

    /// Entries in `[low, high)`, bounded by `max_size`.
    fn entries(
        &self,
        low: u64,
        high: u64,
        max_size: impl Into<Option<u64>>,
        _context: GetEntriesContext,
    ) -> raft::Result<Vec<Entry>> {
        let core = self.wl();
        if low < core.first_index() {
            return Err(RaftError::Store(StorageError::Compacted));
        }
        if high > core.last_index().saturating_add(1) {
            panic!(
                "index out of bound (last: {}, high: {})",
                core.last_index(),
                high
            );
        }
        let offset = core.first_index();
        let mut entries = core.entries[(low - offset) as usize..(high - offset) as usize].to_vec();
        limit_size(&mut entries, max_size.into());
        Ok(entries)
    }

    /// Term of one index.
    fn term(&self, index: u64) -> raft::Result<u64> {
        let core = self.wl();
        if index == core.snapshot_metadata.index {
            return Ok(core.snapshot_metadata.term);
        }
        if index < core.first_index() {
            return Err(RaftError::Store(StorageError::Compacted));
        }
        if index > core.last_index() {
            return Err(RaftError::Store(StorageError::Unavailable));
        }
        let offset = core.first_index();
        Ok(core.entries[(index - offset) as usize].term)
    }

    /// First index the log can serve.
    fn first_index(&self) -> raft::Result<u64> {
        Ok(self.wl().first_index())
    }

    /// Last index in the log.
    fn last_index(&self) -> raft::Result<u64> {
        Ok(self.wl().last_index())
    }

    /// The retained snapshot, if it covers what was asked for.
    ///
    /// When it does not, this says "not yet" rather than fabricating one at the
    /// requested index. `MemStorage` fabricates, because it *is* the state
    /// machine; a durable store is not, and a snapshot whose payload does not
    /// match its claimed position would be sent to a follower as truth.
    fn snapshot(&self, request_index: u64, _to: u64) -> raft::Result<Snapshot> {
        let core = self.wl();
        if core.snapshot_metadata.index < request_index || !core.payload_matches_state() {
            return Err(RaftError::Store(
                StorageError::SnapshotTemporarilyUnavailable,
            ));
        }
        Ok(core.snapshot())
    }
}

/// The length of the record `encode_record` writes for `entry`: header, both payloads, digest.
fn record_len(entry: &Entry) -> u64 {
    (RECORD_MAGIC.len() + 8 + 8 + 1 + 4 + 4 + 32 + entry.context.len() + entry.data.len()) as u64
}

/// Encode one log record: magic, position, kind, payloads, digest.
///
/// The digest covers the whole record, so a torn write is detected on read
/// instead of decoding into a plausible entry. Records are not chained: a Raft log
/// legitimately loses its tail to truncation, so a chain would only describe the
/// prefix. A record removed from the middle is caught by the index continuity
/// check on load, which is the property a chain would have been protecting.
fn encode_record(entry: &Entry) -> Vec<u8> {
    let mut record = Vec::with_capacity(48 + entry.data.len() + entry.context.len());
    record.extend_from_slice(RECORD_MAGIC);
    record.extend_from_slice(&entry.index.to_le_bytes());
    record.extend_from_slice(&entry.term.to_le_bytes());
    record.push(entry_type_code(entry.get_entry_type()));
    record.extend_from_slice(&(entry.context.len() as u32).to_le_bytes());
    record.extend_from_slice(&entry.context);
    record.extend_from_slice(&(entry.data.len() as u32).to_le_bytes());
    record.extend_from_slice(&entry.data);
    let digest = sha256(&record);
    record.extend_from_slice(&digest);
    record
}

/// Replace the log file with one that holds exactly `entries`: written beside it, synced and
/// renamed over it. Returns the new handle and the offset of each entry.
fn rewrite_log(log_path: &Path, entries: &[Entry]) -> io::Result<(File, Vec<u64>)> {
    let scratch = write_scratch_log(log_path, entries)?;
    install_scratch_log(log_path, scratch)
}

/// A log written and synced beside the real one, not yet in its place.
struct ScratchLog {
    file: File,
    offsets: Vec<u64>,
}

/// Write the replacement log beside `log_path` and sync it.
fn write_scratch_log(log_path: &Path, entries: &[Entry]) -> io::Result<ScratchLog> {
    let temporary = log_path.with_extension("rewrite");
    let mut rewritten = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temporary)?;
    write_log_header(&mut rewritten)?;
    let mut offsets = Vec::with_capacity(entries.len());
    let mut position = LOG_MAGIC.len() as u64;
    for entry in entries {
        let record = encode_record(entry);
        rewritten.write_all(&record)?;
        offsets.push(position);
        position += record.len() as u64;
    }
    rewritten.sync_all()?;
    Ok(ScratchLog {
        file: rewritten,
        offsets,
    })
}

/// Rename the scratch log over `log_path` and make the rename durable.
fn install_scratch_log(log_path: &Path, scratch: ScratchLog) -> io::Result<(File, Vec<u64>)> {
    fs::rename(log_path.with_extension("rewrite"), log_path)?;
    if let Some(parent) = log_path.parent() {
        sync_dir(parent)?;
    }
    Ok((scratch.file, scratch.offsets))
}

/// Write the log file's header at offset zero.
fn write_log_header(log: &mut File) -> io::Result<()> {
    log.seek(SeekFrom::Start(0))?;
    log.write_all(LOG_MAGIC)?;
    log.sync_data()
}

/// Read the records, checking each digest and that indexes are consecutive.
///
/// Records the snapshot covers (index at or below `snapshot_index`) are skipped: a restart can
/// find them under a state that already names the snapshot.
///
/// The first thing that is not a whole record ends the log, and what follows it is cut off only
/// when it can be shown never to have been acknowledged as a record: it is all zero (space the
/// file system gave and nothing filled), or it holds no record that parses anywhere after the
/// break (an append that stopped part-way). A damaged record with valid data after it is
/// refused, because those records may have been acknowledged and cutting them would forget a
/// promise; a log that ends below the commit index lost committed entries and is refused too.
/// What was cut is reported (`LogImage::trimmed`). A record that is whole and valid but out of
/// sequence is never trimmed.
fn read_log(log: &mut File, snapshot_index: u64, commit: u64) -> io::Result<LogImage> {
    log.seek(SeekFrom::Start(0))?;
    if log.metadata()?.len() > MAX_LOG_BYTES {
        return Err(invalid("raft log is over the size bound"));
    }
    let mut bytes = Vec::new();
    // Bounded again while reading: the file can grow between the check and the read.
    Read::take(&mut *log, MAX_LOG_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_LOG_BYTES {
        return Err(invalid("raft log is over the size bound"));
    }
    if bytes.len() < LOG_MAGIC.len() || &bytes[..LOG_MAGIC.len()] != LOG_MAGIC {
        return Err(invalid("not a PTR raft log"));
    }

    let mut entries: Vec<Entry> = Vec::new();
    let mut offsets: Vec<u64> = Vec::new();
    let mut at = LOG_MAGIC.len();
    let mut skipped = false;
    let mut torn_at: Option<usize> = None;
    while at < bytes.len() {
        let start = at;
        let Some(record) = parse_record(&bytes, at) else {
            torn_at = Some(start);
            break;
        };
        let (entry, end) = record?;
        at = end;
        if entry.index <= snapshot_index && entries.is_empty() {
            skipped = true;
            continue;
        }
        let expected = entries
            .last()
            .map_or(snapshot_index, |last: &Entry| last.index)
            .checked_add(1)
            .ok_or_else(|| invalid("raft log index is exhausted"))?;
        if entry.index != expected {
            return Err(invalid(format!(
                "raft log jumps from {expected} to {}, so a record is missing or duplicated",
                entry.index
            )));
        }
        entries.push(entry);
        offsets.push(start as u64);
    }
    // Whether the log ends in a torn record or exactly at a record boundary (a suffix lost, an
    // older copy restored), it must hold everything that was committed: raft itself would panic
    // on a commit index past its log, and that is not a thing disk input may cause.
    let last = entries.last().map_or(snapshot_index, |entry| entry.index);
    if last < commit {
        return Err(invalid(format!(
            "raft log ends at {last}, below the committed index {commit}: committed entries are missing"
        )));
    }
    let mut trimmed = 0;
    if let Some(cut) = torn_at {
        let rest = &bytes[cut..];
        if !rest.iter().all(|byte| *byte == 0) && record_follows(&bytes, cut) {
            return Err(invalid(format!(
                "raft log is damaged at byte {cut} and valid records follow it: they may have \
                 been acknowledged, so nothing is cut"
            )));
        }
        log.set_len(cut as u64)?;
        log.sync_data()?;
        trimmed = rest.len() as u64;
    }
    Ok(LogImage {
        entries,
        offsets,
        skipped,
        trimmed,
    })
}

/// Whether a whole, valid record starts anywhere after `cut`.
fn record_follows(bytes: &[u8], cut: usize) -> bool {
    (cut + 1..bytes.len())
        .filter(|at| bytes[*at..].starts_with(RECORD_MAGIC))
        .any(|at| matches!(parse_record(bytes, at), Some(Ok(_))))
}

/// Parse the record at `at`. `None` when the bytes there are not a whole record (cut short, or
/// not a record: another magic, a hole, a wrong digest), which only a caller that knows the
/// file's synced length can tell from corruption; `Some(Err(_))` for a whole record that cannot
/// be used.
fn parse_record(bytes: &[u8], at: usize) -> Option<io::Result<(Entry, usize)>> {
    let header_end = at.checked_add(RECORD_MAGIC.len() + 8 + 8 + 1 + 4)?;
    let header = bytes.get(at..header_end)?;
    if &header[..RECORD_MAGIC.len()] != RECORD_MAGIC {
        return None;
    }
    let index = u64::from_le_bytes(header[8..16].try_into().ok()?);
    let term = u64::from_le_bytes(header[16..24].try_into().ok()?);
    let kind_code = header[24];
    let context_len = u32::from_le_bytes(header[25..29].try_into().ok()?) as usize;
    let mut cursor = header_end;
    let context = bytes
        .get(cursor..cursor.checked_add(context_len)?)?
        .to_vec();
    cursor += context_len;
    let data_len =
        u32::from_le_bytes(bytes.get(cursor..cursor.checked_add(4)?)?.try_into().ok()?) as usize;
    cursor += 4;
    let data = bytes.get(cursor..cursor.checked_add(data_len)?)?.to_vec();
    cursor += data_len;
    let recorded = bytes.get(cursor..cursor.checked_add(32)?)?;
    if sha256(&bytes[at..cursor]) != recorded {
        return None;
    }
    cursor += 32;
    let kind = match entry_type_from_code(kind_code) {
        Ok(kind) => kind,
        Err(error) => return Some(Err(error)),
    };
    let mut entry = Entry {
        index,
        term,
        context,
        data,
        ..Entry::default()
    };
    entry.set_entry_type(kind);
    Some(Ok((entry, cursor)))
}

/// Encode the state file: hard state, configuration, snapshot position and payload digest,
/// fence, applied count, digest.
fn encode_state(image: &StateImage) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(STATE_MAGIC);
    bytes.extend_from_slice(&image.hard_state.term.to_le_bytes());
    bytes.extend_from_slice(&image.hard_state.vote.to_le_bytes());
    bytes.extend_from_slice(&image.hard_state.commit.to_le_bytes());
    put_ids(&mut bytes, &image.conf_state.voters);
    put_ids(&mut bytes, &image.conf_state.learners);
    put_ids(&mut bytes, &image.conf_state.voters_outgoing);
    put_ids(&mut bytes, &image.conf_state.learners_next);
    bytes.push(u8::from(image.conf_state.auto_leave));
    bytes.extend_from_slice(&image.snapshot.index.to_le_bytes());
    bytes.extend_from_slice(&image.snapshot.term.to_le_bytes());
    bytes.extend_from_slice(&image.snapshot_digest);
    bytes.extend_from_slice(&image.fence.to_le_bytes());
    match image.applied {
        AppliedEvents::Counted(count) => {
            bytes.push(0);
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        AppliedEvents::SnapshotPending => {
            bytes.push(1);
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
    }
    let digest = sha256(&bytes);
    bytes.extend_from_slice(&digest);
    bytes
}

/// Append a length-prefixed list of node ids.
fn put_ids(bytes: &mut Vec<u8>, ids: &[u64]) {
    bytes.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
}

/// Read the state file, or refuse it.
fn read_state(path: &Path) -> io::Result<StateImage> {
    let bytes = read_bounded(path, MAX_STATE_BYTES, "raft state file")?;
    if bytes.len() < STATE_MAGIC.len() + 32 || &bytes[..STATE_MAGIC.len()] != STATE_MAGIC {
        return Err(invalid("not a PTR raft state file"));
    }
    let body = &bytes[..bytes.len() - 32];
    if sha256(body) != bytes[bytes.len() - 32..] {
        return Err(invalid("raft state file does not match its digest"));
    }
    let mut at = STATE_MAGIC.len();
    let take_u64 = |at: &mut usize| -> io::Result<u64> {
        let slice = body
            .get(*at..*at + 8)
            .ok_or_else(|| invalid("raft state file ends inside a field"))?;
        *at += 8;
        Ok(u64::from_le_bytes(slice.try_into().expect("u64 bytes")))
    };
    let hard_state = HardState {
        term: take_u64(&mut at)?,
        vote: take_u64(&mut at)?,
        commit: take_u64(&mut at)?,
    };

    let voters = take_ids(body, &mut at)?;
    let learners = take_ids(body, &mut at)?;
    let voters_outgoing = take_ids(body, &mut at)?;
    let learners_next = take_ids(body, &mut at)?;
    let auto_leave = match body.get(at) {
        Some(0) => false,
        Some(1) => true,
        Some(other) => return Err(invalid(format!("invalid auto_leave byte {other}"))),
        None => return Err(invalid("raft state file ends before auto_leave")),
    };
    at += 1;
    let conf_state = ConfState {
        voters,
        learners,
        voters_outgoing,
        learners_next,
        auto_leave,
    };

    let snapshot = SnapshotMetadata {
        index: take_u64(&mut at)?,
        term: take_u64(&mut at)?,
        conf_state: None,
    };
    let snapshot_digest: [u8; 32] = body
        .get(at..at + 32)
        .ok_or_else(|| invalid("raft state file ends inside the snapshot digest"))?
        .try_into()
        .expect("digest bytes");
    at += 32;
    let fence = take_u64(&mut at)?;
    let applied_tag = *body
        .get(at)
        .ok_or_else(|| invalid("raft state file ends before the applied count"))?;
    at += 1;
    let applied_count = take_u64(&mut at)?;
    let applied = match (applied_tag, applied_count) {
        (0, count) => AppliedEvents::Counted(count),
        (1, 0) => AppliedEvents::SnapshotPending,
        _ => return Err(invalid("raft state file has an invalid applied count")),
    };
    if at != body.len() {
        return Err(invalid("raft state file has trailing bytes"));
    }
    Ok(StateImage {
        hard_state,
        conf_state,
        snapshot,
        snapshot_digest,
        fence,
        applied,
    })
}

/// Read a file whose size is checked against `max` before it is read, and again while reading.
fn read_bounded(path: &Path, max: u64, what: &str) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    if file.metadata()?.len() > max {
        return Err(invalid(format!("{what} is over the size bound")));
    }
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(invalid(format!("{what} is over the size bound")));
    }
    Ok(bytes)
}

/// Read a length-prefixed list of node ids.
fn take_ids(body: &[u8], at: &mut usize) -> io::Result<Vec<u64>> {
    let count_bytes = body
        .get(*at..*at + 4)
        .ok_or_else(|| invalid("raft state file ends inside an id count"))?;
    let count = u32::from_le_bytes(count_bytes.try_into().expect("count bytes")) as usize;
    *at += 4;
    let mut ids = Vec::with_capacity(count.min(64));
    for _ in 0..count {
        let slice = body
            .get(*at..*at + 8)
            .ok_or_else(|| invalid("raft state file ends inside an id"))?;
        ids.push(u64::from_le_bytes(slice.try_into().expect("id bytes")));
        *at += 8;
    }
    Ok(ids)
}

/// Write a file by writing a new one and renaming it over the old.
///
/// A half-written state file is worse than an old one: it describes a term or a
/// vote that was never actually held.
fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let temporary = atomic_scratch(path);
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("raft.atomic.after_write_before_sync");
        file.sync_all()?;
    }
    #[cfg(feature = "failpoints")]
    fail::fail_point!("raft.atomic.after_sync_before_rename");
    fs::rename(&temporary, path)
}

/// The file `atomic_write` writes beside `path` before renaming it over.
fn atomic_scratch(path: &Path) -> PathBuf {
    path.with_extension("tmp")
}

/// Make the state file's rename durable. The failpoint can stop the process here or return an
/// error in place of the directory sync.
fn sync_state_dir(dir: &Path) -> io::Result<()> {
    #[cfg(feature = "failpoints")]
    fail::fail_point!("raft.state.after_rename_before_dir_sync", |_| Err(
        injected_io_error()
    ));
    sync_dir(dir)
}

/// Make a log rename durable. The failpoint can stop the process here or return an error in
/// place of the directory sync.
fn sync_after_rename(dir: &Path) -> io::Result<()> {
    #[cfg(feature = "failpoints")]
    fail::fail_point!("raft.compact.after_rename_before_dir_sync", |_| Err(
        injected_io_error()
    ));
    sync_dir(dir)
}

/// Remove a scratch file a stop left behind. Best effort.
fn remove_leftover(path: &Path) {
    let _ = fs::remove_file(path);
}

/// The error a test-only failpoint returns in place of an I/O operation.
#[cfg(feature = "failpoints")]
fn injected_io_error() -> io::Error {
    io::Error::other("PTR_RAFT_INJECTED_IO_ERROR")
}

/// Flush a log read at open: a record the previous process wrote and did not sync is made
/// durable before this process acknowledges anything that depends on it.
fn sync_opened_log(log: &File) -> io::Result<()> {
    #[cfg(feature = "failpoints")]
    fail::fail_point!("raft.open.log_sync", |_| Err(injected_io_error()));
    log.sync_data()
}

/// File name under which the snapshot payload with this digest is kept.
fn payload_name(digest: &[u8; 32]) -> String {
    let mut name = String::from("snapshot-");
    for byte in digest {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}

/// Write a payload under the name of its own digest and make the name durable.
fn write_payload(dir: &Path, digest: &[u8; 32], data: &[u8]) -> io::Result<()> {
    atomic_write(&dir.join(payload_name(digest)), data)?;
    sync_dir(dir)
}

/// The payload the state names, or a refusal: absent or altered bytes are not a snapshot.
fn load_payload(dir: &Path, digest: &[u8; 32]) -> io::Result<Vec<u8>> {
    let path = dir.join(payload_name(digest));
    let data = match read_bounded(&path, MAX_SNAPSHOT_BYTES as u64, "retained raft snapshot") {
        Ok(data) => data,
        Err(error) if error.kind() == io::ErrorKind::NotFound && *digest == sha256(&[]) => {
            return Ok(Vec::new())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(invalid("the snapshot payload the state names is missing"))
        }
        Err(error) => return Err(error),
    };
    if data.len() > MAX_SNAPSHOT_BYTES {
        return Err(invalid("retained raft snapshot is over the bound"));
    }
    if sha256(&data) != *digest {
        return Err(invalid(
            "retained raft snapshot does not match the digest the state names",
        ));
    }
    Ok(data)
}

/// Remove payload files (and their scratch files) the state does not name. Best effort: a file
/// that stays is named by nothing and is removed on the next open.
fn remove_other_payloads(dir: &Path, keep: &[u8; 32]) {
    let kept = payload_name(keep);
    let Ok(listing) = fs::read_dir(dir) else {
        return;
    };
    for entry in listing.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if is_payload_file(name) && name != kept {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// Whether `name` is a payload file this module wrote: `snapshot-` and 64 hex digits, with the
/// `.tmp` suffix of a payload being written. Anything else in the directory is not ours to remove.
fn is_payload_file(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("snapshot-") else {
        return false;
    };
    let digest = rest.strip_suffix(".tmp").unwrap_or(rest);
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Flush a directory entry, so a rename survives a crash.
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}
