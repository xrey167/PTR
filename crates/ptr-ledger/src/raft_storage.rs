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
use crate::integrity::sha256;
use raft::prelude::{ConfState, Entry, EntryType, HardState, Snapshot, SnapshotMetadata};
use raft::util::limit_size;
use raft::{Error as RaftError, GetEntriesContext, RaftState, Storage, StorageError};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// Marks the state file. A foreign file is refused rather than parsed.
const STATE_MAGIC: &[u8; 8] = b"PTRRST02";
/// Marks each log record.
const RECORD_MAGIC: &[u8; 8] = b"PTRRFR01";
/// Marks the log file, once, at offset zero.
const LOG_MAGIC: &[u8; 8] = b"PTRRLG01";
/// Bound on one entry's payload. An entry past this is refused on the way in, so
/// the file can never hold one that cannot be read back.
pub const MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;
/// Bound on the retained snapshot payload.
pub const MAX_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;

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
        let snapshot_path = dir.join("snapshot");

        let existing = state_path.is_file();
        let mut log = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&log_path)?;

        let mut core = if existing {
            let (hard_state, conf_state, snapshot_metadata, fence, applied_events) =
                read_state(&state_path)?;
            let snapshot_data = if snapshot_path.is_file() {
                let data = fs::read(&snapshot_path)?;
                if data.len() > MAX_SNAPSHOT_BYTES {
                    return Err(invalid("retained raft snapshot is over the bound"));
                }
                data
            } else {
                Vec::new()
            };
            let (entries, offsets) = read_log(&mut log, snapshot_metadata.index)?;
            Core {
                dir: dir.to_path_buf(),
                state_path,
                log_path,
                snapshot_path,
                log,
                hard_state,
                conf_state,
                snapshot_metadata,
                snapshot_data,
                entries,
                offsets,
                fence,
                applied_events,
            }
        } else {
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
                snapshot_path,
                log,
                hard_state: HardState::default(),
                conf_state,
                snapshot_metadata: SnapshotMetadata::default(),
                snapshot_data: Vec::new(),
                entries: Vec::new(),
                offsets: Vec::new(),
                fence: 0,
                applied_events: 0,
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
    snapshot_path: PathBuf,
    log: File,
    hard_state: HardState,
    conf_state: ConfState,
    snapshot_metadata: SnapshotMetadata,
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
    applied_events: u64,
    /// The highest term this storage has ever accepted a write under.
    ///
    /// This is the fencing token. It is kept in memory only as a cache of what is
    /// on disk — every write re-reads the file, because the whole point is to
    /// notice a *different process* that has moved it on.
    fence: u64,
}

impl Core {
    /// Index of the first entry the log can still serve.
    pub fn first_index(&self) -> u64 {
        match self.entries.first() {
            Some(entry) => entry.index,
            None => self.snapshot_metadata.index + 1,
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

    /// Record term, vote and commit, flushed before returning.
    pub fn set_hard_state(&mut self, hard_state: &HardState) -> io::Result<()> {
        self.hard_state = hard_state.clone();
        self.persist_state()
    }

    /// Record a new commit index, flushed before returning.
    pub fn set_commit(&mut self, commit: u64) -> io::Result<()> {
        self.hard_state.commit = commit;
        self.persist_state()
    }

    /// How many ledger events this member has applied in total.
    pub fn applied_events(&self) -> u64 {
        self.applied_events
    }

    /// Record how many events the member has applied, flushed before returning.
    pub fn set_applied_events(&mut self, applied_events: u64) -> io::Result<()> {
        self.applied_events = applied_events;
        self.persist_state()
    }

    /// Record a new configuration, flushed before returning.
    pub fn set_conf_state(&mut self, conf_state: ConfState) -> io::Result<()> {
        self.conf_state = conf_state;
        self.persist_state()
    }

    /// Append entries, replacing any that conflict.
    ///
    /// Returns an error rather than panicking where `MemStorage` panics: a gap or
    /// an overwrite of compacted history is a refusal this layer can report, and
    /// a process that aborts inside a write is the situation durability exists to
    /// survive.
    pub fn append(&mut self, entries: &[Entry]) -> io::Result<()> {
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
        if first.index > self.last_index() + 1 {
            return Err(invalid(format!(
                "append at {} leaves a gap after {}",
                first.index,
                self.last_index()
            )));
        }
        for (position, entry) in entries.iter().enumerate() {
            if entry.index != first.index + position as u64 {
                return Err(invalid("appended raft entries are not consecutive"));
            }
            if entry.data.len() > MAX_ENTRY_BYTES || entry.context.len() > MAX_ENTRY_BYTES {
                return Err(invalid("raft entry payload is over the bound"));
            }
        }

        // Truncate first. The file is shortened to the offset the new entries
        // start at, so what is on disk is always a prefix of one history rather
        // than two spliced together.
        let keep = (first.index - self.first_index()) as usize;
        if keep < self.entries.len() {
            let offset = self.offsets[keep];
            self.log.set_len(offset)?;
            self.log.seek(SeekFrom::Start(offset))?;
            self.entries.truncate(keep);
            self.offsets.truncate(keep);
        } else {
            let end = self.log.seek(SeekFrom::End(0))?;
            debug_assert_eq!(
                end,
                self.offsets.last().map_or(LOG_MAGIC.len() as u64, |_| end)
            );
        }

        let mut position = self.log.seek(SeekFrom::End(0))?;
        for entry in entries {
            let record = encode_record(entry);
            self.log.write_all(&record)?;
            self.entries.push(entry.clone());
            self.offsets.push(position);
            position += record.len() as u64;
        }
        self.log.sync_data()
    }

    /// Replace the log with a snapshot's position and payload.
    pub fn apply_snapshot(&mut self, snapshot: Snapshot) -> io::Result<()> {
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
        // Fenced before anything is written, because this path truncates the log
        // before it persists state: a fenced writer must not get as far as that.
        self.fenced_to()?;

        // The payload lands before the state that points at it: a crash between
        // the two leaves a snapshot file nothing refers to, which is recoverable,
        // where the other order leaves a state naming a payload that is not there.
        atomic_write(&self.snapshot_path, &snapshot.data)?;
        self.snapshot_data = snapshot.data.to_vec();

        self.entries.clear();
        self.offsets.clear();
        self.log.set_len(0)?;
        self.log.seek(SeekFrom::Start(0))?;
        write_log_header(&mut self.log)?;

        self.hard_state.term = self.hard_state.term.max(metadata.term);
        self.hard_state.commit = metadata.index;
        self.conf_state = metadata.get_conf_state().clone();
        self.snapshot_metadata = metadata;
        self.persist_state()
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

    /// Record a snapshot this node produced itself, discarding the log it covers.
    ///
    /// This is both halves of what `MemStorage` splits between `compact` and a
    /// fabricated snapshot: a durable store cannot invent a snapshot it does not
    /// hold, so the application hands one in.
    pub fn record_snapshot(&mut self, index: u64, data: Vec<u8>) -> io::Result<()> {
        if index > self.hard_state.commit {
            return Err(invalid(format!(
                "snapshot at {index} is ahead of the committed index {}",
                self.hard_state.commit
            )));
        }
        if index < self.snapshot_metadata.index {
            return Err(invalid("snapshot moves the covered position backwards"));
        }
        let term = self.term_of(index)?;
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

        atomic_write(&self.snapshot_path, &data)?;
        self.snapshot_data = data;
        self.snapshot_metadata = snapshot_metadata;
        self.compact_to(index)?;
        self.persist_state()
    }

    /// Discard log entries at or below `index`, rewriting the file without them.
    ///
    /// A prefix cannot be removed from a file in place, so the remainder is
    /// written beside the old log and renamed over it. The cost is proportional to
    /// what is kept, which is the opposite of what compaction is for — acceptable
    /// because compaction is rare, and stated rather than hidden.
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
        let drop = (index + 1 - self.first_index()) as usize;
        let kept: Vec<Entry> = self.entries.split_off(drop);
        self.entries = kept;
        self.offsets.clear();

        let temporary = self.log_path.with_extension("rewrite");
        let mut rewritten = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary)?;
        write_log_header(&mut rewritten)?;
        let mut position = LOG_MAGIC.len() as u64;
        for entry in &self.entries {
            let record = encode_record(entry);
            rewritten.write_all(&record)?;
            self.offsets.push(position);
            position += record.len() as u64;
        }
        rewritten.sync_all()?;
        fs::rename(&temporary, &self.log_path)?;
        sync_dir(&self.dir)?;
        self.log = rewritten;
        Ok(())
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
            Ok((_, _, _, fence, _)) => fence,
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

    fn persist_state(&mut self) -> io::Result<()> {
        let fence = self.fenced_to()?;
        let bytes = encode_state(
            &self.hard_state,
            &self.conf_state,
            &self.snapshot_metadata,
            fence,
            self.applied_events,
        );
        atomic_write(&self.state_path, &bytes)?;
        sync_dir(&self.dir)
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
        if high > core.last_index() + 1 {
            panic!(
                "index out of bound (last: {}, high: {})",
                core.last_index() + 1,
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
        if core.snapshot_metadata.index < request_index {
            return Err(RaftError::Store(
                StorageError::SnapshotTemporarilyUnavailable,
            ));
        }
        Ok(core.snapshot())
    }
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

/// Write the log file's header at offset zero.
fn write_log_header(log: &mut File) -> io::Result<()> {
    log.seek(SeekFrom::Start(0))?;
    log.write_all(LOG_MAGIC)?;
    log.sync_data()
}

/// Read every record, checking each digest and that indexes are consecutive.
fn read_log(log: &mut File, snapshot_index: u64) -> io::Result<(Vec<Entry>, Vec<u64>)> {
    log.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    log.read_to_end(&mut bytes)?;
    if bytes.len() < LOG_MAGIC.len() || &bytes[..LOG_MAGIC.len()] != LOG_MAGIC {
        return Err(invalid("not a PTR raft log"));
    }

    let mut entries: Vec<Entry> = Vec::new();
    let mut offsets: Vec<u64> = Vec::new();
    let mut at = LOG_MAGIC.len();
    while at < bytes.len() {
        let start = at;
        let header_end = at
            .checked_add(RECORD_MAGIC.len() + 8 + 8 + 1 + 4)
            .ok_or_else(|| invalid("raft log record header overflows"))?;
        let header = bytes
            .get(at..header_end)
            .ok_or_else(|| invalid("raft log ends inside a record header"))?;
        if &header[..RECORD_MAGIC.len()] != RECORD_MAGIC {
            return Err(invalid("raft log record has the wrong magic"));
        }
        let index = u64::from_le_bytes(header[8..16].try_into().expect("index bytes"));
        let term = u64::from_le_bytes(header[16..24].try_into().expect("term bytes"));
        let kind = entry_type_from_code(header[24])?;
        let context_len =
            u32::from_le_bytes(header[25..29].try_into().expect("context length")) as usize;
        at = header_end;
        let context = bytes
            .get(at..at + context_len)
            .ok_or_else(|| invalid("raft log ends inside an entry context"))?
            .to_vec();
        at += context_len;
        let data_len_bytes = bytes
            .get(at..at + 4)
            .ok_or_else(|| invalid("raft log ends inside a data length"))?;
        let data_len = u32::from_le_bytes(data_len_bytes.try_into().expect("data length")) as usize;
        at += 4;
        let data = bytes
            .get(at..at + data_len)
            .ok_or_else(|| invalid("raft log ends inside an entry payload"))?
            .to_vec();
        at += data_len;
        let recorded = bytes
            .get(at..at + 32)
            .ok_or_else(|| invalid("raft log ends inside a record digest"))?;
        if sha256(&bytes[start..at]) != recorded {
            return Err(invalid("raft log record does not match its digest"));
        }
        at += 32;

        let expected = entries
            .last()
            .map_or(snapshot_index + 1, |last: &Entry| last.index + 1);
        if index != expected {
            return Err(invalid(format!(
                "raft log jumps from {expected} to {index}, so a record is missing or duplicated"
            )));
        }

        let mut entry = Entry {
            index,
            term,
            context,
            data,
            ..Entry::default()
        };
        entry.set_entry_type(kind);
        entries.push(entry);
        offsets.push(start as u64);
    }
    Ok((entries, offsets))
}

/// Encode the state file: hard state, configuration, snapshot position, digest.
fn encode_state(
    hard_state: &HardState,
    conf_state: &ConfState,
    snapshot: &SnapshotMetadata,
    fence: u64,
    applied_events: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(96);
    bytes.extend_from_slice(STATE_MAGIC);
    bytes.extend_from_slice(&hard_state.term.to_le_bytes());
    bytes.extend_from_slice(&hard_state.vote.to_le_bytes());
    bytes.extend_from_slice(&hard_state.commit.to_le_bytes());
    put_ids(&mut bytes, &conf_state.voters);
    put_ids(&mut bytes, &conf_state.learners);
    put_ids(&mut bytes, &conf_state.voters_outgoing);
    put_ids(&mut bytes, &conf_state.learners_next);
    bytes.push(u8::from(conf_state.auto_leave));
    bytes.extend_from_slice(&snapshot.index.to_le_bytes());
    bytes.extend_from_slice(&snapshot.term.to_le_bytes());
    bytes.extend_from_slice(&fence.to_le_bytes());
    bytes.extend_from_slice(&applied_events.to_le_bytes());
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
fn read_state(path: &Path) -> io::Result<(HardState, ConfState, SnapshotMetadata, u64, u64)> {
    let bytes = fs::read(path)?;
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
    let fence = take_u64(&mut at)?;
    let applied_events = take_u64(&mut at)?;
    if at != body.len() {
        return Err(invalid("raft state file has trailing bytes"));
    }
    Ok((hard_state, conf_state, snapshot, fence, applied_events))
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
    let temporary = path.with_extension("tmp");
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)
}

/// Flush a directory entry, so a rename survives a crash.
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}
