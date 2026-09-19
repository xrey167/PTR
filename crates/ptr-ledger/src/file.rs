//! Strict file lifecycle and explicit, externally anchored tail recovery.
use crate::acknowledged::{AcknowledgedError, Split, TailPolicy, TailRecovery};
use crate::anchor::{ChainOrigin, ProtectedAnchor};
use crate::integrity::{
    self, LogAnchor, VerifiedLog, FRAME_HEADER_BYTES, LOG_MAGIC, MAX_LOG_BYTES,
};
use crate::{CommittedEvent, LedgerEvent};
use ptr_types::CommitIndex;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

/// Single-writer reference ledger. New files use PTRLOG02 framing. Existing
/// legacy/incomplete/corrupt files are never silently migrated or truncated.
/// `open_at` checks an independently retained complete-history anchor.
pub struct FileLedger {
    path: PathBuf,
    file: LockedFile,
    events: Vec<CommittedEvent>,
    anchor: LogAnchor,
    /// Compaction floor this handle was opened against. Records at or below it
    /// are not in this file, so its own bytes cannot establish where it starts.
    base: LogAnchor,
    length: usize,
    poisoned: bool,
}

impl std::fmt::Debug for FileLedger {
    /// The locked handle is deliberately omitted; a file descriptor is not a
    /// useful diagnostic and printing it invites treating it as an identity.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileLedger")
            .field("path", &self.path)
            .field("records", &self.events.len())
            .field("anchor_index", &self.anchor.index)
            .field("base_index", &self.base.index)
            .field("length", &self.length)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

/// An explicitly opened legacy source held exclusively throughout migration.
/// It grants no authenticity guarantee for the old checksum-free contents.
pub struct LegacyLog {
    _source: LockedFile,
    events: Vec<CommittedEvent>,
}
impl LegacyLog {
    pub fn events(&self) -> &[CommittedEvent] {
        &self.events
    }
}

// Explicit unlock releases logical ownership even when a concurrent process
// spawn temporarily inherited the open-file description before exec. Never
// expose or clone this guard; close alone waits for all duplicate descriptors.
struct LockedFile(File);
impl LockedFile {
    fn new(file: File) -> io::Result<Self> {
        fs4::FileExt::try_lock(&file).map_err(io::Error::from)?;
        Ok(Self(file))
    }
}
impl Deref for LockedFile {
    type Target = File;
    fn deref(&self) -> &File {
        &self.0
    }
}
impl DerefMut for LockedFile {
    fn deref_mut(&mut self) -> &mut File {
        &mut self.0
    }
}
impl Drop for LockedFile {
    fn drop(&mut self) {
        let _ = fs4::FileExt::unlock(&self.0);
    }
}
fn lock_existing(path: &Path) -> io::Result<LockedFile> {
    LockedFile::new(OpenOptions::new().read(true).write(true).open(path)?)
}
fn read_bounded(file: &mut File) -> io::Result<Vec<u8>> {
    if file.metadata()?.len() > MAX_LOG_BYTES as u64 {
        return Err(integrity::invalid("PTR_LOG_SIZE_LIMIT"));
    }
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.take(MAX_LOG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_LOG_BYTES {
        return Err(integrity::invalid("PTR_LOG_SIZE_LIMIT"));
    }
    Ok(bytes)
}

/// Unix directory synchronization closes the new-name persistence window.
/// Non-Unix platforms still sync the file; no portable directory-sync guarantee
/// or hardware power-loss claim is made there.
pub(crate) fn sync_parent(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

impl FileLedger {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        match OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)
        {
            Ok(file) => {
                let mut file = LockedFile::new(file)?;
                file.write_all(LOG_MAGIC)?;
                file.sync_all()?;
                sync_parent(path)?;
                Self::from_verified(
                    path,
                    file,
                    integrity::decode_log(LOG_MAGIC)?,
                    LOG_MAGIC.len(),
                )
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let mut file = lock_existing(path)?;
                let bytes = read_bounded(&mut file)?;
                let verified = integrity::decode_log(&bytes)?;
                Self::from_verified(path, file, verified, bytes.len())
            }
            Err(error) => Err(error),
        }
    }

    /// Create a log that must not already exist.
    ///
    /// Unlike [`Self::open`] this never adopts an existing file, so a caller that
    /// is establishing a new anchored history cannot accidentally inherit a
    /// history it has no anchor for.
    pub fn create_new(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)?;
        let mut file = LockedFile::new(file)?;
        file.write_all(LOG_MAGIC)?;
        file.sync_all()?;
        sync_parent(path)?;
        Self::from_verified(
            path,
            file,
            integrity::decode_log(LOG_MAGIC)?,
            LOG_MAGIC.len(),
        )
    }

    /// Acquire the single-writer lock and validate every complete frame above
    /// `base`, without writing anything.
    ///
    /// The returned handle keeps the lock, so the classification it produces
    /// cannot go stale before the repair that acts on it.
    pub fn lock_for_recovery(
        path: impl AsRef<Path>,
        base: LogAnchor,
    ) -> io::Result<RecoverableLog> {
        let path = path.as_ref();
        let mut file = lock_existing(path)?;
        let bytes = read_bounded(&mut file)?;
        let prefix = integrity::scan_from(&bytes, base)?;
        Ok(RecoverableLog {
            path: path.to_owned(),
            file,
            base,
            prefix,
            length: bytes.len(),
        })
    }

    /// Require exact history identity, not merely a valid hash chain. Missing
    /// files fail rather than creating an empty replacement.
    pub fn open_at(path: impl AsRef<Path>, trusted: LogAnchor) -> io::Result<Self> {
        let path = path.as_ref();
        let mut file = lock_existing(path)?;
        let bytes = read_bounded(&mut file)?;
        let verified = integrity::decode_log(&bytes)?;
        verified.require_anchor(trusted)?;
        Self::from_verified(path, file, verified, bytes.len())
    }

    /// Explicit repair only after matching the complete prefix to an anchor
    /// retained independently by the host. No complete extra record is removed.
    /// Damaged complete headers/records and wrong anchors leave all bytes intact.
    pub fn recover_unacknowledged_tail(
        path: impl AsRef<Path>,
        trusted: LogAnchor,
    ) -> io::Result<Self> {
        let path = path.as_ref();
        let mut file = lock_existing(path)?;
        let bytes = read_bounded(&mut file)?;
        let prefix = integrity::scan(&bytes)?;
        prefix.verified.require_anchor(trusted)?;
        if prefix.incomplete {
            file.set_len(prefix.complete_bytes as u64)?;
            file.sync_all()?;
        }
        Self::from_verified(path, file, prefix.verified, prefix.complete_bytes)
    }

    /// Create a new destination only after complete byte/anchor validation.
    /// Never overwrite a live log. An I/O failure may leave an incomplete new
    /// destination; strict open rejects it and the source remains untouched.
    pub fn create_from_log(
        path: impl AsRef<Path>,
        bytes: &[u8],
        trusted: LogAnchor,
    ) -> io::Result<Self> {
        let verified = integrity::decode_log(bytes)?;
        verified.require_anchor(trusted)?;
        let path = path.as_ref();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path)?;
        let mut file = LockedFile::new(file)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        sync_parent(path)?;
        Self::from_verified(path, file, verified, bytes.len())
    }

    /// Explicit inspection of an old unchecked format. Runtime migration must
    /// validate semantic and lifecycle replay before writing a new v2 file.
    pub fn read_legacy(path: impl AsRef<Path>) -> io::Result<Vec<CommittedEvent>> {
        Ok(Self::open_legacy_for_migration(path)?.events)
    }

    pub fn open_legacy_for_migration(path: impl AsRef<Path>) -> io::Result<LegacyLog> {
        let mut file = lock_existing(path.as_ref())?;
        let events = integrity::decode_legacy_log(&read_bounded(&mut file)?)?;
        Ok(LegacyLog {
            _source: file,
            events,
        })
    }

    fn from_verified(
        path: &Path,
        file: LockedFile,
        verified: VerifiedLog,
        length: usize,
    ) -> io::Result<Self> {
        Self::from_verified_above(path, file, verified, length, LogAnchor::empty())
    }

    fn from_verified_above(
        path: &Path,
        mut file: LockedFile,
        verified: VerifiedLog,
        length: usize,
        base: LogAnchor,
    ) -> io::Result<Self> {
        file.seek(SeekFrom::End(0))?;
        Ok(Self {
            path: path.to_owned(),
            file,
            events: verified.events,
            anchor: verified.anchor,
            base,
            length,
            poisoned: false,
        })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Compaction floor this handle was opened against.
    pub fn base(&self) -> LogAnchor {
        self.base
    }

    /// The origin this log can prove on its own, which is the digest of commit
    /// index 1.
    ///
    /// `None` once compaction has discarded that record: the origin then exists
    /// only in protected anchor storage, and re-deriving it from whatever record
    /// now happens to be first would let a substituted log name itself.
    pub fn derived_origin(&self) -> Option<ChainOrigin> {
        if self.base != LogAnchor::empty() {
            return None;
        }
        let first = self.events.first()?;
        let (_, after) = integrity::encode_record(&first.event, LogAnchor::empty()).ok()?;
        Some(ChainOrigin::from_first_record(after.digest))
    }
    pub fn events(&self) -> &[CommittedEvent] {
        &self.events
    }
    pub fn anchor(&self) -> io::Result<LogAnchor> {
        if self.poisoned {
            return Err(io::Error::other("PTR_LOG_INDETERMINATE"));
        }
        Ok(self.anchor)
    }
    pub fn append_durable(&mut self, event: LedgerEvent) -> io::Result<CommitIndex> {
        if self.poisoned {
            return Err(io::Error::other("PTR_LOG_INDETERMINATE"));
        }
        let (frame, anchor) = integrity::encode_record(&event, self.anchor)?;
        let new_length = self
            .length
            .checked_add(frame.len())
            .filter(|n| *n <= MAX_LOG_BYTES)
            .ok_or_else(|| integrity::invalid("PTR_LOG_SIZE_LIMIT"))?;
        // Do not write using stale coordinates if the trusted host violated the
        // cooperative locking contract by editing/truncating the open file.
        if self.file.metadata()?.len() != self.length as u64 {
            self.poisoned = true;
            return Err(integrity::invalid("PTR_LOG_EXTERNAL_LENGTH_CHANGE"));
        }
        self.poisoned = true;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("ledger.before_record_write");
        self.file.write_all(&frame[..FRAME_HEADER_BYTES])?;
        // Historical failpoint name retained; now after the full checked header.
        #[cfg(feature = "failpoints")]
        fail::fail_point!("ledger.after_length_before_payload");
        self.file.write_all(&frame[FRAME_HEADER_BYTES..])?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("ledger.after_payload_before_sync");
        self.file.flush()?;
        self.file.sync_data()?;
        #[cfg(feature = "failpoints")]
        fail::fail_point!("ledger.after_sync_before_memory");
        self.events.push(CommittedEvent {
            index: anchor.index,
            event,
        });
        self.anchor = anchor;
        self.length = new_length;
        self.poisoned = false;
        Ok(anchor.index)
    }
}

/// A log held under its single-writer lock while its relationship to protected
/// anchor storage is decided.
///
/// Nothing is written until [`Self::repair`] is called, and [`Self::classify`] is
/// a pure function of the bytes already read plus the anchor handed to it. That
/// split keeps the trust direction visible: the verdict is derived from trusted
/// input applied to untrusted bytes, never the other way around.
pub struct RecoverableLog {
    path: PathBuf,
    file: LockedFile,
    base: LogAnchor,
    prefix: integrity::Prefix,
    length: usize,
}

impl RecoverableLog {
    /// Anchor of the last complete record present.
    pub fn tail(&self) -> LogAnchor {
        self.prefix.verified.anchor
    }

    /// Whether a partial frame follows the last complete record.
    pub fn has_incomplete_frame(&self) -> bool {
        self.prefix.incomplete
    }

    /// Origin derivable from these bytes; `None` for a compacted log.
    pub fn derived_origin(&self) -> Option<ChainOrigin> {
        if self.base != LogAnchor::empty() {
            return None;
        }
        self.prefix
            .milestones
            .get(1)
            .map(|(_, after)| ChainOrigin::from_first_record(after.digest))
    }

    /// Decide how this log stands relative to `anchor`. Pure: no file is touched.
    pub fn classify(&self, anchor: ProtectedAnchor) -> Split {
        if anchor.base != self.base {
            return Split::BaseMismatch;
        }
        if let (Some(derived), true) = (self.derived_origin(), anchor.origin.is_set()) {
            if derived != anchor.origin {
                return Split::OriginMismatch;
            }
        }
        match self.prefix.records_after(anchor.log) {
            Some(0) if !self.prefix.incomplete => Split::Aligned,
            Some(complete_records) => Split::Unacknowledged {
                complete_records,
                incomplete_frame: self.prefix.incomplete,
                tail: self.tail(),
            },
            None if anchor.log.index.0 > self.tail().index.0 => Split::LostSuffix {
                acknowledged: anchor.log.index,
                present: self.tail().index,
            },
            None => Split::Diverged,
        }
    }

    /// Repair only the tail and return an open ledger.
    ///
    /// A partial trailing frame is always truncated. Complete records above the
    /// anchor are kept or removed per `policy`. Any unrecoverable split leaves
    /// every byte in place and fails.
    pub fn repair(
        self,
        anchor: ProtectedAnchor,
        policy: TailPolicy,
    ) -> io::Result<(FileLedger, TailRecovery)> {
        let (complete_records, incomplete_frame) = match self.classify(anchor) {
            Split::Aligned => (0usize, false),
            Split::Unacknowledged {
                complete_records,
                incomplete_frame,
                ..
            } => (complete_records, incomplete_frame),
            fault => return Err(unrecoverable(fault)),
        };
        if policy == TailPolicy::Refuse && (complete_records > 0 || incomplete_frame) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                AcknowledgedError::TailRefused {
                    complete_records,
                    incomplete_frame,
                }
                .code(),
            ));
        }
        let discard = policy == TailPolicy::Discard && complete_records > 0;
        let target = if discard { anchor.log } else { self.tail() };
        let Self {
            path,
            file,
            base,
            prefix,
            length,
        } = self;
        let keep_bytes = if discard {
            prefix
                .offset_of(anchor.log)
                .ok_or_else(|| unrecoverable(Split::Diverged))?
        } else {
            prefix.complete_bytes
        };
        if keep_bytes != length {
            file.set_len(keep_bytes as u64)?;
            file.sync_all()?;
        }
        let keep_events = target.index.0.saturating_sub(base.index.0) as usize;
        let mut events = prefix.verified.events;
        events.truncate(keep_events);
        let verified = VerifiedLog {
            events,
            anchor: target,
        };
        let ledger = FileLedger::from_verified_above(&path, file, verified, keep_bytes, base)?;
        let recovery = TailRecovery {
            acknowledged_records: if discard { 0 } else { complete_records },
            discarded_records: if discard { complete_records } else { 0 },
            incomplete_frame_removed: incomplete_frame,
            tail: target,
            origin: ledger.derived_origin(),
        };
        Ok((ledger, recovery))
    }
}

fn unrecoverable(fault: Split) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        AcknowledgedError::Unrecoverable(fault).code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn releasing_guard_unlocks_even_while_duplicate_descriptor_exists() {
        let path = std::env::temp_dir().join(format!(
            "ptr-lock-guard-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let guard = LockedFile::new(file).unwrap();
        let inherited = guard.try_clone().unwrap();
        assert!(lock_existing(&path).is_err());
        drop(guard);
        let replacement = lock_existing(&path).unwrap();
        drop(replacement);
        drop(inherited);
        std::fs::remove_file(path).unwrap();
    }
}
