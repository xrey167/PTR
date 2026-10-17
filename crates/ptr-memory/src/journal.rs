//! Append-only framed journal with torn-tail recovery and atomic compaction.
//!
//! # File format
//!
//! ```text
//! header (8 bytes): "PTMJ" | version u8 (=1) | kind u8 | 0 0
//! frame:            payload_len u32 LE | crc32(payload) u32 LE | payload
//! ```
//!
//! The CRC is IEEE CRC-32 (polynomial `0xEDB88320`) computed in this crate. The payload encoding
//! is owned by the store that uses the journal (see `DurableKnowledgeStore` and
//! `DurableMemoryStore`); the journal treats it as opaque bytes.
//!
//! # Recovery rules on open
//!
//! * An incomplete final frame (short header, or fewer payload bytes than announced) is a torn
//!   write: it is truncated away and reported in [`JournalRecovery::truncated_bytes`].
//! * A final frame that is complete but fails its CRC and ends exactly at end-of-file is also a
//!   torn write (the payload was only partially persisted) and is truncated.
//! * A frame that fails its CRC (or announces an impossible length) with more data after it is
//!   [`JournalError::Corrupt`]: the file is never silently repaired in the middle.
//!
//! # Compaction
//!
//! [`MemoryJournal::rewrite`] writes a complete replacement journal to a temporary sibling file,
//! syncs it and renames it over the journal. A crash before the rename leaves the old journal
//! intact; a stale temporary file is removed on the next open.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: [u8; 4] = *b"PTMJ";
const VERSION: u8 = 1;
const HEADER_LEN: usize = 8;
const FRAME_HEADER_LEN: usize = 8;

/// Hard ceiling for [`JournalOptions::max_record_bytes`]: 64 MiB. A payload is held in memory
/// several times over (frame buffer, decoded copy, decoded value), so a frame this large is
/// already far beyond any memory delta or knowledge operation; it is also well under the `u32`
/// frame length field.
pub const MAX_JOURNAL_RECORD_BYTES: usize = 64 * 1024 * 1024;

/// Hard ceiling for [`JournalOptions::max_file_bytes`]: 1 GiB. Opening reads the whole journal
/// into memory and decodes every record (roughly twice the file size resident), so this keeps
/// recovery bounded to a few GiB; compaction keeps real journals far below it.
pub const MAX_JOURNAL_FILE_BYTES: u64 = 1024 * 1024 * 1024;

/// Identifies which store owns a journal file so one cannot be opened as the other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JournalKind {
    Knowledge,
    Memory,
}

impl JournalKind {
    fn tag(self) -> u8 {
        match self {
            Self::Knowledge => 1,
            Self::Memory => 2,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Knowledge),
            2 => Some(Self::Memory),
            _ => None,
        }
    }
}

/// Bounds and durability options of a journal (Invariant 8: nothing is unbounded).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JournalOptions {
    /// Largest accepted frame payload (at most [`MAX_JOURNAL_RECORD_BYTES`]).
    pub max_record_bytes: usize,
    /// Largest journal file that will be loaded on open (at least the 8-byte header, at most
    /// [`MAX_JOURNAL_FILE_BYTES`]).
    pub max_file_bytes: u64,
    /// Call `sync_data` after every append.
    pub sync_on_append: bool,
}

impl Default for JournalOptions {
    fn default() -> Self {
        Self {
            max_record_bytes: 4 * 1024 * 1024,
            max_file_bytes: 256 * 1024 * 1024,
            sync_on_append: true,
        }
    }
}

/// What kind of damage was found in the middle of a journal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CorruptionReason {
    /// The stored CRC does not match the payload.
    ChecksumMismatch,
    /// The announced payload length exceeds the configured maximum.
    OversizedFrame,
}

/// Typed journal failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JournalError {
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        message: String,
    },
    /// The file does not start with a valid journal header.
    BadHeader,
    /// The header names an unsupported format version.
    UnsupportedVersion(u8),
    /// The file belongs to another store kind.
    KindMismatch {
        expected: JournalKind,
        found: Option<JournalKind>,
    },
    /// Damage in the middle of the file at byte `offset`. The file is left untouched.
    Corrupt {
        offset: u64,
        reason: CorruptionReason,
    },
    /// An append or rewrite payload exceeds `max_record_bytes`.
    RecordTooLarge { len: usize, max: usize },
    /// The file exceeds `max_file_bytes`. On open the file is refused; on append or rewrite the
    /// operation is refused before anything is written (`len` is the projected size).
    FileTooLarge { len: u64, max: u64 },
    /// An earlier write failed; the journal refuses further appends because the file tail is
    /// unknown. Reopen it to recover.
    Poisoned,
}

fn io_error(operation: &'static str, error: &io::Error) -> JournalError {
    JournalError::Io {
        operation,
        kind: error.kind(),
        message: error.to_string(),
    }
}

/// Result of opening a journal: the intact payloads in append order.
#[derive(Debug, Eq, PartialEq)]
pub struct JournalRecovery {
    pub records: Vec<Vec<u8>>,
    /// Bytes removed from the tail because of a torn write (0 when the file was clean).
    pub truncated_bytes: u64,
}

const fn crc_table() -> [u32; 256] {
    let mut table = [0_u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = crc_table();

/// IEEE CRC-32 of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for byte in data {
        let index = ((crc ^ u32::from(*byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[index];
    }
    !crc
}

fn header(kind: JournalKind) -> [u8; HEADER_LEN] {
    let mut header = [0_u8; HEADER_LEN];
    header[..4].copy_from_slice(&MAGIC);
    header[4] = VERSION;
    header[5] = kind.tag();
    header
}

fn encode_frame(out: &mut Vec<u8>, payload: &[u8], max: usize) -> Result<(), JournalError> {
    if payload.len() > max {
        return Err(JournalError::RecordTooLarge {
            len: payload.len(),
            max,
        });
    }
    let len = u32::try_from(payload.len()).map_err(|_| JournalError::RecordTooLarge {
        len: payload.len(),
        max,
    })?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&crc32(payload).to_le_bytes());
    out.extend_from_slice(payload);
    Ok(())
}

/// An open append-only journal.
#[derive(Debug)]
pub struct MemoryJournal {
    path: PathBuf,
    kind: JournalKind,
    options: JournalOptions,
    file: File,
    /// Current file length; every frame is checked against `max_file_bytes` before it is written.
    len: u64,
    appended_since_open_or_rewrite: u64,
    poisoned: bool,
    #[cfg(test)]
    fault: FaultInjection,
}

/// Test-only fault seam: makes the next sync and/or the rollback truncate fail.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaultInjection {
    pub(crate) fail_sync: bool,
    pub(crate) fail_truncate: bool,
    /// Makes the directory sync after a compaction rename fail.
    pub(crate) fail_sync_parent: bool,
}

impl MemoryJournal {
    /// Opens (creating if absent) the journal at `path`, recovering a torn tail.
    pub fn open(
        path: impl AsRef<Path>,
        kind: JournalKind,
        options: JournalOptions,
    ) -> Result<(Self, JournalRecovery), JournalError> {
        let path = path.as_ref().to_path_buf();
        // Every journal starts with its header, so a smaller cap could never hold a valid file:
        // refuse it before anything is created or rewritten.
        let header_len = HEADER_LEN as u64;
        if options.max_file_bytes < header_len {
            return Err(JournalError::FileTooLarge {
                len: header_len,
                max: options.max_file_bytes,
            });
        }
        // Hard ceilings: `open` reads the whole file into memory, so an unbounded cap would let
        // a caller (or a growing file) exhaust it.
        if options.max_file_bytes > MAX_JOURNAL_FILE_BYTES {
            return Err(JournalError::FileTooLarge {
                len: options.max_file_bytes,
                max: MAX_JOURNAL_FILE_BYTES,
            });
        }
        if options.max_record_bytes > MAX_JOURNAL_RECORD_BYTES {
            return Err(JournalError::RecordTooLarge {
                len: options.max_record_bytes,
                max: MAX_JOURNAL_RECORD_BYTES,
            });
        }
        let tmp = temp_path(&path);
        match fs::remove_file(&tmp) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(io_error("remove stale compaction file", &error)),
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| io_error("open", &e))?;
        let len = file.metadata().map_err(|e| io_error("stat", &e))?.len();
        if len > options.max_file_bytes {
            return Err(JournalError::FileTooLarge {
                len,
                max: options.max_file_bytes,
            });
        }
        // Another writer may append after the metadata check: never read past the cap, and
        // refuse a file that grew beyond it instead of allocating for the growth.
        let bytes = read_bounded(&mut file, len, options.max_file_bytes)?;

        let expected = header(kind);
        if bytes.len() < HEADER_LEN {
            // An empty file, or a header that was only partly written when the process died.
            if !expected.starts_with(&bytes) {
                return Err(JournalError::BadHeader);
            }
            file.set_len(0).map_err(|e| io_error("truncate", &e))?;
            drop(file);
            write_new(&path, &expected)?;
            let journal = Self::reopen(path, kind, options)?;
            return Ok((
                journal,
                JournalRecovery {
                    records: Vec::new(),
                    truncated_bytes: 0,
                },
            ));
        }
        if bytes[..4] != MAGIC {
            return Err(JournalError::BadHeader);
        }
        if bytes[4] != VERSION {
            return Err(JournalError::UnsupportedVersion(bytes[4]));
        }
        if bytes[5] != kind.tag() {
            return Err(JournalError::KindMismatch {
                expected: kind,
                found: JournalKind::from_tag(bytes[5]),
            });
        }

        let (records, valid_len) = scan(&bytes, options.max_record_bytes)?;
        let truncated_bytes = (bytes.len() - valid_len) as u64;
        if truncated_bytes > 0 {
            file.set_len(valid_len as u64)
                .map_err(|e| io_error("truncate torn tail", &e))?;
            file.sync_all().map_err(|e| io_error("sync", &e))?;
        }
        drop(file);
        let journal = Self::reopen(path, kind, options)?;
        Ok((
            journal,
            JournalRecovery {
                records,
                truncated_bytes,
            },
        ))
    }

    fn reopen(
        path: PathBuf,
        kind: JournalKind,
        options: JournalOptions,
    ) -> Result<Self, JournalError> {
        let file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| io_error("open for append", &e))?;
        let len = file.metadata().map_err(|e| io_error("stat", &e))?.len();
        Ok(Self {
            path,
            kind,
            options,
            file,
            len,
            appended_since_open_or_rewrite: 0,
            poisoned: false,
            #[cfg(test)]
            fault: FaultInjection::default(),
        })
    }

    /// True once a failed write could not be rolled back; the journal then refuses appends.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[cfg(test)]
    pub(crate) fn inject_fault(&mut self, fault: FaultInjection) {
        self.fault = fault;
    }

    fn sync_appended(&self) -> io::Result<()> {
        #[cfg(test)]
        if self.fault.fail_sync {
            return Err(io::Error::other("injected sync failure"));
        }
        self.file.sync_data()
    }

    /// Cuts the file back to `len` after a failed append, so a frame the caller was told failed
    /// can never become visible on a later open.
    ///
    /// This uses a separate handle opened with plain `write(true)`: the journal's own handle is
    /// append-only, and on Windows an append-only handle lacks FILE_WRITE_DATA and cannot be
    /// truncated.
    fn roll_back_to(&self, len: u64) -> io::Result<()> {
        #[cfg(test)]
        if self.fault.fail_truncate {
            return Err(io::Error::other("injected truncate failure"));
        }
        let file = OpenOptions::new().write(true).open(&self.path)?;
        file.set_len(len)?;
        file.sync_all()
    }

    fn persist_rename(&self) -> io::Result<()> {
        #[cfg(test)]
        if self.fault.fail_sync_parent {
            return Err(io::Error::other("injected directory sync failure"));
        }
        sync_parent(&self.path)
    }

    /// Path of the journal file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Payloads appended since the journal was opened or last rewritten.
    pub fn appended_since_rewrite(&self) -> u64 {
        self.appended_since_open_or_rewrite
    }

    /// Reads the intact payloads currently on disk without modifying the file (a torn tail is
    /// ignored, not truncated). Used to rebuild in-memory state after a failed append.
    pub(crate) fn read_records(&self) -> Result<Vec<Vec<u8>>, JournalError> {
        let mut file = File::open(&self.path).map_err(|e| io_error("open", &e))?;
        let bytes = read_bounded(&mut file, self.len, self.options.max_file_bytes)?;
        if bytes.len() < HEADER_LEN || bytes[..HEADER_LEN] != header(self.kind) {
            return Err(JournalError::BadHeader);
        }
        scan(&bytes, self.options.max_record_bytes).map(|(records, _)| records)
    }

    /// Appends one framed payload. If the write, flush or sync fails the file is truncated back
    /// to its previous length, so the failed frame is never visible, and the journal stays usable.
    /// Only when that rollback also fails is the journal poisoned.
    pub fn append(&mut self, payload: &[u8]) -> Result<(), JournalError> {
        if self.poisoned {
            return Err(JournalError::Poisoned);
        }
        let mut frame = Vec::with_capacity(FRAME_HEADER_LEN + payload.len());
        encode_frame(&mut frame, payload, self.options.max_record_bytes)?;
        // Refuse before writing: a file past `max_file_bytes` could not be opened again.
        let projected = self.len.saturating_add(frame.len() as u64);
        if projected > self.options.max_file_bytes {
            return Err(JournalError::FileTooLarge {
                len: projected,
                max: self.options.max_file_bytes,
            });
        }
        let result = self.file.write_all(&frame).and_then(|()| {
            if self.options.sync_on_append {
                self.sync_appended()
            } else {
                Ok(())
            }
        });
        match result {
            Ok(()) => {
                self.len = projected;
                self.appended_since_open_or_rewrite =
                    self.appended_since_open_or_rewrite.saturating_add(1);
                Ok(())
            }
            Err(error) => {
                if self.roll_back_to(self.len).is_err() {
                    self.poisoned = true;
                }
                Err(io_error("append", &error))
            }
        }
    }

    /// Atomically replaces the journal content with exactly `payloads` (compaction).
    pub fn rewrite(&mut self, payloads: &[Vec<u8>]) -> Result<(), JournalError> {
        if self.poisoned {
            return Err(JournalError::Poisoned);
        }
        let mut image = header(self.kind).to_vec();
        for payload in payloads {
            encode_frame(&mut image, payload, self.options.max_record_bytes)?;
        }
        let image_len = image.len() as u64;
        if image_len > self.options.max_file_bytes {
            return Err(JournalError::FileTooLarge {
                len: image_len,
                max: self.options.max_file_bytes,
            });
        }
        let tmp = temp_path(&self.path);
        let write = || -> io::Result<()> {
            let mut file = File::create(&tmp)?;
            file.write_all(&image)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&tmp, &self.path)
        };
        if let Err(error) = write() {
            // The old journal is untouched; drop the half-written temp file.
            let _ = fs::remove_file(&tmp);
            return Err(io_error("rewrite", &error));
        }
        if let Err(error) = self.persist_rename() {
            // The rename may not be durable: a crash could bring the pre-compaction journal
            // back, and anything acknowledged against the replacement would be lost with it.
            // The handle also still points at the replaced inode. Refuse further writes.
            self.poisoned = true;
            return Err(io_error("sync journal directory", &error));
        }
        match OpenOptions::new().append(true).open(&self.path) {
            Ok(file) => {
                self.file = file;
                self.len = image_len;
                self.appended_since_open_or_rewrite = 0;
                Ok(())
            }
            Err(error) => {
                self.poisoned = true;
                Err(io_error("reopen after rewrite", &error))
            }
        }
    }
}

/// Reads at most `max + 1` bytes: a file that grew past `max` after its length was checked is
/// refused as [`JournalError::FileTooLarge`] instead of being read (and allocated) in full.
/// `expected_len` only sizes the initial reservation.
fn read_bounded(
    source: &mut impl Read,
    expected_len: u64,
    max: u64,
) -> Result<Vec<u8>, JournalError> {
    let mut bytes = Vec::with_capacity(usize::try_from(expected_len.min(max)).unwrap_or(0));
    source
        .take(max.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| io_error("read", &e))?;
    if bytes.len() as u64 > max {
        return Err(JournalError::FileTooLarge {
            len: bytes.len() as u64,
            max,
        });
    }
    Ok(bytes)
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".compact.tmp");
    PathBuf::from(name)
}

fn write_new(path: &Path, header: &[u8]) -> Result<(), JournalError> {
    let mut file = File::create(path).map_err(|e| io_error("create", &e))?;
    file.write_all(header)
        .map_err(|e| io_error("write header", &e))?;
    file.sync_all().map_err(|e| io_error("sync", &e))
}

/// Persists the rename itself by syncing the parent directory; errors are returned.
#[cfg(unix)]
fn sync_parent(path: &Path) -> io::Result<()> {
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    File::open(parent)?.sync_all()
}

/// Windows (and other non-Unix targets): std cannot open a directory as a `File`, and NTFS
/// persists the rename through the file-system journal, so there is nothing to sync here.
#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// Parses the frames after the header. Returns the payloads and the length of the valid prefix.
fn scan(bytes: &[u8], max_record: usize) -> Result<(Vec<Vec<u8>>, usize), JournalError> {
    let mut records = Vec::new();
    let mut pos = HEADER_LEN;
    while pos < bytes.len() {
        let remaining = bytes.len() - pos;
        if remaining < FRAME_HEADER_LEN {
            return Ok((records, pos)); // torn frame header
        }
        let mut len_raw = [0_u8; 4];
        len_raw.copy_from_slice(&bytes[pos..pos + 4]);
        let mut crc_raw = [0_u8; 4];
        crc_raw.copy_from_slice(&bytes[pos + 4..pos + 8]);
        let len = u32::from_le_bytes(len_raw) as usize;
        let stored_crc = u32::from_le_bytes(crc_raw);
        if len > max_record {
            return Err(JournalError::Corrupt {
                offset: pos as u64,
                reason: CorruptionReason::OversizedFrame,
            });
        }
        let body_start = pos + FRAME_HEADER_LEN;
        if remaining - FRAME_HEADER_LEN < len {
            return Ok((records, pos)); // torn payload
        }
        let payload = &bytes[body_start..body_start + len];
        if crc32(payload) != stored_crc {
            if body_start + len == bytes.len() {
                return Ok((records, pos)); // last frame only partially persisted
            }
            return Err(JournalError::Corrupt {
                offset: pos as u64,
                reason: CorruptionReason::ChecksumMismatch,
            });
        }
        records.push(payload.to_vec());
        pos = body_start + len;
    }
    Ok((records, pos))
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A unique temp path removed (with its compaction sibling) on drop.
    pub(crate) struct TempPath(pub PathBuf);

    impl TempPath {
        pub(crate) fn new(tag: &str) -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "ptr-memory-{tag}-{}-{n}.journal",
                std::process::id()
            )))
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            let mut tmp = self.0.as_os_str().to_owned();
            tmp.push(".compact.tmp");
            let _ = std::fs::remove_file(PathBuf::from(tmp));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::TempPath;
    use super::*;

    fn open(path: &Path) -> (MemoryJournal, JournalRecovery) {
        MemoryJournal::open(path, JournalKind::Memory, JournalOptions::default()).unwrap()
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn append_and_reopen_round_trip() {
        let path = TempPath::new("rt");
        let (mut journal, recovery) = open(&path.0);
        assert!(recovery.records.is_empty());
        journal.append(b"one").unwrap();
        journal.append(b"").unwrap();
        journal.append(b"three").unwrap();
        drop(journal);
        let (_, recovery) = open(&path.0);
        assert_eq!(
            recovery.records,
            vec![b"one".to_vec(), Vec::new(), b"three".to_vec()]
        );
        assert_eq!(recovery.truncated_bytes, 0);
    }

    #[test]
    fn kind_mismatch_and_bad_header_are_typed() {
        let path = TempPath::new("kind");
        let (journal, _) = open(&path.0);
        drop(journal);
        let err = MemoryJournal::open(&path.0, JournalKind::Knowledge, JournalOptions::default())
            .unwrap_err();
        assert_eq!(
            err,
            JournalError::KindMismatch {
                expected: JournalKind::Knowledge,
                found: Some(JournalKind::Memory)
            }
        );
        fs::write(&path.0, b"NOTAJOURNALFILE").unwrap();
        assert_eq!(
            MemoryJournal::open(&path.0, JournalKind::Memory, JournalOptions::default())
                .unwrap_err(),
            JournalError::BadHeader
        );
    }

    #[test]
    fn a_cap_below_the_header_is_refused_before_anything_is_written() {
        let path = TempPath::new("tinycap");
        let options = JournalOptions {
            max_file_bytes: HEADER_LEN as u64 - 1,
            ..JournalOptions::default()
        };
        let err = MemoryJournal::open(&path.0, JournalKind::Memory, options).unwrap_err();
        assert_eq!(
            err,
            JournalError::FileTooLarge {
                len: HEADER_LEN as u64,
                max: HEADER_LEN as u64 - 1
            }
        );
        // Nothing was created, so the same configuration fails the same way on every open.
        assert!(!path.0.exists());
        assert_eq!(
            MemoryJournal::open(&path.0, JournalKind::Memory, options).unwrap_err(),
            err
        );
    }

    #[test]
    fn oversized_append_is_rejected_without_poisoning() {
        let path = TempPath::new("big");
        let options = JournalOptions {
            max_record_bytes: 4,
            ..JournalOptions::default()
        };
        let (mut journal, _) = MemoryJournal::open(&path.0, JournalKind::Memory, options).unwrap();
        assert_eq!(
            journal.append(b"12345"),
            Err(JournalError::RecordTooLarge { len: 5, max: 4 })
        );
        journal.append(b"1234").unwrap();
    }

    #[test]
    fn rewrite_replaces_content_atomically() {
        let path = TempPath::new("rw");
        let (mut journal, _) = open(&path.0);
        for i in 0..10_u8 {
            journal.append(&[i]).unwrap();
        }
        assert_eq!(journal.appended_since_rewrite(), 10);
        journal.rewrite(&[b"snap".to_vec()]).unwrap();
        assert_eq!(journal.appended_since_rewrite(), 0);
        journal.append(b"after").unwrap();
        drop(journal);
        let (_, recovery) = open(&path.0);
        assert_eq!(recovery.records, vec![b"snap".to_vec(), b"after".to_vec()]);
        assert!(!temp_path(&path.0).exists());
    }

    #[test]
    fn sync_failure_rolls_the_frame_back_and_keeps_the_journal_usable() {
        let path = TempPath::new("syncfail");
        let (mut journal, _) = open(&path.0);
        journal.append(b"one").unwrap();
        let before = fs::read(&path.0).unwrap();
        journal.inject_fault(FaultInjection {
            fail_sync: true,
            ..FaultInjection::default()
        });
        assert!(matches!(
            journal.append(b"two"),
            Err(JournalError::Io {
                operation: "append",
                ..
            })
        ));
        assert_eq!(fs::read(&path.0).unwrap(), before);
        assert!(!journal.is_poisoned());
        journal.inject_fault(FaultInjection::default());
        journal.append(b"two").unwrap();
        drop(journal);
        let (_, recovery) = open(&path.0);
        assert_eq!(recovery.records, vec![b"one".to_vec(), b"two".to_vec()]);
    }

    #[test]
    fn failed_rollback_poisons_the_journal() {
        let path = TempPath::new("truncfail");
        let (mut journal, _) = open(&path.0);
        journal.inject_fault(FaultInjection {
            fail_sync: true,
            fail_truncate: true,
            ..FaultInjection::default()
        });
        assert!(journal.append(b"x").is_err());
        assert!(journal.is_poisoned());
        journal.inject_fault(FaultInjection::default());
        assert_eq!(journal.append(b"y"), Err(JournalError::Poisoned));
    }

    #[test]
    fn stale_compaction_file_is_removed_on_open() {
        let path = TempPath::new("stale");
        fs::write(temp_path(&path.0), b"half written").unwrap();
        let (_, _) = open(&path.0);
        assert!(!temp_path(&path.0).exists());
    }

    #[test]
    fn directory_sync_failure_after_rename_is_reported_and_poisons() {
        let path = TempPath::new("dirsync");
        let (mut journal, _) = open(&path.0);
        journal.append(b"before").unwrap();
        journal.inject_fault(FaultInjection {
            fail_sync_parent: true,
            ..FaultInjection::default()
        });
        assert!(matches!(
            journal.rewrite(&[b"snap".to_vec()]),
            Err(JournalError::Io {
                operation: "sync journal directory",
                ..
            })
        ));
        assert!(journal.is_poisoned());
        journal.inject_fault(FaultInjection::default());
        // No acknowledged append can land on a replacement whose rename may not be durable.
        assert_eq!(journal.append(b"after"), Err(JournalError::Poisoned));
        assert_eq!(journal.rewrite(&[]), Err(JournalError::Poisoned));
    }

    #[cfg(unix)]
    #[test]
    fn sync_parent_reports_a_missing_directory_and_accepts_a_bare_file_name() {
        let missing = std::env::temp_dir().join("ptr-memory-no-such-dir-xyz/file");
        assert!(sync_parent(&missing).is_err());
        assert!(sync_parent(Path::new("journal-in-cwd")).is_ok());
        assert!(sync_parent(&std::env::temp_dir().join("anything")).is_ok());
    }

    #[test]
    fn open_rejects_limits_above_the_hard_ceilings_before_touching_the_file() {
        let path = TempPath::new("ceilings");
        let file_options = JournalOptions {
            max_file_bytes: u64::MAX,
            max_record_bytes: usize::MAX,
            ..JournalOptions::default()
        };
        assert_eq!(
            MemoryJournal::open(&path.0, JournalKind::Memory, file_options).unwrap_err(),
            JournalError::FileTooLarge {
                len: u64::MAX,
                max: MAX_JOURNAL_FILE_BYTES
            }
        );
        let record_options = JournalOptions {
            max_record_bytes: MAX_JOURNAL_RECORD_BYTES + 1,
            ..JournalOptions::default()
        };
        assert_eq!(
            MemoryJournal::open(&path.0, JournalKind::Memory, record_options).unwrap_err(),
            JournalError::RecordTooLarge {
                len: MAX_JOURNAL_RECORD_BYTES + 1,
                max: MAX_JOURNAL_RECORD_BYTES
            }
        );
        assert!(!path.0.exists());
        let at_ceiling = JournalOptions {
            max_file_bytes: MAX_JOURNAL_FILE_BYTES,
            max_record_bytes: MAX_JOURNAL_RECORD_BYTES,
            sync_on_append: false,
        };
        MemoryJournal::open(&path.0, JournalKind::Memory, at_ceiling).unwrap();
    }

    #[test]
    fn default_options_are_within_the_ceilings() {
        let defaults = JournalOptions::default();
        assert!(defaults.max_record_bytes <= MAX_JOURNAL_RECORD_BYTES);
        assert!(defaults.max_file_bytes <= MAX_JOURNAL_FILE_BYTES);
    }

    #[test]
    fn bounded_read_refuses_growth_past_the_cap_without_reading_it_all() {
        // The source is far longer than the cap, as if another writer kept appending after the
        // metadata check; only `max + 1` bytes may be consumed.
        let mut source = io::repeat(7).take(10_000_000);
        assert_eq!(
            read_bounded(&mut source, 16, 32),
            Err(JournalError::FileTooLarge { len: 33, max: 32 })
        );
        assert_eq!(source.limit(), 10_000_000 - 33);
        let mut exact = io::Cursor::new(vec![1_u8; 32]);
        assert_eq!(read_bounded(&mut exact, 32, 32).unwrap().len(), 32);
    }

    #[test]
    fn read_records_refuses_a_file_that_grew_past_the_cap_since_open() {
        let path = TempPath::new("grown");
        let options = JournalOptions {
            max_file_bytes: 8 + 8 + 4,
            max_record_bytes: 64,
            ..JournalOptions::default()
        };
        let (mut journal, _) = MemoryJournal::open(&path.0, JournalKind::Memory, options).unwrap();
        journal.append(b"1234").unwrap();
        // A second writer appends beyond the cap behind the journal's back.
        let mut other = OpenOptions::new().append(true).open(&path.0).unwrap();
        other.write_all(&[0_u8; 4096]).unwrap();
        assert!(matches!(
            journal.read_records(),
            Err(JournalError::FileTooLarge { max, .. }) if max == options.max_file_bytes
        ));
    }
}
