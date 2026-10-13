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
    /// Largest accepted frame payload.
    pub max_record_bytes: usize,
    /// Largest journal file that will be loaded on open.
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
    /// The file exceeds `max_file_bytes`.
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
    appended_since_open_or_rewrite: u64,
    poisoned: bool,
}

impl MemoryJournal {
    /// Opens (creating if absent) the journal at `path`, recovering a torn tail.
    pub fn open(
        path: impl AsRef<Path>,
        kind: JournalKind,
        options: JournalOptions,
    ) -> Result<(Self, JournalRecovery), JournalError> {
        let path = path.as_ref().to_path_buf();
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
        let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
        file.read_to_end(&mut bytes)
            .map_err(|e| io_error("read", &e))?;

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
        Ok(Self {
            path,
            kind,
            options,
            file,
            appended_since_open_or_rewrite: 0,
            poisoned: false,
        })
    }

    /// Path of the journal file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Payloads appended since the journal was opened or last rewritten.
    pub fn appended_since_rewrite(&self) -> u64 {
        self.appended_since_open_or_rewrite
    }

    /// Appends one framed payload. After an I/O failure the journal is poisoned.
    pub fn append(&mut self, payload: &[u8]) -> Result<(), JournalError> {
        if self.poisoned {
            return Err(JournalError::Poisoned);
        }
        let mut frame = Vec::with_capacity(FRAME_HEADER_LEN + payload.len());
        encode_frame(&mut frame, payload, self.options.max_record_bytes)?;
        let result = self.file.write_all(&frame).and_then(|()| {
            if self.options.sync_on_append {
                self.file.sync_data()
            } else {
                Ok(())
            }
        });
        match result {
            Ok(()) => {
                self.appended_since_open_or_rewrite =
                    self.appended_since_open_or_rewrite.saturating_add(1);
                Ok(())
            }
            Err(error) => {
                self.poisoned = true;
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
        sync_parent(&self.path);
        match OpenOptions::new().append(true).open(&self.path) {
            Ok(file) => {
                self.file = file;
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

/// Best effort: persist the rename itself. Unsupported on some platforms, so errors are ignored.
fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
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
    fn stale_compaction_file_is_removed_on_open() {
        let path = TempPath::new("stale");
        fs::write(temp_path(&path.0), b"half written").unwrap();
        let (_, _) = open(&path.0);
        assert!(!temp_path(&path.0).exists());
    }
}
