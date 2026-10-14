//! A persistent, append-only, segmented event log on the local file system (std only).
//!
//! Layout inside the log directory:
//!
//! * `segment-<first offset, 20 digits>.log`: framed records (see `codec`), in offset order;
//! * `offsets.bin`: the durable consumer offsets, rewritten atomically (temp file + rename).
//!
//! The log has a single writer: nothing locks the directory, so two processes must not open it
//! at the same time. It is a local stand-in for a durable event stream, not an Apache Iggy
//! adapter and not replicated.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use crate::bus::{
    validate_consumer, validate_new, BusError, BusRecord, CorruptionKind, EventConsumer,
    EventProducer, NewRecord, Offset, DEFAULT_MAX_RECORD_BYTES, MAX_CONSUMERS,
};
use crate::codec::{
    crc32, decode_body, encode_frame, max_body, FRAME_HEADER, MAX_FIELD_BYTES, MIN_BODY,
};
use crate::policy::{
    slowest, BackpressurePolicy, ClassPolicies, Compaction, ConsumerLag, Limit, PolicyViolation,
    RetentionPolicy,
};

const OFFSETS_FILE: &str = "offsets.bin";
const OFFSETS_TEMP: &str = "offsets.bin.tmp";
const OFFSETS_MAGIC: &[u8; 8] = b"PTROFF01";
const MAX_OFFSETS_BYTES: u64 = (MAX_CONSUMERS * (2 + MAX_FIELD_BYTES + 8) + 16) as u64;
/// Default segment size.
pub const DEFAULT_SEGMENT_BYTES: u64 = 8 << 20;

/// When appended data is forced to stable storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FsyncPolicy {
    /// `sync_data` after every record: an acknowledged publish survives a crash.
    Always,
    /// `sync_data` after every N records: up to N-1 acknowledged records may be lost on a crash.
    EveryRecords(NonZeroU32),
    /// Never sync from the log itself (rollover and offsets included); the OS decides.
    Never,
}

/// Configuration of a [`FileEventLog`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileLogConfig {
    /// A segment is closed before an append that would grow it past this many bytes (a single
    /// larger record still gets a segment of its own).
    pub segment_max_bytes: u64,
    pub fsync: FsyncPolicy,
    /// Bound on records some registered consumer has not acknowledged.
    pub max_unacked_records: usize,
    /// Bound on one record's payload.
    pub max_record_bytes: usize,
    /// The file log cannot drop individual records, so the telemetry policy must be `Reject`.
    pub policies: ClassPolicies,
    /// Retention of acknowledged history, applied by deleting whole closed segments.
    pub retention: RetentionPolicy,
}

impl FileLogConfig {
    pub fn new(max_unacked_records: usize) -> Self {
        Self {
            segment_max_bytes: DEFAULT_SEGMENT_BYTES,
            fsync: FsyncPolicy::Always,
            max_unacked_records,
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
            policies: ClassPolicies::reject_all(),
            retention: RetentionPolicy::acked_only(),
        }
    }

    fn validate(&self) -> Result<(), BusError> {
        if self.max_unacked_records == 0 || self.segment_max_bytes == 0 {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::ZeroBound,
            });
        }
        if max_body(self.max_record_bytes).is_none() {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::RecordLimitTooLarge,
            });
        }
        if self.retention.is_unbounded() {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::UnboundedRetention,
            });
        }
        if self.policies.telemetry() == BackpressurePolicy::DropOldestTelemetry {
            return Err(BusError::InvalidPolicy {
                violation: PolicyViolation::DropUnsupportedByBackend,
            });
        }
        Ok(())
    }
}

/// A partially written or checksum-damaged last frame removed during recovery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TornTail {
    pub segment: String,
    pub discarded_bytes: u64,
}

/// What opening the log found.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveryReport {
    pub segments: usize,
    /// Records retained after recovery.
    pub records: u64,
    pub torn_tail: Option<TornTail>,
}

#[derive(Clone, Debug)]
struct Segment {
    path: PathBuf,
    first: u64,
    last: Option<u64>,
    bytes: u64,
    newest_tick: u64,
}

impl Segment {
    fn name(&self) -> String {
        segment_name(self.first)
    }
}

fn segment_name(first: u64) -> String {
    format!("segment-{first:020}.log")
}

fn parse_segment_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("segment-")?.strip_suffix(".log")?;
    if digits.len() != 20 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Tear {
    Incomplete,
    ChecksumAtEnd,
}

enum Frame {
    End,
    Torn(Tear),
    Record { record: BusRecord, end_pos: u64 },
}

/// The largest frame body for a record bound; the config validator refuses bounds without one.
fn frame_limit(max_record_bytes: usize) -> Result<usize, BusError> {
    max_body(max_record_bytes).ok_or(BusError::InvalidPolicy {
        violation: PolicyViolation::RecordLimitTooLarge,
    })
}

fn corrupt(file: &str, position: u64, kind: CorruptionKind) -> BusError {
    BusError::Corrupt {
        file: file.to_owned(),
        position,
        kind,
    }
}

fn corrupt_from_tear(file: &str, position: u64, tear: Tear) -> BusError {
    corrupt(
        file,
        position,
        match tear {
            Tear::Incomplete => CorruptionKind::IncompleteFrame,
            Tear::ChecksumAtEnd => CorruptionKind::ChecksumMismatch,
        },
    )
}

/// Read the frame at `pos`; bytes at and after `limit` do not exist.
fn read_frame<R: Read>(
    reader: &mut R,
    pos: u64,
    limit: u64,
    max_body_len: usize,
    file: &str,
) -> Result<Frame, BusError> {
    let remaining = limit.saturating_sub(pos);
    if remaining == 0 {
        return Ok(Frame::End);
    }
    if remaining < FRAME_HEADER as u64 {
        return Ok(Frame::Torn(Tear::Incomplete));
    }
    let mut header = [0u8; FRAME_HEADER];
    reader
        .read_exact(&mut header)
        .map_err(|e| BusError::io("read segment", &e))?;
    let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let crc = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if !(MIN_BODY..=max_body_len).contains(&len) {
        return Err(corrupt(file, pos, CorruptionKind::BadLength));
    }
    let frame_len = (FRAME_HEADER + len) as u64;
    if remaining < frame_len {
        return Ok(Frame::Torn(Tear::Incomplete));
    }
    let mut body = vec![0u8; len];
    reader
        .read_exact(&mut body)
        .map_err(|e| BusError::io("read segment", &e))?;
    if crc32(&body) != crc {
        return if remaining == frame_len {
            Ok(Frame::Torn(Tear::ChecksumAtEnd))
        } else {
            Err(corrupt(file, pos, CorruptionKind::ChecksumMismatch))
        };
    }
    let record =
        decode_body(&body).ok_or_else(|| corrupt(file, pos, CorruptionKind::BadEncoding))?;
    Ok(Frame::Record {
        record,
        end_pos: pos + frame_len,
    })
}

struct Scan {
    last: Option<u64>,
    valid_bytes: u64,
    newest_tick: u64,
    count: u64,
    tear: Option<Tear>,
}

fn scan_segment(
    path: &Path,
    name: &str,
    base: u64,
    file_len: u64,
    max_body_len: usize,
) -> Result<Scan, BusError> {
    let file = File::open(path).map_err(|e| BusError::io("open segment", &e))?;
    let mut reader = BufReader::new(file);
    let mut scan = Scan {
        last: None,
        valid_bytes: 0,
        newest_tick: 0,
        count: 0,
        tear: None,
    };
    let mut expected = base;
    loop {
        match read_frame(&mut reader, scan.valid_bytes, file_len, max_body_len, name)? {
            Frame::End => return Ok(scan),
            Frame::Torn(tear) => {
                scan.tear = Some(tear);
                return Ok(scan);
            }
            Frame::Record { record, end_pos } => {
                if record.offset.0 != expected {
                    return Err(corrupt(
                        name,
                        scan.valid_bytes,
                        CorruptionKind::OffsetGap {
                            expected,
                            found: record.offset.0,
                        },
                    ));
                }
                // `u64::MAX` can never be followed by a next offset, so this log never writes it.
                let Some(next_expected) = expected.checked_add(1) else {
                    return Err(corrupt(name, scan.valid_bytes, CorruptionKind::BadEncoding));
                };
                scan.last = Some(expected);
                scan.newest_tick = record.tick;
                scan.count += 1;
                scan.valid_bytes = end_pos;
                expected = next_expected;
            }
        }
    }
}

fn sync_dir(dir: &Path) {
    // Directory fsync is how a rename or create becomes durable; it is not supported everywhere.
    if let Ok(handle) = File::open(dir) {
        let _ = handle.sync_all();
    }
}

fn write_atomic(dir: &Path, bytes: &[u8], sync: bool) -> Result<(), BusError> {
    let temp = dir.join(OFFSETS_TEMP);
    let mut file = File::create(&temp).map_err(|e| BusError::io("create offsets temp", &e))?;
    file.write_all(bytes)
        .map_err(|e| BusError::io("write offsets temp", &e))?;
    if sync {
        file.sync_all()
            .map_err(|e| BusError::io("sync offsets temp", &e))?;
    }
    drop(file);
    fs::rename(&temp, dir.join(OFFSETS_FILE)).map_err(|e| BusError::io("rename offsets", &e))?;
    if sync {
        sync_dir(dir);
    }
    Ok(())
}

fn encode_offsets(committed: &BTreeMap<String, Offset>) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(OFFSETS_MAGIC);
    bytes.extend_from_slice(&(committed.len() as u32).to_le_bytes());
    for (name, offset) in committed {
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&offset.0.to_le_bytes());
    }
    let crc = crc32(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());
    bytes
}

fn decode_offsets(bytes: &[u8]) -> Result<BTreeMap<String, Offset>, BusError> {
    let bad = |position: usize, kind| corrupt(OFFSETS_FILE, position as u64, kind);
    if bytes.len() < OFFSETS_MAGIC.len() + 4 + 4 {
        return Err(bad(0, CorruptionKind::IncompleteFrame));
    }
    if &bytes[..8] != OFFSETS_MAGIC {
        return Err(bad(0, CorruptionKind::BadMagic));
    }
    let (body, tail) = bytes.split_at(bytes.len() - 4);
    let stored = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    if crc32(body) != stored {
        return Err(bad(body.len(), CorruptionKind::ChecksumMismatch));
    }
    let count = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
    if count > MAX_CONSUMERS {
        return Err(bad(8, CorruptionKind::BadLength));
    }
    let mut at = 12;
    let mut out = BTreeMap::new();
    for _ in 0..count {
        let name_len = body
            .get(at..at + 2)
            .map(|s| usize::from(u16::from_le_bytes([s[0], s[1]])))
            .ok_or_else(|| bad(at, CorruptionKind::IncompleteFrame))?;
        at += 2;
        let name = body
            .get(at..at + name_len)
            .and_then(|s| std::str::from_utf8(s).ok())
            .ok_or_else(|| bad(at, CorruptionKind::BadEncoding))?
            .to_owned();
        at += name_len;
        let value = body
            .get(at..at + 8)
            .and_then(|s| <[u8; 8]>::try_from(s).ok())
            .map(u64::from_le_bytes)
            .ok_or_else(|| bad(at, CorruptionKind::IncompleteFrame))?;
        at += 8;
        out.insert(name, Offset(value));
    }
    if at != body.len() {
        return Err(bad(at, CorruptionKind::BadLength));
    }
    Ok(out)
}

/// Test-only failure injection for the durability step of an append.
#[cfg(test)]
#[derive(Debug, Default)]
struct Faults {
    sync: bool,
    truncate: bool,
    /// Make deleting an expired segment fail.
    delete: bool,
}

/// The persistent log: one writer, durable consumer offsets, replay, rollover and retention.
///
/// Authoritative records are never deleted while a registered consumer has not acknowledged
/// them: only whole closed segments entirely at or below the slowest consumer's offset (and
/// beyond the [`RetentionPolicy`] window) are deleted, so the low-water mark advances in
/// segment steps. Publishing past `max_unacked_records` returns [`BusError::Full`].
#[derive(Debug)]
pub struct FileEventLog {
    dir: PathBuf,
    config: FileLogConfig,
    segments: Vec<Segment>,
    active: Option<File>,
    next: u64,
    committed: BTreeMap<String, Offset>,
    tick: u64,
    unsynced: u32,
    /// Set when a failed append could not be rolled back; every publish then fails.
    poisoned: bool,
    compaction_failures: u64,
    last_compaction_error: Option<BusError>,
    #[cfg(test)]
    faults: Faults,
    recovery: RecoveryReport,
}

impl FileEventLog {
    /// Open (creating if needed) the log in `dir` and recover it.
    ///
    /// Recovery validates every frame. An incomplete last frame of the newest segment (a torn
    /// write), or a last frame there whose checksum fails and that reaches the end of the
    /// file, is truncated away and reported in [`FileEventLog::recovery`]. Any other damage is a
    /// typed [`BusError::Corrupt`]; nothing is repaired silently.
    ///
    /// # Errors
    /// `InvalidPolicy` for a refused configuration, `Corrupt`, `OffsetsAheadOfLog`, `Truncated`
    /// (an offset below the low-water mark) or `Io`.
    pub fn open(dir: impl AsRef<Path>, config: FileLogConfig) -> Result<Self, BusError> {
        config.validate()?;
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|e| BusError::io("create log dir", &e))?;
        let _ = fs::remove_file(dir.join(OFFSETS_TEMP));
        let max_body_len = frame_limit(config.max_record_bytes)?;

        let mut files = Vec::new();
        for entry in fs::read_dir(&dir).map_err(|e| BusError::io("list log dir", &e))? {
            let entry = entry.map_err(|e| BusError::io("list log dir", &e))?;
            let name = entry.file_name();
            if let Some(base) = name.to_str().and_then(parse_segment_name) {
                files.push((base, entry.path()));
            }
        }
        files.sort();

        let mut segments = Vec::with_capacity(files.len());
        let mut expected: Option<u64> = None;
        let mut torn_tail = None;
        let mut records = 0u64;
        let mut tick = 0u64;
        for (index, (base, path)) in files.iter().enumerate() {
            let name = segment_name(*base);
            // Offsets start at 1 (`next` is 1 on a fresh log, and a segment is named after the
            // first offset it holds), so a zero base was never written by this log. Refuse it
            // before `first - 1` arithmetic would underflow.
            if *base == 0 {
                return Err(corrupt(
                    &name,
                    0,
                    CorruptionKind::SegmentGap {
                        expected: 1,
                        found: 0,
                    },
                ));
            }
            if let Some(exp) = expected {
                if *base != exp {
                    return Err(corrupt(
                        &name,
                        0,
                        CorruptionKind::SegmentGap {
                            expected: exp,
                            found: *base,
                        },
                    ));
                }
            }
            let file_len = fs::metadata(path)
                .map_err(|e| BusError::io("stat segment", &e))?
                .len();
            let scan = scan_segment(path, &name, *base, file_len, max_body_len)?;
            if let Some(tear) = scan.tear {
                if index + 1 != files.len() {
                    return Err(corrupt_from_tear(&name, scan.valid_bytes, tear));
                }
                let file = OpenOptions::new()
                    .write(true)
                    .open(path)
                    .map_err(|e| BusError::io("open segment for repair", &e))?;
                file.set_len(scan.valid_bytes)
                    .map_err(|e| BusError::io("truncate torn tail", &e))?;
                file.sync_all()
                    .map_err(|e| BusError::io("sync repaired segment", &e))?;
                torn_tail = Some(TornTail {
                    segment: name.clone(),
                    discarded_bytes: file_len - scan.valid_bytes,
                });
            }
            records += scan.count;
            tick = tick.max(scan.newest_tick);
            expected = Some(scan.last.map_or(*base, |last| last + 1));
            segments.push(Segment {
                path: path.clone(),
                first: *base,
                last: scan.last,
                bytes: scan.valid_bytes,
                newest_tick: scan.newest_tick,
            });
        }
        let next = expected.unwrap_or(1);
        let low_water = segments.first().map_or(next - 1, |s| s.first - 1);

        let committed = match fs::metadata(dir.join(OFFSETS_FILE)) {
            Ok(meta) => {
                if meta.len() > MAX_OFFSETS_BYTES {
                    return Err(corrupt(OFFSETS_FILE, 0, CorruptionKind::BadLength));
                }
                let bytes = fs::read(dir.join(OFFSETS_FILE))
                    .map_err(|e| BusError::io("read offsets", &e))?;
                decode_offsets(&bytes)?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(BusError::io("stat offsets", &e)),
        };
        for (consumer, offset) in &committed {
            if offset.0 >= next {
                return Err(BusError::OffsetsAheadOfLog {
                    consumer: consumer.clone(),
                    committed: *offset,
                    end: Offset(next - 1),
                });
            }
            if offset.0 < low_water {
                return Err(BusError::Truncated {
                    requested: *offset,
                    low_water: Offset(low_water),
                });
            }
        }
        let recovery = RecoveryReport {
            segments: segments.len(),
            records,
            torn_tail,
        };
        Ok(Self {
            dir,
            config,
            segments,
            active: None,
            next,
            committed,
            tick,
            unsynced: 0,
            poisoned: false,
            compaction_failures: 0,
            last_compaction_error: None,
            #[cfg(test)]
            faults: Faults::default(),
            recovery,
        })
    }

    /// What the last `open` found, including a truncated torn tail.
    pub fn recovery(&self) -> &RecoveryReport {
        &self.recovery
    }

    /// The log directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Segment files currently on disk.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// The current logical tick (restored from the newest record on open).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Number of retention passes run after a durable consumer commit that failed (for example a
    /// segment could not be deleted). Such a failure never fails the commit; retention is
    /// retried on later commits and publishes.
    pub fn compaction_failures(&self) -> u64 {
        self.compaction_failures
    }

    /// The most recent post-commit retention failure, cleared by the next successful pass.
    pub fn last_compaction_error(&self) -> Option<&BusError> {
        self.last_compaction_error.as_ref()
    }

    /// Retention after the triggering change is already durable: record a failure instead of
    /// returning it.
    fn compact_best_effort(&mut self) {
        match self.compact() {
            Ok(_) => self.last_compaction_error = None,
            Err(error) => {
                self.compaction_failures += 1;
                self.last_compaction_error = Some(error);
            }
        }
    }

    /// Force appended data to stable storage regardless of the fsync policy.
    ///
    /// # Errors
    /// `Io` if the sync fails.
    pub fn sync(&mut self) -> Result<(), BusError> {
        if let Some(file) = &self.active {
            file.sync_data()
                .map_err(|e| BusError::io("sync segment", &e))?;
        }
        self.unsynced = 0;
        Ok(())
    }

    /// Move the logical clock forward and apply age retention.
    ///
    /// # Errors
    /// `ClockRegression` if `now` is behind the current tick; `Io` if deleting a segment fails.
    pub fn advance_tick(&mut self, now: u64) -> Result<Compaction, BusError> {
        if now < self.tick {
            return Err(BusError::ClockRegression {
                current: self.tick,
                requested: now,
            });
        }
        self.tick = now;
        self.compact()
    }

    /// Delete closed segments that are fully acknowledged and outside the retention window,
    /// oldest first, and advance the low-water mark.
    ///
    /// # Errors
    /// `Io` if a segment cannot be deleted.
    pub fn compact(&mut self) -> Result<Compaction, BusError> {
        let floor = self.floor().0;
        let retention = self.config.retention;
        let mut removed_records = 0u64;
        let mut removed_segments = 0usize;
        while self.segments.len() > 1 {
            let Some(last) = self.segments[0].last else {
                break;
            };
            if last > floor {
                break;
            }
            let history = floor - (self.segments[0].first - 1);
            let history_bytes: u64 = self
                .segments
                .iter()
                .take_while(|s| s.last.is_some_and(|l| l <= floor))
                .map(|s| s.bytes)
                .sum();
            let age = self.tick.saturating_sub(self.segments[0].newest_tick);
            let expired = retention.max_records.is_some_and(|m| history > m as u64)
                || retention
                    .max_bytes
                    .is_some_and(|m| history_bytes > m as u64)
                || retention.max_age_ticks.is_some_and(|m| age > m);
            if !expired {
                break;
            }
            #[cfg(test)]
            if self.faults.delete {
                return Err(BusError::io(
                    "delete segment",
                    &std::io::Error::other("injected delete failure"),
                ));
            }
            fs::remove_file(&self.segments[0].path)
                .map_err(|e| BusError::io("delete segment", &e))?;
            let gone = self.segments.remove(0);
            removed_records += last - gone.first + 1;
            removed_segments += 1;
        }
        if removed_segments > 0 && self.config.fsync != FsyncPolicy::Never {
            sync_dir(&self.dir);
        }
        Ok(Compaction {
            removed_records,
            removed_segments,
            low_water: self.low_water(),
        })
    }

    fn end_offset(&self) -> Offset {
        Offset(self.next - 1)
    }

    fn floor(&self) -> Offset {
        self.committed
            .values()
            .min()
            .copied()
            .unwrap_or_else(|| self.end_offset())
    }

    fn committed_of(&self, consumer: &str) -> Result<Offset, BusError> {
        self.committed
            .get(consumer)
            .copied()
            .ok_or_else(|| BusError::UnknownConsumer {
                consumer: consumer.to_owned(),
            })
    }

    fn persist_offsets(&self) -> Result<(), BusError> {
        write_atomic(
            &self.dir,
            &encode_offsets(&self.committed),
            self.config.fsync != FsyncPolicy::Never,
        )
    }

    fn start_segment(&mut self, first: u64) -> Result<(), BusError> {
        if let Some(old) = self.active.take() {
            if self.config.fsync != FsyncPolicy::Never {
                old.sync_data()
                    .map_err(|e| BusError::io("sync closed segment", &e))?;
            }
        }
        let path = self.dir.join(segment_name(first));
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
            .map_err(|e| BusError::io("create segment", &e))?;
        if self.config.fsync != FsyncPolicy::Never {
            sync_dir(&self.dir);
        }
        self.segments.push(Segment {
            path,
            first,
            last: None,
            bytes: 0,
            newest_tick: self.tick,
        });
        self.active = Some(file);
        Ok(())
    }

    fn append(&mut self, frame: &[u8], offset: u64) -> Result<(), BusError> {
        let roll = self.segments.last().is_none_or(|s| {
            s.last.is_some()
                && s.bytes.saturating_add(frame.len() as u64) > self.config.segment_max_bytes
        });
        if roll {
            self.start_segment(offset)?;
        } else if self.active.is_none() {
            let path = self.segments.last().map(|s| s.path.clone());
            if let Some(path) = path {
                let file = OpenOptions::new()
                    .append(true)
                    .open(path)
                    .map_err(|e| BusError::io("open segment for append", &e))?;
                self.active = Some(file);
            }
        }
        let good_len = self.segments.last().map_or(0, |s| s.bytes);
        let Some(file) = self.active.as_mut() else {
            return Err(BusError::Io {
                operation: "active segment",
                kind: std::io::ErrorKind::NotFound,
            });
        };
        if let Err(error) = file.write_all(frame) {
            // Do not leave a partial frame in front of the next append.
            self.discard_tail(good_len);
            return Err(BusError::io("append record", &error));
        }
        if let Err(error) = self.sync_after_write() {
            // The frame is complete on disk but the publish fails: take it back, otherwise a
            // retry would append the same offset twice and a reopen would resurrect a record
            // whose publish returned an error.
            self.discard_tail(good_len);
            return Err(BusError::io("sync record", &error));
        }
        if let Some(segment) = self.segments.last_mut() {
            segment.last = Some(offset);
            segment.bytes += frame.len() as u64;
            segment.newest_tick = self.tick;
        }
        Ok(())
    }

    /// Cut the active segment back to `good_len` after a failed append. If the cut itself
    /// fails the on-disk tail is unknown, so the writer is poisoned until the log is reopened
    /// (recovery then decides what is on disk).
    fn discard_tail(&mut self, good_len: u64) {
        #[cfg(test)]
        let forced_failure = self.faults.truncate;
        #[cfg(not(test))]
        let forced_failure = false;
        // The cut goes through a separate write handle on the segment path, not through
        // `self.active`: that handle is opened append-only, and on Windows an append-only handle
        // lacks FILE_WRITE_DATA, so `set_len` (SetEndOfFile) on it fails and the writer would be
        // poisoned although the cut is possible.
        let cut = !forced_failure
            && self.segments.last().is_some_and(|segment| {
                OpenOptions::new()
                    .write(true)
                    .open(&segment.path)
                    .and_then(|file| file.set_len(good_len))
                    .is_ok()
            });
        if !cut {
            self.poisoned = true;
        }
        self.active = None;
    }

    /// Apply the fsync policy to the frame just written.
    fn sync_after_write(&mut self) -> std::io::Result<()> {
        let due = match self.config.fsync {
            FsyncPolicy::Always => true,
            FsyncPolicy::EveryRecords(every) => {
                self.unsynced += 1;
                self.unsynced >= every.get()
            }
            FsyncPolicy::Never => false,
        };
        if !due {
            return Ok(());
        }
        #[cfg(test)]
        if self.faults.sync {
            return Err(std::io::Error::other("injected sync failure"));
        }
        if let Some(file) = self.active.as_ref() {
            file.sync_data()?;
        }
        self.unsynced = 0;
        Ok(())
    }

    fn read_after(&self, after: u64, max: usize) -> Result<Vec<BusRecord>, BusError> {
        let mut out = Vec::new();
        if max == 0 {
            return Ok(out);
        }
        let max_body_len = frame_limit(self.config.max_record_bytes)?;
        for segment in &self.segments {
            if segment.last.is_none_or(|last| last <= after) {
                continue;
            }
            let name = segment.name();
            let file = File::open(&segment.path).map_err(|e| BusError::io("open segment", &e))?;
            let mut reader = BufReader::new(file);
            let mut pos = 0u64;
            loop {
                match read_frame(&mut reader, pos, segment.bytes, max_body_len, &name)? {
                    Frame::End => break,
                    Frame::Torn(tear) => return Err(corrupt_from_tear(&name, pos, tear)),
                    Frame::Record { record, end_pos } => {
                        pos = end_pos;
                        if record.offset.0 > after {
                            out.push(record);
                            if out.len() >= max {
                                return Ok(out);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

impl EventProducer for FileEventLog {
    /// Append a record durably (per the fsync policy) and return its offset.
    ///
    /// # Errors
    /// `FieldTooLong`, `Full` when `max_unacked_records` records are unacknowledged (nothing is
    /// dropped; `blocked_by` names the slowest consumer), or `Io`. A refused publish consumes
    /// no offset.
    fn publish(&mut self, record: NewRecord) -> Result<Offset, BusError> {
        if self.poisoned {
            return Err(BusError::WriterPoisoned);
        }
        validate_new(&record, self.config.max_record_bytes)?;
        self.compact()?;
        let depth = usize::try_from(self.end_offset().0 - self.floor().0).unwrap_or(usize::MAX);
        if depth >= self.config.max_unacked_records {
            return Err(BusError::Full {
                limit: Limit::Records,
                capacity: self.config.max_unacked_records,
                depth,
                blocked_by: self.blocked_by().map(|lag| lag.consumer),
            });
        }
        let offset = self.next;
        // Offset `u64::MAX` is not recoverable (see `scan_segment`), so the offset space ends
        // one short of it; refuse before writing anything.
        let Some(next) = offset.checked_add(1).filter(|n| *n != u64::MAX) else {
            return Err(BusError::Full {
                limit: Limit::Records,
                capacity: self.config.max_unacked_records,
                depth,
                blocked_by: None,
            });
        };
        let frame = encode_frame(Offset(offset), self.tick, &record);
        self.append(&frame, offset)?;
        self.next = next;
        Ok(Offset(offset))
    }
}

impl EventConsumer for FileEventLog {
    /// Register a consumer at `from` and persist it.
    fn register(&mut self, consumer: &str, from: Offset) -> Result<(), BusError> {
        let end = self.end_offset();
        if from > end {
            return Err(BusError::CommitBeyondEnd {
                requested: from,
                end,
            });
        }
        let low_water = self.low_water();
        if from < low_water {
            return Err(BusError::Truncated {
                requested: from,
                low_water,
            });
        }
        validate_consumer(&self.committed, consumer)?;
        let previous = self.committed.insert(consumer.to_owned(), from);
        if let Err(error) = self.persist_offsets() {
            match previous {
                Some(offset) => self.committed.insert(consumer.to_owned(), offset),
                None => self.committed.remove(consumer),
            };
            return Err(error);
        }
        Ok(())
    }

    fn poll(&self, consumer: &str, max: usize) -> Result<Vec<BusRecord>, BusError> {
        let after = self.committed_of(consumer)?;
        self.read_after(after.0, max)
    }

    /// Acknowledge through `upto`; the new offset is on stable storage (atomic file replace)
    /// before this returns, and only then are fully acknowledged segments eligible for deletion.
    fn commit(&mut self, consumer: &str, upto: Offset) -> Result<(), BusError> {
        let end = self.end_offset();
        if upto > end {
            return Err(BusError::CommitBeyondEnd {
                requested: upto,
                end,
            });
        }
        let previous = self.committed_of(consumer)?;
        if upto > previous {
            self.committed.insert(consumer.to_owned(), upto);
            if let Err(error) = self.persist_offsets() {
                self.committed.insert(consumer.to_owned(), previous);
                return Err(error);
            }
        }
        // The offset is durable: a retention I/O error must not turn it into a failed commit
        // (a reopen would resume after `upto` although the caller was told the commit failed).
        self.compact_best_effort();
        Ok(())
    }

    fn committed(&self, consumer: &str) -> Result<Offset, BusError> {
        self.committed_of(consumer)
    }

    fn lag(&self, consumer: &str) -> Result<u64, BusError> {
        Ok(self.end_offset().0 - self.committed_of(consumer)?.0)
    }

    fn end(&self) -> Offset {
        self.end_offset()
    }

    fn low_water(&self) -> Offset {
        Offset(self.segments.first().map_or(self.next - 1, |s| s.first - 1))
    }

    fn replay(&self, after: Offset, max: usize) -> Result<Vec<BusRecord>, BusError> {
        let end = self.end_offset();
        if after > end {
            return Err(BusError::CommitBeyondEnd {
                requested: after,
                end,
            });
        }
        let low_water = self.low_water();
        if after < low_water {
            return Err(BusError::Truncated {
                requested: after,
                low_water,
            });
        }
        self.read_after(after.0, max)
    }

    fn blocked_by(&self) -> Option<ConsumerLag> {
        slowest(&self.committed, self.end_offset())
    }

    /// The file log never drops records, so this is `0` for every registered consumer.
    fn dropped(&self, consumer: &str) -> Result<u64, BusError> {
        self.committed_of(consumer)?;
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::{check_len, Field};

    #[test]
    fn segment_names_round_trip_and_reject_lookalikes() {
        assert_eq!(parse_segment_name(&segment_name(42)), Some(42));
        assert_eq!(parse_segment_name("segment-42.log"), None);
        assert_eq!(parse_segment_name("offsets.bin"), None);
    }

    #[test]
    fn offsets_file_round_trips_and_detects_damage() {
        let mut map = BTreeMap::new();
        map.insert("a".to_owned(), Offset(3));
        map.insert("b".to_owned(), Offset(0));
        let bytes = encode_offsets(&map);
        assert_eq!(decode_offsets(&bytes).unwrap(), map);
        let mut flipped = bytes.clone();
        flipped[13] ^= 1;
        assert!(matches!(
            decode_offsets(&flipped),
            Err(BusError::Corrupt {
                kind: CorruptionKind::ChecksumMismatch,
                ..
            })
        ));
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(matches!(
            decode_offsets(&magic),
            Err(BusError::Corrupt {
                kind: CorruptionKind::BadMagic,
                ..
            })
        ));
        assert!(decode_offsets(&bytes[..6]).is_err());
    }

    #[test]
    fn config_refuses_zero_bounds_and_telemetry_dropping() {
        let mut zero = FileLogConfig::new(0);
        assert_eq!(
            zero.validate(),
            Err(BusError::InvalidPolicy {
                violation: PolicyViolation::ZeroBound
            })
        );
        zero.max_unacked_records = 1;
        zero.policies = ClassPolicies::drop_oldest_telemetry();
        assert_eq!(
            zero.validate(),
            Err(BusError::InvalidPolicy {
                violation: PolicyViolation::DropUnsupportedByBackend
            })
        );
    }

    #[test]
    fn name_length_is_checked_by_the_shared_validator() {
        assert!(check_len(Field::ConsumerName, MAX_FIELD_BYTES + 1, MAX_FIELD_BYTES).is_err());
    }

    fn unit_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ptr-events-unit-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn rec(key: &str) -> crate::bus::NewRecord {
        crate::bus::NewRecord {
            class: crate::bus::EventClass::Projection,
            topic: "t".into(),
            key: key.into(),
            payload: key.as_bytes().to_vec(),
        }
    }

    #[test]
    fn a_failed_sync_rolls_the_frame_back_so_retry_and_reopen_stay_consistent() {
        let dir = unit_dir("sync-fail");
        let mut log = FileEventLog::open(&dir, FileLogConfig::new(8)).unwrap();
        assert_eq!(log.publish(rec("a")).unwrap(), Offset(1));
        let len_before = fs::metadata(&log.segments[0].path).unwrap().len();

        log.faults.sync = true;
        assert!(matches!(
            log.publish(rec("b")),
            Err(BusError::Io {
                operation: "sync record",
                ..
            })
        ));
        assert_eq!(
            fs::metadata(&log.segments[0].path).unwrap().len(),
            len_before
        );

        log.faults.sync = false;
        assert_eq!(log.publish(rec("b")).unwrap(), Offset(2));
        drop(log);

        let reopened = FileEventLog::open(&dir, FileLogConfig::new(8)).unwrap();
        let keys: Vec<String> = reopened
            .read_after(0, 10)
            .unwrap()
            .into_iter()
            .map(|r| r.key)
            .collect();
        assert_eq!(keys, ["a", "b"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_sync_that_cannot_be_rolled_back_poisons_the_writer_until_reopen() {
        let dir = unit_dir("poison");
        let mut log = FileEventLog::open(&dir, FileLogConfig::new(8)).unwrap();
        log.publish(rec("a")).unwrap();
        log.faults.sync = true;
        log.faults.truncate = true;
        assert!(log.publish(rec("b")).is_err());
        log.faults.sync = false;
        log.faults.truncate = false;
        assert_eq!(log.publish(rec("c")), Err(BusError::WriterPoisoned));
        assert_eq!(log.publish(rec("c")), Err(BusError::WriterPoisoned));
        drop(log);
        // Recovery decides what is on disk; the log is usable again.
        let mut reopened = FileEventLog::open(&dir, FileLogConfig::new(8)).unwrap();
        let next = reopened.publish(rec("c")).unwrap();
        assert!(next.0 >= 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_refuses_age_only_retention() {
        let mut config = FileLogConfig::new(1);
        config.retention = RetentionPolicy::with_max_age_ticks(10);
        assert_eq!(
            config.validate(),
            Err(BusError::InvalidPolicy {
                violation: PolicyViolation::UnboundedRetention
            })
        );
    }

    #[test]
    fn config_refuses_unbounded_retention() {
        let mut config = FileLogConfig::new(1);
        config.retention = RetentionPolicy {
            max_records: None,
            max_bytes: None,
            max_age_ticks: None,
        };
        assert_eq!(
            config.validate(),
            Err(BusError::InvalidPolicy {
                violation: PolicyViolation::UnboundedRetention
            })
        );
    }

    fn retention_one_record_config() -> FileLogConfig {
        let mut config = FileLogConfig::new(8);
        config.segment_max_bytes = 1;
        config.retention = RetentionPolicy::with_max_records(0);
        config
    }

    #[test]
    fn a_retention_failure_after_a_durable_commit_does_not_fail_the_commit() {
        let dir = unit_dir("retention-fail");
        let mut log = FileEventLog::open(&dir, retention_one_record_config()).unwrap();
        log.register("c", Offset(0)).unwrap();
        for key in ["a", "b", "c"] {
            log.publish(rec(key)).unwrap();
        }
        assert!(log.segment_count() >= 3);

        log.faults.delete = true;
        assert_eq!(log.commit("c", Offset(3)), Ok(()));
        assert_eq!(log.committed("c"), Ok(Offset(3)));
        assert_eq!(log.compaction_failures(), 1);
        assert!(matches!(
            log.last_compaction_error(),
            Some(BusError::Io {
                operation: "delete segment",
                ..
            })
        ));
        drop(log);
        // The commit really is durable: a reopen resumes after it.
        let mut reopened = FileEventLog::open(&dir, retention_one_record_config()).unwrap();
        assert_eq!(reopened.committed("c"), Ok(Offset(3)));

        // Retention is retried on a later commit and the error clears.
        reopened.publish(rec("d")).unwrap();
        reopened.commit("c", Offset(4)).unwrap();
        assert_eq!(reopened.compaction_failures(), 0);
        assert!(reopened.last_compaction_error().is_none());
        assert!(reopened.low_water() > Offset(0));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_retention_failure_is_counted_and_cleared_by_a_later_success() {
        let dir = unit_dir("retention-retry");
        let mut log = FileEventLog::open(&dir, retention_one_record_config()).unwrap();
        log.register("c", Offset(0)).unwrap();
        for key in ["a", "b", "c"] {
            log.publish(rec(key)).unwrap();
        }
        log.faults.delete = true;
        log.commit("c", Offset(3)).unwrap();
        log.commit("c", Offset(3)).unwrap();
        assert_eq!(log.compaction_failures(), 2);
        log.faults.delete = false;
        log.commit("c", Offset(3)).unwrap();
        assert_eq!(log.compaction_failures(), 2);
        assert!(log.last_compaction_error().is_none());
        assert!(log.low_water() > Offset(0));
        let _ = fs::remove_dir_all(&dir);
    }

    fn write_segment_with_record_at(dir: &Path, offset: u64) {
        fs::create_dir_all(dir).unwrap();
        let frame = encode_frame(Offset(offset), 0, &rec("x"));
        fs::write(dir.join(segment_name(offset)), frame).unwrap();
    }

    #[test]
    fn a_record_at_the_last_representable_offset_is_corruption_not_an_overflow() {
        let dir = unit_dir("max-offset");
        write_segment_with_record_at(&dir, u64::MAX);
        let result = FileEventLog::open(&dir, FileLogConfig::new(8));
        assert!(
            matches!(result, Err(BusError::Corrupt { .. })),
            "expected typed corruption, got {:?}",
            result.map(|_| ())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn publishing_at_the_last_representable_offset_is_refused_without_writing() {
        let dir = unit_dir("max-offset-publish");
        write_segment_with_record_at(&dir, u64::MAX - 1);
        let mut log = FileEventLog::open(&dir, FileLogConfig::new(8)).unwrap();
        let before = fs::metadata(&log.segments[0].path).unwrap().len();
        assert!(matches!(log.publish(rec("y")), Err(BusError::Full { .. })));
        assert_eq!(fs::metadata(&log.segments[0].path).unwrap().len(), before);
        drop(log);
        assert!(FileEventLog::open(&dir, FileLogConfig::new(8)).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_limit_that_overflows_the_frame_is_refused() {
        for max_record_bytes in [usize::MAX, usize::MAX - 100, u32::MAX as usize] {
            let mut config = FileLogConfig::new(1);
            config.max_record_bytes = max_record_bytes;
            assert_eq!(
                config.validate(),
                Err(BusError::InvalidPolicy {
                    violation: PolicyViolation::RecordLimitTooLarge
                })
            );
        }
        let mut config = FileLogConfig::new(1);
        config.max_record_bytes = crate::codec::MAX_FRAME_BODY - 2 * MAX_FIELD_BYTES - MIN_BODY;
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(
            max_body(config.max_record_bytes),
            Some(crate::codec::MAX_FRAME_BODY)
        );
        assert_eq!(max_body(config.max_record_bytes + 1), None);
    }
}
