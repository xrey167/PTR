//! An append-only JSON Lines file of [`TrajectoryRecord`]s with size-based rotation.
//!
//! Each [`TrajectoryLog::append`] writes one complete line. When the next line would push a
//! non-empty file past `max_bytes`, the files are rotated first (`log` becomes `log.1`, `log.1`
//! becomes `log.2`, up to `keep` files; the oldest is dropped). Every step of the rotation is a
//! rename, so a reader never sees a half-moved file; a crash between renames can leave a gap in
//! the numbering but never a truncated or merged record. A line torn by a crash is detected by
//! the reader and, in [`ReadMode::SkipCorrupt`], skipped.

use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::record::{RecordError, TrajectoryRecord, MAX_LINE_BYTES};

/// Smallest accepted rotation size.
pub const MIN_ROTATION_BYTES: u64 = 512;
/// Most rotated files kept.
pub const MAX_KEEP: u32 = 64;

/// Why a log operation failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogError {
    /// `max_bytes` is below [`MIN_ROTATION_BYTES`] or `keep` is zero or above [`MAX_KEEP`].
    InvalidConfig,
    /// The path has no file name.
    InvalidPath,
    /// A filesystem operation failed.
    Io(io::ErrorKind),
    /// The record could not be encoded.
    Record(RecordError),
    /// Strict reading met a bad line (1-based line number).
    Corrupt { line: usize, error: RecordError },
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid log configuration"),
            Self::InvalidPath => f.write_str("the log path has no file name"),
            Self::Io(kind) => write!(f, "log I/O failed: {kind}"),
            Self::Record(error) => write!(f, "record error: {error}"),
            Self::Corrupt { line, error } => write!(f, "line {line} is corrupt: {error}"),
        }
    }
}

impl Error for LogError {}

impl From<io::Error> for LogError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}

/// Where the log lives and when it rotates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogConfig {
    pub path: PathBuf,
    /// Rotate before a line would take a non-empty file past this size.
    pub max_bytes: u64,
    /// How many rotated files (`.1` to `.keep`) are kept.
    pub keep: u32,
}

/// What an append did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendReport {
    pub rotated: bool,
    /// Bytes written, newline included.
    pub bytes: usize,
}

/// How [`read_log`] treats a line that does not decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadMode {
    /// Stop at the first bad line with [`LogError::Corrupt`].
    Strict,
    /// Skip bad lines and report them.
    SkipCorrupt,
}

/// The records of one file and the lines that were skipped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReadOutcome {
    pub records: Vec<TrajectoryRecord>,
    /// 1-based line numbers and reasons.
    pub skipped: Vec<(usize, RecordError)>,
}

/// The append side of a trajectory log.
#[derive(Clone, Debug)]
pub struct TrajectoryLog {
    config: LogConfig,
}

/// Path of the `n`-th rotated file of `base`: `base.n`.
pub fn rotated_path(base: &Path, n: u32) -> PathBuf {
    let mut name = base.as_os_str().to_owned();
    name.push(format!(".{n}"));
    PathBuf::from(name)
}

impl TrajectoryLog {
    /// # Errors
    /// [`LogError::InvalidConfig`] or [`LogError::InvalidPath`].
    pub fn open(config: LogConfig) -> Result<Self, LogError> {
        if config.max_bytes < MIN_ROTATION_BYTES || config.keep == 0 || config.keep > MAX_KEEP {
            return Err(LogError::InvalidConfig);
        }
        if config.path.file_name().is_none() {
            return Err(LogError::InvalidPath);
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &LogConfig {
        &self.config
    }

    /// Encode (and so redact and bound) `record` and append it as one line.
    ///
    /// # Errors
    /// [`LogError::Record`] when the record does not encode, [`LogError::Io`] otherwise.
    pub fn append(&self, record: &TrajectoryRecord) -> Result<AppendReport, LogError> {
        let mut line = record.encode().map_err(LogError::Record)?;
        line.push('\n');
        let path = &self.config.path;
        let size = match fs::metadata(path) {
            Ok(meta) => meta.len(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        let rotated = size > 0 && size + line.len() as u64 > self.config.max_bytes;
        if rotated {
            self.rotate()?;
        }
        let mut payload = Vec::with_capacity(line.len() + 1);
        if !rotated && size > 0 && !ends_with_newline(path)? {
            // A previous write was torn; start this record on its own line.
            payload.push(b'\n');
        }
        payload.extend_from_slice(line.as_bytes());
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(&payload)?;
        file.flush()?;
        file.sync_data()?;
        Ok(AppendReport {
            rotated,
            bytes: line.len(),
        })
    }

    fn rotate(&self) -> Result<(), LogError> {
        let base = &self.config.path;
        let keep = self.config.keep;
        remove_if_present(&rotated_path(base, keep))?;
        for n in (1..keep).rev() {
            let from = rotated_path(base, n);
            if from.exists() {
                fs::rename(&from, rotated_path(base, n + 1))?;
            }
        }
        fs::rename(base, rotated_path(base, 1))?;
        Ok(())
    }
}

fn remove_if_present(path: &Path) -> Result<(), LogError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn ends_with_newline(path: &Path) -> Result<bool, LogError> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::End(-1))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)?;
    Ok(last[0] == b'\n')
}

enum RawLine {
    Text(Vec<u8>),
    TooLong,
}

/// Read one line of at most [`MAX_LINE_BYTES`] without ever buffering more than that.
fn read_bounded_line(reader: &mut impl BufRead) -> Result<Option<RawLine>, LogError> {
    let mut line: Vec<u8> = Vec::new();
    let mut too_long = false;
    let mut any = false;
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            break;
        }
        any = true;
        let newline = chunk.iter().position(|byte| *byte == b'\n');
        let take = newline.unwrap_or(chunk.len());
        if !too_long {
            if line.len() + take > MAX_LINE_BYTES {
                too_long = true;
                line.clear();
            } else {
                line.extend_from_slice(&chunk[..take]);
            }
        }
        let consumed = newline.map_or(chunk.len(), |at| at + 1);
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }
    if !any {
        return Ok(None);
    }
    Ok(Some(if too_long {
        RawLine::TooLong
    } else {
        RawLine::Text(line)
    }))
}

/// Read every record of one log file. Blank lines are ignored.
///
/// # Errors
/// [`LogError::Io`] when the file cannot be read; in [`ReadMode::Strict`] also
/// [`LogError::Corrupt`] at the first bad line.
pub fn read_log(path: &Path, mode: ReadMode) -> Result<ReadOutcome, LogError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut outcome = ReadOutcome::default();
    let mut number = 0usize;
    while let Some(raw) = read_bounded_line(&mut reader)? {
        number += 1;
        let decoded = match raw {
            RawLine::TooLong => Err(RecordError::LineTooLong),
            RawLine::Text(bytes) => match std::str::from_utf8(&bytes) {
                Err(_) => Err(RecordError::NotUtf8),
                Ok(text) if text.trim().is_empty() => continue,
                Ok(text) => TrajectoryRecord::decode(text),
            },
        };
        match decoded {
            Ok(record) => outcome.records.push(record),
            Err(error) => match mode {
                ReadMode::Strict => {
                    return Err(LogError::Corrupt {
                        line: number,
                        error,
                    })
                }
                ReadMode::SkipCorrupt => outcome.skipped.push((number, error)),
            },
        }
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{OutcomeRecord, StepRecord, UsageRecord};
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("ptr-feedback-log-unit-{}-{n}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn record(id: &str) -> TrajectoryRecord {
        TrajectoryRecord {
            run_id: id.into(),
            outcome: OutcomeRecord::Verified {
                candidate: "c0".into(),
            },
            usage: UsageRecord::default(),
            steps: vec![StepRecord {
                candidate: "c0".into(),
                origin: "initial".into(),
                from: None,
                rewrite_reason: None,
                text: "t".into(),
                text_truncated: false,
                status: "pass".into(),
                level: "deterministic".into(),
                score: 1.0,
                issues: vec![],
                repairable: None,
                decision: "accepted".into(),
                stop_reason: None,
            }],
        }
    }

    fn config(dir: &TempDir, max_bytes: u64, keep: u32) -> LogConfig {
        LogConfig {
            path: dir.0.join("traj.jsonl"),
            max_bytes,
            keep,
        }
    }

    #[test]
    fn configuration_is_validated() {
        let dir = TempDir::new();
        assert_eq!(
            TrajectoryLog::open(config(&dir, 10, 1)).unwrap_err(),
            LogError::InvalidConfig
        );
        assert_eq!(
            TrajectoryLog::open(config(&dir, 4096, 0)).unwrap_err(),
            LogError::InvalidConfig
        );
        let no_name = LogConfig {
            path: PathBuf::from("/"),
            max_bytes: 4096,
            keep: 1,
        };
        assert_eq!(
            TrajectoryLog::open(no_name).unwrap_err(),
            LogError::InvalidPath
        );
    }

    #[test]
    fn appended_records_read_back_in_order() {
        let dir = TempDir::new();
        let log = TrajectoryLog::open(config(&dir, 1 << 20, 2)).unwrap();
        for id in ["a", "b", "c"] {
            let report = log.append(&record(id)).unwrap();
            assert!(!report.rotated);
        }
        let outcome = read_log(&log.config().path, ReadMode::Strict).unwrap();
        let ids: Vec<&str> = outcome.records.iter().map(|r| r.run_id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    #[test]
    fn rotation_keeps_the_newest_files_and_drops_the_oldest() {
        let dir = TempDir::new();
        let line_len = record("r0").encode().unwrap().len() as u64 + 1;
        let log = TrajectoryLog::open(config(&dir, line_len * 2 + 10, 2)).unwrap();
        let mut rotations = 0;
        for i in 0..9 {
            if log.append(&record(&format!("r{i}"))).unwrap().rotated {
                rotations += 1;
            }
        }
        assert_eq!(rotations, 4);
        let ids = |path: &Path| -> Vec<String> {
            read_log(path, ReadMode::Strict)
                .unwrap()
                .records
                .into_iter()
                .map(|r| r.run_id)
                .collect()
        };
        let base = &log.config().path;
        assert_eq!(ids(base), ["r8"]);
        assert_eq!(ids(&rotated_path(base, 1)), ["r6", "r7"]);
        assert_eq!(ids(&rotated_path(base, 2)), ["r4", "r5"]);
        assert!(!rotated_path(base, 3).exists(), "beyond keep is dropped");
    }

    #[test]
    fn corrupt_and_torn_lines_are_typed_and_skippable() {
        let dir = TempDir::new();
        let log = TrajectoryLog::open(config(&dir, 1 << 20, 1)).unwrap();
        log.append(&record("good1")).unwrap();
        let path = log.config().path.clone();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"{\"v\":1,\"run_id\":\"torn").unwrap();
        file.write_all(b"\n\xff\xfe not utf8\n").unwrap();
        drop(file);
        log.append(&record("good2")).unwrap();
        match read_log(&path, ReadMode::Strict).unwrap_err() {
            LogError::Corrupt { line, .. } => assert_eq!(line, 2),
            other => panic!("unexpected {other:?}"),
        }
        let outcome = read_log(&path, ReadMode::SkipCorrupt).unwrap();
        assert_eq!(outcome.records.len(), 2);
        assert_eq!(outcome.skipped.len(), 2);
        assert_eq!(outcome.skipped[0].0, 2);
        assert!(matches!(outcome.skipped[0].1, RecordError::Syntax { .. }));
        assert_eq!(outcome.skipped[1], (3, RecordError::NotUtf8));
    }

    #[test]
    fn an_append_after_a_torn_final_line_starts_a_fresh_line() {
        let dir = TempDir::new();
        let log = TrajectoryLog::open(config(&dir, 1 << 20, 1)).unwrap();
        let path = log.config().path.clone();
        fs::write(&path, b"{\"v\":1,\"run_id\":\"torn").unwrap();
        log.append(&record("after")).unwrap();
        let outcome = read_log(&path, ReadMode::SkipCorrupt).unwrap();
        assert_eq!(outcome.records.len(), 1);
        assert_eq!(outcome.records[0].run_id, "after");
        assert_eq!(outcome.skipped.len(), 1);
    }

    #[test]
    fn an_oversized_line_is_reported_without_being_buffered() {
        let dir = TempDir::new();
        let path = dir.0.join("big.jsonl");
        let mut bytes = vec![b'a'; MAX_LINE_BYTES + 10];
        bytes.push(b'\n');
        fs::write(&path, &bytes).unwrap();
        let outcome = read_log(&path, ReadMode::SkipCorrupt).unwrap();
        assert_eq!(outcome.skipped, [(1, RecordError::LineTooLong)]);
        assert_eq!(
            read_log(&path, ReadMode::Strict).unwrap_err(),
            LogError::Corrupt {
                line: 1,
                error: RecordError::LineTooLong
            }
        );
    }

    #[test]
    fn a_missing_file_is_an_io_error() {
        let dir = TempDir::new();
        assert_eq!(
            read_log(&dir.0.join("none"), ReadMode::Strict).unwrap_err(),
            LogError::Io(io::ErrorKind::NotFound)
        );
    }
}
