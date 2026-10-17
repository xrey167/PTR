//! An append-only JSON Lines file of [`TrajectoryRecord`]s with size-based rotation.
//!
//! Each [`TrajectoryLog::append`] writes one complete line. When the next line would push a
//! non-empty file past `max_bytes`, the files are rotated first (`log` becomes `log.1`, `log.1`
//! becomes `log.2`, up to `keep` files; the oldest is dropped). Every step of the rotation is a
//! rename, so a reader never sees a half-moved file. The oldest file is first staged
//! (`log.rotating`) rather than deleted, and a rename that fails is undone in reverse, so a failed
//! rotation leaves the files it found (an undo that fails too poisons the log). A crash, as opposed
//! to a failed rename, between renames can still leave a gap in the numbering or a stale
//! `log.rotating`, but never a truncated or merged record. A line torn by a crash is detected by
//! the reader and, in [`ReadMode::SkipCorrupt`], skipped.

use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::record::{RecordError, TrajectoryRecord, MAX_LINE_BYTES};

/// Smallest accepted rotation size.
pub const MIN_ROTATION_BYTES: u64 = 512;
/// Largest accepted rotation size (1 GiB). A larger limit would never trigger rotation in
/// practice, so the log would grow without a retained bound; [`TrajectoryLog::open`] refuses it.
pub const MAX_ROTATION_BYTES: u64 = 1 << 30;
/// Most rotated files kept.
pub const MAX_KEEP: u32 = 64;

/// Why a log operation failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogError {
    /// `max_bytes` is outside [`MIN_ROTATION_BYTES`]..=[`MAX_ROTATION_BYTES`] or `keep` is zero or above [`MAX_KEEP`].
    InvalidConfig,
    /// The path has no file name.
    InvalidPath,
    /// A filesystem operation failed.
    Io(io::ErrorKind),
    /// The record could not be encoded.
    Record(RecordError),
    /// A failed append could not be rolled back, so the file tail is unknown; the log refuses
    /// appends until it is reopened (and the file inspected with [`read_log`]).
    Poisoned,
    /// Strict reading met a bad line (1-based line number).
    Corrupt { line: usize, error: RecordError },
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid log configuration"),
            Self::InvalidPath => f.write_str("the log path has no file name"),
            Self::Io(kind) => write!(f, "log I/O failed: {kind}"),
            Self::Poisoned => {
                f.write_str("the log is poisoned by an append that could not be undone")
            }
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
    /// Rotate before a line would take a non-empty file past this size; from
    /// [`MIN_ROTATION_BYTES`] to [`MAX_ROTATION_BYTES`].
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
///
/// Clones share one lock, so the size check, the rotation and the write of an append are
/// serialized across threads that use clones of the same log. Two logs opened independently for
/// the same path (or in different processes) are not coordinated with each other.
#[derive(Clone, Debug)]
pub struct TrajectoryLog {
    config: LogConfig,
    guard: Arc<Mutex<()>>,
    /// Set (under `guard`) when a failed append could not be rolled back; shared by clones.
    poisoned: Arc<AtomicBool>,
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
        if !(MIN_ROTATION_BYTES..=MAX_ROTATION_BYTES).contains(&config.max_bytes)
            || config.keep == 0
            || config.keep > MAX_KEEP
        {
            return Err(LogError::InvalidConfig);
        }
        if config.path.file_name().is_none() {
            return Err(LogError::InvalidPath);
        }
        Ok(Self {
            config,
            guard: Arc::new(Mutex::new(())),
            poisoned: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn config(&self) -> &LogConfig {
        &self.config
    }

    /// Encode (and so redact and bound) `record` and append it as one line. The size check,
    /// rotation and write run under the lock shared by clones of this log.
    ///
    /// # Errors
    /// [`LogError::Record`] when the record does not encode, [`LogError::Io`] otherwise (the file is
    /// then cut back to its length before the call), [`LogError::Poisoned`] after a failure that
    /// could not be cut back.
    pub fn append(&self, record: &TrajectoryRecord) -> Result<AppendReport, LogError> {
        let mut line = record.encode().map_err(LogError::Record)?;
        line.push('\n');
        let _held: MutexGuard<'_, ()> = self
            .guard
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(LogError::Poisoned);
        }
        let path = &self.config.path;
        let size = match fs::metadata(path) {
            Ok(meta) => meta.len(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error.into()),
        };
        #[cfg(test)]
        test_seam::after_size_check();
        let rotated = size > 0 && size + line.len() as u64 > self.config.max_bytes;
        if rotated {
            self.rotate()?;
        }
        // Length the file must return to if this record cannot be made durable.
        let good_len = if rotated { 0 } else { size };
        let mut payload = Vec::with_capacity(line.len() + 1);
        if !rotated && size > 0 && !ends_with_newline(path)? {
            // A previous write was torn; start this record on its own line.
            payload.push(b'\n');
        }
        payload.extend_from_slice(line.as_bytes());
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        let durable = file
            .write_all(&payload)
            .and_then(|()| file.flush())
            .and_then(|()| sync_file(&file));
        if let Err(error) = durable {
            // The line may be complete on disk although the append reports failure: take it
            // back, otherwise a retry would duplicate the trajectory and a reopen would resurrect
            // a record whose append returned an error. If the cut itself fails the tail is
            // unknown, so the log refuses appends until it is reopened.
            if truncate_path(path, good_len).is_err() {
                self.poisoned.store(true, Ordering::SeqCst);
            }
            return Err(error.into());
        }
        Ok(AppendReport {
            rotated,
            bytes: line.len(),
        })
    }

    /// Shift `base` to `base.1`, `base.n` to `base.n+1`, dropping `base.keep`, failure-atomically.
    ///
    /// The oldest generation is first *staged* (renamed to `base.rotating`), never deleted. The
    /// shifts then only move a file into a slot that was just vacated, so no rename ever
    /// overwrites a retained file. On any failing step the completed moves are undone in reverse
    /// (each undo moves a file back into the slot its own move vacated) and the staged file is
    /// put back, so a failed rotation leaves exactly the files it found and a retry loses
    /// nothing. If an undo itself fails the numbering is unknown and the log is poisoned. Only
    /// after every move succeeded is the staged oldest generation deleted (a failure to delete
    /// it leaves a stray `base.rotating`, which the next rotation replaces).
    fn rotate(&self) -> Result<(), LogError> {
        let base = &self.config.path;
        let keep = self.config.keep;
        let stage = staging_path(base);
        let oldest = rotated_path(base, keep);
        let staged = oldest.exists();
        if staged {
            rename_file(&oldest, &stage)?;
        }
        // (from, to) of every move done so far, in order.
        let mut done: Vec<(PathBuf, PathBuf)> = Vec::new();
        let mut moves: Vec<(PathBuf, PathBuf)> = (1..keep)
            .rev()
            .map(|n| (rotated_path(base, n), rotated_path(base, n + 1)))
            .filter(|(from, _)| from.exists())
            .collect();
        moves.push((base.clone(), rotated_path(base, 1)));
        for (from, to) in moves {
            if let Err(error) = rename_file(&from, &to) {
                let mut restored = true;
                for (undo_to, undo_from) in done.iter().rev() {
                    restored &= rename_file(undo_from, undo_to).is_ok();
                }
                if staged {
                    restored &= rename_file(&stage, &oldest).is_ok();
                }
                if !restored {
                    self.poisoned.store(true, Ordering::SeqCst);
                }
                return Err(error.into());
            }
            done.push((from, to));
        }
        if staged {
            let _ = fs::remove_file(&stage);
        }
        Ok(())
    }
}

/// Where the oldest generation waits while a rotation is in flight: `base.rotating`.
fn staging_path(base: &Path) -> PathBuf {
    let mut name = base.as_os_str().to_owned();
    name.push(".rotating");
    PathBuf::from(name)
}

#[cfg(test)]
mod test_seam {
    use std::cell::{Cell, RefCell};

    type Hook = Box<dyn Fn()>;

    thread_local! {
        static AFTER_SIZE_CHECK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    pub(super) fn set(hook: Hook) {
        AFTER_SIZE_CHECK.with(|slot| *slot.borrow_mut() = Some(hook));
    }

    thread_local! {
        static FAIL_SYNC: Cell<bool> = const { Cell::new(false) };
        static FAIL_TRUNCATE: Cell<bool> = const { Cell::new(false) };
        static FAIL_ROLLBACK_SYNC: Cell<bool> = const { Cell::new(false) };
        static FAIL_RENAME: Cell<Option<(usize, bool)>> = const { Cell::new(None) };
        static RENAME_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    /// Fail the rename numbered `at` (0-based, counted from this call); with `persist`, every
    /// later rename fails too (so an undo cannot succeed either).
    pub(super) fn fail_rename_at(at: Option<usize>, persist: bool) {
        RENAME_COUNT.with(|count| count.set(0));
        FAIL_RENAME.with(|slot| slot.set(at.map(|at| (at, persist))));
    }

    pub(super) fn rename_fails() -> bool {
        let number = RENAME_COUNT.with(|count| {
            let number = count.get();
            count.set(number + 1);
            number
        });
        FAIL_RENAME
            .with(Cell::get)
            .is_some_and(|(at, persist)| number == at || (persist && number > at))
    }

    pub(super) fn fail_sync(on: bool) {
        FAIL_SYNC.with(|flag| flag.set(on));
    }

    pub(super) fn fail_truncate(on: bool) {
        FAIL_TRUNCATE.with(|flag| flag.set(on));
    }

    pub(super) fn fail_rollback_sync(on: bool) {
        FAIL_ROLLBACK_SYNC.with(|flag| flag.set(on));
    }

    pub(super) fn rollback_sync_fails() -> bool {
        FAIL_ROLLBACK_SYNC.with(Cell::get)
    }

    pub(super) fn sync_fails() -> bool {
        FAIL_SYNC.with(Cell::get)
    }

    pub(super) fn truncate_fails() -> bool {
        FAIL_TRUNCATE.with(Cell::get)
    }

    pub(super) fn after_size_check() {
        AFTER_SIZE_CHECK.with(|slot| {
            if let Some(hook) = slot.borrow().as_ref() {
                hook();
            }
        });
    }
}

fn sync_file(file: &File) -> io::Result<()> {
    #[cfg(test)]
    if test_seam::sync_fails() {
        return Err(io::Error::other("injected sync failure"));
    }
    file.sync_data()
}

/// Cut `path` to `len` through a fresh write handle. The append handle cannot do it: on Windows
/// an append-only handle lacks `FILE_WRITE_DATA`, so `set_len` on it fails and the rollback would
/// never happen.
fn truncate_path(path: &Path, len: u64) -> io::Result<()> {
    #[cfg(test)]
    if test_seam::truncate_fails() {
        return Err(io::Error::other("injected truncate failure"));
    }
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(len)?;
    // Without this a crash could restore the line the failed append already reported as lost.
    #[cfg(test)]
    if test_seam::rollback_sync_fails() {
        return Err(io::Error::other("injected rollback sync failure"));
    }
    file.sync_data()
}

fn rename_file(from: &Path, to: &Path) -> io::Result<()> {
    #[cfg(test)]
    if test_seam::rename_fails() {
        return Err(io::Error::other("injected rename failure"));
    }
    fs::rename(from, to)
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
    fn rotation_size_has_a_hard_maximum() {
        let dir = TempDir::new();
        for max_bytes in [u64::MAX, MAX_ROTATION_BYTES + 1] {
            assert_eq!(
                TrajectoryLog::open(config(&dir, max_bytes, 1)).unwrap_err(),
                LogError::InvalidConfig
            );
        }
        assert!(TrajectoryLog::open(config(&dir, MAX_ROTATION_BYTES, 1)).is_ok());
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

    fn ids_of(path: &Path) -> Vec<String> {
        read_log(path, ReadMode::Strict)
            .unwrap()
            .records
            .into_iter()
            .map(|r| r.run_id)
            .collect()
    }

    fn long_record(id: &str) -> TrajectoryRecord {
        record(&format!("{}{id}", "x".repeat(200)))
    }

    /// A log whose base and `keep` rotated files all exist, one record each (`g0` newest in
    /// `.1`, `g{keep-1}` oldest in `.keep`; the base holds `base`). The next append rotates.
    fn full_log(dir: &TempDir, keep: u32) -> (TrajectoryLog, Vec<String>) {
        let log = TrajectoryLog::open(config(dir, MIN_ROTATION_BYTES, keep)).unwrap();
        let base = log.config().path.clone();
        let mut all = Vec::new();
        let line = |id: &str| {
            let mut text = long_record(id).encode().unwrap();
            text.push('\n');
            text
        };
        fs::write(&base, line("base")).unwrap();
        all.push(long_record("base").run_id);
        for n in 1..=keep {
            let id = format!("g{}", n - 1);
            fs::write(rotated_path(&base, n), line(&id)).unwrap();
            all.push(long_record(&id).run_id);
        }
        (log, all)
    }

    fn all_ids(log: &TrajectoryLog) -> Vec<String> {
        let base = &log.config().path;
        let mut ids = Vec::new();
        for path in std::iter::once(base.clone())
            .chain((1..=MAX_KEEP).map(|n| rotated_path(base, n)))
            .filter(|path| path.exists())
        {
            ids.extend(ids_of(&path));
        }
        ids
    }

    #[test]
    fn a_failed_rename_at_any_step_loses_nothing_and_a_retry_rotates_cleanly() {
        for keep in 1..=3u32 {
            for step in 0..=(keep as usize + 2) {
                let dir = TempDir::new();
                let (log, before) = full_log(&dir, keep);
                let base = log.config().path.clone();
                test_seam::fail_rename_at(Some(step), false);
                let result = log.append(&long_record("new"));
                test_seam::fail_rename_at(None, false);
                if result.is_ok() {
                    continue; // fewer renames than `step`: nothing was injected
                }
                assert!(
                    matches!(result, Err(LogError::Io(_))),
                    "keep {keep} step {step}"
                );
                let mut after = all_ids(&log);
                after.sort();
                let mut expected = before.clone();
                expected.sort();
                assert_eq!(
                    after, expected,
                    "failed append lost data: keep {keep} step {step}"
                );
                assert!(ids_of(&base).contains(&long_record("base").run_id));

                log.append(&long_record("new")).unwrap();
                let ids = all_ids(&log);
                assert_eq!(ids.first(), Some(&long_record("new").run_id));
                assert_eq!(ids.len(), keep as usize + 1, "keep {keep} step {step}");
                // newest generations survive; only the oldest one is dropped by the rotation
                for kept in before.iter().take(keep as usize) {
                    assert!(ids.contains(kept), "lost {kept}: keep {keep} step {step}");
                }
            }
        }
    }

    #[test]
    fn a_rotation_that_cannot_be_undone_poisons_the_log() {
        let dir = TempDir::new();
        let (log, _) = full_log(&dir, 3);
        let clone = log.clone();
        // Fail the second rename and every one after it, so the undo fails too.
        test_seam::fail_rename_at(Some(2), true);
        let failed = log.append(&long_record("new"));
        test_seam::fail_rename_at(None, false);
        assert!(matches!(failed, Err(LogError::Io(_))));
        assert_eq!(log.append(&long_record("again")), Err(LogError::Poisoned));
        assert_eq!(clone.append(&long_record("again")), Err(LogError::Poisoned));
    }

    #[test]
    fn a_failed_sync_rolls_the_line_back_so_a_retry_does_not_duplicate_it() {
        let dir = TempDir::new();
        let log = TrajectoryLog::open(config(&dir, 1 << 20, 2)).unwrap();
        log.append(&record("a")).unwrap();
        let before = fs::read(&log.config().path).unwrap();

        test_seam::fail_sync(true);
        let failed = log.append(&record("b"));
        test_seam::fail_sync(false);
        assert!(matches!(failed, Err(LogError::Io(_))));
        assert_eq!(
            fs::read(&log.config().path).unwrap(),
            before,
            "tail cut back"
        );

        log.append(&record("b")).unwrap();
        assert_eq!(ids_of(&log.config().path), ["a", "b"]);
    }

    #[test]
    fn a_failed_sync_after_rotation_leaves_the_new_file_empty() {
        let dir = TempDir::new();
        let line_len = record("r0").encode().unwrap().len() as u64 + 1;
        let log =
            TrajectoryLog::open(config(&dir, MIN_ROTATION_BYTES.max(line_len + 1), 1)).unwrap();
        let big = |id: &str| {
            let mut r = record(id);
            r.run_id = format!("{}{id}", "x".repeat(200));
            r
        };
        log.append(&big("a")).unwrap();
        test_seam::fail_sync(true);
        let failed = log.append(&big("b"));
        test_seam::fail_sync(false);
        assert!(matches!(failed, Err(LogError::Io(_))));
        let base = &log.config().path;
        assert_eq!(fs::metadata(base).unwrap().len(), 0);
        log.append(&big("b")).unwrap();
        assert_eq!(ids_of(base), [format!("{}b", "x".repeat(200))]);
        assert_eq!(
            ids_of(&rotated_path(base, 1)),
            [format!("{}a", "x".repeat(200))]
        );
    }

    #[test]
    fn a_failure_that_cannot_be_rolled_back_poisons_the_log_until_reopen() {
        let dir = TempDir::new();
        let log = TrajectoryLog::open(config(&dir, 1 << 20, 2)).unwrap();
        let clone = log.clone();
        log.append(&record("a")).unwrap();

        test_seam::fail_sync(true);
        test_seam::fail_truncate(true);
        let failed = log.append(&record("b"));
        test_seam::fail_sync(false);
        test_seam::fail_truncate(false);
        assert!(matches!(failed, Err(LogError::Io(_))));

        assert_eq!(log.append(&record("c")), Err(LogError::Poisoned));
        assert_eq!(clone.append(&record("c")), Err(LogError::Poisoned));
        assert!(!LogError::Poisoned.to_string().is_empty());

        // Reopening clears the poison; the file then holds what the failed append left.
        let reopened = TrajectoryLog::open(log.config().clone()).unwrap();
        reopened.append(&record("c")).unwrap();
        let ids = ids_of(&reopened.config().path);
        assert_eq!(ids.first().map(String::as_str), Some("a"));
        assert_eq!(ids.last().map(String::as_str), Some("c"));
    }

    #[test]
    fn a_rollback_whose_sync_fails_poisons_the_log() {
        let dir = TempDir::new();
        let log = TrajectoryLog::open(config(&dir, 1 << 20, 2)).unwrap();
        log.append(&record("a")).unwrap();

        test_seam::fail_sync(true);
        test_seam::fail_rollback_sync(true);
        let failed = log.append(&record("b"));
        test_seam::fail_sync(false);
        test_seam::fail_rollback_sync(false);
        assert!(matches!(failed, Err(LogError::Io(_))));
        assert_eq!(log.append(&record("c")), Err(LogError::Poisoned));
    }

    #[test]
    fn clones_serialize_the_size_check_rotation_and_append() {
        use std::sync::mpsc;
        use std::time::Duration;

        // Long ids so that a valid (>= MIN_ROTATION_BYTES) limit still holds only one line.
        let id = |tag: &str| format!("{}{tag}", "x".repeat(200));
        let dir = TempDir::new();
        let line_len = record(&id("r0")).encode().unwrap().len() as u64 + 1;
        assert!(line_len > MIN_ROTATION_BYTES / 2, "two lines must not fit");
        let max_bytes = (line_len + 10).max(MIN_ROTATION_BYTES);
        let log = TrajectoryLog::open(config(&dir, max_bytes, 1)).unwrap();
        log.append(&record(&id("r0"))).unwrap();

        let (at_check_tx, at_check_rx) = mpsc::channel::<()>();
        let (b_done_tx, b_done_rx) = mpsc::channel::<()>();
        let a_log = log.clone();
        let a = std::thread::spawn(move || {
            // Pause A between its size check and its rotation until B has finished (or, when B
            // is correctly held back by the shared lock, until the wait times out).
            test_seam::set(Box::new(move || {
                let _ = at_check_tx.send(());
                let _ = b_done_rx.recv_timeout(Duration::from_millis(300));
            }));
            a_log.append(&record(&id("rA")))
        });
        at_check_rx.recv().unwrap();
        let b_log = log.clone();
        let b = std::thread::spawn(move || {
            let result = b_log.append(&record(&id("rB")));
            let _ = b_done_tx.send(());
            result
        });
        let a_report = a.join().unwrap().unwrap();
        let b_report = b.join().unwrap().unwrap();
        assert!(a_report.rotated && b_report.rotated);

        let ids = |path: &Path| -> Vec<String> {
            read_log(path, ReadMode::Strict)
                .unwrap()
                .records
                .into_iter()
                .map(|r| r.run_id)
                .collect()
        };
        let base = &log.config().path;
        // A ran first (rotated r0 away), then B rotated A's record to .1.
        assert_eq!(ids(base), [id("rB")]);
        assert_eq!(ids(&rotated_path(base, 1)), [id("rA")]);
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
