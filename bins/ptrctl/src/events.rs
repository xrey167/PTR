//! `ptrctl events`: list a `ptr_events::FileEventLog` without changing it.
//!
//! `FileEventLog::open` creates the directory, deletes a stale `offsets.bin.tmp` and truncates
//! a torn tail, and the crate has no non-mutating open. So the listing runs over a snapshot:
//! the segment files and `offsets.bin` are copied into a private temp directory (bounded
//! total size, regular files only) and the log is opened there. Nothing in the source
//! directory is created, truncated or rewritten. If opening the snapshot finds a torn tail,
//! the command refuses instead of listing a log whose on-disk state a writer would repair.
//! The snapshot is opened with the default record bound; a log written with a larger
//! `max_record_bytes` is reported as corrupt.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use ptr_events::{
    BusError, BusRecord, EventConsumer, FileEventLog, FileLogConfig, FsyncPolicy, Offset,
};

use crate::cli::{
    flag_number, flag_value, is_help, usage_error, EXIT_CODES_HELP, EXIT_INPUT, EXIT_OK,
};

const USAGE: &str = "usage: ptrctl events <dir> [--consumer NAME] [--from OFFSET] [--limit N]";
const DEFAULT_LIMIT: usize = 100;
const MAX_LIMIT: usize = 10_000;
/// Largest total size of the files copied into the snapshot.
const MAX_SNAPSHOT_BYTES: u64 = 512 * 1024 * 1024;
const OFFSETS_FILE: &str = "offsets.bin";

fn help() -> String {
    format!(
        "{USAGE}

List the records of a file event log: offset, class, authority, topic and payload length,
then the consumer offsets and lag. Read-only: the log is opened on a snapshot copy in a
temporary directory, and a log with a torn tail is refused rather than repaired.

options:
  --consumer NAME  also print that consumer's committed offset and lag
  --from OFFSET    list records after OFFSET (default: the low-water mark, i.e. all retained)
  --limit N        list at most N records (default {DEFAULT_LIMIT}, at most {MAX_LIMIT})

{EXIT_CODES_HELP}"
    )
}

struct Request {
    dir: PathBuf,
    consumer: Option<String>,
    from: Option<u64>,
    limit: usize,
}

pub(crate) fn command(args: Vec<String>) -> i32 {
    if args.first().is_some_and(|arg| is_help(arg)) {
        println!("{}", help());
        return EXIT_OK;
    }
    let request = match parse(args) {
        Ok(request) => request,
        Err(code) => return code,
    };
    match list(&request) {
        Ok(text) => {
            print!("{text}");
            EXIT_OK
        }
        Err(message) => {
            eprintln!("ptrctl events: {message}");
            EXIT_INPUT
        }
    }
}

fn parse(args: Vec<String>) -> Result<Request, i32> {
    let mut dir = None;
    let mut consumer = None;
    let mut from = None;
    let mut limit = DEFAULT_LIMIT;
    let mut rest = args.into_iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--consumer" => {
                consumer = Some(flag_value(&mut rest, "events", "--consumer", USAGE)?);
            }
            "--from" => from = Some(flag_number(&mut rest, "events", "--from", USAGE)?),
            "--limit" => limit = flag_number(&mut rest, "events", "--limit", USAGE)?,
            flag if flag.starts_with("--") => {
                return Err(usage_error(
                    "events",
                    &format!("unknown option {flag}"),
                    USAGE,
                ))
            }
            _ => {
                if dir.replace(PathBuf::from(&arg)).is_some() {
                    return Err(usage_error(
                        "events",
                        "only one directory is accepted",
                        USAGE,
                    ));
                }
            }
        }
    }
    if limit > MAX_LIMIT {
        return Err(usage_error(
            "events",
            &format!("--limit is at most {MAX_LIMIT}"),
            USAGE,
        ));
    }
    match dir {
        Some(dir) => Ok(Request {
            dir,
            consumer,
            from,
            limit,
        }),
        None => Err(usage_error("events", "a log directory is required", USAGE)),
    }
}

/// A private temp directory removed on drop.
struct Snapshot(PathBuf);

impl Drop for Snapshot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn is_log_file(name: &str) -> bool {
    name == OFFSETS_FILE || (name.starts_with("segment-") && name.ends_with(".log"))
}

/// Copy the log files of `source` into a fresh temp directory.
fn snapshot(source: &Path) -> Result<Snapshot, String> {
    snapshot_with(source, || ())
}

/// `snapshot`, running `between` after the sizes are counted and before any file is copied.
/// It exists so a test can grow a file in that window, as a live writer could.
fn snapshot_with(source: &Path, between: impl FnOnce()) -> Result<Snapshot, String> {
    let metadata = fs::metadata(source)
        .map_err(|e| format!("cannot read log directory {}: {e}", source.display()))?;
    if !metadata.is_dir() {
        return Err(format!("{} is not a directory", source.display()));
    }
    let mut files = Vec::new();
    let mut total = 0u64;
    let entries = fs::read_dir(source)
        .map_err(|e| format!("cannot list log directory {}: {e}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("cannot list {}: {e}", source.display()))?;
        let name = entry.file_name();
        let Some(name) = name.to_str().filter(|name| is_log_file(name)) else {
            continue;
        };
        let meta =
            fs::symlink_metadata(entry.path()).map_err(|e| format!("cannot stat {name}: {e}"))?;
        if !meta.is_file() {
            continue;
        }
        total = total.saturating_add(meta.len());
        if total > MAX_SNAPSHOT_BYTES {
            return Err(format!(
                "log files exceed the {MAX_SNAPSHOT_BYTES} byte snapshot bound"
            ));
        }
        files.push((name.to_owned(), entry.path(), meta.len()));
    }
    if files.is_empty() {
        return Err(format!(
            "{} holds no event log files (segment-*.log or {OFFSETS_FILE})",
            source.display()
        ));
    }
    between();
    let target = std::env::temp_dir().join(format!(
        "ptrctl-events-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    private_dir_builder()
        .create(&target)
        .map_err(|e| format!("cannot create snapshot directory: {e}"))?;
    let guard = Snapshot(target);
    for (name, path, counted) in files {
        copy_counted(&path, &guard.0.join(&name), counted)
            .map_err(|e| format!("cannot copy {name} into the snapshot: {e}"))?;
    }
    Ok(guard)
}

/// Copy at most `counted` bytes of `from` (its size when the snapshot bound was checked). A
/// file that has grown since is refused rather than truncated: a prefix cut mid-record would
/// look like a torn tail, and an unbounded copy would exceed the snapshot bound.
fn copy_counted(from: &Path, to: &Path, counted: u64) -> Result<(), String> {
    let source = fs::File::open(from).map_err(|e| e.to_string())?;
    let mut bounded = source.take(counted.saturating_add(1));
    let mut target = private_file_options().open(to).map_err(|e| e.to_string())?;
    let copied = io::copy(&mut bounded, &mut target).map_err(|e| e.to_string())?;
    if copied > counted {
        return Err(format!(
            "the file grew beyond the {counted} bytes counted (is a writer appending?); \
             retry when the log is quiescent"
        ));
    }
    Ok(())
}

/// The snapshot holds full copies of a possibly protected log in the shared temp directory, so
/// on Unix it is created owner-only (0700) regardless of the umask.
fn private_dir_builder() -> fs::DirBuilder {
    // Only the Unix branch below mutates the builder.
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
}

/// Snapshot files are created fresh (never opened through an existing path) and, on Unix,
/// owner-only (0600).
fn private_file_options() -> fs::OpenOptions {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

fn describe(error: &BusError) -> String {
    format!("{error}")
}

fn list(request: &Request) -> Result<String, String> {
    let snapshot = snapshot(&request.dir)?;
    let mut config = FileLogConfig::new(1);
    config.fsync = FsyncPolicy::Never;
    let log = FileEventLog::open(&snapshot.0, config).map_err(|e| {
        format!(
            "log {} is not readable: {}",
            request.dir.display(),
            describe(&e)
        )
    })?;
    if let Some(tear) = &log.recovery().torn_tail {
        return Err(format!(
            "log {} has a torn tail in {} ({} bytes) that opening for write would truncate; \
             refusing to read a log that needs repair",
            request.dir.display(),
            tear.segment,
            tear.discarded_bytes
        ));
    }
    let from = request.from.map_or_else(|| log.low_water(), Offset);
    let records = log
        .replay(from, request.limit)
        .map_err(|e| format!("cannot list from offset {}: {}", from.0, describe(&e)))?;

    let mut out = String::new();
    out.push_str(&format!(
        "log: {}\nsegments: {} records: {} low_water: {} end: {}\n",
        request.dir.display(),
        log.recovery().segments,
        log.recovery().records,
        log.low_water().0,
        log.end().0
    ));
    if let Some(name) = &request.consumer {
        let committed = log.committed(name).map_err(|e| describe(&e))?;
        let lag = log.lag(name).map_err(|e| describe(&e))?;
        out.push_str(&format!(
            "consumer {name}: committed {} lag {lag}\n",
            committed.0
        ));
    }
    match log.blocked_by() {
        Some(slowest) => out.push_str(&format!(
            "slowest consumer {}: committed {} lag {}\n",
            slowest.consumer, slowest.committed.0, slowest.lag
        )),
        None => out.push_str("slowest consumer: none registered\n"),
    }
    out.push_str("offset\tclass\tauthority\ttopic\tpayload_bytes\n");
    for record in &records {
        out.push_str(&row(record));
    }
    out.push_str(&format!("listed {} record(s)\n", records.len()));
    Ok(out)
}

fn row(record: &BusRecord) -> String {
    format!(
        "{}\t{:?}\t{:?}\t{}\t{}\n",
        record.offset.0,
        record.class,
        record.authority,
        record.topic.escape_debug(),
        record.payload.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_that_grows_after_counting_is_refused_and_leaves_no_snapshot() {
        let dir = std::env::temp_dir().join(format!("ptrctl-grow-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("dir");
        let segment = dir.join("segment-00000000000000000000.log");
        fs::write(&segment, b"0123456789").expect("segment");

        let result = snapshot_with(&dir, || {
            use std::io::Write;
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&segment)
                .expect("open");
            file.write_all(b"grown").expect("append");
        });
        let message = result.err().expect("growth must be refused");
        assert!(message.contains("grew"), "{message}");
        assert_eq!(fs::read(&segment).expect("read"), b"0123456789grown");
        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn the_snapshot_directory_and_its_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("ptrctl-mode-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join(OFFSETS_FILE), b"abc").expect("offsets");
        fs::write(dir.join("segment-00000000000000000000.log"), b"payload").expect("segment");
        let copy = snapshot(&dir).expect("snapshot");
        let mode = |path: &Path| fs::metadata(path).expect("stat").permissions().mode() & 0o777;
        assert_eq!(mode(&copy.0), 0o700, "snapshot directory");
        for entry in fs::read_dir(&copy.0).expect("list") {
            let path = entry.expect("entry").path();
            assert_eq!(mode(&path), 0o600, "{}", path.display());
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unchanged_file_is_copied_exactly() {
        let dir = std::env::temp_dir().join(format!("ptrctl-same-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join(OFFSETS_FILE), b"abc").expect("offsets");
        let copy = snapshot(&dir).expect("snapshot");
        assert_eq!(fs::read(copy.0.join(OFFSETS_FILE)).expect("copy"), b"abc");
        let _ = fs::remove_dir_all(&dir);
    }
}
