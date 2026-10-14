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
        files.push((name.to_owned(), entry.path()));
    }
    if files.is_empty() {
        return Err(format!(
            "{} holds no event log files (segment-*.log or {OFFSETS_FILE})",
            source.display()
        ));
    }
    let target = std::env::temp_dir().join(format!(
        "ptrctl-events-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    fs::create_dir(&target).map_err(|e| format!("cannot create snapshot directory: {e}"))?;
    let guard = Snapshot(target);
    for (name, path) in files {
        fs::copy(&path, guard.0.join(&name))
            .map_err(|e| format!("cannot copy {name} into the snapshot: {e}"))?;
    }
    Ok(guard)
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
