//! Routing telemetry: what was decided, never the secret that sampled it.
//!
//! A [`RoutingRecord`] carries the decision id (the request id), the nonce, the non-secret key
//! identifier, the propensity, the chosen option and the constraints outcome. It has no field
//! that could hold the [`crate::SamplingKey`] or the keyed draw (Invariant 13), so no sink can
//! write them. Sinks are bounded: [`MemorySink`] by record count, [`JsonLinesSink`] by file
//! size, and both refuse further records with a typed error instead of growing.

use crate::{CheckedCandidate, ConstraintsOutcome};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Longest decision id or chosen-option text a record accepts, in bytes.
pub const MAX_FIELD_BYTES: usize = 256;

/// One routing decision as telemetry.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingRecord {
    decision_id: String,
    nonce: [u8; 16],
    key_id: [u8; 8],
    propensity: f64,
    chosen: String,
    constraints: ConstraintsOutcome,
}

impl RoutingRecord {
    /// The record of a checked candidate. `chosen` is the option the caller finally selected
    /// (for example the Pod id or the model backend id).
    pub fn from_candidate(
        candidate: &CheckedCandidate,
        chosen: &str,
        constraints: ConstraintsOutcome,
    ) -> Result<Self, TelemetryError> {
        let decision_id = candidate.request_id().to_string();
        if decision_id.len() > MAX_FIELD_BYTES {
            return Err(TelemetryError::FieldTooLong {
                field: "decision_id",
            });
        }
        if chosen.len() > MAX_FIELD_BYTES {
            return Err(TelemetryError::FieldTooLong { field: "chosen" });
        }
        Ok(Self {
            decision_id,
            nonce: *candidate.nonce().as_bytes(),
            key_id: candidate.key_id(),
            propensity: candidate.propensity().get(),
            chosen: chosen.to_owned(),
            constraints,
        })
    }

    pub fn decision_id(&self) -> &str {
        &self.decision_id
    }
    pub fn nonce(&self) -> [u8; 16] {
        self.nonce
    }
    pub fn key_id(&self) -> [u8; 8] {
        self.key_id
    }
    pub fn propensity(&self) -> f64 {
        self.propensity
    }
    pub fn chosen(&self) -> &str {
        &self.chosen
    }
    pub fn constraints(&self) -> ConstraintsOutcome {
        self.constraints
    }

    /// One JSON object without a trailing newline. Hand-written; strings are escaped.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(192);
        out.push_str("{\"v\":1,\"decision_id\":");
        push_json_string(&mut out, &self.decision_id);
        out.push_str(",\"nonce\":\"");
        push_hex(&mut out, &self.nonce);
        out.push_str("\",\"key_id\":\"");
        push_hex(&mut out, &self.key_id);
        out.push_str("\",\"propensity\":");
        // Propensity is validated finite, in (0, 1].
        let _ = write!(out, "{:?}", self.propensity);
        out.push_str(",\"chosen\":");
        push_json_string(&mut out, &self.chosen);
        out.push_str(",\"constraints\":");
        match self.constraints {
            ConstraintsOutcome::NotApplied => out.push_str("{\"kind\":\"not_applied\"}"),
            ConstraintsOutcome::Satisfied { eligible, feasible } => {
                let _ = write!(
                    out,
                    "{{\"kind\":\"satisfied\",\"eligible\":{eligible},\"feasible\":{feasible}}}"
                );
            }
            ConstraintsOutcome::Infeasible { options } => {
                let _ = write!(out, "{{\"kind\":\"infeasible\",\"options\":{options}}}");
            }
        }
        out.push('}');
        out
    }
}

fn push_hex(out: &mut String, bytes: &[u8]) {
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
}

fn push_json_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Why a record was not accepted or written.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TelemetryError {
    FieldTooLong {
        field: &'static str,
    },
    /// The sink is at its bound; the record was dropped.
    Full,
    Io {
        kind: ErrorKind,
    },
    /// The sink can no longer be trusted: its lock was poisoned, or a failed write could not be
    /// rolled back and the file may end in a partial line. Later records are refused.
    Poisoned,
}

impl std::fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FieldTooLong { field } => write!(f, "telemetry field {field} is too long"),
            Self::Full => write!(f, "telemetry sink is full"),
            Self::Io { kind } => write!(f, "telemetry io error: {kind}"),
            Self::Poisoned => write!(f, "telemetry sink is poisoned"),
        }
    }
}

impl std::error::Error for TelemetryError {}

/// A destination for [`RoutingRecord`]s. Routing must not depend on it: callers may ignore
/// the error.
pub trait RoutingTelemetry {
    fn record(&self, record: &RoutingRecord) -> Result<(), TelemetryError>;
}

/// Keeps the first `capacity` records and rejects the rest with [`TelemetryError::Full`],
/// counting them.
#[derive(Debug)]
pub struct MemorySink {
    capacity: usize,
    state: Mutex<MemoryState>,
}

#[derive(Debug, Default)]
struct MemoryState {
    records: Vec<RoutingRecord>,
    rejected: u64,
}

impl MemorySink {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(MemoryState::default()),
        }
    }

    pub fn records(&self) -> Vec<RoutingRecord> {
        self.state
            .lock()
            .map(|state| state.records.clone())
            .unwrap_or_default()
    }

    /// Records refused because the sink was full.
    pub fn rejected(&self) -> u64 {
        self.state.lock().map(|state| state.rejected).unwrap_or(0)
    }
}

impl RoutingTelemetry for MemorySink {
    fn record(&self, record: &RoutingRecord) -> Result<(), TelemetryError> {
        let mut state = self.state.lock().map_err(|_| TelemetryError::Poisoned)?;
        if state.records.len() >= self.capacity {
            state.rejected = state.rejected.saturating_add(1);
            return Err(TelemetryError::Full);
        }
        state.records.push(record.clone());
        Ok(())
    }
}

/// Appends one JSON line per record to a file and stops at a byte bound. It never rewrites
/// existing content; the bound counts the bytes already in the file at open. A write that
/// fails part-way is rolled back to the length before the record, so the file only ever ends
/// on a whole line and the bound holds; if that rollback fails the sink is poisoned.
#[derive(Debug)]
pub struct JsonLinesSink {
    max_bytes: u64,
    path: PathBuf,
    state: Mutex<FileState>,
}

#[derive(Debug)]
struct FileState {
    file: File,
    written: u64,
    poisoned: bool,
    refused: u64,
    #[cfg(test)]
    faults: tests::Faults,
}

impl JsonLinesSink {
    pub fn open(path: &Path, max_bytes: u64) -> Result<Self, TelemetryError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| TelemetryError::Io { kind: error.kind() })?;
        let written = file
            .metadata()
            .map_err(|error| TelemetryError::Io { kind: error.kind() })?
            .len();
        Ok(Self {
            max_bytes,
            path: path.to_path_buf(),
            state: Mutex::new(FileState {
                file,
                written,
                poisoned: false,
                refused: 0,
                #[cfg(test)]
                faults: tests::Faults::default(),
            }),
        })
    }

    /// Records refused because the sink is poisoned by an unrecoverable write failure.
    pub fn refused(&self) -> u64 {
        self.state.lock().map(|state| state.refused).unwrap_or(0)
    }
}

impl FileState {
    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        #[cfg(test)]
        if let Some(fault) = self.faults.write.take() {
            return fault.apply(&mut self.file, bytes);
        }
        self.file.write_all(bytes)?;
        self.file.flush()
    }

    /// Cut the file back to `len` through a separate handle. The append handle must not be
    /// used: on Windows an append-only handle lacks FILE_WRITE_DATA, so `set_len` on it fails.
    fn truncate_to(&self, path: &Path, len: u64) -> io::Result<()> {
        #[cfg(test)]
        if self.faults.truncate_fails {
            return Err(io::Error::from(ErrorKind::PermissionDenied));
        }
        OpenOptions::new().write(true).open(path)?.set_len(len)
    }
}

impl RoutingTelemetry for JsonLinesSink {
    fn record(&self, record: &RoutingRecord) -> Result<(), TelemetryError> {
        let mut line = record.to_json();
        line.push('\n');
        // The lock is held across measure, write and rollback so clones sharing the sink or
        // other threads cannot truncate each other's successful writes.
        let mut state = self.state.lock().map_err(|_| TelemetryError::Poisoned)?;
        if state.poisoned {
            state.refused = state.refused.saturating_add(1);
            return Err(TelemetryError::Poisoned);
        }
        let before = state
            .file
            .metadata()
            .map_err(|error| TelemetryError::Io { kind: error.kind() })?
            .len();
        state.written = before;
        let len = line.len() as u64;
        if before.saturating_add(len) > self.max_bytes {
            return Err(TelemetryError::Full);
        }
        match state.append(line.as_bytes()) {
            Ok(()) => {
                state.written = before + len;
                Ok(())
            }
            Err(error) => {
                if state.truncate_to(&self.path, before).is_err() {
                    state.poisoned = true;
                } else {
                    state.written = state
                        .file
                        .metadata()
                        .map(|meta| meta.len())
                        .unwrap_or(before);
                }
                Err(TelemetryError::Io { kind: error.kind() })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fault injected into the next append of a [`JsonLinesSink`].
    #[derive(Clone, Copy, Debug)]
    pub(super) enum WriteFault {
        /// Write this many bytes of the line, then fail like a full disk.
        Partial(usize),
        /// Write the whole line, then fail at flush.
        Flush,
    }

    impl WriteFault {
        pub(super) fn apply(self, file: &mut File, bytes: &[u8]) -> io::Result<()> {
            match self {
                Self::Partial(n) => {
                    file.write_all(&bytes[..n.min(bytes.len())])?;
                    Err(io::Error::from(ErrorKind::StorageFull))
                }
                Self::Flush => {
                    file.write_all(bytes)?;
                    Err(io::Error::from(ErrorKind::Other))
                }
            }
        }
    }

    #[derive(Debug, Default)]
    pub(super) struct Faults {
        pub(super) write: Option<WriteFault>,
        pub(super) truncate_fails: bool,
    }

    #[test]
    fn json_strings_are_escaped() {
        let mut out = String::new();
        push_json_string(&mut out, "a\"b\\c\nd\u{1}é");
        assert_eq!(out, "\"a\\\"b\\\\c\\nd\\u0001é\"");
    }

    fn temp_file(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ptr-router-telemetry-unit-{}-{tag}.jsonl",
            std::process::id()
        ))
    }

    fn sample(id: &str) -> RoutingRecord {
        RoutingRecord {
            decision_id: id.to_string(),
            nonce: [3; 16],
            key_id: [4; 8],
            propensity: 1.0,
            chosen: "zeta".to_string(),
            constraints: ConstraintsOutcome::NotApplied,
        }
    }

    fn set_faults(sink: &JsonLinesSink, write: Option<WriteFault>, truncate_fails: bool) {
        sink.state.lock().unwrap().faults = Faults {
            write,
            truncate_fails,
        };
    }

    fn assert_file_matches(sink: &JsonLinesSink, path: &Path, expected: &[u8]) {
        assert_eq!(std::fs::read(path).unwrap(), expected);
        assert_eq!(sink.state.lock().unwrap().written, expected.len() as u64);
    }

    fn well_formed(bytes: &[u8]) -> bool {
        bytes.is_empty()
            || (bytes.ends_with(b"\n")
                && bytes
                    .split(|b| *b == b'\n')
                    .filter(|l| !l.is_empty())
                    .all(|l| l.starts_with(b"{\"v\":1") && l.ends_with(b"}")))
    }

    fn run_rolled_back(tag: &str, fault: WriteFault) {
        let path = temp_file(tag);
        let _ = std::fs::remove_file(&path);
        let first = sample("a");
        let one = (first.to_json().len() + 1) as u64;
        let sink = JsonLinesSink::open(&path, one * 3).unwrap();
        sink.record(&first).unwrap();
        let before = std::fs::read(&path).unwrap();
        for _ in 0..5 {
            set_faults(&sink, Some(fault), false);
            assert!(matches!(
                sink.record(&sample("b")),
                Err(TelemetryError::Io { .. })
            ));
            assert_file_matches(&sink, &path, &before);
        }
        sink.record(&sample("c")).unwrap();
        let after = std::fs::read(&path).unwrap();
        assert!(well_formed(&after));
        assert!(after.len() as u64 <= one * 3);
        assert_eq!(after.iter().filter(|b| **b == b'\n').count(), 2);
        assert_eq!(sink.state.lock().unwrap().written, after.len() as u64);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn partial_write_is_rolled_back() {
        run_rolled_back("partial", WriteFault::Partial(40));
    }

    #[test]
    fn flush_failure_is_rolled_back() {
        run_rolled_back("flush", WriteFault::Flush);
    }

    #[test]
    fn failed_truncate_poisons_the_sink() {
        let path = temp_file("poison");
        let _ = std::fs::remove_file(&path);
        let sink = JsonLinesSink::open(&path, 10_000).unwrap();
        sink.record(&sample("a")).unwrap();
        set_faults(&sink, Some(WriteFault::Partial(40)), true);
        assert!(matches!(
            sink.record(&sample("b")),
            Err(TelemetryError::Io { .. })
        ));
        let frozen = std::fs::read(&path).unwrap();
        set_faults(&sink, None, false);
        assert_eq!(sink.record(&sample("c")), Err(TelemetryError::Poisoned));
        assert_eq!(sink.record(&sample("d")), Err(TelemetryError::Poisoned));
        assert_eq!(sink.refused(), 2);
        assert_eq!(std::fs::read(&path).unwrap(), frozen);
        std::fs::remove_file(&path).unwrap();
    }
}
