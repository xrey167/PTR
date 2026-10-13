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
use std::io::{ErrorKind, Write};
use std::path::Path;
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
    Poisoned,
}

impl std::fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FieldTooLong { field } => write!(f, "telemetry field {field} is too long"),
            Self::Full => write!(f, "telemetry sink is full"),
            Self::Io { kind } => write!(f, "telemetry io error: {kind}"),
            Self::Poisoned => write!(f, "telemetry sink lock is poisoned"),
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

/// Appends one JSON line per record to a file and stops at a byte bound. It never truncates or
/// rewrites existing content; the bound counts the bytes already in the file at open.
#[derive(Debug)]
pub struct JsonLinesSink {
    max_bytes: u64,
    state: Mutex<FileState>,
}

#[derive(Debug)]
struct FileState {
    file: File,
    written: u64,
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
            state: Mutex::new(FileState { file, written }),
        })
    }
}

impl RoutingTelemetry for JsonLinesSink {
    fn record(&self, record: &RoutingRecord) -> Result<(), TelemetryError> {
        let mut line = record.to_json();
        line.push('\n');
        let mut state = self.state.lock().map_err(|_| TelemetryError::Poisoned)?;
        let len = line.len() as u64;
        if state.written.saturating_add(len) > self.max_bytes {
            return Err(TelemetryError::Full);
        }
        state
            .file
            .write_all(line.as_bytes())
            .map_err(|error| TelemetryError::Io { kind: error.kind() })?;
        state.written += len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_strings_are_escaped() {
        let mut out = String::new();
        push_json_string(&mut out, "a\"b\\c\nd\u{1}é");
        assert_eq!(out, "\"a\\\"b\\\\c\\nd\\u0001é\"");
    }
}
