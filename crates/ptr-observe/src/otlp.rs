//! OTLP-JSON-format file exporter for finished spans.
//!
//! This writes the *field layout* of the OpenTelemetry OTLP/JSON trace data model
//! (`resourceSpans[].scopeSpans[].spans[]` with `traceId`, `spanId`, `parentSpanId`, `name`,
//! `kind`, `startTimeUnixNano`, `endTimeUnixNano`, `attributes`, `events`, `status`) as one JSON
//! document per line. It is a hand-written file writer, not an OpenTelemetry SDK, collector or
//! OTLP network exporter.
//!
//! Notes on fidelity: 64-bit integers are JSON strings as in proto3 JSON; ids are lowercase hex;
//! the tracer clock is monotonic with an arbitrary origin, so `startTimeUnixNano` is
//! `unix_epoch_offset_ns + tick` and is only wall-clock time if the caller supplies a real offset
//! (default 0). Every span is passed through a [`Redactor`] before it is serialized.

use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::{Redactor, Span, SpanStatus, TraceError, TraceLevel, TraceValue};

#[derive(Clone, Debug)]
pub struct OtlpFileConfig {
    pub service_name: String,
    pub scope_name: String,
    pub unix_epoch_offset_ns: u64,
    pub redactor: Redactor,
}

impl Default for OtlpFileConfig {
    fn default() -> Self {
        Self {
            service_name: "ptr".to_owned(),
            scope_name: "ptr-observe".to_owned(),
            unix_epoch_offset_ns: 0,
            redactor: Redactor::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OtlpJsonFileExporter {
    path: PathBuf,
    config: OtlpFileConfig,
    /// Set when a failed append could not be rolled back; shared by clones, so no clone ever
    /// appends after an unknown tail.
    poisoned: Arc<AtomicBool>,
    /// Serializes open, length measurement, append and rollback across clones, so a rollback
    /// can never truncate to a length measured before another clone's successful append.
    append_lock: Arc<Mutex<()>>,
    #[cfg(test)]
    faults: Arc<Faults>,
}

/// Test-only failure injection for the append step.
#[cfg(test)]
#[derive(Debug, Default)]
struct Faults {
    /// Write only this many bytes of the next line, then fail (consumed on use).
    partial_write: std::sync::Mutex<Option<usize>>,
    /// Make the rollback truncate fail.
    truncate: AtomicBool,
    /// Runs once between measuring the pre-write length and appending (clone interleaving).
    after_measure: std::sync::Mutex<Option<Hook>>,
}

/// Debug-printable boxed closure for the test seam.
#[cfg(test)]
struct Hook(Box<dyn FnOnce() + Send>);

#[cfg(test)]
impl std::fmt::Debug for Hook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Hook")
    }
}

const SINK_NAME: &str = "otlp-json-file";

/// Appends `text` as a JSON string literal (with quotes) to `out`.
pub(crate) fn push_json_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_any_value(out: &mut String, value: &TraceValue) {
    let int = |out: &mut String, n: u64| match i64::try_from(n) {
        Ok(_) => {
            let _ = write!(out, "{{\"intValue\":\"{n}\"}}");
        }
        Err(_) => {
            let _ = write!(out, "{{\"stringValue\":\"{n}\"}}");
        }
    };
    match value {
        TraceValue::String(text) => {
            out.push_str("{\"stringValue\":");
            push_json_string(out, text);
            out.push('}');
        }
        TraceValue::RequestId(id) => {
            out.push_str("{\"stringValue\":");
            push_json_string(out, &id.0);
            out.push('}');
        }
        TraceValue::Bool(flag) => {
            let _ = write!(out, "{{\"boolValue\":{flag}}}");
        }
        TraceValue::U64(n) => int(out, *n),
        TraceValue::Revision(v) => int(out, v.0),
        TraceValue::Generation(v) => int(out, v.0),
        TraceValue::CommitIndex(v) => int(out, v.0),
    }
}

fn push_attributes<'a>(
    out: &mut String,
    attributes: impl Iterator<Item = (&'a str, &'a TraceValue)>,
) {
    out.push('[');
    for (index, (key, value)) in attributes.enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str("{\"key\":");
        push_json_string(out, key);
        out.push_str(",\"value\":");
        push_any_value(out, value);
        out.push('}');
    }
    out.push(']');
}

fn severity(level: TraceLevel) -> &'static str {
    match level {
        TraceLevel::Error => "error",
        TraceLevel::Warn => "warn",
        TraceLevel::Info => "info",
        TraceLevel::Debug => "debug",
        TraceLevel::Trace => "trace",
    }
}

impl OtlpJsonFileExporter {
    pub fn new(path: impl Into<PathBuf>, config: OtlpFileConfig) -> Self {
        Self {
            path: path.into(),
            config,
            poisoned: Arc::new(AtomicBool::new(false)),
            append_lock: Arc::new(Mutex::new(())),
            #[cfg(test)]
            faults: Arc::new(Faults::default()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Renders one OTLP/JSON trace document (single line, no trailing newline) for `spans`.
    pub fn render(&self, spans: &[Span]) -> String {
        let offset = self.config.unix_epoch_offset_ns;
        let nano = |tick: u64| offset.saturating_add(tick);
        let mut out = String::new();
        out.push_str("{\"resourceSpans\":[{\"resource\":{\"attributes\":[");
        out.push_str("{\"key\":\"service.name\",\"value\":{\"stringValue\":");
        push_json_string(&mut out, &self.config.service_name);
        out.push_str("}}]},\"scopeSpans\":[{\"scope\":{\"name\":");
        push_json_string(&mut out, &self.config.scope_name);
        out.push_str("},\"spans\":[");
        for (index, original) in spans.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let (span, _) = self.config.redactor.redact_span(original);
            let _ = write!(
                out,
                "{{\"traceId\":\"{}\",\"spanId\":\"{}\"",
                span.trace_id, span.id
            );
            if let Some(parent) = span.parent {
                let _ = write!(out, ",\"parentSpanId\":\"{parent}\"");
            }
            out.push_str(",\"name\":");
            push_json_string(&mut out, &span.name);
            let _ = write!(
                out,
                ",\"kind\":1,\"startTimeUnixNano\":\"{}\",\"endTimeUnixNano\":\"{}\",\"attributes\":",
                nano(span.start_tick),
                nano(span.end_tick)
            );
            push_attributes(&mut out, span.fields.iter().map(|(k, v)| (k.as_str(), v)));
            out.push_str(",\"events\":[");
            for (event_index, entry) in span.events.iter().enumerate() {
                if event_index > 0 {
                    out.push(',');
                }
                let _ = write!(out, "{{\"timeUnixNano\":\"{}\",\"name\":", nano(entry.tick));
                push_json_string(&mut out, &entry.event.name);
                out.push_str(",\"attributes\":");
                let severity_value = TraceValue::String(severity(entry.event.level).to_owned());
                push_attributes(
                    &mut out,
                    std::iter::once(("ptr.level", &severity_value))
                        .chain(entry.event.iter_fields()),
                );
                out.push('}');
            }
            out.push_str("],\"status\":");
            match &span.status {
                SpanStatus::Unset => out.push_str("{\"code\":0}"),
                SpanStatus::Ok => out.push_str("{\"code\":1}"),
                SpanStatus::Error(message) => {
                    out.push_str("{\"code\":2,\"message\":");
                    push_json_string(&mut out, message);
                    out.push('}');
                }
            }
            out.push('}');
        }
        out.push_str("]}]}]}");
        out
    }

    /// Appends one JSON line for `spans` to the file (created if missing). Empty input is a
    /// no-op. Returns the number of spans written.
    pub fn export(&self, spans: &[Span]) -> Result<usize, TraceError> {
        if spans.is_empty() {
            return Ok(0);
        }
        let mut line = self.render(spans);
        line.push('\n');
        let export_error = |error: std::io::Error| TraceError::Export {
            sink: SINK_NAME.to_owned(),
            message: error.to_string(),
        };
        let _guard = self
            .append_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.poisoned.load(Ordering::Acquire) {
            return Err(TraceError::ExporterPoisoned {
                sink: SINK_NAME.to_owned(),
            });
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(export_error)?;
        let good_len = file.metadata().map_err(export_error)?.len();
        #[cfg(test)]
        if let Some(hook) = self
            .faults
            .after_measure
            .lock()
            .ok()
            .and_then(|mut h| h.take())
        {
            (hook.0)();
        }
        if let Err(error) = self.append(&mut file, line.as_bytes()) {
            // A failed write may have left a prefix of the line; cut it off so the next export
            // is not glued onto a fragment. If the cut fails the tail is unknown: poison.
            if !self.truncate_to(good_len) {
                self.poisoned.store(true, Ordering::Release);
            }
            return Err(export_error(error));
        }
        Ok(spans.len())
    }

    fn append(&self, file: &mut File, bytes: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if let Some(prefix) = self
            .faults
            .partial_write
            .lock()
            .ok()
            .and_then(|mut f| f.take())
        {
            file.write_all(&bytes[..prefix.min(bytes.len())])?;
            return Err(std::io::Error::other("injected partial write"));
        }
        file.write_all(bytes)?;
        file.flush()
    }

    /// Cuts the file back to `len` through a separate write handle: on Windows a handle opened
    /// with `append(true)` lacks FILE_WRITE_DATA, so `set_len` on it would fail.
    fn truncate_to(&self, len: u64) -> bool {
        #[cfg(test)]
        if self.faults.truncate.load(Ordering::Relaxed) {
            return false;
        }
        OpenOptions::new()
            .write(true)
            .open(&self.path)
            .and_then(|file| file.set_len(len))
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_string_escaping_covers_quotes_backslashes_and_controls() {
        let mut out = String::new();
        push_json_string(&mut out, "a\"b\\c\nd\te\u{1}\u{7f}\u{e9}");
        assert_eq!(out, "\"a\\\"b\\\\c\\nd\\te\\u0001\u{7f}\u{e9}\"");
    }

    fn unit_path(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let name = format!(
            "ptr-observe-otlp-{tag}-{}-{nanos}.jsonl",
            std::process::id()
        );
        std::env::temp_dir().join(name)
    }

    fn one_span(name: &str) -> Span {
        let tracer = crate::Tracer::new(
            crate::TracerConfig::default(),
            Arc::new(crate::ManualClock::new(0)),
        );
        tracer.root_span(name).expect("root").finish();
        tracer.drain_finished().remove(0)
    }

    fn assert_well_formed_lines(bytes: &[u8], expected: usize) {
        let text = std::str::from_utf8(bytes).expect("utf8");
        assert!(
            text.is_empty() || text.ends_with('\n'),
            "no dangling fragment"
        );
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), expected);
        for line in lines {
            assert!(
                line.starts_with("{\"resourceSpans\"") && line.ends_with("]}]}]}"),
                "{line}"
            );
        }
    }

    #[test]
    fn a_partial_write_is_rolled_back_so_the_next_export_stays_well_formed() {
        let path = unit_path("partial");
        let exporter = OtlpJsonFileExporter::new(&path, OtlpFileConfig::default());
        assert_eq!(exporter.export(&[one_span("first")]), Ok(1));
        let before = std::fs::read(&path).expect("read");
        *exporter.faults.partial_write.lock().expect("lock") = Some(25);
        assert!(matches!(
            exporter.export(&[one_span("lost")]),
            Err(TraceError::Export { .. })
        ));
        assert_eq!(std::fs::read(&path).expect("read"), before);
        assert_eq!(exporter.export(&[one_span("second")]), Ok(1));
        assert_well_formed_lines(&std::fs::read(&path).expect("read"), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_failed_rollback_poisons_the_exporter_and_its_clones() {
        let path = unit_path("poison");
        let exporter = OtlpJsonFileExporter::new(&path, OtlpFileConfig::default());
        let clone = exporter.clone();
        *exporter.faults.partial_write.lock().expect("lock") = Some(10);
        exporter.faults.truncate.store(true, Ordering::Relaxed);
        assert!(matches!(
            exporter.export(&[one_span("x")]),
            Err(TraceError::Export { .. })
        ));
        let tail = std::fs::read(&path).expect("read");
        for target in [&exporter, &clone] {
            assert!(matches!(
                target.export(&[one_span("y")]),
                Err(TraceError::ExporterPoisoned { .. })
            ));
        }
        assert_eq!(
            std::fs::read(&path).expect("read"),
            tail,
            "nothing appended after the tail"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_clone_rollback_never_deletes_another_clones_successful_append() {
        let path = unit_path("interleave");
        let a = OtlpJsonFileExporter::new(&path, OtlpFileConfig::default());
        let b = a.clone();
        let faults = Arc::clone(&a.faults);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        *a.faults.after_measure.lock().expect("lock") = Some(Hook(Box::new(move || {
            // Clone `b` exports while `a` sits between measure and write. With the append lock
            // `b` blocks until `a` is finished, so the wait times out; without it `b` is done.
            std::thread::spawn(move || {
                let _ = done_tx.send(b.export(&[one_span("survivor")]));
            });
            let _ = done_rx.recv_timeout(std::time::Duration::from_millis(300));
            *faults.partial_write.lock().expect("lock") = Some(25);
        })));
        assert!(matches!(
            a.export(&[one_span("lost")]),
            Err(TraceError::Export { .. })
        ));
        // Wait for `b` (if it was blocked) to finish before reading the file.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut bytes = std::fs::read(&path).expect("read");
        while !std::str::from_utf8(&bytes).is_ok_and(|t| t.contains("survivor"))
            && std::time::Instant::now() < deadline
        {
            std::thread::yield_now();
            bytes = std::fs::read(&path).unwrap_or_default();
        }
        assert_well_formed_lines(&bytes, 1);
        assert!(String::from_utf8(bytes).expect("utf8").contains("survivor"));
        let _ = std::fs::remove_file(&path);
    }
}
