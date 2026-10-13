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
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

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
}

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
            sink: "otlp-json-file".to_owned(),
            message: error.to_string(),
        };
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(export_error)?;
        file.write_all(line.as_bytes()).map_err(export_error)?;
        file.flush().map_err(export_error)?;
        Ok(spans.len())
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
}
