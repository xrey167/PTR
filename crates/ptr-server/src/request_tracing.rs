//! Redacted request tracing with a bounded OTLP/JSON-shaped file export.
//!
//! One `ptr_observe` span per request with only these fields: request id, route (the matched
//! route pattern, never the concrete path), status, latency and outcome. Request bodies and
//! headers are never read. Every span passes through `ptr_observe::Redactor` inside
//! `OtlpJsonFileExporter::render` before a byte is written (Invariant 13), and every
//! client-chosen request id is replaced by a SHA-256 digest prefix (never recorded raw). The file is
//! capped: a batch that would exceed `max_export_bytes` is dropped and counted.

use crate::options::{RequestTracingOptions, ServerOptionsError};
use axum::{
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::Response,
};
use ptr_observe::{
    fields, MonotonicClock, OtlpFileConfig, OtlpJsonFileExporter, SpanStatus, Tracer, TracerConfig,
};
use sha2::{Digest, Sha256};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Set by the request handler on its response so the tracing layer can record the request id
/// without reading any body.
#[derive(Clone, Debug)]
pub(crate) struct TracedRequestId(pub(crate) String);

pub(crate) struct RequestTracing {
    tracer: Tracer,
    exporter: OtlpJsonFileExporter,
    max_bytes: u64,
    /// Serialises drain + write and holds the bytes the file is known to hold.
    written: Mutex<u64>,
    dropped_exports: AtomicU64,
    export_errors: AtomicU64,
    poisoned: AtomicBool,
    #[cfg(test)]
    faults: Mutex<tests::Faults>,
}

impl RequestTracing {
    pub(crate) fn new(options: RequestTracingOptions) -> Result<Self, ServerOptionsError> {
        if options.max_export_bytes == 0 {
            return Err(ServerOptionsError::ZeroTraceExportCap);
        }
        let unavailable = |error: std::io::Error| ServerOptionsError::TraceExportUnavailable {
            path: options.export_path.clone(),
            kind: error.kind(),
        };
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&options.export_path)
            .map_err(unavailable)?;
        let existing = file.metadata().map_err(unavailable)?.len();
        let clock = Arc::new(MonotonicClock::new());
        let epoch_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let config = OtlpFileConfig {
            service_name: "ptr-server".to_owned(),
            scope_name: "ptr-server.request".to_owned(),
            unix_epoch_offset_ns: epoch_ns,
            ..OtlpFileConfig::default()
        };
        Ok(Self {
            tracer: Tracer::new(TracerConfig::default(), clock),
            exporter: OtlpJsonFileExporter::new(options.export_path, config),
            max_bytes: options.max_export_bytes,
            written: Mutex::new(existing),
            dropped_exports: AtomicU64::new(0),
            export_errors: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
            #[cfg(test)]
            faults: Mutex::new(tests::Faults::default()),
        })
    }

    /// True once a failed write could not be rolled back: the file may end in a partial line,
    /// so nothing more is written (further batches are counted in `export_errors`).
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Relaxed)
    }

    pub(crate) fn dropped_exports(&self) -> u64 {
        self.dropped_exports.load(Ordering::Relaxed)
    }

    pub(crate) fn export_errors(&self) -> u64 {
        self.export_errors.load(Ordering::Relaxed)
    }

    fn write_line(&self, file: &mut std::fs::File, line: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if let Some(fault) = self.faults.lock().ok().and_then(|mut f| f.write.take()) {
            return fault.apply(file, line);
        }
        file.write_all(line)?;
        file.flush()
    }

    fn export_finished(&self) {
        let mut written = self
            .written
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let spans = self.tracer.drain_finished();
        if spans.is_empty() {
            return;
        }
        if self.is_poisoned() {
            self.export_errors.fetch_add(1, Ordering::Relaxed);
            return;
        }
        // `render` redacts every span before serialising.
        let mut line = self.exporter.render(&spans);
        line.push('\n');
        let size = line.len() as u64;
        if written.saturating_add(size) > self.max_bytes {
            self.dropped_exports
                .fetch_add(spans.len() as u64, Ordering::Relaxed);
            return;
        }
        let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.exporter.path())
        else {
            self.export_errors.fetch_add(1, Ordering::Relaxed);
            return;
        };
        // The length before the write is what a failed write is rolled back to.
        let Ok(before) = file.metadata().map(|metadata| metadata.len()) else {
            self.export_errors.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let result = self.write_line(&mut file, line.as_bytes());
        drop(file);
        match result {
            Ok(()) => *written = before.saturating_add(size),
            Err(_) => {
                self.export_errors.fetch_add(1, Ordering::Relaxed);
                // `write_all` may have written part of the line, and a failed `flush` may
                // follow a complete write: either way the file must go back to `before`.
                match self.truncate_to(before) {
                    Ok(()) => *written = before,
                    // The tail of the file is unknown: stop writing instead of guessing.
                    Err(_) => self.poisoned.store(true, Ordering::Relaxed),
                }
            }
        }
    }

    fn truncate_to(&self, len: u64) -> std::io::Result<()> {
        #[cfg(test)]
        if self
            .faults
            .lock()
            .is_ok_and(|mut f| std::mem::take(&mut f.truncate_fails))
        {
            return Err(std::io::Error::other("injected truncate failure"));
        }
        // A separate write handle: an append-only handle cannot be truncated on Windows because
        // `FILE_APPEND_DATA` alone lacks `FILE_WRITE_DATA`, which `set_len` needs.
        std::fs::OpenOptions::new()
            .write(true)
            .open(self.exporter.path())?
            .set_len(len)
    }
}

/// Every client-chosen request id is hashed before it enters a span. A short alphanumeric id
/// can be a low-entropy credential (`hunter2`) that the redactor's entropy threshold does not
/// recognise, so no value is trusted as plain. The same id always maps to the same digest, so
/// traces stay correlatable; the raw value is never recorded.
fn hashed_id(id: &str) -> String {
    let digest = Sha256::digest(id.as_bytes());
    let hex: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

fn outcome(status: u16) -> &'static str {
    match status {
        200..=399 => "ok",
        400..=499 => "client_error",
        _ => "server_error",
    }
}

/// Axum middleware (`from_fn_with_state`): wraps one request in a span.
pub(crate) async fn trace_request(
    State(tracing): State<Arc<RequestTracing>>,
    request: Request,
    next: Next,
) -> Response {
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "unmatched".to_owned(), |path| path.as_str().to_owned());
    // A full tracer (too many open spans) skips this request; it is never an error.
    let guard = tracing.tracer.root_span("http.request").ok();
    let started = tracing.tracer.now_tick();
    let response = next.run(request).await;
    if let Some(guard) = guard {
        let status = response.status().as_u16();
        let id = response
            .extensions()
            .get::<TracedRequestId>()
            .map_or_else(|| "none".to_owned(), |id| hashed_id(&id.0));
        guard.set_field(fields::REQUEST_ID, id.as_str());
        guard.set_field("http.route", route.as_str());
        guard.set_field("http.status", u64::from(status));
        guard.set_field(
            fields::LATENCY_NS,
            tracing.tracer.now_tick().saturating_sub(started),
        );
        guard.set_field(fields::OUTCOME, outcome(status));
        if status >= 500 {
            guard.set_status(SpanStatus::Error("server_error".to_owned()));
        }
        guard.finish();
        tracing.export_finished();
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only fault seam: the next write fails after `Partial` bytes or at flush time.
    pub(super) enum WriteFault {
        Partial(usize),
        Flush,
    }

    impl WriteFault {
        pub(super) fn apply(self, file: &mut std::fs::File, line: &[u8]) -> std::io::Result<()> {
            match self {
                Self::Partial(n) => {
                    file.write_all(&line[..n.min(line.len())])?;
                    Err(std::io::Error::other("injected partial write"))
                }
                Self::Flush => {
                    file.write_all(line)?;
                    Err(std::io::Error::other("injected flush failure"))
                }
            }
        }
    }

    #[derive(Default)]
    pub(super) struct Faults {
        pub(super) write: Option<WriteFault>,
        pub(super) truncate_fails: bool,
    }

    fn temp_file(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ptr-server-rt-{tag}-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ))
    }

    fn tracing_at(path: &std::path::Path, cap: u64) -> RequestTracing {
        RequestTracing::new(RequestTracingOptions {
            export_path: path.to_owned(),
            max_export_bytes: cap,
        })
        .unwrap()
    }

    fn emit(tracing: &RequestTracing, id: &str) {
        let guard = tracing.tracer.root_span("http.request").unwrap();
        guard.set_field(fields::REQUEST_ID, hashed_id(id).as_str());
        guard.finish();
        tracing.export_finished();
    }

    fn assert_well_formed(path: &std::path::Path, lines: usize) {
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.is_empty() || text.ends_with('\n'), "{text:?}");
        assert_eq!(text.lines().count(), lines);
        for line in text.lines() {
            serde_json::from_str::<serde_json::Value>(line).unwrap();
        }
    }

    fn set_fault(tracing: &RequestTracing, write: Option<WriteFault>, truncate_fails: bool) {
        *tracing.faults.lock().unwrap() = Faults {
            write,
            truncate_fails,
        };
    }

    #[test]
    fn a_partial_write_is_rolled_back_and_the_cap_stays_exact() {
        let path = temp_file("partial");
        let tracing = tracing_at(&path, 100_000);
        emit(&tracing, "a");
        let before = std::fs::read(&path).unwrap();
        set_fault(&tracing, Some(WriteFault::Partial(40)), false);
        emit(&tracing, "b");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(tracing.export_errors(), 1);
        assert_eq!(*tracing.written.lock().unwrap(), before.len() as u64);
        emit(&tracing, "c");
        assert_well_formed(&path, 2);
        assert_eq!(
            *tracing.written.lock().unwrap(),
            std::fs::metadata(&path).unwrap().len()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_flush_failure_after_a_full_write_is_rolled_back_and_the_cap_stays_exact() {
        let path = temp_file("flush");
        let tracing = tracing_at(&path, 100_000);
        emit(&tracing, "a");
        let before = std::fs::read(&path).unwrap();
        set_fault(&tracing, Some(WriteFault::Flush), false);
        emit(&tracing, "b");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(*tracing.written.lock().unwrap(), before.len() as u64);
        emit(&tracing, "c");
        assert_well_formed(&path, 2);
        assert_eq!(
            *tracing.written.lock().unwrap(),
            std::fs::metadata(&path).unwrap().len()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_cap_holds_across_repeated_flush_failures() {
        let path = temp_file("cap");
        let probe = temp_file("probe");
        let one = {
            let t = tracing_at(&probe, 100_000);
            emit(&t, "x");
            std::fs::metadata(&probe).unwrap().len()
        };
        let _ = std::fs::remove_file(probe);
        // Room for exactly two lines.
        let tracing = tracing_at(&path, one * 2);
        for _ in 0..3 {
            set_fault(&tracing, Some(WriteFault::Flush), false);
            emit(&tracing, "x");
        }
        emit(&tracing, "x");
        emit(&tracing, "x");
        emit(&tracing, "x");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), one * 2);
        assert_well_formed(&path, 2);
        assert_eq!(tracing.dropped_exports(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_failed_truncate_poisons_the_writer_and_later_spans_are_counted_errors() {
        let path = temp_file("poison");
        let tracing = tracing_at(&path, 100_000);
        emit(&tracing, "a");
        let before = std::fs::metadata(&path).unwrap().len();
        set_fault(&tracing, Some(WriteFault::Partial(40)), true);
        emit(&tracing, "b");
        assert!(tracing.is_poisoned());
        let after_failure = std::fs::read(&path).unwrap();
        assert!(after_failure.len() as u64 <= 100_000 && after_failure.len() as u64 > before);
        emit(&tracing, "c");
        emit(&tracing, "d");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            after_failure,
            "no write once poisoned"
        );
        assert_eq!(tracing.export_errors(), 3);
        let _ = std::fs::remove_file(path);
    }
}
