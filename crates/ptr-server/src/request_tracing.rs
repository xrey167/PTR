//! Redacted request tracing with a bounded OTLP/JSON-shaped file export.
//!
//! One `ptr_observe` span per request with only these fields: request id, route (the matched
//! route pattern, never the concrete path), status, latency and outcome. Request bodies and
//! headers are never read. Every span passes through `ptr_observe::Redactor` inside
//! `OtlpJsonFileExporter::render` before a byte is written (Invariant 13), and a client-chosen
//! request id that is not a short plain identifier is replaced by a digest prefix. The file is
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Set by the request handler on its response so the tracing layer can record the request id
/// without reading any body.
#[derive(Clone, Debug)]
pub(crate) struct TracedRequestId(pub(crate) String);

const MAX_PLAIN_ID_BYTES: usize = 64;

pub(crate) struct RequestTracing {
    tracer: Tracer,
    exporter: OtlpJsonFileExporter,
    max_bytes: u64,
    /// Serialises drain + write and holds the bytes the file is known to hold.
    written: Mutex<u64>,
    dropped_exports: AtomicU64,
    export_errors: AtomicU64,
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
        })
    }

    pub(crate) fn dropped_exports(&self) -> u64 {
        self.dropped_exports.load(Ordering::Relaxed)
    }

    pub(crate) fn export_errors(&self) -> u64 {
        self.export_errors.load(Ordering::Relaxed)
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
        // `render` redacts every span before serialising.
        let mut line = self.exporter.render(&spans);
        line.push('\n');
        let size = line.len() as u64;
        if written.saturating_add(size) > self.max_bytes {
            self.dropped_exports
                .fetch_add(spans.len() as u64, Ordering::Relaxed);
            return;
        }
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.exporter.path())
            .and_then(|mut file| file.write_all(line.as_bytes()).and_then(|()| file.flush()));
        match result {
            Ok(()) => *written += size,
            Err(_) => {
                self.export_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn plain_id(id: &str) -> String {
    let plain = !id.is_empty()
        && id.len() <= MAX_PLAIN_ID_BYTES
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-'));
    if plain {
        return id.to_owned();
    }
    let digest = Sha256::digest(id.as_bytes());
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
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
            .map_or_else(|| "none".to_owned(), |id| plain_id(&id.0));
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
