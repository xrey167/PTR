//! Opt-in server capabilities. Everything here defaults to off; with the default
//! [`ServerOptions`] the server behaves exactly as it did before they existed.

use crate::{ingress_diagnostics::IngressDiagnosticsRunner, request_tracing::RequestTracing};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// Default cap on the size of the trace export file, in bytes.
pub const DEFAULT_MAX_TRACE_EXPORT_BYTES: u64 = 8 * 1024 * 1024;

/// Whether `POST /v1/requests` reports advisory ingress diagnostics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IngressDiagnosticsMode {
    /// No ingress run, no extra response field (the default).
    #[default]
    Off,
    /// Run `ptr_ingress::IngressPipeline` on the request text and add an
    /// `ingress_diagnostics` field. The result is advisory: the runtime still receives the raw
    /// text unchanged and nothing is blocked, rewritten or authorized by it.
    Advisory,
}

/// Redacted request tracing: one span per request, exported as OTLP/JSON-shaped lines.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestTracingOptions {
    /// File the finished spans are appended to (created if missing).
    pub export_path: PathBuf,
    /// The export file never grows beyond this many bytes; a span batch that would exceed it
    /// is dropped and counted (see `ServerState::dropped_trace_exports`). Must be non-zero.
    pub max_export_bytes: u64,
}

impl RequestTracingOptions {
    pub fn new(export_path: impl Into<PathBuf>) -> Self {
        Self {
            export_path: export_path.into(),
            max_export_bytes: DEFAULT_MAX_TRACE_EXPORT_BYTES,
        }
    }
}

/// The opt-in capabilities of a [`crate::ServerState`]; `Default` is everything off.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ServerOptions {
    pub request_tracing: Option<RequestTracingOptions>,
    pub ingress_diagnostics: IngressDiagnosticsMode,
}

/// Why [`crate::ServerState::with_options`] refused its options.
#[derive(Debug)]
pub enum ServerOptionsError {
    /// `max_export_bytes` was zero, so no span could ever be exported.
    ZeroTraceExportCap,
    /// The export file could not be created or opened for appending.
    TraceExportUnavailable {
        path: PathBuf,
        kind: std::io::ErrorKind,
    },
}

impl fmt::Display for ServerOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTraceExportCap => f.write_str("max_export_bytes must be non-zero"),
            Self::TraceExportUnavailable { path, kind } => {
                write!(f, "trace export file {} unusable: {kind}", path.display())
            }
        }
    }
}

impl std::error::Error for ServerOptionsError {}

/// The built form of [`ServerOptions`] held by the state.
#[derive(Clone, Default)]
pub(crate) struct ActiveOptions {
    pub(crate) tracing: Option<Arc<RequestTracing>>,
    pub(crate) ingress: Option<Arc<IngressDiagnosticsRunner>>,
}

impl ActiveOptions {
    pub(crate) fn build(options: ServerOptions) -> Result<Self, ServerOptionsError> {
        let tracing = options
            .request_tracing
            .map(|tracing| RequestTracing::new(tracing).map(Arc::new))
            .transpose()?;
        let ingress = match options.ingress_diagnostics {
            IngressDiagnosticsMode::Off => None,
            IngressDiagnosticsMode::Advisory => Some(Arc::new(IngressDiagnosticsRunner::new())),
        };
        Ok(Self { tracing, ingress })
    }
}
