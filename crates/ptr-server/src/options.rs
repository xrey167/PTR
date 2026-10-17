//! Opt-in server capabilities. Everything here defaults to off; with the default
//! [`ServerOptions`] the server behaves exactly as it did before they existed.

use crate::{ingress_diagnostics::IngressDiagnosticsRunner, request_tracing::RequestTracing};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// Default cap on the size of the trace export file, in bytes.
pub const DEFAULT_MAX_TRACE_EXPORT_BYTES: u64 = 8 * 1024 * 1024;
/// Hard maximum of [`RequestTracingOptions::max_export_bytes`] (1 GiB): a larger cap would let
/// the export file grow without a practical bound.
pub const MAX_TRACE_EXPORT_BYTES: u64 = 1 << 30;

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

/// Default number of ingress analyses that may run at once.
pub const DEFAULT_MAX_CONCURRENT_INGRESS_DIAGNOSTICS: usize = 2;
/// Default cap on the request-text prefix that is analysed, in bytes.
pub const DEFAULT_MAX_INGRESS_ANALYZED_BYTES: usize = 64 * 1024;
/// Hard ceiling on `max_concurrent * max_analyzed_bytes`, the bytes that may be under analysis
/// at once. At about 56 bytes of offset tables per analysed character this bounds the
/// diagnostics to roughly 56 MiB however the limits are chosen (the defaults use 128 KiB of it).
pub const MAX_INGRESS_DIAGNOSTICS_BUDGET_BYTES: usize = 1024 * 1024;

/// Resource bounds of the advisory ingress diagnostics (Invariant #8). The pipeline builds
/// per-character offset tables (about 56 bytes per character), so the bounds are separate from,
/// and much tighter than, the request admission and body caps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IngressDiagnosticsLimits {
    /// Analyses allowed at once. A request that finds none free skips its diagnostics
    /// (`"skipped": "busy"`) instead of waiting. Must be non-zero.
    pub max_concurrent: usize,
    /// Only this many leading bytes of the request text (cut at a character boundary) are
    /// analysed; the report then carries `"truncated": true`. Must be non-zero, and
    /// `max_concurrent * max_analyzed_bytes` may not exceed
    /// [`MAX_INGRESS_DIAGNOSTICS_BUDGET_BYTES`].
    pub max_analyzed_bytes: usize,
}

impl Default for IngressDiagnosticsLimits {
    fn default() -> Self {
        Self {
            max_concurrent: DEFAULT_MAX_CONCURRENT_INGRESS_DIAGNOSTICS,
            max_analyzed_bytes: DEFAULT_MAX_INGRESS_ANALYZED_BYTES,
        }
    }
}

/// Redacted request tracing: one span per request, exported as OTLP/JSON-shaped lines.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestTracingOptions {
    /// File the finished spans are appended to (created if missing).
    pub export_path: PathBuf,
    /// The export file never grows beyond this many bytes; a span batch that would exceed it
    /// is dropped and counted (see `ServerState::dropped_trace_exports`). Must be non-zero and at
    /// most [`MAX_TRACE_EXPORT_BYTES`].
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
    /// `max_export_bytes` was above [`MAX_TRACE_EXPORT_BYTES`].
    TraceExportCapTooLarge { max: u64 },
    /// `IngressDiagnosticsLimits::max_concurrent` was zero.
    ZeroIngressConcurrency,
    /// `IngressDiagnosticsLimits::max_analyzed_bytes` was zero.
    ZeroIngressAnalysisCap,
    /// `max_concurrent * max_analyzed_bytes` exceeds the hard budget
    /// ([`MAX_INGRESS_DIAGNOSTICS_BUDGET_BYTES`]).
    IngressDiagnosticsBudgetExceeded { budget: usize },
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
            Self::TraceExportCapTooLarge { max } => {
                write!(f, "max_export_bytes must not exceed {max}")
            }
            Self::ZeroIngressConcurrency => f.write_str("max_concurrent must be non-zero"),
            Self::ZeroIngressAnalysisCap => f.write_str("max_analyzed_bytes must be non-zero"),
            Self::IngressDiagnosticsBudgetExceeded { budget } => write!(
                f,
                "max_concurrent * max_analyzed_bytes must not exceed {budget} bytes"
            ),
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
    pub(crate) ingress_limits: IngressDiagnosticsLimits,
}

impl ActiveOptions {
    pub(crate) fn build(
        options: ServerOptions,
        ingress_limits: IngressDiagnosticsLimits,
    ) -> Result<Self, ServerOptionsError> {
        let tracing = options
            .request_tracing
            .map(|tracing| RequestTracing::new(tracing).map(Arc::new))
            .transpose()?;
        let ingress = Self::runner(options.ingress_diagnostics, ingress_limits)?;
        Ok(Self {
            tracing,
            ingress,
            ingress_limits,
        })
    }

    /// Replace the ingress diagnostics bounds; the runner (if enabled) is rebuilt.
    pub(crate) fn with_ingress_limits(
        mut self,
        limits: IngressDiagnosticsLimits,
    ) -> Result<Self, ServerOptionsError> {
        let mode = if self.ingress.is_some() {
            IngressDiagnosticsMode::Advisory
        } else {
            IngressDiagnosticsMode::Off
        };
        self.ingress = Self::runner(mode, limits)?;
        self.ingress_limits = limits;
        Ok(self)
    }

    fn runner(
        mode: IngressDiagnosticsMode,
        limits: IngressDiagnosticsLimits,
    ) -> Result<Option<Arc<IngressDiagnosticsRunner>>, ServerOptionsError> {
        match mode {
            IngressDiagnosticsMode::Off => Ok(None),
            IngressDiagnosticsMode::Advisory => {
                IngressDiagnosticsRunner::new(limits).map(|runner| Some(Arc::new(runner)))
            }
        }
    }
}
