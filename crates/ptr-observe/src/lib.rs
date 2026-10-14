mod active;
mod clock;
mod error;
mod flow;
mod macros;
mod otlp;
mod plan;
mod redact;
mod sinks;
mod span;
mod trace;
#[cfg(feature = "tracing-adapter")]
mod tracing_adapter;

pub mod fields;

pub use active::{ActiveSpan, ActiveTree, StuckSpan, STUCK_PATH_MAX_BYTES, STUCK_PATH_MAX_DEPTH};
pub use clock::{Clock, ManualClock, MonotonicClock};
pub use error::TraceError;
pub use flow::FlowSignature;
pub use otlp::{OtlpFileConfig, OtlpJsonFileExporter};
pub use plan::{
    correlate, CorrelationReport, CorrelationSummary, PlanError, PlanRecord, PlannedStep,
    StepOutcome, StepReport, MAX_PLAN_STEPS,
};
pub use redact::{RedactingSink, RedactionPolicy, Redactor, REDACTED};
pub use sinks::{FanoutSink, MemorySink, SamplingRates, SamplingSink};
pub use span::{
    Span, SpanContext, SpanError, SpanEvent, SpanGuard, SpanId, SpanStatus, TraceId, Tracer,
    TracerConfig, TracerLimits, TracerStats,
};
pub use trace::{NoopTraceSink, TraceEvent, TraceLevel, TraceSink, TraceValue};
#[cfg(feature = "tracing-adapter")]
pub use tracing_adapter::TracingSink;
