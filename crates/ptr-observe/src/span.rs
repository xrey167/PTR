//! Bounded span model: deterministic ids, a `Tracer` with an injectable clock and RAII guards.
//!
//! Bounds (Invariant 8) and their failure behavior:
//! - `max_open_spans`: starting a span beyond it is a typed error (`SpanError::TooManyOpenSpans`).
//! - `max_finished_spans`: the finished-span buffer is a ring; the oldest span is dropped and
//!   counted in `TracerStats::dropped_finished`.
//! - `max_fields_per_event`, `max_events_per_span`, `max_value_len`: excess fields/events are
//!   dropped (in key order, deterministic) and over-long strings are truncated at a char
//!   boundary; both are counted in `TracerStats`, never an error on the hot path.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ptr_types::RequestId;

use crate::{Clock, TraceEvent, TraceValue};

/// Observability trace identifier (distinct from `ptr_types::TraceId`).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TraceId(pub u128);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SpanId(pub u64);

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// Identity of a span, enough to parent another span on it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpanContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpanEvent {
    pub tick: u64,
    pub event: TraceEvent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Span {
    pub trace_id: TraceId,
    pub id: SpanId,
    pub parent: Option<SpanId>,
    pub name: String,
    pub start_tick: u64,
    pub end_tick: u64,
    pub fields: BTreeMap<String, TraceValue>,
    pub events: Vec<SpanEvent>,
    pub status: SpanStatus,
}

impl Span {
    pub fn duration(&self) -> u64 {
        self.end_tick.saturating_sub(self.start_tick)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TracerLimits {
    pub max_open_spans: usize,
    pub max_finished_spans: usize,
    pub max_fields_per_event: usize,
    pub max_events_per_span: usize,
    pub max_value_len: usize,
}

impl TracerLimits {
    /// Hard maximum of `max_open_spans` (16 times the default).
    pub const HARD_MAX_OPEN_SPANS: usize = 16_384;
    /// Hard maximum of `max_finished_spans` (16 times the default).
    pub const HARD_MAX_FINISHED_SPANS: usize = 65_536;
    /// Hard maximum of `max_fields_per_event` (32 times the default).
    pub const HARD_MAX_FIELDS_PER_EVENT: usize = 1_024;
    /// Hard maximum of `max_events_per_span` (64 times the default).
    pub const HARD_MAX_EVENTS_PER_SPAN: usize = 4_096;
    /// Hard maximum of `max_value_len` (64 times the default).
    pub const HARD_MAX_VALUE_LEN: usize = 65_536;

    /// Returns these limits with every field capped at its `HARD_MAX_*` constant. [`Tracer::new`]
    /// applies this, so a host config of `usize::MAX` cannot make a buffer unbounded.
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            max_open_spans: self.max_open_spans.min(Self::HARD_MAX_OPEN_SPANS),
            max_finished_spans: self.max_finished_spans.min(Self::HARD_MAX_FINISHED_SPANS),
            max_fields_per_event: self
                .max_fields_per_event
                .min(Self::HARD_MAX_FIELDS_PER_EVENT),
            max_events_per_span: self.max_events_per_span.min(Self::HARD_MAX_EVENTS_PER_SPAN),
            max_value_len: self.max_value_len.min(Self::HARD_MAX_VALUE_LEN),
        }
    }
}

impl Default for TracerLimits {
    fn default() -> Self {
        Self {
            max_open_spans: 1024,
            max_finished_spans: 4096,
            max_fields_per_event: 32,
            max_events_per_span: 64,
            max_value_len: 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TracerConfig {
    /// Seed of the deterministic id generator; equal seeds yield equal id sequences.
    pub id_seed: u64,
    pub limits: TracerLimits,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TracerStats {
    pub spans_opened: u64,
    pub spans_finished: u64,
    pub rejected_opens: u64,
    pub dropped_finished: u64,
    pub dropped_fields: u64,
    pub dropped_events: u64,
    pub truncated_values: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpanError {
    TooManyOpenSpans { limit: usize },
    ParentNotOpen { span_id: SpanId },
}

impl SpanError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::TooManyOpenSpans { .. } => "too_many_open_spans",
            Self::ParentNotOpen { .. } => "parent_not_open",
        }
    }
}

impl fmt::Display for SpanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyOpenSpans { limit } => write!(f, "open span limit {limit} reached"),
            Self::ParentNotOpen { span_id } => write!(f, "parent span {span_id} is not open"),
        }
    }
}

impl std::error::Error for SpanError {}

/// One currently open span as seen by diagnostics.
pub(crate) struct OpenSpanView {
    pub(crate) id: SpanId,
    pub(crate) trace_id: TraceId,
    pub(crate) parent: Option<SpanId>,
    pub(crate) name: String,
    pub(crate) start_tick: u64,
}

struct OpenSpan {
    trace_id: TraceId,
    parent: Option<SpanId>,
    name: String,
    start_tick: u64,
    fields: BTreeMap<String, TraceValue>,
    events: Vec<SpanEvent>,
    status: SpanStatus,
}

struct State {
    seed: u64,
    counter: u64,
    open: BTreeMap<SpanId, OpenSpan>,
    finished: VecDeque<Span>,
    stats: TracerStats,
}

impl State {
    fn next_u64(&mut self) -> u64 {
        loop {
            self.counter = self.counter.wrapping_add(1);
            let mut z = self
                .seed
                .wrapping_add(self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if z != 0 {
                return z;
            }
        }
    }
}

struct Inner {
    clock: Arc<dyn Clock>,
    limits: TracerLimits,
    state: Mutex<State>,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn truncate(text: &str, max: usize, stats: &mut TracerStats) -> String {
    if text.len() <= max {
        return text.to_owned();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    stats.truncated_values += 1;
    text[..end].to_owned()
}

fn bound_fields(
    fields: BTreeMap<String, TraceValue>,
    limits: &TracerLimits,
    stats: &mut TracerStats,
) -> BTreeMap<String, TraceValue> {
    let mut bounded = BTreeMap::new();
    for (key, value) in fields {
        if bounded.len() >= limits.max_fields_per_event {
            stats.dropped_fields += 1;
            continue;
        }
        let key = truncate(&key, limits.max_value_len, stats);
        let value = match value {
            TraceValue::String(text) => {
                TraceValue::String(truncate(&text, limits.max_value_len, stats))
            }
            // A client may supply any string as a request id, so it is bounded like a string.
            TraceValue::RequestId(id) => {
                TraceValue::RequestId(RequestId(truncate(&id.0, limits.max_value_len, stats)))
            }
            value @ (TraceValue::U64(_)
            | TraceValue::Bool(_)
            | TraceValue::Revision(_)
            | TraceValue::Generation(_)
            | TraceValue::CommitIndex(_)) => value,
        };
        bounded.insert(key, value);
    }
    bounded
}

/// Shared, cloneable span tracer. Thread-safe; all state is bounded.
#[derive(Clone)]
pub struct Tracer {
    inner: Arc<Inner>,
}

impl fmt::Debug for Tracer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tracer")
            .field("limits", &self.inner.limits)
            .field("stats", &self.stats())
            .finish()
    }
}

impl Tracer {
    pub fn new(config: TracerConfig, clock: Arc<dyn Clock>) -> Self {
        Self {
            inner: Arc::new(Inner {
                clock,
                limits: config.limits.clamped(),
                state: Mutex::new(State {
                    seed: config.id_seed,
                    counter: 0,
                    open: BTreeMap::new(),
                    finished: VecDeque::new(),
                    stats: TracerStats::default(),
                }),
            }),
        }
    }

    pub fn limits(&self) -> TracerLimits {
        self.inner.limits
    }

    pub fn now_tick(&self) -> u64 {
        self.inner.clock.now_tick()
    }

    /// Starts a new trace with a root span.
    pub fn root_span(&self, name: impl Into<String>) -> Result<SpanGuard, SpanError> {
        self.start_span(name, None)
    }

    /// Starts a span; `parent` must be an open span (it supplies the trace id).
    pub fn start_span(
        &self,
        name: impl Into<String>,
        parent: Option<SpanContext>,
    ) -> Result<SpanGuard, SpanError> {
        let tick = self.inner.clock.now_tick();
        let name = name.into();
        let mut state = self.inner.lock();
        if state.open.len() >= self.inner.limits.max_open_spans {
            state.stats.rejected_opens += 1;
            return Err(SpanError::TooManyOpenSpans {
                limit: self.inner.limits.max_open_spans,
            });
        }
        let (trace_id, parent_id) = match parent {
            Some(ctx) => match state.open.get(&ctx.span_id) {
                Some(open) => (open.trace_id, Some(ctx.span_id)),
                None => {
                    return Err(SpanError::ParentNotOpen {
                        span_id: ctx.span_id,
                    })
                }
            },
            None => {
                let hi = u128::from(state.next_u64());
                let lo = u128::from(state.next_u64());
                (TraceId((hi << 64) | lo), None)
            }
        };
        let span_id = SpanId(state.next_u64());
        let mut stats = state.stats;
        let name = truncate(&name, self.inner.limits.max_value_len, &mut stats);
        state.stats = stats;
        state.open.insert(
            span_id,
            OpenSpan {
                trace_id,
                parent: parent_id,
                name,
                start_tick: tick,
                fields: BTreeMap::new(),
                events: Vec::new(),
                status: SpanStatus::Unset,
            },
        );
        state.stats.spans_opened += 1;
        Ok(SpanGuard {
            inner: Arc::clone(&self.inner),
            context: SpanContext { trace_id, span_id },
            closed: false,
        })
    }

    /// Removes and returns buffered finished spans in completion order.
    pub fn drain_finished(&self) -> Vec<Span> {
        self.inner.lock().finished.drain(..).collect()
    }

    /// Copies buffered finished spans without removing them.
    pub fn finished(&self) -> Vec<Span> {
        self.inner.lock().finished.iter().cloned().collect()
    }

    pub fn open_span_count(&self) -> usize {
        self.inner.lock().open.len()
    }

    pub fn stats(&self) -> TracerStats {
        self.inner.lock().stats
    }

    pub(crate) fn open_snapshot(&self) -> (u64, Vec<OpenSpanView>) {
        let now = self.inner.clock.now_tick();
        let state = self.inner.lock();
        let spans = state
            .open
            .iter()
            .map(|(id, open)| OpenSpanView {
                id: *id,
                trace_id: open.trace_id,
                parent: open.parent,
                name: open.name.clone(),
                start_tick: open.start_tick,
            })
            .collect();
        (now, spans)
    }
}

/// RAII handle of an open span; dropping it finishes the span with its current status.
pub struct SpanGuard {
    inner: Arc<Inner>,
    context: SpanContext,
    closed: bool,
}

impl fmt::Debug for SpanGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanGuard")
            .field("context", &self.context)
            .finish()
    }
}

impl SpanGuard {
    pub fn context(&self) -> SpanContext {
        self.context
    }

    /// Starts a child span of this one.
    pub fn child(&self, name: impl Into<String>) -> Result<SpanGuard, SpanError> {
        Tracer {
            inner: Arc::clone(&self.inner),
        }
        .start_span(name, Some(self.context))
    }

    /// Sets (or replaces) a span field, subject to the field and value bounds.
    pub fn set_field(&self, key: impl Into<String>, value: impl Into<TraceValue>) {
        let limits = self.inner.limits;
        let mut one = BTreeMap::new();
        one.insert(key.into(), value.into());
        let mut state = self.inner.lock();
        let mut stats = state.stats;
        let bounded = bound_fields(one, &limits, &mut stats);
        state.stats = stats;
        let Some(open) = state.open.get_mut(&self.context.span_id) else {
            return;
        };
        let mut dropped = 0;
        for (key, value) in bounded {
            if open.fields.contains_key(&key) || open.fields.len() < limits.max_fields_per_event {
                open.fields.insert(key, value);
            } else {
                dropped += 1;
            }
        }
        state.stats.dropped_fields += dropped;
    }

    pub fn set_status(&self, status: SpanStatus) {
        let max = self.inner.limits.max_value_len;
        let mut state = self.inner.lock();
        let mut stats = state.stats;
        let status = match status {
            SpanStatus::Error(message) => SpanStatus::Error(truncate(&message, max, &mut stats)),
            other => other,
        };
        state.stats = stats;
        if let Some(open) = state.open.get_mut(&self.context.span_id) {
            open.status = status;
        }
    }

    /// Attaches an event to this span at the current tick, subject to the event bounds.
    pub fn event(&self, event: TraceEvent) {
        let limits = self.inner.limits;
        let tick = self.inner.clock.now_tick();
        let mut state = self.inner.lock();
        let mut stats = state.stats;
        let name = truncate(&event.name, limits.max_value_len, &mut stats);
        let fields = bound_fields(event.fields, &limits, &mut stats);
        state.stats = stats;
        let Some(open) = state.open.get_mut(&self.context.span_id) else {
            return;
        };
        if open.events.len() >= limits.max_events_per_span {
            state.stats.dropped_events += 1;
            return;
        }
        open.events.push(SpanEvent {
            tick,
            event: TraceEvent {
                level: event.level,
                name,
                fields,
            },
        });
    }

    /// Finishes the span with `SpanStatus::Ok` unless a status was already set.
    pub fn finish(mut self) {
        self.close(SpanStatus::Ok);
    }

    /// Finishes the span with an error status unless a status was already set.
    pub fn fail(mut self, message: impl Into<String>) {
        self.close(SpanStatus::Error(message.into()));
    }

    fn close(&mut self, default_status: SpanStatus) {
        if self.closed {
            return;
        }
        self.closed = true;
        let end_tick = self.inner.clock.now_tick();
        let max = self.inner.limits.max_value_len;
        let cap = self.inner.limits.max_finished_spans.max(1);
        let mut state = self.inner.lock();
        let Some(open) = state.open.remove(&self.context.span_id) else {
            return;
        };
        let mut stats = state.stats;
        let status = match open.status {
            SpanStatus::Unset => match default_status {
                SpanStatus::Error(message) => {
                    SpanStatus::Error(truncate(&message, max, &mut stats))
                }
                other => other,
            },
            existing => existing,
        };
        state.stats = stats;
        let span = Span {
            trace_id: open.trace_id,
            id: self.context.span_id,
            parent: open.parent,
            name: open.name,
            start_tick: open.start_tick,
            end_tick: end_tick.max(open.start_tick),
            fields: open.fields,
            events: open.events,
            status,
        };
        if state.finished.len() >= cap {
            state.finished.pop_front();
            state.stats.dropped_finished += 1;
        }
        state.finished.push_back(span);
        state.stats.spans_finished += 1;
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        // A dropped guard keeps whatever status was set explicitly (Unset otherwise).
        self.close(SpanStatus::Unset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ManualClock, TraceLevel};

    fn tracer(limits: TracerLimits) -> (Tracer, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new(100));
        let tracer = Tracer::new(TracerConfig { id_seed: 7, limits }, clock.clone());
        (tracer, clock)
    }

    #[test]
    fn ids_are_deterministic_for_equal_seeds_and_nonzero() {
        let (a, _) = tracer(TracerLimits::default());
        let (b, _) = tracer(TracerLimits::default());
        let ga = a.root_span("x").expect("open");
        let gb = b.root_span("x").expect("open");
        assert_eq!(ga.context(), gb.context());
        assert_ne!(ga.context().span_id.0, 0);
        assert_eq!(ga.context().trace_id.to_string().len(), 32);
        assert_eq!(ga.context().span_id.to_string().len(), 16);
    }

    #[test]
    fn lifecycle_nesting_and_ticks() {
        let (t, clock) = tracer(TracerLimits::default());
        let root = t.root_span("root").expect("root");
        clock.advance(10);
        let child = root.child("child").expect("child");
        clock.advance(5);
        child.set_field("k", "v");
        child.finish();
        clock.advance(5);
        drop(root);
        let spans = t.drain_finished();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].name, "child");
        assert_eq!(spans[0].parent, Some(spans[1].id));
        assert_eq!(spans[0].trace_id, spans[1].trace_id);
        assert_eq!((spans[0].start_tick, spans[0].end_tick), (110, 115));
        assert_eq!((spans[1].start_tick, spans[1].end_tick), (100, 120));
        assert_eq!(spans[0].status, SpanStatus::Ok);
        assert_eq!(spans[1].status, SpanStatus::Unset);
        assert_eq!(t.open_span_count(), 0);
    }

    #[test]
    fn open_span_limit_is_typed_error() {
        let limits = TracerLimits {
            max_open_spans: 1,
            ..TracerLimits::default()
        };
        let (t, _) = tracer(limits);
        let _g = t.root_span("a").expect("first");
        let err = t.root_span("b").expect_err("limit");
        assert_eq!(err, SpanError::TooManyOpenSpans { limit: 1 });
        assert_eq!(err.code(), "too_many_open_spans");
        assert_eq!(t.stats().rejected_opens, 1);
    }

    #[test]
    fn closed_parent_is_rejected() {
        let (t, _) = tracer(TracerLimits::default());
        let root = t.root_span("a").expect("root");
        let ctx = root.context();
        root.finish();
        let err = t.start_span("b", Some(ctx)).expect_err("closed");
        assert_eq!(err.code(), "parent_not_open");
        assert!(err.to_string().contains("not open"));
    }

    #[test]
    fn finished_buffer_is_a_ring() {
        let limits = TracerLimits {
            max_finished_spans: 2,
            ..TracerLimits::default()
        };
        let (t, _) = tracer(limits);
        for name in ["a", "b", "c"] {
            t.root_span(name).expect("open").finish();
        }
        let names: Vec<_> = t.finished().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["b", "c"]);
        assert_eq!(t.stats().dropped_finished, 1);
    }

    #[test]
    fn fields_events_and_values_are_bounded() {
        let limits = TracerLimits {
            max_fields_per_event: 2,
            max_events_per_span: 1,
            max_value_len: 4,
            ..TracerLimits::default()
        };
        let (t, _) = tracer(limits);
        let g = t.root_span("abcdefgh").expect("open");
        g.set_field("a", "long-value");
        g.set_field("b", 1u64);
        g.set_field("c", 2u64);
        g.set_field("a", "zz");
        g.event(
            TraceEvent::new(TraceLevel::Info, "evt-long")
                .with_field("x", 1u64)
                .with_field("y", 1u64)
                .with_field("z", 1u64),
        );
        g.event(TraceEvent::new(TraceLevel::Info, "second"));
        g.finish();
        let s = t.drain_finished().remove(0);
        assert_eq!(s.name, "abcd");
        assert_eq!(s.fields.len(), 2);
        assert_eq!(s.fields["a"], TraceValue::String("zz".into()));
        assert_eq!(s.events.len(), 1);
        assert_eq!(s.events[0].event.name, "evt-");
        assert_eq!(s.events[0].event.fields.len(), 2);
        let stats = t.stats();
        assert_eq!(stats.dropped_events, 1);
        assert_eq!(stats.dropped_fields, 2);
        assert!(stats.truncated_values >= 3);
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        let mut stats = TracerStats::default();
        assert_eq!(truncate("a\u{e9}", 2, &mut stats), "a");
    }

    #[test]
    fn fail_sets_error_status_and_explicit_status_wins_over_finish() {
        let (t, _) = tracer(TracerLimits::default());
        t.root_span("a").expect("open").fail("boom");
        let g = t.root_span("b").expect("open");
        g.set_status(SpanStatus::Error("x".into()));
        g.finish();
        let spans = t.drain_finished();
        assert_eq!(spans[0].status, SpanStatus::Error("boom".into()));
        assert_eq!(spans[1].status, SpanStatus::Error("x".into()));
    }
}
