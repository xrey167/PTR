//! Recording and routing sinks: `MemorySink`, `FanoutSink`, `SamplingSink`.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{TraceError, TraceEvent, TraceLevel, TraceSink};

/// Bounded in-memory ring of events, inspectable in tests. When full, the oldest event is
/// evicted and counted in [`MemorySink::dropped`]. A capacity of 0 is treated as 1.
#[derive(Debug)]
pub struct MemorySink {
    capacity: usize,
    events: Mutex<VecDeque<TraceEvent>>,
    dropped: AtomicU64,
}

impl MemorySink {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            events: Mutex::new(VecDeque::new()),
            dropped: AtomicU64::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn events(&self) -> Vec<TraceEvent> {
        self.lock().iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<TraceEvent>> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl TraceSink for MemorySink {
    fn emit(&self, event: &TraceEvent) -> Result<(), TraceError> {
        let mut events = self.lock();
        if events.len() >= self.capacity {
            events.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        events.push_back(event.clone());
        Ok(())
    }
}

/// Delivers every event to all child sinks, in order. A failing child does not stop delivery to
/// the others; the first error (in child order) is returned after all children were tried.
#[derive(Default)]
pub struct FanoutSink {
    sinks: Vec<Arc<dyn TraceSink>>,
}

impl fmt::Debug for FanoutSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FanoutSink")
            .field("sinks", &self.sinks.len())
            .finish()
    }
}

impl FanoutSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sink(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    pub fn len(&self) -> usize {
        self.sinks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }
}

impl TraceSink for FanoutSink {
    fn emit(&self, event: &TraceEvent) -> Result<(), TraceError> {
        let mut first_error = None;
        for sink in &self.sinks {
            if let Err(error) = sink.emit(event) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Per-level 1-in-N rates: `1` keeps every event, `N` keeps events number 0, N, 2N, ... of that
/// level, and `0` drops every event of the level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingRates {
    pub error: u32,
    pub warn: u32,
    pub info: u32,
    pub debug: u32,
    pub trace: u32,
}

impl SamplingRates {
    pub const KEEP_ALL: Self = Self {
        error: 1,
        warn: 1,
        info: 1,
        debug: 1,
        trace: 1,
    };

    pub fn with_rate(mut self, level: TraceLevel, one_in: u32) -> Self {
        match level {
            TraceLevel::Error => self.error = one_in,
            TraceLevel::Warn => self.warn = one_in,
            TraceLevel::Info => self.info = one_in,
            TraceLevel::Debug => self.debug = one_in,
            TraceLevel::Trace => self.trace = one_in,
        }
        self
    }

    fn rate(&self, level: TraceLevel) -> u32 {
        match level {
            TraceLevel::Error => self.error,
            TraceLevel::Warn => self.warn,
            TraceLevel::Info => self.info,
            TraceLevel::Debug => self.debug,
            TraceLevel::Trace => self.trace,
        }
    }
}

impl Default for SamplingRates {
    fn default() -> Self {
        Self::KEEP_ALL
    }
}

fn level_index(level: TraceLevel) -> usize {
    match level {
        TraceLevel::Error => 0,
        TraceLevel::Warn => 1,
        TraceLevel::Info => 2,
        TraceLevel::Debug => 3,
        TraceLevel::Trace => 4,
    }
}

/// Deterministic counter-based sampler: no randomness, so equal event sequences give equal
/// outputs.
#[derive(Debug)]
pub struct SamplingSink<S: TraceSink> {
    inner: S,
    rates: SamplingRates,
    seen: [AtomicU64; 5],
    sampled_out: AtomicU64,
}

impl<S: TraceSink> SamplingSink<S> {
    pub fn new(inner: S, rates: SamplingRates) -> Self {
        Self {
            inner,
            rates,
            seen: Default::default(),
            sampled_out: AtomicU64::new(0),
        }
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    pub fn sampled_out(&self) -> u64 {
        self.sampled_out.load(Ordering::Relaxed)
    }
}

impl<S: TraceSink> TraceSink for SamplingSink<S> {
    fn emit(&self, event: &TraceEvent) -> Result<(), TraceError> {
        let rate = u64::from(self.rates.rate(event.level));
        let index = self.seen[level_index(event.level)].fetch_add(1, Ordering::Relaxed);
        if rate == 0 || !index.is_multiple_of(rate) {
            self.sampled_out.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        self.inner.emit(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailingSink;

    impl TraceSink for FailingSink {
        fn emit(&self, _event: &TraceEvent) -> Result<(), TraceError> {
            Err(TraceError::SinkUnavailable {
                sink: "failing".into(),
            })
        }
    }

    fn event(level: TraceLevel, name: &str) -> TraceEvent {
        TraceEvent::new(level, name)
    }

    #[test]
    fn memory_sink_is_a_bounded_ring() {
        let sink = MemorySink::new(2);
        for name in ["a", "b", "c"] {
            sink.emit(&event(TraceLevel::Info, name)).expect("emit");
        }
        let names: Vec<_> = sink.events().into_iter().map(|e| e.name).collect();
        assert_eq!(names, ["b", "c"]);
        assert_eq!(sink.dropped(), 1);
        assert_eq!(sink.len(), 2);
        sink.clear();
        assert!(sink.is_empty());
        assert_eq!(MemorySink::new(0).capacity(), 1);
    }

    #[test]
    fn fanout_delivers_to_all_and_reports_first_error() {
        let memory = Arc::new(MemorySink::new(8));
        let fanout = FanoutSink::new()
            .with_sink(Arc::new(FailingSink))
            .with_sink(memory.clone());
        let result = fanout.emit(&event(TraceLevel::Info, "x"));
        assert_eq!(result.expect_err("error").code(), "sink_unavailable");
        assert_eq!(memory.len(), 1);
        assert_eq!(fanout.len(), 2);
        assert!(FanoutSink::new()
            .emit(&event(TraceLevel::Info, "x"))
            .is_ok());
    }

    #[test]
    fn sampling_is_deterministic_per_level() {
        let run = || {
            let rates = SamplingRates::KEEP_ALL
                .with_rate(TraceLevel::Debug, 3)
                .with_rate(TraceLevel::Trace, 0);
            let sink = SamplingSink::new(MemorySink::new(64), rates);
            for i in 0..7 {
                sink.emit(&event(TraceLevel::Debug, &format!("d{i}")))
                    .expect("emit");
                sink.emit(&event(TraceLevel::Error, &format!("e{i}")))
                    .expect("emit");
                sink.emit(&event(TraceLevel::Trace, "t")).expect("emit");
            }
            let names: Vec<_> = sink.inner().events().into_iter().map(|e| e.name).collect();
            (names, sink.sampled_out())
        };
        let (names, out) = run();
        let debug: Vec<_> = names.iter().filter(|n| n.starts_with('d')).collect();
        assert_eq!(debug, ["d0", "d3", "d6"]);
        assert_eq!(names.iter().filter(|n| n.starts_with('e')).count(), 7);
        assert!(!names.contains(&"t".to_owned()));
        assert_eq!(out, 4 + 7);
        assert_eq!(run(), (names, out));
    }
}
