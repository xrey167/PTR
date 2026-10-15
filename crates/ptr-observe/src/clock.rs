use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Monotonic tick source. One tick is one nanosecond; the origin is arbitrary, so ticks are only
/// meaningful as differences (it is not wall-clock time).
pub trait Clock: Send + Sync {
    fn now_tick(&self) -> u64;
}

/// Manually driven clock for deterministic tests and replay.
#[derive(Debug, Default)]
pub struct ManualClock {
    tick: AtomicU64,
}

impl ManualClock {
    pub fn new(start: u64) -> Self {
        Self {
            tick: AtomicU64::new(start),
        }
    }

    pub fn set(&self, tick: u64) {
        self.tick.store(tick, Ordering::SeqCst);
    }

    /// Advances the clock, saturating at `u64::MAX`.
    pub fn advance(&self, delta: u64) {
        let _ = self
            .tick
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |t| {
                Some(t.saturating_add(delta))
            });
    }
}

impl Clock for ManualClock {
    fn now_tick(&self) -> u64 {
        self.tick.load(Ordering::SeqCst)
    }
}

/// `std::time::Instant`-backed monotonic clock; ticks are nanoseconds since construction.
#[derive(Debug)]
pub struct MonotonicClock {
    origin: Instant,
}

impl MonotonicClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_tick(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_sets_and_advances_with_saturation() {
        let clock = ManualClock::new(5);
        clock.advance(10);
        assert_eq!(clock.now_tick(), 15);
        clock.set(u64::MAX - 1);
        clock.advance(10);
        assert_eq!(clock.now_tick(), u64::MAX);
    }

    #[test]
    fn monotonic_clock_never_goes_backwards() {
        let clock = MonotonicClock::new();
        let first = clock.now_tick();
        assert!(clock.now_tick() >= first);
    }
}
