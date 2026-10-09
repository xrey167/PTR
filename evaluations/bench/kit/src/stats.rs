//! Quantiles, latency summaries and the repetition protocol.
//!
//! Quantiles use the **nearest-rank** method on integer basis points: for `n`
//! sorted samples and `bp` (1/100 of a percent), the rank is
//! `ceil(n * bp / 10_000)` clamped to `1..=n` and the answer is the sample at
//! that rank. Integer arithmetic avoids `0.95 * 10_000 = 9500.000000000002`
//! style off-by-one ranks, and the result is always an observed sample.

use std::time::Instant;

/// A latency summary needs at least this many samples for p99 to mean anything.
pub const MIN_LATENCY_SAMPLES: usize = 10_000;
/// The protocol requires at least this many measured repetitions after warmup.
pub const MIN_MEASURED_REPETITIONS: usize = 5;

/// Why a summary or repetition plan was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatsError {
    /// No samples at all.
    Empty,
    /// Fewer samples than the summary kind requires.
    TooFewSamples {
        /// Samples supplied.
        got: usize,
        /// Samples required.
        need: usize,
    },
    /// Fewer measured repetitions than the protocol allows.
    TooFewRepetitions {
        /// Measured repetitions requested.
        got: usize,
    },
}

impl std::fmt::Display for StatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "no samples"),
            Self::TooFewSamples { got, need } => {
                write!(f, "{got} samples, at least {need} required")
            }
            Self::TooFewRepetitions { got } => write!(
                f,
                "{got} measured repetitions, at least {MIN_MEASURED_REPETITIONS} required"
            ),
        }
    }
}

impl std::error::Error for StatsError {}

/// Nearest-rank quantile of an ascending slice at `bp` basis points
/// (5000 = median, 9900 = p99). `None` for an empty slice.
pub fn quantile_bp(sorted: &[u64], bp: u32) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let n = sorted.len() as u128;
    let rank = (n * u128::from(bp.min(10_000))).div_ceil(10_000);
    let rank = rank.clamp(1, n) as usize;
    Some(sorted[rank - 1])
}

/// A distribution summary of `u64` samples (nanoseconds for latency).
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Number of samples.
    pub samples: usize,
    /// Smallest sample.
    pub min: u64,
    /// Median (nearest rank).
    pub p50: u64,
    /// 95th percentile (nearest rank).
    pub p95: u64,
    /// 99th percentile (nearest rank).
    pub p99: u64,
    /// Largest sample.
    pub max: u64,
    /// Arithmetic mean.
    pub mean: f64,
}

impl Summary {
    /// Summarise `samples` (any order) requiring at least `min_samples` of them.
    pub fn from_samples(samples: &[u64], min_samples: usize) -> Result<Self, StatsError> {
        if samples.is_empty() {
            return Err(StatsError::Empty);
        }
        if samples.len() < min_samples {
            return Err(StatsError::TooFewSamples {
                got: samples.len(),
                need: min_samples,
            });
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let sum: u128 = sorted.iter().map(|&s| u128::from(s)).sum();
        Ok(Self {
            samples: sorted.len(),
            min: sorted[0],
            p50: quantile_bp(&sorted, 5000).unwrap_or(0),
            p95: quantile_bp(&sorted, 9500).unwrap_or(0),
            p99: quantile_bp(&sorted, 9900).unwrap_or(0),
            max: sorted[sorted.len() - 1],
            mean: sum as f64 / sorted.len() as f64,
        })
    }

    /// A latency summary: refuses fewer than [`MIN_LATENCY_SAMPLES`] samples.
    pub fn latency(samples_ns: &[u64]) -> Result<Self, StatsError> {
        Self::from_samples(samples_ns, MIN_LATENCY_SAMPLES)
    }
}

/// Time one call in nanoseconds (saturating at `u64::MAX`).
pub fn time_ns<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let start = Instant::now();
    let value = f();
    (
        value,
        u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
    )
}

/// Collect `n` per-call latencies of `f`, which receives the call index.
pub fn sample_latencies(n: usize, mut f: impl FnMut(usize)) -> Vec<u64> {
    (0..n).map(|i| time_ns(|| f(i)).1).collect()
}

/// Operations per second for `ops` operations in `elapsed_ns`; `None` when no
/// time elapsed.
pub fn ops_per_second(ops: u64, elapsed_ns: u64) -> Option<f64> {
    (elapsed_ns > 0).then(|| ops as f64 * 1e9 / elapsed_ns as f64)
}

/// Median of `f64` values (mean of the two middle values for even counts);
/// `None` when empty or when any value is NaN.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    })
}

/// The repetition protocol: `warmup` discarded runs, then `measured` kept runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Repetitions {
    /// Discarded warmup runs.
    pub warmup: usize,
    /// Kept runs; at least [`MIN_MEASURED_REPETITIONS`].
    pub measured: usize,
}

impl Repetitions {
    /// The protocol default: one warmup and five measured runs.
    pub const DEFAULT: Self = Self {
        warmup: 1,
        measured: MIN_MEASURED_REPETITIONS,
    };

    /// Validate a plan.
    pub fn new(warmup: usize, measured: usize) -> Result<Self, StatsError> {
        if measured < MIN_MEASURED_REPETITIONS {
            return Err(StatsError::TooFewRepetitions { got: measured });
        }
        Ok(Self { warmup, measured })
    }

    /// Run `f` (given the global run index) `warmup + measured` times and return
    /// only the measured outputs.
    pub fn run<T>(&self, mut f: impl FnMut(usize) -> T) -> Vec<T> {
        let mut kept = Vec::with_capacity(self.measured);
        for i in 0..self.warmup + self.measured {
            let out = f(i);
            if i >= self.warmup {
                kept.push(out);
            }
        }
        kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_to(n: u64) -> Vec<u64> {
        (1..=n).collect()
    }

    #[test]
    fn quantiles_of_one_to_ten_match_hand_computation() {
        let v = one_to(10);
        // rank = ceil(10 * bp / 10000)
        assert_eq!(quantile_bp(&v, 5000), Some(5)); // ceil(5.0) = 5
        assert_eq!(quantile_bp(&v, 9500), Some(10)); // ceil(9.5) = 10
        assert_eq!(quantile_bp(&v, 9900), Some(10)); // ceil(9.9) = 10
        assert_eq!(quantile_bp(&v, 1000), Some(1)); // ceil(1.0) = 1
        assert_eq!(quantile_bp(&v, 1001), Some(2)); // ceil(1.001) = 2
        assert_eq!(quantile_bp(&v, 0), Some(1)); // rank clamps up to 1
        assert_eq!(quantile_bp(&v, 10_000), Some(10));
        assert_eq!(quantile_bp(&v, 99_999), Some(10)); // bp clamps to 100 percent
    }

    #[test]
    fn quantiles_of_one_to_ten_thousand_have_exact_ranks() {
        let v = one_to(10_000);
        // 0.95 * 10_000 is 9500.000000000002 in f64; integer ranks must not drift.
        assert_eq!(quantile_bp(&v, 5000), Some(5000));
        assert_eq!(quantile_bp(&v, 9500), Some(9500));
        assert_eq!(quantile_bp(&v, 9900), Some(9900));
        assert_eq!(quantile_bp(&v, 9990), Some(9990));
    }

    #[test]
    fn quantile_of_a_single_sample_and_of_nothing() {
        assert_eq!(quantile_bp(&[42], 9900), Some(42));
        assert_eq!(quantile_bp(&[42], 0), Some(42));
        assert_eq!(quantile_bp(&[], 5000), None);
    }

    #[test]
    fn quantile_returns_observed_values_of_a_skewed_set() {
        let mut v = vec![1u64; 98];
        v.push(1000);
        v.push(5000);
        // n = 100: p50 rank 50 -> 1; p95 rank 95 -> 1; p99 rank 99 -> 1000.
        assert_eq!(quantile_bp(&v, 5000), Some(1));
        assert_eq!(quantile_bp(&v, 9500), Some(1));
        assert_eq!(quantile_bp(&v, 9900), Some(1000));
        assert_eq!(quantile_bp(&v, 10_000), Some(5000));
    }

    #[test]
    fn summary_sorts_input_and_reports_every_field() {
        let mut v = one_to(100);
        v.reverse();
        let s = Summary::from_samples(&v, 1).unwrap();
        assert_eq!(s.samples, 100);
        assert_eq!((s.min, s.p50, s.p95, s.p99, s.max), (1, 50, 95, 99, 100));
        assert!((s.mean - 50.5).abs() < 1e-12);
    }

    #[test]
    fn summary_refuses_empty_and_too_few_samples() {
        assert_eq!(Summary::from_samples(&[], 0), Err(StatsError::Empty));
        assert_eq!(
            Summary::from_samples(&[1, 2, 3], 4),
            Err(StatsError::TooFewSamples { got: 3, need: 4 })
        );
        assert_eq!(
            Summary::latency(&one_to(9_999)),
            Err(StatsError::TooFewSamples {
                got: 9_999,
                need: 10_000
            })
        );
    }

    #[test]
    fn latency_accepts_exactly_ten_thousand_samples() {
        let s = Summary::latency(&one_to(10_000)).unwrap();
        assert_eq!(
            (s.p50, s.p95, s.p99, s.min, s.max),
            (5000, 9500, 9900, 1, 10_000)
        );
        assert!((s.mean - 5000.5).abs() < 1e-9);
    }

    #[test]
    fn summary_mean_does_not_overflow_on_huge_samples() {
        let s = Summary::from_samples(&[u64::MAX, u64::MAX], 1).unwrap();
        assert!((s.mean - u64::MAX as f64).abs() / (u64::MAX as f64) < 1e-12);
    }

    #[test]
    fn time_ns_returns_the_value_and_a_plausible_duration() {
        let (value, ns) = time_ns(|| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            7
        });
        assert_eq!(value, 7);
        assert!(ns >= 4_000_000, "slept 5ms, measured {ns}ns");
    }

    #[test]
    fn sample_latencies_calls_f_once_per_index_in_order() {
        let mut seen = Vec::new();
        let samples = sample_latencies(25, |i| seen.push(i));
        assert_eq!(samples.len(), 25);
        assert_eq!(seen, (0..25).collect::<Vec<_>>());
        assert!(sample_latencies(0, |_| unreachable!()).is_empty());
    }

    #[test]
    fn ops_per_second_is_exact_and_refuses_zero_time() {
        assert_eq!(ops_per_second(1_000, 1_000_000_000), Some(1_000.0));
        assert_eq!(ops_per_second(3, 500_000_000), Some(6.0));
        assert_eq!(ops_per_second(5, 0), None);
    }

    #[test]
    fn median_handles_odd_even_empty_and_nan() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[9.0]), Some(9.0));
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[1.0, f64::NAN]), None);
    }

    #[test]
    fn repetitions_refuse_fewer_than_five_measured_runs() {
        assert_eq!(
            Repetitions::new(1, 4),
            Err(StatsError::TooFewRepetitions { got: 4 })
        );
        assert_eq!(
            Repetitions::new(0, 5),
            Ok(Repetitions {
                warmup: 0,
                measured: 5
            })
        );
        assert_eq!(
            Repetitions::DEFAULT,
            Repetitions {
                warmup: 1,
                measured: 5
            }
        );
    }

    #[test]
    fn repetitions_discard_warmup_and_keep_measured_outputs() {
        let plan = Repetitions::new(2, 5).unwrap();
        let mut calls = 0;
        let kept = plan.run(|i| {
            calls += 1;
            i * 10
        });
        assert_eq!(calls, 7);
        assert_eq!(kept, vec![20, 30, 40, 50, 60]);
    }

    #[test]
    fn errors_display_their_numbers() {
        assert_eq!(StatsError::Empty.to_string(), "no samples");
        assert_eq!(
            StatsError::TooFewSamples { got: 3, need: 10 }.to_string(),
            "3 samples, at least 10 required"
        );
        assert!(StatsError::TooFewRepetitions { got: 2 }
            .to_string()
            .contains("2 measured"));
    }
}
