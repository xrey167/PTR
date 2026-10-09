//! The harness's own binomial arithmetic. It shares no code with `ptr_analytics`: the product's
//! Clopper-Pearson bound is checked against the bisection written here, and the held-out tests
//! and the critical value of the coverage test are decided here.
//!
//! Every probability is a sum of point masses computed in log space from a table of
//! `ln(i)` sums, so a count in the thousands loses nothing to overflow. The sums are direct
//! (no recurrence from the mode), which is slower than the product's and needs no argument for
//! its accuracy: the longest sum in a run has a few thousand terms.

use std::sync::OnceLock;

/// The largest number of trials any function here is called with.
pub const MAX_TRIALS: usize = 20_000;

fn ln_factorials() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Vec::with_capacity(MAX_TRIALS + 1);
        let mut sum = 0.0_f64;
        table.push(0.0);
        for i in 1..=MAX_TRIALS {
            sum += (i as f64).ln();
            table.push(sum);
        }
        table
    })
}

/// `ln P(X = k)` for `X ~ Binomial(n, p)`, `0 < p < 1`, `k <= n <= MAX_TRIALS`.
fn ln_pmf(k: usize, n: usize, p: f64) -> f64 {
    let table = ln_factorials();
    table[n] - table[k] - table[n - k] + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()
}

/// `P(X <= k)`, the lower tail, for `X ~ Binomial(n, p)`.
pub fn cdf_le(k: usize, n: usize, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    (0..=k).map(|x| ln_pmf(x, n, p).exp()).sum::<f64>().min(1.0)
}

/// `P(X >= k)`, the upper tail, summed directly from the top so that a small tail keeps its
/// relative accuracy.
pub fn tail_ge(k: usize, n: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    (k..=n)
        .rev()
        .map(|x| ln_pmf(x, n, p).exp())
        .sum::<f64>()
        .min(1.0)
}

/// The one-sided Clopper-Pearson upper bound with confidence `1 - delta`: the `p` at which
/// `P(X <= k) = delta`, found by bisection. One when there are no trials or all succeeded.
pub fn cp_upper(k: usize, n: usize, delta: f64) -> f64 {
    if n == 0 || k >= n {
        return 1.0;
    }
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        if cdf_le(k, n, mid) > delta {
            low = mid;
        } else {
            high = mid;
        }
    }
    high
}

/// The one-sided Clopper-Pearson lower bound with confidence `1 - delta`: the `p` at which
/// `P(X >= k) = delta`. Zero when nothing succeeded.
pub fn cp_lower(k: usize, n: usize, delta: f64) -> f64 {
    if n == 0 || k == 0 {
        return 0.0;
    }
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        if tail_ge(k, n, mid) < delta {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

/// Whether the Learn-then-Test test of one threshold passes: the upper bound of `k` harmful
/// among `n` admitted is at most `alpha` at confidence `1 - delta`. Decided without a bound:
/// the bound is at most `alpha` iff the lower tail at `alpha` is at most `delta`.
pub fn ltt_passes(k: usize, n: usize, alpha: f64, delta: f64) -> bool {
    n > 0 && cdf_le(k, n, alpha) <= delta
}

/// The exact one-sided p-value of "the harm rate is at most `alpha`" given `k` harmful among
/// `n`: `P(Bin(n, alpha) >= k)`.
pub fn upper_p_value(k: usize, n: usize, alpha: f64) -> f64 {
    tail_ge(k, n, alpha)
}

/// The largest count `c` such that `P(Bin(replicates, delta) >= c) >= level`: a count of
/// coverage violations above it has probability below `level` if the guarantee holds.
pub fn coverage_critical(replicates: usize, delta: f64, level: f64) -> usize {
    let mut critical = 0;
    for c in 0..=replicates {
        if tail_ge(c, replicates, delta) >= level {
            critical = c;
        } else {
            break;
        }
    }
    critical
}

/// Mean and sample standard deviation of a list.
pub fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_closed_forms_hold() {
        // k = 0: the bound is 1 - delta^(1/n).
        for n in [1_usize, 5, 59, 300] {
            let closed = 1.0 - 0.05_f64.powf(1.0 / n as f64);
            assert!((cp_upper(0, n, 0.05) - closed).abs() < 1e-12, "{n}");
        }
        assert_eq!(cp_upper(7, 7, 0.05), 1.0);
        assert_eq!(cp_upper(0, 0, 0.05), 1.0);
        // The lower bound mirrors it: k = n gives delta^(1/n).
        for n in [1_usize, 5, 59] {
            let closed = 0.05_f64.powf(1.0 / n as f64);
            assert!((cp_lower(n, n, 0.05) - closed).abs() < 1e-12, "{n}");
        }
        assert_eq!(cp_lower(0, 10, 0.05), 0.0);
    }

    #[test]
    fn fifty_nine_clean_samples_is_the_fewest_that_certify() {
        assert!(!ltt_passes(0, 58, 0.05, 0.05));
        assert!(ltt_passes(0, 59, 0.05, 0.05));
        assert!(!ltt_passes(1, 59, 0.05, 0.05));
    }

    #[test]
    fn the_bound_and_the_tail_test_agree() {
        for (k, n) in [(0, 100), (1, 100), (3, 150), (4, 152), (9, 346), (20, 600)] {
            let by_bound = cp_upper(k, n, 0.05) <= 0.05;
            assert_eq!(by_bound, ltt_passes(k, n, 0.05, 0.05), "{k}/{n}");
        }
    }

    #[test]
    fn tails_sum_to_one() {
        for (k, n, p) in [(3, 10, 0.3), (0, 5, 0.5), (49, 100, 0.5)] {
            let total = cdf_le(k, n, p) + tail_ge(k + 1, n, p);
            assert!((total - 1.0).abs() < 1e-12, "{k} {n} {p}");
        }
        assert_eq!(tail_ge(0, 4, 0.2), 1.0);
        assert_eq!(tail_ge(5, 4, 0.2), 0.0);
    }

    #[test]
    fn the_critical_value_is_the_edge_of_the_level() {
        let c = coverage_critical(2000, 0.05, 0.001);
        assert!(tail_ge(c, 2000, 0.05) >= 0.001);
        assert!(tail_ge(c + 1, 2000, 0.05) < 0.001);
        assert!((120..150).contains(&c), "{c}");
    }

    #[test]
    fn mean_and_sd() {
        let (m, s) = mean_sd(&[1.0, 2.0, 3.0, 4.0]);
        assert!((m - 2.5).abs() < 1e-12);
        assert!((s - 1.2909944487358056).abs() < 1e-12);
    }
}
