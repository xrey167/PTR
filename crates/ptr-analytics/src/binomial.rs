use crate::error::StatsError;

/// A binomial proportion with an interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateEstimate {
    pub successes: u64,
    pub trials: u64,
    pub point: f64,
    pub low: f64,
    pub high: f64,
}

/// Wilson score interval for `successes` out of `trials` at normal quantile
/// `z` (1.96 for a two-sided 95% interval).
///
/// Valid for an unweighted binomial proportion with independent trials. It is
/// too narrow for weighted, clustered or censored data; use
/// [`crate::weighted_rate`] for importance-weighted samples.
///
/// # Errors
/// Returns an error for zero trials, successes exceeding trials, or a
/// nonfinite or nonpositive `z`.
pub fn wilson_interval(successes: u64, trials: u64, z: f64) -> Result<RateEstimate, StatsError> {
    if successes > trials {
        return Err(StatsError::InvalidCount { successes, trials });
    }
    if trials == 0 {
        return Err(StatsError::Empty { field: "trials" });
    }
    if !(z.is_finite() && z > 0.0) {
        return Err(StatsError::InvalidParameter {
            field: "z",
            message: "must be finite and positive",
        });
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let (low, high) = wilson_bounds(p, n, z);
    Ok(RateEstimate {
        successes,
        trials,
        point: p,
        low,
        high,
    })
}

pub(crate) fn wilson_bounds(p: f64, n: f64, z: f64) -> (f64, f64) {
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denominator;
    let half = z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt() / denominator;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// One-sided Clopper-Pearson upper bound: the `p` at which observing at most
/// `successes` in `trials` has probability `delta`. With probability at least
/// `1 - delta`, the true rate is below it. Found by bisection on the exact
/// binomial CDF, which is decreasing in `p`.
///
/// Returns `1` when there are no trials or every trial succeeded.
///
/// # Errors
/// Returns an error when successes exceed trials or `delta` is not finite
/// and strictly between zero and one.
pub fn clopper_pearson_upper(successes: u64, trials: u64, delta: f64) -> Result<f64, StatsError> {
    check_delta(delta)?;
    if successes > trials {
        return Err(StatsError::InvalidCount { successes, trials });
    }
    if trials == 0 || successes == trials {
        return Ok(1.0);
    }
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        if binomial_cdf(successes, trials, mid) > delta {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(high)
}

/// `P(X <= k)` for `X ~ Binomial(n, p)`, accumulating probabilities from
/// log-space terms. Very small probabilities can still underflow to zero.
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return if k >= n { 1.0 } else { 0.0 };
    }
    let log_p = p.ln();
    let log_q = (-p).ln_1p();
    let mut log_pmf = n as f64 * log_q;
    let mut total = log_pmf.exp();
    for i in 0..k.min(n) {
        log_pmf += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + log_p - log_q;
        total += log_pmf.exp();
    }
    total.min(1.0)
}

fn check_delta(delta: f64) -> Result<(), StatsError> {
    if delta.is_finite() && delta > 0.0 && delta < 1.0 {
        Ok(())
    } else {
        Err(StatsError::InvalidParameter {
            field: "delta",
            message: "must lie in (0, 1)",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_matches_a_textbook_value() {
        // 8 of 10 at z = 1.96: [0.4902, 0.9433].
        let rate = wilson_interval(8, 10, 1.96).unwrap();
        assert!((rate.low - 0.4902).abs() < 1e-3, "{}", rate.low);
        assert!((rate.high - 0.9433).abs() < 1e-3, "{}", rate.high);
    }

    #[test]
    fn wilson_never_leaves_the_unit_interval() {
        let none = wilson_interval(0, 5, 1.96).unwrap();
        let all = wilson_interval(5, 5, 1.96).unwrap();
        assert_eq!(none.low, 0.0);
        assert_eq!(all.high, 1.0);
    }

    #[test]
    fn clopper_pearson_with_no_failures_solves_the_closed_form() {
        // k = 0: (1 - p)^n = delta.
        let bound = clopper_pearson_upper(0, 50, 0.05).unwrap();
        let exact = 1.0 - 0.05_f64.powf(1.0 / 50.0);
        assert!((bound - exact).abs() < 1e-9, "{bound} {exact}");
        assert_eq!(clopper_pearson_upper(3, 3, 0.05).unwrap(), 1.0);
    }

    #[test]
    fn clopper_pearson_can_find_a_root_below_the_observed_rate() {
        // For one success in two trials, CDF(p) = 1 - p^2.
        for delta in [0.01_f64, 0.5, 0.9, 0.999] {
            let bound = clopper_pearson_upper(1, 2, delta).unwrap();
            assert!((bound - (1.0 - delta).sqrt()).abs() < 1e-12);
        }
    }

    #[test]
    fn the_binomial_cdf_sums_to_one() {
        assert!((binomial_cdf(20, 20, 0.3) - 1.0).abs() < 1e-12);
        assert!((binomial_cdf(0, 3, 0.5) - 0.125).abs() < 1e-12);
    }

    #[test]
    fn impossible_counts_are_refused() {
        assert!(wilson_interval(3, 2, 1.96).is_err());
        assert!(clopper_pearson_upper(3, 2, 0.1).is_err());
        assert!(clopper_pearson_upper(1, 2, 1.5).is_err());
    }
}
