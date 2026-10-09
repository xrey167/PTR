//! The significance tests of the expectation check: exact for a success count, non-asymptotic
//! for a sum of bounded per-task costs. A normal approximation would call a single failure in a
//! cell whose expected number of failures is 0.01 a ten-sigma event; neither test here has that
//! fault, so a correct product is not falsified by a rare event in a saturated cell.

/// `ln(n!)` by summation (exact enough for the counts of a run; `n` is the task count).
fn ln_factorials(n: usize) -> Vec<f64> {
    let mut table = Vec::with_capacity(n + 1);
    let mut sum = 0.0;
    table.push(0.0);
    for i in 1..=n {
        sum += (i as f64).ln();
        table.push(sum);
    }
    table
}

/// The two-sided exact binomial p-value of `observed` events out of `n` with probability `p`:
/// twice the smaller tail, at most one.
pub fn binomial_p_value(n: usize, observed: usize, p: f64) -> f64 {
    assert!(observed <= n, "more events than trials");
    if p <= 0.0 {
        return if observed == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if observed == n { 1.0 } else { 0.0 };
    }
    let ln_fact = ln_factorials(n);
    let (ln_p, ln_q) = (p.ln(), (-p).ln_1p());
    let pmf = |k: usize| -> f64 {
        (ln_fact[n] - ln_fact[k] - ln_fact[n - k] + k as f64 * ln_p + (n - k) as f64 * ln_q).exp()
    };
    let lower: f64 = (0..=observed).map(pmf).sum();
    let upper: f64 = (observed..=n).map(pmf).sum();
    (2.0 * lower.min(upper)).min(1.0)
}

/// The Bernstein bound on `P(|S - E S| >= deviation)` for the sum `S` of `n` independent
/// variables with the same mean, variance `variance` each, and every `|X - E X|` at most
/// `bound`. A bound of one means the observation is as expected or more.
pub fn bernstein_p_value(n: usize, deviation: f64, variance: f64, bound: f64) -> f64 {
    let t = deviation.abs();
    if t < 1e-9 {
        return 1.0;
    }
    let spread = n as f64 * variance;
    if spread < 1e-12 && bound < 1e-12 {
        return 0.0;
    }
    let exponent = t * t / (2.0 * (spread + bound * t / 3.0));
    (2.0 * (-exponent).exp()).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_binomial_p_value_matches_known_values() {
        // Bin(10, 0.5): P(X <= 2) = 56/1024.
        let p = binomial_p_value(10, 2, 0.5);
        assert!((p - 2.0 * 56.0 / 1024.0).abs() < 1e-12, "{p}");
        // The mode is never surprising.
        assert!((binomial_p_value(10, 5, 0.5) - 1.0).abs() < 1e-12);
        // Degenerate probabilities.
        assert_eq!(binomial_p_value(10, 0, 0.0), 1.0);
        assert_eq!(binomial_p_value(10, 1, 0.0), 0.0);
        assert_eq!(binomial_p_value(10, 10, 1.0), 1.0);
        assert_eq!(binomial_p_value(10, 9, 1.0), 0.0);
    }

    #[test]
    fn one_failure_in_a_nearly_saturated_cell_is_not_a_five_sigma_event() {
        // Expected failures 0.01 per 2000 tasks: one failure has probability about 1%.
        let p = binomial_p_value(2000, 1, 0.01 / 2000.0);
        assert!(p > 0.01, "{p}");
        // Thirty failures are not.
        assert!(binomial_p_value(2000, 30, 0.01 / 2000.0) < 1e-12);
    }

    #[test]
    fn a_large_success_shift_is_detected() {
        // 2000 tasks at 0.8: 1400 successes is eleven standard deviations off.
        assert!(binomial_p_value(2000, 1400, 0.8) < 1e-9);
        assert!(binomial_p_value(2000, 1500, 0.8) < 1e-6);
        assert!(binomial_p_value(2000, 1600, 0.8) > 0.5);
    }

    #[test]
    fn the_bernstein_bound_is_valid_and_detects_a_shift() {
        // 2000 tasks, mean 2, variance 1.5, bound 8: a shift of 600 is far outside.
        assert!(bernstein_p_value(2000, 600.0, 1.5, 8.0) < 1e-9);
        // A deviation of two standard deviations is not.
        let sd = (2000.0f64 * 1.5).sqrt();
        assert!(bernstein_p_value(2000, 2.0 * sd, 1.5, 8.0) > 0.01);
        // Three rare long runs in a small cell stay within the bound.
        assert!(bernstein_p_value(120, 23.0, 1.5, 8.0) > 1e-3);
        assert_eq!(bernstein_p_value(10, 0.0, 1.0, 1.0), 1.0);
    }
}
