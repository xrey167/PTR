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
/// The interval always contains the point estimate, `low <= point <= high`.
/// With no successes `low` is exactly `0`; with every trial a success `high`
/// is exactly `1`.
///
/// # Errors
/// Returns an error for zero trials, successes exceeding trials, or a
/// nonfinite or nonpositive `z`, or one too large for a finite interval
/// (its square overflows).
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
    let (low, high) = wilson_bounds(p, n, z)?;
    Ok(RateEstimate {
        successes,
        trials,
        point: p,
        low,
        high,
    })
}

/// Wilson bounds for a finite proportion `p` in `[0, 1]` on a finite
/// positive `n`.
///
/// Refuses the interval unless its centre and half-width are finite: the
/// clamps to `[0, 1]` would otherwise turn a NaN into the vacuous interval
/// and hide it. With `p` and `n` finite, only a `z` whose square overflows
/// gets there.
///
/// Each bound is clamped to `[0, 1]` and to `p`, so `low <= p <= high`
/// always holds. The exact interval contains `p`, and at `p = 0` its lower
/// bound is `0` (at `p = 1` its upper bound is `1`), which the rounded
/// centre and half-width can miss by a rounding residue. The clamp
/// moves a bound only when rounding has put it on the wrong side of `p`, so
/// `p = 0` gives a lower bound of exactly `0` and `p = 1` an upper bound of
/// exactly `1`.
pub(crate) fn wilson_bounds(p: f64, n: f64, z: f64) -> Result<(f64, f64), StatsError> {
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denominator;
    let half = z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt() / denominator;
    if !(centre.is_finite() && half.is_finite()) {
        return Err(StatsError::InvalidParameter {
            field: "z",
            message: "is too large for a finite interval",
        });
    }
    Ok((
        (centre - half).max(0.0).min(p),
        (centre + half).min(1.0).max(p),
    ))
}

/// One-sided Clopper-Pearson upper bound: the `p` at which observing at most
/// `successes` in `trials` has probability `delta`. With probability at least
/// `1 - delta`, the true rate is below it. Found by 100 bisection steps over
/// `[0, 1]` on [`binomial_cdf`], which is decreasing in `p`; each step
/// evaluates one binomial tail, whose cost [`binomial_cdf`] states.
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
        if cdf(successes, trials, mid) > delta {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(high)
}

/// `P(X <= k)` for `X ~ Binomial(n, p)`.
///
/// Exact where the answer is simple: `1` for `k >= n` or `p = 0`, and `0`
/// for `k < n` at `p = 1`. Otherwise it sums one tail outward from `k`: the
/// lower tail `P(X <= k)` when `k < n p`, else the upper tail `P(X > k)`,
/// returned as its complement; `k` is then at least the median, so the
/// complement is at least one half. `k < n p` is decided exactly, on the
/// integer mantissa of `p`, so no count, however large, picks the other
/// side. The first term is a point probability from Loader's saddle-point
/// expansion, which never subtracts `ln(k!)` and `ln((n - k)!)` from
/// `ln(n!)`; each further term is the one before times the ratio of the two,
/// formed from the counts and `p`, and the sum stops once a geometric bound
/// on the terms left is at most `2^-60` of it. So `k = 0` gives
/// `exp(n ln(1 - p))` and `k = n - 1` (for `n - 1 >= n p`) gives
/// `1 - exp(n ln p)`, one term each; `k = 0` is computed without the ratio
/// `(1 - p) / p`, which overflows for a `p` below `1 / f64::MAX`, so a
/// subnormal `p` gives `exp(n ln(1 - p))` too.
///
/// The number of terms summed grows with the spread `sqrt(n p (1 - p))`, not
/// in proportion to `n` or `k`: near the centre it is about nine standard
/// deviations (some 14,000 terms for `n = 10^7` at `p = 1/2`), fewer in the
/// tails. It is not constant time: for trials in the trillions a tail near
/// the centre sums millions of terms, and near `u64::MAX` billions.
///
/// `1 - p`, the ratio `q / p` (or `p / q`) every term reuses, and the means
/// `n p` and `n q` are carried to about twice `f64` precision, so their
/// rounding does not compound along the tail or into the deviances, and the
/// sum is compensated; each term still rounds on its own. No error bound is
/// proven. Against 70-digit references at 5,995 points for up to `10^7`
/// trials, with `p` from `10^-6` to `1 - 10^-6` and `k` from the centre out
/// into both tails, the largest absolute error was `4.8e-15`, and the
/// largest relative error for tails above `1e-300` was `2.9e-13`; these are
/// measurements on those points, not bounds for other inputs. The relative
/// error of a far tail grows with `-ln` of the tail: the logarithm of its
/// first term (about `-690` near `1e-300`) is formed in `f64`, and each
/// rounding of it moves the tail by up to about `690 2^-53` of itself. The
/// absolute error grows with the number of terms. The tests pin twelve of
/// these references, seven of them to `1e-14` absolute near the centre or
/// `5e-13` relative in the tails and the other five to `1e-13` relative.
/// Counts above `2^53` are rounded to the nearest `f64` in the arithmetic
/// (though not in choosing the side), and no figure above covers them. A
/// tail too small for an `f64` underflows to zero.
///
/// # Errors
/// Returns `StatsError::InvalidParameter` unless `p` is a probability in
/// `[0, 1]`. A NaN, infinite or out-of-range `p` has no binomial
/// distribution; answering for the nearest endpoint instead would report a
/// confident `0` or `1` for it.
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> Result<f64, StatsError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::InvalidParameter {
            field: "p",
            message: "must lie in [0, 1]",
        });
    }
    Ok(cdf(k, n, p))
}

/// [`binomial_cdf`] for a `p` already known to lie in `[0, 1]`.
fn cdf(k: u64, n: u64, p: f64) -> f64 {
    cdf_and_terms(k, n, p).0
}

/// [`binomial_cdf`] and the number of point probabilities it summed.
fn cdf_and_terms(k: u64, n: u64, p: f64) -> (f64, u64) {
    if k >= n || p <= 0.0 {
        return (1.0, 0);
    }
    if p >= 1.0 {
        return (0.0, 0);
    }
    let q = complement(p);
    let (value, terms) = if k == 0 {
        // P(X <= 0) = (1 - p)^n, one term and no ratio. k = 0 is below the
        // mean n p > 0, but the ratio q / p that side reuses is not formed
        // for it: below 1 / f64::MAX (about 5.56e-309) it overflows, and its
        // infinite high part made the low part, and the value, NaN.
        (log_pmf(0, n, p, q).exp(), 1)
    } else if below_mean(k, n, p) {
        // P(X = i - 1) / P(X = i) = i q / ((n - i + 1) p), from i = k down.
        // 1 <= k < n p gives p > 1 / n >= 2^-64, so q / p < 2^64 is finite.
        let odds = quotient(q, (p, 0.0));
        let (sum, terms) = relative_tail(
            (1..=k)
                .rev()
                .map(|i| i as f64 / (n - i + 1) as f64 * odds.0),
            odds.1 / odds.0,
        );
        ((log_pmf(k, n, p, q) + sum.ln()).exp(), terms)
    } else {
        // P(X = i + 1) / P(X = i) = (n - i) p / ((i + 1) q), from i = k + 1 up.
        // p < 1 leaves q >= 2^-53, so p / q <= 2^53 is finite.
        let odds = quotient((p, 0.0), q);
        let (sum, terms) = relative_tail(
            (k + 1..n).map(|i| (n - i) as f64 / (i + 1) as f64 * odds.0),
            odds.1 / odds.0,
        );
        (1.0 - (log_pmf(k + 1, n, p, q) + sum.ln()).exp(), terms)
    };
    // Only rounding takes the value outside [0, 1]; a clamp, unlike `min`,
    // would still pass a NaN through rather than turn it into a certainty.
    debug_assert!(!value.is_nan(), "P(X <= {k}) of {n} at {p:e} is NaN");
    (value.clamp(0.0, 1.0), terms)
}

/// Whether `k < n p` in exact arithmetic, for `0 < p < 1`: the side
/// [`binomial_cdf`] sums the lower tail on. `p` is `m 2^-s` for an integer
/// mantissa `m < 2^53` and `s >= 53`, so the comparison is `k 2^s < n m`,
/// decided on the 117-bit product `n m` without rounding any count.
fn below_mean(k: u64, n: u64, p: f64) -> bool {
    let bits = p.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1 << 52) - 1);
    let (mantissa, shift) = if exponent == 0 {
        (fraction, 1074)
    } else {
        (fraction | (1 << 52), 1075 - exponent)
    };
    let product = u128::from(n) * u128::from(mantissa);
    if shift >= 128 {
        // n p < 2^117 / 2^128 < 1: only k = 0 lies below it.
        return k == 0 && product > 0;
    }
    let whole = product >> shift;
    let fractional = product & ((1_u128 << shift) - 1) != 0;
    u128::from(k) < whole || (u128::from(k) == whole && fractional)
}

/// A number held as the unevaluated sum `hi + lo` of two `f64`, `lo` far
/// below an ulp of `hi`: about twice the precision of either.
type Pair = (f64, f64);

/// `1 - p` for `0 <= p <= 1`, exactly: `1 - p` rounded, and the rounding
/// error, which is an `f64` (Dekker's two-sum, valid since `1 >= p`).
fn complement(p: f64) -> Pair {
    let hi = 1.0 - p;
    (hi, (1.0 - hi) - p)
}

/// `a / b` for positive pairs with a finite quotient, to a relative error
/// near `2^-104`: the quotient rounded, and the remainder of that rounding
/// divided by `b`. `mul_add` makes `a.0 - hi b.0` exact.
fn quotient(a: Pair, b: Pair) -> Pair {
    let hi = a.0 / b.0;
    (hi, (hi.mul_add(-b.0, a.0) + a.1 - hi * b.1) / b.0)
}

/// `n factor` as a pair: exact when `n <= 2^53`, `factor.1` is zero and the
/// residue does not underflow (the error of the rounded product is what
/// `mul_add` leaves), else to about twice the precision of an `f64`.
fn product(n: u64, factor: Pair) -> Pair {
    let total = n as f64;
    let hi = total * factor.0;
    (hi, total.mul_add(factor.0, -hi) + total * factor.1)
}

/// Relative size, `2^-60`, below which the terms left of a tail are dropped.
const TAIL_TOLERANCE: f64 = 1.0 / (1_u64 << 60) as f64;

/// Sums the terms of a binomial tail relative to its first, given the ratio
/// of each term to the one before it, and returns the sum and the number of
/// terms. Along either tail these ratios never grow, so after a term `t`
/// whose next ratio is `r < 1` the terms left sum to at most `t r / (1 - r)`;
/// the sum stops once that is at most [`TAIL_TOLERANCE`] of it.
///
/// Each ratio is off by the same relative `drift` every time: it multiplies
/// only the high part of a pair (`q / p` or `p / q`), because the low part
/// is below half an ulp of the product and would be lost if added to it.
/// Term `j` is then off by `(1 + drift)^j`, a bias that compounds along the
/// tail (about `sqrt(n p q)` ulps of the sum when it was left in). The sum
/// returned adds `drift` times the terms' first moment, `sum_j j t_j`, which
/// removes it to first order; with `|drift| <= 2^-52`, the second-order
/// remainder, about `(j drift)^2 / 2` of term `j`, is below `2^-58` of it
/// for the first `10^7` terms.
///
/// The sum is compensated: every term is at most the first, so at most the
/// sum, and the rounding error of each addition, `(sum - next) + term`, is
/// exact and kept in `carry` (Kahan and Babuska). Uncompensated, those
/// roundings were the larger part of the error near the centre, where some
/// ten thousand terms are added.
fn relative_tail(ratios: impl Iterator<Item = f64>, drift: f64) -> (f64, u64) {
    let (mut sum, mut carry, mut moment) = (1.0_f64, 0.0_f64, 0.0_f64);
    let (mut term, mut terms) = (1.0_f64, 1_u64);
    for ratio in ratios {
        if ratio < 1.0 && term * ratio <= TAIL_TOLERANCE * sum * (1.0 - ratio) {
            break;
        }
        term *= ratio;
        let next = sum + term;
        carry += (sum - next) + term;
        sum = next;
        moment += terms as f64 * term;
        terms += 1;
    }
    (sum + (carry + drift * moment), terms)
}

/// `ln P(X = x)` for `X ~ Binomial(n, p)` with `0 < p < 1`, `q = 1 - p` as
/// a pair and `x <= n`, by Loader's saddle-point expansion ("Fast and
/// Accurate Computation of Binomial Probabilities", 2000). It is built from
/// Stirling errors and deviances, so it never subtracts `ln(x!)` and
/// `ln((n - x)!)` from `ln(n!)`, a cancellation that grows with `n`. The
/// means `n p` and `n q` enter the deviances as pairs: rounded to an `f64`,
/// each moved a deviance by up to `|x - mean| 2^-53`, which in a far tail
/// is the relative error of the result.
fn log_pmf(x: u64, n: u64, p: f64, q: Pair) -> f64 {
    let total = n as f64;
    if x == 0 {
        return total * (-p).ln_1p();
    }
    if x == n {
        return total * p.ln();
    }
    let (hits, misses) = (x as f64, (n - x) as f64);
    stirling_error(n)
        - stirling_error(x)
        - stirling_error(n - x)
        - deviance(hits, product(n, (p, 0.0)))
        - deviance(misses, product(n, q))
        - 0.5 * (LN_TAU + hits.ln() + (misses / total).ln())
}

/// `ln(2 pi)`.
const LN_TAU: f64 = 1.8378770664093453;

/// `ln(m!) - ln(sqrt(2 pi m) (m / e)^m)`, the error of Stirling's formula:
/// tabulated to 15, then its asymptotic series cut where Loader cuts it,
/// which keeps its error below `1.1e-16` (largest at `m = 16`).
fn stirling_error(m: u64) -> f64 {
    const TABLE: [f64; 16] = [
        0.0,
        0.08106146679532726,
        0.0413406959554093,
        0.02767792568499834,
        0.020790672103765093,
        0.016644691189821193,
        0.013876128823070748,
        0.01189670994589177,
        0.010411265261972096,
        0.009255462182712733,
        0.00833056343336287,
        0.007573675487951841,
        0.00694284010720953,
        0.006408994188004207,
        0.0059513701127588475,
        0.005554733551962801,
    ];
    const S0: f64 = 1.0 / 12.0;
    const S1: f64 = 1.0 / 360.0;
    const S2: f64 = 1.0 / 1260.0;
    const S3: f64 = 1.0 / 1680.0;
    const S4: f64 = 1.0 / 1188.0;
    if m <= 15 {
        return TABLE[m as usize];
    }
    let m = m as f64;
    let square = m * m;
    if m > 500.0 {
        (S0 - S1 / square) / m
    } else if m > 80.0 {
        (S0 - (S1 - S2 / square) / square) / m
    } else if m > 35.0 {
        (S0 - (S1 - (S2 - S3 / square) / square) / square) / m
    } else {
        (S0 - (S1 - (S2 - (S3 - S4 / square) / square) / square) / square) / m
    }
}

/// The deviance `x ln(x / mean) + mean - x` for `x > 0` and a positive
/// `mean` given as a pair: at `mean.0`, plus its derivative in the mean,
/// `1 - x / mean.0`, times `mean.1`. The next term, `x (mean.1 / mean.0)^2
/// / 2`, is below `x 2^-104`.
fn deviance(x: f64, mean: Pair) -> f64 {
    deviance_at(x, mean.0) + (mean.0 - x) * (mean.1 / mean.0)
}

/// The deviance `x ln(x / mean) + mean - x` for `x > 0` and `mean > 0`, by a
/// series in `v = (x - mean) / (x + mean)` while `|v| < 1/2`, where the
/// direct form cancels: at `x = 1.5 mean` it subtracts `0.5 mean` from
/// `0.61 mean`, and the rounding of `x / mean` alone can move the result by
/// some seven ulps. The series converges as `v^2`, so it takes at most about
/// thirty terms.
fn deviance_at(x: f64, mean: f64) -> f64 {
    if (x - mean).abs() < 0.5 * (x + mean) {
        let v = (x - mean) / (x + mean);
        let square = v * v;
        let mut sum = (x - mean) * v;
        let mut power = 2.0 * x * v;
        for j in 1..1000_u32 {
            power *= square;
            let next = sum + power / f64::from(2 * j + 1);
            if next == sum {
                return sum;
            }
            sum = next;
        }
    }
    x * (x / mean).ln() + mean - x
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
        assert!((binomial_cdf(20, 20, 0.3).unwrap() - 1.0).abs() < 1e-12);
        assert!((binomial_cdf(0, 3, 0.5).unwrap() - 0.125).abs() < 1e-12);
    }

    /// The summation this module used before it summed the shorter tail:
    /// every term from `(1 - p)^n` up to `k`, accumulated in log space.
    fn summed_from_zero(k: u64, n: u64, p: f64) -> f64 {
        let log_p = p.ln();
        let log_q = (-p).ln_1p();
        let mut log_pmf = n as f64 * log_q;
        let mut total = log_pmf.exp();
        for i in 0..k.min(n) {
            log_pmf += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + log_p - log_q;
            total += log_pmf.exp();
        }
        total.clamp(0.0, 1.0)
    }

    #[test]
    fn the_cdf_agrees_with_the_summation_from_zero_on_small_samples() {
        let probabilities = [
            1e-6,
            0.01,
            0.1,
            0.123_456_789,
            0.25,
            1.0 / 3.0,
            0.5,
            0.7,
            0.9,
            0.99,
            1.0 - 1e-6,
        ];
        for n in 1..=60_u64 {
            for p in probabilities {
                for k in 0..=n {
                    let (tail, summed) = (cdf(k, n, p), summed_from_zero(k, n, p));
                    assert!(
                        (tail - summed).abs() <= 1e-15 + 1e-12 * summed,
                        "{k} of {n} at {p}: {tail} {summed}"
                    );
                }
                // No successes is the one term both sum, to the bit.
                assert_eq!(cdf(0, n, p), summed_from_zero(0, n, p));
            }
        }
    }

    #[test]
    fn the_cdf_sums_terms_in_proportion_to_its_spread_not_its_count() {
        for n in [1_000_000_u64, 10_000_000, 1_000_000_000] {
            for p in [0.5, 0.1, 0.01, 0.999] {
                let spread = (n as f64 * p * (1.0 - p)).sqrt();
                for step in -24..=24 {
                    let k = (n as f64 * p + f64::from(step) * spread / 4.0).round() as u64;
                    let (_, terms) = cdf_and_terms(k, n, p);
                    assert!(
                        terms as f64 <= 10.0 * spread + 64.0,
                        "{k} of {n} at {p}: {terms} terms, spread {spread}"
                    );
                }
            }
        }
        // At most one failure in millions takes one term above the mean and a
        // handful below it, where the summation from zero took one per trial.
        assert_eq!(
            cdf_and_terms(4_999_999, 5_000_000, 0.5),
            (1.0 - 0.5_f64.powi(5_000_000), 1)
        );
        let (tail, terms) = cdf_and_terms(4_999_999, 5_000_000, 1.0 - 1e-8);
        assert!(terms <= 16, "{terms}");
        assert!(
            (tail - (1.0 - (1.0 - 1e-8_f64).powf(5e6))).abs() < 1e-15,
            "{tail}"
        );
        assert_eq!(cdf_and_terms(0, 5_000_000, 1e-3).1, 1);
    }

    #[test]
    fn the_tail_is_chosen_by_comparing_k_with_n_p_exactly_at_any_count() {
        let (n, p) = (1_u64 << 60, 0.25);
        // n p = 2^58 exactly, and 2^58 - 1 rounds to 2^58 as an f64.
        assert!(below_mean((1 << 58) - 1, n, p));
        assert!(!below_mean(1 << 58, n, p));
        assert!(!below_mean((1 << 58) + 1, n, p));
        // The same on the side above half the trials, for p near one.
        let p = 0.75;
        assert!(below_mean(3 * (1 << 58) - 1, n, p));
        assert!(!below_mean(3 * (1 << 58), n, p));
        // n p = 3 (2^64 - 1) / 4 is not a whole number: its floor is below
        // the mean, the next count is not.
        let n = u64::MAX;
        let floor = u64::MAX / 4 * 3 + 2;
        assert!(below_mean(floor, n, p));
        assert!(!below_mean(floor + 1, n, p));
        // The f64 0.01 lies above 1/100, so 100 of it is above one although
        // the product rounds to exactly one; the f64 1/3 lies below 1/3.
        assert!(below_mean(1, 100, 0.01));
        assert!(below_mean(1, 10, 0.1));
        assert!(!below_mean(1, 3, 1.0 / 3.0));
        assert!(below_mean(0, 1, f64::from_bits(1)));
    }

    #[test]
    fn impossible_counts_are_refused() {
        assert!(wilson_interval(3, 2, 1.96).is_err());
        assert!(clopper_pearson_upper(3, 2, 0.1).is_err());
        assert!(clopper_pearson_upper(1, 2, 1.5).is_err());
    }
}
