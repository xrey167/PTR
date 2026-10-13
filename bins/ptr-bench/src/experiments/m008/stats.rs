//! The paired bootstrap of the recall gain, in integers.
//!
//! The gain of arm A over arm B on a seed is `sum(a_i - b_i) / n` over the `n` answerable
//! queries, where `a_i`, `b_i` are 1 when the arm's top fact is the current one. Each query
//! contributes a paired difference in {-1, 0, 1}. The lower bound is the
//! `BOOTSTRAP_LOWER_QUANTILE_PERMILLE` quantile (one-sided 95%) of the means of
//! `BOOTSTRAP_RESAMPLES` resamples with replacement, drawn from a SplitMix64 stream seeded from the
//! pinned bootstrap seed, the run's seed and the comparison's label. Means are reported as integer
//! permille rounded towards minus infinity (`1000 * sum / n` by Euclidean division).

use super::params;
use crate::experiments::rng::Rng;

/// `(point, lower)` of the paired gain in permille.
pub fn paired_gain(diffs: &[i64], seed: u64, label: u64) -> (i64, i64) {
    let n = diffs.len() as i64;
    if n == 0 {
        return (0, 0);
    }
    let point = (1000 * diffs.iter().sum::<i64>()).div_euclid(n);
    let mut rng = Rng::new(params::BOOTSTRAP_SEED ^ seed).fork(label);
    let mut sums: Vec<i64> = (0..params::BOOTSTRAP_RESAMPLES)
        .map(|_| {
            (0..diffs.len())
                .map(|_| diffs[rng.index(diffs.len())])
                .sum()
        })
        .collect();
    sums.sort_unstable();
    let index = params::BOOTSTRAP_RESAMPLES * params::BOOTSTRAP_LOWER_QUANTILE_PERMILLE / 1000;
    let lower = (1000 * sums[index]).div_euclid(n);
    (point, lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_wins_have_a_positive_bound_and_all_ties_a_zero_one() {
        let wins = vec![1_i64; 320];
        assert_eq!(paired_gain(&wins, 17, 0), (1000, 1000));
        let ties = vec![0_i64; 320];
        assert_eq!(paired_gain(&ties, 17, 0), (0, 0));
        let losses = vec![-1_i64; 320];
        assert_eq!(paired_gain(&losses, 17, 0), (-1000, -1000));
        assert_eq!(paired_gain(&[], 17, 0), (0, 0));
    }

    #[test]
    fn a_mixed_sample_has_a_bound_below_its_point_and_the_stream_is_pinned() {
        let mut diffs = vec![1_i64; 60];
        diffs.extend(vec![-1_i64; 20]);
        diffs.extend(vec![0_i64; 240]);
        let (point, lower) = paired_gain(&diffs, 29, 3);
        assert_eq!(point, 125);
        assert!(lower < point && lower > -125, "{lower}");
        assert_eq!(paired_gain(&diffs, 29, 3), (point, lower), "deterministic");
        assert_ne!(
            paired_gain(&diffs, 29, 4).1,
            i64::MIN,
            "another comparison draws its own stream"
        );
    }

    #[test]
    fn rounding_is_towards_minus_infinity() {
        let diffs = vec![-1_i64, 0, 0];
        assert_eq!(paired_gain(&diffs, 1, 0).0, -334);
    }
}
