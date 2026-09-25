use crate::binomial::wilson_bounds;
use crate::error::StatsError;

/// One observation of an importance-weighted binary outcome. The weight is the
/// inverse of the probability that this unit was observed (for example
/// `1 / calibration_rate` for a randomly audited branch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedOutcome {
    pub weight: f64,
    pub success: bool,
}

/// A Horvitz-Thompson (self-normalised) rate with a Wilson interval computed
/// on the Kish effective sample size rather than the raw count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedRate {
    pub estimate: f64,
    pub effective_n: f64,
    pub low: f64,
    pub high: f64,
}

/// Estimate a rate from importance-weighted observations.
///
/// The interval uses `n_eff = (sum w)^2 / sum w^2`, which shrinks towards the
/// number of heavily weighted units; a Wilson interval on the raw count would
/// be far too narrow when a few audited units stand in for many.
pub fn weighted_rate(observations: &[WeightedOutcome], z: f64) -> Result<WeightedRate, StatsError> {
    if observations.is_empty() {
        return Err(StatsError::Empty {
            field: "observations",
        });
    }
    if !(z.is_finite() && z > 0.0) {
        return Err(StatsError::InvalidParameter {
            field: "z",
            message: "must be finite and positive",
        });
    }
    if observations
        .iter()
        .any(|o| !(o.weight.is_finite() && o.weight > 0.0))
    {
        return Err(StatsError::InvalidParameter {
            field: "weight",
            message: "every weight must be finite and positive",
        });
    }
    let total: f64 = observations.iter().map(|o| o.weight).sum();
    let squares: f64 = observations.iter().map(|o| o.weight * o.weight).sum();
    let successes: f64 = observations
        .iter()
        .filter(|o| o.success)
        .map(|o| o.weight)
        .sum();
    let estimate = successes / total;
    let effective_n = total * total / squares;
    let (low, high) = wilson_bounds(estimate, effective_n, z);
    Ok(WeightedRate {
        estimate,
        effective_n,
        low,
        high,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_weights_reduce_to_the_plain_rate_and_count() {
        let observations: Vec<_> = (0..10)
            .map(|i| WeightedOutcome {
                weight: 2.0,
                success: i < 3,
            })
            .collect();
        let rate = weighted_rate(&observations, 1.96).unwrap();
        assert!((rate.estimate - 0.3).abs() < 1e-12);
        assert!((rate.effective_n - 10.0).abs() < 1e-9);
    }

    #[test]
    fn a_few_heavy_units_shrink_the_effective_sample() {
        let mut observations = vec![
            WeightedOutcome {
                weight: 1.0,
                success: false
            };
            90
        ];
        observations.extend(vec![
            WeightedOutcome {
                weight: 50.0,
                success: true
            };
            2
        ]);
        let rate = weighted_rate(&observations, 1.96).unwrap();
        assert!(rate.effective_n < 10.0, "{}", rate.effective_n);
        assert!(rate.high - rate.low > 0.3);
    }
}
