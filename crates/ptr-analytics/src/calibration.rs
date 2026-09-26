use std::collections::BTreeMap;

use crate::error::StatsError;

/// How predictions are grouped for calibration error.
///
/// Any positive bin count is accepted. Only bins that hold a prediction are
/// ever materialised, so memory grows with the number of predictions, never
/// with the count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Binning {
    /// `n` equal-width confidence intervals over `[0, 1]`.
    EqualWidth(usize),
    /// `n` bins holding (as nearly as possible) the same number of predictions,
    /// ordered by confidence. Less sensitive to where predictions cluster.
    /// With at least as many bins as predictions, each prediction is a bin of
    /// its own.
    EqualMass(usize),
}

/// Multiclass Brier score: mean squared distance between each predicted
/// distribution and the one-hot truth. Lower is better; `0` is perfect.
pub fn brier_score(probabilities: &[Vec<f64>], truth: &[usize]) -> Result<f64, StatsError> {
    check_aligned(probabilities, truth)?;
    let total: f64 = probabilities
        .iter()
        .zip(truth)
        .map(|(p, &t)| {
            p.iter()
                .enumerate()
                .map(|(class, &q)| {
                    let target = if class == t { 1.0 } else { 0.0 };
                    (q - target) * (q - target)
                })
                .sum::<f64>()
        })
        .sum();
    Ok(total / truth.len() as f64)
}

/// Expected calibration error with the top-class probability as confidence.
///
/// Meaningful only on a gold set sampled uniformly from the population the
/// model labels; labels chosen by active learning are a biased sample and
/// overstate or understate calibration.
pub fn expected_calibration_error(
    probabilities: &[Vec<f64>],
    truth: &[usize],
    binning: Binning,
) -> Result<f64, StatsError> {
    check_aligned(probabilities, truth)?;
    let mut scored: Vec<(f64, bool)> = probabilities
        .iter()
        .zip(truth)
        .map(|(p, &t)| {
            let (class, top) = p
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("aligned distributions are non-empty");
            (top, class == t)
        })
        .collect();
    let bins: Vec<Bin> = match binning {
        Binning::EqualWidth(count) => {
            check_bins(count)?;
            let mut bins: BTreeMap<usize, Bin> = BTreeMap::new();
            for (confidence, correct) in scored {
                let bin = ((confidence * count as f64) as usize).min(count - 1);
                bins.entry(bin).or_default().add(confidence, correct);
            }
            bins.into_values().collect()
        }
        Binning::EqualMass(count) => {
            check_bins(count)?;
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            // Bin `b` of `count` holds sorted items `b * n / count` up to
            // `(b + 1) * n / count`. With `count >= n` each holds at most one
            // item, so `n` bins give exactly the non-empty ones; the bounds
            // are computed in u128, where `b * n` cannot overflow.
            let n = scored.len();
            let count = count.min(n);
            let bound = |b: usize| (b as u128 * n as u128 / count as u128) as usize;
            (0..count)
                .map(|b| {
                    let mut bin = Bin::default();
                    for &(confidence, correct) in &scored[bound(b)..bound(b + 1)] {
                        bin.add(confidence, correct);
                    }
                    bin
                })
                .collect()
        }
    };
    let n = truth.len() as f64;
    Ok(bins
        .iter()
        .filter(|bin| bin.size > 0)
        .map(|bin| {
            let size = bin.size as f64;
            let confidence = bin.confidence / size;
            let accuracy = bin.correct as f64 / size;
            (size / n) * (accuracy - confidence).abs()
        })
        .sum())
}

/// The predictions that fell into one calibration bin, as running totals.
#[derive(Clone, Copy, Debug, Default)]
struct Bin {
    size: usize,
    confidence: f64,
    correct: usize,
}

impl Bin {
    fn add(&mut self, confidence: f64, correct: bool) {
        self.size += 1;
        self.confidence += confidence;
        self.correct += usize::from(correct);
    }
}

fn check_bins(count: usize) -> Result<(), StatsError> {
    if count == 0 {
        return Err(StatsError::InvalidParameter {
            field: "bins",
            message: "at least one bin is required",
        });
    }
    Ok(())
}

fn check_aligned(probabilities: &[Vec<f64>], truth: &[usize]) -> Result<(), StatsError> {
    if truth.is_empty() {
        return Err(StatsError::Empty { field: "truth" });
    }
    if probabilities.len() != truth.len() {
        return Err(StatsError::LengthMismatch {
            expected: truth.len(),
            actual: probabilities.len(),
        });
    }
    for (item, (p, &t)) in probabilities.iter().zip(truth).enumerate() {
        let total: f64 = p.iter().sum();
        let valid = !p.is_empty()
            && t < p.len()
            && p.iter().all(|q| q.is_finite() && *q >= 0.0)
            && (total - 1.0).abs() < 1e-6;
        if !valid {
            return Err(StatsError::InvalidDistribution { item });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_perfect_forecast_has_zero_brier_and_zero_ece() {
        let p = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(brier_score(&p, &[0, 1]).unwrap(), 0.0);
        assert_eq!(
            expected_calibration_error(&p, &[0, 1], Binning::EqualWidth(10)).unwrap(),
            0.0
        );
    }

    #[test]
    fn overconfidence_shows_up_under_both_binnings() {
        // Always 90% confident, right half the time: ECE = |0.5 - 0.9| = 0.4.
        let p = vec![vec![0.9, 0.1]; 4];
        for binning in [Binning::EqualWidth(10), Binning::EqualMass(2)] {
            let ece = expected_calibration_error(&p, &[0, 1, 0, 1], binning).unwrap();
            assert!((ece - 0.4).abs() < 1e-12, "{binning:?}: {ece}");
        }
    }

    #[test]
    fn a_distribution_that_does_not_sum_to_one_is_refused() {
        assert_eq!(
            brier_score(&[vec![0.5, 0.2]], &[0]).unwrap_err(),
            StatsError::InvalidDistribution { item: 0 }
        );
    }
}
