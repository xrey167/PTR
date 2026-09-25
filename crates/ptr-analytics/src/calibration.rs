use crate::error::StatsError;

/// How predictions are grouped for calibration error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Binning {
    /// `n` equal-width confidence intervals over `[0, 1]`.
    EqualWidth(usize),
    /// `n` bins holding (as nearly as possible) the same number of predictions,
    /// ordered by confidence. Less sensitive to where predictions cluster.
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
    let bins: Vec<Vec<(f64, bool)>> = match binning {
        Binning::EqualWidth(count) => {
            check_bins(count)?;
            let mut bins = vec![Vec::new(); count];
            for (confidence, correct) in scored {
                let bin = ((confidence * count as f64) as usize).min(count - 1);
                bins[bin].push((confidence, correct));
            }
            bins
        }
        Binning::EqualMass(count) => {
            check_bins(count)?;
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            let n = scored.len();
            (0..count)
                .map(|b| scored[b * n / count..(b + 1) * n / count].to_vec())
                .collect()
        }
    };
    let n = truth.len() as f64;
    Ok(bins
        .iter()
        .filter(|bin| !bin.is_empty())
        .map(|bin| {
            let size = bin.len() as f64;
            let confidence: f64 = bin.iter().map(|(c, _)| c).sum::<f64>() / size;
            let accuracy = bin.iter().filter(|(_, correct)| *correct).count() as f64 / size;
            (size / n) * (accuracy - confidence).abs()
        })
        .sum())
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
