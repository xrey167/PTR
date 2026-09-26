use crate::error::StatsError;

/// Streaming mean and variance by Welford's algorithm: one pass, numerically
/// stable, mergeable across shards with Chan's update.
///
/// The mean and the sum of squared deviations are always finite: an update
/// that would make either nonfinite is refused and leaves the summary as it
/// was.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunningMoments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningMoments {
    /// Include one observation in the count, mean, and variance.
    ///
    /// # Errors
    /// Returns `StatsError::InvalidParameter` for a nonfinite value, or a
    /// finite one whose squared deviation overflows the sum of squares (for
    /// example `f64::MAX` after `-f64::MAX`); the summary is unchanged.
    pub fn push(&mut self, value: f64) -> Result<(), StatsError> {
        let count = self.count + 1;
        let delta = value - self.mean;
        let mean = self.mean + delta / count as f64;
        let m2 = self.m2 + delta * (value - mean);
        *self = Self::checked(count, mean, m2, "value")?;
        Ok(())
    }

    /// Combine two independent summaries as if every value had been pushed
    /// into one.
    ///
    /// Chan's cross term `delta^2 * n1 * n2 / n` is evaluated as
    /// `delta * (delta * (n1 * (n2 / n)))`: the count factor is at most `n1`,
    /// and `delta` times it is at most the larger of `n1` and the term, so no
    /// intermediate overflows unless the term itself does.
    ///
    /// # Errors
    /// Returns `StatsError::InvalidParameter` when the combined mean or sum
    /// of squared deviations would not be finite.
    pub fn merge(&self, other: &Self) -> Result<Self, StatsError> {
        if self.count == 0 {
            return Ok(*other);
        }
        if other.count == 0 {
            return Ok(*self);
        }
        let count = self.count + other.count;
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * other.count as f64 / count as f64;
        let factor = self.count as f64 * (other.count as f64 / count as f64);
        let m2 = self.m2 + other.m2 + delta * (delta * factor);
        Self::checked(count, mean, m2, "summary")
    }

    fn checked(count: u64, mean: f64, m2: f64, field: &'static str) -> Result<Self, StatsError> {
        if !(mean.is_finite() && m2.is_finite()) {
            return Err(StatsError::InvalidParameter {
                field,
                message: "would make the mean or the sum of squared deviations non-finite",
            });
        }
        Ok(Self { count, mean, m2 })
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// `None` before the first value.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Unbiased sample variance; `None` below two values.
    pub fn sample_variance(&self) -> Option<f64> {
        (self.count > 1).then(|| self.m2 / (self.count - 1) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_match_the_two_pass_definition() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut moments = RunningMoments::default();
        for value in values {
            moments.push(value).unwrap();
        }
        assert_eq!(moments.mean(), Some(5.0));
        assert!((moments.sample_variance().unwrap() - 32.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn merging_shards_equals_one_pass() {
        let values = [1.0, 3.0, 8.0, 13.0, 21.0, 34.0];
        let mut all = RunningMoments::default();
        let mut left = RunningMoments::default();
        let mut right = RunningMoments::default();
        for (i, v) in values.iter().enumerate() {
            all.push(*v).unwrap();
            if i < 2 {
                left.push(*v).unwrap();
            } else {
                right.push(*v).unwrap();
            }
        }
        let merged = left.merge(&right).unwrap();
        assert_eq!(merged.count(), all.count());
        assert!((merged.mean().unwrap() - all.mean().unwrap()).abs() < 1e-12);
        assert!((merged.sample_variance().unwrap() - all.sample_variance().unwrap()).abs() < 1e-9);
    }

    #[test]
    fn an_empty_summary_has_no_mean() {
        assert_eq!(RunningMoments::default().mean(), None);
        assert_eq!(RunningMoments::default().sample_variance(), None);
    }
}
