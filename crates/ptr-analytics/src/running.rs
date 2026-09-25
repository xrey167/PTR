/// Streaming mean and variance by Welford's algorithm: one pass, numerically
/// stable, mergeable across shards with Chan's update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunningMoments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl RunningMoments {
    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }

    /// Combine two independent summaries as if every value had been pushed
    /// into one.
    pub fn merge(&self, other: &Self) -> Self {
        if self.count == 0 {
            return *other;
        }
        if other.count == 0 {
            return *self;
        }
        let count = self.count + other.count;
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * other.count as f64 / count as f64;
        let m2 = self.m2
            + other.m2
            + delta * delta * self.count as f64 * other.count as f64 / count as f64;
        Self { count, mean, m2 }
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
        values.iter().for_each(|v| moments.push(*v));
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
            all.push(*v);
            if i < 2 {
                left.push(*v);
            } else {
                right.push(*v);
            }
        }
        let merged = left.merge(&right);
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
