use crate::core_model::CoreModel;
use crate::error::CoreError;
use crate::tensor::sigmoid;
use ptr_types::Probability;

#[derive(Clone, Debug, PartialEq)]
pub struct CoreVerification {
    pub confidence: Probability,
    pub contradiction_score: Probability,
    pub needs_external_verification: bool,
}

impl CoreModel {
    /// Verifier head: learned probability that the pooled state passes.
    ///
    /// This is a learned score only. It never substitutes for a deterministic
    /// verifier result (Invariant 11): callers must combine it with, not use it
    /// instead of, the deterministic contract.
    pub fn verify_probability(&self, pooled: &[f32]) -> Result<Probability, CoreError> {
        self.check_pooled(pooled)?;
        let logit = self.verifier.forward_vec(pooled)?[0];
        Probability::new(sigmoid(logit).clamp(0.0, 1.0)).ok_or(CoreError::NonFinite {
            what: "verifier probability",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CoreConfig;

    fn model() -> CoreModel {
        CoreModel::new(CoreConfig {
            d_model: 8,
            heads: 2,
            d_ff: 16,
            ..CoreConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn probability_is_in_unit_interval_and_deterministic() {
        let m = model();
        let p = m.verify_probability(&[0.4; 8]).unwrap();
        assert!((0.0..=1.0).contains(&p.get()));
        assert_eq!(p, m.verify_probability(&[0.4; 8]).unwrap());
    }

    #[test]
    fn bias_drives_the_probability_to_the_expected_side() {
        let mut m = model();
        m.verifier
            .w
            .as_mut_slice()
            .iter_mut()
            .for_each(|w| *w = 0.0);
        m.verifier.b.as_mut_slice()[0] = 0.0;
        assert_eq!(m.verify_probability(&[1.0; 8]).unwrap().get(), 0.5);
        m.verifier.b.as_mut_slice()[0] = 10.0;
        assert!(m.verify_probability(&[1.0; 8]).unwrap().get() > 0.999);
        m.verifier.b.as_mut_slice()[0] = -10.0;
        assert!(m.verify_probability(&[1.0; 8]).unwrap().get() < 0.001);
    }

    #[test]
    fn wrong_width_is_refused() {
        assert!(matches!(
            model().verify_probability(&[0.0; 2]),
            Err(CoreError::PooledWidth { .. })
        ));
    }
}
