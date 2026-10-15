use crate::core_model::{CoreModel, CODEBOOK};
use crate::error::CoreError;
use crate::tensor::softmax;
use ptr_types::{CognitiveType, Probability};

pub use ptr_types::ReasoningOperator;

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedOperator {
    pub operator: ReasoningOperator,
    pub weight: Probability,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterDecision {
    pub operators: Vec<WeightedOperator>,
}

impl RouterDecision {
    pub fn top(&self) -> Option<ReasoningOperator> {
        self.operators
            .iter()
            .max_by(|a, b| a.weight.get().total_cmp(&b.weight.get()))
            .map(|x| x.operator)
    }
}

/// The router head's operator classes in ptr-types code order.
pub fn operator_classes() -> &'static [ReasoningOperator] {
    ReasoningOperator::table(CODEBOOK.version()).unwrap_or(&[])
}

impl CoreModel {
    /// Router logits for a pooled state (`d_model` wide).
    pub fn router_logits(&self, pooled: &[f32]) -> Result<Vec<f32>, CoreError> {
        self.check_pooled(pooled)?;
        Ok(self.router.forward_vec(pooled)?)
    }

    /// Softmax over the router logits, in [`operator_classes`] order.
    pub fn router_probabilities(&self, pooled: &[f32]) -> Result<Vec<f32>, CoreError> {
        let p = softmax(&self.router_logits(pooled)?);
        if p.iter().all(|x| x.is_finite()) {
            Ok(p)
        } else {
            Err(CoreError::NonFinite {
                what: "router probabilities",
            })
        }
    }

    /// Operator router head: pooled state to a weighted operator decision.
    pub fn route(&self, pooled: &[f32]) -> Result<RouterDecision, CoreError> {
        let probs = self.router_probabilities(pooled)?;
        let operators = operator_classes()
            .iter()
            .zip(probs)
            .map(|(operator, p)| {
                Probability::new(p.clamp(0.0, 1.0))
                    .map(|weight| WeightedOperator {
                        operator: *operator,
                        weight,
                    })
                    .ok_or(CoreError::NonFinite {
                        what: "router weight",
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RouterDecision { operators })
    }

    pub(crate) fn check_pooled(&self, pooled: &[f32]) -> Result<(), CoreError> {
        if pooled.len() != self.config.d_model {
            return Err(CoreError::PooledWidth {
                expected: self.config.d_model,
                found: pooled.len(),
            });
        }
        if !pooled.iter().all(|v| v.is_finite()) {
            return Err(CoreError::NonFinite { what: "pooled" });
        }
        Ok(())
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
    fn classes_cover_the_whole_operator_codebook() {
        assert_eq!(
            operator_classes().len(),
            usize::from(CODEBOOK.cardinality_of::<ReasoningOperator>())
        );
        assert_eq!(operator_classes()[0], ReasoningOperator::Semantic);
    }

    #[test]
    fn route_is_a_distribution_over_every_operator() {
        let m = model();
        let d = m.route(&[0.3; 8]).unwrap();
        assert_eq!(d.operators.len(), operator_classes().len());
        let sum: f32 = d.operators.iter().map(|w| w.weight.get()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(d.top().is_some());
        assert_eq!(d, m.route(&[0.3; 8]).unwrap());
    }

    #[test]
    fn wrong_width_and_non_finite_are_refused() {
        let m = model();
        assert_eq!(
            m.route(&[0.0; 3]),
            Err(CoreError::PooledWidth {
                expected: 8,
                found: 3
            })
        );
        let mut v = [0.0; 8];
        v[0] = f32::INFINITY;
        assert!(matches!(m.route(&v), Err(CoreError::NonFinite { .. })));
    }

    #[test]
    fn top_is_the_highest_weight() {
        let m = model();
        let mut m2 = m.clone();
        // Bias the head hard toward class 4.
        m2.router.b.as_mut_slice()[4] = 50.0;
        assert_eq!(
            m2.route(&[0.0; 8]).unwrap().top(),
            Some(operator_classes()[4])
        );
    }
}
