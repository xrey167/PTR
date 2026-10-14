pub use ptr_types::ActionIr;

use crate::core_model::CoreModel;
use crate::error::CoreError;
use crate::tensor::{fnv1a64, softmax, Matrix, SeededRng};

/// Largest candidate list the head ranks (Invariant 8).
pub const MAX_ACTION_CANDIDATES: usize = 256;

/// The head's selection among caller-supplied candidates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionChoice {
    /// Index into the candidate slice the caller passed in.
    pub index: usize,
    /// Learned bilinear score of the selected candidate.
    pub score: f32,
    /// Softmax probability among the candidates that passed the deterministic
    /// check.
    pub probability: f32,
}

/// Deterministic hashed feature vector of a candidate's semantic fields
/// (operation, target, capability, effect, input type). Generation, revision and
/// payload are deliberately not features: they are authority data, not content to
/// be scored.
fn candidate_features(action: &ActionIr, d: usize) -> Vec<f32> {
    let fields: [(&str, String); 5] = [
        ("op", action.operation.clone()),
        ("target", action.target.clone()),
        ("cap", action.capability.0.clone()),
        ("effect", format!("{:?}", action.effect)),
        ("type", action.input_type.0.clone()),
    ];
    let mut feat = vec![0.0_f32; d];
    for (tag, value) in fields {
        let mut bytes = tag.as_bytes().to_vec();
        bytes.push(0);
        bytes.extend_from_slice(value.as_bytes());
        let mut rng = SeededRng::new(fnv1a64(&bytes));
        for f in feat.iter_mut() {
            *f += rng.uniform(-1.0, 1.0);
        }
    }
    let scale = 1.0 / (d as f32).sqrt();
    feat.iter_mut().for_each(|f| *f *= scale);
    feat
}

impl CoreModel {
    /// Learned scores of `candidates` for a pooled state, without any filtering.
    pub fn score_actions(
        &self,
        pooled: &[f32],
        candidates: &[ActionIr],
    ) -> Result<Vec<f32>, CoreError> {
        self.check_pooled(pooled)?;
        if candidates.is_empty() {
            return Err(CoreError::NoCandidates);
        }
        if candidates.len() > MAX_ACTION_CANDIDATES {
            return Err(CoreError::TooManyCandidates {
                got: candidates.len(),
                max: MAX_ACTION_CANDIDATES,
            });
        }
        let d = self.config.d_model;
        let query = Matrix::new(1, d, pooled.to_vec())?.matmul(&self.action_w)?;
        let scores: Vec<f32> = candidates
            .iter()
            .map(|c| {
                query
                    .as_slice()
                    .iter()
                    .zip(candidate_features(c, d))
                    .map(|(q, f)| q * f)
                    .sum::<f32>()
            })
            .collect();
        if scores.iter().all(|s| s.is_finite()) {
            Ok(scores)
        } else {
            Err(CoreError::NonFinite {
                what: "action scores",
            })
        }
    }

    /// ActionIR head: rank caller-supplied candidates and pick one.
    ///
    /// # Authority
    ///
    /// The head only **ranks** the list it is given and returns an index into
    /// it. It never creates or edits an [`ActionIr`], so it cannot invent a
    /// target, revision or generation. `deterministic_check` is the
    /// deterministic verifier contract (Invariant 11: a learned score cannot
    /// override a deterministic verifier contradiction): a candidate for which it
    /// returns `false` is excluded before the softmax and can never be selected,
    /// however high its learned score. If every candidate is rejected the result
    /// is [`CoreError::AllCandidatesRejected`], not a fallback to the best
    /// learned score. Passing the choice on still requires the runtime's own
    /// generation/revision/capability checks.
    pub fn rank_actions<F>(
        &self,
        pooled: &[f32],
        candidates: &[ActionIr],
        deterministic_check: F,
    ) -> Result<ActionChoice, CoreError>
    where
        F: Fn(&ActionIr) -> bool,
    {
        let scores = self.score_actions(pooled, candidates)?;
        let allowed: Vec<usize> = (0..candidates.len())
            .filter(|i| deterministic_check(&candidates[*i]))
            .collect();
        let mut best: Option<usize> = None;
        for &i in &allowed {
            if best.is_none_or(|b| scores[i] > scores[b]) {
                best = Some(i);
            }
        }
        let index = best.ok_or(CoreError::AllCandidatesRejected)?;
        let allowed_scores: Vec<f32> = allowed.iter().map(|i| scores[*i]).collect();
        let probs = softmax(&allowed_scores);
        let pos = allowed.iter().position(|i| *i == index).unwrap_or(0);
        Ok(ActionChoice {
            index,
            score: scores[index],
            probability: probs[pos],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CoreConfig;
    use ptr_types::{CapabilityId, Effect, Generation, Revision, TypeId};

    fn model() -> CoreModel {
        CoreModel::new(CoreConfig {
            d_model: 8,
            heads: 2,
            d_ff: 16,
            ..CoreConfig::default()
        })
        .unwrap()
    }

    fn action(op: &str, target: &str) -> ActionIr {
        ActionIr {
            operation: op.into(),
            target: target.into(),
            capability: CapabilityId::from("cap.write"),
            effect: Effect::Mutation,
            input_type: TypeId::from("doc"),
            generation: Generation(3),
            revision: Revision(7),
            payload: vec![1, 2, 3],
        }
    }

    fn candidates() -> Vec<ActionIr> {
        (0..6)
            .map(|i| action(&format!("op{i}"), &format!("res{i}")))
            .collect()
    }

    #[test]
    fn selects_the_highest_scored_candidate_and_returns_an_index() {
        let m = model();
        let c = candidates();
        let pooled = [0.5, -0.2, 0.9, 0.1, -0.7, 0.3, 0.8, -0.4];
        let scores = m.score_actions(&pooled, &c).unwrap();
        let argmax = (0..c.len())
            .max_by(|a, b| scores[*a].total_cmp(&scores[*b]))
            .unwrap();
        let choice = m.rank_actions(&pooled, &c, |_| true).unwrap();
        assert_eq!(choice.index, argmax);
        assert_eq!(choice.score, scores[argmax]);
        assert!(choice.probability > 0.0 && choice.probability <= 1.0);
        // The chosen candidate is exactly the supplied one: nothing is invented.
        assert_eq!(c[choice.index].generation, Generation(3));
        assert_eq!(c[choice.index].revision, Revision(7));
    }

    #[test]
    fn candidate_rejected_by_the_deterministic_check_is_never_selected() {
        let m = model();
        let c = candidates();
        let pooled = [0.5, -0.2, 0.9, 0.1, -0.7, 0.3, 0.8, -0.4];
        let best = m.rank_actions(&pooled, &c, |_| true).unwrap().index;
        // Reject the learned favourite: it must not come back, whatever its score.
        let rejected = c[best].clone();
        let choice = m.rank_actions(&pooled, &c, |a| *a != rejected).unwrap();
        assert_ne!(choice.index, best);
        // Walk the whole ranking by rejecting one winner at a time.
        let mut banned: Vec<ActionIr> = Vec::new();
        while let Ok(ch) = m.rank_actions(&pooled, &c, |a| !banned.contains(a)) {
            assert!(!banned.contains(&c[ch.index]));
            banned.push(c[ch.index].clone());
        }
        assert_eq!(banned.len(), c.len());
        assert_eq!(
            m.rank_actions(&pooled, &c, |_| false).err(),
            Some(CoreError::AllCandidatesRejected)
        );
    }

    #[test]
    fn probabilities_are_over_the_admitted_candidates_only() {
        let m = model();
        let c = candidates();
        let pooled = [0.2; 8];
        // Only one candidate passes: it is chosen with probability exactly 1.
        let only = c[4].clone();
        let ch = m.rank_actions(&pooled, &c, |a| *a == only).unwrap();
        assert_eq!(ch.index, 4);
        assert_eq!(ch.probability, 1.0);
    }

    #[test]
    fn candidate_list_bounds_and_widths_are_typed() {
        let m = model();
        assert_eq!(
            m.rank_actions(&[0.0; 8], &[], |_| true).err(),
            Some(CoreError::NoCandidates)
        );
        let many = vec![action("a", "b"); MAX_ACTION_CANDIDATES + 1];
        assert!(matches!(
            m.rank_actions(&[0.0; 8], &many, |_| true),
            Err(CoreError::TooManyCandidates { .. })
        ));
        assert!(matches!(
            m.rank_actions(&[0.0; 2], &candidates(), |_| true),
            Err(CoreError::PooledWidth { .. })
        ));
    }

    #[test]
    fn features_ignore_authority_fields() {
        let mut a = action("op", "t");
        let b = a.clone();
        a.generation = Generation(99);
        a.revision = Revision(99);
        a.payload = vec![9];
        assert_eq!(candidate_features(&a, 8), candidate_features(&b, 8));
        assert_ne!(
            candidate_features(&action("op", "t1"), 8),
            candidate_features(&action("op", "t2"), 8)
        );
    }
}
