//! Shadow learning of the vcw service (`[vcw] learning = "shadow"`).
//!
//! Each served kind keeps a [`RoutingLearner`] that the loop tells the verifier's verdict on
//! every routed Pod. After a fixed number of decisions the learner is frozen into a candidate
//! policy; from then on the decisions logged after the freeze (made, as always, by the uniform
//! router) are the held-out evidence, and the doubly robust estimate of the candidate's value
//! over the uniform policy is logged as a runtime event at geometrically growing log sizes.
//!
//! Nothing here routes. The loop's scorer stays [`ptr_router::UniformScorer`]; the candidate is
//! never handed to the router, and promoting one is a host action
//! ([`ptr_router::PolicyBook::promote`]) this service does not take and has no configuration for.
//!
//! What the estimates are, and are not: monitoring values from a one-step per-route clear rate
//! (not request-level acceptance or cost), computed repeatedly on a growing log (so they are
//! uncorrected for peeking) with decisions treated as independent. They are never promotion
//! evidence; the promotion gate assesses a candidate once, on a log whose size was fixed in
//! advance. The estimate is computed by the learner's own log only after a cheap length check
//! (`RoutingLearner::log_len_from`), so a request that is not due does not copy the log. When the
//! log is so full that the next estimate point could never be reached, the candidate is
//! re-frozen from the current posteriors and the held-out log restarts, so estimates continue.

use std::sync::Arc;

use ptr_config::VcwTaskKind;
use ptr_router::{
    compare_policies, Confidence, Estimator, Exploration, LearnedPolicy, LearnerConfig,
    RoutingLearner, UniformScorer,
};
use ptr_runtime::PtrRuntime;
use ptr_types::Digest;

/// Decisions the learner sees before its first candidate is frozen.
pub(super) const TRAIN_DECISIONS: u64 = 256;
/// Fewest held-out decisions before an estimate is logged.
pub(super) const FIRST_ESTIMATE: usize = 128;
/// Largest log the learner keeps (and so the largest an estimate is computed from).
pub(super) const MAX_LOG: usize = 16_384;

/// The latest logged estimate of a kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowEstimate {
    /// Held-out decisions the estimate used.
    pub records: usize,
    /// Doubly robust value of the candidate minus the uniform policy's.
    pub difference: f64,
    /// One-sided 95% lower bound of the difference.
    pub lower_bound: f64,
}

/// What a kind has learned so far.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearningReport {
    pub kind: VcwTaskKind,
    /// Verdicts the learner has been told.
    pub settled: u64,
    /// The digest of the frozen candidate, once there is one.
    pub candidate: Option<Digest>,
    pub latest: Option<ShadowEstimate>,
}

pub(super) struct Shadow {
    kind: VcwTaskKind,
    learner: Arc<RoutingLearner>,
    candidate: Option<LearnedPolicy>,
    last_estimate_at: usize,
    latest: Option<ShadowEstimate>,
}

impl Shadow {
    /// # Errors
    /// The reason the learner configuration is refused.
    pub(super) fn new(
        kind: VcwTaskKind,
        seed: u64,
        exploration: Exploration,
    ) -> Result<Self, String> {
        let learner = RoutingLearner::new(LearnerConfig {
            exploration,
            max_log: MAX_LOG,
            ..LearnerConfig::new(kind.as_str(), seed)
        })
        .map_err(|error| error.to_string())?;
        Ok(Self {
            kind,
            learner: Arc::new(learner),
            candidate: None,
            last_estimate_at: 0,
            latest: None,
        })
    }

    pub(super) fn learner(&self) -> Arc<RoutingLearner> {
        Arc::clone(&self.learner)
    }

    /// Freeze the candidate when due and log an estimate when the held-out log has grown enough.
    pub(super) fn tick(&mut self, runtime: &mut PtrRuntime) {
        let Some(candidate) = &self.candidate else {
            if self.learner.settled() >= TRAIN_DECISIONS {
                self.candidate = Some(self.learner.snapshot());
            }
            return;
        };
        let due = self.last_estimate_at + self.last_estimate_at / 2 + 1;
        let held_out = self.learner.log_len_from(candidate.cutoff());
        if held_out < FIRST_ESTIMATE.max(due) {
            return;
        }
        let candidate = candidate.clone();
        let log = self.learner.log_from(candidate.cutoff());
        self.last_estimate_at = log.len();
        // The uniform policy is what routes; the candidate is the reward model, trained on
        // decisions before the freeze, so the evaluation log is independent of it.
        let comparison = compare_policies(
            &log,
            &candidate,
            &UniformScorer,
            &candidate,
            Estimator::DoublyRobust,
            Confidence::ONE_SIDED_95,
        );
        // The next estimate point is `1.5 * len + 1`. If the log can never hold that many, this
        // candidate has had its last estimate: freeze a fresh one and start its log over.
        let rotate = log.len() + log.len() / 2 + 1 > MAX_LOG;
        let estimate = comparison.ok().map(|comparison| ShadowEstimate {
            records: log.len(),
            difference: comparison.difference,
            lower_bound: comparison.difference_lcb,
        });
        if let Some(estimate) = &estimate {
            runtime.record_vcw_shadow_estimate(
                &format!("vcw-{}", self.kind.as_str()),
                candidate.digest(),
                log.len() as u64,
                estimate.difference,
                estimate.lower_bound,
            );
        }
        self.conclude(estimate, rotate);
    }

    /// Store the round's estimate, or, when the candidate is rotated, drop it: the report pairs
    /// `latest` with the current candidate's digest, and an estimate of the previous candidate
    /// must never be shown next to a fresh one (its own estimate is in the runtime events).
    fn conclude(&mut self, estimate: Option<ShadowEstimate>, rotate: bool) {
        if rotate {
            self.candidate = Some(self.learner.snapshot());
            self.last_estimate_at = 0;
            self.latest = None;
        } else if estimate.is_some() {
            self.latest = estimate;
        }
    }

    pub(super) fn report(&self) -> LearningReport {
        LearningReport {
            kind: self.kind,
            settled: self.learner.settled(),
            candidate: self.candidate.as_ref().map(LearnedPolicy::digest),
            latest: self.latest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotating_the_candidate_drops_the_previous_candidates_estimate() {
        let mut shadow =
            Shadow::new(VcwTaskKind::Arithmetic, 7, Exploration::new(0.1).unwrap()).unwrap();
        shadow.candidate = Some(shadow.learner.snapshot());
        let estimate = ShadowEstimate {
            records: 10,
            difference: 0.25,
            lower_bound: 0.1,
        };
        shadow.conclude(Some(estimate), false);
        assert_eq!(shadow.report().latest, Some(estimate));
        shadow.conclude(Some(estimate), true);
        assert_eq!(shadow.report().latest, None);
        assert!(shadow.report().candidate.is_some());
    }
}
