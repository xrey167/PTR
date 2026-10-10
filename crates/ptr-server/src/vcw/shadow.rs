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
        let log = self.learner.log_from(candidate.cutoff());
        let due = self.last_estimate_at + self.last_estimate_at / 2 + 1;
        if log.len() < FIRST_ESTIMATE.max(due) {
            return;
        }
        self.last_estimate_at = log.len();
        // The uniform policy is what routes; the candidate is the reward model, trained on
        // decisions before the freeze, so the evaluation log is independent of it.
        let Ok(comparison) = compare_policies(
            &log,
            candidate,
            &UniformScorer,
            candidate,
            Estimator::DoublyRobust,
            Confidence::ONE_SIDED_95,
        ) else {
            return;
        };
        let estimate = ShadowEstimate {
            records: log.len(),
            difference: comparison.difference,
            lower_bound: comparison.difference_lcb,
        };
        runtime.record_vcw_shadow_estimate(
            &format!("vcw-{}", self.kind.as_str()),
            candidate.digest(),
            log.len() as u64,
            estimate.difference,
            estimate.lower_bound,
        );
        self.latest = Some(estimate);
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
