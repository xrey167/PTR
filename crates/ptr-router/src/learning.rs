//! L1 learning for the finding router: a contextual bandit over the repair Pods of a finding
//! code, trained only on verifier verdicts, usable only as an [`AdmissibleScorer`].
//!
//! ```text
//!  FindingRouter ---- FindingRouteRecord (support, propensity) ----> RoutingLearner::settle
//!       ^                                                                  ^
//!       |                                          VerifierReward::from_class(classify output)
//!       |                                                                  |
//!   ActiveScorer <--- PolicyBook::promote <--- PromotionGate::assess <--- LoggedDecision log
//!   (reads only)       (host action)            (IPS / SNIPS / DR, LCB)   (held-out of training)
//! ```
//!
//! # What is learned, and what is not
//!
//! The learner keeps one Beta posterior per `(class, Pod)`: the class is the finding code of a
//! repair route (or "regenerate" for the proposer draw) and the scope of the learner names the
//! task kind (one learner per task kind, so a `(finding code, task kind, Pod)` triple is a
//! posterior). A [`LearnedPolicy`] is an immutable snapshot of those posteriors. It implements
//! [`AdmissibleScorer`]: it receives the verifier-determined admissible Pods of one class and
//! returns a non-negative weight for each of them, by Thompson sampling with a seeded generator
//! (the weights are a deterministic function of the snapshot, the class and the admissible set,
//! so the propensity an estimator assigns to the policy is exact).
//!
//! It has no handle on anything else. The module names no gate, level, threshold, verifier,
//! eligibility view, registry update or `Accepted` proof (a source-scan test pins this), a
//! snapshot cannot be given to the router except through [`crate::PolicyBook`], and the router,
//! not the policy, owns the exploration floor and the eligibility filter: a weight on a Pod
//! outside the admissible set is refused and every eligible Pod keeps propensity of at least
//! `epsilon / n` whatever the policy says.
//!
//! # Reward
//!
//! The only reward is [`VerifierReward`]: 1 when [`classify`](crate::classify) accepted the
//! candidate that the routed Pod produced and 0 otherwise (a declined repair, a failed repair, a
//! timeout). It is built from a [`ReportClass`], not from a number, so a score, a confidence or
//! a statement of a model cannot become a reward, and a candidate with a hard finding is 0
//! whatever its score (invariant 11). When one repair attempt routed several codes, the single
//! verdict on the repaired candidate is the reward of each of those routes (joint attribution).
//!
//! # Not online
//!
//! Learning updates posteriors; it does not change routing. Routing changes only when a host
//! promotes a [`LearnedPolicy`] through [`crate::PolicyBook`] with evidence from
//! [`crate::PromotionGate`], computed on decisions logged after the snapshot was taken.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use ptr_types::PodId;
use sha2::{Digest as _, Sha256};

use crate::findings::{
    AdmissibleScorer, Exploration, FindingRouteRecord, ReportClass, RouteClass, ScoringContext,
};
use crate::router::{RouteDistribution, RouteError};

const POLICY_DOMAIN: &[u8] = b"ptr.router.learning.policy.v1";
const STREAM_DOMAIN: &[u8] = b"ptr.router.learning.stream.v1";

/// Most arms (class, Pod) a learner or policy holds.
pub const MAX_ARMS: usize = 1 << 14;
/// Most logged decisions a learner keeps.
pub const MAX_LOG: usize = 1 << 20;
/// Thompson draws per weight computation: bounds the work of one routing.
pub const MAX_DRAWS: u32 = 1 << 14;
/// Most Pods a single weight computation considers (the registry's own cap is lower).
const MAX_WEIGHTED_PODS: usize = 1024;

/// The verifier's verdict on the candidate a routed Pod produced.
///
/// ```compile_fail
/// // The tuple field is private: a number cannot become a reward.
/// let reward = ptr_router::VerifierReward(true);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifierReward(bool);

impl VerifierReward {
    /// The routed Pod produced nothing that was accepted (it declined, or its candidate was not
    /// accepted).
    pub const NOT_CLEARED: Self = Self(false);

    /// 1 exactly when `class` is [`ReportClass::Accept`], which only
    /// [`classify`](crate::classify) can build.
    pub fn from_class(class: &ReportClass) -> Self {
        Self(matches!(class, ReportClass::Accept(_)))
    }

    /// A logged verdict replayed from a record (or a synthetic log in a test). The caller states
    /// what the verifier said; no score is involved.
    pub fn replayed(passed: bool) -> Self {
        Self(passed)
    }

    pub fn passed(self) -> bool {
        self.0
    }

    /// 1.0 or 0.0.
    pub fn value(self) -> f64 {
        if self.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// The label of a route class that keys a posterior: `repair/<code>` or `regenerate` (the
/// proposer draw does not depend on why a regeneration was asked for).
pub(crate) fn class_label(class: &RouteClass) -> String {
    match class {
        RouteClass::Repair(code) => format!("repair/{}", code.as_str()),
        RouteClass::Regenerate(_) => "regenerate".to_owned(),
    }
}

/// Why a log record or a learner configuration is refused.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LearningError {
    /// A configuration value is out of range (named).
    Config(&'static str),
    /// The support of a replayed decision is empty, has a zero width, does not sum to `2^53`, is
    /// out of `PodId` order or does not contain the chosen Pod.
    BadSupport,
    /// The exploration floor of a logged decision is zero: the log cannot be used off-policy.
    NoFloor,
    /// The learner and the router disagree on the exploration rate.
    FloorMismatch,
}

impl fmt::Display for LearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(what) => write!(f, "learner configuration: {what}"),
            Self::BadSupport => f.write_str("logged support is not a valid sampling support"),
            Self::NoFloor => f.write_str("a logged decision without an exploration floor"),
            Self::FloorMismatch => f.write_str("learner and router exploration rates differ"),
        }
    }
}

impl std::error::Error for LearningError {}

/// One routing decision with its verdict, as the off-policy estimator needs it.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedDecision {
    seq: u64,
    class: RouteClass,
    support: Vec<(PodId, u64)>,
    chosen: PodId,
    reward: VerifierReward,
    floor: f64,
}

impl LoggedDecision {
    /// A decision as the router logged it. `floor` is the router's exploration rate, which the
    /// estimator needs to rebuild a target policy's distribution exactly.
    ///
    /// # Errors
    /// [`LearningError::NoFloor`] when `floor` is not positive.
    pub fn from_record(
        seq: u64,
        record: &FindingRouteRecord,
        reward: VerifierReward,
        floor: Exploration,
    ) -> Result<Self, LearningError> {
        if floor.epsilon() <= 0.0 {
            return Err(LearningError::NoFloor);
        }
        Ok(Self {
            seq,
            class: record.class().clone(),
            support: record.support().to_vec(),
            chosen: record.pod().clone(),
            reward,
            floor: floor.epsilon(),
        })
    }

    /// A decision rebuilt from its parts: a logged record read back, or a synthetic log.
    ///
    /// # Errors
    /// [`LearningError::BadSupport`] unless `support` is non-empty, in strictly ascending `PodId`
    /// order, has positive widths that sum to exactly `2^53` and contains `chosen`;
    /// [`LearningError::NoFloor`] when `floor` is not positive.
    pub fn replayed(
        seq: u64,
        class: RouteClass,
        support: Vec<(PodId, u64)>,
        chosen: PodId,
        reward: VerifierReward,
        floor: Exploration,
    ) -> Result<Self, LearningError> {
        if floor.epsilon() <= 0.0 {
            return Err(LearningError::NoFloor);
        }
        let ordered = support.windows(2).all(|pair| pair[0].0 < pair[1].0);
        let total: u128 = support.iter().map(|(_, width)| u128::from(*width)).sum();
        let positive = support.iter().all(|(_, width)| *width > 0);
        let holds_chosen = support.iter().any(|(pod, _)| *pod == chosen);
        if support.is_empty()
            || !ordered
            || !positive
            || total != u128::from(RouteDistribution::RESOLUTION)
            || !holds_chosen
        {
            return Err(LearningError::BadSupport);
        }
        Ok(Self {
            seq,
            class,
            support,
            chosen,
            reward,
            floor: floor.epsilon(),
        })
    }

    /// Position in the learner's log; training uses decisions below a snapshot's cutoff.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn class(&self) -> &RouteClass {
        &self.class
    }

    pub fn support(&self) -> &[(PodId, u64)] {
        &self.support
    }

    pub fn chosen(&self) -> &PodId {
        &self.chosen
    }

    pub fn reward(&self) -> VerifierReward {
        self.reward
    }

    /// The exploration rate of the router that logged the decision.
    pub fn floor(&self) -> f64 {
        self.floor
    }

    /// The probability with which the logging policy chose [`Self::chosen`].
    pub fn propensity(&self) -> f64 {
        let width = self
            .support
            .iter()
            .find(|(pod, _)| *pod == self.chosen)
            .map_or(0, |(_, width)| *width);
        width as f64 / RouteDistribution::RESOLUTION as f64
    }
}

/// Settings of a [`RoutingLearner`] and of the policies it snapshots.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnerConfig {
    /// Names what the learner is for, typically the task kind (`arithmetic`, `code`, ...). Part
    /// of every snapshot's digest.
    pub scope: String,
    /// Beta prior `(prior, prior)`, at least 1 (a flat or favourable-to-nothing prior).
    pub prior: f64,
    /// Thompson draws per weight computation, in `[16, MAX_DRAWS]`.
    pub draws: u32,
    /// Seed of the draws.
    pub seed: u64,
    /// The router's exploration rate. Must be positive: positivity is what makes the log usable
    /// off-policy.
    pub exploration: Exploration,
    pub max_arms: usize,
    pub max_log: usize,
}

impl LearnerConfig {
    /// Defaults: flat prior, 512 draws, the router's default exploration, bounded tables.
    pub fn new(scope: &str, seed: u64) -> Self {
        Self {
            scope: scope.to_owned(),
            prior: 1.0,
            draws: 512,
            seed,
            exploration: Exploration::default(),
            max_arms: 4096,
            max_log: 1 << 16,
        }
    }

    fn validate(&self) -> Result<(), LearningError> {
        if self.scope.is_empty() || self.scope.len() > 64 {
            return Err(LearningError::Config("scope is empty or over 64 bytes"));
        }
        if !(self.prior.is_finite() && (1.0..=1e6).contains(&self.prior)) {
            return Err(LearningError::Config("prior is not in [1, 1e6]"));
        }
        if !(16..=MAX_DRAWS).contains(&self.draws) {
            return Err(LearningError::Config("draws are not in [16, 16384]"));
        }
        if self.exploration.epsilon() <= 0.0 {
            return Err(LearningError::Config("exploration must be positive"));
        }
        if self.max_arms == 0 || self.max_arms > MAX_ARMS {
            return Err(LearningError::Config("max_arms is zero or above the cap"));
        }
        if self.max_log == 0 || self.max_log > MAX_LOG {
            return Err(LearningError::Config("max_log is zero or above the cap"));
        }
        Ok(())
    }
}

/// Verdict counts of one arm.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArmCounts {
    pub cleared: u64,
    pub not_cleared: u64,
}

type ArmKey = (String, PodId);

struct LearnerState {
    arms: BTreeMap<ArmKey, ArmCounts>,
    log: VecDeque<LoggedDecision>,
    next_seq: u64,
    untracked_arms: u64,
    dropped_log: u64,
}

/// Counts every verdict of every routed Pod and logs the decision. Shared (`Arc`) between the
/// loop, which settles, and the host, which snapshots.
pub struct RoutingLearner {
    config: LearnerConfig,
    state: Mutex<LearnerState>,
}

impl fmt::Debug for RoutingLearner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RoutingLearner")
            .field("scope", &self.config.scope)
            .finish_non_exhaustive()
    }
}

impl RoutingLearner {
    /// # Errors
    /// [`LearningError::Config`] for a value out of range.
    pub fn new(config: LearnerConfig) -> Result<Self, LearningError> {
        config.validate()?;
        Ok(Self {
            config,
            state: Mutex::new(LearnerState {
                arms: BTreeMap::new(),
                log: VecDeque::new(),
                next_seq: 0,
                untracked_arms: 0,
                dropped_log: 0,
            }),
        })
    }

    pub fn config(&self) -> &LearnerConfig {
        &self.config
    }

    fn state(&self) -> std::sync::MutexGuard<'_, LearnerState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record the verdict on the candidate a routed Pod produced: one count, one logged
    /// decision. A full arm table keeps counting known arms and counts the rest as untracked; a
    /// full log drops its oldest decision (by age, never by outcome).
    pub fn settle(&self, record: &FindingRouteRecord, reward: VerifierReward) {
        let mut state = self.state();
        let seq = state.next_seq;
        state.next_seq = seq.saturating_add(1);
        let key = (class_label(record.class()), record.pod().clone());
        if state.arms.contains_key(&key) || state.arms.len() < self.config.max_arms {
            let counts = state.arms.entry(key).or_default();
            if reward.passed() {
                counts.cleared = counts.cleared.saturating_add(1);
            } else {
                counts.not_cleared = counts.not_cleared.saturating_add(1);
            }
        } else {
            state.untracked_arms = state.untracked_arms.saturating_add(1);
        }
        // `from_record` fails only for a zero floor, which the configuration refuses.
        if let Ok(decision) =
            LoggedDecision::from_record(seq, record, reward, self.config.exploration)
        {
            if state.log.len() >= self.config.max_log {
                state.log.pop_front();
                state.dropped_log = state.dropped_log.saturating_add(1);
            }
            state.log.push_back(decision);
        }
    }

    /// Decisions settled so far (the next decision's `seq`).
    pub fn settled(&self) -> u64 {
        self.state().next_seq
    }

    /// Settled verdicts that did not fit the arm table.
    pub fn untracked_arms(&self) -> u64 {
        self.state().untracked_arms
    }

    /// Logged decisions dropped because the log was full.
    pub fn dropped_log(&self) -> u64 {
        self.state().dropped_log
    }

    /// The logged decisions with `seq >= from`, oldest first.
    pub fn log_from(&self, from: u64) -> Vec<LoggedDecision> {
        self.state()
            .log
            .iter()
            .filter(|decision| decision.seq >= from)
            .cloned()
            .collect()
    }

    /// An immutable policy of the posteriors as they are now. Decisions with `seq` below
    /// [`LearnedPolicy::cutoff`] trained it; an evaluation must use decisions from the cutoff on.
    pub fn snapshot(&self) -> LearnedPolicy {
        let state = self.state();
        LearnedPolicy::from_parts(&self.config, state.arms.clone(), state.next_seq)
    }
}

/// An immutable snapshot of a learner's posteriors, usable as an [`AdmissibleScorer`].
#[derive(Clone, Debug)]
pub struct LearnedPolicy {
    scope: String,
    prior: f64,
    draws: u32,
    seed: u64,
    exploration: Exploration,
    arms: Arc<BTreeMap<ArmKey, ArmCounts>>,
    cutoff: u64,
    digest: [u8; 32],
}

impl LearnedPolicy {
    fn from_parts(config: &LearnerConfig, arms: BTreeMap<ArmKey, ArmCounts>, cutoff: u64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(POLICY_DOMAIN);
        hasher.update((config.scope.len() as u64).to_le_bytes());
        hasher.update(config.scope.as_bytes());
        hasher.update(config.prior.to_bits().to_le_bytes());
        hasher.update(config.draws.to_le_bytes());
        hasher.update(config.seed.to_le_bytes());
        hasher.update(config.exploration.epsilon().to_bits().to_le_bytes());
        hasher.update(cutoff.to_le_bytes());
        hasher.update((arms.len() as u64).to_le_bytes());
        for ((class, pod), counts) in &arms {
            for text in [class.as_str(), pod.0.as_str()] {
                hasher.update((text.len() as u64).to_le_bytes());
                hasher.update(text.as_bytes());
            }
            hasher.update(counts.cleared.to_le_bytes());
            hasher.update(counts.not_cleared.to_le_bytes());
        }
        Self {
            scope: config.scope.clone(),
            prior: config.prior,
            draws: config.draws,
            seed: config.seed,
            exploration: config.exploration,
            arms: Arc::new(arms),
            cutoff,
            digest: hasher.finalize().into(),
        }
    }

    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Identity of the snapshot: scope, settings, cutoff and every posterior.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Decisions with `seq` below this trained the snapshot.
    pub fn cutoff(&self) -> u64 {
        self.cutoff
    }

    /// The exploration rate of the router the policy was trained under.
    pub fn exploration(&self) -> Exploration {
        self.exploration
    }

    /// The verdict counts of an arm, zero for an arm never seen.
    pub fn counts(&self, class: &RouteClass, pod: &PodId) -> ArmCounts {
        self.arms
            .get(&(class_label(class), pod.clone()))
            .copied()
            .unwrap_or_default()
    }

    /// Posterior mean clear probability of an arm, `(prior + cleared) / (2 prior + total)`.
    pub fn mean(&self, class: &RouteClass, pod: &PodId) -> f64 {
        let counts = self.counts(class, pod);
        (self.prior + counts.cleared as f64)
            / (2.0 * self.prior + counts.cleared as f64 + counts.not_cleared as f64)
    }

    fn beta_parameters(&self, label: &str, pod: &PodId) -> (f64, f64) {
        let counts = self
            .arms
            .get(&(label.to_owned(), pod.clone()))
            .copied()
            .unwrap_or_default();
        (
            self.prior + counts.cleared as f64,
            self.prior + counts.not_cleared as f64,
        )
    }

    /// The seed of the draw stream for one class and admissible set: a function of the snapshot,
    /// the class, the admissible Pods and their counts only, never of a request.
    fn stream_seed(&self, label: &str, admissible: &[PodId]) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(STREAM_DOMAIN);
        hasher.update(self.seed.to_le_bytes());
        hasher.update(self.digest);
        hasher.update((label.len() as u64).to_le_bytes());
        hasher.update(label.as_bytes());
        for pod in admissible {
            hasher.update((pod.0.len() as u64).to_le_bytes());
            hasher.update(pod.0.as_bytes());
        }
        let digest = hasher.finalize();
        let mut head = [0_u8; 8];
        head.copy_from_slice(&digest[..8]);
        u64::from_le_bytes(head)
    }
}

impl AdmissibleScorer for LearnedPolicy {
    /// Thompson sampling: the weight of a Pod is one plus the number of the `draws` posterior
    /// samples in which it had the largest sampled clear probability. It names only the Pods it
    /// was given.
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        let admissible = context.admissible;
        let label = class_label(context.class);
        if admissible.len() > MAX_WEIGHTED_PODS {
            return Ok(admissible.iter().map(|pod| (pod.clone(), 1.0)).collect());
        }
        let parameters: Vec<(f64, f64)> = admissible
            .iter()
            .map(|pod| self.beta_parameters(&label, pod))
            .collect();
        let mut rng = SplitMix64::new(self.stream_seed(&label, admissible));
        let mut wins = vec![0_u32; admissible.len()];
        for _ in 0..self.draws {
            let mut best = 0;
            let mut best_value = f64::NEG_INFINITY;
            for (index, (a, b)) in parameters.iter().enumerate() {
                let value = rng.beta(*a, *b);
                if value > best_value {
                    best_value = value;
                    best = index;
                }
            }
            if let Some(count) = wins.get_mut(best) {
                *count += 1;
            }
        }
        Ok(admissible
            .iter()
            .zip(wins)
            .map(|(pod, won)| (pod.clone(), f64::from(won) + 1.0))
            .collect())
    }
}

/// A reward model for the doubly robust estimator: the predicted clear probability of an arm.
pub trait RewardModel {
    /// A probability in `[0, 1]`.
    fn predict(&self, class: &RouteClass, pod: &PodId) -> f64;
}

impl RewardModel for LearnedPolicy {
    fn predict(&self, class: &RouteClass, pod: &PodId) -> f64 {
        self.mean(class, pod)
    }
}

/// SplitMix64: a small seeded generator with a full 2^64 period, enough for sampling draws that
/// must be reproducible from a seed.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval (0, 1).
    pub(crate) fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Gamma(shape, 1) for `shape >= 1` (Marsaglia and Tsang).
    fn gamma(&mut self, shape: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let t = 1.0 + c * x;
            if t <= 0.0 {
                continue;
            }
            let v = t * t * t;
            let u = self.unit();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    pub(crate) fn beta(&mut self, a: f64, b: f64) -> f64 {
        let x = self.gamma(a);
        let y = self.gamma(b);
        x / (x + y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_beta_sampler_has_the_right_mean_and_stays_in_the_unit_interval() {
        let mut rng = SplitMix64::new(11);
        for (a, b) in [(1.0, 1.0), (3.0, 7.0), (40.0, 4.0), (1.0, 25.0)] {
            let n = 20_000;
            let mut sum = 0.0;
            for _ in 0..n {
                let x = rng.beta(a, b);
                assert!((0.0..=1.0).contains(&x));
                sum += x;
            }
            let mean = sum / f64::from(n);
            assert!((mean - a / (a + b)).abs() < 0.01, "{a} {b} {mean}");
        }
    }

    #[test]
    fn the_unit_draw_never_reaches_the_ends() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..10_000 {
            let u = rng.unit();
            assert!(u > 0.0 && u < 1.0);
        }
    }

    #[test]
    fn a_replayed_decision_needs_a_valid_support_and_a_floor() {
        let a = PodId::from("a");
        let b = PodId::from("b");
        let half = RouteDistribution::RESOLUTION / 2;
        let class = RouteClass::Repair(crate::FindingCode::from("c"));
        let floor = Exploration::default();
        let ok = LoggedDecision::replayed(
            0,
            class.clone(),
            vec![(a.clone(), half), (b.clone(), half)],
            a.clone(),
            VerifierReward::replayed(true),
            floor,
        )
        .unwrap();
        assert!((ok.propensity() - 0.5).abs() < 1e-12);
        for bad in [
            vec![],
            vec![(a.clone(), half), (b.clone(), half - 1)],
            vec![(b.clone(), half), (a.clone(), half)],
            vec![(a.clone(), RouteDistribution::RESOLUTION), (b.clone(), 0)],
        ] {
            assert_eq!(
                LoggedDecision::replayed(
                    0,
                    class.clone(),
                    bad,
                    a.clone(),
                    VerifierReward::NOT_CLEARED,
                    floor
                ),
                Err(LearningError::BadSupport)
            );
        }
        assert_eq!(
            LoggedDecision::replayed(
                0,
                class,
                vec![(a.clone(), half), (b, half)],
                a,
                VerifierReward::NOT_CLEARED,
                Exploration::NONE
            ),
            Err(LearningError::NoFloor)
        );
    }

    fn context<'a>(
        class: &'a RouteClass,
        admissible: &'a [PodId],
        request: &'a ptr_types::RequestId,
    ) -> ScoringContext<'a> {
        ScoringContext {
            class,
            admissible,
            request_id: request,
            request_digest: [0; 32],
        }
    }

    fn policy_with(counts: &[(&str, u64, u64)]) -> LearnedPolicy {
        let config = LearnerConfig::new("t", 3);
        let class = class_label(&RouteClass::Repair(crate::FindingCode::from("c")));
        let arms = counts
            .iter()
            .map(|(pod, cleared, not_cleared)| {
                (
                    (class.clone(), PodId::from(*pod)),
                    ArmCounts {
                        cleared: *cleared,
                        not_cleared: *not_cleared,
                    },
                )
            })
            .collect();
        LearnedPolicy::from_parts(&config, arms, 0)
    }

    #[test]
    fn weights_name_exactly_the_admissible_pods_and_are_positive() {
        let class = RouteClass::Repair(crate::FindingCode::from("c"));
        let request = ptr_types::RequestId::from("r");
        let policy = policy_with(&[("a", 30, 2), ("b", 3, 30), ("outsider", 99, 0)]);
        let admissible = [PodId::from("a"), PodId::from("b"), PodId::from("c")];
        let weights = policy
            .weights(&context(&class, &admissible, &request))
            .unwrap();
        let named: Vec<&PodId> = weights.iter().map(|(pod, _)| pod).collect();
        assert_eq!(named, admissible.iter().collect::<Vec<_>>());
        assert!(weights.iter().all(|(_, w)| w.is_finite() && *w >= 1.0));
        // The Pod with the best record gets the most, the worst the least; an unseen Pod sits
        // between them under a flat prior.
        let weight = |name: &str| weights.iter().find(|(p, _)| p.0 == name).unwrap().1;
        assert!(
            weight("a") > weight("c") && weight("c") > weight("b"),
            "{weights:?}"
        );
        // The same call gives the same answer.
        assert_eq!(
            weights,
            policy
                .weights(&context(&class, &admissible, &request))
                .unwrap()
        );
        // A Pod that is not admissible is not weighted whatever the record says.
        let narrow = [PodId::from("b")];
        let only = policy.weights(&context(&class, &narrow, &request)).unwrap();
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].0, PodId::from("b"));
    }

    #[test]
    fn a_class_with_no_record_is_uniform_up_to_sampling_noise() {
        let class = RouteClass::Repair(crate::FindingCode::from("unseen"));
        let request = ptr_types::RequestId::from("r");
        let policy = policy_with(&[("a", 30, 2)]);
        let admissible = [PodId::from("a"), PodId::from("b")];
        let weights = policy
            .weights(&context(&class, &admissible, &request))
            .unwrap();
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        for (_, w) in &weights {
            assert!((w / total - 0.5).abs() < 0.15, "{weights:?}");
        }
    }

    #[test]
    fn the_configuration_is_range_checked() {
        let mut config = LearnerConfig::new("arith", 1);
        assert!(RoutingLearner::new(config.clone()).is_ok());
        config.prior = 0.5;
        assert!(RoutingLearner::new(config.clone()).is_err());
        config.prior = 1.0;
        config.draws = 3;
        assert!(RoutingLearner::new(config.clone()).is_err());
        config.draws = 64;
        config.exploration = Exploration::NONE;
        assert!(RoutingLearner::new(config).is_err());
        assert!(RoutingLearner::new(LearnerConfig::new("", 1)).is_err());
    }
}
