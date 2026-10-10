//! Off-policy evaluation of routing policies and the promotion gate.
//!
//! The estimators are the categorical IPS, self-normalised IPS (SNIPS) and doubly robust (DR)
//! family over [`LoggedDecision`]s. The triage versions in `ptr-branch` (`evaluate_off_policy`,
//! `doubly_robust`) are specific to its three-action space and a score-dependent target; the
//! router's action space is "which of the eligible Pods", with the logging distribution recorded
//! exactly in [`FindingRouteRecord::support`](crate::FindingRouteRecord::support), so the
//! estimators here are written over that, with the same refusal discipline (an empty log, a
//! logging policy without positivity, a non-finite estimate are errors, never an estimate).
//!
//! # What is estimated
//!
//! For a target policy `pi` (an [`AdmissibleScorer`], deployed under the router's exploration
//! floor `epsilon`) and a log of decisions `(class_i, support_i, a_i, r_i)` made by a logging
//! policy `mu_i` with `r_i` in `{0, 1}` the verifier's verdict:
//!
//! ```text
//!   w_i   = pi_i(a_i) / mu_i(a_i)                    pi_i = mixed_distribution(pi, class_i, support_i)
//!   IPS   = mean w_i r_i
//!   SNIPS = sum w_i r_i / sum w_i
//!   DR    = mean [ sum_a pi_i(a) rhat(class_i, a)  +  w_i (r_i - rhat(class_i, a_i)) ]
//! ```
//!
//! `pi_i` is rebuilt by the function the router itself samples with, so a policy's estimated
//! probability is the probability it would route with, floor included. Each estimator carries a
//! standard error from the per-decision influence terms (a normal approximation, so it needs a
//! log of some hundreds of decisions; [`PromotionRule::min_records`] and
//! [`PromotionRule::min_effective_sample`] refuse a log too small to trust it).
//!
//! # Preconditions (refused, not assumed)
//!
//! * the log's exploration floor is positive and every logged propensity is at least
//!   `epsilon / |support|` (positivity: a target cannot put mass where the logger had none);
//! * the policy names only Pods of the logged support;
//! * the reward model's predictions are in `[0, 1]`;
//! * for a promotion, no decision of the evaluation log trained the candidate
//!   ([`OpeError::OverlapsTraining`]): a policy evaluated on the data it was fitted to is
//!   optimistic, so the gate evaluates it on decisions logged after the snapshot.
//!
//! # Promotion
//!
//! [`PromotionGate::assess`] compares the candidate with the incumbent on the same log with a
//! paired difference per decision (the two estimates share the log's noise), and returns
//! [`Verdict::Promote`] with a sealed [`PromotionEvidence`] only when the one-sided lower
//! confidence bound of the difference exceeds the margin. The estimator is named in the rule
//! before the data are seen. Evidence is a value, not an action: [`crate::PolicyBook::promote`]
//! is the host action that consumes it.
//!
//! The gate reads the evaluation log from the [`RoutingLearner`] itself and requires the
//! candidate to be a snapshot of that learner and the incumbent to be the book's
//! [`ActiveScorer`] under the candidate's exploration rate. There is no way to hand it a log,
//! a reward or an incumbent digest: a log assembled by a caller cannot produce evidence.
//!
//! # Statistical assumptions (stated, not corrected)
//!
//! * **One-step value.** What is compared is the per-route clear rate: the probability that the
//!   candidate produced by the routed Pod is accepted by the verifier. It is not request-level
//!   acceptance, cost or latency, and it ignores state dependence (a round-2 repair exists only
//!   after a round-1 failure).
//! * **Independence.** Decisions are treated as independent. Routes of one attempt share a
//!   verdict (joint attribution) and rounds of one request are dependent, so the standard error
//!   is optimistic; the floors below are chosen with that in mind.
//! * **Peeking.** The test uses a fixed one-sided quantile. Assessing the same learner on a
//!   growing log repeatedly (as the server's shadow mode does for its estimates) inflates the
//!   false-promotion rate; the estimates in shadow mode are for monitoring, and a promotion
//!   should rest on one assessment of a log whose size was fixed in advance.
//! * **Scorer purity.** The target and incumbent are evaluated with placeholder `request_id`
//!   and `request_digest`; they must be functions of the class and admissible set only (see
//!   [`AdmissibleScorer`]).
//!
//! # Floors on the gate's strength
//!
//! A caller can tighten but not weaken the rule below [`MIN_RECORDS_FLOOR`] decisions,
//! [`MIN_EFFECTIVE_SAMPLE_FLOOR`] effective samples and the one-sided 95% quantile
//! ([`Z_FLOOR`]), and the margin cannot be negative (a candidate must beat the incumbent, not
//! tie it). 100 records and an effective sample of 30 are the smallest sizes at which a normal
//! approximation of a mean of bounded influence terms is not absurd (the usual n of 30 rule,
//! with a factor for the dependence above); the quantile floor is the conventional one-sided
//! 5% level.

use std::collections::BTreeMap;
use std::fmt;

use ptr_analytics::RunningMoments;
use ptr_types::{PodId, RequestId};
use sha2::{Digest as _, Sha256};

use crate::findings::{
    mixed_distribution, AdmissibleScorer, FindingRouteErrorKind, ScoringContext,
};
use crate::learning::{class_label, LearnedPolicy, LoggedDecision, RewardModel, RoutingLearner};
use crate::policy_book::ActiveScorer;
use crate::router::RouteDistribution;

/// Fewest evaluation decisions a [`PromotionRule`] may demand.
pub const MIN_RECORDS_FLOOR: usize = 100;
/// Smallest effective sample size a [`PromotionRule`] may demand.
pub const MIN_EFFECTIVE_SAMPLE_FLOOR: f64 = 30.0;
/// Smallest one-sided normal quantile a [`PromotionRule`] may use (one-sided 95%).
pub const Z_FLOOR: f64 = 1.644_853_626_951_472_7 - 1e-12;

const EVIDENCE_DOMAIN: &[u8] = b"ptr.router.ope.evidence.v1";
const LOG_DOMAIN: &[u8] = b"ptr.router.ope.log.v1";

/// Which estimator decides a comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Estimator {
    /// Inverse propensity scoring: unbiased, high variance.
    Ips,
    /// Self-normalised IPS: bounded by the reward range, slightly biased, lower variance.
    Snips,
    /// Doubly robust: unbiased when the propensities are right, low variance when the reward
    /// model is good.
    DoublyRobust,
}

impl Estimator {
    fn code(self) -> u8 {
        match self {
            Self::Ips => 0,
            Self::Snips => 1,
            Self::DoublyRobust => 2,
        }
    }
}

/// A one-sided confidence level as a normal quantile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Confidence(f64);

impl Confidence {
    /// One-sided 95%.
    pub const ONE_SIDED_95: Self = Self(1.644_853_626_951_472_7);
    /// One-sided 99%.
    pub const ONE_SIDED_99: Self = Self(2.326_347_874_040_841);

    /// # Errors
    /// [`OpeError::Config`] unless `z` is finite and in `(0, 6]`.
    pub fn from_quantile(z: f64) -> Result<Self, OpeError> {
        if z.is_finite() && z > 0.0 && z <= 6.0 {
            Ok(Self(z))
        } else {
            Err(OpeError::Config("the quantile is not in (0, 6]"))
        }
    }

    pub fn quantile(self) -> f64 {
        self.0
    }
}

/// Why an evaluation or an assessment was refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum OpeError {
    EmptyLog,
    /// A configuration value is out of range.
    Config(&'static str),
    /// A logged decision has no exploration floor, or a propensity below `floor / |support|`.
    BelowFloor {
        seq: u64,
    },
    /// The policy named a Pod outside the logged admissible set.
    PolicyOutsideAdmissibleSet {
        seq: u64,
        pod: PodId,
    },
    /// The policy failed or gave no mass on a record.
    PolicyFailed {
        seq: u64,
    },
    /// The reward model returned a value outside `[0, 1]`.
    RewardModelOutOfRange,
    /// The target policy gives every logged action probability zero, or an estimate is not
    /// finite.
    NonFiniteEstimate(&'static str),
    /// An evaluation decision (`seq`) is below the candidate's training cutoff.
    OverlapsTraining {
        seq: u64,
        cutoff: u64,
    },
    /// The candidate is not a snapshot of the learner whose log was offered.
    ForeignLearner,
    /// The incumbent's exploration rate is not the candidate's.
    ExplorationMismatch,
}

impl fmt::Display for OpeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLog => f.write_str("the log is empty"),
            Self::Config(what) => write!(f, "invalid configuration: {what}"),
            Self::BelowFloor { seq } => {
                write!(f, "decision {seq} was logged without the exploration floor")
            }
            Self::PolicyOutsideAdmissibleSet { seq, pod } => {
                write!(
                    f,
                    "the policy named {pod:?} outside the admissible set at {seq}"
                )
            }
            Self::PolicyFailed { seq } => write!(f, "the policy gave no usable weights at {seq}"),
            Self::RewardModelOutOfRange => f.write_str("reward model prediction is not in [0, 1]"),
            Self::NonFiniteEstimate(which) => write!(f, "the {which} estimate is not finite"),
            Self::OverlapsTraining { seq, cutoff } => write!(
                f,
                "decision {seq} is below the training cutoff {cutoff}: the candidate saw it"
            ),
            Self::ForeignLearner => f.write_str("the candidate is not a snapshot of this learner"),
            Self::ExplorationMismatch => {
                f.write_str("the incumbent and the candidate use different exploration rates")
            }
        }
    }
}

impl std::error::Error for OpeError {}

/// The three estimates of one policy on one log, with standard errors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyEstimate {
    pub n: usize,
    pub ips: f64,
    pub snips: f64,
    pub doubly_robust: f64,
    pub se_ips: f64,
    pub se_snips: f64,
    pub se_doubly_robust: f64,
    /// `(sum w)^2 / sum w^2`: how many logged decisions the target effectively reuses.
    pub effective_sample_size: f64,
}

impl PolicyEstimate {
    pub fn value(&self, estimator: Estimator) -> f64 {
        match estimator {
            Estimator::Ips => self.ips,
            Estimator::Snips => self.snips,
            Estimator::DoublyRobust => self.doubly_robust,
        }
    }

    pub fn standard_error(&self, estimator: Estimator) -> f64 {
        match estimator {
            Estimator::Ips => self.se_ips,
            Estimator::Snips => self.se_snips,
            Estimator::DoublyRobust => self.se_doubly_robust,
        }
    }
}

/// The importance weights and reward-model terms of one policy on one log.
struct Weighted {
    w: Vec<f64>,
    r: Vec<f64>,
    direct: Vec<f64>,
    predicted: Vec<f64>,
}

impl Weighted {
    fn build(
        log: &[LoggedDecision],
        policy: &dyn AdmissibleScorer,
        model: &dyn RewardModel,
    ) -> Result<Self, OpeError> {
        if log.is_empty() {
            return Err(OpeError::EmptyLog);
        }
        let request = RequestId::from("ope");
        let mut cache: BTreeMap<(String, Vec<PodId>, u64), RouteDistribution> = BTreeMap::new();
        let mut out = Self {
            w: Vec::with_capacity(log.len()),
            r: Vec::with_capacity(log.len()),
            direct: Vec::with_capacity(log.len()),
            predicted: Vec::with_capacity(log.len()),
        };
        for decision in log {
            let admissible: Vec<PodId> = decision
                .support()
                .iter()
                .map(|(pod, _)| pod.clone())
                .collect();
            let floor = decision.floor();
            if !(floor.is_finite() && floor > 0.0) {
                return Err(OpeError::BelowFloor {
                    seq: decision.seq(),
                });
            }
            let minimum = floor / admissible.len() as f64;
            if decision.propensity() < minimum * (1.0 - 1e-9) {
                return Err(OpeError::BelowFloor {
                    seq: decision.seq(),
                });
            }
            let key = (
                class_label(decision.class()),
                admissible.clone(),
                floor.to_bits(),
            );
            if !cache.contains_key(&key) {
                let context = ScoringContext {
                    class: decision.class(),
                    admissible: &admissible,
                    request_id: &request,
                    request_digest: [0; 32],
                };
                let distribution =
                    mixed_distribution(policy, &context, floor).map_err(|kind| match kind {
                        FindingRouteErrorKind::OutsideAdmissibleSet { pod } => {
                            OpeError::PolicyOutsideAdmissibleSet {
                                seq: decision.seq(),
                                pod,
                            }
                        }
                        _ => OpeError::PolicyFailed {
                            seq: decision.seq(),
                        },
                    })?;
                cache.insert(key.clone(), distribution);
            }
            let distribution = cache.get(&key).ok_or(OpeError::PolicyFailed {
                seq: decision.seq(),
            })?;
            let target = distribution.probability(decision.chosen());
            let mut direct = 0.0;
            for (pod, probability) in distribution.iter() {
                let predicted = model.predict(decision.class(), pod);
                if !(predicted.is_finite() && (0.0..=1.0).contains(&predicted)) {
                    return Err(OpeError::RewardModelOutOfRange);
                }
                direct += probability * predicted;
            }
            let predicted_logged = model.predict(decision.class(), decision.chosen());
            if !(predicted_logged.is_finite() && (0.0..=1.0).contains(&predicted_logged)) {
                return Err(OpeError::RewardModelOutOfRange);
            }
            out.w.push(target / decision.propensity());
            out.r.push(decision.reward().value());
            out.direct.push(direct);
            out.predicted.push(predicted_logged);
        }
        Ok(out)
    }

    fn n(&self) -> usize {
        self.w.len()
    }

    fn ips_terms(&self) -> Vec<f64> {
        self.w.iter().zip(&self.r).map(|(w, r)| w * r).collect()
    }

    fn dr_terms(&self) -> Vec<f64> {
        (0..self.n())
            .map(|i| self.direct[i] + self.w[i] * (self.r[i] - self.predicted[i]))
            .collect()
    }

    fn snips(&self) -> Result<f64, OpeError> {
        let total: f64 = self.w.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(OpeError::NonFiniteEstimate("snips"));
        }
        let numerator: f64 = self.w.iter().zip(&self.r).map(|(w, r)| w * r).sum();
        Ok(numerator / total)
    }

    /// The influence of each decision on SNIPS, `w_i (r_i - snips) / mean w`.
    fn snips_influence(&self, snips: f64) -> Vec<f64> {
        let mean_w = self.w.iter().sum::<f64>() / self.n() as f64;
        self.w
            .iter()
            .zip(&self.r)
            .map(|(w, r)| w * (r - snips) / mean_w)
            .collect()
    }

    fn effective_sample_size(&self) -> f64 {
        let sum: f64 = self.w.iter().sum();
        let squares: f64 = self.w.iter().map(|w| w * w).sum();
        if squares > 0.0 {
            sum * sum / squares
        } else {
            0.0
        }
    }
}

fn mean_and_se(values: &[f64]) -> Result<(f64, f64), OpeError> {
    let mut moments = RunningMoments::default();
    for value in values {
        moments
            .push(*value)
            .map_err(|_| OpeError::NonFiniteEstimate("moments"))?;
    }
    let mean = moments.mean().ok_or(OpeError::EmptyLog)?;
    let variance = moments.sample_variance().unwrap_or(0.0);
    let se = (variance / values.len() as f64).sqrt();
    if mean.is_finite() && se.is_finite() {
        Ok((mean, se))
    } else {
        Err(OpeError::NonFiniteEstimate("moments"))
    }
}

/// The IPS, SNIPS and DR estimates of `policy` on `log`.
///
/// `policy` is evaluated with the placeholder `request_id` `"ope"` and a zero `request_digest`,
/// so it must be a function of the route class, the admissible set and its own immutable state
/// only (it must ignore both request fields); otherwise the probability estimated for a logged
/// action is not the probability the router sampled it with. See
/// [`AdmissibleScorer`](crate::AdmissibleScorer). The estimates are a one-step per-route value
/// with decisions treated as independent; see the module documentation.
///
/// # Errors
/// [`OpeError`]: see the module documentation for the preconditions.
pub fn estimate_policy(
    log: &[LoggedDecision],
    policy: &dyn AdmissibleScorer,
    model: &dyn RewardModel,
) -> Result<PolicyEstimate, OpeError> {
    let weighted = Weighted::build(log, policy, model)?;
    summarise(&weighted)
}

fn summarise(weighted: &Weighted) -> Result<PolicyEstimate, OpeError> {
    let (ips, se_ips) = mean_and_se(&weighted.ips_terms())?;
    let (doubly_robust, se_doubly_robust) = mean_and_se(&weighted.dr_terms())?;
    let snips = weighted.snips()?;
    let (_, se_snips) = mean_and_se(&weighted.snips_influence(snips))?;
    Ok(PolicyEstimate {
        n: weighted.n(),
        ips,
        snips,
        doubly_robust,
        se_ips,
        se_snips,
        se_doubly_robust,
        effective_sample_size: weighted.effective_sample_size(),
    })
}

/// A target policy compared with an incumbent on one log.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub estimator: Estimator,
    pub target: PolicyEstimate,
    pub incumbent: PolicyEstimate,
    /// Target minus incumbent under `estimator`.
    pub difference: f64,
    /// Standard error of the paired difference.
    pub difference_se: f64,
    /// `difference - z * difference_se`.
    pub difference_lcb: f64,
}

/// Compares `target` with `incumbent` on `log` with a paired difference.
///
/// # Errors
/// [`OpeError`] as for [`estimate_policy`], for either policy.
pub fn compare_policies(
    log: &[LoggedDecision],
    target: &dyn AdmissibleScorer,
    incumbent: &dyn AdmissibleScorer,
    model: &dyn RewardModel,
    estimator: Estimator,
    confidence: Confidence,
) -> Result<Comparison, OpeError> {
    let t = Weighted::build(log, target, model)?;
    let i = Weighted::build(log, incumbent, model)?;
    let target_estimate = summarise(&t)?;
    let incumbent_estimate = summarise(&i)?;
    let (terms_t, terms_i) = match estimator {
        Estimator::Ips => (t.ips_terms(), i.ips_terms()),
        Estimator::DoublyRobust => (t.dr_terms(), i.dr_terms()),
        Estimator::Snips => (
            t.snips_influence(target_estimate.snips),
            i.snips_influence(incumbent_estimate.snips),
        ),
    };
    let paired: Vec<f64> = terms_t.iter().zip(&terms_i).map(|(a, b)| a - b).collect();
    let (mean, se) = mean_and_se(&paired)?;
    // The influence terms of SNIPS average to zero; its difference is the difference of the
    // two ratios.
    let difference = match estimator {
        Estimator::Snips => target_estimate.snips - incumbent_estimate.snips,
        Estimator::Ips | Estimator::DoublyRobust => mean,
    };
    Ok(Comparison {
        estimator,
        target: target_estimate,
        incumbent: incumbent_estimate,
        difference,
        difference_se: se,
        difference_lcb: difference - confidence.quantile() * se,
    })
}

/// What a candidate must show before the gate lets a host promote it. Fixed before the data are
/// seen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionRule {
    pub estimator: Estimator,
    pub confidence: Confidence,
    /// The lower confidence bound of the value difference must exceed this, in `[0, 1]`
    /// (never negative: a tie is not an improvement).
    pub margin: f64,
    /// Fewest decisions in the evaluation log, at least [`MIN_RECORDS_FLOOR`].
    pub min_records: usize,
    /// Smallest effective sample size of the candidate's weights, at least
    /// [`MIN_EFFECTIVE_SAMPLE_FLOOR`].
    pub min_effective_sample: f64,
}

impl PromotionRule {
    /// Doubly robust, one-sided 95%, the given margin and size floors ([`PromotionGate::new`]
    /// refuses values below the module's floors).
    pub fn new(margin: f64, min_records: usize, min_effective_sample: f64) -> Self {
        Self {
            estimator: Estimator::DoublyRobust,
            confidence: Confidence::ONE_SIDED_95,
            margin,
            min_records,
            min_effective_sample,
        }
    }

    fn validate(&self) -> Result<(), OpeError> {
        if !(self.margin.is_finite() && (0.0..=1.0).contains(&self.margin)) {
            return Err(OpeError::Config("margin is not in [0, 1]"));
        }
        if self.min_records < MIN_RECORDS_FLOOR {
            return Err(OpeError::Config("min_records is below the floor of 100"));
        }
        if !(self.min_effective_sample.is_finite()
            && self.min_effective_sample >= MIN_EFFECTIVE_SAMPLE_FLOOR)
        {
            return Err(OpeError::Config(
                "min_effective_sample is below the floor of 30",
            ));
        }
        if self.confidence.quantile() < Z_FLOOR {
            return Err(OpeError::Config(
                "the quantile is below the one-sided 95% floor",
            ));
        }
        Ok(())
    }
}

/// Why the gate refused to let a candidate be promoted.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PromotionRefusal {
    TooFewRecords {
        have: usize,
        need: usize,
    },
    EffectiveSampleTooSmall {
        have: f64,
        need: f64,
    },
    /// The lower bound of the difference does not exceed the margin.
    LowerBoundNotAboveMargin {
        lcb: f64,
        margin: f64,
    },
}

/// A candidate that cleared the gate. Sealed: only [`PromotionGate::assess`] builds one, and
/// [`crate::PolicyBook::promote`] checks it against the candidate and the incumbent it names.
#[derive(Clone, Debug, PartialEq)]
pub struct PromotionEvidence {
    candidate: [u8; 32],
    incumbent: [u8; 32],
    log_digest: [u8; 32],
    estimator: Estimator,
    records: usize,
    difference: f64,
    lcb: f64,
    margin: f64,
    cutoff: u64,
}

impl PromotionEvidence {
    pub fn candidate(&self) -> &[u8; 32] {
        &self.candidate
    }

    pub fn incumbent(&self) -> &[u8; 32] {
        &self.incumbent
    }

    pub fn log_digest(&self) -> &[u8; 32] {
        &self.log_digest
    }

    pub fn estimator(&self) -> Estimator {
        self.estimator
    }

    pub fn records(&self) -> usize {
        self.records
    }

    pub fn difference(&self) -> f64 {
        self.difference
    }

    pub fn lower_bound(&self) -> f64 {
        self.lcb
    }

    pub fn margin(&self) -> f64 {
        self.margin
    }

    pub fn cutoff(&self) -> u64 {
        self.cutoff
    }

    /// A digest of every field, for the record a host keeps of a promotion.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(EVIDENCE_DOMAIN);
        hasher.update(self.candidate);
        hasher.update(self.incumbent);
        hasher.update(self.log_digest);
        hasher.update([self.estimator.code()]);
        hasher.update((self.records as u64).to_le_bytes());
        hasher.update(self.difference.to_bits().to_le_bytes());
        hasher.update(self.lcb.to_bits().to_le_bytes());
        hasher.update(self.margin.to_bits().to_le_bytes());
        hasher.update(self.cutoff.to_le_bytes());
        hasher.finalize().into()
    }
}

/// The outcome of an assessment.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Promote(PromotionEvidence),
    Refuse(PromotionRefusal),
}

/// An assessment: the numbers and the verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct Assessment {
    pub comparison: Comparison,
    pub verdict: Verdict,
}

impl Assessment {
    /// The evidence when the candidate may be promoted.
    pub fn evidence(&self) -> Option<&PromotionEvidence> {
        match &self.verdict {
            Verdict::Promote(evidence) => Some(evidence),
            Verdict::Refuse(_) => None,
        }
    }
}

/// Decides, from logged decisions, whether a learned candidate may replace the incumbent.
#[derive(Clone, Copy, Debug)]
pub struct PromotionGate {
    rule: PromotionRule,
}

fn digest_of_log(log: &[LoggedDecision]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(LOG_DOMAIN);
    hasher.update((log.len() as u64).to_le_bytes());
    for decision in log {
        hasher.update(decision.seq().to_le_bytes());
        let label = class_label(decision.class());
        hasher.update((label.len() as u64).to_le_bytes());
        hasher.update(label.as_bytes());
        hasher.update((decision.support().len() as u64).to_le_bytes());
        for (pod, width) in decision.support() {
            hasher.update((pod.0.len() as u64).to_le_bytes());
            hasher.update(pod.0.as_bytes());
            hasher.update(width.to_le_bytes());
        }
        hasher.update((decision.chosen().0.len() as u64).to_le_bytes());
        hasher.update(decision.chosen().0.as_bytes());
        hasher.update([u8::from(decision.reward().passed())]);
        hasher.update(decision.floor().to_bits().to_le_bytes());
    }
    hasher.finalize().into()
}

impl PromotionGate {
    /// # Errors
    /// [`OpeError::Config`] for a rule out of range.
    pub fn new(rule: PromotionRule) -> Result<Self, OpeError> {
        rule.validate()?;
        Ok(Self { rule })
    }

    pub fn rule(&self) -> &PromotionRule {
        &self.rule
    }

    /// Assesses `candidate` against `incumbent` on the decisions `learner` logged after the
    /// candidate was frozen.
    ///
    /// The log is read from `learner` (decisions with `seq >= candidate.cutoff()`), so it holds
    /// only router records and verifier verdicts the learner was told, none of which the
    /// candidate trained on. `incumbent` is the book's active handle: its digest is the one the
    /// evidence names.
    ///
    /// # Errors
    /// [`OpeError`] for a log or policy that cannot be evaluated, a candidate that did not come
    /// from `learner` ([`OpeError::ForeignLearner`]) or an incumbent under another exploration
    /// rate ([`OpeError::ExplorationMismatch`]). A candidate that is evaluated but does not
    /// clear the rule is `Ok` with a [`Verdict::Refuse`].
    pub fn assess(
        &self,
        learner: &RoutingLearner,
        candidate: &LearnedPolicy,
        incumbent: &ActiveScorer,
    ) -> Result<Assessment, OpeError> {
        if candidate.origin() != learner.id() {
            return Err(OpeError::ForeignLearner);
        }
        if incumbent.exploration() != candidate.exploration() {
            return Err(OpeError::ExplorationMismatch);
        }
        let log = learner.log_from(candidate.cutoff());
        let log = log.as_slice();
        let incumbent_digest = incumbent.digest();
        if let Some(decision) = log.iter().find(|d| d.seq() < candidate.cutoff()) {
            return Err(OpeError::OverlapsTraining {
                seq: decision.seq(),
                cutoff: candidate.cutoff(),
            });
        }
        let comparison = compare_policies(
            log,
            candidate,
            incumbent,
            candidate,
            self.rule.estimator,
            self.rule.confidence,
        )?;
        let refusal = if log.len() < self.rule.min_records {
            Some(PromotionRefusal::TooFewRecords {
                have: log.len(),
                need: self.rule.min_records,
            })
        } else if comparison.target.effective_sample_size < self.rule.min_effective_sample {
            Some(PromotionRefusal::EffectiveSampleTooSmall {
                have: comparison.target.effective_sample_size,
                need: self.rule.min_effective_sample,
            })
        } else if comparison.difference_lcb <= self.rule.margin {
            Some(PromotionRefusal::LowerBoundNotAboveMargin {
                lcb: comparison.difference_lcb,
                margin: self.rule.margin,
            })
        } else {
            None
        };
        let verdict = match refusal {
            Some(refusal) => Verdict::Refuse(refusal),
            None => Verdict::Promote(PromotionEvidence {
                candidate: candidate.digest(),
                incumbent: incumbent_digest,
                log_digest: digest_of_log(log),
                estimator: self.rule.estimator,
                records: log.len(),
                difference: comparison.difference,
                lcb: comparison.difference_lcb,
                margin: self.rule.margin,
                cutoff: candidate.cutoff(),
            }),
        };
        Ok(Assessment {
            comparison,
            verdict,
        })
    }
}
