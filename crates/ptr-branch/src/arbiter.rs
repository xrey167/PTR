use sha2::{Digest, Sha256};

use ptr_analytics::clopper_pearson_upper;

use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

use crate::branch::BranchId;
use crate::error::ArbiterError;

/// What happens to a certified branch next.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TriageDecision {
    /// Propose the merge plan for commit. This is not authorisation: the plan
    /// still goes through the runtime's revision check and required
    /// verification, exactly like a human-approved plan.
    AutoPropose,
    /// Ask a person.
    Escalate,
    /// Drop the branch.
    Discard,
}

/// The score above which an eligible branch is auto-proposed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AutoThreshold {
    /// Nothing is auto-proposed.
    Never,
    /// Branches scoring at least this are auto-proposed.
    AtLeast(f32),
}

impl AutoThreshold {
    fn admits(self, score: f32) -> bool {
        match self {
            Self::Never => false,
            Self::AtLeast(threshold) => score >= threshold,
        }
    }
}

/// A triage policy: a calibrated threshold plus the rate at which eligible
/// branches are sent to a person irrespective of score.
///
/// The calibration slice is what makes the threshold honest. Outcomes of
/// auto-proposed branches are observed only through later reverts, and
/// escalated branches only below the threshold, so neither is an exchangeable
/// sample of the branches the threshold decides about. Escalating a uniform
/// random fraction of *all* eligible branches for adjudication is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriagePolicy {
    threshold: AutoThreshold,
    calibration_rate: f64,
}

/// One triage, with what is needed to learn from it later.
#[derive(Clone, Debug, PartialEq)]
pub struct TriageOutcome {
    pub decision: TriageDecision,
    /// Whether the rules left the decision to the policy at all. Ineligible
    /// branches are decided by verification alone.
    pub eligible: bool,
    /// Whether this branch was escalated as part of the calibration slice.
    pub calibration_slice: bool,
    /// The score the policy saw.
    pub score: f32,
    /// Probability the logging policy assigned to auto-proposing this branch.
    pub auto_propensity: f64,
}

/// A human adjudication of a calibration-slice branch: would auto-proposing it
/// have been harmful?
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationSample {
    score: f32,
    harmful: bool,
}

impl TriagePolicy {
    /// Create a policy with a fraction `calibration_rate` of eligible branches
    /// reserved for human calibration. Zero disables that slice; the threshold
    /// is stored without validation.
    ///
    /// # Errors
    /// Returns `ArbiterError::InvalidExploration` unless the rate is finite
    /// and in `[0, 1)`.
    pub fn new(threshold: AutoThreshold, calibration_rate: f64) -> Result<Self, ArbiterError> {
        if !(calibration_rate.is_finite() && (0.0..1.0).contains(&calibration_rate)) {
            return Err(ArbiterError::InvalidExploration {
                rate: calibration_rate,
            });
        }
        Ok(Self {
            threshold,
            calibration_rate,
        })
    }

    pub fn threshold(&self) -> AutoThreshold {
        self.threshold
    }

    /// Decide one branch. `draw` is a uniform number in `[0, 1)` that the caller
    /// derives reproducibly, for example with [`calibration_draw`].
    ///
    /// Verification is consulted first and cannot be outvoted: a failed report
    /// discards, and a disputed, unknown or below-`FullSemantic` report
    /// escalates, whatever the score. Only a passing, full-semantic or
    /// deterministic report with no hard findings makes a branch eligible for
    /// the threshold. A passing report with a hard finding escalates.
    ///
    /// # Errors
    /// Returns `ArbiterError::InvalidDraw` unless `draw` is finite and in
    /// `[0, 1)`, whatever the report. A NaN or a draw of at least one would
    /// keep every eligible branch out of the calibration slice, and a negative
    /// draw would put every one in it, while the logged `auto_propensity`
    /// still claimed `1 - calibration_rate`: the slice would stop being a
    /// uniform sample and off-policy estimates would reweight by a propensity
    /// the policy never had.
    pub fn triage(
        &self,
        report: &VerificationReport,
        score: Probability,
        draw: f64,
    ) -> Result<TriageOutcome, ArbiterError> {
        if !(draw.is_finite() && (0.0..1.0).contains(&draw)) {
            return Err(ArbiterError::InvalidDraw { draw });
        }
        let score = score.get();
        let forced = |decision| {
            Ok(TriageOutcome {
                decision,
                eligible: false,
                calibration_slice: false,
                score,
                auto_propensity: 0.0,
            })
        };
        match report.status {
            VerificationStatus::Fail => return forced(TriageDecision::Discard),
            VerificationStatus::Disputed | VerificationStatus::Unknown => {
                return forced(TriageDecision::Escalate)
            }
            VerificationStatus::Pass => {}
        }
        match report.level {
            VerificationLevel::FullSemantic | VerificationLevel::Deterministic => {}
            VerificationLevel::Unverified
            | VerificationLevel::LatentAgreement
            | VerificationLevel::SampleVerified => return forced(TriageDecision::Escalate),
        }
        // A hard finding under a passing status is still a finding no score
        // may outweigh: a person decides.
        if report.findings.iter().any(|finding| finding.hard) {
            return forced(TriageDecision::Escalate);
        }

        let admitted = self.threshold.admits(score);
        let auto_propensity = if admitted {
            1.0 - self.calibration_rate
        } else {
            0.0
        };
        let calibration_slice = draw < self.calibration_rate;
        let decision = if !calibration_slice && admitted {
            TriageDecision::AutoPropose
        } else {
            TriageDecision::Escalate
        };
        Ok(TriageOutcome {
            decision,
            eligible: true,
            calibration_slice,
            score,
            auto_propensity,
        })
    }

    /// Probability this policy takes `action` on a logged record.
    fn probability(&self, record: &LoggedTriage, action: TriageDecision) -> f64 {
        if !record.eligible {
            return if action == record.decision { 1.0 } else { 0.0 };
        }
        let auto = if self.threshold.admits(record.score) {
            1.0 - self.calibration_rate
        } else {
            0.0
        };
        match action {
            TriageDecision::AutoPropose => auto,
            TriageDecision::Escalate => 1.0 - auto,
            TriageDecision::Discard => 0.0,
        }
    }
}

impl TriageOutcome {
    /// Turn a calibration-slice triage into a calibration sample once a person
    /// has adjudicated it. Any other triage is refused: its outcome is not an
    /// exchangeable sample of eligible branches.
    pub fn adjudicate(&self, harmful: bool) -> Option<CalibrationSample> {
        (self.eligible && self.calibration_slice).then_some(CalibrationSample {
            score: self.score,
            harmful,
        })
    }
}

/// A reproducible uniform draw in `[0, 1)` for one branch under one seed.
pub fn calibration_draw(branch: &BranchId, seed: u64) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-branch/calibration-draw/v1");
    hasher.update(seed.to_le_bytes());
    hasher.update(branch.0.as_bytes());
    let digest = hasher.finalize();
    let bits = u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"));
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

/// Number of steps of the fixed threshold grid `{0, 1/G, ..., 1}`.
pub const THRESHOLD_GRID_STEPS: u32 = 1000;

/// The fixed grid thresholds are chosen from, ascending. Both calibration
/// rules need a candidate set fixed before the data; the observed scores are
/// not.
pub fn threshold_grid() -> impl DoubleEndedIterator<Item = f32> {
    (0..=THRESHOLD_GRID_STEPS).map(|step| step as f32 / THRESHOLD_GRID_STEPS as f32)
}

/// Choose the auto-propose threshold by conformal risk control (Angelopoulos,
/// Bates, Fisch, Lei and Schuster, "Conformal Risk Control").
///
/// The loss of a branch at threshold `t` is `1` when it would be
/// auto-proposed (`score >= t`) and was adjudicated harmful, else `0`; it can
/// only fall as `t` rises along the fixed grid, and `B = 1`. The result is the
/// smallest grid threshold `t` with
///
/// ```text
/// n/(n+1) * mean_i loss_i(t) + 1/(n+1) <= alpha
/// ```
///
/// What this bounds is the *expected joint* probability that the next
/// eligible branch is auto-proposed **and** harmful, marginally over
/// calibration and test draws, for a branch exchangeable with the samples. It
/// does not bound the harm rate among auto-proposed branches (that is
/// [`certify_threshold`]). When no grid threshold qualifies — always when
/// `n + 1 < 1/alpha` — nothing is auto-proposed.
pub fn calibrate_threshold(
    samples: &[CalibrationSample],
    alpha: f64,
) -> Result<AutoThreshold, ArbiterError> {
    check_level("alpha", alpha)?;
    if samples.is_empty() {
        return Err(ArbiterError::EmptyCalibration);
    }
    let n = samples.len() as f64;
    for threshold in threshold_grid() {
        let harmful_admitted = samples
            .iter()
            .filter(|sample| sample.harmful && sample.score >= threshold)
            .count() as f64;
        let bound = (n / (n + 1.0)) * (harmful_admitted / n) + 1.0 / (n + 1.0);
        if bound <= alpha {
            return Ok(AutoThreshold::AtLeast(threshold));
        }
    }
    Ok(AutoThreshold::Never)
}

/// Choose the auto-propose threshold by Learn-then-Test fixed-sequence testing
/// with exact binomial bounds (Angelopoulos, Bates, Candès, Jordan and Lei,
/// "Learn then Test"; the threshold rule of Jung, Brahman and Choi, "Trust or
/// Escalate").
///
/// For a grid threshold `t`, let `n_t` calibration branches score at least `t`
/// and `k_t` of them be harmful. Grid thresholds are tested from high to low;
/// each passes when the one-sided Clopper-Pearson upper bound
/// `UCB(k_t, n_t; delta)` is at most `alpha`. Testing stops at the first
/// failure and the lowest passing threshold is returned. Thresholds admitting
/// fewer than `ceil(ln delta / ln(1 - alpha))` samples could not pass even
/// with no harm, so the sequence starts below them; that start depends on the
/// scores only, never on the harm labels under test. Then, with probability at
/// least `1 - delta` over the calibration sample, the harm rate *among
/// auto-proposed branches* is at most `alpha`.
///
/// This is the recommended rule. The unit is the branch, so decisions that
/// share a branch are never counted as independent evidence.
pub fn certify_threshold(
    samples: &[CalibrationSample],
    alpha: f64,
    delta: f64,
) -> Result<AutoThreshold, ArbiterError> {
    check_level("alpha", alpha)?;
    check_level("delta", delta)?;
    if samples.is_empty() {
        return Err(ArbiterError::EmptyCalibration);
    }
    let minimum_admitted = (delta.ln() / (1.0 - alpha).ln()).ceil() as u64;
    let mut certified = AutoThreshold::Never;
    for threshold in threshold_grid().rev() {
        let admitted = samples.iter().filter(|s| s.score >= threshold).count() as u64;
        if admitted < minimum_admitted {
            continue;
        }
        let harmful = samples
            .iter()
            .filter(|s| s.harmful && s.score >= threshold)
            .count() as u64;
        let bound = clopper_pearson_upper(harmful, admitted, delta).map_err(|_| {
            ArbiterError::InvalidRisk {
                field: "delta",
                value: delta,
            }
        })?;
        if bound <= alpha {
            certified = AutoThreshold::AtLeast(threshold);
        } else {
            break;
        }
    }
    Ok(certified)
}

fn check_level(field: &'static str, value: f64) -> Result<(), ArbiterError> {
    if value.is_finite() && value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(ArbiterError::InvalidRisk { field, value })
    }
}

/// One logged triage with the reward its decision earned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoggedTriage {
    pub eligible: bool,
    pub score: f32,
    pub decision: TriageDecision,
    /// The logging policy's probability of auto-proposing this record.
    pub auto_propensity: f64,
    pub reward: f64,
}

impl From<&TriageOutcome> for LoggedTriage {
    fn from(outcome: &TriageOutcome) -> Self {
        Self {
            eligible: outcome.eligible,
            score: outcome.score,
            decision: outcome.decision,
            auto_propensity: outcome.auto_propensity,
            reward: 0.0,
        }
    }
}

impl LoggedTriage {
    fn logging_probability(&self, action: TriageDecision) -> f64 {
        if !self.eligible {
            return if action == self.decision { 1.0 } else { 0.0 };
        }
        match action {
            TriageDecision::AutoPropose => self.auto_propensity,
            TriageDecision::Escalate => 1.0 - self.auto_propensity,
            TriageDecision::Discard => 0.0,
        }
    }
}

/// Off-policy estimates of a candidate policy's mean reward from a log.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OffPolicyEstimate {
    /// Inverse propensity scoring: unbiased under positivity, high variance.
    pub ips: f64,
    /// Self-normalised IPS: slightly biased, much lower variance.
    pub snips: f64,
    /// Kish effective sample size of the importance weights.
    pub effective_sample_size: f64,
}

/// Estimate `target`'s mean reward from decisions logged under another policy.
///
/// Refuses when `target` would take an action the logging policy could never
/// have taken on some record. That is the common case for a lower threshold:
/// the logging policy never auto-proposes below its own threshold, so no log it
/// produced can say what auto-proposing there would have earned, and an
/// estimate that ignored this would silently report the escalation reward
/// instead.
pub fn evaluate_off_policy(
    log: &[LoggedTriage],
    target: &TriagePolicy,
) -> Result<OffPolicyEstimate, ArbiterError> {
    let weights = importance_weights(log, target)?;
    let n = log.len() as f64;
    let weighted: f64 = weights
        .iter()
        .zip(log)
        .map(|(weight, record)| weight * record.reward)
        .sum();
    let total: f64 = weights.iter().sum();
    let squares: f64 = weights.iter().map(|weight| weight * weight).sum();
    Ok(OffPolicyEstimate {
        ips: weighted / n,
        snips: if total > 0.0 { weighted / total } else { 0.0 },
        effective_sample_size: if squares > 0.0 {
            total * total / squares
        } else {
            0.0
        },
    })
}

/// Doubly robust estimate: a reward model's prediction for the target policy,
/// corrected by importance-weighted residuals on the logged actions. Unbiased
/// when either the propensities or the reward model are correct.
pub fn doubly_robust<M>(
    log: &[LoggedTriage],
    target: &TriagePolicy,
    reward_model: M,
) -> Result<f64, ArbiterError>
where
    M: Fn(&LoggedTriage, TriageDecision) -> f64,
{
    let weights = importance_weights(log, target)?;
    let actions = [
        TriageDecision::AutoPropose,
        TriageDecision::Escalate,
        TriageDecision::Discard,
    ];
    let total: f64 = log
        .iter()
        .zip(&weights)
        .map(|(record, weight)| {
            let direct: f64 = actions
                .iter()
                .map(|&action| target.probability(record, action) * reward_model(record, action))
                .sum();
            direct + weight * (record.reward - reward_model(record, record.decision))
        })
        .sum();
    Ok(total / log.len() as f64)
}

fn importance_weights(
    log: &[LoggedTriage],
    target: &TriagePolicy,
) -> Result<Vec<f64>, ArbiterError> {
    if log.is_empty() {
        return Err(ArbiterError::EmptyLog);
    }
    let actions = [
        TriageDecision::AutoPropose,
        TriageDecision::Escalate,
        TriageDecision::Discard,
    ];
    let mut weights = Vec::with_capacity(log.len());
    for (index, record) in log.iter().enumerate() {
        if !(record.auto_propensity.is_finite() && (0.0..=1.0).contains(&record.auto_propensity)) {
            return Err(ArbiterError::InvalidPropensity {
                index,
                value: record.auto_propensity,
            });
        }
        for action in actions {
            if target.probability(record, action) > 0.0 && record.logging_probability(action) == 0.0
            {
                return Err(ArbiterError::PositivityViolation { index });
            }
        }
        let logged = record.logging_probability(record.decision);
        if logged <= 0.0 {
            return Err(ArbiterError::InvalidPropensity {
                index,
                value: logged,
            });
        }
        weights.push(target.probability(record, record.decision) / logged);
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(score: f32, harmful: bool) -> CalibrationSample {
        CalibrationSample { score, harmful }
    }

    #[test]
    fn with_too_few_samples_nothing_is_auto_proposed() {
        // 1/(n+1) = 0.25 > alpha = 0.1 for n = 3.
        let samples = [sample(0.9, false), sample(0.8, false), sample(0.7, false)];
        assert_eq!(
            calibrate_threshold(&samples, 0.1).unwrap(),
            AutoThreshold::Never
        );
    }

    #[test]
    fn the_crc_threshold_sits_just_above_the_harmful_score_it_must_exclude() {
        // n = 21: one admitted harmful sample gives 2/22 <= 0.1, two give 3/22.
        let mut samples: Vec<_> = (1..=19).map(|i| sample(i as f32 * 0.05, false)).collect();
        samples.push(sample(0.6, true));
        samples.push(sample(0.75, true));
        match calibrate_threshold(&samples, 0.1).unwrap() {
            AutoThreshold::AtLeast(threshold) => {
                assert!(threshold > 0.6 && threshold <= 0.601 + 1e-6, "{threshold}");
            }
            AutoThreshold::Never => panic!("a threshold exists"),
        }
    }

    #[test]
    fn learn_then_test_certifies_the_clean_region_and_bounds_the_harm_rate() {
        let mut samples: Vec<_> = (0..60)
            .map(|i| sample(0.5 + i as f32 / 200.0, false))
            .collect();
        samples.extend((0..20).map(|i| sample(0.1 + i as f32 / 100.0, i % 2 == 0)));
        match certify_threshold(&samples, 0.1, 0.1).unwrap() {
            AutoThreshold::AtLeast(threshold) => {
                assert!(
                    threshold <= 0.5,
                    "the clean region is certified: {threshold}"
                );
                let admitted: Vec<_> = samples.iter().filter(|s| s.score >= threshold).collect();
                let harmful = admitted.iter().filter(|s| s.harmful).count();
                assert!(harmful as f64 / admitted.len() as f64 <= 0.1);
            }
            AutoThreshold::Never => panic!("sixty clean samples certify a threshold"),
        }
        let few = [sample(0.9, false), sample(0.8, false)];
        assert_eq!(
            certify_threshold(&few, 0.05, 0.05).unwrap(),
            AutoThreshold::Never
        );
    }

    #[test]
    fn levels_outside_the_unit_interval_are_refused() {
        let samples = [sample(0.5, false)];
        assert!(matches!(
            certify_threshold(&samples, 0.1, 1.0),
            Err(ArbiterError::InvalidRisk { field: "delta", .. })
        ));
        assert!(matches!(
            calibrate_threshold(&samples, 0.0),
            Err(ArbiterError::InvalidRisk { field: "alpha", .. })
        ));
    }

    #[test]
    fn the_grid_is_fixed_and_spans_the_unit_interval() {
        let grid: Vec<f32> = threshold_grid().collect();
        assert_eq!(grid.len(), THRESHOLD_GRID_STEPS as usize + 1);
        assert_eq!(grid[0], 0.0);
        assert_eq!(*grid.last().unwrap(), 1.0);
    }

    #[test]
    fn the_calibration_draw_is_reproducible_and_seed_dependent() {
        let branch = BranchId::from("b-1");
        let draw = calibration_draw(&branch, 7);
        assert_eq!(draw, calibration_draw(&branch, 7));
        assert_ne!(draw, calibration_draw(&branch, 8));
        assert!((0.0..1.0).contains(&draw));
    }
}
