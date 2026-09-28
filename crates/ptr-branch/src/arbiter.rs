use sha2::{Digest, Sha256};

use ptr_analytics::clopper_pearson_upper;

use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

use crate::branch::BranchId;
use crate::error::ArbiterError;

/// What happens to a certified branch next.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TriageDecision {
    /// Propose the merge plan for commit. This is not authorisation over
    /// verification: the runtime triages only a merge its grant's verifiers
    /// admitted, and a human-approved plan is verified exactly the same way.
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
    /// Branches scoring at least this are auto-proposed. A policy holds only
    /// a finite threshold in `[0, 1]`, the range of a score:
    /// [`TriagePolicy::new`] refuses any other.
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
///
/// # Guarantees
/// Its fields are private, and it is built only by [`TriagePolicy::triage`]
/// and by [`TriageOutcome::from_parts`], which storage rebuilds a logged
/// triage with and which refuses parts no policy produces. So every
/// `TriageOutcome` equals (`==`) what [`TriagePolicy::triage`] returns for
/// some policy, verification report, score and calibration draw: its score is
/// a finite number in `[0, 1]`; a triage verification decided is discarded or
/// escalated, outside the calibration slice, with auto-propose propensity
/// zero; an eligible one is never discarded, is escalated in the slice with a
/// propensity below one, and outside it is auto-proposed exactly when its
/// propensity is positive. [`TriageOutcome::adjudicate`], and so every
/// [`CalibrationSample`], rests on that, and checks it again.
///
/// Which policy made a triage is not part of it, so nothing here says that
/// the policy a log cites for it made it: [`TriagePolicy::explains`] checks
/// that against the cited policy.
///
/// No field can be reached to change a built triage:
///
/// ```compile_fail
/// fn forge(mut triage: ptr_branch::TriageOutcome) -> ptr_branch::TriageOutcome {
///     triage.score = f32::NAN;
///     triage
/// }
/// ```
///
/// nor to build one around the constructor:
///
/// ```compile_fail
/// use ptr_branch::{TriageDecision, TriageOutcome};
/// let forged = TriageOutcome {
///     decision: TriageDecision::AutoPropose,
///     eligible: true,
///     calibration_slice: true,
///     score: 0.9,
///     auto_propensity: 0.8,
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TriageOutcome {
    decision: TriageDecision,
    eligible: bool,
    calibration_slice: bool,
    score: f32,
    auto_propensity: f64,
}

/// The parts of a triage: what [`TriageOutcome::into_parts`] returns and
/// [`TriageOutcome::from_parts`] validates, for example when storage rebuilds
/// a logged triage from its row.
///
/// Holding parts grants nothing: only a [`TriageOutcome`] is adjudicated into
/// a [`CalibrationSample`], and one exists only once its parts are a triage
/// some policy produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriageOutcomeParts {
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
///
/// Its fields are private and only [`TriageOutcome::adjudicate`] builds one,
/// from an eligible calibration-slice triage, so its score is a finite number
/// in `[0, 1]`: [`calibrate_threshold`] and [`certify_threshold`] compare it
/// with every grid threshold, and a NaN would keep a harmful sample out of
/// every admitted set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationSample {
    score: f32,
    harmful: bool,
}

impl CalibrationSample {
    /// The score of the adjudicated triage: a finite number in `[0, 1]`.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Whether the person judged that auto-proposing the branch would have
    /// been harmful.
    pub fn harmful(&self) -> bool {
        self.harmful
    }
}

impl TriagePolicy {
    /// Create a policy with a fraction `calibration_rate` of eligible branches
    /// reserved for human calibration. Zero disables that slice.
    ///
    /// Every way a policy is built — [`PolicyRecord::manual`],
    /// [`PolicyRecord::calibrate`] and [`PolicyRecord::from_parts`], which
    /// storage rebuilds records with — goes through here, so no policy holds
    /// a threshold outside `[0, 1]`.
    ///
    /// A positive rate is one for which `1 - calibration_rate`, the
    /// propensity the policy logs for an admitted score, is below one: every
    /// positive rate above `2^-54`. So a calibration-slice triage always has
    /// propensity below one, the escalation it records has a positive
    /// probability under the policy that made it, and [`Self::explains`],
    /// storage and off-policy evaluation agree on every triage the policy
    /// makes.
    ///
    /// # Errors
    /// Returns `ArbiterError::InvalidExploration` unless the rate is finite
    /// and in `[0, 1)`, or for a positive rate of at most `2^-54`, for which
    /// `1 - rate` rounds to one: its slice triages would be logged with
    /// auto-propose propensity one, as if escalating them were impossible.
    /// Returns `ArbiterError::InvalidThreshold` for an `AtLeast` threshold
    /// that is NaN, infinite or outside `[0, 1]`: a NaN admits no score and a
    /// negative threshold every score, so either would silently turn the
    /// policy into "never" or "always" auto-propose, and one above one is as
    /// unreachable as NaN while claiming otherwise.
    pub fn new(threshold: AutoThreshold, calibration_rate: f64) -> Result<Self, ArbiterError> {
        // `contains` is false for NaN and both infinities.
        if !(0.0..1.0).contains(&calibration_rate)
            || (calibration_rate > 0.0 && 1.0 - calibration_rate == 1.0)
        {
            return Err(ArbiterError::InvalidExploration {
                rate: calibration_rate,
            });
        }
        if let AutoThreshold::AtLeast(value) = threshold {
            // `contains` is false for NaN and both infinities.
            if !(0.0..=1.0).contains(&value) {
                return Err(ArbiterError::InvalidThreshold { value });
            }
        }
        Ok(Self {
            threshold,
            calibration_rate,
        })
    }

    pub fn threshold(&self) -> AutoThreshold {
        self.threshold
    }

    /// The fraction of eligible branches reserved for human calibration.
    pub fn calibration_rate(&self) -> f64 {
        self.calibration_rate
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

    /// Whether this policy can have produced `triage`: whether some
    /// verification report, the triage's score and some calibration draw in
    /// `[0, 1)` make [`Self::triage`] return exactly this decision, slice flag
    /// and auto-propose propensity. A triage log cites the policy that made
    /// each row; calibration and off-policy evaluation trust that citation,
    /// so storage checks it with this before a row is written.
    ///
    /// The rules are those of [`Self::triage`]. A branch verification decided
    /// (`eligible` false) is discarded or escalated, outside the calibration
    /// slice, with propensity zero, whatever the policy. An eligible branch is
    /// never discarded; its propensity is `1 - calibration_rate` when the
    /// threshold admits its score and zero otherwise, compared exactly (the
    /// policy computes it the same way, and storage keeps both bit for bit);
    /// in the calibration slice, which only a positive rate has, it is
    /// escalated, and outside it auto-proposed exactly when the threshold
    /// admits its score.
    ///
    /// Every [`TriageOutcome`] is one some policy produces
    /// ([`TriageOutcome::from_parts`]), so what this decides for one built
    /// through this crate is whether this policy is among them: whether its
    /// propensity is the one this policy logs for its score, and whether a
    /// slice triage cites a policy with a positive rate. The other rules
    /// are checked again rather than trusted.
    ///
    /// # Errors
    /// Returns `ArbiterError::UnexplainedTriage` naming the first rule the
    /// triage breaks, the first being a score that is not a finite number in
    /// `[0, 1]`: every triage sees a `Probability`.
    pub fn explains(&self, triage: &TriageOutcome) -> Result<(), ArbiterError> {
        let refuse = |reason| Err(ArbiterError::UnexplainedTriage { reason });
        if !(0.0..=1.0).contains(&triage.score) {
            return refuse("the score is not a probability");
        }
        if !triage.eligible {
            if triage.decision == TriageDecision::AutoPropose {
                return refuse("verification alone never auto-proposes");
            }
            if triage.calibration_slice {
                return refuse("a calibration-slice branch is eligible");
            }
            if triage.auto_propensity != 0.0 {
                return refuse("a branch verification decided has auto-propose propensity zero");
            }
            return Ok(());
        }
        if triage.decision == TriageDecision::Discard {
            return refuse("an eligible branch is never discarded");
        }
        let admitted = self.threshold.admits(triage.score);
        let propensity = if admitted {
            1.0 - self.calibration_rate
        } else {
            0.0
        };
        if triage.auto_propensity != propensity {
            return refuse(
                "the auto-propose propensity is not the one the policy logs for this score",
            );
        }
        if triage.calibration_slice {
            // `draw < calibration_rate` for a draw in [0, 1) needs a positive
            // rate.
            if self.calibration_rate == 0.0 {
                return refuse("a policy with calibration rate zero has no calibration slice");
            }
            if triage.decision != TriageDecision::Escalate {
                return refuse("a calibration-slice branch is escalated");
            }
            return Ok(());
        }
        // A rate below one leaves draws outside the slice.
        let expected = if admitted {
            TriageDecision::AutoPropose
        } else {
            TriageDecision::Escalate
        };
        if triage.decision != expected {
            return refuse(
                "outside the calibration slice the policy auto-proposes exactly the scores its \
                 threshold admits",
            );
        }
        Ok(())
    }

    /// Probability this policy takes `action` on a logged record. An
    /// ineligible record was decided by verification, which no policy
    /// outvotes, so any policy takes its decision with probability one.
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
    /// Rebuild a triage from its parts, for example ones storage read back,
    /// refusing any that no policy produces for any verification report,
    /// score and calibration draw. Checked in this order:
    ///
    /// - the score is a finite number in `[0, 1]`: every triage sees a
    ///   [`Probability`];
    /// - the auto-propose propensity `p` is a finite number in `[0, 1]`, and
    ///   a positive one is `1 - r` for some calibration rate `r` a policy can
    ///   hold, the propensity a policy logs for a score its threshold admits
    ///   (checked as `1 - (1 - p) == p`, which holds for every `p` in
    ///   `[1/2, 1]` and, below that, for the multiples of `2^-53`);
    /// - a triage verification decided (`eligible` false) is not
    ///   auto-proposed, not in the calibration slice, and has propensity zero;
    /// - an eligible triage is not discarded;
    /// - an eligible triage in the calibration slice is escalated and has
    ///   propensity below one: only a positive rate has a slice, and it logs
    ///   `1 - rate` or zero;
    /// - outside the slice, an eligible triage is auto-proposed exactly when
    ///   its propensity is positive: a policy auto-proposes there exactly the
    ///   scores its threshold admits, and logs a positive propensity for
    ///   exactly those.
    ///
    /// The rules are also sufficient: parts that pass equal what
    /// [`TriagePolicy::triage`] returns for some policy, report, score and
    /// draw (for a positive propensity `p`, the policy with rate `1 - p` and
    /// the score as its threshold). Whether the policy a log cites is one of
    /// them is what [`TriagePolicy::explains`] checks.
    ///
    /// # Errors
    /// `ArbiterError::ImpossibleOutcome` naming the first rule the parts
    /// break; nothing is built.
    pub fn from_parts(parts: TriageOutcomeParts) -> Result<Self, ArbiterError> {
        let triage = Self {
            decision: parts.decision,
            eligible: parts.eligible,
            calibration_slice: parts.calibration_slice,
            score: parts.score,
            auto_propensity: parts.auto_propensity,
        };
        match triage.contradiction() {
            Some(reason) => Err(ArbiterError::ImpossibleOutcome { reason }),
            None => Ok(triage),
        }
    }

    /// Give up the triage for its parts. Building one again goes through
    /// [`TriageOutcome::from_parts`].
    pub fn into_parts(self) -> TriageOutcomeParts {
        TriageOutcomeParts {
            decision: self.decision,
            eligible: self.eligible,
            calibration_slice: self.calibration_slice,
            score: self.score,
            auto_propensity: self.auto_propensity,
        }
    }

    pub fn decision(&self) -> TriageDecision {
        self.decision
    }

    /// Whether the rules left the decision to the policy at all. Ineligible
    /// branches are decided by verification alone.
    pub fn eligible(&self) -> bool {
        self.eligible
    }

    /// Whether this branch was escalated as part of the calibration slice.
    pub fn calibration_slice(&self) -> bool {
        self.calibration_slice
    }

    /// The score the policy saw: a finite number in `[0, 1]`.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Probability the logging policy assigned to auto-proposing this branch.
    pub fn auto_propensity(&self) -> f64 {
        self.auto_propensity
    }

    /// Turn a calibration-slice triage into a calibration sample once a person
    /// has adjudicated it, `harmful` saying whether auto-proposing it would
    /// have been harmful.
    ///
    /// The triage is checked against every rule [`TriageOutcome::from_parts`]
    /// checks before anything is built, although every `TriageOutcome` passed
    /// them when it was built: a sample's score is compared with every grid
    /// threshold by [`calibrate_threshold`] and [`certify_threshold`], so a
    /// score that is not a probability, or a sample of a triage no policy
    /// placed in the slice, would bias the threshold chosen from it rather
    /// than be refused there.
    ///
    /// # Errors
    /// `ArbiterError::ImpossibleOutcome` for a triage no policy produces, as
    /// [`TriageOutcome::from_parts`] names it; then
    /// `ArbiterError::NotCalibrationSlice` for any triage outside the
    /// calibration slice (auto-proposed, escalated below the threshold, or
    /// decided by verification): its outcome is not an exchangeable sample of
    /// eligible branches.
    pub fn adjudicate(&self, harmful: bool) -> Result<CalibrationSample, ArbiterError> {
        if let Some(reason) = self.contradiction() {
            return Err(ArbiterError::ImpossibleOutcome { reason });
        }
        // The rules above admit a slice triage only if it is eligible.
        if !self.calibration_slice {
            return Err(ArbiterError::NotCalibrationSlice);
        }
        Ok(CalibrationSample {
            score: self.score,
            harmful,
        })
    }

    /// The first rule of [`TriageOutcome::from_parts`] this triage breaks,
    /// if any.
    fn contradiction(&self) -> Option<&'static str> {
        // `contains` is false for NaN and both infinities.
        if !(0.0..=1.0).contains(&self.score) {
            return Some("the score is not a probability");
        }
        let propensity = self.auto_propensity;
        if !(0.0..=1.0).contains(&propensity) {
            return Some("the auto-propose propensity is not a probability");
        }
        if propensity > 0.0 && 1.0 - (1.0 - propensity) != propensity {
            return Some("a positive auto-propose propensity is 1 - r for a calibration rate r");
        }
        if !self.eligible {
            if self.decision == TriageDecision::AutoPropose {
                return Some("verification alone never auto-proposes");
            }
            if self.calibration_slice {
                return Some("a calibration-slice branch is eligible");
            }
            if propensity != 0.0 {
                return Some("a branch verification decided has auto-propose propensity zero");
            }
            return None;
        }
        if self.decision == TriageDecision::Discard {
            return Some("an eligible branch is never discarded");
        }
        if self.calibration_slice {
            if self.decision != TriageDecision::Escalate {
                return Some("a calibration-slice branch is escalated");
            }
            if propensity == 1.0 {
                return Some("a calibration-slice branch has auto-propose propensity below one");
            }
            return None;
        }
        if (self.decision == TriageDecision::AutoPropose) != (propensity > 0.0) {
            return Some(
                "outside the calibration slice an eligible branch is auto-proposed exactly when \
                 its auto-propose propensity is positive",
            );
        }
        None
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
/// failure and the lowest passing threshold is returned. Thresholds at which
/// even no harm could not pass, `UCB(0, n_t; delta) > alpha`, are skipped, so
/// the sequence starts at the first threshold admitting enough samples to
/// pass; that start is decided by the same bound the test uses, on the scores
/// only, never on the harm labels under test. Then, with probability at
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
    let upper = |harmful, admitted| {
        clopper_pearson_upper(harmful, admitted, delta).map_err(|_| ArbiterError::InvalidRisk {
            field: "delta",
            value: delta,
        })
    };
    let mut certified = AutoThreshold::Never;
    // The admitted count only grows as the threshold falls, and the no-harm
    // bound only falls as it grows, so the skipped thresholds are a prefix.
    let mut started = false;
    for threshold in threshold_grid().rev() {
        let admitted = samples.iter().filter(|s| s.score >= threshold).count() as u64;
        if !started {
            if upper(0, admitted)? > alpha {
                continue;
            }
            started = true;
        }
        let harmful = samples
            .iter()
            .filter(|s| s.harmful && s.score >= threshold)
            .count() as u64;
        let bound = upper(harmful, admitted)?;
        if bound <= alpha {
            certified = AutoThreshold::AtLeast(threshold);
        } else {
            break;
        }
    }
    Ok(certified)
}

/// How a recorded policy's threshold was chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThresholdRule {
    /// Set by a person without calibration, for example
    /// [`AutoThreshold::Never`] before any adjudication exists.
    Manual,
    /// [`calibrate_threshold`] at risk level `alpha`.
    ConformalRiskControl { alpha: f64 },
    /// [`certify_threshold`] at risk level `alpha` and confidence `1 - delta`.
    LearnThenTest { alpha: f64, delta: f64 },
}

impl ThresholdRule {
    /// Stable machine name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::ConformalRiskControl { .. } => "conformal_risk_control",
            Self::LearnThenTest { .. } => "learn_then_test",
        }
    }

    fn check(self) -> Result<(), ArbiterError> {
        match self {
            Self::Manual => Ok(()),
            Self::ConformalRiskControl { alpha } => check_level("alpha", alpha),
            Self::LearnThenTest { alpha, delta } => {
                check_level("alpha", alpha)?;
                check_level("delta", delta)
            }
        }
    }
}

/// A triage policy as it is recorded and cited by triage logs: its version,
/// the policy, the rule its threshold is attributed to, and the adjudicated
/// calibration-slice branches it names as the ones that rule saw.
///
/// Naming the calibration branches is what keeps a later evaluation honest:
/// a policy's harm rate may only be estimated on adjudications it was not
/// calibrated on ([`PolicyRecord::held_out`]), which F003 requires. The unit
/// is the branch, so a branch appears at most once.
///
/// The attribution holds by construction only for a record
/// [`PolicyRecord::calibrate`] builds: it runs the rule on the samples it is
/// given and names their branches. [`PolicyRecord::from_parts`], which
/// storage rebuilds records with, has no adjudications to rerun the rule on
/// and takes the threshold and the calibration branches as given, so a
/// record it builds may name branches the rule, run on their adjudications,
/// would not choose its threshold from. Whoever records a policy checks that
/// against the stored adjudications (ptr-pg's `record_policy` reruns the rule
/// in the transaction that writes the record).
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyRecord {
    version: String,
    policy: TriagePolicy,
    rule: ThresholdRule,
    calibrated_on: Vec<BranchId>,
}

impl PolicyRecord {
    /// A policy whose threshold a person set, calibrated on nothing.
    ///
    /// # Errors
    /// Refuses an empty version.
    pub fn manual(version: impl Into<String>, policy: TriagePolicy) -> Result<Self, ArbiterError> {
        Self::from_parts(
            version,
            policy.threshold(),
            policy.calibration_rate,
            ThresholdRule::Manual,
            Vec::new(),
        )
    }

    /// Choose a threshold with `rule` from adjudicated calibration-slice
    /// samples, keyed by their branch, and record which branches it used.
    ///
    /// # Errors
    /// Refuses an empty version, a manual rule (use [`PolicyRecord::manual`]),
    /// an invalid calibration rate or risk level, no samples, and a branch
    /// that appears twice.
    pub fn calibrate(
        version: impl Into<String>,
        rule: ThresholdRule,
        calibration_rate: f64,
        samples: &[(BranchId, CalibrationSample)],
    ) -> Result<Self, ArbiterError> {
        let scored: Vec<CalibrationSample> = samples.iter().map(|(_, sample)| *sample).collect();
        let threshold = match rule {
            ThresholdRule::Manual => {
                return Err(ArbiterError::InvalidRule {
                    rule: rule.name(),
                    message: "a manual threshold is recorded with PolicyRecord::manual",
                })
            }
            ThresholdRule::ConformalRiskControl { alpha } => calibrate_threshold(&scored, alpha)?,
            ThresholdRule::LearnThenTest { alpha, delta } => {
                certify_threshold(&scored, alpha, delta)?
            }
        };
        Self::from_parts(
            version,
            threshold,
            calibration_rate,
            rule,
            samples.iter().map(|(branch, _)| branch.clone()).collect(),
        )
    }

    /// Rebuild a record, for example one read back from storage. The
    /// calibration branches are kept in branch order.
    ///
    /// The threshold and the calibration branches are taken as given:
    /// nothing here reruns the rule, so the record names the branches its
    /// threshold is attributed to without showing that the rule chose it
    /// from them (see [`PolicyRecord`]).
    ///
    /// # Errors
    /// Refuses an empty version, an invalid calibration rate or risk level, a
    /// manual rule that names calibration branches, a calibrated rule that
    /// names none, and a branch that appears twice.
    pub fn from_parts(
        version: impl Into<String>,
        threshold: AutoThreshold,
        calibration_rate: f64,
        rule: ThresholdRule,
        mut calibrated_on: Vec<BranchId>,
    ) -> Result<Self, ArbiterError> {
        let version = version.into();
        if version.is_empty() {
            return Err(ArbiterError::EmptyVersion);
        }
        rule.check()?;
        let policy = TriagePolicy::new(threshold, calibration_rate)?;
        match (rule, calibrated_on.is_empty()) {
            (ThresholdRule::Manual, false) => {
                return Err(ArbiterError::InvalidRule {
                    rule: rule.name(),
                    message: "a manual threshold must not name calibration branches",
                })
            }
            (ThresholdRule::Manual, true) => {}
            (_, true) => return Err(ArbiterError::EmptyCalibration),
            (_, false) => {}
        }
        calibrated_on.sort();
        if let Some(pair) = calibrated_on.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(ArbiterError::DuplicateCalibrationBranch {
                branch: pair[0].0.clone(),
            });
        }
        Ok(Self {
            version,
            policy,
            rule,
            calibrated_on,
        })
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn policy(&self) -> TriagePolicy {
        self.policy
    }

    pub fn rule(&self) -> ThresholdRule {
        self.rule
    }

    /// The branches the threshold was calibrated on, in branch order.
    pub fn calibrated_on(&self) -> &[BranchId] {
        &self.calibrated_on
    }

    /// The adjudicated samples this policy was not calibrated on: the only
    /// ones its harm rate may be estimated from.
    ///
    /// Disjointness is necessary for an honest held-out estimate, not
    /// sufficient. The calibration subset, the rule and its levels must have
    /// been fixed before anyone looked at the outcomes of these samples, as
    /// the F003 pre-registration requires; a policy tuned until its held-out
    /// harm looked acceptable has seen them. This returns the complement of
    /// the recorded calibration set and cannot check that precondition.
    pub fn held_out<'a>(
        &self,
        adjudicated: &'a [(BranchId, CalibrationSample)],
    ) -> Vec<&'a (BranchId, CalibrationSample)> {
        adjudicated
            .iter()
            .filter(|(branch, _)| self.calibrated_on.binary_search(branch).is_err())
            .collect()
    }
}

fn check_level(field: &'static str, value: f64) -> Result<(), ArbiterError> {
    if value.is_finite() && value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(ArbiterError::InvalidRisk { field, value })
    }
}

/// One logged triage with the reward its decision earned.
///
/// The fields are public, so a record can say anything. The off-policy
/// estimators ([`evaluate_off_policy`], [`doubly_robust`]) refuse a log
/// holding a record that [`TriagePolicy::triage`] cannot return under any
/// threshold and calibration rate (`ArbiterError::ImpossibleTriage`), checked
/// before the record's importance weight is computed and before any estimate
/// is formed or a reward model consulted:
///
/// - an ineligible record (verification decided it) is discarded or
///   escalated, never auto-proposed, and has auto-propose propensity zero;
/// - an eligible record is never discarded;
/// - an eligible auto-proposed record has a positive propensity, since only a
///   score the threshold admits is auto-proposed and its propensity is
///   `1 - calibration_rate`;
/// - an eligible escalated record has a propensity below one, since
///   propensity one means calibration rate zero and an admitted score, which
///   is always auto-proposed.
///
/// A record does not name the policy that logged it, so the estimators cannot
/// check that its propensity is the one that policy gives its score, nor
/// that its decision is the one that policy makes for it outside the
/// calibration slice: [`TriagePolicy::explains`] checks that against the
/// cited policy, and storage runs it before a triage row is written. Nor do
/// they check that a positive propensity is one some calibration rate yields
/// (every positive propensity a policy logs is at least `2^-53`): any
/// propensity in `[0, 1]` that fits the rules above is used as given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoggedTriage {
    pub eligible: bool,
    /// The score the logging policy saw: a finite number in `[0, 1]`, which
    /// the off-policy estimators check before reweighting by it.
    pub score: f32,
    pub decision: TriageDecision,
    /// The logging policy's probability of auto-proposing this record: zero
    /// for an ineligible record.
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
    /// The first rule of [`TriagePolicy::triage`] this record breaks whatever
    /// the policy, if any. Checked only once its propensity is known to be in
    /// `[0, 1]`.
    fn contradiction(&self) -> Option<&'static str> {
        if !self.eligible {
            if self.decision == TriageDecision::AutoPropose {
                return Some("verification alone never auto-proposes");
            }
            if self.auto_propensity != 0.0 {
                return Some("a branch verification decided has auto-propose propensity zero");
            }
            return None;
        }
        match self.decision {
            TriageDecision::Discard => Some("an eligible branch is never discarded"),
            TriageDecision::AutoPropose if self.auto_propensity == 0.0 => Some(
                "an eligible branch is auto-proposed only with positive auto-propose propensity",
            ),
            TriageDecision::Escalate if self.auto_propensity == 1.0 => {
                Some("an eligible branch with auto-propose propensity one is never escalated")
            }
            TriageDecision::AutoPropose | TriageDecision::Escalate => None,
        }
    }

    /// The logging policy's probability of `action` on this record. An
    /// ineligible record was decided by verification, so its decision had
    /// probability one; the record's propensity is zero, which
    /// [`Self::contradiction`] checks before this is used.
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
///
/// SNIPS and the effective sample size do not change when every weight is
/// multiplied by one factor, so both are computed on the weights divided by
/// the largest: SNIPS as a convex combination of the rewards, at most the
/// largest reward's magnitude up to rounding, and the effective sample size
/// from sums between one and the log's length. IPS is SNIPS times the mean
/// weight. No estimate overflows merely because a sum of weighted rewards or
/// of squared weights would, and one that is still not finite is refused.
///
/// # Errors
/// Refuses an empty log (`EmptyLog`), then, at the first record that shows
/// one and in this order for each record: a propensity outside `[0, 1]`
/// (`InvalidPropensity`), a score that is not a finite number in `[0, 1]`
/// (`InvalidScore`, before any target probability is computed from it), a
/// reward that is not finite (`InvalidReward`), a record no policy can have
/// produced, such as an ineligible auto-proposal or an ineligible record with
/// a nonzero propensity (`ImpossibleTriage`, naming the rule it breaks; see
/// [`LoggedTriage`]), a positivity violation (`PositivityViolation`), and a
/// logged action whose probability is so small that the importance weight is
/// not finite (`InvalidPropensity`); then an estimate that is not finite
/// (`NonFiniteEstimate`).
/// That includes a log on which `target` gives every logged action
/// probability zero: every weight is zero, so SNIPS is `0 / 0` and the
/// effective sample size is undefined, and the log says nothing about what
/// `target` would earn (`NonFiniteEstimate { estimate: "snips" }`, rather
/// than a reward of zero). Nothing nonfinite is ever returned as an
/// estimate.
pub fn evaluate_off_policy(
    log: &[LoggedTriage],
    target: &TriagePolicy,
) -> Result<OffPolicyEstimate, ArbiterError> {
    let weights = importance_weights(log, target)?;
    let largest = weights.iter().copied().fold(0.0, f64::max);
    if largest == 0.0 {
        // The target takes none of the logged actions: every weight is zero
        // and SNIPS is 0 / 0.
        return Err(ArbiterError::NonFiniteEstimate { estimate: "snips" });
    }
    let scaled: Vec<f64> = weights.iter().map(|weight| weight / largest).collect();
    let total: f64 = scaled.iter().sum();
    let squares: f64 = scaled.iter().map(|weight| weight * weight).sum();
    let snips: f64 = scaled
        .iter()
        .zip(log)
        .map(|(weight, record)| weight / total * record.reward)
        .sum();
    let estimate = OffPolicyEstimate {
        ips: snips * (largest * (total / log.len() as f64)),
        snips,
        effective_sample_size: total * total / squares,
    };
    for (name, value) in [
        ("ips", estimate.ips),
        ("snips", estimate.snips),
        ("effective_sample_size", estimate.effective_sample_size),
    ] {
        if !value.is_finite() {
            return Err(ArbiterError::NonFiniteEstimate { estimate: name });
        }
    }
    Ok(estimate)
}

/// Doubly robust estimate: a reward model's prediction for the target policy,
/// corrected by importance-weighted residuals on the logged actions. Unbiased
/// when either the propensities or the reward model are correct.
///
/// The estimate is `D + 2 L R`, evaluated as `2 (D / 2 + L R)`: `D` averages
/// the direct predictions, each divided by the log's length `n` before they
/// are added, and `R` sums each record's weight divided by the largest, `L`,
/// times its residual `reward / 2n - prediction / 2n`. Every term of `D` and
/// `R` is then at most the largest reward or prediction over `n` in
/// magnitude, and `L` multiplies back only once, so a large finite weight, a
/// residual near the largest finite reward or a long log does not overflow an
/// estimate that is itself finite (up to rounding at the limit of `f64`).
///
/// `reward_model` is called exactly once for each record and each of the
/// three actions: in log order, and for each record for `AutoPropose`,
/// `Escalate` and `Discard` in that order. The logged action's prediction is
/// the one used both in `D` and in the record's residual, so the two
/// cancel as the estimator requires even when the model is stochastic,
/// stateful or an inference call whose answer can change, and the estimate
/// rests on one prediction per record and action.
///
/// # Errors
/// Refuses the logs [`evaluate_off_policy`] refuses for their records, with
/// the same error and before `reward_model` is called: an empty log, a
/// nonfinite logged reward, a score outside `[0, 1]`, a record no policy can
/// have produced (`ImpossibleTriage`) or an infinite importance weight
/// included. Then refuses an estimate that is not finite
/// (`NonFiniteEstimate`), as a reward model's nonfinite prediction makes it.
pub fn doubly_robust<M>(
    log: &[LoggedTriage],
    target: &TriagePolicy,
    mut reward_model: M,
) -> Result<f64, ArbiterError>
where
    M: FnMut(&LoggedTriage, TriageDecision) -> f64,
{
    let weights = importance_weights(log, target)?;
    let n = log.len() as f64;
    // Every weight is finite and nonnegative; when all are zero, dividing
    // them by one keeps them zero.
    let largest = weights.iter().copied().fold(0.0, f64::max);
    let scale = if largest > 0.0 { largest } else { 1.0 };
    let mut direct_mean = 0.0;
    let mut residuals = 0.0;
    for (record, weight) in log.iter().zip(&weights) {
        let [auto, escalate, discard] = [
            TriageDecision::AutoPropose,
            TriageDecision::Escalate,
            TriageDecision::Discard,
        ]
        .map(|action| reward_model(record, action));
        let direct = target.probability(record, TriageDecision::AutoPropose) * auto
            + target.probability(record, TriageDecision::Escalate) * escalate
            + target.probability(record, TriageDecision::Discard) * discard;
        direct_mean += direct / n;
        let prediction = match record.decision {
            TriageDecision::AutoPropose => auto,
            TriageDecision::Escalate => escalate,
            TriageDecision::Discard => discard,
        };
        residuals += weight / scale * (record.reward / (2.0 * n) - prediction / (2.0 * n));
    }
    let estimate = 2.0 * (direct_mean / 2.0 + scale * residuals);
    if !estimate.is_finite() {
        return Err(ArbiterError::NonFiniteEstimate {
            estimate: "doubly_robust",
        });
    }
    Ok(estimate)
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
        // `contains` is false for NaN and both infinities. Checked before
        // any target probability, which is computed from the score.
        if !(0.0..=1.0).contains(&record.score) {
            return Err(ArbiterError::InvalidScore {
                index,
                value: record.score,
            });
        }
        if !record.reward.is_finite() {
            return Err(ArbiterError::InvalidReward {
                index,
                value: record.reward,
            });
        }
        // Both probability helpers trust the record's eligibility, decision
        // and propensity to fit together: an ineligible auto-proposal would
        // otherwise be reweighted by one under every target.
        if let Some(reason) = record.contradiction() {
            return Err(ArbiterError::ImpossibleTriage { index, reason });
        }
        for action in actions {
            if target.probability(record, action) > 0.0 && record.logging_probability(action) == 0.0
            {
                return Err(ArbiterError::PositivityViolation { index });
            }
        }
        let logged = record.logging_probability(record.decision);
        let weight = target.probability(record, record.decision) / logged;
        // A positive probability so small that the weight overflows. A zero
        // one contradicts the record's own fields and was refused above; it
        // is checked again so that no division by zero is ever returned.
        if logged <= 0.0 || !weight.is_finite() {
            return Err(ArbiterError::InvalidPropensity {
                index,
                value: logged,
            });
        }
        weights.push(weight);
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

    /// A triage built without [`TriageOutcome::from_parts`], as no caller
    /// outside this module can build one.
    fn unchecked(
        decision: TriageDecision,
        eligible: bool,
        calibration_slice: bool,
        score: f32,
        auto_propensity: f64,
    ) -> TriageOutcome {
        TriageOutcome {
            decision,
            eligible,
            calibration_slice,
            score,
            auto_propensity,
        }
    }

    #[test]
    fn adjudication_checks_the_triage_again_rather_than_trusting_its_constructor() {
        for (forged, reason) in [
            (
                unchecked(TriageDecision::Escalate, true, true, f32::NAN, 0.0),
                "the score is not a probability",
            ),
            (
                unchecked(TriageDecision::AutoPropose, true, true, 0.9, 0.8),
                "a calibration-slice branch is escalated",
            ),
            (
                unchecked(TriageDecision::Escalate, true, true, 0.9, 1.0),
                "a calibration-slice branch has auto-propose propensity below one",
            ),
            (
                unchecked(TriageDecision::Escalate, false, true, 0.9, 0.0),
                "a calibration-slice branch is eligible",
            ),
        ] {
            assert_eq!(
                forged.adjudicate(true),
                Err(ArbiterError::ImpossibleOutcome { reason }),
                "{forged:?}"
            );
        }
        let slice = unchecked(TriageDecision::Escalate, true, true, 0.9, 0.8);
        assert_eq!(slice.adjudicate(true), Ok(sample(0.9, true)));
    }

    #[test]
    fn explains_checks_again_every_rule_the_constructor_enforces() {
        let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.25).unwrap();
        let outside = "outside the calibration slice the policy auto-proposes exactly the \
                       scores its threshold admits";
        for (forged, reason) in [
            (
                unchecked(TriageDecision::Escalate, true, false, f32::NAN, 0.0),
                "the score is not a probability",
            ),
            (
                unchecked(TriageDecision::Escalate, true, false, 0.9, 0.75),
                outside,
            ),
            (
                unchecked(TriageDecision::AutoPropose, true, false, 0.2, 0.0),
                outside,
            ),
            (
                unchecked(TriageDecision::Discard, true, false, 0.2, 0.0),
                "an eligible branch is never discarded",
            ),
            (
                unchecked(TriageDecision::AutoPropose, true, true, 0.9, 0.75),
                "a calibration-slice branch is escalated",
            ),
            (
                unchecked(TriageDecision::AutoPropose, false, false, 0.9, 0.0),
                "verification alone never auto-proposes",
            ),
            (
                unchecked(TriageDecision::Discard, false, true, 0.9, 0.0),
                "a calibration-slice branch is eligible",
            ),
            (
                unchecked(TriageDecision::Discard, false, false, 0.9, 0.75),
                "a branch verification decided has auto-propose propensity zero",
            ),
        ] {
            assert!(forged.contradiction().is_some(), "{forged:?}");
            assert_eq!(
                policy.explains(&forged),
                Err(ArbiterError::UnexplainedTriage { reason }),
                "{forged:?}"
            );
        }
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
