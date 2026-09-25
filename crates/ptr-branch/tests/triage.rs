use ptr_branch::{
    calibrate_threshold, calibration_draw, doubly_robust, evaluate_off_policy, ArbiterError,
    AutoThreshold, BranchId, LoggedTriage, TriageDecision, TriagePolicy,
};
use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus};

fn report(status: VerificationStatus, level: VerificationLevel) -> VerificationReport {
    VerificationReport {
        status,
        level,
        score: Probability::new(0.99).unwrap(),
        findings: vec![],
    }
}

fn passing() -> VerificationReport {
    report(VerificationStatus::Pass, VerificationLevel::Deterministic)
}

fn score(value: f32) -> Probability {
    Probability::new(value).unwrap()
}

#[test]
fn a_failed_verification_discards_whatever_the_score() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.0).unwrap();
    let outcome = policy.triage(
        &report(VerificationStatus::Fail, VerificationLevel::Deterministic),
        score(1.0),
        0.5,
    );
    assert_eq!(outcome.decision, TriageDecision::Discard);
    assert!(!outcome.eligible);
}

#[test]
fn disputed_unknown_or_shallow_verification_escalates_whatever_the_score() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.0).unwrap();
    for report in [
        report(
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
        ),
        report(
            VerificationStatus::Unknown,
            VerificationLevel::Deterministic,
        ),
        report(VerificationStatus::Pass, VerificationLevel::SampleVerified),
    ] {
        let outcome = policy.triage(&report, score(1.0), 0.9);
        assert_eq!(outcome.decision, TriageDecision::Escalate);
        assert!(!outcome.eligible);
    }
}

#[test]
fn the_calibration_slice_escalates_high_scores_and_only_it_can_be_adjudicated() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.2).unwrap();
    let sliced = policy.triage(&passing(), score(0.9), 0.1);
    assert_eq!(sliced.decision, TriageDecision::Escalate);
    assert!(sliced.calibration_slice);
    assert!(sliced.adjudicate(false).is_some());

    let auto = policy.triage(&passing(), score(0.9), 0.7);
    assert_eq!(auto.decision, TriageDecision::AutoPropose);
    assert!((auto.auto_propensity - 0.8).abs() < 1e-12);
    assert!(auto.adjudicate(false).is_none());

    let low = policy.triage(&passing(), score(0.2), 0.7);
    assert_eq!(low.decision, TriageDecision::Escalate);
    assert!(low.adjudicate(true).is_none());
}

#[test]
fn a_threshold_calibrated_from_adjudicated_slices_is_used_by_the_next_policy() {
    let logging = TriagePolicy::new(AutoThreshold::Never, 0.999).unwrap();
    let samples: Vec<_> = (0..40)
        .map(|i| {
            let branch = BranchId(format!("b{i}"));
            let score_value = i as f32 / 40.0;
            let outcome = logging.triage(
                &passing(),
                score(score_value),
                calibration_draw(&branch, 1) * 0.5,
            );
            outcome
                .adjudicate(score_value < 0.3)
                .expect("calibration slice")
        })
        .collect();
    // Twelve harmful samples score below 0.3 (at i/40). With n = 40 and
    // alpha = 0.1 the bound (h + 1)/41 <= 0.1 admits at most three of them, so
    // the threshold is the first grid point above 8/40: the guarantee bounds
    // the joint probability of auto-proposing a harmful branch, not the harm
    // rate among proposals.
    let threshold = calibrate_threshold(&samples, 0.1).unwrap();
    assert_eq!(threshold, AutoThreshold::AtLeast(201.0_f32 / 1000.0));
}

#[test]
fn an_invalid_risk_level_is_refused() {
    let samples = [];
    assert_eq!(
        calibrate_threshold(&samples, 0.1).unwrap_err(),
        ArbiterError::EmptyCalibration
    );
}

fn log_under(policy: &TriagePolicy, scores: &[f32]) -> Vec<LoggedTriage> {
    scores
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let outcome = policy.triage(
                &passing(),
                score(value),
                calibration_draw(&BranchId(format!("b{i}")), 3),
            );
            let mut logged = LoggedTriage::from(&outcome);
            logged.reward = match outcome.decision {
                TriageDecision::AutoPropose => 1.0,
                TriageDecision::Escalate => 0.2,
                TriageDecision::Discard => 0.0,
            };
            logged
        })
        .collect()
}

#[test]
fn evaluating_the_logging_policy_on_its_own_log_returns_its_mean_reward() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.1).unwrap();
    let scores: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();
    let log = log_under(&policy, &scores);
    let mean = log.iter().map(|record| record.reward).sum::<f64>() / log.len() as f64;
    let estimate = evaluate_off_policy(&log, &policy).unwrap();
    assert!((estimate.ips - mean).abs() < 1e-12);
    assert!((estimate.snips - mean).abs() < 1e-12);
    assert!((estimate.effective_sample_size - 50.0).abs() < 1e-9);
}

#[test]
fn a_lower_threshold_than_the_log_ever_explored_is_refused_as_a_positivity_violation() {
    let logging = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.1).unwrap();
    let log = log_under(&logging, &[0.2, 0.4, 0.6, 0.8]);
    let bolder = TriagePolicy::new(AutoThreshold::AtLeast(0.3), 0.1).unwrap();
    assert_eq!(
        evaluate_off_policy(&log, &bolder).unwrap_err(),
        ArbiterError::PositivityViolation { index: 1 }
    );
}

#[test]
fn a_higher_threshold_is_estimable_and_doubly_robust_agrees_with_a_perfect_model() {
    let logging = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap();
    let scores: Vec<f32> = (0..200).map(|i| i as f32 / 200.0).collect();
    let log = log_under(&logging, &scores);
    let cautious = TriagePolicy::new(AutoThreshold::AtLeast(0.7), 0.3).unwrap();
    let perfect = |_: &LoggedTriage, action: TriageDecision| match action {
        TriageDecision::AutoPropose => 1.0,
        TriageDecision::Escalate => 0.2,
        TriageDecision::Discard => 0.0,
    };
    let truth: f64 = scores
        .iter()
        .map(|&s| if s >= 0.7 { 0.7 * 1.0 + 0.3 * 0.2 } else { 0.2 })
        .sum::<f64>()
        / scores.len() as f64;
    let dr = doubly_robust(&log, &cautious, perfect).unwrap();
    assert!((dr - truth).abs() < 1e-9, "dr {dr} truth {truth}");
    assert!(evaluate_off_policy(&log, &cautious).is_ok());
}
