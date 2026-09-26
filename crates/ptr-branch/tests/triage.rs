use ptr_branch::{
    calibrate_threshold, calibration_draw, certify_threshold, doubly_robust, evaluate_off_policy,
    ArbiterError, AutoThreshold, BranchId, CalibrationSample, LoggedTriage, PolicyRecord,
    ThresholdRule, TriageDecision, TriagePolicy,
};
use ptr_types::{Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

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
fn threshold_equality_is_admitted_and_calibration_equality_is_not_sampled() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.25).unwrap();
    let boundary = policy.triage(&passing(), score(0.5), 0.25).unwrap();
    assert_eq!(boundary.decision, TriageDecision::AutoPropose);
    assert!(boundary.eligible);
    assert!(!boundary.calibration_slice);
    assert_eq!(boundary.auto_propensity, 0.75);
    let sampled = policy.triage(&passing(), score(0.5), 0.0).unwrap();
    assert_eq!(sampled.decision, TriageDecision::Escalate);
    assert!(sampled.adjudicate(false).is_some());
    let below = policy.triage(&passing(), score(0.49), 0.25).unwrap();
    assert_eq!(below.decision, TriageDecision::Escalate);
    assert_eq!(below.auto_propensity, 0.0);
}

#[test]
fn a_draw_outside_the_unit_interval_is_refused_rather_than_skewing_the_slice() {
    // With a NaN or a draw of at least one, `draw < rate` is false for every
    // branch, so an admitted branch would always be auto-proposed while its
    // logged propensity still claimed 1 - rate; a negative draw would put
    // every branch in the slice.
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.25).unwrap();
    for draw in [
        f64::NAN,
        1.0,
        1.5,
        f64::INFINITY,
        -0.1,
        -f64::MIN_POSITIVE,
        f64::NEG_INFINITY,
    ] {
        let refused = policy.triage(&passing(), score(0.9), draw);
        assert!(
            matches!(refused, Err(ArbiterError::InvalidDraw { .. })),
            "draw {draw} gave {refused:?}"
        );
        assert_eq!(refused.unwrap_err().code(), "PTR_ARBITER_INVALID_DRAW");
        // The draw is the caller's input, so it is refused whatever the
        // verification report says.
        let failed = report(VerificationStatus::Fail, VerificationLevel::Deterministic);
        assert!(matches!(
            policy.triage(&failed, score(0.9), draw),
            Err(ArbiterError::InvalidDraw { .. })
        ));
    }
    // The endpoints of the documented range are accepted.
    assert!(policy.triage(&passing(), score(0.9), 0.0).is_ok());
    assert!(policy
        .triage(&passing(), score(0.9), 1.0 - f64::EPSILON / 2.0)
        .is_ok());
}

#[test]
fn a_hard_finding_excludes_a_passing_report_from_proposals_and_calibration() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.5).unwrap();
    let mut verified = passing();
    verified.findings.push(Finding {
        code: "unsafe".into(),
        message: "constraint failed".into(),
        hard: true,
    });
    let outcome = policy.triage(&verified, score(1.0), 0.0).unwrap();
    assert_eq!(outcome.decision, TriageDecision::Escalate);
    assert!(!outcome.eligible);
    assert!(!outcome.calibration_slice);
    assert_eq!(outcome.auto_propensity, 0.0);
    assert_eq!(outcome.adjudicate(false), None);
}

#[test]
fn off_policy_estimators_refuse_an_empty_log() {
    let target = TriagePolicy::new(AutoThreshold::Never, 0.0).unwrap();
    assert_eq!(
        evaluate_off_policy(&[], &target),
        Err(ArbiterError::EmptyLog)
    );
    assert_eq!(
        doubly_robust(&[], &target, |_, _| panic!(
            "empty log must be rejected first"
        )),
        Err(ArbiterError::EmptyLog)
    );
}

#[test]
fn a_failed_verification_discards_whatever_the_score() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.0).unwrap();
    let outcome = policy
        .triage(
            &report(VerificationStatus::Fail, VerificationLevel::Deterministic),
            score(1.0),
            0.5,
        )
        .unwrap();
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
        let outcome = policy.triage(&report, score(1.0), 0.9).unwrap();
        assert_eq!(outcome.decision, TriageDecision::Escalate);
        assert!(!outcome.eligible);
    }
}

#[test]
fn the_calibration_slice_escalates_high_scores_and_only_it_can_be_adjudicated() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.2).unwrap();
    let sliced = policy.triage(&passing(), score(0.9), 0.1).unwrap();
    assert_eq!(sliced.decision, TriageDecision::Escalate);
    assert!(sliced.calibration_slice);
    assert!(sliced.adjudicate(false).is_some());

    let auto = policy.triage(&passing(), score(0.9), 0.7).unwrap();
    assert_eq!(auto.decision, TriageDecision::AutoPropose);
    assert!((auto.auto_propensity - 0.8).abs() < 1e-12);
    assert!(auto.adjudicate(false).is_none());

    let low = policy.triage(&passing(), score(0.2), 0.7).unwrap();
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
            let outcome = logging
                .triage(
                    &passing(),
                    score(score_value),
                    calibration_draw(&branch, 1) * 0.5,
                )
                .unwrap();
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

    // The next policy auto-proposes what the calibrated threshold admits and
    // escalates the rest.
    let next = TriagePolicy::new(threshold, 0.0).unwrap();
    assert_eq!(
        next.triage(&passing(), score(0.5), 0.5).unwrap().decision,
        TriageDecision::AutoPropose
    );
    assert_eq!(
        next.triage(&passing(), score(0.1), 0.5).unwrap().decision,
        TriageDecision::Escalate
    );
}

#[test]
fn a_certified_threshold_bounds_the_harm_rate_among_what_the_next_policy_proposes() {
    let logging = TriagePolicy::new(AutoThreshold::Never, 0.999).unwrap();
    // Sixty clean branches score high; twenty score low and every second one
    // of those is harmful.
    let labelled: Vec<(f32, bool)> = (0..60)
        .map(|i| (0.5 + i as f32 / 200.0, false))
        .chain((0..20).map(|i| (0.1 + i as f32 / 100.0, i % 2 == 0)))
        .collect();
    let samples: Vec<_> = labelled
        .iter()
        .enumerate()
        .map(|(i, &(value, harmful))| {
            let branch = BranchId(format!("c{i}"));
            logging
                .triage(&passing(), score(value), calibration_draw(&branch, 7) * 0.5)
                .unwrap()
                .adjudicate(harmful)
                .expect("calibration slice")
        })
        .collect();
    // Learn-then-Test with Clopper-Pearson bounds at alpha = delta = 0.1.
    let threshold = certify_threshold(&samples, 0.1, 0.1).unwrap();
    let AutoThreshold::AtLeast(cut) = threshold else {
        panic!("sixty clean samples certify a threshold");
    };
    let admitted: Vec<_> = labelled.iter().filter(|(value, _)| *value >= cut).collect();
    let harmful = admitted.iter().filter(|(_, harmful)| *harmful).count();
    assert!(!admitted.is_empty());
    assert!(harmful as f64 / admitted.len() as f64 <= 0.1, "{cut}");

    let next = TriagePolicy::new(threshold, 0.05).unwrap();
    assert_eq!(
        next.triage(&passing(), score(0.9), 0.5).unwrap().decision,
        TriageDecision::AutoPropose
    );
    assert_eq!(
        next.triage(&passing(), score(0.1), 0.5).unwrap().decision,
        TriageDecision::Escalate
    );
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
            let outcome = policy
                .triage(
                    &passing(),
                    score(value),
                    calibration_draw(&BranchId(format!("b{i}")), 3),
                )
                .unwrap();
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

#[test]
fn a_nonfinite_logged_reward_is_refused_by_every_off_policy_estimate() {
    let logging = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap();
    let cautious = TriagePolicy::new(AutoThreshold::AtLeast(0.7), 0.3).unwrap();
    let model = |_: &LoggedTriage, _: TriageDecision| 0.5;
    for reward in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut log = log_under(&logging, &[0.2, 0.6, 0.8]);
        log[1].reward = reward;
        for target in [&logging, &cautious] {
            let refused = evaluate_off_policy(&log, target).unwrap_err();
            assert!(
                matches!(refused, ArbiterError::InvalidReward { index: 1, value }
                    if value.to_bits() == reward.to_bits()),
                "{reward}: {refused:?}"
            );
            assert_eq!(refused.code(), "PTR_ARBITER_INVALID_REWARD");
            assert!(
                matches!(
                    doubly_robust(&log, target, model),
                    Err(ArbiterError::InvalidReward { index: 1, .. })
                ),
                "{reward}"
            );
        }
    }
    // A finite reward is not refused.
    let mut log = log_under(&logging, &[0.2, 0.6, 0.8]);
    log[1].reward = -1e300;
    assert!(evaluate_off_policy(&log, &logging).is_ok());
    assert!(doubly_robust(&log, &logging, model).is_ok());
}

/// Forty adjudicated calibration-slice branches, keyed by branch, scoring
/// i/40 and harmful below 0.3.
fn adjudicated(prefix: &str) -> Vec<(BranchId, CalibrationSample)> {
    let logging = TriagePolicy::new(AutoThreshold::Never, 0.999).unwrap();
    (0..40)
        .map(|i| {
            let branch = BranchId(format!("{prefix}{i}"));
            let score_value = i as f32 / 40.0;
            let sample = logging
                .triage(&passing(), score(score_value), 0.0)
                .unwrap()
                .adjudicate(score_value < 0.3)
                .expect("calibration slice");
            (branch, sample)
        })
        .collect()
}

#[test]
fn a_recorded_policy_names_its_calibration_branches_and_holds_out_the_rest() {
    let calibration = adjudicated("c");
    let rule = ThresholdRule::ConformalRiskControl { alpha: 0.1 };
    let record = PolicyRecord::calibrate("policy-2", rule, 0.05, &calibration).unwrap();
    // The same threshold as calibrating the bare samples.
    let scored: Vec<CalibrationSample> = calibration.iter().map(|(_, s)| *s).collect();
    assert_eq!(
        record.policy().threshold(),
        calibrate_threshold(&scored, 0.1).unwrap()
    );
    assert_eq!(record.policy().calibration_rate(), 0.05);
    assert_eq!(record.rule(), rule);
    assert_eq!(record.version(), "policy-2");
    assert_eq!(record.calibrated_on().len(), 40);
    assert!(record
        .calibrated_on()
        .windows(2)
        .all(|pair| pair[0] < pair[1]));

    // Only adjudications the policy did not see are held out for evaluating it.
    let mut later = calibration.clone();
    later.extend(adjudicated("h"));
    let held_out = record.held_out(&later);
    assert_eq!(held_out.len(), 40);
    assert!(held_out.iter().all(|(branch, _)| branch.0.starts_with('h')));

    // A certified rule records its confidence too, and round-trips through parts.
    let certified = PolicyRecord::calibrate(
        "policy-3",
        ThresholdRule::LearnThenTest {
            alpha: 0.2,
            delta: 0.1,
        },
        0.05,
        &calibration,
    )
    .unwrap();
    let mut reversed = certified.calibrated_on().to_vec();
    reversed.reverse();
    assert_eq!(
        PolicyRecord::from_parts(
            "policy-3",
            certified.policy().threshold(),
            0.05,
            certified.rule(),
            reversed
        )
        .unwrap(),
        certified
    );
}

#[test]
fn a_recorded_policy_refuses_what_would_make_its_evaluation_dishonest() {
    let calibration = adjudicated("c");
    let crc = ThresholdRule::ConformalRiskControl { alpha: 0.1 };
    let mut twice = calibration.clone();
    twice.push(calibration[3].clone());
    assert_eq!(
        PolicyRecord::calibrate("v", crc, 0.05, &twice).unwrap_err(),
        ArbiterError::DuplicateCalibrationBranch {
            branch: "c3".into()
        }
    );
    assert_eq!(
        PolicyRecord::calibrate("", crc, 0.05, &calibration).unwrap_err(),
        ArbiterError::EmptyVersion
    );
    assert_eq!(
        PolicyRecord::calibrate("v", crc, 0.05, &[]).unwrap_err(),
        ArbiterError::EmptyCalibration
    );
    assert!(matches!(
        PolicyRecord::calibrate("v", ThresholdRule::Manual, 0.05, &calibration),
        Err(ArbiterError::InvalidRule { rule: "manual", .. })
    ));
    assert!(matches!(
        PolicyRecord::calibrate(
            "v",
            ThresholdRule::ConformalRiskControl { alpha: f64::NAN },
            0.05,
            &calibration
        ),
        Err(ArbiterError::InvalidRisk { field: "alpha", .. })
    ));
    assert!(matches!(
        PolicyRecord::calibrate("v", crc, 1.0, &calibration),
        Err(ArbiterError::InvalidExploration { .. })
    ));
    // A manual policy is calibrated on nothing, and a calibrated one on something.
    let manual = PolicyRecord::manual(
        "bootstrap",
        TriagePolicy::new(AutoThreshold::Never, 0.1).unwrap(),
    )
    .unwrap();
    assert_eq!(manual.rule(), ThresholdRule::Manual);
    assert!(manual.calibrated_on().is_empty());
    assert_eq!(manual.held_out(&calibration).len(), 40);
    let named = PolicyRecord::from_parts(
        "v",
        AutoThreshold::Never,
        0.1,
        ThresholdRule::Manual,
        vec![BranchId::from("c1")],
    )
    .unwrap_err();
    assert_eq!(
        named,
        ArbiterError::InvalidRule {
            rule: "manual",
            message: "a manual threshold must not name calibration branches",
        }
    );
    assert_eq!(
        named.to_string(),
        "rule manual: a manual threshold must not name calibration branches"
    );
    assert_eq!(
        PolicyRecord::from_parts("v", AutoThreshold::Never, 0.1, crc, vec![]).unwrap_err(),
        ArbiterError::EmptyCalibration
    );
}
