use ptr_branch::{
    calibrate_threshold, calibration_draw, certify_threshold, doubly_robust, evaluate_off_policy,
    ArbiterError, AutoThreshold, BranchId, CalibrationSample, LoggedTriage, OffPolicyEstimate,
    PolicyRecord, ThresholdRule, TriageDecision, TriageOutcome, TriageOutcomeParts, TriagePolicy,
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
    assert_eq!(boundary.decision(), TriageDecision::AutoPropose);
    assert!(boundary.eligible());
    assert!(!boundary.calibration_slice());
    assert_eq!(boundary.auto_propensity(), 0.75);
    let sampled = policy.triage(&passing(), score(0.5), 0.0).unwrap();
    assert_eq!(sampled.decision(), TriageDecision::Escalate);
    assert!(sampled.adjudicate(false).is_ok());
    let below = policy.triage(&passing(), score(0.49), 0.25).unwrap();
    assert_eq!(below.decision(), TriageDecision::Escalate);
    assert_eq!(below.auto_propensity(), 0.0);
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
    assert_eq!(outcome.decision(), TriageDecision::Escalate);
    assert!(!outcome.eligible());
    assert!(!outcome.calibration_slice());
    assert_eq!(outcome.auto_propensity(), 0.0);
    assert_eq!(
        outcome.adjudicate(false),
        Err(ArbiterError::NotCalibrationSlice)
    );
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
    assert_eq!(outcome.decision(), TriageDecision::Discard);
    assert!(!outcome.eligible());
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
        assert_eq!(outcome.decision(), TriageDecision::Escalate);
        assert!(!outcome.eligible());
    }
}

#[test]
fn the_calibration_slice_escalates_high_scores_and_only_it_can_be_adjudicated() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.2).unwrap();
    let sliced = policy.triage(&passing(), score(0.9), 0.1).unwrap();
    assert_eq!(sliced.decision(), TriageDecision::Escalate);
    assert!(sliced.calibration_slice());
    let sample = sliced.adjudicate(false).unwrap();
    assert_eq!((sample.score(), sample.harmful()), (0.9, false));

    let auto = policy.triage(&passing(), score(0.9), 0.7).unwrap();
    assert_eq!(auto.decision(), TriageDecision::AutoPropose);
    assert!((auto.auto_propensity() - 0.8).abs() < 1e-12);
    let refused = auto.adjudicate(false).unwrap_err();
    assert_eq!(refused, ArbiterError::NotCalibrationSlice);
    assert_eq!(refused.code(), "PTR_ARBITER_NOT_CALIBRATION_SLICE");

    let low = policy.triage(&passing(), score(0.2), 0.7).unwrap();
    assert_eq!(low.decision(), TriageDecision::Escalate);
    assert_eq!(low.adjudicate(true), Err(ArbiterError::NotCalibrationSlice));
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
        next.triage(&passing(), score(0.5), 0.5).unwrap().decision(),
        TriageDecision::AutoPropose
    );
    assert_eq!(
        next.triage(&passing(), score(0.1), 0.5).unwrap().decision(),
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
        next.triage(&passing(), score(0.9), 0.5).unwrap().decision(),
        TriageDecision::AutoPropose
    );
    assert_eq!(
        next.triage(&passing(), score(0.1), 0.5).unwrap().decision(),
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
            logged.reward = match outcome.decision() {
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

/// An eligible record scoring above a 0.5 threshold, logged with any
/// propensity and reward the log's checks accept.
fn eligible(decision: TriageDecision, auto_propensity: f64, reward: f64) -> LoggedTriage {
    LoggedTriage {
        eligible: true,
        score: 0.8,
        decision,
        auto_propensity,
        reward,
    }
}

#[test]
fn off_policy_estimates_of_extreme_but_valid_logs_are_finite() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap();
    // Unit weights and the largest finite reward: the sum of the weighted
    // rewards overflows, but none of the estimates does.
    let log = [eligible(TriageDecision::AutoPropose, 0.7, f64::MAX); 2];
    assert_eq!(
        evaluate_off_policy(&log, &policy),
        Ok(OffPolicyEstimate {
            ips: f64::MAX,
            snips: f64::MAX,
            effective_sample_size: 2.0,
        })
    );
    assert_eq!(doubly_robust(&log, &policy, |_, _| 0.0), Ok(f64::MAX));
    // A propensity of 1e-300 gives a weight near 1e300, whose square
    // overflows; the effective sample size, SNIPS and IPS do not.
    let log = [
        eligible(TriageDecision::AutoPropose, 1e-300, 0.0),
        eligible(TriageDecision::Escalate, 1e-300, 1.0),
    ];
    let estimate = evaluate_off_policy(&log, &policy).unwrap();
    assert!(
        (estimate.effective_sample_size - 1.0).abs() < 1e-12,
        "{estimate:?}"
    );
    assert!((estimate.ips - 0.15).abs() < 1e-12, "{estimate:?}");
    let snips = 0.3 / (0.7 / 1e-300 + 0.3);
    assert!(
        (estimate.snips - snips).abs() <= 1e-12 * snips,
        "{estimate:?}"
    );
    let dr = doubly_robust(&log, &policy, |_, _| 0.0).unwrap();
    assert!((dr - 0.15).abs() < 1e-12, "{dr}");
}

#[test]
fn an_infinite_importance_weight_or_a_nonfinite_estimate_is_refused() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap();
    let model = |_: &LoggedTriage, _: TriageDecision| 0.0;
    // A positive propensity so small that 0.7 / propensity is infinite.
    let log = [
        eligible(TriageDecision::Escalate, 0.5, 1.0),
        eligible(TriageDecision::AutoPropose, 5e-324, 0.0),
    ];
    let refused = evaluate_off_policy(&log, &policy).unwrap_err();
    assert_eq!(
        refused,
        ArbiterError::InvalidPropensity {
            index: 1,
            value: 5e-324,
        }
    );
    assert_eq!(refused.code(), "PTR_ARBITER_INVALID_PROPENSITY");
    assert_eq!(doubly_robust(&log, &policy, model), Err(refused));
    // Finite weights and rewards whose IPS and doubly robust estimates are
    // beyond f64: 0.7e300 * 1e300 / 2.
    let log = [
        eligible(TriageDecision::AutoPropose, 1e-300, 1e300),
        eligible(TriageDecision::Escalate, 1e-300, 1.0),
    ];
    let refused = evaluate_off_policy(&log, &policy).unwrap_err();
    assert_eq!(refused, ArbiterError::NonFiniteEstimate { estimate: "ips" });
    assert_eq!(refused.code(), "PTR_ARBITER_NONFINITE_ESTIMATE");
    assert_eq!(
        doubly_robust(&log, &policy, model),
        Err(ArbiterError::NonFiniteEstimate {
            estimate: "doubly_robust"
        })
    );
    // A reward model that predicts NaN makes the doubly robust estimate NaN.
    let log = log_under(&policy, &[0.2, 0.6, 0.8]);
    assert_eq!(
        doubly_robust(&log, &policy, |_, _| f64::NAN),
        Err(ArbiterError::NonFiniteEstimate {
            estimate: "doubly_robust"
        })
    );
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
    // Learn-then-Test at these levels, not conformal risk control at the same
    // alpha and not Learn-then-Test with the levels swapped.
    let ltt = certify_threshold(&scored, 0.2, 0.1).unwrap();
    assert_eq!(certified.policy().threshold(), ltt);
    assert_ne!(ltt, calibrate_threshold(&scored, 0.2).unwrap());
    assert_ne!(ltt, certify_threshold(&scored, 0.1, 0.2).unwrap());
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
fn a_record_rebuilt_from_parts_refuses_a_risk_level_or_confidence_outside_the_unit_interval() {
    // Nothing else is wrong with these records, so only the rule's own level
    // check can refuse them: from_parts is how a stored policy is read back,
    // and no threshold computation runs there to catch a corrupt level.
    let rebuild = |rule| {
        PolicyRecord::from_parts(
            "v",
            AutoThreshold::AtLeast(0.5),
            0.1,
            rule,
            vec![BranchId::from("c1")],
        )
    };
    for (rule, field) in [
        (
            ThresholdRule::ConformalRiskControl { alpha: f64::NAN },
            "alpha",
        ),
        (ThresholdRule::ConformalRiskControl { alpha: 0.0 }, "alpha"),
        (
            ThresholdRule::LearnThenTest {
                alpha: 1.0,
                delta: 0.1,
            },
            "alpha",
        ),
        (
            ThresholdRule::LearnThenTest {
                alpha: 0.1,
                delta: 1.5,
            },
            "delta",
        ),
        (
            ThresholdRule::LearnThenTest {
                alpha: 0.1,
                delta: f64::NAN,
            },
            "delta",
        ),
    ] {
        let refused = rebuild(rule).unwrap_err();
        assert!(
            matches!(
                refused,
                ArbiterError::InvalidRisk { field: refused_field, .. } if refused_field == field
            ),
            "{rule:?}: {refused:?}"
        );
    }
    assert!(rebuild(ThresholdRule::LearnThenTest {
        alpha: 0.1,
        delta: 0.1
    })
    .is_ok());
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

#[test]
fn a_threshold_outside_the_unit_interval_is_refused_wherever_a_policy_is_built() {
    // NaN admits no score and a negative threshold every score; above one is
    // as unreachable as NaN. Each would silently make the policy "never" or
    // "always" auto-propose.
    for value in [f32::NAN, -0.5, f32::NEG_INFINITY, 1.5, f32::INFINITY] {
        let refused = TriagePolicy::new(AutoThreshold::AtLeast(value), 0.0).unwrap_err();
        assert!(
            matches!(refused, ArbiterError::InvalidThreshold { value: got }
                if got.to_bits() == value.to_bits()),
            "{value}: {refused:?}"
        );
        assert_eq!(refused.code(), "PTR_ARBITER_INVALID_THRESHOLD");
        // A stored record is rebuilt through from_parts, which refuses it too.
        assert!(
            matches!(
                PolicyRecord::from_parts(
                    "v1",
                    AutoThreshold::AtLeast(value),
                    0.0,
                    ThresholdRule::Manual,
                    vec![],
                ),
                Err(ArbiterError::InvalidThreshold { .. })
            ),
            "{value}"
        );
    }
    // The endpoints are scores: zero admits every eligible branch, one only a
    // perfect score.
    let everything = TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.0).unwrap();
    let perfect = TriagePolicy::new(AutoThreshold::AtLeast(1.0), 0.0).unwrap();
    assert!(PolicyRecord::manual("v1", perfect).is_ok());
    let decide = |policy: &TriagePolicy, value| {
        policy
            .triage(&passing(), score(value), 0.5)
            .unwrap()
            .decision()
    };
    assert_eq!(decide(&everything, 0.0), TriageDecision::AutoPropose);
    assert_eq!(decide(&perfect, 1.0), TriageDecision::AutoPropose);
    assert_eq!(decide(&perfect, 0.999), TriageDecision::Escalate);
}

#[test]
fn learn_then_test_starts_where_its_own_bound_can_first_pass() {
    // At these levels the closed-form start ceil(ln delta / ln(1 - alpha))
    // is one sample short of what the Clopper-Pearson test itself passes
    // with no harm (4 rather than 3, and 3 rather than 2), so starting there
    // failed the first test and certified nothing.
    let logging = TriagePolicy::new(AutoThreshold::Never, 0.999).unwrap();
    let samples: Vec<CalibrationSample> = (0..400)
        .map(|i| {
            logging
                .triage(&passing(), score(i as f32 / 400.0), 0.0)
                .unwrap()
                .adjudicate(false)
                .expect("calibration slice")
        })
        .collect();
    for (alpha, delta) in [(0.5, 0.125), (0.3, 0.49), (0.5, 0.0625)] {
        assert_eq!(
            certify_threshold(&samples, alpha, delta).unwrap(),
            AutoThreshold::AtLeast(0.0),
            "alpha {alpha} delta {delta}"
        );
    }
}

#[test]
fn doubly_robust_scales_before_it_multiplies_so_a_finite_estimate_is_returned() {
    // One auto-proposed record with a weight of 1e300 and a reward of 1e10
    // among 999 ineligible ones: weight * reward overflows, the estimate
    // (1e307, the IPS with a zero reward model) does not.
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.0).unwrap();
    let mut log = vec![LoggedTriage {
        eligible: true,
        score: 0.9,
        decision: TriageDecision::AutoPropose,
        auto_propensity: 1e-300,
        reward: 1e10,
    }];
    log.extend(
        [LoggedTriage {
            eligible: false,
            score: 0.1,
            decision: TriageDecision::Escalate,
            auto_propensity: 0.0,
            reward: 0.0,
        }; 999],
    );
    let ips = evaluate_off_policy(&log, &policy).unwrap().ips;
    let dr = doubly_robust(&log, &policy, |_, _| 0.0).unwrap();
    assert!((dr - ips).abs() <= 1e-12 * ips, "{dr} {ips}");
    assert!((dr - 1e307).abs() <= 1e-12 * 1e307, "{dr}");

    // A single record whose residual, the largest reward minus the smallest
    // prediction, overflows although the weighted estimate does not.
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap();
    let log = [eligible(TriageDecision::Escalate, 0.3, f64::MAX)];
    let model = |_: &LoggedTriage, action| match action {
        TriageDecision::Escalate => -f64::MAX,
        TriageDecision::AutoPropose | TriageDecision::Discard => 0.0,
    };
    // -0.3 * MAX + (0.3 / 0.7) * 2 * MAX, about 0.557 * MAX.
    let expected = 2.0 * (-0.15 * f64::MAX + (0.3 / 0.7) * f64::MAX);
    let dr = doubly_robust(&log, &policy, model).unwrap();
    assert!((dr - expected).abs() <= 1e-12 * expected, "{dr} {expected}");
}

#[test]
fn a_logged_score_that_is_not_a_probability_is_refused_before_any_reweighting() {
    // An eligible escalated record with a NaN score and propensity zero: NaN
    // admits nothing under an `AtLeast` target, so the target escalates it
    // with probability one and the record used to be reweighted by one as if
    // its score meant something. A score above one or an infinite one used
    // to surface as a positivity violation instead of as the bad score.
    let target = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap();
    let model = |_: &LoggedTriage, _: TriageDecision| 0.5;
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.25, 1.5] {
        for eligible_record in [true, false] {
            let log = [
                eligible(TriageDecision::AutoPropose, 0.7, 1.0),
                LoggedTriage {
                    eligible: eligible_record,
                    score: value,
                    decision: TriageDecision::Escalate,
                    auto_propensity: 0.0,
                    reward: 0.2,
                },
            ];
            let refused = evaluate_off_policy(&log, &target).unwrap_err();
            assert!(
                matches!(refused, ArbiterError::InvalidScore { index: 1, value: seen }
                    if seen.to_bits() == value.to_bits()),
                "{value} eligible={eligible_record}: {refused:?}"
            );
            assert_eq!(refused.code(), "PTR_ARBITER_INVALID_SCORE");
            assert!(
                matches!(
                    doubly_robust(&log, &target, model),
                    Err(ArbiterError::InvalidScore { index: 1, .. })
                ),
                "{value} eligible={eligible_record}"
            );
        }
    }
    // The ends of the unit interval are scores.
    for value in [0.0, 1.0] {
        let mut record = eligible(TriageDecision::Escalate, 0.0, 0.2);
        record.score = value;
        if value >= 0.5 {
            record.decision = TriageDecision::AutoPropose;
            record.auto_propensity = 0.7;
        }
        assert!(evaluate_off_policy(&[record], &target).is_ok(), "{value}");
        assert!(doubly_robust(&[record], &target, model).is_ok(), "{value}");
    }
}

#[test]
fn a_logged_triage_no_policy_can_produce_is_refused_before_any_reweighting() {
    // `LoggedTriage` has public fields, so a log can hold records no
    // `TriagePolicy::triage` returns. An ineligible auto-proposal was
    // reweighted by one under every target, as if verification alone had
    // proposed it; a propensity on an ineligible record was ignored; and the
    // eligible contradictions surfaced, if at all, as a zero-probability
    // propensity or a positivity violation instead of as what they are.
    let ineligible = |decision, auto_propensity| LoggedTriage {
        eligible: false,
        score: 0.8,
        decision,
        auto_propensity,
        reward: 1.0,
    };
    let impossible = [
        (
            ineligible(TriageDecision::AutoPropose, 0.0),
            "verification alone never auto-proposes",
        ),
        (
            ineligible(TriageDecision::AutoPropose, 0.7),
            "verification alone never auto-proposes",
        ),
        (
            ineligible(TriageDecision::Escalate, 0.5),
            "a branch verification decided has auto-propose propensity zero",
        ),
        (
            ineligible(TriageDecision::Discard, 1.0),
            "a branch verification decided has auto-propose propensity zero",
        ),
        (
            eligible(TriageDecision::Discard, 0.7, 0.0),
            "an eligible branch is never discarded",
        ),
        (
            eligible(TriageDecision::Discard, 0.0, 0.0),
            "an eligible branch is never discarded",
        ),
        (
            eligible(TriageDecision::AutoPropose, 0.0, 1.0),
            "an eligible branch is auto-proposed only with positive auto-propose propensity",
        ),
        (
            eligible(TriageDecision::Escalate, 1.0, 0.2),
            "an eligible branch with auto-propose propensity one is never escalated",
        ),
    ];
    let targets = [
        TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap(),
        TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.0).unwrap(),
        TriagePolicy::new(AutoThreshold::Never, 0.0).unwrap(),
    ];
    let model = |_: &LoggedTriage, _: TriageDecision| 0.5;
    for (record, reason) in impossible {
        let log = [eligible(TriageDecision::AutoPropose, 0.7, 1.0), record];
        for target in &targets {
            let refused = evaluate_off_policy(&log, target).unwrap_err();
            assert_eq!(
                refused,
                ArbiterError::ImpossibleTriage { index: 1, reason },
                "{record:?} under {target:?}"
            );
            assert_eq!(refused.code(), "PTR_ARBITER_IMPOSSIBLE_TRIAGE");
            assert_eq!(
                refused.to_string(),
                format!("record 1 is not a triage any policy produces: {reason}")
            );
            assert_eq!(
                doubly_robust(&log, target, |_, _| panic!(
                    "the reward model is consulted only for a log that passed every check"
                )),
                Err(refused),
                "{record:?} under {target:?}"
            );
        }
        // The same record is refused wherever it sits in the log.
        assert_eq!(
            doubly_robust(&[record], &targets[0], model),
            Err(ArbiterError::ImpossibleTriage { index: 0, reason })
        );
    }
    // A propensity outside [0, 1] is refused as such before the rules that
    // compare it with zero and one.
    assert!(matches!(
        evaluate_off_policy(&[ineligible(TriageDecision::AutoPropose, 1.5)], &targets[0]),
        Err(ArbiterError::InvalidPropensity { index: 0, .. })
    ));
    // So is a score that is not a probability.
    let mut unscored = ineligible(TriageDecision::AutoPropose, 0.0);
    unscored.score = f32::NAN;
    assert!(matches!(
        evaluate_off_policy(&[unscored], &targets[0]),
        Err(ArbiterError::InvalidScore { index: 0, .. })
    ));
}

#[test]
fn every_triage_a_policy_produces_is_accepted_by_both_off_policy_estimates() {
    // The control for the refusals above: the boundary combinations they sit
    // next to (an ineligible escalation or discard with propensity zero, an
    // eligible escalation with propensity zero or below one, an eligible
    // auto-proposal with propensity one) are all triages some policy makes.
    let mut hard = passing();
    hard.findings.push(Finding {
        code: "unsafe".into(),
        message: "constraint failed".into(),
        hard: true,
    });
    let reports = [
        passing(),
        report(VerificationStatus::Fail, VerificationLevel::Deterministic),
        report(
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
        ),
        report(VerificationStatus::Pass, VerificationLevel::SampleVerified),
        hard,
    ];
    let policies = [
        TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.25).unwrap(),
        TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.0).unwrap(),
        TriagePolicy::new(AutoThreshold::AtLeast(0.0), 0.0).unwrap(),
        TriagePolicy::new(AutoThreshold::Never, 0.5).unwrap(),
        TriagePolicy::new(AutoThreshold::AtLeast(0.5), 2f64.powi(-53)).unwrap(),
    ];
    let mut seen = std::collections::BTreeSet::new();
    for policy in &policies {
        for verified in &reports {
            for value in [0.0, 0.2, 0.5, 0.9, 1.0] {
                for draw in [0.0, 0.1, 0.9] {
                    let triage = policy.triage(verified, score(value), draw).unwrap();
                    let mut record = LoggedTriage::from(&triage);
                    record.reward = 0.5;
                    seen.insert((
                        record.eligible,
                        record.decision,
                        record.auto_propensity == 0.0,
                        record.auto_propensity == 1.0,
                    ));
                    assert!(
                        evaluate_off_policy(&[record], policy).is_ok(),
                        "{record:?} under {policy:?}"
                    );
                    assert!(
                        doubly_robust(&[record], policy, |_, _| 0.5).is_ok(),
                        "{record:?} under {policy:?}"
                    );
                }
            }
        }
    }
    // Every shape a triage can take was exercised.
    for shape in [
        (false, TriageDecision::Discard, true, false),
        (false, TriageDecision::Escalate, true, false),
        (true, TriageDecision::Escalate, true, false),
        (true, TriageDecision::Escalate, false, false),
        (true, TriageDecision::AutoPropose, false, false),
        (true, TriageDecision::AutoPropose, false, true),
    ] {
        assert!(seen.contains(&shape), "{shape:?} not exercised: {seen:?}");
    }
    assert_eq!(seen.len(), 6, "{seen:?}");
}

#[test]
fn a_policy_explains_every_triage_it_produces_and_nothing_else() {
    let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.25).unwrap();
    let mut hard = passing();
    hard.findings.push(Finding {
        code: "unsafe".into(),
        message: "constraint failed".into(),
        hard: true,
    });
    let reports = [
        passing(),
        report(VerificationStatus::Pass, VerificationLevel::FullSemantic),
        report(VerificationStatus::Fail, VerificationLevel::Deterministic),
        report(
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
        ),
        report(VerificationStatus::Pass, VerificationLevel::SampleVerified),
        hard,
    ];
    for verified in &reports {
        for value in [0.0, 0.25, 0.49, 0.5, 0.9, 1.0] {
            for draw in [0.0, 0.1, 0.25, 0.9] {
                let triage = policy.triage(verified, score(value), draw).unwrap();
                assert_eq!(policy.explains(&triage), Ok(()), "{triage:?}");
            }
        }
    }

    let unexplained = |triage: &TriageOutcome| match policy.explains(triage) {
        Err(ArbiterError::UnexplainedTriage { reason }) => reason,
        other => panic!("{triage:?} was explained: {other:?}"),
    };
    // Another policy's triages, cited as this one's: its propensity, or its
    // decision for this score, is not one this policy logs.
    let other = TriagePolicy::new(AutoThreshold::AtLeast(0.8), 0.1).unwrap();
    let theirs = other.triage(&passing(), score(0.85), 0.5).unwrap();
    assert_eq!(theirs.decision(), TriageDecision::AutoPropose);
    assert_eq!(
        unexplained(&theirs),
        "the auto-propose propensity is not the one the policy logs for this score"
    );
    let below_theirs = other.triage(&passing(), score(0.6), 0.5).unwrap();
    assert_eq!(below_theirs.decision(), TriageDecision::Escalate);
    assert_eq!(
        unexplained(&below_theirs),
        "the auto-propose propensity is not the one the policy logs for this score"
    );
    // A slice triage cited as the triage of a policy that has no slice.
    let never_sliced = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.0).unwrap();
    let slice = policy.triage(&passing(), score(0.2), 0.1).unwrap();
    assert!(slice.calibration_slice());
    assert_eq!(
        never_sliced.explains(&slice),
        Err(ArbiterError::UnexplainedTriage {
            reason: "a policy with calibration rate zero has no calibration slice",
        })
    );
    // Every other rule explains checks is one that no policy breaks, so a
    // triage breaking it can no longer be built: from_parts refuses its
    // parts, naming the rule, where a triage with public fields used to be
    // built and reach explains. (Explains still checks those rules; the
    // arbiter unit tests show it on a triage built around the constructor.)
    let edited = |triage: TriageOutcome, edit: fn(&mut TriageOutcomeParts)| {
        let mut parts = triage.into_parts();
        edit(&mut parts);
        match TriageOutcome::from_parts(parts) {
            Err(ArbiterError::ImpossibleOutcome { reason }) => reason,
            other => panic!("{parts:?} was built: {other:?}"),
        }
    };
    let auto = policy.triage(&passing(), score(0.9), 0.5).unwrap();
    let below = policy.triage(&passing(), score(0.2), 0.5).unwrap();
    let sliced = policy.triage(&passing(), score(0.9), 0.1).unwrap();
    let failed = policy
        .triage(
            &report(VerificationStatus::Fail, VerificationLevel::Deterministic),
            score(0.9),
            0.5,
        )
        .unwrap();
    let outside = "outside the calibration slice an eligible branch is auto-proposed exactly \
                   when its auto-propose propensity is positive";
    for (triage, edit, reason) in [
        // The right propensity with a decision the threshold does not make.
        (
            auto.clone(),
            (|parts| parts.decision = TriageDecision::Escalate) as fn(&mut TriageOutcomeParts),
            outside,
        ),
        (
            below.clone(),
            |parts| parts.decision = TriageDecision::AutoPropose,
            outside,
        ),
        // Eligible rows the policy never writes.
        (
            below,
            |parts| parts.decision = TriageDecision::Discard,
            "an eligible branch is never discarded",
        ),
        (
            sliced,
            |parts| parts.decision = TriageDecision::AutoPropose,
            "a calibration-slice branch is escalated",
        ),
        // Rows verification decided, but not the way verification decides.
        (
            failed.clone(),
            |parts| parts.decision = TriageDecision::AutoPropose,
            "verification alone never auto-proposes",
        ),
        (
            failed.clone(),
            |parts| parts.calibration_slice = true,
            "a calibration-slice branch is eligible",
        ),
        (
            failed,
            |parts| parts.auto_propensity = 0.75,
            "a branch verification decided has auto-propose propensity zero",
        ),
    ] {
        assert_eq!(edited(triage, edit), reason);
    }
    // A score no triage sees.
    for value in [f32::NAN, -0.1, 1.5, f32::INFINITY] {
        let mut parts = auto.clone().into_parts();
        parts.score = value;
        let refused = TriageOutcome::from_parts(parts).unwrap_err();
        assert_eq!(refused.code(), "PTR_ARBITER_IMPOSSIBLE_OUTCOME");
        assert_eq!(
            refused,
            ArbiterError::ImpossibleOutcome {
                reason: "the score is not a probability"
            }
        );
    }
}

#[test]
fn a_calibration_rate_too_small_to_lower_the_propensity_is_refused_wherever_a_policy_is_built() {
    // For a positive rate of at most 2^-54, `1 - rate` rounds to one: a
    // calibration-slice triage was logged with auto-propose propensity one,
    // so the escalation it records had probability zero under the policy
    // that made it. `explains` accepted that triage, the table's CHECK
    // refused it as a raw database error, and off-policy evaluation refused
    // it as an invalid propensity.
    let largest_degenerate = 2f64.powi(-54);
    assert_eq!(1.0 - largest_degenerate, 1.0);
    for rate in [1e-17, largest_degenerate, 1e-300, f64::MIN_POSITIVE, 5e-324] {
        let refused = TriagePolicy::new(AutoThreshold::AtLeast(0.5), rate).unwrap_err();
        assert!(
            matches!(refused, ArbiterError::InvalidExploration { rate: got }
                if got.to_bits() == rate.to_bits()),
            "{rate}: {refused:?}"
        );
        assert_eq!(refused.code(), "PTR_ARBITER_INVALID_EXPLORATION");
        assert_eq!(
            refused.to_string(),
            format!("calibration rate {rate} is so small that 1 - rate rounds to one")
        );
        // A stored record is rebuilt through from_parts, which refuses it too.
        assert!(
            matches!(
                PolicyRecord::from_parts(
                    "v1",
                    AutoThreshold::AtLeast(0.5),
                    rate,
                    ThresholdRule::Manual,
                    vec![],
                ),
                Err(ArbiterError::InvalidExploration { .. })
            ),
            "{rate}"
        );
    }
    // Zero disables the slice, and the smallest rate that lowers the
    // propensity is kept: its slice triages have propensity below one, the
    // policy explains them, and its own log is reweighted by one.
    let off = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.0).unwrap();
    assert!(!off
        .triage(&passing(), score(0.9), 0.0)
        .unwrap()
        .calibration_slice());
    let smallest = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 2f64.powi(-53)).unwrap();
    let slice = smallest.triage(&passing(), score(0.9), 0.0).unwrap();
    assert!(slice.calibration_slice());
    assert_eq!(slice.decision(), TriageDecision::Escalate);
    assert!(slice.auto_propensity() < 1.0, "{}", slice.auto_propensity());
    assert_eq!(smallest.explains(&slice), Ok(()));
    let estimate = evaluate_off_policy(&[LoggedTriage::from(&slice)], &smallest).unwrap();
    assert_eq!(estimate.effective_sample_size, 1.0);
}

#[test]
fn a_log_on_which_the_target_takes_no_logged_action_is_refused_rather_than_scored_zero() {
    // Every logged branch was auto-proposed at propensity 0.5 and cost 1.
    // A policy that never auto-proposes passes positivity (the log also
    // escalated with probability 0.5) but gives every logged action
    // probability zero: SNIPS is 0 / 0, and the log says nothing about what
    // the policy earns, so no estimate is returned that could rank it above
    // the policy that was logged.
    let log = [eligible(TriageDecision::AutoPropose, 0.5, -1.0); 50];
    let never = TriagePolicy::new(AutoThreshold::Never, 0.0).unwrap();
    let refused = evaluate_off_policy(&log, &never).unwrap_err();
    assert_eq!(
        refused,
        ArbiterError::NonFiniteEstimate { estimate: "snips" }
    );
    assert_eq!(refused.code(), "PTR_ARBITER_NONFINITE_ESTIMATE");
    // The logged policy itself is estimated from the same log.
    let logged = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.5).unwrap();
    let estimate = evaluate_off_policy(&log, &logged).unwrap();
    assert!((estimate.snips + 1.0).abs() < 1e-12, "{estimate:?}");
    assert!(
        (estimate.effective_sample_size - 50.0).abs() < 1e-9,
        "{estimate:?}"
    );
    // The doubly robust estimate still has the reward model to go on.
    let dr = doubly_robust(&log, &never, |_, _| -0.5).unwrap();
    assert!((dr + 0.5).abs() < 1e-12, "{dr}");
}

/// Whether some policy, verification report and draw make `triage` return
/// exactly `parts`: the witnesses `TriageOutcome::from_parts` documents (for
/// a positive propensity `p`, rate `1 - p` and the score as threshold), and
/// a few more.
fn some_policy_produces(parts: TriageOutcomeParts) -> bool {
    let Some(value) = Probability::new(parts.score) else {
        return false;
    };
    let mut rates = vec![0.0, 0.5];
    if parts.auto_propensity > 0.0 && parts.auto_propensity <= 1.0 {
        rates.push(1.0 - parts.auto_propensity);
    }
    let mut thresholds = vec![
        AutoThreshold::Never,
        AutoThreshold::AtLeast(0.0),
        AutoThreshold::AtLeast(1.0),
    ];
    if (0.0..=1.0).contains(&parts.score) {
        thresholds.push(AutoThreshold::AtLeast(parts.score));
    }
    let reports = [
        passing(),
        report(VerificationStatus::Fail, VerificationLevel::Deterministic),
        report(
            VerificationStatus::Disputed,
            VerificationLevel::Deterministic,
        ),
    ];
    rates.iter().any(|&rate| {
        thresholds.iter().any(|&threshold| {
            let Ok(policy) = TriagePolicy::new(threshold, rate) else {
                return false;
            };
            reports.iter().any(|verified| {
                [0.0, 0.5, rate].iter().any(|&draw| {
                    policy
                        .triage(verified, value, draw)
                        .is_ok_and(|triage| triage.into_parts() == parts)
                })
            })
        })
    })
}

#[test]
fn a_triage_is_rebuilt_from_parts_exactly_when_some_policy_produces_it() {
    // The fields of a triage were public, so a caller could build or edit
    // one no policy produces (a NaN score, an auto-proposed slice branch)
    // and adjudicate it into a calibration sample. Now parts are rebuilt
    // only if some policy returns exactly them, and every such parts are.
    let scores = [0.0, 0.2, 0.5, 0.9, 1.0, f32::NAN, -0.1, 1.5, f32::INFINITY];
    let propensities = [
        0.0,
        2f64.powi(-53),
        0.25,
        0.5,
        0.75,
        0.9,
        1.0 - 2f64.powi(-53),
        1.0,
        // Positive, but 1 - r for no calibration rate r.
        0.1,
        1e-300,
        f64::NAN,
        -0.5,
        1.5,
        f64::INFINITY,
    ];
    let (mut built, mut refused) = (0, 0);
    for decision in [
        TriageDecision::AutoPropose,
        TriageDecision::Escalate,
        TriageDecision::Discard,
    ] {
        for eligible in [false, true] {
            for calibration_slice in [false, true] {
                for score in scores {
                    for auto_propensity in propensities {
                        let parts = TriageOutcomeParts {
                            decision,
                            eligible,
                            calibration_slice,
                            score,
                            auto_propensity,
                        };
                        let produced = some_policy_produces(parts);
                        match TriageOutcome::from_parts(parts) {
                            Ok(triage) => {
                                assert!(produced, "{parts:?} was built");
                                assert_eq!(triage.into_parts(), parts);
                                built += 1;
                            }
                            Err(error) => {
                                assert!(!produced, "{parts:?} was refused: {error:?}");
                                assert!(
                                    matches!(error, ArbiterError::ImpossibleOutcome { .. }),
                                    "{error:?}"
                                );
                                assert_eq!(error.code(), "PTR_ARBITER_IMPOSSIBLE_OUTCOME");
                                refused += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(built > 0 && refused > 0, "{built} built, {refused} refused");
}

#[test]
fn triage_parts_no_policy_produces_are_refused_naming_the_rule_they_break() {
    let parts =
        |decision, eligible, calibration_slice, score, auto_propensity| TriageOutcomeParts {
            decision,
            eligible,
            calibration_slice,
            score,
            auto_propensity,
        };
    use TriageDecision::{AutoPropose, Discard, Escalate};
    for (parts, reason) in [
        (
            parts(Escalate, true, true, f32::NAN, 0.0),
            "the score is not a probability",
        ),
        (
            parts(Escalate, false, false, 0.5, f64::NAN),
            "the auto-propose propensity is not a probability",
        ),
        (
            parts(AutoPropose, true, false, 0.5, 1.5),
            "the auto-propose propensity is not a probability",
        ),
        (
            parts(AutoPropose, true, false, 0.5, 0.1),
            "a positive auto-propose propensity is 1 - r for a calibration rate r",
        ),
        (
            parts(AutoPropose, false, false, 0.5, 0.0),
            "verification alone never auto-proposes",
        ),
        (
            parts(Escalate, false, true, 0.5, 0.0),
            "a calibration-slice branch is eligible",
        ),
        (
            parts(Discard, false, false, 0.5, 0.5),
            "a branch verification decided has auto-propose propensity zero",
        ),
        (
            parts(Discard, true, false, 0.5, 0.0),
            "an eligible branch is never discarded",
        ),
        (
            parts(AutoPropose, true, true, 0.9, 0.8),
            "a calibration-slice branch is escalated",
        ),
        (
            parts(Escalate, true, true, 0.9, 1.0),
            "a calibration-slice branch has auto-propose propensity below one",
        ),
        (
            parts(AutoPropose, true, false, 0.9, 0.0),
            "outside the calibration slice an eligible branch is auto-proposed exactly when \
             its auto-propose propensity is positive",
        ),
        (
            parts(Escalate, true, false, 0.9, 0.75),
            "outside the calibration slice an eligible branch is auto-proposed exactly when \
             its auto-propose propensity is positive",
        ),
    ] {
        let refused = TriageOutcome::from_parts(parts).unwrap_err();
        assert_eq!(
            refused,
            ArbiterError::ImpossibleOutcome { reason },
            "{parts:?}"
        );
        assert_eq!(
            refused.to_string(),
            format!("no triage policy produces this triage: {reason}")
        );
    }
    // The shapes next to them are triages, and a slice triage among them
    // is adjudicated.
    for parts in [
        parts(Escalate, true, true, 0.9, 0.8),
        parts(Escalate, true, true, 0.2, 0.0),
        parts(AutoPropose, true, false, 0.9, 1.0),
        parts(AutoPropose, true, false, 0.9, 2f64.powi(-53)),
        parts(Escalate, true, false, 0.2, 0.0),
        parts(Discard, false, false, 0.9, 0.0),
        parts(Escalate, false, false, 0.9, 0.0),
    ] {
        let triage = TriageOutcome::from_parts(parts).unwrap();
        assert_eq!(
            triage.adjudicate(true).is_ok(),
            parts.calibration_slice,
            "{parts:?}"
        );
    }
}

#[test]
fn no_calibration_sample_hides_harm_behind_a_score_no_triage_sees() {
    // Twenty clean calibration branches at 0.9 and twenty harmful ones. With
    // the harmful ones' score NaN, no grid threshold admitted them, and
    // Learn-then-Test certified a threshold of zero for a slice that was
    // half harmful. Such a triage can no longer be built, so the samples
    // carry their scores and the harm is seen.
    let slice = |score| TriageOutcomeParts {
        decision: TriageDecision::Escalate,
        eligible: true,
        calibration_slice: true,
        score,
        auto_propensity: 0.0,
    };
    assert_eq!(
        TriageOutcome::from_parts(slice(f32::NAN)),
        Err(ArbiterError::ImpossibleOutcome {
            reason: "the score is not a probability"
        })
    );
    let adjudicate = |score, harmful| {
        TriageOutcome::from_parts(slice(score))
            .unwrap()
            .adjudicate(harmful)
            .unwrap()
    };
    let mut samples: Vec<CalibrationSample> = (0..20).map(|_| adjudicate(0.9, false)).collect();
    samples.extend((0..20).map(|_| adjudicate(0.9, true)));
    assert_eq!(
        certify_threshold(&samples, 0.2, 0.1).unwrap(),
        AutoThreshold::Never
    );
    // Nor is an auto-proposed triage, whose outcome was never adjudicated as
    // part of the slice, turned into a sample however it is flagged.
    let mut auto = slice(0.9);
    auto.decision = TriageDecision::AutoPropose;
    auto.auto_propensity = 0.8;
    assert_eq!(
        TriageOutcome::from_parts(auto),
        Err(ArbiterError::ImpossibleOutcome {
            reason: "a calibration-slice branch is escalated"
        })
    );
    auto.calibration_slice = false;
    assert_eq!(
        TriageOutcome::from_parts(auto).unwrap().adjudicate(true),
        Err(ArbiterError::NotCalibrationSlice)
    );
}

#[test]
fn doubly_robust_consults_the_reward_model_once_per_record_and_action_and_reuses_the_logged_prediction(
) {
    // A stateful model answers each call differently, as a sampled or re-run
    // inference can. The logged action used to be predicted twice, once for
    // the direct term and once for its residual, so the two did not cancel
    // and the estimate depended on the order of the calls.
    let logging = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.3).unwrap();
    let target = TriagePolicy::new(AutoThreshold::AtLeast(0.7), 0.3).unwrap();
    let log = log_under(&logging, &[0.2, 0.6, 0.8, 0.9]);
    let mut calls: Vec<(f32, TriageDecision, f64)> = Vec::new();
    let model = |record: &LoggedTriage, action| {
        let prediction = 0.125 * calls.len() as f64;
        calls.push((record.score, action, prediction));
        prediction
    };
    let dr = doubly_robust(&log, &target, model).unwrap();

    // Once per record and action, in log order and in the documented order
    // of the actions.
    let actions = [
        TriageDecision::AutoPropose,
        TriageDecision::Escalate,
        TriageDecision::Discard,
    ];
    let expected: Vec<(f32, TriageDecision)> = log
        .iter()
        .flat_map(|record| actions.map(|action| (record.score, action)))
        .collect();
    let seen: Vec<(f32, TriageDecision)> = calls
        .iter()
        .map(|&(score, action, _)| (score, action))
        .collect();
    assert_eq!(seen, expected);

    // The estimate is the doubly robust formula on exactly those
    // predictions: every record is eligible, the target auto-proposes from
    // 0.7 with probability 0.7, and the logging policy's probability of the
    // logged action is its propensity or one minus it.
    let n = log.len() as f64;
    let formula: f64 = log
        .iter()
        .zip(calls.chunks(3))
        .map(|(record, predictions)| {
            let [auto, escalate, discard] = [predictions[0].2, predictions[1].2, predictions[2].2];
            let target_auto = if record.score >= 0.7 { 0.7 } else { 0.0 };
            let direct = target_auto * auto + (1.0 - target_auto) * escalate + 0.0 * discard;
            let (target_logged, logged, prediction) = match record.decision {
                TriageDecision::AutoPropose => (target_auto, record.auto_propensity, auto),
                TriageDecision::Escalate => {
                    (1.0 - target_auto, 1.0 - record.auto_propensity, escalate)
                }
                TriageDecision::Discard => unreachable!("an eligible record is never discarded"),
            };
            direct + target_logged / logged * (record.reward - prediction)
        })
        .sum::<f64>()
        / n;
    assert!((dr - formula).abs() < 1e-12, "dr {dr} formula {formula}");
}

#[test]
fn a_record_rebuilt_from_parts_names_its_calibration_branches_without_proving_them() {
    // `from_parts` has no adjudications to rerun the rule on, so it takes a
    // threshold and a calibration set as given; only rerunning the rule on
    // the named branches' adjudications, as `calibrate` does and storage
    // does when a record is written, tells an invented threshold apart.
    let calibration = adjudicated("c");
    let rule = ThresholdRule::LearnThenTest {
        alpha: 0.2,
        delta: 0.1,
    };
    let honest = PolicyRecord::calibrate("v", rule, 0.05, &calibration).unwrap();
    let invented = PolicyRecord::from_parts(
        "v",
        AutoThreshold::AtLeast(0.0),
        0.05,
        rule,
        honest.calibrated_on().to_vec(),
    )
    .unwrap();
    assert_ne!(invented.policy().threshold(), honest.policy().threshold());
    assert_eq!(invented.calibrated_on(), honest.calibrated_on());
    let rerun =
        PolicyRecord::calibrate(invented.version(), invented.rule(), 0.05, &calibration).unwrap();
    assert_eq!(rerun, honest);
    assert_ne!(rerun, invented);
}
