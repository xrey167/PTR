use ptr_analytics::{brier_score, expected_calibration_error, wilson_interval, Binning};
use ptr_feedback::Candidate;
use ptr_router::{
    evaluate_prequential, shuffled_labels, CalibrationTracker, CodeOutcome, FeatureMask,
    FindingCode, GlobalRateProbe, HintPolicy, LinearProbe, LookupTableProbe, PodRef, Propensity,
    RouterLimits, SelfContext, SelfModel, SelfModelHint, SelfModelLimits, SelfOutcome,
    SelfPrediction, SelfRole, SelfTraceStep, TraceLearner, OVERSIZED_CODE,
};
use ptr_types::{CandidateId, Generation, PodId, Probability, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

const REQUIRED: VerificationLevel = VerificationLevel::SampleVerified;

fn pod(name: &str, generation: u64) -> PodRef {
    PodRef::new(PodId::from(name), Generation(generation))
}

fn ctx(
    role: SelfRole,
    producer: Option<&str>,
    round: u32,
    budget: u32,
    codes: &[&str],
) -> SelfContext {
    SelfContext {
        role,
        producer: producer.map(|p| pod(p, 1)),
        round,
        budget_left: budget,
        required: REQUIRED,
        last_codes: codes.iter().map(|c| FindingCode::new(*c)).collect(),
    }
}

fn step(
    context: SelfContext,
    passed: bool,
    code: Option<&str>,
    routed: Option<&str>,
) -> SelfTraceStep {
    SelfTraceStep {
        context,
        outcome: SelfOutcome {
            passed,
            primary_code: code.map(FindingCode::new),
            routed_pod: routed.map(|p| pod(p, 1)),
            propensity: routed.map(|_| Propensity::new(0.5).unwrap()),
        },
    }
}

/// Seeded splitmix64 so the traces are fixed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn calibration_matches_hand_computed_values() {
    let mut tracker = CalibrationTracker::new();
    // (p_pass, passed): bins 9, 8, 7, 9.
    for (p, passed) in [(0.95, true), (0.85, false), (0.25, false), (0.95, true)] {
        tracker.record_binary(p, passed).unwrap();
    }
    let report = tracker.report();
    assert_eq!(report.n, 4);
    // Multiclass Brier terms: 0.005, 1.445, 0.125, 0.005 -> mean 0.395.
    assert!((report.brier.unwrap() - 0.395).abs() < 1e-12);
    // Top-label ECE: bin 9: 0.5 * 0.05; bin 8: 0.25 * 0.85; bin 7: 0.25 * 0.25.
    assert!((report.ece.unwrap() - 0.3).abs() < 1e-12);
    let accuracy = report.accuracy.unwrap();
    assert_eq!((accuracy.successes, accuracy.trials), (3, 4));
    assert_eq!(accuracy, wilson_interval(3, 4, 1.959_964).unwrap());
    assert_eq!(report.bins.len(), 3);
    assert_eq!(CalibrationTracker::new().report().brier, None);
}

#[test]
fn calibration_agrees_with_ptr_analytics_on_a_seeded_sequence() {
    let mut rng = Rng(11);
    let mut tracker = CalibrationTracker::new();
    let mut probabilities = Vec::new();
    let mut truth = Vec::new();
    for _ in 0..500 {
        let p = rng.unit();
        let passed = rng.unit() < p;
        tracker.record_binary(p, passed).unwrap();
        probabilities.push(vec![1.0 - p, p]);
        truth.push(usize::from(passed));
    }
    let report = tracker.report();
    let brier = brier_score(&probabilities, &truth).unwrap();
    let ece = expected_calibration_error(&probabilities, &truth, Binning::EqualWidth(10)).unwrap();
    assert!((report.brier.unwrap() - brier).abs() < 1e-12);
    assert!((report.ece.unwrap() - ece).abs() < 1e-12);
}

#[test]
fn calibration_tracker_refuses_invalid_input_and_stays_unchanged() {
    let mut tracker = CalibrationTracker::new();
    tracker.record_binary(0.7, true).unwrap();
    let before = tracker.clone();
    for p in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(tracker.record_binary(p, true).is_err());
        assert!(tracker.record(p, true, 0.0).is_err());
    }
    assert!(tracker.record(0.5, true, f64::NAN).is_err());
    assert!(tracker.record(0.5, true, -1.0).is_err());
    assert_eq!(tracker, before);
}

#[test]
fn prequential_order_is_pinned_by_exact_equalities() {
    // Updating before predicting would change at least one of these exactly.
    let mut model = SelfModel::new(SelfModelLimits::default());
    let steps = trace(3, 400);
    let mut expected = Vec::new();
    for s in &steps {
        let before = model.predict(&s.context);
        let returned = model.observe(s);
        assert_eq!(before, returned);
        expected.push(before);
    }
    let log: Vec<_> = model.log().collect();
    assert_eq!(log.len(), 400);
    let mut rebuilt = CalibrationTracker::new();
    for (i, entry) in log.iter().enumerate() {
        assert_eq!(entry.seq, i as u64);
        assert_eq!(entry.version, i as u64);
        assert_eq!(entry.prediction, expected[i]);
        rebuilt
            .record_binary(entry.prediction.pass_probability, entry.outcome.passed)
            .unwrap();
    }
    assert_eq!(model.report().pass, rebuilt.report());
    // The first prediction is the prior: nothing has been learned yet.
    assert_eq!(expected[0].pass_probability, 0.5);
    assert_eq!(expected[0].evidence, 0);
    let reliability: u64 = model.report().pass_reliability.iter().map(|b| b.n).sum();
    assert_eq!(reliability, 400);
}

#[test]
fn contexts_separate_what_they_predict() {
    let mut model = SelfModel::new(SelfModelLimits::default());
    let easy = ctx(SelfRole::Proposal, Some("a"), 0, 9, &[]);
    let hard = ctx(SelfRole::Proposal, Some("b"), 3, 1, &["target", "effect"]);
    for _ in 0..60 {
        model.observe(&step(easy.clone(), true, None, Some("a")));
        model.observe(&step(hard.clone(), false, Some("target"), Some("fix")));
    }
    let p_easy = model.predict(&easy);
    let p_hard = model.predict(&hard);
    assert!(p_easy.pass_probability > 0.9, "{}", p_easy.pass_probability);
    assert!(p_hard.pass_probability < 0.1, "{}", p_hard.pass_probability);
    assert_eq!(
        p_hard.code.unwrap().0,
        CodeOutcome::Code(FindingCode::new("target"))
    );
    assert_eq!(p_hard.route.unwrap().0, pod("fix", 1));
    assert_eq!(p_easy.evidence, 60);
    let unseen = ctx(SelfRole::Repair, None, 1, 2, &["zzz"]);
    let p_unseen = model.predict(&unseen);
    assert_eq!(p_unseen.evidence, 0);
    assert!((p_unseen.pass_probability - 0.5).abs() < 0.05);
    // The required level is a feature: the same context at another level is another context.
    let mut other = easy.clone();
    other.required = VerificationLevel::Deterministic;
    assert_eq!(model.predict(&other).evidence, 0);
}

#[test]
fn route_statistics_are_keyed_by_pod_generation() {
    let mut model = SelfModel::new(SelfModelLimits::default());
    let context = ctx(SelfRole::Proposal, None, 0, 5, &[]);
    for _ in 0..30 {
        let mut s = step(context.clone(), true, None, None);
        s.outcome.routed_pod = Some(pod("p", 1));
        model.observe(&s);
    }
    for _ in 0..60 {
        let mut s = step(context.clone(), true, None, None);
        s.outcome.routed_pod = Some(pod("p", 2));
        model.observe(&s);
    }
    // The successor generation is a different class; the old record is not lent to it.
    assert_eq!(model.predict(&context).route.unwrap().0, pod("p", 2));
    let first = ctx(SelfRole::Proposal, Some("x"), 0, 5, &[]);
    assert_eq!(model.predict(&first).evidence, 0);
    // Same producer Pod at another generation is a different context.
    let mut g1 = ctx(SelfRole::Proposal, None, 0, 5, &[]);
    g1.producer = Some(pod("x", 1));
    let mut g2 = g1.clone();
    g2.producer = Some(pod("x", 2));
    model.observe(&step(g1.clone(), false, Some("t"), None));
    assert_eq!(model.predict(&g1).evidence, 1);
    assert_eq!(model.predict(&g2).evidence, 0);
}

#[test]
fn tables_are_bounded_with_overflow_buckets() {
    let limits = SelfModelLimits {
        max_contexts: 2,
        max_codes_per_context: 2,
        max_pods_per_context: 2,
        max_repair_pairs: 2,
        max_repair_codes: 2,
        max_context_codes: 1,
        max_code_len: 8,
        max_log: 5,
    };
    let mut model = SelfModel::new(limits);
    for i in 0..50u32 {
        let code = format!("code-{i}");
        let pod_name = format!("pod-{i}");
        let context = ctx(SelfRole::Repair, Some(&pod_name), i, i, &[&code, "x", "y"]);
        model.observe(&step(context, i % 2 == 0, Some(&code), Some(&pod_name)));
    }
    assert!(model.contexts_tracked() <= 2);
    assert_eq!(model.report().contexts_tracked, 2);
    assert!(model.report().overflow_observations >= 40);
    assert_eq!(model.repair_pairs_tracked(), 2);
    assert!(model.repair_pairs_evicted() >= 40);
    assert_eq!(model.log().count(), 5);
    assert_eq!(model.log_evicted(), 45);
    let context = ctx(SelfRole::Repair, Some("pod-0"), 0, 0, &["code-0", "x", "y"]);
    assert!(model.code_distribution(&context).len() <= 4);
    // Oversized codes and Pod ids are replaced by distinct, bounded markers.
    let mut small = SelfModel::new(limits);
    let (long_a, long_b) = ("A".repeat(10_000), "B".repeat(10_000));
    for long in [&long_a, &long_b] {
        let mut s = step(
            ctx(SelfRole::Proposal, Some(long), 0, 0, &[long]),
            false,
            Some(long),
            Some(long),
        );
        s.outcome.propensity = None;
        small.observe(&s);
    }
    let logged: Vec<_> = small.log().collect();
    let code_a = logged[0].outcome.primary_code.clone().unwrap();
    let code_b = logged[1].outcome.primary_code.clone().unwrap();
    assert!(code_a.as_str().starts_with(OVERSIZED_CODE) && code_a.as_str().len() < 64);
    assert_ne!(code_a, code_b, "distinct long codes stay distinct");
    assert!(logged[0].outcome.routed_pod.as_ref().unwrap().pod.0.len() < 64);
    // Zero capacity everywhere is legal and never panics.
    let zero = SelfModelLimits {
        max_contexts: 0,
        max_codes_per_context: 0,
        max_pods_per_context: 0,
        max_repair_pairs: 0,
        max_repair_codes: 0,
        max_context_codes: 0,
        max_code_len: 0,
        max_log: 0,
    };
    let mut model = SelfModel::new(zero);
    let s = step(
        ctx(SelfRole::Repair, Some("p"), 1, 1, &["c"]),
        true,
        Some("c"),
        Some("p"),
    );
    model.observe(&s);
    let _ = model.predict(&s.context);
    let _ = model.hints(&s.context, &HintPolicy::default());
    assert_eq!(model.log().count(), 0);
}

#[test]
fn pair_eviction_keeps_the_best_evidenced_pairs() {
    let limits = SelfModelLimits {
        max_repair_pairs: 3,
        ..SelfModelLimits::default()
    };
    let mut model = SelfModel::new(limits);
    let repair = |code: &str, p: &str| {
        step(
            ctx(SelfRole::Repair, None, 0, 5, &[code]),
            true,
            None,
            Some(p),
        )
    };
    for _ in 0..30 {
        model.observe(&repair("target", "veteran"));
        model.observe(&repair("target", "regular"));
    }
    for _ in 0..10 {
        model.observe(&repair("target", "weaker"));
    }
    // A flood of novel pairs (first-come would keep the table frozen; plain LFU would let every
    // newcomer evict an established pair): it displaces at most the weakest established pair and
    // then only itself.
    for i in 0..100 {
        model.observe(&repair("target", &format!("novel-{i}")));
    }
    let code = FindingCode::new("target");
    assert!(model.repair_value(&code, &pod("veteran", 1)).is_some());
    assert!(model.repair_value(&code, &pod("regular", 1)).is_some());
    assert_eq!(model.repair_pairs_tracked(), 3);
    assert!(model.repair_pairs_evicted() >= 99);
}

#[test]
fn extreme_inputs_do_not_panic() {
    let mut model = SelfModel::new(SelfModelLimits::default());
    let context = ctx(SelfRole::Proposal, None, u32::MAX, u32::MAX, &[]);
    model.observe(&step(context.clone(), true, None, None));
    let _ = model.predict(&context);
    let _ = model.hints(
        &context,
        &HintPolicy {
            escalate_below: f64::NAN,
            min_evidence: 0,
            prefer_margin: f64::NAN,
        },
    );
    let many: Vec<String> = (0..2000).map(|i| format!("c{i}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    let big = ctx(SelfRole::Repair, Some("p"), 1, 1, &refs);
    model.observe(&step(big.clone(), false, Some("c1"), Some("p")));
    let _ = model.predict(&big);
}

fn trace(seed: u64, n: usize) -> Vec<SelfTraceStep> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let producer = ["a", "b", "c"][(rng.next() % 3) as usize];
            let code = ["target", "effect"][(rng.next() % 2) as usize];
            let context = ctx(
                SelfRole::Repair,
                Some(producer),
                (rng.next() % 4) as u32,
                (rng.next() % 6) as u32,
                &[code],
            );
            let p = if producer == "a" { 0.9 } else { 0.2 };
            let passed = rng.unit() < p;
            step(
                context,
                passed,
                (!passed).then_some(code),
                Some(["x", "y"][(rng.next() % 2) as usize]),
            )
        })
        .collect()
}

#[test]
fn replay_is_deterministic_and_the_digest_tracks_state() {
    let steps = trace(5, 300);
    let run = |steps: &[SelfTraceStep]| {
        let mut model = SelfModel::new(SelfModelLimits::default());
        let predictions: Vec<SelfPrediction> = steps.iter().map(|s| model.observe(s)).collect();
        (model, predictions)
    };
    let (a, pa) = run(&steps);
    let (b, pb) = run(&steps);
    assert_eq!(pa, pb);
    assert_eq!(a.digest(), b.digest());
    assert_eq!(a.report(), b.report());
    assert_eq!(a.version(), 300);
    let (c, _) = run(&trace(6, 300));
    assert_ne!(a.digest(), c.digest());
    let mut swapped = steps.clone();
    swapped.swap(0, 299);
    let (d, _) = run(&swapped);
    assert_ne!(a.digest(), d.digest());
    let (e, _) = run(&steps[..299]);
    assert_ne!(a.digest(), e.digest());
    // A feature mask is part of the model identity.
    let masked = SelfModel::with_features(
        SelfModelLimits::default(),
        FeatureMask {
            pod: false,
            ..FeatureMask::ALL
        },
    );
    assert_ne!(
        masked.digest(),
        SelfModel::new(SelfModelLimits::default()).digest()
    );
}

/// A repair trace where the failing producer is the same for every step and only the routed
/// repair Pod decides the outcome: `good` passes 90%, `bad` 20%, chosen uniformly (propensity
/// 0.5).
fn repair_trace(n: usize, seed: u64, codes: [&[&str]; 2]) -> Vec<SelfTraceStep> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|i| {
            let (routed, p) = if rng.next().is_multiple_of(2) {
                ("good", 0.9)
            } else {
                ("bad", 0.2)
            };
            let passed = rng.unit() < p;
            step(
                ctx(
                    SelfRole::Repair,
                    Some("failing-producer"),
                    1,
                    4,
                    codes[i % 2],
                ),
                passed,
                (!passed).then_some("target"),
                Some(routed),
            )
        })
        .collect()
}

#[test]
fn prefer_repair_hint_names_the_routed_pod_not_the_producer() {
    // Both orders of the same two codes: learning and hints use one canonical list, so a code
    // that is second in a report is learned exactly like a first one.
    let steps = repair_trace(600, 7, [&["target", "effect"], &["effect", "target"]]);
    let mut model = SelfModel::new(SelfModelLimits::default());
    for s in &steps {
        model.observe(s);
    }
    let context = ctx(
        SelfRole::Repair,
        Some("failing-producer"),
        1,
        4,
        &["effect", "target"],
    );
    let hints = model.hints(&context, &HintPolicy::default());
    let prefer: Vec<_> = hints
        .iter()
        .filter_map(|h| match h {
            SelfModelHint::PreferRepairPod {
                code,
                pod,
                generation,
                value_lcb,
                runner_up_ucb,
                ..
            } => Some((
                code.as_str(),
                pod.0.as_str(),
                *generation,
                *value_lcb,
                *runner_up_ucb,
            )),
            _ => None,
        })
        .collect();
    assert_eq!(prefer.len(), 2, "one hint per code: {prefer:?}");
    for (_, pod, generation, lcb, ucb) in &prefer {
        assert_eq!(*pod, "good");
        assert_eq!(*generation, Generation(1));
        assert!(lcb > ucb);
    }
    assert!(!prefer.iter().any(|(_, pod, ..)| *pod == "failing-producer"));
    let code = FindingCode::new("target");
    let (good, glow, ghigh, n) = model.repair_value(&code, &pod("good", 1)).unwrap();
    let (bad, ..) = model.repair_value(&code, &pod("bad", 1)).unwrap();
    // The inverse-propensity value of "always good" is its pass rate, 0.9.
    assert!(
        (good - 0.9).abs() < 0.08 && glow < good && good < ghigh,
        "{good}"
    );
    assert!((bad - 0.2).abs() < 0.08, "{bad}");
    assert!(n > 100);
}

#[test]
fn hints_need_evidence_two_pods_and_a_clear_gap() {
    let steps = repair_trace(600, 9, [&["target"], &["target"]]);
    let mut model = SelfModel::new(SelfModelLimits::default());
    for s in &steps {
        model.observe(s);
    }
    let context = ctx(
        SelfRole::Repair,
        Some("failing-producer"),
        1,
        4,
        &["target"],
    );
    let strict = HintPolicy {
        min_evidence: 10_000,
        ..HintPolicy::default()
    };
    assert!(model.hints(&context, &strict).is_empty());
    let impossible = HintPolicy {
        prefer_margin: 5.0,
        ..HintPolicy::default()
    };
    assert!(!model
        .hints(&context, &impossible)
        .iter()
        .any(|h| matches!(h, SelfModelHint::PreferRepairPod { .. })));
    // A negative margin counts as zero: the hint still needs lcb > ucb.
    let negative = HintPolicy {
        prefer_margin: -1.0,
        ..HintPolicy::default()
    };
    assert_eq!(
        model.hints(&context, &negative),
        model.hints(&context, &HintPolicy::default())
    );
    // A single Pod can never win against an absent runner-up.
    let mut single = SelfModel::new(SelfModelLimits::default());
    for _ in 0..200 {
        single.observe(&step(context.clone(), true, None, Some("only")));
    }
    assert!(single.hints(&context, &HintPolicy::default()).is_empty());
    // Equally good Pods give no hint.
    let mut rng = Rng(4);
    let mut tie = SelfModel::new(SelfModelLimits::default());
    for i in 0..600 {
        let routed = if i % 2 == 0 { "p" } else { "q" };
        tie.observe(&step(context.clone(), rng.unit() < 0.5, None, Some(routed)));
    }
    assert!(!tie
        .hints(&context, &HintPolicy::default())
        .iter()
        .any(|h| matches!(h, SelfModelHint::PreferRepairPod { .. })));
    // Hints are pure.
    let digest = model.digest();
    let _ = model.hints(&context, &HintPolicy::default());
    assert_eq!(digest, model.digest());
}

#[test]
fn escalate_early_needs_a_low_confidence_bound() {
    let mut model = SelfModel::new(SelfModelLimits::default());
    let low = ctx(SelfRole::Proposal, Some("a"), 2, 1, &["target"]);
    let few = ctx(SelfRole::Proposal, Some("b"), 2, 1, &["target"]);
    for _ in 0..80 {
        model.observe(&step(low.clone(), false, Some("target"), None));
    }
    for _ in 0..3 {
        model.observe(&step(few.clone(), false, Some("target"), None));
    }
    let policy = HintPolicy::default();
    assert!(model
        .hints(&low, &policy)
        .iter()
        .any(|h| matches!(h, SelfModelHint::EscalateEarly { evidence, .. } if *evidence == 80)));
    assert!(
        model.hints(&few, &policy).is_empty(),
        "3 observations are not evidence"
    );
}

/// Pass depends on the producer and on the required level: information a context-free probe
/// does not have.
fn informative_trace(seed: u64, n: usize) -> Vec<SelfTraceStep> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| {
            let producer = ["a", "b", "c", "d"][(rng.next() % 4) as usize];
            let required = if rng.next().is_multiple_of(2) {
                VerificationLevel::Unverified
            } else {
                VerificationLevel::Deterministic
            };
            let base = if producer == "a" || producer == "b" {
                0.9
            } else {
                0.15
            };
            let p = if required == VerificationLevel::Deterministic {
                base * 0.5
            } else {
                base
            };
            let mut context = ctx(SelfRole::Proposal, Some(producer), 0, 5, &[]);
            context.required = required;
            let passed = rng.unit() < p;
            step(context, passed, None, None)
        })
        .collect()
}

#[test]
fn c002_harness_compares_equal_information_learners_with_skill_and_ranking_metrics() {
    let steps = informative_trace(21, 3000);
    let full =
        evaluate_prequential(&mut SelfModel::new(SelfModelLimits::default()), &steps).unwrap();
    let constant = evaluate_prequential(&mut GlobalRateProbe::default(), &steps).unwrap();
    let lookup = evaluate_prequential(&mut LookupTableProbe::new(256), &steps).unwrap();
    let linear = evaluate_prequential(&mut LinearProbe::new(64, 0.2), &steps).unwrap();
    let ablated = evaluate_prequential(
        &mut SelfModel::with_features(
            SelfModelLimits::default(),
            FeatureMask {
                pod: false,
                required: false,
                ..FeatureMask::ALL
            },
        ),
        &steps,
    )
    .unwrap();
    // A calibrated constant has near-zero skill and chance-level ranking: ECE alone would
    // reward it, skill and AUROC do not.
    assert!(constant.brier_skill.unwrap().abs() < 0.02);
    assert!((constant.auroc.unwrap() - 0.5).abs() < 0.05);
    assert!(full.brier_skill.unwrap() > 0.25, "{:?}", full.brier_skill);
    assert!(full.auroc.unwrap() > 0.8);
    assert!(full.log_loss < constant.log_loss);
    assert!(lookup.brier_skill.unwrap() > 0.25);
    assert!(linear.brier_skill.unwrap() > constant.brier_skill.unwrap() + 0.05);
    // The feature-ablated self-model loses the information and the skill.
    assert!(ablated.brier_skill.unwrap() < full.brier_skill.unwrap() - 0.15);
    assert!(ablated.brier_skill.unwrap().abs() < 0.05);
    // The same trace length for all, and binary Brier is half the multiclass one.
    assert_eq!(full.n, constant.n);
    assert!((full.brier_binary * 2.0 - full.calibration.brier.unwrap()).abs() < 1e-12);
    // Negative control: with shuffled labels no learner has skill.
    let shuffled = shuffled_labels(&steps, 99);
    assert_eq!(shuffled.len(), steps.len());
    assert_eq!(shuffled, shuffled_labels(&steps, 99));
    assert_eq!(
        shuffled.iter().filter(|s| s.outcome.passed).count(),
        steps.iter().filter(|s| s.outcome.passed).count()
    );
    for report in [
        evaluate_prequential(&mut SelfModel::new(SelfModelLimits::default()), &shuffled).unwrap(),
        evaluate_prequential(&mut LookupTableProbe::new(256), &shuffled).unwrap(),
        evaluate_prequential(&mut LinearProbe::new(64, 0.2), &shuffled).unwrap(),
    ] {
        assert!(
            report.brier_skill.unwrap() < 0.03,
            "{:?}",
            report.brier_skill
        );
        assert!(
            (report.auroc.unwrap() - 0.5).abs() < 0.08,
            "{:?}",
            report.auroc
        );
    }
}

struct Scripted(std::cell::RefCell<Vec<f64>>);
impl TraceLearner for Scripted {
    fn predict(&self, _: &SelfContext) -> SelfPrediction {
        SelfPrediction {
            pass_probability: self.0.borrow_mut().remove(0),
            evidence: 0,
            code: None,
            route: None,
        }
    }
    fn learn(&mut self, _: &SelfTraceStep) {}
}

#[test]
fn ranking_and_log_loss_match_hand_computed_values() {
    // The classic AUC example: scores 0.1 0.4 0.35 0.8, labels 0 0 1 1 -> 3 of 4 pairs ordered.
    let labels = [false, false, true, true];
    let steps: Vec<SelfTraceStep> = labels
        .iter()
        .map(|passed| {
            step(
                ctx(SelfRole::Proposal, None, 0, 0, &[]),
                *passed,
                None,
                None,
            )
        })
        .collect();
    let mut learner = Scripted(std::cell::RefCell::new(vec![0.1, 0.4, 0.35, 0.8]));
    let report = evaluate_prequential(&mut learner, &steps).unwrap();
    assert!((report.auroc.unwrap() - 0.75).abs() < 1e-12);
    // Binary Brier: (0.01 + 0.16 + 0.4225 + 0.04) / 4; base rate 0.5 -> skill 1 - brier / 0.25.
    let brier = (0.01 + 0.16 + 0.65f64 * 0.65 + 0.04) / 4.0;
    assert!((report.brier_binary - brier).abs() < 1e-12);
    assert!((report.brier_skill.unwrap() - (1.0 - brier / 0.25)).abs() < 1e-12);
    let log_loss = -(0.9f64.ln() + 0.6f64.ln() + 0.35f64.ln() + 0.8f64.ln()) / 4.0;
    assert!((report.log_loss - log_loss).abs() < 1e-12);
    // Ties count half: a constant scorer is at 0.5; one class absent gives None.
    let mut constant = Scripted(std::cell::RefCell::new(vec![0.5; 4]));
    assert_eq!(
        evaluate_prequential(&mut constant, &steps).unwrap().auroc,
        Some(0.5)
    );
    let one_class: Vec<SelfTraceStep> = steps[..2].to_vec();
    let mut learner = Scripted(std::cell::RefCell::new(vec![0.1, 0.2]));
    assert_eq!(
        evaluate_prequential(&mut learner, &one_class)
            .unwrap()
            .auroc,
        None
    );
}

struct Broken;
impl TraceLearner for Broken {
    fn predict(&self, _: &SelfContext) -> SelfPrediction {
        SelfPrediction {
            pass_probability: f64::NAN,
            evidence: 0,
            code: None,
            route: None,
        }
    }
    fn learn(&mut self, _: &SelfTraceStep) {}
}

#[test]
fn a_probe_with_an_invalid_prediction_is_refused_not_scored() {
    assert!(evaluate_prequential(&mut Broken, &trace(1, 3)).is_err());
}

#[test]
fn the_route_head_scores_the_other_bucket_as_a_class() {
    let limits = SelfModelLimits {
        max_pods_per_context: 1,
        ..SelfModelLimits::default()
    };
    let mut model = SelfModel::new(limits);
    let context = ctx(SelfRole::Proposal, None, 0, 5, &[]);
    let names = ["first", "o1", "o2", "o3", "o4", "o5", "o6", "o7"];
    for name in names.iter().cycle().take(40) {
        model.observe(&step(context.clone(), true, None, Some(name)));
    }
    // Every step after the first has a route prediction, including those where the top class is
    // the Other bucket: skipping them would bias the calibration.
    assert_eq!(model.report().route.n, 39);
}

#[test]
fn outcomes_built_from_reports_follow_the_verifier_alone() {
    let limits = RouterLimits::default();
    let statuses = [
        VerificationStatus::Pass,
        VerificationStatus::Fail,
        VerificationStatus::Disputed,
        VerificationStatus::Unknown,
    ];
    let levels = [
        VerificationLevel::Unverified,
        VerificationLevel::LatentAgreement,
        VerificationLevel::SampleVerified,
        VerificationLevel::FullSemantic,
        VerificationLevel::Deterministic,
    ];
    let id = CandidateId::from("c");
    for status in statuses {
        for level in levels {
            for required in levels {
                for findings in [
                    vec![],
                    vec![("style", false)],
                    vec![("target", true)],
                    vec![("a", false), ("target", true), ("b", true)],
                ] {
                    let report = VerificationReport {
                        status,
                        level,
                        score: Probability::new(0.5).unwrap(),
                        findings: findings
                            .iter()
                            .map(|(c, h)| Finding {
                                code: (*c).into(),
                                message: String::new(),
                                hard: *h,
                            })
                            .collect(),
                    };
                    let outcome = SelfOutcome::from_report(&report, required, &limits, None);
                    let candidate = Candidate {
                        id: id.clone(),
                        value: (),
                        verification: Some(report.clone()),
                    };
                    assert_eq!(
                        outcome.passed,
                        candidate.passes(required),
                        "{status:?} {level:?} {required:?} {findings:?}"
                    );
                    let first_hard = findings.iter().find(|(_, h)| *h).map(|(c, _)| *c);
                    assert_eq!(
                        outcome.primary_code.as_ref().map(FindingCode::as_str),
                        first_hard
                    );
                    let context =
                        SelfContext::from_report(SelfRole::Repair, None, 0, 1, required, &report);
                    let hard: Vec<&str> = findings
                        .iter()
                        .filter(|(_, h)| *h)
                        .map(|(c, _)| *c)
                        .collect();
                    let codes: Vec<&str> =
                        context.last_codes.iter().map(FindingCode::as_str).collect();
                    assert_eq!(codes, hard);
                    assert_eq!(context.required, required);
                }
            }
        }
    }
}

#[test]
fn refusal_codes_extend_the_context_without_duplicates() {
    let context = ctx(SelfRole::Proposal, None, 0, 0, &["target"])
        .with_refusal_codes(["gate", "target", "gate"]);
    let codes: Vec<&str> = context.last_codes.iter().map(FindingCode::as_str).collect();
    assert_eq!(codes, ["target", "gate"]);
}

#[test]
fn a_model_trained_to_expect_passes_does_not_change_what_the_verifier_says() {
    // The model's belief never enters `SelfOutcome::from_report`: the label of a forged Pass is
    // fail whatever the model predicts, and the next observation corrects the model, not the
    // other way round.
    let mut model = SelfModel::new(SelfModelLimits::default());
    let context = ctx(SelfRole::Proposal, None, 0, 3, &[]);
    for _ in 0..200 {
        model.observe(&step(context.clone(), true, None, None));
    }
    let belief = model.predict(&context).pass_probability;
    assert!(belief > 0.95);
    let forged = VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(1.0).unwrap(),
        findings: vec![Finding {
            code: "target".into(),
            message: String::new(),
            hard: true,
        }],
    };
    let outcome = SelfOutcome::from_report(
        &forged,
        VerificationLevel::Unverified,
        &RouterLimits::default(),
        None,
    );
    assert!(!outcome.passed);
    model.observe(&SelfTraceStep {
        context: context.clone(),
        outcome,
    });
    assert!(model.predict(&context).pass_probability < belief);
}
