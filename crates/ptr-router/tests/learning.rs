//! L1 learning of the finding router: a Thompson-sampling bandit over the repair Pods of a code,
//! trained on verifier verdicts only, evaluated off-policy before it may be promoted by a host.

use ptr_router::{
    classify, compare_policies, estimate_policy, AdmissibleScorer, Confidence, EligibilitySnapshot,
    Estimator, Exploration, FallbackPolicy, FindingCode, FindingRegistry, FindingRouteErrorKind,
    FindingRouteRecord, FindingRouteRequest, FindingRouter, LearnedPolicy, LearnerConfig,
    LoggedDecision, NonceSequence, OpeError, PolicyAction, PolicyBook, PolicyBookError,
    PromotionGate, PromotionRefusal, PromotionRule, RegistryChange, RegistryUpdate, ReportClass,
    RewardModel, RouteClass, RouteDistribution, RouteError, RouterConfig, RouterLimits,
    RoutingLearner, SamplingKey, ScoringContext, UniformScorer, Verdict, VerifierReward,
};
use ptr_types::{Generation, PodId, Probability, RequestId, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};

const CODE: &str = "wrong_answer";

fn pod(id: &str) -> PodId {
    PodId::from(id)
}

fn repair_class() -> RouteClass {
    RouteClass::Repair(FindingCode::new(CODE))
}

/// A seeded generator for the simulated world (not the router's).
struct World(u64);

impl World {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn hard_fail() -> VerificationReport {
    VerificationReport {
        status: VerificationStatus::Fail,
        level: VerificationLevel::Deterministic,
        score: Probability::new(0.0).unwrap(),
        findings: vec![Finding {
            code: CODE.to_owned(),
            message: String::new(),
            hard: true,
        }],
    }
}

fn registry(pods: &[&str]) -> FindingRegistry {
    let mut registry = FindingRegistry::new();
    for id in pods {
        let expected = registry.generation();
        registry
            .apply(RegistryUpdate {
                expected,
                change: RegistryChange::AddRepair {
                    code: FindingCode::new(CODE),
                    pod: pod(id),
                },
            })
            .unwrap();
    }
    registry
}

fn eligible(pods: &[&str]) -> EligibilitySnapshot {
    let mut view = EligibilitySnapshot::new();
    for id in pods {
        view.set(pod(id), Generation(1));
    }
    view
}

fn config(epsilon: f64) -> RouterConfig {
    RouterConfig {
        fallback: FallbackPolicy::default(),
        limits: RouterLimits::default(),
        exploration: Exploration::new(epsilon).unwrap(),
    }
}

fn router<S: AdmissibleScorer>(
    pods: &[&str],
    epsilon: f64,
    scorer: S,
    key: u8,
) -> FindingRouter<S> {
    FindingRouter::new(
        registry(pods),
        config(epsilon),
        scorer,
        SamplingKey::from_bytes([key; 32]),
        NonceSequence::starting_at([key; 8], 0),
    )
}

fn route_record<S: AdmissibleScorer>(
    router: &FindingRouter<S>,
    view: &EligibilitySnapshot,
    id: &str,
) -> FindingRouteRecord {
    let request = RequestId::from(id);
    let report = hard_fail();
    let outcome = router
        .route(&FindingRouteRequest {
            request_id: &request,
            request_digest: [3; 32],
            report: &report,
            required: VerificationLevel::Deterministic,
            eligibility: view,
        })
        .unwrap();
    outcome.records()[0].clone()
}

const EPSILON: f64 = 0.1;

/// Route `n` requests under `scorer`, settle each with the world's verdict, return the learner.
fn collect<S: AdmissibleScorer>(
    learner: &RoutingLearner,
    router: &FindingRouter<S>,
    truth: &[(&str, f64)],
    n: usize,
    tag: &str,
    world: &mut World,
    invert: bool,
) {
    let view = eligible(&truth.iter().map(|(p, _)| *p).collect::<Vec<_>>());
    for i in 0..n {
        let record = route_record(router, &view, &format!("{tag}-{i}"));
        let p = truth.iter().find(|(p, _)| *p == record.pod().0).unwrap().1;
        let passed = world.unit() < p;
        learner.settle(&record, VerifierReward::replayed(passed != invert));
    }
}

fn learner(seed: u64) -> RoutingLearner {
    RoutingLearner::new(LearnerConfig {
        exploration: Exploration::new(EPSILON).unwrap(),
        ..LearnerConfig::new("arith", seed)
    })
    .unwrap()
}

const TRUTH: [(&str, f64); 2] = [("bad", 0.2), ("good", 0.9)];

#[test]
fn the_bandit_prefers_the_better_pod_and_the_router_floor_keeps_the_other_reachable() {
    let learner = learner(5);
    let uniform = router(&["bad", "good"], EPSILON, UniformScorer, 1);
    let mut world = World(11);
    collect(&learner, &uniform, &TRUTH, 400, "t", &mut world, false);
    let policy = learner.snapshot();
    assert!(policy.mean(&repair_class(), &pod("good")) > 0.8);
    assert!(policy.mean(&repair_class(), &pod("bad")) < 0.35);
    assert!(policy.counts(&repair_class(), &pod("good")).cleared > 100);

    // Routed by the learned policy: the better pod gets almost everything, and the pod the
    // policy has written off keeps at least the router's floor of epsilon / 2.
    let learned = router(&["bad", "good"], EPSILON, policy, 2);
    let view = eligible(&["bad", "good"]);
    let n = 4000;
    let mut good = 0;
    let mut min_propensity: f64 = 1.0;
    for i in 0..n {
        let record = route_record(&learned, &view, &format!("l-{i}"));
        good += usize::from(record.pod().0 == "good");
        min_propensity = min_propensity.min(record.propensity().get());
    }
    let share = good as f64 / n as f64;
    assert!(share > 0.88, "{share}");
    assert!(share < 0.99, "the floor is not respected: {share}");
    assert!(min_propensity >= EPSILON / 2.0 - 1e-9, "{min_propensity}");
}

#[test]
fn learned_weights_are_a_function_of_the_snapshot_and_change_with_evidence() {
    let learner = learner(9);
    let uniform = router(&["bad", "good"], EPSILON, UniformScorer, 1);
    let mut world = World(2);
    let before = learner.snapshot();
    collect(&learner, &uniform, &TRUTH, 50, "d", &mut world, false);
    let after = learner.snapshot();
    assert_ne!(before.digest(), after.digest());
    assert_eq!(after.cutoff(), 50);
    assert_eq!(learner.snapshot().digest(), after.digest());
    // The same snapshot routes the same way for the same keyed draws.
    let a = router(&["bad", "good"], EPSILON, after.clone(), 4);
    let b = router(&["bad", "good"], EPSILON, after, 4);
    let view = eligible(&["bad", "good"]);
    for i in 0..50 {
        let (x, y) = (
            route_record(&a, &view, &format!("r{i}")),
            route_record(&b, &view, &format!("r{i}")),
        );
        assert_eq!(x.pod(), y.pod());
        assert_eq!(x.support(), y.support());
    }
}

/// Train on 400 decisions, snapshot, then log `eval` more under the uniform incumbent.
fn trained_and_logged(
    seed: u64,
    eval: usize,
    invert: bool,
) -> (RoutingLearner, LearnedPolicy, Vec<LoggedDecision>) {
    let learner = learner(seed);
    let uniform = router(&["bad", "good"], EPSILON, UniformScorer, 1);
    let mut world = World(seed.wrapping_mul(7919));
    collect(&learner, &uniform, &TRUTH, 400, "train", &mut world, invert);
    let policy = learner.snapshot();
    let fresh = router(&["bad", "good"], EPSILON, UniformScorer, 6);
    collect(&learner, &fresh, &TRUTH, eval, "eval", &mut world, false);
    let log = learner.log_from(policy.cutoff());
    (learner, policy, log)
}

fn gate() -> PromotionGate {
    PromotionGate::new(PromotionRule::new(0.05, 500, 100.0)).unwrap()
}

#[test]
fn a_clearly_better_policy_is_promoted_on_held_out_decisions_and_can_be_rolled_back() {
    let (_, policy, log) = trained_and_logged(21, 1500, false);
    assert_eq!(log.len(), 1500);
    assert!(log.iter().all(|d| d.seq() >= policy.cutoff()));
    let mut book = PolicyBook::new("arith");
    let incumbent = book.scorer();
    let assessment = gate()
        .assess(&log, &policy, &incumbent, book.active_digest())
        .unwrap();
    let evidence = match &assessment.verdict {
        Verdict::Promote(evidence) => evidence.clone(),
        other => panic!(
            "expected promotion, got {other:?} {:?}",
            assessment.comparison
        ),
    };
    // Truth: learned ~ 0.95 * 0.9 + 0.05 * 0.2 = 0.865, uniform 0.55, difference ~ 0.31.
    assert!(
        (evidence.difference() - 0.31).abs() < 0.08,
        "{}",
        evidence.difference()
    );
    assert!(evidence.lower_bound() > 0.05);
    assert_eq!(evidence.records(), 1500);

    // Promotion is a host action that records an event; nothing promoted before it.
    assert!(book.events().is_empty());
    let initial = book.active_digest();
    let event = book.promote(policy.clone(), evidence.clone()).unwrap();
    assert_eq!(event.action, PolicyAction::Promoted);
    assert_eq!(event.from, initial);
    assert_eq!(event.to, policy.digest());
    assert_eq!(event.evidence, Some(evidence.digest()));
    assert!(book.is_learned());
    assert_eq!(book.active_digest(), policy.digest());

    // The router that holds the book's handle now routes by the learned policy.
    let live = router(&["bad", "good"], EPSILON, book.scorer(), 8);
    let view = eligible(&["bad", "good"]);
    let good = (0..2000)
        .filter(|i| route_record(&live, &view, &format!("p{i}")).pod().0 == "good")
        .count();
    assert!(good > 1700, "{good}");

    // The same evidence cannot be applied twice: its incumbent is no longer the active policy.
    assert_eq!(
        book.promote(policy.clone(), evidence).unwrap_err(),
        PolicyBookError::StaleEvidence
    );
    // Rolled back, routing is uniform again and an event says so.
    let rolled = book.rollback().unwrap();
    assert_eq!(rolled.action, PolicyAction::RolledBack);
    assert_eq!(rolled.from, policy.digest());
    assert_eq!(rolled.to, initial);
    assert!(!book.is_learned());
    let good = (0..2000)
        .filter(|i| route_record(&live, &view, &format!("q{i}")).pod().0 == "good")
        .count();
    assert!((800..1200).contains(&good), "{good}");
    assert_eq!(
        book.rollback().unwrap_err(),
        PolicyBookError::NothingToRollBack
    );
    assert_eq!(book.events().len(), 2);
    assert_eq!(book.events()[1].sequence, 2);
}

#[test]
fn evidence_is_bound_to_its_candidate_its_incumbent_and_its_scope() {
    let (_, policy, log) = trained_and_logged(31, 1500, false);
    let (_, other, _) = trained_and_logged(32, 10, false);
    let mut book = PolicyBook::new("arith");
    let evidence = gate()
        .assess(&log, &policy, &book.scorer(), book.active_digest())
        .unwrap()
        .evidence()
        .cloned()
        .expect("promotable");
    assert_eq!(
        book.promote(other, evidence.clone()).unwrap_err(),
        PolicyBookError::EvidenceForAnotherCandidate
    );
    let mut elsewhere = PolicyBook::new("code");
    assert_eq!(
        elsewhere
            .promote(policy.clone(), evidence.clone())
            .unwrap_err(),
        PolicyBookError::ScopeMismatch
    );
    // Evidence sealed against another incumbent digest is stale for this book.
    let against_other = gate()
        .assess(&log, &policy, &book.scorer(), [9; 32])
        .unwrap()
        .evidence()
        .cloned()
        .unwrap();
    assert_eq!(
        book.promote(policy.clone(), against_other).unwrap_err(),
        PolicyBookError::StaleEvidence
    );
    // Nothing was installed by any refusal.
    assert!(!book.is_learned());
    assert!(book.events().is_empty());
    book.promote(policy, evidence).unwrap();
}

#[test]
fn promotion_is_refused_when_the_lower_bound_does_not_beat_the_incumbent() {
    // A policy that learned the wrong way round is worse than the incumbent.
    let (_, worse, log) = trained_and_logged(41, 1500, true);
    let book = PolicyBook::new("arith");
    let assessment = gate()
        .assess(&log, &worse, &book.scorer(), book.active_digest())
        .unwrap();
    assert!(
        assessment.comparison.difference < 0.0,
        "{:?}",
        assessment.comparison
    );
    assert!(matches!(
        assessment.verdict,
        Verdict::Refuse(PromotionRefusal::LowerBoundNotAboveMargin { .. })
    ));

    // A policy trained on nothing is the prior: its value is the incumbent's, within noise.
    let (_, _, log) = trained_and_logged(42, 1500, false);
    let untrained = learner(43).snapshot();
    let assessment = gate()
        .assess(&log, &untrained, &book.scorer(), book.active_digest())
        .unwrap();
    assert!(
        assessment.evidence().is_none(),
        "{:?}",
        assessment.comparison
    );

    // A real improvement still fails a margin it cannot reach.
    let (_, policy, log) = trained_and_logged(44, 1500, false);
    let strict = PromotionGate::new(PromotionRule::new(0.6, 500, 100.0)).unwrap();
    let assessment = strict
        .assess(&log, &policy, &book.scorer(), book.active_digest())
        .unwrap();
    assert!(matches!(
        assessment.verdict,
        Verdict::Refuse(PromotionRefusal::LowerBoundNotAboveMargin { .. })
    ));
    // And too short a log, whatever it shows.
    let assessment = gate()
        .assess(&log[..100], &policy, &book.scorer(), book.active_digest())
        .unwrap();
    assert!(matches!(
        assessment.verdict,
        Verdict::Refuse(PromotionRefusal::TooFewRecords {
            have: 100,
            need: 500
        })
    ));
}

#[test]
fn a_candidate_is_never_evaluated_on_decisions_it_was_trained_on() {
    let (learner, policy, _) = trained_and_logged(51, 800, false);
    let everything = learner.log_from(0);
    assert!(everything.len() > 800);
    let book = PolicyBook::new("arith");
    let error = gate()
        .assess(&everything, &policy, &book.scorer(), book.active_digest())
        .unwrap_err();
    assert!(
        matches!(error, OpeError::OverlapsTraining { seq: 0, .. }),
        "{error:?}"
    );
}

// --- the estimators -------------------------------------------------------------------------

/// A target policy that prefers its weights as given.
struct Fixed(Vec<(&'static str, f64)>);

impl AdmissibleScorer for Fixed {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        Ok(context
            .admissible
            .iter()
            .map(|p| {
                let w = self
                    .0
                    .iter()
                    .find(|(n, _)| *n == p.0)
                    .map_or(0.0, |(_, w)| *w);
                (p.clone(), w)
            })
            .collect())
    }
}

struct Constant(f64);

impl RewardModel for Constant {
    fn predict(&self, _: &RouteClass, _: &PodId) -> f64 {
        self.0
    }
}

struct Truth;

impl RewardModel for Truth {
    fn predict(&self, _: &RouteClass, p: &PodId) -> f64 {
        match p.0.as_str() {
            "a" => 0.2,
            "b" => 0.5,
            _ => 0.8,
        }
    }
}

fn synthetic_log(n: usize, world: &mut World) -> Vec<LoggedDecision> {
    let logging =
        RouteDistribution::new(vec![(pod("a"), 0.5), (pod("b"), 0.3), (pod("c"), 0.2)]).unwrap();
    let support: Vec<(PodId, u64)> = ["a", "b", "c"]
        .iter()
        .map(|p| (pod(p), logging.width(&pod(p))))
        .collect();
    let floor = Exploration::new(0.1).unwrap();
    (0..n)
        .map(|i| {
            let draw = world.next() >> 11;
            let mut upper = 0;
            let mut chosen = pod("c");
            for (p, width) in &support {
                upper += width;
                if draw < upper {
                    chosen = p.clone();
                    break;
                }
            }
            let truth = Truth.predict(&repair_class(), &chosen);
            LoggedDecision::replayed(
                i as u64,
                repair_class(),
                support.clone(),
                chosen,
                VerifierReward::replayed(world.unit() < truth),
                floor,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn ips_snips_and_dr_are_unbiased_on_a_log_with_known_value_and_cover() {
    // Target mixture at epsilon 0.1 over weights (0.1, 0.2, 0.7): pi = 0.9 w + 0.1 / 3.
    let target = Fixed(vec![("a", 0.1), ("b", 0.2), ("c", 0.7)]);
    let pi = [
        0.9 * 0.1 + 0.1 / 3.0,
        0.9 * 0.2 + 0.1 / 3.0,
        0.9 * 0.7 + 0.1 / 3.0,
    ];
    let truth = pi[0] * 0.2 + pi[1] * 0.5 + pi[2] * 0.8;
    let reps = 40;
    let mut world = World(77);
    let mut sums = [0.0_f64; 3];
    let mut squares = [0.0_f64; 3];
    let mut covered = [0_usize; 3];
    let wrong_model = Constant(0.5);
    for _ in 0..reps {
        let log = synthetic_log(3000, &mut world);
        // DR with a deliberately wrong reward model is still unbiased: the propensities are right.
        let estimate = estimate_policy(&log, &target, &wrong_model).unwrap();
        let values = [estimate.ips, estimate.snips, estimate.doubly_robust];
        let errors = [
            estimate.se_ips,
            estimate.se_snips,
            estimate.se_doubly_robust,
        ];
        for k in 0..3 {
            sums[k] += values[k];
            squares[k] += values[k] * values[k];
            if (values[k] - truth).abs() <= 1.96 * errors[k] {
                covered[k] += 1;
            }
        }
        assert!(estimate.effective_sample_size > 1000.0);
    }
    for k in 0..3 {
        let mean = sums[k] / reps as f64;
        let variance = squares[k] / reps as f64 - mean * mean;
        let se_of_mean = (variance / reps as f64).sqrt();
        assert!(
            (mean - truth).abs() < 3.0 * se_of_mean + 1e-3,
            "estimator {k}: mean {mean} truth {truth} se {se_of_mean}"
        );
        assert!(
            covered[k] >= 34,
            "estimator {k} covered {} of {reps}",
            covered[k]
        );
    }
}

#[test]
fn doubly_robust_with_a_good_model_has_a_smaller_error_than_ips() {
    let target = Fixed(vec![("a", 0.1), ("b", 0.2), ("c", 0.7)]);
    let mut world = World(5);
    let log = synthetic_log(3000, &mut world);
    let good = estimate_policy(&log, &target, &Truth).unwrap();
    assert!(good.se_doubly_robust < good.se_ips, "{good:?}");
}

#[test]
fn the_comparison_of_a_policy_with_itself_is_exactly_zero() {
    let target = Fixed(vec![("a", 0.1), ("b", 0.2), ("c", 0.7)]);
    let mut world = World(8);
    let log = synthetic_log(500, &mut world);
    for estimator in [Estimator::Ips, Estimator::Snips, Estimator::DoublyRobust] {
        let c = compare_policies(
            &log,
            &target,
            &target,
            &Truth,
            estimator,
            Confidence::ONE_SIDED_95,
        )
        .unwrap();
        assert!(c.difference.abs() < 1e-12);
        assert!(c.difference_se.abs() < 1e-12);
    }
}

#[test]
fn estimation_refuses_what_it_cannot_trust() {
    let target = Fixed(vec![("a", 1.0), ("b", 1.0), ("c", 1.0)]);
    assert_eq!(
        estimate_policy(&[], &target, &Truth).unwrap_err(),
        OpeError::EmptyLog
    );
    let mut world = World(1);
    let log = synthetic_log(50, &mut world);

    // A policy that names a Pod outside the logged admissible set.
    struct Escaping;
    impl AdmissibleScorer for Escaping {
        fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
            Ok(vec![(PodId::from("zzz"), 1.0)])
        }
    }
    assert!(matches!(
        estimate_policy(&log, &Escaping, &Truth).unwrap_err(),
        OpeError::PolicyOutsideAdmissibleSet { .. }
    ));

    // A policy with no mass at all.
    struct Silent;
    impl AdmissibleScorer for Silent {
        fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
            Ok(Vec::new())
        }
    }
    assert!(matches!(
        estimate_policy(&log, &Silent, &Truth).unwrap_err(),
        OpeError::PolicyFailed { .. }
    ));

    // A reward model outside [0, 1].
    assert_eq!(
        estimate_policy(&log, &target, &Constant(1.5)).unwrap_err(),
        OpeError::RewardModelOutOfRange
    );
    assert_eq!(
        estimate_policy(&log, &target, &Constant(f64::NAN)).unwrap_err(),
        OpeError::RewardModelOutOfRange
    );

    // A log whose logger had no floor: a propensity far below epsilon / n.
    let support = vec![
        (pod("a"), 1_u64 << 20),
        (pod("b"), (1_u64 << 53) - (1 << 20)),
    ];
    let thin = LoggedDecision::replayed(
        0,
        repair_class(),
        support,
        pod("a"),
        VerifierReward::NOT_CLEARED,
        Exploration::new(0.1).unwrap(),
    )
    .unwrap();
    assert_eq!(
        estimate_policy(&[thin], &target, &Truth).unwrap_err(),
        OpeError::BelowFloor { seq: 0 }
    );
    // A log without any exploration floor cannot even be built.
    assert!(LoggedDecision::replayed(
        0,
        repair_class(),
        vec![(pod("a"), 1_u64 << 53)],
        pod("a"),
        VerifierReward::NOT_CLEARED,
        Exploration::NONE,
    )
    .is_err());
}

// --- authority --------------------------------------------------------------------------------

#[test]
fn a_scorer_cannot_choose_an_ineligible_pod_or_remove_the_floor() {
    // A scorer that names a pod the eligibility view does not list is refused.
    struct Everyone;
    impl AdmissibleScorer for Everyone {
        fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
            Ok(vec![(pod("a"), 1.0), (pod("b"), 1.0)])
        }
    }
    let router_ab = router(&["a", "b"], EPSILON, Everyone, 3);
    let only_a = eligible(&["a"]);
    let request = RequestId::from("x");
    let report = hard_fail();
    let error = router_ab
        .route(&FindingRouteRequest {
            request_id: &request,
            request_digest: [0; 32],
            report: &report,
            required: VerificationLevel::Deterministic,
            eligibility: &only_a,
        })
        .unwrap_err();
    assert!(matches!(
        error.kind(),
        FindingRouteErrorKind::OutsideAdmissibleSet { pod } if pod.0 == "b"
    ));

    // A learned policy wrapped so that it also names a third pod is refused too.
    struct Padded(LearnedPolicy);
    impl AdmissibleScorer for Padded {
        fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
            let mut weights = self.0.weights(context)?;
            weights.push((pod("not-admissible"), 5.0));
            Ok(weights)
        }
    }
    let padded = router(&["a", "b"], EPSILON, Padded(learner(1).snapshot()), 3);
    let both = eligible(&["a", "b"]);
    assert!(matches!(
        padded
            .route(&FindingRouteRequest {
                request_id: &request,
                request_digest: [0; 32],
                report: &report,
                required: VerificationLevel::Deterministic,
                eligibility: &both,
            })
            .unwrap_err()
            .kind(),
        FindingRouteErrorKind::OutsideAdmissibleSet { .. }
    ));

    // A scorer that puts all the mass on one pod cannot take the floor from the other.
    let skewed = router(&["a", "b", "c"], 0.3, Fixed(vec![("a", 1.0)]), 3);
    let all = eligible(&["a", "b", "c"]);
    for i in 0..200 {
        let record = route_record(&skewed, &all, &format!("f{i}"));
        for (_, width) in record.support() {
            let share = *width as f64 / RouteDistribution::RESOLUTION as f64;
            assert!(share >= 0.3 / 3.0 - 1e-9, "{share}");
        }
    }
    // A scorer that gives no pod any weight is refused rather than defaulted.
    let silent = router(&["a", "b"], EPSILON, Fixed(vec![]), 3);
    assert!(matches!(
        silent
            .route(&FindingRouteRequest {
                request_id: &request,
                request_digest: [0; 32],
                report: &report,
                required: VerificationLevel::Deterministic,
                eligibility: &both,
            })
            .unwrap_err()
            .kind(),
        FindingRouteErrorKind::Distribution(_) | FindingRouteErrorKind::NoMass
    ));
}

#[test]
fn a_reward_is_the_verifier_verdict_and_a_score_cannot_make_one() {
    // A forged pass: score 1.0, status Pass, but a hard finding. Not accepted, so reward zero.
    let forged = VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(1.0).unwrap(),
        findings: vec![Finding {
            code: CODE.to_owned(),
            message: String::new(),
            hard: true,
        }],
    };
    let limits = RouterLimits::default();
    let class = classify(&forged, VerificationLevel::Deterministic, &limits);
    assert!(!matches!(class, ReportClass::Accept(_)));
    assert_eq!(
        VerifierReward::from_class(&class),
        VerifierReward::NOT_CLEARED
    );
    // A clean pass at the required level is the only reward of one.
    let clean = VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(0.0).unwrap(),
        findings: Vec::new(),
    };
    let class = classify(&clean, VerificationLevel::Deterministic, &limits);
    assert!(VerifierReward::from_class(&class).passed());
    // The same clean pass below the required level is zero, whatever the model says about it.
    assert_eq!(VerifierReward::from_class(&class).value(), 1.0);
    let weak = VerificationReport {
        level: VerificationLevel::LatentAgreement,
        ..clean
    };
    let class = classify(&weak, VerificationLevel::Deterministic, &limits);
    assert_eq!(VerifierReward::from_class(&class).value(), 0.0);
}

fn code_lines(source: &str) -> Vec<(usize, &str)> {
    source
        .lines()
        .enumerate()
        .map(|(i, line)| (i + 1, line.trim_start()))
        .filter(|(_, line)| !line.starts_with("//"))
        .collect()
}

#[test]
fn the_learning_sources_have_no_handle_on_gates_thresholds_verifiers_or_eligibility() {
    let sources = [
        ("learning.rs", include_str!("../src/learning.rs")),
        ("ope.rs", include_str!("../src/ope.rs")),
        ("policy_book.rs", include_str!("../src/policy_book.rs")),
    ];
    let forbidden = [
        "FindingRegistry",
        "RegistryUpdate",
        "RegistryChange",
        "update_registry",
        "Accepted",
        "FindingRouter",
        "RouterConfig",
        "FallbackPolicy",
        "FallbackAction",
        "RouterLimits",
        "PodEligibility",
        "EligibilitySnapshot",
        "VerificationLevel",
        "VerificationReport",
        "VerificationStatus",
        "ptr_verifier",
        "ptr_feedback",
        "classify(",
        "threshold",
        "required",
        "Verifier<",
    ];
    for (name, source) in sources {
        for (line_number, line) in code_lines(source) {
            for token in forbidden {
                assert!(
                    !line.contains(token),
                    "{name}:{line_number} names `{token}`: {line}"
                );
            }
        }
    }
    // The only implementors of the scorer trait in the learning sources are the policy and the
    // read-only handle; none takes a request, a registry or a report as a parameter.
    let mut implementors = Vec::new();
    for (name, source) in sources {
        for (_, line) in code_lines(source) {
            if line.contains("AdmissibleScorer for") {
                implementors.push((name, line.to_owned()));
            }
        }
    }
    assert_eq!(implementors.len(), 2, "{implementors:?}");
}
