//! L1 learning of the finding router: a Thompson-sampling bandit over the repair Pods of a code,
//! trained on verifier verdicts only, evaluated off-policy before it may be promoted by a host.

use ptr_router::{
    classify, AdmissibleScorer, EligibilitySnapshot, Exploration, FallbackPolicy, FindingCode,
    FindingRegistry, FindingRouteErrorKind, FindingRouteRecord, FindingRouteRequest, FindingRouter,
    LearnedPolicy, LearnerConfig, NonceSequence, OpeError, PolicyAction, PolicyBook,
    PolicyBookError, PolicyEvent, PromotionGate, PromotionRefusal, PromotionRule, RegistryChange,
    RegistryUpdate, ReportClass, RouteClass, RouteDistribution, RouteError, RouterBuildError,
    RouterConfig, RouterLimits, RoutingLearner, SamplingKey, ScoringContext, UniformScorer,
    Verdict, VerifierReward, MAX_EVENTS,
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
    .unwrap()
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

/// The verifier's verdict as the loop would hand it to the learner: `Accept` is built only by
/// `classify` from a clean passing report at the required level.
fn verdict(passed: bool) -> VerifierReward {
    if !passed {
        return VerifierReward::NOT_CLEARED;
    }
    let clean = VerificationReport {
        status: VerificationStatus::Pass,
        level: VerificationLevel::Deterministic,
        score: Probability::new(0.0).unwrap(),
        findings: Vec::new(),
    };
    let class = classify(
        &clean,
        VerificationLevel::Deterministic,
        &RouterLimits::default(),
    );
    VerifierReward::from_class(&class)
}

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
        learner.settle(&record, verdict(passed != invert));
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
fn trained_and_logged(seed: u64, eval: usize, invert: bool) -> (RoutingLearner, LearnedPolicy) {
    let learner = learner(seed);
    let uniform = router(&["bad", "good"], EPSILON, UniformScorer, 1);
    let mut world = World(seed.wrapping_mul(7919));
    collect(&learner, &uniform, &TRUTH, 400, "train", &mut world, invert);
    let policy = learner.snapshot();
    let fresh = router(&["bad", "good"], EPSILON, UniformScorer, 6);
    collect(&learner, &fresh, &TRUTH, eval, "eval", &mut world, false);
    (learner, policy)
}

fn gate() -> PromotionGate {
    PromotionGate::new(PromotionRule::new(0.05, 500, 100.0)).unwrap()
}

fn book() -> PolicyBook {
    PolicyBook::new("arith", Exploration::new(EPSILON).unwrap()).unwrap()
}

#[test]
fn a_clearly_better_policy_is_promoted_on_held_out_decisions_and_can_be_rolled_back() {
    let (learner, policy) = trained_and_logged(21, 1500, false);
    let log = learner.log_from(policy.cutoff());
    assert_eq!(log.len(), 1500);
    assert_eq!(learner.log_len_from(policy.cutoff()), 1500);
    assert!(log.iter().all(|d| d.seq() >= policy.cutoff()));
    let mut book = book();
    let incumbent = book.scorer();
    let assessment = gate().assess(&learner, &policy, &incumbent).unwrap();
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
    let mut sink: Vec<PolicyEvent> = Vec::new();
    let event = book
        .promote(policy.clone(), evidence.clone(), &mut sink)
        .unwrap();
    assert_eq!(event.action, PolicyAction::Promoted);
    assert_eq!(event.from, initial);
    assert_eq!(event.to, policy.digest());
    assert_eq!(event.evidence, Some(evidence.digest()));
    // The sink saw exactly the event the book returned: a promotion cannot skip the record.
    assert_eq!(sink, vec![event.clone()]);
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
        book.promote(policy.clone(), evidence, &mut sink)
            .unwrap_err(),
        PolicyBookError::StaleEvidence
    );
    assert_eq!(sink.len(), 1, "a refused promotion records nothing");
    // Rolled back, routing is uniform again and an event says so.
    let rolled = book.rollback(&mut sink).unwrap();
    assert_eq!(rolled.action, PolicyAction::RolledBack);
    assert_eq!(rolled.from, policy.digest());
    assert_eq!(rolled.to, initial);
    assert_eq!(sink.len(), 2);
    assert!(!book.is_learned());
    let good = (0..2000)
        .filter(|i| route_record(&live, &view, &format!("q{i}")).pod().0 == "good")
        .count();
    assert!((800..1200).contains(&good), "{good}");
    assert_eq!(
        book.rollback(&mut sink).unwrap_err(),
        PolicyBookError::NothingToRollBack
    );
    assert_eq!(book.events().len(), 2);
    assert_eq!(book.events()[1].sequence, 2);
}

#[test]
fn evidence_is_bound_to_its_candidate_its_incumbent_its_scope_and_its_exploration() {
    let (learner, policy) = trained_and_logged(31, 1500, false);
    let (_, other) = trained_and_logged(32, 10, false);
    let mut book = book();
    let mut sink: Vec<PolicyEvent> = Vec::new();
    let evidence = gate()
        .assess(&learner, &policy, &book.scorer())
        .unwrap()
        .evidence()
        .cloned()
        .expect("promotable");
    assert_eq!(
        book.promote(other, evidence.clone(), &mut sink)
            .unwrap_err(),
        PolicyBookError::EvidenceForAnotherCandidate
    );
    let mut elsewhere = PolicyBook::new("code", Exploration::new(EPSILON).unwrap()).unwrap();
    assert_eq!(
        elsewhere
            .promote(policy.clone(), evidence.clone(), &mut sink)
            .unwrap_err(),
        PolicyBookError::ScopeMismatch
    );
    // A book for another exploration rate refuses the candidate, and the gate refuses to
    // compare it with that book's handle at all.
    let mut other_floor = PolicyBook::new("arith", Exploration::new(0.2).unwrap()).unwrap();
    assert_eq!(
        other_floor
            .promote(policy.clone(), evidence.clone(), &mut sink)
            .unwrap_err(),
        PolicyBookError::ExplorationMismatch
    );
    assert_eq!(
        gate()
            .assess(&learner, &policy, &other_floor.scorer())
            .unwrap_err(),
        OpeError::ExplorationMismatch
    );
    // Evidence sealed against another incumbent is stale for this book: after a promotion and
    // a rollback the book is back at the uniform policy, so use the learned one as the stale
    // incumbent instead.
    book.promote(policy.clone(), evidence.clone(), &mut sink)
        .unwrap();
    assert_eq!(
        book.promote(policy.clone(), evidence, &mut sink)
            .unwrap_err(),
        PolicyBookError::StaleEvidence
    );
    assert_eq!(sink.len(), 1);
}

#[test]
fn a_candidate_can_only_be_assessed_on_the_log_of_the_learner_it_came_from() {
    let (learner_a, policy_a) = trained_and_logged(61, 1500, false);
    let (learner_b, _) = trained_and_logged(61, 1500, false);
    let book = book();
    assert_eq!(
        gate()
            .assess(&learner_b, &policy_a, &book.scorer())
            .unwrap_err(),
        OpeError::ForeignLearner
    );
    assert!(gate().assess(&learner_a, &policy_a, &book.scorer()).is_ok());
}

#[test]
fn a_log_in_which_nothing_was_cleared_cannot_yield_evidence() {
    // The only rewards a learner can be given are verifier verdicts. Settling every evaluation
    // decision as not cleared leaves the candidate no better than the incumbent.
    let learner = learner(71);
    let uniform = router(&["bad", "good"], EPSILON, UniformScorer, 1);
    let mut world = World(5);
    collect(&learner, &uniform, &TRUTH, 400, "train", &mut world, false);
    let policy = learner.snapshot();
    let view = eligible(&["bad", "good"]);
    for i in 0..1500 {
        let record = route_record(&uniform, &view, &format!("nothing-{i}"));
        learner.settle(&record, VerifierReward::NOT_CLEARED);
    }
    let book = book();
    let assessment = gate().assess(&learner, &policy, &book.scorer()).unwrap();
    assert!(
        assessment.evidence().is_none(),
        "{:?}",
        assessment.comparison
    );
}

#[test]
fn a_router_cannot_be_built_with_a_learned_scorer_under_another_exploration_floor() {
    let book = book();
    let build = |epsilon: f64| {
        FindingRouter::new(
            registry(&["bad", "good"]),
            config(epsilon),
            book.scorer(),
            SamplingKey::from_bytes([1; 32]),
            NonceSequence::starting_at([1; 8], 0),
        )
    };
    // No floor at all: refused.
    assert!(matches!(
        build(0.0),
        Err(RouterBuildError::ExplorationMismatch { .. })
    ));
    // Another floor: refused.
    assert!(matches!(
        build(0.2),
        Err(RouterBuildError::ExplorationMismatch { .. })
    ));
    // The book's own floor: accepted.
    assert!(build(EPSILON).is_ok());
    // A learned snapshot used directly is bound too.
    let snapshot = learner(3).snapshot();
    let direct = |epsilon: f64| {
        FindingRouter::new(
            registry(&["bad", "good"]),
            config(epsilon),
            snapshot.clone(),
            SamplingKey::from_bytes([1; 32]),
            NonceSequence::starting_at([1; 8], 0),
        )
    };
    assert!(direct(0.0).is_err());
    assert!(direct(0.3).is_err());
    assert!(direct(EPSILON).is_ok());
    // A book needs a positive floor of its own.
    assert_eq!(
        PolicyBook::new("arith", Exploration::NONE).unwrap_err(),
        PolicyBookError::NoFloor
    );
}

#[test]
fn rollback_still_works_after_more_events_than_the_history_holds() {
    let (learner, policy) = trained_and_logged(81, 1500, false);
    let mut book = book();
    let mut sink: Vec<PolicyEvent> = Vec::new();
    let evidence = gate()
        .assess(&learner, &policy, &book.scorer())
        .unwrap()
        .evidence()
        .cloned()
        .expect("promotable");
    let initial = book.active_digest();
    // The same evidence applies again whenever the book is back at the incumbent it names.
    let rounds = MAX_EVENTS / 2 + 40;
    for _ in 0..rounds {
        book.promote(policy.clone(), evidence.clone(), &mut sink)
            .unwrap();
        book.rollback(&mut sink).unwrap();
    }
    assert!(sink.len() > MAX_EVENTS);
    assert_eq!(book.events().len(), MAX_EVENTS);
    assert!(book.dropped_events() > 0);
    // One more promotion, then the rollback that must not be blocked.
    book.promote(policy.clone(), evidence, &mut sink).unwrap();
    assert!(book.is_learned());
    let rolled = book.rollback(&mut sink).unwrap();
    assert_eq!(rolled.to, initial);
    assert!(!book.is_learned());
    // Sequence numbers keep counting past the dropped events.
    assert_eq!(rolled.sequence, sink.len() as u64);
}

#[test]
fn the_gate_cannot_be_weakened_below_its_floors() {
    let weak = [
        PromotionRule::new(0.05, 2, 100.0),
        PromotionRule::new(0.05, 99, 100.0),
        PromotionRule::new(0.05, 500, 1.0),
        PromotionRule::new(0.05, 500, 29.0),
        PromotionRule::new(-0.01, 500, 100.0),
        PromotionRule {
            confidence: ptr_router::Confidence::from_quantile(1.0).unwrap(),
            ..PromotionRule::new(0.05, 500, 100.0)
        },
    ];
    for rule in weak {
        assert!(matches!(PromotionGate::new(rule), Err(OpeError::Config(_))));
    }
    // The floors themselves are accepted.
    assert!(PromotionGate::new(PromotionRule::new(0.0, 100, 30.0)).is_ok());
}

#[test]
fn promotion_is_refused_when_the_lower_bound_does_not_beat_the_incumbent() {
    // A policy that learned the wrong way round is worse than the incumbent.
    let (learner, worse) = trained_and_logged(41, 1500, true);
    let book = book();
    let assessment = gate().assess(&learner, &worse, &book.scorer()).unwrap();
    assert!(
        assessment.comparison.difference < 0.0,
        "{:?}",
        assessment.comparison
    );
    assert!(matches!(
        assessment.verdict,
        Verdict::Refuse(PromotionRefusal::LowerBoundNotAboveMargin { .. })
    ));

    // A policy trained on nothing is the prior: its value is the incumbent's, within noise. It
    // has to come from the learner whose log is used, so snapshot a learner before any traffic.
    let learner = self::learner(43);
    let untrained = learner.snapshot();
    let uniform = router(&["bad", "good"], EPSILON, UniformScorer, 6);
    let mut world = World(9);
    collect(&learner, &uniform, &TRUTH, 1500, "eval", &mut world, false);
    let assessment = gate().assess(&learner, &untrained, &book.scorer()).unwrap();
    assert!(
        assessment.evidence().is_none(),
        "{:?}",
        assessment.comparison
    );

    // A real improvement still fails a margin it cannot reach.
    let (learner, policy) = trained_and_logged(44, 1500, false);
    let strict = PromotionGate::new(PromotionRule::new(0.6, 500, 100.0)).unwrap();
    let assessment = strict.assess(&learner, &policy, &book.scorer()).unwrap();
    assert!(matches!(
        assessment.verdict,
        Verdict::Refuse(PromotionRefusal::LowerBoundNotAboveMargin { .. })
    ));
    // And too short a log, whatever it shows.
    let (short_learner, short) = trained_and_logged(45, 100, false);
    let assessment = gate()
        .assess(&short_learner, &short, &book.scorer())
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
    // The gate takes the log from the learner starting at the candidate's cutoff, so a stale
    // snapshot (taken later than the decisions) sees only what came after it; one taken before
    // any settling sees everything, and that is exactly the held-out log.
    let (learner, policy) = trained_and_logged(51, 800, false);
    assert_eq!(learner.log_len_from(policy.cutoff()), 800);
    assert_eq!(learner.log_len_from(0), 1200);
    assert!(learner
        .log_from(policy.cutoff())
        .iter()
        .all(|d| d.seq() >= policy.cutoff()));
    let book = book();
    let assessment = gate().assess(&learner, &policy, &book.scorer()).unwrap();
    assert_eq!(assessment.comparison.target.n, 800);
}

// --- authority helpers ---------------------------------------------------------------------

/// A target policy that prefers its weights as given.
struct Fixed(Vec<(&'static str, f64)>);

impl AdmissibleScorer for Fixed {
    fn bound_exploration(&self) -> Option<ptr_router::Exploration> {
        None
    }

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

// --- authority --------------------------------------------------------------------------------

#[test]
fn a_scorer_cannot_choose_an_ineligible_pod_or_remove_the_floor() {
    // A scorer that names a pod the eligibility view does not list is refused.
    struct Everyone;
    impl AdmissibleScorer for Everyone {
        fn bound_exploration(&self) -> Option<ptr_router::Exploration> {
            None
        }

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
        fn bound_exploration(&self) -> Option<ptr_router::Exploration> {
            None
        }

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
