use ptr_router::{
    AdmissibleScorer, AttentionSchema, BalanceStrength, CodeRoute, ControlArm, ControlConfig,
    ControlledScorer, EligibilitySnapshot, EscalationReason, Exploration, FallbackPolicy,
    FallbackReason, FindingCode, FindingRegistry, FindingRouteErrorKind, FindingRouteRecord,
    FindingRouteRequest, FindingRouter, HintPolicy, NonceSequence, PodEligibility, PodRef,
    RegistryChange, RegistryUpdate, RouteError, RouteOutcome, RouterConfig, RouterLimits,
    SamplingKey, SchemaError, SchemaLimits, ScoringContext, SelfContext, SelfModel, SelfModelHint,
    SelfModelLimits, SelfOutcome, SelfRole, SelfTraceStep, TargetPower, UniformScorer,
};
use ptr_types::{Digest, Generation, PodId, Probability, RequestId, VerificationLevel};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};
use std::cell::RefCell;
use std::sync::Arc;

const STATUSES: [VerificationStatus; 4] = [
    VerificationStatus::Pass,
    VerificationStatus::Fail,
    VerificationStatus::Disputed,
    VerificationStatus::Unknown,
];
const LEVELS: [VerificationLevel; 5] = [
    VerificationLevel::Unverified,
    VerificationLevel::LatentAgreement,
    VerificationLevel::SampleVerified,
    VerificationLevel::FullSemantic,
    VerificationLevel::Deterministic,
];
const ARMS: [ControlArm; 6] = [
    ControlArm::InnerOnly,
    ControlArm::ObserveOnly,
    ControlArm::BalanceOnly,
    ControlArm::HintOnly,
    ControlArm::Full,
    ControlArm::FullMismatched,
];

fn pod(id: &str) -> PodId {
    PodId::from(id)
}

fn report(
    status: VerificationStatus,
    level: VerificationLevel,
    findings: &[(&str, bool)],
) -> VerificationReport {
    VerificationReport {
        status,
        level,
        score: Probability::new(0.5).unwrap(),
        findings: findings
            .iter()
            .map(|(code, hard)| Finding {
                code: (*code).into(),
                message: String::new(),
                hard: *hard,
            })
            .collect(),
    }
}

fn digest(i: u32) -> Digest {
    let mut d = [0u8; 32];
    d[..4].copy_from_slice(&i.to_be_bytes());
    d
}

fn registry(repair: &[(&str, &str)], proposers: &[&str]) -> FindingRegistry {
    let mut registry = FindingRegistry::new();
    for (code, id) in repair {
        let expected = registry.generation();
        registry
            .apply(RegistryUpdate {
                expected,
                change: RegistryChange::AddRepair {
                    code: FindingCode::new(*code),
                    pod: pod(id),
                },
            })
            .unwrap();
    }
    for id in proposers {
        let expected = registry.generation();
        registry
            .apply(RegistryUpdate {
                expected,
                change: RegistryChange::AddProposer { pod: pod(id) },
            })
            .unwrap();
    }
    registry
}

fn router<S: AdmissibleScorer>(epsilon: f64, scorer: S) -> FindingRouter<S> {
    FindingRouter::new(
        registry(
            &[
                ("target", "a"),
                ("target", "b"),
                ("target", "c"),
                ("effect", "a"),
                ("effect", "b"),
            ],
            &["gen1", "gen2"],
        ),
        RouterConfig {
            fallback: FallbackPolicy::default(),
            limits: RouterLimits::default(),
            exploration: Exploration::new(epsilon).unwrap(),
        },
        scorer,
        SamplingKey::from_bytes([9; 32]),
        NonceSequence::starting_at([3; 8], 0),
    )
    .unwrap()
}

/// Every Pod is eligible at one generation.
struct Gen(u64);
impl PodEligibility for Gen {
    fn current_generation(&self, _: &PodId) -> Option<Generation> {
        Some(Generation(self.0))
    }
}

fn route_with(
    router: &FindingRouter<impl AdmissibleScorer>,
    request: &str,
    i: u32,
    report: &VerificationReport,
    eligibility: &dyn PodEligibility,
) -> Result<RouteOutcome, ptr_router::FindingRouteError> {
    let id = RequestId::from(request);
    router.route(&FindingRouteRequest {
        request_id: &id,
        request_digest: digest(i),
        report,
        required: VerificationLevel::SampleVerified,
        eligibility,
    })
}

fn route(
    router: &FindingRouter<impl AdmissibleScorer>,
    request: &str,
    i: u32,
    report: &VerificationReport,
) -> RouteOutcome {
    route_with(router, request, i, report, &Gen(1)).unwrap()
}

fn target_failure() -> VerificationReport {
    report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    )
}

fn target_record(
    router: &FindingRouter<impl AdmissibleScorer>,
    tag: &str,
    i: u32,
) -> FindingRouteRecord {
    route(router, &format!("{tag}-{i}"), i, &target_failure()).records()[0].clone()
}

fn shape(outcome: &RouteOutcome) -> String {
    match outcome {
        RouteOutcome::Accept(_) => "accept".into(),
        RouteOutcome::Repair { routes } => {
            let parts: Vec<String> = routes
                .iter()
                .map(|r| match r {
                    CodeRoute::Routed(rec) => format!("routed:{:?}", rec.class()),
                    CodeRoute::Unroutable(code) => format!("unroutable:{code}"),
                })
                .collect();
            format!("repair[{}]", parts.join(","))
        }
        RouteOutcome::Regenerate { reason, proposer } => {
            format!("regenerate:{reason:?}:{:?}", proposer.class())
        }
        RouteOutcome::Escalate { reason } => match reason {
            EscalationReason::Policy(r) => format!("escalate-policy:{r:?}"),
            EscalationReason::NoProposer(r) => format!("escalate-noproposer:{r:?}"),
        },
    }
}

/// Seeded splitmix64.
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

fn new_schema(limits: SchemaLimits) -> Arc<AttentionSchema> {
    Arc::new(AttentionSchema::new(limits).unwrap())
}

fn controlled<S: AdmissibleScorer>(
    inner: S,
    schema: &Arc<AttentionSchema>,
    arm: ControlArm,
) -> ControlledScorer<S> {
    ControlledScorer::new(inner, Arc::clone(schema), arm, ControlConfig::default())
}

/// A scorer with fixed weights, standing in for a learned inner router policy.
#[derive(Debug)]
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
                    .find(|(n, _)| pod(n) == *p)
                    .map_or(1.0, |(_, w)| *w);
                (p.clone(), w)
            })
            .collect())
    }
}

type Seen = RefCell<Vec<(Vec<PodId>, Vec<(PodId, f64)>)>>;
struct Recording<'a, S>(&'a S, &'a Seen);
impl<S: AdmissibleScorer> AdmissibleScorer for Recording<'_, S> {
    fn bound_exploration(&self) -> Option<ptr_router::Exploration> {
        self.0.bound_exploration()
    }

    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        let weights = self.0.weights(context)?;
        self.1
            .borrow_mut()
            .push((context.admissible.to_vec(), weights.clone()));
        Ok(weights)
    }
}

/// The weights the scorer returned for the last scored class of one routing with only the
/// listed Pods eligible (at generation 1). `ScoringContext` cannot be built outside the crate, so
/// this goes through a real router.
fn weights_seen<S: AdmissibleScorer>(
    scorer: &S,
    report: &VerificationReport,
    eligible: &[&str],
) -> Vec<(PodId, f64)> {
    let seen = Seen::default();
    let router = router(0.0, Recording(scorer, &seen));
    let mut snapshot = EligibilitySnapshot::new();
    for name in eligible {
        snapshot.set(pod(name), Generation(1));
    }
    route_with(&router, "w", 0, report, &snapshot).unwrap();
    let seen = seen.borrow();
    seen.last().expect("the scorer was consulted").1.clone()
}

fn weight(weights: &[(PodId, f64)], name: &str) -> f64 {
    weights.iter().find(|(p, _)| *p == pod(name)).unwrap().1
}

/// Observes and settles `n` target repairs routed by `router`; `passes` decides the outcome.
fn train(
    schema: &AttentionSchema,
    router: &FindingRouter<impl AdmissibleScorer>,
    tag: &str,
    range: std::ops::Range<u32>,
    passes: impl Fn(&FindingRouteRecord, u32) -> bool,
) {
    for i in range {
        let record = target_record(router, tag, i);
        schema.observe(&record);
        schema.settle(&record, passes(&record, i));
    }
}

fn valid_hint(code: &str, name: &str) -> SelfModelHint {
    SelfModelHint::PreferRepairPod {
        code: FindingCode::new(code),
        pod: pod(name),
        generation: Generation(1),
        value_lcb: 0.9,
        runner_up_ucb: 0.1,
        evidence: 50,
    }
}

/// A schema trained on a skewed router, with outcomes where "a" passes, plus a hint for "b".
fn trained() -> Arc<AttentionSchema> {
    let schema = new_schema(SchemaLimits::default());
    let skewed = router(0.0, Fixed(vec![("a", 10.0), ("b", 1.0), ("c", 1.0)]));
    train(&schema, &skewed, "train", 0..300, |r, _| {
        *r.pod() == pod("a")
    });
    assert_eq!(schema.apply_hints(&[valid_hint("target", "b")]), 1);
    schema
}

#[test]
fn class_decisions_are_unchanged_for_every_arm_status_level_and_finding() {
    let schema = trained();
    let uniform = router(0.05, UniformScorer);
    let config = ControlConfig::default();
    let seen: Vec<Seen> = ARMS.iter().map(|_| Seen::default()).collect();
    let controlled: Vec<ControlledScorer<UniformScorer>> = ARMS
        .iter()
        .map(|arm| controlled(UniformScorer, &schema, *arm))
        .collect();
    let routers: Vec<_> = controlled
        .iter()
        .zip(&seen)
        .map(|(scorer, seen)| router(0.05, Recording(scorer, seen)))
        .collect();
    let mut combos = 0;
    for status in STATUSES {
        for level in LEVELS {
            for findings in [
                vec![],
                vec![("style", false)],
                vec![("target", true)],
                vec![("target", true), ("effect", true), ("target", true)],
                vec![("unregistered", true)],
                vec![("target", false), ("effect", true)],
            ] {
                let r = report(status, level, &findings);
                let i = combos;
                combos += 1;
                let id = format!("m-{i}");
                let expected = shape(&route(&uniform, &id, i, &r));
                for (arm, router) in ARMS.iter().zip(&routers) {
                    assert_eq!(
                        shape(&route(router, &id, i, &r)),
                        expected,
                        "{arm:?} {status:?} {level:?} {findings:?}"
                    );
                }
            }
        }
    }
    assert_eq!(combos, 120);
    // Every correction stayed inside [1/R, R] of the inner weight (1.0 here), whatever the arm.
    let ratio = config.max_ratio();
    let mut consulted = 0;
    for (arm, seen) in ARMS.iter().zip(&seen) {
        for (admissible, weights) in seen.borrow().iter() {
            consulted += 1;
            assert_eq!(weights.len(), admissible.len());
            for (p, w) in weights {
                assert!(admissible.contains(p));
                assert!(
                    w.is_finite() && *w >= 1.0 / ratio - 1e-12 && *w <= ratio + 1e-12,
                    "{arm:?} {w}"
                );
            }
        }
    }
    assert!(consulted > 0);
    // Accept is only reachable through the verifier, whatever the schema believes.
    let accept = report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        &[],
    );
    assert!(route(&routers[4], "acc", 1, &accept).is_accept());
    let forged = report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    assert!(matches!(
        route(&routers[4], "forged", 2, &forged),
        RouteOutcome::Escalate {
            reason: EscalationReason::Policy(FallbackReason::ForgedPass)
        }
    ));
}

#[derive(Debug)]
struct Rogue;
impl AdmissibleScorer for Rogue {
    fn bound_exploration(&self) -> Option<ptr_router::Exploration> {
        None
    }

    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        let mut weights: Vec<(PodId, f64)> = context
            .admissible
            .iter()
            .map(|p| (p.clone(), 1.0))
            .collect();
        weights.push((pod("not-admissible"), 1000.0));
        Ok(weights)
    }
}

#[test]
fn the_wrapper_never_repairs_or_adds_pods_and_the_router_still_refuses() {
    let schema = trained();
    for arm in ARMS {
        let router = router(0.1, controlled(Rogue, &schema, arm));
        let err = route_with(&router, "r", 1, &target_failure(), &Gen(1)).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                FindingRouteErrorKind::OutsideAdmissibleSet { .. }
            ),
            "{arm:?}"
        );
    }
    // A zero inner weight stays zero: the schema cannot revive a Pod the inner policy dropped.
    let inner = Fixed(vec![("a", 0.0), ("b", 1.0), ("c", 1.0)]);
    let controlled_full = controlled(inner, &schema, ControlArm::Full);
    let w = weights_seen(&controlled_full, &target_failure(), &["a", "b", "c"]);
    assert_eq!(weight(&w, "a"), 0.0);
}

/// A self-model whose beliefs and hints are as optimistic as they can be, applied to a schema in
/// the strongest arm, cannot change a class.
#[test]
fn self_model_hints_flowing_into_the_schema_cannot_change_a_class() {
    let mut model = SelfModel::new(SelfModelLimits::default());
    let mut rng = Rng(5);
    for _ in 0..400 {
        let good = rng.next().is_multiple_of(2);
        let routed = PodRef::new(pod(if good { "a" } else { "b" }), Generation(1));
        let passed = rng.unit() < if good { 0.95 } else { 0.05 };
        model.observe(&SelfTraceStep {
            context: SelfContext {
                role: SelfRole::Repair,
                producer: None,
                round: 1,
                budget_left: 3,
                required: VerificationLevel::SampleVerified,
                last_codes: vec![FindingCode::new("target")],
            },
            outcome: SelfOutcome {
                passed,
                primary_code: None,
                routed_pod: Some(routed),
                propensity: Some(ptr_router::Propensity::new(0.5).unwrap()),
            },
        });
    }
    let context = SelfContext {
        role: SelfRole::Repair,
        producer: None,
        round: 1,
        budget_left: 3,
        required: VerificationLevel::SampleVerified,
        last_codes: vec![FindingCode::new("target")],
    };
    let hints = model.hints(&context, &HintPolicy::default());
    assert!(hints.iter().any(
        |h| matches!(h, SelfModelHint::PreferRepairPod { pod, .. } if *pod == self::pod("a"))
    ));
    let schema = new_schema(SchemaLimits::default());
    let seed = router(0.1, UniformScorer);
    train(&schema, &seed, "seed", 0..50, |_, _| true);
    assert!(schema.apply_hints(&hints) >= 1);
    let full = router(0.05, controlled(UniformScorer, &schema, ControlArm::Full));
    let plain = router(0.05, UniformScorer);
    let forged = report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    for i in 0..50 {
        let id = format!("h-{i}");
        let a = route(&plain, &id, i, &forged);
        let b = route(&full, &id, i, &forged);
        assert_eq!(shape(&a), shape(&b));
        assert!(!b.is_accept() && !b.is_repair());
    }
}

#[test]
fn the_exploration_floor_still_holds_with_the_full_arm_under_maximum_boost() {
    let epsilon = 0.2;
    let config = ControlConfig::new(
        BalanceStrength::Two,
        64.0,
        64.0,
        0.0,
        TargetPower::Four,
        1,
        1,
    )
    .unwrap();
    let schema = new_schema(SchemaLimits::default());
    let skewed = router(0.0, Fixed(vec![("a", 50.0), ("b", 1.0), ("c", 1.0)]));
    train(&schema, &skewed, "floor-train", 0..400, |r, _| {
        *r.pod() == pod("a")
    });
    schema.apply_hints(&[valid_hint("target", "a")]);
    let scorer =
        ControlledScorer::new(UniformScorer, Arc::clone(&schema), ControlArm::Full, config);
    let router = router(epsilon, scorer);
    let mut counts = [0u32; 3];
    for i in 0..3000 {
        let record = target_record(&router, "floor", i);
        assert_eq!(record.support().len(), 3);
        assert!(record.propensity().get() >= epsilon / 3.0 - 1e-9);
        for (_, width) in record.support() {
            assert!(*width as f64 / 9_007_199_254_740_992.0 >= epsilon / 3.0 - 1e-9);
        }
        let index = ["a", "b", "c"]
            .iter()
            .position(|n| pod(n) == *record.pod())
            .unwrap();
        counts[index] += 1;
    }
    assert!(counts.iter().all(|c| *c > 0), "{counts:?}");
}

#[test]
fn inner_only_and_observe_only_routing_equals_the_inner_scorer_exactly() {
    let schema = trained();
    for arm in [ControlArm::InnerOnly, ControlArm::ObserveOnly] {
        let inner = || Fixed(vec![("a", 3.0), ("b", 1.0), ("c", 2.0)]);
        let plain = router(0.1, inner());
        let wrapped = router(0.1, controlled(inner(), &schema, arm));
        for i in 0..500 {
            let a = target_record(&plain, "same", i);
            let b = target_record(&wrapped, "same", i);
            assert_eq!(a, b, "{arm:?}");
            schema.observe(&b);
        }
        let below = report(VerificationStatus::Pass, VerificationLevel::Unverified, &[]);
        for i in 0..50 {
            let id = format!("g-{i}");
            assert_eq!(
                route(&plain, &id, i, &below),
                route(&wrapped, &id, i, &below)
            );
        }
    }
}

/// The reference hit computed outside the schema, before it learns.
fn external_expected(schema: &AttentionSchema, record: &FindingRouteRecord) -> f64 {
    let pods: Vec<PodId> = record.support().iter().map(|(p, _)| p.clone()).collect();
    let forecast = schema.forecast(record.class(), &pods);
    let mut best: Option<(&PodId, f64)> = None;
    for (p, q) in &forecast {
        if best.is_none_or(|(_, b)| *q > b) {
            best = Some((p, *q));
        }
    }
    let top = best.unwrap().0;
    let width = record.support().iter().find(|(p, _)| p == top).unwrap().1;
    width as f64 / 9_007_199_254_740_992.0
}

#[test]
fn schema_accuracy_is_the_expected_hit_rate_and_follows_the_stationary_process_and_a_shift() {
    let schema = new_schema(SchemaLimits {
        window: 8,
        ..SchemaLimits::default()
    });
    // The inner policy is fixed and external, so the routing process is stationary and does not
    // depend on the schema; "c" is heavily preferred (the schema's tie-break names "a").
    let before = router(0.05, Fixed(vec![("a", 1.0), ("b", 1.0), ("c", 20.0)]));
    let mut external = 0.0;
    let mut first = None;
    let mut late = Vec::new();
    for i in 0..2000 {
        let record = target_record(&before, "stationary", i);
        let expected = external_expected(&schema, &record);
        external += expected;
        if i == 0 {
            first = Some(expected);
        }
        if i >= 1500 {
            late.push(expected);
        }
        schema.observe(&record);
    }
    // The order is pinned exactly: scoring after learning could not reproduce these sums.
    let report = schema.report();
    assert_eq!(report.observations, 2000);
    assert!((report.expected_accuracy.unwrap() * 2000.0 - external).abs() < 1e-9);
    let first = first.unwrap();
    let late = late.iter().sum::<f64>() / late.len() as f64;
    assert!(first < 0.1, "cold start names the rarest Pod: {first}");
    assert!(late > 0.8, "{late}");
    // Chance and oracle levels are reported, and the skill is high on a stationary process.
    assert!(report.chance_accuracy.unwrap() > 0.3 && report.chance_accuracy.unwrap() < 0.4);
    assert!(report.oracle_accuracy.unwrap() > report.expected_accuracy.unwrap() - 1e-9);
    assert!(report.skill.unwrap() > 0.9, "{:?}", report.skill);
    assert!(report.mean_total_variation.unwrap() < 0.05);
    let stationary = report.windowed_expected_accuracy.unwrap();
    assert!(stationary > 0.8, "{stationary}");
    assert!(report.attention_entropy.unwrap() < 3f64.ln() * 0.6);

    // Distribution shift: "a" becomes the preferred Pod; the forecast lags for a few records.
    let after = router(0.05, Fixed(vec![("a", 20.0), ("b", 1.0), ("c", 1.0)]));
    for i in 0..8 {
        schema.observe(&target_record(&after, "shifted", i));
    }
    let shifted = schema.report().windowed_expected_accuracy.unwrap();
    assert!(
        shifted < stationary - 0.5,
        "stationary {stationary} shifted {shifted}"
    );
    for i in 8..200 {
        schema.observe(&target_record(&after, "shifted", i));
    }
    let recovered = schema.report().windowed_expected_accuracy.unwrap();
    assert!(
        recovered > shifted + 0.5,
        "shifted {shifted} recovered {recovered}"
    );
}

#[test]
fn closed_loop_accuracy_is_judged_against_chance_not_against_raw_hits() {
    let schema = new_schema(SchemaLimits::default());
    let full = router(0.1, controlled(UniformScorer, &schema, ControlArm::Full));
    train(&schema, &full, "loop", 0..3000, |r, i| {
        let rate = if *r.pod() == pod("a") { 0.8 } else { 0.3 };
        Rng(u64::from(i) ^ 0xABCD).unit() < rate
    });
    let report = schema.report();
    let chance = report.chance_accuracy.unwrap();
    let expected = report.expected_accuracy.unwrap();
    let oracle = report.oracle_accuracy.unwrap();
    // The router steers towards the verifier-grounded target, so the composite is not uniform
    // (a is favoured), the forecast tracks it, and it is reported relative to 1/n.
    assert!(
        oracle > chance + 0.05,
        "the controlled composite is not uniform: {oracle} {chance}"
    );
    assert!(expected > chance, "{expected} {chance}");
    assert!(expected <= oracle + 1e-9);
    assert!(report.skill.unwrap() > 0.5, "{:?}", report.skill);
    assert!(report.raw_hits > 0);
}

#[test]
fn control_reweights_by_the_verifier_grounded_target_and_stays_bounded() {
    let schema = new_schema(SchemaLimits::default());
    let uniform = router(0.1, UniformScorer);
    // Only "a" passes; the router is uniform, so the forecast is flat and the target is not.
    train(&schema, &uniform, "target", 0..600, |r, _| {
        *r.pod() == pod("a")
    });
    let config = ControlConfig::default();
    let failure = target_failure();
    let full = controlled(UniformScorer, &schema, ControlArm::Full);
    let w = weights_seen(&full, &failure, &["a", "b", "c"]);
    assert!(
        weight(&w, "a") > weight(&w, "b") && weight(&w, "a") > weight(&w, "c"),
        "{w:?}"
    );
    assert!(w
        .iter()
        .all(|(_, x)| *x >= 1.0 / config.max_ratio() && *x <= config.max_ratio()));
    // The balance-only arm has a uniform target: with a flat forecast it is near-neutral.
    let balance = controlled(UniformScorer, &schema, ControlArm::BalanceOnly);
    let wb = weights_seen(&balance, &failure, &["a", "b", "c"]);
    assert!(
        (weight(&wb, "a") / weight(&wb, "b") - 1.0).abs() < 0.3,
        "{wb:?}"
    );
    // Restricting the eligible set restricts the weights.
    let w = weights_seen(&full, &failure, &["a", "c"]);
    assert_eq!(w.len(), 2);
    assert!(w.iter().all(|(p, _)| *p != pod("b")));
    // A Pod the schema has never seen is not flooded: its correction is exactly neutral.
    let fresh = new_schema(SchemaLimits::default());
    let two_only = router(0.0, Fixed(vec![("a", 1.0), ("b", 1.0), ("c", 0.0)]));
    train(&fresh, &two_only, "newcomer", 0..300, |r, _| {
        *r.pod() == pod("a")
    });
    let scorer = controlled(UniformScorer, &fresh, ControlArm::Full);
    let w = weights_seen(&scorer, &failure, &["a", "b", "c"]);
    assert_eq!(weight(&w, "c"), 1.0, "{w:?}");
    // Regeneration classes take no repair preference.
    let below = report(VerificationStatus::Pass, VerificationLevel::Unverified, &[]);
    let w = weights_seen(
        &controlled(UniformScorer, &schema, ControlArm::HintOnly),
        &below,
        &["gen1", "gen2"],
    );
    assert_eq!(w, vec![(pod("gen1"), 1.0), (pod("gen2"), 1.0)]);
}

#[test]
fn the_mismatched_forecast_arm_is_a_distinct_falsifier_of_the_same_rule() {
    let schema = new_schema(SchemaLimits::default());
    let skewed = router(0.0, Fixed(vec![("a", 20.0), ("b", 2.0), ("c", 1.0)]));
    for i in 0..400 {
        schema.observe(&target_record(&skewed, "mis", i));
    }
    let failure = target_failure();
    let balance = weights_seen(
        &controlled(UniformScorer, &schema, ControlArm::BalanceOnly),
        &failure,
        &["a", "b", "c"],
    );
    let mismatched = weights_seen(
        &controlled(UniformScorer, &schema, ControlArm::FullMismatched),
        &failure,
        &["a", "b", "c"],
    );
    // The aligned forecast damps the dominant Pod; rotating it hands that Pod another Pod's
    // small forecast and so a larger weight.
    assert!(weight(&balance, "a") < weight(&balance, "c"), "{balance:?}");
    assert!(
        weight(&mismatched, "a") > weight(&balance, "a"),
        "{mismatched:?}"
    );
    assert_ne!(balance, mismatched);
}

#[test]
fn hint_only_boosts_exactly_the_hinted_pod_inside_the_ratio() {
    let schema = trained();
    let w = weights_seen(
        &controlled(UniformScorer, &schema, ControlArm::HintOnly),
        &target_failure(),
        &["a", "b", "c"],
    );
    assert_eq!(weight(&w, "b"), ControlConfig::default().hint_boost());
    assert_eq!(weight(&w, "a"), 1.0);
    assert_eq!(weight(&w, "c"), 1.0);
    // Hints do nothing in the arms that do not use them.
    for arm in [
        ControlArm::InnerOnly,
        ControlArm::ObserveOnly,
        ControlArm::BalanceOnly,
    ] {
        let w = weights_seen(
            &controlled(UniformScorer, &schema, arm),
            &target_failure(),
            &["a", "b", "c"],
        );
        assert!(
            w.iter()
                .all(|(p, x)| *p == pod("b") && *x != 0.0 || *x <= 8.0),
            "{arm:?}"
        );
        if arm != ControlArm::BalanceOnly {
            assert!(w.iter().all(|(_, x)| *x == 1.0), "{arm:?}");
        }
    }
}

#[test]
fn forged_oversized_and_unfounded_hints_are_rejected_and_hints_are_replaced_wholesale() {
    let schema = new_schema(SchemaLimits::default());
    let long = "cbbcbbacabababaacaaabbacacaacbbacaabacbbabbbabbcbbaacaaabcacaaacbbbaaccccabbabbaaaacbbbaabaabaaabbaa".repeat(10_000);
    let bad = |code: &str, name: &str, lcb: f64, ucb: f64, evidence: u64| {
        SelfModelHint::PreferRepairPod {
            code: FindingCode::new(code),
            pod: pod(name),
            generation: Generation(1),
            value_lcb: lcb,
            runner_up_ucb: ucb,
            evidence,
        }
    };
    let forged = [
        bad(&long, "a", 0.9, 0.1, 50),
        bad("target", &long, 0.9, 0.1, 50),
        bad("target", "a", 0.9, 0.1, 0),
        bad("target", "a", f64::NAN, 0.1, 50),
        bad("target", "a", 0.9, f64::INFINITY, 50),
        bad("target", "a", 1.5, 0.1, 50),
        bad("target", "a", 0.3, 0.3, 50),
        bad("target", "a", 0.2, 0.6, 50),
        SelfModelHint::EscalateEarly {
            predicted_pass: 0.0,
            threshold: 0.2,
            evidence: 99,
        },
    ];
    assert_eq!(schema.apply_hints(&forged), 0);
    assert_eq!(schema.report().preferences, 0);
    // Valid hints are stored; a repeated code keeps the larger lower bound.
    let mut stronger = valid_hint("target", "c");
    if let SelfModelHint::PreferRepairPod { value_lcb, .. } = &mut stronger {
        *value_lcb = 0.95;
    }
    assert_eq!(
        schema.apply_hints(&[
            valid_hint("target", "b"),
            stronger,
            valid_hint("effect", "a")
        ]),
        2
    );
    // Wholesale replacement: the new set is the whole set, and an empty set clears.
    assert_eq!(schema.apply_hints(&[valid_hint("effect", "b")]), 1);
    assert_eq!(schema.report().preferences, 1);
    assert_eq!(schema.apply_hints(&[]), 0);
    assert_eq!(schema.report().preferences, 0);
    // The table is capped.
    let capped = new_schema(SchemaLimits {
        max_preferences: 2,
        ..SchemaLimits::default()
    });
    let many: Vec<SelfModelHint> = (0..10)
        .map(|i| valid_hint(&format!("code-{i}"), "a"))
        .collect();
    assert_eq!(capped.apply_hints(&many), 2);
}

#[test]
fn preferences_expire_and_do_not_survive_a_new_pod_generation() {
    let config = ControlConfig::new(
        BalanceStrength::Off,
        4.0,
        8.0,
        0.25,
        TargetPower::Two,
        30,
        1,
    )
    .unwrap();
    let schema = new_schema(SchemaLimits {
        preference_ttl: 20,
        ..SchemaLimits::default()
    });
    let uniform = router(0.1, UniformScorer);
    train(&schema, &uniform, "ttl", 0..60, |_, _| true);
    schema.apply_hints(&[valid_hint("target", "b")]);
    let scorer = ControlledScorer::new(
        UniformScorer,
        Arc::clone(&schema),
        ControlArm::HintOnly,
        config,
    );
    let w = weights_seen(&scorer, &target_failure(), &["a", "b", "c"]);
    assert_eq!(weight(&w, "b"), 4.0);
    // Time to live: 21 more observed records and the stale preference is inert.
    for i in 60..81 {
        schema.observe(&target_record(&uniform, "ttl", i));
    }
    let w = weights_seen(&scorer, &target_failure(), &["a", "b", "c"]);
    assert_eq!(weight(&w, "b"), 1.0);
    // Generation safety: a preference for generation 1 never applies after the Pod is seen at
    // generation 2, and the Pod's statistics are dropped with it.
    schema.apply_hints(&[valid_hint("target", "b")]);
    let w = weights_seen(&scorer, &target_failure(), &["a", "b", "c"]);
    assert_eq!(weight(&w, "b"), 4.0);
    let mut moved = None;
    for i in 0..200 {
        let outcome = route_with(
            &uniform,
            &format!("gen-{i}"),
            1000 + i,
            &target_failure(),
            &Gen(2),
        )
        .unwrap();
        let record = outcome.records()[0].clone();
        if *record.pod() == pod("b") {
            moved = Some(record);
            break;
        }
    }
    schema.observe(&moved.expect("b is sampled eventually"));
    let w = weights_seen(&scorer, &target_failure(), &["a", "b", "c"]);
    assert_eq!(
        weight(&w, "b"),
        1.0,
        "the stale generation's hint is inert: {w:?}"
    );
}

/// The loop the review warned about: a hint boosts a Pod, the Pod gets more traffic, and raw
/// evidence for it would grow. Both Pods pass at the same rate; the inverse-propensity estimate
/// of the boosted Pod must not be inflated, and no preference hint may follow from it.
#[test]
fn a_hint_cannot_reinforce_itself_through_the_traffic_it_causes() {
    let schema = new_schema(SchemaLimits {
        preference_ttl: 1_000_000,
        ..SchemaLimits::default()
    });
    let config = ControlConfig::new(
        BalanceStrength::Off,
        8.0,
        8.0,
        0.25,
        TargetPower::Two,
        30,
        1,
    )
    .unwrap();
    let boosted = ControlledScorer::new(
        UniformScorer,
        Arc::clone(&schema),
        ControlArm::HintOnly,
        config,
    );
    let two = FindingRouter::new(
        registry(&[("target", "a"), ("target", "b")], &[]),
        RouterConfig {
            fallback: FallbackPolicy::default(),
            limits: RouterLimits::default(),
            exploration: Exploration::new(0.1).unwrap(),
        },
        boosted,
        SamplingKey::from_bytes([2; 32]),
        NonceSequence::starting_at([4; 8], 0),
    )
    .unwrap();
    // A confident-looking but baseless hint, as a naive self-model could produce it.
    let seed_router = router(0.1, UniformScorer);
    train(&schema, &seed_router, "warm", 0..80, |_, _| true);
    assert_eq!(schema.apply_hints(&[valid_hint("target", "a")]), 1);
    let mut model = SelfModel::new(SelfModelLimits::default());
    let mut rng = Rng(77);
    let (mut a_count, mut b_count) = (0u32, 0u32);
    let context = SelfContext {
        role: SelfRole::Repair,
        producer: None,
        round: 1,
        budget_left: 3,
        required: VerificationLevel::SampleVerified,
        last_codes: vec![FindingCode::new("target")],
    };
    for i in 0..6000 {
        let record = target_record(&two, "loop", i);
        let passed = rng.unit() < 0.5;
        schema.observe(&record);
        schema.settle(&record, passed);
        if *record.pod() == pod("a") {
            a_count += 1;
        } else {
            b_count += 1;
        }
        model.observe(&SelfTraceStep {
            context: context.clone(),
            outcome: SelfOutcome {
                passed,
                primary_code: None,
                routed_pod: Some(PodRef::new(record.pod().clone(), record.pod_generation())),
                propensity: Some(record.propensity()),
            },
        });
    }
    // The boost really did send far more traffic to "a" ...
    assert!(
        f64::from(a_count) > 2.0 * f64::from(b_count),
        "{a_count} {b_count}"
    );
    // ... but its inverse-propensity value is the common pass rate, not inflated, and neither
    // Pod is preferred.
    let code = FindingCode::new("target");
    let (a, ..) = model
        .repair_value(&code, &PodRef::new(pod("a"), Generation(1)))
        .unwrap();
    let (b, ..) = model
        .repair_value(&code, &PodRef::new(pod("b"), Generation(1)))
        .unwrap();
    assert!((a - 0.5).abs() < 0.06, "a {a}");
    assert!((b - 0.5).abs() < 0.08, "b {b}");
    assert!(!model
        .hints(&context, &HintPolicy::default())
        .iter()
        .any(|h| matches!(h, SelfModelHint::PreferRepairPod { .. })));
}

#[test]
fn logged_propensities_are_calibrated_for_a_state_changing_schema_and_ips_is_unbiased() {
    let schema = new_schema(SchemaLimits::default());
    let full = router(0.1, controlled(UniformScorer, &schema, ControlArm::Full));
    let names = ["a", "b", "c"];
    let rates = [0.8, 0.5, 0.2];
    let mut chosen = [0u32; 3];
    let mut expected = [0.0f64; 3];
    let mut variance = [0.0f64; 3];
    let mut value_a = 0.0;
    let n = 6000;
    let mut rng = Rng(123);
    for i in 0..n {
        let record = target_record(&full, "ope", i);
        let index = names
            .iter()
            .position(|nm| pod(nm) == *record.pod())
            .unwrap();
        for (p, width) in record.support() {
            let k = names.iter().position(|nm| pod(nm) == *p).unwrap();
            let prob = *width as f64 / 9_007_199_254_740_992.0;
            expected[k] += prob;
            variance[k] += prob * (1.0 - prob);
        }
        chosen[index] += 1;
        let passed = rng.unit() < rates[index];
        if index == 0 {
            value_a += if passed {
                1.0 / record.propensity().get()
            } else {
                0.0
            };
        }
        schema.observe(&record);
        schema.settle(&record, passed);
    }
    // Each Pod's realised count matches the sum of its logged probabilities (the schema changed
    // state throughout): a mislogged propensity would break this.
    for k in 0..3 {
        let sd = variance[k].sqrt();
        assert!(
            (f64::from(chosen[k]) - expected[k]).abs() < 4.5 * sd,
            "{k}: chosen {} expected {} sd {sd}",
            chosen[k],
            expected[k]
        );
    }
    // IPS of "always a" recovers a's pass rate although the schema steered the traffic.
    let estimate = value_a / f64::from(n);
    assert!((estimate - 0.8).abs() < 0.06, "{estimate}");
    // And the controller did steer: the best Pod got the most traffic.
    assert!(chosen[0] > chosen[2]);
}

#[test]
fn tables_are_bounded_duplicates_are_ignored_and_reports_are_consistent() {
    let schema = new_schema(SchemaLimits {
        max_contexts: 2,
        max_pods_per_context: 2,
        window: 10,
        max_preferences: 1,
        dedup_window: 4,
        ..SchemaLimits::default()
    });
    let mut reg = FindingRegistry::new();
    for code in ["c0", "c1", "c2", "c3"] {
        for p in ["p0", "p1", "p2", "p3"] {
            let expected = reg.generation();
            reg.apply(RegistryUpdate {
                expected,
                change: RegistryChange::AddRepair {
                    code: FindingCode::new(code),
                    pod: pod(p),
                },
            })
            .unwrap();
        }
    }
    let wide = FindingRouter::new(
        reg,
        RouterConfig::default(),
        UniformScorer,
        SamplingKey::from_bytes([1; 32]),
        NonceSequence::starting_at([1; 8], 0),
    )
    .unwrap();
    let mut rng = Rng(3);
    let mut last = None;
    for i in 0..600u32 {
        let code = format!("c{}", rng.next() % 4);
        let r = report(
            VerificationStatus::Fail,
            VerificationLevel::Deterministic,
            &[(&code, true)],
        );
        let record = route(&wide, &format!("b-{i}"), i, &r).records()[0].clone();
        schema.observe(&record);
        last = Some(record);
    }
    let report = schema.report();
    assert_eq!(report.contexts_tracked, 2);
    assert!(report.overflow_observations > 200);
    assert_eq!(report.observations, 600);
    assert_eq!(report.epoch, 600);
    // Duplicates inside the dedup window are ignored and counted; the window is bounded.
    let last = last.unwrap();
    schema.observe(&last);
    schema.settle(&last, true);
    schema.settle(&last, true);
    let after = schema.report();
    assert_eq!(after.observations, 600);
    assert_eq!(after.duplicates_ignored, 2);
    assert_eq!(after.digest, schema.digest());
    // Once the ring has moved on the same record counts again (the ring is the only memory).
    for i in 600..606u32 {
        let r = report_for(i);
        schema.observe(&route(&wide, &format!("b-{i}"), i, &r).records()[0].clone());
    }
    schema.observe(&last);
    assert_eq!(schema.report().duplicates_ignored, 2);
}

fn report_for(i: u32) -> VerificationReport {
    report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[(&format!("c{}", i % 4), true)],
    )
}

#[test]
fn concurrent_observation_and_scoring_stay_consistent() {
    let schema = new_schema(SchemaLimits::default());
    let uniform = Arc::new(router(
        0.1,
        controlled(UniformScorer, &schema, ControlArm::Full),
    ));
    let handles: Vec<_> = (0..4u32)
        .map(|t| {
            let schema = Arc::clone(&schema);
            let router = Arc::clone(&uniform);
            std::thread::spawn(move || {
                for i in 0..200u32 {
                    let record = target_record(&router, &format!("t{t}"), t * 1000 + i);
                    schema.observe(&record);
                    schema.settle(&record, i % 2 == 0);
                    let _ = schema.report();
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let report = schema.report();
    assert_eq!(report.observations, 800);
    assert_eq!(report.epoch, 800);
    assert_eq!(report.digest, schema.digest());
}

#[test]
fn invalid_configuration_is_refused() {
    let limits = SchemaLimits {
        window: 0,
        ..SchemaLimits::default()
    };
    assert_eq!(
        AttentionSchema::new(limits).unwrap_err(),
        SchemaError::InvalidLimits { field: "window" }
    );
    let ok = |b, h, r, m, k| ControlConfig::new(b, h, r, m, TargetPower::Two, 1, k);
    let one = BalanceStrength::One;
    assert!(ok(one, 2.0, 8.0, 0.25, 8).is_ok());
    for (h, r, m, k) in [
        (0.9, 8.0, 0.25, 8),
        (f64::NAN, 8.0, 0.25, 8),
        (9.0, 8.0, 0.25, 8),
        (2.0, 65.0, 0.25, 8),
        (2.0, 0.5, 0.25, 8),
        (2.0, f64::INFINITY, 0.25, 8),
        (2.0, 8.0, -0.1, 8),
        (2.0, 8.0, 1.1, 8),
        (2.0, 8.0, f64::NAN, 8),
        (2.0, 8.0, 0.25, 0),
    ] {
        assert!(ok(one, h, r, m, k).is_err(), "{h} {r} {m} {k}");
    }
}

#[test]
fn replay_gives_identical_picks_reports_and_digest_in_every_arm() {
    for arm in ARMS {
        let run = || {
            let schema = trained();
            let router = router(0.05, controlled(UniformScorer, &schema, arm));
            let mut picks = Vec::new();
            for i in 0..300 {
                let record = target_record(&router, "replay", i);
                picks.push((
                    record.pod().clone(),
                    record.draw(),
                    record.support().to_vec(),
                ));
                schema.observe(&record);
                schema.settle(&record, i % 3 == 0);
            }
            (picks, schema.digest(), schema.report())
        };
        let (p1, d1, r1) = run();
        let (p2, d2, r2) = run();
        assert_eq!(p1, p2, "{arm:?}");
        assert_eq!(d1, d2);
        assert_eq!(r1, r2);
        assert_ne!(d1, new_schema(SchemaLimits::default()).digest());
    }
}

#[test]
fn the_whole_control_path_uses_only_exactly_rounded_operations() {
    // A pinned digest of a fixed run: it changes if any platform-dependent function (exp, pow,
    // ln) ever enters the update or the weights.
    let schema = new_schema(SchemaLimits::default());
    let uniform = router(0.1, UniformScorer);
    train(&schema, &uniform, "pin", 0..200, |r, i| {
        (*r.pod() == pod("a")) ^ (i % 5 == 0)
    });
    schema.apply_hints(&[valid_hint("target", "b")]);
    let router = router(0.1, controlled(UniformScorer, &schema, ControlArm::Full));
    let mut picks = String::new();
    for i in 1000..1100 {
        let record = target_record(&router, "pin2", i);
        picks.push_str(&record.pod().0);
        schema.observe(&record);
        schema.settle(&record, i % 2 == 0);
    }
    let hex: String = schema.digest().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(picks, "cbbcbbacabababaacaaabbacacaacbbacaabacbbabbbabbcbbaacaaabcacaaacbbbaaccccabbabbaaaacbbbaabaabaaabbaa");
    // Digest layout v3 (issue #141: the generation table's last-seen epochs and evictions);
    // the picks above are unchanged from v2.
    assert_eq!(
        hex,
        "1676fba406718c9d2d85ec48160f16011879818ded8f49ad026151540d4b5ac2"
    );
}
