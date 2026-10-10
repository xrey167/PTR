use ptr_feedback::{Candidate, Critique};
use ptr_router::{
    classify, AdmissibleScorer, CodeRoute, EligibilitySnapshot, EscalationReason, Exploration,
    FallbackAction, FallbackPolicy, FallbackReason, FindingCode, FindingRegistry,
    FindingRouteErrorKind, FindingRouteRequest, FindingRouter, NonceSequence, PodEligibility,
    RegistryChange, RegistryError, RegistryGeneration, RegistryUpdate, ReportClass, RouteClass,
    RouteError, RouteOutcome, RouterConfig, RouterLimits, SamplingKey, ScoringContext,
    UniformScorer,
};
use ptr_types::{
    CandidateId, Digest, Generation, PodId, Probability, RequestId, VerificationLevel,
};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::error::Error;

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

fn pod(id: &str) -> PodId {
    PodId::from(id)
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

fn config(fallback: FallbackPolicy) -> RouterConfig {
    RouterConfig {
        fallback,
        limits: RouterLimits::default(),
        exploration: Exploration::NONE,
    }
}

fn router_with<S: AdmissibleScorer>(
    registry: FindingRegistry,
    policy: FallbackPolicy,
    scorer: S,
) -> FindingRouter<S> {
    router_cfg(registry, config(policy), scorer)
}

fn router_cfg<S: AdmissibleScorer>(
    registry: FindingRegistry,
    config: RouterConfig,
    scorer: S,
) -> FindingRouter<S> {
    FindingRouter::new(
        registry,
        config,
        scorer,
        SamplingKey::from_bytes([7; 32]),
        NonceSequence::starting_at([1; 8], 0),
    )
    .unwrap()
}

fn router(registry: FindingRegistry) -> FindingRouter {
    router_with(registry, FallbackPolicy::default(), UniformScorer)
}

/// Every pod is eligible at generation 1 unless a test says otherwise.
struct Everyone;
impl PodEligibility for Everyone {
    fn current_generation(&self, _: &PodId) -> Option<Generation> {
        Some(Generation(1))
    }
}

fn try_route(
    router: &FindingRouter<impl AdmissibleScorer>,
    i: u32,
    report: &VerificationReport,
    required: VerificationLevel,
    eligibility: &dyn PodEligibility,
) -> Result<RouteOutcome, ptr_router::FindingRouteError> {
    let id = RequestId::from(format!("req-{i}").as_str());
    router.route(&FindingRouteRequest {
        request_id: &id,
        request_digest: digest(i),
        report,
        required,
        eligibility,
    })
}

fn route(
    router: &FindingRouter<impl AdmissibleScorer>,
    i: u32,
    report: &VerificationReport,
    required: VerificationLevel,
) -> RouteOutcome {
    try_route(router, i, report, required, &Everyone).unwrap()
}

#[test]
fn every_status_finding_level_combination_maps_by_the_verifier_alone() {
    let router = router(registry(
        &[("target", "fix-target"), ("effect", "fix-effect")],
        &["gen"],
    ));
    let required = VerificationLevel::SampleVerified;
    for status in STATUSES {
        for level in LEVELS {
            for findings in [
                vec![],
                vec![("style", false)],
                vec![("target", true)],
                vec![("target", true), ("effect", true), ("target", true)],
            ] {
                let hard = findings.iter().any(|(_, h)| *h);
                let report = report(status, level, &findings);
                let outcome = route(&router, 1, &report, required);
                let accepted = status == VerificationStatus::Pass
                    && !hard
                    && LEVELS.iter().position(|l| *l == level)
                        >= LEVELS.iter().position(|l| *l == required);
                let repaired = status == VerificationStatus::Fail && hard;
                assert_eq!(
                    outcome.is_accept(),
                    accepted,
                    "{status:?} {level:?} {findings:?}"
                );
                assert_eq!(
                    outcome.is_repair(),
                    repaired,
                    "{status:?} {level:?} {findings:?}"
                );
                if repaired {
                    let mut distinct: Vec<&str> = Vec::new();
                    for (code, hard) in &findings {
                        if *hard && !distinct.contains(code) {
                            distinct.push(code);
                        }
                    }
                    assert_eq!(outcome.records().len(), distinct.len());
                }
            }
        }
    }
}

#[test]
fn repair_routes_each_distinct_code_in_report_order() {
    let router = router(registry(
        &[("target", "fix-target"), ("effect", "fix-effect")],
        &[],
    ));
    let r = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[
            ("effect", true),
            ("style", false),
            ("target", true),
            ("effect", true),
        ],
    );
    let RouteOutcome::Repair { routes } = route(&router, 1, &r, VerificationLevel::Deterministic)
    else {
        panic!("expected repair");
    };
    let pods: Vec<_> = routes
        .iter()
        .map(|route| match route {
            CodeRoute::Routed(record) => (record.class().clone(), record.pod().clone()),
            CodeRoute::Unroutable(_) => panic!("both codes are registered"),
        })
        .collect();
    assert_eq!(
        pods,
        [
            (RouteClass::Repair("effect".into()), pod("fix-effect")),
            (RouteClass::Repair("target".into()), pod("fix-target")),
        ]
    );
}

#[test]
fn forged_pass_with_a_hard_finding_is_never_accepted_or_repaired() {
    for action in [FallbackAction::Regenerate, FallbackAction::Escalate] {
        let policy = FallbackPolicy {
            forged_pass: action,
            ..FallbackPolicy::default()
        };
        let router = router_with(
            registry(&[("target", "fix")], &["gen"]),
            policy,
            UniformScorer,
        );
        for level in LEVELS {
            let r = report(VerificationStatus::Pass, level, &[("target", true)]);
            assert_eq!(
                classify(&r, VerificationLevel::Unverified, &RouterLimits::default()),
                ReportClass::Fallback(FallbackReason::ForgedPass)
            );
            let outcome = route(&router, 3, &r, VerificationLevel::Unverified);
            assert!(!outcome.is_accept() && !outcome.is_repair(), "{outcome:?}");
        }
    }
}

struct Counting<'a>(&'a Cell<u32>);
impl AdmissibleScorer for Counting<'_> {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        self.0.set(self.0.get() + 1);
        Ok(context
            .admissible
            .iter()
            .map(|p| (p.clone(), 1.0))
            .collect())
    }
}

#[test]
fn the_scorer_is_never_consulted_for_accept_or_escalate() {
    let calls = Cell::new(0);
    let policy = FallbackPolicy {
        unknown: FallbackAction::Escalate,
        ..FallbackPolicy::default()
    };
    let router = router_with(
        registry(&[("target", "fix")], &[]),
        policy,
        Counting(&calls),
    );
    let ok = report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        &[],
    );
    assert!(route(&router, 1, &ok, VerificationLevel::Deterministic).is_accept());
    let disputed = report(
        VerificationStatus::Disputed,
        VerificationLevel::Deterministic,
        &[],
    );
    assert!(matches!(
        route(&router, 1, &disputed, VerificationLevel::Deterministic),
        RouteOutcome::Escalate {
            reason: EscalationReason::Policy(FallbackReason::Disputed)
        }
    ));
    let forged = report(
        VerificationStatus::Pass,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    assert!(matches!(
        route(&router, 1, &forged, VerificationLevel::Deterministic),
        RouteOutcome::Escalate {
            reason: EscalationReason::Policy(FallbackReason::ForgedPass)
        }
    ));
    let unknown = report(
        VerificationStatus::Unknown,
        VerificationLevel::Deterministic,
        &[],
    );
    assert!(matches!(
        route(&router, 1, &unknown, VerificationLevel::Deterministic),
        RouteOutcome::Escalate {
            reason: EscalationReason::Policy(FallbackReason::Unknown)
        }
    ));
    let soft = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[],
    );
    assert!(matches!(
        route(&router, 1, &soft, VerificationLevel::Deterministic),
        RouteOutcome::Escalate {
            reason: EscalationReason::NoProposer(_)
        }
    ));
    assert_eq!(calls.get(), 0);
}

#[test]
fn unknown_and_disputed_are_never_repaired_even_with_hard_findings() {
    for unknown in [FallbackAction::Regenerate, FallbackAction::Escalate] {
        for disputed in [FallbackAction::Regenerate, FallbackAction::Escalate] {
            let policy = FallbackPolicy {
                unknown,
                disputed,
                ..FallbackPolicy::default()
            };
            let router = router_with(
                registry(&[("target", "fix")], &["gen"]),
                policy,
                UniformScorer,
            );
            for (status, action, reason) in [
                (
                    VerificationStatus::Unknown,
                    unknown,
                    FallbackReason::Unknown,
                ),
                (
                    VerificationStatus::Disputed,
                    disputed,
                    FallbackReason::Disputed,
                ),
            ] {
                for findings in [vec![], vec![("target", true)]] {
                    let r = report(status, VerificationLevel::Deterministic, &findings);
                    match route(&router, 5, &r, VerificationLevel::Unverified) {
                        RouteOutcome::Regenerate {
                            reason: got,
                            proposer,
                        } => {
                            assert_eq!(action, FallbackAction::Regenerate);
                            assert_eq!(got, reason);
                            assert_eq!(proposer.pod(), &pod("gen"));
                            assert_eq!(proposer.class(), &RouteClass::Regenerate(reason));
                        }
                        RouteOutcome::Escalate { reason: got } => {
                            assert_eq!(action, FallbackAction::Escalate);
                            assert_eq!(got, EscalationReason::Policy(reason));
                        }
                        other => panic!("{status:?} must not give {other:?}"),
                    }
                }
            }
        }
    }
}

#[test]
fn fail_without_hard_finding_and_below_level_pass_do_not_accept_or_repair() {
    let router = router(registry(&[("target", "fix")], &["gen"]));
    let soft = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("style", false)],
    );
    assert_eq!(
        classify(
            &soft,
            VerificationLevel::Deterministic,
            &RouterLimits::default()
        ),
        ReportClass::Fallback(FallbackReason::FailWithoutHardFinding)
    );
    assert!(matches!(
        route(&router, 1, &soft, VerificationLevel::Deterministic),
        RouteOutcome::Regenerate {
            reason: FallbackReason::FailWithoutHardFinding,
            ..
        }
    ));
    let weak = report(
        VerificationStatus::Pass,
        VerificationLevel::LatentAgreement,
        &[],
    );
    assert!(matches!(
        route(&router, 1, &weak, VerificationLevel::FullSemantic),
        RouteOutcome::Regenerate {
            reason: FallbackReason::BelowLevel,
            ..
        }
    ));
    assert!(route(&router, 1, &weak, VerificationLevel::LatentAgreement).is_accept());
}

#[test]
fn regenerate_without_a_proposer_escalates_explicitly() {
    let router = router(registry(&[], &[]));
    let r = report(
        VerificationStatus::Unknown,
        VerificationLevel::Unverified,
        &[],
    );
    assert!(matches!(
        route(&router, 1, &r, VerificationLevel::Unverified),
        RouteOutcome::Escalate {
            reason: EscalationReason::NoProposer(FallbackReason::Unknown)
        }
    ));
}

#[test]
fn unroutable_codes_are_reported_and_counted_not_dropped() {
    let router = router(registry(&[("target", "fix")], &[]));
    let r = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[
            ("payload-size", true),
            ("target", true),
            ("payload-size", true),
        ],
    );
    for i in 0..3 {
        let outcome = route(&router, i, &r, VerificationLevel::Deterministic);
        assert_eq!(
            outcome.unroutable_codes(),
            [&FindingCode::new("payload-size")]
        );
        assert_eq!(outcome.records().len(), 1);
    }
    let only = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("novel", true)],
    );
    let outcome = route(&router, 9, &only, VerificationLevel::Deterministic);
    assert!(outcome.is_repair() && outcome.records().is_empty());
    assert_eq!(outcome.unroutable_codes(), [&FindingCode::new("novel")]);
    let counts = router.unroutable_counts();
    assert_eq!(counts.get(&FindingCode::new("payload-size")), Some(&3));
    assert_eq!(counts.get(&FindingCode::new("novel")), Some(&1));
    assert_eq!(counts.len(), 2);
}

#[test]
fn multi_pod_propensities_sum_to_one_and_match_empirical_frequencies() {
    let router = router(registry(
        &[
            ("target", "a"),
            ("target", "b"),
            ("target", "c"),
            ("target", "d"),
        ],
        &[],
    ));
    let r = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    let n = 4000;
    let mut counts: BTreeMap<PodId, u32> = BTreeMap::new();
    let mut propensity_by_pod: BTreeMap<PodId, f64> = BTreeMap::new();
    for i in 0..n {
        let outcome = route(&router, i, &r, VerificationLevel::Deterministic);
        let record = outcome.records()[0];
        assert_eq!(record.support().len(), 4);
        assert_eq!(record.pod_generation(), Generation(1));
        assert_eq!(record.key_id(), router.key_id());
        *counts.entry(record.pod().clone()).or_default() += 1;
        let p = record.propensity().get();
        assert!(p > 0.0 && p <= 1.0);
        propensity_by_pod.insert(record.pod().clone(), p);
    }
    assert_eq!(propensity_by_pod.len(), 4);
    let sum: f64 = propensity_by_pod.values().sum();
    assert!((sum - 1.0).abs() < 1e-12, "{sum}");
    for (pod, count) in counts {
        let freq = f64::from(count) / f64::from(n);
        assert!(
            (freq - propensity_by_pod[&pod]).abs() < 0.03,
            "{pod:?} {freq}"
        );
    }
}

#[test]
fn routing_is_deterministic_for_a_fixed_key_and_nonce_sequence() {
    let make = || {
        router(registry(
            &[("target", "a"), ("target", "b"), ("target", "c")],
            &["g1", "g2"],
        ))
    };
    let (one, two) = (make(), make());
    let hard = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    let unknown = report(
        VerificationStatus::Unknown,
        VerificationLevel::Unverified,
        &[],
    );
    for i in 0..50 {
        let r = if i % 2 == 0 { &hard } else { &unknown };
        assert_eq!(
            route(&one, i, r, VerificationLevel::Deterministic),
            route(&two, i, r, VerificationLevel::Deterministic)
        );
    }
    // A different key changes at least one of the draws.
    let other = FindingRouter::new(
        registry(&[("target", "a"), ("target", "b"), ("target", "c")], &[]),
        config(FallbackPolicy::default()),
        UniformScorer,
        SamplingKey::from_bytes([8; 32]),
        NonceSequence::starting_at([1; 8], 0),
    )
    .unwrap();
    let base = make();
    let differs = (0..50).any(|i| {
        route(&base, i, &hard, VerificationLevel::Deterministic).records()[0].draw()
            != route(&other, i, &hard, VerificationLevel::Deterministic).records()[0].draw()
    });
    assert!(differs);
}

#[test]
fn stale_generation_registrations_are_refused_and_change_nothing() {
    let mut router = router(registry(&[("target", "a")], &[]));
    let current = router.registry().generation();
    assert_eq!(current.get(), RegistryGeneration::INITIAL.get() + 1);
    let stale = RegistryUpdate {
        expected: RegistryGeneration::INITIAL,
        change: RegistryChange::AddRepair {
            code: "target".into(),
            pod: pod("b"),
        },
    };
    assert_eq!(
        router.update_registry(stale),
        Err(RegistryError::StaleGeneration {
            expected: RegistryGeneration::INITIAL,
            current
        })
    );
    assert_eq!(router.registry().generation(), current);
    assert_eq!(router.registry().repair_pods(&"target".into()), [pod("a")]);
    let fresh = RegistryUpdate {
        expected: current,
        change: RegistryChange::AddRepair {
            code: "target".into(),
            pod: pod("b"),
        },
    };
    let next = router.update_registry(fresh.clone()).unwrap();
    assert!(next > current);
    // Replaying the same update is now stale, and a duplicate at the new generation is refused.
    assert!(matches!(
        router.update_registry(fresh),
        Err(RegistryError::StaleGeneration { .. })
    ));
    assert_eq!(
        router.update_registry(RegistryUpdate {
            expected: next,
            change: RegistryChange::AddRepair {
                code: "target".into(),
                pod: pod("b")
            },
        }),
        Err(RegistryError::AlreadyRegistered)
    );
    assert_eq!(
        router.update_registry(RegistryUpdate {
            expected: next,
            change: RegistryChange::RemoveProposer { pod: pod("zzz") },
        }),
        Err(RegistryError::NotRegistered)
    );
    assert_eq!(router.registry().generation(), next);
}

struct Skewed;
impl AdmissibleScorer for Skewed {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        Ok(context
            .admissible
            .iter()
            .map(|p| (p.clone(), if p.0 == "a" { 3.0 } else { 1.0 }))
            .collect())
    }
}

struct Escaping;
impl AdmissibleScorer for Escaping {
    fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        Ok(vec![(PodId::from("not-admitted"), 1.0)])
    }
}

struct Silent;
impl AdmissibleScorer for Silent {
    fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        Ok(Vec::new())
    }
}

#[test]
fn a_scorer_reranks_inside_the_admissible_set_only() {
    let hard = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    let reg = registry(&[("target", "a"), ("target", "b")], &[]);

    let skewed = router_with(reg.clone(), FallbackPolicy::default(), Skewed);
    for i in 0..200 {
        let outcome = route(&skewed, i, &hard, VerificationLevel::Deterministic);
        let record = outcome.records()[0];
        let expected = if record.pod() == &pod("a") {
            0.75
        } else {
            0.25
        };
        assert!((record.propensity().get() - expected).abs() < 1e-12);
    }

    for (name, router) in [
        (
            "escaping",
            router_with(
                reg.clone(),
                FallbackPolicy::default(),
                EscapingBox(Box::new(Escaping)),
            ),
        ),
        (
            "silent",
            router_with(
                reg,
                FallbackPolicy::default(),
                EscapingBox(Box::new(Silent)),
            ),
        ),
    ] {
        let err = try_route(
            &router,
            0,
            &hard,
            VerificationLevel::Deterministic,
            &Everyone,
        )
        .unwrap_err();
        assert_eq!(err.request_id(), &RequestId::from("req-0"));
        assert_eq!(err.request_digest(), digest(0));
        assert!(err.to_string().contains("req-0"));
        match (name, err.kind()) {
            ("escaping", FindingRouteErrorKind::OutsideAdmissibleSet { pod: p }) => {
                assert_eq!(p, &pod("not-admitted"));
                assert!(err.source().is_none());
            }
            ("silent", FindingRouteErrorKind::NoMass) => {}
            (name, other) => panic!("{name}: {other:?}"),
        }
        assert!(router.unroutable_counts().is_empty());
        assert_eq!(router.unroutable_overflow(), 0);
    }
}

struct EscapingBox(Box<dyn AdmissibleScorer>);
impl AdmissibleScorer for EscapingBox {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        self.0.weights(context)
    }
}

#[test]
fn ips_over_logged_records_of_a_uniform_policy_recovers_the_target_value() {
    let pods = ["p0", "p1", "p2", "p3"];
    let router = router(registry(
        &pods.iter().map(|p| ("target", *p)).collect::<Vec<_>>(),
        &[],
    ));
    let reward = |p: &PodId| match p.0.as_str() {
        "p0" => 0.2,
        "p1" => 0.4,
        "p2" => 0.6,
        _ => 0.8,
    };
    let hard = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    let n = 4000;
    let mut ips_p2 = 0.0;
    let mut ips_uniform = 0.0;
    let mut weights = 0.0;
    for i in 0..n {
        let outcome = route(&router, i, &hard, VerificationLevel::Deterministic);
        let record = outcome.records()[0];
        let r = reward(record.pod());
        // Target A: always p2 (true value 0.6). Target B: the logging policy (true value 0.5).
        let target_p2 = f64::from(u8::from(record.pod() == &pod("p2")));
        ips_p2 += record.ips_weight(Probability::new(target_p2 as f32).unwrap()) * r;
        ips_uniform += record.ips_weight(Probability::new(0.25).unwrap()) * r;
        weights += record.ips_weight(Probability::new(target_p2 as f32).unwrap());
    }
    let n = f64::from(n);
    assert!((ips_p2 / n - 0.6).abs() < 0.08, "{}", ips_p2 / n);
    assert!((ips_uniform / n - 0.5).abs() < 0.03, "{}", ips_uniform / n);
    // SNIPS (self-normalized) recovers it too, and the mean weight is near one.
    assert!((ips_p2 / weights - 0.6).abs() < 0.03);
    assert!((weights / n - 1.0).abs() < 0.1);
}

#[test]
fn classify_matches_ptr_feedback_over_the_full_matrix() {
    let limits = RouterLimits::default();
    let id = CandidateId::from("c");
    let findings_cases: [Vec<(&str, bool)>; 5] = [
        vec![],
        vec![("style", false)],
        vec![("target", true)],
        vec![("target", true), ("effect", true), ("target", true)],
        vec![("style", false), ("effect", true)],
    ];
    for status in STATUSES {
        for level in LEVELS {
            for required in LEVELS {
                for findings in &findings_cases {
                    let r = report(status, level, findings);
                    let candidate = Candidate {
                        id: id.clone(),
                        value: (),
                        verification: Some(r.clone()),
                    };
                    let critique = Critique::from_report(&id, &r);
                    let class = classify(&r, required, &limits);
                    assert_eq!(
                        matches!(class, ReportClass::Accept(_)),
                        candidate.passes(required),
                        "{status:?} {level:?} {required:?} {findings:?}"
                    );
                    match class {
                        ReportClass::Repair(codes) => {
                            assert!(critique.repairable);
                            let codes: Vec<&str> = codes.iter().map(FindingCode::as_str).collect();
                            assert_eq!(codes, critique.issues);
                        }
                        ReportClass::Accept(_) | ReportClass::Fallback(_) => {
                            assert!(!critique.repairable)
                        }
                    }
                }
            }
        }
    }
}

type Seen = std::cell::RefCell<Vec<(RouteClass, Vec<PodId>)>>;
struct Recording<'a>(&'a Seen);
impl AdmissibleScorer for Recording<'_> {
    fn weights(&self, context: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        self.0
            .borrow_mut()
            .push((context.class.clone(), context.admissible.to_vec()));
        Ok(context
            .admissible
            .iter()
            .map(|p| (p.clone(), 1.0))
            .collect())
    }
}

#[test]
fn the_scorer_sees_exactly_the_eligible_pods_of_each_class() {
    let seen = Seen::default();
    let reg = registry(
        &[
            ("target", "a"),
            ("target", "b"),
            ("effect", "e"),
            ("effect", "f"),
        ],
        &["g1", "g2"],
    );
    let router = router_with(reg, FallbackPolicy::default(), Recording(&seen));
    let mut view = EligibilitySnapshot::new();
    for p in ["a", "b", "e", "g1", "g2"] {
        view.set(pod(p), Generation(3));
    }
    let two = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true), ("effect", true)],
    );
    try_route(&router, 1, &two, VerificationLevel::Deterministic, &view).unwrap();
    let unknown = report(
        VerificationStatus::Unknown,
        VerificationLevel::Unverified,
        &[],
    );
    try_route(&router, 2, &unknown, VerificationLevel::Unverified, &view).unwrap();
    assert_eq!(
        *seen.borrow(),
        [
            (
                RouteClass::Repair("target".into()),
                vec![pod("a"), pod("b")]
            ),
            (RouteClass::Repair("effect".into()), vec![pod("e")]),
            (
                RouteClass::Regenerate(FallbackReason::Unknown),
                vec![pod("g1"), pod("g2")]
            ),
        ]
    );
}

#[test]
fn ineligible_pods_are_never_chosen_and_are_counted() {
    let reg = registry(
        &[("target", "a"), ("target", "b"), ("solo", "s")],
        &["g1", "g2"],
    );
    let router = router(reg);
    let mut view = EligibilitySnapshot::new();
    view.set(pod("a"), Generation(7));
    view.set(pod("g2"), Generation(9));
    let hard = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true), ("solo", true)],
    );
    let unknown = report(
        VerificationStatus::Unknown,
        VerificationLevel::Unverified,
        &[],
    );
    for i in 0..100 {
        let outcome =
            try_route(&router, i, &hard, VerificationLevel::Deterministic, &view).unwrap();
        assert_eq!(outcome.unroutable_codes(), [&FindingCode::new("solo")]);
        let record = outcome.records()[0];
        assert_eq!(record.pod(), &pod("a"));
        assert_eq!(record.propensity().get(), 1.0);
        assert_eq!(record.pod_generation(), Generation(7));
        let RouteOutcome::Regenerate { proposer, .. } =
            try_route(&router, i, &unknown, VerificationLevel::Unverified, &view).unwrap()
        else {
            panic!("expected regenerate");
        };
        assert_eq!(proposer.pod(), &pod("g2"));
        assert_eq!(proposer.pod_generation(), Generation(9));
    }
    // Per round: b and s skipped in the repair, g1 in the regeneration.
    assert_eq!(router.ineligible_skipped(), 300);
    assert_eq!(
        router.unroutable_counts().get(&FindingCode::new("solo")),
        Some(&100)
    );

    let none = EligibilitySnapshot::new();
    let outcome = try_route(&router, 1, &hard, VerificationLevel::Deterministic, &none).unwrap();
    assert!(outcome.records().is_empty());
    assert_eq!(outcome.unroutable_codes().len(), 2);
    assert!(matches!(
        try_route(&router, 1, &unknown, VerificationLevel::Unverified, &none).unwrap(),
        RouteOutcome::Escalate {
            reason: EscalationReason::NoProposer(FallbackReason::Unknown)
        }
    ));
}

#[test]
fn limits_bound_codes_code_length_and_counter_state() {
    let limits = RouterLimits {
        max_codes_per_report: 3,
        max_code_len: 8,
        max_tracked_unroutable_codes: 2,
    };
    let cfg = RouterConfig {
        limits,
        ..config(FallbackPolicy::default())
    };
    let router = router_cfg(registry(&[], &[]), cfg, UniformScorer);
    let many: Vec<String> = (0..4).map(|i| format!("c{i}")).collect();
    let pairs: Vec<(&str, bool)> = many.iter().map(|c| (c.as_str(), true)).collect();
    let r = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &pairs,
    );
    assert_eq!(
        classify(&r, VerificationLevel::Deterministic, &limits),
        ReportClass::Fallback(FallbackReason::TooManyCodes)
    );
    let repeated = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[
            ("a", true),
            ("a", true),
            ("a", true),
            ("a", true),
            ("b", true),
        ],
    );
    assert!(matches!(
        classify(&repeated, VerificationLevel::Deterministic, &limits),
        ReportClass::Repair(codes) if codes.len() == 2
    ));
    let long = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("123456789", true)],
    );
    assert_eq!(
        classify(&long, VerificationLevel::Deterministic, &limits),
        ReportClass::Fallback(FallbackReason::OversizedCode)
    );
    assert!(matches!(
        route(&router, 1, &r, VerificationLevel::Deterministic),
        RouteOutcome::Escalate {
            reason: EscalationReason::Policy(FallbackReason::TooManyCodes)
        }
    ));
    assert!(matches!(
        route(&router, 1, &long, VerificationLevel::Deterministic),
        RouteOutcome::Escalate {
            reason: EscalationReason::Policy(FallbackReason::OversizedCode)
        }
    ));
    for code in ["x1", "x2", "x3", "x4", "x1"] {
        let r = report(
            VerificationStatus::Fail,
            VerificationLevel::Deterministic,
            &[(code, true)],
        );
        route(&router, 1, &r, VerificationLevel::Deterministic);
    }
    let counts = router.unroutable_counts();
    assert_eq!(counts.len(), 2);
    assert_eq!(counts.get(&FindingCode::new("x1")), Some(&2));
    assert_eq!(router.unroutable_overflow(), 2);
}

#[test]
fn registry_capacity_is_bounded() {
    let mut registry = FindingRegistry::with_capacity(1, 2);
    let mut add = |code: &str, id: &str| {
        let expected = registry.generation();
        registry.apply(RegistryUpdate {
            expected,
            change: RegistryChange::AddRepair {
                code: code.into(),
                pod: pod(id),
            },
        })
    };
    add("a", "p1").unwrap();
    add("a", "p2").unwrap();
    assert_eq!(add("a", "p3"), Err(RegistryError::CapacityExceeded));
    assert_eq!(add("b", "p1"), Err(RegistryError::CapacityExceeded));
    assert_eq!(registry.generation().get(), 2);
}

struct OnlyA;
impl AdmissibleScorer for OnlyA {
    fn weights(&self, _: &ScoringContext<'_>) -> Result<Vec<(PodId, f64)>, RouteError> {
        Ok(vec![(PodId::from("a"), 1.0)])
    }
}

#[test]
fn exploration_floor_keeps_every_eligible_pod_reachable() {
    let reg = registry(&[("target", "a"), ("target", "b")], &[]);
    let cfg = RouterConfig {
        exploration: Exploration::new(0.2).unwrap(),
        ..config(FallbackPolicy::default())
    };
    let router = router_cfg(reg, cfg, OnlyA);
    let hard = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    let n = 4000;
    let mut b = 0u32;
    for i in 0..n {
        let outcome = route(&router, i, &hard, VerificationLevel::Deterministic);
        let record = outcome.records()[0];
        let total: u64 = record.support().iter().map(|(_, w)| w).sum();
        assert_eq!(total, 1u64 << 53);
        assert_eq!(record.support().len(), 2);
        let expected = if record.pod() == &pod("a") { 0.9 } else { 0.1 };
        assert!((record.propensity().get() - expected).abs() < 1e-9);
        b += u32::from(record.pod() == &pod("b"));
    }
    assert!((f64::from(b) / f64::from(n) - 0.1).abs() < 0.02);
    assert!(Exploration::new(f64::NAN).is_err());
    assert!(Exploration::new(1.5).is_err());
    assert!(Exploration::new(1e-9).is_err());
    assert!(Exploration::new(0.0).is_ok());
}

#[test]
fn skewed_logging_policy_matches_empirical_frequency_and_stays_unbiased() {
    let reg = registry(&[("target", "a"), ("target", "b")], &[]);
    let router = router_with(reg, FallbackPolicy::default(), Skewed);
    let hard = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    let n = 4000;
    let mut a = 0u32;
    let (mut ips, mut num, mut den) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let outcome = route(&router, i, &hard, VerificationLevel::Deterministic);
        let record = outcome.records()[0];
        let is_a = record.pod() == &pod("a");
        a += u32::from(is_a);
        // Target: always "a"; the reward of "a" is 1 and of "b" 0, so the true value is 1.
        let w = record.ips_weight(Probability::new(f32::from(u8::from(is_a))).unwrap());
        let reward = f64::from(u8::from(is_a));
        ips += w * reward;
        num += w * reward;
        den += w;
    }
    assert!((f64::from(a) / f64::from(n) - 0.75).abs() < 0.03);
    assert!((ips / f64::from(n) - 1.0).abs() < 0.05);
    assert!((num / den - 1.0).abs() < 1e-9);
}

#[test]
fn records_carry_the_registry_generation_of_route_time() {
    let mut router = router(registry(&[("target", "a")], &[]));
    let hard = report(
        VerificationStatus::Fail,
        VerificationLevel::Deterministic,
        &[("target", true)],
    );
    let before = router.registry().generation();
    let record = route(&router, 1, &hard, VerificationLevel::Deterministic).records()[0].clone();
    assert_eq!(record.generation(), before);
    let after = router
        .update_registry(RegistryUpdate {
            expected: before,
            change: RegistryChange::AddRepair {
                code: "target".into(),
                pod: pod("b"),
            },
        })
        .unwrap();
    let outcome = route(&router, 2, &hard, VerificationLevel::Deterministic);
    let later = outcome.records()[0];
    assert_eq!(later.generation(), after);
    assert_eq!(later.support().len(), 2);
    let mut generation = after;
    for p in ["a", "b"] {
        generation = router
            .update_registry(RegistryUpdate {
                expected: generation,
                change: RegistryChange::RemoveRepair {
                    code: "target".into(),
                    pod: pod(p),
                },
            })
            .unwrap();
    }
    let outcome = route(&router, 3, &hard, VerificationLevel::Deterministic);
    assert_eq!(outcome.unroutable_codes(), [&FindingCode::new("target")]);
}
