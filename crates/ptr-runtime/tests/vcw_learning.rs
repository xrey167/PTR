//! L1 learning in the verifier-closed loop: the loop reports the verifier's verdict on every
//! routed Pod to a learner, a bandit learns which repair Pod is better from those verdicts, an
//! off-policy gate decides whether a host may promote it, and promotion is a host action the loop
//! never takes. All experts are the seeded programmatic ones: where one repair Pod is much better
//! than the other for a code, that is a configured fact of the simulation, not a claim about a
//! model.

use std::sync::Arc;

use ptr_config::PtrConfig;
use ptr_events::RuntimeEvent;
use ptr_router::LearnerConfig;
use ptr_router::{
    ActiveScorer, EligibilitySnapshot, Exploration, FindingCode, NonceSequence, PolicyAction,
    PolicyBook, PromotionGate, PromotionRule, RouteClass, RouterConfig, RoutingLearner,
    SamplingKey, SelfModel, SelfModelLimits, Verdict,
};
use ptr_runtime::expert::{StructurePolicy, StructurePolicyConfig};
use ptr_runtime::vcw::experts::{DefectDoc, DefectDocVerifier, SimulatedExperts, SimulationConfig};
use ptr_runtime::vcw::{
    RequestBudget, Roster, VcwConfig, VcwError, VcwLoop, VcwOutcome, VcwParts, VcwRequest,
};
use ptr_runtime::PtrRuntime;
use ptr_types::{Generation, PodId, ProjectId, RequestId, VerificationLevel};
use sha2::{Digest as _, Sha256};

fn pod(name: &str) -> PodId {
    PodId::from(name)
}

fn code() -> FindingCode {
    FindingCode::new("x")
}

fn view() -> EligibilitySnapshot {
    let mut view = EligibilitySnapshot::new();
    for name in ["gen", "fix_good", "fix_bad"] {
        view.set(pod(name), Generation(1));
    }
    view
}

fn make_loop(
    seed: u8,
    scorer: ActiveScorer,
    learner: Option<&Arc<RoutingLearner>>,
) -> VcwLoop<ActiveScorer> {
    let roster = Roster {
        proposers: vec![pod("gen")],
        repairers: vec![
            (pod("fix_good"), vec![code()]),
            (pod("fix_bad"), vec![code()]),
        ],
    };
    let vcw = VcwLoop::new(VcwParts {
        config: VcwConfig::new("learning-test", [3; 32]),
        roster,
        router: RouterConfig::default(),
        scorer,
        key: SamplingKey::from_bytes([seed; 32]),
        nonces: NonceSequence::starting_at([seed; 8], 0),
        self_model: SelfModel::new(SelfModelLimits::default()),
        schema: None,
        structure: StructurePolicy::new(StructurePolicyConfig::default()).unwrap(),
    })
    .unwrap();
    match learner {
        Some(learner) => vcw.with_learner(Arc::clone(learner)).unwrap(),
        None => vcw,
    }
}

/// Every proposal carries the defect `x`; `fix_good` clears it with probability `good`,
/// `fix_bad` with probability `bad`.
fn experts(seed: u64, good: f64, bad: f64, decline: bool) -> SimulatedExperts<DefectDoc> {
    let mut config = SimulationConfig::new(seed)
        .with_error_rate("x", 1.0)
        .with_pod_repair_success("fix_good", "x", good)
        .with_pod_repair_success("fix_bad", "x", bad);
    config.decline_on_failure = decline;
    let mut experts = SimulatedExperts::new(config, "task").unwrap();
    experts.add_proposer(pod("gen")).unwrap();
    experts.add_repairer(pod("fix_good"), &["x"]).unwrap();
    experts.add_repairer(pod("fix_bad"), &["x"]).unwrap();
    experts
}

fn learner(seed: u64) -> Arc<RoutingLearner> {
    Arc::new(RoutingLearner::new(LearnerConfig::new("defect-doc", seed)).unwrap())
}

#[derive(Default, Debug)]
struct Tally {
    requests: u32,
    verified: u32,
    routes: u64,
    repair_to_good: u64,
    repair_to_bad: u64,
}

fn run_requests(
    vcw: &mut VcwLoop<ActiveScorer>,
    experts: &mut SimulatedExperts<DefectDoc>,
    prefix: &str,
    n: u32,
) -> Tally {
    let view = view();
    let mut tally = Tally::default();
    for i in 0..n {
        let id = format!("{prefix}-{i}");
        let request = VcwRequest {
            request_id: RequestId::from(id.as_str()),
            project: ProjectId::from("proj"),
            request_digest: Sha256::digest(id.as_bytes()).into(),
            required: VerificationLevel::SampleVerified,
            budget: RequestBudget::new(8, 6).unwrap(),
        };
        let run = vcw
            .run(
                &request,
                &view,
                &DefectDocVerifier,
                &mut experts.guarded(&view),
            )
            .unwrap();
        tally.requests += 1;
        tally.verified += u32::from(matches!(run.outcome, VcwOutcome::Verified(_)));
        for record in run.trace.routes() {
            tally.routes += 1;
            if matches!(record.class(), RouteClass::Repair(c) if c == &code()) {
                match record.pod().0.as_str() {
                    "fix_good" => tally.repair_to_good += 1,
                    "fix_bad" => tally.repair_to_bad += 1,
                    other => panic!("unexpected repairer {other}"),
                }
            }
        }
    }
    tally
}

fn gate() -> PromotionGate {
    PromotionGate::new(PromotionRule::new(0.05, 500, 100.0)).unwrap()
}

#[test]
fn the_bandit_learns_the_better_repair_pod_and_a_host_promotes_it_on_held_out_evidence() {
    let mut book = PolicyBook::new("defect-doc");
    let learner = learner(7);
    let mut vcw = make_loop(1, book.scorer(), Some(&learner));
    let mut sim = experts(21, 0.9, 0.1, false);

    // Phase A: the incumbent (uniform) routes; the learner settles every verdict.
    let before = run_requests(&mut vcw, &mut sim, "a", 300);
    let share =
        before.repair_to_good as f64 / (before.repair_to_good + before.repair_to_bad) as f64;
    assert!(
        (0.4..0.6).contains(&share),
        "the incumbent is uniform: {share}"
    );
    let candidate = learner.snapshot();
    let good_mean = candidate.mean(&RouteClass::Repair(code()), &pod("fix_good"));
    let bad_mean = candidate.mean(&RouteClass::Repair(code()), &pod("fix_bad"));
    assert!(good_mean > 0.75 && bad_mean < 0.3, "{good_mean} {bad_mean}");
    let calls_before = vcw.counters().verifier_calls;

    // Phase B: more incumbent traffic, logged after the snapshot: the held-out evidence.
    run_requests(&mut vcw, &mut sim, "b", 900);
    let log = learner.log_from(candidate.cutoff());
    assert!(log.len() > 1500, "{}", log.len());

    // The learner changed nothing about routing by itself.
    assert!(!book.is_learned());
    assert!(book.events().is_empty());

    let assessment = gate()
        .assess(&log, &candidate, &book.scorer(), book.active_digest())
        .unwrap();
    let evidence = match assessment.verdict {
        Verdict::Promote(evidence) => evidence,
        Verdict::Refuse(reason) => panic!("{reason:?} {:?}", assessment.comparison),
    };
    assert!(evidence.lower_bound() > 0.05);

    // The host promotes, and records it as a runtime event.
    let event = book.promote(candidate.clone(), evidence).unwrap();
    assert_eq!(event.action, PolicyAction::Promoted);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.record_vcw_policy_event(&event);
    let recorded = runtime.events().iter().any(|envelope| {
        matches!(
            &envelope.event,
            RuntimeEvent::VcwPolicyChanged { action, to, evidence: Some(_), .. }
                if action == "promoted" && *to == candidate.digest()
        )
    });
    assert!(recorded);

    // Phase C: the same loop now routes by the learned policy, inside the admissible set and
    // under the router's floor.
    let calls_mid = vcw.counters().verifier_calls;
    let after = run_requests(&mut vcw, &mut sim, "c", 600);
    let learned_share =
        after.repair_to_good as f64 / (after.repair_to_good + after.repair_to_bad) as f64;
    assert!(learned_share > 0.85, "{learned_share}");
    assert!(
        learned_share < 0.99,
        "the exploration floor was removed: {learned_share}"
    );
    assert!(after.repair_to_bad > 0, "the worse pod must stay reachable");
    // The better pod finishes requests in fewer verifier calls and exhausts fewer of them.
    let per_request_before = (calls_mid - calls_before) as f64 / 900.0;
    let per_request_after = (vcw.counters().verifier_calls - calls_mid) as f64 / 600.0;
    assert!(
        per_request_after < per_request_before * 0.85,
        "{per_request_after} vs {per_request_before}"
    );
    assert!(f64::from(after.verified) / 600.0 >= f64::from(before.verified) / 300.0);

    // Rolling back restores the incumbent and is recorded too.
    let rolled = book.rollback().unwrap();
    runtime.record_vcw_policy_event(&rolled);
    assert_eq!(rolled.action, PolicyAction::RolledBack);
    let mut vcw_after = make_loop(2, book.scorer(), None);
    let uniform_again = run_requests(&mut vcw_after, &mut sim, "d", 300);
    let uniform_share = uniform_again.repair_to_good as f64
        / (uniform_again.repair_to_good + uniform_again.repair_to_bad) as f64;
    assert!((0.4..0.6).contains(&uniform_share), "{uniform_share}");
}

#[test]
fn every_routed_pod_is_settled_exactly_once_including_declined_repairs() {
    for decline in [false, true] {
        let book = PolicyBook::new("defect-doc");
        let learner = learner(3);
        let mut vcw = make_loop(4, book.scorer(), Some(&learner));
        let mut sim = experts(5, 0.6, 0.3, decline);
        let tally = run_requests(&mut vcw, &mut sim, "s", 200);
        assert_eq!(
            learner.settled(),
            tally.routes,
            "decline_on_failure = {decline}: routed records and settled verdicts differ"
        );
        assert_eq!(learner.dropped_log(), 0);
        assert_eq!(learner.untracked_arms(), 0);
    }
}

#[test]
fn a_declined_repair_is_a_failure_of_the_routed_pod_not_a_missing_observation() {
    // The bad pod declines whenever it fails; the good pod never fails. Declines must lower the
    // bad pod's posterior exactly as failures do.
    let book = PolicyBook::new("defect-doc");
    let learner = learner(11);
    let mut vcw = make_loop(6, book.scorer(), Some(&learner));
    let mut sim = experts(8, 1.0, 0.0, true);
    run_requests(&mut vcw, &mut sim, "decline", 200);
    let policy = learner.snapshot();
    let class = RouteClass::Repair(code());
    let bad = policy.counts(&class, &pod("fix_bad"));
    let good = policy.counts(&class, &pod("fix_good"));
    assert!(bad.not_cleared > 20 && bad.cleared == 0, "{bad:?}");
    assert!(good.cleared > 20 && good.not_cleared == 0, "{good:?}");
}

#[test]
fn the_reward_is_the_verifiers_verdict_whatever_the_pods_do() {
    // The simulation's "repairs" never succeed (a pod that claims to repair does not): no
    // candidate is ever accepted, so no routed Pod is ever credited.
    let book = PolicyBook::new("defect-doc");
    let learner = learner(13);
    let mut vcw = make_loop(9, book.scorer(), Some(&learner));
    let mut sim = experts(10, 0.0, 0.0, false);
    let tally = run_requests(&mut vcw, &mut sim, "never", 100);
    assert_eq!(tally.verified, 0);
    let policy = learner.snapshot();
    let class = RouteClass::Repair(code());
    for name in ["fix_good", "fix_bad"] {
        let counts = policy.counts(&class, &pod(name));
        assert_eq!(counts.cleared, 0, "{name}");
        assert!(counts.not_cleared > 0, "{name}");
    }
    assert!(learner.log_from(0).iter().all(|d| !d.reward().passed()));
}

#[test]
fn a_learner_must_use_the_routers_exploration_rate() {
    let book = PolicyBook::new("defect-doc");
    let other = Arc::new(
        RoutingLearner::new(LearnerConfig {
            exploration: Exploration::new(0.2).unwrap(),
            ..LearnerConfig::new("defect-doc", 1)
        })
        .unwrap(),
    );
    let vcw = make_loop(1, book.scorer(), None);
    assert!(matches!(vcw.with_learner(other), Err(VcwError::Config(_))));
}

#[test]
fn the_loop_holds_the_learner_and_the_read_only_scorer_but_no_way_to_change_the_policy() {
    // The loop's scorer type can only be read from; the book that changes it is not reachable
    // from the loop. The source scan of the loop's module (tests/vcw_loop.rs) pins that the
    // module never names the book or its actions.
    fn scorer_is_read_only<S: ptr_router::AdmissibleScorer>(_: &VcwLoop<S>) {}
    let book = PolicyBook::new("defect-doc");
    let vcw = make_loop(1, book.scorer(), None);
    scorer_is_read_only(&vcw);
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/vcw");
    for entry in std::fs::read_dir(src).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).unwrap();
            for line in text.lines().filter(|l| !l.trim_start().starts_with("//")) {
                for token in ["PolicyBook", ".promote(", ".rollback(", "PromotionGate"] {
                    assert!(!line.contains(token), "{}: {line}", path.display());
                }
            }
        }
    }
}
