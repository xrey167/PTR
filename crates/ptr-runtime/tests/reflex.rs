//! Failover to an explicit default pod, and counters that do not claim more than happened.
//!
//! The default pod is held to everything a routed pod is: a registry candidate of the project,
//! a live generation (checked at every use), no effect beyond pure or read, and the same
//! verifier path. The counters are telemetry and never decide.

use ptr_config::PtrConfig;
use ptr_events::RuntimeEvent;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_router::{
    AbstentionReason, CheckedRouter, FailoverCoverage, FallbackPlan, MissReason, NonceSequence,
    PodRouter, RouteDistribution, RouteError, RouteRequest, Router, SamplingKey, UncoveredReason,
};
use ptr_runtime::{
    pod_lifecycle_target, FallbackConfigError, PodSampler, PtrRuntime, RuntimeError,
    SetFallbackPlan, MAX_FALLBACK_PLANS,
};
use ptr_types::{
    CapabilityId, CapsuleId, Effect, Generation, NamespaceId, PodAddress, PodId,
    PodRevisionAddress, Probability, ProjectId, TypeId, VerificationLevel,
};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const PROJECT: &str = "project";

fn project() -> ProjectId {
    ProjectId::from(PROJECT)
}

fn capability() -> CapabilityId {
    CapabilityId::from("Echo<Text>")
}

struct Ask {
    capability: &'static str,
}

impl InferenceBackend for Ask {
    fn name(&self) -> &'static str {
        "ask"
    }

    fn infer(&self, _: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::PodRequested {
            capability: CapabilityId::from(self.capability),
            input_type: TypeId::from("Text"),
            payload: b"question".to_vec(),
        }])
    }
}

impl ResumableInferenceBackend for Ask {
    fn resume(&self, _: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        Ok(vec![ModelEvent::Finished])
    }
}

#[derive(Clone, Copy)]
enum Behaviour {
    /// Returns its input.
    Echo,
    /// Returns its input behind a prefix.
    Prefix(&'static str),
    /// Fails when invoked.
    Fail,
}

struct TestPod {
    manifest: PodManifest,
    behaviour: Behaviour,
    invocations: Arc<AtomicUsize>,
}

impl DynPod for TestPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        match self.behaviour {
            Behaviour::Echo => Ok(input),
            Behaviour::Prefix(prefix) => Ok(TypedPayload {
                type_id: input.type_id,
                bytes: [prefix.as_bytes(), &input.bytes].concat(),
            }),
            Behaviour::Fail => Err("pod failed".into()),
        }
    }
}

struct PodSpec {
    project: &'static str,
    id: &'static str,
    effect: Effect,
    behaviour: Behaviour,
}

fn spec(id: &'static str, behaviour: Behaviour) -> PodSpec {
    PodSpec {
        project: PROJECT,
        id,
        effect: Effect::Pure,
        behaviour,
    }
}

/// A registry of the given pods and, per pod id, how often it was invoked.
fn registry(specs: &[PodSpec]) -> (PodRegistry, Vec<(&'static str, Arc<AtomicUsize>)>) {
    let mut registry = PodRegistry::default();
    let mut counters = Vec::new();
    for spec in specs {
        let invocations = Arc::new(AtomicUsize::new(0));
        counters.push((spec.id, Arc::clone(&invocations)));
        registry.register(Arc::new(TestPod {
            manifest: PodManifest {
                project: ProjectId::from(spec.project),
                id: PodId::from(spec.id),
                capabilities: vec![capability()],
                accepts: vec![TypeId::from("Text")],
                produces: vec![TypeId::from("Text")],
                effects: vec![spec.effect],
                protocol_version: 1,
            },
            behaviour: spec.behaviour,
            invocations,
        }));
    }
    (registry, counters)
}

fn invocations(counters: &[(&'static str, Arc<AtomicUsize>)], id: &str) -> usize {
    counters
        .iter()
        .find(|(name, _)| *name == id)
        .map(|(_, count)| count.load(Ordering::SeqCst))
        .expect("pod is in the registry")
}

/// Passes everything except an output that starts with `reject`.
struct Verdict {
    reject: &'static str,
}

impl Verifier<TypedPayload> for Verdict {
    fn verify(&self, candidate: &TypedPayload) -> VerificationReport {
        let contradicted = candidate.bytes.starts_with(self.reject.as_bytes());
        VerificationReport {
            status: if contradicted {
                VerificationStatus::Fail
            } else {
                VerificationStatus::Pass
            },
            level: VerificationLevel::Deterministic,
            score: Probability::new(1.0).unwrap(),
            findings: vec![],
        }
    }
}

const PASS: Verdict = Verdict { reject: "\u{0}" };

/// A learned router that declines or has nothing to score.
type MakeError = fn() -> RouteError;

struct Declines(MakeError);

impl Router for Declines {
    fn policy(&self, _: &RouteRequest<'_>) -> Result<RouteDistribution, RouteError> {
        Err((self.0)())
    }
}

fn abstains(reason: AbstentionReason) -> MakeError {
    match reason {
        AbstentionReason::LowScore => || abstained(AbstentionReason::LowScore),
        AbstentionReason::SmallMargin => || abstained(AbstentionReason::SmallMargin),
        AbstentionReason::PartialCoverage => || abstained(AbstentionReason::PartialCoverage),
    }
}

fn abstained(reason: AbstentionReason) -> RouteError {
    RouteError::Abstained {
        best: PodId::from("primary"),
        runner_up: None,
        best_score: 0.4,
        margin: 0.4,
        reason,
    }
}

fn no_representation() -> RouteError {
    RouteError::NoRepresentation { candidates: 2 }
}

fn checked<R: Router>(router: R) -> CheckedRouter<R> {
    CheckedRouter::new(
        router,
        SamplingKey::from_bytes([7; 32]),
        NonceSequence::starting_at([1; 8], 0),
    )
}

fn runtime() -> PtrRuntime {
    PtrRuntime::new(PtrConfig::default()).unwrap()
}

fn revision(pod: &str, generation: u64) -> PodRevisionAddress {
    PodRevisionAddress {
        address: PodAddress::new(project(), NamespaceId::from("ns"), PodId::from(pod)).unwrap(),
        semantic_revision: [generation as u8 + 1; 32],
        generation: Generation(generation),
    }
}

/// Makes `revision`'s generation the one the lifecycle authority holds live for its pod.
fn make_live(runtime: &mut PtrRuntime, revision: &PodRevisionAddress) {
    let target = pod_lifecycle_target(revision);
    let event = match runtime.live_generation(&target) {
        None => LedgerEvent::CapsuleCommitted {
            project: revision.address.project.0.as_str().into(),
            capsule: CapsuleId::from(target.as_str()),
            generation: revision.generation,
        },
        Some(old) => LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(target.as_str()),
            old,
            new: revision.generation,
        },
    };
    runtime.commit(event).unwrap();
}

fn revoke(runtime: &mut PtrRuntime, revision: &PodRevisionAddress) {
    runtime
        .commit(LedgerEvent::Revoked {
            subject: pod_lifecycle_target(revision),
            generation: revision.generation,
        })
        .unwrap();
}

/// Sets the plan of `capability()` in the project: `default` is served by `revision`.
fn plan(runtime: &mut PtrRuntime, default: &str, max_attempts: u8, revision: PodRevisionAddress) {
    runtime
        .set_fallback_plan(SetFallbackPlan {
            project: project(),
            capability: capability(),
            plan: FallbackPlan::new(Some(PodId::from(default)), max_attempts).unwrap(),
            default_revision: Some(revision),
        })
        .unwrap();
}

/// Runs one Pod request through `router` and returns the outcome.
fn run_routed<R: Router + Send + Sync>(
    runtime: &mut PtrRuntime,
    id: &str,
    pods: &PodRegistry,
    router: &CheckedRouter<R>,
    verifier: &Verdict,
) -> Result<(), RuntimeError> {
    let sampler: &dyn PodSampler = router;
    runtime
        .run_resumable_with_pods_using_router(
            id.into(),
            &project(),
            "go",
            &Ask {
                capability: "Echo<Text>",
            },
            pods,
            sampler,
            verifier,
            2,
        )
        .map(|_| ())
}

/// Runs one Pod request through the plain deterministic selector.
fn run_plain(
    runtime: &mut PtrRuntime,
    id: &str,
    pods: &PodRegistry,
    verifier: &Verdict,
) -> Result<(), RuntimeError> {
    runtime
        .run_resumable_with_pods(
            id.into(),
            &project(),
            "go",
            &Ask {
                capability: "Echo<Text>",
            },
            pods,
            verifier,
            2,
        )
        .map(|_| ())
}

fn events(runtime: &PtrRuntime) -> Vec<RuntimeEvent> {
    runtime
        .events()
        .iter()
        .map(|envelope| envelope.event.clone())
        .collect()
}

fn invoked_pods(runtime: &PtrRuntime) -> Vec<String> {
    events(runtime)
        .into_iter()
        .filter_map(|event| match event {
            RuntimeEvent::PodInvoked(pod) => Some(pod),
            _ => None,
        })
        .collect()
}

fn count<F: Fn(&RuntimeEvent) -> bool>(runtime: &PtrRuntime, matches: F) -> usize {
    events(runtime)
        .iter()
        .filter(|event| matches(event))
        .count()
}

fn low_score_router() -> CheckedRouter<Declines> {
    checked(Declines(abstains(AbstentionReason::LowScore)))
}

mod without_a_default {
    use super::*;

    #[test]
    fn a_miss_with_no_plan_is_pod_unavailable_and_counted_as_uncovered() {
        let (pods, _) = registry(&[]);
        let mut runtime = runtime();
        let outcome = run_plain(&mut runtime, "r1", &pods, &PASS);
        assert!(matches!(outcome, Err(RuntimeError::PodUnavailable { .. })));

        let counters = runtime.reflex_counters();
        assert_eq!(counters.misses(), 1);
        assert_eq!(counters.failovers(), 0);
        assert_eq!(
            counters.miss_reasons().get(&MissReason::NoCandidate),
            Some(&1)
        );
        assert_eq!(
            counters.uncovered_reasons().get(&UncoveredReason::NoPlan),
            Some(&1)
        );
        assert_eq!(
            counters.coverage(),
            FailoverCoverage::Uncovered {
                misses: 1,
                failovers: 0
            }
        );
        assert!(!counters.all_misses_covered());
    }

    #[test]
    fn an_abstention_with_no_plan_still_fails_with_a_pod_free_reason() {
        let (pods, counts) = registry(&[spec("primary", Behaviour::Echo)]);
        let mut runtime = runtime();
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert_eq!(
            outcome,
            Err(RuntimeError::RouteFailed {
                reason: "routing abstained"
            })
        );
        assert_eq!(invocations(&counts, "primary"), 0);
        assert_eq!(
            runtime
                .reflex_counters()
                .miss_reasons()
                .get(&MissReason::Abstained(AbstentionReason::LowScore)),
            Some(&1)
        );
    }

    #[test]
    fn a_plan_that_names_no_default_covers_nothing() {
        let (pods, _) = registry(&[spec("primary", Behaviour::Echo)]);
        let mut runtime = runtime();
        runtime
            .set_fallback_plan(SetFallbackPlan {
                project: project(),
                capability: capability(),
                plan: FallbackPlan::new(None, 2).unwrap(),
                default_revision: None,
            })
            .unwrap();
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert!(matches!(outcome, Err(RuntimeError::RouteFailed { .. })));
        assert_eq!(
            runtime
                .reflex_counters()
                .uncovered_reasons()
                .get(&UncoveredReason::NoDefault),
            Some(&1)
        );
    }

    #[test]
    fn a_router_failure_that_is_not_a_miss_is_not_covered_by_a_default() {
        let (pods, counts) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        let broken = checked(Declines(|| RouteError::MissingChoice { candidates: 2 }));
        let outcome = run_routed(&mut runtime, "r1", &pods, &broken, &PASS);
        assert_eq!(
            outcome,
            Err(RuntimeError::RouteFailed {
                reason: "routing policy made no choice"
            })
        );
        assert_eq!(invocations(&counts, "fallback"), 0);
        assert_eq!(runtime.reflex_counters().misses(), 0);
    }
}

mod with_a_default {
    use super::*;

    fn with_fallback() -> (
        PtrRuntime,
        PodRegistry,
        Vec<(&'static str, Arc<AtomicUsize>)>,
    ) {
        let (pods, counts) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        (runtime, pods, counts)
    }

    #[test]
    fn an_abstention_fails_over_and_the_miss_is_covered() {
        let (mut runtime, pods, counts) = with_fallback();
        let committed = runtime.committed_events().len();
        run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS).unwrap();

        assert_eq!(invocations(&counts, "primary"), 0);
        assert_eq!(invocations(&counts, "fallback"), 1);
        assert_eq!(invoked_pods(&runtime), vec!["fallback".to_owned()]);
        // The request and the promoted output of the default pod.
        assert_eq!(runtime.committed_events().len(), committed + 2);

        let counters = runtime.reflex_counters();
        assert_eq!(counters.misses(), 1);
        assert_eq!(counters.failovers(), 1);
        assert_eq!(counters.hits(), 0);
        assert_eq!(counters.coverage(), FailoverCoverage::Covered { misses: 1 });
        assert!(counters.all_misses_covered());

        let seen = events(&runtime);
        let miss = seen
            .iter()
            .position(|event| matches!(event, RuntimeEvent::ReflexMiss { .. }))
            .expect("miss event");
        let invoked = seen
            .iter()
            .position(|event| matches!(event, RuntimeEvent::PodInvoked(_)))
            .expect("invoked event");
        let failover = seen
            .iter()
            .position(|event| matches!(event, RuntimeEvent::ReflexFailover { .. }))
            .expect("failover event");
        assert!(miss < invoked && invoked < failover);
        assert_eq!(
            seen[miss],
            RuntimeEvent::ReflexMiss {
                reason: "abstained_low_score".into()
            }
        );
        assert_eq!(
            seen[failover],
            RuntimeEvent::ReflexFailover {
                to: "fallback".into()
            }
        );
        // A failover is not a sampled decision: no propensity is logged for it.
        assert_eq!(
            count(&runtime, |e| matches!(e, RuntimeEvent::PodRouted { .. })),
            0
        );
    }

    #[test]
    fn abstention_reasons_and_no_representation_are_counted_apart() {
        let (mut runtime, pods, _) = with_fallback();
        let routers: [(&str, MakeError); 4] = [
            ("a", abstains(AbstentionReason::LowScore)),
            ("b", abstains(AbstentionReason::SmallMargin)),
            ("c", abstains(AbstentionReason::PartialCoverage)),
            ("d", no_representation),
        ];
        for (id, make) in routers {
            run_routed(&mut runtime, id, &pods, &checked(Declines(make)), &PASS).unwrap();
        }
        let counters = runtime.reflex_counters();
        let reasons = counters.miss_reasons();
        for reason in [
            MissReason::Abstained(AbstentionReason::LowScore),
            MissReason::Abstained(AbstentionReason::SmallMargin),
            MissReason::Abstained(AbstentionReason::PartialCoverage),
            MissReason::NoRepresentation,
        ] {
            assert_eq!(reasons.get(&reason), Some(&1), "{reason:?}");
        }
        assert_eq!(reasons.len(), 4);
        assert_eq!(counters.coverage(), FailoverCoverage::Covered { misses: 4 });
    }

    #[test]
    fn all_misses_covered_turns_false_when_one_miss_is_not_covered() {
        let (mut runtime, pods, counts) = with_fallback();
        run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS).unwrap();
        assert!(runtime.reflex_counters().all_misses_covered());

        // The default's generation is revoked after the first, covered miss.
        revoke(&mut runtime, &revision("fallback", 1));
        let outcome = run_routed(&mut runtime, "r2", &pods, &low_score_router(), &PASS);
        assert_eq!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                miss: MissReason::Abstained(AbstentionReason::LowScore),
                reason: UncoveredReason::GenerationNotLive,
            })
        );
        assert_eq!(invocations(&counts, "fallback"), 1);

        let counters = runtime.reflex_counters();
        assert_eq!(
            counters.coverage(),
            FailoverCoverage::Uncovered {
                misses: 2,
                failovers: 1
            }
        );
        assert!(!counters.all_misses_covered());
    }

    #[test]
    fn a_superseded_default_generation_is_refused_at_use() {
        let (mut runtime, pods, counts) = with_fallback();
        make_live(&mut runtime, &revision("fallback", 2));
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert_eq!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                miss: MissReason::Abstained(AbstentionReason::LowScore),
                reason: UncoveredReason::GenerationNotLive,
            })
        );
        assert_eq!(invocations(&counts, "fallback"), 0);
    }

    #[test]
    fn a_default_the_lifecycle_never_made_live_fails_closed() {
        let (pods, counts) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        plan(&mut runtime, "fallback", 2, revision("fallback", 1));
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert!(matches!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                reason: UncoveredReason::GenerationNotLive,
                ..
            })
        ));
        assert_eq!(invocations(&counts, "fallback"), 0);
    }

    #[test]
    fn a_default_that_is_not_a_registry_candidate_is_refused() {
        let (pods, counts) = registry(&[spec("primary", Behaviour::Echo)]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert!(matches!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                reason: UncoveredReason::NotACandidate,
                ..
            })
        ));
        assert_eq!(invocations(&counts, "primary"), 0);
    }

    #[test]
    fn a_default_registered_in_another_project_is_not_a_candidate() {
        let (pods, counts) = registry(&[
            spec("primary", Behaviour::Echo),
            PodSpec {
                project: "other",
                ..spec("fallback", Behaviour::Prefix("fb:"))
            },
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert!(matches!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                reason: UncoveredReason::NotACandidate,
                ..
            })
        ));
        assert_eq!(invocations(&counts, "fallback"), 0);
    }

    #[test]
    fn a_default_pod_with_an_effect_beyond_read_is_refused_and_never_invoked() {
        let (pods, counts) = registry(&[
            spec("primary", Behaviour::Echo),
            PodSpec {
                effect: Effect::Mutation,
                ..spec("fallback", Behaviour::Prefix("fb:"))
            },
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        let committed = runtime.committed_events().len();
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert!(matches!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                reason: UncoveredReason::EffectNotAllowed,
                ..
            })
        ));
        assert_eq!(invocations(&counts, "fallback"), 0);
        assert!(invoked_pods(&runtime).is_empty());
        // Only the request was ingested.
        assert_eq!(runtime.committed_events().len(), committed + 1);
    }

    #[test]
    fn a_read_effect_default_is_allowed() {
        let (pods, _) = registry(&[
            spec("primary", Behaviour::Echo),
            PodSpec {
                effect: Effect::Read,
                ..spec("fallback", Behaviour::Prefix("fb:"))
            },
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS).unwrap();
        assert_eq!(runtime.reflex_counters().failovers(), 1);
    }

    #[test]
    fn a_verifier_contradiction_of_the_default_output_is_never_promoted() {
        let (mut runtime, pods, counts) = with_fallback();
        let committed = runtime.committed_events().len();
        let contradicting = Verdict { reject: "fb:" };
        let outcome = run_routed(
            &mut runtime,
            "r1",
            &pods,
            &low_score_router(),
            &contradicting,
        );
        assert_eq!(
            outcome,
            Err(RuntimeError::PodVerificationFailed {
                pod: "fallback".into()
            })
        );
        // The default pod ran, its output was judged by the verifier and refused: only the
        // request was committed, the output was not.
        assert_eq!(invocations(&counts, "fallback"), 1);
        assert_eq!(runtime.committed_events().len(), committed + 1);
        assert_eq!(
            count(&runtime, |e| matches!(
                e,
                RuntimeEvent::VerifierResult { passed: false, .. }
            )),
            1
        );
        assert_eq!(
            count(&runtime, |e| matches!(
                e,
                RuntimeEvent::ReflexFailover { .. }
            )),
            0
        );

        let counters = runtime.reflex_counters();
        assert_eq!(counters.failovers(), 0);
        assert_eq!(
            counters
                .uncovered_reasons()
                .get(&UncoveredReason::DefaultFailed),
            Some(&1)
        );
        assert!(!counters.all_misses_covered());
    }

    #[test]
    fn a_default_pod_that_fails_when_invoked_leaves_the_miss_uncovered() {
        let (pods, counts) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Fail),
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        let outcome = run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS);
        assert_eq!(outcome, Err(RuntimeError::Pod("pod failed".into())));
        assert_eq!(invocations(&counts, "fallback"), 1);
        assert!(!runtime.reflex_counters().all_misses_covered());
    }

    #[test]
    fn a_failing_primary_pod_is_a_dispatch_miss_and_fails_over() {
        // The plain selector breaks ties by the greater id: "zeta" is chosen and fails.
        let (pods, counts) = registry(&[
            spec("zeta", Behaviour::Fail),
            spec("alpha", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        let fallback = revision("alpha", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "alpha", 2, fallback);
        run_plain(&mut runtime, "r1", &pods, &PASS).unwrap();

        assert_eq!(invocations(&counts, "zeta"), 1);
        assert_eq!(invocations(&counts, "alpha"), 1);
        assert_eq!(
            invoked_pods(&runtime),
            vec!["zeta".to_owned(), "alpha".to_owned()]
        );
        let counters = runtime.reflex_counters();
        assert_eq!(
            counters.miss_reasons().get(&MissReason::DispatchError),
            Some(&1)
        );
        assert_eq!(counters.coverage(), FailoverCoverage::Covered { misses: 1 });
    }

    #[test]
    fn the_attempt_budget_counts_the_primary_dispatch() {
        let (pods, counts) = registry(&[
            spec("zeta", Behaviour::Fail),
            spec("alpha", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        let fallback = revision("alpha", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "alpha", 1, fallback);
        let outcome = run_plain(&mut runtime, "r1", &pods, &PASS);
        assert_eq!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                miss: MissReason::DispatchError,
                reason: UncoveredReason::AttemptsExhausted,
            })
        );
        assert_eq!(invocations(&counts, "alpha"), 0);
    }

    #[test]
    fn a_single_attempt_still_serves_a_miss_that_dispatched_nothing() {
        let (pods, _) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 1, fallback);
        run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS).unwrap();
        assert!(runtime.reflex_counters().all_misses_covered());
    }

    #[test]
    fn the_default_is_not_retried_when_it_is_the_pod_that_just_failed() {
        let (pods, counts) = registry(&[spec("alpha", Behaviour::Fail)]);
        let mut runtime = runtime();
        let fallback = revision("alpha", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "alpha", 2, fallback);
        let outcome = run_plain(&mut runtime, "r1", &pods, &PASS);
        assert_eq!(
            outcome,
            Err(RuntimeError::FailoverRefused {
                miss: MissReason::DispatchError,
                reason: UncoveredReason::DefaultIsFailedPod,
            })
        );
        assert_eq!(invocations(&counts, "alpha"), 1);
    }

    #[test]
    fn a_plan_of_one_project_does_not_cover_another() {
        let (pods, _) = registry(&[]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        let outcome = runtime.run_resumable_with_pods(
            "r1".into(),
            &ProjectId::from("elsewhere"),
            "go",
            &Ask {
                capability: "Echo<Text>",
            },
            &pods,
            &PASS,
            2,
        );
        assert!(matches!(outcome, Err(RuntimeError::PodUnavailable { .. })));
        assert_eq!(
            runtime
                .reflex_counters()
                .uncovered_reasons()
                .get(&UncoveredReason::NoPlan),
            Some(&1)
        );
    }
}

mod counters {
    use super::*;

    #[test]
    fn nothing_observed_is_not_a_green_statement() {
        let runtime = runtime();
        let counters = runtime.reflex_counters();
        assert_eq!(counters.coverage(), FailoverCoverage::NoObservations);
        assert!(!counters.all_misses_covered());
    }

    #[test]
    fn a_hit_is_counted_and_a_run_without_misses_says_it_never_exercised_failover() {
        let (pods, _) = registry(&[spec("primary", Behaviour::Echo)]);
        let mut runtime = runtime();
        run_plain(&mut runtime, "r1", &pods, &PASS).unwrap();
        let counters = runtime.reflex_counters();
        assert_eq!(counters.hits(), 1);
        assert_eq!(counters.misses(), 0);
        assert_eq!(counters.coverage(), FailoverCoverage::NoMisses { hits: 1 });
        assert_eq!(
            count(&runtime, |e| matches!(e, RuntimeEvent::ReflexHit { .. })),
            1
        );
    }

    #[test]
    fn a_primary_answer_the_verifier_refuses_is_neither_hit_nor_miss() {
        let (pods, _) = registry(&[spec("primary", Behaviour::Prefix("bad:"))]);
        let mut runtime = runtime();
        let outcome = run_plain(&mut runtime, "r1", &pods, &Verdict { reject: "bad:" });
        assert!(matches!(
            outcome,
            Err(RuntimeError::PodVerificationFailed { .. })
        ));
        let counters = runtime.reflex_counters();
        assert_eq!(counters.hits(), 0);
        assert_eq!(counters.misses(), 0);
        assert_eq!(counters.rejected(), 1);
    }

    #[test]
    fn failed_requests_are_in_the_latency_window() {
        let (empty, _) = registry(&[]);
        let (pods, _) = registry(&[spec("primary", Behaviour::Echo)]);
        let mut runtime = runtime();
        run_plain(&mut runtime, "r1", &pods, &PASS).unwrap();
        run_plain(&mut runtime, "r2", &empty, &PASS).unwrap_err();
        let latency = runtime.reflex_counters().latency().clone();
        assert_eq!(latency.recorded(), 2);
        assert_eq!(latency.failed(), 1);
        assert_eq!(latency.failed_in_window(), 1);
    }

    #[test]
    fn a_replayed_runtime_starts_with_no_counters_and_no_plans() {
        let (pods, _) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Prefix("fb:")),
        ]);
        let mut original = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut original, &fallback);
        plan(&mut original, "fallback", 2, fallback);
        run_routed(&mut original, "r1", &pods, &low_score_router(), &PASS).unwrap();
        assert!(original.reflex_counters().all_misses_covered());

        let replayed = PtrRuntime::replay(PtrConfig::default(), original.committed_events())
            .expect("the log replays");
        let counters = replayed.reflex_counters();
        // Replay reads the ledger; the counters are not in it, so nothing is claimed.
        assert_eq!(counters.coverage(), FailoverCoverage::NoObservations);
        assert!(!counters.all_misses_covered());
        // Neither is the plan: a restarted runtime does not fail over until the host sets it.
        assert_eq!(replayed.fallback_plan(&project(), &capability()), None);
    }

    #[test]
    fn the_snapshot_is_a_copy_not_a_handle() {
        let (pods, _) = registry(&[spec("primary", Behaviour::Echo)]);
        let mut runtime = runtime();
        let before = runtime.reflex_counters();
        run_plain(&mut runtime, "r1", &pods, &PASS).unwrap();
        assert_eq!(before.hits(), 0);
        assert_eq!(runtime.reflex_counters().hits(), 1);
    }

    #[test]
    fn counters_do_not_change_what_the_default_pod_checks() {
        // Many covered misses do not buy the default an exemption: the next miss is judged by
        // the live generation again.
        let (pods, _) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback.clone());
        for index in 0..5 {
            run_routed(
                &mut runtime,
                &format!("r{index}"),
                &pods,
                &low_score_router(),
                &PASS,
            )
            .unwrap();
        }
        assert_eq!(runtime.reflex_counters().failovers(), 5);
        revoke(&mut runtime, &fallback);
        assert!(run_routed(&mut runtime, "r9", &pods, &low_score_router(), &PASS).is_err());
    }
}

mod configuration {
    use super::*;

    fn request(
        default: Option<&str>,
        default_revision: Option<PodRevisionAddress>,
    ) -> SetFallbackPlan {
        SetFallbackPlan {
            project: project(),
            capability: capability(),
            plan: FallbackPlan::new(default.map(PodId::from), 2).unwrap(),
            default_revision,
        }
    }

    fn refused(runtime: &mut PtrRuntime, request: SetFallbackPlan) -> FallbackConfigError {
        match runtime.set_fallback_plan(request) {
            Err(RuntimeError::FallbackConfig(error)) => error,
            other => panic!("expected a configuration refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_default_without_a_revision_is_refused() {
        let mut runtime = runtime();
        assert_eq!(
            refused(&mut runtime, request(Some("fallback"), None)),
            FallbackConfigError::MissingDefaultRevision {
                pod: PodId::from("fallback")
            }
        );
    }

    #[test]
    fn a_revision_without_a_default_is_refused() {
        let mut runtime = runtime();
        assert_eq!(
            refused(&mut runtime, request(None, Some(revision("fallback", 1)))),
            FallbackConfigError::UnexpectedDefaultRevision
        );
    }

    #[test]
    fn a_revision_of_another_pod_names_both() {
        let mut runtime = runtime();
        assert_eq!(
            refused(
                &mut runtime,
                request(Some("fallback"), Some(revision("other", 1)))
            ),
            FallbackConfigError::PodMismatch {
                plan: PodId::from("fallback"),
                revision: PodId::from("other"),
            }
        );
    }

    #[test]
    fn a_revision_of_another_project_names_both() {
        let mut runtime = runtime();
        let foreign = PodRevisionAddress {
            address: PodAddress::new(
                ProjectId::from("elsewhere"),
                NamespaceId::from("ns"),
                PodId::from("fallback"),
            )
            .unwrap(),
            semantic_revision: [2; 32],
            generation: Generation(1),
        };
        assert_eq!(
            refused(&mut runtime, request(Some("fallback"), Some(foreign))),
            FallbackConfigError::ProjectMismatch {
                expected: project(),
                actual: ProjectId::from("elsewhere"),
            }
        );
    }

    #[test]
    fn a_malformed_revision_is_refused() {
        let mut runtime = runtime();
        let malformed = PodRevisionAddress {
            semantic_revision: [0; 32],
            ..revision("fallback", 1)
        };
        assert_eq!(
            refused(&mut runtime, request(Some("fallback"), Some(malformed))),
            FallbackConfigError::InvalidRevision
        );
    }

    #[test]
    fn the_number_of_plans_is_bounded_but_an_existing_plan_can_be_replaced() {
        let mut runtime = runtime();
        for index in 0..MAX_FALLBACK_PLANS {
            runtime
                .set_fallback_plan(SetFallbackPlan {
                    project: project(),
                    capability: CapabilityId::from(format!("cap{index}").as_str()),
                    plan: FallbackPlan::new(None, 1).unwrap(),
                    default_revision: None,
                })
                .unwrap();
        }
        let over = SetFallbackPlan {
            project: project(),
            capability: CapabilityId::from("one-too-many"),
            plan: FallbackPlan::new(None, 1).unwrap(),
            default_revision: None,
        };
        assert_eq!(
            refused(&mut runtime, over),
            FallbackConfigError::TooManyPlans {
                limit: MAX_FALLBACK_PLANS
            }
        );
        runtime
            .set_fallback_plan(SetFallbackPlan {
                project: project(),
                capability: CapabilityId::from("cap0"),
                plan: FallbackPlan::new(None, 3).unwrap(),
                default_revision: None,
            })
            .unwrap();
        assert_eq!(
            runtime
                .fallback_plan(&project(), &CapabilityId::from("cap0"))
                .map(FallbackPlan::max_attempts),
            Some(3)
        );
    }

    #[test]
    fn a_cleared_plan_stops_failing_over() {
        let (pods, _) = registry(&[
            spec("primary", Behaviour::Echo),
            spec("fallback", Behaviour::Prefix("fb:")),
        ]);
        let mut runtime = runtime();
        let fallback = revision("fallback", 1);
        make_live(&mut runtime, &fallback);
        plan(&mut runtime, "fallback", 2, fallback);
        assert!(runtime.clear_fallback_plan(&project(), &capability()));
        assert!(!runtime.clear_fallback_plan(&project(), &capability()));
        assert!(run_routed(&mut runtime, "r1", &pods, &low_score_router(), &PASS).is_err());
    }
}

#[test]
fn the_plain_router_is_unchanged_when_nothing_misses() {
    // PodRouter, wrapped, still picks what select picks and is a hit.
    let (pods, counts) = registry(&[
        spec("zeta", Behaviour::Echo),
        spec("alpha", Behaviour::Echo),
    ]);
    let mut runtime = runtime();
    run_routed(&mut runtime, "r1", &pods, &checked(PodRouter), &PASS).unwrap();
    assert_eq!(invocations(&counts, "zeta"), 1);
    assert_eq!(invocations(&counts, "alpha"), 0);
    assert_eq!(runtime.reflex_counters().hits(), 1);
}
