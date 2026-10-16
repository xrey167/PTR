//! A router's output as a prefetch signal: it plans residency and nothing else. A target the
//! caller's eligibility source rejects is dropped, one without a grant is never moved, and
//! the controller's state after planning is what it was before.

use ptr_router::{
    CodeRoute, EligibilitySnapshot, Exploration, FallbackPolicy, FindingCode, FindingRegistry,
    FindingRouteRequest, FindingRouter, NonceSequence, RegistryChange, RegistryUpdate,
    RouteDecision, RouteOutcome, RouteScore, RouterConfig, RouterLimits, SamplingKey,
    UniformScorer,
};
use ptr_runtime::{
    plan_prefetch, EligibleTargets, LookaheadPrefetchPolicy, ManualTierClock, PodEligibilityGate,
    PrefetchDropReason, PrefetchObjectIndex, PrefetchOutcome, PrefetchPlanRequest, PrefetchSignal,
    PrefetchSignalError, PrefetchSignalSource, PrefetchTarget, ReplicaState, TierClock, TierGrant,
    TierGrantLimits, TierResidencyController, TierRuntimeError, MAX_PREFETCH_TARGETS,
};
use ptr_storage::{
    ChunkDescriptor, ChunkReceipt, CpuTierBackend, StorageTier, TierBackend, TierBackendId,
    TierCapabilities, TierFuture, TierHealth, TierIntegrityReport, TierObjectDomain,
    TierObjectManifest,
};
use ptr_types::{
    Digest, Generation, PodId, Probability, ProjectId, RequestId, Revision, SessionId,
};
use ptr_verifier::{Finding, VerificationReport, VerificationStatus};
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// A CPU backend that reports a chosen tier.
struct TieredBackend {
    inner: CpuTierBackend,
    tier: StorageTier,
}

impl TieredBackend {
    fn new(id: &str, tier: StorageTier) -> Arc<Self> {
        Arc::new(Self {
            inner: CpuTierBackend::new(TierBackendId::from(id)),
            tier,
        })
    }
}

impl TierBackend for TieredBackend {
    fn identity(&self) -> TierBackendId {
        self.inner.identity()
    }

    fn capabilities(&self) -> TierCapabilities {
        let mut capabilities = self.inner.capabilities();
        capabilities.tier = self.tier;
        capabilities
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        self.inner.put_chunk(chunk, bytes)
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        self.inner.get_chunk(chunk)
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        self.inner.verify_chunk(chunk)
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        self.inner.delete_chunk(chunk)
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        self.inner.health()
    }
}

const SOURCE: &str = "ram";
const DESTINATION: &str = "gpu";
const ARTIFACT: Digest = [5; 32];

/// Per object: its replicas as (backend, state, pins).
type Snapshot = Vec<(Digest, Vec<(TierBackendId, ReplicaState, u64)>)>;

struct World {
    controller: TierResidencyController,
    clock: Arc<ManualTierClock>,
    source: Arc<TieredBackend>,
    index: PrefetchObjectIndex,
    experts: Vec<(u32, TierObjectManifest)>,
}

fn id(name: &str) -> TierBackendId {
    TierBackendId::from(name)
}

fn project() -> ProjectId {
    ProjectId::from("alpha")
}

fn expert(expert_id: u32) -> PrefetchTarget {
    PrefetchTarget::Expert {
        artifact_digest: ARTIFACT,
        expert_id,
    }
}

/// Four experts of one artifact registered under project alpha on the RAM tier, plus one
/// expert of project beta.
fn world() -> World {
    let clock = Arc::new(ManualTierClock::new(1_000));
    let controller = TierResidencyController::default()
        .with_grants(
            clock.clone() as Arc<dyn TierClock>,
            TierGrantLimits::default(),
        )
        .unwrap();
    let source = TieredBackend::new(SOURCE, StorageTier::Cpu);
    let destination = TieredBackend::new(DESTINATION, StorageTier::Gpu);
    block_on(controller.attach_backend(source.clone())).unwrap();
    block_on(controller.attach_backend(destination)).unwrap();
    let mut index = PrefetchObjectIndex::default();
    let mut experts = Vec::new();
    for expert_id in 0..4u32 {
        let manifest = register(
            &controller,
            &source,
            "alpha",
            &format!("expert-{expert_id}"),
            expert_id,
        );
        index
            .insert(project(), expert(expert_id), manifest.root_digest)
            .unwrap();
        experts.push((expert_id, manifest));
    }
    World {
        controller,
        clock,
        source,
        index,
        experts,
    }
}

fn register(
    controller: &TierResidencyController,
    source: &Arc<TieredBackend>,
    project: &str,
    logical_id: &str,
    seed: u32,
) -> TierObjectManifest {
    let bytes = format!("{project}-{logical_id}-{seed}-weights").into_bytes();
    let manifest = TierObjectManifest::new(
        TierObjectDomain::ModelWeight,
        logical_id,
        Generation(1),
        Revision(1),
        [3; 32],
        vec![ChunkDescriptor::for_bytes(0, 0, &bytes)],
    )
    .unwrap();
    block_on(source.put_chunk(manifest.chunks[0].clone(), bytes)).unwrap();
    block_on(controller.register_project_object(
        ProjectId::from(project),
        manifest.clone(),
        id(SOURCE),
    ))
    .unwrap();
    manifest
}

impl World {
    fn object(&self, expert_id: u32) -> Digest {
        self.experts
            .iter()
            .find(|(candidate, _)| *candidate == expert_id)
            .unwrap()
            .1
            .root_digest
    }

    fn grant(&self, expert_id: u32, nonce: u128) -> TierGrant {
        TierGrant {
            project: project(),
            object: self.object(expert_id),
            source: id(SOURCE),
            destination: id(DESTINATION),
            write_authorization: None,
            receiver: SessionId::from("worker-1"),
            nonce,
            deadline_millis: self.clock.now_millis() + 10_000,
        }
    }

    /// Everything observable about residency, to compare before and after planning.
    fn snapshot(&self) -> Snapshot {
        let mut digests: Vec<_> = self.experts.iter().map(|(_, m)| m.root_digest).collect();
        digests.sort();
        digests
            .into_iter()
            .map(|digest| {
                let record = self.controller.residency(&digest).unwrap();
                (
                    digest,
                    record
                        .replicas
                        .iter()
                        .map(|replica| (replica.backend.clone(), replica.state, replica.pins))
                        .collect(),
                )
            })
            .collect()
    }

    fn plan(
        &self,
        signal: &PrefetchSignal,
        eligibility: &dyn ptr_runtime::PrefetchEligibility,
        max_objects: usize,
    ) -> ptr_runtime::PrefetchPlan {
        plan_prefetch(&PrefetchPlanRequest {
            signal,
            index: &self.index,
            eligibility,
            controller: &self.controller,
            policy: &LookaheadPrefetchPolicy { max_objects },
            requested_tiers: vec![StorageTier::Gpu],
        })
        .unwrap()
    }
}

fn allow(experts: &[u32]) -> EligibleTargets {
    let mut allowed = EligibleTargets::default();
    for expert_id in experts {
        allowed.allow(project(), expert(*expert_id));
    }
    allowed
}

fn ranking(experts: &[u32]) -> PrefetchSignal {
    PrefetchSignal::from_expert_ranking(project(), ARTIFACT, experts.iter().copied()).unwrap()
}

#[test]
fn a_top_k_signal_becomes_a_plan_in_ranked_order_limited_by_the_policy() {
    let world = world();
    let before = world.snapshot();
    let plan = world.plan(&ranking(&[2, 0, 3, 1]), &allow(&[0, 1, 2, 3]), 2);
    assert_eq!(
        plan.items
            .iter()
            .map(|item| item.object)
            .collect::<Vec<_>>(),
        vec![world.object(2), world.object(0)]
    );
    assert_eq!(plan.destinations, vec![StorageTier::Gpu]);
    let not_chosen: Vec<_> = plan
        .dropped
        .iter()
        .filter(|drop| drop.reason == PrefetchDropReason::NotChosen)
        .map(|drop| drop.object.unwrap())
        .collect();
    assert_eq!(not_chosen, vec![world.object(3), world.object(1)]);
    assert_eq!(world.snapshot(), before, "planning changes no residency");
}

#[test]
fn a_signal_naming_an_ineligible_expert_changes_nothing_and_admits_nothing() {
    let world = world();
    let before = world.snapshot();
    // The router ranked expert 1 first, but the eligibility source does not allow it.
    let signal = ranking(&[1, 2]);
    let plan = world.plan(&signal, &allow(&[2]), 8);
    assert_eq!(
        plan.items
            .iter()
            .map(|item| item.target.clone())
            .collect::<Vec<_>>(),
        vec![expert(2)]
    );
    assert!(plan
        .dropped
        .iter()
        .any(|drop| { drop.target == expert(1) && drop.reason == PrefetchDropReason::Ineligible }));
    // Executing the plan with a grant for the ineligible expert's object still does not move
    // it: the plan decides what is tried, and expert 1 is not in it.
    let grants = [world.grant(1, 1), world.grant(2, 2)];
    let outcomes = block_on(world.controller.execute_prefetch_plan(
        &plan,
        &grants,
        &SessionId::from("worker-1"),
    ));
    assert_eq!(outcomes.len(), 1);
    assert!(matches!(outcomes[0].1, PrefetchOutcome::Ready(_)));
    let after = world.snapshot();
    let expert1 = after
        .iter()
        .find(|(digest, _)| *digest == world.object(1))
        .unwrap();
    assert_eq!(
        expert1.1.len(),
        1,
        "no replica was created for the ineligible expert"
    );
    for (digest, replicas) in &after {
        assert!(
            replicas.iter().all(|(_, _, pins)| *pins == 0),
            "{digest:?}: no pin, so nothing was admitted"
        );
    }
    let before_other: Vec<_> = before
        .iter()
        .filter(|(digest, _)| *digest != world.object(2))
        .collect();
    let after_other: Vec<_> = after
        .iter()
        .filter(|(digest, _)| *digest != world.object(2))
        .collect();
    assert_eq!(before_other, after_other, "only the planned object moved");
}

#[test]
fn a_signal_never_adds_eligibility_even_when_it_names_every_target() {
    let world = world();
    let before = world.snapshot();
    let plan = world.plan(&ranking(&[0, 1, 2, 3]), &allow(&[]), 8);
    assert!(plan.items.is_empty());
    assert_eq!(plan.dropped.len(), 4);
    assert!(plan
        .dropped
        .iter()
        .all(|drop| drop.reason == PrefetchDropReason::Ineligible));
    assert_eq!(world.snapshot(), before);
}

#[test]
fn an_eligible_expert_without_a_grant_is_planned_but_never_moved() {
    let world = world();
    let before = world.snapshot();
    let plan = world.plan(&ranking(&[0, 1]), &allow(&[0, 1]), 8);
    assert_eq!(plan.items.len(), 2);
    // Only expert 0 has a grant.
    let grants = [world.grant(0, 1)];
    let outcomes = block_on(world.controller.execute_prefetch_plan(
        &plan,
        &grants,
        &SessionId::from("worker-1"),
    ));
    assert!(matches!(outcomes[0].1, PrefetchOutcome::Ready(_)));
    assert_eq!(
        outcomes[1].1,
        PrefetchOutcome::Skipped(TierRuntimeError::GrantRequired)
    );
    let after = world.snapshot();
    let find = |snapshot: &Snapshot, digest| {
        snapshot
            .iter()
            .find(|(candidate, _)| *candidate == digest)
            .unwrap()
            .1
            .clone()
    };
    assert_eq!(
        find(&after, world.object(1)),
        find(&before, world.object(1)),
        "the unauthorized expert's residency is untouched"
    );
    assert_eq!(find(&after, world.object(0)).len(), 2);
}

#[test]
fn a_grant_of_another_project_cannot_carry_a_prefetch() {
    let world = world();
    let plan = world.plan(&ranking(&[0]), &allow(&[0]), 8);
    let mut foreign = world.grant(0, 1);
    foreign.project = ProjectId::from("beta");
    let outcomes = block_on(world.controller.execute_prefetch_plan(
        &plan,
        &[foreign],
        &SessionId::from("worker-1"),
    ));
    assert_eq!(
        outcomes[0].1,
        PrefetchOutcome::Skipped(TierRuntimeError::ProjectMismatch)
    );
    assert_eq!(
        world
            .controller
            .residency(&world.object(0))
            .unwrap()
            .replicas
            .len(),
        1
    );
}

#[test]
fn objects_of_another_project_are_never_planned_from_this_projects_signal() {
    let mut world = world();
    let beta = register(&world.controller, &world.source, "beta", "beta-expert", 9);
    // A stale or hostile index entry pointing project alpha's target at beta's object.
    world
        .index
        .insert(project(), expert(9), beta.root_digest)
        .unwrap();
    let plan = world.plan(&ranking(&[9]), &allow(&[9]), 8);
    assert!(plan.items.is_empty());
    assert_eq!(plan.dropped[0].reason, PrefetchDropReason::NotInNamespace);
    // And an index that holds nothing for the project resolves nothing.
    let other =
        PrefetchSignal::from_expert_ranking(ProjectId::from("beta"), ARTIFACT, [0u32]).unwrap();
    let mut beta_allowed = EligibleTargets::default();
    beta_allowed.allow(ProjectId::from("beta"), expert(0));
    let plan = world.plan(&other, &beta_allowed, 8);
    assert_eq!(plan.dropped[0].reason, PrefetchDropReason::Unresolved);
}

#[test]
fn an_already_resident_object_is_not_planned_again() {
    let world = world();
    block_on(
        world
            .controller
            .transfer_with_grant(&world.grant(0, 1), &SessionId::from("worker-1")),
    )
    .unwrap();
    let plan = world.plan(&ranking(&[0, 1]), &allow(&[0, 1]), 8);
    assert_eq!(plan.items.len(), 1);
    assert_eq!(plan.items[0].object, world.object(1));
    assert!(plan.dropped.iter().any(|drop| {
        drop.object == Some(world.object(0)) && drop.reason == PrefetchDropReason::AlreadyResident
    }));
}

// ---- signals from the router ----

fn route_decision() -> RouteDecision {
    let score = |target: &str, value: f32| RouteScore {
        target: target.into(),
        score: Probability::new(value).unwrap(),
    };
    RouteDecision {
        candidates: vec![
            score("pod-b", 0.5),
            score("pod-a", 0.5),
            score("pod-c", 0.9),
            score("pod-d", 0.1),
        ],
    }
}

#[test]
fn a_route_decision_gives_its_top_k_best_score_first_ties_by_name() {
    let signal = PrefetchSignal::from_route_decision(project(), &route_decision(), 3).unwrap();
    assert_eq!(signal.source(), PrefetchSignalSource::RouterTopK);
    assert_eq!(
        signal.targets(),
        &[
            PrefetchTarget::Pod(PodId::from("pod-c")),
            PrefetchTarget::Pod(PodId::from("pod-a")),
            PrefetchTarget::Pod(PodId::from("pod-b")),
        ]
    );
}

#[test]
fn a_finding_router_record_gives_its_eligible_set_and_eligibility_is_asked_again() {
    let mut registry = FindingRegistry::new();
    for pod in ["fix-a", "fix-b", "fix-c"] {
        let expected = registry.generation();
        registry
            .apply(RegistryUpdate {
                expected,
                change: RegistryChange::AddRepair {
                    code: FindingCode::new("target"),
                    pod: PodId::from(pod),
                },
            })
            .unwrap();
    }
    let router = FindingRouter::new(
        registry,
        RouterConfig {
            fallback: FallbackPolicy::default(),
            limits: RouterLimits::default(),
            exploration: Exploration::NONE,
        },
        UniformScorer,
        SamplingKey::from_bytes([7; 32]),
        NonceSequence::starting_at([1; 8], 0),
    )
    .unwrap();
    let mut eligible = EligibilitySnapshot::new();
    for pod in ["fix-a", "fix-b", "fix-c"] {
        eligible.set(PodId::from(pod), Generation(1));
    }
    let report = VerificationReport {
        status: VerificationStatus::Fail,
        level: ptr_types::VerificationLevel::Deterministic,
        score: Probability::new(0.5).unwrap(),
        findings: vec![Finding {
            code: "target".into(),
            message: String::new(),
            hard: true,
        }],
    };
    let request_id = RequestId::from("req-1");
    let outcome = router
        .route(&FindingRouteRequest {
            request_id: &request_id,
            request_digest: [1; 32],
            report: &report,
            required: ptr_types::VerificationLevel::Deterministic,
            eligibility: &eligible,
        })
        .unwrap();
    let RouteOutcome::Repair { routes } = outcome else {
        panic!("a hard finding is a repair");
    };
    let CodeRoute::Routed(record) = &routes[0] else {
        panic!("a repair Pod is registered");
    };
    let signal = PrefetchSignal::from_finding_route(project(), record).unwrap();
    assert_eq!(signal.source(), PrefetchSignalSource::FindingEligibleSet);
    assert_eq!(signal.targets().len(), 3);
    assert_eq!(
        signal.targets()[0],
        PrefetchTarget::Pod(record.pod().clone())
    );

    // The registry revoked fix-b after the route was sampled. The signal still names it; the
    // gate does not let it through.
    eligible.revoke(&PodId::from("fix-b"));
    let gate = PodEligibilityGate(&eligible);
    let mut index = PrefetchObjectIndex::default();
    let world = world();
    for (n, pod) in ["fix-a", "fix-b", "fix-c"].iter().enumerate() {
        index
            .insert(
                project(),
                PrefetchTarget::Pod(PodId::from(*pod)),
                world.object(n as u32),
            )
            .unwrap();
    }
    let plan = plan_prefetch(&PrefetchPlanRequest {
        signal: &signal,
        index: &index,
        eligibility: &gate,
        controller: &world.controller,
        policy: &LookaheadPrefetchPolicy { max_objects: 8 },
        requested_tiers: vec![StorageTier::Gpu],
    })
    .unwrap();
    assert_eq!(plan.items.len(), 2);
    assert!(plan
        .items
        .iter()
        .all(|item| { item.target != PrefetchTarget::Pod(PodId::from("fix-b")) }));
    assert!(plan.dropped.iter().any(|drop| {
        drop.target == PrefetchTarget::Pod(PodId::from("fix-b"))
            && drop.reason == PrefetchDropReason::Ineligible
    }));
}

#[test]
fn a_signal_is_bounded_and_deduplicated() {
    let many = (0..=MAX_PREFETCH_TARGETS as u32).map(expert);
    assert_eq!(
        PrefetchSignal::new(PrefetchSignalSource::ExpertRanking, project(), many),
        Err(PrefetchSignalError::TooManyTargets)
    );
    let repeated = ranking(&[1, 1, 2, 1]);
    assert_eq!(repeated.targets(), &[expert(1), expert(2)]);
    assert_eq!(
        PrefetchSignal::new(
            PrefetchSignalSource::ExpertRanking,
            ProjectId::from(" "),
            []
        ),
        Err(PrefetchSignalError::EmptyProject)
    );
}
