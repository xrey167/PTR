//! The version 2 downstream stage against a real ledger and a real placement controller: an
//! entry derived from a version 2 frame is served while its chain, its typed inputs and its
//! own stage are live, refused the moment the ledger revokes or supersedes any one of them (or
//! an expert is fenced through the real authority path), whether or not anyone told the cache;
//! an Approximate frame lives in its own namespace and its taint reaches the output; and
//! nothing persists. The expert fence epoch is the runtime's own (`RuntimeStageLiveness`
//! answers it from the expert registry).

use ptr_config::PtrConfig;
use ptr_ledger::{ExpertFenceCause, LedgerEvent};
use ptr_lineage::{
    AccuracyMatrix, AdapterId, BoundGateReport, ForgettingGate, GateSubject, PublicSuite,
};
use ptr_model_api::{KvReuseMode, KvReusePolicy, KvReuseTaint};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertLevel, ExpertPodSpec, ExpertProposal,
    ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef, RationaleDigest, RetiredReason,
    StructuralOp,
};
use ptr_protocol::activation::{
    admit_v2, bijection_violations, chunk_digest_v2, chunk_payload_v2, frame_digest_v2,
    ActivationChunkV2, ActivationFrameV2, AllowlistedPayloadV2, BoundaryDtype, ChainElement,
    ChunkHeaderV2, DependencyKind, FrameMode, FrameTaint, InputLineage, LineageInput, LineageKind,
    UpstreamChain,
};
use ptr_runtime::expert::{
    expert_evidence_artifact_digest, expert_evidence_base_model_digest, ExpertAdmissionGrant,
    ExpertFenceGrant, ExpertFenceRequest,
};
use ptr_runtime::{
    admit_frame_v2, input_target, AdmissionRequestV2, ChainFault, DeviceHealth, DeviceRecord,
    DownstreamKey, FrameFault, InputLiveness, NodeHealth, NodeRecord, PersistOp,
    PodPlacementController, PtrRuntime, RuntimeInputLiveness, RuntimeStageLiveness,
    SplitStageCacheV2, StageLiveness, UseFaultV2,
};
use ptr_types::{
    ArtifactId, CapsuleId, CommitIndex, DeviceId, Generation, NodeId, PodId, ProjectId, Timestamp,
    Validity,
};
use std::collections::BTreeMap;

mod expert_support;

use expert_support::{binding, prime, requirement, EVAL_MANIFEST};

/// Chain dependencies the ledger tracks as capsules. The expert is not one: its generation,
/// state, manifest digest and fence epoch all come from the expert registry.
const TARGETS: [&str; 4] = ["stage", "model", "adapter", "down"];
const INPUTS: [(LineageKind, &str); 3] = [
    (LineageKind::Semantic, "sem"),
    (LineageKind::Memory, "mem"),
    (LineageKind::Derived, "drv"),
];
const EXPERT: &str = "expert:proj/x";

fn digest(name: &str, generation: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = name.len() as u8;
    out[1] = generation as u8;
    out[2] = name.as_bytes()[0];
    out
}

fn project() -> ProjectId {
    ProjectId::from("proj")
}

fn gate() -> ForgettingGate {
    ForgettingGate {
        max_average_forgetting: 0.5,
        max_task_forgetting: 0.5,
        min_backward_transfer: -0.5,
        max_public_regression: 0.02,
    }
}

/// The spec of expert `x` at `generation` (manifest digest `[generation; 32]`).
fn expert_spec(generation: u64) -> ExpertPodSpec {
    let base = BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
        1,
    )
    .unwrap();
    ExpertPodSpec::new(
        project(),
        PodId::from("x"),
        ManifestDigest::new([generation as u8; 32]).unwrap(),
        ExpertLevel::Token { layer: 3, index: 1 },
        base,
        Generation(generation),
    )
    .unwrap()
}

/// Birth the expert `x` through the real admission path: it serves at state version 1.
fn birth_expert(rt: &mut PtrRuntime) {
    prime(rt);
    admit_expert_op(
        rt,
        StructuralOp::Birth {
            result: expert_spec(1),
        },
    );
}

/// The registry's current view of expert `x` as an operand.
fn expert_target(rt: &PtrRuntime) -> ExpertTarget {
    let entry = rt
        .expert_registry()
        .get(&project(), &PodId::from("x"))
        .unwrap()
        .clone();
    ExpertTarget::new(ExpertTargetInit {
        project: project(),
        pod: PodId::from("x"),
        generation: entry.generation,
        level: entry.level,
        base_model: entry.base_model,
        manifest_digest: entry.manifest_digest,
        state: entry.state,
        state_version: entry.state_version,
    })
    .unwrap()
}

/// Revoke expert `x` through the registry: a committed `Retire`, and with `supersede` a
/// `Reactivate` that commits its generation 2.
fn invalidate_expert(rt: &mut PtrRuntime, supersede: bool) {
    admit_expert_op(
        rt,
        StructuralOp::Retire {
            target: expert_target(rt),
            reason: RetiredReason::GateFailed,
        },
    );
    if supersede {
        admit_expert_op(
            rt,
            StructuralOp::Reactivate {
                target: expert_target(rt),
                result: Some(expert_spec(2)),
            },
        );
    }
}

/// Admit one structural operation on expert `x` through the real admission path.
fn admit_expert_op(rt: &mut PtrRuntime, op: StructuralOp) {
    let proposer = ProposerRef::new(project(), PodId::from("meta"), Generation(1)).unwrap();
    let proposal = ExpertProposal::new(
        project(),
        proposer.clone(),
        CommitIndex(1),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    let subject = GateSubject::new(
        *proposal.proposal_id().as_bytes(),
        *proposal.operation_id().as_bytes(),
        expert_evidence_artifact_digest(proposal.op()),
        expert_evidence_base_model_digest(proposal.op()).unwrap(),
        EVAL_MANIFEST,
    );
    let matrix = AccuracyMatrix::new(vec![vec![0.9, 0.0], vec![0.9, 0.9]]).unwrap();
    let report: BoundGateReport = gate()
        .evaluate(
            &AdapterId::from("candidate"),
            &matrix,
            PublicSuite {
                serving: 0.8,
                candidate: 0.8,
            },
        )
        .unwrap()
        .bind(subject);
    let grant = ExpertAdmissionGrant::new(
        project(),
        proposer,
        requirement(7, gate().config_digest().unwrap()),
        binding(),
    );
    rt.admit_expert_proposal(&proposal, &report, &grant)
        .unwrap();
}

struct World {
    runtime: PtrRuntime,
    placement: PodPlacementController,
    manifests: BTreeMap<(String, u64), [u8; 32]>,
    input_digests: BTreeMap<(LineageKind, String, u64), [u8; 32]>,
}

fn world() -> World {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let capsules = TARGETS
        .iter()
        .map(|name| (*name).to_owned())
        .chain(INPUTS.iter().map(|(kind, name)| input_target(*kind, name)));
    for name in capsules {
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from(name.as_str()),
                generation: Generation(1),
            })
            .unwrap();
    }
    birth_expert(&mut runtime);
    let mut placement = PodPlacementController::default();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("n1"),
            zone: "z".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cpu"),
                vram_bytes: 10,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .unwrap();
    placement
        .assign(
            PodId::from("stage"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("n1"),
            DeviceId::from("cpu"),
            1,
        )
        .unwrap();
    World {
        runtime,
        placement,
        manifests: TARGETS
            .iter()
            .map(|name| (((*name).to_owned(), 1), digest(name, 1)))
            .collect(),
        input_digests: INPUTS
            .iter()
            .map(|(kind, name)| (((*kind), (*name).to_owned(), 1), digest(name, 1)))
            .collect(),
    }
}

/// The runtime's own stage liveness (expert fence epochs included) and input liveness.
struct Live<'a> {
    stage: RuntimeStageLiveness<'a>,
    inputs: RuntimeInputLiveness<'a>,
}

impl StageLiveness for Live<'_> {
    fn validity(&self, target: &str, generation: Generation) -> Option<Validity> {
        self.stage.validity(target, generation)
    }

    fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
        self.stage.manifest_digest(target, generation)
    }

    fn placement_epoch(&self, stage: &str) -> Option<u64> {
        self.stage.placement_epoch(stage)
    }

    fn fence_epoch(&self, expert: &str) -> Option<u64> {
        self.stage.fence_epoch(expert)
    }
}

impl InputLiveness for Live<'_> {
    fn validity(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<Validity> {
        self.inputs.validity(kind, identity, generation)
    }

    fn content_digest(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<[u8; 32]> {
        self.inputs.content_digest(kind, identity, generation)
    }
}

fn live(world: &World) -> Live<'_> {
    Live {
        stage: RuntimeStageLiveness {
            runtime: &world.runtime,
            placement: &world.placement,
            manifests: &world.manifests,
        },
        inputs: RuntimeInputLiveness {
            runtime: &world.runtime,
            digests: &world.input_digests,
        },
    }
}

fn element(name: &str) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: digest(name, 1),
        fence_epoch: 0,
    }
}

fn own_chain() -> UpstreamChain {
    UpstreamChain {
        stage: element("down"),
        base_model: element("model"),
        adapters: vec![],
        experts: vec![],
    }
}

fn input(kind: LineageKind, name: &str) -> LineageInput {
    LineageInput {
        kind,
        identity: name.to_owned(),
        generation: 1,
        digest: digest(name, 1),
    }
}

/// The expert's committed fence epoch: its state version in the registry.
fn expert_epoch(world: &World) -> u64 {
    world
        .runtime
        .expert_registry()
        .get(&project(), &PodId::from("x"))
        .unwrap()
        .state_version
}

fn frame(
    world: &World,
    mode: FrameMode,
    taint: FrameTaint,
    raw: Vec<[u8; 32]>,
) -> ActivationFrameV2 {
    ActivationFrameV2 {
        tokens: 1,
        width: 2,
        dtype: BoundaryDtype::F32,
        positions: vec![0],
        upstream_chain: UpstreamChain {
            stage: element("stage"),
            base_model: element("model"),
            adapters: vec![element("adapter")],
            experts: vec![ChainElement {
                // The registry's manifest digest of the expert's generation 1.
                digest: [1; 32],
                fence_epoch: expert_epoch(world),
                ..element(EXPERT)
            }],
        },
        input_lineage: InputLineage::canonical(
            INPUTS
                .iter()
                .map(|(kind, name)| input(*kind, name))
                .collect(),
            raw,
        ),
        mode,
        taint,
        placement_epoch: world
            .placement
            .placement(&PodId::from("stage"))
            .unwrap()
            .epoch
            .0,
        tensor: [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn exact(world: &World) -> ActivationFrameV2 {
    frame(world, FrameMode::Exact, FrameTaint::Exact, vec![[1; 32]])
}

fn approximate(world: &World) -> ActivationFrameV2 {
    frame(
        world,
        FrameMode::Approximate,
        FrameTaint::Approximate {
            report_digest: [6; 32],
        },
        vec![[1; 32]],
    )
}

/// The ledger target of a dependency: an input is a target of its kind.
fn ledger_target(subject: &str) -> String {
    INPUTS
        .iter()
        .find(|(_, name)| *name == subject)
        .map_or_else(
            || subject.to_owned(),
            |(kind, name)| input_target(*kind, name),
        )
}

fn invalidate(world: &mut World, subject: &str, supersede: bool) {
    if subject == EXPERT {
        return invalidate_expert(&mut world.runtime, supersede);
    }
    let target = ledger_target(subject);
    let event = if supersede {
        LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from(target.as_str()),
            old: Generation(1),
            new: Generation(2),
        }
    } else {
        LedgerEvent::Revoked {
            subject: target,
            generation: Generation(1),
        }
    };
    world.runtime.commit(event).unwrap();
}

/// Fence the expert through the real authority path (`fence_expert` under the revocation
/// authority's grant): it goes to `Fenced` at the next epoch.
fn fence_expert(world: &mut World) {
    let request = ExpertFenceRequest {
        project: project(),
        pod: PodId::from("x"),
        generation: Generation(1),
        expected_state_version: expert_epoch(world),
        cause: ExpertFenceCause::AuthorityQuarantine,
        revocation: None,
    };
    world
        .runtime
        .fence_expert(
            &request,
            &ExpertFenceGrant::revocation(project(), binding()),
        )
        .unwrap();
}

#[test]
fn an_invalidation_the_cache_was_never_told_of_ends_the_entry_for_every_dependency() {
    for subject in [
        "stage", "model", "adapter", EXPERT, "down", "sem", "mem", "drv",
    ] {
        for supersede in [false, true] {
            let mut world = world();
            let mut cache: SplitStageCacheV2<&str> = SplitStageCacheV2::new(4);
            let key = {
                let live = live(&world);
                let admitted = admit_frame_v2(
                    AdmissionRequestV2 {
                        frame: &exact(&world),
                        policy: KvReusePolicy::ExactOnly,
                        expected_start: 0,
                    },
                    &live,
                    &live,
                )
                .unwrap();
                cache
                    .store(&[&admitted], &own_chain(), "kv", &live, &live)
                    .unwrap()
            };
            {
                let live = live(&world);
                let served = cache.lookup(&key, &live, &live).unwrap();
                assert_eq!(*served.value(), "kv");
                cache.abandon_use(served).unwrap();
            }
            invalidate(&mut world, subject, supersede);
            let live = live(&world);
            let verdict = cache.lookup(&key, &live, &live);
            assert!(
                matches!(verdict, Err(UseFaultV2::Stale(_))),
                "{subject} supersede={supersede}: {verdict:?}"
            );
            assert!(cache.is_empty());
            // A frame of the same lineage is refused as well, except when only the
            // downstream stage's own generation moved (the frame does not name it).
            let refused = admit_frame_v2(
                AdmissionRequestV2 {
                    frame: &exact(&world),
                    policy: KvReusePolicy::ExactOnly,
                    expected_start: 0,
                },
                &live,
                &live,
            )
            .is_err();
            assert_eq!(refused, subject != "down", "{subject}");
        }
    }
}

#[test]
fn a_fenced_expert_ends_the_entry_and_a_readmission_never_revives_it() {
    let mut world = world();
    assert_eq!(expert_epoch(&world), 1);
    let mut cache: SplitStageCacheV2<u8> = SplitStageCacheV2::new(4);
    let key = {
        let live = live(&world);
        // The runtime answers the epoch the chain recorded.
        assert_eq!(live.fence_epoch(EXPERT), Some(1));
        let admitted = admit_frame_v2(
            AdmissionRequestV2 {
                frame: &exact(&world),
                policy: KvReusePolicy::ExactOnly,
                expected_start: 0,
            },
            &live,
            &live,
        )
        .unwrap();
        cache
            .store(&[&admitted], &own_chain(), 1, &live, &live)
            .unwrap()
    };
    let served = {
        let live = live(&world);
        cache.lookup(&key, &live, &live).unwrap()
    };
    let old_frame = exact(&world);
    // The expert is fenced through the authority path while the use runs.
    fence_expert(&mut world);
    assert_eq!(expert_epoch(&world), 2);
    let live = live(&world);
    assert_eq!(
        live.fence_epoch(EXPERT),
        None,
        "a fenced expert serves nowhere"
    );
    assert!(matches!(
        cache.finish_use(served, 0, &live, &live),
        Err(UseFaultV2::Stale(FrameFault::Chain(
            ChainFault::StaleFence { current: None, .. }
        )))
    ));
    assert!(matches!(
        admit_frame_v2(
            AdmissionRequestV2 {
                frame: &old_frame,
                policy: KvReusePolicy::ExactOnly,
                expected_start: 0
            },
            &live,
            &live
        ),
        Err(FrameFault::Chain(ChainFault::StaleFence { .. }))
    ));
}

#[test]
fn a_chain_expert_is_admitted_only_at_the_committed_epoch() {
    let world = world();
    let live = live(&world);
    let mut wrong = exact(&world);
    wrong.upstream_chain.experts[0].fence_epoch = 5;
    assert!(matches!(
        admit_frame_v2(
            AdmissionRequestV2 {
                frame: &wrong,
                policy: KvReusePolicy::ExactOnly,
                expected_start: 0
            },
            &live,
            &live
        ),
        Err(FrameFault::Chain(ChainFault::StaleFence {
            recorded: 5,
            current: Some(1),
            ..
        }))
    ));
    let mut unrecorded = exact(&world);
    unrecorded.upstream_chain.experts[0].fence_epoch = 0;
    assert!(matches!(
        admit_frame_v2(
            AdmissionRequestV2 {
                frame: &unrecorded,
                policy: KvReusePolicy::ExactOnly,
                expected_start: 0
            },
            &live,
            &live
        ),
        Err(FrameFault::Chain(ChainFault::StaleFence {
            recorded: 0,
            ..
        }))
    ));
    // An expert the registry does not know fails closed, as does a malformed identity.
    for identity in ["expert:proj/unknown", "expert:proj", "x"] {
        assert_eq!(live.fence_epoch(identity), None, "{identity}");
    }
    assert!(admit_frame_v2(
        AdmissionRequestV2 {
            frame: &exact(&world),
            policy: KvReusePolicy::ExactOnly,
            expected_start: 0
        },
        &live,
        &live
    )
    .is_ok());
}

#[test]
fn an_approximate_frame_lives_in_its_own_namespace_and_its_taint_reaches_the_output() {
    let world = world();
    let live = live(&world);
    assert_eq!(
        admit_frame_v2(
            AdmissionRequestV2 {
                frame: &approximate(&world),
                policy: KvReusePolicy::ExactOnly,
                expected_start: 0
            },
            &live,
            &live
        ),
        Err(FrameFault::ApproximateNotAllowed)
    );
    let approximate = admit_frame_v2(
        AdmissionRequestV2 {
            frame: &approximate(&world),
            policy: KvReusePolicy::AllowApproximate,
            expected_start: 0,
        },
        &live,
        &live,
    )
    .unwrap();
    let exact = admit_frame_v2(
        AdmissionRequestV2 {
            frame: &exact(&world),
            policy: KvReusePolicy::ExactOnly,
            expected_start: 0,
        },
        &live,
        &live,
    )
    .unwrap();
    let mut cache: SplitStageCacheV2<&str> = SplitStageCacheV2::new(4);
    let approx_key = cache
        .store(&[&approximate], &own_chain(), "approx", &live, &live)
        .unwrap();
    assert_eq!(approx_key.namespace, KvReuseMode::Approximate);
    // The Exact lookup of the same digest, and of the exact frame's key, misses.
    let exact_view = DownstreamKey {
        namespace: KvReuseMode::Exact,
        digest: approx_key.digest,
    };
    assert!(matches!(
        cache.lookup(&exact_view, &live, &live),
        Err(UseFaultV2::Miss)
    ));
    let exact_key = ptr_runtime::prefix_key_v2(&[&exact], &own_chain());
    assert!(matches!(
        cache.lookup(&exact_key, &live, &live),
        Err(UseFaultV2::Miss)
    ));
    // The Approximate hit carries the taint to the released output.
    let served = cache.lookup(&approx_key, &live, &live).unwrap();
    let released = cache.finish_use(served, "out", &live, &live).unwrap();
    assert_eq!(
        released.into_parts(),
        (
            "out",
            KvReuseTaint::Approximate {
                report_digest: [6; 32]
            }
        )
    );
    let stored = cache
        .store(&[&exact], &own_chain(), "exact", &live, &live)
        .unwrap();
    let served = cache.lookup(&stored, &live, &live).unwrap();
    assert_eq!(
        (*served.value(), served.taint()),
        ("exact", KvReuseTaint::Exact)
    );
    cache.abandon_use(served).unwrap();
}

#[test]
fn an_approximate_hit_cannot_be_released_after_an_invalidation_between_lookup_and_finish() {
    let mut world = world();
    let mut cache: SplitStageCacheV2<&str> = SplitStageCacheV2::new(4);
    let key = {
        let live = live(&world);
        let approximate = admit_frame_v2(
            AdmissionRequestV2 {
                frame: &approximate(&world),
                policy: KvReusePolicy::AllowApproximate,
                expected_start: 0,
            },
            &live,
            &live,
        )
        .unwrap();
        cache
            .store(&[&approximate], &own_chain(), "approx", &live, &live)
            .unwrap()
    };
    // The hit is an open use that owns the only handle on the value; the taint is readable.
    let served = {
        let live = live(&world);
        cache.lookup(&key, &live, &live).unwrap()
    };
    assert!(served.taint().is_approximate());
    assert_eq!(cache.open_uses(), 1);
    let output = served.value().to_uppercase();
    // A source is invalidated without the Revoke message reaching the cache.
    invalidate(&mut world, "sem", false);
    let live = live(&world);
    assert!(matches!(
        cache.finish_use(served, output, &live, &live),
        Err(UseFaultV2::Stale(FrameFault::InputNotLive { .. }))
    ));
    assert_eq!(cache.open_uses(), 0);
}

#[test]
fn a_removed_raw_ancestor_is_a_miss_with_every_typed_input_chain_and_position_equal() {
    let world = world();
    let live = live(&world);
    let long = frame(
        &world,
        FrameMode::Exact,
        FrameTaint::Exact,
        vec![[1; 32], [2; 32]],
    );
    let short = frame(&world, FrameMode::Exact, FrameTaint::Exact, vec![[2; 32]]);
    assert_eq!(long.positions, short.positions);
    assert_eq!(long.upstream_chain, short.upstream_chain);
    assert_eq!(long.input_lineage.inputs, short.input_lineage.inputs);
    let mut cache: SplitStageCacheV2<u8> = SplitStageCacheV2::new(4);
    let long = admit_frame_v2(
        AdmissionRequestV2 {
            frame: &long,
            policy: KvReusePolicy::ExactOnly,
            expected_start: 0,
        },
        &live,
        &live,
    )
    .unwrap();
    let short = admit_frame_v2(
        AdmissionRequestV2 {
            frame: &short,
            policy: KvReusePolicy::ExactOnly,
            expected_start: 0,
        },
        &live,
        &live,
    )
    .unwrap();
    cache
        .store(&[&long], &own_chain(), 1, &live, &live)
        .unwrap();
    let probe = ptr_runtime::prefix_key_v2(&[&short], &own_chain());
    assert!(matches!(
        cache.lookup(&probe, &live, &live),
        Err(UseFaultV2::Miss)
    ));
}

#[test]
fn two_input_kinds_of_one_identity_are_two_lifecycle_targets() {
    let mut world = world();
    // A memory input with the identity of the semantic one: revoking the memory target leaves
    // the semantic one live.
    world
        .runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from(input_target(LineageKind::Memory, "sem").as_str()),
            generation: Generation(1),
        })
        .unwrap();
    world
        .input_digests
        .insert((LineageKind::Memory, "sem".to_owned(), 1), digest("sem", 1));
    world
        .runtime
        .commit(LedgerEvent::Revoked {
            subject: input_target(LineageKind::Memory, "sem"),
            generation: Generation(1),
        })
        .unwrap();
    let live = live(&world);
    let validity = |kind| InputLiveness::validity(&live, kind, "sem", Generation(1));
    assert_eq!(validity(LineageKind::Memory), Some(Validity::Revoked));
    assert_eq!(validity(LineageKind::Semantic), Some(Validity::Live));
    assert!(admit_frame_v2(
        AdmissionRequestV2 {
            frame: &exact(&world),
            policy: KvReusePolicy::ExactOnly,
            expected_start: 0
        },
        &live,
        &live
    )
    .is_ok());
}

#[test]
fn nothing_persists_and_a_restart_drops_every_entry() {
    let world = world();
    let live = live(&world);
    let admitted = admit_frame_v2(
        AdmissionRequestV2 {
            frame: &exact(&world),
            policy: KvReusePolicy::ExactOnly,
            expected_start: 0,
        },
        &live,
        &live,
    )
    .unwrap();
    let mut cache: SplitStageCacheV2<u8> = SplitStageCacheV2::new(4);
    let key = cache
        .store(&[&admitted], &own_chain(), 1, &live, &live)
        .unwrap();
    for op in [
        PersistOp::Persist,
        PersistOp::Snapshot,
        PersistOp::TierWrite,
        PersistOp::Restore,
    ] {
        assert_eq!(cache.persist(op, &key).unwrap_err().op, op);
    }
    let restarted: SplitStageCacheV2<u8> = SplitStageCacheV2::new(4);
    assert_eq!(restarted.len(), 0);
}

#[test]
fn a_frame_crosses_the_v2_wire_and_is_admitted_with_its_lineage_intact() {
    let world = world();
    let frame = approximate(&world);
    let bytes = frame.to_bytes();
    let mut header = ChunkHeaderV2 {
        frame_id: 1,
        parent_operation_id: 2,
        chunk_index: 0,
        chunk_total: 1,
        signed_absolute_deadline_ms: 9_000,
        frame_digest: frame_digest_v2(&bytes),
        chunk_digest: [0; 32],
        upstream_chain: frame.upstream_chain.clone(),
        input_lineage: frame.input_lineage.clone(),
        mode: frame.mode,
        taint: frame.taint,
    };
    header.chunk_digest = chunk_digest_v2(&header, &bytes);
    let chunk = ActivationChunkV2 {
        header,
        signature: [1; 64],
        bytes,
    };
    let Ok(AllowlistedPayloadV2::ChunkV2(received)) = admit_v2(&chunk_payload_v2(&chunk)) else {
        panic!("the v2 chunk was not admitted");
    };
    let decoded = ActivationFrameV2::from_bytes(&received.bytes).unwrap();
    assert!(received.header.matches_frame(&decoded));
    assert_eq!(decoded, frame);
    let live = live(&world);
    assert!(admit_frame_v2(
        AdmissionRequestV2 {
            frame: &decoded,
            policy: KvReusePolicy::AllowApproximate,
            expected_start: 0
        },
        &live,
        &live
    )
    .is_ok());
    // The frame holds an element of every matrix kind, and every one is classified once.
    assert_eq!(bijection_violations(&[&decoded]), 0);
    assert_eq!(DependencyKind::ALL.len(), 7);
}
