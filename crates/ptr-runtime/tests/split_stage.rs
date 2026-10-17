//! The downstream stage cache against a real ledger and a real placement controller: an entry
//! derived from an upstream chain is served while the chain is live and refused the moment the
//! ledger revokes or supersedes the stage, or the stage is re-placed, whether or not anyone
//! told the cache.

use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_model_api::KvReuseMode;
use ptr_protocol::activation::{ActivationFrame, BoundaryDtype, ChainElement, UpstreamChain};
use ptr_runtime::{
    admit_frame, admit_frame_in, revalidate_chain_in, ChainFault, DeviceHealth, DeviceRecord,
    NodeHealth, NodeRecord, PodPlacementController, ProjectPlacementRequest, PtrRuntime,
    RuntimeStageLiveness, SplitStageCache, StageLiveness, UseFault,
};
use ptr_types::{
    ArtifactId, CapsuleId, DeviceId, Generation, NodeId, PodId, ProjectId, Timestamp, Validity,
};
use std::collections::BTreeMap;

fn digest(name: &str, generation: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = name.len() as u8;
    out[1] = generation as u8;
    out
}

struct World {
    runtime: PtrRuntime,
    placement: PodPlacementController,
    manifests: BTreeMap<(String, u64), [u8; 32]>,
}

fn commit(runtime: &mut PtrRuntime, name: &str, generation: u64) {
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from(name),
            generation: Generation(generation),
        })
        .unwrap();
}

fn node(name: &str) -> NodeRecord {
    NodeRecord {
        node_id: NodeId::from(name),
        zone: "z".into(),
        health: NodeHealth::Healthy,
        last_heartbeat: Timestamp(1),
        devices: vec![DeviceRecord {
            device_id: DeviceId::from("cpu"),
            vram_bytes: 10,
            used_vram_bytes: 0,
            health: DeviceHealth::Healthy,
        }],
    }
}

fn place(placement: &mut PodPlacementController, node: &str) {
    placement
        .assign(
            PodId::from("stage"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from(node),
            DeviceId::from("cpu"),
            1,
        )
        .unwrap();
}

fn world() -> World {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    commit(&mut runtime, "stage", 1);
    commit(&mut runtime, "model", 1);
    commit(&mut runtime, "down", 1);
    commit(&mut runtime, "adapter", 1);
    let mut placement = PodPlacementController::default();
    placement.register_node(node("n1")).unwrap();
    placement.register_node(node("n2")).unwrap();
    place(&mut placement, "n1");
    let manifests = [("stage", 1), ("model", 1), ("down", 1), ("adapter", 1)]
        .into_iter()
        .map(|(name, generation)| ((name.to_owned(), generation), digest(name, generation)))
        .collect();
    World {
        runtime,
        placement,
        manifests,
    }
}

fn own_chain() -> UpstreamChain {
    UpstreamChain {
        stage: chain_element("down"),
        base_model: chain_element("model"),
        adapters: vec![],
        experts: vec![],
    }
}

fn chain_element(name: &str) -> ChainElement {
    ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: digest(name, 1),
        fence_epoch: 0,
    }
}

fn frame_with(world: &World, chain: UpstreamChain) -> ActivationFrame {
    ActivationFrame {
        upstream_chain: chain,
        ..frame(world)
    }
}

fn frame(world: &World) -> ActivationFrame {
    let element = |name: &str| ChainElement {
        identity: name.to_owned(),
        generation: 1,
        digest: digest(name, 1),
        fence_epoch: 0,
    };
    ActivationFrame {
        tokens: 1,
        width: 2,
        dtype: BoundaryDtype::F32,
        positions: vec![0],
        upstream_chain: UpstreamChain {
            stage: element("stage"),
            base_model: element("model"),
            adapters: vec![],
            experts: vec![],
        },
        placement_epoch: world
            .placement
            .placement(&PodId::from("stage"))
            .unwrap()
            .epoch
            .0,
        tensor: [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn live(world: &World) -> RuntimeStageLiveness<'_> {
    RuntimeStageLiveness {
        runtime: &world.runtime,
        placement: &world.placement,
        manifests: &world.manifests,
    }
}

#[test]
fn a_revocation_the_cache_was_never_told_of_ends_the_entry_at_its_next_use() {
    let mut world = world();
    let admitted = admit_frame(&frame(&world), KvReuseMode::Exact, &live(&world)).unwrap();
    let mut cache: SplitStageCache<&str> = SplitStageCache::new(2);
    let key = cache
        .store(&[&admitted], &own_chain(), "kv", &live(&world))
        .unwrap();
    assert_eq!(cache.lookup(&key, &live(&world)), Ok(&"kv"));
    world
        .runtime
        .commit(LedgerEvent::Revoked {
            subject: "stage".to_owned(),
            generation: Generation(1),
        })
        .unwrap();
    assert_eq!(
        world.runtime.generation_validity("stage", Generation(1)),
        Some(Validity::Revoked)
    );
    assert!(matches!(
        cache.lookup(&key, &live(&world)),
        Err(UseFault::Stale(ChainFault::NotLive {
            standing: Some(Validity::Revoked),
            ..
        }))
    ));
    // The frame of the same chain is refused as well.
    assert!(matches!(
        admit_frame(&frame(&world), KvReuseMode::Exact, &live(&world)),
        Err(ChainFault::NotLive { .. })
    ));
}

#[test]
fn a_stage_re_placed_without_a_new_generation_refuses_the_old_workers_frame() {
    let mut world = world();
    let old = frame(&world);
    world
        .placement
        .mark_node_failed(&NodeId::from("n1"))
        .unwrap();
    place(&mut world.placement, "n2");
    assert!(matches!(
        admit_frame(&old, KvReuseMode::Exact, &live(&world)),
        Err(ChainFault::StalePlacement { .. })
    ));
    assert!(admit_frame(&frame(&world), KvReuseMode::Exact, &live(&world)).is_ok());
}

#[test]
fn a_supersession_ends_the_old_generations_entry() {
    let mut world = world();
    let admitted = admit_frame(&frame(&world), KvReuseMode::Exact, &live(&world)).unwrap();
    let mut cache: SplitStageCache<u8> = SplitStageCache::new(2);
    let key = cache
        .store(&[&admitted], &own_chain(), 1, &live(&world))
        .unwrap();
    world
        .runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("stage"),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    assert!(matches!(
        cache.begin_use(&key, &live(&world)),
        Err(UseFault::Stale(ChainFault::NotLive {
            standing: Some(Validity::Superseded),
            ..
        }))
    ));
}

fn revoke(world: &mut World, subject: &str) {
    world
        .runtime
        .commit(LedgerEvent::Revoked {
            subject: subject.to_owned(),
            generation: Generation(1),
        })
        .unwrap();
}

#[test]
fn a_revoked_base_model_or_adapter_ends_the_frame_and_the_entry() {
    for subject in ["model", "adapter"] {
        let mut world = world();
        let chain = UpstreamChain {
            adapters: vec![chain_element("adapter")],
            ..frame(&world).upstream_chain
        };
        let admitted = admit_frame(
            &frame_with(&world, chain.clone()),
            KvReuseMode::Exact,
            &live(&world),
        )
        .unwrap();
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(2);
        let key = cache
            .store(&[&admitted], &own_chain(), 1, &live(&world))
            .unwrap();
        revoke(&mut world, subject);
        assert!(
            matches!(
                cache.lookup(&key, &live(&world)),
                Err(UseFault::Stale(ChainFault::NotLive { ref identity, .. })) if identity == subject
            ),
            "{subject}"
        );
        assert!(matches!(
            admit_frame(
                &frame_with(&world, chain),
                KvReuseMode::Exact,
                &live(&world)
            ),
            Err(ChainFault::NotLive { .. })
        ));
    }
}

#[test]
fn the_downstream_stage_revoked_ends_its_own_entry() {
    let mut world = world();
    let admitted = admit_frame(&frame(&world), KvReuseMode::Exact, &live(&world)).unwrap();
    let mut cache: SplitStageCache<u8> = SplitStageCache::new(2);
    let key = cache
        .store(&[&admitted], &own_chain(), 1, &live(&world))
        .unwrap();
    revoke(&mut world, "down");
    assert!(matches!(
        cache.lookup(&key, &live(&world)),
        Err(UseFault::Stale(ChainFault::NotLive { ref identity, .. })) if identity == "down"
    ));
}

#[test]
fn a_chain_expert_that_is_no_registry_expert_is_refused_fail_closed() {
    // Renamed: the runtime now answers fence epochs from its expert registry, so an expert chain
    // is refused when the identity is no registry key (here a plain capsule name), not for all.
    // The positive and fenced cases are in tests/split_stage_v2.rs.
    let world = world();
    let mut expert = chain_element("model");
    expert.fence_epoch = 3;
    let chain = UpstreamChain {
        experts: vec![expert],
        ..frame(&world).upstream_chain
    };
    assert!(matches!(
        admit_frame(
            &frame_with(&world, chain),
            KvReuseMode::Exact,
            &live(&world)
        ),
        Err(ChainFault::StaleFence { current: None, .. })
    ));
}

#[test]
fn an_approximate_upstream_is_refused_and_a_full_cache_recovers_from_a_lost_revoke() {
    let mut world = world();
    assert_eq!(
        admit_frame(&frame(&world), KvReuseMode::Approximate, &live(&world)),
        Err(ChainFault::NotExact {
            mode: KvReuseMode::Approximate
        })
    );
    let admitted = admit_frame(&frame(&world), KvReuseMode::Exact, &live(&world)).unwrap();
    let mut cache: SplitStageCache<u8> = SplitStageCache::new(1);
    cache
        .store(&[&admitted], &own_chain(), 1, &live(&world))
        .unwrap();
    let mut other = frame(&world);
    other.tensor = [5.0f32, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let other = admit_frame(&other, KvReuseMode::Exact, &live(&world)).unwrap();
    assert!(cache
        .store(&[&other], &own_chain(), 2, &live(&world))
        .is_err());
    // The ledger revokes the stage; the cache is never told. Its dead entry gives way.
    revoke(&mut world, "stage");
    commit(&mut world.runtime, "stage", 2);
    assert_eq!(cache.evict_dead(&live(&world)), 1);
    assert!(cache.is_empty());
}

#[test]
fn a_frame_produced_on_a_device_that_then_failed_is_refused() {
    let mut world = world();
    let old = frame(&world);
    let admitted = admit_frame(&old, KvReuseMode::Exact, &live(&world)).unwrap();
    let mut cache: SplitStageCache<u8> = SplitStageCache::new(2);
    let key = cache
        .store(&[&admitted], &own_chain(), 1, &live(&world))
        .unwrap();
    world
        .placement
        .heartbeat(
            &NodeId::from("n1"),
            Timestamp(2),
            vec![DeviceRecord {
                device_id: DeviceId::from("cpu"),
                vram_bytes: 10,
                used_vram_bytes: 0,
                health: DeviceHealth::Failed,
            }],
        )
        .unwrap();
    assert!(matches!(
        admit_frame(&old, KvReuseMode::Exact, &live(&world)),
        Err(ChainFault::StalePlacement { .. })
    ));
    assert!(matches!(
        cache.lookup(&key, &live(&world)),
        Err(UseFault::Stale(ChainFault::StalePlacement { .. }))
    ));
    // The entry was evicted at that use, so it is gone afterwards.
    assert!(cache.lookup(&key, &live(&world)).is_err());
}

#[test]
fn a_frame_produced_before_its_node_was_replaced_is_refused() {
    let mut world = world();
    let old = frame(&world);
    admit_frame(&old, KvReuseMode::Exact, &live(&world)).unwrap();
    // A new node process registers under the same id with the same healthy device.
    world.placement.register_node(node("n1")).unwrap();
    assert!(matches!(
        admit_frame(&old, KvReuseMode::Exact, &live(&world)),
        Err(ChainFault::StalePlacement { .. })
    ));
}

fn place_in(placement: &mut PodPlacementController, project: &str, node: &str) -> u64 {
    placement
        .assign_in(ProjectPlacementRequest {
            project: ProjectId::from(project),
            pod_id: PodId::from("stage"),
            artifact_id: ArtifactId::from("artifact"),
            generation: Generation(1),
            node_id: NodeId::from(node),
            device_id: DeviceId::from("cpu"),
            required_vram: 1,
        })
        .unwrap()
        .epoch
        .0
}

/// `frame` produced under the placement epoch `epoch`.
fn frame_at(world: &World, epoch: u64) -> ActivationFrame {
    ActivationFrame {
        placement_epoch: epoch,
        ..frame(world)
    }
}

#[test]
fn a_frame_of_one_project_is_refused_against_the_same_named_stage_of_another() {
    let mut world = world();
    let (a, b) = (ProjectId::from("a"), ProjectId::from("b"));
    let epoch_a = place_in(&mut world.placement, "a", "n1");
    let epoch_b = place_in(&mut world.placement, "b", "n1");
    let from_a = frame_at(&world, epoch_a);
    let from_b = frame_at(&world, epoch_b);

    // Each frame is admitted for its own project, and only for it.
    assert!(admit_frame_in(&a, &from_a, KvReuseMode::Exact, &live(&world)).is_ok());
    assert!(admit_frame_in(&b, &from_b, KvReuseMode::Exact, &live(&world)).is_ok());
    assert_eq!(
        admit_frame_in(&b, &from_a, KvReuseMode::Exact, &live(&world)).unwrap_err(),
        ChainFault::StalePlacement {
            recorded: epoch_a,
            current: Some(epoch_b)
        }
    );
    assert_eq!(
        admit_frame_in(&a, &from_b, KvReuseMode::Exact, &live(&world)).unwrap_err(),
        ChainFault::StalePlacement {
            recorded: epoch_b,
            current: Some(epoch_a)
        }
    );
    // A project with no placement of its own is not served by another project's.
    assert_eq!(
        admit_frame_in(
            &ProjectId::from("c"),
            &from_a,
            KvReuseMode::Exact,
            &live(&world)
        )
        .unwrap_err(),
        ChainFault::StalePlacement {
            recorded: epoch_a,
            current: None
        }
    );
    // The legacy path reads the legacy namespace only: it does not see project placements.
    let legacy_epoch = world
        .placement
        .placement(&PodId::from("stage"))
        .unwrap()
        .epoch
        .0;
    assert_ne!(legacy_epoch, epoch_a);
    assert!(matches!(
        admit_frame(&from_a, KvReuseMode::Exact, &live(&world)),
        Err(ChainFault::StalePlacement { .. })
    ));
    // Revalidation of a bare chain names the project too.
    assert!(revalidate_chain_in(&a, &from_a.upstream_chain, epoch_a, &live(&world)).is_ok());
    assert!(revalidate_chain_in(&b, &from_a.upstream_chain, epoch_a, &live(&world)).is_err());
}

#[test]
fn fencing_one_project_leaves_the_frames_and_cache_entries_of_the_other_live() {
    let mut world = world();
    let (a, b) = (ProjectId::from("a"), ProjectId::from("b"));
    let epoch_a = place_in(&mut world.placement, "a", "n1");
    let epoch_b = place_in(&mut world.placement, "b", "n1");
    let admitted_a = admit_frame_in(
        &a,
        &frame_at(&world, epoch_a),
        KvReuseMode::Exact,
        &live(&world),
    )
    .unwrap();
    let admitted_b = admit_frame_in(
        &b,
        &frame_at(&world, epoch_b),
        KvReuseMode::Exact,
        &live(&world),
    )
    .unwrap();
    assert_eq!(admitted_a.project(), Some(&a));
    let mut cache: SplitStageCache<&str> = SplitStageCache::new(4);
    let key_a = cache
        .store(&[&admitted_a], &own_chain(), "kv-a", &live(&world))
        .unwrap();
    let key_b = cache
        .store(&[&admitted_b], &own_chain(), "kv-b", &live(&world))
        .unwrap();
    // The two projects' entries never share a key.
    assert_ne!(key_a, key_b);
    // One cache entry cannot mix frames of two projects.
    assert!(cache
        .store(
            &[&admitted_a, &admitted_b],
            &own_chain(),
            "mixed",
            &live(&world)
        )
        .is_err());

    // Project a's stage is re-placed (new epoch); project b's placement is untouched.
    let moved = place_in(&mut world.placement, "a", "n1");
    assert_ne!(moved, epoch_a);
    assert!(matches!(
        cache.lookup(&key_a, &live(&world)),
        Err(UseFault::Stale(ChainFault::StalePlacement { .. }))
    ));
    assert_eq!(cache.lookup(&key_b, &live(&world)), Ok(&"kv-b"));
    assert!(admitted_b.revalidate(&live(&world)).is_ok());
    assert!(admitted_a.revalidate(&live(&world)).is_err());
}

/// An implementor that does not know projects answers nothing for a project-scoped query
/// (it fails closed), while its legacy answer is unchanged.
#[test]
fn a_liveness_that_does_not_know_projects_refuses_a_project_scoped_frame() {
    struct LegacyOnly<'a>(RuntimeStageLiveness<'a>);
    impl StageLiveness for LegacyOnly<'_> {
        fn validity(&self, target: &str, generation: Generation) -> Option<Validity> {
            self.0.validity(target, generation)
        }
        fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
            self.0.manifest_digest(target, generation)
        }
        fn placement_epoch(&self, stage: &str) -> Option<u64> {
            self.0.placement_epoch(stage)
        }
        fn fence_epoch(&self, expert: &str) -> Option<u64> {
            self.0.fence_epoch(expert)
        }
    }
    let mut world = world();
    let epoch = place_in(&mut world.placement, "a", "n1");
    let legacy = LegacyOnly(live(&world));
    // The legacy path still answers (the legacy namespace holds the world's own placement).
    assert!(admit_frame(&frame(&world), KvReuseMode::Exact, &legacy).is_ok());
    assert_eq!(
        admit_frame_in(
            &ProjectId::from("a"),
            &frame_at(&world, epoch),
            KvReuseMode::Exact,
            &legacy
        )
        .unwrap_err(),
        ChainFault::StalePlacement {
            recorded: epoch,
            current: None
        }
    );
}
