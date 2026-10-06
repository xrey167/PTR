//! `ManagedKvRegistry::fork_prefix` admits a fork from its own committed state and the
//! backend forks only under a registry-minted witness (doc 36 defect 2).

use ptr_pods::{
    InMemoryPagedKvBackend, KvBackendError, KvPageSize, KvTensorDType, KvTensorSchema,
    PagedKvTensorBackend, TensorRef,
};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, ManagedKvHandle, ManagedKvPageContext, ManagedKvRegistry,
    NodeHealth, NodeRecord, PodPlacementController, TensorKvError,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, PrincipalId,
    StateId, Timestamp,
};

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("model"),
        adapter: AdapterVersion::from("adapter"),
        device: DeviceId::from("cpu:0"),
        layer_count: 1,
        attention_heads: 1,
        key_value_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

struct Fixture {
    placement: PodPlacementController,
    registry: ManagedKvRegistry<InMemoryPagedKvBackend>,
    backend: InMemoryPagedKvBackend,
}

fn fixture() -> Fixture {
    let mut placement = PodPlacementController::default();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("node-a"),
            zone: "a".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cpu:0"),
                vram_bytes: 64,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .unwrap();
    let backend = InMemoryPagedKvBackend::new(KvPageSize::new(2).unwrap());
    Fixture {
        placement,
        registry: ManagedKvRegistry::new(backend.clone()),
        backend,
    }
}

fn pod(f: &mut Fixture, name: &str, generation: u64) -> PodId {
    let pod_id = PodId::from(name);
    f.placement
        .assign(
            pod_id.clone(),
            ArtifactId::from(name),
            Generation(generation),
            NodeId::from("node-a"),
            DeviceId::from("cpu:0"),
            1,
        )
        .unwrap();
    pod_id
}

fn alloc(f: &mut Fixture, name: &str, pod_id: &PodId) -> ManagedKvHandle {
    f.registry
        .allocate_paged(
            &mut f.placement,
            StateId::from(name),
            pod_id,
            schema(),
            8,
            ManagedKvPageContext {
                execution_manifest: [9; 32],
                principal: PrincipalId::from("alice"),
                page_tokens: KvPageSize::new(2).unwrap(),
            },
        )
        .unwrap()
}

fn publish(f: &mut Fixture, handle: &ManagedKvHandle) -> ptr_pods::KvSharedPrefix {
    let rows = |v: f32| TensorRef {
        layer: 0,
        shape: vec![4, 2],
        values: vec![v; 8],
    };
    f.registry
        .append(&f.placement, handle, &[rows(1.0)], &[rows(2.0)])
        .unwrap();
    f.registry.publish_prefix(&f.placement, handle).unwrap()
}

#[test]
fn a_current_binding_forks_through_the_registry_witness() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let target = alloc(&mut f, "target", &p);
    let prefix = publish(&mut f, &source);
    f.registry
        .fork_prefix(&f.placement, &target, &prefix)
        .unwrap();
    assert_eq!(f.registry.metadata(&target).unwrap().sequence_length, 4);
    assert!(f.registry.page_metrics(&target).unwrap().shared_pages > 0);
}

#[test]
fn a_stale_generation_binding_cannot_fork_and_shares_no_pages() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let prefix = publish(&mut f, &source);
    let newer = pod(&mut f, "p2", 4);
    let target = alloc(&mut f, "target", &newer);
    assert_eq!(
        f.registry.fork_prefix(&f.placement, &target, &prefix),
        Err(TensorKvError::Backend(KvBackendError::PrefixMismatch))
    );
    assert_eq!(f.registry.metadata(&target).unwrap().sequence_length, 0);
    assert_eq!(f.registry.page_metrics(&target).unwrap().shared_pages, 0);
}

#[test]
fn an_invalidated_target_cannot_fork_and_shares_no_pages() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let target = alloc(&mut f, "target", &p);
    let prefix = publish(&mut f, &source);
    f.registry.invalidate(&mut f.placement, &target).unwrap();
    assert_eq!(
        f.registry.fork_prefix(&f.placement, &target, &prefix),
        Err(TensorKvError::Invalidated)
    );
    assert_eq!(f.registry.page_metrics(&target).unwrap().shared_pages, 0);
}

#[test]
fn a_retired_prefix_cannot_fork_through_the_registry() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let target = alloc(&mut f, "target", &p);
    let prefix = publish(&mut f, &source);
    f.backend.retire_prefix(&prefix.key).unwrap();
    assert_eq!(
        f.registry.fork_prefix(&f.placement, &target, &prefix),
        Err(TensorKvError::Backend(KvBackendError::PrefixNotFound))
    );
    assert_eq!(f.registry.metadata(&target).unwrap().sequence_length, 0);
    assert_eq!(f.registry.page_metrics(&target).unwrap().shared_pages, 0);
}

#[test]
fn the_registry_fails_closed_when_the_fork_authority_is_already_taken() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let target = alloc(&mut f, "target", &p);
    let prefix = publish(&mut f, &source);
    let _held_elsewhere = f.backend.take_fork_authority().unwrap();
    assert!(matches!(
        f.registry.fork_prefix(&f.placement, &target, &prefix),
        Err(TensorKvError::Backend(KvBackendError::Backend(_)))
    ));
    assert_eq!(f.registry.page_metrics(&target).unwrap().shared_pages, 0);
}

#[test]
fn a_prefix_of_an_invalidated_source_cannot_be_forked_by_a_fresh_cache() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let fresh = alloc(&mut f, "fresh", &p);
    let prefix = publish(&mut f, &source);
    f.registry.invalidate(&mut f.placement, &source).unwrap();
    assert_eq!(
        f.registry.fork_prefix(&f.placement, &fresh, &prefix),
        Err(TensorKvError::PrefixNotLive)
    );
    assert_eq!(f.registry.metadata(&fresh).unwrap().sequence_length, 0);
    assert_eq!(f.registry.page_metrics(&fresh).unwrap().shared_pages, 0);
    // The backend retired it too, so even a direct backend retire now reports it absent.
    assert_eq!(
        f.backend.retire_prefix(&prefix.key),
        Err(KvBackendError::PrefixNotFound)
    );
}

#[test]
fn a_prefix_of_a_released_source_cannot_be_forked() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let fresh = alloc(&mut f, "fresh", &p);
    let prefix = publish(&mut f, &source);
    f.registry.release(&mut f.placement, &source).unwrap();
    assert_eq!(
        f.registry.fork_prefix(&f.placement, &fresh, &prefix),
        Err(TensorKvError::PrefixNotLive)
    );
    assert_eq!(f.registry.page_metrics(&fresh).unwrap().shared_pages, 0);
}

#[test]
fn an_invalidated_record_cannot_publish() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    f.registry.invalidate(&mut f.placement, &source).unwrap();
    assert_eq!(
        f.registry.publish_prefix(&f.placement, &source),
        Err(TensorKvError::Invalidated)
    );
}

#[test]
fn a_prefix_never_published_through_the_registry_is_not_live() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let target = alloc(&mut f, "target", &p);
    let mut prefix = publish(&mut f, &source);
    prefix.key.prefix_digest = [0xEE; 32];
    assert_eq!(
        f.registry.fork_prefix(&f.placement, &target, &prefix),
        Err(TensorKvError::PrefixNotLive)
    );
    assert_eq!(f.registry.page_metrics(&target).unwrap().shared_pages, 0);
}

#[test]
fn forged_length_and_offset_do_not_reach_registry_metadata() {
    let mut f = fixture();
    let p = pod(&mut f, "p1", 3);
    let source = alloc(&mut f, "source", &p);
    let target = alloc(&mut f, "target", &p);
    let mut prefix = publish(&mut f, &source);
    let real = (prefix.sequence_length, prefix.position_offset);
    prefix.sequence_length = 1;
    prefix.position_offset = 7;
    f.registry
        .fork_prefix(&f.placement, &target, &prefix)
        .unwrap();
    let metadata = f.registry.metadata(&target).unwrap();
    assert_eq!((metadata.sequence_length, metadata.position_offset), real);
}
