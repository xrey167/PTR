use ptr_config::PtrConfig;
use ptr_pods::{
    InMemoryKvTensorBackend, InMemoryPagedKvBackend, KvPageSize, KvTensorDType, KvTensorSchema,
    KvTierBinding, TensorRef,
};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, ManagedKvPageContext, ManagedKvRegistry, NodeHealth, NodeRecord,
    PodPlacementController, PtrRuntime, TensorKvError, TierResidencyController,
};
use ptr_storage::{CpuTierBackend, FileTierBackend, TierBackend, TierBackendId};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, PrincipalId,
    Revision, StateId, Timestamp,
};
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

#[test]
fn managed_kv_snapshot_roundtrips_cpu_nvme_and_restores_under_new_lease() {
    let mut placement = PodPlacementController::default();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("node-a"),
            zone: "zone-a".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![
                DeviceRecord {
                    device_id: DeviceId::from("cpu"),
                    vram_bytes: 1 << 20,
                    used_vram_bytes: 0,
                    health: DeviceHealth::Healthy,
                },
                DeviceRecord {
                    device_id: DeviceId::from("cpu-other"),
                    vram_bytes: 1 << 20,
                    used_vram_bytes: 0,
                    health: DeviceHealth::Healthy,
                },
            ],
        })
        .unwrap();
    let pod = placement
        .assign(
            PodId::from("kv-pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cpu"),
            1024,
        )
        .unwrap();
    let schema = KvTensorSchema {
        model: ModelVersion::from("model"),
        adapter: AdapterVersion::from("adapter"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 1,
        key_value_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let mut registry = ManagedKvRegistry::new(InMemoryKvTensorBackend);
    let original = registry
        .allocate(
            &mut placement,
            StateId::from("state-1"),
            &pod.pod_id,
            schema,
            8,
        )
        .unwrap();
    registry
        .append(
            &placement,
            &original,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![1.0, 2.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![3.0, 4.0],
            }],
        )
        .unwrap();
    let metadata = registry.metadata(&original).unwrap();
    let binding = KvTierBinding {
        logical_id: "state-1".into(),
        generation: metadata.generation,
        revision: Revision(1),
        context_digest: [7; 32],
        placement_epoch: metadata.placement_epoch.0,
        fencing_token: metadata.fencing_token.0,
    };
    let prepared = registry
        .snapshot_to_tier(&placement, &original, &binding, 32)
        .unwrap();
    let mut stale_binding = binding.clone();
    stale_binding.fencing_token += 1;
    assert_eq!(
        registry.snapshot_to_tier(&placement, &original, &stale_binding, 32),
        Err(TensorKvError::TierBindingMismatch)
    );

    let tiers = TierResidencyController::default();
    let cpu = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-tier")));
    let root = std::env::temp_dir().join(format!("ptr-kv-tier-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let nvme = Arc::new(FileTierBackend::new(TierBackendId::from("nvme-tier"), &root).unwrap());
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, cpu.clone(), Revision(1))).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, nvme.clone(), Revision(1))).unwrap();
    for (descriptor, bytes) in prepared.manifest.chunks.iter().zip(&prepared.chunks) {
        block_on(cpu.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    }
    block_on(runtime.register_tier_object_journaled(
        &tiers,
        prepared.manifest.clone(),
        cpu.identity(),
        Revision(1),
    ))
    .unwrap();
    block_on(runtime.transfer_tier_replica_journaled(
        &tiers,
        prepared.manifest.root_digest,
        &cpu.identity(),
        &nvme.identity(),
        None,
        Revision(1),
    ))
    .unwrap();
    block_on(tiers.evict(prepared.manifest.root_digest, &cpu.identity())).unwrap();
    let durable =
        block_on(tiers.read_admitted_object(prepared.manifest.root_digest, &nvme.identity()))
            .unwrap();
    let rogue_tiers = TierResidencyController::default();
    let rogue = Arc::new(CpuTierBackend::new(TierBackendId::from("rogue-tier")));
    block_on(rogue_tiers.attach_backend(rogue.clone())).unwrap();
    for (descriptor, bytes) in prepared.manifest.chunks.iter().zip(&prepared.chunks) {
        block_on(rogue.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    }
    block_on(rogue_tiers.register_object(prepared.manifest.clone(), rogue.identity())).unwrap();
    let unauthoritative = block_on(
        rogue_tiers.read_admitted_object(prepared.manifest.root_digest, &rogue.identity()),
    )
    .unwrap();
    assert_eq!(
        registry.restore_from_tier(
            &mut placement,
            &rogue_tiers,
            runtime.tier_journal(),
            StateId::from("state-unadmitted"),
            &pod.pod_id,
            &unauthoritative,
            &binding,
        ),
        Err(TensorKvError::TierBindingMismatch)
    );
    registry.invalidate(&mut placement, &original).unwrap();
    placement
        .assign(
            pod.pod_id.clone(),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cpu-other"),
            1024,
        )
        .unwrap();
    assert_eq!(
        registry.restore_from_tier(
            &mut placement,
            &tiers,
            runtime.tier_journal(),
            StateId::from("state-wrong-device"),
            &pod.pod_id,
            &durable,
            &binding,
        ),
        Err(TensorKvError::DeviceMismatch)
    );
    placement
        .assign(
            pod.pod_id.clone(),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-a"),
            DeviceId::from("cpu"),
            1024,
        )
        .unwrap();
    let restored = registry
        .restore_from_tier(
            &mut placement,
            &tiers,
            runtime.tier_journal(),
            StateId::from("state-2"),
            &pod.pod_id,
            &durable,
            &binding,
        )
        .unwrap();
    assert_ne!(original.state_id(), restored.state_id());
    assert_eq!(registry.metadata(&restored).unwrap().sequence_length, 1);
    registry
        .append(
            &placement,
            &restored,
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![5.0, 6.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![1, 2],
                values: vec![7.0, 8.0],
            }],
        )
        .unwrap();
    assert_eq!(registry.metadata(&restored).unwrap().sequence_length, 2);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn paged_kv_snapshot_roundtrips_cpu_nvme_with_page_aligned_chunks() {
    let mut placement = PodPlacementController::default();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("node-paged"),
            zone: "zone-a".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cpu"),
                vram_bytes: 1 << 20,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .unwrap();
    let pod = placement
        .assign(
            PodId::from("paged-kv-pod"),
            ArtifactId::from("artifact"),
            Generation(1),
            NodeId::from("node-paged"),
            DeviceId::from("cpu"),
            1024,
        )
        .unwrap();
    let schema = KvTensorSchema {
        model: ModelVersion::from("model"),
        adapter: AdapterVersion::from("adapter"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 1,
        key_value_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let page_size = KvPageSize::new(2).unwrap();
    let mut registry = ManagedKvRegistry::new(InMemoryPagedKvBackend::new(page_size));
    let original = registry
        .allocate_paged(
            &mut placement,
            StateId::from("paged-state"),
            &pod.pod_id,
            schema,
            8,
            ManagedKvPageContext {
                execution_manifest: [8; 32],
                principal: PrincipalId::from("alice"),
                page_tokens: page_size,
            },
        )
        .unwrap();
    registry
        .append(
            &placement,
            &original,
            &[TensorRef {
                layer: 0,
                shape: vec![3, 2],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }],
            &[TensorRef {
                layer: 0,
                shape: vec![3, 2],
                values: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            }],
        )
        .unwrap();
    let online = registry
        .seal_online_snapshot(&placement, &original)
        .unwrap();
    let binding = KvTierBinding {
        logical_id: online.state_id().0.clone(),
        generation: online.generation(),
        revision: Revision(1),
        context_digest: [7; 32],
        placement_epoch: online.placement_epoch().0,
        fencing_token: online.fencing_token().0,
    };
    let prepared = registry
        .prepare_online_snapshot_to_tier(&placement, &online, &binding)
        .unwrap();
    assert_eq!(prepared.chunks.len(), 3);

    let tiers = TierResidencyController::default();
    let cpu = Arc::new(CpuTierBackend::new(TierBackendId::from("paged-cpu")));
    let root = std::env::temp_dir().join(format!("ptr-paged-kv-tier-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let nvme = Arc::new(FileTierBackend::new(TierBackendId::from("paged-nvme"), &root).unwrap());
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, cpu.clone(), Revision(1))).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, nvme.clone(), Revision(1))).unwrap();
    for (descriptor, bytes) in prepared.manifest.chunks.iter().zip(&prepared.chunks) {
        block_on(cpu.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    }
    block_on(runtime.register_tier_object_journaled(
        &tiers,
        prepared.manifest.clone(),
        cpu.identity(),
        Revision(1),
    ))
    .unwrap();
    block_on(runtime.transfer_tier_replica_journaled(
        &tiers,
        prepared.manifest.root_digest,
        &cpu.identity(),
        &nvme.identity(),
        None,
        Revision(2),
    ))
    .unwrap();
    let durable =
        block_on(tiers.read_admitted_object(prepared.manifest.root_digest, &nvme.identity()))
            .unwrap();
    let old_token = registry.metadata(&original).unwrap().fencing_token;
    registry.invalidate(&mut placement, &original).unwrap();
    let restored = registry
        .restore_paged_from_tier(
            &mut placement,
            &tiers,
            runtime.tier_journal(),
            StateId::from("paged-state-restored"),
            &pod.pod_id,
            &durable,
            &binding,
        )
        .unwrap();
    let restored_metadata = registry.metadata(&restored).unwrap();
    assert_eq!(restored_metadata.sequence_length, 3);
    assert_ne!(restored_metadata.fencing_token, old_token);
    std::fs::remove_dir_all(root).unwrap();
}
