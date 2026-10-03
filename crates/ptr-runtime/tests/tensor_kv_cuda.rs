#![cfg(feature = "candle-cuda")]

use ptr_config::PtrConfig;
use ptr_pods::{
    CandleGpuTierBackend, CandleKvTensorBackend, KvPageSize, KvTensorDType, KvTensorSchema,
    KvTierBinding, TensorRef,
};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, ManagedKvPageContext, ManagedKvRegistry, NodeHealth, NodeRecord,
    PodPlacementController, PtrRuntime, TierResidencyController,
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
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn setup() -> (PodPlacementController, ptr_runtime::Placement) {
    let mut placement = PodPlacementController::default();
    placement
        .register_node(NodeRecord {
            node_id: NodeId::from("cuda-node"),
            zone: "local".into(),
            health: NodeHealth::Healthy,
            last_heartbeat: Timestamp(1),
            devices: vec![DeviceRecord {
                device_id: DeviceId::from("cuda:0"),
                vram_bytes: 24 * 1024 * 1024 * 1024,
                used_vram_bytes: 0,
                health: DeviceHealth::Healthy,
            }],
        })
        .unwrap();
    let pod = placement
        .assign(
            PodId::from("cuda-pod"),
            ArtifactId::from("cuda-artifact"),
            Generation(1),
            NodeId::from("cuda-node"),
            DeviceId::from("cuda:0"),
            1,
        )
        .unwrap();
    (placement, pod)
}

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("cuda-model"),
        adapter: AdapterVersion::from("cuda-adapter"),
        device: DeviceId::from("cuda:0"),
        layer_count: 1,
        attention_heads: 1,
        key_value_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

#[test]
fn registry_runs_real_candle_kv_cache_append_snapshot_and_restore() {
    let (mut placement, pod) = setup();
    let page_size = KvPageSize::new(2).unwrap();
    let backend = CandleKvTensorBackend::with_page_size(0, page_size);
    let mut registry = ManagedKvRegistry::new(backend);
    let handle = registry
        .allocate_paged(
            &mut placement,
            StateId::from("cuda-state"),
            &pod.pod_id,
            schema(),
            8,
            ManagedKvPageContext {
                execution_manifest: [3; 32],
                principal: PrincipalId::from("cuda-test"),
                page_tokens: page_size,
            },
        )
        .unwrap();
    registry
        .append(
            &placement,
            &handle,
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
    let snapshot_lease = registry.seal_online_snapshot(&placement, &handle).unwrap();
    let binding = KvTierBinding {
        logical_id: snapshot_lease.state_id().0.clone(),
        generation: snapshot_lease.generation(),
        revision: Revision(1),
        context_digest: [7; 32],
        placement_epoch: snapshot_lease.placement_epoch().0,
        fencing_token: snapshot_lease.fencing_token().0,
    };
    let prepared = registry
        .prepare_online_snapshot_to_tier(&placement, &snapshot_lease, &binding)
        .unwrap();
    assert_eq!(prepared.chunks.len(), 3);

    let tiers = TierResidencyController::default();
    let gpu = Arc::new(CandleGpuTierBackend::new(TierBackendId::from("cuda-tier"), 0).unwrap());
    let gpu_restore =
        Arc::new(CandleGpuTierBackend::new(TierBackendId::from("cuda-tier-restored"), 0).unwrap());
    let cpu = Arc::new(CpuTierBackend::new(TierBackendId::from("cpu-tier")));
    let root = std::env::temp_dir().join(format!("ptr-cuda-tier-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let nvme = Arc::new(FileTierBackend::new(TierBackendId::from("nvme-tier"), &root).unwrap());
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, gpu.clone(), Revision(1))).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, cpu.clone(), Revision(2))).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, nvme.clone(), Revision(3))).unwrap();
    block_on(runtime.attach_tier_backend_journaled(&tiers, gpu_restore.clone(), Revision(4)))
        .unwrap();
    for (descriptor, bytes) in prepared.manifest.chunks.iter().zip(&prepared.chunks) {
        block_on(gpu.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    }
    block_on(runtime.register_tier_object_journaled(
        &tiers,
        prepared.manifest.clone(),
        gpu.identity(),
        Revision(5),
    ))
    .unwrap();
    block_on(runtime.transfer_tier_replica_journaled(
        &tiers,
        prepared.manifest.root_digest,
        &gpu.identity(),
        &cpu.identity(),
        None,
        Revision(6),
    ))
    .unwrap();
    block_on(runtime.transfer_tier_replica_journaled(
        &tiers,
        prepared.manifest.root_digest,
        &cpu.identity(),
        &nvme.identity(),
        None,
        Revision(7),
    ))
    .unwrap();
    block_on(runtime.transfer_tier_replica_journaled(
        &tiers,
        prepared.manifest.root_digest,
        &nvme.identity(),
        &gpu_restore.identity(),
        None,
        Revision(8),
    ))
    .unwrap();
    let admitted = block_on(
        tiers.read_admitted_object(prepared.manifest.root_digest, &gpu_restore.identity()),
    )
    .unwrap();

    let old_token = registry.metadata(&handle).unwrap().fencing_token;
    registry.release(&mut placement, &handle).unwrap();
    let restored = registry
        .restore_paged_from_tier(
            &mut placement,
            &tiers,
            runtime.tier_journal(),
            StateId::from("cuda-restored"),
            &pod.pod_id,
            &admitted,
            &binding,
        )
        .unwrap();
    assert_eq!(registry.metadata(&restored).unwrap().sequence_length, 3);
    assert_ne!(
        registry.metadata(&restored).unwrap().fencing_token,
        old_token
    );
    std::fs::remove_dir_all(root).unwrap();
}
