#![cfg(feature = "candle-cuda")]

use ptr_pods::{CandleKvTensorBackend, KvTensorDType, KvTensorSchema, TensorRef};
use ptr_runtime::{
    DeviceHealth, DeviceRecord, ManagedKvRegistry, NodeHealth, NodeRecord, PodPlacementController,
};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, NodeId, PodId, StateId,
    Timestamp,
};

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
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

#[test]
fn registry_runs_real_candle_kv_cache_append_snapshot_and_restore() {
    let (mut placement, pod) = setup();
    let backend = CandleKvTensorBackend::new(0);
    let mut registry = ManagedKvRegistry::new(backend);
    let handle = registry
        .allocate(
            &mut placement,
            StateId::from("cuda-state"),
            &pod.pod_id,
            schema(),
            4,
        )
        .unwrap();
    registry
        .append(
            &placement,
            &handle,
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
    let snapshot = registry.snapshot(&placement, &handle).unwrap();
    assert_eq!(snapshot.sequence_length, 1);

    registry.release(&mut placement, &handle).unwrap();
    let restored = registry
        .restore(
            &mut placement,
            StateId::from("cuda-restored"),
            &pod.pod_id,
            snapshot,
        )
        .unwrap();
    assert_eq!(registry.metadata(&restored).unwrap().sequence_length, 1);
}
