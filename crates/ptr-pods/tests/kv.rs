use ptr_pods::{
    InMemoryKvTensorBackend, KvCacheLayout, KvCacheTier, KvTensorBackend, KvTensorDType,
    KvTensorSchema, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, ModelVersion};

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        device: DeviceId::from("cuda:0"),
        layer_count: 2,
        attention_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn tensor(layer: usize, tokens: usize, value: f32) -> TensorRef {
    TensorRef {
        layer,
        shape: vec![tokens, 4],
        values: vec![value; tokens * 4],
    }
}

#[test]
fn append_snapshot_restore_and_truncate_preserve_kv_state() {
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema(), 8).unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 2, 1.0), tensor(1, 2, 2.0)],
            &[tensor(0, 2, 3.0), tensor(1, 2, 4.0)],
        )
        .unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 1, 5.0), tensor(1, 1, 6.0)],
            &[tensor(0, 1, 7.0), tensor(1, 1, 8.0)],
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    assert_eq!(snapshot.sequence_length, 3);
    let mut restored = backend
        .restore(snapshot, &DeviceId::from("cuda:0"))
        .unwrap();
    assert_eq!(restored.sequence_length, 3);
    backend
        .append(
            &mut restored,
            &[tensor(0, 1, 9.0), tensor(1, 1, 9.0)],
            &[tensor(0, 1, 9.0), tensor(1, 1, 9.0)],
        )
        .unwrap();
    assert_eq!(restored.sequence_length, 4);
    backend.truncate(&mut cache, 1).unwrap();
    assert_eq!(cache.sequence_length, 1);
}

#[test]
fn snapshot_device_and_digest_are_bound() {
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema(), 4).unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
        )
        .unwrap();
    let mut snapshot = backend.snapshot(&cache).unwrap();
    assert!(backend
        .restore(snapshot.clone(), &DeviceId::from("cuda:1"))
        .is_err());
    snapshot.layers[0].keys.values[0] += 1.0;
    assert!(backend
        .restore(snapshot, &DeviceId::from("cuda:0"))
        .is_err());
    let mut schema_snapshot = backend.snapshot(&cache).unwrap();
    schema_snapshot.schema.model = ModelVersion::from("other-model");
    assert_eq!(
        backend.restore(schema_snapshot, &DeviceId::from("cuda:0")),
        Err(ptr_pods::KvBackendError::SnapshotDigestMismatch)
    );
}

#[test]
fn allocation_rejects_zero_schema_dimensions_and_capacity() {
    let backend = InMemoryKvTensorBackend;
    let mut invalid = schema();
    invalid.layer_count = 0;
    assert!(matches!(
        backend.allocate(invalid, 4),
        Err(ptr_pods::KvBackendError::InvalidSchema(_))
    ));
    assert!(matches!(
        backend.allocate(schema(), 0),
        Err(ptr_pods::KvBackendError::InvalidSchema(_))
    ));
}

#[test]
fn append_rejects_wrong_layer_shape_and_capacity() {
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema(), 2).unwrap();
    assert_eq!(
        backend.append(&mut cache, &[tensor(0, 1, 1.0)], &[tensor(0, 1, 1.0)]),
        Err(ptr_pods::KvBackendError::InvalidLayer(1))
    );
    let mut malformed = tensor(0, 1, 1.0);
    malformed.shape = vec![1, 3];
    assert_eq!(
        backend.append(
            &mut cache,
            &[malformed, tensor(1, 1, 1.0)],
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)]
        ),
        Err(ptr_pods::KvBackendError::ShapeMismatch)
    );
    assert!(backend
        .append(
            &mut cache,
            &[tensor(0, 2, 1.0), tensor(1, 2, 1.0)],
            &[tensor(0, 2, 1.0), tensor(1, 2, 1.0)]
        )
        .is_ok());
    assert!(matches!(
        backend.append(
            &mut cache,
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)]
        ),
        Err(ptr_pods::KvBackendError::InvalidSchema(_))
    ));
}

#[test]
fn restore_rejects_snapshot_with_wrong_layer_count() {
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema(), 2).unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
        )
        .unwrap();
    let mut snapshot = backend.snapshot(&cache).unwrap();
    snapshot.layers.pop();
    assert_eq!(
        backend.restore(snapshot, &DeviceId::from("cuda:0")),
        Err(ptr_pods::KvBackendError::SnapshotDigestMismatch)
    );
}

#[test]
fn restore_preserves_capacity_and_rejects_appends_beyond_it() {
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema(), 1).unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    let mut restored = backend
        .restore(snapshot, &DeviceId::from("cuda:0"))
        .unwrap();
    assert_eq!(
        backend.append(
            &mut restored,
            &[tensor(0, 1, 2.0), tensor(1, 1, 2.0)],
            &[tensor(0, 1, 2.0), tensor(1, 1, 2.0)]
        ),
        Err(ptr_pods::KvBackendError::InvalidSchema(
            "KV capacity exceeded".into()
        ))
    );
}

#[test]
fn restore_rejects_a_snapshot_with_a_wrong_device_even_when_the_digest_is_valid() {
    let backend = InMemoryKvTensorBackend;
    let cache = backend.allocate(schema(), 1).unwrap();
    let mut snapshot = backend.snapshot(&cache).unwrap();
    snapshot.schema.device = DeviceId::from("cuda:1");
    assert_eq!(
        backend.restore(snapshot, &DeviceId::from("cuda:0")),
        Err(ptr_pods::KvBackendError::DeviceMismatch)
    );
}

#[test]
fn snapshot_digest_changes_when_tensor_schema_changes() {
    let backend = InMemoryKvTensorBackend;
    let cache = backend.allocate(schema(), 1).unwrap();
    let mut snapshot = backend.snapshot(&cache).unwrap();
    snapshot.schema.head_dim += 1;
    assert!(!snapshot.verify_digest());
}

#[test]
fn truncate_rejects_length_beyond_current_sequence() {
    let backend = InMemoryKvTensorBackend;
    let mut cache = backend.allocate(schema(), 4).unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
            &[tensor(0, 1, 1.0), tensor(1, 1, 1.0)],
        )
        .unwrap();
    assert_eq!(
        backend.truncate(&mut cache, 2),
        Err(ptr_pods::KvBackendError::InvalidTruncation)
    );
}

#[test]
fn snapshot_binds_cache_layout_and_quantization_contract() {
    let backend = InMemoryKvTensorBackend;
    let cache = backend.allocate(schema(), 4).unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    assert_eq!(snapshot.layout, KvCacheLayout::default());
    assert_eq!(snapshot.layout.tier, KvCacheTier::Cpu);
    assert_eq!(snapshot.layout.dtype, KvTensorDType::F32);
    let mut changed = snapshot.clone();
    changed.layout.tier = KvCacheTier::Nvme;
    assert!(!changed.verify_digest());
}

#[test]
fn current_f32_backend_rejects_unimplemented_quantized_storage() {
    let backend = InMemoryKvTensorBackend;
    let mut quantized = schema();
    quantized.dtype = KvTensorDType::Fp8E4M3;
    assert_eq!(
        backend.allocate(quantized, 4),
        Err(ptr_pods::KvBackendError::UnsupportedDtype)
    );
}

#[test]
fn page_table_allocates_and_releases_logical_pages() {
    let mut pages = ptr_pods::KvPageTable::allocate(2, 5).unwrap();
    pages.reserve_tokens(3).unwrap();
    assert_eq!(pages.logical_to_physical.len(), 2);
    assert_eq!(pages.free_physical.len(), 1);
    pages.truncate(1).unwrap();
    assert_eq!(pages.logical_to_physical.len(), 1);
    assert_eq!(pages.free_physical.len(), 2);
    assert!(pages.validate().is_ok());
}
