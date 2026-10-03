use ptr_pods::{
    ArtifactTierAdapter, ArtifactTierBinding, InMemoryKvTensorBackend, InMemoryPagedKvBackend,
    KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType, KvTensorSchema, KvTierAdapter,
    KvTierBinding, ModelWeightTierAdapter, ModelWeightTierBinding, PagedKvTensorBackend,
    PagedKvTierAdapter, PodTierError, TensorRef,
};
use ptr_storage::{tier_digest, PreparedTierObject, TierObjectDomain};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId, Revision};

fn snapshot() -> ptr_pods::KvTensorSnapshot {
    let backend = InMemoryKvTensorBackend;
    let schema = KvTensorSchema {
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let mut cache = backend.allocate(schema, 8).unwrap();
    let tensor = TensorRef {
        layer: 0,
        shape: vec![2, 4],
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    };
    backend
        .append(
            &mut cache,
            std::slice::from_ref(&tensor),
            std::slice::from_ref(&tensor),
        )
        .unwrap();
    backend.snapshot(&cache).unwrap()
}

fn kv_binding() -> KvTierBinding {
    KvTierBinding {
        logical_id: "session-a/state-1".into(),
        generation: Generation(3),
        revision: Revision(9),
        context_digest: [7; 32],
        placement_epoch: 11,
        fencing_token: 13,
    }
}

fn paged_snapshot() -> ptr_pods::KvPagedSnapshot {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let schema = KvTensorSchema {
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let binding = KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(3),
        execution_manifest: [8; 32],
        principal: PrincipalId::from("alice"),
        device: schema.device.clone(),
        dtype: schema.dtype,
        page_tokens: page_size,
    };
    let mut cache = backend.allocate_bound(schema, 8, binding).unwrap();
    let tensor = TensorRef {
        layer: 0,
        shape: vec![3, 4],
        values: (0..12).map(|value| value as f32).collect(),
    };
    backend
        .append(
            &mut cache,
            std::slice::from_ref(&tensor),
            std::slice::from_ref(&tensor),
        )
        .unwrap();
    let lease = backend.seal_snapshot(&mut cache).unwrap();
    backend.snapshot_from_lease(&lease).unwrap()
}

#[test]
fn kv_snapshot_roundtrips_through_chunked_tier_object() {
    let snapshot = snapshot();
    let binding = kv_binding();
    let object = KvTierAdapter::prepare(&snapshot, &binding, 37).unwrap();
    assert!(object.chunks.len() > 1);
    assert_eq!(KvTierAdapter::restore(&object, &binding).unwrap(), snapshot);
}

#[test]
fn kv_restore_rejects_changed_fencing_binding_and_corruption() {
    let snapshot = snapshot();
    let binding = kv_binding();
    let object = KvTierAdapter::prepare(&snapshot, &binding, 41).unwrap();
    let mut stale = binding.clone();
    stale.fencing_token += 1;
    assert_eq!(
        KvTierAdapter::restore(&object, &stale),
        Err(PodTierError::InvalidBinding)
    );

    let mut damaged = object;
    damaged.chunks[0][0] ^= 1;
    assert!(matches!(
        KvTierAdapter::restore(&damaged, &binding),
        Err(PodTierError::Storage(_))
    ));
}

#[test]
fn paged_kv_roundtrip_preserves_one_chunk_per_page() {
    let snapshot = paged_snapshot();
    let binding = kv_binding();
    let object = PagedKvTierAdapter::prepare(&snapshot, &binding).unwrap();
    assert_eq!(object.chunks.len(), snapshot.pages.len() + 1);
    assert_eq!(
        PagedKvTierAdapter::restore(&object, &binding).unwrap(),
        snapshot
    );
}

#[test]
fn paged_kv_roundtrip_preserves_compact_grouped_query_heads() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let schema = KvTensorSchema {
        model: ModelVersion::from("qwen2"),
        adapter: AdapterVersion::from("base"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 4,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let page_binding = KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(3),
        execution_manifest: [8; 32],
        principal: PrincipalId::from("alice"),
        device: schema.device.clone(),
        dtype: schema.dtype,
        page_tokens: page_size,
    };
    let mut cache = backend.allocate_bound(schema, 8, page_binding).unwrap();
    let tensor = TensorRef {
        layer: 0,
        shape: vec![3, 4],
        values: (0..12).map(|value| value as f32).collect(),
    };
    backend
        .append(
            &mut cache,
            std::slice::from_ref(&tensor),
            std::slice::from_ref(&tensor),
        )
        .unwrap();
    let lease = backend.seal_snapshot(&mut cache).unwrap();
    let snapshot = backend.snapshot_from_lease(&lease).unwrap();
    let binding = kv_binding();
    let object = PagedKvTierAdapter::prepare(&snapshot, &binding).unwrap();
    let restored = PagedKvTierAdapter::restore(&object, &binding).unwrap();
    assert_eq!(restored.schema.attention_heads, 4);
    assert_eq!(restored.schema.key_value_heads, 2);
    assert_eq!(restored.pages[0].layers[0].keys.shape, vec![2, 4]);
    assert_eq!(restored, snapshot);
}

#[test]
fn paged_kv_restore_rejects_corrupt_page_chunk() {
    let snapshot = paged_snapshot();
    let binding = kv_binding();
    let mut object = PagedKvTierAdapter::prepare(&snapshot, &binding).unwrap();
    object.chunks[1][16] ^= 1;
    assert!(matches!(
        PagedKvTierAdapter::restore(&object, &binding),
        Err(PodTierError::Storage(_))
    ));
}

#[test]
fn kv_decoder_rejects_hostile_lengths_before_allocating() {
    let binding = kv_binding();
    let mut bytes = b"PTRKVTR1".to_vec();
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    let object = PreparedTierObject::from_bytes(
        TierObjectDomain::KvSnapshot,
        binding.logical_id.clone(),
        binding.generation,
        binding.revision,
        [1; 32],
        &bytes,
        64,
    )
    .unwrap();
    assert!(matches!(
        KvTierAdapter::restore(&object, &binding),
        Err(PodTierError::InvalidSnapshot | PodTierError::Overflow)
    ));
}

#[test]
fn weight_schema_binds_manifest_model_tensor_shape_and_dtype() {
    let binding = ModelWeightTierBinding {
        logical_id: "model-a/layer-0/weight".into(),
        generation: Generation(2),
        revision: Revision(4),
        execution_manifest: [1; 32],
        artifact_digest: [2; 32],
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        tensor_name: "layer.0.weight".into(),
        shape: vec![2, 4],
        dtype: "f32".into(),
    };
    let first = ModelWeightTierAdapter::prepare(b"weight-bytes", &binding, 4).unwrap();
    let mut changed = binding;
    changed.dtype = "bf16".into();
    let second = ModelWeightTierAdapter::prepare(b"weight-bytes", &changed, 4).unwrap();
    assert_ne!(first.manifest.root_digest, second.manifest.root_digest);
}

#[test]
fn artifact_frontend_requires_content_digest_and_lineage() {
    let bytes = b"artifact-content";
    let binding = ArtifactTierBinding {
        logical_id: "artifact-a".into(),
        generation: Generation(1),
        revision: Revision(2),
        artifact_digest: tier_digest(bytes),
        execution_manifest: [4; 32],
        provenance_digest: [5; 32],
    };
    assert!(ArtifactTierAdapter::prepare(bytes, &binding, 8).is_ok());
    assert_eq!(
        ArtifactTierAdapter::prepare(b"different", &binding, 8),
        Err(PodTierError::InvalidBinding)
    );
}
