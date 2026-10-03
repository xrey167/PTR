use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryKvTensorBackend, InMemoryPagedKvBackend,
    KvCacheLayout, KvCacheTier, KvContinuationExecutor, KvPageBinding, KvPageSize, KvPageState,
    KvTensorBackend, KvTensorDType, KvTensorSchema, PagedKvTensorBackend, Qwen2Activation,
    Qwen2DecoderInput, Qwen2DecoderWeights, Qwen2LayerConfig, Qwen2LinearWeights,
    Qwen2PagedDecoder, Qwen2RopeScaling, ReferenceCausalAttentionExecutor, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        device: DeviceId::from("cuda:0"),
        layer_count: 2,
        attention_heads: 2,
        key_value_heads: 2,
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

fn page_binding(principal: &str, page_tokens: KvPageSize) -> KvPageBinding {
    KvPageBinding {
        model: ModelVersion::from("model-v1"),
        adapter: AdapterVersion::from("adapter-v1"),
        generation: Generation(1),
        execution_manifest: [7; 32],
        principal: PrincipalId::from(principal),
        device: DeviceId::from("cuda:0"),
        dtype: KvTensorDType::F32,
        page_tokens,
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

#[test]
fn paged_backend_validates_page_size_and_appends_across_boundaries() {
    assert!(KvPageSize::new(0).is_err());
    assert!(KvPageSize::new(3).is_err());
    assert!(KvPageSize::new(257).is_err());
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let mut cache = backend
        .allocate_bound(schema(), 8, page_binding("alice", page_size))
        .unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 3, 1.0), tensor(1, 3, 2.0)],
            &[tensor(0, 3, 3.0), tensor(1, 3, 4.0)],
        )
        .unwrap();
    let pages = cache.page_table();
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].valid_tokens, 2);
    assert_eq!(pages[0].state, KvPageState::Sealed);
    assert_eq!(pages[1].valid_tokens, 1);
    assert_eq!(pages[1].state, KvPageState::Writable);
}

#[test]
fn online_snapshot_is_immutable_while_tail_continues_with_cow() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let mut cache = backend
        .allocate_bound(schema(), 8, page_binding("alice", page_size))
        .unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 3, 1.0), tensor(1, 3, 2.0)],
            &[tensor(0, 3, 3.0), tensor(1, 3, 4.0)],
        )
        .unwrap();
    let lease = backend.seal_snapshot(&mut cache).unwrap();
    let frozen = backend.snapshot_from_lease(&lease).unwrap();
    let old_tail = cache.page_table()[1].page_id;

    backend
        .append(
            &mut cache,
            &[tensor(0, 1, 9.0), tensor(1, 1, 9.0)],
            &[tensor(0, 1, 8.0), tensor(1, 1, 8.0)],
        )
        .unwrap();

    assert_eq!(frozen.sequence_length, 3);
    assert_eq!(backend.snapshot_from_lease(&lease).unwrap(), frozen);
    assert_eq!(cache.sequence_length, 4);
    assert_ne!(cache.page_table()[1].page_id, old_tail);
    assert_eq!(backend.page_metrics(&cache).cow_copies, 1);
}

#[test]
fn prefix_sharing_is_bound_to_principal_and_manifest() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let mut source = backend
        .allocate_bound(schema(), 8, page_binding("alice", page_size))
        .unwrap();
    backend
        .append(
            &mut source,
            &[tensor(0, 4, 1.0), tensor(1, 4, 2.0)],
            &[tensor(0, 4, 3.0), tensor(1, 4, 4.0)],
        )
        .unwrap();
    let prefix = backend.publish_prefix(&mut source).unwrap();

    let mut same_principal = backend
        .allocate_bound(schema(), 8, page_binding("alice", page_size))
        .unwrap();
    backend.fork_prefix(&mut same_principal, &prefix).unwrap();
    assert_eq!(same_principal.sequence_length, 4);
    assert_eq!(backend.page_metrics(&same_principal).shared_pages, 2);

    let mut other_principal = backend
        .allocate_bound(schema(), 8, page_binding("bob", page_size))
        .unwrap();
    assert_eq!(
        backend.fork_prefix(&mut other_principal, &prefix),
        Err(ptr_pods::KvBackendError::PrefixMismatch)
    );
}

#[test]
fn paged_oom_does_not_mutate_cache() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let mut cache = backend
        .allocate_bound(schema(), 2, page_binding("alice", page_size))
        .unwrap();
    let before = backend.page_metrics(&cache);
    assert_eq!(
        backend.append(
            &mut cache,
            &[tensor(0, 3, 1.0), tensor(1, 3, 1.0)],
            &[tensor(0, 3, 1.0), tensor(1, 3, 1.0)]
        ),
        Err(ptr_pods::KvBackendError::OutOfMemory)
    );
    assert_eq!(cache.sequence_length, 0);
    assert_eq!(backend.page_metrics(&cache), before);
}

#[test]
fn paged_snapshot_digest_binds_tensor_schema_and_page_structure() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let mut cache = backend
        .allocate_bound(schema(), 8, page_binding("alice", page_size))
        .unwrap();
    backend
        .append(
            &mut cache,
            &[tensor(0, 3, 1.0), tensor(1, 3, 2.0)],
            &[tensor(0, 3, 3.0), tensor(1, 3, 4.0)],
        )
        .unwrap();
    let lease = backend.seal_snapshot(&mut cache).unwrap();
    let snapshot = backend.snapshot_from_lease(&lease).unwrap();

    let mut schema_tamper = snapshot.clone();
    schema_tamper.schema.head_dim += 1;
    assert!(!schema_tamper.verify_digest());

    let mut ordinal_tamper = snapshot;
    ordinal_tamper.pages[1].ordinal = 0;
    assert!(!ordinal_tamper.verify_digest());
}

#[test]
fn paged_prefill_then_decode_matches_full_causal_recompute() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let mut attention_schema = schema();
    attention_schema.layer_count = 1;
    let mut continued = backend
        .allocate_bound(
            attention_schema.clone(),
            8,
            page_binding("alice", page_size),
        )
        .unwrap();
    let mut recomputed = backend
        .allocate_bound(attention_schema, 8, page_binding("alice", page_size))
        .unwrap();
    let executor =
        ReferenceCausalAttentionExecutor::new(CausalAttentionWeights::identity(4).unwrap(), 2)
            .unwrap();
    let first = vec![1.0, 0.0, 0.5, -0.5];
    let second = vec![0.0, 1.0, -0.5, 0.5];
    let third = vec![0.5, 0.5, 1.0, 0.0];
    executor
        .prefill(
            &backend,
            &mut continued,
            CausalAttentionInput {
                tokens: vec![first.clone(), second.clone()],
            },
        )
        .unwrap();
    let decoded = executor
        .decode(
            &backend,
            &mut continued,
            CausalAttentionInput {
                tokens: vec![third.clone()],
            },
        )
        .unwrap();
    let full = executor
        .prefill(
            &backend,
            &mut recomputed,
            CausalAttentionInput {
                tokens: vec![first, second, third],
            },
        )
        .unwrap();
    let expected = full.hidden_states.last().unwrap();
    let actual = decoded.hidden_states.last().unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(decoded.first_position, 2);
    assert_eq!(decoded.next_position, 3);
    assert_eq!(continued.sequence_length, recomputed.sequence_length);
}

fn linear(output_size: usize, input_size: usize, diagonal: bool, bias: bool) -> Qwen2LinearWeights {
    let mut weight = vec![0.0; output_size * input_size];
    if diagonal {
        for index in 0..output_size.min(input_size) {
            weight[index * input_size + index] = 1.0;
        }
    }
    Qwen2LinearWeights {
        output_size,
        input_size,
        weight,
        bias: bias.then(|| vec![0.0; output_size]),
    }
}

fn qwen_config() -> Qwen2LayerConfig {
    Qwen2LayerConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_position_embeddings: 32,
        rope_theta: 10_000.0,
        rope_scaling: Qwen2RopeScaling::None,
        rms_norm_eps: 1e-6,
        hidden_act: Qwen2Activation::Silu,
        sliding_window: None,
    }
}

fn qwen_weights() -> Qwen2DecoderWeights {
    Qwen2DecoderWeights {
        input_layernorm: vec![1.0; 4],
        query: linear(4, 4, true, true),
        key: linear(2, 4, true, true),
        value: linear(2, 4, true, true),
        output: linear(4, 4, true, false),
        post_attention_layernorm: vec![1.0; 4],
        gate: linear(8, 4, false, false),
        up: linear(8, 4, false, false),
        down: linear(4, 8, false, false),
    }
}

#[test]
fn qwen_config_rejects_invalid_grouped_query_attention() {
    let mut config = qwen_config();
    config.num_attention_heads = 3;
    assert!(config.validate().is_err());
    config.num_attention_heads = 2;
    config.rope_theta = 0.0;
    assert!(config.validate().is_err());
    config.rope_theta = 10_000.0;
    config.rope_scaling = Qwen2RopeScaling::Unsupported("yarn".into());
    assert!(config.validate().is_err());
}

#[test]
fn qwen_gqa_prefill_decode_matches_recompute_and_keeps_compact_kv() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = InMemoryPagedKvBackend::new(page_size);
    let config = qwen_config();
    let executor = Qwen2PagedDecoder::new(config.clone(), qwen_weights()).unwrap();
    let qwen_schema = KvTensorSchema {
        model: ModelVersion::from("qwen2-test"),
        adapter: AdapterVersion::from("base"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: config.num_attention_heads,
        key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim().unwrap(),
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let binding = KvPageBinding {
        model: qwen_schema.model.clone(),
        adapter: qwen_schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [21; 32],
        principal: PrincipalId::from("alice"),
        device: qwen_schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: page_size,
    };
    let mut continued = backend
        .allocate_bound(qwen_schema.clone(), 8, binding.clone())
        .unwrap();
    let mut recomputed = backend.allocate_bound(qwen_schema, 8, binding).unwrap();
    let first = vec![1.0, 0.0, 0.5, -0.5];
    let second = vec![0.0, 1.0, -0.5, 0.5];
    let third = vec![0.5, 0.5, 1.0, 0.0];

    executor
        .prefill(
            &backend,
            &mut continued,
            Qwen2DecoderInput {
                hidden_states: vec![first.clone(), second.clone()],
            },
        )
        .unwrap();
    let decoded = executor
        .decode(
            &backend,
            &mut continued,
            Qwen2DecoderInput {
                hidden_states: vec![third.clone()],
            },
        )
        .unwrap();
    let full = executor
        .prefill(
            &backend,
            &mut recomputed,
            Qwen2DecoderInput {
                hidden_states: vec![first, second, third],
            },
        )
        .unwrap();

    for (actual, expected) in decoded.hidden_states[0]
        .iter()
        .zip(full.hidden_states.last().unwrap())
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    let snapshot = backend.snapshot(&continued).unwrap();
    assert_eq!(snapshot.layers[0].keys.shape, vec![3, 2]);
    assert_eq!(snapshot.layers[0].values.shape, vec![3, 2]);
    assert_eq!(decoded.first_position, 2);
    assert_eq!(decoded.next_position, 3);
}
