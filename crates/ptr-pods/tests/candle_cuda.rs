#![cfg(feature = "candle-cuda")]

use ptr_pods::{
    CandleDenseExecutor, CandleGpuTierBackend, CandleKvTensorBackend, CausalAttentionInput,
    CausalAttentionWeights, KvContinuationExecutor, KvPageBinding, KvPageSize, KvTensorBackend,
    KvTensorDType, KvTensorSchema, LeaseState, NeuralPodDescriptor, NeuralPodExecutor,
    NeuralPodType, PagedKvTensorBackend, PodLifecycle, Qwen2Activation, Qwen2DecoderInput,
    Qwen2DecoderWeights, Qwen2LayerConfig, Qwen2PagedDecoder, Qwen2RopeScaling,
    ReferenceCausalAttentionExecutor, ResourceRequirements, TensorContract, TensorDType, TensorRef,
    TypedPayload,
};
use ptr_storage::{ChunkDescriptor, TierBackend, TierBackendId};
use ptr_types::{
    AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, PodId, PrincipalId, TypeId,
};
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use std::future::Future;
use std::path::PathBuf;
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

struct TestTensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for TestTensor {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn descriptor() -> NeuralPodDescriptor {
    NeuralPodDescriptor {
        artifact_id: ArtifactId::from("cuda-test-artifact"),
        pod_id: PodId::from("cuda-test-pod"),
        manifest_hash: [1; 32],
        generation: Generation(1),
        pod_type: NeuralPodType::Model,
        input_schema: TypeId::from("vector"),
        output_schema: TypeId::from("vector"),
        capabilities: vec![],
        provenance: vec!["test:safetensors".into()],
        lifecycle: PodLifecycle::Ready,
        resources: ResourceRequirements {
            ram_bytes: 1,
            vram_bytes: 1,
            device: None,
            max_concurrency: 1,
        },
        tensor: TensorContract {
            dtype: TensorDType::F32,
            input_len: 2,
            output_len: 2,
        },
    }
}

#[test]
fn safetensors_matmul_runs_on_cuda_device() {
    let path: PathBuf =
        std::env::temp_dir().join(format!("ptr-candle-{}.safetensors", std::process::id()));
    let tensors = vec![
        (
            "weight",
            TestTensor {
                shape: vec![2, 2],
                bytes: f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            },
        ),
        (
            "bias",
            TestTensor {
                shape: vec![2],
                bytes: f32_bytes(&[0.5, -0.5]),
            },
        ),
    ];
    safetensors::tensor::serialize_to_file(tensors, None, &path).unwrap();
    let executor =
        CandleDenseExecutor::from_safetensors(descriptor(), &path, "weight", Some("bias"), 0)
            .unwrap();
    let mut lease = executor.activate(&descriptor()).unwrap();
    let output = executor
        .infer(
            &mut lease,
            TypedPayload {
                type_id: TypeId::from("vector"),
                bytes: f32_bytes(&[1.0, 2.0]),
            },
        )
        .unwrap();
    assert_eq!(output.type_id, TypeId::from("vector"));
    assert_eq!(output.bytes.len(), 2 * std::mem::size_of::<f32>());
    assert_eq!(lease.state, LeaseState::Ready);
    executor.release(lease).unwrap();
    std::fs::remove_file(path).unwrap();
}

#[test]
fn failed_cuda_kv_append_releases_tensor_use_for_recovery() {
    let backend = CandleKvTensorBackend::new(0);
    let cache_schema = KvTensorSchema {
        model: ModelVersion::from("cuda-model"),
        adapter: AdapterVersion::from("cuda-adapter"),
        device: DeviceId::from("cuda:0"),
        layer_count: 1,
        attention_heads: 1,
        key_value_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let mut cache = backend.allocate(cache_schema, 2).unwrap();
    let invalid = backend.append(
        &mut cache,
        &[TensorRef {
            layer: 0,
            shape: vec![1, 3],
            values: vec![1.0, 2.0, 3.0],
        }],
        &[TensorRef {
            layer: 0,
            shape: vec![1, 3],
            values: vec![1.0, 2.0, 3.0],
        }],
    );
    assert!(invalid.is_err());
    backend
        .append(
            &mut cache,
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
    backend.release(cache).unwrap();
}

#[test]
fn paged_cuda_snapshot_uses_cow_and_continues_causal_attention() {
    let page_size = KvPageSize::new(2).unwrap();
    let backend = CandleKvTensorBackend::with_page_size(0, page_size);
    let cache_schema = KvTensorSchema {
        model: ModelVersion::from("cuda-attention"),
        adapter: AdapterVersion::from("cuda-adapter"),
        device: DeviceId::from("cuda:0"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let binding = KvPageBinding {
        model: cache_schema.model.clone(),
        adapter: cache_schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [5; 32],
        principal: PrincipalId::from("cuda-test"),
        device: cache_schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: page_size,
    };
    let mut cache = backend.allocate_bound(cache_schema, 8, binding).unwrap();
    let attention_path: PathBuf = std::env::temp_dir().join(format!(
        "ptr-candle-attention-{}.safetensors",
        std::process::id()
    ));
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let tensors = ["q_proj", "k_proj", "v_proj", "o_proj"]
        .into_iter()
        .map(|name| {
            (
                name,
                TestTensor {
                    shape: vec![4, 4],
                    bytes: f32_bytes(&identity),
                },
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(tensors, None, &attention_path).unwrap();
    let weights = CausalAttentionWeights::from_safetensors(
        &attention_path,
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    .unwrap();
    let executor = ReferenceCausalAttentionExecutor::new(weights, 2).unwrap();
    executor
        .prefill(
            &backend,
            &mut cache,
            CausalAttentionInput {
                tokens: vec![
                    vec![1.0, 0.0, 0.5, -0.5],
                    vec![0.0, 1.0, -0.5, 0.5],
                    vec![0.5, 0.5, 1.0, 0.0],
                ],
            },
        )
        .unwrap();
    let lease = backend.seal_snapshot(&mut cache).unwrap();
    let frozen = backend.snapshot_from_lease(&lease).unwrap();
    let old_tail = cache.page_table()[1].page_id;
    let decoded = executor
        .decode(
            &backend,
            &mut cache,
            CausalAttentionInput {
                tokens: vec![vec![0.25, 0.75, 0.0, 1.0]],
            },
        )
        .unwrap();
    assert_eq!(frozen.sequence_length, 3);
    assert_eq!(cache.sequence_length(), 4);
    assert_ne!(cache.page_table()[1].page_id, old_tail);
    assert_eq!(decoded.first_position, 3);
    assert_eq!(backend.page_metrics(&cache).cow_copies, 1);
    backend.release(cache).unwrap();
    std::fs::remove_file(attention_path).unwrap();
}

#[test]
fn cuda_gpu_tier_backend_roundtrips_verified_chunk() {
    let backend = CandleGpuTierBackend::new(TierBackendId::from("cuda-tier"), 0).unwrap();
    let bytes = b"paged-kv-gpu-chunk".to_vec();
    let descriptor = ChunkDescriptor::for_bytes(0, 0, &bytes);
    block_on(backend.put_chunk(descriptor.clone(), bytes.clone())).unwrap();
    assert_eq!(block_on(backend.get_chunk(&descriptor)).unwrap(), bytes);
    assert!(block_on(backend.verify_chunk(&descriptor)).unwrap().valid);
    block_on(backend.delete_chunk(&descriptor)).unwrap();
    assert!(block_on(backend.get_chunk(&descriptor)).is_err());
}

#[test]
fn qwen2_safetensors_layer_runs_gqa_on_paged_cuda_cache() {
    let config = Qwen2LayerConfig {
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
    };
    let path = std::env::temp_dir().join(format!(
        "ptr-candle-qwen2-{}.safetensors",
        std::process::id()
    ));
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut tensors = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>, values: &[f32]| {
        tensors.push((
            name.to_owned(),
            TestTensor {
                shape,
                bytes: f32_bytes(values),
            },
        ));
    };
    push("model.layers.0.input_layernorm.weight", vec![4], &[1.0; 4]);
    push(
        "model.layers.0.post_attention_layernorm.weight",
        vec![4],
        &[1.0; 4],
    );
    push(
        "model.layers.0.self_attn.q_proj.weight",
        vec![4, 4],
        &identity,
    );
    push(
        "model.layers.0.self_attn.k_proj.weight",
        vec![2, 4],
        &identity[..8],
    );
    push(
        "model.layers.0.self_attn.v_proj.weight",
        vec![2, 4],
        &identity[..8],
    );
    push(
        "model.layers.0.self_attn.o_proj.weight",
        vec![4, 4],
        &identity,
    );
    for projection in ["q_proj", "k_proj", "v_proj"] {
        let length = if projection == "q_proj" { 4 } else { 2 };
        push(
            &format!("model.layers.0.self_attn.{projection}.bias"),
            vec![length],
            &vec![0.0; length],
        );
    }
    push(
        "model.layers.0.mlp.gate_proj.weight",
        vec![8, 4],
        &[0.0; 32],
    );
    push("model.layers.0.mlp.up_proj.weight", vec![8, 4], &[0.0; 32]);
    push(
        "model.layers.0.mlp.down_proj.weight",
        vec![4, 8],
        &[0.0; 32],
    );
    safetensors::tensor::serialize_to_file(tensors, None, &path).unwrap();

    let weights = Qwen2DecoderWeights::from_safetensors(&path, 0, &config).unwrap();
    let executor = Qwen2PagedDecoder::new(config.clone(), weights).unwrap();
    let page_size = KvPageSize::new(2).unwrap();
    let backend = CandleKvTensorBackend::with_page_size(0, page_size);
    let schema = KvTensorSchema {
        model: ModelVersion::from("qwen2-test"),
        adapter: AdapterVersion::from("base"),
        device: DeviceId::from("cuda:0"),
        layer_count: 1,
        attention_heads: config.num_attention_heads,
        key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim().unwrap(),
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let binding = KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [27; 32],
        principal: PrincipalId::from("qwen-cuda"),
        device: schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: page_size,
    };
    let mut cache = backend.allocate_bound(schema, 8, binding).unwrap();
    executor
        .prefill(
            &backend,
            &mut cache,
            Qwen2DecoderInput {
                hidden_states: vec![vec![1.0, 0.0, 0.5, -0.5], vec![0.0, 1.0, -0.5, 0.5]],
            },
        )
        .unwrap();
    executor
        .decode(
            &backend,
            &mut cache,
            Qwen2DecoderInput {
                hidden_states: vec![vec![0.5, 0.5, 1.0, 0.0]],
            },
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    assert_eq!(snapshot.layers[0].keys.shape, vec![3, 2]);
    assert_eq!(snapshot.layers[0].values.shape, vec![3, 2]);
    backend.release(cache).unwrap();
    std::fs::remove_file(path).unwrap();
}
