#![cfg(feature = "candle-cuda")]

use ptr_pods::{
    CandleDenseExecutor, CandleKvTensorBackend, KvTensorBackend, KvTensorDType, KvTensorSchema,
    LeaseState, NeuralPodDescriptor, NeuralPodExecutor, NeuralPodType, PodLifecycle,
    ResourceRequirements, TensorContract, TensorDType, TensorRef, TypedPayload,
};
use ptr_types::{AdapterVersion, ArtifactId, DeviceId, Generation, ModelVersion, PodId, TypeId};
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use std::path::PathBuf;

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
