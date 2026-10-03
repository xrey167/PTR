#![cfg(feature = "candle-cuda")]

use crate::{
    DescriptorError, DeviceLease, DeviceLeaseError, InMemoryKvTensorBackend, KvBackendError,
    KvCacheLayout, KvCacheTier, KvLayerSnapshot, KvTensorBackend, KvTensorDType, KvTensorSchema,
    KvTensorSnapshot, LeaseState, NeuralPodDescriptor, NeuralPodError, NeuralPodExecutor,
    NeuralPodLease, TensorDType, TensorRef, TypedPayload,
};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

#[derive(Debug)]
pub enum CandleExecutorError {
    Descriptor(DescriptorError),
    Candle(candle_core::Error),
    InvalidInput(String),
    InvalidLease(NeuralPodError),
}

pub struct CandleDenseExecutor {
    descriptor: NeuralPodDescriptor,
    device: Device,
    weight: Tensor,
    bias: Option<Tensor>,
}

pub struct CandleKvCache {
    schema: KvTensorSchema,
    layout: KvCacheLayout,
    capacity_tokens: usize,
    sequence_length: usize,
    position_offset: usize,
    layers: Vec<(Tensor, Tensor)>,
    device_lease: DeviceLease,
}

#[derive(Clone, Copy, Debug)]
pub struct CandleKvTensorBackend {
    pub device_id: usize,
}

impl CandleKvTensorBackend {
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    fn device(&self) -> Result<Device, KvBackendError> {
        Device::new_cuda(self.device_id).map_err(|error| KvBackendError::Backend(error.to_string()))
    }

    fn tensor_from_ref(&self, value: &TensorRef) -> Result<Tensor, KvBackendError> {
        let device = self.device()?;
        if value.shape.len() != 2 || value.values.len() != value.shape[0] * value.shape[1] {
            return Err(KvBackendError::ShapeMismatch);
        }
        Tensor::from_vec(
            value.values.clone(),
            (value.shape[0], value.shape[1]),
            &device,
        )
        .map_err(|error| KvBackendError::Backend(error.to_string()))
    }

    fn tensor_to_ref(layer: usize, tensor: &Tensor) -> Result<TensorRef, KvBackendError> {
        let values = tensor
            .to_vec2::<f32>()
            .map_err(|error| KvBackendError::Backend(error.to_string()))?;
        let rows = values.len();
        let cols = values.first().map_or(0, Vec::len);
        Ok(TensorRef {
            layer,
            shape: vec![rows, cols],
            values: values.into_iter().flatten().collect(),
        })
    }

    fn lease_error(error: DeviceLeaseError) -> KvBackendError {
        KvBackendError::Backend(format!("device lease: {error:?}"))
    }
}

impl KvTensorBackend for CandleKvTensorBackend {
    type Cache = CandleKvCache;

    fn allocate(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
    ) -> Result<Self::Cache, KvBackendError> {
        InMemoryKvTensorBackend::validate_schema(&schema)?;
        if schema.device.0 != format!("cuda:{}", self.device_id) || capacity_tokens == 0 {
            return Err(KvBackendError::DeviceMismatch);
        }
        let _ = self.device()?;
        let device_id = schema.device.clone();
        Ok(CandleKvCache {
            schema,
            layout: KvCacheLayout {
                page_tokens: 1,
                tier: KvCacheTier::Gpu,
                dtype: KvTensorDType::F32,
            },
            capacity_tokens,
            sequence_length: 0,
            position_offset: 0,
            layers: Vec::new(),
            device_lease: DeviceLease::new(device_id, 0, 0, 0),
        })
    }

    fn append(
        &self,
        cache: &mut Self::Cache,
        keys: &[TensorRef],
        values: &[TensorRef],
    ) -> Result<(), KvBackendError> {
        cache
            .device_lease
            .begin_tensor()
            .map_err(Self::lease_error)?;
        let result = (|| {
            if keys.len() != cache.schema.layer_count || values.len() != keys.len() {
                return Err(KvBackendError::InvalidLayer(keys.len()));
            }
            let width = cache.schema.attention_heads * cache.schema.head_dim;
            let token_count = keys.first().map_or(0, |tensor| tensor.shape[0]);
            if token_count == 0 || cache.sequence_length + token_count > cache.capacity_tokens {
                return Err(KvBackendError::ShapeMismatch);
            }
            for layer in 0..cache.schema.layer_count {
                if keys[layer].shape != vec![token_count, width]
                    || values[layer].shape != vec![token_count, width]
                {
                    return Err(KvBackendError::ShapeMismatch);
                }
                if keys[layer].values.len() != token_count * width
                    || values[layer].values.len() != token_count * width
                {
                    return Err(KvBackendError::ShapeMismatch);
                }
                let key = self.tensor_from_ref(&keys[layer])?;
                let value = self.tensor_from_ref(&values[layer])?;
                if cache.layers.is_empty() {
                    cache.layers.push((key, value));
                } else {
                    let old = &mut cache.layers[layer];
                    old.0 = Tensor::cat(&[old.0.clone(), key], 0)
                        .map_err(|error| KvBackendError::Backend(error.to_string()))?;
                    old.1 = Tensor::cat(&[old.1.clone(), value], 0)
                        .map_err(|error| KvBackendError::Backend(error.to_string()))?;
                }
            }
            cache.sequence_length += token_count;
            Ok(())
        })();
        cache.device_lease.end_tensor().map_err(Self::lease_error)?;
        result
    }

    fn truncate(&self, cache: &mut Self::Cache, new_length: usize) -> Result<(), KvBackendError> {
        cache
            .device_lease
            .begin_tensor()
            .map_err(Self::lease_error)?;
        let result = (|| {
            if new_length > cache.sequence_length {
                return Err(KvBackendError::InvalidTruncation);
            }
            for (key, value) in &mut cache.layers {
                *key = key
                    .narrow(0, 0, new_length)
                    .map_err(|error| KvBackendError::Backend(error.to_string()))?;
                *value = value
                    .narrow(0, 0, new_length)
                    .map_err(|error| KvBackendError::Backend(error.to_string()))?;
            }
            cache.sequence_length = new_length;
            Ok(())
        })();
        cache.device_lease.end_tensor().map_err(Self::lease_error)?;
        result
    }

    fn snapshot(&self, cache: &Self::Cache) -> Result<KvTensorSnapshot, KvBackendError> {
        let layers = cache
            .layers
            .iter()
            .enumerate()
            .map(|(layer, (key, value))| {
                Ok(KvLayerSnapshot {
                    keys: Self::tensor_to_ref(layer, key)?,
                    values: Self::tensor_to_ref(layer, value)?,
                })
            })
            .collect::<Result<Vec<_>, KvBackendError>>()?;
        let mut snapshot = KvTensorSnapshot {
            schema: cache.schema.clone(),
            layout: cache.layout,
            capacity_tokens: cache.capacity_tokens,
            sequence_length: cache.sequence_length,
            position_offset: cache.position_offset,
            layers,
            digest: [0; 32],
        };
        snapshot.digest = InMemoryKvTensorBackend::digest(&snapshot);
        Ok(snapshot)
    }

    fn restore(
        &self,
        snapshot: KvTensorSnapshot,
        device: &ptr_types::DeviceId,
    ) -> Result<Self::Cache, KvBackendError> {
        if device != &snapshot.schema.device
            || snapshot.schema.device.0 != format!("cuda:{}", self.device_id)
            || snapshot.layout.tier != KvCacheTier::Gpu
            || InMemoryKvTensorBackend::digest(&snapshot) != snapshot.digest
        {
            return Err(KvBackendError::DeviceMismatch);
        }
        let mut cache = self.allocate(snapshot.schema.clone(), snapshot.capacity_tokens)?;
        let keys: Vec<_> = snapshot
            .layers
            .iter()
            .map(|layer| layer.keys.clone())
            .collect();
        let values: Vec<_> = snapshot
            .layers
            .iter()
            .map(|layer| layer.values.clone())
            .collect();
        self.append(&mut cache, &keys, &values)?;
        cache.position_offset = snapshot.position_offset;
        Ok(cache)
    }

    fn release(&self, mut cache: Self::Cache) -> Result<(), KvBackendError> {
        cache.device_lease.release().map_err(Self::lease_error)?;
        Ok(())
    }
}

impl CandleDenseExecutor {
    /// Loads an F32 `[output, input]` weight and optional F32 bias directly
    /// from Safetensors onto the selected CUDA device.
    pub fn from_safetensors<P: AsRef<Path>>(
        descriptor: NeuralPodDescriptor,
        path: P,
        weight_name: &str,
        bias_name: Option<&str>,
        device_id: usize,
    ) -> Result<Self, CandleExecutorError> {
        descriptor
            .validate()
            .map_err(CandleExecutorError::Descriptor)?;
        let device = Device::new_cuda(device_id).map_err(CandleExecutorError::Candle)?;
        let weights = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(CandleExecutorError::Candle)?;
        let weight = weights
            .load(weight_name, &device)
            .map_err(CandleExecutorError::Candle)?;
        if weight.dtype() != DType::F32 || weight.dims().len() != 2 {
            return Err(CandleExecutorError::InvalidInput(
                "weight must be an F32 rank-2 tensor".into(),
            ));
        }
        let dims = weight.dims();
        if descriptor.tensor.dtype != TensorDType::F32
            || dims[1] != descriptor.tensor.input_len
            || dims[0] != descriptor.tensor.output_len
        {
            return Err(CandleExecutorError::InvalidInput(
                "weight dimensions do not match descriptor tensor contract".into(),
            ));
        }
        let bias = bias_name
            .map(|name| weights.load(name, &device))
            .transpose()
            .map_err(CandleExecutorError::Candle)?;
        if let Some(bias) = &bias {
            if bias.dtype() != DType::F32 || bias.dims() != [descriptor.tensor.output_len] {
                return Err(CandleExecutorError::InvalidInput(
                    "bias must be an F32 vector matching output dimension".into(),
                ));
            }
        }
        Ok(Self {
            descriptor,
            device,
            weight,
            bias,
        })
    }

    fn infer_inner(&self, input: TypedPayload) -> Result<TypedPayload, CandleExecutorError> {
        if input.type_id != self.descriptor.input_schema {
            return Err(CandleExecutorError::InvalidInput(
                "input type does not match descriptor".into(),
            ));
        }
        if !input.bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
            return Err(CandleExecutorError::InvalidInput(
                "input bytes are not F32-aligned".into(),
            ));
        }
        let (chunks, remainder) = input.bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        let values: Vec<f32> = chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        let expected = self.weight.dims()[1];
        if values.len() != expected {
            return Err(CandleExecutorError::InvalidInput(format!(
                "input length {} does not match weight dimension {expected}",
                values.len()
            )));
        }
        let input = Tensor::from_vec(values, (expected, 1), &self.device)
            .map_err(CandleExecutorError::Candle)?;
        let mut output = self
            .weight
            .matmul(&input)
            .map_err(CandleExecutorError::Candle)?;
        if let Some(bias) = &self.bias {
            let bias = bias
                .reshape((self.descriptor.tensor.output_len, 1))
                .map_err(CandleExecutorError::Candle)?;
            output = output
                .broadcast_add(&bias)
                .map_err(CandleExecutorError::Candle)?;
        }
        let values = output
            .reshape((self.descriptor.tensor.output_len,))
            .map_err(CandleExecutorError::Candle)?
            .to_vec1::<f32>()
            .map_err(CandleExecutorError::Candle)?;
        let bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
        Ok(TypedPayload {
            type_id: self.descriptor.output_schema.clone(),
            bytes,
        })
    }
}

impl NeuralPodExecutor for CandleDenseExecutor {
    type Lease = NeuralPodLease;
    type Error = CandleExecutorError;

    fn activate(&self, descriptor: &NeuralPodDescriptor) -> Result<Self::Lease, Self::Error> {
        if descriptor != &self.descriptor {
            return Err(CandleExecutorError::Descriptor(
                DescriptorError::InvalidLifecycle,
            ));
        }
        Ok(NeuralPodLease {
            pod_id: descriptor.pod_id.clone(),
            generation: descriptor.generation,
            state: LeaseState::Ready,
        })
    }

    fn infer(
        &self,
        lease: &mut Self::Lease,
        input: TypedPayload,
    ) -> Result<TypedPayload, Self::Error> {
        lease.begin().map_err(CandleExecutorError::InvalidLease)?;
        let result = self.infer_inner(input);
        let finish = lease.finish().map_err(CandleExecutorError::InvalidLease);
        match (result, finish) {
            (Ok(output), Ok(())) => Ok(output),
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }

    fn release(&self, mut lease: Self::Lease) -> Result<(), Self::Error> {
        lease.release().map_err(CandleExecutorError::InvalidLease)
    }

    fn health(&self) -> bool {
        true
    }
}
