use ptr_types::{AdapterVersion, DeviceId, ModelVersion};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvTensorDType {
    F32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvTensorSchema {
    pub model: ModelVersion,
    pub adapter: AdapterVersion,
    pub device: DeviceId,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    pub batch_size: usize,
    pub dtype: KvTensorDType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorRef {
    pub layer: usize,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvLayerSnapshot {
    pub keys: TensorRef,
    pub values: TensorRef,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvTensorSnapshot {
    pub schema: KvTensorSchema,
    pub capacity_tokens: usize,
    pub sequence_length: usize,
    pub position_offset: usize,
    pub layers: Vec<KvLayerSnapshot>,
    pub digest: [u8; 32],
}

impl KvTensorSnapshot {
    pub fn verify_digest(&self) -> bool {
        InMemoryKvTensorBackend::digest(self) == self.digest
    }

    pub fn rebind_device(&self, device: DeviceId) -> Self {
        let mut rebound = self.clone();
        rebound.schema.device = device;
        rebound.digest = InMemoryKvTensorBackend::digest(&rebound);
        rebound
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvBackendError {
    Backend(String),
    InvalidSchema(String),
    InvalidLayer(usize),
    ShapeMismatch,
    DtypeMismatch,
    DeviceMismatch,
    SnapshotDigestMismatch,
    SnapshotSchemaMismatch,
    InvalidTruncation,
}

pub trait KvTensorBackend: Send + Sync {
    type Cache: Send;

    fn allocate(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
    ) -> Result<Self::Cache, KvBackendError>;

    fn append(
        &self,
        cache: &mut Self::Cache,
        keys: &[TensorRef],
        values: &[TensorRef],
    ) -> Result<(), KvBackendError>;

    fn truncate(&self, cache: &mut Self::Cache, new_length: usize) -> Result<(), KvBackendError>;

    fn snapshot(&self, cache: &Self::Cache) -> Result<KvTensorSnapshot, KvBackendError>;

    fn restore(
        &self,
        snapshot: KvTensorSnapshot,
        device: &DeviceId,
    ) -> Result<Self::Cache, KvBackendError>;

    fn release(&self, cache: Self::Cache) -> Result<(), KvBackendError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct InMemoryKvCache {
    pub schema: KvTensorSchema,
    pub capacity_tokens: usize,
    pub sequence_length: usize,
    pub position_offset: usize,
    pub layers: Vec<KvLayerSnapshot>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InMemoryKvTensorBackend;

impl InMemoryKvTensorBackend {
    pub(crate) fn validate_schema(schema: &KvTensorSchema) -> Result<(), KvBackendError> {
        if schema.layer_count == 0
            || schema.attention_heads == 0
            || schema.head_dim == 0
            || schema.batch_size == 0
        {
            return Err(KvBackendError::InvalidSchema(
                "zero tensor dimension".into(),
            ));
        }
        Ok(())
    }

    fn validate_tensor(
        tensor: &TensorRef,
        schema: &KvTensorSchema,
        expected_layer: usize,
    ) -> Result<usize, KvBackendError> {
        if tensor.layer != expected_layer {
            return Err(KvBackendError::InvalidLayer(tensor.layer));
        }
        let expected_width = schema.attention_heads * schema.head_dim;
        if tensor.shape.len() != 2
            || tensor.shape[1] != expected_width
            || tensor.shape[0] == 0
            || tensor.values.len() != tensor.shape[0] * tensor.shape[1]
        {
            return Err(KvBackendError::ShapeMismatch);
        }
        Ok(tensor.shape[0])
    }

    pub(crate) fn digest(snapshot: &KvTensorSnapshot) -> [u8; 32] {
        let mut bytes = Vec::new();
        for value in [
            &snapshot.schema.model.0,
            &snapshot.schema.adapter.0,
            &snapshot.schema.device.0,
        ] {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        bytes.push(match snapshot.schema.dtype {
            KvTensorDType::F32 => 0,
        });
        bytes.extend_from_slice(&(snapshot.schema.layer_count as u64).to_le_bytes());
        bytes.extend_from_slice(&(snapshot.schema.attention_heads as u64).to_le_bytes());
        bytes.extend_from_slice(&(snapshot.schema.head_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(snapshot.schema.batch_size as u64).to_le_bytes());
        bytes.extend_from_slice(&(snapshot.capacity_tokens as u64).to_le_bytes());
        bytes.extend_from_slice(&(snapshot.sequence_length as u64).to_le_bytes());
        bytes.extend_from_slice(&(snapshot.position_offset as u64).to_le_bytes());
        for layer in &snapshot.layers {
            for tensor in [&layer.keys, &layer.values] {
                bytes.extend_from_slice(&(tensor.layer as u64).to_le_bytes());
                for dimension in &tensor.shape {
                    bytes.extend_from_slice(&(*dimension as u64).to_le_bytes());
                }
                for value in &tensor.values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Sha256::digest(bytes).into()
    }
}

impl KvTensorBackend for InMemoryKvTensorBackend {
    type Cache = InMemoryKvCache;

    fn allocate(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
    ) -> Result<Self::Cache, KvBackendError> {
        Self::validate_schema(&schema)?;
        if capacity_tokens == 0 {
            return Err(KvBackendError::InvalidSchema("zero token capacity".into()));
        }
        Ok(InMemoryKvCache {
            layers: Vec::new(),
            schema,
            capacity_tokens,
            sequence_length: 0,
            position_offset: 0,
        })
    }

    fn append(
        &self,
        cache: &mut Self::Cache,
        keys: &[TensorRef],
        values: &[TensorRef],
    ) -> Result<(), KvBackendError> {
        if keys.len() != cache.schema.layer_count || values.len() != keys.len() {
            return Err(KvBackendError::InvalidLayer(keys.len()));
        }
        let mut token_count = None;
        for layer in 0..cache.schema.layer_count {
            let key_tokens = Self::validate_tensor(&keys[layer], &cache.schema, layer)?;
            let value_tokens = Self::validate_tensor(&values[layer], &cache.schema, layer)?;
            if key_tokens != value_tokens || token_count.is_some_and(|count| count != key_tokens) {
                return Err(KvBackendError::ShapeMismatch);
            }
            token_count = Some(key_tokens);
        }
        let token_count = token_count.ok_or(KvBackendError::InvalidLayer(0))?;
        if cache.sequence_length.saturating_add(token_count) > cache.capacity_tokens {
            return Err(KvBackendError::InvalidSchema("KV capacity exceeded".into()));
        }
        if cache.layers.is_empty() {
            cache.layers = keys
                .iter()
                .zip(values)
                .map(|(key, value)| KvLayerSnapshot {
                    keys: key.clone(),
                    values: value.clone(),
                })
                .collect();
        } else {
            for layer in 0..cache.schema.layer_count {
                cache.layers[layer]
                    .keys
                    .values
                    .extend_from_slice(&keys[layer].values);
                cache.layers[layer]
                    .values
                    .values
                    .extend_from_slice(&values[layer].values);
                cache.layers[layer].keys.shape[0] += token_count;
                cache.layers[layer].values.shape[0] += token_count;
            }
        }
        cache.sequence_length += token_count;
        Ok(())
    }

    fn truncate(&self, cache: &mut Self::Cache, new_length: usize) -> Result<(), KvBackendError> {
        if new_length > cache.sequence_length {
            return Err(KvBackendError::InvalidTruncation);
        }
        let width = cache.schema.attention_heads * cache.schema.head_dim;
        for layer in &mut cache.layers {
            layer.keys.values.truncate(new_length * width);
            layer.values.values.truncate(new_length * width);
            layer.keys.shape[0] = new_length;
            layer.values.shape[0] = new_length;
        }
        cache.sequence_length = new_length;
        Ok(())
    }

    fn snapshot(&self, cache: &Self::Cache) -> Result<KvTensorSnapshot, KvBackendError> {
        let mut snapshot = KvTensorSnapshot {
            schema: cache.schema.clone(),
            capacity_tokens: cache.capacity_tokens,
            sequence_length: cache.sequence_length,
            position_offset: cache.position_offset,
            layers: cache.layers.clone(),
            digest: [0; 32],
        };
        snapshot.digest = Self::digest(&snapshot);
        Ok(snapshot)
    }

    fn restore(
        &self,
        snapshot: KvTensorSnapshot,
        device: &DeviceId,
    ) -> Result<Self::Cache, KvBackendError> {
        if &snapshot.schema.device != device {
            return Err(KvBackendError::DeviceMismatch);
        }
        if Self::digest(&snapshot) != snapshot.digest {
            return Err(KvBackendError::SnapshotDigestMismatch);
        }
        if snapshot.layers.len() != snapshot.schema.layer_count {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }
        Ok(InMemoryKvCache {
            schema: snapshot.schema,
            capacity_tokens: snapshot.capacity_tokens,
            sequence_length: snapshot.sequence_length,
            position_offset: snapshot.position_offset,
            layers: snapshot.layers,
        })
    }

    fn release(&self, _cache: Self::Cache) -> Result<(), KvBackendError> {
        Ok(())
    }
}
