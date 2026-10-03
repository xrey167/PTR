#![cfg(feature = "candle-cuda")]

use crate::{
    DescriptorError, DeviceLease, DeviceLeaseError, InMemoryKvTensorBackend, KvBackendError,
    KvCacheTier, KvLayerSnapshot, KvPageBinding, KvPageEntry, KvPageId, KvPageMetrics, KvPageSize,
    KvPageSnapshot, KvPageState, KvPagedSnapshot, KvPrefixKey, KvRuntimeLeaseBinding,
    KvSharedPrefix, KvTensorBackend, KvTensorSchema, KvTensorSnapshot, LeaseState,
    NeuralPodDescriptor, NeuralPodError, NeuralPodExecutor, NeuralPodLease, PagedKvTensorBackend,
    TensorDType, TensorRef, TypedPayload,
};
use candle_core::{DType, Device, Tensor};
use ptr_storage::{
    tier_digest, ChunkDescriptor, ChunkReceipt, StorageTier, TierBackend, TierBackendId,
    TierCapabilities, TierError, TierFuture, TierHealth, TierIntegrityReport,
};
use ptr_types::Digest;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

#[derive(Clone)]
struct CandlePageBundle {
    page_id: KvPageId,
    valid_tokens: u16,
    state: KvPageState,
    digest: Option<Digest>,
    layers: Vec<(Tensor, Tensor)>,
}

impl CandlePageBundle {
    fn entry(&self) -> KvPageEntry {
        KvPageEntry {
            page_id: self.page_id,
            valid_tokens: self.valid_tokens,
            digest: self.digest,
            state: self.state,
        }
    }
}

#[derive(Clone)]
struct CandlePrefixRecord {
    pages: Vec<Arc<CandlePageBundle>>,
    sequence_length: usize,
    position_offset: usize,
}

#[derive(Default)]
struct CandlePagedState {
    next_page_id: AtomicU64,
    next_stream_id: AtomicU64,
    prefixes: Mutex<BTreeMap<KvPrefixKey, CandlePrefixRecord>>,
}

pub struct CandlePagedKvCache {
    schema: KvTensorSchema,
    binding: KvPageBinding,
    capacity_tokens: usize,
    sequence_length: usize,
    position_offset: usize,
    pages: Vec<Arc<CandlePageBundle>>,
    device: Device,
    device_lease: DeviceLease,
    runtime_lease: Option<KvRuntimeLeaseBinding>,
    revoked: bool,
    cow_copies: u64,
}

impl CandlePagedKvCache {
    pub fn page_table(&self) -> Vec<KvPageEntry> {
        self.pages.iter().map(|page| page.entry()).collect()
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

pub type CandleKvCache = CandlePagedKvCache;

pub struct CandleKvSnapshotLease {
    schema: KvTensorSchema,
    binding: KvPageBinding,
    capacity_tokens: usize,
    sequence_length: usize,
    position_offset: usize,
    digest: Digest,
    pages: Vec<Arc<CandlePageBundle>>,
    _device: Device,
}

#[derive(Clone)]
pub struct CandlePagedKvBackend {
    pub device_id: usize,
    page_tokens: KvPageSize,
    state: Arc<CandlePagedState>,
}

pub type CandleKvTensorBackend = CandlePagedKvBackend;

pub struct CandleGpuTierBackend {
    id: TierBackendId,
    device: Device,
    chunks: Mutex<BTreeMap<Digest, Tensor>>,
}

impl CandleGpuTierBackend {
    pub fn new(id: TierBackendId, device_id: usize) -> Result<Self, TierError> {
        if id.0.trim().is_empty() || id.0 != id.0.trim() {
            return Err(TierError::InvalidBackendId);
        }
        let device = Device::new_cuda_with_stream(device_id)
            .map_err(|error| TierError::Backend(error.to_string()))?;
        Ok(Self {
            id,
            device,
            chunks: Mutex::new(BTreeMap::new()),
        })
    }

    fn read_tensor(tensor: &Tensor) -> Result<Vec<u8>, TierError> {
        tensor
            .to_vec1::<u8>()
            .map_err(|error| TierError::Backend(error.to_string()))
    }
}

impl TierBackend for CandleGpuTierBackend {
    fn identity(&self) -> TierBackendId {
        self.id.clone()
    }

    fn capabilities(&self) -> TierCapabilities {
        TierCapabilities {
            tier: StorageTier::Gpu,
            persistent: false,
            range_reads: false,
            conditional_create: true,
        }
    }

    fn put_chunk(&self, chunk: ChunkDescriptor, bytes: Vec<u8>) -> TierFuture<'_, ChunkReceipt> {
        Box::pin(async move {
            chunk.validate_bytes(&bytes)?;
            let mut chunks = self
                .chunks
                .lock()
                .map_err(|_| TierError::Backend("GPU tier lock poisoned".into()))?;
            if let Some(existing) = chunks.get(&chunk.digest) {
                if Self::read_tensor(existing)? != bytes {
                    return Err(TierError::ImmutableConflict);
                }
            } else {
                let length = bytes.len();
                let tensor = Tensor::from_vec(bytes, length, &self.device)
                    .map_err(|error| TierError::Backend(error.to_string()))?;
                chunks.insert(chunk.digest, tensor);
            }
            Ok(ChunkReceipt {
                backend: self.id.clone(),
                digest: chunk.digest,
                length: chunk.length,
            })
        })
    }

    fn get_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, Vec<u8>> {
        Box::pin(async move {
            let chunks = self
                .chunks
                .lock()
                .map_err(|_| TierError::Backend("GPU tier lock poisoned".into()))?;
            let tensor = chunks.get(&chunk.digest).ok_or(TierError::UnknownChunk)?;
            let bytes = Self::read_tensor(tensor)?;
            chunk
                .validate_bytes(&bytes)
                .map_err(|_| TierError::CorruptChunk)?;
            Ok(bytes)
        })
    }

    fn verify_chunk<'a>(
        &'a self,
        chunk: &'a ChunkDescriptor,
    ) -> TierFuture<'a, TierIntegrityReport> {
        Box::pin(async move {
            let chunks = self
                .chunks
                .lock()
                .map_err(|_| TierError::Backend("GPU tier lock poisoned".into()))?;
            let tensor = chunks.get(&chunk.digest).ok_or(TierError::UnknownChunk)?;
            let bytes = Self::read_tensor(tensor)?;
            Ok(TierIntegrityReport {
                digest: tier_digest(&bytes),
                length: bytes.len() as u64,
                valid: chunk.validate_bytes(&bytes).is_ok(),
            })
        })
    }

    fn delete_chunk<'a>(&'a self, chunk: &'a ChunkDescriptor) -> TierFuture<'a, ()> {
        Box::pin(async move {
            self.chunks
                .lock()
                .map_err(|_| TierError::Backend("GPU tier lock poisoned".into()))?
                .remove(&chunk.digest);
            Ok(())
        })
    }

    fn health(&self) -> TierFuture<'_, TierHealth> {
        Box::pin(async move {
            self.device
                .synchronize()
                .map_err(|error| TierError::Backend(error.to_string()))?;
            Ok(TierHealth::Healthy)
        })
    }
}

impl CandlePagedKvBackend {
    pub fn new(device_id: usize) -> Self {
        Self::with_page_size(device_id, KvPageSize::DEFAULT)
    }

    pub fn with_page_size(device_id: usize, page_tokens: KvPageSize) -> Self {
        Self {
            device_id,
            page_tokens,
            state: Arc::new(CandlePagedState::default()),
        }
    }

    fn new_device(&self) -> Result<Device, KvBackendError> {
        Device::new_cuda_with_stream(self.device_id).map_err(Self::candle_error)
    }

    fn next_page_id(&self) -> Result<KvPageId, KvBackendError> {
        let value = self.state.next_page_id.fetch_add(1, Ordering::Relaxed);
        if value == u64::MAX {
            return Err(KvBackendError::OutOfMemory);
        }
        Ok(KvPageId(value + 1))
    }

    fn next_stream_id(&self) -> Result<u64, KvBackendError> {
        let value = self.state.next_stream_id.fetch_add(1, Ordering::Relaxed);
        value.checked_add(1).ok_or(KvBackendError::OutOfMemory)
    }

    fn candle_error(error: candle_core::Error) -> KvBackendError {
        let message = error.to_string();
        if message.to_ascii_lowercase().contains("out of memory") {
            KvBackendError::OutOfMemory
        } else {
            KvBackendError::Backend(message)
        }
    }

    fn tensor_from_values(
        values: Vec<f32>,
        tokens: usize,
        width: usize,
        device: &Device,
    ) -> Result<Tensor, KvBackendError> {
        Tensor::from_vec(values, (tokens, width), device).map_err(Self::candle_error)
    }

    fn tensor_to_ref(layer: usize, tensor: &Tensor) -> Result<TensorRef, KvBackendError> {
        let values = tensor.to_vec2::<f32>().map_err(Self::candle_error)?;
        let rows = values.len();
        let cols = values.first().map_or(0, Vec::len);
        Ok(TensorRef {
            layer,
            shape: vec![rows, cols],
            values: values.into_iter().flatten().collect(),
        })
    }

    fn lease_error(error: DeviceLeaseError) -> KvBackendError {
        match error {
            DeviceLeaseError::Revoked | DeviceLeaseError::Released | DeviceLeaseError::Failed => {
                KvBackendError::StaleLease
            }
            _ => KvBackendError::Backend(format!("device lease: {error:?}")),
        }
    }

    fn validate_append(
        cache: &CandlePagedKvCache,
        keys: &[TensorRef],
        values: &[TensorRef],
    ) -> Result<usize, KvBackendError> {
        if cache.revoked {
            return Err(KvBackendError::StaleLease);
        }
        if keys.len() != cache.schema.layer_count || values.len() != keys.len() {
            return Err(KvBackendError::InvalidLayer(keys.len()));
        }
        let width = cache.schema.key_value_heads * cache.schema.head_dim;
        let tokens = keys
            .first()
            .and_then(|tensor| tensor.shape.first())
            .copied()
            .unwrap_or(0);
        if tokens == 0 || cache.sequence_length.saturating_add(tokens) > cache.capacity_tokens {
            return Err(KvBackendError::OutOfMemory);
        }
        for layer in 0..cache.schema.layer_count {
            for tensor in [&keys[layer], &values[layer]] {
                if tensor.layer != layer
                    || tensor.shape != [tokens, width]
                    || tensor.values.len() != tokens * width
                {
                    return Err(KvBackendError::ShapeMismatch);
                }
            }
        }
        Ok(tokens)
    }

    fn new_page(
        &self,
        cache: &CandlePagedKvCache,
        keys: &[TensorRef],
        values: &[TensorRef],
        start: usize,
        count: usize,
    ) -> Result<CandlePageBundle, KvBackendError> {
        let width = cache.schema.key_value_heads * cache.schema.head_dim;
        let mut layers = Vec::with_capacity(cache.schema.layer_count);
        for layer in 0..cache.schema.layer_count {
            let range = start * width..(start + count) * width;
            layers.push((
                Self::tensor_from_values(
                    keys[layer].values[range.clone()].to_vec(),
                    count,
                    width,
                    &cache.device,
                )?,
                Self::tensor_from_values(
                    values[layer].values[range].to_vec(),
                    count,
                    width,
                    &cache.device,
                )?,
            ));
        }
        Ok(CandlePageBundle {
            page_id: self.next_page_id()?,
            valid_tokens: u16::try_from(count).map_err(|_| KvBackendError::OutOfMemory)?,
            state: KvPageState::Writable,
            digest: None,
            layers,
        })
    }

    fn page_snapshot(
        binding: &KvPageBinding,
        ordinal: usize,
        page: &CandlePageBundle,
    ) -> Result<KvPageSnapshot, KvBackendError> {
        let layers = page
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
        let digest = crate::paged_kv::page_digest(binding, ordinal, page.valid_tokens, &layers);
        Ok(KvPageSnapshot {
            ordinal,
            valid_tokens: page.valid_tokens,
            layers,
            digest,
        })
    }

    fn seal_page(
        binding: &KvPageBinding,
        ordinal: usize,
        page: &mut CandlePageBundle,
    ) -> Result<(), KvBackendError> {
        if matches!(
            page.state,
            KvPageState::Revoked | KvPageState::Corrupt | KvPageState::Released
        ) {
            return Err(KvBackendError::InvalidPageState);
        }
        let snapshot = Self::page_snapshot(binding, ordinal, page)?;
        page.digest = Some(snapshot.digest);
        page.state = KvPageState::Sealed;
        Ok(())
    }
}

impl KvTensorBackend for CandlePagedKvBackend {
    type Cache = CandlePagedKvCache;

    fn allocate(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
    ) -> Result<Self::Cache, KvBackendError> {
        let binding = KvPageBinding::legacy(&schema, self.page_tokens);
        self.allocate_bound(schema, capacity_tokens, binding)
    }

    fn append(
        &self,
        cache: &mut Self::Cache,
        keys: &[TensorRef],
        values: &[TensorRef],
    ) -> Result<(), KvBackendError> {
        let token_count = Self::validate_append(cache, keys, values)?;
        cache
            .device_lease
            .begin_tensor()
            .map_err(Self::lease_error)?;
        let result = (|| {
            let page_tokens = cache.binding.page_tokens.get();
            let width = cache.schema.key_value_heads * cache.schema.head_dim;
            let originally_shared = cache
                .pages
                .iter()
                .map(|page| Arc::strong_count(page) > 1)
                .collect::<Vec<_>>();
            let mut next_pages = cache.pages.clone();
            let mut next_cow_copies = cache.cow_copies;
            let mut consumed = 0usize;
            while consumed < token_count {
                let needs_page = next_pages
                    .last()
                    .is_none_or(|page| usize::from(page.valid_tokens) == page_tokens);
                if needs_page {
                    let count = (token_count - consumed).min(page_tokens);
                    let mut page = self.new_page(cache, keys, values, consumed, count)?;
                    if count == page_tokens {
                        Self::seal_page(&cache.binding, next_pages.len(), &mut page)?;
                    }
                    next_pages.push(Arc::new(page));
                    consumed += count;
                    continue;
                }

                let ordinal = next_pages.len() - 1;
                let old = &next_pages[ordinal];
                let mut candidate = (**old).clone();
                if originally_shared.get(ordinal).copied().unwrap_or(false)
                    || old.state == KvPageState::Sealed
                {
                    candidate.page_id = self.next_page_id()?;
                    next_cow_copies = next_cow_copies.saturating_add(1);
                }
                candidate.state = KvPageState::Writable;
                candidate.digest = None;
                let available = page_tokens - usize::from(candidate.valid_tokens);
                let count = (token_count - consumed).min(available);
                for layer in 0..cache.schema.layer_count {
                    let range = consumed * width..(consumed + count) * width;
                    let key = Self::tensor_from_values(
                        keys[layer].values[range.clone()].to_vec(),
                        count,
                        width,
                        &cache.device,
                    )?;
                    let value = Self::tensor_from_values(
                        values[layer].values[range].to_vec(),
                        count,
                        width,
                        &cache.device,
                    )?;
                    candidate.layers[layer].0 =
                        Tensor::cat(&[candidate.layers[layer].0.clone(), key], 0)
                            .map_err(Self::candle_error)?;
                    candidate.layers[layer].1 =
                        Tensor::cat(&[candidate.layers[layer].1.clone(), value], 0)
                            .map_err(Self::candle_error)?;
                }
                candidate.valid_tokens = candidate
                    .valid_tokens
                    .checked_add(u16::try_from(count).map_err(|_| KvBackendError::OutOfMemory)?)
                    .ok_or(KvBackendError::OutOfMemory)?;
                if usize::from(candidate.valid_tokens) == page_tokens {
                    Self::seal_page(&cache.binding, ordinal, &mut candidate)?;
                }
                next_pages[ordinal] = Arc::new(candidate);
                consumed += count;
            }
            cache.pages = next_pages;
            cache.cow_copies = next_cow_copies;
            cache.sequence_length += token_count;
            Ok(())
        })();
        let finish = cache.device_lease.end_tensor().map_err(Self::lease_error);
        match (result, finish) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn truncate(&self, cache: &mut Self::Cache, new_length: usize) -> Result<(), KvBackendError> {
        if cache.revoked {
            return Err(KvBackendError::StaleLease);
        }
        if new_length > cache.sequence_length {
            return Err(KvBackendError::InvalidTruncation);
        }
        cache
            .device_lease
            .begin_tensor()
            .map_err(Self::lease_error)?;
        let result = (|| {
            let page_tokens = cache.binding.page_tokens.get();
            cache.pages.truncate(new_length.div_ceil(page_tokens));
            let keep = new_length % page_tokens;
            if keep != 0 {
                let ordinal = cache.pages.len() - 1;
                if keep < usize::from(cache.pages[ordinal].valid_tokens) {
                    let mut page = (*cache.pages[ordinal]).clone();
                    page.page_id = self.next_page_id()?;
                    page.state = KvPageState::Writable;
                    page.digest = None;
                    for (key, value) in &mut page.layers {
                        *key = key.narrow(0, 0, keep).map_err(Self::candle_error)?;
                        *value = value.narrow(0, 0, keep).map_err(Self::candle_error)?;
                    }
                    page.valid_tokens =
                        u16::try_from(keep).map_err(|_| KvBackendError::OutOfMemory)?;
                    cache.pages[ordinal] = Arc::new(page);
                    cache.cow_copies = cache.cow_copies.saturating_add(1);
                }
            }
            cache.sequence_length = new_length;
            Ok(())
        })();
        let finish = cache.device_lease.end_tensor().map_err(Self::lease_error);
        match (result, finish) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    fn snapshot(&self, cache: &Self::Cache) -> Result<KvTensorSnapshot, KvBackendError> {
        let mut cloned = CandlePagedKvCache {
            schema: cache.schema.clone(),
            binding: cache.binding.clone(),
            capacity_tokens: cache.capacity_tokens,
            sequence_length: cache.sequence_length,
            position_offset: cache.position_offset,
            pages: cache.pages.clone(),
            device: cache.device.clone(),
            device_lease: cache.device_lease.clone(),
            runtime_lease: cache.runtime_lease,
            revoked: cache.revoked,
            cow_copies: cache.cow_copies,
        };
        let lease = self.seal_snapshot(&mut cloned)?;
        self.snapshot_from_lease(&lease)?.to_dense(KvCacheTier::Gpu)
    }

    fn restore(
        &self,
        snapshot: KvTensorSnapshot,
        device: &ptr_types::DeviceId,
    ) -> Result<Self::Cache, KvBackendError> {
        if device != &snapshot.schema.device
            || snapshot.schema.device.0 != format!("cuda:{}", self.device_id)
            || snapshot.layout.tier != KvCacheTier::Gpu
            || !snapshot.verify_digest()
        {
            return Err(KvBackendError::DeviceMismatch);
        }
        if snapshot.layout.page_tokens != self.page_tokens.get() {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }
        let mut cache = self.allocate(snapshot.schema.clone(), snapshot.capacity_tokens)?;
        if snapshot.sequence_length > 0 {
            let keys = snapshot
                .layers
                .iter()
                .map(|layer| layer.keys.clone())
                .collect::<Vec<_>>();
            let values = snapshot
                .layers
                .iter()
                .map(|layer| layer.values.clone())
                .collect::<Vec<_>>();
            self.append(&mut cache, &keys, &values)?;
        }
        cache.position_offset = snapshot.position_offset;
        Ok(cache)
    }

    fn release(&self, mut cache: Self::Cache) -> Result<(), KvBackendError> {
        cache.device.synchronize().map_err(Self::candle_error)?;
        cache.pages.clear();
        cache.device_lease.release().map_err(Self::lease_error)
    }

    fn revoke(&self, cache: &mut Self::Cache) -> Result<(), KvBackendError> {
        self.revoke_runtime_lease(cache)
    }
}

impl PagedKvTensorBackend for CandlePagedKvBackend {
    type SnapshotLease = CandleKvSnapshotLease;

    fn allocate_bound(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
        binding: KvPageBinding,
    ) -> Result<Self::Cache, KvBackendError> {
        InMemoryKvTensorBackend::validate_schema(&schema)?;
        binding.validate(&schema)?;
        if capacity_tokens == 0
            || binding.page_tokens != self.page_tokens
            || schema.device.0 != format!("cuda:{}", self.device_id)
        {
            return Err(KvBackendError::DeviceMismatch);
        }
        let device = self.new_device()?;
        let stream_id = self.next_stream_id()?;
        Ok(CandlePagedKvCache {
            schema: schema.clone(),
            binding,
            capacity_tokens,
            sequence_length: 0,
            position_offset: 0,
            pages: Vec::new(),
            device,
            device_lease: DeviceLease::new(schema.device, stream_id, 0, 0),
            runtime_lease: None,
            revoked: false,
            cow_copies: 0,
        })
    }

    fn bind_runtime_lease(
        &self,
        cache: &mut Self::Cache,
        binding: KvRuntimeLeaseBinding,
    ) -> Result<(), KvBackendError> {
        binding.validate()?;
        if cache.revoked
            || cache
                .runtime_lease
                .is_some_and(|current| current != binding)
        {
            return Err(KvBackendError::StaleLease);
        }
        cache.runtime_lease = Some(binding);
        cache.device_lease.placement_epoch = binding.placement_epoch;
        cache.device_lease.fencing_token = binding.fencing_token;
        Ok(())
    }

    fn revoke_runtime_lease(&self, cache: &mut Self::Cache) -> Result<(), KvBackendError> {
        cache.revoked = true;
        cache.runtime_lease = None;
        cache.device_lease.revoke();
        cache.device.synchronize().map_err(Self::candle_error)
    }

    fn seal_snapshot(
        &self,
        cache: &mut Self::Cache,
    ) -> Result<Self::SnapshotLease, KvBackendError> {
        if cache.revoked {
            return Err(KvBackendError::StaleLease);
        }
        cache.device.synchronize().map_err(Self::candle_error)?;
        for (ordinal, page) in cache.pages.iter_mut().enumerate() {
            if page.state != KvPageState::Sealed {
                if Arc::strong_count(page) > 1 {
                    let mut copied = (**page).clone();
                    copied.page_id = self.next_page_id()?;
                    *page = Arc::new(copied);
                    cache.cow_copies = cache.cow_copies.saturating_add(1);
                }
                let page = Arc::get_mut(page).ok_or(KvBackendError::InvalidPageState)?;
                Self::seal_page(&cache.binding, ordinal, page)?;
            }
        }
        let page_digests = cache
            .pages
            .iter()
            .map(|page| page.digest.ok_or(KvBackendError::InvalidPageState))
            .collect::<Result<Vec<_>, _>>()?;
        let digest = crate::paged_kv::paged_snapshot_digest_parts(
            &cache.schema,
            &cache.binding,
            cache.capacity_tokens,
            cache.sequence_length,
            cache.position_offset,
            page_digests,
        );
        Ok(CandleKvSnapshotLease {
            schema: cache.schema.clone(),
            binding: cache.binding.clone(),
            capacity_tokens: cache.capacity_tokens,
            sequence_length: cache.sequence_length,
            position_offset: cache.position_offset,
            digest,
            pages: cache.pages.clone(),
            _device: cache.device.clone(),
        })
    }

    fn snapshot_from_lease(
        &self,
        lease: &Self::SnapshotLease,
    ) -> Result<KvPagedSnapshot, KvBackendError> {
        let pages = lease
            .pages
            .iter()
            .enumerate()
            .map(|(ordinal, page)| {
                if page.state != KvPageState::Sealed {
                    return Err(KvBackendError::InvalidPageState);
                }
                let snapshot = Self::page_snapshot(&lease.binding, ordinal, page)?;
                if page.digest != Some(snapshot.digest) {
                    return Err(KvBackendError::SnapshotDigestMismatch);
                }
                Ok(snapshot)
            })
            .collect::<Result<Vec<_>, KvBackendError>>()?;
        let snapshot = KvPagedSnapshot {
            schema: lease.schema.clone(),
            binding: lease.binding.clone(),
            capacity_tokens: lease.capacity_tokens,
            sequence_length: lease.sequence_length,
            position_offset: lease.position_offset,
            pages,
            digest: lease.digest,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    fn restore_paged(
        &self,
        snapshot: KvPagedSnapshot,
        binding: KvPageBinding,
    ) -> Result<Self::Cache, KvBackendError> {
        if !snapshot.verify_digest()
            || snapshot.binding.principal != binding.principal
            || snapshot.binding.execution_manifest != binding.execution_manifest
            || snapshot.binding.model != binding.model
            || snapshot.binding.adapter != binding.adapter
            || snapshot.binding.generation != binding.generation
            || snapshot.binding.dtype != binding.dtype
            || snapshot.binding.page_tokens != binding.page_tokens
        {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }
        let dense = snapshot.to_dense(KvCacheTier::Gpu)?;
        let mut schema = dense.schema.clone();
        schema.device = binding.device.clone();
        let mut cache = self.allocate_bound(schema, dense.capacity_tokens, binding)?;
        if dense.sequence_length > 0 {
            let keys = dense
                .layers
                .iter()
                .map(|layer| layer.keys.clone())
                .collect::<Vec<_>>();
            let values = dense
                .layers
                .iter()
                .map(|layer| layer.values.clone())
                .collect::<Vec<_>>();
            self.append(&mut cache, &keys, &values)?;
        }
        cache.position_offset = dense.position_offset;
        Ok(cache)
    }

    fn publish_prefix(&self, cache: &mut Self::Cache) -> Result<KvSharedPrefix, KvBackendError> {
        let _ = self.seal_snapshot(cache)?;
        let pages = cache
            .pages
            .iter()
            .take_while(|page| usize::from(page.valid_tokens) == cache.binding.page_tokens.get())
            .cloned()
            .collect::<Vec<_>>();
        if pages.is_empty() {
            return Err(KvBackendError::PrefixMismatch);
        }
        let page_digests = pages
            .iter()
            .map(|page| page.digest.ok_or(KvBackendError::InvalidPageState))
            .collect::<Result<Vec<_>, _>>()?;
        let key = KvPrefixKey {
            principal: cache.binding.principal.clone(),
            execution_manifest: cache.binding.execution_manifest,
            model: cache.binding.model.clone(),
            adapter: cache.binding.adapter.clone(),
            generation: cache.binding.generation,
            schema_digest: crate::paged_kv::schema_digest(&cache.schema, cache.binding.page_tokens),
            prefix_digest: crate::paged_kv::digest_list(b"ptr-kv-prefix-v1\0", &page_digests),
        };
        let sequence_length = pages.len() * cache.binding.page_tokens.get();
        self.state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("CUDA prefix catalog poisoned".into()))?
            .insert(
                key.clone(),
                CandlePrefixRecord {
                    pages,
                    sequence_length,
                    position_offset: cache.position_offset,
                },
            );
        Ok(KvSharedPrefix {
            key,
            sequence_length,
            position_offset: cache.position_offset,
            page_digests,
        })
    }

    fn fork_prefix(
        &self,
        cache: &mut Self::Cache,
        prefix: &KvSharedPrefix,
    ) -> Result<(), KvBackendError> {
        if cache.revoked || cache.sequence_length != 0 || !cache.pages.is_empty() {
            return Err(KvBackendError::PrefixMismatch);
        }
        let expected = KvPrefixKey {
            principal: cache.binding.principal.clone(),
            execution_manifest: cache.binding.execution_manifest,
            model: cache.binding.model.clone(),
            adapter: cache.binding.adapter.clone(),
            generation: cache.binding.generation,
            schema_digest: crate::paged_kv::schema_digest(&cache.schema, cache.binding.page_tokens),
            prefix_digest: prefix.key.prefix_digest,
        };
        if expected != prefix.key || prefix.sequence_length > cache.capacity_tokens {
            return Err(KvBackendError::PrefixMismatch);
        }
        let record = self
            .state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("CUDA prefix catalog poisoned".into()))?
            .get(&prefix.key)
            .cloned()
            .ok_or(KvBackendError::PrefixMismatch)?;
        cache.pages = record.pages;
        cache.sequence_length = record.sequence_length;
        cache.position_offset = record.position_offset;
        Ok(())
    }

    fn synchronize(&self, cache: &Self::Cache) -> Result<(), KvBackendError> {
        if cache.revoked {
            return Err(KvBackendError::StaleLease);
        }
        cache.device.synchronize().map_err(Self::candle_error)
    }

    fn page_metrics(&self, cache: &Self::Cache) -> KvPageMetrics {
        let width = cache.schema.key_value_heads * cache.schema.head_dim;
        let values = cache
            .pages
            .iter()
            .map(|page| usize::from(page.valid_tokens) * width * cache.schema.layer_count * 2)
            .sum::<usize>();
        KvPageMetrics {
            allocated_pages: cache.pages.len(),
            writable_pages: cache
                .pages
                .iter()
                .filter(|page| page.state == KvPageState::Writable)
                .count(),
            sealed_pages: cache
                .pages
                .iter()
                .filter(|page| page.state == KvPageState::Sealed)
                .count(),
            shared_pages: cache
                .pages
                .iter()
                .filter(|page| Arc::strong_count(page) > 1)
                .count(),
            cow_copies: cache.cow_copies,
            resident_bytes: u64::try_from(values.saturating_mul(4)).unwrap_or(u64::MAX),
        }
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
