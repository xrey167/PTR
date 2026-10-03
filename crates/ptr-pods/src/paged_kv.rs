use crate::{
    InMemoryKvTensorBackend, KvBackendError, KvCacheLayout, KvCacheTier, KvLayerSnapshot,
    KvPageTable, KvTensorBackend, KvTensorDType, KvTensorSchema, KvTensorSnapshot, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, Digest, Generation, ModelVersion, PrincipalId};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvPageSize(u16);

impl KvPageSize {
    pub const DEFAULT: Self = Self(16);

    pub fn new(tokens: u16) -> Result<Self, KvBackendError> {
        if !(1..=256).contains(&tokens) || !tokens.is_power_of_two() {
            return Err(KvBackendError::InvalidSchema(
                "KV page size must be a power of two in 1..=256".into(),
            ));
        }
        Ok(Self(tokens))
    }

    pub fn get(self) -> usize {
        usize::from(self.0)
    }
}

impl Default for KvPageSize {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvPageId(pub u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvPageState {
    Writable,
    Sealed,
    Transferring,
    Revoked,
    Corrupt,
    Released,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvPageBinding {
    pub model: ModelVersion,
    pub adapter: AdapterVersion,
    pub generation: Generation,
    pub execution_manifest: Digest,
    pub principal: PrincipalId,
    pub device: DeviceId,
    pub dtype: KvTensorDType,
    pub page_tokens: KvPageSize,
}

impl KvPageBinding {
    pub fn validate(&self, schema: &KvTensorSchema) -> Result<(), KvBackendError> {
        if self.generation.0 == 0
            || self.execution_manifest == [0; 32]
            || self.principal.0.trim().is_empty()
            || self.model != schema.model
            || self.adapter != schema.adapter
            || self.device != schema.device
            || self.dtype != schema.dtype
            || schema.batch_size != 1
        {
            return Err(KvBackendError::InvalidSchema(
                "KV page binding does not match tensor schema".into(),
            ));
        }
        KvPageSize::new(self.page_tokens.0).map(|_| ())
    }

    pub(crate) fn legacy(schema: &KvTensorSchema, page_tokens: KvPageSize) -> Self {
        let mut bytes = b"ptr-legacy-paged-kv-v1\0".to_vec();
        put_text(&mut bytes, &schema.model.0);
        put_text(&mut bytes, &schema.adapter.0);
        put_text(&mut bytes, &schema.device.0);
        Self {
            model: schema.model.clone(),
            adapter: schema.adapter.clone(),
            generation: Generation(1),
            execution_manifest: Sha256::digest(bytes).into(),
            principal: PrincipalId::from("ptr:legacy-local"),
            device: schema.device.clone(),
            dtype: schema.dtype,
            page_tokens,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPageEntry {
    pub page_id: KvPageId,
    pub valid_tokens: u16,
    pub digest: Option<Digest>,
    pub state: KvPageState,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KvRuntimeLeaseBinding {
    pub placement_epoch: u64,
    pub fencing_token: u128,
}

impl KvRuntimeLeaseBinding {
    pub fn validate(self) -> Result<(), KvBackendError> {
        if self.placement_epoch == 0 || self.fencing_token == 0 {
            return Err(KvBackendError::StaleLease);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KvPrefixKey {
    pub principal: PrincipalId,
    pub execution_manifest: Digest,
    pub model: ModelVersion,
    pub adapter: AdapterVersion,
    pub generation: Generation,
    pub schema_digest: Digest,
    pub prefix_digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvSharedPrefix {
    pub key: KvPrefixKey,
    pub sequence_length: usize,
    pub position_offset: usize,
    pub page_digests: Vec<Digest>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvPageSnapshot {
    pub ordinal: usize,
    pub valid_tokens: u16,
    pub layers: Vec<KvLayerSnapshot>,
    pub digest: Digest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvPagedSnapshot {
    pub schema: KvTensorSchema,
    pub binding: KvPageBinding,
    pub capacity_tokens: usize,
    pub sequence_length: usize,
    pub position_offset: usize,
    pub pages: Vec<KvPageSnapshot>,
    pub digest: Digest,
}

impl KvPagedSnapshot {
    pub fn verify_digest(&self) -> bool {
        self.validate().is_ok()
    }

    pub fn validate(&self) -> Result<(), KvBackendError> {
        self.binding.validate(&self.schema)?;
        if self.capacity_tokens == 0
            || self.sequence_length > self.capacity_tokens
            || self.pages.len()
                > self
                    .capacity_tokens
                    .div_ceil(self.binding.page_tokens.get())
        {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }
        let width = self.schema.key_value_heads * self.schema.head_dim;
        let mut total_tokens = 0usize;
        for (expected_ordinal, page) in self.pages.iter().enumerate() {
            let tokens = usize::from(page.valid_tokens);
            if page.ordinal != expected_ordinal
                || tokens == 0
                || tokens > self.binding.page_tokens.get()
                || (expected_ordinal + 1 != self.pages.len()
                    && tokens != self.binding.page_tokens.get())
                || page.layers.len() != self.schema.layer_count
            {
                return Err(KvBackendError::SnapshotSchemaMismatch);
            }
            for (layer_index, layer) in page.layers.iter().enumerate() {
                for tensor in [&layer.keys, &layer.values] {
                    if tensor.layer != layer_index
                        || tensor.shape != [tokens, width]
                        || tensor.values.len() != tokens * width
                    {
                        return Err(KvBackendError::SnapshotSchemaMismatch);
                    }
                }
            }
            if page.digest
                != page_digest(&self.binding, page.ordinal, page.valid_tokens, &page.layers)
            {
                return Err(KvBackendError::SnapshotDigestMismatch);
            }
            total_tokens = total_tokens
                .checked_add(tokens)
                .ok_or(KvBackendError::OutOfMemory)?;
        }
        if total_tokens != self.sequence_length || self.digest != paged_snapshot_digest(self) {
            return Err(KvBackendError::SnapshotDigestMismatch);
        }
        Ok(())
    }

    pub fn to_dense(&self, tier: KvCacheTier) -> Result<KvTensorSnapshot, KvBackendError> {
        if !self.verify_digest() {
            return Err(KvBackendError::SnapshotDigestMismatch);
        }
        let mut layers = empty_layers(&self.schema);
        for page in &self.pages {
            append_layer_data(&mut layers, &page.layers, usize::from(page.valid_tokens))?;
        }
        let page_count = self
            .capacity_tokens
            .div_ceil(self.binding.page_tokens.get());
        let mut snapshot = KvTensorSnapshot {
            schema: self.schema.clone(),
            layout: KvCacheLayout {
                page_tokens: self.binding.page_tokens.get(),
                tier,
                dtype: self.binding.dtype,
            },
            page_table: KvPageTable {
                page_tokens: self.binding.page_tokens.get(),
                capacity_tokens: self.capacity_tokens,
                logical_to_physical: (0..self.pages.len()).collect(),
                free_physical: (self.pages.len()..page_count).rev().collect(),
            },
            capacity_tokens: self.capacity_tokens,
            sequence_length: self.sequence_length,
            position_offset: self.position_offset,
            layers,
            digest: [0; 32],
        };
        snapshot.digest = InMemoryKvTensorBackend::digest(&snapshot);
        Ok(snapshot)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KvPageMetrics {
    pub allocated_pages: usize,
    pub writable_pages: usize,
    pub sealed_pages: usize,
    pub shared_pages: usize,
    pub cow_copies: u64,
    pub resident_bytes: u64,
}

pub trait PagedKvTensorBackend: KvTensorBackend {
    type SnapshotLease: Send;

    fn allocate_bound(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
        binding: KvPageBinding,
    ) -> Result<Self::Cache, KvBackendError>;

    fn bind_runtime_lease(
        &self,
        cache: &mut Self::Cache,
        binding: KvRuntimeLeaseBinding,
    ) -> Result<(), KvBackendError>;

    fn revoke_runtime_lease(&self, cache: &mut Self::Cache) -> Result<(), KvBackendError>;

    fn seal_snapshot(&self, cache: &mut Self::Cache)
        -> Result<Self::SnapshotLease, KvBackendError>;

    fn snapshot_from_lease(
        &self,
        lease: &Self::SnapshotLease,
    ) -> Result<KvPagedSnapshot, KvBackendError>;

    fn restore_paged(
        &self,
        snapshot: KvPagedSnapshot,
        binding: KvPageBinding,
    ) -> Result<Self::Cache, KvBackendError>;

    fn publish_prefix(&self, cache: &mut Self::Cache) -> Result<KvSharedPrefix, KvBackendError>;

    fn fork_prefix(
        &self,
        cache: &mut Self::Cache,
        prefix: &KvSharedPrefix,
    ) -> Result<(), KvBackendError>;

    fn synchronize(&self, cache: &Self::Cache) -> Result<(), KvBackendError>;

    fn page_metrics(&self, cache: &Self::Cache) -> KvPageMetrics;
}

#[derive(Clone, Debug)]
struct HostPageBundle {
    page_id: KvPageId,
    valid_tokens: u16,
    state: KvPageState,
    digest: Option<Digest>,
    layers: Vec<KvLayerSnapshot>,
}

impl HostPageBundle {
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
struct PrefixRecord {
    pages: Vec<Arc<HostPageBundle>>,
    sequence_length: usize,
    position_offset: usize,
}

#[derive(Default)]
struct PagedBackendState {
    next_page_id: AtomicU64,
    prefixes: Mutex<BTreeMap<KvPrefixKey, PrefixRecord>>,
}

#[derive(Clone)]
pub struct InMemoryPagedKvBackend {
    page_tokens: KvPageSize,
    state: Arc<PagedBackendState>,
}

impl Default for InMemoryPagedKvBackend {
    fn default() -> Self {
        Self::new(KvPageSize::DEFAULT)
    }
}

impl InMemoryPagedKvBackend {
    pub fn new(page_tokens: KvPageSize) -> Self {
        Self {
            page_tokens,
            state: Arc::new(PagedBackendState::default()),
        }
    }

    fn next_page_id(&self) -> Result<KvPageId, KvBackendError> {
        let id = self.state.next_page_id.fetch_add(1, Ordering::Relaxed);
        if id == u64::MAX {
            return Err(KvBackendError::OutOfMemory);
        }
        Ok(KvPageId(id + 1))
    }

    fn new_page(
        &self,
        schema: &KvTensorSchema,
        keys: &[TensorRef],
        values: &[TensorRef],
        start: usize,
        count: usize,
    ) -> Result<HostPageBundle, KvBackendError> {
        let width = schema.key_value_heads * schema.head_dim;
        let mut layers = Vec::with_capacity(schema.layer_count);
        for layer in 0..schema.layer_count {
            let range = start * width..(start + count) * width;
            layers.push(KvLayerSnapshot {
                keys: TensorRef {
                    layer,
                    shape: vec![count, width],
                    values: keys[layer].values[range.clone()].to_vec(),
                },
                values: TensorRef {
                    layer,
                    shape: vec![count, width],
                    values: values[layer].values[range].to_vec(),
                },
            });
        }
        Ok(HostPageBundle {
            page_id: self.next_page_id()?,
            valid_tokens: u16::try_from(count).map_err(|_| KvBackendError::OutOfMemory)?,
            state: KvPageState::Writable,
            digest: None,
            layers,
        })
    }

    fn seal_page(
        page: &mut HostPageBundle,
        binding: &KvPageBinding,
        ordinal: usize,
    ) -> Result<(), KvBackendError> {
        if matches!(
            page.state,
            KvPageState::Revoked | KvPageState::Corrupt | KvPageState::Released
        ) {
            return Err(KvBackendError::InvalidPageState);
        }
        let digest = page_digest(binding, ordinal, page.valid_tokens, &page.layers);
        page.digest = Some(digest);
        page.state = KvPageState::Sealed;
        Ok(())
    }

    fn validate_append(
        cache: &InMemoryPagedKvCache,
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
        let token_count = keys
            .first()
            .map_or(0, |tensor| tensor.shape.first().copied().unwrap_or(0));
        if token_count == 0
            || cache.sequence_length.saturating_add(token_count) > cache.capacity_tokens
        {
            return Err(KvBackendError::OutOfMemory);
        }
        for layer in 0..cache.schema.layer_count {
            for tensor in [&keys[layer], &values[layer]] {
                if tensor.layer != layer
                    || tensor.shape != [token_count, width]
                    || tensor.values.len() != token_count * width
                {
                    return Err(KvBackendError::ShapeMismatch);
                }
            }
        }
        Ok(token_count)
    }

    fn paged_snapshot(cache: &InMemoryPagedKvCache) -> Result<KvPagedSnapshot, KvBackendError> {
        let mut pages = Vec::with_capacity(cache.pages.len());
        for (ordinal, page) in cache.pages.iter().enumerate() {
            if page.state != KvPageState::Sealed {
                return Err(KvBackendError::InvalidPageState);
            }
            let digest = page.digest.ok_or(KvBackendError::InvalidPageState)?;
            pages.push(KvPageSnapshot {
                ordinal,
                valid_tokens: page.valid_tokens,
                layers: page.layers.clone(),
                digest,
            });
        }
        let mut snapshot = KvPagedSnapshot {
            schema: cache.schema.clone(),
            binding: cache.binding.clone(),
            capacity_tokens: cache.capacity_tokens,
            sequence_length: cache.sequence_length,
            position_offset: cache.position_offset,
            pages,
            digest: [0; 32],
        };
        snapshot.digest = paged_snapshot_digest(&snapshot);
        Ok(snapshot)
    }
}

#[derive(Clone, Debug)]
pub struct InMemoryPagedKvCache {
    pub schema: KvTensorSchema,
    pub binding: KvPageBinding,
    pub capacity_tokens: usize,
    pub sequence_length: usize,
    pub position_offset: usize,
    pages: Vec<Arc<HostPageBundle>>,
    runtime_lease: Option<KvRuntimeLeaseBinding>,
    revoked: bool,
    cow_copies: u64,
}

impl InMemoryPagedKvCache {
    pub fn page_table(&self) -> Vec<KvPageEntry> {
        self.pages.iter().map(|page| page.entry()).collect()
    }
}

#[derive(Clone, Debug)]
pub struct InMemoryKvSnapshotLease {
    snapshot: KvPagedSnapshot,
    _pages: Vec<Arc<HostPageBundle>>,
}

impl KvTensorBackend for InMemoryPagedKvBackend {
    type Cache = InMemoryPagedKvCache;

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
        let page_tokens = cache.binding.page_tokens.get();
        let width = cache.schema.key_value_heads * cache.schema.head_dim;
        let mut consumed = 0usize;
        while consumed < token_count {
            let needs_page = cache
                .pages
                .last()
                .is_none_or(|page| usize::from(page.valid_tokens) == page_tokens);
            if needs_page {
                let count = (token_count - consumed).min(page_tokens);
                cache.pages.push(Arc::new(self.new_page(
                    &cache.schema,
                    keys,
                    values,
                    consumed,
                    count,
                )?));
                consumed += count;
            } else {
                let ordinal = cache.pages.len() - 1;
                let page = &cache.pages[ordinal];
                let was_shared_or_sealed =
                    Arc::strong_count(page) > 1 || page.state == KvPageState::Sealed;
                if was_shared_or_sealed {
                    let mut copied = (**page).clone();
                    copied.page_id = self.next_page_id()?;
                    copied.state = KvPageState::Writable;
                    copied.digest = None;
                    cache.pages[ordinal] = Arc::new(copied);
                    cache.cow_copies = cache.cow_copies.saturating_add(1);
                }
                let page = Arc::get_mut(&mut cache.pages[ordinal])
                    .ok_or(KvBackendError::InvalidPageState)?;
                let available = page_tokens - usize::from(page.valid_tokens);
                let count = (token_count - consumed).min(available);
                for layer in 0..cache.schema.layer_count {
                    let range = consumed * width..(consumed + count) * width;
                    page.layers[layer]
                        .keys
                        .values
                        .extend_from_slice(&keys[layer].values[range.clone()]);
                    page.layers[layer]
                        .values
                        .values
                        .extend_from_slice(&values[layer].values[range]);
                    page.layers[layer].keys.shape[0] += count;
                    page.layers[layer].values.shape[0] += count;
                }
                page.valid_tokens = page
                    .valid_tokens
                    .checked_add(u16::try_from(count).map_err(|_| KvBackendError::OutOfMemory)?)
                    .ok_or(KvBackendError::OutOfMemory)?;
                consumed += count;
            }
            let ordinal = cache.pages.len() - 1;
            if usize::from(cache.pages[ordinal].valid_tokens) == page_tokens {
                let page = Arc::get_mut(&mut cache.pages[ordinal])
                    .ok_or(KvBackendError::InvalidPageState)?;
                Self::seal_page(page, &cache.binding, ordinal)?;
            }
        }
        cache.sequence_length += token_count;
        Ok(())
    }

    fn truncate(&self, cache: &mut Self::Cache, new_length: usize) -> Result<(), KvBackendError> {
        if cache.revoked {
            return Err(KvBackendError::StaleLease);
        }
        if new_length > cache.sequence_length {
            return Err(KvBackendError::InvalidTruncation);
        }
        let page_tokens = cache.binding.page_tokens.get();
        let required_pages = new_length.div_ceil(page_tokens);
        cache.pages.truncate(required_pages);
        if let Some(last) = cache.pages.last_mut() {
            let keep = new_length % page_tokens;
            if keep != 0 && keep < usize::from(last.valid_tokens) {
                let mut page = (**last).clone();
                page.page_id = self.next_page_id()?;
                page.state = KvPageState::Writable;
                page.digest = None;
                let width = cache.schema.key_value_heads * cache.schema.head_dim;
                for layer in &mut page.layers {
                    layer.keys.values.truncate(keep * width);
                    layer.values.values.truncate(keep * width);
                    layer.keys.shape[0] = keep;
                    layer.values.shape[0] = keep;
                }
                page.valid_tokens = u16::try_from(keep).map_err(|_| KvBackendError::OutOfMemory)?;
                *last = Arc::new(page);
                cache.cow_copies = cache.cow_copies.saturating_add(1);
            }
        }
        cache.sequence_length = new_length;
        Ok(())
    }

    fn snapshot(&self, cache: &Self::Cache) -> Result<KvTensorSnapshot, KvBackendError> {
        let mut cloned = cache.clone();
        self.seal_snapshot(&mut cloned)?
            .snapshot
            .to_dense(KvCacheTier::Cpu)
    }

    fn restore(
        &self,
        snapshot: KvTensorSnapshot,
        device: &DeviceId,
    ) -> Result<Self::Cache, KvBackendError> {
        if &snapshot.schema.device != device {
            return Err(KvBackendError::DeviceMismatch);
        }
        if !snapshot.verify_digest() {
            return Err(KvBackendError::SnapshotDigestMismatch);
        }
        let page_tokens = KvPageSize::new(
            u16::try_from(snapshot.layout.page_tokens)
                .map_err(|_| KvBackendError::SnapshotSchemaMismatch)?,
        )?;
        if page_tokens != self.page_tokens {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }
        let binding = KvPageBinding::legacy(&snapshot.schema, page_tokens);
        let mut cache =
            self.allocate_bound(snapshot.schema.clone(), snapshot.capacity_tokens, binding)?;
        if snapshot.sequence_length > 0 {
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
        }
        cache.position_offset = snapshot.position_offset;
        Ok(cache)
    }

    fn release(&self, mut cache: Self::Cache) -> Result<(), KvBackendError> {
        cache.revoked = true;
        cache.pages.clear();
        Ok(())
    }

    fn revoke(&self, cache: &mut Self::Cache) -> Result<(), KvBackendError> {
        self.revoke_runtime_lease(cache)
    }
}

impl PagedKvTensorBackend for InMemoryPagedKvBackend {
    type SnapshotLease = InMemoryKvSnapshotLease;

    fn allocate_bound(
        &self,
        schema: KvTensorSchema,
        capacity_tokens: usize,
        binding: KvPageBinding,
    ) -> Result<Self::Cache, KvBackendError> {
        InMemoryKvTensorBackend::validate_schema(&schema)?;
        binding.validate(&schema)?;
        if binding.page_tokens != self.page_tokens || capacity_tokens == 0 {
            return Err(KvBackendError::InvalidSchema(
                "invalid paged KV allocation".into(),
            ));
        }
        Ok(InMemoryPagedKvCache {
            schema,
            binding,
            capacity_tokens,
            sequence_length: 0,
            position_offset: 0,
            pages: Vec::new(),
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
        Ok(())
    }

    fn revoke_runtime_lease(&self, cache: &mut Self::Cache) -> Result<(), KvBackendError> {
        cache.revoked = true;
        cache.runtime_lease = None;
        Ok(())
    }

    fn seal_snapshot(
        &self,
        cache: &mut Self::Cache,
    ) -> Result<Self::SnapshotLease, KvBackendError> {
        if cache.revoked {
            return Err(KvBackendError::StaleLease);
        }
        for (ordinal, page) in cache.pages.iter_mut().enumerate() {
            if page.state != KvPageState::Sealed {
                if Arc::strong_count(page) > 1 {
                    let mut copied = (**page).clone();
                    copied.page_id = self.next_page_id()?;
                    *page = Arc::new(copied);
                    cache.cow_copies = cache.cow_copies.saturating_add(1);
                }
                let page = Arc::get_mut(page).ok_or(KvBackendError::InvalidPageState)?;
                Self::seal_page(page, &cache.binding, ordinal)?;
            }
        }
        let snapshot = Self::paged_snapshot(cache)?;
        Ok(InMemoryKvSnapshotLease {
            snapshot,
            _pages: cache.pages.clone(),
        })
    }

    fn snapshot_from_lease(
        &self,
        lease: &Self::SnapshotLease,
    ) -> Result<KvPagedSnapshot, KvBackendError> {
        if !lease.snapshot.verify_digest() {
            return Err(KvBackendError::SnapshotDigestMismatch);
        }
        Ok(lease.snapshot.clone())
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
        let dense = snapshot.to_dense(KvCacheTier::Cpu)?;
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
        let full_pages: Vec<_> = cache
            .pages
            .iter()
            .take_while(|page| usize::from(page.valid_tokens) == cache.binding.page_tokens.get())
            .cloned()
            .collect();
        if full_pages.is_empty() {
            return Err(KvBackendError::PrefixMismatch);
        }
        let page_digests = full_pages
            .iter()
            .map(|page| page.digest.ok_or(KvBackendError::InvalidPageState))
            .collect::<Result<Vec<_>, _>>()?;
        let prefix_digest = digest_list(b"ptr-kv-prefix-v1\0", &page_digests);
        let key = KvPrefixKey {
            principal: cache.binding.principal.clone(),
            execution_manifest: cache.binding.execution_manifest,
            model: cache.binding.model.clone(),
            adapter: cache.binding.adapter.clone(),
            generation: cache.binding.generation,
            schema_digest: schema_digest(&cache.schema, cache.binding.page_tokens),
            prefix_digest,
        };
        let sequence_length = full_pages.len() * cache.binding.page_tokens.get();
        self.state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?
            .insert(
                key.clone(),
                PrefixRecord {
                    pages: full_pages,
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
            schema_digest: schema_digest(&cache.schema, cache.binding.page_tokens),
            prefix_digest: prefix.key.prefix_digest,
        };
        if expected != prefix.key || prefix.sequence_length > cache.capacity_tokens {
            return Err(KvBackendError::PrefixMismatch);
        }
        let record = self
            .state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?
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
        Ok(())
    }

    fn page_metrics(&self, cache: &Self::Cache) -> KvPageMetrics {
        let width = cache.schema.key_value_heads * cache.schema.head_dim;
        let resident_values: usize = cache
            .pages
            .iter()
            .map(|page| usize::from(page.valid_tokens) * width * cache.schema.layer_count * 2)
            .sum();
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
            resident_bytes: u64::try_from(resident_values.saturating_mul(4)).unwrap_or(u64::MAX),
        }
    }
}

fn empty_layers(schema: &KvTensorSchema) -> Vec<KvLayerSnapshot> {
    let width = schema.key_value_heads * schema.head_dim;
    (0..schema.layer_count)
        .map(|layer| KvLayerSnapshot {
            keys: TensorRef {
                layer,
                shape: vec![0, width],
                values: Vec::new(),
            },
            values: TensorRef {
                layer,
                shape: vec![0, width],
                values: Vec::new(),
            },
        })
        .collect()
}

fn append_layer_data(
    target: &mut [KvLayerSnapshot],
    source: &[KvLayerSnapshot],
    tokens: usize,
) -> Result<(), KvBackendError> {
    if target.len() != source.len() {
        return Err(KvBackendError::SnapshotSchemaMismatch);
    }
    for (target, source) in target.iter_mut().zip(source) {
        if source.keys.shape.first().copied() != Some(tokens)
            || source.values.shape.first().copied() != Some(tokens)
        {
            return Err(KvBackendError::SnapshotSchemaMismatch);
        }
        target.keys.values.extend_from_slice(&source.keys.values);
        target
            .values
            .values
            .extend_from_slice(&source.values.values);
        target.keys.shape[0] += tokens;
        target.values.shape[0] += tokens;
    }
    Ok(())
}

pub(crate) fn schema_digest(schema: &KvTensorSchema, page_tokens: KvPageSize) -> Digest {
    let mut bytes = b"ptr-paged-kv-schema-v1\0".to_vec();
    put_text(&mut bytes, &schema.model.0);
    put_text(&mut bytes, &schema.adapter.0);
    put_text(&mut bytes, &schema.device.0);
    for value in [
        schema.layer_count,
        schema.attention_heads,
        schema.key_value_heads,
        schema.head_dim,
        schema.batch_size,
        page_tokens.get(),
    ] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    bytes.push(dtype_code(schema.dtype));
    Sha256::digest(bytes).into()
}

pub(crate) fn page_digest(
    binding: &KvPageBinding,
    ordinal: usize,
    valid_tokens: u16,
    layers: &[KvLayerSnapshot],
) -> Digest {
    let mut bytes = b"ptr-paged-kv-page-v1\0".to_vec();
    put_binding(&mut bytes, binding);
    bytes.extend_from_slice(&(ordinal as u64).to_le_bytes());
    bytes.extend_from_slice(&valid_tokens.to_le_bytes());
    for layer in layers {
        for tensor in [&layer.keys, &layer.values] {
            bytes.extend_from_slice(&(tensor.layer as u64).to_le_bytes());
            for dimension in &tensor.shape {
                bytes.extend_from_slice(&(*dimension as u64).to_le_bytes());
            }
            for value in &tensor.values {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    Sha256::digest(bytes).into()
}

pub(crate) fn paged_snapshot_digest(snapshot: &KvPagedSnapshot) -> Digest {
    paged_snapshot_digest_parts(
        &snapshot.schema,
        &snapshot.binding,
        snapshot.capacity_tokens,
        snapshot.sequence_length,
        snapshot.position_offset,
        snapshot.pages.iter().map(|page| page.digest),
    )
}

pub(crate) fn paged_snapshot_digest_parts(
    schema: &KvTensorSchema,
    binding: &KvPageBinding,
    capacity_tokens: usize,
    sequence_length: usize,
    position_offset: usize,
    page_digests: impl IntoIterator<Item = Digest>,
) -> Digest {
    let mut bytes = b"ptr-paged-kv-snapshot-v1\0".to_vec();
    bytes.extend_from_slice(&schema_digest(schema, binding.page_tokens));
    put_binding(&mut bytes, binding);
    bytes.extend_from_slice(&(capacity_tokens as u64).to_le_bytes());
    bytes.extend_from_slice(&(sequence_length as u64).to_le_bytes());
    bytes.extend_from_slice(&(position_offset as u64).to_le_bytes());
    for digest in page_digests {
        bytes.extend_from_slice(&digest);
    }
    Sha256::digest(bytes).into()
}

fn put_binding(bytes: &mut Vec<u8>, binding: &KvPageBinding) {
    put_text(bytes, &binding.model.0);
    put_text(bytes, &binding.adapter.0);
    bytes.extend_from_slice(&binding.generation.0.to_le_bytes());
    bytes.extend_from_slice(&binding.execution_manifest);
    put_text(bytes, &binding.principal.0);
    put_text(bytes, &binding.device.0);
    bytes.push(dtype_code(binding.dtype));
    bytes.extend_from_slice(&(binding.page_tokens.get() as u64).to_le_bytes());
}

fn put_text(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn dtype_code(dtype: KvTensorDType) -> u8 {
    match dtype {
        KvTensorDType::F32 => 0,
        KvTensorDType::Fp8E4M3 => 1,
        KvTensorDType::Fp8E5M2 => 2,
        KvTensorDType::NvFp4 => 3,
        KvTensorDType::Fp4MxBlock16 => 4,
    }
}

pub(crate) fn digest_list(domain: &[u8], digests: &[Digest]) -> Digest {
    let mut bytes = domain.to_vec();
    for digest in digests {
        bytes.extend_from_slice(digest);
    }
    Sha256::digest(bytes).into()
}
