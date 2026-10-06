use crate::{
    InMemoryKvTensorBackend, KvBackendError, KvCacheLayout, KvCacheTier, KvLayerSnapshot,
    KvPageTable, KvTensorBackend, KvTensorDType, KvTensorSchema, KvTensorSnapshot, TensorRef,
};
use ptr_types::{AdapterVersion, DeviceId, Digest, Generation, ModelVersion, PrincipalId};
use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
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

/// The layers of a model a published prefix covers. `Full` is every layer (the only scope
/// before layer-scoped publishing, and the canonical form of a span that covers all layers,
/// so one content has one key); `Span` is the half-open range `start..end`. `Range<u16>`
/// itself is not `Ord`, which a map key needs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvLayerRange {
    Full,
    Span { start: u16, end: u16 },
}

impl KvLayerRange {
    /// A non-empty span `start..end`. It is not checked against a layer count here; the
    /// backend resolves it against the cache's schema.
    pub fn span(start: u16, end: u16) -> Result<Self, KvBackendError> {
        if start >= end {
            return Err(KvBackendError::InvalidSchema(
                "KV layer range must be non-empty".into(),
            ));
        }
        Ok(Self::Span { start, end })
    }

    /// The covered layers as `start..end` for a model with `layer_count` layers.
    pub fn resolve(self, layer_count: usize) -> Result<std::ops::Range<usize>, KvBackendError> {
        match self {
            Self::Full => Ok(0..layer_count),
            Self::Span { start, end } => {
                let (start, end) = (usize::from(start), usize::from(end));
                if start >= end || end > layer_count {
                    return Err(KvBackendError::InvalidLayer(end));
                }
                Ok(start..end)
            }
        }
    }

    /// The canonical scope of `range` in a model with `layer_count` layers.
    fn canonical(
        range: std::ops::Range<usize>,
        layer_count: usize,
    ) -> Result<Self, KvBackendError> {
        if range.start == 0 && range.end == layer_count {
            return Ok(Self::Full);
        }
        let to_u16 =
            |value: usize| u16::try_from(value).map_err(|_| KvBackendError::InvalidLayer(value));
        Self::span(to_u16(range.start)?, to_u16(range.end)?)
    }
}

/// How an adapter is applied across layers: layers below `first_adapted_layer` are computed
/// by `base_adapter` alone (a LoRA that starts at layer f leaves layers `0..f` untouched), so
/// their KV rows are the base model's and may be shared across adapters.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KvAdapterLayering {
    pub base_adapter: AdapterVersion,
    pub first_adapted_layer: u16,
}

/// The adapter that actually determined the KV rows of `range` in a cache bound to
/// `cache_adapter`: the base adapter when the whole range lies below the first adapted layer.
fn effective_adapter(
    cache_adapter: &AdapterVersion,
    layering: Option<&KvAdapterLayering>,
    range: &std::ops::Range<usize>,
) -> AdapterVersion {
    match layering {
        Some(layering) if range.end <= usize::from(layering.first_adapted_layer) => {
            layering.base_adapter.clone()
        }
        _ => cache_adapter.clone(),
    }
}

fn check_layering(
    layering: Option<&KvAdapterLayering>,
    layer_count: usize,
) -> Result<(), KvBackendError> {
    match layering {
        Some(layering) if usize::from(layering.first_adapted_layer) > layer_count => Err(
            KvBackendError::InvalidLayer(usize::from(layering.first_adapted_layer)),
        ),
        _ => Ok(()),
    }
}

/// A published prefix's identity. `adapter` is the EFFECTIVE adapter of `layers`: for a span
/// below a LoRA's first adapted layer it is the base adapter, not the publishing cache's
/// adapter, and `schema_digest` is computed with that adapter. For a `Full` key it is the
/// cache's adapter, as before layer scoping.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KvPrefixKey {
    pub principal: PrincipalId,
    pub execution_manifest: Digest,
    pub model: ModelVersion,
    pub adapter: AdapterVersion,
    pub generation: Generation,
    pub schema_digest: Digest,
    pub prefix_digest: Digest,
    pub layers: KvLayerRange,
}

/// Publish the full pages of `layers` only.
#[derive(Clone, Debug)]
pub struct KvLayerPublishRequest {
    pub layers: KvLayerRange,
    /// How the publishing cache's adapter is layered; `None` means the cache's adapter
    /// determines every layer.
    pub layering: Option<KvAdapterLayering>,
}

/// Fork a cache from several layer-scoped prefixes that together cover every layer.
///
/// Layering is **caller-trusted**: the backend checks each part's witness, key and adapter
/// layering against `layering`, but not that a part was the right entry for its layer in the
/// caller's own catalog. The caller (the runtime) must choose each part by that layer's own
/// chain; a part chosen by any other rule is not detected here.
#[derive(Debug)]
pub struct KvLayerForkRequest {
    /// One published prefix with its single-use witness per layer span.
    pub parts: Vec<(KvSharedPrefix, KvForkWitness)>,
    /// How the receiving cache's adapter is layered (the adapter check compares each part's
    /// effective adapter for its span).
    pub layering: Option<KvAdapterLayering>,
}

/// Checked adoption of a layer-scoped prefix into a cache that already computed its rows.
///
/// Layering is **caller-trusted** as for [`KvLayerForkRequest`]: adoption compares every row
/// bitwise, which catches a wrong prefix as a mismatch, but the choice of which layer's entry
/// to offer is the caller's and must follow that layer's own chain.
#[derive(Debug)]
pub struct KvLayerAdoptRequest {
    pub prefix: KvSharedPrefix,
    pub witness: KvForkWitness,
    pub layering: Option<KvAdapterLayering>,
}

/// A successful checked adoption.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvAdoptReport {
    /// Rows compared, summed over adopted layers.
    pub compared_rows: usize,
    /// Layer pages that now share the published allocation (already shared ones excluded).
    pub swapped_layer_pages: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvAdoptError {
    /// The request was refused before any row was compared.
    Backend(KvBackendError),
    /// At least one row differs bitwise from the published row; the cache is unchanged.
    Mismatch {
        mismatched_rows: usize,
        compared_rows: usize,
    },
}

impl From<KvBackendError> for KvAdoptError {
    fn from(error: KvBackendError) -> Self {
        Self::Backend(error)
    }
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

/// Paged KV backend. The only way to fork a published prefix is
/// [`PagedKvTensorBackend::fork_prefix_admitted`], which needs a single-use
/// [`KvForkWitness`]; the former binding-trusting `fork_prefix` no longer exists.
///
/// ```compile_fail
/// use ptr_pods::{InMemoryPagedKvBackend, InMemoryPagedKvCache, KvSharedPrefix, PagedKvTensorBackend};
///
/// fn fork_without_witness(
///     backend: &InMemoryPagedKvBackend,
///     cache: &mut InMemoryPagedKvCache,
///     prefix: &KvSharedPrefix,
/// ) {
///     // The caller's binding is no longer trusted: this method does not exist.
///     backend.fork_prefix(cache, prefix).unwrap();
/// }
/// ```
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

    /// Publish the full pages of a layer span. The default is
    /// [`KvBackendError::Unsupported`].
    fn publish_prefix_layers(
        &self,
        _cache: &mut Self::Cache,
        _request: KvLayerPublishRequest,
    ) -> Result<KvSharedPrefix, KvBackendError> {
        Err(KvBackendError::Unsupported("publish_prefix_layers"))
    }

    /// Fork an empty cache from layer-scoped prefixes covering every layer disjointly with
    /// identical page boundaries, under one single-use witness per part. The default is
    /// [`KvBackendError::Unsupported`].
    fn fork_prefix_layers_admitted(
        &self,
        _cache: &mut Self::Cache,
        _request: KvLayerForkRequest,
    ) -> Result<(), KvBackendError> {
        Err(KvBackendError::Unsupported("fork_prefix_layers_admitted"))
    }

    /// Checked adoption: compare every published row bitwise with the rows the cache already
    /// computed and, only if all are equal, share the published allocation. The default is
    /// [`KvBackendError::Unsupported`].
    fn adopt_prefix_layers_admitted(
        &self,
        _cache: &mut Self::Cache,
        _request: KvLayerAdoptRequest,
    ) -> Result<KvAdoptReport, KvAdoptError> {
        Err(KvAdoptError::Backend(KvBackendError::Unsupported(
            "adopt_prefix_layers_admitted",
        )))
    }

    /// Remove a published prefix so it can no longer be forked. A key that is not
    /// registered is [`KvBackendError::PrefixNotFound`], so a double retire is visible.
    /// Caches that already forked the prefix keep their pages until they are released.
    /// The default is [`KvBackendError::Unsupported`]: a backend that cannot retire
    /// must not be given a catalog that depends on retirement.
    fn retire_prefix(&self, _key: &KvPrefixKey) -> Result<(), KvBackendError> {
        Err(KvBackendError::Unsupported("retire_prefix"))
    }

    /// Hand out this backend's [`KvForkAuthority`], exactly once. The runtime's prefix
    /// catalog takes it when it is built; nobody else can then mint a witness.
    fn take_fork_authority(&self) -> Result<KvForkAuthority, KvBackendError> {
        Err(KvBackendError::Unsupported("take_fork_authority"))
    }

    /// Fork a published prefix under a runtime-minted, single-use witness instead of
    /// trusting the caller's binding generation (doc 36 defect 2). This is the only fork
    /// path: the former binding-trusting `fork_prefix` is removed from the trait. The
    /// witness is consumed whether or not the fork then succeeds.
    fn fork_prefix_admitted(
        &self,
        _cache: &mut Self::Cache,
        _prefix: &KvSharedPrefix,
        _witness: KvForkWitness,
    ) -> Result<(), KvBackendError> {
        Err(KvBackendError::Unsupported("fork_prefix_admitted"))
    }

    /// Read-only per-page byte sizes of a published prefix: one `(digest, bytes)` per
    /// unique page digest of `prefix.page_digests`, in first-occurrence order. `bytes` is
    /// the page's resident size in the same unit as [`KvPageMetrics::resident_bytes`]
    /// (K and V values of every layer for the page's valid tokens, 4 bytes per value), so a
    /// catalog can charge each shared page once. A prefix the backend no longer holds is
    /// [`KvBackendError::PrefixNotFound`]; digests that do not match the stored pages are
    /// [`KvBackendError::PrefixMismatch`]. The default is [`KvBackendError::Unsupported`].
    fn prefix_page_bytes(
        &self,
        _prefix: &KvSharedPrefix,
    ) -> Result<Vec<(Digest, u64)>, KvBackendError> {
        Err(KvBackendError::Unsupported("prefix_page_bytes"))
    }

    fn synchronize(&self, cache: &Self::Cache) -> Result<(), KvBackendError>;

    fn page_metrics(&self, cache: &Self::Cache) -> KvPageMetrics;
}

/// Why a [`KvForkWitness`] was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvWitnessRejection {
    /// Minted by the authority of a different backend.
    ForeignBackend,
    /// Minted for a different prefix than the one being forked.
    PrefixMismatch,
    /// Already used for a fork.
    Consumed,
    /// Revoked by a dependency invalidation before it was used.
    Revoked,
}

/// Lifecycle of a witness, shared between the witness and its [`KvWitnessRevoker`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvWitnessState {
    Pending,
    Consumed,
    Revoked,
}

const WITNESS_PENDING: u8 = 0;
const WITNESS_CONSUMED: u8 = 1;
const WITNESS_REVOKED: u8 = 2;

static NEXT_BACKEND_ID: AtomicU64 = AtomicU64::new(1);

/// The capability to mint [`KvForkWitness`] values for one backend.
///
/// Compiler-enforced scope: a witness has private fields and its only constructor is
/// [`KvForkAuthority::admit`], which needs a `KvForkAuthority`. The authority has no
/// public constructor and is not `Clone`; a backend hands it out once through
/// [`PagedKvTensorBackend::take_fork_authority`], and the runtime catalog takes it when
/// it is built. A caller that only holds the backend can therefore not forge a witness.
/// This is a type-level capability, not a defence against code that already controls the
/// process.
#[derive(Debug)]
pub struct KvForkAuthority {
    backend_id: u64,
}

impl KvForkAuthority {
    /// Mint a single-use witness for `prefix`, with the handle that can revoke it.
    pub fn admit(&self, prefix: &KvSharedPrefix) -> (KvForkWitness, KvWitnessRevoker) {
        let state = Arc::new(AtomicU8::new(WITNESS_PENDING));
        (
            KvForkWitness {
                backend_id: self.backend_id,
                prefix: prefix.key.clone(),
                state: Arc::clone(&state),
            },
            KvWitnessRevoker { state },
        )
    }
}

/// Single-use proof that the runtime admitted one fork of one prefix. Not `Clone`; it is
/// consumed by value, so a second fork with the same witness does not compile:
///
/// ```compile_fail
/// use ptr_pods::{InMemoryPagedKvBackend, InMemoryPagedKvCache, KvForkWitness, KvSharedPrefix, PagedKvTensorBackend};
///
/// fn reuse(
///     backend: &InMemoryPagedKvBackend,
///     a: &mut InMemoryPagedKvCache,
///     b: &mut InMemoryPagedKvCache,
///     prefix: &KvSharedPrefix,
///     witness: KvForkWitness,
/// ) {
///     let _ = backend.fork_prefix_admitted(a, prefix, witness);
///     let _ = backend.fork_prefix_admitted(b, prefix, witness);
/// }
/// ```
#[derive(Debug)]
pub struct KvForkWitness {
    backend_id: u64,
    prefix: KvPrefixKey,
    state: Arc<AtomicU8>,
}

impl KvForkWitness {
    pub fn prefix(&self) -> &KvPrefixKey {
        &self.prefix
    }

    fn consume(&self, backend_id: u64, prefix: &KvPrefixKey) -> Result<(), KvBackendError> {
        let reject = |why| Err(KvBackendError::ForkWitnessRejected(why));
        if self.backend_id != backend_id {
            return reject(KvWitnessRejection::ForeignBackend);
        }
        if &self.prefix != prefix {
            return reject(KvWitnessRejection::PrefixMismatch);
        }
        match self.state.compare_exchange(
            WITNESS_PENDING,
            WITNESS_CONSUMED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(WITNESS_REVOKED) => reject(KvWitnessRejection::Revoked),
            Err(_) => reject(KvWitnessRejection::Consumed),
        }
    }
}

/// Held by the catalog to revoke a witness that has not been used yet.
#[derive(Clone, Debug)]
pub struct KvWitnessRevoker {
    state: Arc<AtomicU8>,
}

impl KvWitnessRevoker {
    /// Revoke a pending witness; returns whether it was still pending.
    pub fn revoke(&self) -> bool {
        self.state
            .compare_exchange(
                WITNESS_PENDING,
                WITNESS_REVOKED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub fn state(&self) -> KvWitnessState {
        match self.state.load(Ordering::Acquire) {
            WITNESS_PENDING => KvWitnessState::Pending,
            WITNESS_CONSUMED => KvWitnessState::Consumed,
            _ => KvWitnessState::Revoked,
        }
    }
}

#[derive(Clone, Debug)]
struct HostPageBundle {
    page_id: KvPageId,
    valid_tokens: u16,
    state: KvPageState,
    digest: Option<Digest>,
    /// One independently shared allocation per layer: cloning a bundle is shallow and a write
    /// to layer `l` copies only layer `l` (`Arc::make_mut`), never the other layers.
    layers: Vec<PageLayer>,
}

/// The leading rows of an allocation owned by someone else.
#[derive(Clone, Debug)]
struct SharedRows {
    source: Arc<KvLayerSnapshot>,
    rows: usize,
}

/// One layer of one page: optionally the first `shared.rows` rows of a shared allocation,
/// followed by the rows of `owned`. Truncating a shared layer keeps the surviving rows
/// physically shared (a view of the shared allocation) and later appends go to `owned`, so a
/// boundary page can share its pre-boundary rows with the page it was cut from. The
/// representation is invisible to digests, snapshots and row access.
#[derive(Clone, Debug)]
struct PageLayer {
    shared: Option<SharedRows>,
    owned: Arc<KvLayerSnapshot>,
}

impl PageLayer {
    fn whole(layer: KvLayerSnapshot) -> Self {
        Self {
            shared: None,
            owned: Arc::new(layer),
        }
    }

    fn width(&self) -> usize {
        self.owned.keys.shape.get(1).copied().unwrap_or(0)
    }

    fn shared_rows(&self) -> usize {
        self.shared.as_ref().map_or(0, |shared| shared.rows)
    }

    fn rows(&self) -> usize {
        self.shared_rows() + self.owned.keys.shape.first().copied().unwrap_or(0)
    }

    /// Key and value values in row order as at most two contiguous segments.
    fn segments(&self) -> ([&[f32]; 2], [&[f32]; 2]) {
        let owned = (&self.owned.keys.values[..], &self.owned.values.values[..]);
        match &self.shared {
            None => ([owned.0, &[]], [owned.1, &[]]),
            Some(shared) => {
                let len = shared.rows * self.width();
                (
                    [&shared.source.keys.values[..len], owned.0],
                    [&shared.source.values.values[..len], owned.1],
                )
            }
        }
    }

    fn row(&self, row: usize) -> (&[f32], &[f32]) {
        let width = self.width();
        let (keys, values) = self.segments();
        let start = row * width;
        let first = keys[0].len();
        if start < first {
            (
                &keys[0][start..start + width],
                &values[0][start..start + width],
            )
        } else {
            let start = start - first;
            (
                &keys[1][start..start + width],
                &values[1][start..start + width],
            )
        }
    }

    fn materialize(&self, index: usize) -> KvLayerSnapshot {
        let (keys, values) = self.segments();
        let shape = vec![self.rows(), self.width()];
        KvLayerSnapshot {
            keys: TensorRef {
                layer: index,
                shape: shape.clone(),
                values: keys.concat(),
            },
            values: TensorRef {
                layer: index,
                shape,
                values: values.concat(),
            },
        }
    }

    fn value_count(&self) -> usize {
        let (keys, values) = self.segments();
        keys.iter()
            .chain(values.iter())
            .map(|part| part.len())
            .sum()
    }

    fn digest(&self, index: usize) -> Digest {
        let (keys, values) = self.segments();
        layer_digest_segments(index, &[self.rows(), self.width()], keys, values)
    }

    fn extend(&mut self, keys: &[f32], values: &[f32], count: usize) {
        let target = Arc::make_mut(&mut self.owned);
        target.keys.values.extend_from_slice(keys);
        target.values.values.extend_from_slice(values);
        target.keys.shape[0] += count;
        target.values.shape[0] += count;
    }

    /// Keep the first `keep` rows (`0 < keep < rows`). A layer that is shared with another
    /// page keeps its rows as a shared view instead of being copied; returns whether rows
    /// had to be copied.
    fn truncate(&mut self, keep: usize) -> bool {
        let width = self.width();
        let layer = self.owned.keys.layer;
        let shared_rows = self.shared_rows();
        if keep <= shared_rows {
            if let Some(shared) = &mut self.shared {
                shared.rows = keep;
            }
            self.owned = Arc::new(empty_layer(layer, width));
            return false;
        }
        if self.shared.is_none() && Arc::strong_count(&self.owned) > 1 {
            let source = std::mem::replace(&mut self.owned, Arc::new(empty_layer(layer, width)));
            self.shared = Some(SharedRows { source, rows: keep });
            return false;
        }
        let copied = Arc::strong_count(&self.owned) > 1;
        let owned = Arc::make_mut(&mut self.owned);
        let rows = keep - shared_rows;
        owned.keys.values.truncate(rows * width);
        owned.values.values.truncate(rows * width);
        owned.keys.shape[0] = rows;
        owned.values.shape[0] = rows;
        copied
    }

    /// The allocation and row count of the first segment, for sharing diagnostics.
    fn leading_segment(&self) -> (*const KvLayerSnapshot, usize) {
        match &self.shared {
            Some(shared) => (Arc::as_ptr(&shared.source), shared.rows),
            None => (
                Arc::as_ptr(&self.owned),
                self.owned.keys.shape.first().copied().unwrap_or(0),
            ),
        }
    }
}

fn empty_layer(layer: usize, width: usize) -> KvLayerSnapshot {
    KvLayerSnapshot {
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
    }
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
enum PrefixRecord {
    /// Every layer of whole pages, published by `publish_prefix`.
    Whole(WholePrefix),
    /// The layers of one span, published by `publish_prefix_layers`.
    Layered(Arc<LayeredPrefix>),
}

#[derive(Clone)]
struct WholePrefix {
    pages: Vec<Arc<HostPageBundle>>,
    sequence_length: usize,
    position_offset: usize,
}

struct LayeredPrefix {
    /// The resolved layer span.
    layers: std::ops::Range<usize>,
    pages: Vec<LayeredPage>,
    sequence_length: usize,
    position_offset: usize,
}

struct LayeredPage {
    valid_tokens: u16,
    digest: Digest,
    /// One entry per layer of the span, in layer order.
    layers: Vec<PageLayer>,
}

struct PagedBackendState {
    id: u64,
    authority_taken: Mutex<bool>,
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
            state: Arc::new(PagedBackendState {
                id: NEXT_BACKEND_ID.fetch_add(1, Ordering::Relaxed),
                authority_taken: Mutex::new(false),
                next_page_id: AtomicU64::new(0),
                prefixes: Mutex::new(BTreeMap::new()),
            }),
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
            layers.push(PageLayer::whole(KvLayerSnapshot {
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
            }));
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
        let digest = page_digest_from_layer_digests(
            binding,
            ordinal,
            page.valid_tokens,
            page.layers
                .iter()
                .enumerate()
                .map(|(index, layer)| layer.digest(index)),
        );
        page.digest = Some(digest);
        page.state = KvPageState::Sealed;
        Ok(())
    }

    /// Returns the cache's tail page for writing. A shared or sealed tail is first replaced by
    /// a shallow copy (new page id, writable, digest cleared) whose layers are still the same
    /// `Arc`s; each layer is copied only when it is itself written (`extend_layer`).
    fn writable_tail<'a>(
        &self,
        cache: &'a mut InMemoryPagedKvCache,
    ) -> Result<&'a mut HostPageBundle, KvBackendError> {
        let ordinal = cache
            .pages
            .len()
            .checked_sub(1)
            .ok_or(KvBackendError::InvalidPageState)?;
        let page = &cache.pages[ordinal];
        if Arc::strong_count(page) > 1 || page.state == KvPageState::Sealed {
            let mut copied = (**page).clone();
            copied.page_id = self.next_page_id()?;
            copied.state = KvPageState::Writable;
            copied.digest = None;
            cache.pages[ordinal] = Arc::new(copied);
            cache.cow_copies = cache.cow_copies.saturating_add(1);
        }
        Arc::get_mut(&mut cache.pages[ordinal]).ok_or(KvBackendError::InvalidPageState)
    }

    /// Copy-on-write append of `count` tokens to one layer of a page: only that layer's
    /// allocation is cloned (and only when it is still shared); sibling layers stay shared.
    fn extend_layer(
        page: &mut HostPageBundle,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        count: usize,
    ) {
        page.layers[layer].extend(keys, values, count);
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
                layers: page
                    .layers
                    .iter()
                    .enumerate()
                    .map(|(index, layer)| layer.materialize(index))
                    .collect(),
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

impl InMemoryPagedKvBackend {
    /// The stored layered record for `prefix`, after checking that the cache may use it: same
    /// principal and model, the cache's EFFECTIVE adapter for the span equals the key's
    /// adapter, and the schema digest (computed with that adapter) matches.
    fn layered_record_for(
        &self,
        cache: &InMemoryPagedKvCache,
        layering: Option<&KvAdapterLayering>,
        prefix: &KvSharedPrefix,
    ) -> Result<Arc<LayeredPrefix>, KvBackendError> {
        let key = &prefix.key;
        let span = key.layers.resolve(cache.schema.layer_count)?;
        let adapter = effective_adapter(&cache.binding.adapter, layering, &span);
        let mut schema = cache.schema.clone();
        schema.adapter = adapter.clone();
        if cache.binding.principal != key.principal
            || cache.binding.model != key.model
            || adapter != key.adapter
            || schema_digest(&schema, cache.binding.page_tokens) != key.schema_digest
        {
            return Err(KvBackendError::PrefixMismatch);
        }
        let record = self
            .state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?
            .get(key)
            .cloned()
            .ok_or(KvBackendError::PrefixNotFound)?;
        match record {
            PrefixRecord::Layered(record) if record.layers == span => Ok(record),
            PrefixRecord::Layered(_) | PrefixRecord::Whole(_) => {
                Err(KvBackendError::PrefixMismatch)
            }
        }
    }

    fn layered_page_bytes(
        record: &LayeredPrefix,
        prefix: &KvSharedPrefix,
    ) -> Result<Vec<(Digest, u64)>, KvBackendError> {
        let stored: Vec<Digest> = record.pages.iter().map(|page| page.digest).collect();
        if stored != prefix.page_digests {
            return Err(KvBackendError::PrefixMismatch);
        }
        let mut charges: Vec<(Digest, u64)> = Vec::new();
        for page in &record.pages {
            if charges.iter().any(|(seen, _)| *seen == page.digest) {
                continue;
            }
            let values: usize = page.layers.iter().map(PageLayer::value_count).sum();
            charges.push((
                page.digest,
                u64::try_from(values.saturating_mul(4)).unwrap_or(u64::MAX),
            ));
        }
        Ok(charges)
    }
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Page digest for a layer-scoped prefix: the binding (with the effective adapter), the
/// page position, the layer span and the span's ordered per-layer digests.
fn layered_page_digest(
    binding: &KvPageBinding,
    ordinal: usize,
    valid_tokens: u16,
    span: &std::ops::Range<usize>,
    layer_digests: impl Iterator<Item = Digest>,
) -> Digest {
    let mut bytes = b"ptr-paged-kv-page-layers-v1\0".to_vec();
    put_binding(&mut bytes, binding);
    bytes.extend_from_slice(&(ordinal as u64).to_le_bytes());
    bytes.extend_from_slice(&valid_tokens.to_le_bytes());
    bytes.extend_from_slice(&(span.start as u64).to_le_bytes());
    bytes.extend_from_slice(&(span.end as u64).to_le_bytes());
    for digest in layer_digests {
        bytes.extend_from_slice(&digest);
    }
    Sha256::digest(bytes).into()
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

    /// Diagnostic: how many leading rows of `layer` in `page` are physically the same
    /// allocation in this cache and in `other` (0 when either does not have the page layer
    /// or they do not start in the same allocation).
    pub fn shared_leading_rows(&self, other: &Self, page: usize, layer: usize) -> usize {
        let (Some(mine), Some(theirs)) = (
            self.pages.get(page).and_then(|page| page.layers.get(layer)),
            other
                .pages
                .get(page)
                .and_then(|page| page.layers.get(layer)),
        ) else {
            return 0;
        };
        let ((mine_ptr, mine_rows), (their_ptr, their_rows)) =
            (mine.leading_segment(), theirs.leading_segment());
        if mine_ptr == their_ptr {
            mine_rows.min(their_rows)
        } else {
            0
        }
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
                let layer_count = cache.schema.layer_count;
                let page = self.writable_tail(cache)?;
                let available = page_tokens - usize::from(page.valid_tokens);
                let count = (token_count - consumed).min(available);
                for layer in 0..layer_count {
                    let range = consumed * width..(consumed + count) * width;
                    Self::extend_layer(
                        page,
                        layer,
                        &keys[layer].values[range.clone()],
                        &values[layer].values[range],
                        count,
                    );
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
                for layer in &mut page.layers {
                    layer.truncate(keep);
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
            layers: KvLayerRange::Full,
        };
        let sequence_length = full_pages.len() * cache.binding.page_tokens.get();
        self.state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?
            .insert(
                key.clone(),
                PrefixRecord::Whole(WholePrefix {
                    pages: full_pages,
                    sequence_length,
                    position_offset: cache.position_offset,
                }),
            );
        Ok(KvSharedPrefix {
            key,
            sequence_length,
            position_offset: cache.position_offset,
            page_digests,
        })
    }

    fn publish_prefix_layers(
        &self,
        cache: &mut Self::Cache,
        request: KvLayerPublishRequest,
    ) -> Result<KvSharedPrefix, KvBackendError> {
        let layer_count = cache.schema.layer_count;
        let span = request.layers.resolve(layer_count)?;
        check_layering(request.layering.as_ref(), layer_count)?;
        let scope = KvLayerRange::canonical(span.clone(), layer_count)?;
        let adapter = effective_adapter(&cache.binding.adapter, request.layering.as_ref(), &span);
        let _ = self.seal_snapshot(cache)?;
        let page_tokens = cache.binding.page_tokens.get();
        let mut binding = cache.binding.clone();
        binding.adapter = adapter.clone();
        let mut pages = Vec::new();
        for (ordinal, page) in cache
            .pages
            .iter()
            .take_while(|page| usize::from(page.valid_tokens) == page_tokens)
            .enumerate()
        {
            let layers: Vec<PageLayer> = page.layers[span.clone()].to_vec();
            let digest = layered_page_digest(
                &binding,
                ordinal,
                page.valid_tokens,
                &span,
                layers
                    .iter()
                    .zip(span.clone())
                    .map(|(layer, index)| layer.digest(index)),
            );
            pages.push(LayeredPage {
                valid_tokens: page.valid_tokens,
                digest,
                layers,
            });
        }
        if pages.is_empty() {
            return Err(KvBackendError::PrefixMismatch);
        }
        let page_digests: Vec<Digest> = pages.iter().map(|page| page.digest).collect();
        let mut schema = cache.schema.clone();
        schema.adapter = adapter.clone();
        let mut digest_bytes = b"ptr-kv-prefix-layers-v1\0".to_vec();
        digest_bytes.extend_from_slice(&(span.start as u64).to_le_bytes());
        digest_bytes.extend_from_slice(&(span.end as u64).to_le_bytes());
        for digest in &page_digests {
            digest_bytes.extend_from_slice(digest);
        }
        let key = KvPrefixKey {
            principal: cache.binding.principal.clone(),
            execution_manifest: cache.binding.execution_manifest,
            model: cache.binding.model.clone(),
            adapter,
            generation: cache.binding.generation,
            schema_digest: schema_digest(&schema, cache.binding.page_tokens),
            prefix_digest: Sha256::digest(digest_bytes).into(),
            layers: scope,
        };
        let sequence_length = pages.len() * page_tokens;
        self.state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?
            .insert(
                key.clone(),
                PrefixRecord::Layered(Arc::new(LayeredPrefix {
                    layers: span,
                    pages,
                    sequence_length,
                    position_offset: cache.position_offset,
                })),
            );
        Ok(KvSharedPrefix {
            key,
            sequence_length,
            position_offset: cache.position_offset,
            page_digests,
        })
    }

    fn fork_prefix_layers_admitted(
        &self,
        cache: &mut Self::Cache,
        request: KvLayerForkRequest,
    ) -> Result<(), KvBackendError> {
        if cache.revoked || cache.sequence_length != 0 || !cache.pages.is_empty() {
            return Err(KvBackendError::PrefixMismatch);
        }
        let layer_count = cache.schema.layer_count;
        check_layering(request.layering.as_ref(), layer_count)?;
        let mut parts = Vec::with_capacity(request.parts.len());
        for (prefix, witness) in &request.parts {
            let record = self.layered_record_for(cache, request.layering.as_ref(), prefix)?;
            parts.push((prefix, witness, record));
        }
        parts.sort_by_key(|(_, _, record)| record.layers.start);
        // Disjoint cover of 0..L: each part starts where the previous one ended.
        let mut next = 0usize;
        for (_, _, record) in &parts {
            if record.layers.start != next {
                return Err(KvBackendError::PrefixMismatch);
            }
            next = record.layers.end;
        }
        let Some((_, _, first)) = parts.first() else {
            return Err(KvBackendError::PrefixMismatch);
        };
        if next != layer_count {
            return Err(KvBackendError::PrefixMismatch);
        }
        // Identical page boundaries and position across parts.
        for (_, _, record) in &parts {
            if record.sequence_length != first.sequence_length
                || record.position_offset != first.position_offset
                || record.pages.len() != first.pages.len()
                || record
                    .pages
                    .iter()
                    .zip(&first.pages)
                    .any(|(a, b)| a.valid_tokens != b.valid_tokens)
            {
                return Err(KvBackendError::PrefixMismatch);
            }
        }
        if first.sequence_length > cache.capacity_tokens {
            return Err(KvBackendError::PrefixMismatch);
        }
        // Every witness is spent by this call, whether or not a later one is refused.
        for (prefix, witness, _) in &parts {
            witness.consume(self.state.id, &prefix.key)?;
        }
        let mut pages = Vec::with_capacity(first.pages.len());
        for ordinal in 0..first.pages.len() {
            let mut layers: Vec<PageLayer> = Vec::with_capacity(layer_count);
            for (_, _, record) in &parts {
                layers.extend(record.pages[ordinal].layers.iter().cloned());
            }
            let mut bundle = HostPageBundle {
                page_id: self.next_page_id()?,
                valid_tokens: first.pages[ordinal].valid_tokens,
                state: KvPageState::Writable,
                digest: None,
                layers,
            };
            Self::seal_page(&mut bundle, &cache.binding, ordinal)?;
            pages.push(Arc::new(bundle));
        }
        cache.pages = pages;
        cache.sequence_length = first.sequence_length;
        cache.position_offset = first.position_offset;
        Ok(())
    }

    fn adopt_prefix_layers_admitted(
        &self,
        cache: &mut Self::Cache,
        request: KvLayerAdoptRequest,
    ) -> Result<KvAdoptReport, KvAdoptError> {
        if cache.revoked {
            return Err(KvBackendError::StaleLease.into());
        }
        check_layering(request.layering.as_ref(), cache.schema.layer_count)?;
        let record = self.layered_record_for(cache, request.layering.as_ref(), &request.prefix)?;
        if record.sequence_length > cache.sequence_length
            || record.position_offset != cache.position_offset
            || record.pages.len() > cache.pages.len()
            || record
                .pages
                .iter()
                .zip(&cache.pages)
                .any(|(stored, page)| stored.valid_tokens != page.valid_tokens)
        {
            return Err(KvBackendError::PrefixMismatch.into());
        }
        // The admission is spent from here on, also when the rows turn out to differ.
        request
            .witness
            .consume(self.state.id, &request.prefix.key)?;
        let mut compared_rows = 0usize;
        let mut mismatched_rows = 0usize;
        for (stored, page) in record.pages.iter().zip(&cache.pages) {
            for (offset, stored_layer) in stored.layers.iter().enumerate() {
                let layer = &page.layers[record.layers.start + offset];
                if layer.rows() != stored_layer.rows() || layer.width() != stored_layer.width() {
                    return Err(KvBackendError::PrefixMismatch.into());
                }
                for row in 0..layer.rows() {
                    compared_rows += 1;
                    let (keys, values) = layer.row(row);
                    let (stored_keys, stored_values) = stored_layer.row(row);
                    if !bits_equal(keys, stored_keys) || !bits_equal(values, stored_values) {
                        mismatched_rows += 1;
                    }
                }
            }
        }
        if mismatched_rows > 0 {
            return Err(KvAdoptError::Mismatch {
                mismatched_rows,
                compared_rows,
            });
        }
        let mut swapped_layer_pages = 0usize;
        for (stored, page) in record.pages.iter().zip(&mut cache.pages) {
            let page = Arc::make_mut(page);
            for (offset, stored_layer) in stored.layers.iter().enumerate() {
                let layer = &mut page.layers[record.layers.start + offset];
                if layer.leading_segment() != stored_layer.leading_segment()
                    || layer.shared_rows() != stored_layer.shared_rows()
                {
                    *layer = stored_layer.clone();
                    swapped_layer_pages += 1;
                }
            }
        }
        Ok(KvAdoptReport {
            compared_rows,
            swapped_layer_pages,
        })
    }

    fn retire_prefix(&self, key: &KvPrefixKey) -> Result<(), KvBackendError> {
        self.state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?
            .remove(key)
            .map(|_| ())
            .ok_or(KvBackendError::PrefixNotFound)
    }

    fn take_fork_authority(&self) -> Result<KvForkAuthority, KvBackendError> {
        let mut taken = self
            .state
            .authority_taken
            .lock()
            .map_err(|_| KvBackendError::Backend("fork authority poisoned".into()))?;
        if *taken {
            return Err(KvBackendError::Backend(
                "fork authority already taken".into(),
            ));
        }
        *taken = true;
        Ok(KvForkAuthority {
            backend_id: self.state.id,
        })
    }

    fn fork_prefix_admitted(
        &self,
        cache: &mut Self::Cache,
        prefix: &KvSharedPrefix,
        witness: KvForkWitness,
    ) -> Result<(), KvBackendError> {
        if cache.revoked || cache.sequence_length != 0 || !cache.pages.is_empty() {
            return Err(KvBackendError::PrefixMismatch);
        }
        // The witness, not the caller's binding, carries admission. The cache must still
        // belong to the same principal, model, adapter and layout as the producer; the
        // producing generation and manifest digest are deliberately not compared (doc 36
        // section D): their liveness is the catalog's use-time check.
        let key = &prefix.key;
        if cache.binding.principal != key.principal
            || cache.binding.model != key.model
            || cache.binding.adapter != key.adapter
            || schema_digest(&cache.schema, cache.binding.page_tokens) != key.schema_digest
        {
            return Err(KvBackendError::PrefixMismatch);
        }
        let record = match self
            .state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?
            .get(key)
            .cloned()
            .ok_or(KvBackendError::PrefixNotFound)?
        {
            PrefixRecord::Whole(record) => record,
            // A layer-scoped prefix cannot stand alone; it forks through the layered path.
            PrefixRecord::Layered(_) => return Err(KvBackendError::PrefixMismatch),
        };
        if record.sequence_length > cache.capacity_tokens {
            return Err(KvBackendError::PrefixMismatch);
        }
        witness.consume(self.state.id, key)?;
        cache.pages = record.pages;
        cache.sequence_length = record.sequence_length;
        cache.position_offset = record.position_offset;
        Ok(())
    }

    fn prefix_page_bytes(
        &self,
        prefix: &KvSharedPrefix,
    ) -> Result<Vec<(Digest, u64)>, KvBackendError> {
        let prefixes = self
            .state
            .prefixes
            .lock()
            .map_err(|_| KvBackendError::Backend("prefix catalog poisoned".into()))?;
        let record = prefixes
            .get(&prefix.key)
            .ok_or(KvBackendError::PrefixNotFound)?;
        let record = match record {
            PrefixRecord::Whole(record) => record,
            PrefixRecord::Layered(record) => return Self::layered_page_bytes(record, prefix),
        };
        let stored: Vec<Option<Digest>> = record.pages.iter().map(|page| page.digest).collect();
        let offered: Vec<Option<Digest>> = prefix.page_digests.iter().map(|d| Some(*d)).collect();
        if stored != offered {
            return Err(KvBackendError::PrefixMismatch);
        }
        let mut charges: Vec<(Digest, u64)> = Vec::new();
        for page in &record.pages {
            let digest = page.digest.ok_or(KvBackendError::InvalidPageState)?;
            if charges.iter().any(|(seen, _)| *seen == digest) {
                continue;
            }
            // Same unit as `page_metrics`: every stored f32 of K and V, over all layers.
            let values: usize = page.layers.iter().map(PageLayer::value_count).sum();
            charges.push((
                digest,
                u64::try_from(values.saturating_mul(4)).unwrap_or(u64::MAX),
            ));
        }
        Ok(charges)
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
    page_digest_from_layer_digests(
        binding,
        ordinal,
        valid_tokens,
        layers
            .iter()
            .enumerate()
            .map(|(index, layer)| layer_digest(index, layer)),
    )
}

/// Page digest v2: domain-separated digest over the binding, position and the ordered
/// per-layer digests, so a page's identity is the composition of its layers' identities.
pub(crate) fn page_digest_from_layer_digests(
    binding: &KvPageBinding,
    ordinal: usize,
    valid_tokens: u16,
    layer_digests: impl ExactSizeIterator<Item = Digest>,
) -> Digest {
    let mut bytes = b"ptr-paged-kv-page-v2\0".to_vec();
    put_binding(&mut bytes, binding);
    bytes.extend_from_slice(&(ordinal as u64).to_le_bytes());
    bytes.extend_from_slice(&valid_tokens.to_le_bytes());
    bytes.extend_from_slice(&(layer_digests.len() as u64).to_le_bytes());
    for digest in layer_digests {
        bytes.extend_from_slice(&digest);
    }
    Sha256::digest(bytes).into()
}

/// Digest of one layer's keys and values within a page (layer index included).
pub(crate) fn layer_digest(index: usize, layer: &KvLayerSnapshot) -> Digest {
    layer_digest_segments(
        index,
        &layer.keys.shape,
        [&layer.keys.values, &[]],
        [&layer.values.values, &[]],
    )
}

/// The layer digest over rows given as contiguous segments: the bytes hashed are exactly
/// those of the same rows in one allocation, so the digest does not depend on how a page
/// layer is represented. Keys and values share `shape`; the tensors' layer index is `index`.
fn layer_digest_segments(
    index: usize,
    shape: &[usize],
    keys: [&[f32]; 2],
    values: [&[f32]; 2],
) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-paged-kv-layer-v1\0");
    hasher.update((index as u64).to_le_bytes());
    for segments in [keys, values] {
        hasher.update((index as u64).to_le_bytes());
        hasher.update((shape.len() as u64).to_le_bytes());
        for dimension in shape {
            hasher.update((*dimension as u64).to_le_bytes());
        }
        hasher.update(
            segments
                .iter()
                .map(|part| part.len() as u64)
                .sum::<u64>()
                .to_le_bytes(),
        );
        for value in segments.into_iter().flatten() {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    hasher.finalize().into()
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

#[cfg(test)]
mod per_layer_page_tests {
    use super::*;

    const LAYERS: usize = 3;
    const WIDTH: usize = 4;

    fn schema() -> KvTensorSchema {
        KvTensorSchema {
            model: ModelVersion::from("m"),
            adapter: AdapterVersion::from("a"),
            device: DeviceId::from("cpu"),
            layer_count: LAYERS,
            attention_heads: 2,
            key_value_heads: 2,
            head_dim: 2,
            batch_size: 1,
            dtype: KvTensorDType::F32,
        }
    }

    fn rows(layer: usize, tokens: usize, seed: f32) -> TensorRef {
        TensorRef {
            layer,
            shape: vec![tokens, WIDTH],
            values: (0..tokens * WIDTH)
                .map(|i| seed + (layer * 100 + i) as f32)
                .collect(),
        }
    }

    /// Cache with one sealed full page (4 tokens) and a writable partial tail (2 tokens).
    fn partial_cache(backend: &InMemoryPagedKvBackend) -> InMemoryPagedKvCache {
        let mut cache = backend.allocate(schema(), 16).unwrap();
        let keys: Vec<_> = (0..LAYERS).map(|l| rows(l, 6, 1.0)).collect();
        let values: Vec<_> = (0..LAYERS).map(|l| rows(l, 6, 9.0)).collect();
        backend.append(&mut cache, &keys, &values).unwrap();
        cache
    }

    fn unique_layer_bytes(caches: &[&InMemoryPagedKvCache]) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        let mut bytes = 0;
        for cache in caches {
            for page in &cache.pages {
                for layer in &page.layers {
                    if seen.insert(Arc::as_ptr(&layer.owned) as usize) {
                        bytes +=
                            (layer.owned.keys.values.len() + layer.owned.values.values.len()) * 4;
                    }
                }
            }
        }
        bytes
    }

    #[test]
    fn writing_one_layer_of_a_forked_tail_copies_only_that_layer() {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
        let mut source = partial_cache(&backend);
        backend.seal_snapshot(&mut source).unwrap();
        let before = InMemoryPagedKvBackend::paged_snapshot(&source).unwrap();
        let mut fork = source.clone();
        let bytes_before = unique_layer_bytes(&[&source, &fork]);
        assert_eq!(bytes_before, unique_layer_bytes(&[&source]));

        let page = backend.writable_tail(&mut fork).unwrap();
        InMemoryPagedKvBackend::extend_layer(page, 1, &[7.0; WIDTH], &[8.0; WIDTH], 1);

        let tail = source.pages.len() - 1;
        for layer in 0..LAYERS {
            let same = Arc::ptr_eq(
                &source.pages[tail].layers[layer].owned,
                &fork.pages[tail].layers[layer].owned,
            );
            assert_eq!(same, layer != 1, "layer {layer}");
            // Fully sealed head page is untouched and still shared whole.
            assert!(Arc::ptr_eq(&source.pages[0], &fork.pages[0]));
        }
        // The source is bitwise unchanged.
        assert_eq!(
            InMemoryPagedKvBackend::paged_snapshot(&source).unwrap(),
            before
        );
        // Resident growth is exactly the modified layer's page bytes (keys + values, 2 tokens).
        let modified_layer_bytes = 2 * (2 * WIDTH) * 4;
        assert_eq!(
            unique_layer_bytes(&[&source, &fork]) - bytes_before,
            modified_layer_bytes + (2 * WIDTH) * 4 // the 1 appended token in that layer
        );
        assert_eq!(fork.cow_copies, 1);
    }

    #[test]
    fn full_append_after_fork_shares_every_sealed_page_and_leaves_source_intact() {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
        let mut source = partial_cache(&backend);
        backend.seal_snapshot(&mut source).unwrap();
        let before = InMemoryPagedKvBackend::paged_snapshot(&source).unwrap();
        let mut fork = source.clone();
        let keys: Vec<_> = (0..LAYERS).map(|l| rows(l, 1, 5.0)).collect();
        let values: Vec<_> = (0..LAYERS).map(|l| rows(l, 1, 6.0)).collect();
        backend.append(&mut fork, &keys, &values).unwrap();
        assert!(Arc::ptr_eq(&source.pages[0], &fork.pages[0]));
        assert_eq!(
            InMemoryPagedKvBackend::paged_snapshot(&source).unwrap(),
            before
        );
        assert_eq!(fork.sequence_length, 7);
    }

    #[test]
    fn page_digest_is_deterministic_and_sensitive_to_every_single_layer() {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
        let a = InMemoryPagedKvBackend::paged_snapshot(&{
            let mut c = partial_cache(&backend);
            backend.seal_snapshot(&mut c).unwrap();
            c
        })
        .unwrap();
        let b = InMemoryPagedKvBackend::paged_snapshot(&{
            let mut c = partial_cache(&backend);
            backend.seal_snapshot(&mut c).unwrap();
            c
        })
        .unwrap();
        assert_eq!(a.pages[0].digest, b.pages[0].digest);
        assert!(a.verify_digest());
        for layer in 0..LAYERS {
            let mut changed = a.pages[0].layers.clone();
            changed[layer].values.values[0] += 1.0;
            let digest = page_digest(&a.binding, 0, a.pages[0].valid_tokens, &changed);
            assert_ne!(digest, a.pages[0].digest, "layer {layer}");
        }
        // A tampered layer is refused by snapshot validation.
        let mut tampered = a.clone();
        tampered.pages[0].layers[2].keys.values[0] += 1.0;
        assert_eq!(
            tampered.validate(),
            Err(KvBackendError::SnapshotDigestMismatch)
        );
    }
}

#[cfg(test)]
mod witness_tests {
    use super::*;

    #[test]
    fn second_consume_of_the_same_witness_state_is_refused() {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::new(2).unwrap());
        let authority = backend.take_fork_authority().unwrap();
        let key = KvPrefixKey {
            principal: PrincipalId::from("alice"),
            execution_manifest: [1; 32],
            model: ModelVersion::from("m"),
            adapter: AdapterVersion::from("a"),
            generation: Generation(1),
            schema_digest: [2; 32],
            prefix_digest: [3; 32],
            layers: KvLayerRange::Full,
        };
        let prefix = KvSharedPrefix {
            key: key.clone(),
            sequence_length: 0,
            position_offset: 0,
            page_digests: Vec::new(),
        };
        let (witness, _) = authority.admit(&prefix);
        witness.consume(backend.state.id, &key).unwrap();
        assert_eq!(
            witness.consume(backend.state.id, &key),
            Err(KvBackendError::ForkWitnessRejected(
                KvWitnessRejection::Consumed
            ))
        );
    }
}

#[cfg(test)]
mod layer_prefix_tests {
    use super::*;

    const LAYERS: usize = 3;
    const WIDTH: usize = 4;

    fn schema() -> KvTensorSchema {
        KvTensorSchema {
            model: ModelVersion::from("m"),
            adapter: AdapterVersion::from("a"),
            device: DeviceId::from("cpu"),
            layer_count: LAYERS,
            attention_heads: 2,
            key_value_heads: 2,
            head_dim: 2,
            batch_size: 1,
            dtype: KvTensorDType::F32,
        }
    }

    fn rows(layer: usize, tokens: usize, seed: f32) -> TensorRef {
        TensorRef {
            layer,
            shape: vec![tokens, WIDTH],
            values: (0..tokens * WIDTH)
                .map(|i| seed + (layer * 100 + i) as f32)
                .collect(),
        }
    }

    fn filled(backend: &InMemoryPagedKvBackend, tokens: usize, seed: f32) -> InMemoryPagedKvCache {
        let mut cache = backend.allocate(schema(), 16).unwrap();
        let keys: Vec<_> = (0..LAYERS).map(|l| rows(l, tokens, seed)).collect();
        let values: Vec<_> = (0..LAYERS).map(|l| rows(l, tokens, 9.0)).collect();
        backend.append(&mut cache, &keys, &values).unwrap();
        cache
    }

    fn admitted(
        authority: &KvForkAuthority,
        prefix: &KvSharedPrefix,
    ) -> (KvForkWitness, KvWitnessRevoker) {
        authority.admit(prefix)
    }

    #[test]
    fn adopted_layers_are_the_published_allocations_and_a_mismatch_changes_nothing() {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
        let authority = backend.take_fork_authority().unwrap();
        let mut source = filled(&backend, 8, 1.0);
        let prefix = backend
            .publish_prefix_layers(
                &mut source,
                KvLayerPublishRequest {
                    layers: KvLayerRange::span(0, 2).unwrap(),
                    layering: None,
                },
            )
            .unwrap();
        let (witness, _) = admitted(&authority, &prefix);
        let mut computed = filled(&backend, 8, 1.0);
        backend
            .adopt_prefix_layers_admitted(
                &mut computed,
                KvLayerAdoptRequest {
                    prefix: prefix.clone(),
                    witness,
                    layering: None,
                },
            )
            .unwrap();
        for page in 0..2 {
            for layer in 0..LAYERS {
                assert_eq!(
                    Arc::ptr_eq(
                        &computed.pages[page].layers[layer].owned,
                        &source.pages[page].layers[layer].owned
                    ),
                    layer < 2,
                    "page {page} layer {layer}"
                );
            }
        }

        // One differing row: refused, every layer allocation of the cache is unchanged.
        let mut other = filled(&backend, 8, 1.0);
        let mut changed = other.pages[1].as_ref().clone();
        Arc::make_mut(&mut changed.layers[0].owned).keys.values[0] += 1.0;
        other.pages[1] = Arc::new(changed);
        let pointers = |cache: &InMemoryPagedKvCache| -> Vec<*const KvLayerSnapshot> {
            cache
                .pages
                .iter()
                .flat_map(|page| page.layers.iter().map(|l| Arc::as_ptr(&l.owned)))
                .collect()
        };
        let before = pointers(&other);
        let (witness, _) = admitted(&authority, &prefix);
        let refused = backend.adopt_prefix_layers_admitted(
            &mut other,
            KvLayerAdoptRequest {
                prefix,
                witness,
                layering: None,
            },
        );
        assert_eq!(
            refused,
            Err(KvAdoptError::Mismatch {
                mismatched_rows: 1,
                compared_rows: 16
            })
        );
        assert_eq!(pointers(&other), before);
    }

    #[test]
    fn truncated_shared_layer_keeps_its_prefix_allocation_across_append() {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap());
        let mut base = filled(&backend, 4, 1.0);
        backend.seal_snapshot(&mut base).unwrap();
        let mut fork = base.clone();
        backend.truncate(&mut fork, 3).unwrap();
        let keys: Vec<_> = (0..LAYERS).map(|l| rows(l, 1, 5.0)).collect();
        backend.append(&mut fork, &keys, &keys).unwrap();
        for layer in 0..LAYERS {
            let forked = &fork.pages[0].layers[layer];
            let shared = forked.shared.as_ref().expect("prefix stays shared");
            assert!(Arc::ptr_eq(
                &shared.source,
                &base.pages[0].layers[layer].owned
            ));
            assert_eq!(shared.rows, 3);
            assert_eq!(forked.owned.keys.shape[0], 1);
            // The segmented layer digests like the same rows in one allocation.
            let flat = forked.materialize(layer);
            assert_eq!(forked.digest(layer), layer_digest(layer, &flat));
        }
        assert_eq!(fork.sequence_length, 4);
    }
}
