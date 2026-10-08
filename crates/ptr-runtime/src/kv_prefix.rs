//! Exact prefix catalog with retirement (doc 36, phase P2).
//!
//! A radix tree over 16-token chunks. A chunk key is computable from the request's
//! tokens alone, before any prefill ([`KvChunkKey`], [`chunk_keys`]); it is the answer to
//! doc 36 defect 1 (the backend's page-digest key cannot be looked up before computing).
//! Each trie node holds per-layer-chain entries keyed by `(KvChainId, lineage digest)`;
//! an entry owns a published [`KvSharedPrefix`], the semantic-input dependencies it was
//! computed from, and a refcount.
//!
//! # Guarantees
//!
//! * **Isolation.** The [`KvSalt`] folds in principal, project, base model, tokenizer,
//!   numerics and schema, and nothing generation- or adapter-specific (doc 36 section D).
//!   Different salts diverge at the first node.
//! * **Use-time liveness (invariant 3).** [`KvPrefixCatalog::lookup`] and
//!   [`KvPrefixCatalog::fork`] ask a [`KvLivenessOracle`] about every recorded dependency
//!   on every use. Invalidation is an eviction optimisation; it is not what makes a stale
//!   entry unreachable, so a delayed or lost invalidation cannot produce a stale hit.
//! * **Witness-gated forks (defect 2).** A hit carries a single-use [`KvForkWitness`]
//!   minted from the backend's one [`KvForkAuthority`], which the catalog owns. A
//!   dependency invalidation revokes every witness not yet used and tombstones the entry,
//!   so an active fork can see that it was cancelled ([`KvPrefixCatalog::pin_is_live`]).
//! * **Bounded, fail closed (invariant 8).** Nodes, KV page bytes and metadata bytes are
//!   limited ([`CatalogLimits`]). [`KvPrefixCatalog::insert`] evicts unreferenced entries
//!   in LRU order and returns [`KvCatalogError::CapacityExhausted`] when only pinned
//!   entries remain; it never exceeds a bound and never evicts a pinned entry.
//! * **Byte accounting per unique page.** [`KvPrefixCatalog::insert_paged`] charges the
//!   page bytes once per unique page digest held by live entries (refcounted), so entries
//!   that share pages do not pay twice and the byte bound binds as configured. The older
//!   [`KvPrefixCatalog::insert`] charges its single `page_bytes` per entry.
//! * **Dependency kinds.** A [`KvDependency`] is a segment, an expert (with the expert's
//!   fence epoch), the base model or an adapter, each with its generation.
//! * **Snapshot without pages.** [`KvPrefixCatalog::export_entries`] writes a canonical,
//!   bounded snapshot of keys, lineage, dependencies and charges (never K/V bytes);
//!   [`KvPrefixCatalog::restore_entries`] strictly decodes it and admits the entries
//!   without a liveness check. The use-time check (and the backend, for pages that are
//!   gone) decides at the first use, so a restored entry whose input was revoked, or whose
//!   pages were lost, misses cleanly.
//! * **Secondary index.** The catalog owns one position-independent
//!   [`SecondaryIndex`] (R006). Every `invalidate_*` function and the
//!   committed-event mapping reach it through the same identities and the same affected
//!   closure as the trie; an invalidation that never arrives (index lag) is covered by the
//!   index's own use-time liveness check. Its runs ride in an optional v4 snapshot section
//!   written only when a live run exists, so a catalog without runs exports the
//!   byte-identical earlier snapshot.
//! * **Canonical positions only.** Only a request whose positions start at 0 without
//!   explicit ids can hit, and only an entry published at `position_offset` 0 is admitted.
//!
//! # Not covered here
//!
//! Per-layer chain keys beyond the opaque [`KvChainId`], RoPE (doc 36 defect 3) and
//! per-layer pages (defect 5) belong to later phases. Invalidation reaches this catalog through
//! `PtrRuntime::apply_committed` for committed revocations, committed supersessions and
//! committed semantic deltas, never for prepared ones.

use crate::kv_secondary::{SecondaryConfig, SecondaryError, SecondaryIndex};
use crate::PtrRuntime;
use ptr_ledger::{InputInvalidationKind, LedgerEvent};
use ptr_model_api::{
    LineageContribution, PromptSegment, SegmentKind, SegmentSource, SegmentedModelRequest,
    TokenContractError, TokenId, Tokenizer, TokenizerIdentity,
};
use ptr_pods::{
    ExpertCommitRecord, ExpertState, KvBackendError, KvForkAuthority, KvForkWitness, KvPrefixKey,
    KvSharedPrefix, KvWitnessRevoker, PagedKvTensorBackend,
};
use ptr_semdb::{PreparedDelta, SemanticValue};
use ptr_types::{
    AdapterVersion, Digest, Generation, ModelVersion, PodId, PrincipalId, ProjectId, Revision,
    Validity,
};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

/// Tokens per trie chunk; equals the default [`ptr_pods::KvPageSize`].
pub const KV_CHUNK_TOKENS: usize = 16;

/// Metadata bytes charged per trie node; the R005 budget is 1 KiB per node.
const NODE_METADATA_BYTES: u64 = 128;
const ENTRY_METADATA_BYTES: u64 = 256;
const DEPENDENCY_METADATA_BYTES: u64 = 96;

const SALT_DOMAIN: &[u8] = b"ptr.kv.salt.v1\0";
const ROOT_DOMAIN: &[u8] = b"ptr.kv.root.v1\0";
const CHUNK_DOMAIN: &[u8] = b"ptr.kv.chunk.v1\0";
const NODE_DOMAIN: &[u8] = b"ptr.kv.node.v1\0";
const PREFIX_LINEAGE_DOMAIN: &[u8] = b"ptr.kv.prefix-lineage.v2\0";

fn put_text(hasher: &mut Sha256, text: &str) {
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

/// The isolation namespace of a catalog entry. Contains neither the adapter, nor the
/// execution-manifest digest, nor any generation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvSalt([u8; 32]);

impl KvSalt {
    /// `H(principal, project, base-model identity, tokenizer, numerics, schema)`.
    pub fn derive(
        principal: &PrincipalId,
        project: &ProjectId,
        base_model: &ModelVersion,
        tokenizer: &TokenizerIdentity,
        numerics: &str,
        schema: &str,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(SALT_DOMAIN);
        put_text(&mut hasher, &principal.0);
        put_text(&mut hasher, &project.0);
        put_text(&mut hasher, &base_model.0);
        put_text(&mut hasher, tokenizer.name());
        put_text(&mut hasher, tokenizer.version());
        put_text(&mut hasher, numerics);
        put_text(&mut hasher, schema);
        Self(hasher.finalize().into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Identity of the per-layer chain an entry was produced under (doc 36 `chain(l)`). The
/// catalog treats it as opaque: an entry is only visible to a lookup with the same id.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvChainId(pub [u8; 32]);

/// One trie edge: parent node digest, the chunk's token digest and the salt.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvChunkKey {
    pub parent: [u8; 32],
    pub chunk_digest: [u8; 32],
    pub salt: [u8; 32],
}

impl KvChunkKey {
    /// Key of `chunk` below `parent`. `chunk` is hashed as given; callers pass full
    /// 16-token chunks ([`chunk_keys`] does).
    pub fn new(parent: [u8; 32], chunk: &[TokenId], salt: &KvSalt) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CHUNK_DOMAIN);
        hasher.update((chunk.len() as u64).to_le_bytes());
        for token in chunk {
            hasher.update(token.0.to_le_bytes());
        }
        Self {
            parent,
            chunk_digest: hasher.finalize().into(),
            salt: salt.0,
        }
    }

    /// Digest of this node, used as the parent of its children.
    pub fn node_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(NODE_DOMAIN);
        hasher.update(self.parent);
        hasher.update(self.chunk_digest);
        hasher.update(self.salt);
        hasher.finalize().into()
    }

    /// The parent digest of a first-level node under `salt`.
    pub fn root_parent(salt: &KvSalt) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(ROOT_DOMAIN);
        hasher.update(salt.0);
        hasher.finalize().into()
    }
}

/// The chunk-key chain for `tokens`, one key per full 16-token chunk. Computable before
/// any prefill. A trailing partial chunk has no key.
pub fn chunk_keys(salt: &KvSalt, tokens: &[TokenId]) -> Vec<KvChunkKey> {
    let mut parent = KvChunkKey::root_parent(salt);
    tokens
        .as_chunks::<KV_CHUNK_TOKENS>()
        .0
        .iter()
        .map(|chunk| {
            let key = KvChunkKey::new(parent, chunk, salt);
            parent = key.node_digest();
            key
        })
        .collect()
}

/// Capacity bounds of one catalog. Defaults are the R005 values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogLimits {
    pub max_trie_nodes: usize,
    pub max_kv_page_bytes: u64,
    pub max_metadata_bytes: u64,
}

impl Default for CatalogLimits {
    fn default() -> Self {
        Self {
            max_trie_nodes: 4096,
            max_kv_page_bytes: 128 * 1024 * 1024,
            max_metadata_bytes: 8 * 1024 * 1024,
        }
    }
}

/// The bound that refused an insert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCapacityResource {
    TrieNodes,
    KvPageBytes,
    MetadataBytes,
}

/// One entry on its way into the trie.
struct Admission {
    path: Vec<KvChunkKey>,
    entry_key: EntryKey,
    prefix: KvSharedPrefix,
    dependencies: Vec<KvDependency>,
    charge: PageCharge,
    /// `Some` for a per-layer entry ([`KvPrefixCatalog::insert_layers`]).
    layer: Option<u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Eviction {
    Lru,
    Refuse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCatalogError {
    /// Pinned entries leave no evictable capacity; the request proceeds as an uncached
    /// full prefill.
    CapacityExhausted(KvCapacityResource),
    /// The limits contain a zero bound.
    InvalidLimits,
    /// The prefix was not published at position offset 0.
    NonCanonicalPosition,
    /// The prefix is empty, not a whole number of chunks, or longer than the request.
    PrefixShape,
    /// A recorded dependency is not live at insert time (an in-flight invalidation).
    DependencyNotLive,
    /// Another live entry with a different backend prefix already holds this slot.
    DuplicateEntry,
    /// A tombstoned entry is still pinned by an active fork and occupies this slot. It is
    /// reclaimed when its last pin is released; until then the slot is refused rather than
    /// overwritten, so accounting and pins stay symmetric.
    SlotDraining,
    /// The per-page charges do not describe the prefix's pages, or one page digest is
    /// charged with two different byte sizes.
    PageChargeMismatch,
    /// A per-layer insert or lookup names no layer, a layer twice, or per-layer prefixes
    /// of different lengths.
    LayerShape,
}

impl fmt::Display for KvCatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExhausted(resource) => {
                write!(f, "KV prefix catalog capacity exhausted ({resource:?})")
            }
            Self::InvalidLimits => write!(f, "KV prefix catalog limits must be non-zero"),
            Self::NonCanonicalPosition => write!(f, "prefix was not published at position 0"),
            Self::PrefixShape => write!(f, "prefix is not a non-empty run of whole chunks"),
            Self::DependencyNotLive => write!(f, "a prefix dependency is no longer live"),
            Self::DuplicateEntry => write!(f, "a different prefix already holds this entry"),
            Self::SlotDraining => write!(f, "the slot is held by a cancelled, still pinned entry"),
            Self::PageChargeMismatch => write!(f, "page charges do not match the prefix pages"),
            Self::LayerShape => write!(f, "per-layer entries are empty, repeated or ragged"),
        }
    }
}

impl std::error::Error for KvCatalogError {}

/// What kind of generation-bearing input a [`KvDependency`] is.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KvDependencyKind {
    /// A prompt segment of the given kind (never `Raw`).
    Segment(SegmentKind),
    /// A pod-MoE expert whose weights shaped the cached K/V (doc 36 section D, P2c). Its
    /// [`KvDependency::fence_epoch`] is the expert's transition epoch at insert.
    Expert,
    /// The base model whose weights shaped the cached K/V.
    BaseModel,
    /// An adapter applied while the K/V was computed.
    Adapter,
}

/// A generation-bearing input an entry's K/V was computed from.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvDependency {
    pub kind: KvDependencyKind,
    /// The lifecycle target string: a semantic key, memory object id or derived evidence
    /// id, which is also what committed lifecycle events name.
    pub identity: String,
    pub generation: Generation,
    pub content_digest: Digest,
    /// For an [`KvDependencyKind::Expert`], the expert's fence (transition) epoch the
    /// K/V was computed under; `0` means none was recorded. Any other kind carries `0`.
    /// A fence at a higher epoch invalidates the entry
    /// ([`KvPrefixCatalog::invalidate_expert_fences`]); an entry that recorded no epoch is
    /// invalidated by every fence (fail closed).
    pub fence_epoch: u64,
}

impl KvDependency {
    /// A base-model dependency under the lifecycle identity `identity`.
    pub fn base_model(identity: &str, generation: Generation, content_digest: Digest) -> Self {
        Self {
            kind: KvDependencyKind::BaseModel,
            identity: identity.to_owned(),
            generation,
            content_digest,
            fence_epoch: 0,
        }
    }

    /// An adapter dependency under the lifecycle identity `identity`.
    pub fn adapter(identity: &str, generation: Generation, content_digest: Digest) -> Self {
        Self {
            kind: KvDependencyKind::Adapter,
            identity: identity.to_owned(),
            generation,
            content_digest,
            fence_epoch: 0,
        }
    }

    /// An expert dependency that also records the expert's fence epoch.
    pub fn expert_fenced(
        project: &ProjectId,
        pod: &PodId,
        generation: Generation,
        content_digest: Digest,
        fence_epoch: u64,
    ) -> Self {
        Self {
            fence_epoch,
            ..Self::expert(project, pod, generation, content_digest)
        }
    }

    /// An expert dependency, identified as `expert:{project}/{pod}`, the key under which
    /// the materialized state records the expert.
    pub fn expert(
        project: &ProjectId,
        pod: &PodId,
        generation: Generation,
        content_digest: Digest,
    ) -> Self {
        Self {
            kind: KvDependencyKind::Expert,
            identity: expert_identity(project, pod),
            generation,
            content_digest,
            fence_epoch: 0,
        }
    }
}

fn expert_identity(project: &ProjectId, pod: &PodId) -> String {
    ptr_state::expert_key(&project.0, &pod.0)
}

/// Answers whether a dependency is live now, as the committed ledger state sees it.
pub trait KvLivenessOracle {
    fn is_live(&self, dependency: &KvDependency) -> bool;
}

/// Positions of the request's tokens. Only [`KvRequestPositions::Canonical`] can hit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvRequestPositions {
    /// Position ids equal the token index, starting at 0, none supplied explicitly.
    Canonical,
    /// A non-zero start offset.
    Offset(usize),
    /// Explicit position ids.
    Explicit(Vec<usize>),
}

/// Everything a lookup is keyed by.
#[derive(Clone, Copy, Debug)]
pub struct KvLookup<'a> {
    pub salt: &'a KvSalt,
    pub chain: &'a KvChainId,
    pub request: &'a SegmentedModelRequest,
    pub positions: &'a KvRequestPositions,
}

/// A finished prefill offered to the catalog.
#[derive(Debug)]
pub struct KvInsert<'a> {
    pub salt: &'a KvSalt,
    pub chain: &'a KvChainId,
    pub request: &'a SegmentedModelRequest,
    pub prefix: KvSharedPrefix,
    /// Resident bytes of the published pages, charged against `max_kv_page_bytes`.
    pub page_bytes: u64,
    /// Dependencies beyond the request's segments, such as the experts (see
    /// [`KvDependency::expert`]) whose weights shaped the pages. They are checked for
    /// liveness at insert and at every use, and can be invalidated by identity.
    pub extra_dependencies: Vec<KvDependency>,
}

/// Bytes of one backend page, charged once however many entries hold it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPageCharge {
    pub digest: Digest,
    pub bytes: u64,
}

/// A finished prefill offered with per-page byte charges ([`KvPrefixCatalog::insert_paged`]).
#[derive(Debug)]
pub struct KvPagedInsert<'a> {
    pub salt: &'a KvSalt,
    pub chain: &'a KvChainId,
    pub request: &'a SegmentedModelRequest,
    pub prefix: KvSharedPrefix,
    /// One charge per unique digest of `prefix.page_digests`, in first-occurrence order.
    pub pages: Vec<KvPageCharge>,
    pub extra_dependencies: Vec<KvDependency>,
}

/// One layer's slice of a [`KvLayeredInsert`]: the layer's chain `chain(l)`, the prefix the
/// backend published for that layer's pages, the per-page charges and the dependencies the
/// layer's K/V was computed from.
#[derive(Debug)]
pub struct KvLayerEntry {
    pub layer: u16,
    pub chain: KvChainId,
    pub prefix: KvSharedPrefix,
    /// One charge per unique digest of `prefix.page_digests`, in first-occurrence order.
    pub pages: Vec<KvPageCharge>,
    /// The layer's own dependencies beyond the request's segments: the base model on every
    /// layer, the adapter only on the layers it touches (`l >= f`), expert `(j, i)` only on
    /// layers above `j`. The caller states them; the catalog does not infer them.
    pub extra_dependencies: Vec<KvDependency>,
}

/// A finished prefill offered as one entry per layer ([`KvPrefixCatalog::insert_layers`]).
#[derive(Debug)]
pub struct KvLayeredInsert<'a> {
    pub salt: &'a KvSalt,
    pub request: &'a SegmentedModelRequest,
    pub layers: Vec<KvLayerEntry>,
}

/// One layer asked for by [`KvPrefixCatalog::lookup_layers`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvLayerQuery {
    pub layer: u16,
    pub chain: KvChainId,
}

/// A per-layer lookup ([`KvPrefixCatalog::lookup_layers`]).
#[derive(Clone, Copy, Debug)]
pub struct KvLayeredLookup<'a> {
    pub salt: &'a KvSalt,
    pub request: &'a SegmentedModelRequest,
    pub positions: &'a KvRequestPositions,
    pub layers: &'a [KvLayerQuery],
}

/// The outcome of one layer of a [`KvPrefixCatalog::lookup_layers`]: its own deepest live
/// node, or a miss, independent of every other layer.
#[derive(Debug)]
pub struct KvLayerHit {
    pub layer: u16,
    pub hit: Option<KvCatalogHit>,
}

/// How an entry's bytes are charged.
#[derive(Clone, Debug, Eq, PartialEq)]
enum PageCharge {
    /// One figure for the whole entry, never shared.
    Lump(u64),
    /// Per unique page digest, shared with every other entry holding the page.
    Pages(Vec<KvPageCharge>),
}

type EntryKey = (KvChainId, Digest);

const LAYER_LINEAGE_DOMAIN: &[u8] = b"ptr.kv.layer-lineage.v1\0";

/// The lineage slot of an entry key: the prefix lineage itself for a whole-stack entry, and
/// the lineage bound to the layer index for a per-layer entry, so two layers whose chains
/// happened to be equal still never share a slot and a per-layer key never aliases a
/// whole-stack one.
fn entry_lineage(lineage: Digest, layer: Option<u16>) -> Digest {
    let Some(layer) = layer else { return lineage };
    let mut hasher = Sha256::new();
    hasher.update(LAYER_LINEAGE_DOMAIN);
    hasher.update(lineage);
    hasher.update(layer.to_le_bytes());
    hasher.finalize().into()
}

fn unique_page_charges_match(prefix: &KvSharedPrefix, pages: &[KvPageCharge]) -> bool {
    let mut unique: Vec<Digest> = Vec::new();
    for digest in &prefix.page_digests {
        if !unique.contains(digest) {
            unique.push(*digest);
        }
    }
    pages.len() == unique.len()
        && pages
            .iter()
            .zip(&unique)
            .all(|(page, digest)| page.digest == *digest)
}

/// Handle on a pinned entry. Release it with [`KvPrefixCatalog::release`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPin {
    node: KvChunkKey,
    entry: EntryKey,
    id: u64,
    layer: Option<u16>,
}

impl KvPin {
    /// The layer of a per-layer pin ([`KvPrefixCatalog::lookup_layers`]); `None` for a
    /// whole-stack pin.
    pub fn layer(&self) -> Option<u16> {
        self.layer
    }
}

/// A hit: the prefix to fork, the witness that authorises forking it once, how many
/// tokens it covers, and the pin that keeps the entry from eviction until released.
#[derive(Debug)]
pub struct KvCatalogHit {
    witness: KvForkWitness,
    prefix: KvSharedPrefix,
    matched_tokens: usize,
    pin: KvPin,
}

impl KvCatalogHit {
    pub fn prefix(&self) -> &KvSharedPrefix {
        &self.prefix
    }

    /// Tokens the prefix covers.
    pub fn matched_tokens(&self) -> usize {
        self.matched_tokens
    }

    pub fn pin(&self) -> &KvPin {
        &self.pin
    }

    /// The layer this hit is for; `None` for a whole-stack hit.
    pub fn layer(&self) -> Option<u16> {
        self.pin.layer
    }

    /// Split into the witness, the prefix and the pin, for a caller that forks through the
    /// backend directly. The pin must still be released.
    pub fn into_parts(self) -> (KvForkWitness, KvSharedPrefix, KvPin) {
        (self.witness, self.prefix, self.pin)
    }
}

#[derive(Debug)]
pub enum KvForkError {
    /// The entry was invalidated or a dependency stopped being live before the fork.
    Cancelled,
    Backend(KvBackendError),
}

impl fmt::Display for KvForkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "prefix fork cancelled by a dependency invalidation"),
            Self::Backend(error) => write!(f, "backend refused the fork: {error:?}"),
        }
    }
}

impl std::error::Error for KvForkError {}

/// Retires published prefixes on the backend that holds them. Implemented for every
/// [`PagedKvTensorBackend`], so a backend is registered as `Arc::new(backend.clone())`.
pub trait KvPrefixRetirer: Send + Sync {
    fn retire(&self, key: &KvPrefixKey) -> Result<(), KvBackendError>;
}

impl<B: PagedKvTensorBackend> KvPrefixRetirer for B {
    fn retire(&self, key: &KvPrefixKey) -> Result<(), KvBackendError> {
        self.retire_prefix(key)
    }
}

/// Counters, for the R005 coverage prerequisites and for tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvCatalogStats {
    pub nodes: usize,
    pub entries: usize,
    pub tombstoned_pinned_entries: usize,
    pub page_bytes: u64,
    pub metadata_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub capacity_refusals: u64,
    /// Entries admitted by [`KvPrefixCatalog::restore_entries`].
    pub restored_entries: u64,
    /// Entries dropped because the backend no longer holds their prefix at fork time.
    pub backend_missing_rejections: u64,
    /// Forks of per-layer entries the backend refused with `PrefixMismatch`: the entry's key
    /// scope (layer range, effective adapter or schema) disagrees with what its layer needs.
    /// That is a bug or a corrupt entry, never a miss; the entry is tombstoned and this stays
    /// zero on every correct path.
    pub layered_scope_mismatches: u64,
    pub noncanonical_misses: u64,
    pub use_time_rejections: u64,
    pub cancelled_pins: u64,
    pub retire_failures: u64,
}

struct Entry {
    prefix: KvSharedPrefix,
    dependencies: Vec<KvDependency>,
    charge: PageCharge,
    metadata_bytes: u64,
    refcount: u32,
    tombstoned: bool,
    /// Whether this entry still counts in `live_key_refs` for its backend prefix.
    live_ref: bool,
    last_used: u64,
    pins: BTreeMap<u64, KvWitnessRevoker>,
    layer: Option<u16>,
}

struct Node {
    parent: Option<KvChunkKey>,
    children: usize,
    entries: BTreeMap<EntryKey, Entry>,
}

/// Bounded radix catalog of exact prefixes. See the module docs.
pub struct KvPrefixCatalog {
    limits: CatalogLimits,
    authority: KvForkAuthority,
    retirer: Option<Arc<dyn KvPrefixRetirer>>,
    nodes: BTreeMap<KvChunkKey, Node>,
    by_identity: BTreeMap<String, BTreeSet<(KvChunkKey, EntryKey)>>,
    live_key_refs: BTreeMap<KvPrefixKey, usize>,
    /// Unique page digest to (bytes, live entries holding it).
    page_refs: BTreeMap<Digest, (u64, usize)>,
    retire_queue: Vec<KvPrefixKey>,
    page_bytes: u64,
    metadata_bytes: u64,
    clock: u64,
    next_pin: u64,
    stats: KvCatalogStats,
    /// The position-independent secondary index (R006). Empty and inert unless a caller
    /// inserts runs; every invalidation below reaches it.
    secondary: SecondaryIndex,
}

impl fmt::Debug for KvPrefixCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvPrefixCatalog")
            .field("limits", &self.limits)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl KvPrefixCatalog {
    /// A catalog that mints witnesses from `authority`, which the backend hands out once
    /// through [`PagedKvTensorBackend::take_fork_authority`].
    pub fn new(authority: KvForkAuthority, limits: CatalogLimits) -> Result<Self, KvCatalogError> {
        if limits.max_trie_nodes == 0
            || limits.max_kv_page_bytes == 0
            || limits.max_metadata_bytes == 0
        {
            return Err(KvCatalogError::InvalidLimits);
        }
        let secondary = SecondaryIndex::new(SecondaryConfig::default())
            .map_err(|_| KvCatalogError::InvalidLimits)?;
        Ok(Self {
            limits,
            authority,
            retirer: None,
            nodes: BTreeMap::new(),
            by_identity: BTreeMap::new(),
            live_key_refs: BTreeMap::new(),
            page_refs: BTreeMap::new(),
            retire_queue: Vec::new(),
            page_bytes: 0,
            metadata_bytes: 0,
            clock: 0,
            next_pin: 0,
            stats: KvCatalogStats::default(),
            secondary,
        })
    }

    /// Replace the (still empty) secondary index with one built from `config`. The default
    /// configuration is what every catalog starts with, so a catalog that never calls this
    /// behaves as before. Refused with [`KvCatalogError::InvalidLimits`] for an invalid
    /// configuration or when the index already holds runs.
    pub fn with_secondary_config(
        mut self,
        config: SecondaryConfig,
    ) -> Result<Self, KvCatalogError> {
        if !self.secondary.is_empty() {
            return Err(KvCatalogError::InvalidLimits);
        }
        self.secondary = SecondaryIndex::new(config).map_err(|_| KvCatalogError::InvalidLimits)?;
        Ok(self)
    }

    /// The secondary index this catalog owns and invalidates.
    pub fn secondary(&self) -> &SecondaryIndex {
        &self.secondary
    }

    /// Mutable access, for inserting and looking up runs and releasing their pins. The
    /// catalog still reaches the same index from every `invalidate_*` function.
    pub fn secondary_mut(&mut self) -> &mut SecondaryIndex {
        &mut self.secondary
    }

    /// Register the backend on which invalidated prefixes are retired by the runtime
    /// hook ([`PtrRuntime::enable_kv_prefix_catalog`]).
    pub fn with_retirer(mut self, retirer: Arc<dyn KvPrefixRetirer>) -> Self {
        self.retirer = Some(retirer);
        self
    }

    pub fn limits(&self) -> CatalogLimits {
        self.limits
    }

    pub fn stats(&self) -> KvCatalogStats {
        let mut stats = self.stats.clone();
        stats.nodes = self.nodes.len();
        stats.entries = self.nodes.values().map(|node| node.entries.len()).sum();
        stats.tombstoned_pinned_entries = self
            .nodes
            .values()
            .flat_map(|node| node.entries.values())
            .filter(|entry| entry.tombstoned)
            .count();
        stats.page_bytes = self.page_bytes;
        stats.metadata_bytes = self.metadata_bytes;
        stats
    }

    /// Find the deepest live entry for the request's token chain and lineage, re-checking
    /// every recorded dependency with `liveness`. A hit pins the entry and mints a witness.
    pub fn lookup(
        &mut self,
        lookup: KvLookup<'_>,
        liveness: &dyn KvLivenessOracle,
    ) -> Option<KvCatalogHit> {
        if *lookup.positions != KvRequestPositions::Canonical {
            self.stats.noncanonical_misses += 1;
            self.stats.misses += 1;
            return None;
        }
        let tokens = lookup.request.tokens();
        let keys = chunk_keys(lookup.salt, &tokens);
        let hit = self.find_live(&keys, lookup.request, &tokens, lookup.chain, None, liveness);
        if hit.is_none() {
            self.stats.misses += 1;
        }
        hit
    }

    /// The deepest node on `keys` holding a live entry for `(chain, layer)`; a candidate
    /// whose dependencies are no longer live is tombstoned and the walk falls back to its
    /// ancestor. Counts the hit, never the miss.
    fn find_live(
        &mut self,
        keys: &[KvChunkKey],
        request: &SegmentedModelRequest,
        tokens: &[TokenId],
        chain: &KvChainId,
        layer: Option<u16>,
        liveness: &dyn KvLivenessOracle,
    ) -> Option<KvCatalogHit> {
        let mut candidates: Vec<(KvChunkKey, EntryKey, usize)> = Vec::new();
        for (index, key) in keys.iter().enumerate() {
            let Some(node) = self.nodes.get(key) else {
                break;
            };
            if !node
                .entries
                .keys()
                .any(|(entry_chain, _)| entry_chain == chain)
            {
                continue;
            }
            let covered = (index + 1) * KV_CHUNK_TOKENS;
            let (lineage, _) = prefix_lineage(request, tokens, covered);
            let entry_key = (*chain, entry_lineage(lineage, layer));
            if node
                .entries
                .get(&entry_key)
                .is_some_and(|entry| !entry.tombstoned && entry.prefix.sequence_length == covered)
            {
                candidates.push((*key, entry_key, covered));
            }
        }
        // Deepest first; a shallower candidate is the walk back to an admitted ancestor.
        while let Some((node_key, entry_key, covered)) = candidates.pop() {
            let live = self
                .entry(&node_key, &entry_key)
                .map(|entry| entry.dependencies.iter().all(|dep| liveness.is_live(dep)));
            match live {
                Some(true) => return Some(self.pin_hit(node_key, entry_key, covered, layer)),
                Some(false) => {
                    self.stats.use_time_rejections += 1;
                    self.tombstone_entry(&node_key, &entry_key);
                }
                None => {}
            }
        }
        None
    }

    fn pin_hit(
        &mut self,
        node_key: KvChunkKey,
        entry_key: EntryKey,
        covered: usize,
        layer: Option<u16>,
    ) -> KvCatalogHit {
        self.clock += 1;
        self.next_pin += 1;
        let (clock, pin_id) = (self.clock, self.next_pin);
        let entry = self
            .nodes
            .get_mut(&node_key)
            .and_then(|node| node.entries.get_mut(&entry_key))
            .expect("candidate entry exists");
        let (witness, revoker) = self.authority.admit(&entry.prefix);
        entry.refcount += 1;
        entry.last_used = clock;
        entry.pins.insert(pin_id, revoker);
        self.stats.hits += 1;
        KvCatalogHit {
            witness,
            prefix: entry.prefix.clone(),
            matched_tokens: covered,
            pin: KvPin {
                node: node_key,
                entry: entry_key,
                id: pin_id,
                layer,
            },
        }
    }

    /// Fork the hit's prefix into `cache` through the backend's witness-gated path.
    ///
    /// Dependencies are re-checked here, at fork time, as well as at lookup. On any error
    /// the pin is released and the caller falls back to a full prefill. On success the
    /// returned pin must be released when the run ends.
    pub fn fork<B: PagedKvTensorBackend>(
        &mut self,
        backend: &B,
        cache: &mut B::Cache,
        hit: KvCatalogHit,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvPin, KvForkError> {
        let KvCatalogHit {
            witness,
            prefix,
            pin,
            ..
        } = hit;
        let admissible = self.entry(&pin.node, &pin.entry).is_some_and(|entry| {
            !entry.tombstoned && entry.dependencies.iter().all(|dep| liveness.is_live(dep))
        });
        if !admissible {
            self.stats.use_time_rejections += 1;
            self.tombstone_entry(&pin.node, &pin.entry);
            self.release(&pin);
            return Err(KvForkError::Cancelled);
        }
        match backend.fork_prefix_admitted(cache, &prefix, witness) {
            Ok(()) => Ok(pin),
            Err(error) => {
                if matches!(error, KvBackendError::PrefixNotFound) {
                    // A restored (or otherwise stale) entry whose pages are gone: drop it so
                    // the next lookup misses without another failed fork.
                    self.stats.backend_missing_rejections += 1;
                    self.tombstone_entry(&pin.node, &pin.entry);
                }
                self.release(&pin);
                Err(KvForkError::Backend(error))
            }
        }
    }

    /// A backend refused the fork or adoption of a per-layer entry as `PrefixMismatch`. Counted
    /// separately from a clean miss so a scope bug cannot hide as one; the entry is tombstoned
    /// (it can never be served) and the pin released.
    pub fn reject_scope_mismatch_pin(&mut self, pin: &KvPin) {
        self.stats.layered_scope_mismatches += 1;
        self.tombstone_entry(&pin.node, &pin.entry);
        self.release(pin);
    }

    /// Drop the entry behind `pin` because the backend no longer holds its pages (a layered
    /// fork or adoption answered `PrefixNotFound`), the per-layer counterpart of what
    /// [`Self::fork`] does for a whole-stack hit: the entry is tombstoned so the next lookup
    /// misses, and the pin is released.
    pub fn reject_missing_backend_pin(&mut self, pin: &KvPin) {
        self.stats.backend_missing_rejections += 1;
        self.tombstone_entry(&pin.node, &pin.entry);
        self.release(pin);
    }

    /// Whether the entry behind `pin` is still reachable. `false` means a dependency
    /// invalidation cancelled the fork: the run must restart from a fresh prefill and no
    /// result produced after the invalidation may escape.
    pub fn pin_is_live(&self, pin: &KvPin) -> bool {
        self.entry(&pin.node, &pin.entry)
            .is_some_and(|entry| !entry.tombstoned && entry.pins.contains_key(&pin.id))
    }

    /// Whether the entry behind `pin` still exists and still holds the pin, tombstoned or
    /// not. `false` means the entry was evicted or reclaimed under the pin, which the
    /// catalog never does while the pin is held (the session audit checks it).
    pub fn pin_held(&self, pin: &KvPin) -> bool {
        self.entry(&pin.node, &pin.entry)
            .is_some_and(|entry| entry.pins.contains_key(&pin.id))
    }

    /// Drop a pin. A tombstoned entry is reclaimed when its last pin is released.
    /// Returns whether the pin was known.
    pub fn release(&mut self, pin: &KvPin) -> bool {
        let Some(entry) = self
            .nodes
            .get_mut(&pin.node)
            .and_then(|node| node.entries.get_mut(&pin.entry))
        else {
            return false;
        };
        if entry.pins.remove(&pin.id).is_none() {
            return false;
        }
        entry.refcount -= 1;
        if entry.tombstoned && entry.refcount == 0 {
            self.stats.cancelled_pins += 1;
            self.remove_entry(&pin.node, &pin.entry);
        }
        true
    }

    /// Offer a finished prefill, charging `page_bytes` to this one entry (nothing is shared
    /// across entries; prefer [`Self::insert_paged`]). Evicts unreferenced entries in LRU
    /// order to make room and returns [`KvCatalogError::CapacityExhausted`] when only pinned
    /// entries remain, so the bounds are never exceeded. Prefixes made unreferenced by
    /// eviction are queued for [`Self::take_retirements`].
    pub fn insert(
        &mut self,
        insert: KvInsert<'_>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<(), KvCatalogError> {
        let KvInsert {
            salt,
            chain,
            request,
            prefix,
            page_bytes,
            extra_dependencies,
        } = insert;
        self.insert_charged(
            (salt, chain, request),
            prefix,
            PageCharge::Lump(page_bytes),
            extra_dependencies,
            liveness,
        )
    }

    /// Like [`Self::insert`], with per-page charges: each unique page digest is charged
    /// once while any live entry holds it, so shared pages are not counted twice.
    pub fn insert_paged(
        &mut self,
        insert: KvPagedInsert<'_>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<(), KvCatalogError> {
        let KvPagedInsert {
            salt,
            chain,
            request,
            prefix,
            pages,
            extra_dependencies,
        } = insert;
        let mut unique: Vec<Digest> = Vec::new();
        for digest in &prefix.page_digests {
            if !unique.contains(digest) {
                unique.push(*digest);
            }
        }
        if pages.len() != unique.len()
            || pages
                .iter()
                .zip(&unique)
                .any(|(page, digest)| page.digest != *digest)
        {
            return Err(KvCatalogError::PageChargeMismatch);
        }
        self.insert_charged(
            (salt, chain, request),
            prefix,
            PageCharge::Pages(pages),
            extra_dependencies,
            liveness,
        )
    }

    fn insert_charged(
        &mut self,
        (salt, chain, request): (&KvSalt, &KvChainId, &SegmentedModelRequest),
        prefix: KvSharedPrefix,
        charge: PageCharge,
        extra_dependencies: Vec<KvDependency>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<(), KvCatalogError> {
        if prefix.position_offset != 0 {
            return Err(KvCatalogError::NonCanonicalPosition);
        }
        let tokens = request.tokens();
        let covered = prefix.sequence_length;
        if covered == 0 || !covered.is_multiple_of(KV_CHUNK_TOKENS) || covered > tokens.len() {
            return Err(KvCatalogError::PrefixShape);
        }
        let (lineage, mut dependencies) = prefix_lineage(request, &tokens, covered);
        for dependency in extra_dependencies {
            if !dependencies.contains(&dependency) {
                dependencies.push(dependency);
            }
        }
        if !dependencies.iter().all(|dep| liveness.is_live(dep)) {
            return Err(KvCatalogError::DependencyNotLive);
        }
        let path: Vec<KvChunkKey> = chunk_keys(salt, &tokens)
            .into_iter()
            .take(covered / KV_CHUNK_TOKENS)
            .collect();
        self.admit(
            Admission {
                path,
                entry_key: (*chain, lineage),
                prefix,
                dependencies,
                charge,
                layer: None,
            },
            Eviction::Lru,
        )
    }

    /// Bytes that admitting `charge` would add: the whole lump, or the pages no live entry
    /// holds yet. A digest charged with two sizes is refused.
    fn added_page_bytes(&self, charge: &PageCharge) -> Result<u64, KvCatalogError> {
        match charge {
            PageCharge::Lump(bytes) => Ok(*bytes),
            PageCharge::Pages(pages) => {
                let mut added = 0u64;
                let mut seen: BTreeMap<Digest, u64> = BTreeMap::new();
                for page in pages {
                    if seen
                        .insert(page.digest, page.bytes)
                        .is_some_and(|b| b != page.bytes)
                    {
                        return Err(KvCatalogError::PageChargeMismatch);
                    }
                    match self.page_refs.get(&page.digest) {
                        Some((bytes, _)) if *bytes != page.bytes => {
                            return Err(KvCatalogError::PageChargeMismatch)
                        }
                        Some(_) => {}
                        None => added = added.saturating_add(page.bytes),
                    }
                }
                Ok(added)
            }
        }
    }

    fn acquire_pages(&mut self, charge: &PageCharge) {
        match charge {
            PageCharge::Lump(bytes) => self.page_bytes += bytes,
            PageCharge::Pages(pages) => {
                let mut counted: BTreeSet<Digest> = BTreeSet::new();
                for page in pages {
                    if !counted.insert(page.digest) {
                        continue;
                    }
                    let slot = self.page_refs.entry(page.digest).or_insert_with(|| {
                        self.page_bytes += page.bytes;
                        (page.bytes, 0)
                    });
                    slot.1 += 1;
                }
            }
        }
    }

    fn release_pages(&mut self, charge: &PageCharge) {
        match charge {
            PageCharge::Lump(bytes) => self.page_bytes -= bytes,
            PageCharge::Pages(pages) => {
                let mut counted: BTreeSet<Digest> = BTreeSet::new();
                for page in pages {
                    if !counted.insert(page.digest) {
                        continue;
                    }
                    if let Some(slot) = self.page_refs.get_mut(&page.digest) {
                        slot.1 -= 1;
                        if slot.1 == 0 {
                            self.page_bytes -= slot.0;
                            self.page_refs.remove(&page.digest);
                        }
                    }
                }
            }
        }
    }

    /// Admit one entry under the bounds. No liveness check: callers that need one (insert)
    /// make it first; a restore deliberately does not.
    fn admit(&mut self, admission: Admission, eviction: Eviction) -> Result<(), KvCatalogError> {
        let Admission {
            path,
            entry_key,
            prefix,
            dependencies,
            charge,
            layer,
        } = admission;
        let leaf = *path.last().ok_or(KvCatalogError::PrefixShape)?;
        if let Some(existing) = self.entry(&leaf, &entry_key) {
            if existing.tombstoned {
                return Err(KvCatalogError::SlotDraining);
            }
            return if existing.prefix.key == prefix.key {
                Ok(())
            } else {
                Err(KvCatalogError::DuplicateEntry)
            };
        }
        let metadata_bytes = ENTRY_METADATA_BYTES
            + DEPENDENCY_METADATA_BYTES * dependencies.len() as u64
            + dependencies
                .iter()
                .map(|dep| dep.identity.len() as u64)
                .sum::<u64>()
            + 32 * prefix.page_digests.len() as u64
            + match &charge {
                PageCharge::Lump(_) => 0,
                PageCharge::Pages(pages) => 8 * pages.len() as u64,
            };
        loop {
            let missing = path
                .iter()
                .filter(|key| !self.nodes.contains_key(key))
                .count();
            let added = self.added_page_bytes(&charge)?;
            let refused = if self.nodes.len() + missing > self.limits.max_trie_nodes {
                Some(KvCapacityResource::TrieNodes)
            } else if self.page_bytes.saturating_add(added) > self.limits.max_kv_page_bytes {
                Some(KvCapacityResource::KvPageBytes)
            } else if self
                .metadata_bytes
                .saturating_add(metadata_bytes)
                .saturating_add(NODE_METADATA_BYTES * missing as u64)
                > self.limits.max_metadata_bytes
            {
                Some(KvCapacityResource::MetadataBytes)
            } else {
                None
            };
            let Some(resource) = refused else { break };
            if eviction == Eviction::Refuse || !self.evict_lru() {
                self.stats.capacity_refusals += 1;
                return Err(KvCatalogError::CapacityExhausted(resource));
            }
        }
        let mut parent = None;
        for key in &path {
            if !self.nodes.contains_key(key) {
                self.nodes.insert(
                    *key,
                    Node {
                        parent,
                        children: 0,
                        entries: BTreeMap::new(),
                    },
                );
                if let Some(parent_node) = parent.and_then(|p| self.nodes.get_mut(&p)) {
                    parent_node.children += 1;
                }
                self.metadata_bytes += NODE_METADATA_BYTES;
            }
            parent = Some(*key);
        }
        self.clock += 1;
        for dependency in &dependencies {
            self.by_identity
                .entry(dependency.identity.clone())
                .or_default()
                .insert((leaf, entry_key));
        }
        *self.live_key_refs.entry(prefix.key.clone()).or_default() += 1;
        self.acquire_pages(&charge);
        self.metadata_bytes += metadata_bytes;
        let node = self.nodes.get_mut(&leaf).expect("leaf node was created");
        node.entries.insert(
            entry_key,
            Entry {
                prefix,
                dependencies,
                charge,
                metadata_bytes,
                refcount: 0,
                tombstoned: false,
                live_ref: true,
                last_used: self.clock,
                pins: BTreeMap::new(),
                layer,
            },
        );
        Ok(())
    }

    /// Evict the least recently used entry that no fork references. Returns whether one
    /// was evicted; `false` means everything left is pinned.
    pub fn evict_lru(&mut self) -> bool {
        let victim = self
            .nodes
            .iter()
            .flat_map(|(node_key, node)| {
                node.entries
                    .iter()
                    .filter(|(_, entry)| entry.refcount == 0)
                    .map(move |(entry_key, entry)| (entry.last_used, *node_key, *entry_key))
            })
            .min();
        let Some((_, node_key, entry_key)) = victim else {
            return false;
        };
        self.stats.evictions += 1;
        self.queue_retire_if_unreferenced(&node_key, &entry_key);
        self.remove_entry(&node_key, &entry_key);
        true
    }

    /// Tombstone every entry that depends on one of `invalidated` `(identity, generation)`
    /// pairs, revoke their unused witnesses, and return every prefix the backend must now
    /// retire (including queued eviction retirements).
    pub fn invalidate_inputs(
        &mut self,
        invalidated: &BTreeSet<(String, Generation)>,
    ) -> Vec<KvPrefixKey> {
        let identities: BTreeSet<&str> = invalidated.iter().map(|(id, _)| id.as_str()).collect();
        self.tombstone_matching(&identities, |dep| {
            invalidated.contains(&(dep.identity.clone(), dep.generation))
        });
        self.take_retirements()
    }

    /// Like [`Self::invalidate_inputs`] for every generation of the named identities, as
    /// a committed semantic delta changes a key whatever generation a prompt cited.
    pub fn invalidate_identities(&mut self, identities: &BTreeSet<String>) -> Vec<KvPrefixKey> {
        let wanted: BTreeSet<&str> = identities.iter().map(String::as_str).collect();
        self.tombstone_matching(&wanted, |_| true);
        self.take_retirements()
    }

    /// Like [`Self::invalidate_identities`] restricted to dependencies of `kind`.
    pub fn invalidate_kind_identities(
        &mut self,
        kind: KvDependencyKind,
        identities: &BTreeSet<String>,
    ) -> Vec<KvPrefixKey> {
        let wanted: BTreeSet<&str> = identities.iter().map(String::as_str).collect();
        self.tombstone_matching(&wanted, |dep| dep.kind == kind);
        self.take_retirements()
    }

    /// Tombstone every entry with an expert dependency. Used when an expert structural
    /// record cannot be decoded and its operands are therefore unknown (fail closed).
    pub fn invalidate_all_experts(&mut self) -> Vec<KvPrefixKey> {
        let identities: Vec<String> = self
            .by_identity
            .range("expert:".to_owned()..)
            .map(|(identity, _)| identity.clone())
            .take_while(|identity| identity.starts_with("expert:"))
            .collect();
        let wanted: BTreeSet<&str> = identities.iter().map(String::as_str).collect();
        self.tombstone_matching(&wanted, |dep| dep.kind == KvDependencyKind::Expert);
        // The secondary index scans its own `expert:` identity range.
        self.secondary.invalidate_all_experts();
        self.take_retirements()
    }

    /// Like [`Self::invalidate_inputs`] for every generation below the new live one, as a
    /// committed supersession replaces generations.
    pub fn invalidate_superseded(
        &mut self,
        live: &BTreeMap<String, Generation>,
    ) -> Vec<KvPrefixKey> {
        let wanted: BTreeSet<&str> = live.keys().map(String::as_str).collect();
        self.tombstone_matching(&wanted, |dep| {
            live.get(&dep.identity)
                .is_some_and(|generation| dep.generation < *generation)
        });
        self.take_retirements()
    }

    /// Tombstone every expert dependency whose recorded fence epoch is below the epoch now
    /// in force for its identity (an entry that recorded none, `0`, counts as below).
    /// Other kinds and other identities are untouched.
    pub fn invalidate_expert_fences(&mut self, fences: &BTreeMap<String, u64>) -> Vec<KvPrefixKey> {
        let wanted: BTreeSet<&str> = fences.keys().map(String::as_str).collect();
        self.tombstone_matching(&wanted, |dep| {
            dep.kind == KvDependencyKind::Expert
                && fences
                    .get(&dep.identity)
                    .is_some_and(|epoch| dep.fence_epoch < *epoch)
        });
        self.take_retirements()
    }

    /// Apply one committed ledger event to this catalog through the mapping the runtime's own
    /// hook uses (`PtrRuntime::enable_kv_prefix_catalog`): the trie entries and the secondary
    /// index runs the event invalidates are tombstoned, and the backend prefixes that retires
    /// are retired on the registered backend (see [`Self::with_retirer`]). Returns how many
    /// prefix keys were retired; secondary index runs hold copies and are never counted.
    ///
    /// This is the host's hook for a catalog it owns outside a runtime. Only committed events
    /// may be passed, never a prepared or discarded one. An event that invalidates nothing is
    /// a no-op returning `0`. A [`LedgerEvent::SemanticDeltaCommitted`] carries no delta, so it
    /// changes nothing here: apply the affected keys with [`Self::invalidate_identities`]. The
    /// input half of a [`LedgerEvent::InputInvalidated`] is applied like the plain records,
    /// the experts it fenced are not in the event and are applied with
    /// [`Self::invalidate_expert_fences`].
    pub fn apply_committed_event(&mut self, event: &LedgerEvent) -> usize {
        match invalidate_catalog_for_event(self, event, None) {
            Some(retire) => {
                let retired = retire.len();
                retire_keys(self, retire);
                retired
            }
            None => 0,
        }
    }

    /// Tombstone every `kind` dependency below the generation now live for its identity: a
    /// base-model or adapter generation bump. Entries of other kinds are untouched even
    /// when they share the identity string.
    pub fn invalidate_kind_superseded(
        &mut self,
        kind: KvDependencyKind,
        live: &BTreeMap<String, Generation>,
    ) -> Vec<KvPrefixKey> {
        let wanted: BTreeSet<&str> = live.keys().map(String::as_str).collect();
        self.tombstone_matching(&wanted, |dep| {
            dep.kind == kind
                && live
                    .get(&dep.identity)
                    .is_some_and(|generation| dep.generation < *generation)
        });
        self.take_retirements()
    }

    /// Prefixes the backend must retire: those tombstoned, evicted or rejected at use
    /// since the last call. Draining.
    pub fn take_retirements(&mut self) -> Vec<KvPrefixKey> {
        std::mem::take(&mut self.retire_queue)
    }

    fn entry(&self, node: &KvChunkKey, entry: &EntryKey) -> Option<&Entry> {
        self.nodes
            .get(node)
            .and_then(|node| node.entries.get(entry))
    }

    fn tombstone_matching(
        &mut self,
        identities: &BTreeSet<&str>,
        mut affected: impl FnMut(&KvDependency) -> bool,
    ) {
        let mut targets: BTreeSet<(KvChunkKey, EntryKey)> = BTreeSet::new();
        for identity in identities {
            for (node_key, entry_key) in self.by_identity.get(*identity).into_iter().flatten() {
                let hit = self.entry(node_key, entry_key).is_some_and(|entry| {
                    entry
                        .dependencies
                        .iter()
                        .any(|dep| dep.identity == *identity && affected(dep))
                });
                if hit {
                    targets.insert((*node_key, *entry_key));
                }
            }
        }
        for (node_key, entry_key) in targets {
            self.tombstone_entry(&node_key, &entry_key);
        }
        // The secondary index sees the same identities and the same affected closure, so
        // every `invalidate_*` function reaches it. It owns row copies, so there is
        // nothing to retire on the backend.
        self.secondary
            .invalidate_matching(identities, &mut affected);
    }

    /// Make an entry unreachable now; reclaim it now if unpinned, else when released.
    fn tombstone_entry(&mut self, node_key: &KvChunkKey, entry_key: &EntryKey) {
        let Some(entry) = self
            .nodes
            .get_mut(node_key)
            .and_then(|node| node.entries.get_mut(entry_key))
        else {
            return;
        };
        if entry.tombstoned {
            return;
        }
        entry.tombstoned = true;
        for revoker in entry.pins.values() {
            revoker.revoke();
        }
        let refcount = entry.refcount;
        self.queue_retire_if_unreferenced(node_key, entry_key);
        if refcount == 0 {
            self.remove_entry(node_key, entry_key);
        }
    }

    /// Drop the entry's live reference to its backend prefix, once, and queue the prefix
    /// for retirement when no other live entry holds the same one.
    fn queue_retire_if_unreferenced(&mut self, node_key: &KvChunkKey, entry_key: &EntryKey) {
        let Some(entry) = self
            .nodes
            .get_mut(node_key)
            .and_then(|node| node.entries.get_mut(entry_key))
        else {
            return;
        };
        if !std::mem::take(&mut entry.live_ref) {
            return;
        }
        let key = entry.prefix.key.clone();
        if let Some(count) = self.live_key_refs.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                self.live_key_refs.remove(&key);
                self.retire_queue.push(key);
            }
        }
    }

    fn remove_entry(&mut self, node_key: &KvChunkKey, entry_key: &EntryKey) {
        let Some(node) = self.nodes.get_mut(node_key) else {
            return;
        };
        let Some(entry) = node.entries.remove(entry_key) else {
            return;
        };
        self.release_pages(&entry.charge);
        self.metadata_bytes -= entry.metadata_bytes;
        for dependency in &entry.dependencies {
            if let Some(set) = self.by_identity.get_mut(&dependency.identity) {
                set.remove(&(*node_key, *entry_key));
                if set.is_empty() {
                    self.by_identity.remove(&dependency.identity);
                }
            }
        }
        self.prune(*node_key);
    }

    /// Remove nodes that hold no entries and have no children, upwards.
    fn prune(&mut self, mut key: KvChunkKey) {
        loop {
            let Some(node) = self.nodes.get(&key) else {
                return;
            };
            if !node.entries.is_empty() || node.children > 0 {
                return;
            }
            let parent = node.parent;
            self.nodes.remove(&key);
            self.metadata_bytes -= NODE_METADATA_BYTES;
            let Some(parent) = parent else { return };
            if let Some(parent_node) = self.nodes.get_mut(&parent) {
                parent_node.children -= 1;
            }
            key = parent;
        }
    }
}

/// The global SemDB revision is deliberately not hashed (doc 36 section D): a prefix's
/// lineage is per segment, and each bound segment's liveness is checked at use against its
/// own identity and generation. Hashing the revision made every committed request miss,
/// because each request commits its text and so advances it. The domain tag is `v2` so a
/// key from the revision-bound `v1` scheme can never alias a `v2` key.
///
/// Lineage digest of the first `covered` tokens, and the dependencies of that prefix.
///
/// The digest covers the tokenizer and, per segment overlapping the
/// prefix: the kind, for a bound segment its identity, generation and whole-content digest,
/// and the number of tokens covered; for a Raw segment the digest of the covered tokens
/// (content-bound, no identity). Requests that share a Raw prefix and differ afterwards
/// therefore share an entry; requests that cite a different generation do not.
fn prefix_lineage(
    request: &SegmentedModelRequest,
    tokens: &[TokenId],
    covered: usize,
) -> (Digest, Vec<KvDependency>) {
    let mut hasher = Sha256::new();
    hasher.update(PREFIX_LINEAGE_DOMAIN);
    put_text(&mut hasher, request.tokenizer().name());
    put_text(&mut hasher, request.tokenizer().version());
    let mut dependencies: Vec<KvDependency> = Vec::new();
    let mut start = 0usize;
    for segment in request.segments() {
        if start >= covered {
            break;
        }
        let used = segment.tokens().len().min(covered - start);
        match segment.lineage_contribution() {
            LineageContribution::Raw { .. } => {
                hasher.update([0u8]);
                hasher.update((used as u64).to_le_bytes());
                for token in &tokens[start..start + used] {
                    hasher.update(token.0.to_le_bytes());
                }
            }
            LineageContribution::Bound {
                kind,
                identity,
                generation,
                content_digest,
            } => {
                hasher.update([1u8]);
                hasher.update([kind_tag(kind)]);
                put_text(&mut hasher, &identity);
                hasher.update(generation.0.to_le_bytes());
                hasher.update(content_digest);
                hasher.update((used as u64).to_le_bytes());
                let dependency = KvDependency {
                    kind: KvDependencyKind::Segment(kind),
                    identity,
                    generation,
                    content_digest,
                    fence_epoch: 0,
                };
                if !dependencies.contains(&dependency) {
                    dependencies.push(dependency);
                }
            }
        }
        start += segment.tokens().len();
    }
    (hasher.finalize().into(), dependencies)
}

fn kind_tag(kind: SegmentKind) -> u8 {
    match kind {
        SegmentKind::Raw => 0,
        SegmentKind::Semantic => 1,
        SegmentKind::Memory => 2,
        SegmentKind::Derived => 3,
    }
}

impl KvPrefixCatalog {
    /// Offer a finished prefill as one entry per layer. Each chunk node holds up to `L`
    /// entries keyed `(chain(l), layer-bound lineage)`, each with its own dependency set
    /// (the request's segments plus the layer's `extra_dependencies`), so an invalidation
    /// reaches exactly the layers that depend on the input. Page bytes are charged once per
    /// unique page digest across all layers and entries.
    ///
    /// All-or-nothing: every dependency of every layer is checked live first, and if any
    /// layer is refused (capacity, a draining slot, a conflicting entry) the layers of this
    /// call admitted before it are tombstoned again and the error is returned. The layers of
    /// one call never evict one another.
    pub fn insert_layers(
        &mut self,
        insert: KvLayeredInsert<'_>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<(), KvCatalogError> {
        let KvLayeredInsert {
            salt,
            request,
            layers,
        } = insert;
        let Some(first) = layers.first() else {
            return Err(KvCatalogError::LayerShape);
        };
        let covered = first.prefix.sequence_length;
        let mut seen: BTreeSet<u16> = BTreeSet::new();
        for layer in &layers {
            if !seen.insert(layer.layer) || layer.prefix.sequence_length != covered {
                return Err(KvCatalogError::LayerShape);
            }
            if layer.prefix.position_offset != 0 {
                return Err(KvCatalogError::NonCanonicalPosition);
            }
            if !unique_page_charges_match(&layer.prefix, &layer.pages) {
                return Err(KvCatalogError::PageChargeMismatch);
            }
        }
        let tokens = request.tokens();
        if covered == 0 || !covered.is_multiple_of(KV_CHUNK_TOKENS) || covered > tokens.len() {
            return Err(KvCatalogError::PrefixShape);
        }
        let (lineage, segment_dependencies) = prefix_lineage(request, &tokens, covered);
        let path: Vec<KvChunkKey> = chunk_keys(salt, &tokens)
            .into_iter()
            .take(covered / KV_CHUNK_TOKENS)
            .collect();
        let mut admissions = Vec::with_capacity(layers.len());
        for layer in layers {
            let mut dependencies = segment_dependencies.clone();
            for dependency in layer.extra_dependencies {
                if !dependencies.contains(&dependency) {
                    dependencies.push(dependency);
                }
            }
            if !dependencies.iter().all(|dep| liveness.is_live(dep)) {
                return Err(KvCatalogError::DependencyNotLive);
            }
            admissions.push(Admission {
                path: path.clone(),
                entry_key: (layer.chain, entry_lineage(lineage, Some(layer.layer))),
                prefix: layer.prefix,
                dependencies,
                charge: PageCharge::Pages(layer.pages),
                layer: Some(layer.layer),
            });
        }
        let leaf = *path.last().ok_or(KvCatalogError::PrefixShape)?;
        // Entries this call created, each held by one refcount so no later layer of the
        // same call can evict it.
        let mut created: Vec<EntryKey> = Vec::new();
        let mut failure = None;
        for admission in admissions {
            let entry_key = admission.entry_key;
            let existed = self.entry(&leaf, &entry_key).is_some();
            match self.admit(admission, Eviction::Lru) {
                Ok(()) if existed => {}
                Ok(()) => {
                    if let Some(entry) = self
                        .nodes
                        .get_mut(&leaf)
                        .and_then(|node| node.entries.get_mut(&entry_key))
                    {
                        entry.refcount += 1;
                    }
                    created.push(entry_key);
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        for entry_key in &created {
            if let Some(entry) = self
                .nodes
                .get_mut(&leaf)
                .and_then(|node| node.entries.get_mut(entry_key))
            {
                entry.refcount -= 1;
            }
        }
        if let Some(error) = failure {
            for entry_key in &created {
                self.tombstone_entry(&leaf, entry_key);
            }
            return Err(error);
        }
        Ok(())
    }

    /// Per-layer lookup: for each requested layer, the deepest node of the request's chunk
    /// chain whose entry for `(chain(l), lineage)` is live. Layers are independent. A layer
    /// whose entries were tombstoned, whose dependency is no longer live (checked now, per
    /// layer) or which was never inserted misses on its own and does not change another
    /// layer's hit, so the matched depth can differ per layer. Every hit pins its entry and
    /// carries a witness for the layer's prefix.
    pub fn lookup_layers(
        &mut self,
        lookup: KvLayeredLookup<'_>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<Vec<KvLayerHit>, KvCatalogError> {
        let mut seen: BTreeSet<u16> = BTreeSet::new();
        if lookup.layers.is_empty() || !lookup.layers.iter().all(|query| seen.insert(query.layer)) {
            return Err(KvCatalogError::LayerShape);
        }
        if *lookup.positions != KvRequestPositions::Canonical {
            let count = lookup.layers.len() as u64;
            self.stats.noncanonical_misses += count;
            self.stats.misses += count;
            return Ok(lookup
                .layers
                .iter()
                .map(|query| KvLayerHit {
                    layer: query.layer,
                    hit: None,
                })
                .collect());
        }
        let tokens = lookup.request.tokens();
        let keys = chunk_keys(lookup.salt, &tokens);
        let mut hits = Vec::with_capacity(lookup.layers.len());
        for query in lookup.layers {
            let hit = self.find_live(
                &keys,
                lookup.request,
                &tokens,
                &query.chain,
                Some(query.layer),
                liveness,
            );
            if hit.is_none() {
                self.stats.misses += 1;
            }
            hits.push(KvLayerHit {
                layer: query.layer,
                hit,
            });
        }
        Ok(hits)
    }

    /// Re-check, at use, that the entry behind a per-layer (or whole-stack) `pin` is still
    /// reachable and that every dependency of that entry is live now. A dependency that is
    /// dead tombstones this entry, so the pin reads as cancelled ([`Self::pin_is_live`]),
    /// without touching the other layers, which are judged by their own dependencies.
    pub fn revalidate_pin(&mut self, pin: &KvPin, liveness: &dyn KvLivenessOracle) -> bool {
        if !self.pin_is_live(pin) {
            return false;
        }
        let live = self
            .entry(&pin.node, &pin.entry)
            .is_some_and(|entry| entry.dependencies.iter().all(|dep| liveness.is_live(dep)));
        if !live {
            self.stats.use_time_rejections += 1;
            self.tombstone_entry(&pin.node, &pin.entry);
        }
        live
    }
}

// Snapshot -------------------------------------------------------------------------------

const SNAPSHOT_MAGIC: &[u8; 8] = b"PTRKVC01";
const SNAPSHOT_DOMAIN: &[u8] = b"ptr.kv.catalog-snapshot.v1\0";
/// Snapshot v2: v1 plus a layer tag per entry. Written only when the catalog holds a
/// per-layer entry, so a catalog without one still exports the byte-identical v1 snapshot.
/// A v1 snapshot always decodes (every entry is a whole-stack entry).
const SNAPSHOT_MAGIC_V2: &[u8; 8] = b"PTRKVL02";
const SNAPSHOT_DOMAIN_V2: &[u8] = b"ptr.kv.catalog-snapshot.v2\0";
/// Snapshot v3: v2 plus the key's layer scope after each entry's layer tag (`0` = `Full`,
/// `1` + `start` + `end` as little-endian `u16` = a non-empty span). The effective adapter of
/// the key was already part of every version's key encoding. This is what the writer emits
/// whenever the catalog holds a per-layer entry. A v2 snapshot still decodes, but its layered
/// entries have an unknown layer scope and are **skipped** on restore (see
/// [`KvRestoreReport::skipped_unscoped_layered`]): serving one under a guessed `Full` scope
/// would be refused by the backend as a scope mismatch, never as a hit.
const SNAPSHOT_MAGIC_V3: &[u8; 8] = b"PTRKVL03";
const SNAPSHOT_DOMAIN_V3: &[u8] = b"ptr.kv.catalog-snapshot.v3\0";
/// Snapshot v4: the v3 body (layer tag and scope on every entry) followed by one
/// length-prefixed secondary-index section (R006, [`crate::SecondaryIndex::export_section`]).
/// Written **only** when the secondary index holds a live run, so a catalog without runs
/// still exports the byte-identical v1 or v3 snapshot. Restoring a v4 snapshot merges the
/// section's runs without rows; rows come back by an idempotent re-insert whose digest
/// matches.
const SNAPSHOT_MAGIC_V4: &[u8; 8] = b"PTRKVS04";
const SNAPSHOT_DOMAIN_V4: &[u8] = b"ptr.kv.catalog-snapshot.v4\0";
/// Largest snapshot exported or accepted.
pub const KV_SNAPSHOT_MAX_BYTES: usize = 64 * 1024 * 1024;
const SNAPSHOT_MAX_DEPENDENCIES: usize = 1024;
const SNAPSHOT_MAX_PAGES: usize = 1 << 16;
const SNAPSHOT_MAX_TEXT: usize = 4096;

/// Why a catalog snapshot could not be exported or restored.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvSnapshotError {
    /// The snapshot exceeds [`KV_SNAPSHOT_MAX_BYTES`].
    TooLarge,
    /// Restore needs an empty catalog.
    NotEmpty,
    /// Wrong magic or version.
    BadMagic,
    /// The trailing digest does not match the body.
    BadChecksum,
    /// The bytes end inside a field.
    Truncated,
    /// A count or length exceeds its bound.
    LimitExceeded(&'static str),
    /// The bytes decode but are not the canonical encoding (order, duplicates, trailing
    /// bytes, an unknown tag) or an entry is internally inconsistent.
    NonCanonical(&'static str),
    /// The entries do not fit this catalog's limits; nothing was admitted.
    Admission(KvCatalogError),
    /// The v4 secondary section was refused (malformed, over a bound, or over this
    /// index's limits); the catalog stays empty.
    Secondary(SecondaryError),
}

impl fmt::Display for KvSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => write!(f, "KV catalog snapshot is too large"),
            Self::NotEmpty => write!(f, "restore needs an empty KV prefix catalog"),
            Self::BadMagic => write!(f, "not a KV catalog snapshot"),
            Self::BadChecksum => write!(f, "KV catalog snapshot digest mismatch"),
            Self::Truncated => write!(f, "KV catalog snapshot is truncated"),
            Self::LimitExceeded(what) => write!(f, "KV catalog snapshot bound exceeded: {what}"),
            Self::NonCanonical(what) => write!(f, "KV catalog snapshot is not canonical: {what}"),
            Self::Admission(error) => write!(f, "snapshot entries were refused: {error}"),
            Self::Secondary(error) => write!(f, "snapshot secondary section: {error}"),
        }
    }
}

impl std::error::Error for KvSnapshotError {}

/// What a restore admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvRestoreReport {
    /// Entries admitted into the catalog.
    pub entries: usize,
    /// Per-layer entries of a v2 snapshot that were not admitted because that format does not
    /// record the key's layer scope. Zero for a v1 or v3 snapshot.
    pub skipped_unscoped_layered: usize,
    /// Secondary-index runs added from a v4 section (without rows). Zero for any other
    /// version, and for runs the index already held.
    pub secondary_runs: usize,
}

struct SnapshotEntry {
    path: Vec<KvChunkKey>,
    entry_key: EntryKey,
    prefix: KvSharedPrefix,
    dependencies: Vec<KvDependency>,
    charge: PageCharge,
    layer: Option<u16>,
    /// The entry came from a v2 snapshot and its key's layer scope is unknown.
    scope_unknown: bool,
}

fn put_u32(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    put_u32(out, text.len());
    out.extend_from_slice(text.as_bytes());
}

fn dependency_tag(kind: KvDependencyKind) -> (u8, u8) {
    match kind {
        KvDependencyKind::Segment(segment) => (0, kind_tag(segment)),
        KvDependencyKind::Expert => (1, 0),
        KvDependencyKind::BaseModel => (2, 0),
        KvDependencyKind::Adapter => (3, 0),
    }
}

fn dependency_from_tag(tag: u8, segment: u8) -> Option<KvDependencyKind> {
    match (tag, segment) {
        (0, 1) => Some(KvDependencyKind::Segment(SegmentKind::Semantic)),
        (0, 2) => Some(KvDependencyKind::Segment(SegmentKind::Memory)),
        (0, 3) => Some(KvDependencyKind::Segment(SegmentKind::Derived)),
        (1, 0) => Some(KvDependencyKind::Expert),
        (2, 0) => Some(KvDependencyKind::BaseModel),
        (3, 0) => Some(KvDependencyKind::Adapter),
        _ => None,
    }
}

fn encode_snapshot(
    entries: &[SnapshotEntry],
    secondary: Option<&[u8]>,
) -> Result<Vec<u8>, KvSnapshotError> {
    let layered = secondary.is_some() || entries.iter().any(|entry| entry.layer.is_some());
    let mut out = Vec::new();
    out.extend_from_slice(match (secondary, layered) {
        (Some(_), _) => SNAPSHOT_MAGIC_V4,
        (None, true) => SNAPSHOT_MAGIC_V3,
        (None, false) => SNAPSHOT_MAGIC,
    });
    put_u32(&mut out, entries.len());
    for entry in entries {
        put_u32(&mut out, entry.path.len());
        for key in &entry.path {
            out.extend_from_slice(&key.parent);
            out.extend_from_slice(&key.chunk_digest);
            out.extend_from_slice(&key.salt);
        }
        out.extend_from_slice(&entry.entry_key.0 .0);
        out.extend_from_slice(&entry.entry_key.1);
        if layered {
            match entry.layer {
                None => out.push(0),
                Some(layer) => {
                    out.push(1);
                    out.extend_from_slice(&layer.to_le_bytes());
                }
            }
            match entry.prefix.key.layers {
                ptr_pods::KvLayerRange::Full => out.push(0),
                ptr_pods::KvLayerRange::Span { start, end } => {
                    out.push(1);
                    out.extend_from_slice(&start.to_le_bytes());
                    out.extend_from_slice(&end.to_le_bytes());
                }
            }
        }
        let key = &entry.prefix.key;
        put_str(&mut out, &key.principal.0);
        out.extend_from_slice(&key.execution_manifest);
        put_str(&mut out, &key.model.0);
        put_str(&mut out, &key.adapter.0);
        out.extend_from_slice(&key.generation.0.to_le_bytes());
        out.extend_from_slice(&key.schema_digest);
        out.extend_from_slice(&key.prefix_digest);
        out.extend_from_slice(&(entry.prefix.sequence_length as u64).to_le_bytes());
        put_u32(&mut out, entry.prefix.page_digests.len());
        for digest in &entry.prefix.page_digests {
            out.extend_from_slice(digest);
        }
        match &entry.charge {
            PageCharge::Lump(bytes) => {
                out.push(0);
                out.extend_from_slice(&bytes.to_le_bytes());
            }
            PageCharge::Pages(pages) => {
                out.push(1);
                put_u32(&mut out, pages.len());
                for page in pages {
                    out.extend_from_slice(&page.digest);
                    out.extend_from_slice(&page.bytes.to_le_bytes());
                }
            }
        }
        put_u32(&mut out, entry.dependencies.len());
        for dependency in &entry.dependencies {
            let (tag, segment) = dependency_tag(dependency.kind);
            out.push(tag);
            out.push(segment);
            put_str(&mut out, &dependency.identity);
            out.extend_from_slice(&dependency.generation.0.to_le_bytes());
            out.extend_from_slice(&dependency.content_digest);
            out.extend_from_slice(&dependency.fence_epoch.to_le_bytes());
        }
        if out.len() > KV_SNAPSHOT_MAX_BYTES {
            return Err(KvSnapshotError::TooLarge);
        }
    }
    if let Some(section) = secondary {
        put_u32(&mut out, section.len());
        out.extend_from_slice(section);
    }
    let mut hasher = Sha256::new();
    hasher.update(match (secondary, layered) {
        (Some(_), _) => SNAPSHOT_DOMAIN_V4,
        (None, true) => SNAPSHOT_DOMAIN_V3,
        (None, false) => SNAPSHOT_DOMAIN,
    });
    hasher.update(&out);
    let trailer: [u8; 32] = hasher.finalize().into();
    out.extend_from_slice(&trailer);
    if out.len() > KV_SNAPSHOT_MAX_BYTES {
        return Err(KvSnapshotError::TooLarge);
    }
    Ok(out)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], KvSnapshotError> {
        let end = self.at.checked_add(len).ok_or(KvSnapshotError::Truncated)?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or(KvSnapshotError::Truncated)?;
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, KvSnapshotError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<usize, KvSnapshotError> {
        let raw: [u8; 4] = self.take(4)?.try_into().expect("four bytes");
        Ok(u32::from_le_bytes(raw) as usize)
    }

    fn u64(&mut self) -> Result<u64, KvSnapshotError> {
        let raw: [u8; 8] = self.take(8)?.try_into().expect("eight bytes");
        Ok(u64::from_le_bytes(raw))
    }

    fn digest(&mut self) -> Result<[u8; 32], KvSnapshotError> {
        Ok(self.take(32)?.try_into().expect("32 bytes"))
    }

    fn text(&mut self) -> Result<String, KvSnapshotError> {
        let len = self.u32()?;
        if len > SNAPSHOT_MAX_TEXT {
            return Err(KvSnapshotError::LimitExceeded("text length"));
        }
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| KvSnapshotError::NonCanonical("text is not UTF-8"))
    }
}

/// A decoded snapshot: the trie entries and, for v4, the raw secondary section.
struct DecodedSnapshot {
    entries: Vec<SnapshotEntry>,
    secondary: Option<Vec<u8>>,
}

fn decode_snapshot(
    bytes: &[u8],
    limits: &CatalogLimits,
) -> Result<DecodedSnapshot, KvSnapshotError> {
    if bytes.len() > KV_SNAPSHOT_MAX_BYTES {
        return Err(KvSnapshotError::TooLarge);
    }
    if bytes.len() < SNAPSHOT_MAGIC.len() + 4 + 32 {
        return Err(if bytes.starts_with(SNAPSHOT_MAGIC) || bytes.len() >= 8 {
            KvSnapshotError::Truncated
        } else {
            KvSnapshotError::BadMagic
        });
    }
    let (layered, scoped, sectioned) = if bytes.starts_with(SNAPSHOT_MAGIC) {
        (false, false, false)
    } else if bytes.starts_with(SNAPSHOT_MAGIC_V2) {
        (true, false, false)
    } else if bytes.starts_with(SNAPSHOT_MAGIC_V3) {
        (true, true, false)
    } else if bytes.starts_with(SNAPSHOT_MAGIC_V4) {
        (true, true, true)
    } else {
        return Err(KvSnapshotError::BadMagic);
    };
    let (body, trailer) = bytes.split_at(bytes.len() - 32);
    let mut hasher = Sha256::new();
    hasher.update(match (layered, scoped, sectioned) {
        (false, _, _) => SNAPSHOT_DOMAIN,
        (true, false, _) => SNAPSHOT_DOMAIN_V2,
        (true, true, false) => SNAPSHOT_DOMAIN_V3,
        (true, true, true) => SNAPSHOT_DOMAIN_V4,
    });
    hasher.update(body);
    let expected: [u8; 32] = hasher.finalize().into();
    if expected != trailer {
        return Err(KvSnapshotError::BadChecksum);
    }
    let mut reader = Reader {
        bytes: body,
        at: SNAPSHOT_MAGIC.len(),
    };
    let count = reader.u32()?;
    if count > limits.max_trie_nodes {
        return Err(KvSnapshotError::LimitExceeded("entry count"));
    }
    let mut entries: Vec<SnapshotEntry> = Vec::new();
    for _ in 0..count {
        let depth = reader.u32()?;
        if depth == 0 {
            return Err(KvSnapshotError::NonCanonical("empty path"));
        }
        if depth > limits.max_trie_nodes {
            return Err(KvSnapshotError::LimitExceeded("path depth"));
        }
        let mut path = Vec::new();
        for _ in 0..depth {
            path.push(KvChunkKey {
                parent: reader.digest()?,
                chunk_digest: reader.digest()?,
                salt: reader.digest()?,
            });
        }
        let salt = KvSalt(path[0].salt);
        let mut parent = KvChunkKey::root_parent(&salt);
        for key in &path {
            if key.salt != salt.0 || key.parent != parent {
                return Err(KvSnapshotError::NonCanonical("path is not a chain"));
            }
            parent = key.node_digest();
        }
        let chain = KvChainId(reader.digest()?);
        let lineage = reader.digest()?;
        let layer = if layered {
            match reader.u8()? {
                0 => None,
                1 => {
                    let raw: [u8; 2] = reader.take(2)?.try_into().expect("two bytes");
                    Some(u16::from_le_bytes(raw))
                }
                _ => return Err(KvSnapshotError::NonCanonical("unknown layer tag")),
            }
        } else {
            None
        };
        let layers = if scoped {
            match reader.u8()? {
                0 => ptr_pods::KvLayerRange::Full,
                1 => {
                    let start = u16::from_le_bytes(reader.take(2)?.try_into().expect("two bytes"));
                    let end = u16::from_le_bytes(reader.take(2)?.try_into().expect("two bytes"));
                    if start >= end {
                        return Err(KvSnapshotError::NonCanonical("empty layer span"));
                    }
                    ptr_pods::KvLayerRange::Span { start, end }
                }
                _ => return Err(KvSnapshotError::NonCanonical("unknown layer scope tag")),
            }
        } else {
            ptr_pods::KvLayerRange::Full
        };
        if layer.is_none() && layers != ptr_pods::KvLayerRange::Full {
            return Err(KvSnapshotError::NonCanonical(
                "whole-stack entry has a layer span",
            ));
        }
        let key = KvPrefixKey {
            principal: PrincipalId::from(reader.text()?.as_str()),
            execution_manifest: reader.digest()?,
            model: ModelVersion::from(reader.text()?.as_str()),
            adapter: AdapterVersion::from(reader.text()?.as_str()),
            generation: Generation(reader.u64()?),
            schema_digest: reader.digest()?,
            prefix_digest: reader.digest()?,
            layers,
        };
        let sequence_length = reader.u64()?;
        if sequence_length != (depth * KV_CHUNK_TOKENS) as u64 {
            return Err(KvSnapshotError::NonCanonical("length does not match path"));
        }
        let page_count = reader.u32()?;
        if page_count > SNAPSHOT_MAX_PAGES {
            return Err(KvSnapshotError::LimitExceeded("page digests"));
        }
        let mut page_digests = Vec::new();
        for _ in 0..page_count {
            page_digests.push(reader.digest()?);
        }
        let charge = match reader.u8()? {
            0 => PageCharge::Lump(reader.u64()?),
            1 => {
                let charged = reader.u32()?;
                if charged > SNAPSHOT_MAX_PAGES {
                    return Err(KvSnapshotError::LimitExceeded("page charges"));
                }
                let mut pages = Vec::new();
                for _ in 0..charged {
                    pages.push(KvPageCharge {
                        digest: reader.digest()?,
                        bytes: reader.u64()?,
                    });
                }
                let mut unique: Vec<Digest> = Vec::new();
                for digest in &page_digests {
                    if !unique.contains(digest) {
                        unique.push(*digest);
                    }
                }
                if pages.len() != unique.len()
                    || pages
                        .iter()
                        .zip(&unique)
                        .any(|(page, digest)| page.digest != *digest)
                {
                    return Err(KvSnapshotError::NonCanonical("charges do not match pages"));
                }
                PageCharge::Pages(pages)
            }
            _ => return Err(KvSnapshotError::NonCanonical("unknown charge tag")),
        };
        let dependency_count = reader.u32()?;
        if dependency_count > SNAPSHOT_MAX_DEPENDENCIES {
            return Err(KvSnapshotError::LimitExceeded("dependencies"));
        }
        let mut dependencies: Vec<KvDependency> = Vec::new();
        for _ in 0..dependency_count {
            let (tag, segment) = (reader.u8()?, reader.u8()?);
            let kind = dependency_from_tag(tag, segment)
                .ok_or(KvSnapshotError::NonCanonical("unknown dependency kind"))?;
            let dependency = KvDependency {
                kind,
                identity: reader.text()?,
                generation: Generation(reader.u64()?),
                content_digest: reader.digest()?,
                fence_epoch: reader.u64()?,
            };
            if dependencies.last().is_some_and(|last| *last >= dependency) {
                return Err(KvSnapshotError::NonCanonical(
                    "dependencies not strictly sorted",
                ));
            }
            dependencies.push(dependency);
        }
        let entry = SnapshotEntry {
            path,
            entry_key: (chain, lineage),
            prefix: KvSharedPrefix {
                key,
                sequence_length: sequence_length as usize,
                position_offset: 0,
                page_digests,
            },
            dependencies,
            charge,
            layer,
            scope_unknown: layer.is_some() && !scoped,
        };
        if entries.last().is_some_and(|last| {
            (last.path.last(), last.entry_key) >= (entry.path.last(), entry.entry_key)
        }) {
            return Err(KvSnapshotError::NonCanonical("entries not strictly sorted"));
        }
        entries.push(entry);
    }
    let secondary = if sectioned {
        let len = reader.u32()?;
        if len == 0 {
            return Err(KvSnapshotError::NonCanonical("empty secondary section"));
        }
        if len > crate::kv_secondary::SECONDARY_SECTION_MAX_BYTES {
            return Err(KvSnapshotError::LimitExceeded("secondary section"));
        }
        Some(reader.take(len)?.to_vec())
    } else {
        None
    };
    if reader.at != body.len() {
        return Err(KvSnapshotError::NonCanonical("trailing bytes"));
    }
    Ok(DecodedSnapshot { entries, secondary })
}

impl KvPrefixCatalog {
    /// A canonical, bounded snapshot of the live entries: trie paths, lineage, backend
    /// prefix keys, dependencies and byte charges, never K/V pages. Tombstoned entries are
    /// not exported. The same catalog state always yields the same bytes. The trailing
    /// digest detects corruption; it is not a signature, so the snapshot belongs in a
    /// store the runtime trusts. Recency is not exported: a restored catalog orders its
    /// entries by key for LRU purposes.
    pub fn export_entries(&self) -> Result<Vec<u8>, KvSnapshotError> {
        let mut entries = Vec::new();
        for (node_key, node) in &self.nodes {
            for (entry_key, entry) in &node.entries {
                if entry.tombstoned {
                    continue;
                }
                let mut path = vec![*node_key];
                let mut at = node.parent;
                while let Some(parent) = at {
                    path.push(parent);
                    at = self.nodes.get(&parent).and_then(|node| node.parent);
                }
                path.reverse();
                let mut dependencies = entry.dependencies.clone();
                dependencies.sort();
                dependencies.dedup();
                entries.push(SnapshotEntry {
                    path,
                    entry_key: *entry_key,
                    prefix: entry.prefix.clone(),
                    dependencies,
                    charge: entry.charge.clone(),
                    layer: entry.layer,
                    scope_unknown: false,
                });
            }
        }
        encode_snapshot(&entries, self.secondary.export_section().as_deref())
    }

    /// Admit the entries of an [`Self::export_entries`] snapshot into this empty catalog.
    ///
    /// The snapshot is decoded strictly and re-checked against this catalog's limits
    /// before anything is admitted; on any error the catalog stays empty. Entries are
    /// admitted **without a liveness check**: an input revoked or superseded while the
    /// snapshot was at rest is rejected at the first lookup or fork by the use-time check
    /// (doc 36), and an entry whose backend prefix is gone is dropped when its fork finds
    /// the prefix missing. Both are clean misses.
    pub fn restore_entries(&mut self, snapshot: &[u8]) -> Result<KvRestoreReport, KvSnapshotError> {
        if !self.nodes.is_empty() {
            return Err(KvSnapshotError::NotEmpty);
        }
        let DecodedSnapshot { entries, secondary } = decode_snapshot(snapshot, &self.limits)?;
        // The secondary section is decoded and checked against its bounds BEFORE any trie
        // entry is admitted, so a malformed section leaves the catalog and every counter
        // exactly as they were.
        let plan = match secondary {
            Some(section) => Some(
                self.secondary
                    .prepare_restore(&section)
                    .map_err(KvSnapshotError::Secondary)?,
            ),
            None => None,
        };
        let stats_before = self.stats.clone();
        let clock_before = self.clock;
        let next_pin_before = self.next_pin;
        let mut count = 0usize;
        let mut skipped = 0usize;
        for entry in entries {
            if entry.scope_unknown {
                skipped += 1;
                continue;
            }
            count += 1;
            let admitted = self.admit(
                Admission {
                    path: entry.path,
                    entry_key: entry.entry_key,
                    prefix: entry.prefix,
                    dependencies: entry.dependencies,
                    charge: entry.charge,
                    layer: entry.layer,
                },
                Eviction::Refuse,
            );
            if let Err(error) = admitted {
                self.clear_trie();
                self.stats = stats_before;
                self.clock = clock_before;
                self.next_pin = next_pin_before;
                return Err(KvSnapshotError::Admission(error));
            }
        }
        // The prepared section cannot fail to apply: the index has not changed since.
        let secondary_runs = plan.map_or(0, |plan| self.secondary.commit_restore(plan));
        self.stats.restored_entries += count as u64;
        Ok(KvRestoreReport {
            entries: count,
            skipped_unscoped_layered: skipped,
            secondary_runs,
        })
    }

    fn clear_trie(&mut self) {
        self.nodes.clear();
        self.by_identity.clear();
        self.live_key_refs.clear();
        self.page_refs.clear();
        self.retire_queue.clear();
        self.page_bytes = 0;
        self.metadata_bytes = 0;
    }
}

// Runtime wiring -------------------------------------------------------------------------

/// Most keys whose last-edit revision [`SemanticEdits`] tracks individually. Past it the
/// per-key history is dropped and the floor rises to the current revision (fail closed).
pub(crate) const MAX_TRACKED_SEMANTIC_KEYS: usize = 1 << 16;

/// Per semantic key, the revision of its last committed edit and the revision at which a
/// committed revocation named it.
///
/// A semantic input has no lifecycle generation, so its segment's `generation` is read as
/// the committed semantic revision at which the cited content was read. The segment is
/// current iff no committed delta has affected the key after that revision (the written
/// keys and the keys derived from them, the same set the catalog invalidates) and the key
/// is not revoked.
///
/// * **Revocation is a floor, not a pair.** A committed `Revoked { subject: K, .. }` for a
///   key the snapshot holds kills every citation of `K` at every generation, not only the
///   cited pair: a caller could otherwise re-cite the same content at another generation
///   inside `[last_edit, current]`. The key stays dead until a later committed edit of
///   `K` writes a new value; a citation at or after that edit names the new value, which
///   the revocation did not cover. (The exact revoked `(K, g)` pair stays dead for ever.)
/// * **Bounded.** An entry for a key the post-apply snapshot no longer holds is pruned (a
///   re-creation records a fresh edit, and the revocation of a key that does not exist
///   cannot outlive its absence). More than [`MAX_TRACKED_SEMANTIC_KEYS`] edited keys
///   drop the per-key history and raise the floor to the current revision. Revocation
///   markers are only kept for keys the snapshot holds and are never dropped by the cap.
/// * **Restore.** After a compacted restore the history is unknown: every key is only
///   known to be unchanged since the snapshot's revision, and every key with a recorded
///   revocation is dead until it is edited again. Replay above the floor re-derives the
///   same answers as the live runtime through the same record calls.
#[derive(Debug)]
pub(crate) struct SemanticEdits {
    floor: Revision,
    edited: BTreeMap<String, Revision>,
    revoked: BTreeMap<String, Revision>,
    cap: usize,
}

impl Default for SemanticEdits {
    fn default() -> Self {
        Self::with_cap(Revision::default(), MAX_TRACKED_SEMANTIC_KEYS)
    }
}

impl SemanticEdits {
    fn with_cap(floor: Revision, cap: usize) -> Self {
        Self {
            floor,
            edited: BTreeMap::new(),
            revoked: BTreeMap::new(),
            cap,
        }
    }

    /// The state of a runtime restored at `floor` whose lifecycle holds revocations for
    /// `revoked` (the keys among them that the snapshot holds are dead until edited).
    pub(crate) fn restored_at(floor: Revision, revoked: impl IntoIterator<Item = String>) -> Self {
        let mut edits = Self::with_cap(floor, MAX_TRACKED_SEMANTIC_KEYS);
        edits.revoked = revoked.into_iter().map(|key| (key, floor)).collect();
        edits
    }

    /// Record a committed delta at `revision`. `held` says whether the post-apply
    /// snapshot holds a key.
    pub(crate) fn record(
        &mut self,
        affected: &BTreeSet<String>,
        held: impl Fn(&str) -> bool,
        revision: Revision,
    ) {
        for key in affected {
            if held(key) {
                self.edited.insert(key.clone(), revision);
            } else {
                self.edited.remove(key);
                self.revoked.remove(key);
            }
        }
        if self.edited.len() > self.cap {
            self.edited.clear();
            self.floor = revision;
        }
    }

    /// Record a committed revocation naming `key` at `revision`; a key the snapshot does
    /// not hold records nothing (it has no content to re-cite and a later creation is a
    /// fresh edit).
    pub(crate) fn record_revocation(&mut self, key: &str, held: bool, revision: Revision) {
        if held {
            let at = self.revoked.entry(key.to_owned()).or_insert(revision);
            *at = (*at).max(revision);
        }
    }

    fn last_edit(&self, key: &str) -> Revision {
        self.edited
            .get(key)
            .copied()
            .unwrap_or(self.floor)
            .max(self.floor)
    }

    /// Whether the key was revoked and no edit of it was committed since.
    fn is_revoked(&self, key: &str) -> bool {
        self.revoked
            .get(key)
            .is_some_and(|at| self.edited.get(key).is_none_or(|edit| edit <= at))
    }

    fn is_current(&self, key: &str, read_at: Revision, current: Revision) -> bool {
        !self.is_revoked(key) && self.last_edit(key) <= read_at && read_at <= current
    }
}

impl KvLivenessOracle for PtrRuntime {
    /// A semantic key carries no lifecycle generation and the lifecycle tables are never
    /// consulted for it (a key that happens to be named like a capsule, constraint or
    /// procedure identity must not borrow that identity's liveness): it is live iff the
    /// committed snapshot holds the key, no committed revocation floor covers it (see
    /// `SemanticEdits`), no tombstone names the exact pair, and the segment's
    /// generation (the committed semantic revision its content was read at) is not older
    /// than the key's last committed edit nor newer than the current revision. So a
    /// committed edit or revocation supersedes every entry built on the earlier content
    /// even when the catalog's invalidation was delayed, lost or not restored.
    ///
    /// Every other dependency is live iff the lifecycle authority knows `identity` at
    /// `generation` and it is neither revoked nor superseded.
    ///
    /// The segment's token-level `content_digest` cannot be compared with the committed
    /// value here: it is a digest of the tokenizer's ids for the rendered text, and the
    /// runtime holds neither the tokenizer nor the rendering. It stays part of the
    /// entry's lineage, so different content never matches an entry. A generic
    /// `PromptSegment::from_text` Semantic segment is therefore caller-asserted; a
    /// production path builds it with [`PtrRuntime::semantic_prompt_segment`].
    fn is_live(&self, dependency: &KvDependency) -> bool {
        match dependency.kind {
            KvDependencyKind::Expert => self.expert_is_serving(
                &dependency.identity,
                dependency.generation,
                dependency.fence_epoch,
            ),
            KvDependencyKind::Segment(SegmentKind::Semantic) => {
                !self.revoked_generations().any(|(id, generation)| {
                    id == dependency.identity && generation == dependency.generation
                }) && self.snapshot().value(&dependency.identity).is_some()
                    && self.semantic_segment_is_current(&dependency.identity, dependency.generation)
            }
            _ => matches!(
                self.generation_validity(&dependency.identity, dependency.generation),
                Some(Validity::Live)
            ),
        }
    }
}

/// Why [`PtrRuntime::semantic_prompt_segment`] built no segment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticSegmentError {
    /// The committed snapshot holds no value at the key.
    MissingKey,
    /// The key is revoked and not edited since.
    Revoked,
    /// The value is a typed payload, which has no canonical text rendering.
    NotText,
    /// The token contract refused the segment (an empty value, an empty key).
    Contract(TokenContractError),
}

impl fmt::Display for SemanticSegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKey => write!(f, "the committed snapshot holds no value at the key"),
            Self::Revoked => write!(f, "the semantic key is revoked and not edited since"),
            Self::NotText => write!(f, "the value is a payload and has no canonical rendering"),
            Self::Contract(error) => write!(f, "token contract: {error}"),
        }
    }
}

impl std::error::Error for SemanticSegmentError {}

impl PtrRuntime {
    /// Whether content of `key` read at semantic revision `generation` is unchanged since.
    fn semantic_segment_is_current(&self, key: &str, generation: Generation) -> bool {
        self.semantic_edits
            .is_current(key, Revision(generation.0), self.revision())
    }

    /// Build the Semantic prompt segment for `key` from the committed snapshot, the only
    /// path whose content the runtime vouches for (a `PromptSegment::from_text` Semantic
    /// segment is caller-asserted: the runtime cannot compare its tokens with the value).
    ///
    /// The canonical rendering is the committed value's text, byte for byte, with no key,
    /// framing or normalisation; a typed payload has none and is refused. It is encoded
    /// with `tokenizer`. The segment's generation is the revision of the key's last
    /// committed edit, the revision the current content was read at: it is live now, it
    /// does not change under unrelated commits (so the same content keeps one lineage
    /// instead of fragmenting per cited generation) and it is dead after the next edit.
    pub fn semantic_prompt_segment(
        &self,
        tokenizer: &dyn Tokenizer,
        key: &str,
    ) -> Result<PromptSegment, SemanticSegmentError> {
        let snapshot = self.snapshot();
        let text = match snapshot.value(key) {
            None => return Err(SemanticSegmentError::MissingKey),
            Some(SemanticValue::Text(text)) => text,
            Some(SemanticValue::Payload(_)) => return Err(SemanticSegmentError::NotText),
        };
        if self.semantic_edits.is_revoked(key) {
            return Err(SemanticSegmentError::Revoked);
        }
        let generation = Generation(self.semantic_edits.last_edit(key).0);
        PromptSegment::from_text(
            tokenizer,
            SegmentSource::Semantic {
                key: key.to_owned(),
                generation,
            },
            text,
        )
        .map_err(SemanticSegmentError::Contract)
    }

    /// Whether the materialized `expert:{project}/{pod}` value, written as
    /// `{generation}:{state}:{state_version}`, names `generation` in a serving or frozen
    /// state. An absent, malformed or other-state value is not live (fail closed). An expert
    /// that was ever fenced is live only for a dependency recorded at an epoch above its highest
    /// fence (never `0`, never an epoch at or below it), also after a readmission.
    fn expert_is_serving(&self, identity: &str, generation: Generation, fence_epoch: u64) -> bool {
        self.materialized_state()
            .values
            .get(identity)
            .is_some_and(|value| expert_value_is_serving(value, generation, fence_epoch))
            && self
                .experts
                .highest_fenced_epoch(identity)
                .is_none_or(|fenced| fence_epoch > fenced)
    }
}

/// With a recorded `fence_epoch` (non-zero) the value's `state_version` must be exactly it:
/// any later transition, a fence included, ends the dependency.
fn expert_value_is_serving(value: &str, generation: Generation, fence_epoch: u64) -> bool {
    let mut parts = value.splitn(3, ':');
    let (Some(live), Some(state)) = (parts.next(), parts.next()) else {
        return false;
    };
    if fence_epoch != 0
        && !parts
            .next()
            .and_then(|version| version.parse::<u64>().ok())
            .is_some_and(|version| version == fence_epoch)
    {
        return false;
    }
    // The names come from the one mapping the materialized projection writes with.
    live.parse::<u64>().is_ok_and(|live| live == generation.0)
        && (state == ptr_state::expert_state_name(ExpertState::Serving)
            || state == ptr_state::expert_state_name(ExpertState::Frozen))
}

/// The `expert:{project}/{pod}` identity of every pod an encoded expert commit record
/// names, as an existing source or an intended result. `None` when the bytes do not decode
/// as a canonical [`ExpertCommitRecord`].
pub fn expert_operand_identities(record: &[u8]) -> Option<Vec<String>> {
    let record = ExpertCommitRecord::decode_canonical(record).ok()?;
    let op = record.proposal().op();
    let mut identities: Vec<String> = op
        .sources()
        .into_iter()
        .map(|target| expert_identity(target.project(), target.pod()))
        .chain(
            op.results()
                .into_iter()
                .map(|result| expert_identity(result.project(), result.pod())),
        )
        .collect();
    identities.sort();
    identities.dedup();
    Some(identities)
}

/// The reason a catalog could not be enabled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvCatalogWiringError {
    AlreadyEnabled,
    NotEnabled,
}

impl fmt::Display for KvCatalogWiringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyEnabled => write!(f, "a KV prefix catalog is already enabled"),
            Self::NotEnabled => write!(f, "no KV prefix catalog is enabled"),
        }
    }
}

impl std::error::Error for KvCatalogWiringError {}

impl PtrRuntime {
    /// Install the prefix catalog. Until this is called the runtime keeps none and every
    /// invalidation hook is a no-op. Exactly one catalog can be installed.
    pub fn enable_kv_prefix_catalog(
        &mut self,
        catalog: KvPrefixCatalog,
    ) -> Result<(), KvCatalogWiringError> {
        if self.kv_catalog.is_some() {
            return Err(KvCatalogWiringError::AlreadyEnabled);
        }
        self.kv_catalog = Some(catalog);
        Ok(())
    }

    pub fn kv_prefix_catalog(&self) -> Option<&KvPrefixCatalog> {
        self.kv_catalog.as_ref()
    }

    /// Look up under this runtime's committed lifecycle state.
    pub fn kv_prefix_lookup(
        &mut self,
        lookup: KvLookup<'_>,
    ) -> Result<Option<KvCatalogHit>, KvCatalogWiringError> {
        let mut catalog = self
            .kv_catalog
            .take()
            .ok_or(KvCatalogWiringError::NotEnabled)?;
        let hit = catalog.lookup(lookup, &*self);
        retire_queued(&mut catalog);
        self.kv_catalog = Some(catalog);
        Ok(hit)
    }

    /// Fork a hit under this runtime's committed lifecycle state.
    pub fn kv_prefix_fork<B: PagedKvTensorBackend>(
        &mut self,
        backend: &B,
        cache: &mut B::Cache,
        hit: KvCatalogHit,
    ) -> Result<Result<KvPin, KvForkError>, KvCatalogWiringError> {
        let mut catalog = self
            .kv_catalog
            .take()
            .ok_or(KvCatalogWiringError::NotEnabled)?;
        let forked = catalog.fork(backend, cache, hit, &*self);
        retire_queued(&mut catalog);
        self.kv_catalog = Some(catalog);
        Ok(forked)
    }

    /// Insert under this runtime's committed lifecycle state, then retire whatever the
    /// insert evicted on the registered backend.
    pub fn kv_prefix_insert(
        &mut self,
        insert: KvInsert<'_>,
    ) -> Result<Result<(), KvCatalogError>, KvCatalogWiringError> {
        let mut catalog = self
            .kv_catalog
            .take()
            .ok_or(KvCatalogWiringError::NotEnabled)?;
        let inserted = catalog.insert(insert, &*self);
        retire_queued(&mut catalog);
        self.kv_catalog = Some(catalog);
        Ok(inserted)
    }

    /// Release a pin taken by a hit.
    pub fn kv_prefix_release(&mut self, pin: &KvPin) -> bool {
        self.kv_catalog
            .as_mut()
            .is_some_and(|catalog| catalog.release(pin))
    }

    /// Tombstone and retire every expert dependency recorded below the fence epochs in `epochs`
    /// (identity `expert:{project}/{pod}` to the epoch now in force): the catalog side of a
    /// compound input invalidation, whose event names no experts and so has no event-only
    /// mapping. A host without a catalog has nothing to do.
    pub(crate) fn invalidate_kv_expert_fences(&mut self, epochs: &BTreeMap<String, u64>) {
        let Some(catalog) = self.kv_catalog.as_mut() else {
            return;
        };
        let retire = catalog.invalidate_expert_fences(epochs);
        retire_keys(catalog, retire);
    }

    /// Called from `apply_committed` for every committed event, before the event's own
    /// state change. Only committed events reach it, never a prepared or discarded
    /// overlay: `semantic` is the delta that the same commit is about to publish.
    pub(crate) fn invalidate_kv_catalog(
        &mut self,
        event: &LedgerEvent,
        semantic: Option<&PreparedDelta>,
    ) {
        let Some(catalog) = self.kv_catalog.as_mut() else {
            return;
        };
        if let Some(retire) = invalidate_catalog_for_event(catalog, event, semantic) {
            retire_keys(catalog, retire);
        }
    }
}

/// The one mapping from a committed event to catalog invalidation, shared by the runtime hook
/// and [`ReferenceKvReuseBackend::apply_committed_event`](crate::ReferenceKvReuseBackend).
/// Returns the backend prefixes to retire, or `None` for an event that invalidates nothing
/// (or a semantic delta event without its delta).
pub(crate) fn invalidate_catalog_for_event(
    catalog: &mut KvPrefixCatalog,
    event: &LedgerEvent,
    semantic: Option<&PreparedDelta>,
) -> Option<Vec<KvPrefixKey>> {
    let retire = match event {
        LedgerEvent::SemanticDeltaCommitted { .. } => {
            catalog.invalidate_identities(semantic?.affected())
        }
        LedgerEvent::Revoked {
            subject,
            generation,
        } => invalidate_revoked_pair(catalog, subject, *generation),
        // The input half of a compound invalidation maps like the plain records: a revocation
        // like `Revoked`, a supersession like a generation bump to the new generation. Which
        // experts the event fences is not in the event; the runtime's apply path invalidates
        // those through `invalidate_kv_expert_fences` with the epochs of the set it derived, and
        // a standalone backend is handed them (see
        // `ReferenceKvReuseBackend::apply_input_invalidation`).
        LedgerEvent::InputInvalidated {
            input,
            generation,
            kind: InputInvalidationKind::Revoked,
            ..
        } => invalidate_revoked_pair(catalog, input, *generation),
        LedgerEvent::InputInvalidated {
            input,
            kind: InputInvalidationKind::Superseded { new_generation, .. },
            ..
        } => catalog.invalidate_superseded(&BTreeMap::from([(input.clone(), *new_generation)])),
        LedgerEvent::ProcedureRevoked { id, generation } => {
            catalog.invalidate_inputs(&BTreeSet::from([(format!("procedure:{id}"), *generation)]))
        }
        LedgerEvent::CapsuleCommitted {
            capsule,
            generation,
            ..
        } => catalog.invalidate_superseded(&BTreeMap::from([(capsule.to_string(), *generation)])),
        LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
            catalog.invalidate_superseded(&BTreeMap::from([(capsule.to_string(), *new)]))
        }
        LedgerEvent::HardConstraintCommitted { key, generation } => catalog.invalidate_superseded(
            &BTreeMap::from([(format!("constraint:{key}"), *generation)]),
        ),
        LedgerEvent::ProcedurePromoted { id, generation } => catalog
            .invalidate_superseded(&BTreeMap::from([(format!("procedure:{id}"), *generation)])),
        // Committed expert structure: every operand identity is invalidated for every
        // generation. Today `commit` refuses this record before it is applied, so the arm
        // is the future admission path's hook; an undecodable record flushes every expert
        // dependency rather than guessing its operands.
        LedgerEvent::ExpertStructuralCommitted { record, .. } => {
            match expert_operand_identities(record) {
                Some(identities) => {
                    catalog.invalidate_identities(&identities.into_iter().collect())
                }
                None => catalog.invalidate_all_experts(),
            }
        }
        // A committed fence or readmission: every dependency recorded below the new epoch,
        // or with none, is tombstoned and retired now rather than at its next use.
        LedgerEvent::ExpertFenceCommitted { .. } => {
            catalog.invalidate_expert_fences(&crate::expert::expert_fence_epochs(event))
        }
        _ => return None,
    };
    Some(retire)
}

/// The catalog invalidation of a revocation of `subject` at `generation`: the exact pair for
/// every kind, and every generation of a semantic key of that name (the oracle treats a
/// revocation as a floor for it).
fn invalidate_revoked_pair(
    catalog: &mut KvPrefixCatalog,
    subject: &str,
    generation: Generation,
) -> Vec<KvPrefixKey> {
    let mut retire = catalog.invalidate_inputs(&BTreeSet::from([(subject.to_owned(), generation)]));
    retire.extend(catalog.invalidate_kind_identities(
        KvDependencyKind::Segment(SegmentKind::Semantic),
        &BTreeSet::from([subject.to_owned()]),
    ));
    retire
}

/// Retire every queued prefix on the catalog's registered backend.
fn retire_queued(catalog: &mut KvPrefixCatalog) {
    let keys = catalog.take_retirements();
    retire_keys(catalog, keys);
}

/// Retire `keys` on the catalog's registered backend. A prefix that is already gone is
/// fine; any other failure is counted, because the entry is already unreachable and a
/// failed retire only leaves backend memory to be reclaimed.
fn retire_keys(catalog: &mut KvPrefixCatalog, keys: Vec<KvPrefixKey>) {
    let Some(retirer) = catalog.retirer.clone() else {
        return;
    };
    for key in keys {
        match retirer.retire(&key) {
            Ok(()) | Err(KvBackendError::PrefixNotFound) => {}
            Err(_) => catalog.stats.retire_failures += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held_all(_: &str) -> bool {
        true
    }

    fn keys(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn semantic_edits_know_the_last_affecting_revision_and_fail_closed_after_a_restore() {
        let mut edits = SemanticEdits::default();
        assert_eq!(edits.last_edit("k"), Revision(0));
        edits.record(&keys(&["k"]), held_all, Revision(3));
        edits.record(&keys(&["k", "j"]), held_all, Revision(5));
        assert_eq!(edits.last_edit("k"), Revision(5));
        assert_eq!(edits.last_edit("j"), Revision(5));
        assert_eq!(edits.last_edit("other"), Revision(0));
        // After a compacted restore at revision 9 the history is unknown: every key is
        // only known to be unchanged since 9.
        let restored = SemanticEdits::restored_at(Revision(9), []);
        assert_eq!(restored.last_edit("k"), Revision(9));
    }

    #[test]
    fn a_revocation_is_a_floor_until_a_later_edit_and_a_missing_key_prunes_its_history() {
        let mut edits = SemanticEdits::default();
        edits.record(&keys(&["k"]), held_all, Revision(2));
        edits.record_revocation("k", true, Revision(4));
        for read_at in 0..=6 {
            assert!(!edits.is_current("k", Revision(read_at), Revision(6)));
        }
        // A key the snapshot does not hold records no revocation.
        edits.record_revocation("absent", false, Revision(4));
        assert!(edits.is_current("absent", Revision(0), Revision(6)));
        edits.record(&keys(&["k"]), held_all, Revision(5));
        assert!(!edits.is_current("k", Revision(4), Revision(6)));
        assert!(edits.is_current("k", Revision(5), Revision(6)));
        assert!(!edits.is_current("k", Revision(7), Revision(6)));
        // A key removed by a delta loses both its edit and its revocation.
        edits.record_revocation("k", true, Revision(6));
        edits.record(&keys(&["k"]), |_| false, Revision(7));
        assert!(edits.edited.is_empty() && edits.revoked.is_empty());
    }

    #[test]
    fn the_history_is_capped_and_overflow_raises_the_floor_without_reviving_a_revoked_key() {
        let mut edits = SemanticEdits::with_cap(Revision(0), 3);
        edits.record_revocation("r", true, Revision(1));
        edits.record(&keys(&["a", "b", "c"]), held_all, Revision(2));
        assert_eq!(edits.edited.len(), 3);
        assert_eq!(edits.floor, Revision(0));
        edits.record(&keys(&["d"]), held_all, Revision(3));
        assert!(
            edits.edited.is_empty(),
            "overflow drops the per-key history"
        );
        assert_eq!(edits.floor, Revision(3));
        // Citations below the floor are dead for every key; the floor itself is live.
        assert!(!edits.is_current("a", Revision(2), Revision(3)));
        assert!(edits.is_current("a", Revision(3), Revision(3)));
        // The revoked key is still dead at the floor.
        assert!(!edits.is_current("r", Revision(3), Revision(3)));
        assert_eq!(edits.revoked.len(), 1);
    }

    mod secondary_events {
        use super::*;
        use crate::kv_secondary::{SecondaryInsert, SecondaryRows, SecondaryRunId};
        use ptr_ledger::{
            ExpertFenceAction, ExpertFenceCause, ExpertFenceResume, ExpertOpKind,
            InputInvalidationKind,
        };
        use ptr_model_api::{ByteTokenizer, KvReuseMode};
        use ptr_pods::{InMemoryPagedKvBackend, KvPageSize};
        use ptr_types::{CapsuleId, EvidenceId, KnowledgeObjectId, RequestId, SemanticContext};

        struct AllLive;

        impl KvLivenessOracle for AllLive {
            fn is_live(&self, _: &KvDependency) -> bool {
                true
            }
        }

        const DIGEST: [u8; 32] = [4; 32];
        const CHAIN: KvChainId = KvChainId([1; 32]);

        fn catalog() -> KvPrefixCatalog {
            let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
            KvPrefixCatalog::new(
                backend.take_fork_authority().unwrap(),
                CatalogLimits::default(),
            )
            .unwrap()
        }

        fn salt() -> KvSalt {
            KvSalt::derive(
                &PrincipalId::from("p"),
                &ProjectId::from("proj"),
                &ModelVersion::from("m"),
                ByteTokenizer::new().identity(),
                "f32",
                "schema",
            )
        }

        fn request(segment: PromptSegment) -> SegmentedModelRequest {
            SegmentedModelRequest::new(
                RequestId::from("r"),
                SemanticContext {
                    revision: Revision(0),
                    entries: Vec::new(),
                },
                ByteTokenizer::new().identity().clone(),
                vec![segment],
            )
            .unwrap()
        }

        fn segment(source: SegmentSource) -> PromptSegment {
            PromptSegment::from_text(&ByteTokenizer::new(), source, "some segment text").unwrap()
        }

        fn run_with(
            catalog: &mut KvPrefixCatalog,
            segment: PromptSegment,
            extras: Vec<KvDependency>,
        ) -> SecondaryRunId {
            let request = request(segment);
            let tokens = request.segments()[0].tokens().len();
            let rows = SecondaryRows::new(
                vec![0.5; tokens * 2].into(),
                vec![0.25; tokens * 2].into(),
                2,
            );
            let salt = salt();
            match catalog
                .secondary_mut()
                .insert(
                    SecondaryInsert {
                        salt: &salt,
                        chain: &CHAIN,
                        mode: KvReuseMode::Exact,
                        request: &request,
                        segment: 0,
                        rows,
                        extra_dependencies: extras,
                    },
                    &AllLive,
                )
                .unwrap()
            {
                crate::SecondaryInsertOutcome::Inserted(id)
                | crate::SecondaryInsertOutcome::AlreadyPresent(id)
                | crate::SecondaryInsertOutcome::RowsAttached(id) => id,
            }
        }

        fn memory(id: &str, generation: u64) -> PromptSegment {
            segment(SegmentSource::Memory {
                id: KnowledgeObjectId::from(id),
                generation: Generation(generation),
            })
        }

        fn alive(catalog: &KvPrefixCatalog, run: SecondaryRunId) -> bool {
            catalog.secondary().contains_live_run(run)
        }

        /// Each event kind of the one mapping kills the run it names in the secondary index
        /// and leaves a bystander alone.
        #[test]
        fn every_event_kind_of_the_mapping_reaches_the_secondary_index() {
            let cases: Vec<(&str, LedgerEvent, &str, u64)> = vec![
                (
                    "Revoked",
                    LedgerEvent::Revoked {
                        subject: "x".into(),
                        generation: Generation(1),
                    },
                    "x",
                    1,
                ),
                (
                    "InputInvalidated revoked",
                    LedgerEvent::InputInvalidated {
                        input: "x".into(),
                        generation: Generation(1),
                        binding_digest: DIGEST,
                        kind: InputInvalidationKind::Revoked,
                        fence_count: 0,
                        fence_digest: DIGEST,
                    },
                    "x",
                    1,
                ),
                (
                    "InputInvalidated superseded",
                    LedgerEvent::InputInvalidated {
                        input: "x".into(),
                        generation: Generation(1),
                        binding_digest: DIGEST,
                        kind: InputInvalidationKind::Superseded {
                            old_generation: Generation(1),
                            new_generation: Generation(2),
                            new_digest: DIGEST,
                            replacement_ref: DIGEST,
                        },
                        fence_count: 0,
                        fence_digest: DIGEST,
                    },
                    "x",
                    1,
                ),
                (
                    "ProcedureRevoked",
                    LedgerEvent::ProcedureRevoked {
                        id: "x".into(),
                        generation: Generation(1),
                    },
                    "procedure:x",
                    1,
                ),
                (
                    "ProcedurePromoted",
                    LedgerEvent::ProcedurePromoted {
                        id: "x".into(),
                        generation: Generation(2),
                    },
                    "procedure:x",
                    1,
                ),
                (
                    "HardConstraintCommitted",
                    LedgerEvent::HardConstraintCommitted {
                        key: "x".into(),
                        generation: Generation(2),
                    },
                    "constraint:x",
                    1,
                ),
                (
                    "CapsuleSuperseded",
                    LedgerEvent::CapsuleSuperseded {
                        capsule: CapsuleId::from("x"),
                        old: Generation(1),
                        new: Generation(2),
                    },
                    "x",
                    1,
                ),
                (
                    "CapsuleCommitted",
                    LedgerEvent::CapsuleCommitted {
                        project: ProjectId::from("proj"),
                        capsule: CapsuleId::from("x"),
                        generation: Generation(2),
                    },
                    "x",
                    1,
                ),
            ];
            for (name, event, identity, generation) in cases {
                let mut catalog = catalog();
                let run = run_with(&mut catalog, memory(identity, generation), Vec::new());
                let bystander = run_with(&mut catalog, memory("bystander", 1), Vec::new());
                let retire = invalidate_catalog_for_event(&mut catalog, &event, None);
                assert!(retire.is_some(), "{name} maps to an invalidation");
                assert!(!alive(&catalog, run), "{name} did not reach the index");
                assert!(alive(&catalog, bystander), "{name} over-invalidated");
            }
        }

        #[test]
        fn a_revocation_reaches_a_semantic_run_at_every_generation() {
            let mut catalog = catalog();
            let semantic = |generation| {
                segment(SegmentSource::Semantic {
                    key: "price:sku".to_owned(),
                    generation: Generation(generation),
                })
            };
            let g5 = run_with(&mut catalog, semantic(5), Vec::new());
            let g9 = run_with(&mut catalog, semantic(9), Vec::new());
            let event = LedgerEvent::Revoked {
                subject: "price:sku".into(),
                generation: Generation(1),
            };
            invalidate_catalog_for_event(&mut catalog, &event, None);
            assert!(!alive(&catalog, g5) && !alive(&catalog, g9));
        }

        #[test]
        fn a_derived_run_dies_with_its_exact_revoked_pair_only() {
            let mut catalog = catalog();
            let derived = |generation| {
                segment(SegmentSource::Derived {
                    id: EvidenceId::from("ev"),
                    generation: Generation(generation),
                })
            };
            let g1 = run_with(&mut catalog, derived(1), Vec::new());
            let g2 = run_with(&mut catalog, derived(2), Vec::new());
            let event = LedgerEvent::Revoked {
                subject: "ev".into(),
                generation: Generation(1),
            };
            invalidate_catalog_for_event(&mut catalog, &event, None);
            assert!(!alive(&catalog, g1));
            assert!(alive(&catalog, g2));
        }

        fn expert(pod: &str, epoch: u64) -> KvDependency {
            KvDependency::expert_fenced(
                &ProjectId::from("proj"),
                &PodId::from(pod),
                Generation(1),
                DIGEST,
                epoch,
            )
        }

        #[test]
        fn a_committed_expert_fence_reaches_the_index_per_epoch() {
            let mut catalog = catalog();
            let below = run_with(&mut catalog, memory("a", 1), vec![expert("e1", 1)]);
            let other = run_with(&mut catalog, memory("b", 1), vec![expert("e2", 1)]);
            let at = run_with(&mut catalog, memory("c", 1), vec![expert("e1", 5)]);
            let event = LedgerEvent::ExpertFenceCommitted {
                project: ProjectId::from("proj"),
                pod: PodId::from("e1"),
                action: ExpertFenceAction::Fence,
                cause: ExpertFenceCause::InputRevoked,
                resume: ExpertFenceResume::Serving,
                generation: Generation(1),
                expected_state_version: 0,
                epoch: 5,
                revocation: None,
                evidence: None,
            };
            assert!(invalidate_catalog_for_event(&mut catalog, &event, None).is_some());
            assert!(!alive(&catalog, below));
            assert!(alive(&catalog, at), "a dependency at the new epoch stays");
            assert!(alive(&catalog, other), "another expert is untouched");
        }

        #[test]
        fn an_undecodable_expert_structure_flushes_every_expert_run() {
            let mut catalog = catalog();
            let expert_run = run_with(&mut catalog, memory("a", 1), vec![expert("e1", 1)]);
            let plain = run_with(&mut catalog, memory("b", 1), Vec::new());
            let event = LedgerEvent::ExpertStructuralCommitted {
                project: ProjectId::from("proj"),
                op_kind: ExpertOpKind::Retire,
                proposal_id: DIGEST,
                operation_id: DIGEST,
                evidence_digest: DIGEST,
                record: vec![0xff, 0xff],
            };
            assert!(invalidate_catalog_for_event(&mut catalog, &event, None).is_some());
            assert!(!alive(&catalog, expert_run));
            assert!(alive(&catalog, plain));
        }

        #[test]
        fn an_event_that_invalidates_nothing_leaves_the_index_alone() {
            let mut catalog = catalog();
            let run = run_with(&mut catalog, memory("a", 1), Vec::new());
            let event = LedgerEvent::VerifierAttested {
                subject: "a".into(),
                passed: true,
            };
            assert!(invalidate_catalog_for_event(&mut catalog, &event, None).is_none());
            assert!(alive(&catalog, run));
        }
    }

    #[test]
    fn expert_value_is_live_only_for_the_same_generation_in_a_serving_state() {
        let g = Generation(2);
        assert!(expert_value_is_serving("2:serving:5", g, 0));
        assert!(expert_value_is_serving(
            &ptr_state::expert_entry(2, ExpertState::Frozen, 5),
            g,
            0
        ));
        assert!(expert_value_is_serving("2:serving:5", g, 5));
        assert!(!expert_value_is_serving("2:serving:5", g, 4));
        assert!(!expert_value_is_serving("2:serving:x", g, 4));
        for dead in [
            "1:serving:5",
            "3:serving:5",
            "2:retired:5",
            "2:fenced:5",
            "2:superseded:5",
            "2:candidate:5",
            "2",
            "x:serving:1",
            "",
        ] {
            assert!(!expert_value_is_serving(dead, g, 0), "{dead}");
        }
    }
}
