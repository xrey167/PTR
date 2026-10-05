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
//! * **Canonical positions only.** Only a request whose positions start at 0 without
//!   explicit ids can hit, and only an entry published at `position_offset` 0 is admitted.
//!
//! # Not covered here
//!
//! The adapter, expert and base-model generation dependencies, per-layer chain keys beyond
//! the opaque [`KvChainId`], RoPE (doc 36 defect 3) and per-layer pages (defect 5) belong
//! to later phases. Invalidation reaches this catalog through
//! `PtrRuntime::apply_committed` for committed revocations, committed supersessions and
//! committed semantic deltas, never for prepared ones.

use crate::PtrRuntime;
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    LineageContribution, SegmentKind, SegmentedModelRequest, TokenId, TokenizerIdentity,
};
use ptr_pods::{
    KvBackendError, KvForkAuthority, KvForkWitness, KvPrefixKey, KvSharedPrefix, KvWitnessRevoker,
    PagedKvTensorBackend,
};
use ptr_semdb::PreparedDelta;
use ptr_types::{Digest, Generation, ModelVersion, PrincipalId, ProjectId, Validity};
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
const PREFIX_LINEAGE_DOMAIN: &[u8] = b"ptr.kv.prefix-lineage.v1\0";

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
        }
    }
}

impl std::error::Error for KvCatalogError {}

/// A generation-bearing input an entry's K/V was computed from.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvDependency {
    pub kind: SegmentKind,
    /// The lifecycle target string: a semantic key, memory object id or derived evidence
    /// id, which is also what committed lifecycle events name.
    pub identity: String,
    pub generation: Generation,
    pub content_digest: Digest,
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
}

type EntryKey = (KvChainId, Digest);

/// Handle on a pinned entry. Release it with [`KvPrefixCatalog::release`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPin {
    node: KvChunkKey,
    entry: EntryKey,
    id: u64,
}

/// A hit: the prefix to fork, the witness that authorises forking it once, how many
/// tokens it covers, and the pin that keeps the entry from eviction until released.
#[derive(Debug)]
pub struct KvCatalogHit {
    pub witness: KvForkWitness,
    pub prefix: KvSharedPrefix,
    pub matched_tokens: usize,
    pub pin: KvPin,
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
    pub noncanonical_misses: u64,
    pub use_time_rejections: u64,
    pub cancelled_pins: u64,
    pub retire_failures: u64,
}

struct Entry {
    prefix: KvSharedPrefix,
    dependencies: Vec<KvDependency>,
    page_bytes: u64,
    metadata_bytes: u64,
    refcount: u32,
    tombstoned: bool,
    /// Whether this entry still counts in `live_key_refs` for its backend prefix.
    live_ref: bool,
    last_used: u64,
    pins: BTreeMap<u64, KvWitnessRevoker>,
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
    retire_queue: Vec<KvPrefixKey>,
    page_bytes: u64,
    metadata_bytes: u64,
    clock: u64,
    next_pin: u64,
    stats: KvCatalogStats,
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
        Ok(Self {
            limits,
            authority,
            retirer: None,
            nodes: BTreeMap::new(),
            by_identity: BTreeMap::new(),
            live_key_refs: BTreeMap::new(),
            retire_queue: Vec::new(),
            page_bytes: 0,
            metadata_bytes: 0,
            clock: 0,
            next_pin: 0,
            stats: KvCatalogStats::default(),
        })
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
        let mut candidates: Vec<(KvChunkKey, EntryKey, usize)> = Vec::new();
        for (index, key) in keys.iter().enumerate() {
            let Some(node) = self.nodes.get(key) else {
                break;
            };
            if !node.entries.keys().any(|(chain, _)| chain == lookup.chain) {
                continue;
            }
            let covered = (index + 1) * KV_CHUNK_TOKENS;
            let (lineage, _) = prefix_lineage(lookup.request, &tokens, covered);
            let entry_key = (*lookup.chain, lineage);
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
                Some(true) => return Some(self.pin_hit(node_key, entry_key, covered)),
                Some(false) => {
                    self.stats.use_time_rejections += 1;
                    self.tombstone_entry(&node_key, &entry_key);
                }
                None => {}
            }
        }
        self.stats.misses += 1;
        None
    }

    fn pin_hit(
        &mut self,
        node_key: KvChunkKey,
        entry_key: EntryKey,
        covered: usize,
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
                self.release(&pin);
                Err(KvForkError::Backend(error))
            }
        }
    }

    /// Whether the entry behind `pin` is still reachable. `false` means a dependency
    /// invalidation cancelled the fork: the run must restart from a fresh prefill and no
    /// result produced after the invalidation may escape.
    pub fn pin_is_live(&self, pin: &KvPin) -> bool {
        self.entry(&pin.node, &pin.entry)
            .is_some_and(|entry| !entry.tombstoned && entry.pins.contains_key(&pin.id))
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

    /// Offer a finished prefill. Evicts unreferenced entries in LRU order to make room and
    /// returns [`KvCatalogError::CapacityExhausted`] when only pinned entries remain, so
    /// the bounds are never exceeded. Prefixes made unreferenced by eviction are queued for
    /// [`Self::take_retirements`].
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
        } = insert;
        if prefix.position_offset != 0 {
            return Err(KvCatalogError::NonCanonicalPosition);
        }
        let tokens = request.tokens();
        let covered = prefix.sequence_length;
        if covered == 0 || covered % KV_CHUNK_TOKENS != 0 || covered > tokens.len() {
            return Err(KvCatalogError::PrefixShape);
        }
        let (lineage, dependencies) = prefix_lineage(request, &tokens, covered);
        if !dependencies.iter().all(|dep| liveness.is_live(dep)) {
            return Err(KvCatalogError::DependencyNotLive);
        }
        let path: Vec<KvChunkKey> = chunk_keys(salt, &tokens)
            .into_iter()
            .take(covered / KV_CHUNK_TOKENS)
            .collect();
        let leaf = *path.last().ok_or(KvCatalogError::PrefixShape)?;
        let entry_key = (*chain, lineage);
        if let Some(existing) = self
            .entry(&leaf, &entry_key)
            .filter(|entry| !entry.tombstoned)
        {
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
            + 32 * prefix.page_digests.len() as u64;
        loop {
            let missing = path
                .iter()
                .filter(|key| !self.nodes.contains_key(key))
                .count();
            let refused = if self.nodes.len() + missing > self.limits.max_trie_nodes {
                Some(KvCapacityResource::TrieNodes)
            } else if self.page_bytes.saturating_add(page_bytes) > self.limits.max_kv_page_bytes {
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
            if !self.evict_lru() {
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
        self.page_bytes += page_bytes;
        self.metadata_bytes += metadata_bytes;
        let node = self.nodes.get_mut(&leaf).expect("leaf node was created");
        node.entries.insert(
            entry_key,
            Entry {
                prefix,
                dependencies,
                page_bytes,
                metadata_bytes,
                refcount: 0,
                tombstoned: false,
                live_ref: true,
                last_used: self.clock,
                pins: BTreeMap::new(),
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
        self.page_bytes -= entry.page_bytes;
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

/// Lineage digest of the first `covered` tokens, and the dependencies of that prefix.
///
/// The digest covers the semantic revision, the tokenizer and, per segment overlapping the
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
    hasher.update(request.revision().0.to_le_bytes());
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
                    kind,
                    identity,
                    generation,
                    content_digest,
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

// Runtime wiring -------------------------------------------------------------------------

impl KvLivenessOracle for PtrRuntime {
    /// Live means the lifecycle authority knows `identity` at `generation` and it is
    /// neither revoked nor superseded. A semantic key carries no lifecycle generation, so
    /// it is live while the committed snapshot still holds the key and no tombstone names
    /// it; committed deltas also invalidate it through the catalog hook. The content digest
    /// of a segment is not re-verified here: it is part of the entry's lineage and of the
    /// request's, so a different content never matches the entry.
    fn is_live(&self, dependency: &KvDependency) -> bool {
        match self.generation_validity(&dependency.identity, dependency.generation) {
            Some(Validity::Live) => true,
            Some(_) => false,
            None => {
                dependency.kind == SegmentKind::Semantic
                    && !self.revoked_generations().any(|(id, generation)| {
                        id == dependency.identity && generation == dependency.generation
                    })
                    && self.snapshot().get(&dependency.identity).is_some()
            }
        }
    }
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
        let retire = match event {
            LedgerEvent::SemanticDeltaCommitted { .. } => match semantic {
                Some(delta) => catalog.invalidate_identities(delta.affected()),
                None => return,
            },
            LedgerEvent::Revoked {
                subject,
                generation,
            } => catalog.invalidate_inputs(&BTreeSet::from([(subject.clone(), *generation)])),
            LedgerEvent::ProcedureRevoked { id, generation } => catalog
                .invalidate_inputs(&BTreeSet::from([(format!("procedure:{id}"), *generation)])),
            LedgerEvent::CapsuleCommitted {
                capsule,
                generation,
                ..
            } => {
                catalog.invalidate_superseded(&BTreeMap::from([(capsule.to_string(), *generation)]))
            }
            LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                catalog.invalidate_superseded(&BTreeMap::from([(capsule.to_string(), *new)]))
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => catalog
                .invalidate_superseded(&BTreeMap::from([(
                    format!("constraint:{key}"),
                    *generation,
                )])),
            LedgerEvent::ProcedurePromoted { id, generation } => catalog
                .invalidate_superseded(&BTreeMap::from([(format!("procedure:{id}"), *generation)])),
            _ => return,
        };
        retire_keys(catalog, retire);
    }
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
