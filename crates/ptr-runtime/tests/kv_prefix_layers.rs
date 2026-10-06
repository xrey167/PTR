//! Per-layer catalog entries (R009/R010 WP3): per-layer dependency sets, per-layer
//! tombstoning, per-layer pins, byte accounting per unique page across layers and the
//! snapshot v2 that carries the layer index.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, TokenId, Tokenizer};
use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryPagedKvBackend, InMemoryPagedKvCache,
    KvContinuationExecutor, KvPageBinding, KvPageSize, KvSharedPrefix, KvTensorDType,
    KvTensorSchema, PagedKvTensorBackend, ReferenceCausalAttentionExecutor,
};
use ptr_runtime::{
    CatalogLimits, KvCapacityResource, KvCatalogError, KvChainId, KvDependency, KvLayerEntry,
    KvLayerHit, KvLayerQuery, KvLayeredInsert, KvLayeredLookup, KvLivenessOracle, KvPageCharge,
    KvPrefixCatalog, KvRequestPositions, KvSalt, KvSnapshotError, KV_CHUNK_TOKENS,
};
use ptr_types::{
    AdapterVersion, DeviceId, Generation, ModelVersion, PodId, PrincipalId, ProjectId, RequestId,
    Revision, SemanticContext,
};
use sha2::{Digest as _, Sha256};

const WIDTH: usize = 4;
const PAGE: u64 = 1000;

/// Dead `(identity, generation)` pairs plus a per-identity minimum live fence epoch.
#[derive(Default)]
struct Dead {
    dead: BTreeSet<(String, Generation)>,
    fences: BTreeMap<String, u64>,
}

impl Dead {
    fn kill(identity: &str) -> Self {
        Self {
            dead: BTreeSet::from([(identity.to_owned(), Generation(1))]),
            fences: BTreeMap::new(),
        }
    }
}

impl KvLivenessOracle for Dead {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        !self
            .dead
            .contains(&(dependency.identity.clone(), dependency.generation))
            && self
                .fences
                .get(&dependency.identity)
                .is_none_or(|epoch| dependency.fence_epoch >= *epoch)
    }
}

fn schema() -> KvTensorSchema {
    KvTensorSchema {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    }
}

fn binding() -> KvPageBinding {
    KvPageBinding {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        generation: Generation(1),
        execution_manifest: [3; 32],
        principal: PrincipalId::from("alice"),
        device: DeviceId::from("cpu"),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::DEFAULT,
    }
}

fn salt() -> KvSalt {
    KvSalt::derive(
        &PrincipalId::from("alice"),
        &ProjectId::from("p1"),
        &ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        "f32-cpu-reference",
        "paged-kv-v1",
    )
}

fn request(segments: Vec<PromptSegment>) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from("r"),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        segments,
    )
    .unwrap()
}

fn raw(text: &str) -> PromptSegment {
    PromptSegment::raw_from_text(&ByteTokenizer::new(), text).unwrap()
}

fn block(label: char) -> String {
    label.to_string().repeat(KV_CHUNK_TOKENS)
}

/// `a` + `second`: two chunks, two pages, the first page shared by every such request.
fn two_chunks(second: char) -> SegmentedModelRequest {
    request(vec![raw(&format!("{}{}", block('a'), block(second)))])
}

fn embed(token: TokenId) -> Vec<f32> {
    (0..WIDTH)
        .map(|i| ((token.0 as f32) + i as f32) / 97.0 - 0.5)
        .collect()
}

fn fresh(backend: &InMemoryPagedKvBackend) -> InMemoryPagedKvCache {
    backend.allocate_bound(schema(), 128, binding()).unwrap()
}

fn publish(backend: &InMemoryPagedKvBackend, tokens: &[TokenId]) -> KvSharedPrefix {
    let mut cache = fresh(backend);
    ReferenceCausalAttentionExecutor::new(CausalAttentionWeights::identity(WIDTH).unwrap(), 2)
        .unwrap()
        .prefill(
            backend,
            &mut cache,
            CausalAttentionInput {
                tokens: tokens.iter().copied().map(embed).collect(),
            },
        )
        .unwrap();
    backend.publish_prefix(&mut cache).unwrap()
}

struct Harness {
    backend: InMemoryPagedKvBackend,
    catalog: KvPrefixCatalog,
}

fn harness(limits: CatalogLimits) -> Harness {
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let authority = backend.take_fork_authority().unwrap();
    let catalog = KvPrefixCatalog::new(authority, limits)
        .unwrap()
        .with_retirer(Arc::new(backend.clone()));
    Harness { backend, catalog }
}

const LAYERS: u16 = 4;
/// The adapter is attached from layer `F` on; the expert sits in layer `J`, so it shapes
/// layers above `J`.
const F: u16 = 2;
const J: u16 = 1;
const ADAPTER: &str = "adapter:lora-a";
const BASE: &str = "model:base";

fn chain(layer: u16) -> KvChainId {
    KvChainId([10 + layer as u8; 32])
}

fn expert_dep(epoch: u64) -> KvDependency {
    KvDependency::expert_fenced(
        &ProjectId::from("p1"),
        &PodId::from("moe-1-2"),
        Generation(1),
        [7; 32],
        epoch,
    )
}

fn expert_id() -> String {
    expert_dep(0).identity
}

/// The dependency set of layer `l` as the caller states it.
fn layer_dependencies(layer: u16) -> Vec<KvDependency> {
    let mut deps = vec![KvDependency::base_model(BASE, Generation(1), [1; 32])];
    if layer >= F {
        deps.push(KvDependency::adapter(ADAPTER, Generation(1), [2; 32]));
    }
    if layer > J {
        deps.push(expert_dep(1));
    }
    deps
}

fn bytes_only(max_kv_page_bytes: u64) -> CatalogLimits {
    CatalogLimits {
        max_trie_nodes: 1 << 20,
        max_kv_page_bytes,
        max_metadata_bytes: 1 << 30,
    }
}

/// Layers `0` and `1` hold the base K/V and share their pages; `2` and `3` own theirs.
fn layer_pages(layer: u16, pages: usize) -> Vec<[u8; 32]> {
    let owner = if layer < F { 0 } else { layer };
    (0..pages)
        .map(|page| {
            let mut digest = [0u8; 32];
            digest[0] = owner as u8 + 1;
            digest[1] = page as u8;
            digest
        })
        .collect()
}

fn layer_entry(
    base: &KvSharedPrefix,
    layer: u16,
    chunks: usize,
    extra: Vec<KvDependency>,
) -> KvLayerEntry {
    let mut prefix = base.clone();
    prefix.sequence_length = chunks * KV_CHUNK_TOKENS;
    prefix.page_digests = layer_pages(layer, chunks);
    let mut unique: Vec<[u8; 32]> = Vec::new();
    for digest in &prefix.page_digests {
        if !unique.contains(digest) {
            unique.push(*digest);
        }
    }
    KvLayerEntry {
        layer,
        chain: chain(layer),
        pages: unique
            .into_iter()
            .map(|digest| KvPageCharge {
                digest,
                bytes: PAGE,
            })
            .collect(),
        prefix,
        extra_dependencies: extra,
    }
}

impl Harness {
    fn insert_layers_with(
        &mut self,
        request: &SegmentedModelRequest,
        chunks: usize,
        extra_for: impl Fn(u16) -> Vec<KvDependency>,
        live: &dyn KvLivenessOracle,
    ) -> Result<(), KvCatalogError> {
        let base = publish(&self.backend, &request.tokens());
        self.catalog.insert_layers(
            KvLayeredInsert {
                salt: &salt(),
                request,
                layers: (0..LAYERS)
                    .map(|l| layer_entry(&base, l, chunks, extra_for(l)))
                    .collect(),
            },
            live,
        )
    }

    fn insert_all(&mut self, request: &SegmentedModelRequest) {
        self.insert_layers_with(request, 2, layer_dependencies, &Dead::default())
            .unwrap();
    }

    fn lookup_layers(
        &mut self,
        request: &SegmentedModelRequest,
        live: &dyn KvLivenessOracle,
    ) -> Vec<KvLayerHit> {
        let queries: Vec<KvLayerQuery> = (0..LAYERS)
            .map(|layer| KvLayerQuery {
                layer,
                chain: chain(layer),
            })
            .collect();
        self.catalog
            .lookup_layers(
                KvLayeredLookup {
                    salt: &salt(),
                    request,
                    positions: &KvRequestPositions::Canonical,
                    layers: &queries,
                },
                live,
            )
            .unwrap()
    }

    /// Matched tokens per layer (`None` for a miss); releases every pin.
    fn depths(
        &mut self,
        request: &SegmentedModelRequest,
        live: &dyn KvLivenessOracle,
    ) -> Vec<Option<usize>> {
        let hits = self.lookup_layers(request, live);
        hits.into_iter()
            .map(|layer| {
                layer.hit.map(|hit| {
                    let matched = hit.matched_tokens();
                    let pin = hit.pin().clone();
                    self.catalog.release(&pin);
                    matched
                })
            })
            .collect()
    }
}

const FULL: Option<usize> = Some(2 * KV_CHUNK_TOKENS);

#[test]
fn adapter_revocation_tombstones_only_the_adapted_layers() {
    let req = two_chunks('b');
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    assert_eq!(h.catalog.stats().entries, LAYERS as usize);
    assert_eq!(h.depths(&req, &Dead::default()), vec![FULL; 4]);

    // The layers here share one backend prefix key, still held by layers 0 and 1, so
    // nothing is retired on the backend yet.
    let retired = h
        .catalog
        .invalidate_identities(&BTreeSet::from([ADAPTER.to_owned()]));
    assert!(retired.is_empty());
    assert_eq!(h.catalog.stats().entries, F as usize);
    assert_eq!(
        h.depths(&req, &Dead::default()),
        vec![FULL, FULL, None, None]
    );
    // A generation bump behaves the same: a fresh catalog, a kind-scoped supersession.
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    h.catalog.invalidate_kind_superseded(
        ptr_runtime::KvDependencyKind::Adapter,
        &BTreeMap::from([(ADAPTER.to_owned(), Generation(2))]),
    );
    assert_eq!(
        h.depths(&req, &Dead::default()),
        vec![FULL, FULL, None, None]
    );
}

#[test]
fn expert_revocation_and_fence_bump_reach_only_layers_above_the_expert() {
    let req = two_chunks('b');
    // Revocation of expert (1, i) by (identity, generation).
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    h.catalog
        .invalidate_inputs(&BTreeSet::from([(expert_id(), Generation(1))]));
    assert_eq!(
        h.depths(&req, &Dead::default()),
        vec![FULL, FULL, None, None]
    );
    assert_eq!(h.catalog.stats().entries, 2);

    // A fence epoch bump to 2 (recorded 1) likewise.
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    h.catalog
        .invalidate_expert_fences(&BTreeMap::from([(expert_id(), 2)]));
    assert_eq!(
        h.depths(&req, &Dead::default()),
        vec![FULL, FULL, None, None]
    );
    // An unrelated expert changes nothing.
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    h.catalog
        .invalidate_expert_fences(&BTreeMap::from([("expert:p1/other".to_owned(), 9)]));
    assert_eq!(h.depths(&req, &Dead::default()), vec![FULL; 4]);
}

#[test]
fn liveness_is_checked_per_layer_at_use_even_when_invalidation_was_lost() {
    let req = two_chunks('b');
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    // The adapter is dead for the oracle but the catalog was never told.
    let dead = Dead::kill(ADAPTER);
    assert_eq!(h.depths(&req, &dead), vec![FULL, FULL, None, None]);
    let stats = h.catalog.stats();
    assert_eq!(
        stats.use_time_rejections, 2,
        "one rejection per adapted layer"
    );
    assert_eq!(stats.entries, 2);
    // A fence behind the recorded epoch kills only layers above the expert.
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    let fenced = Dead {
        fences: BTreeMap::from([(expert_id(), 2)]),
        ..Dead::default()
    };
    assert_eq!(h.depths(&req, &fenced), vec![FULL, FULL, None, None]);
    // The base model dies on every layer.
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    assert_eq!(h.depths(&req, &Dead::kill(BASE)), vec![None; 4]);
}

#[test]
fn lookup_layers_returns_the_deepest_live_node_independently_per_layer() {
    let req = two_chunks('b');
    let mut h = harness(CatalogLimits::default());
    // A shallow (one chunk) entry on every layer, then a deep one in which layer 3 also
    // depends on a segment-free extra input that dies.
    h.insert_layers_with(&req, 1, layer_dependencies, &Dead::default())
        .unwrap();
    let extra = KvDependency::base_model("model:extra", Generation(1), [9; 32]);
    h.insert_layers_with(
        &req,
        2,
        |l| {
            let mut deps = layer_dependencies(l);
            if l == 3 {
                deps.push(extra.clone());
            }
            deps
        },
        &Dead::default(),
    )
    .unwrap();
    assert_eq!(h.catalog.stats().entries, 8);
    let half = Some(KV_CHUNK_TOKENS);
    assert_eq!(h.depths(&req, &Dead::default()), vec![FULL; 4]);
    // Layer 3's deep entry dies; it falls back to the shallow ancestor on its own while
    // layers 0..=2 stay at the deep node.
    let dead = Dead::kill("model:extra");
    assert_eq!(h.depths(&req, &dead), vec![FULL, FULL, FULL, half]);
    // A layer that was never inserted misses without disturbing the others.
    let queries = [
        KvLayerQuery {
            layer: 0,
            chain: chain(0),
        },
        KvLayerQuery {
            layer: 9,
            chain: chain(9),
        },
    ];
    let hits = h
        .catalog
        .lookup_layers(
            KvLayeredLookup {
                salt: &salt(),
                request: &req,
                positions: &KvRequestPositions::Canonical,
                layers: &queries,
            },
            &Dead::default(),
        )
        .unwrap();
    assert!(hits[0].hit.is_some() && hits[1].hit.is_none());
    assert_eq!((hits[0].layer, hits[1].layer), (0, 9));
    // Another request's chunks, or a non-canonical position, never hit.
    assert_eq!(
        h.depths(&two_chunks('c'), &Dead::default()),
        vec![Some(KV_CHUNK_TOKENS); 4],
        "the shared first chunk hits on every layer"
    );
    let offset = [KvLayerQuery {
        layer: 0,
        chain: chain(0),
    }];
    let hits = h
        .catalog
        .lookup_layers(
            KvLayeredLookup {
                salt: &salt(),
                request: &req,
                positions: &KvRequestPositions::Offset(3),
                layers: &offset,
            },
            &Dead::default(),
        )
        .unwrap();
    assert!(hits[0].hit.is_none());
}

#[test]
fn pins_are_per_layer_and_report_cancellation_per_layer() {
    let req = two_chunks('b');
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    let hits = h.lookup_layers(&req, &Dead::default());
    let pins: Vec<_> = hits
        .into_iter()
        .map(|layer| layer.hit.expect("all layers hit").pin().clone())
        .collect();
    assert_eq!(
        pins.iter().map(|pin| pin.layer()).collect::<Vec<_>>(),
        (0..LAYERS).map(Some).collect::<Vec<_>>()
    );
    assert!(pins.iter().all(|pin| h.catalog.pin_is_live(pin)));

    h.catalog
        .invalidate_identities(&BTreeSet::from([ADAPTER.to_owned()]));
    let live: Vec<bool> = pins.iter().map(|pin| h.catalog.pin_is_live(pin)).collect();
    assert_eq!(live, vec![true, true, false, false]);
    // The cancelled layers stay held (draining) until released, never reclaimed under a pin.
    assert!(pins.iter().all(|pin| h.catalog.pin_held(pin)));
    assert_eq!(h.catalog.stats().tombstoned_pinned_entries, 2);

    // revalidate_pin asks the oracle per layer at use: a dead expert cancels layer 1's
    // sibling above it only, here layer 1 stays alive.
    let dead = Dead::kill(&expert_id());
    let results: Vec<bool> = pins
        .iter()
        .map(|pin| h.catalog.revalidate_pin(pin, &dead))
        .collect();
    assert_eq!(results, vec![true, true, false, false]);
    for pin in &pins {
        assert!(h.catalog.release(pin));
    }
    assert_eq!(h.catalog.stats().entries, 2, "reclaimed on release");
}

#[test]
fn pages_are_charged_once_per_unique_digest_across_layers() {
    let req = two_chunks('b');
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    // Layers 0 and 1 share two pages; layers 2 and 3 own two each: six unique pages.
    assert_eq!(h.catalog.stats().page_bytes, 6 * PAGE);
    h.catalog
        .invalidate_identities(&BTreeSet::from([ADAPTER.to_owned()]));
    assert_eq!(
        h.catalog.stats().page_bytes,
        2 * PAGE,
        "dropping layers 2 and 3 frees their own pages and keeps the shared ones"
    );
    // Dropping one of the two sharers keeps the shared pages charged for the other.
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    h.catalog
        .invalidate_expert_fences(&BTreeMap::from([(expert_id(), 2)]));
    assert_eq!(h.catalog.stats().page_bytes, 2 * PAGE);
    // A byte bound that fits the unique pages but not the per-entry sum accepts the insert.
    let mut tight = harness(bytes_only(6 * PAGE));
    tight.insert_all(&req);
    assert_eq!(tight.catalog.stats().page_bytes, 6 * PAGE);
}

#[test]
fn insert_layers_is_all_or_nothing_and_refuses_ragged_shapes() {
    let req = two_chunks('b');
    // Five pages fit but six are needed: nothing stays admitted.
    let mut h = harness(bytes_only(5 * PAGE));
    let error = h
        .insert_layers_with(&req, 2, layer_dependencies, &Dead::default())
        .unwrap_err();
    assert_eq!(
        error,
        KvCatalogError::CapacityExhausted(KvCapacityResource::KvPageBytes)
    );
    let stats = h.catalog.stats();
    assert_eq!((stats.entries, stats.page_bytes, stats.nodes), (0, 0, 0));
    assert_eq!(stats.metadata_bytes, 0);

    // One dead dependency on one layer refuses the whole insert.
    let mut h = harness(CatalogLimits::default());
    let error = h
        .insert_layers_with(&req, 2, layer_dependencies, &Dead::kill(ADAPTER))
        .unwrap_err();
    assert_eq!(error, KvCatalogError::DependencyNotLive);
    assert_eq!(h.catalog.stats().entries, 0);

    // Empty, repeated and ragged layer sets.
    let base = publish(&h.backend, &req.tokens());
    let layer_salt = salt();
    let insert = |layers: Vec<KvLayerEntry>| KvLayeredInsert {
        salt: &layer_salt,
        request: &req,
        layers,
    };
    assert_eq!(
        h.catalog
            .insert_layers(insert(Vec::new()), &Dead::default()),
        Err(KvCatalogError::LayerShape)
    );
    assert_eq!(
        h.catalog.insert_layers(
            insert(vec![
                layer_entry(&base, 0, 2, Vec::new()),
                layer_entry(&base, 0, 2, Vec::new())
            ]),
            &Dead::default()
        ),
        Err(KvCatalogError::LayerShape)
    );
    assert_eq!(
        h.catalog.insert_layers(
            insert(vec![
                layer_entry(&base, 0, 2, Vec::new()),
                layer_entry(&base, 1, 1, Vec::new())
            ]),
            &Dead::default()
        ),
        Err(KvCatalogError::LayerShape)
    );
    // A charge that does not describe the layer's pages.
    let mut bad = layer_entry(&base, 0, 2, Vec::new());
    bad.pages.pop();
    assert_eq!(
        h.catalog.insert_layers(insert(vec![bad]), &Dead::default()),
        Err(KvCatalogError::PageChargeMismatch)
    );
    // An offset prefix is not canonical.
    let mut offset = layer_entry(&base, 0, 2, Vec::new());
    offset.prefix.position_offset = 4;
    assert_eq!(
        h.catalog
            .insert_layers(insert(vec![offset]), &Dead::default()),
        Err(KvCatalogError::NonCanonicalPosition)
    );
    // Lookups refuse empty and repeated layers.
    let dup = [
        KvLayerQuery {
            layer: 1,
            chain: chain(1),
        },
        KvLayerQuery {
            layer: 1,
            chain: chain(1),
        },
    ];
    for layers in [&dup[..], &[][..]] {
        assert!(matches!(
            h.catalog.lookup_layers(
                KvLayeredLookup {
                    salt: &salt(),
                    request: &req,
                    positions: &KvRequestPositions::Canonical,
                    layers,
                },
                &Dead::default()
            ),
            Err(KvCatalogError::LayerShape)
        ));
    }
    assert_eq!(h.catalog.stats().entries, 0);
}

#[test]
fn reinserting_the_same_layers_is_idempotent_and_layered_entries_do_not_alias_whole_stack_ones() {
    let req = two_chunks('b');
    let mut h = harness(CatalogLimits::default());
    h.insert_all(&req);
    h.insert_all(&req);
    assert_eq!(h.catalog.stats().entries, LAYERS as usize);
    assert_eq!(h.catalog.stats().page_bytes, 6 * PAGE);
    // A whole-stack lookup under a layer's chain does not see the layered entry.
    let hit = h.catalog.lookup(
        ptr_runtime::KvLookup {
            salt: &salt(),
            chain: &chain(0),
            request: &req,
            positions: &KvRequestPositions::Canonical,
        },
        &Dead::default(),
    );
    assert!(hit.is_none());
}

fn reseal_v3(body: &[u8]) -> Vec<u8> {
    reseal_with(b"ptr.kv.catalog-snapshot.v3\0", body)
}

fn reseal_v2(body: &[u8]) -> Vec<u8> {
    reseal_with(b"ptr.kv.catalog-snapshot.v2\0", body)
}

fn reseal_with(domain: &[u8], body: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(body);
    let mut out = body.to_vec();
    out.extend_from_slice(&hasher.finalize());
    out
}

fn reseal_v1(body: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr.kv.catalog-snapshot.v1\0");
    hasher.update(body);
    let mut out = body.to_vec();
    out.extend_from_slice(&hasher.finalize());
    out
}

fn restore(snapshot: &[u8]) -> Result<ptr_runtime::KvRestoreReport, KvSnapshotError> {
    harness(CatalogLimits::default())
        .catalog
        .restore_entries(snapshot)
}

#[test]
fn snapshot_v3_round_trips_with_the_layer_index() {
    let req = two_chunks('b');
    let mut source = harness(CatalogLimits::default());
    source.insert_all(&req);
    let snapshot = source.catalog.export_entries().unwrap();
    assert!(snapshot.starts_with(b"PTRKVL03"));
    assert_eq!(
        snapshot,
        source.catalog.export_entries().unwrap(),
        "deterministic"
    );

    let mut target = harness(CatalogLimits::default());
    let report = target.catalog.restore_entries(&snapshot).unwrap();
    assert_eq!(report.entries, LAYERS as usize);
    assert_eq!(
        target.catalog.export_entries().unwrap(),
        snapshot,
        "restore then export is the identity"
    );
    assert_eq!(target.catalog.stats().page_bytes, 6 * PAGE);
    // Each layer is addressable by its layer index and chain, with its own dependencies.
    assert_eq!(target.depths(&req, &Dead::default()), vec![FULL; 4]);
    assert_eq!(
        target.depths(&req, &Dead::kill(ADAPTER)),
        vec![FULL, FULL, None, None]
    );

    // A snapshot with whole-stack and layered entries together round-trips as well.
    let mut mixed = harness(CatalogLimits::default());
    mixed.insert_all(&req);
    let prefix = publish(&mixed.backend, &two_chunks('c').tokens());
    let other = two_chunks('c');
    mixed
        .catalog
        .insert_paged(
            ptr_runtime::KvPagedInsert {
                salt: &salt(),
                chain: &chain(0),
                request: &other,
                pages: layer_entry(&prefix, 0, 2, Vec::new()).pages,
                prefix: {
                    let mut p = prefix.clone();
                    p.page_digests = layer_pages(0, 2);
                    p
                },
                extra_dependencies: Vec::new(),
            },
            &Dead::default(),
        )
        .unwrap();
    let snapshot = mixed.catalog.export_entries().unwrap();
    assert!(snapshot.starts_with(b"PTRKVL03"));
    let mut again = harness(CatalogLimits::default());
    again.catalog.restore_entries(&snapshot).unwrap();
    assert_eq!(again.catalog.export_entries().unwrap(), snapshot);
    assert_eq!(again.catalog.stats().entries, LAYERS as usize + 1);
}

#[test]
fn a_whole_stack_catalog_still_exports_and_restores_v1() {
    let req = two_chunks('b');
    let mut source = harness(CatalogLimits::default());
    let base = publish(&source.backend, &req.tokens());
    let entry = layer_entry(&base, 0, 2, Vec::new());
    source
        .catalog
        .insert_paged(
            ptr_runtime::KvPagedInsert {
                salt: &salt(),
                chain: &chain(0),
                request: &req,
                prefix: entry.prefix,
                pages: entry.pages,
                extra_dependencies: Vec::new(),
            },
            &Dead::default(),
        )
        .unwrap();
    let snapshot = source.catalog.export_entries().unwrap();
    assert!(
        snapshot.starts_with(b"PTRKVC01"),
        "no layered entry, so the byte-identical v1 snapshot"
    );
    assert_eq!(restore(&snapshot).unwrap().entries, 1);
    // A v1 body with the v2 digest domain (or the reverse) is a checksum failure.
    let body = &snapshot[..snapshot.len() - 32];
    assert_eq!(restore(&reseal_v3(body)), Err(KvSnapshotError::BadChecksum));
}

#[test]
fn tampered_v3_snapshots_are_rejected_and_admit_nothing() {
    let req = two_chunks('b');
    let mut source = harness(CatalogLimits::default());
    source.insert_all(&req);
    let good = source.catalog.export_entries().unwrap();
    restore(&good).unwrap();

    for at in [0, 7, 9, 40, good.len() / 2, good.len() - 1] {
        let mut bad = good.clone();
        bad[at] ^= 1;
        let error = restore(&bad).unwrap_err();
        assert!(
            matches!(
                error,
                KvSnapshotError::BadChecksum | KvSnapshotError::BadMagic
            ),
            "{at}: {error:?}"
        );
    }
    assert_eq!(
        restore(&good[..good.len() - 1]).unwrap_err(),
        KvSnapshotError::BadChecksum
    );
    // A forger who recomputes the digest meets the strict decoder: the first entry's layer
    // tag sits after the two-chunk path (4 + 2 * 96 bytes), chain and lineage.
    let body = good[..good.len() - 32].to_vec();
    let tag_at = 8 + 4 + 4 + 2 * 96 + 64;
    assert_eq!(body[tag_at], 1, "the first entry is layered");
    let mut unknown = body.clone();
    unknown[tag_at] = 7;
    assert_eq!(
        restore(&reseal_v3(&unknown)),
        Err(KvSnapshotError::NonCanonical("unknown layer tag"))
    );
    // Declaring the entry whole-stack leaves two layer bytes where the page count starts:
    // the record no longer parses to a canonical entry.
    let mut shifted = body.clone();
    shifted[tag_at] = 0;
    assert!(restore(&reseal_v3(&shifted)).is_err());
    // Trailing bytes, a lying count and a truncated body.
    let mut trailing = body.clone();
    trailing.push(0);
    assert_eq!(
        restore(&reseal_v3(&trailing)),
        Err(KvSnapshotError::NonCanonical("trailing bytes"))
    );
    let mut lying = body.clone();
    lying[8..12].copy_from_slice(&(LAYERS as u32 + 1).to_le_bytes());
    assert_eq!(restore(&reseal_v3(&lying)), Err(KvSnapshotError::Truncated));
    // A v3 body under the v1 magic is rejected, and an unknown magic is BadMagic.
    let mut as_v1 = body.clone();
    as_v1[..8].copy_from_slice(b"PTRKVC01");
    assert!(restore(&reseal_v1(&as_v1)).is_err());
    let mut unknown_magic = body;
    unknown_magic[..8].copy_from_slice(b"PTRKVL04");
    assert_eq!(
        restore(&reseal_v3(&unknown_magic)),
        Err(KvSnapshotError::BadMagic)
    );
    // A failed restore leaves the catalog empty and usable.
    let mut target = harness(CatalogLimits::default());
    assert!(target.catalog.restore_entries(&good[..40]).is_err());
    assert_eq!(target.catalog.stats().entries, 0);
    target.catalog.restore_entries(&good).unwrap();
}

#[test]
fn a_restored_layered_entry_whose_dependency_died_misses_only_on_that_layer() {
    let req = two_chunks('b');
    let mut source = harness(CatalogLimits::default());
    source.insert_all(&req);
    let snapshot = source.catalog.export_entries().unwrap();
    let mut target = harness(CatalogLimits::default());
    target.catalog.restore_entries(&snapshot).unwrap();
    let fenced = Dead {
        fences: BTreeMap::from([(expert_id(), 2)]),
        ..Dead::default()
    };
    assert_eq!(target.depths(&req, &fenced), vec![FULL, FULL, None, None]);
    assert_eq!(target.catalog.stats().entries, 2);
}

/// The `(offset of the layer tag within an entry)` for the two-chunk requests of these tests:
/// the path (4 + 2 * 96 bytes), then chain and lineage.
const ENTRY_TAG_AT: usize = 4 + 2 * 96 + 64;

fn splice(body: &[u8], at: usize, remove: usize, insert: &[u8]) -> Vec<u8> {
    let mut out = body[..at].to_vec();
    out.extend_from_slice(insert);
    out.extend_from_slice(&body[at + remove..]);
    out
}

#[test]
fn v3_scope_bytes_are_strictly_canonical() {
    let req = two_chunks('b');
    let mut source = harness(CatalogLimits::default());
    source.insert_all(&req);
    let good = source.catalog.export_entries().unwrap();
    let body = good[..good.len() - 32].to_vec();
    let tag_at = 8 + 4 + ENTRY_TAG_AT;
    assert_eq!(body[tag_at], 1, "the first entry is layered");
    assert_eq!(body[tag_at + 3], 0, "a Full key encodes the scope tag 0");

    let mut unknown = body.clone();
    unknown[tag_at + 3] = 9;
    assert_eq!(
        restore(&reseal_v3(&unknown)),
        Err(KvSnapshotError::NonCanonical("unknown layer scope tag"))
    );
    // A span with start == end (and one with start > end) is not a scope.
    for (start, end) in [(2u16, 2u16), (3, 1)] {
        let mut insert = vec![1u8];
        insert.extend_from_slice(&start.to_le_bytes());
        insert.extend_from_slice(&end.to_le_bytes());
        let empty = splice(&body, tag_at + 3, 1, &insert);
        assert_eq!(
            restore(&reseal_v3(&empty)),
            Err(KvSnapshotError::NonCanonical("empty layer span"))
        );
    }
    // A whole-stack entry cannot carry a layer span.
    let mut insert = vec![0u8, 1];
    insert.extend_from_slice(&0u16.to_le_bytes());
    insert.extend_from_slice(&1u16.to_le_bytes());
    let whole_with_span = splice(&body, tag_at, 4, &insert);
    assert_eq!(
        restore(&reseal_v3(&whole_with_span)),
        Err(KvSnapshotError::NonCanonical(
            "whole-stack entry has a layer span"
        ))
    );
    // A truncated scope is a truncated body.
    let cut = splice(&body, tag_at + 3, 1, &[1, 0]);
    assert!(restore(&reseal_v3(&cut)).is_err());
}

#[test]
fn a_v2_snapshot_still_decodes_but_its_layered_entries_are_skipped() {
    let req = two_chunks('b');
    let mut source = harness(CatalogLimits::default());
    source.insert_all(&req);
    let good = source.catalog.export_entries().unwrap();
    let body = good[..good.len() - 32].to_vec();
    // Down-convert: drop the scope byte of every entry (all share the same path prefix).
    let header = body[12..12 + 4 + 2 * 96].to_vec();
    let mut starts = Vec::new();
    let mut at = 12;
    while at + header.len() <= body.len() {
        if body[at..at + header.len()] == header[..] {
            starts.push(at);
            at += header.len();
        } else {
            at += 1;
        }
    }
    assert_eq!(starts.len(), LAYERS as usize);
    let mut v2 = body.clone();
    for start in starts.iter().rev() {
        v2 = splice(&v2, start + ENTRY_TAG_AT + 3, 1, &[]);
    }
    v2[..8].copy_from_slice(b"PTRKVL02");
    let mut target = harness(CatalogLimits::default());
    let report = target.catalog.restore_entries(&reseal_v2(&v2)).unwrap();
    assert_eq!(report.entries, 0);
    assert_eq!(report.skipped_unscoped_layered, LAYERS as usize);
    assert_eq!(target.catalog.stats().entries, 0);
    assert_eq!(target.catalog.stats().restored_entries, 0);
    // A v3 restore reports no skips.
    let report = restore(&good).unwrap();
    assert_eq!(report.skipped_unscoped_layered, 0);
    assert_eq!(report.entries, LAYERS as usize);
}
