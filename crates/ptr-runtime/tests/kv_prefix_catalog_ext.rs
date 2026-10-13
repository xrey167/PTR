//! Prefix catalog extensions (doc 36 P2): per-unique-page byte accounting, base-model,
//! adapter and expert-fence dependencies, and the canonical entry snapshot.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ptr_model_api::{
    ByteTokenizer, PromptSegment, SegmentSource, SegmentedModelRequest, TokenId, Tokenizer,
};
use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryPagedKvBackend, InMemoryPagedKvCache,
    KvBackendError, KvContinuationExecutor, KvPageBinding, KvPageSize, KvSharedPrefix,
    KvTensorDType, KvTensorSchema, PagedKvTensorBackend, ReferenceCausalAttentionExecutor,
};
use ptr_runtime::{
    CatalogLimits, KvCapacityResource, KvCatalogError, KvChainId, KvDependency, KvDependencyKind,
    KvForkError, KvInsert, KvLivenessOracle, KvLookup, KvPageCharge, KvPagedInsert,
    KvPrefixCatalog, KvRequestPositions, KvSalt, KvSnapshotError, KV_CHUNK_TOKENS,
};
use ptr_types::{
    AdapterVersion, DeviceId, Generation, ModelVersion, PodId, PrincipalId, ProjectId, RequestId,
    Revision, SemanticContext,
};
use sha2::{Digest as _, Sha256};

const WIDTH: usize = 4;
const CHAIN: KvChainId = KvChainId([1; 32]);
const PAGE: u64 = 1000;

#[derive(Default)]
struct Dead(BTreeSet<(String, Generation)>);

impl KvLivenessOracle for Dead {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        !self
            .0
            .contains(&(dependency.identity.clone(), dependency.generation))
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

fn semantic(key: &str, generation: u64, text: &str) -> PromptSegment {
    PromptSegment::from_text(
        &ByteTokenizer::new(),
        SegmentSource::Semantic {
            key: key.to_owned(),
            generation: Generation(generation),
        },
        text,
    )
    .unwrap()
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

fn bytes_only(max_kv_page_bytes: u64) -> CatalogLimits {
    CatalogLimits {
        max_trie_nodes: 1 << 20,
        max_kv_page_bytes,
        max_metadata_bytes: 1 << 30,
    }
}

fn charges(prefix: &KvSharedPrefix) -> Vec<KvPageCharge> {
    let mut unique: Vec<[u8; 32]> = Vec::new();
    for digest in &prefix.page_digests {
        if !unique.contains(digest) {
            unique.push(*digest);
        }
    }
    unique
        .into_iter()
        .map(|digest| KvPageCharge {
            digest,
            bytes: PAGE,
        })
        .collect()
}

impl Harness {
    fn insert_with(
        &mut self,
        request: &SegmentedModelRequest,
        extra: Vec<KvDependency>,
        live: &dyn KvLivenessOracle,
    ) -> Result<KvSharedPrefix, KvCatalogError> {
        let prefix = publish(&self.backend, &request.tokens());
        self.catalog.insert_paged(
            KvPagedInsert {
                salt: &salt(),
                chain: &CHAIN,
                request,
                prefix: prefix.clone(),
                pages: charges(&prefix),
                extra_dependencies: extra,
            },
            live,
        )?;
        Ok(prefix)
    }

    fn insert(&mut self, request: &SegmentedModelRequest) -> KvSharedPrefix {
        self.insert_with(request, Vec::new(), &Dead::default())
            .unwrap()
    }

    fn lookup(
        &mut self,
        request: &SegmentedModelRequest,
        live: &dyn KvLivenessOracle,
    ) -> Option<ptr_runtime::KvCatalogHit> {
        self.catalog.lookup(
            KvLookup {
                salt: &salt(),
                chain: &CHAIN,
                request,
                positions: &KvRequestPositions::Canonical,
            },
            live,
        )
    }

    fn hits(&mut self, request: &SegmentedModelRequest) -> bool {
        match self.lookup(request, &Dead::default()) {
            Some(hit) => {
                let pin = hit.pin().clone();
                self.catalog.release(&pin);
                true
            }
            None => false,
        }
    }
}

#[test]
fn the_byte_bound_binds_on_a_bytes_only_workload_and_shared_pages_count_once() {
    // Three pages fit. Per-entry charging (the whole cache's bytes each) would refuse B.
    let mut h = harness(bytes_only(3 * PAGE));
    let (a, b, c, d) = (
        two_chunks('b'),
        two_chunks('c'),
        two_chunks('d'),
        two_chunks('e'),
    );
    let pa = h.insert(&a);
    let pb = h.insert(&b);
    assert_eq!(
        pa.page_digests[0], pb.page_digests[0],
        "first page is shared"
    );
    assert_ne!(pa.page_digests[1], pb.page_digests[1]);
    assert_eq!(h.catalog.stats().entries, 2);
    assert_eq!(h.catalog.stats().page_bytes, 3 * PAGE);

    // C needs one more page: A, the least recently used, is evicted and frees only its own
    // page; the shared one stays charged for B.
    h.insert(&c);
    let stats = h.catalog.stats();
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.page_bytes, 3 * PAGE);
    assert!(!h.hits(&a) && h.hits(&b) && h.hits(&c));

    // Pin everything: the bound still binds and fails closed.
    let hit_b = h.lookup(&b, &Dead::default()).unwrap();
    let hit_c = h.lookup(&c, &Dead::default()).unwrap();
    let refused = h.insert_with(&d, Vec::new(), &Dead::default());
    assert_eq!(
        refused.unwrap_err(),
        KvCatalogError::CapacityExhausted(KvCapacityResource::KvPageBytes)
    );
    assert_eq!(h.catalog.stats().page_bytes, 3 * PAGE);
    h.catalog.release(&hit_b.pin().clone());
    h.catalog.release(&hit_c.pin().clone());

    // Dropping every holder frees every byte.
    while h.catalog.evict_lru() {}
    assert_eq!(h.catalog.stats().page_bytes, 0);
    assert_eq!(h.catalog.stats().entries, 0);
}

#[test]
fn the_per_entry_insert_still_charges_its_lump_per_entry() {
    let mut h = harness(bytes_only(10 * PAGE));
    for second in ['b', 'c'] {
        let req = two_chunks(second);
        let prefix = publish(&h.backend, &req.tokens());
        h.catalog
            .insert(
                KvInsert {
                    salt: &salt(),
                    chain: &CHAIN,
                    request: &req,
                    prefix,
                    page_bytes: 2 * PAGE,
                    extra_dependencies: Vec::new(),
                },
                &Dead::default(),
            )
            .unwrap();
    }
    assert_eq!(h.catalog.stats().page_bytes, 4 * PAGE);
}

#[test]
fn page_charges_must_describe_the_prefix_and_agree_across_entries() {
    let mut h = harness(bytes_only(10 * PAGE));
    let req = two_chunks('b');
    let prefix = publish(&h.backend, &req.tokens());
    let mut wrong = charges(&prefix);
    wrong.pop();
    let refused = |h: &mut Harness, pages: Vec<KvPageCharge>| {
        h.catalog.insert_paged(
            KvPagedInsert {
                salt: &salt(),
                chain: &CHAIN,
                request: &req,
                prefix: prefix.clone(),
                pages,
                extra_dependencies: Vec::new(),
            },
            &Dead::default(),
        )
    };
    assert_eq!(
        refused(&mut h, wrong),
        Err(KvCatalogError::PageChargeMismatch)
    );
    let mut swapped = charges(&prefix);
    swapped.reverse();
    assert_eq!(
        refused(&mut h, swapped),
        Err(KvCatalogError::PageChargeMismatch)
    );
    assert_eq!(h.catalog.stats().entries, 0);
    assert_eq!(h.catalog.stats().page_bytes, 0);

    // One digest charged with two sizes.
    h.insert(&req);
    let other = two_chunks('c');
    let other_prefix = publish(&h.backend, &other.tokens());
    let mut resized = charges(&other_prefix);
    resized[0].bytes = PAGE + 1;
    let result = h.catalog.insert_paged(
        KvPagedInsert {
            salt: &salt(),
            chain: &CHAIN,
            request: &other,
            prefix: other_prefix,
            pages: resized,
            extra_dependencies: Vec::new(),
        },
        &Dead::default(),
    );
    assert_eq!(result, Err(KvCatalogError::PageChargeMismatch));
    assert_eq!(h.catalog.stats().page_bytes, 2 * PAGE);
}

fn expert(pod: &str, epoch: u64) -> KvDependency {
    KvDependency::expert_fenced(
        &ProjectId::from("proj"),
        &PodId::from(pod),
        Generation(1),
        [0; 32],
        epoch,
    )
}

#[test]
fn a_base_model_or_adapter_generation_bump_invalidates_only_its_dependents() {
    let mut h = harness(CatalogLimits::default());
    let (base, adapter, plain) = (two_chunks('b'), two_chunks('c'), two_chunks('d'));
    h.insert_with(
        &base,
        vec![KvDependency::base_model(
            "model:base",
            Generation(1),
            [7; 32],
        )],
        &Dead::default(),
    )
    .unwrap();
    h.insert_with(
        &adapter,
        vec![KvDependency::adapter(
            "adapter:lora",
            Generation(1),
            [8; 32],
        )],
        &Dead::default(),
    )
    .unwrap();
    h.insert(&plain);

    // A bump scoped to the wrong kind or an unrelated identity changes nothing.
    let bump = |identity: &str, generation: u64| {
        BTreeMap::from([(identity.to_owned(), Generation(generation))])
    };
    assert!(h
        .catalog
        .invalidate_kind_superseded(KvDependencyKind::Adapter, &bump("model:base", 2))
        .is_empty());
    assert!(h
        .catalog
        .invalidate_kind_superseded(KvDependencyKind::BaseModel, &bump("model:other", 2))
        .is_empty());
    assert!(h
        .catalog
        .invalidate_kind_superseded(KvDependencyKind::BaseModel, &bump("model:base", 1))
        .is_empty());
    assert!(h.hits(&base) && h.hits(&adapter) && h.hits(&plain));

    assert_eq!(
        h.catalog
            .invalidate_kind_superseded(KvDependencyKind::BaseModel, &bump("model:base", 2))
            .len(),
        1
    );
    assert!(!h.hits(&base) && h.hits(&adapter) && h.hits(&plain));

    // The exact (identity, generation) form reaches an adapter too.
    let retired = h.catalog.invalidate_inputs(&BTreeSet::from([(
        "adapter:lora".to_owned(),
        Generation(1),
    )]));
    assert_eq!(retired.len(), 1);
    assert!(!h.hits(&adapter) && h.hits(&plain));
}

#[test]
fn an_expert_fence_epoch_invalidates_only_older_epochs_of_that_expert() {
    let mut h = harness(CatalogLimits::default());
    let (old, current, other, unrecorded, plain) = (
        two_chunks('b'),
        two_chunks('c'),
        two_chunks('d'),
        two_chunks('e'),
        two_chunks('f'),
    );
    h.insert_with(&old, vec![expert("x", 3)], &Dead::default())
        .unwrap();
    h.insert_with(&current, vec![expert("x", 5)], &Dead::default())
        .unwrap();
    h.insert_with(&other, vec![expert("y", 3)], &Dead::default())
        .unwrap();
    h.insert_with(&unrecorded, vec![expert("x", 0)], &Dead::default())
        .unwrap();
    h.insert(&plain);

    let fence = |epoch: u64| BTreeMap::from([("expert:proj/x".to_owned(), epoch)]);
    assert!(h.catalog.invalidate_expert_fences(&fence(3)).len() == 1);
    // epoch 3 is not below 3, so only the entry that recorded nothing went.
    assert!(h.hits(&old) && h.hits(&current) && h.hits(&other) && !h.hits(&unrecorded));
    assert_eq!(h.catalog.invalidate_expert_fences(&fence(4)).len(), 1);
    assert!(!h.hits(&old) && h.hits(&current) && h.hits(&other) && h.hits(&plain));
}

/// Three entries with a shared first page, mixed dependency kinds and both charge kinds.
fn populated() -> Harness {
    let mut h = harness(CatalogLimits::default());
    h.insert_with(
        &request(vec![
            raw(&block('a')),
            semantic("price:sku", 1, &block('s')),
        ]),
        vec![
            KvDependency::base_model("model:base", Generation(1), [7; 32]),
            expert("x", 4),
        ],
        &Dead::default(),
    )
    .unwrap();
    h.insert(&two_chunks('c'));
    let req = two_chunks('d');
    let prefix = publish(&h.backend, &req.tokens());
    h.catalog
        .insert(
            KvInsert {
                salt: &salt(),
                chain: &CHAIN,
                request: &req,
                prefix,
                page_bytes: 123,
                extra_dependencies: vec![KvDependency::adapter(
                    "adapter:lora",
                    Generation(2),
                    [9; 32],
                )],
            },
            &Dead::default(),
        )
        .unwrap();
    h
}

#[test]
fn export_then_restore_round_trips_equal_and_never_carries_pages() {
    let source = populated();
    let snapshot = source.catalog.export_entries().unwrap();
    assert_eq!(
        snapshot,
        source.catalog.export_entries().unwrap(),
        "deterministic"
    );
    assert!(
        snapshot.len() < 8 * 1024,
        "keys and metadata only: {}",
        snapshot.len()
    );

    let mut target = harness(CatalogLimits::default());
    let report = target.catalog.restore_entries(&snapshot).unwrap();
    assert_eq!(report.entries, 3);
    let (src, dst) = (source.catalog.stats(), target.catalog.stats());
    assert_eq!(
        (src.nodes, src.entries, src.page_bytes, src.metadata_bytes),
        (dst.nodes, dst.entries, dst.page_bytes, dst.metadata_bytes)
    );
    assert_eq!(dst.restored_entries, 3);
    assert_eq!(snapshot, target.catalog.export_entries().unwrap());

    // A tombstoned entry is not exported.
    let mut source = source;
    source
        .catalog
        .invalidate_inputs(&BTreeSet::from([("price:sku".to_owned(), Generation(1))]));
    let smaller = source.catalog.export_entries().unwrap();
    let mut again = harness(CatalogLimits::default());
    assert_eq!(again.catalog.restore_entries(&smaller).unwrap().entries, 2);
}

#[test]
fn a_restored_entry_hits_and_forks_when_the_backend_still_holds_its_pages() {
    let source = populated();
    let snapshot = source.catalog.export_entries().unwrap();
    let mut target = harness(CatalogLimits::default());
    target.catalog.restore_entries(&snapshot).unwrap();
    // The target backend recomputes the same prefix: keys are content-addressed.
    let req = two_chunks('c');
    publish(&target.backend, &req.tokens());
    let hit = target
        .lookup(&req, &Dead::default())
        .expect("restored entry hits");
    let mut cache = fresh(&target.backend);
    let backend = target.backend.clone();
    let pin = target
        .catalog
        .fork(&backend, &mut cache, hit, &Dead::default())
        .unwrap();
    assert_eq!(cache.sequence_length, 32);
    target.catalog.release(&pin);
}

#[test]
fn a_restored_entry_whose_pages_are_gone_misses_cleanly() {
    let source = populated();
    let snapshot = source.catalog.export_entries().unwrap();
    let mut target = harness(CatalogLimits::default());
    target.catalog.restore_entries(&snapshot).unwrap();
    let req = two_chunks('c');
    let hit = target
        .lookup(&req, &Dead::default())
        .expect("admitted without a backend check");
    let mut cache = fresh(&target.backend);
    let backend = target.backend.clone();
    let error = target
        .catalog
        .fork(&backend, &mut cache, hit, &Dead::default())
        .unwrap_err();
    assert!(matches!(
        error,
        KvForkError::Backend(KvBackendError::PrefixNotFound)
    ));
    assert_eq!(cache.sequence_length, 0);
    assert_eq!(target.catalog.stats().backend_missing_rejections, 1);
    assert!(
        target.lookup(&req, &Dead::default()).is_none(),
        "dropped, not retried"
    );
    assert_eq!(target.catalog.stats().entries, 2);
}

#[test]
fn restored_pre_revocation_is_observed_in_the_catalog_and_misses_at_use() {
    let req = request(vec![
        raw(&block('a')),
        semantic("price:sku", 1, &block('s')),
    ]);
    let mut source = harness(CatalogLimits::default());
    source
        .insert_with(&req, Vec::new(), &Dead::default())
        .unwrap();
    let snapshot = source.catalog.export_entries().unwrap();

    // The input is revoked while the snapshot is at rest; the restore does not ask.
    let revoked = Dead(BTreeSet::from([("price:sku".to_owned(), Generation(1))]));
    let mut target = harness(CatalogLimits::default());
    target.catalog.restore_entries(&snapshot).unwrap();
    publish(&target.backend, &req.tokens());
    assert_eq!(
        target.catalog.stats().entries,
        1,
        "admitted despite the revocation"
    );
    assert!(target.lookup(&req, &revoked).is_none());
    let stats = target.catalog.stats();
    assert_eq!(stats.use_time_rejections, 1);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.page_bytes, 0);
    // Insert, by contrast, does ask.
    assert_eq!(
        target.insert_with(&req, Vec::new(), &revoked).unwrap_err(),
        KvCatalogError::DependencyNotLive
    );
}

fn reseal(body: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr.kv.catalog-snapshot.v1\0");
    hasher.update(body);
    let mut out = body.to_vec();
    out.extend_from_slice(&hasher.finalize());
    out
}

fn records(snapshot: &[u8]) -> Vec<u8> {
    snapshot[12..snapshot.len() - 32].to_vec()
}

fn assemble(count: u32, records: &[u8]) -> Vec<u8> {
    let mut body = b"PTRKVC01".to_vec();
    body.extend_from_slice(&count.to_le_bytes());
    body.extend_from_slice(records);
    reseal(&body)
}

fn single(second: char) -> Vec<u8> {
    let mut h = harness(CatalogLimits::default());
    h.insert(&two_chunks(second));
    h.catalog.export_entries().unwrap()
}

fn restore(snapshot: &[u8]) -> Result<(), KvSnapshotError> {
    harness(CatalogLimits::default())
        .catalog
        .restore_entries(snapshot)
        .map(|_| ())
}

#[test]
fn tampered_and_non_canonical_snapshots_are_rejected_and_admit_nothing() {
    let good = populated().catalog.export_entries().unwrap();
    restore(&good).unwrap();

    // Any flipped byte fails the digest (or the magic).
    for at in [0, 9, 40, good.len() / 2, good.len() - 1] {
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
    assert_eq!(restore(b"short").unwrap_err(), KvSnapshotError::BadMagic);
    assert_eq!(restore(&[]).unwrap_err(), KvSnapshotError::BadMagic);

    // A forger who recomputes the digest still hits the strict decoder.
    let (r1, r2) = (records(&single('b')), records(&single('c')));
    let (first, second) = if r1 < r2 { (&r1, &r2) } else { (&r2, &r1) };
    // Right order is accepted; the reverse is not canonical, nor is a repeat.
    let ordered = [first.as_slice(), second.as_slice()].concat();
    let reversed = [second.as_slice(), first.as_slice()].concat();
    let repeated = [first.as_slice(), first.as_slice()].concat();
    let sorted_ok = restore(&assemble(2, &ordered));
    let reverse_ok = restore(&assemble(2, &reversed));
    assert!(
        sorted_ok.is_ok() != reverse_ok.is_ok(),
        "exactly one of the two orders is canonical: {sorted_ok:?} {reverse_ok:?}"
    );
    assert!(matches!(
        restore(&assemble(2, &repeated)),
        Err(KvSnapshotError::NonCanonical(_))
    ));
    // Trailing bytes, a count that lies, a count over the bound, and a bad version.
    let mut trailing = records(&single('b'));
    trailing.push(0);
    assert_eq!(
        restore(&assemble(1, &trailing)),
        Err(KvSnapshotError::NonCanonical("trailing bytes"))
    );
    assert_eq!(
        restore(&assemble(2, &records(&single('b')))),
        Err(KvSnapshotError::Truncated)
    );
    assert_eq!(
        restore(&assemble(u32::MAX, &[])),
        Err(KvSnapshotError::LimitExceeded("entry count"))
    );
    let mut body = b"PTRKVC02".to_vec();
    body.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(restore(&reseal(&body)), Err(KvSnapshotError::BadMagic));
}

#[test]
fn restore_needs_an_empty_catalog_and_fails_closed_when_the_entries_do_not_fit() {
    let snapshot = populated().catalog.export_entries().unwrap();
    let mut busy = populated();
    assert_eq!(
        busy.catalog.restore_entries(&snapshot),
        Err(KvSnapshotError::NotEmpty)
    );

    let mut small = harness(bytes_only(PAGE));
    let error = small.catalog.restore_entries(&snapshot).unwrap_err();
    assert_eq!(
        error,
        KvSnapshotError::Admission(KvCatalogError::CapacityExhausted(
            KvCapacityResource::KvPageBytes
        ))
    );
    let stats = small.catalog.stats();
    assert_eq!(
        (
            stats.entries,
            stats.nodes,
            stats.page_bytes,
            stats.metadata_bytes
        ),
        (0, 0, 0, 0)
    );
    // The catalog is still usable afterwards.
    small.catalog.restore_entries(&single('b')).unwrap_err();

    let mut few = harness(CatalogLimits {
        max_trie_nodes: 2,
        ..CatalogLimits::default()
    });
    assert_eq!(
        few.catalog.restore_entries(&snapshot),
        Err(KvSnapshotError::LimitExceeded("entry count"))
    );
}
