//! Exact prefix catalog with retirement (doc 36, phase P2).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, PromptSegment, SegmentSource, SegmentedModelRequest, TokenId, Tokenizer,
};
use ptr_pods::{
    BaseModelDigest, BaseModelId, BaseModelRef, ExpertCommitRecord, ExpertLevel, ExpertPodSpec,
    ExpertProposal, ExpertState, ExpertTarget, ExpertTargetInit, ManifestDigest, ProposerRef,
    RationaleDigest, RetiredReason, StructuralOp,
};
use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryPagedKvBackend, InMemoryPagedKvCache,
    KvBackendError, KvContinuationExecutor, KvPageBinding, KvPageSize, KvTensorDType,
    KvTensorSchema, KvWitnessRejection, PagedKvTensorBackend, ReferenceCausalAttentionExecutor,
};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{
    chunk_keys, expert_operand_identities, CatalogLimits, KvCapacityResource, KvCatalogError,
    KvChainId, KvDependency, KvForkError, KvInsert, KvLivenessOracle, KvLookup, KvPrefixCatalog,
    KvRequestPositions, KvSalt, PtrRuntime, KV_CHUNK_TOKENS,
};
use ptr_semdb::SemanticDelta;
use ptr_types::{
    AdapterVersion, CapsuleId, CommitIndex, DeviceId, Generation, KnowledgeObjectId, ModelVersion,
    PodId, PrincipalId, ProjectId, RequestId, Revision, SemanticContext, VerificationLevel,
};
use ptr_verifier::VerificationStatus;

#[path = "common/semantic.rs"]
mod semantic_common;

const WIDTH: usize = 4;

/// Liveness the test scripts: everything is live except the named generations.
#[derive(Default)]
struct Dead(BTreeSet<(String, Generation)>);

impl Dead {
    fn kill(&mut self, identity: &str, generation: u64) {
        self.0.insert((identity.to_owned(), Generation(generation)));
    }
}

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

fn binding(principal: &str, generation: u64) -> KvPageBinding {
    KvPageBinding {
        model: ModelVersion::from("base-v1"),
        adapter: AdapterVersion::from("none"),
        generation: Generation(generation),
        execution_manifest: [3; 32],
        principal: PrincipalId::from(principal),
        device: DeviceId::from("cpu"),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::DEFAULT,
    }
}

fn salt(principal: &str, project: &str) -> KvSalt {
    KvSalt::derive(
        &PrincipalId::from(principal),
        &ProjectId::from(project),
        &ModelVersion::from("base-v1"),
        ByteTokenizer::new().identity(),
        "f32-cpu-reference",
        "paged-kv-v1",
    )
}

const CHAIN: KvChainId = KvChainId([1; 32]);

fn context(revision: u64) -> SemanticContext {
    SemanticContext {
        revision: Revision(revision),
        entries: Vec::new(),
    }
}

fn request(segments: Vec<PromptSegment>) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from("r"),
        context(0),
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

fn memory(id: &str, generation: u64, text: &str) -> PromptSegment {
    PromptSegment::from_text(
        &ByteTokenizer::new(),
        SegmentSource::Memory {
            id: KnowledgeObjectId::from(id),
            generation: Generation(generation),
        },
        text,
    )
    .unwrap()
}

/// 16 bytes of text, so one segment is exactly one chunk.
fn block(label: char) -> String {
    label.to_string().repeat(KV_CHUNK_TOKENS)
}

fn embed(token: TokenId) -> Vec<f32> {
    (0..WIDTH)
        .map(|i| ((token.0 as f32) + i as f32) / 97.0 - 0.5)
        .collect()
}

fn executor() -> ReferenceCausalAttentionExecutor {
    ReferenceCausalAttentionExecutor::new(CausalAttentionWeights::identity(WIDTH).unwrap(), 2)
        .unwrap()
}

fn fresh(backend: &InMemoryPagedKvBackend) -> InMemoryPagedKvCache {
    backend
        .allocate_bound(schema(), 128, binding("alice", 1))
        .unwrap()
}

/// Prefill `tokens` and publish the full pages; returns the prefix and its page bytes.
fn prefill_and_publish(
    backend: &InMemoryPagedKvBackend,
    tokens: &[TokenId],
) -> (ptr_pods::KvSharedPrefix, u64) {
    let mut cache = fresh(backend);
    executor()
        .prefill(
            backend,
            &mut cache,
            CausalAttentionInput {
                tokens: tokens.iter().copied().map(embed).collect(),
            },
        )
        .unwrap();
    let prefix = backend.publish_prefix(&mut cache).unwrap();
    let bytes = backend.page_metrics(&cache).resident_bytes;
    (prefix, bytes)
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

impl Harness {
    fn insert(
        &mut self,
        salt: &KvSalt,
        request: &SegmentedModelRequest,
        live: &dyn KvLivenessOracle,
    ) -> Result<ptr_pods::KvSharedPrefix, KvCatalogError> {
        let mut tokens = request.tokens();
        tokens.truncate(tokens.len() / KV_CHUNK_TOKENS * KV_CHUNK_TOKENS);
        let (prefix, page_bytes) = prefill_and_publish(&self.backend, &tokens);
        self.catalog.insert(
            KvInsert {
                salt,
                chain: &CHAIN,
                request,
                prefix: prefix.clone(),
                page_bytes,
                extra_dependencies: Vec::new(),
            },
            live,
        )?;
        Ok(prefix)
    }

    fn lookup(
        &mut self,
        salt: &KvSalt,
        request: &SegmentedModelRequest,
        live: &dyn KvLivenessOracle,
    ) -> Option<ptr_runtime::KvCatalogHit> {
        self.catalog.lookup(
            KvLookup {
                salt,
                chain: &CHAIN,
                request,
                positions: &KvRequestPositions::Canonical,
            },
            live,
        )
    }

    fn retire(&self, keys: &[ptr_pods::KvPrefixKey]) {
        for key in keys {
            self.backend.retire_prefix(key).unwrap();
        }
    }
}

#[test]
fn keys_are_computable_from_tokens_before_any_prefill() {
    let s = salt("alice", "p1");
    let tokens = ByteTokenizer::new().encode(&format!("{}{}tail", block('a'), block('b')));
    let keys = chunk_keys(&s, &tokens);
    assert_eq!(keys.len(), 2, "a trailing partial chunk has no key");
    assert_eq!(keys[1].parent, keys[0].node_digest());
    let longer =
        ByteTokenizer::new().encode(&format!("{}{}{}", block('a'), block('b'), block('c')));
    let longer_keys = chunk_keys(&s, &longer);
    assert_eq!(&longer_keys[..2], &keys[..], "a prefix has the same keys");
    assert_eq!(keys, chunk_keys(&s, &tokens), "deterministic");
    let other = chunk_keys(&salt("alice", "p2"), &tokens);
    assert_ne!(other[0], keys[0], "salts diverge at the first node");
}

#[test]
fn raw_identical_prefix_hits_and_the_tail_does_not_matter() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    h.insert(
        &s,
        &request(vec![raw(&format!("{}{}one", block('a'), block('b')))]),
        &live,
    )
    .unwrap();
    let hit = h
        .lookup(
            &s,
            &request(vec![raw(&format!("{}{}two!!", block('a'), block('b')))]),
            &live,
        )
        .expect("same raw prefix, different tail");
    assert_eq!(hit.matched_tokens(), 2 * KV_CHUNK_TOKENS);
    h.catalog.release(hit.pin());
}

#[test]
fn another_principal_or_project_never_hits_the_same_tokens() {
    let mut h = harness(CatalogLimits::default());
    let live = Dead::default();
    let req = request(vec![raw(&format!("{}{}", block('a'), block('b')))]);
    h.insert(&salt("alice", "p1"), &req, &live).unwrap();
    assert!(
        h.lookup(&salt("bob", "p1"), &req, &live).is_none(),
        "cross principal"
    );
    assert!(
        h.lookup(&salt("alice", "p2"), &req, &live).is_none(),
        "cross project"
    );
    assert!(h.lookup(&salt("alice", "p1"), &req, &live).is_some());
}

#[test]
fn identical_tokens_with_different_lineage_hold_separate_entries() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let text = block('s');
    let gen1 = request(vec![raw(&block('a')), semantic("price:sku", 1, &text)]);
    let gen2 = request(vec![raw(&block('a')), semantic("price:sku", 2, &text)]);
    assert_eq!(gen1.tokens(), gen2.tokens());
    h.insert(&s, &gen1, &live).unwrap();
    assert!(h.lookup(&s, &gen2, &live).is_none(), "no cross-lineage hit");
    h.insert(&s, &gen2, &live).unwrap();
    assert_eq!(h.catalog.stats().entries, 2);
    let first = h.lookup(&s, &gen1, &live).unwrap();
    let second = h.lookup(&s, &gen2, &live).unwrap();
    // Identical tokens publish identical pages, so both entries name one backend prefix.
    assert_eq!(first.prefix().key, second.prefix().key);
    assert_eq!(first.matched_tokens(), 2 * KV_CHUNK_TOKENS);
    h.catalog.release(first.pin());
    h.catalog.release(second.pin());
    // Retiring one lineage must not retire the backend prefix the other still uses.
    let retire = h
        .catalog
        .invalidate_inputs(&BTreeSet::from([("price:sku".to_owned(), Generation(1))]));
    assert!(retire.is_empty());
    assert!(h.lookup(&s, &gen2, &live).is_some());
    // A raw-only request with the same tokens is a third lineage and also misses.
    let as_raw = request(vec![raw(&format!("{}{}", block('a'), text))]);
    assert!(h.lookup(&s, &as_raw, &live).is_none());
    // A different semantic revision is a different lineage too.
    let moved = SegmentedModelRequest::new(
        RequestId::from("r"),
        context(7),
        ByteTokenizer::new().identity().clone(),
        gen1.segments().to_vec(),
    )
    .unwrap();
    assert!(h.lookup(&s, &moved, &live).is_none());
}

#[test]
fn a_revoked_generation_stops_hitting_and_its_prefix_is_retired() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![
        raw(&block('a')),
        semantic("price:sku", 1, &block('s')),
    ]);
    let prefix = h.insert(&s, &req, &live).unwrap();
    let hit = h.lookup(&s, &req, &live).unwrap();
    h.catalog.release(hit.pin());

    // An unrelated generation of the same identity leaves the entry alone.
    let unrelated = BTreeSet::from([("price:sku".to_owned(), Generation(9))]);
    assert!(h.catalog.invalidate_inputs(&unrelated).is_empty());

    let revoked = BTreeSet::from([("price:sku".to_owned(), Generation(1))]);
    let retire = h.catalog.invalidate_inputs(&revoked);
    assert_eq!(retire, vec![prefix.key.clone()]);
    h.retire(&retire);
    assert!(h.lookup(&s, &req, &live).is_none());
    assert_eq!(h.catalog.stats().entries, 0);
    assert_eq!(h.catalog.stats().nodes, 0, "emptied nodes are pruned");
    assert_eq!(
        h.backend.retire_prefix(&prefix.key),
        Err(KvBackendError::PrefixNotFound)
    );
}

#[test]
fn a_superseded_generation_stops_hitting_through_the_supersession_hook() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![raw(&block('a')), memory("mem:1", 1, &block('m'))]);
    h.insert(&s, &req, &live).unwrap();
    let retire = h
        .catalog
        .invalidate_superseded(&BTreeMap::from([("mem:1".to_owned(), Generation(2))]));
    assert_eq!(retire.len(), 1);
    assert!(h.lookup(&s, &req, &live).is_none());
}

#[test]
fn liveness_is_rechecked_at_use_when_invalidation_was_lost() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let mut dead = Dead::default();
    let req = request(vec![
        raw(&block('a')),
        semantic("price:sku", 1, &block('s')),
    ]);
    h.insert(&s, &req, &dead).unwrap();
    assert!(h.lookup(&s, &req, &dead).is_some());
    // The revocation committed but `invalidate_inputs` never ran (lag, or lost in restore).
    dead.kill("price:sku", 1);
    assert!(h.lookup(&s, &req, &dead).is_none());
    let stats = h.catalog.stats();
    assert_eq!(stats.use_time_rejections, 1);
    assert_eq!(
        h.catalog.take_retirements().len(),
        1,
        "queued for retirement"
    );

    // An insert racing a revocation is refused outright.
    let again = h.insert(&s, &req, &dead);
    assert_eq!(again.unwrap_err(), KvCatalogError::DependencyNotLive);
}

#[test]
fn a_dependency_of_an_earlier_segment_guards_a_later_chunk() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let mut dead = Dead::default();
    // The semantic segment is only in chunk 1; the entry covers chunks 1 and 2.
    let req = request(vec![semantic("doc:a", 1, &block('a')), raw(&block('b'))]);
    h.insert(&s, &req, &dead).unwrap();
    dead.kill("doc:a", 1);
    assert!(h.lookup(&s, &req, &dead).is_none());
}

#[test]
fn forged_stale_and_cancelled_witnesses_cannot_fork() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![
        raw(&block('a')),
        semantic("price:sku", 1, &block('s')),
    ]);
    let prefix = h.insert(&s, &req, &live).unwrap();

    // Stale: a genuine witness held across a dependency invalidation.
    let hit = h.lookup(&s, &req, &live).unwrap();
    let pin = hit.pin().clone();
    assert!(h.catalog.pin_is_live(&pin));
    h.catalog
        .invalidate_inputs(&BTreeSet::from([("price:sku".to_owned(), Generation(1))]));
    assert!(!h.catalog.pin_is_live(&pin), "the pinned fork is cancelled");
    let mut target = fresh(&h.backend);
    let (witness, hit_prefix, _) = hit.into_parts();
    assert_eq!(
        h.backend
            .fork_prefix_admitted(&mut target, &hit_prefix, witness),
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::Revoked
        ))
    );
    assert_eq!(target.sequence_length, 0);
    assert!(h.catalog.release(&pin));
    assert_eq!(h.catalog.stats().entries, 0, "reclaimed on last release");

    // Forged: a witness from a different backend's authority.
    let other_backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let other_catalog_authority = other_backend.take_fork_authority().unwrap();
    let mut other =
        KvPrefixCatalog::new(other_catalog_authority, CatalogLimits::default()).unwrap();
    let (other_prefix, bytes) = prefill_and_publish(&other_backend, &req.tokens()[..32]);
    other
        .insert(
            KvInsert {
                salt: &s,
                chain: &CHAIN,
                request: &req,
                prefix: other_prefix,
                page_bytes: bytes,
                extra_dependencies: Vec::new(),
            },
            &live,
        )
        .unwrap();
    let foreign = other
        .lookup(
            KvLookup {
                salt: &s,
                chain: &CHAIN,
                request: &req,
                positions: &KvRequestPositions::Canonical,
            },
            &live,
        )
        .unwrap();
    let mut target = fresh(&h.backend);
    let again = h.insert(&s, &req, &live).unwrap();
    assert_eq!(
        again.key, prefix.key,
        "same tokens publish the same backend key"
    );
    let (foreign_witness, _, _) = foreign.into_parts();
    assert_eq!(
        h.backend
            .fork_prefix_admitted(&mut target, &prefix, foreign_witness),
        Err(KvBackendError::ForkWitnessRejected(
            KvWitnessRejection::ForeignBackend
        ))
    );
}

#[test]
fn fork_rechecks_liveness_and_releases_the_pin_on_refusal() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let mut dead = Dead::default();
    let req = request(vec![
        raw(&block('a')),
        semantic("price:sku", 1, &block('s')),
    ]);
    h.insert(&s, &req, &dead).unwrap();
    let hit = h.lookup(&s, &req, &dead).unwrap();
    dead.kill("price:sku", 1);
    let mut target = fresh(&h.backend);
    let refused = h.catalog.fork(&h.backend, &mut target, hit, &dead);
    assert!(matches!(refused, Err(KvForkError::Cancelled)));
    assert_eq!(target.sequence_length, 0);
    assert_eq!(
        h.catalog.stats().entries,
        0,
        "pin released, entry reclaimed"
    );
}

#[test]
fn pinned_entries_are_never_evicted_and_saturation_fails_closed() {
    let limits = CatalogLimits {
        max_trie_nodes: 2,
        ..CatalogLimits::default()
    };
    let mut h = harness(limits);
    let s = salt("alice", "p1");
    let live = Dead::default();
    let first = request(vec![raw(&format!("{}{}", block('a'), block('b')))]);
    h.insert(&s, &first, &live).unwrap();
    let pinned = h.lookup(&s, &first, &live).unwrap();

    let second = request(vec![raw(&format!("{}{}", block('c'), block('d')))]);
    let error = h.insert(&s, &second, &live).unwrap_err();
    assert_eq!(
        error,
        KvCatalogError::CapacityExhausted(KvCapacityResource::TrieNodes)
    );
    let stats = h.catalog.stats();
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.capacity_refusals, 1);
    assert_eq!(stats.nodes, 2);
    assert!(h.catalog.pin_is_live(pinned.pin()));

    // Once the fork ends the entry is evictable and the insert fits.
    h.catalog.release(pinned.pin());
    h.insert(&s, &second, &live).unwrap();
    assert_eq!(h.catalog.stats().evictions, 1);
    assert!(h.lookup(&s, &first, &live).is_none());
    assert!(h.lookup(&s, &second, &live).is_some());
}

#[test]
fn unique_prompts_beyond_every_limit_evict_and_stay_within_bounds() {
    let one_entry_pages = {
        let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
        prefill_and_publish(
            &backend,
            &ByteTokenizer::new().encode(&format!("{}{}", block('a'), block('b'))),
        )
        .1
    };
    let cases = [
        CatalogLimits {
            max_trie_nodes: 4,
            ..CatalogLimits::default()
        },
        CatalogLimits {
            max_kv_page_bytes: 2 * one_entry_pages,
            ..CatalogLimits::default()
        },
        CatalogLimits {
            max_metadata_bytes: 2 * 1024,
            ..CatalogLimits::default()
        },
    ];
    for limits in cases {
        let mut h = harness(limits);
        let s = salt("alice", "p1");
        let live = Dead::default();
        for i in 0..12u8 {
            let a = char::from(b'A' + i);
            let b = char::from(b'a' + i);
            let req = request(vec![raw(&format!("{}{}", block(a), block(b)))]);
            h.insert(&s, &req, &live).unwrap();
            let stats = h.catalog.stats();
            assert!(stats.nodes <= limits.max_trie_nodes);
            assert!(stats.page_bytes <= limits.max_kv_page_bytes);
            assert!(stats.metadata_bytes <= limits.max_metadata_bytes);
        }
        assert!(h.catalog.stats().evictions >= 1, "{limits:?}");
        // Evicted prefixes are queued so the backend can retire them.
        let queued = h.catalog.take_retirements();
        assert_eq!(queued.len() as u64, h.catalog.stats().evictions);
        h.retire(&queued);
    }
}

#[test]
fn defaults_are_the_r005_limits() {
    let limits = CatalogLimits::default();
    assert_eq!(limits.max_trie_nodes, 4096);
    assert_eq!(limits.max_kv_page_bytes, 128 * 1024 * 1024);
    assert_eq!(limits.max_metadata_bytes, 8 * 1024 * 1024);
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let zero = CatalogLimits {
        max_trie_nodes: 0,
        ..limits
    };
    assert_eq!(
        KvPrefixCatalog::new(backend.take_fork_authority().unwrap(), zero).unwrap_err(),
        KvCatalogError::InvalidLimits
    );
}

#[test]
fn only_canonical_positions_hit_or_insert() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![raw(&format!("{}{}", block('a'), block('b')))]);
    let prefix = h.insert(&s, &req, &live).unwrap();
    for positions in [
        KvRequestPositions::Offset(16),
        KvRequestPositions::Explicit((0..32).collect()),
    ] {
        let hit = h.catalog.lookup(
            KvLookup {
                salt: &s,
                chain: &CHAIN,
                request: &req,
                positions: &positions,
            },
            &live,
        );
        assert!(hit.is_none(), "{positions:?}");
    }
    assert_eq!(h.catalog.stats().noncanonical_misses, 2);
    assert!(h.lookup(&s, &req, &live).is_some());

    let mut shifted = prefix;
    shifted.position_offset = 16;
    let error = h
        .catalog
        .insert(
            KvInsert {
                salt: &salt("alice", "p9"),
                chain: &CHAIN,
                request: &req,
                prefix: shifted,
                page_bytes: 1,
                extra_dependencies: Vec::new(),
            },
            &live,
        )
        .unwrap_err();
    assert_eq!(error, KvCatalogError::NonCanonicalPosition);
}

#[test]
fn a_chain_mismatch_never_hits() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![raw(&format!("{}{}", block('a'), block('b')))]);
    h.insert(&s, &req, &live).unwrap();
    let hit = h.catalog.lookup(
        KvLookup {
            salt: &s,
            chain: &KvChainId([2; 32]),
            request: &req,
            positions: &KvRequestPositions::Canonical,
        },
        &live,
    );
    assert!(hit.is_none());
}

#[test]
fn catalog_fork_then_decode_is_bitwise_identical_to_full_prefill() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let prompt = format!("{}{}!", block('a'), block('b'));
    let req = request(vec![raw(&prompt)]);
    h.insert(&s, &req, &live).unwrap();

    let hit = h.lookup(&s, &req, &live).unwrap();
    assert_eq!(hit.matched_tokens(), 32);
    let mut forked = fresh(&h.backend);
    let pin = h.catalog.fork(&h.backend, &mut forked, hit, &live).unwrap();
    let tokens = req.tokens();
    let decoded = executor()
        .decode(
            &h.backend,
            &mut forked,
            CausalAttentionInput {
                tokens: vec![embed(tokens[32])],
            },
        )
        .unwrap();

    let mut recomputed = fresh(&h.backend);
    let full = executor()
        .prefill(
            &h.backend,
            &mut recomputed,
            CausalAttentionInput {
                tokens: tokens.iter().copied().map(embed).collect(),
            },
        )
        .unwrap();
    let actual = decoded.hidden_states.last().unwrap();
    let expected = full.hidden_states.last().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert!(h.catalog.release(&pin));
}

fn lifecycle_runtime() -> (PtrRuntime, InMemoryPagedKvBackend) {
    let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let catalog = KvPrefixCatalog::new(
        backend.take_fork_authority().unwrap(),
        CatalogLimits::default(),
    )
    .unwrap()
    .with_retirer(Arc::new(backend.clone()));
    runtime.enable_kv_prefix_catalog(catalog).unwrap();
    (runtime, backend)
}

fn runtime_insert(
    runtime: &mut PtrRuntime,
    backend: &InMemoryPagedKvBackend,
    s: &KvSalt,
    req: &SegmentedModelRequest,
) -> ptr_pods::KvSharedPrefix {
    let tokens = req.tokens();
    let (prefix, page_bytes) = prefill_and_publish(backend, &tokens[..tokens.len() / 16 * 16]);
    runtime
        .kv_prefix_insert(KvInsert {
            salt: s,
            chain: &CHAIN,
            request: req,
            prefix: prefix.clone(),
            page_bytes,
            extra_dependencies: Vec::new(),
        })
        .unwrap()
        .unwrap();
    prefix
}

fn runtime_lookup(
    runtime: &mut PtrRuntime,
    s: &KvSalt,
    req: &SegmentedModelRequest,
) -> Option<ptr_runtime::KvCatalogHit> {
    runtime
        .kv_prefix_lookup(KvLookup {
            salt: s,
            chain: &CHAIN,
            request: req,
            positions: &KvRequestPositions::Canonical,
        })
        .unwrap()
}

#[test]
fn committed_revocation_and_supersession_invalidate_and_retire_on_the_backend() {
    let (mut runtime, backend) = lifecycle_runtime();
    let s = salt("alice", "p1");
    runtime
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p1"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    let req = request(vec![raw(&block('a')), memory("capsule:a", 1, &block('m'))]);
    let prefix = runtime_insert(&mut runtime, &backend, &s, &req);
    let hit = runtime_lookup(&mut runtime, &s, &req).expect("live generation hits");
    assert!(runtime.kv_prefix_release(hit.pin()));

    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();
    assert!(runtime_lookup(&mut runtime, &s, &req).is_none());
    assert_eq!(
        backend.retire_prefix(&prefix.key),
        Err(KvBackendError::PrefixNotFound),
        "the hook already retired it on the backend"
    );

    // Supersession: generation 2 replaces 1.
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(1),
            new: Generation(2),
        })
        .unwrap();
    let req2 = request(vec![raw(&block('c')), memory("capsule:a", 2, &block('n'))]);
    let prefix2 = runtime_insert(&mut runtime, &backend, &s, &req2);
    assert!(runtime_lookup(&mut runtime, &s, &req2).is_some());
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(2),
            new: Generation(3),
        })
        .unwrap();
    assert!(runtime_lookup(&mut runtime, &s, &req2).is_none());
    assert_eq!(
        backend.retire_prefix(&prefix2.key),
        Err(KvBackendError::PrefixNotFound)
    );
    assert_eq!(
        runtime.kv_prefix_catalog().unwrap().stats().retire_failures,
        0
    );
}

#[test]
fn only_a_committed_semantic_delta_invalidates_a_refused_one_does_not() {
    use semantic_common::{finding, host_write_now, pass, report, scripted_runtime};
    let (mut runtime, verifier) = scripted_runtime(
        RequiredVerification::Deterministic,
        pass(VerificationLevel::Deterministic),
    );
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    runtime
        .enable_kv_prefix_catalog(
            KvPrefixCatalog::new(
                backend.take_fork_authority().unwrap(),
                CatalogLimits::default(),
            )
            .unwrap()
            .with_retirer(Arc::new(backend.clone())),
        )
        .unwrap();
    let write = |value: &str| {
        let mut delta = SemanticDelta::default();
        delta.upserts.insert("price".into(), value.into());
        delta
    };
    host_write_now(&mut runtime, write("10")).unwrap();

    let s = salt("alice", "p1");
    let req = request(vec![raw(&block('a')), semantic("price", 1, &block('p'))]);
    let prefix = runtime_insert(&mut runtime, &backend, &s, &req);
    let hit = runtime_lookup(&mut runtime, &s, &req).expect("key is in the committed snapshot");
    runtime.kv_prefix_release(hit.pin());

    // A write the verifier refuses is prepared but never committed: nothing is invalidated.
    let mut refused = report(VerificationStatus::Pass, VerificationLevel::Deterministic);
    refused.findings.push(finding("constraint", true));
    verifier.set(refused);
    assert!(host_write_now(&mut runtime, write("11")).is_err());
    let hit = runtime_lookup(&mut runtime, &s, &req).expect("discarded overlay changed nothing");
    runtime.kv_prefix_release(hit.pin());
    assert_eq!(runtime.kv_prefix_catalog().unwrap().stats().entries, 1);

    verifier.set(pass(VerificationLevel::Deterministic));
    host_write_now(&mut runtime, write("12")).unwrap();
    assert!(runtime_lookup(&mut runtime, &s, &req).is_none());
    assert_eq!(
        backend.retire_prefix(&prefix.key),
        Err(KvBackendError::PrefixNotFound)
    );
}

#[test]
fn runtime_without_a_catalog_reports_it_and_the_hook_is_inert() {
    let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
    let s = salt("alice", "p1");
    let req = request(vec![raw(&block('a'))]);
    assert!(runtime
        .kv_prefix_lookup(KvLookup {
            salt: &s,
            chain: &CHAIN,
            request: &req,
            positions: &KvRequestPositions::Canonical,
        })
        .is_err());
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "x".into(),
            generation: Generation(1),
        })
        .unwrap();
}

#[test]
fn a_tombstoned_pinned_slot_is_refused_not_overwritten_and_accounting_stays_symmetric() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![
        raw(&block('a')),
        semantic("price:sku", 1, &block('s')),
    ]);
    h.insert(&s, &req, &live).unwrap();
    let one = h.catalog.stats();
    let hit = h.lookup(&s, &req, &live).unwrap();
    let pin = hit.pin().clone();

    // Repeat the cycle: the re-insert of the same slot is refused while pinned.
    for _ in 0..3 {
        h.catalog
            .invalidate_identities(&BTreeSet::from(["price:sku".to_owned()]));
        assert!(!h.catalog.pin_is_live(&pin));
        assert_eq!(
            h.insert(&s, &req, &live).unwrap_err(),
            KvCatalogError::SlotDraining
        );
        let during = h.catalog.stats();
        assert_eq!(during.page_bytes, one.page_bytes);
        assert_eq!(during.metadata_bytes, one.metadata_bytes);
        assert_eq!(during.tombstoned_pinned_entries, 1);
    }
    assert!(h.catalog.release(&pin), "the original pin still releases");
    let drained = h.catalog.stats();
    assert_eq!(
        (drained.page_bytes, drained.metadata_bytes, drained.nodes),
        (0, 0, 0)
    );

    h.insert(&s, &req, &live).unwrap();
    let again = h.catalog.stats();
    assert_eq!(again.page_bytes, one.page_bytes);
    assert_eq!(again.metadata_bytes, one.metadata_bytes);
    assert!(h.lookup(&s, &req, &live).is_some());
}

fn expert_target(pod: &str, state: ExpertState) -> ExpertTarget {
    ExpertTarget::new(ExpertTargetInit {
        project: ProjectId::from("proj"),
        pod: PodId::from(pod),
        generation: Generation(1),
        level: ExpertLevel::Token {
            layer: 3,
            index: u16::from(pod.as_bytes()[0]),
        },
        base_model: base_model(),
        manifest_digest: ManifestDigest::new([1; 32]).unwrap(),
        state,
        state_version: 1,
    })
    .unwrap()
}

fn base_model() -> BaseModelRef {
    BaseModelRef::new(
        BaseModelId::new("base-a").unwrap(),
        BaseModelDigest::new([9; 32]).unwrap(),
        1,
    )
    .unwrap()
}

fn record(op: StructuralOp) -> Vec<u8> {
    let proposal = ExpertProposal::new(
        ProjectId::from("proj"),
        ProposerRef::new(ProjectId::from("proj"), PodId::from("meta"), Generation(1)).unwrap(),
        CommitIndex(10),
        op,
        RationaleDigest::new([7; 32]).unwrap(),
    )
    .unwrap();
    ExpertCommitRecord::new(proposal, [5; 32], 1)
        .unwrap()
        .encode_canonical()
}

fn merged_result() -> ExpertPodSpec {
    ExpertPodSpec::new(
        ProjectId::from("proj"),
        PodId::from("merged"),
        ManifestDigest::new([2; 32]).unwrap(),
        ExpertLevel::Token { layer: 3, index: 1 },
        base_model(),
        Generation(1),
    )
    .unwrap()
}

#[test]
fn operand_identities_come_from_the_decoded_record_of_each_operation() {
    let x = expert_target("x", ExpertState::Serving);
    let y = expert_target("y", ExpertState::Serving);
    let retire = record(StructuralOp::Retire {
        target: expert_target("x", ExpertState::Serving),
        reason: RetiredReason::Superseded,
    });
    assert_eq!(
        expert_operand_identities(&retire),
        Some(vec!["expert:proj/x".to_owned()])
    );
    let prune = record(StructuralOp::Prune {
        targets: vec![
            expert_target("x", ExpertState::Serving),
            expert_target("y", ExpertState::Serving),
        ],
    });
    assert_eq!(
        expert_operand_identities(&prune),
        Some(vec!["expert:proj/x".to_owned(), "expert:proj/y".to_owned()])
    );
    let merge = record(StructuralOp::Merge {
        sources: vec![x, y],
        result: merged_result(),
    });
    assert_eq!(
        expert_operand_identities(&merge),
        Some(vec![
            "expert:proj/merged".to_owned(),
            "expert:proj/x".to_owned(),
            "expert:proj/y".to_owned()
        ])
    );
    assert_eq!(expert_operand_identities(b"PTREXR01-opaque"), None);
}

#[test]
fn an_expert_dependency_stops_hitting_when_its_operand_identity_is_invalidated() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![raw(&format!("{}{}", block('a'), block('b')))]);
    let (prefix, page_bytes) = prefill_and_publish(&h.backend, &req.tokens());
    let expert = |pod: &str| {
        KvDependency::expert(
            &ProjectId::from("proj"),
            &PodId::from(pod),
            Generation(1),
            [0; 32],
        )
    };
    h.catalog
        .insert(
            KvInsert {
                salt: &s,
                chain: &CHAIN,
                request: &req,
                prefix,
                page_bytes,
                extra_dependencies: vec![expert("x"), expert("z")],
            },
            &live,
        )
        .unwrap();
    assert!(h.lookup(&s, &req, &live).is_some());

    // What the hook does for a committed Retire of x: invalidate its decoded operands.
    let retire = record(StructuralOp::Retire {
        target: expert_target("x", ExpertState::Serving),
        reason: RetiredReason::Superseded,
    });
    let identities = expert_operand_identities(&retire).unwrap();
    let retired = h
        .catalog
        .invalidate_identities(&identities.into_iter().collect());
    assert_eq!(retired.len(), 1);
    assert!(h.lookup(&s, &req, &live).is_none());
}

#[test]
fn an_undecodable_record_flushes_every_expert_dependency() {
    let mut h = harness(CatalogLimits::default());
    let s = salt("alice", "p1");
    let live = Dead::default();
    let req = request(vec![raw(&format!("{}{}", block('a'), block('b')))]);
    let (prefix, page_bytes) = prefill_and_publish(&h.backend, &req.tokens());
    h.catalog
        .insert(
            KvInsert {
                salt: &s,
                chain: &CHAIN,
                request: &req,
                prefix,
                page_bytes,
                extra_dependencies: vec![KvDependency::expert(
                    &ProjectId::from("proj"),
                    &PodId::from("q"),
                    Generation(1),
                    [0; 32],
                )],
            },
            &live,
        )
        .unwrap();
    assert_eq!(h.catalog.invalidate_all_experts().len(), 1);
    assert!(h.lookup(&s, &req, &live).is_none());
}

#[test]
fn the_runtime_oracle_treats_an_expert_with_no_serving_record_as_not_live() {
    let (mut runtime, backend) = lifecycle_runtime();
    let s = salt("alice", "p1");
    let req = request(vec![raw(&format!("{}{}", block('a'), block('b')))]);
    let (prefix, page_bytes) = prefill_and_publish(&backend, &req.tokens());
    let refused = runtime
        .kv_prefix_insert(KvInsert {
            salt: &s,
            chain: &CHAIN,
            request: &req,
            prefix,
            page_bytes,
            extra_dependencies: vec![KvDependency::expert(
                &ProjectId::from("proj"),
                &PodId::from("x"),
                Generation(1),
                [0; 32],
            )],
        })
        .unwrap();
    assert_eq!(refused, Err(KvCatalogError::DependencyNotLive));
}
