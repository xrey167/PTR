//! Use-time liveness of Semantic KV dependencies under `PtrRuntime` as the oracle:
//! revocation as a floor over every citation of a key, the runtime-built segment,
//! isolation from the lifecycle tables, bounded history and replay equality.

use std::sync::Arc;

use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, PromptSegment, SegmentKind, SegmentSource, SegmentedModelRequest, TokenId,
    Tokenizer,
};
use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryPagedKvBackend, KvContinuationExecutor,
    KvPageBinding, KvPageSize, KvTensorDType, KvTensorSchema, PagedKvTensorBackend,
    ReferenceCausalAttentionExecutor,
};
use ptr_runtime::execution::RequiredVerification;
use ptr_runtime::{
    CatalogLimits, KvChainId, KvDependency, KvDependencyKind, KvInsert, KvLivenessOracle, KvLookup,
    KvPrefixCatalog, KvRequestPositions, KvSalt, PtrRuntime, SemanticSegmentError, KV_CHUNK_TOKENS,
};
use ptr_semdb::SemanticDelta;
use ptr_types::{
    AdapterVersion, CapsuleId, DeviceId, Generation, ModelVersion, PrincipalId, ProjectId,
    RequestId, Revision, SemanticContext,
};

#[path = "common/semantic.rs"]
mod semantic_common;

const CHAIN: KvChainId = KvChainId([1; 32]);

fn runtime() -> PtrRuntime {
    let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
    runtime
        .install_semantic_grant(
            ptr_runtime::SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(semantic_common::AcceptAll)
                .allow_host_writes(),
        )
        .unwrap();
    runtime
}

fn edit(runtime: &mut PtrRuntime, key: &str, value: &str) {
    let mut delta = SemanticDelta::default();
    delta.upserts.insert(key.into(), value.into());
    semantic_common::host_write_now(runtime, delta).unwrap();
}

fn remove(runtime: &mut PtrRuntime, key: &str) {
    let mut delta = SemanticDelta::default();
    delta.removals.insert(key.into());
    semantic_common::host_write_now(runtime, delta).unwrap();
}

fn revoke(runtime: &mut PtrRuntime, key: &str, generation: u64) {
    runtime
        .commit(LedgerEvent::Revoked {
            subject: key.into(),
            generation: Generation(generation),
        })
        .unwrap();
}

fn dep(key: &str, generation: u64) -> KvDependency {
    KvDependency {
        kind: KvDependencyKind::Segment(SegmentKind::Semantic),
        identity: key.to_owned(),
        generation: Generation(generation),
        content_digest: [0; 32],
        fence_epoch: 0,
    }
}

/// Which generations in `0..=current + 2` are live for `key`.
fn live_set(runtime: &PtrRuntime, key: &str) -> Vec<u64> {
    (0..=runtime.revision().0 + 2)
        .filter(|g| runtime.is_live(&dep(key, *g)))
        .collect()
}

#[test]
fn a_revocation_kills_every_citation_of_the_key_not_only_the_cited_pair() {
    let mut rt = runtime();
    edit(&mut rt, "doc:a", "v1"); // revision 1
    edit(&mut rt, "doc:other", "x"); // 2
    edit(&mut rt, "doc:other", "y"); // 3
    assert_eq!(live_set(&rt, "doc:a"), vec![1, 2, 3]);
    revoke(&mut rt, "doc:a", 2);
    // Every g' in [last_edit, current] is dead, and the future is dead too.
    assert_eq!(live_set(&rt, "doc:a"), Vec::<u64>::new());
    // Unrelated keys are untouched.
    assert_eq!(live_set(&rt, "doc:other"), vec![3]);
    // Unrelated commits do not resurrect it.
    edit(&mut rt, "doc:other", "z");
    assert_eq!(live_set(&rt, "doc:a"), Vec::<u64>::new());
    // The runtime constructor refuses the revoked key too.
    assert_eq!(
        rt.semantic_prompt_segment(&ByteTokenizer::new(), "doc:a")
            .unwrap_err(),
        SemanticSegmentError::Revoked
    );
}

#[test]
fn a_later_genuine_edit_readmits_only_citations_at_or_after_it() {
    let mut rt = runtime();
    edit(&mut rt, "doc:a", "v1"); // 1
    revoke(&mut rt, "doc:a", 1);
    edit(&mut rt, "doc:a", "v2"); // 2: new committed content
    assert_eq!(live_set(&rt, "doc:a"), vec![2]);
    // The revoked pair itself stays dead for good.
    assert!(!rt.is_live(&dep("doc:a", 1)));
    edit(&mut rt, "doc:b", "x"); // 3
    assert_eq!(live_set(&rt, "doc:a"), vec![2, 3]);
}

#[test]
fn a_revocation_of_a_key_the_snapshot_does_not_hold_records_nothing() {
    let mut rt = runtime();
    revoke(&mut rt, "doc:later", 9);
    edit(&mut rt, "doc:later", "v1"); // created after the tombstone: fresh content
    assert_eq!(live_set(&rt, "doc:later"), vec![1]);
}

#[test]
fn a_semantic_key_named_like_a_lifecycle_identity_never_borrows_its_liveness() {
    let mut rt = runtime();
    // A capsule identity with a live generation 7, and a semantic key of the same name.
    rt.commit(LedgerEvent::CapsuleCommitted {
        project: ProjectId::from("p1"),
        capsule: CapsuleId::from("capsule:a"),
        generation: Generation(7),
    })
    .unwrap();
    // The lifecycle says (capsule:a, 7) is live, but no semantic key exists: dead at 7.
    assert!(!rt.is_live(&dep("capsule:a", 7)));
    assert!(!rt.is_live(&dep("capsule:a", 0)));
    edit(&mut rt, "capsule:a", "text"); // revision 1
    assert_eq!(live_set(&rt, "capsule:a"), vec![1]);
    // The lifecycle generation 7 does not make a semantic citation at 7 live.
    assert!(!rt.is_live(&dep("capsule:a", 7)));
    // A lifecycle supersession to generation 8 does not kill the semantic citation.
    rt.commit(LedgerEvent::CapsuleSuperseded {
        capsule: CapsuleId::from("capsule:a"),
        old: Generation(7),
        new: Generation(8),
    })
    .unwrap();
    assert!(rt.is_live(&dep("capsule:a", 1)));
}

#[test]
fn the_runtime_builds_the_semantic_segment_from_the_committed_value() {
    let mut rt = runtime();
    let tokenizer = ByteTokenizer::new();
    assert_eq!(
        rt.semantic_prompt_segment(&tokenizer, "doc:a").unwrap_err(),
        SemanticSegmentError::MissingKey
    );
    edit(&mut rt, "doc:a", "committed text");
    let segment = rt.semantic_prompt_segment(&tokenizer, "doc:a").unwrap();
    assert_eq!(
        tokenizer.decode(segment.tokens()).unwrap(),
        "committed text"
    );
    let SegmentSource::Semantic { key, generation } = segment.source().clone() else {
        panic!("not a semantic segment");
    };
    assert_eq!(key, "doc:a");
    assert!(rt.is_live(&dep("doc:a", generation.0)));
    // Unrelated commits do not change the lineage of unchanged content.
    edit(&mut rt, "doc:b", "other");
    assert_eq!(
        rt.semantic_prompt_segment(&tokenizer, "doc:a").unwrap(),
        segment
    );
    // After an edit the old segment is dead and the rebuilt one names the new content.
    edit(&mut rt, "doc:a", "newer text");
    assert!(!rt.is_live(&dep("doc:a", generation.0)));
    let rebuilt = rt.semantic_prompt_segment(&tokenizer, "doc:a").unwrap();
    assert_ne!(rebuilt, segment);
    assert!(rt.is_live(&dep("doc:a", rebuilt_generation(&rebuilt))));
    // An empty value has no segment; the contract error is passed through.
    edit(&mut rt, "doc:empty", "");
    assert!(matches!(
        rt.semantic_prompt_segment(&tokenizer, "doc:empty"),
        Err(SemanticSegmentError::Contract(_))
    ));
}

fn rebuilt_generation(segment: &PromptSegment) -> u64 {
    match segment.source() {
        SegmentSource::Semantic { generation, .. } => generation.0,
        other => panic!("{other:?}"),
    }
}

// ---- catalog level -------------------------------------------------------------------------

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

fn embed(token: TokenId) -> Vec<f32> {
    (0..4)
        .map(|i| ((token.0 as f32) + i as f32) / 97.0 - 0.5)
        .collect()
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

fn raw(label: char) -> PromptSegment {
    PromptSegment::raw_from_text(
        &ByteTokenizer::new(),
        &label.to_string().repeat(KV_CHUNK_TOKENS),
    )
    .unwrap()
}

fn runtime_with_catalog() -> (PtrRuntime, InMemoryPagedKvBackend) {
    let mut rt = runtime();
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let catalog = KvPrefixCatalog::new(
        backend.take_fork_authority().unwrap(),
        CatalogLimits::default(),
    )
    .unwrap()
    .with_retirer(Arc::new(backend.clone()));
    rt.enable_kv_prefix_catalog(catalog).unwrap();
    (rt, backend)
}

fn insert(
    rt: &mut PtrRuntime,
    backend: &InMemoryPagedKvBackend,
    req: &SegmentedModelRequest,
) -> Result<(), ptr_runtime::KvCatalogError> {
    let tokens = req.tokens();
    let tokens = &tokens[..tokens.len() / KV_CHUNK_TOKENS * KV_CHUNK_TOKENS];
    let mut cache = backend
        .allocate_bound(
            schema(),
            128,
            KvPageBinding {
                model: ModelVersion::from("base-v1"),
                adapter: AdapterVersion::from("none"),
                generation: Generation(1),
                execution_manifest: [3; 32],
                principal: PrincipalId::from("alice"),
                device: DeviceId::from("cpu"),
                dtype: KvTensorDType::F32,
                page_tokens: KvPageSize::DEFAULT,
            },
        )
        .unwrap();
    ReferenceCausalAttentionExecutor::new(CausalAttentionWeights::identity(4).unwrap(), 2)
        .unwrap()
        .prefill(
            backend,
            &mut cache,
            CausalAttentionInput {
                tokens: tokens.iter().copied().map(embed).collect(),
            },
        )
        .unwrap();
    let prefix = backend.publish_prefix(&mut cache).unwrap();
    let page_bytes = backend.page_metrics(&cache).resident_bytes;
    rt.kv_prefix_insert(KvInsert {
        salt: &salt(),
        chain: &CHAIN,
        request: req,
        prefix,
        page_bytes,
        extra_dependencies: Vec::new(),
    })
    .unwrap()
}

fn hits(rt: &mut PtrRuntime, req: &SegmentedModelRequest) -> bool {
    let hit = rt
        .kv_prefix_lookup(KvLookup {
            salt: &salt(),
            chain: &CHAIN,
            request: req,
            positions: &KvRequestPositions::Canonical,
        })
        .unwrap();
    match hit {
        Some(hit) => {
            rt.kv_prefix_release(hit.pin());
            true
        }
        None => false,
    }
}

fn segment_text(key: &str, generation: u64, text: &str) -> PromptSegment {
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

#[test]
fn the_constructed_segment_hits_on_repeat_and_is_dead_after_an_edit() {
    let (mut rt, backend) = runtime_with_catalog();
    edit(&mut rt, "doc:a", &"v".repeat(KV_CHUNK_TOKENS));
    let tokenizer = ByteTokenizer::new();
    let req = request(vec![
        raw('a'),
        rt.semantic_prompt_segment(&tokenizer, "doc:a").unwrap(),
    ]);
    insert(&mut rt, &backend, &req).unwrap();
    assert!(hits(&mut rt, &req));
    // The same content rebuilt after an unrelated commit is the same lineage: a hit.
    edit(&mut rt, "doc:b", "x");
    let again = request(vec![
        raw('a'),
        rt.semantic_prompt_segment(&tokenizer, "doc:a").unwrap(),
    ]);
    assert!(hits(&mut rt, &again));
    edit(&mut rt, "doc:a", &"w".repeat(KV_CHUNK_TOKENS));
    assert!(!hits(&mut rt, &req), "stale hit");
    assert_eq!(
        insert(&mut rt, &backend, &req),
        Err(ptr_runtime::KvCatalogError::DependencyNotLive)
    );
}

#[test]
fn a_revocation_leaves_no_hit_and_no_insert_at_any_generation() {
    let (mut rt, backend) = runtime_with_catalog();
    edit(&mut rt, "doc:a", "v1"); // 1
    edit(&mut rt, "doc:b", "x"); // 2
    let text = "t".repeat(KV_CHUNK_TOKENS);
    let at = |g: u64| request(vec![raw('a'), segment_text("doc:a", g, &text)]);
    // Same tokens at three citation generations: three lineages, all live.
    for g in 1..=2 {
        insert(&mut rt, &backend, &at(g)).unwrap();
        assert!(hits(&mut rt, &at(g)));
    }
    revoke(&mut rt, "doc:a", 1);
    // The catalog hook invalidated the other generation's entry too.
    assert_eq!(rt.kv_prefix_catalog().unwrap().stats().entries, 0);
    for g in 0..=4 {
        assert!(!hits(&mut rt, &at(g)), "hit at {g}");
        assert_eq!(
            insert(&mut rt, &backend, &at(g)),
            Err(ptr_runtime::KvCatalogError::DependencyNotLive),
            "insert at {g}"
        );
    }
    assert_eq!(rt.kv_prefix_catalog().unwrap().stats().entries, 0);
}

// ---- bounded history and replay ----------------------------------------------------------------

#[test]
fn replay_gives_the_same_liveness_answers_as_the_live_runtime_after_churn() {
    let mut rt = runtime();
    edit(&mut rt, "doc:a", "1");
    edit(&mut rt, "doc:b", "1");
    edit(&mut rt, "doc:a", "2");
    revoke(&mut rt, "doc:b", 2);
    remove(&mut rt, "doc:a");
    edit(&mut rt, "doc:c", "1");
    edit(&mut rt, "doc:a", "3"); // re-creation records a fresh edit
    revoke(&mut rt, "doc:c", 5);
    edit(&mut rt, "doc:b", "9");
    revoke(&mut rt, "doc:never", 1);
    edit(&mut rt, "doc:d", "1");
    let events = rt.committed_events().to_vec();
    let replayed = PtrRuntime::replay(ptr_config::PtrConfig::default(), &events).unwrap();
    assert_eq!(replayed.revision(), rt.revision());
    for key in ["doc:a", "doc:b", "doc:c", "doc:d", "doc:never", "doc:zzz"] {
        assert_eq!(live_set(&replayed, key), live_set(&rt, key), "{key}");
    }
    // The churn itself: a re-created key is live only from its re-creation, a revoked
    // and never re-edited key is dead, a revoked and re-edited key is live again.
    assert_eq!(live_set(&rt, "doc:a"), vec![6, 7, 8]);
    assert_eq!(live_set(&rt, "doc:c"), Vec::<u64>::new());
    assert_eq!(live_set(&rt, "doc:b"), vec![7, 8]);
}

#[test]
fn a_compacted_restore_keeps_a_revoked_key_dead_until_it_is_edited() {
    let mut rt = runtime();
    edit(&mut rt, "doc:a", "1");
    edit(&mut rt, "doc:b", "1");
    revoke(&mut rt, "doc:a", 1);
    let snapshot = rt.export_compacted_snapshot().unwrap();
    let mut restored = PtrRuntime::restore_compacted(
        ptr_config::PtrConfig::default(),
        snapshot.bytes(),
        snapshot.anchor(),
        &[],
    )
    .unwrap();
    restored
        .install_semantic_grant(
            ptr_runtime::SemanticGrant::new(RequiredVerification::Deterministic)
                .with_verifier(semantic_common::AcceptAll)
                .allow_host_writes(),
        )
        .unwrap();
    assert_eq!(live_set(&restored, "doc:a"), Vec::<u64>::new());
    // doc:b: only known unchanged since the snapshot revision (2).
    assert_eq!(live_set(&restored, "doc:b"), vec![2]);
    edit(&mut restored, "doc:a", "2");
    assert_eq!(live_set(&restored, "doc:a"), vec![3]);
}
