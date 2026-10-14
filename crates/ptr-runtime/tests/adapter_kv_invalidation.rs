//! A promoted adapter generation that is superseded, and one that is revoked, retire the KV
//! entries recorded under its lifecycle key, pinned forks included, and replay agrees.
mod adapter_support;

use adapter_support::{proposal, SPLITS, WEIGHTS};
use ptr_config::PtrConfig;
use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentedModelRequest, TokenId, Tokenizer};
use ptr_pods::{
    CausalAttentionInput, CausalAttentionWeights, InMemoryPagedKvBackend, KvContinuationExecutor,
    KvPageBinding, KvPageSize, KvTensorDType, KvTensorSchema, PagedKvTensorBackend,
    ReferenceCausalAttentionExecutor,
};
use ptr_runtime::adapter::{PromoteAdapter, RevokeAdapter};
use ptr_runtime::{
    CatalogLimits, KvChainId, KvDependency, KvInsert, KvLivenessOracle, KvLookup, KvPrefixCatalog,
    KvRequestPositions, KvSalt, PtrRuntime, KV_CHUNK_TOKENS,
};
use ptr_types::{
    AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId, ProjectId, RequestId,
    Revision, SemanticContext,
};
use std::sync::Arc;

const CHAIN: KvChainId = KvChainId([1; 32]);
const KEY: &str = "ptr-adapter:alpha/coder";

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

fn request() -> SegmentedModelRequest {
    let text = "a".repeat(KV_CHUNK_TOKENS);
    SegmentedModelRequest::new(
        RequestId::from("r"),
        SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        vec![PromptSegment::raw_from_text(&ByteTokenizer::new(), &text).unwrap()],
    )
    .unwrap()
}

fn promote(runtime: &mut PtrRuntime, generation: u64) {
    runtime
        .promote_adapter(PromoteAdapter {
            project: ProjectId::from("alpha"),
            proposal: proposal("coder"),
            frozen_split_manifest: SPLITS,
            generation: Generation(generation),
            artifact: WEIGHTS,
        })
        .expect("promotes");
}

fn adapter_dependency(generation: u64) -> KvDependency {
    KvDependency::adapter(KEY, Generation(generation), [9; 32])
}

/// A runtime with a catalog, adapter generation 2 promoted, and one entry recorded under it.
fn harness() -> (PtrRuntime, InMemoryPagedKvBackend) {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).expect("runtime");
    let backend = InMemoryPagedKvBackend::new(KvPageSize::DEFAULT);
    let catalog = KvPrefixCatalog::new(
        backend.take_fork_authority().unwrap(),
        CatalogLimits::default(),
    )
    .unwrap()
    .with_retirer(Arc::new(backend.clone()));
    runtime.enable_kv_prefix_catalog(catalog).unwrap();
    promote(&mut runtime, 2);

    let req = request();
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
            &backend,
            &mut cache,
            CausalAttentionInput {
                tokens: tokens.iter().copied().map(embed).collect(),
            },
        )
        .unwrap();
    let prefix = backend.publish_prefix(&mut cache).unwrap();
    let page_bytes = backend.page_metrics(&cache).resident_bytes;
    runtime
        .kv_prefix_insert(KvInsert {
            salt: &salt(),
            chain: &CHAIN,
            request: &req,
            prefix,
            page_bytes,
            extra_dependencies: vec![adapter_dependency(2)],
        })
        .unwrap()
        .unwrap();
    (runtime, backend)
}

fn lookup(runtime: &mut PtrRuntime) -> Option<ptr_runtime::KvPin> {
    runtime
        .kv_prefix_lookup(KvLookup {
            salt: &salt(),
            chain: &CHAIN,
            request: &request(),
            positions: &KvRequestPositions::Canonical,
        })
        .unwrap()
        .map(|hit| hit.pin().clone())
}

fn pin_is_live(runtime: &PtrRuntime, pin: &ptr_runtime::KvPin) -> bool {
    runtime.kv_prefix_catalog().unwrap().pin_is_live(pin)
}

#[test]
fn revoking_the_adapter_generation_cancels_a_pinned_fork_and_retires_the_entry() {
    let (mut runtime, _backend) = harness();
    let pin = lookup(&mut runtime).expect("the entry hits while the adapter serves");
    assert!(pin_is_live(&runtime, &pin));

    runtime
        .revoke_adapter(RevokeAdapter {
            project: ProjectId::from("alpha"),
            adapter: "coder".into(),
            generation: Generation(2),
            reason: "training input revoked".into(),
        })
        .expect("revokes");
    assert!(!pin_is_live(&runtime, &pin), "the pinned fork is cancelled");
    assert!(lookup(&mut runtime).is_none());

    let replayed =
        PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).expect("replays");
    assert!(!replayed.is_live(&adapter_dependency(2)));
    assert_eq!(
        replayed.is_live(&adapter_dependency(2)),
        runtime.is_live(&adapter_dependency(2))
    );
}

#[test]
fn promoting_a_later_generation_supersedes_a_pinned_fork_of_the_earlier_one() {
    let (mut runtime, _backend) = harness();
    let pin = lookup(&mut runtime).expect("hits");
    assert!(pin_is_live(&runtime, &pin));

    promote(&mut runtime, 3);
    assert!(!pin_is_live(&runtime, &pin), "generation 2 is superseded");
    assert!(lookup(&mut runtime).is_none());

    let replayed =
        PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).expect("replays");
    for generation in [2, 3] {
        assert_eq!(
            replayed.is_live(&adapter_dependency(generation)),
            runtime.is_live(&adapter_dependency(generation)),
            "generation {generation}"
        );
    }
    assert!(!runtime.is_live(&adapter_dependency(2)));
    assert!(runtime.is_live(&adapter_dependency(3)));
}

#[test]
fn an_entry_under_another_adapters_key_is_left_alone() {
    let (mut runtime, _backend) = harness();
    let pin = lookup(&mut runtime).expect("hits");
    // Another project's adapter of the same name has another lifecycle key.
    let beta = PromoteAdapter {
        project: ProjectId::from("beta"),
        proposal: adapter_support::proposal_for("beta", "coder", WEIGHTS, Some(0.7))
            .expect("proposal"),
        frozen_split_manifest: SPLITS,
        generation: Generation(5),
        artifact: WEIGHTS,
    };
    runtime.promote_adapter(beta).expect("promotes in beta");
    assert!(pin_is_live(&runtime, &pin));
}
