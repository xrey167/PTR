//! Reference KV-reuse backend: real reuse through the runtime prefix catalog.
//! Public API only. Not an R005 result.

use ptr_config::PtrConfig;
use ptr_model_api::{
    InferenceBackend, KvReuseHint, KvReuseMode, KvReusePolicy, KvReuseTaint, KvReusingBackend,
    ModelEvent, ModelRequest, PromptSegment, SegmentSource, SegmentedModelRequest, Tokenizer,
    TokenizerIdentity,
};
use ptr_runtime::{
    CatalogLimits, PtrRuntime, ReferenceKvError, ReferenceKvLimits, ReferenceKvOutcome,
    ReferenceKvReuseBackend, ReferenceKvScope,
};
use ptr_types::{Generation, PrincipalId, ProjectId, RequestId, Revision, SemanticContext};
use std::collections::BTreeSet;

fn scope(principal: &str) -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from(principal),
        project: ProjectId::from("p"),
    }
}

fn backend_with(limits: ReferenceKvLimits) -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::new(limits, scope("alice")).unwrap()
}

fn backend() -> ReferenceKvReuseBackend {
    backend_with(ReferenceKvLimits::default())
}

fn context(revision: u64) -> SemanticContext {
    SemanticContext {
        revision: Revision(revision),
        entries: Vec::new(),
    }
}

/// `count` ASCII bytes derived from `seed`, never containing a repeated 16-byte chunk.
fn text(count: usize, seed: usize) -> String {
    (0..count)
        .map(|i| char::from(b'a' + ((i * 7 + seed * 13 + i / 16 * 5) % 26) as u8))
        .collect()
}

fn raw(
    backend: &ReferenceKvReuseBackend,
    id: &str,
    revision: u64,
    text: &str,
) -> SegmentedModelRequest {
    let segment = PromptSegment::raw_from_text(backend.tokenizer(), text).unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        context(revision),
        backend.tokenizer().identity().clone(),
        vec![segment],
    )
    .unwrap()
}

fn run(
    backend: &ReferenceKvReuseBackend,
    scope: &ReferenceKvScope,
    request: &SegmentedModelRequest,
) -> ReferenceKvOutcome {
    backend.infer_scoped(scope, request, None).unwrap()
}

fn output_text(outcome: &ReferenceKvOutcome) -> String {
    outcome
        .run
        .events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Token(text) => Some(text.clone()),
            _ => None,
        })
        .expect("a token event")
}

fn assert_same_result(a: &ReferenceKvOutcome, b: &ReferenceKvOutcome) {
    assert_eq!(output_text(a), output_text(b));
    assert_eq!(a.generated, b.generated);
    assert_eq!(a.logits_digest, b.logits_digest);
}

#[test]
fn warm_run_with_a_shared_32_token_prefix_hits_exactly_32_and_is_bitwise_equal_to_cold() {
    let prefix = text(32, 1);
    let first = format!("{prefix}{}", text(10, 2));
    let second = format!("{prefix}{}", text(10, 3));

    // Cold reference: a fresh backend, no reuse at all.
    let cold_backend = backend();
    let cold_request = raw(&cold_backend, "cold", 1, &second);
    let cold = cold_backend
        .infer_cold(&scope("alice"), &cold_request)
        .unwrap();
    assert_eq!(cold.hit_tokens, 0);
    assert_eq!(cold.prefilled_tokens, 42);
    assert_eq!(cold.inserted_chunks, 0);

    // Warm: the first prompt populates the catalog, the second shares 32 tokens with it.
    let warm_backend = backend();
    let first_out = run(
        &warm_backend,
        &scope("alice"),
        &raw(&warm_backend, "a", 1, &first),
    );
    assert_eq!(first_out.hit_tokens, 0);
    assert_eq!(first_out.inserted_chunks, 2, "chunk boundaries 16 and 32");
    let warm = run(
        &warm_backend,
        &scope("alice"),
        &raw(&warm_backend, "b", 1, &second),
    );
    assert_eq!(warm.hit_tokens, 32);
    assert_eq!(warm.prefilled_tokens, 10);
    assert_eq!(warm.run.kv_report.unwrap().hit_tokens(), 32);
    assert_eq!(warm.run.kv_report.unwrap().mode(), KvReuseMode::Exact);
    assert_same_result(&cold, &warm);
    assert!(!output_text(&warm).is_empty());
    assert_ne!(first_out.logits_digest, warm.logits_digest);

    let stats = warm_backend.catalog_stats().unwrap();
    assert_eq!(stats.hits, 1);
}

#[test]
fn a_prompt_that_is_entirely_cached_is_prefilled_again_and_still_equal() {
    let prompt = text(32, 4);
    let backend = backend();
    let request = raw(&backend, "a", 1, &prompt);
    let first = run(&backend, &scope("alice"), &request);
    let second = run(&backend, &scope("alice"), &raw(&backend, "b", 1, &prompt));
    // The deepest entry covers the whole prompt, and the last prompt token's logits are not
    // in a cache, so the hit is refused rather than skipping the logits.
    assert_eq!(second.hit_tokens, 0);
    assert_eq!(second.prefilled_tokens, 32);
    assert_same_result(&first, &second);
}

#[test]
fn another_principal_project_or_revision_never_hits() {
    let prefix = text(32, 5);
    let prompt = format!("{prefix}{}", text(6, 6));
    let backend = backend();
    let baseline = run(&backend, &scope("alice"), &raw(&backend, "a", 1, &prompt));
    let own = run(&backend, &scope("alice"), &raw(&backend, "a2", 1, &prompt));
    assert_eq!(own.hit_tokens, 32);

    let bob = run(&backend, &scope("bob"), &raw(&backend, "b", 1, &prompt));
    assert_eq!(bob.hit_tokens, 0, "different principal");
    assert_same_result(&baseline, &bob);

    let other_project = ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: ProjectId::from("other"),
    };
    let moved = run(&backend, &other_project, &raw(&backend, "c", 1, &prompt));
    assert_eq!(moved.hit_tokens, 0, "different project (salt)");
    assert_same_result(&baseline, &moved);

    let later = run(&backend, &scope("alice"), &raw(&backend, "d", 2, &prompt));
    assert_eq!(later.hit_tokens, 0, "different semantic revision (lineage)");
    assert_same_result(&baseline, &later);
}

fn semantic_request(
    backend: &ReferenceKvReuseBackend,
    id: &str,
    generation: u64,
    semantic: &str,
    suffix: &str,
) -> SegmentedModelRequest {
    let bound = PromptSegment::from_text(
        backend.tokenizer(),
        SegmentSource::Semantic {
            key: "doc:one".to_owned(),
            generation: Generation(generation),
        },
        semantic,
    )
    .unwrap();
    let tail = PromptSegment::raw_from_text(backend.tokenizer(), suffix).unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        context(1),
        backend.tokenizer().identity().clone(),
        vec![bound, tail],
    )
    .unwrap()
}

#[test]
fn invalidating_a_bound_input_makes_the_next_run_a_miss_with_identical_output() {
    let semantic = text(32, 7);
    let backend = backend();
    let first = run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "a", 1, &semantic, &text(8, 8)),
    );
    let warm = run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "b", 1, &semantic, &text(8, 9)),
    );
    assert_eq!(warm.hit_tokens, 32);

    let invalidated: BTreeSet<(String, Generation)> = [("doc:one".to_owned(), Generation(1))]
        .into_iter()
        .collect();
    assert!(backend.invalidate_inputs(&invalidated).unwrap() >= 1);
    assert_eq!(backend.catalog_stats().unwrap().entries, 0);

    let after = run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "c", 1, &semantic, &text(8, 9)),
    );
    assert_eq!(after.hit_tokens, 0, "the revoked input is not reusable");
    assert_eq!(
        after.inserted_chunks, 0,
        "and nothing derived from it is cached"
    );
    assert!(after.insert_refusals >= 1);
    assert_same_result(&warm, &after);
    assert_ne!(first.logits_digest, after.logits_digest);

    // The same content under a newer generation is another lineage, computed fresh.
    let newer = run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "d", 2, &semantic, &text(8, 9)),
    );
    assert_eq!(newer.hit_tokens, 0);
    assert_same_result(&warm, &newer);
}

#[test]
fn flushing_retires_every_entry_and_the_next_run_is_an_identical_miss() {
    let prompt = format!("{}{}", text(32, 10), text(5, 11));
    let backend = backend();
    let first = run(&backend, &scope("alice"), &raw(&backend, "a", 1, &prompt));
    let warm = run(&backend, &scope("alice"), &raw(&backend, "b", 1, &prompt));
    assert_eq!(warm.hit_tokens, 32);
    assert_eq!(backend.flush().unwrap(), 2);
    assert_eq!(backend.catalog_stats().unwrap().entries, 0);
    let again = run(&backend, &scope("alice"), &raw(&backend, "c", 1, &prompt));
    assert_eq!(again.hit_tokens, 0);
    assert_same_result(&first, &again);
}

#[test]
fn capacity_exhaustion_is_a_graceful_insert_miss_and_never_changes_the_output() {
    let limits = ReferenceKvLimits {
        catalog: CatalogLimits {
            max_trie_nodes: 2,
            ..CatalogLimits::default()
        },
        ..ReferenceKvLimits::default()
    };
    let backend = backend_with(limits);
    let short = text(32, 12);
    let long = format!("{short}{}", text(16, 13));
    run(&backend, &scope("alice"), &raw(&backend, "a", 1, &short));
    assert_eq!(backend.catalog_stats().unwrap().nodes, 2);

    // The 32-token entry is pinned by this run's fork, so a third node cannot be made.
    let capped = run(&backend, &scope("alice"), &raw(&backend, "b", 1, &long));
    assert_eq!(capped.hit_tokens, 32);
    assert!(capped.insert_refusals >= 1);
    assert_eq!(capped.inserted_chunks, 0);
    let stats = backend.catalog_stats().unwrap();
    assert!(stats.nodes <= 2);
    assert!(stats.capacity_refusals >= 1);

    let reference = backend_with(ReferenceKvLimits::default());
    let cold = reference
        .infer_cold(&scope("alice"), &raw(&reference, "b", 1, &long))
        .unwrap();
    assert_same_result(&cold, &capped);
    // The refused chunk left nothing behind in the backend.
    assert_eq!(backend.catalog_stats().unwrap().entries, 1);
}

#[test]
fn the_trait_run_is_exact_and_the_runtime_ingests_it() {
    let backend = backend();
    let prompt = format!("{}{}", text(32, 14), text(4, 15));
    let hint = KvReuseHint {
        requested_prefix_tokens: 32,
        policy: KvReusePolicy::ExactOnly,
    };
    let first =
        KvReusingBackend::infer_segmented(&backend, &raw(&backend, "a", 1, &prompt), Some(&hint))
            .unwrap();
    assert_eq!(first.kv_report.unwrap().hit_tokens(), 0);
    let second = backend
        .infer_segmented(&raw(&backend, "b", 1, &prompt), Some(&hint))
        .unwrap();
    let report = second.kv_report.unwrap();
    assert_eq!(
        (report.hit_tokens(), report.mode()),
        (32, KvReuseMode::Exact)
    );
    assert_eq!(second.taint(), KvReuseTaint::Exact);
    assert_eq!(second.events, first.events);

    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let resumable = runtime.ingest_model_run(&second).unwrap();
    assert_eq!(resumable.kv_taint, KvReuseTaint::Exact);
    assert!(resumable.action.is_none());
    assert_eq!(resumable.model_events, second.events);
}

#[test]
fn the_plain_inference_backend_is_the_cold_path() {
    let backend = backend();
    let prompt = text(40, 16);
    let request = ModelRequest {
        request_id: RequestId::from("plain"),
        semantic_context: context(1),
        raw_text: prompt.clone(),
    };
    let events = InferenceBackend::infer(&backend, &request).unwrap();
    assert_eq!(InferenceBackend::name(&backend), "reference-kv-reuse");
    assert_eq!(backend.catalog_stats().unwrap().entries, 0, "no insert");
    assert_eq!(backend.catalog_stats().unwrap().hits, 0);

    let reused = run(&backend, &scope("alice"), &raw(&backend, "r", 1, &prompt));
    assert_eq!(events, reused.run.events);
    // Cold again after the catalog is warm: still no lookup.
    let events_again = InferenceBackend::infer(&backend, &request).unwrap();
    assert_eq!(events_again, events);
    assert_eq!(backend.catalog_stats().unwrap().hits, 0);
}

#[test]
fn oversized_prompts_foreign_tokenizers_and_bad_limits_are_typed_errors() {
    let backend = backend();
    let too_long = raw(&backend, "x", 1, &text(257, 17));
    assert!(matches!(
        backend.infer_scoped(&scope("alice"), &too_long, None),
        Err(ReferenceKvError::PromptTooLong {
            tokens: 257,
            maximum: 256
        })
    ));
    let at_limit = raw(&backend, "y", 1, &text(256, 18));
    assert!(backend
        .infer_scoped(&scope("alice"), &at_limit, None)
        .is_ok());

    let foreign = TokenizerIdentity::new("other", "9").unwrap();
    let segment =
        PromptSegment::raw_from_tokens(foreign.clone(), vec![ptr_model_api::TokenId(1)]).unwrap();
    let request =
        SegmentedModelRequest::new(RequestId::from("z"), context(1), foreign, vec![segment])
            .unwrap();
    assert!(matches!(
        backend.infer_scoped(&scope("alice"), &request, None),
        Err(ReferenceKvError::TokenizerMismatch { .. })
    ));
    assert!(KvReusingBackend::infer_segmented(&backend, &request, None).is_err());

    let blank = ReferenceKvScope {
        principal: PrincipalId::from(""),
        project: ProjectId::from("p"),
    };
    assert!(matches!(
        backend.infer_scoped(&blank, &raw(&backend, "q", 1, "hello"), None),
        Err(ReferenceKvError::InvalidScope)
    ));
    assert!(matches!(
        ReferenceKvReuseBackend::new(
            ReferenceKvLimits {
                max_prompt_tokens: 0,
                ..ReferenceKvLimits::default()
            },
            scope("alice")
        ),
        Err(ReferenceKvError::InvalidLimits)
    ));
}

#[test]
fn runs_are_deterministic_across_backends_and_depend_on_the_weights() {
    let prompt = text(40, 19);
    let a = backend();
    let b = backend();
    let ra = run(&a, &scope("alice"), &raw(&a, "a", 1, &prompt));
    let rb = run(&b, &scope("alice"), &raw(&b, "a", 1, &prompt));
    assert_same_result(&ra, &rb);

    let other = ReferenceKvReuseBackend::with_seed(ReferenceKvLimits::default(), scope("alice"), 1)
        .unwrap();
    let ro = run(&other, &scope("alice"), &raw(&other, "a", 1, &prompt));
    assert_ne!(ra.logits_digest, ro.logits_digest);
    assert_ne!(a.chain_id(), other.chain_id());
}
