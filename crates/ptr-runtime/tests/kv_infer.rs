//! Reference KV-reuse backend: real reuse through the runtime prefix catalog.
//! Public API only. Not an R005 result.

use ptr_config::PtrConfig;
use ptr_model_api::{
    InferenceBackend, KvReuseHint, KvReuseMode, KvReusePolicy, KvReuseTaint, KvReusingBackend,
    ModelEvent, ModelRequest, PromptSegment, SegmentSource, SegmentedModelRequest, Tokenizer,
    TokenizerIdentity,
};
use ptr_pods::Qwen2RopeScaling;
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
fn another_principal_or_project_never_hits() {
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
}

#[test]
fn a_raw_prefix_hits_at_a_later_revision_and_is_bitwise_equal_to_cold() {
    let prompt = format!("{}{}", text(32, 5), text(6, 6));
    let backend = backend();
    run(&backend, &scope("alice"), &raw(&backend, "a", 1, &prompt));
    let later = run(&backend, &scope("alice"), &raw(&backend, "d", 9, &prompt));
    assert_eq!(later.hit_tokens, 32, "the semantic revision is not lineage");
    let cold_backend = backend_with(ReferenceKvLimits::default());
    let cold = cold_backend
        .infer_cold(&scope("alice"), &raw(&cold_backend, "d", 9, &prompt))
        .unwrap();
    assert_same_result(&cold, &later);
}

#[test]
fn a_bound_segment_with_another_generation_or_content_never_hits() {
    let semantic = text(32, 7);
    let backend = backend();
    run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "a", 1, &semantic, &text(8, 8)),
    );
    let same = run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "b", 1, &semantic, &text(8, 9)),
    );
    assert_eq!(same.hit_tokens, 32);
    let newer = run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "c", 2, &semantic, &text(8, 9)),
    );
    assert_eq!(newer.hit_tokens, 0, "another generation");
    assert_same_result(&same, &newer);
    let other = run(
        &backend,
        &scope("alice"),
        &semantic_request(&backend, "d", 1, &text(32, 8), &text(8, 9)),
    );
    assert_eq!(other.hit_tokens, 0, "another content digest");
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

#[test]
fn chain_id_is_stable_for_equal_weights_and_split_by_weights_and_limits_shape() {
    let a = backend();
    let b = backend();
    assert_eq!(a.chain_id(), b.chain_id());
    // A different context capacity changes no stored K/V byte, so the chain is unchanged.
    let longer = backend_with(ReferenceKvLimits {
        max_prompt_tokens: 512,
        ..ReferenceKvLimits::default()
    });
    assert_eq!(a.chain_id(), longer.chain_id());
    let seeded =
        ReferenceKvReuseBackend::with_seed(ReferenceKvLimits::default(), scope("alice"), 2)
            .unwrap();
    assert_ne!(a.chain_id(), seeded.chain_id());
}

/// Prompts that share a long prefix: `shared` tokens plus a distinct 16-token suffix.
fn shared_prefix_prompts(count: usize) -> Vec<String> {
    let shared = text(64, 21);
    (0..count)
        .map(|i| {
            // A suffix that differs from every other prompt's from its first byte on.
            let suffix: String = (0..16)
                .map(|k| char::from(b'a' + ((k * 7 + i * 3 + 1) % 26) as u8))
                .collect();
            format!("{shared}{suffix}")
        })
        .collect()
}

fn one_page_bytes() -> u64 {
    let backend = backend();
    run(
        &backend,
        &scope("alice"),
        &raw(&backend, "p", 1, &text(16, 3)),
    );
    let stats = backend.catalog_stats().unwrap();
    assert_eq!(stats.entries, 1);
    stats.page_bytes
}

#[test]
fn catalog_page_bytes_count_each_unique_page_once_not_quadratically() {
    let page = one_page_bytes();
    assert!(page > 0);
    let backend = backend();
    for (i, prompt) in shared_prefix_prompts(6).iter().enumerate() {
        run(
            &backend,
            &scope("alice"),
            &raw(&backend, &format!("r{i}"), 1, prompt),
        );
    }
    let stats = backend.catalog_stats().unwrap();
    // 4 shared pages + one distinct suffix page per prompt = 10 unique pages. The former
    // whole-cache `resident_bytes` lump charged 1+2+3+4+5 = 15 pages for the first prompt
    // and 5 for each of the 5 hits: 40 pages, growing with every prompt and with prefix depth.
    assert_eq!(stats.page_bytes, 10 * page);
    assert!(stats.entries >= 6);
}

#[test]
fn a_bytes_only_limit_evicts_as_configured_and_outputs_stay_equal_to_cold() {
    let page = one_page_bytes();
    let limits = ReferenceKvLimits {
        catalog: CatalogLimits {
            max_kv_page_bytes: 7 * page,
            ..CatalogLimits::default()
        },
        ..ReferenceKvLimits::default()
    };
    let backend = backend_with(limits);
    let reference = backend_with(ReferenceKvLimits::default());
    for (i, prompt) in shared_prefix_prompts(6).iter().enumerate() {
        let warm = run(
            &backend,
            &scope("alice"),
            &raw(&backend, &format!("r{i}"), 1, prompt),
        );
        let cold = reference
            .infer_cold(
                &scope("alice"),
                &raw(&reference, &format!("r{i}"), 1, prompt),
            )
            .unwrap();
        assert_same_result(&cold, &warm);
        assert!(backend.catalog_stats().unwrap().page_bytes <= 7 * page);
    }
    let stats = backend.catalog_stats().unwrap();
    // 10 unique pages do not fit in 7: the byte limit, not the node limit, bound the catalog.
    assert!(stats.evictions + stats.capacity_refusals >= 1);
    assert!(stats.nodes < CatalogLimits::default().max_trie_nodes);
}

// ---- dynamic NTK: the RoPE regime is fixed at admission ----

const NTK_OMP: usize = 24;

fn ntk_backend() -> ReferenceKvReuseBackend {
    ntk_backend_with(ReferenceKvLimits::default())
}

fn ntk_backend_with(limits: ReferenceKvLimits) -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::with_rope_scaling(
        limits,
        scope("alice"),
        20240607,
        Qwen2RopeScaling::DynamicNtk {
            factor: 4.0,
            original_max_position_embeddings: NTK_OMP,
        },
    )
    .unwrap()
}

fn cold(backend: &ReferenceKvReuseBackend, request: &SegmentedModelRequest) -> ReferenceKvOutcome {
    backend.infer_cold(&scope("alice"), request).unwrap()
}

#[test]
fn decode_crossing_omp_under_the_admission_fixed_regime_is_exact_and_bitwise_equal_to_cold() {
    let backend = ntk_backend();
    let prefix = text(16, 4);
    let first = format!("{prefix}{}", text(4, 5));
    let second = format!("{prefix}{}", text(4, 6));
    // 20 prompt tokens + 8 new tokens = 28 > omp = 24: the decode crosses omp (position 24),
    // yet both requests run under the one regime fixed at admission.
    let a = raw(&backend, "a", 1, &first);
    let b = raw(&backend, "b", 1, &second);
    let first_out = run(&backend, &scope("alice"), &a);
    assert_eq!(first_out.rope_regime, backend.admit_regime(20).unwrap());
    assert!(first_out.rope_regime.effective_base() > 10_000.0);
    assert_eq!(first_out.rope_regime.admitted_sequence_length(), 28);
    assert_same_result(&first_out, &cold(&backend, &a));

    let warm = run(&backend, &scope("alice"), &b);
    assert_eq!(warm.hit_tokens, 16);
    assert_eq!(warm.rope_regime, first_out.rope_regime);
    assert_eq!(
        warm.run.kv_report.as_ref().unwrap().mode(),
        KvReuseMode::Exact
    );
    assert_same_result(&warm, &cold(&backend, &b));
}

#[test]
fn requests_in_different_admission_regimes_never_share_entries() {
    let backend = ntk_backend();
    let shared = text(32, 7);
    let long = format!("{shared}{}", text(8, 1)); // 40 + 8 = 48
    let shorter = format!("{shared}{}", text(4, 2)); // 36 + 8 = 44
    let same_as_long = format!("{shared}{}", text(8, 3)); // 40 + 8 = 48

    let a = raw(&backend, "a", 1, &long);
    let b = raw(&backend, "b", 1, &shorter);
    let c = raw(&backend, "c", 1, &same_as_long);
    assert_ne!(
        backend.chain_id_for_admission(40).unwrap(),
        backend.chain_id_for_admission(36).unwrap()
    );
    assert_eq!(
        backend.chain_id_for_admission(40).unwrap(),
        backend.chain_id_for_admission(40).unwrap()
    );

    let a_out = run(&backend, &scope("alice"), &a);
    assert_eq!(a_out.hit_tokens, 0);
    assert!(a_out.inserted_chunks > 0);

    // Same 32-token prefix, different admission regime: a miss, recomputed, Exact, and
    // bitwise equal to its own cold run.
    let b_out = run(&backend, &scope("alice"), &b);
    assert_ne!(b_out.rope_regime, a_out.rope_regime);
    assert_eq!(b_out.hit_tokens, 0);
    assert_eq!(b_out.prefilled_tokens, 36);
    assert_eq!(
        b_out.run.kv_report.as_ref().unwrap().mode(),
        KvReuseMode::Exact
    );
    assert_eq!(b_out.run.kv_report.as_ref().unwrap().hit_tokens(), 0);
    assert_same_result(&b_out, &cold(&backend, &b));

    // Control: the same regime as `a` does share.
    let c_out = run(&backend, &scope("alice"), &c);
    assert_eq!(c_out.rope_regime, a_out.rope_regime);
    assert_eq!(c_out.hit_tokens, 32);
    assert_same_result(&c_out, &cold(&backend, &c));

    // The regime changes what the model computes: the unscaled backend differs.
    let plain = backend_with(ReferenceKvLimits::default());
    let plain_out = cold(&plain, &raw(&plain, "a", 1, &long));
    assert_ne!(plain_out.logits_digest, cold(&backend, &a).logits_digest);
}

#[test]
fn a_forced_regime_transition_recomputes_instead_of_serving_the_cached_prefix() {
    let backend = ntk_backend_with(ReferenceKvLimits {
        max_new_tokens: 2,
        ..ReferenceKvLimits::default()
    });
    // 16 + 2 = 18 <= omp = 24: the plain-base regime, cached after the first run.
    let prefix = text(16, 9);
    let short = raw(&backend, "short", 1, &prefix);
    let short_out = run(&backend, &scope("alice"), &short);
    assert_eq!(short_out.rope_regime.effective_base(), 10_000.0);
    assert_eq!(short_out.inserted_chunks, 1);
    assert_eq!(
        backend.chain_id_for_admission(16).unwrap(),
        backend.chain_id_for_admission(5).unwrap(),
        "every length up to omp is one regime"
    );

    // The longer request extends the same prefix but is admitted into a larger regime.
    let longer_text = format!("{prefix}{}", text(20, 10));
    let longer = raw(&backend, "longer", 1, &longer_text);
    let longer_out = run(&backend, &scope("alice"), &longer);
    assert!(longer_out.rope_regime.effective_base() > 10_000.0);
    assert_eq!(longer_out.hit_tokens, 0);
    assert_eq!(longer_out.prefilled_tokens, 36);
    assert_same_result(&longer_out, &cold(&backend, &longer));

    // Back in the plain regime the original entry is still served.
    let again = run(
        &backend,
        &scope("alice"),
        &raw(&backend, "again", 1, &format!("{prefix}x")),
    );
    assert_eq!(
        again.rope_regime.effective_base_bits(),
        short_out.rope_regime.effective_base_bits()
    );
    assert_eq!(again.hit_tokens, 16);
    assert_same_result(
        &again,
        &cold(&backend, &raw(&backend, "again", 1, &format!("{prefix}x"))),
    );
}

#[test]
fn invalid_rope_scaling_is_refused_by_the_backend() {
    for scaling in [
        Qwen2RopeScaling::DynamicNtk {
            factor: 0.5,
            original_max_position_embeddings: 24,
        },
        Qwen2RopeScaling::DynamicNtk {
            factor: f32::NAN,
            original_max_position_embeddings: 24,
        },
        Qwen2RopeScaling::DynamicNtk {
            factor: 2.0,
            original_max_position_embeddings: 0,
        },
        Qwen2RopeScaling::Unsupported("yarn".into()),
    ] {
        assert!(matches!(
            ReferenceKvReuseBackend::with_rope_scaling(
                ReferenceKvLimits::default(),
                scope("alice"),
                1,
                scaling,
            ),
            Err(ReferenceKvError::Stack(_))
        ));
    }
}

#[test]
fn default_backend_reports_the_plain_base_regime_and_one_chain() {
    let backend = backend();
    let out = run(
        &backend,
        &scope("alice"),
        &raw(&backend, "a", 1, &text(20, 1)),
    );
    assert_eq!(out.rope_regime.effective_base(), 10_000.0);
    assert_eq!(
        backend.chain_id_for_admission(1).unwrap(),
        backend.chain_id()
    );
    assert_eq!(
        backend.chain_id_for_admission(200).unwrap(),
        backend.chain_id()
    );
}
