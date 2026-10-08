//! R007 WP1: runtime probes for span-forget on the CPU reference backend. Public API only.
//!
//! A span is a revocable `Derived` prompt segment. Forgetting it is: revoke its lineage
//! identity (every catalog entry citing it dies), then run the prompt without it. The prefix
//! before the span is reused from the catalog and everything after it is recomputed, and the
//! run must be bitwise equal (prompt-position logits, KV-state digest, decode logits digest)
//! to a cold run of the prompt without the span. The stale-donor probe mode is the negative
//! control showing that comparison is not vacuous. Not an R007 result.

use ptr_model_api::{PromptSegment, SegmentSource, SegmentedModelRequest, TokenId, Tokenizer};
use ptr_runtime::{
    CatalogLimits, IndexUpdate, ReferenceKvError, ReferenceKvFaults, ReferenceKvLimits,
    ReferenceKvProbe, ReferenceKvProbeMode, ReferenceKvReuseBackend, ReferenceKvScope,
    KV_CHUNK_TOKENS,
};
use ptr_types::{EvidenceId, Generation, PrincipalId, ProjectId, RequestId, Revision};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

const SPAN_ID: &str = "span:canary-1";

fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: ProjectId::from("p"),
    }
}

/// Raised limits for a prompt of up to `prompt` tokens. The catalog bounds are raised too: a
/// 4224-token run inserts one entry per 16-token boundary.
fn limits(prompt: usize) -> ReferenceKvLimits {
    ReferenceKvLimits {
        max_prompt_tokens: prompt,
        catalog: CatalogLimits {
            max_trie_nodes: 1 << 16,
            max_kv_page_bytes: 1 << 32,
            max_metadata_bytes: 1 << 30,
        },
        ..ReferenceKvLimits::default()
    }
}

fn backend(prompt: usize) -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::new(limits(prompt), scope()).unwrap()
}

/// Deterministic byte tokens: an LCG stream per `stream`, `count` long.
fn tokens(stream: u64, count: usize) -> Vec<TokenId> {
    let mut state = stream.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            TokenId((state >> 33) as u32 & 0xff)
        })
        .collect()
}

/// Token counts of a prompt `prefix | span | after`.
#[derive(Clone, Copy)]
struct Shape {
    prefix: usize,
    span: usize,
    after: usize,
}

impl Shape {
    /// R007 full size: 64 + 1024 = 1088 prefix tokens, a 256-token span, 2816 + 64 after it.
    const FULL: Self = Self {
        prefix: 1088,
        span: 256,
        after: 2880,
    };
    const SMALL: Self = Self {
        prefix: 64,
        span: 32,
        after: 96,
    };

    fn with_span_tokens(self) -> usize {
        self.prefix + self.span + self.after
    }
}

fn request(
    backend: &ReferenceKvReuseBackend,
    id: &str,
    shape: Shape,
    with_span: bool,
) -> SegmentedModelRequest {
    let identity = backend.tokenizer().identity().clone();
    let mut segments =
        vec![PromptSegment::raw_from_tokens(identity.clone(), tokens(1, shape.prefix)).unwrap()];
    if with_span {
        segments.push(
            PromptSegment::from_tokens(
                identity.clone(),
                SegmentSource::Derived {
                    id: EvidenceId::from(SPAN_ID),
                    generation: Generation(1),
                },
                tokens(2, shape.span),
            )
            .unwrap(),
        );
    }
    segments
        .push(PromptSegment::raw_from_tokens(identity.clone(), tokens(3, shape.after)).unwrap());
    SegmentedModelRequest::new(
        RequestId::from(id),
        ptr_types::SemanticContext {
            revision: Revision(1),
            entries: Vec::new(),
        },
        identity,
        segments,
    )
    .unwrap()
}

fn span_revocation() -> BTreeSet<(String, Generation)> {
    [(SPAN_ID.to_owned(), Generation(1))].into_iter().collect()
}

/// A cold probe of the prompt without the span on a fresh backend: the baseline.
fn baseline(shape: Shape) -> ReferenceKvProbe {
    let fresh = backend(shape.with_span_tokens());
    let request = request(&fresh, "baseline", shape, false);
    fresh
        .probe(&scope(), &request, ReferenceKvProbeMode::Cold)
        .unwrap()
}

fn assert_bitwise_equal(a: &ReferenceKvProbe, b: &ReferenceKvProbe) {
    assert_eq!(a.prompt_logits.len(), 256);
    assert_eq!(a.logit_bits(), b.logit_bits(), "prompt logits differ");
    assert_eq!(a.kv_state_digest, b.kv_state_digest, "KV state differs");
    assert_eq!(a.outcome.logits_digest, b.outcome.logits_digest);
    assert_eq!(a.outcome.generated, b.outcome.generated);
}

fn forget_matches_cold(shape: Shape) {
    let aligned = shape.prefix / KV_CHUNK_TOKENS * KV_CHUNK_TOKENS;
    assert_eq!(aligned, shape.prefix, "the test prefix is chunk aligned");
    let expected = baseline(shape);
    assert_eq!(expected.outcome.prompt_tokens, shape.prefix + shape.after);

    let backend = backend(shape.with_span_tokens());
    let with_span = request(&backend, "with-span", shape, true);
    let without = request(&backend, "without", shape, false);

    // Insert: the span is in the cache.
    let warm = backend
        .probe(&scope(), &with_span, ReferenceKvProbeMode::Warm)
        .unwrap();
    assert_eq!(warm.outcome.hit_tokens, 0);
    assert_eq!(
        warm.outcome.insert_refusals, 0,
        "{:?}",
        warm.outcome.insert_refusal_reasons
    );
    assert_eq!(
        warm.outcome.inserted_chunks,
        shape.with_span_tokens() / KV_CHUNK_TOKENS
    );
    // The span changes the state: the probes see it.
    assert_ne!(warm.kv_state_digest, expected.kv_state_digest);
    assert_ne!(warm.logit_bits(), expected.logit_bits());

    // Evict: revoke the span; every entry that cites it dies, the prefix entries do not.
    let retired = backend.invalidate_inputs(&span_revocation()).unwrap();
    assert!(retired > 0, "entries citing the span must be retired");
    let after_span_entries = (shape.span + shape.after) / KV_CHUNK_TOKENS;
    assert_eq!(retired, after_span_entries);

    // A request that still cites the span reuses only the prefix that does not cite it.
    let stale_cite = backend
        .probe(&scope(), &with_span, ReferenceKvProbeMode::Warm)
        .unwrap();
    assert_eq!(stale_cite.outcome.hit_tokens, shape.prefix);

    // Forget: the prompt without the span reuses the prefix and recomputes everything after.
    let forgotten = backend
        .probe(&scope(), &without, ReferenceKvProbeMode::Warm)
        .unwrap();
    assert_eq!(forgotten.outcome.hit_tokens, aligned);
    assert_eq!(forgotten.outcome.prefilled_tokens, shape.after);
    assert_bitwise_equal(&forgotten, &expected);
}

#[test]
fn forgetting_a_span_is_bitwise_equal_to_a_cold_run_without_it() {
    forget_matches_cold(Shape::SMALL);
}

#[test]
fn forgetting_a_span_beyond_the_default_prompt_limit_works_with_raised_limits() {
    let shape = Shape {
        prefix: 256,
        span: 32,
        after: 128,
    };
    assert!(shape.with_span_tokens() > ReferenceKvLimits::default().max_prompt_tokens);
    forget_matches_cold(shape);
}

/// The full R007 shape. One cold 4224-token prefill takes about a minute in release on the
/// one-token-at-a-time reference stack (the whole test about three minutes) and far longer in
/// a debug build, so it is opt-in: `cargo test --release -p ptr-runtime --test kv_span_forget
/// -- --ignored`.
#[test]
#[ignore = "about three minutes in release; run with --ignored"]
fn forgetting_a_4224_token_span_prompt_is_bitwise_equal_to_cold_with_raised_limits() {
    let shape = Shape::FULL;
    assert_eq!(shape.with_span_tokens(), 4224);
    forget_matches_cold(shape);
}

#[test]
fn a_warm_run_without_any_span_is_bitwise_equal_to_cold() {
    let shape = Shape::SMALL;
    let backend = backend(shape.with_span_tokens());
    // Populate the catalog with the prefix alone, then run the whole prompt warm.
    let prefix_only = request(&backend, "prefix", Shape { after: 16, ..shape }, false);
    backend
        .probe(&scope(), &prefix_only, ReferenceKvProbeMode::Warm)
        .unwrap();
    let without = request(&backend, "without", shape, false);
    let warm = backend
        .probe(&scope(), &without, ReferenceKvProbeMode::Warm)
        .unwrap();
    // The populating request shares its first 80 tokens (prefix plus one chunk) with the prompt.
    assert_eq!(warm.outcome.hit_tokens, shape.prefix + 16);
    assert_bitwise_equal(&warm, &baseline(shape));
    // A whole-prompt hit is refused (the last position's logits are not in the cache), so the
    // repeat recomputes from scratch and is still equal.
    let repeat = backend
        .probe(&scope(), &without, ReferenceKvProbeMode::Warm)
        .unwrap();
    assert_eq!(repeat.outcome.hit_tokens, 0);
    assert_bitwise_equal(&repeat, &warm);
}

#[test]
fn the_probes_are_deterministic_and_sensitive_to_a_one_token_change() {
    let shape = Shape::SMALL;
    let a = baseline(shape);
    assert_bitwise_equal(&a, &baseline(shape));

    let backend = backend(shape.with_span_tokens());
    let identity = backend.tokenizer().identity().clone();
    let mut changed = tokens(3, shape.after);
    changed[shape.after - 1] = TokenId((changed[shape.after - 1].0 + 1) & 0xff);
    let request = SegmentedModelRequest::new(
        RequestId::from("changed"),
        ptr_types::SemanticContext {
            revision: Revision(1),
            entries: Vec::new(),
        },
        identity.clone(),
        vec![
            PromptSegment::raw_from_tokens(identity.clone(), tokens(1, shape.prefix)).unwrap(),
            PromptSegment::raw_from_tokens(identity, changed).unwrap(),
        ],
    )
    .unwrap();
    let b = backend
        .probe(&scope(), &request, ReferenceKvProbeMode::Cold)
        .unwrap();
    assert_ne!(a.kv_state_digest, b.kv_state_digest);
    assert_ne!(a.logit_bits(), b.logit_bits());
}

#[test]
fn injected_stale_reuse_after_the_span_is_detected_by_the_bitwise_check() {
    let shape = Shape::SMALL;
    let expected = baseline(shape);
    let backend = backend(shape.with_span_tokens());
    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .unwrap();
    let with_span = request(&backend, "with-span", shape, true);
    let without = request(&backend, "without", shape, false);
    backend
        .probe(&scope(), &with_span, ReferenceKvProbeMode::Warm)
        .unwrap();
    // The revocation reaches the liveness set but not the catalog index: the entries citing
    // the span are still in the index.
    assert_eq!(backend.invalidate_inputs(&span_revocation()).unwrap(), 0);

    // The honest path is still correct: the use-time liveness check keeps the stale entries
    // from being served (a hit on the prefix only, which does not cite the span).
    let honest = backend
        .probe(&scope(), &without, ReferenceKvProbeMode::Warm)
        .unwrap();
    assert_eq!(honest.outcome.hit_tokens, shape.prefix);
    assert_bitwise_equal(&honest, &expected);

    // The negative control serves the donor's rows after the span as the prompt's own.
    let stale = backend
        .probe(
            &scope(),
            &without,
            ReferenceKvProbeMode::StaleDonor(&with_span),
        )
        .unwrap();
    let cut = (shape.prefix + shape.after - 1) / KV_CHUNK_TOKENS * KV_CHUNK_TOKENS;
    assert!(
        stale.outcome.hit_tokens > shape.prefix,
        "downstream rows were reused"
    );
    assert_eq!(
        stale.outcome.hit_tokens,
        cut.min(shape.prefix + shape.span + shape.after)
    );
    assert_ne!(stale.kv_state_digest, expected.kv_state_digest);
    assert_ne!(stale.logit_bits(), expected.logit_bits());
    // The control inserted nothing and did not disturb the honest result.
    assert_eq!(stale.outcome.inserted_chunks, 0);
    let again = backend
        .probe(&scope(), &without, ReferenceKvProbeMode::Warm)
        .unwrap();
    assert_bitwise_equal(&again, &expected);
}

#[test]
fn default_limits_are_unchanged_and_the_tiny_model_config_digest_is_pinned() {
    let defaults = ReferenceKvLimits::default();
    assert_eq!(defaults.max_prompt_tokens, 256);
    assert_eq!(defaults.max_new_tokens, 8);
    let default_backend =
        ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let raised = backend(4224);
    assert_eq!(raised.limits().max_prompt_tokens, 4224);

    // The default still refuses a prompt over 256 tokens.
    let long = request(&default_backend, "long", Shape::SMALL, true);
    assert_eq!(long.total_tokens(), 192);
    let over = request(
        &default_backend,
        "over",
        Shape {
            prefix: 256,
            span: 16,
            after: 16,
        },
        true,
    );
    assert!(matches!(
        default_backend.probe(&scope(), &over, ReferenceKvProbeMode::Cold),
        Err(ReferenceKvError::PromptTooLong {
            tokens: 288,
            maximum: 256
        })
    ));

    // The limits change max_position_embeddings but not the pinned tiny_model_config.
    for backend in [&default_backend, &raised] {
        let config = backend.stack_config();
        let json = format!(
            "{{\"head_dim\":{},\"hidden_size\":{},\"layers\":{},\"num_heads\":{},\
             \"num_kv_heads\":{},\"vocab_size\":{},\"weight_seed\":{}}}",
            config.layer.hidden_size / config.layer.num_attention_heads,
            config.layer.hidden_size,
            config.num_layers,
            config.layer.num_attention_heads,
            config.layer.num_key_value_heads,
            config.vocab_size,
            ptr_runtime::REFERENCE_KV_WEIGHT_SEED,
        );
        let hex: String = Sha256::digest(json.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let digest = format!("sha256:{hex}");
        assert_eq!(
            digest,
            "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a"
        );
    }
    assert_eq!(
        default_backend.stack_config().layer.max_position_embeddings,
        256 + 8
    );
    assert_eq!(
        raised.stack_config().layer.max_position_embeddings,
        4224 + 8
    );
    // The position bound is not part of the chain id (weights, RoPE regime, build and tokenizer
    // are), so the pinned config and chain derivation are untouched by raising the limit.
    assert_eq!(default_backend.chain_id(), raised.chain_id());
}
