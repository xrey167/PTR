//! R006 WP3: positional (secondary index) reuse on the one-layer CPU reference backend.
//! Public API only.
//!
//! A whole prompt segment `C` that sat at one offset in an earlier request is found by
//! content at another offset, its stored layer-0 `PreRope` rows are repacked into the cache
//! at the new positions and the run must be bitwise equal (prompt logits, KV-state digest,
//! generated tokens) to a cold prefill of the same request. In a one-layer model the rows
//! of a token are a pure function of the token, so staleness is invisible in any output:
//! every revocation, supersession and omission check below is structural (what the lookup
//! returned or admitted, and what a session emitted or inserted after the event). The
//! `PostRope` stack and the liveness-bypass fault mode show those checks can fail. Not an
//! R006 result.

use std::collections::BTreeSet;

use ptr_model_api::{
    KvReuseMode, PromptSegment, SegmentSource, SegmentedModelRequest, TokenId, Tokenizer,
};
use ptr_pods::{KvKeyEncoding, Qwen2RopeScaling};
use ptr_runtime::{
    CatalogLimits, IndexUpdate, KvCancelReason, KvLivenessOracle, KvSession, KvSessionFinish,
    KvSessionPhase, KvSessionStep, PositionalControl, PositionalOptions, ReferenceKvError,
    ReferenceKvFaults, ReferenceKvLimits, ReferenceKvOutcome, ReferenceKvProbe,
    ReferenceKvProbeMode, ReferenceKvReuseBackend, ReferenceKvScope, ReferenceKvStackShape,
    SecondaryConfig, SecondaryLimits, REFERENCE_KV_WEIGHT_SEED,
};
use ptr_types::{Generation, KnowledgeObjectId, PrincipalId, ProjectId, RequestId, Revision};
use sha2::{Digest as _, Sha256};

const BASE_MODEL: &str = "r005-tiny";
const SEGMENT_TOKENS: usize = 48;
const TAIL_TOKENS: usize = 16;

fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: ProjectId::from("p"),
    }
}

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

/// A one-layer backend with positional reuse opted in.
fn positional(prompt: usize) -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::with_layers(limits(prompt), scope(), REFERENCE_KV_WEIGHT_SEED, 1)
        .unwrap()
        .with_positional_reuse(PositionalOptions::default())
        .unwrap()
}

fn positional_with(prompt: usize, control: PositionalControl) -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::with_layers(limits(prompt), scope(), REFERENCE_KV_WEIGHT_SEED, 1)
        .unwrap()
        .with_positional_reuse(PositionalOptions {
            secondary: SecondaryConfig::default(),
            control,
        })
        .unwrap()
}

/// Deterministic byte tokens: an LCG stream per `stream`.
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

fn identity() -> ptr_model_api::TokenizerIdentity {
    ptr_model_api::ByteTokenizer::new().identity().clone()
}

fn raw(stream: u64, count: usize) -> PromptSegment {
    PromptSegment::raw_from_tokens(identity(), tokens(stream, count)).unwrap()
}

fn memory(id: &str, generation: u64, stream: u64, count: usize) -> PromptSegment {
    PromptSegment::from_tokens(
        identity(),
        SegmentSource::Memory {
            id: KnowledgeObjectId::from(id),
            generation: Generation(generation),
        },
        tokens(stream, count),
    )
    .unwrap()
}

fn semantic(key: &str, generation: u64, stream: u64, count: usize) -> PromptSegment {
    PromptSegment::from_tokens(
        identity(),
        SegmentSource::Semantic {
            key: key.to_owned(),
            generation: Generation(generation),
        },
        tokens(stream, count),
    )
    .unwrap()
}

fn request(segments: Vec<PromptSegment>) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from("r"),
        ptr_types::SemanticContext {
            revision: Revision(0),
            entries: Vec::new(),
        },
        identity(),
        segments,
    )
    .unwrap()
}

/// The reused segment.
fn doc() -> PromptSegment {
    memory("mem:doc", 1, 11, SEGMENT_TOKENS)
}

fn tail() -> PromptSegment {
    raw(12, TAIL_TOKENS)
}

/// `[C, tail]`: the origin run that populates the index.
fn origin() -> SegmentedModelRequest {
    request(vec![doc(), tail()])
}

/// `[filler of s tokens, C, tail]`: the shifted request and its baseline.
fn shifted(shift: usize) -> SegmentedModelRequest {
    request(vec![raw(13, shift), doc(), tail()])
}

fn cold(backend: &ReferenceKvReuseBackend, request: &SegmentedModelRequest) -> ReferenceKvProbe {
    backend
        .probe(&scope(), request, ReferenceKvProbeMode::Cold)
        .unwrap()
}

fn run_positional(
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
) -> ReferenceKvProbe {
    backend
        .probe_positional(&scope(), request, &backend.own_liveness())
        .unwrap()
}

/// Bitwise comparison of two probes: logits bits, KV-state digest, generated tokens, decode
/// logits digest and the emitted events.
fn assert_bitwise_equal(label: &str, expected: &ReferenceKvProbe, actual: &ReferenceKvProbe) {
    assert_eq!(
        expected.logit_bits(),
        actual.logit_bits(),
        "{label}: prompt logits"
    );
    assert_eq!(
        expected.kv_state_digest, actual.kv_state_digest,
        "{label}: KV state"
    );
    assert_eq!(
        expected.outcome.generated, actual.outcome.generated,
        "{label}: generated"
    );
    assert_eq!(
        expected.outcome.logits_digest, actual.outcome.logits_digest,
        "{label}: decode logits"
    );
    assert_eq!(
        expected.outcome.run.events, actual.outcome.run.events,
        "{label}: events"
    );
}

fn report(outcome: &ReferenceKvOutcome) -> &ptr_runtime::PositionalReport {
    outcome
        .positional
        .as_ref()
        .expect("a positional session reports")
}

fn prefill_until_done(session: &mut KvSession<'_>, live: &dyn KvLivenessOracle) {
    while session.phase() == KvSessionPhase::Prefilling {
        match session.prefill_chunk(live).unwrap() {
            KvSessionStep::Prefilled { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }
}

fn completed(finish: KvSessionFinish) -> ReferenceKvOutcome {
    match finish {
        KvSessionFinish::Completed(outcome) => *outcome,
        KvSessionFinish::Cancelled(cancelled) => panic!("cancelled: {cancelled:?}"),
    }
}

fn revoke(backend: &ReferenceKvReuseBackend, identity: &str, generation: u64) {
    backend
        .invalidate_inputs(&BTreeSet::from([(
            identity.to_owned(),
            Generation(generation),
        )]))
        .unwrap();
}

fn lag(backend: &ReferenceKvReuseBackend) {
    backend
        .set_faults(ReferenceKvFaults {
            index_update: IndexUpdate::Suppressed,
        })
        .unwrap();
}

// ---- parity -----------------------------------------------------------------------------

/// One origin run, then the shifted request cold and through the secondary index.
fn assert_shift_parity(backend: &ReferenceKvReuseBackend, shift: usize) -> ReferenceKvProbe {
    let origin = run_positional(backend, &origin());
    let origin_report = report(&origin.outcome);
    assert_eq!(origin_report.inserted_segments, 2, "C and the tail");
    assert_eq!(
        origin_report.reused_tokens, 0,
        "the first run reuses nothing"
    );
    assert_eq!(origin.outcome.positional.as_ref().unwrap().lookups, 1);

    let request = shifted(shift);
    let baseline = cold(backend, &request);
    let warm = run_positional(backend, &request);
    let warm_report = report(&warm.outcome);
    assert_eq!(warm_report.reused_tokens, SEGMENT_TOKENS, "shift {shift}");
    assert_eq!(warm_report.splices, 1, "shift {shift}");
    assert_eq!(warm_report.admitted_splices, 1, "shift {shift}");
    assert_eq!(warm.outcome.hit_tokens, 0, "the primary prefix cannot hit");
    assert_eq!(
        warm.outcome.prefilled_tokens,
        shift + TAIL_TOKENS,
        "shift {shift}: only the filler and the tail are computed"
    );
    assert_eq!(
        warm.outcome.prompt_tokens,
        warm.outcome.hit_tokens + warm_report.reused_tokens + warm.outcome.prefilled_tokens
    );
    assert!(warm_report.insert_refusals.is_empty());
    let reuse = warm.outcome.run.kv_report.unwrap();
    assert_eq!(reuse.mode(), KvReuseMode::Exact);
    assert_eq!(reuse.hit_tokens(), SEGMENT_TOKENS);
    assert_eq!(baseline.outcome.run.kv_report.unwrap().hit_tokens(), 0);
    assert_bitwise_equal(&format!("shift {shift}"), &baseline, &warm);
    warm
}

#[test]
fn positional_reuse_is_bitwise_equal_to_a_cold_prefill_at_small_shifts() {
    for shift in [1usize, 7, 16, 17, 64] {
        let backend = positional(256);
        assert_shift_parity(&backend, shift);
        let stats = backend.secondary_stats().unwrap();
        assert_eq!(stats.active_pins, 0, "every pin is released at finish");
        assert_eq!(stats.reused_tokens, SEGMENT_TOKENS as u64);
    }
}

#[test]
fn positional_reuse_is_bitwise_equal_to_a_cold_prefill_at_shift_513() {
    let backend = positional(640);
    assert_shift_parity(&backend, 513);
}

/// R006's longest shift. A cold prefill of 4,160 tokens costs about a minute per run in
/// release (the stack re-snapshots the whole cache for every token), so this is release-only.
#[test]
#[ignore = "release-only: three prefills of up to 4,160 tokens"]
fn positional_reuse_is_bitwise_equal_to_a_cold_prefill_at_shift_4096() {
    let backend = positional(4352);
    assert_shift_parity(&backend, 4096);
}

#[test]
fn non_aligned_insertions_of_one_and_seven_tokens_are_reused_bit_for_bit() {
    for inserted in [1usize, 7] {
        let backend = positional(256);
        let prefix = raw(21, 24);
        let origin_request = request(vec![prefix.clone(), doc(), tail()]);
        let origin = run_positional(&backend, &origin_request);
        assert_eq!(report(&origin.outcome).inserted_segments, 3);

        let request = request(vec![prefix, raw(22, inserted), doc(), tail()]);
        let start = 24 + inserted;
        assert_ne!(start % 16, 0, "the segment moved to an unaligned offset");
        let baseline = cold(&backend, &request);
        let warm = run_positional(&backend, &request);
        let warm_report = report(&warm.outcome);
        assert_eq!(warm_report.reused_tokens, SEGMENT_TOKENS, "+{inserted}");
        assert_eq!(warm_report.splices, 1);
        assert_bitwise_equal(&format!("+{inserted}"), &baseline, &warm);
    }
}

#[test]
fn a_short_segment_is_found_by_its_short_key_at_a_new_offset() {
    let backend = positional(256);
    let short = semantic("sem:short", 1, 31, 5);
    let origin_request = request(vec![short.clone(), doc(), tail()]);
    run_positional(&backend, &origin_request);
    let request = request(vec![raw(32, 9), short, doc(), tail()]);
    let baseline = cold(&backend, &request);
    let warm = run_positional(&backend, &request);
    let warm_report = report(&warm.outcome);
    assert_eq!(warm_report.splices, 2);
    assert_eq!(warm_report.reused_tokens, 5 + SEGMENT_TOKENS);
    assert_bitwise_equal("short", &baseline, &warm);
}

#[test]
fn several_spans_are_repacked_in_request_order_and_reordering_is_allowed() {
    let backend = positional(256);
    let first = memory("mem:a", 1, 41, 40);
    let second = semantic("sem:b", 1, 42, 33);
    let origin_request = request(vec![first.clone(), second.clone(), tail()]);
    run_positional(&backend, &origin_request);
    // The second segment now comes first; the first one follows a new filler.
    let request = request(vec![raw(43, 5), second, raw(44, 3), first, tail()]);
    let baseline = cold(&backend, &request);
    let warm = run_positional(&backend, &request);
    let warm_report = report(&warm.outcome);
    assert_eq!(warm_report.splices, 2, "{warm_report:?}");
    assert_eq!(warm_report.reused_tokens, 33 + 40);
    assert_eq!(warm.outcome.prefilled_tokens, 5 + 3 + TAIL_TOKENS);
    assert_bitwise_equal("two spans", &baseline, &warm);
}

#[test]
fn a_span_moving_to_offset_zero_is_repacked_as_the_first_step() {
    let backend = positional(256);
    let ancestor = raw(51, 20);
    run_positional(&backend, &request(vec![ancestor.clone(), doc(), tail()]));
    // C moves to offset 0 (a negative shift) and its raw ancestor behind it: the multiset of
    // the whole request still holds it, so reordering is allowed.
    let request = request(vec![doc(), ancestor, tail()]);
    let baseline = cold(&backend, &request);
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session_positional(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(session.next_splice_start(), Some(0));
    assert_eq!(
        session.prefill_chunk(&live).unwrap(),
        KvSessionStep::Prefilled {
            position: SEGMENT_TOKENS
        },
        "the repack is a step of its own"
    );
    assert_eq!(session.next_splice_start(), Some(SEGMENT_TOKENS));
    assert_eq!(
        session.prefill_chunk(&live).unwrap(),
        KvSessionStep::Prefilled {
            position: SEGMENT_TOKENS + 20
        },
        "a second span follows directly"
    );
    assert_eq!(session.next_splice_start(), None);
    prefill_until_done(&mut session, &live);
    let logits = session.prompt_logits().unwrap().to_vec();
    let digest = session.kv_state_digest().unwrap();
    let outcome = completed(session.run_to_end(&live).unwrap());
    assert_eq!(report(&outcome).reused_tokens, SEGMENT_TOKENS + 20);
    assert_eq!(report(&outcome).splices, 2);
    assert_eq!(
        logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        baseline.logit_bits()
    );
    assert_eq!(digest, baseline.kv_state_digest);
    assert_eq!(outcome.generated, baseline.outcome.generated);
}

#[test]
fn a_span_below_the_primary_hit_is_dropped_and_its_pin_released() {
    let backend = positional(256);
    // 74 tokens: the primary catalog caches the boundaries up to 64 and may hit there.
    let request = request(vec![doc(), tail(), raw(14, 10)]);
    // A warm non-positional run fills the primary catalog; the positional run then finds the
    // same segment at its original offset, below the primary hit.
    run_positional(&backend, &request);
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session_positional(&scope(), &request, None, &live)
        .unwrap();
    assert!(session.hit_tokens() > 0, "the primary catalog hits");
    assert_eq!(
        session.positional_report().unwrap().dropped_below_primary,
        2,
        "C and the 16-token segment after it start below the primary hit"
    );
    assert_eq!(session.positional_report().unwrap().splices, 0);
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 0);
    let baseline = cold(&backend, &request);
    let outcome = completed(session.run_to_end(&live).unwrap());
    assert_eq!(report(&outcome).reused_tokens, 0);
    assert_eq!(outcome.generated, baseline.outcome.generated);
    assert_eq!(outcome.logits_digest, baseline.outcome.logits_digest);
}

#[test]
fn a_primary_hit_followed_by_a_shifted_span_uses_both() {
    let backend = positional(256);
    let head = raw(61, 40);
    run_positional(&backend, &request(vec![head.clone(), doc(), tail()]));
    // The same head (primary prefix hit for its first chunks), then a new filler, then C.
    let request = request(vec![head, raw(62, 9), doc(), tail()]);
    let baseline = cold(&backend, &request);
    let warm = run_positional(&backend, &request);
    let warm_report = report(&warm.outcome);
    assert!(warm.outcome.hit_tokens >= 32, "{}", warm.outcome.hit_tokens);
    assert_eq!(warm_report.reused_tokens, SEGMENT_TOKENS);
    assert_bitwise_equal("primary + secondary", &baseline, &warm);
}

// ---- opt-in --------------------------------------------------------------------------------

#[test]
fn positional_reuse_is_off_by_default_and_changes_no_existing_path() {
    let plain =
        ReferenceKvReuseBackend::with_layers(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED, 1)
            .unwrap();
    assert!(matches!(
        plain.begin_session_positional(&scope(), &origin(), None, &plain.own_liveness()),
        Err(ReferenceKvError::PositionalNotEnabled)
    ));
    assert!(matches!(
        plain.probe_positional(&scope(), &origin(), &plain.own_liveness()),
        Err(ReferenceKvError::PositionalNotEnabled)
    ));

    // The same requests through every existing entry point of a backend that opted in leave
    // the index untouched and give exactly the outcomes of a backend that did not.
    let opted = positional(256);
    for request in [origin(), shifted(7), shifted(7), origin()] {
        let expected = plain.infer_scoped(&scope(), &request, None).unwrap();
        let actual = opted.infer_scoped(&scope(), &request, None).unwrap();
        assert_eq!(expected, actual);
        assert!(actual.positional.is_none());
    }
    for mode in [ReferenceKvProbeMode::Cold, ReferenceKvProbeMode::Warm] {
        assert_eq!(
            plain.probe(&scope(), &shifted(16), mode).unwrap(),
            opted.probe(&scope(), &shifted(16), mode).unwrap()
        );
    }
    for backend in [&plain, &opted] {
        let live = backend.own_liveness();
        let mut session = backend
            .begin_session(&scope(), &shifted(16), None, &live)
            .unwrap();
        let outcome = completed(session.run_to_end(&live).unwrap());
        assert!(outcome.positional.is_none());
    }
    let stats = opted.secondary_stats().unwrap();
    assert_eq!((stats.lookups, stats.inserts, stats.runs), (0, 0, 0));
    assert_eq!(
        plain.catalog_stats().unwrap(),
        opted.catalog_stats().unwrap()
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Everything the opt-in-off path must keep reproducing bit for bit.
fn default_backend_fingerprint() -> [String; 5] {
    let default =
        ReferenceKvReuseBackend::with_seed(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED).unwrap();
    let config = hex(&Sha256::digest(
        format!("{:?}", default.stack_config()).as_bytes(),
    ));
    let probe = default
        .probe(&scope(), &origin(), ReferenceKvProbeMode::Cold)
        .unwrap();
    let warm = default.infer_scoped(&scope(), &origin(), None).unwrap();
    [
        hex(&default.chain_id().0),
        config,
        hex(&probe.kv_state_digest),
        hex(&probe.outcome.logits_digest),
        hex(&warm.logits_digest),
    ]
}

/// Literals computed from the code at the base commit cb2db074, before any R006 positional
/// change: the opt-in-off path must not drift. `config` is the SHA-256 of the `Debug` text of
/// the stack configuration.
#[test]
fn the_default_backend_fingerprint_is_pinned_to_the_pre_r006_literals() {
    let [chain, config, kv_state, probe_logits, infer_logits] = default_backend_fingerprint();
    assert_eq!(
        chain,
        "9a14f47059404a447f1c12044908d91dec08a158ed6e1fdfe6f6bc872b7861ee"
    );
    assert_eq!(
        config,
        "064a7c65166ca7fafd2986f906dfc63d0d5d2e044fb23403eb4f25575d0fcb05"
    );
    assert_eq!(
        kv_state,
        "97e1598049890c3aff69568318364ebde5479ef034937551a1157c4f58dec180"
    );
    assert_eq!(
        probe_logits,
        "ff315dbdb6cf20f6aa65bc5f988c459b6a7d5727df500f9050bdd73be26b09ab"
    );
    assert_eq!(infer_logits, probe_logits);
}

#[test]
fn the_default_four_layer_backend_is_unchanged_and_refuses_positional_reuse() {
    let default =
        ReferenceKvReuseBackend::with_seed(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED).unwrap();
    let explicit =
        ReferenceKvReuseBackend::with_layers(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED, 4)
            .unwrap();
    assert_eq!(default.chain_id(), explicit.chain_id());
    assert_eq!(default.stack_config().num_layers, 4);
    assert_eq!(
        default.infer_scoped(&scope(), &origin(), None).unwrap(),
        explicit.infer_scoped(&scope(), &origin(), None).unwrap()
    );
    assert!(matches!(
        explicit.with_positional_reuse(PositionalOptions::default()),
        Err(ReferenceKvError::PositionalNeedsOneLayer { layers: 4 })
    ));
    let one =
        ReferenceKvReuseBackend::with_layers(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED, 1)
            .unwrap();
    assert_ne!(one.chain_id(), default.chain_id());
    let post_rope = ReferenceKvReuseBackend::with_stack_shape(
        limits(256),
        scope(),
        REFERENCE_KV_WEIGHT_SEED,
        Qwen2RopeScaling::None,
        ReferenceKvStackShape {
            layers: 1,
            key_encoding: KvKeyEncoding::PostRope,
        },
    )
    .unwrap();
    assert!(matches!(
        post_rope.with_positional_reuse(PositionalOptions::default()),
        Err(ReferenceKvError::PositionalNeedsPreRope)
    ));
    assert!(matches!(
        ReferenceKvReuseBackend::with_layers(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED, 0),
        Err(ReferenceKvError::Stack(_))
    ));
}

#[test]
fn the_one_layer_config_digest_is_the_one_r006_pins() {
    let backend = positional(256);
    let config = backend.stack_config();
    let layer = &config.layer;
    let canonical = format!(
        "{{\"head_dim\":{},\"hidden_size\":{},\"layers\":{},\"num_heads\":{},\"num_kv_heads\":{},\"vocab_size\":{},\"weight_seed\":{}}}",
        layer.hidden_size / layer.num_attention_heads,
        layer.hidden_size,
        config.num_layers,
        layer.num_attention_heads,
        layer.num_key_value_heads,
        config.vocab_size,
        REFERENCE_KV_WEIGHT_SEED
    );
    let digest: [u8; 32] = Sha256::digest(canonical.as_bytes()).into();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        hex,
        "a77e017c8ee3b56bb06a0b22512aae3a70a1b65d4ae283bf2d8bb02353029c2e"
    );
}

// ---- typed refusals through a session ----------------------------------------------------------

/// Run `request` positionally and require zero reuse, a run bitwise equal to cold, and return
/// the report for the caller to check the refusal counts.
fn refused_run(
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
) -> ptr_runtime::PositionalReport {
    refused_run_but(backend, request, 0)
}

/// Like [`refused_run`] when a segment unrelated to the refusal is still reused:
/// `reused` tokens, all of them admitted spans.
fn refused_run_but(
    backend: &ReferenceKvReuseBackend,
    request: &SegmentedModelRequest,
    reused: usize,
) -> ptr_runtime::PositionalReport {
    let baseline = cold(backend, request);
    let warm = run_positional(backend, request);
    let warm_report = report(&warm.outcome).clone();
    assert_eq!(warm_report.reused_tokens, reused, "{warm_report:?}");
    assert_eq!(warm_report.splices, warm_report.admitted_splices);
    assert_bitwise_equal("refused run", &baseline, &warm);
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 0);
    warm_report
}

#[test]
fn an_own_lineage_that_differs_is_refused_for_the_named_reason() {
    // Same tokens, another generation.
    let backend = positional(256);
    run_positional(&backend, &origin());
    let r = refused_run(
        &backend,
        &request(vec![
            raw(13, 7),
            memory("mem:doc", 2, 11, SEGMENT_TOKENS),
            tail(),
        ]),
    );
    assert!(r.refusals.source_generation_mismatch >= 1, "{r:?}");
    assert_eq!(r.refusals.source_identity_mismatch, 0);
    assert_eq!(r.refusals.source_kind_mismatch, 0);
    assert_eq!(r.refusals.source_digest_mismatch, 0);

    // Same tokens, another identity.
    let backend = positional(256);
    run_positional(&backend, &origin());
    let r = refused_run(
        &backend,
        &request(vec![
            raw(13, 7),
            memory("mem:other", 1, 11, SEGMENT_TOKENS),
            tail(),
        ]),
    );
    assert!(r.refusals.source_identity_mismatch >= 1, "{r:?}");
    assert_eq!(r.refusals.source_generation_mismatch, 0);

    // Same tokens, another kind.
    let backend = positional(256);
    run_positional(&backend, &origin());
    let r = refused_run(
        &backend,
        &request(vec![
            raw(13, 7),
            semantic("mem:doc", 1, 11, SEGMENT_TOKENS),
            tail(),
        ]),
    );
    assert!(r.refusals.source_kind_mismatch >= 1, "{r:?}");
}

#[test]
fn an_edit_inside_a_segment_gives_zero_reuse_and_a_digest_refusal() {
    let backend = positional(256);
    run_positional(&backend, &origin());
    let mut edited = tokens(11, SEGMENT_TOKENS);
    edited[SEGMENT_TOKENS / 2] = TokenId((edited[SEGMENT_TOKENS / 2].0 + 1) % 256);
    let edited_segment = PromptSegment::from_tokens(
        identity(),
        SegmentSource::Memory {
            id: KnowledgeObjectId::from("mem:doc"),
            generation: Generation(1),
        },
        edited,
    )
    .unwrap();
    let r = refused_run(&backend, &request(vec![raw(13, 7), edited_segment, tail()]));
    assert!(r.refusals.source_digest_mismatch >= 1, "{r:?}");
    assert_eq!(r.refusals.source_generation_mismatch, 0);
}

#[test]
fn an_omitted_original_upstream_segment_is_refused_for_the_named_reason() {
    let backend = positional(256);
    let upstream = semantic("sem:u", 1, 71, 20);
    run_positional(&backend, &request(vec![upstream.clone(), doc(), tail()]));
    // The request keeps C but no longer carries U.
    let r = refused_run(&backend, &request(vec![raw(13, 7), doc(), tail()]));
    assert!(r.refusals.original_upstream_omitted >= 1, "{r:?}");
    assert_eq!(r.refusals.original_upstream_dead, 0);

    // The multiplicity case: the run saw U twice, the request has it once.
    let backend = positional(256);
    run_positional(
        &backend,
        &request(vec![upstream.clone(), upstream.clone(), doc(), tail()]),
    );
    // (The first copy of U, which had nothing before it, is still reused: 20 tokens.)
    let r = refused_run_but(
        &backend,
        &request(vec![raw(13, 7), upstream.clone(), doc(), tail()]),
        20,
    );
    assert!(r.refusals.original_upstream_omitted >= 1, "{r:?}");

    // Reordering is not omission: U and C swap places and the run is admitted.
    let backend = positional(256);
    run_positional(&backend, &request(vec![upstream.clone(), doc(), tail()]));
    let reordered = request(vec![doc(), raw(13, 7), upstream, tail()]);
    let baseline = cold(&backend, &reordered);
    let warm = run_positional(&backend, &reordered);
    assert!(report(&warm.outcome).reused_tokens >= SEGMENT_TOKENS);
    assert_bitwise_equal("reordered", &baseline, &warm);
}

#[test]
fn a_removed_raw_ancestor_is_refused_for_the_named_reason() {
    let backend = positional(256);
    run_positional(&backend, &request(vec![raw(81, 20), doc(), tail()]));
    let r = refused_run(&backend, &request(vec![raw(13, 7), doc(), tail()]));
    assert!(r.refusals.raw_ancestor_omitted >= 1, "{r:?}");
    assert_eq!(r.refusals.original_upstream_omitted, 0);
}

#[test]
fn a_dead_original_upstream_is_refused_when_the_index_lags() {
    let backend = positional(256);
    let upstream = semantic("sem:u", 1, 71, 20);
    run_positional(&backend, &request(vec![upstream.clone(), doc(), tail()]));
    lag(&backend);
    revoke(&backend, "sem:u", 1);
    assert_eq!(
        backend.secondary_stats().unwrap().invalidated,
        0,
        "the lagging index never heard of it"
    );
    let r = refused_run(
        &backend,
        &request(vec![raw(13, 7), upstream, doc(), tail()]),
    );
    assert!(r.refusals.original_upstream_dead >= 1, "{r:?}");
    assert_eq!(r.refusals.original_upstream_omitted, 0);
    assert!(backend.secondary_stats().unwrap().tombstoned_at_lookup >= 1);
}

#[test]
fn a_dead_binding_dependency_is_refused_when_the_index_lags() {
    let backend = positional(256);
    run_positional(&backend, &origin());
    lag(&backend);
    revoke(&backend, BASE_MODEL, 1);
    let r = refused_run(&backend, &shifted(7));
    assert!(r.refusals.binding_dependency_dead >= 1, "{r:?}");
}

// ---- revocation and supersession ------------------------------------------------------------------

#[test]
fn an_applied_revocation_or_supersession_makes_the_run_unreachable() {
    for supersede in [false, true] {
        let backend = positional(256);
        let upstream = semantic("sem:u", 1, 71, 20);
        run_positional(&backend, &request(vec![upstream.clone(), doc(), tail()]));
        assert_eq!(backend.secondary_stats().unwrap().runs, 3);
        if supersede {
            backend
                .supersede_inputs(&std::collections::BTreeMap::from([(
                    "sem:u".to_owned(),
                    Generation(2),
                )]))
                .unwrap();
        } else {
            revoke(&backend, "sem:u", 1);
        }
        let stats = backend.secondary_stats().unwrap();
        assert!(stats.invalidated >= 2, "C's run and U's own run");
        let lookups_before = stats.lookups;
        // A request citing the same (now dead) U and the same content: the index no longer
        // reaches any run, so there is no candidate, no refusal and no admission.
        let request = request(vec![raw(13, 7), upstream.clone(), doc(), tail()]);
        let baseline = cold(&backend, &request);
        let warm = run_positional(&backend, &request);
        let warm_report = report(&warm.outcome);
        assert_eq!(warm_report.reused_tokens, 0, "supersede {supersede}");
        assert_eq!(warm_report.admitted_splices, 0);
        assert_eq!(warm_report.refusals.original_upstream_dead, 0);
        assert_eq!(
            backend.secondary_stats().unwrap().lookups,
            lookups_before + 1
        );
        assert_bitwise_equal("fresh prefill after the event", &baseline, &warm);
        // The dead lineage was not live at begin, so nothing from this request entered the
        // index either.
        assert!(
            warm_report.inserted_segments <= 1,
            "only the live raw segments"
        );
    }
}

#[test]
fn the_liveness_bypass_admits_a_dead_run_that_the_normal_path_refuses() {
    let upstream = semantic("sem:u", 1, 71, 20);
    let setup = |control| {
        let backend = positional_with(256, control);
        run_positional(&backend, &request(vec![upstream.clone(), doc(), tail()]));
        lag(&backend);
        revoke(&backend, "sem:u", 1);
        backend
    };
    let shifted_request = request(vec![raw(13, 7), upstream.clone(), doc(), tail()]);

    let normal = setup(PositionalControl::None);
    let warm = run_positional(&normal, &shifted_request);
    assert_eq!(report(&warm.outcome).reused_tokens, 0);
    assert!(report(&warm.outcome).refusals.original_upstream_dead >= 1);
    assert_eq!(report(&warm.outcome).bypass_dead_admissions, 0);

    let bypass = setup(PositionalControl::BypassUseTimeLiveness);
    let warm = run_positional(&bypass, &shifted_request);
    let bypass_report = report(&warm.outcome);
    assert!(bypass_report.liveness_bypassed);
    assert!(
        bypass_report.admitted_splices >= 1 && bypass_report.reused_tokens > 0,
        "the dead run is admitted: {bypass_report:?}"
    );
    assert!(bypass_report.bypass_dead_admissions >= 1);
    assert_eq!(bypass_report.refusals.original_upstream_dead, 0);
    // In one layer the output cannot tell: it is still bit-equal to a cold prefill. Only the
    // structural counters above show the stale hit.
    assert_bitwise_equal(
        "stale admission is numerically invisible",
        &cold(&bypass, &shifted_request),
        &warm,
    );
}

// ---- in-flight ---------------------------------------------------------------------------------------

/// `[filler of 40, C, tail]` after an origin run, with a positional session open and not yet
/// stepped.
struct InFlight {
    backend: ReferenceKvReuseBackend,
    request: SegmentedModelRequest,
}

const FILLER: usize = 40;

fn in_flight(control: PositionalControl, doc_segment: PromptSegment) -> InFlight {
    let backend = positional_with(256, control);
    let origin_request = request(vec![doc_segment.clone(), tail()]);
    run_positional(&backend, &origin_request);
    let request = request(vec![raw(13, FILLER), doc_segment, tail()]);
    InFlight { backend, request }
}

fn step_to(session: &mut KvSession<'_>, live: &dyn KvLivenessOracle, position: usize) {
    while session.position() < position {
        match session.prefill_chunk(live).unwrap() {
            KvSessionStep::Prefilled { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(session.position(), position);
}

/// The event hits at `when`; the next step must be cancelled with `reason`, nothing may be
/// emitted or inserted after it, every pin is released and a cold restart equals a cold run.
#[derive(Clone, Copy, Debug)]
enum When {
    BeforeSplice,
    AfterSpliceBeforeTail,
    AfterPrefillBeforeDecode,
}

fn inflight_case(
    doc_segment: PromptSegment,
    event: &dyn Fn(&ReferenceKvReuseBackend),
    when: When,
    reason: KvCancelReason,
) {
    let InFlight { backend, request } = in_flight(PositionalControl::None, doc_segment);
    let live = backend.own_liveness();
    let baseline = cold(&backend, &request);
    let mut session = backend
        .begin_session_positional(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(session.next_splice_start(), Some(FILLER));
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 1);
    step_to(&mut session, &live, FILLER);
    match when {
        When::BeforeSplice => {}
        When::AfterSpliceBeforeTail => {
            session.prefill_chunk(&live).unwrap();
            assert_eq!(session.positional_report().unwrap().splices, 1);
        }
        When::AfterPrefillBeforeDecode => {
            prefill_until_done(&mut session, &live);
            assert_eq!(session.phase(), KvSessionPhase::Decoding);
        }
    }
    let inserts_before = backend.secondary_stats().unwrap().inserts;
    let entries_before = backend.catalog_stats().unwrap().entries;
    let splices_before = session.positional_report().unwrap().splices;
    event(&backend);

    let step = match when {
        When::BeforeSplice | When::AfterSpliceBeforeTail => session.prefill_chunk(&live).unwrap(),
        When::AfterPrefillBeforeDecode => session.decode_step(&live).unwrap(),
    };
    let KvSessionStep::Cancelled(cancelled) = step else {
        panic!("{when:?}: expected a cancellation, got {step:?}");
    };
    assert_eq!(cancelled.reason, reason, "{when:?}");
    assert_eq!(
        cancelled.tokens_emitted, 0,
        "{when:?}: no token after the event"
    );
    assert_eq!(session.phase(), KvSessionPhase::Cancelled);
    assert!(session.generated().is_empty());
    assert_eq!(
        session.positional_report().unwrap().splices,
        splices_before,
        "{when:?}: no span repacked after the event"
    );
    let stats = backend.secondary_stats().unwrap();
    assert_eq!(
        stats.inserts, inserts_before,
        "{when:?}: no insert after the event"
    );
    // The event itself may retire primary entries; nothing was added after it.
    assert!(backend.catalog_stats().unwrap().entries <= entries_before);
    assert_eq!(
        stats.active_pins, 0,
        "{when:?}: the pin is released at cancel"
    );

    // The restart is cold: no lookup, no insert, no pin, bitwise equal to a cold run.
    let lookups = stats.lookups;
    let restarted = session.restart_with(&live).unwrap();
    assert!(restarted.restarted_after_cancel);
    let restart_report = report(&restarted);
    assert!(restart_report.cold_restart);
    assert_eq!(
        (
            restart_report.reused_tokens,
            restart_report.splices,
            restart_report.lookups
        ),
        (0, 0, 0)
    );
    assert_eq!(restarted.hit_tokens, 0);
    assert_eq!(restarted.prefilled_tokens, request.tokens().len());
    assert_eq!(restarted.generated, baseline.outcome.generated);
    assert_eq!(restarted.logits_digest, baseline.outcome.logits_digest);
    assert_eq!(restarted.run.events, baseline.outcome.run.events);
    assert_eq!(
        restarted.run.kv_report.unwrap().hit_tokens(),
        0,
        "a cold restart reports no reuse"
    );
    assert_eq!(session.kv_state_digest().unwrap(), baseline.kv_state_digest);
    let after = backend.secondary_stats().unwrap();
    assert_eq!(
        after.inserts, inserts_before,
        "{when:?}: the restart inserts nothing"
    );
    assert_eq!(
        after.lookups, lookups,
        "{when:?}: the restart looks nothing up"
    );
    assert_eq!(after.active_pins, 0);
}

const ALL_WHEN: [When; 3] = [
    When::BeforeSplice,
    When::AfterSpliceBeforeTail,
    When::AfterPrefillBeforeDecode,
];

#[test]
fn a_revoked_memory_segment_cancels_a_positional_session_before_and_after_the_splice() {
    for when in ALL_WHEN {
        inflight_case(
            memory("mem:doc", 1, 11, SEGMENT_TOKENS),
            &|backend| revoke(backend, "mem:doc", 1),
            when,
            KvCancelReason::InputNotLive,
        );
    }
}

#[test]
fn a_revoked_semantic_segment_cancels_a_positional_session_before_and_after_the_splice() {
    for when in ALL_WHEN {
        inflight_case(
            semantic("sem:doc", 1, 11, SEGMENT_TOKENS),
            &|backend| revoke(backend, "sem:doc", 1),
            when,
            KvCancelReason::InputNotLive,
        );
    }
}

#[test]
fn a_superseded_segment_cancels_a_positional_session() {
    for when in ALL_WHEN {
        inflight_case(
            memory("mem:doc", 1, 11, SEGMENT_TOKENS),
            &|backend| {
                backend
                    .supersede_inputs(&std::collections::BTreeMap::from([(
                        "mem:doc".to_owned(),
                        Generation(2),
                    )]))
                    .unwrap();
            },
            when,
            KvCancelReason::InputNotLive,
        );
    }
}

#[test]
fn a_revoked_base_model_cancels_a_positional_session_with_a_reason_naming_it() {
    for when in ALL_WHEN {
        inflight_case(
            memory("mem:doc", 1, 11, SEGMENT_TOKENS),
            &|backend| revoke(backend, BASE_MODEL, 1),
            when,
            KvCancelReason::SecondaryDependencyNotLive,
        );
    }
}

#[test]
fn a_lagging_index_does_not_stop_the_use_time_check_from_cancelling() {
    for when in ALL_WHEN {
        inflight_case(
            memory("mem:doc", 1, 11, SEGMENT_TOKENS),
            &|backend| {
                lag(backend);
                revoke(backend, BASE_MODEL, 1);
            },
            when,
            KvCancelReason::SecondaryDependencyNotLive,
        );
    }
}

#[test]
fn without_use_time_liveness_a_lagging_session_keeps_emitting_after_the_event() {
    // The negative control: the same lag and the same base-model revocation as above, with
    // the bypass. The session is not cancelled, so tokens are emitted and inserts are made
    // after the event, which the structural counters of the experiment must catch.
    let InFlight { backend, request } = in_flight(
        PositionalControl::BypassUseTimeLiveness,
        memory("mem:doc", 1, 11, SEGMENT_TOKENS),
    );
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session_positional(&scope(), &request, None, &live)
        .unwrap();
    step_to(&mut session, &live, FILLER);
    lag(&backend);
    revoke(&backend, BASE_MODEL, 1);
    session.prefill_chunk(&live).unwrap();
    prefill_until_done(&mut session, &live);
    let step = session.decode_step(&live).unwrap();
    assert!(
        matches!(step, KvSessionStep::Token(_)),
        "a token escaped after the event: {step:?}"
    );
    let outcome = completed(session.run_to_end(&live).unwrap());
    assert_eq!(report(&outcome).splices, 1);
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 0);
}

#[test]
fn an_applied_invalidation_still_tombstones_the_run_under_the_bypass() {
    // Only the use-time re-check is bypassed; an invalidation that did reach the index
    // still revokes the pin, and the session cancels naming the tombstoned run.
    let InFlight { backend, request } = in_flight(
        PositionalControl::BypassUseTimeLiveness,
        memory("mem:doc", 1, 11, SEGMENT_TOKENS),
    );
    let live = backend.own_liveness();
    let mut session = backend
        .begin_session_positional(&scope(), &request, None, &live)
        .unwrap();
    step_to(&mut session, &live, FILLER);
    revoke(&backend, BASE_MODEL, 1);
    let step = session.prefill_chunk(&live).unwrap();
    let KvSessionStep::Cancelled(cancelled) = step else {
        panic!("expected a cancellation, got {step:?}");
    };
    assert_eq!(cancelled.reason, KvCancelReason::SecondaryRunTombstoned);
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 0);
}

#[test]
fn every_secondary_pin_is_released_on_finish_and_on_drop() {
    let backend = positional(256);
    run_positional(&backend, &origin());
    let live = backend.own_liveness();
    let request = shifted(16);
    // Held while the session is open.
    let mut session = backend
        .begin_session_positional(&scope(), &request, None, &live)
        .unwrap();
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 1);
    session.run_to_end(&live).unwrap();
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 0);
    drop(session);
    // Released by an abandoned session as well.
    let session = backend
        .begin_session_positional(&scope(), &request, None, &live)
        .unwrap();
    // The first shifted run also taught the index its filler, so two spans are admitted now.
    let admitted = session.positional_report().unwrap().admitted_splices;
    assert_eq!(admitted, 2);
    assert_eq!(
        backend.secondary_stats().unwrap().active_pins,
        admitted as u64
    );
    drop(session);
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 0);
}

// ---- restore -------------------------------------------------------------------------------------------

#[test]
fn restore_carries_no_rows_and_a_re_insert_brings_the_reuse_back() {
    let source = positional(256);
    run_positional(&source, &origin());
    let snapshot = source.export_catalog().unwrap();
    assert!(snapshot.starts_with(b"PTRKVS04"));

    let target = positional(256);
    let restored = target.restore_catalog(&snapshot).unwrap();
    assert_eq!(restored.secondary_runs, 2);
    let request = shifted(16);
    // No rows survive a snapshot: the restored runs miss.
    let miss = run_positional(&target, &request);
    assert_eq!(report(&miss.outcome).reused_tokens, 0);
    assert_eq!(report(&miss.outcome).admitted_splices, 0);
    assert!(target.secondary_stats().unwrap().rows_missing_misses >= 1);
    // Re-running the origin content attaches the rows (digest-checked) ...
    let again = run_positional(&target, &origin());
    assert_eq!(report(&again.outcome).rows_attached, 2);
    // ... and the shifted request is served from the index again.
    let baseline = cold(&target, &request);
    let warm = run_positional(&target, &request);
    // C, and the 16-token filler the first shifted run above taught the index.
    assert_eq!(report(&warm.outcome).reused_tokens, SEGMENT_TOKENS + 16);
    assert_bitwise_equal("after restore", &baseline, &warm);
}

#[test]
fn a_restored_run_whose_upstream_died_at_rest_never_gets_its_rows_back() {
    let source = positional(256);
    let upstream = semantic("sem:u", 1, 71, 20);
    let origin_request = request(vec![upstream, doc(), tail()]);
    run_positional(&source, &origin_request);
    let snapshot = source.export_catalog().unwrap();

    let target = positional(256);
    target.restore_catalog(&snapshot).unwrap();
    revoke(&target, "sem:u", 1);
    let again = run_positional(&target, &origin_request);
    let again_report = report(&again.outcome);
    assert_eq!(again_report.rows_attached, 0);
    assert!(
        !again_report.insert_refusals.is_empty(),
        "DependencyNotLive is a graceful insert refusal"
    );
}

fn reseal_v4(mut body: Vec<u8>) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"ptr.kv.catalog-snapshot.v4\0");
    hasher.update(&body);
    body.extend_from_slice(&hasher.finalize());
    body
}

/// A v4 snapshot whose run for `doc()` records a wrong rows digest (re-sealed).
fn snapshot_with_wrong_doc_digest(snapshot: &[u8]) -> Vec<u8> {
    let pattern: Vec<u8> = tokens(11, SEGMENT_TOKENS)
        .iter()
        .flat_map(|token| token.0.to_le_bytes())
        .collect();
    let at = snapshot
        .windows(pattern.len())
        .position(|window| window == pattern.as_slice())
        .expect("the doc run's tokens are in the section");
    let mut body = snapshot[..snapshot.len() - 32].to_vec();
    body[at + pattern.len()] ^= 0xff; // first byte of the recorded rows digest
    reseal_v4(body)
}

#[test]
fn a_conflicting_restored_run_is_refused_and_tombstoned_and_never_fails_a_request() {
    let source = positional(256);
    run_positional(&source, &origin());
    let crafted = snapshot_with_wrong_doc_digest(&source.export_catalog().unwrap());

    let target = positional(256);
    assert_eq!(target.restore_catalog(&crafted).unwrap().secondary_runs, 2);

    // Re-running the origin offers the doc's rows: the digest conflicts. The request
    // completes, the conflict is a typed insert refusal and the other run gets its rows.
    let again = run_positional(&target, &origin());
    let again_report = report(&again.outcome);
    assert_eq!(
        again_report.insert_refusals,
        vec![ptr_runtime::SecondaryError::RowsDigestMismatch]
    );
    assert_eq!(again_report.rows_attached, 1);
    let stats = target.secondary_stats().unwrap();
    assert_eq!(stats.conflict_tombstones, 1);
    assert_eq!(stats.runs, 1, "the conflicting run is gone");

    // The conflicting run is never spliced, and a later request is not failed by it.
    let request = shifted(16);
    let baseline = cold(&target, &request);
    let first = run_positional(&target, &request);
    assert_eq!(report(&first.outcome).reused_tokens, 0);
    assert!(report(&first.outcome).insert_refusals.is_empty());
    assert_bitwise_equal("conflicting run not spliced", &baseline, &first);
    // The content re-entered as a fresh, correct run and now serves.
    let second = run_positional(&target, &request);
    assert_eq!(report(&second.outcome).reused_tokens, SEGMENT_TOKENS + 16);
    assert_bitwise_equal("fresh run serves", &baseline, &second);
}

#[test]
fn opting_in_is_refused_when_the_index_already_holds_restored_runs() {
    let source = positional(256);
    run_positional(&source, &origin());
    let snapshot = source.export_catalog().unwrap();

    let restored_first =
        ReferenceKvReuseBackend::with_layers(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED, 1)
            .unwrap();
    restored_first.restore_catalog(&snapshot).unwrap();
    assert!(matches!(
        restored_first.with_positional_reuse(PositionalOptions::default()),
        Err(ReferenceKvError::PositionalIndexNotEmpty)
    ));

    // Opting in first, then restoring, is the supported order.
    let target = positional(256);
    assert_eq!(target.restore_catalog(&snapshot).unwrap().secondary_runs, 2);
}

// ---- negative control: PostRope --------------------------------------------------------------------------

#[test]
fn a_post_rope_stack_mismatches_bitwise_which_makes_the_comparison_non_vacuous() {
    let backend = ReferenceKvReuseBackend::with_stack_shape(
        limits(256),
        scope(),
        REFERENCE_KV_WEIGHT_SEED,
        Qwen2RopeScaling::None,
        ReferenceKvStackShape {
            layers: 1,
            key_encoding: KvKeyEncoding::PostRope,
        },
    )
    .unwrap()
    .with_positional_reuse(PositionalOptions {
        secondary: SecondaryConfig::default(),
        control: PositionalControl::PermitPostRopeKeys,
    })
    .unwrap();
    run_positional(&backend, &origin());
    let request = shifted(7);
    let baseline = cold(&backend, &request);
    let warm = run_positional(&backend, &request);
    assert_eq!(
        report(&warm.outcome).reused_tokens,
        SEGMENT_TOKENS,
        "the span is reused ..."
    );
    // ... and its keys, rotated at the old position, are wrong at the new one.
    assert_ne!(baseline.kv_state_digest, warm.kv_state_digest);
    assert_ne!(baseline.logit_bits(), warm.logit_bits());
}

// ---- bounds ------------------------------------------------------------------------------------------------

#[test]
fn a_full_index_declines_the_insert_gracefully_and_the_output_is_unchanged() {
    let backend =
        ReferenceKvReuseBackend::with_layers(limits(256), scope(), REFERENCE_KV_WEIGHT_SEED, 1)
            .unwrap()
            .with_positional_reuse(PositionalOptions {
                secondary: SecondaryConfig {
                    window_tokens: 8,
                    limits: SecondaryLimits {
                        max_row_bytes: 1,
                        ..SecondaryLimits::default()
                    },
                },
                control: PositionalControl::None,
            })
            .unwrap();
    let request = origin();
    let baseline = cold(&backend, &request);
    let warm = run_positional(&backend, &request);
    let warm_report = report(&warm.outcome);
    assert_eq!(warm_report.inserted_segments, 0);
    assert_eq!(warm_report.insert_refusals.len(), 2);
    assert_bitwise_equal("capacity", &baseline, &warm);
}

#[test]
fn a_prompt_over_the_limit_is_refused_before_anything_is_pinned() {
    let backend = positional(64);
    assert!(matches!(
        backend.begin_session_positional(&scope(), &shifted(40), None, &backend.own_liveness()),
        Err(ReferenceKvError::PromptTooLong { .. })
    ));
    assert_eq!(backend.secondary_stats().unwrap().active_pins, 0);
}
