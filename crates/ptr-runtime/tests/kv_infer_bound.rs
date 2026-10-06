//! Bound sessions of the reference KV-reuse backend (WP4): per-layer entries for adapters,
//! aLoRA and MoE experts, fork and compute-then-adopt assembly, Approximate routing and the
//! re-resolving restart. Public API only. Not an R009 or R010 result.

use ptr_model_api::{
    KvReuseHint, KvReuseMode, KvReusePolicy, PromptSegment, SegmentSource, SegmentedModelRequest,
    Tokenizer,
};
use ptr_pods::{
    Qwen2Adapter, Qwen2AdapterKind, Qwen2AdapterSpec, Qwen2Moe, Qwen2MoeSpec, Qwen2RoutingMode,
    RoutingDeterminism,
};
use ptr_runtime::{
    BindingError, BindingSuccessors, BoundExpert, ExpertSuccessor, KvCancelReason, KvDependency,
    KvDependencyKind, KvLivenessOracle, KvSession, KvSessionFinish, KvSessionPhase, KvSessionStep,
    LayerAssembly, MoeBinding, ReferenceKvError, ReferenceKvLimits, ReferenceKvOutcome,
    ReferenceKvReuseBackend, ReferenceKvScope, StackBinding,
};
use ptr_types::{Generation, PodId, PrincipalId, ProjectId, RequestId, Revision, SemanticContext};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};

const SEED: u64 = 20240607;
const LAYERS: usize = 4;
const PROMPT: usize = 70;

fn scope() -> ReferenceKvScope {
    ReferenceKvScope {
        principal: PrincipalId::from("alice"),
        project: ProjectId::from("p"),
    }
}

fn backend() -> ReferenceKvReuseBackend {
    ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap()
}

fn request(backend: &ReferenceKvReuseBackend, id: &str, body: &str) -> SegmentedModelRequest {
    let segment = PromptSegment::raw_from_text(backend.tokenizer(), body).unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        SemanticContext {
            revision: Revision(1),
            entries: Vec::new(),
        },
        backend.tokenizer().identity().clone(),
        vec![segment],
    )
    .unwrap()
}

/// `count` ASCII bytes, no repeated 16-byte chunk.
fn varied(count: usize, seed: usize) -> String {
    (0..count)
        .map(|i| char::from(b'a' + ((i * 7 + seed * 3 + i / 16 * 5) % 26) as u8))
        .collect()
}

fn adapter(
    backend: &ReferenceKvReuseBackend,
    identity: &str,
    generation: u64,
    kind: Qwen2AdapterKind,
    first_layer: usize,
) -> Qwen2Adapter {
    Qwen2Adapter::seeded(
        backend.stack_config(),
        &Qwen2AdapterSpec::pinned(identity, generation, kind, first_layer),
        SEED,
    )
    .unwrap()
}

fn lora(backend: &ReferenceKvReuseBackend, generation: u64) -> Qwen2Adapter {
    adapter(backend, "lora-a", generation, Qwen2AdapterKind::Qkv, 2)
}

fn dead(identity: &str, generation: u64) -> BTreeSet<(String, Generation)> {
    [(identity.to_owned(), Generation(generation))]
        .into_iter()
        .collect()
}

fn run(session: &mut KvSession<'_>, oracle: &dyn KvLivenessOracle) -> ReferenceKvOutcome {
    match session.run_to_end(oracle).unwrap() {
        KvSessionFinish::Completed(outcome) => *outcome,
        KvSessionFinish::Cancelled(cancelled) => panic!("cancelled: {cancelled:?}"),
    }
}

fn same_result(a: &ReferenceKvOutcome, b: &ReferenceKvOutcome) {
    assert_eq!(a.generated, b.generated);
    assert_eq!(a.logits_digest, b.logits_digest);
    assert_eq!(a.run.events, b.run.events);
}

fn allow() -> KvReuseHint {
    KvReuseHint {
        requested_prefix_tokens: 0,
        policy: KvReusePolicy::AllowApproximate,
    }
}

fn prefill_chunks(session: &mut KvSession<'_>, oracle: &dyn KvLivenessOracle, chunks: usize) {
    for _ in 0..chunks {
        assert!(matches!(
            session.prefill_chunk(oracle).unwrap(),
            KvSessionStep::Prefilled { .. }
        ));
    }
}

fn expert_identity(project: &ProjectId, pod: &PodId) -> String {
    KvDependency::expert(project, pod, Generation(1), [0; 32]).identity
}

/// The pod of expert `(layer, index)`: every expert has its own lifecycle identity.
fn pod_of(pod: &PodId, layer: usize, index: usize) -> PodId {
    PodId::from(format!("{pod}-{layer}-{index}").as_str())
}

// ---- base to LoRA sharing -------------------------------------------------------------------

#[test]
fn base_to_lora_shares_layers_below_the_first_adapted_layer_and_matches_a_full_adapter_prefill() {
    let backend = backend();
    let live = backend.own_liveness();
    let body = varied(PROMPT, 3);
    let req = request(&backend, "p", &body);
    let adapter = lora(&backend, 1);
    let binding = StackBinding::with_adapter(adapter);

    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_base = reference.infer_cold(&scope(), &req).unwrap();
    let cold_lora = reference
        .infer_cold_bound(&scope(), &req, None, &binding)
        .unwrap();
    // The adapter is not a no-op.
    assert_ne!(cold_base.logits_digest, cold_lora.logits_digest);

    // The base request runs layered and bitwise equals the default binding.
    let mut base = backend
        .begin_session_bound(&scope(), &req, None, &StackBinding::base_layered(), &live)
        .unwrap();
    let base_out = run(&mut base, &live);
    same_result(&base_out, &cold_base);
    assert_eq!(base_out.inserted_chunks, 4);
    assert_eq!(
        base_out.layered.as_ref().unwrap().inserted_layer_entries,
        16
    );

    // The LoRA request (f = 2) shares layers 0 and 1 and computes layers 2 and 3.
    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    let out = run(&mut sess, &live);
    same_result(&out, &cold_lora);
    let layered = out.layered.clone().unwrap();
    assert_eq!(layered.assembly, LayerAssembly::ComputeThenAdopt);
    assert_eq!(layered.layer_count, LAYERS);
    assert_eq!(layered.shared_layers, 2);
    assert_eq!(layered.computed_layers, 2);
    assert_eq!(layered.shared_rows, 2 * 64);
    assert_eq!(layered.total_rows, LAYERS * PROMPT);
    assert_eq!(layered.computed_rows, LAYERS * PROMPT - 2 * 64);
    assert!(layered.adopted_compared_rows >= 2 * 64);
    assert!(layered.adopted_swapped_layer_pages >= 2 * 4);
    assert_eq!(layered.adoption_mismatches, 0);
    // Full prefill ran (compute-then-adopt), nothing was skipped.
    assert_eq!(out.hit_tokens, 0);
    assert_eq!(out.prefilled_tokens, PROMPT);
    assert_eq!(out.binding.adapter, Some(("lora-a".to_owned(), 1)));
    // Physical sharing: layers < f share the base pages, layers >= f do not.
    for page in 0..4 {
        for layer in 0..LAYERS {
            let shared = sess.shared_leading_rows(&base, page, layer);
            assert_eq!(shared, if layer < 2 { 16 } else { 0 }, "{page}/{layer}");
        }
    }
    assert_eq!(backend.session_stats().unwrap().adoption_mismatches, 0);
    drop(base);
    drop(sess);

    // A second LoRA request now finds every layer live and forks the assembled prefix.
    let second = backend
        .infer_scoped_bound(&scope(), &req, None, &binding)
        .unwrap();
    same_result(&second, &cold_lora);
    let layered = second.layered.clone().unwrap();
    assert_eq!(layered.assembly, LayerAssembly::Fork);
    assert_eq!(second.hit_tokens, 64);
    assert_eq!(second.prefilled_tokens, PROMPT - 64);
    assert_eq!(
        (layered.shared_layers, layered.shared_rows),
        (LAYERS, LAYERS * 64)
    );
}

#[test]
fn qonly_and_mlp_only_share_one_more_layer_than_qkv() {
    for (kind, shared) in [
        (Qwen2AdapterKind::Qkv, 2),
        (Qwen2AdapterKind::QOnly, 3),
        (Qwen2AdapterKind::MlpOnly, 3),
    ] {
        let backend = backend();
        let live = backend.own_liveness();
        let req = request(&backend, "p", &varied(PROMPT, 5));
        let binding = StackBinding::with_adapter(adapter(&backend, "lora-b", 1, kind, 2));
        let mut base = backend
            .begin_session_bound(&scope(), &req, None, &StackBinding::base_layered(), &live)
            .unwrap();
        run(&mut base, &live);
        let out = backend
            .infer_scoped_bound(&scope(), &req, None, &binding)
            .unwrap();
        let cold = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope())
            .unwrap()
            .infer_cold_bound(&scope(), &req, None, &binding)
            .unwrap();
        same_result(&out, &cold);
        let layered = out.layered.unwrap();
        assert_eq!(layered.shared_layers, shared, "{kind:?}");
        assert_eq!(layered.computed_layers, LAYERS - shared, "{kind:?}");
        assert_eq!(layered.adoption_mismatches, 0, "{kind:?}");
    }
}

// ---- aLoRA ------------------------------------------------------------------------------------

#[test]
fn alora_shares_the_boundary_page_rows_before_the_invocation_and_never_files_post_rows_under_a_base_key(
) {
    let backend = backend();
    let live = backend.own_liveness();
    let req = request(&backend, "p", &varied(PROMPT, 9));
    let alora = adapter(
        &backend,
        "alora-a",
        1,
        Qwen2AdapterKind::Activated { invocation: 39 },
        0,
    );
    let binding = StackBinding::with_adapter(alora);
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_alora = reference
        .infer_cold_bound(&scope(), &req, None, &binding)
        .unwrap();
    let cold_base = reference.infer_cold(&scope(), &req).unwrap();
    assert_ne!(cold_alora.logits_digest, cold_base.logits_digest);

    let mut base = backend
        .begin_session_bound(&scope(), &req, None, &StackBinding::base_layered(), &live)
        .unwrap();
    run(&mut base, &live);

    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    // The base prefix is forked and truncated at the invocation.
    assert_eq!(sess.hit_tokens(), 39);
    let out = run(&mut sess, &live);
    same_result(&out, &cold_alora);
    let layered = out.layered.clone().unwrap();
    assert_eq!(layered.assembly, LayerAssembly::Fork);
    assert_eq!(layered.boundary_rows, 39 % 16);
    assert_eq!(
        (layered.shared_layers, layered.shared_rows),
        (LAYERS, LAYERS * 39)
    );
    assert_eq!(layered.post_invocation_rows_under_base_key, 0);
    assert_eq!(
        backend
            .session_stats()
            .unwrap()
            .post_invocation_rows_under_base_key,
        0
    );
    for layer in 0..LAYERS {
        assert_eq!(sess.shared_leading_rows(&base, 0, layer), 16);
        assert_eq!(sess.shared_leading_rows(&base, 1, layer), 16);
        // The boundary page keeps exactly the seven pre-invocation rows.
        assert_eq!(
            sess.shared_leading_rows(&base, 2, layer),
            7,
            "layer {layer}"
        );
        assert_eq!(
            sess.shared_leading_rows(&base, 3, layer),
            0,
            "layer {layer}"
        );
    }
    // Chunks 48 and 64 are post-invocation and went in under the adapted chain.
    assert_eq!(layered.inserted_layer_entries, 2 * LAYERS);
    drop(base);
    drop(sess);

    // A base request is served at most the pre-invocation boundary of its own earlier
    // entries (its entries are untouched), and the aLoRA entries never reach it: its output
    // equals a cold base run.
    let again = backend
        .infer_scoped_bound(&scope(), &req, None, &StackBinding::base_layered())
        .unwrap();
    same_result(&again, &cold_base);
    assert_eq!(again.hit_tokens, 64);
}

#[test]
fn a_cold_alora_run_files_pre_invocation_chunks_under_the_base_key_and_a_base_request_reads_them() {
    let backend = backend();
    let req = request(&backend, "p", &varied(PROMPT, 11));
    let binding = StackBinding::with_adapter(adapter(
        &backend,
        "alora-b",
        1,
        Qwen2AdapterKind::Activated { invocation: 39 },
        0,
    ));
    let out = backend
        .infer_scoped_bound(&scope(), &req, None, &binding)
        .unwrap();
    let layered = out.layered.unwrap();
    // Chunks 16 and 32 end before the invocation (base-keyed, no adapter dependency), 48 and
    // 64 contain post-invocation rows (adapted-keyed).
    assert_eq!(layered.inserted_layer_entries, 4 * LAYERS);
    assert_eq!(layered.post_invocation_rows_under_base_key, 0);

    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_base = reference.infer_cold(&scope(), &req).unwrap();
    let base = backend
        .infer_scoped_bound(&scope(), &req, None, &StackBinding::base_layered())
        .unwrap();
    // Only the two pre-invocation chunks exist under the base chain.
    assert_eq!(base.hit_tokens, 32);
    same_result(&base, &cold_base);
}

// ---- revocation and supersession --------------------------------------------------------------

#[test]
fn adapter_revocation_cancels_and_restart_re_resolves_to_the_base_model() {
    let backend = backend();
    let live = backend.own_liveness();
    let req = request(&backend, "r", &varied(PROMPT, 13));
    let binding = StackBinding::with_adapter(lora(&backend, 1));
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_base = reference.infer_cold(&scope(), &req).unwrap();

    // A completed LoRA run fills per-layer entries: layers 0 and 1 base-keyed, 2 and 3 adapted.
    backend
        .infer_scoped_bound(&scope(), &req, None, &binding)
        .unwrap();
    assert_eq!(backend.catalog_stats().unwrap().entries, 4 * LAYERS);

    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    // The prefix (64 tokens) was forked from the per-layer entries; one chunk is left.
    assert_eq!(sess.hit_tokens(), 64);
    prefill_chunks(&mut sess, &live, 1);
    // Revoking the adapter tombstones only the entries that depend on it (layers >= 2).
    backend.invalidate_inputs(&dead("lora-a", 1)).unwrap();
    let step = sess.decode_step(&live).unwrap();
    let KvSessionStep::Cancelled(cancelled) = step else {
        panic!("expected cancellation, got {step:?}")
    };
    assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
    assert_eq!(cancelled.tokens_emitted, 0);
    assert_eq!(sess.phase(), KvSessionPhase::Cancelled);
    // The cancelled fork released its pins, so the dead entries are gone.
    assert_eq!(backend.catalog_stats().unwrap().entries, 4 * 2);
    let restarted = sess.restart_with(&live).unwrap();
    same_result(&restarted, &cold_base);
    assert!(restarted.restarted_after_cancel);
    assert_eq!(restarted.binding.adapter, None);
    assert!(restarted.binding.re_resolved && restarted.binding.adapter_revoked_to_base);
    assert!(backend
        .session_stats()
        .unwrap()
        .restarted_requests
        .contains(&RequestId::from("r")));

    // The revoked adapter is refused at begin, and the surviving lower layers still serve a
    // base request (partial hit, adopted).
    assert!(matches!(
        backend.begin_session_bound(&scope(), &req, None, &binding, &live),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
    let base = backend
        .infer_scoped_bound(&scope(), &req, None, &StackBinding::base_layered())
        .unwrap();
    same_result(&base, &cold_base);
    let layered = base.layered.unwrap();
    assert_eq!(layered.assembly, LayerAssembly::ComputeThenAdopt);
    assert_eq!(layered.shared_layers, 2);
}

#[test]
fn adapter_supersession_restarts_on_the_new_generation() {
    let backend = backend();
    let live = backend.own_liveness();
    let req = request(&backend, "s", &varied(PROMPT, 17));
    let gen1 = lora(&backend, 1);
    let gen2 = lora(&backend, 2);
    let binding = StackBinding::with_adapter(gen1).with_successors(BindingSuccessors {
        adapters: vec![gen2.clone()],
        experts: Vec::new(),
    });
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_gen2 = reference
        .infer_cold_bound(&scope(), &req, None, &StackBinding::with_adapter(gen2))
        .unwrap();
    let cold_gen1 = reference
        .infer_cold_bound(&scope(), &req, None, &binding)
        .unwrap();
    assert_ne!(cold_gen1.logits_digest, cold_gen2.logits_digest);

    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    prefill_chunks(&mut sess, &live, 1);
    let live_floor: BTreeMap<String, Generation> =
        [("lora-a".to_owned(), Generation(2))].into_iter().collect();
    backend.supersede_inputs(&live_floor).unwrap();
    assert!(matches!(
        sess.prefill_chunk(&live).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    let restarted = sess.restart_with(&live).unwrap();
    same_result(&restarted, &cold_gen2);
    assert_eq!(restarted.binding.adapter, Some(("lora-a".to_owned(), 2)));
    assert_eq!(restarted.binding.adapter_superseded, Some((1, 2)));
    assert!(!restarted.binding.adapter_revoked_to_base);
}

// ---- MoE experts ------------------------------------------------------------------------------

fn moe_binding(
    backend: &ReferenceKvReuseBackend,
    project: &ProjectId,
    pod: &PodId,
    _index: usize,
    mode: Qwen2RoutingMode,
) -> StackBinding {
    let spec = Qwen2MoeSpec {
        mode,
        ..Qwen2MoeSpec::pinned(7, LAYERS)
    };
    let moe = Qwen2Moe::seeded(backend.stack_config(), &spec).unwrap();
    StackBinding::with_moe(MoeBinding {
        experts: all_experts(&moe, project, pod),
        moe,
    })
}

/// Every live expert of every MoE layer, each under its own pod (`pod_of`).
fn all_experts(moe: &Qwen2Moe, project: &ProjectId, pod: &PodId) -> Vec<BoundExpert> {
    let mut experts = Vec::new();
    for layer in 0..LAYERS {
        let Some(moe_layer) = moe.layer(layer) else {
            continue;
        };
        for index in 0..moe_layer.experts.len() {
            experts.push(BoundExpert {
                layer,
                index,
                project: project.clone(),
                pod: pod_of(pod, layer, index),
                fence_epoch: 0,
            });
        }
    }
    experts
}

/// The first expert of layer 1 whose masking changes the output on `req` (a routed one).
fn routed_expert(
    backend: &ReferenceKvReuseBackend,
    req: &SegmentedModelRequest,
    project: &ProjectId,
    pod: &PodId,
) -> usize {
    let plain = moe_binding(backend, project, pod, 0, Qwen2RoutingMode::Dropless);
    let digest = backend
        .infer_cold_bound(&scope(), req, None, &plain)
        .unwrap()
        .logits_digest;
    (0..4)
        .find(|index| {
            let masked = masked_binding(backend, project, pod, *index, 1);
            backend
                .infer_cold_bound(&scope(), req, None, &masked)
                .unwrap()
                .logits_digest
                != digest
        })
        .expect("some expert of layer 1 is routed to on the probe prompt")
}

fn masked_binding(
    backend: &ReferenceKvReuseBackend,
    project: &ProjectId,
    pod: &PodId,
    index: usize,
    layer: usize,
) -> StackBinding {
    let StackBinding {
        moe: Some(bound), ..
    } = moe_binding(backend, project, pod, index, Qwen2RoutingMode::Dropless)
    else {
        unreachable!()
    };
    StackBinding::with_moe(MoeBinding {
        moe: bound.moe.set_expert_live(layer, index, false).unwrap(),
        experts: bound.experts,
    })
}

#[test]
fn expert_revocation_kills_only_entries_above_its_layer_and_restart_masks_it() {
    let backend = backend();
    let live = backend.own_liveness();
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    let req = request(&backend, "e", &varied(PROMPT, 19));
    let index = routed_expert(&backend, &req, &project, &pod);
    let binding = moe_binding(&backend, &project, &pod, index, Qwen2RoutingMode::Dropless);
    let masked = masked_binding(&backend, &project, &pod, index, 1);
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_masked = reference
        .infer_cold_bound(&scope(), &req, None, &masked)
        .unwrap();
    let cold_full = reference
        .infer_cold_bound(&scope(), &req, None, &binding)
        .unwrap();
    // The expert is routed to, so masking it changes the output (a vacuous run is refused).
    assert_ne!(cold_masked.logits_digest, cold_full.logits_digest);
    assert_eq!(
        cold_full.binding.routing,
        RoutingDeterminism::BatchInvariant
    );

    backend
        .infer_scoped_bound(&scope(), &req, None, &binding)
        .unwrap();
    assert_eq!(backend.catalog_stats().unwrap().entries, 4 * LAYERS);

    let identity = expert_identity(&project, &pod_of(&pod, 1, index));
    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    prefill_chunks(&mut sess, &live, 1);
    backend.invalidate_inputs(&dead(&identity, 1)).unwrap();
    assert!(matches!(
        sess.decode_step(&live).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    // Expert (1, i): layers 0 and 1 stay, layers 2 and 3 are dead.
    assert_eq!(backend.catalog_stats().unwrap().entries, 4 * 2);
    let restarted = sess.restart_with(&live).unwrap();
    same_result(&restarted, &cold_masked);
    assert_eq!(
        restarted.binding.experts_masked_by_resolution,
        vec![(1, index)]
    );
    assert_eq!(restarted.binding.masked_experts, vec![(1, index)]);

    // The surviving lower layers serve the masked binding (their rows do not depend on the
    // expert), and the result equals its cold run.
    let served = backend
        .infer_scoped_bound(&scope(), &req, None, &masked)
        .unwrap();
    same_result(&served, &cold_masked);
    let layered = served.layered.unwrap();
    assert_eq!(layered.assembly, LayerAssembly::ComputeThenAdopt);
    assert_eq!(layered.shared_layers, 2);
    assert_eq!(layered.adoption_mismatches, 0);
}

#[test]
fn a_superseded_expert_restarts_on_its_successor_generation() {
    let backend = backend();
    let live = backend.own_liveness();
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    let req = request(&backend, "g", &varied(PROMPT, 23));
    let index = routed_expert(&backend, &req, &project, &pod);
    let StackBinding {
        moe: Some(bound), ..
    } = moe_binding(&backend, &project, &pod, index, Qwen2RoutingMode::Dropless)
    else {
        unreachable!()
    };
    let successor = StackBinding::with_moe(MoeBinding {
        moe: bound
            .moe
            .clone()
            .with_expert_generation(backend.stack_config(), 1, index, 2)
            .unwrap(),
        experts: bound.experts.clone(),
    });
    let binding = StackBinding::with_moe(bound).with_successors(BindingSuccessors {
        adapters: Vec::new(),
        experts: vec![ExpertSuccessor {
            layer: 1,
            index,
            generation: 2,
            fence_epoch: 0,
        }],
    });
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_successor = reference
        .infer_cold_bound(&scope(), &req, None, &successor)
        .unwrap();

    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    prefill_chunks(&mut sess, &live, 1);
    let floor: BTreeMap<String, Generation> = [(
        expert_identity(&project, &pod_of(&pod, 1, index)),
        Generation(2),
    )]
    .into_iter()
    .collect();
    backend.supersede_inputs(&floor).unwrap();
    assert!(matches!(
        sess.prefill_chunk(&live).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    let restarted = sess.restart_with(&live).unwrap();
    same_result(&restarted, &cold_successor);
    assert_eq!(restarted.binding.experts_superseded, vec![(1, index, 1, 2)]);
    assert!(restarted.binding.experts_masked_by_resolution.is_empty());
}

// ---- batch-variant routing --------------------------------------------------------------------

#[test]
fn batch_variant_routing_is_approximate_and_never_touches_the_exact_catalog() {
    let backend = backend();
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    let req = request(&backend, "b", &varied(PROMPT, 29));
    let exact = moe_binding(&backend, &project, &pod, 0, Qwen2RoutingMode::Dropless);
    let variant = moe_binding(
        &backend,
        &project,
        &pod,
        0,
        Qwen2RoutingMode::BatchVariantCapacity { capacity: 1 },
    );
    // Populate the Exact catalog first.
    let filled = backend
        .infer_scoped_bound(&scope(), &req, None, &exact)
        .unwrap();
    assert_eq!(filled.run.kv_report.unwrap().mode(), KvReuseMode::Exact);
    let entries = backend.catalog_stats().unwrap().entries;
    assert!(entries > 0);
    let stats = backend.catalog_stats().unwrap();

    // ExactOnly refuses it, the allow policy tags it Approximate.
    assert!(matches!(
        backend.infer_scoped_bound(&scope(), &req, None, &variant),
        Err(ReferenceKvError::Binding(
            BindingError::ApproximateNotAllowed
        ))
    ));
    let out = backend
        .infer_scoped_bound(&scope(), &req, Some(&allow()), &variant)
        .unwrap();
    let report = out.run.kv_report.unwrap();
    assert_eq!(report.mode(), KvReuseMode::Approximate);
    assert_eq!(report.hit_tokens(), 0);
    assert_eq!(
        (out.hit_tokens, out.inserted_chunks, out.insert_refusals),
        (0, 0, 0)
    );
    assert_eq!(out.prefilled_tokens, PROMPT);
    assert_eq!(out.binding.routing, RoutingDeterminism::BatchVariant);
    // No catalog movement at all: no hit, no insert, no lookup miss.
    assert_eq!(backend.catalog_stats().unwrap(), stats);
    let session_stats = backend.session_stats().unwrap();
    assert_eq!(session_stats.approximate_sessions, 1);
    assert!(session_stats.approximate_inserts_refused >= 4);

    // And the Exact binding afterwards still hits its own entries only.
    let again = backend
        .infer_scoped_bound(&scope(), &req, None, &exact)
        .unwrap();
    assert_eq!(again.hit_tokens, 64);
    same_result(&again, &filled);
}

// ---- refusals and unchanged defaults ----------------------------------------------------------

#[test]
fn malformed_bindings_are_refused_and_the_default_binding_is_the_unbound_path() {
    let backend = backend();
    let live = backend.own_liveness();
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    let req = request(&backend, "n", &varied(PROMPT, 31));

    let StackBinding {
        moe: Some(mut bound),
        ..
    } = moe_binding(&backend, &project, &pod, 0, Qwen2RoutingMode::Dropless)
    else {
        unreachable!()
    };
    let mut unknown = bound.clone();
    unknown.experts.push(BoundExpert {
        layer: 1,
        index: 9,
        project: project.clone(),
        pod: pod_of(&pod, 1, 9),
        fence_epoch: 0,
    });
    assert!(matches!(
        backend.begin_session_bound(
            &scope(),
            &req,
            None,
            &StackBinding::with_moe(unknown),
            &live
        ),
        Err(ReferenceKvError::Binding(BindingError::UnknownExpert {
            layer: 1,
            index: 9
        }))
    ));
    bound.experts.push(bound.experts[0].clone());
    assert!(matches!(
        backend.begin_session_bound(&scope(), &req, None, &StackBinding::with_moe(bound), &live),
        Err(ReferenceKvError::Binding(BindingError::DuplicateExpert {
            layer: 0,
            index: 0
        }))
    ));

    // A revoked tracked expert is refused at begin, a cold run of it is not (it depends on
    // no cached state).
    let revoked = moe_binding(&backend, &project, &pod, 0, Qwen2RoutingMode::Dropless);
    backend
        .invalidate_inputs(&dead(&expert_identity(&project, &pod_of(&pod, 1, 0)), 1))
        .unwrap();
    assert!(matches!(
        backend.begin_session_bound(&scope(), &req, None, &revoked, &live),
        Err(ReferenceKvError::Binding(BindingError::NotLive { .. }))
    ));
    assert!(backend
        .infer_cold_bound(&scope(), &req, None, &revoked)
        .is_ok());

    // The default binding is bitwise the unbound path and keeps whole-stack entries.
    let bound_before = backend.session_stats().unwrap().bound_sessions;
    let unbound = backend.infer_scoped(&scope(), &req, None).unwrap();
    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &StackBinding::base(), &live)
        .unwrap();
    let bound = run(&mut sess, &live);
    same_result(&bound, &unbound);
    assert_eq!(bound.hit_tokens, 64);
    assert!(bound.layered.is_none());
    assert_eq!(bound.binding, ptr_runtime::BindingRecord::default());
    assert_eq!(
        backend.session_stats().unwrap().bound_sessions,
        bound_before
    );
}

// ---- guarded restart, tracked experts, per-class guards ----------------------------------------

/// The backend's own liveness, except that dependencies of `kind` (and `generation`, when
/// given) answer live for their first `after` queries and dead from then on.
struct DieAfter {
    inner: ptr_runtime::ReferenceKvLiveness,
    kind: KvDependencyKind,
    generation: Option<Generation>,
    after: usize,
    seen: AtomicUsize,
}

impl DieAfter {
    fn new(
        backend: &ReferenceKvReuseBackend,
        kind: KvDependencyKind,
        generation: Option<u64>,
        after: usize,
    ) -> Self {
        Self {
            inner: backend.own_liveness(),
            kind,
            generation: generation.map(Generation),
            after,
            seen: AtomicUsize::new(0),
        }
    }
}

impl KvLivenessOracle for DieAfter {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        if !self.inner.is_live(dependency) {
            return false;
        }
        let matches = dependency.kind == self.kind
            && self
                .generation
                .is_none_or(|generation| generation == dependency.generation);
        !matches || self.seen.fetch_add(1, Ordering::SeqCst) < self.after
    }
}

#[test]
fn a_revocation_during_the_restart_cancels_it_and_re_resolves_to_the_base_model() {
    let backend = backend();
    let live = backend.own_liveness();
    let req = request(&backend, "m", &varied(PROMPT, 37));
    let gen1 = lora(&backend, 1);
    let gen2 = lora(&backend, 2);
    let binding = StackBinding::with_adapter(gen1).with_successors(BindingSuccessors {
        adapters: vec![gen2],
        experts: Vec::new(),
    });
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_base = reference.infer_cold(&scope(), &req).unwrap();

    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    prefill_chunks(&mut sess, &live, 1);
    let floor: BTreeMap<String, Generation> =
        [("lora-a".to_owned(), Generation(2))].into_iter().collect();
    backend.supersede_inputs(&floor).unwrap();
    assert!(matches!(
        sess.prefill_chunk(&live).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    // The successor (generation 2) is live for the resolution and the first two chunk
    // checks of the restart, and is revoked before its third chunk.
    let oracle = DieAfter::new(&backend, KvDependencyKind::Adapter, Some(2), 3);
    let restarted = sess.restart_with(&oracle).unwrap();
    // The cancelled attempt produced nothing that escaped: the outcome is the base model's.
    same_result(&restarted, &cold_base);
    assert!(restarted.restarted_after_cancel);
    assert_eq!(restarted.binding.adapter, None);
    assert!(restarted.binding.re_resolved && restarted.binding.adapter_revoked_to_base);
    let stats = backend.session_stats().unwrap();
    // The original cancellation and the cancelled first attempt of the restart.
    assert_eq!(stats.sessions_cancelled, 2);
    assert_eq!(stats.sessions_restarted, 1);
}

#[test]
fn a_restart_cancelled_on_every_attempt_surfaces_cancelled_and_emits_nothing() {
    let backend = backend();
    let live = backend.own_liveness();
    let req = request(&backend, "x", &varied(PROMPT, 41));
    let binding = StackBinding::with_adapter(lora(&backend, 1));
    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    prefill_chunks(&mut sess, &live, 1);
    backend.invalidate_inputs(&dead("lora-a", 1)).unwrap();
    assert!(matches!(
        sess.prefill_chunk(&live).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    // The base model (the resolution's fallback) answers live for two chunk checks and is
    // then fenced for good: every attempt is cancelled.
    let oracle = DieAfter::new(&backend, KvDependencyKind::BaseModel, None, 2);
    let error = sess.restart_with(&oracle).unwrap_err();
    let ReferenceKvError::RestartCancelled { attempts, last } = error else {
        panic!("expected RestartCancelled, got {error:?}")
    };
    assert_eq!(attempts, 3);
    assert_eq!(last.reason, KvCancelReason::InputNotLive);
    assert_eq!((last.tokens_emitted, last.prefilled_tokens), (0, 0));
    assert_eq!(sess.phase(), KvSessionPhase::Cancelled);
    assert!(sess.generated().is_empty());
    let stats = backend.session_stats().unwrap();
    assert_eq!(stats.sessions_completed, 0);
    assert_eq!(stats.sessions_cancelled, 1 + 3);
    // The session never produced an outcome; a bound session has no oracle-less restart.
    assert!(matches!(
        sess.restart(),
        Err(ReferenceKvError::Binding(BindingError::RestartNeedsOracle))
    ));
    assert_eq!(sess.phase(), KvSessionPhase::Cancelled);
}

fn semantic_request(backend: &ReferenceKvReuseBackend, id: &str) -> SegmentedModelRequest {
    let head = PromptSegment::from_text(
        backend.tokenizer(),
        SegmentSource::Semantic {
            key: "doc:one".to_owned(),
            generation: Generation(1),
        },
        &varied(32, 5),
    )
    .unwrap();
    let tail = PromptSegment::raw_from_text(backend.tokenizer(), &varied(38, 6)).unwrap();
    SegmentedModelRequest::new(
        RequestId::from(id),
        SemanticContext {
            revision: Revision(1),
            entries: Vec::new(),
        },
        backend.tokenizer().identity().clone(),
        vec![head, tail],
    )
    .unwrap()
}

#[test]
fn a_default_binding_restart_is_unchanged_and_needs_no_oracle() {
    let backend = backend();
    let live = backend.own_liveness();
    let req = semantic_request(&backend, "d");
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold = reference.infer_cold(&scope(), &req).unwrap();
    let mut sess = backend.begin_session(&scope(), &req, None, &live).unwrap();
    prefill_chunks(&mut sess, &live, 1);
    // A revoked input cancels a default session; its restart is the plain cold run, which
    // does not re-check the revoked segment.
    backend.invalidate_inputs(&dead("doc:one", 1)).unwrap();
    assert!(matches!(
        sess.prefill_chunk(&live).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    let restarted = sess.restart().unwrap();
    same_result(&restarted, &cold);
    assert!(restarted.restarted_after_cancel && restarted.layered.is_none());
    assert_eq!(restarted.binding, ptr_runtime::BindingRecord::default());
}

#[test]
fn a_dead_segment_at_begin_does_not_disable_the_adapter_guard() {
    let backend = backend();
    let live = backend.own_liveness();
    let req = semantic_request(&backend, "q");
    // The segment is already dead when the session begins (the caller's policy question).
    backend.invalidate_inputs(&dead("doc:one", 1)).unwrap();
    let binding = StackBinding::with_adapter(lora(&backend, 1));
    let mut sess = backend
        .begin_session_bound(&scope(), &req, None, &binding, &live)
        .unwrap();
    prefill_chunks(&mut sess, &live, 1);
    // The adapter dies mid-session: it still cancels.
    backend.invalidate_inputs(&dead("lora-a", 1)).unwrap();
    let KvSessionStep::Cancelled(cancelled) = sess.prefill_chunk(&live).unwrap() else {
        panic!("the adapter guard must stay armed")
    };
    assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
}

#[test]
fn a_binding_that_omits_a_live_expert_is_refused_and_a_masked_one_needs_no_tracking() {
    let backend = backend();
    let live = backend.own_liveness();
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    let req = request(&backend, "u", &varied(PROMPT, 47));
    let StackBinding {
        moe: Some(bound), ..
    } = moe_binding(&backend, &project, &pod, 0, Qwen2RoutingMode::Dropless)
    else {
        unreachable!()
    };
    let omit = |layer: usize, index: usize| {
        let mut partial = bound.clone();
        partial
            .experts
            .retain(|tracked| (tracked.layer, tracked.index) != (layer, index));
        StackBinding::with_moe(partial)
    };
    for (layer, index) in [(0, 0), (2, 3), (LAYERS - 1, 1)] {
        let refused = omit(layer, index);
        assert!(matches!(
            backend.begin_session_bound(&scope(), &req, None, &refused, &live),
            Err(ReferenceKvError::Binding(BindingError::UntrackedLiveExpert { layer: l, index: i }))
                if (l, i) == (layer, index)
        ));
        // A cold run depends on the same binding and is refused too.
        assert!(matches!(
            backend.infer_cold_bound(&scope(), &req, None, &refused),
            Err(ReferenceKvError::Binding(
                BindingError::UntrackedLiveExpert { .. }
            ))
        ));
    }
    // Masking the omitted expert makes its absence harmless.
    let mut partial = bound.clone();
    partial
        .experts
        .retain(|tracked| (tracked.layer, tracked.index) != (2, 3));
    let masked = StackBinding::with_moe(MoeBinding {
        moe: partial.moe.set_expert_live(2, 3, false).unwrap(),
        experts: partial.experts,
    });
    assert!(backend
        .begin_session_bound(&scope(), &req, None, &masked, &live)
        .is_ok());
}

#[test]
fn revoking_any_live_expert_tombstones_the_layers_above_it_and_cancels_the_session() {
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    for layer in 0..LAYERS {
        for index in 0..4 {
            let backend = backend();
            let live = backend.own_liveness();
            let req = request(&backend, "k", &varied(PROMPT, 53));
            let binding = moe_binding(&backend, &project, &pod, 0, Qwen2RoutingMode::Dropless);
            backend
                .infer_scoped_bound(&scope(), &req, None, &binding)
                .unwrap();
            assert_eq!(backend.catalog_stats().unwrap().entries, 4 * LAYERS);
            let mut sess = backend
                .begin_session_bound(&scope(), &req, None, &binding, &live)
                .unwrap();
            prefill_chunks(&mut sess, &live, 1);
            let identity = expert_identity(&project, &pod_of(&pod, layer, index));
            backend.invalidate_inputs(&dead(&identity, 1)).unwrap();
            let KvSessionStep::Cancelled(cancelled) = sess.decode_step(&live).unwrap() else {
                panic!("expert ({layer}, {index}) did not cancel the session")
            };
            assert_eq!(cancelled.reason, KvCancelReason::InputNotLive);
            // Layers up to and including `layer` do not depend on its MLP.
            assert_eq!(
                backend.catalog_stats().unwrap().entries,
                4 * (layer + 1),
                "expert ({layer}, {index})"
            );
        }
    }
}

// ---- successor fence epochs and the backend's own fence hook ----------------------------------

/// An oracle with a materialised `(generation, epoch)` per expert identity, the way the
/// registry judges a dependency recorded at a non-zero fence epoch: the generation and the
/// epoch must both be exactly the current ones. Everything else is live.
#[derive(Default)]
struct EpochOracle {
    experts: std::sync::Mutex<BTreeMap<String, (u64, u64)>>,
}

impl EpochOracle {
    fn set(&self, identity: &str, generation: u64, epoch: u64) {
        self.experts
            .lock()
            .unwrap()
            .insert(identity.to_owned(), (generation, epoch));
    }
}

impl KvLivenessOracle for EpochOracle {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        if dependency.kind != KvDependencyKind::Expert {
            return true;
        }
        self.experts
            .lock()
            .unwrap()
            .get(&dependency.identity)
            .is_some_and(|(generation, epoch)| {
                dependency.generation.0 == *generation && dependency.fence_epoch == *epoch
            })
    }
}

/// `binding` with the tracked expert `(layer, index)` recorded at `fence_epoch`.
fn with_tracked_epoch(
    binding: StackBinding,
    layer: usize,
    index: usize,
    fence_epoch: u64,
) -> StackBinding {
    let StackBinding {
        moe: Some(mut bound),
        ..
    } = binding
    else {
        unreachable!()
    };
    for tracked in &mut bound.experts {
        if (tracked.layer, tracked.index) == (layer, index) {
            tracked.fence_epoch = fence_epoch;
        }
    }
    StackBinding::with_moe(bound)
}

struct EpochCase {
    backend: ReferenceKvReuseBackend,
    oracle: EpochOracle,
    req: SegmentedModelRequest,
    index: usize,
    identity: String,
    /// Tracked at generation 1, epoch 3, with `successor` as the only candidate.
    binding: StackBinding,
    /// The same MoE on generation 2 tracked at epoch 5 (what a restart should run).
    cold_successor: ReferenceKvOutcome,
    successor: StackBinding,
}

fn epoch_case(successor_epoch: u64) -> EpochCase {
    let backend = backend();
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    let req = request(&backend, "ep", &varied(PROMPT, 31));
    let index = routed_expert(&backend, &req, &project, &pod);
    let recorded = with_tracked_epoch(
        moe_binding(&backend, &project, &pod, index, Qwen2RoutingMode::Dropless),
        1,
        index,
        3,
    );
    let StackBinding {
        moe: Some(bound), ..
    } = recorded.clone()
    else {
        unreachable!()
    };
    let successor = with_tracked_epoch(
        StackBinding::with_moe(MoeBinding {
            moe: bound
                .moe
                .clone()
                .with_expert_generation(backend.stack_config(), 1, index, 2)
                .unwrap(),
            experts: bound.experts.clone(),
        }),
        1,
        index,
        5,
    );
    let binding = recorded.with_successors(BindingSuccessors {
        adapters: Vec::new(),
        experts: vec![ExpertSuccessor {
            layer: 1,
            index,
            generation: 2,
            fence_epoch: successor_epoch,
        }],
    });
    let reference = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
    let cold_successor = reference
        .infer_cold_bound(&scope(), &req, None, &successor)
        .unwrap();
    let identity = expert_identity(&project, &pod_of(&pod, 1, index));
    let oracle = EpochOracle::default();
    // Every other expert is serving at its recorded epoch (0), this one at (1, 3).
    for layer in 0..LAYERS {
        for other in 0..4 {
            oracle.set(
                &expert_identity(&project, &pod_of(&pod, layer, other)),
                1,
                0,
            );
        }
    }
    oracle.set(&identity, 1, 3);
    EpochCase {
        backend,
        oracle,
        req,
        index,
        identity,
        binding,
        cold_successor,
        successor,
    }
}

#[test]
fn a_fenced_then_readmitted_expert_restarts_on_its_successor_at_the_successor_epoch() {
    let case = epoch_case(5);
    let mut sess = case
        .backend
        .begin_session_bound(&scope(), &case.req, None, &case.binding, &case.oracle)
        .unwrap();
    prefill_chunks(&mut sess, &case.oracle, 1);
    // The expert is fenced (epoch 4) and readmitted as generation 2 (epoch 5).
    case.oracle.set(&case.identity, 2, 5);
    assert!(matches!(
        sess.prefill_chunk(&case.oracle).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    let restarted = sess.restart_with(&case.oracle).unwrap();
    same_result(&restarted, &case.cold_successor);
    assert_eq!(
        restarted.binding.experts_superseded,
        vec![(1, case.index, 1, 2)]
    );
    assert!(restarted.binding.experts_masked_by_resolution.is_empty());
    assert!(restarted.binding.masked_experts.is_empty());

    // The restart tracked the successor at its own epoch: its guards named epoch 5, so the
    // run completed under the oracle (an inherited epoch 3 would have cancelled it), and
    // the successor binding (tracked at epoch 5) is accepted at begin.
    case.backend
        .begin_session_bound(&scope(), &case.req, None, &case.successor, &case.oracle)
        .unwrap();
}

#[test]
fn a_successor_recorded_at_a_stale_epoch_is_not_live_and_the_expert_is_masked() {
    // The old behaviour: the successor inherited the superseded binding's epoch (3), which
    // the registry no longer accepts once the expert sits at epoch 5.
    for stale in [0, 3, 4] {
        let case = epoch_case(stale);
        let mut sess = case
            .backend
            .begin_session_bound(&scope(), &case.req, None, &case.binding, &case.oracle)
            .unwrap();
        prefill_chunks(&mut sess, &case.oracle, 1);
        case.oracle.set(&case.identity, 2, 5);
        assert!(matches!(
            sess.prefill_chunk(&case.oracle).unwrap(),
            KvSessionStep::Cancelled(_)
        ));
        let restarted = sess.restart_with(&case.oracle).unwrap();
        assert_eq!(
            restarted.binding.experts_masked_by_resolution,
            vec![(1, case.index)],
            "epoch {stale}"
        );
        assert!(restarted.binding.experts_superseded.is_empty());
    }
}

#[test]
fn the_backend_applies_a_committed_fence_through_the_same_hook_as_the_runtime() {
    use ptr_ledger::{ExpertFenceAction, ExpertFenceCause, ExpertFenceResume, LedgerEvent};
    let backend = backend();
    let project = ProjectId::from("p");
    let pod = PodId::from("moe-pod");
    let req = request(&backend, "fh", &varied(PROMPT, 37));
    let index = routed_expert(&backend, &req, &project, &pod);
    let binding = moe_binding(&backend, &project, &pod, index, Qwen2RoutingMode::Dropless);
    backend
        .infer_scoped_bound(&scope(), &req, None, &binding)
        .unwrap();
    assert_eq!(backend.catalog_stats().unwrap().entries, 4 * LAYERS);

    let fence = |pod: PodId, epoch: u64| LedgerEvent::ExpertFenceCommitted {
        project: project.clone(),
        pod,
        action: ExpertFenceAction::Fence,
        cause: ExpertFenceCause::AuthorityQuarantine,
        resume: ExpertFenceResume::Serving,
        generation: Generation(1),
        expected_state_version: 1,
        epoch,
        revocation: None,
        evidence: None,
    };
    // Negative: an unrelated pod, an event that names nothing the catalog holds, and a
    // semantic delta (which carries no delta here) change nothing.
    assert_eq!(
        backend
            .apply_committed_event(&fence(PodId::from("other"), 2))
            .unwrap(),
        0
    );
    assert_eq!(
        backend
            .apply_committed_event(&LedgerEvent::Revoked {
                subject: "doc:unrelated".to_owned(),
                generation: Generation(1)
            })
            .unwrap(),
        0
    );
    assert_eq!(backend.catalog_stats().unwrap().entries, 4 * LAYERS);

    // The fence of expert (1, index): its dependants (layers 2 and 3) are tombstoned and
    // their backend prefixes retired; layers 0 and 1 do not depend on it.
    let retired = backend
        .apply_committed_event(&fence(pod_of(&pod, 1, index), 2))
        .unwrap();
    assert_eq!(retired, 4 * 2);
    assert_eq!(backend.catalog_stats().unwrap().entries, 4 * 2);

    // The backend's own liveness follows: a dependency recorded below the fence epoch is
    // not live, one at or above it is, and other experts are unaffected.
    let live = backend.own_liveness();
    let identity = expert_identity(&project, &pod_of(&pod, 1, index));
    let at = |epoch: u64| {
        KvDependency::expert_fenced(
            &project,
            &pod_of(&pod, 1, index),
            Generation(1),
            [0; 32],
            epoch,
        )
    };
    assert_eq!(at(0).identity, identity);
    assert!(!live.is_live(&at(0)));
    assert!(!live.is_live(&at(1)));
    assert!(live.is_live(&at(2)));
    assert!(live.is_live(&KvDependency::expert(
        &project,
        &pod_of(&pod, 1, (index + 1) % 4),
        Generation(1),
        [0; 32]
    )));
    // Replaying the same event retires nothing more.
    assert_eq!(
        backend
            .apply_committed_event(&fence(pod_of(&pod, 1, index), 2))
            .unwrap(),
        0
    );
}
