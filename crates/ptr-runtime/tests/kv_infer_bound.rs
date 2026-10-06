//! Bound sessions of the reference KV-reuse backend (WP4): per-layer entries for adapters,
//! aLoRA and MoE experts, fork and compute-then-adopt assembly, Approximate routing and the
//! re-resolving restart. Public API only. Not an R009 or R010 result.

use ptr_model_api::{
    KvReuseHint, KvReuseMode, KvReusePolicy, PromptSegment, SegmentedModelRequest, Tokenizer,
};
use ptr_pods::{
    Qwen2Adapter, Qwen2AdapterKind, Qwen2AdapterSpec, Qwen2Moe, Qwen2MoeSpec, Qwen2RoutingMode,
    RoutingDeterminism,
};
use ptr_runtime::{
    BindingError, BindingSuccessors, BoundExpert, ExpertSuccessor, KvCancelReason, KvDependency,
    KvLivenessOracle, KvSession, KvSessionFinish, KvSessionPhase, KvSessionStep, LayerAssembly,
    MoeBinding, ReferenceKvError, ReferenceKvLimits, ReferenceKvOutcome, ReferenceKvReuseBackend,
    ReferenceKvScope, StackBinding,
};
use ptr_types::{Generation, PodId, PrincipalId, ProjectId, RequestId, Revision, SemanticContext};
use std::collections::{BTreeMap, BTreeSet};

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
    let restarted = sess.restart().unwrap();
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
    let restarted = sess.restart().unwrap();
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
    index: usize,
    mode: Qwen2RoutingMode,
) -> StackBinding {
    let spec = Qwen2MoeSpec {
        mode,
        ..Qwen2MoeSpec::pinned(7, LAYERS)
    };
    let moe = Qwen2Moe::seeded(backend.stack_config(), &spec).unwrap();
    StackBinding::with_moe(MoeBinding {
        moe,
        experts: vec![BoundExpert {
            layer: 1,
            index,
            project: project.clone(),
            pod: pod.clone(),
            fence_epoch: 0,
        }],
    })
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

    let identity = expert_identity(&project, &pod);
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
    let restarted = sess.restart().unwrap();
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
    let floor: BTreeMap<String, Generation> = [(expert_identity(&project, &pod), Generation(2))]
        .into_iter()
        .collect();
    backend.supersede_inputs(&floor).unwrap();
    assert!(matches!(
        sess.prefill_chunk(&live).unwrap(),
        KvSessionStep::Cancelled(_)
    ));
    let restarted = sess.restart().unwrap();
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
    unknown.experts[0].index = 9;
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
            layer: 1,
            index: 0
        }))
    ));

    // A revoked tracked expert is refused at begin, a cold run of it is not (it depends on
    // no cached state).
    let revoked = moe_binding(&backend, &project, &pod, 0, Qwen2RoutingMode::Dropless);
    backend
        .invalidate_inputs(&dead(&expert_identity(&project, &pod), 1))
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
