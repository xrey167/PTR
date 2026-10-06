//! Tiny MoE MLP on the reference stack: routing rule, batch invariance, the batch-variant
//! negative control, masking and chain placement. Public API only.

use ptr_pods::{
    derive_kv_layer_chains, InMemoryPagedKvBackend, KvChainIdentity, KvKeyEncoding, KvLayerChains,
    KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType, KvTensorSchema,
    PagedKvTensorBackend, Qwen2Activation, Qwen2Binding, Qwen2LayerConfig, Qwen2Moe, Qwen2MoeError,
    Qwen2MoeSpec, Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeScaling, Qwen2RoutingMode,
    Qwen2StackConfig, Qwen2StackError, Qwen2StackInput, RoutingCounts, RoutingDeterminism,
};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

const SEED: u64 = 20240607;
const MOE_SEED: u64 = 4242;
const CAPACITY: usize = 64;
const LAYERS: usize = 4;

fn config() -> Qwen2StackConfig {
    Qwen2StackConfig {
        layer: Qwen2LayerConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: CAPACITY,
            rope_theta: 10_000.0,
            rope_scaling: Qwen2RopeScaling::None,
            rms_norm_eps: 1e-6,
            hidden_act: Qwen2Activation::Silu,
            sliding_window: None,
        },
        num_layers: LAYERS,
        vocab_size: 256,
        key_encoding: KvKeyEncoding::PreRope,
        qk_norm: Qwen2QkNorm::Disabled,
    }
}

fn base_stack() -> Qwen2PagedStack {
    Qwen2PagedStack::reference(config(), SEED).unwrap()
}

fn moe(mode: Qwen2RoutingMode) -> Qwen2Moe {
    Qwen2Moe::seeded(
        &config(),
        &Qwen2MoeSpec {
            mode,
            ..Qwen2MoeSpec::pinned(MOE_SEED, LAYERS)
        },
    )
    .unwrap()
}

fn moe_stack(moe: Qwen2Moe) -> Qwen2PagedStack {
    base_stack().with_moe(moe).unwrap()
}

fn dropless() -> Qwen2PagedStack {
    moe_stack(moe(Qwen2RoutingMode::Dropless))
}

fn identity() -> KvChainIdentity {
    KvChainIdentity {
        base_model: "r005-tiny".to_owned(),
        generation: 1,
        build: "test-build".to_owned(),
        tokenizer: "ptr.byte@1".to_owned(),
    }
}

fn backend() -> InMemoryPagedKvBackend {
    InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap())
}

fn new_cache(
    backend: &InMemoryPagedKvBackend,
    stack: &Qwen2PagedStack,
) -> <InMemoryPagedKvBackend as KvTensorBackend>::Cache {
    let schema: KvTensorSchema = stack
        .kv_schema(
            ModelVersion::from("r005-tiny"),
            AdapterVersion::from("base"),
            DeviceId::from("cpu"),
        )
        .unwrap();
    let binding = KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [9; 32],
        principal: PrincipalId::from("alice"),
        device: schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::new(4).unwrap(),
    };
    backend.allocate_bound(schema, CAPACITY, binding).unwrap()
}

fn tokens(count: usize, salt: u32) -> Vec<u32> {
    (0..count as u32).map(|i| (i * 37 + salt) % 256).collect()
}

fn input(tokens: &[u32]) -> Qwen2StackInput {
    Qwen2StackInput {
        tokens: tokens.to_vec(),
    }
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

fn solo(stack: &Qwen2PagedStack, sequence: &[u32]) -> Vec<Vec<u32>> {
    traced(stack, sequence).0
}

fn traced(stack: &Qwen2PagedStack, sequence: &[u32]) -> (Vec<Vec<u32>>, RoutingCounts) {
    let backend = backend();
    let mut cache = new_cache(&backend, stack);
    let regime = stack.admit_rope_regime(sequence.len(), 0).unwrap();
    let (output, counts) = stack
        .prefill_traced_in_regime(
            &backend,
            &mut cache,
            input(sequence),
            &regime,
            &Qwen2Binding::default(),
        )
        .unwrap();
    (bits(&output.logits), counts)
}

/// Co-batch `sequences` in `order`; the logits come back per request index.
fn cobatch(
    stack: &Qwen2PagedStack,
    sequences: &[Vec<u32>],
    order: &[usize],
) -> (Vec<Vec<Vec<u32>>>, RoutingCounts) {
    let backend = backend();
    let mut caches: Vec<_> = sequences
        .iter()
        .map(|_| new_cache(&backend, stack))
        .collect();
    let longest = sequences.iter().map(Vec::len).max().unwrap();
    let regime = stack.admit_rope_regime(longest, 0).unwrap();
    let result = stack
        .prefill_cobatched_in_regime(
            &backend,
            &mut caches,
            sequences.iter().map(|s| input(s)).collect(),
            order,
            &regime,
            &Qwen2Binding::default(),
        )
        .unwrap();
    (
        result.outputs.iter().map(|o| bits(&o.logits)).collect(),
        result.routing,
    )
}

fn chains(stack: &Qwen2PagedStack) -> KvLayerChains {
    derive_kv_layer_chains(stack, &identity()).unwrap()
}

fn differing(a: &KvLayerChains, b: &KvLayerChains) -> Vec<usize> {
    (0..LAYERS).filter(|l| a.layer(*l) != b.layer(*l)).collect()
}

#[test]
fn routing_determinism_is_reported_by_the_stack() {
    assert_eq!(
        base_stack().routing_determinism(),
        RoutingDeterminism::NotRouted
    );
    assert_eq!(
        dropless().routing_determinism(),
        RoutingDeterminism::BatchInvariant
    );
    let capacity = moe_stack(moe(Qwen2RoutingMode::BatchVariantCapacity { capacity: 1 }));
    assert_eq!(
        capacity.routing_determinism(),
        RoutingDeterminism::BatchVariant
    );
}

#[test]
fn dense_weights_are_unchanged_by_the_moe_and_experts_use_their_own_domain() {
    let base = base_stack();
    let stack = dropless();
    assert_eq!(stack.weights(), base.weights());
    // The MoE seed domain does not depend on the dense seed or the other way round.
    let other = Qwen2Moe::seeded(&config(), &Qwen2MoeSpec::pinned(MOE_SEED + 1, LAYERS)).unwrap();
    assert_ne!(other, *stack.moe().unwrap());
    let moe = stack.moe().unwrap();
    for layer in 0..LAYERS {
        let layer = moe.layer(layer).unwrap();
        assert_eq!(layer.router.output_size, 4);
        assert_eq!(layer.router.input_size, 64);
        assert_eq!(layer.experts.len(), 4);
        for expert in &layer.experts {
            assert_eq!(expert.gate.output_size, 64);
            assert_eq!(expert.up.output_size, 64);
            assert_eq!((expert.down.output_size, expert.down.input_size), (64, 64));
            assert!(expert.live && expert.generation == 1);
        }
    }
    assert_eq!(moe.top_k(), 2);
    // The experts differ from each other.
    assert_ne!(
        moe.layer(0).unwrap().experts[0],
        moe.layer(0).unwrap().experts[1]
    );
}

#[test]
fn the_moe_changes_the_outputs_and_every_token_selects_exactly_top_k_experts() {
    let sequence = tokens(12, 5);
    let (moe_logits, counts) = traced(&dropless(), &sequence);
    let (dense_logits, dense_counts) = traced(&base_stack(), &sequence);
    assert_ne!(moe_logits, dense_logits);
    assert_eq!(dense_counts, RoutingCounts::default());
    for layer in 0..LAYERS {
        assert_eq!(
            counts.layer_selected(layer).iter().sum::<u64>(),
            2 * 12,
            "layer {layer}"
        );
        assert_eq!(counts.layer_selected(layer).len(), 4);
    }
    assert_eq!(counts.total_dropped(), 0);
    // The helper the experiment uses to pick the revoked expert: layer 1's most routed one.
    let most = counts.most_routed(1).unwrap();
    assert!(counts.selected(1, most) >= 1);
    assert!((0..4).all(|e| counts.selected(1, e) <= counts.selected(1, most)));
    assert_eq!(counts.most_routed(LAYERS), None);
}

#[test]
fn dropless_routing_is_independent_of_batch_order_and_of_co_batched_requests() {
    let stack = dropless();
    let a = tokens(10, 5);
    let b = tokens(10, 91);
    let c = tokens(7, 13);
    let sequences = [a.clone(), b.clone(), c.clone()];
    let alone: Vec<_> = sequences.iter().map(|s| solo(&stack, s)).collect();
    for order in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]] {
        let (batched, counts) = cobatch(&stack, &sequences, &order);
        assert_eq!(batched, alone, "order {order:?}");
        assert_eq!(counts.total_dropped(), 0);
    }
}

#[test]
fn batch_variant_capacity_diverges_with_co_batched_load_and_with_order() {
    let stack = moe_stack(moe(Qwen2RoutingMode::BatchVariantCapacity { capacity: 1 }));
    let a = tokens(10, 5);
    let b = tokens(10, 91);
    let sequences = [a.clone(), b.clone()];
    let alone: Vec<_> = sequences.iter().map(|s| solo(&stack, s)).collect();
    // A lone request never drops, and equals the dropless stack.
    let (_, solo_counts) = traced(&stack, &a);
    assert_eq!(solo_counts.total_dropped(), 0);
    assert_eq!(alone[0], solo(&dropless(), &a));

    let (forward, forward_counts) = cobatch(&stack, &sequences, &[0, 1]);
    let (reverse, reverse_counts) = cobatch(&stack, &sequences, &[1, 0]);
    // Drops depend on the load of the co-batched request...
    assert!(forward_counts.total_dropped() > 0);
    assert!(reverse_counts.total_dropped() > 0);
    assert_ne!(
        forward[1], alone[1],
        "the second request lost capacity to the first"
    );
    assert_ne!(reverse[0], alone[0], "and the other way round");
    // ...and the same request gets different outputs depending on who runs first.
    assert_ne!(forward, reverse);
    assert_ne!(forward[1], reverse[1]);
    // The first request of a step wins every contested slot, so it keeps its solo output.
    assert_eq!(forward[0], alone[0]);
    assert_eq!(reverse[1], alone[1]);
}

#[test]
fn cobatch_rejects_a_malformed_order_without_running_anything() {
    let stack = dropless();
    let backend = backend();
    let sequences = [tokens(4, 1), tokens(4, 2)];
    let regime = stack.admit_rope_regime(4, 0).unwrap();
    for order in [vec![0], vec![0, 0], vec![0, 2], vec![0, 1, 1]] {
        let mut caches: Vec<_> = sequences
            .iter()
            .map(|_| new_cache(&backend, &stack))
            .collect();
        let before = backend.snapshot(&caches[0]).unwrap();
        let error = stack
            .prefill_cobatched_in_regime(
                &backend,
                &mut caches,
                sequences.iter().map(|s| input(s)).collect(),
                &order,
                &regime,
                &Qwen2Binding::default(),
            )
            .unwrap_err();
        assert_eq!(error, Qwen2StackError::InvalidBatch, "{order:?}");
        assert_eq!(backend.snapshot(&caches[0]).unwrap(), before);
    }
}

#[test]
fn masking_changes_outputs_only_when_the_expert_would_have_been_routed() {
    let stack = dropless();
    // One token: exactly two of the four experts are routed at layer 1.
    let single = tokens(1, 5);
    let (reference, counts) = traced(&stack, &single);
    let unrouted = (0..4).find(|e| counts.selected(1, *e) == 0).unwrap();
    let routed = (0..4).find(|e| counts.selected(1, *e) == 1).unwrap();

    let masked = |expert: usize| {
        moe_stack(
            stack
                .moe()
                .unwrap()
                .clone()
                .set_expert_live(1, expert, false)
                .unwrap(),
        )
    };
    // Masking an expert the token was not routed to is bit-identical.
    assert_eq!(solo(&masked(unrouted), &single), reference);
    // Masking a routed one changes the output, and nothing is routed to it any more.
    let (changed, masked_counts) = traced(&masked(routed), &single);
    assert_ne!(changed, reference);
    assert_eq!(masked_counts.selected(1, routed), 0);
    assert_eq!(masked_counts.layer_selected(1).iter().sum::<u64>(), 2);

    // On a longer prompt: masking the most-routed expert of layer 1 changes the output.
    let sequence = tokens(12, 5);
    let (full, counts) = traced(&stack, &sequence);
    let most = counts.most_routed(1).unwrap();
    let (after, after_counts) = traced(&masked(most), &sequence);
    assert_ne!(after, full);
    assert_eq!(after_counts.selected(1, most), 0);
    // Unmasking restores the original output bit for bit.
    let restored = moe_stack(
        masked(most)
            .moe()
            .unwrap()
            .clone()
            .set_expert_live(1, most, true)
            .unwrap(),
    );
    assert_eq!(solo(&restored, &sequence), full);
}

#[test]
fn a_layer_without_a_live_expert_is_refused_without_touching_the_cache() {
    let mut moe = moe(Qwen2RoutingMode::Dropless);
    for expert in 0..4 {
        moe = moe.set_expert_live(1, expert, false).unwrap();
    }
    let stack = moe_stack(moe);
    let backend = backend();
    let mut cache = new_cache(&backend, &stack);
    let regime = stack.admit_rope_regime(3, 0).unwrap();
    let before = backend.snapshot(&cache).unwrap();
    let error = stack
        .prefill_bound_in_regime(
            &backend,
            &mut cache,
            input(&tokens(3, 5)),
            &regime,
            &Qwen2Binding::default(),
        )
        .unwrap_err();
    assert_eq!(
        error,
        Qwen2StackError::Moe(Qwen2MoeError::NoLiveExpert { layer: 1 })
    );
    assert_eq!(backend.snapshot(&cache).unwrap(), before);
}

#[test]
fn an_expert_of_layer_j_reaches_chain_l_for_l_above_j_only() {
    let stack = dropless();
    let reference = chains(&stack);
    let base = chains(&base_stack());
    // The MoE of layer 0 is bound into chain(1..), never chain(0).
    assert_eq!(differing(&base, &reference), vec![1, 2, 3]);
    assert_eq!(base.root(), reference.root());

    let moe = stack.moe().unwrap().clone();
    for j in 0..LAYERS {
        let above: Vec<usize> = (j + 1..LAYERS).collect();
        let variants = [
            // A new generation of the pod with new weights.
            moe.clone()
                .with_expert_generation(&config(), j, 2, 2)
                .unwrap(),
            // A generation bump without a weight change.
            moe.clone().with_expert_generation_only(j, 2, 2).unwrap(),
            // A revoked (masked) expert.
            moe.clone().set_expert_live(j, 2, false).unwrap(),
        ];
        for (index, variant) in variants.into_iter().enumerate() {
            let changed = chains(&moe_stack(variant));
            assert_eq!(
                differing(&reference, &changed),
                above,
                "expert (j={j}, variant {index})"
            );
        }
    }
    // Another expert of the same layer, another router seed and the routing mode differ too.
    let capacity = chains(&moe_stack(
        moe.clone()
            .with_mode(Qwen2RoutingMode::BatchVariantCapacity { capacity: 1 })
            .unwrap(),
    ));
    assert_eq!(differing(&reference, &capacity), vec![1, 2, 3]);
    let other_expert = chains(&moe_stack(
        moe.with_expert_generation_only(1, 0, 2).unwrap(),
    ));
    let this_expert = chains(&moe_stack(
        stack
            .moe()
            .unwrap()
            .clone()
            .with_expert_generation_only(1, 3, 2)
            .unwrap(),
    ));
    assert_ne!(other_expert.layer(2), this_expert.layer(2));
}

#[test]
fn a_generation_bump_alone_leaves_the_logits_but_not_the_chain() {
    let stack = dropless();
    let bumped = moe_stack(
        stack
            .moe()
            .unwrap()
            .clone()
            .with_expert_generation_only(1, 0, 2)
            .unwrap(),
    );
    let sequence = tokens(12, 5);
    assert_eq!(solo(&bumped, &sequence), solo(&stack, &sequence));
    assert_ne!(chains(&bumped), chains(&stack));
    // New weights for the new generation do change what a routed token computes.
    let (_, counts) = traced(&stack, &sequence);
    let routed = counts.most_routed(1).unwrap();
    let reseeded = moe_stack(
        stack
            .moe()
            .unwrap()
            .clone()
            .with_expert_generation(&config(), 1, routed, 2)
            .unwrap(),
    );
    assert_ne!(solo(&reseeded, &sequence), solo(&stack, &sequence));
}

#[test]
fn invalid_moe_specs_and_mismatched_stacks_are_refused() {
    let spec = Qwen2MoeSpec::pinned(MOE_SEED, LAYERS);
    let seeded = |spec: &Qwen2MoeSpec| Qwen2Moe::seeded(&config(), spec);
    assert_eq!(
        seeded(&Qwen2MoeSpec {
            moe_layers: vec![],
            ..spec.clone()
        })
        .unwrap_err(),
        Qwen2MoeError::NoLayers
    );
    assert_eq!(
        seeded(&Qwen2MoeSpec {
            moe_layers: vec![0, 0],
            ..spec.clone()
        })
        .unwrap_err(),
        Qwen2MoeError::DuplicateLayer(0)
    );
    assert!(matches!(
        seeded(&Qwen2MoeSpec {
            moe_layers: vec![LAYERS],
            ..spec.clone()
        })
        .unwrap_err(),
        Qwen2MoeError::LayerOutOfRange { .. }
    ));
    assert_eq!(
        seeded(&Qwen2MoeSpec {
            experts: 0,
            ..spec.clone()
        })
        .unwrap_err(),
        Qwen2MoeError::ZeroExperts
    );
    assert_eq!(
        seeded(&Qwen2MoeSpec {
            expert_intermediate: 0,
            ..spec.clone()
        })
        .unwrap_err(),
        Qwen2MoeError::ZeroExpertIntermediate
    );
    for top_k in [0, 5] {
        assert!(matches!(
            seeded(&Qwen2MoeSpec {
                top_k,
                ..spec.clone()
            })
            .unwrap_err(),
            Qwen2MoeError::InvalidTopK { .. }
        ));
    }
    assert_eq!(
        seeded(&Qwen2MoeSpec {
            mode: Qwen2RoutingMode::BatchVariantCapacity { capacity: 0 },
            ..spec.clone()
        })
        .unwrap_err(),
        Qwen2MoeError::ZeroCapacity
    );
    let moe = seeded(&spec).unwrap();
    assert_eq!(
        moe.clone().set_expert_live(0, 4, false).unwrap_err(),
        Qwen2MoeError::ExpertOutOfRange {
            layer: 0,
            expert: 4
        }
    );
    // A MoE shaped for another hidden size does not fit this stack.
    let mut small = config();
    small.layer.hidden_size = 32;
    small.layer.num_attention_heads = 2;
    small.layer.num_key_value_heads = 1;
    let foreign = Qwen2Moe::seeded(&small, &spec).unwrap();
    assert!(base_stack().with_moe(foreign).is_err());
    // A MoE over only some layers leaves the others dense.
    let partial = Qwen2Moe::seeded(
        &config(),
        &Qwen2MoeSpec {
            moe_layers: vec![2],
            ..spec
        },
    )
    .unwrap();
    assert!(partial.is_moe_layer(2) && !partial.is_moe_layer(1));
    let partial_stack = moe_stack(partial);
    let partial_chains = chains(&partial_stack);
    assert_eq!(differing(&partial_chains, &chains(&base_stack())), vec![3]);
    assert_eq!(
        partial_stack
            .moe()
            .unwrap()
            .clone()
            .set_expert_live(1, 0, false)
            .unwrap_err(),
        Qwen2MoeError::NotAMoeLayer(1)
    );
}
