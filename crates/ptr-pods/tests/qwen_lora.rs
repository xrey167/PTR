//! Unmerged LoRA / aLoRA on the reference stack: chain placement, identity binding, logits
//! and the aLoRA pre-invocation rows. Public API only.

use ptr_pods::{
    derive_adapted_kv_layer_chains, derive_kv_layer_chains, InMemoryPagedKvBackend,
    KvChainIdentity, KvKeyEncoding, KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType,
    KvTensorSchema, PagedKvTensorBackend, Qwen2Activation, Qwen2Adapter, Qwen2AdapterError,
    Qwen2AdapterKind, Qwen2AdapterProjection, Qwen2AdapterSpec, Qwen2Binding, Qwen2LayerConfig,
    Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeScaling, Qwen2StackConfig, Qwen2StackError,
    Qwen2StackInput,
};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

const SEED: u64 = 20240607;
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

fn stack() -> Qwen2PagedStack {
    Qwen2PagedStack::reference(config(), SEED).unwrap()
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

fn tokens(count: usize) -> Vec<u32> {
    (0..count as u32).map(|i| (i * 37 + 5) % 256).collect()
}

fn make_adapter(kind: Qwen2AdapterKind, first_layer: usize) -> Qwen2Adapter {
    Qwen2Adapter::seeded(
        &config(),
        &Qwen2AdapterSpec::pinned("adapter-a", 1, kind, first_layer),
        SEED,
    )
    .unwrap()
}

struct Run {
    logits: Vec<Vec<u32>>,
    keys: Vec<Vec<u32>>,
    values: Vec<Vec<u32>>,
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn run(stack: &Qwen2PagedStack, adapter: Option<&Qwen2Adapter>, sequence: &[u32]) -> Run {
    let backend = backend();
    let mut cache = new_cache(&backend, stack);
    let regime = stack.admit_rope_regime(sequence.len(), 0).unwrap();
    let output = stack
        .prefill_bound_in_regime(
            &backend,
            &mut cache,
            Qwen2StackInput {
                tokens: sequence.to_vec(),
            },
            &regime,
            &Qwen2Binding { adapter },
        )
        .unwrap();
    let snapshot = backend.snapshot(&cache).unwrap();
    Run {
        logits: output.logits.iter().map(|row| bits(row)).collect(),
        keys: snapshot
            .layers
            .iter()
            .map(|l| bits(&l.keys.values))
            .collect(),
        values: snapshot
            .layers
            .iter()
            .map(|l| bits(&l.values.values))
            .collect(),
    }
}

fn differing_layers(a: &ptr_pods::KvLayerChains, b: &ptr_pods::KvLayerChains) -> Vec<usize> {
    (0..LAYERS).filter(|l| a.layer(*l) != b.layer(*l)).collect()
}

#[test]
fn adapter_seeds_are_pinned_nonzero_and_attached_at_every_layer_from_first() {
    let adapter = make_adapter(Qwen2AdapterKind::Qkv, 2);
    assert_eq!((adapter.rank(), adapter.scale()), (4, 2.0));
    assert!(adapter.layer(0).is_none() && adapter.layer(1).is_none());
    for layer in 2..LAYERS {
        let layer = adapter
            .layer(layer)
            .expect("attached at every layer >= first");
        for pair in [&layer.q, &layer.k, &layer.v] {
            let pair = pair.as_ref().unwrap();
            assert!(
                pair.b.iter().any(|value| *value != 0.0),
                "B must be non-zero"
            );
            assert!(pair.a.iter().any(|value| *value != 0.0));
        }
        assert!(layer.mlp.is_none());
    }
    // Seeded from (weight_seed, identity, generation): each of them changes the bits.
    let spec = Qwen2AdapterSpec::pinned("adapter-a", 1, Qwen2AdapterKind::Qkv, 2);
    let same = Qwen2Adapter::seeded(&config(), &spec, SEED).unwrap();
    assert_eq!(adapter, same);
    let other_seed = Qwen2Adapter::seeded(&config(), &spec, SEED + 1).unwrap();
    let other_generation = Qwen2Adapter::seeded(
        &config(),
        &Qwen2AdapterSpec::pinned("adapter-a", 2, Qwen2AdapterKind::Qkv, 2),
        SEED,
    )
    .unwrap();
    let other_identity = Qwen2Adapter::seeded(
        &config(),
        &Qwen2AdapterSpec::pinned("adapter-b", 1, Qwen2AdapterKind::Qkv, 2),
        SEED,
    )
    .unwrap();
    for other in [&other_seed, &other_generation, &other_identity] {
        assert_ne!(adapter.layer(2), other.layer(2));
    }
    let mlp = make_adapter(Qwen2AdapterKind::MlpOnly, 1);
    assert!(mlp.layer(1).unwrap().mlp.is_some() && mlp.layer(1).unwrap().q.is_none());
}

#[test]
fn invalid_adapter_specs_are_refused() {
    let seeded = |spec: Qwen2AdapterSpec| Qwen2Adapter::seeded(&config(), &spec, SEED);
    let good = Qwen2AdapterSpec::pinned("a", 1, Qwen2AdapterKind::Qkv, 0);
    assert_eq!(
        seeded(Qwen2AdapterSpec {
            identity: " ".into(),
            ..good.clone()
        })
        .unwrap_err(),
        Qwen2AdapterError::BlankIdentity
    );
    assert_eq!(
        seeded(Qwen2AdapterSpec {
            rank: 0,
            ..good.clone()
        })
        .unwrap_err(),
        Qwen2AdapterError::ZeroRank
    );
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            seeded(Qwen2AdapterSpec {
                scale,
                ..good.clone()
            })
            .unwrap_err(),
            Qwen2AdapterError::InvalidScale
        );
    }
    assert!(matches!(
        seeded(Qwen2AdapterSpec {
            first_layer: LAYERS,
            ..good
        })
        .unwrap_err(),
        Qwen2AdapterError::FirstLayerOutOfRange { .. }
    ));
}

#[test]
fn an_adapter_shaped_for_another_stack_is_refused_without_touching_the_cache() {
    let mut small = config();
    small.layer.hidden_size = 32;
    small.layer.num_attention_heads = 2;
    small.layer.num_key_value_heads = 1;
    let foreign = Qwen2Adapter::seeded(
        &small,
        &Qwen2AdapterSpec::pinned("a", 1, Qwen2AdapterKind::Qkv, 0),
        SEED,
    )
    .unwrap();
    let stack = stack();
    let backend = backend();
    let mut cache = new_cache(&backend, &stack);
    let before = backend.snapshot(&cache).unwrap();
    let regime = stack.admit_rope_regime(3, 0).unwrap();
    let error = stack
        .prefill_bound_in_regime(
            &backend,
            &mut cache,
            Qwen2StackInput { tokens: tokens(3) },
            &regime,
            &Qwen2Binding {
                adapter: Some(&foreign),
            },
        )
        .unwrap_err();
    assert_eq!(error, Qwen2StackError::Adapter(Qwen2AdapterError::Shape));
    assert_eq!(backend.snapshot(&cache).unwrap(), before);
}

#[test]
fn qkv_chains_equal_base_below_first_layer_and_differ_from_it_up() {
    let stack = stack();
    let base = derive_kv_layer_chains(&stack, &identity()).unwrap();
    for first in [0, 2] {
        let adapter = make_adapter(Qwen2AdapterKind::Qkv, first);
        let chains = derive_adapted_kv_layer_chains(&stack, &identity(), &adapter).unwrap();
        assert_eq!(chains.base(), &base);
        let differing = differing_layers(&base, chains.adapted());
        assert_eq!(
            differing,
            (first..LAYERS).collect::<Vec<_>>(),
            "f = {first}"
        );
        assert_eq!(chains.adapted().root(), base.root());
    }
}

#[test]
fn q_only_and_mlp_only_chains_change_only_above_the_adapted_layer() {
    let stack = stack();
    let base = derive_kv_layer_chains(&stack, &identity()).unwrap();
    for kind in [Qwen2AdapterKind::QOnly, Qwen2AdapterKind::MlpOnly] {
        for first in [0, 2] {
            let adapter = make_adapter(kind, first);
            let chains = derive_adapted_kv_layer_chains(&stack, &identity(), &adapter).unwrap();
            let differing = differing_layers(&base, chains.adapted());
            assert_eq!(
                differing,
                (first + 1..LAYERS).collect::<Vec<_>>(),
                "{kind:?} f = {first}"
            );
        }
    }
    // Last layer: a Q-only adapter there changes no chain at all (no layer above it).
    let last = make_adapter(Qwen2AdapterKind::QOnly, LAYERS - 1);
    let chains = derive_adapted_kv_layer_chains(&stack, &identity(), &last).unwrap();
    assert!(differing_layers(&base, chains.adapted()).is_empty());
}

#[test]
fn identity_generation_and_weight_bits_are_each_bound_into_the_chain() {
    let stack = stack();
    let original = make_adapter(Qwen2AdapterKind::Qkv, 0);
    let chains_of = |adapter: &Qwen2Adapter| {
        derive_adapted_kv_layer_chains(&stack, &identity(), adapter)
            .unwrap()
            .adapted()
            .clone()
    };
    let reference = chains_of(&original);
    // Identical weights, different identity.
    let relabelled = original.clone().relabelled("adapter-b", 1);
    assert_eq!(relabelled.layer(0), original.layer(0));
    assert_eq!(
        differing_layers(&reference, &chains_of(&relabelled)),
        vec![0, 1, 2, 3]
    );
    // Identical weights, different generation.
    let regenerated = original.clone().relabelled("adapter-a", 2);
    assert_eq!(regenerated.layer(0), original.layer(0));
    assert_eq!(
        differing_layers(&reference, &chains_of(&regenerated)),
        vec![0, 1, 2, 3]
    );
    // Same identity and generation, one B bit changed at layer 2. A K or V bit is a
    // stored-row dependency of layer 2 itself; a Q bit only reaches chain(3).
    let perturbed = |projection| {
        original
            .clone()
            .with_perturbed_b(2, projection, 1e-3)
            .unwrap()
    };
    for projection in [Qwen2AdapterProjection::K, Qwen2AdapterProjection::V] {
        assert_eq!(
            differing_layers(&reference, &chains_of(&perturbed(projection))),
            vec![2, 3],
            "{projection:?}"
        );
    }
    assert_eq!(
        differing_layers(
            &reference,
            &chains_of(&perturbed(Qwen2AdapterProjection::Q))
        ),
        vec![3]
    );
    assert!(original
        .clone()
        .with_perturbed_b(2, Qwen2AdapterProjection::Mlp, 1.0)
        .is_none());
    // The same adapter derives the same chains.
    assert_eq!(reference, chains_of(&original));
    // Q-only: identity binds through the layer term, one layer later.
    let q = make_adapter(Qwen2AdapterKind::QOnly, 1);
    let q_relabelled = q.clone().relabelled("adapter-b", 1);
    let a = chains_of(&q);
    let b = chains_of(&q_relabelled);
    assert_eq!(differing_layers(&a, &b), vec![2, 3]);
}

#[test]
fn the_adapter_changes_the_logits_and_the_base_path_is_untouched() {
    let stack = stack();
    let sequence = tokens(12);
    let unbound = run(&stack, None, &sequence);
    for kind in [
        Qwen2AdapterKind::Qkv,
        Qwen2AdapterKind::QOnly,
        Qwen2AdapterKind::MlpOnly,
    ] {
        let adapter = make_adapter(kind, 1);
        let adapted = run(&stack, Some(&adapter), &sequence);
        assert_ne!(
            adapted.logits, unbound.logits,
            "{kind:?} must change the logits"
        );
        // Layer 0 is below first_layer: its rows are the base rows.
        assert_eq!(adapted.keys[0], unbound.keys[0], "{kind:?}");
        assert_eq!(adapted.values[0], unbound.values[0], "{kind:?}");
    }
    // K/V adapters change stored rows from first_layer; Q-only changes them one layer later.
    let qkv = run(
        &stack,
        Some(&make_adapter(Qwen2AdapterKind::Qkv, 1)),
        &sequence,
    );
    assert_ne!(qkv.keys[1], unbound.keys[1]);
    assert_ne!(qkv.values[1], unbound.values[1]);
    let q_only = run(
        &stack,
        Some(&make_adapter(Qwen2AdapterKind::QOnly, 1)),
        &sequence,
    );
    assert_eq!(q_only.keys[1], unbound.keys[1]);
    assert_ne!(q_only.keys[2], unbound.keys[2]);
    // The default binding is the base model.
    let backend = backend();
    let mut cache = new_cache(&backend, &stack);
    let regime = stack.admit_rope_regime(sequence.len(), 0).unwrap();
    let default = stack
        .prefill_bound_in_regime(
            &backend,
            &mut cache,
            Qwen2StackInput {
                tokens: sequence.clone(),
            },
            &regime,
            &Qwen2Binding::default(),
        )
        .unwrap();
    let default: Vec<Vec<u32>> = default.logits.iter().map(|row| bits(row)).collect();
    assert_eq!(default, unbound.logits);
}

#[test]
fn prefill_then_decode_with_an_adapter_equals_one_full_prefill() {
    let stack = stack();
    for kind in [
        Qwen2AdapterKind::Qkv,
        Qwen2AdapterKind::Activated { invocation: 5 },
    ] {
        let adapter = make_adapter(kind, 0);
        let sequence = tokens(10);
        let full = run(&stack, Some(&adapter), &sequence);
        let backend = backend();
        let mut cache = new_cache(&backend, &stack);
        let regime = stack.admit_rope_regime(sequence.len(), 0).unwrap();
        let binding = Qwen2Binding {
            adapter: Some(&adapter),
        };
        let mut logits: Vec<Vec<u32>> = stack
            .prefill_bound_in_regime(
                &backend,
                &mut cache,
                Qwen2StackInput {
                    tokens: sequence[..3].to_vec(),
                },
                &regime,
                &binding,
            )
            .unwrap()
            .logits
            .iter()
            .map(|row| bits(row))
            .collect();
        for token in &sequence[3..] {
            let out = stack
                .decode_bound_in_regime(
                    &backend,
                    &mut cache,
                    Qwen2StackInput {
                        tokens: vec![*token],
                    },
                    &regime,
                    &binding,
                )
                .unwrap();
            logits.push(bits(&out.logits[0]));
        }
        assert_eq!(logits, full.logits, "{kind:?}");
    }
}

#[test]
fn alora_rows_before_the_invocation_are_bitwise_the_base_rows() {
    const INVOCATION: usize = 39;
    let stack = stack();
    let sequence = tokens(CAPACITY);
    let base = run(&stack, None, &sequence);
    let adapter = make_adapter(
        Qwen2AdapterKind::Activated {
            invocation: INVOCATION,
        },
        0,
    );
    let adapted = run(&stack, Some(&adapter), &sequence);
    let kv_width = 2 * (64 / 4);
    let boundary = INVOCATION * kv_width;
    for layer in 0..LAYERS {
        assert_eq!(
            adapted.keys[layer][..boundary],
            base.keys[layer][..boundary],
            "keys {layer}"
        );
        assert_eq!(
            adapted.values[layer][..boundary],
            base.values[layer][..boundary],
            "values {layer}"
        );
        assert_ne!(
            adapted.keys[layer][boundary..],
            base.keys[layer][boundary..]
        );
        assert_ne!(
            adapted.values[layer][boundary..],
            base.values[layer][boundary..]
        );
    }
    assert_eq!(adapted.logits[..INVOCATION], base.logits[..INVOCATION]);
    assert_ne!(adapted.logits[INVOCATION], base.logits[INVOCATION]);
    // An invocation past the prompt never fires: the adapter is the identity on this run.
    let never = make_adapter(
        Qwen2AdapterKind::Activated {
            invocation: CAPACITY,
        },
        0,
    );
    assert_eq!(run(&stack, Some(&never), &sequence).logits, base.logits);
}

#[test]
fn alora_effective_chain_is_base_before_the_invocation_and_binds_it_after() {
    const INVOCATION: usize = 39;
    let stack = stack();
    let base = derive_kv_layer_chains(&stack, &identity()).unwrap();
    let adapter = make_adapter(
        Qwen2AdapterKind::Activated {
            invocation: INVOCATION,
        },
        0,
    );
    let chains = derive_adapted_kv_layer_chains(&stack, &identity(), &adapter).unwrap();
    assert_eq!(chains.activation_position(), INVOCATION);
    for layer in 0..LAYERS {
        // Wholly before the invocation, up to and including position 38: base chain.
        assert_eq!(
            chains.effective_layer(layer, 0..INVOCATION),
            base.layer(layer)
        );
        assert_eq!(
            chains.effective_layer(layer, 32..INVOCATION),
            base.layer(layer)
        );
        // At or after it, or straddling it: the adapted chain.
        for range in [INVOCATION..INVOCATION + 1, 40..44, 36..40, 0..64] {
            assert_eq!(
                chains.effective_layer(layer, range),
                chains.adapted().layer(layer)
            );
        }
        assert_ne!(chains.adapted().layer(layer), base.layer(layer));
        assert!(chains.effective_layer(layer, 5..5).is_none());
    }
    assert!(chains.effective_layer(LAYERS, 0..1).is_none());
    // The adapted chain binds the invocation position: same weights, other invocation.
    let moved = adapter_with_invocation(INVOCATION + 1);
    let moved_chains = derive_adapted_kv_layer_chains(&stack, &identity(), &moved).unwrap();
    assert_eq!(
        differing_layers(chains.adapted(), moved_chains.adapted()),
        vec![0, 1, 2, 3]
    );
    // A non-activated adapter is adapted throughout.
    let qkv = make_adapter(Qwen2AdapterKind::Qkv, 2);
    let qkv_chains = derive_adapted_kv_layer_chains(&stack, &identity(), &qkv).unwrap();
    assert_eq!(
        qkv_chains.effective_layer(3, 0..4),
        qkv_chains.adapted().layer(3)
    );
    assert_eq!(qkv_chains.effective_layer(1, 0..4), base.layer(1));
}

fn adapter_with_invocation(invocation: usize) -> Qwen2Adapter {
    make_adapter(Qwen2AdapterKind::Activated { invocation }, 0)
}
