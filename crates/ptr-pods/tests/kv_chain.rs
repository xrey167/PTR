//! Per-layer KV chain keys follow the weights and configuration (doc 36 section D).
//! Public API only.

use ptr_pods::{
    derive_kv_layer_chains, derive_kv_layer_chains_in_regime, KvChainError, KvChainIdentity,
    KvKeyEncoding, KvLayerChains, Qwen2Activation, Qwen2LayerConfig, Qwen2PagedStack, Qwen2QkNorm,
    Qwen2RopeScaling, Qwen2StackConfig, Qwen2StackError, Qwen2StackWeights,
};

const LAYERS: usize = 4;

fn config(encoding: KvKeyEncoding, qk_norm: Qwen2QkNorm) -> Qwen2StackConfig {
    Qwen2StackConfig {
        layer: Qwen2LayerConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rope_scaling: Qwen2RopeScaling::None,
            rms_norm_eps: 1e-6,
            hidden_act: Qwen2Activation::Silu,
            sliding_window: None,
        },
        num_layers: LAYERS,
        vocab_size: 32,
        key_encoding: encoding,
        qk_norm,
    }
}

fn identity() -> KvChainIdentity {
    KvChainIdentity {
        base_model: "tiny".to_owned(),
        generation: 1,
        build: "build-a".to_owned(),
        tokenizer: "ptr.byte@1".to_owned(),
    }
}

fn base(encoding: KvKeyEncoding, qk_norm: Qwen2QkNorm) -> (Qwen2StackConfig, Qwen2StackWeights) {
    let config = config(encoding, qk_norm);
    let stack = Qwen2PagedStack::reference(config.clone(), 5).unwrap();
    (config, stack.weights().clone())
}

fn chains(config: &Qwen2StackConfig, weights: &Qwen2StackWeights) -> KvLayerChains {
    chains_with(config, weights, &identity())
}

fn chains_with(
    config: &Qwen2StackConfig,
    weights: &Qwen2StackWeights,
    identity: &KvChainIdentity,
) -> KvLayerChains {
    let stack = Qwen2PagedStack::new(config.clone(), weights.clone()).unwrap();
    derive_kv_layer_chains(&stack, identity).unwrap()
}

/// Layers whose chain differs between `a` and `b`.
fn differing(a: &KvLayerChains, b: &KvLayerChains) -> Vec<usize> {
    (0..LAYERS).filter(|l| a.layer(*l) != b.layer(*l)).collect()
}

#[test]
fn chain_is_deterministic_and_layers_are_distinct() {
    let (config, weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Enabled);
    let a = chains(&config, &weights);
    assert_eq!(a, chains(&config, &weights));
    assert_eq!(a.layer_count(), LAYERS);
    assert_eq!(a.layer(LAYERS), None);
    assert_eq!(a.final_id(), a.layer(LAYERS - 1).unwrap());
    let mut all: Vec<[u8; 32]> = (0..LAYERS).map(|l| a.layer(l).unwrap()).collect();
    all.push(a.root());
    all.sort();
    all.dedup();
    assert_eq!(all.len(), LAYERS + 1);
}

#[test]
fn key_norm_change_at_layer_l_changes_chain_from_l_only() {
    let (config, base_weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Enabled);
    let reference = chains(&config, &base_weights);
    for l in 0..LAYERS {
        let mut weights = base_weights.clone();
        weights.layers[l].key_norm.as_mut().unwrap()[0] += 0.5;
        let changed = chains(&config, &weights);
        assert_eq!(
            differing(&reference, &changed),
            (l..LAYERS).collect::<Vec<_>>()
        );
        assert_eq!(reference.root(), changed.root());
    }
}

#[test]
fn query_norm_change_at_layer_l_changes_chain_after_l_only() {
    let (config, base_weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Enabled);
    let reference = chains(&config, &base_weights);
    let mut weights = base_weights.clone();
    weights.layers[1].query_norm.as_mut().unwrap()[0] += 0.5;
    assert_eq!(
        differing(&reference, &chains(&config, &weights)),
        vec![2, 3]
    );
}

#[test]
fn ffn_change_at_layer_l_changes_chain_after_l_only() {
    let (config, base_weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);
    let reference = chains(&config, &base_weights);
    for l in 0..LAYERS - 1 {
        for part in 0..3 {
            let mut weights = base_weights.clone();
            let decoder = &mut weights.layers[l].decoder;
            match part {
                0 => decoder.gate.weight[3] += 0.25,
                1 => decoder.up.weight[3] += 0.25,
                _ => decoder.down.weight[3] += 0.25,
            }
            let changed = chains(&config, &weights);
            assert_eq!(
                differing(&reference, &changed),
                (l + 1..LAYERS).collect::<Vec<_>>(),
                "ffn part {part} at layer {l}"
            );
        }
    }
    // The last layer's FFN runs after the last K/V is stored: no chain moves.
    let mut weights = base_weights.clone();
    weights.layers[LAYERS - 1].decoder.down.weight[0] += 0.25;
    assert!(differing(&reference, &chains(&config, &weights)).is_empty());
}

#[test]
fn kv_projection_and_input_norm_change_chain_from_l() {
    let (config, base_weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);
    let reference = chains(&config, &base_weights);
    let edits: [fn(&mut Qwen2StackWeights); 4] = [
        |w| w.layers[2].decoder.key.weight[0] += 0.5,
        |w| w.layers[2].decoder.value.weight[0] += 0.5,
        |w| w.layers[2].decoder.key.bias.as_mut().unwrap()[0] += 0.5,
        |w| w.layers[2].decoder.input_layernorm[0] += 0.5,
    ];
    for edit in edits {
        let mut weights = base_weights.clone();
        edit(&mut weights);
        assert_eq!(
            differing(&reference, &chains(&config, &weights)),
            vec![2, 3]
        );
    }
}

#[test]
fn signed_zero_is_kept_bitwise() {
    let (config, base_weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);
    let mut positive = base_weights.clone();
    positive.layers[0].decoder.key.weight[0] = 0.0;
    let mut negative = base_weights;
    negative.layers[0].decoder.key.weight[0] = -0.0;
    assert_ne!(chains(&config, &positive), chains(&config, &negative));
}

#[test]
fn rope_theta_binds_layers_from_one_for_pre_rope_and_all_for_post_rope() {
    let (mut config, weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);
    let pre = chains(&config, &weights);
    config.layer.rope_theta = 500_000.0;
    assert_eq!(differing(&pre, &chains(&config, &weights)), vec![1, 2, 3]);

    let (mut config, weights) = base(KvKeyEncoding::PostRope, Qwen2QkNorm::Disabled);
    let post = chains(&config, &weights);
    config.layer.rope_theta = 500_000.0;
    assert_eq!(
        differing(&post, &chains(&config, &weights)),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn build_identity_embedding_base_model_and_tokenizer_change_every_chain() {
    let (config, weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);
    let reference = chains(&config, &weights);
    let all = (0..LAYERS).collect::<Vec<_>>();

    let mut other = identity();
    other.build = "build-b".to_owned();
    assert_eq!(
        differing(&reference, &chains_with(&config, &weights, &other)),
        all
    );
    let mut other = identity();
    other.generation = 2;
    assert_eq!(
        differing(&reference, &chains_with(&config, &weights, &other)),
        all
    );
    let mut other = identity();
    other.base_model = "other".to_owned();
    assert_eq!(
        differing(&reference, &chains_with(&config, &weights, &other)),
        all
    );
    let mut other = identity();
    other.tokenizer = "ptr.byte@2".to_owned();
    assert_eq!(
        differing(&reference, &chains_with(&config, &weights, &other)),
        all
    );

    let mut edited = weights;
    edited.embedding[7] += 0.5;
    let changed = chains(&config, &edited);
    assert_eq!(differing(&reference, &changed), all);
    assert_ne!(reference.root(), changed.root());
}

#[test]
fn same_seed_different_weights_get_different_chains() {
    let config = config(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);
    let a = Qwen2PagedStack::reference(config.clone(), 9).unwrap();
    let again = Qwen2PagedStack::reference(config.clone(), 9).unwrap();
    let mut weights = a.weights().clone();
    weights.layers[0].decoder.query.weight[0] += 1e-3;
    let perturbed = Qwen2PagedStack::new(config, weights).unwrap();
    let id = identity();
    let reference = derive_kv_layer_chains(&a, &id).unwrap();
    assert_eq!(reference, derive_kv_layer_chains(&again, &id).unwrap());
    assert_ne!(
        reference.final_id(),
        derive_kv_layer_chains(&perturbed, &id).unwrap().final_id()
    );
}

#[test]
fn blank_identity_and_nan_weights_are_refused() {
    let (config, weights) = base(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);
    let stack = Qwen2PagedStack::new(config.clone(), weights.clone()).unwrap();
    for field in ["base_model", "build", "tokenizer"] {
        let mut blank = identity();
        match field {
            "base_model" => blank.base_model = " ".to_owned(),
            "build" => blank.build.clear(),
            _ => blank.tokenizer = String::new(),
        }
        assert!(matches!(
            derive_kv_layer_chains(&stack, &blank),
            Err(KvChainError::BlankIdentity(name)) if name == field
        ));
    }
    let mut nan = weights.clone();
    nan.embedding[0] = f32::NAN;
    assert_eq!(
        Qwen2PagedStack::new(config.clone(), nan).unwrap_err(),
        Qwen2StackError::InvalidWeights("non-finite weight")
    );
    let mut infinite = weights;
    infinite.layers[2].decoder.key.bias.as_mut().unwrap()[0] = f32::INFINITY;
    assert_eq!(
        Qwen2PagedStack::new(config, infinite).unwrap_err(),
        Qwen2StackError::InvalidWeights("non-finite weight")
    );
}

// ---- RoPE regime fixed at admission ----

fn ntk(encoding: KvKeyEncoding, factor: f32) -> Qwen2PagedStack {
    let mut config = config(encoding, Qwen2QkNorm::Disabled);
    config.layer.rope_scaling = Qwen2RopeScaling::DynamicNtk {
        factor,
        original_max_position_embeddings: 8,
    };
    Qwen2PagedStack::reference(config, 5).unwrap()
}

fn in_regime(stack: &Qwen2PagedStack, admitted: usize) -> KvLayerChains {
    let regime = stack.admit_rope_regime(admitted, 0).unwrap();
    derive_kv_layer_chains_in_regime(stack, &identity(), &regime).unwrap()
}

#[test]
fn chain_follows_the_regime_per_the_rope_rule() {
    // PostRope: the stored key is rotated, so the regime enters every layer including 0.
    let post = ntk(KvKeyEncoding::PostRope, 4.0);
    assert_eq!(
        differing(&in_regime(&post, 12), &in_regime(&post, 30)),
        vec![0, 1, 2, 3]
    );
    // PreRope: layer 0 projects from the embedding before any attention, so layers >= 1.
    let pre = ntk(KvKeyEncoding::PreRope, 4.0);
    assert_eq!(
        differing(&in_regime(&pre, 12), &in_regime(&pre, 30)),
        vec![1, 2, 3]
    );
    // Deterministic, and one regime is one chain.
    assert_eq!(in_regime(&pre, 30), in_regime(&pre, 30));
}

#[test]
fn lengths_within_the_original_maximum_share_one_regime() {
    for encoding in [KvKeyEncoding::PreRope, KvKeyEncoding::PostRope] {
        let stack = ntk(encoding, 4.0);
        // Both give the plain base, so the chains are equal and entries may be shared.
        let a = in_regime(&stack, 3);
        assert_eq!(a, in_regime(&stack, 8));
        assert_ne!(a, in_regime(&stack, 9));
    }
}

#[test]
fn scaling_policy_changes_the_chain_even_at_equal_base() {
    let four = ntk(KvKeyEncoding::PostRope, 4.0);
    let two = ntk(KvKeyEncoding::PostRope, 2.0);
    // Within omp both stacks use the plain base, yet the policy is bound.
    assert_eq!(
        differing(&in_regime(&four, 4), &in_regime(&two, 4)),
        vec![0, 1, 2, 3]
    );
    // No scaling at all is a different policy from dynamic NTK at the plain base.
    let (config, weights) = base(KvKeyEncoding::PostRope, Qwen2QkNorm::Disabled);
    assert_eq!(
        differing(&chains(&config, &weights), &in_regime(&four, 4)),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn regime_is_required_and_checked() {
    let stack = ntk(KvKeyEncoding::PreRope, 4.0);
    assert_eq!(
        derive_kv_layer_chains(&stack, &identity()).unwrap_err(),
        KvChainError::RopeRegimeRequired
    );
    // A regime of another configuration is refused.
    let other = ntk(KvKeyEncoding::PreRope, 2.0);
    let foreign = other.admit_rope_regime(30, 0).unwrap();
    assert_eq!(
        derive_kv_layer_chains_in_regime(&stack, &identity(), &foreign).unwrap_err(),
        KvChainError::RopeRegimeMismatch
    );
}
