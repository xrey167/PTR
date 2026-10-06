use ptr_pods::{
    InMemoryPagedKvBackend, KvBackendError, KvContinuationExecutor, KvKeyEncoding, KvPageBinding,
    KvPageSize, KvTensorBackend, KvTensorDType, KvTensorSchema, PagedKvTensorBackend,
    Qwen2Activation, Qwen2DecoderInput, Qwen2DecoderWeights, Qwen2LayerConfig, Qwen2LinearWeights,
    Qwen2PagedDecoder, Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeRegime, Qwen2RopeScaling,
    Qwen2StackConfig, Qwen2StackError, Qwen2StackInput,
};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

const SEED: u64 = 20240607;
const CAPACITY: usize = 64;

fn config(encoding: KvKeyEncoding, qk_norm: Qwen2QkNorm) -> Qwen2StackConfig {
    // R005 tiny_model_config: 4 layers, hidden 64, 4 heads, 2 KV heads, head_dim 16, vocab 256.
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
        num_layers: 4,
        vocab_size: 256,
        key_encoding: encoding,
        qk_norm,
    }
}

fn stack(encoding: KvKeyEncoding) -> Qwen2PagedStack {
    Qwen2PagedStack::reference(config(encoding, Qwen2QkNorm::Disabled), SEED).unwrap()
}

fn backend() -> InMemoryPagedKvBackend {
    InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap())
}

fn schema_for(stack: &Qwen2PagedStack) -> KvTensorSchema {
    stack
        .kv_schema(
            ModelVersion::from("r005-tiny"),
            AdapterVersion::from("base"),
            DeviceId::from("cpu"),
        )
        .unwrap()
}

fn binding(schema: &KvTensorSchema) -> KvPageBinding {
    KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [9; 32],
        principal: PrincipalId::from("alice"),
        device: schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::new(4).unwrap(),
    }
}

fn new_cache(
    backend: &InMemoryPagedKvBackend,
    schema: KvTensorSchema,
) -> <InMemoryPagedKvBackend as KvTensorBackend>::Cache {
    let binding = binding(&schema);
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

fn bits(vector: &[f32]) -> Vec<u32> {
    vector.iter().map(|value| value.to_bits()).collect()
}

/// Logits of every position of `sequence`, one prefill call.
fn full_logits(stack: &Qwen2PagedStack, sequence: &[u32]) -> Vec<Vec<u32>> {
    let backend = backend();
    let mut cache = new_cache(&backend, schema_for(stack));
    stack
        .prefill(&backend, &mut cache, input(sequence))
        .unwrap()
        .logits
        .iter()
        .map(|logits| bits(logits))
        .collect()
}

#[test]
fn four_layer_prefill_then_decode_is_bitwise_equal_to_full_prefill() {
    for encoding in [KvKeyEncoding::PreRope, KvKeyEncoding::PostRope] {
        let stack = stack(encoding);
        let sequence = tokens(12, 5);
        let expected = full_logits(&stack, &sequence);

        let backend = backend();
        let mut cache = new_cache(&backend, schema_for(&stack));
        let prefix = 7;
        let mut actual: Vec<Vec<u32>> = stack
            .prefill(&backend, &mut cache, input(&sequence[..prefix]))
            .unwrap()
            .logits
            .iter()
            .map(|logits| bits(logits))
            .collect();
        for token in &sequence[prefix..] {
            let out = stack
                .decode(&backend, &mut cache, input(&[*token]))
                .unwrap();
            actual.push(bits(&out.logits[0]));
        }
        assert_eq!(actual, expected, "{encoding:?}");
        assert!(expected.iter().flatten().any(|bits| *bits != 0));
    }
}

#[test]
fn pre_and_post_rope_logits_are_bitwise_equal_at_canonical_positions() {
    for qk_norm in [Qwen2QkNorm::Disabled, Qwen2QkNorm::Enabled] {
        let build = |encoding| Qwen2PagedStack::reference(config(encoding, qk_norm), SEED).unwrap();
        let sequence = tokens(16, 11);
        assert_eq!(
            full_logits(&build(KvKeyEncoding::PreRope), &sequence),
            full_logits(&build(KvKeyEncoding::PostRope), &sequence),
            "{qk_norm:?}"
        );
    }
}

#[test]
fn qk_norm_changes_the_model_and_is_off_by_default_in_tests() {
    let sequence = tokens(6, 3);
    let off = full_logits(&stack(KvKeyEncoding::PreRope), &sequence);
    let on = Qwen2PagedStack::reference(config(KvKeyEncoding::PreRope, Qwen2QkNorm::Enabled), SEED)
        .unwrap();
    assert_ne!(off, full_logits(&on, &sequence));
}

#[test]
fn schema_digest_differs_between_encodings_and_caches_are_not_interchangeable() {
    let pre = stack(KvKeyEncoding::PreRope);
    let post = stack(KvKeyEncoding::PostRope);
    let (pre_schema, post_schema) = (schema_for(&pre), schema_for(&post));
    assert_eq!(pre_schema.key_encoding(), KvKeyEncoding::PreRope);
    assert_eq!(post_schema.key_encoding(), KvKeyEncoding::PostRope);
    assert_ne!(pre_schema, post_schema);
    // Applying an encoding is idempotent and reversible.
    assert_eq!(
        pre_schema.clone().with_key_encoding(KvKeyEncoding::PreRope),
        pre_schema
    );
    assert_eq!(
        pre_schema
            .clone()
            .with_key_encoding(KvKeyEncoding::PostRope),
        post_schema
    );

    let backend = backend();
    let sequence = tokens(5, 1);
    let mut pre_cache = new_cache(&backend, pre_schema);
    let mut post_cache = new_cache(&backend, post_schema);
    pre.prefill(&backend, &mut pre_cache, input(&sequence))
        .unwrap();
    post.prefill(&backend, &mut post_cache, input(&sequence))
        .unwrap();
    let pre_snapshot = backend.snapshot(&pre_cache).unwrap();
    let post_snapshot = backend.snapshot(&post_cache).unwrap();
    assert_ne!(pre_snapshot.digest, post_snapshot.digest);
    assert!(pre_snapshot.verify_digest() && post_snapshot.verify_digest());

    // Reuse across encodings is refused, in both directions, and leaves the cache unchanged.
    let before = backend.snapshot(&post_cache).unwrap();
    assert_eq!(
        pre.prefill(&backend, &mut post_cache, input(&sequence[..1]))
            .unwrap_err(),
        Qwen2StackError::KeyEncodingMismatch {
            expected: KvKeyEncoding::PreRope,
            found: KvKeyEncoding::PostRope,
        }
    );
    assert_eq!(
        post.decode(&backend, &mut pre_cache, input(&sequence[..1]))
            .unwrap_err(),
        Qwen2StackError::KeyEncodingMismatch {
            expected: KvKeyEncoding::PostRope,
            found: KvKeyEncoding::PreRope,
        }
    );
    assert_eq!(backend.snapshot(&post_cache).unwrap(), before);
}

#[test]
fn pre_rope_layer0_keys_are_position_independent_while_post_rope_keys_are_not() {
    let prefix = tokens(6, 17);
    let shift = 5;
    let padding = tokens(shift, 99);
    let shifted: Vec<u32> = padding.iter().chain(&prefix).copied().collect();
    let width = 2 * 16;

    let layer0_prefix_rows = |encoding: KvKeyEncoding, sequence: &[u32], skip: usize| {
        let stack = stack(encoding);
        let backend = backend();
        let mut cache = new_cache(&backend, schema_for(&stack));
        stack
            .prefill(&backend, &mut cache, input(sequence))
            .unwrap();
        let keys = backend.snapshot(&cache).unwrap().layers[0].keys.clone();
        assert_eq!(keys.shape[1], width);
        bits(&keys.values[skip * width..(skip + prefix.len()) * width])
    };

    let pre_at_zero = layer0_prefix_rows(KvKeyEncoding::PreRope, &prefix, 0);
    let pre_shifted = layer0_prefix_rows(KvKeyEncoding::PreRope, &shifted, shift);
    assert_eq!(pre_at_zero, pre_shifted, "pre-RoPE rows must not move");

    let post_at_zero = layer0_prefix_rows(KvKeyEncoding::PostRope, &prefix, 0);
    let post_shifted = layer0_prefix_rows(KvKeyEncoding::PostRope, &shifted, shift);
    assert_ne!(
        post_at_zero, post_shifted,
        "post-RoPE rows are position bound"
    );
    // Position 0 is the identity rotation, so the first row alone is equal in both encodings.
    assert_eq!(pre_at_zero[..width], post_at_zero[..width]);
    assert_ne!(pre_at_zero[width..], post_at_zero[width..]);
}

#[test]
fn invalid_configs_are_rejected_with_typed_errors() {
    let valid = config(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled);

    let mut zero_layers = valid.clone();
    zero_layers.num_layers = 0;
    assert_eq!(
        Qwen2PagedStack::reference(zero_layers, SEED).unwrap_err(),
        Qwen2StackError::ZeroLayers
    );

    let mut zero_vocab = valid.clone();
    zero_vocab.vocab_size = 0;
    assert_eq!(
        Qwen2PagedStack::reference(zero_vocab, SEED).unwrap_err(),
        Qwen2StackError::ZeroVocab
    );

    let mut heads = valid.clone();
    heads.layer.num_key_value_heads = 3;
    assert_eq!(
        Qwen2PagedStack::reference(heads, SEED).unwrap_err(),
        Qwen2StackError::HeadsNotDivisibleByKvHeads {
            heads: 4,
            kv_heads: 3
        }
    );

    let mut scaling = valid;
    scaling.layer.rope_scaling = Qwen2RopeScaling::Unsupported("yarn".into());
    assert!(matches!(
        Qwen2PagedStack::reference(scaling, SEED).unwrap_err(),
        Qwen2StackError::InvalidLayer(KvBackendError::InvalidSchema(_))
    ));
}

#[test]
fn invalid_inputs_are_rejected_without_touching_the_cache() {
    let stack = stack(KvKeyEncoding::PreRope);
    let backend = backend();
    let mut cache = new_cache(&backend, schema_for(&stack));
    let empty = backend.snapshot(&cache).unwrap();
    assert_eq!(
        stack.prefill(&backend, &mut cache, input(&[])).unwrap_err(),
        Qwen2StackError::EmptyInput
    );
    assert_eq!(
        stack
            .prefill(&backend, &mut cache, input(&[1, 256]))
            .unwrap_err(),
        Qwen2StackError::TokenOutOfVocabulary {
            token: 256,
            vocab_size: 256
        }
    );
    assert_eq!(
        stack
            .decode(&backend, &mut cache, input(&[1, 2]))
            .unwrap_err(),
        Qwen2StackError::DecodeNeedsOneToken { tokens: 2 }
    );
    assert_eq!(
        stack
            .prefill(&backend, &mut cache, input(&tokens(CAPACITY + 1, 0)))
            .unwrap_err(),
        Qwen2StackError::ContextExceeded {
            required: CAPACITY + 1,
            maximum: CAPACITY
        }
    );
    assert_eq!(backend.snapshot(&cache).unwrap(), empty);

    // A cache with a different layer count is a schema mismatch.
    let mut wrong = schema_for(&stack);
    wrong.layer_count = 2;
    let mut other = new_cache(&backend, wrong);
    assert_eq!(
        stack
            .prefill(&backend, &mut other, input(&[1]))
            .unwrap_err(),
        Qwen2StackError::CacheSchemaMismatch
    );
}

#[test]
fn one_layer_decoder_refuses_a_pre_rope_cache() {
    let layer = Qwen2LayerConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_position_embeddings: 8,
        rope_theta: 10_000.0,
        rope_scaling: Qwen2RopeScaling::None,
        rms_norm_eps: 1e-6,
        hidden_act: Qwen2Activation::Silu,
        sliding_window: None,
    };
    let linear = |output: usize, input: usize| Qwen2LinearWeights {
        output_size: output,
        input_size: input,
        weight: vec![0.1; output * input],
        bias: None,
    };
    let decoder = Qwen2PagedDecoder::new(
        layer,
        Qwen2DecoderWeights {
            input_layernorm: vec![1.0; 4],
            query: linear(4, 4),
            key: linear(2, 4),
            value: linear(2, 4),
            output: linear(4, 4),
            post_attention_layernorm: vec![1.0; 4],
            gate: linear(8, 4),
            up: linear(8, 4),
            down: linear(4, 8),
        },
    )
    .unwrap();
    let schema = KvTensorSchema {
        model: ModelVersion::from("one-layer"),
        adapter: AdapterVersion::from("base"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 1,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let backend = backend();
    let hidden = Qwen2DecoderInput {
        hidden_states: vec![vec![1.0, 0.0, 0.5, -0.5]],
    };
    let mut post = new_cache(&backend, schema.clone());
    decoder
        .prefill(&backend, &mut post, hidden.clone())
        .unwrap();
    let mut pre = new_cache(&backend, schema.with_key_encoding(KvKeyEncoding::PreRope));
    assert_eq!(
        decoder.prefill(&backend, &mut pre, hidden).unwrap_err(),
        KvBackendError::SnapshotSchemaMismatch
    );
}

#[test]
fn spoofed_model_marker_is_refused_and_a_legitimate_marker_is_accepted() {
    let backend = backend();
    let stack = stack(KvKeyEncoding::PreRope);
    let legit = schema_for(&stack);
    assert!(legit.validate_key_encoding().is_ok());
    assert!(backend
        .allocate_bound(legit.clone(), CAPACITY, binding(&legit))
        .is_ok());

    for name in [
        "m#kv-key-encoding=post-rope",
        "m#kv-key-encoding=pre-rope-x",
        "m#kv-key-encoding=#kv-key-encoding=pre-rope",
        "#kv-key-encoding=",
    ] {
        let mut spoof = legit.clone();
        spoof.model = ModelVersion::from(name);
        assert!(spoof.validate_key_encoding().is_err(), "{name}");
        assert!(
            matches!(
                backend.allocate_bound(spoof.clone(), CAPACITY, binding(&spoof)),
                Err(KvBackendError::InvalidSchema(_))
            ),
            "{name}"
        );
    }
}

#[test]
fn reference_attention_executor_is_rope_free_and_refuses_a_pre_rope_cache() {
    use ptr_pods::{
        CausalAttentionInput, CausalAttentionWeights, ReferenceCausalAttentionExecutor,
    };
    let executor =
        ReferenceCausalAttentionExecutor::new(CausalAttentionWeights::identity(4).unwrap(), 2)
            .unwrap();
    let schema = KvTensorSchema {
        model: ModelVersion::from("attn"),
        adapter: AdapterVersion::from("base"),
        device: DeviceId::from("cpu"),
        layer_count: 1,
        attention_heads: 2,
        key_value_heads: 2,
        head_dim: 2,
        batch_size: 1,
        dtype: KvTensorDType::F32,
    };
    let backend = backend();
    let tokens = CausalAttentionInput {
        tokens: vec![vec![1.0, 0.0, 0.5, -0.5]],
    };
    let mut post = new_cache(&backend, schema.clone());
    executor
        .prefill(&backend, &mut post, tokens.clone())
        .unwrap();
    let mut pre = new_cache(&backend, schema.with_key_encoding(KvKeyEncoding::PreRope));
    assert_eq!(
        executor.prefill(&backend, &mut pre, tokens).unwrap_err(),
        KvBackendError::SnapshotSchemaMismatch
    );
}

// ---- dynamic NTK scaling, regime fixed at admission (doc 36 section D) ----

const OMP: usize = 8;
const FACTOR: f32 = 4.0;

fn ntk_config(encoding: KvKeyEncoding) -> Qwen2StackConfig {
    let mut config = config(encoding, Qwen2QkNorm::Disabled);
    config.layer.rope_scaling = Qwen2RopeScaling::DynamicNtk {
        factor: FACTOR,
        original_max_position_embeddings: OMP,
    };
    config
}

fn ntk_stack(encoding: KvKeyEncoding) -> Qwen2PagedStack {
    Qwen2PagedStack::reference(ntk_config(encoding), SEED).unwrap()
}

/// Independent closed form of the Hugging Face `_compute_dynamic_ntk_parameters` base.
fn expected_base(seq_len: usize) -> f32 {
    if seq_len <= OMP {
        return 10_000.0;
    }
    let (factor, head_dim) = (f64::from(FACTOR), 16.0_f64);
    let scale = factor * seq_len as f64 / OMP as f64 - (factor - 1.0);
    (10_000.0_f64 * scale.powf(head_dim / (head_dim - 2.0))) as f32
}

fn in_regime_logits(
    stack: &Qwen2PagedStack,
    sequence: &[u32],
    prefix: usize,
    regime: &Qwen2RopeRegime,
) -> Vec<Vec<u32>> {
    let backend = backend();
    let mut cache = new_cache(&backend, schema_for(stack));
    let mut logits: Vec<Vec<u32>> = stack
        .prefill_in_regime(&backend, &mut cache, input(&sequence[..prefix]), regime)
        .unwrap()
        .logits
        .iter()
        .map(|row| bits(row))
        .collect();
    for token in &sequence[prefix..] {
        let out = stack
            .decode_in_regime(&backend, &mut cache, input(&[*token]), regime)
            .unwrap();
        logits.push(bits(&out.logits[0]));
    }
    logits
}

#[test]
fn regime_follows_the_hugging_face_dynamic_ntk_formula() {
    let stack = ntk_stack(KvKeyEncoding::PreRope);
    for admitted in [1, 7, OMP, OMP + 1, 14, 33, 60] {
        let regime = stack.admit_rope_regime(admitted, 0).unwrap();
        assert_eq!(
            regime.effective_base().to_bits(),
            expected_base(admitted).to_bits(),
            "admitted {admitted}"
        );
        assert_eq!(regime.admitted_sequence_length(), admitted);
    }
    // Every length up to omp is the plain base, so one regime; above it the base grows.
    let plain = stack.admit_rope_regime(3, 2).unwrap();
    assert_eq!(plain.effective_base(), 10_000.0);
    assert_eq!(
        plain.effective_base_bits(),
        stack
            .admit_rope_regime(OMP, 0)
            .unwrap()
            .effective_base_bits()
    );
    let longer = stack.admit_rope_regime(OMP + 1, 0).unwrap();
    assert!(longer.effective_base() > 10_000.0);
    assert!(stack.admit_rope_regime(40, 0).unwrap().effective_base() > longer.effective_base());
    // The split into prompt and max_new_tokens does not matter, only the sum.
    assert_eq!(
        stack.admit_rope_regime(6, 8).unwrap(),
        stack.admit_rope_regime(10, 4).unwrap()
    );
    assert!(stack.admit_rope_regime(0, 0).is_err());
}

#[test]
fn decode_crossing_omp_under_the_admission_fixed_regime_is_bitwise_equal_to_full_prefill() {
    for encoding in [KvKeyEncoding::PreRope, KvKeyEncoding::PostRope] {
        let stack = ntk_stack(encoding);
        let sequence = tokens(14, 3);
        let prompt = 6;
        // Admitted at prompt + max_new_tokens = 14 > omp = 8; the decode crosses omp.
        let regime = stack
            .admit_rope_regime(prompt, sequence.len() - prompt)
            .unwrap();
        assert!(regime.effective_base() > 10_000.0);
        assert!(prompt < OMP && OMP < sequence.len());

        let expected = in_regime_logits(&stack, &sequence, sequence.len(), &regime);
        let actual = in_regime_logits(&stack, &sequence, prompt, &regime);
        assert_eq!(actual, expected, "{encoding:?}");
        assert!(expected.iter().flatten().any(|bits| *bits != 0));
    }
}

#[test]
fn the_regime_is_observable_in_the_logits() {
    let stack = ntk_stack(KvKeyEncoding::PreRope);
    let sequence = tokens(OMP, 9);
    let plain = stack.admit_rope_regime(OMP, 0).unwrap();
    let stretched = stack.admit_rope_regime(20, 0).unwrap();
    let a = in_regime_logits(&stack, &sequence, sequence.len(), &plain);
    let b = in_regime_logits(&stack, &sequence, sequence.len(), &stretched);
    // Position 0 has angle 0 under any base; later positions rotate differently.
    assert_eq!(a[0], b[0]);
    assert_ne!(a[1..], b[1..]);
    // Under the plain regime the NTK stack equals the unscaled stack bit for bit.
    let unscaled =
        Qwen2PagedStack::reference(config(KvKeyEncoding::PreRope, Qwen2QkNorm::Disabled), SEED)
            .unwrap();
    assert_eq!(a, full_logits(&unscaled, &sequence));
}

#[test]
fn regime_misuse_is_refused_without_touching_the_cache() {
    let stack = ntk_stack(KvKeyEncoding::PreRope);
    let backend = backend();
    let mut cache = new_cache(&backend, schema_for(&stack));
    let empty = backend.snapshot(&cache).unwrap();

    // The regime-less trait entry points cannot choose a regime for a scaled stack.
    assert_eq!(
        stack
            .prefill(&backend, &mut cache, input(&[1, 2]))
            .unwrap_err(),
        Qwen2StackError::RopeRegimeRequired
    );
    assert_eq!(
        stack.decode(&backend, &mut cache, input(&[1])).unwrap_err(),
        Qwen2StackError::RopeRegimeRequired
    );

    // A run past the admitted length is a regime transition: refused, never served.
    let regime = stack.admit_rope_regime(3, 2).unwrap();
    assert_eq!(
        stack
            .prefill_in_regime(&backend, &mut cache, input(&tokens(6, 1)), &regime)
            .unwrap_err(),
        Qwen2StackError::RopeRegimeExceeded {
            required: 6,
            admitted: 5
        }
    );

    // A regime of another configuration (other factor, or unscaled) is not accepted.
    let mut other = ntk_config(KvKeyEncoding::PreRope);
    other.layer.rope_scaling = Qwen2RopeScaling::DynamicNtk {
        factor: 2.0,
        original_max_position_embeddings: OMP,
    };
    let other = Qwen2PagedStack::reference(other, SEED).unwrap();
    let foreign = other.admit_rope_regime(20, 0).unwrap();
    assert_eq!(
        stack
            .prefill_in_regime(&backend, &mut cache, input(&[1]), &foreign)
            .unwrap_err(),
        Qwen2StackError::RopeRegimeMismatch
    );
    let unscaled = self::stack(KvKeyEncoding::PreRope);
    let mut unscaled_cache = new_cache(&backend, schema_for(&unscaled));
    assert_eq!(
        unscaled
            .prefill_in_regime(&backend, &mut unscaled_cache, input(&[1]), &foreign)
            .unwrap_err(),
        Qwen2StackError::RopeRegimeMismatch
    );
    assert_eq!(backend.snapshot(&cache).unwrap(), empty);
}

#[test]
fn unscaled_stack_default_path_is_unchanged() {
    let stack = stack(KvKeyEncoding::PreRope);
    let sequence = tokens(12, 5);
    let regime = stack.admit_rope_regime(7, 5).unwrap();
    assert_eq!(regime.effective_base(), 10_000.0);
    assert_eq!(
        in_regime_logits(&stack, &sequence, 7, &regime),
        full_logits(&stack, &sequence)
    );
}

#[test]
fn invalid_dynamic_ntk_parameters_are_rejected() {
    let invalid = |factor: f32, omp: usize| {
        let mut config = ntk_config(KvKeyEncoding::PreRope);
        config.layer.rope_scaling = Qwen2RopeScaling::DynamicNtk {
            factor,
            original_max_position_embeddings: omp,
        };
        Qwen2PagedStack::reference(config, SEED).unwrap_err()
    };
    for (factor, omp) in [
        (0.99, OMP),
        (0.0, OMP),
        (-2.0, OMP),
        (f32::NAN, OMP),
        (f32::INFINITY, OMP),
        (FACTOR, 0),
    ] {
        assert!(
            matches!(
                invalid(factor, omp),
                Qwen2StackError::InvalidLayer(KvBackendError::InvalidSchema(_))
            ),
            "factor {factor} omp {omp}"
        );
    }
    // factor == 1 is valid (boundary); the formula then reduces to (seq_len / omp)^(d/(d-2)).
    let mut config = ntk_config(KvKeyEncoding::PreRope);
    config.layer.rope_scaling = Qwen2RopeScaling::DynamicNtk {
        factor: 1.0,
        original_max_position_embeddings: OMP,
    };
    let boundary = Qwen2PagedStack::reference(config, SEED).unwrap();
    let expected = (10_000.0_f64 * (50.0_f64 / OMP as f64).powf(16.0 / 14.0)) as f32;
    assert_eq!(
        boundary
            .admit_rope_regime(50, 0)
            .unwrap()
            .effective_base()
            .to_bits(),
        expected.to_bits()
    );

    // head_dim 2 divides by zero in head_dim / (head_dim - 2).
    let mut narrow = ntk_config(KvKeyEncoding::PreRope);
    narrow.layer.hidden_size = 8;
    narrow.layer.num_attention_heads = 4;
    narrow.layer.num_key_value_heads = 2;
    narrow.layer.intermediate_size = 16;
    assert!(matches!(
        Qwen2PagedStack::reference(narrow, SEED).unwrap_err(),
        Qwen2StackError::InvalidLayer(KvBackendError::InvalidSchema(_))
    ));

    // The single-layer decoder has no regime and must not silently ignore scaling.
    let layer = ntk_config(KvKeyEncoding::PostRope).layer;
    let weights = ntk_stack(KvKeyEncoding::PostRope).weights().layers[0]
        .decoder
        .clone();
    assert!(matches!(
        Qwen2PagedDecoder::new(layer, weights),
        Err(KvBackendError::InvalidSchema(_))
    ));
}
