use ptr_pods::{
    InMemoryPagedKvBackend, KvBackendError, KvContinuationExecutor, KvKeyEncoding, KvPageBinding,
    KvPageSize, KvTensorBackend, KvTensorDType, KvTensorSchema, PagedKvTensorBackend,
    Qwen2Activation, Qwen2DecoderInput, Qwen2DecoderWeights, Qwen2LayerConfig, Qwen2LinearWeights,
    Qwen2PagedDecoder, Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeScaling, Qwen2StackConfig,
    Qwen2StackError, Qwen2StackInput,
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
