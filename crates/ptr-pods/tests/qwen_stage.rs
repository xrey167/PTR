//! Stage execution of the reference stack: running the layers in two spans, the residual
//! rows handed from one to the other, is the whole stack bit for bit, in logits and in the
//! K/V rows of each span's cache.

use ptr_pods::{
    InMemoryPagedKvBackend, KvContinuationExecutor, KvKeyEncoding, KvPageBinding, KvPageSize,
    KvTensorBackend, KvTensorDType, KvTensorSchema, PagedKvTensorBackend, Qwen2Activation,
    Qwen2LayerConfig, Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeScaling, Qwen2StackConfig,
    Qwen2StackError, Qwen2StackInput, StageRows, StageRun,
};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

const SEED: u64 = 20240607;
const CAPACITY: usize = 64;

fn stack() -> Qwen2PagedStack {
    let config = Qwen2StackConfig {
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
        key_encoding: KvKeyEncoding::PreRope,
        qk_norm: Qwen2QkNorm::Disabled,
    };
    Qwen2PagedStack::reference(config, SEED).unwrap()
}

fn backend() -> InMemoryPagedKvBackend {
    InMemoryPagedKvBackend::new(KvPageSize::new(4).unwrap())
}

fn cache(
    backend: &InMemoryPagedKvBackend,
    schema: KvTensorSchema,
) -> <InMemoryPagedKvBackend as KvTensorBackend>::Cache {
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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn two_stages_are_the_whole_stack_bit_for_bit_in_logits_and_in_each_stages_kv() {
    let stack = stack();
    let sequence = tokens(12);
    let (model, adapter, device) = (
        ModelVersion::from("m"),
        AdapterVersion::from("base"),
        DeviceId::from("cpu"),
    );

    // The whole stack: prefill 9 tokens, decode the rest one at a time.
    let whole_backend = backend();
    let mut whole = cache(
        &whole_backend,
        stack
            .kv_schema(model.clone(), adapter.clone(), device.clone())
            .unwrap(),
    );
    let mut expected: Vec<Vec<u32>> = stack
        .prefill(
            &whole_backend,
            &mut whole,
            Qwen2StackInput {
                tokens: sequence[..9].to_vec(),
            },
        )
        .unwrap()
        .logits
        .iter()
        .map(|row| bits(row))
        .collect();
    for token in &sequence[9..] {
        let out = stack
            .decode(
                &whole_backend,
                &mut whole,
                Qwen2StackInput {
                    tokens: vec![*token],
                },
            )
            .unwrap();
        expected.push(bits(&out.logits[0]));
    }

    // Two stages: [0, 2) and [2, 4), each with its own cache and backend.
    let (first, second) = (0..2, 2..4);
    let backend_a = backend();
    let backend_b = backend();
    let mut cache_a = cache(
        &backend_a,
        stack
            .stage_kv_schema(&first, model.clone(), adapter.clone(), device.clone())
            .unwrap(),
    );
    let mut cache_b = cache(
        &backend_b,
        stack
            .stage_kv_schema(&second, model, adapter, device)
            .unwrap(),
    );
    let regime = stack.admit_rope_regime(sequence.len(), 0).unwrap();
    let mut actual: Vec<Vec<u32>> = Vec::new();
    let mut run = |rows: &[u32], decode: bool| {
        let embedded = stack.embed_tokens(rows).unwrap();
        let mid = stack
            .run_stage_in_regime(
                &backend_a,
                &mut cache_a,
                StageRun::new(first.clone(), &embedded, rows.len(), shape(decode)),
                &regime,
            )
            .unwrap();
        let end = stack
            .run_stage_in_regime(
                &backend_b,
                &mut cache_b,
                StageRun::new(second.clone(), &mid.residual, mid.rows, shape(decode)),
                &regime,
            )
            .unwrap();
        assert_eq!(mid.first_position, end.first_position);
        for row in stack.logits_from_residual(&end.residual).unwrap() {
            actual.push(bits(&row));
        }
    };
    run(&sequence[..9], false);
    for token in &sequence[9..] {
        run(&[*token], true);
    }
    assert_eq!(actual, expected);

    // The K/V rows of each stage's cache are the matching layers of the whole cache.
    let whole_snapshot = whole_backend.snapshot(&whole).unwrap();
    let snapshot_a = backend_a.snapshot(&cache_a).unwrap();
    let snapshot_b = backend_b.snapshot(&cache_b).unwrap();
    assert_eq!(snapshot_a.layers.len(), 2);
    assert_eq!(snapshot_b.layers.len(), 2);
    for (index, layer) in snapshot_a.layers.iter().enumerate() {
        assert_eq!(
            bits(&layer.keys.values),
            bits(&whole_snapshot.layers[index].keys.values)
        );
        assert_eq!(
            bits(&layer.values.values),
            bits(&whole_snapshot.layers[index].values.values)
        );
    }
    for (index, layer) in snapshot_b.layers.iter().enumerate() {
        let whole_layer = &whole_snapshot.layers[2 + index];
        assert_eq!(bits(&layer.keys.values), bits(&whole_layer.keys.values));
        assert_eq!(bits(&layer.values.values), bits(&whole_layer.values.values));
    }
}

#[test]
fn a_bad_span_a_bad_shape_and_a_wrong_cache_are_refused() {
    let stack = stack();
    let backend = backend();
    let regime = stack.admit_rope_regime(8, 0).unwrap();
    let schema = |span: &std::ops::Range<usize>| {
        stack.stage_kv_schema(
            span,
            ModelVersion::from("m"),
            AdapterVersion::from("base"),
            DeviceId::from("cpu"),
        )
    };
    // 3..2 is reversed on purpose: an inverted span must be refused.
    #[allow(clippy::reversed_empty_ranges)]
    for span in [0..0, 3..2, 0..5, 4..4] {
        assert!(matches!(
            schema(&span),
            Err(Qwen2StackError::InvalidLayerSpan { .. })
        ));
    }
    let mut stage = cache(&backend, schema(&(0..2)).unwrap());
    let embedded = stack.embed_tokens(&[1, 2]).unwrap();
    // The rows do not match the declared count.
    assert!(matches!(
        stack.run_stage_in_regime(
            &backend,
            &mut stage,
            StageRun::new(0..2, &embedded, 3, StageRows::Prefill),
            &regime
        ),
        Err(Qwen2StackError::StageInputShape { .. })
    ));
    // A decode step takes one row.
    assert!(matches!(
        stack.run_stage_in_regime(
            &backend,
            &mut stage,
            StageRun::new(0..2, &embedded, 2, StageRows::DecodeStep),
            &regime
        ),
        Err(Qwen2StackError::DecodeNeedsOneToken { tokens: 2 })
    ));
    // A cache of two layers is no cache for a three-layer span.
    assert_eq!(
        stack.run_stage_in_regime(
            &backend,
            &mut stage,
            StageRun::new(0..3, &embedded, 2, StageRows::Prefill),
            &regime
        ),
        Err(Qwen2StackError::CacheSchemaMismatch)
    );
    // A token outside the vocabulary has no embedding.
    assert!(matches!(
        stack.embed_tokens(&[256]),
        Err(Qwen2StackError::TokenOutOfVocabulary { .. })
    ));
    // No rows, no logits.
    assert!(stack.logits_from_residual(&[]).is_err());
}

#[test]
fn the_rows_of_a_frame_must_sit_at_the_next_positions_of_the_cache() {
    let stack = stack();
    let backend = backend();
    let regime = stack.admit_rope_regime(8, 0).unwrap();
    let schema = stack
        .stage_kv_schema(
            &(0..2),
            ModelVersion::from("m"),
            AdapterVersion::from("base"),
            DeviceId::from("cpu"),
        )
        .unwrap();
    let mut stage = cache(&backend, schema);
    let embedded = stack.embed_tokens(&[1, 2]).unwrap();
    let run = |positions: &'static [u32]| {
        StageRun::new(0..2, &embedded, 2, StageRows::Prefill).at_positions(positions)
    };
    // A gap, a repeat and a wrong count are refused, and the cache stays where it was.
    for bad in [&[1u32, 2][..], &[0, 0], &[0]] {
        let bad: &'static [u32] = Vec::leak(bad.to_vec());
        assert!(matches!(
            stack.run_stage_in_regime(&backend, &mut stage, run(bad), &regime),
            Err(Qwen2StackError::StagePositions {
                expected_first: 0,
                rows: 2
            })
        ));
    }
    assert!(stack
        .run_stage_in_regime(&backend, &mut stage, run(&[0, 1]), &regime)
        .is_ok());
    // The same frame again is a replay at the wrong place: the cache is at position 2.
    assert!(matches!(
        stack.run_stage_in_regime(&backend, &mut stage, run(&[0, 1]), &regime),
        Err(Qwen2StackError::StagePositions {
            expected_first: 2,
            rows: 2
        })
    ));
}

fn shape(decode: bool) -> StageRows {
    if decode {
        StageRows::DecodeStep
    } else {
        StageRows::Prefill
    }
}
