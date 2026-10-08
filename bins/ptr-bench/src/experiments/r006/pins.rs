//! The one-layer model pins of R006.
//!
//! The shared pin helpers of `layered_kv::pins` describe the four-layer model of R005, R007,
//! R009 and R010 (`SHAPE.layers = 4`). R006 pins a one-layer model: the same seeded weights,
//! hidden size, heads and vocabulary, with `layers = 1`, whose canonical configuration digest
//! is the `tiny_model_config_digest` of `config.toml`.
//!
//! The checks of this module need no backend. A backend with a configurable layer count does
//! not exist yet (the planned `ReferenceKvReuseBackend::with_layers`), so the comparison of the
//! backend's chain id and cold run with the independent one-layer stack built here is part of
//! the harness (`cases`, not built yet); nothing here pretends to have made it.
//!
//! The checks are recorded the way the shared ones are (a [`PinChecks`], one mismatch per
//! failed check), and the checkpoint and the topology have no real counterpart: both are the
//! string `none: seeded random` / `none: dense tiny decoder`, as in R007.

use super::params;
use crate::experiments::layered_kv::emit::Json;
use crate::experiments::layered_kv::lab::{fail, Res};
use crate::experiments::layered_kv::pins as shared;
pub use crate::experiments::layered_kv::pins::{canonical_digest, PinChecks};
use ptr_model_api::{ByteTokenizer, Tokenizer};
use ptr_pods::{
    derive_kv_layer_chains_in_regime, KvChainIdentity, KvKeyEncoding, Qwen2Activation,
    Qwen2LayerConfig, Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeScaling, Qwen2StackConfig,
};

/// The tiny model of R006 (`tiny_model_config`): the shared shape with one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub layers: u64,
    pub hidden_size: u64,
    pub num_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub vocab_size: u64,
    pub weight_seed: u64,
}

pub const SHAPE: Shape = Shape {
    layers: 1,
    hidden_size: 64,
    num_heads: 4,
    num_kv_heads: 2,
    head_dim: 16,
    vocab_size: 256,
    weight_seed: 20_240_607,
};

/// The value the checkpoint pin takes: R006 loads no checkpoint.
pub const BASE_CHECKPOINT: &str = "none: seeded random";

/// The model name the chain binds (the backend's `MODEL_NAME`; the same for any layer count).
const MODEL_NAME: &str = "r005-tiny";

/// The fields of `tiny_model_config`, for [`canonical_digest`].
pub fn shape_fields() -> Vec<(&'static str, u64)> {
    vec![
        ("layers", SHAPE.layers),
        ("hidden_size", SHAPE.hidden_size),
        ("num_heads", SHAPE.num_heads),
        ("num_kv_heads", SHAPE.num_kv_heads),
        ("head_dim", SHAPE.head_dim),
        ("vocab_size", SHAPE.vocab_size),
        ("weight_seed", SHAPE.weight_seed),
    ]
}

/// The stack configuration of `layers` layers (R006 uses 1; the tests use 2 as a control) for a
/// prompt of up to `max_prompt_tokens` tokens: `max_position_embeddings` is the prompt limit
/// plus the generated tokens, as in the backend.
pub fn stack_config_with(layers: usize, max_prompt_tokens: usize) -> Qwen2StackConfig {
    Qwen2StackConfig {
        layer: Qwen2LayerConfig {
            hidden_size: SHAPE.hidden_size as usize,
            intermediate_size: shared::INTERMEDIATE_SIZE,
            num_attention_heads: SHAPE.num_heads as usize,
            num_key_value_heads: SHAPE.num_kv_heads as usize,
            max_position_embeddings: max_prompt_tokens + shared::MAX_NEW_TOKENS,
            rope_theta: shared::ROPE_THETA,
            rope_scaling: Qwen2RopeScaling::None,
            rms_norm_eps: shared::RMS_NORM_EPS,
            hidden_act: Qwen2Activation::Silu,
            sliding_window: None,
        },
        num_layers: layers,
        vocab_size: SHAPE.vocab_size as usize,
        key_encoding: KvKeyEncoding::PreRope,
        qk_norm: Qwen2QkNorm::Disabled,
    }
}

/// The one-layer stack configuration of the pinned shape with the shared prompt limit.
pub fn stack_config() -> Qwen2StackConfig {
    stack_config_with(SHAPE.layers as usize, shared::MAX_PROMPT_TOKENS)
}

/// The independent one-layer stack (dense, seeded random weights).
pub fn reference_stack() -> Res<Qwen2PagedStack> {
    Qwen2PagedStack::reference(stack_config(), SHAPE.weight_seed)
        .map_err(fail("the independent one-layer stack"))
}

fn chain_identity() -> KvChainIdentity {
    let tokenizer = ByteTokenizer::new();
    KvChainIdentity {
        base_model: MODEL_NAME.to_string(),
        generation: 1,
        build: "ptr-bench-r006-pin-check".to_string(),
        tokenizer: format!(
            "{}@{}",
            tokenizer.identity().name(),
            tokenizer.identity().version()
        ),
    }
}

/// The chain id of layer 0 and the final chain id of `stack` under the plain rotary regime.
fn chain_ids(stack: &Qwen2PagedStack) -> Res<([u8; 32], [u8; 32])> {
    let regime = stack
        .config()
        .layer
        .rope_regime(1)
        .map_err(fail("the plain regime"))?;
    let chains = derive_kv_layer_chains_in_regime(stack, &chain_identity(), &regime)
        .map_err(fail("the chain"))?;
    let first = chains.layer(0).ok_or("the chain has no layer 0")?;
    Ok((first, chains.final_id()))
}

/// The backend-free checks of the one-layer pins, recorded in `checks`:
///
/// * the canonical digest of [`shape_fields`] equals `pinned_digest`;
/// * the tokenizer is the byte tokenizer with vocabulary 256;
/// * the stack configuration built here has the pinned shape and one layer;
/// * for one layer the final chain id equals the layer-0 chain id (the secondary index is keyed
///   on layer 0, which for one layer is the whole model);
/// * two independently built one-layer stacks reproduce each other's cold run (generated tokens
///   and logits digest) bit for bit.
///
/// The comparison with a backend's own chain id and cold run is the harness's.
pub fn verify_independent(checks: &mut PinChecks, pinned_digest: &str) -> Res<()> {
    let canonical = canonical_digest(&shape_fields());
    checks.check(
        "tiny_model_config_digest",
        canonical == pinned_digest,
        Json::str(canonical),
    );
    let tokenizer = ByteTokenizer::new();
    let identity = tokenizer.identity();
    checks.check(
        "tokenizer",
        identity.name() == "ptr.byte"
            && identity.version() == "1"
            && SHAPE.vocab_size == 256
            && params::TOKENIZER_PIN.starts_with("ByteTokenizer, vocabulary 256"),
        Json::str(format!("{}@{}", identity.name(), identity.version())),
    );
    let config = stack_config();
    let shape_ok = config.num_layers as u64 == SHAPE.layers
        && SHAPE.layers == 1
        && config.layer.hidden_size as u64 == SHAPE.hidden_size
        && config.layer.num_attention_heads as u64 == SHAPE.num_heads
        && config.layer.num_key_value_heads as u64 == SHAPE.num_kv_heads
        && (config.layer.hidden_size / config.layer.num_attention_heads) as u64 == SHAPE.head_dim
        && config.vocab_size as u64 == SHAPE.vocab_size
        && config.key_encoding == KvKeyEncoding::PreRope;
    checks.check(
        "shape",
        shape_ok,
        Json::str("one layer, hidden, heads, kv heads, head_dim, vocab, PreRope"),
    );
    let stack = reference_stack()?;
    let (first, last) = chain_ids(&stack)?;
    checks.check(
        "single_layer_chain",
        first == last,
        Json::str(format!(
            "sha256:{}",
            crate::experiments::layered_kv::corpus::hex(&first)
        )),
    );
    let tokens = crate::experiments::layered_kv::corpus::sequence(
        SHAPE.weight_seed,
        9_999,
        3 * crate::experiments::layered_kv::corpus::CHUNK_TOKENS + 5,
    );
    let first_run = shared::independent_cold_run(&stack, None, &tokens)?;
    let second_run = shared::independent_cold_run(&reference_stack()?, None, &tokens)?;
    checks.check(
        "independent_cold_run_reproduces",
        first_run == second_run,
        Json::obj([
            ("generated", Json::Int(first_run.0.len() as i64)),
            (
                "logits_digest",
                Json::str(format!(
                    "sha256:{}",
                    crate::experiments::layered_kv::corpus::hex(&first_run.1)
                )),
            ),
        ]),
    );
    Ok(())
}

/// The pinned values the run echoes under `model_pins.pinned`.
pub fn pinned_json() -> Json {
    Json::obj([
        (
            "tiny_model_config_digest",
            Json::str(params::TINY_MODEL_CONFIG_DIGEST),
        ),
        ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
        ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
        ("weights", Json::str(params::WEIGHTS)),
        ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
        ("layers", Json::Int(params::TEST_MODEL_LAYERS as i64)),
        ("base_checkpoint", Json::str(BASE_CHECKPOINT)),
    ])
}

/// The values the config leaves open, as `model_pins.chosen` records them.
pub fn chosen_json(max_prompt_tokens: usize) -> Json {
    Json::obj([
        (
            "intermediate_size",
            Json::Int(shared::INTERMEDIATE_SIZE as i64),
        ),
        ("rope_theta", Json::str("10000.0")),
        ("rms_norm_eps", Json::str("1e-6")),
        ("qk_norm", Json::str("disabled")),
        ("key_encoding", Json::str("PreRope")),
        ("max_new_tokens", Json::Int(shared::MAX_NEW_TOKENS as i64)),
        ("max_prompt_tokens", Json::Int(max_prompt_tokens as i64)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit::{MetricList, Tally};

    fn tally() -> Tally {
        Tally::new(MetricList {
            names: &[],
            grouped: &[],
        })
    }

    #[test]
    fn the_one_layer_digest_is_the_configs_and_differs_from_the_four_layer_pin() {
        assert_eq!(
            canonical_digest(&shape_fields()),
            "sha256:a77e017c8ee3b56bb06a0b22512aae3a70a1b65d4ae283bf2d8bb02353029c2e"
        );
        assert_eq!(
            canonical_digest(&shape_fields()),
            params::TINY_MODEL_CONFIG_DIGEST
        );
        assert_ne!(
            canonical_digest(&shape_fields()),
            canonical_digest(&shared::shape_fields()),
            "the four-layer digest of R005 to R010 is another model"
        );
    }

    #[test]
    fn the_independent_checks_hold_for_the_pinned_digest_and_fail_for_another() {
        let mut checks = PinChecks::default();
        verify_independent(&mut checks, params::TINY_MODEL_CONFIG_DIGEST).unwrap();
        let mut counter = tally();
        let pins = checks.finish(&mut counter, pinned_json(), chosen_json(256), None);
        assert_eq!(
            counter.get("model_pin_mismatches"),
            0,
            "{}",
            pins.json.render()
        );

        let mut checks = PinChecks::default();
        verify_independent(
            &mut checks,
            "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a",
        )
        .unwrap();
        let mut counter = tally();
        checks.finish(&mut counter, Json::Null, Json::Null, None);
        assert_eq!(
            counter.get("model_pin_mismatches"),
            1,
            "the four-layer digest is a mismatch for the one-layer model"
        );
    }

    #[test]
    fn the_stack_has_one_layer_and_stores_pre_rope_keys() {
        let stack = reference_stack().unwrap();
        assert_eq!(stack.config().num_layers, 1);
        assert_eq!(stack.config().key_encoding, KvKeyEncoding::PreRope);
        assert_eq!(stack_config_with(4, 256).num_layers, 4);
        assert_eq!(
            stack_config_with(1, 4352).layer.max_position_embeddings,
            4352 + shared::MAX_NEW_TOKENS
        );
    }

    #[test]
    fn in_one_layer_the_stored_rows_of_a_token_are_a_pure_function_of_the_token() {
        // The premise of R006 and the reason staleness is structural: layer 0 sees only the
        // token, so a token's stored (pre-RoPE) K and V rows are the same bits wherever the
        // token sits and whatever surrounds it.
        let tokens = [5u32, 9, 5, 200, 5, 9];
        let rows =
            shared::independent_prefill_rows(&reference_stack().unwrap(), None, &tokens).unwrap();
        let width = rows.kv_width;
        let (keys, values) = &rows.layers[0];
        let row = |data: &Vec<f32>, index: usize| -> Vec<u32> {
            data[index * width..(index + 1) * width]
                .iter()
                .map(|value| value.to_bits())
                .collect()
        };
        for other in [2, 4] {
            assert_eq!(
                row(keys, 0),
                row(keys, other),
                "K row of token 5 at {other}"
            );
            assert_eq!(
                row(values, 0),
                row(values, other),
                "V row of token 5 at {other}"
            );
        }
        assert_eq!(row(keys, 1), row(keys, 5));
        assert_ne!(row(keys, 0), row(keys, 1), "different tokens differ");
    }

    #[test]
    fn in_two_layers_the_rows_above_layer_zero_depend_on_the_context() {
        // The control that shows the premise is not trivially true: layer 1 attends over the
        // context, so the same token at different positions gets different rows. This is why
        // the multi-layer case is Approximate (M016) and R006 is one layer.
        let tokens = [5u32, 9, 5, 200, 5, 9];
        let stack =
            Qwen2PagedStack::reference(stack_config_with(2, 256), SHAPE.weight_seed).unwrap();
        let rows = shared::independent_prefill_rows(&stack, None, &tokens).unwrap();
        let width = rows.kv_width;
        let bits = |data: &Vec<f32>, index: usize| -> Vec<u32> {
            data[index * width..(index + 1) * width]
                .iter()
                .map(|value| value.to_bits())
                .collect()
        };
        let (layer0, layer1) = (&rows.layers[0].0, &rows.layers[1].0);
        assert_eq!(bits(layer0, 0), bits(layer0, 2), "layer 0 is still pure");
        assert_ne!(bits(layer1, 0), bits(layer1, 2), "layer 1 sees the context");
    }

    #[test]
    fn the_final_chain_id_of_one_layer_is_the_layer_zero_chain_and_two_layers_differ() {
        let one = reference_stack().unwrap();
        let (first, last) = chain_ids(&one).unwrap();
        assert_eq!(first, last);
        let two = Qwen2PagedStack::reference(stack_config_with(2, 256), SHAPE.weight_seed).unwrap();
        let (first_two, last_two) = chain_ids(&two).unwrap();
        assert_eq!(
            first_two, first,
            "layer 0 of the two-layer model is the same chain"
        );
        assert_ne!(last_two, first_two);
    }
}
