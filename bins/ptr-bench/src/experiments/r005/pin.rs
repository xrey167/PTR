//! The model pins and the producer environment of a run.
//!
//! The backend under test hard-codes its tiny model (`stack_config` in
//! `crates/ptr-runtime/src/kv_infer.rs`). Nothing in its public API reports the
//! configuration it runs, so the pins are checked from outside, three ways:
//!
//! 1. the digest of `tiny_model_config` (the rule in `config.toml`) equals the pinned digest;
//! 2. a stack built here from the pinned shape and the chosen values derives the same KV chain
//!    id as the backend (the chain hashes every weight, the shape, the rotary parameters, the
//!    key encoding, the tokenizer and the build identity), and
//! 3. a cold run on that stack, driven here chunk by chunk and token by token, produces the
//!    same generated tokens and the same logits digest as the backend's cold run.
//!
//! A difference in any of them is a model-pin mismatch.

use super::lab::{self, fail, Res};
use super::metrics::{Json, Tally};
use super::params;
use super::workload::{hex, raw_request, sequence};
use ptr_ledger::integrity::sha256;
use ptr_model_api::{ByteTokenizer, Tokenizer};
use ptr_pods::{
    derive_kv_layer_chains_in_regime, InMemoryPagedKvBackend, KvChainIdentity, KvKeyEncoding,
    KvPageBinding, KvPageSize, KvTensorDType, PagedKvTensorBackend, Qwen2Activation,
    Qwen2LayerConfig, Qwen2PagedStack, Qwen2QkNorm, Qwen2RopeScaling, Qwen2StackConfig,
    Qwen2StackInput,
};
use ptr_runtime::ReferenceKvReuseBackend;
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

/// The model name the backend binds into its chain (`MODEL_NAME` in `kv_infer.rs`).
const BACKEND_MODEL_NAME: &str = "r005-tiny";
/// The numerics label the backend binds into its build identity (`NUMERICS`).
const BACKEND_NUMERICS: &str = "cpu-f32-sequential-reference";
/// The domain of the logits digest (`LOGITS_DOMAIN`).
const LOGITS_DOMAIN: &[u8] = b"ptr.kv.reference-logits.v1\0";

/// The tiny-model stack configuration the pins and the chosen values describe.
pub fn stack_config(qk_norm: Qwen2QkNorm, rms_norm_eps: f32) -> Qwen2StackConfig {
    Qwen2StackConfig {
        layer: Qwen2LayerConfig {
            hidden_size: params::TINY_MODEL.hidden_size as usize,
            intermediate_size: params::INTERMEDIATE_SIZE,
            num_attention_heads: params::TINY_MODEL.num_heads as usize,
            num_key_value_heads: params::TINY_MODEL.num_kv_heads as usize,
            max_position_embeddings: params::MAX_PROMPT_TOKENS + params::MAX_NEW_TOKENS,
            rope_theta: params::ROPE_THETA,
            rope_scaling: Qwen2RopeScaling::None,
            rms_norm_eps,
            hidden_act: Qwen2Activation::Silu,
            sliding_window: None,
        },
        num_layers: params::TINY_MODEL.layers as usize,
        vocab_size: params::TINY_MODEL.vocab_size as usize,
        key_encoding: KvKeyEncoding::PreRope,
        qk_norm,
    }
}

/// `sha256:` of the canonical JSON (sorted keys, no spaces) of `tiny_model_config`.
pub fn tiny_model_digest() -> String {
    let model = params::TINY_MODEL;
    let canonical = format!(
        "{{\"head_dim\":{},\"hidden_size\":{},\"layers\":{},\"num_heads\":{},\"num_kv_heads\":{},\"vocab_size\":{},\"weight_seed\":{}}}",
        model.head_dim,
        model.hidden_size,
        model.layers,
        model.num_heads,
        model.num_kv_heads,
        model.vocab_size,
        model.weight_seed,
    );
    format!("sha256:{}", hex(&sha256(canonical.as_bytes())))
}

/// The build identity the backend reports to its chain, as this crate reconstructs it:
/// verified against the backend's chain id before it is trusted (`verify`).
fn reconstructed_build() -> String {
    format!(
        "ptr-runtime@{}|{BACKEND_NUMERICS}|{}-{}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

/// The chain identity of the pinned model with `build`.
pub fn chain_identity(build: &str) -> KvChainIdentity {
    let tokenizer = ByteTokenizer::new();
    KvChainIdentity {
        base_model: BACKEND_MODEL_NAME.to_string(),
        generation: 1,
        build: build.to_string(),
        tokenizer: format!(
            "{}@{}",
            tokenizer.identity().name(),
            tokenizer.identity().version()
        ),
    }
}

/// The logits digest and generated tokens of a cold run on `stack`, driven by hand.
fn independent_cold_run(stack: &Qwen2PagedStack, tokens: &[u32]) -> Res<(Vec<u32>, [u8; 32])> {
    let paged = InMemoryPagedKvBackend::new(
        KvPageSize::new(params::CHUNK_TOKENS as u16).map_err(fail("the page size"))?,
    );
    let schema = stack
        .kv_schema(
            ModelVersion::from("r005-tiny"),
            AdapterVersion::from("base"),
            DeviceId::from("cpu"),
        )
        .map_err(fail("the schema"))?;
    let binding = KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [1; 32],
        principal: PrincipalId::from(params::PRINCIPAL),
        device: schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::new(params::CHUNK_TOKENS as u16).map_err(fail("the page size"))?,
    };
    let mut cache = paged
        .allocate_bound(
            schema,
            params::MAX_PROMPT_TOKENS + params::MAX_NEW_TOKENS,
            binding,
        )
        .map_err(fail("a cache"))?;
    let regime = stack
        .admit_rope_regime(tokens.len(), params::MAX_NEW_TOKENS)
        .map_err(fail("the regime"))?;
    let mut last = Vec::new();
    for chunk in tokens.chunks(params::CHUNK_TOKENS) {
        let output = stack
            .prefill_in_regime(
                &paged,
                &mut cache,
                Qwen2StackInput {
                    tokens: chunk.to_vec(),
                },
                &regime,
            )
            .map_err(fail("a prefill chunk"))?;
        last = output.logits.into_iter().next_back().unwrap_or_default();
    }
    let mut material = LOGITS_DOMAIN.to_vec();
    let mut generated = Vec::new();
    for step in 0..params::MAX_NEW_TOKENS {
        for value in &last {
            material.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let mut best = 0usize;
        for (index, value) in last.iter().enumerate() {
            if *value > last[best] {
                best = index;
            }
        }
        generated.push(best as u32);
        if step + 1 < params::MAX_NEW_TOKENS {
            let output = stack
                .decode_in_regime(
                    &paged,
                    &mut cache,
                    Qwen2StackInput {
                        tokens: vec![best as u32],
                    },
                    &regime,
                )
                .map_err(fail("a decode step"))?;
            last = output.logits.into_iter().next().unwrap_or_default();
        }
    }
    Ok((generated, sha256(&material)))
}

/// What `verify` found.
pub struct Pins {
    pub json: Json,
    /// The build identity string of the backend, when it could be verified.
    pub build: Option<String>,
}

/// Check the model pins against the backend under test; one mismatch per failed check.
pub fn verify(tally: &mut Tally, backend: &ReferenceKvReuseBackend) -> Res<Pins> {
    let mut mismatches = 0i64;
    let mut checks: Vec<(&str, Json)> = Vec::new();
    let mut check = |name: &'static str, ok: bool, detail: Json| {
        if !ok {
            mismatches += 1;
        }
        checks.push((
            name,
            Json::obj([("holds", Json::Bool(ok)), ("detail", detail)]),
        ));
    };

    let digest = tiny_model_digest();
    check(
        "tiny_model_config_digest",
        digest == params::TINY_MODEL_CONFIG_DIGEST,
        Json::str(digest.clone()),
    );

    let tokenizer = backend.tokenizer().identity();
    let tokenizer_text = format!("{}@{}", tokenizer.name(), tokenizer.version());
    check(
        "tokenizer",
        tokenizer.name() == "ptr.byte"
            && tokenizer.version() == "1"
            && params::VOCAB_SIZE == 256
            && params::TOKENIZER_PIN.starts_with("ByteTokenizer, vocabulary 256"),
        Json::str(tokenizer_text),
    );

    let config = stack_config(Qwen2QkNorm::Disabled, params::RMS_NORM_EPS);
    let stack = Qwen2PagedStack::reference(config.clone(), params::TINY_MODEL.weight_seed)
        .map_err(fail("the independent stack"))?;
    let shape_ok = config.num_layers as u64 == params::TINY_MODEL.layers
        && config.layer.hidden_size as u64 == params::TINY_MODEL.hidden_size
        && config.layer.num_attention_heads as u64 == params::TINY_MODEL.num_heads
        && config.layer.num_key_value_heads as u64 == params::TINY_MODEL.num_kv_heads
        && (config.layer.hidden_size / config.layer.num_attention_heads) as u64
            == params::TINY_MODEL.head_dim
        && config.vocab_size as u64 == params::TINY_MODEL.vocab_size;
    check(
        "shape",
        shape_ok,
        Json::str("layers, hidden, heads, kv heads, head_dim, vocab"),
    );

    // The chain id of the backend: its build identity is verified by reconstructing it.
    let regime = stack
        .config()
        .layer
        .rope_regime(1)
        .map_err(fail("the plain regime"))?;
    let build = reconstructed_build();
    let derived = derive_kv_layer_chains_in_regime(&stack, &chain_identity(&build), &regime)
        .map_err(fail("the chain"))?
        .final_id();
    let chain_ok = derived == backend.chain_id().0;
    check(
        "chain_id",
        chain_ok,
        Json::obj([
            (
                "backend",
                Json::str(format!("sha256:{}", hex(&backend.chain_id().0))),
            ),
            (
                "independent",
                Json::str(format!("sha256:{}", hex(&derived))),
            ),
        ]),
    );

    // The cold run, reproduced by hand.
    let tokens: Vec<u32> = sequence(9_999, 3 * params::CHUNK_TOKENS + 5);
    let request = raw_request("pin-probe", &tokens);
    let observed = lab::cold(backend, &lab::scope(), &request)?;
    let (generated, logits) = independent_cold_run(&stack, &tokens)?;
    let observed_tokens: Vec<u32> = observed.generated.iter().map(|token| token.0).collect();
    check(
        "cold_run",
        generated == observed_tokens && logits == observed.logits_digest,
        Json::obj([
            ("generated", Json::Int(generated.len() as i64)),
            (
                "logits_digest",
                Json::str(format!("sha256:{}", hex(&logits))),
            ),
        ]),
    );
    check(
        "moe_topology",
        params::MOE_TOPOLOGY.starts_with("none: dense"),
        Json::str(params::MOE_TOPOLOGY),
    );

    let json = Json::Obj(
        checks
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .chain([
                (
                    "pinned".to_string(),
                    Json::obj([
                        (
                            "tiny_model_config_digest",
                            Json::str(params::TINY_MODEL_CONFIG_DIGEST),
                        ),
                        ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
                        ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
                        ("weights", Json::str(params::WEIGHTS)),
                        ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
                        (
                            "base_checkpoint",
                            Json::str("none: seeded random weights, no checkpoint"),
                        ),
                    ]),
                ),
                (
                    "chosen".to_string(),
                    Json::obj([
                        (
                            "intermediate_size",
                            Json::Int(params::INTERMEDIATE_SIZE as i64),
                        ),
                        ("rope_theta", Json::Float(f64::from(params::ROPE_THETA))),
                        ("rms_norm_eps", Json::Float(f64::from(params::RMS_NORM_EPS))),
                        ("qk_norm_main", Json::str(params::QK_NORM_MAIN)),
                        (
                            "qk_norm_transform_arm",
                            Json::str(params::QK_NORM_TRANSFORM_ARM),
                        ),
                        ("key_encoding", Json::str(params::KEY_ENCODING)),
                        ("max_new_tokens", Json::Int(params::MAX_NEW_TOKENS as i64)),
                        ("ntk_factor", Json::Float(f64::from(params::NTK_FACTOR))),
                        (
                            "ntk_original_max_positions",
                            Json::Int(params::NTK_ORIGINAL_MAX_POSITIONS as i64),
                        ),
                    ]),
                ),
            ])
            .collect(),
    );
    tally.add("model_pin_mismatches", mismatches);
    Ok(Pins {
        json,
        build: chain_ok.then_some(build),
    })
}

fn command_line(program: &str, argument: &str) -> Option<String> {
    let output = std::process::Command::new(program)
        .arg(argument)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
}

fn cpu_model() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    info.lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split_once(':'))
        .map(|(_, model)| model.trim().to_string())
}

/// The toolchain channel of `rust-toolchain.toml`, read at compile time.
fn pinned_toolchain() -> Option<String> {
    include_str!("../../../../../rust-toolchain.toml")
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == "channel").then(|| value.trim().trim_matches('"').to_string())
        })
}

/// The producer environment (`numerics_binding_environment_fields`) and the host facts.
pub fn environment(tally: &mut Tally, pins: &Pins) -> Json {
    let cpu = cpu_model();
    let architecture = format!(
        "cpu:{}:{}",
        std::env::consts::ARCH,
        cpu.clone().unwrap_or_else(|| "unknown".to_string())
    );
    let build = match &pins.build {
        Some(build) => Json::str(build.clone()),
        None => {
            tally.unsupported(
                "producer_environment_fields.backend_build_identity",
                "the backend does not expose its build identity and the reconstruction did not reproduce its chain id",
            );
            Json::Null
        }
    };
    let fields = Json::obj([
        ("gpu_or_cpu_architecture", Json::str(architecture)),
        ("driver_version", Json::str("not-applicable:cpu-reference")),
        (
            "cuda_runtime_version",
            Json::str("not-applicable:cpu-reference"),
        ),
        (
            "attention_gemm_kernel_library",
            Json::str(
                "ptr-pods reference f32 kernels, sequential reduction order, no external library",
            ),
        ),
        ("backend_build_identity", build),
    ]);
    tally.set_json("producer_environment_fields", fields);
    Json::obj([
        ("os", Json::str(std::env::consts::OS)),
        ("arch", Json::str(std::env::consts::ARCH)),
        ("cpu_model", cpu.map_or(Json::Null, Json::Str)),
        (
            "logical_cpus",
            std::thread::available_parallelism()
                .map_or(Json::Null, |count| Json::Int(count.get() as i64)),
        ),
        (
            "rustc",
            command_line("rustc", "--version").map_or(Json::Null, Json::Str),
        ),
        (
            "toolchain_pin",
            pinned_toolchain().map_or(Json::Null, Json::Str),
        ),
        (
            "profile",
            Json::str(if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }),
        ),
        ("ptr_bench_version", Json::str(env!("CARGO_PKG_VERSION"))),
    ])
}

/// The build identity a different environment would report: one build-identity field changed.
pub fn changed_build(build: &str) -> String {
    format!("{build}|driver-changed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tiny_model_digest_is_the_pinned_one() {
        assert_eq!(tiny_model_digest(), params::TINY_MODEL_CONFIG_DIGEST);
    }

    #[test]
    fn the_backend_runs_the_pinned_model() {
        let backend = lab::main_backend().unwrap();
        let mut tally = Tally::default();
        let pins = verify(&mut tally, &backend).unwrap();
        assert_eq!(
            tally.get("model_pin_mismatches"),
            0,
            "{}",
            pins.json.render()
        );
        assert!(
            pins.build.is_some(),
            "the build identity reproduces the chain id"
        );
        assert_eq!(tally.get("model_pin_mismatches"), 0);
    }

    #[test]
    fn a_perturbed_choice_is_a_pin_mismatch() {
        // The chain of a stack with another epsilon is not the backend's.
        let backend = lab::main_backend().unwrap();
        let config = stack_config(Qwen2QkNorm::Disabled, params::RMS_NORM_EPS * 2.0);
        let stack = Qwen2PagedStack::reference(config, params::TINY_MODEL.weight_seed).unwrap();
        let regime = stack.config().layer.rope_regime(1).unwrap();
        let chain = derive_kv_layer_chains_in_regime(
            &stack,
            &chain_identity(&reconstructed_build()),
            &regime,
        )
        .unwrap()
        .final_id();
        assert_ne!(chain, backend.chain_id().0);
    }

    #[test]
    fn a_changed_build_field_is_another_build() {
        let build = reconstructed_build();
        assert_ne!(changed_build(&build), build);
    }

    #[test]
    fn the_environment_names_the_pinned_fields_and_records_the_toolchain() {
        let mut tally = Tally::default();
        let pins = Pins {
            json: Json::Null,
            build: Some("b".to_string()),
        };
        let environment = environment(&mut tally, &pins).render();
        assert!(environment.contains("\"arch\""));
        let fields = tally_fields(&tally);
        for field in params::NUMERICS_BINDING_ENVIRONMENT_FIELDS {
            assert!(fields.contains(field), "{field}");
        }
    }

    fn tally_fields(tally: &Tally) -> String {
        let (metrics, _) = tally.metrics();
        metrics
            .into_iter()
            .find(|(name, _)| name == "producer_environment_fields")
            .map(|(_, value)| value.render())
            .unwrap_or_default()
    }
}
