//! The model pins and the producer environment of a run.
//!
//! The backend under test hard-codes its tiny model (`stack_config` in
//! `crates/ptr-runtime/src/kv_infer.rs`). Nothing in its public API reports the configuration
//! it runs, so the pins are checked from outside:
//!
//! 1. the digest of `tiny_model_config` (the rule in `config.toml`) equals the pinned digest;
//! 2. a stack built here from the pinned shape and the chosen values derives the same KV chain
//!    id as the backend (the chain hashes every weight, the shape, the rotary parameters, the
//!    key encoding, the tokenizer and the build identity);
//! 3. a cold run on that stack, driven here chunk by chunk and token by token, produces the
//!    same generated tokens and the same logits digest as the backend's cold run, and so does
//!    a cold run under the experiment's own binding (adapter or MoE), which the experiment's
//!    `verify` adds.
//!
//! A difference in any of them is a model-pin mismatch.

use super::corpus::{hex, raw_request, sequence, CHUNK_TOKENS};
use super::emit::{Json, Tally};
use super::lab::{self, fail, Res};
use ptr_ledger::integrity::sha256;
use ptr_model_api::{ByteTokenizer, Tokenizer};
use ptr_pods::{
    derive_kv_layer_chains_in_regime, InMemoryPagedKvBackend, KvChainIdentity, KvKeyEncoding,
    KvPageBinding, KvPageSize, KvTensorBackend, KvTensorDType, PagedKvTensorBackend,
    Qwen2Activation, Qwen2Adapter, Qwen2Binding, Qwen2LayerConfig, Qwen2PagedStack, Qwen2QkNorm,
    Qwen2RopeScaling, Qwen2StackConfig, Qwen2StackInput,
};
use ptr_runtime::{ReferenceKvReuseBackend, StackBinding};
use ptr_types::{AdapterVersion, DeviceId, Generation, ModelVersion, PrincipalId};

/// The tiny model of both experiments (`tiny_model_config`, without the MoE keys).
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
    layers: 4,
    hidden_size: 64,
    num_heads: 4,
    num_kv_heads: 2,
    head_dim: 16,
    vocab_size: 256,
    weight_seed: 20_240_607,
};

/// Values the config leaves open, inherited from R005 (the backend hard-codes them).
pub const INTERMEDIATE_SIZE: usize = 128;
pub const ROPE_THETA: f32 = 10_000.0;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const MAX_NEW_TOKENS: usize = 8;
pub const MAX_PROMPT_TOKENS: usize = 256;

/// The model name the backend binds into its chain (`MODEL_NAME` in `kv_infer.rs`).
const BACKEND_MODEL_NAME: &str = "r005-tiny";
/// The numerics label the backend binds into its build identity (`NUMERICS`).
const BACKEND_NUMERICS: &str = "cpu-f32-sequential-reference";
/// The domain of the logits digest (`LOGITS_DOMAIN`).
const LOGITS_DOMAIN: &[u8] = b"ptr.kv.reference-logits.v1\0";

/// The tiny-model stack configuration the pins and the chosen values describe.
pub fn stack_config() -> Qwen2StackConfig {
    Qwen2StackConfig {
        layer: Qwen2LayerConfig {
            hidden_size: SHAPE.hidden_size as usize,
            intermediate_size: INTERMEDIATE_SIZE,
            num_attention_heads: SHAPE.num_heads as usize,
            num_key_value_heads: SHAPE.num_kv_heads as usize,
            max_position_embeddings: MAX_PROMPT_TOKENS + MAX_NEW_TOKENS,
            rope_theta: ROPE_THETA,
            rope_scaling: Qwen2RopeScaling::None,
            rms_norm_eps: RMS_NORM_EPS,
            hidden_act: Qwen2Activation::Silu,
            sliding_window: None,
        },
        num_layers: SHAPE.layers as usize,
        vocab_size: SHAPE.vocab_size as usize,
        key_encoding: KvKeyEncoding::PreRope,
        qk_norm: Qwen2QkNorm::Disabled,
    }
}

/// The independent stack of the pinned shape (dense).
pub fn reference_stack() -> Res<Qwen2PagedStack> {
    Qwen2PagedStack::reference(stack_config(), SHAPE.weight_seed)
        .map_err(fail("the independent stack"))
}

/// `sha256:` of the canonical JSON (sorted keys, no spaces, integer values) of `fields`: the
/// rule of `tiny_model_config_digest_rule`.
pub fn canonical_digest(fields: &[(&str, u64)]) -> String {
    let mut sorted: Vec<&(&str, u64)> = fields.iter().collect();
    sorted.sort_by_key(|(key, _)| *key);
    let body: Vec<String> = sorted
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect();
    format!(
        "sha256:{}",
        hex(&sha256(format!("{{{}}}", body.join(",")).as_bytes()))
    )
}

/// The fields of `tiny_model_config` of a dense model.
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

/// The build identity the backend reports to its chain, as this crate reconstructs it:
/// verified against the backend's chain id before it is trusted.
fn reconstructed_build() -> String {
    format!(
        "ptr-runtime@{}|{BACKEND_NUMERICS}|{}-{}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

fn chain_identity(build: &str) -> KvChainIdentity {
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

fn fresh_cache(
    paged: &InMemoryPagedKvBackend,
    stack: &Qwen2PagedStack,
) -> Res<<InMemoryPagedKvBackend as ptr_pods::KvTensorBackend>::Cache> {
    let schema = stack
        .kv_schema(
            ModelVersion::from(BACKEND_MODEL_NAME),
            AdapterVersion::from("base"),
            DeviceId::from("cpu"),
        )
        .map_err(fail("the schema"))?;
    let binding = KvPageBinding {
        model: schema.model.clone(),
        adapter: schema.adapter.clone(),
        generation: Generation(1),
        execution_manifest: [1; 32],
        principal: PrincipalId::from(lab::PRINCIPAL),
        device: schema.device.clone(),
        dtype: KvTensorDType::F32,
        page_tokens: KvPageSize::new(CHUNK_TOKENS as u16).map_err(fail("the page size"))?,
    };
    paged
        .allocate_bound(schema, MAX_PROMPT_TOKENS + MAX_NEW_TOKENS, binding)
        .map_err(fail("a cache"))
}

/// The generated tokens and the logits digest of a cold run on `stack` under `adapter`,
/// driven by hand chunk by chunk and token by token.
pub fn independent_cold_run(
    stack: &Qwen2PagedStack,
    adapter: Option<&Qwen2Adapter>,
    tokens: &[u32],
) -> Res<(Vec<u32>, [u8; 32])> {
    let paged = InMemoryPagedKvBackend::new(
        KvPageSize::new(CHUNK_TOKENS as u16).map_err(fail("the page size"))?,
    );
    let mut cache = fresh_cache(&paged, stack)?;
    let regime = stack
        .admit_rope_regime(tokens.len(), MAX_NEW_TOKENS)
        .map_err(fail("the regime"))?;
    let binding = Qwen2Binding { adapter };
    let mut last = Vec::new();
    for chunk in tokens.chunks(CHUNK_TOKENS) {
        let output = stack
            .prefill_bound_in_regime(
                &paged,
                &mut cache,
                Qwen2StackInput {
                    tokens: chunk.to_vec(),
                },
                &regime,
                &binding,
            )
            .map_err(fail("a prefill chunk"))?;
        last = output.logits.into_iter().next_back().unwrap_or_default();
    }
    let mut material = LOGITS_DOMAIN.to_vec();
    let mut generated = Vec::new();
    for step in 0..MAX_NEW_TOKENS {
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
        if step + 1 < MAX_NEW_TOKENS {
            let output = stack
                .decode_bound_in_regime(
                    &paged,
                    &mut cache,
                    Qwen2StackInput {
                        tokens: vec![best as u32],
                    },
                    &regime,
                    &binding,
                )
                .map_err(fail("a decode step"))?;
            last = output.logits.into_iter().next().unwrap_or_default();
        }
    }
    Ok((generated, sha256(&material)))
}

/// The K and V rows a prefill of `tokens` on `stack` under `adapter` leaves in the cache,
/// per layer, as the exact `f32` values: the independent reference the sharing checks
/// compare bit patterns against.
pub struct PrefillRows {
    /// `layers[l]` is `(keys, values)`, each `tokens.len() * kv_width` values.
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub kv_width: usize,
}

pub fn independent_prefill_rows(
    stack: &Qwen2PagedStack,
    adapter: Option<&Qwen2Adapter>,
    tokens: &[u32],
) -> Res<PrefillRows> {
    let paged = InMemoryPagedKvBackend::new(
        KvPageSize::new(CHUNK_TOKENS as u16).map_err(fail("the page size"))?,
    );
    let mut cache = fresh_cache(&paged, stack)?;
    let regime = stack
        .admit_rope_regime(tokens.len(), MAX_NEW_TOKENS)
        .map_err(fail("the regime"))?;
    let binding = Qwen2Binding { adapter };
    for chunk in tokens.chunks(CHUNK_TOKENS) {
        stack
            .prefill_bound_in_regime(
                &paged,
                &mut cache,
                Qwen2StackInput {
                    tokens: chunk.to_vec(),
                },
                &regime,
                &binding,
            )
            .map_err(fail("a prefill chunk"))?;
    }
    let state = paged.snapshot(&cache).map_err(fail("the snapshot"))?;
    Ok(PrefillRows {
        layers: state
            .layers
            .iter()
            .map(|layer| (layer.keys.values.clone(), layer.values.values.clone()))
            .collect(),
        kv_width: (SHAPE.num_kv_heads * SHAPE.head_dim) as usize,
    })
}

/// How often each expert of each MoE layer is selected when `stack` prefills `tokens`, and
/// the logits of the last prompt token: the routing trace the revoked expert is chosen from.
pub fn routing_trace(stack: &Qwen2PagedStack, tokens: &[u32]) -> Res<ptr_pods::RoutingCounts> {
    let paged = InMemoryPagedKvBackend::new(
        KvPageSize::new(CHUNK_TOKENS as u16).map_err(fail("the page size"))?,
    );
    let mut cache = fresh_cache(&paged, stack)?;
    let regime = stack
        .admit_rope_regime(tokens.len(), MAX_NEW_TOKENS)
        .map_err(fail("the regime"))?;
    let (_, counts) = stack
        .prefill_traced_in_regime(
            &paged,
            &mut cache,
            Qwen2StackInput {
                tokens: tokens.to_vec(),
            },
            &regime,
            &Qwen2Binding::default(),
        )
        .map_err(fail("the traced prefill"))?;
    Ok(counts)
}

/// The last-token logits of each request of `batches` prefilled **co-batched** on `stack` in
/// the given `order`, and the total of selections the capacity rule dropped.
pub fn cobatched_last_logits(
    stack: &Qwen2PagedStack,
    batches: &[Vec<u32>],
    order: &[usize],
) -> Res<(Vec<Vec<f32>>, u64)> {
    let paged = InMemoryPagedKvBackend::new(
        KvPageSize::new(CHUNK_TOKENS as u16).map_err(fail("the page size"))?,
    );
    let mut caches = Vec::new();
    for _ in batches {
        caches.push(fresh_cache(&paged, stack)?);
    }
    let longest = batches.iter().map(Vec::len).max().unwrap_or(1);
    let regime = stack
        .admit_rope_regime(longest, MAX_NEW_TOKENS)
        .map_err(fail("the regime"))?;
    let output = stack
        .prefill_cobatched_in_regime(
            &paged,
            &mut caches,
            batches
                .iter()
                .map(|tokens| Qwen2StackInput {
                    tokens: tokens.clone(),
                })
                .collect(),
            order,
            &regime,
            &Qwen2Binding::default(),
        )
        .map_err(fail("the co-batched prefill"))?;
    Ok((
        output
            .outputs
            .iter()
            .map(|out| out.logits.last().cloned().unwrap_or_default())
            .collect(),
        output.routing.total_dropped(),
    ))
}

/// The checks of a model-pin verification, one mismatch per failed one.
#[derive(Default)]
pub struct PinChecks {
    mismatches: i64,
    items: Vec<(String, Json)>,
}

impl PinChecks {
    pub fn check(&mut self, name: &str, ok: bool, detail: Json) {
        if !ok {
            self.mismatches += 1;
        }
        self.items.push((
            name.to_string(),
            Json::obj([("holds", Json::Bool(ok)), ("detail", detail)]),
        ));
    }

    /// Record the mismatch count in `tally` and assemble the `model_pins` object.
    pub fn finish(
        self,
        tally: &mut Tally,
        pinned: Json,
        chosen: Json,
        build: Option<String>,
    ) -> Pins {
        tally.add("model_pin_mismatches", self.mismatches);
        let json = Json::Obj(
            self.items
                .into_iter()
                .chain([
                    ("pinned".to_string(), pinned),
                    ("chosen".to_string(), chosen),
                ])
                .collect(),
        );
        Pins { json, build }
    }
}

/// What `verify` found.
pub struct Pins {
    pub json: Json,
    /// The build identity string of the backend, when it could be verified.
    pub build: Option<String>,
}

/// The checks both experiments share: the digest, the tokenizer, the shape, the chain id and
/// the cold run of the dense model. Returns the verified build identity.
pub fn base_checks(
    checks: &mut PinChecks,
    backend: &ReferenceKvReuseBackend,
    canonical: &str,
    pinned_digest: &str,
    tokenizer_pin: &str,
) -> Res<Option<String>> {
    checks.check(
        "tiny_model_config_digest",
        canonical == pinned_digest,
        Json::str(canonical),
    );
    let tokenizer = backend.tokenizer().identity();
    checks.check(
        "tokenizer",
        tokenizer.name() == "ptr.byte"
            && tokenizer.version() == "1"
            && SHAPE.vocab_size == 256
            && tokenizer_pin.starts_with("ByteTokenizer, vocabulary 256"),
        Json::str(lab::tokenizer_text(backend)),
    );
    let config = stack_config();
    let stack = reference_stack()?;
    let shape_ok = config.num_layers as u64 == SHAPE.layers
        && config.layer.hidden_size as u64 == SHAPE.hidden_size
        && config.layer.num_attention_heads as u64 == SHAPE.num_heads
        && config.layer.num_key_value_heads as u64 == SHAPE.num_kv_heads
        && (config.layer.hidden_size / config.layer.num_attention_heads) as u64 == SHAPE.head_dim
        && config.vocab_size as u64 == SHAPE.vocab_size
        && backend.stack_config().num_layers == config.num_layers
        && backend.stack_config().layer.hidden_size == config.layer.hidden_size
        && backend.stack_config().layer.intermediate_size == config.layer.intermediate_size;
    checks.check(
        "shape",
        shape_ok,
        Json::str("layers, hidden, heads, kv heads, head_dim, vocab, intermediate"),
    );

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
    checks.check(
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

    let tokens: Vec<u32> = sequence(SHAPE.weight_seed, 9_999, 3 * CHUNK_TOKENS + 5);
    let observed = lab::cold(
        backend,
        &raw_request("pin-probe", &tokens),
        &StackBinding::base(),
    )?;
    let (generated, logits) = independent_cold_run(&stack, None, &tokens)?;
    let observed_tokens: Vec<u32> = observed.generated.iter().map(|token| token.0).collect();
    checks.check(
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
    Ok(chain_ok.then_some(build))
}

/// The pinned values both experiments echo.
pub fn chosen_json() -> Json {
    Json::obj([
        ("intermediate_size", Json::Int(INTERMEDIATE_SIZE as i64)),
        ("rope_theta", Json::str("10000.0")),
        ("rms_norm_eps", Json::str("1e-6")),
        ("qk_norm", Json::str("disabled")),
        ("key_encoding", Json::str("PreRope")),
        ("max_new_tokens", Json::Int(MAX_NEW_TOKENS as i64)),
        ("max_prompt_tokens", Json::Int(MAX_PROMPT_TOKENS as i64)),
    ])
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

/// The producer environment and the host facts.
pub fn environment(pins: &Pins) -> Json {
    let cpu = cpu_model();
    let build = pins.build.clone().map_or(Json::Null, Json::Str);
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
        ("backend_build_identity", build),
        ("ptr_bench_version", Json::str(env!("CARGO_PKG_VERSION"))),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_digest_is_the_r009_pin() {
        assert_eq!(
            canonical_digest(&shape_fields()),
            "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a"
        );
    }

    #[test]
    fn the_moe_digest_is_the_r010_pin() {
        let mut fields = shape_fields();
        fields.push(("num_experts", 4));
        fields.push(("top_k", 2));
        assert_eq!(
            canonical_digest(&fields),
            "sha256:fccf7bd315c89853010d486c2f1e11344f5d1e29fa0c8791eef69b5e246f84f0"
        );
    }

    #[test]
    fn the_backend_runs_the_pinned_dense_model() {
        let backend = lab::backend().unwrap();
        let mut checks = PinChecks::default();
        let build = base_checks(
            &mut checks,
            &backend,
            &canonical_digest(&shape_fields()),
            "sha256:91be3ab2b3493b3ab5dbc99a03c36a0a599b42c8d359a6443e629c2ce4ba433a",
            "ByteTokenizer, vocabulary 256 (no external revision)",
        )
        .unwrap();
        assert!(
            build.is_some(),
            "the build identity reproduces the chain id"
        );
        let mut tally = Tally::new(super::super::emit::MetricList {
            names: &[],
            grouped: &[],
        });
        let pins = checks.finish(&mut tally, Json::Null, Json::Null, build);
        assert_eq!(
            tally.get("model_pin_mismatches"),
            0,
            "{}",
            pins.json.render()
        );
    }

    #[test]
    fn another_epsilon_is_another_chain() {
        let backend = lab::backend().unwrap();
        let mut config = stack_config();
        config.layer.rms_norm_eps *= 2.0;
        let stack = Qwen2PagedStack::reference(config, SHAPE.weight_seed).unwrap();
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
}
