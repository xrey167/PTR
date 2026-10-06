//! Experiment R009, layer-shared LoRA KV, as a harness: `ptr-bench layer-shared-lora-kv
//! <seed>` runs every configuration, arm and matrix cell of the experiment's `config.toml`
//! once for one seed and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`; a run at a reduced scale (the smoke test) is no evidence. The code
//! under test is the CPU reference path: `ReferenceKvReuseBackend` and its bound `KvSession`s
//! over the tiny seeded Qwen2 stack with seeded LoRA and aLoRA adapters, `PtrRuntime` as the
//! ledger-anchored liveness oracle.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names
//! (a number, an object, a digest string, or `null` with its reason under `unsupported`),
//! the procedure parameters, the model pins, the producer environment, the corpus selection
//! digest, the per-configuration and per-cell details, the two judgements (`coverage` and
//! `falsifiers`), and the counters that belong to no metric (`internal`). The exit status is 1
//! when a coverage prerequisite is unmet, a falsifier is tripped or a metric has no value and
//! no declared reason, and 2 when the harness itself cannot run a case.

pub mod arms;
pub mod metrics;
pub mod params;

use super::layered_kv::emit::{self, count, judge, Check, Json, Status, Tally};
use super::layered_kv::lab::{self, fail, Ids, Res};
use super::layered_kv::pins::{self, PinChecks};
use super::layered_kv::{corpus, Scale};
use params::CONFIGS;
use ptr_runtime::StackBinding;
use std::time::Instant;

/// Counters whose natural value is zero (violations, mismatches): a run in which none happens
/// still prints a zero. A counter of attempts or cases is never seeded, so a case that never
/// ran is a gap or an unmet prerequisite, not a silent zero.
const ZERO_BASED: [&str; 14] = [
    "bitwise_mismatch_count",
    "adapter_logits_bitwise_mismatch_count",
    "base_pages_dropped_on_revoke",
    "revoked_adapter_entry_hits",
    "cross_adapter_hits",
    "old_generation_hits",
    "revoked_lineage_entry_hits",
    "stale_hits",
    "outputs_after_adapter_revocation",
    "outputs_after_input_invalidation",
    "adapter_restart_output_mismatches",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "model_pin_mismatches",
];

/// Internal violation counters the judgements read.
const ZERO_BASED_INTERNAL: [&str; 8] = [
    "per_layer_fraction_mismatches",
    "prediction_structure_mismatches",
    "report_physical_mismatches",
    "adoption_mismatches",
    "post_invocation_rows_under_base_key",
    "restart_resolution_mismatches",
    "matrix_selectivity_mismatches",
    "internal_entry_accounting_mismatches",
];

/// The seeds the manifest declares, read from its `seeds = [...]` line.
fn declared_seeds() -> Vec<u64> {
    params::EXPERIMENT_TOML
        .lines()
        .find(|line| line.trim_start().starts_with("seeds"))
        .and_then(|line| line.split_once('['))
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inner, _)| {
            inner
                .split(',')
                .filter_map(|seed| seed.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

fn selection(tally: &mut Tally) -> Json {
    let computed = arms::selection_digest();
    let matches = computed == params::SELECTION_DIGEST;
    tally.add("selection_digest_mismatches", i64::from(!matches));
    Json::obj([
        ("corpus", Json::str(params::CORPUS)),
        ("rule", Json::str(params::CORPUS_SELECTION_RULE)),
        ("generator_seed", Json::Int(params::CORPUS_GENERATOR_SEED as i64)),
        ("domain", Json::str(params::SELECTION_DOMAIN)),
        ("computed", Json::str(computed)),
        ("pinned", Json::str(params::SELECTION_DIGEST)),
        ("matches", Json::Bool(matches)),
        (
            "sequences",
            Json::Int(arms::selection_sequences().len() as i64),
        ),
    ])
}

fn ints(values: &[u64]) -> Json {
    Json::ints(values.iter().map(|value| *value as i64))
}

/// The procedure parameters of the run, as `procedure_parameters_recorded_per_cell` lists
/// them (each cell and configuration also carries what it observed, and a mismatch is
/// counted there).
fn procedure_json(seed: u64, scale: Scale) -> Json {
    let seeds = declared_seeds();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        ("seeds", ints(&seeds)),
        ("adapter_types_run", Json::strs(&params::ADAPTER_TYPES_RUN)),
        ("qkv_lora_first_layers", ints(&params::QKV_LORA_FIRST_LAYERS)),
        ("q_only_lora_first_layers", ints(&params::Q_ONLY_LORA_FIRST_LAYERS)),
        ("mlp_only_lora_first_layers", ints(&params::MLP_ONLY_LORA_FIRST_LAYERS)),
        (
            "alora_invocation_position_tokens",
            Json::Int(params::ALORA_INVOCATION_POSITION_TOKENS as i64),
        ),
        ("prompt_tokens", Json::Int(params::PROMPT_TOKENS as i64)),
        ("test_model_layers", Json::Int(params::TEST_MODEL_LAYERS as i64)),
        ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
        ("hidden_size", Json::Int(params::HIDDEN_SIZE as i64)),
        ("num_heads", Json::Int(params::NUM_HEADS as i64)),
        ("num_kv_heads", Json::Int(params::NUM_KV_HEADS as i64)),
        ("head_dim", Json::Int(params::HEAD_DIM as i64)),
        ("vocab_size", Json::Int(params::VOCAB_SIZE as i64)),
        ("corpus_generator_seed", Json::Int(params::CORPUS_GENERATOR_SEED as i64)),
        (
            "required_interior_first_adapted_layer",
            Json::Int(params::REQUIRED_INTERIOR_FIRST_ADAPTED_LAYER as i64),
        ),
        (
            "alora_invocation_offset_in_page",
            Json::Int(params::ALORA_INVOCATION_OFFSET_IN_PAGE as i64),
        ),
        ("page_tokens", Json::Int(params::PAGE_TOKENS as i64)),
        (
            "invalidation_coverage_matrix",
            Json::obj([
                ("kinds", Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_KINDS)),
                ("types", Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_TYPES)),
                ("paths", Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_PATHS)),
                (
                    "cells",
                    Json::Int(params::INVALIDATION_COVERAGE_MATRIX_CELLS.len() as i64),
                ),
            ]),
        ),
        ("tiny_model_config_digest", Json::str(params::TINY_MODEL_CONFIG_DIGEST)),
        ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
        ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
        (
            "declared_dependency_kinds",
            Json::strs(&params::DECLARED_DEPENDENCY_KINDS),
        ),
        (
            "chosen",
            Json::obj([
                ("max_new_tokens", Json::Int(params::MAX_NEW_TOKENS as i64)),
                ("lora_rank", Json::Int(params::LORA_RANK as i64)),
                ("lora_scale", Json::str(params::LORA_SCALE)),
                ("adapter_seed", Json::Int(params::ADAPTER_SEED as i64)),
                ("adapter_layers", Json::str(params::ADAPTER_LAYERS)),
                ("alora_projections", Json::str(params::ALORA_PROJECTIONS)),
                ("mlp_adapter_form", Json::str(params::MLP_ADAPTER_FORM)),
                ("partial_hit_mode", Json::str(params::PARTIAL_HIT_MODE)),
                ("restart_resolution", Json::str(params::RESTART_RESOLUTION)),
                ("total_kv_rows_counts", Json::str(params::TOTAL_KV_ROWS_COUNTS)),
                ("per_seed_role", Json::str(params::PER_SEED_ROLE)),
                (
                    "matrix_adapter",
                    Json::str(format!(
                        "QKV-LoRA {} first attached at layer {}",
                        params::MATRIX_ADAPTER,
                        params::MATRIX_ADAPTER_FIRST_LAYER
                    )),
                ),
                (
                    "catalog_limits",
                    Json::str("ReferenceKvLimits::default (backend default)"),
                ),
            ]),
        ),
    ])
}

/// The model pins: the dense checks plus a cold run under an adapter of each kind that
/// reproduces the backend's bound cold run.
fn verify_pins(tally: &mut Tally) -> Res<pins::Pins> {
    let backend = lab::backend()?;
    let mut checks = PinChecks::default();
    let build = pins::base_checks(
        &mut checks,
        &backend,
        &pins::canonical_digest(&pins::shape_fields()),
        params::TINY_MODEL_CONFIG_DIGEST,
        params::TOKENIZER_PIN,
    )?;
    let stack = pins::reference_stack()?;
    for config in &CONFIGS {
        let adapter = ptr_pods::Qwen2Adapter::seeded(
            backend.stack_config(),
            &ptr_pods::Qwen2AdapterSpec::pinned(
                params::ADAPTER_A,
                1,
                config.kind,
                config.first_layer,
            ),
            params::ADAPTER_SEED,
        )
        .map_err(fail("a pin adapter"))?;
        let tokens = corpus::sequence(super::layered_kv::SEQUENCE_SEED, 9_998, 40);
        let observed = lab::cold(
            &backend,
            &corpus::raw_request("pin-adapter", &tokens),
            &StackBinding::with_adapter(adapter.clone()),
        )?;
        let (generated, logits) = pins::independent_cold_run(&stack, Some(&adapter), &tokens)?;
        let observed_tokens: Vec<u32> = observed.generated.iter().map(|token| token.0).collect();
        checks.check(
            &format!("cold_run_{}", config.name),
            generated == observed_tokens && logits == observed.logits_digest,
            Json::Null,
        );
    }
    checks.check(
        "moe_topology",
        params::MOE_TOPOLOGY.starts_with("none: dense"),
        Json::str(params::MOE_TOPOLOGY),
    );
    Ok(checks.finish(
        tally,
        Json::obj([
            ("tiny_model_config_digest", Json::str(params::TINY_MODEL_CONFIG_DIGEST)),
            ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
            ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
            ("weights", Json::str(params::WEIGHTS)),
            ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
            (
                "base_checkpoint",
                Json::str("none: seeded random weights, no checkpoint"),
            ),
        ]),
        pins::chosen_json(),
        build,
    ))
}

/// What a run produced, before it is printed.
pub struct Report {
    pub line: String,
    /// The same object as `line`, before it is rendered (read by the criteria test).
    #[cfg_attr(not(test), allow(dead_code))]
    pub json: Json,
    pub coverage: Vec<Check>,
    pub falsifiers: Vec<Check>,
    pub gaps: Vec<(String, String)>,
}

impl Report {
    pub fn coverage_unmet(&self) -> usize {
        count(&self.coverage, Status::Fail)
    }

    pub fn falsifiers_tripped(&self) -> usize {
        count(&self.falsifiers, Status::Fail)
    }

    /// Whether the run is a clean pass: nothing unmet, nothing tripped, no gap.
    pub fn clean(&self) -> bool {
        self.coverage_unmet() == 0 && self.falsifiers_tripped() == 0 && self.gaps.is_empty()
    }
}

/// Run every case once for `seed` at `scale` and assemble the one JSON object. A case the
/// harness cannot run is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let mut tally = Tally::new(metrics::LIST);
    let mut adapter_ids = Ids::default();
    for name in ZERO_BASED.iter().chain(&ZERO_BASED_INTERNAL) {
        tally.add(name, 0);
    }
    tally.seed_group("adapter_revocations_performed_by_type", &params::ADAPTER_TYPES_RUN);
    tally.seed_group("adapter_reuse_attempts_by_type", &params::ADAPTER_TYPES_RUN);
    tally.seed_group("inflight_adapter_revocation_race_cases", &params::INVALIDATION_TYPES);
    tally.seed_group("inflight_input_race_cases", &params::INVALIDATION_TYPES);
    tally.seed_group(
        "invalidation_matrix_cases",
        &params::INVALIDATION_COVERAGE_MATRIX_CELLS,
    );
    tally.seed_group(
        "invalidation_matrix_stale_hits",
        &params::INVALIDATION_COVERAGE_MATRIX_CELLS,
    );
    for config in &CONFIGS {
        tally.add(&format!("config_runs.{}", config.name), 0);
    }
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut lap = Instant::now();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };
    let pins = verify_pins(&mut tally)?;
    let environment = pins::environment(&pins);
    let selection = selection(&mut tally);
    phase("pins_and_selection", &mut lap);

    let order = corpus::order(seed, 9, CONFIGS.len());
    let mut run_order = Vec::new();
    for index in order {
        let config = &CONFIGS[index];
        if scale == Scale::Reduced && !matches!(config.name, "qkv_lora_f2" | "alora_inv39") {
            continue;
        }
        arms::config_arm(&mut tally, &mut adapter_ids, seed, config)?;
        run_order.push(Json::str(config.name));
        phase(config.name, &mut lap);
    }
    let cells = arms::run_matrix(&mut tally, &mut adapter_ids, seed, scale)?;
    phase("matrix", &mut lap);
    arms::empty_restart(&mut tally)?;
    phase("empty_restart", &mut lap);

    // The identity-set digests: what the backends' own bookkeeping saw cancelled and restarted
    // among the adapter-using requests.
    tally.set_json(
        "inflight_adapter_cancelled_request_id_set_digest",
        Json::str(adapter_ids.cancelled_digest()),
    );
    tally.set_json(
        "inflight_adapter_restarted_request_id_set_digest",
        Json::str(adapter_ids.restarted_digest()),
    );
    tally.set("adapter_ids_cancelled", adapter_ids.cancelled.len() as i64);
    tally.set("adapter_ids_restarted", adapter_ids.restarted.len() as i64);

    // Fractions as integer numerators and denominators.
    let fraction = |tally: &Tally| -> (Json, Json) {
        let mut actual = Vec::new();
        let mut predicted = Vec::new();
        for config in &CONFIGS {
            let value = tally.json_or_null(&format!("fraction.{}", config.name));
            if let Some(object) = (value != Json::Null).then_some(value) {
                let field = |key: &str| emit::resolve(&object, key).cloned().unwrap_or(Json::Null);
                predicted.push((
                    config.name.to_string(),
                    Json::obj([
                        ("numerator", field("predicted_numerator")),
                        ("denominator", field("predicted_denominator")),
                    ]),
                ));
                actual.push((config.name.to_string(), object));
            }
        }
        (Json::Obj(actual), Json::Obj(predicted))
    };
    let (actual, predicted) = fraction(&tally);
    tally.set_json("shared_row_fraction", actual);
    tally.set_json("predicted_shared_row_fraction", predicted);

    // What the code under test cannot honestly produce, or produces differently.
    tally.unsupported(
        "binding_carried_post_event_lookups",
        "a request on a binding with a dead dependency (base model, adapter) is refused by begin_session_bound before any catalog lookup, so post-event lookups on stale-eligible entries (post_event_lookups*, restored/stale-eligible counters) are counted for the segment cells only; the use-time rejection of an entry that depends on such a dependency is exercised in flight (a session cancelled at its next step) and by the adapter-revocation probes in the configuration arms, which are begin refusals",
    );
    tally.unsupported(
        "base_model_inflight_restart",
        "no live base model exists to re-resolve to: the in-flight base-model cells check cancellation, silence and no insert, and do not restart (a restart would compute from the revoked base model)",
    );
    tally.unsupported(
        "prefill_flops_saving",
        "the claim is a KV-memory saving only (claim_scope kv_memory_only): partial hits are computed in full and adopted in checked mode, so no forward-compute saving exists to measure",
    );
    let unsupported_names = tally.unsupported_names();

    let (printed, gaps) = tally.metrics();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("R009")),
        (
            "status_note".to_string(),
            Json::str("one seed of the R009 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
        (
            "elapsed_ms".to_string(),
            Json::Float(started.elapsed().as_secs_f64() * 1000.0),
        ),
    ];
    pairs.extend(printed);
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([("phases", Json::Obj(phases))]),
    ));
    pairs.push(("procedure".to_string(), procedure_json(seed, scale)));
    pairs.push(("model_pins".to_string(), pins.json));
    pairs.push(("environment".to_string(), environment));
    pairs.push(("selection".to_string(), selection));
    pairs.push(("config_order".to_string(), Json::Arr(run_order)));
    pairs.push((
        "configs".to_string(),
        Json::Obj(
            CONFIGS
                .iter()
                .map(|config| {
                    (
                        config.name.to_string(),
                        Json::obj([
                            (
                                "per_layer_shared_rows",
                                tally.json_or_null(&format!("per_layer_shared_rows.{}", config.name)),
                            ),
                            (
                                "procedure_observed",
                                tally.json_or_null(&format!("procedure_config.{}", config.name)),
                            ),
                            (
                                "alora_boundary_page_shared_rows",
                                tally.json_or_null(&format!(
                                    "alora_boundary_page_shared_rows.{}",
                                    config.name
                                )),
                            ),
                        ]),
                    )
                })
                .collect(),
        ),
    ));
    pairs.push((
        "matrix".to_string(),
        Json::Arr(cells.iter().map(matrix_json).collect()),
    ));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "shared_kv_rows",
                Json::str("physical sharing: rows of each page layer that are the same allocation in the adapter request's cache and the base request's cache (KvSession::shared_leading_rows); checked against the report's shared_rows and against bitwise comparison of independent prefills' K/V rows"),
            ),
            (
                "total_kv_rows",
                Json::str("layers x prompt tokens (generated tokens' rows are not counted)"),
            ),
            (
                "alora_base_request",
                Json::str("an aLoRA configuration's base request is the first 48 tokens of its prompt and the adapter request appends 16 others, so a base prefix shorter than the prompt can be forked and truncated at the invocation; with the whole prompt cached under the base key the same aLoRA request shares nothing (internal.alora_full_prompt_cached_shared_rows)"),
            ),
            (
                "base_pages",
                Json::str("a page here is one layer's page of one 16-token chunk, i.e. one catalog entry"),
            ),
        ]),
    ));
    pairs.push(("internal".to_string(), tally.internal()));
    let root = Json::Obj(pairs.clone());
    let coverage = judge(&root, &unsupported_names, &metrics::coverage());
    let falsifiers = judge(&root, &unsupported_names, &metrics::falsifiers());
    pairs.push((
        "coverage".to_string(),
        Json::Arr(coverage.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "falsifiers".to_string(),
        Json::Arr(falsifiers.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "summary".to_string(),
        Json::obj([
            ("coverage_unmet", Json::Int(count(&coverage, Status::Fail) as i64)),
            (
                "coverage_unsupported",
                Json::Int(count(&coverage, Status::Unsupported) as i64),
            ),
            (
                "falsifiers_tripped",
                Json::Int(count(&falsifiers, Status::Fail) as i64),
            ),
            (
                "falsifiers_unsupported",
                Json::Int(count(&falsifiers, Status::Unsupported) as i64),
            ),
        ]),
    ));
    pairs.push((
        "harness_gaps".to_string(),
        Json::Arr(
            gaps.iter()
                .map(|(metric, reason)| {
                    Json::obj([
                        ("metric", Json::str(metric.clone())),
                        ("reason", Json::str(reason.clone())),
                    ])
                })
                .collect(),
        ),
    ));
    let json = Json::Obj(pairs);
    Ok(Report {
        line: json.render(),
        json,
        coverage,
        falsifiers,
        gaps,
    })
}

fn matrix_json(cell: &super::layered_kv::matrix::CellInfo) -> Json {
    cell.json()
}

/// Run the experiment's harness for `seed` and print its one JSON object. Exits 1 when the
/// run is not clean and 2 when the harness cannot run a case.
pub fn run(seed: u64) {
    let seeds = declared_seeds();
    if !seeds.contains(&seed) {
        eprintln!("seed {seed} is not one of the manifest's seeds {seeds:?}; this is a dry run");
    }
    match execute(seed, Scale::Pinned) {
        Ok(report) => {
            println!("{}", report.line);
            for check in report.coverage.iter().chain(&report.falsifiers) {
                if check.status == Status::Fail {
                    eprintln!("{} ({})", check.rule, check.observed.render());
                }
            }
            for (metric, reason) in &report.gaps {
                eprintln!("harness gap {metric}: {reason}");
            }
            if !report.clean() {
                std::process::exit(1);
            }
        }
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    }
}
