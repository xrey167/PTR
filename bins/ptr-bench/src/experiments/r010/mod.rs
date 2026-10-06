//! Experiment R010, MoE expert KV revocation, as a harness: `ptr-bench
//! moe-expert-kv-revocation <seed>` runs every configuration, arm and matrix cell of the experiment's `config.toml`
//! once for one seed and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`; a run at a reduced scale (the smoke test) is no evidence. The code
//! under test is the CPU reference path: `ReferenceKvReuseBackend` and its bound `KvSession`s
//! over the tiny seeded Qwen2 stack with the seeded tiny MoE and the real expert admission and fence path, `PtrRuntime` as the
//! ledger-anchored liveness oracle.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names
//! (a number, an object, a digest string, or `null` with its reason under `unsupported`),
//! the procedure parameters, the model pins, the producer environment, the corpus selection
//! digest, the per-cell details, the two judgements (`coverage` and `falsifiers`), and the
//! counters that belong to no metric (`internal`). The exit status is 1 when a coverage
//! prerequisite is unmet, a falsifier is tripped or a metric has no value and no declared
//! reason, and 2 when the harness itself cannot run a case (the revoked expert is never
//! routed to, a fixture does not build).

pub mod arms;
pub mod metrics;
pub mod params;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::layered_kv::lab::{self, fail, Res};
use super::layered_kv::pins::{self, PinChecks};
use super::layered_kv::{corpus, Scale};
use arms::World;
use ptr_runtime::StackBinding;
use std::time::Instant;

/// Counters whose natural value is zero (violations, mismatches): a run in which none happens
/// still prints a zero. A counter of attempts or cases is never seeded, so a case that never
/// ran is a gap or an unmet prerequisite, not a silent zero.
const ZERO_BASED: [&str; 14] = [
    "stale_pages_served",
    "wrongly_dropped_pages",
    "nondeterministic_exact_tags",
    "bitwise_mismatch_count",
    "mismatching_runs_tagged_exact",
    "revoked_lineage_entry_hits",
    "stale_hits",
    "stale_expert_generation_hits",
    "outputs_after_expert_revocation",
    "outputs_after_input_invalidation",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "model_pin_mismatches",
    "kv_reuse_after_expert_fence",
];

/// Internal violation counters the judgements read.
const ZERO_BASED_INTERNAL: [&str; 7] = [
    "adoption_mismatches",
    "restart_resolution_mismatches",
    "matrix_selectivity_mismatches",
    "layered_scope_mismatches",
    "batch_invariance_mismatches",
    "approximate_catalog_movement",
    "expert_restart_output_mismatches",
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
        (
            "generator_seed",
            Json::Int(params::CORPUS_GENERATOR_SEED as i64),
        ),
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

/// The procedure parameters of the run, as `procedure_parameters_recorded_per_cell` lists
/// them (each cell also carries what it observed, and a mismatch is counted there).
fn procedure_json(seed: u64, scale: Scale, world: &World) -> Json {
    let seeds = declared_seeds();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        ("seeds", Json::ints(seeds.iter().map(|seed| *seed as i64))),
        (
            "test_model_layers",
            Json::Int(params::TEST_MODEL_LAYERS as i64),
        ),
        (
            "revoked_expert_layer_j",
            Json::Int(params::REVOKED_EXPERT_LAYER_J as i64),
        ),
        ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
        ("hidden_size", Json::Int(params::HIDDEN_SIZE as i64)),
        ("num_heads", Json::Int(params::NUM_HEADS as i64)),
        ("num_kv_heads", Json::Int(params::NUM_KV_HEADS as i64)),
        ("head_dim", Json::Int(params::HEAD_DIM as i64)),
        ("vocab_size", Json::Int(params::VOCAB_SIZE as i64)),
        (
            "corpus_generator_seed",
            Json::Int(params::CORPUS_GENERATOR_SEED as i64),
        ),
        (
            "tiny_model_config_digest",
            Json::str(params::TINY_MODEL_CONFIG_DIGEST),
        ),
        ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
        (
            "moe_topology",
            Json::obj([
                (
                    "num_experts",
                    Json::Int(params::MOE_TOPOLOGY_NUM_EXPERTS as i64),
                ),
                ("top_k", Json::Int(params::MOE_TOPOLOGY_TOP_K as i64)),
                (
                    "router_layout",
                    Json::str(params::MOE_TOPOLOGY_ROUTER_LAYOUT),
                ),
            ]),
        ),
        (
            "declared_dependency_kinds",
            Json::strs(&params::DECLARED_DEPENDENCY_KINDS),
        ),
        (
            "chosen",
            Json::obj([
                ("max_new_tokens", Json::Int(params::MAX_NEW_TOKENS as i64)),
                ("prompt_tokens", Json::Int(params::PROMPT_TOKENS as i64)),
                ("page_tokens", Json::Int(params::PAGE_TOKENS as i64)),
                ("moe_seed", Json::Int(params::MOE_SEED as i64)),
                ("moe_layers", Json::str(params::MOE_LAYERS)),
                (
                    "expert_intermediate_size",
                    Json::Int(params::EXPERT_INTERMEDIATE_SIZE as i64),
                ),
                (
                    "top_k_normalisation",
                    Json::str(params::TOP_K_NORMALISATION),
                ),
                ("tie_break", Json::str(params::TIE_BREAK)),
                ("masked_expert", Json::str(params::MASKED_EXPERT)),
                (
                    "revoked_expert_index_pinned",
                    Json::Int(params::REVOKED_EXPERT_INDEX as i64),
                ),
                (
                    "revoked_expert_index_observed",
                    Json::Int(world.index as i64),
                ),
                (
                    "revoked_expert_index_matches",
                    Json::Bool(world.matches_pinned),
                ),
                (
                    "batch_variant_capacity",
                    Json::Int(params::BATCH_VARIANT_CAPACITY as i64),
                ),
                (
                    "cobatch_requests",
                    Json::Int(params::COBATCH_REQUESTS as i64),
                ),
                ("expert_tracking", Json::str(params::EXPERT_TRACKING)),
                ("partial_hit_mode", Json::str(params::PARTIAL_HIT_MODE)),
                ("restart_resolution", Json::str(params::RESTART_RESOLUTION)),
                ("per_seed_role", Json::str(params::PER_SEED_ROLE)),
                (
                    "catalog_limits",
                    Json::str("ReferenceKvLimits::default (backend default)"),
                ),
            ]),
        ),
    ])
}

/// The model pins: the dense checks plus the MoE topology of the run's own MoE and a cold run
/// under it that reproduces an independent MoE stack's.
fn verify_pins(tally: &mut Tally, world: &World) -> Res<pins::Pins> {
    let backend = lab::backend()?;
    let mut checks = PinChecks::default();
    let mut fields = pins::shape_fields();
    fields.push(("num_experts", params::MOE_TOPOLOGY_NUM_EXPERTS));
    fields.push(("top_k", params::MOE_TOPOLOGY_TOP_K));
    let build = pins::base_checks(
        &mut checks,
        &backend,
        &pins::canonical_digest(&fields),
        params::TINY_MODEL_CONFIG_DIGEST,
        params::TOKENIZER_PIN,
    )?;
    let moe = &world.moe;
    let topology_ok = (0..params::TEST_MODEL_LAYERS as usize).all(|layer| {
        moe.layer(layer)
            .is_some_and(|l| l.experts.len() as u64 == params::MOE_TOPOLOGY_NUM_EXPERTS)
    }) && moe.top_k() as u64 == params::MOE_TOPOLOGY_TOP_K
        && moe.mode() == ptr_pods::Qwen2RoutingMode::Dropless
        && moe.expert_intermediate() as u64 == params::EXPERT_INTERMEDIATE_SIZE
        && moe.routing_determinism() == ptr_pods::RoutingDeterminism::BatchInvariant;
    checks.check(
        "moe_topology",
        topology_ok,
        Json::obj([
            (
                "num_experts",
                Json::Int(params::MOE_TOPOLOGY_NUM_EXPERTS as i64),
            ),
            ("top_k", Json::Int(moe.top_k() as i64)),
            (
                "router_layout",
                Json::str(params::MOE_TOPOLOGY_ROUTER_LAYOUT),
            ),
        ]),
    );
    let stack = pins::reference_stack()?
        .with_moe(moe.clone())
        .map_err(fail("the MoE stack"))?;
    let tokens = corpus::sequence(super::layered_kv::SEQUENCE_SEED, 9_998, 40);
    let binding = world_binding(world)?;
    let observed = lab::cold(&backend, &corpus::raw_request("pin-moe", &tokens), &binding)?;
    let (generated, logits) = pins::independent_cold_run(&stack, None, &tokens)?;
    let observed_tokens: Vec<u32> = observed.generated.iter().map(|token| token.0).collect();
    checks.check(
        "cold_run_moe",
        generated == observed_tokens && logits == observed.logits_digest,
        Json::Null,
    );
    Ok(checks.finish(
        tally,
        Json::obj([
            (
                "tiny_model_config_digest",
                Json::str(params::TINY_MODEL_CONFIG_DIGEST),
            ),
            ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
            (
                "moe_topology",
                Json::obj([
                    (
                        "num_experts",
                        Json::Int(params::MOE_TOPOLOGY_NUM_EXPERTS as i64),
                    ),
                    ("top_k", Json::Int(params::MOE_TOPOLOGY_TOP_K as i64)),
                    (
                        "router_layout",
                        Json::str(params::MOE_TOPOLOGY_ROUTER_LAYOUT),
                    ),
                ]),
            ),
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

fn world_binding(world: &World) -> Res<StackBinding> {
    let mut authority = lab::Authority::new(&[])?;
    arms::birth_all(&mut authority)?;
    Ok(world.binding(&authority, false))
}

/// What a run produced, before it is printed.
pub struct Report {
    pub line: String,
    /// The same object as `line`, before it is rendered (read by the criteria test).
    #[cfg_attr(not(test), allow(dead_code))]
    pub json: Json,
    pub coverage: Vec<Check>,
    pub falsifiers: Vec<Check>,
    pub gaps: Gaps,
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
    for name in ZERO_BASED.iter().chain(&ZERO_BASED_INTERNAL) {
        tally.add(name, 0);
    }
    tally.seed_group(
        "inflight_expert_revocation_race_cases",
        &params::INVALIDATION_TYPES,
    );
    tally.seed_group("inflight_input_race_cases", &params::INVALIDATION_TYPES);
    tally.seed_group("inflight_race_cases_by_cell", &metrics::inflight_cells());
    tally.seed_group(
        "inflight_outputs_after_invalidation",
        &metrics::inflight_cells(),
    );
    tally.seed_group(
        "invalidation_matrix_cases",
        &params::INVALIDATION_COVERAGE_MATRIX_CELLS,
    );
    tally.seed_group(
        "invalidation_matrix_stale_hits",
        &params::INVALIDATION_COVERAGE_MATRIX_CELLS,
    );
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut lap = Instant::now();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };
    let world = World::new()?;
    if !world.matches_pinned {
        tally.add("procedure_mismatches", 1);
    }
    let pins = verify_pins(&mut tally, &world)?;
    let environment = pins::environment(&pins);
    let selection = selection(&mut tally);
    let procedure = procedure_json(seed, scale, &world);
    phase("pins_and_selection", &mut lap);

    // The arms run in a seeded order.
    let mut arm_order = Vec::new();
    for index in corpus::order(seed, 10, 3) {
        match index {
            0 => {
                arms::deterministic_arm(&mut tally, &world)?;
                arm_order.push(Json::str("deterministic_revocation"));
            }
            1 => {
                arms::quarantine_arm(&mut tally, &world)?;
                arm_order.push(Json::str("quarantine"));
            }
            _ => {
                arms::empty_restart(&mut tally, &world)?;
                arm_order.push(Json::str("empty_restart"));
            }
        }
        phase("arm", &mut lap);
    }
    let revoked_index = world.index;
    let cells = arms::run_matrix(&mut tally, world, seed, scale)?;
    phase("matrix", &mut lap);

    tally.unsupported(
        "binding_carried_post_event_lookups",
        "a request on a binding with a dead dependency (base model, expert) is refused by begin_session_bound before any catalog lookup, so post-event lookups on stale-eligible entries (post_event_lookups*, restored/stale-eligible counters) are counted for the segment cells only; the use-time rejection of an entry that depends on such a dependency is exercised in flight (a session cancelled at its next step) and by the post-revocation probes of the deterministic arm, which are begin refusals and masked-router lookups under other chains",
    );
    tally.unsupported(
        "base_model_inflight_restart",
        "no live base model exists to re-resolve to: the in-flight base-model cells check cancellation, silence and no insert, and do not restart (a restart would compute from the revoked base model)",
    );
    let unsupported_names = tally.unsupported_names();

    let (printed, gaps) = tally.metrics();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("R010")),
        (
            "status_note".to_string(),
            Json::str("one seed of the R010 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
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
    pairs.push(("procedure".to_string(), procedure));
    pairs.push(("model_pins".to_string(), pins.json));
    pairs.push(("environment".to_string(), environment));
    pairs.push(("selection".to_string(), selection));
    pairs.push(("arm_order".to_string(), Json::Arr(arm_order)));
    pairs.push((
        "revoked_expert".to_string(),
        Json::obj([
            ("layer", Json::Int(params::REVOKED_EXPERT_LAYER_J as i64)),
            ("index", Json::Int(revoked_index as i64)),
        ]),
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
                "pages",
                Json::str("a page is one layer's page of one 16-token chunk, i.e. one catalog entry; pages above j are those of layers 2 and 3 of the 4 chunks of the 64-token prompt (8), at or below j those of layers 0 and 1 (8)"),
            ),
            (
                "page_sets",
                Json::str("probed_page_set_digest and cached_page_set_digest_above_j are digests of the sets of (layer, chunk) pages above j that the post-revocation probe looks up and that the warm run cached; both are derived from the request geometry and the run's own insert count (the backend exposes no entry listing), and the probe's physical sharing with the pre-revocation cache is measured with KvSession::shared_leading_rows"),
            ),
            (
                "stale_pages_served",
                Json::str("page layers above j of the masked-router probe that share any row with the pre-revocation run's cache"),
            ),
            (
                "wrongly_dropped_pages",
                Json::str("pages at or below j minus those the masked-router probe shares as whole pages with the pre-revocation run's cache"),
            ),
            (
                "nondeterministic_divergence_observed",
                Json::str("two requests co-batched in both orders under capacity-limited routing (capacity 1 per expert per step) whose last-token logits differ in some bit; the dropless control must not differ (internal.batch_invariance_mismatches)"),
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
            (
                "coverage_unmet",
                Json::Int(count(&coverage, Status::Fail) as i64),
            ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/runtime/R010-moe-expert-kv-revocation/PREREGISTRATION.md"
    );
    const CRITERIA: &str = include_str!(
        "../../../../../experiments/runtime/R010-moe-expert-kv-revocation/criteria.toml"
    );

    /// The shared reduced run (a subset of the arms; it takes a few seconds in a debug build).
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(17, Scale::Reduced).expect("a reduced run"))
    }

    fn rules(table: &toml::Table, key: &str) -> Vec<String> {
        table[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is an array"))
            .iter()
            .map(|rule| rule.as_str().expect("a rule is a string").to_string())
            .collect()
    }

    #[test]
    fn the_manifest_seeds_are_read() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
    }

    #[test]
    fn the_pinned_selection_digest_is_what_the_corpus_generator_produces() {
        assert_eq!(arms::selection_digest(), params::SELECTION_DIGEST);
        assert!(PREREGISTRATION.contains(params::SELECTION_DIGEST));
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("R010"));
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_a_path_the_harness_prints() {
        let report = reduced_report();
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            assert_eq!(tokens.len(), 3, "{rule}");
            for token in [tokens[0], tokens[2]] {
                let literal = token
                    .strip_prefix('-')
                    .unwrap_or(token)
                    .bytes()
                    .all(|byte| byte.is_ascii_digit());
                if !literal {
                    assert!(
                        emit::resolve(&report.json, token).is_some(),
                        "{rule}: {token} does not resolve in the harness JSON"
                    );
                }
            }
        }
    }

    #[test]
    fn the_manifest_entrypoint_is_the_subcommand_and_the_table_is_the_compiled_choices() {
        assert_eq!(params::BENCHMARK, "moe-expert-kv-revocation");
        let config: toml::Table = params::CONFIG_TOML.parse().expect("config.toml parses");
        let table = config["preregistration"]
            .as_table()
            .expect("[preregistration]");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        assert_eq!(manifest["entrypoint"].as_str(), Some(params::ENTRYPOINT));
        assert_eq!(table["seeds"], manifest["seeds"]);
        assert_eq!(table["harness"].as_str(), Some(params::BENCHMARK));
        assert_eq!(
            table["corpus_selection_digest"].as_str(),
            Some(params::SELECTION_DIGEST)
        );
        assert_eq!(
            table["moe_seed"].as_integer(),
            Some(params::MOE_SEED as i64)
        );
        assert_eq!(
            table["max_new_tokens"].as_integer(),
            Some(params::MAX_NEW_TOKENS as i64)
        );
        assert_eq!(
            table["prompt_tokens"].as_integer(),
            Some(params::PROMPT_TOKENS as i64)
        );
        assert_eq!(
            table["page_tokens"].as_integer(),
            Some(params::PAGE_TOKENS as i64)
        );
        assert_eq!(
            table["revoked_expert_index"].as_integer(),
            Some(params::REVOKED_EXPERT_INDEX as i64)
        );
        assert_eq!(
            table["expert_intermediate_size"].as_integer(),
            Some(params::EXPERT_INTERMEDIATE_SIZE as i64)
        );
        assert_eq!(
            table["batch_variant_capacity"].as_integer(),
            Some(params::BATCH_VARIANT_CAPACITY as i64)
        );
        assert_eq!(
            table["partial_hit_mode"].as_str(),
            Some(params::PARTIAL_HIT_MODE)
        );
        assert_eq!(
            table["restart_resolution"].as_str(),
            Some(params::RESTART_RESOLUTION)
        );
        assert_eq!(table["masked_expert"].as_str(), Some(params::MASKED_EXPERT));
        assert_eq!(
            table["expert_tracking"].as_str(),
            Some(params::EXPERT_TRACKING)
        );
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric_and_no_tripped_falsifier() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"moe-expert-kv-revocation\""));
        for name in metrics::METRIC_NAMES {
            assert!(report.line.contains(&format!("\"{name}\":")), "{name}");
        }
        assert!(report.line.contains("\"unsupported\":{"));
        assert!(report.line.contains("\"harness_gaps\":["));
        let tripped: Vec<&str> = report
            .falsifiers
            .iter()
            .filter(|check| check.status == emit::Status::Fail)
            .map(|check| check.rule.as_str())
            .filter(|rule| {
                !rule.starts_with("invalidation_matrix_stale_hits.")
                    && !rule.contains("expert_generation")
            })
            .collect();
        assert!(tripped.is_empty(), "{tripped:?}");
    }

    #[test]
    fn the_procedure_records_every_parameter_the_config_lists() {
        let procedure = emit::resolve(&reduced_report().json, "procedure").expect("procedure");
        for name in params::PROCEDURE_PARAMETERS_RECORDED_PER_CELL {
            assert!(emit::resolve(procedure, name).is_some(), "{name}");
        }
    }

    /// Regression for a runtime defect this harness found (restored per-layer entries
    /// decoded with the whole-stack key scope and could never fork): a live restored entry
    /// whose pages exist again forks, and no scope mismatch is counted.
    #[test]
    fn a_restored_per_layer_entry_forks_once_its_pages_exist() {
        use crate::experiments::layered_kv::lab::Ids;
        use crate::experiments::layered_kv::matrix::{self, Invalidation, Kind, Path};
        let mut tally = Tally::new(metrics::LIST);
        let mut ids = Ids::default();
        let mut infos = Vec::new();
        matrix::run_cell(
            &mut tally,
            &mut ids,
            &mut infos,
            &arms::R010Rig {
                world: arms::World::new().unwrap(),
            },
            17,
            Kind::BaseModel,
            Invalidation::Revocation,
            Path::Restore,
        )
        .unwrap();
        assert_eq!(tally.get("restore_control_hit_tokens"), 32);
        assert_eq!(tally.get("layered_scope_mismatches"), 0);
    }
}
