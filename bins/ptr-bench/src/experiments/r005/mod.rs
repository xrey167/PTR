//! Experiment R005, exact prefix reuse (the Exact half), as a harness: `ptr-bench
//! exact-prefix-reuse <seed>` runs every case, arm and workload of the experiment's
//! `config.toml` once for one seed and prints exactly one JSON object on stdout.
//!
//! R005 is `planned`. Nothing here is frozen, no record is written, and a run of this harness
//! is not a result: it is the harness the preregistration will name, and the dry run that
//! measures it. The code under test is the CPU reference path: `ReferenceKvReuseBackend`
//! and `KvSession` over the tiny seeded Qwen2 stack, with `PtrRuntime` as the
//! ledger-anchored liveness oracle wherever the configuration asks for committed-ledger
//! liveness, and a bare `KvPrefixCatalog` for the dependency kinds, the forged witnesses and
//! the positions that the stack backend does not carry.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names
//! (a number, an object, a digest string, or `null` with its reason under `unsupported`),
//! the procedure parameters, the model pins, the producer environment, the corpus selection
//! digests, the per-cell matrix, the per-member breakdown, the two judgements (`coverage` and
//! `falsifiers`) and the counters that belong to no metric (`internal`). The exit status is 1
//! when a coverage prerequisite is unmet, a falsifier is tripped or a metric has no value and
//! no declared reason, and 2 when the harness itself cannot run a case.

pub mod arms;
pub mod lab;
pub mod matrix;
pub mod metrics;
pub mod params;
pub mod pin;
pub mod workload;

use arms::CapacityPlan;
use matrix::Ids;
use metrics::{Check, Json, Status, Tally};
use params::Workload;
use std::time::Instant;

/// How big the run is. Only [`Scale::Pinned`] is the experiment; [`Scale::Reduced`] runs the
/// same phases on a handful of requests and small catalogs, for the smoke test, and says so
/// in its output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the smoke test only.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
}

impl Scale {
    fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
        }
    }

    fn main_requests(self) -> usize {
        match self {
            Self::Pinned => params::N_REQUESTS as usize,
            Self::Reduced => 6,
        }
    }

    /// Reduced prompts are 64 tokens sharing 32: the same 50% shared prefix.
    fn main_shape(self) -> workload::MainShape {
        match self {
            Self::Pinned => workload::MainShape::PINNED,
            Self::Reduced => workload::MainShape {
                prompt: 4 * params::CHUNK_TOKENS,
                shared: 2 * params::CHUNK_TOKENS,
            },
        }
    }

    fn plan(self, kind: Workload) -> CapacityPlan {
        match self {
            Self::Pinned => CapacityPlan::pinned(kind),
            Self::Reduced => CapacityPlan::reduced(kind),
        }
    }
}

/// Counters whose natural value is zero (violations, mismatches, leaks): see `execute`.
const ZERO_BASED: [&str; 28] = [
    "bitwise_mismatch_count",
    "revoked_fork_count",
    "stale_hits",
    "revoked_lineage_entry_hits",
    "revoked_pinned_page_served_after_revocation",
    "cross_principal_hits",
    "cross_project_hits",
    "discarded_overlay_entry_hits",
    "unreachable_revoked_page_bytes_unreclaimed",
    "cross_lineage_hits",
    "outputs_escaped_after_revocation",
    "forged_witness_forks_accepted",
    "cancelled_run_effects_admitted",
    "capacity_fallback_output_mismatches",
    "capacity_refusal_cache_inserts",
    "cache_inserts_after_supersession",
    "outputs_after_supersession_semantic",
    "outputs_after_supersession_memory",
    "stale_exact_hits",
    "stale_or_reused_witness_forks_admitted",
    "fixed_regime_crossing_parity_failures",
    "exact_outputs_across_effective_regime_transition_without_recompute",
    "exact_hits_with_mismatched_positions",
    "exact_hits_across_environment",
    "procedure_mismatches",
    "selection_digest_mismatches",
    "model_pin_mismatches",
    "active_fork_page_evictions",
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

/// The pinned selection digest of each workload's corpus.
fn pinned_selection_digest(kind: Workload) -> &'static str {
    match kind {
        Workload::Main => params::SELECTION_DIGEST_MAIN,
        Workload::NodesOnly => params::SELECTION_DIGEST_NODES_ONLY,
        Workload::BytesOnly => params::SELECTION_DIGEST_BYTES_ONLY,
    }
}

fn selection(tally: &mut Tally, scale: Scale) -> Json {
    let mut rows = Vec::new();
    for kind in Workload::ALL {
        let computed = workload::selection_digest(kind);
        let pinned = pinned_selection_digest(kind);
        // A reduced run draws fewer requests, so only the pinned scale is compared.
        let matches = computed == pinned;
        if scale == Scale::Pinned && !matches {
            tally.add("selection_digest_mismatches", 1);
        }
        rows.push((
            kind.name().to_string(),
            Json::obj([
                ("computed", Json::str(computed)),
                ("pinned", Json::str(pinned)),
                ("matches", Json::Bool(matches)),
            ]),
        ));
    }
    tally.add("selection_digest_mismatches", 0);
    Json::obj([
        ("corpus", Json::str(params::CORPUS)),
        ("rule", Json::str(params::CORPUS_SELECTION_RULE)),
        (
            "generator_seed",
            Json::Int(params::CORPUS_GENERATOR_SEED as i64),
        ),
        ("digests", Json::Obj(rows)),
    ])
}

/// The procedure parameters of the run, as `procedure_parameters_recorded_per_cell` lists
/// them (each cell also carries the limits it ran under, and a mismatch is counted there).
fn procedure_json(seed: u64, scale: Scale) -> Json {
    let seeds = declared_seeds();
    let limits = |kind: Workload| {
        let limits = kind.limits();
        Json::obj([
            ("max_trie_nodes", Json::Int(limits.max_trie_nodes as i64)),
            (
                "max_kv_page_bytes",
                Json::Int(limits.max_kv_page_bytes as i64),
            ),
            (
                "max_metadata_bytes",
                Json::Int(limits.max_metadata_bytes as i64),
            ),
        ])
    };
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        (
            "seeds",
            Json::Arr(seeds.iter().map(|seed| Json::Int(*seed as i64)).collect()),
        ),
        ("n_requests", Json::Int(params::N_REQUESTS as i64)),
        ("prompt_tokens", Json::Int(params::PROMPT_TOKENS as i64)),
        (
            "shared_prefix_tokens",
            Json::Int(params::SHARED_PREFIX_TOKENS as i64),
        ),
        ("concurrency", Json::Int(params::CONCURRENCY as i64)),
        ("max_trie_nodes", Json::Int(params::MAX_TRIE_NODES as i64)),
        (
            "max_kv_page_bytes",
            Json::Int(params::MAX_KV_PAGE_BYTES as i64),
        ),
        (
            "test_model_layers",
            Json::Int(params::TEST_MODEL_LAYERS as i64),
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
            "kv_page_bytes_per_node",
            Json::Int(params::KV_PAGE_BYTES_PER_NODE as i64),
        ),
        (
            "metadata_bytes_per_node",
            Json::Int(params::METADATA_BYTES_PER_NODE as i64),
        ),
        (
            "nodes_only_run_max_kv_page_bytes",
            Json::Int(params::NODES_ONLY_RUN_MAX_KV_PAGE_BYTES as i64),
        ),
        (
            "bytes_only_run_max_trie_nodes",
            Json::Int(params::BYTES_ONLY_RUN_MAX_TRIE_NODES as i64),
        ),
        (
            "workload_limits",
            Json::obj([
                ("main", limits(Workload::Main)),
                ("nodes_only", limits(Workload::NodesOnly)),
                ("bytes_only", limits(Workload::BytesOnly)),
            ]),
        ),
        (
            "invalidation_coverage_matrix",
            Json::obj([
                ("kinds", Json::strs(&params::MATRIX_KINDS)),
                ("types", Json::strs(&params::MATRIX_TYPES)),
                ("paths", Json::strs(&params::MATRIX_PATHS)),
                ("cells", Json::Int(params::MATRIX_CELLS.len() as i64)),
            ]),
        ),
        (
            "superseded_lookup_cells",
            Json::Arr(
                params::SUPERSEDED_LOOKUP_CELLS
                    .iter()
                    .map(|pair| Json::strs(pair))
                    .collect(),
            ),
        ),
        (
            "saturation_definition",
            Json::str(params::SATURATION_DEFINITION),
        ),
        (
            "chain_layer_covers",
            Json::strs(&params::CHAIN_LAYER_COVERS),
        ),
        (
            "tiny_model_config_digest",
            Json::str(params::TINY_MODEL_CONFIG_DIGEST),
        ),
        ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
        ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
        (
            "declared_dependency_kinds",
            Json::strs(&params::DECLARED_DEPENDENCY_KINDS),
        ),
        ("production_modes", Json::strs(&params::PRODUCTION_MODES)),
        (
            "derived_state_carriers",
            Json::obj([(
                "ptr_catalog_key",
                Json::strs(&params::DERIVED_STATE_CARRIERS_PTR_CATALOG_KEY),
            )]),
        ),
        (
            "numerics_binding_environment_fields",
            Json::strs(&params::NUMERICS_BINDING_ENVIRONMENT_FIELDS),
        ),
        (
            "chosen",
            Json::obj([
                ("max_new_tokens", Json::Int(params::MAX_NEW_TOKENS as i64)),
                (
                    "over_capacity_prompt_tokens",
                    Json::Int(params::OVER_CAPACITY_PROMPT_TOKENS as i64),
                ),
                (
                    "over_capacity_fill_prompts",
                    Json::Int(params::OVER_CAPACITY_FILL_PROMPTS as i64),
                ),
                (
                    "over_capacity_extra_prompts",
                    Json::Int(params::OVER_CAPACITY_EXTRA_PROMPTS as i64),
                ),
                (
                    "saturation_max_sessions",
                    Json::Int(params::SATURATION_MAX_SESSIONS as i64),
                ),
                (
                    "saturation_excess_requests",
                    Json::Int(params::SATURATION_EXCESS_REQUESTS as i64),
                ),
            ]),
        ),
    ])
}

/// The per-member breakdown: {semantic, memory} x {revocation, supersession}.
fn members_json(tally: &Tally) -> Json {
    Json::Obj(
        metrics::MEMBERS
            .iter()
            .map(|member| {
                (
                    member.to_string(),
                    Json::obj([
                        (
                            "race_cases",
                            Json::Int(tally.get(&format!("member_race_cases.{member}"))),
                        ),
                        (
                            "cancelled_forks",
                            Json::Int(tally.get(&format!("member_cancelled_forks.{member}"))),
                        ),
                    ]),
                )
            })
            .collect(),
    )
}

/// What a run produced, before it is printed.
pub struct Report {
    pub line: String,
    pub coverage: Vec<Check>,
    pub falsifiers: Vec<Check>,
    pub gaps: Vec<String>,
}

impl Report {
    pub fn coverage_unmet(&self) -> usize {
        metrics::count(&self.coverage, Status::Fail)
    }

    pub fn falsifiers_tripped(&self) -> usize {
        metrics::count(&self.falsifiers, Status::Fail)
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
    let mut tally = Tally::default();
    let mut ids = Ids::default();

    // Counters whose natural value is zero: a violation adds to them, so a run in which none
    // happens still prints a zero (a counter of attempts or cases is never seeded).
    for name in ZERO_BASED {
        tally.add(name, 0);
    }
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut phase = |name: &str, since: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(since.elapsed().as_secs_f64() * 1000.0),
        ));
        *since = Instant::now();
    };
    let mut lap = Instant::now();
    let pin_backend = lab::main_backend()?;
    let pins = pin::verify(&mut tally, &pin_backend)?;
    let environment = pin::environment(&mut tally, &pins);
    let selection = selection(&mut tally, scale);
    let build = pins
        .build
        .clone()
        .unwrap_or_else(|| "unverified-build".to_string());
    drop(pin_backend);
    phase("pins_and_environment", &mut lap);

    arms::main_workload(&mut tally, seed, scale.main_requests(), scale.main_shape())?;
    phase("main_workload", &mut lap);
    for kind in [Workload::NodesOnly, Workload::BytesOnly] {
        arms::over_capacity(&mut tally, &mut ids, seed, kind, scale.plan(kind))?;
        phase(kind.name(), &mut lap);
    }
    let cells = matrix::run_all(&mut tally, &mut ids, seed)?;
    phase("matrix", &mut lap);
    arms::isolated_cases(&mut tally, &mut ids, &build)?;
    phase("isolated_cases", &mut lap);

    // The identity-set digests: what the backends' own bookkeeping saw cancelled and
    // restarted, and what was refused for capacity and completed after the refusal.
    let digest = |set: &std::collections::BTreeSet<String>| {
        Json::str(lab::id_set_digest(set.iter().map(String::as_str)))
    };
    tally.set_json("cancelled_fork_id_set_digest", digest(&ids.cancelled));
    tally.set_json("restarted_fork_id_set_digest", digest(&ids.restarted));
    tally.set_json(
        "capacity_refused_request_id_set_digest",
        digest(&ids.refused),
    );
    tally.set_json(
        "capacity_completed_request_id_set_digest",
        digest(&ids.completed_after_refusal),
    );
    tally.set("cancelled_fork_ids", ids.cancelled.len() as i64);
    tally.set("restarted_fork_ids", ids.restarted.len() as i64);

    // Totals the manifest names without a workload.
    for (total, workload) in [
        ("peak_trie_nodes", "peak_trie_nodes_by_workload.main"),
        ("peak_kv_page_bytes", "peak_kv_page_bytes_by_workload.main"),
        (
            "peak_metadata_bytes",
            "peak_metadata_bytes_by_workload.main",
        ),
    ] {
        let value = tally.get(workload);
        tally.set(total, value);
    }

    // What the code under test cannot honestly produce.
    tally.unsupported(
        "revoked_pinned_page_bytes_reclaimed_after_release.backend_resident_bytes",
        "the paged KV backend exposes no resident-byte accounting of its prefix registry through the public API; reclamation is measured in the catalog's charged bytes",
    );
    tally.unsupported(
        "base_model_inflight_outputs_after_invalidation",
        "no inference step carries a base-model dependency: ReferenceKvReuseBackend records only the request's prompt segments, so an in-flight base-model invalidation can be observed on a bare catalog (a pin cancelled, an insert refused: reported under base_model_inflight_cache_inserts and in the matrix) but no output exists to count",
    );
    for info in &cells {
        if info.evidence == "catalog" && info.cell.contains("_inflight_") {
            tally.unsupported(
                &format!("invalidation_matrix.{}.escaped_outputs", info.cell),
                "catalog-layer cell: the stale hit is a still-live pin or an accepted insert after the event; no inference step runs, so an escaped output is not observable",
            );
        }
    }

    let (printed, gaps) = tally.metrics();
    let coverage = metrics::coverage(&tally);
    let falsifiers = metrics::falsifiers(&tally);
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str("exact-prefix-reuse")),
        ("experiment".to_string(), Json::str("R005")),
        (
            "status_note".to_string(),
            Json::str("R005 is planned: this is the harness, not a result; nothing here is frozen or recorded"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
        ("elapsed_ms".to_string(), Json::Float(started.elapsed().as_secs_f64() * 1000.0)),
        (
            "peak_scope".to_string(),
            Json::str("peak_trie_nodes, peak_kv_page_bytes and peak_metadata_bytes are the main workload's; the *_by_workload metrics hold every workload's"),
        ),
    ];
    pairs.extend(printed);
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([
            ("main_warm_and_cold_twins", tally.json_or_null("timing_ms")),
            ("phases", Json::Obj(phases)),
        ]),
    ));
    pairs.push(("procedure".to_string(), procedure_json(seed, scale)));
    pairs.push((
        "procedure_observed".to_string(),
        Json::obj([
            ("main", tally.json_or_null("procedure_main")),
            ("nodes_only", tally.json_or_null("procedure_nodes_only")),
            ("bytes_only", tally.json_or_null("procedure_bytes_only")),
        ]),
    ));
    pairs.push(("model_pins".to_string(), pins.json));
    pairs.push(("environment".to_string(), environment));
    pairs.push(("selection".to_string(), selection));
    pairs.push((
        "matrix".to_string(),
        Json::Arr(cells.iter().map(matrix::CellInfo::json).collect()),
    ));
    pairs.push(("members".to_string(), members_json(&tally)));
    pairs.push((
        "witness_reuse_static_evidence_present".to_string(),
        tally.json_or_null("witness_reuse_static_evidence_present"),
    ));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "kv_page_bytes",
                Json::str("catalog-charged bytes (one charge per unique page digest, 16 KiB a page in the pinned shape), read from KvCatalogStats; the backend's resident memory is not observable through the public API"),
            ),
            (
                "peaks",
                Json::str("sampled after every prefill chunk, the only step that inserts, and at the end of a run; a bound is checked by the catalog before it admits, so no peak between samples can exceed it"),
            ),
            (
                "revoked_pinned_page_bytes_reclaimed_after_release",
                Json::str("catalog accounting: bytes charged while the revoked entry is pinned minus bytes charged after the fork released it"),
            ),
            (
                "cancelled_run_effects",
                Json::str("a cancelled session yields no outcome; the attempts are finish, one more token, and the action on the revoked generation at the current revision against the hard boundary (authorize_action)"),
            ),
            (
                "evictions",
                Json::str("all evictions of the three workloads; evictions_*_run are those of the unique-prompt phase that follows the fill"),
            ),
            (
                "wall_time_ms",
                Json::str("main workload, warm arm (reuse), one request at a time, catalog sampling included; the cold twins' time is under timing_ms"),
            ),
        ]),
    ));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
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
                Json::Int(metrics::count(&coverage, Status::Fail) as i64),
            ),
            (
                "coverage_unsupported",
                Json::Int(metrics::count(&coverage, Status::Unsupported) as i64),
            ),
            (
                "falsifiers_tripped",
                Json::Int(metrics::count(&falsifiers, Status::Fail) as i64),
            ),
            (
                "falsifiers_unsupported",
                Json::Int(metrics::count(&falsifiers, Status::Unsupported) as i64),
            ),
            ("bitwise_compared", Json::Int(tally.get("bitwise_compared"))),
        ]),
    ));
    pairs.push((
        "harness_gaps".to_string(),
        Json::Arr(gaps.iter().map(|gap| Json::str(gap.clone())).collect()),
    ));
    pairs.push(("internal".to_string(), tally.internal()));
    Ok(Report {
        line: Json::Obj(pairs).render(),
        coverage,
        falsifiers,
        gaps,
    })
}

/// Run the experiment's harness for `seed` and print its one JSON object. Exits 1 when the
/// run is not clean (an unmet prerequisite, a tripped falsifier, a metric without a value or a
/// reason) and 2 when the harness cannot run a case.
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
                    eprintln!(
                        "{}: {} ({})",
                        check.name,
                        check.rule,
                        check.observed.render()
                    );
                }
            }
            for gap in &report.gaps {
                eprintln!("no value and no reason for {gap}");
            }
            if !report.clean() {
                std::process::exit(1);
            }
        }
        Err(reason) => {
            eprintln!("exact-prefix-reuse: {reason}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_seeds_are_read() {
        assert_eq!(declared_seeds(), vec![17, 29, 43, 71, 101]);
    }

    #[test]
    fn the_pinned_selection_digests_are_what_the_corpus_generator_produces() {
        for kind in Workload::ALL {
            assert_eq!(
                workload::selection_digest(kind),
                pinned_selection_digest(kind),
                "{}",
                kind.name()
            );
        }
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric_and_judgement() {
        let report = execute(17, Scale::Reduced).expect("a reduced run");
        if let Some(at) = report.line.find("\"phases\"") {
            eprintln!("PHASES {}", &report.line[at..at + 220]);
        }
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"exact-prefix-reuse\""));
        for name in metrics::METRIC_NAMES {
            assert!(report.line.contains(&format!("\"{name}\":")), "{name}");
        }
        assert!(report.gaps.is_empty(), "{:?}", report.gaps);
        // No falsifier trips on the reduced workload but the ones of the known finding: a
        // semantic input superseded by a committed edit is not seen by the runtime's use-time
        // liveness check when the catalog's invalidation is lost (`PtrRuntime::is_live`, see
        // the draft preregistration). A fix of it may only shrink this set.
        const KNOWN_FINDING: [&str; 8] = [
            "stale_hits",
            "unreachable_revoked_page_bytes_unreclaimed",
            "invalidation_matrix_stale_hits.semantic_input_normal_supersession",
            "invalidation_matrix_stale_hits.semantic_input_lag_supersession",
            "invalidation_matrix_stale_hits.semantic_input_restore_supersession",
            "superseded_stale_hits.semantic_normal",
            "superseded_stale_hits.semantic_lag",
            "superseded_stale_hits.semantic_restore",
        ];
        let tripped: Vec<&str> = report
            .falsifiers
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.name.as_str())
            .filter(|name| !KNOWN_FINDING.contains(name))
            .collect();
        assert!(tripped.is_empty(), "{tripped:?}");
        // The only unmet prerequisites of a reduced run are the ones that scale defines:
        // the procedure the pinned scale fixes.
        let unmet: Vec<&str> = report
            .coverage
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.name.as_str())
            .collect();
        assert_eq!(unmet, ["procedure_mismatches"], "{unmet:?}");
        assert!(report.line.contains("\"unsupported\":{"));
        assert!(report
            .line
            .contains("\"witness_reuse_static_evidence_passing\":null"));
    }
}
