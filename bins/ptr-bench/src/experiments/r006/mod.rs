//! Experiment R006, the one-layer positional (shifted) reuse oracle, as a harness: `ptr-bench
//! positional-reuse-oracle <seed>` runs the cases C1 to C11 of the experiment's draft
//! preregistration once for one seed and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`; a run at a reduced scale (the smoke test) is no evidence. The code
//! under test is the CPU reference KV backend with positional reuse opted in
//! (`ReferenceKvReuseBackend::with_positional_reuse`: the secondary index of its catalog and
//! the positional session), driven only through its public API, on a one-layer seeded random
//! model whose pre-RoPE K and V rows of a token are a pure function of the token. That is why
//! staleness is structural here and not numeric (see [`lab`]).
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names
//! (a JSON integer, an object of integers for a grouped metric, or a digest string; the
//! timing-only `reuse_latency_ms` is an object of milliseconds), the timings (never read by a
//! rule), the procedure parameters, the model pins, the producer environment, the corpus
//! selection digest, the per-case records, the two judgements (`coverage` and `falsifiers`),
//! and the counters that belong to no metric (`internal`). The exit status is 1 when a
//! coverage prerequisite is unmet, a falsifier is tripped or a metric has no value and no
//! declared reason, and 2 when the harness itself cannot run a case (a refusal of the wrong
//! type, a fixture that does not build, an accounting that does not add up, a live control that
//! reuses nothing, a negative control that does not trip).

// The configuration constants echo pinned strings that no code reads (the drift test compares
// them with `config.toml`), so some of them are used only by the tests.
#![allow(dead_code)]

pub mod cases;
pub mod lab;
pub mod matrix;
pub mod metrics;
pub mod params;
pub mod pins;

use crate::experiments::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use crate::experiments::layered_kv::lab::Res;
use crate::experiments::layered_kv::pins::{environment, Pins};
use lab::{Anchor, CaseIds, Lab, Material, Observed, Procedure, Scale, Shape, Timings};
use std::time::Instant;

fn selection(tally: &mut Tally, material: &Material) -> Json {
    let computed = material.selection_digest();
    let matches = computed == params::SELECTION_DIGEST;
    tally.set("selection_digest_mismatches", i64::from(!matches));
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
        ("sequences", Json::Int(params::SELECTION_SEQUENCES as i64)),
        (
            "sequence_tokens",
            Json::Int(params::SELECTION_SEQUENCE_TOKENS as i64),
        ),
        ("per_seed_role", Json::str(params::PER_SEED_ROLE)),
        ("role_order", Json::ints(material.roles())),
        ("role_names", Json::strs(&cases::role_names())),
    ])
}

/// The run's identity and the choices of the harness, as `procedure` records them next to the
/// observed parameters (which the cases fill in).
fn chosen_json(seed: u64, scale: Scale) -> Json {
    let shape = scale.shape();
    let seeds = params::declared_seeds();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        (
            "shape_is_pinned",
            Json::Bool(shape == lab::PINNED_SHAPE && shape.limits_hold()),
        ),
        (
            "shape",
            Json::obj([
                (
                    "shifts",
                    Json::ints(shape.shifts.iter().map(|shift| *shift as i64)),
                ),
                (
                    "inserted",
                    Json::ints(shape.inserted.iter().map(|shift| *shift as i64)),
                ),
                ("segment", Json::Int(shape.segment as i64)),
                ("tail", Json::Int(shape.tail as i64)),
                ("origin_prefix", Json::Int(shape.origin_prefix as i64)),
            ]),
        ),
        (
            "longest_prompt_tokens",
            Json::Int(shape.longest_prompt() as i64),
        ),
        (
            "max_prompt_tokens",
            Json::Int(lab::limits(shape).max_prompt_tokens as i64),
        ),
        (
            "max_new_tokens",
            Json::Int(lab::limits(shape).max_new_tokens as i64),
        ),
        (
            "catalog_limits",
            Json::str("max_trie_nodes 65536, max_kv_page_bytes 2^32, max_metadata_bytes 2^30; secondary index at its default bounds (an insert refusal on the honest path is exit 2)"),
        ),
        ("positional_reuse", Json::str(params::POSITIONAL_REUSE_DEFAULT)),
        ("staleness_measurement", Json::str(params::STALENESS_MEASUREMENT)),
    ])
}

/// The model pins: the checks of the independent one-layer stack and of the backend the cases
/// run (digest of the configuration it reports, tokenizer, shape, chain id, cold run reproduced
/// by the independent stack), with the raised limits.
fn verify_pins(tally: &mut Tally, shape: Shape) -> Res<Pins> {
    let backend = lab::positional_backend(shape, ptr_runtime::PositionalControl::None)?;
    let default_chain = ptr_runtime::ReferenceKvReuseBackend::with_layers(
        ptr_runtime::ReferenceKvLimits::default(),
        crate::experiments::layered_kv::lab::scope(),
        ptr_runtime::REFERENCE_KV_WEIGHT_SEED,
        params::TEST_MODEL_LAYERS as usize,
    )
    .map_err(|error| format!("a default-limits backend: {error:?}"))?
    .chain_id()
    .0;
    let mut checks = pins::PinChecks::default();
    pins::verify_independent(&mut checks, params::TINY_MODEL_CONFIG_DIGEST)?;
    let build = pins::verify_backend(
        &mut checks,
        &backend,
        default_chain,
        shape.max_prompt_tokens,
        params::TINY_MODEL_CONFIG_DIGEST,
    )?;
    Ok(checks.finish(
        tally,
        pins::pinned_json(),
        pins::chosen_json(shape.max_prompt_tokens),
        build,
    ))
}

/// The procedure parameters of `procedure_parameters_recorded_per_cell`, each observed from
/// the objects the cases built and compared with the pinned value.
fn procedure(lab: &Lab<'_>, scale: Scale) -> Res<Procedure> {
    let shape = scale.shape();
    let backend = lab::positional_backend(shape, ptr_runtime::PositionalControl::None)?;
    let config = backend.stack_config();
    let tokenizer = backend.tokenizer();
    let identity = ptr_model_api::Tokenizer::identity(tokenizer);
    let tokenizer_text = if identity.name() == "ptr.byte" && identity.version() == "1" {
        format!(
            "ByteTokenizer, vocabulary {} (no external revision)",
            config.vocab_size
        )
    } else {
        format!("{}@{}", identity.name(), identity.version())
    };
    let observed: &Observed = &lab.observed;
    let mut procedure = Procedure::default();
    procedure.ints("seeds", &params::declared_seeds(), &[17, 29, 43, 71, 101]);
    procedure.ints("position_shift_tokens", &observed.shifts, &shape.shifts);
    procedure.ints(
        "nonaligned_insertion_tokens",
        &observed.inserted,
        &shape.inserted,
    );
    procedure.int(
        "test_model_layers",
        config.num_layers as u64,
        params::TEST_MODEL_LAYERS,
    );
    procedure.int(
        "weight_seed",
        ptr_runtime::REFERENCE_KV_WEIGHT_SEED,
        params::WEIGHT_SEED,
    );
    procedure.int(
        "hidden_size",
        config.layer.hidden_size as u64,
        params::HIDDEN_SIZE,
    );
    procedure.int(
        "num_heads",
        config.layer.num_attention_heads as u64,
        params::NUM_HEADS,
    );
    procedure.int(
        "num_kv_heads",
        config.layer.num_key_value_heads as u64,
        params::NUM_KV_HEADS,
    );
    procedure.int(
        "head_dim",
        (config.layer.hidden_size / config.layer.num_attention_heads) as u64,
        params::HEAD_DIM,
    );
    procedure.int("vocab_size", config.vocab_size as u64, params::VOCAB_SIZE);
    procedure.int(
        "corpus_generator_seed",
        crate::experiments::layered_kv::SEQUENCE_SEED,
        params::CORPUS_GENERATOR_SEED,
    );
    procedure.text(
        "tiny_model_config_digest",
        &pins::observed_config_digest(&backend),
        params::TINY_MODEL_CONFIG_DIGEST,
    );
    procedure.text("tokenizer_pin", &tokenizer_text, params::TOKENIZER_PIN);
    procedure.constant("moe_topology", Json::str(params::MOE_TOPOLOGY));
    let kinds: Vec<String> = params::DECLARED_DEPENDENCY_KINDS
        .iter()
        .filter(|kind| observed.declared_kinds.contains(**kind))
        .map(|kind| (*kind).to_string())
        .collect();
    procedure.strs(
        "declared_dependency_kinds",
        &kinds,
        &params::DECLARED_DEPENDENCY_KINDS,
    );
    let modes: Vec<String> = params::PRODUCTION_MODES
        .iter()
        .filter(|mode| observed.production_modes.contains(**mode))
        .map(|mode| (*mode).to_string())
        .collect();
    procedure.strs("production_modes", &modes, &params::PRODUCTION_MODES);
    procedure.constant(
        "derived_state_carriers",
        Json::obj([(
            "secondary_index_entry",
            Json::strs(&params::DERIVED_STATE_CARRIERS_SECONDARY_INDEX_ENTRY),
        )]),
    );
    procedure.int(
        "rolling_window_tokens",
        lab::observed_window()?,
        params::ROLLING_WINDOW_TOKENS,
    );
    procedure.int(
        "reused_segment_tokens",
        observed.segment_tokens.unwrap_or(0),
        shape.segment as u64,
    );
    procedure.int(
        "tail_tokens",
        observed.tail_tokens.unwrap_or(0),
        shape.tail as u64,
    );
    procedure.int(
        "nonaligned_origin_prefix_tokens",
        observed.origin_prefix_tokens.unwrap_or(0),
        shape.origin_prefix as u64,
    );
    procedure.int(
        "harness_max_prompt_tokens",
        backend.limits().max_prompt_tokens as u64,
        shape.max_prompt_tokens as u64,
    );
    procedure.int(
        "harness_max_new_tokens",
        backend.limits().max_new_tokens as u64,
        params::HARNESS_MAX_NEW_TOKENS,
    );
    procedure.constant("base_checkpoint", Json::str(pins::BASE_CHECKPOINT));
    let missing: Vec<&str> = params::PROCEDURE_PARAMETERS_RECORDED_PER_CELL
        .iter()
        .copied()
        .filter(|name| !procedure.has(name))
        .collect();
    if !missing.is_empty() {
        return Err(format!("procedure parameters not recorded: {missing:?}"));
    }
    Ok(procedure)
}

/// Every counter the manifest names starts at zero, so a case that never ran prints a zero
/// (an unmet prerequisite) and not a missing value.
fn seed_counters(tally: &mut Tally, shape: Shape) {
    let shifts: Vec<String> = params::POSITION_SHIFT_TOKENS
        .iter()
        .map(|shift| params::shift_member(*shift))
        .collect();
    let insertions: Vec<String> = params::NONALIGNED_INSERTION_TOKENS
        .iter()
        .map(|tokens| params::insertion_member(*tokens))
        .collect();
    let cells = params::INVALIDATION_COVERAGE_MATRIX_CELLS;
    let inflight = params::inflight_members();
    let _ = shape;
    for name in metrics::METRIC_NAMES {
        if metrics::GROUPED.contains(&name) {
            match name {
                "nonaligned_insertion_cases" => tally.seed_group(name, &insertions),
                "invalidation_matrix_cases" | "invalidation_matrix_stale_hits" => {
                    tally.seed_group(name, &cells)
                }
                "shifted_reuse_inflight_race_cases" => tally.seed_group(name, &inflight),
                _ => tally.seed_group(name, &shifts),
            }
        } else if !matches!(
            name,
            "reuse_latency_ms"
                | "invalidated_case_id_set_digest"
                | "fresh_prefill_completed_case_id_set_digest"
        ) {
            tally.set(name, 0);
        }
    }
}

/// What a run produced, before it is printed.
pub struct Report {
    pub line: String,
    /// The same object as `line`, before it is rendered (read by the tests).
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

fn note(text: &str) -> Json {
    Json::str(text)
}

/// Run every case once for `seed` at `scale` and assemble the one JSON object. A case the
/// harness cannot run is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape();
    if !shape.limits_hold() {
        return Err(format!(
            "max_prompt_tokens {} does not cover the longest prompt {} plus the new tokens",
            shape.max_prompt_tokens,
            shape.longest_prompt()
        ));
    }
    let mut tally = Tally::new(metrics::LIST);
    seed_counters(&mut tally, shape);
    let mut lap = Instant::now();
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };

    let material = Material::new(seed);
    let pins = verify_pins(&mut tally, shape)?;
    let environment = environment(&pins);
    let selection = selection(&mut tally, &material);
    let chosen = chosen_json(seed, scale);
    let anchor = Anchor::new(shape.max_prompt_tokens)?;
    phase("pins_and_selection", &mut lap);

    let mut lab = Lab {
        tally: &mut tally,
        shape,
        material: &material,
        anchor: &anchor,
        timings: Timings::default(),
        ids: CaseIds::default(),
        observed: Observed::default(),
        cases: Vec::new(),
    };
    cases::run_all(&mut lab)?;
    phase("cases", &mut lap);
    matrix::run_all(&mut lab)?;
    phase("matrix", &mut lap);
    let procedure = procedure(&lab, scale)?;
    let Lab {
        timings,
        ids,
        cases,
        ..
    } = lab;
    tally.set("procedure_mismatches", procedure.mismatches);
    tally.set_json(
        "invalidated_case_id_set_digest",
        Json::str(ids.invalidated_digest()),
    );
    tally.set_json(
        "fresh_prefill_completed_case_id_set_digest",
        Json::str(ids.completed_digest()),
    );

    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let mut procedure_pairs = match procedure.json() {
        Json::Obj(pairs) => pairs,
        other => return Err(format!("the procedure record is not an object: {other:?}")),
    };
    if let Json::Obj(extra) = chosen {
        procedure_pairs.extend(extra);
    }
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("R006")),
        (
            "status_note".to_string(),
            note("one seed of the R006 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([
            (
                "elapsed",
                Json::Float(started.elapsed().as_secs_f64() * 1000.0),
            ),
            ("phases", Json::Obj(phases)),
            ("runs", timings.json()),
        ]),
    ));
    pairs.push(("procedure".to_string(), Json::Obj(procedure_pairs)));
    pairs.push(("model_pins".to_string(), pins.json));
    pairs.push(("environment".to_string(), environment));
    pairs.push(("selection".to_string(), selection));
    pairs.push(("cases".to_string(), Json::Obj(cases)));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "staleness",
                note("structural (owner decision 5): a stale hit is a lookup that returned or admitted a run with a dead dependency; an escaped output is tokens emitted, spans repacked or runs inserted after the event step; no output comparison is used, because in one layer a stale page is bit-identical to a fresh one; the controls of cases.C10 show the counters can trip"),
            ),
            (
                "counts",
                note("reused_tokens_per_shift is PositionalReport.reused_tokens (the repacked tokens) and the primary hit is ReferenceKvOutcome.hit_tokens; neither is read from KvReuseReport::hit_tokens(), which adds them; every count is read from the runtime's outcome, never asserted from a constant"),
            ),
            (
                "bitwise_mismatch_count",
                note("per shift member: one per differing KV-state digest, one per differing logits entry (f32 bit patterns), one per differing generated token, one for a differing decode logits digest and one per non-finite logits entry on either side, for the shifted run against the cold baseline, plus the generated tokens and decode digest against the independent one-layer stack; the non-aligned insertions of 1 and 7 tokens count into shift_1 and shift_7"),
            ),
            (
                "evidence_labels",
                note("semantic, memory and derived segments: session_lookup; base model: session_recorded_dependency (every positional session records it); adapter and expert: declared_dependency, runs inserted directly into a standalone catalog under a real ledger and a real expert registry; their in-flight cells exercise the index's pin protocol, not a session, and are not in shifted_reuse_inflight_*"),
            ),
            (
                "weak_metrics",
                note("model_pin_mismatches and procedure_mismatches are weak (checkpoint, topology and generator seed are constants here); fresh_prefill_after_revocation records that a fresh prefill followed the revocation, in one layer there are no layers above 0 to drop; raw_ancestor_omission_cases covers direct Raw segments only; the omission, own-lineage, raw-ancestor and superseded-reachability criteria have no fault-injection control, only the cases that would admit"),
            ),
            (
                "limits",
                note("tiny seeded random one-layer model, CPU reference path, one machine profile, bitwise on f32 bit patterns with no tolerance; reuse_latency_ms is timing only and no rule reads it"),
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

/// Run the experiment's harness for `seed` and print its one JSON object. Exits 1 when the
/// run is not clean and 2 when the harness cannot run a case.
pub fn run(seed: u64) {
    let seeds = params::declared_seeds();
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

    const DRAFT: &str = include_str!(
        "../../../../../experiments/runtime/R006-positional-reuse-oracle/DRAFT_PREREGISTRATION.md"
    );
    const CRITERIA: &str = include_str!(
        "../../../../../experiments/runtime/R006-positional-reuse-oracle/criteria.toml"
    );

    /// The shared reduced run (every case and all 52 cells at the reduced scale; a few tens of
    /// seconds in a debug build).
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| {
            let report = execute(17, Scale::Reduced).expect("a reduced run");
            // For the reader of a failing test: `PTR_R006_DUMP=<path>` keeps the JSON.
            if let Ok(path) = std::env::var("PTR_R006_DUMP") {
                std::fs::write(path, &report.line).expect("the dump is writable");
            }
            report
        })
    }

    fn int(json: &Json, path: &str) -> i64 {
        match emit::resolve(json, path) {
            Some(Json::Int(value)) => *value,
            other => panic!("{path} is {other:?}"),
        }
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "positional-reuse-oracle");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert!(DRAFT.contains(params::ENTRYPOINT));
        assert!(params::ENTRYPOINT.contains(params::BENCHMARK));
    }

    #[test]
    fn the_draft_quotes_the_owner_decisions_verbatim_and_pins_the_selection_digest() {
        for decision in [
            params::OWNER_DECISION_2026_10_08_1_GRANULARITY,
            params::OWNER_DECISION_2026_10_08_2_MATRIX,
            params::OWNER_DECISION_2026_10_08_3_REFUSALS,
            params::OWNER_DECISION_2026_10_08_4_GO,
            params::OWNER_DECISION_2026_10_08_5_STALENESS,
            params::OWNER_DECISION_2026_10_08_6_MULTISET,
            params::OWNER_DECISION_2026_10_08_7_REUSE_UNIT,
            params::OWNER_DECISION_2026_10_08_8_DECLARED_AND_SNAPSHOT,
            params::OWNER_DECISION_2026_10_08_9_CONFIG,
        ] {
            assert!(DRAFT.contains(decision), "{decision}");
        }
        assert!(DRAFT.contains(params::SELECTION_DIGEST));
        assert!(DRAFT.contains(params::TINY_MODEL_CONFIG_DIGEST));
        assert!(DRAFT.contains("not frozen"));
    }

    #[test]
    fn the_draft_maps_every_metric_to_its_cases() {
        for name in metrics::METRIC_NAMES {
            assert!(DRAFT.contains(&format!("`{name}`")), "{name}");
        }
        for (case, _) in metrics::CASE_MAP {
            assert!(DRAFT.contains(&format!("| {case} ")), "{case}");
        }
    }

    #[test]
    fn the_matrix_has_the_52_cells_of_the_config_in_its_order() {
        let cells = matrix::cell_names();
        assert_eq!(cells.len(), 52);
        let unique: std::collections::BTreeSet<&str> = cells.iter().copied().collect();
        assert_eq!(unique.len(), 52);
    }

    #[test]
    fn the_scales_are_named_and_only_the_pinned_one_has_the_pinned_shape() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Reduced.name(), "reduced");
        assert_eq!(Scale::Pinned.shape(), lab::PINNED_SHAPE);
        assert_ne!(Scale::Reduced.shape(), lab::PINNED_SHAPE);
        // The harness limits are the config's and cover the longest prompt.
        assert!(lab::PINNED_SHAPE.limits_hold());
        assert_eq!(lab::PINNED_SHAPE.longest_prompt(), 4096 + 48 + 16);
        assert!(lab::REDUCED_SHAPE.limits_hold());
        let smaller = Shape {
            max_prompt_tokens: 4167,
            ..lab::PINNED_SHAPE
        };
        assert!(!smaller.limits_hold(), "4167 < 4160 + 8");
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        let rules = |key: &str| -> Vec<String> {
            table[key]
                .as_array()
                .expect("an array")
                .iter()
                .map(|rule| rule.as_str().expect("a rule is a string").to_string())
                .collect()
        };
        assert_eq!(rules("coverage"), metrics::coverage());
        assert_eq!(rules("falsifiers"), metrics::falsifiers());
        assert_eq!(rules("coverage").len(), 93);
        assert_eq!(rules("falsifiers").len(), 68);
    }

    /// Every path a criterion names resolves to an integer (or a digest string, for the one
    /// equality rule) in the JSON the harness emits for a seed.
    #[test]
    fn every_criteria_path_resolves_to_an_emitted_member() {
        let report = reduced_report();
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        let mut checked = 0;
        for key in ["coverage", "falsifiers"] {
            for rule in table[key].as_array().expect("an array") {
                let rule = rule.as_str().expect("a rule is a string");
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
                            matches!(
                                emit::resolve(&report.json, token),
                                Some(Json::Int(_) | Json::Str(_))
                            ),
                            "{rule}: {token} does not resolve to an integer or a digest in the emitted JSON"
                        );
                        checked += 1;
                    }
                }
            }
        }
        // 93 + 68 rules: 161 left sides and the two right-hand references.
        assert_eq!(checked, 161 + 2);
    }

    /// Every falsifier (and coverage) rule can fail: mutate the emitted metric it reads so
    /// that it violates the rule, and exactly that rule must fail on the emitted result.
    #[test]
    fn every_rule_can_fail_on_the_emitted_result_and_fails_alone() {
        fn set(root: &mut Json, path: &str, value: Json) {
            let Json::Obj(pairs) = root else {
                panic!("not an object at {path}");
            };
            match path.split_once('.') {
                None => {
                    let slot = pairs
                        .iter_mut()
                        .find(|(key, _)| key == path)
                        .unwrap_or_else(|| panic!("{path} is not emitted"));
                    slot.1 = value;
                }
                Some((head, rest)) => {
                    let slot = pairs
                        .iter_mut()
                        .find(|(key, _)| key == head)
                        .unwrap_or_else(|| panic!("{head} is not emitted"));
                    set(&mut slot.1, rest, value);
                }
            }
        }
        let report = reduced_report();
        let rules: Vec<String> = metrics::coverage()
            .into_iter()
            .chain(metrics::falsifiers())
            .collect();
        for rule in &rules {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let (left, op, right) = (tokens[0], tokens[1], tokens[2]);
            let mut mutated = report.json.clone();
            let current = emit::resolve(&mutated, left).cloned().expect("emitted");
            let broken = match (&current, op) {
                (Json::Str(_), "==") => Json::str("sha256:mutated"),
                (Json::Int(_), ">=") => {
                    let bound = right
                        .parse::<i64>()
                        .unwrap_or_else(|_| int(&report.json, right));
                    Json::Int(bound - 1)
                }
                (Json::Int(value), "<=") => {
                    Json::Int(value.max(&right.parse::<i64>().unwrap()) + 1)
                }
                other => panic!("{rule}: {other:?}"),
            };
            set(&mut mutated, left, broken);
            let failing: Vec<String> = metrics::coverage()
                .into_iter()
                .chain(metrics::falsifiers())
                .filter(|candidate| emit::evaluate(&mutated, candidate) == Ok(false))
                .collect();
            let baseline: Vec<String> = metrics::coverage()
                .into_iter()
                .chain(metrics::falsifiers())
                .filter(|candidate| emit::evaluate(&report.json, candidate) == Ok(false))
                .collect();
            let mut expected = baseline.clone();
            if !expected.contains(rule) {
                expected.push(rule.clone());
            }
            let mut got = failing.clone();
            got.sort();
            expected.sort();
            // A rule reading the mutated member fails, and any other rule reading it too
            // (a counter shared by two rules); no unrelated rule changes.
            assert!(failing.contains(rule), "{rule} did not fail when mutated");
            for other in &failing {
                let other_left = other.split(' ').next().unwrap();
                let other_right = other.split(' ').nth(2).unwrap();
                assert!(
                    baseline.contains(other) || other_left == left || other_right == left,
                    "mutating {left} also failed {other}"
                );
            }
        }
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"positional-reuse-oracle\",\"experiment\":\"R006\""));
        for name in metrics::METRIC_NAMES {
            assert!(report.line.contains(&format!("\"{name}\":")), "{name}");
        }
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        assert_eq!(int(&report.json, "seed"), 17);
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "reduced"
        ));
        // Every metric is an integer, an object, or a digest: no null.
        for name in metrics::METRIC_NAMES {
            let value = emit::resolve(&report.json, name).expect(name);
            assert!(!matches!(value, Json::Null), "{name}");
        }
    }

    /// The reduced run holds every coverage rule and trips no falsifier except the ones the
    /// reduced scale itself marks (a clean pass: exit status 0).
    #[test]
    fn a_reduced_run_holds_every_coverage_rule_and_trips_no_falsifier() {
        let report = reduced_report();
        let unmet: Vec<&str> = report
            .coverage
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(unmet.is_empty(), "{unmet:?}");
        let tripped: Vec<&str> = report
            .falsifiers
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(tripped.is_empty(), "{tripped:?}");
        assert!(report.clean(), "{}", report.line);
        assert_eq!(int(&report.json, "bitwise_mismatch_count.shift_7"), 0);
        assert_eq!(int(&report.json, "procedure_mismatches"), 0);
        assert_eq!(int(&report.json, "model_pin_mismatches"), 0);
        assert_eq!(int(&report.json, "selection_digest_mismatches"), 0);
        assert!(int(&report.json, "internal.negative_control_bypass_admissions") > 0);
        assert!(
            int(
                &report.json,
                "internal.negative_control_postrope_mismatches"
            ) > 0
        );
    }

    #[test]
    fn the_case_id_sets_are_equal_and_not_empty() {
        let json = &reduced_report().json;
        let (invalidated, completed) = (
            emit::resolve(json, "invalidated_case_id_set_digest").cloned(),
            emit::resolve(json, "fresh_prefill_completed_case_id_set_digest").cloned(),
        );
        assert_eq!(invalidated, completed);
        let empty = crate::experiments::layered_kv::lab::id_set_digest(std::iter::empty());
        assert_ne!(invalidated, Some(Json::str(empty)));
    }

    #[test]
    fn the_matrix_has_52_cells_with_the_evidence_labels_of_the_config() {
        let json = &reduced_report().json;
        let Some(Json::Arr(cells)) = emit::resolve(json, "cases.C9.cells") else {
            panic!("no cells");
        };
        assert_eq!(cells.len(), 52);
        let mut labels = std::collections::BTreeMap::<String, usize>::new();
        for cell in cells {
            let Some(Json::Str(evidence)) = emit::resolve(cell, "evidence") else {
                panic!("a cell without evidence");
            };
            *labels.entry(evidence.clone()).or_default() += 1;
        }
        assert_eq!(labels["session_lookup"], 24);
        assert_eq!(labels["session_recorded_dependency"], 8);
        assert_eq!(labels["declared_dependency"], 20);
    }

    #[test]
    fn the_procedure_records_every_parameter_the_config_lists() {
        let procedure = emit::resolve(&reduced_report().json, "procedure").expect("procedure");
        for name in params::PROCEDURE_PARAMETERS_RECORDED_PER_CELL {
            assert!(emit::resolve(procedure, name).is_some(), "{name}");
        }
        assert!(matches!(
            emit::resolve(procedure, "differing"),
            Some(Json::Arr(items)) if items.is_empty()
        ));
    }

    #[test]
    fn the_model_pins_hold_with_the_raised_limits() {
        let mut tally = Tally::new(metrics::LIST);
        let pins = verify_pins(&mut tally, lab::PINNED_SHAPE).unwrap();
        assert_eq!(
            tally.get("model_pin_mismatches"),
            0,
            "{}",
            pins.json.render()
        );
    }

    #[test]
    fn timings_are_in_their_own_fields_and_no_rule_reads_them() {
        let json = &reduced_report().json;
        assert!(matches!(
            emit::resolve(json, "timing_ms.elapsed"),
            Some(Json::Float(_))
        ));
        assert!(matches!(
            emit::resolve(json, "reuse_latency_ms.shift_7"),
            Some(Json::Float(_))
        ));
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("latency"), "{rule}");
        }
    }

    /// The full R006 shape for one seed: about two minutes in release and far longer in a
    /// debug build, so it is opt-in: `cargo test --release -p ptr-bench -- --ignored r006`.
    #[test]
    #[ignore = "minutes in release; run with --ignored"]
    fn r006_the_pinned_shape_holds_every_rule() {
        let report = execute(17, Scale::Pinned).expect("a pinned run");
        assert!(report.clean(), "{}", report.line);
        assert_eq!(int(&report.json, "reused_tokens_per_shift.shift_4096"), 48);
    }
}
