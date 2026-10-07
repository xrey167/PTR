//! Experiment R012, containment of Approximate-KV taint, as a harness: `ptr-bench
//! approximate-taint-containment <seed>` runs the nine cases C1 to C9 of the experiment's
//! preregistration once for one seed and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`; a run at a reduced scale (the smoke test) is no evidence. The code
//! under test is `PtrRuntime` (its taint registry, derivation hops, Raw rule, Pod, SemDB and
//! merge containment) over a durable `FileLedger`, driven only through its public API. The
//! Approximate producer is the harness-only stub of [`stub`], which runs a genuine cold prefill
//! of the R005 tiny model and labels the report; this experiment tests containment of a
//! labelled Approximate result, not any real Approximate producer.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names
//! (a number, or `null` with its reason under `unsupported`), the procedure parameters, the
//! model pins, the producer environment, the corpus selection digest, the stub, the per-case
//! counters, the sub-cases with their violations, the two judgements (`coverage` and
//! `falsifiers`), and the counters that belong to no metric (`internal`). The exit status is 1
//! when a coverage prerequisite is unmet, a falsifier is tripped or a metric has no value and
//! no declared reason, and 2 when the harness itself cannot run a case (a refusal that is not
//! the expected typed error, a fixture that does not build, a control that fails).

pub mod cases;
pub mod metrics;
pub mod params;
pub mod stub;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::layered_kv::lab::Res;
use super::layered_kv::pins::{self, PinChecks};
use super::layered_kv::Scale;
use cases::Cx;
use std::time::Instant;

/// Counters whose natural value is zero (violations, mismatches): a run in which none happens
/// still prints a zero. A counter of attempts or cases is never seeded, so a case that never
/// ran is an unmet prerequisite, not a silent zero.
pub(crate) const ZERO_BASED: [&str; 11] = [
    "approximate_run_actions_admitted",
    "approximate_semdb_promotions",
    "approximate_candidate_exact_admissions",
    "taint_lost",
    "approximate_pod_result_promotions_accepted",
    "misclassified_raw_admissions",
    "laundered_exact_admissions",
    "taint_cleared_by_echo",
    "selection_digest_mismatches",
    "procedure_mismatches",
    "model_pin_mismatches",
];

fn selection(tally: &mut Tally) -> Json {
    let computed = cases::selection_digest();
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
        ("sequences", Json::Int(params::SELECTION_SEQUENCES as i64)),
        (
            "prompt_text",
            Json::str(
                "token t of a sequence is the character '!' + (t % 94); a prompt is that text",
            ),
        ),
    ])
}

/// The procedure parameters of the run, as `procedure_parameters_recorded_per_cell` lists
/// them (each case also checks what it observed, and a mismatch is counted there).
fn procedure_json(seed: u64, scale: Scale) -> Json {
    let seeds = params::declared_seeds();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        ("seeds", Json::ints(seeds.iter().map(|seed| *seed as i64))),
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
            "tiny_model_config_digest",
            Json::str(params::TINY_MODEL_CONFIG_DIGEST),
        ),
        ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
        ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
        (
            "chosen",
            Json::obj([
                ("max_new_tokens", Json::Int(params::MAX_NEW_TOKENS as i64)),
                ("prompt_tokens", Json::Int(params::PROMPT_TOKENS as i64)),
                ("segment_tokens", Json::Int(params::SEGMENT_TOKENS as i64)),
                ("hops_per_chain", Json::Int(params::HOPS_PER_CHAIN as i64)),
                (
                    "max_live_taint_subjects",
                    Json::Int(params::MAX_LIVE_TAINT_SUBJECTS as i64),
                ),
                ("per_seed_role", Json::str(params::PER_SEED_ROLE)),
                (
                    "ledger",
                    Json::str("FileLedger in a temporary directory, one per case"),
                ),
                (
                    "catalog_limits",
                    Json::str(
                        "not used: the stub runs the backend's cold path (no lookup, no insert)",
                    ),
                ),
            ]),
        ),
    ])
}

/// The model pins: the dense checks of the backend (digest, tokenizer, shape, chain id, cold
/// run reproduced by an independent stack) and the stub's (Approximate, same content, identity).
fn verify_pins(
    tally: &mut Tally,
    backend: &ptr_runtime::ReferenceKvReuseBackend,
) -> Res<pins::Pins> {
    let mut checks = PinChecks::default();
    let build = pins::base_checks(
        &mut checks,
        backend,
        &pins::canonical_digest(&pins::shape_fields()),
        params::TINY_MODEL_CONFIG_DIGEST,
        params::TOKENIZER_PIN,
    )?;
    cases::stub_pin_checks(&mut checks, backend)?;
    Ok(checks.finish(
        tally,
        Json::obj([
            (
                "tiny_model_config_digest",
                Json::str(params::TINY_MODEL_CONFIG_DIGEST),
            ),
            ("tokenizer_pin", Json::str(params::TOKENIZER_PIN)),
            ("moe_topology", Json::str(params::MOE_TOPOLOGY)),
            ("weights", Json::str(params::WEIGHTS)),
            ("weight_seed", Json::Int(params::WEIGHT_SEED as i64)),
            ("base_checkpoint", Json::str(params::BASE_CHECKPOINT)),
            (
                "approximate_producer",
                Json::str(params::APPROXIMATE_PRODUCER),
            ),
        ]),
        pins::chosen_json(),
        build,
    ))
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

/// Run every case once for `seed` at `scale` and assemble the one JSON object. A case the
/// harness cannot run is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let mut cx = Cx::new(seed, scale)?;
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut lap = Instant::now();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };
    let pins = verify_pins(&mut cx.tally, &cx.backend)?;
    let environment = pins::environment(&pins);
    let selection = selection(&mut cx.tally);
    let procedure = procedure_json(seed, scale);
    phase("pins_and_selection", &mut lap);

    let case_order = cases::run_all(&mut cx)?;
    phase("cases", &mut lap);

    // The chain must reach the preregistered hop count at the pinned scale unless a measured
    // violation broke it (then the falsifier says so).
    if scale == Scale::Pinned
        && (cx.tally.get("candidate_taint_hops") as u64) < params::HOPS_PER_CHAIN
        && cx.tally.get("approximate_candidate_exact_admissions") == 0
        && cx.tally.get("hop_refused_unknown") == 0
    {
        return Err(format!(
            "candidate_taint_hops {} is below hops_per_chain {}",
            cx.tally.get("candidate_taint_hops"),
            params::HOPS_PER_CHAIN
        ));
    }

    let (printed, gaps) = cx.tally.metrics();
    let unsupported_names = cx.tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("R012")),
        (
            "status_note".to_string(),
            Json::str("one seed of the R012 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
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
    pairs.push((
        "stub".to_string(),
        Json::obj([
            ("name", Json::str(params::STUB_NAME)),
            ("version", Json::str(params::STUB_VERSION)),
            ("identity", Json::str(stub::identity())),
            ("mode", Json::str("Approximate")),
            ("policy", Json::str("AllowApproximate")),
            (
                "content",
                Json::str("a genuine cold prefill of the R005 tiny model (ReferenceKvReuseBackend::infer_cold: no lookup, no insert); the Approximate label is the stub's"),
            ),
            (
                "generation_parameters",
                Json::str(params::GENERATION_PARAMETERS),
            ),
            (
                "controls",
                Json::str("Cold: no report; ExactWarm: an Exact report with a hit of one chunk over the same genuine content"),
            ),
        ]),
    ));
    pairs.push(("case_order".to_string(), Json::Arr(case_order)));
    pairs.push((
        "prompt_indexes".to_string(),
        Json::obj([
            ("C1", Json::ints(cx.prompt_indexes(0))),
            ("C2", Json::ints(cx.prompt_indexes(1))),
            ("C3", Json::ints(cx.prompt_indexes(2))),
            ("C4", Json::ints(cx.prompt_indexes(3))),
            ("C5", Json::ints(cx.prompt_indexes(4))),
            ("C6", Json::ints(cx.prompt_indexes(5))),
            ("C7", Json::ints(cx.prompt_indexes(6))),
            ("C8", Json::ints(cx.prompt_indexes(7))),
            ("C9", Json::ints(cx.prompt_indexes(8))),
        ]),
    ));
    pairs.push(("cases".to_string(), Json::Obj(cx.cases.clone())));
    pairs.push(("subcases".to_string(), cx.subcases_json()));
    pairs.push((
        "wp4b_dependent_subcases".to_string(),
        Json::strs(&cases::WP4B_SUBCASES),
    ));
    pairs.push(("unsupported".to_string(), cx.tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "attempts",
                Json::str(params::ATTEMPT_DEFINITION),
            ),
            (
                "hops",
                Json::str(params::HOP_DEFINITION),
            ),
            (
                "candidate_taint_hops",
                Json::str("the longest chain, over C3 and its restarts, in which every hop's output stayed Approximate and its action stayed refused at authorize_action; hop 2 carries an Exact warm report (a stand-in over the same genuine content), hops 1 and 3 are cold; the runtime is reopened after hop 1 and compacted after hop 2"),
            ),
            (
                "taint_lost",
                Json::str("a persisted taint subject (the action, the candidate segment, its text) that reads Exact after a reopen or a compaction restore, plus any negative recompute control that cleared; weak: it compares with what this harness itself persisted, so a lying backend report or a tampered log is not covered"),
            ),
            (
                "model_pin_mismatches",
                Json::str("weak: base checkpoint, tokenizer revision and topology are constants here, so it checks the tiny-model config digest, the tokenizer, the shape, the chain id, the backend's and the stub's cold run against an independent stack, and the stub's identity and label"),
            ),
            (
                "misclassified_revocable_as_raw_attempts",
                Json::str("counts the ingress-equality rule (a Raw segment is admitted only when it equals the request's own ingress text); each presentation is tried through admit_segments and through a derivation hop"),
            ),
            (
                "echo",
                Json::str("an echo is cleared if a Clear record is committed, the tainted action is authorized, the tainted derived segment is admitted Exact, or the echo's own output is Exact"),
            ),
            (
                "limits",
                Json::str("exact whole-segment digest only: a paraphrase, substring or whitespace edit of tainted text is not detected (limit_whitespace_edit_detected under internal records the probe and is no pass); the backend's report is trusted; the Exact-only catalog guard is structural"),
            ),
        ]),
    ));
    pairs.push(("internal".to_string(), cx.tally.internal()));
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

/// A backend for the tests that need only the pins.
#[cfg(test)]
fn backend() -> ptr_runtime::ReferenceKvReuseBackend {
    crate::experiments::layered_kv::lab::backend().expect("the reference backend")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/runtime/R012-approximate-taint-containment/PREREGISTRATION.md"
    );
    const CRITERIA: &str = include_str!(
        "../../../../../experiments/runtime/R012-approximate-taint-containment/criteria.toml"
    );

    /// The shared reduced run (every case once at the reduced scale; a few seconds in a debug
    /// build).
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

    fn int(json: &Json, path: &str) -> i64 {
        match emit::resolve(json, path) {
            Some(Json::Int(value)) => *value,
            other => panic!("{path} is {other:?}"),
        }
    }

    #[test]
    fn the_metric_names_are_the_manifests() {
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let listed: Vec<&str> = manifest["metrics"]
            .as_str()
            .expect("metrics is a string")
            .split(", ")
            .collect();
        assert_eq!(listed, metrics::METRIC_NAMES);
    }

    #[test]
    fn the_pinned_selection_digest_is_what_the_corpus_generator_produces() {
        assert_eq!(cases::selection_digest(), params::SELECTION_DIGEST);
        assert!(PREREGISTRATION.contains(params::SELECTION_DIGEST));
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("R012"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some("every_seed_must_pass")
        );
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
                        matches!(emit::resolve(&report.json, token), Some(Json::Int(_))),
                        "{rule}: {token} does not resolve to an integer in the harness JSON"
                    );
                }
            }
        }
    }

    #[test]
    fn the_rules_mirror_the_falsification_text_of_the_manifest() {
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let text = manifest["falsification"].as_str().expect("falsification");
        // Every metric the text bounds with "< N" or "> 0" has a rule on it.
        for rule in metrics::coverage() {
            let metric = rule.split(' ').next().unwrap();
            assert!(
                text.contains(metric),
                "{metric} is not in the falsification text"
            );
        }
        for rule in metrics::falsifiers() {
            let metric = rule.split(' ').next().unwrap();
            assert!(
                text.contains(&format!("{metric} > 0")),
                "{metric} > 0 is not in the falsification text"
            );
        }
        // Every "<metric> > 0" and "<metric> < N" of the text has a rule.
        let ruled: Vec<String> = metrics::coverage()
            .into_iter()
            .chain(metrics::falsifiers())
            .map(|rule| rule.split(' ').next().unwrap().to_string())
            .collect();
        for name in metrics::METRIC_NAMES {
            if text.contains(&format!("{name} >")) || text.contains(&format!("{name} <")) {
                assert!(ruled.iter().any(|ruled| ruled == name), "{name}");
            }
        }
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "approximate-taint-containment");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"approximate-taint-containment\""));
        for name in metrics::METRIC_NAMES {
            assert!(report.line.contains(&format!("\"{name}\":")), "{name}");
        }
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        assert_eq!(int(&report.json, "seed"), 17);
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "reduced"
        ));
    }

    /// The reduced run holds every coverage rule, trips no falsifier except in the sub-cases
    /// that depend on WP4b of the runtime (`cases::WP4B_SUBCASES`; once the runtime has
    /// that behaviour the list is emptied and this test demands a clean pass, exit status 0).
    #[test]
    fn a_reduced_run_holds_every_coverage_rule_and_trips_no_falsifier_outside_the_wp4b_subcases() {
        let report = reduced_report();
        let unmet: Vec<&str> = report
            .coverage
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(unmet.is_empty(), "{unmet:?}");
        for name in [
            "approximate_action_admission_attempts",
            "approximate_semdb_promotion_attempts",
            "approximate_candidate_path_attempts",
            "tainted_candidate_reload_attempts",
            "fresh_exact_recomputation_controls",
            "approximate_pod_result_promotion_attempts",
            "misclassified_revocable_as_raw_attempts",
            "approximate_output_reintroduced_as_raw_attempts",
            "exact_echo_cases",
        ] {
            assert!(int(&report.json, name) >= 1, "{name}");
        }
        assert!(int(&report.json, "candidate_taint_hops") >= 2);
        let Some(Json::Obj(subcases)) = emit::resolve(&report.json, "subcases") else {
            panic!("subcases");
        };
        let outside: Vec<String> = subcases
            .iter()
            .filter(|(label, _)| !cases::WP4B_SUBCASES.contains(&label.as_str()))
            .filter(|(_, value)| int(value, "violations") > 0)
            .map(|(label, _)| label.clone())
            .collect();
        assert!(
            outside.is_empty(),
            "violations outside the WP4b sub-cases: {outside:?}"
        );
        if cases::WP4B_SUBCASES.is_empty() {
            assert!(report.clean(), "{}", report.line);
        }
    }

    #[test]
    fn the_procedure_records_every_parameter_the_config_lists() {
        let procedure = emit::resolve(&reduced_report().json, "procedure").expect("procedure");
        for name in params::PROCEDURE_PARAMETERS_RECORDED_PER_CELL {
            assert!(emit::resolve(procedure, name).is_some(), "{name}");
        }
    }

    #[test]
    fn the_stub_is_recorded_with_its_pinned_name_and_version() {
        let json = &reduced_report().json;
        for (path, expected) in [
            ("stub.name", "r012-stub-approximate-producer"),
            ("stub.version", "1"),
            ("stub.mode", "Approximate"),
        ] {
            assert!(
                matches!(emit::resolve(json, path), Some(Json::Str(text)) if text == expected),
                "{path}"
            );
        }
    }

    #[test]
    fn the_model_pins_hold() {
        let mut tally = Tally::new(metrics::LIST);
        let pins = verify_pins(&mut tally, &backend()).unwrap();
        assert_eq!(
            tally.get("model_pin_mismatches"),
            0,
            "{}",
            pins.json.render()
        );
    }

    #[test]
    fn every_case_ran_and_is_reported_with_its_counters() {
        let json = &reduced_report().json;
        for case in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"] {
            assert!(
                matches!(
                    emit::resolve(json, &format!("cases.{case}")),
                    Some(Json::Obj(_))
                ),
                "{case}"
            );
        }
    }
}
