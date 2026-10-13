//! Experiment R011, self-modification safety of the expert registry, as a harness: `ptr-bench
//! self-modification-safety <seed>` runs the thirteen cases C1 to C13 of the experiment's
//! preregistration once for one seed and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`; a run at a reduced scale (the smoke test) is no evidence. The code
//! under test is `PtrRuntime` (its expert registry, the gate evidence binding, the commit-time
//! authority, the dependency index with the compound `InputInvalidated` event, the authority
//! epoch and the use-time `expert_route_admission`) over a durable `FileLedger`, driven only
//! through its public API, and, for the in-flight cases, the R010 `KvSession` on the tiny
//! seeded MoE. There is no router and no training; "route" is a probe of
//! `expert_route_admission`, and "direct append" is the public `PtrRuntime::commit` surface
//! (no claim against a raw ledger writer).
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names
//! (an integer, an object of integers whose `total` is its scalar aggregate, or a digest
//! string), the procedure parameters, the model pins, the producer environment, the corpus
//! selection digest, the fixture choices the seed made, the per-case counters, the sub-cases
//! with their violations, the two judgements (`coverage` and `falsifiers`), and the counters
//! that belong to no metric (`internal`). The exit status is 1 when a coverage prerequisite is
//! unmet, a falsifier is tripped or a metric has no value and no declared reason, and 2 when
//! the harness itself cannot run a case (a refusal that is not the expected typed error, a
//! fixture that does not build, a control that fails).

pub mod cases;
pub mod metrics;
pub mod params;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status};
use super::layered_kv::pins as model_pins;
use super::layered_kv::Scale;
use cases::Cx;
use std::time::Instant;

fn selection(cx: &Cx) -> Json {
    let computed = cases::selection_digest();
    Json::obj([
        ("corpus", Json::str(params::CORPUS)),
        ("rule", Json::str(params::CORPUS_SELECTION_RULE)),
        ("domain", Json::str(params::SELECTION_DOMAIN)),
        ("computed", Json::str(computed.clone())),
        ("pinned", Json::str(params::SELECTION_DIGEST)),
        ("matches", Json::Bool(computed == params::SELECTION_DIGEST)),
        ("cases", Json::Int(params::ADVERSARIAL_CASES.len() as i64)),
        (
            "cases_attempted",
            Json::Int(
                cx.tally
                    .group("adversarial_attempts_per_case")
                    .iter()
                    .filter(|(_, attempts)| *attempts > 0)
                    .count() as i64,
            ),
        ),
        (
            "sequence_of_a_case",
            Json::str("the bytes of the case name, in listed order"),
        ),
    ])
}

/// The procedure parameters of the run, as `procedure_parameters_recorded_per_cell` lists them
/// (each case also checks what it observed, and a mismatch is counted there).
fn procedure_json(seed: u64, scale: Scale) -> Json {
    let seeds = params::declared_seeds();
    let pairs: Vec<String> = params::INFLIGHT_INPUT_RACE_CELLS
        .iter()
        .map(|cell| (*cell).to_string())
        .collect();
    Json::Obj(vec![
        ("scale".to_string(), Json::str(scale.name())),
        ("seed".to_string(), Json::Int(seed as i64)),
        (
            "seed_is_declared".to_string(),
            Json::Bool(seeds.contains(&seed)),
        ),
        (
            "seeds".to_string(),
            Json::ints(seeds.iter().map(|seed| *seed as i64)),
        ),
        (
            "test_model_layers".to_string(),
            Json::Int(params::TEST_MODEL_LAYERS as i64),
        ),
        (
            "weight_seed".to_string(),
            Json::Int(params::WEIGHT_SEED as i64),
        ),
        (
            "hidden_size".to_string(),
            Json::Int(params::HIDDEN_SIZE as i64),
        ),
        ("num_heads".to_string(), Json::Int(params::NUM_HEADS as i64)),
        (
            "num_kv_heads".to_string(),
            Json::Int(params::NUM_KV_HEADS as i64),
        ),
        ("head_dim".to_string(), Json::Int(params::HEAD_DIM as i64)),
        (
            "vocab_size".to_string(),
            Json::Int(params::VOCAB_SIZE as i64),
        ),
        (
            "adversarial_cases".to_string(),
            Json::strs(&params::ADVERSARIAL_CASES),
        ),
        (
            "min_attempts_per_case".to_string(),
            Json::Int(params::MIN_ATTEMPTS_PER_CASE as i64),
        ),
        (
            "inflight_expert_call_levels".to_string(),
            Json::strs(&params::INFLIGHT_EXPERT_CALL_LEVELS),
        ),
        (
            "structural_dependency_kinds".to_string(),
            Json::strs(&params::STRUCTURAL_DEPENDENCY_KINDS),
        ),
        (
            "invalidation_types".to_string(),
            Json::strs(&params::INVALIDATION_TYPES),
        ),
        ("input_kinds".to_string(), Json::strs(&params::INPUT_KINDS)),
        (
            "multi_operand_op_kinds".to_string(),
            Json::strs(&params::MULTI_OPERAND_OP_KINDS),
        ),
        (
            "direct_append_event_types".to_string(),
            Json::strs(&params::DIRECT_APPEND_EVENT_TYPES),
        ),
        (
            "gate_policy_epoch_items".to_string(),
            Json::strs(&params::GATE_POLICY_EPOCH_ITEMS),
        ),
        (
            "cas_operand_fields".to_string(),
            Json::strs(&params::CAS_OPERAND_FIELDS),
        ),
        (
            "invalidation_coverage_matrix".to_string(),
            Json::obj([
                (
                    "kinds",
                    Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_KINDS),
                ),
                (
                    "types",
                    Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_TYPES),
                ),
                (
                    "paths",
                    Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_PATHS),
                ),
                (
                    "authority_kinds",
                    Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_AUTHORITY_KINDS),
                ),
                (
                    "authority_types",
                    Json::strs(&params::INVALIDATION_COVERAGE_MATRIX_AUTHORITY_TYPES),
                ),
                (
                    "cells",
                    Json::Int(params::INVALIDATION_COVERAGE_MATRIX_CELLS.len() as i64),
                ),
            ]),
        ),
        ("inflight_input_race_cells".to_string(), Json::strs(&pairs)),
        (
            "impersonation_op_kinds".to_string(),
            Json::strs(&params::IMPERSONATION_OP_KINDS),
        ),
        (
            "producing_op_kinds".to_string(),
            Json::strs(&params::PRODUCING_OP_KINDS),
        ),
        (
            "result_identity_cas_covers".to_string(),
            Json::strs(&params::RESULT_IDENTITY_CAS_COVERS),
        ),
        (
            "authority_kinds".to_string(),
            Json::strs(&params::AUTHORITY_KINDS),
        ),
        (
            "takeover_kinds".to_string(),
            Json::strs(&params::TAKEOVER_KINDS),
        ),
        (
            "tightening_rotation_definition".to_string(),
            Json::str(params::TIGHTENING_ROTATION_DEFINITION),
        ),
        (
            "same_project_impersonation_op_kinds".to_string(),
            Json::strs(&params::SAME_PROJECT_IMPERSONATION_OP_KINDS),
        ),
        (
            "fence_cause_types".to_string(),
            Json::strs(&params::FENCE_CAUSE_TYPES),
        ),
        (
            "proposer_grant_op_kinds".to_string(),
            Json::strs(&params::PROPOSER_GRANT_OP_KINDS),
        ),
        (
            "gate_host_event_kinds".to_string(),
            Json::strs(&params::GATE_HOST_EVENT_KINDS),
        ),
        (
            "tiny_model_config_digest".to_string(),
            Json::str(params::TINY_MODEL_CONFIG_DIGEST),
        ),
        (
            "tokenizer_pin".to_string(),
            Json::str(params::TOKENIZER_PIN),
        ),
        (
            "moe_topology".to_string(),
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
            "lineage_edge_types".to_string(),
            Json::strs(&params::LINEAGE_EDGE_TYPES),
        ),
        (
            "identity_dependencies".to_string(),
            Json::strs(&params::IDENTITY_DEPENDENCIES),
        ),
        (
            "commit_liveness_cas_covers".to_string(),
            Json::strs(&params::COMMIT_LIVENESS_CAS_COVERS),
        ),
        (
            "chosen".to_string(),
            Json::obj([
                (
                    "ledger",
                    Json::str("FileLedger in a temporary directory, one per case or sub-case"),
                ),
                ("per_seed_role", Json::str(params::PER_SEED_ROLE)),
                (
                    "fanout_experts",
                    Json::Int(match scale {
                        Scale::Pinned => params::FANOUT_EXPERTS,
                        Scale::Reduced => params::FANOUT_EXPERTS_REDUCED,
                    } as i64),
                ),
                ("fanout_id_len", Json::Int(params::FANOUT_ID_LEN as i64)),
                (
                    "inflight_prompt_tokens",
                    Json::Int(params::INFLIGHT_PROMPT_TOKENS as i64),
                ),
                ("max_new_tokens", Json::Int(params::MAX_NEW_TOKENS as i64)),
                (
                    "pre_regime_gate_epoch",
                    Json::Int(params::PRE_REGIME_GATE_EPOCH as i64),
                ),
                (
                    "inflight_layer_j",
                    Json::Int(params::INFLIGHT_LAYER_J as i64),
                ),
            ]),
        ),
    ])
}

/// The object of a family: the members as nested keys (`a.b` is the key `b` of the object `a`)
/// and the scalar aggregate `total`, the sum of every leaf.
fn family_json(members: &[(String, i64)]) -> Json {
    fn insert(level: &mut Vec<(String, Json)>, parts: &[&str], value: i64) {
        let Some((first, rest)) = parts.split_first() else {
            return;
        };
        if rest.is_empty() {
            level.push(((*first).to_string(), Json::Int(value)));
            return;
        }
        let at = level.iter().position(|(name, _)| name == first);
        let index = at.unwrap_or_else(|| {
            level.push(((*first).to_string(), Json::Obj(Vec::new())));
            level.len() - 1
        });
        if let Json::Obj(inner) = &mut level[index].1 {
            insert(inner, rest, value);
        }
    }
    let mut root = Vec::new();
    let mut total = 0;
    for (member, value) in members {
        let parts: Vec<&str> = member.split('.').collect();
        insert(&mut root, &parts, *value);
        total += value;
    }
    root.push(("total".to_string(), Json::Int(total)));
    Json::Obj(root)
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
    let pins = cases::pins::verify_pins(&mut cx)?;
    let environment = model_pins::environment(&pins);
    let procedure = procedure_json(seed, scale);
    phase("pins", &mut lap);

    let case_order = cases::run_all(&mut cx)?;
    phase("cases", &mut lap);

    cases::invalidation::finish_sets(&mut cx);
    let selection = selection(&cx);
    // The objects: every family of cells is printed with its members and its total.
    for name in metrics::NESTED {
        let json = family_json(&cx.tally.group(name));
        cx.tally.set_json(name, json);
    }
    let (printed, gaps) = cx.tally.metrics();
    let unsupported_names = cx.tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("R011")),
        (
            "status_note".to_string(),
            Json::str("one seed of the R011 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
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
    pairs.push(("case_order".to_string(), Json::Arr(case_order)));
    pairs.push(("fixture".to_string(), Json::Obj(cx.fixture.clone())));
    pairs.push(("cases".to_string(), Json::Obj(cx.cases.clone())));
    pairs.push(("subcases".to_string(), cx.subcases_json()));
    pairs.push(("unsupported".to_string(), cx.tally.unsupported_json()));
    pairs.push(("measurement_notes".to_string(), cases::measurement_notes()));
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
        "../../../../../experiments/runtime/R011-self-modification-safety/PREREGISTRATION.md"
    );
    const CRITERIA: &str = include_str!(
        "../../../../../experiments/runtime/R011-self-modification-safety/criteria.toml"
    );
    const README: &str =
        include_str!("../../../../../experiments/runtime/R011-self-modification-safety/README.md");

    /// The shared reduced run (every case once at the reduced scale; a few tens of seconds in a
    /// debug build).
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
        assert_eq!(listed.len(), 131);
    }

    #[test]
    fn the_pinned_selection_digest_is_what_the_suite_produces() {
        assert_eq!(cases::selection_digest(), params::SELECTION_DIGEST);
        assert!(PREREGISTRATION.contains(params::SELECTION_DIGEST));
        // The suite is the 91 cases of the manifest's falsification text, in its order.
        let manifest: toml::Table = params::EXPERIMENT_TOML.parse().expect("experiment.toml");
        let text = manifest["falsification"].as_str().expect("falsification");
        assert_eq!(params::ADVERSARIAL_CASES.len(), 91);
        let start = text.find("uncommitted_change, ").expect("the first case");
        let end = text
            .find(", non_tightening_rotation_control;")
            .expect("the last case");
        let listed: Vec<&str> = text[start..end + ", non_tightening_rotation_control".len()]
            .split(", ")
            .collect();
        assert_eq!(listed, params::ADVERSARIAL_CASES);
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("R011"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some("every_seed_must_pass")
        );
    }

    /// Every counter a rule names resolves in the harness JSON to an integer, or, under `==`
    /// between two references, to a digest string.
    #[test]
    fn every_rule_names_a_path_the_harness_prints() {
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
                if literal {
                    continue;
                }
                match emit::resolve(&report.json, token) {
                    Some(Json::Int(_)) => {}
                    Some(Json::Str(digest)) if tokens[1] == "==" => {
                        assert!(digest.starts_with("sha256:"), "{rule}: {digest}");
                    }
                    other => panic!("{rule}: {token} is {other:?}"),
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
        // The rule names are the manifest's metric names, and the text bounds each of them the
        // way the rule does.
        for rule in metrics::coverage() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let metric = tokens[0].split('.').next().unwrap();
            assert!(metrics::METRIC_NAMES.contains(&metric), "{rule}");
            match tokens[1] {
                ">=" => {
                    // "metric < N" for a scalar, "metric < N" for a family too.
                    assert!(
                        text.contains(&format!("{metric} < ")) || text.contains(metric),
                        "{rule}"
                    );
                }
                "<=" => assert!(text.contains(&format!("{metric} > 0")), "{rule}"),
                _ => assert!(
                    text.contains(&format!("{} != {}", tokens[0], tokens[2])),
                    "{rule}"
                ),
            }
        }
        for rule in metrics::falsifiers() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let metric = tokens[0].split('.').next().unwrap();
            assert!(metrics::METRIC_NAMES.contains(&metric), "{rule}");
            match tokens[1] {
                "<=" => {
                    let scalar = !tokens[0].contains('.');
                    if scalar {
                        // A scalar violation is "metric > 0" in the text, or one of the
                        // families the text words as "any nonzero".
                        assert!(
                            text.contains(&format!("{metric} > 0"))
                                || text.contains(&format!("any nonzero {metric}")),
                            "{rule}"
                        );
                    } else {
                        assert!(text.contains(metric), "{rule}");
                    }
                }
                "==" => assert!(
                    text.contains(&format!("{} != {}", tokens[0], tokens[2])),
                    "{rule}"
                ),
                other => panic!("{rule}: {other}"),
            }
        }
        // Every "<metric> > 0" and "<metric> < N" of the text has a rule.
        let ruled: Vec<String> = metrics::coverage()
            .into_iter()
            .chain(metrics::falsifiers())
            .map(|rule| {
                rule.split(' ')
                    .next()
                    .unwrap()
                    .split('.')
                    .next()
                    .unwrap()
                    .to_string()
            })
            .collect();
        for name in metrics::METRIC_NAMES {
            if text.contains(&format!("{name} >")) || text.contains(&format!("{name} <")) {
                assert!(ruled.iter().any(|ruled| ruled == name), "{name}");
            }
        }
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "self-modification-safety");
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
            .starts_with("{\"benchmark\":\"self-modification-safety\""));
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

    /// Every metric is an integer, an object of integers (with its `total`) or a digest.
    #[test]
    fn every_metric_has_the_shape_the_manifest_promises() {
        let json = &reduced_report().json;
        for name in metrics::METRIC_NAMES {
            let value = emit::resolve(json, name).unwrap_or_else(|| panic!("{name}"));
            match value {
                Json::Int(_) => assert!(!metrics::NESTED.contains(&name), "{name}"),
                Json::Str(digest) => {
                    assert!(metrics::DIGESTS.contains(&name) && digest.starts_with("sha256:"))
                }
                Json::Obj(_) => {
                    assert!(metrics::NESTED.contains(&name), "{name}");
                    assert!(matches!(emit::resolve(value, "total"), Some(Json::Int(_))));
                }
                other => panic!("{name} is {other:?}"),
            }
        }
    }

    /// The reduced run holds every coverage rule and trips no falsifier: a clean pass, exit
    /// status 0, with no violation in any sub-case.
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
        let Some(Json::Obj(subcases)) = emit::resolve(&report.json, "subcases") else {
            panic!("subcases");
        };
        let violating: Vec<String> = subcases
            .iter()
            .filter(|(_, value)| int(value, "violations") > 0)
            .map(|(label, _)| label.clone())
            .collect();
        assert!(
            violating.is_empty(),
            "violations in sub-cases: {violating:?}"
        );
        assert!(report.clean(), "{}", report.line);
    }

    #[test]
    fn the_reduced_fan_out_says_it_is_below_the_record_limit() {
        let json = &reduced_report().json;
        assert_eq!(int(json, "internal.internal_fanout_experts"), 64);
        assert_eq!(
            int(json, "internal.internal_fanout_exceeds_max_record_bytes"),
            0
        );
    }

    #[test]
    fn the_pinned_fan_out_would_not_fit_in_one_record() {
        let member = 8 + params::FANOUT_ID_LEN + 8 + params::FANOUT_ID_LEN + 8 + 8 + 1 + 1;
        assert!(params::FANOUT_EXPERTS * member > params::MAX_RECORD_BYTES as usize);
        assert!(params::FANOUT_EXPERTS_REDUCED * member < params::MAX_RECORD_BYTES as usize);
    }

    #[test]
    fn the_procedure_records_every_parameter_the_config_lists() {
        let procedure = emit::resolve(&reduced_report().json, "procedure").expect("procedure");
        for name in params::PROCEDURE_PARAMETERS_RECORDED_PER_CELL {
            assert!(emit::resolve(procedure, name).is_some(), "{name}");
        }
    }

    #[test]
    fn the_model_pins_hold() {
        let mut cx = Cx::new(17, Scale::Reduced).expect("a context");
        let pins = cases::pins::verify_pins(&mut cx).unwrap();
        assert_eq!(
            cx.tally.get("model_pin_mismatches"),
            0,
            "{}",
            pins.json.render()
        );
        let _ = backend();
    }

    #[test]
    fn every_case_ran_and_is_reported_with_its_counters() {
        let json = &reduced_report().json;
        for case in [
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
        ] {
            assert!(
                matches!(
                    emit::resolve(json, &format!("cases.{case}")),
                    Some(Json::Obj(_))
                ),
                "{case}"
            );
        }
        // The last case in the order is C13, which reads what the others recorded.
        let Some(Json::Arr(order)) = emit::resolve(json, "case_order") else {
            panic!("case_order");
        };
        assert_eq!(order.len(), 13);
        assert_eq!(order.last(), Some(&Json::str("C13")));
    }

    #[test]
    fn the_seed_changes_the_order_and_the_race_points_but_not_the_suite() {
        let a: Vec<usize> =
            crate::experiments::layered_kv::corpus::order(17, params::CASE_ORDER_LABEL, 12);
        let b: Vec<usize> =
            crate::experiments::layered_kv::corpus::order(29, params::CASE_ORDER_LABEL, 12);
        assert_ne!(a, b);
        assert_eq!(cases::selection_digest(), params::SELECTION_DIGEST);
    }

    #[test]
    fn a_case_outside_the_pinned_suite_cannot_be_counted_and_a_wrong_error_is_a_harness_error() {
        let mut cx = Cx::new(17, Scale::Reduced).expect("a context");
        assert!(cx.attempt("not_a_case_of_the_suite").is_err());
        assert!(cx.attempt("uncommitted_change").is_ok());
        // The runtime's answer decides: Ok is a violation, the expected error a refusal, any
        // other error ends the run.
        let admitted = cases::classify("x", Ok::<(), ptr_runtime::RuntimeError>(()), |_| true);
        assert_eq!(admitted, Ok(cases::Verdict::Admitted));
        let other = ptr_runtime::RuntimeError::StructuralRecordOutsideAdmissionPath;
        let wrong = cases::classify("x", Err::<(), _>(other.clone()), |_| false);
        assert!(wrong.is_err());
        let refused = cases::classify("x", Err::<(), _>(other), |e| {
            matches!(
                e,
                ptr_runtime::RuntimeError::StructuralRecordOutsideAdmissionPath
            )
        });
        assert_eq!(refused, Ok(cases::Verdict::Refused));
    }

    /// R011's attempts are in the harness record of every seed
    /// (`adversarial_attempts_per_case`), which aggregate.py carries into run.json and
    /// metrics.json: no file of the experiment names a separate attempts artifact that nothing
    /// writes.
    #[test]
    fn no_required_artifact_is_one_that_nothing_writes() {
        let manifest: toml::Table = params::EXPERIMENT_TOML.parse().expect("experiment.toml");
        let required: Vec<&str> = manifest["required_artifacts"]
            .as_array()
            .expect("required_artifacts")
            .iter()
            .map(|name| name.as_str().expect("a name"))
            .collect();
        assert_eq!(required, ["run.json", "metrics.json"]);
        for (file, text) in [
            ("experiment.toml", params::EXPERIMENT_TOML),
            ("config.toml", params::CONFIG_TOML),
            ("PREREGISTRATION.md", PREREGISTRATION),
            ("criteria.toml", CRITERIA),
            ("README.md", README),
        ] {
            assert!(!text.contains("adversarial_attempts.json"), "{file}");
        }
        assert!(!params::CONFIG_TOML.contains("attempts_artifact"));
    }

    /// The counts the preregistration states are the counts of the rules and of the suite.
    #[test]
    fn the_preregistration_states_the_counts_the_harness_has() {
        let suite = params::ADVERSARIAL_CASES.len();
        assert!(PREREGISTRATION.contains(&format!(
            "{} coverage rules and {} falsifier rules",
            metrics::coverage().len(),
            metrics::falsifiers().len()
        )));
        assert!(PREREGISTRATION.contains(&format!("the {suite} `adversarial_cases`")));
        assert!(PREREGISTRATION.contains(&format!("per case of the {suite}")));
        let manifest: toml::Table = params::EXPERIMENT_TOML.parse().expect("experiment.toml");
        let text = manifest["falsification"].as_str().expect("falsification");
        assert!(text.contains(&format!("adversarial_cases, {suite} cases")));
        // The direct-append family of the text is the config's, with both compound events.
        for name in params::DIRECT_APPEND_EVENT_TYPES {
            assert!(
                text.contains(&format!("direct_ledger_append_{name}")),
                "{name}"
            );
        }
        assert_eq!(params::DIRECT_APPEND_EVENT_TYPES.len(), 11);
    }

    /// The attempts of the two C7 families are the calls that reached the runtime: the proposal
    /// constructor's refusals are not attempts and are only noted.
    #[test]
    fn constructor_refusals_are_not_attempts() {
        let json = &reduced_report().json;
        for (case, counter, note) in [
            (
                "fenced_gate_retire",
                "fenced_gate_retire_attempts",
                "internal.fenced_gate_retire_constructor_refusals",
            ),
            (
                "reactivate_from_revoked",
                "reactivate_from_revoked_attempts",
                "internal.reactivate_from_revoked_constructor_refusals",
            ),
        ] {
            let attempts = int(json, &format!("adversarial_attempts_per_case.{case}"));
            assert_eq!(attempts, int(json, counter), "{case}");
            assert_eq!(
                attempts, 2,
                "{case}: one per reason or form, from a stale view"
            );
            assert_eq!(int(json, note), 2, "{case}");
        }
    }

    /// Both compound events were offered to the public commit in every run, and the sets of the
    /// attempted and the admitted event types hold them.
    #[test]
    fn the_direct_append_family_has_the_compound_events() {
        let json = &reduced_report().json;
        for name in ["InputInvalidated", "ExpertAuthorityChanged"] {
            assert_eq!(
                int(
                    json,
                    &format!("adversarial_attempts_per_case.direct_ledger_append_{name}")
                ),
                1
            );
            assert_eq!(
                int(
                    json,
                    &format!("direct_ledger_append_accepted_by_type.{name}")
                ),
                0
            );
        }
        assert_eq!(
            emit::resolve(json, "direct_append_attempted_event_type_set_digest"),
            emit::resolve(json, "admitted_event_type_set_digest")
        );
    }

    /// C11 ran its un-invalidated control in every cell, drew every race point after at least
    /// one step, and Reactivate's two forms in C4 are each four cells.
    #[test]
    fn the_inflight_cells_have_a_control_and_a_step_before_the_event() {
        let json = &reduced_report().json;
        assert_eq!(int(json, "internal.c11_control_sessions_completed"), 6);
        let Some(Json::Obj(fixture)) = emit::resolve(json, "fixture") else {
            panic!("fixture");
        };
        let points: Vec<i64> = fixture
            .iter()
            .filter(|(name, _)| name.starts_with("race_point."))
            .map(|(_, value)| match value {
                Json::Int(point) => *point,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(points.len(), 6);
        assert!(points.iter().all(|point| *point >= 1), "{points:?}");
        assert_eq!(int(json, "internal.c4_reactivate_frozen_form_cells"), 4);
        assert_eq!(int(json, "internal.c4_reactivate_retired_form_cells"), 4);
    }

    #[test]
    fn the_inflight_layer_is_r010s() {
        assert_eq!(
            params::INFLIGHT_LAYER_J,
            crate::experiments::r010::params::J
        );
    }
}
