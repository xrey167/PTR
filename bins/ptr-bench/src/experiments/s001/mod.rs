//! Experiment S001, dependency-aware invalidation, as a harness: `ptr-bench invalidation <seed>`
//! runs the experiment's update streams over seeded graphs once for one seed and prints exactly
//! one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `invalidation <operations> <seed>`) is no evidence. The code under test is
//! `ptr_semdb::SemanticHost` (`prepare_delta`, its `affected()` set and its eviction of affected
//! derived values), driven only through its public API. The harness supplies the derivation:
//! `ptr-semdb` has no query engine, so "recompute" is a pure function the harness evaluates for
//! the keys SemDB marks affected (arm A) or for every derived key (arm G), and an independent
//! oracle (`oracle.rs`, no code shared with `DependencyGraph`) supplies the truth.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names (a
//! JSON integer), `hard_failures`, the timings (never read by a rule), the procedure, the
//! negative controls, the per-graph counters, the notes, the counters that belong to no metric
//! (`internal`), and the two judgements (`coverage` and `falsifiers`). The exit status is 1 when
//! a coverage prerequisite is unmet, a falsifier is tripped or a metric has no value and no
//! declared reason, and 2 when the harness itself cannot run: the oracle and the global
//! baseline disagree, the generator produced a delta the oracle does not call what it meant, or
//! a negative control does not trip.

pub mod derive;
pub mod generator;
pub mod metrics;
pub mod oracle;
pub mod params;
pub mod runner;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use params::{Scale, Topology};
use runner::{Counters, Mode, Pass};
use std::time::Instant;

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

fn tally_of(pass: &Pass) -> Tally {
    let mut tally = Tally::new(metrics::LIST);
    metrics::fill(&mut tally, &pass.counters, pass.topologies.len());
    tally
}

fn metric_object(tally: &Tally) -> Json {
    let (printed, _) = tally.metrics();
    Json::Obj(printed)
}

fn timing_json(pass: &Pass, elapsed_ns: u64) -> Json {
    let by_size = pass
        .timing
        .by_size
        .iter()
        .map(|(size, t)| {
            (
                size.to_string(),
                Json::obj([
                    ("deltas", Json::Int(t.deltas)),
                    ("phase_one_ns", Json::Int(t.phase_one_ns as i64)),
                    ("phase_two_ns", Json::Int(t.phase_two_ns as i64)),
                ]),
            )
        })
        .collect();
    Json::obj([
        ("elapsed_ns", Json::Int(elapsed_ns as i64)),
        (
            "recompute_incremental_ns",
            Json::Int(pass.timing.recompute_incremental_ns as i64),
        ),
        (
            "recompute_global_ns",
            Json::Int(pass.timing.recompute_global_ns as i64),
        ),
        ("by_size", Json::Obj(by_size)),
    ])
}

fn classes_json(pass: &Pass) -> Json {
    Json::Obj(
        pass.classes
            .iter()
            .map(|(name, s)| {
                (
                    name.clone(),
                    Json::obj([
                        ("deltas", Json::Int(s.deltas)),
                        ("derived_keys_at_end", Json::Int(s.derived_keys)),
                        ("evals_global", Json::Int(s.evals_global)),
                        ("evals_incremental", Json::Int(s.evals_incremental)),
                        ("closure_derived_total", Json::Int(s.closure_derived_total)),
                        ("local_class_deltas", Json::Int(s.local_class_deltas)),
                        (
                            "locality_ratio_permille_local",
                            Json::Int(metrics::permille(s.local_incremental, s.local_global, 0)),
                        ),
                    ]),
                )
            })
            .collect(),
    )
}

/// One negative control: the same graph and stream with a doctored affected set, judged by the
/// same falsifier rules; the rule it exists to trip must fail, else the harness cannot see the
/// defect and the run is not valid.
fn control(seed: u64, mode: Mode, required_rule: &str) -> Result<(Json, i64), String> {
    let pass = runner::run_pass(seed, &runner::control_plan(), mode)?;
    let tally = tally_of(&pass);
    let root = metric_object(&tally);
    let checks = judge(&root, &[], &metrics::falsifiers());
    let tripped: Vec<&Check> = checks.iter().filter(|c| c.status == Status::Fail).collect();
    let required_tripped = tripped.iter().any(|check| check.rule == required_rule);
    if !required_tripped {
        return Err(format!(
            "the negative control {mode:?} does not trip {required_rule:?}: the harness cannot see the defect it plants ({})",
            root.render()
        ));
    }
    let json = Json::obj([
        ("mode", Json::str(format!("{mode:?}"))),
        ("required_rule", Json::str(required_rule)),
        ("required_rule_tripped", Json::Bool(required_tripped)),
        (
            "tripped",
            Json::Arr(
                tripped
                    .iter()
                    .map(|check| Json::str(check.rule.clone()))
                    .collect(),
            ),
        ),
        (
            "counters",
            Json::obj([
                ("deltas", Json::Int(tally.get("deltas"))),
                (
                    "missed_invalidations",
                    Json::Int(tally.get("missed_invalidations")),
                ),
                (
                    "closure_mismatches",
                    Json::Int(tally.get("closure_mismatches")),
                ),
                (
                    "semantic_divergences",
                    Json::Int(tally.get("semantic_divergences")),
                ),
                (
                    "locality_ratio_permille_local",
                    Json::Int(tally.get("locality_ratio_permille_local")),
                ),
                (
                    "local_class_deltas",
                    Json::Int(tally.get("local_class_deltas")),
                ),
            ]),
        ),
    ]);
    Ok((json, tripped.len() as i64))
}

/// Run every graph once for `seed` at `scale` and assemble the one JSON object. A run the
/// harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape();
    let plan = runner::plan_of(&shape);
    let pass = runner::run_pass(seed, &plan, Mode::Mechanism)?;
    let (drop_one, _) = control(seed, Mode::DropOne, "missed_invalidations <= 0")?;
    let (everything, _) = control(
        seed,
        Mode::Everything,
        "locality_ratio_permille_local <= 250",
    )?;
    let mut tally = tally_of(&pass);
    let c: &Counters = &pass.counters;
    tally.set("stale_survivors", c.stale_survivors);
    tally.set("unsatisfied_recomputes", c.unsatisfied_recomputes);
    tally.set("phase_two_refusals", c.phase_two_refusals);
    tally.set("value_changed_deltas", c.value_changed_deltas);
    tally.set("local_incremental_evals", c.local_incremental);
    tally.set("local_global_evals", c.local_global);
    tally.set("hub_incremental_evals", c.hub_incremental);
    tally.set("hub_global_evals", c.hub_global);
    tally.set("recomputed_total", c.recomputed_total);
    tally.set("changed_keys", c.changed_keys);
    tally.set("lossy_evaluations", c.lossy_evaluations);
    let hard = metrics::hard_failures(&tally);

    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let topologies: Vec<&str> = Topology::ALL.iter().map(|t| t.name()).collect();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("S001")),
        (
            "status_note".to_string(),
            note("one seed of the S001 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push((
        "timing_ns".to_string(),
        timing_json(&pass, started.elapsed().as_nanos() as u64),
    ));
    pairs.push((
        "procedure".to_string(),
        Json::obj([
            (
                "seed_is_declared",
                Json::Bool(params::SEEDS.contains(&seed)),
            ),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("sizes", Json::ints(shape.sizes.iter().map(|s| *s as i64))),
            ("operations_per_size", Json::ints(shape.deltas.iter().map(|s| *s as i64))),
            ("topology_classes", Json::strs(&topologies)),
            (
                "local_class",
                note("applied deltas whose oracle structural closure holds at least one derived key and at most 5% of the derived keys of the graph after the delta (closure_derived * 1000 <= 50 * derived_keys); no-op deltas, cycle attempts and probes are not in it"),
            ),
            (
                "hub_class",
                note("applied edits of the designated hub of the hub-and-spoke class (every derived key lists it, so the closure is the whole derived set and no advantage is expected: about 1000 per mille); an ungated control. The other classes' hub edits (their largest-cone ground key) are ordinary deltas"),
            ),
            (
                "evals",
                note("evals_global counts every derived key of the graph after every accepted delta (arm G); evals_incremental counts the derivation evaluations of arm A, which evaluates only the derived keys the host marks affected"),
            ),
        ]),
    ));
    pairs.push((
        "controls".to_string(),
        Json::obj([("drop_one", drop_one), ("everything", everything)]),
    ));
    pairs.push(("classes".to_string(), classes_json(&pass)));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "recompute",
                note("a harness-supplied pure derivation (derive.rs: a salted hash chain, the minimum, the parity of the sum) evaluated for the keys SemDB marks affected; ptr-semdb has no query engine, so what 'recomputing' costs is a count of evaluations, not a SemDB feature"),
            ),
            (
                "closure_mismatches",
                note("the keys on which the host's affected() (the set arm A uses) and the oracle's structural closure over the old and the new edges differ, summed over deltas"),
            ),
            (
                "missed_invalidations",
                note("per delta, the derived keys whose fresh value changed and that the host did not mark affected, together with the derived values still present after prepare_delta whose fresh oracle value differs (stale survivors)"),
            ),
            (
                "semantic_divergences",
                note("after both phases, the keys on which the host's committed state and arm G's full recomputation differ (missing, extra or another value), summed over deltas"),
            ),
            (
                "precision",
                note("precision_structural_permille compares the affected set with the oracle closure; precision_value_permille is the share of recomputed derived keys whose value really changed (descriptive: SemDB has no early cutoff, so lossy derivations show unnecessary invalidations by design)"),
            ),
            (
                "limits",
                note("prepare_delta clones the whole state and affected_by rebuilds the reverse map per call (O(edges)), so only the recomputation, not the invalidation step, has locality; the wall times are in timing_ns and are never judged"),
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

fn print_and_exit(result: Result<Report, String>) {
    match result {
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

/// The seed and scale the arguments after the subcommand name; strict, so a typo stops the run.
fn parse_args(args: &[String]) -> Result<(u64, Scale), String> {
    let parse = |text: &String, what: &str| {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    match args {
        [seed] => parse(seed, "seed").map(|seed| (seed, Scale::Pinned)),
        [operations, seed] => parse(operations, "operations").and_then(|operations| {
            if operations == 0 {
                return Err("operations must be at least 1".to_string());
            }
            parse(seed, "seed").map(|seed| (seed, Scale::Custom(operations as usize)))
        }),
        _ => Err("usage: invalidation <seed> | invalidation <operations> <seed>".to_string()),
    }
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after
/// the subcommand: `<seed>` (the pinned scale) or `<operations> <seed>` (the mutation
/// checker's form: the pinned graph sizes with that many operations per graph, no evidence).
/// Exits 1 when the run is not clean and 2 when the harness cannot run.
pub fn run(args: &[String]) {
    let (seed, scale) = match parse_args(args) {
        Ok(parsed) => parsed,
        Err(reason) => {
            eprintln!("{}: {reason}", params::BENCHMARK);
            std::process::exit(2);
        }
    };
    if !params::SEEDS.contains(&seed) {
        eprintln!(
            "seed {seed} is not one of the manifest's seeds {:?}; this is a dry run",
            params::SEEDS
        );
    }
    print_and_exit(execute(seed, scale));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const CRITERIA: &str =
        include_str!("../../../../../experiments/semdb/S001-invalidation/criteria.toml");
    const MUTATIONS: &str =
        include_str!("../../../../../experiments/semdb/S001-invalidation/tests/mutations.toml");
    const PREREGISTRATION: &str =
        include_str!("../../../../../experiments/semdb/S001-invalidation/DRAFT_PREREGISTRATION.md");

    /// The shared reduced run (small graphs of every topology class; a few seconds in a debug
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

    fn is_literal(token: &str) -> bool {
        let digits = token.strip_prefix('-').unwrap_or(token);
        !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
    }

    /// `json` with the member `path` (dotted) replaced.
    fn with(json: &Json, path: &str, value: Json) -> Json {
        fn set(at: &mut Json, keys: &[&str], value: Json) {
            let Json::Obj(pairs) = at else {
                panic!("not an object")
            };
            let slot = pairs
                .iter_mut()
                .find(|(name, _)| name == keys[0])
                .expect("member");
            if keys.len() == 1 {
                slot.1 = value;
            } else {
                set(&mut slot.1, &keys[1..], value);
            }
        }
        let mut copy = json.clone();
        set(&mut copy, &path.split('.').collect::<Vec<_>>(), value);
        copy
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
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("S001"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_an_integer_the_harness_prints() {
        let report = reduced_report();
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            assert_eq!(tokens.len(), 3, "{rule}");
            for token in [tokens[0], tokens[2]] {
                if !is_literal(token) {
                    assert!(
                        matches!(emit::resolve(&report.json, token), Some(Json::Int(_))),
                        "{rule}: {token} does not resolve to an integer in the harness JSON"
                    );
                }
            }
        }
    }

    /// Every rule can fail: set the counter it names to a violating value in the emitted JSON
    /// and the harness's own judge flags that rule.
    #[test]
    fn every_rule_fails_when_the_emitted_counter_is_set_to_a_violation() {
        let report = reduced_report();
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let current = int(&report.json, tokens[0]);
            let violating = match tokens[1] {
                ">=" => tokens[2].parse::<i64>().unwrap() - 1,
                "<=" => tokens[2].parse::<i64>().unwrap() + 1,
                _ => int(&report.json, tokens[2]) + 1,
            };
            assert_ne!(current, violating, "{rule}");
            let broken = with(&report.json, tokens[0], Json::Int(violating));
            let failed: Vec<String> = judge(&broken, &[], std::slice::from_ref(rule))
                .into_iter()
                .filter(|check| check.status == Status::Fail)
                .map(|check| check.rule)
                .collect();
            assert_eq!(failed, vec![rule.clone()], "{rule}");
        }
    }

    /// Every rule is a reading of a sentence of the manifest's falsification text.
    #[test]
    fn the_rules_mirror_the_falsification_text_of_the_manifest() {
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let text = manifest["falsification"].as_str().expect("falsification");
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let sentence = match tokens[1] {
                ">=" => format!("{} < {}", tokens[0], tokens[2]),
                "<=" => format!("{} > {}", tokens[0], tokens[2]),
                _ => format!("{} != {}", tokens[0], tokens[2]),
            };
            assert!(
                text.contains(&sentence),
                "{rule}: no '{sentence}' in the text"
            );
        }
        assert_eq!(metrics::falsifiers().len(), 9);
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "invalidation");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
    }

    #[test]
    fn the_hard_counters_of_the_mutation_plan_are_the_harnesss() {
        let plan: toml::Table = MUTATIONS.parse().expect("mutations.toml is TOML");
        let listed: Vec<String> = plan["hard_counters"]
            .as_array()
            .expect("hard_counters")
            .iter()
            .map(|name| name.as_str().expect("a name").to_string())
            .collect();
        assert_eq!(listed, metrics::HARD_COUNTERS);
        assert_eq!(plan["subcommand"].as_str(), Some(params::BENCHMARK));
        // Every hard counter is expected by at least one mutation.
        let mutations = plan["mutation"].as_array().expect("mutations");
        for counter in metrics::HARD_COUNTERS {
            assert!(
                mutations.iter().any(|m| m["expect"]
                    .as_array()
                    .expect("expect")
                    .iter()
                    .any(|name| name.as_str() == Some(counter))),
                "{counter} is expected by no mutation"
            );
        }
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"invalidation\",\"experiment\":\"S001\""));
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        assert_eq!(int(&report.json, "seed"), 17);
        for name in metrics::METRIC_NAMES {
            assert!(
                matches!(emit::resolve(&report.json, name), Some(Json::Int(_))),
                "{name}"
            );
        }
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "reduced"
        ));
        assert!(matches!(
            emit::resolve(&report.json, "procedure.shape_is_pinned"),
            Some(Json::Bool(false))
        ));
    }

    /// The real mechanism trips no falsifier on the reduced run, and the reduced run covers
    /// every topology class, every kind of operation and the lossy class.
    #[test]
    fn a_reduced_run_trips_no_falsifier_and_is_a_real_comparison() {
        let report = reduced_report();
        let tripped: Vec<&str> = report
            .falsifiers
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(tripped.is_empty(), "{tripped:?}");
        let json = &report.json;
        assert_eq!(int(json, "hard_failures"), 0);
        assert_eq!(int(json, "topology_classes"), 5);
        assert!(int(json, "deltas") >= 1000, "{}", int(json, "deltas"));
        assert!(int(json, "cycle_attempts") >= 1);
        assert_eq!(int(json, "cycle_attempts"), int(json, "cycle_refusals"));
        assert!(int(json, "missing_dependency_probes") >= 1);
        assert!(
            int(json, "rewires") >= 1
                && int(json, "removals") >= 1
                && int(json, "noop_writes") >= 1
        );
        assert!(int(json, "lossy_class_deltas") >= 1 && int(json, "value_changed_total") >= 1);
        // The mechanism's affected set is the oracle's closure, and it recomputes fewer keys than
        // the global baseline.
        assert_eq!(
            int(json, "affected_total"),
            int(json, "structural_closure_total")
        );
        assert_eq!(int(json, "precision_structural_permille"), 1000);
        assert_eq!(int(json, "recall_permille"), 1000);
        assert!(int(json, "evals_incremental") < int(json, "evals_global"));
        // Lossy derivations show unnecessary invalidations (no early cutoff): descriptive.
        assert!(int(json, "unnecessary_invalidations") > 0);
        assert!(int(json, "precision_value_permille") < 1000);
        // Arm A really evaluated what it invalidated, and no recompute lacked an input.
        assert_eq!(int(json, "internal.unsatisfied_recomputes"), 0);
        assert_eq!(int(json, "internal.phase_two_refusals"), 0);
        assert_eq!(int(json, "internal.stale_survivors"), 0);
    }

    #[test]
    fn the_same_seed_gives_the_same_json_apart_from_the_timings_and_another_seed_does_not() {
        let strip = |json: &Json| {
            let Json::Obj(pairs) = json else {
                panic!("object")
            };
            Json::Obj(
                pairs
                    .iter()
                    .filter(|(k, _)| k != "timing_ns")
                    .cloned()
                    .collect(),
            )
            .render()
        };
        let first = strip(&reduced_report().json);
        let again = strip(&execute(17, Scale::Reduced).expect("a second run").json);
        assert_eq!(first, again);
        let other = strip(&execute(29, Scale::Reduced).expect("another seed").json);
        assert_ne!(first, other);
    }

    /// Both negative controls trip the rule they exist for, on the same graph and stream, and
    /// the over-invalidating control pushes locality to the whole graph.
    #[test]
    fn the_negative_controls_trip_their_rules() {
        let json = &reduced_report().json;
        assert!(matches!(
            emit::resolve(json, "controls.drop_one.required_rule_tripped"),
            Some(Json::Bool(true))
        ));
        assert!(int(json, "controls.drop_one.counters.missed_invalidations") >= 1);
        assert!(int(json, "controls.drop_one.counters.semantic_divergences") >= 1);
        assert!(matches!(
            emit::resolve(json, "controls.everything.required_rule_tripped"),
            Some(Json::Bool(true))
        ));
        assert_eq!(
            int(
                json,
                "controls.everything.counters.locality_ratio_permille_local"
            ),
            1000
        );
        assert!(int(json, "controls.everything.counters.closure_mismatches") >= 1);
        // The mechanism's own locality on the same kind of delta is far below it.
        assert!(int(json, "locality_ratio_permille_local") <= 250);
    }

    #[test]
    fn a_control_that_does_not_trip_is_a_harness_failure() {
        // The mechanism does not trip the missed-invalidation rule: as a "control" it is invalid.
        let outcome = control(17, Mode::Mechanism, "missed_invalidations <= 0");
        assert!(outcome.is_err());
    }

    #[test]
    fn timings_are_in_their_own_field_and_no_rule_reads_them() {
        let json = &reduced_report().json;
        assert!(int(json, "timing_ns.elapsed_ns") > 0);
        assert!(int(json, "timing_ns.recompute_global_ns") > 0);
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("timing") && !rule.contains("_ns"), "{rule}");
        }
    }

    #[test]
    fn the_hub_class_is_a_control_near_the_whole_graph_on_hub_and_spoke() {
        let json = &reduced_report().json;
        let hub = int(json, "locality_ratio_permille_hub");
        assert!(
            hub >= 800,
            "the hub's closure is nearly every derived key: {hub}"
        );
        assert!(hub > int(json, "locality_ratio_permille_local"));
        assert!(int(json, "hub_deltas") >= 1);
    }

    #[test]
    fn the_arguments_are_a_seed_or_operations_and_a_seed_and_nothing_else() {
        let args = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_args(&args(&["17"])), Ok((17, Scale::Pinned)));
        assert_eq!(
            parse_args(&args(&["60", "17"])),
            Ok((17, Scale::Custom(60)))
        );
        for bad in [
            args(&[]),
            args(&["x"]),
            args(&["-1"]),
            args(&["0", "17"]),
            args(&["60", "x"]),
            args(&["1", "2", "3"]),
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_custom_scale_run_is_marked_and_the_mechanism_still_holds_every_hard_counter() {
        let report = execute(17, Scale::Custom(6)).expect("a custom run");
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "custom"
        ));
        assert!(matches!(
            emit::resolve(&report.json, "procedure.shape_is_pinned"),
            Some(Json::Bool(false))
        ));
        assert_eq!(int(&report.json, "hard_failures"), 0);
        assert!(
            report.coverage_unmet() > 0,
            "six operations per graph cannot cover the pinned shape"
        );
    }
}
