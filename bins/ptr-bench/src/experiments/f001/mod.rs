//! Experiment F001, the generate-verify-repair loop, as a harness: `ptr-bench
//! generate-verify-repair <seed>` runs the experiment's tasks once for one seed and prints exactly
//! one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `generate-verify-repair <tasks> <seed>`) is no evidence. The code under test is
//! `ptr_feedback::RepairLoop` and `Population::best_verified` over the real
//! `ptr_verifier::VerifierFabric` and `ActionContractVerifier`, driven only through their public
//! APIs. The claim is the mechanism: a stub generator with a stated skill on a synthetic
//! verifiable task shows that the loop uses the verifier's verdicts correctly and never lets a
//! rejected candidate pass. Nothing is said about the quality of any model's generations.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names (a
//! JSON integer, or an object of integers), `hard_failures`, the procedure, the negative
//! controls, the cells compared with the analytic expectation, the notes, the two judgements
//! (`coverage` and `falsifiers`). The exit status is 1 when a coverage prerequisite is unmet, a
//! falsifier is tripped or a metric has no value and no declared reason, and 2 when the harness
//! itself cannot run: a stub's labelled defects differ from the oracle's, a stub's empirical
//! frequency leaves its stated probability, or a negative control does not trip. A panic of the
//! product is a counted falsifier (`product_panics`), never exit 2.

pub mod dp;
pub mod metrics;
pub mod oracle;
pub mod params;
pub mod runner;
pub mod stats;
pub mod stub;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status};
use params::Scale;
use runner::{Arm, Pass, Variant};
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

fn metric_object(pass: &Pass) -> Json {
    let (printed, _) = pass.tally.metrics();
    Json::Obj(printed)
}

/// One negative control: the same machinery with a doctored loop, judged by the same falsifier
/// rules; the rule it exists to trip must fail, else the harness cannot see the defect and the
/// run is not valid. The controls run the harness' own reference loop, not the product, so they
/// show that the counters can fail, not that a defective product is caught (the mutation plan
/// does that).
fn control(
    seed: u64,
    variant: Variant,
    required_rule: &str,
    counters: &[&str],
) -> Result<Json, String> {
    let mut pass = runner::run_pass(
        seed,
        params::CONTROL_TASKS_PER_PROFILE,
        variant,
        metrics::LIST,
    )?;
    metrics::fill(&mut pass);
    let root = metric_object(&pass);
    let checks = judge(&root, &[], &metrics::falsifiers());
    let tripped: Vec<&Check> = checks.iter().filter(|c| c.status == Status::Fail).collect();
    let required_tripped = tripped.iter().any(|check| check.rule == required_rule);
    if !required_tripped {
        return Err(format!(
            "the negative control {variant:?} does not trip {required_rule:?}: the harness cannot see the defect it plants ({})",
            root.render()
        ));
    }
    Ok(Json::obj([
        ("mode", Json::str(format!("{variant:?}"))),
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
            Json::Obj(
                counters
                    .iter()
                    .map(|name| (name.to_string(), Json::Int(pass.tally.get(name))))
                    .collect(),
            ),
        ),
    ]))
}

fn cells_json(pass: &Pass) -> Json {
    Json::Arr(
        pass.comparisons
            .iter()
            .map(|row| {
                Json::obj([
                    ("profile", Json::str(row.profile)),
                    ("arm", Json::str(row.arm.name())),
                    ("budget", Json::Int(i64::from(row.budget))),
                    ("tasks", Json::Int(row.tasks)),
                    ("successes", Json::Int(row.successes)),
                    (
                        "expected_successes_milli",
                        Json::Int(row.expected_successes_milli),
                    ),
                    ("calls", Json::Int(row.calls)),
                    ("expected_calls_milli", Json::Int(row.expected_calls_milli)),
                    ("draws", Json::Int(row.draws)),
                    ("expected_draws_milli", Json::Int(row.expected_draws_milli)),
                    ("worst_z_milli", Json::Int(row.worst_z_milli)),
                    ("violated", Json::Bool(row.violated)),
                ])
            })
            .collect(),
    )
}

/// Success per profile and budget, for the reader (no rule reads it).
fn by_profile_json(pass: &Pass) -> Json {
    Json::Obj(
        params::PROFILES
            .iter()
            .enumerate()
            .map(|(profile, (name, _))| {
                let arm = |arm: Arm| {
                    Json::Obj(
                        params::BUDGETS
                            .iter()
                            .map(|budget| {
                                let stats = pass
                                    .cells
                                    .get(&(profile, arm, *budget))
                                    .copied()
                                    .unwrap_or_default();
                                (format!("b{budget}"), Json::Int(stats.successes))
                            })
                            .collect(),
                    )
                };
                (
                    (*name).to_string(),
                    Json::obj([
                        ("tasks", Json::Int(pass.profile_tasks[profile])),
                        ("rewrite", arm(Arm::Rewrite)),
                        ("repair", arm(Arm::Repair)),
                    ]),
                )
            })
            .collect(),
    )
}

/// Run every task once for `seed` at `scale` and assemble the one JSON object. A run the
/// harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    runner::selftest(seed)?;
    let tasks = scale.tasks_per_profile();
    let mut pass = runner::run_pass(seed, tasks, Variant::Mechanism, metrics::LIST)?;
    metrics::fill(&mut pass);
    let score_accept = control(
        seed,
        Variant::ScoreAccept,
        "false_accepts <= 0",
        &[
            "false_accepts",
            "unverified_returns",
            "selection_mismatches",
        ],
    )?;
    let blind_repair = control(
        seed,
        Variant::BlindRepair,
        "collateral_edits <= 0",
        &[
            "collateral_edits",
            "false_accepts",
            "expectation_violations",
        ],
    )?;
    let hard = metrics::hard_failures(&pass.tally);
    let (printed, gaps) = pass.tally.metrics();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("F001")),
        (
            "status_note".to_string(),
            note("one seed of the F001 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules; mechanism scope with a seeded stub generator: no claim about the quality of any model's generations"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push((
        "elapsed_ns".to_string(),
        Json::Int(started.elapsed().as_nanos() as i64),
    ));
    pairs.push((
        "procedure".to_string(),
        Json::obj([
            (
                "seed_is_declared",
                Json::Bool(params::SEEDS.contains(&seed)),
            ),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("tasks_per_profile", Json::Int(tasks as i64)),
            (
                "budgets",
                Json::ints(params::BUDGETS.iter().map(|b| i64::from(*b))),
            ),
            ("profiles", Json::str(params::profiles_text())),
            (
                "repair_q_permille",
                Json::Int(params::REPAIR_Q_PERMILLE as i64),
            ),
            ("required_level", Json::str(params::REQUIRED_LEVEL)),
            (
                "expectation_sigma",
                Json::Int(i64::from(params::EXPECTATION_SIGMA)),
            ),
            (
                "expectation_alpha_per_billion",
                Json::Int(params::EXPECTATION_ALPHA_PER_BILLION as i64),
            ),
            ("stub_selftest_passed", Json::Bool(true)),
            (
                "noise_permille",
                Json::obj([
                    ("unknown", Json::Int(params::MODE_UNKNOWN_PERMILLE as i64)),
                    ("disputed", Json::Int(params::MODE_DISPUTED_PERMILLE as i64)),
                    ("shallow", Json::Int(params::MODE_SHALLOW_PERMILLE as i64)),
                    ("lying", Json::Int(params::MODE_LYING_PERMILLE as i64)),
                    ("forged", Json::Int(params::MODE_FORGED_PERMILLE as i64)),
                ]),
            ),
        ]),
    ));
    pairs.push((
        "controls".to_string(),
        Json::obj([
            ("score_accept", score_accept),
            ("blind_repair", blind_repair),
        ]),
    ));
    pairs.push(("by_profile".to_string(), by_profile_json(&pass)));
    pairs.push(("cells".to_string(), cells_json(&pass)));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "scope",
                note("mechanism only: the generator is a seeded stub whose per-slot skill is stated in the preregistration, the task is a synthetic ActionContract with seven independent defect slots, and the comparative rules (repair >= rewrite >= single, repair cheaper per success) are consequences of that stub skill: they test that the loop uses the verifier's verdicts and critiques correctly, not that any model repairs better than it rewrites"),
            ),
            (
                "success",
                note("a task counts as a success only when the loop returned a candidate the harness's own contract check finds defect-free and whose report the harness accepts (Pass, no hard finding, Deterministic level); pooled over the three profiles"),
            ),
            (
                "generator_draws",
                note("slot draws: a fresh generation draws all seven slots, a repair draws only the slots its critique names; counted by the stubs, not by the loop"),
            ),
            (
                "draws_per_success",
                note("1000 * slot draws / successes for the arm and budget (a very large number when the arm never succeeded)"),
            ),
            (
                "verifier_call_mismatches",
                note("the loop's own claims (verifier calls, generator draws, repairs, declined repairs, rounds) against the call-counting boundary around the fabric and the stubs' counts"),
            ),
            (
                "expectation_violations",
                note("measured successes, verifier calls and slot draws of each (profile, arm, budget) cell against the exact analytic expectation (dp.rs): the success count by the exact two-sided binomial tail, the calls and the draws (sums of bounded per-task costs of exactly known mean and variance) by the Bernstein bound; a cell is violated when a p-value is below 573 per billion (the two-sided tail of five standard deviations, chosen over four because a seed makes several hundred such comparisons). A normal approximation is not used: it would call one failure in a saturated cell a ten-sigma event"),
            ),
            (
                "collateral_edits",
                note("a slot the verifier's report did not name (a slot with no defect before the repair) that is defective after it, or a payload that changed although neither payload slot was defective"),
            ),
            (
                "lying_refused",
                note("verifier calls whose report was the lying stub's (a Fail carrying a made-up hard finding) or a forged Pass that keeps its hard findings, and that the harness's acceptance rule refuses: observed from the boundary's log"),
            ),
            (
                "by_construction",
                note("tasks_per_profile is a count of the plan; the four status counters are observed from the boundary's log"),
            ),
        ]),
    ));
    let root = Json::Obj(pairs.clone());
    let coverage = judge(&root, &[], &metrics::coverage());
    let falsifiers = judge(&root, &[], &metrics::falsifiers());
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
                "falsifiers_tripped",
                Json::Int(count(&falsifiers, Status::Fail) as i64),
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

/// The seed and scale the arguments after the subcommand; strict, so a typo stops the run.
fn parse_args(args: &[String]) -> Result<(u64, Scale), String> {
    let parse = |text: &String, what: &str| {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    match args {
        [seed] => parse(seed, "seed").map(|seed| (seed, Scale::Pinned)),
        [tasks, seed] => parse(tasks, "tasks").and_then(|tasks| {
            if tasks == 0 {
                return Err("tasks must be at least 1".to_string());
            }
            parse(seed, "seed").map(|seed| (seed, Scale::Custom(tasks as usize)))
        }),
        _ => Err(
            "usage: generate-verify-repair <seed> | generate-verify-repair <tasks> <seed>"
                .to_string(),
        ),
    }
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after
/// the subcommand: `<seed>` (the pinned scale) or `<tasks> <seed>` (the mutation checker's form:
/// that many tasks per profile, no evidence). Exits 1 when the run is not clean and 2 when the
/// harness cannot run.
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
    // A product panic is counted; do not print one message per contained panic.
    std::panic::set_hook(Box::new(|_| {}));
    print_and_exit(execute(seed, scale));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const CRITERIA: &str = include_str!(
        "../../../../../experiments/feedback/F001-generate-verify-repair/criteria.toml"
    );
    const MUTATIONS: &str = include_str!(
        "../../../../../experiments/feedback/F001-generate-verify-repair/tests/mutations.toml"
    );
    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/feedback/F001-generate-verify-repair/PREREGISTRATION.md"
    );

    /// The shared reduced run.
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
        assert_eq!(table["experiment"].as_str(), Some("F001"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_integers_the_harness_prints() {
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
            let right = if is_literal(tokens[2]) {
                tokens[2].parse::<i64>().unwrap()
            } else {
                int(&report.json, tokens[2])
            };
            let violating = match tokens[1] {
                ">=" => right - 1,
                "<=" => right + 1,
                _ => right + 1,
            };
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
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "generate-verify-repair");
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
            .starts_with("{\"benchmark\":\"generate-verify-repair\",\"experiment\":\"F001\""));
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        assert_eq!(int(&report.json, "seed"), 17);
        for name in metrics::METRIC_NAMES {
            assert!(emit::resolve(&report.json, name).is_some(), "{name}");
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

    /// The real mechanism trips no hard counter on the reduced run, and the run exercises every
    /// path the criteria speak of.
    #[test]
    fn a_reduced_run_trips_no_hard_counter_and_is_a_real_comparison() {
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
        assert_eq!(int(json, "profiles.uniform.tasks"), 250);
        assert!(int(json, "runs") > 8000);
        for status in ["pass", "fail", "unknown", "disputed"] {
            assert!(int(json, &format!("status_{status}")) > 0, "{status}");
        }
        assert!(int(json, "lying_refused") > 0 && int(json, "shallow_refused") > 0);
        assert!(int(json, "repaired_successes") > 0);
        assert!(int(json, "exhausted_runs") > 0);
        // The skill of the stub shows: repair beats rewrite beats single, and is cheaper per
        // success, at the larger budgets.
        assert!(int(json, "success.repair.b4") > int(json, "success.rewrite.b4"));
        assert!(int(json, "success.rewrite.b4") > int(json, "success.single"));
        assert!(
            int(json, "draws_per_success.repair.b16") < int(json, "draws_per_success.rewrite.b16")
        );
        // One verifier call per candidate: the calls of the budget-1 arms are the tasks.
        assert_eq!(int(json, "verifier_calls.single"), 750);
        assert_eq!(int(json, "verifier_calls.repair.b1"), 750);
    }

    #[test]
    fn the_same_seed_gives_the_same_json_apart_from_the_timing_and_another_seed_does_not() {
        let strip = |json: &Json| {
            let Json::Obj(pairs) = json else {
                panic!("object")
            };
            Json::Obj(
                pairs
                    .iter()
                    .filter(|(k, _)| k != "elapsed_ns")
                    .cloned()
                    .collect(),
            )
            .render()
        };
        let first = strip(&reduced_report().json);
        let again = strip(&execute(17, Scale::Reduced).expect("a second run").json);
        assert_eq!(first, again);
        let other = strip(&execute(5, Scale::Reduced).expect("another seed").json);
        assert_ne!(first, other);
    }

    /// Both negative controls trip the rule they exist for.
    #[test]
    fn the_negative_controls_trip_their_rules() {
        let json = &reduced_report().json;
        for name in ["score_accept", "blind_repair"] {
            assert!(
                matches!(
                    emit::resolve(json, &format!("controls.{name}.required_rule_tripped")),
                    Some(Json::Bool(true))
                ),
                "{name}"
            );
        }
        assert!(int(json, "controls.score_accept.counters.false_accepts") >= 1);
        assert!(int(json, "controls.blind_repair.counters.collateral_edits") >= 1);
        // The mechanism itself trips neither.
        assert_eq!(int(json, "false_accepts"), 0);
        assert_eq!(int(json, "collateral_edits"), 0);
    }

    #[test]
    fn a_control_that_does_not_trip_is_a_harness_failure() {
        // The reference loop with the right rule and a repairer that reads the critique trips
        // no false accept: as a "control" for it, the run is invalid.
        let outcome = control(
            17,
            Variant::BlindRepair,
            "false_accepts <= 0",
            &["false_accepts"],
        );
        assert!(outcome.is_err());
    }

    #[test]
    fn both_negative_controls_trip_for_all_five_seeds() {
        for seed in params::PILOT_SEEDS {
            control(
                seed,
                Variant::ScoreAccept,
                "false_accepts <= 0",
                &["false_accepts"],
            )
            .unwrap_or_else(|reason| panic!("seed {seed}: {reason}"));
            control(
                seed,
                Variant::BlindRepair,
                "collateral_edits <= 0",
                &["collateral_edits"],
            )
            .unwrap_or_else(|reason| panic!("seed {seed}: {reason}"));
        }
    }

    #[test]
    fn no_rule_reads_the_timing() {
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("elapsed") && !rule.contains("_ns"), "{rule}");
        }
    }

    #[test]
    fn the_arguments_are_a_seed_or_tasks_and_a_seed_and_nothing_else() {
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
        let report = execute(17, Scale::Custom(40)).expect("a custom run");
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "custom"
        ));
        assert_eq!(int(&report.json, "hard_failures"), 0);
        assert!(
            report.coverage_unmet() > 0,
            "forty tasks per profile cannot cover the pinned shape"
        );
    }
}
