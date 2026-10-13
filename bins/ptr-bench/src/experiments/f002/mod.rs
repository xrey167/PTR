//! Experiment F002, weak supervision with verifier precedence, as a harness:
//! `ptr-bench weak-supervision <seed>` runs three synthetic labelling worlds once for one seed
//! and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`. The code under test is `ptr-labeling` (`VoteMatrix`, `fit_label_model`,
//! `resolve`, `rank_for_annotation`, `evaluate`, `function_accuracy`, `EvaluationSet`) and the
//! statistics kernel of `ptr-analytics` it uses, driven through their public APIs only. The
//! harness supplies the worlds (`generator.rs`: known truth, probabilistic functions with
//! known confusion behaviour, verifiers that rule a class out), the baselines (majority vote,
//! verifier-only) and an independent oracle (`oracle.rs`: a second Dawid-Skene EM, an expected
//! resolution, Brier, ECE, Krippendorff alpha and Wilson computed from their definitions).
//!
//! The object holds, in this order: the identity of the run, every metric (a JSON integer),
//! `hard_failures`, the procedure, the negative controls, the per-world descriptive counters,
//! the notes, the counters that belong to no metric (`internal`), and the two judgements
//! (`coverage` and `falsifiers`). The exit status is 1 when a coverage prerequisite is unmet, a
//! falsifier is tripped or a metric has no value and no declared reason, and 2 when the harness
//! itself cannot run: the oracle fails its own check, the generator is not deterministic or
//! breaks its own structure, the product refuses a vote matrix built from valid votes, or a
//! negative control does not trip. A panic or an error of the product on a valid input is a
//! counted falsification, never exit 2.

pub mod generator;
pub mod metrics;
pub mod oracle;
pub mod params;
pub mod runner;

use std::time::Instant;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use params::{Scale, World};

const USAGE: &str = "usage: weak-supervision <seed>";

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

/// The seed of `weak-supervision <seed>`. `scripts/mutation_check.py` appends the seed twice
/// (`<cases> <seed>`), so the same value twice is also accepted; anything else is a usage error.
pub fn seed_from_args(args: &[String]) -> Result<u64, String> {
    let parse = |value: &String| {
        value
            .parse::<u64>()
            .map_err(|_| format!("seed must be a non-negative integer, got {value:?}"))
    };
    match args.get(2..).unwrap_or(&[]) {
        [seed] => parse(seed),
        [seed, again] if seed == again => parse(seed),
        [] => Err("seed is required: weak-supervision <seed>".to_string()),
        _ => Err(USAGE.to_string()),
    }
}

/// Run the three worlds once for `seed` at `scale` and assemble the one JSON object. A run the
/// harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape();
    oracle::self_check()?;
    let mut tally = Tally::new(metrics::LIST);
    for name in metrics::ZEROED {
        tally.set(name, 0);
    }

    let mut worlds: Vec<(String, Json)> = Vec::new();
    let mut sound = None;
    let mut contaminated = None;
    for world in World::ALL {
        let generated = generator::generate(seed, world, shape);
        if generated != generator::generate(seed, world, shape) {
            return Err(format!(
                "the generator is not deterministic for seed {seed}, world {}",
                world.key()
            ));
        }
        let run = runner::run_world(&mut tally, seed, world, shape, &generated)?;
        worlds.push((world.key().to_string(), run.descriptive));
        match world {
            World::Sound => sound = Some(generated),
            World::Contaminated => contaminated = Some((generated, run.pipeline)),
            World::Homogeneous => {}
        }
    }
    let sound = sound.ok_or("the sound world did not run")?;
    let cell = runner::run_two_function_cell(&mut tally, &sound)?;

    let gold_items = World::ALL
        .iter()
        .map(|world| tally.get(&format!("{}_gold_items", world.key())))
        .min()
        .unwrap_or(0);
    tally.set("gold_items", gold_items);
    let violations = runner::gain_gate_violations(&runner::GainView {
        brier_gain: tally.get("w1_brier_gain_ppm"),
        ece_gain: tally.get("w1_ece_gain_ppm"),
        brier_lower: tally.get("w1_brier_gain_lower_ppm"),
        ece_lower: tally.get("w1_ece_gain_lower_ppm"),
        accuracy_ds: tally.get("w1_census_accuracy_ds_permille"),
        accuracy_mv: tally.get("w1_census_accuracy_mv_permille"),
        gate_brier_on_lower: params::BRIER_GATE_ON_LOWER_BOUND,
        gate_ece_on_lower: params::ECE_GATE_ON_LOWER_BOUND,
    });
    tally.set("gain_violations", violations as i64);

    // The negative controls need the contaminated world's pipeline; if a product failure
    // stopped it (counted above), they cannot be built and the run is falsified anyway.
    let controls = match contaminated {
        Some((generated, Some(pipeline))) => {
            let results = runner::negative_controls(&pipeline, &generated);
            if let Some(missed) = results.iter().find(|result| !result.tripped()) {
                return Err(format!(
                    "the negative control {:?} does not trip {}: the harness cannot see the defect it plants",
                    missed.name, missed.counter
                ));
            }
            tally.set("negative_controls_tripped", results.len() as i64);
            Json::Arr(results.iter().map(runner::ControlResult::json).collect())
        }
        _ => {
            tally.set("negative_controls_tripped", 0);
            Json::str("not built: the contaminated world's pipeline did not run")
        }
    };

    let hard: i64 = metrics::HARD_COUNTERS
        .iter()
        .map(|name| tally.get(name))
        .sum();
    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("F002")),
        (
            "status_note".to_string(),
            note("one seed of the F002 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([("elapsed", Json::Int(started.elapsed().as_millis() as i64))]),
    ));
    pairs.push((
        "procedure".to_string(),
        Json::obj([
            (
                "seed_is_declared",
                Json::Bool(params::SEEDS.contains(&seed)),
            ),
            (
                "seed_is_pilot",
                Json::Bool(params::PILOT_SEEDS.contains(&seed)),
            ),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("items_per_world", Json::Int(shape.items as i64)),
            ("gold_items_per_world", Json::Int(shape.gold as i64)),
            ("acquisition_budget", Json::Int(shape.budget as i64)),
            ("bootstrap_resamples", Json::Int(shape.resamples as i64)),
            (
                "worlds",
                note("w1: sound verifiers, heterogeneous accuracies (a gain is expected by construction); w2: contaminated verifiers, used for the hard veto check and the Disputed rate; w3: homogeneous accuracies and no verifier (no gain expected; a control that the harness is not rigged)"),
            ),
            (
                "arms",
                note("ds: ptr-labeling Dawid-Skene with verifier precedence; mv: majority vote over the class votes, posterior (n_c + 1) / (n + C) with ties to the lowest class, vetoes ignored; mvp (descriptive): the same majority vote restricted to the classes no verifier rules out; verifier-only (descriptive): only items the verifiers determine are labelled. All arms get the same class votes (the same annotation budget) and the same gold items"),
            ),
        ]),
    ));
    pairs.push(("negative_controls".to_string(), controls));
    pairs.push(("two_function_cell".to_string(), cell));
    pairs.push(("worlds".to_string(), Json::Obj(worlds)));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "disputed_items_in_scoring",
                note("a disputed item has no distribution (every class is ruled out); the scorer is given the uniform distribution over the classes for it, in every arm, and the veto check exempts exactly that substitute"),
            ),
            (
                "gold",
                note("gold is GoldSource::Oracle, the known truth of the synthetic world, sampled uniformly: a mechanism experiment, not a claim about real annotators or model labellers"),
            ),
            (
                "ties",
                note("the argmax of a distribution is its last maximal class (ptr-labeling's evaluate and ptr-analytics' ECE); the majority-vote posterior carries a 1e-9 per-class-rank epsilon so that ties go to the lowest class, a convention generous to the baseline because the lower classes are the more frequent ones; equal-mass calibration bins cut a stable sort of the confidences, so ties in confidence fall in the gold order"),
            ),
            (
                "gains",
                note("brier_gain_ppm and ece_gain_ppm are majority vote minus the label model, from the scores the product reported, rounded to parts per million; the lower bounds are the 2.5th percentile of a paired bootstrap over the gold items computed with the oracle's metrics"),
            ),
            (
                "gold_guard_probes",
                note("each probe is an input the guards must refuse (a duplicate gold item with the same or another class, a human label in an oracle set, a gold item without a posterior, a posterior that is no distribution); a refusal is the pass"),
            ),
        ]),
    ));
    let internal = match tally.internal() {
        Json::Obj(pairs) => pairs,
        other => vec![("counters".to_string(), other)],
    };
    pairs.push(("internal".to_string(), Json::Obj(internal)));
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

/// Run the experiment's harness at the pinned scale and print its one JSON object. Exits 1 when
/// the run is not clean and 2 when the harness cannot run.
pub fn run(seed: u64) {
    if !params::SEEDS.contains(&seed) {
        eprintln!(
            "seed {seed} is not one of the manifest's seeds {:?}; this is a dry run",
            params::SEEDS
        );
    }
    runner::silence_contained_panics();
    print_and_exit(execute(seed, Scale::Pinned));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const CRITERIA: &str =
        include_str!("../../../../../experiments/feedback/F002-weak-supervision/criteria.toml");
    const MUTATIONS: &str = include_str!(
        "../../../../../experiments/feedback/F002-weak-supervision/tests/mutations.toml"
    );
    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/feedback/F002-weak-supervision/PREREGISTRATION.md"
    );

    /// A seed no confirmatory run uses; the unit tests never run a declared seed.
    const TEST_SEED: u64 = 1001;

    /// The shared reduced run (small worlds of the same structure; seconds in a debug build).
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(TEST_SEED, Scale::Reduced).expect("a reduced run"))
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

    fn all_rules() -> Vec<String> {
        metrics::coverage()
            .into_iter()
            .chain(metrics::falsifiers())
            .collect()
    }

    #[test]
    fn the_seed_is_one_number_or_the_same_number_twice() {
        let args = |items: &[&str]| {
            ["ptr-bench", "weak-supervision"]
                .iter()
                .chain(items)
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(seed_from_args(&args(&["17"])), Ok(17));
        assert_eq!(seed_from_args(&args(&["17", "17"])), Ok(17));
        for bad in [
            args(&[]),
            args(&["17", "18"]),
            args(&["x"]),
            args(&["-1"]),
            args(&["1", "1", "1"]),
        ] {
            assert!(seed_from_args(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("F002"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
    }

    #[test]
    fn every_metric_is_a_json_integer_in_the_harness_output() {
        let report = reduced_report();
        for name in metrics::METRIC_NAMES {
            assert!(
                matches!(emit::resolve(&report.json, name), Some(Json::Int(_))),
                "{name}"
            );
        }
        let unique: std::collections::BTreeSet<&str> = metrics::METRIC_NAMES.into_iter().collect();
        assert_eq!(unique.len(), metrics::METRIC_NAMES.len());
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_an_integer_the_harness_prints() {
        let report = reduced_report();
        for rule in all_rules() {
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
        for rule in all_rules() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let bound = if is_literal(tokens[2]) {
                tokens[2].parse::<i64>().unwrap()
            } else {
                int(&report.json, tokens[2])
            };
            let violating = match tokens[1] {
                ">=" => bound - 1,
                "<=" => bound + 1,
                _ => bound + 1,
            };
            assert_ne!(int(&report.json, tokens[0]), violating, "{rule}");
            let broken = with(&report.json, tokens[0], Json::Int(violating));
            let failed: Vec<String> = judge(&broken, &[], std::slice::from_ref(&rule))
                .into_iter()
                .filter(|check| check.status == Status::Fail)
                .map(|check| check.rule)
                .collect();
            assert_eq!(failed, vec![rule.clone()], "{rule}");
        }
    }

    #[test]
    fn the_rules_name_the_claims_of_the_manifest() {
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let text = manifest["falsification"].as_str().expect("falsification");
        for counter in [
            "veto_contradictions",
            "w1_brier_gain_lower_ppm",
            "w1_ece_gain_lower_ppm",
            "w1_census_accuracy_ds_permille",
        ] {
            assert!(text.contains(counter), "{counter} is not in the text");
        }
        assert_eq!(metrics::falsifiers().len(), 15);
        assert_eq!(metrics::coverage().len(), 14);
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "weak-supervision");
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
            assert!(
                metrics::falsifiers().contains(&format!("{counter} <= 0")),
                "{counter} has no falsifier rule"
            );
        }
        // The plan plants defects in the products and in the harness.
        for prefix in [
            "crates/ptr-labeling/",
            "crates/ptr-analytics/",
            "bins/ptr-bench/",
        ] {
            assert!(
                mutations
                    .iter()
                    .any(|m| m["file"].as_str().is_some_and(|f| f.starts_with(prefix))),
                "{prefix}"
            );
        }
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_and_is_clean_but_for_the_pinned_sizes() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"weak-supervision\",\"experiment\":\"F002\""));
        assert_eq!(int(&report.json, "seed"), TEST_SEED as i64);
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "reduced"
        ));
        assert!(matches!(
            emit::resolve(&report.json, "procedure.shape_is_pinned"),
            Some(Json::Bool(false))
        ));
        let tripped: Vec<&str> = report
            .falsifiers
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(tripped.is_empty(), "{tripped:?}");
        assert_eq!(int(&report.json, "hard_failures"), 0);
        // The reduced worlds hold 150 gold items, below the 500 the pinned run needs; nothing
        // else may be unmet.
        let unmet: Vec<&str> = report
            .coverage
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(
            unmet.iter().all(|rule| rule.starts_with("gold_items")),
            "{unmet:?}"
        );
    }

    #[test]
    fn the_run_sees_every_outcome_and_the_arms_are_a_real_comparison() {
        let json = &reduced_report().json;
        for outcome in ["determined", "estimated", "unknown", "disputed"] {
            assert!(int(json, &format!("{outcome}_seen")) >= 1, "{outcome}");
        }
        assert_eq!(
            int(json, "w1_disputed_items"),
            0,
            "no verifier is wrong in w1"
        );
        assert!(int(json, "w2_disputed_items") >= 1);
        assert_eq!(int(json, "w3_determined_items"), 0, "w3 has no verifier");
        assert!(
            int(json, "w1_mv_veto_violating_items") > 0,
            "the baseline ignores vetoes"
        );
        assert_eq!(int(json, "w1_mvp_veto_violating_items"), 0);
        assert!(int(json, "w1_brier_gain_ppm") > 0 && int(json, "w1_ece_gain_ppm") > 0);
        assert!(int(json, "w1_brier_gain_lower_ppm") <= int(json, "w1_brier_gain_ppm"));
        // The label model is a different arm from the baseline: the same words mean different numbers.
        assert_ne!(int(json, "w1_brier_ds_ppm"), int(json, "w1_brier_mv_ppm"));
        assert_eq!(int(json, "gold_items"), Scale::Reduced.shape().gold as i64);
        assert_eq!(int(json, "negative_controls_tripped"), 7);
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
                    .filter(|(k, _)| k != "timing_ms")
                    .cloned()
                    .collect(),
            )
            .render()
        };
        let first = strip(&reduced_report().json);
        let again = strip(
            &execute(TEST_SEED, Scale::Reduced)
                .expect("a second run")
                .json,
        );
        assert_eq!(first, again);
        let other = strip(&execute(1002, Scale::Reduced).expect("another seed").json);
        assert_ne!(first, other);
    }

    #[test]
    fn every_negative_control_trips_the_counter_built_for_it() {
        let json = &reduced_report().json;
        let Some(Json::Arr(controls)) = emit::resolve(json, "negative_controls") else {
            panic!("negative_controls is an array");
        };
        assert_eq!(controls.len(), 7);
        for control in controls {
            assert!(
                matches!(emit::resolve(control, "tripped"), Some(Json::Bool(true))),
                "{}",
                control.render()
            );
            let Some(Json::Str(counter)) = emit::resolve(control, "counter") else {
                panic!("a counter");
            };
            assert!(
                metrics::HARD_COUNTERS.contains(&counter.as_str()),
                "{counter}"
            );
        }
    }

    #[test]
    fn timings_are_in_their_own_field_and_no_rule_reads_them() {
        assert!(int(&reduced_report().json, "timing_ms.elapsed") >= 0);
        for rule in all_rules() {
            assert!(!rule.contains("timing") && !rule.contains("_ms"), "{rule}");
        }
    }

    #[test]
    fn the_gate_form_constants_match_the_rules() {
        let rules = metrics::falsifiers();
        assert_eq!(
            rules.contains(&"w1_brier_gain_lower_ppm >= 1".to_string()),
            params::BRIER_GATE_ON_LOWER_BOUND
        );
        assert_eq!(
            rules.contains(&"w1_ece_gain_lower_ppm >= 1".to_string()),
            params::ECE_GATE_ON_LOWER_BOUND
        );
    }
}
