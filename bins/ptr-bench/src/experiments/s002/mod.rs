//! Experiment S002, snapshot cancellation, as a harness: `ptr-bench snapshot-cancellation <seed>`
//! runs the experiment's cases once for one seed and prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `snapshot-cancellation <cases> <seed>`) is no evidence. The code under test is the steppable
//! resumable run of `ptr-runtime` (`begin_resumable_run`, `step_resumable_run`, its cooperative
//! cancellation when the semantic revision moves under a run) and the hard effect boundary
//! (`authorize_action`, `prepare_execution_once`, `execute_prepared`), driven only through their
//! public API. Every case is a scheduled run and at most one writer (a host write that moves the
//! revision, one that does not, a generation superseded or revoked) acting at one of five
//! positions of the run's life, executed on a fresh runtime in arm A0 (cancellation off, the
//! baseline) and in arm A1 (cancellation on). The harness' own model (the revision mirror and
//! the ledger scan of `oracle`) decides what must happen; the runtime's answers are compared
//! with it.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names (a
//! JSON integer), `hard_failures`, the timings (never read by a rule), the procedure, the
//! negative controls, the per-arm and per-class tables, the notes, the counters that belong to no
//! metric (`internal`), and the two judgements (`coverage` and `falsifiers`). The exit status is
//! 1 when a coverage prerequisite is unmet, a falsifier is tripped (a panic or a hang of the
//! product included: the harness contains both) or a metric has no value and no declared
//! reason, and 2 when the harness itself cannot run: a write it makes is refused, a control does
//! not trip, its scheduler is inconsistent, or an argument does not parse. A defect of the code
//! under test is never exit 2.

pub mod controls;
pub mod metrics;
pub mod oracle;
pub mod params;
pub mod plan;
pub mod runner;
pub mod world;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use metrics::{Counters, A0, A1};
use params::Scale;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

fn map_json(map: &std::collections::BTreeMap<&'static str, i64>) -> Json {
    Json::Obj(
        map.iter()
            .map(|(name, n)| (name.to_string(), Json::Int(*n)))
            .collect(),
    )
}

fn arm_json(c: &Counters, arm: usize) -> Json {
    Json::obj([
        (
            "stale_boundary_attempts",
            Json::Int(c.stale_boundary_attempts[arm]),
        ),
        ("stale_executed", Json::Int(c.stale_executed[arm])),
        ("boundary_refusals", Json::Int(c.boundary_refusals[arm])),
        ("wasted_units", Json::Int(c.wasted_units[arm])),
        ("total_units", Json::Int(c.total_units[arm])),
    ])
}

fn tally_of(c: &Counters) -> Tally {
    let mut tally = Tally::new(metrics::LIST);
    metrics::fill(&mut tally, c);
    tally.set("executions", c.executions);
    tally.set("retries_refused", c.retries_refused);
    tally.set("cancelled_step_work_units", c.cancelled_step_work_units);
    tally.set(
        "generation_boundary_refusals",
        c.generation_boundary_refusals,
    );
    tally.set("stale_permit_refusals", c.stale_permit_refusals);
    tally.set("revision_refusals", c.revision_refusals);
    tally.set(
        "late_stale_refusals.authorize",
        c.late_stale_refusals_by_checkpoint[0],
    );
    tally.set(
        "late_stale_refusals.prepare",
        c.late_stale_refusals_by_checkpoint[1],
    );
    tally.set(
        "late_stale_refusals.execute",
        c.late_stale_refusals_by_checkpoint[2],
    );
    tally
}

/// The one JSON object of a run, assembled from its counters.
fn assemble(
    seed: u64,
    scale: Scale,
    counters: &Counters,
    controls: &(Json, Json),
    elapsed: Duration,
) -> Report {
    let tally = tally_of(counters);
    let hard = metrics::hard_failures(&tally);
    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("S002")),
        (
            "status_note".to_string(),
            note("one seed of the S002 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push((
        "timing_ns".to_string(),
        Json::obj([("elapsed_ns", Json::Int(elapsed.as_nanos() as i64))]),
    ));
    let positions: Vec<&str> = params::Position::ALL.iter().map(|p| p.name()).collect();
    let effects: Vec<&str> = params::EFFECTS
        .iter()
        .map(|e| params::effect_name(*e))
        .collect();
    pairs.push((
        "procedure".to_string(),
        Json::obj([
            (
                "seed_is_declared",
                Json::Bool(params::SEEDS.contains(&seed)),
            ),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("cases", Json::Int(scale.cases() as i64)),
            ("rounds_min", Json::Int(params::ROUNDS_MIN as i64)),
            ("rounds_max", Json::Int(params::ROUNDS_MAX as i64)),
            ("positions", Json::strs(&positions)),
            ("effects", Json::strs(&effects)),
            (
                "arms",
                Json::strs(&["A0-cancellation-off", "A1-cancellation-on"]),
            ),
            (
                "scheduler",
                note("a deterministic virtual-time event queue ordered by (tick, seq); a run step and each boundary checkpoint is 10 ticks after the previous one and a writer acts 5 ticks before the event it precedes; no clock is read"),
            ),
            (
                "backend",
                note("a scripted backend: the case's rounds of Pod requests, then an action of the case's effect stamped with the revision of the context the backend was last shown; the Pod echoes; the verifiers pass"),
            ),
            (
                "host_policy",
                note("the same in both arms: a run that is cancelled, or whose action the boundary refuses as stale (revision or permit) on a live generation, is begun again once on the current revision and its action executed under the case's at-most-once key; the boundary pass calls authorize_action, prepare_execution_once and execute_prepared without stopping at the first refusal; after an execution the same key and action are offered once more and must be answered from history"),
            ),
            (
                "wasted_units",
                note("steps (a Pod round, or the step that hands out the action) a run takes after a bumping foreign write replaced its snapshot, counted from the runtime's own work (the Pod's invocation count) and the step results, until the run ends; a cancelled step that did work counts its work"),
            ),
            (
                "total_units",
                note("every step of every run of the arm, restarts included; net_units_a1_minus_a0 is arm A1's total minus arm A0's, with its sign (positive: cancelling and beginning again cost more steps in total than going on). Descriptive, ungated, and a headline of the result: cancelling a run that would have resumed on the new snapshot and been accepted costs the pre-write steps again"),
            ),
        ]),
    ));
    pairs.push((
        "controls".to_string(),
        Json::obj([
            ("bypass", controls.0.clone()),
            ("ledger_scan", controls.1.clone()),
        ]),
    ));
    pairs.push((
        "arms".to_string(),
        Json::obj([
            ("A0", arm_json(counters, A0)),
            ("A1", arm_json(counters, A1)),
        ]),
    ));
    pairs.push((
        "cases".to_string(),
        Json::obj([
            ("by_class", map_json(&counters.cases_by_class)),
            ("by_position", map_json(&counters.cases_by_position)),
            ("by_effect", map_json(&counters.cases_by_effect)),
            ("by_writer", map_json(&counters.cases_by_writer)),
            (
                "cancel_latency_histogram",
                Json::Obj(
                    counters
                        .latency_histogram
                        .iter()
                        .map(|(latency, n)| (latency.to_string(), Json::Int(*n)))
                        .collect(),
                ),
            ),
        ]),
    ));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "mirror",
                note("the harness counts the writes it makes (the run's ingest and promoted outputs, the writer's keys) and never reads the runtime's revision into its model; mirror_drift counts the checks where the runtime's revision differs from it, the runs that did not take the shape the scripted chain implies, and the cases where the schedule's label (stale or not, generation dead or not) and the mirror's observation disagree"),
            ),
            (
                "stale",
                note("an action is stale at a checkpoint when its revision is not the revision of the committed ledger replayed to that point (the last committed delta's own revision), or its generation is not the live one there or was revoked; the harness' mirror is compared with that verdict at every checkpoint and a difference is mirror_drift, so stale_actions_executed and stale_checkpoint_admissions are never judged against a lagging mirror; stale_actions_executed counts executor calls made on an action the ledger replay called stale just before the call; stale_effect_attempts_committed counts the EffectAttempted records whose revision or generation was not the one live at the record's own place in a replay of the committed ledger"),
            ),
            (
                "cancel_latency_rounds_max",
                note("steps started since the writer acted, up to and including the step that cancelled: 1 when the run is cancelled by the very next step"),
            ),
            (
                "gated_claim",
                note("cancellation never weakens the boundary (no stale action executes or is attempted in either arm, judged against the committed ledger, and the boundary answers a post-run action identically in both arms) and a cancelled run takes no step on a replaced snapshot (wasted_units_a1 is 0: this restates the cancellation contract, missed_cancellations is its independent check, and it is not a compute saving); not claimed: that cancelling is cheaper in total (total_units_a0, total_units_a1 and net_units_a1_minus_a0 are descriptive headline numbers)"),
            ),
        ]),
    ));
    let mut internal = match tally.internal() {
        Json::Obj(pairs) => pairs,
        other => vec![("counters".to_string(), other)],
    };
    internal.push((
        "stale_refusal_by_checkpoint".to_string(),
        Json::Arr(
            counters
                .late_stale_refusals_by_checkpoint
                .iter()
                .map(|n| Json::Int(*n))
                .collect(),
        ),
    ));
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
    Report {
        line: json.render(),
        json,
        coverage,
        falsifiers,
        gaps,
    }
}

/// What the watchdog sees of a run in progress.
struct Progress {
    cases_done: AtomicU64,
    finished: AtomicBool,
    snapshot: Mutex<Counters>,
}

/// Prints the run as it stands with one product hang counted and exits 1: a case that does not
/// end is a falsification, not a harness that went silent.
fn watchdog(
    progress: Arc<Progress>,
    seed: u64,
    scale: Scale,
    controls: (Json, Json),
    started: Instant,
    hang_seconds: u64,
) {
    let mut last = progress.cases_done.load(Ordering::SeqCst);
    let mut idle = 0u64;
    loop {
        std::thread::sleep(Duration::from_secs(1));
        if progress.finished.load(Ordering::SeqCst) {
            return;
        }
        let now = progress.cases_done.load(Ordering::SeqCst);
        if now != last {
            last = now;
            idle = 0;
            continue;
        }
        idle += 1;
        if idle < hang_seconds {
            continue;
        }
        let mut counters = progress
            .snapshot
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        counters.product_hangs += 1;
        let report = assemble(seed, scale, &counters, &controls, started.elapsed());
        println!("{}", report.line);
        eprintln!(
            "{}: the product did not return from case {now} for {hang_seconds} seconds",
            params::BENCHMARK
        );
        std::process::exit(1);
    }
}

/// Run every case of `seed` at `scale` and assemble the one JSON object. A run the harness
/// cannot make is an `Err` with the reason. `hang_seconds` starts the watchdog.
pub fn execute(seed: u64, scale: Scale, hang_seconds: Option<u64>) -> Result<Report, String> {
    let started = Instant::now();
    let mut counters = Counters::default();
    let controls = controls::run(&mut counters)?;
    let progress = Arc::new(Progress {
        cases_done: AtomicU64::new(0),
        finished: AtomicBool::new(false),
        snapshot: Mutex::new(counters.clone()),
    });
    if let Some(seconds) = hang_seconds {
        let (progress, controls) = (progress.clone(), controls.clone());
        std::thread::Builder::new()
            .name("s002-watchdog".to_string())
            .spawn(move || watchdog(progress, seed, scale, controls, started, seconds))
            .map_err(|error| format!("the watchdog could not start: {error}"))?;
    }
    let outcome = run_cases(seed, scale, &mut counters, &progress);
    progress.finished.store(true, Ordering::SeqCst);
    outcome?;
    Ok(assemble(
        seed,
        scale,
        &counters,
        &controls,
        started.elapsed(),
    ))
}

fn run_cases(
    seed: u64,
    scale: Scale,
    counters: &mut Counters,
    progress: &Progress,
) -> Result<(), String> {
    for index in 0..scale.cases() {
        let case = plan::plan_of(seed, index);
        counters.runs += 1;
        counters.stale_runs_by_schedule += i64::from(case.stale_by_schedule());
        *counters
            .cases_by_class
            .entry(case.class.name())
            .or_insert(0) += 1;
        *counters
            .cases_by_effect
            .entry(params::effect_name(case.effect))
            .or_insert(0) += 1;
        if let Some(injection) = case.injection {
            *counters
                .cases_by_position
                .entry(injection.position.name())
                .or_insert(0) += 1;
        }
        let a0 = runner::run_arm(&case, A0, counters)?;
        let a1 = runner::run_arm(&case, A1, counters)?;
        // Where both arms handed the boundary an action, it must be the same action and the
        // boundary must answer it the same way: cancellation does not touch the boundary.
        if !a0.aborted && !a1.aborted {
            if let (Some(first), Some(second)) = (&a0.first_pass, &a1.first_pass) {
                if first != second {
                    counters.boundary_arm_divergences += 1;
                }
            }
        }
        if let Some(injection) = case.injection {
            *counters
                .cases_by_writer
                .entry(injection.writer.name())
                .or_insert(0) += 1;
        }
        if let Ok(mut snapshot) = progress.snapshot.lock() {
            *snapshot = counters.clone();
        }
        progress.cases_done.fetch_add(1, Ordering::SeqCst);
    }
    Ok(())
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
        [cases, seed] => parse(cases, "cases").and_then(|cases| {
            if cases == 0 {
                return Err("cases must be at least 1".to_string());
            }
            parse(seed, "seed").map(|seed| (seed, Scale::Custom(cases as usize)))
        }),
        _ => Err(
            "usage: snapshot-cancellation <seed> | snapshot-cancellation <cases> <seed>"
                .to_string(),
        ),
    }
}

/// The first few panics of the product are printed; the rest are only counted, so a defective
/// product that panics in every case does not bury the result in stderr.
fn quiet_panics() {
    let printed = std::sync::atomic::AtomicUsize::new(0);
    std::panic::set_hook(Box::new(move |info| {
        if printed.fetch_add(1, Ordering::SeqCst) < 3 {
            eprintln!("panic in the code under test: {info}");
        }
    }));
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after
/// the subcommand: `<seed>` (the pinned scale) or `<cases> <seed>` (the mutation checker's form:
/// that many cases, no evidence). Exits 1 when the run is not clean and 2 when the harness
/// cannot run.
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
    quiet_panics();
    print_and_exit(execute(seed, scale, Some(params::HANG_SECONDS)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const CRITERIA: &str =
        include_str!("../../../../../experiments/semdb/S002-snapshot-cancellation/criteria.toml");
    const MUTATIONS: &str = include_str!(
        "../../../../../experiments/semdb/S002-snapshot-cancellation/tests/mutations.toml"
    );
    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/semdb/S002-snapshot-cancellation/PREREGISTRATION.md"
    );

    /// The shared reduced run (a few hundred cases; well under a second in a debug build).
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(17, Scale::Reduced, None).expect("a reduced run"))
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
        assert_eq!(table["experiment"].as_str(), Some("S002"));
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
            let bound = if is_literal(tokens[2]) {
                tokens[2].parse::<i64>().unwrap()
            } else {
                int(&report.json, tokens[2])
            };
            let violating = match tokens[1] {
                ">=" => bound - 1,
                _ => bound + 1,
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
        assert_eq!(metrics::falsifiers().len(), 17);
        assert_eq!(metrics::coverage().len(), 15);
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "snapshot-cancellation");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
        assert_eq!(manifest["status"].as_str(), Some("planned"));
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
        assert_eq!(mutations.len(), 21);
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"snapshot-cancellation\",\"experiment\":\"S002\""));
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

    /// The real runtime trips no falsifier on the reduced run, and the reduced run is a real
    /// comparison: both arms reach the boundary with stale actions, arm A1 cancels, arm A0
    /// does the work the cancellation saves.
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
        assert_eq!(int(json, "runs"), params::REDUCED_CASES as i64);
        assert!(int(json, "boundary_refusals_a0") >= 1);
        assert!(int(json, "boundary_refusals_a1") >= 1);
        assert!(int(json, "boundary_refusals_a0") > int(json, "boundary_refusals_a1"));
        assert!(int(json, "cancelled_runs") >= 1);
        assert_eq!(int(json, "cancel_latency_rounds_max"), 1);
        assert!(int(json, "wasted_units_a0") >= 1);
        assert_eq!(int(json, "wasted_units_a1"), 0);
        assert_eq!(int(json, "effect_kinds_covered"), 4);
        assert_eq!(int(json, "injection_positions_covered"), 5);
        assert!(int(json, "generation_stale_cases") >= 1);
        assert!(int(json, "noop_writes_observed") >= 1);
        assert!(int(json, "retries_replayed") >= 1);
        assert!(int(json, "restarted_runs") >= 1);
        assert_eq!(int(json, "control_trips"), 5);
        // Nothing stale executed or was attempted, in either arm, and the baseline met stale
        // attempts that it refused.
        assert_eq!(int(json, "stale_actions_executed"), 0);
        assert_eq!(int(json, "stale_action_rate_permille_a0"), 0);
        assert!(int(json, "stale_boundary_attempts_a0") >= int(json, "stale_boundary_attempts_a1"));
        assert_eq!(
            int(json, "stale_boundary_attempts_a0"),
            int(json, "boundary_refusals_a0"),
            "every stale attempt of the baseline was refused"
        );
        assert_eq!(
            int(json, "stale_boundary_attempts_a1"),
            int(json, "boundary_refusals_a1")
        );
        // Each case executed at most once per arm.
        assert!(int(json, "internal.executions") <= 2 * int(json, "runs"));
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
        let again = strip(
            &execute(17, Scale::Reduced, None)
                .expect("a second run")
                .json,
        );
        assert_eq!(first, again);
        let other = strip(
            &execute(29, Scale::Reduced, None)
                .expect("another seed")
                .json,
        );
        assert_ne!(first, other);
    }

    #[test]
    fn the_negative_controls_trip_and_are_in_the_json() {
        let json = &reduced_report().json;
        for control in ["bypass", "ledger_scan"] {
            assert!(
                matches!(
                    emit::resolve(json, &format!("controls.{control}.required_rule_tripped")),
                    Some(Json::Bool(true))
                ),
                "{control}"
            );
        }
        assert_eq!(int(json, "controls.ledger_scan.planted_stale_attempts"), 1);
        assert_eq!(int(json, "controls.ledger_scan.clean_stale_attempts"), 0);
    }

    #[test]
    fn a_hang_is_a_falsification_and_the_run_still_prints_a_complete_object() {
        // What the watchdog prints: the counters so far with one hang counted.
        let counters = Counters {
            runs: 5,
            product_hangs: 1,
            ..Counters::default()
        };
        let report = assemble(
            17,
            Scale::Pinned,
            &counters,
            &(Json::Null, Json::Null),
            Duration::from_secs(1),
        );
        assert_eq!(int(&report.json, "product_hangs"), 1);
        assert_eq!(int(&report.json, "hard_failures"), 1);
        assert!(report
            .falsifiers
            .iter()
            .any(|check| check.rule == "product_hangs <= 0" && check.status == Status::Fail));
        assert!(report.gaps.is_empty());
        for name in metrics::METRIC_NAMES {
            assert!(
                matches!(emit::resolve(&report.json, name), Some(Json::Int(_))),
                "{name}"
            );
        }
    }

    #[test]
    fn timings_are_in_their_own_field_and_no_rule_reads_them() {
        let json = &reduced_report().json;
        assert!(int(json, "timing_ns.elapsed_ns") > 0);
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("timing") && !rule.contains("_ns"), "{rule}");
            assert!(!rule.contains("total_units"), "{rule}");
        }
    }

    #[test]
    fn the_arguments_are_a_seed_or_cases_and_a_seed_and_nothing_else() {
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
    fn a_custom_scale_run_is_marked_and_the_runtime_still_holds_every_hard_counter() {
        let report = execute(17, Scale::Custom(30), None).expect("a custom run");
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
            "thirty cases cannot cover the pinned shape"
        );
    }

    #[test]
    fn the_preregistration_draft_states_its_limits_and_the_review_it_waits_for() {
        for needle in [
            "Not freezable until the invariant review",
            "Total compute is not claimed to fall",
            "Two runs interleaved on one `PtrRuntime` cancel each other",
            "A **lifecycle record moves no semantic revision**",
            "Equivalent mutants",
            "a panic or a hang of the\nproduct is a falsification",
        ] {
            assert!(PREREGISTRATION.contains(needle), "{needle}");
        }
    }
}
