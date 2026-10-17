//! Experiment E008 Part A (CPU, no network) as a harness: `ptr-bench split-pod-extensions
//! <seed>` runs every Part A case once for one seed and prints exactly one JSON object on
//! stdout. `split-pod-extensions <cases> <seed>` (the mutation checker's form) scales the three
//! seeded case families down or up; the matrix and the controls always run in full, and a run
//! at another scale is no evidence.
//!
//! E008 is `planned`: nothing here is frozen, no `PREREGISTRATION.md`, `criteria.toml` or
//! `aggregate.py` exists yet, and the coverage, falsifier and validity lists of
//! [`metrics`] are this harness's reading of `experiment.toml` for the head to pin.
//!
//! What runs, against the version 2 frame of `ptr-protocol` and the version 2 downstream stage
//! of `ptr-runtime` (a real `PtrRuntime` ledger and a real placement controller):
//!
//! * A1, A2, A5: the 45-cell invalidation coverage matrix ([`matrix`]), every cell on its own;
//! * A1: the Raw ancestry cases, A3: the mode and taint cases, A4: the persistence cases
//!   ([`cases`]);
//! * the three control baselines ([`controls`]), each of which must produce the failure it
//!   exists to show;
//! * the kind checker: every element of every frame built maps to exactly one matrix kind and
//!   every kind is reached (`matrix_frame_bijection_violations`).
//!
//! The exit status, as [`Report::exit_status`] and `run_standing` implement it:
//!
//! * 2 (the run is broken or vacuous, whatever else fired) when a case could not run, a
//!   validity counter fired (the procedure differs from the planned one), a coverage
//!   prerequisite is unmet, a metric has a gap, or a control did not trip (a dead control
//!   proves nothing);
//! * 1 only when a falsifier is tripped by an observed product violation and nothing in the
//!   first list applies; the one exception is a dead control beside an observed product
//!   violation with every validity counter at 0, which does not become 2 (controls come
//!   after falsifiers);
//! * 0 when every rule holds.

pub mod cases;
pub mod controls;
pub mod matrix;
pub mod metrics;
pub mod params;
pub mod world;

use crate::experiments::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use crate::experiments::layered_kv::lab::Res;
use ptr_protocol::activation::{
    bijection_violations, ActivationFrameV2, DependencyKind, MAX_LINEAGE_INPUTS, MAX_RAW_ANCESTRY,
};
use std::time::Instant;
use world::{FrameSpec, World};

/// The most cases a custom scale may ask for per family.
pub const MAX_CASES: u64 = 10_000;

/// How big the run is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    Custom(u64),
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Custom(_) => "custom",
        }
    }

    pub fn cases(self) -> u64 {
        match self {
            Self::Pinned => params::PINNED_CASES,
            Self::Custom(count) => count,
        }
    }
}

pub struct Report {
    pub line: String,
    #[cfg_attr(not(test), allow(dead_code))]
    pub json: Json,
    pub coverage: Vec<Check>,
    pub falsifiers: Vec<Check>,
    /// The rules that say the record describes the planned run; a failing one is the harness's
    /// fault.
    pub validity: Vec<Check>,
    pub gaps: Gaps,
}

impl Report {
    pub fn coverage_unmet(&self) -> usize {
        count(&self.coverage, Status::Fail)
    }

    pub fn falsifiers_tripped(&self) -> usize {
        count(&self.falsifiers, Status::Fail)
    }

    pub fn validity_violated(&self) -> usize {
        count(&self.validity, Status::Fail)
    }

    /// The exit status of the printed record (AGENTS.md: 1 is a product finding, 2 a harness
    /// failure): 2 when a validity counter is violated, a coverage prerequisite is unmet or a
    /// metric has a gap (a vacuous or broken run, whatever else fired), else 1 when a
    /// falsifier is tripped, else 0.
    pub fn exit_status(&self) -> i32 {
        if self.validity_violated() > 0 || self.coverage_unmet() > 0 || !self.gaps.is_empty() {
            2
        } else if self.falsifiers_tripped() > 0 {
            1
        } else {
            0
        }
    }
}

/// Whether an observed product violation fired.
fn observed_product_fault(get: &dyn Fn(&str) -> i64) -> bool {
    metrics::product_counters().iter().any(|name| get(name) > 0)
}

/// Whether a validity counter fired.
fn validity_fired(get: &dyn Fn(&str) -> i64) -> bool {
    metrics::HARNESS_COUNTERS.iter().any(|name| get(name) > 0)
}

/// Whether a product violation is the finding of this run: observed, and no validity counter
/// fired (validity first).
fn product_finding(get: &dyn Fn(&str) -> i64) -> bool {
    !validity_fired(get) && observed_product_fault(get)
}

/// Whether the cases and controls leave a run that can be judged. `Err` is the harness's fault
/// (exit 2); `Ok` lets the record be printed, with exit 1 if a falsifier fired. A case that
/// could not run is a validity failure, which precedes every falsifier; a dead control only
/// yields to an observed product finding (controls come after falsifiers).
fn run_standing(
    get: &dyn Fn(&str) -> i64,
    control_trips: &dyn Fn(&str) -> i64,
    failures: &[String],
) -> Result<(), String> {
    if !failures.is_empty() {
        return Err(format!("a case could not run: {}", failures.join("; ")));
    }
    for control in controls::CONTROLS {
        if control_trips(control) < 1 && !product_finding(get) {
            return Err(format!(
                "the control {control} did not trip: its probe proves nothing"
            ));
        }
    }
    Ok(())
}

/// The procedure as the run observed it against the planned one; a difference is a procedure
/// mismatch.
fn procedure(tally: &mut Tally, seed: u64, scale: Scale) -> Json {
    let pairs: usize = DependencyKind::ALL
        .iter()
        .map(|kind| kind.invalidations().len())
        .sum();
    let observed = [
        (matrix::cells().len() as u64, params::MATRIX_CELLS as u64),
        (DependencyKind::ALL.len() as u64, 7),
        (pairs as u64, 15),
        (matrix::Path::ALL.len() as u64, params::PATHS.len() as u64),
        (MAX_LINEAGE_INPUTS as u64, 64),
        (MAX_RAW_ANCESTRY as u64, 256),
    ];
    let mismatches = observed
        .iter()
        .filter(|(seen, pinned)| seen != pinned)
        .count() as i64;
    tally.add("procedure_mismatches", mismatches);
    let seeds = params::declared_seeds();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(seeds.contains(&seed))),
        ("seeds", Json::ints(seeds.iter().map(|seed| *seed as i64))),
        ("cases_per_family", Json::Int(scale.cases() as i64)),
        (
            "dependency_kinds",
            Json::strs(
                &DependencyKind::ALL
                    .iter()
                    .map(|kind| kind.name())
                    .collect::<Vec<_>>(),
            ),
        ),
        ("paths", Json::strs(&params::PATHS)),
        ("matrix_cells", Json::Int(matrix::cells().len() as i64)),
        ("matrix_cell_names", Json::strs(&matrix::cell_names())),
        ("max_lineage_inputs", Json::Int(MAX_LINEAGE_INPUTS as i64)),
        ("max_raw_ancestry", Json::Int(MAX_RAW_ANCESTRY as i64)),
        ("frame_rows", Json::Int(i64::from(params::FRAME_ROWS))),
        ("frame_width", Json::Int(i64::from(params::FRAME_WIDTH))),
        ("cache_capacity", Json::Int(params::CACHE_CAPACITY as i64)),
    ])
}

/// Run every Part A case for `seed`.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let mut tally = Tally::new(metrics::LIST);
    for name in metrics::ZERO_BASED.iter().chain(&metrics::INTERNAL) {
        tally.add(name, 0);
    }
    for name in [
        "raw_ancestor_omission_cases",
        "input_lineage_revalidation_cases",
        "upstream_expert_fence_cases",
        "upstream_adapter_invalidation_race_cases",
        "upstream_base_model_invalidation_cases",
        "approximate_frame_downstream_exact_lookup_cases",
        "downstream_persist_rejection_cases",
        "procedure_mismatches",
    ] {
        tally.add(name, 0);
    }
    let cell_names = matrix::cell_names();
    tally.seed_group("invalidation_matrix_cases", &cell_names);
    tally.seed_group("invalidation_matrix_stale_hits", &cell_names);
    tally.seed_group("control_trips", &controls::CONTROLS);

    let procedure_json = procedure(&mut tally, seed, scale);
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut lap = Instant::now();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };

    // A case that cannot run is a failure of the run. When a falsifier counter has fired as
    // well, the product is at fault and the record is still printed (status 1); otherwise it
    // is the harness's (status 2).
    let mut failures: Vec<String> = Vec::new();
    macro_rules! case {
        ($name:expr, $body:expr) => {
            if let Err(error) = $body {
                failures.push(format!("{}: {error}", $name));
            }
        };
    }
    let cases_per_family = scale.cases();
    let mut frames: Vec<ActivationFrameV2> = Vec::new();
    case!("matrix", matrix::run(&mut tally, seed, &mut frames));
    phase("matrix", &mut lap);
    case!(
        "raw ancestry",
        cases::raw_ancestry_cases(&mut tally, seed, cases_per_family)
    );
    case!(
        "modes",
        cases::mode_cases(&mut tally, seed, cases_per_family)
    );
    case!(
        "persistence",
        cases::persistence_cases(&mut tally, seed, cases_per_family)
    );
    phase("cases", &mut lap);
    case!(
        "control: revoke-trusting cache",
        controls::revoke_trusting(&mut tally, seed)
    );
    phase("controls", &mut lap);

    // The kind checker over a reference frame (every kind present) and every frame the matrix
    // built.
    case!(
        "bijection",
        (|| -> Res<()> {
            let reference = World::new()?.frame(&FrameSpec::exact(vec![[1; 32]], seed));
            let mut all: Vec<&ActivationFrameV2> = vec![&reference];
            all.extend(frames.iter());
            tally.add(
                "matrix_frame_bijection_violations",
                bijection_violations(&all) as i64,
            );
            Ok(())
        })()
    );
    phase("bijection", &mut lap);

    run_standing(
        &|name| tally.get(name),
        &|control| {
            tally
                .group("control_trips")
                .into_iter()
                .find(|(name, _)| name == control)
                .map_or(0, |(_, value)| value)
        },
        &failures,
    )?;
    tally.set("case_failures", failures.len() as i64);

    tally.unsupported(
        "restore_path",
        "no KV is restored on a Pod, and the downstream cache has no persistence (no snapshot, tier write or restore): the restore path of the invalidation coverage matrix does not apply",
    );
    tally.unsupported(
        "loopback_and_real_model_metrics",
        "Part B (loopback iroh) and Part C (a real model on a GPU) are other harnesses; this one is Part A, CPU, no network",
    );
    let unsupported_names = tally.unsupported_names();
    let (printed, gaps) = tally.metrics();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("E008")),
        ("part".to_string(), Json::str("A")),
        (
            "status_note".to_string(),
            Json::str("one seed of the E008 Part A harness: E008 is planned, so this is no evidence until the head has frozen it; the exit status and the coverage and falsifiers lists below are the harness's own reading of the manifest"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
        (
            "elapsed_ms".to_string(),
            Json::Float(started.elapsed().as_secs_f64() * 1000.0),
        ),
    ];
    pairs.extend(printed);
    for name in [
        "verdict_mismatches",
        "attempts_without_typed_rejection",
        "control_inconsistencies",
    ] {
        pairs.push((name.to_string(), Json::Int(tally.get(name))));
    }
    for cell in &cell_names {
        let hits = tally
            .group("invalidation_matrix_stale_hits")
            .into_iter()
            .find(|(name, _)| name == cell)
            .map_or(0, |(_, value)| value);
        pairs.push((metrics::cell_counter(cell), Json::Int(hits)));
    }
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([("phases", Json::Obj(phases))]),
    ));
    pairs.push(("procedure".to_string(), procedure_json));
    pairs.push(("unsupported".to_string(), tally.unsupported_json()));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "invalidation_matrix_cases",
                Json::str("one or two cases per cell, each on a fresh world with a real ledger and placement controller; a stale hit is the entry served after the invalidation, a use finished or the buffered frame still live, or the old frame admitted again"),
            ),
            (
                "matrix_frame_bijection_violations",
                Json::str("the slot/kind mapping checked as a bijection both ways, every element of every frame built classified once, every one of the seven kinds reached"),
            ),
            (
                "downstream_restore_attempts_admitted",
                Json::str("a lookup or a finished use on a restarted or cleared cache that hits; the persist attempts themselves cannot succeed (their success type is uninhabited)"),
            ),
            (
                "untainted_outputs_from_approximate_frames",
                Json::str("an Approximate hit whose released taint is not the frame's, or whose taint the runtime's effect boundary (authorize_action_with_taint) does not refuse"),
            ),
        ]),
    ));
    pairs.push(("internal".to_string(), tally.internal()));
    let root = Json::Obj(pairs.clone());
    let coverage = judge(&root, &unsupported_names, &metrics::coverage());
    let falsifiers = judge(&root, &unsupported_names, &metrics::falsifiers());
    let validity = judge(&root, &unsupported_names, &metrics::validity());
    let tripped = count(&falsifiers, Status::Fail);
    pairs.insert(7, ("hard_failures".to_string(), Json::Int(tripped as i64)));
    pairs.push((
        "coverage".to_string(),
        Json::Arr(coverage.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "falsifiers".to_string(),
        Json::Arr(falsifiers.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "validity".to_string(),
        Json::Arr(validity.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "summary".to_string(),
        Json::obj([
            (
                "coverage_unmet",
                Json::Int(count(&coverage, Status::Fail) as i64),
            ),
            ("falsifiers_tripped", Json::Int(tripped as i64)),
            (
                "validity_violated",
                Json::Int(count(&validity, Status::Fail) as i64),
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
        validity,
        gaps,
    })
}

fn print_and_exit(report: &Report) {
    println!("{}", report.line);
    for check in report
        .coverage
        .iter()
        .chain(&report.falsifiers)
        .chain(&report.validity)
    {
        if check.status == Status::Fail {
            eprintln!("{} ({})", check.rule, check.observed.render());
        }
    }
    for (metric, reason) in &report.gaps {
        eprintln!("harness gap {metric}: {reason}");
    }
    match report.exit_status() {
        0 => {}
        status => std::process::exit(status),
    }
}

/// `split-pod-extensions <seed>` | `split-pod-extensions <cases> <seed>`: Part A.
pub fn run(args: &[String]) {
    let parsed: Result<(Scale, u64), String> = match args {
        [seed] => seed
            .parse::<u64>()
            .map(|seed| (Scale::Pinned, seed))
            .map_err(|_| format!("seed must be a non-negative integer, got {seed:?}")),
        [cases, seed] => cases
            .parse::<u64>()
            .ok()
            .filter(|count| (1..=MAX_CASES).contains(count))
            .zip(seed.parse::<u64>().ok())
            .map(|(count, seed)| (Scale::Custom(count), seed))
            .ok_or_else(|| {
                format!(
                    "usage: split-pod-extensions <cases in 1..={MAX_CASES}> <seed>, got {args:?}"
                )
            }),
        _ => Err("usage: split-pod-extensions <seed> | <cases> <seed>".to_string()),
    };
    let (scale, seed) = match parsed {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("ptr-bench: {message}");
            std::process::exit(2);
        }
    };
    let seeds = params::declared_seeds();
    if !seeds.contains(&seed) {
        eprintln!("seed {seed} is not one of the manifest's seeds {seeds:?}; this is a dry run");
    }
    match execute(seed, scale) {
        Ok(report) => print_and_exit(&report),
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

    const EXPERIMENT_TOML: &str = params::EXPERIMENT_TOML;

    /// One reduced run (two cases per family), shared by the tests that read its output.
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(17, Scale::Custom(2)).expect("a reduced run"))
    }

    #[test]
    fn the_metric_list_is_the_manifests_part_a_metrics_in_its_order() {
        let manifest: Vec<&str> = EXPERIMENT_TOML
            .lines()
            .find(|line| line.starts_with("metrics"))
            .and_then(|line| line.split_once('"'))
            .and_then(|(_, rest)| rest.rsplit_once('"'))
            .map(|(names, _)| names.split(", ").collect())
            .expect("the manifest names its metrics");
        let mut at = 0;
        for name in &metrics::METRIC_NAMES[..20] {
            let found = manifest[at..]
                .iter()
                .position(|candidate| candidate == name)
                .unwrap_or_else(|| panic!("{name} is not in the manifest, in order"));
            at += found + 1;
        }
        assert_eq!(metrics::METRIC_NAMES[20], "procedure_mismatches");
        assert!(manifest.contains(&"procedure_mismatches"));
        assert_eq!(
            params::declared_seeds(),
            vec![17, 29, 43, 71, 101],
            "the manifest's seeds"
        );
    }

    #[test]
    fn every_rule_is_three_tokens_and_names_a_path_the_harness_prints() {
        let report = reduced_report();
        for rule in metrics::coverage()
            .iter()
            .chain(&metrics::falsifiers())
            .chain(&metrics::validity())
        {
            let tokens: Vec<&str> = rule.split(' ').collect();
            assert_eq!(tokens.len(), 3, "{rule}");
            for token in [tokens[0], tokens[2]] {
                if token.parse::<i64>().is_err() {
                    assert!(
                        emit::resolve(&report.json, token).is_some(),
                        "{rule}: {token} does not resolve in the harness JSON"
                    );
                }
            }
        }
    }

    fn with(root: &Json, path: &str, value: Json) -> Json {
        fn go(at: &Json, parts: &[&str], value: &Json) -> Json {
            match (at, parts.split_first()) {
                (_, None) => value.clone(),
                (Json::Obj(pairs), Some((first, rest))) => Json::Obj(
                    pairs
                        .iter()
                        .map(|(key, inner)| {
                            if key == first {
                                (key.clone(), go(inner, rest, value))
                            } else {
                                (key.clone(), inner.clone())
                            }
                        })
                        .collect(),
                ),
                _ => at.clone(),
            }
        }
        go(root, &path.split('.').collect::<Vec<_>>(), &value)
    }

    #[test]
    fn every_rule_can_fail_when_its_counter_is_set_to_a_violation() {
        let report = reduced_report();
        for rule in metrics::coverage()
            .iter()
            .chain(&metrics::falsifiers())
            .chain(&metrics::validity())
        {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let (left, op, right) = (tokens[0], tokens[1], tokens[2]);
            let bound: i64 = right.parse().expect("a literal bound");
            let value = if op == ">=" { bound - 1 } else { bound + 1 };
            let violating = with(&report.json, left, Json::Int(value));
            assert_eq!(
                emit::evaluate(&violating, rule),
                Ok(false),
                "{rule} did not fail when violated"
            );
        }
    }

    #[test]
    fn the_reduced_run_trips_no_falsifier_covers_the_45_cells_and_no_control_is_dead() {
        let report = reduced_report();
        assert_eq!(report.falsifiers_tripped(), 0, "{:?}", report.falsifiers);
        assert_eq!(report.validity_violated(), 0, "{:?}", report.validity);
        assert_eq!(report.coverage_unmet(), 0, "{:?}", report.coverage);
        assert!(report.gaps.is_empty(), "{:?}", report.gaps);
        assert_eq!(report.exit_status(), 0);
        let Some(Json::Obj(cells)) = emit::resolve(&report.json, "invalidation_matrix_cases")
        else {
            panic!("the matrix cases are not printed");
        };
        assert_eq!(cells.len(), 45);
        for control in controls::CONTROLS {
            let path = format!("control_trips.{control}");
            let Some(Json::Int(trips)) = emit::resolve(&report.json, &path) else {
                panic!("{path} is not printed");
            };
            assert!(*trips >= 1, "{control}");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_record_apart_from_timing() {
        fn stable(json: &Json) -> String {
            let Json::Obj(pairs) = json else { panic!() };
            Json::Obj(
                pairs
                    .iter()
                    .filter(|(key, _)| key != "elapsed_ms" && key != "timing_ms")
                    .cloned()
                    .collect(),
            )
            .render()
        }
        let first = stable(&reduced_report().json);
        let second = execute(17, Scale::Custom(2)).expect("a second run");
        assert_eq!(first, stable(&second.json));
        let other = execute(29, Scale::Custom(2)).expect("another seed");
        assert_ne!(first, stable(&other.json), "the seed must matter");
    }

    #[test]
    fn a_hard_counter_is_a_top_level_integer_of_the_record() {
        let report = reduced_report();
        for name in metrics::hard_counters() {
            assert!(
                matches!(emit::resolve(&report.json, &name), Some(Json::Int(_))),
                "{name} is no top-level integer"
            );
        }
    }

    fn counters<'a>(fired: &'a [&'a str]) -> impl Fn(&str) -> i64 + 'a {
        move |name| i64::from(fired.contains(&name))
    }

    #[test]
    fn a_dead_control_or_a_failed_case_with_no_finding_is_the_harness_fault() {
        let none = counters(&[]);
        let failures = vec!["matrix: boom".to_string()];
        assert!(run_standing(&none, &|_| 1, &failures).is_err());
        assert!(run_standing(&none, &|_| 0, &[]).is_err());
        assert!(run_standing(&none, &|_| 1, &[]).is_ok());
    }

    #[test]
    fn an_observed_violation_takes_precedence_over_a_dead_control_and_validity_over_both() {
        let stale = counters(&["outputs_after_upstream_invalidation"]);
        assert!(run_standing(&stale, &|_| 0, &[]).is_ok());
        // A case that could not run is the harness's failure even beside a product finding.
        assert!(run_standing(&stale, &|_| 1, &["x".to_string()]).is_err());
        let invalid = counters(&[
            "outputs_after_upstream_invalidation",
            "procedure_mismatches",
        ]);
        assert!(run_standing(&invalid, &|_| 0, &[]).is_err());
        assert!(validity_fired(&invalid));
        assert!(!product_finding(&invalid));
        // The harness's own counters are no product finding.
        assert!(!observed_product_fault(&counters(&[
            "procedure_mismatches",
            "control_inconsistencies"
        ])));
    }

    fn check(rule: &str, status: Status) -> Check {
        Check {
            rule: rule.to_string(),
            observed: Json::Null,
            status,
        }
    }

    fn report_with(falsifier: Status, validity: Status, coverage: Status) -> Report {
        Report {
            line: String::new(),
            json: Json::Null,
            coverage: vec![check("c", coverage)],
            falsifiers: vec![check("f", falsifier)],
            validity: vec![check("v", validity)],
            gaps: Vec::new(),
        }
    }

    #[test]
    fn the_exit_status_reserves_1_for_a_product_finding_and_2_for_a_harness_failure() {
        use Status::{Fail, Pass};
        assert_eq!(report_with(Pass, Pass, Pass).exit_status(), 0);
        assert_eq!(report_with(Fail, Pass, Pass).exit_status(), 1);
        // A vacuous run (unmet coverage) is the harness's failure, whatever else fired.
        assert_eq!(report_with(Pass, Pass, Fail).exit_status(), 2);
        assert_eq!(report_with(Fail, Pass, Fail).exit_status(), 2);
        assert_eq!(report_with(Fail, Fail, Fail).exit_status(), 2);
        assert_eq!(report_with(Pass, Fail, Pass).exit_status(), 2);
        // So is a metric with a gap.
        let mut gap = report_with(Fail, Pass, Pass);
        gap.gaps.push(("m".to_string(), "no data".to_string()));
        assert_eq!(gap.exit_status(), 2);
    }
}
