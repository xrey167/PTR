//! Experiment F007, the whole VCW loop on seeded stub experts, as a harness:
//! `ptr-bench vcw-loop <seed>` runs the pinned plan for one seed and prints exactly one JSON
//! object on stdout; `ptr-bench vcw-loop <requests> <seed>` is the mutation checker's reduced
//! form (no evidence); `ptr-bench vcw-loop --preflight` runs one tiny request per arm.
//!
//! One run is one seed's evidence. It measures, for every world and arm, the verified-success of
//! each request, paired across arms, and audits every request against an independent oracle
//! (the planted defects and the clean text) and against the harness's own restatement of the
//! loop's rules. The verdict across seeds is the aggregator's (`aggregate.py`, `criteria.toml`):
//! the harness reports counts and the integrity of the run.
//!
//! The simulated Pods' per-code skill tables are the assumption under test (`world.rs`). A gain
//! measured here is a consequence of those tables, bounded by them; nothing is said about any
//! model.
//!
//! Exit status: 0 clean; 1 when a coverage prerequisite is unmet, a falsifier is tripped or a
//! control fails (the object says which); 2 when the harness itself cannot run (a training
//! stream fails, the arguments are wrong, the harness panics). A panic of the product is a
//! counted falsifier (`product_panics`), never exit 2.

pub mod arms;
pub mod lane;
pub mod learn;
pub mod metrics;
pub mod params;
pub mod task;
pub mod world;

use std::time::Instant;

use super::layered_kv::emit::{count, judge, Check, Json, Status};
use arms::{run_world, Arm, Integrity, Pair, WorldRun, ARMS};
use params::Scale;
use world::WorldId;

/// What a run produced, before it is printed.
pub struct Report {
    pub line: String,
    #[cfg_attr(not(test), allow(dead_code))]
    pub json: Json,
    pub coverage: Vec<Check>,
    pub falsifiers: Vec<Check>,
    pub invalid: Vec<Check>,
}

impl Report {
    pub fn clean(&self) -> bool {
        count(&self.coverage, Status::Fail) == 0
            && count(&self.falsifiers, Status::Fail) == 0
            && count(&self.invalid, Status::Fail) == 0
    }
}

/// The harness sources, the crate manifest, the lock file and the pinned configuration, as compiled: their digest is in every result, so
/// that the code a result came from can be recognised (the freeze re-runs a pilot on the committed
/// tree and requires the same outcome digests).
const SOURCES: [(&str, &str); 12] = [
    ("mod.rs", include_str!("mod.rs")),
    ("params.rs", include_str!("params.rs")),
    ("task.rs", include_str!("task.rs")),
    ("world.rs", include_str!("world.rs")),
    ("lane.rs", include_str!("lane.rs")),
    ("learn.rs", include_str!("learn.rs")),
    ("arms.rs", include_str!("arms.rs")),
    ("metrics.rs", include_str!("metrics.rs")),
    ("Cargo.toml", include_str!("../../../Cargo.toml")),
    ("Cargo.lock", include_str!("../../../../../Cargo.lock")),
    ("config.toml", params::CONFIG_TOML),
    (
        "criteria.toml",
        include_str!("../../../../../experiments/feedback/F007-vcw-loop/criteria.toml"),
    ),
];

pub fn source_digest() -> String {
    let mut bytes = Vec::new();
    for (name, text) in SOURCES {
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    }
    hex(&ptr_ledger::integrity::sha256(&bytes))
}

fn int(value: u64) -> Json {
    Json::Int(i64::try_from(value).unwrap_or(i64::MAX))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn arm_json(arm: &arms::ArmResult) -> Json {
    let kinds = task::KINDS
        .iter()
        .map(|kind| {
            (
                kind.name().to_string(),
                Json::obj([
                    ("requests", int(arm.by_kind[kind.index()][0])),
                    ("successes", int(arm.by_kind[kind.index()][1])),
                ]),
            )
        })
        .collect();
    let bits: Vec<u8> = arm.success.iter().map(|ok| u8::from(*ok)).collect();
    Json::obj([
        ("requests", int(arm.requests)),
        ("successes", int(arm.successes)),
        ("verified", int(arm.verified)),
        ("first_pass", int(arm.first_pass)),
        ("broadcast_reads", int(arm.broadcast_reads)),
        ("exhausted", int(arm.exhausted)),
        ("escalated", int(arm.escalated)),
        ("verifier_calls", int(arm.verifier_calls)),
        ("pod_calls", int(arm.pod_calls)),
        ("rounds", int(arm.rounds)),
        (
            "elapsed_ns",
            Json::Int(i64::try_from(arm.elapsed_ns).unwrap_or(i64::MAX)),
        ),
        ("births", int(arm.births)),
        ("l3_refusals", int(arm.l3_refusals)),
        ("repairs_applied", int(arm.repairs_applied)),
        ("repairs_declined", int(arm.repairs_declined)),
        ("regenerations", int(arm.regenerations)),
        ("unroutable_codes", int(arm.unroutable_codes)),
        ("by_kind", Json::Obj(kinds)),
        (
            "hard_by_code",
            Json::Obj(
                arm.hard_by_code
                    .iter()
                    .map(|(code, n)| (code.clone(), int(*n)))
                    .collect(),
            ),
        ),
        (
            "outcome_digest",
            Json::str(hex(&ptr_ledger::integrity::sha256(&bits))),
        ),
    ])
}

fn pairs_json(world: &WorldRun) -> Json {
    let mut pairs = Vec::new();
    for (i, a) in ARMS.iter().enumerate() {
        for b in &ARMS[i + 1..] {
            let table = Pair::of(world.arm(*a), world.arm(*b));
            pairs.push((
                format!("{}__{}", a.name(), b.name()),
                Json::obj([
                    ("both", int(table.both)),
                    ("only_a", int(table.only_a)),
                    ("only_b", int(table.only_b)),
                    ("neither", int(table.neither)),
                ]),
            ));
        }
    }
    Json::Obj(pairs)
}

fn float_milli(value: f64) -> Json {
    Json::Int((value * 1_000_000.0).round() as i64)
}

fn training_json(world: &WorldRun) -> Json {
    Json::Obj(
        world
            .trained
            .kinds
            .iter()
            .map(|kind| {
                (
                    kind.kind.name().to_string(),
                    Json::obj([
                        ("training_requests", int(kind.training_requests)),
                        (
                            "seeds",
                            Json::ints(kind.seeds.iter().map(|s| i64::try_from(*s).unwrap_or(0))),
                        ),
                        ("vcw_promoted", Json::Bool(kind.vcw_promoted)),
                        ("vcw_prior", int(kind.vcw_prior)),
                        ("vcw_difference_micro", float_milli(kind.vcw_difference)),
                        ("vcw_lcb_micro", float_milli(kind.vcw_lcb)),
                        ("vcw_records", int(kind.vcw_records as u64)),
                        ("vcw_policy_digest", Json::str(hex(&kind.vcw_digest))),
                        (
                            "vcw_candidates_micro",
                            Json::Arr(
                                kind.vcw_candidates
                                    .iter()
                                    .map(|(p, v)| Json::Arr(vec![int(*p), float_milli(*v)]))
                                    .collect(),
                            ),
                        ),
                        ("vcw_edge", Json::Bool(kind.vcw_edge)),
                        ("vcw_edge_excess_micro", float_milli(kind.vcw_edge_excess)),
                        ("gate_edge_excess_micro", float_milli(kind.gate_edge_excess)),
                        ("bon_edge_excess_micro", float_milli(kind.bon_edge_excess)),
                        ("vcw_edge_se_micro", float_milli(kind.vcw_edge_se)),
                        ("gate_edge_se_micro", float_milli(kind.gate_edge_se)),
                        ("bon_edge_se_micro", float_milli(kind.bon_edge_se)),
                        ("gate_edge_test_sees", Json::Bool(kind.gate_edge_test_sees)),
                        ("bon_edge_test_sees", Json::Bool(kind.bon_edge_test_sees)),
                        ("gate_edge", Json::Bool(kind.gate_edge)),
                        ("bon_edge", Json::Bool(kind.bon_edge)),
                        ("bon_promoted", Json::Bool(kind.bon_promoted)),
                        ("bon_tau_permille", int(kind.bon_tau)),
                        ("bon_difference_micro", float_milli(kind.bon_difference)),
                        ("bon_lcb_micro", float_milli(kind.bon_lcb)),
                        (
                            "bon_candidates_micro",
                            Json::Arr(
                                kind.bon_candidates
                                    .iter()
                                    .map(|(t, v)| Json::Arr(vec![int(*t), float_milli(*v)]))
                                    .collect(),
                            ),
                        ),
                        ("gate_promoted", Json::Bool(kind.gate_promoted)),
                        ("gate_tau_permille", int(kind.gate_tau)),
                        ("gate_difference_micro", float_milli(kind.gate_difference)),
                        ("gate_lcb_micro", float_milli(kind.gate_lcb)),
                        ("gate_records", int(kind.gate_records as u64)),
                        (
                            "gate_candidates_micro",
                            Json::Arr(
                                kind.gate_candidates
                                    .iter()
                                    .map(|(t, v)| Json::Arr(vec![int(*t), float_milli(*v)]))
                                    .collect(),
                            ),
                        ),
                    ]),
                )
            })
            .collect(),
    )
}

fn world_json(world: &WorldRun) -> Json {
    Json::obj([
        ("requests", int(world.requests as u64)),
        ("gated", Json::Bool(world.world.gated())),
        ("training", training_json(world)),
        (
            "arms",
            Json::Obj(
                ARMS.iter()
                    .map(|arm| (arm.name().to_string(), arm_json(world.arm(*arm))))
                    .collect(),
            ),
        ),
        ("pairs", pairs_json(world)),
        ("l3_population", l3_population_json(world)),
        (
            "first_candidates_differ",
            int(world.first_draft_differences()),
        ),
    ])
}

/// The requests whose first candidate carried the unrepairable code as a hard finding, as the 2x2
/// table of the full loop against the loop without L3. The first candidate is the shared first draft,
/// identical in every arm (`first_candidates_differ` at world level pins that), so the population is
/// fixed before any arm acts.
fn l3_population_json(world: &WorldRun) -> Json {
    let (full, blind) = (world.arm(Arm::Full), world.arm(Arm::NoL3));
    let mut table = Pair::default();
    for (index, carries) in blind.gap_first.iter().enumerate() {
        if !*carries {
            continue;
        }
        match (full.success[index], blind.success[index]) {
            (true, true) => table.both += 1,
            (true, false) => table.only_a += 1,
            (false, true) => table.only_b += 1,
            (false, false) => table.neither += 1,
        }
    }
    Json::obj([
        ("both", int(table.both)),
        ("only_a", int(table.only_a)),
        ("only_b", int(table.only_b)),
        ("neither", int(table.neither)),
    ])
}

fn worlds_to_run(scale: Scale) -> Vec<WorldId> {
    match scale {
        Scale::Pinned => WorldId::ALL.to_vec(),
        // The reduced forms skip the descriptive `mild` world; the controls need the others.
        Scale::Custom(_) => vec![WorldId::Hetero, WorldId::Homog, WorldId::Ctx],
    }
}

/// Run every world once for `seed` at `scale` and assemble the one JSON object. A run the
/// harness cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let mut worlds = Vec::new();
    for world in worlds_to_run(scale) {
        worlds.push(run_world(world, seed, scale)?);
    }
    let mut integrity = Integrity::default();
    for world in &worlds {
        for arm in &world.arms {
            integrity.add(&arm.integrity);
        }
    }
    for world in &worlds {
        for kind in &world.trained.kinds {
            integrity.loop_errors += kind.failures.loop_errors;
            integrity.product_panics += kind.failures.panics;
        }
    }
    let controls = metrics::controls(&worlds, arms::ctx_probe(seed, 90));
    let hetero = worlds
        .iter()
        .find(|w| w.world == WorldId::Hetero)
        .ok_or("no primary world")?;
    let full = hetero.arm(Arm::Full);
    let shares = full
        .by_kind
        .iter()
        .map(|[n, _]| (*n as i64) * 1000 / full.requests.max(1) as i64);
    let min_share = shares.min().unwrap_or(0);
    let min_events = task::all_codes()
        .iter()
        .map(|code| {
            hetero
                .arms
                .iter()
                .map(|a| a.hard_by_code.get(*code).copied().unwrap_or(0))
                .sum::<u64>() as i64
        })
        .min()
        .unwrap_or(0);
    let exhausted = full.exhausted;
    let counters = [
        ("false_accepts", integrity.false_accepts),
        ("unverified_returns", integrity.unverified_returns),
        ("double_verifications", integrity.double_verifications),
        ("budget_violations", integrity.budget_violations),
        (
            "routing_outside_admissible",
            integrity.routing_outside_admissible,
        ),
        ("verdict_mismatches", integrity.verdict_mismatches),
        ("verdict_overrides", integrity.verdict_overrides),
        (
            "verifier_call_mismatches",
            integrity.verifier_call_mismatches,
        ),
        ("calls_after_acceptance", integrity.calls_after_acceptance),
        ("l3_unadmitted", integrity.l3_unadmitted),
        ("loop_errors", integrity.loop_errors),
        ("product_panics", integrity.product_panics),
        ("exhausted_as_success", integrity.exhausted_as_success),
    ];
    debug_assert!(counters
        .iter()
        .map(|(name, _)| *name)
        .eq(metrics::FALSIFIER_COUNTERS));
    let control_total: i64 = controls.counters.iter().map(|(_, n)| *n).sum();
    let coverage_object = Json::obj([
        ("min_kind_share_permille", Json::Int(min_share)),
        ("min_hard_events_per_code", Json::Int(min_events)),
        ("repairs_applied", int(full.repairs_applied)),
        ("exhausted_runs", int(exhausted)),
        ("births", int(full.births)),
    ]);
    let coverage_checks = judge(
        &Json::Obj(vec![("coverage".to_string(), coverage_object.clone())]),
        &[],
        &metrics::coverage(),
    );
    let coverage_unmet = count(&coverage_checks, Status::Fail) as i64;
    let hard: i64 =
        counters.iter().map(|(_, n)| *n as i64).sum::<i64>() + control_total + coverage_unmet;

    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("F007")),
        (
            "status_note".to_string(),
            Json::str("one seed of the F007 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml. Simulated Pods with declared per-code skill tables: a gain measured here is a consequence of those tables; no claim about any model."),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    for (name, value) in counters {
        pairs.push((name.to_string(), int(value)));
    }
    for (name, failures) in &controls.counters {
        pairs.push(((*name).to_string(), Json::Int(*failures)));
    }
    pairs.push(("coverage_unmet".to_string(), Json::Int(coverage_unmet)));
    for (name, check) in metrics::COVERAGE_COUNTERS.iter().zip(&coverage_checks) {
        pairs.push((
            (*name).to_string(),
            Json::Int(i64::from(check.status == Status::Fail)),
        ));
    }
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push(("coverage".to_string(), coverage_object));
    pairs.push((
        "control_observations_permille".to_string(),
        Json::Obj(
            controls
                .observed
                .iter()
                .map(|(name, value)| (name.clone(), Json::Int(*value)))
                .collect(),
        ),
    ));
    pairs.push((
        "worlds".to_string(),
        Json::Obj(
            worlds
                .iter()
                .map(|world| (world.world.name().to_string(), world_json(world)))
                .collect(),
        ),
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
                Json::Bool(seed >= params::PILOT_SEED_FLOOR),
            ),
            ("source_digest", Json::str(source_digest())),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("requests_per_world", int(scale.requests() as u64)),
            (
                "verifier_call_budget",
                Json::Int(i64::from(params::VERIFIER_CALL_BUDGET)),
            ),
            (
                "training_seeds",
                Json::ints(
                    params::training_seeds(seed)
                        .iter()
                        .map(|s| i64::try_from(*s).unwrap_or(0)),
                ),
            ),
            (
                "worlds",
                Json::strs(&worlds.iter().map(|w| w.world.name()).collect::<Vec<_>>()),
            ),
            (
                "arms",
                Json::strs(&ARMS.iter().map(|a| a.name()).collect::<Vec<_>>()),
            ),
            (
                "elapsed_ns",
                Json::Int(i64::try_from(started.elapsed().as_nanos()).unwrap_or(i64::MAX)),
            ),
        ]),
    ));
    let root = Json::Obj(pairs.clone());
    let coverage = judge(&root, &[], &metrics::coverage());
    let falsifiers = judge(&root, &[], &metrics::falsifiers());
    let invalid = judge(&root, &[], &metrics::invalid_if());
    pairs.push((
        "coverage_checks".to_string(),
        Json::Arr(coverage.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "falsifier_checks".to_string(),
        Json::Arr(falsifiers.iter().map(Check::json).collect()),
    ));
    pairs.push((
        "invalid_checks".to_string(),
        Json::Arr(invalid.iter().map(Check::json).collect()),
    ));
    let json = Json::Obj(pairs);
    Ok(Report {
        line: json.render(),
        json,
        coverage,
        falsifiers,
        invalid,
    })
}

/// One tiny request per arm, to be sure before a seed starts that every arm builds and serves.
/// Prints a line and returns an error naming the arm that failed.
pub fn preflight() -> Result<String, String> {
    let scale = Scale::Custom(30);
    let seed = params::PILOT_SEED_FLOOR + 90;
    let trained = learn::train(WorldId::Hetero, seed, Scale::Custom(60))?;
    let tasks = arms::request_stream(seed, 6);
    let mut served = Vec::new();
    for arm in ARMS {
        let result = arms::run_arm(arm, WorldId::Hetero, seed, &trained, &tasks);
        let bad = result.integrity.loop_errors + result.integrity.product_panics;
        if result.requests != tasks.len() as u64 || bad != 0 {
            return Err(format!(
                "preflight: arm {} served {} of {} requests with {} loop errors or panics",
                arm.name(),
                result.requests,
                tasks.len(),
                bad
            ));
        }
        served.push(format!("{}={}", arm.name(), result.requests));
    }
    let _ = scale;
    Ok(format!("preflight ok: {}", served.join(" ")))
}

/// The command line of the subcommand: `<seed>`, `<requests> <seed>` or `--preflight`.
pub enum Mode {
    Run { seed: u64, scale: Scale },
    Preflight,
}

pub fn parse_args(args: &[String]) -> Result<Mode, String> {
    let number = |text: &str, what: &str| -> Result<u64, String> {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    match args {
        [flag] if flag == "--preflight" => Ok(Mode::Preflight),
        [seed] => Ok(Mode::Run {
            seed: number(seed, "seed")?,
            scale: Scale::Pinned,
        }),
        [requests, seed] => {
            let requests = number(requests, "requests")? as usize;
            if requests == 0 {
                return Err("requests must be positive".to_string());
            }
            Ok(Mode::Run {
                seed: number(seed, "seed")?,
                scale: Scale::Custom(requests),
            })
        }
        _ => Err(
            "usage: vcw-loop <seed> | vcw-loop <requests> <seed> | vcw-loop --preflight"
                .to_string(),
        ),
    }
}

/// Entry point of the subcommand; never returns.
pub fn run(args: &[String]) -> ! {
    let mode = match parse_args(args) {
        Ok(mode) => mode,
        Err(message) => {
            eprintln!("vcw-loop: {message}");
            std::process::exit(2);
        }
    };
    let outcome = std::panic::catch_unwind(|| match mode {
        Mode::Preflight => preflight().map(|line| (line, true)),
        Mode::Run { seed, scale } => execute(seed, scale).map(|report| {
            let clean = report.clean();
            (report.line, clean)
        }),
    });
    match outcome {
        Ok(Ok((line, clean))) => {
            println!("{line}");
            std::process::exit(i32::from(!clean));
        }
        Ok(Err(message)) => {
            eprintln!("vcw-loop: the harness cannot run: {message}");
            std::process::exit(2);
        }
        Err(_) => {
            eprintln!("vcw-loop: the harness panicked");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arms::{arm_config, audit_for_tests, request_stream, run_arm, Pair};
    use lane::{EndTag, Lane, LaneConfig, RequestRecord, RouteScorer};
    use std::sync::Arc;
    use task::{Kind, Task, KINDS};
    use world::RegistryMode;

    const CRITERIA_TOML: &str =
        include_str!("../../../../../experiments/feedback/F007-vcw-loop/criteria.toml");

    /// Seeds of the unit tests: pilot seeds, never a declared one.
    const TEST_SEED: u64 = 1001;

    fn rules(key: &str) -> Vec<String> {
        let parsed: toml::Table = CRITERIA_TOML.parse().expect("criteria.toml parses");
        parsed[key]
            .as_array()
            .expect("a list")
            .iter()
            .map(|rule| rule.as_str().expect("a string").to_string())
            .collect()
    }

    fn uniform_lane(
        world: WorldId,
        kind: Kind,
        registry: RegistryMode,
        l3: bool,
        broadcast: bool,
    ) -> Lane {
        Lane::new(LaneConfig {
            world,
            kind,
            registry,
            scorer: RouteScorer::Active(super::learn::book_for(kind).scorer()),
            hints_steer: false,
            schema: false,
            l3,
            broadcast,
            sim_seed: 5,
            router_seed: 9,
            learner: None,
            calls: params::VERIFIER_CALL_BUDGET,
            shared_first_draft: true,
        })
    }

    fn serve_all(lane: &mut Lane, kind: Kind, n: usize) -> Vec<(bool, u32)> {
        (0..n)
            .map(|index| {
                let task = Arc::new(Task::generate(TEST_SEED, index as u64, kind));
                let record = lane.serve(&format!("t-{index}"), &task);
                let ok = matches!(record.tag, EndTag::Verified)
                    && record
                        .accepted
                        .as_ref()
                        .is_some_and(|p| task::Judge::independent_pass(&task, p));
                let calls = record
                    .trace
                    .as_ref()
                    .map_or(0, |t| t.spent().verifier_calls);
                (ok, calls)
            })
            .collect()
    }

    #[test]
    fn the_rule_lists_of_criteria_toml_are_the_harness_lists_in_order() {
        assert_eq!(rules("coverage"), metrics::coverage());
        assert_eq!(rules("falsifiers"), metrics::falsifiers());
        assert_eq!(rules("invalid_if"), metrics::invalid_if());
        let parsed: toml::Table = CRITERIA_TOML.parse().expect("parses");
        assert_eq!(parsed["experiment"].as_str(), Some("F007"));
        assert!(parsed.contains_key("decision"));
    }

    #[test]
    fn a_small_run_resolves_every_rule_and_every_rule_can_fail() {
        let report = execute(TEST_SEED + 20, Scale::Custom(40)).expect("a small run");
        let json = report.json.clone();
        for key in ["coverage", "falsifiers", "invalid_if"] {
            for rule in rules(key) {
                let checks = judge(&json, &[], std::slice::from_ref(&rule));
                assert!(
                    checks[0].status != Status::Unsupported,
                    "{rule} did not resolve: {:?}",
                    checks[0]
                );
                // Set the counter to a violation and the same rule must fail.
                let tokens: Vec<&str> = rule.split(' ').collect();
                let (left, op, right) = (tokens[0], tokens[1], tokens[2]);
                let bad = if op == "<=" {
                    right.parse::<i64>().expect("literal") + 1
                } else {
                    right.parse::<i64>().expect("literal") - 1
                };
                let doctored = set_path(&json, left, Json::Int(bad));
                let checks = judge(&doctored, &[], std::slice::from_ref(&rule));
                assert_eq!(checks[0].status, Status::Fail, "{rule} cannot fail");
            }
        }
        assert_eq!(report.falsifiers.len(), metrics::FALSIFIER_COUNTERS.len());
        assert_eq!(
            count(&report.falsifiers, Status::Fail),
            0,
            "a correct product trips nothing"
        );
    }

    fn set_path(root: &Json, path: &str, value: Json) -> Json {
        fn go(node: &Json, keys: &[&str], value: &Json) -> Json {
            match (node, keys) {
                (_, []) => value.clone(),
                (Json::Obj(pairs), [first, rest @ ..]) => Json::Obj(
                    pairs
                        .iter()
                        .map(|(name, child)| {
                            if name == first {
                                (name.clone(), go(child, rest, value))
                            } else {
                                (name.clone(), child.clone())
                            }
                        })
                        .collect(),
                ),
                _ => node.clone(),
            }
        }
        let keys: Vec<&str> = path.split('.').collect();
        go(root, &keys, &value)
    }

    #[test]
    fn a_run_is_a_pure_function_of_its_seed() {
        let a = execute(TEST_SEED + 21, Scale::Custom(30)).expect("run");
        let b = execute(TEST_SEED + 21, Scale::Custom(30)).expect("run");
        let strip = |json: &Json| {
            // elapsed times differ between runs; everything else must not
            let text = json.render();
            let mut out = String::new();
            let mut skip = false;
            for part in text.split("\"elapsed_ns\":") {
                if skip {
                    out.push_str(part.trim_start_matches(|c: char| c.is_ascii_digit()));
                } else {
                    out.push_str(part);
                }
                skip = true;
            }
            out
        };
        assert_eq!(strip(&a.json), strip(&b.json));
        let c = execute(TEST_SEED + 22, Scale::Custom(30)).expect("run");
        assert_ne!(strip(&a.json), strip(&c.json));
    }

    #[test]
    fn a_pilot_run_is_not_marked_confirmatory_and_a_custom_run_is_not_pinned() {
        let report = execute(TEST_SEED + 23, Scale::Custom(20)).expect("run");
        let procedure = json_field(&report.json, "procedure");
        assert_eq!(
            json_field(procedure, "seed_is_declared"),
            &Json::Bool(false)
        );
        assert_eq!(json_field(procedure, "seed_is_pilot"), &Json::Bool(true));
        assert_eq!(json_field(procedure, "shape_is_pinned"), &Json::Bool(false));
        assert_eq!(json_field(&report.json, "scale"), &Json::str("custom"));
    }

    fn json_field<'a>(json: &'a Json, key: &str) -> &'a Json {
        super::super::layered_kv::emit::resolve(json, key).unwrap_or_else(|| panic!("{key}"))
    }

    #[test]
    fn every_arm_serves_a_stream_with_a_clean_audit_in_every_kind_of_world() {
        let trained =
            learn::train(WorldId::Hetero, TEST_SEED + 24, Scale::Custom(60)).expect("training");
        let tasks = request_stream(TEST_SEED + 24, 40);
        for world in WorldId::ALL {
            for arm in arms::ARMS {
                let result = run_arm(arm, world, TEST_SEED + 24, &trained, &tasks);
                assert_eq!(result.requests, 40);
                let i = &result.integrity;
                let total = i.false_accepts
                    + i.unverified_returns
                    + i.double_verifications
                    + i.budget_violations
                    + i.routing_outside_admissible
                    + i.verdict_mismatches
                    + i.verdict_overrides
                    + i.verifier_call_mismatches
                    + i.calls_after_acceptance
                    + i.l3_unadmitted
                    + i.loop_errors
                    + i.product_panics
                    + i.exhausted_as_success;
                assert_eq!(total, 0, "{} in {}: {:?}", arm.name(), world.name(), i);
            }
        }
    }

    #[test]
    fn the_budget_is_matched_no_arm_ever_spends_more_verifier_calls_than_it_is_given() {
        let trained =
            learn::train(WorldId::Hetero, TEST_SEED + 25, Scale::Custom(60)).expect("training");
        let tasks = request_stream(TEST_SEED + 25, 60);
        for arm in arms::ARMS {
            let result = run_arm(arm, WorldId::Hetero, TEST_SEED + 25, &trained, &tasks);
            assert!(
                result.verifier_calls <= u64::from(params::VERIFIER_CALL_BUDGET) * result.requests,
                "{}",
                arm.name()
            );
            assert!(
                result.verifier_calls >= result.requests,
                "every request verifies once at least"
            );
        }
    }

    #[test]
    fn the_true_map_beats_the_shuffled_map_with_uniform_routing_and_the_null_world_ties() {
        for kind in KINDS {
            let mut rates = Vec::new();
            for registry in [RegistryMode::True, RegistryMode::Shuffled] {
                let mut lane = uniform_lane(WorldId::Hetero, kind, registry, false, true);
                let served = serve_all(&mut lane, kind, 240);
                rates.push(served.iter().filter(|(ok, _)| *ok).count());
            }
            assert!(rates[0] > rates[1] + 12, "{kind:?}: {rates:?}");
        }
        let kind = Kind::Json;
        let mut rates = Vec::new();
        for registry in [
            RegistryMode::True,
            RegistryMode::Shuffled,
            RegistryMode::Generic,
        ] {
            let mut lane = uniform_lane(WorldId::Homog, kind, registry, false, true);
            rates.push(
                serve_all(&mut lane, kind, 400)
                    .iter()
                    .filter(|(ok, _)| *ok)
                    .count() as i64,
            );
        }
        assert!(
            (rates[0] - rates[1]).abs() < 40 && (rates[0] - rates[2]).abs() < 40,
            "{rates:?}"
        );
    }

    #[test]
    fn withholding_the_broadcast_changes_nothing_unless_a_pod_reads_it() {
        let kind = Kind::Arithmetic;
        let run = |world: WorldId, broadcast: bool| -> Vec<bool> {
            let mut lane = uniform_lane(world, kind, RegistryMode::True, false, broadcast);
            serve_all(&mut lane, kind, 200)
                .into_iter()
                .map(|(ok, _)| ok)
                .collect()
        };
        assert_eq!(run(WorldId::Hetero, true), run(WorldId::Hetero, false));
        assert_ne!(run(WorldId::Ctx, true), run(WorldId::Ctx, false));
    }

    #[test]
    fn l3_births_a_specialist_for_the_code_nobody_repairs_only_through_admission() {
        let kind = Kind::Code;
        let mut with = uniform_lane(WorldId::Hetero, kind, RegistryMode::True, true, true);
        let before = serve_all(&mut with, kind, 360);
        assert!(with.births >= 1, "no Birth in 360 code requests");
        assert_eq!(
            with.audit_l3(),
            0,
            "every structural commit is an admitted proposal"
        );
        let mut without = uniform_lane(WorldId::Hetero, kind, RegistryMode::True, false, true);
        let blind = serve_all(&mut without, kind, 360);
        let wins = |s: &[(bool, u32)]| s.iter().filter(|(ok, _)| *ok).count();
        assert!(
            wins(&before) >= wins(&blind),
            "{} vs {}",
            wins(&before),
            wins(&blind)
        );
        assert_eq!(without.births, 0);
        // The homogeneous world has no gap: nothing is unroutable and nothing is born.
        let mut homog = uniform_lane(WorldId::Homog, kind, RegistryMode::True, true, true);
        serve_all(&mut homog, kind, 360);
        assert_eq!(homog.births, 0);
    }

    fn record_with(
        tag: EndTag,
        accepted: Option<task::Served>,
        events: Vec<task::VerifyEvent>,
        task: &Arc<Task>,
    ) -> RequestRecord {
        RequestRecord {
            tag,
            accepted,
            events,
            trace: None,
            task: Arc::clone(task),
            elapsed_ns: 0,
        }
    }

    fn event(
        clean: bool,
        passes: bool,
        planted: &[&str],
        codes: &[&str],
        digest: u8,
    ) -> task::VerifyEvent {
        task::VerifyEvent {
            digest: [digest; 32],
            clean,
            status: if passes {
                ptr_verifier::VerificationStatus::Pass
            } else {
                ptr_verifier::VerificationStatus::Fail
            },
            level: ptr_types::VerificationLevel::Deterministic,
            hard_codes: codes.iter().map(|c| (*c).to_string()).collect(),
            any_hard: !codes.is_empty(),
            planted: planted.iter().map(|c| (*c).to_string()).collect(),
        }
    }

    #[test]
    fn the_audit_counts_each_integrity_violation_it_is_built_to_see() {
        use ptr_runtime::vcw::experts::DefectPayload;
        let task = Arc::new(Task::generate(TEST_SEED, 3, Kind::Arithmetic));
        let lane = uniform_lane(
            WorldId::Hetero,
            Kind::Arithmetic,
            RegistryMode::True,
            false,
            true,
        );
        let clean = task::with_task(&task, || task::Served::clean("x"));
        let defective = clean.with_defect("arith.value");
        // A defective candidate returned as verified: a false accept, with the verifier lying.
        let lying = event(false, true, &["arith.value"], &[], 1);
        let result = audit_for_tests(
            &record_with(
                EndTag::Verified,
                Some(defective.clone()),
                vec![lying],
                &task,
            ),
            &lane,
        );
        assert_eq!(result.false_accepts, 1);
        assert_eq!(
            result.verdict_mismatches, 1,
            "the verifier passed what the oracle rejects"
        );
        // A returned candidate that no verifier call accepted.
        let result = audit_for_tests(
            &record_with(
                EndTag::Verified,
                Some(clean.clone()),
                vec![event(true, false, &[], &["arith.value"], 2)],
                &task,
            ),
            &lane,
        );
        assert_eq!(result.unverified_returns, 1);
        assert_eq!(result.verdict_mismatches, 1);
        // Calls after an accepted report.
        let ok = event(true, true, &[], &[], 3);
        let result = audit_for_tests(
            &record_with(
                EndTag::Verified,
                Some(clean.clone()),
                vec![ok.clone(), ok.clone()],
                &task,
            ),
            &lane,
        );
        assert_eq!(result.calls_after_acceptance, 1);
        // A Fail that names a code nobody planted.
        let result = audit_for_tests(
            &record_with(
                EndTag::Exhausted,
                None,
                vec![event(false, false, &["arith.value"], &["arith.bounds"], 4)],
                &task,
            ),
            &lane,
        );
        assert_eq!(result.verdict_mismatches, 1);
        // A loop error and a panic are product defects, not harness failures.
        let result = audit_for_tests(&record_with(EndTag::Error, None, vec![], &task), &lane);
        assert_eq!(result.loop_errors, 1);
        let result = audit_for_tests(&record_with(EndTag::Panic, None, vec![], &task), &lane);
        assert_eq!(result.product_panics, 1);
        // A Verified outcome without an answer.
        let result = audit_for_tests(&record_with(EndTag::Verified, None, vec![], &task), &lane);
        assert_eq!(result.exhausted_as_success, 1);
    }

    #[test]
    fn the_gate_scorer_cannot_see_a_finding_code_and_prefers_the_pod_that_clears() {
        use ptr_router::{FallbackReason, FindingCode, RouteClass};
        use ptr_types::PodId;
        let trained =
            learn::train(WorldId::Hetero, TEST_SEED + 26, Scale::Custom(300)).expect("training");
        let gate = &trained.of(Kind::Json).gate;
        let repair = |code: &str| RouteClass::Repair(FindingCode::new(code));
        for pod in world::REPAIRERS {
            let pod = PodId::from(pod.id);
            assert_eq!(
                gate.mean(&repair("json.type"), &pod),
                gate.mean(&repair("json.extra"), &pod),
                "the gate does not depend on the finding code"
            );
        }
        let generalist = PodId::from("gen_r");
        let off_role = PodId::from("fmt_1");
        assert!(
            gate.mean(&repair("json.type"), &generalist)
                > gate.mean(&repair("json.type"), &off_role),
            "the best Pod over the kind is the generalist"
        );
        let regen = RouteClass::Regenerate(FallbackReason::Unknown);
        let proposer = PodId::from("gen_2");
        assert_ne!(
            gate.mean(&regen, &proposer),
            gate.mean(&repair("json.type"), &proposer)
        );
    }

    #[test]
    fn training_uses_pilot_streams_and_a_tuning_budget_equal_for_both_learned_arms() {
        let trained =
            learn::train(WorldId::Hetero, TEST_SEED + 27, Scale::Custom(60)).expect("training");
        for kind in &trained.kinds {
            assert!(kind.seeds.iter().all(|s| *s >= params::PILOT_SEED_FLOOR));
            assert!(kind.seeds.iter().all(|s| !params::SEEDS.contains(s)));
            assert_eq!(kind.vcw_candidates.len(), params::PRIORS.len());
            assert_eq!(kind.gate_candidates.len(), params::GATE_TAU_PERMILLE.len());
            assert_eq!(params::PRIORS.len(), params::GATE_TAU_PERMILLE.len());
        }
    }

    #[test]
    fn preflight_serves_every_arm_and_the_arguments_are_strict() {
        assert!(preflight().expect("preflight").starts_with("preflight ok"));
        assert!(matches!(
            parse_args(&["--preflight".into()]),
            Ok(Mode::Preflight)
        ));
        assert!(matches!(
            parse_args(&["17".into()]),
            Ok(Mode::Run {
                seed: 17,
                scale: Scale::Pinned
            })
        ));
        assert!(matches!(
            parse_args(&["50".into(), "9".into()]),
            Ok(Mode::Run {
                seed: 9,
                scale: Scale::Custom(50)
            })
        ));
        for bad in [
            vec![],
            vec!["x".into()],
            vec!["0".into(), "1".into()],
            vec!["1".into(), "2".into(), "3".into()],
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_pair_table_counts_the_same_requests_for_both_arms() {
        let trained =
            learn::train(WorldId::Hetero, TEST_SEED + 28, Scale::Custom(60)).expect("training");
        let tasks = request_stream(TEST_SEED + 28, 50);
        let a = run_arm(Arm::Full, WorldId::Hetero, TEST_SEED + 28, &trained, &tasks);
        let b = run_arm(
            Arm::NoWorkspace,
            WorldId::Hetero,
            TEST_SEED + 28,
            &trained,
            &tasks,
        );
        let pair = Pair::of(&a, &b);
        assert_eq!(pair.both + pair.only_a + pair.only_b + pair.neither, 50);
        assert_eq!(
            pair.only_a + pair.only_b,
            0,
            "no Pod reads the broadcast here"
        );
        let cfg = arm_config(Arm::Shuffled, WorldId::Hetero, Kind::Json, 17, &trained);
        assert_eq!(cfg.registry, RegistryMode::Shuffled);
    }

    #[test]
    fn the_edge_neighbour_is_the_inner_candidate_of_an_edge_only() {
        use learn::edge_neighbour;
        assert_eq!(edge_neighbour(9, 0), Some(1));
        assert_eq!(edge_neighbour(9, 8), Some(7));
        assert_eq!(edge_neighbour(9, 4), None);
        assert_eq!(edge_neighbour(1, 0), None);
    }

    #[test]
    fn the_oracle_reference_uses_the_best_pod_and_the_good_proposer_and_best_of_n_beats_single_shot(
    ) {
        let trained =
            learn::train(WorldId::Hetero, TEST_SEED + 29, Scale::Custom(120)).expect("training");
        let tasks = request_stream(TEST_SEED + 29, 120);
        let oracle = run_arm(
            Arm::Oracle,
            WorldId::Hetero,
            TEST_SEED + 29,
            &trained,
            &tasks,
        );
        let uniform_pool = run_arm(
            Arm::NoFinding,
            WorldId::Hetero,
            TEST_SEED + 29,
            &trained,
            &tasks,
        );
        assert!(
            oracle.successes > uniform_pool.successes + 8,
            "{} {}",
            oracle.successes,
            uniform_pool.successes
        );
        assert_eq!(oracle.integrity.routing_outside_admissible, 0);
        let bon = run_arm(
            Arm::BestOfN,
            WorldId::Hetero,
            TEST_SEED + 29,
            &trained,
            &tasks,
        );
        assert!(
            bon.successes > bon.first_pass + 10,
            "{} {}",
            bon.successes,
            bon.first_pass
        );
        let full = run_arm(Arm::Full, WorldId::Hetero, TEST_SEED + 29, &trained, &tasks);
        let blind = run_arm(Arm::NoL3, WorldId::Hetero, TEST_SEED + 29, &trained, &tasks);
        assert_eq!(
            full.gap_first, blind.gap_first,
            "the first candidate is the same with and without L3"
        );
        assert_eq!(
            full.first_planted, bon.first_planted,
            "the first draft is the same in arms with different scorers and registries"
        );
        assert_eq!(full.first_planted, oracle.first_planted);
    }

    #[test]
    fn every_counter_of_the_harness_is_a_hard_counter_of_the_plan_and_some_mutation_expects_it() {
        const PLAN: &str =
            include_str!("../../../../../experiments/feedback/F007-vcw-loop/tests/mutations.toml");
        let plan: toml::Table = PLAN.parse().expect("mutations.toml parses");
        let hard: Vec<&str> = plan["hard_counters"]
            .as_array()
            .expect("hard_counters")
            .iter()
            .map(|c| c.as_str().expect("string"))
            .collect();
        let expected: Vec<&str> = plan["mutation"]
            .as_array()
            .expect("mutations")
            .iter()
            .flat_map(|m| m["expect"].as_array().expect("expect").iter())
            .map(|c| c.as_str().expect("string"))
            .collect();
        let all = metrics::FALSIFIER_COUNTERS
            .iter()
            .chain(metrics::CONTROL_COUNTERS.iter())
            .chain(metrics::COVERAGE_COUNTERS.iter())
            .chain(std::iter::once(&"coverage_unmet"));
        for counter in all {
            assert!(
                hard.contains(counter),
                "{counter} is not a hard counter of the plan"
            );
            assert!(expected.contains(counter), "no mutation expects {counter}");
        }
        for counter in &expected {
            assert!(hard.contains(counter), "{counter}");
        }
    }
}
