//! Experiment L001-v2, revocation under crash, power-loss and leadership schedules, as a
//! harness: `ptr-bench revocation-crash <seed>` runs the experiment's four arms once for one
//! seed and prints exactly one JSON object on stdout. (It supersedes L001, whose probes checked
//! one subject and one stale action.)
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `revocation-crash <cases> <seed>`) is no evidence. The code under test is `ptr_ledger`
//! (`FileLedger`, `AcknowledgedLedger` with its `ProtectedAnchor`, `RaftEngineLedger`,
//! `RaftNode`) and `ptr_runtime::PtrRuntime` (replay, `authorize_action`, `generation_validity`),
//! driven only through their public APIs, with the failpoints of `ptr-ledger`'s test features.
//! An independent oracle (`model.rs`, about sixty lines, no code shared with the product) says
//! which (subject, generation) pairs may be acted on after any prefix of a history.
//!
//! The object holds, in this order: the identity of the run, every metric the manifest names (a
//! JSON integer), `hard_failures`, the timings (never read by a rule), the procedure, the
//! controls, the environment, the notes, the counters that belong to no metric (`internal`),
//! and the two judgements (`coverage` and `falsifiers`). The exit status is 1 when a coverage
//! prerequisite is unmet or a falsifier is tripped, and 2 when the harness itself cannot run: a
//! synthesized image outside the honest model, a baseline lie that fails to fool the naive
//! recovery, a durability probe that cannot see a dirty page, a child process that does not die
//! the way the protocol says, or a plan that is not the same twice. Every panic and every hang
//! of the product is contained and counted (`product_panics`, `product_hangs`) and falsifies.

pub mod abort_arm;
pub mod check;
pub mod child;
pub mod engine_arm;
pub mod file_arm;
pub mod image;
pub mod metrics;
pub mod model;
pub mod params;
pub mod probe;
pub mod raft_arm;
pub mod router;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::rng::Rng;
use file_arm::{Kind, RunDir};
use model::{generate, Workload};
use params::{Scale, Shape};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
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

/// Whether the child processes of the abort arms can run: the executable must be `ptr-bench`
/// itself, which knows the `revocation-crash-child` subcommand (a test binary does not).
fn children_available() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .is_some_and(|stem| stem == "ptr-bench")
}

static RUNS: AtomicU64 = AtomicU64::new(0);

/// A digest of the workloads a seed and shape generate, for the check that the plan is the same
/// twice.
fn plan_digest(seed: u64, shape: Shape) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |text: String| {
        for byte in text.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (salt, count) in [
        (SALT_FILE, shape.file),
        (SALT_ACK, shape.ack),
        (SALT_ENGINE, shape.engine),
    ] {
        let mut rng = Rng::new(seed ^ salt);
        for _ in 0..count {
            let workload_seed = rng.next_u64();
            let workload = generate(
                &mut Rng::new(workload_seed),
                params::EVENTS_MIN,
                params::EVENTS_MAX,
            );
            feed(format!("{workload_seed}{:?}", workload.events));
        }
    }
    hash
}

const SALT_FILE: u64 = 0xA1A1_A1A1;
const SALT_ACK: u64 = 0xB2B2_B2B2;
const SALT_ENGINE: u64 = 0xC3C3_C3C3;
const SALT_RAFT: u64 = 0xD4D4_D4D4;

/// What the workloads of a run looked like.
#[derive(Default)]
struct Shapes {
    events_min: Option<usize>,
    subjects_min: Option<usize>,
    recommits: i64,
}

impl Shapes {
    fn note(&mut self, tally: &mut Tally, workload: &Workload) {
        let events = workload.events.len();
        let subjects = workload.subjects;
        self.events_min = Some(self.events_min.map_or(events, |m| m.min(events)));
        self.subjects_min = Some(self.subjects_min.map_or(subjects, |m| m.min(subjects)));
        if workload.has_revoke_then_recommit() {
            self.recommits += 1;
            tally.add("revoke_then_recommit_histories", 1);
        }
    }
}

/// Arms A and B for `count` workloads of `kind`.
#[allow(clippy::too_many_arguments)]
fn files_arm(
    tally: &mut Tally,
    runs: &mut RunDir,
    shapes: &mut Shapes,
    kind: Kind,
    count: usize,
    rng: &mut Rng,
    probing: bool,
    children: bool,
) -> Result<(), String> {
    for _ in 0..count {
        let workload_seed = rng.next_u64();
        let workload = Arc::new(generate(
            &mut Rng::new(workload_seed),
            params::EVENTS_MIN,
            params::EVENTS_MAX,
        ));
        shapes.note(tally, &workload);
        tally.add(&format!("workloads_{}", kind.arm()), 1);
        if check::failpoints_blocked(tally) || check::hang_limit_reached(tally) {
            break;
        }
        let observed = runs.fresh();
        // The run without a fault is under a watchdog like every recovery: a product call that
        // panics or hangs on the reference path is the product's defect, counted, and the
        // workload ends there.
        let (reference_workload, reference_dir) = ((*workload).clone(), observed.clone());
        let outcome = check::guarded(
            std::time::Duration::from_millis(params::HANG_WATCHDOG_MS),
            move || file_arm::observe(kind, &reference_workload, &reference_dir, probing),
        );
        let Some(obs) = check::counted_append(tally, kind.arm(), outcome) else {
            let _ = std::fs::remove_dir_all(&observed);
            continue;
        };
        absorb_file_observation(tally, kind, &obs)?;
        file_arm::reference(tally, kind, &workload, &observed, &obs);
        let _ = std::fs::remove_dir_all(&observed);
        let mut image_rng = rng.fork(0x11);
        for prepared in file_arm::images(kind, &workload, &obs, &mut image_rng, tally)? {
            file_arm::run_prepared(tally, runs, kind, &workload, &prepared);
        }
        let mut crash_rng = rng.fork(0x22);
        abort_arm::unwind_runs(tally, runs, kind, &workload, &mut crash_rng)?;
        if children {
            abort_arm::child_runs(tally, runs, kind, &workload, workload_seed, &mut crash_rng)?;
        }
    }
    Ok(())
}

fn absorb_file_observation(
    tally: &mut Tally,
    kind: Kind,
    obs: &file_arm::Observation,
) -> Result<(), String> {
    for (point, hits) in &obs.hits {
        tally.add(
            &format!("failpoint_hits.{}", params::hit_name(point)),
            *hits,
        );
    }
    tally.add("probe_clean_checked", obs.clean_checked);
    check::add(tally, kind.arm(), "unsynced_acks", obs.unsynced_acks);
    // Creating the log synchronizes its directory, and so does each publication of an anchor:
    // the set once more at its creation and once per acknowledged append. A shortfall is a
    // directory the product answered for without synchronizing.
    let appended = obs.boundaries.len().saturating_sub(1) as i64;
    let expected_dir_syncs = match kind {
        Kind::File => 1,
        Kind::Ack => 2 + appended,
    };
    tally.add(
        &format!(
            "failpoint_hits.{}",
            params::hit_name(params::SYNC_PARENT_POINT)
        ),
        obs.sync_parent_hits,
    );
    check::add(
        tally,
        kind.arm(),
        "unsynced_acks",
        (expected_dir_syncs - obs.sync_parent_hits).max(0),
    );
    check::add(
        tally,
        kind.arm(),
        "reference_mismatches",
        obs.append_failures,
    );
    tally.add("append_failures", obs.append_failures);
    Ok(())
}

fn engine_arm(
    tally: &mut Tally,
    runs: &mut RunDir,
    shapes: &mut Shapes,
    count: usize,
    rng: &mut Rng,
    probing: bool,
    children: bool,
) -> Result<(), String> {
    for _ in 0..count {
        let workload_seed = rng.next_u64();
        let workload = Arc::new(generate(
            &mut Rng::new(workload_seed),
            params::EVENTS_MIN,
            params::EVENTS_MAX,
        ));
        shapes.note(tally, &workload);
        tally.add("workloads_engine", 1);
        let mut stream = rng.fork(0x33);
        let Ok(run) = catch_unwind(AssertUnwindSafe(|| {
            engine_arm::run_workload(
                tally,
                runs,
                &workload,
                workload_seed,
                &mut stream,
                probing,
                children,
            )
        })) else {
            check::bump(tally, "engine", "product_panics");
            continue;
        };
        let obs = run?;
        for (point, hits) in &obs.hits {
            tally.add(
                &format!("failpoint_hits.{}", params::hit_name(point)),
                *hits,
            );
        }
        tally.add("probe_clean_checked", obs.clean_checked);
        check::add(tally, "engine", "unsynced_acks", obs.unsynced_acks);
        check::add(
            tally,
            "engine",
            "reference_mismatches",
            obs.append_failures + obs.reopen_mismatches,
        );
        tally.add("append_failures", obs.append_failures);
    }
    Ok(())
}

/// The controls: the fsync lie must fool the naive recovery on the arms whose anchored product
/// is judged, and there must be lies to fool it with. A run where it does not cannot tell a
/// recovery that notices lost records from one that does not.
fn check_controls(tally: &Tally) -> Result<(), String> {
    for arm in ["file", "ack"] {
        let lies = tally.get(&format!("arms.{arm}.lie_images"));
        let fooled = tally.get(&format!("arms.{arm}.b1_lie_images_accepted_stale"));
        if lies == 0 || fooled == 0 {
            return Err(format!(
                "the fsync-lie control does not trip in the {arm} arm: {lies} lie images, {fooled} accepted as healthy by the naive recovery with a stale revocation; the experiment could not tell a recovery that notices lost records from one that does not"
            ));
        }
    }
    Ok(())
}

/// A run whose durability probe cannot tell a written file from a synced one cannot observe
/// what its clean readings are meant to show: the harness cannot run (status 2).
fn require_probe(calibration: probe::Calibration) -> Result<(), String> {
    if calibration.available() {
        Ok(())
    } else {
        Err(format!(
            "the durability probe cannot tell a written file from a synced one on the scratch directory (dirty read {}, clean read {}): it needs a filesystem that writes back to a disk, and a clean reading would prove nothing",
            calibration.sees_dirty, calibration.sees_clean
        ))
    }
}

/// The filesystem the run's scratch directory is on, from the mount table.
fn filesystem_of(path: &std::path::Path) -> String {
    let Ok(real) = path.canonicalize() else {
        return "unknown".into();
    };
    let Ok(table) = std::fs::read_to_string("/proc/self/mountinfo") else {
        return "unknown".into();
    };
    table
        .lines()
        .filter_map(|line| {
            let (before, after) = line.split_once(" - ")?;
            let mount = before.split(' ').nth(4)?;
            let kind = after.split(' ').next()?;
            real.starts_with(mount)
                .then(|| (mount.len(), kind.to_string()))
        })
        .max_by_key(|(length, _)| *length)
        .map(|(_, kind)| kind)
        .unwrap_or_else(|| "unknown".into())
}

/// Run every arm once for `seed` at `scale` and assemble the one JSON object. A run the harness
/// cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    static QUIET: std::sync::Once = std::sync::Once::new();
    QUIET.call_once(check::quiet_injected_panics);
    // The oracle against its table first: a wrong oracle is the harness's defect (status 2) and
    // must not be read as the product's.
    model::self_check()?;
    let started = Instant::now();
    let shape = scale.shape();
    if plan_digest(seed, shape) != plan_digest(seed, shape) {
        return Err("the plan of a seed differs between two derivations".into());
    }
    let children = children_available();
    if scale == Scale::Pinned && !children {
        return Err("the pinned run needs the ptr-bench executable for its child processes".into());
    }
    let label = RUNS.fetch_add(1, Ordering::SeqCst);
    let mut runs = RunDir::new(seed, &format!("run{label}"));
    let filesystem = filesystem_of(runs.path());
    let calibration = probe::calibrate(runs.path());
    require_probe(calibration)?;
    let probing = calibration.available();
    let mut tally = Tally::new(metrics::LIST);
    tally.set("probe_calibrated", i64::from(probing));
    tally.set("probe_dirty_seen", i64::from(calibration.sees_dirty));
    let mut shapes = Shapes::default();

    let mut rng = Rng::new(seed ^ SALT_FILE);
    files_arm(
        &mut tally,
        &mut runs,
        &mut shapes,
        Kind::File,
        shape.file,
        &mut rng,
        probing,
        children,
    )?;
    let mut rng = Rng::new(seed ^ SALT_ACK);
    files_arm(
        &mut tally,
        &mut runs,
        &mut shapes,
        Kind::Ack,
        shape.ack,
        &mut rng,
        probing,
        children,
    )?;
    let mut rng = Rng::new(seed ^ SALT_ENGINE);
    engine_arm(
        &mut tally,
        &mut runs,
        &mut shapes,
        shape.engine,
        &mut rng,
        probing,
        children,
    )?;
    let mut rng = Rng::new(seed ^ SALT_RAFT);
    raft_arm::run(&mut tally, &mut runs, &mut rng, shape.raft_per_kind)?;
    // A run that already tripped a falsifier (a hang that cut it short, an append of a valid
    // workload that failed so that no revocation was ever acknowledged) may never have built its
    // lie images: it is falsified by what it counted, and its controls are not a harness failure.
    if metrics::hard_failures(&tally) == 0 {
        check_controls(&tally)?;
    }

    // The manifest's names from the counters of the arms.
    for (metric, source) in [
        ("recoveries_file", "arms.file.recoveries"),
        ("recoveries_ack", "arms.ack.recoveries"),
        ("recoveries_engine", "arms.engine.recoveries"),
        ("members_replayed", "arms.raft.members_replayed"),
        ("child_aborts_file", "arms.file.child_aborts"),
        ("child_aborts_ack", "arms.ack.child_aborts"),
        ("child_aborts_engine", "arms.engine.child_aborts"),
        ("scenarios_raft", "arms.raft.scenarios"),
        ("scenarios_checked", "arms.raft.scenarios_checked"),
        ("raft_scenario_errors", "arms.raft.raft_scenario_errors"),
    ] {
        let value = tally.get(source);
        tally.set(metric, value);
    }
    // Of all recoveries of a damaged image (a tail of zeros included), the share the product
    // refused; above the pinned bound it is a falsifier.
    let damaged = tally.get("recoveries_damaged");
    let refused = tally.get("refusals_damaged");
    let share = if damaged == 0 {
        0
    } else {
        1000 * refused / damaged
    };
    tally.set("refused_share_damaged_permille", share);
    if share > params::MAX_REFUSED_SHARE_DAMAGED_PERMILLE {
        tally.set("refusal_share_excess", 1);
    }
    tally.set("events_min", shapes.events_min.unwrap_or(0) as i64);
    tally.set("subjects_min", shapes.subjects_min.unwrap_or(0) as i64);
    tally.set("revoke_then_recommit_histories", shapes.recommits);
    for name in metrics::METRIC_NAMES {
        if name != "failpoint_hits" {
            let value = tally.get(name);
            tally.set(name, value);
        }
    }
    tally.seed_group("failpoint_hits", &metrics::HIT_POINTS);
    let hard = metrics::hard_failures(&tally);

    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str(params::EXPERIMENT_ID)),
        (
            "status_note".to_string(),
            note("one seed of the L001-v2 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push(("hard_failures".to_string(), Json::Int(hard)));
    pairs.push((
        "timing_ns".to_string(),
        Json::obj([("elapsed_ns", Json::Int(started.elapsed().as_nanos() as i64))]),
    ));
    pairs.push((
        "procedure".to_string(),
        Json::obj([
            ("seed_is_declared", Json::Bool(params::SEEDS.contains(&seed))),
            ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
            ("children_ran", Json::Bool(children)),
            ("durability_probe_available", Json::Bool(probing)),
            (
                "workloads",
                Json::obj([
                    ("file", Json::Int(shape.file as i64)),
                    ("ack", Json::Int(shape.ack as i64)),
                    ("engine", Json::Int(shape.engine as i64)),
                    ("raft_scenarios_per_kind", Json::Int(shape.raft_per_kind as i64)),
                ]),
            ),
            (
                "failure_classes",
                note("process abort keeps the page cache, so every written byte survives: an in-process panic at a failpoint (the unwinding drops the ledger) and a child process that calls abort() at it, both recovered from the real files they leave; power loss keeps the synced prefix of each file and a seeded subset of what was written after it (none, all, a prefix, half, 512-byte holes), an fsync-honest crash-image model and not a real power cut; leadership schedules run three in-process Raft members on a deterministic router (isolation, drop, duplicate, reorder, restart)"),
            ),
            (
                "expectation",
                note("recovered history = acknowledged prefix + at most the record in flight, exact events and indices 1..n; an unsafe log/anchor pair is refused (fail closed); the classification of the pair is compared with the harness's own table; the runtime replayed over the recovered events must agree with the oracle for every (subject, generation) of the universe"),
            ),
            (
                "baselines",
                note("B0 is the same workload without a fault, recovered through the same path and held to the same checks; B1 is naive recovery (the log alone, tail trimmed, whatever is left trusted) run on the fsync-lie images, where it must accept a stale revocation or the run is invalid"),
            ),
        ]),
    ));
    let lie = |arm: &str| {
        Json::obj([
            (
                "lie_images",
                Json::Int(tally.get(&format!("arms.{arm}.lie_images"))),
            ),
            (
                "naive_accepted_stale",
                Json::Int(tally.get(&format!("arms.{arm}.b1_lie_images_accepted_stale"))),
            ),
            (
                "anchored_refused",
                Json::Int(tally.get(&format!("arms.{arm}.refusals_lost_ack"))),
            ),
            ("required_to_trip", Json::Bool(true)),
        ])
    };
    pairs.push((
        "controls".to_string(),
        Json::obj([
            ("file_lie", lie("file")),
            ("ack_lie", lie("ack")),
            (
                "engine_lie",
                Json::obj([
                    ("lie_images", Json::Int(tally.get("arms.engine.lie_images"))),
                    (
                        "undetected_rollback",
                        Json::Int(tally.get("undetected_rollback")),
                    ),
                    ("refused", Json::Int(tally.get("engine_lie_refused"))),
                    ("required_to_trip", Json::Bool(false)),
                ]),
            ),
        ]),
    ));
    pairs.push((
        "environment".to_string(),
        Json::obj([
            ("filesystem", Json::str(filesystem)),
            (
                "power_loss_model",
                Json::str("fsync-honest crash images; no real power cut"),
            ),
        ]),
    ));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            (
                "raft_engine_anchor",
                note("RaftEngineLedger has no hash chain and no protected anchor: after a lie that drops acknowledged bytes it opens without error on fewer records. undetected_rollback counts those and engine_lie_stale_accepts the stale accepts that follow; both are descriptive, no rule reads them, and no product work is attempted on it (owner decision of 2026-10-09)"),
            ),
            (
                "durability_probe",
                note("the probe asks the kernel (posix_fadvise DONTNEED, then mincore) whether a file still holds pages it has not written to its disk. probe_calibrated is 1 when, on the scratch directory, a written file reads dirty and a synced one clean, and probe_dirty_seen is 1 when the dirty reading was seen; inside a run only clean readings are taken (asking about a dirty file starts its writeback), where the product claims a record durable, before it writes the next one, before it renames the anchor's scratch file and before it moves the anchor; probe_clean_checked counts them and unsynced_acks the readings that found a page. It sees file data pages only, not directory entries, so a missing directory fsync is outside what it detects"),
            ),
            (
                "refusals",
                note("refusals_lost_ack are lie images the anchored product refused (required); refusals_corrupt_tail are complete-length frames with a hole that the product refused (allowed: fail closed); any other refusal is unjustified_refusals"),
            ),
            (
                "reference_mismatches",
                note("a baseline run (no fault) whose recovery trips any other falsifier, or whose append fails, or whose engine does not reopen on the records it acknowledged; counted as the product's defect and falsifies, it is not exit 2"),
            ),
        ]),
    ));
    let mut internal = match tally.internal() {
        Json::Obj(pairs) => pairs,
        other => vec![("counters".to_string(), other)],
    };
    internal.push(("unsupported".to_string(), tally.unsupported_json()));
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
        [cases, seed] => parse(cases, "cases").and_then(|cases| {
            if cases == 0 {
                return Err("cases must be at least 1".to_string());
            }
            parse(seed, "seed").map(|seed| (seed, Scale::Custom(cases as usize)))
        }),
        _ => Err("usage: revocation-crash <seed> | revocation-crash <cases> <seed>".to_string()),
    }
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after
/// the subcommand: `<seed>` (the pinned scale) or `<cases> <seed>` (the mutation checker's
/// form: that many workloads per arm, no evidence). Exits 1 when the run is not clean and 2
/// when the harness cannot run.
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
        include_str!("../../../../../experiments/lifecycle/L001-v2-revocation-crash/criteria.toml");
    const MUTATIONS: &str = include_str!(
        "../../../../../experiments/lifecycle/L001-v2-revocation-crash/tests/mutations.toml"
    );
    const PREREGISTRATION: &str = include_str!(
        "../../../../../experiments/lifecycle/L001-v2-revocation-crash/PREREGISTRATION.md"
    );

    /// The shared reduced run (one workload per arm, no child processes: the test binary is
    /// not `ptr-bench`).
    fn reduced_report() -> &'static Report {
        static REPORT: std::sync::OnceLock<Report> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| execute(1001, Scale::Reduced).expect("a reduced run"))
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
        assert_eq!(table["experiment"].as_str(), Some(params::EXPERIMENT_ID));
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
                // One below the threshold, or 0 where the reduced run is already there.
                ">=" => {
                    let below = tokens[2].parse::<i64>().unwrap() - 1;
                    if below == current {
                        below - 1
                    } else {
                        below
                    }
                }
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
        assert_eq!(metrics::falsifiers().len(), 14);
    }

    #[test]
    fn the_entrypoint_is_the_subcommand_and_the_status_is_planned_or_later() {
        assert_eq!(params::BENCHMARK, "revocation-crash");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert_eq!(entrypoint, params::ENTRYPOINT);
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
        assert!(matches!(
            manifest["status"].as_str(),
            Some("planned" | "prepared" | "completed" | "failed")
        ));
        assert_eq!(manifest["id"].as_str(), Some(params::EXPERIMENT_ID));
    }

    #[test]
    fn the_mutation_plan_names_the_harnesss_hard_counters_and_each_is_expected_by_one() {
        let plan: toml::Table = MUTATIONS.parse().expect("mutations.toml is TOML");
        let listed: Vec<String> = plan["hard_counters"]
            .as_array()
            .expect("hard_counters")
            .iter()
            .map(|name| name.as_str().expect("a name").to_string())
            .collect();
        assert_eq!(listed, metrics::HARD_COUNTERS);
        assert_eq!(plan["subcommand"].as_str(), Some(params::BENCHMARK));
        assert_eq!(plan["features"].as_str(), Some("ledger-experiments"));
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
            .starts_with("{\"benchmark\":\"revocation-crash\",\"experiment\":\"L001-v2\""));
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        assert_eq!(int(&report.json, "seed"), 1001);
        for name in metrics::METRIC_NAMES {
            if name != "failpoint_hits" {
                assert!(
                    matches!(emit::resolve(&report.json, name), Some(Json::Int(_))),
                    "{name}"
                );
            }
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

    /// The real product trips no falsifier on the reduced run, and the reduced run really
    /// exercises what it names: failpoints fired, bytes lost, the lie fooled the naive
    /// recovery and not the anchored one, Raft elected twice and dropped messages.
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
        for point in metrics::HIT_POINTS {
            assert!(
                int(json, &format!("failpoint_hits.{point}")) >= 1,
                "{point}"
            );
        }
        assert!(int(json, "images_lost_bytes") >= 10);
        assert!(int(json, "rename_image_pairs") >= 1);
        assert!(int(json, "lie_images") >= 4);
        assert!(int(json, "unwind_panics_observed") >= 8);
        assert!(int(json, "leader_changes") >= 2);
        assert!(int(json, "minority_proposals") >= 1);
        assert!(int(json, "deposed_leader_restarts") >= 1);
        assert!(int(json, "messages_dropped") >= 1);
        assert!(int(json, "probe_calibrated") == 1 || int(json, "probe_clean_checked") == 0);
        assert!(int(json, "refusals_lost_ack") >= 1);
        assert!(
            int(json, "b1_stale_accepts") >= 1,
            "the naive recovery accepts a stale revocation"
        );
        assert!(
            int(json, "follow_on_appends") >= 1,
            "recovery leaves a ledger that goes on"
        );
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
        let again = strip(&execute(1001, Scale::Reduced).expect("a second run").json);
        assert_eq!(first, again);
        let other = strip(&execute(1002, Scale::Reduced).expect("another seed").json);
        assert_ne!(first, other);
    }

    /// The controls named in the output are the controls the run checks.
    #[test]
    fn the_lie_controls_trip_in_the_two_anchored_arms_and_are_descriptive_in_the_engine() {
        let json = &reduced_report().json;
        for arm in ["file_lie", "ack_lie"] {
            assert!(
                int(json, &format!("controls.{arm}.lie_images")) >= 1,
                "{arm}"
            );
            assert!(
                int(json, &format!("controls.{arm}.naive_accepted_stale")) >= 1,
                "{arm}"
            );
            assert_eq!(
                int(json, &format!("controls.{arm}.anchored_refused")),
                int(json, &format!("controls.{arm}.lie_images")),
                "{arm}: the anchored product refuses every lie"
            );
        }
        assert!(matches!(
            emit::resolve(json, "controls.engine_lie.required_to_trip"),
            Some(Json::Bool(false))
        ));
        assert!(int(json, "controls.engine_lie.undetected_rollback") >= 1);
    }

    #[test]
    fn a_control_that_does_not_trip_is_a_harness_failure() {
        let mut tally = Tally::new(metrics::LIST);
        assert!(check_controls(&tally).is_err(), "no lies at all");
        for arm in ["file", "ack"] {
            tally.add(&format!("arms.{arm}.lie_images"), 3);
        }
        assert!(
            check_controls(&tally).is_err(),
            "lies the naive recovery rejects too"
        );
        for arm in ["file", "ack"] {
            tally.add(&format!("arms.{arm}.b1_lie_images_accepted_stale"), 1);
        }
        assert_eq!(check_controls(&tally), Ok(()));
    }

    #[test]
    fn a_probe_that_cannot_tell_dirty_from_clean_ends_the_run_with_status_2() {
        let blind = probe::Calibration {
            sees_dirty: false,
            sees_clean: true,
        };
        assert!(require_probe(blind).is_err());
        assert!(require_probe(probe::Calibration {
            sees_dirty: true,
            sees_clean: true
        })
        .is_ok());
        // The directory of this run's own calibration is the one the tests run on.
        let dir = std::env::temp_dir();
        let _ = require_probe(probe::calibrate(&dir));
    }

    #[test]
    fn the_plan_of_a_seed_is_the_same_twice_and_differs_between_seeds() {
        let shape = Scale::Pinned.shape();
        assert_eq!(plan_digest(1001, shape), plan_digest(1001, shape));
        assert_ne!(plan_digest(1001, shape), plan_digest(1002, shape));
    }

    #[test]
    fn timings_are_in_their_own_field_and_no_rule_reads_them() {
        let json = &reduced_report().json;
        assert!(int(json, "timing_ns.elapsed_ns") > 0);
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(!rule.contains("timing") && !rule.contains("_ns"), "{rule}");
        }
    }

    #[test]
    fn the_filesystem_is_recorded_from_the_mount_table() {
        let json = &reduced_report().json;
        assert!(matches!(
            emit::resolve(json, "environment.filesystem"),
            Some(Json::Str(name)) if !name.is_empty()
        ));
        assert_ne!(
            filesystem_of(std::path::Path::new("/definitely/not/there")),
            "ext4"
        );
    }

    #[test]
    fn the_arguments_are_a_seed_or_cases_and_a_seed_and_nothing_else() {
        let args = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_args(&args(&["17"])), Ok((17, Scale::Pinned)));
        assert_eq!(parse_args(&args(&["4", "17"])), Ok((17, Scale::Custom(4))));
        for bad in [
            args(&[]),
            args(&["x"]),
            args(&["-1"]),
            args(&["0", "17"]),
            args(&["4", "x"]),
            args(&["1", "2", "3"]),
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_pinned_run_refuses_to_start_where_it_cannot_run_its_children() {
        // The test binary is not `ptr-bench`, so the pinned scale (which needs the children)
        // is a harness error and not a silently smaller run.
        assert!(!children_available());
        assert!(execute(1001, Scale::Pinned).is_err());
    }
}
