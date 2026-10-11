//! Experiment L002, recovery of a clustered ledger after crashes, partitions and restarts, as a
//! harness: `ptr-bench raft-recovery <seed>` runs the experiment's arms once for one seed and
//! prints exactly one JSON object on stdout. `raft-recovery --preflight` checks, without
//! recording anything, that the harness can run on this machine at all.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`; a run at another scale (the unit tests, or the mutation checker's
//! `raft-recovery <cases> <seed>`) is no evidence. The code under test is `ptr_ledger`
//! (`AcknowledgedLedger::compact` with its protected anchor, `FileRaftStorage` and `RaftNode`)
//! and `ptr_runtime::PtrRuntime` (replay, the compacted snapshot, `authorize_action`,
//! `generation_validity`), driven only through their public APIs, with the failpoints of
//! `ptr-ledger`'s test features. The oracle and the workload generator are L001-v2's
//! (`l001::model`, about sixty lines, no code shared with the product).
//!
//! Arm K crashes the compaction cutover of an anchored log (`cutover`); arms S and G crash the
//! durable-state transitions of three Raft members and hold the group to a single-node
//! reference (`group`). The exit status is 1 when a coverage prerequisite is unmet or a
//! falsifier is tripped, and 2 when the harness itself cannot run: the oracle failing its table,
//! a synthesized image outside the honest model, a control that does not trip, a durability
//! probe that cannot see a dirty page. Every panic and every hang of the product is contained
//! and counted (`product_panics`, `product_hangs`) and falsifies.

pub mod cutover;
pub mod group;
pub mod metrics;
pub mod params;
pub mod view;

use super::l001::check::{self, bump};
use super::l001::model::{self, generate, Workload};
use super::l001::probe;
use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::rng::Rng;
use cutover::RunDir;
use group::{Inject, Run};
use params::{Scale, Shape};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
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

static RUNS: AtomicU64 = AtomicU64::new(0);

const SALT_CUTOVER: u64 = 0xE5E5_E5E5;
const SALT_GROUP: u64 = 0xF6F6_F6F6;

fn watchdog() -> Duration {
    Duration::from_millis(params::HANG_WATCHDOG_MS)
}

/// A digest of the workloads a seed and shape generate, for the check that the plan is the same
/// twice.
fn plan_digest(seed: u64, shape: Shape) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |text: String| {
        for byte in text.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let mut rng = Rng::new(seed ^ SALT_CUTOVER);
    for _ in 0..shape.cutover_workloads {
        let workload = generate(&mut rng, params::EVENTS_MIN, params::EVENTS_MAX);
        feed(format!("{:?}", workload.events));
        feed(format!("{:?}", cutover::cuts_for(&workload, &mut rng)));
    }
    let mut rng = Rng::new(seed ^ SALT_GROUP);
    for kind in params::SCENARIO_KINDS {
        let spec = group::spec_for(kind, &mut rng);
        feed(format!(
            "{kind}{:?}{}{}",
            spec.workload.events, spec.a, spec.b
        ));
    }
    hash
}

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

/// Arm K over `shape.cutover_workloads` workloads.
fn cutover_arm(
    tally: &mut Tally,
    runs: &mut RunDir,
    shapes: &mut Shapes,
    shape: Shape,
    rng: &mut Rng,
    probing: bool,
) -> Result<(), String> {
    for _ in 0..shape.cutover_workloads {
        let workload = Arc::new(generate(rng, params::EVENTS_MIN, params::EVENTS_MAX));
        shapes.note(tally, &workload);
        tally.add("workloads_cutover", 1);
        for cut in cutover::cuts_for(&workload, rng) {
            if check::failpoints_blocked(tally) || tally.get("product_hangs") >= params::HANG_LIMIT
            {
                return Ok(());
            }
            let dir = runs.fresh();
            let (call_workload, call_dir) = (Arc::clone(&workload), dir.clone());
            let outcome = check::guarded(watchdog(), move || {
                cutover::observe(&call_workload, cut, &call_dir, probing)
            });
            let Some(obs) = check::counted_append(tally, cutover::ARM, outcome) else {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            };
            let _ = std::fs::remove_dir_all(&dir);
            let Some(case) = cutover::absorb_and_reference(tally, runs, &obs) else {
                continue;
            };
            cutover::unwind_runs(tally, runs, &case);
            cutover::above_floor_runs(tally, runs, &case, rng);
            let mut image_rng = rng.fork(0x51);
            cutover::images(tally, runs, &case, &obs, &mut image_rng)?;
        }
    }
    Ok(())
}

/// One schedule under a watchdog; its counters are added to the run's.
fn run_scenario(
    tally: &mut Tally,
    runs: &mut RunDir,
    spec: &group::Spec,
    inject: Inject,
    probing: bool,
    capture: bool,
) -> Option<Run> {
    let dir = runs.fresh();
    let (call_dir, call_spec) = (dir.clone(), spec.clone());
    let outcome = check::guarded(watchdog(), move || {
        group::run_life(call_dir, call_spec, inject, probing, capture)
    });
    let _ = std::fs::remove_dir_all(&dir);
    let run = check::counted_append(tally, group::ARM, outcome)?;
    // The longest wait of a run is a maximum, not a sum.
    let (maxima, sums): (Vec<_>, Vec<_>) = run
        .counters
        .iter()
        .cloned()
        .partition(|(name, _)| name == "rounds_max_run");
    tally.absorb(&sums);
    for (_, rounds) in maxima {
        let longest = tally.get("rounds_to_converge_max").max(rounds);
        tally.set("rounds_to_converge_max", longest);
    }
    Some(run)
}

fn absorb_hits(tally: &mut Tally, run: &Run) {
    for (point, hits) in &run.hits {
        tally.add(
            &format!("failpoint_hits.{}", params::hit_name(point)),
            *hits as i64,
        );
    }
}

/// Arms S and G: every kind of scenario of the plan.
fn group_arm(
    tally: &mut Tally,
    runs: &mut RunDir,
    shapes: &mut Shapes,
    shape: Shape,
    rng: &mut Rng,
    probing: bool,
) -> Result<(), String> {
    let stop = |tally: &Tally| {
        check::failpoints_blocked(tally) || tally.get("product_hangs") >= params::HANG_LIMIT
    };
    // A schedule that crashes or fails a write: the observing run first (baseline B0, and the
    // count of hits), then the same schedule with the fault.
    let faulted = |tally: &mut Tally,
                   runs: &mut RunDir,
                   shapes: &mut Shapes,
                   rng: &mut Rng,
                   point: &'static str,
                   io: bool| {
        let spec = group::spec_for("crash", rng);
        shapes.note(tally, &spec.workload);
        tally.add("scenarios_group", 1);
        let before = metrics::hard_sum(tally);
        let Some(observed) = run_scenario(tally, runs, &spec, Inject::None, probing, false) else {
            return;
        };
        absorb_hits(tally, &observed);
        if metrics::hard_sum(tally) > before {
            bump(tally, group::ARM, "reference_mismatches");
        }
        let Some(target) = group::target_for(&observed.hits, point, rng) else {
            check::add(tally, group::ARM, "reference_mismatches", 1);
            return;
        };
        let inject = if io {
            Inject::IoError { point, target }
        } else {
            Inject::Crash { point, target }
        };
        let checked_before = tally.get("arms.member.members_replayed");
        let injected_before = tally.get("arms.member.injected_io_errors");
        let Some(run) = run_scenario(tally, runs, &spec, inject, probing, false) else {
            return;
        };
        let happened = if io {
            tally.get("arms.member.injected_io_errors") > injected_before
        } else {
            run.fired
        };
        if !happened {
            bump(tally, group::ARM, "reference_mismatches");
        }
        if tally.get("arms.member.members_replayed") > checked_before {
            tally.add("scenarios_checked", 1);
        }
    };
    for point in params::MEMBER_POINTS {
        for _ in 0..shape.crash_per_point {
            if stop(tally) {
                return Ok(());
            }
            faulted(tally, runs, shapes, rng, point, false);
        }
    }
    for point in params::ERROR_POINTS {
        for _ in 0..shape.ioerr_per_point {
            if stop(tally) {
                return Ok(());
            }
            faulted(tally, runs, shapes, rng, point, true);
        }
    }
    for kind in ["heal", "tamper"] {
        let count = if kind == "heal" {
            shape.heal
        } else {
            shape.tamper
        };
        for _ in 0..count {
            if stop(tally) {
                return Ok(());
            }
            let spec = group::spec_for(kind, rng);
            shapes.note(tally, &spec.workload);
            tally.add("scenarios_group", 1);
            let checked_before = tally.get("arms.member.members_replayed");
            let Some(run) = run_scenario(tally, runs, &spec, Inject::None, probing, false) else {
                continue;
            };
            absorb_hits(tally, &run);
            if tally.get("arms.member.members_replayed") > checked_before {
                tally.add("scenarios_checked", 1);
            }
        }
    }
    for _ in 0..shape.images {
        if stop(tally) {
            return Ok(());
        }
        let spec = group::spec_for("images", rng);
        shapes.note(tally, &spec.workload);
        tally.add("scenarios_group", 1);
        let checked_before = tally.get("arms.member.members_replayed");
        let Some(run) = run_scenario(tally, runs, &spec, Inject::None, probing, true) else {
            continue;
        };
        absorb_hits(tally, &run);
        let mut image_rng = rng.fork(0x77);
        group::image_cases(tally, runs, &spec, &run.captures, &mut image_rng)?;
        if tally.get("arms.member.members_replayed") > checked_before {
            tally.add("scenarios_checked", 1);
        }
    }
    Ok(())
}

/// The controls: the lies must fool the naive recovery, and the forged snapshot must fool the
/// unanchored take of it. A run where they do not cannot tell a recovery that notices lost
/// records from one that does not.
fn check_controls(tally: &Tally) -> Result<(), String> {
    let lies = tally.get("arms.cutover.lie_images");
    let fooled = tally.get("arms.cutover.b1_lie_images_accepted_stale");
    if lies == 0 || fooled == 0 {
        return Err(format!(
            "the lie control does not trip in the cutover arm: {lies} lie images, {fooled} accepted by the naive recovery with a stale revocation; the experiment could not tell a recovery that notices lost records from one that does not"
        ));
    }
    let tampered = tally.get("tamper_controls");
    let forged = tally.get("arms.member.b1_lie_images_accepted_stale");
    if tampered == 0 || forged == 0 {
        return Err(format!(
            "the forged-snapshot control does not trip: {tampered} refused by the anchor match, {forged} forged snapshots whose unanchored take resurrects a revoked generation; the experiment could not tell a member that vets a snapshot from one that does not"
        ));
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

/// What `--preflight` checks: the harness can run on this machine. It records nothing and
/// reads no seed.
pub fn preflight() -> Result<String, String> {
    model::self_check()?;
    if !fail::has_failpoints() {
        return Err("the failpoints of ptr-ledger are not compiled in".into());
    }
    let label = RUNS.fetch_add(1, Ordering::SeqCst);
    let runs = RunDir::new(0, &format!("preflight{label}"));
    let calibration = probe::calibrate(runs.path());
    require_probe(calibration)?;
    let filesystem = filesystem_of(runs.path());
    // The cutover and a member's snapshot are exercised once, end to end, on a fixed schedule.
    let workload = Arc::new(generate(
        &mut Rng::new(1),
        params::EVENTS_MIN,
        params::EVENTS_MAX,
    ));
    let cut = cutover::Cut { k: 6, keep: 1 };
    let mut runs = runs;
    let dir = runs.fresh();
    let obs = cutover::observe(&workload, cut, &dir, true);
    if let Some(failure) = obs.failure {
        return Err(format!("the cutover of the preflight failed: {failure}"));
    }
    let mut rng = Rng::new(2);
    let spec = group::spec_for("crash", &mut rng);
    let dir = runs.fresh();
    let run = group::run_life(dir, spec, Inject::None, true, false);
    let broken = run
        .counters
        .iter()
        .filter(|(name, value)| metrics::HARD_COUNTERS.contains(&name.as_str()) && *value > 0)
        .count();
    if broken > 0 {
        return Err(format!(
            "the unfaulted group of the preflight tripped {broken} hard counters"
        ));
    }
    Ok(format!(
        "{{\"preflight\":\"ok\",\"benchmark\":\"{}\",\"filesystem\":\"{filesystem}\",\"probe_sees_dirty\":{}}}",
        params::BENCHMARK,
        calibration.sees_dirty
    ))
}

/// Run every arm once for `seed` at `scale` and assemble the one JSON object. A run the harness
/// cannot make is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    static QUIET: std::sync::Once = std::sync::Once::new();
    QUIET.call_once(check::quiet_injected_panics);
    // The oracle against its table first: a wrong oracle is the harness's defect (status 2).
    model::self_check()?;
    let started = Instant::now();
    let shape = scale.shape();
    if plan_digest(seed, shape) != plan_digest(seed, shape) {
        return Err("the plan of a seed differs between two derivations".into());
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

    let mut rng = Rng::new(seed ^ SALT_CUTOVER);
    cutover_arm(&mut tally, &mut runs, &mut shapes, shape, &mut rng, probing)?;
    let mut rng = Rng::new(seed ^ SALT_GROUP);
    group_arm(&mut tally, &mut runs, &mut shapes, shape, &mut rng, probing)?;
    // A run that already tripped a falsifier may never have built its lie images: it is
    // falsified by what it counted, and its controls are not a harness failure.
    if metrics::hard_failures(&tally) == 0 {
        check_controls(&tally)?;
    }

    // The manifest's names from the counters of the arms.
    let cutover_recoveries = tally.get("arms.cutover.recoveries");
    tally.set("recoveries_cutover", cutover_recoveries);
    tally.set("events_min", shapes.events_min.unwrap_or(0) as i64);
    tally.set("subjects_min", shapes.subjects_min.unwrap_or(0) as i64);
    tally.set("revoke_then_recommit_histories", shapes.recommits);
    tally.seed_group("failpoint_hits", &metrics::HIT_POINTS);
    for name in metrics::METRIC_NAMES {
        if name != "failpoint_hits" {
            let value = tally.get(name);
            tally.set(name, value);
        }
    }
    let hard = metrics::hard_failures(&tally);

    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str(params::EXPERIMENT_ID)),
        (
            "status_note".to_string(),
            note("one seed of the L002 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
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
            ("durability_probe_available", Json::Bool(probing)),
            (
                "plan",
                Json::obj([
                    ("cutover_workloads", Json::Int(shape.cutover_workloads as i64)),
                    ("crash_scenarios_per_point", Json::Int(shape.crash_per_point as i64)),
                    ("ioerr_scenarios_per_point", Json::Int(shape.ioerr_per_point as i64)),
                    ("heal_scenarios", Json::Int(shape.heal as i64)),
                    ("tamper_scenarios", Json::Int(shape.tamper as i64)),
                    ("image_scenarios", Json::Int(shape.images as i64)),
                ]),
            ),
            (
                "failure_classes",
                note("process abort keeps the page cache, so every written byte survives: an in-process panic at a failpoint (the unwinding drops the handles, and no destructor of the code under test writes), recovered from the real files it leaves; power loss keeps the synced prefix of each file and a seeded subset of what was written after it (none, all, a prefix, half, 512-byte holes), an fsync-honest crash-image model and not a real power cut; an I/O error returned in place of a write; partitions, restarts, drops, duplicates and reorders of messages on a deterministic router of three in-process Raft members"),
            ),
            (
                "expectation",
                note("after a crash of the compaction cutover the live floor is the old one before the anchor was renamed and the new one after, the retained records are the acknowledged ones above it, the runtime restored from the snapshot and those records answers every (subject, generation) as the oracle does for the whole history, orphans are reclaimed without touching the live log, the cutover can be retried and the log goes on; a Raft member opened again holds what it had, every vote and acknowledged append in a message is already in its files, a call whose write failed releases no message, and after the faults every member holds the workload within the pinned number of heartbeat rounds; a snapshot that omits a revocation is refused by the anchor match and stays refused across a restart"),
            ),
            (
                "baselines",
                note("B0 is the same schedule without a fault, recovered through the same path and held to the same checks (the single-node reference is the workload itself, fed to the runtime in order); B1 is the naive recovery on the lie images (the anchor file trusted without the witness, the log alone with its tail trimmed) and the unanchored take of a forged snapshot, which must accept a stale revocation or the run is invalid"),
            ),
        ]),
    ));
    pairs.push((
        "controls".to_string(),
        Json::obj([
            (
                "cutover_lie",
                Json::obj([
                    (
                        "lie_images",
                        Json::Int(tally.get("arms.cutover.lie_images")),
                    ),
                    (
                        "naive_accepted_stale",
                        Json::Int(tally.get("arms.cutover.b1_lie_images_accepted_stale")),
                    ),
                    (
                        "anchored_refused",
                        Json::Int(tally.get("arms.cutover.refusals_lost_ack")),
                    ),
                    ("required_to_trip", Json::Bool(true)),
                ]),
            ),
            (
                "forged_snapshot",
                Json::obj([
                    ("refused_by_anchor", Json::Int(tally.get("tamper_controls"))),
                    (
                        "naive_accepted_stale",
                        Json::Int(tally.get("arms.member.b1_lie_images_accepted_stale")),
                    ),
                    ("required_to_trip", Json::Bool(true)),
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
                "recovery_time",
                note("recovery time is measured in leader heartbeat rounds (rounds_to_converge_total, rounds_to_converge_max), a logical count and not a duration; the pinned bound is judged by convergence_bound_exceeded and every wall-clock time (timing_ns) is descriptive"),
            ),
            (
                "durability_probe",
                note("the probe asks the kernel (posix_fadvise DONTNEED, then mincore) whether a file still holds pages it has not written to its disk; inside a run only clean readings are taken, after a member's call returned and where the cutover claims the new log or the anchor's scratch file durable; it sees file data pages only, not directory entries"),
            ),
            (
                "raft_ordering",
                note("the members return all the messages of a call from one call, so the order of persisting and sending inside RaftNode::drain is not observable by a router; what is checked is the contract of the call: nothing it returns promises a vote or an append that is not in the files when it returns, and a call whose write failed returns an error and no message"),
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

/// What the arguments after the subcommand ask for.
#[derive(Debug, Eq, PartialEq)]
enum Request {
    Run(u64, Scale),
    Preflight,
}

/// The seed and scale, or the preflight; strict, so a typo stops the run.
fn parse_args(args: &[String]) -> Result<Request, String> {
    let parse = |text: &String, what: &str| {
        text.parse::<u64>()
            .map_err(|_| format!("{what} must be a non-negative integer, got {text:?}"))
    };
    match args {
        [flag] if flag == "--preflight" => Ok(Request::Preflight),
        [seed] => parse(seed, "seed").map(|seed| Request::Run(seed, Scale::Pinned)),
        [cases, seed] => parse(cases, "cases").and_then(|cases| {
            if cases == 0 {
                return Err("cases must be at least 1".to_string());
            }
            parse(seed, "seed").map(|seed| Request::Run(seed, Scale::Custom(cases as usize)))
        }),
        _ => Err(
            "usage: raft-recovery <seed> | raft-recovery <cases> <seed> | raft-recovery --preflight"
                .to_string(),
        ),
    }
}

/// Run the experiment's harness and print its one JSON object. `args` are the arguments after
/// the subcommand: `<seed>` (the pinned scale), `<cases> <seed>` (the mutation checker's form:
/// that many workloads, no evidence) or `--preflight`. Exits 1 when the run is not clean and 2
/// when the harness cannot run.
pub fn run(args: &[String]) {
    let (seed, scale) = match parse_args(args) {
        Ok(Request::Preflight) => match preflight() {
            Ok(line) => {
                println!("{line}");
                return;
            }
            Err(reason) => {
                eprintln!("{}: preflight: {reason}", params::BENCHMARK);
                std::process::exit(2);
            }
        },
        Ok(Request::Run(seed, scale)) => (seed, scale),
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
        include_str!("../../../../../experiments/lifecycle/L002-raft-recovery/criteria.toml");
    const MUTATIONS: &str = include_str!(
        "../../../../../experiments/lifecycle/L002-raft-recovery/tests/mutations.toml"
    );
    const PREREGISTRATION: &str =
        include_str!("../../../../../experiments/lifecycle/L002-raft-recovery/PREREGISTRATION.md");

    /// The shared reduced run (one case of each kind).
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
        assert_eq!(metrics::falsifiers().len(), 18);
    }

    #[test]
    fn the_entrypoint_and_the_preflight_are_the_subcommand_and_the_status_is_planned_or_later() {
        assert_eq!(params::BENCHMARK, "raft-recovery");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert_eq!(entrypoint, params::ENTRYPOINT);
        assert_eq!(
            manifest["preflight"].as_str().expect("preflight"),
            params::PREFLIGHT
        );
        assert!(PREREGISTRATION.contains(params::ENTRYPOINT));
        assert!(PREREGISTRATION.contains(params::PREFLIGHT));
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
            .starts_with("{\"benchmark\":\"raft-recovery\",\"experiment\":\"L002\""));
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
    }

    /// The real product trips no falsifier on the reduced run, and the reduced run really
    /// exercises what it names.
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
        assert!(int(json, "cuts") >= 2);
        assert!(int(json, "lie_images") >= 1 || int(json, "cuts") >= 1);
        assert!(int(json, "snapshots_recorded") >= 2);
        assert!(int(json, "snapshots_installed") >= 1);
        assert!(int(json, "promises_checked") >= 10);
        assert!(int(json, "injected_io_errors") >= 1);
        assert!(int(json, "tamper_controls") >= 1);
        assert!(int(json, "b1_stale_accepts") >= 1);
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
        // A reduced run is small enough that a seed can draw no lie image at all, which the
        // harness refuses as a failed control: the first of a few other seeds that does not is
        // the one compared.
        let other = (1002..1008)
            .find_map(|seed| execute(seed, Scale::Reduced).ok())
            .expect("another seed gives a run");
        assert_ne!(first, strip(&other.json));
    }

    #[test]
    fn a_control_that_does_not_trip_is_a_harness_failure() {
        let mut tally = Tally::new(metrics::LIST);
        assert!(check_controls(&tally).is_err(), "no lies at all");
        tally.add("arms.cutover.lie_images", 3);
        assert!(
            check_controls(&tally).is_err(),
            "lies the naive recovery rejects too"
        );
        tally.add("arms.cutover.b1_lie_images_accepted_stale", 1);
        assert!(check_controls(&tally).is_err(), "no forged snapshot");
        tally.add("tamper_controls", 1);
        assert!(
            check_controls(&tally).is_err(),
            "a forged snapshot the unanchored take rejects"
        );
        tally.add("arms.member.b1_lie_images_accepted_stale", 1);
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
    }

    #[test]
    fn the_arguments_are_a_seed_or_cases_and_a_seed_or_the_preflight_and_nothing_else() {
        let args = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            parse_args(&args(&["17"])),
            Ok(Request::Run(17, Scale::Pinned))
        );
        assert_eq!(
            parse_args(&args(&["4", "17"])),
            Ok(Request::Run(17, Scale::Custom(4)))
        );
        assert_eq!(parse_args(&args(&["--preflight"])), Ok(Request::Preflight));
        for bad in [
            args(&[]),
            args(&["x"]),
            args(&["-1"]),
            args(&["0", "17"]),
            args(&["4", "x"]),
            args(&["1", "2", "3"]),
            args(&["--preflight", "17"]),
            args(&["--prefight"]),
        ] {
            assert!(parse_args(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_preflight_passes_on_a_machine_that_can_run_the_harness() {
        let line = preflight().expect("the preflight passes here");
        assert!(line.starts_with("{\"preflight\":\"ok\""));
    }
}
