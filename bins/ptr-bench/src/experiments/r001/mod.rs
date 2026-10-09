//! Experiment R001, mailbox backpressure, as a harness: `ptr-bench mailbox-backpressure <seed>`
//! runs the cases C1 to C9 of the experiment's preregistration once for one seed and
//! prints exactly one JSON object on stdout.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py`
//! under `criteria.toml`. The code under test is `ptr_exec::Mailbox` and `MailboxSender`,
//! driven only through their public API. The latency claim is gated in virtual ticks of a
//! discrete-event simulation (see `sim`), never in wall-clock time: the threaded arm records
//! wall-clock latency and the resident set as descriptive numbers that no rule reads, but its
//! ownership ledger and its capacity counts are checked like the simulation's.
//!
//! The object holds, in this order: the identity of the run, every metric (a JSON integer),
//! the timings (never read by a rule), the procedure, the per-case tables, the per-arm
//! summaries, the controls, the descriptive wall-clock numbers, the analytic expectations, the
//! counters that belong to no metric (`internal`), and the two judgements (`coverage` and
//! `falsifiers`). The exit status is 1 when a coverage prerequisite is unmet or a falsifier is
//! tripped, and 2 when the harness itself cannot run: a thread that cannot start, the oracle
//! disagreeing with the reference queue or the sync channel, or a negative control that does
//! not trip (the experiment could not then tell a broken queue from a sound one).
//!
//! A panic or a hang of the product is not a harness fault but a finding: every call into
//! `ptr_exec` is contained (`guard`), a panic is counted in the hard counter `product_panics`
//! and a threaded stage that misses its watchdog in `product_hangs`, and both falsify (exit 1).
//! Only a panic in harness code itself ends the process with status 101, which the aggregator
//! reads as INVALID.

pub mod cases;
mod guard;
mod ledger;
pub mod metrics;
pub mod params;
mod payload;
mod queue;
mod sim;
mod threaded;

use std::time::Instant;

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use cases::{CellRun, Pass, Probe, UnitOutcome};
use params::{PhaseSpec, ThreadedShape};
use sim::percentile_99;

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the smoke test only: the same simulation, a short threaded arm.
    #[cfg_attr(not(test), allow(dead_code))]
    Reduced,
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Reduced => "reduced",
        }
    }

    pub fn threaded(self) -> ThreadedShape {
        match self {
            Self::Pinned => params::THREADED_PINNED,
            Self::Reduced => params::THREADED_REDUCED,
        }
    }
}

/// Why the pair form exists: `scripts/mutation_check.py` runs every harness as `<cases> <seed>`.
const USAGE: &str = "usage: mailbox-backpressure <seed>; the form `<n> <seed>` that scripts/mutation_check.py uses is accepted only when both values are the same seed (R001 has no case count, and a pair that differs is refused rather than read as another seed)";

/// The seed of `mailbox-backpressure <seed>`. `scripts/mutation_check.py` appends the seed
/// twice (`<cases> <seed>`), so the same value twice is also accepted; anything else is a
/// usage error.
pub fn seed_from_args(args: &[String]) -> Result<u64, String> {
    let parse = |value: &String| {
        value
            .parse::<u64>()
            .map_err(|_| format!("seed must be a non-negative integer, got {value:?}"))
    };
    match args.get(2..).unwrap_or(&[]) {
        [seed] => parse(seed),
        [seed, again] if seed == again => parse(seed),
        [] => Err("seed is required: mailbox-backpressure <seed>".to_string()),
        _ => Err(USAGE.to_string()),
    }
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

fn int(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// The worst cell by the ratio `value / bound`, compared in integers.
fn worst_ratio(pairs: impl Iterator<Item = (u64, u64)>) -> (u64, u64) {
    let mut worst: Option<(u64, u64)> = None;
    for (value, bound) in pairs {
        worst = match worst {
            Some((w, b))
                if u128::from(w) * u128::from(bound) >= u128::from(value) * u128::from(b) =>
            {
                Some((w, b))
            }
            _ => Some((value, bound)),
        };
    }
    worst.unwrap_or((0, 1))
}

/// Fill every metric from the outcomes. Pure arithmetic over what the arms measured.
pub fn measure(
    pass: &Pass,
    determinism_mismatches: u64,
    probes: &[Probe],
    threaded: &threaded::Outcome,
    zero: &UnitOutcome,
    owner: &UnitOutcome,
) -> Tally {
    let mut t = Tally::new(metrics::LIST);
    let sum = |t: &mut Tally, name: &str, value: u64| t.add(name, int(value));
    let (mut overload_offered, mut overload_refused) = (0u64, 0u64);
    let mut capacities_filled: Vec<usize> = Vec::new();
    let mut classes_alive: Vec<&str> = Vec::new();
    for run in &pass.runs {
        let (b, closed, u) = (&run.b, &run.closed, &run.u);
        sum(&mut t, "offered", b.report.offered);
        sum(&mut t, "admitted", b.report.admitted);
        sum(&mut t, "refused_full", b.report.refused_full);
        sum(&mut t, "refused_closed", b.report.refused_closed);
        sum(&mut t, "delivered", b.report.delivered);
        for outcome in [b, closed] {
            sum(&mut t, "product_panics", outcome.product_panics);
            sum(&mut t, "silent_drops", outcome.report.silent_drops);
            sum(
                &mut t,
                "ownership_violations",
                outcome.report.ownership_violations(),
            );
            sum(&mut t, "capacity_violations", outcome.capacity_violations);
            sum(&mut t, "fifo_violations", outcome.report.fifo_violations);
            sum(&mut t, "model_divergences", outcome.model_divergences);
            sum(&mut t, "spurious_refusals", outcome.spurious_refusals);
            sum(
                &mut t,
                "wait_bound_violations",
                outcome.wait_bound_violations,
            );
            sum(
                &mut t,
                "payload_bound_violations",
                outcome.payload_bound_violations,
            );
            sum(
                &mut t,
                "closed_path_mismatches",
                outcome.closed_path_mismatches,
            );
            for (cause, value) in [
                ("duplicate_deliveries", outcome.report.duplicate_deliveries),
                ("phantom_deliveries", outcome.report.phantom_deliveries),
                ("payload_mutations", outcome.report.payload_mutations),
                ("wrong_returns", outcome.report.wrong_returns),
                ("dropped_while_owned", outcome.report.dropped_while_owned),
                ("live_mismatches", outcome.report.live_mismatches),
                ("spurious_completions", outcome.report.spurious_completions),
                (
                    "drop_count_mismatches",
                    outcome.report.drop_count_mismatches,
                ),
            ] {
                sum(&mut t, &format!("ownership_causes.{cause}"), value);
            }
        }
        // The real mailbox against both oracles, whole run against whole run.
        sum(
            &mut t,
            "model_divergences",
            cases::against_reference(b, &run.r)
                + cases::against_reference(b, &run.s)
                + cases::against_reference(closed, &run.closed_s),
        );
        t.add(
            "closed_path_cases",
            i64::from(closed.report.refused_closed > 0),
        );
        t.add(
            "closed_while_full_cells",
            i64::from(closed.closed_over_full > 0),
        );
        t.add(
            "closed_while_nonempty_cells",
            i64::from(closed.drained_after_close > 0),
        );
        if b.peak_depth == run.cell.capacity {
            capacities_filled.push(run.cell.capacity);
        }
        if b.peak_bytes >= run.cell.class.bytes as i64 {
            classes_alive.push(run.cell.class.name);
        }
        t.add("recoveries_observed", i64::from(b.recovered));
        t.add(
            "unbounded_exceeding_cells",
            i64::from(u.peak_depth > run.cell.capacity),
        );
        for (stat, range) in b.phases.iter().zip(&pass.main.ranges) {
            // An overload phase counts when the real mailbox was observed refusing in it.
            if range.overload && stat.offered > 0 && stat.refused_full > 0 {
                t.add("overload_phases", 1);
                overload_offered += stat.offered;
                overload_refused += stat.refused_full;
            }
        }
    }
    t.set(
        "refusal_permille_overload",
        i64::try_from(overload_refused * 1000 / overload_offered.max(1)).unwrap_or(i64::MAX),
    );
    // The unit cases and the threaded arm add to the hard counters of the same names.
    for unit in [zero, owner] {
        sum(&mut t, "silent_drops", unit.report.silent_drops);
        sum(
            &mut t,
            "ownership_violations",
            unit.report.ownership_violations(),
        );
        sum(&mut t, "capacity_violations", unit.capacity_violations);
        sum(&mut t, "model_divergences", unit.model_divergences);
        sum(
            &mut t,
            "closed_path_mismatches",
            unit.closed_path_mismatches,
        );
        sum(&mut t, "product_panics", unit.product_panics);
    }
    t.set("owner_drop_cases", int(owner.owner_drop_cases));
    sum(
        &mut t,
        "closed_path_mismatches",
        threaded.closed_path_mismatches,
    );
    sum(&mut t, "product_panics", threaded.product_panics);
    t.set("product_hangs", int(threaded.product_hangs));
    t.add(
        "closed_path_cases",
        i64::from(zero.report.refused_closed > 0),
    );
    t.set("determinism_mismatches", int(determinism_mismatches));
    t.set(
        "canary_misses",
        probes
            .iter()
            .filter(|probe| !probe.negative_control && !probe.tripped)
            .count() as i64,
    );
    t.set(
        "negative_controls_tripped",
        probes
            .iter()
            .filter(|probe| probe.negative_control && probe.tripped)
            .count() as i64,
    );
    t.set(
        "threaded_ledger_violations",
        int(threaded.ledger_violations()),
    );
    t.set(
        "threaded_capacity_violations",
        int(threaded.capacity_violations),
    );
    t.set(
        "threaded_admitted_after_close",
        int(threaded.admitted_after_close),
    );
    t.set("threaded_producers", int(threaded.producers));
    t.set("threaded_fill_rounds", int(threaded.fill_rounds));
    t.set(
        "threaded_producers_with_admissions",
        int(threaded.producers_with_admissions),
    );
    t.set(
        "threaded_interleaved_fills",
        int(threaded.interleaved_fills),
    );
    t.set("threaded_round2_admitted", int(threaded.contend_admitted));
    t.set("threaded_round2_full", int(threaded.contend_full));
    t.set("threaded_round2_closed", int(threaded.contend_closed));
    t.set("threaded_offered", int(threaded.offered));
    t.set("threaded_admitted", int(threaded.admitted));
    t.set("threaded_refused_full", int(threaded.refused_full));
    t.set("threaded_refused_closed", int(threaded.refused_closed));
    t.set("threaded_delivered", int(threaded.delivered));

    // Peaks, and the worst cell for the two analytic bounds.
    let bounded = |select: fn(&CellRun) -> u64| pass.runs.iter().map(select).max().unwrap_or(0);
    t.set(
        "bounded_peak_depth",
        int(bounded(|run| run.b.peak_depth as u64)),
    );
    t.set(
        "unbounded_peak_depth",
        int(bounded(|run| run.u.peak_depth as u64)),
    );
    t.set(
        "unbounded_peak_payload_bytes",
        int(bounded(|run| run.u.peak_bytes.max(0) as u64)),
    );
    t.set(
        "unbounded_p99_wait_ticks",
        int(bounded(|run| run.u.p99_wait())),
    );
    t.set(
        "bounded_max_wait_ticks",
        int(bounded(|run| run.b.max_wait())),
    );
    let (p99, bound) = worst_ratio(
        pass.runs
            .iter()
            .map(|run| (run.b.p99_wait(), params::wait_bound(run.cell.capacity))),
    );
    t.set("bounded_p99_wait_ticks", int(p99));
    t.set("bounded_wait_bound_ticks", int(bound));
    let (bytes, bytes_bound) = worst_ratio(pass.runs.iter().map(|run| {
        (
            run.b.peak_bytes.max(0) as u64,
            params::payload_bound(run.cell.capacity, run.cell.class.bytes) as u64,
        )
    }));
    t.set("bounded_peak_payload_bytes", int(bytes));
    t.set("bounded_payload_bytes_bound", int(bytes_bound));
    // Derived from what was observed: a capacity at which the queue was seen full, a payload
    // class of which a body was seen alive.
    capacities_filled.sort_unstable();
    capacities_filled.dedup();
    t.set("capacities_exercised", capacities_filled.len() as i64);
    classes_alive.sort_unstable();
    classes_alive.dedup();
    t.set("payload_classes", classes_alive.len() as i64);

    let hard: i64 = metrics::HARD_COUNTERS.iter().map(|name| t.get(name)).sum();
    t.set("hard_failures", hard);
    t
}

fn phase_table(pass: &Pass, name: &str) -> Json {
    let rows = pass
        .runs
        .iter()
        .filter_map(|run| {
            let at = pass
                .main
                .ranges
                .iter()
                .position(|range| range.name == name)?;
            let stat = run.b.phases.get(at)?;
            Some(Json::obj([
                ("capacity", Json::Int(run.cell.capacity as i64)),
                ("class", Json::str(run.cell.class.name)),
                ("offered", Json::Int(int(stat.offered))),
                ("admitted", Json::Int(int(stat.admitted))),
                ("refused_full", Json::Int(int(stat.refused_full))),
                (
                    "refusal_permille",
                    Json::Int(int(stat.refused_full * 1000 / stat.offered.max(1))),
                ),
            ]))
        })
        .collect();
    Json::Arr(rows)
}

fn sweep_table(pass: &Pass) -> Json {
    Json::Arr(
        pass.runs
            .iter()
            .map(|run| {
                let (b, u, r) = (&run.b, &run.u, &run.r);
                Json::obj([
                    ("capacity", Json::Int(run.cell.capacity as i64)),
                    ("class", Json::str(run.cell.class.name)),
                    ("payload_bytes", Json::Int(run.cell.class.bytes as i64)),
                    ("offered", Json::Int(int(b.report.offered))),
                    ("admitted", Json::Int(int(b.report.admitted))),
                    ("refused_full", Json::Int(int(b.report.refused_full))),
                    ("delivered", Json::Int(int(b.report.delivered))),
                    ("peak_depth", Json::Int(b.peak_depth as i64)),
                    ("p99_wait_ticks", Json::Int(int(b.p99_wait()))),
                    ("max_wait_ticks", Json::Int(int(b.max_wait()))),
                    (
                        "wait_bound_ticks",
                        Json::Int(int(params::wait_bound(run.cell.capacity))),
                    ),
                    ("peak_payload_bytes", Json::Int(b.peak_bytes)),
                    (
                        "payload_bytes_bound",
                        Json::Int(params::payload_bound(
                            run.cell.capacity,
                            run.cell.class.bytes,
                        )),
                    ),
                    ("recovered", Json::Bool(b.recovered)),
                    (
                        "admitted_after_recovery",
                        Json::Int(int(b.admitted_after_recovery)),
                    ),
                    (
                        "refused_after_recovery",
                        Json::Int(int(b.refused_after_recovery)),
                    ),
                    ("end_tick", Json::Int(int(b.end_tick))),
                    (
                        "matches_reference",
                        Json::Bool(cases::against_reference(b, r) == 0),
                    ),
                    ("trace_digest", Json::str(format!("{:016x}", b.digest))),
                    ("unbounded_peak_depth", Json::Int(u.peak_depth as i64)),
                    ("unbounded_p99_wait_ticks", Json::Int(int(u.p99_wait()))),
                    ("unbounded_max_wait_ticks", Json::Int(int(u.max_wait()))),
                    ("unbounded_peak_payload_bytes", Json::Int(u.peak_bytes)),
                    (
                        "unbounded_capacity_violations",
                        Json::Int(int(u.capacity_violations)),
                    ),
                    (
                        "unbounded_wait_bound_violations",
                        Json::Int(int(u.wait_bound_violations)),
                    ),
                ])
            })
            .collect(),
    )
}

fn closed_table(pass: &Pass) -> Json {
    Json::Arr(
        pass.runs
            .iter()
            .map(|run| {
                let c = &run.closed;
                Json::obj([
                    ("capacity", Json::Int(run.cell.capacity as i64)),
                    ("class", Json::str(run.cell.class.name)),
                    (
                        "closed_before_arrival",
                        Json::Int(i64::from(params::CLOSE_BEFORE_ID)),
                    ),
                    ("offered", Json::Int(int(c.report.offered))),
                    ("admitted", Json::Int(int(c.report.admitted))),
                    ("refused_full", Json::Int(int(c.report.refused_full))),
                    ("refused_closed", Json::Int(int(c.report.refused_closed))),
                    ("delivered", Json::Int(int(c.report.delivered))),
                    (
                        "closed_path_mismatches",
                        Json::Int(int(c.closed_path_mismatches)),
                    ),
                    ("model_divergences", Json::Int(int(c.model_divergences))),
                ])
            })
            .collect(),
    )
}

fn probe_table(probes: &[Probe]) -> Json {
    Json::Arr(
        probes
            .iter()
            .map(|probe| {
                Json::obj([
                    ("name", Json::str(probe.name)),
                    (
                        "kind",
                        Json::str(if probe.negative_control {
                            "negative_control"
                        } else {
                            "canary"
                        }),
                    ),
                    ("expected", Json::str(probe.expected)),
                    ("tripped", Json::Bool(probe.tripped)),
                    (
                        "silent_drops",
                        Json::Int(int(probe.observed.report.silent_drops)),
                    ),
                    (
                        "ownership_violations",
                        Json::Int(int(probe.observed.report.ownership_violations())),
                    ),
                    (
                        "capacity_violations",
                        Json::Int(int(probe.observed.capacity_violations)),
                    ),
                    (
                        "model_divergences",
                        Json::Int(int(probe.observed.model_divergences)),
                    ),
                    (
                        "wait_bound_violations",
                        Json::Int(int(probe.observed.wait_bound_violations)),
                    ),
                    (
                        "spurious_refusals",
                        Json::Int(int(probe.observed.spurious_refusals)),
                    ),
                    (
                        "fifo_violations",
                        Json::Int(int(probe.observed.report.fifo_violations)),
                    ),
                    (
                        "closed_path_mismatches",
                        Json::Int(int(probe.observed.closed_path_mismatches)),
                    ),
                ])
            })
            .collect(),
    )
}

fn phases_json(phases: &[PhaseSpec]) -> Json {
    Json::Arr(
        phases
            .iter()
            .map(|phase| {
                Json::obj([
                    ("name", Json::str(phase.name)),
                    ("count", Json::Int(i64::from(phase.count))),
                    ("gap_lo", Json::Int(phase.gap_lo as i64)),
                    ("gap_hi", Json::Int(phase.gap_hi as i64)),
                    ("overload", Json::Bool(phase.overload())),
                ])
            })
            .collect(),
    )
}

fn procedure_json(seed: u64, scale: Scale) -> Json {
    let shape = scale.threaded();
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        ("seed_is_declared", Json::Bool(params::DECLARED_SEEDS.contains(&seed))),
        ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
        (
            "capacities",
            Json::ints(params::CAPACITIES.iter().map(|c| *c as i64)),
        ),
        (
            "payload_classes",
            Json::Arr(
                params::PAYLOAD_CLASSES
                    .iter()
                    .map(|class| {
                        Json::obj([
                            ("name", Json::str(class.name)),
                            ("bytes", Json::Int(class.bytes as i64)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("service_ticks_min", Json::Int(params::SERVICE_MIN as i64)),
        ("service_ticks_max", Json::Int(params::SERVICE_MAX as i64)),
        ("phases", phases_json(&params::PHASES)),
        ("closed_phases", phases_json(&params::CLOSED_PHASES)),
        ("close_before_id", Json::Int(i64::from(params::CLOSE_BEFORE_ID))),
        (
            "tie_break",
            Json::str("(tick, seq); every arrival is scheduled before any completion, so at an equal tick the arrival fires first"),
        ),
        (
            "consumer",
            Json::str("one server; it takes a message the moment it is idle and one is queued; wait = tick the service starts - tick of arrival"),
        ),
        (
            "threaded",
            Json::obj([
                ("producers", Json::Int(shape.producers as i64)),
                ("capacity", Json::Int(shape.capacity as i64)),
                ("fill_rounds", Json::Int(i64::from(shape.fill_rounds))),
                ("fill_per_producer", Json::Int(i64::from(shape.fill_per_producer))),
                ("watchdog_seconds", Json::Int(threaded::WATCHDOG.as_secs() as i64)),
                ("contend_per_producer", Json::Int(i64::from(shape.contend_per_producer))),
                ("closed_per_producer", Json::Int(i64::from(shape.closed_per_producer))),
                ("close_after_received", Json::Int(i64::from(shape.close_after_received))),
            ]),
        ),
        ("control_capacity", Json::Int(params::CONTROL_CAPACITY as i64)),
    ])
}

fn percentile_ns(values: &[u64], numerator: usize) -> i64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * numerator).div_ceil(100).max(1);
    int(sorted[rank - 1])
}

/// Run every case once for `seed` at `scale` and assemble the one JSON object. A case the
/// harness cannot run is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let mut lap = Instant::now();
    let mut phases: Vec<(String, Json)> = Vec::new();
    let mut phase = |name: &str, lap: &mut Instant| {
        phases.push((
            name.to_string(),
            Json::Float(lap.elapsed().as_secs_f64() * 1000.0),
        ));
        *lap = Instant::now();
    };

    let first = cases::pass(seed)?;
    phase("pass_1", &mut lap);
    let second = cases::pass(seed)?;
    let determinism = cases::determinism_mismatches(&first.digests, &second.digests);
    drop(second);
    phase("pass_2_determinism", &mut lap);
    let probes = cases::controls(&first)?;
    phase("controls", &mut lap);
    let zero = cases::capacity_zero(seed);
    let owner = cases::owner_drop(seed);
    let threaded = threaded::run(seed, scale.threaded())?;
    phase("threaded", &mut lap);

    let tally = measure(&first, determinism, &probes, &threaded, &zero, &owner);
    let (printed, gaps) = tally.metrics();
    let unsupported_names = tally.unsupported_names();

    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("R001")),
        (
            "status_note".to_string(),
            Json::str("one seed of the R001 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the exit status and the coverage and falsifiers lists below are the harness's own reading of the same rules"),
        ),
        ("seed".to_string(), Json::Int(seed as i64)),
        ("scale".to_string(), Json::str(scale.name())),
    ];
    pairs.extend(printed);
    pairs.push((
        "timing_ms".to_string(),
        Json::obj([
            (
                "elapsed",
                Json::Float(started.elapsed().as_secs_f64() * 1000.0),
            ),
            ("phases", Json::Obj(phases)),
            ("threaded", Json::Float(threaded.elapsed_ms)),
        ]),
    ));
    pairs.push(("procedure".to_string(), procedure_json(seed, scale)));
    pairs.push((
        "cases".to_string(),
        Json::obj([
            ("C1", Json::obj([("phase", Json::str("underload")), ("rows", phase_table(&first, "underload"))])),
            ("C2", Json::obj([("phase", Json::str("overload_x2")), ("rows", phase_table(&first, "overload_x2"))])),
            ("C3", Json::obj([("phase", Json::str("overload_x8")), ("rows", phase_table(&first, "overload_x8"))])),
            (
                "C4",
                Json::obj([
                    ("burst", phase_table(&first, "burst")),
                    ("recovery", phase_table(&first, "recovery")),
                ]),
            ),
            ("C5", Json::obj([("sweep", sweep_table(&first))])),
            (
                "C6",
                Json::obj([(
                    "note",
                    Json::str("the payload classes are the cells of C5 with the same capacity and different class; the peak payload bytes are read from the drop-tracker and bounded by (capacity + 1) * bytes"),
                )]),
            ),
            (
                "C7",
                Json::obj([
                    ("capacity_zero_offers", Json::Int(int(zero.offers))),
                    ("capacity_zero_refused_full", Json::Int(int(zero.report.refused_full))),
                    ("capacity_zero_refused_closed", Json::Int(int(zero.report.refused_closed))),
                    ("owner_drop_cases", Json::Int(int(owner.owner_drop_cases))),
                    ("owner_drop_closed_path_mismatches", Json::Int(int(owner.closed_path_mismatches))),
                    ("scheduled_close", closed_table(&first)),
                ]),
            ),
            (
                "C8",
                Json::obj([
                    ("producers", Json::Int(int(threaded.producers))),
                    ("offered", Json::Int(int(threaded.offered))),
                    ("admitted", Json::Int(int(threaded.admitted))),
                    ("refused_full", Json::Int(int(threaded.refused_full))),
                    ("refused_closed", Json::Int(int(threaded.refused_closed))),
                    ("delivered", Json::Int(int(threaded.delivered))),
                    ("peak_depth", Json::Int(threaded.peak_depth as i64)),
                    ("order_violations", Json::Int(int(threaded.order_violations))),
                    ("fill_rounds", Json::Int(int(threaded.fill_rounds))),
                    ("interleaved_fills", Json::Int(int(threaded.interleaved_fills))),
                    ("admitted_after_close", Json::Int(int(threaded.admitted_after_close))),
                    ("round2_admitted", Json::Int(int(threaded.contend_admitted))),
                    ("round2_full", Json::Int(int(threaded.contend_full))),
                    ("round2_closed", Json::Int(int(threaded.contend_closed))),
                ]),
            ),
            ("C9", Json::obj([("probes", probe_table(&probes))])),
        ]),
    ));
    pairs.push((
        "descriptive".to_string(),
        Json::obj([
            (
                "note",
                Json::str("wall-clock and resident-set numbers of the threaded arm; descriptive only, no rule reads anything under this key"),
            ),
            ("threaded_latency_ns_p50", Json::Int(percentile_ns(&threaded.latencies_ns, 50))),
            ("threaded_latency_ns_p99", Json::Int(percentile_ns(&threaded.latencies_ns, 99))),
            ("threaded_latency_ns_max", Json::Int(percentile_ns(&threaded.latencies_ns, 100))),
            ("threaded_peak_depth", Json::Int(threaded.peak_depth as i64)),
            (
                "peak_rss_kib",
                threaded.peak_rss_kib.map_or(Json::Null, |kib| Json::Int(int(kib))),
            ),
            (
                "virtual_p99_all_cells_ticks",
                Json::Int(int(percentile_99(
                    &first.runs.iter().flat_map(|run| run.b.waits.iter().copied()).collect::<Vec<_>>(),
                ))),
            ),
        ]),
    ));
    pairs.push((
        "expected".to_string(),
        Json::obj([
            ("wait_bound", Json::str("an admitted message has at most capacity - 1 messages ahead of it in the mailbox and at most one in service with at most SERVICE_MAX ticks left, so its wait is at most capacity * SERVICE_MAX ticks")),
            ("payload_bound", Json::str("between events at most capacity queued payloads and one in service are alive, so at most (capacity + 1) * body bytes")),
            ("messages_per_schedule", Json::Int(i64::from(params::message_count(&params::PHASES)))),
            ("cells", Json::Int(metrics::cell_count() as i64)),
        ]),
    ));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            ("scope", Json::str("ptr_exec::Mailbox and MailboxSender only; supervision, restart, async mailboxes and the RequestAdmission queue are out of scope")),
            ("bounded_p99_wait_ticks", Json::str("the p99 wait (nearest rank) of the cell whose p99 / (capacity * SERVICE_MAX) is largest, printed with that cell's bound, so the rule holds iff it holds in every cell; wait_bound_violations counts every single message over the bound")),
            ("bounded_peak_payload_bytes", Json::str("the largest live payload bytes between events of the cell whose peak / bound is largest, printed with that cell's bound")),
            ("silent_drops", Json::str("offered ids with no terminal state at the end (delivered, refused Full or refused Closed), summed over the sweep, the scheduled close and the unit cases")),
            ("threaded", Json::str("exact counts only: every one of the fill rounds admits exactly the capacity, an Ok after a returned close() is a violation, the contended stage must show admitted, Full and Closed outcomes, the last stage is all Closed, the ledger is checked after the threads join; a stage that misses the watchdog is product_hangs; wall-clock latency is descriptive")),
            ("oracles", Json::str("the shadow model (RefModel) runs in lockstep at every event; arm R (the same model as a queue) and arm S (a bounded std::sync::mpsc::sync_channel, independent code) are compared with the real mailbox whole run against whole run; the peak depth the mailbox records must equal the shadow's peak")),
            ("panics", Json::str("every call into ptr_exec is contained; a panic is counted in product_panics and falsifies, a panic in harness code is exit 101 (INVALID)")),
            ("derived_coverage", Json::str("overload_phases, capacities_exercised, payload_classes, closed_while_full_cells, closed_while_nonempty_cells and the threaded round-2 counts are counted from observed events (refusals seen, queue seen full, bodies seen alive, offers refused Closed while full, messages delivered after the close), not from the configuration")),
            ("controls", Json::str("the unbounded channel and a drop-oldest wrapper are negative controls (exit 2 if one does not trip); the other broken queues are canaries of the oracle (canary_misses)")),
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

/// Run the experiment's harness for `seed` and print its one JSON object. Exits 1 when the
/// run is not clean and 2 when the harness cannot run.
pub fn run(seed: u64) {
    // Panics of the product are contained and counted; their messages would only flood stderr.
    guard::silence_contained_panics();
    if !params::DECLARED_SEEDS.contains(&seed) {
        eprintln!(
            "seed {seed} is not one of the manifest's seeds {:?}; this is a dry run",
            params::DECLARED_SEEDS
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::layered_kv::emit;

    const CRITERIA: &str =
        include_str!("../../../../../experiments/runtime/R001-mailbox-backpressure/criteria.toml");

    /// The shared reduced run (every simulation case at full size, a short threaded arm).
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

    fn int_at(json: &Json, path: &str) -> i64 {
        match emit::resolve(json, path) {
            Some(Json::Int(value)) => *value,
            other => panic!("{path} is {other:?}"),
        }
    }

    #[test]
    fn the_criteria_file_is_exactly_the_rules_the_harness_judges_in_its_order() {
        let table: toml::Table = CRITERIA.parse().expect("criteria.toml is TOML");
        assert_eq!(rules(&table, "coverage"), metrics::coverage());
        assert_eq!(rules(&table, "falsifiers"), metrics::falsifiers());
        assert_eq!(table["experiment"].as_str(), Some("R001"));
        assert_eq!(
            table["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
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
    fn the_draft_preregistration_names_every_metric_and_states_every_rule() {
        for name in metrics::METRIC_NAMES {
            assert!(
                params::DRAFT_TEXT.contains(name),
                "{name} is not in the draft"
            );
        }
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(
                params::DRAFT_TEXT.contains(rule.as_str()),
                "{rule} is not in the draft"
            );
        }
        assert!(params::DRAFT_TEXT.contains(params::ENTRYPOINT));
        assert!(params::DRAFT_TEXT.contains("RequestAdmission"));
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "mailbox-backpressure");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert_eq!(manifest["id"].as_str(), Some("R001"));
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|seed| seed.as_integer().unwrap() as u64)
            .collect();
        assert_eq!(seeds, params::DECLARED_SEEDS);
    }

    #[test]
    fn the_seed_is_one_argument_or_the_mutation_checkers_pair() {
        let args = |rest: &[&str]| -> Vec<String> {
            ["ptr-bench", "mailbox-backpressure"]
                .iter()
                .chain(rest)
                .map(|s| s.to_string())
                .collect()
        };
        assert_eq!(seed_from_args(&args(&["17"])), Ok(17));
        assert_eq!(seed_from_args(&args(&["17", "17"])), Ok(17));
        assert!(seed_from_args(&args(&[])).is_err());
        assert!(seed_from_args(&args(&["17", "18"])).is_err());
        assert!(seed_from_args(&args(&["x"])).is_err());
        assert!(seed_from_args(&args(&["-1"])).is_err());
        assert!(seed_from_args(&args(&["1", "1", "1"])).is_err());
    }

    #[test]
    fn a_reduced_run_assembles_one_json_object_with_every_metric_as_an_integer() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report
            .line
            .starts_with("{\"benchmark\":\"mailbox-backpressure\",\"experiment\":\"R001\""));
        assert!(report.line.contains("\"harness_gaps\":[]"), "no gap");
        for name in metrics::METRIC_NAMES {
            assert!(
                matches!(emit::resolve(&report.json, name), Some(Json::Int(_))),
                "{name}"
            );
        }
        assert_eq!(int_at(&report.json, "seed"), 17);
        assert!(matches!(
            emit::resolve(&report.json, "scale"),
            Some(Json::Str(scale)) if scale == "reduced"
        ));
    }

    #[test]
    fn a_reduced_run_is_a_clean_pass_with_the_expected_exact_counts() {
        let report = reduced_report();
        // The reduced run repeats the exact fill 100 times, not the pinned 2000, so the two
        // coverage rules on the number of fills (and of fills with producers meeting) may be unmet,
        // and no other; nothing is tripped.
        let unmet: Vec<&str> = report
            .coverage
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        assert!(
            unmet.iter().all(|rule| {
                rule.starts_with("threaded_fill_rounds")
                    || rule.starts_with("threaded_interleaved_fills")
            }),
            "{unmet:?} {}",
            report.line
        );
        assert!(unmet.contains(&"threaded_fill_rounds >= 2000"));
        assert_eq!(report.falsifiers_tripped(), 0, "{}", report.line);
        assert!(report.gaps.is_empty());
        let j = &report.json;
        assert_eq!(int_at(j, "hard_failures"), 0);
        let cells = metrics::cell_count() as i64;
        assert_eq!(int_at(j, "offered"), 2300 * cells);
        assert_eq!(
            int_at(j, "delivered") + int_at(j, "refused_full") + int_at(j, "refused_closed"),
            int_at(j, "offered")
        );
        assert_eq!(int_at(j, "admitted"), int_at(j, "delivered"));
        assert_eq!(int_at(j, "silent_drops"), 0);
        assert_eq!(int_at(j, "ownership_violations"), 0);
        assert_eq!(int_at(j, "determinism_mismatches"), 0);
        assert_eq!(int_at(j, "canary_misses"), 0);
        assert_eq!(int_at(j, "negative_controls_tripped"), 2);
        assert_eq!(int_at(j, "unbounded_exceeding_cells"), cells);
        assert_eq!(int_at(j, "recoveries_observed"), cells);
        assert_eq!(int_at(j, "capacities_exercised"), 3);
        assert_eq!(int_at(j, "payload_classes"), 2);
        assert_eq!(int_at(j, "overload_phases"), 3 * cells);
        assert!(int_at(j, "bounded_peak_depth") <= 128);
        assert!(int_at(j, "unbounded_peak_depth") > 128);
        assert!(int_at(j, "bounded_p99_wait_ticks") <= int_at(j, "bounded_wait_bound_ticks"));
        assert!(int_at(j, "bounded_max_wait_ticks") <= 128 * 32);
        assert!(
            int_at(j, "unbounded_p99_wait_ticks") > int_at(j, "bounded_wait_bound_ticks"),
            "the baseline's tail is far above the bounded bound"
        );
        assert!(int_at(j, "refusal_permille_overload") > 0);
        assert!(int_at(j, "refusal_permille_overload") < 1000);
        assert_eq!(int_at(j, "threaded_producers"), 4);
        assert_eq!(int_at(j, "product_panics"), 0);
        assert_eq!(int_at(j, "product_hangs"), 0);
        assert_eq!(int_at(j, "threaded_admitted_after_close"), 0);
        assert_eq!(int_at(j, "owner_drop_cases"), 1);
        assert!(int_at(j, "closed_while_full_cells") >= 4);
        assert_eq!(int_at(j, "closed_while_nonempty_cells"), cells);
        assert_eq!(
            int_at(j, "threaded_fill_rounds"),
            i64::from(params::THREADED_REDUCED.fill_rounds)
        );
        for outcome in ["admitted", "full", "closed"] {
            assert!(
                int_at(j, &format!("threaded_round2_{outcome}")) >= 1,
                "{outcome}"
            );
        }
        assert!(int_at(j, "threaded_producers_with_admissions") >= 2);
        assert_eq!(int_at(j, "threaded_ledger_violations"), 0);
        assert_eq!(int_at(j, "threaded_capacity_violations"), 0);
        assert_eq!(
            int_at(j, "threaded_admitted"),
            int_at(j, "threaded_delivered")
        );
        assert_eq!(int_at(j, "closed_path_cases"), cells + 1);
        // The first rows of the case tables are the lowest capacity, the small class.
        assert!(matches!(
            emit::resolve(j, "cases.C5.sweep"),
            Some(Json::Arr(rows)) if rows.len() == metrics::cell_count()
        ));
        assert!(matches!(
            emit::resolve(j, "cases.C9.probes"),
            Some(Json::Arr(rows)) if rows.len() == 11
        ));
    }

    #[test]
    fn a_reduced_run_underload_never_refuses() {
        let json = &reduced_report().json;
        let Some(Json::Arr(rows)) = emit::resolve(json, "cases.C1.rows") else {
            panic!("C1 rows");
        };
        assert_eq!(rows.len(), metrics::cell_count());
        for row in rows {
            assert_eq!(emit::resolve(row, "refused_full"), Some(&Json::Int(0)));
            assert_eq!(emit::resolve(row, "offered"), Some(&Json::Int(200)));
        }
    }

    #[test]
    fn no_rule_reads_a_timing_or_a_descriptive_number() {
        let json = &reduced_report().json;
        assert!(matches!(
            emit::resolve(json, "timing_ms.elapsed"),
            Some(Json::Float(_))
        ));
        assert!(emit::resolve(json, "descriptive.threaded_latency_ns_p99").is_some());
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            for word in ["timing", "elapsed", "descriptive", "latency", "rss", "_ns"] {
                assert!(!rule.contains(word), "{rule}");
            }
        }
    }

    #[test]
    fn the_measures_fail_when_the_pass_is_broken() {
        // The same arithmetic over a pass whose bounded arm was replaced by the unbounded one
        // trips the hard counters, so the rules are not satisfied by construction.
        let mut pass = cases::pass(17).unwrap();
        let probes = cases::controls(&pass).unwrap();
        let threaded = threaded::run(17, params::THREADED_REDUCED).unwrap();
        let zero = cases::capacity_zero(17);
        let owner = cases::owner_drop(17);
        let clean = measure(&pass, 0, &probes, &threaded, &zero, &owner);
        assert_eq!(clean.get("hard_failures"), 0);
        for run in &mut pass.runs {
            std::mem::swap(&mut run.b, &mut run.u);
        }
        let broken = measure(&pass, 0, &probes, &threaded, &zero, &owner);
        assert!(broken.get("capacity_violations") > 0);
        assert!(broken.get("model_divergences") > 0);
        assert!(broken.get("wait_bound_violations") > 0);
        assert!(broken.get("payload_bound_violations") > 0);
        assert!(broken.get("hard_failures") > 0);
        assert!(broken.get("bounded_p99_wait_ticks") > broken.get("bounded_wait_bound_ticks"));
        // A nondeterministic harness and a missed canary count too.
        let determinism = measure(
            &cases::pass(17).unwrap(),
            3,
            &probes,
            &threaded,
            &zero,
            &owner,
        );
        assert_eq!(determinism.get("determinism_mismatches"), 3);
        assert_eq!(determinism.get("hard_failures"), 3);
    }

    #[test]
    fn the_worst_ratio_is_the_cell_nearest_its_bound() {
        assert_eq!(worst_ratio([(5, 10), (9, 10), (3, 2)].into_iter()), (3, 2));
        assert_eq!(worst_ratio([(10, 10), (20, 20)].into_iter()), (10, 10));
        assert_eq!(worst_ratio(std::iter::empty()), (0, 1));
        // A violation in the smallest cell is not hidden by slack in a large one.
        assert_eq!(worst_ratio([(33, 32), (100, 4096)].into_iter()), (33, 32));
    }

    #[test]
    fn the_scales_are_named_and_only_the_pinned_one_has_the_pinned_shape() {
        assert_eq!(Scale::Pinned.name(), "pinned");
        assert_eq!(Scale::Reduced.name(), "reduced");
        assert_eq!(Scale::Pinned.threaded(), params::THREADED_PINNED);
        assert_ne!(Scale::Reduced.threaded(), params::THREADED_PINNED);
    }
}
