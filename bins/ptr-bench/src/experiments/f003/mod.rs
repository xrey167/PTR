//! Experiment F003, the calibrated arbiter, as a harness: `ptr-bench calibrated-arbiter <seed>`
//! runs the experiment once for one seed and prints exactly one JSON object on stdout.
//!
//! The code under test is `ptr_branch`'s arbiter: `TriagePolicy::triage`, `calibration_draw`,
//! `TriageOutcome::adjudicate`, `PolicyRecord::calibrate` and `held_out`, `certify_threshold`,
//! `calibrate_threshold`, `evaluate_off_policy` and `doubly_robust`, driven only through the
//! public API, with `ptr_analytics::clopper_pearson_upper` as the bound they rest on. The
//! adjudicator is a simulation with a known error rate (PROTOCOL.md): no person adjudicates
//! anything here, and the claim is the mechanism's, not human adjudication's.
//!
//! One run is one seed's evidence, judged mechanically by the experiment's `aggregate.py` under
//! `criteria.toml`. The object holds, in this order: the identity of the run, every metric (a
//! JSON integer), the timings (never read by a rule), the procedure, the per-policy tables, the
//! arms, the Monte-Carlo and off-policy tables, the controls, the measurement notes, the
//! counters that belong to no metric (`internal`), and the two judgements (`coverage` and
//! `falsifiers`). The exit status is 1 when a coverage prerequisite is unmet or a falsifier is
//! tripped, and 2 when the harness itself cannot run: the grid and the integer rule disagree,
//! a run is not reproducible, or a negative control does not trip. A panic of the product is
//! contained, counted in `product_panics` and falsifies (exit 1); only a panic in harness code
//! ends the process with status 101, which the aggregator reads as INVALID.

mod guard;
pub mod heldout;
pub mod mc;
pub mod metrics;
pub mod model;
pub mod ope;
pub mod oracle;
pub mod params;
pub mod stats;
pub mod stream;

use std::time::Instant;

use ptr_branch::{AutoThreshold, PolicyRecord};

use super::layered_kv::emit::{count, judge, Check, Gaps, Json, Status, Tally};
use super::rng::Rng;
use params::Shape;

/// The scale of a run: only the pinned one is the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Pinned,
    /// Used by the unit tests only: the same code on a shorter stream and fewer replicates.
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

    pub fn shape(self) -> Shape {
        match self {
            Self::Pinned => params::PINNED,
            Self::Reduced => params::REDUCED,
        }
    }
}

const USAGE: &str = "usage: calibrated-arbiter <seed>; the form `<n> <seed>` that scripts/mutation_check.py uses is accepted only when both values are the same seed (F003 has no case count, and a pair that differs is refused rather than read as another seed)";

/// The seed of `calibrated-arbiter <seed>`. `scripts/mutation_check.py` appends the seed twice
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
        [] => Err("seed is required: calibrated-arbiter <seed>".to_string()),
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

fn ppm(value: f64) -> i64 {
    (value * 1_000_000.0).round() as i64
}

fn opt_ppm(value: Option<f64>) -> i64 {
    value.map_or(-1, ppm)
}

fn opt_permille(j: Option<usize>) -> i64 {
    j.map_or(-1, |j| j as i64)
}

fn ratio_ppm(numerator: u64, denominator: u64) -> i64 {
    if denominator == 0 {
        0
    } else {
        (numerator as u128 * 1_000_000 / denominator as u128) as i64
    }
}

fn policy_row(index: usize, report: &heldout::PolicyReport) -> Json {
    Json::obj([
        ("version", Json::str(report.version.clone())),
        ("index", Json::Int(index as i64)),
        ("calibration_adjudications", Json::Int(report.n_cal as i64)),
        (
            "threshold_permille",
            Json::Int(opt_permille(report.product_j)),
        ),
        (
            "oracle_threshold_permille",
            Json::Int(opt_permille(report.oracle_j)),
        ),
        ("heldout_adjudications", Json::Int(report.heldout as i64)),
        ("heldout_candidates", Json::Int(report.candidates as i64)),
        ("heldout_harmful", Json::Int(report.harmful as i64)),
        ("p_value_ppm", Json::Int(ppm(report.p_value))),
        (
            "rejected_at_bonferroni_level",
            Json::Int(i64::from(report.violation)),
        ),
        (
            "true_adjudicated_harm_ppm",
            Json::Int(opt_ppm(report.true_harm)),
        ),
        (
            "true_harm_exceeds_alpha",
            Json::Int(i64::from(report.true_exceeds)),
        ),
        ("literal_ucb_ppm", Json::Int(opt_ppm(report.literal_ucb))),
        (
            "exact_lower_bound_ppm",
            Json::Int(opt_ppm(report.lower_bound)),
        ),
        (
            "literal_ucb_exceeds_alpha",
            Json::Int(i64::from(report.literal_exceeds)),
        ),
        (
            "breakdown_by_namespace",
            Json::Obj(
                report
                    .breakdown
                    .iter()
                    .map(|(ns, candidates, harmful)| {
                        (
                            ns.clone(),
                            Json::obj([
                                ("candidates", Json::Int(*candidates as i64)),
                                ("harmful", Json::Int(*harmful as i64)),
                            ]),
                        )
                    })
                    .collect(),
            ),
        ),
    ])
}

fn arm_row(name: &str, arm: &stream::ArmTally) -> Json {
    Json::obj([
        ("arm", Json::str(name)),
        ("eligible_branches", Json::Int(arm.eligible as i64)),
        ("auto_proposed", Json::Int(arm.auto as i64)),
        ("slice_escalations", Json::Int(arm.slice as i64)),
        (
            "reviews_saved_permille",
            Json::Int(ratio_ppm(arm.auto, arm.eligible) / 1000),
        ),
        ("harmful_auto_proposals", Json::Int(arm.harmful_auto as i64)),
        (
            "realized_true_harm_ppm",
            Json::Int(ratio_ppm(arm.harmful_auto, arm.auto)),
        ),
    ])
}

/// The drift control: a held-out set drawn from a world whose harm is several times the
/// model's, tested at the threshold of the last certified policy. The held-out test must
/// reject it. Returns `(p_value, rejected)`.
fn drift_control(seed: u64, j: usize) -> (f64, bool) {
    let mut rng = Rng::new(seed ^ 0xF003_D21F_0000_0004);
    let (mut n, mut k) = (0_usize, 0_usize);
    while n < params::DRIFT_SAMPLES {
        let ns = model::draw_namespace(&mut rng);
        let m = model::draw_score(ns, &mut rng);
        if !model::admits(j, m) {
            continue;
        }
        n += 1;
        let harm = model::draw_harm(ns, m, params::DRIFT_HARM_FACTOR, &mut rng);
        let flipped = rng.below(1000) < u64::from(params::EPSILON_PERMILLE);
        k += usize::from(harm != flipped);
    }
    heldout::rejects(k, n)
}

/// Run the experiment for `seed` at `scale` and assemble the one JSON object. A run the harness
/// cannot carry out is an `Err` with the reason.
pub fn execute(seed: u64, scale: Scale) -> Result<Report, String> {
    let started = Instant::now();
    let shape = scale.shape();
    model::grid_self_check()?;
    let truth = model::Truth::new();
    let mut t = Tally::new(metrics::LIST);
    // A counter that never fires prints as zero, not as a gap.
    for name in metrics::HARD_COUNTERS.iter().chain(&["nonslice_probes"]) {
        t.add(name, 0);
    }

    // The stream through the real pipeline, and its determinism rerun.
    let drafts = stream::generate(seed, &shape);
    let lap = Instant::now();
    let pass = stream::run(seed, &shape, &drafts, &mut t)?;
    let stream_ms = lap.elapsed().as_secs_f64() * 1000.0;
    let repeat = stream::run(seed, &shape, &drafts, &mut Tally::new(metrics::LIST))?;
    if repeat.digest != pass.digest {
        return Err("the stream pass is not reproducible (digest differs between two runs)".into());
    }

    // Held-out evaluation of every recorded policy, and of the uncalibrated baseline.
    let mut reports = Vec::new();
    for recorded in &pass.recorded {
        reports.push(heldout::evaluate(
            &mut t,
            &drafts,
            &pass.adjudicated,
            &pass.index,
            &recorded.record,
            recorded.n_cal,
            recorded.oracle_j,
            &truth,
        ));
    }
    let unc_policy = stream::make_policy(
        &mut t,
        AutoThreshold::AtLeast(params::UNCALIBRATED_THRESHOLD_PERMILLE as f32 / 1000.0),
        params::slice_rate(),
    );
    let baseline_record = unc_policy.and_then(|policy| {
        stream::product(&mut t, || PolicyRecord::manual("uncalibrated", policy))
    });
    let baseline = match &baseline_record {
        Some(record) => heldout::evaluate(
            &mut t,
            &drafts,
            &pass.adjudicated,
            &pass.index,
            record,
            0,
            Some(params::UNCALIBRATED_THRESHOLD_PERMILLE as usize),
            &truth,
        ),
        None => heldout::PolicyReport::absent("uncalibrated"),
    };
    let (bound_mismatches, bound_pairs) = heldout::bound_grid(&mut t);

    let (slice_tests, slice_violations, slice_worst_milli) =
        stream::slice_rate_tests(&pass.slice_groups);

    // Monte-Carlo coverage and off-policy evaluation, each with a rerun of its prefix.
    let lap = Instant::now();
    let mc = mc::run(
        seed,
        &shape,
        shape.mc_replicates,
        &truth,
        params::EPSILON_PERMILLE,
        &mut t,
    )?;
    let mc_ms = lap.elapsed().as_secs_f64() * 1000.0;
    let prefix = shape.mc_replicates.min(params::DETERMINISM_REPLICATES);
    let mc_again = mc::run(
        seed,
        &shape,
        prefix,
        &truth,
        params::EPSILON_PERMILLE,
        &mut Tally::new(metrics::LIST),
    )?;
    if mc_again.digest != mc.prefix_digest {
        return Err("the Monte-Carlo replicates are not reproducible".into());
    }
    // The descriptive arm: the same replicates at an adjudicator error rate of 20 per mille.
    let mc_noisy = mc::run(
        seed,
        &shape,
        shape.mc_replicates,
        &truth,
        params::NOISY_EPSILON_PERMILLE,
        &mut t,
    )?;
    let lap = Instant::now();
    let ope = ope::run(seed, &shape, shape.ope_logs, &truth, &mut t)?;
    let ope_ms = lap.elapsed().as_secs_f64() * 1000.0;
    let ope_again = ope::run(
        seed,
        &shape,
        shape.ope_logs.min(params::DETERMINISM_LOGS),
        &truth,
        &mut Tally::new(metrics::LIST),
    )?;
    if ope_again.digest != ope.digest {
        return Err("the off-policy estimates are not reproducible".into());
    }

    // Judgements of the tests.
    let critical = stats::coverage_critical(
        shape.mc_replicates,
        params::delta(),
        params::coverage_level(),
    );
    t.set("ltt_coverage_violations", mc.violations);
    t.set("ltt_coverage_critical", critical as i64);
    t.set(
        "ltt_coverage_excess",
        (mc.violations - critical as i64).max(0),
    );
    let mut ips_trips = 0;
    let mut dr_trips = 0;
    let mut ope_table = Vec::new();
    let mut unweighted_tripped = false;
    for (ti, target) in ope::TARGETS.iter().enumerate() {
        let mut rows = Vec::new();
        for (ei, name) in ope::ESTIMATORS.iter().enumerate() {
            let list = &ope.errors[ti][ei];
            let (mean, se, trips) = if list.errors.is_empty() {
                (0.0, 0.0, true)
            } else {
                ope::bias_test(list)
            };
            match *name {
                "ips" => ips_trips += i64::from(trips),
                "dr_wrong_model" | "dr_near_right_model" => dr_trips += i64::from(trips),
                "unweighted_control" => unweighted_tripped |= trips,
                _ => {}
            }
            rows.push(Json::obj([
                ("estimator", Json::str(*name)),
                ("logs", Json::Int(list.errors.len() as i64)),
                ("mean_error_ppm", Json::Int(ppm(mean))),
                ("standard_error_ppm", Json::Int(ppm(se))),
                ("outside_bias_bound", Json::Int(i64::from(trips))),
            ]));
        }
        ope_table.push(Json::obj([
            ("target", Json::str(target.name)),
            (
                "threshold_permille",
                Json::Int(i64::from(target.threshold_permille)),
            ),
            (
                "slice_rate_permille",
                Json::Int(i64::from(target.rate_permille)),
            ),
            ("true_value_ppm", Json::Int(ppm(ope.truth_values[ti]))),
            ("estimators", Json::Arr(rows)),
        ]));
    }
    t.set("ips_bias_violations", ips_trips);
    t.set("dr_bias_violations", dr_trips);
    t.set("slice_rate_deviation_violations", slice_violations);
    t.set(
        "heldout_policy_violations",
        reports.iter().filter(|r| r.violation).count() as i64,
    );
    t.set(
        "heldout_violations_literal_ucb",
        reports.iter().filter(|r| r.literal_exceeds).count() as i64,
    );

    // Negative controls: each must trip, or the experiment could not tell a defect from a pass.
    let last_certified = reports.iter().rev().find_map(|r| r.product_j);
    let (drift_p, drift_rejected) = drift_control(
        seed,
        last_certified.unwrap_or(params::CONTROL_FALLBACK_THRESHOLD_PERMILLE as usize),
    );
    let naive_tripped = mc.naive_violations > critical as i64;
    let baseline_rejected = baseline.violation;
    let controls = [
        ("drifted_heldout", drift_rejected),
        ("point_estimate_without_bound", naive_tripped),
        ("uncalibrated_threshold_heldout", baseline_rejected),
        ("unweighted_off_policy_mean", unweighted_tripped),
    ];
    // A product that starves the harness of data (no adjudications, no completed replicates)
    // leaves controls that cannot be tested: that is a product defect, counted and falsifying
    // (exit 1). Only when the data were there and a control still did not trip is the harness at
    // fault (exit 2).
    let starved = pass.adjudicated.len() < shape.policy_sizes[0]
        || mc.replicates < (shape.mc_replicates / 2) as i64;
    t.set("data_starvation", i64::from(starved));
    if !starved {
        if let Some((name, _)) = controls.iter().find(|(_, tripped)| !tripped) {
            return Err(format!(
                "the negative control {name} did not trip: the experiment could not tell a defect from a pass"
            ));
        }
    }

    // Counters read by coverage rules.
    let certified: Vec<&heldout::PolicyReport> =
        reports.iter().filter(|r| r.product_j.is_some()).collect();
    t.set("stream_branches", drafts.len() as i64);
    t.set("eligible_branches", pass.arms[0].eligible as i64);
    t.set("adjudications", pass.adjudicated.len() as i64);
    t.set("policies_recorded", pass.recorded.len() as i64);
    t.set("policies_certified", certified.len() as i64);
    t.set(
        "ineligible_kinds_seen",
        pass.kinds_seen.iter().filter(|seen| **seen).count() as i64,
    );
    t.set("mc_replicates", mc.replicates);
    t.set("mc_certified_replicates", mc.certified);
    t.set("mc_never_replicates", mc.never);
    t.set("mc_naive_violations", mc.naive_violations);
    t.set("ope_logs", ope.logs);
    t.set("ope_estimates", ope.estimates);
    t.set("positivity_probes", ope.positivity_probes);
    t.set("slice_rate_tests", slice_tests);
    t.set(
        "heldout_candidates_min_certified",
        certified
            .iter()
            .map(|r| r.candidates as i64)
            .min()
            .unwrap_or(0),
    );
    t.set(
        "ltt_reviews_saved_permille",
        ratio_ppm(pass.arms[2].auto, pass.arms[2].eligible) / 1000,
    );
    t.set(
        "uncalibrated_reviews_saved_permille",
        ratio_ppm(pass.arms[1].auto, pass.arms[1].eligible) / 1000,
    );
    t.set(
        "stream_realized_harm_ltt_ppm",
        ratio_ppm(pass.arms[2].harmful_auto, pass.arms[2].auto),
    );
    t.set(
        "stream_realized_harm_uncalibrated_ppm",
        ratio_ppm(pass.arms[1].harmful_auto, pass.arms[1].auto),
    );
    t.set(
        "uncalibrated_true_harm_ppm",
        ppm(truth.adjudicated_harm_above(params::UNCALIBRATED_THRESHOLD_PERMILLE as usize)),
    );
    t.set(
        "uncalibrated_baseline_rejected",
        i64::from(baseline_rejected),
    );
    t.set(
        "negative_controls_tripped",
        controls.iter().filter(|(_, tripped)| *tripped).count() as i64,
    );
    let hard: i64 = metrics::HARD_COUNTERS.iter().map(|name| t.get(name)).sum();
    t.set("hard_failures", hard);

    let (printed, gaps) = t.metrics();
    let unsupported_names = t.unsupported_names();
    let mut pairs: Vec<(String, Json)> = vec![
        ("benchmark".to_string(), Json::str(params::BENCHMARK)),
        ("experiment".to_string(), Json::str("F003")),
        (
            "status_note".to_string(),
            Json::str("one seed of the F003 harness: evidence only when run at the pinned scale from the frozen commit and judged by aggregate.py under criteria.toml; the adjudicator is a simulation with a known error rate, not a person"),
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
            ("stream_pass", Json::Float(stream_ms)),
            ("monte_carlo", Json::Float(mc_ms)),
            ("off_policy", Json::Float(ope_ms)),
        ]),
    ));
    pairs.push(("procedure".to_string(), procedure_json(seed, scale, &shape)));
    pairs.push((
        "policies".to_string(),
        Json::Arr(
            reports
                .iter()
                .enumerate()
                .map(|(i, r)| policy_row(i + 1, r))
                .collect(),
        ),
    ));
    pairs.push((
        "uncalibrated_baseline".to_string(),
        policy_row(0, &baseline),
    ));
    pairs.push((
        "arms".to_string(),
        Json::Arr(vec![
            arm_row("escalate_all", &pass.arms[0]),
            arm_row("uncalibrated_threshold", &pass.arms[1]),
            arm_row("learn_then_test_in_force", &pass.arms[2]),
        ]),
    ));
    pairs.push((
        "monte_carlo".to_string(),
        Json::obj([
            ("replicates", Json::Int(mc.replicates)),
            ("certified", Json::Int(mc.certified)),
            ("never", Json::Int(mc.never)),
            ("violations", Json::Int(mc.violations)),
            ("critical", Json::Int(critical as i64)),
            (
                "violation_permille",
                Json::Int(ratio_ppm(mc.violations as u64, mc.replicates.max(1) as u64) / 1000),
            ),
            ("naive_certified", Json::Int(mc.naive_certified)),
            ("naive_violations", Json::Int(mc.naive_violations)),
            ("crc_checked", Json::Int(mc.crc_checked)),
            ("chosen_threshold_by_hundred", Json::ints(mc.by_hundred)),
        ]),
    ));
    pairs.push(("off_policy".to_string(), Json::Arr(ope_table)));
    pairs.push((
        "controls".to_string(),
        Json::obj([
            (
                "drifted_heldout",
                Json::obj([
                    ("p_value_ppm", Json::Int(ppm(drift_p))),
                    ("required_rule_tripped", Json::Bool(drift_rejected)),
                ]),
            ),
            (
                "point_estimate_without_bound",
                Json::obj([
                    ("violations", Json::Int(mc.naive_violations)),
                    ("critical", Json::Int(critical as i64)),
                    ("required_rule_tripped", Json::Bool(naive_tripped)),
                ]),
            ),
            (
                "uncalibrated_threshold_heldout",
                Json::obj([
                    ("p_value_ppm", Json::Int(ppm(baseline.p_value))),
                    ("required_rule_tripped", Json::Bool(baseline_rejected)),
                ]),
            ),
            (
                "unweighted_off_policy_mean",
                Json::obj([("required_rule_tripped", Json::Bool(unweighted_tripped))]),
            ),
        ]),
    ));
    pairs.push((
        "descriptive".to_string(),
        Json::obj([
            (
                "note",
                Json::str("descriptive only; no rule reads anything under this key"),
            ),
            ("bound_oracle_pairs_compared", Json::Int(bound_pairs)),
            ("bound_oracle_pair_mismatches", Json::Int(bound_mismatches)),
            ("slice_rate_worst_abs_z_milli", Json::Int(slice_worst_milli)),
            (
                "ineligible_kinds_decided_by_verification",
                Json::strs(
                    &model::Kind::ALL
                        .iter()
                        .filter(|kind| pass.kinds_seen[kind.index()])
                        .map(|kind| kind.name())
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "slice_rate_z_limit_milli",
                Json::Int(params::SLICE_Z_LIMIT_MILLI),
            ),
            ("heldout_level_ppm", Json::Int(ppm(params::heldout_level()))),
            (
                "coverage_level_ppm",
                Json::Int(i64::from(params::COVERAGE_TEST_LEVEL_PPM)),
            ),
            (
                "harm_breakdown_namespaces",
                Json::str("first ':'-separated segment of each changed key; descriptive only"),
            ),
            ("noisy_adjudicator_arm", noisy_arm_json(&mc_noisy, &shape)),
        ]),
    ));
    pairs.push((
        "measurement_notes".to_string(),
        Json::obj([
            ("adjudicator", Json::str("simulated: label = true harm XOR a hash-keyed flip at adjudicator_error_permille; it is shown the changed keys and verifier codes and never the intent; no person adjudicates anything")),
            ("coverage_test", Json::str("over the Monte-Carlo replicates, the fraction of certified thresholds whose exact adjudicated harm rate exceeds alpha must not exceed delta by more than an exact-binomial tolerance: violations <= the largest c with P(Bin(R, delta) >= c) >= the level; a replicate that certifies nothing cannot violate")),
            ("heldout_test", Json::str("per recorded policy, exact one-sided binomial test of harm <= alpha on the held-out adjudications the policy would auto-propose, rejected at familywise level / tests; the literal upper bound against alpha is printed (literal_ucb_exceeds_alpha) and judged by nothing, because a correct policy fails it by design")),
            ("derived_coverage", Json::str("every coverage counter is counted from events the run observed (slice_rate_tests counts the groups with observed branches, negative_controls_tripped the controls that tripped), except stream_branches (the generated length) and uncalibrated_true_harm_ppm (the model's exact value, whose observed counterpart is uncalibrated_baseline_rejected)")),
            ("oracles", Json::str("eligibility from the specification, the decision a policy must reach, the Learn-then-Test fixed-sequence rule (exact binomial tail, no bound), the conformal rule, the Clopper-Pearson bound by bisection on a log-space binomial CDF, the held-out sets by set difference, and the exact true harm by summing the model's tables; none reads a product value to compute an expectation")),
            ("panics", Json::str("every call into ptr_branch and ptr_analytics is contained (each stream step, record view, policy construction, estimator and bound call); a panic is counted in product_panics and falsifies, a refusal where success is required in product_errors, a product that starves the harness of data in data_starvation (all exit 1), and only a panic in harness code is exit 101 (INVALID); exit 2 is a harness fault (grid check, nondeterminism, or a control that did not trip although the data were there)")),
            ("calibration_draw", Json::str("the product's calibration_draw is also the draw the oracle's expected decision uses, so the slice itself is not derived independently; its uniformity is judged by the slice-rate z tests and its seed separation by the test below")),
            ("grid_ties", Json::str("30 percent of scores sit exactly on a grid point: a test-sensitivity device that makes >= against > observable, not a claim about real scorers")),
            ("detection_range", Json::str("the held-out test rejects an observed harm rate above about 7 percent at its level and the coverage test a violation rate above about 6.5 percent; smaller deviations are seen only by the exact oracle comparisons. The coverage test is one-sided at 1e-3 per seed (not Bonferroni-split over the seeds)")),
            ("start_rule", Json::str("the validity of the fixed sequence's data-dependent start (the first threshold with at least 59 admitted samples) is shown empirically for this scenario by the coverage count, not proven")),
            ("controls", Json::str("a drifted held-out set, a rule that estimates without a bound, the uncalibrated threshold under the held-out test and an unweighted off-policy mean are negative controls (exit 2 if one does not trip with the data present)")),
        ]),
    ));
    pairs.push(("internal".to_string(), t.internal()));
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

/// The ungated Monte-Carlo arm at the preregistered noisy adjudicator error rate: how often
/// Learn-then-Test certifies nothing, and how often what it certifies violates alpha.
fn noisy_arm_json(mc: &mc::McOutcome, shape: &Shape) -> Json {
    Json::obj([
        (
            "note",
            Json::str("ungated: the same replicates with the adjudicator wrong on 20 per mille of branches; a utility result (the start rule rarely certifies), not a gate"),
        ),
        ("adjudicator_error_permille", Json::Int(i64::from(params::NOISY_EPSILON_PERMILLE))),
        ("replicates", Json::Int(mc.replicates)),
        ("certified", Json::Int(mc.certified)),
        ("never", Json::Int(mc.never)),
        ("violations", Json::Int(mc.violations)),
        ("policy_sizes", Json::ints(shape.policy_sizes.iter().map(|n| *n as i64))),
        ("certified_by_size", Json::ints(mc.certified_by_size.iter().copied())),
        ("never_by_size", Json::ints(mc.never_by_size.iter().copied())),
        ("violations_by_size", Json::ints(mc.violations_by_size.iter().copied())),
    ])
}

fn procedure_json(seed: u64, scale: Scale, shape: &Shape) -> Json {
    Json::obj([
        ("scale", Json::str(scale.name())),
        ("seed", Json::Int(seed as i64)),
        (
            "seed_is_declared",
            Json::Bool(params::DECLARED_SEEDS.contains(&seed)),
        ),
        ("shape_is_pinned", Json::Bool(scale == Scale::Pinned)),
        ("stream_branches", Json::Int(shape.stream_branches as i64)),
        (
            "policy_sizes",
            Json::ints(shape.policy_sizes.iter().map(|n| *n as i64)),
        ),
        ("mc_replicates", Json::Int(shape.mc_replicates as i64)),
        ("ope_logs", Json::Int(shape.ope_logs as i64)),
        ("ope_log_branches", Json::Int(shape.ope_log_branches as i64)),
        (
            "alpha_permille",
            Json::Int(i64::from(params::ALPHA_PERMILLE)),
        ),
        (
            "delta_permille",
            Json::Int(i64::from(params::DELTA_PERMILLE)),
        ),
        (
            "slice_rate_permille",
            Json::Int(i64::from(params::SLICE_RATE_PERMILLE)),
        ),
        (
            "adjudicator_error_permille",
            Json::Int(i64::from(params::EPSILON_PERMILLE)),
        ),
        (
            "adjudicator",
            Json::str("simulated, deterministic, hash-keyed flip; never shown the intent"),
        ),
        (
            "uncalibrated_threshold_permille",
            Json::Int(i64::from(params::UNCALIBRATED_THRESHOLD_PERMILLE)),
        ),
        (
            "namespaces",
            Json::Arr(
                model::NAMESPACES
                    .iter()
                    .map(|ns| {
                        Json::obj([
                            ("name", Json::str(ns.name)),
                            ("weight_permille", Json::Int(i64::from(ns.weight_permille))),
                            ("score_is_max_of", Json::Int(i64::from(ns.maxes))),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

/// Run the experiment's harness for `seed` and print its one JSON object. Exits 1 when the run
/// is not clean and 2 when the harness cannot run.
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
    use ptr_branch::TriagePolicy;

    const CRITERIA: &str =
        include_str!("../../../../../experiments/feedback/F003-calibrated-arbiter/criteria.toml");

    /// The shared reduced run (the full code on a shorter stream and fewer replicates).
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
        assert_eq!(table["experiment"].as_str(), Some("F003"));
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
    fn every_falsifier_fails_when_the_emitted_counter_is_set_to_a_violation() {
        let report = reduced_report();
        let Json::Obj(pairs) = &report.json else {
            panic!("an object");
        };
        for rule in metrics::falsifiers() {
            let tokens: Vec<&str> = rule.split(' ').collect();
            let mut damaged = pairs.clone();
            let slot = damaged
                .iter_mut()
                .find(|(key, _)| key == tokens[0])
                .unwrap_or_else(|| panic!("{} is printed", tokens[0]));
            let Json::Int(current) = slot.1 else {
                panic!("an integer");
            };
            // `<= 0` rules are violated by one; the cross-reference rule by exceeding its bound.
            let bound = if tokens[2] == "0" {
                0
            } else {
                int_at(&report.json, tokens[2])
            };
            slot.1 = Json::Int(current.max(bound) + 1);
            let checks = emit::judge(&Json::Obj(damaged), &[], std::slice::from_ref(&rule));
            assert_eq!(checks[0].status, Status::Fail, "{rule}");
        }
        // And the same JSON undamaged passes every one.
        assert!(report.falsifiers.iter().all(|c| c.status == Status::Pass));
    }

    #[test]
    fn the_preregistration_states_every_rule_and_the_simulation() {
        for rule in metrics::coverage().iter().chain(&metrics::falsifiers()) {
            assert!(
                params::PREREGISTRATION_TEXT.contains(rule.as_str()),
                "{rule} is not in the preregistration"
            );
        }
        assert!(params::PREREGISTRATION_TEXT.contains(params::ENTRYPOINT));
        assert!(params::PREREGISTRATION_TEXT.contains("simulat"));
        for ns in model::NAMESPACES {
            assert!(
                params::PREREGISTRATION_TEXT.contains(ns.name),
                "{}",
                ns.name
            );
        }
    }

    #[test]
    fn the_entrypoint_is_empty_until_the_freeze_and_then_the_subcommand() {
        assert_eq!(params::BENCHMARK, "calibrated-arbiter");
        let manifest: toml::Table = params::EXPERIMENT_TOML
            .parse()
            .expect("experiment.toml parses");
        let entrypoint = manifest["entrypoint"].as_str().expect("entrypoint");
        assert!(entrypoint.is_empty() || entrypoint == params::ENTRYPOINT);
        assert_eq!(manifest["id"].as_str(), Some("F003"));
        let seeds: Vec<u64> = manifest["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|seed| seed.as_integer().unwrap() as u64)
            .collect();
        assert_eq!(seeds, params::DECLARED_SEEDS);
    }

    fn ints(table: &toml::Table, key: &str) -> Vec<i64> {
        table[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is an array"))
            .iter()
            .map(|v| v.as_integer().expect("an integer"))
            .collect()
    }

    fn int(table: &toml::Table, key: &str) -> i64 {
        table[key]
            .as_integer()
            .unwrap_or_else(|| panic!("{key} is an integer"))
    }

    #[test]
    fn the_compiled_parameters_are_the_configuration_files() {
        let config: toml::Table = params::CONFIG_TOML.parse().expect("config.toml parses");
        let pre = config["preregistration"]
            .as_table()
            .expect("[preregistration]");
        let pinned = config["pinned_thresholds"]
            .as_table()
            .expect("[pinned_thresholds]");
        // Every key is mirrored: the sets of keys are exactly these.
        let mut pre_keys: Vec<&str> = pre.keys().map(String::as_str).collect();
        pre_keys.sort_unstable();
        let mut expected_pre = vec![
            "seeds",
            "harness",
            "alpha_permille",
            "delta_permille",
            "threshold_rule",
            "threshold_grid_steps",
            "min_adjudications_per_policy",
            "uncalibrated_threshold_permille",
            "calibration_set_rule",
            "adjudicators_see_intent",
            "harm_breakdown_namespaces",
            "adjudication_protocol",
            "adjudication_protocol_sha256",
            "stream_branches",
            "slice_rate_permille",
            "adjudicator_error_permille",
            "policy_sizes",
            "mc_replicates",
            "coverage_test_level_ppm",
            "heldout_familywise_permille",
            "heldout_tests",
            "ope_logs",
            "ope_log_branches",
            "ope_bias_sigmas",
            "preregistration_text",
            "preregistration_text_sha256",
            "criteria",
            "criteria_sha256",
        ];
        expected_pre.sort_unstable();
        assert_eq!(pre_keys, expected_pre);
        assert_eq!(pre["harness"].as_str(), Some(params::BENCHMARK));
        assert_eq!(ints(pre, "seeds"), params::DECLARED_SEEDS.map(|s| s as i64));
        assert_eq!(
            int(pre, "alpha_permille"),
            i64::from(params::ALPHA_PERMILLE)
        );
        assert_eq!(
            int(pre, "delta_permille"),
            i64::from(params::DELTA_PERMILLE)
        );
        assert_eq!(pre["threshold_rule"].as_str(), Some("learn_then_test"));
        assert_eq!(int(pre, "threshold_grid_steps"), 1000);
        assert_eq!(int(pre, "min_adjudications_per_policy"), 59);
        assert_eq!(
            int(pre, "uncalibrated_threshold_permille"),
            i64::from(params::UNCALIBRATED_THRESHOLD_PERMILLE)
        );
        assert_eq!(
            int(pre, "stream_branches"),
            params::PINNED.stream_branches as i64
        );
        assert_eq!(
            int(pre, "slice_rate_permille"),
            i64::from(params::SLICE_RATE_PERMILLE)
        );
        assert_eq!(
            int(pre, "adjudicator_error_permille"),
            i64::from(params::EPSILON_PERMILLE)
        );
        assert_eq!(
            ints(pre, "policy_sizes"),
            params::PINNED
                .policy_sizes
                .iter()
                .map(|n| *n as i64)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            int(pre, "mc_replicates"),
            params::PINNED.mc_replicates as i64
        );
        assert_eq!(
            int(pre, "coverage_test_level_ppm"),
            i64::from(params::COVERAGE_TEST_LEVEL_PPM)
        );
        assert_eq!(
            int(pre, "heldout_familywise_permille"),
            i64::from(params::HELDOUT_FAMILYWISE_PERMILLE)
        );
        assert_eq!(int(pre, "heldout_tests"), i64::from(params::HELDOUT_TESTS));
        assert_eq!(int(pre, "ope_logs"), params::PINNED.ope_logs as i64);
        assert_eq!(
            int(pre, "ope_log_branches"),
            params::PINNED.ope_log_branches as i64
        );
        assert_eq!(int(pre, "ope_bias_sigmas") as f64, params::OPE_BIAS_SIGMAS);
        // The Bonferroni split is policies x declared seeds.
        assert_eq!(
            params::HELDOUT_TESTS as usize,
            params::POLICY_SIZES.len() * params::DECLARED_SEEDS.len()
        );
        // The harness-only numerics.
        let mut pinned_keys: Vec<&str> = pinned.keys().map(String::as_str).collect();
        pinned_keys.sort_unstable();
        let mut expected_pinned = vec![
            "seed_aggregation",
            "eligible_permille",
            "kind_weights_permille",
            "namespace_weights_permille",
            "namespace_score_maxes",
            "tied_permille",
            "slice_z_limit_milli",
            "crc_every",
            "determinism_replicates",
            "determinism_logs",
            "ope_logging_threshold_permille",
            "ope_target_thresholds_permille",
            "ope_target_slice_rates_permille",
            "ope_unsupported_threshold_permille",
            "reward_safe_milli",
            "reward_harm_milli",
            "drift_harm_factor",
            "drift_samples",
            "control_fallback_threshold_permille",
            "noisy_epsilon_permille",
        ];
        expected_pinned.sort_unstable();
        assert_eq!(pinned_keys, expected_pinned);
        assert_eq!(
            pinned["seed_aggregation"].as_str(),
            Some(params::SEED_AGGREGATION)
        );
        assert_eq!(
            int(pinned, "eligible_permille"),
            i64::from(params::ELIGIBLE_PERMILLE)
        );
        assert_eq!(
            ints(pinned, "kind_weights_permille"),
            params::KIND_WEIGHTS_PERMILLE.map(i64::from)
        );
        assert_eq!(
            ints(pinned, "namespace_weights_permille"),
            model::NAMESPACES.map(|ns| i64::from(ns.weight_permille))
        );
        assert_eq!(
            ints(pinned, "namespace_score_maxes"),
            model::NAMESPACES.map(|ns| i64::from(ns.maxes))
        );
        assert_eq!(
            int(pinned, "tied_permille"),
            i64::from(params::TIED_PERMILLE)
        );
        assert_eq!(
            int(pinned, "slice_z_limit_milli"),
            params::SLICE_Z_LIMIT_MILLI
        );
        assert_eq!(int(pinned, "crc_every"), params::CRC_EVERY as i64);
        assert_eq!(
            int(pinned, "determinism_replicates"),
            params::DETERMINISM_REPLICATES as i64
        );
        assert_eq!(
            int(pinned, "determinism_logs"),
            params::DETERMINISM_LOGS as i64
        );
        assert_eq!(
            int(pinned, "ope_logging_threshold_permille"),
            i64::from(params::OPE_LOGGING_THRESHOLD_PERMILLE)
        );
        assert_eq!(
            ints(pinned, "ope_target_thresholds_permille"),
            ope::TARGETS.map(|t| i64::from(t.threshold_permille))
        );
        assert_eq!(
            ints(pinned, "ope_target_slice_rates_permille"),
            ope::TARGETS.map(|t| i64::from(t.rate_permille))
        );
        assert_eq!(
            int(pinned, "ope_unsupported_threshold_permille"),
            i64::from(ope::UNSUPPORTED.threshold_permille)
        );
        assert_eq!(
            int(pinned, "reward_safe_milli") as f64 / 1000.0,
            params::REWARD_SAFE
        );
        assert_eq!(
            int(pinned, "reward_harm_milli") as f64 / 1000.0,
            params::REWARD_HARM
        );
        assert_eq!(int(pinned, "drift_harm_factor"), params::DRIFT_HARM_FACTOR);
        assert_eq!(int(pinned, "drift_samples"), params::DRIFT_SAMPLES as i64);
        assert_eq!(
            int(pinned, "noisy_epsilon_permille"),
            i64::from(params::NOISY_EPSILON_PERMILLE)
        );
        assert_eq!(
            int(pinned, "control_fallback_threshold_permille"),
            i64::from(params::CONTROL_FALLBACK_THRESHOLD_PERMILLE)
        );
    }

    #[test]
    fn the_seed_is_one_argument_or_the_mutation_checkers_pair() {
        let args = |rest: &[&str]| -> Vec<String> {
            ["ptr-bench", "calibrated-arbiter"]
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
    fn a_reduced_run_is_clean_where_the_scale_does_not_matter() {
        let report = reduced_report();
        assert!(!report.line.contains('\n'));
        assert!(report.line.contains("\"harness_gaps\":[]"));
        assert_eq!(report.falsifiers_tripped(), 0, "{:?}", report.line);
        assert!(report.gaps.is_empty());
        let unmet: Vec<&str> = report
            .coverage
            .iter()
            .filter(|check| check.status == Status::Fail)
            .map(|check| check.rule.as_str())
            .collect();
        // Only the rules written for the pinned sizes are unmet at the reduced scale.
        for rule in &unmet {
            let name = rule.split(' ').next().unwrap();
            assert!(
                [
                    "stream_branches",
                    "eligible_branches",
                    "adjudications",
                    "policies_recorded",
                    "nonslice_probes",
                    "mc_replicates",
                    "mc_certified_replicates",
                    "ope_logs",
                    "ope_estimates",
                    "positivity_probes",
                    "heldout_candidates_min_certified",
                ]
                .contains(&name),
                "{rule} is unmet in a reduced run: {}",
                report.line
            );
        }
        let j = &report.json;
        assert_eq!(int_at(j, "hard_failures"), 0);
        assert_eq!(int_at(j, "negative_controls_tripped"), 4);
        assert_eq!(int_at(j, "ineligible_kinds_seen"), 6);
        assert_eq!(
            int_at(j, "stream_branches"),
            params::REDUCED.stream_branches as i64
        );
        assert_eq!(
            int_at(j, "policies_recorded"),
            params::REDUCED.policy_sizes.len() as i64
        );
        assert!(int_at(j, "policies_certified") >= 1);
        assert_eq!(
            int_at(j, "mc_replicates"),
            params::REDUCED.mc_replicates as i64
        );
        assert_eq!(int_at(j, "ope_logs"), params::REDUCED.ope_logs as i64);
        assert_eq!(
            int_at(j, "ope_estimates"),
            (params::REDUCED.ope_logs * 8) as i64
        );
        assert_eq!(
            int_at(j, "positivity_probes"),
            (params::REDUCED.ope_logs * 2) as i64
        );
        assert_eq!(int_at(j, "slice_rate_tests"), 6);
        assert!(int_at(j, "ltt_coverage_critical") >= int_at(j, "ltt_coverage_violations"));
        assert!(int_at(j, "uncalibrated_true_harm_ppm") > 50_000);
        assert_eq!(int_at(j, "uncalibrated_baseline_rejected"), 1);
        assert_eq!(
            int_at(j, "procedure.mc_replicates"),
            params::REDUCED.mc_replicates as i64
        );
        assert_eq!(emit::resolve(j, "scale"), Some(&Json::str("reduced")));
        assert_eq!(
            emit::resolve(j, "procedure.shape_is_pinned"),
            Some(&Json::Bool(false))
        );
        assert_eq!(
            emit::resolve(j, "procedure.seed_is_declared"),
            Some(&Json::Bool(true))
        );
    }

    #[test]
    fn a_reduced_run_is_reproducible_apart_from_its_timings() {
        let again = execute(17, Scale::Reduced).expect("a reduced run");
        let strip = |json: &Json| -> String {
            let Json::Obj(pairs) = json else {
                panic!("an object");
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
        assert_eq!(strip(&reduced_report().json), strip(&again.json));
        // Another seed is another stream.
        let other = execute(29, Scale::Reduced).expect("a reduced run");
        assert_ne!(strip(&reduced_report().json), strip(&other.json));
    }

    #[test]
    fn the_stream_is_deterministic_and_has_the_declared_mix() {
        let shape = params::REDUCED;
        let a = stream::generate(5, &shape);
        let b = stream::generate(5, &shape);
        assert_eq!(a.len(), shape.stream_branches);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!((&x.id, x.m, x.harm, x.label), (&y.id, y.m, y.harm, y.label));
        }
        let eligible = a.iter().filter(|d| d.kind == model::Kind::Eligible).count();
        let share = eligible as f64 / a.len() as f64;
        assert!((0.72..0.78).contains(&share), "{share}");
        // Ineligible branches carry adversarially high scores; ties sit on the grid.
        assert!(a
            .iter()
            .filter(|d| d.kind != model::Kind::Eligible)
            .all(|d| d.m >= 900_000));
        let tied = a
            .iter()
            .filter(|d| d.kind == model::Kind::Eligible && d.m % 1000 == 0)
            .count();
        assert!(tied as f64 / eligible as f64 > 0.25);
        // Every branch changes at least one key with a namespace.
        assert!(a.iter().all(|d| !d.keys.is_empty()
            && oracle::namespace_of(&d.keys[0]) == model::NAMESPACES[d.ns].name));
    }

    #[test]
    fn the_exact_truth_matches_the_generator() {
        // The oracle's exact harm rates are sums over the model's tables; the generator draws
        // from the same tables with its own code, so the two must agree within sampling error.
        let truth = model::Truth::new();
        let mut rng = Rng::new(99);
        let samples: Vec<(u32, bool)> = (0..400_000)
            .map(|_| {
                let ns = model::draw_namespace(&mut rng);
                let m = model::draw_score(ns, &mut rng);
                (m, model::draw_harm(ns, m, 1, &mut rng))
            })
            .collect();
        for j in [0_usize, 400, 500, 700, 850] {
            let admitted: Vec<&(u32, bool)> = samples
                .iter()
                .filter(|(m, _)| model::admits(j, *m))
                .collect();
            let n = admitted.len() as f64;
            let rate = admitted.iter().filter(|(_, h)| *h).count() as f64 / n;
            let mass = n / samples.len() as f64;
            let exact = truth.harm_above(j);
            let se = (exact * (1.0 - exact) / n).sqrt().max(1e-4);
            assert!((rate - exact).abs() < 4.5 * se, "j {j}: {rate} vs {exact}");
            assert!((mass - truth.mass_above(j)).abs() < 0.004, "j {j} mass");
        }
    }

    #[test]
    fn off_policy_estimates_are_unbiased_over_many_logs() {
        // Far more logs than a run uses: the mean error of IPS and doubly robust estimates of
        // the exact value is within four standard errors, and the unweighted mean is not.
        let truth = model::Truth::new();
        let shape = Shape {
            ope_logs: 400,
            ope_log_branches: 1_000,
            ..params::REDUCED
        };
        let out = ope::run(
            3,
            &shape,
            shape.ope_logs,
            &truth,
            &mut Tally::new(metrics::LIST),
        )
        .expect("logs");
        for (ti, target) in ope::TARGETS.iter().enumerate() {
            for (ei, name) in ope::ESTIMATORS.iter().enumerate() {
                let (mean, se, trips) = ope::bias_test(&out.errors[ti][ei]);
                if *name == "unweighted_control" {
                    assert!(trips, "{} {name}", target.name);
                } else if *name != "snips" {
                    assert!(!trips, "{} {name}: {mean} +- {se}", target.name);
                }
            }
        }
    }

    #[test]
    fn the_drift_control_rejects_and_the_model_itself_does_not() {
        let j = 700;
        let (p, rejected) = drift_control(1, j);
        assert!(rejected && p < 1e-6, "{p}");
        // The same procedure at harm factor one, at a threshold the model keeps clean, is not
        // rejected: the control trips because of the drift, not because the test always rejects.
        let mut rng = Rng::new(7);
        let (mut n, mut k) = (0, 0);
        while n < params::DRIFT_SAMPLES {
            let ns = model::draw_namespace(&mut rng);
            let m = model::draw_score(ns, &mut rng);
            if !model::admits(900, m) {
                continue;
            }
            n += 1;
            k += usize::from(model::draw_harm(ns, m, 1, &mut rng));
        }
        assert!(!heldout::rejects(k, n).1, "{k}/{n}");
    }

    #[test]
    fn the_coverage_critical_value_is_pinned_to_the_exact_binomial() {
        // P(Bin(2000, 0.05) >= 131) = 1.307e-3 >= 1e-3 > P(>= 132) = 0.955e-3.
        assert_eq!(
            stats::coverage_critical(2000, params::delta(), params::coverage_level()),
            131
        );
    }

    #[test]
    fn an_empty_slice_group_is_a_violation_and_is_no_test() {
        let (tests, violations, _) = stream::slice_rate_tests(&[(1000, 150), (0, 0)]);
        assert_eq!((tests, violations), (1, 1));
        let (tests, violations, _) = stream::slice_rate_tests(&[(1000, 150), (1000, 800)]);
        assert_eq!((tests, violations), (2, 1));
    }

    #[test]
    fn the_calibration_draw_is_reproducible_and_separated_by_seed() {
        let branch = ptr_branch::BranchId::from("f003-17-000001");
        let a = ptr_branch::calibration_draw(&branch, 17);
        assert_eq!(a, ptr_branch::calibration_draw(&branch, 17));
        assert_ne!(a, ptr_branch::calibration_draw(&branch, 29));
        let other = ptr_branch::BranchId::from("f003-17-000002");
        assert_ne!(a, ptr_branch::calibration_draw(&other, 17));
    }

    #[test]
    fn the_noisy_arm_certifies_far_less_than_the_preregistered_regime() {
        let report = reduced_report();
        let j = &report.json;
        let noisy = |key: &str| int_at(j, &format!("descriptive.noisy_adjudicator_arm.{key}"));
        assert_eq!(noisy("replicates"), params::REDUCED.mc_replicates as i64);
        assert!(
            noisy("certified") * 2 < int_at(j, "mc_certified_replicates"),
            "{}",
            report.line
        );
        assert_eq!(noisy("certified") + noisy("never"), noisy("replicates"));
    }

    #[test]
    fn a_product_that_adjudicates_everything_is_seen_by_the_probe() {
        // The probe of the pipeline: a non-slice triage offered to `adjudicate` must be refused.
        let report = model::Kind::Eligible.report(0);
        let policy = TriagePolicy::new(AutoThreshold::AtLeast(0.5), 0.15).unwrap();
        let prob = ptr_types::Probability::new(0.9).unwrap();
        let outside = policy.triage(&report, prob, 0.9).unwrap();
        assert!(outside.adjudicate(true).is_err());
        let inside = policy.triage(&report, prob, 0.01).unwrap();
        assert!(inside.adjudicate(true).is_ok());
    }
}
