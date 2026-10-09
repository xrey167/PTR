//! The runs of F001: every task of every profile through the single-shot, rewrite-only and repair
//! arms at every budget, each run judged against the independent oracle and the call-counting
//! boundary, and the measured cells compared with the exact analytic expectation.
//!
//! The product under test is `ptr_feedback::RepairLoop` (and `Population::best_verified`) over
//! the real `VerifierFabric` and `ActionContractVerifier`. A panic of the product is caught and
//! counted as `product_panics`. A disagreement of the harness with itself (a stub whose labelled
//! defects differ from the oracle's, a stub whose empirical frequency leaves its stated
//! probability) is an `Err`: the harness cannot run.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use ptr_feedback::{
    Candidate, Critique, Decision, ExhaustionReason, Generator, Origin, Outcome, Population,
    RepairBudget, RepairLoop, RepairPolicy, RepairRun, Repairer, RewriteReason, Step, Trajectory,
    Usage,
};
use ptr_types::{ActionIr, VerificationLevel};
use ptr_verifier::{VerificationReport, VerificationStatus, Verifier};

use super::dp::{ArmKind, Model};
use super::oracle::{self, acceptable, bit, defect_mask, expected, level_rank, Mode, ALL_SLOTS};
use super::params::{
    BUDGETS, EXPECTATION_ALPHA_PER_BILLION, EXPECTATION_SIGMA, PROFILES, REPAIR_Q_PERMILLE,
    SELFTEST_SAMPLES, SLOTS, SLOT_CODES,
};
use super::stats::{bernstein_p_value, binomial_p_value};
use super::stub::{
    stream, Boundary, Kind, NeverRepairs, Noise, Produced, Reads, Serial, SharedLog, StubGenerator,
    StubLog, StubRepairer,
};
use crate::experiments::layered_kv::emit::{MetricList, Tally};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Arm {
    Single,
    Rewrite,
    Repair,
}

impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Rewrite => "rewrite",
            Self::Repair => "repair",
        }
    }
}

/// What a pass runs: the product, or one of the two negative controls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Variant {
    Mechanism,
    /// The harness' reference loop with its acceptance rule replaced by `score >= 0.5`.
    ScoreAccept,
    /// The harness' reference loop (correct acceptance rule) with a repairer that ignores the
    /// critique and redraws every slot.
    BlindRepair,
}

/// Measured totals of one (profile, arm, budget) cell.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellStats {
    pub tasks: i64,
    pub successes: i64,
    pub calls: i64,
    pub draws: i64,
}

/// One cell compared with its analytic expectation.
#[derive(Clone, Debug)]
pub struct Comparison {
    pub profile: &'static str,
    pub arm: Arm,
    pub budget: u32,
    pub tasks: i64,
    pub successes: i64,
    pub expected_successes_milli: i64,
    pub calls: i64,
    pub expected_calls_milli: i64,
    pub draws: i64,
    pub expected_draws_milli: i64,
    /// The largest |z| of the three comparisons, in thousandths.
    pub worst_z_milli: i64,
    pub violated: bool,
}

pub struct Pass {
    pub tally: Tally,
    pub cells: BTreeMap<(usize, Arm, u32), CellStats>,
    pub profile_tasks: Vec<i64>,
    pub comparisons: Vec<Comparison>,
}

/// The facts of one finished run the aggregation needs.
struct Facts {
    success: bool,
    calls: i64,
    draws: i64,
    exhausted: bool,
    first_call_accepted: bool,
    repaired_success: bool,
}

fn is_repaired(stub: &StubLog, serial: u64) -> bool {
    stub.produced
        .iter()
        .any(|(s, how, _)| *s == serial && *how == Produced::Repaired)
}

/// The harness-owned reference loop of the two negative controls. It has the product loop's
/// shape but is the harness' own code (it builds its critiques itself), so the controls do not
/// depend on the code under test; `accept` is its acceptance rule.
fn reference_run(
    verifier: &dyn Verifier<ActionIr>,
    budget: u32,
    accept: fn(&VerificationReport) -> bool,
    generator: &mut dyn Generator<ActionIr>,
    repairer: &mut dyn Repairer<ActionIr>,
) -> RepairRun<ActionIr> {
    let mut usage = Usage::default();
    let mut steps: Vec<Step<ActionIr>> = Vec::new();
    let mut value = generator.generate();
    usage.generated += 1;
    let mut origin = Origin::Initial;
    loop {
        let id = ptr_types::CandidateId::from(format!("c{}", usage.verifier_calls).as_str());
        let report = verifier.verify(&value);
        usage.verifier_calls += 1;
        let accepted = accept(&report);
        let issues = named_by(&report);
        let repairable = report.status == VerificationStatus::Fail && !issues.is_empty();
        let critique = Critique {
            candidate: id.clone(),
            issues,
            repairable,
        };
        let candidate = Candidate {
            id,
            value,
            verification: Some(report),
        };
        if accepted {
            steps.push(Step {
                candidate: candidate.clone(),
                origin,
                critique: None,
                decision: Decision::Accepted,
            });
            return RepairRun {
                outcome: Outcome::Verified(candidate),
                trajectory: Trajectory { steps },
                usage,
            };
        }
        if usage.verifier_calls >= budget {
            let reason = ExhaustionReason::VerifierCalls;
            steps.push(Step {
                candidate,
                origin,
                critique: Some(critique),
                decision: Decision::Stop(reason),
            });
            return RepairRun {
                outcome: Outcome::Exhausted { reason },
                trajectory: Trajectory { steps },
                usage,
            };
        }
        let repaired = if critique.repairable {
            repairer.repair(&candidate, &critique)
        } else {
            None
        };
        let from = candidate.id.clone();
        let (next, next_origin, decision) = match repaired {
            Some(next) => {
                usage.repaired += 1;
                (next, Origin::Repaired { from }, Decision::Repair)
            }
            None => {
                let reason = if critique.repairable {
                    usage.declined += 1;
                    RewriteReason::Declined
                } else {
                    RewriteReason::NotRepairable
                };
                usage.generated += 1;
                (
                    generator.generate(),
                    Origin::Rewritten { from, reason },
                    Decision::Rewrite(reason),
                )
            }
        };
        usage.rounds += 1;
        steps.push(Step {
            candidate,
            origin,
            critique: Some(critique),
            decision,
        });
        value = next;
        origin = next_origin;
    }
}

/// The hard-finding codes of `report`, each once, in report order: what a critique must name.
fn named_by(report: &VerificationReport) -> Vec<String> {
    let mut issues: Vec<String> = Vec::new();
    for code in oracle::hard_codes(report) {
        if !issues.contains(&code) {
            issues.push(code);
        }
    }
    issues
}

/// Judge one finished run against the boundary's log and the stubs' log. Counters go to
/// `tally`; an `Err` is a harness fault.
fn check_run(
    tally: &mut Tally,
    run: &RepairRun<ActionIr>,
    boundary: &Boundary,
    stub: &StubLog,
    budget: u32,
) -> Result<Facts, String> {
    let log = boundary.log.borrow();
    let calls = log.len() as i64;

    // The stubs' labelled defects against the oracle's reading, checked where the stub produced
    // the action (before the loop saw it): a disagreement is a fault of the harness.
    if stub.label_faults > 0 {
        return Err(format!(
            "{} candidates whose labelled defects differ from the oracle's reading at the stub's output",
            stub.label_faults
        ));
    }
    // What reached the verifier against what the stubs handed out: the loop must verify the
    // candidate it was given, unaltered, and no other.
    for entry in log.iter() {
        let handed = stub
            .actions
            .iter()
            .find(|(serial, _)| *serial == entry.serial)
            .map(|(_, action)| action);
        if handed != Some(&entry.action) {
            tally.add("altered_candidates", 1);
        }
    }

    // Every verdict against the fabric's required report.
    for entry in log.iter() {
        let want = expected(entry.mask, entry.mode);
        if !oracle::matches_expected(&entry.report, &want) {
            tally.add("verdict_mismatches", 1);
        }
        match entry.report.status {
            VerificationStatus::Pass => tally.add("status_pass", 1),
            VerificationStatus::Fail => tally.add("status_fail", 1),
            VerificationStatus::Unknown => tally.add("status_unknown", 1),
            VerificationStatus::Disputed => tally.add("status_disputed", 1),
        }
        let refused = !acceptable(&entry.report);
        if refused && (entry.mode == Mode::Lying || entry.forged) {
            tally.add("lying_refused", 1);
        }
        if refused
            && entry.report.status == VerificationStatus::Pass
            && level_rank(entry.report.level) < level_rank(VerificationLevel::Deterministic)
        {
            tally.add("shallow_refused", 1);
        }
    }

    // The loop's own claims against the counts of the boundary and the stubs.
    let usage = run.usage;
    let claims = [
        (i64::from(usage.verifier_calls), calls),
        (i64::from(usage.generated), i64::from(stub.generated)),
        (i64::from(usage.repaired), i64::from(stub.repairs_returned)),
        (i64::from(usage.declined), i64::from(stub.repairs_declined)),
        (
            i64::from(usage.rounds),
            i64::from(stub.generated) + i64::from(stub.repairs_returned) - 1,
        ),
    ];
    for (claimed, counted) in claims {
        if claimed != counted {
            tally.add("verifier_call_mismatches", 1);
        }
    }

    let mut serials: Vec<u64> = log.iter().map(|entry| entry.serial).collect();
    serials.sort_unstable();
    let distinct = {
        let mut unique = serials.clone();
        unique.dedup();
        unique.len()
    };
    tally.add("double_verifications", (serials.len() - distinct) as i64);

    if calls > i64::from(budget) || i64::from(usage.rounds) > i64::from(budget) {
        tally.add("budget_violations", 1);
    }

    let first_acceptable = log.iter().position(|entry| acceptable(&entry.report));
    if let Some(position) = first_acceptable {
        tally.add("calls_after_acceptance", calls - position as i64 - 1);
    }

    let mut success = false;
    let mut exhausted = false;
    let mut repaired_success = false;
    match &run.outcome {
        Outcome::Verified(candidate) => {
            let serial = candidate.value.revision.0;
            let logged = log.iter().find(|entry| entry.serial == serial);
            let report_ok = candidate.verification.as_ref().is_some_and(acceptable);
            let clean = defect_mask(&candidate.value) == 0;
            if !report_ok || !clean {
                tally.add("false_accepts", 1);
            }
            match logged {
                Some(entry) if candidate.verification.as_ref() == Some(&entry.report) => {}
                _ => tally.add("unverified_returns", 1),
            }
            if first_acceptable.is_none() {
                tally.add("exhausted_as_success", 1);
            }
            success = report_ok && clean;
            repaired_success = success && is_repaired(stub, serial);
        }
        Outcome::Exhausted { reason } => {
            exhausted = true;
            tally.add("exhausted_runs", 1);
            if first_acceptable.is_some() {
                tally.add("missed_acceptances", 1);
            }
            if calls < i64::from(budget) || *reason == ExhaustionReason::Rounds {
                tally.add("premature_exhaustions", 1);
            }
        }
    }

    // The trajectory against the log.
    let steps = &run.trajectory.steps;
    let mut mismatched = steps.len() != log.len();
    for (index, step) in steps.iter().enumerate() {
        let Some(entry) = log.get(index) else {
            break;
        };
        let last = index + 1 == steps.len();
        mismatched |= step.candidate.verification.as_ref() != Some(&entry.report);
        mismatched |= step.candidate.value.revision.0 != entry.serial;
        mismatched |= step.candidate.id.to_string() != format!("c{index}");
        mismatched |= match &step.critique {
            None => step.decision != Decision::Accepted,
            Some(critique) => {
                let issues = named_by(&entry.report);
                let repairable =
                    entry.report.status == VerificationStatus::Fail && !issues.is_empty();
                critique.issues != issues
                    || critique.repairable != repairable
                    || step.decision == Decision::Accepted
            }
        };
        mismatched |= !matches!(
            (&step.decision, last, &run.outcome),
            (Decision::Accepted, true, Outcome::Verified(_))
                | (Decision::Stop(_), true, Outcome::Exhausted { .. })
                | (Decision::Repair | Decision::Rewrite(_), false, _)
        );
        mismatched |= match (index, &step.origin) {
            (0, Origin::Initial) => false,
            (0, _) | (_, Origin::Initial) => true,
            (_, Origin::Repaired { from } | Origin::Rewritten { from, .. }) => {
                from.to_string() != format!("c{}", index - 1)
            }
        };
        if index > 0 {
            let before = &steps[index - 1];
            mismatched |= match (&before.decision, &step.origin) {
                (Decision::Repair, Origin::Repaired { .. }) => false,
                (Decision::Rewrite(reason), Origin::Rewritten { reason: got, .. }) => reason != got,
                _ => true,
            };
        }
    }
    if mismatched {
        tally.add("trajectory_mismatches", 1);
    }

    // Repairs touch only what the report names.
    for (before, after) in &stub.edits {
        let was = defect_mask(before);
        let now = defect_mask(after);
        let broken = now & !was & ALL_SLOTS;
        tally.add("collateral_edits", i64::from(broken.count_ones()));
        let payload_slots = bit(5) | bit(6);
        if was & payload_slots == 0 && before.payload != after.payload {
            tally.add("collateral_edits", 1);
        }
    }

    // Selection over the whole trajectory agrees with the loop.
    let mut population: Population<ActionIr> = Population::default();
    for step in steps {
        population.push(step.candidate.clone());
    }
    let best = population.best_verified(VerificationLevel::Deterministic);
    let agrees = match (&run.outcome, best) {
        (Outcome::Verified(candidate), Some(best)) => best.id == candidate.id,
        (Outcome::Exhausted { .. }, None) => true,
        _ => false,
    };
    if !agrees {
        tally.add("selection_mismatches", 1);
    }

    Ok(Facts {
        success,
        calls,
        draws: stub.slot_draws as i64,
        exhausted,
        first_call_accepted: first_acceptable == Some(0),
        repaired_success,
    })
}

/// Run one task through one arm at one budget.
fn run_one(
    seed: u64,
    profile: usize,
    task: usize,
    arm: Arm,
    budget: u32,
    variant: Variant,
    tally: &mut Tally,
) -> Result<Facts, String> {
    let noise = Noise::pinned();
    let p = PROFILES[profile].1;
    let serial: Serial = Rc::default();
    let log: SharedLog = Rc::default();
    let boundary = Boundary::new(seed, profile, task, noise);
    let mut generator = StubGenerator {
        seed,
        profile,
        task,
        p_permille: p,
        serial: Rc::clone(&serial),
        log: Rc::clone(&log),
    };
    let mut never = NeverRepairs {
        log: Rc::clone(&log),
    };
    let mut repairing = StubRepairer {
        seed,
        profile,
        task,
        q_permille: REPAIR_Q_PERMILLE,
        reads: if variant == Variant::BlindRepair {
            Reads::Everything
        } else {
            Reads::Critique
        },
        serial: Rc::clone(&serial),
        log: Rc::clone(&log),
    };
    let repairer: &mut dyn Repairer<ActionIr> = match arm {
        Arm::Single | Arm::Rewrite => &mut never,
        Arm::Repair => &mut repairing,
    };
    let policy = RepairPolicy {
        required_level: VerificationLevel::Deterministic,
        budget: RepairBudget {
            max_verifier_calls: budget,
            max_rounds: budget,
        },
    };
    let ran = catch_unwind(AssertUnwindSafe(|| match variant {
        Variant::Mechanism => RepairLoop::new(&boundary, policy)
            .expect("a positive budget")
            .run(&mut generator, repairer),
        Variant::ScoreAccept => reference_run(
            &boundary,
            budget,
            |r| r.score.get() >= 0.5,
            &mut generator,
            repairer,
        ),
        Variant::BlindRepair => {
            reference_run(&boundary, budget, acceptable, &mut generator, repairer)
        }
    }));
    tally.add("runs", 1);
    let Ok(run) = ran else {
        tally.add("product_panics", 1);
        let stub = log.borrow();
        return Ok(Facts {
            success: false,
            calls: boundary.log.borrow().len() as i64,
            draws: stub.slot_draws as i64,
            exhausted: false,
            first_call_accepted: false,
            repaired_success: false,
        });
    };
    let stub = log.borrow();
    check_run(tally, &run, &boundary, &stub, budget)
}

fn z_score(observed: f64, mean: f64, variance: f64) -> f64 {
    let spread = variance.sqrt();
    if spread < 1e-12 {
        if (observed - mean).abs() < 1e-6 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (observed - mean) / spread
    }
}

/// The two-sided tail probability, in nine decimal digits per one, that the pinned sigma cuts
/// off: the exact significance a measured cell must stay above.
fn alpha() -> f64 {
    EXPECTATION_ALPHA_PER_BILLION as f64 / 1e9
}

/// Compare every measured (profile, arm, budget) cell with the exact analytic expectation. The
/// success count is tested with the exact binomial tail, the verifier calls and the slot draws
/// (sums of bounded per-task costs with exactly known mean and variance) with the Bernstein
/// bound; a cell is violated when any p-value is below the pinned significance. The z-scores
/// are descriptive.
fn compare(cells: &BTreeMap<(usize, Arm, u32), CellStats>, tally: &mut Tally) -> Vec<Comparison> {
    let mut rows = Vec::new();
    for (&(profile, arm, budget), stats) in cells {
        let (name, p) = PROFILES[profile];
        let kind = match arm {
            Arm::Single | Arm::Rewrite => ArmKind::Rewrite,
            Arm::Repair => ArmKind::Repair,
        };
        let cell = Model::new(&p, REPAIR_Q_PERMILLE, &Noise::pinned()).cell(kind, budget);
        let tasks = stats.tasks as usize;
        let n = stats.tasks as f64;
        // The failures are the rare event in a saturated cell: test them, not the successes.
        let failure_p = binomial_p_value(
            tasks,
            tasks - stats.successes as usize,
            (1.0 - cell.success).max(0.0),
        );
        let calls_p = bernstein_p_value(
            tasks,
            stats.calls as f64 - n * cell.calls.mean,
            cell.calls.variance,
            f64::from(budget),
        );
        let draws_p = bernstein_p_value(
            tasks,
            stats.draws as f64 - n * cell.draws.mean,
            cell.draws.variance,
            (SLOTS as f64) * f64::from(budget),
        );
        let worst = [
            z_score(
                stats.successes as f64,
                n * cell.success,
                n * cell.success * (1.0 - cell.success),
            ),
            z_score(
                stats.calls as f64,
                n * cell.calls.mean,
                n * cell.calls.variance,
            ),
            z_score(
                stats.draws as f64,
                n * cell.draws.mean,
                n * cell.draws.variance,
            ),
        ]
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
        let violated = failure_p < alpha() || calls_p < alpha() || draws_p < alpha();
        if violated {
            tally.add("expectation_violations", 1);
        }
        rows.push(Comparison {
            profile: name,
            arm,
            budget,
            tasks: stats.tasks,
            successes: stats.successes,
            expected_successes_milli: (n * cell.success * 1000.0).round() as i64,
            calls: stats.calls,
            expected_calls_milli: (n * cell.calls.mean * 1000.0).round() as i64,
            draws: stats.draws,
            expected_draws_milli: (n * cell.draws.mean * 1000.0).round() as i64,
            worst_z_milli: if worst.is_finite() {
                (worst * 1000.0).round() as i64
            } else {
                i64::MAX
            },
            violated,
        });
    }
    rows
}

/// Every task of every profile through every arm and budget.
pub fn run_pass(
    seed: u64,
    tasks_per_profile: usize,
    variant: Variant,
    list: MetricList,
) -> Result<Pass, String> {
    let mut tally = Tally::new(list);
    let mut cells: BTreeMap<(usize, Arm, u32), CellStats> = BTreeMap::new();
    let mut profile_tasks = vec![0i64; PROFILES.len()];
    let longest = *BUDGETS.last().expect("budgets");
    for (profile, tasks_of_profile) in profile_tasks.iter_mut().enumerate() {
        for task in 0..tasks_per_profile {
            let mut arms: Vec<(Arm, u32)> = vec![(Arm::Single, 1)];
            for budget in BUDGETS {
                arms.push((Arm::Rewrite, budget));
                arms.push((Arm::Repair, budget));
            }
            for (arm, budget) in arms {
                let facts = run_one(seed, profile, task, arm, budget, variant, &mut tally)?;
                let stats = cells.entry((profile, arm, budget)).or_default();
                stats.tasks += 1;
                stats.successes += i64::from(facts.success);
                stats.calls += facts.calls;
                stats.draws += facts.draws;
                if arm == Arm::Repair && budget == longest {
                    tally.add("tasks", 1);
                    *tasks_of_profile += 1;
                    if !facts.first_call_accepted {
                        tally.add("initial_fail_tasks", 1);
                    }
                    if facts.repaired_success {
                        tally.add("repaired_successes", 1);
                    }
                }
                let _ = facts.exhausted;
            }
        }
    }
    let comparisons = compare(&cells, &mut tally);
    Ok(Pass {
        tally,
        cells,
        profile_tasks,
        comparisons,
    })
}

/// The harness' check of its own stubs: the empirical frequencies of the generator, the
/// repairer and the verifier noise against their stated probabilities, each within the
/// expectation's sigma. A failure is a fault of the harness.
pub fn selftest(seed: u64) -> Result<(), String> {
    let sigma = f64::from(EXPECTATION_SIGMA);
    let within = |what: String, hits: u64, n: usize, p: f64| -> Result<(), String> {
        let z = z_score(hits as f64, n as f64 * p, n as f64 * p * (1.0 - p));
        if z.abs() > sigma {
            Err(format!(
                "self-test: {what}: {hits} of {n} against probability {p} (z = {z:.2})"
            ))
        } else {
            Ok(())
        }
    };
    for (profile, (name, p)) in PROFILES.iter().enumerate() {
        let generator = StubGenerator {
            seed,
            profile,
            task: 0,
            p_permille: *p,
            serial: Rc::default(),
            log: Rc::default(),
        };
        let mut right = [0u64; SLOTS];
        for i in 0..SELFTEST_SAMPLES {
            let mut rng = stream(seed, profile, i, Kind::Selftest, 0);
            let (action, _) = generator.draw(&mut rng, 0);
            let mask = defect_mask(&action);
            for (slot, count) in right.iter_mut().enumerate() {
                *count += u64::from(mask & bit(slot) == 0);
            }
        }
        for slot in 0..SLOTS {
            within(
                format!("generator {name} slot {}", SLOT_CODES[slot]),
                right[slot],
                SELFTEST_SAMPLES,
                f64::from(p[slot]) / 1000.0,
            )?;
        }
    }
    // The repairer fixes a named slot with probability q.
    let mut fixed = [0u64; SLOTS];
    let log: SharedLog = Rc::default();
    for i in 0..SELFTEST_SAMPLES {
        let mut repairer = StubRepairer {
            seed,
            profile: 0,
            task: i,
            q_permille: REPAIR_Q_PERMILLE,
            reads: Reads::Critique,
            serial: Rc::default(),
            log: Rc::clone(&log),
        };
        let generator = StubGenerator {
            seed,
            profile: 0,
            task: i,
            p_permille: [1; SLOTS],
            serial: Rc::default(),
            log: Rc::default(),
        };
        let mut rng = stream(seed, 0, i, Kind::Selftest, 1);
        let (broken, _) = generator.draw(&mut rng, 0);
        let candidate = Candidate {
            id: ptr_types::CandidateId::from("c0"),
            value: broken,
            verification: None,
        };
        let critique = Critique {
            candidate: candidate.id.clone(),
            issues: SLOT_CODES.iter().map(|code| code.to_string()).collect(),
            repairable: true,
        };
        if defect_mask(&candidate.value) != ALL_SLOTS {
            // With p = 0.001 a slot is rarely right: skip the rare draw instead of mislabelling.
            continue;
        }
        let repaired = repairer
            .repair(&candidate, &critique)
            .ok_or("self-test: the repairer declined")?;
        let mask = defect_mask(&repaired);
        for (slot, count) in fixed.iter_mut().enumerate() {
            *count += u64::from(mask & bit(slot) == 0);
        }
    }
    let attempts = log.borrow().repairs_returned as usize;
    for slot in 0..SLOTS {
        within(
            format!("repairer slot {}", SLOT_CODES[slot]),
            fixed[slot],
            attempts,
            REPAIR_Q_PERMILLE as f64 / 1000.0,
        )?;
    }
    // The verifier noise.
    let noise = Noise::pinned();
    let mut counts = [0u64; 6];
    for i in 0..SELFTEST_SAMPLES {
        let mut rng = stream(seed, 0, i, Kind::Selftest, 2);
        counts[noise.draw(&mut rng).code() as usize] += 1;
    }
    let shares = [
        1000 - noise.rejecting_permille() - noise.forged,
        noise.unknown,
        noise.disputed,
        noise.shallow,
        noise.lying,
        noise.forged,
    ];
    for (mode, (count, share)) in Mode::ALL.iter().zip(counts.iter().zip(shares)) {
        within(
            format!("verifier noise {}", mode.name()),
            *count,
            SELFTEST_SAMPLES,
            share as f64 / 1000.0,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::f001::metrics;

    fn pass(variant: Variant) -> Pass {
        run_pass(17, 120, variant, metrics::LIST).expect("a pass")
    }

    #[test]
    fn the_mechanism_trips_no_hard_counter_on_a_small_pass() {
        let pass = pass(Variant::Mechanism);
        for name in metrics::HARD_COUNTERS {
            assert_eq!(pass.tally.get(name), 0, "{name}");
        }
        assert!(pass.tally.get("runs") > 3000);
        assert_eq!(pass.profile_tasks, [120, 120, 120]);
    }

    #[test]
    fn the_score_accept_control_false_accepts() {
        let pass = pass(Variant::ScoreAccept);
        assert!(pass.tally.get("false_accepts") > 0);
    }

    #[test]
    fn the_blind_repair_control_edits_collaterally() {
        let pass = pass(Variant::BlindRepair);
        assert!(pass.tally.get("collateral_edits") > 0);
    }

    #[test]
    fn the_stubs_pass_their_own_frequency_test_for_every_seed() {
        for seed in super::super::params::PILOT_SEEDS {
            selftest(seed).unwrap_or_else(|reason| panic!("seed {seed}: {reason}"));
        }
    }

    #[test]
    fn an_exact_zero_variance_cell_must_match_exactly() {
        assert_eq!(z_score(5.0, 5.0, 0.0), 0.0);
        assert!(z_score(6.0, 5.0, 0.0).is_infinite());
        assert!((z_score(12.0, 10.0, 4.0) - 1.0).abs() < 1e-12);
    }
}
