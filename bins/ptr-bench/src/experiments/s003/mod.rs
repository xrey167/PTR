//! Experiment S003, certified branches: do dependency-certified agent branches
//! avoid lost updates and phantoms while merging more concurrent work than
//! serial execution? The harness runs in memory, with no PostgreSQL, in every
//! build; its parameters are the preregistered table in the experiment's
//! `config.toml` (`params`).
//!
//! A run is `iterations` cases. Each case draws one workload and runs it
//! under every arm (`arms`), with the canaries that judge the oracle
//! (`canary`) and the adversarial probes (`probes`). The first case is run a
//! second time and must give the same digest. The result is one JSON line on
//! stdout; everything else goes to stderr, and the exit code is 1 exactly
//! when a hard counter is above zero.

pub mod arms;
pub mod canary;
pub mod metrics;
pub mod model;
pub mod oracle;
pub mod params;
pub mod probes;
pub mod program;
pub mod scratch;
pub mod verifier;
pub mod workload;
pub mod world;

use std::path::PathBuf;
use std::time::Instant;

use arms::{Arm, Run, RunStats};
use metrics::{fnv, Metrics};
use scratch::Scratch;
use workload::Case;

use super::json;

/// The private directory of the durable round trip of case `index`, for the
/// cases that make one; `label` tells a case from its rerun. It is removed when
/// dropped.
fn durable_scratch(seed: u64, index: usize, label: &str) -> Option<Scratch> {
    (index % params::DURABLE_ROUNDTRIP_EVERY_CASES == 0).then(|| {
        Scratch::create(&format!("{seed}-{index}-{label}")).unwrap_or_else(|error| {
            eprintln!("no scratch directory for a durable round trip: {error}");
            std::process::exit(2);
        })
    })
}

/// The most notes of one case that are printed.
const NOTES_PER_CASE: usize = 8;

/// One case, as the result lists it.
struct CaseResult {
    index: usize,
    level: usize,
    groups: usize,
    serial_ticks: u64,
    runs: Vec<RunStats>,
    metrics: Metrics,
    notes: Vec<String>,
    /// FNV-1a over every run's digest and the counters of the probes.
    digest: u64,
}

impl CaseResult {
    /// Bind every deterministic case counter, including canary outcomes, to
    /// the run/probe digest compared with the independent rerun.
    fn deterministic_digest(&self) -> u64 {
        self.metrics.digest_into(self.digest)
    }
}

/// Run one case: the canaries, the probes and every arm. `durable` names the
/// file of the durable round trip, when this case makes one.
fn run_case(seed: u64, index: usize, durable: Option<PathBuf>) -> CaseResult {
    let case = Case::new(seed, index);
    let tasks = case.tasks();
    let mut metrics = Metrics::default();
    let mut notes: Vec<String> = Vec::new();
    let mut runs: Vec<RunStats> = Vec::new();
    let mut digest = fnv(None, &(index as u64).to_le_bytes());

    let (ran, missed) = canary::run(&canary::canaries());
    metrics.canaries_run += ran;
    metrics.canary_misses += missed.len() as u64;
    notes.extend(
        missed
            .iter()
            .map(|name| format!("the oracle missed the {name} canary")),
    );

    match probes::run_all(&case, seed) {
        Ok(outcome) => {
            digest = outcome.metrics.digest_into(digest);
            metrics.absorb_probes(&outcome.metrics);
            notes.extend(outcome.notes);
        }
        Err(error) => {
            metrics.harness_errors += 1;
            notes.push(format!("the probes stopped: {error}"));
        }
    }

    let mut serial_ticks = 0;
    let mut plan: Vec<(Arm, usize)> = vec![(Arm::Serial, 1)];
    for agents in params::AGENTS {
        for arm in [Arm::Certified, Arm::CertifiedReview, Arm::Lww, Arm::Occ] {
            plan.push((arm, agents));
        }
    }
    for (arm, agents) in plan {
        let round_trip = (arm == Arm::Certified && agents == params::AGENTS[0])
            .then(|| durable.clone())
            .flatten();
        let run: Run = arms::run_arm(&case, &tasks, arm, agents, round_trip.as_deref());
        if arm == Arm::Serial {
            serial_ticks = run.stats.ticks;
        }
        digest = fnv(Some(digest), &run.digest.to_le_bytes());
        metrics.absorb(&run.metrics);
        notes.extend(
            run.notes
                .iter()
                .map(|note| format!("{} with {agents}: {note}", arm.name())),
        );
        if let Some(error) = &run.failed {
            notes.push(format!("{} with {agents} stopped: {error}", arm.name()));
        }
        runs.push(run.stats);
    }
    CaseResult {
        index,
        level: case.level,
        groups: case.groups,
        serial_ticks,
        runs,
        metrics,
        notes,
        digest,
    }
}

fn run_json(stats: &RunStats) -> String {
    format!(
        "{{\"arm\":{},\"agents\":{},\"ticks\":{},\"attempts\":{},\"merged\":{},\"no_change\":{},\
         \"conflicts\":{},\"lifecycle_refusals\":{},\"verification_holds\":{},\"escalations\":{},\
         \"review_voids\":{},\"abandoned\":{},\"wasted_ticks\":{},\"unnecessary_refusals\":{},\
         \"complete\":{}}}",
        json::json_string(stats.arm.name()),
        stats.agents,
        stats.ticks,
        stats.attempts,
        stats.merged,
        stats.no_change,
        stats.conflicts,
        stats.lifecycle_refusals,
        stats.verification_holds,
        stats.escalations,
        stats.review_voids,
        stats.abandoned,
        stats.wasted_ticks,
        stats.unnecessary_refusals,
        stats.complete,
    )
}

/// The result line: identity and provenance, every counter, the merge-time
/// histogram, each case's runs, and the sum of the hard counters.
fn result_line(
    iterations: usize,
    seed: u64,
    preregistration: &str,
    elapsed_ns: u128,
    metrics: &Metrics,
    cases: &[CaseResult],
) -> String {
    let mut line = format!(
        "{{\"benchmark\":\"certified-branches\",\"iterations\":{iterations},\"seed\":{seed},\
         \"server\":\"in-memory\",\"preregistration\":{},\"elapsed_ns\":{elapsed_ns}",
        json::json_string(preregistration),
    );
    for (name, value) in metrics
        .hard()
        .into_iter()
        .chain(metrics.coverage())
        .chain(metrics.descriptive())
    {
        line.push_str(&format!(",\"{name}\":{value}"));
    }
    line.push_str(&format!(
        ",\"merge_wall_ns\":{},\"merge_calls\":{},\"merge_wall_le_10us\":{},\
         \"merge_wall_le_100us\":{},\"merge_wall_le_1ms\":{},\"merge_wall_le_10ms\":{},\
         \"merge_wall_gt_10ms\":{}",
        metrics.merge_wall_ns,
        metrics.merge_calls,
        metrics.merge_wall_le_10us,
        metrics.merge_wall_le_100us,
        metrics.merge_wall_le_1ms,
        metrics.merge_wall_le_10ms,
        metrics.merge_wall_gt_10ms,
    ));
    line.push_str(",\"cases\":[");
    for (position, case) in cases.iter().enumerate() {
        if position > 0 {
            line.push(',');
        }
        let runs: Vec<String> = case.runs.iter().map(run_json).collect();
        line.push_str(&format!(
            "{{\"case\":{},\"level\":{},\"groups\":{},\"serial_ticks\":{},\"runs\":[{}]}}",
            case.index,
            case.level,
            case.groups,
            case.serial_ticks,
            runs.join(",")
        ));
    }
    line.push_str(&format!(
        "],\"hard_failures\":{}}}",
        metrics.hard_failures()
    ));
    line
}

/// Run `iterations` cases of seed `seed`, print the result line, and exit 1
/// if any hard counter is above zero.
pub fn run(iterations: usize, seed: u64) {
    let preregistration = match params::echo() {
        Ok(text) => text,
        Err(error) => {
            eprintln!("the preregistration cannot be read: {error}");
            std::process::exit(2);
        }
    };
    let started = Instant::now();
    let mut metrics = Metrics::default();
    let mut cases: Vec<CaseResult> = Vec::with_capacity(iterations);
    for index in 0..iterations {
        let scratch = durable_scratch(seed, index, "first");
        let case = run_case(seed, index, scratch.as_ref().map(|s| s.file("journal.log")));
        drop(scratch);
        eprintln!(
            "case {index} level {} groups {}: serial {} ticks, {} hard, {} notes",
            case.level,
            case.groups,
            case.serial_ticks,
            case.metrics.hard_failures(),
            case.notes.len()
        );
        for note in case.notes.iter().take(NOTES_PER_CASE) {
            eprintln!("  {note}");
        }
        metrics.absorb(&case.metrics);
        cases.push(case);
    }
    // The first case again, from nothing but the seed: every counter and every
    // final state must come out the same.
    if let Some(first) = cases
        .iter()
        .find(|case| case.index == params::NONDETERMINISM_RERUN_CASE)
    {
        let scratch = durable_scratch(seed, first.index, "again");
        let again = run_case(
            seed,
            first.index,
            scratch.as_ref().map(|s| s.file("journal.log")),
        );
        drop(scratch);
        if again.deterministic_digest() != first.deterministic_digest() {
            metrics.nondeterminism += 1;
            eprintln!(
                "case {} is not reproducible: digest {:016x}, then {:016x}",
                first.index,
                first.deterministic_digest(),
                again.deterministic_digest()
            );
        }
    }
    let elapsed_ns = started.elapsed().as_nanos();
    println!(
        "{}",
        result_line(
            iterations,
            seed,
            &preregistration,
            elapsed_ns,
            &metrics,
            &cases
        )
    );
    if metrics.hard_failures() > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_case_runs_every_arm_and_is_reproducible() {
        let mut first = run_case(17, 0, None);
        assert_eq!(first.notes, Vec::<String>::new());
        assert_eq!(first.metrics.hard(), Metrics::default().hard());
        // The serial run and, for each of the four agent counts, the four other arms.
        assert_eq!(first.runs.len(), 1 + 4 * params::AGENTS.len());
        assert_eq!(first.runs[0].arm, Arm::Serial);
        assert!(first.serial_ticks > 0);
        assert_eq!(first.metrics.canaries_run, 9);
        let second = run_case(17, 0, None);
        assert_eq!(first.deterministic_digest(), second.deterministic_digest());
        assert_ne!(
            first.deterministic_digest(),
            run_case(29, 0, None).deterministic_digest()
        );
        first.metrics.canary_misses += 1;
        assert_ne!(first.deterministic_digest(), second.deterministic_digest());
    }

    #[test]
    fn a_durable_case_reopens_its_journal_and_leaves_no_file() {
        let scratch = Scratch::create("mod-test").expect("a scratch directory");
        let path = scratch.file("journal.log");
        let case = run_case(17, 0, Some(path.clone()));
        assert_eq!(case.metrics.hard(), Metrics::default().hard());
        assert_eq!(case.metrics.durable_roundtrips, 1);
        assert!(!path.exists());
    }

    #[test]
    fn the_result_line_is_one_line_of_json_with_the_preregistration_and_every_counter() {
        let case = run_case(17, 0, None);
        let metrics = case.metrics.clone();
        let text = params::echo().expect("the preregistration");
        let line = result_line(1, 17, &text, 5, &metrics, &[case]);
        assert!(!line.contains('\n'));
        assert!(line.starts_with(
            r#"{"benchmark":"certified-branches","iterations":1,"seed":17,"server":"in-memory","preregistration":"{\"agents\":"#
        ));
        for name in metrics::HARD
            .iter()
            .chain(metrics::COVERAGE)
            .chain(metrics::DESCRIPTIVE)
        {
            assert!(line.contains(&format!("\"{name}\":")), "{name}");
        }
        assert!(line.contains(r#""merge_wall_gt_10ms":"#));
        assert!(line.contains(r#""cases":[{"case":0,"level":0,"groups":4,"serial_ticks":"#));
        assert!(line.ends_with(r#""hard_failures":0}"#));
        assert!(line.contains(r#"{"arm":"certified","agents":2,"ticks":"#));
    }
}
