//! The five arms and the tick scheduler they share.
//!
//! An arm runs one case's tasks with some number of agents over a
//! [`World`]: `serial` (one agent, with the background held back between
//! tasks), `certified` (every agent merges through the runtime under the
//! auto policy), `certified-review` (the same under the review policy, with
//! one reviewer for what the policy does not auto-propose), and two
//! baselines that skip certification: `lww` (last writer wins) and `occ`
//! (key-level optimistic concurrency).
//!
//! Time is integer ticks. Within a tick: (1) the background actors act;
//! (2) merges that are due go through one lane, one merge per
//! `merge_ticks`, oldest first; (3) idle agents open the next task. An
//! attempt takes one tick per call its program makes plus a think time drawn
//! per (task, attempt), so attempt `k` of a task lasts the same in every arm.
//! A task that is refused is opened again, up to `max_attempts`; one the
//! verifiers reject is dropped in every arm.

use std::collections::VecDeque;
use std::path::Path;

use super::metrics::{fnv, Metrics};
use super::oracle;
use super::params;
use super::workload::{Background, Case, Task};
use super::world::{Attempt, Authority, GrantKind, HostOutcome, PlanKey, Settled, World};

/// The most ticks a run may take before it is called stuck.
const TICK_LIMIT: u64 = 2_000_000;

/// An arm of the experiment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arm {
    Serial,
    Certified,
    CertifiedReview,
    Lww,
    Occ,
}

impl Arm {
    pub const ALL: [Arm; 5] = [
        Arm::Serial,
        Arm::Certified,
        Arm::CertifiedReview,
        Arm::Lww,
        Arm::Occ,
    ];

    /// The arm's name in the result.
    pub fn name(self) -> &'static str {
        match self {
            Arm::Serial => "serial",
            Arm::Certified => "certified",
            Arm::CertifiedReview => "certified-review",
            Arm::Lww => "lww",
            Arm::Occ => "occ",
        }
    }

    /// Whether the arm merges through the runtime, and so is judged by the
    /// oracle, replayed and counted in the hard counters.
    pub fn certifies(self) -> bool {
        matches!(self, Arm::Serial | Arm::Certified | Arm::CertifiedReview)
    }
}

/// What one run of an arm did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunStats {
    pub arm: Arm,
    pub agents: usize,
    /// The makespan in ticks.
    pub ticks: u64,
    /// Branches opened.
    pub attempts: u64,
    /// Merges committed.
    pub merged: u64,
    /// Merges the verifiers admitted that changed nothing.
    pub no_change: u64,
    pub conflicts: u64,
    pub lifecycle_refusals: u64,
    /// Changes the verifiers did not admit.
    pub verification_holds: u64,
    pub escalations: u64,
    /// Approvals a changed plan voided.
    pub review_voids: u64,
    /// Tasks given up after the most attempts, or refused for a reason no
    /// retry changes.
    pub abandoned: u64,
    /// Ticks of attempts that did not commit.
    pub wasted_ticks: u64,
    /// Refusals where merging would have given the serial state.
    pub unnecessary_refusals: u64,
}

impl RunStats {
    fn new(arm: Arm, agents: usize) -> Self {
        Self {
            arm,
            agents,
            ticks: 0,
            attempts: 0,
            merged: 0,
            no_change: 0,
            conflicts: 0,
            lifecycle_refusals: 0,
            verification_holds: 0,
            escalations: 0,
            review_voids: 0,
            abandoned: 0,
            wasted_ticks: 0,
            unnecessary_refusals: 0,
        }
    }

    /// Tasks that reached a final state.
    pub fn settled_tasks(&self) -> u64 {
        self.merged + self.no_change + self.verification_holds + self.abandoned
    }
}

/// One run: what it did, what it counted, and whether it stopped early.
#[derive(Debug)]
pub struct Run {
    pub stats: RunStats,
    pub metrics: Metrics,
    pub notes: Vec<String>,
    /// Why the run stopped before its tasks were done, if it did.
    pub failed: Option<String>,
    /// FNV-1a over the run's counters and its final state.
    pub digest: u64,
}

/// A branch an agent is working on.
struct Flight {
    task: usize,
    number: usize,
    attempt: Attempt,
    opened_at: u64,
    due: u64,
}

/// A branch waiting for the reviewer.
struct Review {
    id: u64,
    flight: Flight,
    digest: [u8; 32],
    plan: PlanKey,
    completes_at: u64,
}

struct Agent {
    idle_at: u64,
    current: Option<Flight>,
}

enum Due {
    Review(u64),
    Agent(usize),
}

struct Simulation<'a> {
    case: &'a Case,
    tasks: &'a [Task],
    arm: Arm,
    world: World,
    stats: RunStats,
    agents: Vec<Agent>,
    next_fresh: usize,
    retries: VecDeque<(usize, usize)>,
    lane_free_at: u64,
    reviewer_free_at: u64,
    reviews: Vec<Review>,
    next_review: u64,
    deferred: Vec<Background>,
}

/// Run `arm` over `tasks` with `agents` agents. A run of an arm that
/// certifies ends with the replay, compaction and (given `durable`) durable
/// round trips.
pub fn run_arm(
    case: &Case,
    tasks: &[Task],
    arm: Arm,
    agents: usize,
    durable: Option<&Path>,
) -> Run {
    let agents = if arm == Arm::Serial { 1 } else { agents };
    let kind = if arm == Arm::CertifiedReview {
        GrantKind::Review
    } else {
        GrantKind::Auto
    };
    let stats = RunStats::new(arm, agents);
    let world = match World::new(case, kind) {
        Ok(world) => world,
        Err(error) => return failed_before_start(stats, error),
    };
    let mut simulation = Simulation {
        case,
        tasks,
        arm,
        world,
        stats,
        agents: (0..agents)
            .map(|_| Agent {
                idle_at: 0,
                current: None,
            })
            .collect(),
        next_fresh: 0,
        retries: VecDeque::new(),
        lane_free_at: 0,
        reviewer_free_at: 0,
        reviews: Vec::new(),
        next_review: 0,
        deferred: Vec::new(),
    };
    let mut failed = simulation.execute().err();
    if failed.is_none() && arm.certifies() {
        if let Err(error) = simulation.world.round_trips(durable) {
            failed = Some(error);
        }
    }
    simulation.finish(failed)
}

fn failed_before_start(stats: RunStats, error: String) -> Run {
    let metrics = Metrics {
        harness_errors: 1,
        ..Metrics::default()
    };
    Run {
        stats,
        metrics,
        notes: vec![error.clone()],
        failed: Some(error),
        digest: 0,
    }
}

impl Simulation<'_> {
    /// Tick until every task is settled; the makespan.
    fn execute(&mut self) -> Result<u64, String> {
        let mut tick = 0u64;
        loop {
            self.background(tick)?;
            self.merges(tick)?;
            self.opens(tick)?;
            if self.finished() {
                self.stats.ticks = tick + params::MERGE_TICKS;
                return Ok(self.stats.ticks);
            }
            tick += 1;
            if tick > TICK_LIMIT {
                return Err(format!("the run is stuck after {TICK_LIMIT} ticks"));
            }
        }
    }

    fn finished(&self) -> bool {
        self.next_fresh >= self.tasks.len()
            && self.retries.is_empty()
            && self.reviews.is_empty()
            && self.agents.iter().all(|agent| agent.current.is_none())
    }

    /// Concurrent arms see the background at its ticks; the serial arm holds
    /// it back until its agent is between tasks.
    fn background(&mut self, tick: u64) -> Result<(), String> {
        let events = self.case.background(tick);
        if self.arm == Arm::Serial {
            self.deferred.extend(events);
            Ok(())
        } else {
            self.world.background(tick, &events)
        }
    }

    // ---- merges ---------------------------------------------------------------

    fn merges(&mut self, tick: u64) -> Result<(), String> {
        // Reviews that completed come before merges of the same age.
        let mut due: Vec<(u64, u8, u64, Due)> = Vec::new();
        for review in &self.reviews {
            if review.completes_at <= tick {
                due.push((review.completes_at, 0, review.id, Due::Review(review.id)));
            }
        }
        for (index, agent) in self.agents.iter().enumerate() {
            if let Some(flight) = &agent.current {
                if flight.due <= tick {
                    due.push((flight.due, 1, index as u64, Due::Agent(index)));
                }
            }
        }
        due.sort_by_key(|(at, rank, order, _)| (*at, *rank, *order));
        for (_, _, _, which) in due {
            if self.lane_free_at > tick {
                break;
            }
            self.lane_free_at = tick + params::MERGE_TICKS;
            match which {
                Due::Agent(index) => self.settle_agent(tick, index)?,
                Due::Review(id) => self.settle_review(tick, id)?,
            }
        }
        Ok(())
    }

    fn settle_agent(&mut self, tick: u64, index: usize) -> Result<(), String> {
        let flight = self.agents[index]
            .current
            .take()
            .ok_or("an agent with no branch was due")?;
        self.agents[index].idle_at = tick + params::MERGE_TICKS;
        match self.arm {
            Arm::Serial | Arm::Certified => {
                let settled = self
                    .world
                    .merge(&flight.attempt, &Authority::Triage { score: 1.0 })?;
                self.after_merge(tick, flight, settled);
            }
            Arm::CertifiedReview => {
                let score = self.tasks[flight.task].scores[flight.number];
                let settled = self
                    .world
                    .merge(&flight.attempt, &Authority::Triage { score })?;
                if let Settled::Escalated { digest, plan } = settled {
                    let latency = self.tasks[flight.task].reviews[flight.number];
                    let start = tick.max(self.reviewer_free_at);
                    self.reviewer_free_at = start + latency;
                    self.next_review += 1;
                    self.reviews.push(Review {
                        id: self.next_review,
                        flight,
                        digest,
                        plan,
                        completes_at: start + latency,
                    });
                } else {
                    self.after_merge(tick, flight, settled);
                }
            }
            Arm::Lww => self.settle_lww(tick, flight)?,
            Arm::Occ => self.settle_occ(tick, flight)?,
        }
        Ok(())
    }

    fn settle_review(&mut self, tick: u64, id: u64) -> Result<(), String> {
        let position = self
            .reviews
            .iter()
            .position(|review| review.id == id)
            .ok_or("a review that was due is gone")?;
        let Review {
            flight,
            digest,
            plan,
            ..
        } = self.reviews.remove(position);
        let authority = Authority::Reviewed {
            digest,
            approved: plan,
        };
        let settled = self.world.merge(&flight.attempt, &authority)?;
        self.after_merge(tick, flight, settled);
        Ok(())
    }

    /// Count what a certified merge settled to, and schedule what follows.
    fn after_merge(&mut self, tick: u64, flight: Flight, settled: Settled) {
        let wasted = tick.saturating_sub(flight.opened_at);
        match settled {
            Settled::Committed => self.stats.merged += 1,
            Settled::NoChange => self.stats.no_change += 1,
            Settled::Conflict | Settled::LifecycleChanged => {
                self.stats.wasted_ticks += wasted;
                self.retry(&flight);
            }
            Settled::PlanChanged => {
                self.stats.review_voids += 1;
                self.stats.wasted_ticks += wasted;
                self.retry(&flight);
            }
            Settled::Rejected => {
                self.stats.verification_holds += 1;
                self.stats.wasted_ticks += wasted;
            }
            Settled::Refused | Settled::Escalated { .. } => {
                self.stats.abandoned += 1;
                self.stats.wasted_ticks += wasted;
            }
        }
    }

    /// Open the task again, or give it up after the most attempts.
    fn retry(&mut self, flight: &Flight) {
        if flight.number + 1 >= params::MAX_ATTEMPTS {
            self.stats.abandoned += 1;
        } else {
            self.retries.push_back((flight.task, flight.number + 1));
        }
    }

    // ---- baselines ------------------------------------------------------------

    fn settle_lww(&mut self, tick: u64, flight: Flight) -> Result<(), String> {
        let Some(delta) = self.world.overlay(&flight.attempt) else {
            self.stats.abandoned += 1;
            return Ok(());
        };
        let outcome = self
            .world
            .baseline_commit(&flight.attempt, &delta, "s003-lww", false)?;
        self.baseline_outcome(tick, &flight, outcome);
        Ok(())
    }

    fn settle_occ(&mut self, tick: u64, flight: Flight) -> Result<(), String> {
        let moved = flight
            .attempt
            .footprint
            .reads
            .iter()
            .any(|(key, seen)| self.world.model.value(key) != seen.as_ref());
        if moved {
            self.stats.conflicts += 1;
            self.stats.wasted_ticks += tick.saturating_sub(flight.opened_at);
            self.retry(&flight);
            return Ok(());
        }
        let Ok(delta) = oracle::merge_delta(&flight.attempt.footprint.ops, &self.world.model)
        else {
            self.stats.abandoned += 1;
            return Ok(());
        };
        let outcome = self
            .world
            .baseline_commit(&flight.attempt, &delta, "s003-occ", true)?;
        self.baseline_outcome(tick, &flight, outcome);
        Ok(())
    }

    fn baseline_outcome(&mut self, tick: u64, flight: &Flight, outcome: HostOutcome) {
        match outcome {
            HostOutcome::Committed => self.stats.merged += 1,
            HostOutcome::NoChange => self.stats.no_change += 1,
            HostOutcome::Refused => {
                self.stats.verification_holds += 1;
                self.stats.wasted_ticks += tick.saturating_sub(flight.opened_at);
            }
        }
    }

    // ---- opening --------------------------------------------------------------

    fn opens(&mut self, tick: u64) -> Result<(), String> {
        for index in 0..self.agents.len() {
            if self.agents[index].current.is_some() || self.agents[index].idle_at > tick {
                continue;
            }
            let Some((task, number)) = self.next_work() else {
                continue;
            };
            if self.arm == Arm::Serial && !self.deferred.is_empty() {
                let events = std::mem::take(&mut self.deferred);
                self.world.background(tick, &events)?;
            }
            let spec = &self.tasks[task];
            let id = format!("c{}-t{task}-a{number}", self.case.index);
            let attempt = self.world.open(&id, index, &spec.program, spec.rely)?;
            self.stats.attempts += 1;
            let due = tick + attempt.steps + spec.thinks[number];
            self.agents[index].current = Some(Flight {
                task,
                number,
                attempt,
                opened_at: tick,
                due,
            });
        }
        Ok(())
    }

    /// What an idle agent works on next: a task to open again, or the next
    /// one in the queue.
    fn next_work(&mut self) -> Option<(usize, usize)> {
        if let Some(work) = self.retries.pop_front() {
            return Some(work);
        }
        if self.next_fresh < self.tasks.len() {
            self.next_fresh += 1;
            return Some((self.next_fresh - 1, 0));
        }
        None
    }

    // ---- end ------------------------------------------------------------------

    fn finish(mut self, failed: Option<String>) -> Run {
        if let Some(error) = &failed {
            self.world.metrics.harness_errors += 1;
            self.world.note(format!("the run stopped: {error}"));
        }
        let metrics = &self.world.metrics;
        self.stats.conflicts += metrics.conflicts;
        self.stats.lifecycle_refusals += metrics.lifecycle_refusals;
        self.stats.escalations += metrics.escalations;
        self.stats.unnecessary_refusals += metrics.unnecessary_refusals;
        if self.arm.certifies() {
            self.stats.verification_holds += metrics.verification_holds;
        }
        self.world.metrics.abandoned_tasks += self.stats.abandoned;
        self.world.metrics.wasted_ticks += self.stats.wasted_ticks;
        self.world.metrics.review_voids += self.stats.review_voids;
        let digest = digest(&self.stats, &self.world);
        Run {
            stats: self.stats,
            metrics: self.world.metrics,
            notes: self.world.notes,
            failed,
            digest,
        }
    }
}

/// The digest of a run: its arm, its statistics, every counter and the final
/// state, with no wall-clock value in it.
fn digest(stats: &RunStats, world: &World) -> u64 {
    let mut state = fnv(None, stats.arm.name().as_bytes());
    let numbers = [
        stats.agents as u64,
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
    ];
    for number in numbers {
        state = fnv(Some(state), &number.to_le_bytes());
    }
    state = world.metrics.digest_into(state);
    for (key, value) in world.model.values() {
        state = fnv(Some(state), key.as_bytes());
        state = fnv(Some(state), format!("{value:?}").as_bytes());
    }
    for (key, inputs) in world.model.dependencies() {
        state = fnv(Some(state), key.as_bytes());
        state = fnv(Some(state), format!("{inputs:?}").as_bytes());
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(case_index: usize, arm: Arm, agents: usize) -> Run {
        let case = Case::new(17, case_index);
        let tasks = case.tasks();
        run_arm(&case, &tasks, arm, agents, None)
    }

    fn assert_clean(run: &Run) {
        assert_eq!(run.failed, None);
        assert_eq!(run.notes, Vec::<String>::new());
        assert_eq!(run.metrics.hard(), Metrics::default().hard());
        assert_eq!(
            run.stats.settled_tasks(),
            params::TASKS_PER_CASE as u64,
            "{:?}",
            run.stats
        );
    }

    #[test]
    fn the_serial_arm_never_conflicts_and_settles_every_task() {
        let run = run(0, Arm::Serial, 4);
        assert_clean(&run);
        assert_eq!(run.stats.agents, 1);
        assert_eq!(run.stats.conflicts + run.stats.lifecycle_refusals, 0);
        assert_eq!(run.stats.review_voids, 0);
        assert!(run.stats.merged > 0);
        assert!(run.metrics.replays == 1 && run.metrics.compaction_roundtrips == 1);
    }

    #[test]
    fn a_certified_run_settles_every_task_with_no_hard_failure() {
        for agents in params::AGENTS {
            let run = run(0, Arm::Certified, agents);
            assert_clean(&run);
            assert_eq!(run.stats.agents, agents);
        }
    }

    #[test]
    fn more_agents_finish_sooner_and_attempts_are_paired_with_the_serial_run() {
        let serial = run(1, Arm::Serial, 1);
        let two = run(1, Arm::Certified, 2);
        let eight = run(1, Arm::Certified, 8);
        assert!(eight.stats.ticks < two.stats.ticks);
        assert!(two.stats.ticks < serial.stats.ticks);
        // Serial opens each task once: nothing conflicts.
        assert_eq!(serial.stats.attempts, params::TASKS_PER_CASE as u64);
    }

    #[test]
    fn a_run_is_reproducible_to_the_digest() {
        let first = run(2, Arm::Certified, 4);
        let second = run(2, Arm::Certified, 4);
        assert_eq!(first.stats, second.stats);
        assert_eq!(first.digest, second.digest);
        assert_ne!(first.digest, run(3, Arm::Certified, 4).digest);
    }

    #[test]
    fn the_review_arm_escalates_reviews_and_stays_clean() {
        let run = run(0, Arm::CertifiedReview, 4);
        assert_clean(&run);
        assert!(run.stats.escalations > 0);
        // A reviewer approves the plan of a state that moves on: most approvals are voided.
        assert!(run.stats.review_voids > 0);
    }

    #[test]
    fn a_short_review_run_approves_some_plans_and_voids_others() {
        // One agent, five tasks, at the smallest state: the state stands still
        // long enough for some approvals and not for others.
        let case = Case::new(17, 12);
        let tasks = case.tasks();
        let run = run_arm(&case, &tasks[..5], Arm::CertifiedReview, 1, None);
        assert_eq!(run.failed, None);
        assert_eq!(run.notes, Vec::<String>::new());
        assert_eq!(run.metrics.hard(), Metrics::default().hard());
        assert!(run.metrics.reviewed_merges > 0, "{:?}", run.stats);
        assert!(run.stats.review_voids > 0, "{:?}", run.stats);
        assert_eq!(run.stats.settled_tasks(), 5, "{:?}", run.stats);
    }

    #[test]
    fn the_baselines_settle_every_task_and_show_the_anomalies_certification_prevents() {
        let lww = run(0, Arm::Lww, 8);
        assert_eq!(lww.failed, None);
        assert_eq!(lww.metrics.hard(), Metrics::default().hard());
        assert!(lww.metrics.lww_lost_updates > 0, "{:?}", lww.metrics);
        assert!(lww.metrics.lww_lost_increments > 0);
        let occ = run(0, Arm::Occ, 8);
        assert_eq!(occ.failed, None);
        assert_eq!(occ.metrics.hard(), Metrics::default().hard());
        assert!(occ.metrics.occ_undetected_phantoms > 0, "{:?}", occ.metrics);
        assert!(
            occ.metrics.occ_stale_input_commits > 0 || occ.metrics.occ_stale_reliance_commits > 0
        );
        for run in [&lww, &occ] {
            assert_eq!(
                run.stats.settled_tasks(),
                params::TASKS_PER_CASE as u64,
                "{:?}",
                run.stats
            );
        }
    }

    #[test]
    fn the_cheaper_contention_levels_run_clean_under_every_certifying_arm() {
        // The larger states take minutes in a debug build; the pilot and the
        // mutation check run every level in release.
        for level in 0..3 {
            for arm in [Arm::Serial, Arm::Certified, Arm::CertifiedReview] {
                let run = run(level, arm, 8);
                assert_clean(&run);
            }
        }
    }
}
