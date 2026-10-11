//! The eight arms, the audit of every request against the independent oracle, and the paired
//! tables the aggregator decides from.
//!
//! All arms serve the same requests, with the same verifier-call budget, the same simulated
//! Pods and the same draws for the same (request, Pod, code, occurrence). They are the same
//! `VcwLoop` configured differently, so a difference between two arms is a difference in what
//! the arms name and nothing else.

use std::collections::BTreeMap;
use std::sync::Arc;

use ptr_router::PolicyBook;
use ptr_runtime::vcw::StepDecision;

use super::super::rng::Rng;
use super::lane::{EndTag, Lane, LaneConfig, RequestRecord, RouteScorer};
use super::learn::{train, Trained};
use super::params::{self, Scale};
use super::task::{all_codes, Judge, Kind, Task, KINDS};
use super::world::{RegistryMode, WorldId};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Arm {
    Full,
    Gate,
    BestOfN,
    NoWorkspace,
    NoFinding,
    NoSelfModel,
    NoL3,
    Shuffled,
    /// Upper reference: the best fixed Pod per code and the good proposer, known with hindsight.
    Oracle,
}

pub const ARMS: [Arm; 9] = [
    Arm::Full,
    Arm::Gate,
    Arm::BestOfN,
    Arm::NoWorkspace,
    Arm::NoFinding,
    Arm::NoSelfModel,
    Arm::NoL3,
    Arm::Shuffled,
    Arm::Oracle,
];

impl Arm {
    pub fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Gate => "gate",
            Self::BestOfN => "bon",
            Self::NoWorkspace => "no_workspace",
            Self::NoFinding => "no_finding",
            Self::NoSelfModel => "no_selfmodel",
            Self::NoL3 => "no_l3",
            Self::Shuffled => "shuffled",
            Self::Oracle => "oracle",
        }
    }

    pub fn number(self) -> usize {
        ARMS.iter().position(|arm| *arm == self).expect("an arm") + 1
    }
}

fn uniform(kind: Kind) -> RouteScorer {
    RouteScorer::Active(PolicyBook::new(kind.name()).scorer())
}

/// The lane an arm serves `kind` with.
pub fn arm_config(
    arm: Arm,
    world: WorldId,
    kind: Kind,
    seed: u64,
    trained: &Trained,
) -> LaneConfig {
    let learned = trained.of(kind);
    let gate = if learned.gate_promoted {
        RouteScorer::Gate(Arc::clone(&learned.gate))
    } else {
        uniform(kind)
    };
    let bon_gate = if learned.bon_promoted {
        RouteScorer::Gate(Arc::clone(&learned.bon_gate))
    } else {
        uniform(kind)
    };
    let full = LaneConfig {
        world,
        kind,
        registry: RegistryMode::True,
        scorer: RouteScorer::Active(learned.vcw_scorer.clone()),
        hints_steer: true,
        schema: true,
        l3: true,
        broadcast: true,
        sim_seed: seed * 1000 + world as u64,
        router_seed: (seed % 250) as u8 + 1,
        learner: None,
        calls: params::VERIFIER_CALL_BUDGET,
        shared_first_draft: true,
    };
    match arm {
        Arm::Full => full,
        Arm::Gate => LaneConfig {
            registry: RegistryMode::Generic,
            scorer: gate,
            hints_steer: false,
            schema: false,
            l3: false,
            ..full
        },
        Arm::BestOfN => LaneConfig {
            registry: RegistryMode::NoRepair,
            scorer: bon_gate,
            calls: params::VERIFIER_CALL_BUDGET,
            shared_first_draft: true,
            hints_steer: false,
            schema: false,
            l3: false,
            ..full
        },
        Arm::NoWorkspace => LaneConfig {
            broadcast: false,
            ..full
        },
        Arm::NoFinding => LaneConfig {
            registry: RegistryMode::Generic,
            scorer: uniform(kind),
            hints_steer: false,
            schema: false,
            l3: false,
            ..full
        },
        Arm::NoSelfModel => LaneConfig {
            hints_steer: false,
            schema: false,
            ..full
        },
        Arm::NoL3 => LaneConfig { l3: false, ..full },
        Arm::Shuffled => LaneConfig {
            registry: RegistryMode::Shuffled,
            ..full
        },
        Arm::Oracle => LaneConfig {
            registry: RegistryMode::Oracle,
            scorer: uniform(kind),
            hints_steer: false,
            schema: false,
            l3: false,
            ..full
        },
    }
}

/// Counters of the integrity falsifiers, summed over every request of an arm.
#[derive(Clone, Debug, Default)]
pub struct Integrity {
    pub false_accepts: u64,
    pub unverified_returns: u64,
    pub double_verifications: u64,
    pub budget_violations: u64,
    pub routing_outside_admissible: u64,
    pub verdict_mismatches: u64,
    pub verdict_overrides: u64,
    pub verifier_call_mismatches: u64,
    pub calls_after_acceptance: u64,
    pub l3_unadmitted: u64,
    pub loop_errors: u64,
    pub product_panics: u64,
    pub exhausted_as_success: u64,
}

impl Integrity {
    pub fn add(&mut self, other: &Integrity) {
        self.false_accepts += other.false_accepts;
        self.unverified_returns += other.unverified_returns;
        self.double_verifications += other.double_verifications;
        self.budget_violations += other.budget_violations;
        self.routing_outside_admissible += other.routing_outside_admissible;
        self.verdict_mismatches += other.verdict_mismatches;
        self.verdict_overrides += other.verdict_overrides;
        self.verifier_call_mismatches += other.verifier_call_mismatches;
        self.calls_after_acceptance += other.calls_after_acceptance;
        self.l3_unadmitted += other.l3_unadmitted;
        self.loop_errors += other.loop_errors;
        self.product_panics += other.product_panics;
        self.exhausted_as_success += other.exhausted_as_success;
    }
}

/// What one arm did over the world's request stream.
#[derive(Clone, Debug)]
pub struct ArmResult {
    pub arm: Arm,
    pub requests: u64,
    pub successes: u64,
    pub verified: u64,
    pub exhausted: u64,
    pub escalated: u64,
    pub verifier_calls: u64,
    pub pod_calls: u64,
    pub rounds: u64,
    pub elapsed_ns: u128,
    pub by_kind: [[u64; 2]; 3],
    pub success: Vec<bool>,
    /// Requests whose first candidate the verifier accepted (the single-shot success).
    pub first_pass: u64,
    /// Repair calls in which a Pod that reads the broadcast saw it name its code.
    pub broadcast_reads: u64,
    /// Per request: the first candidate carried the unrepairable code as a hard finding.
    pub gap_first: Vec<bool>,
    /// Per request: the defect codes planted in the first candidate (the shared first draft).
    pub first_planted: Vec<String>,
    pub births: u64,
    pub l3_refusals: u64,
    pub repairs_applied: u64,
    pub repairs_declined: u64,
    pub regenerations: u64,
    pub unroutable_codes: u64,
    pub hard_by_code: BTreeMap<String, u64>,
    pub integrity: Integrity,
}

impl ArmResult {
    fn new(arm: Arm) -> Self {
        Self {
            arm,
            requests: 0,
            successes: 0,
            verified: 0,
            exhausted: 0,
            escalated: 0,
            verifier_calls: 0,
            pod_calls: 0,
            rounds: 0,
            elapsed_ns: 0,
            by_kind: [[0; 2]; 3],
            success: Vec::new(),
            first_pass: 0,
            broadcast_reads: 0,
            gap_first: Vec::new(),
            first_planted: Vec::new(),
            births: 0,
            l3_refusals: 0,
            repairs_applied: 0,
            repairs_declined: 0,
            regenerations: 0,
            unroutable_codes: 0,
            hard_by_code: all_codes()
                .into_iter()
                .map(|c| (c.to_string(), 0))
                .collect(),
            integrity: Integrity::default(),
        }
    }
}

/// Judge one request against the independent oracle and the harness's own restatement of the
/// loop's rules. Returns whether the request counts as a success.
fn audit(record: &RequestRecord, lane: &Lane, result: &mut ArmResult) -> bool {
    result.first_pass += u64::from(record.events.first().is_some_and(|e| e.accepts()));
    result.gap_first.push(
        record
            .events
            .first()
            .is_some_and(|e| e.hard_codes.iter().any(|c| c == super::world::GAP_CODE)),
    );
    result.first_planted.push(
        record
            .events
            .first()
            .map(|e| e.planted.iter().cloned().collect::<Vec<_>>().join(","))
            .unwrap_or_default(),
    );
    let integrity = &mut result.integrity;
    let mut success = false;
    match &record.tag {
        EndTag::Error => integrity.loop_errors += 1,
        EndTag::Panic => integrity.product_panics += 1,
        _ => {}
    }
    for event in &record.events {
        for code in &event.hard_codes {
            *result.hard_by_code.entry(code.clone()).or_default() += 1;
        }
        let disagree = event.accepts() != event.clean;
        let unnamed = event.status == ptr_verifier::VerificationStatus::Fail
            && (event.hard_codes.is_empty()
                || event
                    .hard_codes
                    .iter()
                    .any(|code| !event.planted.contains(code)));
        if disagree || unnamed {
            integrity.verdict_mismatches += 1;
        }
    }
    if let Some(first) = record.events.iter().position(|event| event.accepts()) {
        integrity.calls_after_acceptance += (record.events.len() - first - 1) as u64;
    }
    if let Some(trace) = &record.trace {
        let spent = trace.spent();
        if record.events.len() as u64 != u64::from(spent.verifier_calls) {
            integrity.verifier_call_mismatches += 1;
        }
        if record.events.len() as u32 > params::VERIFIER_CALL_BUDGET
            || spent.rounds > params::ROUND_BUDGET
        {
            integrity.budget_violations += 1;
        }
        let candidates = trace.candidates().count();
        if record.events.len() > candidates {
            integrity.double_verifications += (record.events.len() - candidates) as u64;
        }
        result.verifier_calls += u64::from(spent.verifier_calls);
        result.pod_calls += u64::from(spent.pod_calls);
        result.rounds += u64::from(spent.rounds);
        // The loop's decision on each candidate against the oracle's verdict on it.
        let mut events = record.events.iter();
        for step in trace.steps() {
            if step.candidate.is_some() {
                if let Some(event) = events.next() {
                    if (step.decision == StepDecision::Accepted) != event.accepts() {
                        integrity.verdict_overrides += 1;
                    }
                }
            }
        }
        for route in trace.routes() {
            let named = lane.is_admissible(route.class(), route.pod());
            let support = route
                .support()
                .iter()
                .all(|(pod, _)| lane.is_admissible(route.class(), pod));
            if !(named && support) {
                integrity.routing_outside_admissible += 1;
            }
        }
    }
    match (&record.tag, &record.accepted) {
        (EndTag::Verified, Some(payload)) => {
            result.verified += 1;
            let independent = Judge::independent_pass(&record.task, payload);
            if !independent {
                integrity.false_accepts += 1;
            }
            let digest = ptr_runtime::vcw::VcwPayload::payload_digest(payload);
            let seen = record
                .events
                .iter()
                .any(|event| event.digest == digest && event.accepts());
            if !seen {
                integrity.unverified_returns += 1;
            }
            success = independent && seen;
        }
        (EndTag::Verified, None) => integrity.exhausted_as_success += 1,
        (EndTag::Exhausted, _) => result.exhausted += 1,
        (EndTag::Escalated, _) => result.escalated += 1,
        _ => {}
    }
    success
}

/// A seeded probe of the broadcast path, independent of any success rate: in the world that
/// declares a Pod reading the broadcast, serve the same requests with the broadcast delivered and
/// withheld, and return how many times a Pod read it (delivered, withheld) and how many requests produced a different
/// sequence of candidates because of it.
pub fn ctx_probe(seed: u64, requests: usize) -> (u64, u64, u64) {
    let (mut reads_on, mut reads_off, mut changed) = (0, 0, 0);
    for kind in KINDS {
        let make = |broadcast: bool| {
            Lane::new(LaneConfig {
                world: WorldId::Ctx,
                kind,
                registry: RegistryMode::True,
                scorer: uniform(kind),
                hints_steer: false,
                schema: false,
                l3: false,
                broadcast,
                sim_seed: seed ^ 0x9E37,
                router_seed: (seed % 250) as u8 + 1,
                learner: None,
                calls: params::VERIFIER_CALL_BUDGET,
                shared_first_draft: true,
            })
        };
        let (mut on, mut off) = (make(true), make(false));
        for index in 0..requests {
            let task = Arc::new(Task::generate(seed ^ 0x51, index as u64, kind));
            let id = format!("probe-{}-{index}", kind.name());
            let seen = |lane: &mut Lane| -> Vec<[u8; 32]> {
                lane.serve(&id, &task)
                    .events
                    .iter()
                    .map(|e| e.digest)
                    .collect()
            };
            if seen(&mut on) != seen(&mut off) {
                changed += 1;
            }
        }
        reads_on += on.broadcast_reads();
        reads_off += off.broadcast_reads();
    }
    (reads_on, reads_off, changed)
}

/// The audit of one record on a fresh accumulator, for the tests.
#[cfg(test)]
pub fn audit_for_tests(record: &RequestRecord, lane: &Lane) -> Integrity {
    let mut result = ArmResult::new(Arm::Full);
    audit(record, lane, &mut result);
    result.integrity
}

/// The request stream of a seed: kinds by the declared weights, tasks pure in (seed, index).
pub fn request_stream(seed: u64, n: usize) -> Vec<(Kind, Arc<Task>)> {
    (0..n)
        .map(|index| {
            let mut rng = Rng::new(seed ^ 0xF007_0000_0000_0001).fork(index as u64);
            let draw = rng.below(1000);
            let mut edge = 0;
            let mut kind = KINDS[2];
            for (candidate, weight) in KINDS.iter().zip(params::KIND_WEIGHTS_PERMILLE) {
                edge += weight;
                if draw < edge {
                    kind = *candidate;
                    break;
                }
            }
            (kind, Arc::new(Task::generate(seed, index as u64, kind)))
        })
        .collect()
}

pub fn run_arm(
    arm: Arm,
    world: WorldId,
    seed: u64,
    trained: &Trained,
    tasks: &[(Kind, Arc<Task>)],
) -> ArmResult {
    let mut lanes: Vec<Lane> = KINDS
        .iter()
        .map(|kind| Lane::new(arm_config(arm, world, *kind, seed, trained)))
        .collect();
    let mut result = ArmResult::new(arm);
    for (index, (kind, task)) in tasks.iter().enumerate() {
        let lane = &mut lanes[kind.index()];
        let record = lane.serve(&format!("e{seed}-{index}"), task);
        let ok = audit(&record, lane, &mut result);
        result.requests += 1;
        result.by_kind[kind.index()][0] += 1;
        if ok {
            result.successes += 1;
            result.by_kind[kind.index()][1] += 1;
        }
        result.elapsed_ns += record.elapsed_ns;
        result.success.push(ok);
    }
    for lane in &mut lanes {
        let bad = lane.audit_l3();
        result.integrity.l3_unadmitted += bad;
        result.births += lane.births;
        result.broadcast_reads += lane.broadcast_reads();
        result.l3_refusals += lane.l3_refusals;
        let counters = lane.counters();
        result.repairs_applied += counters.repairs_applied;
        result.repairs_declined += counters.repairs_declined;
        result.regenerations += counters.regenerations;
        result.unroutable_codes += counters.unroutable_codes;
    }
    result
}

/// One world of one seed: its training, its arms.
pub struct WorldRun {
    pub world: WorldId,
    pub trained: Trained,
    pub arms: Vec<ArmResult>,
    pub requests: usize,
}

impl WorldRun {
    pub fn arm(&self, arm: Arm) -> &ArmResult {
        &self.arms[arm.number() - 1]
    }

    /// The number of (arm, request) pairs whose first candidate carries other planted defects than
    /// the full arm's. The first draft is shared by construction, so this is 0 unless arm state
    /// reaches round 0.
    pub fn first_draft_differences(&self) -> u64 {
        let full = self.arm(Arm::Full);
        self.arms
            .iter()
            .map(|arm| {
                arm.first_planted
                    .iter()
                    .zip(&full.first_planted)
                    .filter(|(a, b)| a != b)
                    .count() as u64
                    + arm.first_planted.len().abs_diff(full.first_planted.len()) as u64
            })
            .sum()
    }
}

pub fn run_world(world: WorldId, seed: u64, scale: Scale) -> Result<WorldRun, String> {
    let trained = train(world, seed, scale)?;
    let tasks = request_stream(seed, scale.requests());
    let arms = ARMS
        .iter()
        .map(|arm| run_arm(*arm, world, seed, &trained, &tasks))
        .collect();
    Ok(WorldRun {
        world,
        trained,
        arms,
        requests: tasks.len(),
    })
}

/// The 2x2 table of two arms over the same requests: both succeed, only `a`, only `b`, neither.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Pair {
    pub both: u64,
    pub only_a: u64,
    pub only_b: u64,
    pub neither: u64,
}

impl Pair {
    pub fn of(a: &ArmResult, b: &ArmResult) -> Self {
        let mut pair = Self::default();
        for (x, y) in a.success.iter().zip(&b.success) {
            match (x, y) {
                (true, true) => pair.both += 1,
                (true, false) => pair.only_a += 1,
                (false, true) => pair.only_b += 1,
                (false, false) => pair.neither += 1,
            }
        }
        pair
    }

    /// `(a - b)` in successes.
    #[cfg(test)]
    pub fn difference(&self) -> i64 {
        self.only_a as i64 - self.only_b as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_tables_count_every_request_once_and_difference_is_antisymmetric() {
        let mut a = ArmResult::new(Arm::Full);
        let mut b = ArmResult::new(Arm::Gate);
        a.success = vec![true, true, false, false, true];
        b.success = vec![true, false, true, false, false];
        let pair = Pair::of(&a, &b);
        assert_eq!(
            pair,
            Pair {
                both: 1,
                only_a: 2,
                only_b: 1,
                neither: 1
            }
        );
        assert_eq!(pair.difference(), 1);
        assert_eq!(Pair::of(&b, &a).difference(), -1);
    }

    #[test]
    fn the_stream_is_pure_and_follows_the_declared_weights() {
        let a = request_stream(17, 2000);
        let b = request_stream(17, 2000);
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.0 == y.0 && x.1.digest() == y.1.digest()));
        let arithmetic = a.iter().filter(|(k, _)| *k == Kind::Arithmetic).count();
        assert!((700..900).contains(&arithmetic), "{arithmetic}");
    }

    #[test]
    fn arm_numbers_are_one_to_nine_in_the_declared_order() {
        let numbers: Vec<usize> = ARMS.iter().map(|arm| arm.number()).collect();
        assert_eq!(numbers, (1..=9).collect::<Vec<_>>());
    }
}
