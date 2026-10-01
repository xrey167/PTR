//! The workload of one case: its contention level, the genesis state, the
//! tasks agents work through, and the background actors' schedule.
//!
//! Everything is drawn from a stream named by the seed, the case and a label,
//! never from a generator that something else advances, so two arms that ask
//! for the same thing get the same answer whatever else either has consumed.
//! That is what makes the arms paired: same genesis, same tasks with the same
//! attempt durations and scores, same background events at the same ticks.
//! An attempt's duration is a property of its task: one tick per call the
//! task's program makes on the case's genesis state, plus a think time, so no
//! arm's own state can lengthen or shorten the same attempt.

use std::collections::BTreeSet;

use super::model::{Delta, Model, Val};
use super::params;
use super::program::{keys, Program, RefView};
use crate::experiments::rng::Rng;

/// The genesis host write's source for counters and sets.
pub const GENESIS_SOURCE: &str = "s003-genesis";
/// The project the policy capsules belong to.
pub const PROJECT: &str = "s003";

const LABEL_CASE: u64 = 1;
const LABEL_GENESIS: u64 = 2;
const LABEL_TASK: u64 = 3;
const LABEL_BACKGROUND: u64 = 4;

/// An independent stream named by `root` and `labels`, whatever was drawn
/// from any other.
fn stream(root: u64, labels: &[u64]) -> Rng {
    let mut state = root;
    for label in labels {
        state = Rng::new(state ^ label.wrapping_mul(0xA24B_AED4_963E_E407)).next_u64();
    }
    Rng::new(state)
}

/// One case of a run: which contention level it runs at and where its draws
/// come from.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub index: usize,
    /// The position in the contention ladder.
    pub level: usize,
    /// The number of groups: fewer groups, more contention.
    pub groups: usize,
    root: u64,
}

impl Case {
    /// Choose the contention level by cycling through the ladder with the
    /// zero-based case index, and derive a repeatable stream from seed and index.
    pub fn new(seed: u64, index: usize) -> Self {
        let level = index % params::GROUPS_LADDER.len();
        Self {
            index,
            level,
            groups: params::GROUPS_LADDER[level],
            root: stream(seed, &[LABEL_CASE, index as u64]).next_u64(),
        }
    }

    /// The state every arm starts from.
    pub fn genesis(&self) -> Genesis {
        let mut rng = stream(self.root, &[LABEL_GENESIS]);
        let mut delta = Delta::default();
        for group in 0..self.groups {
            let mut inputs = BTreeSet::new();
            let mut sum = 0i64;
            let mut last = 0i64;
            for item in 0..params::ITEMS_PER_GROUP {
                last = rng.range(0, 100) as i64;
                sum += last;
                delta
                    .upserts
                    .insert(keys::item(group, item), Val::text(last.to_string()));
                inputs.insert(keys::item(group, item));
            }
            // The spare starts equal to the last item, so the rewirer can swap
            // one for the other without changing what the total is.
            delta
                .upserts
                .insert(keys::spare(group), Val::text(last.to_string()));
            delta
                .upserts
                .insert(keys::total(group), Val::text(sum.to_string()));
            delta.dependencies.insert(keys::total(group), inputs);
            let entries = params::ITEMS_PER_GROUP + 1;
            delta.upserts.insert(
                keys::audit(group),
                Val::text(format!("{entries},{}", sum + last)),
            );
        }
        for counter in 0..params::COUNTERS {
            delta.upserts.insert(
                keys::counter(counter),
                Val::counter(params::COUNTER_START, GENESIS_SOURCE),
            );
        }
        for set in 0..params::SETS {
            let members: BTreeSet<String> = (0..params::SET_MEMBERS)
                .filter(|_| rng.chance(0.5))
                .map(keys::member)
                .collect();
            delta
                .upserts
                .insert(keys::set(set), Val::set(&members, GENESIS_SOURCE));
        }
        Genesis {
            delta,
            policies: (0..params::POLICIES).map(keys::policy).collect(),
        }
    }

    /// The tasks of the case, in queue order.
    pub fn tasks(&self) -> Vec<Task> {
        let genesis = self.genesis();
        let mut state = Model::default();
        state
            .apply(&genesis.delta)
            .expect("the genesis delta applies to an empty model");
        for policy in &genesis.policies {
            state.lifecycle.set_live(policy, 1);
        }
        (0..params::TASKS_PER_CASE)
            .map(|index| self.task(index, &state))
            .collect()
    }

    /// Draw a task and its per-attempt think/review times in ticks.
    /// Measure program steps against `genesis` so durations are shared by all arms.
    fn task(&self, index: usize, genesis: &Model) -> Task {
        let mut rng = stream(self.root, &[LABEL_TASK, index as u64]);
        let program = draw_program(&mut rng, self.groups, index);
        let rely = rng
            .chance(params::RELY_PERMILLE as f64 / 1000.0)
            .then(|| rng.index(params::POLICIES));
        let mut thinks = [0u64; params::MAX_ATTEMPTS];
        let mut reviews = [0u64; params::MAX_ATTEMPTS];
        let mut scores = [0f32; params::MAX_ATTEMPTS];
        for attempt in 0..params::MAX_ATTEMPTS {
            // Attempt `k` of a task has one duration, one review latency and
            // one score in every arm.
            let mut attempt_rng = stream(self.root, &[LABEL_TASK, index as u64, attempt as u64]);
            thinks[attempt] = attempt_rng.range(params::THINK_MIN_TICKS, params::THINK_MAX_TICKS);
            reviews[attempt] =
                attempt_rng.range(params::REVIEW_MIN_TICKS, params::REVIEW_MAX_TICKS);
            scores[attempt] = attempt_rng.unit() as f32;
        }
        let mut view = RefView::open(genesis);
        program.run(&mut view, rely);
        let (log, _, _) = view.finish();
        let steps = log.events.len() as u64 * params::STEP_TICKS;
        Task {
            index,
            program,
            rely,
            steps,
            thinks,
            reviews,
            scores,
        }
    }

    /// What the background actors do at `tick`, in a fixed order: ingress, an
    /// external writer, the lifecycle authority, the rewirer. The same for
    /// every arm; how a rewire is completed depends on the state it finds.
    pub fn background(&self, tick: u64) -> Vec<Background> {
        let mut rng = stream(self.root, &[LABEL_BACKGROUND, tick]);
        let mut events = Vec::new();
        if occurs(&mut rng, params::INGRESS_PERMILLE_PER_TICK) {
            events.push(Background::Ingress {
                request: format!("r{tick}"),
                text: format!("request received at tick {tick}"),
            });
        }
        if occurs(&mut rng, params::EXTERNAL_WRITE_PERMILLE_PER_TICK) {
            events.push(Background::External {
                group: rng.index(self.groups),
                item: rng.index(params::ITEMS_PER_GROUP),
                value: rng.range(0, 100) as i64,
            });
        }
        if occurs(&mut rng, params::LIFECYCLE_PERMILLE_PER_TICK) {
            events.push(Background::Lifecycle {
                policy: rng.index(params::POLICIES),
                revoke: rng.chance(0.5),
            });
        }
        if occurs(&mut rng, params::REWIRE_PERMILLE_PER_TICK) {
            events.push(Background::Rewire {
                group: rng.index(self.groups),
                swap_after: rng.range(params::REWIRE_SWAP_MIN_TICKS, params::REWIRE_SWAP_MAX_TICKS),
            });
        }
        events
    }
}

/// The state every arm of a case starts from: one host write, and the
/// capsules whose generations programs may rely on.
#[derive(Clone, Debug)]
pub struct Genesis {
    pub delta: Delta,
    /// The lifecycle targets, each committed at generation 1.
    pub policies: Vec<String>,
}

/// One task: what the agent does and how long each attempt takes.
#[derive(Clone, Debug)]
pub struct Task {
    pub index: usize,
    pub program: Program,
    /// A policy the branch also relies on, when it is live at open.
    pub rely: Option<usize>,
    /// The calls the program makes on the genesis state, one tick each: the
    /// same in every arm, whatever state an arm's attempt opens on.
    pub steps: u64,
    /// Think time in ticks, per attempt.
    pub thinks: [u64; params::MAX_ATTEMPTS],
    /// The reviewer's latency in ticks, per attempt.
    pub reviews: [u64; params::MAX_ATTEMPTS],
    /// The host's score for the branch, per attempt, for the review policy.
    pub scores: [f32; params::MAX_ATTEMPTS],
}

/// What a background actor does. The rewirer's two steps are one event: the
/// prime now, and the swap `swap_after` ticks later if the state allows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Background {
    /// `ingest_text` of a request.
    Ingress { request: String, text: String },
    /// A host write of one item.
    External {
        group: usize,
        item: usize,
        value: i64,
    },
    /// A policy is superseded, or revoked and committed again a tick later.
    Lifecycle { policy: usize, revoke: bool },
    /// Prime a group's total for a swap of its input set.
    Rewire { group: usize, swap_after: u64 },
}

/// Whether an actor of `permille` fires on this tick.
fn occurs(rng: &mut Rng, permille: u64) -> bool {
    rng.below(1000) < permille
}

/// A signed amount of magnitude `1..=max`.
fn signed(rng: &mut Rng, max: i64) -> i64 {
    let magnitude = rng.range(1, max as u64) as i64;
    if rng.chance(0.5) {
        magnitude
    } else {
        -magnitude
    }
}

/// Draw a program using the preregistered weights and operand ranges.
/// `groups` bounds group selection; `task` identifies a capped insertion's key.
fn draw_program(rng: &mut Rng, groups: usize, task: usize) -> Program {
    let mut pick = rng.below(1000);
    let mut which = params::PROGRAM_WEIGHTS_PERMILLE.len() - 1;
    for (index, weight) in params::PROGRAM_WEIGHTS_PERMILLE.iter().enumerate() {
        if pick < *weight {
            which = index;
            break;
        }
        pick -= weight;
    }
    let group = rng.index(groups);
    match params::PROGRAMS[which] {
        "rmw" => Program::Rmw {
            group,
            item: rng.index(params::ITEMS_PER_GROUP),
            delta: signed(rng, params::RMW_DELTA_MAX),
        },
        "write_skew" => {
            let first = rng.index(params::ITEMS_PER_GROUP);
            let second =
                (first + 1 + rng.index(params::ITEMS_PER_GROUP - 1)) % params::ITEMS_PER_GROUP;
            Program::WriteSkew {
                group,
                first,
                second,
            }
        }
        "insert_capped" => Program::InsertCapped { group, task },
        "remove_extra" => Program::RemoveExtra { group },
        "audit" => Program::Audit { group },
        "group_total" => Program::GroupTotal {
            group,
            item: rng.index(params::ITEMS_PER_GROUP),
            delta: signed(rng, params::RMW_DELTA_MAX),
        },
        "counter_add" => Program::CounterAdd {
            counter: rng.index(params::COUNTERS),
            amount: signed(rng, params::COUNTER_ADD_MAX),
        },
        "set_op" => Program::SetOp {
            set: rng.index(params::SETS),
            member: rng.index(params::SET_MEMBERS),
            insert: rng.chance(0.5),
        },
        "guarded_decrement" => Program::GuardedDecrement {
            counter: rng.index(params::COUNTERS),
        },
        other => unreachable!("the preregistration lists no program {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::s003::model::Model;

    #[test]
    fn a_case_is_the_same_whatever_was_drawn_before_it() {
        let first = Case::new(17, 3);
        let second = Case::new(17, 3);
        assert_eq!(
            format!("{:?}", first.tasks()),
            format!("{:?}", second.tasks())
        );
        assert_eq!(first.genesis().delta, second.genesis().delta);
        // Ticks asked for out of order give what they give in order.
        let forward: Vec<_> = (0..500).map(|tick| first.background(tick)).collect();
        let backward: Vec<_> = (0..500).rev().map(|tick| second.background(tick)).collect();
        let backward: Vec<_> = backward.into_iter().rev().collect();
        assert_eq!(forward, backward);
    }

    #[test]
    fn another_seed_or_case_draws_another_workload() {
        let base = format!("{:?}", Case::new(17, 3).tasks());
        assert_ne!(base, format!("{:?}", Case::new(29, 3).tasks()));
        assert_ne!(base, format!("{:?}", Case::new(17, 9).tasks()));
    }

    #[test]
    fn a_tasks_steps_are_the_calls_of_its_program_on_the_genesis_state() {
        let case = Case::new(17, 1);
        let tasks = case.tasks();
        let mut genesis = Model::default();
        genesis
            .apply(&case.genesis().delta)
            .expect("the genesis applies");
        for policy in &case.genesis().policies {
            genesis.lifecycle.set_live(policy, 1);
        }
        for task in &tasks {
            let mut view = RefView::open(&genesis);
            task.program.run(&mut view, task.rely);
            let (log, _, _) = view.finish();
            assert!(task.steps >= 1);
            assert_eq!(task.steps, log.events.len() as u64 * params::STEP_TICKS);
        }
        let again: Vec<u64> = case.tasks().iter().map(|task| task.steps).collect();
        assert_eq!(
            again,
            tasks.iter().map(|task| task.steps).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_level_is_the_case_modulo_the_ladder_and_sets_the_groups() {
        for index in 0..24 {
            let case = Case::new(17, index);
            assert_eq!(case.level, index % params::GROUPS_LADDER.len());
            assert_eq!(case.groups, params::GROUPS_LADDER[case.level]);
        }
        assert_eq!(Case::new(17, 0).groups, 4);
    }

    #[test]
    fn the_genesis_is_a_state_the_model_accepts_with_every_total_matching_its_inputs() {
        for index in 0..6 {
            let case = Case::new(29, index);
            let genesis = case.genesis();
            let mut model = Model::default();
            model.apply(&genesis.delta).expect("the genesis applies");
            for group in 0..case.groups {
                let total = keys::total(group);
                let inputs = model.inputs(&total);
                assert_eq!(inputs.len(), params::ITEMS_PER_GROUP);
                let sum: i64 = inputs
                    .iter()
                    .map(|input| {
                        model
                            .value(input)
                            .unwrap()
                            .as_text()
                            .unwrap()
                            .parse::<i64>()
                            .unwrap()
                    })
                    .sum();
                assert_eq!(
                    model.value(&total).unwrap().as_text().unwrap(),
                    sum.to_string()
                );
                let last = keys::item(group, params::ITEMS_PER_GROUP - 1);
                assert_eq!(model.value(&keys::spare(group)), model.value(&last));
                let entries = model.entries_under(&keys::items(group));
                let audited: i64 = entries
                    .iter()
                    .map(|(_, value)| value.as_text().unwrap().parse::<i64>().unwrap())
                    .sum();
                assert_eq!(
                    model.value(&keys::audit(group)).unwrap().as_text().unwrap(),
                    format!("{},{audited}", entries.len())
                );
            }
            for counter in 0..params::COUNTERS {
                assert_eq!(
                    model.value(&keys::counter(counter)).unwrap().as_counter(),
                    Some(params::COUNTER_START)
                );
            }
            for set in 0..params::SETS {
                assert!(model.value(&keys::set(set)).unwrap().as_set().is_some());
            }
            assert_eq!(genesis.policies.len(), params::POLICIES);
        }
    }

    #[test]
    fn a_case_has_the_preregistered_number_of_tasks_and_their_draws_are_in_range() {
        let case = Case::new(43, 5);
        let tasks = case.tasks();
        assert_eq!(tasks.len(), params::TASKS_PER_CASE);
        for task in &tasks {
            assert!(task.rely.is_none_or(|policy| policy < params::POLICIES));
            for attempt in 0..params::MAX_ATTEMPTS {
                assert!((params::THINK_MIN_TICKS..=params::THINK_MAX_TICKS)
                    .contains(&task.thinks[attempt]));
                assert!((params::REVIEW_MIN_TICKS..=params::REVIEW_MAX_TICKS)
                    .contains(&task.reviews[attempt]));
                assert!((0.0..=1.0).contains(&task.scores[attempt]));
            }
            match &task.program {
                Program::Rmw { group, item, delta }
                | Program::GroupTotal { group, item, delta } => {
                    assert!(*group < case.groups && *item < params::ITEMS_PER_GROUP);
                    assert!(*delta != 0 && delta.abs() <= params::RMW_DELTA_MAX);
                }
                Program::WriteSkew {
                    group,
                    first,
                    second,
                } => {
                    assert!(*group < case.groups);
                    assert!(
                        first != second
                            && *first < params::ITEMS_PER_GROUP
                            && *second < params::ITEMS_PER_GROUP
                    );
                }
                Program::CounterAdd { counter, amount } => {
                    assert!(*counter < params::COUNTERS);
                    assert!(*amount != 0 && amount.abs() <= params::COUNTER_ADD_MAX);
                }
                Program::SetOp { set, member, .. } => {
                    assert!(*set < params::SETS && *member < params::SET_MEMBERS);
                }
                Program::GuardedDecrement { counter } => assert!(*counter < params::COUNTERS),
                Program::InsertCapped { group, task: index } => {
                    assert!(*group < case.groups);
                    assert_eq!(*index, task.index);
                }
                Program::RemoveExtra { group } | Program::Audit { group } => {
                    assert!(*group < case.groups);
                }
            }
        }
    }

    #[test]
    fn the_program_mix_follows_the_preregistered_weights() {
        let mut counts = std::collections::BTreeMap::new();
        let draws = 40_000u64;
        let mut rng = stream(7, &[99]);
        for task in 0..draws as usize {
            let program = draw_program(&mut rng, 16, task);
            *counts.entry(program.name()).or_insert(0u64) += 1;
        }
        for (name, weight) in params::PROGRAMS
            .iter()
            .zip(params::PROGRAM_WEIGHTS_PERMILLE)
        {
            let expected = draws as f64 * weight as f64 / 1000.0;
            let observed = counts.get(name).copied().unwrap_or(0) as f64;
            let sigma =
                (draws as f64 * (weight as f64 / 1000.0) * (1.0 - weight as f64 / 1000.0)).sqrt();
            assert!(
                (observed - expected).abs() < 5.0 * sigma,
                "{name}: {observed} draws, {expected} expected"
            );
        }
    }

    #[test]
    fn the_background_actors_fire_at_their_preregistered_rates() {
        let case = Case::new(71, 2);
        let ticks = 200_000u64;
        let (mut ingress, mut external, mut lifecycle, mut rewire) = (0u64, 0u64, 0u64, 0u64);
        for tick in 0..ticks {
            for event in case.background(tick) {
                match event {
                    Background::Ingress { .. } => ingress += 1,
                    Background::External { group, item, value } => {
                        assert!(group < case.groups && item < params::ITEMS_PER_GROUP);
                        assert!((0..=100).contains(&value));
                        external += 1;
                    }
                    Background::Lifecycle { policy, .. } => {
                        assert!(policy < params::POLICIES);
                        lifecycle += 1;
                    }
                    Background::Rewire { group, swap_after } => {
                        assert!(group < case.groups);
                        assert!(
                            (params::REWIRE_SWAP_MIN_TICKS..=params::REWIRE_SWAP_MAX_TICKS)
                                .contains(&swap_after)
                        );
                        rewire += 1;
                    }
                }
            }
        }
        let close = |count: u64, permille: u64| {
            let expected = ticks as f64 * permille as f64 / 1000.0;
            let sigma = expected.sqrt();
            (count as f64 - expected).abs() < 5.0 * sigma
        };
        assert!(
            close(ingress, params::INGRESS_PERMILLE_PER_TICK),
            "ingress {ingress}"
        );
        assert!(
            close(external, params::EXTERNAL_WRITE_PERMILLE_PER_TICK),
            "external {external}"
        );
        assert!(
            close(lifecycle, params::LIFECYCLE_PERMILLE_PER_TICK),
            "lifecycle {lifecycle}"
        );
        assert!(
            close(rewire, params::REWIRE_PERMILLE_PER_TICK),
            "rewire {rewire}"
        );
    }

    #[test]
    fn a_request_id_is_a_valid_identifier_that_no_tick_repeats() {
        let case = Case::new(101, 4);
        let mut seen = BTreeSet::new();
        for tick in 0..5_000 {
            for event in case.background(tick) {
                if let Background::Ingress { request, .. } = event {
                    assert!(seen.insert(request));
                }
            }
        }
        assert!(!seen.is_empty());
    }
}
