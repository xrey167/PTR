//! The schedule of a seed: one [`CasePlan`] per case, drawn from the seed alone.
//!
//! A case is one run of `rounds` Pod requests ending in an action of `effect`, plus at most one
//! writer that acts at a [`Position`] of the run's life. The same plan is executed in both
//! arms. The labels here (is the run stale by schedule, must arm A1 cancel it) are the
//! schedule's reading of its own intent; the mirror in `oracle` is the independent count they
//! are cross-checked with.

use super::params::{self, Position};
use crate::experiments::rng::Rng;
use ptr_types::Effect;

/// What the writer does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Writer {
    /// A host write of a key nobody wrote: the revision moves.
    Bump,
    /// A host write of a key to the value it holds: the revision does not move.
    NoOp,
    /// The target's generation is superseded (no semantic revision moves).
    Supersede,
    /// The target's generation 1 is revoked (no semantic revision moves).
    Revoke,
}

impl Writer {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bump => "bump",
            Self::NoOp => "noop",
            Self::Supersede => "supersede",
            Self::Revoke => "revoke",
        }
    }

    pub fn moves_revision(self) -> bool {
        matches!(self, Self::Bump)
    }

    pub fn kills_generation(self) -> bool {
        matches!(self, Self::Supersede | Self::Revoke)
    }
}

/// The class of a case, for the mix and the tables.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Class {
    Never,
    BumpEarly,
    BumpLate,
    NoOp,
    Lifecycle,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::BumpEarly => "bump-early",
            Self::BumpLate => "bump-late",
            Self::NoOp => "noop",
            Self::Lifecycle => "lifecycle",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Injection {
    pub writer: Writer,
    pub position: Position,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CasePlan {
    pub index: usize,
    pub seed: u64,
    pub rounds: u64,
    pub effect: Effect,
    pub class: Class,
    pub injection: Option<Injection>,
}

impl CasePlan {
    /// Whether the schedule says the run is stale when the writer has acted: the revision
    /// moved under it.
    pub fn stale_by_schedule(&self) -> bool {
        self.injection
            .is_some_and(|injection| injection.writer.moves_revision())
    }

    /// Whether the schedule says the action cannot be executed on the generation it names.
    pub fn generation_dead_by_schedule(&self) -> bool {
        self.injection
            .is_some_and(|injection| injection.writer.kills_generation())
    }

    /// Whether the schedule leaves the run and its action fresh to the end.
    pub fn fresh_by_schedule(&self) -> bool {
        matches!(self.class, Class::Never | Class::NoOp)
    }

    /// The case's key for the at-most-once guarantee.
    pub fn key(&self, arm: &str) -> String {
        format!("s002-{}-{}-{arm}", self.seed, self.index)
    }
}

fn position_among(rng: &mut Rng, positions: &[Position]) -> Position {
    *rng.pick(positions)
}

/// The plan of case `index` of `seed`.
pub fn plan_of(seed: u64, index: usize) -> CasePlan {
    let mut rng = Rng::new(seed ^ 0x5002_5EED_0000_0000).fork(index as u64);
    let rounds = rng.range(params::ROUNDS_MIN, params::ROUNDS_MAX);
    let effect = params::EFFECTS[index % params::EFFECTS.len()];
    let draw = rng.below(1000);
    let mut edge = params::CLASS_NEVER_PERMILLE;
    let class = if draw < edge {
        Class::Never
    } else {
        edge += params::CLASS_BUMP_EARLY_PERMILLE;
        if draw < edge {
            Class::BumpEarly
        } else {
            edge += params::CLASS_BUMP_LATE_PERMILLE;
            if draw < edge {
                Class::BumpLate
            } else {
                edge += params::CLASS_NOOP_PERMILLE;
                if draw < edge {
                    Class::NoOp
                } else {
                    Class::Lifecycle
                }
            }
        }
    };
    let early = [
        Position::BeforeRound1,
        Position::MidRun,
        Position::JustBeforeAction,
    ];
    let late = [
        Position::BetweenActionAndAuthorize,
        Position::BetweenPrepareAndExecute,
    ];
    let injection = match class {
        Class::Never => None,
        Class::BumpEarly => Some(Injection {
            writer: Writer::Bump,
            position: position_among(&mut rng, &early),
        }),
        Class::BumpLate => Some(Injection {
            writer: Writer::Bump,
            position: position_among(&mut rng, &late),
        }),
        Class::NoOp => Some(Injection {
            writer: Writer::NoOp,
            position: position_among(&mut rng, &Position::ALL),
        }),
        Class::Lifecycle => {
            let writer = if rng.below(1000) < params::LIFECYCLE_REVOKE_PERMILLE {
                Writer::Revoke
            } else {
                Writer::Supersede
            };
            Some(Injection {
                writer,
                position: position_among(&mut rng, &Position::ALL),
            })
        }
    };
    CasePlan {
        index,
        seed,
        rounds,
        effect,
        class,
        injection,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn a_plan_depends_on_the_seed_and_the_index_alone() {
        for index in [0, 1, 17, 1499] {
            assert_eq!(plan_of(17, index), plan_of(17, index));
        }
        assert_ne!(
            (0..200).map(|i| plan_of(17, i)).collect::<Vec<_>>(),
            (0..200).map(|i| plan_of(29, i)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_mix_follows_the_pinned_permille_within_sampling_error() {
        for seed in params::SEEDS {
            let mut counts = std::collections::BTreeMap::new();
            for index in 0..params::CASES_PER_SEED {
                *counts.entry(plan_of(seed, index).class).or_insert(0usize) += 1;
            }
            let pinned = [
                (Class::Never, params::CLASS_NEVER_PERMILLE),
                (Class::BumpEarly, params::CLASS_BUMP_EARLY_PERMILLE),
                (Class::BumpLate, params::CLASS_BUMP_LATE_PERMILLE),
                (Class::NoOp, params::CLASS_NOOP_PERMILLE),
                (Class::Lifecycle, params::CLASS_LIFECYCLE_PERMILLE),
            ];
            for (class, permille) in pinned {
                let expected = params::CASES_PER_SEED * permille as usize / 1000;
                let actual = counts[&class];
                assert!(
                    actual.abs_diff(expected) * 5 <= expected,
                    "seed {seed} {class:?}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn every_position_and_every_effect_and_both_lifecycle_writers_occur_in_every_seed() {
        for seed in params::SEEDS {
            let plans: Vec<CasePlan> = (0..params::CASES_PER_SEED)
                .map(|i| plan_of(seed, i))
                .collect();
            let positions: BTreeSet<Position> = plans
                .iter()
                .filter_map(|p| p.injection.map(|i| i.position))
                .collect();
            assert_eq!(positions.len(), Position::ALL.len(), "seed {seed}");
            let effects: BTreeSet<&str> = plans
                .iter()
                .map(|p| params::effect_name(p.effect))
                .collect();
            assert_eq!(effects.len(), 4);
            let writers: BTreeSet<&str> = plans
                .iter()
                .filter_map(|p| p.injection.map(|i| i.writer.name()))
                .collect();
            assert_eq!(writers.len(), 4);
            // The three early positions each occur for the bumping writer.
            for position in [
                Position::BeforeRound1,
                Position::MidRun,
                Position::JustBeforeAction,
            ] {
                assert!(
                    plans.iter().any(|p| p.injection
                        == Some(Injection {
                            writer: Writer::Bump,
                            position
                        })),
                    "seed {seed} {position:?}"
                );
            }
        }
    }

    #[test]
    fn the_rounds_are_in_range_and_a_mid_run_writer_has_a_round_to_interrupt() {
        for seed in params::SEEDS {
            for index in 0..params::CASES_PER_SEED {
                let plan = plan_of(seed, index);
                assert!((params::ROUNDS_MIN..=params::ROUNDS_MAX).contains(&plan.rounds));
                if plan.injection.map(|i| i.position) == Some(Position::MidRun) {
                    assert!(plan.rounds >= 2);
                }
            }
        }
    }

    #[test]
    fn the_schedule_labels_follow_the_class() {
        for index in 0..400 {
            let plan = plan_of(43, index);
            match plan.class {
                Class::Never | Class::NoOp => {
                    assert!(plan.fresh_by_schedule() && !plan.stale_by_schedule());
                }
                Class::BumpEarly => {
                    assert!(
                        plan.stale_by_schedule() && plan.injection.unwrap().position.is_early()
                    );
                }
                Class::BumpLate => {
                    assert!(
                        plan.stale_by_schedule() && !plan.injection.unwrap().position.is_early()
                    );
                }
                Class::Lifecycle => {
                    assert!(plan.generation_dead_by_schedule() && !plan.stale_by_schedule());
                }
            }
        }
    }

    #[test]
    fn the_keys_are_valid_identifiers_that_differ_by_case_and_arm() {
        let a = plan_of(17, 3).key("a0");
        assert_eq!(a, "s002-17-3-a0");
        assert_ne!(a, plan_of(17, 3).key("a1"));
        assert_ne!(a, plan_of(17, 4).key("a0"));
    }
}
