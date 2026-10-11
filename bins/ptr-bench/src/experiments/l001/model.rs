//! The harness's own reading of "revoked", and the seeded workloads it judges.
//!
//! [`Model`] is the oracle: it holds nothing but the history of events the harness fed it and
//! answers `authorizable` and `validity` from that history alone. It never reads a ledger, a
//! runtime or any other state of the code under test, and shares no code with it: the product's
//! answer comes from `PtrRuntime::replay` and `authorize_action` (`images`/`arms` compare the two).
//! [`Ev`] is the harness's own event type; only [`Ev::to_ledger`] touches the product's.

use super::super::rng::Rng;
use ptr_ledger::LedgerEvent;
use ptr_types::{CapsuleId, Generation, ProjectId};
use std::collections::{BTreeMap, BTreeSet};

/// The project every workload commits its capsules under.
pub const PROJECT: &str = "l001";

/// One lifecycle event of a workload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ev {
    Commit { subject: String, generation: u64 },
    Supersede { subject: String, old: u64, new: u64 },
    Revoke { subject: String, generation: u64 },
}

impl Ev {
    pub fn to_ledger(&self) -> LedgerEvent {
        match self {
            Ev::Commit {
                subject,
                generation,
            } => LedgerEvent::CapsuleCommitted {
                project: ProjectId::from(PROJECT),
                capsule: CapsuleId(subject.clone()),
                generation: Generation(*generation),
            },
            Ev::Supersede { subject, old, new } => LedgerEvent::CapsuleSuperseded {
                capsule: CapsuleId(subject.clone()),
                old: Generation(*old),
                new: Generation(*new),
            },
            Ev::Revoke {
                subject,
                generation,
            } => LedgerEvent::Revoked {
                subject: subject.clone(),
                generation: Generation(*generation),
            },
        }
    }

    pub fn is_revocation(&self) -> bool {
        matches!(self, Ev::Revoke { .. })
    }
}

/// What a generation of a subject is, as the harness reads a history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Standing {
    Live,
    Superseded,
    Revoked,
    /// Never produced by the oracle: a runtime that reports a disputed generation disagrees.
    Disputed,
}

/// The oracle: the generation each subject lives at, and every revoked pair.
#[derive(Clone, Debug, Default)]
pub struct Model {
    live: BTreeMap<String, u64>,
    revoked: BTreeSet<(String, u64)>,
}

impl Model {
    pub fn of(history: &[Ev]) -> Self {
        let mut model = Self::default();
        history.iter().for_each(|event| model.apply(event));
        model
    }

    pub fn apply(&mut self, event: &Ev) {
        match event {
            Ev::Commit {
                subject,
                generation,
            } => {
                self.live.insert(subject.clone(), *generation);
            }
            Ev::Supersede { subject, new, .. } => {
                self.live.insert(subject.clone(), *new);
            }
            Ev::Revoke {
                subject,
                generation,
            } => {
                self.revoked.insert((subject.clone(), *generation));
            }
        }
    }

    /// An action on `subject` at `generation` may proceed: it is the live generation and no
    /// revocation names it.
    pub fn authorizable(&self, subject: &str, generation: u64) -> bool {
        self.live.get(subject) == Some(&generation)
            && !self.revoked.contains(&(subject.to_owned(), generation))
    }

    /// The standing of a generation; `None` for one the history never made live.
    pub fn standing(&self, subject: &str, generation: u64) -> Option<Standing> {
        if self.revoked.contains(&(subject.to_owned(), generation)) {
            return Some(Standing::Revoked);
        }
        match self.live.get(subject) {
            Some(live) if *live == generation => Some(Standing::Live),
            Some(live) if *live > generation => Some(Standing::Superseded),
            _ => None,
        }
    }
}

/// The oracle against a table written by hand, run at the start of every run. A history of
/// three subjects (a revoke-then-recommit, a supersession through generation 3 and a revoked
/// later generation) and what a correct reading of each prefix says. A disagreement is a defect
/// of the harness, not of the product: the run ends with status 2.
pub fn self_check() -> Result<(), String> {
    let commit = |s: &str, g: u64| Ev::Commit {
        subject: s.into(),
        generation: g,
    };
    let revoke = |s: &str, g: u64| Ev::Revoke {
        subject: s.into(),
        generation: g,
    };
    let supersede = |s: &str, old: u64, new: u64| Ev::Supersede {
        subject: s.into(),
        old,
        new,
    };
    let history = [
        commit("a", 1),
        revoke("a", 1),
        commit("a", 2),
        commit("b", 1),
        supersede("b", 1, 2),
        supersede("b", 2, 3),
        commit("c", 1),
        commit("c", 4),
        revoke("c", 4),
        revoke("c", 3),
    ];
    // (events applied, subject, generation, authorizable, standing)
    let table = [
        (1, "a", 1, true, Some(Standing::Live)),
        (2, "a", 1, false, Some(Standing::Revoked)),
        (3, "a", 1, false, Some(Standing::Revoked)),
        (3, "a", 2, true, Some(Standing::Live)),
        (3, "a", 3, false, None),
        (5, "b", 1, false, Some(Standing::Superseded)),
        (5, "b", 2, true, Some(Standing::Live)),
        (6, "b", 2, false, Some(Standing::Superseded)),
        (6, "b", 3, true, Some(Standing::Live)),
        (6, "b", 4, false, None),
        (8, "c", 1, false, Some(Standing::Superseded)),
        (8, "c", 4, true, Some(Standing::Live)),
        (9, "c", 4, false, Some(Standing::Revoked)),
        (10, "c", 3, false, Some(Standing::Revoked)),
        (10, "c", 4, false, Some(Standing::Revoked)),
        (10, "d", 1, false, None),
    ];
    for (applied, subject, generation, authorizable, standing) in table {
        let model = Model::of(&history[..applied]);
        if model.authorizable(subject, generation) != authorizable
            || model.standing(subject, generation) != standing
        {
            return Err(format!(
                "the oracle disagrees with its hand-written table on ({subject}, {generation}) after {applied} events: it says authorizable={} standing={:?}",
                model.authorizable(subject, generation),
                model.standing(subject, generation)
            ));
        }
    }
    Ok(())
}

/// A seeded history and the pairs every judgement ranges over.
#[derive(Clone, Debug)]
pub struct Workload {
    pub events: Vec<Ev>,
    /// Every (subject, generation) from 1 to one past the highest generation the history used.
    pub universe: Vec<(String, u64)>,
    pub subjects: usize,
}

impl Workload {
    /// Whether some subject is revoked at a generation and later made live at a higher one.
    pub fn has_revoke_then_recommit(&self) -> bool {
        let mut revoked: BTreeMap<&str, u64> = BTreeMap::new();
        for event in &self.events {
            match event {
                Ev::Revoke {
                    subject,
                    generation,
                } => {
                    let slot = revoked.entry(subject).or_insert(0);
                    *slot = (*slot).max(*generation);
                }
                Ev::Commit {
                    subject,
                    generation: new,
                }
                | Ev::Supersede { subject, new, .. } => {
                    if revoked.get(subject.as_str()).is_some_and(|r| new > r) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// A workload over an explicit history: the universe is every pair up to one past the
    /// highest generation each subject reaches.
    pub fn from_events(events: Vec<Ev>) -> Self {
        let mut highest: BTreeMap<String, u64> = BTreeMap::new();
        for event in &events {
            let (subject, generation) = match event {
                Ev::Commit {
                    subject,
                    generation,
                } => (subject, *generation),
                Ev::Supersede { subject, new, .. } => (subject, *new),
                Ev::Revoke {
                    subject,
                    generation,
                } => (subject, *generation),
            };
            let slot = highest.entry(subject.clone()).or_insert(0);
            *slot = (*slot).max(generation);
        }
        let universe = highest
            .iter()
            .flat_map(|(subject, top)| (1..=top + 1).map(move |g| (subject.clone(), g)))
            .collect();
        Self {
            subjects: highest.len(),
            events,
            universe,
        }
    }

    /// The history cut to its first `count` events.
    pub fn prefix(&self, count: usize) -> &[Ev] {
        &self.events[..count]
    }
}

/// Generate one workload: 6 to 14 events over 3 or 4 subjects. Every history holds the
/// revoke-then-recommit triple of subject 0 (commit, revoke that generation, commit a higher
/// one), and the other slots are random events valid under the history so far.
pub fn generate(rng: &mut Rng, min_events: usize, max_events: usize) -> Workload {
    let subjects = 3 + rng.index(2);
    let count = rng.range(min_events as u64, max_events as u64) as usize;
    // Three distinct slots, in order, for subject 0's triple.
    let mut slots: Vec<usize> = Vec::new();
    while slots.len() < 3 {
        let slot = rng.index(count);
        if !slots.contains(&slot) {
            slots.push(slot);
        }
    }
    slots.sort_unstable();
    let name = |k: usize| format!("capsule:l001:{k}");
    let mut live: BTreeMap<String, u64> = BTreeMap::new();
    let mut highest: BTreeMap<String, u64> = BTreeMap::new();
    let mut events: Vec<Ev> = Vec::new();
    for slot in 0..count {
        let event = if slot == slots[0] || slot == slots[2] {
            let subject = name(0);
            let generation = highest.get(&subject).copied().unwrap_or(0) + 1;
            Ev::Commit {
                subject,
                generation,
            }
        } else if slot == slots[1] {
            let subject = name(0);
            let generation = *live.get(&subject).expect("the first slot made it live");
            Ev::Revoke {
                subject,
                generation,
            }
        } else {
            let subject = name(rng.index(subjects));
            let next = highest.get(&subject).copied().unwrap_or(0) + 1;
            let roll = rng.below(1000);
            match live.get(&subject).copied() {
                Some(current) if roll < 350 => Ev::Supersede {
                    subject,
                    old: current,
                    new: next,
                },
                Some(current) if roll < 700 => Ev::Revoke {
                    subject,
                    generation: if rng.chance(0.25) && current > 1 {
                        current - 1
                    } else {
                        current
                    },
                },
                _ => Ev::Commit {
                    subject,
                    generation: next,
                },
            }
        };
        match &event {
            Ev::Commit {
                subject,
                generation,
            } => {
                live.insert(subject.clone(), *generation);
                highest.insert(subject.clone(), *generation);
            }
            Ev::Supersede { subject, new, .. } => {
                live.insert(subject.clone(), *new);
                highest.insert(subject.clone(), *new);
            }
            Ev::Revoke { .. } => {}
        }
        events.push(event);
    }
    let universe = (0..subjects)
        .map(name)
        .flat_map(|subject| {
            let top = highest.get(&subject).copied().unwrap_or(0) + 1;
            (1..=top).map(move |g| (subject.clone(), g))
        })
        .collect();
    Workload {
        events,
        universe,
        subjects,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(subject: &str, generation: u64) -> Ev {
        Ev::Commit {
            subject: subject.into(),
            generation,
        }
    }

    fn revoke(subject: &str, generation: u64) -> Ev {
        Ev::Revoke {
            subject: subject.into(),
            generation,
        }
    }

    /// The oracle against a hand-written table: a revoked generation is never authorizable,
    /// a recommit at a higher generation is, and the standings follow.
    #[test]
    fn the_oracle_answers_a_table_written_by_hand() {
        let history = [
            commit("a", 1),
            revoke("a", 1),
            commit("a", 2),
            commit("b", 1),
            Ev::Supersede {
                subject: "b".into(),
                old: 1,
                new: 2,
            },
        ];
        let model = Model::of(&history);
        let table = [
            ("a", 1, false, Some(Standing::Revoked)),
            ("a", 2, true, Some(Standing::Live)),
            ("a", 3, false, None),
            ("b", 1, false, Some(Standing::Superseded)),
            ("b", 2, true, Some(Standing::Live)),
            ("c", 1, false, None),
        ];
        for (subject, generation, authorizable, standing) in table {
            assert_eq!(model.authorizable(subject, generation), authorizable);
            assert_eq!(model.standing(subject, generation), standing);
        }
        assert!(!Model::of(&history[..2]).authorizable("a", 1));
        assert!(Model::of(&history[..1]).authorizable("a", 1));
    }

    #[test]
    fn the_oracle_passes_its_own_table_every_run() {
        assert_eq!(self_check(), Ok(()));
    }

    #[test]
    fn every_workload_is_in_bounds_has_a_recommit_and_is_reproducible() {
        for seed in 0..200 {
            let mut rng = Rng::new(seed);
            let workload = generate(&mut rng, 6, 14);
            assert!((6..=14).contains(&workload.events.len()));
            assert!((3..=4).contains(&workload.subjects));
            assert!(workload.has_revoke_then_recommit(), "seed {seed}");
            assert!(workload.events.iter().any(Ev::is_revocation));
            let again = generate(&mut Rng::new(seed), 6, 14);
            assert_eq!(workload.events, again.events);
            assert!(!workload.universe.is_empty());
        }
    }
}
