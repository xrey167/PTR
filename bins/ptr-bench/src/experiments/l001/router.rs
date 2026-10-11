//! A deterministic message router for in-process Raft members: no sockets, no timers.
//!
//! A message is delivered when the router says so. It can be dropped (a member is down, or a
//! seeded fault takes it), delivered twice, or delivered out of order, all from one seeded
//! stream, so a schedule is reproducible from the seed alone. The router is deliberately dumb:
//! one queue and a set of members that hear nothing.

use super::super::rng::Rng;
use ptr_ledger::{RaftMessage, RaftNode};
use std::collections::{BTreeSet, VecDeque};

/// How often the router interferes, in per mille of messages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Faults {
    pub drop: u64,
    pub duplicate: u64,
    pub reorder: u64,
}

/// What the router did, counted as it did it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    pub delivered: i64,
    /// Dropped because the sender or the receiver was isolated.
    pub dropped_isolated: i64,
    /// Dropped by a seeded fault.
    pub dropped_fault: i64,
    pub duplicated: i64,
    pub reordered: i64,
}

impl Stats {
    pub fn dropped(&self) -> i64 {
        self.dropped_isolated + self.dropped_fault
    }
}

pub struct Router {
    /// Members whose messages are dropped in both directions.
    pub isolated: BTreeSet<u64>,
    pub faults: Faults,
    pub stats: Stats,
    rng: Rng,
}

/// The bound on one delivery run: a group that never goes quiet is a defect, reported as an
/// error and not waited for.
const MAX_STEPS: usize = 20_000;

impl Router {
    pub fn new(seed: u64) -> Self {
        Self {
            isolated: BTreeSet::new(),
            faults: Faults::default(),
            stats: Stats::default(),
            rng: Rng::new(seed),
        }
    }

    /// Deliver `initial` and everything it causes until nothing is in flight.
    pub fn settle(
        &mut self,
        nodes: &mut [RaftNode],
        initial: Vec<RaftMessage>,
    ) -> Result<(), String> {
        let mut queue: VecDeque<RaftMessage> = initial.into();
        for _ in 0..MAX_STEPS {
            if queue.is_empty() {
                return Ok(());
            }
            let position = if queue.len() > 1 && self.rng.below(1000) < self.faults.reorder {
                self.stats.reordered += 1;
                self.rng.index(queue.len())
            } else {
                0
            };
            let message = queue.remove(position).expect("position is in range");
            if self.isolated.contains(&message.from) || self.isolated.contains(&message.to) {
                self.stats.dropped_isolated += 1;
                continue;
            }
            if self.rng.below(1000) < self.faults.drop {
                self.stats.dropped_fault += 1;
                continue;
            }
            if self.rng.below(1000) < self.faults.duplicate {
                self.stats.duplicated += 1;
                queue.push_back(message.clone());
            }
            let Some(node) = nodes.iter_mut().find(|node| node.id() == message.to) else {
                return Err(format!(
                    "a message is addressed to unknown member {}",
                    message.to
                ));
            };
            self.stats.delivered += 1;
            // A member that cannot step a message is the product refusing it; the harness
            // treats the refusal as the message being lost.
            if let Ok(produced) = node.step(message) {
                queue.extend(produced);
            }
        }
        Err("the group did not go quiet".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_makes_the_same_choices() {
        let draw = |seed| {
            let mut router = Router::new(seed);
            router.faults = Faults {
                drop: 300,
                duplicate: 300,
                reorder: 300,
            };
            (0..64).map(|_| router.rng.below(1000)).collect::<Vec<_>>()
        };
        assert_eq!(draw(4), draw(4));
        assert_ne!(draw(4), draw(5));
    }

    #[test]
    fn an_empty_queue_is_quiet_and_stats_add_up() {
        let mut router = Router::new(1);
        assert_eq!(router.settle(&mut [], Vec::new()), Ok(()));
        let total = Stats {
            delivered: 2,
            dropped_isolated: 1,
            dropped_fault: 3,
            duplicated: 4,
            reordered: 5,
        };
        assert_eq!(total.dropped(), 4);
    }
}
