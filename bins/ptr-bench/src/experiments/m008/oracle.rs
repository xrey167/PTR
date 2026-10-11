//! The harness-owned truth: the current fact of each cue after the session's writes.
//!
//! Two independent readings share no code with each other or with any arm: [`Truth`] folds the
//! write list forwards (the table every arm is scored against) and [`naive_current`] scans it
//! backwards for the last write of the cue. The generator records a third reading in each query
//! (`expected`). A disagreement among the three is `oracle_label_mismatches`, a harness defect:
//! the aggregator judges such a seed invalid, never falsified.

use super::session::{Event, Session};
use std::collections::BTreeMap;

/// The current fact (pool row) of every cue written.
pub struct Truth {
    current: BTreeMap<i64, usize>,
}

impl Truth {
    pub fn fold(events: &[Event]) -> Self {
        let mut truth = Self {
            current: BTreeMap::new(),
        };
        for event in events {
            truth.current.insert(event.cue, event.row);
        }
        truth
    }

    /// The row of the current fact of `cue`, or `None` when the cue was never written.
    pub fn current(&self, cue: i64) -> Option<usize> {
        self.current.get(&cue).copied()
    }

    pub fn len(&self) -> usize {
        self.current.len()
    }
}

/// The last write of `cue`, found by scanning backwards.
pub fn naive_current(events: &[Event], cue: i64) -> Option<usize> {
    events
        .iter()
        .rev()
        .find(|event| event.cue == cue)
        .map(|event| event.row)
}

/// How many cues or queries the three readings disagree on.
pub fn mismatches(session: &Session, truth: &Truth) -> i64 {
    let mut count = 0;
    for &cue in &session.written {
        if truth.current(cue) != naive_current(&session.events, cue) {
            count += 1;
        }
    }
    for query in &session.queries {
        let table = truth.current(query.cue);
        if table != naive_current(&session.events, query.cue) || table != query.expected {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::super::params::Shape;
    use super::super::pool::tests::shared;
    use super::*;

    #[test]
    fn the_three_readings_agree_on_every_pilot_seed() {
        let pool = shared();
        for seed in super::super::params::PILOT_SEEDS {
            let session = Session::generate(pool, seed, Shape::PINNED).unwrap();
            let truth = Truth::fold(&session.events);
            assert_eq!(truth.len(), 400);
            assert_eq!(mismatches(&session, &truth), 0, "seed {seed}");
        }
    }

    #[test]
    fn a_wrong_table_is_counted() {
        let pool = shared();
        let session = Session::generate(pool, 1001, Shape::PINNED).unwrap();
        // First write wins: the table of a cue that was updated now names the stale fact.
        let mut wrong = Truth::fold(&session.events);
        let updated: Vec<Event> = session
            .events
            .iter()
            .filter(|e| e.version == 2)
            .copied()
            .collect();
        for event in &updated {
            wrong.current.insert(event.cue, pool.fact(event.cue, 1));
        }
        assert!(mismatches(&session, &wrong) >= 200);
    }
}
