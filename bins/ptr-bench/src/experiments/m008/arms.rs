//! What every arm shares (a ranked candidate list and the abstention rule), the recency buffer and
//! the three negative controls.
//!
//! An arm answers a query with a ranked list of at most `RANKED_LIMIT` facts (pool rows), best
//! first; the empty list is Unknown. The arms that decode or rank first produce a [`Ranked`] that
//! does not depend on any threshold, and a [`Policy`] turns it into the answer. That is how the
//! operating points are set on the pilot seeds (`calibrate.rs`) and applied unchanged on the
//! confirmatory ones.

use super::oracle::Truth;
use super::params;
use super::pool::{cosine, Pool};
use super::session::{QueryKind, Session};
use crate::experiments::rng::Rng;

/// The candidates of one arm for one query, before any abstention: the top `RANKED_LIMIT`, best
/// first, each with the arm's own score, and the score of the runner-up over all candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct Ranked {
    pub rows: Vec<usize>,
    pub scores: Vec<f32>,
    pub runner_up: f32,
}

impl Ranked {
    pub fn empty() -> Self {
        Self {
            rows: Vec::new(),
            scores: Vec::new(),
            runner_up: 0.0,
        }
    }
}

/// When a ranked list is confident enough to be an answer: the rule of
/// `ptr_fastmem::decode_readout`, which every arm uses at its own pinned thresholds. The best
/// score must reach `min_score` and lead the runner-up by `min_margin`; only candidates that
/// reach `min_score` are kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Policy {
    pub min_score: f32,
    pub min_margin: f32,
}

impl Policy {
    /// No abstention: every candidate is kept whatever its score (the descriptive reading).
    pub fn unthresholded() -> Self {
        Self {
            min_score: -2.0,
            min_margin: -2.0,
        }
    }

    pub fn milli(min_score_milli: i64, min_margin_milli: i64) -> Self {
        Self {
            min_score: min_score_milli as f32 / 1000.0,
            min_margin: min_margin_milli as f32 / 1000.0,
        }
    }

    /// Why the list is Unknown under this policy, or `None` when it is an answer.
    pub fn refusal(&self, ranked: &Ranked) -> Option<Refusal> {
        let best = *ranked.scores.first()?;
        if best < self.min_score {
            Some(Refusal::Score)
        } else if best - ranked.runner_up < self.min_margin {
            Some(Refusal::Margin)
        } else {
            None
        }
    }

    /// The answer: the candidates that reach `min_score`, or nothing (Unknown).
    pub fn apply(&self, ranked: &Ranked) -> Vec<usize> {
        if ranked.rows.is_empty() || self.refusal(ranked).is_some() {
            return Vec::new();
        }
        ranked
            .rows
            .iter()
            .zip(&ranked.scores)
            .filter(|(_, score)| **score >= self.min_score)
            .map(|(row, _)| *row)
            .take(params::RANKED_LIMIT)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Refusal {
    Score,
    Margin,
}

/// The recency buffer: the last `window` writes, answered by the cosine between the query and
/// each buffered fact on the frozen vectors, the newest first on a tie.
///
/// The window is `RECENCY_WINDOW` writes (the budget buys that many facts of the longest length).
pub fn recency(pool: &Pool, session: &Session) -> (Vec<Ranked>, usize) {
    let window = params::RECENCY_WINDOW;
    let start = session.events.len().saturating_sub(window);
    let buffered = &session.events[start..];
    let ranked = session
        .queries
        .iter()
        .map(|query| {
            let vector = pool.vector(query.row);
            let mut scored: Vec<(f32, usize, usize)> = buffered
                .iter()
                .enumerate()
                .map(|(age, event)| {
                    let score = cosine(vector, pool.vector(event.row)) as f32;
                    (score, age, event.row)
                })
                .collect();
            // Best cosine first; on a tie the newer write (the larger index) first.
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
            let runner_up = scored.get(1).map_or(0.0, |entry| entry.0);
            scored.truncate(params::RANKED_LIMIT);
            Ranked {
                rows: scored.iter().map(|entry| entry.2).collect(),
                scores: scored.iter().map(|entry| entry.0).collect(),
                runner_up,
            }
        })
        .collect();
    (ranked, buffered.len())
}

/// The three negative controls, each an answer list per query. None of them runs code under
/// test; they show the scoring and the corpus can tell a perfect arm, a blind one and a
/// stale-answering one apart.
pub struct Controls {
    /// Answers the current fact of every answerable query and abstains on the others.
    pub oracle_cheat: Vec<Vec<usize>>,
    /// Answers a random current fact of the session to every query.
    pub random: Vec<Vec<usize>>,
    /// Answers the oldest version of the cue when it has several, a random current fact otherwise.
    pub stale_first: Vec<Vec<usize>>,
}

pub fn controls(session: &Session, truth: &Truth, seed: u64) -> Controls {
    let mut rng = Rng::new(seed ^ 0x4d30_3038_c0de_c0de);
    let live: Vec<usize> = session
        .written
        .iter()
        .filter_map(|cue| truth.current(*cue))
        .collect();
    let mut oracle_cheat = Vec::new();
    let mut random = Vec::new();
    let mut stale_first = Vec::new();
    for query in &session.queries {
        let current = if query.kind.answerable() {
            truth.current(query.cue)
        } else {
            None
        };
        oracle_cheat.push(current.map(|row| vec![row]).unwrap_or_default());
        random.push(vec![*rng.pick(&live)]);
        let stale = match (query.kind, query.stale.first()) {
            (QueryKind::Updated, Some(row)) => *row,
            _ => *rng.pick(&live),
        };
        stale_first.push(vec![stale]);
    }
    Controls {
        oracle_cheat,
        random,
        stale_first,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked(scores: &[f32], runner_up: f32) -> Ranked {
        Ranked {
            rows: (0..scores.len()).collect(),
            scores: scores.to_vec(),
            runner_up,
        }
    }

    #[test]
    fn the_policy_refuses_by_score_then_by_margin_and_keeps_confident_candidates() {
        let policy = Policy::milli(300, 100);
        assert_eq!(
            policy.refusal(&ranked(&[0.2, 0.1], 0.1)),
            Some(Refusal::Score)
        );
        assert_eq!(
            policy.refusal(&ranked(&[0.5, 0.45], 0.45)),
            Some(Refusal::Margin)
        );
        assert_eq!(policy.refusal(&ranked(&[0.5, 0.2], 0.2)), None);
        assert_eq!(policy.apply(&ranked(&[0.5, 0.35, 0.2], 0.35)), vec![0, 1]);
        assert!(policy.apply(&ranked(&[0.5, 0.45], 0.45)).is_empty());
        assert!(policy.apply(&Ranked::empty()).is_empty());
        // A single candidate leads the zero runner-up.
        assert_eq!(policy.apply(&ranked(&[0.4], 0.0)), vec![0]);
    }

    #[test]
    fn a_zero_margin_policy_never_refuses_by_margin() {
        let policy = Policy::milli(0, 0);
        assert_eq!(policy.refusal(&ranked(&[0.5, 0.5], 0.5)), None);
    }
}
