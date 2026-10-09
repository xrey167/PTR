//! Descriptive evidence, never judged by a rule: what the arms do without any abstention, how the
//! memory fares when asked with the cue text it was keyed by, how recall changes with the number of
//! cues written (the capacity table), and a sweep of the score threshold.
//!
//! It is part of every confirmatory run so that it is frozen with it. The matched-operating-point
//! numbers say how the arms compare at one threshold; these say whether the comparison depends on
//! that threshold and in which regime the memory holds its cues at all.

use super::arms::{self, Policy, Ranked};
use super::fast::{self, FastConfig};
use super::hybrid;
use super::metrics::permille;
use super::oracle::Truth;
use super::params::{self, Shape};
use super::pool::Pool;
use super::score;
use super::session::Session;
use crate::experiments::layered_kv::emit::{MetricList, Tally};
use crate::experiments::rag_bridge::Bridge;

/// Recall@1 in permille of the answerable queries when each query's ranked candidates are taken
/// as they are (no abstention): the top candidate is the answer.
pub fn unthresholded_recall(
    pool: &Pool,
    session: &Session,
    truth: &Truth,
    ranked: &[Ranked],
) -> i64 {
    let policy = Policy::unthresholded();
    let answers: Vec<Vec<usize>> = ranked.iter().map(|r| policy.apply(r)).collect();
    let result = score::score(pool, session, truth, &answers);
    permille(result.correct, result.answerable)
}

/// Recall@1 of the memory when asked with the cue text itself, in permille of the answerable queries.
pub fn exact_cue_recall(session: &Session, truth: &Truth, tops: &[Option<usize>]) -> i64 {
    let (mut right, mut total) = (0, 0);
    for (query, top) in session.queries.iter().zip(tops) {
        if query.kind.answerable() {
            total += 1;
            if *top == truth.current(query.cue) && top.is_some() {
                right += 1;
            }
        }
    }
    permille(right, total)
}

/// Recall@1 and the share of unanswerable queries answered at each score threshold of the sweep
/// (no margin), for the arm `name`.
pub fn sweep(
    pool: &Pool,
    session: &Session,
    truth: &Truth,
    name: &str,
    ranked: &[Ranked],
    tally: &mut Tally,
) {
    for threshold in params::SWEEP_THRESHOLDS_MILLI {
        let policy = Policy::milli(threshold, 0);
        let answers: Vec<Vec<usize>> = ranked.iter().map(|r| policy.apply(r)).collect();
        let result = score::score(pool, session, truth, &answers);
        tally.set(
            &format!("sweep_recall_at_1_permille.{name}_t{threshold:03}"),
            permille(result.correct, result.answerable),
        );
        tally.set(
            &format!("sweep_unanswerable_answered_permille.{name}_t{threshold:03}"),
            permille(result.answered_unanswerable, result.unanswerable),
        );
    }
}

/// The capacity table: unthresholded recall@1 of the memory (asked with a paraphrase and with the
/// cue text), of the recency buffer and of live hybrid retrieval on sessions of
/// `CAPACITY_CASES` cues drawn from the run's seed. The 400-cue point is the run itself.
pub fn capacity(
    pool: &Pool,
    seed: u64,
    bridge: &mut Bridge,
    tally: &mut Tally,
) -> Result<(), String> {
    for cases in params::CAPACITY_CASES {
        let session = Session::generate(pool, seed, Shape::of_cases(cases)?)?;
        let truth = Truth::fold(&session.events);
        let mut scratch = Tally::new(MetricList {
            names: &[],
            grouped: &[],
        });
        let run = fast::run(
            pool,
            &session,
            &FastConfig::pinned(),
            Policy::unthresholded(),
            &mut scratch,
        );
        let fast_ranked: Vec<Ranked> = run.queries.iter().map(|q| q.ranked.clone()).collect();
        let tops: Vec<Option<usize>> = run.queries.iter().map(|q| q.exact_cue_top).collect();
        let (recency_ranked, _) = arms::recency(pool, &session);
        let live = hybrid::run(bridge, pool, &session, hybrid::Index::Live)
            .map_err(|error| error.to_string())?;
        for (arm, value) in [
            (
                "fast",
                unthresholded_recall(pool, &session, &truth, &fast_ranked),
            ),
            ("fast_exact_cue", exact_cue_recall(&session, &truth, &tops)),
            (
                "recency",
                unthresholded_recall(pool, &session, &truth, &recency_ranked),
            ),
            (
                "hybrid_live",
                unthresholded_recall(pool, &session, &truth, &live.queries),
            ),
        ] {
            tally.set(
                &format!("capacity_recall_at_1_permille.{arm}_c{cases:03}"),
                value,
            );
        }
    }
    Ok(())
}
