//! Scoring an arm's answers against the oracle's table.
//!
//! One function scores every arm and every control, so a control that is not separated from the
//! arms shows the scoring or the corpus wrong. An answer is a ranked list of pool rows, best first;
//! the empty list is Unknown.

use super::oracle::Truth;
use super::params;
use super::pool::Pool;
use super::session::{QueryKind, Session};

#[derive(Clone, Debug, Default)]
pub struct Score {
    pub queries: i64,
    pub answerable: i64,
    pub unanswerable: i64,
    /// Answerable queries whose top fact is the current one.
    pub correct: i64,
    /// Answerable queries whose current fact is in the list.
    pub correct_in_list: i64,
    /// Queries answered with a wrong top fact on an answerable query.
    pub wrong: i64,
    /// Of those, the top fact was a stale version of the queried cue.
    pub stale_top1: i64,
    /// Unanswerable queries that got any answer.
    pub answered_unanswerable: i64,
    /// Answerable queries answered Unknown.
    pub unknown_answerable: i64,
    /// All queries answered Unknown.
    pub unknown: i64,
    pub tokens: i64,
    /// Queries whose list is longer than the token budget.
    pub budget_violations: i64,
    pub updated: i64,
    pub updated_correct: i64,
    pub old: i64,
    pub old_correct: i64,
    pub recent: i64,
    pub recent_correct: i64,
    /// Per answerable query, in query order: 1 when the top fact is the current one.
    pub hits: Vec<i64>,
}

impl Score {
    /// Wrong answers over all queries: a stale or other-cue top fact on an answerable query, or any
    /// answer to an unanswerable one.
    pub fn false_recall(&self) -> i64 {
        self.wrong + self.answered_unanswerable
    }
}

/// Whether the current fact of an answerable query was written before the recency window.
pub fn is_old(session: &Session, current_row: usize) -> bool {
    let index = session.write_index(current_row).unwrap_or(0);
    index + params::RECENCY_WINDOW < session.events.len()
}

pub fn score(pool: &Pool, session: &Session, truth: &Truth, answers: &[Vec<usize>]) -> Score {
    let mut result = Score::default();
    for (query, answer) in session.queries.iter().zip(answers) {
        result.queries += 1;
        let tokens: usize = answer.iter().map(|row| pool.row(*row).tokens).sum();
        result.tokens += tokens as i64;
        if tokens > params::TOKEN_BUDGET {
            result.budget_violations += 1;
        }
        if answer.is_empty() {
            result.unknown += 1;
        }
        if !query.kind.answerable() {
            result.unanswerable += 1;
            if !answer.is_empty() {
                result.answered_unanswerable += 1;
            }
            continue;
        }
        let Some(current) = truth.current(query.cue) else {
            // An answerable query about a cue the table does not know: the table is wrong, which
            // `oracle_label_mismatches` counts; here the query cannot be satisfied.
            result.answerable += 1;
            result.hits.push(0);
            continue;
        };
        result.answerable += 1;
        let old = is_old(session, current);
        let correct = answer.first() == Some(&current);
        let updated = query.kind == QueryKind::Updated;
        result.updated += i64::from(updated);
        result.old += i64::from(old);
        result.recent += i64::from(!old);
        if correct {
            result.correct += 1;
            result.updated_correct += i64::from(updated);
            result.old_correct += i64::from(old);
            result.recent_correct += i64::from(!old);
        }
        if answer.contains(&current) {
            result.correct_in_list += 1;
        }
        match answer.first() {
            None => result.unknown_answerable += 1,
            Some(top) if *top != current => {
                result.wrong += 1;
                let row = pool.row(*top);
                if row.cue == query.cue {
                    result.stale_top1 += 1;
                }
            }
            Some(_) => {}
        }
        result.hits.push(i64::from(correct));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::super::params::Shape;
    use super::super::pool::tests::shared;
    use super::*;

    fn setup() -> (&'static Pool, Session, Truth) {
        let pool = shared();
        let session = Session::generate(pool, 1001, Shape::PINNED).unwrap();
        let truth = Truth::fold(&session.events);
        (pool, session, truth)
    }

    #[test]
    fn the_oracle_answers_score_perfectly_and_a_stale_answer_is_false_recall() {
        let (pool, session, truth) = setup();
        let perfect: Vec<Vec<usize>> = session
            .queries
            .iter()
            .map(|q| {
                if q.kind.answerable() {
                    vec![truth.current(q.cue).unwrap()]
                } else {
                    Vec::new()
                }
            })
            .collect();
        let score = score(pool, &session, &truth, &perfect);
        assert_eq!((score.answerable, score.unanswerable), (320, 80));
        assert_eq!(score.correct, 320);
        assert_eq!(score.false_recall(), 0);
        assert_eq!(score.hits.iter().sum::<i64>(), 320);
        assert_eq!(score.updated, 160);
        assert_eq!(score.old + score.recent, 320);

        let stale_first: Vec<Vec<usize>> = session
            .queries
            .iter()
            .map(|q| match q.kind {
                QueryKind::Updated => vec![q.stale[0]],
                QueryKind::Stable => vec![truth.current(q.cue).unwrap()],
                _ => Vec::new(),
            })
            .collect();
        let stale = super::score(pool, &session, &truth, &stale_first);
        assert_eq!(stale.stale_top1, 160);
        assert_eq!(stale.wrong, 160);
        assert_eq!(stale.correct, 160);
        assert_eq!(stale.false_recall(), 160);
    }

    #[test]
    fn an_answer_to_an_unanswerable_query_and_a_long_list_are_counted() {
        let (pool, session, truth) = setup();
        let every_fact: Vec<usize> = (0..80).map(|i| session.events[i].row).collect();
        let answers: Vec<Vec<usize>> = session.queries.iter().map(|_| every_fact.clone()).collect();
        let result = score(pool, &session, &truth, &answers);
        assert_eq!(result.answered_unanswerable, 80);
        assert_eq!(
            result.budget_violations, 400,
            "eighty facts exceed 512 tokens"
        );
        assert_eq!(result.unknown, 0);
    }
}
