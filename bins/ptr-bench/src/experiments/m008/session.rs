//! The seeded session: which cues are written, which are updated and when, and the queries.
//!
//! A cue is an entity x attribute pair of the pool. A session writes `written` cues (version 1),
//! then, at later random moments, updates `updated` of them (version 2: a near-duplicate fact whose
//! value differs). It then asks `queries` questions, each a paraphrase (1..=3) of a cue, never the
//! cue text itself (paraphrase 0), which is what a write is keyed by:
//!
//! - updated cues (40%): the right answer is the version-2 fact, the version-1 fact is stale;
//! - never-updated cues (40%): the right answer is the version-1 fact;
//! - unwritten cues (10%): a question about a cue of the pool nobody wrote, often about an entity or
//!   an attribute that other written cues share: the right answer is Unknown;
//! - off-topic (10%): a question about an entity no cue of the pool mentions: Unknown.
//!
//! The generator also records what it believes the right answer is (`expected`), which the oracle
//! (`oracle.rs`) reads independently and the harness compares.

use super::params::{self, Shape};
use super::pool::Pool;
use crate::experiments::rng::Rng;

/// One write of the session, in order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Event {
    pub cue: i64,
    pub version: usize,
    /// The pool row of the fact written.
    pub row: usize,
    /// The pool row of the cue text (paraphrase 0) the memory is keyed by.
    pub key_row: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryKind {
    Updated,
    Stable,
    Unwritten,
    Offtopic,
}

impl QueryKind {
    pub fn answerable(self) -> bool {
        matches!(self, Self::Updated | Self::Stable)
    }
}

#[derive(Clone, Debug)]
pub struct Query {
    pub kind: QueryKind,
    /// The pool row of the question.
    pub row: usize,
    /// The cue asked about, `-1` for an off-topic question.
    pub cue: i64,
    /// What the generator believes the current fact is: the pool row, or `None` for Unknown.
    pub expected: Option<usize>,
    /// The stale versions of the cue: pool rows of facts that were once current.
    pub stale: Vec<usize>,
}

pub struct Session {
    pub events: Vec<Event>,
    pub queries: Vec<Query>,
    /// The cues written, in the order they were first written.
    pub written: Vec<i64>,
}

fn shuffle<T>(rng: &mut Rng, items: &mut [T]) {
    for index in (1..items.len()).rev() {
        items.swap(index, rng.index(index + 1));
    }
}

impl Session {
    /// The session of `seed` at `shape` over `pool`. A shape the pool cannot fill is an `Err`.
    pub fn generate(pool: &Pool, seed: u64, shape: Shape) -> Result<Self, String> {
        let cues = params::POOL_CUES;
        if shape.written + shape.queries_unwritten > cues
            || shape.updated > shape.written
            || shape.queries_updated > shape.updated
            || shape.queries_stable > shape.written - shape.updated
            || shape.queries_offtopic > pool.offtopic.len()
        {
            return Err(format!("the pool cannot fill the shape {shape:?}"));
        }
        let mut rng = Rng::new(seed ^ 0x4d30_3038_5e55_1011);
        let mut order: Vec<i64> = (0..cues as i64).collect();
        shuffle(&mut rng, &mut order);
        let written: Vec<i64> = order[..shape.written].to_vec();
        let unwritten: Vec<i64> =
            order[shape.written..shape.written + shape.queries_unwritten].to_vec();

        // Which written cues are updated, and when everything is written.
        let mut picks = written.clone();
        shuffle(&mut rng, &mut picks);
        let updated: Vec<i64> = picks[..shape.updated].to_vec();
        let stable: Vec<i64> = picks[shape.updated..].to_vec();
        let mut timed: Vec<(f64, usize, Event)> = Vec::new();
        let mut first_time = std::collections::BTreeMap::new();
        for &cue in &written {
            let time = rng.unit();
            first_time.insert(cue, time);
            timed.push((time, 1, Self::event(pool, cue, 1)));
        }
        for &cue in &updated {
            let after = first_time[&cue];
            let time = after + (1.0 - after) * rng.unit();
            timed.push((time, 2, Self::event(pool, cue, 2)));
        }
        timed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let events: Vec<Event> = timed.into_iter().map(|(_, _, event)| event).collect();

        let mut queries = Vec::with_capacity(shape.queries());
        let mut updated_pool = updated.clone();
        shuffle(&mut rng, &mut updated_pool);
        for &cue in &updated_pool[..shape.queries_updated] {
            queries.push(Self::query(
                pool,
                &mut rng,
                QueryKind::Updated,
                cue,
                Some(2),
                vec![1],
            ));
        }
        let mut stable_pool = stable;
        shuffle(&mut rng, &mut stable_pool);
        for &cue in &stable_pool[..shape.queries_stable] {
            queries.push(Self::query(
                pool,
                &mut rng,
                QueryKind::Stable,
                cue,
                Some(1),
                vec![],
            ));
        }
        for &cue in &unwritten {
            queries.push(Self::query(
                pool,
                &mut rng,
                QueryKind::Unwritten,
                cue,
                None,
                vec![],
            ));
        }
        let mut offtopic = pool.offtopic.clone();
        shuffle(&mut rng, &mut offtopic);
        for &row in &offtopic[..shape.queries_offtopic] {
            queries.push(Query {
                kind: QueryKind::Offtopic,
                row,
                cue: -1,
                expected: None,
                stale: Vec::new(),
            });
        }
        shuffle(&mut rng, &mut queries);
        Ok(Self {
            events,
            queries,
            written,
        })
    }

    fn event(pool: &Pool, cue: i64, version: usize) -> Event {
        Event {
            cue,
            version,
            row: pool.fact(cue, version),
            key_row: pool.query(cue, 0),
        }
    }

    fn query(
        pool: &Pool,
        rng: &mut Rng,
        kind: QueryKind,
        cue: i64,
        expected_version: Option<usize>,
        stale_versions: Vec<usize>,
    ) -> Query {
        let paraphrase = 1 + rng.index(params::POOL_PARAPHRASES - 1);
        Query {
            kind,
            row: pool.query(cue, paraphrase),
            cue,
            expected: expected_version.map(|version| pool.fact(cue, version)),
            stale: stale_versions
                .into_iter()
                .map(|version| pool.fact(cue, version))
                .collect(),
        }
    }

    /// The position in the write order of the write of `row`.
    pub fn write_index(&self, row: usize) -> Option<usize> {
        self.events.iter().position(|event| event.row == row)
    }
}

#[cfg(test)]
mod tests {
    use super::super::pool::tests::shared;
    use super::*;

    #[test]
    fn a_seed_reproduces_its_session_and_another_seed_differs() {
        let pool = shared();
        let a = Session::generate(pool, 1001, Shape::PINNED).unwrap();
        let b = Session::generate(pool, 1001, Shape::PINNED).unwrap();
        assert_eq!(a.events, b.events);
        assert_eq!(
            a.queries.iter().map(|q| q.row).collect::<Vec<_>>(),
            b.queries.iter().map(|q| q.row).collect::<Vec<_>>()
        );
        let c = Session::generate(pool, 1002, Shape::PINNED).unwrap();
        assert_ne!(a.events, c.events);
    }

    #[test]
    fn the_pinned_session_has_the_promised_shape() {
        let pool = shared();
        let session = Session::generate(pool, 1001, Shape::PINNED).unwrap();
        assert_eq!(session.events.len(), 600);
        assert_eq!(session.written.len(), 400);
        let count = |kind| session.queries.iter().filter(|q| q.kind == kind).count();
        assert_eq!(count(QueryKind::Updated), 160);
        assert_eq!(count(QueryKind::Stable), 160);
        assert_eq!(count(QueryKind::Unwritten), 40);
        assert_eq!(count(QueryKind::Offtopic), 40);
        // An update is written after its original, and every cue is written once or twice.
        for event in session.events.iter().filter(|e| e.version == 2) {
            let original = session
                .events
                .iter()
                .position(|e| e.cue == event.cue && e.version == 1)
                .expect("the original");
            let update = session.events.iter().position(|e| e == event).unwrap();
            assert!(original < update);
        }
        // Queries never use the cue text the writes are keyed by, and never name a fact.
        for query in &session.queries {
            let row = pool.row(query.row);
            if query.kind != QueryKind::Offtopic {
                assert!((1..4).contains(&row.paraphrase), "{row:?}");
            }
        }
        // Unwritten cues really are unwritten.
        for query in session.queries.iter().filter(|q| !q.kind.answerable()) {
            assert!(session
                .events
                .iter()
                .all(|e| e.cue != query.cue || query.cue == -1));
            assert_eq!(query.expected, None);
        }
    }

    #[test]
    fn a_shape_the_pool_cannot_fill_is_refused() {
        let pool = shared();
        let too_many = Shape {
            queries_offtopic: 701,
            ..Shape::PINNED
        };
        assert!(Session::generate(pool, 1, too_many).is_err());
        let too_few_stable = Shape {
            queries_stable: 201,
            ..Shape::PINNED
        };
        assert!(Session::generate(pool, 1, too_few_stable).is_err());
    }
}
