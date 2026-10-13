//! The hybrid retrieval arms: the pinned `rag_reference` stack (BM25 and cosine over the frozen
//! vectors, fused by reciprocal rank, k = 60), run by the digest-pinned Python driver through
//! `rag_bridge`.
//!
//! - `hybrid_live` is the gated baseline: it indexes the live facts only. An update deletes the old
//!   version and indexes the new one, so the index holds the current fact of every cue written.
//!   Whatever stale version it ever returns is the harness's or the driver's defect
//!   (`stale_in_live_index`), not a property of retrieval.
//! - `hybrid_append` is reported, not gated: the literal "plain retrieval", an append-only index
//!   that keeps every version written.
//!
//! The driver ranks by fused rank score, which is ordinal and so cannot separate a good hit from a
//! bad one; the harness gives these arms an abstention rule on the cosine between the query and the
//! top hit, computed from the same frozen vectors (`Policy` with a score threshold only).

use super::arms::Ranked;
use super::params;
use super::pool::{cosine, Pool};
use super::session::Session;
use crate::experiments::rag_bridge::{Bridge, BridgeError, Doc};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Index {
    Live,
    Append,
}

pub struct Run {
    pub queries: Vec<Ranked>,
    /// The documents the driver held when the queries were asked.
    pub documents: usize,
    pub query_ns: u128,
}

/// Build the index of `index` kind from the session's writes, in order, then answer every query.
pub fn run(
    bridge: &mut Bridge,
    pool: &Pool,
    session: &Session,
    index: Index,
) -> Result<Run, BridgeError> {
    bridge.reset()?;
    let mut by_id: HashMap<&str, usize> = HashMap::new();
    for event in &session.events {
        let row = pool.row(event.row);
        by_id.insert(row.id.as_str(), event.row);
        if index == Index::Live && event.version > 1 {
            bridge.delete(&pool.row(pool.fact(event.cue, event.version - 1)).id)?;
        }
        bridge.upsert(&Doc {
            id: &row.id,
            text: &row.text,
            embedding: pool.vector(event.row),
        })?;
    }
    let documents = bridge.size()?;
    let started = Instant::now();
    let mut queries = Vec::with_capacity(session.queries.len());
    for query in &session.queries {
        let row = pool.row(query.row);
        let vector = pool.vector(query.row);
        let hits = bridge.search(&row.text, vector, params::RANKED_LIMIT)?;
        let mut rows = Vec::with_capacity(hits.len());
        let mut scores = Vec::with_capacity(hits.len());
        for hit in &hits {
            let fact = *by_id.get(hit.id.as_str()).ok_or_else(|| {
                BridgeError::Protocol(format!("a hit names {:?}, which was never indexed", hit.id))
            })?;
            rows.push(fact);
            scores.push(cosine(vector, pool.vector(fact)) as f32);
        }
        queries.push(Ranked {
            rows,
            scores,
            runner_up: 0.0,
        });
    }
    Ok(Run {
        queries,
        documents,
        query_ns: started.elapsed().as_nanos(),
    })
}
