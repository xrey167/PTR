//! The fast-weight memory arm (`ptr-fastmem`, the code under test).
//!
//! Each write of the session becomes a `FastMemory::write`: the key is the projected frozen
//! embedding of the cue text (paraphrase 0 of the cue), the value is the identifier code of the
//! fact (capsule `m008:c:<cue>` at generation `<version>`), beta is 1 so an update under the same
//! cue replaces the association, and the decay is the pinned scalar. A query is the projected
//! embedding of a paraphrase; the readout is decoded against the codes of every fact the journal
//! holds (stale versions included) by `decode_readout`.
//!
//! The harness never trusts the product's decode: it scores every candidate itself from the
//! readout and the codes (`independent`, no `decode_readout` code) and applies the same rule, and a
//! product answer that differs is `decode_contract_violations`. It also checks, from the readout
//! alone, that a write reads back exactly (`readback_failures`), that an exact-key overwrite leaves
//! no stale association (`stale_residual_violations`) and that the key projection binding refuses
//! a query of another projection or of none (`projection_mismatch_accepted`).
//!
//! A product error or panic is a counter (`product_errors`, `product_panics`), never an abort.

use super::arms::{Policy, Ranked, Refusal};
use super::params;
use super::pool::Pool;
use super::session::{QueryKind, Session};
use crate::experiments::layered_kv::emit::Tally;
use ptr_fastmem::{
    decode_readout, Decay, DecodePolicy, FactCode, FastMemory, FastMemoryConfig, FastMemoryError,
    IdentifierCodebook, ProjectionSpec, Query, Readout, Recall, SeededProjection, SourceRef,
    WriteRequest,
};
use ptr_ledger::integrity::sha256;
use ptr_types::{CapsuleId, Generation};
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

/// The shape and write parameters of the memory.
#[derive(Clone, Copy, Debug)]
pub struct FastConfig {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub beta_permille: i64,
    pub decay_permille: i64,
    pub checkpoint_interval: u32,
    pub projection_seed: u64,
    pub codebook_seed: u64,
}

impl FastConfig {
    pub fn pinned() -> Self {
        Self {
            heads: params::HEADS,
            key_dim: params::KEY_DIM,
            value_dim: params::VALUE_DIM,
            beta_permille: params::BETA_PERMILLE,
            decay_permille: params::DECAY_PERMILLE,
            checkpoint_interval: params::CHECKPOINT_INTERVAL,
            projection_seed: params::PROJECTION_SEED,
            codebook_seed: params::CODEBOOK_SEED,
        }
    }

    fn memory(&self, writes: usize) -> FastMemoryConfig {
        FastMemoryConfig {
            heads: self.heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            checkpoint_interval: self.checkpoint_interval,
            max_writes: (writes as u32).max(1),
        }
    }

    fn decay(&self) -> Decay {
        if self.decay_permille >= 1000 {
            Decay::None
        } else {
            Decay::Scalar(self.decay_permille as f32 / 1000.0)
        }
    }
}

/// What the memory did for one query.
pub struct QueryRun {
    /// The harness's independent scoring of every candidate fact, best first.
    pub scored: Vec<(usize, f32)>,
    /// The top `RANKED_LIMIT` of `scored` and the runner-up, before any threshold.
    pub ranked: Ranked,
    /// The product's answer under the policy it was given (rows, best first, empty = Unknown),
    /// or `None` when the product failed to produce one.
    pub product: Option<Vec<usize>>,
    /// Descriptive, never gated: the memory's top fact when asked with the cue text itself
    /// (paraphrase 0, the text the write was keyed by), unthresholded. `None` for a query that has
    /// no answer, or when the read failed.
    pub exact_cue_top: Option<usize>,
}

pub struct Run {
    pub queries: Vec<QueryRun>,
    /// Wall-clock nanoseconds spent reading and decoding all queries (descriptive).
    pub query_ns: u128,
    /// The facts the journal held, which every query was decoded against.
    pub candidates: usize,
}

/// Run a call of the code under test and contain a panic: it is counted as `product_panics`.
fn contained<T>(tally: &mut Tally, call: impl FnOnce() -> T) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(value) => Some(value),
        Err(_) => {
            tally.add("product_panics", 1);
            None
        }
    }
}

/// Run a fallible call of the code under test: an `Err` is `product_errors` and a panic is
/// `product_panics`; both are falsifying counters (exit 1), never a harness failure.
fn guarded<T>(tally: &mut Tally, call: impl FnOnce() -> Result<T, FastMemoryError>) -> Option<T> {
    match contained(tally, call)? {
        Ok(value) => Some(value),
        Err(_) => {
            tally.add("product_errors", 1);
            None
        }
    }
}

fn capsule(cue: i64) -> String {
    format!("m008:c:{cue:04}")
}

/// The harness's own decode: every candidate scored as `<readout, code>` in the product's order of
/// summation, sorted by score, then capsule, then generation. It reads only what the product
/// publishes (the readout's values and each fact's code).
fn independent(
    values: &[f32],
    facts: &[FactCode],
    rows: &HashMap<(String, u64), usize>,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(f32, &str, u64, usize)> = facts
        .iter()
        .map(|fact| {
            let score: f32 = values.iter().zip(fact.code()).map(|(r, c)| r * c).sum();
            let name = fact.capsule().0.as_str();
            let generation = fact.generation().0;
            (
                score,
                name,
                generation,
                rows[&(name.to_string(), generation)],
            )
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    scored
        .into_iter()
        .map(|(score, _, _, row)| (row, score))
        .collect()
}

fn ranked_of(scored: &[(usize, f32)]) -> Ranked {
    Ranked {
        rows: scored
            .iter()
            .take(params::RANKED_LIMIT)
            .map(|entry| entry.0)
            .collect(),
        scores: scored
            .iter()
            .take(params::RANKED_LIMIT)
            .map(|entry| entry.1)
            .collect(),
        runner_up: scored.get(1).map_or(0.0, |entry| entry.1),
    }
}

/// The product's decode of `readout` under `decode_policy`, compared with the harness's own decode
/// (`ranked` under `policy`): a different answer, or a hit that names no fact of the session, is
/// `decode_contract_violations`. Returns the product's answer (rows, empty = Unknown), or `None`
/// when the product failed.
#[allow(clippy::too_many_arguments)]
fn decode_and_compare(
    tally: &mut Tally,
    readout: &Readout,
    facts: &[FactCode],
    rows: &HashMap<(String, u64), usize>,
    policy: Policy,
    decode_policy: DecodePolicy,
    scored: &[(usize, f32)],
    ranked: &Ranked,
) -> Option<Vec<usize>> {
    let decoded = guarded(tally, || {
        decode_readout(readout, facts, decode_policy).map(|recall| match recall {
            Recall::Unknown => None,
            Recall::Hits(hits) => Some(
                hits.iter()
                    .map(|hit| (hit.capsule().0.clone(), hit.generation().0, hit.score))
                    .collect::<Vec<_>>(),
            ),
        })
    });
    // `Hits` always names at least one fact; an empty `Hits` is the contract broken, not Unknown.
    if matches!(&decoded, Some(Some(hits)) if hits.is_empty()) {
        tally.add("decode_contract_violations", 1);
    }
    let product = decoded.and_then(|hits| {
        let mut answer = Vec::new();
        let mut checked = Vec::new();
        for (name, generation, score) in hits.unwrap_or_default() {
            // A hit that names no fact of the session is the product's defect.
            answer.push(*rows.get(&(name, generation))?);
            checked.push(score);
        }
        Some((answer, checked))
    });
    let expected = policy.apply(ranked);
    let agrees = match &product {
        None => false,
        Some((answer, scores)) => {
            *answer == expected
                && answer.iter().zip(scores).all(|(row, score)| {
                    scored
                        .iter()
                        .find(|entry| entry.0 == *row)
                        .is_some_and(|entry| {
                            (entry.1 - score).abs() <= params::DECODE_SCORE_TOLERANCE
                        })
                })
        }
    };
    if !agrees {
        tally.add("decode_contract_violations", 1);
    }
    product.map(|(answer, _)| answer)
}

/// Probes of the decode contract on small memories whose readout is known by construction. Two
/// facts are written under orthogonal unit keys (the same coordinates in every head); a query along
/// one key reads one fact (the decode must answer it), a query along the sum of both reads both at
/// the same strength (the readout is ambiguous, the decode must say Unknown by margin). The harness's
/// own decode judges each and the product's answer is compared with it, like every organic query.
/// These probes are by construction: the margin refusals they produce are counted apart
/// (`contract_probe_margin_refusals`) from the organic ones.
fn contract_probes(
    config: &FastConfig,
    memory_config: FastMemoryConfig,
    policy: Policy,
    decode_policy: DecodePolicy,
    tally: &mut Tally,
) {
    let small = FastMemoryConfig {
        max_writes: 4,
        ..memory_config
    };
    let Some(codebook) = guarded(tally, || {
        IdentifierCodebook::new(config.codebook_seed, small.value_len())
    }) else {
        return;
    };
    let raw = |first: usize, second: Option<usize>| -> Vec<f32> {
        let mut key = vec![0.0_f32; small.key_len()];
        for head in 0..small.heads {
            key[head * small.key_dim + first % small.key_dim] = 1.0;
            if let Some(second) = second {
                key[head * small.key_dim + second % small.key_dim] = 1.0;
            }
        }
        key
    };
    for probe in 0..PROBES {
        let (a, b) = (2 * probe, 2 * probe + 1);
        let Some(mut memory) = guarded(tally, || FastMemory::new(small, codebook)) else {
            return;
        };
        let mut rows: HashMap<(String, u64), usize> = HashMap::new();
        for (index, (name, key)) in [("a", raw(a, None)), ("b", raw(b, None))]
            .into_iter()
            .enumerate()
        {
            let capsule_name = format!("m008:probe:{probe:03}:{name}");
            rows.insert((capsule_name.clone(), 1), index);
            let Some(value) = contained(tally, || {
                codebook.code_for(&CapsuleId::from(capsule_name.as_str()), Generation(1))
            }) else {
                return;
            };
            let request = WriteRequest {
                source: SourceRef {
                    key: capsule_name.clone(),
                    generation: Generation(1),
                    input_digest: sha256(capsule_name.as_bytes()),
                },
                key,
                value,
                beta: 1.0,
                decay: Decay::None,
            };
            if guarded(tally, || memory.write(request).map(|_| ())).is_none() {
                return;
            }
        }
        let Some(facts) = contained(tally, || memory.fact_codes()) else {
            return;
        };
        for (kind, query) in [("clear", raw(a, None)), ("ambiguous", raw(a, Some(b)))] {
            let Some(readout) = guarded(tally, || {
                memory.read_admitted(&Query::new(&small, query)?, |_| true)
            }) else {
                continue;
            };
            let scored = independent(&readout.values, &facts, &rows);
            let ranked = ranked_of(&scored);
            tally.add("contract_probes", 1);
            if policy.refusal(&ranked) == Some(Refusal::Margin) {
                tally.add("contract_probe_margin_refusals", 1);
            }
            let answer = decode_and_compare(
                tally,
                &readout,
                &facts,
                &rows,
                policy,
                decode_policy,
                &scored,
                &ranked,
            );
            // The construction itself: a clear query is answered, an ambiguous one is not.
            let answered = answer.as_ref().is_some_and(|rows| !rows.is_empty());
            if (kind == "clear") != answered && answer.is_some() {
                tally.add("decode_contract_violations", 1);
            }
        }
    }
}

/// Probe memories per run (two decodes each).
const PROBES: usize = 40;

/// The memory and everything drawn for it. Built through contained calls: a constructor of the
/// product that refuses the pinned parameters, or panics, is a counter, never a harness failure.
struct Setup {
    memory_config: FastMemoryConfig,
    projection: SeededProjection,
    other: SeededProjection,
    codebook: IdentifierCodebook,
    memory: FastMemory,
}

fn setup(config: &FastConfig, writes: usize, tally: &mut Tally) -> Option<Setup> {
    let memory_config = config.memory(writes);
    let projection = guarded(tally, || {
        SeededProjection::new(ProjectionSpec {
            input_dim: params::EMBEDDING_DIM,
            heads: config.heads,
            head_dim: config.key_dim,
            seed: config.projection_seed,
        })
    })?;
    let other = guarded(tally, || {
        SeededProjection::new(ProjectionSpec {
            input_dim: params::EMBEDDING_DIM,
            heads: config.heads,
            head_dim: config.key_dim,
            seed: params::OTHER_PROJECTION_SEED,
        })
    })?;
    let codebook = guarded(tally, || {
        IdentifierCodebook::new(config.codebook_seed, memory_config.value_len())
    })?;
    let digest = contained(tally, || projection.digest())?;
    let memory = guarded(tally, || {
        FastMemory::with_projection(memory_config, codebook, digest)
    })?;
    Some(Setup {
        memory_config,
        projection,
        other,
        codebook,
        memory,
    })
}

/// A run in which the memory could not be built: every query is unanswered by the product.
fn failed_run(session: &Session) -> Run {
    Run {
        queries: session
            .queries
            .iter()
            .map(|_| QueryRun {
                scored: Vec::new(),
                ranked: Ranked::empty(),
                product: None,
                exact_cue_top: None,
            })
            .collect(),
        query_ns: 0,
        candidates: 0,
    }
}

/// Write the session into a fresh memory, read every query back and decode it under `policy`.
/// No call of the code under test can fail the run: an error is `product_errors`, a panic
/// `product_panics`.
pub fn run(
    pool: &Pool,
    session: &Session,
    config: &FastConfig,
    policy: Policy,
    tally: &mut Tally,
) -> Run {
    let Some(Setup {
        memory_config,
        projection,
        other,
        codebook,
        mut memory,
    }) = setup(config, session.events.len(), tally)
    else {
        return failed_run(session);
    };

    let mut rows: HashMap<(String, u64), usize> = HashMap::new();
    for event in &session.events {
        rows.insert((capsule(event.cue), event.version as u64), event.row);
    }

    // The writes, each read back through its own key at once.
    for event in &session.events {
        let name = capsule(event.cue);
        let generation = Generation(event.version as u64);
        let Some(key) = guarded(tally, || projection.project(pool.vector(event.key_row))) else {
            continue;
        };
        let Some(value) = contained(tally, || {
            codebook.code_for(&CapsuleId::from(name.as_str()), generation)
        }) else {
            continue;
        };
        let request = WriteRequest {
            source: SourceRef {
                key: name.clone(),
                generation,
                input_digest: sha256(pool.row(event.row).text.as_bytes()),
            },
            key,
            value: value.clone(),
            beta: config.beta_permille as f32 / 1000.0,
            decay: config.decay(),
        };
        if guarded(tally, || memory.write(request).map(|_| ())).is_none() {
            continue;
        }
        let readout = guarded(tally, || {
            let query = Query::project(&memory_config, &projection, pool.vector(event.key_row))?;
            memory.read_admitted(&query, |_| true)
        });
        if let Some(readout) = readout {
            tally.add("readback_checks", 1);
            let score: f32 = readout.values.iter().zip(&value).map(|(r, c)| r * c).sum();
            if score < params::READBACK_MIN_MILLI as f32 / 1000.0 {
                tally.add("readback_failures", 1);
            }
        }
    }

    // The queries.
    let Some(facts) = contained(tally, || memory.fact_codes()) else {
        return failed_run(session);
    };
    let decode_policy = DecodePolicy {
        limit: params::RANKED_LIMIT,
        min_score: policy.min_score,
        min_margin: policy.min_margin,
    };
    let mut queries = Vec::with_capacity(session.queries.len());
    let started = Instant::now();
    for query in &session.queries {
        let vector = pool.vector(query.row);
        let readout = guarded(tally, || {
            let probe = Query::project(&memory_config, &projection, vector)?;
            memory.read_admitted(&probe, |_| true)
        });
        let Some(readout) = readout else {
            queries.push(QueryRun {
                scored: Vec::new(),
                ranked: Ranked::empty(),
                product: None,
                exact_cue_top: None,
            });
            continue;
        };
        let scored = independent(&readout.values, &facts, &rows);
        let ranked = ranked_of(&scored);
        match policy.refusal(&ranked) {
            Some(Refusal::Score) => tally.add("fast_score_abstentions", 1),
            Some(Refusal::Margin) => tally.add("fast_margin_abstentions", 1),
            None => {}
        }
        if query.kind == QueryKind::Updated {
            if let Some(stale) = query.stale.first() {
                if let Some((_, score)) = scored.iter().find(|entry| entry.0 == *stale) {
                    tally.add("stale_residual_checks", 1);
                    if *score > params::STALE_RESIDUAL_MAX_MILLI as f32 / 1000.0 {
                        tally.add("stale_residual_violations", 1);
                    }
                }
            }
        }
        let product = decode_and_compare(
            tally,
            &readout,
            &facts,
            &rows,
            policy,
            decode_policy,
            &scored,
            &ranked,
        );
        // Descriptive: the same cue asked with the text the write was keyed by.
        let exact_cue_top = if query.kind.answerable() {
            let cue_vector = pool.vector(pool.query(query.cue, 0));
            guarded(tally, || {
                let probe = Query::project(&memory_config, &projection, cue_vector)?;
                memory.read_admitted(&probe, |_| true)
            })
            .and_then(|exact| {
                independent(&exact.values, &facts, &rows)
                    .first()
                    .map(|top| top.0)
            })
        } else {
            None
        };
        queries.push(QueryRun {
            scored,
            ranked,
            product,
            exact_cue_top,
        });
    }
    let query_ns = started.elapsed().as_nanos();

    contract_probes(config, memory_config, policy, decode_policy, tally);

    // The projection binding: a query of another projection, or one that states none, is refused.
    for query in session.queries.iter().take(64) {
        let vector = pool.vector(query.row);
        let wrong = guarded(tally, || Query::project(&memory_config, &other, vector));
        let unstated = guarded(tally, || {
            projection
                .project(vector)
                .and_then(|raw| Query::new(&memory_config, raw))
        });
        for probe in [wrong, unstated].into_iter().flatten() {
            tally.add("projection_probes", 1);
            match contained(tally, || memory.read_admitted(&probe, |_| true)) {
                Some(Err(FastMemoryError::ProjectionMismatch { .. })) => {}
                Some(Ok(_)) => tally.add("projection_mismatch_accepted", 1),
                Some(Err(_)) => tally.add("product_errors", 1),
                None => {}
            }
        }
    }
    Run {
        queries,
        query_ns,
        candidates: facts.len(),
    }
}
