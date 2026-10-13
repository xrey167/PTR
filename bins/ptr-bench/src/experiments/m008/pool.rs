//! The frozen text pool and its frozen vectors (`research/embeddings`), loaded and verified.
//!
//! Every text of the experiment (the facts, the cue texts, the query paraphrases, the off-topic
//! queries) is a row of `pool-m008.jsonl`, and every vector is the same row of
//! `vectors-m008.f16` (little-endian IEEE half, 384 values per row, produced once by the pinned
//! MiniLM on CPU). The harness embeds nothing. A pool or a vector file that is not the pinned one,
//! a token count that differs from the recount by the pinned pattern, a pool that is not the
//! shape the generator promises: each is a harness-integrity failure (`Err`, exit 2).

use super::params;
use crate::experiments::rag_bridge::{self, Value};
use ptr_ledger::integrity::sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Fact,
    Query,
    Offtopic,
}

/// One row of the pool.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: String,
    pub text: String,
    pub tokens: usize,
    pub kind: Kind,
    /// The cue (an entity x attribute pair), `-1` for an off-topic query.
    pub cue: i64,
    /// `1..=3` for a fact, `0` otherwise.
    pub version: usize,
    /// The paraphrase index of a query, `0` otherwise.
    pub paraphrase: usize,
}

pub struct Pool {
    pub rows: Vec<Row>,
    vectors: Vec<f32>,
    facts: BTreeMap<(i64, usize), usize>,
    queries: BTreeMap<(i64, usize), usize>,
    pub offtopic: Vec<usize>,
    pub pool_sha256: String,
    pub vectors_sha256: String,
    pub max_fact_tokens: usize,
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn repo_file(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

/// The token count of `text` under `[A-Za-z0-9]+|[^\sA-Za-z0-9]`: a run of ASCII letters and
/// digits is one token, any other character that is not whitespace is one token. The pool is
/// ASCII (the loader refuses anything else), so Unicode whitespace never arises.
pub fn count_tokens(text: &str) -> usize {
    let mut tokens = 0;
    let mut in_word = false;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            if !in_word {
                tokens += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !character.is_whitespace() {
                tokens += 1;
            }
        }
    }
    tokens
}

/// An IEEE half as an `f32`; `None` for an infinity or a NaN.
pub fn f16_to_f32(bits: u16) -> Option<f32> {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x3ff);
    match exponent {
        0 => {
            let magnitude = fraction as f32 * (1.0 / 16_777_216.0);
            Some(if sign == 0 { magnitude } else { -magnitude })
        }
        31 => None,
        _ => Some(f32::from_bits(
            sign | ((exponent + 112) << 23) | (fraction << 13),
        )),
    }
}

fn int_field(row: &Value, key: &str) -> Result<i64, String> {
    row.get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("pool row has no integer {key}"))
}

fn str_field<'a>(row: &'a Value, key: &str) -> Result<&'a str, String> {
    row.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("pool row has no string {key}"))
}

impl Pool {
    /// Load the pinned pool and vectors from the repository, verifying their digests, the token
    /// counts and the structure.
    pub fn load() -> Result<Self, String> {
        let pool_bytes = std::fs::read(repo_file(params::POOL_FILE))
            .map_err(|error| format!("cannot read {}: {error}", params::POOL_FILE))?;
        let vector_bytes = std::fs::read(repo_file(params::VECTORS_FILE))
            .map_err(|error| format!("cannot read {}: {error}", params::VECTORS_FILE))?;
        Self::from_bytes(&pool_bytes, &vector_bytes)
    }

    /// The same from bytes (the digests are checked against the pinned ones).
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn from_bytes(pool_bytes: &[u8], vector_bytes: &[u8]) -> Result<Self, String> {
        let pool_sha256 = hex(&sha256(pool_bytes));
        if pool_sha256 != params::POOL_SHA256 {
            return Err(format!(
                "the pool is not the pinned one: sha256 {pool_sha256}, pinned {}",
                params::POOL_SHA256
            ));
        }
        let vectors_sha256 = hex(&sha256(vector_bytes));
        if vectors_sha256 != params::VECTORS_SHA256 {
            return Err(format!(
                "the vectors are not the pinned ones: sha256 {vectors_sha256}, pinned {}",
                params::VECTORS_SHA256
            ));
        }
        let text = std::str::from_utf8(pool_bytes)
            .map_err(|error| format!("pool is not UTF-8: {error}"))?;
        let mut rows = Vec::with_capacity(params::POOL_ROWS);
        for line in text.lines() {
            let value = rag_bridge::parse(line)?;
            let kind = match str_field(&value, "kind")? {
                "fact" => Kind::Fact,
                "query" => Kind::Query,
                "offtopic" => Kind::Offtopic,
                other => return Err(format!("unknown pool kind {other:?}")),
            };
            let row_text = str_field(&value, "text")?.to_string();
            if !row_text.is_ascii() {
                return Err(format!("pool text is not ASCII: {row_text:?}"));
            }
            let recount = count_tokens(&row_text);
            let tokens = int_field(&value, "tokens")?;
            if tokens != recount as i64 {
                return Err(format!(
                    "token count of {row_text:?}: the pool says {tokens}, the pinned pattern counts {recount}"
                ));
            }
            rows.push(Row {
                id: str_field(&value, "id")?.to_string(),
                text: row_text,
                tokens: recount,
                kind,
                cue: int_field(&value, "cue")?,
                version: int_field(&value, "version")? as usize,
                paraphrase: int_field(&value, "paraphrase")? as usize,
            });
        }
        if rows.len() != params::POOL_ROWS {
            return Err(format!(
                "the pool has {} rows, expected {}",
                rows.len(),
                params::POOL_ROWS
            ));
        }
        let expected_bytes = params::POOL_ROWS * params::EMBEDDING_DIM * 2;
        if vector_bytes.len() != expected_bytes {
            return Err(format!(
                "the vector file has {} bytes, expected {expected_bytes}",
                vector_bytes.len()
            ));
        }
        let mut vectors = Vec::with_capacity(params::POOL_ROWS * params::EMBEDDING_DIM);
        for pair in vector_bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            vectors.push(f16_to_f32(bits).ok_or("a vector holds an infinity or a NaN")?);
        }
        let mut pool = Self {
            rows,
            vectors,
            facts: BTreeMap::new(),
            queries: BTreeMap::new(),
            offtopic: Vec::new(),
            pool_sha256,
            vectors_sha256,
            max_fact_tokens: 0,
        };
        pool.index()?;
        pool.check_vectors()?;
        Ok(pool)
    }

    fn index(&mut self) -> Result<(), String> {
        let mut ids = BTreeSet::new();
        for (index, row) in self.rows.iter().enumerate() {
            if !ids.insert(row.id.clone()) {
                return Err(format!("duplicate pool id {}", row.id));
            }
            match row.kind {
                Kind::Fact => {
                    self.max_fact_tokens = self.max_fact_tokens.max(row.tokens);
                    if self.facts.insert((row.cue, row.version), index).is_some() {
                        return Err(format!(
                            "two facts for cue {} version {}",
                            row.cue, row.version
                        ));
                    }
                }
                Kind::Query => {
                    if self
                        .queries
                        .insert((row.cue, row.paraphrase), index)
                        .is_some()
                    {
                        return Err(format!(
                            "two queries for cue {} paraphrase {}",
                            row.cue, row.paraphrase
                        ));
                    }
                }
                Kind::Offtopic => self.offtopic.push(index),
            }
        }
        for cue in 0..params::POOL_CUES as i64 {
            for version in 1..=params::POOL_VERSIONS {
                if !self.facts.contains_key(&(cue, version)) {
                    return Err(format!("no fact for cue {cue} version {version}"));
                }
            }
            for paraphrase in 0..params::POOL_PARAPHRASES {
                if !self.queries.contains_key(&(cue, paraphrase)) {
                    return Err(format!("no query for cue {cue} paraphrase {paraphrase}"));
                }
            }
        }
        if self.facts.len() != params::POOL_CUES * params::POOL_VERSIONS
            || self.queries.len() != params::POOL_CUES * params::POOL_PARAPHRASES
            || self.offtopic.len() != params::POOL_OFFTOPIC
        {
            return Err(
                "the pool holds rows beyond the cues, versions and paraphrases it promises".into(),
            );
        }
        Ok(())
    }

    /// Every vector is finite and unit length to the precision of a half.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn check_vectors(&self) -> Result<(), String> {
        for (index, vector) in self.vectors.chunks_exact(params::EMBEDDING_DIM).enumerate() {
            let norm = vector
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>()
                .sqrt();
            if (norm - 1.0).abs() > 2.0e-3 {
                return Err(format!("vector of row {index} has norm {norm}"));
            }
        }
        Ok(())
    }

    /// The row of the fact of `cue` at `version` (1..=3).
    pub fn fact(&self, cue: i64, version: usize) -> usize {
        self.facts[&(cue, version)]
    }

    /// The row of the query of `cue` at `paraphrase` (0 is the cue text a write is keyed by, 1..=3
    /// are the paraphrases a query asks with).
    pub fn query(&self, cue: i64, paraphrase: usize) -> usize {
        self.queries[&(cue, paraphrase)]
    }

    pub fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[row * params::EMBEDDING_DIM..(row + 1) * params::EMBEDDING_DIM]
    }

    pub fn row(&self, row: usize) -> &Row {
        &self.rows[row]
    }
}

/// The cosine of two vectors, accumulated in f64 (deterministic: no library function but the
/// exact IEEE square root).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// The pool is loaded once for the whole test binary (it is read and hashed, 13 MB).
    pub(crate) fn shared() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(|| Pool::load().expect("the pinned pool and vectors load"))
    }

    #[test]
    fn halves_convert_exactly() {
        assert_eq!(f16_to_f32(0x3c00), Some(1.0));
        assert_eq!(f16_to_f32(0xc000), Some(-2.0));
        assert_eq!(f16_to_f32(0x0000), Some(0.0));
        assert_eq!(f16_to_f32(0x0001), Some(2.0_f32.powi(-24)));
        assert_eq!(f16_to_f32(0x7bff), Some(65504.0));
        assert_eq!(f16_to_f32(0x0400), Some(2.0_f32.powi(-14)));
        assert_eq!(f16_to_f32(0x3555), Some(0.333_251_95));
        assert_eq!(f16_to_f32(0x7c00), None);
        assert_eq!(f16_to_f32(0xfe00), None);
    }

    #[test]
    fn tokens_are_counted_by_the_pinned_pattern() {
        assert_eq!(
            count_tokens("Renata Ironside prefers lilac over every other colour."),
            9
        );
        assert_eq!(count_tokens("a1b2 c-d"), 4);
        assert_eq!(count_tokens("  ?! "), 2);
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn the_pinned_pool_loads_with_its_promised_shape() {
        let pool = shared();
        assert_eq!(pool.rows.len(), params::POOL_ROWS);
        assert_eq!(pool.max_fact_tokens, params::MAX_FACT_TOKENS);
        assert_eq!(pool.pool_sha256, params::POOL_SHA256);
        assert_eq!(pool.vectors_sha256, params::VECTORS_SHA256);
        let fact = pool.row(pool.fact(0, 2));
        assert_eq!((fact.cue, fact.version, fact.kind), (0, 2, Kind::Fact));
        let cue_text = pool.row(pool.query(0, 0));
        assert_eq!((cue_text.cue, cue_text.paraphrase), (0, 0));
        assert!(cosine(pool.vector(pool.fact(0, 1)), pool.vector(pool.fact(0, 1))) > 0.999);
        // The fact versions of a cue are near-duplicates and still distinct.
        let near = cosine(pool.vector(pool.fact(0, 1)), pool.vector(pool.fact(0, 2)));
        assert!((0.3..0.9999).contains(&near), "{near}");
    }

    #[test]
    fn a_changed_pool_or_vector_file_is_refused_by_digest() {
        let pool = std::fs::read(repo_file(params::POOL_FILE)).unwrap();
        let vectors = std::fs::read(repo_file(params::VECTORS_FILE)).unwrap();
        let mut bad_pool = pool.clone();
        bad_pool[10] ^= 1;
        assert!(Pool::from_bytes(&bad_pool, &vectors)
            .err()
            .unwrap()
            .contains("not the pinned one"));
        let mut bad_vectors = vectors.clone();
        bad_vectors[100] ^= 1;
        assert!(Pool::from_bytes(&pool, &bad_vectors)
            .err()
            .unwrap()
            .contains("not the pinned ones"));
        assert!(Pool::from_bytes(&pool, &vectors[..vectors.len() - 2]).is_err());
    }
}
