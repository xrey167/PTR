use std::collections::BTreeSet;

use ptr_search::SearchHit;
use ptr_types::{CapsuleId, Generation};

use crate::error::FastMemoryError;
use crate::memory::FastMemory;
use crate::projection::IdentifierCodebook;
use crate::state::Readout;

/// Backend label carried by every hit this crate produces.
pub const FASTMEM_BACKEND: &str = "fastmem";

/// An explicit fact a readout may be decoded into: a capsule at one generation
/// and the identifier code its writes stored as value.
#[derive(Clone, Debug, PartialEq)]
pub struct FactCode {
    pub capsule: CapsuleId,
    pub generation: Generation,
    pub code: Vec<f32>,
}

impl IdentifierCodebook {
    /// The value code of a fact at one generation. Writers store exactly this
    /// vector as the value of a write about the fact, and decoding compares
    /// readouts against it; a new generation is a new fact identity.
    pub fn code_for(&self, capsule: &CapsuleId, generation: Generation) -> Vec<f32> {
        self.code(&format!("{capsule}@{}", generation.0))
    }
}

impl FastMemory {
    /// The candidate facts a readout may decode into: every distinct source
    /// the journal holds, and nothing else. A fact that was never written, or
    /// whose writes were revoked, is not a candidate.
    pub fn fact_codes(&self, book: &IdentifierCodebook) -> Vec<FactCode> {
        let sources: BTreeSet<(String, Generation)> = self
            .writes()
            .iter()
            .map(|write| (write.source().key.clone(), write.source().generation))
            .collect();
        sources
            .into_iter()
            .map(|(key, generation)| {
                let capsule = CapsuleId::from(key.as_str());
                let code = book.code_for(&capsule, generation);
                FactCode {
                    capsule,
                    generation,
                    code,
                }
            })
            .collect()
    }
}

/// When a readout is confident enough to name a fact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodePolicy {
    pub limit: usize,
    /// Minimum `<readout, code>` for a fact to be returned at all.
    pub min_score: f32,
    /// Minimum lead of the best fact over the runner-up. Below it the readout
    /// is ambiguous and the answer is `Unknown`.
    pub min_margin: f32,
}

/// The result of decoding a readout.
#[derive(Clone, Debug, PartialEq)]
pub enum Recall {
    /// Named facts, best first, each at the lowest evidence stage.
    Hits(Vec<SearchHit>),
    /// No fact scores high enough, or the best does not lead clearly. Unknown
    /// is a valid answer, not a failure.
    Unknown,
}

/// Decode a readout against candidate fact codes.
///
/// The score of a fact is `<readout, code>`: with unit-length, nearly
/// orthogonal codes it estimates the memory's weight on that fact. Hits start
/// at the lowest evidence stage, so a recall must still be resolved against
/// live generations and verified before it is relied on. Ties are broken by
/// capsule, then generation.
pub fn decode_readout<'a, I>(
    readout: &Readout,
    facts: I,
    policy: DecodePolicy,
) -> Result<Recall, FastMemoryError>
where
    I: IntoIterator<Item = &'a FactCode>,
{
    let mut scored = Vec::new();
    for fact in facts {
        if fact.code.len() != readout.values.len() {
            return Err(FastMemoryError::DimensionMismatch {
                field: "fact code",
                expected: readout.values.len(),
                actual: fact.code.len(),
            });
        }
        let score: f32 = readout
            .values
            .iter()
            .zip(&fact.code)
            .map(|(r, c)| r * c)
            .sum();
        scored.push((score, fact));
    }
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.capsule.cmp(&right.1.capsule))
            .then_with(|| left.1.generation.cmp(&right.1.generation))
    });
    let Some(best) = scored.first().map(|(score, _)| *score) else {
        return Ok(Recall::Unknown);
    };
    let runner_up = scored.get(1).map(|(score, _)| *score).unwrap_or(0.0);
    if best < policy.min_score || best - runner_up < policy.min_margin {
        return Ok(Recall::Unknown);
    }
    Ok(Recall::Hits(
        scored
            .into_iter()
            .filter(|(score, _)| *score >= policy.min_score)
            .take(policy.limit)
            .map(|(score, fact)| {
                SearchHit::new(
                    fact.capsule.clone(),
                    fact.generation,
                    score,
                    FASTMEM_BACKEND,
                )
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::WriteSeq;
    use ptr_search::EvidenceStage;

    fn fact(id: &str, code: Vec<f32>) -> FactCode {
        FactCode {
            capsule: CapsuleId::from(id),
            generation: Generation(2),
            code,
        }
    }

    fn policy() -> DecodePolicy {
        DecodePolicy {
            limit: 5,
            min_score: 0.5,
            min_margin: 0.2,
        }
    }

    #[test]
    fn a_clear_winner_is_named_and_starts_as_a_search_candidate() {
        let readout = Readout {
            values: vec![0.9, 0.1],
            as_of: WriteSeq(3),
        };
        let facts = [fact("b", vec![0.0, 1.0]), fact("a", vec![1.0, 0.0])];
        let Recall::Hits(hits) = decode_readout(&readout, &facts, policy()).unwrap() else {
            panic!("a clear winner is named");
        };
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].capsule, CapsuleId::from("a"));
        assert_eq!(hits[0].stage(), EvidenceStage::SearchCandidate);
        assert_eq!(hits[0].backend, FASTMEM_BACKEND);
    }

    #[test]
    fn an_ambiguous_or_weak_readout_is_unknown() {
        let facts = [fact("a", vec![1.0, 0.0]), fact("b", vec![0.0, 1.0])];
        let ambiguous = Readout {
            values: vec![0.7, 0.65],
            as_of: WriteSeq(1),
        };
        assert_eq!(
            decode_readout(&ambiguous, &facts, policy()).unwrap(),
            Recall::Unknown
        );
        let weak = Readout {
            values: vec![0.3, 0.0],
            as_of: WriteSeq(1),
        };
        assert_eq!(
            decode_readout(&weak, &facts, policy()).unwrap(),
            Recall::Unknown
        );
        assert_eq!(
            decode_readout(&weak, std::iter::empty(), policy()).unwrap(),
            Recall::Unknown
        );
    }
}
