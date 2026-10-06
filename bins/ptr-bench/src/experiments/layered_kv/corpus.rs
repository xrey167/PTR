//! The corpus and the requests of the layered-KV experiments: synthetic token sequences from
//! a seeded generator and the request shapes the arms build from them.
//!
//! Sequence `i` is a pure function of `corpus_generator_seed` and the stream index `i`, so the
//! corpus does not depend on the run's seed. The run seed only orders the arms and places the
//! in-flight race points (`order`, `scenario`). The digest of a corpus selection covers the
//! sequences in stream order and is what `selection_digest_mismatches` compares with the
//! pinned digest.

use super::super::rng::Rng;
use ptr_ledger::integrity::sha256;
use ptr_model_api::{
    ByteTokenizer, PromptSegment, SegmentKind, SegmentSource, SegmentedModelRequest, TokenId,
    Tokenizer,
};
use ptr_types::{EvidenceId, Generation, KnowledgeObjectId, RequestId, Revision, SemanticContext};

/// Tokens of one KV page and of one catalog chunk.
pub const CHUNK_TOKENS: usize = 16;
/// The byte tokenizer's vocabulary.
pub const VOCAB: u64 = 256;

/// `len` tokens of stream `index`: a pure function of the corpus seed, the index and `len`.
pub fn sequence(corpus_seed: u64, index: u64, len: usize) -> Vec<u32> {
    let mut rng = Rng::new(corpus_seed ^ index.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    (0..len).map(|_| (rng.next_u64() % VOCAB) as u32).collect()
}

/// The digest of a corpus selection: `domain`, then every sequence length-prefixed in stream
/// order. Text form `sha256:<hex>`.
pub fn selection_digest(domain: &str, sequences: &[Vec<u32>]) -> String {
    let mut material = Vec::new();
    material.extend_from_slice(domain.as_bytes());
    material.push(0);
    material.extend_from_slice(&(sequences.len() as u64).to_le_bytes());
    for tokens in sequences {
        material.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            material.extend_from_slice(&token.to_le_bytes());
        }
    }
    format!("sha256:{}", hex(&sha256(&material)))
}

/// Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A seeded permutation of `0..count` (Fisher-Yates), the order a set of arms runs in.
pub fn order(seed: u64, label: u64, count: usize) -> Vec<usize> {
    let mut rng = Rng::new(seed ^ label.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut indexes: Vec<usize> = (0..count).collect();
    for position in (1..count).rev() {
        indexes.swap(position, rng.index(position + 1));
    }
    indexes
}

/// The run seed's source of scenario choices (race points), one stream per label.
pub fn scenario(seed: u64, label: &str) -> Rng {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for byte in label.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    Rng::new(seed ^ hash)
}

/// One segment of a request.
#[derive(Clone, Debug)]
pub enum Part {
    Raw(Vec<u32>),
    Bound {
        kind: SegmentKind,
        identity: String,
        generation: u64,
        tokens: Vec<u32>,
    },
}

impl Part {
    pub fn raw(tokens: Vec<u32>) -> Self {
        Self::Raw(tokens)
    }

    pub fn bound(kind: SegmentKind, identity: &str, generation: u64, tokens: Vec<u32>) -> Self {
        Self::Bound {
            kind,
            identity: identity.to_string(),
            generation,
            tokens,
        }
    }

    fn segment(&self) -> PromptSegment {
        let tokenizer = ByteTokenizer::new();
        let ids = |tokens: &[u32]| {
            tokens
                .iter()
                .map(|token| TokenId(*token))
                .collect::<Vec<_>>()
        };
        match self {
            Self::Raw(tokens) => {
                PromptSegment::raw_from_tokens(tokenizer.identity().clone(), ids(tokens))
            }
            Self::Bound {
                kind,
                identity,
                generation,
                tokens,
            } => {
                let generation = Generation(*generation);
                let source = match kind {
                    SegmentKind::Semantic => SegmentSource::Semantic {
                        key: identity.clone(),
                        generation,
                    },
                    SegmentKind::Memory => SegmentSource::Memory {
                        id: KnowledgeObjectId::from(identity.as_str()),
                        generation,
                    },
                    SegmentKind::Derived => SegmentSource::Derived {
                        id: EvidenceId::from(identity.as_str()),
                        generation,
                    },
                    SegmentKind::Raw => SegmentSource::Raw,
                };
                PromptSegment::from_tokens(tokenizer.identity().clone(), source, ids(tokens))
            }
        }
        .expect("a valid segment: non-empty tokens and a named source")
    }
}

/// A request of `parts` under the byte tokenizer.
pub fn request(id: &str, parts: &[Part]) -> SegmentedModelRequest {
    SegmentedModelRequest::new(
        RequestId::from(id),
        SemanticContext {
            revision: Revision(1),
            entries: Vec::new(),
        },
        ByteTokenizer::new().identity().clone(),
        parts.iter().map(Part::segment).collect(),
    )
    .expect("a valid request: segments under one tokenizer")
}

/// A request of one raw segment.
pub fn raw_request(id: &str, tokens: &[u32]) -> SegmentedModelRequest {
    request(id, &[Part::raw(tokens.to_vec())])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 20_240_607;

    #[test]
    fn a_sequence_is_a_pure_function_of_its_index_and_length() {
        assert_eq!(sequence(SEED, 7, 32), sequence(SEED, 7, 32));
        assert_ne!(sequence(SEED, 7, 32), sequence(SEED, 8, 32));
        assert_ne!(sequence(SEED, 7, 32), sequence(SEED + 1, 7, 32));
        assert_eq!(sequence(SEED, 7, 16), sequence(SEED, 7, 32)[..16].to_vec());
        assert!(sequence(SEED, 3, 4096)
            .iter()
            .all(|token| u64::from(*token) < VOCAB));
    }

    #[test]
    fn a_selection_digest_depends_on_domain_order_and_content() {
        let a = vec![sequence(SEED, 1, 8), sequence(SEED, 2, 8)];
        let digest = selection_digest("d", &a);
        assert_eq!(digest, selection_digest("d", &a));
        assert_ne!(digest, selection_digest("e", &a));
        assert_ne!(
            digest,
            selection_digest("d", &[a[1].clone(), a[0].clone()])
        );
        assert!(digest.starts_with("sha256:") && digest.len() == 7 + 64);
    }

    #[test]
    fn an_order_is_a_seeded_permutation() {
        let a = order(17, 1, 6);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..6).collect::<Vec<_>>());
        assert_eq!(a, order(17, 1, 6));
        assert_ne!(order(17, 1, 64), order(29, 1, 64));
    }

    #[test]
    fn requests_carry_their_parts_and_lineage() {
        let bound = request(
            "r",
            &[
                Part::bound(SegmentKind::Memory, "mem:one", 1, sequence(SEED, 1, 16)),
                Part::raw(sequence(SEED, 2, 16)),
            ],
        );
        assert_eq!(bound.total_tokens(), 32);
        assert_eq!(bound.segments().len(), 2);
        assert_eq!(raw_request("r", &sequence(SEED, 1, 16)).segments().len(), 1);
    }
}
