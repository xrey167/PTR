//! The corpus and the requests of R005: synthetic token sequences from a seeded generator,
//! and the request shapes the arms build from them.
//!
//! Sequence `i` is derived from `corpus_generator_seed` and the index `i` alone, so the
//! corpus does not depend on the run's seed; the run seed only orders the requests of a
//! workload and places the in-flight races (`scenario`). The digests of the corpus
//! selection (`selection_digest`) cover the sequences in index order and are what
//! `selection_digest_mismatches` compares with the pinned digests.

use super::super::rng::Rng;
use super::params;
use ptr_ledger::integrity::sha256;
use ptr_model_api::{
    ByteTokenizer, PromptSegment, SegmentKind, SegmentSource, SegmentedModelRequest, TokenId,
    Tokenizer,
};
use ptr_types::{EvidenceId, Generation, KnowledgeObjectId, RequestId, Revision, SemanticContext};

/// Stream index of the prefix every main-workload request shares.
const SHARED_STREAM: u64 = 0;
/// Stream index of the unique suffix of main-workload request `i` is `1 + i`.
const MAIN_UNIQUE_BASE: u64 = 1;
/// Unique prompts of the nodes-only workload: stream `NODES_ONLY_BASE + j`.
const NODES_ONLY_BASE: u64 = 1_000;
/// Unique prompts of the bytes-only workload: stream `BYTES_ONLY_BASE + j`.
const BYTES_ONLY_BASE: u64 = 3_000;
/// Fresh unique requests offered to a saturated catalog: `SATURATION_BASE + k` (nodes-only)
/// and `SATURATION_BASE + 100 + k` (bytes-only).
const SATURATION_BASE: u64 = 5_000;
/// Blocks of the small cases: stream `BLOCK_BASE + label`.
const BLOCK_BASE: u64 = 10_000;

/// `len` tokens of stream `index`: a pure function of the corpus seed, the index and `len`.
pub fn sequence(index: u64, len: usize) -> Vec<u32> {
    let mut rng =
        Rng::new(params::CORPUS_GENERATOR_SEED ^ index.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    (0..len)
        .map(|_| (rng.next_u64() % params::VOCAB_SIZE) as u32)
        .collect()
}

/// One 16-token block of a small case; distinct labels give distinct blocks.
pub fn block(label: u64) -> Vec<u32> {
    sequence(BLOCK_BASE + label, params::CHUNK_TOKENS)
}

/// `blocks` blocks starting at `label`, concatenated.
pub fn blocks(label: u64, count: u64) -> Vec<u32> {
    (label..label + count).flat_map(block).collect()
}

/// The shape of the main workload's prompts: `prompt` tokens of which the first `shared` are
/// common to every request. The pinned shape is 256 and 128.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MainShape {
    pub prompt: usize,
    pub shared: usize,
}

impl MainShape {
    pub const PINNED: MainShape = MainShape {
        prompt: params::PROMPT_TOKENS as usize,
        shared: params::SHARED_PREFIX_TOKENS as usize,
    };
}

/// Prompt `i` of the main workload: the shared prefix, then the tokens unique to `i`.
pub fn main_prompt(index: usize, shape: MainShape) -> Vec<u32> {
    let mut tokens = sequence(SHARED_STREAM, shape.shared);
    tokens.extend(sequence(
        MAIN_UNIQUE_BASE + index as u64,
        shape.prompt - shape.shared,
    ));
    tokens
}

/// Unique prompt `j` of an over-capacity workload, `tokens` long.
pub fn unique_prompt(workload: params::Workload, index: usize, tokens: usize) -> Vec<u32> {
    let base = match workload {
        params::Workload::NodesOnly => NODES_ONLY_BASE,
        params::Workload::BytesOnly => BYTES_ONLY_BASE,
        params::Workload::Main => MAIN_UNIQUE_BASE,
    };
    sequence(base + index as u64, tokens)
}

/// Fresh unique request `k` offered to the saturated catalog of `workload`.
pub fn excess_prompt(workload: params::Workload, index: usize, tokens: usize) -> Vec<u32> {
    let offset = match workload {
        params::Workload::BytesOnly => 100,
        _ => 0,
    };
    sequence(SATURATION_BASE + offset + index as u64, tokens)
}

/// Tokens of a tail no pinned prompt shares: 16 tokens that extend a 128-token prompt to a
/// request longer than the prompt, so a hit on the whole prompt is possible.
pub fn tail(label: u64) -> Vec<u32> {
    sequence(BLOCK_BASE + 5_000 + label, params::CHUNK_TOKENS)
}

/// How many prompts every over-capacity workload offers, in order.
pub fn over_capacity_prompts() -> usize {
    params::OVER_CAPACITY_FILL_PROMPTS + params::OVER_CAPACITY_EXTRA_PROMPTS
}

/// The digest of the corpus selection of `workload`, over its sequences in index order.
/// The text is the digest as `sha256:<hex>`.
pub fn selection_digest(workload: params::Workload) -> String {
    let mut material = Vec::new();
    material.extend_from_slice(params::SELECTION_DOMAIN.as_bytes());
    material.push(0);
    material.extend_from_slice(workload.name().as_bytes());
    material.push(0);
    let mut absorb = |tokens: &[u32]| {
        material.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            material.extend_from_slice(&token.to_le_bytes());
        }
    };
    match workload {
        params::Workload::Main => {
            for index in 0..params::N_REQUESTS as usize {
                absorb(&main_prompt(index, MainShape::PINNED));
            }
        }
        params::Workload::NodesOnly | params::Workload::BytesOnly => {
            for index in 0..over_capacity_prompts() {
                absorb(&unique_prompt(
                    workload,
                    index,
                    params::OVER_CAPACITY_PROMPT_TOKENS,
                ));
            }
            for index in 0..params::SATURATION_EXCESS_REQUESTS {
                absorb(&excess_prompt(
                    workload,
                    index,
                    params::OVER_CAPACITY_PROMPT_TOKENS,
                ));
            }
        }
    }
    format!("sha256:{}", hex(&sha256(&material)))
}

/// Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A seeded permutation of `0..count` (Fisher-Yates), the order a workload runs in.
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
    /// A segment built elsewhere, such as by `PtrRuntime::semantic_prompt_segment`.
    Built(PromptSegment),
}

impl Part {
    pub fn raw(tokens: Vec<u32>) -> Self {
        Self::Raw(tokens)
    }

    pub fn built(segment: PromptSegment) -> Self {
        Self::Built(segment)
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
            Self::Built(segment) => Ok(segment.clone()),
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

    #[test]
    fn a_sequence_is_a_pure_function_of_its_index_and_length() {
        assert_eq!(sequence(7, 32), sequence(7, 32));
        assert_ne!(sequence(7, 32), sequence(8, 32));
        assert_eq!(sequence(7, 16), sequence(7, 32)[..16].to_vec());
        assert!(sequence(3, 4096)
            .iter()
            .all(|token| u64::from(*token) < params::VOCAB_SIZE));
    }

    #[test]
    fn main_prompts_share_exactly_the_pinned_prefix() {
        let (a, b) = (
            main_prompt(0, MainShape::PINNED),
            main_prompt(1, MainShape::PINNED),
        );
        let shared = params::SHARED_PREFIX_TOKENS as usize;
        assert_eq!(a.len(), params::PROMPT_TOKENS as usize);
        assert_eq!(a[..shared], b[..shared]);
        assert_ne!(
            a[shared..shared + params::CHUNK_TOKENS],
            b[shared..shared + params::CHUNK_TOKENS]
        );
    }

    #[test]
    fn over_capacity_prompts_are_unique_and_the_workloads_are_independent() {
        let tokens = params::OVER_CAPACITY_PROMPT_TOKENS;
        let nodes = unique_prompt(params::Workload::NodesOnly, 0, tokens);
        let bytes = unique_prompt(params::Workload::BytesOnly, 0, tokens);
        assert_ne!(nodes, bytes);
        let mut seen = std::collections::BTreeSet::new();
        for index in 0..over_capacity_prompts() {
            for chunk in unique_prompt(params::Workload::NodesOnly, index, tokens)
                .chunks(params::CHUNK_TOKENS)
            {
                assert!(
                    seen.insert(chunk.to_vec()),
                    "a repeated chunk in prompt {index}"
                );
            }
        }
    }

    #[test]
    fn the_selection_digests_are_stable_and_distinct() {
        let digests: Vec<String> = params::Workload::ALL
            .iter()
            .map(|w| selection_digest(*w))
            .collect();
        assert_eq!(
            digests,
            params::Workload::ALL
                .iter()
                .map(|w| selection_digest(*w))
                .collect::<Vec<_>>()
        );
        assert_ne!(digests[0], digests[1]);
        assert_ne!(digests[1], digests[2]);
        assert!(digests[0].starts_with("sha256:") && digests[0].len() == 7 + 64);
    }

    #[test]
    fn an_order_is_a_seeded_permutation() {
        let a = order(17, 1, 64);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_eq!(a, order(17, 1, 64));
        assert_ne!(a, order(29, 1, 64));
        assert_ne!(a, order(17, 2, 64));
    }

    #[test]
    fn requests_carry_their_parts_and_lineage() {
        let tokens = block(1);
        let bound = request(
            "r",
            &[
                Part::bound(SegmentKind::Memory, "mem:one", 1, tokens.clone()),
                Part::raw(block(2)),
            ],
        );
        assert_eq!(bound.total_tokens(), 32);
        assert_eq!(bound.segments().len(), 2);
        assert_eq!(bound.tokens().len(), 32);
        let raw = raw_request("r", &tokens);
        assert_eq!(raw.segments().len(), 1);
    }
}
