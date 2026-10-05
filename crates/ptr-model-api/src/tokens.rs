//! Token-aware request contract for KV-cache reuse (doc 36, phase P1).
//!
//! This module is contract only: it defines token ids, the tokenizer port, prompt
//! segments with their lineage contribution, the segmented request, the reuse hint and
//! report types and the [`KvReusingBackend`] trait. No existing backend implements it,
//! and nothing here enforces reuse admission; that is P2 and later.
//!
//! # Exact versus Approximate
//!
//! A reuse hit is either [`KvReuseMode::Exact`] (the reused state is what a fresh prefill
//! of the same tokens and lineage would have produced) or [`KvReuseMode::Approximate`].
//! **Output derived from an Approximate hit must never reach an effect path.** The types
//! document that rule; enforcement belongs to the runtime and cache phases (P2+).

use crate::{ModelError, ModelEvent};
use ptr_types::{Digest, EvidenceId, Generation, KnowledgeObjectId, RequestId, SemanticContext};
use sha2::{Digest as _, Sha256};
use std::fmt;

const SEGMENT_DOMAIN: &[u8] = b"ptr.kv.segment.v1\0";
const LINEAGE_DOMAIN: &[u8] = b"ptr.kv.lineage.v2\0";

/// One token id in a tokenizer's vocabulary.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenId(pub u32);

/// Contract violations raised by constructors in this module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenContractError {
    EmptyTokenizerName,
    EmptyTokenizerVersion,
    EmptySemanticKey,
    EmptySourceId,
    /// `Raw` was passed to a constructor that is not `raw_from_*`.
    RawSourceNeedsRawConstructor,
    /// Approximate reuse reported although the policy is `ExactOnly`.
    ApproximateNotAllowed,
    EmptySegment,
    EmptySegmentList,
    TokenizerMismatch {
        expected: TokenizerIdentity,
        found: TokenizerIdentity,
    },
    HitExceedsTotal {
        hit_tokens: usize,
        total_tokens: usize,
    },
}

impl fmt::Display for TokenContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTokenizerName => write!(f, "tokenizer name is empty"),
            Self::EmptyTokenizerVersion => write!(f, "tokenizer version is empty"),
            Self::EmptySemanticKey => write!(f, "semantic segment key is empty"),
            Self::EmptySourceId => write!(f, "memory or derived segment id is empty"),
            Self::RawSourceNeedsRawConstructor => {
                write!(
                    f,
                    "Raw segments must be built with raw_from_text or raw_from_tokens"
                )
            }
            Self::ApproximateNotAllowed => {
                write!(f, "approximate reuse reported under an ExactOnly policy")
            }
            Self::EmptySegment => write!(f, "prompt segment has no tokens"),
            Self::EmptySegmentList => write!(f, "segmented request has no segments"),
            Self::TokenizerMismatch { expected, found } => write!(
                f,
                "segment tokenizer {found} differs from request tokenizer {expected}"
            ),
            Self::HitExceedsTotal {
                hit_tokens,
                total_tokens,
            } => write!(
                f,
                "hit of {hit_tokens} tokens exceeds total of {total_tokens}"
            ),
        }
    }
}

impl std::error::Error for TokenContractError {}

/// Failure to turn token ids back into text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizeError {
    /// The id is outside the tokenizer's vocabulary.
    UnknownToken(TokenId),
    /// The decoded bytes are not valid UTF-8.
    InvalidUtf8,
}

impl fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownToken(id) => write!(f, "token id {} is not in the vocabulary", id.0),
            Self::InvalidUtf8 => write!(f, "decoded bytes are not valid UTF-8"),
        }
    }
}

impl std::error::Error for TokenizeError {}

/// Name and version of a tokenizer; token ids are only comparable within one identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenizerIdentity {
    name: String,
    version: String,
}

impl TokenizerIdentity {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self, TokenContractError> {
        let name = name.into();
        let version = version.into();
        if name.trim().is_empty() {
            return Err(TokenContractError::EmptyTokenizerName);
        }
        if version.trim().is_empty() {
            return Err(TokenContractError::EmptyTokenizerVersion);
        }
        Ok(Self { name, version })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

impl fmt::Display for TokenizerIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Text to token ids and back. A real HF tokenizer arrives later in `ptr-tokenizers`.
pub trait Tokenizer: Send + Sync {
    fn identity(&self) -> &TokenizerIdentity;
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> Result<String, TokenizeError>;
}

/// One token per UTF-8 byte. Identity `ptr.byte` version `1`.
#[derive(Clone, Debug)]
pub struct ByteTokenizer {
    identity: TokenizerIdentity,
}

impl ByteTokenizer {
    pub fn new() -> Self {
        Self {
            identity: TokenizerIdentity {
                name: "ptr.byte".to_owned(),
                version: "1".to_owned(),
            },
        }
    }
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for ByteTokenizer {
    fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    fn encode(&self, text: &str) -> Vec<TokenId> {
        text.bytes().map(|b| TokenId(u32::from(b))).collect()
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String, TokenizeError> {
        let bytes = tokens
            .iter()
            .map(|t| u8::try_from(t.0).map_err(|_| TokenizeError::UnknownToken(*t)))
            .collect::<Result<Vec<u8>, _>>()?;
        String::from_utf8(bytes).map_err(|_| TokenizeError::InvalidUtf8)
    }
}

/// Where a prompt segment came from, with the identity and generation that make it
/// revocable. `Raw` carries neither: it is content-bound only (doc 36 section D).
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum SegmentSource {
    /// Caller-supplied text that need not be revocable. Runtime output is never Raw.
    Raw,
    /// A semantic snapshot entry, by key.
    Semantic { key: String, generation: Generation },
    /// A lifecycle-managed memory object.
    Memory {
        id: KnowledgeObjectId,
        generation: Generation,
    },
    /// Runtime-produced content (model, tool or derived text) with its provenance.
    /// Approximate taint propagates through this kind until an exact recomputation.
    Derived {
        id: EvidenceId,
        generation: Generation,
    },
}

/// Kind tag of a [`SegmentSource`], used in lineage contributions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SegmentKind {
    Raw,
    Semantic,
    Memory,
    Derived,
}

impl SegmentKind {
    fn tag(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::Semantic => 1,
            Self::Memory => 2,
            Self::Derived => 3,
        }
    }
}

impl SegmentSource {
    pub fn kind(&self) -> SegmentKind {
        match self {
            Self::Raw => SegmentKind::Raw,
            Self::Semantic { .. } => SegmentKind::Semantic,
            Self::Memory { .. } => SegmentKind::Memory,
            Self::Derived { .. } => SegmentKind::Derived,
        }
    }

    fn validate(&self) -> Result<(), TokenContractError> {
        match self {
            Self::Semantic { key, .. } if key.is_empty() => {
                Err(TokenContractError::EmptySemanticKey)
            }
            Self::Memory { id, .. } if id.0.is_empty() => Err(TokenContractError::EmptySourceId),
            Self::Derived { id, .. } if id.0.is_empty() => Err(TokenContractError::EmptySourceId),
            _ => Ok(()),
        }
    }
}

/// A segment's contribution to a request's lineage. Raw has no identity or generation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum LineageContribution {
    Raw {
        content_digest: Digest,
    },
    Bound {
        kind: SegmentKind,
        identity: String,
        generation: Generation,
        content_digest: Digest,
    },
}

impl LineageContribution {
    pub fn kind(&self) -> SegmentKind {
        match self {
            Self::Raw { .. } => SegmentKind::Raw,
            Self::Bound { kind, .. } => *kind,
        }
    }

    pub fn content_digest(&self) -> &Digest {
        match self {
            Self::Raw { content_digest } | Self::Bound { content_digest, .. } => content_digest,
        }
    }

    fn absorb(&self, hasher: &mut Sha256) {
        hasher.update([self.kind().tag()]);
        if let Self::Bound {
            identity,
            generation,
            ..
        } = self
        {
            put_text(hasher, identity);
            hasher.update(generation.0.to_le_bytes());
        }
        hasher.update(self.content_digest());
    }
}

fn put_text(hasher: &mut Sha256, text: &str) {
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
}

/// A run of tokens with its source and content digest.
///
/// `content_digest` is SHA-256 over a domain tag, the tokenizer identity and the token
/// ids (little-endian `u32`). It does not cover the source; the source enters through
/// [`PromptSegment::lineage_contribution`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptSegment {
    source: SegmentSource,
    tokenizer: TokenizerIdentity,
    tokens: Vec<TokenId>,
    content_digest: Digest,
}

impl PromptSegment {
    /// Encode request raw input with `tokenizer` as a `Raw` segment.
    ///
    /// Use this only for raw input that arrived with the request. Runtime-produced
    /// content (model output, tool output, derived text) must never be Raw; build it
    /// with [`PromptSegment::from_text`] and a `Derived` source. Revocable or forgettable
    /// content must use `Semantic` or `Memory`. The type system cannot tell the two
    /// apart, so full enforcement is the P2 catalog's job.
    pub fn raw_from_text<T: Tokenizer + ?Sized>(
        tokenizer: &T,
        text: &str,
    ) -> Result<Self, TokenContractError> {
        Self::build(
            tokenizer.identity().clone(),
            SegmentSource::Raw,
            tokenizer.encode(text),
        )
    }

    /// Like [`PromptSegment::raw_from_text`] for ids already encoded under `tokenizer`.
    pub fn raw_from_tokens(
        tokenizer: TokenizerIdentity,
        tokens: Vec<TokenId>,
    ) -> Result<Self, TokenContractError> {
        Self::build(tokenizer, SegmentSource::Raw, tokens)
    }

    /// Encode `text` as a `Semantic`, `Memory` or `Derived` segment. Rejects `Raw`.
    pub fn from_text<T: Tokenizer + ?Sized>(
        tokenizer: &T,
        source: SegmentSource,
        text: &str,
    ) -> Result<Self, TokenContractError> {
        Self::from_tokens(tokenizer.identity().clone(), source, tokenizer.encode(text))
    }

    /// Build a `Semantic`, `Memory` or `Derived` segment from ids already encoded under
    /// `tokenizer`. Rejects `Raw`.
    pub fn from_tokens(
        tokenizer: TokenizerIdentity,
        source: SegmentSource,
        tokens: Vec<TokenId>,
    ) -> Result<Self, TokenContractError> {
        if source == SegmentSource::Raw {
            return Err(TokenContractError::RawSourceNeedsRawConstructor);
        }
        Self::build(tokenizer, source, tokens)
    }

    fn build(
        tokenizer: TokenizerIdentity,
        source: SegmentSource,
        tokens: Vec<TokenId>,
    ) -> Result<Self, TokenContractError> {
        source.validate()?;
        if tokens.is_empty() {
            return Err(TokenContractError::EmptySegment);
        }
        let content_digest = segment_digest(&tokenizer, &tokens);
        Ok(Self {
            source,
            tokenizer,
            tokens,
            content_digest,
        })
    }

    pub fn source(&self) -> &SegmentSource {
        &self.source
    }

    pub fn tokenizer(&self) -> &TokenizerIdentity {
        &self.tokenizer
    }

    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn content_digest(&self) -> &Digest {
        &self.content_digest
    }

    /// Raw: (kind, content digest) only. Others: kind, identity, generation and digest.
    pub fn lineage_contribution(&self) -> LineageContribution {
        let content_digest = self.content_digest;
        match &self.source {
            SegmentSource::Raw => LineageContribution::Raw { content_digest },
            SegmentSource::Semantic { key, generation } => LineageContribution::Bound {
                kind: SegmentKind::Semantic,
                identity: key.clone(),
                generation: *generation,
                content_digest,
            },
            SegmentSource::Memory { id, generation } => LineageContribution::Bound {
                kind: SegmentKind::Memory,
                identity: id.0.clone(),
                generation: *generation,
                content_digest,
            },
            SegmentSource::Derived { id, generation } => LineageContribution::Bound {
                kind: SegmentKind::Derived,
                identity: id.0.clone(),
                generation: *generation,
                content_digest,
            },
        }
    }
}

fn segment_digest(tokenizer: &TokenizerIdentity, tokens: &[TokenId]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(SEGMENT_DOMAIN);
    put_text(&mut hasher, tokenizer.name());
    put_text(&mut hasher, tokenizer.version());
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(token.0.to_le_bytes());
    }
    hasher.finalize().into()
}

/// A model request whose prompt is an ordered list of provenance-tagged segments.
/// It is separate from `ModelRequest`, which is unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentedModelRequest {
    request_id: RequestId,
    semantic_context: SemanticContext,
    tokenizer: TokenizerIdentity,
    segments: Vec<PromptSegment>,
}

impl SegmentedModelRequest {
    /// Rejects an empty segment list and any segment encoded under another tokenizer.
    pub fn new(
        request_id: RequestId,
        semantic_context: SemanticContext,
        tokenizer: TokenizerIdentity,
        segments: Vec<PromptSegment>,
    ) -> Result<Self, TokenContractError> {
        if segments.is_empty() {
            return Err(TokenContractError::EmptySegmentList);
        }
        if let Some(bad) = segments.iter().find(|s| s.tokenizer() != &tokenizer) {
            return Err(TokenContractError::TokenizerMismatch {
                expected: tokenizer,
                found: bad.tokenizer().clone(),
            });
        }
        Ok(Self {
            request_id,
            semantic_context,
            tokenizer,
            segments,
        })
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn semantic_context(&self) -> &SemanticContext {
        &self.semantic_context
    }

    pub fn tokenizer(&self) -> &TokenizerIdentity {
        &self.tokenizer
    }

    pub fn segments(&self) -> &[PromptSegment] {
        &self.segments
    }

    pub fn revision(&self) -> ptr_types::Revision {
        self.semantic_context.revision
    }

    /// Concatenation of every segment's tokens, in order.
    pub fn tokens(&self) -> Vec<TokenId> {
        self.segments
            .iter()
            .flat_map(|s| s.tokens().iter().copied())
            .collect()
    }

    pub fn total_tokens(&self) -> usize {
        self.segments.iter().map(|s| s.tokens().len()).sum()
    }

    /// Order-dependent SHA-256 over the semantic revision, the tokenizer identity and the
    /// segments' lineage contributions, so identical raw tokens under different revisions
    /// never share a digest.
    pub fn lineage_digest(&self) -> Digest {
        let mut hasher = Sha256::new();
        hasher.update(LINEAGE_DOMAIN);
        hasher.update(self.semantic_context.revision.0.to_le_bytes());
        put_text(&mut hasher, self.tokenizer.name());
        put_text(&mut hasher, self.tokenizer.version());
        hasher.update((self.segments.len() as u64).to_le_bytes());
        for segment in &self.segments {
            segment.lineage_contribution().absorb(&mut hasher);
        }
        hasher.finalize().into()
    }
}

/// Strength of a KV reuse hit. Approximate output must never reach an effect path;
/// enforcement is P2+.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvReuseMode {
    Exact,
    Approximate,
}

/// What reuse modes the caller accepts.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KvReusePolicy {
    ExactOnly,
    /// Only for paths that can never produce effects.
    AllowApproximate,
}

/// Caller's request to the backend about reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvReuseHint {
    /// Longest prefix, in tokens, the caller expects to be reusable.
    pub requested_prefix_tokens: usize,
    pub policy: KvReusePolicy,
}

/// What the backend actually reused. `hit_tokens` never exceeds the request's total.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvReuseReport {
    hit_tokens: usize,
    mode: KvReuseMode,
}

impl KvReuseReport {
    pub fn new(
        hit_tokens: usize,
        mode: KvReuseMode,
        total_tokens: usize,
        policy: KvReusePolicy,
    ) -> Result<Self, TokenContractError> {
        if mode == KvReuseMode::Approximate && policy == KvReusePolicy::ExactOnly {
            return Err(TokenContractError::ApproximateNotAllowed);
        }
        if hit_tokens > total_tokens {
            return Err(TokenContractError::HitExceedsTotal {
                hit_tokens,
                total_tokens,
            });
        }
        Ok(Self { hit_tokens, mode })
    }

    pub fn hit_tokens(&self) -> usize {
        self.hit_tokens
    }

    pub fn mode(&self) -> KvReuseMode {
        self.mode
    }
}

/// Result of a segmented inference: the events and, when the backend reused a cache,
/// the report of that reuse.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelRun {
    pub events: Vec<ModelEvent>,
    pub kv_report: Option<KvReuseReport>,
}

/// Backend that accepts a segmented request and may reuse cached K/V state.
/// Existing `InferenceBackend` implementations are unaffected.
pub trait KvReusingBackend: Send + Sync {
    fn name(&self) -> &str;
    fn infer_segmented(
        &self,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
    ) -> Result<ModelRun, ModelError>;
}
