//! Position-independent secondary KV index (R006 WP1).
//!
//! The primary prefix catalog ([`crate::kv_prefix`]) finds K/V by an ordered token prefix,
//! so a segment that moved to another offset can never hit. This index finds K/V by the
//! *content* of one admitted prompt segment, wherever that segment now sits, and hands back
//! the stored rows so a consumer can splice them in at the new positions.
//!
//! # Soundness boundary
//!
//! The index stores layer-0 rows with `PreRope` keys: a pure, deterministic function of the
//! token id, independent of position, neighbours and cache content. Re-using such rows at a
//! different position is bit-equal to a fresh prefill only when the consumer rotates them
//! at use (see the plan, section 2). Whether that holds is the caller's contract; this
//! module only guarantees what it can check: exact tokens, exact lineage, liveness of every
//! recorded dependency at the time of the lookup, and the multiset presence of the
//! original prefix.
//!
//! # Key
//!
//! * The namespace is `H(salt, chain(0), mode)`. It has **no parent**: a run is found
//!   without knowing what preceded it.
//! * Inside a namespace a polynomial rolling fingerprint (`u64`, modulo `2^61 - 1`, base
//!   derived from the namespace hash) over every stride-1 window of `W` tokens points to
//!   `(run, offset)`. A run shorter than `W` is keyed once by `H(len, tokens)`.
//! * The fingerprint is a locator only. A candidate is accepted only after its tokens are
//!   compared exactly, so a fingerprint collision can never produce a hit and never
//!   produces a refusal either.
//!
//! # Value
//!
//! A [`SecondaryRows`] pair of bit-exact `f32` buffers and the metadata that makes the
//! reuse admissible: the segment's own lineage (kind, identity, generation, digest), the
//! multiset of bound segments and of raw segments that preceded it in the original request,
//! the dependencies it was computed from (those segments, itself, and caller-declared
//! extras such as the base model, an adapter or an expert with its fence epoch), the mode,
//! pins, a tombstone flag and a logical use clock.
//!
//! # Admission and the typed refusals
//!
//! One run is one whole segment. An edit inside a segment changes its digest, so the run
//! is refused ([`SecondaryRefusalReason::SourceDigestMismatch`]); a span with no full match
//! is recomputed, never partly reused. Candidates are judged in a fixed order and the first
//! failing step is the one refusal:
//!
//! 1. unusable (tombstoned, rows missing): a silent miss, not a refusal;
//! 2. own lineage against the request segment: kind, identity, generation, digest;
//! 3. [`SecondaryRefusalReason::OriginalUpstreamDead`]: an upstream or own bound
//!    dependency is not live (revoked, superseded or never committed);
//! 4. [`SecondaryRefusalReason::BindingDependencyDead`]: a declared extra is not live;
//! 5. [`SecondaryRefusalReason::OriginalUpstreamOmitted`]: some original-prefix bound
//!    segment has a smaller count in the **whole** request than in the run (the span's own
//!    occurrence is not counted for it; reordering is allowed);
//! 6. [`SecondaryRefusalReason::RawAncestorOmitted`]: the same over raw segments.
//!
//! A run found dead is tombstoned at once. A splice never covers the last prompt token,
//! because the consumer needs that token's logits.
//!
//! # Bounds
//!
//! [`SecondaryLimits`] bounds runs, row bytes, indexed windows, tokens per run, metadata
//! bytes and candidates per window. An insert evicts unpinned runs in least-recently-used
//! order and returns [`SecondaryError::CapacityExhausted`] when only pinned runs remain; a
//! pinned run is never evicted, and nothing is evicted when the insert cannot fit anyway.
//!
//! # Snapshot
//!
//! [`SecondaryIndex::export_section`] writes canonical, bounded bytes of metadata and
//! `rows_digest` only. [`SecondaryIndex::restore_section`] decodes them strictly into runs
//! without rows; those runs miss until an insert of the same content under a live oracle
//! attaches rows whose digest matches.

use crate::kv_prefix::{
    KvChainId, KvDependency, KvDependencyKind, KvLivenessOracle, KvRequestPositions, KvSalt,
};
use ptr_model_api::{
    KvReuseMode, LineageContribution, SegmentKind, SegmentedModelRequest, TokenId,
};
use ptr_types::{Digest, Generation};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

const NAMESPACE_DOMAIN: &[u8] = b"ptr.kv.secondary.namespace.v1\0";
const SHORT_DOMAIN: &[u8] = b"ptr.kv.secondary.short.v1\0";
const RUN_KEY_DOMAIN: &[u8] = b"ptr.kv.secondary.run.v1\0";
const ROWS_DOMAIN: &[u8] = b"ptr.kv.secondary.rows.v1\0";

/// Magic of the exported section.
const SECTION_MAGIC: &[u8; 8] = b"PTRSEC01";
/// Largest section [`SecondaryIndex::restore_section`] reads.
pub const SECONDARY_SECTION_MAX_BYTES: usize = 16 * 1024 * 1024;
const SECTION_MAX_TEXT_BYTES: usize = 4096;
const SECTION_MAX_ENTRIES: usize = 65_536;

const RUN_METADATA_BYTES: u64 = 256;
const DEPENDENCY_METADATA_BYTES: u64 = 96;
const TOKEN_METADATA_BYTES: u64 = 4;

/// The Mersenne prime `2^61 - 1`, the modulus of the rolling fingerprint.
const FINGERPRINT_MODULUS: u64 = (1 << 61) - 1;
/// Smallest rolling base, so a base is never a degenerate small value.
const MIN_BASE: u64 = 1 << 16;

/// Default tokens per rolling window; R006 pins 8.
pub const DEFAULT_WINDOW_TOKENS: usize = 8;

// ---------------------------------------------------------------------------------------
// Rolling fingerprint
// ---------------------------------------------------------------------------------------

fn reduce(value: u128) -> u64 {
    let modulus = u128::from(FINGERPRINT_MODULUS);
    let folded = (value & modulus) + (value >> 61);
    let folded = (folded & modulus) + (folded >> 61);
    let folded = if folded >= modulus {
        folded - modulus
    } else {
        folded
    };
    // After the folds the value is below the modulus, so it fits a u64.
    u64::try_from(folded).unwrap_or(0)
}

fn mul_mod(a: u64, b: u64) -> u64 {
    reduce(u128::from(a) * u128::from(b))
}

fn add_mod(a: u64, b: u64) -> u64 {
    reduce(u128::from(a) + u128::from(b))
}

fn sub_mod(a: u64, b: u64) -> u64 {
    add_mod(a, FINGERPRINT_MODULUS - b)
}

fn pow_mod(base: u64, mut exponent: usize) -> u64 {
    let mut result = 1;
    let mut square = base;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, square);
        }
        square = mul_mod(square, square);
        exponent >>= 1;
    }
    result
}

/// The value a token contributes. `+ 1` keeps token 0 from vanishing at the leading
/// position.
fn token_value(token: TokenId) -> u64 {
    u64::from(token.0) + 1
}

/// Fingerprint of one window, computed from scratch.
fn fingerprint(window: &[TokenId], base: u64) -> u64 {
    window.iter().fold(0, |hash, token| {
        add_mod(mul_mod(hash, base), token_value(*token))
    })
}

/// Fingerprints of every stride-1 window of `width` tokens, by the rolling recurrence.
fn window_fingerprints(tokens: &[TokenId], width: usize, base: u64) -> Vec<u64> {
    if width == 0 || tokens.len() < width {
        return Vec::new();
    }
    let top = pow_mod(base, width - 1);
    let mut hash = fingerprint(&tokens[..width], base);
    let mut out = Vec::with_capacity(tokens.len() - width + 1);
    out.push(hash);
    for index in width..tokens.len() {
        let leaving = mul_mod(token_value(tokens[index - width]), top);
        hash = add_mod(
            mul_mod(sub_mod(hash, leaving), base),
            token_value(tokens[index]),
        );
        out.push(hash);
    }
    out
}

fn short_key(tokens: &[TokenId]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SHORT_DOMAIN);
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(token.0.to_le_bytes());
    }
    hasher.finalize().into()
}

// ---------------------------------------------------------------------------------------
// Public value types
// ---------------------------------------------------------------------------------------

/// Identifier of a run inside one index. Strictly increasing with insertion order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SecondaryRunId(pub u64);

/// Handle on a pinned run. Release it with [`SecondaryIndex::release`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SecondaryPin {
    run: SecondaryRunId,
    id: u64,
}

impl SecondaryPin {
    pub fn run(&self) -> SecondaryRunId {
        self.run
    }
}

/// Whether a pinned run may still be consumed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecondaryPinState {
    Live,
    /// The run was tombstoned (invalidated or found dead) or is gone.
    Revoked,
}

/// Bit-exact layer-0 key and value rows of one run: `tokens * row_width` `f32` each.
#[derive(Clone, Debug)]
pub struct SecondaryRows {
    pub keys: Arc<[f32]>,
    pub values: Arc<[f32]>,
    /// `f32` values per token row (`kv_heads * head_dim`).
    pub row_width: usize,
}

impl SecondaryRows {
    pub fn new(keys: Arc<[f32]>, values: Arc<[f32]>, row_width: usize) -> Self {
        Self {
            keys,
            values,
            row_width,
        }
    }

    /// Number of token rows the buffers hold, or 0 for a zero width.
    pub fn row_count(&self) -> usize {
        self.keys.len().checked_div(self.row_width).unwrap_or(0)
    }

    fn bytes(&self) -> u64 {
        ((self.keys.len() + self.values.len()) as u64) * 4
    }

    /// SHA-256 over a domain tag, the row width, the row count and every `f32` bit pattern.
    pub fn digest(&self) -> Digest {
        let mut hasher = Sha256::new();
        hasher.update(ROWS_DOMAIN);
        hasher.update((self.row_width as u64).to_le_bytes());
        hasher.update((self.row_count() as u64).to_le_bytes());
        for value in self.keys.iter().chain(self.values.iter()) {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.finalize().into()
    }
}

impl PartialEq for SecondaryRows {
    fn eq(&self, other: &Self) -> bool {
        self.row_width == other.row_width
            && self.keys.len() == other.keys.len()
            && self.values.len() == other.values.len()
            && self
                .keys
                .iter()
                .zip(other.keys.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for SecondaryRows {}

/// Capacity bounds of one index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SecondaryLimits {
    pub max_runs: usize,
    pub max_row_bytes: u64,
    pub max_indexed_windows: u64,
    pub max_tokens_per_run: usize,
    pub max_metadata_bytes: u64,
    pub max_candidates_per_window: usize,
}

impl Default for SecondaryLimits {
    fn default() -> Self {
        Self {
            max_runs: 4096,
            max_row_bytes: 128 * 1024 * 1024,
            max_indexed_windows: 1 << 20,
            max_tokens_per_run: 8192,
            max_metadata_bytes: 8 * 1024 * 1024,
            max_candidates_per_window: 64,
        }
    }
}

/// Construction parameters of an index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SecondaryConfig {
    /// Tokens per rolling window (`W`). R006 pins 8.
    pub window_tokens: usize,
    pub limits: SecondaryLimits,
}

impl Default for SecondaryConfig {
    fn default() -> Self {
        Self {
            window_tokens: DEFAULT_WINDOW_TOKENS,
            limits: SecondaryLimits::default(),
        }
    }
}

/// The bound that refused an insert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecondaryResource {
    Runs,
    RowBytes,
    IndexedWindows,
    TokensPerRun,
    MetadataBytes,
    CandidatesPerWindow,
}

/// Why a candidate run was refused, in the order the checks run.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SecondaryRefusalReason {
    SourceKindMismatch,
    SourceIdentityMismatch,
    SourceGenerationMismatch,
    SourceDigestMismatch,
    OriginalUpstreamDead,
    BindingDependencyDead,
    OriginalUpstreamOmitted,
    RawAncestorOmitted,
}

impl SecondaryRefusalReason {
    pub const ALL: [Self; 8] = [
        Self::SourceKindMismatch,
        Self::SourceIdentityMismatch,
        Self::SourceGenerationMismatch,
        Self::SourceDigestMismatch,
        Self::OriginalUpstreamDead,
        Self::BindingDependencyDead,
        Self::OriginalUpstreamOmitted,
        Self::RawAncestorOmitted,
    ];

    /// Stable machine-readable code.
    pub fn code(self) -> &'static str {
        match self {
            Self::SourceKindMismatch => "source_kind_mismatch",
            Self::SourceIdentityMismatch => "source_identity_mismatch",
            Self::SourceGenerationMismatch => "source_generation_mismatch",
            Self::SourceDigestMismatch => "source_digest_mismatch",
            Self::OriginalUpstreamDead => "original_upstream_dead",
            Self::BindingDependencyDead => "binding_dependency_dead",
            Self::OriginalUpstreamOmitted => "original_upstream_omitted",
            Self::RawAncestorOmitted => "raw_ancestor_omitted",
        }
    }
}

impl fmt::Display for SecondaryRefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Number of refusals per reason.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SecondaryRefusalCounts {
    pub source_kind_mismatch: u64,
    pub source_identity_mismatch: u64,
    pub source_generation_mismatch: u64,
    pub source_digest_mismatch: u64,
    pub original_upstream_dead: u64,
    pub binding_dependency_dead: u64,
    pub original_upstream_omitted: u64,
    pub raw_ancestor_omitted: u64,
}

impl SecondaryRefusalCounts {
    pub fn get(&self, reason: SecondaryRefusalReason) -> u64 {
        match reason {
            SecondaryRefusalReason::SourceKindMismatch => self.source_kind_mismatch,
            SecondaryRefusalReason::SourceIdentityMismatch => self.source_identity_mismatch,
            SecondaryRefusalReason::SourceGenerationMismatch => self.source_generation_mismatch,
            SecondaryRefusalReason::SourceDigestMismatch => self.source_digest_mismatch,
            SecondaryRefusalReason::OriginalUpstreamDead => self.original_upstream_dead,
            SecondaryRefusalReason::BindingDependencyDead => self.binding_dependency_dead,
            SecondaryRefusalReason::OriginalUpstreamOmitted => self.original_upstream_omitted,
            SecondaryRefusalReason::RawAncestorOmitted => self.raw_ancestor_omitted,
        }
    }

    pub fn total(&self) -> u64 {
        SecondaryRefusalReason::ALL
            .iter()
            .map(|reason| self.get(*reason))
            .sum()
    }

    fn bump(&mut self, reason: SecondaryRefusalReason) {
        let slot = match reason {
            SecondaryRefusalReason::SourceKindMismatch => &mut self.source_kind_mismatch,
            SecondaryRefusalReason::SourceIdentityMismatch => &mut self.source_identity_mismatch,
            SecondaryRefusalReason::SourceGenerationMismatch => {
                &mut self.source_generation_mismatch
            }
            SecondaryRefusalReason::SourceDigestMismatch => &mut self.source_digest_mismatch,
            SecondaryRefusalReason::OriginalUpstreamDead => &mut self.original_upstream_dead,
            SecondaryRefusalReason::BindingDependencyDead => &mut self.binding_dependency_dead,
            SecondaryRefusalReason::OriginalUpstreamOmitted => &mut self.original_upstream_omitted,
            SecondaryRefusalReason::RawAncestorOmitted => &mut self.raw_ancestor_omitted,
        };
        *slot += 1;
    }
}

/// Counters and gauges of one index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SecondaryStats {
    // Gauges.
    pub runs: u64,
    pub tombstoned_pinned_runs: u64,
    pub row_bytes: u64,
    pub indexed_windows: u64,
    pub metadata_bytes: u64,
    pub active_pins: u64,
    // Counters.
    pub inserts: u64,
    pub duplicate_inserts: u64,
    pub rows_attached: u64,
    pub evictions: u64,
    pub capacity_refusals: u64,
    pub lookups: u64,
    pub windows_probed: u64,
    pub candidates_examined: u64,
    /// Postings whose exact token comparison failed: fingerprint collisions.
    pub fingerprint_collisions: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub last_token_skips: u64,
    pub tombstoned_skips: u64,
    pub rows_missing_misses: u64,
    /// Refusals of a candidate for a segment that another candidate then served.
    pub suppressed_refusals: u64,
    pub tombstoned_at_lookup: u64,
    pub invalidated: u64,
    pub refusals: SecondaryRefusalCounts,
}

/// Why an operation on the index failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SecondaryError {
    /// A limit is zero or the window is zero or larger than a run may be.
    InvalidConfig,
    /// Only [`KvReuseMode::Exact`] runs are admitted.
    NotExact,
    /// Lookups need canonical positions (start 0, no explicit ids).
    NonCanonicalPosition,
    /// The segment index is not in the request.
    SegmentOutOfRange,
    /// The buffers are not `tokens * row_width` values each, or the width is zero.
    RowShape,
    /// A recorded dependency is not live at insert time.
    DependencyNotLive,
    /// A tombstoned, still pinned run occupies this slot.
    SlotDraining,
    /// The same run key holds different tokens.
    TokensConflict,
    /// The supplied rows do not match the run's `rows_digest`.
    RowsDigestMismatch,
    /// A bound would be exceeded and no unpinned run can make room.
    CapacityExhausted(SecondaryResource),
    /// The exported section is malformed.
    Section(SecondarySectionError),
}

impl fmt::Display for SecondaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "secondary index configuration is invalid"),
            Self::NotExact => write!(f, "only Exact runs are admitted"),
            Self::NonCanonicalPosition => write!(f, "request positions are not canonical"),
            Self::SegmentOutOfRange => write!(f, "segment index is outside the request"),
            Self::RowShape => write!(f, "rows do not match the token count and row width"),
            Self::DependencyNotLive => write!(f, "a run dependency is not live"),
            Self::SlotDraining => write!(f, "the slot is held by a cancelled, still pinned run"),
            Self::TokensConflict => write!(f, "the run key holds different tokens"),
            Self::RowsDigestMismatch => write!(f, "rows do not match the recorded rows digest"),
            Self::CapacityExhausted(resource) => {
                write!(f, "secondary index capacity exhausted ({resource:?})")
            }
            Self::Section(error) => write!(f, "secondary section: {error}"),
        }
    }
}

impl std::error::Error for SecondaryError {}

/// Why an exported section was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecondarySectionError {
    BadMagic,
    Truncated,
    TrailingBytes,
    TooLarge,
    /// A length or count is above its bound.
    OutOfBounds(&'static str),
    /// A value is out of its domain.
    InvalidValue(&'static str),
    /// Entries are not in strictly ascending canonical order, or a derived field differs.
    NotCanonical(&'static str),
}

impl fmt::Display for SecondarySectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic"),
            Self::Truncated => write!(f, "truncated"),
            Self::TrailingBytes => write!(f, "trailing bytes"),
            Self::TooLarge => write!(f, "section is too large"),
            Self::OutOfBounds(what) => write!(f, "{what} is out of bounds"),
            Self::InvalidValue(what) => write!(f, "{what} is invalid"),
            Self::NotCanonical(what) => write!(f, "{what} is not canonical"),
        }
    }
}

impl std::error::Error for SecondarySectionError {}

impl From<SecondarySectionError> for SecondaryError {
    fn from(error: SecondarySectionError) -> Self {
        Self::Section(error)
    }
}

/// The lineage of one segment as the index records it. A `Raw` segment has an empty
/// identity and generation 0.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SecondaryLineage {
    pub kind: SegmentKind,
    pub identity: String,
    pub generation: Generation,
    pub content_digest: Digest,
}

impl SecondaryLineage {
    fn from_contribution(contribution: &LineageContribution) -> Self {
        match contribution {
            LineageContribution::Raw { content_digest } => Self {
                kind: SegmentKind::Raw,
                identity: String::new(),
                generation: Generation(0),
                content_digest: *content_digest,
            },
            LineageContribution::Bound {
                kind,
                identity,
                generation,
                content_digest,
            } => Self {
                kind: *kind,
                identity: identity.clone(),
                generation: *generation,
                content_digest: *content_digest,
            },
        }
    }

    fn dependency(&self) -> Option<KvDependency> {
        match self.kind {
            SegmentKind::Raw => None,
            kind => Some(KvDependency {
                kind: KvDependencyKind::Segment(kind),
                identity: self.identity.clone(),
                generation: self.generation,
                content_digest: self.content_digest,
                fence_epoch: 0,
            }),
        }
    }

    fn metadata_bytes(&self) -> u64 {
        DEPENDENCY_METADATA_BYTES + self.identity.len() as u64
    }
}

/// A finished prompt segment offered to the index.
#[derive(Debug)]
pub struct SecondaryInsert<'a> {
    pub salt: &'a KvSalt,
    /// `chain(0)` of the stack the rows were computed by.
    pub chain: &'a KvChainId,
    pub mode: KvReuseMode,
    pub request: &'a SegmentedModelRequest,
    /// Index of the completed segment inside `request`.
    pub segment: usize,
    pub rows: SecondaryRows,
    /// Dependencies beyond the request's segments: the base model, an adapter, an expert
    /// with its fence epoch. They are checked at insert and at every use.
    pub extra_dependencies: Vec<KvDependency>,
}

/// What an insert did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecondaryInsertOutcome {
    Inserted(SecondaryRunId),
    /// The same run was already present with the same rows.
    AlreadyPresent(SecondaryRunId),
    /// The run was present without rows (restored); the rows matched its digest and are
    /// attached now.
    RowsAttached(SecondaryRunId),
}

/// A lookup of the request's segments, wherever they now sit.
#[derive(Clone, Copy, Debug)]
pub struct SecondaryLookup<'a> {
    pub salt: &'a KvSalt,
    pub chain: &'a KvChainId,
    pub mode: KvReuseMode,
    pub request: &'a SegmentedModelRequest,
    pub positions: &'a KvRequestPositions,
}

/// One admitted run to splice into the request.
#[derive(Clone, Debug)]
pub struct SecondarySplice {
    /// Token offset of the span in the request.
    pub request_start: usize,
    pub len: usize,
    /// `request_start - original_start`: how far the rows move.
    pub delta: i64,
    pub original_start: usize,
    pub run: SecondaryRunId,
    pub rows: SecondaryRows,
    /// Everything the rows depend on; add it to the session's binding guards.
    pub dependencies: Vec<KvDependency>,
    pub pin: SecondaryPin,
}

/// One refused candidate, for a request segment no run could serve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SecondaryRefusal {
    /// Index of the request segment.
    pub segment: usize,
    pub run: SecondaryRunId,
    pub reason: SecondaryRefusalReason,
}

/// The result of [`SecondaryIndex::lookup_positional`].
#[derive(Clone, Debug, Default)]
pub struct SecondaryLookupOutcome {
    /// Admitted spans in ascending request order.
    pub splices: Vec<SecondarySplice>,
    /// Refused candidates of segments with no splice, ordered by segment then run id.
    pub refusals: Vec<SecondaryRefusal>,
    /// Segments whose only admitted run would have covered the last prompt token.
    pub last_token_skips: usize,
}

impl SecondaryLookupOutcome {
    pub fn reused_tokens(&self) -> usize {
        self.splices.iter().map(|splice| splice.len).sum()
    }

    pub fn refusals_for(&self, reason: SecondaryRefusalReason) -> usize {
        self.refusals
            .iter()
            .filter(|refusal| refusal.reason == reason)
            .count()
    }
}

// ---------------------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------------------

type Multiset<K> = BTreeMap<K, u32>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Namespace([u8; 32]);

impl Namespace {
    fn derive(salt: &KvSalt, chain: &KvChainId, mode: KvReuseMode) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(NAMESPACE_DOMAIN);
        hasher.update(salt.as_bytes());
        hasher.update(chain.0);
        hasher.update([mode_tag(mode)]);
        Self(hasher.finalize().into())
    }

    /// The rolling base, derived from the namespace hash.
    fn base(&self) -> u64 {
        let mut head = [0u8; 8];
        head.copy_from_slice(&self.0[..8]);
        u64::from_le_bytes(head) % (FINGERPRINT_MODULUS - MIN_BASE) + MIN_BASE
    }
}

fn mode_tag(mode: KvReuseMode) -> u8 {
    match mode {
        KvReuseMode::Exact => 0,
        KvReuseMode::Approximate => 1,
    }
}

fn kind_tag(kind: SegmentKind) -> u8 {
    match kind {
        SegmentKind::Raw => 0,
        SegmentKind::Semantic => 1,
        SegmentKind::Memory => 2,
        SegmentKind::Derived => 3,
    }
}

fn kind_from_tag(tag: u8) -> Option<SegmentKind> {
    match tag {
        0 => Some(SegmentKind::Raw),
        1 => Some(SegmentKind::Semantic),
        2 => Some(SegmentKind::Memory),
        3 => Some(SegmentKind::Derived),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum PostingKey {
    Rolling(u64),
    Short([u8; 32]),
}

type RunKey = [u8; 32];

fn put_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn put_text(hasher: &mut Sha256, text: &str) {
    put_u64(hasher, text.len() as u64);
    hasher.update(text.as_bytes());
}

fn absorb_lineage(hasher: &mut Sha256, lineage: &SecondaryLineage) {
    hasher.update([kind_tag(lineage.kind)]);
    put_text(hasher, &lineage.identity);
    put_u64(hasher, lineage.generation.0);
    hasher.update(lineage.content_digest);
}

fn dependency_code(kind: KvDependencyKind) -> u8 {
    match kind {
        KvDependencyKind::Expert => 1,
        KvDependencyKind::BaseModel => 2,
        KvDependencyKind::Adapter => 3,
        KvDependencyKind::Segment(segment) => 0x10 + kind_tag(segment),
    }
}

fn dependency_from_code(code: u8) -> Option<KvDependencyKind> {
    match code {
        1 => Some(KvDependencyKind::Expert),
        2 => Some(KvDependencyKind::BaseModel),
        3 => Some(KvDependencyKind::Adapter),
        0x10..=0x13 => kind_from_tag(code - 0x10).map(KvDependencyKind::Segment),
        _ => None,
    }
}

fn absorb_dependency(hasher: &mut Sha256, dependency: &KvDependency) {
    hasher.update([dependency_code(dependency.kind)]);
    put_text(hasher, &dependency.identity);
    put_u64(hasher, dependency.generation.0);
    hasher.update(dependency.content_digest);
    put_u64(hasher, dependency.fence_epoch);
}

/// Everything a run is built from.
struct RunParts {
    namespace: Namespace,
    mode: KvReuseMode,
    tokens: Vec<TokenId>,
    original_start: usize,
    row_width: usize,
    rows: Option<SecondaryRows>,
    rows_digest: Digest,
    own: SecondaryLineage,
    upstream: Multiset<SecondaryLineage>,
    raw: Multiset<Digest>,
    extras: BTreeSet<KvDependency>,
}

struct Run {
    id: SecondaryRunId,
    key: RunKey,
    namespace: Namespace,
    mode: KvReuseMode,
    tokens: Vec<TokenId>,
    original_start: usize,
    row_width: usize,
    rows: Option<SecondaryRows>,
    rows_digest: Digest,
    own: SecondaryLineage,
    upstream: Multiset<SecondaryLineage>,
    raw: Multiset<Digest>,
    extras: BTreeSet<KvDependency>,
    /// Upstream bound segments and the run's own bound lineage (reason 3).
    lineage_dependencies: Vec<KvDependency>,
    /// Lineage dependencies and extras together.
    dependencies: Vec<KvDependency>,
    windows: Vec<(PostingKey, u32)>,
    metadata_bytes: u64,
    pins: BTreeSet<u64>,
    tombstoned: bool,
    last_used: u64,
}

impl Run {
    fn charge(&self) -> Charge {
        Charge {
            runs: 1,
            row_bytes: self.rows.as_ref().map_or(0, SecondaryRows::bytes),
            windows: self.windows.len() as u64,
            metadata: self.metadata_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Charge {
    runs: u64,
    row_bytes: u64,
    windows: u64,
    metadata: u64,
}

impl Charge {
    fn add(self, other: Self) -> Self {
        Self {
            runs: self.runs + other.runs,
            row_bytes: self.row_bytes + other.row_bytes,
            windows: self.windows + other.windows,
            metadata: self.metadata + other.metadata,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            runs: self.runs.saturating_sub(other.runs),
            row_bytes: self.row_bytes.saturating_sub(other.row_bytes),
            windows: self.windows.saturating_sub(other.windows),
            metadata: self.metadata.saturating_sub(other.metadata),
        }
    }

    fn first_exceeded(self, limits: &SecondaryLimits) -> Option<SecondaryResource> {
        if self.runs > limits.max_runs as u64 {
            Some(SecondaryResource::Runs)
        } else if self.row_bytes > limits.max_row_bytes {
            Some(SecondaryResource::RowBytes)
        } else if self.windows > limits.max_indexed_windows {
            Some(SecondaryResource::IndexedWindows)
        } else if self.metadata > limits.max_metadata_bytes {
            Some(SecondaryResource::MetadataBytes)
        } else {
            None
        }
    }
}

#[derive(Default)]
struct Counters {
    inserts: u64,
    duplicate_inserts: u64,
    rows_attached: u64,
    evictions: u64,
    capacity_refusals: u64,
    lookups: u64,
    windows_probed: u64,
    candidates_examined: u64,
    fingerprint_collisions: u64,
    hits: u64,
    reused_tokens: u64,
    last_token_skips: u64,
    tombstoned_skips: u64,
    rows_missing_misses: u64,
    suppressed_refusals: u64,
    tombstoned_at_lookup: u64,
    invalidated: u64,
    refusals: SecondaryRefusalCounts,
}

/// The request seen as multisets, computed once per lookup.
struct RequestView {
    starts: Vec<usize>,
    lineages: Vec<SecondaryLineage>,
    bound: Multiset<SecondaryLineage>,
    raw: Multiset<Digest>,
    total_tokens: usize,
}

impl RequestView {
    fn new(request: &SegmentedModelRequest) -> Self {
        let mut starts = Vec::with_capacity(request.segments().len());
        let mut lineages = Vec::with_capacity(request.segments().len());
        let mut bound: Multiset<SecondaryLineage> = BTreeMap::new();
        let mut raw: Multiset<Digest> = BTreeMap::new();
        let mut cursor = 0usize;
        for segment in request.segments() {
            starts.push(cursor);
            cursor += segment.tokens().len();
            let lineage = SecondaryLineage::from_contribution(&segment.lineage_contribution());
            match lineage.kind {
                SegmentKind::Raw => *raw.entry(lineage.content_digest).or_insert(0) += 1,
                _ => *bound.entry(lineage.clone()).or_insert(0) += 1,
            }
            lineages.push(lineage);
        }
        Self {
            starts,
            lineages,
            bound,
            raw,
            total_tokens: cursor,
        }
    }
}

/// How one candidate fared.
enum Verdict {
    Admit,
    Refuse(SecondaryRefusalReason),
    /// Exact token comparison failed although the lineage matched (never expected for a
    /// real digest; counted as a collision).
    Unverified,
}

/// The position-independent secondary KV index.
pub struct SecondaryIndex {
    window: usize,
    limits: SecondaryLimits,
    runs: BTreeMap<SecondaryRunId, Run>,
    by_key: BTreeMap<RunKey, SecondaryRunId>,
    postings: BTreeMap<(Namespace, PostingKey), Vec<(SecondaryRunId, u32)>>,
    by_identity: BTreeMap<String, BTreeSet<SecondaryRunId>>,
    next_run: u64,
    next_pin: u64,
    clock: u64,
    totals: Charge,
    counters: Counters,
}

impl fmt::Debug for SecondaryIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecondaryIndex")
            .field("window", &self.window)
            .field("runs", &self.runs.len())
            .finish_non_exhaustive()
    }
}

impl SecondaryIndex {
    /// An empty index. Fails with [`SecondaryError::InvalidConfig`] for a zero limit, a
    /// zero window or a window longer than a run may be.
    pub fn new(config: SecondaryConfig) -> Result<Self, SecondaryError> {
        let limits = config.limits;
        if config.window_tokens == 0
            || config.window_tokens > limits.max_tokens_per_run
            || limits.max_runs == 0
            || limits.max_row_bytes == 0
            || limits.max_indexed_windows == 0
            || limits.max_tokens_per_run == 0
            || limits.max_metadata_bytes == 0
            || limits.max_candidates_per_window == 0
        {
            return Err(SecondaryError::InvalidConfig);
        }
        Ok(Self {
            window: config.window_tokens,
            limits,
            runs: BTreeMap::new(),
            by_key: BTreeMap::new(),
            postings: BTreeMap::new(),
            by_identity: BTreeMap::new(),
            next_run: 0,
            next_pin: 0,
            clock: 0,
            totals: Charge::default(),
            counters: Counters::default(),
        })
    }

    pub fn window_tokens(&self) -> usize {
        self.window
    }

    pub fn limits(&self) -> SecondaryLimits {
        self.limits
    }

    /// Runs held, tombstoned-but-pinned ones included.
    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Whether the run is present and not tombstoned.
    pub fn contains_live_run(&self, run: SecondaryRunId) -> bool {
        self.runs.get(&run).is_some_and(|run| !run.tombstoned)
    }

    pub fn stats(&self) -> SecondaryStats {
        let counters = &self.counters;
        SecondaryStats {
            runs: self.runs.len() as u64,
            tombstoned_pinned_runs: self.runs.values().filter(|run| run.tombstoned).count() as u64,
            row_bytes: self.totals.row_bytes,
            indexed_windows: self.totals.windows,
            metadata_bytes: self.totals.metadata,
            active_pins: self.runs.values().map(|run| run.pins.len() as u64).sum(),
            inserts: counters.inserts,
            duplicate_inserts: counters.duplicate_inserts,
            rows_attached: counters.rows_attached,
            evictions: counters.evictions,
            capacity_refusals: counters.capacity_refusals,
            lookups: counters.lookups,
            windows_probed: counters.windows_probed,
            candidates_examined: counters.candidates_examined,
            fingerprint_collisions: counters.fingerprint_collisions,
            hits: counters.hits,
            reused_tokens: counters.reused_tokens,
            last_token_skips: counters.last_token_skips,
            tombstoned_skips: counters.tombstoned_skips,
            rows_missing_misses: counters.rows_missing_misses,
            suppressed_refusals: counters.suppressed_refusals,
            tombstoned_at_lookup: counters.tombstoned_at_lookup,
            invalidated: counters.invalidated,
            refusals: counters.refusals,
        }
    }

    // ----- insert ----------------------------------------------------------------------

    /// Offer one finished segment. Idempotent: the same content under the same lineage is
    /// one run. See [`SecondaryInsertOutcome`].
    pub fn insert(
        &mut self,
        insert: SecondaryInsert<'_>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<SecondaryInsertOutcome, SecondaryError> {
        if insert.mode != KvReuseMode::Exact {
            return Err(SecondaryError::NotExact);
        }
        let segments = insert.request.segments();
        let segment = segments
            .get(insert.segment)
            .ok_or(SecondaryError::SegmentOutOfRange)?;
        let tokens = segment.tokens();
        if tokens.len() > self.limits.max_tokens_per_run {
            self.counters.capacity_refusals += 1;
            return Err(SecondaryError::CapacityExhausted(
                SecondaryResource::TokensPerRun,
            ));
        }
        let rows = insert.rows;
        if rows.row_width == 0
            || rows.keys.len() != tokens.len() * rows.row_width
            || rows.values.len() != rows.keys.len()
        {
            return Err(SecondaryError::RowShape);
        }
        let mut original_start = 0usize;
        let mut upstream: Multiset<SecondaryLineage> = BTreeMap::new();
        let mut raw: Multiset<Digest> = BTreeMap::new();
        for earlier in &segments[..insert.segment] {
            original_start += earlier.tokens().len();
            let lineage = SecondaryLineage::from_contribution(&earlier.lineage_contribution());
            match lineage.kind {
                SegmentKind::Raw => *raw.entry(lineage.content_digest).or_insert(0) += 1,
                _ => *upstream.entry(lineage).or_insert(0) += 1,
            }
        }
        let own = SecondaryLineage::from_contribution(&segment.lineage_contribution());
        let namespace = Namespace::derive(insert.salt, insert.chain, insert.mode);
        let rows_digest = rows.digest();
        let parts = RunParts {
            namespace,
            mode: insert.mode,
            tokens: tokens.to_vec(),
            original_start,
            row_width: rows.row_width,
            rows: Some(rows),
            rows_digest,
            own,
            upstream,
            raw,
            extras: insert.extra_dependencies.into_iter().collect(),
        };
        let run = self.build_run(parts);
        if !run
            .dependencies
            .iter()
            .all(|dependency| liveness.is_live(dependency))
        {
            return Err(SecondaryError::DependencyNotLive);
        }
        self.admit_run(run)
    }

    fn admit_run(&mut self, run: Run) -> Result<SecondaryInsertOutcome, SecondaryError> {
        if let Some(existing_id) = self.by_key.get(&run.key).copied() {
            return self.merge_into_existing(existing_id, run);
        }
        // Per-window candidate bound, judged against the current state.
        for (key, _) in &run.windows {
            let held = self
                .postings
                .get(&(run.namespace, *key))
                .map_or(0, Vec::len);
            if held >= self.limits.max_candidates_per_window {
                self.counters.capacity_refusals += 1;
                return Err(SecondaryError::CapacityExhausted(
                    SecondaryResource::CandidatesPerWindow,
                ));
            }
        }
        self.make_room(run.charge(), None)?;
        self.counters.inserts += 1;
        self.add_run(run);
        let id = SecondaryRunId(self.next_run - 1);
        self.touch(id);
        Ok(SecondaryInsertOutcome::Inserted(id))
    }

    fn merge_into_existing(
        &mut self,
        existing_id: SecondaryRunId,
        incoming: Run,
    ) -> Result<SecondaryInsertOutcome, SecondaryError> {
        let (tombstoned, tokens_equal, has_rows, digest_equal) = {
            let existing = &self.runs[&existing_id];
            (
                existing.tombstoned,
                existing.tokens == incoming.tokens,
                existing.rows.is_some(),
                existing.rows_digest == incoming.rows_digest,
            )
        };
        if tombstoned {
            return Err(SecondaryError::SlotDraining);
        }
        if !tokens_equal {
            return Err(SecondaryError::TokensConflict);
        }
        if !digest_equal {
            return Err(SecondaryError::RowsDigestMismatch);
        }
        if let Some(existing) = self.runs.get_mut(&existing_id) {
            existing.original_start = existing.original_start.min(incoming.original_start);
        }
        if has_rows {
            self.counters.duplicate_inserts += 1;
            self.touch(existing_id);
            return Ok(SecondaryInsertOutcome::AlreadyPresent(existing_id));
        }
        let rows = incoming.rows;
        let Some(rows) = rows else {
            return Err(SecondaryError::RowShape);
        };
        let need = Charge {
            row_bytes: rows.bytes(),
            ..Charge::default()
        };
        self.make_room(need, Some(existing_id))?;
        if let Some(existing) = self.runs.get_mut(&existing_id) {
            existing.rows = Some(rows);
            existing.row_width = incoming.row_width;
        }
        self.totals = self.totals.add(need);
        self.counters.rows_attached += 1;
        self.touch(existing_id);
        Ok(SecondaryInsertOutcome::RowsAttached(existing_id))
    }

    fn build_run(&self, parts: RunParts) -> Run {
        let mut lineage_dependencies: Vec<KvDependency> = parts
            .upstream
            .keys()
            .chain(std::iter::once(&parts.own))
            .filter_map(SecondaryLineage::dependency)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        lineage_dependencies.sort();
        let dependencies: Vec<KvDependency> = lineage_dependencies
            .iter()
            .cloned()
            .chain(parts.extras.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let key = run_key(&parts);
        let windows = self.posting_keys(&parts.namespace, &parts.tokens);
        let metadata_bytes = RUN_METADATA_BYTES
            + parts.tokens.len() as u64 * TOKEN_METADATA_BYTES
            + dependencies
                .iter()
                .map(|dependency| DEPENDENCY_METADATA_BYTES + dependency.identity.len() as u64)
                .sum::<u64>()
            + parts
                .upstream
                .keys()
                .map(SecondaryLineage::metadata_bytes)
                .sum::<u64>()
            + parts.raw.len() as u64 * DEPENDENCY_METADATA_BYTES;
        Run {
            id: SecondaryRunId(0),
            key,
            namespace: parts.namespace,
            mode: parts.mode,
            tokens: parts.tokens,
            original_start: parts.original_start,
            row_width: parts.row_width,
            rows: parts.rows,
            rows_digest: parts.rows_digest,
            own: parts.own,
            upstream: parts.upstream,
            raw: parts.raw,
            extras: parts.extras,
            lineage_dependencies,
            dependencies,
            windows,
            metadata_bytes,
            pins: BTreeSet::new(),
            tombstoned: false,
            last_used: 0,
        }
    }

    /// The unique posting keys of a run with the first offset of each.
    fn posting_keys(&self, namespace: &Namespace, tokens: &[TokenId]) -> Vec<(PostingKey, u32)> {
        if tokens.len() < self.window {
            return vec![(PostingKey::Short(short_key(tokens)), 0)];
        }
        let mut unique: BTreeMap<PostingKey, u32> = BTreeMap::new();
        for (offset, print) in window_fingerprints(tokens, self.window, namespace.base())
            .into_iter()
            .enumerate()
        {
            unique
                .entry(PostingKey::Rolling(print))
                .or_insert(offset as u32);
        }
        unique.into_iter().collect()
    }

    fn add_run(&mut self, mut run: Run) {
        let id = SecondaryRunId(self.next_run);
        self.next_run += 1;
        run.id = id;
        self.totals = self.totals.add(run.charge());
        self.by_key.insert(run.key, id);
        for (key, offset) in &run.windows {
            self.postings
                .entry((run.namespace, *key))
                .or_default()
                .push((id, *offset));
        }
        for dependency in &run.dependencies {
            self.by_identity
                .entry(dependency.identity.clone())
                .or_default()
                .insert(id);
        }
        self.runs.insert(id, run);
    }

    fn remove_run(&mut self, id: SecondaryRunId) {
        let Some(run) = self.runs.remove(&id) else {
            return;
        };
        self.totals = self.totals.sub(run.charge());
        self.by_key.remove(&run.key);
        for (key, _) in &run.windows {
            let slot = (run.namespace, *key);
            if let Some(list) = self.postings.get_mut(&slot) {
                list.retain(|(held, _)| *held != id);
                if list.is_empty() {
                    self.postings.remove(&slot);
                }
            }
        }
        for dependency in &run.dependencies {
            if let Some(set) = self.by_identity.get_mut(&dependency.identity) {
                set.remove(&id);
                if set.is_empty() {
                    self.by_identity.remove(&dependency.identity);
                }
            }
        }
    }

    fn touch(&mut self, id: SecondaryRunId) {
        self.clock += 1;
        let clock = self.clock;
        if let Some(run) = self.runs.get_mut(&id) {
            run.last_used = clock;
        }
    }

    /// Make `need` fit by evicting unpinned runs in LRU order. Nothing is evicted when it
    /// cannot fit even then.
    fn make_room(
        &mut self,
        need: Charge,
        protect: Option<SecondaryRunId>,
    ) -> Result<(), SecondaryError> {
        let limits = self.limits;
        let mut total = self.totals.add(need);
        if total.first_exceeded(&limits).is_none() {
            return Ok(());
        }
        let mut order: Vec<(u64, SecondaryRunId)> = self
            .runs
            .values()
            .filter(|run| run.pins.is_empty() && Some(run.id) != protect)
            .map(|run| (run.last_used, run.id))
            .collect();
        order.sort();
        let mut victims = Vec::new();
        for (_, id) in order {
            total = total.sub(self.runs[&id].charge());
            victims.push(id);
            if total.first_exceeded(&limits).is_none() {
                break;
            }
        }
        if let Some(resource) = total.first_exceeded(&limits) {
            self.counters.capacity_refusals += 1;
            return Err(SecondaryError::CapacityExhausted(resource));
        }
        for victim in victims {
            self.remove_run(victim);
            self.counters.evictions += 1;
        }
        Ok(())
    }

    // ----- lookup ----------------------------------------------------------------------

    /// Find, for every request segment, a stored run that serves it at its new position.
    ///
    /// Requires canonical positions. Admitted runs are pinned; release every
    /// [`SecondarySplice::pin`] when the consumer is done. A run found dead is tombstoned.
    pub fn lookup_positional(
        &mut self,
        lookup: SecondaryLookup<'_>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<SecondaryLookupOutcome, SecondaryError> {
        if !matches!(lookup.positions, KvRequestPositions::Canonical) {
            return Err(SecondaryError::NonCanonicalPosition);
        }
        self.counters.lookups += 1;
        let namespace = Namespace::derive(lookup.salt, lookup.chain, lookup.mode);
        let view = RequestView::new(lookup.request);
        let mut outcome = SecondaryLookupOutcome::default();
        for (index, segment) in lookup.request.segments().iter().enumerate() {
            let candidates = self.candidates(&namespace, segment.tokens());
            let mut admitted: Vec<SecondaryRunId> = Vec::new();
            let mut refused: Vec<(SecondaryRunId, SecondaryRefusalReason)> = Vec::new();
            let mut dead: Vec<SecondaryRunId> = Vec::new();
            for id in candidates {
                let run = &self.runs[&id];
                self.counters.candidates_examined += 1;
                if run.tombstoned {
                    self.counters.tombstoned_skips += 1;
                    continue;
                }
                if run.rows.is_none() {
                    self.counters.rows_missing_misses += 1;
                    continue;
                }
                match judge(run, &view, index, segment.tokens(), liveness) {
                    Verdict::Admit => admitted.push(id),
                    Verdict::Refuse(reason) => {
                        if matches!(
                            reason,
                            SecondaryRefusalReason::OriginalUpstreamDead
                                | SecondaryRefusalReason::BindingDependencyDead
                        ) {
                            dead.push(id);
                        }
                        refused.push((id, reason));
                    }
                    Verdict::Unverified => self.counters.fingerprint_collisions += 1,
                }
            }
            for id in dead {
                self.counters.tombstoned_at_lookup += 1;
                self.tombstone_run(id);
            }
            let len = segment.tokens().len();
            let request_start = view.starts[index];
            let best = admitted
                .iter()
                .copied()
                .min_by_key(|id| (std::cmp::Reverse(len), self.runs[id].original_start, *id));
            let Some(best) = best else {
                for (run, reason) in refused {
                    self.counters.refusals.bump(reason);
                    outcome.refusals.push(SecondaryRefusal {
                        segment: index,
                        run,
                        reason,
                    });
                }
                continue;
            };
            self.counters.suppressed_refusals += refused.len() as u64;
            if request_start + len >= view.total_tokens {
                self.counters.last_token_skips += 1;
                outcome.last_token_skips += 1;
                continue;
            }
            outcome
                .splices
                .push(self.take_splice(best, request_start, len));
        }
        Ok(outcome)
    }

    fn take_splice(
        &mut self,
        id: SecondaryRunId,
        request_start: usize,
        len: usize,
    ) -> SecondarySplice {
        self.next_pin += 1;
        let pin_id = self.next_pin;
        self.touch(id);
        self.counters.hits += 1;
        self.counters.reused_tokens += len as u64;
        let run = self.runs.get_mut(&id).expect("admitted run is present");
        run.pins.insert(pin_id);
        let rows = run.rows.clone().expect("admitted run has rows");
        SecondarySplice {
            request_start,
            len,
            delta: request_start as i64 - run.original_start as i64,
            original_start: run.original_start,
            run: id,
            rows,
            dependencies: run.dependencies.clone(),
            pin: SecondaryPin {
                run: id,
                id: pin_id,
            },
        }
    }

    /// Runs that share at least one exactly verified window (or the exact short key) with
    /// `tokens`.
    fn candidates(
        &mut self,
        namespace: &Namespace,
        tokens: &[TokenId],
    ) -> BTreeSet<SecondaryRunId> {
        let mut found = BTreeSet::new();
        let mut collisions = 0u64;
        let mut probes = 0u64;
        if tokens.len() < self.window {
            probes += 1;
            let key = (*namespace, PostingKey::Short(short_key(tokens)));
            for (id, _) in self.postings.get(&key).into_iter().flatten() {
                if self.runs[id].tokens == tokens {
                    found.insert(*id);
                } else {
                    collisions += 1;
                }
            }
        } else {
            let prints = window_fingerprints(tokens, self.window, namespace.base());
            for (at, print) in prints.into_iter().enumerate() {
                probes += 1;
                let key = (*namespace, PostingKey::Rolling(print));
                for (id, offset) in self.postings.get(&key).into_iter().flatten() {
                    let run = &self.runs[id];
                    let offset = *offset as usize;
                    let verified = run.tokens.get(offset..offset + self.window)
                        == Some(&tokens[at..at + self.window]);
                    if verified {
                        found.insert(*id);
                    } else {
                        collisions += 1;
                    }
                }
            }
        }
        self.counters.windows_probed += probes;
        self.counters.fingerprint_collisions += collisions;
        found
    }

    // ----- pins ------------------------------------------------------------------------

    /// Whether the pinned run may still be consumed.
    pub fn pin_state(&self, pin: &SecondaryPin) -> SecondaryPinState {
        match self.runs.get(&pin.run) {
            Some(run) if !run.tombstoned && run.pins.contains(&pin.id) => SecondaryPinState::Live,
            _ => SecondaryPinState::Revoked,
        }
    }

    /// Re-check every dependency of the pinned run now; a dead one tombstones the run.
    pub fn revalidate(
        &mut self,
        pin: &SecondaryPin,
        liveness: &dyn KvLivenessOracle,
    ) -> SecondaryPinState {
        if self.pin_state(pin) != SecondaryPinState::Live {
            return SecondaryPinState::Revoked;
        }
        let alive = self.runs[&pin.run]
            .dependencies
            .iter()
            .all(|dependency| liveness.is_live(dependency));
        if alive {
            return SecondaryPinState::Live;
        }
        self.counters.tombstoned_at_lookup += 1;
        self.tombstone_run(pin.run);
        SecondaryPinState::Revoked
    }

    /// Release a pin. A tombstoned run is reclaimed with its last pin. Returns whether the
    /// pin was held.
    pub fn release(&mut self, pin: SecondaryPin) -> bool {
        let Some(run) = self.runs.get_mut(&pin.run) else {
            return false;
        };
        if !run.pins.remove(&pin.id) {
            return false;
        }
        if run.tombstoned && run.pins.is_empty() {
            self.remove_run(pin.run);
        }
        true
    }

    // ----- invalidation ----------------------------------------------------------------

    fn tombstone_run(&mut self, id: SecondaryRunId) {
        let Some(run) = self.runs.get_mut(&id) else {
            return;
        };
        if run.tombstoned {
            return;
        }
        run.tombstoned = true;
        if run.pins.is_empty() {
            self.remove_run(id);
        }
    }

    /// Tombstone every run with a dependency on one of `identities` for which `affected`
    /// holds. Mirrors the primary catalog's `tombstone_matching`, so every `invalidate_*`
    /// of the catalog can funnel here. Returns the number of runs tombstoned.
    pub fn invalidate_matching(
        &mut self,
        identities: &BTreeSet<&str>,
        mut affected: impl FnMut(&KvDependency) -> bool,
    ) -> usize {
        let mut targets: BTreeSet<SecondaryRunId> = BTreeSet::new();
        for identity in identities {
            for id in self.by_identity.get(*identity).into_iter().flatten() {
                let hit = self.runs.get(id).is_some_and(|run| {
                    !run.tombstoned
                        && run.dependencies.iter().any(|dependency| {
                            dependency.identity == *identity && affected(dependency)
                        })
                });
                if hit {
                    targets.insert(*id);
                }
            }
        }
        let count = targets.len();
        for id in targets {
            self.tombstone_run(id);
        }
        self.counters.invalidated += count as u64;
        count
    }

    /// Tombstone every run that depends on one of the `(identity, generation)` pairs.
    pub fn invalidate_inputs(&mut self, invalidated: &BTreeSet<(String, Generation)>) -> usize {
        let identities: BTreeSet<&str> = invalidated.iter().map(|(id, _)| id.as_str()).collect();
        self.invalidate_matching(&identities, |dependency| {
            invalidated.contains(&(dependency.identity.clone(), dependency.generation))
        })
    }

    /// Tombstone every run that depends on any generation of the named identities.
    pub fn invalidate_identities(&mut self, identities: &BTreeSet<String>) -> usize {
        let wanted: BTreeSet<&str> = identities.iter().map(String::as_str).collect();
        self.invalidate_matching(&wanted, |_| true)
    }

    /// Like [`Self::invalidate_identities`] restricted to dependencies of `kind`.
    pub fn invalidate_kind_identities(
        &mut self,
        kind: KvDependencyKind,
        identities: &BTreeSet<String>,
    ) -> usize {
        let wanted: BTreeSet<&str> = identities.iter().map(String::as_str).collect();
        self.invalidate_matching(&wanted, |dependency| dependency.kind == kind)
    }

    /// Tombstone every run with an expert dependency (the range of `expert:` identities).
    pub fn invalidate_all_experts(&mut self) -> usize {
        let identities: Vec<String> = self
            .by_identity
            .range("expert:".to_owned()..)
            .map(|(identity, _)| identity.clone())
            .take_while(|identity| identity.starts_with("expert:"))
            .collect();
        let wanted: BTreeSet<&str> = identities.iter().map(String::as_str).collect();
        self.invalidate_matching(&wanted, |dependency| {
            dependency.kind == KvDependencyKind::Expert
        })
    }

    /// Tombstone every dependency below the generation now live for its identity.
    pub fn invalidate_superseded(&mut self, live: &BTreeMap<String, Generation>) -> usize {
        let wanted: BTreeSet<&str> = live.keys().map(String::as_str).collect();
        self.invalidate_matching(&wanted, |dependency| {
            live.get(&dependency.identity)
                .is_some_and(|generation| dependency.generation < *generation)
        })
    }

    /// Like [`Self::invalidate_superseded`] restricted to dependencies of `kind`.
    pub fn invalidate_kind_superseded(
        &mut self,
        kind: KvDependencyKind,
        live: &BTreeMap<String, Generation>,
    ) -> usize {
        let wanted: BTreeSet<&str> = live.keys().map(String::as_str).collect();
        self.invalidate_matching(&wanted, |dependency| {
            dependency.kind == kind
                && live
                    .get(&dependency.identity)
                    .is_some_and(|generation| dependency.generation < *generation)
        })
    }

    /// Tombstone every expert dependency whose recorded fence epoch is below the epoch now
    /// in force (a run that recorded none, `0`, counts as below).
    pub fn invalidate_expert_fences(&mut self, fences: &BTreeMap<String, u64>) -> usize {
        let wanted: BTreeSet<&str> = fences.keys().map(String::as_str).collect();
        self.invalidate_matching(&wanted, |dependency| {
            dependency.kind == KvDependencyKind::Expert
                && fences
                    .get(&dependency.identity)
                    .is_some_and(|epoch| dependency.fence_epoch < *epoch)
        })
    }

    // ----- snapshot --------------------------------------------------------------------

    /// Canonical bytes of every live run's metadata and `rows_digest`, never rows.
    /// `None` when there is no live run, so a caller can leave its own snapshot unchanged.
    pub fn export_section(&self) -> Option<Vec<u8>> {
        let mut live: Vec<&Run> = self.runs.values().filter(|run| !run.tombstoned).collect();
        if live.is_empty() {
            return None;
        }
        live.sort_by_key(|run| run.key);
        let mut out = Vec::new();
        out.extend_from_slice(SECTION_MAGIC);
        out.extend_from_slice(&(live.len() as u64).to_le_bytes());
        for run in live {
            encode_run(&mut out, run);
        }
        Some(out)
    }

    /// Admit the runs of an exported section, without rows and without a liveness check;
    /// the use-time check decides at the first use. Atomic: on any error the index is
    /// unchanged. Runs already present are left as they are. Nothing is evicted to make
    /// room. Returns the number of runs added.
    pub fn restore_section(&mut self, bytes: &[u8]) -> Result<usize, SecondaryError> {
        let decoded = self.decode_section(bytes)?;
        let fresh: Vec<Run> = decoded
            .into_iter()
            .filter(|run| !self.by_key.contains_key(&run.key))
            .collect();
        let mut need = Charge::default();
        let mut per_window: BTreeMap<(Namespace, PostingKey), usize> = BTreeMap::new();
        for run in &fresh {
            need = need.add(run.charge());
            for (key, _) in &run.windows {
                *per_window.entry((run.namespace, *key)).or_insert(0) += 1;
            }
        }
        for (slot, added) in &per_window {
            let held = self.postings.get(slot).map_or(0, Vec::len);
            if held + added > self.limits.max_candidates_per_window {
                self.counters.capacity_refusals += 1;
                return Err(SecondaryError::CapacityExhausted(
                    SecondaryResource::CandidatesPerWindow,
                ));
            }
        }
        if let Some(resource) = self.totals.add(need).first_exceeded(&self.limits) {
            self.counters.capacity_refusals += 1;
            return Err(SecondaryError::CapacityExhausted(resource));
        }
        let added = fresh.len();
        for run in fresh {
            self.add_run(run);
        }
        Ok(added)
    }

    fn decode_section(&self, bytes: &[u8]) -> Result<Vec<Run>, SecondarySectionError> {
        if bytes.len() > SECONDARY_SECTION_MAX_BYTES {
            return Err(SecondarySectionError::TooLarge);
        }
        let mut reader = Reader { bytes, at: 0 };
        if reader.take(SECTION_MAGIC.len())? != SECTION_MAGIC {
            return Err(SecondarySectionError::BadMagic);
        }
        let count = reader.count("run count", self.limits.max_runs)?;
        let mut runs: Vec<Run> = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let parts = self.decode_run(&mut reader)?;
            let run = self.build_run(parts);
            if runs.last().is_some_and(|previous| previous.key >= run.key) {
                return Err(SecondarySectionError::NotCanonical("run order"));
            }
            runs.push(run);
        }
        if reader.at != bytes.len() {
            return Err(SecondarySectionError::TrailingBytes);
        }
        Ok(runs)
    }

    fn decode_run(&self, reader: &mut Reader<'_>) -> Result<RunParts, SecondarySectionError> {
        let mut namespace = [0u8; 32];
        namespace.copy_from_slice(reader.take(32)?);
        if reader.u8()? != mode_tag(KvReuseMode::Exact) {
            return Err(SecondarySectionError::InvalidValue("mode"));
        }
        let original_start = reader.index("original start")?;
        let row_width = reader.index("row width")?;
        if row_width == 0 {
            return Err(SecondarySectionError::InvalidValue("row width"));
        }
        let token_count = reader.count("token count", self.limits.max_tokens_per_run)?;
        if token_count == 0 {
            return Err(SecondarySectionError::InvalidValue("token count"));
        }
        let mut tokens = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            tokens.push(TokenId(reader.u32()?));
        }
        let mut rows_digest = [0u8; 32];
        rows_digest.copy_from_slice(reader.take(32)?);
        let own = reader.lineage()?;
        let upstream = reader.multiset(|reader| {
            let lineage = reader.lineage()?;
            if lineage.kind == SegmentKind::Raw {
                return Err(SecondarySectionError::InvalidValue("upstream kind"));
            }
            Ok(lineage)
        })?;
        let raw = reader.multiset(|reader| {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(reader.take(32)?);
            Ok(digest)
        })?;
        let extra_count = reader.count("extra count", SECTION_MAX_ENTRIES)?;
        let mut extras = BTreeSet::new();
        let mut previous: Option<KvDependency> = None;
        for _ in 0..extra_count {
            let dependency = reader.dependency()?;
            if previous
                .as_ref()
                .is_some_and(|before| *before >= dependency)
            {
                return Err(SecondarySectionError::NotCanonical("extra order"));
            }
            previous = Some(dependency.clone());
            extras.insert(dependency);
        }
        Ok(RunParts {
            namespace: Namespace(namespace),
            mode: KvReuseMode::Exact,
            tokens,
            original_start,
            row_width,
            rows: None,
            rows_digest,
            own,
            upstream,
            raw,
            extras,
        })
    }
}

fn run_key(parts: &RunParts) -> RunKey {
    let mut hasher = Sha256::new();
    hasher.update(RUN_KEY_DOMAIN);
    hasher.update(parts.namespace.0);
    absorb_lineage(&mut hasher, &parts.own);
    put_u64(&mut hasher, parts.upstream.len() as u64);
    for (lineage, count) in &parts.upstream {
        absorb_lineage(&mut hasher, lineage);
        hasher.update(count.to_le_bytes());
    }
    put_u64(&mut hasher, parts.raw.len() as u64);
    for (digest, count) in &parts.raw {
        hasher.update(digest);
        hasher.update(count.to_le_bytes());
    }
    put_u64(&mut hasher, parts.extras.len() as u64);
    for dependency in &parts.extras {
        absorb_dependency(&mut hasher, dependency);
    }
    hasher.finalize().into()
}

/// Judge one usable candidate against request segment `index`. Steps 2 to 6 of the module
/// documentation, in order.
fn judge(
    run: &Run,
    view: &RequestView,
    index: usize,
    tokens: &[TokenId],
    liveness: &dyn KvLivenessOracle,
) -> Verdict {
    let own = &view.lineages[index];
    if run.own.kind != own.kind {
        return Verdict::Refuse(SecondaryRefusalReason::SourceKindMismatch);
    }
    if run.own.identity != own.identity {
        return Verdict::Refuse(SecondaryRefusalReason::SourceIdentityMismatch);
    }
    if run.own.generation != own.generation {
        return Verdict::Refuse(SecondaryRefusalReason::SourceGenerationMismatch);
    }
    if run.own.content_digest != own.content_digest {
        return Verdict::Refuse(SecondaryRefusalReason::SourceDigestMismatch);
    }
    // Lineage equal: the tokens must be equal too, or the digest is not what it claims.
    if run.tokens != tokens {
        return Verdict::Unverified;
    }
    if !run
        .lineage_dependencies
        .iter()
        .all(|dependency| liveness.is_live(dependency))
    {
        return Verdict::Refuse(SecondaryRefusalReason::OriginalUpstreamDead);
    }
    if !run
        .extras
        .iter()
        .all(|dependency| liveness.is_live(dependency))
    {
        return Verdict::Refuse(SecondaryRefusalReason::BindingDependencyDead);
    }
    let span_is_bound = own.kind != SegmentKind::Raw;
    for (lineage, needed) in &run.upstream {
        let own_occurrence = u32::from(span_is_bound && lineage == own);
        let have = view
            .bound
            .get(lineage)
            .copied()
            .unwrap_or(0)
            .saturating_sub(own_occurrence);
        if have < *needed {
            return Verdict::Refuse(SecondaryRefusalReason::OriginalUpstreamOmitted);
        }
    }
    for (digest, needed) in &run.raw {
        let own_occurrence = u32::from(!span_is_bound && *digest == own.content_digest);
        let have = view
            .raw
            .get(digest)
            .copied()
            .unwrap_or(0)
            .saturating_sub(own_occurrence);
        if have < *needed {
            return Verdict::Refuse(SecondaryRefusalReason::RawAncestorOmitted);
        }
    }
    Verdict::Admit
}

// ---------------------------------------------------------------------------------------
// Canonical encoding
// ---------------------------------------------------------------------------------------

fn put_bytes_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_text_bytes(out: &mut Vec<u8>, text: &str) {
    put_bytes_u64(out, text.len() as u64);
    out.extend_from_slice(text.as_bytes());
}

fn put_lineage(out: &mut Vec<u8>, lineage: &SecondaryLineage) {
    out.push(kind_tag(lineage.kind));
    put_text_bytes(out, &lineage.identity);
    put_bytes_u64(out, lineage.generation.0);
    out.extend_from_slice(&lineage.content_digest);
}

fn encode_run(out: &mut Vec<u8>, run: &Run) {
    out.extend_from_slice(&run.namespace.0);
    out.push(mode_tag(run.mode));
    put_bytes_u64(out, run.original_start as u64);
    put_bytes_u64(out, run.row_width as u64);
    put_bytes_u64(out, run.tokens.len() as u64);
    for token in &run.tokens {
        out.extend_from_slice(&token.0.to_le_bytes());
    }
    out.extend_from_slice(&run.rows_digest);
    put_lineage(out, &run.own);
    put_bytes_u64(out, run.upstream.len() as u64);
    for (lineage, count) in &run.upstream {
        put_lineage(out, lineage);
        out.extend_from_slice(&count.to_le_bytes());
    }
    put_bytes_u64(out, run.raw.len() as u64);
    for (digest, count) in &run.raw {
        out.extend_from_slice(digest);
        out.extend_from_slice(&count.to_le_bytes());
    }
    put_bytes_u64(out, run.extras.len() as u64);
    for dependency in &run.extras {
        out.push(dependency_code(dependency.kind));
        put_text_bytes(out, &dependency.identity);
        put_bytes_u64(out, dependency.generation.0);
        out.extend_from_slice(&dependency.content_digest);
        put_bytes_u64(out, dependency.fence_epoch);
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], SecondarySectionError> {
        let end = self
            .at
            .checked_add(len)
            .ok_or(SecondarySectionError::Truncated)?;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or(SecondarySectionError::Truncated)?;
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, SecondarySectionError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, SecondarySectionError> {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(raw))
    }

    fn u64(&mut self) -> Result<u64, SecondarySectionError> {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(raw))
    }

    /// A `u64` that must fit a `usize` below `u32::MAX`.
    fn index(&mut self, what: &'static str) -> Result<usize, SecondarySectionError> {
        let value = self.u64()?;
        u32::try_from(value)
            .ok()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(SecondarySectionError::OutOfBounds(what))
    }

    /// A count bounded by `max`.
    fn count(&mut self, what: &'static str, max: usize) -> Result<usize, SecondarySectionError> {
        let value = self.u64()?;
        match usize::try_from(value) {
            Ok(value) if value <= max => Ok(value),
            _ => Err(SecondarySectionError::OutOfBounds(what)),
        }
    }

    fn text(&mut self) -> Result<String, SecondarySectionError> {
        let len = self.count("text length", SECTION_MAX_TEXT_BYTES)?;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| SecondarySectionError::InvalidValue("text"))
    }

    fn lineage(&mut self) -> Result<SecondaryLineage, SecondarySectionError> {
        let kind = kind_from_tag(self.u8()?).ok_or(SecondarySectionError::InvalidValue("kind"))?;
        let identity = self.text()?;
        let generation = Generation(self.u64()?);
        let mut content_digest = [0u8; 32];
        content_digest.copy_from_slice(self.take(32)?);
        if kind == SegmentKind::Raw && (!identity.is_empty() || generation != Generation(0)) {
            return Err(SecondarySectionError::InvalidValue("raw lineage"));
        }
        if kind != SegmentKind::Raw && identity.is_empty() {
            return Err(SecondarySectionError::InvalidValue("bound identity"));
        }
        Ok(SecondaryLineage {
            kind,
            identity,
            generation,
            content_digest,
        })
    }

    fn dependency(&mut self) -> Result<KvDependency, SecondarySectionError> {
        let kind = dependency_from_code(self.u8()?)
            .ok_or(SecondarySectionError::InvalidValue("dependency kind"))?;
        let identity = self.text()?;
        let generation = Generation(self.u64()?);
        let mut content_digest = [0u8; 32];
        content_digest.copy_from_slice(self.take(32)?);
        let fence_epoch = self.u64()?;
        if identity.is_empty() {
            return Err(SecondarySectionError::InvalidValue("dependency identity"));
        }
        if kind != KvDependencyKind::Expert && fence_epoch != 0 {
            return Err(SecondarySectionError::InvalidValue("fence epoch"));
        }
        Ok(KvDependency {
            kind,
            identity,
            generation,
            content_digest,
            fence_epoch,
        })
    }

    /// A strictly ascending multiset with counts of at least 1.
    fn multiset<K: Ord + Clone>(
        &mut self,
        mut key: impl FnMut(&mut Self) -> Result<K, SecondarySectionError>,
    ) -> Result<Multiset<K>, SecondarySectionError> {
        let entries = self.count("multiset size", SECTION_MAX_ENTRIES)?;
        let mut out = BTreeMap::new();
        let mut previous: Option<K> = None;
        for _ in 0..entries {
            let item = key(self)?;
            let count = self.u32()?;
            if count == 0 {
                return Err(SecondarySectionError::InvalidValue("multiset count"));
            }
            if previous.as_ref().is_some_and(|before| *before >= item) {
                return Err(SecondarySectionError::NotCanonical("multiset order"));
            }
            previous = Some(item.clone());
            out.insert(item, count);
        }
        Ok(out)
    }
}

#[cfg(test)]
impl SecondaryIndex {
    /// Make `run` a posting of `print` in `namespace`, as a fingerprint collision would.
    fn inject_posting(&mut self, lookup: &SecondaryLookup<'_>, print: u64, run: SecondaryRunId) {
        let namespace = Namespace::derive(lookup.salt, lookup.chain, lookup.mode);
        self.postings
            .entry((namespace, PostingKey::Rolling(print)))
            .or_default()
            .push((run, 0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_model_api::{ByteTokenizer, PromptSegment, SegmentSource, Tokenizer};
    use ptr_types::{EvidenceId, KnowledgeObjectId, RequestId, Revision, SemanticContext};

    const ROW: usize = 6;

    // ----- fixtures --------------------------------------------------------------------

    struct AllLive;

    impl KvLivenessOracle for AllLive {
        fn is_live(&self, _: &KvDependency) -> bool {
            true
        }
    }

    /// Everything is live except the listed `(identity, generation)` pairs.
    struct Dead(BTreeSet<(String, u64)>);

    impl Dead {
        fn of(items: &[(&str, u64)]) -> Self {
            Self(items.iter().map(|(id, g)| ((*id).to_owned(), *g)).collect())
        }
    }

    impl KvLivenessOracle for Dead {
        fn is_live(&self, dependency: &KvDependency) -> bool {
            !self
                .0
                .contains(&(dependency.identity.clone(), dependency.generation.0))
        }
    }

    fn salt() -> &'static KvSalt {
        static SALT: std::sync::OnceLock<KvSalt> = std::sync::OnceLock::new();
        SALT.get_or_init(|| {
            KvSalt::derive(
                &ptr_types::PrincipalId::from("p"),
                &ptr_types::ProjectId::from("proj"),
                &ptr_types::ModelVersion::from("m"),
                ByteTokenizer::new().identity(),
                "f32",
                "schema",
            )
        })
    }

    fn other_salt() -> &'static KvSalt {
        static SALT: std::sync::OnceLock<KvSalt> = std::sync::OnceLock::new();
        SALT.get_or_init(|| {
            KvSalt::derive(
                &ptr_types::PrincipalId::from("q"),
                &ptr_types::ProjectId::from("proj"),
                &ptr_types::ModelVersion::from("m"),
                ByteTokenizer::new().identity(),
                "f32",
                "schema",
            )
        })
    }

    const CHAIN: KvChainId = KvChainId([7; 32]);
    const OTHER_CHAIN: KvChainId = KvChainId([8; 32]);

    /// Deterministic tokens from a seed (xorshift), values below 256.
    fn toks(seed: u64, len: usize) -> Vec<TokenId> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                TokenId((state % 256) as u32)
            })
            .collect()
    }

    fn ident() -> ptr_model_api::TokenizerIdentity {
        ByteTokenizer::new().identity().clone()
    }

    fn raw(tokens: Vec<TokenId>) -> PromptSegment {
        PromptSegment::raw_from_tokens(ident(), tokens).unwrap()
    }

    fn sem(key: &str, generation: u64, tokens: Vec<TokenId>) -> PromptSegment {
        PromptSegment::from_tokens(
            ident(),
            SegmentSource::Semantic {
                key: key.to_owned(),
                generation: Generation(generation),
            },
            tokens,
        )
        .unwrap()
    }

    fn mem(id: &str, generation: u64, tokens: Vec<TokenId>) -> PromptSegment {
        PromptSegment::from_tokens(
            ident(),
            SegmentSource::Memory {
                id: KnowledgeObjectId::from(id),
                generation: Generation(generation),
            },
            tokens,
        )
        .unwrap()
    }

    fn derived(id: &str, generation: u64, tokens: Vec<TokenId>) -> PromptSegment {
        PromptSegment::from_tokens(
            ident(),
            SegmentSource::Derived {
                id: EvidenceId::from(id),
                generation: Generation(generation),
            },
            tokens,
        )
        .unwrap()
    }

    fn request(segments: Vec<PromptSegment>) -> SegmentedModelRequest {
        SegmentedModelRequest::new(
            RequestId::from("r"),
            SemanticContext {
                revision: Revision(0),
                entries: Vec::new(),
            },
            ident(),
            segments,
        )
        .unwrap()
    }

    /// Rows that are a pure function of the token ids.
    fn rows_for(tokens: &[TokenId]) -> SecondaryRows {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for token in tokens {
            for column in 0..ROW {
                keys.push(token.0 as f32 * 0.25 + column as f32 - 3.0);
                values.push(token.0 as f32 * -0.5 + column as f32 * 0.125);
            }
        }
        SecondaryRows::new(keys.into(), values.into(), ROW)
    }

    fn config(window: usize) -> SecondaryConfig {
        SecondaryConfig {
            window_tokens: window,
            limits: SecondaryLimits::default(),
        }
    }

    fn index() -> SecondaryIndex {
        SecondaryIndex::new(SecondaryConfig::default()).unwrap()
    }

    fn insert_of<'a>(
        salt: &'a KvSalt,
        chain: &'a KvChainId,
        request: &'a SegmentedModelRequest,
        segment: usize,
    ) -> SecondaryInsert<'a> {
        SecondaryInsert {
            salt,
            chain,
            mode: KvReuseMode::Exact,
            request,
            segment,
            rows: rows_for(request.segments()[segment].tokens()),
            extra_dependencies: Vec::new(),
        }
    }

    fn put(
        index: &mut SecondaryIndex,
        request: &SegmentedModelRequest,
        segment: usize,
    ) -> SecondaryRunId {
        match index
            .insert(insert_of(salt(), &CHAIN, request, segment), &AllLive)
            .unwrap()
        {
            SecondaryInsertOutcome::Inserted(id)
            | SecondaryInsertOutcome::AlreadyPresent(id)
            | SecondaryInsertOutcome::RowsAttached(id) => id,
        }
    }

    fn find(
        index: &mut SecondaryIndex,
        request: &SegmentedModelRequest,
        liveness: &dyn KvLivenessOracle,
    ) -> SecondaryLookupOutcome {
        find_in(index, salt(), &CHAIN, KvReuseMode::Exact, request, liveness)
    }

    fn find_in(
        index: &mut SecondaryIndex,
        salt: &KvSalt,
        chain: &KvChainId,
        mode: KvReuseMode,
        request: &SegmentedModelRequest,
        liveness: &dyn KvLivenessOracle,
    ) -> SecondaryLookupOutcome {
        index
            .lookup_positional(
                SecondaryLookup {
                    salt,
                    chain,
                    mode,
                    request,
                    positions: &KvRequestPositions::Canonical,
                },
                liveness,
            )
            .unwrap()
    }

    fn release_all(index: &mut SecondaryIndex, outcome: &SecondaryLookupOutcome) {
        for splice in &outcome.splices {
            assert!(index.release(splice.pin));
        }
    }

    // ----- rolling hash ----------------------------------------------------------------

    #[test]
    fn rolling_fingerprints_equal_naive_recomputation() {
        let base = Namespace::derive(salt(), &CHAIN, KvReuseMode::Exact).base();
        let mut tokens = toks(1, 300);
        tokens[3] = TokenId(0);
        tokens[4] = TokenId(u32::MAX);
        for width in [1usize, 2, 7, 8, 9, 64, 300] {
            let rolled = window_fingerprints(&tokens, width, base);
            assert_eq!(rolled.len(), tokens.len() - width + 1);
            for (offset, print) in rolled.iter().enumerate() {
                assert_eq!(*print, fingerprint(&tokens[offset..offset + width], base));
                assert!(*print < FINGERPRINT_MODULUS);
            }
        }
        assert!(window_fingerprints(&tokens, 301, base).is_empty());
        assert!(window_fingerprints(&tokens, 0, base).is_empty());
        // A leading token 0 is not the same window as the shifted one.
        assert_ne!(
            fingerprint(&[TokenId(0), TokenId(1)], base),
            fingerprint(&[TokenId(1)], base)
        );
    }

    #[test]
    fn base_depends_on_the_namespace_and_stays_in_range() {
        let a = Namespace::derive(salt(), &CHAIN, KvReuseMode::Exact).base();
        let b = Namespace::derive(salt(), &OTHER_CHAIN, KvReuseMode::Exact).base();
        assert_ne!(a, b);
        for base in [a, b] {
            assert!((MIN_BASE..FINGERPRINT_MODULUS).contains(&base));
        }
    }

    // ----- hits at shifts --------------------------------------------------------------

    fn shifted_pair(
        shift: usize,
        span_len: usize,
    ) -> (SegmentedModelRequest, SegmentedModelRequest) {
        let span = sem("doc", 1, toks(11, span_len));
        let tail = raw(toks(12, 5));
        let origin = request(vec![span.clone(), tail.clone()]);
        let mut segments = Vec::new();
        if shift > 0 {
            segments.push(raw(toks(13, shift)));
        }
        segments.push(span);
        segments.push(tail);
        (origin, request(segments))
    }

    #[test]
    fn a_segment_hits_at_every_preregistered_shift_with_bit_equal_rows() {
        for shift in [1usize, 7, 16, 17, 64, 513] {
            let (origin, shifted) = shifted_pair(shift, 20);
            let mut idx = index();
            let id = put(&mut idx, &origin, 0);
            let outcome = find(&mut idx, &shifted, &AllLive);
            assert!(outcome.refusals.is_empty(), "shift {shift}");
            assert_eq!(outcome.splices.len(), 1, "shift {shift}");
            let splice = &outcome.splices[0];
            assert_eq!(splice.run, id);
            assert_eq!(splice.request_start, shift);
            assert_eq!(splice.len, 20);
            assert_eq!(splice.delta, shift as i64);
            assert_eq!(splice.original_start, 0);
            assert_eq!(splice.rows, rows_for(origin.segments()[0].tokens()));
            assert_eq!(idx.pin_state(&splice.pin), SecondaryPinState::Live);
            assert_eq!(outcome.reused_tokens(), 20);
        }
    }

    #[test]
    fn a_segment_shorter_than_the_window_hits_by_its_short_key() {
        for (shift, len) in [(1usize, 1usize), (7, 7), (3, 5)] {
            let (origin, shifted) = shifted_pair(shift, len);
            let mut idx = index();
            put(&mut idx, &origin, 0);
            let outcome = find(&mut idx, &shifted, &AllLive);
            assert_eq!(outcome.splices.len(), 1, "shift {shift} len {len}");
            assert_eq!(outcome.splices[0].len, len);
            assert_eq!(outcome.splices[0].request_start, shift);
        }
    }

    #[test]
    fn a_short_segment_with_one_token_edited_finds_no_candidate_and_no_refusal() {
        let origin = request(vec![mem("m", 1, toks(1, 5)), raw(toks(2, 4))]);
        let mut edited_tokens = toks(1, 5);
        edited_tokens[2] = TokenId(edited_tokens[2].0 ^ 1);
        let edited = request(vec![
            raw(toks(3, 2)),
            mem("m", 1, edited_tokens),
            raw(toks(2, 4)),
        ]);
        let mut idx = index();
        put(&mut idx, &origin, 0);
        let outcome = find(&mut idx, &edited, &AllLive);
        assert!(outcome.splices.is_empty() && outcome.refusals.is_empty());
    }

    #[test]
    fn non_aligned_whole_segment_insertions_keep_the_neighbours_reusable() {
        for inserted in [1usize, 7] {
            let a = sem("a", 1, toks(21, 24));
            let b = mem("b", 2, toks(22, 5));
            let tail = raw(toks(23, 3));
            let origin = request(vec![a.clone(), b.clone(), tail.clone()]);
            let shifted = request(vec![a, raw(toks(24, inserted)), b, tail]);
            let mut idx = index();
            put(&mut idx, &origin, 0);
            put(&mut idx, &origin, 1);
            let outcome = find(&mut idx, &shifted, &AllLive);
            assert_eq!(outcome.splices.len(), 2, "insertion {inserted}");
            assert_eq!(outcome.splices[0].delta, 0);
            assert_eq!(outcome.splices[1].request_start, 24 + inserted);
            assert_eq!(outcome.splices[1].delta, inserted as i64);
        }
    }

    #[test]
    fn the_last_prompt_token_is_never_covered() {
        let span = sem("doc", 1, toks(31, 12));
        let only = request(vec![span.clone()]);
        let mut idx = index();
        put(&mut idx, &only, 0);
        let outcome = find(&mut idx, &only, &AllLive);
        assert!(outcome.splices.is_empty());
        assert_eq!(outcome.last_token_skips, 1);
        assert_eq!(idx.stats().last_token_skips, 1);
        assert_eq!(idx.stats().active_pins, 0);
        // Followed by one more token the same span is served.
        let longer = request(vec![span, raw(toks(32, 1))]);
        assert_eq!(find(&mut idx, &longer, &AllLive).splices.len(), 1);
    }

    #[test]
    fn rows_are_bit_exact_including_nan_payloads_and_negative_zero() {
        let origin = request(vec![mem("m", 1, toks(41, 9)), raw(toks(42, 2))]);
        let mut rows = rows_for(origin.segments()[0].tokens());
        let mut keys = rows.keys.to_vec();
        keys[0] = f32::from_bits(0x7fc0_0001);
        keys[1] = -0.0;
        rows.keys = keys.into();
        let mut idx = index();
        idx.insert(
            SecondaryInsert {
                rows: rows.clone(),
                ..insert_of(salt(), &CHAIN, &origin, 0)
            },
            &AllLive,
        )
        .unwrap();
        let outcome = find(&mut idx, &origin, &AllLive);
        let got = &outcome.splices[0].rows;
        assert_eq!(got.keys[0].to_bits(), 0x7fc0_0001);
        assert_eq!(got.keys[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(*got, rows);
    }

    // ----- refusals --------------------------------------------------------------------

    /// Origin `[U, R, S, tail]`; the run is S with upstream `{U}` and raw `{R}`.
    struct Fixture {
        idx: SecondaryIndex,
        u: PromptSegment,
        r: PromptSegment,
        s_tokens: Vec<TokenId>,
        tail: PromptSegment,
    }

    fn fixture_with(extras: Vec<KvDependency>) -> Fixture {
        let u = sem("u", 1, toks(51, 10));
        let r = raw(toks(52, 10));
        let s_tokens = toks(53, 14);
        let tail = raw(toks(54, 4));
        let origin = request(vec![
            u.clone(),
            r.clone(),
            mem("s", 1, s_tokens.clone()),
            tail.clone(),
        ]);
        let mut idx = index();
        idx.insert(
            SecondaryInsert {
                extra_dependencies: extras,
                ..insert_of(salt(), &CHAIN, &origin, 2)
            },
            &AllLive,
        )
        .unwrap();
        Fixture {
            idx,
            u,
            r,
            s_tokens,
            tail,
        }
    }

    fn fixture() -> Fixture {
        fixture_with(Vec::new())
    }

    fn base_dep() -> KvDependency {
        KvDependency::base_model("base", Generation(1), [9; 32])
    }

    /// Run the shifted request `[filler, U, R, span, tail]` and return the outcome.
    fn run_case(
        fx: &mut Fixture,
        span: PromptSegment,
        with_u: bool,
        with_r: bool,
        oracle: &dyn KvLivenessOracle,
    ) -> SecondaryLookupOutcome {
        let mut segments = vec![raw(toks(55, 3))];
        if with_u {
            segments.push(fx.u.clone());
        }
        if with_r {
            segments.push(fx.r.clone());
        }
        segments.push(span);
        segments.push(fx.tail.clone());
        find(&mut fx.idx, &request(segments), oracle)
    }

    fn assert_only(outcome: &SecondaryLookupOutcome, reason: SecondaryRefusalReason) {
        assert!(outcome.splices.is_empty(), "no splice for {reason}");
        assert_eq!(outcome.refusals.len(), 1, "one refusal for {reason}");
        assert_eq!(outcome.refusals[0].reason, reason);
    }

    #[test]
    fn the_fixture_hits_when_everything_is_in_order() {
        let mut fx = fixture();
        let span = mem("s", 1, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_eq!(outcome.splices.len(), 1);
        assert!(outcome.refusals.is_empty());
        // Reordering the original prefix is allowed.
        let reordered = request(vec![
            fx.r.clone(),
            mem("s", 1, fx.s_tokens.clone()),
            fx.u.clone(),
            fx.tail.clone(),
        ]);
        assert_eq!(find(&mut fx.idx, &reordered, &AllLive).splices.len(), 1);
    }

    #[test]
    fn source_kind_mismatch_is_the_only_refusal_when_the_kind_differs() {
        let mut fx = fixture();
        let span = derived("s", 1, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::SourceKindMismatch);
        assert_eq!(fx.idx.stats().refusals.source_kind_mismatch, 1);
        assert_eq!(fx.idx.stats().refusals.total(), 1);
        // The run is untouched by a lineage refusal.
        assert!(fx.idx.contains_live_run(outcome.refusals[0].run));
    }

    #[test]
    fn source_identity_mismatch_is_the_only_refusal_when_the_identity_differs() {
        let mut fx = fixture();
        let span = mem("s2", 1, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::SourceIdentityMismatch);
    }

    #[test]
    fn source_generation_mismatch_is_the_only_refusal_when_the_generation_differs() {
        let mut fx = fixture();
        let span = mem("s", 2, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::SourceGenerationMismatch);
    }

    #[test]
    fn source_digest_mismatch_is_the_only_refusal_when_a_token_inside_the_segment_changed() {
        let mut fx = fixture();
        let mut edited = fx.s_tokens.clone();
        let last = edited.len() - 1;
        edited[last] = TokenId(edited[last].0 ^ 0x55);
        let span = mem("s", 1, edited);
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::SourceDigestMismatch);
        assert_eq!(outcome.reused_tokens(), 0);
        // No partial reuse and the stored run survives the refusal.
        assert!(fx.idx.contains_live_run(outcome.refusals[0].run));
    }

    #[test]
    fn original_upstream_dead_is_reported_and_tombstones_the_run() {
        let mut fx = fixture();
        let span = mem("s", 1, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span.clone(), true, true, &Dead::of(&[("u", 1)]));
        assert_only(&outcome, SecondaryRefusalReason::OriginalUpstreamDead);
        assert!(!fx.idx.contains_live_run(outcome.refusals[0].run));
        assert_eq!(fx.idx.stats().tombstoned_at_lookup, 1);
        // The dead run is gone: a later lookup with everything live finds nothing.
        let later = run_case(&mut fx, span, true, true, &AllLive);
        assert!(later.splices.is_empty() && later.refusals.is_empty());
    }

    #[test]
    fn a_dead_own_segment_is_an_original_upstream_dead_refusal() {
        let mut fx = fixture();
        let span = mem("s", 1, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &Dead::of(&[("s", 1)]));
        assert_only(&outcome, SecondaryRefusalReason::OriginalUpstreamDead);
    }

    #[test]
    fn binding_dependency_dead_covers_base_adapter_and_expert() {
        let project = ptr_types::ProjectId::from("proj");
        let pod = ptr_types::PodId::from("pod");
        let expert = KvDependency::expert_fenced(&project, &pod, Generation(1), [3; 32], 4);
        let expert_identity = expert.identity.clone();
        let extras = vec![
            base_dep(),
            KvDependency::adapter("lora", Generation(1), [4; 32]),
            expert,
        ];
        for dead in [
            Dead::of(&[("base", 1)]),
            Dead::of(&[("lora", 1)]),
            Dead::of(&[(expert_identity.as_str(), 1)]),
        ] {
            let mut fx = fixture_with(extras.clone());
            let span = mem("s", 1, fx.s_tokens.clone());
            let outcome = run_case(&mut fx, span, true, true, &dead);
            assert_only(&outcome, SecondaryRefusalReason::BindingDependencyDead);
            assert!(!fx.idx.contains_live_run(outcome.refusals[0].run));
        }
    }

    #[test]
    fn original_upstream_omitted_is_reported_when_a_bound_ancestor_is_missing() {
        let mut fx = fixture();
        let span = mem("s", 1, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, false, true, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::OriginalUpstreamOmitted);
        // Omission does not tombstone: the run is fine for a request that has U.
        assert!(fx.idx.contains_live_run(outcome.refusals[0].run));
    }

    #[test]
    fn a_superseded_ancestor_is_dead_not_omitted() {
        // The request cites U at generation 2; the run recorded U at generation 1, which
        // is dead. The dead check runs first, so this is not an omission.
        let mut fx = fixture();
        let u2 = sem("u", 2, toks(51, 10));
        let request = request(vec![
            u2,
            fx.r.clone(),
            mem("s", 1, fx.s_tokens.clone()),
            fx.tail.clone(),
        ]);
        let outcome = find(&mut fx.idx, &request, &Dead::of(&[("u", 1)]));
        assert_only(&outcome, SecondaryRefusalReason::OriginalUpstreamDead);
    }

    #[test]
    fn raw_ancestor_omitted_is_reported_when_a_raw_ancestor_is_missing() {
        let mut fx = fixture();
        let span = mem("s", 1, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, false, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::RawAncestorOmitted);
    }

    #[test]
    fn several_failing_checks_report_the_first_in_the_fixed_order() {
        let order = [
            // Lineage beats everything below it.
            (
                mem("s", 2, toks(53, 14)),
                false,
                false,
                Dead::of(&[("u", 1), ("base", 1)]),
                SecondaryRefusalReason::SourceGenerationMismatch,
            ),
            // Upstream dead beats binding dead, omitted and raw omitted.
            (
                mem("s", 1, toks(53, 14)),
                false,
                false,
                Dead::of(&[("u", 1), ("base", 1)]),
                SecondaryRefusalReason::OriginalUpstreamDead,
            ),
            // Binding dead beats omitted and raw omitted.
            (
                mem("s", 1, toks(53, 14)),
                false,
                false,
                Dead::of(&[("base", 1)]),
                SecondaryRefusalReason::BindingDependencyDead,
            ),
            // Upstream omitted beats raw omitted.
            (
                mem("s", 1, toks(53, 14)),
                false,
                false,
                Dead::of(&[]),
                SecondaryRefusalReason::OriginalUpstreamOmitted,
            ),
            (
                mem("s", 1, toks(53, 14)),
                true,
                false,
                Dead::of(&[]),
                SecondaryRefusalReason::RawAncestorOmitted,
            ),
        ];
        for (span, with_u, with_r, oracle, expected) in order {
            let mut fx = fixture_with(vec![base_dep()]);
            let outcome = run_case(&mut fx, span, with_u, with_r, &oracle);
            assert_only(&outcome, expected);
        }
        // Kind, identity and generation are ordered as well.
        let mut fx = fixture();
        let span = derived("other", 9, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::SourceKindMismatch);
        let span = mem("other", 9, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::SourceIdentityMismatch);
    }

    #[test]
    fn refusals_of_a_segment_that_another_run_serves_are_suppressed_and_counted() {
        let mut fx = fixture();
        // A second run for the same content under another generation of the identity.
        let other_origin = request(vec![mem("s", 2, fx.s_tokens.clone()), raw(toks(56, 3))]);
        put(&mut fx.idx, &other_origin, 0);
        let span = mem("s", 2, fx.s_tokens.clone());
        let outcome = run_case(&mut fx, span, true, true, &AllLive);
        assert_eq!(outcome.splices.len(), 1);
        assert!(outcome.refusals.is_empty());
        assert_eq!(fx.idx.stats().suppressed_refusals, 1);
        assert_eq!(fx.idx.stats().refusals.total(), 0);
    }

    // ----- multiset --------------------------------------------------------------------

    #[test]
    fn multiplicity_of_original_prefix_segments_is_enforced() {
        let a = sem("a", 1, toks(61, 9));
        let b = sem("b", 1, toks(62, 9));
        let s = mem("s", 1, toks(63, 11));
        let tail = raw(toks(64, 2));
        let origin = request(vec![
            a.clone(),
            a.clone(),
            b.clone(),
            s.clone(),
            tail.clone(),
        ]);
        let mut idx = index();
        put(&mut idx, &origin, 3);
        // One copy of `a` is not enough.
        let one = request(vec![a.clone(), b.clone(), s.clone(), tail.clone()]);
        let outcome = find(&mut idx, &one, &AllLive);
        assert_only(&outcome, SecondaryRefusalReason::OriginalUpstreamOmitted);
        // Two copies, in any order and with other segments between, are enough.
        let reordered = request(vec![
            b.clone(),
            a.clone(),
            raw(toks(65, 3)),
            a.clone(),
            s.clone(),
            tail.clone(),
        ]);
        let outcome = find(&mut idx, &reordered, &AllLive);
        assert_eq!(outcome.splices.len(), 1);
        // The multiset is judged over the whole request, so the ancestors may also come
        // after the span.
        let after = request(vec![s, a.clone(), b, a, tail]);
        assert_eq!(find(&mut idx, &after, &AllLive).splices.len(), 1);
    }

    #[test]
    fn the_span_itself_does_not_count_as_its_own_ancestor() {
        // Origin `[X, X']` where both segments have identical lineage: the run for the
        // second one needs one earlier X.
        let x = sem("x", 1, toks(71, 10));
        let tail = raw(toks(72, 2));
        let origin = request(vec![x.clone(), x.clone(), tail.clone()]);
        let mut idx = index();
        put(&mut idx, &origin, 1);
        let single = request(vec![raw(toks(73, 2)), x.clone(), tail.clone()]);
        assert_only(
            &find(&mut idx, &single, &AllLive),
            SecondaryRefusalReason::OriginalUpstreamOmitted,
        );
        // With a second copy present both copies are served by the one run.
        let double = request(vec![x.clone(), x, tail]);
        assert_eq!(find(&mut idx, &double, &AllLive).splices.len(), 2);
    }

    #[test]
    fn raw_ancestor_multiplicity_is_enforced() {
        let r = raw(toks(81, 8));
        let s = mem("s", 1, toks(82, 10));
        let tail = raw(toks(83, 2));
        let origin = request(vec![r.clone(), r.clone(), s.clone(), tail.clone()]);
        let mut idx = index();
        put(&mut idx, &origin, 2);
        let one = request(vec![r.clone(), s.clone(), tail.clone()]);
        assert_only(
            &find(&mut idx, &one, &AllLive),
            SecondaryRefusalReason::RawAncestorOmitted,
        );
        let two = request(vec![r.clone(), raw(toks(84, 3)), r, s, tail]);
        assert_eq!(find(&mut idx, &two, &AllLive).splices.len(), 1);
    }

    // ----- namespaces ------------------------------------------------------------------

    #[test]
    fn namespaces_separate_by_salt_chain_and_mode() {
        let (origin, shifted) = shifted_pair(7, 20);
        let mut idx = index();
        put(&mut idx, &origin, 0);
        let hit = |idx: &mut SecondaryIndex, salt: &KvSalt, chain: &KvChainId, mode| {
            let outcome = find_in(idx, salt, chain, mode, &shifted, &AllLive);
            assert!(outcome.refusals.is_empty());
            outcome.splices.len()
        };
        assert_eq!(hit(&mut idx, salt(), &CHAIN, KvReuseMode::Exact), 1);
        assert_eq!(hit(&mut idx, other_salt(), &CHAIN, KvReuseMode::Exact), 0);
        assert_eq!(hit(&mut idx, salt(), &OTHER_CHAIN, KvReuseMode::Exact), 0);
        assert_eq!(hit(&mut idx, salt(), &CHAIN, KvReuseMode::Approximate), 0);
        // The same content under another salt coexists as a second run.
        idx.insert(insert_of(other_salt(), &CHAIN, &origin, 0), &AllLive)
            .unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(hit(&mut idx, other_salt(), &CHAIN, KvReuseMode::Exact), 1);
    }

    #[test]
    fn an_approximate_run_is_refused() {
        let (origin, _) = shifted_pair(1, 20);
        let mut idx = index();
        let error = idx
            .insert(
                SecondaryInsert {
                    mode: KvReuseMode::Approximate,
                    ..insert_of(salt(), &CHAIN, &origin, 0)
                },
                &AllLive,
            )
            .unwrap_err();
        assert_eq!(error, SecondaryError::NotExact);
        assert!(idx.is_empty());
    }

    #[test]
    fn non_canonical_positions_are_refused() {
        let (origin, shifted) = shifted_pair(1, 20);
        let mut idx = index();
        put(&mut idx, &origin, 0);
        for positions in [
            KvRequestPositions::Offset(3),
            KvRequestPositions::Explicit(vec![0; shifted.total_tokens()]),
        ] {
            let error = idx
                .lookup_positional(
                    SecondaryLookup {
                        salt: salt(),
                        chain: &CHAIN,
                        mode: KvReuseMode::Exact,
                        request: &shifted,
                        positions: &positions,
                    },
                    &AllLive,
                )
                .unwrap_err();
            assert_eq!(error, SecondaryError::NonCanonicalPosition);
        }
        assert_eq!(idx.stats().lookups, 0);
    }

    // ----- selection and dedup ---------------------------------------------------------

    #[test]
    fn candidates_are_chosen_by_lowest_original_start_then_lowest_run_id() {
        let c = sem("c", 1, toks(91, 12));
        let tail = raw(toks(92, 3));
        let x = raw(toks(93, 3));
        let y = raw(toks(94, 3));
        let mut idx = index();
        // Run ids 0, 1, 2 with original starts 0, 3, 3.
        let id0 = put(&mut idx, &request(vec![c.clone(), tail.clone()]), 0);
        let id1 = put(
            &mut idx,
            &request(vec![x.clone(), c.clone(), tail.clone()]),
            1,
        );
        let id2 = put(
            &mut idx,
            &request(vec![y.clone(), c.clone(), tail.clone()]),
            1,
        );
        assert!(id0 < id1 && id1 < id2);
        let all = request(vec![x.clone(), y.clone(), c.clone(), tail.clone()]);
        let outcome = find(&mut idx, &all, &AllLive);
        assert_eq!(outcome.splices[0].run, id0);
        release_all(&mut idx, &outcome);
        // Without the run at start 0 (its request needs nothing, but remove it).
        idx.invalidate_identities(&["c".to_owned()].into_iter().collect());
        assert!(idx.is_empty());
        let id1 = put(
            &mut idx,
            &request(vec![x.clone(), c.clone(), tail.clone()]),
            1,
        );
        let id2 = put(
            &mut idx,
            &request(vec![y.clone(), c.clone(), tail.clone()]),
            1,
        );
        let outcome = find(&mut idx, &all, &AllLive);
        assert_eq!(outcome.splices[0].run, id1.min(id2));
        // Longest span first: a candidate can only be as long as its segment, so the rule
        // is exercised by the equal-length tie above.
        assert_eq!(outcome.splices[0].len, 12);
    }

    #[test]
    fn insert_is_idempotent_and_dedups_by_the_prefix_multiset() {
        let c = sem("c", 1, toks(101, 12));
        let tail = raw(toks(102, 2));
        let x = raw(toks(103, 4));
        let y = raw(toks(104, 3));
        let mut idx = index();
        let late = request(vec![x.clone(), y.clone(), c.clone(), tail.clone()]);
        let early = request(vec![c.clone(), tail.clone()]);
        let first = idx
            .insert(insert_of(salt(), &CHAIN, &late, 2), &AllLive)
            .unwrap();
        let again = idx
            .insert(insert_of(salt(), &CHAIN, &late, 2), &AllLive)
            .unwrap();
        assert!(matches!(first, SecondaryInsertOutcome::Inserted(_)));
        assert!(matches!(again, SecondaryInsertOutcome::AlreadyPresent(_)));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.stats().duplicate_inserts, 1);
        // Another prefix multiset is another run.
        idx.insert(insert_of(salt(), &CHAIN, &early, 0), &AllLive)
            .unwrap();
        assert_eq!(idx.len(), 2);
        // The same multiset in another order is the same run.
        let swapped = request(vec![y, x, c, tail]);
        let outcome = idx
            .insert(insert_of(salt(), &CHAIN, &swapped, 2), &AllLive)
            .unwrap();
        assert!(matches!(outcome, SecondaryInsertOutcome::AlreadyPresent(_)));
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn insert_validates_its_input() {
        let (origin, _) = shifted_pair(1, 20);
        let mut idx = index();
        let mut outside = insert_of(salt(), &CHAIN, &origin, 0);
        outside.segment = 9;
        assert_eq!(
            idx.insert(outside, &AllLive).unwrap_err(),
            SecondaryError::SegmentOutOfRange
        );
        let mut bad = insert_of(salt(), &CHAIN, &origin, 0);
        bad.rows.row_width = 5;
        assert_eq!(
            idx.insert(bad, &AllLive).unwrap_err(),
            SecondaryError::RowShape
        );
        let mut short = insert_of(salt(), &CHAIN, &origin, 0);
        short.rows.values = short.rows.values[1..].to_vec().into();
        assert_eq!(
            idx.insert(short, &AllLive).unwrap_err(),
            SecondaryError::RowShape
        );
        assert_eq!(
            idx.insert(
                insert_of(salt(), &CHAIN, &origin, 0),
                &Dead::of(&[("doc", 1)])
            )
            .unwrap_err(),
            SecondaryError::DependencyNotLive
        );
        let dead_extra = SecondaryInsert {
            extra_dependencies: vec![base_dep()],
            ..insert_of(salt(), &CHAIN, &origin, 0)
        };
        assert_eq!(
            idx.insert(dead_extra, &Dead::of(&[("base", 1)]))
                .unwrap_err(),
            SecondaryError::DependencyNotLive
        );
        assert!(idx.is_empty());
        assert_eq!(idx.stats().inserts, 0);
    }

    #[test]
    fn a_conflicting_reinsert_is_a_typed_error_and_changes_nothing() {
        let (origin, _) = shifted_pair(1, 20);
        let mut idx = index();
        put(&mut idx, &origin, 0);
        let mut other = insert_of(salt(), &CHAIN, &origin, 0);
        let mut keys = other.rows.keys.to_vec();
        keys[0] += 1.0;
        other.rows.keys = keys.into();
        assert_eq!(
            idx.insert(other, &AllLive).unwrap_err(),
            SecondaryError::RowsDigestMismatch
        );
        let outcome = find(&mut idx, &origin, &AllLive);
        assert_eq!(outcome.last_token_skips, 0);
        assert_eq!(idx.len(), 1);
    }

    // ----- bounds and LRU --------------------------------------------------------------

    fn small(limits: SecondaryLimits) -> SecondaryIndex {
        SecondaryIndex::new(SecondaryConfig {
            window_tokens: 4,
            limits,
        })
        .unwrap()
    }

    fn solo(tag: u64, len: usize) -> SegmentedModelRequest {
        request(vec![
            mem(&format!("m{tag}"), 1, toks(200 + tag, len)),
            raw(toks(300 + tag, 2)),
        ])
    }

    #[test]
    fn invalid_configs_are_refused() {
        let zero = |edit: fn(&mut SecondaryLimits)| {
            let mut limits = SecondaryLimits::default();
            edit(&mut limits);
            SecondaryIndex::new(SecondaryConfig {
                window_tokens: 8,
                limits,
            })
            .unwrap_err()
        };
        assert_eq!(zero(|l| l.max_runs = 0), SecondaryError::InvalidConfig);
        assert_eq!(zero(|l| l.max_row_bytes = 0), SecondaryError::InvalidConfig);
        assert_eq!(
            zero(|l| l.max_indexed_windows = 0),
            SecondaryError::InvalidConfig
        );
        assert_eq!(
            zero(|l| l.max_tokens_per_run = 0),
            SecondaryError::InvalidConfig
        );
        assert_eq!(
            zero(|l| l.max_metadata_bytes = 0),
            SecondaryError::InvalidConfig
        );
        assert_eq!(
            zero(|l| l.max_candidates_per_window = 0),
            SecondaryError::InvalidConfig
        );
        assert_eq!(
            SecondaryIndex::new(config(0)).unwrap_err(),
            SecondaryError::InvalidConfig
        );
        assert_eq!(
            SecondaryIndex::new(SecondaryConfig {
                window_tokens: 9,
                limits: SecondaryLimits {
                    max_tokens_per_run: 8,
                    ..SecondaryLimits::default()
                },
            })
            .unwrap_err(),
            SecondaryError::InvalidConfig
        );
    }

    #[test]
    fn the_run_bound_evicts_the_least_recently_used_unpinned_run() {
        let mut idx = small(SecondaryLimits {
            max_runs: 3,
            ..SecondaryLimits::default()
        });
        let requests: Vec<_> = (0..4).map(|tag| solo(tag, 8)).collect();
        let ids: Vec<_> = requests[..3].iter().map(|r| put(&mut idx, r, 0)).collect();
        // Use run 0 so run 1 is now the least recently used.
        let used = find(&mut idx, &requests[0], &AllLive);
        release_all(&mut idx, &used);
        let new = put(&mut idx, &requests[3], 0);
        assert_eq!(idx.len(), 3);
        assert!(idx.contains_live_run(ids[0]));
        assert!(!idx.contains_live_run(ids[1]));
        assert!(idx.contains_live_run(ids[2]));
        assert!(idx.contains_live_run(new));
        assert_eq!(idx.stats().evictions, 1);
    }

    #[test]
    fn a_pinned_run_is_never_evicted_and_a_full_pinned_index_refuses_typed() {
        let mut idx = small(SecondaryLimits {
            max_runs: 2,
            ..SecondaryLimits::default()
        });
        let requests: Vec<_> = (0..3).map(|tag| solo(tag, 8)).collect();
        put(&mut idx, &requests[0], 0);
        put(&mut idx, &requests[1], 0);
        let first = find(&mut idx, &requests[0], &AllLive);
        let second = find(&mut idx, &requests[1], &AllLive);
        let error = idx
            .insert(insert_of(salt(), &CHAIN, &requests[2], 0), &AllLive)
            .unwrap_err();
        assert_eq!(
            error,
            SecondaryError::CapacityExhausted(SecondaryResource::Runs)
        );
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.stats().evictions, 0);
        assert_eq!(idx.stats().capacity_refusals, 1);
        // Releasing one makes room for exactly that one to be evicted.
        release_all(&mut idx, &first);
        put(&mut idx, &requests[2], 0);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.stats().evictions, 1);
        assert!(idx.contains_live_run(second.splices[0].run));
    }

    #[test]
    fn row_bytes_windows_metadata_tokens_and_candidates_are_bounded() {
        // Row bytes: 8 tokens * 6 * 2 * 4 = 384 per run.
        let mut idx = small(SecondaryLimits {
            max_row_bytes: 400,
            ..SecondaryLimits::default()
        });
        let requests: Vec<_> = (0..3).map(|tag| solo(tag, 8)).collect();
        put(&mut idx, &requests[0], 0);
        put(&mut idx, &requests[1], 0);
        assert_eq!(idx.len(), 1, "the second evicts the first");
        assert!(idx.stats().row_bytes <= 400);
        let too_big = solo(9, 9);
        assert_eq!(
            idx.insert(insert_of(salt(), &CHAIN, &too_big, 0), &AllLive)
                .unwrap_err(),
            SecondaryError::CapacityExhausted(SecondaryResource::RowBytes)
        );
        assert_eq!(idx.len(), 1, "a refusal evicts nothing");

        // Indexed windows: an 8-token run has up to 5 windows at W=4.
        let mut idx = small(SecondaryLimits {
            max_indexed_windows: 6,
            ..SecondaryLimits::default()
        });
        put(&mut idx, &requests[0], 0);
        put(&mut idx, &requests[1], 0);
        assert_eq!(idx.len(), 1);
        assert!(idx.stats().indexed_windows <= 6);

        // Metadata bytes.
        let mut idx = small(SecondaryLimits {
            max_metadata_bytes: 400,
            ..SecondaryLimits::default()
        });
        put(&mut idx, &requests[0], 0);
        assert_eq!(
            idx.insert(insert_of(salt(), &CHAIN, &solo(7, 80), 0), &AllLive)
                .unwrap_err(),
            SecondaryError::CapacityExhausted(SecondaryResource::MetadataBytes)
        );

        // Tokens per run.
        let mut idx = small(SecondaryLimits {
            max_tokens_per_run: 8,
            ..SecondaryLimits::default()
        });
        assert_eq!(
            idx.insert(insert_of(salt(), &CHAIN, &solo(8, 9), 0), &AllLive)
                .unwrap_err(),
            SecondaryError::CapacityExhausted(SecondaryResource::TokensPerRun)
        );

        // Candidates per window: runs that share a window beyond the bound are refused.
        let mut idx = small(SecondaryLimits {
            max_candidates_per_window: 2,
            ..SecondaryLimits::default()
        });
        let shared = toks(400, 8);
        for tag in 0..3u64 {
            let req = request(vec![
                mem(&format!("c{tag}"), 1, shared.clone()),
                raw(toks(401, 1)),
            ]);
            let result = idx.insert(insert_of(salt(), &CHAIN, &req, 0), &AllLive);
            if tag < 2 {
                result.unwrap();
            } else {
                assert_eq!(
                    result.unwrap_err(),
                    SecondaryError::CapacityExhausted(SecondaryResource::CandidatesPerWindow)
                );
            }
        }
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn accounting_returns_to_zero_when_every_run_is_gone() {
        let mut idx = index();
        let requests: Vec<_> = (0..5).map(|tag| solo(tag, 12)).collect();
        for r in &requests {
            put(&mut idx, r, 0);
        }
        assert!(idx.stats().row_bytes > 0 && idx.stats().indexed_windows > 0);
        let all: BTreeSet<String> = (0..5).map(|tag| format!("m{tag}")).collect();
        assert_eq!(idx.invalidate_identities(&all), 5);
        let stats = idx.stats();
        assert_eq!(
            (
                stats.runs,
                stats.row_bytes,
                stats.indexed_windows,
                stats.metadata_bytes
            ),
            (0, 0, 0, 0)
        );
        assert!(idx.postings.is_empty() && idx.by_identity.is_empty() && idx.by_key.is_empty());
    }

    // ----- pins, tombstones, invalidation ----------------------------------------------

    #[test]
    fn a_tombstoned_pinned_run_is_unreachable_and_reclaimed_at_the_last_release() {
        let (origin, shifted) = shifted_pair(7, 20);
        let mut idx = index();
        let id = put(&mut idx, &origin, 0);
        let first = find(&mut idx, &shifted, &AllLive);
        let second = find(&mut idx, &shifted, &AllLive);
        assert_eq!(idx.stats().active_pins, 2);
        assert_eq!(
            idx.invalidate_identities(&["doc".to_owned()].into_iter().collect()),
            1
        );
        // Still held for the pins, but dead for everyone.
        assert_eq!(idx.len(), 1);
        assert!(!idx.contains_live_run(id));
        assert_eq!(
            idx.pin_state(&first.splices[0].pin),
            SecondaryPinState::Revoked
        );
        assert_eq!(
            idx.pin_state(&second.splices[0].pin),
            SecondaryPinState::Revoked
        );
        let miss = find(&mut idx, &shifted, &AllLive);
        assert!(miss.splices.is_empty() && miss.refusals.is_empty());
        assert_eq!(idx.stats().tombstoned_skips, 1);
        assert_eq!(idx.stats().tombstoned_pinned_runs, 1);
        // The slot is draining: the same content is refused rather than overwritten.
        assert_eq!(
            idx.insert(insert_of(salt(), &CHAIN, &origin, 0), &AllLive)
                .unwrap_err(),
            SecondaryError::SlotDraining
        );
        assert!(idx.release(first.splices[0].pin));
        assert_eq!(idx.len(), 1);
        assert!(!idx.release(first.splices[0].pin), "a pin releases once");
        assert!(idx.release(second.splices[0].pin));
        assert!(idx.is_empty());
        assert_eq!(idx.stats().row_bytes, 0);
        // And the slot is free again.
        put(&mut idx, &origin, 0);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn an_unpinned_run_is_removed_at_once_and_a_live_pin_is_revalidated() {
        let (origin, shifted) = shifted_pair(7, 20);
        let mut idx = index();
        put(&mut idx, &origin, 0);
        let outcome = find(&mut idx, &shifted, &AllLive);
        let pin = outcome.splices[0].pin;
        assert_eq!(idx.revalidate(&pin, &AllLive), SecondaryPinState::Live);
        assert_eq!(
            idx.revalidate(&pin, &Dead::of(&[("doc", 1)])),
            SecondaryPinState::Revoked
        );
        assert_eq!(idx.pin_state(&pin), SecondaryPinState::Revoked);
        assert_eq!(idx.revalidate(&pin, &AllLive), SecondaryPinState::Revoked);
        idx.release(pin);
        assert!(idx.is_empty());
        // A pin on nothing is revoked.
        assert_eq!(idx.pin_state(&pin), SecondaryPinState::Revoked);
    }

    #[test]
    fn invalidation_reaches_runs_by_identity_generation_kind_and_supersession() {
        let build = |extras: Vec<KvDependency>| {
            let mut idx = index();
            let origin = request(vec![
                sem("u", 1, toks(111, 9)),
                mem("m", 3, toks(112, 9)),
                raw(toks(113, 3)),
            ]);
            idx.insert(
                SecondaryInsert {
                    extra_dependencies: extras,
                    ..insert_of(salt(), &CHAIN, &origin, 1)
                },
                &AllLive,
            )
            .unwrap();
            idx
        };
        let set = |items: &[&str]| -> BTreeSet<String> {
            items.iter().map(|item| (*item).to_owned()).collect()
        };
        // By (identity, generation): the upstream's and the own one's.
        for (identity, generation) in [("u", 1u64), ("m", 3)] {
            let mut idx = build(Vec::new());
            let pairs: BTreeSet<(String, Generation)> =
                [(identity.to_owned(), Generation(generation))]
                    .into_iter()
                    .collect();
            assert_eq!(idx.invalidate_inputs(&pairs), 1, "{identity}");
            assert!(idx.is_empty());
        }
        // A different generation of the same identity does not match inputs, but does
        // match identities.
        let mut idx = build(Vec::new());
        let pairs: BTreeSet<(String, Generation)> =
            [("u".to_owned(), Generation(2))].into_iter().collect();
        assert_eq!(idx.invalidate_inputs(&pairs), 0);
        assert_eq!(idx.invalidate_identities(&set(&["u"])), 1);
        // By kind: a Segment(Semantic) dependency is not a Segment(Memory) one.
        let mut idx = build(Vec::new());
        assert_eq!(
            idx.invalidate_kind_identities(
                KvDependencyKind::Segment(SegmentKind::Memory),
                &set(&["u"])
            ),
            0
        );
        assert_eq!(
            idx.invalidate_kind_identities(
                KvDependencyKind::Segment(SegmentKind::Semantic),
                &set(&["u"])
            ),
            1
        );
        // Supersession: strictly below the live generation.
        let mut idx = build(Vec::new());
        let live: BTreeMap<String, Generation> =
            [("u".to_owned(), Generation(1))].into_iter().collect();
        assert_eq!(idx.invalidate_superseded(&live), 0);
        let live: BTreeMap<String, Generation> =
            [("u".to_owned(), Generation(2))].into_iter().collect();
        assert_eq!(idx.invalidate_superseded(&live), 1);
        // Base model and adapter supersession are kind-scoped.
        let mut idx = build(vec![
            base_dep(),
            KvDependency::adapter("lora", Generation(1), [4; 32]),
        ]);
        let live: BTreeMap<String, Generation> =
            [("lora".to_owned(), Generation(2))].into_iter().collect();
        assert_eq!(
            idx.invalidate_kind_superseded(KvDependencyKind::BaseModel, &live),
            0
        );
        assert_eq!(
            idx.invalidate_kind_superseded(KvDependencyKind::Adapter, &live),
            1
        );
        assert_eq!(idx.stats().invalidated, 1);
    }

    #[test]
    fn expert_dependencies_are_reached_by_range_and_by_fence_epoch() {
        let project = ptr_types::ProjectId::from("proj");
        let pods = ["a", "b"].map(ptr_types::PodId::from);
        let build = || {
            let mut idx = index();
            for (n, pod) in pods.iter().enumerate() {
                let origin = request(vec![
                    mem(&format!("m{n}"), 1, toks(120 + n as u64, 9)),
                    raw(toks(125, 2)),
                ]);
                idx.insert(
                    SecondaryInsert {
                        extra_dependencies: vec![KvDependency::expert_fenced(
                            &project,
                            pod,
                            Generation(1),
                            [5; 32],
                            2 + n as u64,
                        )],
                        ..insert_of(salt(), &CHAIN, &origin, 0)
                    },
                    &AllLive,
                )
                .unwrap();
            }
            // One run without any expert.
            put(&mut idx, &solo(40, 9), 0);
            idx
        };
        let mut idx = build();
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.invalidate_all_experts(), 2);
        assert_eq!(idx.len(), 1);

        let identity = |pod: &ptr_types::PodId| ptr_state::expert_key(&project.0, &pod.0);
        let mut idx = build();
        // Epoch 2 is not below 2: only the run that recorded epoch 2 is spared; a fence at
        // epoch 3 reaches both runs only for the one that recorded 2.
        let fences: BTreeMap<String, u64> = [(identity(&pods[0]), 2)].into_iter().collect();
        assert_eq!(idx.invalidate_expert_fences(&fences), 0);
        let fences: BTreeMap<String, u64> = [(identity(&pods[0]), 3)].into_iter().collect();
        assert_eq!(idx.invalidate_expert_fences(&fences), 1);
        assert_eq!(idx.len(), 2);
        // The other expert is untouched by a fence naming only the first.
        let fences: BTreeMap<String, u64> = [(identity(&pods[1]), 3)].into_iter().collect();
        assert_eq!(idx.invalidate_expert_fences(&fences), 0);
    }

    // ----- snapshot --------------------------------------------------------------------

    fn populated(order: &[usize]) -> SecondaryIndex {
        let origins: Vec<SegmentedModelRequest> = vec![
            request(vec![
                sem("u", 1, toks(131, 9)),
                mem("a", 1, toks(132, 14)),
                raw(toks(133, 2)),
            ]),
            request(vec![
                raw(toks(134, 5)),
                derived("d", 4, toks(135, 6)),
                raw(toks(136, 2)),
            ]),
            request(vec![mem("e", 2, toks(137, 10)), raw(toks(138, 2))]),
        ];
        let segments = [1usize, 1, 0];
        let mut idx = index();
        for n in order {
            let extras = if *n == 2 {
                vec![base_dep()]
            } else {
                Vec::new()
            };
            idx.insert(
                SecondaryInsert {
                    extra_dependencies: extras,
                    ..insert_of(salt(), &CHAIN, &origins[*n], segments[*n])
                },
                &AllLive,
            )
            .unwrap();
        }
        idx
    }

    #[test]
    fn an_empty_index_exports_nothing_and_the_section_is_canonical() {
        assert!(index().export_section().is_none());
        let forward = populated(&[0, 1, 2]).export_section().unwrap();
        let backward = populated(&[2, 1, 0]).export_section().unwrap();
        assert_eq!(forward, backward, "independent of insertion order");
        assert!(forward.starts_with(SECTION_MAGIC));
        // Tombstoned runs are not exported.
        let mut idx = populated(&[0, 1, 2]);
        idx.invalidate_identities(&["d".to_owned()].into_iter().collect());
        let fewer = idx.export_section().unwrap();
        assert!(fewer.len() < forward.len());
    }

    #[test]
    fn a_restored_section_has_no_rows_and_misses_until_rows_are_reattached() {
        let origin = request(vec![
            sem("u", 1, toks(131, 9)),
            mem("a", 1, toks(132, 14)),
            raw(toks(133, 2)),
        ]);
        let mut source = index();
        put(&mut source, &origin, 1);
        let section = source.export_section().unwrap();

        let mut restored = index();
        assert_eq!(restored.restore_section(&section).unwrap(), 1);
        assert_eq!(restored.stats().row_bytes, 0);
        assert_eq!(restored.restore_section(&section).unwrap(), 0, "idempotent");
        let shifted = request(vec![
            raw(toks(139, 5)),
            sem("u", 1, toks(131, 9)),
            mem("a", 1, toks(132, 14)),
            raw(toks(133, 2)),
        ]);
        let miss = find(&mut restored, &shifted, &AllLive);
        assert!(miss.splices.is_empty() && miss.refusals.is_empty());
        assert_eq!(restored.stats().rows_missing_misses, 1);

        // A wrong row buffer is a typed error and the run stays row-less.
        let mut wrong = insert_of(salt(), &CHAIN, &origin, 1);
        let mut keys = wrong.rows.keys.to_vec();
        keys[3] = f32::from_bits(keys[3].to_bits() ^ 1);
        wrong.rows.keys = keys.into();
        assert_eq!(
            restored.insert(wrong, &AllLive).unwrap_err(),
            SecondaryError::RowsDigestMismatch
        );
        assert_eq!(restored.stats().row_bytes, 0);
        // A dead dependency refuses the attach.
        assert_eq!(
            restored
                .insert(
                    insert_of(salt(), &CHAIN, &origin, 1),
                    &Dead::of(&[("u", 1)])
                )
                .unwrap_err(),
            SecondaryError::DependencyNotLive
        );
        // The right rows under a live oracle attach and the run serves.
        let attached = restored
            .insert(insert_of(salt(), &CHAIN, &origin, 1), &AllLive)
            .unwrap();
        assert!(matches!(attached, SecondaryInsertOutcome::RowsAttached(_)));
        assert_eq!(restored.stats().rows_attached, 1);
        assert!(restored.stats().row_bytes > 0);
        let hit = find(&mut restored, &shifted, &AllLive);
        assert_eq!(hit.splices.len(), 1);
        assert_eq!(hit.splices[0].rows, rows_for(origin.segments()[1].tokens()));
        // Attaching again is a duplicate.
        assert!(matches!(
            restored
                .insert(insert_of(salt(), &CHAIN, &origin, 1), &AllLive)
                .unwrap(),
            SecondaryInsertOutcome::AlreadyPresent(_)
        ));
        // The restored run exports the same bytes as the source.
        release_all(&mut restored, &hit);
        assert_eq!(restored.export_section().unwrap(), section);
    }

    #[test]
    fn a_restored_run_whose_input_died_is_refused_at_the_first_use() {
        let origin = request(vec![
            sem("u", 1, toks(131, 9)),
            mem("a", 1, toks(132, 14)),
            raw(toks(133, 2)),
        ]);
        let mut source = index();
        put(&mut source, &origin, 1);
        let section = source.export_section().unwrap();
        let mut restored = index();
        restored.restore_section(&section).unwrap();
        // Rows cannot be attached to a dead run, so the run never serves.
        assert_eq!(
            restored
                .insert(
                    insert_of(salt(), &CHAIN, &origin, 1),
                    &Dead::of(&[("u", 1)])
                )
                .unwrap_err(),
            SecondaryError::DependencyNotLive
        );
        let miss = find(&mut restored, &origin, &Dead::of(&[("u", 1)]));
        assert!(miss.splices.is_empty());
    }

    #[test]
    fn the_section_is_decoded_strictly() {
        let good = populated(&[0, 1, 2]).export_section().unwrap();
        let decode = |bytes: &[u8]| index().restore_section(bytes);
        assert_eq!(decode(&good).unwrap(), 3);

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 1;
        assert_eq!(
            decode(&bad_magic).unwrap_err(),
            SecondaryError::Section(SecondarySectionError::BadMagic)
        );
        for cut in [0usize, 5, 8, 12, good.len() / 2, good.len() - 1] {
            assert_eq!(
                decode(&good[..cut]).unwrap_err(),
                SecondaryError::Section(SecondarySectionError::Truncated),
                "cut at {cut}"
            );
        }
        let mut trailing = good.clone();
        trailing.push(0);
        assert_eq!(
            decode(&trailing).unwrap_err(),
            SecondaryError::Section(SecondarySectionError::TrailingBytes)
        );
        let mut too_many = good.clone();
        too_many[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(
            decode(&too_many).unwrap_err(),
            SecondaryError::Section(SecondarySectionError::OutOfBounds("run count"))
        );
        // A count above the index's own bound is refused as well.
        let mut tiny = SecondaryIndex::new(SecondaryConfig {
            window_tokens: 8,
            limits: SecondaryLimits {
                max_runs: 2,
                ..SecondaryLimits::default()
            },
        })
        .unwrap();
        assert_eq!(
            tiny.restore_section(&good).unwrap_err(),
            SecondaryError::Section(SecondarySectionError::OutOfBounds("run count"))
        );
        let oversized = vec![0u8; SECONDARY_SECTION_MAX_BYTES + 1];
        assert_eq!(
            decode(&oversized).unwrap_err(),
            SecondaryError::Section(SecondarySectionError::TooLarge)
        );
        // Flipping the mode byte of the first run (after magic, count and namespace).
        let mut approximate = good.clone();
        approximate[16 + 32] = 1;
        assert_eq!(
            decode(&approximate).unwrap_err(),
            SecondaryError::Section(SecondarySectionError::InvalidValue("mode"))
        );
        // A duplicated run breaks the strictly ascending key order.
        let one = populated(&[0]).export_section().unwrap();
        let mut twice = Vec::new();
        twice.extend_from_slice(SECTION_MAGIC);
        twice.extend_from_slice(&2u64.to_le_bytes());
        twice.extend_from_slice(&one[16..]);
        twice.extend_from_slice(&one[16..]);
        assert_eq!(
            decode(&twice).unwrap_err(),
            SecondaryError::Section(SecondarySectionError::NotCanonical("run order"))
        );
        // A failed restore leaves the index unchanged.
        let mut idx = index();
        assert!(idx.restore_section(&bad_magic).is_err());
        assert!(idx.is_empty() && idx.stats().indexed_windows == 0);
    }

    #[test]
    fn a_restore_that_does_not_fit_changes_nothing() {
        let good = populated(&[0, 1, 2]).export_section().unwrap();
        let mut idx = SecondaryIndex::new(SecondaryConfig {
            window_tokens: 8,
            limits: SecondaryLimits {
                max_indexed_windows: 5,
                ..SecondaryLimits::default()
            },
        })
        .unwrap();
        assert_eq!(
            idx.restore_section(&good).unwrap_err(),
            SecondaryError::CapacityExhausted(SecondaryResource::IndexedWindows)
        );
        assert!(idx.is_empty());
    }

    // ----- determinism and collisions --------------------------------------------------

    #[test]
    fn the_same_operations_give_the_same_outcomes_and_bytes() {
        let drive = || {
            let mut idx = populated(&[0, 1, 2]);
            let shifted = request(vec![
                raw(toks(140, 3)),
                sem("u", 1, toks(131, 9)),
                mem("a", 1, toks(132, 14)),
                raw(toks(133, 2)),
            ]);
            let outcome = find(&mut idx, &shifted, &AllLive);
            let bad = request(vec![
                raw(toks(141, 3)),
                mem("a", 1, toks(132, 14)),
                raw(toks(133, 2)),
            ]);
            let refused = find(&mut idx, &bad, &AllLive);
            (
                format!("{outcome:?}"),
                format!("{refused:?}"),
                format!("{:?}", idx.stats()),
                idx.export_section(),
            )
        };
        assert_eq!(drive(), drive());
    }

    #[test]
    fn a_forced_fingerprint_collision_never_yields_a_hit_or_a_refusal() {
        let mut idx = index();
        let origin = request(vec![mem("a", 1, toks(151, 16)), raw(toks(152, 2))]);
        let id = put(&mut idx, &origin, 0);
        // A request of unrelated content whose first window the run is injected under.
        let unrelated = request(vec![mem("a", 1, toks(153, 16)), raw(toks(154, 2))]);
        let base = Namespace::derive(salt(), &CHAIN, KvReuseMode::Exact).base();
        let print = fingerprint(&unrelated.segments()[0].tokens()[..8], base);
        let lookup = SecondaryLookup {
            salt: salt(),
            chain: &CHAIN,
            mode: KvReuseMode::Exact,
            request: &unrelated,
            positions: &KvRequestPositions::Canonical,
        };
        idx.inject_posting(&lookup, print, id);
        let outcome = find(&mut idx, &unrelated, &AllLive);
        assert!(outcome.splices.is_empty());
        assert!(
            outcome.refusals.is_empty(),
            "no verified window, no refusal"
        );
        assert!(idx.stats().fingerprint_collisions >= 1);
        assert_eq!(idx.stats().hits, 0);
        // The true content still hits.
        let again = request(vec![
            raw(toks(155, 4)),
            mem("a", 1, toks(151, 16)),
            raw(toks(152, 2)),
        ]);
        assert_eq!(find(&mut idx, &again, &AllLive).splices.len(), 1);
    }

    #[test]
    fn a_short_key_collision_never_yields_a_hit() {
        let mut idx = index();
        let origin = request(vec![mem("a", 1, toks(161, 5)), raw(toks(162, 2))]);
        let id = put(&mut idx, &origin, 0);
        let other = request(vec![mem("a", 1, toks(163, 5)), raw(toks(162, 2))]);
        let namespace = Namespace::derive(salt(), &CHAIN, KvReuseMode::Exact);
        idx.postings
            .entry((
                namespace,
                PostingKey::Short(short_key(other.segments()[0].tokens())),
            ))
            .or_default()
            .push((id, 0));
        let outcome = find(&mut idx, &other, &AllLive);
        assert!(outcome.splices.is_empty() && outcome.refusals.is_empty());
        assert_eq!(idx.stats().fingerprint_collisions, 1);
    }

    #[test]
    fn runs_with_shared_windows_but_different_content_report_the_lineage_reason() {
        // A run that shares one verified window with the request segment but is another
        // segment is refused for its identity, never reused.
        let shared = toks(171, 8);
        let mut first = shared.clone();
        first.extend(toks(172, 6));
        let mut second = toks(173, 6);
        second.extend(shared);
        let origin = request(vec![mem("one", 1, first), raw(toks(174, 2))]);
        let mut idx = index();
        put(&mut idx, &origin, 0);
        let probe = request(vec![mem("two", 1, second), raw(toks(174, 2))]);
        let outcome = find(&mut idx, &probe, &AllLive);
        assert!(outcome.splices.is_empty());
        assert_eq!(outcome.refusals.len(), 1);
        assert_eq!(
            outcome.refusals[0].reason,
            SecondaryRefusalReason::SourceIdentityMismatch
        );
    }

    #[test]
    fn refusal_codes_are_distinct_and_counts_cover_every_reason() {
        let codes: BTreeSet<&str> = SecondaryRefusalReason::ALL
            .iter()
            .map(|reason| reason.code())
            .collect();
        assert_eq!(codes.len(), 8);
        let mut counts = SecondaryRefusalCounts::default();
        for reason in SecondaryRefusalReason::ALL {
            assert_eq!(counts.get(reason), 0);
            counts.bump(reason);
            assert_eq!(counts.get(reason), 1);
        }
        assert_eq!(counts.total(), 8);
        assert_eq!(
            SecondaryRefusalReason::OriginalUpstreamDead.to_string(),
            "original_upstream_dead"
        );
    }
}
