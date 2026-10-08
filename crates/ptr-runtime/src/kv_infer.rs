//! CPU reference backend that really reuses KV through the runtime prefix catalog
//! (doc 36, phase P3 skeleton).
//!
//! [`ReferenceKvReuseBackend`] owns one in-memory paged KV backend, one
//! [`KvPrefixCatalog`] over it and the R005 tiny reference Qwen2 stack (seeded random
//! weights, `PreRope` keys, `f32`, one-token-at-a-time CPU math, so every run is bitwise
//! reproducible). For a [`SegmentedModelRequest`] it
//!
//! 1. computes chunk keys from the tokens alone and asks the catalog for the longest live
//!    exact prefix of the request's lineage,
//! 2. forks that prefix through the backend's witness-gated path,
//! 3. prefills only the remaining tokens, 16 tokens (one chunk) at a time, publishing and
//!    inserting every completed chunk boundary of the prompt into the catalog,
//! 4. decodes greedily for a bounded number of tokens and
//! 5. returns a [`ModelRun`] whose `KvReuseReport` is `KvReuseMode::Exact`. The default
//!    binding never produces Approximate output; only a [`StackBinding`] whose MoE routes
//!    batch-variantly does, and it then neither looks up nor inserts.
//!
//! # Chain id
//!
//! The catalog's [`KvChainId`] is opaque to it. It is `chain(L-1)` of the per-layer chain
//! of `ptr_pods::derive_kv_layer_chains` (doc 36 section D), computed from the actual
//! weights and configuration of the stack: embedding digest, every layer's norms, K/V
//! projections, QK-norm, attention and MLP weights, RoPE configuration (per the PreRope /
//! PostRope rule), the base model identity and generation, the tokenizer and the backend
//! build identity (also folded into the catalog salt). Two stacks with the same seed but
//! any different weight get different ids. Adapters and MoE experts of a [`StackBinding`]
//! are covered per layer by [`crate::kv_infer_layered`].
//!
//! # RoPE regime
//!
//! A backend built with [`ReferenceKvReuseBackend::with_rope_scaling`] and
//! [`Qwen2RopeScaling::DynamicNtk`] has a length-dependent effective rotary base. At
//! admission the backend fixes that regime once from `prompt_tokens + max_new_tokens`
//! (`Qwen2LayerConfig::rope_regime`); prefill and every decode step of the request run
//! under that one value. The chain id is derived **per request** under its regime, so
//! requests in different regimes never share a catalog entry: a request whose admission
//! regime differs from a cached entry's misses and is recomputed (Exact, never served
//! across a transition). Lengths up to the original maximum positions all give the plain
//! base and so one shared regime. [`ReferenceKvReuseBackend::chain_id`] is the chain of
//! that plain-base regime (the only one without scaling).
//!
//! # Bounds and failure
//!
//! Prompt length, generated length and catalog capacity are bounded
//! ([`ReferenceKvLimits`]). An oversized prompt is refused with
//! [`ReferenceKvError::PromptTooLong`]. A catalog that cannot take a chunk (capacity
//! exhausted, a slot still draining, a dependency no longer live) is a graceful miss on
//! the *insert* side: the chunk is not cached, the refusal is counted in the outcome and
//! the output is unchanged. Capacity never changes a result.
//!
//! # Sessions
//!
//! Every run is a [`KvSession`] ([`ReferenceKvReuseBackend::begin_session`]): the one-shot
//! entry points below drive one to completion under a single-request gate, so their
//! behaviour is unchanged. A caller that drives the steps itself can interleave a
//! revocation between two decode steps deterministically, which is how the R005 race arms
//! are tested. See [`crate::kv_infer_session`].
//!
//! # Positional reuse (R006)
//!
//! [`ReferenceKvReuseBackend::with_layers`] builds the same stack with another layer count
//! (the default stays 4; chain ids and pinned digests of existing configurations do not
//! change) and [`ReferenceKvReuseBackend::with_positional_reuse`] opts the backend into the
//! catalog's secondary index. It is off by default and only
//! [`ReferenceKvReuseBackend::begin_session_positional`] and
//! [`ReferenceKvReuseBackend::probe_positional`] use it; every other entry point behaves
//! exactly as without it. Only a one-layer `PreRope` stack is accepted: the layer-0 K/V rows
//! of a token are a pure function of the token, the cache index of a row equals its absolute
//! position and the stack rotates every cached key at use, so a stored row appended at a new
//! offset gives the same bits as a fresh prefill there. A positional session
//!
//! 1. looks up the primary prefix as before,
//! 2. asks the index for whole segments found at another offset (spans that start below the
//!    primary hit are dropped and their pins released),
//! 3. prefills up to the next span start, appends the span's stored rows as a step of its own
//!    and continues,
//! 4. offers every completed segment's layer-0 rows to the index after a fresh liveness check
//!    (never in a restarted run), recording the base model as a dependency of the run, and
//! 5. guards the dependencies of every span it used (and the base model), revalidates its
//!    secondary pins before every token and insert, and releases every pin when it finishes,
//!    is cancelled, restarts or is dropped. A restart is cold: no lookup, no insert.
//!
//! The result is reported in [`ReferenceKvOutcome::positional`]. Two controls exist only so
//! an experiment can show its checks can fail ([`PositionalControl`]): a `PostRope` stack, and
//! a mode that bypasses the index's use-time liveness. Neither is a serving path.
//!
//! # Scope
//!
//! Liveness is asked of a [`KvLivenessOracle`] the caller passes to every session step;
//! `PtrRuntime` implements it from the committed lifecycle state, and
//! [`ReferenceKvReuseBackend::own_liveness`] answers from this backend's own revocation
//! set ([`ReferenceKvReuseBackend::invalidate_inputs`],
//! [`ReferenceKvReuseBackend::supersede_inputs`]) as the fallback. The one-shot entry
//! points use the fallback; nothing wires a one-shot run to the ledger. This is an
//! experiment skeleton, not an R005 result.

use crate::kv_infer_layered::{BindingError, BindingRecord, LayeredReuseReport, StackBinding};
use crate::kv_infer_session::{KvSession, KvSessionFinish, KvSessionStats};
use crate::{
    CatalogLimits, KvCatalogError, KvCatalogStats, KvChainId, KvDependency, KvDependencyKind,
    KvLivenessOracle, KvPin, KvPrefixCatalog, KvRestoreReport, KvSalt, KvSnapshotError,
    SecondaryConfig, SecondaryError, SecondaryIndex, SecondaryRefusalCounts, SecondaryStats,
    KV_CHUNK_TOKENS,
};
use ptr_ledger::{InputInvalidationKind, LedgerEvent};
use ptr_model_api::{
    ByteTokenizer, InferenceBackend, KvReuseHint, KvReusePolicy, KvReusingBackend, ModelError,
    ModelEvent, ModelRequest, ModelRun, PromptSegment, SegmentedModelRequest, TokenContractError,
    TokenId, Tokenizer, TokenizerIdentity,
};
use ptr_pods::{
    derive_kv_layer_chains_in_regime, InMemoryPagedKvBackend, KvBackendError, KvChainError,
    KvChainIdentity, KvKeyEncoding, KvPageBinding, KvPageSize, KvPrefixKey, KvTensorDType,
    KvTensorSchema, PagedKvTensorBackend, Qwen2Activation, Qwen2LayerConfig, Qwen2PagedStack,
    Qwen2QkNorm, Qwen2RopeRegime, Qwen2RopeScaling, Qwen2StackConfig, Qwen2StackError,
};
use ptr_types::{
    AdapterVersion, DeviceId, Digest, Generation, ModelVersion, PrincipalId, ProjectId,
};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

pub(crate) const LOGITS_DOMAIN: &[u8] = b"ptr.kv.reference-logits.v1\0";
const MANIFEST_DOMAIN: &[u8] = b"ptr.kv.reference-manifest.v1\0";

/// R005 `weight_seed`.
pub const REFERENCE_KV_WEIGHT_SEED: u64 = 20240607;
const MODEL_NAME: &str = "r005-tiny";
pub(crate) const BASE_ADAPTER_NAME: &str = "base";
const DEVICE_NAME: &str = "cpu";
const NUMERICS: &str = "cpu-f32-sequential-reference";
const SCHEMA_NAME: &str = "ptr-reference-kv-v1";
const BACKEND_NAME: &str = "reference-kv-reuse";

/// Bounds of one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceKvLimits {
    /// Longest prompt, in tokens, a request may carry.
    pub max_prompt_tokens: usize,
    /// Tokens generated per request by greedy decoding.
    pub max_new_tokens: usize,
    /// Most [`KvSession`]s open at once on one backend. One more is refused with
    /// [`ReferenceKvError::SessionLimit`]; the one-shot entry points are serialized by a
    /// gate and do not count.
    pub max_sessions: usize,
    pub catalog: CatalogLimits,
}

impl Default for ReferenceKvLimits {
    fn default() -> Self {
        Self {
            max_prompt_tokens: 256,
            max_new_tokens: 8,
            max_sessions: 8,
            catalog: CatalogLimits::default(),
        }
    }
}

/// Whose cache namespace a request runs in. Folded into the catalog salt and the KV
/// page binding, so two scopes never share an entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReferenceKvScope {
    pub principal: PrincipalId,
    pub project: ProjectId,
}

/// Explicit fault configuration of one backend, for the R005 lag arm. The default is no
/// fault.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReferenceKvFaults {
    pub index_update: IndexUpdate,
}

/// Whether [`ReferenceKvReuseBackend::invalidate_inputs`] and
/// [`ReferenceKvReuseBackend::supersede_inputs`] update the catalog index.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IndexUpdate {
    /// Tombstone and retire the affected entries (production behaviour).
    #[default]
    Applied,
    /// Record the revocation in the backend's liveness set only. The catalog index stays
    /// stale, so only the use-time liveness check keeps a stale entry from being served
    /// (doc 36: invalidation is an optimisation, not the safety mechanism).
    Suppressed,
}

/// Typed failures of the reference backend.
#[derive(Debug)]
pub enum ReferenceKvError {
    InvalidLimits,
    InvalidScope,
    /// The request was tokenized by something other than `ptr.byte@1`.
    TokenizerMismatch {
        found: TokenizerIdentity,
    },
    PromptTooLong {
        tokens: usize,
        maximum: usize,
    },
    Stack(Qwen2StackError),
    Chain(KvChainError),
    Kv(KvBackendError),
    /// A catalog refusal that is not a graceful insert miss (a bug in this backend).
    Catalog(KvCatalogError),
    Contract(TokenContractError),
    NonFiniteLogits,
    /// The state lock was poisoned by a panic in an earlier request.
    Poisoned,
    /// [`ReferenceKvLimits::max_sessions`] sessions are already open.
    SessionLimit {
        maximum: usize,
    },
    /// A session step was called in a phase that does not allow it.
    SessionPhase {
        operation: &'static str,
        phase: crate::kv_infer_session::KvSessionPhase,
    },
    Snapshot(KvSnapshotError),
    /// A [`StackBinding`] was refused (see [`BindingError`]).
    Binding(BindingError),
    /// [`KvSession::restart_with`] was cancelled by a revocation or fence on every one of its
    /// attempts. Nothing produced by those attempts was returned.
    RestartCancelled {
        attempts: usize,
        last: crate::kv_infer_session::KvCancellation,
    },
    /// [`ReferenceKvReuseBackend::apply_input_invalidation`] was handed an event that is not an
    /// `InputInvalidated`.
    NotAnInputInvalidation,
    /// [`ReferenceKvReuseBackend::apply_committed_event`] was handed an `InputInvalidated`,
    /// which names no experts: the fence set is not in the event, so applying it without the
    /// fences would leave the fenced experts' entries live. Use `apply_input_invalidation`.
    InputInvalidationNeedsFences,
    /// [`ReferenceKvReuseBackend::apply_authority_change`] was handed an event that is not an
    /// `ExpertAuthorityChanged`.
    NotAnAuthorityChange,
    /// [`ReferenceKvReuseBackend::apply_committed_event`] was handed an `ExpertAuthorityChanged`,
    /// which names no experts: the fence set is not in the event, so applying it without the
    /// fences would leave the fenced experts' entries live. Use `apply_authority_change`.
    AuthorityChangeNeedsFences,
    /// The secondary (positional) index refused an operation that is not a graceful insert
    /// miss.
    Secondary(SecondaryError),
    /// A positional session or probe was requested on a backend that was not built with
    /// [`ReferenceKvReuseBackend::with_positional_reuse`].
    PositionalNotEnabled,
    /// Positional reuse is sound only for a one-layer stack: the layer-0 K/V rows of a token
    /// are a pure function of the token, while the rows of a deeper layer depend on the
    /// context and the positions before them.
    PositionalNeedsOneLayer {
        layers: usize,
    },
    /// Positional reuse needs `PreRope` keys, rotated at use at the row's new position; a
    /// `PostRope` stack is only accepted as the explicit negative control
    /// ([`PositionalControl::PermitPostRopeKeys`]).
    PositionalNeedsPreRope,
}

impl fmt::Display for ReferenceKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => write!(f, "reference KV limits must be non-zero"),
            Self::InvalidScope => {
                write!(f, "reference KV scope names an empty principal or project")
            }
            Self::TokenizerMismatch { found } => {
                write!(f, "request tokenizer {found} is not ptr.byte@1")
            }
            Self::PromptTooLong { tokens, maximum } => {
                write!(
                    f,
                    "prompt of {tokens} tokens exceeds the {maximum} token limit"
                )
            }
            Self::Stack(error) => write!(f, "reference stack: {error}"),
            Self::Chain(error) => write!(f, "kv chain: {error}"),
            Self::Kv(error) => write!(f, "paged KV backend: {error:?}"),
            Self::Catalog(error) => write!(f, "prefix catalog: {error}"),
            Self::Contract(error) => write!(f, "token contract: {error}"),
            Self::NonFiniteLogits => write!(f, "the reference stack produced non-finite logits"),
            Self::Poisoned => write!(f, "reference KV state lock poisoned"),
            Self::SessionLimit { maximum } => {
                write!(f, "{maximum} reference KV sessions are already open")
            }
            Self::SessionPhase { operation, phase } => {
                write!(f, "{operation} is not allowed in session phase {phase:?}")
            }
            Self::Snapshot(error) => write!(f, "catalog snapshot: {error}"),
            Self::Binding(error) => write!(f, "stack binding: {error}"),
            Self::RestartCancelled { attempts, last } => write!(
                f,
                "restart cancelled on all {attempts} attempts (last: {:?})",
                last.reason
            ),
            Self::NotAnInputInvalidation => {
                write!(f, "the event is not an input invalidation")
            }
            Self::InputInvalidationNeedsFences => write!(
                f,
                "an input invalidation names no experts; apply it with its fence epochs"
            ),
            Self::NotAnAuthorityChange => {
                write!(f, "the event is not an expert authority change")
            }
            Self::AuthorityChangeNeedsFences => write!(
                f,
                "an authority change names no experts; apply it with its fence epochs"
            ),
            Self::Secondary(error) => write!(f, "secondary index: {error}"),
            Self::PositionalNotEnabled => {
                write!(f, "positional reuse is not enabled on this backend")
            }
            Self::PositionalNeedsOneLayer { layers } => write!(
                f,
                "positional reuse needs a one-layer stack, this one has {layers} layers"
            ),
            Self::PositionalNeedsPreRope => {
                write!(f, "positional reuse needs PreRope key storage")
            }
        }
    }
}

impl std::error::Error for ReferenceKvError {}

impl From<Qwen2StackError> for ReferenceKvError {
    fn from(error: Qwen2StackError) -> Self {
        Self::Stack(error)
    }
}

impl From<KvChainError> for ReferenceKvError {
    fn from(error: KvChainError) -> Self {
        Self::Chain(error)
    }
}

impl From<KvBackendError> for ReferenceKvError {
    fn from(error: KvBackendError) -> Self {
        Self::Kv(error)
    }
}

impl From<SecondaryError> for ReferenceKvError {
    fn from(error: SecondaryError) -> Self {
        Self::Secondary(error)
    }
}

impl From<TokenContractError> for ReferenceKvError {
    fn from(error: TokenContractError) -> Self {
        Self::Contract(error)
    }
}

impl From<ReferenceKvError> for ModelError {
    fn from(error: ReferenceKvError) -> Self {
        ModelError(error.to_string())
    }
}

/// Everything one run produced, beyond the [`ModelRun`] the trait returns.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceKvOutcome {
    pub run: ModelRun,
    /// The greedily generated tokens.
    pub generated: Vec<TokenId>,
    /// SHA-256 over the bit patterns of the logits vector of every decode step (the vector
    /// each generated token was chosen from), in order. Equal digests mean bitwise equal
    /// logits.
    pub logits_digest: Digest,
    pub prompt_tokens: usize,
    /// The effective RoPE regime fixed at admission and used for the whole request.
    pub rope_regime: Qwen2RopeRegime,
    /// Prompt tokens served from the catalog.
    pub hit_tokens: usize,
    /// Prompt tokens actually run through the stack.
    pub prefilled_tokens: usize,
    /// Chunk boundaries newly offered to and accepted by the catalog.
    pub inserted_chunks: usize,
    /// Chunk boundaries the catalog declined (capacity, draining slot, dead dependency).
    pub insert_refusals: usize,
    /// Why each declined boundary was declined, in order (`CapacityExhausted` is the
    /// saturation behaviour: the run continues uncached and bitwise equal to cold).
    pub insert_refusal_reasons: Vec<KvCatalogError>,
    /// The fork was cancelled by an invalidation and the run restarted from a fresh prefill.
    pub restarted_after_cancel: bool,
    /// The binding the run executed under: the request's own, or for a restart the
    /// re-resolved one (see [`BindingRecord::re_resolved`]).
    pub binding: BindingRecord,
    /// Layer-scoped reuse accounting of a bound run; `None` for the default binding.
    pub layered: Option<LayeredReuseReport>,
    /// Positional (secondary index) accounting; `None` unless the run was a positional
    /// session. `prompt_tokens == hit_tokens + positional.reused_tokens + prefilled_tokens`
    /// for a positional run that was not restarted.
    pub positional: Option<PositionalReport>,
}

/// Which stack a backend builds. The default is the R005 stack (4 layers, `PreRope`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceKvStackShape {
    /// Decoder layers; R006 uses 1.
    pub layers: usize,
    /// Where RoPE is applied. `PostRope` exists only as R006's negative control: its stored
    /// keys are rotated at the position they were computed at.
    pub key_encoding: KvKeyEncoding,
}

impl Default for ReferenceKvStackShape {
    fn default() -> Self {
        Self {
            layers: DEFAULT_LAYERS,
            key_encoding: KvKeyEncoding::PreRope,
        }
    }
}

impl ReferenceKvStackShape {
    /// The R005 shape with `layers` layers and `PreRope` keys.
    pub fn with_layers(layers: usize) -> Self {
        Self {
            layers,
            key_encoding: KvKeyEncoding::PreRope,
        }
    }
}

/// A negative control or fault of the positional path. Everything but `None` is **not a
/// serving path**: it exists so a harness can show its counters can trip.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PositionalControl {
    /// The serving behaviour.
    #[default]
    None,
    /// Use-time liveness is bypassed for the secondary index: the lookup asks an always-live
    /// oracle, the session adds no guard for a spliced run's dependencies and does not
    /// revalidate its pins against the oracle (an already tombstoned pin still cancels). A
    /// dead run whose invalidation never reached the index (lag) is then admitted and
    /// consumed, which is what the structural counters must detect.
    BypassUseTimeLiveness,
    /// Accept a `PostRope` stack. Its stored keys are rotated at the old position, so a
    /// spliced span is **not** bit-equal to a fresh prefill: the control that shows the
    /// bitwise comparison can fail.
    PermitPostRopeKeys,
}

/// Opt-in configuration of positional reuse on one backend.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PositionalOptions {
    /// The secondary index: rolling window and bounds.
    pub secondary: SecondaryConfig,
    pub control: PositionalControl,
}

/// What a positional session did with the secondary index.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PositionalReport {
    /// Positional lookups made at begin (0 after a cold restart, else 1).
    pub lookups: usize,
    /// Spans the lookup admitted.
    pub admitted_splices: usize,
    /// Admitted spans dropped because they start below the primary catalog's hit (their
    /// pins were released at once).
    pub dropped_below_primary: usize,
    /// Spans repacked into the cache at their new positions.
    pub splices: usize,
    /// Prompt tokens whose stored rows were repacked instead of recomputed.
    pub reused_tokens: usize,
    /// Refused candidates by reason (see [`crate::SecondaryRefusalReason`]).
    pub refusals: SecondaryRefusalCounts,
    /// Segments whose only admitted run would have covered the last prompt token.
    pub last_token_skips: usize,
    /// Segments newly offered to the index.
    pub inserted_segments: usize,
    /// Segments already present with the same rows.
    pub already_present_segments: usize,
    /// Restored row-less runs that got their rows back.
    pub rows_attached: usize,
    /// Segments the index declined (capacity, a draining slot, a dependency not live).
    pub insert_refusals: Vec<SecondaryError>,
    /// The session ran with [`PositionalControl::BypassUseTimeLiveness`].
    pub liveness_bypassed: bool,
    /// Under the bypass: admitted spans with a dependency the real oracle called dead. Zero
    /// without the bypass, where such a span is refused instead.
    pub bypass_dead_admissions: usize,
    /// The session was cancelled and restarted cold: nothing above survives except this flag.
    pub cold_restart: bool,
}

/// How [`ReferenceKvReuseBackend::probe`] runs a request.
#[derive(Clone, Copy, Debug)]
pub enum ReferenceKvProbeMode<'a> {
    /// No lookup and no insert: the cold path.
    Cold,
    /// Look up and insert as [`ReferenceKvReuseBackend::infer_scoped`] does.
    Warm,
    /// Negative control, **not a serving path**. The catalog is asked for `donor`'s prefix
    /// under an always-live oracle (the use-time liveness check is bypassed on purpose), the
    /// forked rows are cut to the longest 16-token boundary below the prompt and served as the
    /// prompt's own rows, and the rest is prefilled. If the donor's prefix holds rows that are
    /// not the prompt's (a span the prompt no longer has), the result differs from a cold run
    /// of the prompt, which is what a bitwise check must detect. The run inserts nothing. The
    /// catalog still has to hold the donor's entries, so revoke the donor's input under
    /// [`IndexUpdate::Suppressed`] (or not at all) before using this mode.
    StaleDonor(&'a SegmentedModelRequest),
}

/// A run plus the two bitwise probes of the R007 span-forget experiment.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceKvProbe {
    pub outcome: ReferenceKvOutcome,
    /// The full next-token logits vector (256 entries) at the last prompt position, before
    /// the first generated token. Compare on `f32::to_bits`; see
    /// [`ReferenceKvProbe::logit_bits`].
    pub prompt_logits: Vec<f32>,
    /// Digest of the K and V rows of every layer for all prompt positions; the definition is
    /// on [`KvSession::kv_state_digest`].
    pub kv_state_digest: Digest,
}

impl ReferenceKvProbe {
    /// The logits as `f32` bit patterns, the unit of the bitwise comparison.
    pub fn logit_bits(&self) -> Vec<u32> {
        self.prompt_logits
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }
}

/// Whether a run may look up and insert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CacheUse {
    Disabled,
    Enabled,
}

/// Liveness as this backend knows it: an input is live unless it was revoked or its
/// generation is below the floor a supersession set for its identity.
#[derive(Debug, Default)]
pub(crate) struct RevokedInputs {
    revoked: BTreeSet<(String, Generation)>,
    floors: BTreeMap<String, Generation>,
    /// Expert identities with a committed fence epoch: an expert dependency recorded below it
    /// (or with none) is not live.
    fences: BTreeMap<String, u64>,
    /// Expert identities a committed structural record named: dead at every generation.
    dead_experts: BTreeSet<String>,
    /// A structural record that could not be decoded: every expert dependency is dead.
    all_experts_dead: bool,
}

impl RevokedInputs {
    /// Record what a committed event makes not live, mirroring the catalog mapping of
    /// `invalidate_catalog_for_event`.
    fn note_event(&mut self, event: &LedgerEvent) {
        match event {
            LedgerEvent::Revoked {
                subject,
                generation,
            } => {
                self.revoked.insert((subject.clone(), *generation));
            }
            LedgerEvent::ProcedureRevoked { id, generation } => {
                self.revoked
                    .insert((format!("procedure:{id}"), *generation));
            }
            LedgerEvent::CapsuleCommitted {
                capsule,
                generation,
                ..
            } => self.raise_floor(capsule.to_string(), *generation),
            LedgerEvent::CapsuleSuperseded { capsule, new, .. } => {
                self.raise_floor(capsule.to_string(), *new);
            }
            LedgerEvent::HardConstraintCommitted { key, generation } => {
                self.raise_floor(format!("constraint:{key}"), *generation);
            }
            LedgerEvent::ProcedurePromoted { id, generation } => {
                self.raise_floor(format!("procedure:{id}"), *generation);
            }
            LedgerEvent::ExpertStructuralCommitted { record, .. } => {
                match crate::kv_prefix::expert_operand_identities(record) {
                    Some(identities) => self.dead_experts.extend(identities),
                    None => self.all_experts_dead = true,
                }
            }
            LedgerEvent::ExpertFenceCommitted { .. } => {
                self.note_fences(crate::expert::expert_fence_epochs(event));
            }
            // The input half of a compound invalidation; the experts it fences are not in the
            // event and arrive through `note_fences` (see `apply_input_invalidation`).
            LedgerEvent::InputInvalidated {
                input,
                generation,
                kind: InputInvalidationKind::Revoked,
                ..
            } => {
                self.revoked.insert((input.clone(), *generation));
            }
            LedgerEvent::InputInvalidated {
                input,
                kind: InputInvalidationKind::Superseded { new_generation, .. },
                ..
            } => self.raise_floor(input.clone(), *new_generation),
            _ => {}
        }
    }

    /// Record the fence epoch now in force per expert identity.
    fn note_fences(&mut self, epochs: BTreeMap<String, u64>) {
        for (identity, epoch) in epochs {
            let known = self.fences.entry(identity).or_insert(epoch);
            *known = (*known).max(epoch);
        }
    }

    fn raise_floor(&mut self, identity: String, generation: Generation) {
        let floor = self.floors.entry(identity).or_insert(generation);
        *floor = (*floor).max(generation);
    }

    fn is_live(&self, dependency: &KvDependency) -> bool {
        let expert_dead = dependency.kind == KvDependencyKind::Expert
            && (self.all_experts_dead
                || self.dead_experts.contains(&dependency.identity)
                || self
                    .fences
                    .get(&dependency.identity)
                    .is_some_and(|epoch| dependency.fence_epoch < *epoch));
        !expert_dead
            && !self
                .revoked
                .contains(&(dependency.identity.clone(), dependency.generation))
            && self
                .floors
                .get(&dependency.identity)
                .is_none_or(|floor| dependency.generation >= *floor)
    }
}

/// The backend's own liveness answer, the fallback oracle when no ledger-anchored one is
/// passed. A clone shares the same set, so it sees revocations made after it was taken. A
/// poisoned set answers "not live" (fail closed). It has its own lock, so it can be asked
/// while a catalog operation holds the state lock.
#[derive(Clone, Debug, Default)]
pub struct ReferenceKvLiveness {
    inner: Arc<Mutex<RevokedInputs>>,
}

impl KvLivenessOracle for ReferenceKvLiveness {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        self.inner
            .lock()
            .is_ok_and(|inputs| inputs.is_live(dependency))
    }
}

pub(crate) struct ReuseState {
    pub(crate) catalog: KvPrefixCatalog,
    /// Backend prefix keys some catalog entry holds or held and whose retirement the
    /// catalog has not yet reported; a refused insert must not retire these.
    pub(crate) held: BTreeSet<KvPrefixKey>,
    pub(crate) faults: ReferenceKvFaults,
    pub(crate) stats: KvSessionStats,
    /// Pins of the open sessions, by session id.
    pub(crate) pins: BTreeMap<u64, Vec<KvPin>>,
    pub(crate) open_sessions: usize,
    pub(crate) next_session: u64,
}

pub(crate) type ReferenceCache = <InMemoryPagedKvBackend as ptr_pods::KvTensorBackend>::Cache;

/// See the module documentation.
pub struct ReferenceKvReuseBackend {
    pub(crate) stack: Qwen2PagedStack,
    tokenizer: ByteTokenizer,
    pub(crate) paged: InMemoryPagedKvBackend,
    chain: KvChainId,
    build: String,
    schema: KvTensorSchema,
    limits: ReferenceKvLimits,
    default_scope: ReferenceKvScope,
    pub(crate) state: Mutex<ReuseState>,
    liveness: ReferenceKvLiveness,
    /// Serializes the one-shot entry points, as the former whole-request lock did.
    oneshot: Mutex<()>,
    /// `Some` when positional reuse was opted into; `None` leaves every path as before.
    pub(crate) positional: Option<PositionalOptions>,
}

impl fmt::Debug for ReferenceKvReuseBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReferenceKvReuseBackend")
            .field("limits", &self.limits)
            .field("default_scope", &self.default_scope)
            .finish_non_exhaustive()
    }
}

/// The R005 layer count, kept for every backend that does not ask for another.
const DEFAULT_LAYERS: usize = 4;

#[cfg(test)]
fn stack_config(limits: &ReferenceKvLimits, rope_scaling: Qwen2RopeScaling) -> Qwen2StackConfig {
    stack_config_for(limits, rope_scaling, ReferenceKvStackShape::default())
}

fn stack_config_for(
    limits: &ReferenceKvLimits,
    rope_scaling: Qwen2RopeScaling,
    shape: ReferenceKvStackShape,
) -> Qwen2StackConfig {
    // R005 tiny_model_config: hidden 64, 4 heads, 2 KV heads, head_dim 16, vocab 256, and
    // 4 layers unless the shape says otherwise (R006 uses 1).
    Qwen2StackConfig {
        layer: Qwen2LayerConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: limits.max_prompt_tokens + limits.max_new_tokens,
            rope_theta: 10_000.0,
            rope_scaling,
            rms_norm_eps: 1e-6,
            hidden_act: Qwen2Activation::Silu,
            sliding_window: None,
        },
        num_layers: shape.layers,
        vocab_size: 256,
        key_encoding: shape.key_encoding,
        qk_norm: Qwen2QkNorm::Disabled,
    }
}

/// Backend build identity folded into the chain root and the catalog salt: engine, version,
/// numerics and producer platform (doc 36: Exact numerics bind the producer environment).
fn build_identity() -> String {
    format!(
        "{}@{}|{NUMERICS}|{}-{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

/// `chain(L-1)` of the actual stack weights and configuration under the RoPE regime
/// fixed at admission (doc 36 section D).
fn derive_chain(
    stack: &Qwen2PagedStack,
    build: &str,
    tokenizer: &TokenizerIdentity,
    regime: &Qwen2RopeRegime,
) -> Result<KvChainId, ReferenceKvError> {
    let identity = KvChainIdentity {
        base_model: MODEL_NAME.to_owned(),
        generation: 1,
        build: build.to_owned(),
        tokenizer: format!("{}@{}", tokenizer.name(), tokenizer.version()),
    };
    Ok(KvChainId(
        derive_kv_layer_chains_in_regime(stack, &identity, regime)?.final_id(),
    ))
}

impl ReferenceKvReuseBackend {
    /// A backend with the R005 reference weights (`REFERENCE_KV_WEIGHT_SEED`).
    pub fn new(
        limits: ReferenceKvLimits,
        default_scope: ReferenceKvScope,
    ) -> Result<Self, ReferenceKvError> {
        Self::with_seed(limits, default_scope, REFERENCE_KV_WEIGHT_SEED)
    }

    pub fn with_seed(
        limits: ReferenceKvLimits,
        default_scope: ReferenceKvScope,
        weight_seed: u64,
    ) -> Result<Self, ReferenceKvError> {
        Self::with_rope_scaling(limits, default_scope, weight_seed, Qwen2RopeScaling::None)
    }

    /// Like [`Self::with_seed`] with a RoPE scaling scheme. With
    /// [`Qwen2RopeScaling::DynamicNtk`] every request fixes its regime at admission (see the
    /// module documentation).
    pub fn with_rope_scaling(
        limits: ReferenceKvLimits,
        default_scope: ReferenceKvScope,
        weight_seed: u64,
        rope_scaling: Qwen2RopeScaling,
    ) -> Result<Self, ReferenceKvError> {
        Self::with_stack_shape(
            limits,
            default_scope,
            weight_seed,
            rope_scaling,
            ReferenceKvStackShape::default(),
        )
    }

    /// Like [`Self::with_seed`] with `layers` decoder layers (R006 uses 1; the default
    /// constructors keep 4). The same seed gives the same embedding and layer-0 weights at any
    /// depth, but the final norm, the head and therefore the chain id differ. Nothing else
    /// changes: a 4-layer backend built here is identical to [`Self::with_seed`].
    pub fn with_layers(
        limits: ReferenceKvLimits,
        default_scope: ReferenceKvScope,
        weight_seed: u64,
        layers: usize,
    ) -> Result<Self, ReferenceKvError> {
        Self::with_stack_shape(
            limits,
            default_scope,
            weight_seed,
            Qwen2RopeScaling::None,
            ReferenceKvStackShape::with_layers(layers),
        )
    }

    /// The general constructor: RoPE scaling and the whole [`ReferenceKvStackShape`], including
    /// a `PostRope` stack for R006's negative control.
    pub fn with_stack_shape(
        limits: ReferenceKvLimits,
        default_scope: ReferenceKvScope,
        weight_seed: u64,
        rope_scaling: Qwen2RopeScaling,
        shape: ReferenceKvStackShape,
    ) -> Result<Self, ReferenceKvError> {
        if limits.max_prompt_tokens == 0 || limits.max_new_tokens == 0 || limits.max_sessions == 0 {
            return Err(ReferenceKvError::InvalidLimits);
        }
        check_scope(&default_scope)?;
        let config = stack_config_for(&limits, rope_scaling, shape);
        let stack = Qwen2PagedStack::reference(config, weight_seed)?;
        let paged = InMemoryPagedKvBackend::new(KvPageSize::new(KV_CHUNK_TOKENS as u16)?);
        let schema = stack.kv_schema(
            ModelVersion::from(MODEL_NAME),
            AdapterVersion::from(BASE_ADAPTER_NAME),
            DeviceId::from(DEVICE_NAME),
        )?;
        let tokenizer = ByteTokenizer::new();
        let build = build_identity();
        // The chain of the plain-base regime; requests derive their own at admission.
        let plain = stack
            .config()
            .layer
            .rope_regime(1)
            .map_err(Qwen2StackError::InvalidLayer)?;
        let chain = derive_chain(&stack, &build, tokenizer.identity(), &plain)?;
        let authority = paged.take_fork_authority()?;
        let catalog = KvPrefixCatalog::new(authority, limits.catalog)
            .map_err(ReferenceKvError::Catalog)?
            .with_retirer(Arc::new(paged.clone()));
        Ok(Self {
            stack,
            tokenizer,
            paged,
            chain,
            build,
            schema,
            limits,
            default_scope,
            state: Mutex::new(ReuseState {
                catalog,
                held: BTreeSet::new(),
                faults: ReferenceKvFaults::default(),
                stats: KvSessionStats::default(),
                pins: BTreeMap::new(),
                open_sessions: 0,
                next_session: 0,
            }),
            liveness: ReferenceKvLiveness::default(),
            oneshot: Mutex::new(()),
            positional: None,
        })
    }

    /// Opt into positional reuse: the catalog's secondary index is rebuilt from
    /// `options.secondary` (it is empty at construction) and
    /// [`Self::begin_session_positional`] and [`Self::probe_positional`] become available.
    /// Every other entry point behaves exactly as without it: no secondary lookup and no
    /// secondary insert. Chain ids, pinned digests and outputs of existing configurations do
    /// not change.
    ///
    /// # Errors
    /// [`ReferenceKvError::PositionalNeedsOneLayer`] unless the stack has exactly one layer,
    /// [`ReferenceKvError::PositionalNeedsPreRope`] for a `PostRope` stack (unless the
    /// options carry [`PositionalControl::PermitPostRopeKeys`]) and
    /// [`ReferenceKvError::Secondary`] for an invalid index configuration.
    pub fn with_positional_reuse(
        mut self,
        options: PositionalOptions,
    ) -> Result<Self, ReferenceKvError> {
        let config = self.stack.config();
        if config.num_layers != 1 {
            return Err(ReferenceKvError::PositionalNeedsOneLayer {
                layers: config.num_layers,
            });
        }
        if config.key_encoding != KvKeyEncoding::PreRope
            && options.control != PositionalControl::PermitPostRopeKeys
        {
            return Err(ReferenceKvError::PositionalNeedsPreRope);
        }
        let index = SecondaryIndex::new(options.secondary)?;
        {
            let mut state = self.lock()?;
            *state.catalog.secondary_mut() = index;
        }
        self.positional = Some(options);
        Ok(self)
    }

    /// Counters of the catalog's secondary index (lookups, hits, refusals per reason,
    /// invalidations, evictions).
    pub fn secondary_stats(&self) -> Result<SecondaryStats, ReferenceKvError> {
        Ok(self.lock()?.catalog.secondary().stats())
    }

    /// The base-model dependency a positional session records: the lifecycle identity of the
    /// reference model, generation 1 and the session's chain as its content digest (as a
    /// bound session records it).
    pub(crate) fn base_model_dependency(&self, chain: &KvChainId) -> KvDependency {
        KvDependency::base_model(MODEL_NAME, Generation(1), chain.0)
    }

    pub fn limits(&self) -> ReferenceKvLimits {
        self.limits
    }

    /// The tiny reference stack's configuration, for building adapters and MoE layers that
    /// fit it.
    pub fn stack_config(&self) -> &Qwen2StackConfig {
        self.stack.config()
    }

    pub fn tokenizer(&self) -> &ByteTokenizer {
        &self.tokenizer
    }

    /// The chain id of the plain-base RoPE regime: every entry of a backend without
    /// scaling is stored under it. With dynamic NTK each request's chain also follows its
    /// admission regime, see [`Self::chain_id_for_admission`].
    pub fn chain_id(&self) -> KvChainId {
        self.chain
    }

    /// The chain id a request of `prompt_tokens` tokens is stored and looked up under
    /// (the regime is fixed from `prompt_tokens + max_new_tokens`).
    pub fn chain_id_for_admission(
        &self,
        prompt_tokens: usize,
    ) -> Result<KvChainId, ReferenceKvError> {
        let regime = self.admit_regime(prompt_tokens)?;
        self.chain_for(&regime)
    }

    /// The RoPE regime a request of `prompt_tokens` tokens is admitted under.
    pub fn admit_regime(&self, prompt_tokens: usize) -> Result<Qwen2RopeRegime, ReferenceKvError> {
        Ok(self
            .stack
            .admit_rope_regime(prompt_tokens, self.limits.max_new_tokens)?)
    }

    pub(crate) fn chain_for(
        &self,
        regime: &Qwen2RopeRegime,
    ) -> Result<KvChainId, ReferenceKvError> {
        derive_chain(&self.stack, &self.build, self.tokenizer.identity(), regime)
    }

    fn manifest_for(chain: &KvChainId) -> Digest {
        let mut hasher = Sha256::new();
        hasher.update(MANIFEST_DOMAIN);
        hasher.update(chain.0);
        hasher.finalize().into()
    }

    pub fn catalog_stats(&self) -> Result<KvCatalogStats, ReferenceKvError> {
        Ok(self.lock()?.catalog.stats())
    }

    pub(crate) fn lock(&self) -> Result<std::sync::MutexGuard<'_, ReuseState>, ReferenceKvError> {
        self.state.lock().map_err(|_| ReferenceKvError::Poisoned)
    }

    /// Tokenize `text` as one Raw segment and run it in `scope` with reuse.
    pub fn infer_text(
        &self,
        scope: &ReferenceKvScope,
        request_id: ptr_types::RequestId,
        semantic_context: ptr_types::SemanticContext,
        text: &str,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        let request = self.text_request(request_id, semantic_context, text)?;
        self.infer_scoped(scope, &request, None)
    }

    /// Like [`Self::infer_text`] with no lookup and no insert.
    pub fn infer_text_cold(
        &self,
        scope: &ReferenceKvScope,
        request_id: ptr_types::RequestId,
        semantic_context: ptr_types::SemanticContext,
        text: &str,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        let request = self.text_request(request_id, semantic_context, text)?;
        self.infer_cold(scope, &request)
    }

    fn text_request(
        &self,
        request_id: ptr_types::RequestId,
        semantic_context: ptr_types::SemanticContext,
        text: &str,
    ) -> Result<SegmentedModelRequest, ReferenceKvError> {
        let segment = PromptSegment::raw_from_text(&self.tokenizer, text)?;
        Ok(SegmentedModelRequest::new(
            request_id,
            semantic_context,
            self.tokenizer.identity().clone(),
            vec![segment],
        )?)
    }

    /// Run `request` with reuse inside `scope`. The trait entry point uses the backend's
    /// default scope.
    ///
    /// `hint.requested_prefix_tokens` is advisory and not used to limit the lookup; the
    /// hint's policy cannot change the result because this backend never reports
    /// Approximate reuse.
    pub fn infer_scoped(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        let policy = hint.map_or(KvReusePolicy::ExactOnly, |hint| hint.policy);
        self.generate(
            scope,
            request,
            policy,
            CacheUse::Enabled,
            &StackBinding::base(),
        )
    }

    /// Like [`Self::infer_scoped`] on a [`StackBinding`] (one shot: a revocation that
    /// cancels the run restarts it cold on the re-resolved binding).
    pub fn infer_scoped_bound(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
        binding: &StackBinding,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        let policy = hint.map_or(KvReusePolicy::ExactOnly, |hint| hint.policy);
        self.generate(scope, request, policy, CacheUse::Enabled, binding)
    }

    /// Like [`Self::infer_cold`] on a [`StackBinding`]: no lookup and no insert, and the
    /// binding is **not checked for liveness**, neither at begin nor during the run (a cold
    /// run depends on no cached state, so a revoked adapter or expert is the caller's policy
    /// question here and the run executes on exactly the binding passed). Use
    /// [`Self::infer_scoped_bound`] or a session for a guarded run. A
    /// batch-variant binding is reported Approximate and needs
    /// [`KvReusePolicy::AllowApproximate`] in `hint`.
    pub fn infer_cold_bound(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
        binding: &StackBinding,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        let policy = hint.map_or(KvReusePolicy::ExactOnly, |hint| hint.policy);
        self.generate(scope, request, policy, CacheUse::Disabled, binding)
    }

    /// Run `request` like the one-shot entry points (same gate, same own-liveness oracle, same
    /// restart on a mid-run cancellation) and also return the full prompt-position logits
    /// vector and the KV-state digest of the run, for the cold path, the warm path or the
    /// stale-reuse negative control (see [`ReferenceKvProbeMode`]). Always the default binding
    /// and Exact reuse.
    pub fn probe(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        mode: ReferenceKvProbeMode<'_>,
    ) -> Result<ReferenceKvProbe, ReferenceKvError> {
        let _gate = self
            .oneshot
            .lock()
            .map_err(|_| ReferenceKvError::Poisoned)?;
        let liveness = self.own_liveness();
        let mut session = match mode {
            ReferenceKvProbeMode::Cold => KvSession::begin(
                self,
                scope,
                request,
                KvReusePolicy::ExactOnly,
                CacheUse::Disabled,
                crate::kv_infer_session::SessionAdmission::OneShot,
                &liveness,
            )?,
            ReferenceKvProbeMode::Warm => KvSession::begin(
                self,
                scope,
                request,
                KvReusePolicy::ExactOnly,
                CacheUse::Enabled,
                crate::kv_infer_session::SessionAdmission::OneShot,
                &liveness,
            )?,
            ReferenceKvProbeMode::StaleDonor(donor) => {
                KvSession::begin_stale_donor(self, scope, request, donor, &liveness)?
            }
        };
        let outcome = match session.run_to_end(&liveness)? {
            KvSessionFinish::Completed(outcome) => *outcome,
            KvSessionFinish::Cancelled(_) => session.restart_with(&liveness)?,
        };
        let prompt_logits =
            session
                .prompt_logits()
                .map(<[f32]>::to_vec)
                .ok_or(ReferenceKvError::SessionPhase {
                    operation: "probe",
                    phase: session.phase(),
                })?;
        let kv_state_digest = session.kv_state_digest()?;
        Ok(ReferenceKvProbe {
            outcome,
            prompt_logits,
            kv_state_digest,
        })
    }

    /// Run `request` with no lookup and no insert: the cold path.
    pub fn infer_cold(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        self.generate(
            scope,
            request,
            KvReusePolicy::ExactOnly,
            CacheUse::Disabled,
            &StackBinding::base(),
        )
    }

    /// The backend's own liveness oracle (the fallback when no ledger-anchored one is
    /// passed to a session).
    pub fn own_liveness(&self) -> ReferenceKvLiveness {
        self.liveness.clone()
    }

    /// Replace the fault configuration (the R005 lag arm).
    pub fn set_faults(&self, faults: ReferenceKvFaults) -> Result<(), ReferenceKvError> {
        self.lock()?.faults = faults;
        Ok(())
    }

    /// Counters of the session lifecycle: cancelled and restarted request ids, saturation,
    /// eviction audit (see [`KvSessionStats`]).
    pub fn session_stats(&self) -> Result<KvSessionStats, ReferenceKvError> {
        Ok(self.lock()?.stats.clone())
    }

    /// Open a stepwise session for `request` in `scope`; see [`KvSession`]. `liveness` is
    /// asked at begin and before every emitted token and catalog insert.
    pub fn begin_session<'a>(
        &'a self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvSession<'a>, ReferenceKvError> {
        let policy = hint.map_or(KvReusePolicy::ExactOnly, |hint| hint.policy);
        KvSession::begin(
            self,
            scope,
            request,
            policy,
            CacheUse::Enabled,
            crate::kv_infer_session::SessionAdmission::Counted,
            liveness,
        )
    }

    /// Open a stepwise session on a [`StackBinding`]; see the [`crate::kv_infer_layered`]
    /// module documentation. The binding's adapter and tracked experts are asked of
    /// `liveness` at begin (a dead one is refused with [`BindingError::NotLive`]) and
    /// before every emitted token and catalog insert. The default binding behaves exactly as
    /// [`Self::begin_session`].
    pub fn begin_session_bound<'a>(
        &'a self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
        binding: &StackBinding,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvSession<'a>, ReferenceKvError> {
        let policy = hint.map_or(KvReusePolicy::ExactOnly, |hint| hint.policy);
        KvSession::begin_bound(
            self,
            scope,
            request,
            policy,
            CacheUse::Enabled,
            crate::kv_infer_session::SessionAdmission::Counted,
            binding,
            liveness,
        )
    }

    /// Open a stepwise **positional** session: after the primary catalog's lookup the
    /// secondary index is asked for whole segments found at another offset, admitted spans
    /// are repacked into the cache at their new positions (one step each) and every completed
    /// segment is offered to the index. The spans' dependencies (and the base model) join the
    /// session's guards, and every secondary pin is released when the session finishes,
    /// cancels, restarts or is dropped. Default binding, Exact only.
    ///
    /// # Errors
    /// [`ReferenceKvError::PositionalNotEnabled`] on a backend built without
    /// [`Self::with_positional_reuse`].
    pub fn begin_session_positional<'a>(
        &'a self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvSession<'a>, ReferenceKvError> {
        let policy = hint.map_or(KvReusePolicy::ExactOnly, |hint| hint.policy);
        KvSession::begin_positional(
            self,
            scope,
            request,
            policy,
            crate::kv_infer_session::SessionAdmission::Counted,
            liveness,
        )
    }

    /// Like [`Self::probe`] in its warm mode for a positional session: the same one-shot gate
    /// and restart on a mid-run cancellation, `liveness` as the oracle, and the logits and
    /// KV-state digest of the run. There is deliberately no [`ReferenceKvProbeMode`] variant
    /// for it: the existing probe modes never touch the secondary index.
    pub fn probe_positional(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<ReferenceKvProbe, ReferenceKvError> {
        let _gate = self
            .oneshot
            .lock()
            .map_err(|_| ReferenceKvError::Poisoned)?;
        let mut session = KvSession::begin_positional(
            self,
            scope,
            request,
            KvReusePolicy::ExactOnly,
            crate::kv_infer_session::SessionAdmission::OneShot,
            liveness,
        )?;
        let outcome = match session.run_to_end(liveness)? {
            KvSessionFinish::Completed(outcome) => *outcome,
            KvSessionFinish::Cancelled(_) => session.restart_with(liveness)?,
        };
        let prompt_logits =
            session
                .prompt_logits()
                .map(<[f32]>::to_vec)
                .ok_or(ReferenceKvError::SessionPhase {
                    operation: "probe_positional",
                    phase: session.phase(),
                })?;
        let kv_state_digest = session.kv_state_digest()?;
        Ok(ReferenceKvProbe {
            outcome,
            prompt_logits,
            kv_state_digest,
        })
    }

    /// Record that `(identity, generation)` pairs are no longer live: every entry that
    /// depends on one is tombstoned, its unused witnesses are revoked and the backend
    /// prefixes it held are retired. Later lookups and session steps also see the
    /// revocation at use time through [`Self::own_liveness`].
    ///
    /// With [`IndexUpdate::Suppressed`] only the liveness set changes and `0` is returned:
    /// the catalog index stays stale on purpose.
    pub fn invalidate_inputs(
        &self,
        invalidated: &BTreeSet<(String, Generation)>,
    ) -> Result<usize, ReferenceKvError> {
        let mut state = self.lock()?;
        self.liveness
            .inner
            .lock()
            .map_err(|_| ReferenceKvError::Poisoned)?
            .revoked
            .extend(invalidated.iter().cloned());
        if state.faults.index_update == IndexUpdate::Suppressed {
            state.stats.suppressed_index_updates += 1;
            return Ok(0);
        }
        let keys = state.catalog.invalidate_inputs(invalidated);
        let retired = keys.len();
        for key in &keys {
            self.retire(&mut state, key)?;
        }
        self.drain_retirements(&mut state)?;
        Ok(retired)
    }

    /// Record that every generation of an identity below `live[identity]` is superseded:
    /// the matching entries are tombstoned and retired like [`Self::invalidate_inputs`],
    /// and the liveness set refuses the older generations from now on (never lowered by a
    /// later call). [`IndexUpdate::Suppressed`] applies here too.
    pub fn supersede_inputs(
        &self,
        live: &BTreeMap<String, Generation>,
    ) -> Result<usize, ReferenceKvError> {
        let mut state = self.lock()?;
        {
            let mut inputs = self
                .liveness
                .inner
                .lock()
                .map_err(|_| ReferenceKvError::Poisoned)?;
            for (identity, generation) in live {
                let floor = inputs.floors.entry(identity.clone()).or_insert(*generation);
                *floor = (*floor).max(*generation);
            }
        }
        if state.faults.index_update == IndexUpdate::Suppressed {
            state.stats.suppressed_index_updates += 1;
            return Ok(0);
        }
        let keys = state.catalog.invalidate_superseded(live);
        let retired = keys.len();
        for key in &keys {
            self.retire(&mut state, key)?;
        }
        self.drain_retirements(&mut state)?;
        Ok(retired)
    }

    /// Apply one committed ledger event to this backend's own catalog through the same
    /// mapping the runtime's `apply_committed` hook uses (`Revoked`, `ProcedureRevoked`,
    /// capsule and constraint supersession, `ExpertFenceCommitted`,
    /// `ExpertStructuralCommitted`; a compound `InputInvalidated` is refused here and goes
    /// through [`Self::apply_input_invalidation`]): the affected entries are tombstoned, their
    /// witnesses
    /// revoked and the backend prefixes they held retired. The backend's own liveness set
    /// follows (revoked pairs, supersession floors, fence epochs, structurally named
    /// experts), so use-time checks agree with the index. Returns how many backend prefixes
    /// were retired.
    ///
    /// A `SemanticDeltaCommitted` event carries no delta here, so it changes nothing; apply
    /// it through [`Self::invalidate_inputs`] with the affected keys. Only committed events
    /// may be passed; this method is the host's fence hook for a standalone backend, not a
    /// replacement for the runtime's. [`IndexUpdate::Suppressed`] applies as for
    /// [`Self::invalidate_inputs`].
    pub fn apply_committed_event(&self, event: &LedgerEvent) -> Result<usize, ReferenceKvError> {
        if matches!(event, LedgerEvent::InputInvalidated { .. }) {
            return Err(ReferenceKvError::InputInvalidationNeedsFences);
        }
        if matches!(event, LedgerEvent::ExpertAuthorityChanged { .. }) {
            return Err(ReferenceKvError::AuthorityChangeNeedsFences);
        }
        let mut state = self.lock()?;
        {
            let mut inputs = self
                .liveness
                .inner
                .lock()
                .map_err(|_| ReferenceKvError::Poisoned)?;
            inputs.note_event(event);
        }
        if state.faults.index_update == IndexUpdate::Suppressed {
            state.stats.suppressed_index_updates += 1;
            return Ok(0);
        }
        let Some(keys) =
            crate::kv_prefix::invalidate_catalog_for_event(&mut state.catalog, event, None)
        else {
            return Ok(0);
        };
        let retired = keys.len();
        for key in &keys {
            self.retire(&mut state, key)?;
        }
        self.drain_retirements(&mut state)?;
        Ok(retired)
    }

    /// Apply one committed compound [`LedgerEvent::InputInvalidated`] to this backend: the input
    /// half through the same mapping the runtime's hook uses (a revocation like `Revoked`, a
    /// supersession like a generation bump), and the experts the event fenced through `fences`,
    /// the fence epochs the runtime derived (`InputInvalidationOutcome::fence_epochs`). The event
    /// carries only the count and digest of that set, so a standalone backend cannot derive it
    /// and must be handed it. An event of another kind is refused. Returns how many backend
    /// prefixes were retired.
    ///
    /// # Errors
    /// [`ReferenceKvError::NotAnInputInvalidation`] for any other event; a poisoned backend.
    pub fn apply_input_invalidation(
        &self,
        event: &LedgerEvent,
        fences: &BTreeMap<String, u64>,
    ) -> Result<usize, ReferenceKvError> {
        if !matches!(event, LedgerEvent::InputInvalidated { .. }) {
            return Err(ReferenceKvError::NotAnInputInvalidation);
        }
        let mut state = self.lock()?;
        {
            let mut inputs = self
                .liveness
                .inner
                .lock()
                .map_err(|_| ReferenceKvError::Poisoned)?;
            inputs.note_event(event);
            inputs.note_fences(fences.clone());
        }
        if state.faults.index_update == IndexUpdate::Suppressed {
            state.stats.suppressed_index_updates += 1;
            return Ok(0);
        }
        let mut keys =
            crate::kv_prefix::invalidate_catalog_for_event(&mut state.catalog, event, None)
                .unwrap_or_default();
        keys.extend(state.catalog.invalidate_expert_fences(fences));
        let retired = keys.len();
        for key in &keys {
            self.retire(&mut state, key)?;
        }
        self.drain_retirements(&mut state)?;
        Ok(retired)
    }

    /// Apply one committed [`LedgerEvent::ExpertAuthorityChanged`] to this backend: the experts
    /// the event fenced die through `fences`, the fence epochs the runtime derived
    /// (`ExpertAuthorityOutcome::fence_epochs`). The event carries only the count and digest of
    /// that set and changes no other KV liveness, so a standalone backend must be handed the
    /// epochs. An event of another kind is refused. Returns how many backend prefixes were
    /// retired.
    ///
    /// # Errors
    /// [`ReferenceKvError::NotAnAuthorityChange`] for any other event; a poisoned backend.
    pub fn apply_authority_change(
        &self,
        event: &LedgerEvent,
        fences: &BTreeMap<String, u64>,
    ) -> Result<usize, ReferenceKvError> {
        if !matches!(event, LedgerEvent::ExpertAuthorityChanged { .. }) {
            return Err(ReferenceKvError::NotAnAuthorityChange);
        }
        let mut state = self.lock()?;
        {
            let mut inputs = self
                .liveness
                .inner
                .lock()
                .map_err(|_| ReferenceKvError::Poisoned)?;
            inputs.note_fences(fences.clone());
        }
        if state.faults.index_update == IndexUpdate::Suppressed {
            state.stats.suppressed_index_updates += 1;
            return Ok(0);
        }
        let keys = state.catalog.invalidate_expert_fences(fences);
        let retired = keys.len();
        for key in &keys {
            self.retire(&mut state, key)?;
        }
        self.drain_retirements(&mut state)?;
        Ok(retired)
    }

    /// The catalog's canonical snapshot (keys, lineage, dependencies; no K/V pages).
    pub fn export_catalog(&self) -> Result<Vec<u8>, ReferenceKvError> {
        self.lock()?
            .catalog
            .export_entries()
            .map_err(ReferenceKvError::Snapshot)
    }

    /// Admit a snapshot into the empty catalog without a liveness check: the first use
    /// decides (the R005 restore arm). An entry whose input was revoked meanwhile misses
    /// through the use-time check; one whose pages this backend does not hold is dropped
    /// when its fork finds them missing.
    pub fn restore_catalog(&self, snapshot: &[u8]) -> Result<KvRestoreReport, ReferenceKvError> {
        self.lock()?
            .catalog
            .restore_entries(snapshot)
            .map_err(ReferenceKvError::Snapshot)
    }

    /// Evict every unpinned entry and retire its backend prefix. Returns how many entries
    /// were evicted.
    pub fn flush(&self) -> Result<usize, ReferenceKvError> {
        let mut state = self.lock()?;
        let mut evicted = 0;
        while state.catalog.evict_lru() {
            evicted += 1;
        }
        self.drain_retirements(&mut state)?;
        crate::kv_infer_session::audit_pins(&mut state);
        Ok(evicted)
    }

    pub(crate) fn retire(
        &self,
        state: &mut ReuseState,
        key: &KvPrefixKey,
    ) -> Result<(), ReferenceKvError> {
        state.held.remove(key);
        match self.paged.retire_prefix(key) {
            Ok(()) | Err(KvBackendError::PrefixNotFound) => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    pub(crate) fn drain_retirements(&self, state: &mut ReuseState) -> Result<(), ReferenceKvError> {
        for key in state.catalog.take_retirements() {
            self.retire(state, &key)?;
        }
        Ok(())
    }

    pub(crate) fn binding(
        &self,
        scope: &ReferenceKvScope,
        chain: &KvChainId,
    ) -> Result<KvPageBinding, ReferenceKvError> {
        Ok(KvPageBinding {
            model: self.schema.model.clone(),
            adapter: self.schema.adapter.clone(),
            generation: Generation(1),
            execution_manifest: Self::manifest_for(chain),
            principal: scope.principal.clone(),
            device: self.schema.device.clone(),
            dtype: KvTensorDType::F32,
            page_tokens: KvPageSize::new(KV_CHUNK_TOKENS as u16)?,
        })
    }

    pub(crate) fn salt(&self, scope: &ReferenceKvScope) -> KvSalt {
        KvSalt::derive(
            &scope.principal,
            &scope.project,
            &self.schema.model,
            self.tokenizer.identity(),
            &self.build,
            SCHEMA_NAME,
        )
    }

    pub(crate) fn new_cache(
        &self,
        scope: &ReferenceKvScope,
        chain: &KvChainId,
    ) -> Result<ReferenceCache, ReferenceKvError> {
        self.new_cache_with(scope, chain, None)
    }

    /// A cache whose schema and page binding carry `adapter` (the base adapter when `None`).
    pub(crate) fn new_cache_with(
        &self,
        scope: &ReferenceKvScope,
        chain: &KvChainId,
        adapter: Option<&AdapterVersion>,
    ) -> Result<ReferenceCache, ReferenceKvError> {
        let capacity = self.limits.max_prompt_tokens + self.limits.max_new_tokens;
        let mut schema = self.schema.clone();
        let mut binding = self.binding(scope, chain)?;
        if let Some(adapter) = adapter {
            schema.adapter = adapter.clone();
            binding.adapter = adapter.clone();
        }
        Ok(self.paged.allocate_bound(schema, capacity, binding)?)
    }

    /// What identifies the producer of this backend's chains.
    pub(crate) fn chain_identity(&self) -> KvChainIdentity {
        KvChainIdentity {
            base_model: MODEL_NAME.to_owned(),
            generation: 1,
            build: self.build.clone(),
            tokenizer: format!(
                "{}@{}",
                self.tokenizer.identity().name(),
                self.tokenizer.identity().version()
            ),
        }
    }

    fn generate(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        policy: KvReusePolicy,
        cache_use: CacheUse,
        binding: &StackBinding,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        let _gate = self
            .oneshot
            .lock()
            .map_err(|_| ReferenceKvError::Poisoned)?;
        let liveness = self.own_liveness();
        let mut session = KvSession::begin_bound(
            self,
            scope,
            request,
            policy,
            cache_use,
            crate::kv_infer_session::SessionAdmission::OneShot,
            binding,
            &liveness,
        )?;
        match session.run_to_end(&liveness)? {
            KvSessionFinish::Completed(outcome) => Ok(*outcome),
            // An invalidation cancelled the run: nothing produced after it escaped, and
            // the restart is a cold run (no lookup, no insert) guarded by the same oracle.
            KvSessionFinish::Cancelled(_) => session.restart_with(&liveness),
        }
    }

    /// `f32` values per cached token row (`kv_heads * head_dim`).
    pub(crate) fn row_width(&self) -> usize {
        let layer = &self.stack.config().layer;
        layer.num_key_value_heads * (layer.hidden_size / layer.num_attention_heads)
    }

    pub(crate) fn max_session_limit(&self) -> usize {
        self.limits.max_sessions
    }

    pub(crate) fn prompt_bound(&self) -> usize {
        self.limits.max_prompt_tokens
    }

    pub(crate) fn max_new(&self) -> usize {
        self.limits.max_new_tokens
    }
}

pub(crate) fn check_scope(scope: &ReferenceKvScope) -> Result<(), ReferenceKvError> {
    if scope.principal.0.trim().is_empty() || scope.project.0.trim().is_empty() {
        return Err(ReferenceKvError::InvalidScope);
    }
    Ok(())
}

/// Index of the first maximum.
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

impl KvReusingBackend for ReferenceKvReuseBackend {
    fn name(&self) -> &str {
        BACKEND_NAME
    }

    fn infer_segmented(
        &self,
        request: &SegmentedModelRequest,
        hint: Option<&KvReuseHint>,
    ) -> Result<ModelRun, ModelError> {
        Ok(self.infer_scoped(&self.default_scope, request, hint)?.run)
    }
}

impl InferenceBackend for ReferenceKvReuseBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    /// The cold path: tokenizes `raw_text` as one Raw segment, never looks up or inserts
    /// and reports no reuse.
    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        let segmented = self.text_request(
            request.request_id.clone(),
            request.semantic_context.clone(),
            &request.raw_text,
        )?;
        let mut outcome = self.infer_cold(&self.default_scope, &segmented)?;
        // The cold path reports no reuse, so its events carry no KV taint either.
        outcome.run.kv_report = None;
        Ok(outcome.run.events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> ReferenceKvScope {
        ReferenceKvScope {
            principal: PrincipalId::from("alice"),
            project: ProjectId::from("p"),
        }
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[0.0, 2.0, 2.0, 1.0]), 1);
    }

    /// The pre-wave derivation: a hash of the seed and the config Debug text only.
    fn legacy_seed_chain(config: &Qwen2StackConfig, seed: u64) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"legacy");
        hasher.update(seed.to_le_bytes());
        hasher.update(format!("{config:?}").as_bytes());
        hasher.finalize().into()
    }

    fn chain_of(stack: &Qwen2PagedStack) -> KvChainId {
        let regime = stack.config().layer.rope_regime(1).unwrap();
        derive_chain(
            stack,
            &build_identity(),
            ByteTokenizer::new().identity(),
            &regime,
        )
        .unwrap()
    }

    #[test]
    fn chain_follows_weights_not_seed() {
        let config = stack_config(&ReferenceKvLimits::default(), Qwen2RopeScaling::None);
        let a = Qwen2PagedStack::reference(config.clone(), 1).unwrap();
        assert_eq!(chain_of(&a), chain_of(&a.clone()));
        let b = Qwen2PagedStack::reference(config.clone(), 2).unwrap();
        assert_ne!(chain_of(&a), chain_of(&b));
        let mut other = config.clone();
        other.num_layers = 3;
        let c = Qwen2PagedStack::reference(other, 1).unwrap();
        assert_ne!(chain_of(&a), chain_of(&c));

        // Same seed and config, one weight perturbed: the old derivation collides.
        let mut weights = a.weights().clone();
        weights.layers[1].decoder.gate.weight[0] += 1.0;
        let perturbed = Qwen2PagedStack::new(config.clone(), weights).unwrap();
        assert_eq!(
            legacy_seed_chain(&config, 1),
            legacy_seed_chain(perturbed.config(), 1)
        );
        assert_ne!(chain_of(&a), chain_of(&perturbed));
    }

    #[test]
    fn build_identity_changes_the_chain() {
        let config = stack_config(&ReferenceKvLimits::default(), Qwen2RopeScaling::None);
        let stack = Qwen2PagedStack::reference(config, 1).unwrap();
        let tokenizer = ByteTokenizer::new();
        let regime = stack.config().layer.rope_regime(1).unwrap();
        let here = derive_chain(&stack, &build_identity(), tokenizer.identity(), &regime).unwrap();
        let other = derive_chain(&stack, "another-build", tokenizer.identity(), &regime).unwrap();
        assert_ne!(here, other);
        assert!(derive_chain(&stack, " ", tokenizer.identity(), &regime).is_err());
    }

    #[test]
    fn invalid_limits_and_scope_are_refused() {
        let zero = ReferenceKvLimits {
            max_new_tokens: 0,
            ..ReferenceKvLimits::default()
        };
        assert!(matches!(
            ReferenceKvReuseBackend::new(zero, scope()),
            Err(ReferenceKvError::InvalidLimits)
        ));
        let blank = ReferenceKvScope {
            principal: PrincipalId::from(" "),
            project: ProjectId::from("p"),
        };
        assert!(matches!(
            ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), blank),
            Err(ReferenceKvError::InvalidScope)
        ));
    }
}
