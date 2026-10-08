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
    KV_CHUNK_TOKENS,
};
use ptr_ledger::LedgerEvent;
use ptr_model_api::{
    ByteTokenizer, InferenceBackend, KvReuseHint, KvReusePolicy, KvReusingBackend, ModelError,
    ModelEvent, ModelRequest, ModelRun, PromptSegment, SegmentedModelRequest, TokenContractError,
    TokenId, Tokenizer, TokenizerIdentity,
};
use ptr_pods::{
    derive_kv_layer_chains_in_regime, InMemoryPagedKvBackend, KvBackendError, KvChainError,
    KvChainIdentity, KvPageBinding, KvPageSize, KvPrefixKey, KvTensorDType, KvTensorSchema,
    PagedKvTensorBackend, Qwen2Activation, Qwen2LayerConfig, Qwen2PagedStack, Qwen2QkNorm,
    Qwen2RopeRegime, Qwen2RopeScaling, Qwen2StackConfig, Qwen2StackError,
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
                for (identity, epoch) in crate::expert::expert_fence_epochs(event) {
                    let known = self.fences.entry(identity).or_insert(epoch);
                    *known = (*known).max(epoch);
                }
            }
            _ => {}
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
}

impl fmt::Debug for ReferenceKvReuseBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReferenceKvReuseBackend")
            .field("limits", &self.limits)
            .field("default_scope", &self.default_scope)
            .finish_non_exhaustive()
    }
}

fn stack_config(limits: &ReferenceKvLimits, rope_scaling: Qwen2RopeScaling) -> Qwen2StackConfig {
    // R005 tiny_model_config: 4 layers, hidden 64, 4 heads, 2 KV heads, head_dim 16, vocab 256.
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
        num_layers: 4,
        vocab_size: 256,
        key_encoding: ptr_pods::KvKeyEncoding::PreRope,
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
        if limits.max_prompt_tokens == 0 || limits.max_new_tokens == 0 || limits.max_sessions == 0 {
            return Err(ReferenceKvError::InvalidLimits);
        }
        check_scope(&default_scope)?;
        let config = stack_config(&limits, rope_scaling);
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
        })
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
    /// `ExpertStructuralCommitted`): the affected entries are tombstoned, their witnesses
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
