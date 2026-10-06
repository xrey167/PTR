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
//! 5. returns a [`ModelRun`] whose [`KvReuseReport`] is always [`KvReuseMode::Exact`].
//!    This backend never produces Approximate output.
//!
//! # Chain id
//!
//! The catalog's [`KvChainId`] is opaque to it. It is `chain(L-1)` of the per-layer chain
//! of `ptr_pods::derive_kv_layer_chains` (doc 36 section D), computed from the actual
//! weights and configuration of the stack: embedding digest, every layer's norms, K/V
//! projections, QK-norm, attention and MLP weights, RoPE configuration (per the PreRope /
//! PostRope rule), the base model identity and generation, the tokenizer and the backend
//! build identity (also folded into the catalog salt). Two stacks with the same seed but
//! any different weight get different ids. Not covered: adapters (none exist).
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
//! # Scope
//!
//! Liveness of bound prompt segments is answered by this backend's own revocation set
//! ([`ReferenceKvReuseBackend::invalidate_inputs`]); it is not wired to the ledger's
//! lifecycle state. This is an experiment skeleton, not an R005 result.

use crate::{
    CatalogLimits, KvCatalogError, KvCatalogStats, KvChainId, KvDependency, KvLivenessOracle,
    KvLookup, KvPageCharge, KvPagedInsert, KvPin, KvPrefixCatalog, KvRequestPositions, KvSalt,
    KV_CHUNK_TOKENS,
};
use ptr_model_api::{
    ByteTokenizer, InferenceBackend, KvReuseHint, KvReuseMode, KvReusePolicy, KvReuseReport,
    KvReusingBackend, ModelError, ModelEvent, ModelRequest, ModelRun, PromptSegment,
    SegmentedModelRequest, TokenContractError, TokenId, Tokenizer, TokenizerIdentity,
};
use ptr_pods::{
    derive_kv_layer_chains_in_regime, InMemoryPagedKvBackend, KvBackendError, KvChainError,
    KvChainIdentity, KvPageBinding, KvPageSize, KvPrefixKey, KvTensorDType, KvTensorSchema,
    PagedKvTensorBackend, Qwen2Activation, Qwen2LayerConfig, Qwen2PagedStack, Qwen2QkNorm,
    Qwen2RopeRegime, Qwen2RopeScaling, Qwen2StackConfig, Qwen2StackError, Qwen2StackInput,
};
use ptr_types::{
    AdapterVersion, DeviceId, Digest, Generation, ModelVersion, PrincipalId, ProjectId,
};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, Mutex};

const LOGITS_DOMAIN: &[u8] = b"ptr.kv.reference-logits.v1\0";
const MANIFEST_DOMAIN: &[u8] = b"ptr.kv.reference-manifest.v1\0";

/// R005 `weight_seed`.
pub const REFERENCE_KV_WEIGHT_SEED: u64 = 20240607;
const MODEL_NAME: &str = "r005-tiny";
const ADAPTER_NAME: &str = "base";
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
    pub catalog: CatalogLimits,
}

impl Default for ReferenceKvLimits {
    fn default() -> Self {
        Self {
            max_prompt_tokens: 256,
            max_new_tokens: 8,
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
    /// The fork was cancelled by an invalidation and the run restarted from a fresh prefill.
    pub restarted_after_cancel: bool,
}

/// Whether a run may look up and insert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CacheUse {
    Disabled,
    Enabled,
}

/// Liveness as this backend knows it: an input is live unless it was revoked.
#[derive(Debug, Default)]
struct RevokedInputs {
    revoked: BTreeSet<(String, Generation)>,
}

impl KvLivenessOracle for RevokedInputs {
    fn is_live(&self, dependency: &KvDependency) -> bool {
        !self
            .revoked
            .contains(&(dependency.identity.clone(), dependency.generation))
    }
}

struct ReuseState {
    catalog: KvPrefixCatalog,
    liveness: RevokedInputs,
    /// Backend prefix keys some catalog entry holds or held and whose retirement the
    /// catalog has not yet reported; a refused insert must not retire these.
    held: BTreeSet<KvPrefixKey>,
}

/// See the module documentation.
pub struct ReferenceKvReuseBackend {
    stack: Qwen2PagedStack,
    tokenizer: ByteTokenizer,
    paged: InMemoryPagedKvBackend,
    chain: KvChainId,
    build: String,
    schema: KvTensorSchema,
    limits: ReferenceKvLimits,
    default_scope: ReferenceKvScope,
    state: Mutex<ReuseState>,
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
        if limits.max_prompt_tokens == 0 || limits.max_new_tokens == 0 {
            return Err(ReferenceKvError::InvalidLimits);
        }
        check_scope(&default_scope)?;
        let config = stack_config(&limits, rope_scaling);
        let stack = Qwen2PagedStack::reference(config, weight_seed)?;
        let paged = InMemoryPagedKvBackend::new(KvPageSize::new(KV_CHUNK_TOKENS as u16)?);
        let schema = stack.kv_schema(
            ModelVersion::from(MODEL_NAME),
            AdapterVersion::from(ADAPTER_NAME),
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
                liveness: RevokedInputs::default(),
                held: BTreeSet::new(),
            }),
        })
    }

    pub fn limits(&self) -> ReferenceKvLimits {
        self.limits
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

    fn chain_for(&self, regime: &Qwen2RopeRegime) -> Result<KvChainId, ReferenceKvError> {
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

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, ReuseState>, ReferenceKvError> {
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
        self.generate(scope, request, policy, CacheUse::Enabled)
    }

    /// Run `request` with no lookup and no insert: the cold path.
    pub fn infer_cold(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        self.generate(scope, request, KvReusePolicy::ExactOnly, CacheUse::Disabled)
    }

    /// Record that `(identity, generation)` pairs are no longer live: every entry that
    /// depends on one is tombstoned, its unused witnesses are revoked and the backend
    /// prefixes it held are retired. Later lookups also see the revocation at use time.
    pub fn invalidate_inputs(
        &self,
        invalidated: &BTreeSet<(String, Generation)>,
    ) -> Result<usize, ReferenceKvError> {
        let mut state = self.lock()?;
        state.liveness.revoked.extend(invalidated.iter().cloned());
        let keys = state.catalog.invalidate_inputs(invalidated);
        let retired = keys.len();
        for key in &keys {
            self.retire(&mut state, key)?;
        }
        self.drain_retirements(&mut state)?;
        Ok(retired)
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
        Ok(evicted)
    }

    fn retire(&self, state: &mut ReuseState, key: &KvPrefixKey) -> Result<(), ReferenceKvError> {
        state.held.remove(key);
        match self.paged.retire_prefix(key) {
            Ok(()) | Err(KvBackendError::PrefixNotFound) => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    fn drain_retirements(&self, state: &mut ReuseState) -> Result<(), ReferenceKvError> {
        for key in state.catalog.take_retirements() {
            self.retire(state, &key)?;
        }
        Ok(())
    }

    fn binding(
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

    fn salt(&self, scope: &ReferenceKvScope) -> KvSalt {
        KvSalt::derive(
            &scope.principal,
            &scope.project,
            &self.schema.model,
            self.tokenizer.identity(),
            &self.build,
            SCHEMA_NAME,
        )
    }

    fn new_cache(
        &self,
        scope: &ReferenceKvScope,
        chain: &KvChainId,
    ) -> Result<<InMemoryPagedKvBackend as ptr_pods::KvTensorBackend>::Cache, ReferenceKvError>
    {
        let capacity = self.limits.max_prompt_tokens + self.limits.max_new_tokens;
        Ok(self
            .paged
            .allocate_bound(self.schema.clone(), capacity, self.binding(scope, chain)?)?)
    }

    fn generate(
        &self,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        policy: KvReusePolicy,
        cache_use: CacheUse,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        check_scope(scope)?;
        if request.tokenizer() != self.tokenizer.identity() {
            return Err(ReferenceKvError::TokenizerMismatch {
                found: request.tokenizer().clone(),
            });
        }
        let total = request.total_tokens();
        if total > self.limits.max_prompt_tokens {
            return Err(ReferenceKvError::PromptTooLong {
                tokens: total,
                maximum: self.limits.max_prompt_tokens,
            });
        }
        let mut state = self.lock()?;
        let mut outcome = self.attempt(&mut state, scope, request, policy, cache_use)?;
        if outcome.cancelled {
            // An invalidation cancelled the fork: nothing produced after it may escape.
            outcome = self.attempt(&mut state, scope, request, policy, CacheUse::Disabled)?;
            outcome.value.restarted_after_cancel = true;
        }
        Ok(outcome.value)
    }

    fn attempt(
        &self,
        state: &mut ReuseState,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        policy: KvReusePolicy,
        cache_use: CacheUse,
    ) -> Result<Attempt, ReferenceKvError> {
        let salt = self.salt(scope);
        let tokens = request.tokens();
        let total = tokens.len();
        // Fixed once here; prefill and every decode step use this value.
        let regime = self.admit_regime(total)?;
        let chain = self.chain_for(&regime)?;
        let ids: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        let mut cache = self.new_cache(scope, &chain)?;
        let mut hit_tokens = 0usize;
        let mut pin: Option<KvPin> = None;

        if cache_use == CacheUse::Enabled {
            let lookup = KvLookup {
                salt: &salt,
                chain: &chain,
                request,
                positions: &KvRequestPositions::Canonical,
            };
            let hit = state.catalog.lookup(lookup, &state.liveness);
            if let Some(hit) = hit {
                let matched = hit.matched_tokens();
                if matched >= total {
                    // The stack needs the logits of the last prompt token, which a cache
                    // does not hold: refuse the whole-prompt hit and prefill from scratch.
                    let hit_pin = hit.pin().clone();
                    state.catalog.release(&hit_pin);
                } else {
                    match state
                        .catalog
                        .fork(&self.paged, &mut cache, hit, &state.liveness)
                    {
                        Ok(forked) => {
                            pin = Some(forked);
                            hit_tokens = matched;
                        }
                        Err(_) => {
                            // The fork was refused before use; start from an untouched cache.
                            cache = self.new_cache(scope, &chain)?;
                        }
                    }
                }
            }
            self.drain_retirements(state)?;
        }

        let result = self.prefill_and_decode(
            state, request, &salt, &chain, &regime, &ids, &mut cache, hit_tokens, cache_use,
        );
        let still_live = pin
            .as_ref()
            .is_none_or(|pin| state.catalog.pin_is_live(pin));
        if let Some(pin) = &pin {
            state.catalog.release(pin);
        }
        self.drain_retirements(state)?;
        let work = result?;
        let prompt_tokens = total;
        let mut outcome = ReferenceKvOutcome {
            run: ModelRun {
                events: Vec::new(),
                kv_report: None,
            },
            generated: work.generated,
            logits_digest: work.logits_digest,
            prompt_tokens,
            rope_regime: regime,
            hit_tokens,
            prefilled_tokens: work.prefilled,
            inserted_chunks: work.inserted,
            insert_refusals: work.refused,
            restarted_after_cancel: false,
        };
        if !still_live {
            return Ok(Attempt {
                cancelled: true,
                value: outcome,
            });
        }
        let bytes: Vec<u8> = outcome
            .generated
            .iter()
            .map(|token| u8::try_from(token.0).unwrap_or(b'?'))
            .collect();
        let report = KvReuseReport::new(hit_tokens, KvReuseMode::Exact, total, policy)?;
        outcome.run = ModelRun {
            events: vec![
                ModelEvent::Token(String::from_utf8_lossy(&bytes).into_owned()),
                ModelEvent::Finished,
            ],
            kv_report: Some(report),
        };
        Ok(Attempt {
            cancelled: false,
            value: outcome,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_and_decode(
        &self,
        state: &mut ReuseState,
        request: &SegmentedModelRequest,
        salt: &KvSalt,
        chain: &KvChainId,
        regime: &Qwen2RopeRegime,
        ids: &[u32],
        cache: &mut <InMemoryPagedKvBackend as ptr_pods::KvTensorBackend>::Cache,
        hit_tokens: usize,
        cache_use: CacheUse,
    ) -> Result<Work, ReferenceKvError> {
        let total = ids.len();
        let mut work = Work::default();
        let mut position = hit_tokens;
        let mut last_logits: Vec<f32> = Vec::new();
        while position < total {
            let next = total.min((position / KV_CHUNK_TOKENS + 1) * KV_CHUNK_TOKENS);
            let output = self.stack.prefill_in_regime(
                &self.paged,
                cache,
                Qwen2StackInput {
                    tokens: ids[position..next].to_vec(),
                },
                regime,
            )?;
            work.prefilled += next - position;
            last_logits = output.logits.into_iter().next_back().unwrap_or_default();
            position = next;
            if cache_use == CacheUse::Enabled && position.is_multiple_of(KV_CHUNK_TOKENS) {
                self.offer_boundary(state, salt, chain, request, cache, &mut work)?;
            }
        }
        let mut hasher = Sha256::new();
        hasher.update(LOGITS_DOMAIN);
        for step in 0..self.limits.max_new_tokens {
            if last_logits.iter().any(|value| !value.is_finite()) {
                return Err(ReferenceKvError::NonFiniteLogits);
            }
            for value in &last_logits {
                hasher.update(value.to_bits().to_le_bytes());
            }
            let token = argmax(&last_logits);
            work.generated.push(TokenId(token));
            if step + 1 < self.limits.max_new_tokens {
                let output = self.stack.decode_in_regime(
                    &self.paged,
                    cache,
                    Qwen2StackInput {
                        tokens: vec![token],
                    },
                    regime,
                )?;
                last_logits = output.logits.into_iter().next().unwrap_or_default();
            }
        }
        work.logits_digest = hasher.finalize().into();
        Ok(work)
    }

    /// Publish the cache's full pages and offer them to the catalog. A refusal that only
    /// means "not cached" is counted; anything else is an error.
    fn offer_boundary(
        &self,
        state: &mut ReuseState,
        salt: &KvSalt,
        chain: &KvChainId,
        request: &SegmentedModelRequest,
        cache: &mut <InMemoryPagedKvBackend as ptr_pods::KvTensorBackend>::Cache,
        work: &mut Work,
    ) -> Result<(), ReferenceKvError> {
        let prefix = self.paged.publish_prefix(cache)?;
        let key = prefix.key.clone();
        // Per-page charges: each unique page digest is charged once with its real bytes, so
        // prompts that share a long prefix do not re-charge it (the former whole-cache
        // `resident_bytes` figure grew quadratically with prefix length).
        let pages = self
            .paged
            .prefix_page_bytes(&prefix)?
            .into_iter()
            .map(|(digest, bytes)| KvPageCharge { digest, bytes })
            .collect();
        let ReuseState {
            catalog, liveness, ..
        } = state;
        let result = catalog.insert_paged(
            KvPagedInsert {
                salt,
                chain,
                request,
                prefix,
                pages,
                extra_dependencies: Vec::new(),
            },
            liveness,
        );
        match result {
            Ok(()) => {
                work.inserted += 1;
                state.held.insert(key);
            }
            Err(
                KvCatalogError::CapacityExhausted(_)
                | KvCatalogError::SlotDraining
                | KvCatalogError::DuplicateEntry
                | KvCatalogError::DependencyNotLive,
            ) => {
                work.refused += 1;
                if !state.held.contains(&key) {
                    match self.paged.retire_prefix(&key) {
                        Ok(()) | Err(KvBackendError::PrefixNotFound) => {}
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            Err(error) => return Err(ReferenceKvError::Catalog(error)),
        }
        self.drain_retirements(state)
    }
}

fn check_scope(scope: &ReferenceKvScope) -> Result<(), ReferenceKvError> {
    if scope.principal.0.trim().is_empty() || scope.project.0.trim().is_empty() {
        return Err(ReferenceKvError::InvalidScope);
    }
    Ok(())
}

/// Index of the first maximum.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

#[derive(Default)]
struct Work {
    generated: Vec<TokenId>,
    logits_digest: Digest,
    prefilled: usize,
    inserted: usize,
    refused: usize,
}

struct Attempt {
    cancelled: bool,
    value: ReferenceKvOutcome,
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
