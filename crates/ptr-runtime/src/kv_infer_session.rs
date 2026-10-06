//! Stepwise sessions of the reference KV-reuse backend (doc 36, section D: liveness at
//! use; R005 race, lag, restore and saturation arms).
//!
//! A [`KvSession`] is one request run in steps the caller drives:
//!
//! 1. [`ReferenceKvReuseBackend::begin_session`](crate::ReferenceKvReuseBackend::begin_session)
//!    looks up the longest live exact prefix and forks it. The fork's **pin is held
//!    between steps**, so the entry cannot be evicted while the session is open.
//! 2. [`KvSession::prefill_chunk`] runs one 16-token chunk and, at a chunk boundary,
//!    offers the prefix to the catalog.
//! 3. [`KvSession::decode_step`] emits one greedy token.
//! 4. [`KvSession::finish`] returns the [`ReferenceKvOutcome`].
//!
//! No lock is held between steps, so a test (or another thread) can commit a revocation or
//! a supersession between two steps; single-threaded, the interleaving is deterministic.
//!
//! # Liveness at use
//!
//! Every step takes a `&dyn` [`KvLivenessOracle`]: the same trait the catalog asks, which
//! `PtrRuntime` implements from the committed lifecycle state (so liveness is anchored in
//! the ledger) and which [`ReferenceKvReuseBackend::own_liveness`](crate::ReferenceKvReuseBackend::own_liveness)
//! answers from the backend's own revocation set as a fallback. **Before every emitted
//! token, before every catalog insert (and at the start of every prefill chunk and at
//! finish)** the session re-checks every segment dependency of its request's lineage
//! against that oracle, and that the catalog entry behind its pin is not tombstoned. A
//! failure cancels: no token is emitted, nothing is inserted, the pin is released, the
//! step returns a typed [`KvCancellation`] and the session reports `Cancelled`.
//! [`KvSession::restart`] then re-runs the same request cold (no lookup, no insert),
//! which is bitwise equal to a fresh cold run.
//!
//! Because the check is made against the oracle on every use, a lost or delayed
//! catalog-index invalidation (see [`IndexUpdate::Suppressed`](crate::IndexUpdate)) cannot
//! produce a stale hit or a token after the revocation point: invalidation is an eviction
//! optimisation, the use-time check is the safety mechanism.
//!
//! A request whose lineage is **already not live at begin** is not cancelled for that
//! reason (a revoked input is the caller's policy question, and the one-shot path always
//! ran such a request uncached); the catalog refuses to hit on it and to insert from it
//! (`DependencyNotLive`, counted as an insert refusal). Only a lineage that was live at
//! begin and dies mid-session cancels.
//!
//! # Bounds and saturation
//!
//! At most [`ReferenceKvLimits::max_sessions`](crate::ReferenceKvLimits) sessions are open
//! (typed [`ReferenceKvError::SessionLimit`]). When the catalog cannot take a chunk because
//! every candidate victim is pinned by an open fork, the insert is refused with the typed
//! `CapacityExhausted` (recorded in the outcome and counted), the session continues with an
//! uncached prefill, and its output is bitwise equal to a cold run. Eviction never takes a
//! pinned entry; [`KvSessionStats::pinned_entry_evictions`] audits that after every insert
//! and flush.
//!
//! # Not covered
//!
//! One thread drives a session; this is not a concurrent serving scheduler. The oracle is
//! whatever the caller passes: the end-to-end ledger-anchored case needs a `PtrRuntime`
//! whose committed state the test advances (`tests/kv_prefix.rs` covers the runtime oracle
//! against the catalog; here the tests use a scripted oracle and the backend's own set).

use crate::kv_infer::{
    argmax, check_scope, CacheUse, ReferenceCache, ReferenceKvError, ReferenceKvOutcome,
    ReferenceKvReuseBackend, ReferenceKvScope, ReuseState, LOGITS_DOMAIN,
};
use crate::kv_infer_layered::{
    adopt_pending, assemble, offer_layers, record_of, resolve_binding, AssembleInput, BindingError,
    BindingRecord, LayeredReuseReport, LayeredState, StackBinding,
};
use crate::{
    KvCatalogError, KvChainId, KvDependency, KvDependencyKind, KvLivenessOracle, KvLookup,
    KvPageCharge, KvPagedInsert, KvPin, KvRequestPositions, KvSalt, KV_CHUNK_TOKENS,
};
use ptr_model_api::{
    KvReuseMode, KvReusePolicy, KvReuseReport, LineageContribution, ModelEvent, ModelRun,
    SegmentedModelRequest, TokenId, Tokenizer,
};
use ptr_pods::{
    KvBackendError, PagedKvTensorBackend, Qwen2Binding, Qwen2RopeRegime, Qwen2StackInput,
};
use ptr_types::{Digest, RequestId};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

/// Counters of the session lifecycle, for the R005 arms and for tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvSessionStats {
    pub sessions_begun: u64,
    pub sessions_completed: u64,
    pub sessions_cancelled: u64,
    pub sessions_restarted: u64,
    /// Request ids of sessions cancelled by a revocation or supersession.
    pub cancelled_requests: BTreeSet<RequestId>,
    /// Request ids that were then re-run cold by [`KvSession::restart`].
    pub restarted_requests: BTreeSet<RequestId>,
    /// `begin_session` calls refused with [`ReferenceKvError::SessionLimit`].
    pub session_limit_refusals: u64,
    /// Chunk inserts refused with `CapacityExhausted` (the uncached fallback ran).
    pub capacity_exhausted_inserts: u64,
    /// Evictions the catalog performed while at least one fork was pinned (the R005
    /// arm only counts as exercised if this is at least one).
    pub evictions_with_active_fork_pinned: u64,
    /// Open sessions' pinned entries found evicted or reclaimed under the pin. Must stay
    /// zero: the catalog never evicts a pinned entry.
    pub pinned_entry_evictions: u64,
    /// Hits the catalog returned that the oracle contradicted right after the fork
    /// (dropped, never served). Must stay zero.
    pub stale_hits: u64,
    /// Invalidations whose catalog index update was suppressed by the fault hook.
    pub suppressed_index_updates: u64,
    /// Sessions opened on a non-default [`StackBinding`] (per-layer entries).
    pub bound_sessions: u64,
    /// Sessions run Approximate (batch-variant routing): no lookup, no insert.
    pub approximate_sessions: u64,
    /// Chunk boundaries at which an Approximate session would have inserted and was refused.
    /// An Approximate run never reaches the Exact catalog; the catalog's own insert counters
    /// do not move for it.
    pub approximate_inserts_refused: u64,
    /// Rows at or after an aLoRA invocation inserted under a base-only key of an adapted
    /// layer, over every session. Must stay zero.
    pub post_invocation_rows_under_base_key: u64,
    /// Rows checked adoption found different from the computed ones. Must stay zero.
    pub adoption_mismatches: u64,
}

/// Where a session is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvSessionPhase {
    /// Prompt tokens remain to be prefilled.
    Prefilling,
    /// The prompt is prefilled; tokens remain to be decoded.
    Decoding,
    /// Every token is decoded; [`KvSession::finish`] is next.
    Finished,
    /// Cancelled by a failed liveness check; only [`KvSession::restart`] is allowed.
    Cancelled,
    /// Completed (or restarted to completion).
    Closed,
}

/// Why a session was cancelled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCancelReason {
    /// A segment dependency of the request's lineage is no longer live.
    InputNotLive,
    /// The catalog entry behind the session's pin was tombstoned.
    PrefixTombstoned,
}

/// A typed cancellation: nothing after this point was emitted or inserted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvCancellation {
    pub request_id: RequestId,
    pub reason: KvCancelReason,
    /// The phase the session was in when the check failed.
    pub at_phase: KvSessionPhase,
    /// Tokens emitted before the check failed; none was emitted at the failing step.
    pub tokens_emitted: usize,
    pub prefilled_tokens: usize,
}

/// What one step did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvSessionStep {
    /// A prefill chunk ran; `position` prompt tokens are now in the cache.
    Prefilled {
        position: usize,
    },
    /// One token was emitted.
    Token(TokenId),
    Cancelled(KvCancellation),
}

/// How a session ended.
#[derive(Clone, Debug, PartialEq)]
pub enum KvSessionFinish {
    Completed(Box<ReferenceKvOutcome>),
    Cancelled(KvCancellation),
}

/// Whether a session counts against `max_sessions`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SessionAdmission {
    Counted,
    /// A one-shot run, serialized by the backend's gate.
    OneShot,
}

#[derive(Default)]
struct Work {
    prefilled: usize,
    inserted: usize,
    refusals: Vec<KvCatalogError>,
}

/// Attempts [`KvSession::restart_with`] makes on a bound session (each re-resolves the
/// binding) before it surfaces [`ReferenceKvError::RestartCancelled`].
pub(crate) const MAX_RESTART_ATTEMPTS: usize = 3;

/// Always live: the oracle of an unbound cold restart, which uses no cache and so depends
/// on no cached state.
struct NoCache;

impl KvLivenessOracle for NoCache {
    fn is_live(&self, _: &KvDependency) -> bool {
        true
    }
}

/// See the module documentation.
pub struct KvSession<'a> {
    backend: &'a ReferenceKvReuseBackend,
    scope: ReferenceKvScope,
    request: SegmentedModelRequest,
    policy: KvReusePolicy,
    cache_use: CacheUse,
    id: u64,
    slot_open: bool,
    salt: KvSalt,
    chain: KvChainId,
    regime: Qwen2RopeRegime,
    ids: Vec<u32>,
    cache: ReferenceCache,
    hit_tokens: usize,
    pin: Option<KvPin>,
    /// Every segment dependency of the request's whole lineage.
    dependencies: Vec<KvDependency>,
    /// The lineage was live at begin, so a later failure cancels. A dead segment at begin
    /// disables only this class.
    lineage_guarded: bool,
    /// The bound session's base model, adapter and tracked live experts that were live when
    /// the session was guarded; each cancels on its own, independent of the lineage.
    binding_guards: Vec<KvDependency>,
    position: usize,
    last_logits: Vec<f32>,
    generated: Vec<TokenId>,
    hasher: Sha256,
    work: Work,
    phase: KvSessionPhase,
    restarted: bool,
    /// The binding currently in force (re-resolved by a restart).
    binding: StackBinding,
    record: BindingRecord,
    /// Per-layer state of a bound session; `None` for the default binding.
    layered: Option<LayeredState>,
    /// `false` when the binding routes batch-variantly: the run is Approximate.
    exact: bool,
    /// The caller asked for the catalog (an Approximate run still counts the refusals).
    catalog_requested: bool,
}

impl std::fmt::Debug for KvSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvSession")
            .field("request", self.request.request_id())
            .field("phase", &self.phase)
            .field("hit_tokens", &self.hit_tokens)
            .field("position", &self.position)
            .field("generated", &self.generated.len())
            .finish_non_exhaustive()
    }
}

fn new_hasher() -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(LOGITS_DOMAIN);
    hasher
}

/// The bound-segment dependencies of the first `covered` tokens of `request`.
fn lineage_dependencies(request: &SegmentedModelRequest, covered: usize) -> Vec<KvDependency> {
    let mut dependencies: Vec<KvDependency> = Vec::new();
    let mut start = 0usize;
    for segment in request.segments() {
        if start >= covered {
            break;
        }
        if let LineageContribution::Bound {
            kind,
            identity,
            generation,
            content_digest,
        } = segment.lineage_contribution()
        {
            let dependency = KvDependency {
                kind: KvDependencyKind::Segment(kind),
                identity,
                generation,
                content_digest,
                fence_epoch: 0,
            };
            if !dependencies.contains(&dependency) {
                dependencies.push(dependency);
            }
        }
        start += segment.tokens().len();
    }
    dependencies
}

/// Count and forget every open session's pin whose entry was evicted or reclaimed under it.
pub(crate) fn audit_pins(state: &mut ReuseState) {
    let mut missing = 0u64;
    let mut emptied: Vec<u64> = Vec::new();
    let ids: Vec<u64> = state.pins.keys().copied().collect();
    for id in ids {
        let Some(pins) = state.pins.get(&id) else {
            continue;
        };
        let held: Vec<KvPin> = pins
            .iter()
            .filter(|pin| state.catalog.pin_held(pin))
            .cloned()
            .collect();
        missing += (pins.len() - held.len()) as u64;
        if held.is_empty() {
            emptied.push(id);
        } else if let Some(slot) = state.pins.get_mut(&id) {
            *slot = held;
        }
    }
    for id in emptied {
        state.pins.remove(&id);
    }
    state.stats.pinned_entry_evictions += missing;
}

impl<'a> KvSession<'a> {
    pub(crate) fn begin(
        backend: &'a ReferenceKvReuseBackend,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        policy: KvReusePolicy,
        cache_use: CacheUse,
        admission: SessionAdmission,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<Self, ReferenceKvError> {
        Self::begin_bound(
            backend,
            scope,
            request,
            policy,
            cache_use,
            admission,
            &StackBinding::base(),
            liveness,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_bound(
        backend: &'a ReferenceKvReuseBackend,
        scope: &ReferenceKvScope,
        request: &SegmentedModelRequest,
        policy: KvReusePolicy,
        cache_use: CacheUse,
        admission: SessionAdmission,
        binding: &StackBinding,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<Self, ReferenceKvError> {
        check_scope(scope)?;
        if request.tokenizer() != backend.tokenizer().identity() {
            return Err(ReferenceKvError::TokenizerMismatch {
                found: request.tokenizer().clone(),
            });
        }
        let tokens = request.tokens();
        let total = tokens.len();
        if total > backend.prompt_bound() {
            return Err(ReferenceKvError::PromptTooLong {
                tokens: total,
                maximum: backend.prompt_bound(),
            });
        }
        let salt = backend.salt(scope);
        // Fixed once here; prefill and every decode step use this value.
        let regime = backend.admit_regime(total)?;
        let mut layered = if binding.is_default() {
            None
        } else {
            Some(LayeredState::prepare(backend, binding, &regime)?)
        };
        let exact = layered
            .as_ref()
            .is_none_or(|layered| !layered.approximate());
        if !exact && policy == KvReusePolicy::ExactOnly {
            return Err(BindingError::ApproximateNotAllowed.into());
        }
        let catalog_requested = cache_use == CacheUse::Enabled;
        // An Approximate run never consults or feeds the Exact catalog.
        let cache_use = if exact { cache_use } else { CacheUse::Disabled };
        let chain = match &layered {
            Some(layered) => layered.chain,
            None => backend.chain_for(&regime)?,
        };
        let mut cache = match &layered {
            Some(layered) => layered.fresh_cache(backend, scope)?,
            None => backend.new_cache(scope, &chain)?,
        };
        let ids: Vec<u32> = tokens.iter().map(|token| token.0).collect();
        let dependencies = lineage_dependencies(request, total);
        let mut binding_guards = Vec::new();
        if let Some(layered) = &layered {
            for dependency in layered.session_dependencies() {
                let live = liveness.is_live(&dependency);
                if catalog_requested && !live {
                    return Err(BindingError::NotLive {
                        identity: dependency.identity,
                    }
                    .into());
                }
                if cache_use == CacheUse::Enabled && live {
                    binding_guards.push(dependency);
                }
            }
        }
        let lineage_guarded = cache_use == CacheUse::Enabled
            && dependencies
                .iter()
                .all(|dependency| liveness.is_live(dependency));

        let mut state = backend.lock()?;
        if admission == SessionAdmission::Counted
            && state.open_sessions >= backend.max_session_limit()
        {
            state.stats.session_limit_refusals += 1;
            return Err(ReferenceKvError::SessionLimit {
                maximum: backend.max_session_limit(),
            });
        }
        let mut hit_tokens = 0usize;
        let mut pin: Option<KvPin> = None;
        if cache_use == CacheUse::Enabled {
            if let Some(layered) = layered.as_mut() {
                hit_tokens = assemble(
                    AssembleInput {
                        backend,
                        scope,
                        salt: &salt,
                        request,
                        total,
                        liveness,
                    },
                    &mut state,
                    &mut cache,
                    layered,
                )?;
            } else {
                let lookup = KvLookup {
                    salt: &salt,
                    chain: &chain,
                    request,
                    positions: &KvRequestPositions::Canonical,
                };
                if let Some(hit) = state.catalog.lookup(lookup, liveness) {
                    let matched = hit.matched_tokens();
                    if matched >= total {
                        // The stack needs the logits of the last prompt token, which a cache
                        // does not hold: refuse the whole-prompt hit and prefill from scratch.
                        let hit_pin = hit.pin().clone();
                        state.catalog.release(&hit_pin);
                    } else {
                        match state
                            .catalog
                            .fork(&backend.paged, &mut cache, hit, liveness)
                        {
                            Ok(forked) => {
                                let covered = lineage_dependencies(request, matched);
                                if covered
                                    .iter()
                                    .all(|dependency| liveness.is_live(dependency))
                                {
                                    pin = Some(forked);
                                    hit_tokens = matched;
                                } else {
                                    // The oracle contradicts a hit the catalog just served:
                                    // drop it, never use it.
                                    state.stats.stale_hits += 1;
                                    state.catalog.release(&forked);
                                    cache = backend.new_cache(scope, &chain)?;
                                }
                            }
                            Err(_) => {
                                // The fork was refused before use; start from an untouched
                                // cache.
                                cache = backend.new_cache(scope, &chain)?;
                            }
                        }
                    }
                }
            }
            backend.drain_retirements(&mut state)?;
        }
        state.next_session += 1;
        let id = state.next_session;
        state.open_sessions += 1;
        state.stats.sessions_begun += 1;
        if layered.is_some() {
            state.stats.bound_sessions += 1;
        }
        if !exact {
            state.stats.approximate_sessions += 1;
        }
        let mut pins: Vec<KvPin> = pin.iter().cloned().collect();
        if let Some(layered) = &layered {
            pins.extend(layered.all_pins());
        }
        if !pins.is_empty() {
            state.pins.insert(id, pins);
        }
        drop(state);
        Ok(Self {
            backend,
            scope: scope.clone(),
            request: request.clone(),
            policy,
            cache_use,
            id,
            slot_open: true,
            salt,
            chain,
            regime,
            ids,
            cache,
            hit_tokens,
            pin,
            dependencies,
            lineage_guarded,
            binding_guards,
            position: hit_tokens,
            last_logits: Vec::new(),
            generated: Vec::new(),
            hasher: new_hasher(),
            work: Work::default(),
            phase: KvSessionPhase::Prefilling,
            restarted: false,
            binding: binding.clone(),
            record: record_of(binding, backend.stack.config()),
            layered,
            exact,
            catalog_requested,
        })
    }

    pub fn request_id(&self) -> &RequestId {
        self.request.request_id()
    }

    pub fn phase(&self) -> KvSessionPhase {
        self.phase
    }

    /// Prompt tokens served from the catalog.
    pub fn hit_tokens(&self) -> usize {
        self.hit_tokens
    }

    /// Prompt tokens in the cache so far (hit plus prefilled).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Tokens emitted so far.
    pub fn generated(&self) -> &[TokenId] {
        &self.generated
    }

    /// The binding the session currently runs under (re-resolved by a restart).
    pub fn binding_record(&self) -> &BindingRecord {
        &self.record
    }

    /// Layer-scoped reuse accounting so far (`None` for the default binding); the counts
    /// that depend on the prompt length are filled in the outcome.
    pub fn layered_report(&self) -> Option<&LayeredReuseReport> {
        self.layered.as_ref().map(|layered| &layered.report)
    }

    /// Diagnostic: how many leading rows of `layer` in cache page `page` are physically the
    /// same allocation in this session's cache and in `other`'s.
    pub fn shared_leading_rows(&self, other: &KvSession<'_>, page: usize, layer: usize) -> usize {
        self.cache.shared_leading_rows(&other.cache, page, layer)
    }

    fn require(
        &self,
        phase: KvSessionPhase,
        operation: &'static str,
    ) -> Result<(), ReferenceKvError> {
        if self.phase == phase {
            Ok(())
        } else {
            Err(ReferenceKvError::SessionPhase {
                operation,
                phase: self.phase,
            })
        }
    }

    /// Run one prefill chunk (up to the next 16-token boundary). At a boundary the prefix
    /// is offered to the catalog, after a liveness check.
    pub fn prefill_chunk(
        &mut self,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvSessionStep, ReferenceKvError> {
        self.require(KvSessionPhase::Prefilling, "prefill_chunk")?;
        if let Some(cancelled) = self.check_live(liveness)? {
            return Ok(KvSessionStep::Cancelled(cancelled));
        }
        let backend = self.backend;
        let total = self.ids.len();
        let from = self.position;
        let next = total.min((from / KV_CHUNK_TOKENS + 1) * KV_CHUNK_TOKENS);
        let stack = self
            .layered
            .as_ref()
            .and_then(|layered| layered.stack.as_ref())
            .unwrap_or(&backend.stack);
        let output = stack.prefill_bound_in_regime(
            &backend.paged,
            &mut self.cache,
            Qwen2StackInput {
                tokens: self.ids[from..next].to_vec(),
            },
            &self.regime,
            &Qwen2Binding {
                adapter: self.binding.adapter.as_ref(),
            },
        )?;
        self.work.prefilled += next - from;
        self.last_logits = output.logits.into_iter().next_back().unwrap_or_default();
        self.position = next;
        if next >= total {
            self.adopt_layers(liveness)?;
        }
        if !self.exact && self.catalog_requested && next.is_multiple_of(KV_CHUNK_TOKENS) {
            self.backend.lock()?.stats.approximate_inserts_refused += 1;
        }
        if self.cache_use == CacheUse::Enabled && next.is_multiple_of(KV_CHUNK_TOKENS) {
            // Re-check immediately before the insert: nothing derived from a revoked input
            // may enter the catalog.
            if let Some(cancelled) = self.check_live(liveness)? {
                return Ok(KvSessionStep::Cancelled(cancelled));
            }
            self.offer_boundary(liveness)?;
        }
        if next >= total {
            self.phase = KvSessionPhase::Decoding;
        }
        Ok(KvSessionStep::Prefilled { position: next })
    }

    /// Emit one greedy token, after a liveness check.
    pub fn decode_step(
        &mut self,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvSessionStep, ReferenceKvError> {
        self.require(KvSessionPhase::Decoding, "decode_step")?;
        if let Some(cancelled) = self.check_live(liveness)? {
            return Ok(KvSessionStep::Cancelled(cancelled));
        }
        let backend = self.backend;
        if self.last_logits.iter().any(|value| !value.is_finite()) {
            return Err(ReferenceKvError::NonFiniteLogits);
        }
        for value in &self.last_logits {
            self.hasher.update(value.to_bits().to_le_bytes());
        }
        let token = argmax(&self.last_logits);
        self.generated.push(TokenId(token));
        if self.generated.len() < backend.max_new() {
            let stack = self
                .layered
                .as_ref()
                .and_then(|layered| layered.stack.as_ref())
                .unwrap_or(&backend.stack);
            let output = stack.decode_bound_in_regime(
                &backend.paged,
                &mut self.cache,
                Qwen2StackInput {
                    tokens: vec![token],
                },
                &self.regime,
                &Qwen2Binding {
                    adapter: self.binding.adapter.as_ref(),
                },
            )?;
            self.last_logits = output.logits.into_iter().next().unwrap_or_default();
        } else {
            self.phase = KvSessionPhase::Finished;
        }
        Ok(KvSessionStep::Token(TokenId(token)))
    }

    /// Complete the session after a last liveness check: release the pin and return the
    /// outcome.
    pub fn finish(
        &mut self,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvSessionFinish, ReferenceKvError> {
        self.require(KvSessionPhase::Finished, "finish")?;
        if let Some(cancelled) = self.check_live(liveness)? {
            return Ok(KvSessionFinish::Cancelled(cancelled));
        }
        self.release_pin()?;
        let total = self.ids.len();
        let bytes: Vec<u8> = self
            .generated
            .iter()
            .map(|token| u8::try_from(token.0).unwrap_or(b'?'))
            .collect();
        let logits_digest: Digest = std::mem::replace(&mut self.hasher, new_hasher())
            .finalize()
            .into();
        let report = KvReuseReport::new(
            self.hit_tokens,
            if self.exact {
                KvReuseMode::Exact
            } else {
                KvReuseMode::Approximate
            },
            total,
            self.policy,
        )?;
        let outcome = ReferenceKvOutcome {
            run: ModelRun {
                events: vec![
                    ModelEvent::Token(String::from_utf8_lossy(&bytes).into_owned()),
                    ModelEvent::Finished,
                ],
                kv_report: Some(report),
            },
            generated: self.generated.clone(),
            logits_digest,
            prompt_tokens: total,
            rope_regime: self.regime,
            hit_tokens: self.hit_tokens,
            prefilled_tokens: self.work.prefilled,
            inserted_chunks: self.work.inserted,
            insert_refusals: self.work.refusals.len(),
            insert_refusal_reasons: std::mem::take(&mut self.work.refusals),
            restarted_after_cancel: self.restarted,
            binding: self.record.clone(),
            layered: self
                .layered
                .as_ref()
                .map(|layered| layered.report.finalized(total)),
        };
        self.phase = KvSessionPhase::Closed;
        self.close_slot(SlotEnd::Completed)?;
        Ok(KvSessionFinish::Completed(Box::new(outcome)))
    }

    /// Drive the session to its end with `liveness`: every remaining prefill chunk, every
    /// decode step and `finish`. Stops at the first cancellation.
    pub fn run_to_end(
        &mut self,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<KvSessionFinish, ReferenceKvError> {
        while self.phase == KvSessionPhase::Prefilling {
            if let KvSessionStep::Cancelled(cancelled) = self.prefill_chunk(liveness)? {
                return Ok(KvSessionFinish::Cancelled(cancelled));
            }
        }
        while self.phase == KvSessionPhase::Decoding {
            if let KvSessionStep::Cancelled(cancelled) = self.decode_step(liveness)? {
                return Ok(KvSessionFinish::Cancelled(cancelled));
            }
        }
        self.finish(liveness)
    }

    /// Re-run a cancelled **default-binding** session's request cold: no lookup, no insert,
    /// nothing carried over from the cancelled attempt. The outcome is bitwise equal
    /// (generated tokens, logits digest, run) to a fresh cold run of the same request and
    /// carries `restarted_after_cancel`. The request id is recorded in
    /// [`KvSessionStats::restarted_requests`].
    ///
    /// A **bound** session is refused with [`BindingError::RestartNeedsOracle`] and stays
    /// cancelled: its restart re-resolves the binding and is guarded, so it needs the oracle
    /// the session was driven with, which the session deliberately does not hold (holding a
    /// borrow across steps would pin the caller's oracle for the session's lifetime). Use
    /// [`Self::restart_with`].
    pub fn restart(&mut self) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        self.require(KvSessionPhase::Cancelled, "restart")?;
        if self.layered.is_some() {
            return Err(BindingError::RestartNeedsOracle.into());
        }
        let liveness = self.backend.own_liveness();
        self.restart_with(&liveness)
    }

    /// Re-run a cancelled session cold. A bound session re-resolves its binding against
    /// `liveness` (a superseded adapter moves to its newest live successor, a revoked one to
    /// the base model, a revoked tracked expert is masked and a superseded one moves to its
    /// live successor generation) and runs cold on it, **guarded by the same oracle**: before
    /// every prefill chunk, every token and at finish the resolved binding's base model,
    /// adapter and tracked experts are re-checked. A revocation or fence during the restart
    /// cancels that attempt (nothing it produced is returned), the binding is re-resolved
    /// and the run restarted, up to `MAX_RESTART_ATTEMPTS` (3) attempts in total; then
    /// [`ReferenceKvError::RestartCancelled`] is returned. The outcome records the binding
    /// ([`ReferenceKvOutcome::binding`]). Segment dependencies are not re-checked by a
    /// restart. A default-binding session restarts exactly as [`Self::restart`].
    pub fn restart_with(
        &mut self,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<ReferenceKvOutcome, ReferenceKvError> {
        self.require(KvSessionPhase::Cancelled, "restart")?;
        self.release_pin()?;
        {
            let mut state = self.backend.lock()?;
            state.stats.sessions_restarted += 1;
            state
                .stats
                .restarted_requests
                .insert(self.request.request_id().clone());
        }
        if self.layered.is_none() {
            self.cache = self.backend.new_cache(&self.scope, &self.chain)?;
            self.reset_for_restart(Vec::new());
            return match self.run_to_end(&NoCache)? {
                KvSessionFinish::Completed(outcome) => Ok(*outcome),
                KvSessionFinish::Cancelled(_) => Err(ReferenceKvError::SessionPhase {
                    operation: "restart",
                    phase: KvSessionPhase::Cancelled,
                }),
            };
        }
        let mut last = None;
        for _ in 0..MAX_RESTART_ATTEMPTS {
            let (binding, record) =
                resolve_binding(&self.binding, self.backend.stack.config(), liveness)?;
            let layered = LayeredState::prepare(self.backend, &binding, &self.regime)?;
            self.cache = layered.fresh_cache(self.backend, &self.scope)?;
            self.chain = layered.chain;
            let guards = layered.session_dependencies();
            self.binding = binding;
            self.record = record;
            self.layered = Some(layered);
            self.reset_for_restart(guards);
            match self.run_to_end(liveness)? {
                KvSessionFinish::Completed(outcome) => return Ok(*outcome),
                KvSessionFinish::Cancelled(cancelled) => last = Some(cancelled),
            }
        }
        match last {
            Some(last) => Err(ReferenceKvError::RestartCancelled {
                attempts: MAX_RESTART_ATTEMPTS,
                last,
            }),
            None => Err(ReferenceKvError::SessionPhase {
                operation: "restart",
                phase: KvSessionPhase::Cancelled,
            }),
        }
    }

    /// Clear the run state for a cold attempt guarded by `guards` (and by nothing else: the
    /// lineage is not re-checked by a restart).
    fn reset_for_restart(&mut self, guards: Vec<KvDependency>) {
        self.cache_use = CacheUse::Disabled;
        self.lineage_guarded = false;
        self.binding_guards = guards;
        self.hit_tokens = 0;
        self.position = 0;
        self.last_logits = Vec::new();
        self.generated = Vec::new();
        self.hasher = new_hasher();
        self.work = Work::default();
        self.restarted = true;
        self.phase = KvSessionPhase::Prefilling;
    }

    /// Ask the oracle about the lineage (outside any lock) and the catalog about the pin;
    /// cancel on a failure.
    fn check_live(
        &mut self,
        liveness: &dyn KvLivenessOracle,
    ) -> Result<Option<KvCancellation>, ReferenceKvError> {
        let lineage_dead = self.lineage_guarded
            && !self
                .dependencies
                .iter()
                .all(|dependency| liveness.is_live(dependency));
        let binding_dead = self
            .binding_guards
            .iter()
            .any(|dependency| !liveness.is_live(dependency));
        let reason = if lineage_dead || binding_dead {
            Some(KvCancelReason::InputNotLive)
        } else {
            let mut pins: Vec<KvPin> = self.pin.iter().cloned().collect();
            if let Some(layered) = &self.layered {
                pins.extend(layered.all_pins());
            }
            if pins.is_empty() {
                None
            } else {
                let state = self.backend.lock()?;
                if pins.iter().all(|pin| state.catalog.pin_is_live(pin)) {
                    None
                } else {
                    Some(KvCancelReason::PrefixTombstoned)
                }
            }
        };
        let Some(reason) = reason else {
            return Ok(None);
        };
        let cancellation = KvCancellation {
            request_id: self.request.request_id().clone(),
            reason,
            at_phase: self.phase,
            tokens_emitted: self.generated.len(),
            prefilled_tokens: self.work.prefilled,
        };
        self.release_pin()?;
        self.phase = KvSessionPhase::Cancelled;
        let mut state = self.backend.lock()?;
        state.stats.sessions_cancelled += 1;
        state
            .stats
            .cancelled_requests
            .insert(self.request.request_id().clone());
        Ok(Some(cancellation))
    }

    fn release_pin(&mut self) -> Result<(), ReferenceKvError> {
        let mut pins: Vec<KvPin> = self.pin.take().into_iter().collect();
        if let Some(layered) = self.layered.as_mut() {
            pins.extend(std::mem::take(&mut layered.forked_pins));
            // Queued adoptions are dropped with their witnesses unused.
            pins.extend(
                std::mem::take(&mut layered.pending)
                    .into_iter()
                    .map(|pending| pending.into_pin()),
            );
        }
        if pins.is_empty() {
            return Ok(());
        }
        let mut state = self.backend.lock()?;
        for pin in &pins {
            state.catalog.release(pin);
        }
        state.pins.remove(&self.id);
        self.backend.drain_retirements(&mut state)
    }

    /// Adopt the live layers queued by a partial hit into the fully computed cache (checked
    /// mode) once the whole prompt is prefilled.
    fn adopt_layers(&mut self, liveness: &dyn KvLivenessOracle) -> Result<(), ReferenceKvError> {
        let Some(layered) = self.layered.as_mut() else {
            return Ok(());
        };
        if layered.pending.is_empty() {
            return Ok(());
        }
        let backend = self.backend;
        let mut state = backend.lock()?;
        let mismatches_before = layered.report.adoption_mismatches;
        adopt_pending(backend, &mut state, &mut self.cache, layered, liveness);
        state.stats.adoption_mismatches +=
            (layered.report.adoption_mismatches - mismatches_before) as u64;
        let remaining = layered.all_pins();
        if remaining.is_empty() {
            state.pins.remove(&self.id);
        } else {
            state.pins.insert(self.id, remaining);
        }
        backend.drain_retirements(&mut state)
    }

    fn close_slot(&mut self, end: SlotEnd) -> Result<(), ReferenceKvError> {
        if !self.slot_open {
            return Ok(());
        }
        self.slot_open = false;
        let mut state = self.backend.lock()?;
        state.open_sessions = state.open_sessions.saturating_sub(1);
        if end == SlotEnd::Completed {
            state.stats.sessions_completed += 1;
        }
        Ok(())
    }

    /// Publish the cache's full pages and offer them to the catalog. A refusal that only
    /// means "not cached" is recorded (capacity exhaustion is counted); anything else is an
    /// error.
    fn offer_boundary(&mut self, liveness: &dyn KvLivenessOracle) -> Result<(), ReferenceKvError> {
        let backend = self.backend;
        if let Some(layered) = self.layered.as_mut() {
            let offered = offer_layers(
                backend,
                &mut self.cache,
                layered,
                &self.request,
                &self.salt,
                self.binding.adapter.as_ref(),
                liveness,
            )?;
            match offered.refusal {
                None => {
                    if offered.inserted_entries > 0 {
                        self.work.inserted += 1;
                        layered.report.inserted_layer_entries += offered.inserted_entries;
                    }
                }
                Some(error) => self.work.refusals.push(error),
            }
            return Ok(());
        }
        let prefix = backend.paged.publish_prefix(&mut self.cache)?;
        let key = prefix.key.clone();
        // Per-page charges: each unique page digest is charged once with its real bytes.
        let pages = backend
            .paged
            .prefix_page_bytes(&prefix)?
            .into_iter()
            .map(|(digest, bytes)| KvPageCharge { digest, bytes })
            .collect();
        let mut state = backend.lock()?;
        let evictions_before = state.catalog.stats().evictions;
        let result = state.catalog.insert_paged(
            KvPagedInsert {
                salt: &self.salt,
                chain: &self.chain,
                request: &self.request,
                prefix,
                pages,
                extra_dependencies: Vec::new(),
            },
            liveness,
        );
        let evicted = state.catalog.stats().evictions - evictions_before;
        if evicted > 0 && !state.pins.is_empty() {
            state.stats.evictions_with_active_fork_pinned += evicted;
        }
        audit_pins(&mut state);
        match result {
            Ok(()) => {
                self.work.inserted += 1;
                state.held.insert(key);
            }
            Err(
                error @ (KvCatalogError::CapacityExhausted(_)
                | KvCatalogError::SlotDraining
                | KvCatalogError::DuplicateEntry
                | KvCatalogError::DependencyNotLive),
            ) => {
                if matches!(error, KvCatalogError::CapacityExhausted(_)) {
                    state.stats.capacity_exhausted_inserts += 1;
                }
                self.work.refusals.push(error);
                if !state.held.contains(&key) {
                    match backend.paged.retire_prefix(&key) {
                        Ok(()) | Err(KvBackendError::PrefixNotFound) => {}
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            Err(error) => return Err(ReferenceKvError::Catalog(error)),
        }
        backend.drain_retirements(&mut state)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotEnd {
    Completed,
    Dropped,
}

impl Drop for KvSession<'_> {
    /// An abandoned session releases its pin and its slot; errors (a poisoned lock) are
    /// ignored because there is nobody to report them to.
    fn drop(&mut self) {
        let _ = self.release_pin();
        let _ = self.close_slot(SlotEnd::Dropped);
    }
}
