//! Per-request stack bindings and layer-scoped KV reuse for the reference backend
//! (experiment R009 / R010 substrate; WP4).
//!
//! A [`StackBinding`] says what a request runs on top of the base model: an optional
//! [`Qwen2Adapter`] (LoRA, QOnly, MLP-only or aLoRA) and an optional MoE with the tracked
//! experts the request depends on. [`ReferenceKvReuseBackend::begin_session_bound`]
//! (crate::ReferenceKvReuseBackend) opens a [`KvSession`](crate::KvSession) on it. The default
//! binding is the plain base model and takes the unchanged whole-stack path.
//!
//! # Per-layer entries
//!
//! A bound session derives the chain of every layer
//! ([`ptr_pods::derive_adapted_kv_layer_chains_in_regime`]; for aLoRA the effective chain of a
//! prefix is the base chain only while the prefix ends at or before the invocation) and keeps
//! one catalog entry per layer ([`KvPrefixCatalog::insert_layers`](crate::KvPrefixCatalog)),
//! each with its own dependency set:
//!
//! * the base model and the request's bound segments on every layer;
//! * the adapter only on layers whose effective chain differs from the base chain, so a LoRA
//!   starting at layer `f` is a dependency of the entries `l >= f` (`l >= f + 1` for Q-only and
//!   MLP-only, whose delta reaches a later layer's K/V through the residual stream);
//! * every tracked expert `(j, i)` on every layer `l > j` (the conservative rule: layers `<= j`
//!   do not depend on layer `j`'s MLP).
//!
//! # Assembly
//!
//! * **Full hit** (every layer live at one common depth `D < prompt`): the per-layer prefixes
//!   are assembled with `fork_prefix_layers_admitted` and only the suffix is prefilled. For
//!   aLoRA the base-chain entries may be forked past the invocation: the fork is truncated to
//!   the invocation, so the boundary page keeps its pre-invocation rows physically shared with
//!   the base pages and no post-invocation row is ever read from a base-keyed entry.
//! * **Partial hit** (some layers live, owner decision 2): the stack has no residual cache, so
//!   the request is **computed in full** and the live layers are then **adopted** in checked
//!   mode (`adopt_prefix_layers_admitted`: every row is compared bitwise before the shared
//!   allocation is swapped in). Peak memory is transiently higher than a true partial
//!   computation; the report records shared and computed layer and row counts as integers.
//! * **Batch-variant routing** ([`RoutingDeterminism::BatchVariant`]) refuses Exact reuse:
//!   no lookup, no insert, the report is tagged Approximate and an `ExactOnly` policy is
//!   refused with [`BindingError::ApproximateNotAllowed`]. Its output never reaches the Exact
//!   catalog.
//!
//! # Restart re-resolves the binding (owner decision 3)
//!
//! `restart` re-resolves the binding against live state: a superseded adapter moves to the
//! newest live successor generation, a revoked one drops to the base model (no adapter), a
//! revoked tracked expert is masked from routing and a superseded one moves to its live
//! successor generation. The restart runs cold on that binding, is bitwise equal to a cold
//! run of it, and records the re-resolved binding in the outcome.
//!
//! The restart is **guarded**: it uses no cache and pins nothing, but before every prefill
//! chunk, every token and at finish it asks the oracle it was given
//! ([`KvSession::restart_with`](crate::KvSession); a bound session has no oracle-less
//! restart, so a restart can never silently use a different oracle than the run it
//! replaces) about the resolved binding's base model, adapter and tracked experts. A
//! revocation or fence during the restart cancels that attempt, nothing it produced is
//! returned, and the binding is re-resolved and the run restarted, up to
//! [`MAX_RESTART_ATTEMPTS`](crate::kv_infer_session) attempts, after which
//! `ReferenceKvError::RestartCancelled` is returned. Segment (lineage) dependencies are not
//! re-checked by a restart, as before: a revoked input is the caller's policy question.
//!
//! # Tracked experts and caller-trusted layering
//!
//! A MoE does not carry the project and pod identity a lifecycle fence names, so the binding
//! must track **every live expert of every MoE layer** ([`MoeBinding::experts`]); an
//! omitted live expert is refused with [`BindingError::UntrackedLiveExpert`]. The layered
//! fork and adopt requests of `ptr-pods` trust the caller's per-layer assignment: this
//! module picks every layer's entry by that layer's own chain (the adapted chain, or the
//! base chain for an unadapted layer or a prefix at or before an aLoRA invocation) and
//! never hands the backend a prefix chosen by anything else.

use crate::kv_infer::{
    ReferenceCache, ReferenceKvError, ReferenceKvReuseBackend, ReferenceKvScope, ReuseState,
};
use crate::{
    KvCatalogError, KvCatalogHit, KvChainId, KvDependency, KvLayerEntry, KvLayerQuery,
    KvLayeredInsert, KvLayeredLookup, KvLivenessOracle, KvPageCharge, KvPin, KvPrefixCatalog,
    KvRequestPositions, KvSalt, KV_CHUNK_TOKENS,
};
use ptr_model_api::SegmentedModelRequest;
use ptr_pods::{
    derive_adapted_kv_layer_chains_in_regime, derive_kv_layer_chains_in_regime, KvAdaptedChains,
    KvAdapterLayering, KvAdoptError, KvBackendError, KvChainIdentity, KvForkWitness,
    KvLayerAdoptRequest, KvLayerForkRequest, KvLayerPublishRequest, KvLayerRange, KvSharedPrefix,
    KvTensorBackend, PagedKvTensorBackend, Qwen2Adapter, Qwen2AdapterKind, Qwen2Moe, Qwen2MoeError,
    Qwen2PagedStack, Qwen2RopeRegime, Qwen2StackConfig, RoutingDeterminism,
};
use ptr_types::{AdapterVersion, Digest, Generation, PodId, ProjectId};
use sha2::{Digest as _, Sha256};
use std::fmt;

const ADAPTER_DIGEST_DOMAIN: &[u8] = b"ptr.kv.bound-adapter.v1\0";
const EXPERT_DIGEST_DOMAIN: &[u8] = b"ptr.kv.bound-expert.v1\0";

/// Why a binding was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BindingError {
    /// A tracked expert names a layer or index the MoE does not have.
    UnknownExpert {
        layer: usize,
        index: usize,
    },
    /// The same `(layer, index)` is tracked twice.
    DuplicateExpert {
        layer: usize,
        index: usize,
    },
    /// A dependency of the binding (base model, adapter or expert) is not live at begin.
    NotLive {
        identity: String,
    },
    /// The binding routes batch-variantly, which can only be reported Approximate, and the
    /// request's policy is `ExactOnly`.
    ApproximateNotAllowed,
    /// A live expert of a MoE layer is not among the tracked experts, so a revocation or
    /// fence of it would not invalidate the entries that depend on it. The MoE carries no
    /// project or pod identity, so the binding must name every live expert; mask an expert
    /// that should not be depended on.
    UntrackedLiveExpert {
        layer: usize,
        index: usize,
    },
    /// The stack derived no chain for a layer (a chain-derivation bug, refused instead of
    /// keying the layer under a placeholder).
    MissingLayerChain {
        layer: usize,
    },
    /// A bound session was restarted through [`KvSession::restart`](crate::KvSession), which
    /// has no oracle; its restart must be guarded by the oracle the session was driven with,
    /// so use `restart_with`.
    RestartNeedsOracle,
    Moe(Qwen2MoeError),
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownExpert { layer, index } => {
                write!(
                    f,
                    "binding tracks expert ({layer}, {index}), which the MoE lacks"
                )
            }
            Self::DuplicateExpert { layer, index } => {
                write!(f, "binding tracks expert ({layer}, {index}) twice")
            }
            Self::NotLive { identity } => {
                write!(f, "binding dependency {identity} is not live")
            }
            Self::ApproximateNotAllowed => write!(
                f,
                "batch-variant routing is Approximate and the policy is ExactOnly"
            ),
            Self::UntrackedLiveExpert { layer, index } => write!(
                f,
                "live expert ({layer}, {index}) is not tracked by the binding"
            ),
            Self::MissingLayerChain { layer } => {
                write!(f, "no KV chain was derived for layer {layer}")
            }
            Self::RestartNeedsOracle => write!(
                f,
                "a bound session restarts with restart_with and the oracle it was driven with"
            ),
            Self::Moe(error) => write!(f, "moe: {error}"),
        }
    }
}

impl std::error::Error for BindingError {}

/// An expert the request depends on: its lifecycle identity is `expert:{project}/{pod}`, the
/// generation and weights come from the binding's [`Qwen2Moe`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundExpert {
    pub layer: usize,
    pub index: usize,
    pub project: ProjectId,
    pub pod: PodId,
    /// The expert's fence epoch recorded in its dependency.
    pub fence_epoch: u64,
}

/// The MoE a request runs on (generations, live masks and routing mode included) and the
/// experts whose lifecycle it tracks. `experts` must cover **every live expert of every MoE
/// layer** (a masked expert needs no tracking); a binding that omits one is refused with
/// [`BindingError::UntrackedLiveExpert`], because the MoE does not carry the project and pod
/// identity a lifecycle fence names, so the runtime cannot derive the dependency itself.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeBinding {
    pub moe: Qwen2Moe,
    pub experts: Vec<BoundExpert>,
}

/// A later generation of a tracked expert a restart may move to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertSuccessor {
    pub layer: usize,
    pub index: usize,
    pub generation: u64,
    /// The successor's own fence epoch (the expert's transition epoch at the commit that
    /// admitted this generation, `0` for an expert that was never fenced). It is NOT inherited
    /// from the superseded binding: a binding recorded at a non-zero epoch is judged by the
    /// registry state version, so a successor carrying the old epoch could never be live.
    pub fence_epoch: u64,
}

/// Candidates a restart re-resolves a revoked or superseded binding to. Empty means a revoked
/// adapter falls to the base model and a revoked expert is masked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BindingSuccessors {
    /// Later generations of the bound adapter (same identity, higher generation).
    pub adapters: Vec<Qwen2Adapter>,
    pub experts: Vec<ExpertSuccessor>,
}

/// What a request binds on top of the base model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StackBinding {
    pub adapter: Option<Qwen2Adapter>,
    pub moe: Option<MoeBinding>,
    pub successors: BindingSuccessors,
    /// Keep per-layer entries even without an adapter or MoE, so that bound requests can
    /// share the base model's lower layers with this one ([`StackBinding::base_layered`]).
    /// An adapter or MoE implies per-layer entries and ignores this flag.
    pub layer_scoped: bool,
}

impl StackBinding {
    /// The plain base model (the unchanged whole-stack path).
    pub fn base() -> Self {
        Self::default()
    }

    /// The plain base model with per-layer catalog entries, the producer side of base to
    /// adapter sharing. Its output is bitwise the default binding's.
    pub fn base_layered() -> Self {
        Self {
            layer_scoped: true,
            ..Self::default()
        }
    }

    pub fn with_adapter(adapter: Qwen2Adapter) -> Self {
        Self {
            adapter: Some(adapter),
            ..Self::default()
        }
    }

    pub fn with_moe(moe: MoeBinding) -> Self {
        Self {
            moe: Some(moe),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_successors(mut self, successors: BindingSuccessors) -> Self {
        self.successors = successors;
        self
    }

    pub(crate) fn is_default(&self) -> bool {
        self.adapter.is_none() && self.moe.is_none() && !self.layer_scoped
    }
}

/// The binding a run actually executed under.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingRecord {
    /// Identity and generation of the adapter, `None` for the base model.
    pub adapter: Option<(String, u64)>,
    pub routing: RoutingDeterminism,
    /// Every expert masked from routing in the effective MoE.
    pub masked_experts: Vec<(usize, usize)>,
    /// `(layer, index, generation)` of every tracked expert.
    pub expert_generations: Vec<(usize, usize, u64)>,
    /// The record was produced by a restart's re-resolution.
    pub re_resolved: bool,
    /// The re-resolution found the adapter revoked and fell back to the base model.
    pub adapter_revoked_to_base: bool,
    /// `(from, to)` generation when the re-resolution moved to a successor adapter.
    pub adapter_superseded: Option<(u64, u64)>,
    /// Experts the re-resolution masked.
    pub experts_masked_by_resolution: Vec<(usize, usize)>,
    /// `(layer, index, from, to)` experts the re-resolution moved to a successor generation.
    pub experts_superseded: Vec<(usize, usize, u64, u64)>,
}

impl Default for BindingRecord {
    fn default() -> Self {
        Self {
            adapter: None,
            routing: RoutingDeterminism::NotRouted,
            masked_experts: Vec::new(),
            expert_generations: Vec::new(),
            re_resolved: false,
            adapter_revoked_to_base: false,
            adapter_superseded: None,
            experts_masked_by_resolution: Vec::new(),
            experts_superseded: Vec::new(),
        }
    }
}

impl BindingRecord {
    fn of(binding: &StackBinding, config: &Qwen2StackConfig, routing: RoutingDeterminism) -> Self {
        let mut masked = Vec::new();
        let mut generations = Vec::new();
        if let Some(bound) = &binding.moe {
            for layer in 0..config.num_layers {
                if let Some(moe_layer) = bound.moe.layer(layer) {
                    for (index, expert) in moe_layer.experts.iter().enumerate() {
                        if !expert.live {
                            masked.push((layer, index));
                        }
                    }
                }
            }
            for tracked in &bound.experts {
                if let Some(expert) = bound
                    .moe
                    .layer(tracked.layer)
                    .and_then(|layer| layer.experts.get(tracked.index))
                {
                    generations.push((tracked.layer, tracked.index, expert.generation));
                }
            }
        }
        Self {
            adapter: binding
                .adapter
                .as_ref()
                .map(|adapter| (adapter.identity().to_owned(), adapter.generation())),
            routing,
            masked_experts: masked,
            expert_generations: generations,
            ..Self::default()
        }
    }
}

/// How the prefix of a bound request was assembled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerAssembly {
    /// Nothing was served from the catalog.
    None,
    /// Every layer was live at one depth: the prefixes were forked and only the suffix was
    /// prefilled.
    Fork,
    /// Some layers were live: the request was computed in full and the live layers adopted
    /// in checked mode.
    ComputeThenAdopt,
}

/// Integer accounting of one bound run's layer-scoped reuse. Fractions are numerators and
/// denominators (`shared_rows / total_rows`, `shared_layers / layer_count`), never floats.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayeredReuseReport {
    pub assembly: LayerAssembly,
    pub layer_count: usize,
    /// Layers whose prefix rows are physically shared with the catalog (forked or adopted).
    pub shared_layers: usize,
    /// `layer_count - shared_layers`: layers with no shared prefix.
    pub computed_layers: usize,
    /// Rows served from the catalog, summed over layers.
    pub shared_rows: usize,
    /// `total_rows - shared_rows`.
    pub computed_rows: usize,
    /// `layer_count * prompt_tokens`.
    pub total_rows: usize,
    /// Rows compared bitwise by checked adoption, summed over adopted layers.
    pub adopted_compared_rows: usize,
    /// Layer pages that now share the published allocation.
    pub adopted_swapped_layer_pages: usize,
    /// Rows a checked adoption found different (must stay zero for an Exact stack).
    pub adoption_mismatches: usize,
    /// Pending adoptions dropped because the entry died or the backend refused.
    pub adoptions_skipped: usize,
    /// Pre-invocation rows of the aLoRA boundary page kept from a base prefix (0 otherwise).
    pub boundary_rows: usize,
    /// Per-layer catalog entries this run inserted.
    pub inserted_layer_entries: usize,
    /// Rows at positions at or after an aLoRA invocation that this run inserted under a
    /// base-only key of an adapted layer. Must be zero: such an entry is refused before it is
    /// published (see [`Self::post_invocation_entries_refused`]), so this never rises.
    pub post_invocation_rows_under_base_key: usize,
    /// Per-layer entries refused because they would have held post-invocation rows under a
    /// base-only key. Zero unless the chain derivation is wrong.
    pub post_invocation_entries_refused: usize,
    /// Rows in those refused entries.
    pub post_invocation_rows_refused: usize,
}

impl LayeredReuseReport {
    fn new(layer_count: usize) -> Self {
        Self {
            assembly: LayerAssembly::None,
            layer_count,
            shared_layers: 0,
            computed_layers: layer_count,
            shared_rows: 0,
            computed_rows: 0,
            total_rows: 0,
            adopted_compared_rows: 0,
            adopted_swapped_layer_pages: 0,
            adoption_mismatches: 0,
            adoptions_skipped: 0,
            boundary_rows: 0,
            inserted_layer_entries: 0,
            post_invocation_rows_under_base_key: 0,
            post_invocation_entries_refused: 0,
            post_invocation_rows_refused: 0,
        }
    }

    pub(crate) fn finalized(&self, prompt_tokens: usize) -> Self {
        let mut report = self.clone();
        report.computed_layers = report.layer_count - report.shared_layers.min(report.layer_count);
        report.total_rows = report.layer_count * prompt_tokens;
        report.computed_rows = report.total_rows.saturating_sub(report.shared_rows);
        report
    }
}

// Digests and dependencies ----------------------------------------------------------------

fn put_bits(hasher: &mut Sha256, values: &[f32]) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

/// The canonical bytes of an adapter's weights: a domain tag, then identity, generation,
/// kind, first layer, rank, scale and the exact bits of every pair, each variable part framed by
/// its length. This is the artifact of an adapter of this stack; its SHA-256 is the adapter's
/// content digest ([`adapter_content_digest`]) and the `artifact_sha256` a promotion must name.
pub(crate) fn adapter_canonical_bytes(adapter: &Qwen2Adapter, num_layers: usize) -> Vec<u8> {
    fn text(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    fn bits(out: &mut Vec<u8>, values: &[f32]) {
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(ADAPTER_DIGEST_DOMAIN);
    text(&mut out, adapter.identity());
    out.extend_from_slice(&adapter.generation().to_le_bytes());
    match adapter.kind() {
        Qwen2AdapterKind::Qkv => out.push(0),
        Qwen2AdapterKind::QOnly => out.push(1),
        Qwen2AdapterKind::MlpOnly => out.push(2),
        Qwen2AdapterKind::Activated { invocation } => {
            out.push(3);
            out.extend_from_slice(&(invocation as u64).to_le_bytes());
        }
    }
    out.extend_from_slice(&(adapter.first_layer() as u64).to_le_bytes());
    out.extend_from_slice(&(adapter.rank() as u64).to_le_bytes());
    out.extend_from_slice(&adapter.scale().to_bits().to_le_bytes());
    for layer in 0..num_layers {
        match adapter.layer(layer) {
            None => out.push(0),
            Some(pairs) => {
                out.push(1);
                for pair in [&pairs.q, &pairs.k, &pairs.v, &pairs.mlp] {
                    match pair {
                        None => out.push(0),
                        Some(pair) => {
                            out.push(1);
                            out.extend_from_slice(&(pair.rank as u64).to_le_bytes());
                            bits(&mut out, &pair.a);
                            bits(&mut out, &pair.b);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Digest of everything that identifies an adapter's behaviour: the SHA-256 of
/// [`adapter_canonical_bytes`].
pub(crate) fn adapter_content_digest(adapter: &Qwen2Adapter, num_layers: usize) -> Digest {
    Sha256::digest(adapter_canonical_bytes(adapter, num_layers)).into()
}

fn adapter_dependency(adapter: &Qwen2Adapter, num_layers: usize) -> KvDependency {
    KvDependency::adapter(
        adapter.identity(),
        Generation(adapter.generation()),
        adapter_content_digest(adapter, num_layers),
    )
}

fn expert_dependency(moe: &MoeBinding, tracked: &BoundExpert) -> Option<KvDependency> {
    let expert = moe.moe.layer(tracked.layer)?.experts.get(tracked.index)?;
    if !expert.live {
        return None;
    }
    let mut hasher = Sha256::new();
    hasher.update(EXPERT_DIGEST_DOMAIN);
    hasher.update((tracked.layer as u64).to_le_bytes());
    hasher.update((tracked.index as u64).to_le_bytes());
    hasher.update(expert.generation.to_le_bytes());
    for linear in [&expert.gate, &expert.up, &expert.down] {
        put_bits(&mut hasher, &linear.weight);
    }
    Some(KvDependency::expert_fenced(
        &tracked.project,
        &tracked.pod,
        Generation(expert.generation),
        hasher.finalize().into(),
        tracked.fence_epoch,
    ))
}

// Re-resolution ---------------------------------------------------------------------------

/// Re-resolve `binding` against `liveness` (see the module documentation).
pub(crate) fn resolve_binding(
    binding: &StackBinding,
    config: &Qwen2StackConfig,
    liveness: &dyn KvLivenessOracle,
) -> Result<(StackBinding, BindingRecord), BindingError> {
    let mut resolved = binding.clone();
    let mut adapter_revoked_to_base = false;
    let mut adapter_superseded = None;
    let mut masked_by_resolution = Vec::new();
    let mut experts_superseded = Vec::new();

    if let Some(adapter) = &binding.adapter {
        if !liveness.is_live(&adapter_dependency(adapter, config.num_layers)) {
            let mut candidates: Vec<&Qwen2Adapter> = binding
                .successors
                .adapters
                .iter()
                .filter(|candidate| {
                    candidate.identity() == adapter.identity()
                        && candidate.generation() > adapter.generation()
                })
                .collect();
            candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.generation()));
            let live = candidates.into_iter().find(|candidate| {
                liveness.is_live(&adapter_dependency(candidate, config.num_layers))
            });
            match live {
                Some(successor) => {
                    adapter_superseded = Some((adapter.generation(), successor.generation()));
                    resolved.adapter = Some(successor.clone());
                }
                None => {
                    adapter_revoked_to_base = true;
                    resolved.adapter = None;
                }
            }
        }
    }

    if let Some(bound) = &binding.moe {
        let mut moe = bound.moe.clone();
        let mut tracked_experts = bound.experts.clone();
        for tracked in &bound.experts {
            let Some(current) = expert_dependency(bound, tracked) else {
                continue; // already masked by the binding
            };
            if liveness.is_live(&current) {
                continue;
            }
            let mut successors: Vec<(u64, u64)> = binding
                .successors
                .experts
                .iter()
                .filter(|successor| {
                    successor.layer == tracked.layer
                        && successor.index == tracked.index
                        && successor.generation > current.generation.0
                })
                .map(|successor| (successor.generation, successor.fence_epoch))
                .collect();
            successors.sort_unstable_by(|a, b| b.cmp(a));
            let mut moved = false;
            for (generation, fence_epoch) in successors {
                let candidate = moe
                    .clone()
                    .with_expert_generation(config, tracked.layer, tracked.index, generation)
                    .map_err(BindingError::Moe)?;
                let moved_tracked = BoundExpert {
                    fence_epoch,
                    ..tracked.clone()
                };
                let candidate_binding = MoeBinding {
                    moe: candidate.clone(),
                    experts: Vec::new(),
                };
                let live = expert_dependency(&candidate_binding, &moved_tracked)
                    .is_some_and(|dependency| liveness.is_live(&dependency));
                if live {
                    // The resolved binding tracks the successor at its own epoch, so the
                    // dependencies of the entries it publishes name that epoch.
                    if let Some(entry) = tracked_experts
                        .iter_mut()
                        .find(|entry| entry.layer == tracked.layer && entry.index == tracked.index)
                    {
                        entry.fence_epoch = fence_epoch;
                    }
                    experts_superseded.push((
                        tracked.layer,
                        tracked.index,
                        current.generation.0,
                        generation,
                    ));
                    moe = candidate;
                    moved = true;
                    break;
                }
            }
            if !moved {
                moe = moe
                    .set_expert_live(tracked.layer, tracked.index, false)
                    .map_err(BindingError::Moe)?;
                masked_by_resolution.push((tracked.layer, tracked.index));
            }
        }
        resolved.moe = Some(MoeBinding {
            moe,
            experts: tracked_experts,
        });
    }

    let routing = routing_of(&resolved);
    let mut record = BindingRecord::of(&resolved, config, routing);
    record.re_resolved = true;
    record.adapter_revoked_to_base = adapter_revoked_to_base;
    record.adapter_superseded = adapter_superseded;
    record.experts_masked_by_resolution = masked_by_resolution;
    record.experts_superseded = experts_superseded;
    Ok((resolved, record))
}

fn routing_of(binding: &StackBinding) -> RoutingDeterminism {
    binding
        .moe
        .as_ref()
        .map_or(RoutingDeterminism::NotRouted, |bound| {
            bound.moe.routing_determinism()
        })
}

// Plan ------------------------------------------------------------------------------------

/// Per-layer chains of one request's stack and binding.
pub(crate) struct LayerPlan {
    pub(crate) root: [u8; 32],
    pub(crate) base: Vec<[u8; 32]>,
    pub(crate) adapted: Option<KvAdaptedChains>,
    /// First position the adapter acts at (`0` unless aLoRA).
    pub(crate) activation: usize,
}

impl LayerPlan {
    fn build(
        stack: &Qwen2PagedStack,
        identity: &KvChainIdentity,
        adapter: Option<&Qwen2Adapter>,
        regime: &Qwen2RopeRegime,
    ) -> Result<Self, ReferenceKvError> {
        let count = stack.config().num_layers;
        match adapter {
            None => {
                let chains = derive_kv_layer_chains_in_regime(stack, identity, regime)?;
                let base = (0..count)
                    .map(|l| {
                        chains
                            .layer(l)
                            .ok_or(BindingError::MissingLayerChain { layer: l })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self {
                    root: chains.root(),
                    base,
                    adapted: None,
                    activation: 0,
                })
            }
            Some(adapter) => {
                let chains =
                    derive_adapted_kv_layer_chains_in_regime(stack, identity, adapter, regime)?;
                let base = (0..count)
                    .map(|l| {
                        chains
                            .base()
                            .layer(l)
                            .ok_or(BindingError::MissingLayerChain { layer: l })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self {
                    root: chains.base().root(),
                    base,
                    activation: chains.activation_position(),
                    adapted: Some(chains),
                })
            }
        }
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.base.len()
    }

    /// The chain a lookup asks layer `layer` under (the adapted chain, or the base chain
    /// without an adapter).
    fn query_chain(&self, layer: usize) -> [u8; 32] {
        self.adapted
            .as_ref()
            .and_then(|chains| chains.adapted().layer(layer))
            .unwrap_or(self.base[layer])
    }

    /// The chain under which rows of `layer` for positions `0..covered` are stored.
    pub(crate) fn chain_at(&self, layer: usize, covered: usize) -> [u8; 32] {
        self.adapted
            .as_ref()
            .and_then(|chains| chains.effective_layer(layer, 0..covered))
            .unwrap_or(self.base[layer])
    }

    /// The layering of a cache whose adapted layers start where the effective chain of
    /// `0..covered` first differs from the base chain; `None` without an adapter.
    pub(crate) fn layering_at(&self, covered: usize) -> Option<KvAdapterLayering> {
        self.adapted.as_ref()?;
        let first = (0..self.layer_count())
            .find(|layer| self.chain_at(*layer, covered) != self.base[*layer])
            .unwrap_or(self.layer_count());
        Some(KvAdapterLayering {
            base_adapter: AdapterVersion::from(crate::kv_infer::BASE_ADAPTER_NAME),
            first_adapted_layer: u16::try_from(first).unwrap_or(u16::MAX),
        })
    }

    /// Every layer base-effective (prefixes keyed by the base adapter only).
    fn layering_base(&self) -> Option<KvAdapterLayering> {
        self.adapted.as_ref()?;
        Some(KvAdapterLayering {
            base_adapter: AdapterVersion::from(crate::kv_infer::BASE_ADAPTER_NAME),
            first_adapted_layer: u16::try_from(self.layer_count()).unwrap_or(u16::MAX),
        })
    }
}

/// A live per-layer hit queued for checked adoption at the end of prefill.
pub(crate) struct PendingAdopt {
    witness: KvForkWitness,
    prefix: KvSharedPrefix,
    pin: KvPin,
    base_keyed: bool,
    depth: usize,
}

impl PendingAdopt {
    pub(crate) fn into_pin(self) -> KvPin {
        self.pin
    }
}

/// Everything a bound session keeps beyond a default one.
pub(crate) struct LayeredState {
    /// The base stack with the binding's MoE, `None` when the binding has no MoE.
    pub(crate) stack: Option<Qwen2PagedStack>,
    pub(crate) plan: LayerPlan,
    /// Root chain, the id the cache binding's manifest is derived from.
    pub(crate) chain: KvChainId,
    pub(crate) adapter_version: Option<AdapterVersion>,
    base_dependency: KvDependency,
    adapter_dependency: Option<KvDependency>,
    /// `(layer, dependency)` of every tracked live expert.
    experts: Vec<(usize, KvDependency)>,
    pub(crate) routing: RoutingDeterminism,
    pub(crate) forked_pins: Vec<KvPin>,
    pub(crate) pending: Vec<PendingAdopt>,
    /// Depth per layer already present in the catalog (no insert below it).
    pub(crate) hit_depth: Vec<usize>,
    pub(crate) report: LayeredReuseReport,
}

impl LayeredState {
    pub(crate) fn prepare(
        backend: &ReferenceKvReuseBackend,
        binding: &StackBinding,
        regime: &Qwen2RopeRegime,
    ) -> Result<Self, ReferenceKvError> {
        let config = backend.stack.config();
        if let Some(bound) = &binding.moe {
            let mut seen = std::collections::BTreeSet::new();
            for tracked in &bound.experts {
                if bound
                    .moe
                    .layer(tracked.layer)
                    .and_then(|layer| layer.experts.get(tracked.index))
                    .is_none()
                {
                    return Err(BindingError::UnknownExpert {
                        layer: tracked.layer,
                        index: tracked.index,
                    }
                    .into());
                }
                if !seen.insert((tracked.layer, tracked.index)) {
                    return Err(BindingError::DuplicateExpert {
                        layer: tracked.layer,
                        index: tracked.index,
                    }
                    .into());
                }
            }
            // Every live expert of every MoE layer must be tracked: an untracked one would
            // feed the KV rows with no dependency to invalidate them by.
            for layer in 0..config.num_layers {
                let Some(moe_layer) = bound.moe.layer(layer) else {
                    continue;
                };
                for (index, expert) in moe_layer.experts.iter().enumerate() {
                    if expert.live && !seen.contains(&(layer, index)) {
                        return Err(BindingError::UntrackedLiveExpert { layer, index }.into());
                    }
                }
            }
        }
        let stack = match &binding.moe {
            Some(bound) => Some(backend.stack.clone().with_moe(bound.moe.clone())?),
            None => None,
        };
        let stack_ref = stack.as_ref().unwrap_or(&backend.stack);
        let identity = backend.chain_identity();
        let plan = LayerPlan::build(stack_ref, &identity, binding.adapter.as_ref(), regime)?;
        let layer_count = plan.layer_count();
        let experts = binding.moe.as_ref().map_or_else(Vec::new, |bound| {
            bound
                .experts
                .iter()
                .filter_map(|tracked| {
                    expert_dependency(bound, tracked).map(|dependency| (tracked.layer, dependency))
                })
                .collect()
        });
        Ok(Self {
            stack,
            chain: KvChainId(plan.root),
            adapter_version: binding.adapter.as_ref().map(|adapter| {
                AdapterVersion::from(&*format!("{}@{}", adapter.identity(), adapter.generation()))
            }),
            base_dependency: KvDependency::base_model(
                &identity.base_model,
                Generation(identity.generation),
                plan.root,
            ),
            adapter_dependency: binding
                .adapter
                .as_ref()
                .map(|adapter| adapter_dependency(adapter, config.num_layers)),
            experts,
            routing: routing_of(binding),
            plan,
            forked_pins: Vec::new(),
            pending: Vec::new(),
            hit_depth: vec![0; layer_count],
            report: LayeredReuseReport::new(layer_count),
        })
    }

    pub(crate) fn approximate(&self) -> bool {
        self.routing == RoutingDeterminism::BatchVariant
    }

    /// The binding's dependencies as a whole, for the session's liveness checks.
    pub(crate) fn session_dependencies(&self) -> Vec<KvDependency> {
        let mut dependencies = vec![self.base_dependency.clone()];
        dependencies.extend(self.adapter_dependency.clone());
        dependencies.extend(
            self.experts
                .iter()
                .map(|(_, dependency)| dependency.clone()),
        );
        dependencies
    }

    /// The dependencies of the entry of `layer` covering positions `0..covered`.
    pub(crate) fn layer_dependencies(&self, layer: usize, covered: usize) -> Vec<KvDependency> {
        let mut dependencies = vec![self.base_dependency.clone()];
        if let Some(adapter) = &self.adapter_dependency {
            if self.plan.chain_at(layer, covered) != self.plan.base[layer] {
                dependencies.push(adapter.clone());
            }
        }
        for (expert_layer, dependency) in &self.experts {
            if layer > *expert_layer {
                dependencies.push(dependency.clone());
            }
        }
        dependencies
    }

    pub(crate) fn all_pins(&self) -> Vec<KvPin> {
        self.forked_pins
            .iter()
            .cloned()
            .chain(self.pending.iter().map(|pending| pending.pin.clone()))
            .collect()
    }

    pub(crate) fn fresh_cache(
        &self,
        backend: &ReferenceKvReuseBackend,
        scope: &ReferenceKvScope,
    ) -> Result<ReferenceCache, ReferenceKvError> {
        backend.new_cache_with(scope, &self.chain, self.adapter_version.as_ref())
    }
}

// Assembly --------------------------------------------------------------------------------

fn lookup_hits(
    catalog: &mut KvPrefixCatalog,
    salt: &KvSalt,
    request: &SegmentedModelRequest,
    queries: &[KvLayerQuery],
    liveness: &dyn KvLivenessOracle,
) -> Vec<Option<KvCatalogHit>> {
    let positions = KvRequestPositions::Canonical;
    match catalog.lookup_layers(
        KvLayeredLookup {
            salt,
            request,
            positions: &positions,
            layers: queries,
        },
        liveness,
    ) {
        Ok(hits) => hits.into_iter().map(|hit| hit.hit).collect(),
        Err(_) => queries.iter().map(|_| None).collect(),
    }
}

fn uniform_depth(depths: &[Option<usize>]) -> Option<usize> {
    let first = (*depths.first()?)?;
    depths
        .iter()
        .all(|depth| *depth == Some(first))
        .then_some(first)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Source {
    Adapted,
    Base,
}

enum Decision {
    Miss,
    Fork {
        sources: Vec<Source>,
        depth: usize,
        base_keyed: bool,
        truncate_to: Option<usize>,
    },
    Adopt {
        picks: Vec<Option<Source>>,
    },
}

pub(crate) struct AssembleInput<'a> {
    pub(crate) backend: &'a ReferenceKvReuseBackend,
    pub(crate) scope: &'a ReferenceKvScope,
    pub(crate) salt: &'a KvSalt,
    pub(crate) request: &'a SegmentedModelRequest,
    pub(crate) total: usize,
    pub(crate) liveness: &'a dyn KvLivenessOracle,
}

/// Look the request up per layer and assemble what is live into `cache` (a fork) or queue it
/// for checked adoption. Returns the prompt tokens served by a fork (`0` for a miss or an
/// adoption).
pub(crate) fn assemble(
    input: AssembleInput<'_>,
    state: &mut ReuseState,
    cache: &mut ReferenceCache,
    layered: &mut LayeredState,
) -> Result<usize, ReferenceKvError> {
    let AssembleInput {
        backend,
        scope,
        salt,
        request,
        total,
        liveness,
    } = input;
    let layer_count = layered.plan.layer_count();
    let activation = layered.plan.activation;
    let alora = layered.plan.adapted.is_some() && activation > 0;
    let differs: Vec<bool> = (0..layer_count)
        .map(|layer| layered.plan.query_chain(layer) != layered.plan.base[layer])
        .collect();

    let queries_a: Vec<KvLayerQuery> = (0..layer_count)
        .map(|layer| KvLayerQuery {
            layer: layer as u16,
            chain: KvChainId(layered.plan.query_chain(layer)),
        })
        .collect();
    let mut adapted = lookup_hits(&mut state.catalog, salt, request, &queries_a, liveness);
    let mut base: Vec<Option<KvCatalogHit>> = (0..layer_count).map(|_| None).collect();

    let depths_a: Vec<Option<usize>> = adapted
        .iter()
        .map(|hit| hit.as_ref().map(KvCatalogHit::matched_tokens))
        .collect();
    let mut decision = Decision::Miss;
    if let Some(depth) = uniform_depth(&depths_a) {
        if depth < total {
            decision = Decision::Fork {
                sources: vec![Source::Adapted; layer_count],
                depth,
                base_keyed: false,
                truncate_to: None,
            };
        }
    }
    if matches!(decision, Decision::Miss) && alora {
        let queries_b: Vec<KvLayerQuery> = (0..layer_count)
            .filter(|layer| differs[*layer])
            .map(|layer| KvLayerQuery {
                layer: layer as u16,
                chain: KvChainId(layered.plan.base[layer]),
            })
            .collect();
        if !queries_b.is_empty() {
            let hits = lookup_hits(&mut state.catalog, salt, request, &queries_b, liveness);
            for (query, hit) in queries_b.iter().zip(hits) {
                base[usize::from(query.layer)] = hit;
            }
        }
        let depths_b: Vec<Option<usize>> = (0..layer_count)
            .map(|layer| {
                let hit = if differs[layer] {
                    base[layer].as_ref()
                } else {
                    adapted[layer].as_ref()
                };
                hit.map(KvCatalogHit::matched_tokens)
            })
            .collect();
        if let Some(depth) = uniform_depth(&depths_b) {
            if depth > 0 && depth < total {
                decision = Decision::Fork {
                    sources: differs
                        .iter()
                        .map(|differs| {
                            if *differs {
                                Source::Base
                            } else {
                                Source::Adapted
                            }
                        })
                        .collect(),
                    depth,
                    base_keyed: true,
                    truncate_to: (depth > activation).then_some(activation),
                };
            }
        }
    }
    if matches!(decision, Decision::Miss) {
        let picks: Vec<Option<Source>> = (0..layer_count)
            .map(|layer| {
                let depth_a = adapted[layer].as_ref().map(KvCatalogHit::matched_tokens);
                // A base-chain hit may be adopted only when every row of it precedes the
                // invocation (a deeper one holds base rows where the adapter acts).
                let depth_b = base[layer]
                    .as_ref()
                    .map(KvCatalogHit::matched_tokens)
                    .filter(|depth| *depth <= activation);
                match (depth_a, depth_b) {
                    (Some(a), Some(b)) if b > a => Some(Source::Base),
                    (Some(_), _) => Some(Source::Adapted),
                    (None, Some(_)) => Some(Source::Base),
                    (None, None) => None,
                }
            })
            .collect();
        if picks.iter().any(Option::is_some) {
            decision = Decision::Adopt { picks };
        }
    }

    let hit_tokens = match decision {
        Decision::Miss => 0,
        Decision::Fork {
            sources,
            depth,
            base_keyed,
            truncate_to,
        } => {
            let mut chosen = Vec::with_capacity(layer_count);
            for (layer, source) in sources.iter().enumerate() {
                let slot = match source {
                    Source::Adapted => &mut adapted[layer],
                    Source::Base => &mut base[layer],
                };
                if let Some(hit) = slot.take() {
                    chosen.push(hit);
                }
            }
            release_all(state, &mut adapted, &mut base);
            fork_chosen(
                backend,
                scope,
                state,
                cache,
                layered,
                chosen,
                (depth, base_keyed, truncate_to),
                liveness,
            )?
        }
        Decision::Adopt { picks } => {
            for (layer, pick) in picks.iter().enumerate() {
                let Some(source) = pick else { continue };
                let slot = match source {
                    Source::Adapted => &mut adapted[layer],
                    Source::Base => &mut base[layer],
                };
                let Some(hit) = slot.take() else { continue };
                let depth = hit.matched_tokens();
                let (witness, prefix, pin) = hit.into_parts();
                layered.hit_depth[layer] = depth;
                layered.pending.push(PendingAdopt {
                    witness,
                    prefix,
                    pin,
                    base_keyed: *source == Source::Base,
                    depth,
                });
            }
            release_all(state, &mut adapted, &mut base);
            layered.report.assembly = LayerAssembly::ComputeThenAdopt;
            0
        }
    };
    release_all(state, &mut adapted, &mut base);
    Ok(hit_tokens)
}

fn release_all(
    state: &mut ReuseState,
    adapted: &mut [Option<KvCatalogHit>],
    base: &mut [Option<KvCatalogHit>],
) {
    for slot in adapted.iter_mut().chain(base.iter_mut()) {
        if let Some(hit) = slot.take() {
            state.catalog.release(hit.pin());
        }
    }
}

fn fork_chosen(
    backend: &ReferenceKvReuseBackend,
    scope: &ReferenceKvScope,
    state: &mut ReuseState,
    cache: &mut ReferenceCache,
    layered: &mut LayeredState,
    chosen: Vec<KvCatalogHit>,
    (depth, base_keyed, truncate_to): (usize, bool, Option<usize>),
    liveness: &dyn KvLivenessOracle,
) -> Result<usize, ReferenceKvError> {
    let layer_count = layered.plan.layer_count();
    let mut all_live = true;
    for hit in &chosen {
        all_live &= state.catalog.revalidate_pin(hit.pin(), liveness);
    }
    if !all_live {
        for hit in &chosen {
            state.catalog.release(hit.pin());
        }
        return Ok(0);
    }
    let pins: Vec<KvPin> = chosen.iter().map(|hit| hit.pin().clone()).collect();
    let parts = chosen
        .into_iter()
        .map(|hit| {
            let (witness, prefix, _) = hit.into_parts();
            (prefix, witness)
        })
        .collect();
    let layering = if base_keyed {
        layered.plan.layering_base()
    } else {
        layered.plan.layering_at(depth)
    };
    match backend
        .paged
        .fork_prefix_layers_admitted(cache, KvLayerForkRequest { parts, layering })
    {
        Ok(()) => {
            let mut served = depth;
            if let Some(keep) = truncate_to {
                if let Err(error) = backend.paged.truncate(cache, keep) {
                    // The fork succeeded but cannot be cut to the invocation: nothing of it
                    // may be used, and the pins taken for it must not leak (the session was
                    // never created, so nobody else would release them).
                    for pin in &pins {
                        state.catalog.release(pin);
                    }
                    *cache = layered.fresh_cache(backend, scope)?;
                    return Err(error.into());
                }
                served = keep;
                layered.report.boundary_rows = keep % KV_CHUNK_TOKENS;
            }
            layered.forked_pins = pins;
            layered.hit_depth = vec![served; layer_count];
            layered.report.assembly = LayerAssembly::Fork;
            layered.report.shared_layers = layer_count;
            layered.report.shared_rows = layer_count * served;
            Ok(served)
        }
        Err(error) => {
            for pin in &pins {
                match error {
                    KvBackendError::PrefixNotFound => state.catalog.reject_missing_backend_pin(pin),
                    // A scope disagreement is a bug (or a corrupt entry), not a miss: count it.
                    KvBackendError::PrefixMismatch => state.catalog.reject_scope_mismatch_pin(pin),
                    _ => {
                        state.catalog.release(pin);
                    }
                }
            }
            *cache = layered.fresh_cache(backend, scope)?;
            Ok(0)
        }
    }
}

/// Adopt the queued live layers into the fully computed `cache` in checked mode, then
/// release their pins. A mismatch is counted (and leaves that layer unshared); it never
/// changes the output, which is the computed one.
pub(crate) fn adopt_pending(
    backend: &ReferenceKvReuseBackend,
    state: &mut ReuseState,
    cache: &mut ReferenceCache,
    layered: &mut LayeredState,
    liveness: &dyn KvLivenessOracle,
) {
    let pending = std::mem::take(&mut layered.pending);
    for adopt in pending {
        let PendingAdopt {
            witness,
            prefix,
            pin,
            base_keyed,
            depth,
        } = adopt;
        if !state.catalog.revalidate_pin(&pin, liveness) {
            state.catalog.release(&pin);
            layered.report.adoptions_skipped += 1;
            continue;
        }
        let layering = if base_keyed {
            layered.plan.layering_base()
        } else {
            layered.plan.layering_at(depth)
        };
        match backend.paged.adopt_prefix_layers_admitted(
            cache,
            KvLayerAdoptRequest {
                prefix,
                witness,
                layering,
            },
        ) {
            Ok(report) => {
                layered.report.adopted_compared_rows += report.compared_rows;
                layered.report.adopted_swapped_layer_pages += report.swapped_layer_pages;
                layered.report.shared_layers += 1;
                layered.report.shared_rows += depth;
                state.catalog.release(&pin);
            }
            Err(KvAdoptError::Mismatch {
                mismatched_rows, ..
            }) => {
                layered.report.adoption_mismatches += mismatched_rows;
                state.catalog.release(&pin);
            }
            Err(KvAdoptError::Backend(KvBackendError::PrefixNotFound)) => {
                layered.report.adoptions_skipped += 1;
                state.catalog.reject_missing_backend_pin(&pin);
            }
            Err(KvAdoptError::Backend(KvBackendError::PrefixMismatch)) => {
                layered.report.adoptions_skipped += 1;
                state.catalog.reject_scope_mismatch_pin(&pin);
            }
            Err(KvAdoptError::Backend(_)) => {
                layered.report.adoptions_skipped += 1;
                state.catalog.release(&pin);
            }
        }
    }
}

/// What offering one chunk boundary to the catalog did.
pub(crate) struct Offered {
    pub(crate) inserted_entries: usize,
    pub(crate) refusal: Option<KvCatalogError>,
}

/// Publish the cache's full pages layer by layer and offer them to the catalog as one
/// all-or-nothing per-layer insert. Layers already present (served by this run's own hit)
/// are not offered again.
pub(crate) fn offer_layers(
    backend: &ReferenceKvReuseBackend,
    cache: &mut ReferenceCache,
    layered: &mut LayeredState,
    request: &SegmentedModelRequest,
    salt: &KvSalt,
    adapter: Option<&Qwen2Adapter>,
    liveness: &dyn KvLivenessOracle,
) -> Result<Offered, ReferenceKvError> {
    let covered = (cache.sequence_length / KV_CHUNK_TOKENS) * KV_CHUNK_TOKENS;
    let layer_count = layered.plan.layer_count();
    let layering = layered.plan.layering_at(covered);
    let mut entries = Vec::new();
    let mut keys = Vec::new();
    let mut refused_rows = 0usize;
    let mut refused_entries = 0usize;
    for layer in 0..layer_count {
        if layered.hit_depth[layer] >= covered {
            continue;
        }
        let chain = layered.plan.chain_at(layer, covered);
        // Fail closed: rows at or after the invocation never go under a base-only key of an
        // adapted layer. The entry is skipped before anything is published for it.
        if let Some(adapter) = adapter {
            if adapter.adapts_kv()
                && adapter.layer(layer).is_some()
                && chain == layered.plan.base[layer]
                && covered > layered.plan.activation
            {
                refused_rows += covered - layered.plan.activation;
                refused_entries += 1;
                continue;
            }
        }
        let prefix = backend.paged.publish_prefix_layers(
            cache,
            KvLayerPublishRequest {
                layers: KvLayerRange::span(layer as u16, layer as u16 + 1)?,
                layering: layering.clone(),
            },
        )?;
        let pages = backend
            .paged
            .prefix_page_bytes(&prefix)?
            .into_iter()
            .map(|(digest, bytes)| KvPageCharge { digest, bytes })
            .collect();
        keys.push(prefix.key.clone());
        entries.push(KvLayerEntry {
            layer: layer as u16,
            chain: KvChainId(chain),
            prefix,
            pages,
            extra_dependencies: layered.layer_dependencies(layer, covered),
        });
    }
    layered.report.post_invocation_entries_refused += refused_entries;
    layered.report.post_invocation_rows_refused += refused_rows;
    if entries.is_empty() {
        return Ok(Offered {
            inserted_entries: 0,
            refusal: None,
        });
    }
    let count = entries.len();
    let mut state = backend.lock()?;
    let evictions_before = state.catalog.stats().evictions;
    let result = state.catalog.insert_layers(
        KvLayeredInsert {
            salt,
            request,
            layers: entries,
        },
        liveness,
    );
    let evicted = state.catalog.stats().evictions - evictions_before;
    if evicted > 0 && !state.pins.is_empty() {
        state.stats.evictions_with_active_fork_pinned += evicted;
    }
    crate::kv_infer_session::audit_pins(&mut state);
    let offered = match result {
        Ok(()) => {
            for key in keys {
                state.held.insert(key);
            }
            Offered {
                inserted_entries: count,
                refusal: None,
            }
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
            for key in keys {
                if !state.held.contains(&key) {
                    match backend.paged.retire_prefix(&key) {
                        Ok(()) | Err(KvBackendError::PrefixNotFound) => {}
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            Offered {
                inserted_entries: 0,
                refusal: Some(error),
            }
        }
        Err(error) => return Err(ReferenceKvError::Catalog(error)),
    };
    backend.drain_retirements(&mut state)?;
    Ok(offered)
}

impl From<BindingError> for ReferenceKvError {
    fn from(error: BindingError) -> Self {
        Self::Binding(error)
    }
}

pub(crate) fn record_of(binding: &StackBinding, config: &Qwen2StackConfig) -> BindingRecord {
    BindingRecord::of(binding, config, routing_of(binding))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReferenceKvLimits;
    use ptr_model_api::{PromptSegment, Tokenizer};
    use ptr_types::{PrincipalId, RequestId, Revision, SemanticContext};

    fn scope() -> ReferenceKvScope {
        ReferenceKvScope {
            principal: PrincipalId::from("alice"),
            project: ProjectId::from("p"),
        }
    }

    /// A failing truncation after a successful fork must release every pin the fork took:
    /// the session is never created, so nothing else would.
    #[test]
    fn a_failed_boundary_truncation_releases_the_fork_pins() {
        let backend = ReferenceKvReuseBackend::new(ReferenceKvLimits::default(), scope()).unwrap();
        let body: String = (0..70)
            .map(|i| char::from(b'a' + ((i * 7 + i / 16 * 5) % 26) as u8))
            .collect();
        let segment = PromptSegment::raw_from_text(backend.tokenizer(), &body).unwrap();
        let request = SegmentedModelRequest::new(
            RequestId::from("t"),
            SemanticContext {
                revision: Revision(1),
                entries: Vec::new(),
            },
            backend.tokenizer().identity().clone(),
            vec![segment],
        )
        .unwrap();
        backend
            .infer_scoped_bound(&scope(), &request, None, &StackBinding::base_layered())
            .unwrap();

        let liveness = backend.own_liveness();
        let regime = backend.admit_regime(70).unwrap();
        let mut layered =
            LayeredState::prepare(&backend, &StackBinding::base_layered(), &regime).unwrap();
        let mut cache = layered.fresh_cache(&backend, &scope()).unwrap();
        let salt = backend.salt(&scope());
        let queries: Vec<KvLayerQuery> = (0..layered.plan.layer_count())
            .map(|layer| KvLayerQuery {
                layer: layer as u16,
                chain: KvChainId(layered.plan.query_chain(layer)),
            })
            .collect();
        let mut state = backend.lock().unwrap();
        let hits: Vec<KvCatalogHit> =
            lookup_hits(&mut state.catalog, &salt, &request, &queries, &liveness)
                .into_iter()
                .map(|hit| hit.expect("every layer was inserted"))
                .collect();
        let pins: Vec<KvPin> = hits.iter().map(|hit| hit.pin().clone()).collect();
        assert!(pins.iter().all(|pin| state.catalog.pin_held(pin)));

        // Truncating beyond the sequence length is refused by the backend.
        let result = fork_chosen(
            &backend,
            &scope(),
            &mut state,
            &mut cache,
            &mut layered,
            hits,
            (64, false, Some(10_000)),
            &liveness,
        );
        assert!(matches!(result, Err(ReferenceKvError::Kv(_))), "{result:?}");
        assert!(pins.iter().all(|pin| !state.catalog.pin_held(pin)));
        assert!(layered.forked_pins.is_empty());
        assert_eq!(cache.sequence_length, 0);
    }
}
