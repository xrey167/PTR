//! The downstream side of a pipeline split, version 2: admission of version 2 activation
//! frames and a downstream cache keyed and revalidated by the frame's input lineage, raw
//! ancestry, mode and taint (doc 36, section D, "Dependency invalidation").
//!
//! This module is additive. Version 1 (`split_stage`) is unchanged: its types, its keys and its
//! paths behave exactly as before. What version 2 adds, all revalidated **at every use**
//! against what the runtime holds now, because a `Revoke` message can be dropped or delayed
//! and so is never the guarantee:
//!
//! * the whole closed upstream list of the frame's `upstream_chain` (stage, base model, every
//!   adapter, every expert with its fence epoch), through the version 1 [`StageLiveness`];
//! * every typed input of the frame's `input_lineage` (semantic, memory, derived: identity,
//!   generation, digest), through the new [`InputLiveness`];
//! * the downstream stage's own chain.
//!
//! The Raw ancestry has no generation to revalidate; it is part of the cache **key** instead
//! ([`prefix_key_v2`]), together with the digest of every frame, the mode and the taint, so an
//! earlier Raw segment removed or changed with the typed inputs, the chain and the positions
//! equal is a miss.
//!
//! **Exact and Approximate are two namespaces** of the cache: an Approximate frame only ever
//! populates and hits the Approximate namespace ([`DownstreamKey::namespace`] is part of the
//! key and picks the table), and the taint of the frame travels to the output that is
//! released from a hit ([`SplitStageCacheV2::finish_use`]), where the runtime's effect
//! boundary refuses an Approximate-tainted action.
//!
//! **Nothing persists.** There is no operation that writes an entry anywhere or reads one
//! back: [`SplitStageCacheV2::persist`] is the typed refusal of a snapshot, a tier write, a
//! persist or a restore, and a restart (a new cache, or [`SplitStageCacheV2::clear`]) drops
//! every entry and every open use.

use crate::placement::PlacementScope;
use crate::split_stage::{revalidate_chain, revalidate_chain_in, ChainFault, StageLiveness};
use crate::PtrRuntime;
use ptr_model_api::{KvReuseMode, KvReusePolicy, KvReuseTaint};
use ptr_protocol::activation::{
    chain_bytes, lineage_bytes, mode_taint_bytes, ActivationFrameV2, ChainElement, FrameMode,
    FrameTaint, InputLineage, LineageInput, LineageKind, SchemaError, UpstreamChain,
    MAX_LINEAGE_INPUTS,
};
use ptr_types::{Generation, ProjectId, Validity};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const KEY_DOMAIN: &[u8] = b"ptr.split.downstream-key.v2\0";
const PROJECT_KEY_DOMAIN: &[u8] = b"ptr.split.downstream-key.v2.project\0";

/// What the downstream stage asks about the typed inputs of a frame's lineage, at use.
pub trait InputLiveness {
    /// Whether the input `identity` of `kind` at `generation` may be used now, as the
    /// lifecycle authority sees it (`None`: unknown).
    fn validity(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<Validity>;
    /// The content digest committed for the input at `generation` (`None`: unknown).
    fn content_digest(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<[u8; 32]>;
}

/// The lifecycle target the runtime tracks a typed input under: the kind and the identity
/// together, so a semantic input and a memory input of the same identity are two targets (one
/// revoked leaves the other live).
pub fn input_target(kind: LineageKind, identity: &str) -> String {
    format!("{}:{identity}", kind.name())
}

/// [`InputLiveness`] over the runtime's committed lifecycle state. An input is the lifecycle
/// target [`input_target`] names. The runtime keeps no digest for an input generation, so the
/// host passes the digests it admitted each generation under, keyed by (kind, identity,
/// generation).
pub struct RuntimeInputLiveness<'a> {
    pub runtime: &'a PtrRuntime,
    pub digests: &'a BTreeMap<(LineageKind, String, u64), [u8; 32]>,
}

impl InputLiveness for RuntimeInputLiveness<'_> {
    fn validity(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<Validity> {
        self.runtime
            .generation_validity(&input_target(kind, identity), generation)
    }

    fn content_digest(
        &self,
        kind: LineageKind,
        identity: &str,
        generation: Generation,
    ) -> Option<[u8; 32]> {
        self.digests
            .get(&(kind, identity.to_owned(), generation.0))
            .copied()
    }
}

/// Why a version 2 frame, or an entry derived from one, is not usable now.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameFault {
    /// An element of the upstream chain, or its placement, is not usable (version 1 fault).
    Chain(ChainFault),
    /// A typed input of the lineage is not live (revoked, superseded, disputed or unknown).
    InputNotLive {
        kind: LineageKind,
        identity: String,
        standing: Option<Validity>,
    },
    /// A typed input's digest is not the one committed for its generation.
    InputDigestMismatch { kind: LineageKind, identity: String },
    /// The frame ran under an Approximate mode and the caller's policy admits Exact only.
    ApproximateNotAllowed,
    /// The frame's mode and taint disagree (a frame built in memory, not decoded).
    ModeTaintMismatch,
    /// The activations hold a NaN or an infinity, at this index of the tensor.
    NonFinite { index: usize },
    /// The frame, built in memory, would not decode: its bytes break the canonical lineage
    /// order, a bound or the tensor shape ([`ActivationFrameV2::validate`]).
    Malformed(SchemaError),
    /// The frame's positions lie wholly before the downstream KV state's next position: a
    /// stale or replayed frame.
    PositionsStale {
        expected_start: u64,
        first: u64,
        last: u64,
    },
    /// The frame starts before the downstream's next position but reaches it: it would
    /// overwrite KV entries the downstream already holds.
    PositionsOverlap { expected_start: u64, first: u64 },
    /// The frame starts after the downstream's next position: the positions in between were
    /// never received, so the KV state would have a hole.
    PositionsSkipped { expected_start: u64, first: u64 },
    /// The frame's own positions are not consecutive: `index` is the first row whose position
    /// is not its predecessor plus one.
    PositionsNotContiguous { index: usize },
}

impl From<ChainFault> for FrameFault {
    fn from(fault: ChainFault) -> Self {
        Self::Chain(fault)
    }
}

fn revalidate_input(input: &LineageInput, inputs: &dyn InputLiveness) -> Result<(), FrameFault> {
    let generation = Generation(input.generation);
    match inputs.validity(input.kind, &input.identity, generation) {
        Some(Validity::Live) => {}
        standing => {
            return Err(FrameFault::InputNotLive {
                kind: input.kind,
                identity: input.identity.clone(),
                standing,
            })
        }
    }
    if inputs.content_digest(input.kind, &input.identity, generation) != Some(input.digest) {
        return Err(FrameFault::InputDigestMismatch {
            kind: input.kind,
            identity: input.identity.clone(),
        });
    }
    Ok(())
}

/// The downstream stage's own chain: its stage, base model, adapters and experts, without a
/// placement (the frame's placement epoch belongs to the upstream stage).
fn revalidate_own(own: &UpstreamChain, live: &dyn StageLiveness) -> Result<(), FrameFault> {
    let revalidate = |element: &ChainElement| -> Result<(), FrameFault> {
        let generation = Generation(element.generation);
        match live.validity(&element.identity, generation) {
            Some(Validity::Live) => {}
            standing => {
                return Err(FrameFault::Chain(ChainFault::NotLive {
                    identity: element.identity.clone(),
                    standing,
                }))
            }
        }
        if live.manifest_digest(&element.identity, generation) != Some(element.digest) {
            return Err(FrameFault::Chain(ChainFault::DigestMismatch {
                identity: element.identity.clone(),
            }));
        }
        Ok(())
    };
    revalidate(&own.stage)?;
    revalidate(&own.base_model)?;
    for adapter in &own.adapters {
        revalidate(adapter)?;
    }
    // An own expert is fenced like an upstream one: a fence changes no generation validity, so
    // the recorded epoch must still be the committed one (fail closed on an unrecorded epoch).
    for expert in &own.experts {
        revalidate(expert)?;
        let current = live.fence_epoch(&expert.identity);
        if expert.fence_epoch == 0 || current != Some(expert.fence_epoch) {
            return Err(FrameFault::Chain(ChainFault::StaleFence {
                identity: expert.identity.clone(),
                recorded: expert.fence_epoch,
                current,
            }));
        }
    }
    Ok(())
}

fn revalidate_scoped(
    scope: &PlacementScope,
    chain: &UpstreamChain,
    placement_epoch: u64,
    live: &dyn StageLiveness,
) -> Result<(), FrameFault> {
    let verdict = match scope {
        PlacementScope::Legacy => revalidate_chain(chain, placement_epoch, live),
        PlacementScope::Project(project) => {
            revalidate_chain_in(project, chain, placement_epoch, live)
        }
    };
    verdict.map_err(FrameFault::Chain)
}

fn namespace_of(mode: FrameMode) -> KvReuseMode {
    match mode {
        FrameMode::Exact => KvReuseMode::Exact,
        FrameMode::Approximate => KvReuseMode::Approximate,
    }
}

fn taint_of(taint: FrameTaint) -> KvReuseTaint {
    match taint {
        FrameTaint::Exact => KvReuseTaint::Exact,
        FrameTaint::Approximate { report_digest } => KvReuseTaint::Approximate { report_digest },
    }
}

/// A version 2 frame whose chain, placement and lineage were live when it was admitted:
/// activations and positions to compute from, and nothing a cache could take for KV.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedFrameV2 {
    scope: PlacementScope,
    frame_digest: [u8; 32],
    chain: UpstreamChain,
    lineage: InputLineage,
    mode: FrameMode,
    taint: FrameTaint,
    placement_epoch: u64,
    rows: u32,
    width: u32,
    positions: Vec<u32>,
    activations: Vec<f32>,
}

impl AdmittedFrameV2 {
    /// The project the frame was admitted for (`None`: the legacy single-project namespace).
    pub fn project(&self) -> Option<&ProjectId> {
        self.scope.project()
    }

    pub fn frame_digest(&self) -> &[u8; 32] {
        &self.frame_digest
    }

    pub fn chain(&self) -> &UpstreamChain {
        &self.chain
    }

    pub fn lineage(&self) -> &InputLineage {
        &self.lineage
    }

    pub fn placement_epoch(&self) -> u64 {
        self.placement_epoch
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// `rows x width` activations, row-major.
    pub fn activations(&self) -> &[f32] {
        &self.activations
    }

    /// The cache namespace the frame belongs to: its reuse mode.
    pub fn namespace(&self) -> KvReuseMode {
        namespace_of(self.mode)
    }

    /// The taint every output derived from the frame carries.
    pub fn taint(&self) -> KvReuseTaint {
        taint_of(self.taint)
    }

    /// Revalidate the frame's chain, placement epoch and lineage, and `own` (the downstream
    /// stage's own chain the output is computed with), against `live` and `inputs` now.
    /// Admission checks once; a caller that computes from the frame on a miss calls this
    /// before it releases the computed output, so an invalidation of either side between
    /// admission and release is caught.
    pub fn revalidate(
        &self,
        own: &UpstreamChain,
        live: &dyn StageLiveness,
        inputs: &dyn InputLiveness,
    ) -> Result<(), FrameFault> {
        revalidate_scoped(&self.scope, &self.chain, self.placement_epoch, live)?;
        revalidate_own(own, live)?;
        for input in &self.lineage.inputs {
            revalidate_input(input, inputs)?;
        }
        Ok(())
    }
}

/// What a downstream stage asks to admit: the frame, the reuse policy and the position the
/// downstream KV state expects the frame to start at.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionRequestV2<'a> {
    pub frame: &'a ActivationFrameV2,
    pub policy: KvReusePolicy,
    /// The next token position of the downstream's KV state: the frame's positions must be
    /// exactly `expected_start`, `expected_start + 1`, ... (contiguous, no overlap, no skip,
    /// not stale).
    pub expected_start: u64,
}

fn check_positions(positions: &[u32], expected_start: u64) -> Result<(), FrameFault> {
    let (Some(&first), Some(&last)) = (positions.first(), positions.last()) else {
        return Ok(());
    };
    let (first, last) = (u64::from(first), u64::from(last));
    if last < expected_start {
        return Err(FrameFault::PositionsStale {
            expected_start,
            first,
            last,
        });
    }
    if first < expected_start {
        return Err(FrameFault::PositionsOverlap {
            expected_start,
            first,
        });
    }
    if first > expected_start {
        return Err(FrameFault::PositionsSkipped {
            expected_start,
            first,
        });
    }
    for (index, pair) in positions.windows(2).enumerate() {
        if u64::from(pair[1]) != u64::from(pair[0]) + 1 {
            return Err(FrameFault::PositionsNotContiguous { index: index + 1 });
        }
    }
    Ok(())
}

/// Admit a version 2 frame for compute: the whole upstream chain, the placement epoch and
/// every typed input of the lineage are revalidated now. A frame that ran under an
/// Approximate mode is admitted only when `policy` allows it, and then only into the
/// Approximate namespace. The activations must be finite.
///
/// The frame's positions must start at the request's `expected_start` and run consecutively
/// (see [`AdmissionRequestV2`]).
///
/// The placement epoch is read in the legacy single-project namespace; a caller that serves a
/// project uses [`admit_frame_v2_in`].
pub fn admit_frame_v2(
    request: AdmissionRequestV2<'_>,
    live: &dyn StageLiveness,
    inputs: &dyn InputLiveness,
) -> Result<AdmittedFrameV2, FrameFault> {
    admit_scoped(PlacementScope::Legacy, request, live, inputs)
}

/// [`admit_frame_v2`] for `project`: the placement epoch is read for that project.
pub fn admit_frame_v2_in(
    project: &ProjectId,
    request: AdmissionRequestV2<'_>,
    live: &dyn StageLiveness,
    inputs: &dyn InputLiveness,
) -> Result<AdmittedFrameV2, FrameFault> {
    admit_scoped(
        PlacementScope::Project(project.clone()),
        request,
        live,
        inputs,
    )
}

fn admit_scoped(
    scope: PlacementScope,
    request: AdmissionRequestV2<'_>,
    live: &dyn StageLiveness,
    inputs: &dyn InputLiveness,
) -> Result<AdmittedFrameV2, FrameFault> {
    let AdmissionRequestV2 {
        frame,
        policy,
        expected_start,
    } = request;
    if (frame.mode == FrameMode::Approximate) != frame.taint.is_approximate() {
        return Err(FrameFault::ModeTaintMismatch);
    }
    if frame.mode == FrameMode::Approximate && policy == KvReusePolicy::ExactOnly {
        return Err(FrameFault::ApproximateNotAllowed);
    }
    // A frame built in memory is held to what its bytes would be held to on the wire.
    frame.validate().map_err(FrameFault::Malformed)?;
    check_positions(&frame.positions, expected_start)?;
    revalidate_scoped(&scope, &frame.upstream_chain, frame.placement_epoch, live)?;
    for input in &frame.input_lineage.inputs {
        revalidate_input(input, inputs)?;
    }
    let activations = frame.tensor_f32();
    if let Some(index) = activations.iter().position(|value| !value.is_finite()) {
        return Err(FrameFault::NonFinite { index });
    }
    Ok(AdmittedFrameV2 {
        scope,
        frame_digest: frame.digest(),
        chain: frame.upstream_chain.clone(),
        lineage: frame.input_lineage.clone(),
        mode: frame.mode,
        taint: frame.taint,
        placement_epoch: frame.placement_epoch,
        rows: frame.tokens,
        width: frame.width,
        positions: frame.positions.clone(),
        activations,
    })
}

/// The key of a downstream entry: the namespace it lives in and the digest that names it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DownstreamKey {
    /// The cache namespace: the reuse mode of the frames. A lookup in one namespace never
    /// sees an entry of the other.
    pub namespace: KvReuseMode,
    pub digest: [u8; 32],
}

fn namespace_index(namespace: KvReuseMode) -> usize {
    match namespace {
        KvReuseMode::Exact => 0,
        KvReuseMode::Approximate => 1,
    }
}

/// The key of the downstream entry derived from `frames`, in order, by the downstream stage
/// described by `own`. It includes, for every frame, the frame digest (which covers the
/// chain, the lineage, the mode, the taint and the activations) and, explicitly and in order,
/// the Raw ancestry; it includes the mode and the taint of the frames and the downstream
/// stage's own chain; frames admitted for a project bind the project. The namespace is the
/// frames' mode (Exact when `frames` is empty).
///
/// Defence in depth: `frame_digest` already covers the chain, the whole lineage (typed inputs
/// and Raw ancestry), the mode and the taint, because the digest is taken over the frame's
/// canonical bytes. The explicit mode, taint, Raw-ancestry and typed-lineage terms repeat that
/// on purpose, so that the key stays correct if the digest's coverage is ever narrowed.
/// Removing one of those terms alone changes no key, so it is an equivalent mutant; only
/// removing it together with the digest's coverage is observable.
pub fn prefix_key_v2(frames: &[&AdmittedFrameV2], own: &UpstreamChain) -> DownstreamKey {
    let namespace = frames
        .first()
        .map_or(KvReuseMode::Exact, |frame| frame.namespace());
    let mut hasher = Sha256::new();
    hasher.update(KEY_DOMAIN);
    if let Some(project) = frames.first().and_then(|frame| frame.project()) {
        hasher.update(PROJECT_KEY_DOMAIN);
        hasher.update((project.0.len() as u64).to_le_bytes());
        hasher.update(project.0.as_bytes());
    }
    let own = chain_bytes(own);
    hasher.update((own.len() as u64).to_le_bytes());
    hasher.update(&own);
    hasher.update((frames.len() as u64).to_le_bytes());
    for frame in frames {
        hasher.update(frame.frame_digest);
        hasher.update(mode_taint_bytes(frame.mode, frame.taint));
        hasher.update((frame.lineage.raw_ancestry.len() as u64).to_le_bytes());
        for raw in &frame.lineage.raw_ancestry {
            hasher.update(raw);
        }
        let typed = lineage_bytes(&InputLineage {
            inputs: frame.lineage.inputs.clone(),
            raw_ancestry: Vec::new(),
        });
        hasher.update((typed.len() as u64).to_le_bytes());
        hasher.update(&typed);
    }
    DownstreamKey {
        namespace,
        digest: hasher.finalize().into(),
    }
}

/// Why an entry could not be stored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoreErrorV2 {
    /// No frames: an entry derives from at least one.
    NoFrames,
    /// The frames do not share one upstream chain, placement epoch and project.
    ChainMismatch,
    /// The frames do not share one mode and taint class: an entry lives in exactly one
    /// namespace, and a mixed one would need a taint that no single frame carries.
    ModeMismatch,
    /// The frames' typed inputs together exceed [`MAX_LINEAGE_INPUTS`].
    TooManyInputs,
    /// The cache holds its maximum of entries.
    Full,
}

/// Why a lookup or a use was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UseFaultV2 {
    /// No entry under the key in its namespace (never stored, dropped by an invalidation, or
    /// stored in the other namespace).
    Miss,
    /// The entry's recorded chain, placement, lineage or own chain is not live now.
    Stale(FrameFault),
    /// The ticket is not an open use of this cache.
    UnknownTicket,
    /// The use was cancelled by an invalidation while it ran.
    Cancelled,
    /// The cache already holds its maximum of open uses.
    TooManyUses { limit: usize },
}

/// An open use of an entry, bound to the cache instance that issued it: a ticket of another
/// cache never ends a use here (it would launder that cache's taint onto this one's output).
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct UseTicketV2 {
    cache: u64,
    id: u64,
}

/// Process-wide source of cache instance identities; never reused, never zero.
static NEXT_CACHE_INSTANCE: AtomicU64 = AtomicU64::new(1);

/// A hit, bound to the open use that [`SplitStageCacheV2::lookup`] began with it: the value
/// (private, readable only by borrow), the taint every output derived from it carries, and the
/// ticket of the use. There is no way to take the value out bare or to end the use without
/// revalidation: the output of a computation over [`value`](Self::value) is released only
/// through [`SplitStageCacheV2::finish_use`], which returns it as a [`Tainted`], or the use is
/// given up with [`SplitStageCacheV2::abandon_use`]. Not `Clone`, so a use is ended once.
#[must_use = "a hit is an open use: end it with finish_use or abandon_use"]
#[derive(Debug)]
pub struct ServedUseV2<V> {
    ticket: UseTicketV2,
    value: Arc<V>,
    taint: KvReuseTaint,
}

impl<V> ServedUseV2<V> {
    /// The cached value, borrowed for the computation of the output.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// The taint every output derived from this hit carries.
    pub fn taint(&self) -> KvReuseTaint {
        self.taint
    }
}

/// An output released from a use, together with the taint of the hit it derives from. It can be
/// built only by [`SplitStageCacheV2::finish_use`], so an output of an Approximate hit cannot
/// be presented as Exact.
#[must_use = "the taint travels with the output"]
#[derive(Debug, Eq, PartialEq)]
pub struct Tainted<O> {
    value: O,
    taint: KvReuseTaint,
}

impl<O> Tainted<O> {
    pub fn value(&self) -> &O {
        &self.value
    }

    pub fn taint(&self) -> KvReuseTaint {
        self.taint
    }

    /// The output and its taint, never one without the other.
    pub fn into_parts(self) -> (O, KvReuseTaint) {
        (self.value, self.taint)
    }
}

/// A way to put an entry somewhere other than this cache's memory. Every one is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistOp {
    Persist,
    Snapshot,
    TierWrite,
    Restore,
}

/// The typed refusal of [`PersistOp`]: downstream split caches are never persisted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistFault {
    pub op: PersistOp,
}

struct Entry<V> {
    scope: PlacementScope,
    own: UpstreamChain,
    chain: UpstreamChain,
    inputs: Vec<LineageInput>,
    placement_epoch: u64,
    taint: KvReuseTaint,
    value: Arc<V>,
}

fn entry_fault<V>(
    entry: &Entry<V>,
    live: &dyn StageLiveness,
    inputs: &dyn InputLiveness,
) -> Option<FrameFault> {
    revalidate_scoped(&entry.scope, &entry.chain, entry.placement_epoch, live)
        .and_then(|()| revalidate_own(&entry.own, live))
        .and_then(|()| {
            entry
                .inputs
                .iter()
                .try_for_each(|input| revalidate_input(input, inputs))
        })
        .err()
}

struct OpenUse {
    namespace: KvReuseMode,
    key: [u8; 32],
    cancelled: bool,
}

/// How many uses may be open per entry of the cache's capacity.
pub const OPEN_USES_PER_ENTRY_V2: usize = 8;

/// The downstream cache of version 2 split frames: bounded, two namespaces, never persisted.
pub struct SplitStageCacheV2<V> {
    namespaces: [BTreeMap<[u8; 32], Entry<V>>; 2],
    uses: BTreeMap<u64, OpenUse>,
    next_use: u64,
    instance: u64,
    capacity: usize,
}

impl<V> SplitStageCacheV2<V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            namespaces: [BTreeMap::new(), BTreeMap::new()],
            uses: BTreeMap::new(),
            next_use: 1,
            instance: NEXT_CACHE_INSTANCE.fetch_add(1, Ordering::Relaxed),
            capacity,
        }
    }

    /// Entries in both namespaces.
    pub fn len(&self) -> usize {
        self.namespaces.iter().map(BTreeMap::len).sum()
    }

    /// Entries in one namespace.
    pub fn len_in(&self, namespace: KvReuseMode) -> usize {
        self.namespaces[namespace_index(namespace)].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Open uses not yet finished or abandoned.
    pub fn open_uses(&self) -> usize {
        self.uses.len()
    }

    /// The most uses that may be open at once.
    pub fn max_open_uses(&self) -> usize {
        self.capacity.saturating_mul(OPEN_USES_PER_ENTRY_V2).max(1)
    }

    /// Store `value`, computed from `frames` in order by the downstream stage `own`, under
    /// their key in the frames' namespace. The entry records the chain, placement epoch,
    /// typed inputs and taint the frames carried, and `own`; all are revalidated at every
    /// use. A full cache first drops the entries that are no longer live.
    pub fn store(
        &mut self,
        frames: &[&AdmittedFrameV2],
        own: &UpstreamChain,
        value: V,
        live: &dyn StageLiveness,
        inputs: &dyn InputLiveness,
    ) -> Result<DownstreamKey, StoreErrorV2> {
        let first = frames.first().ok_or(StoreErrorV2::NoFrames)?;
        if frames.iter().any(|frame| {
            frame.chain != first.chain
                || frame.placement_epoch != first.placement_epoch
                || frame.scope != first.scope
        }) {
            return Err(StoreErrorV2::ChainMismatch);
        }
        if frames
            .iter()
            .any(|frame| frame.mode != first.mode || frame.taint != first.taint)
        {
            return Err(StoreErrorV2::ModeMismatch);
        }
        let mut typed: Vec<LineageInput> = frames
            .iter()
            .flat_map(|frame| frame.lineage.inputs.iter().cloned())
            .collect();
        typed.sort_by(|a, b| {
            (a.kind, &a.identity, a.generation).cmp(&(b.kind, &b.identity, b.generation))
        });
        typed.dedup();
        if typed.len() > MAX_LINEAGE_INPUTS {
            return Err(StoreErrorV2::TooManyInputs);
        }
        let key = prefix_key_v2(frames, own);
        let table = namespace_index(key.namespace);
        if !self.namespaces[table].contains_key(&key.digest) && self.len() >= self.capacity {
            self.evict_dead(live, inputs);
            if self.len() >= self.capacity {
                return Err(StoreErrorV2::Full);
            }
        }
        self.namespaces[table].insert(
            key.digest,
            Entry {
                scope: first.scope.clone(),
                own: own.clone(),
                chain: first.chain.clone(),
                inputs: typed,
                placement_epoch: first.placement_epoch,
                taint: first.taint(),
                value: Arc::new(value),
            },
        );
        Ok(key)
    }

    /// Drop every entry that fails revalidation now, cancelling the uses of it. Returns the
    /// number dropped.
    pub fn evict_dead(&mut self, live: &dyn StageLiveness, inputs: &dyn InputLiveness) -> usize {
        let mut dead: Vec<DownstreamKey> = Vec::new();
        for (index, table) in self.namespaces.iter().enumerate() {
            let namespace = if index == 0 {
                KvReuseMode::Exact
            } else {
                KvReuseMode::Approximate
            };
            for (digest, entry) in table {
                if entry_fault(entry, live, inputs).is_some() {
                    dead.push(DownstreamKey {
                        namespace,
                        digest: *digest,
                    });
                }
            }
        }
        for key in &dead {
            self.drop_entry(key.namespace, &key.digest);
        }
        dead.len()
    }

    fn drop_entry(&mut self, namespace: KvReuseMode, digest: &[u8; 32]) {
        self.namespaces[namespace_index(namespace)].remove(digest);
        for open in self.uses.values_mut() {
            if open.namespace == namespace && open.key == *digest {
                open.cancelled = true;
            }
        }
    }

    /// The entry under `key` in its namespace if everything it depends on is live now. An
    /// entry that is not live is evicted on the spot.
    fn live_entry(
        &mut self,
        key: &DownstreamKey,
        live: &dyn StageLiveness,
        inputs: &dyn InputLiveness,
    ) -> Result<&Entry<V>, UseFaultV2> {
        let table = namespace_index(key.namespace);
        let entry = self.namespaces[table]
            .get(&key.digest)
            .ok_or(UseFaultV2::Miss)?;
        if let Some(fault) = entry_fault(entry, live, inputs) {
            self.drop_entry(key.namespace, &key.digest);
            return Err(UseFaultV2::Stale(fault));
        }
        self.namespaces[table]
            .get(&key.digest)
            .ok_or(UseFaultV2::Miss)
    }

    /// The hit under `key`, if the entry is live now, with the use it opened: the value is
    /// readable only by borrow from the returned [`ServedUseV2`], whose use is registered so
    /// an invalidation during the computation cancels it. The output is released only by
    /// [`finish_use`](Self::finish_use). An entry that is not live is evicted on the spot.
    pub fn lookup(
        &mut self,
        key: &DownstreamKey,
        live: &dyn StageLiveness,
        inputs: &dyn InputLiveness,
    ) -> Result<ServedUseV2<V>, UseFaultV2> {
        let (value, taint) = self
            .live_entry(key, live, inputs)
            .map(|entry| (Arc::clone(&entry.value), entry.taint))?;
        if self.uses.len() >= self.max_open_uses() {
            return Err(UseFaultV2::TooManyUses {
                limit: self.max_open_uses(),
            });
        }
        let id = self.next_use;
        self.next_use += 1;
        self.uses.insert(
            id,
            OpenUse {
                namespace: key.namespace,
                key: key.digest,
                cancelled: false,
            },
        );
        Ok(ServedUseV2 {
            ticket: UseTicketV2 {
                cache: self.instance,
                id,
            },
            value,
            taint,
        })
    }

    /// End a use and release `output`, computed from the hit. It is released only when this
    /// returns `Ok`, and it carries the taint of the hit: the entry must still exist, the use
    /// must not have been cancelled, and everything the entry depends on must be live at this
    /// moment too.
    pub fn finish_use<O>(
        &mut self,
        served: ServedUseV2<V>,
        output: O,
        live: &dyn StageLiveness,
        inputs: &dyn InputLiveness,
    ) -> Result<Tainted<O>, UseFaultV2> {
        let ServedUseV2 { ticket, .. } = served;
        if ticket.cache != self.instance {
            return Err(UseFaultV2::UnknownTicket);
        }
        let open = self
            .uses
            .remove(&ticket.id)
            .ok_or(UseFaultV2::UnknownTicket)?;
        if open.cancelled {
            return Err(UseFaultV2::Cancelled);
        }
        let key = DownstreamKey {
            namespace: open.namespace,
            digest: open.key,
        };
        self.live_entry(&key, live, inputs).map(|entry| Tainted {
            value: output,
            taint: entry.taint,
        })
    }

    /// Give a use up without releasing any output.
    pub fn abandon_use(&mut self, served: ServedUseV2<V>) -> Result<(), UseFaultV2> {
        let ServedUseV2 { ticket, .. } = served;
        if ticket.cache != self.instance {
            return Err(UseFaultV2::UnknownTicket);
        }
        self.uses
            .remove(&ticket.id)
            .map(|_| ())
            .ok_or(UseFaultV2::UnknownTicket)
    }

    /// The `Revoke` message path: drop every entry that depends on `identity` at `generation`
    /// (as a stage, base model, adapter or expert of its chain, as a typed input of its
    /// lineage or as part of the downstream stage's own chain) and cancel the uses of it. A
    /// fence of an expert is the same message for the expert's generation. This is an
    /// optimisation of freeing early, never the guarantee: lookups and uses revalidate.
    pub fn invalidate_dependency(&mut self, identity: &str, generation: Generation) -> usize {
        let names = |element: &ChainElement| {
            element.identity == identity && element.generation == generation.0
        };
        let chain_names = |chain: &UpstreamChain| {
            names(&chain.stage)
                || names(&chain.base_model)
                || chain.adapters.iter().any(names)
                || chain.experts.iter().any(names)
        };
        let mut doomed: Vec<DownstreamKey> = Vec::new();
        for (namespace, table) in [KvReuseMode::Exact, KvReuseMode::Approximate]
            .into_iter()
            .zip(&self.namespaces)
        {
            for (digest, entry) in table {
                let depends = chain_names(&entry.chain)
                    || chain_names(&entry.own)
                    || entry.inputs.iter().any(|input| {
                        input.identity == identity && input.generation == generation.0
                    });
                if depends {
                    doomed.push(DownstreamKey {
                        namespace,
                        digest: *digest,
                    });
                }
            }
        }
        for key in &doomed {
            self.drop_entry(key.namespace, &key.digest);
        }
        doomed.len()
    }

    /// The typed refusal of every way to persist an entry or to bring one back: a snapshot, a
    /// tier write, a persist and a restore. The cache has no operation that serialises or
    /// reloads an entry, so this never returns `Ok` (its success type is uninhabited).
    pub fn persist(&self, op: PersistOp, key: &DownstreamKey) -> Result<Infallible, PersistFault> {
        let _ = key;
        Err(PersistFault { op })
    }

    /// Drop everything, as a restart does: the entries of both namespaces and the open uses
    /// (a ticket from before the call is [`UseFaultV2::UnknownTicket`]).
    pub fn clear(&mut self) {
        for table in &mut self.namespaces {
            table.clear();
        }
        self.uses.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_protocol::activation::BoundaryDtype;
    use std::cell::RefCell;

    #[derive(Default)]
    struct Live {
        standing: RefCell<BTreeMap<(String, u64), Validity>>,
        digests: RefCell<BTreeMap<(String, u64), [u8; 32]>>,
        placement: RefCell<BTreeMap<String, u64>>,
        fence: RefCell<BTreeMap<String, u64>>,
    }

    impl Live {
        fn commit(&self, identity: &str, generation: u64) {
            self.standing
                .borrow_mut()
                .insert((identity.to_owned(), generation), Validity::Live);
            self.digests
                .borrow_mut()
                .insert((identity.to_owned(), generation), [generation as u8; 32]);
        }

        fn set(&self, identity: &str, generation: u64, standing: Validity) {
            self.standing
                .borrow_mut()
                .insert((identity.to_owned(), generation), standing);
        }
    }

    impl StageLiveness for Live {
        fn validity(&self, target: &str, generation: Generation) -> Option<Validity> {
            self.standing
                .borrow()
                .get(&(target.to_owned(), generation.0))
                .copied()
        }

        fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
            self.digests
                .borrow()
                .get(&(target.to_owned(), generation.0))
                .copied()
        }

        fn placement_epoch(&self, stage: &str) -> Option<u64> {
            self.placement.borrow().get(stage).copied()
        }

        fn fence_epoch(&self, expert: &str) -> Option<u64> {
            self.fence.borrow().get(expert).copied()
        }
    }

    impl InputLiveness for Live {
        fn validity(
            &self,
            _kind: LineageKind,
            identity: &str,
            generation: Generation,
        ) -> Option<Validity> {
            StageLiveness::validity(self, identity, generation)
        }

        fn content_digest(
            &self,
            _kind: LineageKind,
            identity: &str,
            generation: Generation,
        ) -> Option<[u8; 32]> {
            self.manifest_digest(identity, generation)
        }
    }

    fn element(name: &str, generation: u64) -> ChainElement {
        ChainElement {
            identity: name.to_owned(),
            generation,
            digest: [generation as u8; 32],
            fence_epoch: 0,
        }
    }

    fn chain() -> UpstreamChain {
        UpstreamChain {
            stage: element("stage", 1),
            base_model: element("model", 1),
            adapters: vec![element("adapter", 1)],
            experts: vec![ChainElement {
                fence_epoch: 4,
                ..element("expert", 1)
            }],
        }
    }

    fn own_chain() -> UpstreamChain {
        UpstreamChain {
            stage: element("down", 1),
            base_model: element("model", 1),
            adapters: vec![],
            experts: vec![],
        }
    }

    fn input(kind: LineageKind, name: &str) -> LineageInput {
        LineageInput {
            kind,
            identity: name.to_owned(),
            generation: 1,
            digest: [1; 32],
        }
    }

    fn lineage(raw: Vec<[u8; 32]>) -> InputLineage {
        InputLineage::canonical(
            vec![
                input(LineageKind::Semantic, "sem"),
                input(LineageKind::Memory, "mem"),
                input(LineageKind::Derived, "drv"),
            ],
            raw,
        )
    }

    fn frame_with(
        lineage: InputLineage,
        mode: FrameMode,
        taint: FrameTaint,
        fill: f32,
    ) -> ActivationFrameV2 {
        ActivationFrameV2 {
            tokens: 2,
            width: 2,
            dtype: BoundaryDtype::F32,
            positions: vec![0, 1],
            upstream_chain: chain(),
            input_lineage: lineage,
            mode,
            taint,
            placement_epoch: 5,
            tensor: [fill, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn frame(raw: Vec<[u8; 32]>, fill: f32) -> ActivationFrameV2 {
        frame_with(lineage(raw), FrameMode::Exact, FrameTaint::Exact, fill)
    }

    fn approximate_frame(raw: Vec<[u8; 32]>) -> ActivationFrameV2 {
        frame_with(
            lineage(raw),
            FrameMode::Approximate,
            FrameTaint::Approximate {
                report_digest: [8; 32],
            },
            1.0,
        )
    }

    fn live() -> Live {
        let live = Live::default();
        for name in [
            "stage", "model", "adapter", "expert", "down", "sem", "mem", "drv",
        ] {
            live.commit(name, 1);
        }
        live.placement.borrow_mut().insert("stage".to_owned(), 5);
        live.fence.borrow_mut().insert("expert".to_owned(), 4);
        live
    }

    fn admit_at(
        live: &Live,
        frame: &ActivationFrameV2,
        expected_start: u64,
    ) -> Result<AdmittedFrameV2, FrameFault> {
        admit_frame_v2(
            AdmissionRequestV2 {
                frame,
                policy: KvReusePolicy::AllowApproximate,
                expected_start,
            },
            live,
            live,
        )
    }

    fn admit(live: &Live, frame: &ActivationFrameV2) -> Result<AdmittedFrameV2, FrameFault> {
        admit_at(live, frame, 0)
    }

    #[test]
    fn a_live_frame_is_admitted_and_each_lineage_kind_is_revalidated() {
        let live = live();
        let admitted = admit(&live, &frame(vec![[1; 32]], 1.0)).unwrap();
        assert_eq!(admitted.rows(), 2);
        assert_eq!(admitted.activations(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(admitted.namespace(), KvReuseMode::Exact);
        assert_eq!(admitted.taint(), KvReuseTaint::Exact);
        for (name, kind) in [
            ("sem", LineageKind::Semantic),
            ("mem", LineageKind::Memory),
            ("drv", LineageKind::Derived),
        ] {
            for standing in [Validity::Revoked, Validity::Superseded] {
                live.set(name, 1, standing);
                assert_eq!(
                    admit(&live, &frame(vec![], 1.0)),
                    Err(FrameFault::InputNotLive {
                        kind,
                        identity: name.to_owned(),
                        standing: Some(standing)
                    })
                );
                live.set(name, 1, Validity::Live);
            }
        }
        live.digests
            .borrow_mut()
            .insert(("mem".to_owned(), 1), [9; 32]);
        assert_eq!(
            admit(&live, &frame(vec![], 1.0)),
            Err(FrameFault::InputDigestMismatch {
                kind: LineageKind::Memory,
                identity: "mem".to_owned()
            })
        );
    }

    #[test]
    fn the_whole_upstream_chain_is_revalidated_at_admission() {
        let live = live();
        for name in ["stage", "model", "adapter", "expert"] {
            live.set(name, 1, Validity::Revoked);
            assert!(
                matches!(
                    admit(&live, &frame(vec![], 1.0)),
                    Err(FrameFault::Chain(ChainFault::NotLive { .. }))
                ),
                "{name}"
            );
            live.set(name, 1, Validity::Live);
        }
        live.fence.borrow_mut().insert("expert".to_owned(), 5);
        assert!(matches!(
            admit(&live, &frame(vec![], 1.0)),
            Err(FrameFault::Chain(ChainFault::StaleFence {
                recorded: 4,
                current: Some(5),
                ..
            }))
        ));
    }

    #[test]
    fn an_approximate_frame_needs_the_policy_and_a_matching_taint() {
        let live = live();
        assert_eq!(
            admit_frame_v2(
                AdmissionRequestV2 {
                    frame: &approximate_frame(vec![]),
                    policy: KvReusePolicy::ExactOnly,
                    expected_start: 0,
                },
                &live,
                &live
            ),
            Err(FrameFault::ApproximateNotAllowed)
        );
        let admitted = admit(&live, &approximate_frame(vec![])).unwrap();
        assert_eq!(admitted.namespace(), KvReuseMode::Approximate);
        assert!(admitted.taint().is_approximate());
        let mut laundered = approximate_frame(vec![]);
        laundered.taint = FrameTaint::Exact;
        assert_eq!(admit(&live, &laundered), Err(FrameFault::ModeTaintMismatch));
    }

    #[test]
    fn a_frame_that_would_not_decode_is_not_admitted_in_memory_either() {
        let live = live();
        let mut unsorted = frame(vec![], 1.0);
        unsorted.input_lineage.inputs.reverse();
        assert_eq!(
            admit(&live, &unsorted),
            Err(FrameFault::Malformed(SchemaError::NonCanonicalLineage))
        );
        let raw = vec![[1; 32]; ptr_protocol::activation::MAX_RAW_ANCESTRY + 1];
        assert_eq!(
            admit(&live, &frame(raw, 1.0)),
            Err(FrameFault::Malformed(SchemaError::OverBound(
                "raw ancestry"
            )))
        );
        let mut short = frame(vec![], 1.0);
        short.tensor.pop();
        assert!(matches!(
            admit(&live, &short),
            Err(FrameFault::Malformed(SchemaError::ShapeMismatch { .. }))
        ));
        let mut positions = frame(vec![], 1.0);
        positions.positions.pop();
        assert!(matches!(
            admit(&live, &positions),
            Err(FrameFault::Malformed(SchemaError::PositionsLength { .. }))
        ));
    }

    #[test]
    fn runtime_input_liveness_keeps_two_kinds_of_one_identity_apart() {
        use ptr_config::PtrConfig;
        use ptr_ledger::LedgerEvent;
        use ptr_types::CapsuleId;
        let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
        for kind in [LineageKind::Semantic, LineageKind::Memory] {
            runtime
                .commit(LedgerEvent::CapsuleCommitted {
                    project: ProjectId::from("p"),
                    capsule: CapsuleId::from(input_target(kind, "same").as_str()),
                    generation: Generation(1),
                })
                .unwrap();
        }
        let digests = BTreeMap::from([
            ((LineageKind::Semantic, "same".to_owned(), 1), [1; 32]),
            ((LineageKind::Memory, "same".to_owned(), 1), [2; 32]),
        ]);
        let live = RuntimeInputLiveness {
            runtime: &runtime,
            digests: &digests,
        };
        let validity = |kind| InputLiveness::validity(&live, kind, "same", Generation(1));
        assert_eq!(validity(LineageKind::Semantic), Some(Validity::Live));
        assert_eq!(validity(LineageKind::Memory), Some(Validity::Live));
        assert_eq!(validity(LineageKind::Derived), None);
        runtime
            .commit(LedgerEvent::Revoked {
                subject: input_target(LineageKind::Memory, "same"),
                generation: Generation(1),
            })
            .unwrap();
        let live = RuntimeInputLiveness {
            runtime: &runtime,
            digests: &digests,
        };
        let validity = |kind| InputLiveness::validity(&live, kind, "same", Generation(1));
        assert_eq!(validity(LineageKind::Memory), Some(Validity::Revoked));
        assert_eq!(validity(LineageKind::Semantic), Some(Validity::Live));
        let digest = |kind| live.content_digest(kind, "same", Generation(1));
        assert_eq!(digest(LineageKind::Semantic), Some([1; 32]));
        assert_eq!(digest(LineageKind::Memory), Some([2; 32]));
        assert_eq!(digest(LineageKind::Derived), None);
    }

    fn frame_at(positions: Vec<u32>) -> ActivationFrameV2 {
        let mut frame = frame(vec![], 1.0);
        frame.positions = positions;
        frame
    }

    #[test]
    fn positions_that_continue_the_downstream_kv_state_exactly_are_admitted() {
        let live = live();
        assert!(admit_at(&live, &frame_at(vec![0, 1]), 0).is_ok());
        assert!(admit_at(&live, &frame_at(vec![7, 8]), 7).is_ok());
    }

    #[test]
    fn a_stale_frame_wholly_before_the_expected_position_is_refused() {
        let live = live();
        assert_eq!(
            admit_at(&live, &frame_at(vec![0, 1]), 2),
            Err(FrameFault::PositionsStale {
                expected_start: 2,
                first: 0,
                last: 1
            })
        );
    }

    #[test]
    fn an_overlapping_frame_is_refused() {
        let live = live();
        assert_eq!(
            admit_at(&live, &frame_at(vec![0, 1]), 1),
            Err(FrameFault::PositionsOverlap {
                expected_start: 1,
                first: 0
            })
        );
    }

    #[test]
    fn a_frame_that_skips_positions_is_refused() {
        let live = live();
        assert_eq!(
            admit_at(&live, &frame_at(vec![3, 4]), 2),
            Err(FrameFault::PositionsSkipped {
                expected_start: 2,
                first: 3
            })
        );
    }

    #[test]
    fn a_frame_whose_own_positions_are_not_consecutive_is_refused() {
        let live = live();
        assert_eq!(
            admit_at(&live, &frame_at(vec![0, 2]), 0),
            Err(FrameFault::PositionsNotContiguous { index: 1 })
        );
        assert_eq!(
            admit_at(&live, &frame_at(vec![0, 0]), 0),
            Err(FrameFault::PositionsNotContiguous { index: 1 })
        );
    }

    #[test]
    fn non_finite_activations_are_refused() {
        let live = live();
        assert_eq!(
            admit(&live, &frame(vec![], f32::NAN)),
            Err(FrameFault::NonFinite { index: 0 })
        );
    }

    #[test]
    fn the_key_includes_the_raw_ancestry_in_order_the_mode_the_taint_and_the_own_chain() {
        let live = live();
        let key = |frame: &ActivationFrameV2, own: &UpstreamChain| {
            let admitted = admit(&live, frame).unwrap();
            prefix_key_v2(&[&admitted], own)
        };
        let base = key(&frame(vec![[1; 32], [2; 32]], 1.0), &own_chain());
        assert_ne!(base, key(&frame(vec![[2; 32]], 1.0), &own_chain()));
        assert_ne!(base, key(&frame(vec![[2; 32], [1; 32]], 1.0), &own_chain()));
        assert_ne!(base, key(&frame(vec![[1; 32], [2; 32]], 9.0), &own_chain()));
        let mut other_own = own_chain();
        other_own.stage.generation = 2;
        assert_ne!(base, key(&frame(vec![[1; 32], [2; 32]], 1.0), &other_own));
        let approximate = key(&approximate_frame(vec![[1; 32], [2; 32]]), &own_chain());
        assert_ne!(base.digest, approximate.digest);
        assert_eq!(approximate.namespace, KvReuseMode::Approximate);
        let mut other_taint = approximate_frame(vec![[1; 32], [2; 32]]);
        other_taint.taint = FrameTaint::Approximate {
            report_digest: [7; 32],
        };
        assert_ne!(approximate, key(&other_taint, &own_chain()));
        // The same inputs give the same key.
        assert_eq!(base, key(&frame(vec![[1; 32], [2; 32]], 1.0), &own_chain()));
    }

    fn stored(
        live: &Live,
        cache: &mut SplitStageCacheV2<u32>,
        frame: &ActivationFrameV2,
        value: u32,
    ) -> DownstreamKey {
        let admitted = admit(live, frame).unwrap();
        cache
            .store(&[&admitted], &own_chain(), value, live, live)
            .unwrap()
    }

    #[test]
    fn an_entry_serves_while_live_and_each_dependency_kind_ends_it() {
        let live = live();
        let mut cache = SplitStageCacheV2::new(8);
        let key = stored(&live, &mut cache, &frame(vec![[1; 32]], 1.0), 7);
        let served = cache.lookup(&key, &live, &live).unwrap();
        assert_eq!((*served.value(), served.taint()), (7, KvReuseTaint::Exact));
        cache.abandon_use(served).unwrap();
        for name in [
            "stage", "model", "adapter", "expert", "down", "sem", "mem", "drv",
        ] {
            live.set(name, 1, Validity::Superseded);
            assert!(
                matches!(cache.lookup(&key, &live, &live), Err(UseFaultV2::Stale(_))),
                "{name}"
            );
            assert_eq!(cache.len(), 0, "{name} evicted on the spot");
            live.set(name, 1, Validity::Live);
            stored(&live, &mut cache, &frame(vec![[1; 32]], 1.0), 7);
        }
        // A fence moves the expert's serving epoch with no generation change.
        live.fence.borrow_mut().insert("expert".to_owned(), 5);
        assert!(matches!(
            cache.lookup(&key, &live, &live),
            Err(UseFaultV2::Stale(FrameFault::Chain(
                ChainFault::StaleFence { .. }
            )))
        ));
    }

    #[test]
    fn a_use_in_flight_is_cancelled_or_refused_at_release_and_carries_the_taint() {
        let live = live();
        let mut cache = SplitStageCacheV2::new(8);
        let key = stored(&live, &mut cache, &approximate_frame(vec![]), 3);
        // The hit is an open use; the output is released only with the taint.
        let served = cache.lookup(&key, &live, &live).unwrap();
        assert_eq!(cache.open_uses(), 1);
        assert_eq!(
            served.taint(),
            KvReuseTaint::Approximate {
                report_digest: [8; 32]
            }
        );
        let output = *served.value() + 1;
        let released = cache.finish_use(served, output, &live, &live).unwrap();
        assert_eq!(
            released.into_parts(),
            (
                4,
                KvReuseTaint::Approximate {
                    report_digest: [8; 32]
                }
            )
        );
        assert_eq!(cache.open_uses(), 0);
        // Invalidated under the use with the Revoke message dropped: refused at release.
        let served = cache.lookup(&key, &live, &live).unwrap();
        live.set("sem", 1, Validity::Revoked);
        assert!(matches!(
            cache.finish_use(served, 0, &live, &live),
            Err(UseFaultV2::Stale(FrameFault::InputNotLive { .. }))
        ));
        live.set("sem", 1, Validity::Live);
        // The message delivered: the use is cancelled.
        let key = stored(&live, &mut cache, &approximate_frame(vec![]), 3);
        let served = cache.lookup(&key, &live, &live).unwrap();
        assert_eq!(cache.invalidate_dependency("mem", Generation(1)), 1);
        assert!(matches!(
            cache.finish_use(served, 0, &live, &live),
            Err(UseFaultV2::Cancelled)
        ));
    }

    #[test]
    fn a_fenced_expert_of_the_own_chain_ends_the_entry_without_the_revoke_message() {
        let live = live();
        live.commit("own-expert", 1);
        live.fence.borrow_mut().insert("own-expert".to_owned(), 2);
        let own = UpstreamChain {
            experts: vec![ChainElement {
                fence_epoch: 2,
                ..element("own-expert", 1)
            }],
            ..own_chain()
        };
        let mut cache = SplitStageCacheV2::new(8);
        let admitted = admit(&live, &frame(vec![], 1.0)).unwrap();
        let key = cache.store(&[&admitted], &own, 7, &live, &live).unwrap();
        let served = cache.lookup(&key, &live, &live).unwrap();
        // Fenced: the generation stays live, only the committed epoch moves.
        live.fence.borrow_mut().insert("own-expert".to_owned(), 3);
        let stale = FrameFault::Chain(ChainFault::StaleFence {
            identity: "own-expert".to_owned(),
            recorded: 2,
            current: Some(3),
        });
        assert!(
            matches!(
                cache.finish_use(served, 0, &live, &live),
                Err(UseFaultV2::Stale(ref fault)) if *fault == stale
            ),
            "refused with the stale fence"
        );
        // The stale entry was evicted on the spot.
        assert!(matches!(
            cache.lookup(&key, &live, &live),
            Err(UseFaultV2::Miss)
        ));
        // An own expert without a recorded epoch is never served (fail closed).
        let unrecorded = UpstreamChain {
            experts: vec![element("own-expert", 1)],
            ..own_chain()
        };
        let key = cache
            .store(&[&admitted], &unrecorded, 8, &live, &live)
            .unwrap();
        assert!(matches!(
            cache.lookup(&key, &live, &live),
            Err(UseFaultV2::Stale(FrameFault::Chain(
                ChainFault::StaleFence { recorded: 0, .. }
            )))
        ));
    }

    #[test]
    fn a_miss_is_not_released_after_its_own_chain_was_invalidated() {
        let live = live();
        let admitted = admit(&live, &frame(vec![], 1.0)).unwrap();
        assert_eq!(admitted.revalidate(&own_chain(), &live, &live), Ok(()));
        // The downstream stage's own generation revoked while the miss computes.
        live.set("down", 1, Validity::Revoked);
        assert_eq!(
            admitted.revalidate(&own_chain(), &live, &live),
            Err(FrameFault::Chain(ChainFault::NotLive {
                identity: "down".to_owned(),
                standing: Some(Validity::Revoked),
            }))
        );
    }

    #[test]
    fn a_ticket_of_another_cache_ends_no_use_here() {
        let live = live();
        let mut exact = SplitStageCacheV2::new(8);
        let mut approximate = SplitStageCacheV2::new(8);
        let exact_key = stored(&live, &mut exact, &frame(vec![], 1.0), 1);
        let approximate_key = stored(&live, &mut approximate, &approximate_frame(vec![]), 2);
        let mine = exact.lookup(&exact_key, &live, &live).unwrap();
        let theirs = approximate.lookup(&approximate_key, &live, &live).unwrap();
        // Both are the first use of their cache, so the ids collide; the cache does not.
        assert_eq!(mine.ticket.id, theirs.ticket.id);
        assert!(matches!(
            exact.finish_use(theirs, 0, &live, &live),
            Err(UseFaultV2::UnknownTicket)
        ));
        assert_eq!(exact.open_uses(), 1);
        let theirs = approximate.lookup(&approximate_key, &live, &live).unwrap();
        assert_eq!(exact.abandon_use(theirs), Err(UseFaultV2::UnknownTicket));
        let released = exact.finish_use(mine, 0, &live, &live).unwrap();
        assert_eq!(released.taint(), KvReuseTaint::Exact);
    }

    #[test]
    fn an_approximate_hit_is_released_only_through_finish_use_and_keeps_its_taint() {
        let live = live();
        let mut cache = SplitStageCacheV2::new(8);
        let key = stored(&live, &mut cache, &approximate_frame(vec![]), 5);
        // The hit is the only handle on the value: it is an open use until it is ended.
        let served = cache.lookup(&key, &live, &live).unwrap();
        assert_eq!(cache.open_uses(), 1);
        assert!(served.taint().is_approximate());
        // A computation over the borrowed value; the release revalidates first, and an
        // invalidation between lookup and finish_use refuses the release.
        let output = served.value().to_string();
        live.set("mem", 1, Validity::Revoked);
        assert!(matches!(
            cache.finish_use(served, output, &live, &live),
            Err(UseFaultV2::Stale(FrameFault::InputNotLive { .. }))
        ));
        assert_eq!(cache.open_uses(), 0);
        live.set("mem", 1, Validity::Live);
        // Released output carries the Approximate taint; it cannot become Exact.
        let key = stored(&live, &mut cache, &approximate_frame(vec![]), 5);
        let served = cache.lookup(&key, &live, &live).unwrap();
        let output = served.value().to_string();
        let released = cache.finish_use(served, output, &live, &live).unwrap();
        assert!(released.taint().is_approximate());
        assert_eq!(released.value(), "5");
        // An abandoned hit releases nothing and frees the use.
        let served = cache.lookup(&key, &live, &live).unwrap();
        cache.abandon_use(served).unwrap();
        assert_eq!(cache.open_uses(), 0);
    }

    #[test]
    fn the_namespaces_are_separate_an_approximate_entry_never_hits_exact() {
        let live = live();
        let mut cache = SplitStageCacheV2::new(8);
        let approximate = stored(&live, &mut cache, &approximate_frame(vec![]), 1);
        assert_eq!(approximate.namespace, KvReuseMode::Approximate);
        assert_eq!(cache.len_in(KvReuseMode::Approximate), 1);
        assert_eq!(cache.len_in(KvReuseMode::Exact), 0);
        // The same digest asked in the Exact namespace is a miss, and so is a lookup that
        // forges the namespace of the key.
        let forged = DownstreamKey {
            namespace: KvReuseMode::Exact,
            digest: approximate.digest,
        };
        assert!(matches!(
            cache.lookup(&forged, &live, &live),
            Err(UseFaultV2::Miss)
        ));
        // An Exact frame never populates or hits the Approximate namespace either.
        let exact = stored(&live, &mut cache, &frame(vec![], 1.0), 2);
        assert_eq!(cache.len_in(KvReuseMode::Exact), 1);
        let forged = DownstreamKey {
            namespace: KvReuseMode::Approximate,
            digest: exact.digest,
        };
        assert!(matches!(
            cache.lookup(&forged, &live, &live),
            Err(UseFaultV2::Miss)
        ));
    }

    #[test]
    fn frames_of_two_modes_chains_or_too_many_inputs_cannot_back_one_entry() {
        let live = live();
        let mut cache: SplitStageCacheV2<u32> = SplitStageCacheV2::new(4);
        let exact = admit(&live, &frame(vec![], 1.0)).unwrap();
        let approximate = admit(&live, &approximate_frame(vec![])).unwrap();
        assert_eq!(
            cache.store(&[&exact, &approximate], &own_chain(), 0, &live, &live),
            Err(StoreErrorV2::ModeMismatch)
        );
        assert_eq!(
            cache.store(&[], &own_chain(), 0, &live, &live),
            Err(StoreErrorV2::NoFrames)
        );
        live.commit("other-stage", 1);
        live.placement
            .borrow_mut()
            .insert("other-stage".to_owned(), 5);
        let mut other = frame(vec![], 2.0);
        other.upstream_chain.stage = element("other-stage", 1);
        let other = admit(&live, &other).unwrap();
        assert_eq!(
            cache.store(&[&exact, &other], &own_chain(), 0, &live, &live),
            Err(StoreErrorV2::ChainMismatch)
        );
        // Frames with distinct inputs together over the bound.
        let mut crowd = Vec::new();
        for index in 0..2 {
            let inputs = (0..MAX_LINEAGE_INPUTS / 2 + 1)
                .map(|number| {
                    let name = format!("in-{index}-{number}");
                    live.commit(&name, 1);
                    LineageInput {
                        kind: LineageKind::Semantic,
                        identity: name,
                        generation: 1,
                        digest: [1; 32],
                    }
                })
                .collect();
            let lineage = InputLineage::canonical(inputs, vec![]);
            crowd.push(
                admit(
                    &live,
                    &frame_with(lineage, FrameMode::Exact, FrameTaint::Exact, 1.0),
                )
                .unwrap(),
            );
        }
        assert_eq!(
            cache.store(&[&crowd[0], &crowd[1]], &own_chain(), 0, &live, &live),
            Err(StoreErrorV2::TooManyInputs)
        );
    }

    #[test]
    fn a_full_cache_drops_dead_entries_when_the_revoke_never_arrived() {
        let live = live();
        let mut cache = SplitStageCacheV2::new(2);
        stored(&live, &mut cache, &frame(vec![[1; 32]], 1.0), 1);
        stored(&live, &mut cache, &frame(vec![[2; 32]], 1.0), 2);
        let third = admit(&live, &frame(vec![[3; 32]], 1.0)).unwrap();
        assert_eq!(
            cache.store(&[&third], &own_chain(), 3, &live, &live),
            Err(StoreErrorV2::Full)
        );
        live.set("model", 1, Validity::Revoked);
        // The model is revoked: a fresh frame cannot be admitted, but the dead entries leave
        // as soon as a store needs the slots (and an unrelated live frame is stored).
        live.commit("model", 2);
        let mut next = frame(vec![[3; 32]], 1.0);
        next.upstream_chain.base_model = element("model", 2);
        let next = admit(&live, &next).unwrap();
        let mut own = own_chain();
        own.base_model = element("model", 2);
        assert!(cache.store(&[&next], &own, 3, &live, &live).is_ok());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn every_way_to_persist_or_restore_an_entry_is_refused_and_a_restart_drops_everything() {
        let live = live();
        let mut cache = SplitStageCacheV2::new(8);
        let exact = stored(&live, &mut cache, &frame(vec![], 1.0), 1);
        let approximate = stored(&live, &mut cache, &approximate_frame(vec![]), 2);
        for key in [&exact, &approximate] {
            for op in [
                PersistOp::Persist,
                PersistOp::Snapshot,
                PersistOp::TierWrite,
                PersistOp::Restore,
            ] {
                assert_eq!(cache.persist(op, key).unwrap_err(), PersistFault { op });
            }
        }
        let served = cache.lookup(&exact, &live, &live).unwrap();
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.open_uses(), 0);
        assert!(matches!(
            cache.lookup(&exact, &live, &live),
            Err(UseFaultV2::Miss)
        ));
        assert!(matches!(
            cache.finish_use(served, 0, &live, &live),
            Err(UseFaultV2::UnknownTicket)
        ));
        // A new cache (a restart) starts empty.
        let fresh: SplitStageCacheV2<u32> = SplitStageCacheV2::new(8);
        assert!(fresh.is_empty());
    }

    #[test]
    fn the_revoke_message_path_drops_by_chain_lineage_or_own_chain() {
        let live = live();
        for (name, expected) in [
            ("stage", 1),
            ("adapter", 1),
            ("expert", 1),
            ("sem", 1),
            ("drv", 1),
            ("down", 1),
            ("nobody", 0),
        ] {
            let mut cache = SplitStageCacheV2::new(8);
            stored(&live, &mut cache, &frame(vec![], 1.0), 1);
            assert_eq!(
                cache.invalidate_dependency(name, Generation(1)),
                expected,
                "{name}"
            );
            assert_eq!(cache.len(), 1 - expected);
        }
        let mut cache = SplitStageCacheV2::new(8);
        stored(&live, &mut cache, &frame(vec![], 1.0), 1);
        assert_eq!(cache.invalidate_dependency("sem", Generation(2)), 0);
    }
}
