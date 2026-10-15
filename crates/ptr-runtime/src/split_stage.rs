//! The downstream side of a pipeline (layer) split: admission of activation frames and the
//! downstream KV cache, both revalidated at use against what the runtime holds now.
//!
//! In a pipeline split the KV of a layer range lives on the Pod hosting it and only
//! activations cross Pods (doc 36, section D). A downstream stage computes from an
//! activation frame and keeps its KV; that KV is derived from the upstream chain the frame
//! carried: the upstream stage Pod (identity, generation, manifest digest), the base model
//! and every adapter and expert of the upstream layer range, and the placement epoch the
//! upstream stage held. A `Revoke` message can be dropped or delayed, so it is not the
//! guarantee. The guarantee is that **the downstream stage revalidates every element of the
//! chain, and the placement epoch, against a [`StageLiveness`] at every use**: admission of
//! a frame ([`admit_frame`]), a lookup, the start of a use and the release of its output.
//!
//! Nothing here persists. A downstream entry is never snapshotted, tiered or restored: the
//! cache has no operation that serialises or reloads an entry, and a restart drops it.
//! Nothing here installs KV from a wire either: an [`AdmittedFrame`] holds activations and
//! positions and the cache stores what the caller computed from them.

use crate::expert::ExpertEntry;
use crate::placement::PlacementScope;
use crate::{PodPlacementController, PtrRuntime};
use ptr_model_api::KvReuseMode;
use ptr_pods::ExpertState;
use ptr_protocol::activation::{
    chain_bytes, frame_digest, ActivationFrame, ChainElement, UpstreamChain,
};
use ptr_types::{Generation, ProjectId, Validity};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;

const KEY_DOMAIN: &[u8] = b"ptr.split.downstream-key.v1\0";
const PROJECT_KEY_DOMAIN: &[u8] = b"ptr.split.downstream-key.project\0";

/// What the downstream stage asks about the upstream chain, at use.
pub trait StageLiveness {
    /// Whether `target` at `generation` may be used now, as the lifecycle authority sees it
    /// (`None`: unknown).
    fn validity(&self, target: &str, generation: Generation) -> Option<Validity>;
    /// The digest committed for `target` at `generation` (`None`: unknown).
    fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]>;
    /// The placement (fence) epoch the stage `stage` holds now in the **legacy single-project
    /// namespace** (`None`: not placed). Kept unchanged for single-project callers and for
    /// completed experiments; a caller that serves a project asks
    /// [`Self::placement_epoch_in`].
    fn placement_epoch(&self, stage: &str) -> Option<u64>;
    /// The placement (fence) epoch the stage `stage` holds now **for `project`** (`None`: not
    /// placed for that project). `PodRegistry` allows one `PodId` in two projects, so a stage
    /// name alone does not identify a placement.
    ///
    /// The default answers `None`: an implementor that does not know projects fails closed
    /// (every project-scoped frame or handle is refused as stale) instead of answering with
    /// another project's epoch. An implementor that holds placements per project overrides it.
    fn placement_epoch_in(&self, project: &ProjectId, stage: &str) -> Option<u64> {
        let _ = (project, stage);
        None
    }
    /// The fence epoch the expert `expert` is serving under now (`None`: not serving).
    fn fence_epoch(&self, expert: &str) -> Option<u64>;
}

/// [`StageLiveness`] over the runtime's committed lifecycle state and its placement
/// controller. The runtime keeps no digest for a capsule generation, so the host passes the
/// digests it admitted each generation under.
pub struct RuntimeStageLiveness<'a> {
    pub runtime: &'a PtrRuntime,
    pub placement: &'a PodPlacementController,
    pub manifests: &'a BTreeMap<(String, u64), [u8; 32]>,
}

impl RuntimeStageLiveness<'_> {
    /// The registry entry behind an `expert:{project}/{pod}` identity. `None` for an identity
    /// that is no registry key and for an expert the registry does not hold.
    fn expert_entry(&self, identity: &str) -> Option<&ExpertEntry> {
        // Project and pod ids never contain `/`, so the key has exactly one split; the same
        // idiom as the registry's own parser (`highest_fenced_epoch`).
        let (project, pod) = identity
            .strip_prefix(ptr_state::EXPERT_PREFIX)?
            .split_once('/')?;
        self.runtime
            .expert_registry()
            .get(&ProjectId::from(project), &ptr_types::PodId::from(pod))
    }
}

/// The standing of `generation` of an expert, from its registry entry alone. The current
/// generation is live while the expert is admitted (`Serving`, `Frozen`, or `Fenced`, which
/// the fence epoch refuses on its own), superseded or revoked when the registry says so; an
/// older generation is superseded; anything else is unknown.
fn expert_validity(entry: &ExpertEntry, generation: Generation) -> Option<Validity> {
    if generation.0 < entry.generation.0 {
        return Some(Validity::Superseded);
    }
    if generation != entry.generation {
        return None;
    }
    match entry.state {
        ExpertState::Serving | ExpertState::Frozen | ExpertState::Fenced => Some(Validity::Live),
        ExpertState::Superseded => Some(Validity::Superseded),
        ExpertState::Retired(_) => Some(Validity::Revoked),
        ExpertState::Candidate | ExpertState::Gated => None,
    }
}

impl StageLiveness for RuntimeStageLiveness<'_> {
    /// An `expert:{project}/{pod}` identity is answered from the expert registry alone (its
    /// generation, state and manifest digest are the same entry the fence epoch comes from); an
    /// expert identity the registry does not hold is unknown. Everything else is a capsule
    /// generation of the runtime's lifecycle state.
    fn validity(&self, target: &str, generation: Generation) -> Option<Validity> {
        if target.starts_with(ptr_state::EXPERT_PREFIX) {
            return self
                .expert_entry(target)
                .and_then(|entry| expert_validity(entry, generation));
        }
        self.runtime.generation_validity(target, generation)
    }

    /// An expert's digest is its registry manifest digest at its current generation; every
    /// other target's digest is the one the host admitted (`manifests`).
    fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
        if target.starts_with(ptr_state::EXPERT_PREFIX) {
            return self
                .expert_entry(target)
                .filter(|entry| entry.generation == generation)
                .map(|entry| *entry.manifest_digest.as_bytes());
        }
        self.manifests
            .get(&(target.to_owned(), generation.0))
            .copied()
    }

    fn placement_epoch(&self, stage: &str) -> Option<u64> {
        self.placement
            .placement(&ptr_types::PodId::from(stage))
            .ok()
            .map(|placement| placement.epoch.0)
    }

    fn placement_epoch_in(&self, project: &ProjectId, stage: &str) -> Option<u64> {
        self.placement
            .placement_in(project, &ptr_types::PodId::from(stage))
            .ok()
            .map(|placement| placement.epoch.0)
    }

    /// The fence epoch the expert serves under now, from the runtime's expert registry (which
    /// is rebuilt from the ledger on replay and carried by compacted snapshots). The identity
    /// is the registry's `expert:{project}/{pod}` key, so the answer is scoped to the project it
    /// names. The epoch is the expert's committed state version: it moves at every fence,
    /// escalation and readmission (so a frame computed before a quarantine never revives) and
    /// at any other state change, which only ever invalidates more. `None` for an identity that
    /// is no registry key and for an expert that is not serving (`Serving` or `Frozen`), so a
    /// fenced, retired or unknown expert fails closed ([`ChainFault::StaleFence`]).
    fn fence_epoch(&self, expert: &str) -> Option<u64> {
        let entry = self.expert_entry(expert)?;
        matches!(entry.state, ExpertState::Serving | ExpertState::Frozen)
            .then_some(entry.state_version)
    }
}

/// Why a chain, or the placement it was produced under, is not usable now.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChainFault {
    /// An element's generation is not live (revoked, superseded, disputed or unknown).
    NotLive {
        identity: String,
        standing: Option<Validity>,
    },
    /// An element's digest is not the one committed for its generation.
    DigestMismatch { identity: String },
    /// The frame was produced under a placement epoch that is not the stage's current one.
    StalePlacement { recorded: u64, current: Option<u64> },
    /// The upstream stage computed under an approximate reuse mode. Version 1 of the split
    /// path is Exact-only: the frame carries no mode or taint, so the caller states the mode
    /// and anything but Exact is refused here rather than silently losing its taint.
    NotExact { mode: KvReuseMode },
    /// The activations hold a NaN or an infinity, at this index of the tensor.
    NonFinite { index: usize },
    /// An expert recorded no fence epoch, or is serving under another one now.
    StaleFence {
        identity: String,
        recorded: u64,
        current: Option<u64>,
    },
}

/// Why `entry` is not usable now, if it is not.
fn entry_fault<V>(entry: &Entry<V>, live: &dyn StageLiveness) -> Option<ChainFault> {
    revalidate_chain_scoped(&entry.scope, &entry.chain, entry.placement_epoch, live)
        .and_then(|()| revalidate_own(&entry.own, live))
        .err()
}

/// Revalidate the elements of the downstream stage's own chain (no placement: the frame's
/// placement epoch belongs to the upstream stage).
fn revalidate_own(own: &UpstreamChain, live: &dyn StageLiveness) -> Result<(), ChainFault> {
    revalidate_element(&own.stage, live)?;
    revalidate_element(&own.base_model, live)?;
    for element in &own.adapters {
        revalidate_element(element, live)?;
    }
    // An own expert is fenced like an upstream one: a fence leaves its generation live, so the
    // recorded epoch must still be the committed one (fail closed on an unrecorded epoch 0).
    for expert in &own.experts {
        revalidate_element(expert, live)?;
        let current = live.fence_epoch(&expert.identity);
        if expert.fence_epoch == 0 || current != Some(expert.fence_epoch) {
            return Err(ChainFault::StaleFence {
                identity: expert.identity.clone(),
                recorded: expert.fence_epoch,
                current,
            });
        }
    }
    Ok(())
}

fn revalidate_element(element: &ChainElement, live: &dyn StageLiveness) -> Result<(), ChainFault> {
    let generation = Generation(element.generation);
    match live.validity(&element.identity, generation) {
        Some(Validity::Live) => {}
        standing => {
            return Err(ChainFault::NotLive {
                identity: element.identity.clone(),
                standing,
            })
        }
    }
    if live.manifest_digest(&element.identity, generation) != Some(element.digest) {
        return Err(ChainFault::DigestMismatch {
            identity: element.identity.clone(),
        });
    }
    Ok(())
}

/// Revalidate every element of `chain` and the placement epoch against `live`: the stage
/// (identity, generation, manifest digest), the base model, every adapter and every expert
/// (generation, digest and fence epoch).
///
/// The placement epoch is read in the **legacy single-project namespace**
/// ([`StageLiveness::placement_epoch`]); use [`revalidate_chain_in`] to check it for a project.
pub fn revalidate_chain(
    chain: &UpstreamChain,
    placement_epoch: u64,
    live: &dyn StageLiveness,
) -> Result<(), ChainFault> {
    revalidate_chain_scoped(&PlacementScope::Legacy, chain, placement_epoch, live)
}

/// [`revalidate_chain`] with the placement epoch read for `project`
/// ([`StageLiveness::placement_epoch_in`]): a stage of the same name placed for another
/// project, at whatever epoch, is not this chain's stage.
pub fn revalidate_chain_in(
    project: &ProjectId,
    chain: &UpstreamChain,
    placement_epoch: u64,
    live: &dyn StageLiveness,
) -> Result<(), ChainFault> {
    revalidate_chain_scoped(
        &PlacementScope::Project(project.clone()),
        chain,
        placement_epoch,
        live,
    )
}

fn revalidate_chain_scoped(
    scope: &PlacementScope,
    chain: &UpstreamChain,
    placement_epoch: u64,
    live: &dyn StageLiveness,
) -> Result<(), ChainFault> {
    revalidate_element(&chain.stage, live)?;
    let current = match scope {
        PlacementScope::Legacy => live.placement_epoch(&chain.stage.identity),
        PlacementScope::Project(project) => live.placement_epoch_in(project, &chain.stage.identity),
    };
    if current != Some(placement_epoch) {
        return Err(ChainFault::StalePlacement {
            recorded: placement_epoch,
            current,
        });
    }
    revalidate_element(&chain.base_model, live)?;
    for adapter in &chain.adapters {
        revalidate_element(adapter, live)?;
    }
    for expert in &chain.experts {
        revalidate_element(expert, live)?;
        let current = live.fence_epoch(&expert.identity);
        if expert.fence_epoch == 0 || current != Some(expert.fence_epoch) {
            return Err(ChainFault::StaleFence {
                identity: expert.identity.clone(),
                recorded: expert.fence_epoch,
                current,
            });
        }
    }
    Ok(())
}

/// A frame whose chain and placement epoch were live when it was admitted: activations and
/// positions to compute from, and nothing a cache could take for KV.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedFrame {
    scope: PlacementScope,
    frame_digest: [u8; 32],
    chain: UpstreamChain,
    placement_epoch: u64,
    rows: u32,
    width: u32,
    positions: Vec<u32>,
    activations: Vec<f32>,
}

impl AdmittedFrame {
    /// The project the frame was admitted for (`None`: admitted through the legacy
    /// single-project [`admit_frame`]).
    pub fn project(&self) -> Option<&ProjectId> {
        self.scope.project()
    }

    pub fn frame_digest(&self) -> &[u8; 32] {
        &self.frame_digest
    }

    pub fn chain(&self) -> &UpstreamChain {
        &self.chain
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

    /// Revalidate the frame's chain and placement epoch against `live` now. Admission checks
    /// once; a caller that computes from the frame (a prefill has no use ticket, unlike a cache
    /// hit) calls this before it releases the computed output, so a revocation between
    /// admission and release is caught. The placement is read in the namespace the frame was
    /// admitted in (its project, or the legacy namespace).
    pub fn revalidate(&self, live: &dyn StageLiveness) -> Result<(), ChainFault> {
        revalidate_chain_scoped(&self.scope, &self.chain, self.placement_epoch, live)
    }

    /// `rows x width` activations, row-major.
    pub fn activations(&self) -> &[f32] {
        &self.activations
    }
}

/// Admit a frame for compute: its chain and placement epoch are revalidated now.
///
/// Version 1 is **Exact-only** (doc 36, section D): `upstream_mode` is the reuse mode the
/// upstream stage ran under, as the caller knows it, and an Approximate upstream is refused
/// with [`ChainFault::NotExact`] because the frame cannot carry a taint. The activations
/// must be finite.
///
/// The placement epoch is read in the **legacy single-project namespace**; a caller that serves
/// a project uses [`admit_frame_in`].
pub fn admit_frame(
    frame: &ActivationFrame,
    upstream_mode: KvReuseMode,
    live: &dyn StageLiveness,
) -> Result<AdmittedFrame, ChainFault> {
    admit_frame_scoped(PlacementScope::Legacy, frame, upstream_mode, live)
}

/// [`admit_frame`] for `project`: the placement epoch is read for that project
/// ([`StageLiveness::placement_epoch_in`]), so a frame whose stage name and epoch match
/// another project's placement is refused with [`ChainFault::StalePlacement`]. The frame
/// stays bound to the project: [`AdmittedFrame::revalidate`], the cache key and every use of
/// a cache entry derived from it check the same project's placement.
pub fn admit_frame_in(
    project: &ProjectId,
    frame: &ActivationFrame,
    upstream_mode: KvReuseMode,
    live: &dyn StageLiveness,
) -> Result<AdmittedFrame, ChainFault> {
    admit_frame_scoped(
        PlacementScope::Project(project.clone()),
        frame,
        upstream_mode,
        live,
    )
}

fn admit_frame_scoped(
    scope: PlacementScope,
    frame: &ActivationFrame,
    upstream_mode: KvReuseMode,
    live: &dyn StageLiveness,
) -> Result<AdmittedFrame, ChainFault> {
    if upstream_mode != KvReuseMode::Exact {
        return Err(ChainFault::NotExact {
            mode: upstream_mode,
        });
    }
    revalidate_chain_scoped(&scope, &frame.upstream_chain, frame.placement_epoch, live)?;
    let activations = frame.tensor_f32();
    if let Some(index) = activations.iter().position(|value| !value.is_finite()) {
        return Err(ChainFault::NonFinite { index });
    }
    Ok(AdmittedFrame {
        scope,
        frame_digest: frame_digest(&frame.to_bytes()),
        chain: frame.upstream_chain.clone(),
        placement_epoch: frame.placement_epoch,
        rows: frame.tokens,
        width: frame.width,
        positions: frame.positions.clone(),
        activations,
    })
}

/// The key of the downstream entry derived from `frames`, in order, by the downstream stage
/// described by `own`: it includes the digest of every frame (a frame's digest covers its
/// chain and placement epoch) and the downstream stage's own chain, so two downstream
/// generations never share a key. Frames admitted for a project also bind the project into
/// the key, so the same frame admitted for two projects never shares an entry; the key of a
/// legacy-admitted frame is unchanged.
pub fn prefix_key(frames: &[&AdmittedFrame], own: &UpstreamChain) -> [u8; 32] {
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
    }
    hasher.finalize().into()
}

/// Why an entry could not be stored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoreError {
    /// No frames: an entry derives from at least one.
    NoFrames,
    /// The frames do not share one upstream chain, placement epoch and project, so no single
    /// chain can be recorded for what was computed from them.
    ChainMismatch,
    /// The cache holds its maximum of entries.
    Full,
}

/// Why a lookup or a use was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UseFault {
    /// No entry under the key (never stored, or dropped by an invalidation).
    Miss,
    /// The entry's recorded chain or placement is not live now.
    Stale(ChainFault),
    /// The ticket is not an open use of this cache.
    UnknownTicket,
    /// The use was cancelled by an invalidation while it ran.
    Cancelled,
    /// The cache already holds its maximum of open uses (`limit`); finish or abandon one first.
    TooManyUses { limit: usize },
}

/// An open use of an entry.
#[derive(Debug, Eq, PartialEq)]
pub struct UseTicket {
    id: u64,
}

struct Entry<V> {
    /// The namespace the placement epoch is read in (the project the frames were admitted for).
    scope: PlacementScope,
    /// The downstream stage's own stage, base model, adapters and experts.
    own: UpstreamChain,
    chain: UpstreamChain,
    placement_epoch: u64,
    value: V,
}

struct OpenUse {
    key: [u8; 32],
    cancelled: bool,
}

/// How many uses may be open per entry of the cache's capacity. An open use is a begun and
/// not yet finished one; the bound keeps a caller that never finishes from growing the table.
pub const OPEN_USES_PER_ENTRY: usize = 8;

/// The downstream cache of split frames, bounded and never persisted.
pub struct SplitStageCache<V> {
    entries: BTreeMap<[u8; 32], Entry<V>>,
    uses: BTreeMap<u64, OpenUse>,
    next_use: u64,
    capacity: usize,
}

impl<V> SplitStageCache<V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            uses: BTreeMap::new(),
            next_use: 1,
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Open uses not yet finished or cancelled-and-collected.
    pub fn open_uses(&self) -> usize {
        self.uses.len()
    }

    /// The most uses that may be open at once: [`OPEN_USES_PER_ENTRY`] per entry of capacity
    /// (at least one).
    pub fn max_open_uses(&self) -> usize {
        self.capacity.saturating_mul(OPEN_USES_PER_ENTRY).max(1)
    }

    /// Store `value`, computed from `frames` in order by the downstream stage `own`, under
    /// their prefix key. The entry records the chain and placement epoch the frames carried
    /// and `own`; both are revalidated at every use. A full cache first drops the entries
    /// that are no longer live (a `Revoke` message that was lost must not leave it stuck).
    pub fn store(
        &mut self,
        frames: &[&AdmittedFrame],
        own: &UpstreamChain,
        value: V,
        live: &dyn StageLiveness,
    ) -> Result<[u8; 32], StoreError> {
        let first = frames.first().ok_or(StoreError::NoFrames)?;
        if frames.iter().any(|frame| {
            frame.chain != first.chain || frame.placement_epoch != first.placement_epoch
        }) {
            return Err(StoreError::ChainMismatch);
        }
        // One entry is revalidated in one project's placement namespace.
        if frames.iter().any(|frame| frame.scope != first.scope) {
            return Err(StoreError::ChainMismatch);
        }
        let key = prefix_key(frames, own);
        if !self.entries.contains_key(&key) && self.entries.len() >= self.capacity {
            self.evict_dead(live);
            if self.entries.len() >= self.capacity {
                return Err(StoreError::Full);
            }
        }
        self.entries.insert(
            key,
            Entry {
                scope: first.scope.clone(),
                own: own.clone(),
                chain: first.chain.clone(),
                placement_epoch: first.placement_epoch,
                value,
            },
        );
        Ok(key)
    }

    /// Drop every entry that fails revalidation now, cancelling the uses of it. Returns the
    /// number dropped.
    pub fn evict_dead(&mut self, live: &dyn StageLiveness) -> usize {
        let dead: Vec<[u8; 32]> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry_fault(entry, live).is_some())
            .map(|(key, _)| *key)
            .collect();
        for key in &dead {
            self.drop_entry(key);
        }
        dead.len()
    }

    fn drop_entry(&mut self, key: &[u8; 32]) {
        self.entries.remove(key);
        for open in self.uses.values_mut() {
            if open.key == *key {
                open.cancelled = true;
            }
        }
    }

    /// The entry under `key` if its chain, its placement and its own stage are live now. An
    /// entry that is not live is **evicted on the spot**, so it neither serves nor occupies
    /// a slot.
    fn live_entry(
        &mut self,
        key: &[u8; 32],
        live: &dyn StageLiveness,
    ) -> Result<&Entry<V>, UseFault> {
        let entry = self.entries.get(key).ok_or(UseFault::Miss)?;
        if let Some(fault) = entry_fault(entry, live) {
            self.drop_entry(key);
            return Err(UseFault::Stale(fault));
        }
        self.entries.get(key).ok_or(UseFault::Miss)
    }

    /// The value under `key`, if the entry's chain, placement and own stage are live now.
    pub fn lookup(&mut self, key: &[u8; 32], live: &dyn StageLiveness) -> Result<&V, UseFault> {
        self.live_entry(key, live).map(|entry| &entry.value)
    }

    /// Begin using the entry under `key`: revalidated now, and registered so an invalidation
    /// during the use cancels it. At most [`Self::max_open_uses`] uses are open at once; the
    /// next is refused [`UseFault::TooManyUses`].
    pub fn begin_use(
        &mut self,
        key: &[u8; 32],
        live: &dyn StageLiveness,
    ) -> Result<UseTicket, UseFault> {
        self.live_entry(key, live)?;
        if self.uses.len() >= self.max_open_uses() {
            return Err(UseFault::TooManyUses {
                limit: self.max_open_uses(),
            });
        }
        let id = self.next_use;
        self.next_use += 1;
        self.uses.insert(
            id,
            OpenUse {
                key: *key,
                cancelled: false,
            },
        );
        Ok(UseTicket { id })
    }

    /// End a use. The output of the use may be released only when this returns `Ok`: the
    /// entry must still exist, the use must not have been cancelled, and the chain and
    /// placement must be live at this moment too.
    pub fn finish_use(
        &mut self,
        ticket: UseTicket,
        live: &dyn StageLiveness,
    ) -> Result<(), UseFault> {
        let open = self
            .uses
            .remove(&ticket.id)
            .ok_or(UseFault::UnknownTicket)?;
        if open.cancelled {
            return Err(UseFault::Cancelled);
        }
        self.live_entry(&open.key, live).map(|_| ())
    }

    /// Give a use up without releasing any output (the compute failed or was dropped between
    /// [`Self::begin_use`] and [`Self::finish_use`]): its slot is freed. A ticket that is not
    /// an open use of this cache (already finished, abandoned or cleared) is
    /// [`UseFault::UnknownTicket`].
    pub fn abandon_use(&mut self, ticket: UseTicket) -> Result<(), UseFault> {
        self.uses
            .remove(&ticket.id)
            .map(|_| ())
            .ok_or(UseFault::UnknownTicket)
    }

    /// The `Revoke` message path: drop every entry derived from the stage `identity` at
    /// `generation` and cancel the uses of them. Returns the entries dropped. This is an
    /// optimisation of freeing early, never the guarantee: lookups and uses revalidate.
    pub fn invalidate_stage(&mut self, identity: &str, generation: Generation) -> usize {
        let doomed: Vec<[u8; 32]> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.chain.stage.identity == identity
                    && entry.chain.stage.generation == generation.0
            })
            .map(|(key, _)| *key)
            .collect();
        for key in &doomed {
            self.drop_entry(key);
        }
        doomed.len()
    }

    /// Drop everything, as a restart does: the entries and the open uses (their slots are
    /// released; a ticket from before the call is [`UseFault::UnknownTicket`], so no output of
    /// it is released either).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.uses.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptr_protocol::activation::BoundaryDtype;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

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

    fn element(name: &str, generation: u64) -> ChainElement {
        ChainElement {
            identity: name.to_owned(),
            generation,
            digest: [generation as u8; 32],
            fence_epoch: 0,
        }
    }

    fn live() -> Live {
        let live = Live::default();
        live.commit("stage", 1);
        live.commit("model", 1);
        live.commit("down", 1);
        live.placement.borrow_mut().insert("stage".to_owned(), 5);
        live
    }

    fn frame(chain: UpstreamChain, epoch: u64, fill: f32) -> ActivationFrame {
        ActivationFrame {
            tokens: 2,
            width: 2,
            dtype: BoundaryDtype::F32,
            positions: vec![0, 1],
            upstream_chain: chain,
            placement_epoch: epoch,
            tensor: [fill, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    fn chain() -> UpstreamChain {
        UpstreamChain {
            stage: element("stage", 1),
            base_model: element("model", 1),
            adapters: vec![],
            experts: vec![],
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

    fn admitted(live: &Live, fill: f32) -> AdmittedFrame {
        admit_frame(&frame(chain(), 5, fill), KvReuseMode::Exact, live).unwrap()
    }

    #[test]
    fn a_live_chain_is_admitted_and_the_activations_read_back() {
        let live = live();
        let frame = admitted(&live, 1.0);
        assert_eq!(frame.rows(), 2);
        assert_eq!(frame.width(), 2);
        assert_eq!(frame.positions(), [0, 1]);
        assert_eq!(frame.activations(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn each_element_of_the_chain_and_the_placement_is_revalidated() {
        let live = live();
        live.commit("adapter", 3);
        let with_adapter = UpstreamChain {
            adapters: vec![element("adapter", 3)],
            ..chain()
        };
        assert!(admit_frame(
            &frame(with_adapter.clone(), 5, 1.0),
            KvReuseMode::Exact,
            &live
        )
        .is_ok());
        live.set("adapter", 3, Validity::Revoked);
        assert!(matches!(
            admit_frame(&frame(with_adapter, 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::NotLive {
                standing: Some(Validity::Revoked),
                ..
            })
        ));
        live.set("model", 1, Validity::Superseded);
        assert!(matches!(
            admit_frame(&frame(chain(), 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::NotLive {
                standing: Some(Validity::Superseded),
                ..
            })
        ));
        live.set("model", 1, Validity::Live);
        live.set("stage", 1, Validity::Revoked);
        assert!(matches!(
            admit_frame(&frame(chain(), 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::NotLive { .. })
        ));
        live.set("stage", 1, Validity::Live);
        // A stage re-placed without a generation change: the old worker's frame is stale.
        live.placement.borrow_mut().insert("stage".to_owned(), 6);
        assert_eq!(
            admit_frame(&frame(chain(), 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::StalePlacement {
                recorded: 5,
                current: Some(6)
            })
        );
        live.placement.borrow_mut().remove("stage");
        assert_eq!(
            admit_frame(&frame(chain(), 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::StalePlacement {
                recorded: 5,
                current: None
            })
        );
    }

    #[test]
    fn a_digest_that_is_not_the_committed_one_is_refused() {
        let live = live();
        let mut forged = chain();
        forged.stage.digest = [9; 32];
        assert_eq!(
            admit_frame(&frame(forged, 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::DigestMismatch {
                identity: "stage".to_owned()
            })
        );
    }

    #[test]
    fn an_expert_must_record_its_fence_epoch_and_still_serve_under_it() {
        let live = live();
        live.commit("expert", 1);
        let with = |epoch: u64| UpstreamChain {
            experts: vec![ChainElement {
                fence_epoch: epoch,
                ..element("expert", 1)
            }],
            ..chain()
        };
        live.fence.borrow_mut().insert("expert".to_owned(), 4);
        assert!(admit_frame(&frame(with(4), 5, 1.0), KvReuseMode::Exact, &live).is_ok());
        assert!(matches!(
            admit_frame(&frame(with(3), 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::StaleFence {
                recorded: 3,
                current: Some(4),
                ..
            })
        ));
        // An entry that recorded no epoch is invalidated by every fence: fail closed.
        assert!(matches!(
            admit_frame(&frame(with(0), 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::StaleFence { recorded: 0, .. })
        ));
        live.fence.borrow_mut().remove("expert");
        assert!(matches!(
            admit_frame(&frame(with(4), 5, 1.0), KvReuseMode::Exact, &live),
            Err(ChainFault::StaleFence { current: None, .. })
        ));
    }

    #[test]
    fn a_fenced_expert_of_the_own_chain_ends_a_v1_entry_without_the_revoke_message() {
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
        let first = admitted(&live, 1.0);
        let mut cache: SplitStageCache<&str> = SplitStageCache::new(4);
        let key = cache.store(&[&first], &own, "kv", &live).unwrap();
        assert_eq!(cache.lookup(&key, &live), Ok(&"kv"));
        // Fenced: the generation stays live, only the committed epoch moves.
        live.fence.borrow_mut().insert("own-expert".to_owned(), 3);
        assert!(cache.lookup(&key, &live).is_err());
    }

    #[test]
    fn an_entry_is_served_while_its_chain_is_live_and_refused_after_a_revocation_nobody_announced()
    {
        let live = live();
        let first = admitted(&live, 1.0);
        let mut cache: SplitStageCache<&str> = SplitStageCache::new(4);
        let key = cache.store(&[&first], &own_chain(), "kv", &live).unwrap();
        assert_eq!(cache.lookup(&key, &live), Ok(&"kv"));
        // The authority revokes; the Revoke message is dropped, so the cache is never told.
        live.set("stage", 1, Validity::Revoked);
        assert!(matches!(
            cache.lookup(&key, &live),
            Err(UseFault::Stale(ChainFault::NotLive {
                standing: Some(Validity::Revoked),
                ..
            }))
        ));
        assert_eq!(cache.len(), 0, "the check at use refused it and evicted it");
    }

    #[test]
    fn a_superseded_stage_serves_nothing_from_its_old_generation_and_the_new_one_its_own() {
        let live = live();
        let old = admitted(&live, 1.0);
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(4);
        let old_key = cache.store(&[&old], &own_chain(), 1, &live).unwrap();
        live.commit("stage", 2);
        live.set("stage", 1, Validity::Superseded);
        assert!(cache.lookup(&old_key, &live).is_err());
        let next_chain = UpstreamChain {
            stage: element("stage", 2),
            ..chain()
        };
        let new = admit_frame(&frame(next_chain, 5, 1.0), KvReuseMode::Exact, &live).unwrap();
        let new_key = cache.store(&[&new], &own_chain(), 2, &live).unwrap();
        assert_ne!(old_key, new_key, "the digest of a frame covers its chain");
        assert_eq!(cache.lookup(&new_key, &live), Ok(&2));
    }

    #[test]
    fn an_admitted_frame_is_revalidated_before_its_output_is_released() {
        let live = live();
        let frame = admitted(&live, 1.0);
        assert_eq!(frame.revalidate(&live), Ok(()));
        live.set("stage", 1, Validity::Revoked);
        assert!(matches!(
            frame.revalidate(&live),
            Err(ChainFault::NotLive { .. })
        ));
    }

    #[test]
    fn open_uses_are_bounded_and_a_finished_use_frees_its_slot() {
        let live = live();
        let frame = admitted(&live, 1.0);
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(1);
        let key = cache.store(&[&frame], &own_chain(), 7, &live).unwrap();
        let limit = cache.max_open_uses();
        assert_eq!(limit, OPEN_USES_PER_ENTRY);
        let mut tickets = Vec::new();
        for _ in 0..limit {
            tickets.push(cache.begin_use(&key, &live).unwrap());
        }
        assert_eq!(cache.open_uses(), limit);
        assert_eq!(
            cache.begin_use(&key, &live),
            Err(UseFault::TooManyUses { limit })
        );
        assert_eq!(cache.open_uses(), limit, "a refusal registers nothing");
        let first = tickets.remove(0);
        assert_eq!(cache.finish_use(first, &live), Ok(()));
        assert!(cache.begin_use(&key, &live).is_ok());
        // A miss is still a miss, not a bound.
        assert_eq!(cache.begin_use(&[0; 32], &live), Err(UseFault::Miss));
    }

    #[test]
    fn an_abandoned_use_frees_its_slot_and_a_clear_frees_them_all() {
        let live = live();
        let frame = admitted(&live, 1.0);
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(1);
        let key = cache.store(&[&frame], &own_chain(), 7, &live).unwrap();
        let limit = cache.max_open_uses();
        let mut tickets: Vec<UseTicket> = (0..limit)
            .map(|_| cache.begin_use(&key, &live).unwrap())
            .collect();
        assert_eq!(
            cache.begin_use(&key, &live),
            Err(UseFault::TooManyUses { limit })
        );
        // A compute that failed gives its use up: the slot is free, no output is released.
        assert_eq!(cache.abandon_use(tickets.remove(0)), Ok(()));
        assert_eq!(cache.open_uses(), limit - 1);
        let again = cache.begin_use(&key, &live).unwrap();
        assert_eq!(cache.finish_use(again, &live), Ok(()));
        // A restart-style clear releases every slot; a ticket of before it is unknown.
        let stale = tickets.remove(0);
        cache.clear();
        assert_eq!(cache.open_uses(), 0);
        assert_eq!(cache.abandon_use(stale), Err(UseFault::UnknownTicket));
        let key = cache.store(&[&frame], &own_chain(), 7, &live).unwrap();
        for _ in 0..limit {
            cache.begin_use(&key, &live).unwrap();
        }
        let leaked = tickets.pop().unwrap();
        assert_eq!(
            cache.finish_use(leaked, &live),
            Err(UseFault::UnknownTicket)
        );
    }

    #[test]
    fn a_use_in_flight_when_the_chain_is_invalidated_releases_no_output() {
        let live = live();
        let frame = admitted(&live, 1.0);
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(4);
        let key = cache.store(&[&frame], &own_chain(), 7, &live).unwrap();
        // Invalidation with the Revoke message dropped: the check at the end of the use.
        let ticket = cache.begin_use(&key, &live).unwrap();
        live.set("stage", 1, Validity::Revoked);
        assert!(matches!(
            cache.finish_use(ticket, &live),
            Err(UseFault::Stale(_))
        ));
        // Invalidation with the message delivered: the use is cancelled.
        live.set("stage", 1, Validity::Live);
        let key = cache.store(&[&frame], &own_chain(), 7, &live).unwrap();
        let ticket = cache.begin_use(&key, &live).unwrap();
        assert_eq!(cache.invalidate_stage("stage", Generation(1)), 1);
        assert_eq!(cache.finish_use(ticket, &live), Err(UseFault::Cancelled));
        assert_eq!(cache.open_uses(), 0);
        assert_eq!(cache.lookup(&key, &live), Err(UseFault::Miss));
        // A finished use of a live entry releases its output once.
        let again = admitted(&live, 1.0);
        let key = cache.store(&[&again], &own_chain(), 7, &live).unwrap();
        let ticket = cache.begin_use(&key, &live).unwrap();
        assert_eq!(cache.finish_use(ticket, &live), Ok(()));
        assert_eq!(
            cache.finish_use(UseTicket { id: 999 }, &live),
            Err(UseFault::UnknownTicket)
        );
    }

    #[test]
    fn a_lost_revoke_cannot_leave_the_cache_stuck_full() {
        let live = live();
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(2);
        let a = admitted(&live, 1.0);
        let b = admitted(&live, 2.0);
        let c = admitted(&live, 3.0);
        cache.store(&[&a], &own_chain(), 1, &live).unwrap();
        cache.store(&[&b], &own_chain(), 2, &live).unwrap();
        assert_eq!(
            cache.store(&[&c], &own_chain(), 3, &live),
            Err(StoreError::Full)
        );
        // The authority revokes the upstream stage; no Revoke message ever arrives. The
        // dead entries give way to a new one instead of wedging the cache.
        live.set("stage", 1, Validity::Revoked);
        live.commit("stage", 2);
        let fresh = admit_frame(
            &frame(
                UpstreamChain {
                    stage: element("stage", 2),
                    ..chain()
                },
                5,
                9.0,
            ),
            KvReuseMode::Exact,
            &live,
        )
        .unwrap();
        assert!(cache.store(&[&fresh], &own_chain(), 4, &live).is_ok());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn the_downstream_stage_itself_is_part_of_the_key_and_is_revalidated() {
        let live = live();
        let frame = admitted(&live, 1.0);
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(4);
        let first = cache.store(&[&frame], &own_chain(), 1, &live).unwrap();
        live.commit("down", 2);
        let second_chain = UpstreamChain {
            stage: element("down", 2),
            ..own_chain()
        };
        let second = cache.store(&[&frame], &second_chain, 2, &live).unwrap();
        assert_ne!(first, second, "two downstream generations share no key");
        // The downstream stage's old generation is revoked: only its own entry goes.
        live.set("down", 1, Validity::Revoked);
        assert!(matches!(
            cache.lookup(&first, &live),
            Err(UseFault::Stale(ChainFault::NotLive { .. }))
        ));
        assert_eq!(cache.lookup(&second, &live), Ok(&2));
    }

    #[test]
    fn version_one_is_exact_only_and_refuses_non_finite_activations() {
        let live = live();
        assert_eq!(
            admit_frame(&frame(chain(), 5, 1.0), KvReuseMode::Approximate, &live),
            Err(ChainFault::NotExact {
                mode: KvReuseMode::Approximate
            })
        );
        assert_eq!(
            admit_frame(&frame(chain(), 5, f32::NAN), KvReuseMode::Exact, &live),
            Err(ChainFault::NonFinite { index: 0 })
        );
        assert_eq!(
            admit_frame(&frame(chain(), 5, f32::INFINITY), KvReuseMode::Exact, &live),
            Err(ChainFault::NonFinite { index: 0 })
        );
    }

    #[test]
    fn frames_of_different_chains_cannot_back_one_entry() {
        let live = live();
        live.commit("stage", 2);
        let a = admitted(&live, 1.0);
        let other = UpstreamChain {
            stage: element("stage", 2),
            ..chain()
        };
        let b = admit_frame(&frame(other, 5, 2.0), KvReuseMode::Exact, &live).unwrap();
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(4);
        assert_eq!(
            cache.store(&[&a, &b], &own_chain(), 0, &live),
            Err(StoreError::ChainMismatch)
        );
        assert_eq!(
            cache.store(&[], &own_chain(), 0, &live),
            Err(StoreError::NoFrames)
        );
        assert!(cache.is_empty());
    }

    #[test]
    fn the_cache_is_bounded_and_a_restart_drops_it() {
        let live = live();
        let mut cache: SplitStageCache<u8> = SplitStageCache::new(1);
        let a = admitted(&live, 1.0);
        let b = admitted(&live, 2.0);
        cache.store(&[&a], &own_chain(), 0, &live).unwrap();
        assert_eq!(
            cache.store(&[&b], &own_chain(), 0, &live),
            Err(StoreError::Full)
        );
        cache.store(&[&a], &own_chain(), 1, &live).unwrap();
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn the_key_covers_the_order_and_the_digest_of_every_frame() {
        let live = live();
        let a = admitted(&live, 1.0);
        let b = admitted(&live, 2.0);
        assert_ne!(
            prefix_key(&[&a, &b], &own_chain()),
            prefix_key(&[&b, &a], &own_chain())
        );
        assert_ne!(
            prefix_key(&[&a], &own_chain()),
            prefix_key(&[&a, &b], &own_chain())
        );
        assert_ne!(
            prefix_key(&[&a], &own_chain()),
            prefix_key(&[&b], &own_chain())
        );
    }
}
