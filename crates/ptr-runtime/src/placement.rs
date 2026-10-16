use ptr_types::{ArtifactId, DeviceId, Generation, NodeId, PodId, ProjectId, StateId, Timestamp};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct PlacementEpoch(pub u64);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FencingToken(pub u128);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeHealth {
    Healthy,
    Stale,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceHealth {
    Healthy,
    Degraded,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceRecord {
    pub device_id: DeviceId,
    pub vram_bytes: u64,
    pub used_vram_bytes: u64,
    pub health: DeviceHealth,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NodeRecord {
    pub node_id: NodeId,
    pub zone: String,
    pub devices: Vec<DeviceRecord>,
    pub health: NodeHealth,
    pub last_heartbeat: Timestamp,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Placement {
    pub pod_id: PodId,
    pub artifact_id: ArtifactId,
    pub generation: Generation,
    pub node_id: NodeId,
    pub device_id: DeviceId,
    pub epoch: PlacementEpoch,
}

/// The namespace a placement, a lease and a freeze live in.
///
/// Nodes and devices are host resources and stay global; what is placed on them belongs to a
/// project. `Legacy` is the single-project namespace of the original API (`assign`,
/// `placement`, `issue_lease`, ...): it is a namespace of its own, never equal to any
/// `ProjectId`, so a project can neither read nor disturb what the legacy API placed.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum PlacementScope {
    Legacy,
    Project(ProjectId),
}

impl PlacementScope {
    pub(crate) fn project(&self) -> Option<&ProjectId> {
        match self {
            Self::Legacy => None,
            Self::Project(project) => Some(project),
        }
    }
}

/// A project-scoped placement request for [`PodPlacementController::assign_in`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectPlacementRequest {
    /// The project the placement belongs to (authority-derived by the caller).
    pub project: ProjectId,
    pub pod_id: PodId,
    pub artifact_id: ArtifactId,
    pub generation: Generation,
    pub node_id: NodeId,
    pub device_id: DeviceId,
    pub required_vram: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub struct FencedStateLease {
    pub(crate) scope: PlacementScope,
    pub(crate) pod_id: PodId,
    pub(crate) state_id: StateId,
    pub(crate) node_id: NodeId,
    pub(crate) device_id: DeviceId,
    pub(crate) placement_epoch: PlacementEpoch,
    pub(crate) fencing_token: FencingToken,
    pub(crate) generation: Generation,
}

impl FencedStateLease {
    /// The project the lease was issued in (`None`: issued through the legacy single-project
    /// API).
    pub fn project(&self) -> Option<&ProjectId> {
        self.scope.project()
    }
    pub fn pod_id(&self) -> &PodId {
        &self.pod_id
    }
    pub fn state_id(&self) -> &StateId {
        &self.state_id
    }
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }
    pub fn placement_epoch(&self) -> PlacementEpoch {
        self.placement_epoch
    }
    pub fn fencing_token(&self) -> FencingToken {
        self.fencing_token
    }
    pub fn generation(&self) -> Generation {
        self.generation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlacementError {
    UnknownNode(NodeId),
    UnknownDevice(DeviceId),
    UnhealthyNode(NodeId),
    UnhealthyDevice(DeviceId),
    InsufficientVram,
    StaleHeartbeat(NodeId),
    StaleLease,
    StateFrozen,
    NoPlacement,
    CounterExhausted,
    /// A node record or heartbeat lists the same device twice; it is refused whole, because
    /// two records of one device could report it healthy and failed at once.
    DuplicateDevice(DeviceId),
}

/// The most revoked placements kept to name a failure in `issue_lease`. Past it the oldest
/// (lowest epoch) is dropped, after which `issue_lease` for that Pod answers `NoPlacement`.
///
/// The bound is shared by every project: a flood of revoked placements in one project can push
/// another project's tombstone out, so that project's `issue_lease_in` answers `NoPlacement`
/// instead of naming the failed node or device. There is no safety impact (both are refusals).
const MAX_REVOKED_PLACEMENTS: usize = 1024;

pub struct PodPlacementController {
    nodes: BTreeMap<NodeId, NodeRecord>,
    /// Keyed by (scope, pod): the same `PodId` may be placed in two projects.
    placements: BTreeMap<(PlacementScope, PodId), Placement>,
    /// Placements revoked by a topology event, kept only so that `issue_lease` can still name
    /// the node or device that failed. They are never served by `placement` and never revived.
    /// At most one entry per (scope, pod) and at most `MAX_REVOKED_PLACEMENTS` entries (the oldest is
    /// dropped first); `assign` and the first `issue_lease` that finds the failure gone prune it.
    revoked: BTreeMap<(PlacementScope, PodId), Placement>,
    leases: BTreeMap<(PlacementScope, StateId), FencedStateLease>,
    frozen: BTreeMap<(PlacementScope, StateId), PlacementEpoch>,
    epoch: PlacementEpoch,
    next_token: u128,
}

impl Default for PodPlacementController {
    fn default() -> Self {
        Self {
            nodes: BTreeMap::new(),
            placements: BTreeMap::new(),
            revoked: BTreeMap::new(),
            leases: BTreeMap::new(),
            frozen: BTreeMap::new(),
            epoch: PlacementEpoch(0),
            next_token: 1,
        }
    }
}

impl PodPlacementController {
    pub fn epoch(&self) -> PlacementEpoch {
        self.epoch
    }

    /// Register a node, or replace the record of a node already registered.
    ///
    /// Registering a `NodeId` that is already known is a **replacement** (a new node process or
    /// a new host under the same id), never a refresh: a refresh is a [`Self::heartbeat`]. The
    /// record carries no incarnation, so the controller cannot tell that the replacement offers
    /// the same devices with the same contents; every placement and every lease on that node is
    /// revoked, and frames produced by the previous incarnation are refused by split-stage
    /// admission (`placement` answers `NoPlacement`). Re-placement goes through `assign` under
    /// a new epoch. The controller epoch moves once for the event.
    pub fn register_node(&mut self, node: NodeRecord) -> Result<(), PlacementError> {
        ensure_unique_devices(&node.devices)?;
        let node_id = node.node_id.clone();
        let replaced = self.nodes.insert(node_id.clone(), node).is_some();
        if replaced {
            self.revoke_placements_on_node(&node_id);
            self.revoke_for_node(&node_id);
        }
        self.bump_epoch()?;
        Ok(())
    }

    pub fn heartbeat(
        &mut self,
        node_id: &NodeId,
        timestamp: Timestamp,
        devices: Vec<DeviceRecord>,
    ) -> Result<(), PlacementError> {
        ensure_unique_devices(&devices)?;
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| PlacementError::UnknownNode(node_id.clone()))?;
        if timestamp.0 < node.last_heartbeat.0 {
            return Err(PlacementError::StaleHeartbeat(node_id.clone()));
        }
        node.last_heartbeat = timestamp;
        node.devices = devices;
        node.health = NodeHealth::Healthy;
        // A heartbeat reports the devices as they are now. Leases on a device
        // that disappeared or failed are revoked here; a lease on a device that
        // is still usable is untouched, and no lease is ever brought back by a
        // heartbeat.
        //
        // A placement on a device that disappeared or reports `Failed` is revoked the
        // same way: it is removed from `placements` and the controller epoch is bumped
        // once for the event, so a frame produced under it is refused by split-stage
        // admission (`placement` answers `NoPlacement`). A `Degraded` device keeps its
        // placements, and a heartbeat that changes nothing keeps every placement and
        // leaves the epoch alone. A later heartbeat never restores a placement;
        // re-placement goes through `assign` under a new epoch.
        let revoked = self.revoke_unavailable_placements(node_id);
        self.revoke_unavailable_leases(node_id);
        if revoked {
            self.bump_epoch()?;
        }
        Ok(())
    }

    pub fn mark_node_failed(&mut self, node_id: &NodeId) -> Result<(), PlacementError> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| PlacementError::UnknownNode(node_id.clone()))?;
        node.health = NodeHealth::Failed;
        // Every placement on the node is revoked, then the epoch moves once.
        self.revoke_unavailable_placements(node_id);
        self.revoke_for_node(node_id);
        self.bump_epoch()
    }

    /// Place a Pod in the **legacy single-project namespace** (the API before placements were
    /// project-scoped). It is kept unchanged for single-project callers and for completed
    /// experiments; new code that serves a project uses [`Self::assign_in`], because this
    /// namespace is shared by every caller of the legacy API.
    pub fn assign(
        &mut self,
        pod_id: PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        node_id: NodeId,
        device_id: DeviceId,
        required_vram: u64,
    ) -> Result<Placement, PlacementError> {
        self.assign_scoped(
            PlacementScope::Legacy,
            pod_id,
            artifact_id,
            generation,
            node_id,
            device_id,
            required_vram,
        )
    }

    /// Place a Pod for `request.project`. The placement, its leases and its freezes live in
    /// that project's namespace: the same `PodId` placed for another project is a different
    /// placement, and neither disturbs the other. The controller epoch is still global (one
    /// monotonic fence), and nodes and devices are shared host resources.
    pub fn assign_in(
        &mut self,
        request: ProjectPlacementRequest,
    ) -> Result<Placement, PlacementError> {
        self.assign_scoped(
            PlacementScope::Project(request.project),
            request.pod_id,
            request.artifact_id,
            request.generation,
            request.node_id,
            request.device_id,
            request.required_vram,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_scoped(
        &mut self,
        scope: PlacementScope,
        pod_id: PodId,
        artifact_id: ArtifactId,
        generation: Generation,
        node_id: NodeId,
        device_id: DeviceId,
        required_vram: u64,
    ) -> Result<Placement, PlacementError> {
        let node = self
            .nodes
            .get(&node_id)
            .ok_or_else(|| PlacementError::UnknownNode(node_id.clone()))?;
        if node.health != NodeHealth::Healthy {
            return Err(PlacementError::UnhealthyNode(node_id));
        }
        let device = node
            .devices
            .iter()
            .find(|device| device.device_id == device_id)
            .ok_or_else(|| PlacementError::UnknownDevice(device_id.clone()))?;
        if device.health == DeviceHealth::Failed {
            return Err(PlacementError::UnhealthyDevice(device_id));
        }
        if device.vram_bytes.saturating_sub(device.used_vram_bytes) < required_vram {
            return Err(PlacementError::InsufficientVram);
        }
        self.bump_epoch()?;
        self.revoke_for_pod(&scope, &pod_id);
        // Hygiene only (issue_lease reads `placements` first): keeps the tombstone bound tight.
        self.revoked.remove(&(scope.clone(), pod_id.clone()));
        let placement = Placement {
            pod_id: pod_id.clone(),
            artifact_id,
            generation,
            node_id,
            device_id,
            epoch: self.epoch,
        };
        self.placements.insert((scope, pod_id), placement.clone());
        Ok(placement)
    }

    /// The placement of `pod_id` in the legacy single-project namespace (see [`Self::assign`]).
    pub fn placement(&self, pod_id: &PodId) -> Result<&Placement, PlacementError> {
        self.placement_scoped(&PlacementScope::Legacy, pod_id)
    }

    /// The placement of `pod_id` in `project`; a placement of another project, or one made
    /// through the legacy API, is `NoPlacement`.
    pub fn placement_in(
        &self,
        project: &ProjectId,
        pod_id: &PodId,
    ) -> Result<&Placement, PlacementError> {
        self.placement_scoped(&PlacementScope::Project(project.clone()), pod_id)
    }

    fn placement_scoped(
        &self,
        scope: &PlacementScope,
        pod_id: &PodId,
    ) -> Result<&Placement, PlacementError> {
        self.placements
            .get(&(scope.clone(), pod_id.clone()))
            .ok_or(PlacementError::NoPlacement)
    }

    /// Issue a lease in the legacy single-project namespace (see [`Self::assign`]).
    pub fn issue_lease(
        &mut self,
        state_id: StateId,
        pod_id: &PodId,
    ) -> Result<FencedStateLease, PlacementError> {
        self.issue_lease_scoped(PlacementScope::Legacy, state_id, pod_id)
    }

    /// Issue a lease for the placement of `pod_id` in `project`. The state id is scoped to the
    /// project too: the same `StateId` leased in two projects is two leases.
    pub fn issue_lease_in(
        &mut self,
        project: &ProjectId,
        state_id: StateId,
        pod_id: &PodId,
    ) -> Result<FencedStateLease, PlacementError> {
        self.issue_lease_scoped(PlacementScope::Project(project.clone()), state_id, pod_id)
    }

    fn issue_lease_scoped(
        &mut self,
        scope: PlacementScope,
        state_id: StateId,
        pod_id: &PodId,
    ) -> Result<FencedStateLease, PlacementError> {
        let pod_key = (scope.clone(), pod_id.clone());
        let Some(placement) = self.placements.get(&pod_key).cloned() else {
            // A revoked placement still names the failure that ended it; a revoked placement
            // whose node and device are usable again is gone, not resurrected.
            let Some(old) = self.revoked.get(&pod_key) else {
                return Err(PlacementError::NoPlacement);
            };
            return match self.ensure_usable(&old.node_id, &old.device_id) {
                Err(error) => Err(error),
                Ok(()) => {
                    // The failure is gone: drop the tombstone so the map cleans itself and a
                    // later failure of the same device is not blamed on this old placement.
                    self.revoked.remove(&pod_key);
                    Err(PlacementError::NoPlacement)
                }
            };
        };
        // The node and device the placement names are checked again before a lease is
        // issued.
        self.ensure_usable(&placement.node_id, &placement.device_id)?;
        let state_key = (scope.clone(), state_id.clone());
        if let Some(frozen_epoch) = self.frozen.get(&state_key).copied() {
            if frozen_epoch == placement.epoch {
                return Err(PlacementError::StateFrozen);
            }
            self.frozen.remove(&state_key);
        }
        let token = FencingToken(self.next_token);
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or(PlacementError::CounterExhausted)?;
        let lease = FencedStateLease {
            scope,
            pod_id: placement.pod_id.clone(),
            state_id: state_id.clone(),
            node_id: placement.node_id,
            device_id: placement.device_id,
            placement_epoch: placement.epoch,
            fencing_token: token,
            generation: placement.generation,
        };
        self.leases.insert(
            state_key,
            FencedStateLease {
                scope: lease.scope.clone(),
                pod_id: lease.pod_id.clone(),
                state_id: lease.state_id.clone(),
                node_id: lease.node_id.clone(),
                device_id: lease.device_id.clone(),
                placement_epoch: lease.placement_epoch,
                fencing_token: lease.fencing_token,
                generation: lease.generation,
            },
        );
        Ok(lease)
    }

    pub fn validate_lease(&self, lease: &FencedStateLease) -> Result<(), PlacementError> {
        let current = self
            .leases
            .get(&(lease.scope.clone(), lease.state_id.clone()))
            .ok_or(PlacementError::StaleLease)?;
        if current != lease {
            return Err(PlacementError::StaleLease);
        }
        Ok(())
    }

    /// Revoke the lease of `state_id` in the legacy single-project namespace.
    pub fn revoke_lease(&mut self, state_id: &StateId) -> Result<(), PlacementError> {
        self.revoke_lease_scoped(&PlacementScope::Legacy, state_id)
    }

    /// Revoke `lease` in the namespace it was issued in (its project, or the legacy namespace),
    /// whatever namespace holds a lease with the same state id.
    pub fn revoke_lease_for(&mut self, lease: &FencedStateLease) -> Result<(), PlacementError> {
        self.revoke_lease_scoped(&lease.scope, lease.state_id())
    }

    /// Revoke the lease of `state_id` in `project`; another project's lease is untouched.
    pub fn revoke_lease_in(
        &mut self,
        project: &ProjectId,
        state_id: &StateId,
    ) -> Result<(), PlacementError> {
        self.revoke_lease_scoped(&PlacementScope::Project(project.clone()), state_id)
    }

    fn revoke_lease_scoped(
        &mut self,
        scope: &PlacementScope,
        state_id: &StateId,
    ) -> Result<(), PlacementError> {
        self.leases
            .remove(&(scope.clone(), state_id.clone()))
            .map(|_| ())
            .ok_or(PlacementError::StaleLease)
    }

    /// Freeze the state a valid lease covers, in the namespace the lease was issued in.
    pub fn freeze_state(&mut self, lease: &FencedStateLease) -> Result<(), PlacementError> {
        self.validate_lease(lease)?;
        self.revoke_lease_scoped(&lease.scope, lease.state_id())?;
        self.frozen.insert(
            (lease.scope.clone(), lease.state_id().clone()),
            lease.placement_epoch(),
        );
        Ok(())
    }

    /// Lift a freeze in the legacy single-project namespace.
    pub fn unfreeze_state(&mut self, state_id: &StateId) {
        self.unfreeze_state_scoped(&PlacementScope::Legacy, state_id);
    }

    pub(crate) fn unfreeze_state_scoped(&mut self, scope: &PlacementScope, state_id: &StateId) {
        self.frozen.remove(&(scope.clone(), state_id.clone()));
    }

    /// Lift a freeze in `project`.
    pub fn unfreeze_state_in(&mut self, project: &ProjectId, state_id: &StateId) {
        self.frozen
            .remove(&(PlacementScope::Project(project.clone()), state_id.clone()));
    }

    fn bump_epoch(&mut self) -> Result<(), PlacementError> {
        self.epoch = PlacementEpoch(
            self.epoch
                .0
                .checked_add(1)
                .ok_or(PlacementError::CounterExhausted)?,
        );
        Ok(())
    }

    fn ensure_usable(&self, node_id: &NodeId, device_id: &DeviceId) -> Result<(), PlacementError> {
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| PlacementError::UnknownNode(node_id.clone()))?;
        if node.health != NodeHealth::Healthy {
            return Err(PlacementError::UnhealthyNode(node_id.clone()));
        }
        let device = node
            .devices
            .iter()
            .find(|device| device.device_id == *device_id)
            .ok_or_else(|| PlacementError::UnknownDevice(device_id.clone()))?;
        if device.health == DeviceHealth::Failed {
            return Err(PlacementError::UnhealthyDevice(device_id.clone()));
        }
        Ok(())
    }

    fn usable_devices(&self, node_id: &NodeId) -> Vec<DeviceId> {
        match self.nodes.get(node_id) {
            Some(node) if node.health != NodeHealth::Failed => node
                .devices
                .iter()
                .filter(|device| device.health != DeviceHealth::Failed)
                .map(|device| device.device_id.clone())
                .collect(),
            _ => Vec::new(),
        }
    }

    fn revoke_unavailable_leases(&mut self, node_id: &NodeId) {
        let usable = self.usable_devices(node_id);
        self.leases
            .retain(|_, lease| lease.node_id != *node_id || usable.contains(&lease.device_id));
    }

    /// Removes every placement on `node_id` whose device is no longer usable and reports
    /// whether any was removed. The caller bumps the epoch once for the event.
    fn revoke_unavailable_placements(&mut self, node_id: &NodeId) -> bool {
        let usable = self.usable_devices(node_id);
        let gone: Vec<(PlacementScope, PodId)> = self
            .placements
            .iter()
            .filter(|(_, p)| p.node_id == *node_id && !usable.contains(&p.device_id))
            .map(|(key, _)| key.clone())
            .collect();
        for key in &gone {
            if let Some(placement) = self.placements.remove(key) {
                self.tombstone(key.clone(), placement);
            }
        }
        !gone.is_empty()
    }

    /// Revoke every placement on `node_id`, usable device or not, keeping a tombstone each.
    fn revoke_placements_on_node(&mut self, node_id: &NodeId) {
        let gone: Vec<(PlacementScope, PodId)> = self
            .placements
            .iter()
            .filter(|(_, p)| p.node_id == *node_id)
            .map(|(key, _)| key.clone())
            .collect();
        for key in &gone {
            if let Some(placement) = self.placements.remove(key) {
                self.tombstone(key.clone(), placement);
            }
        }
    }

    /// Keep a revoked placement to name its failure, dropping the oldest past the bound.
    fn tombstone(&mut self, key: (PlacementScope, PodId), placement: Placement) {
        self.revoked.insert(key, placement);
        while self.revoked.len() > MAX_REVOKED_PLACEMENTS {
            let oldest = self
                .revoked
                .iter()
                .min_by_key(|(_, placement)| placement.epoch)
                .map(|(key, _)| key.clone());
            match oldest {
                Some(key) => {
                    self.revoked.remove(&key);
                }
                None => break,
            }
        }
    }

    /// The number of revoked placements kept to name a failure (bounded).
    pub fn revoked_placements(&self) -> usize {
        self.revoked.len()
    }

    fn revoke_for_pod(&mut self, scope: &PlacementScope, pod_id: &PodId) {
        self.leases
            .retain(|_, lease| lease.scope != *scope || lease.pod_id != *pod_id);
    }

    fn revoke_for_node(&mut self, node_id: &NodeId) {
        self.leases.retain(|_, lease| lease.node_id != *node_id);
    }
}

/// Refuse a device list that names one device twice.
fn ensure_unique_devices(devices: &[DeviceRecord]) -> Result<(), PlacementError> {
    let mut seen = std::collections::BTreeSet::new();
    for device in devices {
        if !seen.insert(&device.device_id) {
            return Err(PlacementError::DuplicateDevice(device.device_id.clone()));
        }
    }
    Ok(())
}
