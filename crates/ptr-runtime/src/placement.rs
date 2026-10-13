use ptr_types::{ArtifactId, DeviceId, Generation, NodeId, PodId, StateId, Timestamp};
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

#[derive(Debug, Eq, PartialEq)]
pub struct FencedStateLease {
    pub(crate) pod_id: PodId,
    pub(crate) state_id: StateId,
    pub(crate) node_id: NodeId,
    pub(crate) device_id: DeviceId,
    pub(crate) placement_epoch: PlacementEpoch,
    pub(crate) fencing_token: FencingToken,
    pub(crate) generation: Generation,
}

impl FencedStateLease {
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
}

pub struct PodPlacementController {
    nodes: BTreeMap<NodeId, NodeRecord>,
    placements: BTreeMap<PodId, Placement>,
    /// Placements revoked by a topology event, kept only so that `issue_lease` can still name
    /// the node or device that failed. They are never served by `placement` and never revived.
    /// At most one entry per pod, so the map is bounded by the same count as `placements`;
    /// `assign` and the first `issue_lease` that finds the failure gone prune it.
    revoked: BTreeMap<PodId, Placement>,
    leases: BTreeMap<StateId, FencedStateLease>,
    frozen: BTreeMap<StateId, PlacementEpoch>,
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

    pub fn register_node(&mut self, node: NodeRecord) -> Result<(), PlacementError> {
        let node_id = node.node_id.clone();
        self.nodes.insert(node_id.clone(), node);
        // Replacing a node record is a topology change: a lease on a device the
        // new record no longer offers, or offers as failed, must not outlive it.
        // The same holds for a placement: it is removed, and the controller epoch moves once
        // for the event (the bump below).
        self.revoke_unavailable_placements(&node_id);
        self.revoke_unavailable_leases(&node_id);
        self.bump_epoch()?;
        Ok(())
    }

    pub fn heartbeat(
        &mut self,
        node_id: &NodeId,
        timestamp: Timestamp,
        devices: Vec<DeviceRecord>,
    ) -> Result<(), PlacementError> {
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

    pub fn assign(
        &mut self,
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
        self.revoke_for_pod(&pod_id);
        // Hygiene only (issue_lease reads `placements` first): keeps the tombstone bound tight.
        self.revoked.remove(&pod_id);
        let placement = Placement {
            pod_id: pod_id.clone(),
            artifact_id,
            generation,
            node_id,
            device_id,
            epoch: self.epoch,
        };
        self.placements.insert(pod_id, placement.clone());
        Ok(placement)
    }

    pub fn placement(&self, pod_id: &PodId) -> Result<&Placement, PlacementError> {
        self.placements
            .get(pod_id)
            .ok_or(PlacementError::NoPlacement)
    }

    pub fn issue_lease(
        &mut self,
        state_id: StateId,
        pod_id: &PodId,
    ) -> Result<FencedStateLease, PlacementError> {
        let Some(placement) = self.placements.get(pod_id).cloned() else {
            // A revoked placement still names the failure that ended it; a revoked placement
            // whose node and device are usable again is gone, not resurrected.
            let Some(old) = self.revoked.get(pod_id) else {
                return Err(PlacementError::NoPlacement);
            };
            return match self.ensure_usable(&old.node_id, &old.device_id) {
                Err(error) => Err(error),
                Ok(()) => {
                    // The failure is gone: drop the tombstone so the map cleans itself and a
                    // later failure of the same device is not blamed on this old placement.
                    self.revoked.remove(pod_id);
                    Err(PlacementError::NoPlacement)
                }
            };
        };
        // The node and device the placement names are checked again before a lease is
        // issued.
        self.ensure_usable(&placement.node_id, &placement.device_id)?;
        if let Some(frozen_epoch) = self.frozen.get(&state_id).copied() {
            if frozen_epoch == placement.epoch {
                return Err(PlacementError::StateFrozen);
            }
            self.frozen.remove(&state_id);
        }
        let token = FencingToken(self.next_token);
        self.next_token = self
            .next_token
            .checked_add(1)
            .ok_or(PlacementError::CounterExhausted)?;
        let lease = FencedStateLease {
            pod_id: placement.pod_id.clone(),
            state_id: state_id.clone(),
            node_id: placement.node_id,
            device_id: placement.device_id,
            placement_epoch: placement.epoch,
            fencing_token: token,
            generation: placement.generation,
        };
        self.leases.insert(
            state_id,
            FencedStateLease {
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
            .get(&lease.state_id)
            .ok_or(PlacementError::StaleLease)?;
        if current != lease {
            return Err(PlacementError::StaleLease);
        }
        Ok(())
    }

    pub fn revoke_lease(&mut self, state_id: &StateId) -> Result<(), PlacementError> {
        self.leases
            .remove(state_id)
            .map(|_| ())
            .ok_or(PlacementError::StaleLease)
    }

    pub fn freeze_state(&mut self, lease: &FencedStateLease) -> Result<(), PlacementError> {
        self.validate_lease(lease)?;
        self.revoke_lease(lease.state_id())?;
        self.frozen
            .insert(lease.state_id().clone(), lease.placement_epoch());
        Ok(())
    }

    pub fn unfreeze_state(&mut self, state_id: &StateId) {
        self.frozen.remove(state_id);
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
        let gone: Vec<PodId> = self
            .placements
            .iter()
            .filter(|(_, p)| p.node_id == *node_id && !usable.contains(&p.device_id))
            .map(|(pod_id, _)| pod_id.clone())
            .collect();
        for pod_id in &gone {
            if let Some(placement) = self.placements.remove(pod_id) {
                self.revoked.insert(pod_id.clone(), placement);
            }
        }
        !gone.is_empty()
    }

    fn revoke_for_pod(&mut self, pod_id: &PodId) {
        self.leases.retain(|_, lease| lease.pod_id != *pod_id);
    }

    fn revoke_for_node(&mut self, node_id: &NodeId) {
        self.leases.retain(|_, lease| lease.node_id != *node_id);
    }
}
