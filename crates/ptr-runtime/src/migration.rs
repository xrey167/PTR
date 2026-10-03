use crate::placement::{FencedStateLease, FencingToken, PlacementError, PodPlacementController};
use ptr_pods::{KvBackendError, KvTensorBackend, KvTensorSnapshot};
use ptr_types::{DeviceId, Generation, NodeId, PodId, StateId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationState {
    Frozen,
    Snapshotted,
    Restored,
    Committed,
    Aborted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MigrationRecord {
    pub migration_id: u64,
    pub state_id: StateId,
    pub pod_id: PodId,
    pub generation: Generation,
    pub source_node: NodeId,
    pub source_device: DeviceId,
    pub source_token: FencingToken,
    pub snapshot_digest: Option<[u8; 32]>,
    pub target_node: Option<NodeId>,
    pub target_device: Option<DeviceId>,
    pub state: MigrationState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MigrationError {
    Placement(PlacementError),
    UnknownMigration(u64),
    InvalidState(MigrationState),
    SnapshotDigestMismatch,
    RestoreFailed(KvBackendError),
    GenerationMismatch,
    SamePlacement,
}

#[derive(Default)]
pub struct KvMigrationController {
    next_id: u64,
    records: BTreeMap<u64, MigrationRecord>,
}

impl KvMigrationController {
    pub fn begin(
        &mut self,
        placement: &mut PodPlacementController,
        source: FencedStateLease,
    ) -> Result<MigrationRecord, MigrationError> {
        placement
            .validate_lease(&source)
            .map_err(MigrationError::Placement)?;
        placement
            .freeze_state(&source)
            .map_err(MigrationError::Placement)?;
        let migration_id = self
            .next_id
            .checked_add(1)
            .ok_or(MigrationError::UnknownMigration(u64::MAX))?;
        self.next_id = migration_id;
        let record = MigrationRecord {
            migration_id,
            state_id: source.state_id().clone(),
            pod_id: source.pod_id().clone(),
            generation: source.generation(),
            source_node: source.node_id().clone(),
            source_device: source.device_id().clone(),
            source_token: source.fencing_token(),
            snapshot_digest: None,
            target_node: None,
            target_device: None,
            state: MigrationState::Frozen,
        };
        self.records.insert(migration_id, record.clone());
        Ok(record)
    }

    pub fn snapshot(
        &mut self,
        migration_id: u64,
        snapshot: &KvTensorSnapshot,
    ) -> Result<MigrationRecord, MigrationError> {
        let record = self.record_mut(migration_id)?;
        if record.state != MigrationState::Frozen {
            return Err(MigrationError::InvalidState(record.state));
        }
        if !snapshot.verify_digest() {
            return Err(MigrationError::SnapshotDigestMismatch);
        }
        record.snapshot_digest = Some(snapshot.digest);
        record.state = MigrationState::Snapshotted;
        Ok(record.clone())
    }

    pub fn restore<B: KvTensorBackend>(
        &mut self,
        placement: &mut PodPlacementController,
        migration_id: u64,
        target: &FencedStateLease,
        backend: &B,
        snapshot: &KvTensorSnapshot,
    ) -> Result<(MigrationRecord, B::Cache), MigrationError> {
        placement
            .validate_lease(target)
            .map_err(MigrationError::Placement)?;
        let record = self.record_mut(migration_id)?;
        if record.state != MigrationState::Snapshotted {
            return Err(MigrationError::InvalidState(record.state));
        }
        if record.snapshot_digest != Some(snapshot.digest) || !snapshot.verify_digest() {
            return Err(MigrationError::SnapshotDigestMismatch);
        }
        if target.state_id() != &record.state_id || target.generation() != record.generation {
            return Err(MigrationError::GenerationMismatch);
        }
        if target.node_id() == &record.source_node && target.device_id() == &record.source_device {
            return Err(MigrationError::SamePlacement);
        }
        let target_snapshot = snapshot.rebind_device(target.device_id().clone());
        let cache = backend
            .restore(target_snapshot, target.device_id())
            .map_err(MigrationError::RestoreFailed)?;
        record.target_node = Some(target.node_id().clone());
        record.target_device = Some(target.device_id().clone());
        record.state = MigrationState::Restored;
        Ok((record.clone(), cache))
    }

    pub fn commit(
        &mut self,
        placement: &mut PodPlacementController,
        migration_id: u64,
        target: &FencedStateLease,
    ) -> Result<MigrationRecord, MigrationError> {
        placement
            .validate_lease(target)
            .map_err(MigrationError::Placement)?;
        let record = self.record_mut(migration_id)?;
        if record.state != MigrationState::Restored {
            return Err(MigrationError::InvalidState(record.state));
        }
        if target.node_id() != record.target_node.as_ref().unwrap()
            || target.device_id() != record.target_device.as_ref().unwrap()
        {
            return Err(MigrationError::Placement(PlacementError::StaleLease));
        }
        record.state = MigrationState::Committed;
        placement.unfreeze_state(&record.state_id);
        Ok(record.clone())
    }

    pub fn abort(
        &mut self,
        placement: &mut PodPlacementController,
        migration_id: u64,
    ) -> Result<MigrationRecord, MigrationError> {
        let record = self.record_mut(migration_id)?;
        if matches!(
            record.state,
            MigrationState::Committed | MigrationState::Aborted
        ) {
            return Err(MigrationError::InvalidState(record.state));
        }
        record.state = MigrationState::Aborted;
        placement.unfreeze_state(&record.state_id);
        Ok(record.clone())
    }

    fn record_mut(&mut self, migration_id: u64) -> Result<&mut MigrationRecord, MigrationError> {
        self.records
            .get_mut(&migration_id)
            .ok_or(MigrationError::UnknownMigration(migration_id))
    }
}
