use ptr_pods::{KvBackendError, KvTensorBackend, KvTensorSchema, KvTensorSnapshot, TensorRef};
use ptr_types::{DeviceId, Generation, PodId, StateId};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ManagedKvHandle {
    state_id: StateId,
    owner: u64,
}

impl ManagedKvHandle {
    pub fn state_id(&self) -> &StateId {
        &self.state_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagedKvMetadata {
    pub state_id: StateId,
    pub parent_state_id: Option<StateId>,
    pub model: ptr_types::ModelVersion,
    pub adapter: ptr_types::AdapterVersion,
    pub device: DeviceId,
    pub generation: Generation,
    pub sequence_length: usize,
    pub position_offset: usize,
    pub placement_epoch: crate::PlacementEpoch,
    pub fencing_token: crate::FencingToken,
    pub valid: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorKvError {
    UnknownHandle,
    ForeignHandle,
    Invalidated,
    StaleLease,
    StateAlreadyExists,
    Backend(KvBackendError),
}

struct Record<C> {
    metadata: ManagedKvMetadata,
    lease: crate::FencedStateLease,
    cache: C,
}

pub struct ManagedKvRegistry<B: KvTensorBackend> {
    backend: B,
    owner: u64,
    records: BTreeMap<StateId, Record<B::Cache>>,
}

impl<B: KvTensorBackend> ManagedKvRegistry<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            records: BTreeMap::new(),
        }
    }

    pub fn allocate(
        &mut self,
        placement: &mut crate::PodPlacementController,
        state_id: StateId,
        pod_id: &PodId,
        schema: KvTensorSchema,
        capacity_tokens: usize,
    ) -> Result<ManagedKvHandle, TensorKvError> {
        if self.records.contains_key(&state_id) {
            return Err(TensorKvError::StateAlreadyExists);
        }
        let cache = self
            .backend
            .allocate(schema.clone(), capacity_tokens)
            .map_err(TensorKvError::Backend)?;
        let lease = placement
            .issue_lease(state_id.clone(), pod_id)
            .map_err(|_| TensorKvError::StaleLease)?;
        let metadata = ManagedKvMetadata {
            state_id: state_id.clone(),
            parent_state_id: None,
            model: schema.model,
            adapter: schema.adapter,
            device: schema.device,
            generation: lease.generation(),
            sequence_length: 0,
            position_offset: 0,
            placement_epoch: lease.placement_epoch(),
            fencing_token: lease.fencing_token(),
            valid: true,
        };
        self.records.insert(
            state_id.clone(),
            Record {
                metadata,
                lease,
                cache,
            },
        );
        Ok(ManagedKvHandle {
            state_id,
            owner: self.owner,
        })
    }

    pub fn restore(
        &mut self,
        placement: &mut crate::PodPlacementController,
        state_id: StateId,
        pod_id: &PodId,
        snapshot: KvTensorSnapshot,
    ) -> Result<ManagedKvHandle, TensorKvError> {
        if self.records.contains_key(&state_id) {
            return Err(TensorKvError::StateAlreadyExists);
        }
        let cache = self
            .backend
            .restore(snapshot.clone(), &snapshot.schema.device)
            .map_err(TensorKvError::Backend)?;
        let lease = placement
            .issue_lease(state_id.clone(), pod_id)
            .map_err(|_| TensorKvError::StaleLease)?;
        let metadata = ManagedKvMetadata {
            state_id: state_id.clone(),
            parent_state_id: None,
            model: snapshot.schema.model,
            adapter: snapshot.schema.adapter,
            device: snapshot.schema.device,
            generation: lease.generation(),
            sequence_length: snapshot.sequence_length,
            position_offset: snapshot.position_offset,
            placement_epoch: lease.placement_epoch(),
            fencing_token: lease.fencing_token(),
            valid: true,
        };
        self.records.insert(
            state_id.clone(),
            Record {
                metadata,
                lease,
                cache,
            },
        );
        Ok(ManagedKvHandle {
            state_id,
            owner: self.owner,
        })
    }

    /// Allocate a fresh root cache after invalidation or migration. A
    /// recompute never reuses the invalidated handle or its fencing token.
    pub fn recompute(
        &mut self,
        placement: &mut crate::PodPlacementController,
        state_id: StateId,
        pod_id: &PodId,
        schema: KvTensorSchema,
        capacity_tokens: usize,
    ) -> Result<ManagedKvHandle, TensorKvError> {
        self.allocate(placement, state_id, pod_id, schema, capacity_tokens)
    }

    pub fn append(
        &mut self,
        placement: &crate::PodPlacementController,
        handle: &ManagedKvHandle,
        keys: &[TensorRef],
        values: &[TensorRef],
    ) -> Result<(), TensorKvError> {
        if handle.owner != self.owner {
            return Err(TensorKvError::ForeignHandle);
        }
        let (backend, records) = (&self.backend, &mut self.records);
        let record = records
            .get_mut(&handle.state_id)
            .ok_or(TensorKvError::UnknownHandle)?;
        if !record.metadata.valid {
            return Err(TensorKvError::Invalidated);
        }
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        backend
            .append(&mut record.cache, keys, values)
            .map_err(TensorKvError::Backend)?;
        let added = keys.first().map(|tensor| tensor.shape[0]).unwrap_or(0);
        record.metadata.sequence_length += added;
        Ok(())
    }

    pub fn snapshot(
        &self,
        placement: &crate::PodPlacementController,
        handle: &ManagedKvHandle,
    ) -> Result<KvTensorSnapshot, TensorKvError> {
        let record = self.resolve(handle)?;
        if !record.metadata.valid {
            return Err(TensorKvError::Invalidated);
        }
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        self.backend
            .snapshot(&record.cache)
            .map_err(TensorKvError::Backend)
    }

    pub fn invalidate(
        &mut self,
        placement: &mut crate::PodPlacementController,
        handle: &ManagedKvHandle,
    ) -> Result<(), TensorKvError> {
        let record = self.resolve_mut(handle)?;
        if !record.metadata.valid {
            return Err(TensorKvError::Invalidated);
        }
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        placement
            .revoke_lease(record.lease.state_id())
            .map_err(|_| TensorKvError::StaleLease)?;
        record.metadata.valid = false;
        Ok(())
    }

    pub fn metadata(&self, handle: &ManagedKvHandle) -> Result<ManagedKvMetadata, TensorKvError> {
        Ok(self.resolve(handle)?.metadata.clone())
    }

    pub fn release(
        &mut self,
        placement: &mut crate::PodPlacementController,
        handle: &ManagedKvHandle,
    ) -> Result<(), TensorKvError> {
        let record = self.resolve(handle)?;
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        let record = self.records.remove(&handle.state_id).unwrap();
        self.backend
            .release(record.cache)
            .map_err(TensorKvError::Backend)
    }

    fn resolve(&self, handle: &ManagedKvHandle) -> Result<&Record<B::Cache>, TensorKvError> {
        if handle.owner != self.owner {
            return Err(TensorKvError::ForeignHandle);
        }
        self.records
            .get(&handle.state_id)
            .ok_or(TensorKvError::UnknownHandle)
    }

    fn resolve_mut(
        &mut self,
        handle: &ManagedKvHandle,
    ) -> Result<&mut Record<B::Cache>, TensorKvError> {
        if handle.owner != self.owner {
            return Err(TensorKvError::ForeignHandle);
        }
        self.records
            .get_mut(&handle.state_id)
            .ok_or(TensorKvError::UnknownHandle)
    }
}
