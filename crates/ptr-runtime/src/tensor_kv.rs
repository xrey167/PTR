use ptr_pods::{
    KvBackendError, KvPageBinding, KvPageMetrics, KvPageSize, KvPagedSnapshot,
    KvRuntimeLeaseBinding, KvSharedPrefix, KvTensorBackend, KvTensorSchema, KvTensorSnapshot,
    KvTierAdapter, KvTierBinding, PagedKvTensorBackend, PagedKvTierAdapter, PodTierError,
    TensorRef,
};
use ptr_storage::PreparedTierObject;
use ptr_types::{DeviceId, Digest, Generation, PodId, PrincipalId, StateId};
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
    DeviceMismatch,
    TierBindingMismatch,
    StateAlreadyExists,
    Backend(KvBackendError),
    Tier(PodTierError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagedKvPageContext {
    pub execution_manifest: Digest,
    pub principal: PrincipalId,
    pub page_tokens: KvPageSize,
}

pub struct ManagedOnlineKvSnapshot<L> {
    owner: u64,
    state_id: StateId,
    generation: Generation,
    placement_epoch: crate::PlacementEpoch,
    fencing_token: crate::FencingToken,
    sequence_length: usize,
    lease: L,
}

impl<L> ManagedOnlineKvSnapshot<L> {
    pub fn state_id(&self) -> &StateId {
        &self.state_id
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn placement_epoch(&self) -> crate::PlacementEpoch {
        self.placement_epoch
    }

    pub fn fencing_token(&self) -> crate::FencingToken {
        self.fencing_token
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
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
        let lease = placement
            .issue_lease(state_id.clone(), pod_id)
            .map_err(|_| TensorKvError::StaleLease)?;
        if snapshot.schema.device != *lease.device_id() {
            let _ = placement.revoke_lease(lease.state_id());
            return Err(TensorKvError::DeviceMismatch);
        }
        let cache = match self.backend.restore(snapshot.clone(), lease.device_id()) {
            Ok(cache) => cache,
            Err(error) => {
                let _ = placement.revoke_lease(lease.state_id());
                return Err(TensorKvError::Backend(error));
            }
        };
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

    /// Quiescent per-state snapshot into the generic tier transport format.
    /// The immutable borrow prevents append through this registry while the
    /// snapshot is captured; other registries and states remain unaffected.
    pub fn snapshot_to_tier(
        &self,
        placement: &crate::PodPlacementController,
        handle: &ManagedKvHandle,
        binding: &KvTierBinding,
        chunk_size: usize,
    ) -> Result<PreparedTierObject, TensorKvError> {
        let metadata = self.metadata(handle)?;
        if binding.logical_id.as_str() != metadata.state_id.0.as_str()
            || binding.generation != metadata.generation
            || binding.placement_epoch != metadata.placement_epoch.0
            || binding.fencing_token != metadata.fencing_token.0
        {
            return Err(TensorKvError::TierBindingMismatch);
        }
        let snapshot = self.snapshot(placement, handle)?;
        KvTierAdapter::prepare(&snapshot, binding, chunk_size).map_err(TensorKvError::Tier)
    }

    pub fn restore_from_tier(
        &mut self,
        placement: &mut crate::PodPlacementController,
        tiers: &crate::TierResidencyController,
        authority: &crate::TierJournalProjection,
        state_id: StateId,
        pod_id: &PodId,
        admitted: &crate::AdmittedTierObject,
        binding: &KvTierBinding,
    ) -> Result<ManagedKvHandle, TensorKvError> {
        tiers
            .validate_admitted_object(admitted)
            .map_err(|_| TensorKvError::TierBindingMismatch)?;
        let object = admitted.object();
        if authority
            .backend(admitted.backend())
            .is_none_or(|backend| backend.state != crate::BackendLifecycleState::Available)
            || authority.object(&object.manifest.root_digest) != Some(&object.manifest)
            || authority
                .replica(&object.manifest.root_digest, admitted.backend())
                .is_none_or(|replica| {
                    replica.state != crate::ReplicaState::Available
                        || replica.generation != object.manifest.generation
                })
        {
            return Err(TensorKvError::TierBindingMismatch);
        }
        let current = placement
            .placement(pod_id)
            .map_err(|_| TensorKvError::StaleLease)?;
        if current.generation != binding.generation || current.epoch.0 < binding.placement_epoch {
            return Err(TensorKvError::TierBindingMismatch);
        }
        let snapshot = KvTierAdapter::restore(object, binding).map_err(TensorKvError::Tier)?;
        self.restore(placement, state_id, pod_id, snapshot)
    }

    pub fn invalidate(
        &mut self,
        placement: &mut crate::PodPlacementController,
        handle: &ManagedKvHandle,
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
        placement
            .revoke_lease(record.lease.state_id())
            .map_err(|_| TensorKvError::StaleLease)?;
        record.metadata.valid = false;
        backend
            .revoke(&mut record.cache)
            .map_err(TensorKvError::Backend)
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
        placement
            .revoke_lease(record.lease.state_id())
            .map_err(|_| TensorKvError::StaleLease)?;
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
}

impl<B: PagedKvTensorBackend> ManagedKvRegistry<B> {
    pub fn allocate_paged(
        &mut self,
        placement: &mut crate::PodPlacementController,
        state_id: StateId,
        pod_id: &PodId,
        schema: KvTensorSchema,
        capacity_tokens: usize,
        context: ManagedKvPageContext,
    ) -> Result<ManagedKvHandle, TensorKvError> {
        if self.records.contains_key(&state_id) {
            return Err(TensorKvError::StateAlreadyExists);
        }
        let lease = placement
            .issue_lease(state_id.clone(), pod_id)
            .map_err(|_| TensorKvError::StaleLease)?;
        if schema.device != *lease.device_id() || context.execution_manifest == [0; 32] {
            let _ = placement.revoke_lease(lease.state_id());
            return Err(TensorKvError::DeviceMismatch);
        }
        let binding = KvPageBinding {
            model: schema.model.clone(),
            adapter: schema.adapter.clone(),
            generation: lease.generation(),
            execution_manifest: context.execution_manifest,
            principal: context.principal,
            device: schema.device.clone(),
            dtype: schema.dtype,
            page_tokens: context.page_tokens,
        };
        let mut cache = match self
            .backend
            .allocate_bound(schema.clone(), capacity_tokens, binding)
        {
            Ok(cache) => cache,
            Err(error) => {
                let _ = placement.revoke_lease(lease.state_id());
                return Err(TensorKvError::Backend(error));
            }
        };
        let runtime_binding = KvRuntimeLeaseBinding {
            placement_epoch: lease.placement_epoch().0,
            fencing_token: lease.fencing_token().0,
        };
        if let Err(error) = self.backend.bind_runtime_lease(&mut cache, runtime_binding) {
            let _ = placement.revoke_lease(lease.state_id());
            let _ = self.backend.release(cache);
            return Err(TensorKvError::Backend(error));
        }
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

    pub fn seal_online_snapshot(
        &mut self,
        placement: &crate::PodPlacementController,
        handle: &ManagedKvHandle,
    ) -> Result<ManagedOnlineKvSnapshot<B::SnapshotLease>, TensorKvError> {
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
        let snapshot = backend
            .seal_snapshot(&mut record.cache)
            .map_err(TensorKvError::Backend)?;
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        Ok(ManagedOnlineKvSnapshot {
            owner: self.owner,
            state_id: record.metadata.state_id.clone(),
            generation: record.metadata.generation,
            placement_epoch: record.metadata.placement_epoch,
            fencing_token: record.metadata.fencing_token,
            sequence_length: record.metadata.sequence_length,
            lease: snapshot,
        })
    }

    pub fn restore_paged_snapshot(
        &mut self,
        placement: &mut crate::PodPlacementController,
        state_id: StateId,
        pod_id: &PodId,
        snapshot: KvPagedSnapshot,
    ) -> Result<ManagedKvHandle, TensorKvError> {
        if self.records.contains_key(&state_id) {
            return Err(TensorKvError::StateAlreadyExists);
        }
        let lease = placement
            .issue_lease(state_id.clone(), pod_id)
            .map_err(|_| TensorKvError::StaleLease)?;
        if snapshot.binding.generation != lease.generation()
            || snapshot.schema.model != snapshot.binding.model
            || snapshot.schema.adapter != snapshot.binding.adapter
        {
            let _ = placement.revoke_lease(lease.state_id());
            return Err(TensorKvError::TierBindingMismatch);
        }
        let mut target_binding = snapshot.binding.clone();
        target_binding.device = lease.device_id().clone();
        let mut cache = match self.backend.restore_paged(snapshot.clone(), target_binding) {
            Ok(cache) => cache,
            Err(error) => {
                let _ = placement.revoke_lease(lease.state_id());
                return Err(TensorKvError::Backend(error));
            }
        };
        let runtime_binding = KvRuntimeLeaseBinding {
            placement_epoch: lease.placement_epoch().0,
            fencing_token: lease.fencing_token().0,
        };
        if let Err(error) = self.backend.bind_runtime_lease(&mut cache, runtime_binding) {
            let _ = placement.revoke_lease(lease.state_id());
            let _ = self.backend.release(cache);
            return Err(TensorKvError::Backend(error));
        }
        let metadata = ManagedKvMetadata {
            state_id: state_id.clone(),
            parent_state_id: None,
            model: snapshot.schema.model,
            adapter: snapshot.schema.adapter,
            device: lease.device_id().clone(),
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

    pub fn restore_paged_from_tier(
        &mut self,
        placement: &mut crate::PodPlacementController,
        tiers: &crate::TierResidencyController,
        authority: &crate::TierJournalProjection,
        state_id: StateId,
        pod_id: &PodId,
        admitted: &crate::AdmittedTierObject,
        binding: &KvTierBinding,
    ) -> Result<ManagedKvHandle, TensorKvError> {
        tiers
            .validate_admitted_object(admitted)
            .map_err(|_| TensorKvError::TierBindingMismatch)?;
        let object = admitted.object();
        if authority
            .backend(admitted.backend())
            .is_none_or(|backend| backend.state != crate::BackendLifecycleState::Available)
            || authority.object(&object.manifest.root_digest) != Some(&object.manifest)
            || authority
                .replica(&object.manifest.root_digest, admitted.backend())
                .is_none_or(|replica| {
                    replica.state != crate::ReplicaState::Available
                        || replica.generation != object.manifest.generation
                })
        {
            return Err(TensorKvError::TierBindingMismatch);
        }
        let current = placement
            .placement(pod_id)
            .map_err(|_| TensorKvError::StaleLease)?;
        if current.generation != binding.generation || current.epoch.0 < binding.placement_epoch {
            return Err(TensorKvError::TierBindingMismatch);
        }
        let snapshot = PagedKvTierAdapter::restore(object, binding).map_err(TensorKvError::Tier)?;
        self.restore_paged_snapshot(placement, state_id, pod_id, snapshot)
    }

    pub fn materialize_online_snapshot(
        &self,
        snapshot: &ManagedOnlineKvSnapshot<B::SnapshotLease>,
    ) -> Result<KvPagedSnapshot, TensorKvError> {
        if snapshot.owner != self.owner {
            return Err(TensorKvError::ForeignHandle);
        }
        self.backend
            .snapshot_from_lease(&snapshot.lease)
            .map_err(TensorKvError::Backend)
    }

    pub fn prepare_online_snapshot_to_tier(
        &self,
        placement: &crate::PodPlacementController,
        snapshot: &ManagedOnlineKvSnapshot<B::SnapshotLease>,
        binding: &KvTierBinding,
    ) -> Result<PreparedTierObject, TensorKvError> {
        if snapshot.owner != self.owner {
            return Err(TensorKvError::ForeignHandle);
        }
        let record = self
            .records
            .get(&snapshot.state_id)
            .ok_or(TensorKvError::UnknownHandle)?;
        if !record.metadata.valid
            || record.metadata.generation != snapshot.generation
            || record.metadata.placement_epoch != snapshot.placement_epoch
            || record.metadata.fencing_token != snapshot.fencing_token
            || binding.logical_id != snapshot.state_id.0
            || binding.generation != snapshot.generation
            || binding.placement_epoch != snapshot.placement_epoch.0
            || binding.fencing_token != snapshot.fencing_token.0
        {
            return Err(TensorKvError::TierBindingMismatch);
        }
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        let materialized = self
            .backend
            .snapshot_from_lease(&snapshot.lease)
            .map_err(TensorKvError::Backend)?;
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        PagedKvTierAdapter::prepare(&materialized, binding).map_err(TensorKvError::Tier)
    }

    pub fn publish_prefix(
        &mut self,
        placement: &crate::PodPlacementController,
        handle: &ManagedKvHandle,
    ) -> Result<KvSharedPrefix, TensorKvError> {
        if handle.owner != self.owner {
            return Err(TensorKvError::ForeignHandle);
        }
        let (backend, records) = (&self.backend, &mut self.records);
        let record = records
            .get_mut(&handle.state_id)
            .ok_or(TensorKvError::UnknownHandle)?;
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        let prefix = backend
            .publish_prefix(&mut record.cache)
            .map_err(TensorKvError::Backend)?;
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        Ok(prefix)
    }

    pub fn fork_prefix(
        &mut self,
        placement: &crate::PodPlacementController,
        handle: &ManagedKvHandle,
        prefix: &KvSharedPrefix,
    ) -> Result<(), TensorKvError> {
        if handle.owner != self.owner {
            return Err(TensorKvError::ForeignHandle);
        }
        let (backend, records) = (&self.backend, &mut self.records);
        let record = records
            .get_mut(&handle.state_id)
            .ok_or(TensorKvError::UnknownHandle)?;
        placement
            .validate_lease(&record.lease)
            .map_err(|_| TensorKvError::StaleLease)?;
        backend
            .fork_prefix(&mut record.cache, prefix)
            .map_err(TensorKvError::Backend)?;
        record.metadata.sequence_length = prefix.sequence_length;
        record.metadata.position_offset = prefix.position_offset;
        Ok(())
    }

    pub fn page_metrics(&self, handle: &ManagedKvHandle) -> Result<KvPageMetrics, TensorKvError> {
        let record = self.resolve(handle)?;
        Ok(self.backend.page_metrics(&record.cache))
    }
}
