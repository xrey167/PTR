use crate::placement::{FencedStateLease, PlacementError, PodPlacementController};
use ptr_memory::{KvStateMetadata, KvValidity};
use ptr_types::{AdapterVersion, Generation, ModelVersion, StateId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvUseRequest {
    pub state_id: StateId,
    pub model: ModelVersion,
    pub adapter: AdapterVersion,
    pub generation: Generation,
    pub context_digest: [u8; 32],
}

#[derive(Debug, Eq, PartialEq)]
pub struct KvUseTicket {
    request: KvUseRequest,
    lease: FencedStateLease,
}

impl KvUseTicket {
    pub fn state_id(&self) -> &StateId {
        &self.request.state_id
    }
    pub fn generation(&self) -> Generation {
        self.request.generation
    }
    pub fn context_digest(&self) -> [u8; 32] {
        self.request.context_digest
    }
    pub fn lease(&self) -> &FencedStateLease {
        &self.lease
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvUseError {
    Placement(PlacementError),
    StateMismatch,
    AlreadyInUse,
    UnknownState,
}

#[derive(Default)]
pub struct RuntimeKvAuthority {
    active: std::collections::BTreeMap<StateId, KvUseTicket>,
}

impl RuntimeKvAuthority {
    pub fn prepare(
        &mut self,
        placement: &PodPlacementController,
        metadata: &KvStateMetadata,
        lease: FencedStateLease,
        request: KvUseRequest,
    ) -> Result<KvUseTicket, KvUseError> {
        placement
            .validate_lease(&lease)
            .map_err(KvUseError::Placement)?;
        if metadata.validity != KvValidity::Valid
            || metadata.state_id != request.state_id
            || metadata.model_version != request.model
            || metadata.adapter_version != request.adapter
            || metadata.context_digest != request.context_digest
            || lease.state_id() != &request.state_id
            || lease.generation() != request.generation
            || (!metadata.pod_generations.is_empty()
                && !metadata.pod_generations.contains(&request.generation))
        {
            return Err(KvUseError::StateMismatch);
        }
        if self.active.contains_key(&request.state_id) {
            return Err(KvUseError::AlreadyInUse);
        }
        let ticket = KvUseTicket { request, lease };
        self.active.insert(
            ticket.request.state_id.clone(),
            KvUseTicket {
                request: ticket.request.clone(),
                lease: FencedStateLease {
                    pod_id: ticket.lease.pod_id().clone(),
                    state_id: ticket.lease.state_id().clone(),
                    node_id: ticket.lease.node_id().clone(),
                    device_id: ticket.lease.device_id().clone(),
                    placement_epoch: ticket.lease.placement_epoch(),
                    fencing_token: ticket.lease.fencing_token(),
                    generation: ticket.lease.generation(),
                },
            },
        );
        Ok(ticket)
    }

    pub fn complete(
        &mut self,
        placement: &PodPlacementController,
        ticket: KvUseTicket,
    ) -> Result<KvUseRequest, KvUseError> {
        let active = self
            .active
            .remove(&ticket.request.state_id)
            .ok_or(KvUseError::UnknownState)?;
        placement
            .validate_lease(&ticket.lease)
            .map_err(KvUseError::Placement)?;
        if active.request != ticket.request || active.lease != ticket.lease {
            return Err(KvUseError::StateMismatch);
        }
        Ok(ticket.request)
    }
}
