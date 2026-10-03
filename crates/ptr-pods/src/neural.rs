use ptr_protocol::TypedPayload;
use ptr_types::{ArtifactId, CapabilityId, DeviceId, Digest, Generation, PodId, TypeId};
use sha2::{Digest as ShaDigest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeuralPodType {
    Model,
    Retrieval,
    Math,
    Router,
    Runtime,
    Vision,
    Embedding,
    Quantization,
    Transport,
    Policy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PodLifecycle {
    Unregistered,
    Admitted,
    Prepared,
    Ready,
    Busy,
    Released,
    Revoked,
    Failed,
}

impl PodLifecycle {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Released | Self::Revoked | Self::Failed)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceRequirements {
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub device: Option<DeviceId>,
    pub max_concurrency: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceLeaseState {
    Active,
    Released,
    Revoked,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceLease {
    pub device_id: DeviceId,
    pub stream_id: u64,
    pub placement_epoch: u64,
    pub fencing_token: u128,
    pub active_tensors: usize,
    pub state: DeviceLeaseState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceLeaseError {
    Busy,
    Released,
    Revoked,
    Failed,
    InvalidTensorRelease,
}

impl DeviceLease {
    pub fn new(
        device_id: DeviceId,
        stream_id: u64,
        placement_epoch: u64,
        fencing_token: u128,
    ) -> Self {
        Self {
            device_id,
            stream_id,
            placement_epoch,
            fencing_token,
            active_tensors: 0,
            state: DeviceLeaseState::Active,
        }
    }

    pub fn begin_tensor(&mut self) -> Result<(), DeviceLeaseError> {
        match self.state {
            DeviceLeaseState::Active => {
                self.active_tensors = self.active_tensors.saturating_add(1);
                Ok(())
            }
            DeviceLeaseState::Released => Err(DeviceLeaseError::Released),
            DeviceLeaseState::Revoked => Err(DeviceLeaseError::Revoked),
            DeviceLeaseState::Failed => Err(DeviceLeaseError::Failed),
        }
    }

    pub fn end_tensor(&mut self) -> Result<(), DeviceLeaseError> {
        if self.active_tensors == 0 {
            return Err(DeviceLeaseError::InvalidTensorRelease);
        }
        self.active_tensors -= 1;
        Ok(())
    }

    pub fn revoke(&mut self) {
        if self.state == DeviceLeaseState::Active {
            self.state = DeviceLeaseState::Revoked;
        }
    }

    pub fn abort(&mut self) {
        if self.state == DeviceLeaseState::Active {
            self.state = DeviceLeaseState::Failed;
        }
    }

    pub fn release(&mut self) -> Result<(), DeviceLeaseError> {
        match self.state {
            DeviceLeaseState::Released => Ok(()),
            DeviceLeaseState::Active if self.active_tensors == 0 => {
                self.state = DeviceLeaseState::Released;
                Ok(())
            }
            DeviceLeaseState::Active => Err(DeviceLeaseError::Busy),
            DeviceLeaseState::Revoked => Err(DeviceLeaseError::Revoked),
            DeviceLeaseState::Failed => Err(DeviceLeaseError::Failed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDType {
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorContract {
    pub dtype: TensorDType,
    pub input_len: usize,
    pub output_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeuralPodDescriptor {
    pub artifact_id: ArtifactId,
    pub pod_id: PodId,
    pub manifest_hash: [u8; 32],
    pub generation: Generation,
    pub pod_type: NeuralPodType,
    pub input_schema: TypeId,
    pub output_schema: TypeId,
    pub capabilities: Vec<CapabilityId>,
    pub provenance: Vec<String>,
    pub lifecycle: PodLifecycle,
    pub resources: ResourceRequirements,
    pub tensor: TensorContract,
}

impl NeuralPodDescriptor {
    /// Canonical digest for artifact lineage binding. This is distinct from
    /// the invocation manifest hash: an artifact includes its executor and
    /// tensor contract as well as the Pod contract.
    pub fn lineage_digest(&self) -> Digest {
        let mut bytes = Vec::new();
        macro_rules! put {
            ($value:expr) => {{
                let value = $value.to_string();
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(value.as_bytes());
            }};
        }
        put!(self.artifact_id.0);
        put!(self.pod_id.0);
        bytes.extend_from_slice(&self.manifest_hash);
        put!(self.generation.0);
        put!(format!("{:?}", self.pod_type));
        put!(self.input_schema.0);
        put!(self.output_schema.0);
        for capability in &self.capabilities {
            put!(capability.0);
        }
        for provenance in &self.provenance {
            put!(provenance);
        }
        put!(format!("{:?}", self.lifecycle));
        put!(self.resources.ram_bytes);
        put!(self.resources.vram_bytes);
        put!(self.resources.max_concurrency);
        put!(format!("{:?}", self.resources.device));
        put!(format!("{:?}", self.tensor));
        Sha256::digest(bytes).into()
    }

    pub fn validate(&self) -> Result<(), DescriptorError> {
        if self.artifact_id.0.trim().is_empty() || self.pod_id.0.trim().is_empty() {
            return Err(DescriptorError::MissingIdentity);
        }
        if self.manifest_hash == [0; 32] {
            return Err(DescriptorError::MissingManifestHash);
        }
        if self.generation.0 == 0 {
            return Err(DescriptorError::InvalidGeneration);
        }
        if self.input_schema.0.trim().is_empty() || self.output_schema.0.trim().is_empty() {
            return Err(DescriptorError::MissingSchema);
        }
        if self.resources.max_concurrency == 0 || self.resources.ram_bytes == 0 {
            return Err(DescriptorError::InvalidResources);
        }
        if self.tensor.input_len == 0 || self.tensor.output_len == 0 {
            return Err(DescriptorError::InvalidTensorContract);
        }
        if !matches!(
            self.lifecycle,
            PodLifecycle::Admitted | PodLifecycle::Prepared | PodLifecycle::Ready
        ) {
            return Err(DescriptorError::InvalidLifecycle);
        }
        if self.provenance.is_empty() {
            return Err(DescriptorError::MissingProvenance);
        }
        Ok(())
    }

    pub fn admits_output_type(&self, output: &TypeId) -> bool {
        &self.output_schema == output
    }

    pub fn transition_to(&mut self, next: PodLifecycle) -> Result<(), DescriptorError> {
        if self.lifecycle.is_terminal() {
            return if self.lifecycle == next {
                Ok(())
            } else {
                Err(DescriptorError::InvalidTransition {
                    from: self.lifecycle,
                    to: next,
                })
            };
        }
        let allowed = matches!(
            (self.lifecycle, next),
            (PodLifecycle::Unregistered, PodLifecycle::Admitted)
                | (PodLifecycle::Admitted, PodLifecycle::Prepared)
                | (PodLifecycle::Prepared, PodLifecycle::Ready)
                | (PodLifecycle::Ready, PodLifecycle::Busy)
                | (PodLifecycle::Busy, PodLifecycle::Ready)
                | (PodLifecycle::Ready, PodLifecycle::Released)
                | (PodLifecycle::Busy, PodLifecycle::Released)
                | (PodLifecycle::Admitted, PodLifecycle::Revoked)
                | (PodLifecycle::Prepared, PodLifecycle::Revoked)
                | (PodLifecycle::Ready, PodLifecycle::Revoked)
                | (PodLifecycle::Busy, PodLifecycle::Revoked)
                | (PodLifecycle::Admitted, PodLifecycle::Failed)
                | (PodLifecycle::Prepared, PodLifecycle::Failed)
                | (PodLifecycle::Ready, PodLifecycle::Failed)
                | (PodLifecycle::Busy, PodLifecycle::Failed)
        );
        if !allowed {
            return Err(DescriptorError::InvalidTransition {
                from: self.lifecycle,
                to: next,
            });
        }
        self.lifecycle = next;
        Ok(())
    }

    pub fn can_activate(&self) -> bool {
        self.validate().is_ok()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DescriptorError {
    MissingIdentity,
    MissingManifestHash,
    InvalidGeneration,
    MissingSchema,
    InvalidResources,
    InvalidTensorContract,
    MissingProvenance,
    InvalidLifecycle,
    InvalidTransition {
        from: PodLifecycle,
        to: PodLifecycle,
    },
    OutputTypeMismatch {
        expected: TypeId,
        actual: TypeId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseState {
    Ready,
    Busy,
    Released,
    Revoked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeuralPodLease {
    pub pod_id: PodId,
    pub generation: Generation,
    pub state: LeaseState,
}

impl NeuralPodLease {
    pub fn begin(&mut self) -> Result<(), NeuralPodError> {
        if self.state != LeaseState::Ready {
            return Err(NeuralPodError::LeaseUnavailable(self.state));
        }
        self.state = LeaseState::Busy;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), NeuralPodError> {
        if self.state != LeaseState::Busy {
            return Err(NeuralPodError::LeaseUnavailable(self.state));
        }
        self.state = LeaseState::Ready;
        Ok(())
    }

    pub fn revoke(&mut self) {
        self.state = LeaseState::Revoked;
    }

    pub fn release(&mut self) -> Result<(), NeuralPodError> {
        if self.state == LeaseState::Released {
            return Ok(());
        }
        if !matches!(
            self.state,
            LeaseState::Ready | LeaseState::Busy | LeaseState::Revoked
        ) {
            return Err(NeuralPodError::LeaseUnavailable(self.state));
        }
        self.state = LeaseState::Released;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NeuralPodError {
    LeaseUnavailable(LeaseState),
    InvalidInputType,
    InvalidOutputType,
    Descriptor(DescriptorError),
}

pub trait NeuralPodExecutor: Send + Sync {
    type Lease;
    type Error;

    fn activate(&self, descriptor: &NeuralPodDescriptor) -> Result<Self::Lease, Self::Error>;
    fn infer(
        &self,
        lease: &mut Self::Lease,
        input: TypedPayload,
    ) -> Result<TypedPayload, Self::Error>;
    fn release(&self, lease: Self::Lease) -> Result<(), Self::Error>;
    fn abort(&self, lease: Self::Lease) -> Result<(), Self::Error> {
        self.release(lease)
    }
    fn health(&self) -> bool;
}

/// Minimal in-process executor used as a typed contract reference. Production
/// model backends can replace it without changing admission or lease semantics.
#[derive(Clone, Debug)]
pub struct ReferenceNeuralExecutor {
    descriptor: NeuralPodDescriptor,
}

impl ReferenceNeuralExecutor {
    pub fn new(descriptor: NeuralPodDescriptor) -> Result<Self, NeuralPodError> {
        descriptor.validate().map_err(NeuralPodError::Descriptor)?;
        Ok(Self { descriptor })
    }
}

impl NeuralPodExecutor for ReferenceNeuralExecutor {
    type Lease = NeuralPodLease;
    type Error = NeuralPodError;

    fn activate(&self, descriptor: &NeuralPodDescriptor) -> Result<Self::Lease, Self::Error> {
        if descriptor != &self.descriptor {
            return Err(NeuralPodError::Descriptor(
                DescriptorError::InvalidLifecycle,
            ));
        }
        Ok(NeuralPodLease {
            pod_id: descriptor.pod_id.clone(),
            generation: descriptor.generation,
            state: LeaseState::Ready,
        })
    }

    fn infer(
        &self,
        lease: &mut Self::Lease,
        input: TypedPayload,
    ) -> Result<TypedPayload, Self::Error> {
        lease.begin()?;
        if input.type_id != self.descriptor.input_schema {
            let _ = lease.finish();
            return Err(NeuralPodError::InvalidInputType);
        }
        let output = TypedPayload {
            type_id: self.descriptor.output_schema.clone(),
            bytes: input.bytes,
        };
        lease.finish()?;
        Ok(output)
    }

    fn release(&self, mut lease: Self::Lease) -> Result<(), Self::Error> {
        lease.release()
    }

    fn health(&self) -> bool {
        true
    }
}
