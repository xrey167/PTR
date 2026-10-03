use crate::{
    NeuralPodDescriptor, NeuralPodError, NeuralPodExecutor, PodManifest, ReferenceNeuralExecutor,
    ResourceRequirements,
};
use ptr_types::{ArtifactId, Generation};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    NotFound(ArtifactId, Generation),
    InvalidLineage,
    Revoked,
    ConflictingGeneration,
}

pub trait ArtifactCatalog: Send + Sync {
    fn resolve(
        &self,
        artifact: &ArtifactId,
        generation: Generation,
    ) -> Result<NeuralPodDescriptor, ArtifactError>;
    fn verify_lineage(&self, descriptor: &NeuralPodDescriptor) -> Result<(), ArtifactError>;
    fn invalidate(
        &mut self,
        artifact: &ArtifactId,
        generation: Generation,
    ) -> Result<(), ArtifactError>;
}

#[derive(Default)]
pub struct InMemoryArtifactCatalog {
    descriptors: BTreeMap<(ArtifactId, Generation), NeuralPodDescriptor>,
    invalidated: BTreeSet<(ArtifactId, Generation)>,
}

impl InMemoryArtifactCatalog {
    pub fn admit_with_manifest(
        &mut self,
        descriptor: NeuralPodDescriptor,
        manifest: &PodManifest,
    ) -> Result<(), ArtifactError> {
        if descriptor.manifest_hash != manifest.digest()
            || descriptor.pod_id != manifest.id
            || !manifest.accepts.contains(&descriptor.input_schema)
            || !manifest.produces.contains(&descriptor.output_schema)
        {
            return Err(ArtifactError::InvalidLineage);
        }
        self.admit(descriptor)
    }

    pub fn admit(&mut self, descriptor: NeuralPodDescriptor) -> Result<(), ArtifactError> {
        descriptor
            .validate()
            .map_err(|_| ArtifactError::InvalidLineage)?;
        let key = (descriptor.artifact_id.clone(), descriptor.generation);
        if self.invalidated.contains(&key) {
            return Err(ArtifactError::Revoked);
        }
        if let Some(existing) = self.descriptors.get(&key) {
            return if existing == &descriptor {
                Ok(())
            } else {
                Err(ArtifactError::ConflictingGeneration)
            };
        }
        self.descriptors.insert(key, descriptor);
        Ok(())
    }
}

impl ArtifactCatalog for InMemoryArtifactCatalog {
    fn resolve(
        &self,
        artifact: &ArtifactId,
        generation: Generation,
    ) -> Result<NeuralPodDescriptor, ArtifactError> {
        let key = (artifact.clone(), generation);
        if self.invalidated.contains(&key) {
            return Err(ArtifactError::Revoked);
        }
        self.descriptors
            .get(&key)
            .cloned()
            .ok_or_else(|| ArtifactError::NotFound(artifact.clone(), generation))
    }

    fn verify_lineage(&self, descriptor: &NeuralPodDescriptor) -> Result<(), ArtifactError> {
        let resolved = self.resolve(&descriptor.artifact_id, descriptor.generation)?;
        (resolved == *descriptor)
            .then_some(())
            .ok_or(ArtifactError::InvalidLineage)
    }

    fn invalidate(
        &mut self,
        artifact: &ArtifactId,
        generation: Generation,
    ) -> Result<(), ArtifactError> {
        let key = (artifact.clone(), generation);
        if !self.descriptors.contains_key(&key) {
            return Err(ArtifactError::NotFound(artifact.clone(), generation));
        }
        self.invalidated.insert(key);
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceLease {
    pub requirements: ResourceRequirements,
    pub lease_id: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResourceError {
    InsufficientCapacity,
    InvalidRequirements,
    UnknownLease,
    LeaseMismatch,
}

pub trait ResourceGovernor {
    fn reserve(
        &mut self,
        requirements: &ResourceRequirements,
    ) -> Result<ResourceLease, ResourceError>;
    fn release(&mut self, lease: ResourceLease) -> Result<(), ResourceError>;
}

#[derive(Default)]
pub struct InMemoryResourceGovernor {
    ram_capacity: u64,
    vram_capacity: u64,
    next_lease_id: u64,
    concurrency_capacity: u32,
    active: BTreeMap<u64, ResourceLease>,
    released: BTreeMap<u64, ResourceLease>,
}

impl InMemoryResourceGovernor {
    pub fn new(ram_capacity: u64, vram_capacity: u64) -> Self {
        Self {
            ram_capacity,
            vram_capacity,
            next_lease_id: 1,
            concurrency_capacity: 1,
            active: BTreeMap::new(),
            released: BTreeMap::new(),
        }
    }

    pub fn with_concurrency_capacity(mut self, capacity: u32) -> Self {
        self.concurrency_capacity = capacity;
        self
    }
}

impl ResourceGovernor for InMemoryResourceGovernor {
    fn reserve(
        &mut self,
        requirements: &ResourceRequirements,
    ) -> Result<ResourceLease, ResourceError> {
        if requirements.ram_bytes == 0 || requirements.max_concurrency == 0 {
            return Err(ResourceError::InvalidRequirements);
        }
        let used_ram = self
            .active
            .values()
            .map(|lease| lease.requirements.ram_bytes)
            .sum::<u64>();
        let used_vram = self
            .active
            .values()
            .map(|lease| lease.requirements.vram_bytes)
            .sum::<u64>();
        let used_concurrency = self
            .active
            .values()
            .map(|lease| lease.requirements.max_concurrency)
            .sum::<u32>();
        if used_ram.saturating_add(requirements.ram_bytes) > self.ram_capacity
            || used_vram.saturating_add(requirements.vram_bytes) > self.vram_capacity
            || used_concurrency.saturating_add(requirements.max_concurrency)
                > self.concurrency_capacity
        {
            return Err(ResourceError::InsufficientCapacity);
        }
        let lease = ResourceLease {
            requirements: requirements.clone(),
            lease_id: self.next_lease_id,
        };
        self.next_lease_id = self.next_lease_id.saturating_add(1);
        self.active.insert(lease.lease_id, lease.clone());
        Ok(lease)
    }

    fn release(&mut self, lease: ResourceLease) -> Result<(), ResourceError> {
        match self.active.get(&lease.lease_id) {
            Some(existing) if existing == &lease => {
                self.active.remove(&lease.lease_id);
                self.released.insert(lease.lease_id, lease);
                Ok(())
            }
            Some(_) => Err(ResourceError::LeaseMismatch),
            None => match self.released.get(&lease.lease_id) {
                Some(existing) if existing == &lease => Ok(()),
                Some(_) => Err(ResourceError::LeaseMismatch),
                None => Err(ResourceError::UnknownLease),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutorError {
    Descriptor(String),
    Activation(String),
    Inference(String),
    Release(String),
    Unhealthy,
}

pub trait ExecutorFactory: Send + Sync {
    type Executor: NeuralPodExecutor;

    fn create(&self, descriptor: &NeuralPodDescriptor) -> Result<Self::Executor, ExecutorError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReferenceExecutorFactory;

impl ExecutorFactory for ReferenceExecutorFactory {
    type Executor = ReferenceNeuralExecutor;

    fn create(&self, descriptor: &NeuralPodDescriptor) -> Result<Self::Executor, ExecutorError> {
        ReferenceNeuralExecutor::new(descriptor.clone()).map_err(|error| match error {
            NeuralPodError::Descriptor(_) => ExecutorError::Descriptor("invalid descriptor".into()),
            other => ExecutorError::Activation(format!("{other:?}")),
        })
    }
}
