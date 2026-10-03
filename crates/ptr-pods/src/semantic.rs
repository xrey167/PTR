use crate::ProtocolBinding;
use ptr_types::{
    ArtifactId, CapabilityId, Digest, Generation, NamespaceId, PodId, PodIdentity, PrincipalId,
    ProjectId, TypeId,
};
use sha2::{Digest as ShaDigest, Sha256};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PodKind {
    Context,
    Model,
    LanguageModel,
    Worker,
    Environment,
    Tool,
    Retrieval,
    Memory,
    Math,
    Reasoning,
    Planner,
    Router,
    Verifier,
    Critic,
    Simulator,
    Vision,
    Embedding,
    Quantization,
    Transport,
    Runtime,
    Agent,
    Composite,
    Policy,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ModelVariant {
    Base,
    Lora,
    Distilled,
    Moe,
    Quantized,
    Speculator,
    Embedding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticPodLifecycle {
    Candidate,
    Trained,
    Evaluated,
    Approved,
    Active,
    Superseded,
    Retired,
    Revoked,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PodResourceProfile {
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub max_concurrency: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodSemanticManifest {
    pub pod_id: PodId,
    pub pod_identity: PodIdentity,
    pub project: ProjectId,
    pub namespace: NamespaceId,
    pub generation: Generation,
    pub artifact_id: ArtifactId,
    pub kind: PodKind,
    pub semantic_role: String,
    pub domain: String,
    pub capabilities: Vec<CapabilityId>,
    pub accepts: Vec<TypeId>,
    pub produces: Vec<TypeId>,
    pub model_variant: Option<ModelVariant>,
    pub model_family: Option<String>,
    pub interface: Option<String>,
    pub adapter_identity: Option<String>,
    pub resources: PodResourceProfile,
    pub protocols: Vec<ProtocolBinding>,
    pub effects: Vec<ptr_types::Effect>,
    pub provenance: Vec<ptr_types::ProvenanceRef>,
    pub principals: Vec<PrincipalId>,
    pub lifecycle: SemanticPodLifecycle,
}

impl PodSemanticManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.pod_id.0.is_empty()
            || self.pod_identity.0.is_empty()
            || self.project.0.is_empty()
            || self.namespace.0.is_empty()
            || self.artifact_id.0.is_empty()
            || self.generation.0 == 0
        {
            return Err(ManifestError::MissingIdentity);
        }
        if self.capabilities.is_empty() || self.accepts.is_empty() || self.produces.is_empty() {
            return Err(ManifestError::MissingTypeContract);
        }
        if self.semantic_role.is_empty() || self.domain.is_empty() || self.provenance.is_empty() {
            return Err(ManifestError::MissingProvenanceOrRole);
        }
        if self.resources.max_concurrency == 0 {
            return Err(ManifestError::InvalidResources);
        }
        if matches!(self.lifecycle, SemanticPodLifecycle::Active) && self.protocols.is_empty() {
            return Err(ManifestError::ActiveWithoutProtocol);
        }
        if matches!(self.kind, PodKind::Model | PodKind::LanguageModel)
            && self.model_variant.is_none()
        {
            return Err(ManifestError::ModelVariantRequired);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest, ManifestError> {
        self.validate()?;
        let mut capabilities = self
            .capabilities
            .iter()
            .map(|value| value.0.clone())
            .collect::<Vec<_>>();
        let mut accepts = self
            .accepts
            .iter()
            .map(|value| value.0.clone())
            .collect::<Vec<_>>();
        let mut produces = self
            .produces
            .iter()
            .map(|value| value.0.clone())
            .collect::<Vec<_>>();
        let mut protocols = self
            .protocols
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>();
        let mut effects = self
            .effects
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>();
        let mut provenance = self
            .provenance
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>();
        let mut principals = self
            .principals
            .iter()
            .map(|value| value.0.clone())
            .collect::<Vec<_>>();
        capabilities.sort();
        accepts.sort();
        produces.sort();
        protocols.sort();
        effects.sort();
        provenance.sort();
        principals.sort();
        let mut bytes = Vec::new();
        macro_rules! put {
            ($value:expr) => {{
                let value = $value.to_string();
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(value.as_bytes());
            }};
        }
        put!(self.pod_id.0);
        put!(self.pod_identity.0);
        put!(self.project.0);
        put!(self.namespace.0);
        put!(self.generation.0);
        put!(self.artifact_id.0);
        put!(format!("{:?}", self.kind));
        put!(self.semantic_role);
        put!(self.domain);
        for item in &capabilities {
            put!(item);
        }
        for item in &accepts {
            put!(item);
        }
        for item in &produces {
            put!(item);
        }
        put!(format!("{:?}", self.model_variant));
        put!(self.model_family.as_deref().unwrap_or_default());
        put!(self.interface.as_deref().unwrap_or_default());
        put!(self.adapter_identity.as_deref().unwrap_or_default());
        put!(self.resources.ram_bytes);
        put!(self.resources.vram_bytes);
        put!(self.resources.max_concurrency);
        for item in &protocols {
            put!(item);
        }
        for item in &effects {
            put!(item);
        }
        for item in &provenance {
            put!(item);
        }
        for item in &principals {
            put!(item);
        }
        put!(format!("{:?}", self.lifecycle));
        Ok(Sha256::digest(bytes).into())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestError {
    MissingIdentity,
    MissingTypeContract,
    MissingProvenanceOrRole,
    InvalidResources,
    ActiveWithoutProtocol,
    ModelVariantRequired,
}
