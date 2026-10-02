use ptr_types::{Digest, Generation, PrincipalId, Revision};
use sha2::{Digest as ShaDigest, Sha256};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LineageBinding {
    pub key: String,
    pub generation: Generation,
    pub digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionManifest {
    pub generation: Generation,
    pub knowledge: Vec<LineageBinding>,
    pub artifacts: Vec<LineageBinding>,
    pub origins: Vec<String>,
    pub reader: Option<LineageBinding>,
    pub snapshot_revision: Revision,
    pub snapshot_digest: Digest,
    pub principal: PrincipalId,
    pub policy_revision: Revision,
    pub manifest_digest: Digest,
}

impl ExecutionManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        generation: Generation,
        mut knowledge: Vec<LineageBinding>,
        mut artifacts: Vec<LineageBinding>,
        mut origins: Vec<String>,
        reader: Option<LineageBinding>,
        snapshot_revision: Revision,
        snapshot_digest: Digest,
        principal: PrincipalId,
        policy_revision: Revision,
    ) -> Result<Self, ManifestError> {
        knowledge.sort_by(|left, right| left.key.cmp(&right.key));
        artifacts.sort_by(|left, right| left.key.cmp(&right.key));
        origins.sort();
        let manifest = Self {
            generation,
            knowledge,
            artifacts,
            origins,
            reader,
            snapshot_revision,
            snapshot_digest,
            principal,
            policy_revision,
            manifest_digest: [0; 32],
        };
        manifest.validate_structure()?;
        let digest = manifest.calculate_digest();
        Ok(Self {
            manifest_digest: digest,
            ..manifest
        })
    }

    pub fn validate(&self) -> Result<(), ManifestError> {
        self.validate_structure()?;
        if self.manifest_digest == [0; 32] {
            return Err(ManifestError::MissingDigest);
        }
        if self.calculate_digest() != self.manifest_digest {
            return Err(ManifestError::DigestMismatch);
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), ManifestError> {
        if self.generation.0 == 0 || self.knowledge.is_empty() || self.artifacts.is_empty() {
            return Err(ManifestError::MissingLineage);
        }
        if self.origins.is_empty() || self.principal.0.is_empty() {
            return Err(ManifestError::MissingProvenance);
        }
        if self.snapshot_digest == [0; 32] {
            return Err(ManifestError::MissingDigest);
        }
        for binding in self
            .knowledge
            .iter()
            .chain(self.artifacts.iter())
            .chain(self.reader.iter())
        {
            if binding.key.is_empty() || binding.generation.0 == 0 || binding.digest == [0; 32] {
                return Err(ManifestError::InvalidBinding);
            }
        }
        if let Some(reader) = &self.reader {
            if reader.generation != self.generation {
                return Err(ManifestError::ReaderGenerationMismatch);
            }
        }
        Ok(())
    }

    fn calculate_digest(&self) -> Digest {
        let mut bytes = Vec::new();
        macro_rules! put {
            ($value:expr) => {{
                let value = $value.to_string();
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(value.as_bytes());
            }};
        }
        put!(self.generation.0);
        for binding in self
            .knowledge
            .iter()
            .chain(self.artifacts.iter())
            .chain(self.reader.iter())
        {
            put!(&binding.key);
            put!(binding.generation.0);
            bytes.extend_from_slice(&binding.digest);
        }
        for origin in &self.origins {
            put!(origin);
        }
        put!(self.snapshot_revision.0);
        bytes.extend_from_slice(&self.snapshot_digest);
        put!(&self.principal.0);
        put!(self.policy_revision.0);
        Sha256::digest(bytes).into()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactLifecycle {
    Candidate,
    Trained,
    Evaluated,
    Approved,
    Active,
    Superseded,
    Retired,
    Revoked,
}

pub struct LifecycleGate;

impl LifecycleGate {
    pub fn transition(
        current: ArtifactLifecycle,
        next: ArtifactLifecycle,
        manifest_valid: bool,
        provenance_valid: bool,
        base_identity_valid: bool,
        verifier_passed: bool,
    ) -> Result<ArtifactLifecycle, ManifestError> {
        let allowed = matches!(
            (current, next),
            (ArtifactLifecycle::Candidate, ArtifactLifecycle::Trained)
                | (ArtifactLifecycle::Trained, ArtifactLifecycle::Evaluated)
                | (ArtifactLifecycle::Evaluated, ArtifactLifecycle::Approved)
                | (ArtifactLifecycle::Approved, ArtifactLifecycle::Active)
                | (ArtifactLifecycle::Active, ArtifactLifecycle::Superseded)
                | (ArtifactLifecycle::Active, ArtifactLifecycle::Retired)
                | (ArtifactLifecycle::Active, ArtifactLifecycle::Revoked)
        );
        if !allowed {
            return Err(ManifestError::InvalidLifecycleTransition);
        }
        if matches!(
            next,
            ArtifactLifecycle::Evaluated | ArtifactLifecycle::Approved | ArtifactLifecycle::Active
        ) && !(manifest_valid && provenance_valid && base_identity_valid && verifier_passed)
        {
            return Err(ManifestError::AdmissionRequired);
        }
        Ok(next)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManifestError {
    MissingLineage,
    MissingProvenance,
    MissingDigest,
    InvalidBinding,
    ReaderGenerationMismatch,
    DigestMismatch,
    InvalidLifecycleTransition,
    AdmissionRequired,
}
