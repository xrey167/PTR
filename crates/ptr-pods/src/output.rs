use ptr_protocol::TypedPayload;
use ptr_types::{Digest, Generation, ProvenanceRef, Revision, TypeId};
use sha2::{Digest as ShaDigest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PodOutputKind {
    Observation,
    Candidate,
    Hypothesis,
    ToolResult,
    EnvironmentObservation,
    StateDelta,
    ActionProposal,
    VerifiedResult,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodOutput {
    pub kind: PodOutputKind,
    pub payload: TypedPayload,
    pub generation: Generation,
    pub manifest_digest: Digest,
    pub artifact_digest: Digest,
    pub provenance: Vec<ProvenanceRef>,
    pub dependencies: Vec<Digest>,
    pub revision: Revision,
    pub verified: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputError {
    MissingDigest,
    MissingProvenance,
    UnverifiedPromotion,
    TypeMismatch { expected: TypeId, actual: TypeId },
}

impl PodOutput {
    pub fn digest(&self) -> Digest {
        let mut bytes = Vec::new();
        bytes.push(self.kind_code());
        put_string(&mut bytes, &self.payload.type_id.0);
        put_bytes(&mut bytes, &self.payload.bytes);
        bytes.extend_from_slice(&self.generation.0.to_le_bytes());
        bytes.extend_from_slice(&self.manifest_digest);
        bytes.extend_from_slice(&self.artifact_digest);
        bytes.extend_from_slice(&self.revision.0.to_le_bytes());
        bytes.push(u8::from(self.verified));
        let mut dependencies = self.dependencies.clone();
        dependencies.sort();
        for dependency in dependencies {
            bytes.extend_from_slice(&dependency);
        }
        let mut provenance = self.provenance.clone();
        provenance.sort_by(|left, right| {
            left.source
                .0
                .cmp(&right.source.0)
                .then_with(|| left.note.cmp(&right.note))
        });
        for item in provenance {
            put_string(&mut bytes, &item.source.0);
            match item.note {
                Some(note) => {
                    bytes.push(1);
                    put_string(&mut bytes, &note);
                }
                None => bytes.push(0),
            }
        }
        Sha256::digest(bytes).into()
    }

    pub fn kind_code(&self) -> u8 {
        match self.kind {
            PodOutputKind::Observation => 0,
            PodOutputKind::Candidate => 1,
            PodOutputKind::Hypothesis => 2,
            PodOutputKind::ToolResult => 3,
            PodOutputKind::EnvironmentObservation => 4,
            PodOutputKind::StateDelta => 5,
            PodOutputKind::ActionProposal => 6,
            PodOutputKind::VerifiedResult => 7,
        }
    }

    pub fn validate(
        &self,
        expected_type: &TypeId,
        allow_promotion: bool,
    ) -> Result<(), OutputError> {
        if self.generation.0 == 0
            || self.manifest_digest == [0; 32]
            || self.artifact_digest == [0; 32]
        {
            return Err(OutputError::MissingDigest);
        }
        if self.provenance.is_empty() {
            return Err(OutputError::MissingProvenance);
        }
        if &self.payload.type_id != expected_type {
            return Err(OutputError::TypeMismatch {
                expected: expected_type.clone(),
                actual: self.payload.type_id.clone(),
            });
        }
        if matches!(
            self.kind,
            PodOutputKind::StateDelta | PodOutputKind::VerifiedResult
        ) && (!allow_promotion || !self.verified)
        {
            return Err(OutputError::UnverifiedPromotion);
        }
        Ok(())
    }
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn put_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}
