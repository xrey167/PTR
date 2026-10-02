use ptr_protocol::TypedPayload;
use ptr_types::{Digest, Generation, ProvenanceRef, Revision, TypeId};

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
