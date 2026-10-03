use crate::NeuralPodDescriptor;
use ptr_protocol::TypedPayload;
use ptr_types::{ArtifactId, Generation, RequestId, Revision, TypeId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodWireRequest {
    pub request_id: RequestId,
    pub artifact_id: ArtifactId,
    pub manifest_hash: [u8; 32],
    pub generation: Generation,
    pub input_type: TypeId,
    pub expect_protocol: u32,
    pub revision: Revision,
    pub payload: TypedPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodWireResponse {
    pub request_id: RequestId,
    pub generation: Generation,
    pub revision: Revision,
    pub payload: TypedPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportError {
    ManifestMismatch,
    GenerationMismatch,
    ArtifactMismatch,
    InputTypeMismatch,
    ProtocolMismatch,
    RevisionMismatch,
    RequestMismatch,
    OutputTypeMismatch,
}

impl PodWireRequest {
    pub fn validate(
        &self,
        descriptor: &NeuralPodDescriptor,
        protocol: u32,
    ) -> Result<(), TransportError> {
        if self.manifest_hash != descriptor.manifest_hash {
            return Err(TransportError::ManifestMismatch);
        }
        if self.artifact_id != descriptor.artifact_id {
            return Err(TransportError::ArtifactMismatch);
        }
        if self.generation != descriptor.generation {
            return Err(TransportError::GenerationMismatch);
        }
        if self.input_type != descriptor.input_schema || self.payload.type_id != self.input_type {
            return Err(TransportError::InputTypeMismatch);
        }
        if self.expect_protocol != protocol {
            return Err(TransportError::ProtocolMismatch);
        }
        Ok(())
    }
}

impl PodWireResponse {
    pub fn validate(
        &self,
        request: &PodWireRequest,
        descriptor: &NeuralPodDescriptor,
    ) -> Result<(), TransportError> {
        if self.request_id != request.request_id {
            return Err(TransportError::RequestMismatch);
        }
        if self.generation != descriptor.generation {
            return Err(TransportError::GenerationMismatch);
        }
        if self.revision != request.revision {
            return Err(TransportError::RevisionMismatch);
        }
        if !descriptor.admits_output_type(&self.payload.type_id) {
            return Err(TransportError::OutputTypeMismatch);
        }
        Ok(())
    }
}
