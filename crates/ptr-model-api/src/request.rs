use ptr_types::{RequestId, Revision, SemanticContext, TypeId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRequest {
    pub request_id: RequestId,
    pub semantic_context: SemanticContext,
    pub raw_text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelObservation {
    pub revision: Revision,
    pub source: String,
    pub type_id: TypeId,
    pub payload: Vec<u8>,
}

impl ModelRequest {
    pub fn revision(&self) -> ptr_types::Revision {
        self.semantic_context.revision
    }
}

impl ModelResumeRequest {
    pub fn revision(&self) -> ptr_types::Revision {
        self.semantic_context.revision
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelResumeRequest {
    pub request_id: RequestId,
    pub semantic_context: SemanticContext,
    pub round: u32,
    pub observation: ModelObservation,
}
