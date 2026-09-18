use ptr_types::{RequestId, Revision, TypeId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRequest {
    pub request_id: RequestId,
    pub revision: Revision,
    pub raw_text: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelObservation {
    pub revision: Revision,
    pub source: String,
    pub type_id: TypeId,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelResumeRequest {
    pub request_id: RequestId,
    pub revision: Revision,
    pub round: u32,
    pub observation: ModelObservation,
}
