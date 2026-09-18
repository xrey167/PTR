use ptr_types::{RequestId, Revision};
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRequest {
    pub request_id: RequestId,
    pub revision: Revision,
    pub raw_text: String,
}
