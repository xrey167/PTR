use ptr_types::{RequestId, Revision};
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApiRequest {
    pub id: RequestId,
    pub text: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApiResponse {
    pub id: RequestId,
    pub revision: Revision,
    pub text: String,
}
