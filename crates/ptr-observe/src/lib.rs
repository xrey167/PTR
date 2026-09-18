use ptr_types::RequestId;

pub mod fields {
    pub const REQUEST_ID: &str = "ptr.request_id";
    pub const PROJECT_ID: &str = "ptr.project_id";
    pub const REVISION: &str = "ptr.revision";
    pub const GENERATION: &str = "ptr.generation";
    pub const CANDIDATE_ID: &str = "ptr.candidate_id";
    pub const POD_ID: &str = "ptr.pod_id";
    pub const CAPABILITY: &str = "ptr.capability";
    pub const REASONING_TYPE: &str = "ptr.reasoning_type";
    pub const EFFECT: &str = "ptr.effect";
    pub const COMMIT_INDEX: &str = "ptr.commit_index";
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlowSignature {
    pub request: RequestId,
    pub operators: Vec<String>,
    pub state_transitions: Vec<String>,
}
