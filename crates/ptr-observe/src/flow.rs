use ptr_types::RequestId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlowSignature {
    pub request: RequestId,
    pub operators: Vec<String>,
    pub state_transitions: Vec<String>,
}
