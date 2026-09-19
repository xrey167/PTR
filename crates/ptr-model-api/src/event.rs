use ptr_types::{CandidateId, CapabilityId, Probability, ReasoningOperator, TypeId};

#[derive(Clone, Debug, PartialEq)]
pub enum ModelEvent {
    SemanticSlotUpdate {
        slot: u32,
        payload: Vec<f32>,
    },
    HypothesisCreated {
        label: String,
        confidence: Probability,
    },
    ConfidenceUpdated {
        slot: u32,
        confidence: Probability,
    },
    OperatorRequested {
        operator: ReasoningOperator,
    },
    PodRequested {
        capability: CapabilityId,
        input_type: TypeId,
        payload: Vec<u8>,
    },
    CandidateReady(CandidateId),
    ActionReady {
        operation: String,
    },
    Token(String),
    Finished,
}
