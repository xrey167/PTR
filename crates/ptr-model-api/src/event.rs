use ptr_types::{CandidateId, Probability};
#[derive(Clone, Debug, PartialEq)]
pub enum ModelEvent {
    SemanticSlotUpdate { slot: u32, payload: Vec<f32> },
    HypothesisCreated { label: String, confidence: Probability },
    ConfidenceUpdated { slot: u32, confidence: Probability },
    OperatorRequested { operator: String },
    PodRequested { capability: String },
    CandidateReady(CandidateId),
    ActionReady { operation: String },
    Token(String),
    Finished,
}
