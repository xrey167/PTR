use ptr_types::{Generation, Probability, ProvenanceRef, Validity};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotKind {
    Goal,
    Constraint,
    Known,
    Observed,
    Hypothesis,
    Distribution,
    Unknown,
    Resource,
    Capability,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticSlot {
    pub id: u32,
    pub kind: SlotKind,
    pub generation: Generation,
    pub validity: Validity,
    pub confidence: Probability,
    pub value: Vec<f32>,
    pub provenance: Vec<ProvenanceRef>,
}
