use ptr_types::{
    EpistemicState, Generation, Probability, ProvenanceRef, SemanticRole, UncertaintyKind, Validity,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticSlot {
    pub id: u32,
    pub role: SemanticRole,
    pub epistemic: EpistemicState,
    pub uncertainty: UncertaintyKind,
    pub generation: Generation,
    pub validity: Validity,
    pub confidence: Probability,
    pub value: Vec<f32>,
    pub provenance: Vec<ProvenanceRef>,
}
