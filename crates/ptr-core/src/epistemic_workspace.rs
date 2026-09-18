use crate::semantic_slots::SemanticSlot;

#[derive(Clone, Debug, Default)]
pub struct EpistemicWorkspace {
    pub slots: Vec<SemanticSlot>,
    pub step: u32,
}

impl EpistemicWorkspace {
    pub fn live_slots(&self) -> impl Iterator<Item = &SemanticSlot> {
        self.slots.iter().filter(|s| matches!(s.validity, ptr_types::Validity::Live))
    }
}
