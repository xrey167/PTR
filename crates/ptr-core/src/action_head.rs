use ptr_types::{CapabilityId, Effect, Generation, Revision, TypeId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionIr {
    pub operation: String,
    pub capability: CapabilityId,
    pub effect: Effect,
    pub input_type: TypeId,
    pub generation: Generation,
    pub revision: Revision,
    pub payload: Vec<u8>,
}
