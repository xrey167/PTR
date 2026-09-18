use ptr_types::{CapabilityId, Effect};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default)]
pub struct PermissionSet { pub capabilities: BTreeSet<CapabilityId>, pub allow_mutation: bool, pub allow_external: bool, pub allow_irreversible: bool }

impl PermissionSet {
    pub fn allows(&self, capability: &CapabilityId, effect: Effect) -> bool {
        if !self.capabilities.contains(capability) { return false; }
        match effect {
            Effect::Pure | Effect::Read => true,
            Effect::Mutation => self.allow_mutation,
            Effect::External => self.allow_external,
            Effect::Irreversible => self.allow_irreversible,
        }
    }
}
