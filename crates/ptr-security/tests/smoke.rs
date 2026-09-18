use ptr_security::PermissionSet;
use ptr_types::{CapabilityId, Effect};
#[test]
fn missing_capability_is_denied() {
    let p = PermissionSet::default();
    assert!(!p.allows(&CapabilityId::from("write"), Effect::Mutation));
}
