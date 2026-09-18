use ptr_pods::{DynPod, PodManifest, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, Effect, PodId, TypeId};
use std::sync::Arc;

struct EchoPod {
    manifest: PodManifest,
}

impl DynPod for EchoPod {
    fn manifest(&self) -> &PodManifest {
        &self.manifest
    }

    fn invoke(&self, input: TypedPayload) -> Result<TypedPayload, String> {
        Ok(input)
    }
}

#[test]
fn registry_resolves_by_semantic_capability_and_type() {
    let manifest = PodManifest {
        id: PodId::from("echo-v2"),
        capabilities: vec![CapabilityId::from("Echo<Text>")],
        accepts: vec![TypeId::from("Text")],
        produces: vec![TypeId::from("Text")],
        effects: vec![Effect::Pure],
        protocol_version: 1,
    };
    let mut registry = PodRegistry::default();
    registry.register(Arc::new(EchoPod { manifest }));

    let found = registry
        .resolve(&CapabilityId::from("Echo<Text>"), &TypeId::from("Text"))
        .expect("semantic capability resolves");
    assert_eq!(found.manifest().id.to_string(), "echo-v2");
    assert!(registry
        .resolve(&CapabilityId::from("Echo<Text>"), &TypeId::from("Bytes"))
        .is_none());
}
