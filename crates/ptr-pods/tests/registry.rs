use ptr_pods::{DynPod, PodManifest, PodRegistrationError, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, Effect, PodId, ProjectId, TypeId};
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

fn echo(project: &str, id: &str) -> Arc<EchoPod> {
    Arc::new(EchoPod {
        manifest: PodManifest {
            project: ProjectId::from(project),
            id: PodId::from(id),
            capabilities: vec![CapabilityId::from("Echo<Text>")],
            accepts: vec![TypeId::from("Text")],
            produces: vec![TypeId::from("Text")],
            effects: vec![Effect::Pure],
            protocol_version: 1,
        },
    })
}

#[test]
fn registry_resolves_by_semantic_capability_and_type() {
    let mut registry = PodRegistry::default();
    registry.register(echo("p", "echo-v2"));

    let found = registry
        .resolve(
            &ProjectId::from("p"),
            &CapabilityId::from("Echo<Text>"),
            &TypeId::from("Text"),
        )
        .expect("semantic capability resolves");
    assert_eq!(found.manifest().id.to_string(), "echo-v2");
    assert!(registry
        .resolve(
            &ProjectId::from("p"),
            &CapabilityId::from("Echo<Text>"),
            &TypeId::from("Bytes"),
        )
        .is_none());
}

#[test]
fn a_pod_registered_for_one_project_is_invisible_to_another() {
    let mut registry = PodRegistry::default();
    registry.register(echo("p", "echo-v2"));

    // Capability and input type match exactly. The project does not, and that
    // is the whole answer.
    assert!(registry
        .resolve(
            &ProjectId::from("other"),
            &CapabilityId::from("Echo<Text>"),
            &TypeId::from("Text"),
        )
        .is_none());
    assert!(registry
        .get(&ProjectId::from("other"), &PodId::from("echo-v2"))
        .is_none());
    assert!(registry
        .get(&ProjectId::from("p"), &PodId::from("echo-v2"))
        .is_some());
}

#[test]
fn two_projects_may_each_register_the_same_pod_id_without_shadowing() {
    let mut registry = PodRegistry::default();
    // Keying by project *and* id is what makes this safe: with an id-only key
    // the second registration would silently replace the first, and one
    // project's requests would land in the other project's Pod.
    assert!(registry.register(echo("p", "echo")).is_none());
    assert!(registry.register(echo("other", "echo")).is_none());
    assert_eq!(registry.len(), 2);

    for project in ["p", "other"] {
        let found = registry
            .resolve(
                &ProjectId::from(project),
                &CapabilityId::from("Echo<Text>"),
                &TypeId::from("Text"),
            )
            .expect("each project resolves its own Pod");
        assert_eq!(found.manifest().project, ProjectId::from(project));
    }

    // Re-registering within one project still replaces, as before.
    assert!(registry.register(echo("p", "echo")).is_some());
    assert_eq!(registry.len(), 2);
}

#[test]
fn register_new_inserts_a_new_id() {
    let mut registry = PodRegistry::default();
    assert_eq!(registry.register_new(echo("p", "echo")), Ok(()));
    // Another project's identical id is a different key.
    assert_eq!(registry.register_new(echo("other", "echo")), Ok(()));
    assert_eq!(registry.len(), 2);
    assert!(registry
        .get(&ProjectId::from("p"), &PodId::from("echo"))
        .is_some());
}

#[test]
fn register_new_rejects_a_duplicate_and_keeps_the_original() {
    let mut registry = PodRegistry::default();
    let original = echo("p", "echo");
    registry.register_new(original.clone()).unwrap();

    let err = registry.register_new(echo("p", "echo")).unwrap_err();
    assert_eq!(
        err,
        PodRegistrationError::AlreadyRegistered {
            project: ProjectId::from("p"),
            pod: PodId::from("echo"),
        }
    );
    assert_eq!(
        err.to_string(),
        "pod echo is already registered in project p"
    );
    assert_eq!(registry.len(), 1);
    let kept = registry
        .get(&ProjectId::from("p"), &PodId::from("echo"))
        .unwrap();
    let original: Arc<dyn DynPod> = original;
    assert!(Arc::ptr_eq(&kept, &original));
}

#[test]
fn register_still_replaces_after_register_new() {
    let mut registry = PodRegistry::default();
    let first = echo("p", "echo");
    registry.register_new(first.clone()).unwrap();
    let second = echo("p", "echo");
    let previous = registry.register(second.clone()).expect("replaces");
    let first: Arc<dyn DynPod> = first;
    let second: Arc<dyn DynPod> = second;
    assert!(Arc::ptr_eq(&previous, &first));
    let now = registry
        .get(&ProjectId::from("p"), &PodId::from("echo"))
        .unwrap();
    assert!(Arc::ptr_eq(&now, &second));
}
