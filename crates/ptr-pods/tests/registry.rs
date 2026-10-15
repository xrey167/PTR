use ptr_pods::{DynPod, PodManifest, PodRegistrationError, PodRegistry};
use ptr_protocol::TypedPayload;
use ptr_types::{CapabilityId, Effect, NamespaceId, PodId, ProjectId, TypeId};
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

fn dynpod(pod: &Arc<EchoPod>) -> Arc<dyn DynPod> {
    pod.clone()
}

fn current(registry: &PodRegistry, project: &str, id: &str) -> Arc<dyn DynPod> {
    registry
        .get(&ProjectId::from(project), &PodId::from(id))
        .expect("registered")
}

#[test]
fn replace_if_current_swaps_when_the_expected_entry_is_current() {
    let mut registry = PodRegistry::default();
    let first = echo("p", "echo");
    registry.register_new(first.clone()).unwrap();
    let second = echo("p", "echo");
    let old = registry
        .replace_if_current(&dynpod(&first), second.clone())
        .unwrap();
    assert!(Arc::ptr_eq(&old, &dynpod(&first)));
    assert!(Arc::ptr_eq(
        &current(&registry, "p", "echo"),
        &dynpod(&second)
    ));
    assert_eq!(registry.len(), 1);
}

#[test]
fn replace_if_current_rejects_a_stale_expectation_and_changes_nothing() {
    let mut registry = PodRegistry::default();
    let first = echo("p", "echo");
    registry.register_new(first.clone()).unwrap();
    let interloper = echo("p", "echo");
    registry.register(interloper.clone());

    let err = registry
        .replace_if_current(&dynpod(&first), echo("p", "echo"))
        .err()
        .expect("rejected");
    assert_eq!(
        err,
        PodRegistrationError::NotCurrent {
            project: ProjectId::from("p"),
            pod: PodId::from("echo"),
        }
    );
    assert_eq!(
        err.to_string(),
        "registered pod echo in project p is not the expected entry"
    );
    assert!(Arc::ptr_eq(
        &current(&registry, "p", "echo"),
        &dynpod(&interloper)
    ));
    assert_eq!(registry.len(), 1);
}

#[test]
fn replace_if_current_rejects_an_absent_key() {
    let mut registry = PodRegistry::default();
    let ghost = echo("p", "echo");
    assert_eq!(
        registry
            .replace_if_current(&dynpod(&ghost), echo("p", "echo"))
            .err()
            .expect("rejected"),
        PodRegistrationError::NotRegistered {
            project: ProjectId::from("p"),
            pod: PodId::from("echo"),
        }
    );
    assert_eq!(registry.len(), 0);
}

#[test]
fn replace_if_current_rejects_a_project_or_id_mismatch() {
    let mut registry = PodRegistry::default();
    let first = echo("p", "echo");
    registry.register_new(first.clone()).unwrap();
    for new in [echo("p", "other"), echo("q", "echo")] {
        let found = (new.manifest.project.clone(), new.manifest.id.clone());
        assert_eq!(
            registry
                .replace_if_current(&dynpod(&first), new)
                .err()
                .expect("rejected"),
            PodRegistrationError::IdentityMismatch {
                expected: (ProjectId::from("p"), PodId::from("echo")),
                found,
            }
        );
    }
    assert!(Arc::ptr_eq(
        &current(&registry, "p", "echo"),
        &dynpod(&first)
    ));
    assert_eq!(registry.len(), 1);
}

/// An entry names the lifecycle namespace of its implementation only while that implementation is
/// registered: a plain re-registration or a compare-and-replace takeover is another
/// implementation and names none; other entries keep theirs.
#[test]
fn the_namespace_of_an_entry_belongs_to_the_implementation_registered_with_it() {
    let (project, alpha, beta) = (
        ProjectId::from("p"),
        PodId::from("alpha"),
        PodId::from("beta"),
    );
    let mut registry = PodRegistry::default();
    assert_eq!(registry.namespace(&project, &alpha), None);

    registry.register_in_namespace(NamespaceId::from("ns-a"), echo("p", "alpha"));
    registry.register_in_namespace(NamespaceId::from("ns-a"), echo("p", "beta"));
    assert_eq!(
        registry.namespace(&project, &alpha),
        Some(&NamespaceId::from("ns-a"))
    );
    // Another project's pod of the same id has its own entry and no namespace.
    registry.register(echo("q", "alpha"));
    assert_eq!(registry.namespace(&ProjectId::from("q"), &alpha), None);

    // Re-registering for another namespace replaces the label with the implementation.
    registry.register_in_namespace(NamespaceId::from("ns-b"), echo("p", "alpha"));
    assert_eq!(
        registry.namespace(&project, &alpha),
        Some(&NamespaceId::from("ns-b"))
    );

    // A plain registration is an implementation of no named namespace.
    registry.register(echo("p", "alpha"));
    assert_eq!(registry.namespace(&project, &alpha), None);

    // A takeover replaces the implementation and drops the label; beta keeps its own.
    registry.register_in_namespace(NamespaceId::from("ns-a"), echo("p", "alpha"));
    let current: Arc<dyn DynPod> = registry.get(&project, &alpha).unwrap();
    registry
        .replace_if_current(&current, echo("p", "alpha"))
        .unwrap();
    assert_eq!(registry.namespace(&project, &alpha), None);
    assert_eq!(
        registry.namespace(&project, &beta),
        Some(&NamespaceId::from("ns-a"))
    );
}

/// A binding resolves only while the entry names the namespace it was bound with: re-registering
/// the same manifest for another namespace, or without one, is another implementation.
#[test]
fn a_binding_does_not_resolve_after_the_entry_moved_to_another_namespace() {
    let project = ProjectId::from("p");
    let alpha = PodId::from("alpha");
    let mut registry = PodRegistry::default();
    registry.register_in_namespace(NamespaceId::from("ns-a"), echo("p", "alpha"));
    let binding = registry.bind_registered(&project, &alpha).unwrap();
    assert_eq!(binding.namespace(), Some(&NamespaceId::from("ns-a")));
    assert!(registry.resolve_bound(&binding).is_some());

    registry.register_in_namespace(NamespaceId::from("ns-b"), echo("p", "alpha"));
    assert!(registry.resolve_bound(&binding).is_none());
    registry.register(echo("p", "alpha"));
    assert!(registry.resolve_bound(&binding).is_none());
    registry.register_in_namespace(NamespaceId::from("ns-a"), echo("p", "alpha"));
    assert!(registry.resolve_bound(&binding).is_some());
}
