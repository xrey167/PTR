use ptr_runtime::{ExecutionScope, ScopeError, ScopeEventKind, ScopeRegistry, ScopeState};
use ptr_types::{ProjectId, ScopeId, SessionId, Timestamp};

fn scope(id: &str, parent: Option<ScopeId>, project: &str) -> ExecutionScope {
    ExecutionScope {
        id: ScopeId::from(id),
        parent,
        session: SessionId::from("session"),
        project: ProjectId::from(project),
        created_at: Timestamp(1),
        deadline: None,
        state: ScopeState::Created,
        cancellation_requested: false,
    }
}

#[test]
fn cleanup_cancels_and_releases_active_children_before_releasing_parent() {
    let mut registry = ScopeRegistry::default();
    let root = ScopeId::from("root");
    let child = ScopeId::from("child");
    registry.create(scope("root", None, "project")).unwrap();
    registry.transition(&root, ScopeState::Admitted).unwrap();
    registry.transition(&root, ScopeState::Started).unwrap();
    registry
        .create(scope("child", Some(root.clone()), "project"))
        .unwrap();
    registry.transition(&child, ScopeState::Admitted).unwrap();
    registry.transition(&child, ScopeState::Started).unwrap();
    registry.cleanup(&root).unwrap();

    assert_eq!(registry.get(&root).unwrap().state, ScopeState::Released);
    assert_eq!(registry.get(&child).unwrap().state, ScopeState::Released);
    assert!(registry.get(&child).unwrap().cancellation_requested);
    assert_eq!(
        registry.events().last().unwrap().kind,
        ScopeEventKind::Released
    );
}

#[test]
fn child_scope_with_a_different_project_is_rejected() {
    let mut registry = ScopeRegistry::default();
    let root = ScopeId::from("root");
    registry.create(scope("root", None, "project-a")).unwrap();

    assert_eq!(
        registry.create(scope("child", Some(root), "project-b")),
        Err(ScopeError::ParentScopeMismatch)
    );
}

#[test]
fn released_scope_cleanup_is_idempotent_without_extra_events() {
    let mut registry = ScopeRegistry::default();
    let root = ScopeId::from("root");
    registry.create(scope("root", None, "project")).unwrap();
    registry.transition(&root, ScopeState::Admitted).unwrap();
    registry.transition(&root, ScopeState::Started).unwrap();
    registry.cancel(&root).unwrap();
    registry.cleanup(&root).unwrap();
    let event_count = registry.events().len();
    registry.cleanup(&root).unwrap();

    assert_eq!(registry.events().len(), event_count);
    assert_eq!(registry.get(&root).unwrap().state, ScopeState::Released);
}
