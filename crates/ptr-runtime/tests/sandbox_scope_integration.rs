use ptr_pods::ResourceRequirements;
use ptr_runtime::{
    ExecutionScope, ExternalSandboxExecutor, NetworkMode, SandboxBackend, SandboxError,
    SandboxExecutor, SandboxProfile, ScopeEventKind, ScopeRegistry, ScopeState,
};
use ptr_types::{CapabilityId, NamespaceId, ProjectId, ScopeId, SessionId, Timestamp};

fn profile() -> SandboxProfile {
    SandboxProfile {
        project: ProjectId::from("project"),
        namespace: NamespaceId::from("namespace"),
        allowed_paths: vec![],
        network: NetworkMode::Deny,
        allow_child_processes: false,
        environment_allowlist: vec![],
        resources: ResourceRequirements {
            ram_bytes: 1,
            vram_bytes: 0,
            device: None,
            max_concurrency: 1,
        },
        timeout_ms: 100,
        capabilities: vec![CapabilityId::from("read")],
    }
}

fn scope(id: &str, parent: Option<ScopeId>) -> ExecutionScope {
    ExecutionScope {
        id: ScopeId::from(id),
        parent,
        session: SessionId::from("session"),
        project: ProjectId::from("project"),
        created_at: Timestamp(1),
        deadline: None,
        state: ScopeState::Created,
        cancellation_requested: false,
    }
}

#[test]
fn cleanup_releases_parent_and_child_scopes_in_dependency_order() {
    let mut registry = ScopeRegistry::default();
    registry.create(scope("root", None)).unwrap();
    registry
        .transition(&ScopeId::from("root"), ScopeState::Admitted)
        .unwrap();
    registry
        .transition(&ScopeId::from("root"), ScopeState::Started)
        .unwrap();
    registry
        .create(scope("child", Some(ScopeId::from("root"))))
        .unwrap();
    registry
        .transition(&ScopeId::from("child"), ScopeState::Admitted)
        .unwrap();
    registry
        .transition(&ScopeId::from("child"), ScopeState::Started)
        .unwrap();
    registry.cleanup(&ScopeId::from("root")).unwrap();

    assert_eq!(
        registry.get(&ScopeId::from("child")).unwrap().state,
        ScopeState::Released
    );
    assert_eq!(
        registry.get(&ScopeId::from("root")).unwrap().state,
        ScopeState::Released
    );
    assert!(registry.events().windows(2).any(|events| {
        events[0].scope == ScopeId::from("child")
            && events[0].kind == ScopeEventKind::Cancelled
            && events[1].scope == ScopeId::from("child")
            && events[1].kind == ScopeEventKind::Released
    }));
}

#[test]
fn unavailable_sandbox_backend_fails_closed_for_every_operation() {
    let executor = ExternalSandboxExecutor::new(SandboxBackend::WinRsBox);
    let admission = executor.admit(&profile());
    assert_eq!(admission, Err(SandboxError::BackendUnavailable("winrsbox")));

    let lease = ptr_runtime::SandboxLease {
        scope: ScopeId::from("unadmitted"),
        project: ProjectId::from("project"),
        profile_digest: [0; 32],
    };
    assert_eq!(
        executor.execute(
            &lease,
            ptr_runtime::SandboxRequest {
                operation: "read".into(),
                payload: vec![1],
            }
        ),
        Err(SandboxError::BackendUnavailable("winrsbox"))
    );
    assert_eq!(
        executor.abort(&lease),
        Err(SandboxError::BackendUnavailable("winrsbox"))
    );
    assert_eq!(
        executor.release(lease),
        Err(SandboxError::BackendUnavailable("winrsbox"))
    );
}
