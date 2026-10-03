use ptr_pods::ResourceRequirements;
use ptr_runtime::{
    ExternalSandboxExecutor, NativeSandboxExecutor, NetworkMode, SandboxBackend, SandboxError,
    SandboxExecutor, SandboxLease, SandboxProfile, SandboxRequest,
};
use ptr_types::{CapabilityId, NamespaceId, ProjectId, ScopeId};

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

#[test]
fn every_unavailable_external_backend_fails_closed_at_admission() {
    for backend in [SandboxBackend::WinRsBox, SandboxBackend::SmolVm] {
        let executor = ExternalSandboxExecutor::new(backend);
        let expected = match backend {
            SandboxBackend::WinRsBox => SandboxError::BackendUnavailable("winrsbox"),
            SandboxBackend::SmolVm => SandboxError::BackendUnavailable("smolvm"),
        };
        assert_eq!(executor.admit(&profile()), Err(expected));
    }
}

#[test]
fn native_executor_rejects_a_lease_that_was_never_admitted() {
    let executor = NativeSandboxExecutor::default();
    let lease = SandboxLease {
        scope: ScopeId::from("forged"),
        project: ProjectId::from("project"),
        profile_digest: [0; 32],
    };

    assert_eq!(
        executor.execute(
            &lease,
            SandboxRequest {
                operation: "read".into(),
                payload: vec![1],
            }
        ),
        Err(SandboxError::UnknownLease)
    );
}
