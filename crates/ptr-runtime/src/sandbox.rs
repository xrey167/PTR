use ptr_pods::ResourceRequirements;
use ptr_types::{CapabilityId, NamespaceId, ProjectId, ScopeId};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetworkMode {
    Deny,
    Allowlist,
    Full,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxProfile {
    pub project: ProjectId,
    pub namespace: NamespaceId,
    pub allowed_paths: Vec<String>,
    pub network: NetworkMode,
    pub allow_child_processes: bool,
    pub environment_allowlist: Vec<String>,
    pub resources: ResourceRequirements,
    pub timeout_ms: u64,
    pub capabilities: Vec<CapabilityId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxLease {
    pub scope: ScopeId,
    pub project: ProjectId,
    pub profile_digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxRequest {
    pub operation: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SandboxResponse {
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SandboxError {
    InvalidProfile,
    UnknownLease,
    RevokedLease,
    ProfileMismatch,
    NativeOnly,
    BackendUnavailable(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SandboxBackend {
    WinRsBox,
    SmolVm,
}

/// Adapter boundary for optional external isolation backends. The production
/// build does not silently fall back to native execution when an external
/// backend is unavailable; admission fails closed instead.
pub struct ExternalSandboxExecutor {
    pub backend: SandboxBackend,
}

impl ExternalSandboxExecutor {
    pub const fn new(backend: SandboxBackend) -> Self {
        Self { backend }
    }

    fn unavailable(&self) -> SandboxError {
        match self.backend {
            SandboxBackend::WinRsBox => SandboxError::BackendUnavailable("winrsbox"),
            SandboxBackend::SmolVm => SandboxError::BackendUnavailable("smolvm"),
        }
    }
}

impl SandboxExecutor for ExternalSandboxExecutor {
    fn admit(&self, _profile: &SandboxProfile) -> Result<SandboxLease, SandboxError> {
        Err(self.unavailable())
    }

    fn execute(
        &self,
        _lease: &SandboxLease,
        _request: SandboxRequest,
    ) -> Result<SandboxResponse, SandboxError> {
        Err(self.unavailable())
    }

    fn abort(&self, _lease: &SandboxLease) -> Result<(), SandboxError> {
        Err(self.unavailable())
    }

    fn release(&self, _lease: SandboxLease) -> Result<(), SandboxError> {
        Err(self.unavailable())
    }
}

pub trait SandboxExecutor: Send + Sync {
    fn admit(&self, profile: &SandboxProfile) -> Result<SandboxLease, SandboxError>;
    fn execute(
        &self,
        lease: &SandboxLease,
        request: SandboxRequest,
    ) -> Result<SandboxResponse, SandboxError>;
    fn abort(&self, lease: &SandboxLease) -> Result<(), SandboxError>;
    fn release(&self, lease: SandboxLease) -> Result<(), SandboxError>;
}

/// Reference native executor. It does not cross a process boundary and therefore
/// deliberately executes only the typed payload pass-through used by tests. Real
/// effects remain owned by `ptr-execwire`.
#[derive(Default)]
pub struct NativeSandboxExecutor {
    leases: Mutex<BTreeMap<ScopeId, (ProjectId, [u8; 32])>>,
    next_lease: AtomicU64,
}

impl SandboxExecutor for NativeSandboxExecutor {
    fn admit(&self, profile: &SandboxProfile) -> Result<SandboxLease, SandboxError> {
        if profile.project.0.is_empty() || profile.namespace.0.is_empty() || profile.timeout_ms == 0
        {
            return Err(SandboxError::InvalidProfile);
        }
        let sequence = self.next_lease.fetch_add(1, Ordering::Relaxed);
        let scope = ScopeId::from(format!("native:{}:{sequence}", profile.project.0).as_str());
        self.leases
            .lock()
            .expect("sandbox lease mutex poisoned")
            .insert(
                scope.clone(),
                (profile.project.clone(), profile_digest(profile)),
            );
        Ok(SandboxLease {
            scope,
            project: profile.project.clone(),
            profile_digest: profile_digest(profile),
        })
    }

    fn execute(
        &self,
        lease: &SandboxLease,
        request: SandboxRequest,
    ) -> Result<SandboxResponse, SandboxError> {
        let valid = self
            .leases
            .lock()
            .expect("sandbox lease mutex poisoned")
            .get(&lease.scope)
            .is_some_and(|(project, digest)| {
                project == &lease.project && digest == &lease.profile_digest
            });
        if !valid {
            return Err(SandboxError::UnknownLease);
        }
        Ok(SandboxResponse {
            payload: request.payload,
        })
    }

    fn abort(&self, lease: &SandboxLease) -> Result<(), SandboxError> {
        self.release(lease.clone())
    }

    fn release(&self, lease: SandboxLease) -> Result<(), SandboxError> {
        let mut leases = self.leases.lock().expect("sandbox lease mutex poisoned");
        if leases.remove(&lease.scope).is_none() {
            return Ok(());
        }
        Ok(())
    }
}

fn profile_digest(profile: &SandboxProfile) -> [u8; 32] {
    let mut canonical = Vec::new();
    push_string(&mut canonical, &profile.project.0);
    push_string(&mut canonical, &profile.namespace.0);
    for path in &profile.allowed_paths {
        push_string(&mut canonical, path);
    }
    canonical.push(profile.network as u8);
    canonical.push(profile.allow_child_processes as u8);
    for variable in &profile.environment_allowlist {
        push_string(&mut canonical, variable);
    }
    canonical.extend_from_slice(&profile.resources.ram_bytes.to_le_bytes());
    canonical.extend_from_slice(&profile.resources.vram_bytes.to_le_bytes());
    if let Some(device) = &profile.resources.device {
        push_string(&mut canonical, &device.0);
    }
    canonical.extend_from_slice(&profile.resources.max_concurrency.to_le_bytes());
    canonical.extend_from_slice(&profile.timeout_ms.to_le_bytes());
    for capability in &profile.capabilities {
        push_string(&mut canonical, &capability.0);
    }
    Sha256::digest(canonical).into()
}

fn push_string(target: &mut Vec<u8>, value: &str) {
    target.extend_from_slice(&(value.len() as u64).to_le_bytes());
    target.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

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
            capabilities: vec![],
        }
    }

    #[test]
    fn native_executor_releases_and_rejects_stale_leases() {
        let executor = NativeSandboxExecutor::default();
        let lease = executor.admit(&profile()).unwrap();
        let response = executor
            .execute(
                &lease,
                SandboxRequest {
                    operation: "read".into(),
                    payload: vec![1],
                },
            )
            .unwrap();
        assert_eq!(response.payload, vec![1]);
        executor.release(lease.clone()).unwrap();
        executor.release(lease.clone()).unwrap();
        assert_eq!(
            executor.execute(
                &lease,
                SandboxRequest {
                    operation: "read".into(),
                    payload: vec![]
                }
            ),
            Err(SandboxError::UnknownLease)
        );
    }

    #[test]
    fn native_leases_are_unique_and_bind_project_and_profile_digest() {
        let executor = NativeSandboxExecutor::default();
        let first = executor.admit(&profile()).unwrap();
        let second = executor.admit(&profile()).unwrap();
        assert_ne!(first.scope, second.scope);

        let forged = SandboxLease {
            scope: first.scope.clone(),
            project: ProjectId::from("other-project"),
            profile_digest: [0; 32],
        };
        assert_eq!(
            executor.execute(
                &forged,
                SandboxRequest {
                    operation: "read".into(),
                    payload: vec![]
                }
            ),
            Err(SandboxError::UnknownLease)
        );
    }

    #[test]
    fn unavailable_external_backend_never_falls_back_to_native() {
        let executor = ExternalSandboxExecutor::new(SandboxBackend::SmolVm);
        assert_eq!(
            executor.admit(&profile()),
            Err(SandboxError::BackendUnavailable("smolvm"))
        );
    }
}
