//! Scoped, verifier-bound synchronous effect dispatch for a trusted embedding host.
//!
//! The host authenticates principals and installs grants/adapters; a model never
//! receives admin access. Handles are process-local capabilities, not wire tokens.

use super::PtrRuntime;
use ptr_core::action_head::ActionIr;
use ptr_security::{ActionAuthorization, AuthorizationDecision, AuthorizationDenial};
use ptr_types::{CapabilityId, Effect, ProjectId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationStatus, Verifier};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_SESSIONS: usize = 1024;
const MAX_GRANTS: usize = 64;

/// An exact match, never a prefix, wildcard or model-supplied permission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionScope {
    pub project: ProjectId,
    pub target: String,
    pub operation: String,
    pub capability: CapabilityId,
    pub input_type: TypeId,
    pub effect: Effect,
}

impl ActionScope {
    fn matches(&self, project: &ProjectId, action: &ActionIr) -> bool {
        self.project == *project
            && self.target == action.target
            && self.operation == action.operation
            && self.capability == action.capability
            && self.input_type == action.input_type
            && self.effect == action.effect
    }

    fn valid(&self) -> bool {
        [
            &self.project.0,
            &self.target,
            &self.operation,
            &self.capability.0,
            &self.input_type.0,
        ]
        .into_iter()
        .all(|value| valid_identifier(value))
    }
}

/// No score threshold may replace the required verification level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequiredVerification {
    FullSemantic,
    Deterministic,
}

impl RequiredVerification {
    fn accepts(self, level: VerificationLevel) -> bool {
        matches!(
            (self, level),
            (_, VerificationLevel::Deterministic)
                | (Self::FullSemantic, VerificationLevel::FullSemantic)
        )
    }
}

/// The executor receives only the exact action verified inside the runtime.
/// Implementations are trusted adapters and must finish the effect synchronously.
/// Detached/background work requires a separate downstream fencing protocol.
pub trait ActionExecutor: Send + Sync {
    fn execute(&self, dispatch: VerifiedDispatch<'_>) -> Result<Vec<u8>, String>;
}

/// Only the runtime may construct a dispatch after all checks.
///
/// ```compile_fail
/// use ptr_core::action_head::ActionIr;
/// use ptr_runtime::execution::VerifiedDispatch;
/// use ptr_types::ProjectId;
/// fn forge<'a>(action: &'a ActionIr, project: &'a ProjectId) -> VerifiedDispatch<'a> {
///     VerifiedDispatch { action, principal: "forged", project }
/// }
/// ```
pub struct VerifiedDispatch<'a> {
    action: &'a ActionIr,
    principal: &'a str,
    project: &'a ProjectId,
}

impl VerifiedDispatch<'_> {
    pub fn action(&self) -> &ActionIr {
        self.action
    }
    pub fn principal(&self) -> &str {
        self.principal
    }
    pub fn project(&self) -> &ProjectId {
        self.project
    }
}

/// Installed by the trusted host, never selected/replaced by an inference caller.
pub struct ExecutionGrant {
    scope: ActionScope,
    required: RequiredVerification,
    verifier: Box<dyn Verifier<ActionIr> + Send + Sync>,
    executor: Box<dyn ActionExecutor>,
}

impl ExecutionGrant {
    pub fn new<V, E>(
        scope: ActionScope,
        required: RequiredVerification,
        verifier: V,
        executor: E,
    ) -> Self
    where
        V: Verifier<ActionIr> + Send + Sync + 'static,
        E: ActionExecutor + 'static,
    {
        Self {
            scope,
            required,
            verifier: Box::new(verifier),
            executor: Box::new(executor),
        }
    }
}

/// An opaque session capability issued by a trusted host after authentication.
/// Clones intentionally share one session; revocation invalidates all clones.
/// No serialized identity string can construct this handle.
#[derive(Clone)]
pub struct ExecutionSession {
    issuer: Arc<()>,
    id: u64,
}

impl fmt::Debug for ExecutionSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ExecutionSession(<opaque>)")
    }
}

/// Single-use, action-bound preparation. Verification is performed at dispatch,
/// not trusted from a cached report or a caller-supplied boolean.
///
/// ```compile_fail
/// use ptr_runtime::execution::ExecutionPermit;
/// fn duplicate(permit: ExecutionPermit) -> ExecutionPermit { permit.clone() }
/// ```
///
/// ```compile_fail
/// use ptr_runtime::execution::ExecutionPermit;
/// use ptr_security::AuthorizationReceipt;
/// fn promote(receipt: AuthorizationReceipt) -> ExecutionPermit { receipt.into() }
/// ```
///
/// ```compile_fail
/// use ptr_runtime::{PtrRuntime, execution::{ExecutionPermit, ExecutionSession}};
/// fn replay(runtime: &mut PtrRuntime, session: &ExecutionSession, permit: ExecutionPermit) {
///     let _ = runtime.execute_prepared(session, permit);
///     let _ = runtime.execute_prepared(session, permit);
/// }
/// ```
pub struct ExecutionPermit {
    session: ExecutionSession,
    epoch: Arc<()>,
    grant: Arc<ExecutionGrant>,
    action: ActionIr,
    expires_at: Instant,
}

impl fmt::Debug for ExecutionPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ExecutionPermit(<opaque>)")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionError {
    InvalidSession,
    InvalidGrant,
    DuplicateScope,
    CapacityExceeded,
    CounterExhausted,
    InvalidTtl,
    ForeignRuntime,
    SessionClosed,
    SessionMismatch,
    Expired,
    StalePermit,
    ScopeDenied,
    ProjectMismatch,
    RuntimeFenced,
    AuthorizationDenied(AuthorizationDenial),
    VerificationRejected {
        status: VerificationStatus,
        level: VerificationLevel,
    },
    HardFinding,
    Executor(String),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PTR_EXECUTION_DENIED: {self:?}")
    }
}

impl std::error::Error for ExecutionError {}

struct SessionRecord {
    principal: String,
    expires_at: Instant,
    grants: Vec<Arc<ExecutionGrant>>,
}

pub(super) struct ExecutionState {
    issuer: Arc<()>,
    epoch: Arc<()>,
    sessions: BTreeMap<u64, SessionRecord>,
    next_session: u64,
    uncertain: bool,
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self {
            issuer: Arc::new(()),
            epoch: Arc::new(()),
            sessions: BTreeMap::new(),
            next_session: 1,
            uncertain: false,
        }
    }
}

impl ExecutionState {
    pub(super) fn is_fenced(&self) -> bool {
        self.uncertain
    }

    fn allocate_session_id(&mut self) -> Result<u64, ExecutionError> {
        let id = self.next_session;
        self.next_session = id.checked_add(1).ok_or(ExecutionError::CounterExhausted)?;
        Ok(id)
    }

    pub(super) fn invalidate_pending(&mut self) {
        self.epoch = Arc::new(());
    }

    pub(super) fn begin_commit(&mut self) {
        self.invalidate_pending();
        self.uncertain = true;
    }

    pub(super) fn complete_commit(&mut self) {
        self.uncertain = false;
    }

    fn session(&self, handle: &ExecutionSession) -> Result<&SessionRecord, ExecutionError> {
        if !Arc::ptr_eq(&self.issuer, &handle.issuer) {
            return Err(ExecutionError::ForeignRuntime);
        }
        if self.uncertain {
            return Err(ExecutionError::RuntimeFenced);
        }
        let record = self
            .sessions
            .get(&handle.id)
            .ok_or(ExecutionError::SessionClosed)?;
        if Instant::now() >= record.expires_at {
            return Err(ExecutionError::Expired);
        }
        Ok(record)
    }
}

impl PtrRuntime {
    /// Privileged embedding API. The host must authenticate `principal` before
    /// calling; this is not a password/token authenticator or a network endpoint.
    /// Only trusted host code may install verifiers, executors and exact grants.
    pub fn register_execution_session(
        &mut self,
        principal: impl Into<String>,
        grants: Vec<ExecutionGrant>,
        ttl: Duration,
    ) -> Result<ExecutionSession, ExecutionError> {
        if self.execution.uncertain {
            return Err(ExecutionError::RuntimeFenced);
        }
        let principal = principal.into();
        if !valid_identifier(&principal) {
            return Err(ExecutionError::InvalidSession);
        }
        let expires_at = deadline(ttl)?;
        if grants.is_empty() || grants.len() > MAX_GRANTS {
            return Err(ExecutionError::InvalidGrant);
        }
        for (index, grant) in grants.iter().enumerate() {
            if !grant.scope.valid() {
                return Err(ExecutionError::InvalidGrant);
            }
            if grants[..index]
                .iter()
                .any(|other| other.scope == grant.scope)
            {
                return Err(ExecutionError::DuplicateScope);
            }
        }
        self.execution
            .sessions
            .retain(|_, record| Instant::now() < record.expires_at);
        if self.execution.sessions.len() >= MAX_SESSIONS {
            return Err(ExecutionError::CapacityExceeded);
        }
        let id = self.execution.allocate_session_id()?;
        self.execution.sessions.insert(
            id,
            SessionRecord {
                principal,
                expires_at,
                grants: grants.into_iter().map(Arc::new).collect(),
            },
        );
        Ok(ExecutionSession {
            issuer: self.execution.issuer.clone(),
            id,
        })
    }

    /// Revocation removes all grants of this session. Re-registration issues a
    /// new identity; old handles/permits never become valid again.
    pub fn revoke_execution_session(
        &mut self,
        handle: &ExecutionSession,
    ) -> Result<(), ExecutionError> {
        if !Arc::ptr_eq(&self.execution.issuer, &handle.issuer) {
            return Err(ExecutionError::ForeignRuntime);
        }
        self.execution
            .sessions
            .remove(&handle.id)
            .ok_or(ExecutionError::SessionClosed)?;
        Ok(())
    }

    pub fn prepare_execution(
        &self,
        session: &ExecutionSession,
        project: &ProjectId,
        action: &ActionIr,
        ttl: Duration,
    ) -> Result<ExecutionPermit, ExecutionError> {
        let expires_at = deadline(ttl)?;
        let record = self.execution.session(session)?;
        let grant = record
            .grants
            .iter()
            .find(|grant| grant.scope.matches(project, action))
            .ok_or(ExecutionError::ScopeDenied)?;
        self.check_execution_action(project, action)?;
        Ok(ExecutionPermit {
            session: session.clone(),
            epoch: self.execution.epoch.clone(),
            grant: grant.clone(),
            action: action.clone(),
            expires_at: expires_at.min(record.expires_at),
        })
    }

    /// Consumes the permit even on rejection, verification failure or panic.
    /// Holds exclusive runtime access through the last check and synchronous
    /// executor invocation. No substitute action, verifier or executor is accepted.
    pub fn execute_prepared(
        &mut self,
        session: &ExecutionSession,
        permit: ExecutionPermit,
    ) -> Result<Vec<u8>, ExecutionError> {
        let record = self.execution.session(session)?;
        if !Arc::ptr_eq(&session.issuer, &permit.session.issuer) || session.id != permit.session.id
        {
            return Err(ExecutionError::SessionMismatch);
        }
        if Instant::now() >= permit.expires_at {
            return Err(ExecutionError::Expired);
        }
        if !Arc::ptr_eq(&self.execution.epoch, &permit.epoch) {
            return Err(ExecutionError::StalePermit);
        }
        if !record
            .grants
            .iter()
            .any(|grant| Arc::ptr_eq(grant, &permit.grant))
        {
            return Err(ExecutionError::ScopeDenied);
        }
        let principal = record.principal.clone();
        let grant = permit.grant;
        self.check_execution_action(&grant.scope.project, &permit.action)?;
        let report = grant.verifier.verify(&permit.action);
        if report.status != VerificationStatus::Pass || !grant.required.accepts(report.level) {
            return Err(ExecutionError::VerificationRejected {
                status: report.status,
                level: report.level,
            });
        }
        if report.findings.iter().any(|finding| finding.hard) {
            return Err(ExecutionError::HardFinding);
        }
        // Verification can be slow: expiry must be checked again at the last
        // admission point. Local permissions/lifecycle cannot change under &mut self.
        self.execution.session(session)?;
        if Instant::now() >= permit.expires_at {
            return Err(ExecutionError::Expired);
        }
        self.execution.uncertain = true;
        self.execution.invalidate_pending();
        let output = grant
            .executor
            .execute(VerifiedDispatch {
                action: &permit.action,
                principal: &principal,
                project: &grant.scope.project,
            })
            .map_err(ExecutionError::Executor)?;
        // Error/panic deliberately leaves this runtime fenced. There is no
        // automatic retry of a possibly applied external effect.
        self.execution.uncertain = false;
        Ok(output)
    }

    fn check_execution_action(
        &self,
        project: &ProjectId,
        action: &ActionIr,
    ) -> Result<(), ExecutionError> {
        if self.capsule_projects.get(&action.target) != Some(&project.0) {
            return Err(ExecutionError::ProjectMismatch);
        }
        let decision = self.permissions.authorize(ActionAuthorization {
            target: action.target.clone(),
            capability: action.capability.clone(),
            effect: action.effect,
            action_revision: action.revision,
            current_revision: self.revision(),
            action_generation: action.generation,
            current_generation: self.live_generation(&action.target),
            generation_revoked: self
                .revoked_generations
                .contains(&(action.target.clone(), action.generation)),
            require_capability: true,
            require_current_revision: true,
            require_live_generation: true,
        });
        match decision {
            AuthorizationDecision::Allow(_) => Ok(()),
            AuthorizationDecision::Deny(reason) => Err(ExecutionError::AuthorizationDenied(reason)),
        }
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control)
}

fn deadline(ttl: Duration) -> Result<Instant, ExecutionError> {
    if ttl.is_zero() {
        return Err(ExecutionError::InvalidTtl);
    }
    Instant::now()
        .checked_add(ttl)
        .ok_or(ExecutionError::InvalidTtl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_exhaustion_never_wraps() {
        let mut state = ExecutionState {
            next_session: u64::MAX,
            ..ExecutionState::default()
        };
        assert_eq!(
            state.allocate_session_id(),
            Err(ExecutionError::CounterExhausted)
        );
        assert_eq!(state.next_session, u64::MAX);
    }

    #[test]
    fn expired_session_is_not_admitted() {
        let mut state = ExecutionState::default();
        state.sessions.insert(
            1,
            SessionRecord {
                principal: "alice".into(),
                expires_at: Instant::now(),
                grants: vec![],
            },
        );
        let session = ExecutionSession {
            issuer: state.issuer.clone(),
            id: 1,
        };
        assert!(matches!(
            state.session(&session),
            Err(ExecutionError::Expired)
        ));
    }

    struct ProbeAdapter(Arc<std::sync::atomic::AtomicUsize>);

    impl Verifier<ActionIr> for ProbeAdapter {
        fn verify(&self, _: &ActionIr) -> ptr_verifier::VerificationReport {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ptr_verifier::VerificationReport {
                status: VerificationStatus::Pass,
                level: VerificationLevel::Deterministic,
                score: ptr_types::Probability::new(1.0).unwrap(),
                findings: vec![],
            }
        }
    }

    impl ActionExecutor for ProbeAdapter {
        fn execute(&self, _: VerifiedDispatch<'_>) -> Result<Vec<u8>, String> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![])
        }
    }

    #[test]
    fn expired_permit_never_reaches_verifier_or_executor() {
        use ptr_ledger::LedgerEvent;
        use ptr_types::{Generation, Revision};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut runtime = PtrRuntime::new(ptr_config::PtrConfig::default()).unwrap();
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: "capsule:a".into(),
                generation: Generation(1),
            })
            .unwrap();
        let action = ActionIr {
            target: "capsule:a".into(),
            operation: "write".into(),
            capability: "write".into(),
            input_type: "Bytes".into(),
            effect: Effect::Mutation,
            payload: vec![],
            revision: Revision(0),
            generation: Generation(1),
        };
        runtime
            .permissions_mut()
            .capabilities
            .insert(action.capability.clone());
        runtime.permissions_mut().allow_mutation = true;
        let calls = Arc::new(AtomicUsize::new(0));
        let grant = ExecutionGrant::new(
            ActionScope {
                project: ProjectId::from("p"),
                target: action.target.clone(),
                operation: action.operation.clone(),
                capability: action.capability.clone(),
                input_type: action.input_type.clone(),
                effect: action.effect,
            },
            RequiredVerification::Deterministic,
            ProbeAdapter(calls.clone()),
            ProbeAdapter(calls.clone()),
        );
        let session = runtime
            .register_execution_session("alice", vec![grant], Duration::from_secs(60))
            .unwrap();
        let mut permit = runtime
            .prepare_execution(
                &session,
                &ProjectId::from("p"),
                &action,
                Duration::from_secs(60),
            )
            .unwrap();
        permit.expires_at = Instant::now();
        assert_eq!(
            runtime.execute_prepared(&session, permit),
            Err(ExecutionError::Expired)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}
