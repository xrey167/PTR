//! Scoped, verifier-bound synchronous effect dispatch for a trusted embedding host.
//!
//! The host authenticates principals and installs grants/adapters; a model never
//! receives admin access. Handles are process-local capabilities, not wire tokens.

use super::{PtrRuntime, RuntimeError};
use ptr_core::action_head::ActionIr;
use ptr_ledger::{effect_code, integrity, LedgerEvent, MAX_RETAINED_RESPONSE};
use ptr_security::{ActionAuthorization, AuthorizationDecision, AuthorizationDenial};
use ptr_types::{CapabilityId, CommitIndex, Effect, NodeId, ProjectId, TypeId, VerificationLevel};
use ptr_verifier::{VerificationStatus, Verifier};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_SESSIONS: usize = 1024;
const MAX_GRANTS: usize = 64;
const ACTION_DOMAIN: &[u8] = b"PTREXEC01-ACTION";

/// The exact action an audit record commits to, domain-separated so a digest of
/// these bytes cannot collide with a digest taken elsewhere in the system.
pub fn action_digest(action: &ActionIr) -> [u8; 32] {
    let mut material = Vec::from(ACTION_DOMAIN);
    for field in [
        action.target.as_str(),
        action.operation.as_str(),
        action.capability.0.as_str(),
        action.input_type.0.as_str(),
    ] {
        material.extend_from_slice(&(field.len() as u64).to_le_bytes());
        material.extend_from_slice(field.as_bytes());
    }
    material.extend_from_slice(&(action.payload.len() as u64).to_le_bytes());
    material.extend_from_slice(&action.payload);
    material.extend_from_slice(&action.revision.0.to_le_bytes());
    material.extend_from_slice(&action.generation.0.to_le_bytes());
    material.push(effect_code(action.effect));
    integrity::sha256(&material)
}

/// An attempt whose outcome this runtime does not know.
///
/// Its presence *is* the fence. There is no separate flag that a restart could
/// drop, because the fence is derived from committed history every time the
/// runtime is built.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsettledEffect {
    pub attempt: CommitIndex,
    pub key: Option<String>,
    pub target: String,
    pub operation: String,
    pub effect: Effect,
}

/// What a settled or reconciled attempt established, addressed by its
/// at-most-once key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SettledOutcome {
    /// The effect applied and its response is still retained, so a retry under
    /// the same key is answered from history.
    Applied { response: Vec<u8> },
    /// The effect applied, but this runtime cannot reproduce the response: it
    /// either exceeded [`MAX_RETAINED_RESPONSE`] or was established by
    /// reconciliation rather than observed. A retry is refused rather than
    /// answered with something else.
    AppliedWithoutResponse,
    /// Reconciliation established that nothing applied, so the at-most-once
    /// budget was never spent and a later attempt may proceed.
    NotApplied,
}

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

/// Which peers may be admitted, as what, and with which grants.
///
/// The table is host policy. A peer never names its own principal or chooses
/// its own grants, and nothing in a request can reach this structure — which is
/// the whole difference between admitting an identity and believing one.
///
/// Grants are produced by a closure rather than stored, because a grant owns its
/// verifier and executor and each session needs its own.
#[derive(Default)]
pub struct AdmissionPolicy {
    entries: BTreeMap<NodeId, PeerAdmission>,
}

struct PeerAdmission {
    principal: String,
    ttl: Duration,
    grants: Box<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
}

impl AdmissionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a peer the transport authenticated to one principal and one grant
    /// set.
    ///
    /// A second entry for the same peer is refused rather than replacing the
    /// first: a silent replacement is how a later, weaker entry widens an
    /// earlier one without anyone deciding to.
    pub fn admit(
        &mut self,
        peer: NodeId,
        principal: impl Into<String>,
        ttl: Duration,
        grants: impl Fn() -> Vec<ExecutionGrant> + Send + Sync + 'static,
    ) -> Result<(), ExecutionError> {
        let principal = principal.into();
        if !valid_identifier(&principal) || !valid_identifier(&peer.0) {
            return Err(ExecutionError::InvalidSession);
        }
        if self.entries.contains_key(&peer) {
            return Err(ExecutionError::DuplicatePeerEntry { peer });
        }
        self.entries.insert(
            peer,
            PeerAdmission {
                principal,
                ttl,
                grants: Box::new(grants),
            },
        );
        Ok(())
    }

    /// Withdraw a peer. Sessions admitted under it stop working immediately,
    /// because a session's authority is re-derived from this table at every use.
    pub fn withdraw(&mut self, peer: &NodeId) -> bool {
        self.entries.remove(peer).is_some()
    }

    pub fn admits(&self, peer: &NodeId) -> bool {
        self.entries.contains_key(peer)
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
/// An optional at-most-once key is part of the preparation rather than of the
/// action, because only the caller knows whether an identical request means
/// "the same one again" or "do it once more".
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
    key: Option<String>,
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
    /// A committed attempt has no settlement, so an external effect may already
    /// have applied. Distinct from `RuntimeFenced`, which is ambiguity about
    /// this runtime's own ledger rather than about the outside world.
    AmbiguousOutcome {
        attempt: CommitIndex,
    },
    /// A retry under a key whose effect applied, but whose response this runtime
    /// no longer holds. Refusing is the only honest answer: re-executing would
    /// apply the effect twice, and inventing a response would be worse.
    ResponseNotRetained {
        attempt: CommitIndex,
    },
    /// Reconciliation named an attempt that is not awaiting one.
    UnknownAttempt {
        attempt: CommitIndex,
    },
    InvalidKey,
    InvalidEvidence,
    /// No admission policy entry for this peer. A peer the host never bound is
    /// not a peer with fewer rights; it is not admitted at all.
    UnknownPeer {
        peer: NodeId,
    },
    /// The policy no longer admits the peer this session was admitted under.
    /// Authority is re-derived at every use, so a withdrawal takes effect at
    /// once rather than when a TTL happens to run out.
    PeerNotAdmitted {
        peer: NodeId,
    },
    DuplicatePeerEntry {
        peer: NodeId,
    },
    /// The audit record itself could not be committed. Nothing was attempted
    /// when this is returned from the attempt, so it denies rather than fences.
    Audit(Box<RuntimeError>),
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
    /// Set when the session came from the admission policy. Such a session is
    /// valid only while the policy still admits that peer.
    peer: Option<NodeId>,
}

pub(super) struct ExecutionState {
    issuer: Arc<()>,
    epoch: Arc<()>,
    sessions: BTreeMap<u64, SessionRecord>,
    next_session: u64,
    uncertain: bool,
    /// Rebuilt from committed history on every open, which is what makes the
    /// fence survive a restart.
    unsettled: BTreeMap<CommitIndex, UnsettledEffect>,
    settled: BTreeMap<String, SettledOutcome>,
    policy: AdmissionPolicy,
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self {
            issuer: Arc::new(()),
            epoch: Arc::new(()),
            sessions: BTreeMap::new(),
            next_session: 1,
            uncertain: false,
            unsettled: BTreeMap::new(),
            settled: BTreeMap::new(),
            policy: AdmissionPolicy::new(),
        }
    }
}

impl ExecutionState {
    /// Either kind of ambiguity blocks execution and commits: this runtime does
    /// not know the outcome of its own last append, or it does not know whether
    /// an external effect applied.
    pub(super) fn is_fenced(&self) -> bool {
        self.uncertain || !self.unsettled.is_empty()
    }

    /// Ambiguity about the ledger alone. Settling an attempt has to be allowed
    /// while the attempt fences the runtime, because settling is what ends it.
    pub(super) fn is_commit_uncertain(&self) -> bool {
        self.uncertain
    }

    pub(super) fn is_unsettled(&self, attempt: CommitIndex) -> bool {
        self.unsettled.contains_key(&attempt)
    }

    pub(super) fn key_in_flight(&self, key: &str) -> bool {
        self.unsettled
            .values()
            .any(|effect| effect.key.as_deref() == Some(key))
    }

    pub(super) fn record_attempt(&mut self, effect: UnsettledEffect) {
        self.unsettled.insert(effect.attempt, effect);
    }

    /// Validation rejects a settlement for an attempt that is not awaiting one,
    /// before append and again during replay, so the miss is unreachable here.
    pub(super) fn settle(&mut self, attempt: CommitIndex, outcome: SettledOutcome) {
        if let Some(effect) = self.unsettled.remove(&attempt) {
            if let Some(key) = effect.key {
                self.settled.insert(key, outcome);
            }
        }
    }

    pub(super) fn unsettled_effects(&self) -> Vec<UnsettledEffect> {
        self.unsettled.values().cloned().collect()
    }

    fn oldest_unsettled(&self) -> Option<CommitIndex> {
        self.unsettled.keys().next().copied()
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
        if let Some(attempt) = self.oldest_unsettled() {
            return Err(ExecutionError::AmbiguousOutcome { attempt });
        }
        let record = self
            .sessions
            .get(&handle.id)
            .ok_or(ExecutionError::SessionClosed)?;
        if Instant::now() >= record.expires_at {
            return Err(ExecutionError::Expired);
        }
        // Re-derived, never remembered: the same rule the neural gate applies to
        // cached state. A session that was admissible when it was issued says
        // nothing about whether its peer is admissible now.
        if let Some(peer) = &record.peer {
            if !self.policy.admits(peer) {
                return Err(ExecutionError::PeerNotAdmitted { peer: peer.clone() });
            }
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
        self.register_session(principal, grants, ttl, None)
    }

    fn register_session(
        &mut self,
        principal: impl Into<String>,
        grants: Vec<ExecutionGrant>,
        ttl: Duration,
        peer: Option<NodeId>,
    ) -> Result<ExecutionSession, ExecutionError> {
        if self.execution.uncertain {
            return Err(ExecutionError::RuntimeFenced);
        }
        if let Some(attempt) = self.execution.oldest_unsettled() {
            return Err(ExecutionError::AmbiguousOutcome { attempt });
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
                peer,
            },
        );
        Ok(ExecutionSession {
            issuer: self.execution.issuer.clone(),
            id,
        })
    }

    /// Install the table that says which peers may be admitted and as what.
    ///
    /// Replacing it withdraws every session whose peer the new table does not
    /// admit, for the same reason a withdrawal does: authority is re-derived.
    pub fn install_admission_policy(&mut self, policy: AdmissionPolicy) {
        self.execution.policy = policy;
    }

    /// Admit a peer the host's transport authenticated.
    ///
    /// This layer admits an identity; it does not establish one. The host passes
    /// what its transport proved — for the iroh backend, the QUIC-authenticated
    /// `Connection::remote_id` — and never a value read out of a request. The
    /// principal and the grants come from the policy, so there is no parameter
    /// through which a caller could name either.
    pub fn admit_peer(&mut self, peer: &NodeId) -> Result<ExecutionSession, ExecutionError> {
        let Some(entry) = self.execution.policy.entries.get(peer) else {
            return Err(ExecutionError::UnknownPeer { peer: peer.clone() });
        };
        let principal = entry.principal.clone();
        let ttl = entry.ttl;
        let grants = (entry.grants)();
        self.register_session(principal, grants, ttl, Some(peer.clone()))
    }

    /// Withdraw a peer from the policy. Sessions admitted under it stop working
    /// immediately.
    pub fn withdraw_peer(&mut self, peer: &NodeId) -> Result<(), ExecutionError> {
        if self.execution.policy.withdraw(peer) {
            Ok(())
        } else {
            Err(ExecutionError::UnknownPeer { peer: peer.clone() })
        }
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

    /// Prepare an execution with no at-most-once guarantee: a retry is a new
    /// request, which is the right default for an action whose repetition is
    /// meaningful.
    pub fn prepare_execution(
        &self,
        session: &ExecutionSession,
        project: &ProjectId,
        action: &ActionIr,
        ttl: Duration,
    ) -> Result<ExecutionPermit, ExecutionError> {
        self.prepare(session, project, action, ttl, None)
    }

    /// Prepare an execution that must apply at most once under `key`.
    ///
    /// The guarantee holds for as long as the attempt's record is retained. A
    /// floor that rises past it discards the memory, which is a retention
    /// obligation rather than something this layer can enforce.
    pub fn prepare_execution_once(
        &self,
        session: &ExecutionSession,
        project: &ProjectId,
        action: &ActionIr,
        ttl: Duration,
        key: impl Into<String>,
    ) -> Result<ExecutionPermit, ExecutionError> {
        let key = key.into();
        if !valid_identifier(&key) {
            return Err(ExecutionError::InvalidKey);
        }
        self.prepare(session, project, action, ttl, Some(key))
    }

    fn prepare(
        &self,
        session: &ExecutionSession,
        project: &ProjectId,
        action: &ActionIr,
        ttl: Duration,
        key: Option<String>,
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
            key,
        })
    }

    /// Attempts whose outcome is unknown, oldest first. An operator needs this
    /// to know what reconciliation is waiting on.
    pub fn unsettled_effects(&self) -> Vec<UnsettledEffect> {
        self.execution.unsettled_effects()
    }

    /// Resolve an attempt this runtime could not observe, with evidence from the
    /// system that received the effect.
    ///
    /// It records what an operator established. It never infers the outcome and
    /// never retries the effect: a runtime that could work out what happened
    /// would not have been fenced.
    pub fn reconcile_effect(
        &mut self,
        attempt: CommitIndex,
        applied: bool,
        evidence: impl Into<String>,
    ) -> Result<CommitIndex, ExecutionError> {
        let evidence = evidence.into();
        if !valid_identifier(&evidence) {
            return Err(ExecutionError::InvalidEvidence);
        }
        if !self.execution.is_unsettled(attempt) {
            return Err(ExecutionError::UnknownAttempt { attempt });
        }
        self.commit_settlement(LedgerEvent::EffectReconciled {
            attempt,
            applied,
            evidence,
        })
        .map_err(audit)
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

        // A key that already carries an outcome is answered from history rather
        // than by applying the effect a second time.
        if let Some(key) = permit.key.as_deref() {
            match self.execution.settled.get(key) {
                Some(SettledOutcome::Applied { response }) => return Ok(response.clone()),
                Some(SettledOutcome::AppliedWithoutResponse) => {
                    return Err(ExecutionError::ResponseNotRetained {
                        attempt: self.settled_attempt(key),
                    })
                }
                Some(SettledOutcome::NotApplied) | None => {}
            }
        }

        // Log first. After this append the runtime is fenced by a record rather
        // than by a flag, so a crash anywhere below still reopens knowing that
        // an external effect may have applied.
        let attempt = self
            .commit(LedgerEvent::EffectAttempted {
                key: permit.key.clone(),
                project: grant.scope.project.clone(),
                principal: principal.clone(),
                target: permit.action.target.clone(),
                operation: permit.action.operation.clone(),
                capability: permit.action.capability.clone(),
                effect: permit.action.effect,
                generation: permit.action.generation,
                revision: permit.action.revision,
                verification: report.level,
                action_digest: action_digest(&permit.action),
            })
            .map_err(audit)?;

        let output = match grant.executor.execute(VerifiedDispatch {
            action: &permit.action,
            principal: &principal,
            project: &grant.scope.project,
        }) {
            Ok(output) => output,
            // An executor error cannot distinguish "did not apply" from "applied
            // and could not say so", and a panic says even less. The attempt
            // stays unsettled either way, so the fence outlives this process and
            // reconciliation is the only way forward.
            Err(error) => return Err(ExecutionError::Executor(error)),
        };

        let response_digest = integrity::sha256(&output);
        let response = (output.len() <= MAX_RETAINED_RESPONSE).then(|| output.clone());
        self.commit_settlement(LedgerEvent::EffectSettled {
            attempt,
            response,
            response_digest,
        })
        .map_err(audit)?;
        Ok(output)
    }

    /// The attempt a retained key was settled at, for a diagnostic that would
    /// otherwise have to say "some earlier attempt".
    fn settled_attempt(&self, key: &str) -> CommitIndex {
        self.committed_events()
            .iter()
            .find(|committed| {
                matches!(
                    &committed.event,
                    LedgerEvent::EffectAttempted { key: Some(recorded), .. } if recorded == key
                )
            })
            .map(|committed| committed.index)
            .unwrap_or_default()
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

fn audit(error: RuntimeError) -> ExecutionError {
    ExecutionError::Audit(Box::new(error))
}

pub(super) fn valid_identifier(value: &str) -> bool {
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
                peer: None,
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
