//! Scoped, verifier-bound synchronous effect dispatch for a trusted embedding host.
//!
//! The host authenticates principals and installs grants/adapters; a model never
//! receives admin access. Handles are process-local capabilities, not wire tokens.

use super::{PtrRuntime, RuntimeError};
use ptr_core::action_head::ActionIr;
use ptr_ledger::{effect_code, integrity, LedgerEvent, MAX_RETAINED_RESPONSE};
use ptr_security::{ActionAuthorization, AuthorizationDecision, AuthorizationDenial};
use ptr_types::{
    CapabilityId, CommitIndex, Effect, Generation, NodeId, ProjectId, Revision, TypeId,
    VerificationLevel,
};
use ptr_verifier::{VerificationStatus, Verifier};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_SESSIONS: usize = 1024;
const MAX_GRANTS: usize = 64;
const ACTION_DOMAIN: &[u8] = b"PTREXEC01-ACTION";

/// The longest at-most-once key, in bytes, that may be spent.
///
/// A compacted snapshot carries every settled key, whatever its outcome, as a
/// string of 1 to this many bytes, as the layout before it did, and every later
/// snapshot carries it again, so a longer key would make every export after its
/// attempt settled fail. [`PtrRuntime::prepare_execution_once`] refuses a longer
/// key that has not settled here with [`ExecutionError::InvalidKey`], before it
/// can be spent, and a keyed `EffectAttempted` being committed that names one is
/// refused with [`RuntimeError::InvalidEffectKey`].
///
/// Earlier builds held a key only to being an identifier, of any length, and
/// the execution wire carries one of up to 64 KiB, so a log may hold a longer
/// key. Replay takes it as it is, and the runtime opens. Preparation lets such a
/// key through once it has settled, so a retry under it is answered from the
/// key's entry as it was before: the same request's outcome, or a refusal for
/// another request. It is still never spent again: an attempt under it, after
/// it was reconciled as not applied, is refused at commit. While its attempt is
/// unsettled the runtime is fenced, so no session exists to prepare under it. Once such a key has
/// settled, every export fails with `PTR_COMPACTED_SECTION_LIMIT`, as it did
/// before this bound.
pub const MAX_KEY_BYTES: usize = super::compacted::MAX_STRING_BYTES;

/// The exact action an audit record commits to, domain-separated so a digest of
/// these bytes cannot collide with a digest taken elsewhere in the system.
pub fn action_digest(action: &ActionIr) -> [u8; 32] {
    digest_at(action, action.revision, action.generation)
}

/// [`action_digest`] of `action` as it would read at `revision` and
/// `generation`: every other field, payload included, is the action's own.
fn digest_at(action: &ActionIr, revision: Revision, generation: Generation) -> [u8; 32] {
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
    material.extend_from_slice(&revision.0.to_le_bytes());
    material.extend_from_slice(&generation.0.to_le_bytes());
    material.push(effect_code(action.effect));
    integrity::sha256(&material)
}

/// Which action an attempt record committed to, and for whom: the fields a
/// permit under an at-most-once key has to repeat to be a retry of the attempt
/// that spent the key.
///
/// [`action_digest`] covers the action, payload included, but not the project
/// or the principal, which the record names beside it. All three are compared,
/// because any one of them left out makes a different request look like a
/// retry: another payload from the same principal, or the same action asked
/// for by another principal.
///
/// The digest also covers the revision and generation the action carried, and
/// those say where the runtime stood rather than what was asked for. Admission
/// requires a permit to carry the current ones, so a digest compared as
/// recorded would refuse every retry once an unrelated semantic delta committed
/// or the target's generation was superseded. A retry is therefore digested at
/// the revision and generation its attempt recorded, which is why both are kept
/// here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionIdentity {
    /// The project of the grant that admitted the attempt.
    pub project: ProjectId,
    /// The principal the attempt's session was admitted as.
    pub principal: String,
    /// The revision the attempt's action carried.
    pub revision: Revision,
    /// The generation the attempt's action carried.
    pub generation: Generation,
    /// [`action_digest`] of the action the attempt dispatched.
    pub action_digest: [u8; 32],
}

impl ActionIdentity {
    /// Whether a permit for `action` in `project` from `principal` repeats the
    /// request this identity recorded: the same project and principal, and the
    /// same digest once `action` is read at the recorded revision and
    /// generation.
    fn is_retry(&self, project: &ProjectId, principal: &str, action: &ActionIr) -> bool {
        self.project == *project
            && self.principal == principal
            && self.action_digest == digest_at(action, self.revision, self.generation)
    }
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
    /// What the attempt record committed to. A settlement that finds the
    /// effect applied binds the attempt's key to it.
    pub identity: ActionIdentity,
}

/// What a settled or reconciled attempt established, addressed by its
/// at-most-once key.
///
/// An outcome in which the effect applied names the attempt that settled the
/// key and what that attempt recorded, so a retry is answered or refused from
/// the key's own entry rather than from anything inferred about it later.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SettledOutcome {
    /// The effect applied and its response is still retained, so a retry of
    /// the same action under the same key is answered from history.
    Applied {
        /// The attempt record that was settled.
        attempt: CommitIndex,
        /// What that attempt recorded; no other action is answered under the
        /// key.
        identity: ActionIdentity,
        response: Vec<u8>,
    },
    /// The effect applied, but this runtime cannot reproduce the response: it
    /// either exceeded [`MAX_RETAINED_RESPONSE`] or was established by
    /// reconciliation rather than observed. A retry is refused rather than
    /// answered with something else.
    AppliedWithoutResponse {
        /// The attempt record that was settled or reconciled.
        attempt: CommitIndex,
        /// What that attempt recorded.
        identity: ActionIdentity,
    },
    /// Reconciliation established that nothing applied, so the at-most-once
    /// budget was never spent. The key binds no action, and a later attempt
    /// under it may proceed for any action, unless the key is longer than
    /// [`MAX_KEY_BYTES`], which a log written by an earlier build may hold: an
    /// attempt under such a key is refused at commit.
    NotApplied,
}

/// What a settlement or reconciliation record established, before
/// [`ExecutionState::settle`] binds it to the attempt it closes.
pub(super) enum Settlement {
    /// The effect applied; the response, when the record retained one.
    Applied(Option<Vec<u8>>),
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
    /// Whether `level` satisfies this requirement: deterministic verification
    /// satisfies either requirement; full semantic verification satisfies only
    /// `FullSemantic`.
    pub(crate) fn accepts(self, level: VerificationLevel) -> bool {
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

/// An adapter whose effect does not finish inside the call.
///
/// `start` hands the work off and returns. It must not report success as though the
/// effect had applied — it has not; all it says is that the work was accepted. What
/// happened is reported later through
/// [`PtrRuntime::settle_detached`](crate::PtrRuntime::settle_detached), and until
/// then the runtime is fenced by the attempt's own record.
///
/// Detached work is a property of the **authority**, not of the call: a grant is
/// either synchronous or detached, decided when the grant is issued. A caller that
/// could choose would be choosing how its own effect is audited.
pub trait DetachedExecutor: Send + Sync {
    /// Accept the work and return. `Err` means it was not accepted.
    fn start(&self, dispatch: VerifiedDispatch<'_>) -> Result<(), String>;
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
    dispatch: Dispatcher,
}

/// How a grant's effect is dispatched.
///
/// Two ways, and a grant is one of them. Which one is authority, not preference: a
/// caller that could pick would be picking how its own effect is audited, and the
/// audited window for work that finishes later is not the same window.
enum Dispatcher {
    /// Finishes inside the call and returns its output.
    Synchronous(Box<dyn ActionExecutor>),
    /// Accepts the work and answers later.
    Detached(Box<dyn DetachedExecutor>),
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
            dispatch: Dispatcher::Synchronous(Box::new(executor)),
        }
    }

    /// A grant whose effect is accepted now and reported later.
    pub fn detached<V, E>(
        scope: ActionScope,
        required: RequiredVerification,
        verifier: V,
        executor: E,
    ) -> Self
    where
        V: Verifier<ActionIr> + Send + Sync + 'static,
        E: DetachedExecutor + 'static,
    {
        Self {
            scope,
            required,
            verifier: Box::new(verifier),
            dispatch: Dispatcher::Detached(Box::new(executor)),
        }
    }

    /// Whether this grant's effect is reported later.
    pub fn is_detached(&self) -> bool {
        matches!(self.dispatch, Dispatcher::Detached(_))
    }
}

/// A dispatch whose outcome will be reported later.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetachedEffect {
    /// The attempt record this work is audited under, and the index a settlement
    /// must name.
    pub attempt: CommitIndex,
    /// Set when an at-most-once key had already been applied, so nothing was handed
    /// out this time. The attempt names the one that settled the key.
    pub already_applied: bool,
}

/// What admission decided: either the key's earlier outcome, or a fresh attempt.
enum Admission {
    /// An at-most-once key already applied for this action: the attempt that
    /// settled it, and its retained response.
    Replayed {
        attempt: CommitIndex,
        response: Vec<u8>,
    },
    /// A committed attempt, awaiting dispatch.
    Attempted(Attempted),
}

/// An attempt that has been committed and not yet dispatched.
struct Attempted {
    attempt: CommitIndex,
    grant: Arc<ExecutionGrant>,
    principal: String,
    action: ActionIr,
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
    /// A permit under an at-most-once key that `attempt` spent on another
    /// action: its project, its principal, or its action digest taken at the
    /// revision and generation that attempt recorded differs from what that
    /// attempt recorded. It is not a retry of that request, so it is neither
    /// answered with that request's outcome nor attempted under a spent key,
    /// and nothing is recorded.
    ///
    /// Keys are one namespace per runtime, shared by every principal and
    /// project, so the attempt may be another principal's.
    KeyBoundToAnotherAction {
        attempt: CommitIndex,
    },
    /// Reconciliation named an attempt that is not awaiting one.
    UnknownAttempt {
        attempt: CommitIndex,
    },
    /// An at-most-once key that is not a well-formed identifier, or one longer
    /// than [`MAX_KEY_BYTES`] that has not settled in this runtime. Refused at
    /// preparation, before it can be spent.
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
    /// The grant dispatches the other way: a detached grant cannot be run
    /// synchronously and a synchronous one cannot be detached. `detached` says which
    /// kind the grant is.
    DispatchMismatch {
        detached: bool,
    },
    /// A settlement named an attempt that is not an outstanding detached one.
    NotDetached {
        attempt: CommitIndex,
    },
    /// The audit record itself could not be committed.
    ///
    /// For the attempt record, nothing was attempted and no unsettled attempt is
    /// left to fence on, so it is a refusal. A failure in the ledger's own append
    /// still leaves this runtime's last append ambiguous, and so the runtime
    /// fenced until it is reopened; a record refused by validation before the
    /// append leaves it unfenced. For a reconciliation, the attempt is left
    /// awaiting one. A settlement that could not be committed after the effect
    /// applied is [`ExecutionError::SettlementNotRecorded`] instead.
    Audit(Box<RuntimeError>),
    /// The effect applied, as the executor or the detached adapter reported,
    /// but its settlement record could not be committed. `error` says why.
    ///
    /// This is not a refusal: the effect applied, and answering "nothing was
    /// attempted" would invite a retry that applies it twice. The runtime stays
    /// fenced: the failed append leaves its own last append ambiguous and the
    /// attempt unsettled, so settling or reconciling it again in this process is
    /// refused too. A reopened runtime replays what was written, and an attempt
    /// still unsettled there fences it until reconciliation records the outcome.
    /// Reconciling is an append of its own, so while the ledger has no index or
    /// record left the fence cannot be lifted in band. The response is not
    /// handed on, since the runtime could not record it; the execution wire
    /// reports this as `AppliedWithoutResponse`.
    SettlementNotRecorded {
        attempt: CommitIndex,
        error: Box<RuntimeError>,
    },
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

/// Borrowed view of the obligations a compacted snapshot carries: the fence, and
/// the at-most-once memory.
pub(super) type RetainedObligations<'a> = (
    &'a BTreeMap<CommitIndex, UnsettledEffect>,
    &'a BTreeMap<String, SettledOutcome>,
);

pub(super) struct ExecutionState {
    issuer: Arc<()>,
    epoch: Arc<()>,
    sessions: BTreeMap<u64, SessionRecord>,
    next_session: u64,
    uncertain: bool,
    /// Rebuilt from committed history on every open, which is what makes the
    /// fence survive a restart.
    unsettled: BTreeMap<CommitIndex, UnsettledEffect>,
    /// Detached attempts this process dispatched.
    ///
    /// Deliberately **not** rebuilt from history: the record says an effect was
    /// attempted, not how it was dispatched, and inventing that distinction on
    /// reopen would let a restarted runtime accept an "adapter answer" for work it
    /// never handed to an adapter. After a restart the fence stands and
    /// reconciliation is the way forward, which is the same answer a crash mid-effect
    /// gets.
    detached: BTreeSet<CommitIndex>,
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
            detached: BTreeSet::new(),
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

    /// The two things a compacted snapshot has to carry, because compaction is
    /// what takes the history they are derived from away.
    ///
    /// `unsettled` is the fence and `settled` is the at-most-once memory. Both
    /// are ordinarily rebuilt by replaying committed records on open, which is
    /// exactly what a raised floor makes impossible — so a snapshot that omits
    /// them describes a runtime that has silently forgotten what it promised.
    ///
    /// The other fields are deliberately **not** here. `sessions`, `policy` and
    /// `detached` are properties of this process rather than of committed
    /// history, and `uncertain` is ambiguity about a commit this process
    /// attempted; carrying any of them across a snapshot would let a restored
    /// runtime claim knowledge it never had.
    pub(super) fn retained_obligations(&self) -> RetainedObligations<'_> {
        (&self.unsettled, &self.settled)
    }

    /// Reinstate the obligations a snapshot carried.
    pub(super) fn restore_obligations(
        &mut self,
        unsettled: BTreeMap<CommitIndex, UnsettledEffect>,
        settled: BTreeMap<String, SettledOutcome>,
    ) {
        self.unsettled = unsettled;
        self.settled = settled;
    }

    /// Ambiguity about the ledger alone. Settling an attempt has to be allowed
    /// while the attempt fences the runtime, because settling is what ends it.
    pub(super) fn is_commit_uncertain(&self) -> bool {
        self.uncertain
    }

    pub(super) fn is_unsettled(&self, attempt: CommitIndex) -> bool {
        self.unsettled.contains_key(&attempt)
    }

    /// Whether a key has settled here, with either outcome.
    pub(super) fn has_settled(&self, key: &str) -> bool {
        self.settled.contains_key(key)
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
    ///
    /// An applied outcome is bound here to the attempt it closes and to what
    /// that attempt recorded, because this is the one place that holds both.
    pub(super) fn settle(&mut self, attempt: CommitIndex, settlement: Settlement) {
        if let Some(effect) = self.unsettled.remove(&attempt) {
            if let Some(key) = effect.key {
                let identity = effect.identity;
                let outcome = match settlement {
                    Settlement::Applied(Some(response)) => SettledOutcome::Applied {
                        attempt,
                        identity,
                        response,
                    },
                    Settlement::Applied(None) => {
                        SettledOutcome::AppliedWithoutResponse { attempt, identity }
                    }
                    Settlement::NotApplied => SettledOutcome::NotApplied,
                };
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
        // A session holds the grants admission returned, and `session()` re-derives
        // only whether the peer is *still admitted* — not what it may do. So a
        // replacement that kept a peer with narrower grants left the live session
        // running under the wider set until its TTL expired, which is the opposite
        // of this layer's own rule: authority is re-derived at use, never
        // remembered.
        //
        // Re-derived here rather than by dropping the session, because dropping it
        // would replace `PeerNotAdmitted` — which says *why* a withdrawn peer is
        // refused — with a bare `SessionClosed`. A session whose peer the new table
        // does not admit is therefore left in place for `session()` to refuse by
        // name.
        //
        // The expiry is deliberately not refreshed: a new policy may narrow what a
        // session can do, but it must not extend how long it lasts.
        let entries = &self.execution.policy.entries;
        for record in self.execution.sessions.values_mut() {
            let Some(peer) = &record.peer else {
                continue;
            };
            if let Some(entry) = entries.get(peer) {
                record.principal = entry.principal.clone();
                record.grants = (entry.grants)().into_iter().map(Arc::new).collect();
            }
        }
        // Permits already issued were scoped by the grants that have just changed.
        self.execution.invalidate_pending();
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
    /// The key names one request. Once an attempt under it has applied, a
    /// permit under it for the same action, project and principal is answered
    /// from that attempt's outcome, and a permit for any other is refused with
    /// [`ExecutionError::KeyBoundToAnotherAction`]. The action is compared at
    /// the revision and generation the attempt recorded, so a retry carrying
    /// the current ones is still answered after either has moved. A key
    /// reconciled as not applied binds nothing.
    ///
    /// Keys are one namespace per runtime, not one per principal or project. A
    /// key one principal spent is refused to every other, so the refusal tells
    /// a requester that somebody else spent that key, and whoever spends a key
    /// first makes it unusable for everyone else. A caller should choose keys
    /// that nobody else can guess.
    ///
    /// The guarantee holds for as long as the attempt's record is retained. A
    /// floor that rises past it discards the memory, which is a retention
    /// obligation rather than something this layer can enforce.
    ///
    /// A key that is not a well-formed identifier is refused with
    /// [`ExecutionError::InvalidKey`] before anything else is checked, and so is
    /// one longer than [`MAX_KEY_BYTES`] that has not settled here: every later
    /// compacted snapshot carries a settled key, so one no snapshot can carry
    /// must not be spent. A longer key that a log written by an earlier build
    /// settled is let through, so a retry under it is answered from its entry as
    /// it was before.
    pub fn prepare_execution_once(
        &self,
        session: &ExecutionSession,
        project: &ProjectId,
        action: &ActionIr,
        ttl: Duration,
        key: impl Into<String>,
    ) -> Result<ExecutionPermit, ExecutionError> {
        let key = key.into();
        if !self.is_usable_execution_key(&key) {
            return Err(ExecutionError::InvalidKey);
        }
        self.prepare(session, project, action, ttl, Some(key))
    }

    /// Whether [`prepare_execution_once`](Self::prepare_execution_once) would
    /// accept `key` at all: a well-formed identifier no longer than
    /// [`MAX_KEY_BYTES`], or a longer one that a log written by an earlier build
    /// already settled. Nothing is reserved or recorded, so a caller can refuse a
    /// key before it does any work that the refusal would otherwise follow.
    pub fn is_usable_execution_key(&self, key: &str) -> bool {
        valid_identifier(key) && (key.len() <= MAX_KEY_BYTES || self.execution.has_settled(key))
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
    ///
    /// A detached attempt it closes is no longer outstanding, as after
    /// [`PtrRuntime::settle_detached`], so an adapter answer that arrives later is
    /// refused with [`ExecutionError::NotDetached`].
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
        let reconciled = self
            .commit_settlement(LedgerEvent::EffectReconciled {
                attempt,
                applied,
                evidence,
            })
            .map_err(audit)?;
        self.execution.detached.remove(&attempt);
        Ok(reconciled)
    }

    /// Consumes the permit even on rejection, verification failure or panic.
    /// Holds exclusive runtime access through the last check and synchronous
    /// executor invocation. No substitute action, verifier or executor is accepted.
    pub fn execute_prepared(
        &mut self,
        session: &ExecutionSession,
        permit: ExecutionPermit,
    ) -> Result<Vec<u8>, ExecutionError> {
        let attempted = match self.admit_effect(session, permit, false)? {
            Admission::Replayed { response, .. } => return Ok(response),
            Admission::Attempted(attempted) => attempted,
        };
        let Dispatcher::Synchronous(executor) = &attempted.grant.dispatch else {
            unreachable!("admit_effect refused the other dispatch kind");
        };
        let output = match executor.execute(VerifiedDispatch {
            action: &attempted.action,
            principal: &attempted.principal,
            project: &attempted.grant.scope.project,
        }) {
            Ok(output) => output,
            // An executor error cannot distinguish "did not apply" from "applied
            // and could not say so", and a panic says even less. The attempt
            // stays unsettled either way, so the fence outlives this process and
            // reconciliation is the only way forward.
            Err(error) => return Err(ExecutionError::Executor(error)),
        };

        self.record_settlement(attempted.attempt, output.clone())
            .map_err(|error| ExecutionError::SettlementNotRecorded {
                attempt: attempted.attempt,
                error: Box::new(error),
            })?;
        Ok(output)
    }

    /// Dispatch an effect whose outcome arrives later.
    ///
    /// The attempt is committed first and is **not** settled, so the runtime stays
    /// fenced until the adapter reports back. That is the whole point: the audited
    /// window for detached work is open from here until
    /// [`PtrRuntime::settle_detached`] or [`PtrRuntime::reconcile_effect`] closes it,
    /// rather than closing when this call returns. A call that returned unfenced
    /// would be claiming the effect had finished.
    ///
    /// Whether an effect is detached is the grant's to say, not the caller's: a
    /// synchronous grant is refused here and a detached one is refused on the
    /// synchronous path.
    pub fn dispatch_detached(
        &mut self,
        session: &ExecutionSession,
        permit: ExecutionPermit,
    ) -> Result<DetachedEffect, ExecutionError> {
        let attempted = match self.admit_effect(session, permit, true)? {
            // A key whose outcome is already known is answered from history, exactly
            // as on the synchronous path: a retry must not hand the work out twice.
            Admission::Replayed { attempt, .. } => {
                return Ok(DetachedEffect {
                    attempt,
                    already_applied: true,
                })
            }
            Admission::Attempted(attempted) => attempted,
        };
        let Dispatcher::Detached(executor) = &attempted.grant.dispatch else {
            unreachable!("admit_effect refused the other dispatch kind");
        };
        match executor.start(VerifiedDispatch {
            action: &attempted.action,
            principal: &attempted.principal,
            project: &attempted.grant.scope.project,
        }) {
            Ok(()) => {}
            // "Not accepted" is not "not applied". An adapter that failed while
            // handing work off may have handed it off, so the attempt stays
            // unsettled and the fence stands — the same answer a synchronous
            // executor's error gets, for the same reason.
            Err(error) => return Err(ExecutionError::Executor(error)),
        }
        self.execution.detached.insert(attempted.attempt);
        Ok(DetachedEffect {
            attempt: attempted.attempt,
            already_applied: false,
        })
    }

    /// Record the answer a detached adapter eventually gave.
    ///
    /// Distinct from [`PtrRuntime::reconcile_effect`], which records what a person
    /// established from the receiving system. This is the adapter's own report, and
    /// it is only available for an attempt *this* runtime dispatched: after a restart
    /// the runtime cannot tell which unsettled attempts were detached, so it does not
    /// pretend to — the fence stands and reconciliation is the way forward.
    ///
    /// An attempt that is not outstanding, because it was never handed out here or
    /// has already been settled or reconciled, is refused with
    /// [`ExecutionError::NotDetached`] before anything is appended. Only an attempt
    /// still awaiting its answer reaches the ledger, so a failure there is an
    /// applied effect whose settlement was not recorded.
    pub fn settle_detached(
        &mut self,
        attempt: CommitIndex,
        response: Vec<u8>,
    ) -> Result<CommitIndex, ExecutionError> {
        if !self.execution.detached.contains(&attempt) || !self.execution.is_unsettled(attempt) {
            return Err(ExecutionError::NotDetached { attempt });
        }
        let settled = self.record_settlement(attempt, response).map_err(|error| {
            ExecutionError::SettlementNotRecorded {
                attempt,
                error: Box::new(error),
            }
        })?;
        self.execution.detached.remove(&attempt);
        Ok(settled)
    }

    /// Attempts this runtime dispatched to a detached adapter and has not heard back
    /// about, oldest first.
    pub fn outstanding_detached(&self) -> Vec<CommitIndex> {
        self.execution.detached.iter().copied().collect()
    }

    /// Commit a settlement with its response digest, retaining the response only
    /// within the bound.
    ///
    /// The digest is unconditional: "we did not keep the response" must never become
    /// "we do not know what happened".
    ///
    /// The effect has applied by the time this runs, so a failure is returned as
    /// the ledger's error for the caller to report as
    /// [`ExecutionError::SettlementNotRecorded`], never as a refusal.
    fn record_settlement(
        &mut self,
        attempt: CommitIndex,
        output: Vec<u8>,
    ) -> Result<CommitIndex, RuntimeError> {
        let response_digest = integrity::sha256(&output);
        let response = (output.len() <= MAX_RETAINED_RESPONSE).then_some(output);
        self.commit_settlement(LedgerEvent::EffectSettled {
            attempt,
            response,
            response_digest,
        })
    }

    /// Every check that must hold before an effect is attempted, and the attempt
    /// record itself.
    ///
    /// One sequence for both dispatch kinds. Two copies would be two chances to drift
    /// on the order that matters — the dispatch kind and every authorization check
    /// come **before** the attempt is committed, because a refusal must leave no
    /// record: a record with nothing behind it is a fence with nothing behind it.
    fn admit_effect(
        &mut self,
        session: &ExecutionSession,
        permit: ExecutionPermit,
        want_detached: bool,
    ) -> Result<Admission, ExecutionError> {
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
        if grant.is_detached() != want_detached {
            return Err(ExecutionError::DispatchMismatch {
                detached: grant.is_detached(),
            });
        }
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

        // What the attempt record would commit to. The record below is built
        // from these same values, so a key's binding and its record cannot
        // disagree.
        let identity = ActionIdentity {
            project: grant.scope.project.clone(),
            principal: principal.clone(),
            revision: permit.action.revision,
            generation: permit.action.generation,
            action_digest: action_digest(&permit.action),
        };

        // A key whose effect applied is answered from history rather than by
        // applying the effect a second time, and only for the action it was
        // spent on: a permit for another action, project or principal is not a
        // retry, and answering it would report an effect that never ran for it.
        // The permit passed the current-revision and live-generation checks
        // above, so it is compared at the position its attempt recorded.
        if let Some(key) = permit.key.as_deref() {
            match self.execution.settled.get(key) {
                Some(
                    SettledOutcome::Applied {
                        attempt,
                        identity: bound,
                        ..
                    }
                    | SettledOutcome::AppliedWithoutResponse {
                        attempt,
                        identity: bound,
                    },
                ) if !bound.is_retry(&identity.project, &identity.principal, &permit.action) => {
                    return Err(ExecutionError::KeyBoundToAnotherAction { attempt: *attempt })
                }
                Some(SettledOutcome::Applied {
                    attempt, response, ..
                }) => {
                    return Ok(Admission::Replayed {
                        attempt: *attempt,
                        response: response.clone(),
                    })
                }
                Some(SettledOutcome::AppliedWithoutResponse { attempt, .. }) => {
                    return Err(ExecutionError::ResponseNotRetained { attempt: *attempt })
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
                project: identity.project,
                principal: identity.principal,
                target: permit.action.target.clone(),
                operation: permit.action.operation.clone(),
                capability: permit.action.capability.clone(),
                effect: permit.action.effect,
                generation: permit.action.generation,
                revision: permit.action.revision,
                verification: report.level,
                action_digest: identity.action_digest,
            })
            .map_err(audit)?;

        Ok(Admission::Attempted(Attempted {
            attempt,
            grant,
            principal,
            action: permit.action,
        }))
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

/// An at-most-once key: an identifier short enough for a compacted snapshot to
/// carry.
pub(super) fn valid_key(value: &str) -> bool {
    valid_identifier(value) && value.len() <= MAX_KEY_BYTES
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
