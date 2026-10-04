//! Scoped, verifier-bound synchronous effect dispatch for a trusted embedding host.
//!
//! The host authenticates principals and installs grants/adapters; a model never
//! receives admin access. Handles are process-local capabilities, not wire tokens.

use super::{PtrRuntime, RuntimeError};
use ptr_core::action_head::ActionIr;
use ptr_ledger::{effect_code, integrity, LedgerEvent, MAX_RETAINED_RESPONSE};
pub use ptr_net::AuthenticatedPeer;
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

/// The capability to record how an ambiguous effect turned out (D6 in issue #23).
///
/// Reconciliation lifts the fence a crash or an unreported adapter left standing,
/// on an operator's evidence, so whoever can call it decides what the audit says
/// happened. It stays a privileged host API — it never crosses a wire — and this
/// type makes "privileged" a property of the code rather than of a comment: it has
/// no public constructor, it is not `Clone`, and a runtime hands out one, once
/// ([`PtrRuntime::take_reconciliation_authority`]).
///
/// ```compile_fail
/// // It cannot be forged...
/// let _ = ptr_runtime::execution::ReconciliationAuthority { issuer: todo!() };
/// ```
///
/// ```compile_fail
/// // ...nor duplicated: there is no owned copy to be had from a borrow.
/// fn copy(
///     a: &ptr_runtime::execution::ReconciliationAuthority,
/// ) -> ptr_runtime::execution::ReconciliationAuthority {
///     a.clone()
/// }
/// ```
pub struct ReconciliationAuthority {
    issuer: Arc<()>,
}

impl fmt::Debug for ReconciliationAuthority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ReconciliationAuthority")
    }
}

/// Bound on a member's node id, principal and grant set name, in bytes.
///
/// A membership is projected into committed state and carried by every compacted
/// snapshot as one string, which that format bounds; holding each field well under
/// it keeps an admission that commits from being one that later fails every export.
pub const MAX_MEMBER_FIELD_BYTES: usize = 256;

/// The host's authority: named grant sets, each with the session lifetime it
/// grants (D5 in issue #23).
///
/// **Membership is history; authority is not.** Which peers are admitted, and
/// under which principal and grant set name, is committed to the ledger by
/// [`PtrRuntime::admit_member`] and replayed on every open. What a grant set
/// *permits* is behaviour — a grant owns its verifier and executor — and is
/// supplied here by the host at every start. A name the host does not supply
/// refuses the peer at use ([`ExecutionError::UnknownGrantSet`]) rather than
/// granting anything by default.
///
/// Grants are produced by a closure rather than stored, because a grant owns its
/// verifier and executor and each session needs its own.
#[derive(Default)]
pub struct GrantSets {
    sets: BTreeMap<String, GrantSet>,
}

struct GrantSet {
    ttl: Duration,
    grants: Box<dyn Fn() -> Vec<ExecutionGrant> + Send + Sync>,
}

impl GrantSets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Define a named grant set.
    ///
    /// A second definition under the same name is refused rather than replacing
    /// the first: a silent replacement is how a later, weaker definition widens
    /// an earlier one without anyone deciding to.
    pub fn define(
        &mut self,
        name: impl Into<String>,
        ttl: Duration,
        grants: impl Fn() -> Vec<ExecutionGrant> + Send + Sync + 'static,
    ) -> Result<(), ExecutionError> {
        let name = name.into();
        if !valid_member_field(&name) {
            return Err(ExecutionError::InvalidSession);
        }
        if self.sets.contains_key(&name) {
            return Err(ExecutionError::DuplicateGrantSet { name });
        }
        self.sets.insert(
            name,
            GrantSet {
                ttl,
                grants: Box::new(grants),
            },
        );
        Ok(())
    }

    /// Whether a grant set of this name is defined.
    pub fn defines(&self, name: &str) -> bool {
        self.sets.contains_key(name)
    }
}

/// One admitted peer, as committed history describes it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Member {
    pub(super) principal: String,
    pub(super) grant_set: String,
    /// The commit index of the admission record, or the floor of the compacted
    /// snapshot it was restored from. Sessions are bound to it.
    pub(super) admitted_at: CommitIndex,
}

/// A node id, principal or grant set name a membership may carry.
pub(super) fn valid_member_field(value: &str) -> bool {
    valid_identifier(value) && value.len() <= MAX_MEMBER_FIELD_BYTES
}

/// Why a membership record cannot be committed or replayed, or `None` if it can.
///
/// One function for both, so a record a live commit refuses is one replay
/// refuses too, and a history cannot hold an admission the runtime would not
/// have written.
pub(super) fn membership_refusal(
    members: &BTreeMap<NodeId, Member>,
    event: &LedgerEvent,
) -> Option<ExecutionError> {
    match event {
        LedgerEvent::PeerAdmitted {
            peer,
            principal,
            grant_set,
        } => {
            if ![peer.0.as_str(), principal, grant_set]
                .into_iter()
                .all(valid_member_field)
            {
                return Some(ExecutionError::InvalidSession);
            }
            if members.contains_key(peer) {
                return Some(ExecutionError::DuplicatePeerEntry { peer: peer.clone() });
            }
            None
        }
        LedgerEvent::PeerWithdrawn { peer } if !members.contains_key(peer) => {
            Some(ExecutionError::UnknownPeer { peer: peer.clone() })
        }
        _ => None,
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
    /// The peer is already a member. Re-admitting under another principal or
    /// grant set is a withdrawal and an admission, never a silent replacement.
    DuplicatePeerEntry {
        peer: NodeId,
    },
    /// A membership names a grant set the host has not defined. Refused rather
    /// than granted anything by default: membership is remembered, authority is
    /// supplied by the host at every start.
    UnknownGrantSet {
        name: String,
    },
    /// A second grant set under a name already defined.
    DuplicateGrantSet {
        name: String,
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
    /// Set when the session came from a peer's membership: the peer, and the
    /// commit index of the admission it was issued under. Such a session is valid
    /// only while that **same** admission stands — a withdrawal ends it, and a
    /// later re-admission is a different record and does not revive it.
    peer: Option<(NodeId, CommitIndex)>,
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
    /// Admitted peers, rebuilt from committed history on every open and from
    /// the projection a compacted snapshot carries.
    members: BTreeMap<NodeId, Member>,
    /// The host's grant sets. Not history: a runtime rebuilt from its log has
    /// none until the host installs them, and refuses every peer until then.
    grant_sets: GrantSets,
    /// Whether the reconciliation authority has been handed out.
    authority_taken: bool,
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
            members: BTreeMap::new(),
            grant_sets: GrantSets::new(),
            authority_taken: false,
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

    /// Whether the key settled with an outcome that a later request under it is
    /// answered from: the effect applied, with or without a retained response.
    /// A key reconciled as not applied binds nothing, so a new attempt under it
    /// is a new attempt and not a replay.
    pub(super) fn has_replayable_settlement(&self, key: &str) -> bool {
        matches!(
            self.settled.get(key),
            Some(SettledOutcome::Applied { .. } | SettledOutcome::AppliedWithoutResponse { .. })
        )
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
        if let Some((peer, admitted_at)) = &record.peer {
            let Some(member) = self
                .members
                .get(peer)
                .filter(|member| member.admitted_at == *admitted_at)
            else {
                return Err(ExecutionError::PeerNotAdmitted { peer: peer.clone() });
            };
            if !self.grant_sets.defines(&member.grant_set) {
                return Err(ExecutionError::UnknownGrantSet {
                    name: member.grant_set.clone(),
                });
            }
        }
        Ok(record)
    }

    /// The admitted peers, for the projection check a compacted restore makes.
    pub(super) fn members(&self) -> &BTreeMap<NodeId, Member> {
        &self.members
    }

    /// Reinstate the membership a compacted snapshot's projection carries.
    pub(super) fn restore_members(&mut self, members: BTreeMap<NodeId, Member>) {
        self.members = members;
    }

    /// Apply a committed membership record. Validated before it was appended
    /// ([`membership_refusal`]), so this only records it.
    pub(super) fn apply_membership(&mut self, index: CommitIndex, event: &LedgerEvent) {
        match event {
            LedgerEvent::PeerAdmitted {
                peer,
                principal,
                grant_set,
            } => {
                self.members.insert(
                    peer.clone(),
                    Member {
                        principal: principal.clone(),
                        grant_set: grant_set.clone(),
                        admitted_at: index,
                    },
                );
            }
            LedgerEvent::PeerWithdrawn { peer } => {
                self.members.remove(peer);
            }
            _ => {}
        }
    }

    /// Re-derive the grants of every live peer session from the grant sets now
    /// in force.
    ///
    /// A session whose admission no longer stands, or whose grant set is not
    /// defined, is left in place for [`ExecutionState::session`] to refuse by
    /// name. The expiry is deliberately not refreshed: a change may narrow what a
    /// session can do, but it must not extend how long it lasts. The principal
    /// is not re-derived because it cannot change: it belongs to the admission
    /// the session is bound to, and a different principal is a different one.
    fn rederive_sessions(&mut self) {
        for record in self.sessions.values_mut() {
            let Some((peer, admitted_at)) = &record.peer else {
                continue;
            };
            let Some(member) = self
                .members
                .get(peer)
                .filter(|member| member.admitted_at == *admitted_at)
            else {
                continue;
            };
            let Some(set) = self.grant_sets.sets.get(&member.grant_set) else {
                continue;
            };
            record.grants = (set.grants)().into_iter().map(Arc::new).collect();
        }
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
        peer: Option<(NodeId, CommitIndex)>,
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

    /// Install the host's grant sets.
    ///
    /// Replacing them re-derives every peer session's grants, for the reason
    /// authority is re-derived at use: a replacement that narrowed a grant set
    /// would otherwise leave a live session running under the wider one until
    /// its TTL expired. A session whose grant set the new table does not define
    /// is kept, so the refusal can name [`ExecutionError::UnknownGrantSet`].
    pub fn install_grant_sets(&mut self, sets: GrantSets) {
        self.execution.grant_sets = sets;
        self.execution.rederive_sessions();
        // Permits already issued were scoped by the grants that have just changed.
        self.execution.invalidate_pending();
    }

    /// Admit a peer as a member: committed to the ledger, so it survives a
    /// restart and a replayed history says who was admitted when (D5 in issue
    /// #23).
    ///
    /// `grant_set` must be defined by the grant sets installed now, which catches
    /// a name the host mistyped while the host is there to see it. That is a
    /// check at admission and not a promise: what the name grants is resolved at
    /// every use, and a later start that does not define it refuses the peer.
    ///
    /// A peer that is already a member is refused rather than re-admitted under a
    /// new principal or grant set: changing either is a withdrawal followed by an
    /// admission, two records a history can show, never one that silently
    /// replaces the other.
    pub fn admit_member(
        &mut self,
        peer: NodeId,
        principal: impl Into<String>,
        grant_set: impl Into<String>,
    ) -> Result<CommitIndex, ExecutionError> {
        let grant_set = grant_set.into();
        let event = LedgerEvent::PeerAdmitted {
            peer,
            principal: principal.into(),
            grant_set: grant_set.clone(),
        };
        if let Some(refusal) = membership_refusal(&self.execution.members, &event) {
            return Err(refusal);
        }
        if !self.execution.grant_sets.defines(&grant_set) {
            return Err(ExecutionError::UnknownGrantSet { name: grant_set });
        }
        self.commit(event).map_err(audit)
    }

    /// Whether committed history admits this peer now.
    pub fn is_member(&self, peer: &NodeId) -> bool {
        self.execution.members.contains_key(peer)
    }

    /// Admit a peer the host's transport authenticated.
    ///
    /// This layer admits an identity; it does not establish one. The identity is an
    /// [`AuthenticatedPeer`], which only `ptr-net`'s transport can construct — from
    /// the QUIC-authenticated `Connection::remote_id` — so a value read out of a
    /// request cannot reach this call: it does not compile (D1 in issue #23). The
    /// principal comes from committed membership and the grants from the host's
    /// grant set of the name that membership records, so there is no parameter
    /// through which a caller could name either.
    ///
    /// ```compile_fail
    /// # fn forge(runtime: &mut ptr_runtime::PtrRuntime) {
    /// // A bare id is not an authenticated peer.
    /// let _ = runtime.admit_peer(&ptr_types::NodeId::from("claimed-in-a-payload"));
    /// # }
    /// ```
    pub fn admit_peer(
        &mut self,
        peer: &AuthenticatedPeer,
    ) -> Result<ExecutionSession, ExecutionError> {
        let peer = peer.id();
        let Some(member) = self.execution.members.get(peer) else {
            return Err(ExecutionError::UnknownPeer { peer: peer.clone() });
        };
        let Some(set) = self.execution.grant_sets.sets.get(&member.grant_set) else {
            return Err(ExecutionError::UnknownGrantSet {
                name: member.grant_set.clone(),
            });
        };
        let principal = member.principal.clone();
        let admitted_at = member.admitted_at;
        let ttl = set.ttl;
        let grants = (set.grants)();
        self.register_session(principal, grants, ttl, Some((peer.clone(), admitted_at)))
    }

    /// Withdraw a peer: committed, so a peer withdrawn before a restart is still
    /// refused after it. Sessions admitted under it stop working immediately.
    pub fn withdraw_peer(&mut self, peer: &NodeId) -> Result<CommitIndex, ExecutionError> {
        let event = LedgerEvent::PeerWithdrawn { peer: peer.clone() };
        if let Some(refusal) = membership_refusal(&self.execution.members, &event) {
            return Err(refusal);
        }
        self.commit(event).map_err(audit)
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
        if !valid_identifier(&key)
            || (key.len() > MAX_KEY_BYTES && !self.execution.has_settled(&key))
        {
            return Err(ExecutionError::InvalidKey);
        }
        self.prepare(session, project, action, ttl, Some(key))
    }

    /// Whether a new request under `key` can be expected to get through
    /// [`prepare_execution_once`](Self::prepare_execution_once) and be committed:
    /// a well-formed identifier no longer than [`MAX_KEY_BYTES`], or a longer one
    /// that a log written by an earlier build settled with an outcome a retry is
    /// answered from. This is stricter than preparation in one case. A longer key
    /// that was only reconciled as not applied passes preparation, because the
    /// runtime holds it, but a new attempt under it is refused at commit, so a
    /// caller that wants to refuse before it does any work should ask here.
    /// Nothing is reserved or recorded.
    pub fn is_usable_execution_key(&self, key: &str) -> bool {
        valid_identifier(key)
            && (key.len() <= MAX_KEY_BYTES || self.execution.has_replayable_settlement(key))
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

    /// Take this runtime's reconciliation authority (D6 in issue #23).
    ///
    /// `Some` exactly once per runtime, `None` after. The host that opens a runtime
    /// takes it before handing the runtime to anything else — a wire host, a
    /// scheduler — so code that is given a runtime is not thereby given the power
    /// to declare what happened to an effect.
    pub fn take_reconciliation_authority(&mut self) -> Option<ReconciliationAuthority> {
        if self.execution.authority_taken {
            return None;
        }
        self.execution.authority_taken = true;
        Some(ReconciliationAuthority {
            issuer: self.execution.issuer.clone(),
        })
    }

    /// Resolve an attempt this runtime could not observe, with evidence from the
    /// system that received the effect.
    ///
    /// It records what an operator established. It never infers the outcome and
    /// never retries the effect: a runtime that could work out what happened
    /// would not have been fenced.
    ///
    /// Only the holder of this runtime's [`ReconciliationAuthority`] may call it,
    /// and it never crosses a wire: a remote peer must not be able to declare
    /// what happened to an effect. An authority taken from another runtime is
    /// refused with [`ExecutionError::ForeignRuntime`].
    ///
    /// A detached attempt it closes is no longer outstanding, as after
    /// [`PtrRuntime::settle_detached`], so an adapter answer that arrives later is
    /// refused with [`ExecutionError::NotDetached`].
    pub fn reconcile_effect(
        &mut self,
        authority: &ReconciliationAuthority,
        attempt: CommitIndex,
        applied: bool,
        evidence: impl Into<String>,
    ) -> Result<CommitIndex, ExecutionError> {
        if !Arc::ptr_eq(&authority.issuer, &self.execution.issuer) {
            return Err(ExecutionError::ForeignRuntime);
        }
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
    fn only_an_applied_settlement_can_be_replayed_under_a_key() {
        let mut state = ExecutionState::default();
        state
            .settled
            .insert("not-applied".into(), SettledOutcome::NotApplied);
        state.settled.insert(
            "applied-unretained".into(),
            SettledOutcome::AppliedWithoutResponse {
                attempt: CommitIndex(1),
                identity: ActionIdentity {
                    project: ProjectId::from("project"),
                    principal: "principal".into(),
                    revision: Revision(1),
                    generation: Generation(1),
                    action_digest: [1; 32],
                },
            },
        );
        assert!(!state.has_replayable_settlement("not-applied"));
        assert!(state.has_replayable_settlement("applied-unretained"));
        assert!(!state.has_replayable_settlement("never-seen"));
    }

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
