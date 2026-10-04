//! Which project's Pods a peer may reach, and which exact capabilities there,
//! as committed history says (the Pod-access follow-up to D5 in issue #23).
//!
//! **Why this lives in the runtime.** `ptr-podwire` decided Pod access from a
//! table the host installed in memory, while execution membership had become
//! committed history. That was two policies with two durability stories: a
//! peer whose execution membership was withdrawn stayed withdrawn across a
//! restart, but a peer whose Pod scope was withdrawn was readmitted by whatever
//! the host installed next. The runtime owns the journal, so it owns this too:
//! [`PtrRuntime::admit_pod_scope`] and [`PtrRuntime::withdraw_pod_scope`] commit
//! records, every open replays them in order, a compacted snapshot carries them
//! across its floor, and [`PtrRuntime::pod_scope`] is what the access decision
//! reads.
//!
//! **Journaled whole.** An execution grant owns a verifier and an executor,
//! which are behaviour, so a membership names its grant set and the host
//! supplies it at every start. A [`PodScope`] is a project and a set of exact
//! `(capability, payload type)` pairs — only data — so the record carries all
//! of it and there is nothing left for a host to supply or to get wrong after
//! a restart.
//!
//! **What stays declared.** Which Pods exist and which project each belongs to
//! is the host's `PodRegistry`, registered at start as before. A scope says
//! what a peer may ask for; the registry still says what there is to answer it.
use super::{PtrRuntime, RuntimeError};
use ptr_ledger::{LedgerEvent, MAX_POD_SCOPE_ENTRIES, MAX_POD_SCOPE_FIELD_BYTES};
use ptr_types::{CapabilityId, CommitIndex, NodeId, ProjectId, TypeId};
use std::collections::{BTreeMap, BTreeSet};

/// The projection of the largest scope a record may carry fits the string
/// bound of a compacted snapshot: `admitted`, a line feed and the project, then
/// a line feed, a capability, a tab and a payload type for every pair. If a
/// bound in `ptr-ledger` grows past this, the build stops here instead of a
/// committed scope later failing every export.
const _: () = assert!(
    "admitted\n".len()
        + MAX_POD_SCOPE_FIELD_BYTES
        + MAX_POD_SCOPE_ENTRIES * (1 + MAX_POD_SCOPE_FIELD_BYTES + 1 + MAX_POD_SCOPE_FIELD_BYTES)
        <= super::compacted::MAX_STRING_BYTES
);

/// What one peer may reach: one project, and an exact set of `(capability,
/// payload type)` pairs inside it.
///
/// Exact, never a prefix or a wildcard, for the reason `ActionScope` gives:
/// a scope that matched by prefix would grow every time somebody registered a
/// Pod with a longer name. The pair is the pair `PodRegistry::resolve` keys on,
/// so a scope cannot be expressed that resolution would not honour.
///
/// A value, not an authority: building one grants nothing. It reaches the
/// access decision only through [`PtrRuntime::admit_pod_scope`], which commits
/// it, and [`PtrRuntime::pod_scope`], which reads what was committed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PodScope {
    project: ProjectId,
    allow: BTreeSet<(CapabilityId, TypeId)>,
}

impl PodScope {
    /// A scope over one project that allows nothing yet.
    pub fn new(project: ProjectId) -> Self {
        Self {
            project,
            allow: BTreeSet::new(),
        }
    }

    /// Allow one exact capability and payload type.
    pub fn allow(mut self, capability: CapabilityId, input_type: TypeId) -> Self {
        self.allow.insert((capability, input_type));
        self
    }

    /// The project this scope is confined to.
    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    /// Whether this exact pair is allowed.
    pub fn covers(&self, capability: &CapabilityId, input_type: &TypeId) -> bool {
        self.allow
            .contains(&(capability.clone(), input_type.clone()))
    }

    /// Every allowed pair, ascending: the canonical order the record carries,
    /// which a set yields by construction, so two equal scopes always commit
    /// the same bytes.
    pub fn entries(&self) -> impl Iterator<Item = &(CapabilityId, TypeId)> {
        self.allow.iter()
    }

    /// The scope a canonical record or projection describes. The caller has
    /// already held it to `ptr_ledger::check_pod_scope`.
    fn from_canonical(project: ProjectId, entries: &[(CapabilityId, TypeId)]) -> Self {
        Self {
            project,
            allow: entries.iter().cloned().collect(),
        }
    }
}

/// The Pod scopes committed history admits, keyed by peer.
pub(super) type PodScopes = BTreeMap<NodeId, PodScope>;

/// Why a Pod-scope record cannot be committed or replayed, or `None` if it can.
///
/// One function for both, as [`super::execution`]'s membership rule is, so a
/// record a live commit refuses is one replay refuses too and a history cannot
/// hold a scope the runtime would not have written:
///
/// - a record that is not canonical by the codec's own rule
///   (`ptr_ledger::check_pod_scope_record`, whose code the reason carries);
/// - an admission of a peer that already has a scope — narrowing or widening
///   one is a withdrawal and an admission, two records a history can show,
///   never one that silently replaces the other;
/// - a withdrawal of a peer that has none.
pub(super) fn pod_scope_refusal(scopes: &PodScopes, event: &LedgerEvent) -> Option<RuntimeError> {
    let refused = |peer: &NodeId, reason: &'static str| {
        Some(RuntimeError::InvalidPodScope {
            peer: peer.clone(),
            reason,
        })
    };
    let (LedgerEvent::PodScopeAdmitted { peer, .. } | LedgerEvent::PodScopeWithdrawn { peer }) =
        event
    else {
        return None;
    };
    if let Err(shape) = ptr_ledger::check_pod_scope_record(event) {
        return refused(peer, shape.code());
    }
    match (event, scopes.contains_key(peer)) {
        (LedgerEvent::PodScopeAdmitted { .. }, true) => {
            refused(peer, "the peer already has a Pod scope")
        }
        (LedgerEvent::PodScopeWithdrawn { .. }, false) => {
            refused(peer, "the peer has no Pod scope")
        }
        _ => None,
    }
}

/// Apply a committed Pod-scope record. Validated before it was appended
/// ([`pod_scope_refusal`]), so this only records it.
pub(super) fn apply_pod_scope(scopes: &mut PodScopes, event: &LedgerEvent) {
    match event {
        LedgerEvent::PodScopeAdmitted {
            peer,
            project,
            entries,
        } => {
            scopes.insert(
                peer.clone(),
                PodScope::from_canonical(project.clone(), entries),
            );
        }
        LedgerEvent::PodScopeWithdrawn { peer } => {
            scopes.remove(peer);
        }
        _ => {}
    }
}

/// Whether a value under [`ptr_state::POD_SCOPE_PREFIX`] is exactly one a
/// committed record projects, for the compacted lifecycle section's check.
pub(super) fn canonical_pod_scope_projection(key: &str, value: &str) -> bool {
    ptr_state::pod_scope_of(key).is_some_and(ptr_ledger::valid_pod_scope_field)
        && ptr_state::parse_pod_scope_entry(value).is_some()
}

/// The Pod scopes a restored projection describes: every admitted one, and
/// none of the withdrawn. Unlike a membership, a scope carries no commit index:
/// nothing is bound to the record that admitted it, because the access decision
/// reads the scope afresh at every request.
pub(super) fn pod_scopes_from_projection(materialized: &BTreeMap<String, String>) -> PodScopes {
    materialized
        .range(ptr_state::POD_SCOPE_PREFIX.to_owned()..)
        .take_while(|(key, _)| key.starts_with(ptr_state::POD_SCOPE_PREFIX))
        .filter_map(|(key, value)| {
            let peer = ptr_state::pod_scope_of(key)?;
            let (project, entries) = ptr_state::parse_pod_scope_entry(value)??;
            Some((
                NodeId::from(peer),
                PodScope::from_canonical(project, &entries),
            ))
        })
        .collect()
}

impl PtrRuntime {
    /// Bind a peer to one Pod scope: committed to the ledger, so it survives a
    /// restart and a replayed history says which peer could reach what, and
    /// when.
    ///
    /// The peer is a [`NodeId`] rather than an `AuthenticatedPeer` because
    /// admitting is the host's decision about a key, made before that key ever
    /// connects; what must be authenticated is the connection that later asks
    /// under it, and the access decision takes only that.
    ///
    /// Refused with [`RuntimeError::InvalidPodScope`], before anything is
    /// appended: a scope the codec would not encode (an empty or oversized field,
    /// more than [`MAX_POD_SCOPE_ENTRIES`] pairs), or a peer that already has a
    /// scope. Changing a scope is [`Self::withdraw_pod_scope`] followed by this,
    /// so the history shows both.
    pub fn admit_pod_scope(
        &mut self,
        peer: NodeId,
        scope: &PodScope,
    ) -> Result<CommitIndex, RuntimeError> {
        self.commit(LedgerEvent::PodScopeAdmitted {
            peer,
            project: scope.project.clone(),
            entries: scope.allow.iter().cloned().collect(),
        })
    }

    /// Withdraw a peer's Pod scope: committed, so a scope withdrawn before a
    /// restart is still withdrawn after it. The next request the peer makes is
    /// refused, because the access decision reads the scope at every request
    /// rather than remembering it.
    pub fn withdraw_pod_scope(&mut self, peer: &NodeId) -> Result<CommitIndex, RuntimeError> {
        self.commit(LedgerEvent::PodScopeWithdrawn { peer: peer.clone() })
    }

    /// The Pod scope committed history binds this peer to now, if any.
    pub fn pod_scope(&self, peer: &NodeId) -> Option<&PodScope> {
        self.pod_scopes.get(peer)
    }
}
