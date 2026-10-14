use ptr_types::{
    ProjectId, ScopeId, ScopeLeaseBinding, ScopeLifecycleEvent, ScopeLifecycleKind, SessionId,
    Timestamp,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScopeState {
    Created,
    Admitted,
    Started,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    Revoked,
    Released,
}

impl ScopeState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Failed
                | Self::Cancelled
                | Self::TimedOut
                | Self::Revoked
                | Self::Released
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionScope {
    pub id: ScopeId,
    pub parent: Option<ScopeId>,
    pub session: SessionId,
    pub project: ProjectId,
    pub created_at: Timestamp,
    pub deadline: Option<Timestamp>,
    pub state: ScopeState,
    pub cancellation_requested: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScopeEventKind {
    Created,
    Admitted,
    Started,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
    Revoked,
    Released,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScopeEvent {
    pub sequence: u64,
    pub scope: ScopeId,
    pub kind: ScopeEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScopeError {
    Duplicate(ScopeId),
    Unknown(ScopeId),
    ParentMissing(ScopeId),
    ParentScopeMismatch,
    InvalidTransition {
        from: ScopeState,
        to: ScopeState,
    },
    Terminal(ScopeState),
    RecoveryRequired(ScopeId),
    InvalidCommittedEvent,
    InvalidLeaseBinding,
    InvalidRevision,
    /// Creating this scope would take the registry past what a compacted snapshot can carry
    /// ([`MAX_SCOPES`] scopes or [`MAX_SCOPE_SNAPSHOT_BYTES`] encoded bytes). Nothing was
    /// committed. Every scope ever created is kept (its id cannot be reused and a released one
    /// can still be named), so the bound is a refusal at creation, never a failing export.
    CapacityExceeded {
        scopes: usize,
        bytes: usize,
    },
}

/// Most scopes a runtime's history may create (released ones included). Below the item bound
/// of a compacted section (65,536), so the scope block can always be counted.
pub const MAX_SCOPES: usize = 16_384;
/// Most bytes the scope block of a compacted snapshot may take: half of one section
/// ([`MAX_SECTION_BYTES`](crate::compacted::MAX_SECTION_BYTES)), the rest staying for the
/// execution obligations it shares the section with.
pub const MAX_SCOPE_SNAPSHOT_BYTES: usize = crate::compacted::MAX_SECTION_BYTES / 2;
/// The bytes of a scope block around its entries: magic, count and the optional revision.
pub(crate) const SCOPE_BLOCK_OVERHEAD: usize = 8 + 4 + 8 + 8;

/// The exact number of bytes one scope takes in a compacted scope block
/// (`compacted::encode_scope_block`): length-prefixed strings, an 8-byte presence flag per
/// optional field, 8-byte numbers and a 16-byte fencing token.
pub(crate) fn compacted_scope_entry_bytes(
    id: &ScopeId,
    parent: Option<&ScopeId>,
    session: &SessionId,
    project: &ProjectId,
    deadline: Option<Timestamp>,
    lease: &ScopeLeaseBinding,
) -> usize {
    let field = |text: &str| 4 + text.len();
    let optional = |inner: Option<usize>| 8 + inner.unwrap_or(0);
    field(&id.0)
        + optional(parent.map(|p| field(&p.0)))
        + field(&session.0)
        + field(&project.0)
        + 8 // created_at
        + optional(deadline.map(|_| 8))
        + 8 // state
        + 8 // cancellation flag
        + optional(lease.state_id.as_ref().map(|v| field(&v.0)))
        + optional(lease.pod_id.as_ref().map(|v| field(&v.0)))
        + optional(lease.placement_epoch.map(|_| 8))
        + optional(lease.fencing_token.map(|_| 16))
        + optional(lease.generation.map(|_| 8))
        + optional(lease.resource_lease_id.as_ref().map(|v| field(v)))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CleanupError {
    pub step: &'static str,
    pub message: String,
}

pub trait ScopeCleanupCoordinator {
    fn stop_intake(&mut self, scope: &ExecutionScope) -> Result<(), CleanupError>;
    fn cancel_children(&mut self, scope: &ExecutionScope) -> Result<(), CleanupError>;
    fn drain_queues(&mut self, scope: &ExecutionScope) -> Result<(), CleanupError>;
    fn release_pod_lease(&mut self, scope: &ExecutionScope) -> Result<(), CleanupError>;
    fn release_resource_lease(&mut self, scope: &ExecutionScope) -> Result<(), CleanupError>;
    fn close_session(&mut self, scope: &ExecutionScope) -> Result<(), CleanupError>;
}

/// The scopes a compacted snapshot carries, by id, each with its lease binding.
pub(crate) type CompactedScopes = BTreeMap<ScopeId, (ExecutionScope, ScopeLeaseBinding)>;

#[derive(Default)]
pub struct ScopeRegistry {
    scopes: BTreeMap<ScopeId, ExecutionScope>,
    events: Vec<ScopeEvent>,
    committed: Vec<ScopeLifecycleEvent>,
    recovery_required: BTreeSet<ScopeId>,
    leases: BTreeMap<ScopeId, ScopeLeaseBinding>,
    last_revision: Option<ptr_types::Revision>,
    /// Bytes the scopes take in a compacted scope block, overhead included once a scope exists.
    snapshot_bytes: usize,
}

impl ScopeRegistry {
    pub fn validate_lease_shape(lease: &ScopeLeaseBinding) -> Result<(), ScopeError> {
        if lease.placement_epoch.is_some() != lease.fencing_token.is_some()
            || lease.fencing_token.is_some()
                && (lease.state_id.is_none()
                    || lease.pod_id.is_none()
                    || lease.generation.is_none())
        {
            return Err(ScopeError::InvalidLeaseBinding);
        }
        Ok(())
    }

    pub fn create(&mut self, scope: ExecutionScope) -> Result<(), ScopeError> {
        if self.scopes.contains_key(&scope.id) {
            return Err(ScopeError::Duplicate(scope.id));
        }
        if let Some(parent_id) = &scope.parent {
            let parent = self
                .scopes
                .get(parent_id)
                .ok_or_else(|| ScopeError::ParentMissing(parent_id.clone()))?;
            if parent.session != scope.session || parent.project != scope.project {
                return Err(ScopeError::ParentScopeMismatch);
            }
        }
        let id = scope.id.clone();
        let lease = ScopeLeaseBinding::default();
        self.account(&scope, &lease);
        self.leases.insert(id.clone(), lease);
        self.scopes.insert(id.clone(), scope);
        self.record(id, ScopeEventKind::Created);
        Ok(())
    }

    pub fn get(&self, id: &ScopeId) -> Result<&ExecutionScope, ScopeError> {
        self.scopes
            .get(id)
            .ok_or_else(|| ScopeError::Unknown(id.clone()))
    }

    pub fn events(&self) -> &[ScopeEvent] {
        &self.events
    }

    pub fn recovery_required(&self, id: &ScopeId) -> bool {
        self.recovery_required.contains(id)
    }

    pub fn recoverable_scopes(&self) -> Vec<ScopeId> {
        self.recovery_required.iter().cloned().collect()
    }

    /// Marks every scope whose cleanup has not completed. A scope that ended
    /// (completed, failed, cancelled, timed out, revoked) but was never released
    /// still holds its pod lease, resource lease and session, so it is pending
    /// recovery exactly like a scope that was still running.
    pub fn mark_recovery_required(&mut self) {
        self.recovery_required = self.unreleased();
    }

    fn unreleased(&self) -> BTreeSet<ScopeId> {
        self.scopes
            .values()
            .filter(|scope| scope.state != ScopeState::Released)
            .map(|scope| scope.id.clone())
            .collect()
    }

    /// Fences one live scope after an external request outcome became
    /// uncertain. The scope remains journal-authoritative; this only records
    /// that normal work is no longer allowed until the runtime performs
    /// recovery cleanup.
    pub fn mark_scope_recovery_required(&mut self, id: &ScopeId) -> Result<(), ScopeError> {
        let scope = self.get(id)?;
        if scope.state.is_terminal() {
            return Err(ScopeError::Terminal(scope.state));
        }
        self.recovery_required.insert(id.clone());
        Ok(())
    }

    pub fn children(&self, id: &ScopeId) -> Vec<ScopeId> {
        self.scopes
            .values()
            .filter(|scope| scope.parent.as_ref() == Some(id))
            .map(|scope| scope.id.clone())
            .collect()
    }

    pub fn ensure_available(&self, id: &ScopeId) -> Result<(), ScopeError> {
        let scope = self.get(id)?;
        // An ended scope is refused as ended even while its cleanup is pending.
        if scope.state.is_terminal() {
            return Err(ScopeError::Terminal(scope.state));
        }
        if self.recovery_required(id) {
            return Err(ScopeError::RecoveryRequired(id.clone()));
        }
        Ok(())
    }

    pub fn lease(&self, id: &ScopeId) -> Result<&ScopeLeaseBinding, ScopeError> {
        self.leases
            .get(id)
            .ok_or_else(|| ScopeError::Unknown(id.clone()))
    }

    /// Materialize an authoritative ledger event. Applying an identical event
    /// twice is safe; a conflicting event is rejected before state changes.
    pub fn apply_committed(&mut self, event: ScopeLifecycleEvent) -> Result<(), ScopeError> {
        if self.committed.contains(&event) {
            return Ok(());
        }
        self.validate_committed(&event)?;
        let next = state_from_kind(event.kind);
        if next == ScopeState::Created {
            if self.scopes.contains_key(&event.scope_id) {
                return Err(ScopeError::Duplicate(event.scope_id));
            }
            let scope = ExecutionScope {
                id: event.scope_id.clone(),
                parent: event.parent_id.clone(),
                session: event.session_id.clone(),
                project: event.project_id.clone(),
                created_at: event.created_at,
                deadline: event.deadline,
                state: ScopeState::Created,
                cancellation_requested: false,
            };
            if let Some(parent_id) = &scope.parent {
                let parent = self
                    .scopes
                    .get(parent_id)
                    .ok_or_else(|| ScopeError::ParentMissing(parent_id.clone()))?;
                if parent.session != scope.session || parent.project != scope.project {
                    return Err(ScopeError::ParentScopeMismatch);
                }
            }
            self.account(&scope, &event.lease);
            self.scopes.insert(scope.id.clone(), scope);
            self.leases
                .insert(event.scope_id.clone(), event.lease.clone());
        } else {
            if self
                .leases
                .get(&event.scope_id)
                .is_some_and(|lease| lease != &event.lease)
            {
                return Err(ScopeError::InvalidLeaseBinding);
            }
            let current = self
                .scopes
                .get(&event.scope_id)
                .ok_or_else(|| ScopeError::Unknown(event.scope_id.clone()))?
                .state;
            if event.previous_kind.map(state_from_kind) != Some(current) {
                return Err(ScopeError::InvalidCommittedEvent);
            }
            if current.is_terminal() && next != ScopeState::Released {
                return Err(ScopeError::Terminal(current));
            }
            let valid = valid_transition(current, next);
            if !valid {
                return Err(ScopeError::InvalidTransition {
                    from: current,
                    to: next,
                });
            }
            let scope = self
                .scopes
                .get_mut(&event.scope_id)
                .expect("scope checked above");
            scope.state = next;
            if matches!(
                next,
                ScopeState::Cancelled | ScopeState::TimedOut | ScopeState::Revoked
            ) {
                scope.cancellation_requested = true;
            }
        }
        self.record(event.scope_id.clone(), event_kind(next));
        self.committed.push(event);
        self.last_revision = self.committed.last().map(|event| event.revision);
        // Only a release ends a pending recovery: a scope that ended any other way still holds
        // its leases and session (see `mark_recovery_required`), and `recover_scope` must still
        // find it.
        if next == ScopeState::Released {
            self.recovery_required
                .remove(&self.committed.last().unwrap().scope_id);
        }
        Ok(())
    }

    pub fn validate_committed(&self, event: &ScopeLifecycleEvent) -> Result<(), ScopeError> {
        Self::validate_lease_shape(&event.lease)?;
        let next = state_from_kind(event.kind);
        if next == ScopeState::Created {
            if self.scopes.contains_key(&event.scope_id) {
                return Err(ScopeError::Duplicate(event.scope_id.clone()));
            }
            if event.previous_kind.is_some() {
                return Err(ScopeError::InvalidCommittedEvent);
            }
            if self
                .last_revision
                .is_some_and(|revision| event.revision < revision)
            {
                return Err(ScopeError::InvalidRevision);
            }
            if let Some(parent_id) = &event.parent_id {
                let parent = self
                    .scopes
                    .get(parent_id)
                    .ok_or_else(|| ScopeError::ParentMissing(parent_id.clone()))?;
                if parent.session != event.session_id || parent.project != event.project_id {
                    return Err(ScopeError::ParentScopeMismatch);
                }
            }
            return Ok(());
        }
        let current = self
            .scopes
            .get(&event.scope_id)
            .ok_or_else(|| ScopeError::Unknown(event.scope_id.clone()))?
            .state;
        let scope = self
            .scopes
            .get(&event.scope_id)
            .expect("scope checked above");
        if scope.parent != event.parent_id
            || scope.session != event.session_id
            || scope.project != event.project_id
            || scope.created_at != event.created_at
            || scope.deadline != event.deadline
        {
            return Err(ScopeError::InvalidCommittedEvent);
        }
        if self
            .last_revision
            .is_some_and(|revision| event.revision < revision)
        {
            return Err(ScopeError::InvalidRevision);
        }
        if event.previous_kind.map(state_from_kind) != Some(current) {
            return Err(ScopeError::InvalidCommittedEvent);
        }
        if current.is_terminal() && next != ScopeState::Released {
            return Err(ScopeError::Terminal(current));
        }
        if !valid_transition(current, next) {
            return Err(ScopeError::InvalidTransition {
                from: current,
                to: next,
            });
        }
        if self
            .leases
            .get(&event.scope_id)
            .is_some_and(|lease| lease != &event.lease)
        {
            return Err(ScopeError::InvalidLeaseBinding);
        }
        Ok(())
    }

    pub fn replay<I>(&mut self, events: I) -> Result<(), ScopeError>
    where
        I: IntoIterator<Item = ScopeLifecycleEvent>,
    {
        for event in events {
            self.apply_committed(event)?;
        }
        self.mark_recovery_required();
        Ok(())
    }

    /// Whether a scope created by `event` still fits the bounds a compacted snapshot can carry
    /// ([`MAX_SCOPES`], [`MAX_SCOPE_SNAPSHOT_BYTES`]). Checked before a new record is committed,
    /// never on replay: a history written before the bound existed still replays.
    pub fn check_capacity(&self, event: &ScopeLifecycleEvent) -> Result<(), ScopeError> {
        if state_from_kind(event.kind) != ScopeState::Created {
            return Ok(());
        }
        let bytes = self.snapshot_bytes.max(SCOPE_BLOCK_OVERHEAD)
            + compacted_scope_entry_bytes(
                &event.scope_id,
                event.parent_id.as_ref(),
                &event.session_id,
                &event.project_id,
                event.deadline,
                &event.lease,
            );
        let scopes = self.scopes.len() + 1;
        if scopes > MAX_SCOPES || bytes > MAX_SCOPE_SNAPSHOT_BYTES {
            return Err(ScopeError::CapacityExceeded { scopes, bytes });
        }
        Ok(())
    }

    /// Bytes the scope block of a compacted snapshot of this registry takes (0 without a scope).
    pub fn snapshot_bytes(&self) -> usize {
        self.snapshot_bytes
    }

    fn account(&mut self, scope: &ExecutionScope, lease: &ScopeLeaseBinding) {
        self.snapshot_bytes = self.snapshot_bytes.max(SCOPE_BLOCK_OVERHEAD)
            + compacted_scope_entry_bytes(
                &scope.id,
                scope.parent.as_ref(),
                &scope.session,
                &scope.project,
                scope.deadline,
                lease,
            );
    }

    /// Every scope the committed history holds, released ones included, with its lease binding,
    /// in id order; and the revision of the last applied scope record. This is what a compacted
    /// snapshot carries across its floor (issue #141): without it a restored runtime forgets
    /// the scopes created below the floor, and with them their recovery obligations.
    pub(crate) fn compacted_parts(&self) -> (CompactedScopes, Option<ptr_types::Revision>) {
        let scopes = self
            .scopes
            .iter()
            .map(|(id, scope)| {
                let lease = self.leases.get(id).cloned().unwrap_or_default();
                (id.clone(), (scope.clone(), lease))
            })
            .collect();
        (scopes, self.last_revision)
    }

    /// Rebuild a registry from [`Self::compacted_parts`]. Checks what is derivable and refuses
    /// the rest with [`ScopeError::InvalidCommittedEvent`]: every key is its scope's id, every
    /// lease has a valid shape, every parent is present, distinct, in the same session and
    /// project and reached without a cycle, a scope's cancellation flag agrees with its state
    /// (set for a cancelled, timed-out or revoked scope, clear for a created, admitted, started,
    /// completed or failed one; either for a released one), and a revision is present exactly
    /// when a scope is. Every unreleased scope is marked recovery-required, as after a replay.
    pub(crate) fn from_compacted_parts(
        scopes: CompactedScopes,
        last_revision: Option<ptr_types::Revision>,
    ) -> Result<Self, ScopeError> {
        let invalid = || ScopeError::InvalidCommittedEvent;
        if scopes.is_empty() != last_revision.is_none() {
            return Err(invalid());
        }
        for (id, (scope, lease)) in &scopes {
            if &scope.id != id {
                return Err(invalid());
            }
            Self::validate_lease_shape(lease)?;
            let flagged = match scope.state {
                ScopeState::Cancelled | ScopeState::TimedOut | ScopeState::Revoked => Some(true),
                ScopeState::Released => None,
                ScopeState::Created
                | ScopeState::Admitted
                | ScopeState::Started
                | ScopeState::Completed
                | ScopeState::Failed => Some(false),
            };
            if flagged.is_some_and(|expected| expected != scope.cancellation_requested) {
                return Err(invalid());
            }
            // Walk to the root: every ancestor exists, matches, and the walk ends.
            let mut child = scope;
            let mut steps = 0usize;
            while let Some(parent_id) = &child.parent {
                let (parent, _) = scopes.get(parent_id).ok_or_else(invalid)?;
                if parent.session != child.session || parent.project != child.project {
                    return Err(ScopeError::ParentScopeMismatch);
                }
                steps += 1;
                if parent_id == id || steps > scopes.len() {
                    return Err(invalid());
                }
                child = parent;
            }
        }
        let mut registry = Self {
            last_revision,
            ..Self::default()
        };
        for (id, (scope, lease)) in scopes {
            registry.account(&scope, &lease);
            registry.leases.insert(id.clone(), lease);
            registry.scopes.insert(id, scope);
        }
        registry.mark_recovery_required();
        Ok(registry)
    }

    pub fn transition(&mut self, id: &ScopeId, next: ScopeState) -> Result<(), ScopeError> {
        let current = self.get(id)?.state;
        if current == next && current == ScopeState::Released {
            return Ok(());
        }
        if current.is_terminal() && next != ScopeState::Released {
            return Err(ScopeError::Terminal(current));
        }
        let valid = matches!(
            (current, next),
            (ScopeState::Created, ScopeState::Admitted)
                | (ScopeState::Admitted, ScopeState::Started)
                | (ScopeState::Started, ScopeState::Completed)
                | (ScopeState::Started, ScopeState::Failed)
                | (ScopeState::Started, ScopeState::Cancelled)
                | (ScopeState::Started, ScopeState::TimedOut)
                | (ScopeState::Started, ScopeState::Revoked)
                | (ScopeState::Admitted, ScopeState::Revoked)
                | (ScopeState::Created, ScopeState::Revoked)
                | (ScopeState::Admitted, ScopeState::Cancelled)
                | (ScopeState::Created, ScopeState::Cancelled)
                | (ScopeState::Completed, ScopeState::Released)
                | (ScopeState::Failed, ScopeState::Released)
                | (ScopeState::Cancelled, ScopeState::Released)
                | (ScopeState::TimedOut, ScopeState::Released)
                | (ScopeState::Revoked, ScopeState::Released)
        );
        if !valid {
            return Err(ScopeError::InvalidTransition {
                from: current,
                to: next,
            });
        }
        let scope = self.scopes.get_mut(id).expect("scope checked above");
        scope.state = next;
        if matches!(
            next,
            ScopeState::Cancelled | ScopeState::TimedOut | ScopeState::Revoked
        ) {
            scope.cancellation_requested = true;
        }
        self.record(id.clone(), event_kind(next));
        Ok(())
    }

    pub fn cancel(&mut self, id: &ScopeId) -> Result<(), ScopeError> {
        self.transition(id, ScopeState::Cancelled)
    }

    pub fn cleanup(&mut self, id: &ScopeId) -> Result<(), ScopeError> {
        let children: Vec<ScopeId> = self
            .scopes
            .values()
            .filter(|scope| scope.parent.as_ref() == Some(id) && !scope.state.is_terminal())
            .map(|scope| scope.id.clone())
            .collect();
        for child in children {
            self.cancel(&child)?;
            self.cleanup(&child)?;
        }
        if self.get(id)?.state == ScopeState::Released {
            return Ok(());
        }
        if !self.get(id)?.state.is_terminal() {
            self.cancel(id)?;
        }
        self.transition(id, ScopeState::Released)
    }

    fn record(&mut self, scope: ScopeId, kind: ScopeEventKind) {
        self.events.push(ScopeEvent {
            sequence: self.events.len() as u64,
            scope,
            kind,
        });
    }
}

fn event_kind(state: ScopeState) -> ScopeEventKind {
    match state {
        ScopeState::Created => ScopeEventKind::Created,
        ScopeState::Admitted => ScopeEventKind::Admitted,
        ScopeState::Started => ScopeEventKind::Started,
        ScopeState::Completed => ScopeEventKind::Completed,
        ScopeState::Failed => ScopeEventKind::Failed,
        ScopeState::Cancelled => ScopeEventKind::Cancelled,
        ScopeState::TimedOut => ScopeEventKind::TimedOut,
        ScopeState::Revoked => ScopeEventKind::Revoked,
        ScopeState::Released => ScopeEventKind::Released,
    }
}

fn state_from_kind(kind: ScopeLifecycleKind) -> ScopeState {
    match kind {
        ScopeLifecycleKind::Created => ScopeState::Created,
        ScopeLifecycleKind::Admitted => ScopeState::Admitted,
        ScopeLifecycleKind::Started => ScopeState::Started,
        ScopeLifecycleKind::Completed => ScopeState::Completed,
        ScopeLifecycleKind::Failed => ScopeState::Failed,
        ScopeLifecycleKind::Cancelled => ScopeState::Cancelled,
        ScopeLifecycleKind::TimedOut => ScopeState::TimedOut,
        ScopeLifecycleKind::Revoked => ScopeState::Revoked,
        ScopeLifecycleKind::Released => ScopeState::Released,
    }
}

fn valid_transition(current: ScopeState, next: ScopeState) -> bool {
    matches!(
        (current, next),
        (ScopeState::Created, ScopeState::Admitted)
            | (ScopeState::Admitted, ScopeState::Started)
            | (ScopeState::Started, ScopeState::Completed)
            | (ScopeState::Started, ScopeState::Failed)
            | (ScopeState::Started, ScopeState::Cancelled)
            | (ScopeState::Started, ScopeState::TimedOut)
            | (ScopeState::Started, ScopeState::Revoked)
            | (ScopeState::Admitted, ScopeState::Revoked)
            | (ScopeState::Created, ScopeState::Revoked)
            | (ScopeState::Admitted, ScopeState::Cancelled)
            | (ScopeState::Created, ScopeState::Cancelled)
            | (ScopeState::Completed, ScopeState::Released)
            | (ScopeState::Failed, ScopeState::Released)
            | (ScopeState::Cancelled, ScopeState::Released)
            | (ScopeState::TimedOut, ScopeState::Released)
            | (ScopeState::Revoked, ScopeState::Released)
    )
}

pub fn lifecycle_event(
    scope: &ExecutionScope,
    previous: Option<ScopeState>,
    next: ScopeState,
    revision: ptr_types::Revision,
    lease: ScopeLeaseBinding,
    reason: Option<String>,
) -> ScopeLifecycleEvent {
    ScopeLifecycleEvent {
        scope_id: scope.id.clone(),
        parent_id: scope.parent.clone(),
        session_id: scope.session.clone(),
        project_id: scope.project.clone(),
        created_at: scope.created_at,
        deadline: scope.deadline,
        kind: state_to_kind(next),
        previous_kind: previous.map(state_to_kind),
        lease,
        revision,
        reason,
    }
}

fn state_to_kind(state: ScopeState) -> ScopeLifecycleKind {
    match state {
        ScopeState::Created => ScopeLifecycleKind::Created,
        ScopeState::Admitted => ScopeLifecycleKind::Admitted,
        ScopeState::Started => ScopeLifecycleKind::Started,
        ScopeState::Completed => ScopeLifecycleKind::Completed,
        ScopeState::Failed => ScopeLifecycleKind::Failed,
        ScopeState::Cancelled => ScopeLifecycleKind::Cancelled,
        ScopeState::TimedOut => ScopeLifecycleKind::TimedOut,
        ScopeState::Revoked => ScopeLifecycleKind::Revoked,
        ScopeState::Released => ScopeLifecycleKind::Released,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn terminal_scopes_cannot_restart_and_cleanup_is_idempotent() {
        let mut registry = ScopeRegistry::default();
        registry.create(scope("root", None)).unwrap();
        registry
            .transition(&ScopeId::from("root"), ScopeState::Admitted)
            .unwrap();
        registry
            .transition(&ScopeId::from("root"), ScopeState::Started)
            .unwrap();
        registry.cancel(&ScopeId::from("root")).unwrap();
        assert!(registry
            .transition(&ScopeId::from("root"), ScopeState::Started)
            .is_err());
        registry.cleanup(&ScopeId::from("root")).unwrap();
        registry.cleanup(&ScopeId::from("root")).unwrap();
        assert_eq!(
            registry.get(&ScopeId::from("root")).unwrap().state,
            ScopeState::Released
        );
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use ptr_types::{Revision, ScopeLifecycleKind};

    fn scope(i: usize) -> ExecutionScope {
        ExecutionScope {
            id: ScopeId(format!("s{i:05}")),
            parent: None,
            session: SessionId::from("session"),
            project: ProjectId::from("project"),
            created_at: Timestamp(1),
            deadline: None,
            state: ScopeState::Released,
            cancellation_requested: true,
        }
    }

    /// Review of #141 (P1): at [`MAX_SCOPES`] the next creation is refused with a typed error.
    #[test]
    fn the_scope_count_is_bounded_with_a_typed_refusal() {
        let full: CompactedScopes = (0..MAX_SCOPES)
            .map(|i| {
                (
                    scope(i).id.clone(),
                    (scope(i), ScopeLeaseBinding::default()),
                )
            })
            .collect();
        let registry = ScopeRegistry::from_compacted_parts(full, Some(Revision(1))).unwrap();
        assert!(registry.snapshot_bytes() <= MAX_SCOPE_SNAPSHOT_BYTES);
        let mut next = scope(MAX_SCOPES);
        next.state = ScopeState::Created;
        next.cancellation_requested = false;
        let event = lifecycle_event(
            &next,
            None,
            ScopeState::Created,
            Revision(1),
            ScopeLeaseBinding::default(),
            None,
        );
        assert_eq!(event.kind, ScopeLifecycleKind::Created);
        assert!(matches!(
            registry.check_capacity(&event),
            Err(ScopeError::CapacityExceeded { scopes, .. }) if scopes == MAX_SCOPES + 1
        ));
        // A transition never grows the registry.
        let mut transition = event.clone();
        transition.kind = ScopeLifecycleKind::Admitted;
        assert!(registry.check_capacity(&transition).is_ok());
    }
}
