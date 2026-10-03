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
    InvalidTransition { from: ScopeState, to: ScopeState },
    Terminal(ScopeState),
    RecoveryRequired(ScopeId),
    InvalidCommittedEvent,
    InvalidLeaseBinding,
    InvalidRevision,
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

#[derive(Default)]
pub struct ScopeRegistry {
    scopes: BTreeMap<ScopeId, ExecutionScope>,
    events: Vec<ScopeEvent>,
    committed: Vec<ScopeLifecycleEvent>,
    recovery_required: BTreeSet<ScopeId>,
    leases: BTreeMap<ScopeId, ScopeLeaseBinding>,
    last_revision: Option<ptr_types::Revision>,
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
        self.leases.insert(id.clone(), ScopeLeaseBinding::default());
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

    pub fn mark_recovery_required(&mut self) {
        self.recovery_required = self
            .scopes
            .values()
            .filter(|scope| !scope.state.is_terminal())
            .map(|scope| scope.id.clone())
            .collect();
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
        if self.recovery_required(id) {
            return Err(ScopeError::RecoveryRequired(id.clone()));
        }
        let scope = self.get(id)?;
        if scope.state.is_terminal() {
            return Err(ScopeError::Terminal(scope.state));
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
        if next.is_terminal() {
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
        self.recovery_required = self
            .scopes
            .values()
            .filter(|scope| !scope.state.is_terminal())
            .map(|scope| scope.id.clone())
            .collect();
        Ok(())
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
