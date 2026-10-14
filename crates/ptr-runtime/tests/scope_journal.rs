use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{
    scopes::lifecycle_event, ExecutionScope, PtrRuntime, RuntimeRecoveryAdapter,
    ScopeCleanupCoordinator, ScopeError, ScopeState,
};
use ptr_types::{
    Generation, PodId, ProjectId, ScopeId, ScopeLeaseBinding, SessionId, StateId,
    StatefulRequestRecovery, Timestamp, UncertainRequest,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

fn scope(id: &str) -> ExecutionScope {
    ExecutionScope {
        id: ScopeId::from(id),
        parent: None,
        session: SessionId::from("session"),
        project: ProjectId::from("project"),
        created_at: Timestamp(1),
        deadline: Some(Timestamp(10)),
        state: ScopeState::Created,
        cancellation_requested: false,
    }
}

#[derive(Default)]
struct CleanupProbe {
    steps: Vec<&'static str>,
}

impl ScopeCleanupCoordinator for CleanupProbe {
    fn stop_intake(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("stop_intake");
        Ok(())
    }

    fn cancel_children(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("cancel_children");
        Ok(())
    }

    fn drain_queues(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("drain_queues");
        Ok(())
    }

    fn release_pod_lease(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("release_pod_lease");
        Ok(())
    }

    fn release_resource_lease(
        &mut self,
        _: &ExecutionScope,
    ) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("release_resource_lease");
        Ok(())
    }

    fn close_session(&mut self, _: &ExecutionScope) -> Result<(), ptr_runtime::CleanupError> {
        self.steps.push("close_session");
        Ok(())
    }
}

fn temp_path() -> std::path::PathBuf {
    // The clock alone can repeat across parallel tests (coarse on Windows), and two tests on one
    // ledger file collide on its exclusive lock; the process id and a counter keep paths unique.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ptr-scope-journal-{}-{nonce}-{n}.ledger",
        std::process::id()
    ))
}

#[test]
fn durable_scope_events_replay_and_mark_open_work_recovery_pending() {
    let path = temp_path();
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        runtime
            .create_scope(scope("turn"), ScopeLeaseBinding::default())
            .unwrap();
        runtime
            .transition_scope(
                &ScopeId::from("turn"),
                ScopeState::Admitted,
                ScopeLeaseBinding::default(),
                None,
            )
            .unwrap();
        runtime
            .transition_scope(
                &ScopeId::from("turn"),
                ScopeState::Started,
                ScopeLeaseBinding::default(),
                None,
            )
            .unwrap();
    }

    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    let id = ScopeId::from("turn");
    assert!(reopened.scopes().recovery_required(&id));
    assert_eq!(reopened.scopes().get(&id).unwrap().created_at, Timestamp(1));
    assert_eq!(
        reopened.scopes().get(&id).unwrap().deadline,
        Some(Timestamp(10))
    );
    assert_eq!(reopened.scopes().recoverable_scopes(), vec![id.clone()]);
    assert!(matches!(
        reopened.transition_scope(
            &id,
            ScopeState::Completed,
            ScopeLeaseBinding::default(),
            None,
        ),
        Err(ptr_runtime::RuntimeError::Scope(
            ScopeError::RecoveryRequired(_)
        ))
    ));

    let mut cleanup = CleanupProbe::default();
    reopened
        .recover_scope(&id, &mut cleanup, ScopeLeaseBinding::default())
        .unwrap();
    assert_eq!(
        cleanup.steps,
        vec![
            "stop_intake",
            "cancel_children",
            "drain_queues",
            "release_pod_lease",
            "release_resource_lease",
            "close_session"
        ]
    );
    assert_eq!(
        reopened.scopes().get(&id).unwrap().state,
        ScopeState::Released
    );
    assert!(!reopened.scopes().recovery_required(&id));

    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_scope_that_ended_but_was_never_released_is_recovery_pending_after_a_restart() {
    let path = temp_path();
    let lease = ScopeLeaseBinding::default;
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        // `finished` completed, and the process stopped before its cleanup ran, so
        // its pod lease, resource lease and session were never released.
        // `released` went through cleanup before the restart.
        for id in ["finished", "released"] {
            runtime.create_scope(scope(id), lease()).unwrap();
            for next in [
                ScopeState::Admitted,
                ScopeState::Started,
                ScopeState::Completed,
            ] {
                runtime
                    .transition_scope(&ScopeId::from(id), next, lease(), None)
                    .unwrap();
            }
        }
        let mut cleanup = CleanupProbe::default();
        runtime
            .cleanup_scope(&ScopeId::from("released"), &mut cleanup, lease())
            .unwrap();
    }

    let mut reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    let finished = ScopeId::from("finished");
    let released = ScopeId::from("released");
    assert_eq!(
        reopened.scopes().get(&finished).unwrap().state,
        ScopeState::Completed
    );
    // Only the scope whose leases were never released is left to recover.
    assert!(reopened.scopes().recovery_required(&finished));
    assert!(!reopened.scopes().recovery_required(&released));
    assert_eq!(
        reopened.scopes().recoverable_scopes(),
        vec![finished.clone()]
    );

    // Recovery releases what the finished scope still held, through the same
    // cleanup path, and leaves nothing pending.
    let mut cleanup = CleanupProbe::default();
    reopened
        .recover_scope(&finished, &mut cleanup, lease())
        .unwrap();
    assert_eq!(
        cleanup.steps,
        vec![
            "stop_intake",
            "cancel_children",
            "drain_queues",
            "release_pod_lease",
            "release_resource_lease",
            "close_session"
        ]
    );
    assert_eq!(
        reopened.scopes().get(&finished).unwrap().state,
        ScopeState::Released
    );
    assert!(reopened.scopes().recoverable_scopes().is_empty());

    drop(reopened);
    std::fs::remove_file(path).unwrap();
}

fn stateful_lease(state: &str, pod: &str, token: u128) -> ScopeLeaseBinding {
    ScopeLeaseBinding {
        state_id: Some(StateId::from(state)),
        pod_id: Some(PodId::from(pod)),
        placement_epoch: Some(1),
        fencing_token: Some(token),
        generation: Some(Generation(1)),
        resource_lease_id: Some(format!("resource-{state}")),
    }
}

#[test]
fn recursive_cleanup_uses_each_child_scope_lease() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let root = scope("root");
    let child = ExecutionScope {
        id: ScopeId::from("child"),
        parent: Some(ScopeId::from("root")),
        ..scope("child")
    };
    let root_lease = stateful_lease("root-state", "root-pod", 1);
    let child_lease = stateful_lease("child-state", "child-pod", 2);
    runtime.create_scope(root, root_lease.clone()).unwrap();
    runtime.create_scope(child, child_lease).unwrap();
    runtime
        .transition_scope(
            &ScopeId::from("root"),
            ScopeState::Admitted,
            root_lease.clone(),
            None,
        )
        .unwrap();
    runtime
        .transition_scope(
            &ScopeId::from("root"),
            ScopeState::Started,
            root_lease.clone(),
            None,
        )
        .unwrap();
    runtime
        .transition_scope(
            &ScopeId::from("child"),
            ScopeState::Admitted,
            stateful_lease("child-state", "child-pod", 2),
            None,
        )
        .unwrap();
    runtime
        .transition_scope(
            &ScopeId::from("child"),
            ScopeState::Started,
            stateful_lease("child-state", "child-pod", 2),
            None,
        )
        .unwrap();

    let mut cleanup = CleanupProbe::default();
    runtime
        .cleanup_scope(&ScopeId::from("root"), &mut cleanup, root_lease)
        .unwrap();
    assert_eq!(
        runtime.scopes().get(&ScopeId::from("root")).unwrap().state,
        ScopeState::Released
    );
    assert_eq!(
        runtime.scopes().get(&ScopeId::from("child")).unwrap().state,
        ScopeState::Released
    );
}

#[test]
fn uncertain_request_revokes_scope_and_runs_recovery_cleanup() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let id = ScopeId::from("uncertain");
    let lease = stateful_lease("state", "pod", 7);
    runtime
        .create_scope(scope("uncertain"), lease.clone())
        .unwrap();
    runtime
        .transition_scope(&id, ScopeState::Admitted, lease.clone(), None)
        .unwrap();
    runtime
        .transition_scope(&id, ScopeState::Started, lease.clone(), None)
        .unwrap();

    let mut cleanup = CleanupProbe::default();
    let mut recovery = RuntimeRecoveryAdapter::new(&mut runtime, id.clone(), &mut cleanup, lease);
    recovery
        .recover_uncertain(UncertainRequest {
            request_id: 777,
            scope_id: id.clone(),
        })
        .unwrap();
    drop(recovery);

    assert_eq!(
        cleanup.steps,
        vec![
            "stop_intake",
            "cancel_children",
            "drain_queues",
            "release_pod_lease",
            "release_resource_lease",
            "close_session"
        ]
    );
    assert_eq!(
        runtime.scopes().get(&id).unwrap().state,
        ScopeState::Released
    );
    assert!(runtime.committed_events().iter().any(|event| matches!(
        &event.event,
        LedgerEvent::ScopeLifecycle(lifecycle)
            if lifecycle.scope_id == id
                && lifecycle.kind == ptr_types::ScopeLifecycleKind::Revoked
                && lifecycle.reason.as_deref() == Some("request-uncertain:777")
    )));
}

#[test]
fn stale_scope_revision_is_rejected_before_append() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let mut event = lifecycle_event(
        &scope("stale"),
        None,
        ScopeState::Created,
        ptr_types::Revision(99),
        ScopeLeaseBinding::default(),
        None,
    );
    event.revision = ptr_types::Revision(99);
    assert!(matches!(
        runtime.commit(LedgerEvent::ScopeLifecycle(event)),
        Err(ptr_runtime::RuntimeError::ScopeRevisionMismatch { .. })
    ));
    assert!(runtime.committed_events().is_empty());
}

#[test]
fn identical_committed_scope_event_is_idempotent_and_conflicting_event_is_rejected() {
    let original = scope("scope");
    let event = lifecycle_event(
        &original,
        None,
        ScopeState::Created,
        ptr_types::Revision(0),
        ScopeLeaseBinding::default(),
        None,
    );
    let mut registry = ptr_runtime::ScopeRegistry::default();
    registry.apply_committed(event.clone()).unwrap();
    registry.apply_committed(event).unwrap();
    assert_eq!(registry.events().len(), 1);

    let conflicting = lifecycle_event(
        &original,
        None,
        ScopeState::Created,
        ptr_types::Revision(1),
        ScopeLeaseBinding::default(),
        Some("different".into()),
    );
    assert_eq!(
        registry.apply_committed(conflicting),
        Err(ScopeError::Duplicate(ScopeId::from("scope")))
    );
}

#[test]
fn committed_transition_cannot_change_scope_identity_or_created_shape() {
    let original = scope("scope");
    let created = lifecycle_event(
        &original,
        None,
        ScopeState::Created,
        ptr_types::Revision(0),
        ScopeLeaseBinding::default(),
        None,
    );
    let mut registry = ptr_runtime::ScopeRegistry::default();
    registry.apply_committed(created.clone()).unwrap();

    let mut forged = lifecycle_event(
        &original,
        Some(ScopeState::Created),
        ScopeState::Admitted,
        ptr_types::Revision(0),
        ScopeLeaseBinding::default(),
        None,
    );
    forged.project_id = ProjectId::from("foreign-project");
    assert_eq!(
        registry.apply_committed(forged),
        Err(ScopeError::InvalidCommittedEvent)
    );

    let mut malformed = created;
    malformed.previous_kind = Some(ptr_types::ScopeLifecycleKind::Released);
    let mut fresh = ptr_runtime::ScopeRegistry::default();
    assert_eq!(
        fresh.apply_committed(malformed),
        Err(ScopeError::InvalidCommittedEvent)
    );
}

#[test]
fn ending_a_scope_that_awaits_recovery_keeps_it_pending_until_it_is_released() {
    let lease = ScopeLeaseBinding::default;
    let id = ScopeId::from("turn");
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.create_scope(scope("turn"), lease()).unwrap();
    for next in [ScopeState::Admitted, ScopeState::Started] {
        runtime.transition_scope(&id, next, lease(), None).unwrap();
    }
    // A restart: the scope is still running in the history and awaits recovery.
    let mut restarted =
        PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert!(restarted.scopes().recovery_required(&id));

    // Cancelling it is allowed, but releases nothing: its leases and session are still held.
    restarted
        .transition_scope(&id, ScopeState::Cancelled, lease(), None)
        .unwrap();
    assert!(restarted.scopes().recovery_required(&id));
    assert_eq!(restarted.scopes().recoverable_scopes(), vec![id.clone()]);

    // Recovery still runs the cleanup and only then is nothing pending.
    let mut cleanup = CleanupProbe::default();
    restarted.recover_scope(&id, &mut cleanup, lease()).unwrap();
    assert_eq!(cleanup.steps.len(), 6);
    assert_eq!(
        restarted.scopes().get(&id).unwrap().state,
        ScopeState::Released
    );
    assert!(!restarted.scopes().recovery_required(&id));
    assert!(restarted.scopes().recoverable_scopes().is_empty());
}
