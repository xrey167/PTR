use ptr_config::PtrConfig;
use ptr_ledger::{CommittedEvent, FileLedger, LedgerEvent};
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapsuleId, CommitIndex, Generation, ProjectId};
use std::time::{SystemTime, UNIX_EPOCH};

fn activate(generation: u64) -> LedgerEvent {
    LedgerEvent::CapsuleCommitted {
        project: ProjectId::from("p"),
        capsule: CapsuleId::from("capsule:a"),
        generation: Generation(generation),
    }
}

#[test]
fn generations_cannot_rewind_or_reactivate_a_tombstone() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.commit(activate(7)).unwrap();
    runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(7),
            new: Generation(8),
        })
        .unwrap();
    for event in [
        activate(7),
        LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(7),
            new: Generation(9),
        },
        LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(8),
            new: Generation(8),
        },
    ] {
        assert!(matches!(
            runtime.commit(event),
            Err(RuntimeError::InvalidLifecycleTransition { .. })
        ));
        assert_eq!(runtime.committed_events().len(), 2);
        assert_eq!(runtime.live_generation("capsule:a"), Some(Generation(8)));
    }
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(8),
        })
        .unwrap();
    assert!(runtime.commit(activate(8)).is_err());
    let persisted = runtime.committed_events().to_vec();
    let mut replayed = PtrRuntime::replay(PtrConfig::default(), &persisted).unwrap();
    assert!(replayed.commit(activate(8)).is_err());
    replayed.commit(activate(9)).unwrap();
    assert_eq!(replayed.live_generation("capsule:a"), Some(Generation(9)));
}

#[test]
fn unknown_supersession_and_cross_project_reuse_fail_before_commit() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert!(runtime
        .commit(LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:a"),
            old: Generation(1),
            new: Generation(2),
        })
        .is_err());
    assert!(runtime.committed_events().is_empty());
    runtime.commit(activate(1)).unwrap();
    assert!(matches!(
        runtime.commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("other"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(2),
        }),
        Err(RuntimeError::CapsuleProjectMismatch { .. })
    ));
    assert_eq!(runtime.committed_events().len(), 1);
}

#[test]
fn reserved_lifecycle_namespaces_cannot_be_impersonated_by_capsules() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    for name in ["constraint:no-network", "procedure:work"] {
        assert!(matches!(
            runtime.commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from(name),
                generation: Generation(1),
            }),
            Err(RuntimeError::ReservedTargetNamespace { .. })
        ));
    }
    assert!(runtime.committed_events().is_empty());
}

#[test]
fn procedure_and_constraint_updates_are_monotone() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::ProcedurePromoted {
            id: "a".into(),
            generation: Generation(3),
        })
        .unwrap();
    runtime
        .commit(LedgerEvent::HardConstraintCommitted {
            key: "a".into(),
            generation: Generation(3),
        })
        .unwrap();
    assert!(runtime
        .commit(LedgerEvent::ProcedurePromoted {
            id: "a".into(),
            generation: Generation(2)
        })
        .is_err());
    assert!(runtime
        .commit(LedgerEvent::HardConstraintCommitted {
            key: "a".into(),
            generation: Generation(2)
        })
        .is_err());
    runtime
        .commit(LedgerEvent::ProcedureRevoked {
            id: "a".into(),
            generation: Generation(4),
        })
        .unwrap();
    assert!(runtime
        .commit(LedgerEvent::ProcedurePromoted {
            id: "a".into(),
            generation: Generation(4)
        })
        .is_err());
    assert_eq!(runtime.live_generation("procedure:a"), Some(Generation(3)));
    assert_eq!(runtime.committed_events().len(), 3);
}

#[test]
fn identical_live_activation_is_idempotent_but_future_snapshot_is_invalid() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.commit(activate(1)).unwrap();
    runtime.commit(activate(1)).unwrap();
    assert_eq!(runtime.live_generation("capsule:a"), Some(Generation(1)));
    assert!(matches!(
        runtime.commit(LedgerEvent::SnapshotCommitted {
            revision: 0,
            covers: CommitIndex(3),
        }),
        Err(RuntimeError::SnapshotBeyondHistory { .. })
    ));
    assert_eq!(runtime.committed_events().len(), 2);
}

#[test]
fn invalid_history_is_rejected_by_memory_and_durable_replay() {
    let events = [
        CommittedEvent {
            index: CommitIndex(1),
            event: activate(2),
        },
        CommittedEvent {
            index: CommitIndex(2),
            event: activate(1),
        },
    ];
    assert!(matches!(
        PtrRuntime::replay(PtrConfig::default(), &events),
        Err(RuntimeError::InvalidLifecycleTransition { .. })
    ));
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "ptr-invalid-history-{}-{nonce}.log",
        std::process::id()
    ));
    {
        let mut log = FileLedger::open(&path).unwrap();
        for committed in &events {
            log.append_durable(committed.event.clone()).unwrap();
        }
    }
    assert!(matches!(
        PtrRuntime::open_durable(PtrConfig::default(), &path),
        Err(RuntimeError::InvalidLifecycleTransition { .. })
    ));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn rejected_transition_never_enters_durable_history() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "ptr-rejected-history-{}-{nonce}.log",
        std::process::id()
    ));
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        runtime.commit(activate(2)).unwrap();
    }
    // Inspect bytes outside the OS-lock lifetime on every platform.
    let bytes = std::fs::read(&path).unwrap();
    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        assert!(runtime.commit(activate(1)).is_err());
        assert_eq!(runtime.committed_events().len(), 1);
    }
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let reopened = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(reopened.live_generation("capsule:a"), Some(Generation(2)));
    assert_eq!(reopened.committed_events().len(), 1);
    drop(reopened);
    std::fs::remove_file(path).unwrap();
}
