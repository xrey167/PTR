use ptr_config::PtrConfig;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapsuleId, Generation, ProjectId};

fn activate(target: &str, generation: u64) -> LedgerEvent {
    LedgerEvent::CapsuleCommitted {
        project: ProjectId::from("enumeration"),
        capsule: CapsuleId::from(target),
        generation: Generation(generation),
    }
}

#[test]
fn new_runtime_has_no_lifecycle_entries() {
    let runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    assert_eq!(runtime.live_generations().count(), 0);
    assert_eq!(runtime.revoked_generations().count(), 0);
}

#[test]
fn live_entries_include_each_namespace_in_target_order_with_latest_generations() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    for event in [
        LedgerEvent::ProcedurePromoted {
            id: "a".into(),
            generation: Generation(3),
        },
        activate("capsule:z", 7),
        LedgerEvent::HardConstraintCommitted {
            key: "a".into(),
            generation: Generation(4),
        },
        activate("capsule:a", 1),
        LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId::from("capsule:z"),
            old: Generation(7),
            new: Generation(8),
        },
        activate("capsule:a", 1),
    ] {
        runtime.commit(event).unwrap();
    }
    assert_eq!(
        runtime.live_generations().collect::<Vec<_>>(),
        [
            ("capsule:a", Generation(1)),
            ("capsule:z", Generation(8)),
            ("constraint:a", Generation(4)),
            ("procedure:a", Generation(3)),
        ]
    );
    assert_eq!(runtime.revoked_generations().count(), 0);
}

#[test]
fn tombstones_include_unknown_targets_and_distant_generations_once_across_replay() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.commit(activate("capsule:a", 2)).unwrap();
    for (subject, generation) in [
        ("unknown", u64::MAX),
        ("capsule:a", 2),
        ("capsule:a", 1000),
        ("capsule:a", 2),
    ] {
        runtime
            .commit(LedgerEvent::Revoked {
                subject: subject.into(),
                generation: Generation(generation),
            })
            .unwrap();
    }
    runtime
        .commit(LedgerEvent::ProcedureRevoked {
            id: "absent".into(),
            generation: Generation(9),
        })
        .unwrap();
    // Revocation retains the current generation, and a newer activation
    // does not erase older or future tombstones.
    assert_eq!(
        runtime.live_generations().collect::<Vec<_>>(),
        [("capsule:a", Generation(2))]
    );
    runtime.commit(activate("capsule:a", 3)).unwrap();
    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    for state in [&runtime, &replayed] {
        assert_eq!(
            state.live_generations().collect::<Vec<_>>(),
            [("capsule:a", Generation(3))]
        );
        assert_eq!(
            state.revoked_generations().collect::<Vec<_>>(),
            [
                ("capsule:a", Generation(2)),
                ("capsule:a", Generation(1000)),
                ("procedure:absent", Generation(9)),
                ("unknown", Generation(u64::MAX)),
            ]
        );
    }
}

#[test]
fn rejected_activation_does_not_add_or_replace_enumerated_entries() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.commit(activate("capsule:a", 5)).unwrap();
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "unknown".into(),
            generation: Generation(1),
        })
        .unwrap();
    assert!(matches!(
        runtime.commit(activate("capsule:a", 4)),
        Err(RuntimeError::InvalidLifecycleTransition { .. })
    ));
    assert!(matches!(
        runtime.commit(activate("unknown", 1)),
        Err(RuntimeError::InvalidLifecycleTransition { .. })
    ));
    assert_eq!(
        runtime.live_generations().collect::<Vec<_>>(),
        [("capsule:a", Generation(5))]
    );
    assert_eq!(
        runtime.revoked_generations().collect::<Vec<_>>(),
        [("unknown", Generation(1))]
    );
}
