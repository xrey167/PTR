use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapabilityId, CapsuleId, Effect, Generation, ProjectId, Revision, TypeId};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn revoked_generation_does_not_resurrect_after_replay() {
    let mut original = PtrRuntime::new(PtrConfig::default()).unwrap();
    original
        .commit(LedgerEvent::CapsuleCommitted {
            project: ProjectId::from("p"),
            capsule: CapsuleId::from("capsule:a"),
            generation: Generation(1),
        })
        .unwrap();
    original
        .commit(LedgerEvent::Revoked {
            subject: "capsule:a".into(),
            generation: Generation(1),
        })
        .unwrap();

    let persisted = original.committed_events().to_vec();
    let mut restarted = PtrRuntime::replay(PtrConfig::default(), &persisted).unwrap();
    restarted
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("memory.write"));
    restarted.permissions_mut().allow_mutation = true;

    let action = ActionIr {
        operation: "update".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("memory.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("SemanticCapsule"),
        generation: Generation(1),
        revision: Revision(0),
        payload: vec![],
    };

    assert_eq!(
        restarted.authorize_action(&action),
        Err(RuntimeError::StaleGeneration {
            target: "capsule:a".into(),
            action: Generation(1),
            current: Some(Generation(1)),
        })
    );
}

#[test]
fn durable_runtime_reopen_preserves_revocation_boundary() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "ptr-runtime-durable-{}-{nonce}.log",
        std::process::id()
    ));

    {
        let mut runtime = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
        runtime
            .commit(LedgerEvent::CapsuleCommitted {
                project: ProjectId::from("p"),
                capsule: CapsuleId::from("capsule:durable"),
                generation: Generation(1),
            })
            .unwrap();
        runtime
            .commit(LedgerEvent::Revoked {
                subject: "capsule:durable".into(),
                generation: Generation(1),
            })
            .unwrap();
    }

    let mut restarted = PtrRuntime::open_durable(PtrConfig::default(), &path).unwrap();
    assert_eq!(restarted.committed_events().len(), 2);
    assert_eq!(restarted.materialized_state().last_applied, 2);
    restarted
        .permissions_mut()
        .capabilities
        .insert(CapabilityId::from("memory.write"));
    restarted.permissions_mut().allow_mutation = true;

    let action = ActionIr {
        operation: "update".into(),
        target: "capsule:durable".into(),
        capability: CapabilityId::from("memory.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("SemanticCapsule"),
        generation: Generation(1),
        revision: Revision(0),
        payload: vec![],
    };

    assert_eq!(
        restarted.authorize_action(&action),
        Err(RuntimeError::StaleGeneration {
            target: "capsule:durable".into(),
            action: Generation(1),
            current: Some(Generation(1)),
        })
    );

    drop(restarted);
    std::fs::remove_file(path).unwrap();
}

