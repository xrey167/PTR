use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_security::{AuthorizationDecision, AuthorizationDenial};
use ptr_types::{CapabilityId, Effect, Generation, RequestId, Revision, TypeId};

fn mutation_action(revision: Revision, generation: Generation) -> ActionIr {
    ActionIr {
        operation: "write".into(),
        target: "artifact:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation,
        revision,
        payload: vec![],
    }
}

#[test]
fn ingest_creates_revision_and_runtime_events() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime.ingest_text(RequestId::from("r1"), "hello");
    assert_eq!(revision, Revision(1));
    assert_eq!(runtime.events().len(), 2);
}

#[test]
fn action_requires_capability() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime.ingest_text(RequestId::from("r1"), "hello");
    let action = mutation_action(revision, Generation(1));
    runtime.set_live_generation(&action.target, action.generation);

    assert_eq!(
        runtime.authorize_action(&action),
        Err(RuntimeError::PermissionDenied)
    );
    runtime
        .permissions_mut()
        .capabilities
        .insert(action.capability.clone());
    runtime.permissions_mut().allow_mutation = true;
    assert!(runtime.authorize_action(&action).is_ok());
}

#[test]
fn stale_revision_is_rejected_before_permission_check() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let old = runtime.ingest_text(RequestId::from("r1"), "first");
    runtime.ingest_text(RequestId::from("r2"), "second");
    let action = mutation_action(old, Generation(1));
    runtime.set_live_generation(&action.target, action.generation);

    assert_eq!(
        runtime.authorize_action(&action),
        Err(RuntimeError::StaleRevision {
            action: old,
            current: Revision(2),
        })
    );
}

#[test]
fn unknown_and_revoked_generations_are_rejected() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime.ingest_text(RequestId::from("r1"), "hello");
    let action = mutation_action(revision, Generation(7));

    assert_eq!(
        runtime.authorize_action(&action),
        Err(RuntimeError::UnknownGeneration {
            target: "artifact:a".into(),
        })
    );

    runtime.set_live_generation("artifact:a", Generation(7));
    runtime
        .commit(LedgerEvent::Revoked {
            subject: "artifact:a".into(),
            generation: Generation(7),
        })
        .unwrap();

    assert_eq!(
        runtime.authorize_action(&action),
        Err(RuntimeError::StaleGeneration {
            target: "artifact:a".into(),
            action: Generation(7),
            current: Some(Generation(7)),
        })
    );

    runtime.set_live_generation("artifact:a", Generation(8));
    let fresh = mutation_action(runtime.revision(), Generation(8));
    assert_eq!(
        runtime.authorize_action(&fresh),
        Err(RuntimeError::PermissionDenied)
    );
}

#[test]
fn committed_event_materializes() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .commit(LedgerEvent::HardConstraintCommitted {
            key: "no-network".into(),
            generation: Generation(3),
        })
        .unwrap();
    assert_eq!(
        runtime
            .materialized_state()
            .values
            .get("constraint:no-network"),
        Some(&"3".to_string())
    );
}

#[test]
fn runtime_exposes_typed_authorization_decision() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    let revision = runtime.ingest_text(RequestId::from("r1"), "hello");
    let action = mutation_action(revision, Generation(1));

    assert_eq!(
        runtime.authorization_decision(&action),
        AuthorizationDecision::Deny(AuthorizationDenial::UnknownGeneration {
            target: "artifact:a".into(),
        })
    );
}
