use ptr_config::PtrConfig;
use ptr_core::action_head::ActionIr;
use ptr_ledger::LedgerEvent;
use ptr_runtime::{PtrRuntime, RuntimeError};
use ptr_types::{CapabilityId, Effect, Generation, RequestId, Revision, TypeId};

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
    let action = ActionIr {
        operation: "write".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision,
        payload: vec![],
    };
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
fn committed_event_materializes() {
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.commit(LedgerEvent::HardConstraintCommitted {
        key: "no-network".into(),
        generation: Generation(3),
    });
    assert_eq!(
        runtime
            .materialized_state()
            .values
            .get("constraint:no-network"),
        Some(&"3".to_string())
    );
}
