#![cfg(feature = "execwire-backend")]

use ptr_core::action_head::ActionIr;
use ptr_execwire::{bind_action_admission, AdmissionBindingError};
use ptr_runtime::{execution::action_digest, ActionAdmission};
use ptr_types::{
    CapabilityId, Effect, Generation, PodId, ProjectId, RequestId, Revision, ScopeId, SessionId,
    TypeId,
};

fn action() -> ActionIr {
    ActionIr {
        operation: "write".into(),
        target: "capsule:a".into(),
        capability: CapabilityId::from("file.write"),
        effect: Effect::Mutation,
        input_type: TypeId::from("Bytes"),
        generation: Generation(1),
        revision: Revision(1),
        payload: b"payload".to_vec(),
    }
}

fn admission(action: &ActionIr) -> ActionAdmission {
    ActionAdmission {
        request_id: RequestId::from("request"),
        session_id: SessionId::from("session"),
        scope_id: ScopeId::from("scope"),
        project: ProjectId::from("project"),
        pod_id: PodId::from("pod"),
        output_digest: [2; 32],
        action_digest: action_digest(action),
    }
}

#[test]
fn admitted_action_binds_to_wire_without_reauthorizing_at_the_bridge() {
    let action = action();
    let request = bind_action_admission(
        &admission(&action),
        action.clone(),
        ProjectId::from("project"),
        "peer",
        7,
        Some("request:action:1".into()),
    )
    .unwrap();
    assert_eq!(request.action, action);
    assert_eq!(request.request_id, 7);
    assert_eq!(request.once_key.as_deref(), Some("request:action:1"));
}

#[test]
fn bridge_rejects_tampered_action_project_and_once_key() {
    let action = action();
    let mut tampered = action.clone();
    tampered.payload.push(0);
    assert_eq!(
        bind_action_admission(
            &admission(&action),
            tampered,
            ProjectId::from("project"),
            "peer",
            7,
            Some("key".into()),
        ),
        Err(AdmissionBindingError::ActionDigestMismatch)
    );
    assert!(matches!(
        bind_action_admission(
            &admission(&action),
            action.clone(),
            ProjectId::from("other"),
            "peer",
            7,
            Some("key".into()),
        ),
        Err(AdmissionBindingError::ProjectMismatch { .. })
    ));
    assert_eq!(
        bind_action_admission(
            &admission(&action),
            action,
            ProjectId::from("project"),
            "peer",
            7,
            Some(String::new()),
        ),
        Err(AdmissionBindingError::EmptyOnceKey)
    );
}
