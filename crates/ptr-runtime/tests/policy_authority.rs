use ed25519_dalek::{Signer, SigningKey};
use ptr_config::PtrConfig;
use ptr_runtime::{
    CapabilityPolicy, ManifestBindingAuthority, PodPolicyBinding, PolicyBundlePayload, PolicyError,
    PolicyTrustStore, ProjectPolicy, PtrRuntime, SignedPolicyBundle,
};
use ptr_types::{CapabilityId, PodId, ProjectId, Revision, SessionId, Timestamp, TypeId};

fn bundle(key: &SigningKey, revision: u64) -> SignedPolicyBundle {
    let payload = PolicyBundlePayload {
        projects: vec![ProjectPolicy {
            project: ProjectId::from("project-a"),
        }],
        capabilities: vec![CapabilityPolicy {
            project: ProjectId::from("project-a"),
            capability: CapabilityId::from("read"),
        }],
        input_types: vec![TypeId::from("input")],
        pod_bindings: vec![PodPolicyBinding {
            project: ProjectId::from("project-a"),
            pod: PodId::from("pod-a"),
            capability: CapabilityId::from("read"),
            input_type: TypeId::from("input"),
        }],
        valid_from: Timestamp(1),
        valid_until: Some(Timestamp(100)),
    };
    let mut result = SignedPolicyBundle {
        revision: Revision(revision),
        key_id: "root".into(),
        payload,
        payload_digest: [0; 32],
        signature: Vec::new(),
    };
    result.payload_digest = result.payload.digest();
    result.signature = key.sign(&result.signing_bytes()).to_bytes().to_vec();
    result
}

#[test]
fn policy_activation_is_verified_and_replayed() {
    let signing = SigningKey::from_bytes(&[21; 32]);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.install_policy_key("root", signing.verifying_key());
    runtime
        .activate_policy_bundle(bundle(&signing, 1), Timestamp(10))
        .unwrap();

    assert_eq!(
        runtime
            .manifest_authority_registry()
            .read()
            .unwrap()
            .current_policy_revision(),
        Revision(1)
    );
    assert!(runtime.committed_events().iter().any(|event| matches!(
        event.event,
        ptr_ledger::LedgerEvent::PolicyBundleActivated { .. }
    )));

    let replayed = PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert_eq!(
        replayed
            .manifest_authority_registry()
            .read()
            .unwrap()
            .current_policy_revision(),
        Revision(1)
    );
    assert!(replayed.policy_authority_ready());
}

#[test]
fn invalid_signature_and_revision_conflict_fail_closed() {
    let signing = SigningKey::from_bytes(&[22; 32]);
    let other = SigningKey::from_bytes(&[23; 32]);
    let mut trust = PolicyTrustStore::default();
    trust.insert("root", signing.verifying_key());
    let invalid = bundle(&other, 1);
    assert_eq!(
        invalid.verify(&trust, Timestamp(10)),
        Err(PolicyError::InvalidSignature)
    );

    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.install_policy_key("root", signing.verifying_key());
    runtime
        .activate_policy_bundle(bundle(&signing, 1), Timestamp(10))
        .unwrap();
    let mut conflicting = bundle(&signing, 1);
    conflicting.payload.projects.push(ProjectPolicy {
        project: ProjectId::from("project-b"),
    });
    conflicting.payload_digest = conflicting.payload.digest();
    conflicting.signature = signing
        .sign(&conflicting.signing_bytes())
        .to_bytes()
        .to_vec();
    assert!(runtime
        .activate_policy_bundle(conflicting, Timestamp(10))
        .is_err());
    assert_eq!(runtime.committed_events().len(), 1);
}

#[test]
fn policy_revocation_is_durable_and_idempotence_is_preserved() {
    let signing = SigningKey::from_bytes(&[24; 32]);
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime.install_policy_key("root", signing.verifying_key());
    runtime
        .activate_policy_bundle(bundle(&signing, 1), Timestamp(10))
        .unwrap();
    runtime
        .revoke_policy_revision(Revision(1), "operator")
        .unwrap();
    assert_eq!(
        runtime
            .manifest_authority_registry()
            .read()
            .unwrap()
            .current_policy_revision(),
        Revision(0)
    );
    assert!(runtime
        .revoke_policy_revision(Revision(1), "operator")
        .is_ok());
    assert_eq!(runtime.committed_events().len(), 2);
}

#[test]
fn session_revocation_is_durable_and_replayed_fail_closed() {
    let session = SessionId::from("session-policy-1");
    let mut runtime = PtrRuntime::new(PtrConfig::default()).unwrap();
    runtime
        .revoke_identity_session(session.clone(), "operator logout")
        .unwrap();
    assert!(runtime.is_identity_session_revoked(&session));
    assert!(runtime.committed_events().iter().any(|event| matches!(
        &event.event,
        ptr_ledger::LedgerEvent::SessionRevoked { session_id, reason }
            if session_id == &session && reason == "operator logout"
    )));

    let mut replayed =
        PtrRuntime::replay(PtrConfig::default(), runtime.committed_events()).unwrap();
    assert!(replayed.is_identity_session_revoked(&session));
    let count = replayed.committed_events().len();
    replayed
        .revoke_identity_session(session.clone(), "operator logout")
        .unwrap();
    assert_eq!(replayed.committed_events().len(), count);
}
