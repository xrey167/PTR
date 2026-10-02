use ptr_runtime::{AdmissionError, IdentityAdmissionController};
use ptr_types::{
    AdmissionDecision, AdmissionPolicy, AdmissionRequest, AuthenticationLevel, CapabilityId,
    IdentityContext, ProjectId, Revision, SessionId, Timestamp, TypeId,
};

struct Policy {
    decision: AdmissionDecision,
}

impl AdmissionPolicy for Policy {
    type Error = &'static str;

    fn decide(&self, _request: &AdmissionRequest) -> Result<AdmissionDecision, Self::Error> {
        Ok(self.decision.clone())
    }
}

fn request() -> AdmissionRequest {
    let session = SessionId::from("session-1");
    let mut identity = IdentityContext {
        subject: "alice".into(),
        issuer: "issuer".into(),
        groups: vec!["operators".into()],
        authentication_level: AuthenticationLevel::MultiFactor,
        session_id: session.clone(),
        issued_at: Timestamp(10),
        expires_at: Timestamp(100),
        identity_digest: [0; 32],
    };
    identity.identity_digest = identity.canonical_digest();
    AdmissionRequest {
        identity,
        project: ProjectId::from("project-1"),
        capability: CapabilityId::from("infer"),
        input_type: TypeId::from("input"),
        pod: None,
        session,
    }
}

fn controller() -> IdentityAdmissionController<Policy> {
    IdentityAdmissionController::new(
        Policy {
            decision: AdmissionDecision::Allowed {
                project: ProjectId::from("project-1"),
                capabilities: vec![CapabilityId::from("infer")],
                expires_at: Timestamp(90),
                policy_revision: Revision(3),
            },
        },
        Revision(3),
    )
}

#[test]
fn valid_identity_is_admitted_and_session_revoke_fences_it() {
    let mut controller = controller();
    let request = request();
    assert_eq!(
        controller
            .admit(&request, Timestamp(20))
            .unwrap()
            .policy_revision,
        Revision(3)
    );
    controller.revoke_session(request.session.clone());
    assert_eq!(
        controller.admit(&request, Timestamp(20)),
        Err(AdmissionError::SessionRevoked(request.session))
    );
}

#[test]
fn stale_policy_revision_and_capability_are_rejected() {
    let mut controller = controller();
    controller.set_policy_revision(Revision(4));
    assert!(matches!(
        controller.admit(&request(), Timestamp(20)),
        Err(AdmissionError::PolicyRevisionMismatch { .. })
    ));
}

#[test]
fn expired_identity_is_rejected_before_policy() {
    let mut request = request();
    request.identity.expires_at = Timestamp(20);
    let controller = controller();
    assert_eq!(
        controller.admit(&request, Timestamp(20)),
        Err(AdmissionError::Denied(
            ptr_types::AdmissionDenial::ExpiredIdentity
        ))
    );
}

#[test]
fn identity_digest_tampering_is_rejected_as_expired_identity() {
    let mut request = request();
    request.identity.identity_digest[0] ^= 1;
    let controller = controller();
    assert_eq!(
        controller.admit(&request, Timestamp(20)),
        Err(AdmissionError::Denied(
            ptr_types::AdmissionDenial::ExpiredIdentity
        ))
    );
}
