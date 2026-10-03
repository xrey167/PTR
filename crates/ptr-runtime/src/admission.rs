use ptr_types::{
    AdmissionDecision, AdmissionDenial, AdmissionPolicy, AdmissionRequest, CapabilityId, ProjectId,
    Revision, SessionId, Timestamp,
};
use std::collections::BTreeSet;

#[derive(Debug, PartialEq, Eq)]
pub enum AdmissionError<E> {
    Policy(E),
    Denied(AdmissionDenial),
    PolicyRevisionMismatch {
        expected: Revision,
        actual: Revision,
    },
    ProjectMismatch {
        expected: ProjectId,
        actual: ProjectId,
    },
    CapabilityMissing(CapabilityId),
    SessionRevoked(SessionId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionGrant {
    pub project: ProjectId,
    pub session: SessionId,
    pub capabilities: Vec<CapabilityId>,
    pub expires_at: Timestamp,
    pub policy_revision: Revision,
}

pub struct IdentityAdmissionController<P> {
    policy: P,
    policy_revision: Revision,
    revoked_sessions: BTreeSet<SessionId>,
}

impl<P> IdentityAdmissionController<P> {
    pub fn new(policy: P, policy_revision: Revision) -> Self {
        Self {
            policy,
            policy_revision,
            revoked_sessions: BTreeSet::new(),
        }
    }

    pub fn policy_revision(&self) -> Revision {
        self.policy_revision
    }

    pub fn set_policy_revision(&mut self, revision: Revision) {
        self.policy_revision = revision;
    }

    pub fn revoke_session(&mut self, session: SessionId) {
        self.revoked_sessions.insert(session);
    }

    pub fn is_session_revoked(&self, session: &SessionId) -> bool {
        self.revoked_sessions.contains(session)
    }
}

impl<P: AdmissionPolicy> IdentityAdmissionController<P> {
    pub fn admit(
        &self,
        request: &AdmissionRequest,
        now: Timestamp,
    ) -> Result<AdmissionGrant, AdmissionError<P::Error>> {
        if !request.identity.is_valid_at(now) {
            return Err(AdmissionError::Denied(AdmissionDenial::ExpiredIdentity));
        }
        if request.identity.session_id != request.session {
            return Err(AdmissionError::Denied(AdmissionDenial::SessionMismatch));
        }
        if self.revoked_sessions.contains(&request.session) {
            return Err(AdmissionError::SessionRevoked(request.session.clone()));
        }

        let decision = self
            .policy
            .decide(request)
            .map_err(AdmissionError::Policy)?;
        let (project, capabilities, expires_at, policy_revision) = match decision {
            AdmissionDecision::Allowed {
                project,
                capabilities,
                expires_at,
                policy_revision,
            } => (project, capabilities, expires_at, policy_revision),
            AdmissionDecision::Denied { reason } => return Err(AdmissionError::Denied(reason)),
        };

        if policy_revision != self.policy_revision {
            return Err(AdmissionError::PolicyRevisionMismatch {
                expected: self.policy_revision,
                actual: policy_revision,
            });
        }
        if project != request.project {
            return Err(AdmissionError::ProjectMismatch {
                expected: request.project.clone(),
                actual: project,
            });
        }
        if !capabilities
            .iter()
            .any(|capability| capability == &request.capability)
        {
            return Err(AdmissionError::CapabilityMissing(
                request.capability.clone(),
            ));
        }
        if expires_at.0 > request.identity.expires_at.0 || expires_at.0 <= now.0 {
            return Err(AdmissionError::Denied(AdmissionDenial::ExpiredIdentity));
        }

        Ok(AdmissionGrant {
            project,
            session: request.session.clone(),
            capabilities,
            expires_at,
            policy_revision,
        })
    }
}
