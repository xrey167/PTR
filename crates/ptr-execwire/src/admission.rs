//! Typed bridge from a runtime ActionAdmission to the effect-free wire request.
//!
//! The bridge carries a runtime decision across the transport boundary; it does
//! not grant authority. The receiving ExecutionHost authenticates the peer and
//! re-runs its policy checks before any effect attempt is journaled.

use ptr_core::action_head::ActionIr;
use ptr_runtime::{execution::action_digest, ActionAdmission};
use ptr_types::ProjectId;
use std::fmt;

use crate::WireRequest;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionBindingError {
    ActionDigestMismatch,
    ProjectMismatch {
        expected: ProjectId,
        actual: ProjectId,
    },
    EmptyAddress,
    EmptyOnceKey,
}

impl fmt::Display for AdmissionBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PTR_EXECWIRE_ADMISSION: {self:?}")
    }
}

impl std::error::Error for AdmissionBindingError {}

/// Bind an already admitted runtime action to one concrete ExecWire request.
///
/// `project` is checked against the admission rather than accepted as a new
/// authorization input. `request_id` is the transport nonce; `once_key` is the
/// durable at-most-once identity and must be supplied for stateful effects.
pub fn bind_action_admission(
    admission: &ActionAdmission,
    action: ActionIr,
    project: ProjectId,
    addressed_to: impl Into<String>,
    request_id: u64,
    once_key: Option<String>,
) -> Result<WireRequest, AdmissionBindingError> {
    if action_digest(&action) != admission.action_digest {
        return Err(AdmissionBindingError::ActionDigestMismatch);
    }
    if project != admission.project {
        return Err(AdmissionBindingError::ProjectMismatch {
            expected: admission.project.clone(),
            actual: project,
        });
    }
    let addressed_to = addressed_to.into();
    if addressed_to.is_empty() {
        return Err(AdmissionBindingError::EmptyAddress);
    }
    if once_key.as_deref() == Some("") {
        return Err(AdmissionBindingError::EmptyOnceKey);
    }
    Ok(WireRequest {
        addressed_to,
        request_id,
        project,
        action,
        once_key,
    })
}
