//! Typed hand-off from an uncertain ExecWire receipt to runtime recovery.
//!
//! ExecWire deliberately cannot decide whether a remote effect should be
//! retried.  An [`WireOutcome::Uncertain`] receipt is therefore converted into
//! the shared `UncertainRequest` contract and handed to the runtime's recovery
//! implementation.  No ledger write or retry is performed in this crate.

use ptr_types::{ScopeId, StatefulRequestRecovery, UncertainRequest};
use std::fmt;

use crate::{WireOutcome, WireReceipt};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecoveryBindingError {
    NotUncertain,
}

impl fmt::Display for RecoveryBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PTR_EXECWIRE_RECOVERY: {self:?}")
    }
}

impl std::error::Error for RecoveryBindingError {}

/// Convert a receipt into the runtime-neutral recovery request.
pub fn uncertain_request(
    receipt: &WireReceipt,
    scope_id: ScopeId,
) -> Result<UncertainRequest, RecoveryBindingError> {
    if !matches!(receipt.outcome, WireOutcome::Uncertain) {
        return Err(RecoveryBindingError::NotUncertain);
    }
    Ok(UncertainRequest {
        request_id: receipt.request_id,
        scope_id,
    })
}

/// Forward an uncertain receipt exactly once to an injected runtime recovery
/// implementation.  The caller must not invoke this again for the same
/// stateful request unless the runtime explicitly makes that operation
/// idempotent; this helper never retries on its own.
pub fn recover_uncertain_receipt<R: StatefulRequestRecovery>(
    recovery: &mut R,
    receipt: &WireReceipt,
    scope_id: ScopeId,
) -> Result<(), RecoveryBindingErrorOr<R::Error>> {
    let request = uncertain_request(receipt, scope_id)?;
    recovery
        .recover_uncertain(request)
        .map_err(RecoveryBindingErrorOr::Recovery)
}

#[derive(Debug)]
pub enum RecoveryBindingErrorOr<E> {
    Binding(RecoveryBindingError),
    Recovery(E),
}

impl<E: fmt::Debug> fmt::Display for RecoveryBindingErrorOr<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding(error) => error.fmt(f),
            Self::Recovery(error) => write!(f, "PTR_EXECWIRE_RECOVERY_RUNTIME: {error:?}"),
        }
    }
}

impl<E: fmt::Debug> std::error::Error for RecoveryBindingErrorOr<E> {}

impl<E> From<RecoveryBindingError> for RecoveryBindingErrorOr<E> {
    fn from(error: RecoveryBindingError) -> Self {
        Self::Binding(error)
    }
}
