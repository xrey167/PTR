//! Egress refusals as committed history.
//!
//! A Pod's link says where it may send. When a send is refused (an unlisted host or port, a
//! topic no filter matches, a lease that spent its frame budget) the refusal happened before
//! anything left the host, so no effect needs reconciling. It still matters to whoever audits
//! the run, and a log line is not evidence. The runtime commits it as a
//! [`LedgerEvent::EgressRefused`] so the refusal replays with the rest of the history.
//!
//! The record carries the trace, the Pod revision that tried, the redacted target (a
//! `host:port` or a topic) and the reason. It carries no payload, no resolved address and no
//! credential: Invariant 13 keeps request bytes out of the audit trail, and
//! `ptr_pods::redact_target` is what a caller uses to get a target fit to record.
use crate::{PtrRuntime, RuntimeError};
use ptr_ledger::LedgerEvent;
use ptr_types::{CommitIndex, EgressRefusalReason, PodRevisionAddress, TraceId};

/// One refusal to commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EgressRefusalRecord {
    /// The trace of the request whose send was refused.
    pub trace_id: TraceId,
    /// The Pod revision that tried to send.
    pub source: PodRevisionAddress,
    /// The redacted `host:port` or topic. Never a payload.
    pub target: String,
    /// Which rule refused it.
    pub reason: EgressRefusalReason,
}

impl PtrRuntime {
    /// Commit an egress refusal. The record is validated with the same rules replay applies, so
    /// a record this refuses is one the ledger would refuse to read back.
    ///
    /// Unlike a generic [`PtrRuntime::commit`], this is the only path that writes the record:
    /// the runtime is the one who saw the refusal.
    ///
    /// # Errors
    /// [`RuntimeError::Ledger`] when the record is malformed (empty trace, invalid source
    /// address, a target that is empty, over its bound or carries whitespace), or the runtime's
    /// own errors when execution is fenced or the append fails.
    pub fn record_egress_refusal(
        &mut self,
        record: EgressRefusalRecord,
    ) -> Result<CommitIndex, RuntimeError> {
        let event = LedgerEvent::EgressRefused {
            trace_id: record.trace_id,
            source: record.source,
            target: record.target,
            reason: record.reason,
        };
        // The same shape check the codec applies, run before the first durable byte.
        ptr_ledger::check_encodable(&event)
            .map_err(|error| RuntimeError::Ledger(error.to_string()))?;
        self.commit_settlement(event)
    }
}
