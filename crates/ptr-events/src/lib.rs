//! Runtime event taxonomy and the event-bus contract.
//!
//! The bus distributes projections and telemetry; it is never causal authority.

mod bus;

pub use bus::{
    BusError, BusRecord, EventClass, EventConsumer, EventProducer, InMemoryBus, NewRecord, Offset,
};

use ptr_types::{CandidateId, CommitIndex, RequestId, Revision, ScopeId, ScopeLifecycleKind};

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeEvent {
    RequestStarted(RequestId),
    SnapshotOpened(Revision),
    CandidateGenerated(CandidateId),
    /// A Pod chosen by the trusted checked routing path, emitted before [`RuntimeEvent::PodInvoked`].
    /// Replaying the decision needs the key named by `key_id`, the `nonce` and the request.
    PodRouted {
        pod: String,
        propensity: f64,
        nonce: [u8; 16],
        draw: u64,
        key_id: [u8; 8],
    },
    PodInvoked(String),
    VerifierResult {
        verifier: String,
        passed: bool,
    },
    CommitApplied(CommitIndex),
    ScopeLifecycle {
        scope: ScopeId,
        kind: ScopeLifecycleKind,
    },
    RequestFinished(RequestId),
    /// A stepped resumable run was ended before its next round because the semantic revision
    /// moved for a reason other than the run's own writes: it was bound to `bound` and the
    /// store was at `current`. It is a saving of work; the effect boundary still refuses every
    /// stale action by itself.
    RunCancelled {
        request: RequestId,
        bound: Revision,
        current: Revision,
    },
    /// A verifier-closed workspace request ended. `outcome` is `verified`, `exhausted`,
    /// `escalated` or `failed`; `trace_digest` is the digest of the run's trace (a function of
    /// the seeds, the Pods' answers and the verifier's reports). Observability only: it carries
    /// no authority, and an accepted answer is committed by the output admission, not by this.
    VcwFinished {
        request: RequestId,
        outcome: String,
        trace_digest: [u8; 32],
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventEnvelope {
    pub sequence: u64,
    pub event: RuntimeEvent,
}
