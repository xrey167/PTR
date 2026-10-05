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
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventEnvelope {
    pub sequence: u64,
    pub event: RuntimeEvent,
}
