//! Runtime event taxonomy and the event-bus contract.
//!
//! The bus distributes projections and telemetry; it is never causal authority.

mod bus;

pub use bus::{
    BusError, BusRecord, EventClass, EventConsumer, EventProducer, InMemoryBus, NewRecord, Offset,
};

use ptr_types::{CandidateId, CommitIndex, RequestId, Revision};

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeEvent {
    RequestStarted(RequestId),
    SnapshotOpened(Revision),
    CandidateGenerated(CandidateId),
    PodInvoked(String),
    VerifierResult { verifier: String, passed: bool },
    CommitApplied(CommitIndex),
    RequestFinished(RequestId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventEnvelope {
    pub sequence: u64,
    pub event: RuntimeEvent,
}
