//! Runtime event taxonomy and the event-bus contract.
//!
//! The bus distributes projections and telemetry; it is never causal authority.

mod bus;
mod codec;
mod file_log;
mod policy;

pub use bus::{
    BusConfig, BusError, BusRecord, CorruptionKind, EventClass, EventConsumer, EventProducer,
    Field, InMemoryBus, NewRecord, Offset, DEFAULT_MAX_RECORD_BYTES, MAX_BUS_RECORD_CAPACITY,
    MAX_CONSUMERS,
};
pub use file_log::{
    FileEventLog, FileLogConfig, FsyncPolicy, RecoveryReport, TornTail, DEFAULT_SEGMENT_BYTES,
    MAX_FILE_LOG_UNACKED_RECORDS,
};
pub use policy::{
    Authority, BackpressurePolicy, ClassPolicies, Compaction, ConsumerLag, Limit, PolicyViolation,
    RetentionPolicy, MAX_RETENTION_BYTES, MAX_RETENTION_RECORDS,
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
    /// The host changed the routing policy of a verifier-closed loop: `action` is `promoted` or
    /// `rolled_back`, `from` and `to` are policy digests, and `evidence` is the digest of the
    /// off-policy evidence a promotion rested on (`None` for a rollback). The record of a host
    /// action, not an authority: routing never changes except through that host action, and
    /// this event is in memory like [`RuntimeEvent::VcwFinished`] (not committed to the ledger).
    ///
    /// Observability, not an audit record: it is lost on restart, nothing replays it, and no
    /// gate record is checked against it. Until a ledger event carries the candidate and
    /// incumbent digests, the evidence and log digests, the cutoff, the scope and the rule,
    /// a host must not rely on it to show that a promotion was justified.
    VcwPolicyChanged {
        scope: String,
        sequence: u64,
        action: String,
        from: [u8; 32],
        to: [u8; 32],
        evidence: Option<[u8; 32]>,
    },
    /// An off-policy estimate logged by a verifier-closed loop that learns in shadow mode
    /// (`[vcw] learning = "shadow"`): the paired value difference of a learned candidate over the
    /// routing policy actually in use, estimated by the doubly robust estimator on `records`
    /// decisions logged after the candidate was frozen, with its one-sided 95% lower bound.
    /// Observability only: routing is never changed by this event or by the mode.
    VcwShadowEstimate {
        scope: String,
        candidate: [u8; 32],
        records: u64,
        difference: f64,
        lower_bound: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct EventEnvelope {
    pub sequence: u64,
    pub event: RuntimeEvent,
}
