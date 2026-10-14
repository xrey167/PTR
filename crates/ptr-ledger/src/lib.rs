pub mod acknowledged;
pub mod anchor;
pub mod compaction;
mod file;
pub mod integrity;
pub mod retention;
pub use acknowledged::{AcknowledgedError, AcknowledgedLedger, Split, TailPolicy, TailRecovery};
pub use compaction::{
    CompactionDecision, CompactionFault, CompactionOutcome, CompactionPlan, LogPaths,
    RetentionFloor, RetentionPolicy,
};
pub use file::{FileLedger, LegacyLog, RecoverableLog};
pub use retention::{retains, ErasureAudit, OutOfReach, Retainer};

use ptr_types::{
    CapabilityId, CapsuleId, CommitIndex, Effect, EgressRefusalReason, FencingToken, Generation,
    MeshEndpointBinding, MeshRouteKind, MeshTunnelEventKind, MeshTunnelLifecycleEvent, NamespaceId,
    PeerId, PodAddress, PodId, PodRevisionAddress, ProjectId, RequestId, Revision, ScopeId,
    ScopeLeaseBinding, ScopeLifecycleEvent, ScopeLifecycleKind, SessionId, Timestamp, TraceId,
    TypeId, VerificationLevel,
};
use std::collections::BTreeSet;
use std::io;
#[cfg(feature = "raft-engine-backend")]
use std::path::Path;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerEvent {
    /// Opaque versioned semantic transaction. ptr-runtime validates its schema
    /// and both revisions before append and again during replay.
    SemanticDeltaCommitted {
        base_revision: Revision,
        revision: Revision,
        encoded_delta: Vec<u8>,
        /// Why the write was allowed, recorded in the same append as the
        /// delta. [`SemanticOrigin::Legacy`] encodes as tag 8, byte for byte
        /// as before origins existed; every other origin encodes as tag 12.
        origin: SemanticOrigin,
    },
    CapsuleCommitted {
        project: ProjectId,
        capsule: CapsuleId,
        generation: Generation,
    },
    CapsuleSuperseded {
        capsule: CapsuleId,
        old: Generation,
        new: Generation,
    },
    Revoked {
        subject: String,
        generation: Generation,
    },
    HardConstraintCommitted {
        key: String,
        generation: Generation,
    },
    VerifierAttested {
        subject: String,
        passed: bool,
    },
    ProcedurePromoted {
        id: String,
        generation: Generation,
    },
    ProcedureRevoked {
        id: String,
        generation: Generation,
    },
    SnapshotCommitted {
        revision: u64,
        covers: CommitIndex,
    },
    /// Committed *before* an external effect is attempted, so a crash inside
    /// the effect window leaves a durable record that the effect may already
    /// have applied. An attempt with no matching settlement *is* the fence:
    /// there is no separate flag to lose, and replay rebuilds it by definition.
    EffectAttempted {
        /// Caller-supplied at-most-once key, absent when the caller asked for
        /// no such guarantee. Only a caller knows whether a repeated request
        /// means "the same one again" or "do it once more".
        key: Option<String>,
        project: ProjectId,
        principal: String,
        target: String,
        operation: String,
        capability: CapabilityId,
        effect: Effect,
        generation: Generation,
        revision: Revision,
        /// The level that admitted this action. A record only exists after the
        /// verifier passed, so there is no status field that could disagree.
        verification: VerificationLevel,
        action_digest: [u8; 32],
    },
    /// The runtime observed the effect's own outcome and recorded it.
    EffectSettled {
        attempt: CommitIndex,
        /// Absent once the response exceeds [`MAX_RETAINED_RESPONSE`]. The
        /// digest is recorded either way, so "not retained" never becomes
        /// "not known".
        response: Option<Vec<u8>>,
        response_digest: [u8; 32],
    },
    /// An operator resolved an ambiguous attempt with evidence this runtime
    /// cannot obtain by itself. Reconciliation never guesses and never retries.
    EffectReconciled {
        attempt: CommitIndex,
        applied: bool,
        evidence: String,
    },
    /// Append-only execution-scope lifecycle transition. Scope state is
    /// materialized by ptr-runtime; the ledger only stores the authoritative
    /// transition and its bindings.
    ScopeLifecycle(ScopeLifecycleEvent),
    /// Durable reference to an encrypted state record. The ledger stores only
    /// binding metadata and digests; ciphertext and key material stay in the
    /// protected state store.
    ProtectedStateCommitted {
        domain: u8,
        logical_id: String,
        generation: Generation,
        revision: Revision,
        plaintext_digest: [u8; 32],
        ciphertext_digest: [u8; 32],
        anchor_digest: [u8; 32],
    },
    MeshTunnelLifecycle(MeshTunnelLifecycleEvent),
    /// Runtime-admitted execution manifest. The payload is a versioned
    /// canonical manifest owned by ptr-pods; the ledger stores it opaquely.
    ExecutionManifestAdmitted {
        manifest_digest: [u8; 32],
        generation: Generation,
        revision: Revision,
        policy_revision: Revision,
        manifest: Vec<u8>,
    },
    ExecutionManifestRevoked {
        manifest_digest: [u8; 32],
        generation: Generation,
        revision: Revision,
    },
    /// A verified, immutable policy bundle activated by the runtime.
    PolicyBundleActivated {
        revision: Revision,
        key_id: String,
        bundle_digest: [u8; 32],
        bundle: Vec<u8>,
    },
    /// A policy revision tombstone. Revocation remains effective on replay.
    PolicyBundleRevoked {
        revision: Revision,
        bundle_digest: [u8; 32],
        reason: String,
    },
    /// A durable session tombstone. Replayed runtimes must reject new
    /// stateful admission for this session; the tombstone is never undone.
    SessionRevoked {
        session_id: SessionId,
        reason: String,
    },
    /// Lifecycle of a configured storage-tier backend. Tier and state use
    /// stable protocol codes owned by ptr-storage/ptr-runtime respectively.
    TierBackendLifecycle {
        backend_id: String,
        tier: u8,
        state: u8,
        revision: Revision,
        event_digest: [u8; 32],
    },
    /// Immutable, canonical tier-object manifest committed by the runtime.
    TierObjectCommitted {
        root_digest: [u8; 32],
        generation: Generation,
        revision: Revision,
        manifest: Vec<u8>,
    },
    /// Lifecycle of one verified replica of a committed tier object.
    TierReplicaLifecycle {
        root_digest: [u8; 32],
        backend_id: String,
        tier: u8,
        state: u8,
        generation: Generation,
        revision: Revision,
        event_digest: [u8; 32],
    },
    /// Opaque, canonical Pod evidence. ptr-pods owns the schema and replay
    /// validation; the ledger owns ordering, durability and record integrity.
    PodEvidenceCommitted {
        session_id: SessionId,
        trace_id: TraceId,
        manifest_digest: [u8; 32],
        artifact_digest: [u8; 32],
        generation: Generation,
        revision: Revision,
        bundle_digest: [u8; 32],
        bundle: Vec<u8>,
    },
    /// Runtime admission record for a typed Pod output. The output payload is
    /// intentionally not stored here; the canonical digest and the later
    /// semantic/branch record remain the authoritative content records.
    PodOutputAdmitted {
        request_id: RequestId,
        session_id: SessionId,
        scope_id: ScopeId,
        pod_id: PodId,
        manifest_digest: [u8; 32],
        artifact_digest: [u8; 32],
        generation: Generation,
        revision: Revision,
        output_kind: u8,
        output_type: TypeId,
        output_digest: [u8; 32],
        verification: Attestation,
    },
    /// Durable, replayable hypothesis materialized from a Pod output. The
    /// payload is retained here because a later branch merge must be able to
    /// reconstruct the exact typed hypothesis without rerunning the Pod.
    PodHypothesisCommitted {
        request_id: RequestId,
        session_id: SessionId,
        scope_id: ScopeId,
        branch_id: String,
        pod_id: PodId,
        manifest_digest: [u8; 32],
        artifact_digest: [u8; 32],
        generation: Generation,
        revision: Revision,
        output_type: TypeId,
        output_digest: [u8; 32],
        payload: Vec<u8>,
        provenance: Vec<(String, Option<String>)>,
        dependencies: Vec<[u8; 32]>,
        confidence_bits: u32,
        latency_millis: u64,
        verification: Attestation,
    },
    /// A structural change to the expert set (birth, merge, split, freeze,
    /// reactivate, prune or retire) committed by the runtime's expert
    /// admission path. `record` is a versioned canonical expert commit record
    /// owned by ptr-pods; the ledger stores it opaquely and checks only its
    /// size. The ids and digests are the committing path's own claim about
    /// that record, carried so a reader can index the event without decoding
    /// the record; this crate cannot check that they agree with it. The
    /// admission path that writes this event MUST derive `proposal_id`,
    /// `operation_id` and `evidence_digest` from the decoded record, never
    /// take them from a caller, and replay MUST re-derive them from the
    /// record and compare, refusing a mismatch.
    ExpertStructuralCommitted {
        project: ProjectId,
        op_kind: ExpertOpKind,
        proposal_id: [u8; 32],
        operation_id: [u8; 32],
        evidence_digest: [u8; 32],
        record: Vec<u8>,
    },
    /// An authority fence of one expert, or its readmission, committed by the runtime's expert
    /// fence path. A fence withdraws the expert from serving without changing its generation
    /// (`Serving` or `Frozen` to `Fenced`); a readmission returns a fenced expert to the state it
    /// left (`resume`) at the same generation; an escalation keeps it fenced and tightens the
    /// cause; a retirement takes a fenced expert of cause `InputRevoked` to
    /// `Retired(InputRevoked)`, final. Each moves the expert's state version, which is its fence
    /// epoch, to `epoch`, so a cache entry recorded under an earlier epoch is never revived.
    /// `expected_state_version` is the compare-and-set operand and `epoch` must be exactly
    /// `expected_state_version + 1`: the codec refuses any other pair. The record is written
    /// only by the runtime's fence path under a typed grant; this crate cannot check that.
    ExpertFenceCommitted {
        project: ProjectId,
        pod: PodId,
        action: ExpertFenceAction,
        cause: ExpertFenceCause,
        /// The state the expert was in before the fence and returns to on readmission. A fence
        /// names the state it leaves (`Serving` or `Frozen`); an escalation and a readmission
        /// repeat the one recorded at the fence. Replay checks each against the registry.
        resume: ExpertFenceResume,
        generation: Generation,
        expected_state_version: u64,
        epoch: u64,
        /// The committed revocation an `InputRevoked` fence, escalation or retirement rests on;
        /// `None` for every other record. The codec requires `Some` for exactly those three.
        revocation: Option<ExpertFenceRevocation>,
        /// The gate evidence a readmission rests on; `None` for a fence, an escalation and a
        /// retirement, none of which waits for a gate. The codec requires `Some` for exactly a
        /// readmission.
        evidence: Option<ExpertFenceEvidence>,
    },
    /// A change to the approximate-KV taint registry, committed by the runtime's taint path
    /// before any result that rests on it is returned (taint first, so a crash leaves at most a
    /// stray taint record, which only refuses more). `subjects` lists one to
    /// [`MAX_KV_TAINT_SUBJECTS`] `(kind, digest)` pairs in strictly ascending canonical order
    /// (kind code, then digest bytes), without duplicates; the codec refuses any other form.
    /// The record is written only by the runtime's taint path; this crate cannot check that,
    /// nor that a `Clear` names a live `Record` (replay does).
    KvTaintCommitted {
        subjects: Vec<(KvTaintSubjectKind, [u8; 32])>,
        change: KvTaintChange,
    },
    /// The compound invalidation of one input (tag 30): `input` at `generation` is revoked or
    /// superseded and, in the same record, every expert that rests on it is fenced. The event
    /// names the invalidated input and commits to the set of experts it fences by count and
    /// digest only (`fence_count`, `fence_digest`), never by an explicit list: the set is
    /// derived from the runtime's dependency index at this commit index, so a large fan-out
    /// costs one small record. `binding_digest` is the content digest of the invalidated input
    /// at `generation` (the digest the first committed use of it fixed, when there was one).
    ///
    /// This crate checks only the shape: the identity is bounded, no digest is zero, a
    /// `Superseded` kind names `old_generation == generation` and `new_generation >
    /// old_generation`. It cannot check that the count and digest are the fence set; the
    /// runtime re-derives the set on commit and on replay and refuses a mismatch. The record is
    /// written only by the runtime's invalidation path; the public `commit` refuses it.
    InputInvalidated {
        input: String,
        generation: Generation,
        binding_digest: [u8; 32],
        kind: InputInvalidationKind,
        fence_count: u32,
        fence_digest: [u8; 32],
    },
    /// A change of the authority expert admission evidence rests on (tag 31): the current
    /// epoch and identity of the gate configuration or of the evaluator moves to `epoch` and
    /// `id_digest`. `class` says what the change does to evidence issued under earlier epochs: a
    /// `Revoke` or a `TighteningRotation` fences, in the same record, every expert whose
    /// admission evidence was issued under an epoch below `epoch`; a `NonTightening` rotation
    /// fences nothing. As for tag 30, the fenced set is committed to by count and digest only
    /// (`fence_count`, `fence_digest`): the runtime derives it from its registry at this commit
    /// index and re-derives it on replay.
    ///
    /// This crate checks only the shape: a non-zero epoch, no all-zero digest, and a
    /// `NonTightening` change that fences nobody. It cannot check that the epoch advances or that
    /// the count and digest are the fence set; the runtime does both on commit and on replay.
    /// The record is written only by the runtime's authority path; the public `commit` refuses it.
    ExpertAuthorityChanged {
        kind: ExpertAuthorityKind,
        id_digest: [u8; 32],
        epoch: u64,
        class: ExpertAuthorityClass,
        fence_count: u32,
        fence_digest: [u8; 32],
    },
    /// A Pod's outbound connection or publication was refused by its egress policy or its
    /// rate limit before anything left the host (tag 37). The record names the trace, the Pod
    /// revision that tried, the redacted `target` (`*`, a hashed `host:` or a hashed `topic:`, never a payload)
    /// and the reason, so the refusal is part of the replayable history rather than a log line.
    ///
    /// This crate checks only the shape: a non-empty trace, a valid source address and a
    /// bounded target without control characters or whitespace. Whether the refusal was
    /// warranted is the runtime's to say; the record is written by the runtime's egress path.
    EgressRefused {
        trace_id: TraceId,
        source: EgressRefusalSource,
        target: String,
        /// How many refusals of this source, target and reason the record stands for: the
        /// runtime coalesces a window's identical refusals into one event. At least one.
        count: u32,
        /// Whether the runtime had to stop counting: the true number of refusals is at least
        /// `count`, which is then `u32::MAX`. Only ever set together with `u32::MAX`.
        saturated: bool,
        reason: EgressRefusalReason,
    },
    /// An adapter checkpoint was promoted to serve (tag 39). The record binds the adapter's
    /// name in `project` and the new `generation` to the digest of the base model it modifies
    /// (`base`), the SHA-256 of its weights (`artifact_sha256`) and the digest of the bound
    /// gate report that admitted it (`gate_report_digest`). The runtime's promotion path
    /// recomputes `artifact_sha256` from the weights and derives the other digests from the
    /// decoded proposal before it appends; replay trusts none of them more than the record.
    ///
    /// This crate checks only the shape: bounded names without control characters, whitespace
    /// or `/`, a non-zero generation and no all-zero digest. The public `commit` of the runtime
    /// refuses the record; only `promote_adapter` writes it.
    AdapterPromoted {
        project: ProjectId,
        adapter: String,
        generation: Generation,
        base: [u8; 32],
        artifact_sha256: [u8; 32],
        gate_report_digest: [u8; 32],
    },
    /// A promoted adapter generation was revoked (tag 40). The tombstone is permanent: that
    /// generation of that adapter cannot serve again whatever cache, restart or replay does
    /// (Invariant 3). `reason` is a non-empty bounded description without control characters.
    /// Written only by the runtime's `revoke_adapter` path.
    AdapterRevoked {
        project: ProjectId,
        adapter: String,
        generation: Generation,
        reason: String,
    },
    /// One self-improvement cycle ended (tag 41). Every cycle that started is recorded, a
    /// completed one and an aborted one alike, because both consumed the budget the cycles of
    /// one `day` (days since the Unix epoch, from the runtime clock) are counted against. The
    /// `policy_digest` pins the improvement policy the cycle ran under. Written only by the
    /// runtime's `record_improvement_cycle` path.
    ImprovementCycleRecorded {
        project: ProjectId,
        day: u64,
        outcome: ImprovementCycleOutcome,
        policy_digest: [u8; 32],
    },
}

/// The longest adapter name or project an adapter event names, in bytes.
pub const MAX_ADAPTER_NAME_BYTES: usize = 256;

/// The longest revocation or abort reason an adapter or cycle event carries, in bytes.
pub const MAX_ADAPTER_REASON_BYTES: usize = 512;

/// How an [`LedgerEvent::ImprovementCycleRecorded`] cycle ended. Codec byte: Completed 0,
/// Aborted 1 (followed by the reason).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImprovementCycleOutcome {
    Completed,
    /// The cycle was stopped before it finished; it still used a unit of the budget.
    Aborted {
        reason: String,
    },
}

/// Who an [`LedgerEvent::EgressRefused`] counts refusals for. The overflow is a variant of its
/// own, not a made-up Pod address, so no real Pod can collide with it (in the codec, the
/// replay or the materialized state). Codec tag byte: Pod 0, Overflow 1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EgressRefusalSource {
    /// Refusals of one Pod revision.
    Pod(PodRevisionAddress),
    /// Refusals of Pods the runtime could no longer track individually while the ledger was
    /// refusing appends, folded per reason. Its target is always `*`.
    Overflow,
}

/// Which authority an [`LedgerEvent::ExpertAuthorityChanged`] moves. The codes are pinned data:
/// GateConfig 0, Evaluator 1.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ExpertAuthorityKind {
    /// The gate configuration admission evidence must have been produced under.
    GateConfig,
    /// The evaluator whose reports count as admission evidence.
    Evaluator,
}

impl ExpertAuthorityKind {
    /// The stored code: GateConfig 0, Evaluator 1.
    pub fn code(self) -> u8 {
        match self {
            Self::GateConfig => 0,
            Self::Evaluator => 1,
        }
    }

    /// The member for a stored code, or `None` for an unknown one.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::GateConfig),
            1 => Some(Self::Evaluator),
            _ => None,
        }
    }
}

/// What an [`LedgerEvent::ExpertAuthorityChanged`] does to evidence issued under earlier epochs.
/// The codes are pinned data: Revoke 0, TighteningRotation 1, NonTightening 2.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertAuthorityClass {
    /// The previous authority is withdrawn outright; its evidence no longer counts.
    Revoke,
    /// The authority rotates to a stricter one; evidence issued under an earlier epoch no longer
    /// meets it.
    TighteningRotation,
    /// The authority rotates without becoming stricter; earlier evidence keeps counting.
    NonTightening,
}

impl ExpertAuthorityClass {
    /// The stored code: Revoke 0, TighteningRotation 1, NonTightening 2.
    pub fn code(self) -> u8 {
        match self {
            Self::Revoke => 0,
            Self::TighteningRotation => 1,
            Self::NonTightening => 2,
        }
    }

    /// The member for a stored code, or `None` for an unknown one.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Revoke),
            1 => Some(Self::TighteningRotation),
            2 => Some(Self::NonTightening),
            _ => None,
        }
    }

    /// Whether evidence issued under an earlier epoch stops counting.
    pub fn fences_earlier_evidence(self) -> bool {
        match self {
            Self::Revoke | Self::TighteningRotation => true,
            Self::NonTightening => false,
        }
    }
}

/// The longest identity an [`LedgerEvent::InputInvalidated`] names, in bytes.
pub const MAX_INPUT_IDENTITY_BYTES: usize = 256;

/// The longest target an [`LedgerEvent::EgressRefused`] names, in bytes.
pub const MAX_EGRESS_TARGET_BYTES: usize = 256;

/// How an [`LedgerEvent::InputInvalidated`] ends the life of its input. The codes are pinned
/// data: Revoked 0, Superseded 1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputInvalidationKind {
    /// The input generation is revoked: no record may rest on it again.
    Revoked,
    /// The input moves from `old_generation` (the event's `generation`) to `new_generation`,
    /// whose content digest is `new_digest`. Every generation below `new_generation` is stale
    /// from here on. `replacement_ref` is a digest of the replacement artifact, kept as
    /// evidence only: the ledger and the runtime never dereference it.
    Superseded {
        old_generation: Generation,
        new_generation: Generation,
        new_digest: [u8; 32],
        replacement_ref: [u8; 32],
    },
}

impl InputInvalidationKind {
    /// The stored code: Revoked 0, Superseded 1.
    pub fn code(self) -> u8 {
        match self {
            Self::Revoked => 0,
            Self::Superseded { .. } => 1,
        }
    }
}

/// What a [`LedgerEvent::KvTaintCommitted`] subject digest identifies. The codes are pinned data.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum KvTaintSubjectKind {
    /// An action identity.
    Action,
    /// A derivation (a minted derived segment).
    Derivation,
    /// A content digest.
    Content,
    /// A Pod output.
    PodOutput,
    /// A request.
    Request,
}

impl KvTaintSubjectKind {
    pub const fn code(self) -> u8 {
        match self {
            Self::Action => 0,
            Self::Derivation => 1,
            Self::Content => 2,
            Self::PodOutput => 3,
            Self::Request => 4,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Action),
            1 => Some(Self::Derivation),
            2 => Some(Self::Content),
            3 => Some(Self::PodOutput),
            4 => Some(Self::Request),
            _ => None,
        }
    }

    /// The stable lowercase name used in projection keys.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Action => "action",
            Self::Derivation => "derivation",
            Self::Content => "content",
            Self::PodOutput => "pod-output",
            Self::Request => "request",
        }
    }
}

/// The change a [`LedgerEvent::KvTaintCommitted`] makes to every one of its subjects. All
/// digests are non-zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvTaintChange {
    /// The subjects are tainted by an approximate run: the reuse report and the recipe it ran.
    Record {
        report_digest: [u8; 32],
        recipe_digest: [u8; 32],
    },
    /// A cold exact recomputation of the same recipe cleared the subjects.
    Clear {
        recipe_digest: [u8; 32],
        recompute_digest: [u8; 32],
    },
}

/// A taint record names at most this many subjects.
pub const MAX_KV_TAINT_SUBJECTS: usize = 16;

/// The committed [`LedgerEvent::Revoked`] (or `ProcedureRevoked`) record an `InputRevoked` fence
/// names: its subject and generation. The runtime checks it against the revocations committed
/// at or before the fence, at commit and on replay; a fence that names none is refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertFenceRevocation {
    pub subject: String,
    pub generation: Generation,
}

/// The state a fenced expert returns to on readmission. The codes are pinned data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertFenceResume {
    Serving,
    Frozen,
}

impl ExpertFenceResume {
    /// The stored code: Serving 0, Frozen 1.
    pub fn code(self) -> u8 {
        match self {
            Self::Serving => 0,
            Self::Frozen => 1,
        }
    }

    /// The member for a stored code, or `None` for an unknown one.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Serving),
            1 => Some(Self::Frozen),
            _ => None,
        }
    }
}

/// The gate evidence embedded in a readmission: the canonical bytes of the bound gate report
/// (owned by ptr-lineage, stored opaquely here and size-checked like an expert record), the
/// gate epoch it was admitted under and the gate configuration digest the grant required for
/// that epoch. Replay re-verifies the report against these and against the event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertFenceEvidence {
    pub gate_epoch: u64,
    pub required_gate_config_digest: [u8; 32],
    pub report: Vec<u8>,
}

/// Whether an [`LedgerEvent::ExpertFenceCommitted`] fences an expert or readmits a fenced one.
/// The codes are pinned data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertFenceAction {
    Fence,
    Readmit,
    /// A `Fenced` expert whose cause is `AuthorityQuarantine` is re-fenced for `InputRevoked`:
    /// the cause only ever tightens, and the epoch moves like for any fence.
    Escalate,
    /// A `Fenced` expert whose cause is `InputRevoked` is retired for good
    /// (`Retired(InputRevoked)`, terminal): the runtime's revocation authority, never a gate,
    /// writes it, it carries no gate evidence and it rests on a committed revocation like the
    /// fence it follows. The epoch moves like for any fence action.
    Retire,
}

impl ExpertFenceAction {
    /// The stored code: Fence 0, Readmit 1, Escalate 2, Retire 3.
    pub fn code(self) -> u8 {
        match self {
            Self::Fence => 0,
            Self::Readmit => 1,
            Self::Escalate => 2,
            Self::Retire => 3,
        }
    }

    /// The member for a stored code, or `None` for an unknown one.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Fence),
            1 => Some(Self::Readmit),
            2 => Some(Self::Escalate),
            3 => Some(Self::Retire),
            _ => None,
        }
    }
}

/// Why an expert is fenced. A readmission names the cause of the fence it clears. The codes are
/// pinned data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertFenceCause {
    /// An input the expert was trained on was revoked. Never readmittable.
    InputRevoked,
    /// An authority the expert's admission rested on was withdrawn or quarantined. Readmittable.
    AuthorityQuarantine,
}

impl ExpertFenceCause {
    /// The stored code: InputRevoked 0, AuthorityQuarantine 1.
    pub fn code(self) -> u8 {
        match self {
            Self::InputRevoked => 0,
            Self::AuthorityQuarantine => 1,
        }
    }

    /// The member for a stored code, or `None` for an unknown one.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::InputRevoked),
            1 => Some(Self::AuthorityQuarantine),
            _ => None,
        }
    }
}

/// Which structural operation an [`LedgerEvent::ExpertStructuralCommitted`]
/// records. The codes are pinned data ([`ExpertOpKind::code`]); inserting a
/// member must not renumber stored records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertOpKind {
    Birth,
    Merge,
    Split,
    Freeze,
    Reactivate,
    Prune,
    Retire,
}

impl ExpertOpKind {
    /// The stored code: Birth 0, Merge 1, Split 2, Freeze 3, Reactivate 4,
    /// Prune 5, Retire 6.
    pub fn code(self) -> u8 {
        match self {
            Self::Birth => 0,
            Self::Merge => 1,
            Self::Split => 2,
            Self::Freeze => 3,
            Self::Reactivate => 4,
            Self::Prune => 5,
            Self::Retire => 6,
        }
    }

    /// The member for a stored code, or `None` for an unknown one.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Birth),
            1 => Some(Self::Merge),
            2 => Some(Self::Split),
            3 => Some(Self::Freeze),
            4 => Some(Self::Reactivate),
            5 => Some(Self::Prune),
            6 => Some(Self::Retire),
            _ => None,
        }
    }
}

/// An expert structural record is at most this many bytes, far below
/// [`integrity::MAX_RECORD_BYTES`] so one record cannot crowd out its log.
pub const MAX_EXPERT_RECORD_BYTES: usize = 1 << 20;

/// Why a semantic write was allowed. ptr-runtime writes it in the same append
/// as the delta and checks it with the same function on append and on every
/// rebuild; the codec checks only its shape.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticOrigin {
    /// Tag 8, byte for byte as records were written before origins existed.
    /// ptr-runtime never writes it, and replays it only before the first
    /// record with an attributed origin. Not the unframed PTRLOG01 format that
    /// `migrate_legacy_log` reads: a record of that format becomes a framed
    /// tag-8 record when it is migrated.
    Legacy,
    /// The raw text of one request, written by ingest.
    Request { request: String },
    /// One Pod's output for a request, promoted after the Pod's verifier
    /// passed with no hard finding, at the level it reported.
    PodOutput {
        request: String,
        pod: String,
        level: VerificationLevel,
    },
    /// A non-authoritative observation/candidate retained for later semantic
    /// review. It is distinct from `PodOutput`: candidates must never be read
    /// as an already promoted fact.
    PodCandidate {
        request: String,
        pod: String,
        output_digest: [u8; 32],
        level: VerificationLevel,
    },
    /// A write by the host, admitted by the installed grant's verifiers.
    Host {
        principal: String,
        verification: Attestation,
    },
    /// A sealed branch the runtime certified, verified and merged.
    Merge(MergeRecord),
}

/// What admitted an attributed write: the level required, the weakest level
/// the grant's verifiers reported, their names in grant order and their soft
/// finding codes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Attestation {
    /// `FullSemantic` or `Deterministic`.
    pub required: VerificationLevel,
    /// The weakest level among the verifiers.
    pub level: VerificationLevel,
    /// 1 to [`MAX_ATTESTATION_VERIFIERS`] names, in grant order.
    pub verifiers: Vec<String>,
    /// Up to [`MAX_ATTESTATION_FINDINGS`] soft-finding codes, sorted and
    /// distinct, each `<verifier>/<code>`.
    pub findings: Vec<String>,
}

/// What a merge record binds: the branch and its author, the seal, plan and
/// dependency digests the runtime computed, the keys certification rebased,
/// the verification and the authority that let the merge commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergeRecord {
    pub branch: String,
    pub author: String,
    /// The sealed branch's digest (`SealedBranch::seal_digest`).
    pub seal: [u8; 32],
    /// The merge plan's digest, recomputable from this record and its delta.
    pub plan: [u8; 32],
    pub dependencies: [u8; 32],
    /// Keys certification rebased onto the target, which the plan digest
    /// covers; empty for a clean merge. At most [`MAX_REBASED_KEYS`].
    pub rebased: BTreeSet<String>,
    pub verification: Attestation,
    pub authority: MergeAuthorityRecord,
}

/// Why a verified merge was allowed to commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MergeAuthorityRecord {
    /// The runtime's own triage under the grant's policy returned
    /// AutoPropose; the score it was given, as `f32` bits.
    Triage {
        policy_version: String,
        score_bits: u32,
    },
    /// A reviewer the grant lists approved exactly this plan.
    Reviewed { reviewer: String },
}

/// Most verifiers an [`Attestation`] names.
pub const MAX_ATTESTATION_VERIFIERS: usize = 16;
/// Most soft-finding codes an [`Attestation`] carries.
pub const MAX_ATTESTATION_FINDINGS: usize = 32;
/// Most rebased keys a [`MergeRecord`] carries: the most items a semantic
/// delta can hold (`ptr_semdb::MAX_DELTA_ITEMS`), since a key is rebased only
/// where the delta writes it or leaves it equal to the target.
pub const MAX_REBASED_KEYS: usize = 16_384;
/// Most provenance entries a `PodHypothesisCommitted` record carries.
pub const MAX_HYPOTHESIS_PROVENANCE: usize = 1024;
/// Most dependency digests a `PodHypothesisCommitted` record carries.
pub const MAX_HYPOTHESIS_DEPENDENCIES: usize = 1024;

/// Largest effect response copied into the journal. Above it only the digest is
/// kept: an unbounded history is its own failure, and a silently truncated
/// response would be a second, wrong answer rather than a missing one.
pub const MAX_RETAINED_RESPONSE: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommittedEvent {
    pub index: CommitIndex,
    pub event: LedgerEvent,
}

pub trait Ledger {
    /// Append one event and return the index it was committed at.
    ///
    /// Fallible because a commit index can run out. Returning the index
    /// unconditionally would force an implementation to invent one at the
    /// ceiling, and the only values available there repeat an index already
    /// handed out — which is worse than refusing, since two records would then
    /// claim the same position in a history that is supposed to order them.
    fn append(&mut self, event: LedgerEvent) -> io::Result<CommitIndex>;
    fn events(&self) -> &[CommittedEvent];
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryLedger {
    base: CommitIndex,
    events: Vec<CommittedEvent>,
}

impl InMemoryLedger {
    /// A ledger that continues above a compaction floor.
    ///
    /// Restoring from a compacted snapshot must not renumber the retained records
    /// from 1: their indices are part of the committed history the snapshot's
    /// floor refers to, and renumbering them would make the two disagree.
    pub fn resuming_above(base: CommitIndex) -> Self {
        Self {
            base,
            events: Vec::new(),
        }
    }

    /// The compaction floor this ledger continues above.
    pub fn base(&self) -> CommitIndex {
        self.base
    }
}

impl Ledger for InMemoryLedger {
    fn append(&mut self, event: LedgerEvent) -> io::Result<CommitIndex> {
        // Checked rather than saturating: saturation hands the ceiling out
        // twice, so the second record silently claims a position the first one
        // already holds. The index is computed before anything is stored, so a
        // refused append leaves the ledger exactly as it was.
        let index = self
            .base
            .0
            .checked_add(self.events.len() as u64)
            .and_then(|count| count.checked_add(1))
            .map(CommitIndex)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "PTR_LEDGER_INDEX_EXHAUSTED")
            })?;
        self.events.push(CommittedEvent { index, event });
        Ok(index)
    }

    fn events(&self) -> &[CommittedEvent] {
        &self.events
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionBarrier {
    pub snapshot_covers: CommitIndex,
    pub all_consumers_caught_up: bool,
    pub unresolved_revocations: usize,
}

impl CompactionBarrier {
    pub fn safe(&self) -> bool {
        self.all_consumers_caught_up && self.unresolved_revocations == 0
    }
}

fn encode_event(event: &LedgerEvent) -> Vec<u8> {
    let mut out = Vec::new();
    match event {
        LedgerEvent::SemanticDeltaCommitted {
            base_revision,
            revision,
            encoded_delta,
            origin,
        } => {
            // Legacy stays tag 8, byte for byte, so every log and anchor
            // written before origins existed keeps its exact encoding.
            out.push(if *origin == SemanticOrigin::Legacy {
                8
            } else {
                12
            });
            put_u64(&mut out, base_revision.0);
            put_u64(&mut out, revision.0);
            put_bytes(&mut out, encoded_delta);
            put_origin(&mut out, origin);
        }
        LedgerEvent::CapsuleCommitted {
            project,
            capsule,
            generation,
        } => {
            out.push(0);
            put_string(&mut out, &project.to_string());
            put_string(&mut out, &capsule.to_string());
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::CapsuleSuperseded { capsule, old, new } => {
            out.push(1);
            put_string(&mut out, &capsule.to_string());
            put_u64(&mut out, old.0);
            put_u64(&mut out, new.0);
        }
        LedgerEvent::Revoked {
            subject,
            generation,
        } => {
            out.push(2);
            put_string(&mut out, subject);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::HardConstraintCommitted { key, generation } => {
            out.push(3);
            put_string(&mut out, key);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::VerifierAttested { subject, passed } => {
            out.push(4);
            put_string(&mut out, subject);
            out.push(u8::from(*passed));
        }
        LedgerEvent::ProcedurePromoted { id, generation } => {
            out.push(5);
            put_string(&mut out, id);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::ProcedureRevoked { id, generation } => {
            out.push(6);
            put_string(&mut out, id);
            put_u64(&mut out, generation.0);
        }
        LedgerEvent::SnapshotCommitted { revision, covers } => {
            out.push(7);
            put_u64(&mut out, *revision);
            put_u64(&mut out, covers.0);
        }
        LedgerEvent::EffectAttempted {
            key,
            project,
            principal,
            target,
            operation,
            capability,
            effect,
            generation,
            revision,
            verification,
            action_digest,
        } => {
            out.push(9);
            put_optional_string(&mut out, key.as_deref());
            put_string(&mut out, &project.to_string());
            put_string(&mut out, principal);
            put_string(&mut out, target);
            put_string(&mut out, operation);
            put_string(&mut out, &capability.to_string());
            out.push(effect_code(*effect));
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            out.push(verification_code(*verification));
            out.extend_from_slice(action_digest);
        }
        LedgerEvent::EffectSettled {
            attempt,
            response,
            response_digest,
        } => {
            out.push(10);
            put_u64(&mut out, attempt.0);
            match response {
                Some(bytes) => {
                    out.push(1);
                    put_bytes(&mut out, bytes);
                }
                None => out.push(0),
            }
            out.extend_from_slice(response_digest);
        }
        LedgerEvent::EffectReconciled {
            attempt,
            applied,
            evidence,
        } => {
            out.push(11);
            put_u64(&mut out, attempt.0);
            out.push(u8::from(*applied));
            put_string(&mut out, evidence);
        }
        LedgerEvent::ScopeLifecycle(event) => {
            out.push(13);
            put_scope_lifecycle(&mut out, event);
        }
        LedgerEvent::ProtectedStateCommitted {
            domain,
            logical_id,
            generation,
            revision,
            plaintext_digest,
            ciphertext_digest,
            anchor_digest,
        } => {
            out.push(14);
            out.push(*domain);
            put_string(&mut out, logical_id);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            out.extend_from_slice(plaintext_digest);
            out.extend_from_slice(ciphertext_digest);
            out.extend_from_slice(anchor_digest);
        }
        LedgerEvent::MeshTunnelLifecycle(event) => {
            out.push(15);
            put_string(&mut out, &event.network_id.0);
            put_string(&mut out, &event.peer_id.0);
            put_optional_string(
                &mut out,
                event.invitation_id.as_ref().map(|id| id.0.as_str()),
            );
            put_optional_string(&mut out, event.tunnel_id.as_deref());
            put_u64(&mut out, event.generation.0);
            put_u64(&mut out, event.revision.0);
            out.push(mesh_event_kind_code(event.kind));
            put_mesh_endpoint(&mut out, event.endpoint.as_ref());
            put_u64(&mut out, event.placement_epoch);
            out.extend_from_slice(&event.fencing_token.0.to_le_bytes());
            out.extend_from_slice(&event.event_digest);
        }
        LedgerEvent::ExecutionManifestAdmitted {
            manifest_digest,
            generation,
            revision,
            policy_revision,
            manifest,
        } => {
            out.push(16);
            out.extend_from_slice(manifest_digest);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            put_u64(&mut out, policy_revision.0);
            put_bytes(&mut out, manifest);
        }
        LedgerEvent::ExpertStructuralCommitted {
            project,
            op_kind,
            proposal_id,
            operation_id,
            evidence_digest,
            record,
        } => {
            out.push(27);
            put_string(&mut out, &project.0);
            out.push(op_kind.code());
            out.extend_from_slice(proposal_id);
            out.extend_from_slice(operation_id);
            out.extend_from_slice(evidence_digest);
            put_bytes(&mut out, record);
        }
        LedgerEvent::ExpertFenceCommitted {
            project,
            pod,
            action,
            cause,
            resume,
            generation,
            expected_state_version,
            epoch,
            revocation,
            evidence,
        } => {
            out.push(28);
            put_string(&mut out, &project.0);
            put_string(&mut out, &pod.0);
            out.push(action.code());
            out.push(cause.code());
            out.push(resume.code());
            put_u64(&mut out, generation.0);
            put_u64(&mut out, *expected_state_version);
            put_u64(&mut out, *epoch);
            match revocation {
                None => out.push(0),
                Some(revocation) => {
                    out.push(1);
                    put_string(&mut out, &revocation.subject);
                    put_u64(&mut out, revocation.generation.0);
                }
            }
            match evidence {
                None => out.push(0),
                Some(evidence) => {
                    out.push(1);
                    put_u64(&mut out, evidence.gate_epoch);
                    out.extend_from_slice(&evidence.required_gate_config_digest);
                    put_bytes(&mut out, &evidence.report);
                }
            }
        }
        LedgerEvent::KvTaintCommitted { subjects, change } => {
            out.push(29);
            out.push(subjects.len().min(usize::from(u8::MAX)) as u8);
            for (kind, digest) in subjects {
                out.push(kind.code());
                out.extend_from_slice(digest);
            }
            match change {
                KvTaintChange::Record {
                    report_digest,
                    recipe_digest,
                } => {
                    out.push(0);
                    out.extend_from_slice(report_digest);
                    out.extend_from_slice(recipe_digest);
                }
                KvTaintChange::Clear {
                    recipe_digest,
                    recompute_digest,
                } => {
                    out.push(1);
                    out.extend_from_slice(recipe_digest);
                    out.extend_from_slice(recompute_digest);
                }
            }
        }
        LedgerEvent::InputInvalidated {
            input,
            generation,
            binding_digest,
            kind,
            fence_count,
            fence_digest,
        } => {
            out.push(30);
            put_string(&mut out, input);
            put_u64(&mut out, generation.0);
            out.extend_from_slice(binding_digest);
            out.push(kind.code());
            if let InputInvalidationKind::Superseded {
                old_generation,
                new_generation,
                new_digest,
                replacement_ref,
            } = kind
            {
                put_u64(&mut out, old_generation.0);
                put_u64(&mut out, new_generation.0);
                out.extend_from_slice(new_digest);
                out.extend_from_slice(replacement_ref);
            }
            put_u32(&mut out, *fence_count);
            out.extend_from_slice(fence_digest);
        }
        LedgerEvent::ExpertAuthorityChanged {
            kind,
            id_digest,
            epoch,
            class,
            fence_count,
            fence_digest,
        } => {
            out.push(31);
            out.push(kind.code());
            out.extend_from_slice(id_digest);
            put_u64(&mut out, *epoch);
            out.push(class.code());
            put_u32(&mut out, *fence_count);
            out.extend_from_slice(fence_digest);
        }
        LedgerEvent::ExecutionManifestRevoked {
            manifest_digest,
            generation,
            revision,
        } => {
            out.push(17);
            out.extend_from_slice(manifest_digest);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
        }
        LedgerEvent::PodEvidenceCommitted {
            session_id,
            trace_id,
            manifest_digest,
            artifact_digest,
            generation,
            revision,
            bundle_digest,
            bundle,
        } => {
            out.push(18);
            put_string(&mut out, &session_id.0);
            put_string(&mut out, &trace_id.0);
            out.extend_from_slice(manifest_digest);
            out.extend_from_slice(artifact_digest);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            out.extend_from_slice(bundle_digest);
            put_bytes(&mut out, bundle);
        }
        LedgerEvent::PodOutputAdmitted {
            request_id,
            session_id,
            scope_id,
            pod_id,
            manifest_digest,
            artifact_digest,
            generation,
            revision,
            output_kind,
            output_type,
            output_digest,
            verification,
        } => {
            out.push(19);
            put_string(&mut out, &request_id.0);
            put_string(&mut out, &session_id.0);
            put_string(&mut out, &scope_id.0);
            put_string(&mut out, &pod_id.0);
            out.extend_from_slice(manifest_digest);
            out.extend_from_slice(artifact_digest);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            out.push(*output_kind);
            put_string(&mut out, &output_type.0);
            out.extend_from_slice(output_digest);
            put_attestation(&mut out, verification);
        }
        LedgerEvent::PodHypothesisCommitted {
            request_id,
            session_id,
            scope_id,
            branch_id,
            pod_id,
            manifest_digest,
            artifact_digest,
            generation,
            revision,
            output_type,
            output_digest,
            payload,
            provenance,
            dependencies,
            confidence_bits,
            latency_millis,
            verification,
        } => {
            out.push(20);
            put_string(&mut out, &request_id.0);
            put_string(&mut out, &session_id.0);
            put_string(&mut out, &scope_id.0);
            put_string(&mut out, branch_id);
            put_string(&mut out, &pod_id.0);
            out.extend_from_slice(manifest_digest);
            out.extend_from_slice(artifact_digest);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            put_string(&mut out, &output_type.0);
            out.extend_from_slice(output_digest);
            put_bytes(&mut out, payload);
            put_u32(&mut out, provenance.len() as u32);
            for (source, note) in provenance {
                put_string(&mut out, source);
                put_optional_string(&mut out, note.as_deref());
            }
            put_u32(&mut out, dependencies.len() as u32);
            for dependency in dependencies {
                out.extend_from_slice(dependency);
            }
            out.extend_from_slice(&confidence_bits.to_le_bytes());
            put_u64(&mut out, *latency_millis);
            put_attestation(&mut out, verification);
        }
        LedgerEvent::PolicyBundleActivated {
            revision,
            key_id,
            bundle_digest,
            bundle,
        } => {
            out.push(21);
            put_u64(&mut out, revision.0);
            put_string(&mut out, key_id);
            out.extend_from_slice(bundle_digest);
            put_bytes(&mut out, bundle);
        }
        LedgerEvent::PolicyBundleRevoked {
            revision,
            bundle_digest,
            reason,
        } => {
            out.push(22);
            put_u64(&mut out, revision.0);
            out.extend_from_slice(bundle_digest);
            put_string(&mut out, reason);
        }
        LedgerEvent::SessionRevoked { session_id, reason } => {
            out.push(23);
            put_string(&mut out, &session_id.0);
            put_string(&mut out, reason);
        }
        LedgerEvent::TierBackendLifecycle {
            backend_id,
            tier,
            state,
            revision,
            event_digest,
        } => {
            out.push(24);
            put_string(&mut out, backend_id);
            out.push(*tier);
            out.push(*state);
            put_u64(&mut out, revision.0);
            out.extend_from_slice(event_digest);
        }
        LedgerEvent::TierObjectCommitted {
            root_digest,
            generation,
            revision,
            manifest,
        } => {
            out.push(25);
            out.extend_from_slice(root_digest);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            put_bytes(&mut out, manifest);
        }
        LedgerEvent::TierReplicaLifecycle {
            root_digest,
            backend_id,
            tier,
            state,
            generation,
            revision,
            event_digest,
        } => {
            out.push(26);
            out.extend_from_slice(root_digest);
            put_string(&mut out, backend_id);
            out.push(*tier);
            out.push(*state);
            put_u64(&mut out, generation.0);
            put_u64(&mut out, revision.0);
            out.extend_from_slice(event_digest);
        }
        LedgerEvent::EgressRefused {
            trace_id,
            source,
            target,
            count,
            saturated,
            reason,
        } => {
            out.push(37);
            put_string(&mut out, &trace_id.0);
            match source {
                EgressRefusalSource::Pod(source) => {
                    out.push(0);
                    put_string(&mut out, &source.address.project.0);
                    put_string(&mut out, &source.address.namespace.0);
                    put_string(&mut out, &source.address.pod_id.0);
                    out.extend_from_slice(&source.semantic_revision);
                    put_u64(&mut out, source.generation.0);
                }
                EgressRefusalSource::Overflow => out.push(1),
            }
            put_string(&mut out, target);
            put_u32(&mut out, *count);
            out.push(u8::from(*saturated));
            out.push(reason.code());
        }
        LedgerEvent::AdapterPromoted {
            project,
            adapter,
            generation,
            base,
            artifact_sha256,
            gate_report_digest,
        } => {
            out.push(39);
            put_string(&mut out, &project.0);
            put_string(&mut out, adapter);
            put_u64(&mut out, generation.0);
            out.extend_from_slice(base);
            out.extend_from_slice(artifact_sha256);
            out.extend_from_slice(gate_report_digest);
        }
        LedgerEvent::AdapterRevoked {
            project,
            adapter,
            generation,
            reason,
        } => {
            out.push(40);
            put_string(&mut out, &project.0);
            put_string(&mut out, adapter);
            put_u64(&mut out, generation.0);
            put_string(&mut out, reason);
        }
        LedgerEvent::ImprovementCycleRecorded {
            project,
            day,
            outcome,
            policy_digest,
        } => {
            out.push(41);
            put_string(&mut out, &project.0);
            put_u64(&mut out, *day);
            match outcome {
                ImprovementCycleOutcome::Completed => out.push(0),
                ImprovementCycleOutcome::Aborted { reason } => {
                    out.push(1);
                    put_string(&mut out, reason);
                }
            }
            out.extend_from_slice(policy_digest);
        }
    }
    out
}

fn decode_event(payload: &[u8]) -> io::Result<LedgerEvent> {
    let mut cursor = Cursor::new(payload);
    let tag = cursor.u8()?;
    let event = match tag {
        8 => LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            encoded_delta: cursor.bytes()?.to_vec(),
            origin: SemanticOrigin::Legacy,
        },
        12 => LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            encoded_delta: cursor.bytes()?.to_vec(),
            origin: attributed_origin(&mut cursor)?,
        },
        0 => LedgerEvent::CapsuleCommitted {
            project: ProjectId(cursor.string()?),
            capsule: CapsuleId(cursor.string()?),
            generation: Generation(cursor.u64()?),
        },
        1 => LedgerEvent::CapsuleSuperseded {
            capsule: CapsuleId(cursor.string()?),
            old: Generation(cursor.u64()?),
            new: Generation(cursor.u64()?),
        },
        2 => LedgerEvent::Revoked {
            subject: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        3 => LedgerEvent::HardConstraintCommitted {
            key: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        4 => LedgerEvent::VerifierAttested {
            subject: cursor.string()?,
            passed: match cursor.u8()? {
                0 => false,
                1 => true,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid bool byte {other}"),
                    ))
                }
            },
        },
        5 => LedgerEvent::ProcedurePromoted {
            id: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        6 => LedgerEvent::ProcedureRevoked {
            id: cursor.string()?,
            generation: Generation(cursor.u64()?),
        },
        7 => LedgerEvent::SnapshotCommitted {
            revision: cursor.u64()?,
            covers: CommitIndex(cursor.u64()?),
        },
        9 => LedgerEvent::EffectAttempted {
            key: cursor.optional_string()?,
            project: ProjectId(cursor.string()?),
            principal: cursor.string()?,
            target: cursor.string()?,
            operation: cursor.string()?,
            capability: CapabilityId(cursor.string()?),
            effect: effect_from_code(cursor.u8()?)?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            verification: verification_from_code(cursor.u8()?)?,
            action_digest: cursor.digest()?,
        },
        10 => LedgerEvent::EffectSettled {
            attempt: CommitIndex(cursor.u64()?),
            response: match cursor.u8()? {
                0 => None,
                1 => Some(cursor.bytes()?.to_vec()),
                other => return Err(presence_byte(other)),
            },
            response_digest: cursor.digest()?,
        },
        11 => LedgerEvent::EffectReconciled {
            attempt: CommitIndex(cursor.u64()?),
            applied: match cursor.u8()? {
                0 => false,
                1 => true,
                other => return Err(presence_byte(other)),
            },
            evidence: cursor.string()?,
        },
        13 => LedgerEvent::ScopeLifecycle(read_scope_lifecycle(&mut cursor)?),
        14 => LedgerEvent::ProtectedStateCommitted {
            domain: cursor.u8()?,
            logical_id: cursor.string()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            plaintext_digest: cursor.digest()?,
            ciphertext_digest: cursor.digest()?,
            anchor_digest: cursor.digest()?,
        },
        15 => LedgerEvent::MeshTunnelLifecycle(MeshTunnelLifecycleEvent {
            network_id: ptr_types::NetworkId(cursor.string()?),
            peer_id: PeerId(cursor.string()?),
            invitation_id: cursor.optional_string()?.map(ptr_types::InvitationId),
            tunnel_id: cursor.optional_string()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            kind: mesh_event_kind_from_code(cursor.u8()?)?,
            endpoint: read_mesh_endpoint(&mut cursor)?,
            placement_epoch: cursor.u64()?,
            fencing_token: FencingToken(cursor.u128()?),
            event_digest: cursor.digest()?,
        }),
        16 => LedgerEvent::ExecutionManifestAdmitted {
            manifest_digest: cursor.digest()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            policy_revision: Revision(cursor.u64()?),
            manifest: cursor.bytes()?.to_vec(),
        },
        17 => LedgerEvent::ExecutionManifestRevoked {
            manifest_digest: cursor.digest()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
        },
        18 => LedgerEvent::PodEvidenceCommitted {
            session_id: SessionId(cursor.string()?),
            trace_id: TraceId(cursor.string()?),
            manifest_digest: cursor.digest()?,
            artifact_digest: cursor.digest()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            bundle_digest: cursor.digest()?,
            bundle: cursor.bytes()?.to_vec(),
        },
        19 => LedgerEvent::PodOutputAdmitted {
            request_id: RequestId(cursor.string()?),
            session_id: SessionId(cursor.string()?),
            scope_id: ScopeId(cursor.string()?),
            pod_id: PodId(cursor.string()?),
            manifest_digest: cursor.digest()?,
            artifact_digest: cursor.digest()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            output_kind: cursor.u8()?,
            output_type: TypeId(cursor.string()?),
            output_digest: cursor.digest()?,
            verification: attestation(&mut cursor)?,
        },
        20 => {
            let request_id = RequestId(cursor.string()?);
            let session_id = SessionId(cursor.string()?);
            let scope_id = ScopeId(cursor.string()?);
            let branch_id = cursor.string()?;
            let pod_id = PodId(cursor.string()?);
            let manifest_digest = cursor.digest()?;
            let artifact_digest = cursor.digest()?;
            let generation = Generation(cursor.u64()?);
            let revision = Revision(cursor.u64()?);
            let output_type = TypeId(cursor.string()?);
            let output_digest = cursor.digest()?;
            let payload = cursor.bytes()?.to_vec();
            let count = cursor.u32()? as usize;
            check_hypothesis_provenance_count(count)?;
            let mut provenance = Vec::with_capacity(count);
            for _ in 0..count {
                provenance.push((cursor.string()?, cursor.optional_string()?));
            }
            let dependency_count = cursor.u32()? as usize;
            check_hypothesis_dependency_count(dependency_count)?;
            let mut dependencies = Vec::with_capacity(dependency_count);
            for _ in 0..dependency_count {
                dependencies.push(cursor.digest()?);
            }
            let confidence_bits = cursor.u32()?;
            let latency_millis = cursor.u64()?;
            let verification = attestation(&mut cursor)?;
            LedgerEvent::PodHypothesisCommitted {
                request_id,
                session_id,
                scope_id,
                branch_id,
                pod_id,
                manifest_digest,
                artifact_digest,
                generation,
                revision,
                output_type,
                output_digest,
                payload,
                provenance,
                dependencies,
                confidence_bits,
                latency_millis,
                verification,
            }
        }
        21 => LedgerEvent::PolicyBundleActivated {
            revision: Revision(cursor.u64()?),
            key_id: cursor.string()?,
            bundle_digest: cursor.digest()?,
            bundle: cursor.bytes()?.to_vec(),
        },
        22 => LedgerEvent::PolicyBundleRevoked {
            revision: Revision(cursor.u64()?),
            bundle_digest: cursor.digest()?,
            reason: cursor.string()?,
        },
        23 => LedgerEvent::SessionRevoked {
            session_id: SessionId(cursor.string()?),
            reason: cursor.string()?,
        },
        24 => LedgerEvent::TierBackendLifecycle {
            backend_id: cursor.string()?,
            tier: cursor.u8()?,
            state: cursor.u8()?,
            revision: Revision(cursor.u64()?),
            event_digest: cursor.digest()?,
        },
        25 => LedgerEvent::TierObjectCommitted {
            root_digest: cursor.digest()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            manifest: cursor.bytes()?.to_vec(),
        },
        26 => LedgerEvent::TierReplicaLifecycle {
            root_digest: cursor.digest()?,
            backend_id: cursor.string()?,
            tier: cursor.u8()?,
            state: cursor.u8()?,
            generation: Generation(cursor.u64()?),
            revision: Revision(cursor.u64()?),
            event_digest: cursor.digest()?,
        },
        27 => {
            let project = ProjectId(cursor.string()?);
            let code = cursor.u8()?;
            let op_kind = ExpertOpKind::from_code(code).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown expert operation code {code}"),
                )
            })?;
            let proposal_id = cursor.digest()?;
            let operation_id = cursor.digest()?;
            let evidence_digest = cursor.digest()?;
            let record = cursor.bytes()?.to_vec();
            check_expert_record(&record)?;
            LedgerEvent::ExpertStructuralCommitted {
                project,
                op_kind,
                proposal_id,
                operation_id,
                evidence_digest,
                record,
            }
        }
        28 => {
            let project = ProjectId(cursor.string()?);
            let pod = PodId(cursor.string()?);
            let action_code = cursor.u8()?;
            let action = ExpertFenceAction::from_code(action_code).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown expert fence action code {action_code}"),
                )
            })?;
            let cause_code = cursor.u8()?;
            let cause = ExpertFenceCause::from_code(cause_code).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown expert fence cause code {cause_code}"),
                )
            })?;
            let resume_code = cursor.u8()?;
            let resume = ExpertFenceResume::from_code(resume_code).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown expert fence resume code {resume_code}"),
                )
            })?;
            let generation = Generation(cursor.u64()?);
            let expected_state_version = cursor.u64()?;
            let epoch = cursor.u64()?;
            let revocation = match cursor.u8()? {
                0 => None,
                1 => Some(ExpertFenceRevocation {
                    subject: cursor.string()?,
                    generation: Generation(cursor.u64()?),
                }),
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown expert fence revocation flag {other}"),
                    ))
                }
            };
            let evidence = match cursor.u8()? {
                0 => None,
                1 => Some(ExpertFenceEvidence {
                    gate_epoch: cursor.u64()?,
                    required_gate_config_digest: cursor.digest()?,
                    report: cursor.bytes()?.to_vec(),
                }),
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown expert fence evidence flag {other}"),
                    ))
                }
            };
            let event = LedgerEvent::ExpertFenceCommitted {
                project,
                pod,
                action,
                cause,
                resume,
                generation,
                expected_state_version,
                epoch,
                revocation,
                evidence,
            };
            check_expert_fence(&event)?;
            event
        }
        29 => {
            let count = usize::from(cursor.u8()?);
            if count == 0 || count > MAX_KV_TAINT_SUBJECTS {
                return Err(integrity::invalid("PTR_LEDGER_KV_TAINT_SUBJECT_LIMIT"));
            }
            let mut subjects = Vec::with_capacity(count);
            for _ in 0..count {
                let code = cursor.u8()?;
                let kind = KvTaintSubjectKind::from_code(code).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown kv taint subject code {code}"),
                    )
                })?;
                subjects.push((kind, cursor.digest()?));
            }
            let change_code = cursor.u8()?;
            let change = match change_code {
                0 => KvTaintChange::Record {
                    report_digest: cursor.digest()?,
                    recipe_digest: cursor.digest()?,
                },
                1 => KvTaintChange::Clear {
                    recipe_digest: cursor.digest()?,
                    recompute_digest: cursor.digest()?,
                },
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown kv taint change code {other}"),
                    ))
                }
            };
            let event = LedgerEvent::KvTaintCommitted { subjects, change };
            check_kv_taint(&event)?;
            event
        }
        30 => {
            let input = cursor.string()?;
            let generation = Generation(cursor.u64()?);
            let binding_digest = cursor.digest()?;
            let kind_code = cursor.u8()?;
            let kind = match kind_code {
                0 => InputInvalidationKind::Revoked,
                1 => InputInvalidationKind::Superseded {
                    old_generation: Generation(cursor.u64()?),
                    new_generation: Generation(cursor.u64()?),
                    new_digest: cursor.digest()?,
                    replacement_ref: cursor.digest()?,
                },
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown input invalidation kind {other}"),
                    ))
                }
            };
            let fence_count = cursor.u32()?;
            let fence_digest = cursor.digest()?;
            let event = LedgerEvent::InputInvalidated {
                input,
                generation,
                binding_digest,
                kind,
                fence_count,
                fence_digest,
            };
            check_input_invalidated(&event)?;
            event
        }
        31 => {
            let kind_code = cursor.u8()?;
            let kind = ExpertAuthorityKind::from_code(kind_code).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown expert authority kind {kind_code}"),
                )
            })?;
            let id_digest = cursor.digest()?;
            let epoch = cursor.u64()?;
            let class_code = cursor.u8()?;
            let class = ExpertAuthorityClass::from_code(class_code).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown expert authority class {class_code}"),
                )
            })?;
            let fence_count = cursor.u32()?;
            let fence_digest = cursor.digest()?;
            let event = LedgerEvent::ExpertAuthorityChanged {
                kind,
                id_digest,
                epoch,
                class,
                fence_count,
                fence_digest,
            };
            check_expert_authority_changed(&event)?;
            event
        }
        37 => {
            let trace_id = TraceId(cursor.string()?);
            let source = match cursor.u8()? {
                0 => {
                    let address = PodAddress {
                        project: ProjectId(cursor.string()?),
                        namespace: NamespaceId(cursor.string()?),
                        pod_id: PodId(cursor.string()?),
                    };
                    let semantic_revision = cursor.digest()?;
                    let generation = Generation(cursor.u64()?);
                    EgressRefusalSource::Pod(PodRevisionAddress {
                        address,
                        semantic_revision,
                        generation,
                    })
                }
                1 => EgressRefusalSource::Overflow,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown egress refusal source tag {other}"),
                    ))
                }
            };
            let target = cursor.string()?;
            let count = cursor.u32()?;
            let saturated = match cursor.u8()? {
                0 => false,
                1 => true,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown egress refusal saturation flag {other}"),
                    ))
                }
            };
            let reason_code = cursor.u8()?;
            let reason = EgressRefusalReason::from_code(reason_code).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown egress refusal reason code {reason_code}"),
                )
            })?;
            let event = LedgerEvent::EgressRefused {
                trace_id,
                source,
                target,
                count,
                saturated,
                reason,
            };
            check_egress_refused(&event)?;
            event
        }
        39 => {
            let project = ProjectId(cursor.string()?);
            let adapter = cursor.string()?;
            let generation = Generation(cursor.u64()?);
            let base = cursor.digest()?;
            let artifact_sha256 = cursor.digest()?;
            let gate_report_digest = cursor.digest()?;
            let event = LedgerEvent::AdapterPromoted {
                project,
                adapter,
                generation,
                base,
                artifact_sha256,
                gate_report_digest,
            };
            check_adapter_promoted(&event)?;
            event
        }
        40 => {
            let project = ProjectId(cursor.string()?);
            let adapter = cursor.string()?;
            let generation = Generation(cursor.u64()?);
            let reason = cursor.string()?;
            let event = LedgerEvent::AdapterRevoked {
                project,
                adapter,
                generation,
                reason,
            };
            check_adapter_revoked(&event)?;
            event
        }
        41 => {
            let project = ProjectId(cursor.string()?);
            let day = cursor.u64()?;
            let outcome = match cursor.u8()? {
                0 => ImprovementCycleOutcome::Completed,
                1 => ImprovementCycleOutcome::Aborted {
                    reason: cursor.string()?,
                },
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown improvement cycle outcome tag {other}"),
                    ))
                }
            };
            let policy_digest = cursor.digest()?;
            let event = LedgerEvent::ImprovementCycleRecorded {
                project,
                day,
                outcome,
                policy_digest,
            };
            check_improvement_cycle(&event)?;
            event
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown ledger event tag {other}"),
            ))
        }
    };

    if !cursor.finished() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "ledger event contains trailing bytes",
        ));
    }

    Ok(event)
}

/// Explicit codes rather than `as` casts. A Rust discriminant follows
/// declaration order, so inserting one variant would renumber every effect
/// after it and every already-written record would then decode as a different
/// effect than it was committed with.
pub fn effect_code(effect: Effect) -> u8 {
    match effect {
        Effect::Pure => 0,
        Effect::Read => 1,
        Effect::Mutation => 2,
        Effect::External => 3,
        Effect::Irreversible => 4,
    }
}

fn effect_from_code(code: u8) -> io::Result<Effect> {
    match code {
        0 => Ok(Effect::Pure),
        1 => Ok(Effect::Read),
        2 => Ok(Effect::Mutation),
        3 => Ok(Effect::External),
        4 => Ok(Effect::Irreversible),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown effect code {other}"),
        )),
    }
}

pub fn verification_code(level: VerificationLevel) -> u8 {
    match level {
        VerificationLevel::Unverified => 0,
        VerificationLevel::LatentAgreement => 1,
        VerificationLevel::SampleVerified => 2,
        VerificationLevel::FullSemantic => 3,
        VerificationLevel::Deterministic => 4,
    }
}

fn verification_from_code(code: u8) -> io::Result<VerificationLevel> {
    match code {
        0 => Ok(VerificationLevel::Unverified),
        1 => Ok(VerificationLevel::LatentAgreement),
        2 => Ok(VerificationLevel::SampleVerified),
        3 => Ok(VerificationLevel::FullSemantic),
        4 => Ok(VerificationLevel::Deterministic),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown verification level code {other}"),
        )),
    }
}

/// Whether `event` meets every rule its encoding carries: the rules the
/// decoder applies to an attributed origin (an attestation's required level
/// and counts, the number of rebased keys), checked by the same functions the
/// decoder calls, and the size of the encoded record
/// ([`integrity::MAX_RECORD_BYTES`]). An event that breaks one is refused
/// with `InvalidData` rather than encoded, since an encoding the decoder
/// refuses would leave a log that cannot be read back. The durable append
/// paths check it, and a writer can check it before appending to a memory
/// ledger, which encodes nothing.
pub fn check_encodable(event: &LedgerEvent) -> Result<(), io::Error> {
    check_origin_bounds(event)?;
    let payload = encode_event(event);
    if payload.len() > integrity::MAX_RECORD_BYTES {
        return Err(integrity::invalid("PTR_LOG_PAYLOAD_LIMIT"));
    }
    Ok(())
}

/// The canonical payload encoding of `event`: the bytes a durable log record frames, whose
/// length [`check_encodable`] bounds. A reader that needs a size or a digest of an event (a
/// summary that must not carry its content) takes it from here instead of rendering the event.
#[must_use]
pub fn encoded_event(event: &LedgerEvent) -> Vec<u8> {
    encode_event(event)
}

/// The rules the decoder applies to an attributed origin, checked before any
/// encoding with the functions the decoder itself calls, in the order it
/// reads the fields they cover, so an append path refuses what the decoder
/// would refuse (and never panics on a count too large for its byte).
pub(crate) fn check_origin_bounds(event: &LedgerEvent) -> Result<(), io::Error> {
    match event {
        LedgerEvent::SemanticDeltaCommitted { origin, .. } => match origin {
            SemanticOrigin::Legacy
            | SemanticOrigin::Request { .. }
            | SemanticOrigin::PodOutput { .. }
            | SemanticOrigin::PodCandidate { .. } => {}
            SemanticOrigin::Host { verification, .. } => check_attestation(verification)?,
            SemanticOrigin::Merge(merge) => {
                check_rebased_key_count(merge.rebased.len())?;
                check_attestation(&merge.verification)?;
            }
        },
        LedgerEvent::PodOutputAdmitted { verification, .. } => check_attestation(verification)?,
        LedgerEvent::PodHypothesisCommitted {
            provenance,
            dependencies,
            verification,
            ..
        } => {
            check_hypothesis_provenance_count(provenance.len())?;
            check_hypothesis_dependency_count(dependencies.len())?;
            check_attestation(verification)?;
        }
        LedgerEvent::ExpertStructuralCommitted { record, .. } => check_expert_record(record)?,
        LedgerEvent::ExpertFenceCommitted { .. } => check_expert_fence(event)?,
        LedgerEvent::KvTaintCommitted { .. } => check_kv_taint(event)?,
        LedgerEvent::InputInvalidated { .. } => check_input_invalidated(event)?,
        LedgerEvent::ExpertAuthorityChanged { .. } => check_expert_authority_changed(event)?,
        LedgerEvent::EgressRefused { .. } => check_egress_refused(event)?,
        LedgerEvent::AdapterPromoted { .. } => check_adapter_promoted(event)?,
        LedgerEvent::AdapterRevoked { .. } => check_adapter_revoked(event)?,
        LedgerEvent::ImprovementCycleRecorded { .. } => check_improvement_cycle(event)?,
        LedgerEvent::MeshTunnelLifecycle(MeshTunnelLifecycleEvent {
            endpoint: Some(endpoint),
            ..
        }) => endpoint
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        _ => {}
    }
    Ok(())
}

/// The one shape a recorded refusal target may have, so a hostname, topic or credential can
/// never ride in it (Invariant 13): `*` (the overflow bucket), `host:<16 lower-case hex>:<len>:<port>`
/// or `topic:<16 lower-case hex>:<len>`.
fn egress_target_is_redacted(target: &str) -> bool {
    fn hex16(part: &str) -> bool {
        part.len() == 16
            && part
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }
    fn number<T: std::str::FromStr>(part: &str) -> bool {
        !part.is_empty()
            && part.bytes().all(|b| b.is_ascii_digit())
            && (part == "0" || !part.starts_with('0'))
            && part.parse::<T>().is_ok()
    }
    if target == "*" {
        return true;
    }
    let parts: Vec<&str> = target.split(':').collect();
    match parts.as_slice() {
        ["host", digest, len, port] => hex16(digest) && number::<u32>(len) && number::<u16>(port),
        ["topic", digest, len] => hex16(digest) && number::<u32>(len),
        _ => false,
    }
}

/// An egress refusal names a non-empty, bounded trace, a positive count, a valid source
/// revision address and a redacted target (see [`egress_target_is_redacted`]). The decoder and
/// the append paths both call this.
fn check_egress_refused(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::EgressRefused {
        trace_id,
        source,
        target,
        count,
        saturated,
        ..
    } = event
    else {
        return Ok(());
    };
    let target_ok = egress_target_is_redacted(target);
    if trace_id.0.is_empty()
        || trace_id.0.len() > MAX_EGRESS_TARGET_BYTES
        || *count == 0
        // Saturation means "at least u32::MAX": nothing else is a saturated count.
        || (*saturated && *count != u32::MAX)
        // `*` stands for several traces, so one refusal cannot carry it.
        || (trace_id.0 == "*" && *count < 2)
        || !target_ok
        || match source {
            EgressRefusalSource::Pod(source) => source.validate().is_err(),
            // The overflow stands for no particular destination.
            EgressRefusalSource::Overflow => target != "*",
        }
    {
        return Err(integrity::invalid("PTR_LEDGER_EGRESS_REFUSED"));
    }
    Ok(())
}

/// A name an adapter event carries: 1 to `max` bytes without control characters, whitespace or
/// the `/` that separates project from adapter in the runtime's lifecycle key, so two distinct
/// (project, adapter) pairs can never share one key.
fn adapter_name_ok(name: &str, max: usize) -> bool {
    !name.is_empty()
        && name.len() <= max
        && !name
            .chars()
            .any(|c| c.is_control() || c.is_whitespace() || c == '/')
}

/// A free-text reason: 1 to [`MAX_ADAPTER_REASON_BYTES`] bytes without control characters.
fn adapter_reason_ok(reason: &str) -> bool {
    !reason.is_empty()
        && reason.len() <= MAX_ADAPTER_REASON_BYTES
        && !reason.chars().any(char::is_control)
}

/// An adapter promotion names a bounded project and adapter, a non-zero generation and three
/// digests none of which is all zero. The decoder and the append paths both call this.
fn check_adapter_promoted(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::AdapterPromoted {
        project,
        adapter,
        generation,
        base,
        artifact_sha256,
        gate_report_digest,
    } = event
    else {
        return Ok(());
    };
    const ZERO: [u8; 32] = [0; 32];
    if !adapter_name_ok(&project.0, MAX_ADAPTER_NAME_BYTES)
        || !adapter_name_ok(adapter, MAX_ADAPTER_NAME_BYTES)
        || generation.0 == 0
        || *base == ZERO
        || *artifact_sha256 == ZERO
        || *gate_report_digest == ZERO
    {
        return Err(integrity::invalid("PTR_LEDGER_ADAPTER_PROMOTED"));
    }
    Ok(())
}

/// An adapter revocation names a bounded project and adapter, a non-zero generation and a
/// bounded reason. The decoder and the append paths both call this.
fn check_adapter_revoked(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::AdapterRevoked {
        project,
        adapter,
        generation,
        reason,
    } = event
    else {
        return Ok(());
    };
    if !adapter_name_ok(&project.0, MAX_ADAPTER_NAME_BYTES)
        || !adapter_name_ok(adapter, MAX_ADAPTER_NAME_BYTES)
        || generation.0 == 0
        || !adapter_reason_ok(reason)
    {
        return Err(integrity::invalid("PTR_LEDGER_ADAPTER_REVOKED"));
    }
    Ok(())
}

/// A cycle record names a bounded project, no all-zero policy digest and, for an aborted
/// cycle, a bounded reason. The decoder and the append paths both call this.
fn check_improvement_cycle(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::ImprovementCycleRecorded {
        project,
        outcome,
        policy_digest,
        ..
    } = event
    else {
        return Ok(());
    };
    let outcome_ok = match outcome {
        ImprovementCycleOutcome::Completed => true,
        ImprovementCycleOutcome::Aborted { reason } => adapter_reason_ok(reason),
    };
    if !adapter_name_ok(&project.0, MAX_ADAPTER_NAME_BYTES)
        || !outcome_ok
        || *policy_digest == [0; 32]
    {
        return Err(integrity::invalid("PTR_LEDGER_IMPROVEMENT_CYCLE"));
    }
    Ok(())
}

/// An authority change carries a non-zero epoch and no all-zero digest, and a `NonTightening`
/// change fences nobody: the one canonical form. The decoder and the append paths both call
/// this.
fn check_expert_authority_changed(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::ExpertAuthorityChanged {
        id_digest,
        epoch,
        class,
        fence_count,
        fence_digest,
        ..
    } = event
    else {
        return Ok(());
    };
    const ZERO: [u8; 32] = [0; 32];
    if *epoch == 0 || *id_digest == ZERO || *fence_digest == ZERO {
        return Err(integrity::invalid("PTR_LEDGER_EXPERT_AUTHORITY_CHANGED"));
    }
    if !class.fences_earlier_evidence() && *fence_count != 0 {
        return Err(integrity::invalid(
            "PTR_LEDGER_EXPERT_AUTHORITY_CHANGED_FENCES",
        ));
    }
    Ok(())
}

/// An input invalidation names a non-empty identity of at most [`MAX_INPUT_IDENTITY_BYTES`]
/// bytes without control characters, a non-zero generation, and carries no all-zero digest. A
/// `Superseded` kind repeats the event's generation as its `old_generation` and moves strictly
/// forward: the one canonical form. The decoder and the append paths both call this.
fn check_input_invalidated(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::InputInvalidated {
        input,
        generation,
        binding_digest,
        kind,
        fence_digest,
        ..
    } = event
    else {
        return Ok(());
    };
    const ZERO: [u8; 32] = [0; 32];
    let identity_ok = !input.is_empty()
        && input.len() <= MAX_INPUT_IDENTITY_BYTES
        && !input.chars().any(char::is_control);
    if !identity_ok || generation.0 == 0 {
        return Err(integrity::invalid("PTR_LEDGER_INPUT_INVALIDATED"));
    }
    let kind_ok = match kind {
        InputInvalidationKind::Revoked => true,
        InputInvalidationKind::Superseded {
            old_generation,
            new_generation,
            new_digest,
            replacement_ref,
        } => {
            old_generation == generation
                && new_generation > old_generation
                && *new_digest != ZERO
                && *replacement_ref != ZERO
        }
    };
    if !kind_ok {
        return Err(integrity::invalid("PTR_LEDGER_INPUT_INVALIDATED"));
    }
    if *binding_digest == ZERO || *fence_digest == ZERO {
        return Err(integrity::invalid(
            "PTR_LEDGER_INPUT_INVALIDATED_ZERO_DIGEST",
        ));
    }
    Ok(())
}

/// A taint record names one to [`MAX_KV_TAINT_SUBJECTS`] subjects in strictly ascending
/// canonical order (kind code, then digest), so without duplicates, and every digest it carries
/// is non-zero: the one canonical form. The decoder and the append paths both call this.
fn check_kv_taint(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::KvTaintCommitted { subjects, change } = event else {
        return Ok(());
    };
    const ZERO: [u8; 32] = [0; 32];
    if subjects.is_empty() || subjects.len() > MAX_KV_TAINT_SUBJECTS {
        return Err(integrity::invalid("PTR_LEDGER_KV_TAINT_SUBJECT_LIMIT"));
    }
    let ordered = subjects
        .windows(2)
        .all(|pair| (pair[0].0.code(), pair[0].1) < (pair[1].0.code(), pair[1].1));
    if !ordered {
        return Err(integrity::invalid("PTR_LEDGER_KV_TAINT_ORDER"));
    }
    let digests_ok = subjects.iter().all(|(_, digest)| *digest != ZERO)
        && match change {
            KvTaintChange::Record {
                report_digest,
                recipe_digest,
            } => *report_digest != ZERO && *recipe_digest != ZERO,
            KvTaintChange::Clear {
                recipe_digest,
                recompute_digest,
            } => *recipe_digest != ZERO && *recompute_digest != ZERO,
        };
    if !digests_ok {
        return Err(integrity::invalid("PTR_LEDGER_KV_TAINT_ZERO_DIGEST"));
    }
    Ok(())
}

/// An expert structural record is non-empty and at most
/// [`MAX_EXPERT_RECORD_BYTES`].
fn check_expert_record(record: &[u8]) -> Result<(), io::Error> {
    if record.is_empty() || record.len() > MAX_EXPERT_RECORD_BYTES {
        return Err(integrity::invalid("PTR_LEDGER_EXPERT_RECORD_LIMIT"));
    }
    Ok(())
}

/// An expert fence record names a non-empty pod and a non-zero generation, its epoch is
/// exactly one above the state version it compares against, and it carries evidence exactly
/// when it is a readmission: the one canonical form.
fn check_expert_fence(event: &LedgerEvent) -> Result<(), io::Error> {
    let LedgerEvent::ExpertFenceCommitted {
        pod,
        generation,
        expected_state_version,
        epoch,
        action,
        cause,
        revocation,
        evidence,
        ..
    } = event
    else {
        return Ok(());
    };
    // An InputRevoked fence, escalation or retirement names its revocation; nothing else does,
    // and an escalation or retirement is only ever for InputRevoked.
    let revocation_ok = match (action, cause, revocation) {
        (
            ExpertFenceAction::Fence | ExpertFenceAction::Escalate | ExpertFenceAction::Retire,
            ExpertFenceCause::InputRevoked,
            Some(revocation),
        ) => !revocation.subject.is_empty(),
        (ExpertFenceAction::Fence, ExpertFenceCause::AuthorityQuarantine, None) => true,
        (ExpertFenceAction::Readmit, _, None) => true,
        _ => false,
    };
    // A readmission carries a non-empty, bounded report; a fence carries none.
    let evidence_ok = match (action, evidence) {
        (
            ExpertFenceAction::Fence | ExpertFenceAction::Escalate | ExpertFenceAction::Retire,
            None,
        ) => true,
        (ExpertFenceAction::Readmit, Some(evidence)) => {
            !evidence.report.is_empty() && evidence.report.len() <= MAX_EXPERT_RECORD_BYTES
        }
        _ => false,
    };
    if !evidence_ok
        || !revocation_ok
        || pod.0.is_empty()
        || generation.0 == 0
        || expected_state_version.checked_add(1) != Some(*epoch)
    {
        return Err(integrity::invalid("PTR_LEDGER_EXPERT_FENCE"));
    }
    Ok(())
}

/// A Pod hypothesis carries at most [`MAX_HYPOTHESIS_PROVENANCE`] provenance
/// entries.
fn check_hypothesis_provenance_count(count: usize) -> Result<(), io::Error> {
    if count > MAX_HYPOTHESIS_PROVENANCE {
        return Err(integrity::invalid("PTR_LEDGER_HYPOTHESIS_LIMIT"));
    }
    Ok(())
}

/// A Pod hypothesis carries at most [`MAX_HYPOTHESIS_DEPENDENCIES`] dependency
/// digests.
fn check_hypothesis_dependency_count(count: usize) -> Result<(), io::Error> {
    if count > MAX_HYPOTHESIS_DEPENDENCIES {
        return Err(integrity::invalid("PTR_LEDGER_HYPOTHESIS_LIMIT"));
    }
    Ok(())
}

/// Every rule an attestation's encoding carries.
fn check_attestation(attestation: &Attestation) -> Result<(), io::Error> {
    check_attestation_requirement(attestation.required)?;
    check_attestation_verifier_count(attestation.verifiers.len())?;
    check_attestation_finding_count(attestation.findings.len())
}

/// An attestation requires `FullSemantic` or `Deterministic`: an attributed
/// write admitted at a lower level is not one this format records.
fn check_attestation_requirement(required: VerificationLevel) -> Result<(), io::Error> {
    if matches!(
        required,
        VerificationLevel::FullSemantic | VerificationLevel::Deterministic
    ) {
        Ok(())
    } else {
        Err(integrity::invalid("PTR_LEDGER_ATTESTATION_REQUIREMENT"))
    }
}

/// An attestation names 1 to [`MAX_ATTESTATION_VERIFIERS`] verifiers.
fn check_attestation_verifier_count(count: usize) -> Result<(), io::Error> {
    if count == 0 || count > MAX_ATTESTATION_VERIFIERS {
        return Err(integrity::invalid("PTR_LEDGER_ATTESTATION_LIMIT"));
    }
    Ok(())
}

/// An attestation carries at most [`MAX_ATTESTATION_FINDINGS`] finding codes.
fn check_attestation_finding_count(count: usize) -> Result<(), io::Error> {
    if count > MAX_ATTESTATION_FINDINGS {
        return Err(integrity::invalid("PTR_LEDGER_ATTESTATION_LIMIT"));
    }
    Ok(())
}

/// A merge record carries at most [`MAX_REBASED_KEYS`] rebased keys.
fn check_rebased_key_count(count: usize) -> Result<(), io::Error> {
    if count > MAX_REBASED_KEYS {
        return Err(integrity::invalid("PTR_LEDGER_REBASED_KEY_LIMIT"));
    }
    Ok(())
}

/// Tag 12's origin after the delta: a kind byte, then the kind's fields.
/// Called only for an attributed origin; `Legacy` has nothing after the delta.
fn put_origin(out: &mut Vec<u8>, origin: &SemanticOrigin) {
    match origin {
        SemanticOrigin::Legacy => {}
        SemanticOrigin::Request { request } => {
            out.push(1);
            put_string(out, request);
        }
        SemanticOrigin::PodOutput {
            request,
            pod,
            level,
        } => {
            out.push(2);
            put_string(out, request);
            put_string(out, pod);
            out.push(verification_code(*level));
        }
        SemanticOrigin::PodCandidate {
            request,
            pod,
            output_digest,
            level,
        } => {
            out.push(5);
            put_string(out, request);
            put_string(out, pod);
            out.extend_from_slice(output_digest);
            out.push(verification_code(*level));
        }
        SemanticOrigin::Host {
            principal,
            verification,
        } => {
            out.push(3);
            put_string(out, principal);
            put_attestation(out, verification);
        }
        SemanticOrigin::Merge(merge) => {
            out.push(4);
            put_string(out, &merge.branch);
            put_string(out, &merge.author);
            out.extend_from_slice(&merge.seal);
            out.extend_from_slice(&merge.plan);
            out.extend_from_slice(&merge.dependencies);
            let count = u32::try_from(merge.rebased.len()).expect("rebased keys are bounded");
            out.extend_from_slice(&count.to_le_bytes());
            for key in &merge.rebased {
                put_string(out, key);
            }
            put_attestation(out, &merge.verification);
            match &merge.authority {
                MergeAuthorityRecord::Triage {
                    policy_version,
                    score_bits,
                } => {
                    out.push(1);
                    put_string(out, policy_version);
                    out.extend_from_slice(&score_bits.to_le_bytes());
                }
                MergeAuthorityRecord::Reviewed { reviewer } => {
                    out.push(2);
                    put_string(out, reviewer);
                }
            }
        }
    }
}

fn put_attestation(out: &mut Vec<u8>, attestation: &Attestation) {
    out.push(verification_code(attestation.required));
    out.push(verification_code(attestation.level));
    // `check_encodable` bounds both counts; a writer checks it before append.
    out.push(u8::try_from(attestation.verifiers.len()).expect("verifier count is bounded"));
    for name in &attestation.verifiers {
        put_string(out, name);
    }
    out.push(u8::try_from(attestation.findings.len()).expect("finding count is bounded"));
    for code in &attestation.findings {
        put_string(out, code);
    }
}

fn attributed_origin(cursor: &mut Cursor<'_>) -> io::Result<SemanticOrigin> {
    Ok(match cursor.u8()? {
        1 => SemanticOrigin::Request {
            request: cursor.string()?,
        },
        2 => SemanticOrigin::PodOutput {
            request: cursor.string()?,
            pod: cursor.string()?,
            level: verification_from_code(cursor.u8()?)?,
        },
        5 => SemanticOrigin::PodCandidate {
            request: cursor.string()?,
            pod: cursor.string()?,
            output_digest: cursor.digest()?,
            level: verification_from_code(cursor.u8()?)?,
        },
        3 => SemanticOrigin::Host {
            principal: cursor.string()?,
            verification: attestation(cursor)?,
        },
        4 => {
            let branch = cursor.string()?;
            let author = cursor.string()?;
            let seal = cursor.digest()?;
            let plan = cursor.digest()?;
            let dependencies = cursor.digest()?;
            let count = cursor.u32()? as usize;
            check_rebased_key_count(count)?;
            let mut rebased = BTreeSet::new();
            let mut previous: Option<String> = None;
            for _ in 0..count {
                let key = cursor.string()?;
                // Strictly ascending, so one set has one encoding.
                if previous.as_ref().is_some_and(|previous| key <= *previous) {
                    return Err(integrity::invalid("PTR_LEDGER_REBASED_KEY_ORDER"));
                }
                previous = Some(key.clone());
                rebased.insert(key);
            }
            let verification = attestation(cursor)?;
            let authority = match cursor.u8()? {
                1 => MergeAuthorityRecord::Triage {
                    policy_version: cursor.string()?,
                    score_bits: cursor.u32()?,
                },
                2 => MergeAuthorityRecord::Reviewed {
                    reviewer: cursor.string()?,
                },
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown merge authority {other}"),
                    ))
                }
            };
            SemanticOrigin::Merge(MergeRecord {
                branch,
                author,
                seal,
                plan,
                dependencies,
                rebased,
                verification,
                authority,
            })
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown semantic origin {other}"),
            ))
        }
    })
}

/// An attestation as encoded, refused by the rules [`check_attestation`]
/// applies before encoding, each as soon as the field it covers is read.
fn attestation(cursor: &mut Cursor<'_>) -> io::Result<Attestation> {
    let required = verification_from_code(cursor.u8()?)?;
    check_attestation_requirement(required)?;
    let level = verification_from_code(cursor.u8()?)?;
    let count = usize::from(cursor.u8()?);
    check_attestation_verifier_count(count)?;
    let verifiers = (0..count)
        .map(|_| cursor.string())
        .collect::<io::Result<Vec<_>>>()?;
    let count = usize::from(cursor.u8()?);
    check_attestation_finding_count(count)?;
    let findings = (0..count)
        .map(|_| cursor.string())
        .collect::<io::Result<Vec<_>>>()?;
    Ok(Attestation {
        required,
        level,
        verifiers,
        findings,
    })
}

fn presence_byte(value: u8) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("invalid presence byte {value}"),
    )
}

fn put_optional_string(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            out.push(1);
            put_string(out, value);
        }
        None => out.push(0),
    }
}

fn put_optional_id<T: std::fmt::Display>(out: &mut Vec<u8>, value: Option<&T>) {
    put_optional_string(out, value.map(ToString::to_string).as_deref());
}

fn put_optional_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            out.push(1);
            put_u64(out, value);
        }
        None => out.push(0),
    }
}

fn put_optional_u128(out: &mut Vec<u8>, value: Option<u128>) {
    match value {
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.to_le_bytes());
        }
        None => out.push(0),
    }
}

fn put_scope_kind(out: &mut Vec<u8>, kind: Option<ScopeLifecycleKind>) {
    match kind {
        Some(kind) => {
            out.push(1);
            out.push(scope_kind_code(kind));
        }
        None => out.push(0),
    }
}

fn put_scope_lifecycle(out: &mut Vec<u8>, event: &ScopeLifecycleEvent) {
    put_string(out, &event.scope_id.to_string());
    put_optional_id(out, event.parent_id.as_ref());
    put_string(out, &event.session_id.to_string());
    put_string(out, &event.project_id.to_string());
    put_u64(out, event.created_at.0);
    put_optional_u64(out, event.deadline.map(|timestamp| timestamp.0));
    out.push(scope_kind_code(event.kind));
    put_scope_kind(out, event.previous_kind);
    put_optional_id(out, event.lease.state_id.as_ref());
    put_optional_id(out, event.lease.pod_id.as_ref());
    put_optional_u64(out, event.lease.placement_epoch);
    put_optional_u128(out, event.lease.fencing_token);
    put_optional_u64(out, event.lease.generation.map(|generation| generation.0));
    put_optional_string(out, event.lease.resource_lease_id.as_deref());
    put_u64(out, event.revision.0);
    put_optional_string(out, event.reason.as_deref());
}

fn put_mesh_endpoint(out: &mut Vec<u8>, endpoint: Option<&MeshEndpointBinding>) {
    let Some(endpoint) = endpoint else {
        out.push(0);
        return;
    };
    out.push(1);
    put_string(out, &endpoint.network_id.0);
    put_string(out, &endpoint.peer_id.0);
    out.push(match endpoint.route {
        MeshRouteKind::Direct => 0,
        MeshRouteKind::Relay => 1,
    });
    put_optional_string(out, endpoint.relay_id.as_ref().map(|peer| peer.0.as_str()));
    put_u64(out, endpoint.membership_generation.0);
    put_u64(out, endpoint.membership_revision.0);
}

fn read_mesh_endpoint(cursor: &mut Cursor<'_>) -> io::Result<Option<MeshEndpointBinding>> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => {
            let network_id = ptr_types::NetworkId(cursor.string()?);
            let peer_id = PeerId(cursor.string()?);
            let route = match cursor.u8()? {
                0 => MeshRouteKind::Direct,
                1 => MeshRouteKind::Relay,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid mesh route {other}"),
                    ))
                }
            };
            let endpoint = MeshEndpointBinding {
                network_id,
                peer_id,
                route,
                relay_id: cursor.optional_string()?.map(PeerId),
                membership_generation: Generation(cursor.u64()?),
                membership_revision: Revision(cursor.u64()?),
            };
            endpoint
                .validate()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            Ok(Some(endpoint))
        }
        other => Err(presence_byte(other)),
    }
}

fn mesh_event_kind_code(kind: MeshTunnelEventKind) -> u8 {
    match kind {
        MeshTunnelEventKind::InvitationCreated => 0,
        MeshTunnelEventKind::InvitationAccepted => 1,
        MeshTunnelEventKind::MembershipActivated => 2,
        MeshTunnelEventKind::MembershipRevoked => 3,
        MeshTunnelEventKind::RouteChanged => 4,
        MeshTunnelEventKind::TunnelEstablished => 5,
        MeshTunnelEventKind::TunnelRevoked => 6,
        MeshTunnelEventKind::TunnelReleased => 7,
    }
}

fn mesh_event_kind_from_code(code: u8) -> io::Result<MeshTunnelEventKind> {
    match code {
        0 => Ok(MeshTunnelEventKind::InvitationCreated),
        1 => Ok(MeshTunnelEventKind::InvitationAccepted),
        2 => Ok(MeshTunnelEventKind::MembershipActivated),
        3 => Ok(MeshTunnelEventKind::MembershipRevoked),
        4 => Ok(MeshTunnelEventKind::RouteChanged),
        5 => Ok(MeshTunnelEventKind::TunnelEstablished),
        6 => Ok(MeshTunnelEventKind::TunnelRevoked),
        7 => Ok(MeshTunnelEventKind::TunnelReleased),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid mesh event {other}"),
        )),
    }
}

fn read_scope_lifecycle(cursor: &mut Cursor<'_>) -> io::Result<ScopeLifecycleEvent> {
    Ok(ScopeLifecycleEvent {
        scope_id: ptr_types::ScopeId(cursor.string()?),
        parent_id: cursor.optional_string()?.map(ptr_types::ScopeId),
        session_id: ptr_types::SessionId(cursor.string()?),
        project_id: ProjectId(cursor.string()?),
        created_at: Timestamp(cursor.u64()?),
        deadline: read_optional_u64(cursor)?.map(Timestamp),
        kind: scope_kind_from_code(cursor.u8()?)?,
        previous_kind: read_optional_scope_kind(cursor)?,
        lease: ScopeLeaseBinding {
            state_id: cursor.optional_string()?.map(ptr_types::StateId),
            pod_id: cursor.optional_string()?.map(ptr_types::PodId),
            placement_epoch: read_optional_u64(cursor)?,
            fencing_token: read_optional_u128(cursor)?,
            generation: read_optional_u64(cursor)?.map(Generation),
            resource_lease_id: cursor.optional_string()?,
        },
        revision: Revision(cursor.u64()?),
        reason: cursor.optional_string()?,
    })
}

fn read_optional_scope_kind(cursor: &mut Cursor<'_>) -> io::Result<Option<ScopeLifecycleKind>> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => Ok(Some(scope_kind_from_code(cursor.u8()?)?)),
        other => Err(presence_byte(other)),
    }
}

fn read_optional_u64(cursor: &mut Cursor<'_>) -> io::Result<Option<u64>> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => Ok(Some(cursor.u64()?)),
        other => Err(presence_byte(other)),
    }
}

fn read_optional_u128(cursor: &mut Cursor<'_>) -> io::Result<Option<u128>> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => {
            let mut bytes = [0; 16];
            for byte in &mut bytes {
                *byte = cursor.u8()?;
            }
            Ok(Some(u128::from_le_bytes(bytes)))
        }
        other => Err(presence_byte(other)),
    }
}

fn scope_kind_code(kind: ScopeLifecycleKind) -> u8 {
    match kind {
        ScopeLifecycleKind::Created => 0,
        ScopeLifecycleKind::Admitted => 1,
        ScopeLifecycleKind::Started => 2,
        ScopeLifecycleKind::Completed => 3,
        ScopeLifecycleKind::Failed => 4,
        ScopeLifecycleKind::Cancelled => 5,
        ScopeLifecycleKind::TimedOut => 6,
        ScopeLifecycleKind::Revoked => 7,
        ScopeLifecycleKind::Released => 8,
    }
}

fn scope_kind_from_code(code: u8) -> io::Result<ScopeLifecycleKind> {
    match code {
        0 => Ok(ScopeLifecycleKind::Created),
        1 => Ok(ScopeLifecycleKind::Admitted),
        2 => Ok(ScopeLifecycleKind::Started),
        3 => Ok(ScopeLifecycleKind::Completed),
        4 => Ok(ScopeLifecycleKind::Failed),
        5 => Ok(ScopeLifecycleKind::Cancelled),
        6 => Ok(ScopeLifecycleKind::TimedOut),
        7 => Ok(ScopeLifecycleKind::Revoked),
        8 => Ok(ScopeLifecycleKind::Released),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown scope lifecycle code {other}"),
        )),
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    let length = u32::try_from(bytes.len()).expect("string length fits u32");
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(bytes);
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn u8(&mut self) -> io::Result<u8> {
        let value = *self.bytes.get(self.offset).ok_or_else(truncated)?;
        self.offset += 1;
        Ok(value)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let end = self.offset.saturating_add(4);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self) -> io::Result<u64> {
        let end = self.offset.saturating_add(8);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(u64::from_le_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn u128(&mut self) -> io::Result<u128> {
        let mut bytes = [0; 16];
        for byte in &mut bytes {
            *byte = self.u8()?;
        }
        Ok(u128::from_le_bytes(bytes))
    }

    fn bytes(&mut self) -> io::Result<&'a [u8]> {
        let length = self.u32()? as usize;
        let end = self.offset.checked_add(length).ok_or_else(truncated)?;
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(bytes)
    }

    fn string(&mut self) -> io::Result<String> {
        String::from_utf8(self.bytes()?.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string"))
    }

    fn optional_string(&mut self) -> io::Result<Option<String>> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.string()?)),
            other => Err(presence_byte(other)),
        }
    }

    fn digest(&mut self) -> io::Result<[u8; 32]> {
        let end = self.offset.saturating_add(32);
        let bytes = self.bytes.get(self.offset..end).ok_or_else(truncated)?;
        self.offset = end;
        Ok(bytes.try_into().expect("thirty-two bytes"))
    }
}

fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "truncated ledger record")
}

#[cfg(feature = "raft-engine-backend")]
mod raft_engine_backend {
    use super::*;
    use raft_engine::{Config, Engine, LogBatch};

    const GROUP_ID: u64 = 1;
    const KEY_PREFIX: &[u8] = b"ptr/event/";

    pub struct RaftEngineLedger {
        engine: Engine,
        events: Vec<CommittedEvent>,
    }

    impl RaftEngineLedger {
        pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
            let config = Config {
                dir: path.as_ref().to_string_lossy().into_owned(),
                ..Default::default()
            };
            let engine = Engine::open(config).map_err(engine_error)?;

            let mut encoded = Vec::<(Vec<u8>, Vec<u8>)>::new();
            engine
                .scan_raw_messages(GROUP_ID, None, None, false, |key, value| {
                    if key.starts_with(KEY_PREFIX) {
                        encoded.push((key.to_vec(), value.to_vec()));
                    }
                    true
                })
                .map_err(engine_error)?;
            encoded.sort_by(|left, right| left.0.cmp(&right.0));

            let mut events = Vec::with_capacity(encoded.len());
            for (offset, (key, payload)) in encoded.into_iter().enumerate() {
                let index = parse_event_key(&key)?;
                let expected = offset as u64 + 1;
                if index != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("raft-engine ledger gap: expected {expected}, found {index}"),
                    ));
                }
                events.push(CommittedEvent {
                    index: CommitIndex(index),
                    event: decode_event(&payload)?,
                });
            }

            Ok(Self { engine, events })
        }

        /// Persist and synchronize an event, then return its one-based commit index.
        ///
        /// # Errors
        ///
        /// Returns an I/O error if the engine rejects the batch entry or the
        /// synchronized write fails. The in-memory event list advances only
        /// after a successful write.
        pub fn append_durable(&mut self, event: LedgerEvent) -> io::Result<CommitIndex> {
            check_encodable(&event)?;
            let index = CommitIndex(self.events.len() as u64 + 1);
            let mut batch = LogBatch::default();
            // `put` refuses a key in raft-engine's reserved namespace. Ours never is,
            // but dropping the refusal would write an empty batch and still record the
            // event as durably committed.
            batch
                .put(
                    GROUP_ID,
                    event_key(index).into_bytes(),
                    encode_event(&event),
                )
                .map_err(engine_error)?;
            self.engine.write(&mut batch, true).map_err(engine_error)?;
            self.events.push(CommittedEvent { index, event });
            Ok(index)
        }

        pub fn events(&self) -> &[CommittedEvent] {
            &self.events
        }

        pub fn sync(&self) -> io::Result<()> {
            self.engine.sync().map_err(engine_error)
        }

        pub fn purge_expired_files(&self) -> io::Result<Vec<u64>> {
            self.engine.purge_expired_files().map_err(engine_error)
        }
    }

    fn event_key(index: CommitIndex) -> String {
        format!("ptr/event/{:020}", index.0)
    }

    fn parse_event_key(key: &[u8]) -> io::Result<u64> {
        if !key.starts_with(KEY_PREFIX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected raft-engine ledger key prefix",
            ));
        }
        let suffix = std::str::from_utf8(&key[KEY_PREFIX.len()..])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "ledger key is not UTF-8"))?;
        suffix
            .parse::<u64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid ledger event index"))
    }

    fn engine_error(error: raft_engine::Error) -> io::Error {
        io::Error::other(error.to_string())
    }
}

#[cfg(feature = "raft-engine-backend")]
pub use raft_engine_backend::RaftEngineLedger;

#[cfg(feature = "raft-rs-backend")]
pub mod raft_node;
#[cfg(feature = "raft-rs-backend")]
pub mod raft_storage;
#[cfg(feature = "raft-rs-backend")]
pub use raft_node::{
    decode_message, encode_message, InstalledSnapshot, MembershipChange, NoSnapshotPending,
    NodeFailed, RaftMessage, RaftNode, RetainedSnapshotAnchor, SnapshotAnchorMismatch,
    SnapshotDelivery, MAX_MESSAGE_BYTES,
};
#[cfg(feature = "raft-rs-backend")]
pub use raft_storage::{
    FileRaftStorage, SnapshotFloorRefusal, MAX_ENTRY_BYTES, MAX_SNAPSHOT_BYTES,
};

#[cfg(feature = "raft-rs-backend")]
mod raft_rs_backend {
    use super::*;
    use crate::raft_node::RaftNode;
    use std::path::Path;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RaftCommitReceipt {
        pub raft_index: u64,
        pub commit_index: CommitIndex,
    }

    /// A one-member group over durable state.
    ///
    /// It is one member, so it demonstrates nothing about partitions or leader
    /// changes — [`RaftNode`] is what composes into a group. What this adds is the
    /// convenience a single member allows: a proposal is decided by the time
    /// `propose` returns, because there is nobody to hear from.
    ///
    /// It is the same node type underneath, so there is one place where entries are
    /// persisted, hard state is flushed and committed entries are applied. Two
    /// copies of that sequence would be two chances to get the ordering wrong.
    pub struct SingleNodeRaftConsensus {
        node: RaftNode,
    }

    impl SingleNodeRaftConsensus {
        /// Open durable state under `dir`, recovering committed history.
        pub fn open(dir: &Path, node_id: u64) -> Result<Self, String> {
            let mut node = RaftNode::open(dir, node_id, &[node_id])?;
            let outbound = node.campaign()?;
            if !outbound.is_empty() {
                return Err("a one-member group has nobody to send votes to".into());
            }
            if !node.is_leader() {
                return Err("single-node raft group failed to become leader".into());
            }
            Ok(Self { node })
        }

        pub fn is_leader(&self) -> bool {
            self.node.is_leader()
        }

        pub fn committed_events(&self) -> &[CommittedEvent] {
            self.node.committed_events()
        }

        /// The durable term this node has seen.
        pub fn term(&self) -> u64 {
            self.node.term()
        }

        pub fn propose(&mut self, event: LedgerEvent) -> Result<RaftCommitReceipt, String> {
            let before = self.node.committed_events().len();
            let outbound = self.node.propose(event)?;
            if !outbound.is_empty() {
                return Err(
                    "a one-member group produced a message with no peer to receive it".into(),
                );
            }

            let committed =
                self.node.committed_events().get(before).ok_or_else(|| {
                    "proposal was not committed in single-node raft group".to_owned()
                })?;
            Ok(RaftCommitReceipt {
                raft_index: self.node.raft_committed(),
                commit_index: committed.index,
            })
        }
    }
}

#[cfg(feature = "raft-rs-backend")]
pub use raft_rs_backend::{RaftCommitReceipt, SingleNodeRaftConsensus};

#[cfg(test)]
mod effect_record_tests {
    use super::*;

    fn attempt_keyed(key: Option<&str>) -> LedgerEvent {
        LedgerEvent::EffectAttempted {
            key: key.map(str::to_owned),
            project: ProjectId::from("p"),
            principal: "alice".into(),
            target: "capsule:a".into(),
            operation: "write".into(),
            capability: CapabilityId::from("file.write"),
            effect: Effect::External,
            generation: Generation(3),
            revision: Revision(9),
            verification: VerificationLevel::FullSemantic,
            action_digest: [7; 32],
        }
    }

    fn attempt() -> LedgerEvent {
        attempt_keyed(Some("invoice-7"))
    }

    /// The codes are data, not discriminants. Inserting a variant must break a
    /// build here rather than silently renumber records that are already stored.
    #[test]
    fn every_effect_and_verification_member_has_a_pinned_code() {
        for (effect, code) in [
            (Effect::Pure, 0),
            (Effect::Read, 1),
            (Effect::Mutation, 2),
            (Effect::External, 3),
            (Effect::Irreversible, 4),
        ] {
            assert_eq!(effect_code(effect), code);
            assert_eq!(effect_from_code(code).unwrap(), effect);
        }
        for (level, code) in [
            (VerificationLevel::Unverified, 0),
            (VerificationLevel::LatentAgreement, 1),
            (VerificationLevel::SampleVerified, 2),
            (VerificationLevel::FullSemantic, 3),
            (VerificationLevel::Deterministic, 4),
        ] {
            assert_eq!(verification_code(level), code);
            assert_eq!(verification_from_code(code).unwrap(), level);
        }
    }

    #[test]
    fn effect_records_round_trip_through_their_canonical_encoding() {
        let records = [
            attempt(),
            attempt_keyed(None),
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(4),
                response: Some(b"executed".to_vec()),
                response_digest: [1; 32],
            },
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(4),
                response: None,
                response_digest: [2; 32],
            },
            LedgerEvent::EffectReconciled {
                attempt: CommitIndex(4),
                applied: true,
                evidence: "operator checked".into(),
            },
            LedgerEvent::EffectReconciled {
                attempt: CommitIndex(4),
                applied: false,
                evidence: "no record downstream".into(),
            },
        ];
        for record in records {
            let encoded = encode_event(&record);
            assert_eq!(decode_event(&encoded).unwrap(), record);
            // A record that decodes from a prefix of itself would mean the
            // framing does not pin its own length.
            assert!(decode_event(&encoded[..encoded.len() - 1]).is_err());
        }
    }

    fn taint_event(subjects: Vec<(KvTaintSubjectKind, [u8; 32])>) -> LedgerEvent {
        LedgerEvent::KvTaintCommitted {
            subjects,
            change: KvTaintChange::Record {
                report_digest: [7; 32],
                recipe_digest: [8; 32],
            },
        }
    }

    const ALL_TAINT_KINDS: [KvTaintSubjectKind; 5] = [
        KvTaintSubjectKind::Action,
        KvTaintSubjectKind::Derivation,
        KvTaintSubjectKind::Content,
        KvTaintSubjectKind::PodOutput,
        KvTaintSubjectKind::Request,
    ];

    #[test]
    fn kv_taint_round_trips_every_kind_and_change() {
        let subjects: Vec<_> = ALL_TAINT_KINDS
            .iter()
            .map(|kind| (*kind, [kind.code() + 1; 32]))
            .collect();
        let changes = [
            KvTaintChange::Record {
                report_digest: [7; 32],
                recipe_digest: [8; 32],
            },
            KvTaintChange::Clear {
                recipe_digest: [8; 32],
                recompute_digest: [9; 32],
            },
        ];
        for change in changes {
            let event = LedgerEvent::KvTaintCommitted {
                subjects: subjects.clone(),
                change,
            };
            check_encodable(&event).unwrap();
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 29);
            assert_eq!(decode_event(&encoded).unwrap(), event);
            assert!(decode_event(&encoded[..encoded.len() - 1]).is_err());
            let mut trailing = encoded.clone();
            trailing.push(0);
            assert!(decode_event(&trailing).is_err());
        }
        for (kind, code) in ALL_TAINT_KINDS.iter().zip(0u8..) {
            assert_eq!(kind.code(), code);
            assert_eq!(KvTaintSubjectKind::from_code(code), Some(*kind));
        }
        assert_eq!(KvTaintSubjectKind::from_code(5), None);
        let single = taint_event(vec![(KvTaintSubjectKind::Request, [1; 32])]);
        assert_eq!(decode_event(&encode_event(&single)).unwrap(), single);
        let full: Vec<_> = (1..=MAX_KV_TAINT_SUBJECTS as u8)
            .map(|n| (KvTaintSubjectKind::Content, [n; 32]))
            .collect();
        let full = taint_event(full);
        check_encodable(&full).unwrap();
        assert_eq!(decode_event(&encode_event(&full)).unwrap(), full);
    }

    #[test]
    fn kv_taint_refuses_malformed_records_on_encode_and_decode() {
        let a = (KvTaintSubjectKind::Action, [1; 32]);
        let b = (KvTaintSubjectKind::Action, [2; 32]);
        let c = (KvTaintSubjectKind::Content, [1; 32]);
        let over: Vec<_> = (1..=(MAX_KV_TAINT_SUBJECTS as u8 + 1))
            .map(|n| (KvTaintSubjectKind::Content, [n; 32]))
            .collect();
        let bad = [
            taint_event(Vec::new()),
            taint_event(over),
            taint_event(vec![b, a]),
            taint_event(vec![c, a]),
            taint_event(vec![a, a]),
            taint_event(vec![(KvTaintSubjectKind::Action, [0; 32])]),
            LedgerEvent::KvTaintCommitted {
                subjects: vec![a],
                change: KvTaintChange::Record {
                    report_digest: [0; 32],
                    recipe_digest: [8; 32],
                },
            },
            LedgerEvent::KvTaintCommitted {
                subjects: vec![a],
                change: KvTaintChange::Record {
                    report_digest: [7; 32],
                    recipe_digest: [0; 32],
                },
            },
            LedgerEvent::KvTaintCommitted {
                subjects: vec![a],
                change: KvTaintChange::Clear {
                    recipe_digest: [0; 32],
                    recompute_digest: [9; 32],
                },
            },
            LedgerEvent::KvTaintCommitted {
                subjects: vec![a],
                change: KvTaintChange::Clear {
                    recipe_digest: [8; 32],
                    recompute_digest: [0; 32],
                },
            },
        ];
        for event in bad {
            assert!(check_encodable(&event).is_err(), "{event:?}");
            assert!(decode_event(&encode_event(&event)).is_err(), "{event:?}");
        }
        // Unknown subject kind and unknown change code, on an otherwise valid record.
        let good = encode_event(&taint_event(vec![a]));
        let mut unknown_kind = good.clone();
        unknown_kind[2] = 5;
        assert!(decode_event(&unknown_kind).is_err());
        let mut unknown_change = good.clone();
        unknown_change[2 + 32 + 1] = 2;
        assert!(decode_event(&unknown_change).is_err());
    }

    fn invalidated_event(kind: InputInvalidationKind) -> LedgerEvent {
        LedgerEvent::InputInvalidated {
            input: "capsule:data".to_owned(),
            generation: Generation(3),
            binding_digest: [1; 32],
            kind,
            fence_count: 2_000,
            fence_digest: [2; 32],
        }
    }

    fn superseded() -> InputInvalidationKind {
        InputInvalidationKind::Superseded {
            old_generation: Generation(3),
            new_generation: Generation(4),
            new_digest: [3; 32],
            replacement_ref: [4; 32],
        }
    }

    #[test]
    fn input_invalidation_round_trips_under_tag_30_and_the_codes_are_pinned() {
        assert_eq!(InputInvalidationKind::Revoked.code(), 0);
        assert_eq!(superseded().code(), 1);
        for kind in [InputInvalidationKind::Revoked, superseded()] {
            let event = invalidated_event(kind);
            check_encodable(&event).unwrap();
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 30);
            assert_eq!(decode_event(&encoded).unwrap(), event);
            // Every strict prefix is a truncation and is refused.
            for len in 0..encoded.len() {
                assert!(decode_event(&encoded[..len]).is_err(), "prefix {len}");
            }
            // Trailing bytes are refused.
            let mut padded = encoded.clone();
            padded.push(0);
            assert!(decode_event(&padded).is_err());
        }
        // The count is carried as a u32 and the largest fan-out round-trips.
        let mut event = invalidated_event(InputInvalidationKind::Revoked);
        if let LedgerEvent::InputInvalidated { fence_count, .. } = &mut event {
            *fence_count = u32::MAX;
        }
        assert_eq!(decode_event(&encode_event(&event)).unwrap(), event);
    }

    fn egress_refused_event(reason: EgressRefusalReason) -> LedgerEvent {
        LedgerEvent::EgressRefused {
            trace_id: TraceId("trace-1".into()),
            source: EgressRefusalSource::Pod(PodRevisionAddress {
                address: PodAddress {
                    project: ProjectId("p".into()),
                    namespace: NamespaceId("ns".into()),
                    pod_id: PodId("pod".into()),
                },
                semantic_revision: [7; 32],
                generation: Generation(3),
            }),
            target: "host:0123456789abcdef:14:1883".into(),
            count: 3,
            saturated: false,
            reason,
        }
    }

    #[test]
    fn egress_refusal_round_trips_under_tag_37_and_the_codes_are_pinned() {
        let pinned = [
            (EgressRefusalReason::HostNotAllowed, 0u8),
            (EgressRefusalReason::PortNotAllowed, 1),
            (EgressRefusalReason::TopicNotAllowed, 2),
            (EgressRefusalReason::InvalidTopic, 3),
            (EgressRefusalReason::RateLimited, 4),
            (EgressRefusalReason::TopicRequired, 5),
            (EgressRefusalReason::TopicUnverifiable, 6),
        ];
        assert_eq!(pinned.len(), EgressRefusalReason::ALL.len());
        for (reason, code) in pinned {
            assert_eq!(reason.code(), code);
            assert_eq!(EgressRefusalReason::from_code(code), Some(reason));
            let event = egress_refused_event(reason);
            check_encodable(&event).unwrap();
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 37);
            assert_eq!(*encoded.last().unwrap(), code, "the reason code is last");
            assert_eq!(
                encoded[encoded.len() - 2],
                0,
                "the saturation flag precedes it"
            );
            assert_eq!(decode_event(&encoded).unwrap(), event);
            for len in 0..encoded.len() {
                assert!(decode_event(&encoded[..len]).is_err(), "prefix {len}");
            }
            let mut padded = encoded.clone();
            padded.push(0);
            assert!(decode_event(&padded).is_err());
        }
        assert_eq!(EgressRefusalReason::from_code(7), None);
        let mut unknown = encode_event(&egress_refused_event(EgressRefusalReason::RateLimited));
        *unknown.last_mut().unwrap() = 7;
        assert!(decode_event(&unknown).is_err());
    }

    #[test]
    fn a_saturated_egress_count_round_trips_and_an_unknown_flag_is_refused() {
        let mut event = egress_refused_event(EgressRefusalReason::RateLimited);
        if let LedgerEvent::EgressRefused {
            count, saturated, ..
        } = &mut event
        {
            *count = u32::MAX;
            *saturated = true;
        }
        check_encodable(&event).unwrap();
        let encoded = encode_event(&event);
        assert_eq!(decode_event(&encoded).unwrap(), event);
        let mut unknown = encoded.clone();
        let flag_at = unknown.len() - 2;
        assert_eq!(unknown[flag_at], 1);
        unknown[flag_at] = 2;
        assert!(decode_event(&unknown).is_err());
        // The flag with any other count is not a canonical form, even if the bytes decode.
        let mut small = encoded;
        let count_at = small.len() - 2 - 4;
        small[count_at..count_at + 4].copy_from_slice(&7u32.to_le_bytes());
        assert!(decode_event(&small).is_err());
    }

    #[test]
    fn tag_33_belongs_to_another_decision_and_decodes_as_unknown() {
        let mut encoded = encode_event(&egress_refused_event(EgressRefusalReason::RateLimited));
        assert_eq!(encoded[0], 37);
        encoded[0] = 33;
        let error = decode_event(&encoded).unwrap_err();
        assert!(error.to_string().contains("unknown ledger event tag 33"));
    }

    #[test]
    fn the_many_traces_marker_needs_a_count_of_at_least_two() {
        for (count, accepted) in [(1u32, false), (2, true), (u32::MAX, true)] {
            let mut event = egress_refused_event(EgressRefusalReason::RateLimited);
            if let LedgerEvent::EgressRefused {
                trace_id, count: c, ..
            } = &mut event
            {
                *trace_id = TraceId("*".into());
                *c = count;
            }
            assert_eq!(check_encodable(&event).is_ok(), accepted, "count {count}");
            assert_eq!(decode_event(&encode_event(&event)).is_ok(), accepted);
        }
    }

    #[test]
    fn egress_refusal_overflow_source_round_trips_with_the_star_target_only() {
        let mut event = egress_refused_event(EgressRefusalReason::RateLimited);
        if let LedgerEvent::EgressRefused { source, target, .. } = &mut event {
            *source = EgressRefusalSource::Overflow;
            *target = "*".to_owned();
        }
        check_encodable(&event).unwrap();
        let encoded = encode_event(&event);
        assert_eq!(encoded[0], 37);
        assert_eq!(decode_event(&encoded).unwrap(), event);
        for len in 0..encoded.len() {
            assert!(decode_event(&encoded[..len]).is_err(), "prefix {len}");
        }
        // The pod and overflow encodings differ in the tag byte and nothing else is shared.
        let pod = encode_event(&egress_refused_event(EgressRefusalReason::RateLimited));
        assert_ne!(encoded, pod);
        // An unknown source tag is refused by the decoder.
        let mut unknown = encoded.clone();
        let tag_at = 1 + 4 + "trace-1".len();
        assert_eq!(unknown[tag_at], 1);
        unknown[tag_at] = 2;
        assert!(decode_event(&unknown).is_err());
    }

    #[test]
    fn egress_refusal_target_must_be_a_redacted_shape() {
        let accepted = [
            "*",
            "host:0123456789abcdef:14:1883",
            "host:0123456789abcdef:0:0",
            "topic:0123456789abcdef:24",
        ];
        let refused = [
            "broker.example:1883",
            "127.0.0.1:1883",
            "tokens/s3cr3t",
            "host:0123456789abcdef",
            "host:0123456789ABCDEF:1:1",
            "host:0123456789abcde:1:1",
            "host:0123456789abcdef:1:65536",
            "host:0123456789abcdef:01:1",
            "host:0123456789abcdef:1:1:1",
            "topic:0123456789abcdef",
            "topic:0123456789abcdef:x",
            "topic:s3cr3t:24",
            "**",
        ];
        for target in accepted.iter().chain(refused.iter()) {
            let mut event = egress_refused_event(EgressRefusalReason::HostNotAllowed);
            if let LedgerEvent::EgressRefused { target: t, .. } = &mut event {
                *t = (*target).to_owned();
            }
            assert_eq!(
                check_encodable(&event).is_ok(),
                accepted.contains(target),
                "{target}"
            );
        }
    }

    #[test]
    fn egress_refusal_refuses_every_non_canonical_form() {
        let with = |edit: &dyn Fn(&mut LedgerEvent)| {
            let mut event = egress_refused_event(EgressRefusalReason::HostNotAllowed);
            edit(&mut event);
            event
        };
        let long = "x".repeat(MAX_EGRESS_TARGET_BYTES + 1);
        let bad = [
            with(&|e| {
                if let LedgerEvent::EgressRefused { target, .. } = e {
                    target.clear();
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused { target, .. } = e {
                    *target = long.clone();
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused { target, .. } = e {
                    *target = "host: 80".to_owned();
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused { target, .. } = e {
                    *target = "a\nb".to_owned();
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused { trace_id, .. } = e {
                    trace_id.0.clear();
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused { trace_id, .. } = e {
                    trace_id.0 = "t".repeat(MAX_EGRESS_TARGET_BYTES + 1);
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused { count, .. } = e {
                    *count = 0;
                }
            }),
            // A saturated count is u32::MAX and nothing else.
            with(&|e| {
                if let LedgerEvent::EgressRefused { saturated, .. } = e {
                    *saturated = true;
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused {
                    source: EgressRefusalSource::Pod(source),
                    ..
                } = e
                {
                    source.semantic_revision = [0; 32];
                }
            }),
            with(&|e| {
                if let LedgerEvent::EgressRefused {
                    source: EgressRefusalSource::Pod(source),
                    ..
                } = e
                {
                    source.address.pod_id = PodId("bad pod".into());
                }
            }),
            // The overflow names no destination: only `*`.
            with(&|e| {
                if let LedgerEvent::EgressRefused { source, .. } = e {
                    *source = EgressRefusalSource::Overflow;
                }
            }),
        ];
        for event in bad {
            assert!(check_encodable(&event).is_err(), "{event:?}");
            // The decoder refuses the same forms, so a log that holds one cannot be read back.
            assert!(decode_event(&encode_event(&event)).is_err(), "{event:?}");
        }
        let overflow = with(&|e| {
            if let LedgerEvent::EgressRefused { target, .. } = e {
                *target = "*".to_owned();
            }
        });
        check_encodable(&overflow).unwrap();
        assert_eq!(decode_event(&encode_event(&overflow)).unwrap(), overflow);
    }

    #[test]
    fn input_invalidation_refuses_every_non_canonical_form() {
        let with = |edit: &dyn Fn(&mut LedgerEvent)| {
            let mut event = invalidated_event(superseded());
            edit(&mut event);
            event
        };
        let long = "x".repeat(MAX_INPUT_IDENTITY_BYTES + 1);
        let bad = [
            with(&|e| {
                if let LedgerEvent::InputInvalidated { input, .. } = e {
                    input.clear();
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { input, .. } = e {
                    *input = long.clone();
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { input, .. } = e {
                    *input = "a\nb".to_owned();
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { generation, .. } = e {
                    *generation = Generation(0);
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { binding_digest, .. } = e {
                    *binding_digest = [0; 32];
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { fence_digest, .. } = e {
                    *fence_digest = [0; 32];
                }
            }),
            // The event's generation and the old generation of a supersession must agree.
            with(&|e| {
                if let LedgerEvent::InputInvalidated { generation, .. } = e {
                    *generation = Generation(2);
                }
            }),
            // A supersession moves strictly forward.
            with(&|e| {
                if let LedgerEvent::InputInvalidated { kind, .. } = e {
                    *kind = InputInvalidationKind::Superseded {
                        old_generation: Generation(3),
                        new_generation: Generation(3),
                        new_digest: [3; 32],
                        replacement_ref: [4; 32],
                    };
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { kind, .. } = e {
                    *kind = InputInvalidationKind::Superseded {
                        old_generation: Generation(3),
                        new_generation: Generation(2),
                        new_digest: [3; 32],
                        replacement_ref: [4; 32],
                    };
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { kind, .. } = e {
                    *kind = InputInvalidationKind::Superseded {
                        old_generation: Generation(3),
                        new_generation: Generation(4),
                        new_digest: [0; 32],
                        replacement_ref: [4; 32],
                    };
                }
            }),
            with(&|e| {
                if let LedgerEvent::InputInvalidated { kind, .. } = e {
                    *kind = InputInvalidationKind::Superseded {
                        old_generation: Generation(3),
                        new_generation: Generation(4),
                        new_digest: [3; 32],
                        replacement_ref: [0; 32],
                    };
                }
            }),
        ];
        for event in bad {
            assert_eq!(
                check_encodable(&event).unwrap_err().kind(),
                io::ErrorKind::InvalidData,
                "{event:?}"
            );
            // What the append path refuses, the decoder refuses too.
            assert!(decode_event(&encode_event(&event)).is_err(), "{event:?}");
        }
        // An identity at the bound is accepted.
        let mut at_bound = invalidated_event(InputInvalidationKind::Revoked);
        if let LedgerEvent::InputInvalidated { input, .. } = &mut at_bound {
            *input = "x".repeat(MAX_INPUT_IDENTITY_BYTES);
        }
        check_encodable(&at_bound).unwrap();
        assert_eq!(decode_event(&encode_event(&at_bound)).unwrap(), at_bound);
    }

    fn authority_event(class: ExpertAuthorityClass) -> LedgerEvent {
        LedgerEvent::ExpertAuthorityChanged {
            kind: ExpertAuthorityKind::Evaluator,
            id_digest: [5; 32],
            epoch: 7,
            class,
            fence_count: if class.fences_earlier_evidence() {
                3
            } else {
                0
            },
            fence_digest: [6; 32],
        }
    }

    #[test]
    fn authority_change_round_trips_under_tag_31_and_the_codes_are_pinned() {
        assert_eq!(ExpertAuthorityKind::GateConfig.code(), 0);
        assert_eq!(ExpertAuthorityKind::Evaluator.code(), 1);
        assert_eq!(ExpertAuthorityKind::from_code(2), None);
        for (class, code) in [
            (ExpertAuthorityClass::Revoke, 0),
            (ExpertAuthorityClass::TighteningRotation, 1),
            (ExpertAuthorityClass::NonTightening, 2),
        ] {
            assert_eq!(class.code(), code);
            assert_eq!(ExpertAuthorityClass::from_code(code), Some(class));
            let event = authority_event(class);
            check_encodable(&event).unwrap();
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 31);
            assert_eq!(decode_event(&encoded).unwrap(), event);
            for len in 0..encoded.len() {
                assert!(decode_event(&encoded[..len]).is_err(), "prefix {len}");
            }
            let mut padded = encoded.clone();
            padded.push(0);
            assert!(decode_event(&padded).is_err());
        }
        assert_eq!(ExpertAuthorityClass::from_code(3), None);
    }

    #[test]
    fn authority_change_refuses_every_non_canonical_form() {
        let with = |edit: &dyn Fn(&mut LedgerEvent)| {
            let mut event = authority_event(ExpertAuthorityClass::Revoke);
            edit(&mut event);
            event
        };
        let bad = [
            with(&|e| {
                if let LedgerEvent::ExpertAuthorityChanged { epoch, .. } = e {
                    *epoch = 0;
                }
            }),
            with(&|e| {
                if let LedgerEvent::ExpertAuthorityChanged { id_digest, .. } = e {
                    *id_digest = [0; 32];
                }
            }),
            with(&|e| {
                if let LedgerEvent::ExpertAuthorityChanged { fence_digest, .. } = e {
                    *fence_digest = [0; 32];
                }
            }),
            // A non-tightening rotation fences nobody.
            with(&|e| {
                if let LedgerEvent::ExpertAuthorityChanged {
                    class, fence_count, ..
                } = e
                {
                    *class = ExpertAuthorityClass::NonTightening;
                    *fence_count = 1;
                }
            }),
        ];
        for event in bad {
            assert_eq!(
                check_encodable(&event).unwrap_err().kind(),
                io::ErrorKind::InvalidData,
                "{event:?}"
            );
            assert!(decode_event(&encode_event(&event)).is_err(), "{event:?}");
        }
        // The largest fan-out round-trips.
        let mut event = authority_event(ExpertAuthorityClass::TighteningRotation);
        if let LedgerEvent::ExpertAuthorityChanged { fence_count, .. } = &mut event {
            *fence_count = u32::MAX;
        }
        assert_eq!(decode_event(&encode_event(&event)).unwrap(), event);
    }

    #[test]
    fn authority_change_refuses_an_unknown_kind_or_class() {
        let encoded = encode_event(&authority_event(ExpertAuthorityClass::Revoke));
        // tag, kind, id digest, epoch, class.
        let kind_at = 1;
        let class_at = 1 + 1 + 32 + 8;
        assert_eq!(encoded[kind_at], 1);
        assert_eq!(encoded[class_at], 0);
        for code in [2u8, 3, 255] {
            let mut unknown_kind = encoded.clone();
            unknown_kind[kind_at] = code;
            assert!(decode_event(&unknown_kind)
                .unwrap_err()
                .to_string()
                .contains("unknown expert authority kind"));
            let mut unknown_class = encoded.clone();
            unknown_class[class_at] = code.max(3);
            assert!(decode_event(&unknown_class)
                .unwrap_err()
                .to_string()
                .contains("unknown expert authority class"));
        }
    }

    #[test]
    fn input_invalidation_refuses_an_unknown_kind() {
        let encoded = encode_event(&invalidated_event(InputInvalidationKind::Revoked));
        // tag, input (4-byte length + 12 bytes), generation, binding digest, then the kind.
        let kind_at = 1 + 4 + "capsule:data".len() + 8 + 32;
        assert_eq!(encoded[kind_at], 0);
        for code in [2u8, 3, 255] {
            let mut unknown = encoded.clone();
            unknown[kind_at] = code;
            let error = decode_event(&unknown).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error
                .to_string()
                .contains("unknown input invalidation kind"));
        }
        // Changing a Revoked record's kind byte to Superseded leaves too few bytes.
        let mut as_superseded = encoded.clone();
        as_superseded[kind_at] = 1;
        assert!(decode_event(&as_superseded).is_err());
    }

    fn expert_event() -> LedgerEvent {
        LedgerEvent::ExpertStructuralCommitted {
            project: ProjectId::from("p"),
            op_kind: ExpertOpKind::Merge,
            proposal_id: [1; 32],
            operation_id: [2; 32],
            evidence_digest: [3; 32],
            record: b"PTREXR01-opaque".to_vec(),
        }
    }

    #[test]
    fn expert_op_kinds_have_pinned_codes() {
        for (kind, code) in [
            (ExpertOpKind::Birth, 0),
            (ExpertOpKind::Merge, 1),
            (ExpertOpKind::Split, 2),
            (ExpertOpKind::Freeze, 3),
            (ExpertOpKind::Reactivate, 4),
            (ExpertOpKind::Prune, 5),
            (ExpertOpKind::Retire, 6),
        ] {
            assert_eq!(kind.code(), code);
            assert_eq!(ExpertOpKind::from_code(code), Some(kind));
        }
        assert_eq!(ExpertOpKind::from_code(7), None);
        assert_eq!(ExpertOpKind::from_code(255), None);
    }

    #[test]
    fn expert_structural_events_round_trip_with_a_stable_tag() {
        for op_kind in (0..=6).map(|c| ExpertOpKind::from_code(c).unwrap()) {
            let event = LedgerEvent::ExpertStructuralCommitted {
                project: ProjectId::from("p"),
                op_kind,
                proposal_id: [1; 32],
                operation_id: [2; 32],
                evidence_digest: [3; 32],
                record: b"PTREXR01-opaque".to_vec(),
            };
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 27);
            assert_eq!(decode_event(&encoded).unwrap(), event);
            check_encodable(&event).unwrap();
            assert!(decode_event(&encoded[..encoded.len() - 1]).is_err());
        }
    }

    #[test]
    fn expert_structural_events_refuse_trailing_bytes_and_unknown_op_codes() {
        let mut encoded = encode_event(&expert_event());
        // tag, project (4-byte length + "p"), then the op code.
        let op_at = 1 + 4 + 1;
        assert_eq!(encoded[op_at], ExpertOpKind::Merge.code());
        let mut unknown = encoded.clone();
        unknown[op_at] = 7;
        let error = decode_event(&unknown).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error
            .to_string()
            .contains("unknown expert operation code 7"));
        encoded.push(0);
        assert!(decode_event(&encoded).is_err());
    }

    #[test]
    fn expert_structural_records_are_bounded_and_non_empty() {
        let with_record = |record: Vec<u8>| LedgerEvent::ExpertStructuralCommitted {
            project: ProjectId::from("p"),
            op_kind: ExpertOpKind::Freeze,
            proposal_id: [1; 32],
            operation_id: [2; 32],
            evidence_digest: [3; 32],
            record,
        };
        for record in [Vec::new(), vec![0; MAX_EXPERT_RECORD_BYTES + 1]] {
            let event = with_record(record);
            assert_eq!(
                check_encodable(&event).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
            // What the append path refuses, the decoder refuses too.
            assert!(decode_event(&encode_event(&event)).is_err());
        }
        let largest = with_record(vec![9; MAX_EXPERT_RECORD_BYTES]);
        check_encodable(&largest).unwrap();
        assert_eq!(decode_event(&encode_event(&largest)).unwrap(), largest);
    }

    fn fence_event(action: ExpertFenceAction) -> LedgerEvent {
        LedgerEvent::ExpertFenceCommitted {
            project: ProjectId::from("p"),
            pod: PodId::from("pod-a"),
            action,
            cause: ExpertFenceCause::AuthorityQuarantine,
            resume: ExpertFenceResume::Frozen,
            generation: Generation(3),
            expected_state_version: 4,
            epoch: 5,
            revocation: None,
            evidence: match action {
                ExpertFenceAction::Fence
                | ExpertFenceAction::Escalate
                | ExpertFenceAction::Retire => None,
                ExpertFenceAction::Readmit => Some(ExpertFenceEvidence {
                    gate_epoch: 7,
                    required_gate_config_digest: [6; 32],
                    report: b"PTRGR-opaque".to_vec(),
                }),
            },
        }
    }

    #[test]
    fn expert_fence_codes_are_pinned_and_events_round_trip_under_tag_28() {
        for (action, code) in [
            (ExpertFenceAction::Fence, 0),
            (ExpertFenceAction::Readmit, 1),
            (ExpertFenceAction::Escalate, 2),
            (ExpertFenceAction::Retire, 3),
        ] {
            assert_eq!(action.code(), code);
            assert_eq!(ExpertFenceAction::from_code(code), Some(action));
        }
        for (cause, code) in [
            (ExpertFenceCause::InputRevoked, 0),
            (ExpertFenceCause::AuthorityQuarantine, 1),
        ] {
            assert_eq!(cause.code(), code);
            assert_eq!(ExpertFenceCause::from_code(code), Some(cause));
        }
        assert_eq!(ExpertFenceAction::from_code(4), None);
        assert_eq!(ExpertFenceResume::from_code(2), None);
        assert_eq!(ExpertFenceResume::Frozen.code(), 1);
        assert_eq!(ExpertFenceCause::from_code(2), None);
        for action in [
            ExpertFenceAction::Fence,
            ExpertFenceAction::Readmit,
            ExpertFenceAction::Retire,
        ] {
            let mut event = fence_event(action);
            if action == ExpertFenceAction::Retire {
                if let LedgerEvent::ExpertFenceCommitted {
                    cause, revocation, ..
                } = &mut event
                {
                    *cause = ExpertFenceCause::InputRevoked;
                    *revocation = Some(ExpertFenceRevocation {
                        subject: "k".to_owned(),
                        generation: Generation(2),
                    });
                }
            }
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 28);
            assert_eq!(decode_event(&encoded).unwrap(), event);
            check_encodable(&event).unwrap();
            assert!(decode_event(&encoded[..encoded.len() - 1]).is_err());
            let mut trailing = encoded.clone();
            trailing.push(0);
            assert!(decode_event(&trailing).is_err());
        }
    }

    #[test]
    fn an_expert_retirement_carries_a_revocation_and_never_gate_evidence() {
        let retire =
            |evidence: Option<ExpertFenceEvidence>, epoch: u64| LedgerEvent::ExpertFenceCommitted {
                project: ProjectId::from("p"),
                pod: PodId::from("pod-a"),
                action: ExpertFenceAction::Retire,
                cause: ExpertFenceCause::InputRevoked,
                resume: ExpertFenceResume::Serving,
                generation: Generation(3),
                expected_state_version: 4,
                epoch,
                revocation: Some(ExpertFenceRevocation {
                    subject: "k".to_owned(),
                    generation: Generation(1),
                }),
                evidence,
            };
        let good = retire(None, 5);
        check_encodable(&good).unwrap();
        assert_eq!(decode_event(&encode_event(&good)).unwrap(), good);
        // Gate evidence on a retirement: it never waits for a gate.
        let with_evidence = retire(
            Some(ExpertFenceEvidence {
                gate_epoch: 7,
                required_gate_config_digest: [6; 32],
                report: b"PTRGR-opaque".to_vec(),
            }),
            5,
        );
        assert!(check_encodable(&with_evidence).is_err());
        assert!(decode_event(&encode_event(&with_evidence)).is_err());
        // The epoch is exactly one above the compared state version, as for every action.
        for epoch in [4, 6] {
            let bad = retire(None, epoch);
            assert!(check_encodable(&bad).is_err());
            assert!(decode_event(&encode_event(&bad)).is_err());
        }
    }

    #[test]
    fn expert_fence_events_refuse_unknown_codes_and_non_canonical_epochs() {
        let encoded = encode_event(&fence_event(ExpertFenceAction::Fence));
        // tag, project (4 + 1), pod (4 + 5), then action and cause.
        let action_at = 1 + 5 + 9;
        let mut unknown_action = encoded.clone();
        unknown_action[action_at] = 4;
        assert!(decode_event(&unknown_action)
            .unwrap_err()
            .to_string()
            .contains("unknown expert fence action code 4"));
        let mut unknown_cause = encoded.clone();
        unknown_cause[action_at + 1] = 9;
        assert!(decode_event(&unknown_cause)
            .unwrap_err()
            .to_string()
            .contains("unknown expert fence cause code 9"));
        let bad = |pod: &str, generation: u64, expected: u64, epoch: u64| {
            LedgerEvent::ExpertFenceCommitted {
                project: ProjectId::from("p"),
                pod: PodId::from(pod),
                action: ExpertFenceAction::Fence,
                cause: ExpertFenceCause::InputRevoked,
                resume: ExpertFenceResume::Serving,
                generation: Generation(generation),
                expected_state_version: expected,
                epoch,
                revocation: Some(ExpertFenceRevocation {
                    subject: "s".to_owned(),
                    generation: Generation(1),
                }),
                evidence: None,
            }
        };
        for event in [
            bad("pod-a", 3, 4, 4),
            bad("pod-a", 3, 4, 6),
            bad("pod-a", 3, u64::MAX, 0),
            bad("pod-a", 0, 4, 5),
            bad("", 3, 4, 5),
        ] {
            assert!(check_encodable(&event).is_err());
            // What the append path refuses, the decoder refuses too.
            assert!(decode_event(&encode_event(&event)).is_err());
        }
        check_encodable(&bad("pod-a", 3, 4, 5)).unwrap();
        // Evidence exactly when readmitting, and never an empty or oversized report.
        let mut fence_with_evidence = fence_event(ExpertFenceAction::Readmit);
        let mut readmit_without = fence_event(ExpertFenceAction::Readmit);
        let mut empty = fence_event(ExpertFenceAction::Readmit);
        let mut oversized = fence_event(ExpertFenceAction::Readmit);
        for (event, change) in [
            (&mut fence_with_evidence, 0),
            (&mut readmit_without, 1),
            (&mut empty, 2),
            (&mut oversized, 3),
        ] {
            if let LedgerEvent::ExpertFenceCommitted {
                action, evidence, ..
            } = event
            {
                match change {
                    0 => *action = ExpertFenceAction::Fence,
                    1 => *evidence = None,
                    2 => evidence.as_mut().unwrap().report.clear(),
                    _ => evidence.as_mut().unwrap().report = vec![0; MAX_EXPERT_RECORD_BYTES + 1],
                }
            }
            assert!(check_encodable(event).is_err());
            assert!(decode_event(&encode_event(event)).is_err());
        }
        let mut bad_flag = encode_event(&fence_event(ExpertFenceAction::Fence));
        *bad_flag.last_mut().unwrap() = 2;
        assert!(decode_event(&bad_flag).is_err());
    }

    #[test]
    fn expert_fence_revocation_is_present_exactly_for_an_input_revoked_fence_or_escalation() {
        let event = |action, cause, revocation: Option<&str>| {
            let mut event = fence_event(action);
            if let LedgerEvent::ExpertFenceCommitted {
                cause: c,
                revocation: r,
                ..
            } = &mut event
            {
                *c = cause;
                *r = revocation.map(|subject| ExpertFenceRevocation {
                    subject: subject.to_owned(),
                    generation: Generation(2),
                });
            }
            event
        };
        use ExpertFenceAction::{Escalate, Fence, Readmit, Retire};
        use ExpertFenceCause::{AuthorityQuarantine, InputRevoked};
        for good in [
            event(Fence, InputRevoked, Some("k")),
            event(Escalate, InputRevoked, Some("k")),
            event(Retire, InputRevoked, Some("k")),
            event(Fence, AuthorityQuarantine, None),
            event(Readmit, AuthorityQuarantine, None),
        ] {
            check_encodable(&good).unwrap();
            assert_eq!(decode_event(&encode_event(&good)).unwrap(), good);
        }
        for bad in [
            event(Fence, InputRevoked, None),
            event(Escalate, InputRevoked, None),
            event(Escalate, AuthorityQuarantine, None),
            // A retirement is for InputRevoked only and always names its revocation.
            event(Retire, InputRevoked, None),
            event(Retire, InputRevoked, Some("")),
            event(Retire, AuthorityQuarantine, None),
            event(Retire, AuthorityQuarantine, Some("k")),
            event(Fence, InputRevoked, Some("")),
            event(Fence, AuthorityQuarantine, Some("k")),
            event(Readmit, AuthorityQuarantine, Some("k")),
            event(Readmit, InputRevoked, Some("k")),
        ] {
            assert!(check_encodable(&bad).is_err(), "{bad:?}");
            assert!(decode_event(&encode_event(&bad)).is_err(), "{bad:?}");
        }
        let mut unknown_resume = encode_event(&fence_event(ExpertFenceAction::Fence));
        unknown_resume[1 + 5 + 9 + 2] = 5;
        assert!(decode_event(&unknown_resume)
            .unwrap_err()
            .to_string()
            .contains("unknown expert fence resume code 5"));
    }

    #[test]
    fn mesh_tunnel_events_round_trip_with_a_stable_tag() {
        let endpoint = MeshEndpointBinding {
            network_id: "mesh-a".into(),
            peer_id: "peer-b".into(),
            route: MeshRouteKind::Relay,
            relay_id: Some("relay-c".into()),
            membership_generation: Generation(2),
            membership_revision: Revision(4),
        };
        let event = LedgerEvent::MeshTunnelLifecycle(MeshTunnelLifecycleEvent {
            network_id: "mesh-a".into(),
            peer_id: "peer-b".into(),
            invitation_id: Some("invite-1".into()),
            tunnel_id: Some("tunnel-1".into()),
            generation: Generation(2),
            revision: Revision(4),
            kind: MeshTunnelEventKind::TunnelEstablished,
            endpoint: Some(endpoint),
            placement_epoch: 7,
            fencing_token: FencingToken(9),
            event_digest: [8; 32],
        });
        let encoded = encode_event(&event);
        assert_eq!(encoded[0], 15);
        assert_eq!(decode_event(&encoded).unwrap(), event);
    }

    #[test]
    fn an_endpoint_the_decoder_refuses_is_refused_before_encoding() {
        // A relayed route without a relay id: `validate` refuses it on decode,
        // so an append path must refuse it too instead of writing a record the
        // log cannot read back.
        let event = LedgerEvent::MeshTunnelLifecycle(MeshTunnelLifecycleEvent {
            network_id: "mesh-a".into(),
            peer_id: "peer-b".into(),
            invitation_id: None,
            tunnel_id: None,
            generation: Generation(2),
            revision: Revision(4),
            kind: MeshTunnelEventKind::RouteChanged,
            endpoint: Some(MeshEndpointBinding {
                network_id: "mesh-a".into(),
                peer_id: "peer-b".into(),
                route: MeshRouteKind::Relay,
                relay_id: None,
                membership_generation: Generation(2),
                membership_revision: Revision(4),
            }),
            placement_epoch: 7,
            fencing_token: FencingToken(9),
            event_digest: [8; 32],
        });
        assert!(decode_event(&encode_event(&event)).is_err());
        let error = check_encodable(&event).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    /// The only byte two records differ in is the field that differs, which locates
    /// it without hand-counting offsets that would rot with the layout.
    fn sole_difference(left: &[u8], right: &[u8]) -> usize {
        assert_eq!(left.len(), right.len());
        let differing: Vec<usize> = (0..left.len()).filter(|i| left[*i] != right[*i]).collect();
        assert_eq!(differing.len(), 1, "expected exactly one differing byte");
        differing[0]
    }

    fn attempt_effect(effect: Effect) -> LedgerEvent {
        match attempt() {
            LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                generation,
                revision,
                verification,
                action_digest,
                ..
            } => LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                effect,
                generation,
                revision,
                verification,
                action_digest,
            },
            other => other,
        }
    }

    fn attempt_verification(verification: VerificationLevel) -> LedgerEvent {
        match attempt() {
            LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                effect,
                generation,
                revision,
                action_digest,
                ..
            } => LedgerEvent::EffectAttempted {
                key,
                project,
                principal,
                target,
                operation,
                capability,
                effect,
                generation,
                revision,
                verification,
                action_digest,
            },
            other => other,
        }
    }

    #[test]
    fn unknown_effect_and_verification_codes_are_refused() {
        assert!(effect_from_code(5).is_err());
        assert!(verification_from_code(5).is_err());

        // A code this build does not assign must stop the record rather than
        // decode as whichever member happens to sit nearby.
        let effect_offset = sole_difference(
            &encode_event(&attempt_effect(Effect::Pure)),
            &encode_event(&attempt_effect(Effect::Read)),
        );
        let mut encoded = encode_event(&attempt());
        assert_eq!(encoded[effect_offset], effect_code(Effect::External));
        encoded[effect_offset] = 5;
        assert!(decode_event(&encoded).is_err());

        let verification_offset = sole_difference(
            &encode_event(&attempt_verification(VerificationLevel::Unverified)),
            &encode_event(&attempt_verification(VerificationLevel::Deterministic)),
        );
        assert_ne!(verification_offset, effect_offset);
        let mut encoded = encode_event(&attempt());
        assert_eq!(
            encoded[verification_offset],
            verification_code(VerificationLevel::FullSemantic)
        );
        encoded[verification_offset] = 5;
        assert!(decode_event(&encoded).is_err());
    }

    #[test]
    fn an_invalid_presence_or_boolean_byte_is_refused() {
        // A third value for a two-valued field would be a second encoding of one
        // of the two meanings.
        let mut encoded = encode_event(&attempt());
        assert_eq!(encoded[1], 1, "the key's presence byte follows the tag");
        encoded[1] = 2;
        assert!(decode_event(&encoded).is_err());

        let mut encoded = encode_event(&LedgerEvent::EffectSettled {
            attempt: CommitIndex(4),
            response: None,
            response_digest: [0; 32],
        });
        assert_eq!(
            encoded[9], 0,
            "the response presence byte follows the index"
        );
        encoded[9] = 2;
        assert!(decode_event(&encoded).is_err());

        let mut encoded = encode_event(&LedgerEvent::EffectReconciled {
            attempt: CommitIndex(4),
            applied: true,
            evidence: "checked".into(),
        });
        assert_eq!(encoded[9], 1, "the applied byte follows the index");
        encoded[9] = 2;
        assert!(decode_event(&encoded).is_err());
    }

    #[test]
    fn a_truncated_digest_is_refused_rather_than_zero_filled() {
        let encoded = encode_event(&attempt());
        for missing in 1..=32 {
            assert!(decode_event(&encoded[..encoded.len() - missing]).is_err());
        }
    }

    #[test]
    fn trailing_bytes_after_an_effect_record_are_refused() {
        for record in [
            attempt(),
            LedgerEvent::EffectSettled {
                attempt: CommitIndex(1),
                response: None,
                response_digest: [0; 32],
            },
            LedgerEvent::EffectReconciled {
                attempt: CommitIndex(1),
                applied: false,
                evidence: "checked".into(),
            },
        ] {
            let mut encoded = encode_event(&record);
            encoded.push(0);
            assert!(decode_event(&encoded).is_err());
        }
    }
}

#[cfg(test)]
mod semantic_origin_tests {
    use super::*;

    fn attestation() -> Attestation {
        Attestation {
            required: VerificationLevel::FullSemantic,
            level: VerificationLevel::Deterministic,
            verifiers: vec!["schema".into(), "replay".into()],
            findings: vec!["replay/slow".into()],
        }
    }

    fn merge(authority: MergeAuthorityRecord, rebased: &[&str]) -> SemanticOrigin {
        SemanticOrigin::Merge(MergeRecord {
            branch: "branch-7".into(),
            author: "alice".into(),
            seal: [1; 32],
            plan: [2; 32],
            dependencies: [3; 32],
            rebased: rebased.iter().map(|key| (*key).to_owned()).collect(),
            verification: attestation(),
            authority,
        })
    }

    fn committed(origin: SemanticOrigin) -> LedgerEvent {
        LedgerEvent::SemanticDeltaCommitted {
            base_revision: Revision(2),
            revision: Revision(3),
            encoded_delta: b"abc".to_vec(),
            origin,
        }
    }

    /// One origin of every attributed kind, every Pod level, both merge
    /// authorities, and an attestation at each of its bounds.
    fn attributed_origins() -> Vec<SemanticOrigin> {
        let mut origins = vec![SemanticOrigin::Request {
            request: "request:1".into(),
        }];
        for level in [
            VerificationLevel::Unverified,
            VerificationLevel::LatentAgreement,
            VerificationLevel::SampleVerified,
            VerificationLevel::FullSemantic,
            VerificationLevel::Deterministic,
        ] {
            origins.push(SemanticOrigin::PodOutput {
                request: "request:1".into(),
                pod: "pod-a".into(),
                level,
            });
        }
        origins.push(SemanticOrigin::Host {
            principal: "operator".into(),
            verification: attestation(),
        });
        origins.push(SemanticOrigin::Host {
            principal: "operator".into(),
            verification: Attestation {
                required: VerificationLevel::Deterministic,
                level: VerificationLevel::Unverified,
                verifiers: (0..MAX_ATTESTATION_VERIFIERS)
                    .map(|n| format!("verifier-{n:02}"))
                    .collect(),
                findings: (0..MAX_ATTESTATION_FINDINGS)
                    .map(|n| format!("verifier-00/finding-{n:02}"))
                    .collect(),
            },
        });
        origins.push(merge(
            MergeAuthorityRecord::Triage {
                policy_version: "policy-3".into(),
                score_bits: 0.75f32.to_bits(),
            },
            &[],
        ));
        origins.push(merge(
            MergeAuthorityRecord::Reviewed {
                reviewer: "bob".into(),
            },
            &["doc:a", "doc:b", "fact:z"],
        ));
        origins
    }

    /// A length-prefixed string, written by hand rather than by the encoder.
    fn text(value: &str) -> Vec<u8> {
        let mut out = u32::try_from(value.len()).unwrap().to_le_bytes().to_vec();
        out.extend_from_slice(value.as_bytes());
        out
    }

    /// The part of `committed`'s encoding before the origin: the tag, both
    /// revisions and the delta.
    fn prefix(tag: u8) -> Vec<u8> {
        let mut out = vec![tag];
        out.extend_from_slice(&2u64.to_le_bytes());
        out.extend_from_slice(&3u64.to_le_bytes());
        out.extend(text("abc"));
        out
    }

    /// An attestation with the counts its lists give, so a test can write
    /// counts the encoder never would.
    fn raw_attestation(required: u8, level: u8, verifiers: &[&str], findings: &[&str]) -> Vec<u8> {
        let mut out = vec![required, level, u8::try_from(verifiers.len()).unwrap()];
        for name in verifiers {
            out.extend(text(name));
        }
        out.push(u8::try_from(findings.len()).unwrap());
        for code in findings {
            out.extend(text(code));
        }
        out
    }

    fn host(attestation: Vec<u8>) -> Vec<u8> {
        let mut out = prefix(12);
        out.push(3);
        out.extend(text("operator"));
        out.extend(attestation);
        out
    }

    /// A merge origin up to its rebased keys, with a raw count.
    fn merge_head(count: u32, rebased: &[&str]) -> Vec<u8> {
        let mut out = prefix(12);
        out.push(4);
        out.extend(text("branch-7"));
        out.extend(text("alice"));
        out.extend_from_slice(&[1; 32]);
        out.extend_from_slice(&[2; 32]);
        out.extend_from_slice(&[3; 32]);
        out.extend_from_slice(&count.to_le_bytes());
        for key in rebased {
            out.extend(text(key));
        }
        out
    }

    fn refusal(payload: &[u8]) -> String {
        decode_event(payload)
            .expect_err("the payload must be refused")
            .to_string()
    }

    #[test]
    fn a_legacy_origin_encodes_as_tag_eight_byte_for_byte() {
        // What every build before origins wrote for this record: the tag, both
        // revisions, the length-prefixed delta, and nothing after it.
        let golden = [
            8, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, b'a', b'b', b'c',
        ];
        let legacy = committed(SemanticOrigin::Legacy);
        assert_eq!(encode_event(&legacy), golden);
        assert_eq!(decode_event(&golden).unwrap(), legacy);
        // Nothing may follow a tag-8 delta, not even a well-formed origin.
        let mut followed = golden.to_vec();
        followed.push(1);
        followed.extend(text("request:1"));
        assert_eq!(refusal(&followed), "ledger event contains trailing bytes");
    }

    #[test]
    fn every_attested_origin_round_trips_through_tag_twelve() {
        for origin in attributed_origins() {
            let event = committed(origin);
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], 12, "{event:?}");
            assert_eq!(decode_event(&encoded).unwrap(), event);
        }

        // The layout of each kind, written out by hand.
        let mut request = prefix(12);
        request.push(1);
        request.extend(text("request:1"));
        assert_eq!(
            encode_event(&committed(SemanticOrigin::Request {
                request: "request:1".into()
            })),
            request
        );

        let mut pod = prefix(12);
        pod.push(2);
        pod.extend(text("request:1"));
        pod.extend(text("pod-a"));
        pod.push(2);
        assert_eq!(
            encode_event(&committed(SemanticOrigin::PodOutput {
                request: "request:1".into(),
                pod: "pod-a".into(),
                level: VerificationLevel::SampleVerified,
            })),
            pod
        );

        let host_layout = host(raw_attestation(
            3,
            4,
            &["schema", "replay"],
            &["replay/slow"],
        ));
        assert_eq!(
            encode_event(&committed(SemanticOrigin::Host {
                principal: "operator".into(),
                verification: attestation(),
            })),
            host_layout
        );

        let mut reviewed = merge_head(3, &["doc:a", "doc:b", "fact:z"]);
        reviewed.extend(raw_attestation(
            3,
            4,
            &["schema", "replay"],
            &["replay/slow"],
        ));
        reviewed.push(2);
        reviewed.extend(text("bob"));
        assert_eq!(
            encode_event(&committed(merge(
                MergeAuthorityRecord::Reviewed {
                    reviewer: "bob".into()
                },
                &["fact:z", "doc:b", "doc:a"],
            ))),
            reviewed,
            "rebased keys are written in ascending order"
        );

        let mut triage = merge_head(0, &[]);
        triage.extend(raw_attestation(
            3,
            4,
            &["schema", "replay"],
            &["replay/slow"],
        ));
        triage.push(1);
        triage.extend(text("policy-3"));
        triage.extend_from_slice(&0.75f32.to_bits().to_le_bytes());
        assert_eq!(
            encode_event(&committed(merge(
                MergeAuthorityRecord::Triage {
                    policy_version: "policy-3".into(),
                    score_bits: 0.75f32.to_bits(),
                },
                &[],
            ))),
            triage
        );
    }

    #[test]
    fn unknown_kinds_levels_counts_key_order_and_trailing_bytes_are_refused() {
        // An origin kind outside 1..=5, or none at all.
        for kind in [0, 6, 255] {
            let mut payload = prefix(12);
            payload.push(kind);
            payload.extend(text("request:1"));
            assert_eq!(refusal(&payload), format!("unknown semantic origin {kind}"));
        }
        assert_eq!(
            decode_event(&prefix(12)).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );

        // A Pod level outside the five codes.
        let mut pod = prefix(12);
        pod.push(2);
        pod.extend(text("request:1"));
        pod.extend(text("pod-a"));
        pod.push(5);
        assert_eq!(refusal(&pod), "unknown verification level code 5");

        // A requirement below FullSemantic, or a level code outside the five.
        for required in [0, 1, 2] {
            assert_eq!(
                refusal(&host(raw_attestation(required, 4, &["v"], &[]))),
                "PTR_LEDGER_ATTESTATION_REQUIREMENT"
            );
        }
        for required in [3, 4] {
            assert!(decode_event(&host(raw_attestation(required, 0, &["v"], &[]))).is_ok());
        }
        assert_eq!(
            refusal(&host(raw_attestation(5, 4, &["v"], &[]))),
            "unknown verification level code 5"
        );
        assert_eq!(
            refusal(&host(raw_attestation(3, 5, &["v"], &[]))),
            "unknown verification level code 5"
        );

        // Verifier counts outside 1..=16 and finding counts above 32.
        let names: Vec<String> = (0..=MAX_ATTESTATION_VERIFIERS)
            .map(|n| format!("v{n}"))
            .collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        assert_eq!(
            refusal(&host(raw_attestation(3, 4, &[], &[]))),
            "PTR_LEDGER_ATTESTATION_LIMIT"
        );
        assert!(decode_event(&host(raw_attestation(
            3,
            4,
            &names[..MAX_ATTESTATION_VERIFIERS],
            &[]
        )))
        .is_ok());
        assert_eq!(
            refusal(&host(raw_attestation(3, 4, &names, &[]))),
            "PTR_LEDGER_ATTESTATION_LIMIT"
        );
        let codes: Vec<String> = (0..=MAX_ATTESTATION_FINDINGS)
            .map(|n| format!("v0/f{n}"))
            .collect();
        let codes: Vec<&str> = codes.iter().map(String::as_str).collect();
        assert!(decode_event(&host(raw_attestation(
            3,
            4,
            &["v0"],
            &codes[..MAX_ATTESTATION_FINDINGS]
        )))
        .is_ok());
        assert_eq!(
            refusal(&host(raw_attestation(3, 4, &["v0"], &codes))),
            "PTR_LEDGER_ATTESTATION_LIMIT"
        );

        // Rebased keys out of order, duplicated, or more than a delta holds.
        // The count is refused before any key is read.
        let tail = |mut head: Vec<u8>| {
            head.extend(raw_attestation(3, 4, &["v"], &[]));
            head.push(2);
            head.extend(text("bob"));
            head
        };
        assert!(decode_event(&tail(merge_head(2, &["a", "b"]))).is_ok());
        assert_eq!(
            refusal(&tail(merge_head(2, &["b", "a"]))),
            "PTR_LEDGER_REBASED_KEY_ORDER"
        );
        assert_eq!(
            refusal(&tail(merge_head(2, &["a", "a"]))),
            "PTR_LEDGER_REBASED_KEY_ORDER"
        );
        let too_many = u32::try_from(MAX_REBASED_KEYS + 1).unwrap();
        assert_eq!(
            refusal(&merge_head(too_many, &[])),
            "PTR_LEDGER_REBASED_KEY_LIMIT"
        );

        // A merge authority outside 1..=2.
        for authority in [0, 3] {
            let mut payload = merge_head(0, &[]);
            payload.extend(raw_attestation(3, 4, &["v"], &[]));
            payload.push(authority);
            payload.extend(text("bob"));
            assert_eq!(
                refusal(&payload),
                format!("unknown merge authority {authority}")
            );
        }

        // Every attributed record refuses a trailing byte and every cut.
        for origin in attributed_origins() {
            let encoded = encode_event(&committed(origin));
            let mut followed = encoded.clone();
            followed.push(0);
            assert_eq!(refusal(&followed), "ledger event contains trailing bytes");
            for end in 0..encoded.len() {
                assert!(decode_event(&encoded[..end]).is_err(), "cut at {end}");
            }
        }
    }

    fn pod_hypothesis(
        provenance: usize,
        dependencies: usize,
        verification: Attestation,
    ) -> LedgerEvent {
        LedgerEvent::PodHypothesisCommitted {
            request_id: RequestId("request".into()),
            session_id: SessionId("session".into()),
            scope_id: ScopeId("scope".into()),
            branch_id: "branch".into(),
            pod_id: PodId("pod".into()),
            manifest_digest: [1; 32],
            artifact_digest: [2; 32],
            generation: Generation(1),
            revision: Revision(1),
            output_type: TypeId("type".into()),
            output_digest: [3; 32],
            payload: b"payload".to_vec(),
            provenance: (0..provenance).map(|n| (format!("s{n}"), None)).collect(),
            dependencies: (0..dependencies).map(|n| [n as u8; 32]).collect(),
            confidence_bits: 0,
            latency_millis: 0,
            verification,
        }
    }

    #[test]
    fn pod_events_are_bounded_before_encoding_and_stay_readable() {
        // Over the decoder's limits: refused up front instead of appended and
        // then unreadable (or, for the finding count, a panic in the encoder).
        for (provenance, dependencies) in [
            (MAX_HYPOTHESIS_PROVENANCE + 1, 0),
            (0, MAX_HYPOTHESIS_DEPENDENCIES + 1),
        ] {
            let error = check_encodable(&pod_hypothesis(provenance, dependencies, attestation()))
                .unwrap_err();
            assert_eq!(error.to_string(), "PTR_LEDGER_HYPOTHESIS_LIMIT");
        }
        let too_many_findings = Attestation {
            findings: (0..=255).map(|n| format!("f{n}")).collect(),
            ..attestation()
        };
        let error = check_encodable(&pod_hypothesis(0, 0, too_many_findings.clone())).unwrap_err();
        assert_eq!(error.to_string(), "PTR_LEDGER_ATTESTATION_LIMIT");
        let admitted = LedgerEvent::PodOutputAdmitted {
            request_id: RequestId("request".into()),
            session_id: SessionId("session".into()),
            scope_id: ScopeId("scope".into()),
            pod_id: PodId("pod".into()),
            manifest_digest: [1; 32],
            artifact_digest: [2; 32],
            generation: Generation(1),
            revision: Revision(1),
            output_kind: 0,
            output_type: TypeId("type".into()),
            output_digest: [3; 32],
            verification: too_many_findings,
        };
        assert!(check_encodable(&admitted).is_err());

        // Exactly at the limits it is accepted and round-trips through the codec.
        let at_limits = pod_hypothesis(
            MAX_HYPOTHESIS_PROVENANCE,
            MAX_HYPOTHESIS_DEPENDENCIES,
            attestation(),
        );
        check_encodable(&at_limits).unwrap();
        let payload = encode_event(&at_limits);
        assert_eq!(decode_event(&payload).unwrap(), at_limits);
    }

    #[test]
    fn an_origin_outside_its_bounds_is_refused_before_it_is_encoded() {
        let host = |verifiers: usize, findings: usize| {
            committed(SemanticOrigin::Host {
                principal: "operator".into(),
                verification: Attestation {
                    verifiers: (0..verifiers).map(|n| format!("v{n}")).collect(),
                    findings: (0..findings).map(|n| format!("v0/f{n}")).collect(),
                    ..attestation()
                },
            })
        };
        for (verifiers, findings) in [
            (0, 0),
            (MAX_ATTESTATION_VERIFIERS + 1, 0),
            (1, MAX_ATTESTATION_FINDINGS + 1),
        ] {
            let error = check_encodable(&host(verifiers, findings)).unwrap_err();
            assert_eq!(error.to_string(), "PTR_LEDGER_ATTESTATION_LIMIT");
        }
        assert!(
            check_encodable(&host(MAX_ATTESTATION_VERIFIERS, MAX_ATTESTATION_FINDINGS)).is_ok()
        );

        let SemanticOrigin::Merge(mut record) = merge(
            MergeAuthorityRecord::Reviewed {
                reviewer: "bob".into(),
            },
            &[],
        ) else {
            unreachable!("merge builds a merge origin")
        };
        record.rebased = (0..=MAX_REBASED_KEYS).map(|n| format!("k{n:05}")).collect();
        let error = check_encodable(&committed(SemanticOrigin::Merge(record.clone()))).unwrap_err();
        assert_eq!(error.to_string(), "PTR_LEDGER_REBASED_KEY_LIMIT");
        record.rebased.pop_last();
        assert!(check_encodable(&committed(SemanticOrigin::Merge(record))).is_ok());

        let mut empty_merge = merge(
            MergeAuthorityRecord::Reviewed {
                reviewer: "bob".into(),
            },
            &[],
        );
        if let SemanticOrigin::Merge(record) = &mut empty_merge {
            record.verification.verifiers.clear();
        }
        let error = check_encodable(&committed(empty_merge)).unwrap_err();
        assert_eq!(error.to_string(), "PTR_LEDGER_ATTESTATION_LIMIT");

        // A requirement below FullSemantic, for a host write and a merge.
        for required in [
            VerificationLevel::Unverified,
            VerificationLevel::LatentAgreement,
            VerificationLevel::SampleVerified,
        ] {
            let verification = Attestation {
                required,
                ..attestation()
            };
            let host_write = committed(SemanticOrigin::Host {
                principal: "operator".into(),
                verification: verification.clone(),
            });
            let SemanticOrigin::Merge(mut record) = merge(
                MergeAuthorityRecord::Reviewed {
                    reviewer: "bob".into(),
                },
                &[],
            ) else {
                unreachable!("merge builds a merge origin")
            };
            record.verification = verification;
            for event in [host_write, committed(SemanticOrigin::Merge(record))] {
                let error = check_encodable(&event).unwrap_err();
                assert_eq!(error.to_string(), "PTR_LEDGER_ATTESTATION_REQUIREMENT");
            }
        }
    }

    /// The writer's check and the decoder cannot drift apart: over every
    /// requirement and every count at and beyond its bounds, an event is
    /// refused before encoding exactly when the decoder refuses its encoding,
    /// with the same code, and an accepted one decodes to itself.
    #[test]
    fn an_event_is_refused_before_encoding_exactly_when_its_encoding_is_refused() {
        let mut events = Vec::new();
        for required in [
            VerificationLevel::Unverified,
            VerificationLevel::LatentAgreement,
            VerificationLevel::SampleVerified,
            VerificationLevel::FullSemantic,
            VerificationLevel::Deterministic,
        ] {
            for verifiers in [
                0,
                1,
                MAX_ATTESTATION_VERIFIERS,
                MAX_ATTESTATION_VERIFIERS + 1,
            ] {
                for findings in [0, MAX_ATTESTATION_FINDINGS, MAX_ATTESTATION_FINDINGS + 1] {
                    let verification = Attestation {
                        required,
                        level: VerificationLevel::Unverified,
                        verifiers: (0..verifiers).map(|n| format!("v{n:02}")).collect(),
                        findings: (0..findings).map(|n| format!("v00/f{n:02}")).collect(),
                    };
                    events.push(committed(SemanticOrigin::Host {
                        principal: "operator".into(),
                        verification: verification.clone(),
                    }));
                    let SemanticOrigin::Merge(mut record) = merge(
                        MergeAuthorityRecord::Triage {
                            policy_version: "policy-3".into(),
                            score_bits: 0.5f32.to_bits(),
                        },
                        &["doc:a"],
                    ) else {
                        unreachable!("merge builds a merge origin")
                    };
                    record.verification = verification;
                    events.push(committed(SemanticOrigin::Merge(record)));
                }
            }
        }
        for keys in [0, MAX_REBASED_KEYS, MAX_REBASED_KEYS + 1] {
            let SemanticOrigin::Merge(mut record) = merge(
                MergeAuthorityRecord::Reviewed {
                    reviewer: "bob".into(),
                },
                &[],
            ) else {
                unreachable!("merge builds a merge origin")
            };
            record.rebased = (0..keys).map(|n| format!("k{n:05}")).collect();
            events.push(committed(SemanticOrigin::Merge(record)));
        }

        let (mut accepted, mut refused) = (0, 0);
        for event in &events {
            match (check_encodable(event), decode_event(&encode_event(event))) {
                (Ok(()), Ok(decoded)) => {
                    assert_eq!(&decoded, event);
                    accepted += 1;
                }
                (Err(before), Err(after)) => {
                    assert_eq!(before.to_string(), after.to_string(), "{event:?}");
                    refused += 1;
                }
                (before, after) => panic!(
                    "the check and the decoder disagree: {before:?} before encoding, {:?} after, for {event:?}",
                    after.map(|_| ())
                ),
            }
        }
        // Both sides of every rule are exercised: 2 of 5 requirements, 2 of 4
        // verifier counts and 2 of 3 finding counts pass, for a host write
        // and a merge, plus two of the three rebased-key counts.
        assert_eq!(accepted, 2 * (2 * 2 * 2) + 2);
        assert_eq!(refused, events.len() - accepted);
    }

    #[test]
    fn protected_state_reference_round_trips_without_payload_or_key_material() {
        let event = LedgerEvent::ProtectedStateCommitted {
            domain: 3,
            logical_id: "kv/session-1".into(),
            generation: Generation(2),
            revision: Revision(9),
            plaintext_digest: [1; 32],
            ciphertext_digest: [2; 32],
            anchor_digest: [3; 32],
        };
        let encoded = encode_event(&event);
        assert_eq!(encoded[0], 14);
        assert!(!encoded.windows(7).any(|window| window == b"secret!"));
        assert_eq!(decode_event(&encoded).unwrap(), event);
    }

    #[test]
    fn encoded_event_is_the_payload_the_decoder_reads_back() {
        let event = LedgerEvent::ProtectedStateCommitted {
            domain: 3,
            logical_id: "kv/session-1".into(),
            generation: Generation(2),
            revision: Revision(9),
            plaintext_digest: [1; 32],
            ciphertext_digest: [2; 32],
            anchor_digest: [3; 32],
        };
        let encoded = encoded_event(&event);
        assert_eq!(encoded, encode_event(&event));
        assert_eq!(decode_event(&encoded).unwrap(), event);
    }
}

#[cfg(test)]
mod adapter_event_tests {
    use super::*;

    fn promoted() -> LedgerEvent {
        LedgerEvent::AdapterPromoted {
            project: ProjectId("proj".into()),
            adapter: "coder-v2".into(),
            generation: Generation(2),
            base: [1; 32],
            artifact_sha256: [2; 32],
            gate_report_digest: [3; 32],
        }
    }

    fn revoked() -> LedgerEvent {
        LedgerEvent::AdapterRevoked {
            project: ProjectId("proj".into()),
            adapter: "coder-v2".into(),
            generation: Generation(2),
            reason: "training input revoked".into(),
        }
    }

    fn cycle(outcome: ImprovementCycleOutcome) -> LedgerEvent {
        LedgerEvent::ImprovementCycleRecorded {
            project: ProjectId("proj".into()),
            day: 20_000,
            outcome,
            policy_digest: [4; 32],
        }
    }

    fn all_events() -> Vec<LedgerEvent> {
        vec![
            promoted(),
            revoked(),
            cycle(ImprovementCycleOutcome::Completed),
            cycle(ImprovementCycleOutcome::Aborted {
                reason: "budget guard".into(),
            }),
        ]
    }

    #[test]
    fn adapter_events_round_trip_with_their_pinned_tags() {
        let tags = [39u8, 40, 41, 41];
        for (event, tag) in all_events().into_iter().zip(tags) {
            let encoded = encode_event(&event);
            assert_eq!(encoded[0], tag);
            assert_eq!(decode_event(&encoded).expect("decodes"), event);
            check_encodable(&event).expect("encodable");
            for len in 0..encoded.len() {
                assert!(decode_event(&encoded[..len]).is_err(), "prefix {len}");
            }
            let mut padded = encoded.clone();
            padded.push(0);
            assert!(decode_event(&padded).is_err(), "trailing byte");
        }
    }

    #[test]
    fn a_malformed_adapter_event_is_refused_by_the_encoder_check_and_the_decoder() {
        let mut bad = Vec::new();
        for field in 0..8 {
            let mut event = promoted();
            if let LedgerEvent::AdapterPromoted {
                project,
                adapter,
                generation,
                base,
                artifact_sha256,
                gate_report_digest,
            } = &mut event
            {
                match field {
                    0 => adapter.clear(),
                    1 => *adapter = "a/b".into(),
                    2 => *adapter = "a b".into(),
                    3 => *adapter = "x".repeat(MAX_ADAPTER_NAME_BYTES + 1),
                    4 => *generation = Generation(0),
                    5 => *base = [0; 32],
                    6 => *artifact_sha256 = [0; 32],
                    _ => {
                        *gate_report_digest = [0; 32];
                        project.0.clear();
                    }
                }
            }
            bad.push(event);
        }
        for reason in ["", "line\nbreak", &"r".repeat(MAX_ADAPTER_REASON_BYTES + 1)] {
            bad.push(LedgerEvent::AdapterRevoked {
                project: ProjectId("proj".into()),
                adapter: "a".into(),
                generation: Generation(1),
                reason: reason.to_owned(),
            });
            bad.push(cycle(ImprovementCycleOutcome::Aborted {
                reason: reason.to_owned(),
            }));
        }
        bad.push(LedgerEvent::ImprovementCycleRecorded {
            project: ProjectId("proj".into()),
            day: 1,
            outcome: ImprovementCycleOutcome::Completed,
            policy_digest: [0; 32],
        });
        for event in bad {
            assert!(check_encodable(&event).is_err(), "{event:?}");
            assert!(decode_event(&encode_event(&event)).is_err(), "{event:?}");
        }
    }

    #[test]
    fn an_unknown_cycle_outcome_tag_is_refused() {
        let mut encoded = encode_event(&cycle(ImprovementCycleOutcome::Completed));
        // tag, project (u32 length + 4 bytes), day (8 bytes), then the outcome byte.
        let outcome_at = 1 + 4 + 4 + 8;
        encoded[outcome_at] = 2;
        assert!(decode_event(&encoded).is_err());
    }
}
