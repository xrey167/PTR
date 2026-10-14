//! Admission of tensor handles (doc 42, section 1): a handle is admitted exactly like an
//! activation frame, before anything is mapped.
//!
//! This module only **decides**. It holds no pointer, maps no memory and opens no file; the
//! carriers that do so are other code. What it adds to the frame path
//! ([`admit_frame`](crate::split_stage::admit_frame)) is the part a frame does not need,
//! because a frame carries its bytes and a handle carries a claim on bytes somebody else holds:
//!
//! 1. **Admission before access** ([`HandleTable::admit`]): the principal is registered, the
//!    handle decodes, its Ed25519 signature verifies (`verify_strict`) over
//!    [`handle_signing_bytes`](ptr_protocol::handle_signing_bytes), the chain's stage is the
//!    stage identity bound to that principal, the handle is addressed to this table's consumer
//!    stage and to the operation of the call, the nonce was not used, the signed deadline is
//!    not reached and not too far ahead, the table has room, and the chain and the placement
//!    epoch are live now ([`revalidate_chain_in`]). The same checks as `ptr-podwire`'s chunk
//!    assembler, in the same order, with nothing table-wide read before the signature verifies.
//!    The handle carries no input lineage (codec v1 has none); the lineage is derived from the
//!    authority instead: [`OperationLiveness`] answers, for the signed operation, whether every
//!    input that operation read (semantic, memory, derived) is still live at the current
//!    committed anchor, and whether the operation belongs to the project of the call, is served
//!    by the recorded consumer worker session and runs under the recorded producer worker
//!    session (the handle carries none of the three: the project and the consumer session come
//!    from the consumer's call context, the producer session from the producer registration).
//!    An unknown operation, another project, a stale session or a dead input is refused. The
//!    runtime provides the implementation when it wires the worker protocol (a follow-up);
//!    until then the caller supplies it and the tests use a fake.
//!    The same record also holds the chain and the payload types the operation expects, so a live
//!    but different adapter or expert, or a payload type the operation does not accept, is
//!    refused.
//!    **Exact-only.** Like `admit_frame`, codec v1 admits only Exact operations: the authority's
//!    record holds the operation's KV reuse mode and taint, and an Approximate or tainted
//!    operation is refused (`OperationFault::NotExact`). Carrying mode and taint in the handle
//!    itself is a later codec version.
//! 2. **Revalidation at use** ([`HandleTable::consume`]): the consumer may touch the memory
//!    only after the chain is revalidated again, because a fence can fall between admission
//!    and use (rule 4), and the operation is checked again for the same reason. The locator
//!    is readable only from the [`ConsumedHandle`] this returns,
//!    never from the [`AdmittedHandle`].
//!    A fault at `consume` cannot cover computation that has already started on the mapped
//!    tensor: the runtime must also revalidate the operation when it admits the computed output
//!    (doc 42 rule 4), for which [`revalidate_operation`] is provided. That obligation belongs
//!    to the worker-protocol wiring.
//! 3. **Single use with a deadline** (rule 2): the nonce is tombstoned on consumption, and the
//!    tombstone is kept until the signed deadline plus the skew, after which the handle is
//!    refused as expired anyway.
//! 4. **Producer-owned memory** (rules 3 and 3b): every claimed buffer is reported once, as a
//!    [`ReclaimNotice`], when the producer may reclaim it, and its bytes stay charged to the
//!    producer's in-flight budget until then. **The producer reclaims on the notice, not
//!    earlier.** For a carrier that is not shared with another process (`InProcess`, `Bytes`,
//!    `NvmeFile`, `Remote`) the notice comes from [`HandleTable::finish`] once the consumer is
//!    done with the [`ConsumedHandle`] (`consume` itself releases nothing), on producer
//!    replacement or removal, or from [`HandleTable::expire`] after the deadline plus the skew
//!    (a consumed handle that was dropped unfinished included).
//!    A **shared** carrier (`HostShm`, `CudaIpc`, `DevicePeer`, `PinnedHost`) is released only
//!    by [`HandleTable::release_after_teardown`], which the supervisor calls after it has
//!    reaped both worker domains and drained their control sockets: `Consumed` is a message
//!    from a worker that may keep its mapping, so it frees nothing, and neither does the
//!    deadline or a producer replacement. A producer whose shared buffers are not torn down
//!    therefore saturates its budget and is refused with a typed error.
//! 5. **Content commitment**: the handle carries the SHA-256 of the tensor bytes under its
//!    signature, but admission reads no memory and cannot check it. The consumer must map the
//!    sealed, immutable region, call [`ConsumedHandle::verify_content`] on the mapped bytes and
//!    use the tensor only if that succeeds. The producer must not write the region after
//!    signing: ownership of the contents passes to the consumer until `Consumed` or reclaim.
//! 6. **Retention through the skew window**: the consumer must stop at the signed deadline
//!    ([`HandleTable::admit`] and [`HandleTable::consume`] refuse `now_ms >= deadline`), but
//!    the producer may reclaim a non-shared buffer only after `finish` or once the deadline
//!    **plus** [`HandlePolicy::max_clock_skew_ms`] has passed, never at the bare deadline: its
//!    clock and the consumer's may differ by the skew, and a consumer still inside its deadline
//!    must not see memory reused. [`HandleTable::expire`] returns a [`ReclaimNotice`] only from
//!    deadline plus skew. Output computed from the tensor is admitted only before the deadline
//!    too ([`revalidate_operation`] takes the current time).
//!
//! **Bounds** (invariant 8). Open handles are bounded table-wide
//! ([`HandlePolicy::max_open_handles`]) and per producer
//! ([`HandlePolicy::max_open_per_principal`], so one producer cannot starve the others); open
//! handles plus tombstones per producer by [`HandlePolicy::max_tombstones_per_principal`]; and
//! the bytes a producer still has claimed by
//! [`HandlePolicy::max_inflight_bytes_per_principal`] (counting handles alone would let a few
//! handles pin gigabytes). A claim is held until its [`ReclaimNotice`] is emitted, so an
//! abandoned handle keeps its bytes until [`HandleTable::expire`] (a shared one until
//! [`HandleTable::release_after_teardown`], its slot kept as well). A full table refuses; it
//! never evicts, and slots are freed only by `expire`. The deadline of an admitted handle is at
//! most `now + max_handle_lifetime_ms + max_clock_skew_ms` (computed with saturating
//! arithmetic), so a tombstone cannot be pinned beyond that horizon, except the slot of a
//! shared claim, which stays until its teardown is reported and the next `expire` after it;
//! those slots still count against [`HandlePolicy::max_tombstones_per_principal`] and the
//! byte budget, so a producer that is never torn down is refused, not grown. The set of
//! principals is bounded by the trusted deployment: [`HandleTable::register_producer`] is a host call, never
//! driven by a peer.
//!
//! **Bounded work before authentication.** Before the signature verifies, `admit` does work
//! only for a registered principal and only within the codec bounds: at most
//! `MAX_INLINE_BYTES` (1 MiB) of inline payload plus the chain, consumer and shape bounds, so
//! about 2 MiB transient (the input and its decoded copy). It reads no table-wide state and
//! allocates nothing that outlives a refusal.
//!
//! **The clock is trusted.** The caller drives time: `now_ms` is clamped monotonic but any
//! forward jump is accepted, so a faulty or hostile clock source can expire every handle at
//! once. The caller must call [`HandleTable::expire`] for slots to be freed.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use ed25519_dalek::{Signature, VerifyingKey};
use ptr_protocol::activation::{BoundaryDtype, UpstreamChain};
use ptr_protocol::{
    handle_digest, handle_signing_bytes, verify_content, Carrier, CarrierLocator, HandleError,
    TensorHandle,
};

use ptr_types::{ProjectId, TypeId};

use crate::host_quota::{QuotaCharge, QuotaRefusal, SharedHostQuota};
use crate::split_stage::{revalidate_chain_in, ChainFault, StageLiveness};

/// Identifies a table, so a ticket of one table is no ticket of another.
static NEXT_TABLE_ID: AtomicU64 = AtomicU64::new(1);

/// The bounds of a [`HandleTable`]. Every bound is non-zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandlePolicy {
    /// Handles open at once, table-wide.
    pub max_open_handles: usize,
    /// Handles open at once for one producer; at most `max_open_handles`.
    pub max_open_per_principal: usize,
    /// Open handles plus tombstones kept for one producer.
    pub max_tombstones_per_principal: usize,
    /// The sum of `byte_len` over the handles of one producer whose buffer is still claimed:
    /// from admission until the [`ReclaimNotice`] is emitted (consume, producer replacement or
    /// removal, or expiry at deadline plus skew). Abandoning a handle releases nothing. A single
    /// handle is bounded by `ptr_protocol::MAX_HANDLE_BYTES`.
    pub max_inflight_bytes_per_principal: u64,
    /// How far ahead of now a signed deadline may lie (before skew).
    pub max_handle_lifetime_ms: u64,
    /// Tolerated clock difference between producer and consumer.
    pub max_clock_skew_ms: u64,
}

/// Which bound of a [`HandlePolicy`], or which part of a table's identity, is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandlePolicyViolation {
    ZeroOpenHandles,
    ZeroOpenPerPrincipal,
    /// `max_open_per_principal` is above `max_open_handles`.
    OpenPerPrincipalAboveTotal,
    ZeroTombstones,
    ZeroInflightBytes,
    ZeroLifetime,
    ZeroClockSkew,
    /// The consumer stage identity of the table is empty.
    EmptyConsumer,
}

impl HandlePolicy {
    /// Every bound must be non-zero, and the per-producer open bound at most the table-wide one.
    pub fn validate(&self) -> Result<(), HandlePolicyViolation> {
        if self.max_open_handles == 0 {
            return Err(HandlePolicyViolation::ZeroOpenHandles);
        }
        if self.max_open_per_principal == 0 {
            return Err(HandlePolicyViolation::ZeroOpenPerPrincipal);
        }
        if self.max_open_per_principal > self.max_open_handles {
            return Err(HandlePolicyViolation::OpenPerPrincipalAboveTotal);
        }
        if self.max_tombstones_per_principal == 0 {
            return Err(HandlePolicyViolation::ZeroTombstones);
        }
        if self.max_inflight_bytes_per_principal == 0 {
            return Err(HandlePolicyViolation::ZeroInflightBytes);
        }
        if self.max_handle_lifetime_ms == 0 {
            return Err(HandlePolicyViolation::ZeroLifetime);
        }
        if self.max_clock_skew_ms == 0 {
            return Err(HandlePolicyViolation::ZeroClockSkew);
        }
        Ok(())
    }
}

/// The random id of one spawn of a worker, drawn by the supervisor at every spawn of that
/// worker. A respawned worker has a new session, so a record bound to the old session no longer
/// matches (doc 42 section 1).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WorkerSession(pub u64);

/// Why an operation's inputs are not usable now.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationFault {
    /// The authority has no record of the operation.
    UnknownOperation { operation: u64 },
    /// An input the operation read is no longer live at the committed anchor.
    InputNotLive { operation: u64, input: String },
    /// The operation does not run in Exact KV reuse mode, or carries a taint. Codec v1 handles
    /// are Exact-only, like `admit_frame`: they carry neither mode nor taint.
    NotExact { operation: u64 },
    /// The handle's upstream chain is not exactly the chain the operation expects.
    ChainMismatch { operation: u64 },
    /// The operation does not accept the handle's payload type.
    PayloadTypeNotAccepted {
        operation: u64,
        payload_type: TypeId,
    },
    /// The operation belongs to another project than the one of the call.
    WrongProject { operation: u64 },
    /// The consumer stage is no longer served by the worker session the operation recorded.
    ConsumerSessionMismatch { operation: u64 },
    /// The producer no longer runs under the worker session the operation recorded.
    ProducerSessionMismatch { operation: u64 },
    /// The handle's signed deadline lies beyond the operation's absolute deadline.
    DeadlineBeyondOperation { operation: u64 },
    /// The stage accepting the handle is not the destination stage the operation recorded.
    ConsumerStageMismatch { operation: u64 },
}

/// What admission and use ask the authority about one handle.
#[derive(Clone, Copy, Debug)]
pub struct OperationCheck<'a> {
    /// The operation the handle was signed for.
    pub parent_operation_id: u64,
    /// The model-side chain the handle claims (stage, base model, adapters, experts).
    pub upstream_chain: &'a UpstreamChain,
    /// The schema and argument slot the handle claims for its tensor.
    pub payload_type: &'a TypeId,
    /// The project of the consumer's call, derived by the authority, never read from the handle.
    pub project: &'a ProjectId,
    /// The consumer stage of the table accepting the handle (the deployment's stage identity,
    /// equal to the handle's `consumer` once admission checked it). One worker session may serve
    /// several stages, so the session alone does not prove the destination: the authority
    /// compares this with the destination stage of its record (`ConsumerStageMismatch`).
    pub consumer_stage: &'a str,
    /// The session of the consumer worker that serves the consumer stage, from the consumer's
    /// call context, never from the handle.
    pub consumer_session: WorkerSession,
    /// The session the producer worker runs under, from the producer registration, never from
    /// the handle.
    pub producer_session: WorkerSession,
    /// The handle's signed absolute deadline; the authority refuses a handle whose deadline lies
    /// beyond the operation's own absolute deadline (`DeadlineBeyondOperation`).
    pub signed_absolute_deadline_ms: u64,
}

/// What the handle admission asks the authority about the operation a handle was produced for.
///
/// The authority process owns a record of every operation it started. The record binds the
/// authority-derived project, the consumer worker's session id and the producer worker's session
/// id (both drawn by the supervisor at every spawn of that worker), and holds the generations of
/// every input the operation read (semantic, memory and derived state), the ordered Raw ancestry it
/// read (`(Raw, content_digest)` pairs, since Raw segments carry no generation), the destination
/// stage and placement (the consumer stage, its placement and epoch when the operation started), the model-side
/// chain it expects (stage, base model, adapters and experts with identity, generation, digest and
/// fence epoch) and the payload types it accepts. Every lookup is keyed by the project: an
/// operation of another project is refused with `WrongProject`, never answered with a detail.
///
/// It answers `Ok` only if the operation is known, belongs to the project of the call, every input
/// is still live at the current committed anchor and every Raw ancestor is still present with its
/// digest (otherwise `InputNotLive`), the accepting stage is the recorded destination stage
/// (`ConsumerStageMismatch`), the consumer still holds the recorded destination placement
/// and is served by the recorded consumer session (`ConsumerSessionMismatch`; re-placing a consumer
/// or respawning its worker cancels its operations, which are then unknown), the producer still
/// runs under the recorded producer session (`ProducerSessionMismatch`), the handle's chain equals
/// the expected chain exactly, the payload type is one the operation accepts, and the operation ran
/// in Exact KV reuse mode without any taint (Approximate or tainted operations are refused with
/// `NotExact`: codec v1 handles carry neither mode nor taint, so they are not admitted; carrying
/// them in the handle is a later codec version). The handle itself carries no input lineage, no
/// project and no session; this is how admission and use prove them from the authority.
///
/// The record is size-bounded: the count of each input category, the size of each identity, the
/// encoded bytes per record and in total. Starting an operation that would exceed a bound is
/// refused with a typed saturation refusal when the operation is started, never later. The record
/// carries an absolute operation deadline minted by the authority: a handle whose
/// `signed_absolute_deadline_ms` lies beyond it is refused (`DeadlineBeyondOperation`),
/// and the record is retained until that deadline plus the clock skew. For a `Remote` handle the
/// destination authority revalidates with the source authority, which owns the record.
///
/// When several faults apply at once, which one is reported is unspecified. Callers must treat
/// every fault as a refusal and must not branch on precedence.
pub trait OperationLiveness {
    fn operation_admits(&self, check: OperationCheck<'_>) -> Result<(), OperationFault>;
}

/// Why a producer registration was refused. Nothing is changed by a refused registration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RegistrationRefused {
    /// The key or the stage identity changed under the same [`WorkerSession`]. A respawned
    /// producer always runs under a new session (the supervisor draws one at every spawn), and
    /// handles already consumed are fenced only through the session the authority checks, so a
    /// new identity under the old session is refused.
    SessionNotRenewed,
}

/// Why a handle was refused, or why an admitted handle could not be used.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HandleRejection {
    /// No producer is registered under the principal.
    UnknownPrincipal,
    /// The bytes or the decoded handle are not a well-formed handle.
    Malformed(HandleError),
    /// The signature does not verify under the principal's registered key.
    BadSignature,
    /// The chain's stage is not the stage identity bound to the principal.
    StageNotBound,
    /// The handle is addressed to another consumer stage than this table's.
    WrongConsumer { expected: String, found: String },
    /// The handle was produced for another operation than the one of the call.
    WrongOperation { expected: u64, found: u64 },
    /// The nonce was consumed already.
    Replayed,
    /// The nonce was abandoned (by the consumer, by a failed use or by a replaced producer).
    Abandoned,
    /// The nonce is open: the handle was admitted and not yet used.
    DuplicateOpen,
    /// The signed deadline is reached: `deadline <= now`.
    Expired,
    /// The signed deadline lies beyond now plus the lifetime bound plus the skew.
    DeadlineTooFar,
    /// The table holds its maximum of open handles.
    TooManyOpenHandles { limit: usize },
    /// The producer holds its maximum of open handles.
    TooManyOpenForPrincipal { limit: usize },
    /// The producer's table holds its maximum of open handles and tombstones.
    TombstoneTableFull { limit: usize },
    /// The footprint the caller charges is smaller than the tensor it holds.
    FootprintBelowLength { footprint: u64, byte_len: u64 },
    /// Admitting the handle would put more than the budget of bytes in flight for the producer.
    InflightBytesExceeded {
        limit: u64,
        in_flight: u64,
        requested: u64,
    },
    /// The host-wide byte quota ([`SharedHostQuota`]) refused the footprint: the producer
    /// session, the consumer session or the host would exceed its cap.
    HostQuota(QuotaRefusal),
    /// The mapped bytes do not match the handle's content commitment.
    Content(HandleError),
    /// The chain or the placement epoch is not live now.
    Chain(ChainFault),
    /// The operation is unknown, or an input it read is no longer live.
    Operation(OperationFault),
    /// The ticket is no open use of this table (another table, swept, or superseded).
    UnknownTicket,
}

/// Why the producer may reclaim a buffer.
///
/// A notice releases this table's admission budget for the handle's bytes. For a carrier that is
/// not shared with another process (`InProcess`, `Bytes`, `NvmeFile`, `Remote`) the notice comes
/// from [`HandleTable::finish`] (with [`ReclaimReason::Consumed`]), on producer replacement or
/// removal, or at the deadline plus the skew. For [`ReclaimReason::Expired`] and [`ReclaimReason::Abandoned`] the consumer has not
/// confirmed that it is done, so the driver must first terminate and reap the consumer worker
/// before the producer frees the buffer.
///
/// A **shared** carrier (`HostShm`, `CudaIpc`, `DevicePeer`, `PinnedHost`) is released only with
/// [`ReclaimReason::TornDown`], from [`HandleTable::release_after_teardown`]: never at
/// `Consumed`, never at the deadline alone and never because the producer was replaced or
/// removed (doc 42 rule 3b). Until then its bytes stay charged to the producer's budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReclaimReason {
    /// The consumer used the handle (non-shared carriers only).
    Consumed,
    /// The deadline plus the skew passed on an open handle.
    Expired,
    /// The handle was abandoned and its deadline plus the skew passed.
    Abandoned,
    /// The producer was registered again under another session (a respawn).
    ProducerReplaced,
    /// The producer was removed with [`HandleTable::unregister_producer`].
    ProducerRemoved,
    /// The supervisor verified the teardown of both worker domains (shared carriers only).
    TornDown,
}

/// What the producer may reclaim, reported once per handle. The locator of a
/// [`Carrier::Bytes`] handle is reported with its inline payload dropped: those bytes were
/// never the table's to hold.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReclaimNotice {
    pub principal: String,
    pub handle_nonce: u64,
    pub handle_digest: [u8; 32],
    pub carrier: Carrier,
    pub locator: CarrierLocator,
    /// The tensor's length.
    pub byte_len: u64,
    /// The allocated footprint that was charged to the budget and is released with this notice.
    pub charged_bytes: u64,
    pub reason: ReclaimReason,
}

/// The handle of a request, still encoded or already decoded.
#[derive(Clone, Copy)]
pub enum HandleSource<'a> {
    Encoded(&'a [u8]),
    Decoded(&'a TensorHandle),
}

impl fmt::Debug for HandleSource<'_> {
    /// Names the form and size only: neither the bytes nor the locator are printed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoded(bytes) => write!(formatter, "Encoded({} bytes)", bytes.len()),
            Self::Decoded(handle) => write!(formatter, "Decoded(nonce {})", handle.handle_nonce),
        }
    }
}

/// One handle presented for admission.
#[derive(Clone, Copy)]
pub struct HandleAdmissionRequest<'a> {
    pub now_ms: u64,
    /// The authenticated principal of the connection, never a value read from the handle.
    pub principal: &'a str,
    /// The operation of the consumer's call, supplied by the consumer's call context, never by
    /// the handle. A handle signed for another operation is refused.
    pub expected_operation: u64,
    pub handle: HandleSource<'a>,
    pub live: &'a dyn StageLiveness,
    /// The authority's answer about the operation's inputs.
    pub operations: &'a dyn OperationLiveness,
    /// The authority-derived project of the consumer's call, supplied by the consumer's call
    /// context, never by the handle.
    pub project: &'a ProjectId,
    /// The session of the consumer worker serving the consumer stage, supplied by the consumer's
    /// call context, never by the handle.
    pub consumer_session: WorkerSession,
    /// The **allocated footprint** of the buffer behind the handle, charged to the budget instead
    /// of the logical `byte_len` (doc 42 rule 3b): the VMM size rounded to the allocation
    /// granularity, a legacy CUDA allocation rounded to 2 MiB, a `memfd` rounded to whole pages,
    /// padding included, so many small handles cannot hold more memory than the cap. It comes
    /// from the supervisor's export record for the handle, never from the handle. A value below
    /// `byte_len` is refused (`FootprintBelowLength`); for a carrier without an allocation of its
    /// own (`Bytes`, `InProcess` under moved ownership) the caller passes `byte_len`.
    pub footprint_bytes: u64,
}

/// One admitted handle presented for use.
pub struct ConsumeRequest<'a> {
    pub now_ms: u64,
    pub admitted: AdmittedHandle,
    pub live: &'a dyn StageLiveness,
    pub operations: &'a dyn OperationLiveness,
}

/// A verified teardown the supervisor reports to [`HandleTable::release_after_teardown`]: the
/// producer worker of `principal` that ran under `producer_session` and the consumer worker that
/// ran under `consumer_session` have both been terminated and reaped, and the old control socket
/// of each has been drained.
#[derive(Clone, Copy, Debug)]
pub struct Teardown<'a> {
    pub principal: &'a str,
    pub producer_session: WorkerSession,
    pub consumer_session: WorkerSession,
}

/// A handle whose signature, stage, audience, deadline, chain and placement epoch were good
/// when it was admitted. It is a single-use ticket: it is not `Clone`, and
/// [`HandleTable::consume`] and [`HandleTable::abandon`] take it by value. It shows what the
/// consumer needs to plan the use; the locator is withheld until [`HandleTable::consume`] has
/// revalidated, and `Debug` omits it.
#[derive(Eq, PartialEq)]
pub struct AdmittedHandle {
    table: u64,
    ticket: u64,
    principal: String,
    handle: TensorHandle,
    /// Captured at admission; `consume` hands them to the authority again.
    project: ProjectId,
    consumer_session: WorkerSession,
    producer_session: WorkerSession,
}

impl fmt::Debug for AdmittedHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmittedHandle")
            .field("principal", &self.principal)
            .field("handle_nonce", &self.handle.handle_nonce)
            .field("carrier", &self.handle.carrier)
            .field("byte_len", &self.handle.byte_len)
            .finish_non_exhaustive()
    }
}

impl AdmittedHandle {
    pub fn principal(&self) -> &str {
        &self.principal
    }

    /// The authority-derived project the handle was admitted for; the placement epoch is
    /// revalidated for it ([`revalidate_chain_in`]).
    pub fn project(&self) -> &ProjectId {
        &self.project
    }

    pub fn handle_nonce(&self) -> u64 {
        self.handle.handle_nonce
    }

    pub fn parent_operation_id(&self) -> u64 {
        self.handle.parent_operation_id
    }

    pub fn handle_digest(&self) -> [u8; 32] {
        handle_digest(&self.handle)
    }

    pub fn carrier(&self) -> Carrier {
        self.handle.carrier
    }

    pub fn dtype(&self) -> BoundaryDtype {
        self.handle.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.handle.shape
    }

    pub fn byte_len(&self) -> u64 {
        self.handle.byte_len
    }

    pub fn placement_epoch(&self) -> u64 {
        self.handle.placement_epoch
    }

    pub fn chain(&self) -> &UpstreamChain {
        &self.handle.upstream_chain
    }

    pub fn signed_absolute_deadline_ms(&self) -> u64 {
        self.handle.signed_absolute_deadline_ms
    }
}

/// A handle revalidated at use: the consumer may now reach the tensor through its locator.
///
/// Its bytes stay charged to the producer's budget while it is in use. When the consumer is done
/// (after [`revalidate_operation`] admitted the output), the driver hands it to
/// [`HandleTable::finish`]: for a carrier that is not shared that returns the [`ReclaimNotice`]
/// that lets the producer reclaim the buffer and releases the bytes then, not earlier (doc 42
/// rule 3). A shared carrier stays charged until [`HandleTable::release_after_teardown`] (doc
/// 42 rule 3b). A handle that is dropped instead keeps its bytes until [`HandleTable::expire`]
/// reports it at the deadline plus the skew.
#[must_use = "finish the handle, or its bytes stay charged until the deadline plus the skew"]
#[derive(Eq, PartialEq)]
pub struct ConsumedHandle {
    table: u64,
    /// The ticket of the use; a later use of the same nonce has another one.
    ticket: u64,
    handle: TensorHandle,
    principal: String,
    /// Captured at admission; `revalidate_operation` hands them to the authority again.
    project: ProjectId,
    consumer_session: WorkerSession,
    producer_session: WorkerSession,
}

impl fmt::Debug for ConsumedHandle {
    /// Redacted like [`AdmittedHandle`]: no locator, no inline bytes, no payload type.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConsumedHandle")
            .field("principal", &self.principal)
            .field("handle_nonce", &self.handle.handle_nonce)
            .field("carrier", &self.handle.carrier)
            .field("byte_len", &self.handle.byte_len)
            .finish_non_exhaustive()
    }
}

impl ConsumedHandle {
    pub fn handle(&self) -> &TensorHandle {
        &self.handle
    }

    pub fn locator(&self) -> &CarrierLocator {
        &self.handle.locator
    }

    /// Verify the bytes the consumer mapped against the handle's content commitment (length
    /// and SHA-256). The consumer must call this after mapping and before using the tensor.
    pub fn verify_content(&self, bytes: &[u8]) -> Result<(), HandleRejection> {
        verify_content(&self.handle, bytes).map_err(HandleRejection::Content)
    }

    pub fn principal(&self) -> &str {
        &self.principal
    }

    /// The authority-derived project captured at admission: pass it to [`revalidate_chain_in`]
    /// when the output is admitted.
    pub fn project(&self) -> &ProjectId {
        &self.project
    }
}

/// Revalidate the operation of a consumed handle against the authority, for the runtime to call
/// at `now_ms` when it admits the output computed from the tensor (doc 42 rule 4): a fault at
/// `consume` cannot cover computation that started before the fault.
///
/// The handle's own signed deadline is checked first: output admitted at or after it
/// (`deadline <= now`) is refused as [`HandleRejection::Expired`], because the consumer had to
/// stop touching the tensor at the deadline and the producer may reclaim the buffer from the
/// deadline plus the skew. A longer-lived operation does not extend the handle. `now_ms` must
/// come from the same clock that drives the [`HandleTable`] of the handle (the table's clock is
/// trusted and clamped monotonic; this function holds no clock of its own).
///
/// This covers **only the authority's operation record**: the operation is known, its inputs are
/// live, it is Exact, the handle's chain equals the expected chain and the payload type is
/// accepted, and the project and both worker sessions captured at admission are still current. It
/// does not check chain liveness or the placement epoch: the caller must also call
/// [`revalidate_chain_in`] (with a [`StageLiveness`]) on the handle's chain and placement epoch, as a
/// separate call, when it admits the computed output (doc 42 rule 4).
pub fn revalidate_operation(
    consumed: &ConsumedHandle,
    operations: &dyn OperationLiveness,
    now_ms: u64,
) -> Result<(), HandleRejection> {
    let handle = consumed.handle();
    if handle.signed_absolute_deadline_ms <= now_ms {
        return Err(HandleRejection::Expired);
    }
    operations
        .operation_admits(OperationCheck {
            parent_operation_id: handle.parent_operation_id,
            upstream_chain: &handle.upstream_chain,
            payload_type: &handle.payload_type,
            project: &consumed.project,
            consumer_stage: &handle.consumer,
            consumer_session: consumed.consumer_session,
            producer_session: consumed.producer_session,
            signed_absolute_deadline_ms: handle.signed_absolute_deadline_ms,
        })
        .map_err(HandleRejection::Operation)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotState {
    Open {
        ticket: u64,
    },
    /// Consumed under `ticket`: only the [`ConsumedHandle`] of that use may finish it, never a
    /// stale one of an earlier use of the same nonce.
    Consumed {
        ticket: u64,
    },
    Abandoned,
}

/// What the producer still owns for a slot (absent once reported).
struct Claim {
    carrier: Carrier,
    locator: CarrierLocator,
    byte_len: u64,
    /// The allocated footprint charged to the budget (at least `byte_len`).
    charged_bytes: u64,
    /// The charge held in the host-wide ledger, if the table has one; handed back exactly once,
    /// when this claim is taken (with the notice that frees the buffer, or at teardown).
    host: Option<QuotaCharge>,
}

struct Slot {
    state: SlotState,
    deadline_ms: u64,
    digest: [u8; 32],
    claim: Option<Claim>,
    /// The worker sessions the handle was admitted under; a teardown releases only the slots
    /// of the pair it reaped.
    producer_session: WorkerSession,
    consumer_session: WorkerSession,
}

struct Producer {
    key: VerifyingKey,
    stage_identity: String,
    /// The session the producer worker runs under; the authority checks it on every use.
    session: WorkerSession,
    /// False once removed with `unregister_producer`; the entry stays while it holds slots,
    /// so the tombstones keep refusing replays.
    active: bool,
    /// Open handles and tombstones by nonce.
    slots: BTreeMap<u64, Slot>,
}

/// The bounded table of handles in flight and of used or abandoned nonces, for the handles
/// addressed to one consumer stage.
pub struct HandleTable {
    id: u64,
    policy: HandlePolicy,
    consumer: String,
    clock_ms: u64,
    next_ticket: u64,
    /// Handles in the `Open` state, table-wide, kept as a counter so that admission reads no
    /// table-wide state by scanning.
    open_total: usize,
    producers: BTreeMap<String, Producer>,
    /// The host-wide byte ledger shared with the other tables of the host, if any.
    quota: Option<SharedHostQuota>,
}

impl HandleTable {
    /// A table under `policy` for the handles addressed to the consumer stage `consumer`.
    pub fn new(policy: HandlePolicy, consumer: &str) -> Result<Self, HandlePolicyViolation> {
        policy.validate()?;
        if consumer.is_empty() {
            return Err(HandlePolicyViolation::EmptyConsumer);
        }
        Ok(Self {
            id: NEXT_TABLE_ID.fetch_add(1, Ordering::Relaxed),
            policy,
            consumer: consumer.to_owned(),
            clock_ms: 0,
            next_ticket: 1,
            open_total: 0,
            producers: BTreeMap::new(),
            quota: None,
        })
    }

    /// Like [`Self::new`], and every handle this table admits is also charged to the host-wide
    /// ledger `quota` (doc 42 rule 3b), which the tables of the host share: against its producer
    /// session, its consumer session and the host, with the footprint rounded up to the ledger's
    /// granularity. The charge is taken last in [`Self::admit`], so a refusal
    /// ([`HandleRejection::HostQuota`]) records nothing, and it is handed back with the notice
    /// that frees the buffer (a carrier that is not shared) or in
    /// [`Self::release_after_teardown`] (a shared carrier), once. Dropping a table does not
    /// release its charges: the host's teardown path owns them
    /// ([`SharedHostQuota::release_pair`] with a [`crate::VerifiedTeardown`];
    /// [`SharedHostQuota::tracked_pairs`] finds them).
    pub fn with_host_quota(
        policy: HandlePolicy,
        consumer: &str,
        quota: SharedHostQuota,
    ) -> Result<Self, HandlePolicyViolation> {
        let mut table = Self::new(policy, consumer)?;
        table.quota = Some(quota);
        Ok(table)
    }

    pub fn policy(&self) -> &HandlePolicy {
        &self.policy
    }

    /// Handles admitted and neither used nor abandoned, table-wide.
    pub fn open_handles(&self) -> usize {
        self.open_total
    }

    /// The bytes `principal` still has claimed: the sum of the charged footprints over its slots
    /// whose buffer has not been reported to the producer yet.
    pub fn inflight_bytes(&self, principal: &str) -> Option<u64> {
        self.producers.get(principal).map(claimed_bytes)
    }

    /// Open handles plus tombstones kept for `principal` (`None`: no entry).
    pub fn slots_held(&self, principal: &str) -> Option<usize> {
        self.producers
            .get(principal)
            .map(|producer| producer.slots.len())
    }

    fn advance(&mut self, now_ms: u64) -> u64 {
        self.clock_ms = self.clock_ms.max(now_ms);
        self.clock_ms
    }

    /// Register the verifying key of the producing runtime of an authenticated principal and
    /// the one stage identity it may speak for. Both are whatever the deployment wrote down
    /// for that principal, never a value read from a handle. This is a host call: the set of
    /// principals is bounded by the trusted deployment, not by peers.
    ///
    /// `session` is the producer worker's session id, drawn by the supervisor at every spawn;
    /// admitted handles keep it and the authority compares it with its record on every use.
    ///
    /// Registering the same key, stage identity and session again (a reconnect) changes nothing.
    /// Registering another key or stage identity under the **same** session is refused with
    /// [`RegistrationRefused::SessionNotRenewed`] and changes nothing: an already consumed
    /// handle captured that session, and only a new session fences it at
    /// [`revalidate_operation`]; a supervisor that re-keys a producer must therefore draw a new
    /// session, inactive (unregistered) producers included. Registering under another session
    /// (a respawned producer, whatever its key and stage) abandons the principal's open handles: they can no longer be consumed, their nonces stay tombstoned
    /// until their deadline plus the skew, and the buffers they claimed on a carrier that is not
    /// shared are returned once, with [`ReclaimReason::ProducerReplaced`], which frees their
    /// bytes from the budget. A shared buffer is not returned here: it stays charged until
    /// [`Self::release_after_teardown`] (doc 42 rule 3b).
    #[must_use = "the returned notices name buffers the producer may reclaim"]
    pub fn register_producer(
        &mut self,
        principal: &str,
        key: VerifyingKey,
        stage_identity: &str,
        session: WorkerSession,
    ) -> Result<Vec<ReclaimNotice>, RegistrationRefused> {
        let Some(producer) = self.producers.get_mut(principal) else {
            self.producers.insert(
                principal.to_owned(),
                Producer {
                    key,
                    stage_identity: stage_identity.to_owned(),
                    session,
                    active: true,
                    slots: BTreeMap::new(),
                },
            );
            return Ok(Vec::new());
        };
        let identity_changed = producer.key != key || producer.stage_identity != stage_identity;
        if identity_changed && producer.session == session {
            return Err(RegistrationRefused::SessionNotRenewed);
        }
        producer.active = true;
        if !identity_changed && producer.session == session {
            return Ok(Vec::new());
        }
        producer.key = key;
        producer.stage_identity = stage_identity.to_owned();
        producer.session = session;
        let (released, closed) = abandon_open(
            producer,
            principal,
            ReclaimReason::ProducerReplaced,
            self.quota.as_ref(),
        );
        self.open_total = self.open_total.saturating_sub(closed);
        Ok(released)
    }

    /// Remove a producer: it can no longer present handles, its open handles are abandoned and
    /// their buffers on a carrier that is not shared returned once, with
    /// [`ReclaimReason::ProducerRemoved`]. Its tombstones and the claims of handles abandoned
    /// earlier stay until [`Self::expire`] drops them (shared claims and their slots until
    /// [`Self::release_after_teardown`]), so a replay of a used nonce stays refused if the
    /// principal is registered again.
    #[must_use = "the returned notices name buffers the producer may reclaim"]
    pub fn unregister_producer(&mut self, principal: &str) -> Vec<ReclaimNotice> {
        let Some(producer) = self.producers.get_mut(principal) else {
            return Vec::new();
        };
        producer.active = false;
        let (released, closed) = abandon_open(
            producer,
            principal,
            ReclaimReason::ProducerRemoved,
            self.quota.as_ref(),
        );
        self.open_total = self.open_total.saturating_sub(closed);
        released
    }

    /// Admit one handle at `request.now_ms`. The checks run in this order and the first
    /// failure is returned, with nothing recorded: the clock is clamped monotonic, the
    /// principal is registered, the handle decodes, the signature verifies, the chain's stage
    /// is the principal's bound stage, the handle is addressed to this table's consumer and to
    /// `request.expected_operation`, the authority admits the operation (it is known, every
    /// input is live, it is Exact, the handle's chain equals the expected chain and the payload
    /// type is accepted; see [`OperationFault`]), the nonce is neither tombstoned nor open, the
    /// deadline is not reached (`deadline <= now` is expired), its deadline is not too far
    /// ahead, the table has room (open handles table-wide and per producer, slots, and the
    /// claimed bytes plus `footprint_bytes`, which must be at least `byte_len`), and the chain
    /// and placement epoch are live now.
    ///
    /// The placement epoch is read for `request.project` through
    /// [`StageLiveness::placement_epoch_in`]. This is a fail-closed hard cut, not an additive
    /// change: a `StageLiveness` that only implements the legacy `placement_epoch`, or a stage
    /// placed with the legacy `assign` instead of `assign_in`, refuses every handle
    /// ([`ChainFault::StalePlacement`]). No caller outside tests uses handle admission yet.
    pub fn admit(
        &mut self,
        request: HandleAdmissionRequest<'_>,
    ) -> Result<AdmittedHandle, HandleRejection> {
        let now_ms = self.advance(request.now_ms);
        let policy = self.policy;
        let producer = self
            .producers
            .get_mut(request.principal)
            .filter(|producer| producer.active)
            .ok_or(HandleRejection::UnknownPrincipal)?;
        let handle: Cow<'_, TensorHandle> = match request.handle {
            HandleSource::Encoded(bytes) => {
                Cow::Owned(TensorHandle::decode(bytes).map_err(HandleRejection::Malformed)?)
            }
            HandleSource::Decoded(handle) => {
                handle.validate().map_err(HandleRejection::Malformed)?;
                Cow::Borrowed(handle)
            }
        };
        let signature = Signature::from_bytes(&handle.signature);
        if producer
            .key
            .verify_strict(&handle_signing_bytes(&handle), &signature)
            .is_err()
        {
            return Err(HandleRejection::BadSignature);
        }
        if handle.upstream_chain.stage.identity != producer.stage_identity {
            return Err(HandleRejection::StageNotBound);
        }
        if handle.consumer != self.consumer {
            return Err(HandleRejection::WrongConsumer {
                expected: self.consumer.clone(),
                found: handle.consumer.clone(),
            });
        }
        if handle.parent_operation_id != request.expected_operation {
            return Err(HandleRejection::WrongOperation {
                expected: request.expected_operation,
                found: handle.parent_operation_id,
            });
        }
        request
            .operations
            .operation_admits(OperationCheck {
                parent_operation_id: handle.parent_operation_id,
                upstream_chain: &handle.upstream_chain,
                payload_type: &handle.payload_type,
                project: request.project,
                consumer_stage: &self.consumer,
                consumer_session: request.consumer_session,
                producer_session: producer.session,
                signed_absolute_deadline_ms: handle.signed_absolute_deadline_ms,
            })
            .map_err(HandleRejection::Operation)?;
        if let Some(slot) = producer.slots.get(&handle.handle_nonce) {
            return Err(match slot.state {
                SlotState::Open { .. } => HandleRejection::DuplicateOpen,
                SlotState::Consumed { .. } => HandleRejection::Replayed,
                SlotState::Abandoned => HandleRejection::Abandoned,
            });
        }
        let deadline_ms = handle.signed_absolute_deadline_ms;
        if deadline_ms <= now_ms {
            return Err(HandleRejection::Expired);
        }
        if deadline_ms
            > now_ms
                .saturating_add(policy.max_handle_lifetime_ms)
                .saturating_add(policy.max_clock_skew_ms)
        {
            return Err(HandleRejection::DeadlineTooFar);
        }
        if self.open_total >= policy.max_open_handles {
            return Err(HandleRejection::TooManyOpenHandles {
                limit: policy.max_open_handles,
            });
        }
        if open_of(producer) >= policy.max_open_per_principal {
            return Err(HandleRejection::TooManyOpenForPrincipal {
                limit: policy.max_open_per_principal,
            });
        }
        if producer.slots.len() >= policy.max_tombstones_per_principal {
            return Err(HandleRejection::TombstoneTableFull {
                limit: policy.max_tombstones_per_principal,
            });
        }
        let footprint = request.footprint_bytes;
        if footprint < handle.byte_len {
            return Err(HandleRejection::FootprintBelowLength {
                footprint,
                byte_len: handle.byte_len,
            });
        }
        let in_flight = claimed_bytes(producer);
        if in_flight.saturating_add(footprint) > policy.max_inflight_bytes_per_principal {
            return Err(HandleRejection::InflightBytesExceeded {
                limit: policy.max_inflight_bytes_per_principal,
                in_flight,
                requested: footprint,
            });
        }
        // The placement is read for the project of the call, never by stage name alone.
        revalidate_chain_in(
            request.project,
            &handle.upstream_chain,
            handle.placement_epoch,
            request.live,
        )
        .map_err(HandleRejection::Chain)?;
        let slot_digest = handle_digest(&handle);
        // The last fallible step: a refusal above or here records nothing, and nothing but the
        // slot insert follows it.
        let host = match &self.quota {
            Some(quota) => Some(
                quota
                    .try_charge(producer.session, request.consumer_session, footprint)
                    .map_err(HandleRejection::HostQuota)?,
            ),
            None => None,
        };
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.open_total += 1;
        producer.slots.insert(
            handle.handle_nonce,
            Slot {
                state: SlotState::Open { ticket },
                deadline_ms,
                digest: slot_digest,
                claim: Some(Claim {
                    carrier: handle.carrier,
                    locator: reclaim_locator(&handle.locator),
                    byte_len: handle.byte_len,
                    charged_bytes: footprint,
                    host,
                }),
                producer_session: producer.session,
                consumer_session: request.consumer_session,
            },
        );
        Ok(AdmittedHandle {
            table: self.id,
            ticket,
            principal: request.principal.to_owned(),
            handle: handle.into_owned(),
            project: request.project.clone(),
            consumer_session: request.consumer_session,
            producer_session: producer.session,
        })
    }

    /// The open slot `admitted` holds the ticket of.
    fn open_slot(&mut self, admitted: &AdmittedHandle) -> Result<&mut Slot, HandleRejection> {
        if admitted.table != self.id {
            return Err(HandleRejection::UnknownTicket);
        }
        let slot = self
            .producers
            .get_mut(&admitted.principal)
            .and_then(|producer| producer.slots.get_mut(&admitted.handle.handle_nonce))
            .ok_or(HandleRejection::UnknownTicket)?;
        match slot.state {
            SlotState::Open { ticket } if ticket == admitted.ticket => Ok(slot),
            SlotState::Open { .. } => Err(HandleRejection::UnknownTicket),
            SlotState::Consumed { .. } => Err(HandleRejection::Replayed),
            SlotState::Abandoned => Err(HandleRejection::Abandoned),
        }
    }

    /// Use an admitted handle at `request.now_ms`: the ticket must still be the open one, the
    /// handle must not have reached its deadline, and the chain and placement epoch are
    /// revalidated against `request.live` now. Only then is the nonce tombstoned and the
    /// locator handed out. The authority is asked again with the same check as at admission, so
    /// every [`OperationFault`] applies at use too: an unknown operation, a dead input, an
    /// operation that is no longer Exact (`NotExact`), a changed expected chain
    /// (`ChainMismatch`) or changed accepted payload types (`PayloadTypeNotAccepted`). Any such
    /// fault, like a stale chain, abandons the handle: no locator is returned, nothing is
    /// released, and the producer reclaims at the deadline plus the skew. A handle at its
    /// deadline stays open for [`Self::expire`] to report.
    pub fn consume(
        &mut self,
        request: ConsumeRequest<'_>,
    ) -> Result<ConsumedHandle, HandleRejection> {
        let now_ms = self.advance(request.now_ms);
        let admitted = request.admitted;
        let slot = self.open_slot(&admitted)?;
        // The consumer stops at the bare deadline; the producer reclaims only after the skew.
        if slot.deadline_ms <= now_ms {
            return Err(HandleRejection::Expired);
        }
        let still_live = revalidate_chain_in(
            &admitted.project,
            &admitted.handle.upstream_chain,
            admitted.handle.placement_epoch,
            request.live,
        )
        .map_err(HandleRejection::Chain)
        .and_then(|()| {
            request
                .operations
                .operation_admits(OperationCheck {
                    parent_operation_id: admitted.handle.parent_operation_id,
                    upstream_chain: &admitted.handle.upstream_chain,
                    payload_type: &admitted.handle.payload_type,
                    project: &admitted.project,
                    // Equal to this table's consumer: admission checked it.
                    consumer_stage: &admitted.handle.consumer,
                    consumer_session: admitted.consumer_session,
                    producer_session: admitted.producer_session,
                    signed_absolute_deadline_ms: admitted.handle.signed_absolute_deadline_ms,
                })
                .map_err(HandleRejection::Operation)
        });
        if let Err(rejection) = still_live {
            slot.state = SlotState::Abandoned;
            self.open_total = self.open_total.saturating_sub(1);
            return Err(rejection);
        }
        // The claim stays: the consumer is using the buffer now. It is released by `finish`
        // (not shared), by `release_after_teardown` (shared) or by `expire`.
        slot.state = SlotState::Consumed {
            ticket: admitted.ticket,
        };
        self.open_total = self.open_total.saturating_sub(1);
        Ok(ConsumedHandle {
            table: self.id,
            ticket: admitted.ticket,
            handle: admitted.handle,
            principal: admitted.principal,
            project: admitted.project,
            consumer_session: admitted.consumer_session,
            producer_session: admitted.producer_session,
        })
    }

    /// The consumer is done with a consumed handle: its output was admitted (see
    /// [`revalidate_operation`]) or given up. For a carrier that is not shared the claim is
    /// released now and the notice returned, once, with [`ReclaimReason::Consumed`]; the driver
    /// delivers it to the producer. A shared carrier returns `None` and stays charged until
    /// [`Self::release_after_teardown`] (doc 42 rule 3b): `Consumed` comes from a worker that
    /// may keep its mapping. `None` also when [`Self::expire`] reported the handle already, or
    /// when the nonce was used again since (a stale handle never releases a later use's claim).
    /// A handle of another table is refused as [`HandleRejection::UnknownTicket`].
    pub fn finish(
        &mut self,
        consumed: ConsumedHandle,
    ) -> Result<Option<ReclaimNotice>, HandleRejection> {
        if consumed.table != self.id {
            return Err(HandleRejection::UnknownTicket);
        }
        let nonce = consumed.handle.handle_nonce;
        let Some(slot) = self
            .producers
            .get_mut(&consumed.principal)
            .and_then(|producer| producer.slots.get_mut(&nonce))
        else {
            return Ok(None);
        };
        if slot.state
            != (SlotState::Consumed {
                ticket: consumed.ticket,
            })
            || slot
                .claim
                .as_ref()
                .is_none_or(|claim| is_shared(claim.carrier))
        {
            return Ok(None);
        }
        Ok(take_notice(
            self.quota.as_ref(),
            &consumed.principal,
            nonce,
            slot,
            ReclaimReason::Consumed,
        ))
    }

    /// The consumer gives an admitted handle up unused. The nonce is tombstoned; the claim and
    /// its bytes are held until [`Self::expire`] reports them after the deadline plus the skew
    /// (a shared claim until [`Self::release_after_teardown`]).
    pub fn abandon(&mut self, admitted: AdmittedHandle) -> Result<(), HandleRejection> {
        self.open_slot(&admitted)?.state = SlotState::Abandoned;
        self.open_total = self.open_total.saturating_sub(1);
        Ok(())
    }

    /// At `now_ms` (clamped monotonic), return every open or abandoned handle whose deadline
    /// **plus the skew** has passed (never the bare deadline: a handle at exactly its deadline is
    /// not yet reclaimable), so the producer reclaims the buffer, and drop every slot
    /// (consumed tombstones included) whose retention has ended, and every removed producer
    /// that holds no slot any more. A handle is reported once. A handle on a shared carrier is
    /// not reported here: an open one is closed, and its slot and bytes stay until
    /// [`Self::release_after_teardown`].
    pub fn expire(&mut self, now_ms: u64) -> Vec<ReclaimNotice> {
        let now_ms = self.advance(now_ms);
        let policy = self.policy;
        let quota = self.quota.as_ref();
        let mut notices = Vec::new();
        let mut closed_open = 0usize;
        for (principal, producer) in &mut self.producers {
            let due: Vec<u64> = producer
                .slots
                .iter()
                .filter(|(_, slot)| is_expired(slot.deadline_ms, &policy, now_ms))
                .map(|(nonce, _)| *nonce)
                .collect();
            for nonce in due {
                // A shared claim outlives the deadline: the slot (and its nonce tombstone)
                // stays, closed, until `release_after_teardown` takes the claim.
                if let Some(slot) = producer.slots.get_mut(&nonce) {
                    if slot
                        .claim
                        .as_ref()
                        .is_some_and(|claim| is_shared(claim.carrier))
                    {
                        if matches!(slot.state, SlotState::Open { .. }) {
                            slot.state = SlotState::Abandoned;
                            closed_open += 1;
                        }
                        continue;
                    }
                }
                let Some(mut slot) = producer.slots.remove(&nonce) else {
                    continue;
                };
                let reason = match slot.state {
                    SlotState::Open { .. } => {
                        closed_open += 1;
                        ReclaimReason::Expired
                    }
                    SlotState::Abandoned => ReclaimReason::Abandoned,
                    SlotState::Consumed { .. } => ReclaimReason::Consumed,
                };
                notices.extend(take_notice(quota, principal, nonce, &mut slot, reason));
            }
        }
        self.open_total = self.open_total.saturating_sub(closed_open);
        self.producers
            .retain(|_, producer| producer.active || !producer.slots.is_empty());
        notices
    }

    /// Release shared buffers after **verified teardown** (doc 42 rule 3b): the supervisor
    /// calls this only once it has terminated and reaped both worker domains named by
    /// `teardown` and drained the old control socket of each. `Consumed`, the deadline and a
    /// producer replacement never release a shared buffer; this does, once, with
    /// [`ReclaimReason::TornDown`], and only for the handles admitted under **exactly** that
    /// producer session and consumer session: a handle of a respawned pair, admitted after the
    /// reap under new sessions, is not touched even if the call arrives late. Open handles of
    /// that pair are closed first (their workers are gone). Nonce tombstones stay until
    /// [`Self::expire`] drops them at the deadline plus the skew. Claims of carriers that are
    /// not shared are left to the other paths.
    #[must_use = "the returned notices name buffers the producer may reclaim"]
    pub fn release_after_teardown(&mut self, teardown: Teardown<'_>) -> Vec<ReclaimNotice> {
        let principal = teardown.principal;
        let Some(producer) = self.producers.get_mut(principal) else {
            return Vec::new();
        };
        let quota = self.quota.as_ref();
        let mut notices = Vec::new();
        let mut closed = 0usize;
        for (nonce, slot) in &mut producer.slots {
            if slot.producer_session != teardown.producer_session
                || slot.consumer_session != teardown.consumer_session
                || !slot
                    .claim
                    .as_ref()
                    .is_some_and(|claim| is_shared(claim.carrier))
            {
                continue;
            }
            if matches!(slot.state, SlotState::Open { .. }) {
                slot.state = SlotState::Abandoned;
                closed += 1;
            }
            notices.extend(take_notice(
                quota,
                principal,
                *nonce,
                slot,
                ReclaimReason::TornDown,
            ));
        }
        self.open_total = self.open_total.saturating_sub(closed);
        notices
    }
}

/// Abandon the open handles of `producer` and take the claims of those whose carrier is not
/// shared: one notice per such handle. A shared claim stays until verified teardown. Returns the
/// notices and the number of handles closed.
fn abandon_open(
    producer: &mut Producer,
    principal: &str,
    reason: ReclaimReason,
    quota: Option<&SharedHostQuota>,
) -> (Vec<ReclaimNotice>, usize) {
    let mut notices = Vec::new();
    let mut closed = 0usize;
    for (nonce, slot) in &mut producer.slots {
        if matches!(slot.state, SlotState::Open { .. }) {
            slot.state = SlotState::Abandoned;
            closed += 1;
            if slot
                .claim
                .as_ref()
                .is_some_and(|claim| !is_shared(claim.carrier))
            {
                notices.extend(take_notice(quota, principal, *nonce, slot, reason));
            }
        }
    }
    (notices, closed)
}

/// The carriers whose buffer another process may still reference after `Consumed`, released
/// only on verified teardown (doc 42 rule 3b).
fn is_shared(carrier: Carrier) -> bool {
    match carrier {
        Carrier::HostShm | Carrier::CudaIpc | Carrier::DevicePeer | Carrier::PinnedHost => true,
        Carrier::InProcess | Carrier::Bytes | Carrier::NvmeFile | Carrier::Remote => false,
    }
}

/// The producer's handles in the `Open` state (a scan of its own, bounded slots).
fn open_of(producer: &Producer) -> usize {
    producer
        .slots
        .values()
        .filter(|slot| matches!(slot.state, SlotState::Open { .. }))
        .count()
}

/// The charged footprints of the producer's claims that have not been reported to it yet.
fn claimed_bytes(producer: &Producer) -> u64 {
    producer
        .slots
        .values()
        .filter_map(|slot| slot.claim.as_ref())
        .fold(0u64, |sum, claim| sum.saturating_add(claim.charged_bytes))
}

/// Retention ends, and a buffer is reclaimable, at the deadline plus the skew.
fn is_expired(deadline_ms: u64, policy: &HandlePolicy, now_ms: u64) -> bool {
    deadline_ms.saturating_add(policy.max_clock_skew_ms) <= now_ms
}

/// The locator as the table reports it: a [`CarrierLocator::Bytes`] payload is not kept.
fn reclaim_locator(locator: &CarrierLocator) -> CarrierLocator {
    match locator {
        CarrierLocator::Bytes { .. } => CarrierLocator::Bytes { inline: Vec::new() },
        other @ (CarrierLocator::HostShm { .. }
        | CarrierLocator::CudaIpc { .. }
        | CarrierLocator::DevicePeer { .. }
        | CarrierLocator::PinnedHost { .. }
        | CarrierLocator::NvmeFile { .. }
        | CarrierLocator::Remote { .. }) => other.clone(),
    }
}

/// Take the claim of `slot`, hand its host charge back and report the buffer. Every path that
/// frees a claim goes through here, so a charge is released exactly once (the claim is gone
/// afterwards) and never without the notice that frees the buffer. A release the ledger refuses
/// is counted there ([`crate::HostQuotaUsage::release_faults`]) and changes nothing.
fn take_notice(
    quota: Option<&SharedHostQuota>,
    principal: &str,
    nonce: u64,
    slot: &mut Slot,
    reason: ReclaimReason,
) -> Option<ReclaimNotice> {
    let mut claim = slot.claim.take()?;
    if let (Some(quota), Some(charge)) = (quota, claim.host.take()) {
        // A refused release is already counted in the ledger; there is nothing to undo here.
        let _ = quota.release(charge);
    }
    Some(notice(principal, nonce, slot.digest, claim, reason))
}

fn notice(
    principal: &str,
    handle_nonce: u64,
    handle_digest: [u8; 32],
    claim: Claim,
    reason: ReclaimReason,
) -> ReclaimNotice {
    ReclaimNotice {
        principal: principal.to_owned(),
        handle_nonce,
        handle_digest,
        carrier: claim.carrier,
        locator: claim.locator,
        byte_len: claim.byte_len,
        charged_bytes: claim.charged_bytes,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use ptr_protocol::activation::ChainElement;
    use ptr_protocol::{content_digest, MAX_HANDLE_BYTES};
    use ptr_types::{Generation, Validity};
    use std::cell::RefCell;

    const PRINCIPAL: &str = "producer";
    const CONSUMER: &str = "down";
    const OP: u64 = 7;
    const PAYLOAD: &str = "ptr.split.activation.v1";
    const CONSUMER_SESSION: WorkerSession = WorkerSession(11);
    const PRODUCER_SESSION: WorkerSession = WorkerSession(22);

    fn project() -> ProjectId {
        ProjectId::from("p")
    }

    #[derive(Default)]
    struct Live {
        standing: RefCell<BTreeMap<(String, u64), Validity>>,
        digests: RefCell<BTreeMap<(String, u64), [u8; 32]>>,
        placement: RefCell<BTreeMap<String, u64>>,
    }

    impl Live {
        fn commit(&self, identity: &str, generation: u64) {
            self.standing
                .borrow_mut()
                .insert((identity.to_owned(), generation), Validity::Live);
            self.digests
                .borrow_mut()
                .insert((identity.to_owned(), generation), [generation as u8; 32]);
        }
    }

    impl StageLiveness for Live {
        fn validity(&self, target: &str, generation: Generation) -> Option<Validity> {
            self.standing
                .borrow()
                .get(&(target.to_owned(), generation.0))
                .copied()
        }

        fn manifest_digest(&self, target: &str, generation: Generation) -> Option<[u8; 32]> {
            self.digests
                .borrow()
                .get(&(target.to_owned(), generation.0))
                .copied()
        }

        fn placement_epoch(&self, stage: &str) -> Option<u64> {
            self.placement.borrow().get(stage).copied()
        }

        /// The fake places stages for the test project only, so a project mix-up is refused.
        fn placement_epoch_in(&self, project: &ProjectId, stage: &str) -> Option<u64> {
            if *project != self::project() {
                return None;
            }
            self.placement_epoch(stage)
        }

        fn fence_epoch(&self, _expert: &str) -> Option<u64> {
            None
        }
    }

    fn live() -> Live {
        let live = Live::default();
        live.commit("stage", 1);
        live.commit("model", 1);
        live.placement.borrow_mut().insert("stage".to_owned(), 5);
        live
    }

    /// Every operation is known and all its inputs are live.
    struct AllLive;

    impl OperationLiveness for AllLive {
        fn operation_admits(&self, _check: OperationCheck<'_>) -> Result<(), OperationFault> {
            Ok(())
        }
    }

    /// The authority's operation record, faked: operation to the inputs it read; an input in
    /// `dead` is no longer live at the committed anchor.
    #[derive(Default)]
    struct Operations {
        inputs: RefCell<BTreeMap<u64, Vec<String>>>,
        dead: RefCell<Vec<String>>,
        expected: RefCell<BTreeMap<u64, (UpstreamChain, Vec<TypeId>)>>,
        /// Operations recorded as Approximate or tainted.
        inexact: RefCell<Vec<u64>>,
        /// The project, consumer session and producer session each operation is bound to.
        bound: RefCell<BTreeMap<u64, (ProjectId, WorkerSession, WorkerSession)>>,
        /// The last check the authority was asked, for what the table passes on.
        last: RefCell<Option<(ProjectId, WorkerSession, WorkerSession)>>,
        /// The destination stage each operation recorded.
        destination: RefCell<BTreeMap<u64, String>>,
    }

    impl OperationLiveness for Operations {
        fn operation_admits(&self, check: OperationCheck<'_>) -> Result<(), OperationFault> {
            let operation = check.parent_operation_id;
            *self.last.borrow_mut() = Some((
                check.project.clone(),
                check.consumer_session,
                check.producer_session,
            ));
            let inputs = self.inputs.borrow();
            let read = inputs
                .get(&operation)
                .ok_or(OperationFault::UnknownOperation { operation })?;
            if let Some((project, consumer, producer)) = self.bound.borrow().get(&operation) {
                if project != check.project {
                    return Err(OperationFault::WrongProject { operation });
                }
                if let Some(stage) = self.destination.borrow().get(&operation) {
                    if stage != check.consumer_stage {
                        return Err(OperationFault::ConsumerStageMismatch { operation });
                    }
                }
                if *consumer != check.consumer_session {
                    return Err(OperationFault::ConsumerSessionMismatch { operation });
                }
                if *producer != check.producer_session {
                    return Err(OperationFault::ProducerSessionMismatch { operation });
                }
            }
            if self.inexact.borrow().contains(&operation) {
                return Err(OperationFault::NotExact { operation });
            }
            let dead = self.dead.borrow();
            if let Some(input) = read.iter().find(|input| dead.contains(input)) {
                return Err(OperationFault::InputNotLive {
                    operation,
                    input: input.clone(),
                });
            }
            if let Some((chain, accepted)) = self.expected.borrow().get(&operation) {
                if chain != check.upstream_chain {
                    return Err(OperationFault::ChainMismatch { operation });
                }
                if !accepted.contains(check.payload_type) {
                    return Err(OperationFault::PayloadTypeNotAccepted {
                        operation,
                        payload_type: check.payload_type.clone(),
                    });
                }
            }
            Ok(())
        }
    }

    impl Operations {
        /// The authority's record of `operation`: the inputs it read, the chain it expects and the
        /// payload types it accepts.
        fn know(&self, operation: u64, inputs: &[&str], chain: UpstreamChain, accepted: &[&str]) {
            self.bound
                .borrow_mut()
                .insert(operation, (project(), CONSUMER_SESSION, PRODUCER_SESSION));
            self.destination
                .borrow_mut()
                .insert(operation, CONSUMER.to_owned());
            self.inputs
                .borrow_mut()
                .insert(operation, inputs.iter().map(|i| (*i).to_owned()).collect());
            self.expected.borrow_mut().insert(
                operation,
                (chain, accepted.iter().map(|t| TypeId::from(*t)).collect()),
            );
        }
    }

    fn signer() -> SigningKey {
        SigningKey::from_bytes(&[0xA1; 32])
    }

    fn policy() -> HandlePolicy {
        HandlePolicy {
            max_open_handles: 4,
            max_open_per_principal: 4,
            max_tombstones_per_principal: 8,
            max_inflight_bytes_per_principal: 64,
            max_handle_lifetime_ms: 1_000,
            max_clock_skew_ms: 100,
        }
    }

    fn table() -> HandleTable {
        let mut table = HandleTable::new(policy(), CONSUMER).expect("a valid policy");
        let replaced = table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                PRODUCER_SESSION,
            )
            .expect("registered");
        assert!(replaced.is_empty());
        table
    }

    fn element(name: &str) -> ChainElement {
        ChainElement {
            identity: name.to_owned(),
            generation: 1,
            digest: [1; 32],
            fence_epoch: 0,
        }
    }

    /// A handle on a carrier that is not shared (`NvmeFile`): released by `finish`, on
    /// producer replacement and at the deadline plus the skew.
    fn signed(nonce: u64, deadline_ms: u64) -> TensorHandle {
        let mut handle = TensorHandle {
            carrier: Carrier::NvmeFile,
            locator: CarrierLocator::NvmeFile {
                chunk_digest: [nonce as u8; 32],
            },
            dtype: BoundaryDtype::F32,
            shape: vec![2, 2],
            byte_len: 16,
            upstream_chain: UpstreamChain {
                stage: element("stage"),
                base_model: element("model"),
                adapters: vec![],
                experts: vec![],
            },
            placement_epoch: 5,
            signed_absolute_deadline_ms: deadline_ms,
            handle_nonce: nonce,
            parent_operation_id: OP,
            consumer: CONSUMER.to_owned(),
            payload_type: TypeId::from(PAYLOAD),
            content_digest: content_digest(&[nonce as u8; 16]),
            signature: [0; 64],
        };
        resign(&mut handle);
        handle
    }

    /// The teardown of the producer and consumer workers every test admits under.
    fn torn_down() -> Teardown<'static> {
        Teardown {
            principal: PRINCIPAL,
            producer_session: PRODUCER_SESSION,
            consumer_session: CONSUMER_SESSION,
        }
    }

    /// The same handle on a shared carrier (`HostShm`): released only after verified teardown.
    fn shared(nonce: u64, deadline_ms: u64) -> TensorHandle {
        let mut handle = signed(nonce, deadline_ms);
        handle.carrier = Carrier::HostShm;
        handle.locator = CarrierLocator::HostShm {
            fd_slot: nonce as u32,
            offset: 0,
        };
        resign(&mut handle);
        handle
    }

    fn resign(handle: &mut TensorHandle) {
        handle.signature = signer().sign(&handle_signing_bytes(handle)).to_bytes();
    }

    fn admit(
        table: &mut HandleTable,
        live: &Live,
        now_ms: u64,
        handle: &TensorHandle,
    ) -> Result<AdmittedHandle, HandleRejection> {
        table.admit(HandleAdmissionRequest {
            now_ms,
            principal: PRINCIPAL,
            expected_operation: OP,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &AllLive,
            handle: HandleSource::Decoded(handle),
            footprint_bytes: handle.byte_len,
            live,
        })
    }

    fn consume(
        table: &mut HandleTable,
        live: &Live,
        now_ms: u64,
        admitted: AdmittedHandle,
    ) -> Result<ConsumedHandle, HandleRejection> {
        table.consume(ConsumeRequest {
            now_ms,
            admitted,
            live,
            operations: &AllLive,
        })
    }

    #[test]
    fn a_zero_bound_is_a_named_policy_violation() {
        let zero = |change: fn(&mut HandlePolicy)| {
            let mut policy = policy();
            change(&mut policy);
            policy.validate()
        };
        assert_eq!(policy().validate(), Ok(()));
        assert_eq!(
            zero(|p| p.max_open_handles = 0),
            Err(HandlePolicyViolation::ZeroOpenHandles)
        );
        assert_eq!(
            zero(|p| p.max_tombstones_per_principal = 0),
            Err(HandlePolicyViolation::ZeroTombstones)
        );
        assert_eq!(
            zero(|p| p.max_open_per_principal = 0),
            Err(HandlePolicyViolation::ZeroOpenPerPrincipal)
        );
        assert_eq!(
            zero(|p| p.max_open_per_principal = p.max_open_handles + 1),
            Err(HandlePolicyViolation::OpenPerPrincipalAboveTotal)
        );
        assert!(matches!(
            HandleTable::new(policy(), ""),
            Err(HandlePolicyViolation::EmptyConsumer)
        ));
        assert_eq!(
            zero(|p| p.max_inflight_bytes_per_principal = 0),
            Err(HandlePolicyViolation::ZeroInflightBytes)
        );
        assert_eq!(
            zero(|p| p.max_handle_lifetime_ms = 0),
            Err(HandlePolicyViolation::ZeroLifetime)
        );
        assert_eq!(
            zero(|p| p.max_clock_skew_ms = 0),
            Err(HandlePolicyViolation::ZeroClockSkew)
        );
        let refused = HandlePolicy {
            max_open_handles: 0,
            ..policy()
        };
        assert!(matches!(
            HandleTable::new(refused, CONSUMER),
            Err(HandlePolicyViolation::ZeroOpenHandles)
        ));
    }

    #[test]
    fn an_encoded_and_a_decoded_handle_are_admitted_alike() {
        let live = live();
        let mut table = table();
        let first = signed(1, 500);
        let bytes = signed(2, 500).encode().expect("encodes");
        assert!(admit(&mut table, &live, 10, &first).is_ok());
        let admitted = table
            .admit(HandleAdmissionRequest {
                now_ms: 10,
                principal: PRINCIPAL,
                expected_operation: OP,
                project: &project(),
                consumer_session: CONSUMER_SESSION,
                operations: &AllLive,
                handle: HandleSource::Encoded(&bytes),
                footprint_bytes: 16,
                live: &live,
            })
            .expect("encoded handle admitted");
        assert_eq!(admitted.handle_nonce(), 2);
        assert_eq!(admitted.shape(), &[2, 2]);
        assert_eq!(table.open_handles(), 2);
    }

    #[test]
    fn an_unregistered_principal_is_refused_before_anything_is_decoded() {
        let live = live();
        let mut table = table();
        let result = table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: "stranger",
            expected_operation: OP,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &AllLive,
            handle: HandleSource::Encoded(&[0xFF; 3]),
            footprint_bytes: 16,
            live: &live,
        });
        assert_eq!(result.unwrap_err(), HandleRejection::UnknownPrincipal);
    }

    #[test]
    fn malformed_bytes_and_an_ill_formed_handle_are_refused() {
        let live = live();
        let mut table = table();
        let result = table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: PRINCIPAL,
            expected_operation: OP,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &AllLive,
            handle: HandleSource::Encoded(&[1, 2, 3, 4, 5, 6, 7, 8]),
            footprint_bytes: 16,
            live: &live,
        });
        assert_eq!(
            result.unwrap_err(),
            HandleRejection::Malformed(HandleError::BadMagic)
        );
        let mut oversize = signed(1, 500);
        oversize.byte_len = MAX_HANDLE_BYTES + 4;
        resign(&mut oversize);
        assert!(matches!(
            admit(&mut table, &live, 10, &oversize),
            Err(HandleRejection::Malformed(HandleError::OverBound(
                "byte_len"
            )))
        ));
        let mut in_process = signed(2, 500);
        in_process.carrier = Carrier::InProcess;
        resign(&mut in_process);
        assert_eq!(
            admit(&mut table, &live, 10, &in_process).unwrap_err(),
            HandleRejection::Malformed(HandleError::InProcessNotEncodable)
        );
    }

    #[test]
    fn a_signature_of_another_key_or_over_other_fields_is_refused() {
        let live = live();
        let mut table = table();
        let mut forged = signed(1, 500);
        forged.signature = SigningKey::from_bytes(&[0xB2; 32])
            .sign(&handle_signing_bytes(&forged))
            .to_bytes();
        assert_eq!(
            admit(&mut table, &live, 10, &forged).unwrap_err(),
            HandleRejection::BadSignature
        );
        let mut moved = signed(2, 500);
        moved.signed_absolute_deadline_ms += 1;
        assert_eq!(
            admit(&mut table, &live, 10, &moved).unwrap_err(),
            HandleRejection::BadSignature
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_chain_of_another_stage_than_the_bound_one_is_refused() {
        let live = live();
        live.commit("other", 1);
        live.placement.borrow_mut().insert("other".to_owned(), 5);
        let mut table = table();
        let mut handle = signed(1, 500);
        handle.upstream_chain.stage = element("other");
        resign(&mut handle);
        assert_eq!(
            admit(&mut table, &live, 10, &handle).unwrap_err(),
            HandleRejection::StageNotBound
        );
    }

    #[test]
    fn a_nonce_in_use_or_used_is_refused_as_duplicate_replayed_or_abandoned() {
        let live = live();
        let mut table = table();
        let handle = signed(1, 500);
        let admitted = admit(&mut table, &live, 10, &handle).expect("admitted");
        assert_eq!(
            admit(&mut table, &live, 10, &handle).unwrap_err(),
            HandleRejection::DuplicateOpen
        );
        let _ = consume(&mut table, &live, 20, admitted).expect("consumed");
        assert_eq!(
            admit(&mut table, &live, 30, &handle).unwrap_err(),
            HandleRejection::Replayed
        );
        let other = signed(2, 500);
        let admitted = admit(&mut table, &live, 30, &other).expect("admitted");
        table.abandon(admitted).expect("abandoned");
        assert_eq!(
            admit(&mut table, &live, 40, &other).unwrap_err(),
            HandleRejection::Abandoned
        );
    }

    #[test]
    fn a_deadline_not_after_now_is_expired_at_admission() {
        let live = live();
        let mut table = table();
        // The consumer stops at the bare deadline 500: alive at 499, dead at 500. Only the
        // retention and the reclaim wait for deadline plus skew.
        let handle = signed(1, 500);
        assert!(admit(&mut table, &live, 499, &handle).is_ok());
        let late = signed(2, 500);
        assert_eq!(
            admit(&mut table, &live, 500, &late).unwrap_err(),
            HandleRejection::Expired
        );
    }

    #[test]
    fn a_deadline_near_u64_max_is_too_far_without_overflow() {
        let live = live();
        let mut table = table();
        for deadline in [u64::MAX, u64::MAX - 1, u64::MAX / 2] {
            assert_eq!(
                admit(&mut table, &live, 10, &signed(1, deadline)).unwrap_err(),
                HandleRejection::DeadlineTooFar
            );
        }
        // A clock near the maximum does not wrap either: the sum saturates.
        assert!(table.expire(u64::MAX - 5).is_empty());
        assert_eq!(
            admit(&mut table, &live, 0, &signed(2, 1_000)).unwrap_err(),
            HandleRejection::Expired
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_deadline_beyond_lifetime_plus_skew_is_too_far() {
        let live = live();
        let mut table = table();
        // now 0 + lifetime 1000 + skew 100.
        assert!(admit(&mut table, &live, 0, &signed(1, 1_100)).is_ok());
        assert_eq!(
            admit(&mut table, &live, 0, &signed(2, 1_101)).unwrap_err(),
            HandleRejection::DeadlineTooFar
        );
    }

    #[test]
    fn the_clock_never_runs_backwards() {
        let live = live();
        let mut table = table();
        // A late call moves the clock to 700; an earlier timestamp cannot revive a dead handle.
        assert!(table.expire(700).is_empty());
        assert_eq!(
            admit(&mut table, &live, 0, &signed(1, 500)).unwrap_err(),
            HandleRejection::Expired
        );
    }

    #[test]
    fn open_handles_are_bounded_table_wide_and_tombstones_per_principal() {
        let live = live();
        let mut table = table();
        for nonce in 1..=4 {
            admit(&mut table, &live, 10, &signed(nonce, 500)).expect("room");
        }
        assert_eq!(
            admit(&mut table, &live, 10, &signed(5, 500)).unwrap_err(),
            HandleRejection::TooManyOpenHandles { limit: 4 }
        );
        assert_eq!(table.open_handles(), 4);

        // Tombstones share the producer's table: 4 consumed + 4 open fills the 8 slots.
        let mut table = HandleTable::new(
            HandlePolicy {
                max_open_handles: 8,
                max_tombstones_per_principal: 3,
                ..policy()
            },
            CONSUMER,
        )
        .expect("policy");
        let _ = table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                PRODUCER_SESSION,
            )
            .expect("registered");
        for nonce in 1..=3 {
            let admitted = admit(&mut table, &live, 10, &signed(nonce, 500)).expect("room");
            let consumed = consume(&mut table, &live, 11, admitted).expect("consumed");
            assert!(table.finish(consumed).expect("this table").is_some());
        }
        assert_eq!(table.open_handles(), 0);
        assert_eq!(
            admit(&mut table, &live, 12, &signed(4, 500)).unwrap_err(),
            HandleRejection::TombstoneTableFull { limit: 3 }
        );
        // Retention ends at deadline + skew: the tombstones go and the slots are free again.
        assert!(table.expire(600).is_empty());
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_stale_placement_or_revoked_generation_is_refused_at_admission() {
        let live = live();
        let mut table = table();
        let mut stale = signed(1, 500);
        stale.placement_epoch = 4;
        resign(&mut stale);
        assert!(matches!(
            admit(&mut table, &live, 10, &stale).unwrap_err(),
            HandleRejection::Chain(ChainFault::StalePlacement {
                recorded: 4,
                current: Some(5)
            })
        ));
        live.standing
            .borrow_mut()
            .insert(("model".to_owned(), 1), Validity::Revoked);
        assert!(matches!(
            admit(&mut table, &live, 10, &signed(2, 500)).unwrap_err(),
            HandleRejection::Chain(ChainFault::NotLive { .. })
        ));
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn consume_revalidates_after_the_epoch_moved_and_abandons_the_handle() {
        let live = live();
        let mut table = table();
        let handle = signed(1, 500);
        let admitted = admit(&mut table, &live, 10, &handle).expect("admitted");
        live.placement.borrow_mut().insert("stage".to_owned(), 6);
        assert!(matches!(
            consume(&mut table, &live, 20, admitted).unwrap_err(),
            HandleRejection::Chain(ChainFault::StalePlacement { .. })
        ));
        assert_eq!(table.open_handles(), 0);
        assert_eq!(
            admit(&mut table, &live, 30, &handle).unwrap_err(),
            HandleRejection::Abandoned
        );
        // The producer reclaims the buffer once, at the deadline plus the skew.
        assert!(table.expire(599).is_empty());
        let notices = table.expire(600);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::Abandoned);
        assert!(table.expire(700).is_empty());
    }

    #[test]
    fn consume_returns_the_locator_and_the_reclaim_notice_once() {
        let live = live();
        let mut table = table();
        let handle = signed(7, 500);
        let admitted = admit(&mut table, &live, 10, &handle).expect("admitted");
        assert_eq!(admitted.handle_digest(), handle_digest(&handle));
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        assert_eq!(consumed.locator(), &handle.locator);
        assert_eq!(consumed.handle(), &handle);
        // `consume` releases nothing: the bytes stay charged while the consumer uses them.
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        let notice = table
            .finish(consumed)
            .expect("a ticket of this table")
            .expect("a non-shared carrier is released at finish");
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
        assert_eq!(notice.reason, ReclaimReason::Consumed);
        assert_eq!(notice.handle_nonce, 7);
        assert_eq!(notice.byte_len, 16);
        assert_eq!(notice.carrier, Carrier::NvmeFile);
        // Nothing is left for the deadline to report.
        assert!(table.expire(1_000).is_empty());
    }

    #[test]
    fn consume_after_the_deadline_is_expired_and_expire_reports_it_once() {
        let live = live();
        let mut table = table();
        let admitted = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        assert_eq!(
            consume(&mut table, &live, 600, admitted).unwrap_err(),
            HandleRejection::Expired
        );
        let notices = table.expire(600);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::Expired);
        assert_eq!(notices[0].handle_nonce, 1);
        assert!(table.expire(601).is_empty());
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_ticket_of_another_table_or_a_swept_slot_is_unknown() {
        let live = live();
        let mut one = table();
        let mut two = table();
        let admitted = admit(&mut one, &live, 10, &signed(1, 500)).expect("admitted");
        let foreign = admit(&mut two, &live, 10, &signed(1, 500)).expect("admitted");
        assert_eq!(
            consume(&mut two, &live, 20, admitted).unwrap_err(),
            HandleRejection::UnknownTicket
        );
        assert_eq!(
            two.abandon(AdmittedHandle {
                table: foreign.table + 1,
                ..foreign
            })
            .unwrap_err(),
            HandleRejection::UnknownTicket
        );
        let swept = admit(&mut one, &live, 30, &signed(2, 500)).expect("admitted");
        let _ = one.expire(700);
        assert_eq!(
            consume(&mut one, &live, 700, swept).unwrap_err(),
            HandleRejection::UnknownTicket
        );
    }

    #[test]
    fn registering_another_key_or_stage_abandons_the_open_handles_and_reports_them() {
        let live = live();
        let mut table = table();
        let kept = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        let used = admit(&mut table, &live, 10, &signed(2, 500)).expect("admitted");
        let used = consume(&mut table, &live, 11, used).expect("consumed");
        assert!(table.finish(used).expect("this table").is_some());

        // The same key and stage again (a reconnect) changes nothing.
        assert!(table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                PRODUCER_SESSION
            )
            .expect("registered")
            .is_empty());
        assert_eq!(table.open_handles(), 1);

        let other = SigningKey::from_bytes(&[0xB2; 32]).verifying_key();
        // Another key or stage under the same session is refused and changes nothing.
        assert_eq!(
            table.register_producer(PRINCIPAL, other, "stage", PRODUCER_SESSION),
            Err(RegistrationRefused::SessionNotRenewed)
        );
        assert_eq!(
            table.register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "other-stage",
                PRODUCER_SESSION
            ),
            Err(RegistrationRefused::SessionNotRenewed)
        );
        assert_eq!(table.open_handles(), 1);
        // A respawned producer comes with a new session.
        let respawned = WorkerSession(PRODUCER_SESSION.0 + 1);
        let notices = table
            .register_producer(PRINCIPAL, other, "stage", respawned)
            .expect("registered");
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].handle_nonce, 1);
        assert_eq!(notices[0].reason, ReclaimReason::ProducerReplaced);
        assert_eq!(table.open_handles(), 0);
        assert_eq!(
            consume(&mut table, &live, 20, kept).unwrap_err(),
            HandleRejection::Abandoned
        );
        // The old key's signature no longer admits.
        assert_eq!(
            admit(&mut table, &live, 20, &signed(3, 500)).unwrap_err(),
            HandleRejection::BadSignature
        );
        // Under the original key again, the used nonce is still a replay and the abandoned
        // one is still abandoned: the tombstones outlive the key change.
        let _ = table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                WorkerSession(PRODUCER_SESSION.0 + 2),
            )
            .expect("registered");
        assert_eq!(
            admit(&mut table, &live, 21, &signed(2, 500)).unwrap_err(),
            HandleRejection::Replayed
        );
        assert_eq!(
            admit(&mut table, &live, 21, &signed(1, 500)).unwrap_err(),
            HandleRejection::Abandoned
        );
        // Reported once: expiry has nothing left to say.
        assert!(table.expire(1_000).is_empty());
    }

    fn wide_table() -> HandleTable {
        let mut table = HandleTable::new(
            HandlePolicy {
                max_open_handles: 16,
                max_open_per_principal: 16,
                max_tombstones_per_principal: 32,
                ..policy()
            },
            CONSUMER,
        )
        .expect("policy");
        let _ = table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                PRODUCER_SESSION,
            )
            .expect("registered");
        table
    }

    #[test]
    fn inflight_bytes_are_budgeted_per_principal_and_released_only_with_the_reclaim_notice() {
        let live = live();
        // Each test handle is 16 bytes; the budget is 64 (four handles).
        let mut table = wide_table();
        let mut held = Vec::new();
        for nonce in 1..=4 {
            held.push(admit(&mut table, &live, 10, &signed(nonce, 500)).expect("budget"));
        }
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(64));
        assert_eq!(
            admit(&mut table, &live, 10, &signed(5, 500)).unwrap_err(),
            HandleRejection::InflightBytesExceeded {
                limit: 64,
                in_flight: 64,
                requested: 16
            }
        );
        // Consume releases nothing while the bytes are in use; finish emits the notice and
        // releases 16 bytes; abandon releases nothing.
        let first = held.remove(0);
        let first = consume(&mut table, &live, 20, first).expect("consumed");
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(64));
        assert!(table.finish(first).expect("this table").is_some());
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(48));
        let second = held.remove(0);
        table.abandon(second).expect("abandoned");
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(48));
        assert!(admit(&mut table, &live, 20, &signed(5, 500)).is_ok());
        assert!(matches!(
            admit(&mut table, &live, 20, &signed(6, 500)),
            Err(HandleRejection::InflightBytesExceeded { .. })
        ));
        // Expiry at deadline plus skew releases the rest.
        let _ = table.expire(600);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }

    #[test]
    fn an_abandoned_handle_keeps_its_bytes_until_expiry_then_the_budget_is_free() {
        let live = live();
        let mut table = wide_table();
        let mut held = Vec::new();
        for nonce in 1..=4 {
            held.push(admit(&mut table, &live, 10, &signed(nonce, 500)).expect("budget"));
        }
        for admitted in held {
            table.abandon(admitted).expect("abandoned");
        }
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(64));
        // Nothing is open any more, but the claims still fill the budget.
        assert!(matches!(
            admit(&mut table, &live, 20, &signed(5, 900)),
            Err(HandleRejection::InflightBytesExceeded { in_flight: 64, .. })
        ));
        assert!(table.expire(599).is_empty());
        assert_eq!(table.expire(600).len(), 4);
        assert!(admit(&mut table, &live, 600, &signed(5, 1_000)).is_ok());
    }

    #[test]
    fn a_consume_refused_by_a_chain_fault_releases_no_bytes() {
        let live = live();
        let mut table = wide_table();
        let admitted = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        live.placement.borrow_mut().insert("stage".to_owned(), 6);
        assert!(consume(&mut table, &live, 20, admitted).is_err());
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        assert_eq!(table.expire(600).len(), 1);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_rejected_admit_leaves_slots_budget_and_tombstones_unchanged() {
        let live = live();
        let mut table = wide_table();
        let admitted = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        let _ = consume(&mut table, &live, 11, admitted).expect("consumed");
        let _open = admit(&mut table, &live, 12, &signed(2, 500)).expect("admitted");
        let before = (
            table.slots_held(PRINCIPAL),
            table.inflight_bytes(PRINCIPAL),
            table.open_handles(),
        );
        let mut stale = signed(3, 500);
        stale.placement_epoch = 4;
        resign(&mut stale);
        let mut forged = signed(4, 500);
        forged.signature[0] ^= 1;
        let mut elsewhere = signed(5, 500);
        elsewhere.consumer = "other".to_owned();
        resign(&mut elsewhere);
        let mut expired = signed(6, 10);
        resign(&mut expired);
        for refused in [
            &stale,
            &forged,
            &elsewhere,
            &expired,
            &signed(1, 500),
            &signed(2, 500),
        ] {
            assert!(admit(&mut table, &live, 13, refused).is_err());
        }
        assert_eq!(
            before,
            (
                table.slots_held(PRINCIPAL),
                table.inflight_bytes(PRINCIPAL),
                table.open_handles()
            )
        );
    }

    #[test]
    fn a_bad_signature_is_reported_for_a_replayed_expired_or_open_nonce_alike() {
        let live = live();
        let mut table = table();
        let admitted = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        let _ = consume(&mut table, &live, 11, admitted).expect("consumed");
        let _open = admit(&mut table, &live, 12, &signed(2, 500)).expect("admitted");
        for (nonce, deadline) in [(1, 500), (2, 500), (3, 5)] {
            let mut forged = signed(nonce, deadline);
            forged.signature[1] ^= 1;
            assert_eq!(
                admit(&mut table, &live, 13, &forged).unwrap_err(),
                HandleRejection::BadSignature,
                "nonce {nonce}"
            );
        }
    }

    #[test]
    fn a_handle_for_another_consumer_or_operation_is_refused_after_the_signature() {
        let live = live();
        let mut table = table();
        let mut elsewhere = signed(1, 500);
        elsewhere.consumer = "somewhere-else".to_owned();
        resign(&mut elsewhere);
        assert_eq!(
            admit(&mut table, &live, 10, &elsewhere).unwrap_err(),
            HandleRejection::WrongConsumer {
                expected: CONSUMER.to_owned(),
                found: "somewhere-else".to_owned()
            }
        );
        // Unsigned retargeting is a signature failure, not an audience failure.
        let mut retargeted = signed(2, 500);
        retargeted.consumer = "somewhere-else".to_owned();
        assert_eq!(
            admit(&mut table, &live, 10, &retargeted).unwrap_err(),
            HandleRejection::BadSignature
        );
        // The same, correctly signed handle under another operation of the call.
        let handle = signed(3, 500);
        let result = table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: PRINCIPAL,
            expected_operation: OP + 1,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &AllLive,
            handle: HandleSource::Decoded(&handle),
            footprint_bytes: handle.byte_len,
            live: &live,
        });
        assert_eq!(
            result.unwrap_err(),
            HandleRejection::WrongOperation {
                expected: OP + 1,
                found: OP
            }
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
        assert_eq!(table.open_handles(), 0);
        // Under its own operation the same handle is admitted.
        assert!(admit(&mut table, &live, 10, &handle).is_ok());
    }

    #[test]
    fn one_producer_cannot_use_more_than_its_share_of_open_handles() {
        let live = live();
        let mut table = HandleTable::new(
            HandlePolicy {
                max_open_per_principal: 2,
                ..policy()
            },
            CONSUMER,
        )
        .expect("policy");
        let _ = table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                PRODUCER_SESSION,
            )
            .expect("registered");
        let neighbour = SigningKey::from_bytes(&[0xE5; 32]);
        let _ = table
            .register_producer(
                "neighbour",
                neighbour.verifying_key(),
                "stage",
                PRODUCER_SESSION,
            )
            .expect("registered");
        admit(&mut table, &live, 10, &signed(1, 500)).expect("first");
        admit(&mut table, &live, 10, &signed(2, 500)).expect("second");
        assert_eq!(
            admit(&mut table, &live, 10, &signed(3, 500)).unwrap_err(),
            HandleRejection::TooManyOpenForPrincipal { limit: 2 }
        );
        // The neighbour still has room.
        let mut theirs = signed(1, 500);
        theirs.signature = neighbour.sign(&handle_signing_bytes(&theirs)).to_bytes();
        let result = table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: "neighbour",
            expected_operation: OP,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &AllLive,
            handle: HandleSource::Decoded(&theirs),
            footprint_bytes: theirs.byte_len,
            live: &live,
        });
        assert!(result.is_ok());
        assert_eq!(table.open_handles(), 3);
    }

    #[test]
    fn unregistering_a_producer_abandons_and_reports_its_open_handles() {
        let live = live();
        let mut table = table();
        let kept = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        let used = admit(&mut table, &live, 10, &signed(2, 500)).expect("admitted");
        let used = consume(&mut table, &live, 11, used).expect("consumed");
        assert!(table.finish(used).expect("this table").is_some());
        let notices = table.unregister_producer(PRINCIPAL);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].handle_nonce, 1);
        assert_eq!(notices[0].reason, ReclaimReason::ProducerRemoved);
        assert_eq!(table.open_handles(), 0);
        assert_eq!(
            consume(&mut table, &live, 12, kept).unwrap_err(),
            HandleRejection::Abandoned
        );
        assert_eq!(
            admit(&mut table, &live, 12, &signed(3, 500)).unwrap_err(),
            HandleRejection::UnknownPrincipal
        );
        assert!(table.unregister_producer("nobody").is_empty());
        // The tombstones outlive the removal: registered again, a used nonce is a replay.
        let _ = table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                PRODUCER_SESSION,
            )
            .expect("registered");
        assert_eq!(
            admit(&mut table, &live, 13, &signed(2, 500)).unwrap_err(),
            HandleRejection::Replayed
        );
        // After retention the entry of a removed producer is dropped.
        let _ = table.unregister_producer(PRINCIPAL);
        assert!(table.expire(600).is_empty());
        assert_eq!(table.slots_held(PRINCIPAL), None);
    }

    fn admit_under(
        table: &mut HandleTable,
        live: &Live,
        operations: &Operations,
        handle: &TensorHandle,
    ) -> Result<AdmittedHandle, HandleRejection> {
        table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: PRINCIPAL,
            expected_operation: OP,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations,
            handle: HandleSource::Decoded(handle),
            footprint_bytes: handle.byte_len,
            live,
        })
    }

    fn standard_chain() -> UpstreamChain {
        signed(1, 500).upstream_chain
    }

    #[test]
    fn a_live_but_different_chain_under_the_same_operation_is_refused_and_records_nothing() {
        let live = live();
        live.commit("adapter", 1);
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        let mut with_adapter = signed(1, 500);
        with_adapter.upstream_chain.adapters = vec![element("adapter")];
        resign(&mut with_adapter);
        assert_eq!(
            admit_under(&mut table, &live, &operations, &with_adapter).unwrap_err(),
            HandleRejection::Operation(OperationFault::ChainMismatch { operation: OP })
        );
        let mut other_generation = signed(2, 500);
        other_generation.upstream_chain.base_model.generation = 2;
        resign(&mut other_generation);
        assert_eq!(
            admit_under(&mut table, &live, &operations, &other_generation).unwrap_err(),
            HandleRejection::Operation(OperationFault::ChainMismatch { operation: OP })
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
        assert_eq!(table.open_handles(), 0);
        // The expected chain itself is admitted.
        assert!(admit_under(&mut table, &live, &operations, &signed(3, 500)).is_ok());
    }

    #[test]
    fn an_approximate_or_tainted_operation_is_refused_at_admit_and_at_consume() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        operations.inexact.borrow_mut().push(OP);
        assert_eq!(
            admit_under(&mut table, &live, &operations, &signed(1, 500)).unwrap_err(),
            HandleRejection::Operation(OperationFault::NotExact { operation: OP })
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
        assert_eq!(table.open_handles(), 0);
        // An Exact operation that turns Approximate between admit and consume is refused at
        // consume and the handle is abandoned with its budget kept.
        operations.inexact.borrow_mut().clear();
        let admitted = admit_under(&mut table, &live, &operations, &signed(2, 500)).expect("ok");
        operations.inexact.borrow_mut().push(OP);
        let result = table.consume(ConsumeRequest {
            now_ms: 20,
            admitted,
            live: &live,
            operations: &operations,
        });
        assert_eq!(
            result.unwrap_err(),
            HandleRejection::Operation(OperationFault::NotExact { operation: OP })
        );
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
    }

    /// Admit under a record, change the record, then consume: the fault must come from the
    /// consume-time check, abandon the handle and keep the budget until the reclaim notice.
    fn assert_consume_refuses_after(change: impl FnOnce(&Operations), fault: OperationFault) {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        let admitted = admit_under(&mut table, &live, &operations, &signed(1, 500)).expect("ok");
        change(&operations);
        let result = table.consume(ConsumeRequest {
            now_ms: 20,
            admitted,
            live: &live,
            operations: &operations,
        });
        // No ConsumedHandle: the locator is never returned.
        assert_eq!(result.unwrap_err(), HandleRejection::Operation(fault));
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        assert!(table.expire(599).is_empty());
        let notices = table.expire(600);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::Abandoned);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }

    #[test]
    fn an_expected_chain_that_changes_between_admit_and_consume_refuses_the_consume() {
        assert_consume_refuses_after(
            |operations| {
                let mut other = standard_chain();
                other.base_model.generation = 2;
                operations
                    .expected
                    .borrow_mut()
                    .insert(OP, (other, vec![TypeId::from(PAYLOAD)]));
            },
            OperationFault::ChainMismatch { operation: OP },
        );
    }

    #[test]
    fn accepted_payload_types_that_change_between_admit_and_consume_refuse_the_consume() {
        assert_consume_refuses_after(
            |operations| {
                operations.expected.borrow_mut().insert(
                    OP,
                    (standard_chain(), vec![TypeId::from("ptr.other.slot.v1")]),
                );
            },
            OperationFault::PayloadTypeNotAccepted {
                operation: OP,
                payload_type: TypeId::from(PAYLOAD),
            },
        );
    }

    #[test]
    fn the_debug_output_of_a_consumed_handle_omits_the_locator_and_payload() {
        let live = live();
        let mut table = table();
        let handle = signed(4, 500);
        let admitted = admit(&mut table, &live, 10, &handle).expect("admitted");
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        let shown = format!("{consumed:?}");
        assert!(!shown.contains("fd_slot"), "{shown}");
        assert!(!shown.contains("locator"), "{shown}");
        assert!(!shown.contains("payload_type"), "{shown}");
        assert!(!shown.contains(PAYLOAD), "{shown}");
        assert!(shown.contains("handle_nonce: 4"), "{shown}");
    }

    #[test]
    fn a_payload_type_the_operation_does_not_accept_is_refused_and_records_nothing() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        let mut other = signed(1, 500);
        other.payload_type = TypeId::from("ptr.kv.page");
        resign(&mut other);
        assert_eq!(
            admit_under(&mut table, &live, &operations, &other).unwrap_err(),
            HandleRejection::Operation(OperationFault::PayloadTypeNotAccepted {
                operation: OP,
                payload_type: TypeId::from("ptr.kv.page")
            })
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
        assert_eq!(table.open_handles(), 0);
    }

    #[test]
    fn the_operation_is_revalidated_for_the_computed_output_after_consume() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        let admitted = admit_under(&mut table, &live, &operations, &signed(1, 500)).expect("ok");
        let consumed = table
            .consume(ConsumeRequest {
                now_ms: 20,
                admitted,
                live: &live,
                operations: &operations,
            })
            .expect("consumed");
        assert_eq!(revalidate_operation(&consumed, &operations, 30), Ok(()));
        operations.dead.borrow_mut().push("semdb/a".to_owned());
        assert_eq!(
            revalidate_operation(&consumed, &operations, 30),
            Err(HandleRejection::Operation(OperationFault::InputNotLive {
                operation: OP,
                input: "semdb/a".to_owned()
            }))
        );
    }

    #[test]
    fn an_unknown_operation_is_refused_at_admission_and_records_nothing() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        assert_eq!(
            admit_under(&mut table, &live, &operations, &signed(1, 500)).unwrap_err(),
            HandleRejection::Operation(OperationFault::UnknownOperation { operation: OP })
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
        assert_eq!(table.open_handles(), 0);
        // A forged handle of an unknown operation is still a signature failure first.
        let mut forged = signed(2, 500);
        forged.signature[0] ^= 1;
        assert_eq!(
            admit_under(&mut table, &live, &operations, &forged).unwrap_err(),
            HandleRejection::BadSignature
        );
    }

    #[test]
    fn an_input_revoked_between_admit_and_consume_refuses_the_consume() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations
            .inputs
            .borrow_mut()
            .insert(OP, vec!["memory/a".to_owned(), "semdb/b".to_owned()]);
        let admitted =
            admit_under(&mut table, &live, &operations, &signed(1, 500)).expect("admitted");
        operations.dead.borrow_mut().push("semdb/b".to_owned());
        let result = table.consume(ConsumeRequest {
            now_ms: 20,
            admitted,
            live: &live,
            operations: &operations,
        });
        // No ConsumedHandle, so the locator is never released.
        assert_eq!(
            result.unwrap_err(),
            HandleRejection::Operation(OperationFault::InputNotLive {
                operation: OP,
                input: "semdb/b".to_owned()
            })
        );
        assert_eq!(table.open_handles(), 0);
        // Like a chain fault: the budget is kept until the reclaim at deadline plus skew.
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        let notices = table.expire(600);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::Abandoned);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }

    #[test]
    fn an_operation_with_a_dead_input_is_refused_at_admission() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations
            .inputs
            .borrow_mut()
            .insert(OP, vec!["memory/a".to_owned()]);
        operations.dead.borrow_mut().push("memory/a".to_owned());
        assert_eq!(
            admit_under(&mut table, &live, &operations, &signed(1, 500)).unwrap_err(),
            HandleRejection::Operation(OperationFault::InputNotLive {
                operation: OP,
                input: "memory/a".to_owned()
            })
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn debug_output_of_a_ticket_and_a_source_omits_the_locator() {
        let live = live();
        let mut table = table();
        let handle = shared(9, 500);
        let admitted = admit(&mut table, &live, 10, &handle).expect("admitted");
        let shown = format!("{admitted:?} {:?}", HandleSource::Decoded(&handle));
        assert!(!shown.contains("fd_slot"), "{shown}");
        assert!(!shown.contains("locator"), "{shown}");
    }

    #[test]
    fn a_decoded_bytes_handle_with_a_wrong_inline_digest_is_malformed() {
        let live = live();
        let mut table = table();
        let mut handle = signed(1, 500);
        handle.carrier = Carrier::Bytes;
        handle.locator = CarrierLocator::Bytes {
            inline: vec![9; 16],
        };
        resign(&mut handle);
        assert_eq!(
            admit(&mut table, &live, 10, &handle).unwrap_err(),
            HandleRejection::Malformed(HandleError::ContentMismatch)
        );
    }

    #[test]
    fn a_handle_larger_than_the_whole_budget_is_refused_whatever_is_in_flight() {
        let live = live();
        let mut table = table();
        let mut big = signed(1, 500);
        big.shape = vec![17];
        big.byte_len = 68;
        resign(&mut big);
        assert_eq!(
            admit(&mut table, &live, 10, &big).unwrap_err(),
            HandleRejection::InflightBytesExceeded {
                limit: 64,
                in_flight: 0,
                requested: 68
            }
        );
    }

    #[test]
    fn a_handle_at_exactly_its_deadline_is_not_reclaimable_until_the_skew_has_passed() {
        let live = live();
        let mut table = table();
        let _admitted = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        assert!(table.expire(500).is_empty());
        assert!(table.expire(599).is_empty());
        assert_eq!(table.open_handles(), 1);
        let notices = table.expire(600);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::Expired);
    }

    #[test]
    fn the_consumer_stops_at_the_deadline_while_the_producer_waits_out_the_skew() {
        let live = live();
        let mut table = table();
        let ok = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        let ok = consume(&mut table, &live, 499, ok).expect("inside the deadline");
        assert!(table.finish(ok).expect("this table").is_some());
        let late = admit(&mut table, &live, 10, &signed(2, 500)).expect("admitted");
        assert_eq!(
            consume(&mut table, &live, 500, late).unwrap_err(),
            HandleRejection::Expired
        );
        // Still open and not reclaimable inside the skew window.
        assert!(table.expire(550).is_empty());
        assert_eq!(table.expire(600).len(), 1);
    }

    #[test]
    fn consumed_bytes_are_verified_against_the_commitment() {
        let live = live();
        let mut table = table();
        let handle = signed(3, 500);
        let admitted = admit(&mut table, &live, 10, &handle).expect("admitted");
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        assert_eq!(consumed.verify_content(&[3u8; 16]), Ok(()));
        assert_eq!(
            consumed.verify_content(&[4u8; 16]),
            Err(HandleRejection::Content(HandleError::ContentMismatch))
        );
    }

    #[test]
    fn a_bytes_handle_reports_no_inline_payload_in_its_notice() {
        let live = live();
        let mut table = table();
        let mut handle = signed(1, 500);
        handle.carrier = Carrier::Bytes;
        handle.locator = CarrierLocator::Bytes {
            inline: vec![9; 16],
        };
        handle.content_digest = content_digest(&[9; 16]);
        resign(&mut handle);
        let admitted = admit(&mut table, &live, 10, &handle).expect("admitted");
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        assert_eq!(consumed.locator(), &handle.locator);
        assert_eq!(
            table
                .finish(consumed)
                .expect("a ticket of this table")
                .expect("bytes are not shared")
                .locator,
            CarrierLocator::Bytes { inline: vec![] }
        );
    }

    #[test]
    fn revalidate_operation_passes_the_values_captured_at_admission() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        let admitted = table
            .admit(HandleAdmissionRequest {
                now_ms: 10,
                principal: PRINCIPAL,
                expected_operation: OP,
                project: &project(),
                consumer_session: CONSUMER_SESSION,
                operations: &operations,
                handle: HandleSource::Decoded(&signed(1, 500)),
                footprint_bytes: 16,
                live: &live,
            })
            .expect("admitted");
        let consumed = table
            .consume(ConsumeRequest {
                now_ms: 20,
                admitted,
                live: &live,
                operations: &operations,
            })
            .expect("consumed");
        *operations.last.borrow_mut() = None;
        assert_eq!(revalidate_operation(&consumed, &operations, 30), Ok(()));
        assert_eq!(
            operations.last.borrow().clone(),
            Some((project(), CONSUMER_SESSION, PRODUCER_SESSION))
        );
        // The authority's record moves to another consumer session: the captured one is refused.
        operations
            .bound
            .borrow_mut()
            .insert(OP, (project(), WorkerSession(99), PRODUCER_SESSION));
        assert_eq!(
            revalidate_operation(&consumed, &operations, 30),
            Err(HandleRejection::Operation(
                OperationFault::ConsumerSessionMismatch { operation: OP }
            ))
        );
    }

    #[test]
    fn a_shared_buffer_stays_charged_after_consume_and_the_deadline_until_teardown() {
        let live = live();
        let mut table = table();
        let admitted = admit(&mut table, &live, 10, &shared(1, 500)).expect("admitted");
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        // `Consumed` comes from the consumer worker: it frees nothing for a shared carrier.
        assert_eq!(table.finish(consumed), Ok(None));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        // Neither does the deadline plus the skew: the slot and its bytes stay.
        assert!(table.expire(600).is_empty());
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        assert_eq!(table.slots_held(PRINCIPAL), Some(1));
        // Verified teardown releases it, once.
        let notices = table.release_after_teardown(torn_down());
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::TornDown);
        assert_eq!(notices[0].carrier, Carrier::HostShm);
        assert_eq!(notices[0].handle_nonce, 1);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
        assert!(table.release_after_teardown(torn_down()).is_empty());
        // The tombstone is dropped at the next expire after its retention.
        assert!(table.expire(600).is_empty());
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn repeated_shared_consumes_without_teardown_saturate_the_budget() {
        let live = live();
        let mut table = table();
        for nonce in 1..=4 {
            let admitted = admit(&mut table, &live, 10, &shared(nonce, 500)).expect("admitted");
            let consumed = consume(&mut table, &live, 11, admitted).expect("consumed");
            assert_eq!(table.finish(consumed), Ok(None));
        }
        assert_eq!(
            admit(&mut table, &live, 12, &shared(5, 500)).unwrap_err(),
            HandleRejection::InflightBytesExceeded {
                limit: 64,
                in_flight: 64,
                requested: 16
            }
        );
        assert_eq!(table.release_after_teardown(torn_down()).len(), 4);
        assert!(admit(&mut table, &live, 13, &shared(5, 500)).is_ok());
    }

    #[test]
    fn an_open_shared_handle_past_its_deadline_is_closed_but_keeps_its_bytes_and_nonce() {
        let live = live();
        let mut table = table();
        let _admitted = admit(&mut table, &live, 10, &shared(1, 500)).expect("admitted");
        assert!(table.expire(600).is_empty());
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        // The nonce stays refused while the slot is kept.
        assert_eq!(
            admit(&mut table, &live, 601, &shared(1, 1_000)).unwrap_err(),
            HandleRejection::Abandoned
        );
        let notices = table.release_after_teardown(torn_down());
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::TornDown);
    }

    #[test]
    fn teardown_closes_open_shared_handles_and_leaves_other_carriers_alone() {
        let live = live();
        let mut table = table();
        let _open = admit(&mut table, &live, 10, &shared(1, 500)).expect("admitted");
        let _other = admit(&mut table, &live, 10, &signed(2, 500)).expect("admitted");
        assert_eq!(table.open_handles(), 2);
        let notices = table.release_after_teardown(torn_down());
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].handle_nonce, 1);
        assert_eq!(table.open_handles(), 1);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
    }

    #[test]
    fn a_replaced_producer_releases_no_shared_buffer() {
        let live = live();
        let mut table = table();
        let _open = admit(&mut table, &live, 10, &shared(1, 500)).expect("admitted");
        let _plain = admit(&mut table, &live, 10, &signed(2, 500)).expect("admitted");
        let released = table
            .register_producer(
                PRINCIPAL,
                signer().verifying_key(),
                "stage",
                WorkerSession(PRODUCER_SESSION.0 + 1),
            )
            .expect("registered");
        // Only the non-shared claim is returned; both handles are closed.
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].handle_nonce, 2);
        assert_eq!(released[0].reason, ReclaimReason::ProducerReplaced);
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        let notices = table.release_after_teardown(torn_down());
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].handle_nonce, 1);
        assert_eq!(table.open_handles(), 0);
    }

    #[test]
    fn output_computed_at_or_after_the_handle_deadline_is_refused() {
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        let admitted = admit_under(&mut table, &live, &operations, &signed(1, 500)).expect("ok");
        let consumed = table
            .consume(ConsumeRequest {
                now_ms: 20,
                admitted,
                live: &live,
                operations: &operations,
            })
            .expect("consumed");
        // The operation outlives the handle; the handle's own deadline still bounds the output.
        assert_eq!(revalidate_operation(&consumed, &operations, 499), Ok(()));
        assert_eq!(
            revalidate_operation(&consumed, &operations, 500),
            Err(HandleRejection::Expired)
        );
        assert_eq!(
            revalidate_operation(&consumed, &operations, 10_000),
            Err(HandleRejection::Expired)
        );
    }

    #[test]
    fn a_late_teardown_of_the_old_pair_leaves_the_handles_of_the_respawned_pair_alone() {
        let live = live();
        let mut table = table();
        let old = admit(&mut table, &live, 10, &shared(1, 500)).expect("admitted");
        let _ = consume(&mut table, &live, 11, old).expect("consumed");
        // The supervisor respawns the producer under a new session; the new worker exports
        // and its handle is admitted before the teardown of the old pair is reported.
        let respawned = WorkerSession(PRODUCER_SESSION.0 + 1);
        assert!(table
            .register_producer(PRINCIPAL, signer().verifying_key(), "stage", respawned)
            .expect("registered")
            .is_empty());
        let _new = admit(&mut table, &live, 12, &shared(2, 500)).expect("admitted");
        let notices = table.release_after_teardown(torn_down());
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].handle_nonce, 1);
        assert_eq!(table.open_handles(), 1);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        // A teardown naming another consumer session releases nothing either.
        assert!(table
            .release_after_teardown(Teardown {
                principal: PRINCIPAL,
                producer_session: respawned,
                consumer_session: WorkerSession(CONSUMER_SESSION.0 + 1),
            })
            .is_empty());
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
    }

    #[test]
    fn closing_a_shared_handle_at_expiry_and_its_teardown_count_it_once() {
        let live = live();
        let mut table = table();
        let _shared = admit(&mut table, &live, 10, &shared(1, 500)).expect("admitted");
        let _plain = admit(&mut table, &live, 10, &signed(2, 1_000)).expect("admitted");
        assert_eq!(table.open_handles(), 2);
        assert!(table.expire(600).is_empty());
        assert_eq!(table.open_handles(), 1);
        assert_eq!(table.release_after_teardown(torn_down()).len(), 1);
        assert_eq!(table.open_handles(), 1);
        let late = table.expire(1_100);
        assert_eq!(late.len(), 1);
        assert_eq!(late[0].reason, ReclaimReason::Expired);
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_removed_producer_keeps_its_shared_bytes_until_teardown_and_then_goes() {
        let live = live();
        let mut table = table();
        let _open = admit(&mut table, &live, 10, &shared(1, 500)).expect("admitted");
        assert!(table.unregister_producer(PRINCIPAL).is_empty());
        assert_eq!(table.open_handles(), 0);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        assert!(table.expire(600).is_empty());
        assert_eq!(table.slots_held(PRINCIPAL), Some(1));
        assert_eq!(table.release_after_teardown(torn_down()).len(), 1);
        assert!(table.expire(600).is_empty());
        assert_eq!(table.slots_held(PRINCIPAL), None);
    }

    #[test]
    fn a_handle_accepted_by_another_stage_than_the_recorded_destination_is_refused() {
        // One worker session may serve two stages: the session alone does not prove the
        // destination, so the authority compares the accepting stage with its record.
        let live = live();
        let mut table = table();
        let operations = Operations::default();
        operations.know(OP, &["semdb/a"], standard_chain(), &[PAYLOAD]);
        operations
            .destination
            .borrow_mut()
            .insert(OP, "another-stage".to_owned());
        assert_eq!(
            admit_under(&mut table, &live, &operations, &signed(1, 500)).unwrap_err(),
            HandleRejection::Operation(OperationFault::ConsumerStageMismatch { operation: OP })
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
        operations
            .destination
            .borrow_mut()
            .insert(OP, CONSUMER.to_owned());
        assert!(admit_under(&mut table, &live, &operations, &signed(1, 500)).is_ok());
    }

    fn admit_with_footprint(
        table: &mut HandleTable,
        live: &Live,
        handle: &TensorHandle,
        footprint_bytes: u64,
    ) -> Result<AdmittedHandle, HandleRejection> {
        table.admit(HandleAdmissionRequest {
            now_ms: 10,
            principal: PRINCIPAL,
            expected_operation: OP,
            project: &project(),
            consumer_session: CONSUMER_SESSION,
            operations: &AllLive,
            handle: HandleSource::Decoded(handle),
            live,
            footprint_bytes,
        })
    }

    #[test]
    fn the_budget_is_charged_with_the_allocated_footprint_not_the_tensor_length() {
        let live = live();
        let mut table = table();
        // A 16-byte tensor in a 32-byte allocation (rounding and padding) is charged 32.
        let admitted = admit_with_footprint(&mut table, &live, &signed(1, 500), 32).expect("ok");
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(32));
        assert_eq!(
            admit_with_footprint(&mut table, &live, &signed(2, 500), 40).unwrap_err(),
            HandleRejection::InflightBytesExceeded {
                limit: 64,
                in_flight: 32,
                requested: 40
            }
        );
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        let notice = table
            .finish(consumed)
            .expect("this table")
            .expect("not shared");
        assert_eq!(notice.byte_len, 16);
        assert_eq!(notice.charged_bytes, 32);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_footprint_below_the_tensor_length_is_refused_and_records_nothing() {
        let live = live();
        let mut table = table();
        assert_eq!(
            admit_with_footprint(&mut table, &live, &signed(1, 500), 15).unwrap_err(),
            HandleRejection::FootprintBelowLength {
                footprint: 15,
                byte_len: 16
            }
        );
        assert_eq!(table.slots_held(PRINCIPAL), Some(0));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }

    #[test]
    fn a_consumed_handle_dropped_unfinished_stays_charged_until_deadline_plus_skew() {
        let live = live();
        let mut table = table();
        let admitted = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        drop(consume(&mut table, &live, 20, admitted).expect("consumed"));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        assert!(table.expire(599).is_empty());
        let notices = table.expire(600);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].reason, ReclaimReason::Consumed);
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }

    #[test]
    fn finish_reports_once_and_refuses_a_handle_of_another_table() {
        let live = live();
        let mut other = table();
        let mut table = table();
        let admitted = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        assert_eq!(other.finish(consumed), Err(HandleRejection::UnknownTicket));
        let admitted = admit(&mut table, &live, 10, &signed(2, 500)).expect("admitted");
        let consumed = consume(&mut table, &live, 20, admitted).expect("consumed");
        assert!(table.finish(consumed).expect("this table").is_some());
        // Only the handle the other table refused is still charged here; expiry reports it,
        // and the finished one is not reported again.
        let notices = table.expire(600);
        assert_eq!(notices.len(), 1);
        assert_eq!(notices[0].handle_nonce, 1);
    }

    #[test]
    fn a_stale_consumed_handle_never_finishes_a_later_use_of_the_same_nonce() {
        let live = live();
        let mut table = table();
        let first = admit(&mut table, &live, 10, &signed(1, 500)).expect("admitted");
        let stale = consume(&mut table, &live, 20, first).expect("consumed");
        // Unfinished, it is reported at the deadline plus the skew and its slot is dropped.
        assert_eq!(table.expire(600).len(), 1);
        // The producer uses the nonce again under a later deadline.
        let again = admit(&mut table, &live, 601, &signed(1, 1_000)).expect("admitted");
        let current = consume(&mut table, &live, 602, again).expect("consumed");
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        // The stale handle releases nothing of the current use.
        assert_eq!(table.finish(stale), Ok(None));
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(16));
        assert!(table.finish(current).expect("this table").is_some());
        assert_eq!(table.inflight_bytes(PRINCIPAL), Some(0));
    }
}
