//! Pod access across a network boundary: one invocation, and the answer to it.
//!
//! `ALPN_PODWIRE` has existed since `ptr-net` declared it. Until now the only
//! bytes that ever travelled over it were `b"ping"` in a transport test — a label
//! borrowed to prove a connection authenticates its peer, which is not a protocol
//! and never claimed to be. This crate is the protocol.
//!
//! **Why this is not `ALPN_EXEC` with a different payload.** The execution wire
//! asks for an *effect*: it commits an attempt before the effect, keeps a fence
//! while the outcome is unknown, and can answer "uncertain". Pod access asks a
//! Pure or Read Pod a question. Nothing applies, so nothing can half-apply, and
//! every mechanism the execution wire needs for that — the attempt, the fence, the
//! at-most-once key, the replay window, two of its four outcomes — is absent here.
//! Absent by derivation, not by omission: each one is named in the place it would
//! have gone, with the reason it is not there. Two protocols sharing one ALPN is
//! how a request meant for one gets parsed by the other, and the parse that
//! succeeds by accident is the dangerous one.
//!
//! **What a requester does not get to say.** A request names no sender, no session
//! and no project. The first two are the execution wire's rule, kept. The third is
//! this protocol's own, and it is the load-bearing one: `PodRegistry` is keyed by
//! project because resolution is the only thing standing between a request and a
//! Pod's data, so a requester that named its project would pick which project's
//! Pods it reaches. The project comes from host policy, keyed by the peer the
//! connection authenticated.
//!
//! **What this host cannot do to itself.** A `PodHost` holds a registry, a
//! policy and a verifier. It holds no runtime and no ledger, so "a Pod invoked
//! over this wire writes nothing to the host's history" is a fact about the type
//! rather than a promise about the code: there is nothing here to write to.
//!
//! What this is not: it is not a daemon. Nothing here loops, ticks or retries. A
//! caller drives `serve_once` and `request`.
mod access;
mod frame;

pub use access::{answer, PodAccessPolicy, PodScope, PolicyError};
pub use frame::{
    decode_answer, decode_request, encode_answer, encode_request, request_digest, FrameError,
    PodAnswer, PodOutcome, PodRequest, RefusalCode, FORMAT_V1, MAX_BODY_BYTES, MAX_FIELD_BYTES,
    MAX_FRAME_BYTES,
};

#[cfg(feature = "podwire-backend")]
mod endpoint;

#[cfg(feature = "podwire-backend")]
pub use endpoint::{PodClient, PodHost, PodWireError, Served, DEFAULT_REQUEST_TIMEOUT};
