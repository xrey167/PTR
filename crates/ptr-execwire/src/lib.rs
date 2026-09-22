//! An action requested across a network boundary, and the receipt that answers it.
//!
//! `ptr-runtime` holds execution authority and must not know about transport;
//! `ptr-net` holds transport and must not know what an action is. This crate is the
//! one place that knows both, and it exists so that neither of those two has to.
//!
//! The sentence the whole layer is built around: **authenticating a peer is not
//! authorizing an action.** A connection proves which key is at the other end and
//! nothing else. What that key may ask for is host policy, looked up at every use,
//! and what the action is audited under is the policy's principal — never a name
//! the request supplied, because a request has no field in which to supply one.
//!
//! Two mechanisms that look alike and are not. A `request_id` is a nonce inside a
//! bounded per-peer window, so a frame captured and sent twice is refused cheaply;
//! it does not survive a restart and it does not span the window. An
//! **at-most-once key** is the runtime's, recorded in the ledger, and it is what
//! makes a retry apply once. Retrying an intent therefore means a *new*
//! `request_id` with the *same* key, which is the only combination that says "this
//! one again" rather than "do it once more".
//!
//! What this is not: it is not a daemon. Nothing here loops, ticks or retries. A
//! caller drives `serve_once` and `request`, which keeps the tests free of sleeps
//! and keeps scheduling decisions where they can be made deliberately.
mod frame;

pub use frame::{
    decode_receipt, decode_request, encode_receipt, encode_request, request_digest, FrameError,
    RefusalCode, WireOutcome, WireReceipt, WireRequest, FORMAT_V1, MAX_BODY_BYTES, MAX_FIELD_BYTES,
    MAX_FRAME_BYTES,
};

#[cfg(feature = "execwire-backend")]
mod endpoint;

#[cfg(feature = "execwire-backend")]
pub use endpoint::{
    ExecutionClient, ExecutionHost, Serviced, WireError, DEFAULT_PERMIT_TTL,
    DEFAULT_REQUEST_TIMEOUT, MAX_REPLAY_WINDOW, MAX_TRACKED_PEERS,
};
