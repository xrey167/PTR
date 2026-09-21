//! Raft messages over an authenticated connection.
//!
//! `ptr-ledger` holds consensus and must not know about transport; `ptr-net` holds
//! transport and must not know about the log. This crate is the one place that
//! knows both, and it exists so that neither of those two has to.
//!
//! The property it adds over plumbing is small and load-bearing: **the sender is
//! the authenticated connection, never the message's own `from` field.** A peer can
//! put any id in a raft message, and a receiver that believes it will accept a vote
//! or an append attributed to a member that never sent it. So a frame is only
//! stepped when the id its messages claim is the id bound to the public key the
//! connection actually authenticated.
//!
//! What this is not: it is not a daemon. There is no background loop, no timer and
//! no retry policy here. A caller drives `serve_once` and `dispatch`, which keeps
//! the tests free of sleeps and keeps the scheduling decisions — how often to tick,
//! when to give up on a peer — where they can be made deliberately.
mod frame;

pub use frame::{decode_batch, encode_batch, FrameError, MAX_BATCH_MESSAGES, MAX_FRAME_BYTES};

#[cfg(feature = "cluster-backend")]
mod member;

#[cfg(feature = "cluster-backend")]
pub use member::{ClusterError, ClusterMember, MemberAddress};
