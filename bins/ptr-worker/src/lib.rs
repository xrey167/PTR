//! Worker host for deterministic CPU reference Pods.
//!
//! The worker speaks the generic control protocol of `ptr-protocol` (Hello negotiation,
//! idempotent `Call`/`Reply`, `Cancel`, `Bye` with drain) over a stdio byte stream and
//! dispatches each call to a small built-in [`ptr_pods::PodRegistry`]. It loads no model
//! weights and opens no network socket; it is a separate process, nothing more isolated.

pub mod pods;
pub mod serve;
pub mod transport;
pub mod wire;
