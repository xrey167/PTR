//! Worker host for deterministic CPU reference Pods.
//!
//! The worker speaks the generic control protocol of `ptr-protocol` (Hello negotiation,
//! idempotent `Call`/`Reply`, `Cancel`, `Bye` with drain) over a stdio byte stream and
//! dispatches each call to a small built-in [`ptr_pods::PodRegistry`]. It loads no model
//! weights and opens no network socket; it is a separate process, nothing more isolated.
//!
//! With the `data-plane` feature (Linux only) the same session also runs over a
//! `SOCK_SEQPACKET` channel inherited from a supervisor (`channel_serve`), tensors travel by
//! handle over it (`data_wire`), and `host` is the supervisor side that spawns confined
//! workers, admits their handles and releases quota charges only when a worker is reaped.

#[cfg(all(feature = "data-plane", target_os = "linux"))]
pub mod channel_serve;
#[cfg(all(feature = "data-plane", target_os = "linux"))]
pub mod data_wire;
#[cfg(all(feature = "data-plane", target_os = "linux"))]
pub mod host;
pub mod pods;
pub mod serve;
pub mod transport;
pub mod wire;
