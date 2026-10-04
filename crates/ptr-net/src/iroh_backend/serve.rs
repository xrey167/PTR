//! The accept loop: mechanism here, policy with the host (D4 in issue #23).
//!
//! Every wire crate answers one connection at a time with `serve_once`, and until
//! now nothing ran that in a loop — on purpose, because a loop decides when to give
//! up, and a loop written inside a protocol crate would make that decision for every
//! deployment without anybody having made it. D4 is decided as a split: **the loop is
//! mechanism and lives here, once**; *when to give up* is a [`ServePolicy`] the host
//! writes down, field by field, with no default to fall back on.
//!
//! What the loop does, and nothing more:
//!
//! - **Bounded concurrency with backpressure, not a queue.** At most
//!   `max_concurrent` connections are being handled. One that arrives over the limit
//!   is refused at once — before its handshake, so it costs the host nothing — and is
//!   not parked until a slot frees: a queue in front of a saturated host is latency
//!   the peer did not agree to, and its length would be one more limit nobody set.
//! - **A deadline per connection.** Handshake, request and handler together get
//!   `per_connection_deadline`; a connection still open at the deadline is closed,
//!   and the slot it held is freed. A peer that stops talking holds one slot for at
//!   most that long and holds nobody else up.
//! - **Shutdown.** Once `shutdown` reads `true` the loop accepts nothing more, waits
//!   for the connections already admitted (each bounded by its deadline), and returns.
//!   A dropped sender is read as shutdown: a loop nobody can stop any more is a loop
//!   that should not keep accepting.
//! - **No retry.** A connection that fails is counted and forgotten. Retrying is the
//!   requester's decision, under its own at-most-once key where effects are involved.
use super::{IrohIncoming, IrohTransport};
use iroh::endpoint::Connection;
use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{watch, Semaphore};
use tokio::task::JoinSet;
use tokio::time::Instant;

/// When the host gives up, written down. Every field is the host's decision and none
/// has a default: a deployment that has not decided how many connections it will
/// hold, or how long it will wait for one, has not decided to serve.
#[derive(Clone, Debug)]
pub struct ServePolicy {
    /// How many connections are handled at once. One more is refused, not queued.
    pub max_concurrent: NonZeroUsize,
    /// How long one connection — handshake, request and answer — may take before it
    /// is closed. Must be greater than zero.
    pub per_connection_deadline: Duration,
    /// Reads `true` when the host wants the loop to stop.
    pub shutdown: watch::Receiver<bool>,
}

/// What one run of the loop did, for the host's own records.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ServeReport {
    /// Connections whose handler ran to completion and reported success.
    pub handled: u64,
    /// Connections refused because `max_concurrent` were already being handled.
    pub refused_over_limit: u64,
    /// Connections closed at their deadline.
    pub cut_at_deadline: u64,
    /// Connections whose handshake, request or handler failed before the deadline.
    pub failed: u64,
}

/// Why the loop did not start.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServeError {
    /// A zero deadline would close every connection before it could be answered.
    ZeroDeadline,
}

impl ServeError {
    /// Stable diagnostic code for this refusal.
    pub fn code(&self) -> &'static str {
        match self {
            Self::ZeroDeadline => "PTR_NET_SERVE_ZERO_DEADLINE",
        }
    }
}

impl std::fmt::Display for ServeError {
    /// Render the stable refusal code.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl std::error::Error for ServeError {}

/// How one connection ended.
enum Ended {
    Handled,
    Cut,
    Failed,
}

/// Accept connections on `transport` until `policy.shutdown` reads `true`, reading
/// one request of at most `max_request` bytes from each and handing it to `handler`.
///
/// The handler is the protocol's own per-connection code — the same code its
/// `serve_once` runs — so the loop adds scheduling and nothing else. Its `Err` is
/// counted as a failure and otherwise belongs to the handler, which is where the
/// protocol's own refusal was already decided and answered.
pub async fn serve<H, F>(
    transport: &IrohTransport,
    max_request: usize,
    policy: ServePolicy,
    handler: H,
) -> Result<ServeReport, ServeError>
where
    H: Fn(IrohIncoming) -> F + Send + Sync + 'static,
    F: Future<Output = Result<(), String>> + Send + 'static,
{
    if policy.per_connection_deadline.is_zero() {
        return Err(ServeError::ZeroDeadline);
    }
    let ServePolicy {
        max_concurrent,
        per_connection_deadline,
        mut shutdown,
    } = policy;
    let handler = Arc::new(handler);
    let permits = Arc::new(Semaphore::new(max_concurrent.get()));
    let mut connections: JoinSet<Ended> = JoinSet::new();
    let mut report = ServeReport::default();

    loop {
        if *shutdown.borrow_and_update() {
            break;
        }
        let incoming = tokio::select! {
            // Shutdown first: a connection arriving in the same instant as the
            // instruction to stop is not one the host still wants.
            biased;
            stop = shutdown.changed() => {
                if stop.is_err() || *shutdown.borrow_and_update() {
                    break;
                }
                continue;
            }
            Some(ended) = connections.join_next(), if !connections.is_empty() => {
                record(&mut report, ended);
                continue;
            }
            incoming = transport.endpoint.accept() => match incoming {
                Some(incoming) => incoming,
                // The endpoint closed: there is nothing left to accept from.
                None => break,
            },
        };

        // Backpressure is a refusal, decided before the handshake.
        let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
            incoming.refuse();
            report.refused_over_limit += 1;
            continue;
        };

        let handler = Arc::clone(&handler);
        connections.spawn(async move {
            let _permit = permit;
            let deadline = Instant::now() + per_connection_deadline;
            // Where the connection is, once its handshake completes, so it can be
            // closed with a reason when the deadline cuts the work short.
            let open: Arc<Mutex<Option<Connection>>> = Arc::new(Mutex::new(None));
            let slot = Arc::clone(&open);
            let work = async move {
                let connection = incoming.await.map_err(|error| error.to_string())?;
                *slot.lock().expect("connection slot is not poisoned") = Some(connection.clone());
                let request = super::read_request(connection, max_request).await?;
                handler(request).await
            };
            match tokio::time::timeout_at(deadline, work).await {
                Ok(Ok(())) => Ended::Handled,
                Ok(Err(_)) => Ended::Failed,
                Err(_) => {
                    if let Some(connection) =
                        open.lock().expect("connection slot is not poisoned").take()
                    {
                        connection.close(1u32.into(), b"ptr connection deadline");
                    }
                    Ended::Cut
                }
            }
        });
    }

    // Accept nothing more; let what was admitted finish, each within its deadline.
    while let Some(ended) = connections.join_next().await {
        record(&mut report, ended);
    }
    Ok(report)
}

/// Count how a connection ended. A handler task that panicked is a failure.
fn record(report: &mut ServeReport, ended: Result<Ended, tokio::task::JoinError>) {
    match ended {
        Ok(Ended::Handled) => report.handled += 1,
        Ok(Ended::Cut) => report.cut_at_deadline += 1,
        Ok(Ended::Failed) | Err(_) => report.failed += 1,
    }
}
