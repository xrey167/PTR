//! Round-robin over interchangeable inference replicas, with bounded failover.
//!
//! [`ReplicaRouter`] implements [`ResumableInferenceBackend`] over replicas that all serve the
//! same model. A request starts at the next replica in rotation; if that replica fails it may
//! be tried on up to `max_failover` others, each at most once. Counters per replica say how
//! many calls reached it, how many failed and how many were failovers onto it.
//!
//! **Which failures fail over.** An allowlist, not a denylist: only overload and a backend
//! failure ([`ModelError::allows_failover`]), the two kinds another replica can change. A
//! refusal of the request itself (validation, policy, kind `Other`), a passed deadline, a
//! cancellation, a revoked generation and a closed replica end the request. The kind is a typed
//! field of the error; the message is never read.
//!
//! **Resume does not fail over by default.** A `resume` continues a run that started on some
//! replica; whether another replica may take it over is a property of the replicas, so it is
//! chosen explicitly with [`ResumeFailover::Interchangeable`], and
//! [`ResumeFailover::Never`] tries the rotation's replica once.

use crate::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// The most replicas one router fronts.
pub const MAX_REPLICAS: usize = 64;

/// Why a router could not be built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplicaError {
    /// No replica was given.
    NoReplicas,
    /// More than [`MAX_REPLICAS`] were given.
    TooManyReplicas { limit: usize, actual: usize },
    /// `max_failover` is not smaller than the number of replicas, so a request would be tried
    /// on the same replica twice.
    FailoverExceedsReplicas { max_failover: u8, replicas: usize },
}

impl fmt::Display for ReplicaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoReplicas => write!(formatter, "a replica router needs at least one replica"),
            Self::TooManyReplicas { limit, actual } => {
                write!(formatter, "{actual} replicas exceed the limit of {limit}")
            }
            Self::FailoverExceedsReplicas {
                max_failover,
                replicas,
            } => write!(
                formatter,
                "max_failover {max_failover} must be below the {replicas} replicas"
            ),
        }
    }
}

impl std::error::Error for ReplicaError {}

/// Whether a `resume` may be retried on another replica.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResumeFailover {
    /// One replica, one try.
    Never,
    /// The replicas are interchangeable for a resume (it carries its whole observation), so it
    /// fails over like `infer`.
    Interchangeable,
}

/// How a router fails over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicaPolicy {
    /// The most replicas, other than the first, one `infer` may be tried on.
    pub max_failover: u8,
    pub resume: ResumeFailover,
}

/// Counters of one replica since the router was built.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplicaStats {
    /// Calls that reached this replica, first tries and failovers alike.
    pub attempts: u64,
    /// Calls that returned an error.
    pub failures: u64,
    /// Calls that reached this replica because an earlier one failed.
    pub failovers: u64,
}

#[derive(Default)]
struct ReplicaCounters {
    attempts: AtomicU64,
    failures: AtomicU64,
    failovers: AtomicU64,
}

struct Replica {
    backend: Arc<dyn ResumableInferenceBackend>,
    counters: ReplicaCounters,
}

/// Round-robin over replicas with failover; see the module documentation.
pub struct ReplicaRouter {
    replicas: Vec<Replica>,
    next: AtomicUsize,
    policy: ReplicaPolicy,
}

impl fmt::Debug for ReplicaRouter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReplicaRouter")
            .field("replicas", &self.replicas.len())
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl ReplicaRouter {
    /// # Errors
    /// [`ReplicaError`] for no replica, more than [`MAX_REPLICAS`], or a `max_failover` that
    /// is not below the replica count.
    pub fn new(
        replicas: Vec<Arc<dyn ResumableInferenceBackend>>,
        policy: ReplicaPolicy,
    ) -> Result<Self, ReplicaError> {
        let max_failover = policy.max_failover;
        if replicas.is_empty() {
            return Err(ReplicaError::NoReplicas);
        }
        if replicas.len() > MAX_REPLICAS {
            return Err(ReplicaError::TooManyReplicas {
                limit: MAX_REPLICAS,
                actual: replicas.len(),
            });
        }
        if usize::from(max_failover) >= replicas.len() {
            return Err(ReplicaError::FailoverExceedsReplicas {
                max_failover,
                replicas: replicas.len(),
            });
        }
        Ok(Self {
            replicas: replicas
                .into_iter()
                .map(|backend| Replica {
                    backend,
                    counters: ReplicaCounters::default(),
                })
                .collect(),
            next: AtomicUsize::new(0),
            policy,
        })
    }

    /// The counters of every replica, in the order they were given.
    pub fn stats(&self) -> Vec<ReplicaStats> {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        self.replicas
            .iter()
            .map(|replica| ReplicaStats {
                attempts: load(&replica.counters.attempts),
                failures: load(&replica.counters.failures),
                failovers: load(&replica.counters.failovers),
            })
            .collect()
    }

    fn route<T>(
        &self,
        max_failover: u8,
        call: impl Fn(&dyn ResumableInferenceBackend) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        let start = self.next.fetch_add(1, Ordering::Relaxed) % self.replicas.len();
        let mut last_error = ModelError::backend_failure();
        for step in 0..=usize::from(max_failover) {
            let replica = &self.replicas[(start + step) % self.replicas.len()];
            replica.counters.attempts.fetch_add(1, Ordering::Relaxed);
            if step > 0 {
                replica.counters.failovers.fetch_add(1, Ordering::Relaxed);
            }
            match call(replica.backend.as_ref()) {
                Ok(value) => return Ok(value),
                Err(error) => {
                    replica.counters.failures.fetch_add(1, Ordering::Relaxed);
                    if !error.allows_failover() {
                        return Err(error);
                    }
                    last_error = error;
                }
            }
        }
        Err(last_error)
    }
}

impl InferenceBackend for ReplicaRouter {
    fn name(&self) -> &'static str {
        "replica-router"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.route(self.policy.max_failover, |backend| backend.infer(request))
    }
}

impl ResumableInferenceBackend for ReplicaRouter {
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        let max_failover = match self.policy.resume {
            ResumeFailover::Never => 0,
            ResumeFailover::Interchangeable => self.policy.max_failover,
        };
        self.route(max_failover, |backend| backend.resume(request))
    }
}
