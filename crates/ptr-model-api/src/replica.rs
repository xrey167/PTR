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
//! **Resume does not fail over by default, and goes back where the run started.** A `resume`
//! continues a run that started on some replica. [`ReplicaRouter::infer_placed`] returns the
//! answer with a [`Placement`] (per invocation; a request id is live until its
//! run ends or the placement is dropped, and a second `infer_placed` under a live id is refused
//! before any replica runs; an unguessable token, bound in a bounded table to the router, the request id and
//! the semantic context, released when the placement is dropped), and
//! [`ReplicaRouter::resume_placed`] checks those and consumes it: under
//! [`ResumeFailover::Never`] the resume goes to exactly that replica, once. The plain trait
//! `resume` carries no placement and takes the rotation's replica, which a stateful replica
//! will refuse. Whether another replica may take a run over is a property of the replicas, so
//! it is chosen explicitly with [`ResumeFailover::Interchangeable`], the only policy under
//! which a resume may move.

use crate::{
    InferenceBackend, ModelError, ModelEvent, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_types::{RequestId, SemanticContext};
use std::collections::HashMap;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

/// The most replicas one router fronts.
pub const MAX_REPLICAS: usize = 64;

/// The most placements one router keeps open at once. When the table is full, a new
/// [`ReplicaRouter::infer_placed`] is refused with [`ModelError::overloaded`]; no open
/// placement is evicted.
pub const MAX_OPEN_PLACEMENTS: usize = 4096;

/// The most placements one principal may hold open at once, reservations of calls still running
/// included. A principal at its quota is refused with [`ModelError::overloaded`] and nothing of
/// anyone else is touched, so one caller cannot take the router-wide table from the others.
/// Calls that name no principal share one anonymous bucket with the same quota; they are
/// limited, not exempt.
pub const MAX_PLACEMENTS_PER_PRINCIPAL: usize = 256;

/// Who an [`ReplicaRouter::infer_placed`] call is made for. `ModelRequest` carries no principal,
/// so the caller states it here.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlacementOptions {
    /// The principal charged for the placement; `None` is the shared anonymous bucket.
    pub principal: Option<String>,
}

/// What the router remembers about one open invocation.
struct OpenPlacement {
    replica: usize,
    request_id: RequestId,
    principal: Option<String>,
    context: [u8; 32],
    /// A resume is running on it: it keeps its slot for the successor and cannot be used twice.
    in_flight: bool,
}

#[derive(Default)]
struct Placements {
    entries: HashMap<u128, OpenPlacement>,
    /// Slots promised to `infer_placed` calls whose replica is still running.
    reserved: usize,
    /// Request ids of every live invocation: an open placement (a resume in flight included) or
    /// an `infer_placed` whose replica is still running. An id is free again once its run ends,
    /// fails or its placement is dropped.
    live: HashMap<RequestId, Option<String>>,
    /// Live invocations per principal (the anonymous bucket is `None`), reservations included.
    held: HashMap<Option<String>, usize>,
}

impl Placements {
    /// Charge `principal` for the live id.
    fn claim(&mut self, id: RequestId, principal: Option<String>) {
        *self.held.entry(principal.clone()).or_insert(0) += 1;
        self.live.insert(id, principal);
    }

    /// The id is no longer live: give the principal's slot back.
    fn release(&mut self, id: &RequestId) {
        if let Some(principal) = self.live.remove(id) {
            if let Some(count) = self.held.get_mut(&principal) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.held.remove(&principal);
                }
            }
        }
    }
}

/// The replica that answered one invocation, handed back by [`ReplicaRouter::infer_placed`] and
/// consumed by [`ReplicaRouter::resume_placed`]. It carries only an unguessable 128-bit
/// invocation token, the id of the router that issued it and a weak reference to that router's
/// table; the replica, the request id and the semantic context live in the table under the
/// token. It is per invocation (a request id is live at most once per router), not
/// `Clone`, has no public constructor, and **dropping it releases its entry**.
pub struct Placement {
    token: u128,
    router: u64,
    table: Weak<Mutex<Placements>>,
}

impl Drop for Placement {
    fn drop(&mut self) {
        if let Some(table) = self.table.upgrade() {
            let mut table = table.lock().unwrap_or_else(PoisonError::into_inner);
            if table
                .entries
                .get(&self.token)
                .is_some_and(|entry| !entry.in_flight)
            {
                if let Some(entry) = table.entries.remove(&self.token) {
                    table.release(&entry.request_id);
                }
            }
        }
    }
}

// Not derived: the token is a credential.
impl fmt::Debug for Placement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Placement").finish_non_exhaustive()
    }
}

/// What [`ReplicaRouter::infer_placed`] returns.
#[derive(Debug)]
pub struct PlacedAnswer {
    pub events: Vec<ModelEvent>,
    /// The placement to resume on; none when the run ended with [`ModelEvent::Finished`]
    /// (nothing is left to resume, so no slot is taken).
    pub placement: Option<Placement>,
}

/// SHA-256 of the rendering of a semantic context: what binds a placement to its context.
fn context_digest(context: &SemanticContext) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let rendered = format!("{context:?}");
    let mut hasher = Sha256::new();
    hasher.update(b"ptr-replica-context-v1");
    hasher.update((rendered.len() as u64).to_le_bytes());
    hasher.update(rendered.as_bytes());
    hasher.finalize().into()
}

static NEXT_ROUTER: AtomicU64 = AtomicU64::new(1);

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
    /// This router's identity, so a placement of another router is recognised.
    instance: u64,
    placements: Arc<Mutex<Placements>>,
    /// Two independently keyed hashers and a counter make invocation tokens unique and
    /// unguessable.
    token_keys: (
        std::collections::hash_map::RandomState,
        std::collections::hash_map::RandomState,
    ),
    next_token: AtomicU64,
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
            instance: NEXT_ROUTER.fetch_add(1, Ordering::Relaxed),
            placements: Arc::new(Mutex::new(Placements::default())),
            token_keys: (
                std::collections::hash_map::RandomState::new(),
                std::collections::hash_map::RandomState::new(),
            ),
            next_token: AtomicU64::new(0),
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

    /// Run `call` from `first` (the rotation's replica when none) on at most `max_failover`
    /// more; returns the value and the index of the replica that produced it.
    fn route<T>(
        &self,
        first: Option<usize>,
        max_failover: u8,
        call: impl Fn(&dyn ResumableInferenceBackend) -> Result<T, ModelError>,
    ) -> Result<(T, usize), ModelError> {
        let start = first.unwrap_or_else(|| self.next.fetch_add(1, Ordering::Relaxed))
            % self.replicas.len();
        let mut last_error = ModelError::backend_failure();
        for step in 0..=usize::from(max_failover) {
            let index = (start + step) % self.replicas.len();
            let replica = &self.replicas[index];
            replica.counters.attempts.fetch_add(1, Ordering::Relaxed);
            if step > 0 {
                replica.counters.failovers.fetch_add(1, Ordering::Relaxed);
            }
            // A replica that panics is a failed replica, not a failed caller: the panic becomes a
            // backend failure, which fails over and is counted like any other.
            let outcome = catch_unwind(AssertUnwindSafe(|| call(replica.backend.as_ref())))
                .unwrap_or_else(|_| Err(ModelError::backend_failure()));
            match outcome {
                Ok(value) => return Ok((value, index)),
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
        self.route(None, self.policy.max_failover, |backend| {
            backend.infer(request)
        })
        .map(|(events, _)| events)
    }
}

impl ReplicaRouter {
    fn table(&self) -> std::sync::MutexGuard<'_, Placements> {
        self.placements
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn issue_token(&self) -> u128 {
        use std::hash::BuildHasher;
        let counter = self.next_token.fetch_add(1, Ordering::Relaxed);
        (u128::from(self.token_keys.0.hash_one(counter)) << 64)
            | u128::from(self.token_keys.1.hash_one(counter))
    }

    fn placement(&self, token: u128) -> Placement {
        Placement {
            token,
            router: self.instance,
            table: Arc::downgrade(&self.placements),
        }
    }

    /// Like `infer`, and also returns the [`Placement`] of the replica that answered. The
    /// router remembers the replica, the request id and the context under an unguessable
    /// token in a table of at most [`MAX_OPEN_PLACEMENTS`]; a full table refuses the call
    /// with [`ModelError::overloaded`] before any replica runs. Each principal
    /// (`options.principal`; `None` is one shared anonymous bucket) may hold at most
    /// [`MAX_PLACEMENTS_PER_PRINCIPAL`] live invocations, reservations of calls still running
    /// included; a principal at its quota is refused the same way and nothing of another's is
    /// taken.
    ///
    /// A request id is **live** from the call until its run finishes, fails or its placement is
    /// dropped (a resume in flight included). A second `infer_placed` under a live id is refused
    /// with [`ModelError::duplicate_request`] before any replica runs: the router forwards the
    /// caller's id unchanged, and a stateful backend could not tell the two invocations apart.
    /// This is checked router-wide, which is stricter than per replica but needs no knowledge of
    /// where failover would land.
    ///
    /// # Errors
    /// As `infer`, [`ModelError::duplicate_request`] for a live id and
    /// [`ModelError::overloaded`] for a full table or a principal at its quota.
    pub fn infer_placed(
        &self,
        request: &ModelRequest,
        options: &PlacementOptions,
    ) -> Result<PlacedAnswer, ModelError> {
        {
            let mut table = self.table();
            // Before the capacity check and before any replica runs: a stateful backend cannot
            // tell two live invocations under one id apart, so the second is refused.
            if table.live.contains_key(&request.request_id) {
                return Err(ModelError::duplicate_request());
            }
            // The principal's quota, before the router-wide bound: a principal at its limit is
            // refused whatever the others hold, and nothing of theirs is taken.
            if table.held.get(&options.principal).copied().unwrap_or(0)
                >= MAX_PLACEMENTS_PER_PRINCIPAL
            {
                return Err(ModelError::overloaded());
            }
            if table.entries.len() + table.reserved >= MAX_OPEN_PLACEMENTS {
                return Err(ModelError::overloaded());
            }
            table.reserved += 1;
            table.claim(request.request_id.clone(), options.principal.clone());
        }
        let routed = self.route(None, self.policy.max_failover, |backend| {
            backend.infer(request)
        });
        let mut table = self.table();
        table.reserved = table.reserved.saturating_sub(1);
        let (events, replica) = match routed {
            Ok(answered) => answered,
            Err(error) => {
                table.release(&request.request_id);
                return Err(error);
            }
        };
        if matches!(events.last(), Some(ModelEvent::Finished)) {
            table.release(&request.request_id);
            return Ok(PlacedAnswer {
                events,
                placement: None,
            });
        }
        let token = self.issue_token();
        table.entries.insert(
            token,
            OpenPlacement {
                replica,
                request_id: request.request_id.clone(),
                principal: options.principal.clone(),
                context: context_digest(&request.semantic_context),
                in_flight: false,
            },
        );
        drop(table);
        Ok(PlacedAnswer {
            events,
            placement: Some(self.placement(token)),
        })
    }

    /// Resume on the replica `placement` names, once. The placement must be this router's, open
    /// and unused, and `request` must carry the request id and the semantic context of the
    /// invocation it was issued for; anything else is [`ModelError::scope_mismatch`] and no
    /// replica runs. Under [`ResumeFailover::Never`] that replica is tried once; under
    /// [`ResumeFailover::Interchangeable`] it is tried first and the others follow like for
    /// `infer`. The answer carries the [`Placement`] of the replica that served it, for the next
    /// round; the slot is held for it while the replica runs.
    ///
    /// # Errors
    /// As `resume`, and [`ModelError::scope_mismatch`].
    pub fn resume_placed(
        &self,
        placement: Placement,
        request: &ModelResumeRequest,
    ) -> Result<PlacedAnswer, ModelError> {
        let replica = {
            let mut table = self.table();
            let usable = placement.router == self.instance
                && table.entries.get(&placement.token).is_some_and(|entry| {
                    !entry.in_flight
                        && entry.request_id == request.request_id
                        && entry.context == context_digest(&request.semantic_context)
                });
            if !usable {
                return Err(ModelError::scope_mismatch());
            }
            let Some(entry) = table.entries.get_mut(&placement.token) else {
                return Err(ModelError::scope_mismatch());
            };
            entry.in_flight = true;
            entry.replica
        };
        let max_failover = match self.policy.resume {
            ResumeFailover::Never => 0,
            ResumeFailover::Interchangeable => self.policy.max_failover,
        };
        let routed = self.route(Some(replica), max_failover, |backend| {
            backend.resume(request)
        });
        let mut table = self.table();
        let Some(mut entry) = table.entries.remove(&placement.token) else {
            return Err(ModelError::scope_mismatch());
        };
        // The id stays live only if the run goes on.
        table.release(&entry.request_id);
        let (events, served) = routed?;
        // A run that finished has nothing left to resume: the entry goes and no placement is made.
        if matches!(events.last(), Some(ModelEvent::Finished)) {
            return Ok(PlacedAnswer {
                events,
                placement: None,
            });
        }
        // The same slot, under a fresh token, on the replica that served this round.
        let token = self.issue_token();
        entry.replica = served;
        entry.in_flight = false;
        table.claim(entry.request_id.clone(), entry.principal.clone());
        table.entries.insert(token, entry);
        drop(table);
        Ok(PlacedAnswer {
            events,
            placement: Some(self.placement(token)),
        })
    }
}

impl ResumableInferenceBackend for ReplicaRouter {
    /// A resume without a [`Placement`]: the rotation's replica, with failover only under
    /// [`ResumeFailover::Interchangeable`].
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        let max_failover = match self.policy.resume {
            ResumeFailover::Never => 0,
            ResumeFailover::Interchangeable => self.policy.max_failover,
        };
        self.route(None, max_failover, |backend| backend.resume(request))
            .map(|(events, _)| events)
    }
}
