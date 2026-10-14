//! An adaptive micro-batcher in front of an inference backend.
//!
//! [`AdaptiveBatcher`] implements [`InferenceBackend`], so it composes where a backend does.
//! Callers block in `infer` while one worker thread collects requests that may share a batch
//! (equal [`BatchKey`]), waits at most the policy's `max_wait` for more, and runs them through
//! [`BatchInferenceBackend::infer_batch`] in one call. The rule that picks batches is
//! `batch_queue` and is deterministic; this module adds the clock, the thread and the counters.
//!
//! **What it guarantees.**
//! * The queue is bounded ([`BatchPolicy::queue_capacity`]); a request that finds it full gets
//!   [`ModelError::overloaded`] at once (Invariant 8). Nothing grows past the bound.
//! * A request whose deadline has passed, or that was cancelled, is answered without a model
//!   call, whether it is refused at the door or found stale in the queue.
//! * Requests with different keys (adapter, principal, generation) never share a batch.
//! * The [`BatchGate`] is asked about the key of every batch immediately before the model runs,
//!   so a generation revoked while a request waited is refused at that use and not only when
//!   it was queued (Invariant 3).
//! * Every caller gets exactly one answer, including when the batcher is dropped or the backend
//!   panics.
//!
//! **What it does not do.** It cannot make a backend faster by itself: a backend whose
//! `infer_batch` is the default sequential loop gains nothing but queueing. It does not retry,
//! and it does not choose adapters; the key is whatever the caller says.
//!
//! The plain [`InferenceBackend::infer`] path carries no adapter, principal or deadline, so it
//! uses [`BatchKey::unscoped`] and no deadline; the scoped path is
//! [`AdaptiveBatcher::infer_with`].

use crate::batch_queue::{BatchQueue, Dropped, Entry, QueueFull};
use crate::{
    InferenceBackend, ModelError, ModelEvent, ModelObservation, ModelRequest, ModelResumeRequest,
    ResumableInferenceBackend,
};
use ptr_types::{Generation, RequestId, SemanticContext, Timestamp};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The most requests one batch may hold.
pub const MAX_BATCH_LIMIT: usize = 1024;

/// The most requests a batcher may queue.
pub const MAX_QUEUE_CAPACITY: usize = 65_536;

/// The longest a request may be held back for company, in milliseconds.
pub const MAX_WAIT_LIMIT_MILLIS: u64 = 10_000;

/// What requests must share to run in one batch: the same project, adapter, principal and
/// generation. The project is part of the key because adapter names are only unique within a
/// project: two projects' adapters of one name must never share a batch or a gate answer.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BatchKey {
    pub project: Option<String>,
    pub adapter: Option<String>,
    pub principal: Option<String>,
    pub generation: Generation,
}

impl BatchKey {
    /// The key of a request that names no adapter and no principal.
    pub fn unscoped() -> Self {
        Self {
            project: None,
            adapter: None,
            principal: None,
            generation: Generation(0),
        }
    }
}

/// How a batcher forms batches. Check it with [`BatchPolicy::validate`]; the batcher does so
/// again when it is built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchPolicy {
    /// The most requests in one model call.
    pub max_batch: usize,
    /// The longest the oldest request of a group waits for more company.
    pub max_wait_millis: u64,
    /// The most requests queued at once.
    pub queue_capacity: usize,
    /// A group is dispatched once its tightest deadline is this close.
    pub deadline_margin_millis: u64,
}

impl BatchPolicy {
    /// # Errors
    /// [`BatchError::InvalidPolicy`] naming the first field outside its range.
    pub fn validate(&self) -> Result<(), BatchError> {
        let invalid = |field, message| Err(BatchError::InvalidPolicy { field, message });
        if self.max_batch == 0 || self.max_batch > MAX_BATCH_LIMIT {
            return invalid("max_batch", "must be between 1 and MAX_BATCH_LIMIT");
        }
        if self.queue_capacity == 0 || self.queue_capacity > MAX_QUEUE_CAPACITY {
            return invalid("queue_capacity", "must be between 1 and MAX_QUEUE_CAPACITY");
        }
        if self.max_wait_millis > MAX_WAIT_LIMIT_MILLIS {
            return invalid("max_wait_millis", "must not exceed MAX_WAIT_LIMIT_MILLIS");
        }
        if self.deadline_margin_millis > MAX_WAIT_LIMIT_MILLIS {
            return invalid(
                "deadline_margin_millis",
                "must not exceed MAX_WAIT_LIMIT_MILLIS",
            );
        }
        Ok(())
    }
}

/// Why a batcher could not be built.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchError {
    InvalidPolicy {
        field: &'static str,
        message: &'static str,
    },
    /// The worker thread could not be started.
    SpawnFailed(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy { field, message } => write!(formatter, "{field}: {message}"),
            Self::SpawnFailed(reason) => write!(formatter, "batch worker not started: {reason}"),
        }
    }
}

impl std::error::Error for BatchError {}

/// The handle of one continuation, returned with its answer by
/// [`AdaptiveBatcher::infer_scoped`]. It is opaque: not `Clone`, not `Copy`, no public
/// constructor, and it is consumed by [`AdaptiveBatcher::resume_continuation`]. It holds only an
/// unguessable 128-bit id. Everything a resume needs (scope, request id, semantic context, the
/// round) is kept in the batcher's bounded table under that id, so nothing a caller holds or
/// writes decides any of it.
pub struct Continuation {
    id: u128,
}

// Not derived: the id is a credential and a log line must not carry it.
impl fmt::Debug for Continuation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Continuation")
            .finish_non_exhaustive()
    }
}

/// What [`AdaptiveBatcher::infer_scoped`] returns: the events and the continuation that a
/// later observation resumes.
#[derive(Debug)]
pub struct ScopedAnswer {
    pub events: Vec<ModelEvent>,
    pub continuation: Continuation,
}

/// What [`AdaptiveBatcher::resume_continuation`] returns: the events and, when the round
/// counter has room, the continuation of the next round.
#[derive(Debug)]
pub struct ResumedAnswer {
    pub events: Vec<ModelEvent>,
    pub next: Option<Continuation>,
}

/// The most continuations a batcher keeps open at once. A flood of answers that are never
/// resumed evicts the oldest of the heaviest principal (see
/// [`MAX_CONTINUATIONS_PER_PRINCIPAL`]); resuming an evicted continuation fails closed.
pub const MAX_OPEN_CONTINUATIONS: usize = 4096;

/// The whole resume context of one open continuation. The resume request is built from this
/// and the caller's observation, never from caller-supplied context.
struct OpenContinuation {
    request_id: RequestId,
    key: BatchKey,
    semantic_context: SemanticContext,
    round: u32,
}

/// The most continuations one principal may hold open at once. At the quota a principal's
/// next continuation evicts that principal's own oldest, so one principal cannot push another's
/// out.
pub const MAX_CONTINUATIONS_PER_PRINCIPAL: usize = 256;

type PrincipalKey = Option<String>;

/// Open continuations: a global bound ([`MAX_OPEN_CONTINUATIONS`]) and a per-principal quota
/// ([`MAX_CONTINUATIONS_PER_PRINCIPAL`]), each principal with its own FIFO. When the table is
/// full, the oldest continuation of the principal that holds the most goes; a principal under
/// its quota is displaced only when it is itself the largest holder. Every operation is
/// logarithmic in the number of principals (a ranking by count) and at most linear in one
/// principal's quota.
#[derive(Default)]
struct Continuations {
    entries: std::collections::HashMap<u128, OpenContinuation>,
    queues: std::collections::HashMap<PrincipalKey, std::collections::VecDeque<u128>>,
    /// (open count, principal) for every principal with at least one open continuation.
    ranking: std::collections::BTreeSet<(usize, PrincipalKey)>,
}

impl Continuations {
    fn evict_oldest_of(&mut self, principal: &PrincipalKey) {
        let Some(queue) = self.queues.get(principal) else {
            return;
        };
        if let Some(oldest) = queue.front().copied() {
            self.close(oldest);
        }
    }

    fn open(&mut self, id: u128, continuation: OpenContinuation) {
        let principal = continuation.key.principal.clone();
        let held = self.queues.get(&principal).map_or(0, |queue| queue.len());
        if held >= MAX_CONTINUATIONS_PER_PRINCIPAL {
            self.evict_oldest_of(&principal);
        }
        while self.entries.len() >= MAX_OPEN_CONTINUATIONS {
            let Some((_, heaviest)) = self.ranking.iter().next_back().cloned() else {
                break;
            };
            self.evict_oldest_of(&heaviest);
        }
        let queue = self.queues.entry(principal.clone()).or_default();
        let before = queue.len();
        queue.push_back(id);
        let after = queue.len();
        if before > 0 {
            self.ranking.remove(&(before, principal.clone()));
        }
        self.ranking.insert((after, principal));
        self.entries.insert(id, continuation);
    }

    fn close(&mut self, id: u128) -> Option<OpenContinuation> {
        let continuation = self.entries.remove(&id)?;
        let principal = continuation.key.principal.clone();
        if let Some(queue) = self.queues.get_mut(&principal) {
            let before = queue.len();
            queue.retain(|open| *open != id);
            let after = queue.len();
            self.ranking.remove(&(before, principal.clone()));
            if after == 0 {
                self.queues.remove(&principal);
            } else {
                self.ranking.insert((after, principal));
            }
        }
        Some(continuation)
    }
}

/// A flag a caller sets to withdraw a request that has not reached the model yet.
#[derive(Clone, Debug, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// What the caller says about one request besides its content.
#[derive(Clone, Debug)]
pub struct BatchAdmission {
    pub key: BatchKey,
    /// When the answer stops being useful, on the batcher's [`BatchClock`] (Unix milliseconds
    /// for [`SystemClock`]). A request past it is refused before the model.
    pub deadline: Option<Timestamp>,
    pub cancel: Option<CancelFlag>,
}

impl BatchAdmission {
    pub fn unscoped() -> Self {
        Self {
            key: BatchKey::unscoped(),
            deadline: None,
            cancel: None,
        }
    }
}

/// Where the batcher reads the time, in milliseconds.
pub trait BatchClock: Send + Sync {
    fn now(&self) -> Timestamp;
}

/// Unix milliseconds. If the system clock cannot be read it answers `u64::MAX`, so every
/// deadline looks passed and nothing with a deadline is served on a time nobody knows.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl BatchClock for SystemClock {
    fn now(&self) -> Timestamp {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(elapsed) => Timestamp(u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)),
            Err(_) => Timestamp(u64::MAX),
        }
    }
}

/// Whether a batch under `key` may still run. Asked immediately before every model call; the
/// host answers it from its lifecycle authority. A key that names an adapter is never served
/// without a gate: a batcher built without one refuses it.
pub trait BatchGate: Send + Sync {
    fn admits(&self, key: &BatchKey) -> bool;
}

/// A backend that can run several requests in one call.
///
/// The provided method runs them one after another, which is correct and gains nothing; a
/// backend that batches overrides it. It returns one result per request, in order; a different
/// count fails every request of the batch.
pub trait BatchInferenceBackend: InferenceBackend {
    fn infer_batch(&self, requests: &[&ModelRequest]) -> Vec<Result<Vec<ModelEvent>, ModelError>> {
        requests.iter().map(|request| self.infer(request)).collect()
    }
}

impl BatchInferenceBackend for crate::ReferenceEchoBackend {}
impl BatchInferenceBackend for crate::ScenarioBackend {}

/// Counters of one batcher since it was built.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchStats {
    /// Requests that entered the queue.
    pub accepted: u64,
    /// Requests refused because the queue was full.
    pub overloaded: u64,
    /// Requests refused or dropped because their deadline had passed.
    pub deadline_drops: u64,
    pub cancelled: u64,
    /// Requests refused because the gate no longer admitted their key.
    pub revoked: u64,
    pub batches: u64,
    /// Requests that were handed to the backend.
    pub dispatched: u64,
    pub largest_batch: u64,
    /// Requests whose backend call failed or panicked.
    pub backend_errors: u64,
}

#[derive(Default)]
struct Counters {
    accepted: AtomicU64,
    overloaded: AtomicU64,
    deadline_drops: AtomicU64,
    cancelled: AtomicU64,
    revoked: AtomicU64,
    batches: AtomicU64,
    dispatched: AtomicU64,
    largest_batch: AtomicU64,
    backend_errors: AtomicU64,
}

fn bump(counter: &AtomicU64, by: usize) {
    counter.fetch_add(u64::try_from(by).unwrap_or(u64::MAX), Ordering::Relaxed);
}

type Reply = Result<Vec<ModelEvent>, ModelError>;

struct Job {
    request: ModelRequest,
    reply: SyncSender<Reply>,
}

// Not derived: a request carries prompt text, which a log line must not.
impl fmt::Debug for Job {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Job")
            .field("request_id", &self.request.request_id)
            .field("raw_text_len", &self.request.raw_text.len())
            .finish_non_exhaustive()
    }
}

struct State {
    queue: BatchQueue<Job>,
    closed: bool,
    continuations: Continuations,
}

struct Shared<B> {
    backend: Arc<B>,
    clock: Arc<dyn BatchClock>,
    gate: Option<Arc<dyn BatchGate>>,
    state: Mutex<State>,
    wake: Condvar,
    counters: Counters,
    /// Two independently keyed hashers (the standard library seeds each from the operating
    /// system) and a counter: a continuation id is unique by the counter and unguessable by the
    /// keys.
    token_keys: (
        std::collections::hash_map::RandomState,
        std::collections::hash_map::RandomState,
    ),
    next_token: AtomicU64,
}

impl<B> Shared<B> {
    fn issue_token(&self) -> u128 {
        use std::hash::BuildHasher;
        let counter = self.next_token.fetch_add(1, Ordering::Relaxed);
        let high = self.token_keys.0.hash_one(counter);
        let low = self.token_keys.1.hash_one(counter);
        (u128::from(high) << 64) | u128::from(low)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn answer(entry: Entry<Job>, reply: Reply) {
    // The caller may have gone; the answer is then dropped.
    let _ = entry.payload.reply.send(reply);
}

/// A batching front for backend `B`; see the module documentation.
pub struct AdaptiveBatcher<B: BatchInferenceBackend + 'static> {
    shared: Arc<Shared<B>>,
    worker: Option<JoinHandle<()>>,
}

impl<B: BatchInferenceBackend + 'static> fmt::Debug for AdaptiveBatcher<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdaptiveBatcher")
            .field("backend", &self.shared.backend.name())
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl<B: BatchInferenceBackend + 'static> AdaptiveBatcher<B> {
    /// A batcher on the system clock with no gate: it serves only keys that name no adapter and
    /// refuses a key that does ([`ModelError::revoked`]), because nothing vouches for it.
    ///
    /// # Errors
    /// As [`AdaptiveBatcher::with_parts`].
    pub fn new(backend: Arc<B>, policy: BatchPolicy) -> Result<Self, BatchError> {
        Self::with_parts(backend, policy, Arc::new(SystemClock), None)
    }

    /// # Errors
    /// [`BatchError::InvalidPolicy`] and [`BatchError::SpawnFailed`].
    pub fn with_parts(
        backend: Arc<B>,
        policy: BatchPolicy,
        clock: Arc<dyn BatchClock>,
        gate: Option<Arc<dyn BatchGate>>,
    ) -> Result<Self, BatchError> {
        policy.validate()?;
        let shared = Arc::new(Shared {
            backend,
            clock,
            gate,
            state: Mutex::new(State {
                queue: BatchQueue::new(policy),
                closed: false,
                continuations: Continuations::default(),
            }),
            wake: Condvar::new(),
            counters: Counters::default(),
            token_keys: (
                std::collections::hash_map::RandomState::new(),
                std::collections::hash_map::RandomState::new(),
            ),
            next_token: AtomicU64::new(0),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("ptr-batcher".into())
            .spawn(move || {
                // Whatever ends the worker, nobody may be left waiting on a queue it no longer
                // serves: the batcher closes and every queued caller is answered.
                let _ = catch_unwind(AssertUnwindSafe(|| run_worker(&worker_shared)));
                close_and_drain(&worker_shared);
            })
            .map_err(|error| BatchError::SpawnFailed(error.to_string()))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    /// Run one request under `admission` and wait for its answer.
    ///
    /// # Errors
    /// [`ModelError::overloaded`] when the queue is full,
    /// [`ModelError::deadline_exceeded`], [`ModelError::cancelled`],
    /// [`ModelError::revoked`] when the gate no longer admits the key, [`ModelError::closed`]
    /// when the batcher is shutting down, and whatever the backend returns.
    pub fn infer_with(
        &self,
        request: &ModelRequest,
        admission: BatchAdmission,
    ) -> Result<Vec<ModelEvent>, ModelError> {
        self.run_inferred(request, admission)
    }

    /// Like [`AdaptiveBatcher::infer_with`], and also opens a continuation: the returned
    /// [`Continuation`] is the one handle a later observation resumes through
    /// ([`AdaptiveBatcher::resume_continuation`]). The batcher keeps the scope (the admission's
    /// key), the request id, the semantic context and the round in a table of at most
    /// [`MAX_OPEN_CONTINUATIONS`] entries. `infer` and `infer_with` open none.
    ///
    /// # Errors
    /// As [`AdaptiveBatcher::infer_with`].
    pub fn infer_scoped(
        &self,
        request: &ModelRequest,
        admission: BatchAdmission,
    ) -> Result<ScopedAnswer, ModelError> {
        let key = admission.key.clone();
        let events = self.run_inferred(request, admission)?;
        let id = self.shared.issue_token();
        self.shared.lock().continuations.open(
            id,
            OpenContinuation {
                request_id: request.request_id.clone(),
                key,
                semantic_context: request.semantic_context.clone(),
                round: 1,
            },
        );
        Ok(ScopedAnswer {
            events,
            continuation: Continuation { id },
        })
    }

    fn run_inferred(
        &self,
        request: &ModelRequest,
        admission: BatchAdmission,
    ) -> Result<Vec<ModelEvent>, ModelError> {
        let counters = &self.shared.counters;
        if admission.key.adapter.is_some() && self.shared.gate.is_none() {
            bump(&counters.revoked, 1);
            return Err(ModelError::revoked());
        }
        let now = self.shared.clock.now().0;
        let deadline = admission.deadline.map(|deadline| deadline.0);
        if deadline.is_some_and(|deadline| deadline <= now) {
            bump(&counters.deadline_drops, 1);
            return Err(ModelError::deadline_exceeded());
        }
        if admission
            .cancel
            .as_ref()
            .is_some_and(CancelFlag::is_cancelled)
        {
            bump(&counters.cancelled, 1);
            return Err(ModelError::cancelled());
        }
        let (reply, answer_rx) = sync_channel(1);
        {
            let mut state = self.shared.lock();
            if state.closed {
                return Err(ModelError::closed());
            }
            let job = Job {
                request: request.clone(),
                reply,
            };
            match state
                .queue
                .push(admission.key, deadline, admission.cancel, job, now)
            {
                Ok(()) => bump(&counters.accepted, 1),
                Err(QueueFull) => {
                    bump(&counters.overloaded, 1);
                    return Err(ModelError::overloaded());
                }
            }
        }
        self.shared.wake.notify_one();
        answer_rx
            .recv()
            .unwrap_or_else(|_| Err(ModelError::closed()))
    }

    /// Resume `continuation` with a new observation, once. The observation is new input of the
    /// caller under the continuation's scope: the gate checks exactly that scope, as it does
    /// for a fresh `infer` under it, and the batcher does not and cannot see where the
    /// caller's bytes came from. What it guarantees is that nothing stale is resumed. The
    /// request id, semantic context and round come from the table entry of this continuation
    /// and from nowhere else; the entry belongs to exactly one scope and is consumed here, so
    /// a continuation is single-use and one from a scope since revoked is refused by the gate.
    ///
    /// `key` is the scope the caller claims. It must be the entry's; any difference, an
    /// unknown, evicted or used continuation alike, is [`ModelError::scope_mismatch`] and
    /// nothing runs (the handle is consumed either way). On success the answer carries the
    /// continuation of the next round.
    ///
    /// # Errors
    /// [`ModelError::scope_mismatch`], [`ModelError::revoked`] when the gate does not admit the
    /// entry's key (or a key naming an adapter has no gate), and whatever the backend returns.
    pub fn resume_continuation(
        &self,
        continuation: Continuation,
        key: &BatchKey,
        observation: ModelObservation,
    ) -> Result<ResumedAnswer, ModelError>
    where
        B: ResumableInferenceBackend,
    {
        let entry = self.shared.lock().continuations.close(continuation.id);
        let Some(entry) = entry.filter(|entry| &entry.key == key) else {
            return Err(ModelError::scope_mismatch());
        };
        let request = ModelResumeRequest {
            request_id: entry.request_id.clone(),
            semantic_context: entry.semantic_context.clone(),
            round: entry.round,
            observation,
        };
        let events = self.run_resume(&request, &entry.key)?;
        let next = entry.round.checked_add(1).map(|round| {
            let id = self.shared.issue_token();
            self.shared
                .lock()
                .continuations
                .open(id, OpenContinuation { round, ..entry });
            Continuation { id }
        });
        Ok(ResumedAnswer { events, next })
    }

    fn run_resume(
        &self,
        request: &ModelResumeRequest,
        key: &BatchKey,
    ) -> Result<Vec<ModelEvent>, ModelError>
    where
        B: ResumableInferenceBackend,
    {
        if !gate_admits(&self.shared, key) {
            bump(&self.shared.counters.revoked, 1);
            return Err(ModelError::revoked());
        }
        // The same containment as a batch: a backend that panics fails this call only.
        match catch_unwind(AssertUnwindSafe(|| self.shared.backend.resume(request))) {
            Ok(result) => {
                if result.is_err() {
                    bump(&self.shared.counters.backend_errors, 1);
                }
                result
            }
            Err(_) => {
                bump(&self.shared.counters.backend_errors, 1);
                Err(ModelError::backend_failure())
            }
        }
    }

    /// The counters since the batcher was built.
    pub fn stats(&self) -> BatchStats {
        let counters = &self.shared.counters;
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        BatchStats {
            accepted: load(&counters.accepted),
            overloaded: load(&counters.overloaded),
            deadline_drops: load(&counters.deadline_drops),
            cancelled: load(&counters.cancelled),
            revoked: load(&counters.revoked),
            batches: load(&counters.batches),
            dispatched: load(&counters.dispatched),
            largest_batch: load(&counters.largest_batch),
            backend_errors: load(&counters.backend_errors),
        }
    }

    /// Requests waiting in the queue now.
    pub fn queued(&self) -> usize {
        self.shared.lock().queue.len()
    }
}

impl<B: BatchInferenceBackend + 'static> Drop for AdaptiveBatcher<B> {
    fn drop(&mut self) {
        self.shared.lock().closed = true;
        self.shared.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            // A worker that panicked has nothing left to join.
            let _ = worker.join();
        }
    }
}

impl<B: BatchInferenceBackend + 'static> InferenceBackend for AdaptiveBatcher<B> {
    fn name(&self) -> &'static str {
        "adaptive-batcher"
    }

    fn infer(&self, request: &ModelRequest) -> Result<Vec<ModelEvent>, ModelError> {
        self.infer_with(request, BatchAdmission::unscoped())
    }
}

impl<B: BatchInferenceBackend + ResumableInferenceBackend + 'static> ResumableInferenceBackend
    for AdaptiveBatcher<B>
{
    /// A resume without a continuation. A request id says nothing about the scope it was
    /// admitted under, so this form is only for a batcher built without a gate (which serves no
    /// adapter at all), where every answer is unscoped. With a gate configured it is refused
    /// with [`ModelError::revoked`]; a caller that serves adapters resumes through
    /// [`AdaptiveBatcher::resume_continuation`] with the continuation of the answer.
    fn resume(&self, request: &ModelResumeRequest) -> Result<Vec<ModelEvent>, ModelError> {
        if self.shared.gate.is_some() {
            bump(&self.shared.counters.revoked, 1);
            return Err(ModelError::revoked());
        }
        self.run_resume(request, &BatchKey::unscoped())
    }
}

/// Whether `key` may run now. A gate that panics has not admitted it: the answer is a refusal,
/// counted with the other revocations, and the worker and the caller carry on.
fn gate_admits<B>(shared: &Shared<B>, key: &BatchKey) -> bool {
    match &shared.gate {
        Some(gate) => catch_unwind(AssertUnwindSafe(|| gate.admits(key))).unwrap_or(false),
        None => key.adapter.is_none(),
    }
}

/// Close the batcher and answer everything still queued. Run when the worker ends, however it
/// ends, so a caller never waits on a queue nothing serves.
fn close_and_drain<B>(shared: &Shared<B>) {
    let mut state = shared.lock();
    state.closed = true;
    for entry in state.queue.drain() {
        answer(entry, Err(ModelError::closed()));
    }
}

fn run_worker<B: BatchInferenceBackend + 'static>(shared: &Shared<B>) {
    loop {
        let ready = {
            let mut state = shared.lock();
            loop {
                if state.closed {
                    for entry in state.queue.drain() {
                        answer(entry, Err(ModelError::closed()));
                    }
                    return;
                }
                let now = shared.clock.now().0;
                let ready = state.queue.take_ready(now);
                if ready.batch.is_some() || !ready.dropped.is_empty() {
                    break ready;
                }
                state = match ready.wake_at {
                    None => shared
                        .wake
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner),
                    Some(at) => {
                        let pause = Duration::from_millis(at.saturating_sub(now).max(1));
                        shared
                            .wake
                            .wait_timeout(state, pause)
                            .unwrap_or_else(PoisonError::into_inner)
                            .0
                    }
                };
            }
        };
        for (why, entry) in ready.dropped {
            match why {
                Dropped::DeadlinePassed => {
                    bump(&shared.counters.deadline_drops, 1);
                    answer(entry, Err(ModelError::deadline_exceeded()));
                }
                Dropped::Cancelled => {
                    bump(&shared.counters.cancelled, 1);
                    answer(entry, Err(ModelError::cancelled()));
                }
            }
        }
        if let Some(batch) = ready.batch {
            run_batch(shared, batch);
        }
    }
}

fn run_batch<B: BatchInferenceBackend + 'static>(shared: &Shared<B>, batch: Vec<Entry<Job>>) {
    let counters = &shared.counters;
    // Asked for every batch, at the moment of use. Without a gate only a key that names no
    // adapter can have been queued; anything else is refused here as well.
    let admitted = batch
        .first()
        .is_some_and(|first| gate_admits(shared, &first.key));
    if !admitted {
        bump(&counters.revoked, batch.len());
        for entry in batch {
            answer(entry, Err(ModelError::revoked()));
        }
        return;
    }
    // The gate may have taken a while: look again at deadlines and cancellations, so nothing
    // stale reaches the model.
    let now = shared.clock.now().0;
    let mut live = Vec::with_capacity(batch.len());
    for entry in batch {
        match entry.stale_at(now) {
            Some(Dropped::DeadlinePassed) => {
                bump(&counters.deadline_drops, 1);
                answer(entry, Err(ModelError::deadline_exceeded()));
            }
            Some(Dropped::Cancelled) => {
                bump(&counters.cancelled, 1);
                answer(entry, Err(ModelError::cancelled()));
            }
            None => live.push(entry),
        }
    }
    let batch = live;
    if batch.is_empty() {
        return;
    }
    bump(&counters.batches, 1);
    bump(&counters.dispatched, batch.len());
    counters.largest_batch.fetch_max(
        u64::try_from(batch.len()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    let requests: Vec<&ModelRequest> = batch.iter().map(|entry| &entry.payload.request).collect();
    let outcome = catch_unwind(AssertUnwindSafe(|| shared.backend.infer_batch(&requests)));
    let results = match outcome {
        Ok(results) if results.len() == batch.len() => results,
        Ok(_) | Err(_) => {
            bump(&counters.backend_errors, batch.len());
            for entry in batch {
                answer(entry, Err(ModelError::backend_failure()));
            }
            return;
        }
    };
    for (entry, result) in batch.into_iter().zip(results) {
        if result.is_err() {
            bump(&counters.backend_errors, 1);
        }
        answer(entry, result);
    }
}
