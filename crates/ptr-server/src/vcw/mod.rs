//! The verifier-closed workspace loop as a request path of the server: `POST /v1/vcw/tasks`.
//!
//! ```text
//!  POST /v1/vcw/tasks  {id, task: {kind, ...}}
//!        |  bounded body (256 KiB), RequestAdmission slot (503 when saturated)
//!        v
//!  validate -> raw text committed (ingest_text) -> real verifier of the task (ptr-verifier)
//!        |
//!        v
//!  VcwLoop::run (one loop per task kind)           expert Pods: simulated | qwen-cpu
//!        |   router: RepairRoleAssignments from the         behind Guarded (call-time check)
//!        |   code table; eligibility: RuntimeStanding       over the same view as the router
//!        v
//!  Verified -> admit_verified_answer -> admit_pod_output (the runtime re-verifies) -> ledger
//!  Exhausted / Escalated -> 422, nothing committed but the request's raw text
//! ```
//!
//! * **A read-back of the last requests.** `GET /v1/vcw/traces/{id}` returns the trace of a
//!   finished request (steps, finding codes, routed Pods with support and propensity, budget
//!   spent) and a summary of the ledger events it committed (read from the ledger when the request finished; the runtime lock is never taken by a query); the last 64
//!   requests are retained in memory. A request that is still running (its trace is stored once the loop has finished, before admission) reads `outcome: "running"` and `ledger: null`; both are filled when it ends. The `bytes` and `digest` of an event come from its canonical encoding (`ptr_ledger::encoded_event`), not from a rendering of it. Observability only, no authority; `make vcw-demo` uses it.
//! * **Off by default.** The route exists only on a [`ServerState`] given a [`VcwService`]
//!   (`ServerState::with_vcw`); every other route, type and signature is unchanged.
//! * **No authority path of its own.** An accepted answer is committed by the runtime's normal
//!   output admission, as a `Candidate` output (a stored candidate, not a promoted fact). The run's
//!   trace digest is bound into the execution manifest of that admission, so the committed
//!   `PodOutputAdmitted` record's manifest digest commits to it, and it is emitted as a
//!   `RuntimeEvent::VcwFinished` for every outcome.
//! * **Never a rejected candidate.** [`VcwResponse`] has no variant that carries one, and an
//!   answer the runtime refuses is not returned.
//! * **One runtime lock, bounded in time.** A request holds the runtime mutex for its whole run,
//!   as the existing path does; a slow model therefore serialises requests, and the admission slot
//!   bounds how many wait. The hold is bounded by `max_request_ms` (default 30 s, at most 120 s)
//!   plus the step in flight: the deadline is checked before every Pod call, never inside one.
//!   Worst case with the model backend: one step is a proposal or one repair call per finding code
//!   (each at most 128 new tokens on the CPU) plus one verifier run; with the code kind the
//!   verifier run is at most 32 tests of 100 000 fuel each. After the deadline the loop is given
//!   no candidate and the request ends `exhausted` with reason `Deadline`. The budgets are capped
//!   at 64 verifier calls and 64 rounds so that the loop after the deadline ends quickly.
//! * **The backend is marked.** Simulated experts return the client's own `reference`, so a
//!   simulated answer is client-authored even though a Pod id signs it. `vcw:backend:simulated`
//!   (for the model: `vcw:backend:qwen-cpu` and the four model-file digests) is in the manifest
//!   origins of every admission, `backend:<name>` is a capability of every served Pod (so the Pod
//!   manifest digest differs by backend), and `backend` is in every response. An enabled section
//!   must name its backend; there is no default.
//! * **A request id is consumed once its raw text is committed.** Whatever follows (an answer,
//!   no answer, an internal error) a retry with the same id answers 409; send a new id for
//!   another attempt. Every such request records one `RuntimeEvent::VcwFinished`.
//! * **The fact kind is an oracle for committed state.** A client can test a guess of
//!   `fact/<subject>/<attribute>`: a verified answer (200) says the claim equals committed state,
//!   an unverified one (422) says it does not. Fine for a loopback-only daemon; do not expose the
//!   path to untrusted or multi-tenant clients without authentication and a per-client quota.
//! * **`hint_mode = "apply-to-schema"` shares state across clients.** The per-kind self-model and
//!   attention schema are learned from all requests, so one client's traffic changes routing
//!   (inside the admissible set only) for the next. The default `trace-only` keeps them
//!   observation-only. Enable the other mode only for a single-tenant deployment.

mod job;
mod pods;
mod report;
mod service;
mod shadow;
mod sim;
mod wire;

#[cfg(feature = "qwen-cpu")]
mod qwen;

pub use job::{
    answer_type, CODE_ANSWER_TYPE, FACT_ANSWER_TYPE, MAX_ENTRY_BYTES, MAX_INSTRUCTION_BYTES,
    MAX_REFERENCE_BYTES, MAX_TEST_ARGS,
};
pub use report::{MAX_REPORTED_EVENTS, MAX_RETAINED_TRACES};
pub use service::{VcwBuildError, VcwService, VcwServiceParts};
pub use shadow::{LearningReport, ShadowEstimate};
pub use sim::plantable;
pub use wire::{
    VcwApiError, VcwAuthor, VcwClaim, VcwCodeTest, VcwCommitted, VcwResponse, VcwSpent, VcwTask,
    VcwTaskRequest, VcwUnanswered, VcwVerified,
};

use axum::{
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};

use crate::ServerState;

/// The path of the endpoint.
pub const VCW_TASKS_PATH: &str = "/v1/vcw/tasks";
/// Largest accepted request body of the endpoint, in bytes.
pub const MAX_VCW_BODY_BYTES: usize = 256 * 1024;

/// The routes of the path, merged into the router only when a service is installed.
pub(crate) fn routes() -> Router<ServerState> {
    Router::new()
        .route(VCW_TASKS_PATH, post(handle))
        .route("/v1/vcw/traces/{id}", get(read_trace))
        .layer(DefaultBodyLimit::max(MAX_VCW_BODY_BYTES))
}

async fn handle(
    State(state): State<ServerState>,
    Json(input): Json<VcwTaskRequest>,
) -> Result<Response, VcwApiError> {
    let service = state
        .vcw
        .clone()
        .ok_or_else(|| VcwApiError::Internal("the vcw path is not installed".into()))?;
    // The same bounded admission as the existing path: the runtime is behind one mutex, and a
    // request that finds the mailbox full is refused instead of waiting.
    let slot = state.admission.try_enter().ok_or(VcwApiError::Saturated)?;
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let mut runtime = state
            .runtime
            .lock()
            .map_err(|_| VcwApiError::Internal("runtime mutex poisoned".into()))?;
        service
            .run_task(&mut runtime, input)
            .map(IntoResponse::into_response)
    })
    .await
    .map_err(|error| VcwApiError::Internal(format!("request task failed: {error}")))?
}

/// `GET /v1/vcw/traces/{id}`: the trace of a finished request (rounds, finding codes, the routed
/// Pods with their propensity, the budget spent) and a summary of the ledger events it committed
/// (index, kind, size, digest; never their content). 404 for an unknown request or one no longer
/// retained. A request still running reads `outcome` `"running"` and `ledger` `null` (the ledger
/// summary is stored when it ends); `bytes` is the size of the event's canonical encoding. It reads only the bounded in-memory store, never the runtime lock, so it answers while
/// a long request runs and needs no admission slot (the store's own lock is held for a clone of
/// small JSON only). Read-only; the same loopback-only caveat as the fact kind applies, because the
/// trace names finding codes.
async fn read_trace(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> Result<Response, VcwApiError> {
    let service = state
        .vcw
        .clone()
        .ok_or_else(|| VcwApiError::Internal("the vcw path is not installed".into()))?;
    match service.trace_report(&id) {
        Some(report) => Ok(Json(report).into_response()),
        None => Ok((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "no retained trace for this request id" })),
        )
            .into_response()),
    }
}
