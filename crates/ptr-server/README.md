# ptr-server — Daemon & Public Control Plane

> **Role:** Exposes PTR to clients through stable APIs without leaking internal crate boundaries or provider-specific interfaces.  
> **Maturity:** see the generated status below; production behavior must be proven by the linked experiments and component evaluations.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-10-10  
**Code footprint:** 9 Rust source files · 3412 nonblank source lines · 8 integration-test files · 72 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- KV path hardening: KvBackend::with_max_prompt_tokens refuses a text that tokenizes beyond the limit with 413 (ApiError::PromptTooLong) before any runtime call, so nothing reaches the ledger; KvTaintRegistryFull is 507 Insufficient Storage without Retry-After (the registry never drains); the KV path answers with an x-ptr-kv-hit-tokens header (KV_HIT_TOKENS_HEADER); serve_with_backend serves either backend and ptrd wires the reference-kv name (CPU reference, seeded random weights, byte text) through it; tests/reference_kv.rs starts ptrd and sees a second request with a shared prefix report x-ptr-kv-hit-tokens > 0
- ServerBackend is either Resumable(Arc<dyn ResumableInferenceBackend>) (the existing path, unchanged: with_dependencies/router_with_dependencies/serve_with_dependencies delegate to it) or KvReusing(KvBackend), a KvReusingBackend with a Tokenizer and optional KvReuseHint (with_backend, router_with_backend, serve_with_backend). On the KV path the request text becomes one Raw PromptSegment of a SegmentedModelRequest and the ModelRun reaches the request flow only through PtrRuntime::run_model_run_resumable, i.e. ingest_model_run (invariant marked KV-INGEST-INVARIANT in lib.rs): an Approximate run's action is refused by authorize_action before any session or EffectAttempted, answered 409 Conflict (not 422: the refusal depends on the server's KV-reuse state, not the request content, and no edit of the request fixes it); a full taint registry is 503; Exact or no reuse gives responses identical to the plain ScenarioBackend path. The KV path does not resume Pods (a run that requests one is 422). Covered by tests/kv_taint.rs with a test double
- Versioned JSON ApiRequest/ApiResponse and HealthResponse contracts
- Axum router with GET /health and POST /v1/requests
- Reference runtime/backend request path returns revisioned response
- POST /v1/requests runs run_resumable_with_pods_using_router against an injected ResumableInferenceBackend, `Arc<dyn PodSampler>` (built with checked_pod_router(SamplingKey, instance), a PodRouter policy sampled through route_checked), PodRegistry and typed-payload Verifier (router_with_dependencies, serve_with_dependencies); ServerState::new and router() keep the ReferenceEchoBackend, an empty registry and default_verifier(), and ApiResponse gains verified_evidence (type:length per observation)
- A mutating action returned by that run is executed once through the runtime's session path (authorize_action, register_execution_session, prepare_execution_once, execute_prepared): ApiRequest gains an optional idempotency_key that mutations require (400 without it), and ApiResponse.effect returns an EffectReceipt with the ledger indices of the EffectAttempted and EffectSettled records and the SHA-256 of the output; a repeated key replays the receipt
- Effect failures map to HTTP: a key bound to another action, a fenced runtime or an ambiguous outcome is 409, other execution errors and a missing grant are 422, and a payload that is not UTF-8 or exceeds 4 KiB is 400
- demo_effect_grants installs a single demo.local-note.create grant (Effect::Mutation, Deterministic verification) whose DemoNoteExecutor replaces effects/demo-note.txt under a data directory atomically (temporary file named from process id, clock and a counter and retried on a name collision, sync_all, rename, then a directory fsync on Unix, and on Windows, where a directory cannot be opened as a file, a flush of the replaced note's own handle to push its metadata out; the Windows path was verified in CI rather than locally, and what a flush guarantees about a rename after a power loss is the file system's, not this crate's), so every distinct idempotency key can write and the latest note wins; the default state installs no grants, so effects there are refused
- default_verifier() (default_verifier_with(PayloadBounds) for other limits) is PayloadBoundsVerifier (non-empty type id without edge whitespace, at most MAX_POD_OUTPUT_BYTES = 64 KiB, UTF-8 for configured textual type ids; hard findings type-id, payload-size, payload-utf8), replacing the allow-all verifier
- demo_effect_grants verifies the action with ptr-verifier's ActionContractVerifier (action-contract) built from the DEMO_* constants, replacing the allow-all action verifier; DemoNoteExecutor's own check and the handler's 400 precheck remain as defence in depth
- Bounded request admission (Invariant #8): RequestAdmission (atomic in-flight counter, compare_exchange; new() returns Result<_, AdmissionError> and refuses capacity 0) sized from runtime.mailbox_capacity; the slot is taken right after the cheap 400 checks and the locked pipeline runs in tokio spawn_blocking with the slot guard moved in, so it is held until the blocking work ends even if the request future is cancelled and async workers stay free for /health; a request that finds it full is refused with 503 Service Unavailable, Retry-After: 1 and {"error":"request mailbox saturated"}, and the slot is released on drop
- The verifier-closed workspace loop as a new, config-selectable request path (src/vcw/, package VS): POST /v1/vcw/tasks exists only on a ServerState given a VcwService (ServerState::with_vcw, built by VcwService::from_config from the [vcw] section of ptr-config, default off; router_from_state and serve_state serve it); every existing route, type and signature is unchanged and the existing tests pass untouched, with a test that /health and /v1/requests answer byte for byte alike with and without it. A request is {id, task:{kind: arithmetic|json_object|code|fact, ...}} (unknown fields refused; a body limit of 256 KiB, field bounds from the verifiers answered 413, a blank or unserved input 400, a reused id 409, the runtime mutex and the existing RequestAdmission slot as for /v1/requests, 503 with Retry-After when saturated). The server commits the task text as raw evidence, builds the real ptr-verifier verifier of the task (arithmetic, JSON Schema subset, code tests, facts against the runtime snapshot), the RoleRoster from RepairRoleAssignments over the code table (one Pod id per role from config), and runs one VcwLoop per served kind with Guarded experts over the runtime own eligibility view (RuntimeStanding). Outcome is typed: Verified {answer, authors, rounds, verifier_calls, pod_calls, trace_digest, committed range} answered 200 only after admit_verified_answer admitted it (the runtime re-verifies with the same task verifier over the state as it is then; liveness, project and payload checks first), Exhausted {reason} and Escalated {reason} answered 422 without any candidate, an admission refusal 422 with no answer. An accepted answer is committed as a Candidate output (PodOutputAdmitted plus a PodCandidate semantic record), not promoted; the trace digest is bound into the execution manifest of that admission (origins) and emitted as RuntimeEvent::VcwFinished for every outcome. Round 2: an enabled section must name its backend (no default; ptr-config vcw.experts.backend is an Option); the backend is marked in every admission (manifest origins vcw:backend:<name>, plus the four model-file digests for qwen-cpu), in the capabilities of the served Pods (backend:<name>, so the Pod manifest digest differs by backend) and as `backend` in Verified, Exhausted and Escalated responses; a per-request wall-clock deadline (max_request_ms, default 30000, at most 120000, checked before each Pod call) ends a request Exhausted with reason Deadline; every request after its raw text is committed records one VcwFinished (outcome failed for internal errors) and its id is consumed (retry needs a new id); install refuses a Pod id that is a capsule of another project (PodIdCollision), a configured generation different from the live generation of a known Pod (GenerationMismatch) and a scope of an earlier incarnation with a non-default lease binding or children (ScopeNotEmpty), and ptrd builds and installs the service before it binds the listener. Experts: simulated (seeded programmatic experts that plant and clear the finding codes of the code table at configured rates, drawing clean candidates from the request reference, which never reaches a verifier) or qwen-cpu (cargo feature qwen-cpu; arithmetic and json_object; model directory and four SHA-256 digests required, a mismatch refuses to start). Covered by tests/vcw.rs (arithmetic wrong then repaired with ledger and event checks, JSON schema and code repair including the resource role, unrepairable task exhausted, wrong reference never passes, unknown fact never accepted, contradicted fact repaired against committed state, a revoked proposer and a revoked repairer never called, disabled has no route and old routes byte-identical, typed 4xx, id used once, saturation 503, determinism for a fixed seed, build-time refusals)
- Shadow learning of the vcw path (src/vcw/shadow.rs, [vcw] learning = "shadow"; package VL): each served kind attaches a ptr_router::RoutingLearner to its loop (VcwLoop::with_learner), freezes a LearnedPolicy candidate after 256 settled verdicts and, from the decisions logged after the freeze, logs the doubly robust value difference of the candidate over the uniform router and its one-sided 95% lower bound as RuntimeEvent::VcwShadowEstimate at geometrically growing log sizes (first at 128 decisions, log capped at 16384); VcwService::learning_report() returns the counts, the candidate digest and the latest estimate per kind (empty with learning off). The loop's scorer stays UniformScorer: nothing learned routes, nothing is promoted, and there is no configuration that promotes

### Missing for the target architecture

- Pod resumption on the KV path; the taint registry is in memory and not rebuilt after a restart (see ptr-runtime)
- Streaming ModelEvent/WebSocket or SSE API
- Authentication/session/policy integration
- Readiness/admin endpoints beyond basic health
- Cancellation propagation (backpressure is bounded admission only; there is no queue, priority or per-client fairness)
- The vcw path holds the runtime mutex for the whole run, bounded by max_request_ms plus the step in flight (a slow model serialises requests); the fact kind is an oracle for committed state (200 versus 422 tells a client whether a guessed claim equals fact/<subject>/<attribute>), acceptable on loopback only; hint_mode apply-to-schema lets client traffic change the shared per-kind self-model and attention schema, so routing influence crosses clients (default trace-only); simulated answers are client-authored (drawn from the request reference) and marked as such; every distinct request id commits raw text and a candidate with no quota or authentication; no streaming, no per-client fairness, no authentication; facts need committed semantic state (an empty runtime answers Unknown); the code and fact kinds have no prompts for the model backend; the qwen-cpu feature is not compiled by CI
- Shadow learning has one Pod per role in the shipped roster, so the candidate cannot differ from uniform there; the estimates become informative only with a roster that has several repair Pods per code. Promotion of a learned policy is not reachable from the server

### Next milestones

- Add streaming ModelEvent SSE/WebSocket endpoint and cancellation propagation
- Replace ReferenceEchoBackend route with configured evaluated backend
- Add authentication/session/policy and per-client fairness

### Linked experiments

- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`
- [E003](../../experiments/system/E003-token-efficiency/README.md) — `planned`

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0001-rust-runtime.md](../../research/decisions/ADR-0001-rust-runtime.md)
- [ADR-0004-backend-independence.md](../../research/decisions/ADR-0004-backend-independence.md)

### Current automated checks

- tests/hardening.rs: grant_verifier_refuses_an_action_the_handler_precheck_admits (422, no effects directory) and a_full_mailbox_is_refused_with_503_and_recovers_when_released (mailbox_capacity 1, gated backend, 503 with Retry-After, backend call count unchanged, A then C succeed) and saturation_is_explicit_with_more_requests_than_workers (2 workers, capacity 4, 6 concurrent requests: exactly 2 get 503 with Retry-After, /health answers 200 while saturated, the 4 admitted succeed)
- tests/http.rs: oversized_id_text_and_body_are_refused_before_the_runtime_sees_them
- scenario_backend_executes_demo_note_once_and_replays_receipt: ScenarioBackend writes the note once, a missing idempotency key is 400, a repeated key returns the same attempt/settlement indices, and the same key with a different request is 409; distinct_idempotency_keys_each_execute_and_malformed_keys_are_rejected_up_front: a second key executes and request id and text are bounded (MAX_REQUEST_ID_BYTES 256, MAX_REQUEST_TEXT_BYTES 1 MiB, 400) and the body is capped at MAX_REQUEST_BODY_BYTES 2 MiB (413), stated here rather than inherited from the framework default; a blank key or one with edge whitespace is 400 before the model run, judged by PtrRuntime::is_usable_execution_key, the rule prepare_execution_once itself applies (a well-formed identifier within MAX_KEY_BYTES, or a longer one an earlier build already settled), so a new over-long key is refused before inference while a retry under a long key that was settled earlier is still answered from its record. The receipt is the most recent settled attempt for the key paired with its own settlement. Whether a request ends in a mutation, and so needs a key, is only known after the model run
- durable_restart_fences_open_effect_until_explicit_reconciliation: over a durable ledger holding an unsettled EffectAttempted, the same key is 409 after reopen and succeeds only after reconcile_effect
- tests: every router_with_dependencies caller passes checked_pod_router with a fixed test key; the unit test checks that checked_pod_router reports the key's identifier (ServerState::new, whose registry is empty, uses a sampler that routes nothing)
- Axum health/request/bad-request HTTP integration tests matching TypeScript SDK contract
- public ApiRequest smoke test
- workspace fmt/check/test/clippy
- tests/vcw.rs: the verifier-closed path with simulated experts (see implemented); tests/vcw_qwen.rs: one ignored test of the qwen-cpu backend (env PTR_QWEN_MODEL_DIR, cargo feature qwen-cpu)
- tests/vcw.rs shadow_learning_learns_and_logs_estimates_but_never_changes_what_is_served: 160 requests under learning off and learning shadow return byte for byte equal responses, off reports nothing, shadow settled more than 256 verdicts, froze a candidate and logged estimates as runtime events, with one Pod per role the estimated gain is zero, and no VcwPolicyChanged event exists; shadow without an exploration floor is refused at build time and by validate

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
    A["Client"] --> B["ptr-server\nDaemon & Public Control Plane"]
    B --> C["PTR ingress/runtime"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-server.mmd`](../../docs/diagrams/components/ptr-server.mmd)

**Upstream:** external clients  
**Downstream:** ptr-ingress, ptr-exec, ptr-observe

## Mission

Exposes PTR to clients through stable APIs without leaking internal crate boundaries or provider-specific interfaces.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- HTTP/WebSocket API
- request/session boundary
- health/readiness endpoints
- streaming response adaptation
- admin/control endpoints

## Explicit non-responsibilities

- core reasoning semantics
- search logic
- distributed consensus

## Data flow

| Direction | Contract |
|---|---|
| Input | Client requests and admin commands |
| Output | API responses and model/runtime event streams |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- Axum primary server framework
- Pingora optional edge/gateway layer
- JSON public API plus binary/streaming options

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Public API versions are explicit.
2. Authentication/authorization is delegated to hard security policy.
3. Backpressure propagates to clients instead of unbounded buffering.

These invariants should be executable wherever possible through unit, property, lifecycle or chaos tests.

## Failure model

The component must fail closed for semantic or effect-safety violations. Infrastructure failures should surface as typed errors that preserve request, revision, generation and provenance context. Retries must be idempotent whenever the operation may cross a process or network boundary.

## Performance model

Measure before optimizing. Benchmarks should record at least latency distribution, throughput, allocations/resident memory, queue depth or working-set size where relevant, and the cost of verification. Performance optimizations may not bypass generation, revision, capability or evidence checks.

## Security and privacy

- Treat external inputs and backend outputs as untrusted until validated.
- Do not put raw secrets/private evidence into generic tracing or inspection.
- Preserve provenance on every promotion from raw/possible evidence to stronger semantic state.
- External effects pass through `ptr-security` even if this component already performed local validation.

## Technology evaluation

- No dedicated technology slot yet; architectural alternatives should be added to `evaluations/components/` before lock-in.

A new candidate should be added with a reproducible benchmark and failure-semantics analysis rather than replacing the default ad hoc.

## Tests required before production use

- Contract/unit tests for all domain transitions.
- Invalid, stale-generation and stale-revision cases.
- Cancellation/retry behavior.
- Property tests for invariants where practical.
- Cross-backend equivalence if more than one backend exists.
- Observability and redaction checks.

## Related architecture

- [System architecture](../../docs/architecture/00-system.md)
- [Technical architecture](../../docs/TECHNICAL_ARCHITECTURE.md)
- [Component contracts](../../docs/COMPONENT_CONTRACTS.md)
- [Global invariants](../../docs/INVARIANTS.md)

