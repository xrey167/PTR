# ptr-runtime — End-to-End Runtime Orchestrator

> **Role:** Compose PTR's semantic, model, execution, verification, security and authority layers into one request lifecycle.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-19  
**Code footprint:** 4 Rust source files · 1554 nonblank source lines · 10 integration-test files · 62 `#[test]` markers

### Implemented now

- Versioned replay-backed recovery snapshots bind complete journal, semantic revision, commit index and independently trusted SHA-256 anchor
- Snapshot validation and semantic/lifecycle replay finish before creating a durable destination; existing destinations are never replaced
- Strict anchored reopen and explicit legacy-to-v2 migration preserve history while restoring no process-local authority
- Ordered semantic journal publication before acknowledgment or model resume
- Semantic payload/dependency/revision reconstruction with schema and transition validation during replay
- Complete typed Pod bytes and source identity are revision-significant
- Fallible ingestion, optimistic base revision and canonical no-op handling
- Scoped synchronous execution gateway with opaque per-runtime sessions, exact grants, frozen ActionIR permits and registered verifier/executor binding
- Consume-time freshness, expiry, ownership and mandatory permission checks; permissions/commit/effect epochs invalidate pending permits
- Executor or ledger ambiguity fences subsequent execution and commits; process-local single-use authority is never replayed
- Ledger-only lifecycle mutation; invalid/rewinding/tombstoned/cross-project transitions rejected before append and during replay
- Central runtime object wiring configuration, SemDB, ledger, materialized state, events and permissions
- Durable standalone runtime constructor opens/replays FileLedger and reconstructs lifecycle/materialized state across restart
- Text ingestion into revisioned semantic state
- Current-revision, live-generation/revocation and capability/effect checks for ActionIR delegated through typed ptr-security authorization decisions
- Committed-event materialization and runtime event emission
- Committed-event replay rebuilds lifecycle state and preserves revocation
- Reference InferenceBackend request loop runs against a revisioned ModelRequest and emits completion event
- Typed model→PodRegistry→Pod→Verifier→SemDB observation loop for Pure/Read cognitive Pods
- Bounded multi-step model resume loop after verified Pod observations advances semantic revision before continuation

### Missing for the target architecture

- Network-authenticated/scoped Pod integration and neural checkpoint admission
- Durable execution audit/idempotency, downstream fencing and compacted materialized snapshots
- Router-driven operator selection around the implemented bounded Pod-resume loop
- Async isolate scheduler integration
- Configured raft-engine/raft-rs/Turso backend composition for production runtime modes
- Streaming client response lifecycle and cancellation

### Next milestones

- Bind opaque model/checkpoint admission to the versioned cognitive and lifecycle contracts; independent trusted anchor storage remains a deployment obligation
- Connect evaluated raft-engine/Turso adapters through typed backend config while preserving FileLedger reference mode
- Add opaque backend checkpoint handles and async streaming around the implemented observation resume contract
- Wire ptrd request handling beyond bootstrap ingestion

### Linked experiments

- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`
- [E003](../../experiments/system/E003-token-efficiency/README.md) — `planned`
- [E004](../../experiments/system/E004-long-horizon/README.md) — `planned`

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0001-rust-runtime.md](../../research/decisions/ADR-0001-rust-runtime.md)
- [ADR-0012-hard-effect-boundary.md](../../research/decisions/ADR-0012-hard-effect-boundary.md)
- [ADR-0013-runtime-orchestrator.md](../../research/decisions/ADR-0013-runtime-orchestrator.md)

### Current automated checks

- snapshot exact-state/byte-corruption/rollback/revision/legacy/overwrite/authority rejection tests
- scoped execution positive/negative integration tests and non-forgeability/single-use compile-fail doctests
- runtime ingestion/revision/generation/capability/materialization integration tests
- typed authorization decision exposure and RuntimeError compatibility test
- reference model-loop integration test
- bounded verified Pod-observation resume-loop tests
- revocation replay/restart integration test
- durable FileLedger runtime reopen preserves revocation and materialized commit position
- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
  I["Ingress / API"] --> R["ptr-runtime"]
  R --> S["SemDB"]
  R --> M["Model / Router / Pods"]
  R --> V["Verifier / Security"]
  R --> L["Ledger / State"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-runtime.mmd`](../../docs/diagrams/components/ptr-runtime.mmd)

## Mission

Keep orchestration out of `ptrd` and out of individual domain crates. The runtime coordinates transitions while each component retains ownership of its semantics.

## Current boundary

The executable reference slice now wires configuration, semantic revisioning, a backend-neutral model call, semantic Pod resolution, Pure/Read Pod execution, verifier-gated observation promotion, action authorization, event emission, ledger materialization and durable FileLedger reopen/replay. The new scoped synchronous execution gateway binds opaque sessions and immutable ActionIR permits to host-registered verifiers/executors. Administrative session registration assumes the embedding host has authenticated the principal; it is not an HTTP authentication endpoint. See [P0.1 execution authority](../../docs/architecture/21-scoped-execution.md) for guarantees, counterexamples and the remaining durable/network gates. P0.2 now reconstructs the actual journaled semantic payloads, dependencies and revisions. The next persistence gate is record integrity and verified snapshot/checkpoint admission, not additional backend breadth.

## P0.2 semantic journal integration

[Durable semantic-state contract](../../docs/architecture/22-durable-semantic-state.md)
records the new code/codec, ownership and replay boundaries. Publication follows
successful journal append. Typed Pod bytes and source identity participate in
semantic revisions. Logical removals do not erase log history; neural checkpoints
and authenticated framing remain separate gates. Execution evidence is in the PR.

## Persistence contract update

See [P0.3 checked records and replay-backed recovery snapshots](../../docs/architecture/23-persistence-integrity.md)
for strict reopen, explicit legacy migration, independent anchors, create-new
restore and format/API compatibility. Old automatic crash-tail repair is replaced
by explicit anchored recovery. No authority or neural checkpoint is restored.
