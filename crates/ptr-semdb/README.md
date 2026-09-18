# ptr-semdb — Incremental Semantic Database

> **Role:** Maintains revisioned ground state and dependency-aware derived semantics using an incremental-compiler model.  
> **Maturity:** architecture + contract scaffold; production behavior must be proven by the linked experiments and component evaluations.

{BEGIN}
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml). Run `python3 scripts/update_component_docs.py --write` after editing metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-18

### Implemented now

- SemanticDelta upsert/removal model
- Dependency graph and transitive affected-closure calculation
- Revisioned SemanticHost and immutable Arc-backed SemanticSnapshot
- Stale snapshot detection
- ProjectSkeleton scaffold
- Unit test proving local dependency invalidation

### Missing for the target architecture

- Typed ground/derived query keys instead of String→String state
- Memoized derived-query engine with dependency capture
- Cancellation tokens for stale in-flight computation
- Concurrent snapshot/read architecture and persistent/rebuildable caches
- Semantic skeleton derivation from typed state

### Next milestones

- Replace string map with typed key/value/query interfaces
- Implement dependency-recording query execution and cache invalidation
- Run S001/S002 before selecting a third-party incremental engine

### Linked experiments

- - [S001](../../experiments/semdb/S001-invalidation/README.md) — `planned`
- - [S002](../../experiments/semdb/S002-snapshot-cancellation/README.md) — `planned`
- - [E004](../../experiments/system/E004-long-horizon/README.md) — `planned`

### Technology evaluations

- - [semantic-db](../../evaluations/components/semantic-db/README.md) — `open`

### Decision records

- - [ADR-0003-raw-and-typed.md](../../research/decisions/ADR-0003-raw-and-typed.md)
- - [ADR-0005-incremental-semdb.md](../../research/decisions/ADR-0005-incremental-semdb.md)

### Current automated checks

- local_change_invalidates_only_dependency_closure unit test
- workspace fmt/check/test/clippy

{END}

## Position in PTR

```mermaid
flowchart LR
    A["Ground State"] --> B["ptr-semdb\nIncremental Semantic Database"]
    B --> C["Immutable snapshot + dirty closure"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-semdb.mmd`](../../docs/diagrams/components/ptr-semdb.mmd)

**Upstream:** ptr-ingress, ptr-state, ptr-verifier  
**Downstream:** ptr-core, ptr-router, ptr-memory

## Mission

Maintains revisioned ground state and dependency-aware derived semantics using an incremental-compiler model.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- ground inputs
- dependency graph
- precise invalidation
- immutable snapshots
- stale-work detection and cancellation
- semantic skeletons

## Explicit non-responsibilities

- durable distributed authority
- vector retrieval
- neural reasoning

## Data flow

| Direction | Contract |
|---|---|
| Input | Validated deltas and observations |
| Output | Immutable SemanticSnapshot<Revision> and affected-query sets |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- Rust-native incremental engine
- rust-analyzer-style host/snapshot architecture
- Salsa and alternative engines remain evaluation candidates

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Reasoning runs against one immutable revision.
2. Only dependent derived values are invalidated.
3. Revision does not encode semantic-object lifecycle Generation.

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

- [semantic-db](../../evaluations/components/semantic-db/README.md)

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

