# ptr-storage — Artifact & Object Storage

> **Role:** Provides typed, content-addressed artifact references while abstracting physical storage locations.  
> **Maturity:** architecture + contract scaffold; production behavior must be proven by the linked experiments and component evaluations.

## Position in PTR

```mermaid
flowchart LR
    A["Bytes"] --> B["ptr-storage\nArtifact & Object Storage"]
    B --> C["Content-addressed ArtifactRef"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-storage.mmd`](../../docs/diagrams/components/ptr-storage.mmd)

**Upstream:** ptr-ingress, ptr-pods  
**Downstream:** ptr-memory, ptr-search, ptr-net

## Mission

Provides typed, content-addressed artifact references while abstracting physical storage locations.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- ArtifactId/ArtifactRef
- object-store abstraction
- content hash identity
- location registry

## Explicit non-responsibilities

- semantic validity
- search ranking
- distributed consensus

## Data flow

| Direction | Contract |
|---|---|
| Input | Raw bytes and typed artifact metadata |
| Output | Stable content identities and retrievable blobs |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- OpenDAL candidate for backend abstraction
- BLAKE3 content identity
- local/S3/object-store backends

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Content identity is independent of location.
2. Raw evidence can be retained even when normalized views change.
3. Secret/private artifact access is policy-gated.

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

- [object-storage](../../evaluations/components/object-storage/README.md)

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

