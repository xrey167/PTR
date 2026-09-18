# ptr-protocol — Semantic Protocol & Codec Boundary

> **Role:** Defines PodWire semantics and converts versioned wire messages into validated PTR domain values.  
> **Maturity:** architecture + contract scaffold; production behavior must be proven by the linked experiments and component evaluations.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `scaffold`  
**Last reviewed:** 2026-09-18  
**Code footprint:** 1 Rust source files · 54 nonblank source lines · 0 `#[test]` markers

### Implemented now

- PodWire v1 semantic version constant
- TypedPayload, CallFrame and CALL/RET/ERR/EVT/REVOKE domain frames
- Frame kind classification independent of transport

### Missing for the target architecture

- Prost/Protobuf adapter and generated-code isolation
- Canonical validation, semantic version negotiation and compatibility matrix
- rkyv/local codec adapter and JSON debug adapter
- Idempotency keys, ACK/CANCEL/BYE/HELLO/DECL frames from the wider PodWire design

### Next milestones

- Implement domain↔prost conversion with rejection tests
- Add canonical wire fixtures and malformed-frame corpus
- Define semantic contract version separately from wire schema version

### Linked experiments

- [R002](../../experiments/runtime/R002-unseen-pod-generalization/README.md) — `planned`
- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`

### Technology evaluations

- [network-codec](../../evaluations/components/network-codec/README.md) — `open`
- [local-serialization](../../evaluations/components/local-serialization/README.md) — `open`

### Decision records

- [ADR-0004-backend-independence.md](../../research/decisions/ADR-0004-backend-independence.md)
- [ADR-0007-model-event-stream.md](../../research/decisions/ADR-0007-model-event-stream.md)

### Current automated checks

- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
    A["Wire bytes"] --> B["ptr-protocol\nSemantic Protocol & Codec Boundary"]
    B --> C["Validated PodWire domain frame"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-protocol.mmd`](../../docs/diagrams/components/ptr-protocol.mmd)

**Upstream:** ptr-types  
**Downstream:** ptr-net, ptr-pods, ptr-server

## Mission

Defines PodWire semantics and converts versioned wire messages into validated PTR domain values.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- PodWire frame semantics
- protocol versioning and compatibility rules
- wire-to-domain validation
- codec boundaries for Protobuf/JSON/rkyv

## Explicit non-responsibilities

- network transport
- Pod execution
- permission authorization

## Data flow

| Direction | Contract |
|---|---|
| Input | Network/local wire frames |
| Output | Validated domain protocol messages |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- Prost/Protobuf candidate for network control
- Serde JSON for public/debug surfaces
- rkyv candidate for trusted local hot paths

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Successful decoding is not semantic validity.
2. Unknown protocol versions cannot silently execute.
3. Domain types never depend on generated Protobuf structs.

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

- [network-codec](../../evaluations/components/network-codec/README.md)
- [local-serialization](../../evaluations/components/local-serialization/README.md)

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

