# ptr-events — Event Distribution Plane

> **Role:** Projects committed and runtime events into scalable streams for analytics, materializers, telemetry and training collectors.  
> **Maturity:** architecture + contract scaffold; production behavior must be proven by the linked experiments and component evaluations.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `scaffold`  
**Last reviewed:** 2026-09-18  
**Code footprint:** 1 Rust source files · 16 nonblank source lines · 1 integration-test files · 1 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- RuntimeEvent taxonomy for request/snapshot/candidate/Pod/verifier/commit lifecycle
- EventEnvelope with sequence number

### Missing for the target architecture

- EventBus producer/consumer contract
- Apache Iggy adapter
- Persistent consumer offsets/replay
- Separation metadata for authoritative projections vs telemetry events
- Backpressure and retention policy

### Next milestones

- Define in-process reference EventBus
- Implement Iggy adapter behind it
- Connect trajectory/training consumer

### Linked experiments

- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`
- [F001](../../experiments/feedback/F001-generate-verify-repair/README.md) — `planned`

### Technology evaluations

- [event-streaming](../../evaluations/components/event-streaming/README.md) — `open`

### Decision records

- [ADR-0004-backend-independence.md](../../research/decisions/ADR-0004-backend-independence.md)
- [ADR-0009-consensus-ledger-state-separation.md](../../research/decisions/ADR-0009-consensus-ledger-state-separation.md)

### Current automated checks

- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
    A["Committed/runtime events"] --> B["ptr-events\nEvent Distribution Plane"]
    B --> C["Consumers"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-events.mmd`](../../docs/diagrams/components/ptr-events.mmd)

**Upstream:** ptr-ledger, ptr-observe  
**Downstream:** training collectors, analytics, materializers

## Mission

Projects committed and runtime events into scalable streams for analytics, materializers, telemetry and training collectors.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- event envelope
- stream/topic abstraction
- subscriber offsets
- projection from authoritative events

## Explicit non-responsibilities

- authoritative ordering of semantic writes
- request execution
- model reasoning

## Data flow

| Direction | Contract |
|---|---|
| Input | Committed events and non-authoritative telemetry events |
| Output | Persistent event streams for consumers |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- Apache Iggy primary candidate
- in-process reference bus
- future enterprise bridges such as RocketMQ/Kafka

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Event streaming cannot create authority.
2. Consumer replay is idempotent.
3. Authoritative and telemetry event classes remain distinguishable.

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

- [event-streaming](../../evaluations/components/event-streaming/README.md)

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

