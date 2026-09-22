# ptr-net — Node Identity & Transport

> **Role:** Connects PTR nodes and remote Pods with authenticated transport while leaving protocol semantics to ptr-protocol.  
> **Maturity:** architecture + contract scaffold; production behavior must be proven by the linked experiments and component evaluations.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-22  
**Code footprint:** 1 Rust source files · 145 nonblank source lines · 2 integration-test files · 1 `#[test]` markers

### Implemented now

- ALPN constants for raft, PodWire, model, blob, events and execution protocols
- The execution wire has its own ALPN rather than sharing the pod wire's: two protocols on one ALPN is how a request meant for one gets parsed by the other, and the parse that succeeds by accident is the dangerous one
- NodeIdentity scaffold
- Provider-independent Transport trait
- Feature-gated Iroh 1.2 direct QUIC adapter with PTR ALPNs and authenticated peer identity
- Iroh response path keeps the connection alive through peer-observed graceful completion
- Iroh 1.2 optional feature requires Rust 1.91; default runtime core retains a separate Rust 1.85 gate
- The endpoint address type is re-exported, so a composing crate names an address without declaring its own iroh dependency: two pins of a transport would be two wire formats

### Missing for the target architecture

- Peer discovery/session lifecycle
- Any traffic on ALPN_MODEL, ALPN_BLOB or ALPN_EVENTS: three of the six ALPNs are still declared and unspoken
- Retry/idempotency and backpressure behavior
- Raft/PodWire/blob stream adapters; ptr-cluster carries raft batches over ALPN_RAFT, ptr-execwire carries execution requests over ALPN_EXEC and ptr-podwire carries Pod access over ALPN_PODWIRE, all three using the request/response path, and a stream adapter would replace that rather than extend it

### Next milestones

- Extend Iroh adapter from local request/response to reusable connection/session management and Raft/PodWire channels
- Add retry/idempotency/backpressure semantics around authenticated Iroh sessions
- Exercise partition/reconnect behavior in L002

### Linked experiments

- [L002](../../experiments/lifecycle/L002-raft-recovery/README.md) — `planned`
- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`

### Technology evaluations

- [network](../../evaluations/components/network/README.md) — `open`

### Decision records

- [ADR-0004-backend-independence.md](../../research/decisions/ADR-0004-backend-independence.md)
- [ADR-0009-consensus-ledger-state-separation.md](../../research/decisions/ADR-0009-consensus-ledger-state-separation.md)

### Current automated checks

- Iroh local direct request/response roundtrip verifies authenticated peer identity and ALPN routing
- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

## Position in PTR

```mermaid
flowchart LR
    A["Node A"] --> B["ptr-net\nNode Identity & Transport"]
    B --> C["Node B"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-net.mmd`](../../docs/diagrams/components/ptr-net.mmd)

**Upstream:** ptr-protocol, ptr-storage  
**Downstream:** ptr-pods, ptr-ledger, ptr-events

## Mission

Connects PTR nodes and remote Pods with authenticated transport while leaving protocol semantics to ptr-protocol.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- NodeId/identity mapping
- transport sessions
- ALPN allocation
- cluster peer connectivity
- remote channel lifecycle

## Explicit non-responsibilities

- PodWire semantics
- Raft state machine
- event semantics

## Data flow

| Direction | Contract |
|---|---|
| Input | Validated protocol frames and blobs |
| Output | Authenticated streams/datagrams and peer events |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- Iroh/QUIC primary candidate
- separate ALPNs for raft, podwire, blobs, model and events

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Peer identity is cryptographically bound to the session.
2. Transport retry cannot duplicate authoritative effects without higher-layer idempotency.
3. Consensus semantics never depend on gossip alone.

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

- [network](../../evaluations/components/network/README.md)

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

