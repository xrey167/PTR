# ptr-security — Capability, Trust & Effect Security

> **Role:** Enforces the hard shell around uncertain reasoning: capabilities, permissions, trust levels, effects, secrets and sandbox requirements.  
> **Maturity:** architecture + contract scaffold; production behavior must be proven by the linked experiments and component evaluations.

## Position in PTR

```mermaid
flowchart LR
    A["ActionIR + policy"] --> B["ptr-security\nCapability, Trust & Effect Security"]
    B --> C["Allow / deny / require verification"]
    B -. "contracts" .-> T["ptr-types"]
    B -. "telemetry" .-> O["ptr-observe"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-security.mmd`](../../docs/diagrams/components/ptr-security.mmd)

**Upstream:** ptr-core, ptr-router, ptr-ingress  
**Downstream:** ptr-exec, ptr-ledger

## Mission

Enforces the hard shell around uncertain reasoning: capabilities, permissions, trust levels, effects, secrets and sandbox requirements.

PTR keeps this responsibility in its own crate so the semantics remain stable even when an external library or implementation is replaced.

## Responsibilities

- capability checks
- permission policy
- effect authorization
- context trust classification
- secret wrappers/redaction policy
- sandbox policy

## Explicit non-responsibilities

- model uncertainty
- consensus ordering
- low-level sandbox implementation

## Data flow

| Direction | Contract |
|---|---|
| Input | ActionIR, principal/session policy, capabilities, context trust and resource scope |
| Output | Authorized action, denial or escalation requirement |
| Failure | Explicit typed error / rejected state; no silent fallback that changes semantics |
| Observability | Standard PTR tracing fields and a stable component span |

## Technical approach

- Rust typestate/newtypes
- Brin-style context-security signals as evidence only
- ROCK/process sandbox integration

External projects are **candidates**, not architectural authority. The PTR-owned traits and domain types must remain usable with a replacement backend.

## Core invariants

1. Mutation/External/Irreversible effects require explicit authority.
2. Untrusted context cannot grant itself capabilities.
3. A learned model cannot override a hard deny.

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

- [security-context](../../evaluations/components/security-context/README.md)

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

