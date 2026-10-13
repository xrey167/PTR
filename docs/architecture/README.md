# PTR Architecture Documentation

## Start here

1. [System architecture](00-system.md)
2. [Technical architecture](../TECHNICAL_ARCHITECTURE.md)
3. [Component contracts](../COMPONENT_CONTRACTS.md)
4. [Global invariants](../INVARIANTS.md)
5. [Technology stack](../TECH_STACK.md)
6. [Rust module and API style](../RUST_API_STYLE.md)
7. [Testing layout](../TESTING.md)
8. [Development environment](../DEVELOPMENT_ENVIRONMENT.md)

## Detailed architecture

- [00-system.md](00-system.md)
- [01-ptr-core.md](01-ptr-core.md)
- [02-semdb.md](02-semdb.md)
- [03-runtime.md](03-runtime.md)
- [04-pods.md](04-pods.md)
- [05-memory-search.md](05-memory-search.md)
- [06-consensus-ledger.md](06-consensus-ledger.md)
- [07-network-protocol.md](07-network-protocol.md)
- [08-verifier-feedback.md](08-verifier-feedback.md)
- [09-training.md](09-training.md)
- [10-observability.md](10-observability.md)
- [11-security.md](11-security.md)
- [12-evaluation.md](12-evaluation.md)
- [13-data-model.md](13-data-model.md)
- [14-request-lifecycle.md](14-request-lifecycle.md)
- [15-deployment-topologies.md](15-deployment-topologies.md)
- [16-failure-consistency.md](16-failure-consistency.md)
- [17-type-system.md](17-type-system.md)
- [18-performance.md](18-performance.md)
- [19-research-method.md](19-research-method.md)
- [20-cognitive-contract-step-01.md](20-cognitive-contract-step-01.md)
- [21-scoped-execution.md](21-scoped-execution.md)
- [22-durable-semantic-state.md](22-durable-semantic-state.md)
- [23-persistence-integrity.md](23-persistence-integrity.md)
- [24-protected-anchors.md](24-protected-anchors.md)
- [25-erasure-and-retention.md](25-erasure-and-retention.md)
- [26-cognitive-codebook.md](26-cognitive-codebook.md)
- [27-neural-state-admission.md](27-neural-state-admission.md)
- [28-durable-execution-audit.md](28-durable-execution-audit.md)
- [29-peer-admission-and-pod-scope.md](29-peer-admission-and-pod-scope.md)
- [30-third-party-notices.md](30-third-party-notices.md)
- [31-cluster-integrity.md](31-cluster-integrity.md)
- [32-execution-wire.md](32-execution-wire.md)
- [33-pod-wire.md](33-pod-wire.md)
- [34-address-authority.md](34-address-authority.md)
- [35-agentic-substrate.md](35-agentic-substrate.md) — PostgreSQL projection substrate, certified agent branches, revocable fast-weight memory, adapter lineage and weak supervision
- [36-kv-reuse.md](36-kv-reuse.md) — KV-cache reuse: control plane in PTR, data plane in engines; exact versus approximate reuse, per-layer chain keys, LoRA and MoE sharing (design, with a CPU reference path exercised by completed mechanism checks R005-R012 and E007)
- [37-pod-moe.md](37-pod-moe.md) — Pod-MoE: experts as pods on three levels (token, request, meta), gated self-modification through proposals, verifier and ledger; training programme and verified literature (design; the R011 and R013 authority slices are implemented on the reference path)
- [38-consciousness-indicators.md](38-consciousness-indicators.md) — functional consciousness-indicator properties (after Butlin, Long et al. 2023) mapped to PTR components and as capability requirements of Pods and the Pod-MoE; principles (indicators are not consciousness, no training on self-reports), CPU/GPU split and the C-series experiment roadmap (design; CPU experiments: C001 NOT_SHOWN, C004 COMPLETED, C007 and C008 FALSIFIED); welfare rules in [`../MODEL_WELFARE.md`](../MODEL_WELFARE.md)
- [39-vcw-moe.md](39-vcw-moe.md) — VCW-MoE: the Verifier-Closed Workspace MoE; formal specification (state, loop, acceptance predicate, invariants as constraints), component and evidence map, delimitation from related work, falsification plan and risks (design; the loop runs on simulated experts in `ptr-runtime` and `ptr-server`, and F007 is COMPLETED as a mechanism check)
- [41-crate-layering.md](41-crate-layering.md) — executable crate layering: layers and backend ownership as data in `research/catalogs/crate-layers.toml`, checked in CI by `scripts/check_crate_layering.py`; shrinking exception ratchet
- [Open architecture catalog](../../research/catalogs/README.md) — unresolved type families, backend slots and component contracts

## Component-level documentation

See [component index](../components/README.md). Every Rust crate has a local README and a dedicated Mermaid source diagram.

## Diagrams

See [diagram index](../diagrams/README.md).
