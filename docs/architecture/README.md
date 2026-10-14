# PTR Architecture Documentation

## Current overview

PTR's root workspace is 33 library crates and four binaries, placed in eight ranked layers by
[`research/catalogs/crate-layers.toml`](../../research/catalogs/crate-layers.toml) and checked in CI
([41](41-crate-layering.md)):

| Rank | Layer | Crates |
|---:|---|---|
| 0 | kernel | `ptr-types` |
| 1 | foundation | `ptr-analytics`, `ptr-config`, `ptr-core`, `ptr-events`, `ptr-exec`, `ptr-ingress`, `ptr-inspect`, `ptr-ledger`, `ptr-lineage`, `ptr-memory`, `ptr-model-api`, `ptr-observe`, `ptr-protocol`, `ptr-security`, `ptr-semdb`, `ptr-storage` |
| 2 | domain | `ptr-labeling`, `ptr-net`, `ptr-search`, `ptr-verifier` |
| 3 | composition | `ptr-branch`, `ptr-cluster`, `ptr-fastmem`, `ptr-feedback`, `ptr-pods` |
| 4 | projection | `ptr-router`, `ptr-state`, `ptr-worker` |
| 5 | orchestration | `ptr-pg`, `ptr-runtime` |
| 6 | surface | `ptr-execwire`, `ptr-podwire`, `ptr-server` |
| 7 | binary | `ptr-bench`, `ptrctl`, `ptrd` |

The diagrams that match the code are in the root [README](../../README.md#architecture): the
crate-layer dependency graph, the request path `ptrd` → `ptr-server` → `ptr-runtime` (ingest
into SemDB, ledger commit, backend, Pod loop with verifier, effect boundary with a journaled
execution audit) and the verifier-closed VCW-MoE loop. In short:

- **Authority.** `ptr-runtime` appends every change to a `FileLedger` (or an in-memory ledger)
  before it is materialized; SemDB deltas carry a recorded origin; effects are fenced by
  `EffectAttempted`/`EffectSettled`. Raft (`ptr-ledger` features, `ptr-cluster`) is a library
  composition that `ptrd` does not run.
- **Request path.** `ptrd` runs standalone on loopback with deterministic plumbing backends or
  the CPU KV-reuse reference; it registers no Pods. `ptr-core` and `ptr-ingress` are not on the
  path.
- **VCW-MoE.** `POST /v1/vcw/tasks` (off by default) runs `VcwLoop` with simulated or `qwen-cpu`
  experts and the deterministic verifiers of `ptr-verifier`; acceptance only through
  `ptr_router::classify`, commit only through `admit_pod_output`.
- **Wires.** Three ALPNs (`ptr-raft/1`, `ptr-exec/1`, `ptr-podwire/1`) over `ptr-net`, each with
  a hand-written bounded frame format.

Where a numbered document below and the code disagree, the code wins; the generated
[component status](../components/STATUS.md) is the inventory.

## Start here

1. [Root README](../../README.md) — what exists, diagrams, how to run
2. [Technical architecture](../TECHNICAL_ARCHITECTURE.md)
3. [Global invariants](../INVARIANTS.md) and [component contracts](../COMPONENT_CONTRACTS.md)
4. [Technology stack](../TECH_STACK.md), [Rust module and API style](../RUST_API_STYLE.md),
   [testing layout](../TESTING.md), [development environment](../DEVELOPMENT_ENVIRONMENT.md)

## Numbered documents

Status vocabulary: **design** (describes intent; little or none of it is built), **partially
implemented** (named parts exist in code, the rest is design), **implemented** (the contract the
document states exists in code with tests; limits the document names still apply), **process**
(a working method that is in force rather than code).

### Vision and planes (written for the scaffold; mostly design)

| Doc | Status | What is true today |
|---|---|---|
| [00 System](00-system.md) | design | The plane diagram is the target; the built request path is in the root README. |
| [01 PTR Core](01-ptr-core.md) | design | `ptr-core` holds data structures only (`research-scaffold`); the trainable model is `model/burn-a0`, not connected to the runtime. |
| [02 SemDB](02-semdb.md) | partially implemented | Revisions, snapshots and dependency invalidation exist (S001); stale-run cancellation only in `SteppedRun` (S002); no memoized query engine. |
| [03 Runtime](03-runtime.md) | design | `ptr-exec` has a bounded `Mailbox` (R001) and nothing else of the isolate model; `ptr-server` serialises requests behind one runtime mutex. |
| [04 Pods](04-pods.md) | partially implemented | Manifests, project-scoped registry, leases, native and remote (PodWire) Pods exist; name-randomised training does not. |
| [05 Memory and search](05-memory-search.md) | design | No engine adapters (Tantivy, Zvec, …); `ptr-search` has evidence promotion (Q002); `ptr-pg` serves dense and lexical search in PostgreSQL. |
| [06 Consensus and ledger](06-consensus-ledger.md) | partially implemented | `FileLedger` is the runtime's log; raft-engine and raft-rs are feature-gated backends not used by the runtime; Turso projection behind a feature. |
| [07 Network and protocol](07-network-protocol.md) | partially implemented | Three ALPNs with hand-written frames over Iroh; no protobuf on a wire, no rkyv or DLPack. |
| [08 Verifier and feedback](08-verifier-feedback.md) | partially implemented | Verifier fabric, deterministic verifiers, `RepairLoop` (F001); no population selection or trajectory archive. |
| [09 Training](09-training.md) | design | `ptr_training.run` writes run manifests (`dry-run` backend); `trl-moe` is verified only as a CPU pipeline smoke. |
| [10 Observability](10-observability.md) | design | `ptr-observe` has field names, `FlowSignature` and a `tracing` sink adapter; no exporters. |
| [11 Security](11-security.md) | partially implemented | Effect boundary, scoped execution, audit and peer admission exist; context-trust model and signed receipts do not. |
| [12 Evaluation](12-evaluation.md) | process | 32 technology slots in `evaluations/registry.toml`, all `open`. |
| [13 Data model](13-data-model.md) | implemented | The listed identifiers and lifecycle types are in `ptr-types`. |
| [14 Request lifecycle](14-request-lifecycle.md) | design | The built flow is the root README's sequence diagram. |
| [15 Deployment topologies](15-deployment-topologies.md) | design | `ptrd` refuses any mode but `standalone` and any non-loopback bind. |
| [16 Failure and consistency](16-failure-consistency.md) | partially implemented | Failpoints in `ptr-ledger`, crash and raft recovery schedules (L001-v2, L002). |
| [17 Type system](17-type-system.md) | partially implemented | Shared axes, codebook and validity masks in `ptr-types`; section 12 lists the gaps. |
| [18 Performance](18-performance.md) | design | Guidelines; no performance budget is measured end to end. |
| [19 Research method](19-research-method.md) | process | Enforced by `scripts/run_experiment.py`, `scripts/check_research_gates.py` and `AGENTS.md` §3. |

### P0 contracts: authority, persistence and trust boundaries

| Doc | Status | Owner in code |
|---|---|---|
| [20 Cognitive contract, step 01](20-cognitive-contract-step-01.md) | implemented | `ptr-types` cognitive axes (first increment, not a model) |
| [21 Scoped execution](21-scoped-execution.md) | implemented | `ptr-runtime::execution` |
| [22 Durable semantic state](22-durable-semantic-state.md) | implemented | `ptr-semdb`, `ptr-ledger`, `ptr-runtime::semantic`, `ptr-state` |
| [23 Persistence integrity](23-persistence-integrity.md) | implemented | `ptr-ledger::{integrity, FileLedger}`, `ptr-runtime::persistence` |
| [24 Protected anchors](24-protected-anchors.md) | implemented | `ptr-ledger::{anchor, acknowledged}` |
| [25 Erasure and retention](25-erasure-and-retention.md) | implemented | `ptr-ledger::retention` (no claim that history is erased) |
| [26 Cognitive codebook](26-cognitive-codebook.md) | implemented | `ptr-types::{codebook, validity_mask}` |
| [27 Neural-state admission](27-neural-state-admission.md) | implemented | `ptr-runtime::neural`, `ptrctl seal` |
| [28 Durable execution audit](28-durable-execution-audit.md) | implemented | `ptr-runtime::execution`, ledger tags 9–11 |
| [29 Peer admission and Pod scope](29-peer-admission-and-pod-scope.md) | implemented | `ptr-runtime` admission policy, `ptr-pods` registry |
| [30 Third-party notices](30-third-party-notices.md) | implemented | `scripts/check_notices.py`, `scripts/check_vendor_retirement.py`, `THIRD-PARTY-NOTICES.md` |

### Cluster and wires

| Doc | Status | Notes |
|---|---|---|
| [31 Cluster integrity](31-cluster-integrity.md) | partially implemented | Durable raft storage, fencing token, retention-floor bound (C7), three-member groups in tests, `ptr-cluster` wire and snapshot transfer; no deployment loop, `ptrd` does not run it. |
| [32 Execution wire](32-execution-wire.md) | implemented | Request/receipt over `ptr-exec/1` (`ptr-execwire`, feature `execwire-backend`); receipts unsigned. |
| [33 Pod wire](33-pod-wire.md) | implemented | Request/answer over `ptr-podwire/1` (`ptr-podwire`, feature `podwire-backend`); answers unsigned. |
| [34 Address authority](34-address-authority.md) | implemented | `ptr-net` address authority; discovery deliberately not built. |

### Agentic substrate, KV reuse and MoE

| Doc | Status | Notes |
|---|---|---|
| [35 Agentic substrate](35-agentic-substrate.md) | partially implemented | Prototypes of `ptr-pg`, `ptr-branch`, `ptr-fastmem`, `ptr-lineage`, `ptr-labeling`, `ptr-analytics`; claims beyond tests are experiments (S003, F002, F003, L003, L004, Q003-v3 completed; M008 falsified; R004, E005 planned). |
| [36 KV reuse](36-kv-reuse.md) | partially implemented | CPU reference control plane in `ptr-pods` and `ptr-runtime` (R005–R007, R009, R010, R012, E007); no engine integration. |
| [37 Pod-MoE](37-pod-moe.md) | design, slices implemented | Expert authority (R011) and attestation (R013) in `ptr-runtime`; training programme planned. |
| [38 Consciousness indicators](38-consciousness-indicators.md) | design | Functional indicators only; C001 NOT_SHOWN, C004 COMPLETED, C007 and C008 FALSIFIED (CPU toys). Welfare rules: [MODEL_WELFARE](../MODEL_WELFARE.md). |
| [39 VCW-MoE](39-vcw-moe.md) | design, loop implemented | `VcwLoop`, finding router, workspace, server path and `make vcw-demo` run with simulated (or `qwen-cpu`) experts; F007 COMPLETED as a mechanism check; no model-backed experiment yet. |
| [41 Crate layering](41-crate-layering.md) | implemented | `scripts/check_crate_layering.py` in CI. |
| [44 Serving and promotion](44-serving-and-promotion.md) | partially implemented | Adapter promotion and revocation as ledger events (tags 39, 40), a promotion refused without a serving baseline, the weights digest recomputed, the improvement-cycle budget counted from the ledger (tag 41), a replay backtest, and a bounded adaptive batcher with a replica router that `ptrd` does not use yet; F008 planned, the batching experiment has no id yet, R004 planned. |

### Pod placement and data plane

| Doc | Status | Notes |
|---|---|---|
| [40 Pod racks and trays](40-pod-rack.md) | design | The NVL72 composition pattern (scale-up domain, tray as the unit of placement and fencing, a host authority per tray, typed interconnect classes) applied to Pod placement; proposed types, work order and mechanism experiments; nothing implemented. |
| [42 Pod data plane](42-pod-data-plane.md) | design | A fenced zero-copy fabric (tensor handles admitted like frames, a closed carrier set) in five stages: one host several processes, several GPUs, RAM offload, NVMe offload, several hosts; experiment ids E009 to E013 reserved; nothing implemented. |

## Related

- [Component index](../components/README.md) and [generated status](../components/STATUS.md)
- [Diagram index](../diagrams/README.md) (older target-architecture diagrams)
- [Open architecture catalog](../../research/catalogs/README.md) — unresolved type families,
  backend slots and component contracts
- [MoE research agenda](../../research/agenda/MOE_RESEARCH_AGENDA.md) and
  [GPU plan](../VCW_MOE_GPU_PLAN.md)
