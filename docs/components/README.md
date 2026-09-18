# PTR Component Documentation

Every runtime/model crate has a local README and a dedicated Mermaid source diagram.\n\n[**Implementation status dashboard →**](STATUS.md)

| Component | Purpose | Diagram |
|---|---|---|
| [ptr-types](../../crates/ptr-types/README.md) | Defines the stable semantic vocabulary shared across model, runtime, storage, verification, and network boundaries. | [diagram](../diagrams/components/ptr-types.mmd) |
| [ptr-protocol](../../crates/ptr-protocol/README.md) | Defines PodWire semantics and converts versioned wire messages into validated PTR domain values. | [diagram](../diagrams/components/ptr-protocol.mmd) |
| [ptr-ingress](../../crates/ptr-ingress/README.md) | Preserves raw evidence while building a provisional typed interpretation that can be cross-verified before reasoning. | [diagram](../diagrams/components/ptr-ingress.mmd) |
| [ptr-semdb](../../crates/ptr-semdb/README.md) | Maintains revisioned ground state and dependency-aware derived semantics using an incremental-compiler model. | [diagram](../diagrams/components/ptr-semdb.mmd) |
| [ptr-core](../../crates/ptr-core/README.md) | Research implementation of the model architecture: raw token states plus typed semantic slots, epistemic state, latent recurrence and operator routing. | [diagram](../diagrams/components/ptr-core.mmd) |
| [ptr-model-api](../../crates/ptr-model-api/README.md) | Keeps the runtime independent of any single inference server or model implementation. | [diagram](../diagrams/components/ptr-model-api.mmd) |
| [ptr-exec](../../crates/ptr-exec/README.md) | Runs PTR subsystems as supervised state machines with bounded typed mailboxes, explicit backpressure, cancellation and replay semantics. | [diagram](../diagrams/components/ptr-exec.mmd) |
| [ptr-pods](../../crates/ptr-pods/README.md) | Defines specialist computational modules through semantic contracts rather than fragile tool names. | [diagram](../diagrams/components/ptr-pods.mmd) |
| [ptr-router](../../crates/ptr-router/README.md) | Chooses how a task should be solved: neural reasoning, search, statistics, simulation, symbolic methods, Pods or combinations. | [diagram](../diagrams/components/ptr-router.mmd) |
| [ptr-verifier](../../crates/ptr-verifier/README.md) | Combines deterministic, executable, statistical and learned checks into explicit verification reports. | [diagram](../diagrams/components/ptr-verifier.mmd) |
| [ptr-feedback](../../crates/ptr-feedback/README.md) | Implements candidate populations, critique, patch/rewrite, selection and trajectory capture for test-time and training-time improvement. | [diagram](../diagrams/components/ptr-feedback.mmd) |
| [ptr-ledger](../../crates/ptr-ledger/README.md) | Records the authoritative ordered lifecycle of semantic changes and, in cluster mode, applies distributed consensus before state materialization. | [diagram](../diagrams/components/ptr-ledger.mmd) |
| [ptr-state](../../crates/ptr-state/README.md) | Projects committed ledger events into queryable current-state views without replacing the ledger as causal authority. | [diagram](../diagrams/components/ptr-state.mmd) |
| [ptr-memory](../../crates/ptr-memory/README.md) | Stores validated semantic, episodic, procedural and epistemic memory as lifecycle-managed domain objects. | [diagram](../diagrams/components/ptr-memory.mmd) |
| [ptr-search](../../crates/ptr-search/README.md) | Routes queries across lexical, semantic, GPU, multimodal, distributed and structural search backends while keeping retrieval non-authoritative. | [diagram](../diagrams/components/ptr-search.mmd) |
| [ptr-storage](../../crates/ptr-storage/README.md) | Provides typed, content-addressed artifact references while abstracting physical storage locations. | [diagram](../diagrams/components/ptr-storage.mmd) |
| [ptr-net](../../crates/ptr-net/README.md) | Connects PTR nodes and remote Pods with authenticated transport while leaving protocol semantics to ptr-protocol. | [diagram](../diagrams/components/ptr-net.mmd) |
| [ptr-events](../../crates/ptr-events/README.md) | Projects committed and runtime events into scalable streams for analytics, materializers, telemetry and training collectors. | [diagram](../diagrams/components/ptr-events.mmd) |
| [ptr-observe](../../crates/ptr-observe/README.md) | Records structured runtime execution without requiring natural-language chain-of-thought logging. | [diagram](../diagrams/components/ptr-observe.mmd) |
| [ptr-inspect](../../crates/ptr-inspect/README.md) | Allows generic inspection and rendering of typed Rust values without collapsing the runtime into untyped JSON. | [diagram](../diagrams/components/ptr-inspect.mmd) |
| [ptr-security](../../crates/ptr-security/README.md) | Enforces the hard shell around uncertain reasoning: capabilities, permissions, trust levels, effects, secrets and sandbox requirements. | [diagram](../diagrams/components/ptr-security.mmd) |
| [ptr-server](../../crates/ptr-server/README.md) | Exposes PTR to clients through stable APIs without leaking internal crate boundaries or provider-specific interfaces. | [diagram](../diagrams/components/ptr-server.mmd) |

## Documentation rule

A new first-class component is not complete until it has:

1. a local README describing ownership and non-ownership;
2. a diagram showing its place in the data/control flow;
3. domain contracts and invariants;
4. failure and security semantics;
5. benchmark/evaluation slots for replaceable technology;
6. tests tied to those invariants.
