# System Architecture

PTR is split into semantic, cognitive, execution, authority and learning planes rather than treating the LLM as the whole application.

```mermaid
flowchart TB
  U["Users / APIs / Files / Sensors / Events"] --> IN["Typed Ingress"]
  IN --> SDB["Incremental Semantic DB"]
  SDB --> SNAP["Immutable Semantic Snapshot"]
  SNAP --> CORE["PTR Core\nRaw + Typed + Epistemic + Latent Reasoning"]
  CORE --> ROUTER["Reasoning / Model / Search / Pod Router"]
  ROUTER --> EXEC["Typed Rust Execution Runtime"]
  EXEC --> PODS["Cognitive Pods"]
  EXEC --> VER["Verifier Fabric"]
  PODS --> OBS["Observations"]
  VER --> OBS
  OBS --> SDB
  EXEC --> SEC["Hard Action Boundary / Security"]
  SEC --> ACT["External Effects"]
  SDB --> LED["Consensus + Causal Ledger"]
  LED --> STATE["Materialized State"]
  STATE --> MEM["Semantic / Episodic / Procedural / Epistemic Memory"]
  MEM --> SEARCH["Derived Search Projections"]
  SEARCH --> ROUTER
  EXEC --> TEL["Tracing / Flow Signatures / Events"]
  TEL --> TRAIN["Training / RL / GEPA / Distillation"]
  TRAIN --> CORE
```

## Planes

1. **Ingress** — preserve raw evidence and produce a typed proposal.
2. **SemDB** — incremental/revisioned semantic world model.
3. **PTR Core** — raw + typed latent cognition.
4. **Router/Execution** — choose and run reasoning operators and Pods.
5. **Verifier/Security** — validate evidence and effects.
6. **Ledger/State** — causal authority and current materialization.
7. **Memory/Search** — lifecycle-managed memory plus derived retrieval.
8. **Observation/Learning** — traces, trajectories, experiments and training.

## Primary dependency direction

Domain semantics should point inward toward `ptr-types`; infrastructure adapters point outward. A search engine, network library or inference server must never become a dependency of the domain vocabulary.

## Truth boundary

See [authority-chain.mmd](../diagrams/authority-chain.mmd) and [Global invariants](../INVARIANTS.md).
