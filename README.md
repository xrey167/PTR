# PTR — Probabilistically Typed Reasoning

PTR is a research-first Rust monorepo for a **typed cognitive runtime and model architecture**. The goal is not to build another LLM tool wrapper. PTR treats language, typed semantic state, uncertainty, reasoning operators, external cognitive modules, durable lifecycle state and verification as separate first-class layers.

## Core thesis

A PTR system keeps **raw language and typed semantics in parallel**, reasons over a **probabilistically typed latent workspace**, routes problems to different reasoning operators or specialist Pods, and only permits external effects after a **hard typed action boundary**.

```text
RAW INPUT ───────────────┐
                         ├─ Cross Verification ─> Incremental SemDB ─> Snapshot
TYPED SEMANTIC PROPOSAL ─┘                                      │
                                                                ▼
                                                         PTR Core Model
                                               language + typed semantic slots
                                                                │
                                                        ModelEvent stream
                                                                │
                                                         Rust Runtime
                                            Pods + Verifiers + Action Boundary
                                                                │
                                               Observation / Semantic Delta
                                                                │
                                        Consensus → Ledger → Materialized State
                                                                │
                                                   Semantic Capsules
                                                                │
                                          Derived Search / Cache Projections
```

## Design rules

1. **Hard shell, soft core.** Reasoning may be uncertain; external effects must be valid.
2. **Raw is never replaced by typed.** Typed semantics are a revisable interpretation of raw evidence.
3. **Revision is not generation.** Revisions identify observed input state; generations identify object lifecycle versions.
4. **Search is derived.** Search hits are candidates, never semantic authority.
5. **Committed causal state is authoritative.** Materialized views and indexes follow committed state.
6. **Pods expose semantic contracts.** Models learn capabilities and types, not fragile tool names.
7. **Every architecture component is benchmarkable and replaceable.** External libraries live behind PTR-owned contracts.
8. **Novelty claims require ablations and strong baselines.** `research/` contains explicit falsification and prior-art tracking.

## Repository map

- `crates/` — Rust runtime and model architecture crates
- `bins/` — `ptrd`, `ptrctl`, `ptr-worker`, `ptr-bench`
- `model/` — model configs, modification specs, reasoning research and checkpoints
- `training/` — SFT/RL/distillation/GEPA/DSPy training workspace
- `datasets/` — dataset registry, schemas, splits and imported bundles
- `experiments/` — model, runtime, lifecycle, retrieval and end-to-end experiments
- `evaluations/` — per-component technology evaluation; no component is permanently locked without evidence
- `benchmarks/` — reusable benchmark suite definitions
- `integrations/` — external systems and adapters, always behind internal contracts
- `research/` — hypotheses, ADRs, novelty matrix, baselines and falsification plans
- `docs/` — detailed architecture documentation and diagrams
- `proto/` — network wire contracts; domain types remain separate

## First milestones

1. Build the Rust foundation: `ptr-types`, `ptr-semdb`, `ptr-exec`, `ptr-protocol`, `ptr-ledger`.
2. Establish strong baselines with an external model backend.
3. Implement PTR-Core A0: semantic slots, typed attention, latent recurrence and operator routing.
4. Run architecture ablations before scaling the model.
5. Add cluster consensus only after standalone lifecycle invariants pass chaos tests.

See `docs/ROADMAP.md` and `docs/DEFINITION_OF_DONE.md`.
