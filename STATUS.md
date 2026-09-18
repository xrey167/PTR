# Repository Status

This repository is a **full architecture and research scaffold**, not a claim that the complete PTR system is already implemented.

## Implemented in the scaffold

- Cargo workspace and domain crate boundaries
- foundational Rust domain types
- revisioned semantic host / dependency invalidation prototype
- bounded mailbox / isolate semantics prototype
- Pod contracts and lease typestate prototype
- model architecture structs for semantic slots, typed attention config, latent reasoning, operator routing, branching, probabilistic reasoning, action/verifier heads, PTR-AR and PTR-Diff
- causal ledger/state/search/storage/network/security contracts
- protocol `.proto` contracts
- 25 component-evaluation workspaces
- 20 planned architecture/system experiments
- 12 explicit model/reasoning modification specifications
- dataset schemas, registry and imported typed-behavior / PodWire bundles
- training workspace and stage configs
- novelty, falsification, ADR and baseline structure
- CI and repository invariant checks

## Intentionally not hard-wired yet

External libraries such as raft-rs, raft-engine, Turso, Iroh, Tantivy, Zvec, cuVS, LanceDB, Havenask, Burn, SGLang, vLLM, tracing, valuable, Iggy, ROCK and others remain behind PTR-owned contracts until their component evaluation is run.

## Validation performed here

- repository invariant script passed
- all TOML and JSON/JSONL files were parsed successfully by the local validation script
- git repository initialized and bundled

The execution environment used to generate this artifact does **not** contain `rustc`/`cargo`, so Rust compilation and tests could not be executed locally. GitHub CI is configured to run formatting, `cargo check`, tests and clippy on a Rust-enabled runner.
