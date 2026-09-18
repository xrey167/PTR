# Repository Status

PTR is a **compiling architecture/research prototype**, not yet a complete model/runtime product and not yet evidence of superiority over strong baselines.

## Implemented and validated

- 24 Rust workspace crates, including typed configuration and a central `ptr-runtime` orchestrator
- foundational domain types, semantic revision/snapshot prototype and dependency invalidation
- bounded-mailbox/isolate contracts and Pod lease typestate
- model architecture structures for semantic slots, epistemic workspace, latent reasoning, routing, branching, ActionIR and PTR-AR/PTR-Diff configs
- causal ledger/materialized-state/search/storage/network/security contracts
- Prost-generated protobuf schemas using vendored `protoc`
- per-component `component.toml`, `config.toml`, README status blocks and tests directories
- Cargo and uv lockfiles
- dataset cards for imported training bundles
- GitHub CI, component-doc freshness checks and repository invariant checks

## Current research/implementation gap

- PTR-Core is not yet a trainable Burn/CubeCL model
- the full model→router→Pod/search→verifier→resume loop is not yet wired
- durable raft-engine/raft-rs/Turso/Iroh/search backends remain unevaluated adapters
- all 20 architecture experiments remain planned until executed
- all component technology decisions remain open until evidence is recorded
- strong RAG/GraphRAG and editable-memory baselines still require full benchmark implementations

## Reproducibility

- `Cargo.lock` pins Rust dependencies
- `training/uv.lock` pins the current Python training utility environment
- `rust-toolchain.toml` declares Rust 1.85.0 as MSRV/default
- CI separately tests MSRV and current stable
- experiment preparation records git SHA, hardware profile and lockfile hashes

See [docs/components/STATUS.md](docs/components/STATUS.md) for per-component maturity.
