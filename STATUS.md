# Repository Status

PTR is a **compiling architecture/research prototype**, not yet a complete model/runtime product and not yet evidence of superiority over strong baselines.

## Implemented and validated

- 24 Rust workspace crates, including typed file/environment/CLI configuration and a central `ptr-runtime` orchestrator
- foundational domain types, semantic revision/snapshot prototype and dependency invalidation
- bounded-mailbox/isolate contracts and Pod lease typestate
- typed model → semantic Pod registry → Pod → verifier → SemDB observation loop for Pure/Read cognitive Pods
- revision, live-generation/revocation and capability/effect checks at the ActionIR boundary
- durable single-node reference `FileLedger` with fsync, reopen/replay and incomplete crash-tail truncation
- monotonic materialization with duplicate/out-of-order/gap rejection
- explicit retrieval evidence-promotion stages that reject stale generations and direct promotion to Known
- Prost-generated protobuf schemas with validated PodCall → domain conversion
- per-component `component.toml`, local `config.toml`, README status blocks and tests directories
- 154 configured workspace areas with matching tests directories
- Cargo and uv lockfiles; Rust 1.85 MSRV plus current-stable Ubuntu/Windows CI
- dataset cards, experiment/evaluation validators, setup scripts, devcontainer/Docker and release scaffolding
- supply-chain checks with cargo-audit/cargo-deny and Dependabot configuration
- `ptrctl doctor` repository/configuration diagnostics
- CI-tested dependency-free TypeScript SDK contract and static documentation landing page
- release packaging with SHA-256 checksums, CycloneDX SBOM generation and GitHub/Sigstore provenance/SBOM attestations

## Executed evidence so far

- **L001 is running, not completed.** The first reference FileLedger crash-tail slice executed 500 cases across five seeds with 0 stale-generation false accepts, 0 recovery-consistency failures and 0 tail-trim failures. This does **not** yet cover real process kills, fail-rs schedules, raft-engine, leader changes or partitions.
- Internal component smoke evidence exists for the custom SemDB and PTR isolate/mailbox path; both candidates remain `evaluating`, not selected.
- A runnable BM25 + caller-supplied dense-vector RRF reference baseline exists, but no claim against strong RAG is valid until a pinned embedding/reranking baseline is executed.

## Current research/implementation gap

- PTR-Core is still a structural research scaffold; there is no trainable Burn/CubeCL implementation yet
- model resume/checkpoint after verified Pod observations and router-driven multi-step execution are not yet wired
- production raft-engine/raft-rs, Turso, Iroh and real search-backend adapters remain unevaluated/unimplemented
- 19 architecture experiments remain planned; L001 is the only experiment with an executed scoped slice
- external component candidates mostly lack comparative benchmark evidence
- strong RAG/GraphRAG/editable-memory and matched plain-model comparative runs are not yet executed
- GitHub main-branch ruleset/repository-admin settings still require manual completion; recommendations are documented in `docs/GITHUB_SETTINGS.md`
- the TypeScript SDK API remains contract-only until `ptr-server` freezes and implements the matching `/v1` endpoints

## Reproducibility

- `Cargo.lock` pins Rust dependencies
- `training/uv.lock` pins the current Python training utility environment
- `rust-toolchain.toml` declares Rust 1.85.0 as MSRV/default
- CI separately tests MSRV and current stable on Linux and current stable on Windows
- experiment preparation records git SHA, hardware profile and lockfile hashes
- small JSON/TOML/Markdown result artifacts are versioned; large datasets/checkpoints remain out of Git

See [docs/components/STATUS.md](docs/components/STATUS.md) for per-component maturity and [experiments/lifecycle/L001-revocation-crash/results/](experiments/lifecycle/L001-revocation-crash/results/) for the first lifecycle evidence.