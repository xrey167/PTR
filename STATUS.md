# Repository Status

PTR is a **compiling architecture/research prototype**, not yet a complete model/runtime product and not yet evidence of superiority over strong baselines.

## Implemented and validated

- 24 Rust workspace crates, including typed file/environment/CLI configuration and a central `ptr-runtime` orchestrator
- foundational domain types, semantic revision/snapshot prototype and dependency invalidation
- bounded-mailbox/isolate contracts and Pod lease typestate
- typed model → semantic Pod registry → Pod → verifier → SemDB observation loop for Pure/Read cognitive Pods
- typed ActionIR authorization decisions in `ptr-security`, with revision/generation freshness, capability/effect denials and audit-ready allow receipts
- durable single-node reference `FileLedger` with fsync, reopen/replay and incomplete crash-tail truncation
- `ptr-runtime` durable standalone mode can open/replay `FileLedger` and reconstruct lifecycle/materialized state across restart
- feature-gated raft-engine durable log, raft-rs single-node consensus, Turso materialized-state and direct Iroh transport adapters with dedicated CI jobs
- monotonic materialization with duplicate/out-of-order/gap rejection
- explicit retrieval evidence-promotion stages that reject stale generations and direct promotion to Known
- Prost-generated protobuf schemas with validated PodCall → domain conversion
- per-component `component.toml`, local `config.toml`, README status blocks and tests directories
- 160 configured workspace areas with matching tests directories
- Cargo and uv lockfiles; Rust 1.85 MSRV plus current-stable Ubuntu/Windows CI
- dataset cards, experiment/evaluation validators, setup scripts, devcontainer/Docker and release scaffolding
- supply-chain checks with cargo-audit/cargo-deny and Dependabot configuration
- `ptrctl doctor` repository/configuration diagnostics
- CI-tested dependency-free TypeScript SDK contract and static documentation landing page
- release packaging with SHA-256 checksums, CycloneDX SBOM generation and GitHub/Sigstore provenance/SBOM attestations

## Executed evidence so far

- **L001 is running, not completed.** The reference in-process crash-tail slice executed 500 cases across five seeds with 0 stale-generation false accepts, 0 recovery-consistency failures and 0 tail-trim failures. A second real child-process abort slice executed 250 cases across five seeds with 0 stale-generation false accepts, 0 recovery failures, 0 tail-trim failures and 0 unexpected child exits. Feature-gated fail-rs fault injection also runs in CI. Production raft-engine, leader-change and network-partition coverage is still missing.
- Internal component smoke evidence exists for the custom SemDB and PTR isolate/mailbox path; both candidates remain `evaluating`, not selected.
- A runnable BM25 + caller-supplied dense-vector RRF reference baseline exists, but no claim against strong RAG is valid until a pinned embedding/reranking baseline is executed.

## Current research/implementation gap

- PTR-Core now has a separate trainable Burn A0 research package with typed metadata, bidirectional cross-attention, recurrent latent refinement and a router; it is still a small architecture probe rather than a pretrained language model
- bounded model resume after verified Pod observations is now wired and revision-advancing; opaque backend checkpoints, async streaming and router-driven operator selection remain open
- hard ActionIR authorization now has typed security decisions/receipts, but principal/session/resource scopes, verifier requirements and durable authorization-audit persistence remain open
- feature-gated raft-engine, single-node raft-rs, Turso and direct Iroh adapters now exist and are under evaluation; multi-node consensus/network sessions and real search-backend adapters remain incomplete
- 19 architecture experiments remain planned; L001 is the only running experiment and now has two executed scoped crash/recovery evidence slices
- external component candidates mostly lack comparative benchmark evidence
- strong RAG/GraphRAG/editable-memory and matched plain-model comparative runs are not yet executed
- GitHub main-branch ruleset/repository-admin settings still require manual completion; recommendations are documented in `docs/GITHUB_SETTINGS.md`
- the TypeScript SDK and Axum server now share `/health` and `/v1/requests`; streaming, auth/session policy and a real configured model backend remain open

## Reproducibility

- `Cargo.lock` pins Rust dependencies
- `training/uv.lock` pins the current Python training utility environment
- `rust-toolchain.toml` declares Rust 1.85.0 as MSRV/default
- CI separately tests MSRV and current stable on Linux and current stable on Windows
- experiment preparation records git SHA, hardware profile and lockfile hashes
- small JSON/TOML/Markdown result artifacts are versioned; large datasets/checkpoints remain out of Git

See [docs/components/STATUS.md](docs/components/STATUS.md) for per-component maturity and [experiments/lifecycle/L001-revocation-crash/results/](experiments/lifecycle/L001-revocation-crash/results/) for the first lifecycle evidence.