# Repository Status

PTR is a **compiling architecture/research prototype**, not yet a complete model/runtime product and not yet evidence of superiority over strong baselines.

## Implemented and validated

- 24 Rust workspace crates, including typed file/environment/CLI configuration and a central `ptr-runtime` orchestrator
- foundational domain types, semantic revision/snapshot prototype and dependency invalidation
- bounded-mailbox/isolate contracts and Pod lease typestate
- typed model → semantic Pod registry → Pod → verifier → SemDB observation loop for Pure/Read cognitive Pods
- typed ActionIR authorization decisions in `ptr-security`, with revision/generation freshness, independent effect authority, capability/effect denials and audit-ready allow receipts
- durable single-node reference `FileLedger` with PTRLOG02 integrity framing, strict reopen and independently anchored explicit crash-tail recovery
- `ptr-runtime` durable standalone mode can open/replay `FileLedger` and reconstruct lifecycle/materialized state across restart
- feature-gated raft-engine durable log, raft-rs single-node consensus, Turso materialized-state and direct Iroh transport adapters with dedicated CI jobs
- monotonic materialization with duplicate/out-of-order/gap rejection
- explicit retrieval evidence-promotion stages that reject stale generations and direct promotion to Known
- Prost-generated protobuf schemas with validated PodCall → domain conversion
- per-component `component.toml`, local `config.toml`, README status blocks and tests directories
- 160 configured workspace areas with matching tests directories
- Cargo and uv lockfiles; Rust 1.99 MSRV plus current-stable Ubuntu/Windows CI
- dataset cards, experiment/evaluation validators, setup scripts, devcontainer/Docker and release scaffolding
- executable experiment and component-evaluation runners with declared no-shell commands and immutable success/failure process evidence
- supply-chain checks with cargo-audit/cargo-deny and Dependabot configuration
- `ptrctl doctor` repository/configuration diagnostics
- CI-tested dependency-free TypeScript SDK contract and static documentation landing page
- release packaging with SHA-256 checksums, CycloneDX SBOM generation and GitHub/Sigstore provenance/SBOM attestations

## Executed evidence so far

- **L001 is running, not completed.** The reference in-process crash-tail slice executed 500 cases across five seeds with 0 stale-generation false accepts, 0 recovery-consistency failures and 0 tail-trim failures. A second real child-process abort slice executed 250 cases across five seeds with 0 stale-generation false accepts, 0 recovery failures, 0 tail-trim failures and 0 unexpected child exits. Feature-gated fail-rs fault injection also runs in CI. Production raft-engine, leader-change and network-partition coverage is still missing.
- Internal component smoke evidence exists for the custom SemDB and PTR isolate/mailbox path; both candidates remain `evaluating`, not selected.
- A runnable BM25 + caller-supplied dense-vector RRF reference baseline exists, but no claim against strong RAG is valid until a pinned embedding/reranking baseline is executed.

### M001/M002 paired evidence freeze

- Historical `M001-v2` and `M002-v2` records remain immutable and are not used as
  positive evidence claims. The current research-gate failure for `M002-v2` is
  an intentional historical-manifest mismatch, not a reason to rewrite its
  records or silently change its entrypoint.
- `M001-v4` and `M002-v4` each have complete paired five-seed aggregate records
  for seeds `17, 29, 43, 71, 101`, finite metrics, and paired 95% confidence
  interval artifacts. Both decisions are now explicitly bound as
  `INCONCLUSIVE/NO-GO`: on `ood_compose_regime`, accuracy improves but the
  lower-is-better NLL and ECE15 deltas are significantly positive. M002-v4 also
  compares `no-typed-attention` with the plain Transformer and therefore does
  not isolate Typed Attention. Their registry status remains `prepared` to
  preserve the frozen lifecycle history; `DECISION.toml` is the scientific
  decision and permits no positive claim.
- The research gate now accepts the historical M002-v2 conflict only through
  its exact commit- and digest-bound `results/NO-GO.toml` marker. The marker
  does not permit positive completion or alter any historical run record.
- `M002-v5` is `superseded`, not evidence. Its complete non-claimable 16-cell
  pilot archive remains immutable, but the prepared confirmatory entrypoint
  supplied bare fold names where the Rust backend correctly requires
  `fold=FNV64`. The resulting first preflight wrote no metrics and consumed no
  seed. Since a prepared study's configuration and decision script may not be
  replaced, it is superseded rather than patched. A successor must bind the
  exact fold digests before it can be prepared; M009 remains locked and no
  learned backend is authorized.
- `M002-v6` is superseded before scientific evidence. Its one committed
  Seed-17 preflight proved a second interface mismatch: the Rust arm catalog
  bound the selected arm names only to M002-v5. It emitted no dataset row,
  metric, checkpoint, or completed record. The failure is preserved in v6's
  committed runner record and `SUPERSEDED.md`; v6 is not patched in place.
- `M002-v7` is superseded before execution. Its versioned runner pair is
  retained, but the already-prepared v7 protocol lacked a complete Git-tree
  binding for the executable A0 implementation. It has no run record,
  checkpoint, metric, or claim. The immutable status record explains why it
  is not amended in place; M009 remains locked.
- `M002-v8` is superseded before execution. It had the intended A0 Git-tree
  binding, but its newly introduced gate implementation itself failed
  verification (wrong Rust source for the pair table; generated pilot
  artifacts compared to a code-source commit). It has no run record, metric,
  checkpoint, or claim. M002-v9 will only be registered after its corrected
  gate passes against a committed unregistered candidate.
- `M002-v9` is superseded after five committed seed records (Seeds 17, 29, 43,
  71, 101) and before any decision. It bound the immutable v5 pilot archive,
  FNV folds, versioned runner arms and their source files, and the complete
  tracked `model/burn-a0` Git tree; the A0 lockfile had to change for
  `ptr-types`' `sha2` dependency, which changes that tree, and the records
  cannot be re-pinned. Records and configuration are unamended
  (`SUPERSEDED.md`). No decision exists and no positive claim, M009 unlock, or
  Learned Backend authorization follows; a further study needs a new
  preregistered successor.

## Current research/implementation gap

- PTR-Core now has a separate trainable Burn A0 research package with typed metadata, bidirectional cross-attention, recurrent latent refinement and a router; it is still a small architecture probe rather than a pretrained language model
- bounded model resume after verified Pod observations is now wired and revision-advancing; opaque backend checkpoints, async streaming and router-driven operator selection remain open
- hard ActionIR authorization has diagnostic typed decisions plus a scoped synchronous execution gateway with opaque host-issued sessions, exact capsule/project grants and registered verifiers/executors; network authentication, scoped Pod access and durable audit/idempotency remain open
- feature-gated raft-engine, single-node raft-rs, Turso and direct Iroh adapters now exist and are under evaluation; multi-node consensus/network sessions and real search-backend adapters remain incomplete
- M002-v5 is superseded and the other planned architecture experiments are not positive evidence; L001 remains the only running experiment and has two executed scoped crash/recovery evidence slices
- external component candidates mostly lack comparative benchmark evidence
- strong RAG/GraphRAG/editable-memory and matched plain-model comparative runs are not yet executed
- GitHub main-branch ruleset/repository-admin settings still require manual completion; recommendations are documented in `docs/GITHUB_SETTINGS.md`
- the TypeScript SDK and Axum server now share `/health` and `/v1/requests`; streaming, auth/session policy and a real configured model backend remain open

## Reproducibility

- `Cargo.lock` pins Rust dependencies
- `training/uv.lock` pins the current Python training utility environment
- `rust-toolchain.toml` declares Rust 1.99.0 as MSRV/default
- CI separately tests MSRV and current stable on Linux and current stable on Windows
- experiment preparation/execution records git SHA, hardware profile, lockfile hashes, exact argv, duration, exit status and process output
- small JSON/TOML/Markdown result artifacts and the frozen operator-routing v1/v2
  datasets are versioned; model checkpoints remain out of Git

See [docs/components/STATUS.md](docs/components/STATUS.md) for per-component maturity and [experiments/lifecycle/L001-revocation-crash/results/](experiments/lifecycle/L001-revocation-crash/results/) for the first lifecycle evidence.

P0.2 adds canonical journaled semantic transactions, typed binary Pod results, transactional dependencies and validated revision reconstruction. Invalidated current derivations are evicted; prior log records and immutable snapshots remain. See `docs/architecture/22-durable-semantic-state.md`. Neural checkpoints, secure erasure, authenticated framing and cluster composition remain open.

## P0.3 persistence reference boundary

[Checked records and recovery snapshots](docs/architecture/23-persistence-integrity.md)
now cover bounded hash-chain verification, explicit legacy migration and complete
replay-backed snapshot restore. Existing files are never overwritten by restore.
Rollback protection requires an independently trusted anchor; default unanchored
open cannot detect a wholesale valid rewrite. This is not log compaction, remote
authentication, secure erasure or neural-state restoration. Earlier L001 numbers
are historical v1-format evidence, not measurements of this new v2 implementation.
