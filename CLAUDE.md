# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

PTR (Probabilistically Typed Reasoning) is a research-first Rust monorepo for a typed cognitive runtime and model architecture. It is a compiling contract/scaffold baseline: a README or doc describing a subsystem does not mean it is implemented. Completion is tracked through tests, experiments and `docs/DEFINITION_OF_DONE.md`; `STATUS.md` and `docs/components/STATUS.md` hold the live status. Every architectural claim must be falsifiable (baseline, ablation, failure criteria), and negative results are first-class artifacts.

## Commands

Toolchain is pinned in `rust-toolchain.toml` (1.99.0, also the workspace MSRV). Cargo aliases in `.cargo/config.toml`: `xtest`, `xcheck`, `xclippy`, `xdoc`. Always pass `--locked`.

```bash
make check                       # cargo check --workspace --all-targets --locked
make test                        # cargo test --workspace --locked
make fmt                         # rustfmt check, root workspace AND model/burn-a0
cargo test -p ptr-ledger --locked                 # one crate
cargo test -p ptr-ledger --locked <test_name>     # one test by name filter
cargo test -p ptr-ledger --test failpoints --features failpoints --locked   # one integration target (feature-gated)

make repo-check                  # scripts/check_repo.py, MSRV alignment, contract citations
make docs                        # regenerate component docs (python3 scripts/update_component_docs.py --write)
make docs-check                  # fail if generated docs are stale
make python-test                 # training/ and scripts/ unittests
make a0                          # model/burn-a0 (separate workspace, needs stable and 1.95.0)
```

CI is mirrored step for step by `make ci-local`; each CI job is a target named `ci-<job>` (for example `ci-quality`, `ci-repository-invariants`, `ci-ledger-raft-rs`). `scripts/tests/test_ci_local.py` fails if a CI step lacks a Makefile counterpart, so add both together. `ci-quality` runs fmt, `clippy -D warnings` and `RUSTDOCFLAGS="-D warnings"` docs on `stable`. `ci-repository-invariants` needs `origin/main` or `BASE=<commit>`. `ci-state-postgres` needs a local PostgreSQL 16+/pgvector (`PTR_PG_TEST_DSN`).

Backends are Cargo features and tested by dedicated jobs, so a plain `--workspace` run does not exercise them. Examples: `ptr-ledger` (`failpoints`, `raft-engine-backend`, `raft-rs-backend`), `ptr-state` (`turso-backend`), `ptr-net` (`iroh-backend`), `ptr-cluster`/`ptr-execwire`/`ptr-podwire` (`*-backend`), `ptr-storage` (`tier-opendal-*`), `ptr-pg` (`postgres-backend`), `ptr-observe` (`tracing-adapter`). Some integration tests are declared with `required-features`, so they silently do not build without the feature.

Experiments and evaluations: `python3 scripts/run_experiment.py validate`, `python3 scripts/run_component_eval.py validate`, `python3 scripts/check_research_gates.py`. New ones come from `scripts/new_experiment.py` and `scripts/new_candidate.py`, which register them in the registries.

## Architecture

The workspace members are `crates/ptr-*`, `bins/ptr-*`, `bins/ptrctl` and `bins/ptrd`. `fuzz`, `model/burn-a0` and `vendor/*` are excluded and are their own workspaces or vendored sources. `bins/` holds composition and bootstrap only; domain behavior belongs in `crates/`.

The request flow spans many crates, so read these together:

- **`ptr-types`** is the shared type kernel (semantic roles, epistemic axes, operators, lifecycle/provenance, identities). Nearly everything depends on it.
- **`ptr-ingress` → `ptr-semdb`**: raw evidence is preserved alongside a provisional typed interpretation. SemDB holds revisioned state with incremental-compiler-style dependency invalidation and hands out immutable snapshots.
- **`ptr-runtime`** is the orchestrator for the end-to-end request, authority and effect lifecycle. It composes `ptr-router` (picks neural/search/Pod/symbolic strategy), `ptr-model-api` (backend-independent inference), `ptr-pods` (semantic-contract specialist modules, leased via typestate), `ptr-verifier`, `ptr-security` and `ptr-exec` (supervised isolates with bounded mailboxes).
- **Authority path**: `ptr-ledger` (ordered, durable causal log; `FileLedger`, optionally raft) is authoritative. `ptr-state` projects committed events into queryable views, `ptr-memory` stores lifecycle-managed memory, and `ptr-search` is derived and non-authoritative. `ptr-pg` hosts anchor-verified projections in PostgreSQL.
- **Wire/cluster**: `ptr-protocol` defines PodWire and converts prost-generated protobuf (`proto/`) into validated domain values. `ptr-net` is authenticated transport (Iroh). `ptr-cluster`, `ptr-execwire` and `ptr-podwire` each carry one protocol on its own ALPN over `ptr-net`.
- **Model/training side**: `ptr-core` (research model: typed slots, latent recurrence), `model/burn-a0`, `ptr-feedback`, `ptr-fastmem`, `ptr-lineage`, `ptr-labeling`, `ptr-branch`, `ptr-analytics`, plus `training/` (Python, locked with `training/uv.lock`).

Invariants in `docs/INVARIANTS.md` are binding: raw is never replaced by typed state, search never self-promotes to truth, uncommitted state is never authoritative, effects pass a hard security boundary requiring current revision/generation, queues are bounded, and a learned score cannot override a deterministic verifier contradiction. `docs/COMPONENT_CONTRACTS.md` and `docs/architecture/` hold the details. Backend-specific types must not leak into domain crates.

## Conventions enforced by CI

- **Docs-as-code freshness**: any change under `crates/<crate>/src/**` or `crates/<crate>/Cargo.toml` must also update `crates/<crate>/component.toml`. Then run `make docs` to refresh the generated README blocks and `docs/components/STATUS.md`. `scripts/check_component_metadata.py` checks this against the base commit.
- Rust style is in `docs/RUST_API_STYLE.md` and testing layout in `docs/TESTING.md`. Key points: thin `lib.rs` facade, private by default (`pub(crate)` for internal contracts), typed request/option structs instead of ambiguous positional booleans or scalars, exhaustive `match` for closed states. `scripts/check_rust_conventions.py` has mechanical checks, for example on `check_`/`validate_`/`ensure_` function names, which should be fallible. Unit tests sit beside the code; integration tests in `tests/` use only the public API. Rustfmt width is 100.
- Each crate has `component.toml`, `config.toml`, a README with a diagram and a `tests/` directory, and `scripts/check_repo.py` asserts this structure. Any new binary or crate must be a workspace member or an explicit exclusion in the root `Cargo.toml`.
- `vendor/` holds patched upstream crates (see `docs/VENDOR_PATCH_POLICY.md`; integrity-checked by `scripts/check_vendor_integrity.py`). Do not edit them casually.
- `.cargo/config.toml` must not contain an `[env]` table (`check_research_gates.py` rejects it); put machine-specific CUDA settings in the user-level Cargo config.
- Experiment records are immutable once prepared. Superseded experiments stay as history, and a successor is created rather than patching a prepared one. See `STATUS.md` for the current evidence freeze.
