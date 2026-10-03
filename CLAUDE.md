# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

PTR (Probabilistically Typed Reasoning) is a research-first Rust monorepo for a typed cognitive runtime and model architecture. It is a compiling contract/scaffold baseline: a README or doc describing a subsystem does not mean it is implemented. Completion is tracked through tests, experiments and `docs/DEFINITION_OF_DONE.md`; `docs/components/STATUS.md` (generated from every crate's `component.toml`) is the current component inventory; the root `STATUS.md` is hand-maintained evidence and status notes that can lag (its crate count is out of date). Every architectural claim must be falsifiable (baseline, ablation, failure criteria), and negative results are first-class artifacts.

## Commands

Toolchain is pinned in `rust-toolchain.toml` (1.99.0, also the workspace MSRV). Cargo aliases in `.cargo/config.toml`: `xtest`, `xcheck` and `xdoc` already include `--locked`. `xclippy` does not, and appending `--locked` to it would land after the `--` separator, so run `cargo clippy --workspace --all-targets --locked -- -D warnings` instead. Otherwise always pass `--locked`.

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
PYTHONPATH=training/src make python-test   # training/ and scripts/ unittests (the target does not set PYTHONPATH itself)
make a0                          # model/burn-a0 (separate workspace, needs stable and 1.95.0)
```

`make ci-local` covers the command steps of `.github/workflows/ci.yml` on Linux; each CI job is a target named `ci-<job>` (for example `ci-quality`, `ci-repository-invariants`, `ci-ledger-raft-rs`). It does not reproduce the editable training install (`pip install -e training`; `PYTHONPATH` is set instead), `security.yml` (needs cargo-audit and cargo-deny), `sdk.yml`, `burn-a0.yml` (use `make a0`), the Windows leg of `rust-stable` or the Python 3.11 leg of `python-training`. `scripts/tests/test_ci_local.py` fails if a CI step lacks a Makefile counterpart, so add both together. `ci-quality` runs fmt, `clippy -D warnings` and `RUSTDOCFLAGS="-D warnings"` docs on `stable`. `ci-repository-invariants` needs `origin/main` or `BASE=<commit>`. `ci-state-postgres` needs a local PostgreSQL 16+/pgvector and reads two variables: `PTR_PG_TEST_DSN` for the `ptr-pg` tests and `PTR_PG_EXPERIMENT_DSN` for the `ptr-bench` experiment steps.

Backends are Cargo features, so a plain `--workspace` run does not exercise them; most have a dedicated CI job. Examples: `ptr-ledger` (`failpoints`, `raft-engine-backend`, `raft-rs-backend`), `ptr-state` (`turso-backend`), `ptr-net` (`iroh-backend`), `ptr-cluster`/`ptr-execwire`/`ptr-podwire` (`*-backend`), `ptr-storage` (`tier-opendal-*`), `ptr-pg` (`postgres-backend`), `ptr-observe` (`tracing-adapter`). Some integration tests are declared with `required-features`, so they silently do not build without the feature. No workflow enables `ptr-net`'s `wireguard-uapi-backend` and `wintun-backend` or the `candle-cuda` feature of `ptr-pods`/`ptr-runtime`, so nothing in CI compiles those paths.

Experiments and evaluations: `python3 scripts/run_experiment.py validate`, `python3 scripts/run_component_eval.py validate`, `python3 scripts/check_research_gates.py`. `scripts/new_experiment.py` creates and registers a new experiment. `scripts/new_candidate.py <component> <id>` only appends a candidate to an existing `evaluations/components/<component>/candidates.toml`; it neither creates an evaluation nor updates `evaluations/registry.toml`.

## Architecture

The workspace members are `crates/ptr-*`, `bins/ptr-*`, `bins/ptrctl` and `bins/ptrd`. `fuzz`, `model/burn-a0` and `vendor/*` are excluded and are their own workspaces or vendored sources. `bins/` holds composition, bootstrap and binary-specific benchmark/experiment harnesses (for example the S003 `certified-branches` harness in `ptr-bench`); reusable domain behavior belongs in `crates/`.

The request flow spans many crates, so read these together:

- **`ptr-types`** is the shared type kernel (semantic roles, epistemic axes, operators, lifecycle/provenance, identities). Nearly everything depends on it.
- **`ptr-ingress` → `ptr-semdb`**: raw evidence is preserved alongside a provisional typed interpretation. SemDB holds revisioned state with incremental-compiler-style dependency invalidation and hands out immutable snapshots.
- **`ptr-runtime`** is the orchestrator for the end-to-end request, authority and effect lifecycle. It composes `ptr-router` (picks neural/search/Pod/symbolic strategy), `ptr-model-api` (backend-independent inference), `ptr-pods` (semantic-contract specialist modules, leased via typestate), `ptr-verifier`, `ptr-security` and `ptr-exec` (supervised isolates with bounded mailboxes).
- **Authority path**: `ptr-ledger` (ordered, durable causal log; `FileLedger`, optionally raft) is authoritative. `ptr-state` projects committed events into queryable views, `ptr-memory` stores lifecycle-managed memory, and `ptr-search` is derived and non-authoritative. `ptr-pg` hosts anchor-verified projections in PostgreSQL.
- **Wire/cluster**: `ptr-protocol` defines PodWire and converts prost-generated protobuf (`proto/`) into validated domain values. `ptr-net` is authenticated transport (Iroh). `ptr-cluster`, `ptr-execwire` and `ptr-podwire` each carry one protocol on its own ALPN over `ptr-net`.
- **Model/training side**: `ptr-core` (research model: typed slots, latent recurrence), `model/burn-a0`, `ptr-feedback`, `ptr-fastmem`, `ptr-lineage`, `ptr-labeling`, `ptr-branch`, `ptr-analytics`, plus `training/` (Python, locked with `training/uv.lock`).

Invariants in `docs/INVARIANTS.md` are binding: raw is never replaced by typed state, search never self-promotes to truth, uncommitted state is never authoritative, effects pass a hard security boundary requiring current revision/generation, queues are bounded, and a learned score cannot override a deterministic verifier contradiction. `docs/COMPONENT_CONTRACTS.md` and `docs/architecture/` hold the details. Backend-specific types must not leak into domain crates.

## Conventions and CI checks

- **Docs-as-code freshness**: any change under `crates/<crate>/src/**` or `crates/<crate>/Cargo.toml` must also update `crates/<crate>/component.toml`. Then run `make docs` to refresh the generated README blocks and `docs/components/STATUS.md`. `scripts/check_component_metadata.py` checks this against the base commit.
- Rust style is in `docs/RUST_API_STYLE.md` and testing layout in `docs/TESTING.md`. Key points: thin `lib.rs` facade, private by default (`pub(crate)` for internal contracts), typed request/option structs instead of ambiguous positional booleans or scalars, exhaustive `match` for closed states. These are repository guidance, not CI checks: `scripts/check_rust_conventions.py` only verifies that `check_`/`validate_`/`ensure_` functions return `Result` and that `tests/common` helpers contain no test annotations. Unit tests sit beside the code; integration tests in `tests/` use only the public API. Rustfmt width is 100.
- Each crate has `component.toml`, `config.toml`, a README with a diagram and a `tests/` directory, and `scripts/check_repo.py` asserts this structure. Any new binary or crate must be a workspace member or an explicit exclusion in the root `Cargo.toml`.
- `vendor/` holds patched upstream crates (see `docs/VENDOR_PATCH_POLICY.md`; integrity-checked by `scripts/check_vendor_integrity.py`). Do not edit them casually.
- `.cargo/config.toml` must not contain an `[env]` table (`check_research_gates.py` rejects it); put machine-specific CUDA settings in the user-level Cargo config.
- CI (`scripts/check_research_gates.py`) freezes prepared experiment records only for experiments listed in `experiments/preregistration.toml`; unlisted prepared records are not covered by that check. Superseded experiments stay as history, and a successor is created rather than patching a prepared one. See `STATUS.md` for the current evidence freeze.
