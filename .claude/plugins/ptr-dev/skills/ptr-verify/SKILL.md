---
name: ptr-verify
description: Derive and run exactly the PTR checks that the current changes require (cargo test/clippy/fmt per changed crate, component.toml freshness, docs-check, feature-gated backend jobs, python tests, experiment and evaluation validators, burn-a0, py-bindings, repo-check) and report each command with pass/fail and real output excerpts. Use when asked to "verify", "run the checks", "is this ready to push/merge", "what CI will fail", or after finishing a change.
---

# Verify the current change

Run only what the diff needs (quick), or everything CI runs (full). Report real output, never a guess.

## 1. Find what changed

```bash
git fetch origin main 2>/dev/null; BASE=$(git merge-base HEAD origin/main 2>/dev/null || echo HEAD~1)
git diff --name-only "$BASE"...HEAD
git status --porcelain
```

Treat committed (`diff`) and uncommitted (`status`) paths together. If `origin/main` is missing, pass an explicit `BASE=<commit>` (the Makefile uses it for `ci-repository-invariants`).

## 2. Map paths to checks

Always pass `--locked` to cargo. Clippy form: `cargo clippy ... --locked -- -D warnings` (`--locked` before `--`; the `xclippy` alias cannot take it).

| Changed path | Run |
|---|---|
| `crates/<c>/**` or `bins/<b>/**` | `cargo test -p <c> --locked`; `cargo clippy -p <c> --all-targets --locked -- -D warnings`; `cargo fmt --all -- --check` |
| `crates/<c>/src/**` or `crates/<c>/Cargo.toml` | `crates/<c>/component.toml` must also be in the diff; then `make docs-check` (if stale: `make docs`, then re-run and include the regenerated files); `python3 scripts/check_component_metadata.py --base "$BASE"` (same as `make meta-check`, which uses `HEAD^`) |
| backend feature code (see below) | the matching `make ci-<job>` |
| `training/**` | `PYTHONPATH=training/src make python-test` (the target does not set `PYTHONPATH`) |
| `experiments/**` | `python3 scripts/run_experiment.py validate` and `python3 scripts/check_research_gates.py` |
| `evaluations/**` | `python3 scripts/run_component_eval.py validate` |
| `bindings/ptr-py/**` | `make py-bindings` (needs Python >= 3.11, network for maturin) |
| `model/burn-a0/**` | `make a0` (needs `stable` and `1.95.0` toolchains); `make fmt` also checks it |
| `scripts/**` | `python3 -m unittest discover -s scripts/tests` |
| `Cargo.toml`, `Cargo.lock`, `.cargo/`, `rust-toolchain.toml` | `make check`, `make test` |
| anything | `make repo-check` (runs `check_repo.py`, `check_msrv_alignment.py`, `check_contract_citations.py`) |

Feature-gated backends. A plain `--workspace` run does not build these, and some integration tests use `required-features`, so they silently do not build without the feature:

| Crate and feature | Target |
|---|---|
| `ptr-ledger` `failpoints` | `make ci-lifecycle-failpoints` |
| `ptr-ledger` `raft-engine-backend` | `make ci-ledger-raft-engine` |
| `ptr-ledger` `raft-rs-backend` | `make ci-ledger-raft-rs` |
| `ptr-state` `turso-backend` | `make ci-state-turso` |
| `ptr-net` `iroh-backend` | `make ci-network-iroh` |
| `ptr-cluster` `cluster-backend` | `make ci-cluster-wire` |
| `ptr-execwire` `execwire-backend` | `make ci-execution-wire` |
| `ptr-podwire` `podwire-backend` | `make ci-pod-wire` |
| `ptr-storage` `tier-opendal-*` | `make ci-storage-tiers` |
| `ptr-pg` `postgres-backend` (also `ptr-bench` experiment steps) | `make ci-state-postgres` (needs local PostgreSQL 16+/pgvector; DSNs `PTR_PG_TEST_DSN`, `PTR_PG_EXPERIMENT_DSN`) |
| `ptr-observe` `tracing-adapter` | part of `make ci-quality` |

Not compiled by any CI job: `ptr-net` `wireguard-uapi-backend` / `wintun-backend`, `candle-cuda` of `ptr-pods`/`ptr-runtime`. Say so if the diff touches them.

## 3. Scale

- **Quick**: the rows above for the changed crates only, plus `make repo-check`.
- **Full workspace**: `make check`, `make test`, `make fmt`, `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- **Everything CI runs on Linux**: `make ci-local` (long). A single CI job: `make ci-quality`, `make ci-rust-stable`, `make ci-rust-msrv`, `make ci-python-training`, `make ci-repository-invariants` (needs `origin/main` or `BASE=<commit>`). `ci-local` does not cover `security.yml`, `sdk.yml`, `burn-a0.yml` (use `make a0`), the Windows leg or the Python 3.11 leg.

## 4. Report

Run each command, capture exit code and the last meaningful lines. Output one table, then details for failures:

```
| command | result | note |
| cargo test -p ptr-ledger --locked | PASS | 112 passed |
| make docs-check | FAIL | docs/components/STATUS.md stale -> run make docs |
```

Quote real output excerpts for every FAIL. Distinguish "not run" (and why: missing toolchain, no PostgreSQL, no network) from "passed". Never report green for a command that was skipped. Do not edit prepared experiment records or `vendor/` to make a check pass.
