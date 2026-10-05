---
name: ptr-crate-change
description: Workflow for changing an existing PTR crate or adding a new one, following docs/RUST_API_STYLE.md, docs/TESTING.md, docs-as-code (component.toml, make docs), README diagram, workspace membership, backend-as-Cargo-feature rules and docs/INVARIANTS.md. Use when asked to "add a crate", "change ptr-<name>", "implement a component", "add a backend", or "touch crates/ptr-*/src".
---

# Change or add a PTR crate

PTR is a contract/scaffold baseline: a README describing a subsystem does not mean it is implemented. Read `CLAUDE.md`, then do the steps below.

## 1. Read before editing

- `docs/RUST_API_STYLE.md`. Key rules: thin `lib.rs` (only `mod` declarations and deliberate `pub use` re-exports); responsibility modules (`types.rs`, `error.rs`, `ports.rs`, `service.rs`, ... create only what is needed); private by default, `pub(super)`/`pub(crate)` for internal contracts, `pub` only for intentional cross-crate contracts; typed request/option structs instead of ambiguous positional booleans or scalars; exhaustive `match` over closed enums (no catch-all `_` hiding new variants); provider-native types confined to `adapters/<backend>.rs`.
- `docs/TESTING.md`. Unit tests sit beside the code in `#[cfg(test)] mod tests` (may use private items via `super::*`; never widen visibility just to test). Integration tests in `tests/*.rs` use only the public API; shared helpers in `tests/common/mod.rs` must contain no `#[test]` annotations. Typical files: `public_contract.rs`, `lifecycle.rs`, `failure.rs`, `adapters.rs`.
- `docs/INVARIANTS.md` (binding) and `docs/COMPONENT_CONTRACTS.md`; architecture details in `docs/architecture/`.
- Invariants to keep: raw is never replaced by typed state; search never self-promotes to truth; uncommitted state is never authoritative; effects pass a hard security boundary requiring current revision/generation; queues are bounded; a learned score cannot override a deterministic verifier contradiction.

## 2. Implement

- Backends are Cargo features. Backend-specific types must not leak into domain crates; expose a PTR-owned trait (port) and put the backend behind a feature in an adapter module. Name features as the existing ones do (`*-backend`).
- Reusable domain behavior belongs in `crates/`; `bins/` holds composition and bootstrap only.
- Do not edit `vendor/` casually (`docs/VENDOR_PATCH_POLICY.md`).
- Rustfmt width is 100.

## 3. Docs-as-code (CI-checked)

Any change under `crates/<crate>/src/**` or `crates/<crate>/Cargo.toml` must also update `crates/<crate>/component.toml`. Then:

```bash
make docs         # refresh generated README blocks and docs/components/STATUS.md
make docs-check   # must pass
```

Each crate needs `component.toml`, `config.toml`, a README with a diagram, and a `tests/` directory (`scripts/check_repo.py` asserts this). A starting point is `templates/rust-crate/`.

## 4. New crate checklist

- Directory `crates/ptr-<name>/` is picked up by the `crates/ptr-*` glob in the root `Cargo.toml`. A binary or crate outside those globs must be a workspace member or an explicit exclusion in the root `Cargo.toml`.
- Add `component.toml`, `config.toml`, `README.md` (with diagram), `tests/`.
- Commit `Cargo.lock` changes; everything runs with `--locked`.
- Toolchain is pinned in `rust-toolchain.toml` (1.99.0, also the MSRV).

## 5. Verify (all with `--locked`)

```bash
cargo test -p <crate> --locked
cargo clippy -p <crate> --all-targets --locked -- -D warnings   # --locked goes before the -- separator
cargo fmt --all -- --check
make docs-check
make repo-check
```

Feature-gated code needs its own job (for example `make ci-ledger-raft-rs`); the `ptr-verify` skill lists them. Use `cargo test -p <crate> --locked <test_name>` for one test and `--test <target> --features <feature>` for one feature-gated integration target.
