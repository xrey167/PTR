# PTR Development Environment

PTR's development contract is editor-independent.

## Authoritative developer commands

The repository/CI commands remain authoritative:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets -- -D warnings
python scripts/check_repo.py
python scripts/check_architecture_catalog.py
python scripts/check_rust_conventions.py
python scripts/run_experiment.py validate
python scripts/run_component_eval.py validate
```

Editor integrations may wrap these commands but may not redefine them.

## Existing environments

- Dev Container: `.devcontainer/`
- VS Code extension hints: rust-analyzer + TOML support
- Emacs optional profile: `tooling/emacs/`

## Emacs Rust candidates

PTR tracks rust-mode, rustic and cargo-mode under `integrations/editor/` and compares their profiles under `evaluations/components/rust-editor-emacs/`.

No editor package is a runtime, build or release dependency.
