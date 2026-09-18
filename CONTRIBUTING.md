# Contributing to PTR

Every change should identify its layer: domain semantics, backend adapter, model modification, experiment, dataset, benchmark, or documentation metadata. Backend-specific types must not leak into PTR domain crates. New architectural claims require an experiment manifest and an update to the novelty/prior-art matrix. Negative results are first-class artifacts.

Rust code must follow [docs/RUST_API_STYLE.md](docs/RUST_API_STYLE.md) and [docs/TESTING.md](docs/TESTING.md). In particular: keep `lib.rs` as a thin facade where practical; group code by responsibility modules; default to private visibility, use `pub(crate)` for crate-internal contracts and `pub` only for intentional cross-crate API; use explicit `use`/`pub use`; prefer descriptive function parameters or typed request/options structs over ambiguous positional scalars/booleans; use `derive`, enums, exhaustive `match`, `Option<T>`, `Result<T, E>`, generics and collections according to their semantics rather than mechanically; prefer `IntoIterator`/`impl Iterator` and standard iterator adapters for sequence APIs, with the weakest appropriate `Fn`/`FnMut`/`FnOnce` closure bound.

## Component documentation is docs-as-code

Every first-class Rust crate owns `crates/<crate>/component.toml`. That file is the human-maintained source of truth for:

- current implementation;
- missing target-architecture work;
- next milestones;
- linked experiments;
- technology evaluations;
- decision records;
- current automated checks.

The generator also derives code footprint metrics directly from `src/` (Rust-file count, nonblank LOC and `#[test]` markers).

### Freshness rule

If a change touches `crates/<crate>/src/**` or `crates/<crate>/Cargo.toml`, the same change must update `crates/<crate>/component.toml`. CI checks this against the previous commit. This forces the implementation-status record to be reviewed whenever implementation changes.

After changing implementation status or links, run:

```bash
python3 scripts/update_component_docs.py --write
```

This refreshes the generated block in each crate README and `docs/components/STATUS.md`. CI rejects stale generated documentation.

Before submitting:

```bash
python3 scripts/check_component_metadata.py --base HEAD^
python3 scripts/update_component_docs.py --check
python3 scripts/check_repo.py
python3 scripts/check_architecture_catalog.py
python3 scripts/check_rust_conventions.py
python3 scripts/report_rust_api.py --public-only > /tmp/ptr-public-api.md
python3 scripts/run_experiment.py validate
python3 scripts/run_component_eval.py validate
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

For a new infrastructure candidate use `python3 scripts/new_candidate.py <component> <id>`.
For a new experiment use `python3 scripts/new_experiment.py <area/id-name> --hypothesis "..."`.