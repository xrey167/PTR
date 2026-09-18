# Contributing to PTR

Every change should identify its layer: domain semantics, backend adapter, model modification, experiment, dataset, benchmark, or documentation metadata. Backend-specific types must not leak into PTR domain crates. New architectural claims require an experiment manifest and an update to the novelty/prior-art matrix. Negative results are first-class artifacts.

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
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

For a new infrastructure candidate use `python3 scripts/new_candidate.py <component> <id>`.
For a new experiment use `python3 scripts/new_experiment.py <area/id-name> --hypothesis "..."`.
