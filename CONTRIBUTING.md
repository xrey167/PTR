# Contributing to PTR

Every change should identify its layer: domain semantics, backend adapter, model modification, experiment, dataset, or benchmark. Backend-specific types must not leak into PTR domain crates. New architectural claims require an experiment manifest and an update to the novelty/prior-art matrix. Negative results are first-class artifacts.

Before submitting:

```bash
python3 scripts/check_repo.py
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

For a new infrastructure candidate use `python3 scripts/new_candidate.py <component> <id>`.
For a new experiment use `python3 scripts/new_experiment.py <area/id-name> --hypothesis "..."`.
