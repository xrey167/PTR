# Contributing to PTR

Every change should identify its layer: domain semantics, backend adapter, model modification, experiment, dataset, benchmark, or documentation metadata. Backend-specific types must not leak into PTR domain crates. New architectural claims require an experiment manifest and an update to the novelty/prior-art matrix. Negative results are first-class artifacts.

Rust code must follow [docs/RUST_API_STYLE.md](docs/RUST_API_STYLE.md) and [docs/TESTING.md](docs/TESTING.md). In particular: keep `lib.rs` as a thin facade where practical; group code by responsibility modules; default to private visibility, use `pub(crate)` for crate-internal contracts and `pub` only for intentional cross-crate API; use explicit `use`/`pub use`; prefer descriptive function parameters or typed request/options structs over ambiguous positional scalars/booleans; use `derive`, enums, exhaustive `match`, `Option<T>`, `Result<T, E>`, generics and collections according to their semantics rather than mechanically; prefer `IntoIterator`/`impl Iterator` and standard iterator adapters for sequence APIs, with the weakest appropriate `Fn`/`FnMut`/`FnOnce` closure bound; use exhaustive `match` for closed semantic states and match guards only for cheap deterministic refinement predicates; prefer functions/traits first and use hygienic `macro_rules!` only for compile-time repetition or small typed DSLs, with `$crate` in exported macros.

## Component documentation is docs-as-code

Every first-class Rust crate owns `crates/<crate>/component.toml`. That file is the human-maintained source of truth for:

- current implementation;
- missing target-architecture work;
- next milestones;
- linked experiments;
- technology evaluations;
- decision records;
- current automated checks.

The generator also derives code footprint metrics directly from `src/` and `tests/` (Rust-file count, nonblank LOC, and `#[test]`/`#[tokio::test]` markers).

### Freshness rule

If a change touches `crates/<crate>/src/**` or `crates/<crate>/Cargo.toml`, the same change must update `crates/<crate>/component.toml`. CI checks this against the previous commit. This forces the implementation-status record to be reviewed whenever implementation changes.

After changing implementation status or links, run:

```bash
python3 scripts/update_component_docs.py --write
```

This refreshes the generated block in each crate README and `docs/components/STATUS.md`. CI rejects stale generated documentation.

Before submitting, run what CI runs:

```bash
make ci-local                  # every job in .github/workflows/ci.yml, step for step
make ci-repository-invariants  # or one job: every target is ci-<job name>
make a0                        # also when you touch model/burn-a0, crates/ptr-types, the root Cargo.toml or the codebook
```

`make ci-local` needs the `stable`, `1.85.0` and `1.91.0` toolchains (it installs
the pinned two the way CI does) and builds every feature backend, so it is slow
the first time. It reads `BASE` from `origin/main`; without that ref, pass
`BASE=<commit>`. `make a0` is burn-a0.yml's two jobs: `a0-stable` formats, tests
and lints on your `stable`, and `a0-msrv` tests on 1.95.0. CI's stable is always
the current release, so run `rustup update stable` first, or a lint newer than
your toolchain passes here and fails there (Burn A0 needs at least 1.95).
`scripts/tests/test_ci_local.py` fails when a CI step, a step's `env:`, or a
multi-line script step has no counterpart in these targets, so they cannot
silently fall behind CI.

`python3 scripts/report_rust_api.py --public-only` prints the public Rust API for
review. It is a report, not a gate, so it is not part of `ci-local`.

For a new infrastructure candidate use `python3 scripts/new_candidate.py <component> <id>`.
For a new experiment use `python3 scripts/new_experiment.py <area>/<ID>-<name> --hypothesis "..." --metrics "..." --baseline "..." --falsification "..."` (optionally `--hardware-profile` and `--seeds`). It writes every key `experiments/schema.toml` requires and registers the experiment.