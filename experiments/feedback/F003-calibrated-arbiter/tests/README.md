# Tests for experiments/feedback/F003-calibrated-arbiter

- `mutations.toml`: the defects the F003 harness must detect (`python3 scripts/mutation_check.py F003`, anchors only with
  `--check`). Twenty-nine are planted in `crates/ptr-branch/src/arbiter.rs` (one in `crates/ptr-analytics/src/binomial.rs`);
  every hard counter is expected by at least one, and the two equivalent mutants a plan might expect are documented in the file.
- The criteria are pinned to the harness by the `f003` unit tests of `bins/ptr-bench` and to the aggregator by
  `scripts/tests/test_f003_aggregate.py`.

This directory holds tests, fixtures, validation cases, or executable checks owned by
`experiments/feedback/F003-calibrated-arbiter`; it is not evidence of implementation by itself.
