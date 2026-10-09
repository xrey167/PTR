# Tests for experiments/feedback/F002-weak-supervision

- `mutations.toml`: the defects the F002 harness must detect (`python3 scripts/mutation_check.py F002`, anchors only with
  `--check`). Twenty are planted in `crates/ptr-labeling/src` and `crates/ptr-analytics/src`, nine in the harness itself.
- The criteria are pinned to the harness by the `f002` unit tests of `bins/ptr-bench` and to the aggregator by
  `scripts/tests/test_f002_aggregate.py`.

This directory holds tests, fixtures, validation cases, or executable checks owned by
`experiments/feedback/F002-weak-supervision`; it is not evidence of implementation by itself.
