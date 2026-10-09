# Tests for experiments/runtime/R001-mailbox-backpressure

- `mutations.toml`: the defects the R001 harness must detect (`python3 scripts/mutation_check.py R001`, anchors only with
  `--check`). Twenty-two are planted in `crates/ptr-exec/src/mailbox.rs`, five in the harness itself.
- The criteria are pinned to the harness by the `r001` unit tests of `bins/ptr-bench` and to the aggregator by
  `scripts/tests/test_r001_aggregate.py`.

This directory holds tests, fixtures, validation cases, or executable checks owned by
`experiments/runtime/R001-mailbox-backpressure`; it is not evidence of implementation by itself.
