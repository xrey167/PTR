# Tests for experiments/model/M008-fast-weight-memory

The harness is `ptr-bench fast-weight-memory`
([`bins/ptr-bench/src/experiments/m008/`](../../../../bins/ptr-bench/src/experiments/m008/mod.rs)), built in every
configuration. It reads the frozen pool and vectors of `research/embeddings` (sha256 pinned, token counts recounted),
runs the code under test (`ptr-fastmem`) and the pinned `rag_reference` driver through
[`rag_bridge.rs`](../../../../bins/ptr-bench/src/experiments/rag_bridge.rs), and needs `python3` (standard library only).
Its unit tests (`cargo test -p ptr-bench m008 rag_bridge`) cover the pool loader (digests, half-float conversion, the
token recount), the seeded session, the oracle's three readings, scoring, the paired bootstrap, the abstention policy
and the pilot's selection rule, the rule lists (pinned to `criteria.toml` in order, every rule resolving to an integer in
a real run's JSON, every rule able to fail), the compiled parameters against `config.toml`, determinism, and a full run
of a pilot seed at the pinned scale. `scripts/tests/test_m008_aggregate.py` holds the aggregator's verdicts.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in `ptr-fastmem` and in the harness itself:

```sh
python3 scripts/mutation_check.py M008          # writes results/mutations.json
python3 scripts/mutation_check.py M008 --check  # anchors only
python3 scripts/mutation_check.py M008 --only <name> ...  # a few, no record
```

The checker runs `ptr-bench fast-weight-memory 400 2001` (the pinned session at seed 2001, which is neither a
confirmatory nor a pilot seed; the result says `"scale":"custom"` and is no evidence). It edits
`crates/ptr-fastmem/src/*.rs` and `bins/ptr-bench/src/experiments/m008/*.rs` in place and rebuilds `target/release`,
so it runs on a quiet tree and never beside another build.

[`pilot_calibration.json`](pilot_calibration.json) is the output of `ptr-bench fast-weight-memory calibrate 1001 1002
1003 1004 1005`, from which the memory shape and the abstention thresholds of `config.toml` were taken
(procedure in `m008/calibrate.rs` and PREREGISTRATION.md). It is pilot evidence only.
