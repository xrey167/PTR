# Tests for experiments/system/E007-split-pod-kv

The harness is `ptr-bench split-pod-kv`
([`bins/ptr-bench/src/experiments/e007/`](../../../../bins/ptr-bench/src/experiments/e007/mod.rs)). It needs no network, no server and no
feature flag. Its unit tests (`cargo test -p ptr-bench --locked e007`) cover the compiled parameters against `config.toml` (drift in
either direction), the relations the design needs (an internal split, the replay window, the 16 MiB frame in five chunks inside the reorder
window, the frame rate against the tombstone table), the rule lists (pinned to `criteria.toml` in order, every rule resolving in a reduced
run, every rule able to fail when its counter is set to a violation), the mutation plan's hard counters (all top-level integers of the
record, each expected by at least one mutation), the independent digest and KV scan, determinism (the same seed gives the same record apart
from timing), the exit-status order (validity, then a falsifier, then the rest) and the two branches of the standing check (a case that cannot run or a control that does not trip ends the run with status 2 unless an observed product violation fired, which prints the record with status 1), and the controls. The product has its own tests:
`crates/ptr-pods/tests/qwen_stage.rs`, `crates/ptr-protocol/src/activation.rs`, `crates/ptr-podwire/src/chunk.rs` and `tests/chunking.rs`,
`crates/ptr-runtime/src/split_stage.rs` and `tests/split_stage.rs`. `scripts/tests/test_e007_aggregate.py` holds the aggregator's verdicts.

[`mutations.toml`](mutations.toml) lists 59 defects planted one at a time in `ptr-podwire`, `ptr-protocol`, `ptr-runtime`, `ptr-pods` and the
harness itself, to show the harness fails on each with a hard counter its plan names (status 1; the five mutations of the harness's own validity say `expect_exit = 2` and are killed by status 2 with their counter):

```sh
python3 scripts/mutation_check.py E007          # writes results/mutations.json
python3 scripts/mutation_check.py E007 --check  # anchors only
python3 scripts/mutation_check.py E007 --only chunk-digest-unchecked
```

The checker runs `ptr-bench split-pod-kv 2 1001` (two corpus sequences, every probe and attack in full; the result says `"scale":"custom"` and is
no evidence). The mutation run edits the crates and the harness in place and rebuilds `target/release`, so it runs on a quiet tree and never
beside another build or another mutation checker.
