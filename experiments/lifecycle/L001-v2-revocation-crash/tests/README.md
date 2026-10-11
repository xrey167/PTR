# Tests for experiments/lifecycle/L001-v2-revocation-crash

The harness is `ptr-bench revocation-crash`
([`bins/ptr-bench/src/experiments/l001/`](../../../../bins/ptr-bench/src/experiments/l001/mod.rs)), built with the
`ledger-experiments` feature of `ptr-bench` (which turns on the failpoints of `ptr-ledger`: the log's, the anchor's and
the vendored raft-engine's). It needs no network and no server. Its unit tests (`cargo test -p ptr-bench --features
ledger-experiments --locked`) cover the oracle against a table written by hand, the workload generator (6 to 14 events,
3 or 4 subjects, a revoke-then-recommit in every history), the crash-image synthesizer (the synced prefix is never
touched, every image is in the honest model, an image outside it is refused), the judgement of one recovery (each
falsifier fires on the recovery it names and on no other), the contained call (a panic and a hang are counted), the
durability probe and its calibration, the router, the rule lists (pinned to `criteria.toml` in order, every rule resolving
to an integer in a reduced run, every rule able to fail when the emitted counter is set to a violation), the controls
(a lie that does not fool the naive recovery is a harness failure), determinism and the compiled parameters against
`config.toml`. `scripts/tests/test_l001v2_aggregate.py` holds the aggregator's verdicts.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in `ptr-ledger`, `ptr-runtime` and the harness
itself, to show the harness fails on each with a hard counter its plan names:

```sh
python3 scripts/mutation_check.py L001-v2          # writes results/mutations.json
python3 scripts/mutation_check.py L001-v2 --check  # anchors only
```

The checker runs `ptr-bench revocation-crash 2 1001` (two workloads per arm, child processes included; the result says
`"scale":"custom"` and is no evidence). The mutation run edits `crates/ptr-ledger`, `crates/ptr-runtime` and the harness
in place and rebuilds `target/release`, so it runs on a quiet tree and never beside another build or another mutation
checker. The directory it uses must be on a disk-backed filesystem (the durability probe says so itself when it is not).
