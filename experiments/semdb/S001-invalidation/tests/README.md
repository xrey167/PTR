# Tests for experiments/semdb/S001-invalidation

The harness is `ptr-bench invalidation`
([`bins/ptr-bench/src/experiments/s001/`](../../../../bins/ptr-bench/src/experiments/s001/mod.rs)), built in every
configuration. It runs in memory and needs nothing but `ptr-semdb`. Its unit tests (`cargo test -p ptr-bench`) cover the
oracle (closure, cycles, conflicts, missing inputs, no-ops, commits), the generator (every topology is a rank-ordered DAG,
every operation valid by construction), the arms against the oracle on every topology, the host-error classification (a refusal of a valid delta, also while a graph is loaded, is a counted falsifier; only errors a wrong harness delta can cause are exit 2), the two negative controls (also at reduced scale for all five seeds), the rule
lists (pinned to `criteria.toml` in order, every rule resolving to an integer in a reduced run's JSON, every rule able to
fail when the emitted counter is set to a violation), determinism (the same seed gives the same JSON apart from the timings)
and the compiled parameters against `config.toml`. `scripts/tests/test_s001_aggregate.py` holds the aggregator's verdicts.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in `ptr-semdb` and in the harness itself, to show the
harness fails on each with a hard counter its plan names:

```sh
python3 scripts/mutation_check.py S001          # writes results/mutations.json
python3 scripts/mutation_check.py S001 --check  # anchors only
```

The checker runs `ptr-bench invalidation 60 17` (60 operations per graph at the pinned sizes; the result says
`"scale":"custom"` and is no evidence). The mutation run edits `crates/ptr-semdb/src/lib.rs` and
`target/release` in place, so it runs on a quiet tree and never beside another build.
