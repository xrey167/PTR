# Tests for experiments/semdb/S002-snapshot-cancellation

The harness is `ptr-bench snapshot-cancellation`
([`bins/ptr-bench/src/experiments/s002/`](../../../../bins/ptr-bench/src/experiments/s002/mod.rs)), built in every configuration.
It runs in memory and needs nothing but `ptr-runtime` and the crates it drives. Its unit tests (`cargo test -p ptr-bench`) cover the
revision mirror and the ledger scan (`oracle.rs`), the schedule (`plan.rs`: determinism, the mix, every position and effect), the
fixture (`world.rs`), the scheduler and the arms against the oracle on hand-built cases (`runner.rs`), the two negative controls, the
panic guard, the rule lists (pinned to `criteria.toml` in order, every rule resolving to an integer in a reduced run's JSON, every rule
able to fail when the emitted counter is set to a violation), determinism (the same seed gives the same JSON apart from the timings)
and the compiled parameters against `config.toml`. `scripts/tests/test_s002_aggregate.py` holds the aggregator's verdicts. The
runtime's own tests are `crates/ptr-runtime/tests/stepped_run.rs` and the unit tests of `src/stepped.rs`.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in `ptr-security`, `ptr-runtime` and the harness itself, to
show the harness fails on each with a hard counter its plan names:

```sh
python3 scripts/mutation_check.py S002          # writes results/mutations.json
python3 scripts/mutation_check.py S002 --check  # anchors only
```

The checker runs `ptr-bench snapshot-cancellation 300 17` (300 cases; the result says `"scale":"custom"` and is no evidence). The
mutation run edits `crates/ptr-security/src/lib.rs`, `crates/ptr-runtime/src/{lib,execution,stepped}.rs` and two harness files in
place and rebuilds `target/release`, so it runs on a quiet tree and never beside another build. The `cancel-hangs` mutation takes the
watchdog's 60 seconds.
