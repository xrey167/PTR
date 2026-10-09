# Tests for experiments/feedback/F001-generate-verify-repair

The harness is `ptr-bench generate-verify-repair`
([`bins/ptr-bench/src/experiments/f001/`](../../../../bins/ptr-bench/src/experiments/f001/mod.rs)), built in every
configuration. It runs in memory and needs nothing but `ptr-feedback` and `ptr-verifier`. Its unit tests
(`cargo test -p ptr-bench f001`) cover the independent oracle (the contract check, the required fabric report per defect set and
noise mode, the harness' own acceptance rule), the stubs (seeded streams, the generator's labelled defects against the oracle, a
repair that edits only the named slots, the boundary that logs every call and forges only a `Fail`), the exact analytic
expectation (a closed form for the rewrite arm, a brute-force check of the moments), the significance tests, the two negative controls
(also for all five seeds), the rule lists (pinned to `criteria.toml` in order, every rule resolving to an integer in a reduced run's
JSON, every rule able to fail when the emitted counter is set to a violation), determinism, and the compiled parameters against
`config.toml`. The crates' own tests (`cargo test -p ptr-feedback -p ptr-verifier`) cover the loop and the acceptance rule.
`scripts/tests/test_f001_aggregate.py` holds the aggregator's verdicts.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in `ptr-feedback`, `ptr-verifier` and the harness itself, to
show the harness fails on each with a hard counter its plan names:

```sh
python3 scripts/mutation_check.py F001          # writes results/mutations.json
python3 scripts/mutation_check.py F001 --check  # anchors only
```

The checker runs `ptr-bench generate-verify-repair 400 17` (400 tasks per profile; the result says `"scale":"custom"` and is no
evidence). The mutation run edits `crates/ptr-feedback`, `crates/ptr-verifier` and the harness in place and `target/release`, so it
runs on a quiet tree and never beside another build.
