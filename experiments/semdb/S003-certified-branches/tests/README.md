# Tests for experiments/semdb/S003-certified-branches

The harness is `ptr-bench certified-branches`
([`bins/ptr-bench/src/experiments/s003/`](../../../../bins/ptr-bench/src/experiments/s003/mod.rs)),
built in every configuration. It runs in memory and exits 1 on any hard failure. Its
unit tests (`cargo test -p ptr-bench`) and `bins/ptr-bench/tests/smoke.rs` run in CI on
Linux and Windows, at the stable and the minimum supported toolchain.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in certification,
the runtime's merge and write paths, the ledger, the state projection and the harness's own
oracle, model and programs, to show the harness fails on each with a hard counter its plan
names:

```sh
python scripts/mutation_check.py S003          # writes results/mutations.json
python scripts/mutation_check.py S003 --check  # anchors only; runs in CI
```

Where the writer-side `validate_semantic_origin` backs up a runtime check (a branch merged
once, ingress keys), the mutation removes both layers, because the harness checks outcomes
and a defect in one layer alone is caught by the other; each layer has its own unit test in
`crates/ptr-runtime/tests/`. The mutation run edits shared sources and `target/release` in
place, so it runs on a quiet tree and never beside another build.

`aggregate.py` reads the five seeds' run records and `mutations.json`, refuses records of
other code or another preregistration, and publishes `results/metrics.json` and
`results/run.json` as one aggregate. `aggregate.py --pilot` reads the pilot outputs in
`pilot/` and prints the `low_cells` list the freeze pins.
