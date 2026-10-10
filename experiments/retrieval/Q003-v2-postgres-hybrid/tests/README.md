# Tests for experiments/retrieval/Q003-v2-postgres-hybrid

The harness is `ptr-bench postgres-hybrid`
([`bins/ptr-bench/src/experiments/q003/`](../../../../bins/ptr-bench/src/experiments/q003/mod.rs)), built with the
`postgres-experiments` feature. It drives `ptr-pg` against a loopback PostgreSQL 16+ server with pgvector 0.8+
(`PTR_PG_EXPERIMENT_DSN`) and two or three Python drivers over a JSON-lines bridge. Its unit tests
(`cargo test -p ptr-bench --features postgres-experiments q003`) cover the workload generator (mix, floors and distinctness for every
seed, determinism, plants and late writes never live), the harness's live table, the ranking and standard-error arithmetic, the
frozen inputs (hashes and the manifest), the JSON reader and the bridge (a driver that exits, stays silent or refuses is an error,
not a hang), the argument parser, the rule lists (pinned to `criteria.toml` in order, every rule a sentence of the manifest's
falsification text), the compiled parameters against `config.toml` and the mutation plan against the harness's hard counters.
Seven more tests need a server and Python and are `#[ignore]`d (`-- --include-ignored`): a reduced end-to-end run, every rule resolving
to an integer in its JSON, every rule able to fail when the emitted counter is set to a violation, the product holding the
lifecycle rules on the reduced world, determinism, and timings never being read. `scripts/tests/test_q003_aggregate.py` holds the
aggregator's verdicts and the re-pin script's behaviour. The product's own tests are `crates/ptr-pg` (`tests/postgres.rs`,
the search tests) and `crates/ptr-search` (`src/fusion.rs`).

[`mutations.toml`](mutations.toml) lists defects planted one at a time in `ptr-pg`, `ptr-search` and the harness itself, to show the
harness fails on each with a hard counter its plan names:

```sh
python3 scripts/mutation_check.py Q003-v2          # writes results/mutations.json
python3 scripts/mutation_check.py Q003-v2 --check  # anchors only
python3 scripts/mutation_check.py Q003-v2 --only <name> ...  # a few, no record
```

The checker runs `ptr-bench postgres-hybrid 400 17` (400 queries at seed 17; the result says `"scale":"custom"`, runs no strong arm and is no evidence; the generator mutation runs the pinned 1000). The mutation run edits `crates/ptr-pg/src/adapters/search.rs`,
`crates/ptr-search/src/fusion.rs` and `bins/ptr-bench/src/experiments/q003/*.rs` in place and rebuilds `target/release`, so it runs on
a quiet tree and never beside another build; it needs `PTR_PG_EXPERIMENT_DSN`.

Every edit of the workload generator changes the generated workload and so the pinned digests (`workload_digest_seed_N` in
`config.toml`, `WORKLOAD_DIGESTS` in `params.rs`). [`pin_workload.py`](pin_workload.py) re-measures them from the harness binary
(`--print-workload-digest`, no server needed) and rewrites both places (`--check` writes nothing and fails when a pin is stale).
