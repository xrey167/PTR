# Tests for experiments/retrieval/Q002-evidence-promotion

The harness is `ptr-bench evidence-promotion`
([`bins/ptr-bench/src/experiments/q002/`](../../../../bins/ptr-bench/src/experiments/q002/mod.rs)), built in every
configuration. It runs in memory and needs nothing but `ptr-search`, `ptr-memory` and `ptr-runtime`. Its unit tests
(`cargo test -p ptr-bench q002`) cover the world generator (shape, classes and floors for every seed, determinism, every poison
hit really being what its class says), the naive oracle against the generator's labels for every seed, the lifecycle authority
(the real runtime's answers and the harness overlay) and the corrupt-replica source, the rule lists (pinned to `criteria.toml` in
order, every rule resolving to an integer in a reduced run's JSON, every rule able to fail when the emitted counter is set to a
violation, every rule a sentence of the manifest's falsification text), the compiled parameters against `config.toml`,
determinism, the baselines' false Known and the mechanism's clean pass on all five pinned seeds.
`scripts/tests/test_q002_aggregate.py` holds the aggregator's verdicts. The product's own tests are `crates/ptr-search`
(`src/source.rs` unit tests, `tests/source_promotion.rs`, the compile-fail doctests).

[`mutations.toml`](mutations.toml) lists defects planted one at a time in `ptr-search` and in the harness itself, to show the
harness fails on each with a hard counter its plan names:

```sh
python3 scripts/mutation_check.py Q002          # writes results/mutations.json
python3 scripts/mutation_check.py Q002 --check  # anchors only
python3 scripts/mutation_check.py Q002 --only <name> ...  # a few, no record
```

The checker runs `ptr-bench evidence-promotion 36 17` (the pinned world at seed 17; the result says `"scale":"custom"` and is no
evidence). The mutation run edits `crates/ptr-search/src/*.rs` and `bins/ptr-bench/src/experiments/q002/*.rs` in place and
rebuilds `target/release`, so it runs on a quiet tree and never beside another build.

Every edit of the world generator changes the generated corpus and so the pinned digests (`corpus_digest_seed_N` in
`config.toml`, `CORPUS_DIGESTS` in `params.rs`). [`pin_corpus.py`](pin_corpus.py) re-measures them from the harness binary and
rewrites both places (`--check` writes nothing and fails when a pin is stale); re-pin once, after the last generator edit.
