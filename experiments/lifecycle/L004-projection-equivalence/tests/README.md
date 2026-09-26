# Tests for experiments/lifecycle/L004-projection-equivalence

The harness is `ptr-bench projection-equivalence`
([`bins/ptr-bench/src/experiments/l004.rs`](../../../../bins/ptr-bench/src/experiments/l004.rs)),
built with the `postgres-experiments` feature (or `turso-oracle`, which adds Turso as
a third implementation). It reads a loopback PostgreSQL server from
`PTR_PG_EXPERIMENT_DSN` and exits 1 on any hard failure.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in the
projector to show the harness fails on them:

```sh
PTR_PG_EXPERIMENT_DSN="host=127.0.0.1 port=5432 user=postgres" \
  python scripts/mutation_check.py L004          # writes results/mutations.json
python scripts/mutation_check.py L004 --check    # anchors only; runs in CI
```

CI lints the harness and runs it for two cases against the `pgvector/pgvector:pg18`
service; archived results come only from the seeds in `experiment.toml`.
