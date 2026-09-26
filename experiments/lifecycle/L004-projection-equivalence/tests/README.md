# Tests for experiments/lifecycle/L004-projection-equivalence

The harness is `ptr-bench projection-equivalence`
([`bins/ptr-bench/src/experiments/l004.rs`](../../../../bins/ptr-bench/src/experiments/l004.rs)),
built with the `postgres-experiments` feature (or `turso-oracle`, which adds Turso as
a third implementation). It reads a loopback PostgreSQL server from
`PTR_PG_EXPERIMENT_DSN` and exits 1 on any hard failure.

Commit faults are deterministic: a deferred constraint trigger, armed from a
separate session around one pre-drawn record, fails that record's transaction at
COMMIT after every statement succeeded. A projector that reports the record applied,
or leaves any of it visible, fails the run. Crash outcomes (whether a killed
transaction had committed) still depend on timing, which is why every seed must
see both.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in the
projector to show the harness fails on them:

```sh
PTR_PG_EXPERIMENT_DSN="host=127.0.0.1 port=5432 user=postgres" \
  python scripts/mutation_check.py L004          # writes results/mutations.json
python scripts/mutation_check.py L004 --check    # anchors only; runs in CI
```

CI lints the harness and runs it for two cases against the `pgvector/pgvector:pg18`
service; archived results come only from the seeds in `experiment.toml`.

Seed runs and a mutation run that writes `results/mutations.json` start only from a
working tree whose sources HEAD holds, and `aggregate.py` refuses records or mutation
evidence whose commit differs from the checkout in code, recording scripts, the
aggregator or, for mutation evidence, the mutation plan ([`scripts/experiment_records.py`](../../../../scripts/experiment_records.py)).
A hard pass needs every seed to exit 0 with no hard failure and to reach every probe. Seeds run without Turso
(`postgres_entrypoint`) aggregate to the verdict `no-turso-oracle`, never to a hard pass.
