# Tests for experiments/lifecycle/L003-fastmem-revocation

The harness is `ptr-bench fastmem-revocation`
([`bins/ptr-bench/src/experiments/l003.rs`](../../../../bins/ptr-bench/src/experiments/l003.rs)),
built with the `postgres-experiments` feature. It reads a loopback PostgreSQL server
from `PTR_PG_EXPERIMENT_DSN` and exits 1 on any hard failure.

[`mutations.toml`](mutations.toml) lists defects planted one at a time in the
projector's cascade, the journal and checkpoint store, and `ptr-fastmem` itself to
show the harness fails on them:

```sh
PTR_PG_EXPERIMENT_DSN="host=127.0.0.1 port=5432 user=postgres" \
  python scripts/mutation_check.py L003          # writes results/mutations.json
python scripts/mutation_check.py L003 --check    # anchors only; runs in CI
```

CI lints the harness and runs it for two cases against the `pgvector/pgvector:pg18`
service; archived results come only from the seeds in `experiment.toml`.
