# ptr-bench

Reference microbenchmarks and lifecycle probes. Run `cargo run -p ptr-bench -- all`.
These probes do not establish model quality or production readiness.

Dependency setup now uses SemanticDelta.dependencies and timed updates use the
fallible typed-value/staged invalidation API. Historical timings of the old
in-place string map are not equivalent baselines. Measure the exact revision.
See [semantic durability](../../docs/architecture/22-durable-semantic-state.md).

The recovery harness now generates real partial PTRLOG02 frames and requires an
anchor derived from its independently known fixture. It does not derive trust
from the damaged file. Ordinary open rejects incomplete tails without rewriting.
Old v1 recovery timings are not comparable to v2 integrity/anchor validation.

## PostgreSQL experiment harnesses

With the `postgres-experiments` feature the binary also runs the harnesses of
[L003](../../experiments/lifecycle/L003-fastmem-revocation/README.md) and
[L004](../../experiments/lifecycle/L004-projection-equivalence/README.md) against a
loopback PostgreSQL 16+ server with pgvector 0.8+ named by `PTR_PG_EXPERIMENT_DSN`:

```sh
export PTR_PG_EXPERIMENT_DSN="host=127.0.0.1 port=5432 user=postgres"
cargo run --release -p ptr-bench --features postgres-experiments -- fastmem-revocation 30 17
cargo +stable run --release -p ptr-bench --features turso-oracle -- projection-equivalence 40 17
```

Each prints one JSON line and exits 1 on any hard failure. Every case gets its own
schemas, tagged with the case so a crash can terminate exactly its sessions, and
drops them when it ends. The feature stays out of default builds, so the default
binary keeps its dependencies and toolchain.
