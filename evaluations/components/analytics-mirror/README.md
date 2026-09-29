# analytics-mirror

PTR contract and candidate comparison for **analytics-mirror**. Add benchmarks under `evidence/` and update `candidates.toml`; keep the architecture-facing trait stable.

Operational metrics over non-authoritative working records, defined once in `ptr-analytics` and compiled per backend. A columnar mirror is an evaluated option, never a dependency of the projection.

The metric SQL on PostgreSQL is the fresh row-store path. No path is chosen by data volume: another engine (a PostgreSQL extension such as `pg_duckdb`, an Iceberg mirror, or a Rust engine such as DataFusion) is admitted only by this evaluation, when it answers every `MetricSpec` with the same `MetricRow`s as the PostgreSQL SQL on a shared fixture. A Rust engine is admitted only in a crate or feature outside the core build. A Parquet export, a cold tier of it on object storage and DataFusion reading it are deferred until a retention or partitioning decision moves rows analysis still needs out of PostgreSQL, or a measured history scan misses its budget; such an archive keys committed history by commit index under the ledger's retention floor, removes branch records when their branch is erased, and never feeds back into state (doc 35 §7).

Design: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).
