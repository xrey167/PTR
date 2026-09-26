# relational-substrate

PTR contract and candidate comparison for **relational-substrate**. Add benchmarks under `evidence/` and update `candidates.toml`; keep the architecture-facing trait stable.

One relational database hosting the ledger projection, the derived search caches and non-authoritative working state behind PTR-owned adapters. It is never a second authority: the projection is verified against ledger anchors and rebuilt by replay (ADR-0016). The evaluating candidate's command needs `PTR_PG_TEST_DSN` naming a loopback PostgreSQL 16+ server with pgvector 0.8+.

Candidates are servers, not drivers: `ptr-pg` uses `tokio-postgres` without an ORM, and the driver choice is reopened here only if TLS, pooling or role separation cannot be built on it within MSRV 1.85. Whether concurrent metric queries raise projector or branch-write latency at a stated working-record volume is measured here; until a measurement shows it, nothing (a reader pool, a logical-replication replica, a columnar mirror) is added to isolate that load.

Design: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).
