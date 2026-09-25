# relational-substrate

PTR contract and candidate comparison for **relational-substrate**. Add benchmarks under `evidence/` and update `candidates.toml`; keep the architecture-facing trait stable.

One relational database hosting the ledger projection, the derived search caches and non-authoritative working state behind PTR-owned adapters. It is never a second authority: the projection is verified against ledger anchors and rebuilt by replay (ADR-0016). The evaluating candidate's command needs `PTR_PG_TEST_DSN` naming a loopback PostgreSQL 16+ server with pgvector 0.7+.

Design: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).
