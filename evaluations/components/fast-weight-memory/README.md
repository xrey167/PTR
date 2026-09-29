# fast-weight-memory

PTR contract and candidate comparison for **fast-weight-memory**. Add benchmarks under `evidence/` and update `candidates.toml`; keep the architecture-facing trait stable.

Session working memory that is a derived, exactly revocable projection of lifecycle-managed inputs (ADR-0018). Candidates are compared on update-aware recall, Unknown rate and false recall at an equal token budget (M008), and on revocation exactness (L003).

Design: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).
