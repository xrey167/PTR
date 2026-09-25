# ADR-0016 — PostgreSQL Is a Projection Substrate, Never a Second Authority

## Status
Accepted for the prototype (`ptr-pg`, feature `postgres-backend`). Production use is gated on L004 and on the `relational-substrate` evaluation.

## Decision
One PostgreSQL database may host PTR state in three schema classes with different contracts. **Projection** is a function of the committed ledger prefix: one commit per transaction behind a watermark, with every record's anchor recomputed from the stored one and compared with the ledger's own (`ptr_ledger::integrity::chain_anchors`), so a discarded tail, a restored backup or a foreign log is refused rather than kept. **Derived** caches (search documents, embeddings) hold live generations only; the projector deletes a generation's row in the transaction that supersedes or revokes it, and writers hold the lifecycle row `FOR SHARE` so they cannot interleave. **Work** state (branches, fast-memory journals, lineage, labels) is non-authoritative, not derived from the ledger, and never dropped by a rebuild. Driver types never cross the crate boundary; every refusal is a typed `PgError`.

## Consequences
Rebuild is drop-and-replay (INVARIANT 14). A reader fences on a commit index and is refused, not served stale data, when the projection is behind. Search hits stay search candidates (ADR-0008). Business tables are the external world and are written only through the effect boundary (ADR-0012), which this ADR does not change. Extensions are probed, never created by the substrate; a BM25 extension is an evaluated variant, not a default.
