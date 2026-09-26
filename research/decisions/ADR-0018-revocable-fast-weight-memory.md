# ADR-0018 — Fast-Weight Working Memory Is a Derived, Exactly Revocable Projection

## Status
Accepted as an experiment (`ptr-fastmem`). Revocation exactness passed L003 on PostgreSQL 18 (2026-09-26; five seeds, no bit difference). Usefulness is still gated on M008.

## Decision
A fast memory is a multi-head matrix state updated by the gated delta rule in its Kimi Delta Attention form and read by `o = S^T q`. Every write names the semantic input, generation and input digest it came from and is journaled exactly as composed. A read is admitted only while every source is still admissible. Revoking a source removes its writes and refolds from the last checkpoint before the first removed write; the fold is deterministic `f32` arithmetic, so the result is bit-identical to a memory that never saw them. Readouts are decoded against identifier codes of the facts actually written into search candidates, or into `Unknown` when no fact leads by a margin. The Postgres substrate deletes a revoked or superseded generation's journal writes and the checkpoints that folded them in the projector's transaction, stores a checkpoint only when its binding digest matches the stored journal prefix it folds, and never returns one that no longer does. The binding digest names the folded writes by source, not their bits; the journal keeps the writes as composed, because re-admitting a normalised key can change its bits.

## Consequences
The memory is never authority and never a higher evidence stage than a search hit. The guarantee is exact revocation, not erasure: copies in storage pages, WAL and backups are the storage's to erase (doc 25). Checkpoints are full `f32`; a narrower encoding would break the bit-identity promise.
