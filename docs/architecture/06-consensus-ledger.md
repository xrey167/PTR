# Consensus, Ledger and State

Cluster mode combines a consensus module (candidate: raft-rs), durable log storage (candidate: raft-engine), and a queryable materialized state (candidate: Turso). Standalone mode does not require distributed consensus.

Only committed events can update authoritative semantic state. Search indexes and caches consume committed materializations asynchronously. Revocation barriers and snapshot/consumer barriers guard log compaction.
