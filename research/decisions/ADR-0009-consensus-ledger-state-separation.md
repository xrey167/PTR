# ADR-0009 — Separate Consensus, Ledger, Materialized State and Event Streaming

## Status
Accepted.

## Decision
Consensus decides cluster order, the ledger stores authoritative history, materialized state serves current queries, and event streaming distributes projections/telemetry. No one subsystem substitutes for the others.

## Consequences
raft-rs, raft-engine, Turso and Iggy are candidates for different roles. Standalone mode preserves the same ledger/state contracts without distributed consensus.
