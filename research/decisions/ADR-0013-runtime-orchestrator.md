# ADR-0013 — Runtime Orchestrator

## Status
Accepted.

## Decision
PTR has a dedicated `ptr-runtime` composition layer. Binaries bootstrap it; domain crates do not orchestrate the complete request lifecycle themselves.

## Consequences
The daemon remains thin, integration tests can exercise one runtime object, and model/search/Pod backends remain replaceable behind their contracts.
