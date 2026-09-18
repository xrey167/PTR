# ADR-0012 — Hard Effect Boundary

## Status
Accepted.

## Decision
Uncertain reasoning may propose actions, but execution requires typed ActionIR plus capability, permission, freshness and required verification checks.

## Consequences
A model confidence score cannot authorize mutation. External/irreversible effects can require stronger verification/durability than reads. Denials/escalations are explicit runtime results.
