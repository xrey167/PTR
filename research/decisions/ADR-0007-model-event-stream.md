# ADR-0007 — Model Event Stream

## Status
Accepted as the long-term model/runtime interface.

## Decision
The PTR model interface is a stream of typed ModelEvents rather than only a final text string. Text-only inference backends are supported as a reduced capability mode.

## Consequences
The runtime can react to operator/Pod/action events before final text completion. Backend capability negotiation is required. Neural events remain proposals until runtime verification/authorization.
