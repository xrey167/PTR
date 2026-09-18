# ADR-0008 — Search Is a Derived Projection

## Status
Accepted.

## Decision
Lexical/vector/graph/structural indexes are disposable projections of lifecycle-managed semantic state. A retrieval hit is evidence candidate data, never authority.

## Consequences
Every hit carries source/generation metadata, resolves to exact evidence, and passes verification before promotion. Index lag after revocation is tolerated only because stale generations are rejected.
