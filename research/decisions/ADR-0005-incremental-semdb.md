# ADR-0005 — Incremental Semantic Database

## Status
Accepted as the target architecture; implementation remains experimental.

## Decision
PTR semantic state is organized as revisioned ground inputs plus dependency-tracked derived queries and immutable snapshots, following incremental-compiler principles.

## Consequences
Local changes should invalidate only dependent semantics. A reasoning run sees one snapshot revision. External side effects and non-deterministic Pods are not hidden inside pure semantic queries; observations re-enter as new ground inputs.
