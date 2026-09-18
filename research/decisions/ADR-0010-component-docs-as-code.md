# ADR-0010 — Component Documentation as Code

## Status
Accepted.

## Decision
Each crate owns a machine-readable `component.toml`. Generated README status blocks and the component dashboard are derived from it and verified by CI.

## Consequences
Implementation/missing/experiment/evaluation/ADR status cannot drift silently from documentation. Hand-written architecture remains outside generated markers.
