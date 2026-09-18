# ADR-0011 — Semantic Pod Contracts

## Status
Accepted as the Pod abstraction.

## Decision
Models and routers target typed semantic capabilities rather than concrete tool/function names. Provider-specific names are resolved by the Pod registry/adapters.

## Consequences
Unseen-Pod generalization becomes measurable. Pod manifests declare accepted/produced types, effects and protocol version. Runtime capability checks remain independent of model preference.
