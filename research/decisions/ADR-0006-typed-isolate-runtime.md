# ADR-0006 — Typed Isolate Runtime

## Status
Accepted as the runtime semantic model; backend remains benchmark-driven.

## Decision
PTR runtime components use single-owner state machines with bounded typed mailboxes, explicit backpressure, supervision and cancellation. Tokio is a substrate, not the semantic contract.

## Consequences
No core path may depend on an implicit unbounded queue. Saturation returns ownership/failure explicitly. Remote and local Pods must preserve equivalent effect and cancellation semantics.
