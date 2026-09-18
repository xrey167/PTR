# ADR-0014 — Typed Configuration Layer

## Status
Accepted.

## Decision
PTR configuration is parsed and validated into `ptr-config` domain structs. Precedence is defaults → file → environment → CLI once all layers are implemented.

## Consequences
Runtime code does not parse ad-hoc environment variables or raw TOML. Provider-specific secrets are referenced, not embedded, and future schema/version migration remains centralized.
