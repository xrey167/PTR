# ptr-config — Typed Configuration Layer

> **Role:** Parse and validate PTR runtime configuration without leaking provider-specific configuration into domain crates.

## Position in PTR

```mermaid
flowchart LR
  F["TOML / env / CLI"] --> C["ptr-config"]
  C --> R["ptr-runtime / ptrd"]
```

Dedicated diagram source: [`docs/diagrams/components/ptr-config.mmd`](../../docs/diagrams/components/ptr-config.mmd)

## Mission

Provide typed defaults and validation for runtime-wide configuration. Backend-specific adapters may extend configuration behind their own contracts, but the central precedence and validation rules remain PTR-owned.

## Responsibilities

- parse repository/default TOML;
- validate basic invariants;
- expose typed runtime/semantic/action/observability/research config.

## Explicit non-responsibilities

- secrets storage;
- business policy;
- backend implementation selection evidence.

## Related architecture

- [Technical architecture](../../docs/TECHNICAL_ARCHITECTURE.md)
- [Technology stack](../../docs/TECH_STACK.md)
