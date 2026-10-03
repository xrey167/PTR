# ptr-config — Typed Configuration Layer

> **Role:** Parse and validate PTR runtime configuration without leaking provider-specific configuration into domain crates.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-09-18  
**Code footprint:** 1 Rust source files · 322 nonblank source lines · 4 integration-test files · 7 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- Typed deserialization/defaults for runtime, semantic, action-boundary, observability and research sections
- Repository default TOML parsing
- Basic configuration validation
- Typed environment override layer for runtime/action/semantic/observability/research settings
- Typed CLI override parser with config-path selection and file→environment→CLI precedence
- Typed server bind configuration with PTR_BIND and --bind overrides
- ServerConfig::data_dir (default data) and a new ModelConfig::backend section (default reference-echo), overridable by PTR_DATA_DIR/--data-dir and PTR_MODEL_BACKEND/--backend, with the same file→environment→CLI precedence
- PtrConfig::validate_daemon, a stricter check for the standalone daemon boundary on top of validate: only runtime.mode standalone, only a loopback server.bind, an explicit server.data_dir that already exists as a directory, and an explicit model.backend. Nothing in the workspace outside ptr-config calls it yet

### Missing for the target architecture

- Per-backend typed configuration sections
- Secret/reference types instead of plaintext secret values
- Config schema generation and migration/version policy

### Next milestones

- Extend typed config to backend-specific adapter sections without leaking secrets
- Generate config schema and example profiles
- Wire ptr-config into ptr-runtime and ptrd

### Linked experiments

- [E001](../../experiments/system/E001-end-to-end/README.md) — `planned`

### Technology evaluations

- None recorded.

### Decision records

- [ADR-0004-backend-independence.md](../../research/decisions/ADR-0004-backend-independence.md)
- [ADR-0014-config-layer.md](../../research/decisions/ADR-0014-config-layer.md)

### Current automated checks

- repository default config parse/validate test
- typed environment override tests
- CLI precedence and unknown-argument tests
- server bind environment/CLI override test
- daemon_validation_rejects_non_loopback_and_missing_persistence (validate_daemon refuses 0.0.0.0 and a missing data_dir, accepts loopback with an existing directory)
- workspace fmt/check/test/clippy

<!-- PTR:STATUS:END -->

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
