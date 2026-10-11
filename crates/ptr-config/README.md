# ptr-config — Typed Configuration Layer

> **Role:** Parse and validate PTR runtime configuration without leaking provider-specific configuration into domain crates.

<!-- PTR:STATUS:BEGIN -->
## Current implementation status

> **Generated section.** Source of truth: [`component.toml`](component.toml) plus code-derived metrics from `src/`. Run `python3 scripts/update_component_docs.py --write` after editing implementation metadata. Do not hand-edit inside this block.

**Maturity:** `prototype`  
**Last reviewed:** 2026-10-10  
**Code footprint:** 2 Rust source files · 1139 nonblank source lines · 6 integration-test files · 28 test markers (`#[test]`, `#[tokio::test]`)

### Implemented now

- Typed deserialization/defaults for runtime, semantic, action-boundary, observability and research sections
- Repository default TOML parsing
- Basic configuration validation
- Typed environment override layer for runtime/action/semantic/observability/research settings
- Typed CLI override parser with config-path selection and file→environment→CLI precedence
- Typed server bind configuration with PTR_BIND and --bind overrides
- ServerConfig::data_dir (default data) and a new ModelConfig::backend section (default reference-echo), overridable by PTR_DATA_DIR/--data-dir and PTR_MODEL_BACKEND/--backend, with the same file→environment→CLI precedence
- PtrConfig::validate_daemon, a stricter check for the standalone daemon boundary on top of validate: only runtime.mode standalone, only a loopback server.bind, an explicit server.data_dir that already exists as a directory, and an explicit model.backend. Nothing in the workspace outside ptr-config calls it yet
- The [vcw] section (VcwServiceConfig, src/vcw.rs, default OFF): enabled, project, the task kinds served (arithmetic, json_object, code, fact), verifier-call and round budgets, hint mode (trace-only or apply-to-schema), the router exploration floor, one Pod id per role, a per-request wall-clock deadline max_request_ms (default 30000, 1..=120000), budgets capped at 64, and the expert backend, which has no default (an enabled section without vcw.experts.backend is refused, so a deployment never serves simulated answers by omission): simulated (seed, per-code error rates, repair success per code and per Pod, decline-on-failure) or qwen-cpu (model directory and the four file SHA-256 digests, from file or the PTR_VCW_QWEN_* environment variables). Its shape is validated whether or not it is enabled; the qwen-cpu requirements (directory, four lowercase hex digests, only arithmetic and json_object kinds, token bound) while it is enabled with that backend. PTR_VCW_ENABLED switches it from the environment; there is no CLI switch. Consumed by ptr-server (VcwService::from_config) and ptrd; covered by tests/vcw.rs
- [vcw.pods.extra_repairers] (role name format|value|schema|code|resource|fact to a list of further repair Pod ids, default empty, VCW_REPAIR_ROLES): several eligible repairers per finding code so the router samples among them; a Pod under several roles covers all their codes; an unknown role, the proposer Pod or an invalid id fails validation
- Every configuration section refuses unknown keys (deny_unknown_fields), so a misspelled key or section fails the load instead of silently falling back to a default, and PtrConfig::from_path validates the loaded configuration before returning it
- PtrConfig::check_supported_settings (run by validate, so file, environment and CLI loads all go through it) refuses with the typed UnsupportedSetting any value the system cannot honour: semantic.retain_raw_evidence=false (raw is never replaced by typed state) and a non-default value of the settings no component reads yet (observability.structured_tracing, observability.async_backtrace, research.allow_unlocked_components, research.require_experiment_manifest, runtime.max_parallel_candidates; 'not supported yet'). FIELD_DISPOSITIONS lists every leaf field as Consumed or RefusedUnlessDefault (semantic.cancel_stale_revisions is Consumed by ptr-runtime's steppable run API only: false is accepted from file and environment and switches the cooperative stale-revision cancellation off, true is the default; NO SHIPPED HOST (ptr-server, ptrd) STEPS RUNS and the blocking entry points force the cancellation off, so the setting has no effect on any production path today) and a test fails when a new field is missing from it
- [vcw] learning = "off" | "shadow" (VcwLearning, default off): shadow makes the vcw path learn a routing policy from the verifier's verdicts and log off-policy estimates, never route by it; validate refuses shadow with exploration_floor = 0 (the estimates rest on the positivity the floor guarantees); a FIELD_DISPOSITIONS row, config/default.toml spells it out, unknown values are refused

### Missing for the target architecture

- Per-backend typed configuration sections
- Secret/reference types instead of plaintext secret values
- Config schema generation and migration/version policy
- The five refused-unless-default settings (observability.structured_tracing, observability.async_backtrace, research.allow_unlocked_components, research.require_experiment_manifest, runtime.max_parallel_candidates; semantic.retain_raw_evidence is refused for the invariant, not for want of a consumer) stay inert: wiring a consumer (and then lifting the refusal) is open work in ptr-runtime and the observability setup

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
- a_misspelled_key_is_refused_instead_of_silently_ignored and loading_from_a_path_validates_the_result
- workspace fmt/check/test/clippy
- tests/unsupported.rs: field table matches the serialized default config, every refused field accepts its default and refuses a changed value with a field-specific error, plus file, environment and CLI paths and retain_raw_evidence=false; stale_revision_cancellation_can_be_switched_off_by_file_and_environment (semantic.cancel_stale_revisions=false is accepted and the field is Consumed)
- tests/vcw.rs: learning is off by default, shadow parses and validates, an unknown mode and shadow with a zero exploration floor are refused

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
