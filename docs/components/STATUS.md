# PTR Component Implementation Status

> Generated from every `crates/*/component.toml` plus code-derived metrics. Do not hand-edit.  
> Refresh with `python3 scripts/update_component_docs.py --write`.

**Component count:** 24  
**Maturity distribution:** `foundation`: 1, `prototype`: 9, `research-scaffold`: 1, `scaffold`: 13  
**Rust footprint:** 40 source files · 2306 nonblank source lines · 40 integration-test files · 52 `#[test]` markers

| Component | Maturity | Rust files | LOC | Test files | Tests | Implemented items | Missing items | Experiments | Evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| [ptr-config](../../crates/ptr-config/README.md) | `prototype` | 1 | 264 | 4 | 6 | 6 | 3 | E001:planned | — |
| [ptr-core](../../crates/ptr-core/README.md) | `research-scaffold` | 13 | 232 | 1 | 1 | 7 | 7 | M001:planned, M002:planned, M003:planned, M004:planned, M005:planned, M006:planned, M007:planned, E001:planned, E003:planned, E004:planned | model-framework:open, inference-serving:open |
| [ptr-events](../../crates/ptr-events/README.md) | `scaffold` | 1 | 16 | 1 | 1 | 2 | 5 | E001:planned, F001:planned | event-streaming:open |
| [ptr-exec](../../crates/ptr-exec/README.md) | `prototype` | 1 | 46 | 1 | 1 | 4 | 5 | R001:planned, E001:planned, E003:planned | execution-runtime:open |
| [ptr-feedback](../../crates/ptr-feedback/README.md) | `scaffold` | 1 | 44 | 2 | 2 | 3 | 4 | F001:planned, E001:planned | — |
| [ptr-ingress](../../crates/ptr-ingress/README.md) | `scaffold` | 1 | 45 | 1 | 1 | 4 | 4 | E001:planned, E004:planned | ingress-classifier:open |
| [ptr-inspect](../../crates/ptr-inspect/README.md) | `scaffold` | 1 | 23 | 1 | 1 | 3 | 5 | E001:planned | introspection:open |
| [ptr-ledger](../../crates/ptr-ledger/README.md) | `prototype` | 1 | 311 | 2 | 3 | 5 | 5 | L001:running, L002:planned, E004:planned | consensus:open, ledger:open |
| [ptr-memory](../../crates/ptr-memory/README.md) | `scaffold` | 1 | 31 | 1 | 1 | 3 | 5 | Q002:planned, E002:planned, E004:planned | — |
| [ptr-model-api](../../crates/ptr-model-api/README.md) | `scaffold` | 4 | 66 | 2 | 2 | 4 | 5 | M007:planned, E001:planned, E003:planned | inference-serving:open |
| [ptr-net](../../crates/ptr-net/README.md) | `scaffold` | 1 | 14 | 1 | 1 | 3 | 5 | L002:planned, E001:planned | network:open |
| [ptr-observe](../../crates/ptr-observe/README.md) | `scaffold` | 1 | 19 | 1 | 1 | 2 | 5 | F001:planned, E003:planned | observability:open |
| [ptr-pods](../../crates/ptr-pods/README.md) | `prototype` | 1 | 125 | 3 | 4 | 6 | 4 | R002:planned, E001:planned | environment-runtime:open |
| [ptr-protocol](../../crates/ptr-protocol/README.md) | `scaffold` | 2 | 104 | 2 | 2 | 5 | 3 | R002:planned, E001:planned | network-codec:open, local-serialization:open |
| [ptr-router](../../crates/ptr-router/README.md) | `scaffold` | 1 | 30 | 1 | 1 | 2 | 5 | M004:planned, R002:planned, E003:planned | inference-serving:open, distributed-data-compute:open |
| [ptr-runtime](../../crates/ptr-runtime/README.md) | `prototype` | 1 | 304 | 4 | 10 | 7 | 4 | E001:planned, E003:planned, E004:planned | — |
| [ptr-search](../../crates/ptr-search/README.md) | `prototype` | 1 | 94 | 2 | 2 | 4 | 5 | Q001:planned, Q002:planned, E002:planned, E003:planned | lexical-search:open, local-vector-search:open, gpu-vector-search:open, distributed-search:open, structural-code-search:open |
| [ptr-security](../../crates/ptr-security/README.md) | `prototype` | 1 | 22 | 1 | 1 | 3 | 5 | L001:running, E001:planned | security-context:open |
| [ptr-semdb](../../crates/ptr-semdb/README.md) | `prototype` | 1 | 119 | 1 | 2 | 6 | 5 | S001:planned, S002:planned, E004:planned | semantic-db:open |
| [ptr-server](../../crates/ptr-server/README.md) | `prototype` | 1 | 105 | 2 | 1 | 3 | 4 | E001:planned, E003:planned | — |
| [ptr-state](../../crates/ptr-state/README.md) | `scaffold` | 1 | 47 | 2 | 2 | 3 | 3 | L001:running, L002:planned, E004:planned | materialized-state:open |
| [ptr-storage](../../crates/ptr-storage/README.md) | `scaffold` | 1 | 23 | 1 | 1 | 3 | 5 | E004:planned | object-storage:open |
| [ptr-types](../../crates/ptr-types/README.md) | `foundation` | 1 | 124 | 1 | 3 | 5 | 4 | M005:planned, L001:running | — |
| [ptr-verifier](../../crates/ptr-verifier/README.md) | `scaffold` | 1 | 98 | 2 | 2 | 4 | 4 | F001:planned, Q002:planned, E001:planned | code-quality-verifier:open |

## Meaning of maturity labels

- `foundation` — small stable domain core already used broadly.
- `prototype` — executable behavior exists, but major target semantics are still missing.
- `scaffold` — contracts/data structures exist; core production implementation is not present.
- `research-scaffold` — architecture hypothesis is represented, but the trainable system is not implemented/proven.

The labels describe implementation maturity, not scientific novelty or production readiness.

## Freshness policy

Changes under `crates/<component>/src/` or that crate's `Cargo.toml` must update the matching `component.toml` in the same change. CI enforces this metadata coupling, then verifies that generated README/status output is synchronized.
