# PTR Component Implementation Status

> Generated from every `crates/*/component.toml` plus code-derived metrics. Do not hand-edit.  
> Refresh with `python3 scripts/update_component_docs.py --write`.

**Component count:** 33  
**Maturity distribution:** `foundation`: 1, `prototype`: 21, `research-scaffold`: 1, `scaffold`: 10  
**Rust footprint:** 181 source files · 65412 nonblank source lines · 151 integration-test files · 1516 test markers (`#[test]`, `#[tokio::test]`)

| Component | Maturity | Rust files | LOC | Test files | Tests | Implemented items | Missing items | Experiments | Evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| [ptr-analytics](../../crates/ptr-analytics/README.md) | `prototype` | 8 | 1303 | 1 | 48 | 8 | 2 | F002:planned, F003:planned | analytics-mirror:open |
| [ptr-branch](../../crates/ptr-branch/README.md) | `prototype` | 7 | 3799 | 3 | 112 | 26 | 7 | S003:planned, F003:planned | — |
| [ptr-cluster](../../crates/ptr-cluster/README.md) | `prototype` | 3 | 726 | 1 | 15 | 13 | 5 | — | — |
| [ptr-config](../../crates/ptr-config/README.md) | `prototype` | 1 | 322 | 4 | 7 | 8 | 3 | E001:planned | — |
| [ptr-core](../../crates/ptr-core/README.md) | `research-scaffold` | 13 | 200 | 1 | 1 | 9 | 6 | M001:planned, M002:planned, M003:planned, M004:planned, M005:planned, M006:planned, M007:planned, E001:planned, E003:planned, E004:planned | model-framework:open, inference-serving:open |
| [ptr-events](../../crates/ptr-events/README.md) | `scaffold` | 2 | 249 | 2 | 8 | 5 | 4 | E001:planned, F001:planned | event-streaming:open |
| [ptr-exec](../../crates/ptr-exec/README.md) | `prototype` | 1 | 46 | 1 | 1 | 4 | 5 | R001:planned, E001:planned, E003:planned | execution-runtime:open |
| [ptr-execwire](../../crates/ptr-execwire/README.md) | `prototype` | 5 | 1597 | 3 | 36 | 18 | 9 | — | — |
| [ptr-fastmem](../../crates/ptr-fastmem/README.md) | `prototype` | 9 | 3189 | 4 | 85 | 13 | 3 | M008:planned, L003:completed | fast-weight-memory:open |
| [ptr-feedback](../../crates/ptr-feedback/README.md) | `scaffold` | 1 | 44 | 2 | 2 | 3 | 4 | F001:planned, E001:planned | — |
| [ptr-ingress](../../crates/ptr-ingress/README.md) | `scaffold` | 1 | 45 | 1 | 1 | 4 | 4 | E001:planned, E004:planned | ingress-classifier:open |
| [ptr-inspect](../../crates/ptr-inspect/README.md) | `scaffold` | 1 | 48 | 1 | 5 | 3 | 5 | E001:planned | introspection:open |
| [ptr-labeling](../../crates/ptr-labeling/README.md) | `prototype` | 6 | 1888 | 1 | 36 | 12 | 5 | F002:planned | label-model:open |
| [ptr-ledger](../../crates/ptr-ledger/README.md) | `prototype` | 9 | 6234 | 14 | 119 | 47 | 8 | L001:running, L002:planned, E004:planned | consensus:open, ledger:open |
| [ptr-lineage](../../crates/ptr-lineage/README.md) | `prototype` | 8 | 3917 | 5 | 102 | 9 | 6 | R004:planned | adapter-serving:open |
| [ptr-memory](../../crates/ptr-memory/README.md) | `scaffold` | 6 | 1256 | 2 | 19 | 14 | 6 | Q002:planned, E002:planned, E004:planned | — |
| [ptr-model-api](../../crates/ptr-model-api/README.md) | `scaffold` | 4 | 126 | 2 | 3 | 9 | 5 | M007:planned, E001:planned, E003:planned | inference-serving:open |
| [ptr-net](../../crates/ptr-net/README.md) | `prototype` | 4 | 1333 | 5 | 34 | 21 | 7 | L002:planned, E001:planned | network:open |
| [ptr-observe](../../crates/ptr-observe/README.md) | `scaffold` | 7 | 351 | 5 | 12 | 9 | 5 | F001:planned, E003:planned | observability:open |
| [ptr-pg](../../crates/ptr-pg/README.md) | `prototype` | 17 | 6444 | 2 | 155 | 42 | 12 | L004:completed, Q003:planned | relational-substrate:open, materialized-state:open, lexical-search:open, local-vector-search:open, event-streaming:open |
| [ptr-pods](../../crates/ptr-pods/README.md) | `prototype` | 20 | 7210 | 12 | 84 | 33 | 4 | R002:planned, E001:planned | environment-runtime:open |
| [ptr-podwire](../../crates/ptr-podwire/README.md) | `prototype` | 4 | 3385 | 3 | 46 | 29 | 8 | — | — |
| [ptr-protocol](../../crates/ptr-protocol/README.md) | `scaffold` | 2 | 121 | 2 | 19 | 6 | 4 | R002:planned, E001:planned | network-codec:open, local-serialization:open |
| [ptr-router](../../crates/ptr-router/README.md) | `scaffold` | 1 | 120 | 1 | 2 | 4 | 5 | M004:planned, R002:planned, E003:planned | inference-serving:open, distributed-data-compute:open |
| [ptr-runtime](../../crates/ptr-runtime/README.md) | `prototype` | 21 | 14405 | 46 | 383 | 87 | 18 | E001:planned, E003:planned, E004:planned | — |
| [ptr-search](../../crates/ptr-search/README.md) | `prototype` | 2 | 652 | 3 | 24 | 8 | 5 | Q001:planned, Q002:planned, E002:planned, E003:planned | lexical-search:open, local-vector-search:open, gpu-vector-search:open, distributed-search:open, structural-code-search:open |
| [ptr-security](../../crates/ptr-security/README.md) | `prototype` | 1 | 126 | 2 | 8 | 6 | 4 | L001:running, E001:planned | security-context:open |
| [ptr-semdb](../../crates/ptr-semdb/README.md) | `prototype` | 2 | 864 | 4 | 37 | 20 | 5 | S001:planned, S002:planned, E004:planned | semantic-db:open |
| [ptr-server](../../crates/ptr-server/README.md) | `prototype` | 1 | 432 | 2 | 8 | 7 | 4 | E001:planned, E003:planned | — |
| [ptr-state](../../crates/ptr-state/README.md) | `prototype` | 1 | 377 | 3 | 10 | 13 | 1 | L001:running, L002:planned, E004:planned | materialized-state:open |
| [ptr-storage](../../crates/ptr-storage/README.md) | `prototype` | 4 | 1565 | 5 | 17 | 12 | 4 | E004:planned | object-storage:open |
| [ptr-types](../../crates/ptr-types/README.md) | `foundation` | 8 | 2932 | 6 | 75 | 32 | 10 | M005:planned, L001:running | — |
| [ptr-verifier](../../crates/ptr-verifier/README.md) | `scaffold` | 1 | 106 | 2 | 2 | 5 | 4 | F001:planned, Q002:planned, E001:planned | code-quality-verifier:open |

## Meaning of maturity labels

- `foundation` — small stable domain core already used broadly.
- `prototype` — executable behavior exists, but major target semantics are still missing.
- `scaffold` — contracts/data structures exist; core production implementation is not present.
- `research-scaffold` — architecture hypothesis is represented, but the trainable system is not implemented/proven.

The labels describe implementation maturity, not scientific novelty or production readiness.

## Freshness policy

Changes under `crates/<component>/src/` or that crate's `Cargo.toml` must update the matching `component.toml` in the same change. CI enforces this metadata coupling, then verifies that generated README/status output is synchronized.
