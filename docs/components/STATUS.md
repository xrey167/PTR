# PTR Component Implementation Status

> Generated from every `crates/*/component.toml` plus code-derived metrics. Do not hand-edit.  
> Refresh with `python3 scripts/update_component_docs.py --write`.

**Component count:** 33  
**Maturity distribution:** `foundation`: 1, `prototype`: 32  
**Rust footprint:** 342 source files · 165415 nonblank source lines · 279 integration-test files · 3830 test markers (`#[test]`, `#[tokio::test]`)

| Component | Maturity | Rust files | LOC | Test files | Tests | Implemented items | Missing items | Experiments | Evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| [ptr-analytics](../../crates/ptr-analytics/README.md) | `prototype` | 8 | 1303 | 1 | 48 | 8 | 2 | F002:frozen, F003:frozen | analytics-mirror:open |
| [ptr-branch](../../crates/ptr-branch/README.md) | `prototype` | 7 | 3799 | 3 | 112 | 26 | 7 | S003:frozen, F003:frozen | — |
| [ptr-cluster](../../crates/ptr-cluster/README.md) | `prototype` | 3 | 749 | 2 | 17 | 15 | 6 | — | — |
| [ptr-config](../../crates/ptr-config/README.md) | `prototype` | 2 | 1167 | 6 | 30 | 13 | 4 | E001:planned | — |
| [ptr-core](../../crates/ptr-core/README.md) | `prototype` | 20 | 3799 | 4 | 78 | 17 | 6 | M001:planned, M002:planned, M003:planned, M004:planned, M005:planned, M006:planned, M007:planned, E001:planned, E003:planned, E004:planned | model-framework:open, inference-serving:open |
| [ptr-events](../../crates/ptr-events/README.md) | `prototype` | 5 | 2292 | 5 | 61 | 16 | 5 | E001:planned, F001:frozen | event-streaming:open |
| [ptr-exec](../../crates/ptr-exec/README.md) | `prototype` | 3 | 458 | 2 | 18 | 7 | 7 | R001:frozen, E001:planned, E003:planned | execution-runtime:open |
| [ptr-execwire](../../crates/ptr-execwire/README.md) | `prototype` | 5 | 1597 | 3 | 36 | 18 | 9 | — | — |
| [ptr-fastmem](../../crates/ptr-fastmem/README.md) | `prototype` | 9 | 3189 | 4 | 85 | 13 | 3 | M008:frozen, L003:completed | fast-weight-memory:open |
| [ptr-feedback](../../crates/ptr-feedback/README.md) | `prototype` | 11 | 4096 | 7 | 85 | 9 | 4 | F001:frozen, E001:planned | — |
| [ptr-ingress](../../crates/ptr-ingress/README.md) | `prototype` | 9 | 3584 | 5 | 90 | 11 | 8 | E001:planned, E004:planned | ingress-classifier:open |
| [ptr-inspect](../../crates/ptr-inspect/README.md) | `prototype` | 8 | 2218 | 2 | 47 | 11 | 5 | E001:planned | introspection:open |
| [ptr-labeling](../../crates/ptr-labeling/README.md) | `prototype` | 6 | 1888 | 1 | 36 | 12 | 5 | F002:frozen | label-model:open |
| [ptr-ledger](../../crates/ptr-ledger/README.md) | `prototype` | 9 | 8947 | 16 | 197 | 68 | 12 | L001:superseded, L001-v2:frozen, L002:frozen, E004:planned | consensus:open, ledger:open |
| [ptr-lineage](../../crates/ptr-lineage/README.md) | `prototype` | 9 | 4573 | 7 | 120 | 11 | 6 | R004:planned | adapter-serving:open |
| [ptr-memory](../../crates/ptr-memory/README.md) | `prototype` | 14 | 4326 | 8 | 89 | 23 | 6 | Q002:frozen, E002:planned, E004:planned | — |
| [ptr-model-api](../../crates/ptr-model-api/README.md) | `prototype` | 10 | 2396 | 8 | 75 | 18 | 7 | M007:planned, E001:planned, E003:planned | inference-serving:open |
| [ptr-net](../../crates/ptr-net/README.md) | `prototype` | 4 | 1399 | 5 | 49 | 22 | 6 | L002:frozen, E001:planned | network:open |
| [ptr-observe](../../crates/ptr-observe/README.md) | `prototype` | 14 | 2328 | 11 | 62 | 16 | 5 | F001:frozen, E003:planned | observability:open |
| [ptr-pg](../../crates/ptr-pg/README.md) | `prototype` | 17 | 6672 | 2 | 161 | 46 | 13 | L004:completed, Q003:superseded, Q003-v2:superseded, Q003-v3:frozen | relational-substrate:open, materialized-state:open, lexical-search:open, local-vector-search:open, event-streaming:open |
| [ptr-pods](../../crates/ptr-pods/README.md) | `prototype` | 39 | 18480 | 26 | 323 | 52 | 12 | R002:planned, E001:planned, E007:frozen | environment-runtime:open |
| [ptr-podwire](../../crates/ptr-podwire/README.md) | `prototype` | 5 | 5097 | 4 | 78 | 33 | 15 | E007:frozen | — |
| [ptr-protocol](../../crates/ptr-protocol/README.md) | `prototype` | 7 | 3693 | 3 | 72 | 12 | 5 | R002:planned, E001:planned, E007:frozen | network-codec:open, local-serialization:open |
| [ptr-router](../../crates/ptr-router/README.md) | `prototype` | 18 | 9184 | 14 | 207 | 30 | 9 | M004:planned, R002:planned, E003:planned | inference-serving:open, distributed-data-compute:open |
| [ptr-runtime](../../crates/ptr-runtime/README.md) | `prototype` | 57 | 47962 | 91 | 1211 | 144 | 48 | E001:planned, E003:planned, E004:planned, E007:frozen, L001-v2:frozen, R011:frozen, R012:frozen | — |
| [ptr-search](../../crates/ptr-search/README.md) | `prototype` | 3 | 1696 | 4 | 55 | 10 | 6 | Q001:planned, Q002:frozen, E002:planned, E003:planned | lexical-search:open, local-vector-search:open, gpu-vector-search:open, distributed-search:open, structural-code-search:open |
| [ptr-security](../../crates/ptr-security/README.md) | `prototype` | 1 | 126 | 2 | 8 | 6 | 4 | L001:superseded, E001:planned | security-context:open |
| [ptr-semdb](../../crates/ptr-semdb/README.md) | `prototype` | 2 | 864 | 4 | 37 | 20 | 5 | S001:frozen, S002:frozen, E004:planned | semantic-db:open |
| [ptr-server](../../crates/ptr-server/README.md) | `prototype` | 13 | 4776 | 9 | 106 | 24 | 8 | E001:planned, E003:planned | — |
| [ptr-state](../../crates/ptr-state/README.md) | `prototype` | 1 | 1150 | 3 | 17 | 18 | 4 | L001:superseded, L002:frozen, E004:planned | materialized-state:open |
| [ptr-storage](../../crates/ptr-storage/README.md) | `prototype` | 4 | 1565 | 5 | 19 | 12 | 4 | E004:planned | object-storage:open |
| [ptr-types](../../crates/ptr-types/README.md) | `foundation` | 8 | 3031 | 6 | 77 | 34 | 10 | M005:planned, L001:superseded | — |
| [ptr-verifier](../../crates/ptr-verifier/README.md) | `prototype` | 11 | 7011 | 6 | 124 | 14 | 7 | F001:frozen, Q002:frozen, E001:planned | code-quality-verifier:open |

## Meaning of maturity labels

- `foundation` — small stable domain core already used broadly.
- `prototype` — executable behavior exists, but major target semantics are still missing.
- `scaffold` — contracts/data structures exist; core production implementation is not present.
- `research-scaffold` — architecture hypothesis is represented, but the trainable system is not implemented/proven.

The labels describe implementation maturity, not scientific novelty or production readiness.

## Freshness policy

Changes under `crates/<component>/src/` or that crate's `Cargo.toml` must update the matching `component.toml` in the same change. CI enforces this metadata coupling, then verifies that generated README/status output is synchronized.
