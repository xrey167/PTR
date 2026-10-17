# PTR Component Implementation Status

> Generated from every `crates/*/component.toml` plus code-derived metrics. Do not hand-edit.  
> Refresh with `python3 scripts/update_component_docs.py --write`.

**Component count:** 34  
**Maturity distribution:** `foundation`: 1, `prototype`: 33  
**Rust footprint:** 392 source files · 200367 nonblank source lines · 327 integration-test files · 4852 test markers (`#[test]`, `#[tokio::test]`)

| Component | Maturity | Rust files | LOC | Test files | Tests | Implemented items | Missing items | Experiments | Evaluations |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| [ptr-analytics](../../crates/ptr-analytics/README.md) | `prototype` | 8 | 1303 | 1 | 48 | 8 | 2 | F002:frozen, F003:frozen | analytics-mirror:open |
| [ptr-branch](../../crates/ptr-branch/README.md) | `prototype` | 7 | 3799 | 3 | 112 | 26 | 7 | S003:frozen, F003:frozen | — |
| [ptr-cluster](../../crates/ptr-cluster/README.md) | `prototype` | 3 | 749 | 2 | 18 | 16 | 6 | — | — |
| [ptr-config](../../crates/ptr-config/README.md) | `prototype` | 2 | 1167 | 6 | 30 | 13 | 4 | E001:planned | — |
| [ptr-core](../../crates/ptr-core/README.md) | `prototype` | 20 | 3959 | 4 | 85 | 20 | 6 | M001:planned, M002:planned, M003:planned, M004:planned, M005:planned, M006:planned, M007:planned, E001:planned, E003:planned, E004:planned | model-framework:open, inference-serving:open |
| [ptr-events](../../crates/ptr-events/README.md) | `prototype` | 5 | 2716 | 5 | 77 | 22 | 5 | E001:planned, F001:frozen | event-streaming:open |
| [ptr-exec](../../crates/ptr-exec/README.md) | `prototype` | 3 | 458 | 2 | 18 | 7 | 7 | R001:frozen, E001:planned, E003:planned | execution-runtime:open |
| [ptr-execwire](../../crates/ptr-execwire/README.md) | `prototype` | 5 | 1597 | 3 | 36 | 18 | 9 | — | — |
| [ptr-fabric](../../crates/ptr-fabric/README.md) | `prototype` | 25 | 9954 | 12 | 185 | 28 | 26 | — | — |
| [ptr-fastmem](../../crates/ptr-fastmem/README.md) | `prototype` | 9 | 3189 | 4 | 85 | 13 | 3 | M008:frozen, L003:completed | fast-weight-memory:open |
| [ptr-feedback](../../crates/ptr-feedback/README.md) | `prototype` | 12 | 4876 | 7 | 102 | 14 | 5 | F001:frozen, E001:planned, F008:planned | — |
| [ptr-ingress](../../crates/ptr-ingress/README.md) | `prototype` | 9 | 3722 | 6 | 97 | 12 | 8 | E001:planned, E004:planned | ingress-classifier:open |
| [ptr-inspect](../../crates/ptr-inspect/README.md) | `prototype` | 8 | 2388 | 4 | 58 | 13 | 5 | E001:planned | introspection:open |
| [ptr-labeling](../../crates/ptr-labeling/README.md) | `prototype` | 6 | 1888 | 1 | 36 | 12 | 5 | F002:frozen | label-model:open |
| [ptr-ledger](../../crates/ptr-ledger/README.md) | `prototype` | 9 | 9970 | 17 | 215 | 76 | 13 | L001:superseded, L001-v2:frozen, L002:frozen, E004:planned | consensus:open, ledger:open |
| [ptr-lineage](../../crates/ptr-lineage/README.md) | `prototype` | 10 | 5022 | 7 | 130 | 13 | 6 | R004:planned | adapter-serving:open |
| [ptr-memory](../../crates/ptr-memory/README.md) | `prototype` | 16 | 4819 | 8 | 108 | 30 | 6 | Q002:frozen, E002:planned, E004:planned | — |
| [ptr-model-api](../../crates/ptr-model-api/README.md) | `prototype` | 13 | 5238 | 11 | 165 | 21 | 11 | M007:planned, E001:planned, E003:planned | inference-serving:open |
| [ptr-net](../../crates/ptr-net/README.md) | `prototype` | 4 | 1399 | 5 | 49 | 22 | 6 | L002:frozen, E001:planned | network:open |
| [ptr-observe](../../crates/ptr-observe/README.md) | `prototype` | 14 | 2629 | 11 | 70 | 21 | 5 | F001:frozen, E003:planned | observability:open |
| [ptr-pg](../../crates/ptr-pg/README.md) | `prototype` | 17 | 6885 | 3 | 171 | 50 | 14 | L004:completed, Q003:superseded, Q003-v2:superseded, Q003-v3:frozen | relational-substrate:open, materialized-state:open, lexical-search:open, local-vector-search:open, event-streaming:open |
| [ptr-pods](../../crates/ptr-pods/README.md) | `prototype` | 40 | 19326 | 28 | 366 | 65 | 15 | R002:planned, E001:planned, E007:frozen | environment-runtime:open |
| [ptr-podwire](../../crates/ptr-podwire/README.md) | `prototype` | 5 | 5623 | 5 | 110 | 45 | 15 | E007:frozen | — |
| [ptr-protocol](../../crates/ptr-protocol/README.md) | `prototype` | 8 | 5682 | 4 | 124 | 18 | 6 | R002:planned, E001:planned, E007:frozen | network-codec:open, local-serialization:open |
| [ptr-router](../../crates/ptr-router/README.md) | `prototype` | 19 | 10843 | 16 | 256 | 40 | 10 | M004:planned, R002:planned, E003:planned, R015:planned | inference-serving:open, distributed-data-compute:open, pod-routing:open |
| [ptr-runtime](../../crates/ptr-runtime/README.md) | `prototype` | 67 | 58581 | 106 | 1581 | 186 | 60 | E001:planned, E003:planned, E004:planned, E007:frozen, L001-v2:frozen, R011:frozen, R012:frozen, R015:planned | — |
| [ptr-search](../../crates/ptr-search/README.md) | `prototype` | 5 | 2480 | 6 | 82 | 14 | 6 | Q001:planned, Q002:frozen, E002:planned, E003:planned | lexical-search:open, local-vector-search:open, gpu-vector-search:open, distributed-search:open, structural-code-search:open |
| [ptr-security](../../crates/ptr-security/README.md) | `prototype` | 1 | 126 | 2 | 8 | 6 | 4 | L001:superseded, E001:planned | security-context:open |
| [ptr-semdb](../../crates/ptr-semdb/README.md) | `prototype` | 2 | 864 | 4 | 37 | 20 | 5 | S001:frozen, S002:frozen, E004:planned | semantic-db:open |
| [ptr-server](../../crates/ptr-server/README.md) | `prototype` | 13 | 5118 | 9 | 114 | 27 | 8 | E001:planned, E003:planned | — |
| [ptr-state](../../crates/ptr-state/README.md) | `prototype` | 1 | 1254 | 4 | 23 | 24 | 4 | L001:superseded, L002:frozen, E004:planned | materialized-state:open |
| [ptr-storage](../../crates/ptr-storage/README.md) | `prototype` | 6 | 2150 | 7 | 41 | 14 | 7 | E004:planned | object-storage:open |
| [ptr-types](../../crates/ptr-types/README.md) | `foundation` | 9 | 3339 | 7 | 85 | 38 | 10 | M005:planned, L001:superseded | — |
| [ptr-verifier](../../crates/ptr-verifier/README.md) | `prototype` | 11 | 7254 | 7 | 130 | 14 | 7 | F001:frozen, Q002:frozen, E001:planned | code-quality-verifier:open |

## Meaning of maturity labels

- `foundation` — small stable domain core already used broadly.
- `prototype` — executable behavior exists, but major target semantics are still missing.
- `scaffold` — contracts/data structures exist; core production implementation is not present.
- `research-scaffold` — architecture hypothesis is represented, but the trainable system is not implemented/proven.

The labels describe implementation maturity, not scientific novelty or production readiness.

## Freshness policy

Changes under `crates/<component>/src/` or that crate's `Cargo.toml` must update the matching `component.toml` in the same change. CI enforces this metadata coupling, then verifies that generated README/status output is synchronized.
