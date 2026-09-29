# PTR Technology Stack

The table distinguishes **PTR semantics** from **current candidate technologies**. A candidate is not permanently selected until evaluation closes. The broader set of intentionally open type/backend slots is tracked in the [open architecture catalog](../research/catalogs/README.md); this table is not a frozen bill of materials.

| Concern | PTR-owned contract | Current candidate(s) | Alternative slot |
|---|---|---|---|
| Configuration | `ptr-config` | serde + TOML | config-rs/other layered sources |
| Runtime orchestration | `ptr-runtime` | PTR-owned composition | no external semantic owner |
| Cognitive + semantic type kernel | `ptr-types` | orthogonal Rust enums/newtypes/generic semantic wrappers | internal design; no model-framework owner |
| Incremental semantics | `ptr-semdb` | custom engine inspired by rust-analyzer | Salsa/other |
| Model framework | `ptr-core` | Burn/CubeCL; PyTorch reference | JAX/PyTorch/native |
| Inference | `ptr-model-api` | SGLang, vLLM, Burn native | TensorRT/custom |
| Execution | `ptr-exec` | Tokio substrate + PTR isolates | Tina/Compio experiments |
| Network | `ptr-net` | Iroh/QUIC | other QUIC/RPC |
| Network codec | `ptr-protocol` | Prost/Protobuf | Cap'n Proto/FlatBuffers/etc |
| Local archive | `ptr-protocol` | rkyv | postcard/bincode/etc |
| Consensus | `ptr-ledger` | tikv/raft-rs | OpenRaft/custom |
| Durable log | `ptr-ledger` | tikv/raft-engine | RocksDB/redb/custom WAL |
| Materialized state | `ptr-state` | Turso/libSQL; PostgreSQL via `ptr-pg` | SQLite/Redb |
| Relational substrate (projection, derived caches, working state) | `ptr-pg` | PostgreSQL 18 + pgvector 0.8 over tokio-postgres | PostgreSQL 16/17; Turso projection-only |
| Lexical retrieval | `ptr-search` | Tantivy; PostgreSQL full text (baseline in `ptr-pg`) | pg_textsearch/ParadeDB BM25 |
| Local vector | `ptr-search` | Zvec; pgvector halfvec HNSW (`ptr-pg`) | usearch/Lance/VectorChord/etc |
| GPU vector | `ptr-search` | cuVS | FAISS GPU/etc |
| Multimodal store | `ptr-search` | LanceDB | alternatives |
| Distributed search | `ptr-search` | Havenask | Quickwit/OpenSearch/etc |
| Code structure | `ptr-search` | GritQL | tree-sitter/custom |
| Object store | `ptr-storage` | OpenDAL | native backend implementations |
| Event streaming | `ptr-events` | in-process reference bus; PostgreSQL projection event log; Apache Iggy | Kafka/NATS/RocketMQ |
| Agent branches and triage | `ptr-branch` | PTR-owned certification and calibrated arbiter | serial execution baseline |
| Working memory | `ptr-fastmem` | PTR-owned gated delta rule | recency buffer; retrieval only |
| Weak supervision | `ptr-labeling` | PTR-owned Dawid-Skene with verifier vetoes | Snorkel label model |
| Adapter lineage and serving | `ptr-lineage` | PTR-owned lineage; vLLM/SGLang multi-LoRA serving | LoRAX |
| Statistics and metrics | `ptr-analytics` | PTR-owned kernel; metric SQL in `ptr-pg` | pg_duckdb/Iceberg/DataFusion mirror |
| Observability | `ptr-observe` | tracing + OTel + NeMo Relay | alternatives |
| Introspection | `ptr-inspect` | Valuable | custom reflection layer |
| Ingress classification | `ptr-ingress` | GLiClass-rs | small classifier/custom head |
| Stateful agent env | Pod/environment contract | ROCK | NeMo Gym/local |
| Code quality verifier | `ptr-verifier` | Sentrux + compiler/tests | custom sensors |
| Training orchestration | `training/` | NeMo RL, Unsloth, Data Designer | replaceable |
| Outer-loop optimization | research/training | GEPA/DSPy | custom search/RL |

## Selection rule

A component becomes a default only after evidence covers correctness, failure semantics, latency, throughput, memory, operational complexity, portability, licensing and research flexibility where relevant.


## Developer tooling (non-runtime)

Developer/editor tools are not part of PTR's runtime bill of materials. They may wrap repository commands but may not redefine build, test, formatting or architecture semantics.

| Concern | PTR rule | Candidate(s) |
|---|---|---|
| Rust language intelligence | editor optional; CLI/CI remains authoritative | rust-analyzer |
| Emacs Rust editing | optional profile | rust-lang/rust-mode |
| Emacs integrated Rust environment | optional profile | emacs-rustic/rustic |
| Emacs Cargo UI | optional profile | ayrat555/cargo-mode |

See [development environment](DEVELOPMENT_ENVIRONMENT.md) and [rust-editor-emacs evaluation](../evaluations/components/rust-editor-emacs/README.md).
