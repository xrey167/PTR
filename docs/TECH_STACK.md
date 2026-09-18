# PTR Technology Stack

The table distinguishes **PTR semantics** from **current candidate technologies**. A candidate is not permanently selected until evaluation closes.

| Concern | PTR-owned contract | Current candidate(s) | Alternative slot |
|---|---|---|---|
| Configuration | `ptr-config` | serde + TOML | config-rs/other layered sources |
| Runtime orchestration | `ptr-runtime` | PTR-owned composition | no external semantic owner |
| Domain types | `ptr-types` | Rust enums/newtypes | internal design |
| Incremental semantics | `ptr-semdb` | custom engine inspired by rust-analyzer | Salsa/other |
| Model framework | `ptr-core` | Burn/CubeCL; PyTorch reference | JAX/PyTorch/native |
| Inference | `ptr-model-api` | SGLang, vLLM, Burn native | TensorRT/custom |
| Execution | `ptr-exec` | Tokio substrate + PTR isolates | Tina/Compio experiments |
| Network | `ptr-net` | Iroh/QUIC | other QUIC/RPC |
| Network codec | `ptr-protocol` | Prost/Protobuf | Cap'n Proto/FlatBuffers/etc |
| Local archive | `ptr-protocol` | rkyv | postcard/bincode/etc |
| Consensus | `ptr-ledger` | tikv/raft-rs | OpenRaft/custom |
| Durable log | `ptr-ledger` | tikv/raft-engine | RocksDB/redb/custom WAL |
| Materialized state | `ptr-state` | Turso/libSQL | SQLite/Postgres/Redb |
| Lexical retrieval | `ptr-search` | Tantivy | alternative FTS |
| Local vector | `ptr-search` | Zvec | usearch/Lance/etc |
| GPU vector | `ptr-search` | cuVS | FAISS GPU/etc |
| Multimodal store | `ptr-search` | LanceDB | alternatives |
| Distributed search | `ptr-search` | Havenask | Quickwit/OpenSearch/etc |
| Code structure | `ptr-search` | GritQL | tree-sitter/custom |
| Object store | `ptr-storage` | OpenDAL | native backend implementations |
| Event streaming | `ptr-events` | Apache Iggy | Kafka/NATS/RocketMQ |
| Observability | `ptr-observe` | tracing + OTel + NeMo Relay | alternatives |
| Introspection | `ptr-inspect` | Valuable | custom reflection layer |
| Ingress classification | `ptr-ingress` | GLiClass-rs | small classifier/custom head |
| Stateful agent env | Pod/environment contract | ROCK | NeMo Gym/local |
| Code quality verifier | `ptr-verifier` | Sentrux + compiler/tests | custom sensors |
| Training orchestration | `training/` | NeMo RL, Unsloth, Data Designer | replaceable |
| Outer-loop optimization | research/training | GEPA/DSPy | custom search/RL |

## Selection rule

A component becomes a default only after evidence covers correctness, failure semantics, latency, throughput, memory, operational complexity, portability, licensing and research flexibility where relevant.
