# Network & Protocol Architecture

```mermaid
flowchart LR
  DOM["Rust Domain Types"] -->|public/debug| JSON["Serde JSON"]
  DOM -->|network control| PROTO["Prost / Protobuf"]
  DOM -->|trusted local hot path| RKYV["rkyv"]
  DOM -->|GPU tensors| DL["DLPack / CUDA"]
  PROTO --> IROH["Iroh / QUIC"]
  RKYV --> IPC["Local IPC"]
  DL --> GPU["GPU Pods / Model"]
```

## Separation

PodWire defines semantics. Prost/Protobuf is a network encoding candidate. Iroh/QUIC is a transport candidate. rkyv is a trusted local data-plane candidate. DLPack carries tensors.

## ALPN design

Planned independent protocols include:
- `ptr-raft/1`
- `ptr-podwire/1`
- `ptr-model/1`
- `ptr-blob/1`
- `ptr-events/1`

## Validation

Network decode is followed by semantic validation into PTR domain types. Network retries use idempotency keys for operations that could be duplicated.
