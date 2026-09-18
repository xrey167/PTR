# Deployment Topologies

## Developer / single-node

`ptrd`, local ledger/state, local model backend or remote vLLM/SGLang, local vector/lexical indexes.

## Multi-GPU workstation

Runtime remains one authority process while model/embedding/vector Pods use separate GPUs/processes. Tensor payloads prefer DLPack/IPC rather than serialization through JSON.

## Cluster

```mermaid
flowchart TB
  subgraph A["PTR Node A"]
    A1["ptrd"] --> A2["raft-rs"]
    A2 --> A3["raft-engine"]
    A3 --> A4["materialized state"]
  end
  subgraph B["PTR Node B"]
    B1["ptrd"] --> B2["raft-rs"]
    B2 --> B3["raft-engine"]
    B3 --> B4["materialized state"]
  end
  subgraph C["PTR Node C"]
    C1["ptrd"] --> C2["raft-rs"]
    C2 --> C3["raft-engine"]
    C3 --> C4["materialized state"]
  end
  A2 <-->|"ptr-raft/1 over Iroh QUIC"| B2
  B2 <-->|"ptr-raft/1"| C2
  C2 <-->|"ptr-raft/1"| A2
```

Consensus covers authoritative semantic state, not every cache/index operation. Retrieval remains local/replicated according to performance requirements.

## Enterprise/federated

Raw private memory remains local by default. Model/verifier adapter updates may use a federated-learning layer when explicitly configured.
