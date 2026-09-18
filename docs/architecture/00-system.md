# System Architecture

PTR consists of eight cooperating planes: ingress, incremental semantic state, PTR Core, execution, knowledge/search, causal authority, distributed transport, and learning/observability.

```mermaid
flowchart TB
  U[User / Files / Events] --> I[Typed Ingress]
  I --> S[Incremental Semantic DB]
  S --> C[PTR Core]
  C --> R[Rust Runtime]
  R --> P[Pods + Verifiers]
  P --> K[Knowledge / Search]
  R --> A[Hard Action Boundary]
  P --> S
  S --> Q[Consensus / Ledger]
  Q --> M[Materialized State]
  M --> K
  R --> O[Tracing / Feedback]
  O --> L[Training / GEPA / RL]
  L --> C
```

Authority flows downward from committed causal state into semantic state and derived indexes. Fast indexes never become authoritative.
