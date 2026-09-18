# End-to-End Request Lifecycle

```mermaid
sequenceDiagram
  participant C as Client
  participant I as Ingress
  participant S as SemDB
  participant M as PTR Core
  participant R as Router/Runtime
  participant P as Pod/Search
  participant V as Verifier
  participant L as Ledger/State
  C->>I: raw request
  I->>S: raw + typed proposal
  S-->>M: snapshot rN
  M-->>R: ModelEvent / operator request
  R->>P: typed operation
  P-->>V: observation / evidence
  V->>S: verified delta
  S-->>M: snapshot rN+1
  M-->>R: ActionIR / final answer
  R->>V: action verification
  V->>L: validated semantic/action event
  L-->>S: committed/materialized update
  R-->>C: streamed result / receipt
```

The key loop is **reason → need observation → execute → verify → commit/update SemDB → resume**. A request may therefore span several semantic revisions while each reasoning segment remains snapshot-isolated.

External effects occur only after ActionIR verification. A response can be streamed independently from authoritative semantic commits, but user-visible claims should retain their evidence/verification status.
