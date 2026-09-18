# Memory & Search Architecture

```mermaid
flowchart TB
  ST["Committed/materialized semantic state"] --> MEM["Typed Memory"]
  MEM --> SEM["Semantic"]
  MEM --> EPI["Episodic"]
  MEM --> PROC["Procedural"]
  MEM --> EPIST["Epistemic"]
  SEM --> PLAN["Retrieval planner"]
  PROC --> PLAN
  EPIST --> PLAN
  PLAN --> T["Tantivy"]
  PLAN --> Z["Zvec"]
  PLAN --> C["cuVS"]
  PLAN --> L["LanceDB"]
  PLAN --> H["Havenask"]
  PLAN --> G["GritQL"]
  T --> F["Fusion"]
  Z --> F
  C --> F
  L --> F
  H --> F
  G --> F
  F --> R["Exact source resolve + generation validation"]
  R --> V["Verifier"]
```

## Memory classes

- semantic: facts, entities, constraints, relations;
- episodic: what happened in an interaction/environment;
- procedural: verified reusable solution procedures;
- epistemic: known/unknown/hypothesis/distribution/conflict state.

## SemanticCapsule

A capsule is lifecycle managed and contains generation, provenance, validity and typed claims. It is not a vector chunk.

## Retrieval

Search engines are projections. A result is `PossibleEvidence` until the exact source/generation is resolved and verified.

## Roles under evaluation

Tantivy, Zvec, cuVS, LanceDB, Havenask and GritQL occupy different retrieval roles. The planner may compose them rather than choosing a single universal store.
