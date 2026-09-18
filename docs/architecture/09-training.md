# Training Architecture

```mermaid
flowchart TB
  REAL["Real tasks"] --> DATA["Dataset registry"]
  SYN["Synthetic / Data Designer"] --> DATA
  DSPY["DSPy teachers"] --> DATA
  DATA --> SFT["SFT / CPT"]
  SFT --> RL["RL / verifier learning"]
  RL --> DIST["Distillation"]
  DIST --> OPT["QAT / ModelOpt / compression"]
  OPT --> SERVE["PTR native / SGLang / vLLM"]
  TRACE["Runtime trajectories"] --> GEPA["GEPA outer loop"]
  GEPA --> DATA
  TRACE --> RL
```

## Stages

1. semantic typing and raw↔typed corruption detection;
2. PodWire / ActionIR native protocol;
3. epistemic calibration and Unknown behavior;
4. reasoning-operator routing;
5. verifier/repair training;
6. environment RL and long-horizon tasks;
7. distillation/quantization.

## Data

All datasets have provenance, schema, split and contamination metadata. OOD sets include unseen nominal types, unseen Pod names, long-horizon state, lifecycle mutation and subtle semantic corruption.

## Loops

- **inner loop:** weights/adapters/verifier models;
- **outer loop:** GEPA/DSPy teacher/system-policy search.

Hard invariants are excluded from learnable optimization.
