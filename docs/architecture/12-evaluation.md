# Component & Architecture Evaluation

Every replaceable technology has a candidate registry under `evaluations/components/`.

## Evaluation axes

- semantic correctness;
- failure semantics and recoverability;
- p50/p95/p99 latency;
- throughput;
- memory/VRAM;
- startup/recovery;
- portability;
- operational complexity;
- licensing;
- integration cost;
- observability;
- research flexibility.

## Architecture experiments

Technology benchmarks are separate from architectural hypothesis tests. Example: “Zvec vs Usearch” is a component evaluation; “typed semantic slots improve OOD Pod generalization” is a model experiment.

## Decision process

Candidate → reproducible evidence → ADR/default decision. Default remains replaceable behind PTR contracts.
