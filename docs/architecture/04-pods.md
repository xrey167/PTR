# Pod Architecture

Pods are semantic cognitive capabilities, not endpoint names.

```mermaid
flowchart LR
  R["Need<Capability<I,O>>"] --> REG["Pod registry"]
  REG --> P["Selected Pod"]
  P --> EX["Execute with lease + policy"]
  EX --> O["Observation<O>"]
  O --> V["Verifier"]
```

A manifest declares accepted/produced types, effects, required capabilities, version and execution properties.

## Modes

- native Rust;
- local isolated process;
- remote node over Iroh/QUIC;
- GPU tensor Pod;
- stateful environment/sandbox Pod.

## Generalization target

Training randomizes Pod names/interfaces while preserving semantic contracts so the model learns `Predict<Features,Distribution<Direction>>`, not one memorized function string.
