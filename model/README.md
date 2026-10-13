# PTR Model Research

This directory contains model-level configs, modification specifications, kernels, checkpoints/artifacts and reference implementations.

## Architecture families

- **PTR-A0:** smallest architecture probe.
- **PTR-AR:** autoregressive backbone plus typed workspace/reasoning.
- **PTR-Diff:** parallel/diffusion-style structured-state refinement research.

## Modification registry

`model/modifications/MOD-001 ... MOD-012` defines hypotheses, implementation points, signals, ablations and falsification criteria.

The production/runtime contract is `crates/ptr-model-api`; `crates/ptr-core` holds the model-side Rust types (a research scaffold), and the only executable neural implementation is `model/burn-a0`, which no runtime path reaches yet.

## Burn A0 executable prototype

[`model/burn-a0`](burn-a0/) is the first tensor/autodiff implementation of PTR's dual raw/typed path. It implements bidirectional raw↔semantic-slot cross-attention and operator-router logits on Burn 0.22.0-pre.3 (Rust 1.95, its own lockfile). It is intentionally a small architecture probe, not yet a pretrained language model.
