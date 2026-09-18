# PTR Training Workspace

Training is separate from the production Rust runtime.

## Stages

1. semantic typing / raw↔typed verification;
2. PodWire + ActionIR;
3. epistemic calibration;
4. operator routing;
5. verifier + repair;
6. environment RL;
7. distillation / QAT / deployment optimization.

Python tooling is managed independently (uv is the preferred environment candidate). NeMo RL, Unsloth, Data Designer, DSPy and GEPA are candidates behind experiment configs rather than runtime dependencies.

## Reproducible run preparation

`ptr-train-run` resolves a versioned training config and dataset registry entry, then writes a run manifest containing the Git commit, lockfile hashes, platform, dataset metadata and training parameters.

```bash
python -m ptr_training.run --config training/configs/run-default.toml
```

The default backend is deliberately `dry-run`: it prepares reproducibility metadata but refuses to claim that training occurred. A concrete Burn/PyTorch/NeMo backend is enabled only after its component/training evaluation is selected.
