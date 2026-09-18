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

`ptr-train-run` resolves a versioned training config and dataset registry entry, verifies the declared dataset archive SHA-256/size, and writes an immutable run manifest containing Git/worktree state, config/model/hardware/dataset/card/lockfile hashes, platform, training parameters and a canonical input fingerprint.

```bash
python -m ptr_training.run --config training/configs/run-default.toml
```

The default backend is deliberately `dry-run`: it prepares reproducibility metadata but refuses to claim that training occurred. A concrete Burn/PyTorch/NeMo backend is enabled only after its component/training evaluation is selected.


The run manifest fails closed when a registered dataset artifact no longer matches its declared hash/size. Default run filenames include microseconds and are created exclusively rather than overwritten. `--execute` also refuses a dirty tracked worktree before any real backend can train weights.
