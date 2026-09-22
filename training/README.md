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

## Continued pretraining (external coding LLM)

`continued-pretraining/` is not part of the numbered stage curriculum above — that curriculum is PTR's own `ptr-a0`/`ptr-ar`/`ptr-diff` model family. Continued pretraining domain-adapts an *external*, already-pretrained open-source coding LLM on PTR's own code corpus, ahead of and separate from that curriculum, so its checkpoint can later back a specialist "coding" Pod (see `research/decisions/ADR-0015-continued-pretraining-coding-pod.md`, tracked in issue #25). Training-backend selection (Axolotl+DeepSpeed ZeRO-3, FSDP, or a custom LOMO integration) goes through `evaluations/components/training-backend/`; the stage stays behind the same `dry-run` gate as everything else until a candidate is selected. See `configs/coding-pod-continued-pretraining.toml` and `configs/run-coding-pod-cpt.toml`, and the linked experiment `experiments/runtime/R003-coding-domain-adaptation/`. (ADR-0015, the training-backend evaluation and R003 land via sibling PRs of issue #25 — plain paths here rather than links, since they don't all exist yet on this branch.)

## Reproducible run preparation

`ptr-train-run` resolves a versioned training config and dataset registry entry, verifies the declared dataset archive SHA-256/size, and writes an immutable run manifest containing Git/worktree state, config/model/hardware/dataset/card/lockfile hashes, platform, training parameters and a canonical input fingerprint.

```bash
python -m ptr_training.run --config training/configs/run-default.toml
```

The default backend is deliberately `dry-run`: it prepares reproducibility metadata but refuses to claim that training occurred. A concrete Burn/PyTorch/NeMo backend is enabled only after its component/training evaluation is selected.


The run manifest fails closed when a registered dataset artifact no longer matches its declared hash/size. Default run filenames include microseconds and are created exclusively rather than overwritten. `--execute` also refuses a dirty tracked worktree before any real backend can train weights.
