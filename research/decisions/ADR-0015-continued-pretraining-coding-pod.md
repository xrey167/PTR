# ADR-0015 — Continued Pretraining for a Coding Pod

## Status
Proposed.

## Decision
Add a full-parameter continued-pretraining track that domain-adapts an external open-source coding LLM checkpoint on PTR's own code corpus. The resulting checkpoint is served through `ptr-model-api`'s `InferenceBackend` trait (a self-hosted serving adapter, not the native GPU/DLPack Pod path) and exposed to the runtime as a specialist "coding" Pod manifest in `ptr-pods`. The training backend (Axolotl+DeepSpeed ZeRO-3, PyTorch FSDP, or a custom LOMO integration) is selected through a new `evaluations/components/training-backend/` evaluation, distinct from `inference-serving` (serving, not training) and `model-framework` (PTR's own tensor framework for `model/burn-a0`).

## Consequences
The checkpoint is not trusted as a Pod backend until experiment R003 (`experiments/runtime/R003-coding-domain-adaptation/`, which defines the held-out benchmark, metric and baseline in its `experiment.toml`) shows measurable improvement over the unmodified base model on held-out code from the user's own repos. Serving-backend selection for this Pod stays gated on the pre-existing `inference-serving` evaluation resolving. When training execution is implemented, training-backend selection must be gated on the new evaluation resolving separately. `training/` keeps `dry-run` as the default; `--execute` currently refuses every real backend because no training backend is implemented.
