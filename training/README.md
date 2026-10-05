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

`continued-pretraining/` is not part of the numbered stage curriculum above — that curriculum is PTR's own `ptr-a0`/`ptr-ar`/`ptr-diff` model family. Continued pretraining domain-adapts an *external*, already-pretrained open-source coding LLM on PTR's own code corpus, ahead of and separate from that curriculum, so its checkpoint can later back a specialist "coding" Pod (see `research/decisions/ADR-0015-continued-pretraining-coding-pod.md`, tracked in issue #25). Training-backend selection (Axolotl+DeepSpeed ZeRO-3, FSDP, or a custom LOMO integration) goes through `evaluations/components/training-backend/`; the stage stays behind the same `dry-run` gate as everything else until a candidate is selected. See `configs/coding-pod-continued-pretraining.toml` and `configs/run-coding-pod-cpt.toml`, and the linked experiment `experiments/runtime/R003-coding-domain-adaptation/`. (ADR-0015 is `research/decisions/ADR-0015-continued-pretraining-coding-pod.md`, the evaluation is `evaluations/components/training-backend/` and R003 is `experiments/runtime/R003-coding-domain-adaptation/`; all three exist.)

## Reproducible run preparation

`ptr-train-run` resolves a versioned training config and dataset registry entry, verifies the declared dataset archive SHA-256/size, and writes an immutable run manifest containing Git/worktree state, config/model/hardware/dataset/card/lockfile hashes, platform, training parameters and a canonical input fingerprint.

```bash
python -m ptr_training.run --config training/configs/run-default.toml
```

The default backend is deliberately `dry-run`: it prepares reproducibility metadata but refuses to claim that training occurred. A concrete Burn/PyTorch/NeMo backend is enabled only after its component/training evaluation is selected.


The run manifest fails closed when a registered dataset artifact no longer matches its declared hash/size. Default run filenames include microseconds and are created exclusively rather than overwritten. `--execute` also refuses a dirty tracked worktree before any real backend can train weights.

## First training (WP8): CPU smoke run

**Pipeline-Beleg, keine Evidenz.** This run shows that the `trl-moe` pipeline (SFT, then GSPO and SAPO arms, routing metrics, `ptr_py` gates) executes end to end on CPU. It is not evidence for any research claim: the model is a 0.24M-parameter random-init OLMoE, one seed, 32 eval tasks, no baseline, no ablation. The backend stays **unselected** (`backend_selection.status = "unselected"`, component `training-backend`, candidate `trl-peft` still `to-evaluate`; the run is labelled "unselected backend, pipeline evidence only").

```bash
PYTHONPATH=training/src python -m ptr_training.run --config training/configs/stage-moe-smoke.toml --execute
```

- Code: git sha `cc77e19ea8419986bec1349989c7f5d48e59851c` (`main` with PR #124 merged), clean tree (`git_dirty: false`), `input_fingerprint_sha256` `e2b83e1d7ddd411100b66d36c79bebe5a01460f8f94a987be850519829035641`, seed 17.
- Hardware: Intel Xeon @ 2.10GHz, 4 cores, 15.7 GiB RAM, no GPU, Linux 6.18.
- Wall time (in-process, `wall_time_s`): total 25.6 s (SFT 6.8 s, GSPO 3.1 s, SAPO 2.0 s, model build 2.1 s, gates 2.4 s each for SFT/GSPO/SAPO, snapshots 0.7 to 1.8 s each).
- Versions: python 3.11.15, torch 2.14.1+cpu, transformers 5.18.0, trl 1.14.1, peft 0.21.2, ptr_py 0.1.0 (rebuilt with `maturin develop` from the clean tree's `bindings/ptr-py`).
- Output: `training/runs/moe-smoke/<run>/metrics.json` (gitignored, not committed).

Eval is 32 held-out tasks (16 arith, 16 json), overlap with training prompts 0. Both RL arms start from the SFT checkpoint.

| Stage | Pass rate | arith / json pass | Routing entropy (L0 / L1) | Load CV (L0 / L1) | Dead experts |
| --- | --- | --- | --- | --- | --- |
| init | 0.000 | 0 / 0 | 2.065 / 2.069 | 0.509 / 0.485 | 0 / 0 |
| after SFT (200 steps, loss 4.15 to 0.39) | 0.469 | 1 / 14 | 1.988 / 1.748 | 0.540 / 0.528 | 0 / 0 |
| after GSPO | 0.313 | 0 / 10 | 2.002 / 1.766 | 0.483 / 0.563 | 0 / 0 |
| after SAPO | 0.563 | 2 / 16 | 2.002 / 1.762 | 0.541 / 0.540 | 0 / 0 |

| RL arm (16 steps, reward logged every second step) | Reward (min to max, mean of 8 logged steps) | KL per step (min to max) | Clip fraction (`clip_ratio/region_mean`, min to max, mean of 16 steps) |
| --- | --- | --- | --- |
| GSPO | 0.125 to 0.75, 0.375 | 0.0 to 0.698 | 0 to 0.75, 0.133 |
| SAPO | 0.3125 to 0.9375, 0.508 | 0.0 to 0.745 | null: trl 1.14.1 logs no clip_ratio for `loss_type=sapo` |

| Quantity | Value |
| --- | --- |
| Routing churn per layer (L0 / L1), init to SFT | 0.986 / 0.865 |
| Routing churn, SFT to GSPO | 0.511 / 0.582 |
| Routing churn, SFT to SAPO | 0.617 / 0.496 |
| Gate AIMER score range (all layers, experts, transitions) | 0.766 to 0.794 |
| Expert-delta subspace overlap (rank 4 of 64, 6 pairs per layer, 36 values) | 0.084 to 0.493, chance 0.0625 (all pairs `informative`) |
| Gate status (`init_to_sft`, `sft_to_rl_gspo`, `sft_to_rl_sapo`) | `ok`, `interference_informative: true` |
| Non-finite values / parameters | none / 0 |

Caveats: GSPO lowered the pass rate relative to SFT and SAPO raised it, but with 32 tasks and one seed the difference is within noise. Churn is measured on 64 tokens. All 36 overlap values (0.084 to 0.493) lie above the chance level of 0.0625 and the gate classifies every pair as informative, but at 0.24M parameters, random init and one seed this is not evidence of expert specialisation. The stored GSPO `clip_ratio_per_step` series matches the TRL log history (`clip_ratio/region_mean`, 0 to 0.75, peak at step 12). An earlier run of the PR #124 head showed all zeros because the series were built from a pre-filtered list of log entries instead of entries grouped by step; TRL logs `clip_ratio/*` on every optimizer step but `reward` only on generation steps, and `merge_log_history` now groups entries by `step` before building any series (commit af1cf71f, test `training/tests/test_moe_rl_series.py`). SAPO has no clip series because TRL 1.14.1 logs no `clip_ratio/*` keys for `loss_type=sapo`.

## GPU path (one 24 GB GPU, RTX 4090 desktop)

Not run yet: these steps were written and unit-tested on CPU only (device selection, gradient accumulation and checkpointing are covered by `training/tests/test_moe_device.py`). The `hardware/rtx4090.toml` profile they name is added separately. All three runs are pipeline smokes on the unselected `trl-moe` backend, not evidence.

`[model] device` is `"auto"` (default: CUDA when available, else CPU), `"cpu"` or `"cuda"`; `"cuda"` raises a `ConfigError` when CUDA is unavailable instead of silently running on CPU. SFT and TRL (`use_cpu`) use the model's device, and `metrics.json` `hardware` records `device`, `gpu`, `torch_cuda` (`torch.version.cuda`) and the peak `torch.cuda.max_memory_allocated`. `[moe.sft] gradient_checkpointing` and `gradient_accumulation_steps` reduce memory; see `src/ptr_training/moe/README.md`.

Run order, each step only after the previous one worked:

1. `make smoke-gpu` runs `configs/stage-moe-smoke-gpu.toml`: the tiny random OLMoE on CUDA (device plumbing check).
2. `make olmoe-sft-smoke` runs `configs/stage-moe-olmoe-sft-smoke.toml`: real `allenai/OLMoE-1B-7B-0125` (pinned revision), bf16, LoRA, SFT only, gradient checkpointing, batch 2 x 4 accumulation steps, max_length 256, 50 steps. Estimated (not measured) to fit in 24 GB: about 12.9 GB of bf16 weights plus LoRA/optimizer state and checkpointed activations.
3. `configs/stage-moe-olmoe.toml` (SFT + GSPO + SAPO, 3090 profile, not opted in) stays unrun until step 2 shows the measured peak memory.

Both Makefile targets are added by a separate change; until then run
`PYTHONPATH=training/src python -m ptr_training.run --config training/configs/<config>.toml --execute` from a clean tree.
