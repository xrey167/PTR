# `trl-moe` training backend

Tiny-random or pretrained OLMoE (`OlmoeForCausalLM`) -> SFT -> RL with TRL's
`GRPOTrainer` configured for GSPO and SAPO -> routing metrics -> optional PTR
governance gates (`ptr_py`) -> a metrics JSON. Selected by `[training] backend = "trl-moe"`
in a stage config; `ptr_training.run --execute` dispatches here (it still refuses a dirty
tracked worktree).

```
tasks.py -> model.py -> sft.py -> rl.py -> routing_metrics.py / gates.py -> pipeline.py -> metrics.json
(seeded)   (OLMoE)     (AdamW)   (GRPO)    (pure Python)        (ptr_py?)    (orchestrator)
```

Pure-Python modules (no torch): `config`, `tasks`, `reward`, `routing_metrics`, `gates`.
Importing the package pulls in nothing heavy; `pip install -e training` (no extras) keeps
`dependencies = []`.

## Evidence status

`evidence_class: "pipeline-smoke"`, note "mechanism check only, not evidence". The
tiny-random run checks that the plumbing executes and produces finite numbers. It says
nothing about GSPO vs SAPO, routing stability, or real OLMoE behaviour.
`configs/stage-moe-olmoe.toml` (real model, 24 GB GPU, LoRA, bf16) has NOT been run.

## Install and run

```bash
cd training
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu   # CPU hosts
uv pip install --python .venv/bin/python -e '.[moe]'
.venv/bin/python -m unittest discover -s tests
# direct entry (no run manifest, works on a dirty tree):
.venv/bin/python -c "from ptr_training.moe.pipeline import run_config; \
  run_config('training/configs/stage-moe-smoke.toml', out='training/runs/smoke-dev')"
```

The `moe` and `cpt` extras conflict (axolotl pins older transformers/trl); `pyproject.toml`
declares this to uv so `uv.lock` resolves both.

## Verified versions (installed and read from source)

| package | version |
|---|---|
| torch | 2.14.1+cpu |
| transformers | 5.18.0 |
| trl | 1.14.1 |
| peft | 0.21.2 |
| datasets | 5.0.1 |
| accelerate | 1.15.0 |
| tokenizers | 0.23.2 |
| python | 3.11.15 |

The `moe` extra lower-bounds these. transformers >= 5 matters: its OLMoE stores experts as
fused 3-D parameters, which changes LoRA targeting (below).

## Verified API facts

TRL 1.14.1 `GRPOConfig` (trl/trainer/grpo_config.py, grpo_trainer.py):

- `importance_sampling_level` in {"token", "sequence"} (GSPO = "sequence").
- `loss_type` includes "grpo", "dapo", "sapo", "cispo", "luspo", "vespo", ...
  For GSPO the trainer's own warning names `loss_type="grpo"` as the paper reproduction, so
  GSPO = `importance_sampling_level="sequence"`, `loss_type="grpo"`, `epsilon=gspo_epsilon`/`epsilon_high=gspo_epsilon_high` (config keys, default 3e-4/4e-4; used only for gspo).
  The GSPO paper uses tiny clip ranges (3e-4 / 4e-4) because sequence ratios are near 1;
  the olmoe config uses those, the smoke config uses 0.2 / 0.28 to make clipping observable.
- SAPO = `loss_type="sapo"` with `sapo_temperature_pos` (default 1.0) and
  `sapo_temperature_neg` (default 1.05); its soft gate replaces clipping, so `epsilon` is unused.
- `beta` (KL coefficient, 0 = no reference model), `num_iterations`, `steps_per_generation`,
  `num_generations`, `max_completion_length`. `max_prompt_length` no longer exists.
- `mini_steps` in our config maps to `steps_per_generation` (optimizer steps per generation
  batch). With 1 the first update is exactly on-policy so clipping/the SAPO gate cannot act;
  the smoke config uses 2.
- Logged per step: `reward`, `reward_std`, `kl` (only when `beta != 0`), `entropy`,
  `aux_loss` (MoE load balancing, enabled because OlmoeConfig has `router_aux_loss_coef`),
  `clip_ratio/*` (logged for the clipped losses, not for SAPO), `completions/*`.
- Reward functions are called as `fn(prompts=, completions=, completion_ids=, **dataset_columns)`.

transformers 5.18.0 OLMoE:

- `OlmoeConfig` fields: `num_experts`, `num_experts_per_tok`, `router_aux_loss_coef`
  (0.01), `output_router_logits`, `norm_topk_prob`, `intermediate_size` (the expert width).
- `model(..., output_router_logits=True)` returns `router_logits`: a tuple with one
  `[batch*seq, num_experts]` tensor per layer, plus `aux_loss`.
- Router: `layers.N.mlp.gate.weight`; experts are fused parameters
  `layers.N.mlp.experts.gate_up_proj [E, 2I, H]` and `.down_proj [E, H, I]` (no per-expert
  `nn.Linear`). Attention: `self_attn.{q,k,v,o}_proj`.
- Real model: HF repo `allenai/OLMoE-1B-7B-0125` exists (not gated, Apache-2.0): 16 layers,
  hidden 2048, 64 experts, top-8, vocab 50304.

PEFT 0.21.2: expert projections are targeted with `target_parameters`
(`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`); attention with `target_modules`.
Verified on the tiny OLMoE: LoRA tensors attach to attention and to `mlp.experts`; the
router stays frozen.

## Reward (mirrors `ptr_verifier::VerificationReport`)

`reward.py` reports all four `VerificationStatus` values (crates/ptr-verifier/src/lib.rs):
`Pass`, `Fail`, `Disputed`, `Unknown`, with `level`, `score` in [0, 1] and `findings`
`{code, message, hard}`. The RL reward is `rl_reward(report)`:

| status | reward |
|---|---|
| Pass | `score`, in (0, 1] (exact-answer tasks: 1.0) |
| Fail | 0.0 |
| Disputed | 0.0 |
| Unknown | 0.0 |

A non-Pass report never receives a passing reward and its own `score` (diagnostic partial
credit such as "right shape, wrong value") is recorded in the report but never used as
reward. A hard finding forces `Fail` (the `VerificationReport` constructor demotes a Pass
carrying one, or a Pass with score 0). Produced here: `Unknown` for an empty completion,
`Disputed` for an arithmetic completion that states contradicting integers, `Pass` only for
the exact answer, `Fail` otherwise. Consequence: RL reward is sparse (exact answers only),
so the SFT stage must leave a non-zero pass rate for RL to see signal.

## Gates (`ptr_py`)

`gates.py` imports `ptr_py` guardedly. Absent: every gate reports
`{"status": "skipped", "reason": "binding not installed: gates skipped"}`. Present
(signatures from bindings/ptr-py/src/governance.rs): `aimer_score(list[float])` on each
expert's flattened gate_up+down weights, `rank_experts_for_pruning([(id, weights)])` per
layer, and `subspace_overlap(a, b)` / `chance_overlap(a, b)` on the `[hidden, intermediate]`
change of pairs of experts' `down_proj` (row-major `dim x k` matrices; tiny model only).
Any exception from the binding is recorded verbatim as `status: "error"` (and the report
status becomes "error") rather than hidden. Gate values are measurements, not decisions:
no threshold or pass/fail verdict is invented here.

## Metrics JSON

`metrics.json` in the output dir: versions, hardware, per-stage wall time, SFT losses,
per-arm RL logs (rewards, kl, clip ratio when logged, aux loss), greedy policy evaluation on
held-out verifiable tasks per stage, routing metrics per layer per stage (entropy, load CV,
dead experts, load), churn between stages (fraction of probe tokens whose top-k set changed),
gate results or the skip reason, `nan_check`, `evidence_class`.
Both RL arms start from the same restored SFT state with the same seed, so their first
generation batch is identical by construction.
