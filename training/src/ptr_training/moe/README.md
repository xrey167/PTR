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
`configs/stage-moe-olmoe.toml` (real model, 24 GB GPU, LoRA, bf16) has NOT been run, nor has
`configs/stage-moe-olmoe-sft-smoke.toml` or the GPU tiny smoke `configs/stage-moe-smoke-gpu.toml`
(see "GPU path").

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

## GPU path

Written and tested on CPU only; nothing here has run on a GPU.

- `[model] device = "auto" | "cpu" | "cuda"` (default `"auto"`). `config.parse` rejects any
  other value; `config.resolve_device` maps it to `"cpu"`/`"cuda"` at run time (`auto` = CUDA
  iff `torch.cuda.is_available()`) and raises `ConfigError` for `"cuda"` when CUDA is
  unavailable. Resolution is at run time, not parse time, so the GPU configs still parse on
  hosts without torch (CI). `model.build_tiny_model` and `build_pretrained` both place the model
  on that device; SFT trains on the model's device and `rl.build_grpo_config(..., device=)` sets
  TRL `use_cpu = (device == "cpu")`.
- `metrics.json`: `hardware.device`, `hardware.gpu`, `hardware.torch_cuda`,
  `hardware.peak_cuda_memory_allocated_bytes` / `_gib` (`null` on CPU; the peak counter is reset at
  the start of `execute`), plus `model.device_requested` and `model.device` (the device the
  parameters ended up on), and `provenance` (`provenance.py`: git commit and tracked-tree dirty
  flag, package versions, host, every visible GPU with name, capability and memory, the driver,
  the CUDA/cuDNN builds, `CUDA_VISIBLE_DEVICES` / `PTR_CUDA_DEVICE`, determinism flags).
- `[moe.sft] gradient_checkpointing` (bool, default false): at SFT start calls
  `gradient_checkpointing_enable()` on the base model (and `enable_input_require_grads()` when LoRA
  is on, so the adapters receive gradients) and switches it off again when SFT ends.
- `[moe.rl] gradient_checkpointing` (bool, default true): passed to `GRPOConfig` explicitly and
  recorded per arm (`rl.<algo>.gradient_checkpointing`). TRL 1.14.1 defaults it to true, so every
  RL arm before this key existed already ran checkpointed (earlier text here said RL does not
  use checkpointing; that was wrong).
- `[moe.sft] gradient_accumulation_steps` (int >= 1, default 1; ceiling 1024 unless
  `allow_large`): one optimizer step = that many micro-batches of `batch`; each loss is scaled by
  1/steps. `sft.steps` counts optimizer steps, the logged loss is one value per optimizer step
  (mean of its micro-batches) and `sft.effective_batch` records `batch * steps`. The SFT task
  sampler draws `batch * steps` tasks per optimizer step.
- Configs: `stage-moe-smoke-gpu.toml` (tiny, fp32, `device = "cuda"`, `hardware/rtx4090.toml`) and
  `stage-moe-olmoe-sft-smoke.toml` (OLMoE-1B-7B, bf16, LoRA, SFT only via `algorithms = []`,
  checkpointing, batch 2 x 4, max_length 256, 50 steps, `allow_unselected_backend = true`).
  Memory: see "Memory (estimate)" below; the real peak lands in `metrics.json`. Both are
  launched with `make smoke-gpu` and `make olmoe-sft-smoke`, which by default
  (`GPU_PROFILE=rtx3090`) run their `-rtx3090` variants naming `hardware/rtx3090.toml`.
- Tests: `training/tests/test_moe_device.py`.

### Memory (estimate, `memory.py`)

`python -m ptr_training.moe.memory --config <stage toml>` (or `ptr-train-run --config <toml>
--memory-report`) and `python -m ptr_training.moe.memory --m011-arm <arm>` print a per-phase
estimate as JSON. Parameter counts are exact (tested against the instantiated model and PEFT);
activation coefficients are fitted to CPU saved-tensor measurements (tests re-measure them, within
15%); the rest is arithmetic stated in the module docstring. Not measured on a GPU. Per 24 GiB card
(budget 95% = 22.8 GiB, CUDA context assumed 0.75 GiB):

| config / arm | trainable | peak phase | estimated peak (allocated) | fits 24 GiB |
|---|---|---|---|---|
| `stage-moe-olmoe-sft-smoke(-rtx3090).toml` | 121,634,816 | SFT (checkpointed, 2 x 256 tokens) | 16.6 GiB | yes, 5.4 GiB headroom |
| `stage-moe-olmoe.toml` as committed | 121,634,816 | SFT without checkpointing | 28.9 GiB | no |
| same with `[moe.sft] gradient_checkpointing = true` | 121,634,816 | RL update (checkpointed) | 17.0 GiB | yes |
| M011 `perft_r_*` (micro-batch 4 x 1024, eval batch 16) | 4,456,448 | eval generation | 17.7 GiB | yes, 4.4 GiB headroom |
| M011 `plain_lora` | 4,456,448 | eval generation | 17.3 GiB | yes, 4.8 GiB headroom |
| M011 `mixlora` | 4,423,680 | eval generation (adapter workspace on 16 x 1024 prefill tokens) | 20.8 GiB | yes, only 1.2 GiB headroom; `--eval-batch` 8 in the estimator gives more |
| tiny GPU smoke | 238,528 | SFT | < 0.01 GiB | yes |

Findings behind the numbers: the bf16 weights are 12.89 GiB (6,919,161,856 parameters). LoRA rank
16 on q/k/v/o plus the fused expert tensors trains 121.6M parameters (PEFT gives each 3-D expert
tensor `A [r*E, d0]`, `B [d1, r*E]`), about 1.8 GiB of fp32 weight, gradient and AdamW state, not
the "tens of MB" an earlier estimate said. PEFT applies expert LoRA by building the merged expert
tensor of a layer on every forward; without checkpointing autograd keeps it for backward, 768 MiB
per layer (measured on CPU), 12 GiB for 16 layers, which is why `stage-moe-olmoe.toml` as
committed does not fit. The M011 eval assumes prompts up to 1024 tokens (`--eval-prompt-tokens`).
The real peak lands in `metrics.json` (`hardware.peak_cuda_memory_allocated_bytes`); correct the
module from it. The pinned checkpoint is stored in float32 on the hub (`torch_dtype` in its
`config.json`): about 27.7 GB download, and the bf16 model is built in host RAM before `.to(cuda)`.

### Two GPUs (2x RTX 3090, PCIe, no NVLink)

Nothing in `training/` trains on two GPUs: no process group, no data sharding, no `device_map`
split, no FSDP/ZeRO. Under `torchrun` or `accelerate launch` with more than one process every
rank would run the same full job on `cuda:0`; `pipeline.execute` and `m011` therefore refuse
`WORLD_SIZE > 1` (`provenance.refuse_distributed_launch`). The second card runs a second,
independent job, pinned with `CUDA_VISIBLE_DEVICES` (inside the job the card is `cuda:0`;
`provenance.device.env` records the variable). `m011` also honours `PTR_CUDA_DEVICE` (OD-3): when
it is set and `CUDA_VISIBLE_DEVICES` is not, it sets `CUDA_VISIBLE_DEVICES` to it before torch is
imported and refuses a conflicting pair.

```bash
# exploratory / smoke runs, one per card, from the repo root (not frozen experiment runs)
mkdir -p training/runs/logs
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.moe.m011 \
  --arm mixlora --seed 1001 --out training/runs/m011-probe/mixlora --model olmoe --corpora hf \
  --max-steps 20 > training/runs/logs/m011-probe-mixlora.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.moe.m011 \
  --arm plain_lora --seed 1001 --out training/runs/m011-probe/plain_lora --model olmoe --corpora hf \
  --max-steps 20 > training/runs/logs/m011-probe-plain_lora.log 2>&1 &
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 30
# trl-moe stage configs: the same pattern
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.run \
  --config training/configs/stage-moe-olmoe-sft-smoke-rtx3090.toml --execute
```

Frozen confirmatory seeds go through `scripts/run_experiment.py`, which passes only declared
`PTR_*` variables: declare `PTR_CUDA_DEVICE` in the manifest's `command_environment` before the
freeze (OD-3), else the run lands on card 0; run them one at a time per card. Not done and not
claimed: data-parallel training over both cards (it would need a process group, per-rank domain
sampling and an adapter-gradient all-reduce, and it changes the pinned batch semantics), and
splitting one model over both cards with `device_map` (sequential over PCIe; only useful if a
config does not fit one card, which none of the estimates above requires). No speedup is claimed
for either.

### Reproducibility

CPU runs with the same seed reproduce bit for bit (`test_m011_run.test_same_seed_is_reproducible`
compares every metric except wall-clock timings for two arms; the pipeline tests compare model
inits). CUDA runs are not expected to: OLMoE's expert loop accumulates with `index_add_` and the
backward of token gathers, which use CUDA atomics in non-deterministic order with torch's default
settings; `provenance.device.bit_reproducible` is false on CUDA. Not measured here (no GPU).

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
  both configs pass those values (`gspo_epsilon = 0.0003`, `gspo_epsilon_high = 0.0004`) to TRL as
  `epsilon`/`epsilon_high` for GSPO. The smoke's `epsilon = 0.2` / `epsilon_high = 0.28` are
  grpo-style token-level values that neither algorithm uses here.
- SAPO = `loss_type="sapo"` with `sapo_temperature_pos` (default 1.0) and
  `sapo_temperature_neg` (default 1.05); its soft gate replaces clipping, so `epsilon` is unused.
- `beta` (KL coefficient, 0 = no reference model), `num_iterations`, `steps_per_generation`,
  `num_generations`, `max_completion_length`. `max_prompt_length` no longer exists in `GRPOConfig`, so `moe.rl.max_prompt_length` is enforced by
  `rl.check_prompt_lengths`: a prompt longer than it (in tokens) is rejected with a `ValueError`,
  never truncated.
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

## Config rules and run hygiene

- Unknown keys are rejected (`ConfigError` naming the table) in `[model]` (manifest keys
  `family`/`config` allowed), `[moe]`, `[moe.sft]`, `[moe.rl]`, `[moe.lora]`, `[moe.output]`.
  Shared manifest tables (`[run]`, `[data]`, `[training]`, `[evaluation]`, `[codebook]`) are not
  validated here.
- `moe.rl.steps` must be >= 1 (transformers ignores `max_steps = 0` and would train the default
  3 epochs). To skip RL set `moe.rl.algorithms = []` (SFT-only run).
- `model.revision` (commit sha or tag; `main`/`master`/`HEAD` rejected) is required for
  `kind = "pretrained"` and is passed to both `from_pretrained` calls. The metrics record
  `model.source` = `{hf_id, revision, resolved_sha, reason}`; `resolved_sha` comes from the Hub
  and is `null` (with `reason`) when offline. `stage-moe-olmoe.toml` pins
  `allenai/OLMoE-1B-7B-0125` at `9b0c1aa8...`.
- Reference policy with LoRA and `beta > 0`: TRL 1.14.1 (`grpo_trainer.py`) copies the live
  `default` adapter to a frozen `ref` adapter when the trainer is built (peft >= 0.20) and scores
  the reference under it, so with the SFT adapter loaded the reference is the SFT state, not the
  pretrained base (adapter disabled is only the fallback for peft < 0.20 with `target_parameters`).
  `rl._assert_peft_reference` checks `ref` exists and equals the SFT adapter before training and
  raises otherwise; the `ref` adapter is deleted after each arm. Cost: a second copy of the LoRA
  weights only.
- Float config values must be finite (nan/inf rejected); `moe.lora.target_modules` /
  `target_parameters` must be lists of non-empty strings.
- `moe.sft.max_length` must exceed the longest task prompt + 1 and `moe.rl.max_completion_length`
  must be >= the longest canonical answer + EOS over the whole task space (14 for the
  built-in tasks; measured in characters at config time, which equals tokens for the tiny
  tokenizer and is conservative for BPE). `run_rl` re-checks with the real tokenizer and
  `sft.collate` refuses an example whose labels were all truncated away.
- `kind = "pretrained"` rejects the tiny-only keys (`layers`, `hidden`, `intermediate`, `heads`,
  `experts`, `top_k`, `max_positions`, `router_aux_loss_coef`); `model.architecture` in the
  metrics records the loaded model's actual values, and `model.class` is the base architecture
  even under LoRA.
- `require_extra` also enforces the `moe` extra minimums (`pipeline.MOE_MINIMUMS`, checked against
  `pyproject.toml` by a test; `packaging` optional). Gate inputs are flattened only when the
  binding is usable and each expert fits `MAX_FLAT_ELEMENTS` (checked on `numel()` first).
  Task splits, prompt/completion/SFT-label length checks run before any hub call or SFT step.
  `[training].precision`, if present, must equal `model.precision`.
- `moe.lora.dropout` must be in [0, 1). `require_extra` checks installed distributions via
  `importlib.metadata` (the repo's top-level `datasets/` directory would satisfy a module-name
  check). A JSON completion with duplicate keys is a `Fail` with hard finding `duplicate-keys`.
- Gate aggregate status is `ok` only if every layer ran, `partial` (with `layers_run` /
  `layers_skipped`) if some were skipped, `skipped` (with reason) if all were; any binding error
  still makes it `error`. Non-finite router logits raise `NonFiniteRouterLogits`; the pipeline
  records the stage under `router_errors` / `nan_check.router_errors`, sets
  `nan_check.nonfinite_values_found = true` and leaves that stage's metrics and churn `null`.
- `model.router_aux_loss_coef` must be finite and >= 0. Metrics record `run.seed` and `model.seed`
  (the init seed actually used; it defaults to the run seed) separately.
- `run.seed` / `model.seed` must be in [0, 2**32). For tiny-random, `model.max_positions` must
  cover max(`moe.sft.max_length`, `max_prompt_length + max_completion_length`) and the head
  dim must be even. Verifiers accept one answer only (see `reward.py`; arithmetic integers
  must be canonical decimal). `nan_check.nonfinite_parameters` records, per stage (after SFT,
  after each RL arm, checked before the SFT state is restored), the count and first names of
  parameters containing NaN/inf, and sets `nonfinite_values_found`.
- Interference gate: expert deltas are reduced to their top-`moe.gates.subspace_rank` left
  singular subspaces (pure-Python orthogonal iteration) before `subspace_overlap` /
  `chance_overlap`, because full-rank subspaces of the ambient space overlap ~1.0 with chance
  1.0. Each pair records `informative` (false if r >= ambient dim, chance_overlap >= 0.99, or the
  binding call failed); layers and the report carry `interference_informative`. The binding's
  `measure_interference` takes LoRA `(layer, B, A)` factors per adapter, not dense expert
  deltas, so the two-argument overlap functions are the right entry point here.
  `binding_version` comes from distribution metadata (`ptr_py` / `ptr-py`).
- Per-step series (`reward`, `kl`, `clip_ratio`, `entropy`, `grad_norm`, `loss`; `*_per_step`) are
  built from TRL's log history grouped by `step` (`rl.merge_log_history`): with
  `steps_per_generation` > 1 the reward is logged only on generation steps while kl/clip_ratio
  are logged every step, so reading series from reward-bearing entries dropped data. The step
  indices of each series are in `series_steps[name]`.
- Metrics TRL does not log are `null` with a reason, not `[]`: `kl_per_step` (beta = 0) and
  `clip_ratio_per_step` (TRL 1.14.1 logs no `clip_ratio/*` and no other gate metric for
  `loss_type="sapo"`; checked in `grpo_trainer.py`), with `kl_unavailable_reason` /
  `clip_ratio_unavailable_reason`.
- `--execute` refuses `trl-moe` unless `[training] allow_unselected_backend = true` (the three smoke configs: CPU, GPU
  and OLMoE SFT; `stage-moe-olmoe.toml` does not): the training-backend component decision is `unlocked` and `trl-peft`
  is `to-evaluate`, and `evaluations/DECISION_PROCESS.md` requires gate tests plus evidence
  comparing at least two candidates before selection. Manifests and metrics carry
  `backend_selection: "unselected backend, pipeline evidence only"`.
- Gates are reporting-only: no value is compared with a threshold, there is no
  `forgetting_gate_evaluate` call, and nothing reacts to a gate. `metrics.json` carries
  `gates_evaluated` and `gates_not_evaluated_reason` at the top level; the OLMoE config
  (pretrained + LoRA) exercises no gate at all (`gates_evaluated = false`). Pair selection is all
  pairs up to the cap, else a deterministic stride over all of them (`pairs_evaluated` /
  `pairs_total` per layer); `run_gates` itself upgrades the status to `error` when any embedded
  call failed (precedence error > skipped > partial > ok).
- GSPO vs SAPO at `mini_steps = 1` (ratio == 1): by TRL 1.14.1's loss (`_compute_loss`), the
  first-step gradients coincide (SAPO's gate has slope 1 at ratio 1; same per-sequence token-mean
  aggregation), a test checks `grad_norm` agrees at step 1, but the logged loss differs by
  construction (gate value 2/tau, tau differs for A > 0 and A < 0) and later steps are not
  bit-identical (Adam amplifies fp noise). It is an approximate control, not a "GSPO == SAPO"
  identity.
- Routing metrics are computed per probe text (`routing.<stage>.per_text`, plus the pooled
  `per_layer`); `moe.output.probe_tokens` caps tokens *per text*, so a long text cannot crowd
  out later ones. Router logits are upcast to fp32 (OLMoE takes its top-k over an fp32 softmax of
  them); our top-k breaks ties toward the lowest expert index whereas `torch.topk` tie order is
  unspecified (`topk_tie_break` in the metrics).
- Not produced by this backend yet (listed in `metrics.json` `not_produced`):
  `policy_ratio_variance_per_token`, `crash_step`, `train_inference_kl`, and rollout-vs-training
  expert churn (M010/M012 quantities).
- `[training]` may only contain `backend`, `precision` (must equal `model.precision`) and
  `allow_unselected_backend`; `learning_rate`, `max_steps`, `batch_size`,
  `gradient_accumulation`, `checkpoint_every` are ignored by this backend (hyperparameters live
  in `[moe.*]`) and rejected. `[evaluation]` must be inert (`every = 0`, `suites = []`). The
  manifest's `[data]` dataset is not used: `metrics.json` records
  `data_source = "synthetic-tasks"`.
- `model.revision` must be a full 40-hex commit sha. Online, `resolve_revision` raises if the hub
  resolves it to a different sha and records `revision_verified = true` otherwise; offline it
  records `revision_verified = false` with the reason. A pretrained model with LoRA disabled
  is a full fine-tune and needs `[moe] allow_full_finetune = true`. Resource keys have sanity
  ceilings (sft.steps/examples 100000, sft.batch 1024, rl.steps 10000, num_generations and
  prompts_per_step 64, max_completion_length 512, probe_tokens 4096) lifted by
  `[moe] allow_large = true`. `sapo_temperature_neg` must exceed `sapo_temperature_pos`.
- The `moe` extra is bounded above (`trl<1.15`, `peft<0.22`, `torch<3`, `transformers<6`,
  `datasets<6`, `accelerate<2`; `pipeline.MOE_UPPER_BOUNDS`, tested against `pyproject.toml`);
  `require_extra` reports too-new packages. A test asserts the `GRPOConfig` fields we pass exist.
- Verifiers: integers are ASCII `-?[0-9]+` (<= 64 chars, else `too-long`); pathological JSON
  nesting is `invalid-json`; a completion that fills `max_completion_length` without EOS is a
  hard `truncated` Fail (via TRL's `completion_ids`; also in the policy evaluation). SFT requires
  `max_length >=` longest prompt + longest answer + EOS (26), so the EOS label always survives.
- `run_moe_tests.py --require-backend` exit codes: 0 ok, 1 test failure, 2 environment missing
  (moe extra absent and no `training/.venv` to re-exec into; not a code defect), 3 tests skipped.
  The RL prompt pool is `ceil(steps / mini_steps) * mini_steps * prompts_per_step` (floor 64):
  TRL draws `mini_steps * prompts_per_step` new prompts per generation batch and slices it over
  `mini_steps` optimizer steps (each step consumes `prompts_per_step` fresh prompts), including
  the final partial batch; the full-fine-tune reference model is reloaded in the policy
  dtype (`model_init_kwargs`), and `config._name_or_path` is restored after each RL arm.
- `moe.rl.mini_steps` must not exceed `moe.rl.steps`, is capped at 64, and the generation batch
  `mini_steps * prompts_per_step * num_generations` at 4096 rollouts (both lifted by
  `[moe] allow_large = true`). With `algorithms = []` no RL prompt pool is built
  (`tasks.rl_pool_size = 0`). After loading, `context_check` compares the tokenizer-measured
  longest SFT example and longest prompt + `max_completion_length` with the model's
  `max_position_embeddings` and raises `ConfigError` if it does not fit.
- Gate pairs record `requested_rank`, `effective_rank_a/b` (basis columns actually passed; a
  low-rank delta yields fewer than requested) and `rank_truncated` (always false: the binding's
  `subspace_overlap` / `chance_overlap` accept unequal ranks).
- Gate snapshots are capped in aggregate (`gates.MAX_TOTAL_GATE_ELEMENTS` = 16,777,216
  Python-list elements over all layers and experts, computed from tensor shapes before any
  conversion; `[moe] allow_large = true` lifts it): over the cap the gate report is `skipped` with
  "snapshot too large (N elements > cap)". With `algorithms = []` the RL-only validations (steps,
  `mini_steps`, generation batch) are skipped. Every RL arm sets `GRPOConfig.disable_dropout`
  (trl 1.14.1 `grpo_config.py` ~452, applied at `grpo_trainer.py` ~988). The SFT state is
  snapshotted only for 2+ arms and restored only before arms 2..n.
- Config sweep (every field is consumed or rejected; every size has a ceiling): `model.hf_id` is
  rejected for tiny-random (like `revision`); `[moe.lora]` keys other than `enabled` are rejected
  while LoRA is off and `[moe.gates]` is rejected for pretrained/LoRA runs (gates never run
  there). RL-only keys stay accepted (and unused) with `algorithms = []`, and are exempt from
  the RL ceilings, but `moe.rl.max_completion_length` (used by both policy evaluations) is
  capped regardless. Ceilings lifted by `[moe] allow_large = true`: the earlier resource keys
  plus `moe.sft.max_length` 4096, `moe.rl.max_prompt_length` 4096 (RL only), `moe.lora.rank` 1024
  (LoRA only), and for tiny-random layers 24, hidden 1024, intermediate 4096, heads 32, experts
  64, top_k 16, max_positions 2048, plus a derived parameter cap of 50M
  (`config.tiny_parameter_count`, tested equal to the real model's count; vocab is fixed at 59,
  kv heads == heads).
- Entry points: the CLI (`ptr_training.run --execute`), `pipeline.run_config` and
  `pipeline.execute` all call `config.require_backend_opt_in` (one message,
  `BackendNotSelected`); `run_config` does so before creating any directory, `execute` before
  `require_extra`, and `execute` also performs the tokenizer-measured context check.
  `run_sft` / `run_rl` are internal building blocks (they re-check prompt/completion lengths).
- Seeding: python/numpy/torch are seeded before model construction and again before LoRA init.
- `precision = "bf16"` is applied to the tiny model too; `model.param_dtypes` records the dtypes
  actually present (PEFT may keep LoRA weights in fp32).
- Each run writes to a fresh `<output.dir>/<run name>-<UTC timestamp (microseconds)>-<config sha8>-<6 hex random>/`;
  `metrics.json` is created exclusively and never overwritten.
- Evaluation tasks are held out first (32 distinct prompts, `tasks.generate_heldout`); SFT and RL
  tasks are sampled with replacement from the remaining prompts, so eval is disjoint by
  construction. `metrics.json` `tasks` records counts and `eval_overlap` (0; a nonzero value
  aborts the run). The prompt space is 310 distinct prompts (200 arith, 110 json); asking for
  more distinct held-out tasks than a kind has raises `ValueError`.

## Reward (mirrors `ptr_verifier::VerificationReport`)

`reward.py` reports all four `VerificationStatus` values (crates/ptr-verifier/src/lib.rs):
`Pass`, `Fail`, `Disputed`, `Unknown`, with `level`, `score` in [0, 1] and `findings`
`{code, message, hard}`. The RL reward is `rl_reward(report)`:

| status | reward |
|---|---|
| Pass, `level` >= `moe.rl.min_verification_level` | `score`, in (0, 1] (exact-answer tasks: 1.0) |
| Pass, `level` below the minimum | 0.0 |
| Fail | 0.0 |
| Disputed | 0.0 |
| Unknown | 0.0 |

`level` is one of `LEVELS` = Unverified < LatentAgreement < SampleVerified < FullSemantic <
Deterministic (mirrors `ptr_types::VerificationLevel`; an unknown level is rejected). The
minimum defaults to `Deterministic`, which is what this module's verifiers produce.

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
gate results or the skip reason, `nan_check`, `evidence_class`, `integrity`, `adapter_sha256`.
Both RL arms start from the same restored SFT state with the same seed, so their first
generation batch is identical by construction.

### Integrity, adapter and preflight

`integrity` (`ptr_training.integrity`) records four properties, all fail-closed: `frozen_unchanged`
(every `requires_grad=False` tensor bit-identical to the start of the run), `adapter_changed`
(training moved the trainable tensors), `steps_exact` (`run_sft` and the GRPO trainer took exactly
the configured number of optimizer steps) and `eval_identity_unchanged` (each policy evaluation
leaves all weights bit-identical). A violation raises `IntegrityError`; `run_config` then writes
`integrity-violation.json` instead of `metrics.json`. A full fine-tune has no frozen tensors, and
`integrity.frozen_tensors` says so (0) instead of implying a check happened.

LoRA runs save the adapter under `adapter/<stage>/` (`after_sft`, `after_rl_<algo>`);
`adapters` lists each path with its SHA-256 and `adapter_sha256` is the final stage's digest.
A full fine-tune records `adapter_sha256: null`.

`python -m ptr_training.run --execute` needs a preflight record from
`python -m ptr_training.preflight --config <stage.toml>` (written to `training/runs/preflight/`)
whose `protocol_sha256` (config file), `runner_sha256` (all `ptr_training` sources) and `device`
(`cpu` or `cuda:<gpu name>`) equal the run's. `--allow-missing-preflight` waives it for
`pipeline-smoke` runs only. The record file stays local (`training/runs/` is ignored); the run
manifest and `metrics.json` carry `preflight = {status, record_sha256, device}` (or `waived`), and a
refused run writes no manifest.

## M011 routed PEFT

CPU-testable implementation of the three adapter topologies, domain-masked routing, corpus
loaders and the entrypoint for experiment M011 (routed PEFT on OLMoE). The experiment stays
`planned` (registry); its manifest is `experiments/model/M011-routed-peft`, its `entrypoint` is
still empty, and the pins below mirror it (`tests/test_m011_*` cross-check them when `PTR_M011_CONFIG` points at
that `config.toml`).

| module | role |
|---|---|
| `routed_peft.py` | arm topologies (torch only): `PerftRBlock`, `MixLoraBlock`, PEFT plain LoRA, `RoutingContext` (domain mask), `route_topk`, `install_arm`, `count_trainable_parameters` |
| `corpora.py` | GSM8K / HumanEval / BFCL-simple at the pinned HF revisions, partition rules, domain labels, selection digests, offline fixtures |
| `scorers.py` | GSM8K exact numeric match, HumanEval pass@1 (subprocess sandbox), BFCL single-call AST check |
| `m011.py` | entrypoint: build model, install arm, SFT, evaluate, `--tune`, writes `run.json` / `metrics.json` / `tuning.json` |

Arms (`--arm`), parameter counts for real OLMoE shapes (H 2048, I 1024, 16 layers):

| arm | topology | trainable |
|---|---|---|
| `perft_r_emergent` | 8 parallel bottleneck experts (r 8, alpha 16) beside each MoE block, top-2 router [H, 8], unmasked | 4,456,448 |
| `perft_r_prescribed` | same, router masked to the domain's experts (code 0-2, math 3-5, tool 6-7) before top-2 | 4,456,448 |
| `mixlora` | 9 LoRA experts (r 4, alpha 8) as deltas on fused `gate_up_proj` / `down_proj`, one delta set shared by all 64 base experts, top-2 router [H, 9] | 4,423,680 |
| `plain_lora` | PEFT `LoraConfig` r 17, alpha 34 on q/k/v/o | 4,456,448 |

Everything else is frozen (checked at install: no non-adapter parameter may require grad). The
count test builds a meta-device model with the real shapes, so no weights are downloaded.
Installation replaces each `OlmoeSparseMoeBlock` (transformers 5.18: `gate` =
`OlmoeTopKRouter`, `experts` = `OlmoeExperts`) by a wrapper holding it as `.base`. Adapter
parameters are fp32; the base may be bf16. The domain label is set per forward pass through the
shared `RoutingContext`; a masked arm without labels raises instead of routing unmasked.

Commands (from the repo root, with the training venv):

```bash
# tiny smoke: random 2-layer OLMoE, in-memory fixture corpora, CPU, nothing downloaded
PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.moe.m011 \
  --arm perft_r_prescribed --seed 1 --out /tmp/m011-smoke --model tiny --max-steps 2 --eval-limit 3
# tests (about 35 s on CPU)
PYTHONPATH=training/src training/.venv/bin/python -m unittest discover -s training/tests -p 'test_m011*'

# real run on one 24 GiB card (an RTX 3090 on the owner's desktop; downloads OLMoE and the three
# corpora at the pinned revisions),
# two commands per arm (TB1 protocol):
# 1. tune ONCE on the pilot seed 1001 (never a confirmatory seed): 8 trials of floor(0.25 * 2000) = 500 steps,
#    seeded random search, selection on validation only -> frozen tuning.json
python -m ptr_training.moe.m011 --arm mixlora --out runs/m011/mixlora/tune --model olmoe --tune
# 2. every pinned seed (17 29 43 71 101) at the full 2000 steps with the frozen hyperparameters
python -m ptr_training.moe.m011 --arm mixlora --out runs/m011/mixlora --model olmoe --all-seeds \
  --hparams runs/m011/mixlora/tune/tuning.json     # writes runs/m011/mixlora/seed<N>/
# or both in one command (tunes into <out>/tune, then runs the seeds):
python -m ptr_training.moe.m011 --arm mixlora --out runs/m011/mixlora --model olmoe --tune --all-seeds
# a single seed with frozen hyperparameters:
python -m ptr_training.moe.m011 --arm mixlora --seed 29 --out runs/m011/mixlora/seed29 \
  --model olmoe --hparams runs/m011/mixlora/tune/tuning.json
```

Tuning protocol (TB1): `--tune` runs on the tuning seed only (default: the pilot seed 1001,
which is not one of the confirmatory seeds 17 29 43 71 101) with `tuning_steps = max(1, floor(0.25 * steps))` per trial (500 for 2000). `--tune-steps`,
`--tune-budget` and another pilot `--seed` (>= 1001) stay available as overrides; a confirmatory
seed or a seed below 1001 is refused for `--tune` (exit 1), and `--hparams` refuses a
`tuning.json` whose `tuning_seed` is one. Every departure from the
pinned seed, steps or 8 trials is written to `deviations` in `tuning.json` and
`tuning_deviations` in `run.json`. `tuning.json` also holds `tuning_seed`, `tuning_steps`,
`tuning_trials_used`, `trial_set_digest`, the trials and the `selected` values
(`frozen_for_all_seeds`). Runs with `--hparams` copy `tuning_seed`, `tuning_steps`,
`tuning_trials_used`, `tuning_trial_set_digest`, `tuning_selected` and `tuning_deviations` into
their own `run.json` and train the full budget.

GPU time per arm, a rough ESTIMATE that has NOT been measured (no GPU was available when this
was written): tuning is 8 x 500 = 4,000 steps and the seeds are 5 x 2,000 = 10,000 steps, so
14,000 optimizer steps per arm plus evaluation generation (greedy up to 256/384/128 new tokens
over GSM8K test 1319 + HumanEval and BFCL held-out items, plus the 747/55/133-item validation
sets after each of the 8 trials). If one step (batch 16-64 split in micro-batches of 4, bf16,
gradient checkpointing, up to 1024 tokens) takes a guessed 1.5-4 s, that is about 6-16 h of
training per arm, so roughly 1-3 days for four arms before evaluation. Measure with the first
`--max-steps 20 --model olmoe` run (`metrics.json` `training`) and rescale; batch 64 trials
will be the slowest.

`--max-steps` overrides the pinned 2000 steps (recorded as `smoke_override`), `--eval-limit`
caps items per domain, `--device auto|cpu|cuda`, `--micro-batch-size` sets gradient accumulation
(default 4 on OLMoE), `--gradient-checkpointing/--no-gradient-checkpointing` (default on for
OLMoE). `--tune --tune-budget 2` exercises tuning cheaply; `--all-seeds` takes no `--seed`.

Outputs: `run.json` (arm, seed and `seeds` per RNG consumer, pins, corpus selection digests per
split, validation-overlap counters, the measured `training_outside_training_partition_count` per
corpus, expert-domain assignment digest vs the pinned one, trainable parameters and their closed
form (a mismatch, or on OLMoE a difference from the pinned count, stops the run), the loaded
model's `topology_observed` with `model_pin_mismatches` (OLMoE: against the pinned topology and
the model and tokenizer commits, resolved on the hub or from the HF cache snapshot; an
unverifiable commit counts as a mismatch and in `model_pin_unverified`; tiny: a config digest),
hyperparameters and their source, `training_config`,
tuning trials used and trial-set digest, versions, device, sandbox mode, git commit,
`provenance` (see "GPU path"), the list of unpinned choices), `metrics.json` (per-domain accuracy
also under the manifest name `task_accuracy_per_domain`, `mean_task_accuracy` = macro average,
`latency_per_token` per domain (evaluation generation seconds / generated tokens; definition in
`UNPINNED_CHOICES`), `training_flops` (measured: the sum of `FlopCounterMode` over every
optimizer step; `training.flops_per_step` lists them), the per-corpus validation digests and
overlap counters keyed by the pinned names `gsm8k_train`, `humaneval`, `bfcl_simple` (also in
`run.json` and `tuning.json`),
`load_cv`, `routing_entropy` and their source, per-domain adapter-router stats, base-router
stats, trainable params, training loss, `not_produced`:
`expert_specialisation_index` has no pinned definition), `tuning.json` (`--tune`: see above).
A multi-process launch (`WORLD_SIZE` > 1) is refused.

Audit fixes (2026-10-13, tests in `training/tests/test_moe_audit*.py`): GSM8K item ids carry the
source split (`gsm8k:train:9` vs `gsm8k:test:9`; before, 131 validation ids collided with test ids
on the real corpus and the overlap counter would have invalidated every run); GSM8K scores the
first `####` answer; HumanEval runs the completion in a sandboxed worker that holds no secret
and never sees the tests, while a trusted harness runs `check()` against a proxy to it and
alone reports the result (an early `sys.exit(0)`, and later a forged pass marker, passed before); tuning on a confirmatory seed is refused.

Implemented and tested on CPU: all four arm topologies and their counts, domain-masked routing
(training and evaluation), corpus partitions and digests, GSM8K / HumanEval / BFCL-simple
scorers, SFT, tuning, the metrics. Not run: nothing here has touched the real OLMoE weights, the
real corpora, a GPU, or a 2000-step budget; the accuracies of the tiny smoke are meaningless.
Known gaps and unpinned choices:

- BFCL scoring is a simplified single-call AST check for the `simple` category, not the official
  BFCL evaluator (the exec check does not apply to `simple`).
- HumanEval runs candidate code in a subprocess (10 s timeout, scrubbed environment, rlimits)
  inside a bubblewrap sandbox (allowlisted read-only root, no network); without bubblewrap it
  refuses unless `PTR_M011_ALLOW_UNSANDBOXED=1` (then `unshare --net` when usable, else only a
  Python-level socket block), recorded in `run.json` `sandbox`.
- Training FLOPs are counted with `FlopCounterMode` on every optimizer step (forward,
  checkpoint recompute, backward) and summed; counting every step adds wall time, accepted
  because the compute-matching criterion needs the measured total. The old first-step x steps
  extrapolation is kept only as `training.extrapolated_total_flops_from_first_step` (not a
  criterion value). The 5% compute-matching check across arms is not computed here.
- Not pinned by the config and chosen here: optimizer (AdamW, constant lr, clip 1.0),
  domain-balanced batch sampling, PERFT activation (SiLU, `W_up` zero-init), gate weights
  (softmax over the selected logits), no balancing loss on the adapter routers, max length,
  generation lengths, the BFCL possible-answer file path, default untuned hyperparameters. The full list is `UNPINNED_CHOICES` in `m011.py`.
- Dataset field names were checked against the dataset cards / files at the pinned revisions
  (GSM8K `question`/`answer`, HumanEval `task_id`/`prompt`/`canonical_solution`/`test`/
  `entry_point`, BFCL `id`/`question`/`function` plus `possible_answer/` `ground_truth`);
  `datasets.load_dataset` itself has not been exercised against the hub here.
