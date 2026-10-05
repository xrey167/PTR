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
- `--execute` refuses `trl-moe` unless `[training] allow_unselected_backend = true` (only the
  smoke config sets it): the training-backend component decision is `unlocked` and `trl-peft`
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
gate results or the skip reason, `nan_check`, `evidence_class`.
Both RL arms start from the same restored SFT state with the same seed, so their first
generation batch is identical by construction.
