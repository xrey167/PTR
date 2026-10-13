# GPU handover: running the GPU-phase work on the local GPU desktop (originally written for one RTX 4090)

Written 2026-10-08 for the PTR owner's Ubuntu 24.04 desktop, originally assuming one NVIDIA RTX 4090
(24 GiB). The machine is two RTX 3090 (24 GiB each); read "4090" below as the reference card of the
original text (section 5 and `DESKTOP_SETUP.md` section 10 state what changes).
It is a plan, not a result. It was assembled from repository files on a machine without a
GPU, nothing in it has run on a GPU, and no experiment mentioned here has evidence yet.

Conventions used throughout:

- A statement without a mark comes from a repository file (the file is named next to it).
- **Estimate** marks anything inferred or computed here, including arithmetic on numbers
  from the files. Treat it as a guess until a run measures it.
- **OWNER DECISION** marks a choice the files leave open. Section 8 collects all of them.
- The setup commands, troubleshooting and result locations are in
  [`docs/DESKTOP_SETUP.md`](DESKTOP_SETUP.md); this document does not repeat them, it adds the
  run order and the experiment-by-experiment gaps. Hosting and the GitLab runner are in
  [`docs/HOSTING.md`](HOSTING.md), the Claude Code plugins in
  [`docs/CLAUDE_TOOLING.md`](CLAUDE_TOOLING.md).

## 0. The plan at a glance

| # | Step | Command or entry | What exists | Blocked by |
|---|---|---|---|---|
| 1 | Tiny GPU smoke | `make smoke-gpu` | config, target | nothing |
| 2 | OLMoE SFT smoke, measure memory | `make olmoe-sft-smoke` | config, target | step 1 |
| 3 | Derive the step-3 config | edit a copy of `stage-moe-olmoe.toml` | the 3090-profile config | step 2 measurement, OD-1 |
| 4 | M011 routed PEFT | `python -m ptr_training.moe.m011` | entrypoint module, pinned manifest | manifest entrypoint, preregistration, OD-1 |
| 5 | M012 LoRA expert balancing | none | pinned manifest | balancing rules, entrypoint |
| 6 | M014 AIMER prune and recovery | none | pinned manifest, AIMER in `ptr-lineage` and `ptr_py` | pruning, MMLU loader, entrypoint |
| 7 | M010 MoE RL stability | none | pinned manifest; GSPO and SAPO in the pipeline | GRPO, routing replay, REINFORCE++, entrypoint |
| 8 | M013 expert lifecycle | none | pinned manifest | lifecycle operations, L3 policy, entrypoint |
| 9 | M015 RLVR vs distillation | none | pinned manifest | 30B teacher does not fit (OD-4) |
| 10 | R008 lossy KV tier | none | pinned manifest | tier mechanism, harness |
| 11 | M016 approximate chunk reuse | none | manifest | R006 completed first, harness, OD-1 |
| 12 | R004 adapter lineage | none | manifest, `ptr-lineage` | two placeholders, backbone, harness |
| 13 | E006 engine prefix reuse | none | manifest | vLLM and SGLang environment, harness |
| 14 | R003, E001 to E005 | none | stubs | pinned backbone (ADR-0021) |

Steps 1 to 3 are configuration runs. Every later step is an experiment and follows the
freeze-then-run discipline of section 3. The VCW-MoE experiments (F004 to F006, C002, C003, C005,
C006) and their checkpoint stage are in section 10, planned in
[`VCW_MOE_GPU_PLAN.md`](VCW_MOE_GPU_PLAN.md); they start only after steps 1 to 4 above.

## 1. Prerequisites and one-time setup

### 1.1 Checklist

Work top to bottom. Each item has a check; do not continue past a failed one.

1. **Ubuntu 24.04 LTS, native** (not WSL2), stock or HWE kernel, `sudo apt update && sudo apt
   full-upgrade`, reboot. At least **200 GB free** after the install (DESKTOP_SETUP step 1).
   Check: `df -h ~`.
2. **Secure Boot.** `mokutil --sb-state`. With Secure Boot on, enrol the MOK key at the first
   reboot after the driver install, or disable Secure Boot (DESKTOP_SETUP step 1).
3. **Driver.** `sudo apt install -y nvidia-driver-580-open`, reboot. Check: `nvidia-smi` shows
   driver 580.x and two lines "NVIDIA GeForce RTX 3090", about 24 GiB each (the original single-4090
   text expected one RTX 4090, 24564 MiB; DESKTOP_SETUP step 2). No system CUDA
   toolkit is needed: the locked CUDA 13 torch wheels bring their own runtime. The experiment
   manifests of M016 and E006 pin driver `580.65.06` and CUDA 13.0 exactly
   (`software_stack`, `max_software_stack_mismatches = 0`). Which 580.x patch version apt
   installs is not stated in any file, so record `nvidia-smi` output and compare it with the
   pin before freezing those two (OD-6).
4. **Base tools and clone.** `sudo apt install -y git curl build-essential pkg-config`, set
   `git config --global user.name/user.email`, clone the repository (DESKTOP_SETUP step 3).
   GitHub is suspended; use the GitLab remote or a git bundle (HOSTING.md, section 6 here).
5. **`make desktop-setup`** (idempotent, never runs `sudo`; `--dry-run` prints the steps). It
   installs rustup with the pinned 1.99.0 toolchain, `uv`, the training venv
   (`uv sync --project training --python 3.11 --extra moe --locked`, CUDA torch wheel from
   `training/uv.lock`) and `ptr_py` into that venv, then runs the doctor. Open a new shell
   afterwards.
6. **`make gpu-doctor`**, preferably `python3 scripts/gpu_doctor.py --require-gpu` (the
   Makefile target runs the plain form, where a missing GPU is a warning). Must exit 0. It
   warns below 60 GB free at the repository and at the Hugging Face cache.
7. **Python tests on the desktop.** `PYTHONPATH=training/src make python-test` (CPU tests;
   the target does not set `PYTHONPATH`). Optionally, with the training venv so the torch
   tests run too:
   `PYTHONPATH=training/src training/.venv/bin/python -m unittest discover -s training/tests`.
   A red test here means the environment differs from the one the code was tested in.
8. **Hugging Face cache and login** (DESKTOP_SETUP step 6). OLMoE is not gated; the login only
   avoids anonymous rate limits. Never put a token in the repository.
   **Keep the default cache location for frozen runs.** `scripts/run_experiment.py` hands a
   listed experiment's command only `PATH, HOME, TMPDIR, TEMP, TMP, LANG, LC_ALL, LC_CTYPE,
   CARGO_HOME, RUSTUP_HOME` plus `PYTHONNOUSERSITE=1` and `PYTHONHASHSEED=0`
   (`COMMAND_ENVIRONMENT`, `FIXED_ENVIRONMENT`). `HF_HOME`, `HF_TOKEN`, `PYTHONPATH` and
   `CUDA_VISIBLE_DEVICES` do not reach it. If the cache lives on another disk, make
   `~/.cache/huggingface` a symlink to it instead of exporting `HF_HOME` (estimate: the
   symlink is resolved by the filesystem and needs no environment).
9. **Model and dataset downloads at pinned revisions** (1.3).
10. **Claude Code and plugins** (DESKTOP_SETUP step 8, section 7 here).
11. **Clean tree rule.** `--execute` refuses a dirty tracked worktree, so commit or stash
    before every run. Run outputs under `training/runs/` are gitignored; never commit them.

### 1.2 Disk, as far as the files state it

| Item | Size | Source |
|---|---|---|
| Free space to keep after the OS install | at least 200 GB | DESKTOP_SETUP step 1 |
| Rust `target/` | "several tens of GB" | DESKTOP_SETUP step 1 |
| Locked CUDA torch wheels in `training/.venv` | about 8 GB | DESKTOP_SETUP step 1 |
| `allenai/OLMoE-1B-7B-0125` in the HF cache | about 14 GB | DESKTOP_SETUP step 6 |
| Doctor warning threshold (repo and HF cache) | 60 GB | DESKTOP_SETUP step 5 |

No file states the size of Qwen/Qwen2.5-0.5B, Qwen/Qwen3-30B-A3B, the datasets, a vLLM or
SGLang environment, checkpoints, or `training/runs/` output. Estimate for the 0.5B model:
about 1 GB in bf16 (0.5e9 parameters times 2 bytes). Estimate for the 30B teacher: about
60 GB in bf16 (30e9 times 2 bytes), which is why M015 needs OD-4 first.

### 1.3 Pinned downloads

Revisions come from the manifests and configs named in the right column. Never use `main`.
Run once, with the training venv, before the timed steps:

```bash
training/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download

# Steps 1-9, 14: base model of M010-M015, E006 (training/configs/*olmoe*.toml, M010-M015 config.toml)
print(snapshot_download("allenai/OLMoE-1B-7B-0125",
                        revision="9b0c1aa87e34a20052389dce1f0cf01da783f654"))

# M011, M013 corpora (M011 config.toml: corpus_*_revision); also used by M010, M012, M014, M015 (GSM8K)
for repo, rev in [("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
                  ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544")]:
    print(snapshot_download(repo, repo_type="dataset", revision=rev))
print(hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "BFCL_v3_simple.json",
                      repo_type="dataset", revision="61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"))

# M013 public suite, M014 gate and quality slices (config.toml: public_suite_revision, quality_suite)
print(snapshot_download("cais/mmlu", repo_type="dataset",
                        revision="c30699e8356da336a370243923dbaf21066bb9fe"))

# R008, M016 (config.toml: agreement_model_revision, test_model_revision)
print(snapshot_download("Qwen/Qwen2.5-0.5B",
                        revision="060db6499f32faf8b98477b0a26969ef7d8b9987"))
PY
```

Not in the snippet on purpose:

- `Qwen/Qwen3-30B-A3B` at `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (M015 teacher). Do not
  download it before OD-4 is decided.
- `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` (strong_rag): their revisions are empty in
  `research/baselines/strong_rag/config.toml` (`status = "blocked-unpinned-model-revisions"`).
- The coding backbone of R003: `training/continued-pretraining/model.toml` says `base_model_id =
  "unspecified"`.
- LongMemEval and LoCoMo (E005): no revision is pinned in any file I read.

### 1.4 Two Python environments, probably three

The training venv (`moe` extra) asks for `torch>=2.14.1`. The M016 and E006 manifests pin
`torch 2.13.0` for vLLM 0.30.0 and SGLang 0.5.21 (`software_stack`, with
`software_stack_basis` saying why), and no vLLM or SGLang package is in `training/uv.lock`
(I found no `vllm` or `sglang` entry). Estimate: M016 and E006 need their own environment,
separate from `training/.venv`, and a harness that records its versions in `run.json`
(`software_stack_recorded_in_run_json`). The `cpt` extra (axolotl, deepspeed, bitsandbytes)
conflicts with `moe` in `training/pyproject.toml`, so R003's training would be a third.

## 2. What the GPU runs may and may not claim

Steps 1 to 3 are pipeline evidence only (DESKTOP_SETUP step 7, the `experiment-run` skill):
one seed, no baseline, no ablation, backend still `unselected`
(`backend_selection.status`). Their outputs say "pipeline evidence only" and must never
be described as improving, outperforming or specialising. A research claim exists only
for an experiment that completed under its frozen preregistration, against its own
falsification criteria, seeds and baselines, with negative results kept. Under ADR-0021,
pretrained-LM comparisons are the GPU phase; A0-scale results stay A0-scale.

## 3. Freeze-then-run discipline (applies to every experiment step)

Sources: `experiments/README.md` (Preregistration, Runner), `experiments/preregistration.toml`,
`scripts/run_experiment.py`, `scripts/check_research_gates.py`, `CLAUDE.md`.

### 3.1 The sequence

1. **Validate.** `python3 scripts/run_experiment.py validate` and
   `python3 scripts/check_research_gates.py` must pass before and after every change.
2. **Prepare the record in a branch, as a normal commit.** While the experiment is still
   `planned`: name the `entrypoint` in `experiment.toml` (it is empty for every experiment in
   this document; the manifest is frozen from the first commit past `planned`, so an
   entrypoint named later could never be named), write the `[preregistration]` table in
   `config.toml`, list the experiment in `experiments/preregistration.toml` with its
   required keys and types, and set `preregistration_sha256` and
   `preregistration_rules_sha256` in the manifest. The full list of rules is in
   `experiments/README.md`; `scripts/experiment_records.py` computes the digests. Follow the
   pattern of R005, R009 to R012 (each has `PREREGISTRATION.md`, `criteria.toml` and
   `aggregate.py`; R006 has them in draft).
3. **Freeze commit.** Every `must-be-pinned-...` and `must-be-signed-...` string must be gone
   (an owner signs a decision by committing the decided value). After the freeze commit the
   manifest, `config.toml` and the preregistration never change; status only moves forward
   (`prepared` to `running` to `completed` or `failed`). A change means a successor
   experiment.
4. **`python3 scripts/run_experiment.py prepare <ID>`** writes the `prepared` record at HEAD.
   Commit it.
5. **One run per seed.** `python3 scripts/run_experiment.py run <ID> --seed <S>` for
   S in 17, 29, 43, 71, 101 (the seeds every manifest lists), one at a time (the runner holds a
   lock per experiment). A listed experiment runs each seed once; a seed that ran is spent, even
   if the run failed or was unwanted. Commit each seed's record before the next launch
   (the runner reads only committed records).
6. **Aggregate once** with the experiment's `aggregate.py` against its `criteria.toml`. This
   writes `results/run.json` (which names the SHA-256 of the `metrics.json` written with it
   and the digest of each seed's record) and `results/metrics.json`. Then set the status to
   `completed` or `failed` and commit.
7. **No reruns.** Never overwrite or delete a result, even a failed or negative one. A rerun
   is a successor experiment with its own preregistration (`CLAUDE.md`; ADR-0021 says the
   same for A0 history).

### 3.2 Facts about listed runs that matter on this machine

- Only experiments named in `experiments/preregistration.toml` get these protections. Today
  the list holds S003, F003, Q003-v2 (superseded; the successor of the superseded Q003), Q003-v3 (the successor of the superseded Q003-v2), R004, M008, E005, R007, R009 to R012 and the A0 records.
  **None of M006, M010 to M016, R002, R003, R008, E001 to E004 or E006 is listed.**
  `CLAUDE.md`: unlisted prepared records are not covered by the freeze check. Listing them
  before the first prepared record is a project decision (OD-2).
- A listed run refuses to start while the repository holds git-ignored files
  (`ignored_files`, `git clean -ndX` lists them) and while it holds files git does not list.
  `.gitignore` ignores `.venv/` and `training/runs/*`, so a `training/.venv` inside the
  checkout makes a listed run refuse. The error text says "a fresh worktree holds none".
  Estimate: for frozen runs, put the Python environment outside the checkout (for example
  with `UV_PROJECT_ENVIRONMENT` at `uv sync` time) and run from a clean worktree; how the
  frozen `entrypoint` then names its interpreter, given that `PYTHONPATH` and the venv path
  are not passed, is not solved anywhere in the repository and must be settled when the first
  GPU entrypoint is written (OD-3).
- The command runs from the repository root with the restricted environment of section 1.1
  item 8. Rust builds go to a scratch directory outside the checkout.
- `make smoke-gpu`, `make olmoe-sft-smoke` and direct `python -m ptr_training.moe.m011` calls
  are not listed-experiment runs and none of the above applies to them. That is also why
  their output is exploratory, not a result.
- The `ptr-dev` skills `ptr-verify`, `experiment-run` and `ptr-results` encode these rules
  (section 7).

## 4. Ordered run plan

Each step lists: what exists, what is missing before the experiment can be preregistered
(placeholders by name, missing code, owner decisions), the expected VRAM fit on 24 GiB, and the
commands.

### Step 1. Tiny GPU smoke

- **Exists.** `training/configs/stage-moe-smoke-gpu.toml` (tiny random OLMoE: 2 layers,
  hidden 64, 8 experts, top-2, fp32, `device = "cuda"`, SFT 200 steps, then GSPO and SAPO 16
  steps each, `hardware/rtx4090.toml`; on this machine the 3090 variant `stage-moe-smoke-gpu-rtx3090.toml`), Makefile target `smoke-gpu`. The config header says it
  has not run on a GPU.
- **Commands.**
  ```bash
  python3 scripts/gpu_doctor.py --require-gpu
  git status --short          # must show no tracked changes
  make smoke-gpu              # PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.run --config training/configs/stage-moe-smoke-gpu.toml --execute
  ```
- **Check.** Exit 0. `metrics.json` under `training/runs/moe-smoke-gpu/` reports
  `hardware.device = "cuda"`, `hardware.gpu` with the 4090 name, `hardware.torch_cuda`, and a
  peak memory value. The run manifest is `training/runs/run-<timestamp>.json`
  (`git_dirty` must be false). Compare with the WP8 CPU table in `training/README.md`
  ("First training (WP8): CPU smoke run"): same range for pass rate, routing entropy, load CV,
  dead experts and SFT loss, with the caveat that WP8 was CPU, one seed, tiny model.
- **VRAM.** Tiny model; no number in the files. Estimate: well under 1 GB.
- **Discipline.** Not an experiment. Record the manifest path and git sha.

### Step 2. OLMoE SFT smoke, measure memory

- **Exists.** `training/configs/stage-moe-olmoe-sft-smoke.toml` (OLMoE-1B-7B-0125 at the
  pinned revision, bf16, LoRA rank 16 / alpha 32 on `q_proj k_proj v_proj o_proj` and the
  fused `mlp.experts.gate_up_proj` / `down_proj` via `target_parameters`, router frozen,
  gradient checkpointing, micro-batch 2 times 4 accumulation steps, `max_length` 256, 50
  optimizer steps, `[moe.rl] algorithms = []`, `allow_unselected_backend = true`,
  `hardware/rtx4090.toml`; on this machine `stage-moe-olmoe-sft-smoke-rtx3090.toml`). Header: "NOT RUN anywhere yet".
- **Commands.** Run it in the background with the `experiment-run` skill, or by hand:
  ```bash
  mkdir -p training/runs/logs
  LOG=training/runs/logs/olmoe-sft-smoke-$(date +%Y%m%dT%H%M%S).log
  nohup make olmoe-sft-smoke > "$LOG" 2>&1 &  echo $! > "${LOG%.log}.pid"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 30
  ```
  Read the measured peak after it finishes:
  ```bash
  ls -t training/runs/*/*/metrics.json | head -1
  python3 -c 'import json,sys; h=json.load(open(sys.argv[1]))["hardware"]; print(h["peak_cuda_memory_allocated_bytes"], h["peak_cuda_memory_allocated_gib"])' <that file>
  ```
- **VRAM.** The config comment's own estimate, not a measurement: about 12.9 GB of bf16
  weights (6.9B parameters times 2 bytes), tens of MB of LoRA and AdamW state, and
  checkpointed activations in the low single-digit GB; several GB of headroom expected.
  `peak_cuda_memory_allocated_*` is the torch allocator peak; `nvidia-smi` shows more (CUDA
  context, cache fragmentation; estimate: about 1 GB for the desktop session alone, as
  DESKTOP_SETUP step 9 says). Record both.
- **If it runs out of memory.** Lower `batch` and raise `gradient_accumulation_steps` first
  (the config says so), set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, keep bf16 and
  checkpointing, and never drop the pinned LoRA targets silently (DESKTOP_SETUP step 9).
  Record every config change in the run notes.
- **Discipline.** Not an experiment. The measured peak is the input to step 3.

### Step 3. Derive the step-3 config

- **Exists.** `training/configs/stage-moe-olmoe.toml`: SFT 100 steps, batch 4, `max_length` 48;
  GSPO and SAPO 50 steps, 8 generations, `mini_steps` 4, `beta` 0.0 (no reference copy of the
  7B model), prompts per step 2, prompt length 32, completion length 24. Its header says it
  names `hardware/rtx3090.toml`, sets neither gradient checkpointing nor
  `allow_unselected_backend`, and that `--execute` therefore refuses it until a step-3 config
  is "derived deliberately from the SFT smoke's measured memory".
- **Do.** Copy it to a new file (for example `training/configs/stage-moe-olmoe-gpu.toml`; the
  name is a suggestion), then change deliberately and list each change in the commit message:
  `[run] name` (unique), `[model] config` (the new path), `hardware_profile` (OD-1),
  `[training] allow_unselected_backend = true`, `[moe.sft] gradient_checkpointing = true` (and
  `gradient_accumulation_steps` if the step-2 peak calls for it), a distinct `[moe.output] dir`.
  Sizes (`batch`, `max_length`, `max_completion_length`, `num_generations`) follow the step-2
  measurement. Commit it, then `make olmoe-sft-smoke OLMOE_SFT_CONFIG=training/configs/<new>.toml`
  (the variable is a plain override of the SFT target; it runs whatever config it is given).
- **VRAM.** RL does not use gradient checkpointing (`moe/README.md`: it is switched off when
  SFT ends), and no file measures RL-phase memory. Estimate: generation with 8 sampled
  completions per prompt adds a KV cache and logits on top of the step-2 footprint; read the
  first RL-phase peak from the new `metrics.json` rather than trusting this.
- **Discipline.** Not an experiment. This config is the reference for the RL-heavy steps 5 to 9.

### Step 4. M011 routed PEFT (PERFT-R) on OLMoE (the first experiment per DESKTOP_SETUP)

- **Exists.** `experiments/model/M011-routed-peft/` (`status = "planned"`): four arms
  (`perft_r_emergent`, `perft_r_prescribed`, `plain_lora`, `mixlora`) at a common target of
  4,456,448 trainable parameters (each within 1%), corpora GSM8K, HumanEval and BFCL
  `BFCL_v3_simple.json` at pinned revisions with pinned split rules, 2000 steps per arm,
  tuning on the pilot seed 1001, never a confirmatory seed (8 trials of 500 steps, search space learning rate 1e-5 to 1e-3 and batch
  size 16, 32 or 64), seeds 17 29 43 71 101, the criteria in `experiment.toml`. The manifest
  and `config.toml` hold no `must-be-pinned` or `must-be-signed` string. The entrypoint module
  `training/src/ptr_training/moe/m011.py` exists with `routed_peft.py`, `corpora.py`,
  `scorers.py` and tests (`test_m011_*`). Its flags: `--arm`, `--seed` or `--all-seeds`,
  `--out`, `--model tiny|olmoe`, `--tune`, `--hparams`, `--max-steps`, `--device`,
  `--micro-batch-size` (default 4 for OLMoE), `--corpora fixture|hf`,
  `--gradient-checkpointing / --no-gradient-checkpointing` (on by default for OLMoE).
- **Missing before it can be preregistered.**
  - `entrypoint` is empty in `experiment.toml`; no `make` target exists (DESKTOP_SETUP step 7).
  - No `[preregistration]` table, no entry in `experiments/preregistration.toml`
    (OD-2), no `criteria.toml`, no `aggregate.py`. Nothing yet maps `m011.py`'s `run.json` and
    `metrics.json` onto the runner's run records and the aggregate (estimate: a thin wrapper or
    an aggregator that reads them is needed).
  - `m011.py` records choices the manifest does not pin in `UNPINNED_CHOICES` (optimizer
    AdamW with constant learning rate and clip 1.0, domain-balanced sampler, loss, PERFT
    activation, maximum lengths, generation lengths, a simplified single-call BFCL scorer
    instead of the official one, HumanEval in a subprocess with a 10 s timeout). Decide whether
    to pin them in the `[preregistration]` table before the freeze (OD-5).
  - OD-1 (hardware profile) and OD-3 (interpreter and environment for frozen runs).
- **VRAM.** Not stated for M011. The base checkpoint is the one of step 2 (about 12.9 GB bf16,
  estimate from the config comment); the four arms add a few million trainable parameters (PERFT-R
  4,456,448). Whether batch sizes up to 64 fit is decided by the micro-batch (4 by default)
  and checkpointing, measured in step 2; estimate: fits, with activations as the open term.
- **Compute, arithmetic from the manifest (estimate for wall time, none stated).** Per arm: 8
  tuning trials of 500 steps plus 5 seeds of 2000 steps is 14,000 optimizer steps; four arms
  are 56,000. The `experiment-run` skill says one arm takes "many hours". Measure the seconds
  per step in a short real run first.
- **Commands.**
  ```bash
  # CPU pipeline check, nothing downloaded (pipeline evidence only)
  PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.moe.m011 \
    --arm perft_r_prescribed --seed 1 --out /tmp/m011-smoke --model tiny --max-steps 2 --eval-limit 3
  # short real-model GPU check before any freeze (exploratory)
  PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.moe.m011 \
    --arm mixlora --seed 1001 --out training/runs/m011/probe --model olmoe --corpora hf --max-steps 20
  ```
  The real runs per arm (tune once, then every seed) are in the `experiment-run` skill, under
  "M011". They run only after the freeze and through `run_experiment.py` as in section 3.
- **Discipline.** Section 3. Tuning happens once, on the pilot seed 1001, inside the preregistered
  procedure; hyperparameters are then frozen for every seed. Any departure from the pinned
  seed, step count or 8 trials is recorded in `deviations` and must be quoted.

### Step 5. M012 LoRA expert balancing

- **Exists.** `experiments/model/M012-lora-expert-balancing/` (`planned`): arms `aux_free_bias`,
  `aux_loss`, `phi_balancing` over 8 LoRA experts (rank 8, alpha 16, top-2, deltas on the fused
  expert tensors) on OLMoE, GSM8K, 2000 steps per arm, evaluation on the first 8192 test
  tokens, tuning 8 trials of 500 steps with learning rate, batch size, `aux_coefficient` and
  `bias_update_rate` in the search space. No `must-be-*` placeholders.
- **Missing.** Entrypoint; the three balancing rules (I found no auxiliary-loss-free bias,
  auxiliary balance loss or phi-balancing code under `training/src`); a LoRA-experts model
  with a router (`routed_peft.py` has a MixLoRA-style arm for M011; whether it can be reused
  is an estimate); preregistration, `criteria.toml`, `aggregate.py`; OD-1.
- **VRAM.** Not stated. Same base model as step 2 (estimate: the adapter is small; the
  footprint is again dominated by weights and activations).
- **Run order reason.** It reuses M011's corpora, tuning procedure and model plumbing, so it
  comes straight after M011.

### Step 6. M014 AIMER prune and recovery

- **Exists.** `experiments/model/M014-aimer-prune-recovery/` (`planned`): AIMER vs random vs
  usage-frequency pruning of 25% (16 of 64 experts) and 50% (32), each followed by SFT (2000
  steps) and GSPO or SAPO recovery (1000 steps), plus unpruned controls with the identical
  recovery per sequence and seed; recovery on all 7473 GSM8K train items; quality on MMLU
  questions 2000 to 3999, ForgettingGate on questions 0 to 1999 with four pinned thresholds
  (0.02, 0.05, -0.02, 0.01). AIMER scoring and ranking exist in `crates/ptr-lineage/src/aimer.rs`
  and `ptr_py.aimer_score` (`bindings/ptr-py`), and the SFT, GSPO and SAPO pipeline exists in
  `ptr_training.moe`. No `must-be-*` placeholders.
- **Missing.** Entrypoint; physical expert removal on OLMoE (`pruned_experts_physically_removed`);
  random and usage-frequency pruning; an MMLU loader (none under `training/src`); a VRAM
  measurement harness (`vram_batch_size = 4`, `vram_sequence_length = 2048`); preregistration,
  `criteria.toml`, `aggregate.py`; OD-1.
- **Size, arithmetic from the manifest (estimate).** Per seed 3 methods times 2 fractions
  times 2 sequences is 12 pruned arms, plus 2 unpruned controls (one per sequence), so 14
  recovery runs per seed and 70 over 5 seeds, each 2000 SFT plus 1000 RL steps.
- **VRAM.** Not stated beyond the two VRAM-measurement parameters. Pruned models are smaller
  than the unpruned control, so the control with RL (batch 64, 8 samples per prompt,
  completions up to 512 tokens in `recovery_protocol`) is the binding case (estimate). That RL
  setting is much larger than step 3's, so step 3's measured peak does not cover it.

### Step 7. M010 MoE RL stability

- **Exists.** `experiments/model/M010-moe-rl-stability/` (`planned`): ten pinned arms (GRPO
  without routing replay, GRPO with replay, GSPO, SAPO, REINFORCE++ at 4 mini-steps, GRPO,
  GSPO and SAPO at 1 mini-step, and two arms with a "separate" rollout engine), 2000 optimizer
  steps per arm, GSM8K, rollout protocol of 8 generations, temperature 1.0, completions up to 512
  tokens, `kl_beta = 0.0`, shared hyperparameters tuned once (8 trials of 500 steps, learning
  rate 1e-7 to 1e-5, batch size 32, 64 or 128). No `must-be-*` placeholders. GSPO and SAPO
  exist in the pipeline (`moe/config.py`: `ALGORITHMS = ("gspo", "sapo")`).
- **Missing.** Entrypoint; GRPO with clip-higher, routing replay and REINFORCE++ (the pipeline
  accepts only `gspo` and `sapo`); a definition and implementation of the "separate" rollout
  engine (the config names the arms but I found no definition beyond the names; whether this
  means vLLM is not stated); the metrics (`expert_churn_per_layer`,
  `policy_ratio_variance_per_token` and others); preregistration, `criteria.toml`,
  `aggregate.py`; OD-1.
- **VRAM.** Not stated. Step 3 measures a 8-completion, 24-token RL setting; this experiment
  uses 512-token completions and batch sizes up to 128 (prompts per step = batch size / 8), so
  memory is likely the hardest of the OLMoE steps (estimate). Do not freeze before a
  short real run of the largest pinned setting has fit.

### Step 8. M013 expert lifecycle

- **Exists.** `experiments/model/M013-expert-lifecycle/` (`planned`): five arms
  (`lifecycle_full_context`, `lifecycle_context_ablated`, `lifecycle_heuristic`,
  `static_expert_set`, `naive_adapter_chain`) over the domain sequence code, math, tool, code
  with `max_recovery_steps = 5000`, 8 initial LoRA experts (rank 8, top-2), corpora
  GSM8K, HumanEval, BFCL and MMLU as public suite (revision pinned). No `must-be-*` placeholders.
- **Missing.** Entrypoint; the lifecycle operations (Birth, Split, retirement, reactivation,
  fresh-expert control) and the L3 proposal policy with its context variants; the heuristic
  baseline; an MMLU loader; the gate-suite and proposal-opportunity machinery;
  preregistration, `criteria.toml`, `aggregate.py`; OD-1. I found no lifecycle code under
  `training/src`.
- **VRAM.** Not stated; same base model and LoRA scale as steps 4 and 5 (estimate).
- **Run order reason.** It needs the most new machinery of the OLMoE family.

### Step 9. M015 RLVR vs distillation (pass@k) - flagged: the teacher does not fit

- **Exists.** `experiments/model/M015-rlvr-vs-distill-passk/` (`planned`): arms `base`, `rlvr`,
  `sft_distilled`; student OLMoE-1B-7B; teacher `Qwen/Qwen3-30B-A3B` at
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, 4 samples per prompt at temperature 0.7, top-p
  0.95, up to 512 new tokens, seed 20240607; 1000 training steps per arm; evaluation of pass@1
  and pass@256 on all 1319 GSM8K test problems. No `must-be-*` placeholders.
- **Does not fit as pinned.** A 30B-parameter teacher is about 60 GB in bf16 (estimate: 30e9
  times 2 bytes), against 24 GiB. The manifest pins no quantization or offload, so running the
  teacher in 4-bit (estimate: about 15 to 17 GB of weights) or with CPU or disk offload would
  be a departure from the pinned procedure, and so would sampling the teacher on other hardware.
  This is OD-4. Student-side steps (RLVR on OLMoE, evaluation) are within the OLMoE budget of
  steps 2 and 3 (estimate).
- **Size, arithmetic from the manifest (estimate).** Teacher: 4 samples times 6726 prompts is
  26,904 generations of up to 512 tokens. Evaluation: 256 samples times 1319 problems is
  337,664 generations per arm and seed, for 3 arms and 5 seeds.
- **Missing.** Entrypoint; teacher-sampling and distillation code; pass@k evaluation with
  confidence intervals; preregistration, `criteria.toml`, `aggregate.py`; OD-1. Run it last of
  the OLMoE family and only after OD-4.

### Step 10. R008 lossy KV tier (Qwen2.5-0.5B)

- **Exists.** `experiments/runtime/R008-lossy-kv-tier/` (`planned`, `hardware/default.toml`):
  claim of at least 2x KV-memory reduction at top-1 next-token agreement of at least 98% on
  pinned `Qwen/Qwen2.5-0.5B` (24 layers, hidden 896) at 512 tokens, 100 prompts, keep
  fractions 0.5, 0.25 and 0.125, always tagged Approximate, with revocation, supersession,
  restart and in-flight matrices. No `must-be-*` placeholders.
- **Missing.** The lossy tier itself: `TierObjectDomain::KvSnapshotLossy` is described in
  `docs/architecture/36-kv-reuse.md` (P6) but I found no `Lossy` tier type in `crates/`. Also
  the harness, entrypoint, preregistration, `criteria.toml`, `aggregate.py`, and the
  declared hardware (OD-1). The mechanism and the invalidation matrices can be built and tested
  on CPU first; only the agreement measurement needs the model.
- **VRAM.** The model is about 1 GB in bf16 (estimate); fits with large margin.
- **Software.** The manifest does not pin a software stack (unlike M016 and E006).

### Step 11. M016 approximate chunk reuse (Qwen2.5-0.5B)

- **Prerequisite.** R006 must be completed first. `docs/architecture/36-kv-reuse.md`: the
  only proven Exact positional-reuse case is a single-layer model "only once R006 has passed";
  M016's multi-layer shifted reuse goes through the secondary index and the shift admission
  rules that R006 establishes. R006 is `planned` (CPU, `hardware/default.toml`); its
  secondary index (WP1), config pins and harness skeleton are merged, but it has not been
  preregistered or run.
- **Exists.** `experiments/model/M016-approximate-chunk-reuse/` (`planned`,
  `hardware/rtx5090.toml`): position shifts 16, 64, 256; recompute candidates k in {4, 7, 10,
  13} per 64-token chunk; top-1 agreement at least 0.99 and median prefill speedup at least
  2.0 on 512-token prompts (3 warm-up and 11 timed runs); 100 calibration and 100 held-out
  GSM8K-derived prompts; a 20-cell `candidate_shift_results.json`; large invalidation and
  taint matrices. No `must-be-*` placeholders.
- **Missing.** R006; the harness and entrypoint; vLLM 0.30.0 environment with the pinned
  stack; preregistration, `criteria.toml`, `aggregate.py`.
- **Hardware.** The manifest declares the 5090 profile (32 GiB). The claim includes a measured
  wall-time speedup, which is hardware dependent, so declaring the wrong profile would
  mislabel the main result (OD-1).
- **VRAM.** 0.5B model, about 1 GB in bf16 (estimate); fits with large margin on 24 GiB.

### Step 12. R004 adapter lineage

- **Exists.** `experiments/runtime/R004-adapter-lineage/` (`planned`, listed in
  `preregistration.toml`): lineage policy (gated, forgetting-driven replay and consolidation)
  against a naive sequential LoRA chain, a single joint retrain and uniform replay; the
  `ptr-lineage` crate (AIMER, forgetting gate, replay, subspace, merge, scale) and `ptr_py`
  bindings; design in `docs/architecture/35-agentic-substrate.md`.
- **Missing, by name.** `public_regression_gate_permille` and
  `backward_transfer_min_permille` are `must-be-pinned-before-prepared` in `config.toml`. The
  manifest also names no backbone, no domain sequence, no public suite and no compute budget,
  and no training driver connects `ptr-lineage` to real adapter chains (the default
  backend is `dry-run`); those are not placeholders, so they would be added to the
  `[preregistration]` table or the manifest before the freeze (OD-8). Also the harness,
  entrypoint, `criteria.toml`, `aggregate.py`.
- **VRAM.** Cannot be stated: no model is named. If it uses the OLMoE-with-LoRA stack of step 2,
  the step-2 measurement applies (estimate).
- **Discipline.** Already listed, so the full freeze machinery applies from the first
  prepared record.

### Step 13. E006 engine prefix reuse (needs vLLM and SGLang)

- **Exists.** `experiments/system/E006-engine-prefix-reuse/` (`planned`,
  `hardware/rtx3090.toml`): vLLM 0.30.0 (batch-invariant mode, `VLLM_BATCH_INVARIANT=1`) and
  SGLang 0.5.21 with OLMoE-1B-7B at the pinned revision, 2048-token prompts with a
  1024-token shared prefix, `cache_salt`, revocation arms normal/lag/restart, a TOST test of
  median TTFT with 1000 samples per arm, a logistic-regression timing attacker. No
  `must-be-*` placeholders. Without a batch-invariant engine the outcome is `inconclusive`
  (`missing_batch_invariance_outcome`).
- **Needs from the 4090.** Batch invariance needs compute capability 8.0 or higher
  (`batch_invariance_min_compute_capability = "8.0"`); the 4090 is sm_89 (hardware profile
  notes), which qualifies. **But the manifest pins `gpu_compute_capability = "8.6"`**
  (the 3090), so on the 4090 the run record would disagree with the pin (OD-1).
  Exact driver `580.65.06` and CUDA 13.0 are also pinned (OD-6).
- **Missing.** vLLM and SGLang installed in their own environment (section 1.4), the engine
  harness (cache salts, per-principal revocation, timing), entrypoint, preregistration,
  `criteria.toml`, `aggregate.py`.
- **VRAM.** OLMoE-1B-7B weights are about 12.9 GB in bf16 (estimate from the step-2 config
  comment), so one engine instance should fit with the rest used for the KV cache. Run the
  two engines one after the other, and never while a training run is on the GPU. The
  engine's memory-utilization setting is not pinned in any file.

### Step 14. R003 and E001 to E005 - blocked on a pinned backbone

ADR-0021 (accepted 2026-10-06): the A0-scale baseline is the A0-matched plain model, and the
pretrained-LM baseline and its backbone are "a separate later decision". It states plainly that
E002 and R003 are not unblocked. Nothing here can start before that decision (OD-9).

| ID | Hardware profile | Entrypoint | What is missing, by name |
|---|---|---|---|
| R003 | `hardware/dual-rtx3090.toml` (2 x 24 GiB, 48 GiB combined, PCIe only) | empty | Backbone (`training/continued-pretraining/model.toml`: all fields "unspecified"); `training/configs/run-coding-pod-cpt.toml` is `backend = "dry-run"` with placeholder hyperparameters; the own-code corpus size (`own_code_corpus_v0_1`); ADR-0015 is Proposed. A single 4090 does not meet the declared profile. |
| E001 | default | empty | A pinned unmodified-backbone agent baseline; the end-to-end harness. |
| E002 | default | empty | Backbone; `strong_rag` is `blocked-unpinned-model-revisions` (dense and reranker revisions and the answer generator are empty); GraphRAG and editable-memory baselines. |
| E003 | default | empty | Matched backbone and runtime; a generated-token count. |
| E004 | default | empty | Full-context and strong RAG baselines with a model. |
| E005 | default | empty | `token_budget`, `full_context_baseline`, `agent_memory_baselines` and `baseline_strong_rag_sha256` are all `must-be-pinned-before-prepared`; LongMemEval and LoCoMo revisions are not pinned anywhere; the published memory systems to re-run (licences). Listed in `preregistration.toml`. |

Not in the plan: M006 (needs a trained diffusion-style model, no GPU profile named) and R002
(needs a trained tool-calling model; `benchmarks/ood-pods` holds no data).

## 5. Hardware profile question (RESOLVED: two RTX 3090, keep `rtx3090.toml`): M010 to M015 name the RTX 3090, the original text assumed a 4090

> **Resolved (owner fact, later than this text).** The owner's desktop has two RTX 3090 (24 GiB
> each), not a 4090. OD-1 therefore resolves in favour of option A: the `hardware/rtx3090.toml`
> profile the M010 to M015 and E006 manifests already name is the right one, and
> `gpu_compute_capability = "8.6"` of E006 is true on this machine. The two-card machine is
> `hardware/dual-rtx3090.toml` (PCIe only, no NVLink). The text below is kept as written; read
> "4090" in it as the reference card of the smoke configs, which name `hardware/rtx4090.toml` and
> should be switched to a 3090 profile (or a derived config) before a run is cited. M016 (`rtx5090`)
> and its measured speedup still need their own decision. Two independent runs, one per card, are
> possible (`DESKTOP_SETUP.md` section 10); nothing in `training/` trains on both.

**Facts.**

- `experiment.toml` of M010, M011, M012, M013, M014 and M015 each set
  `hardware_profile = "hardware/rtx3090.toml"`. E006 does too, and additionally pins
  `gpu_compute_capability = "8.6"` in `config.toml`.
- `hardware/rtx3090.toml`: RTX 3090, 24 GiB, precisions fp32, fp16, bf16, int8.
  `hardware/rtx4090.toml`: RTX 4090, 24 GiB, precisions fp32, fp16, bf16, fp8, int8 (notes:
  Ada, sm_89). Both say to record the actual CPU, RAM, driver and CUDA in each run manifest.
- The smoke configs of steps 1 and 2 already name `hardware/rtx4090.toml`.
- Other profiles in the plan: M016 names `rtx5090` (32 GiB) and its `config.toml` pins
  `hardware_profile` as a recorded procedure parameter with `max_hardware_profile_mismatches = 0`;
  R003 names `dual-rtx3090`; R004, R008 and E001 to E005 name `default.toml`.
- None of M010 to M015 pins the profile inside `config.toml`; the profile lives in the
  manifest, which freezes at the first commit past `planned`.

**OWNER DECISION OD-1.** Options:

- **A. Keep `rtx3090.toml` as the declared class** because the VRAM is equal (24 GiB).
  Argument for: no manifest edit. Against: the record would name a GPU that did not run it;
  the 4090 has a different architecture (sm_89 against sm_86), so kernels, bf16 behaviour and any
  timing differ (estimate), and M011 reports `latency_per_token`; "equal VRAM" justifies
  only that the run fits, not that it is the same class. It would also leave E006's
  `gpu_compute_capability = "8.6"` untrue.
- **B. Edit the manifests to `hardware/rtx4090.toml` before the freeze.** The manifests are
  still `planned` and unlisted, so the edit is allowed and is just a normal commit; after the
  freeze commit it would be impossible without a successor experiment. Also change E006's
  `gpu_compute_capability` to the value the 4090 reports.

**Recommendation (superseded, kept as history; resolved in favour of A, see the note at the head of this section): B**, for M010 to M015 and E006, decided before the first prepared record of
each. Reason: the profile is provenance, the owner's machine is a 4090, and `rtx4090.toml`
already exists and is what the smoke configs use. For M016 (`rtx5090`, 32 GiB, with a
measured speedup as the headline) the same logic applies: switch to `rtx4090` or do not run it
on this machine. For R003 (`dual-rtx3090`), the declared hardware is not this machine at all.
Whichever option is chosen, write down the actual GPU, driver and CUDA from `nvidia-smi` in the
run record (the profile files ask for it).

## 6. Hand results back

Neither route needs GitHub (suspended; HOSTING.md).

- **Git bundle (works offline).** From the desktop checkout, on a branch that holds the
  committed run records and any config you derived:
  ```bash
  python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py create --out ../ptr-export
  # copy ../ptr-export to the other machine, then there:
  python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py join ../ptr-export
  python3 .claude/plugins/ptr-dev/skills/ptr-offline-sync/scripts/ptr_bundle.py apply ../ptr-export/<bundle>
  ```
  `apply` writes only `import/*` branches and never touches the current branch; merge them
  explicitly. A bundle contains the full history, so keep it private.
- **GitLab push.** `python3 scripts/publish_remote.py --remote-url <url> --dry-run` first,
  then without `--dry-run` (default remote name `gitlab`, default branches `main,claude/*`).
  It pushes everything in one `git push`, never forces (refspecs have no `+`), refuses URLs
  with embedded credentials, and a second run is a no-op. Exit code 3 means the remote
  rejected a ref (for example non-fast-forward): fix it by merging, **never force-push**.
  Batch pushes and avoid loops of pipeline runs or comments (HOSTING.md, "Low-automation
  rules"). The manual `gpu-smoke` GitLab job needs a runner tagged `ptr-gpu`
  (HOSTING.md) and is pipeline evidence only.
- **What goes back.** Committed: the derived config, wrapper or entrypoint changes, the
  frozen experiment files, the run records and aggregates under each experiment's `results/`,
  and the write-up. Not committed: `training/runs/` (gitignored), logs, model weights, the
  Hugging Face cache, tokens.
- **What the write-up must contain** (the `ptr-results` skill): what ran (command, config
  path, git sha, `git_dirty`, seed, hardware, driver, versions, wall time), the numbers from
  `metrics.json`, the evidence class (smoke = "pipeline evidence only"), the baselines and
  seeds actually used, the pinned falsification criteria and whether each held, and negative or
  inconclusive outcomes stated the same way.

## 7. Instructing Claude Code on the desktop

Install the plugins once from the repository root (DESKTOP_SETUP step 8):

```bash
claude plugin marketplace add ./
claude plugin install ptr-dev@ptr
claude plugin install ptr-mods@ptr
```

Then start `claude` in the repository and use requests like these. The skills are invoked by
asking for the task, or as `/ptr-dev:<name>`.

- **`experiment-run`** (run and monitor): "Use the experiment-run skill to run
  `make smoke-gpu` in the background, watch the log and `nvidia-smi`, and report the peak
  memory and the metrics against the WP8 table, labelled pipeline evidence." Same for
  `make olmoe-sft-smoke`. The skill requires `make gpu-doctor` first, a clean tracked tree,
  logs in `training/runs/logs/`, and refuses to claim results from smoke runs.
- **`ptr-verify`** (checks derived from a diff): "Run ptr-verify on this branch" after any
  config, Python or manifest change; it picks the validators, `component.toml` and
  `make docs-check` where needed and reports each with real output.
- **`ptr-results`** (write-up): "Use ptr-results to write up the SFT-smoke run" or the
  completed experiment, with the evidence class, seeds, baselines and intervals and no
  over-claiming.
- **`ptr-new-experiment`** (successors): "Create a successor to M011 with ptr-new-experiment"
  when a prepared record would otherwise be patched.
- **`ptr-offline-sync`, `ptr-gitlab`** for section 6.

**Head sessions.** The end-to-end path from a powered-on machine to a running head session is in
[`LOCAL_START.md`](LOCAL_START.md); the head prompt, the staged goal (this document's run order) and the
channel, reviewer and fact-check brief templates are in [`agents/KICKOFF.md`](agents/KICKOFF.md). Use them
instead of composing the requests by hand when a head session drives the work.

Rules to state in the first message of each GPU session (they come from `CLAUDE.md`, the
`experiment-run` skill and sections 2 and 3):

1. Never edit a prepared or frozen experiment record, and never edit `experiments/` to make
   a run pass; a change is a successor experiment.
2. Never commit under `training/runs/`; never print or store `HF_TOKEN`.
3. Smoke runs are pipeline evidence; no "improves", "outperforms" or "specialises".
4. One run per seed, no reruns, no tuning on the held-out partition; negative results are kept.
5. Do not push, force-push or change hosting settings without being asked.
6. Do not install system packages, drivers or the GitLab runner without being asked.

## 8. Owner decisions

All of them, by experiment and key. "Placeholder" means a `must-be-pinned-...` or
`must-be-signed-...` string in the files; "gap" means the files leave it open without a
placeholder.

| ID | Applies to | Key or item | Kind | Options | Recommendation |
|---|---|---|---|---|---|
| OD-1 | M010, M011, M012, M013, M014, M015, E006 (also M016, R003) | `hardware_profile` in `experiment.toml`; E006 `gpu_compute_capability = "8.6"`; M016 `rtx5090` | gap | keep 3090 as declared class / switch to `rtx4090` before freezing (section 5) | resolved: the machine has two RTX 3090, keep `rtx3090.toml` (section 5, note at its head); the recommendation to switch to `rtx4090` no longer applies |
| OD-2 | M006, M010 to M016, R002, R003, R008, E001 to E004, E006 | listing in `experiments/preregistration.toml` | gap | list before the first prepared record / leave unlisted (no freeze protection, per `CLAUDE.md`) | list, with the required keys of each, so the freeze gate covers them |
| OD-3 | every GPU experiment | interpreter and environment of a frozen entrypoint (a listed run refuses git-ignored files such as `training/.venv`, and passes no `PYTHONPATH`) | gap | environment outside the checkout and a clean worktree / a wrapper in the repository / run unlisted | settle with the first entrypoint (M011) and reuse it **Decided 2026-10-13 (pinning part):** GPU pinning of frozen seed runs (OD-3, decided 2026-10-13, no runner change): a GPU experiment's manifest declares `PTR_CUDA_DEVICE` in `command_environment` before its freeze (the runner passes declared `PTR_*` variables through, `scripts/run_experiment.py`), its entrypoint wrapper sets `CUDA_VISIBLE_DEVICES=$PTR_CUDA_DEVICE` and prints the index into its result JSON (the run record names the declared variable, never its value), and the freeze script's `gpu` field exports both variables. An experiment frozen without that declaration runs on card 0, one at a time. |
| OD-4 | M015 | teacher `Qwen/Qwen3-30B-A3B` does not fit 24 GiB (bf16 about 60 GB, estimate) | gap | pin a quantized or offloaded teacher / sample the teacher on other hardware and pin its output digest / choose a smaller teacher (successor manifest) / defer M015 | defer M015; if wanted, pin the teacher's precision and sampling location in the manifest before the freeze |
| OD-5 | M011 (and the others reusing `m011.py` choices) | `UNPINNED_CHOICES` of `m011.py`: optimizer, sampler, loss, PERFT activation, maximum lengths, simplified BFCL scorer | gap | pin them in `[preregistration]` / leave them recorded only | pin the ones that can move the result (optimizer, learning-rate schedule, scorer) |
| OD-6 | E006, M016 | `software_stack.driver_version = "580.65.06"`, `cuda_runtime_version = "13.0"`, `torch_version = "2.13.0"` with `max_software_stack_mismatches = 0` | gap | install exactly the pinned stack / edit the pin to the installed one before freezing | check `nvidia-smi` first; edit the pin before freezing if it differs |
| OD-7 | M010 | meaning of the `rollout_engine = "separate"` arms (`grpo_no_replay_ms4_separate`, `grpo_replay_ms4_separate`) | gap | define the separate engine (for example vLLM) in `config.toml` / drop the two arms in a revised manifest before the freeze | define it before the freeze, since the manifest's arm set is pinned |
| OD-8 | R004 | placeholders `public_regression_gate_permille`, `backward_transfer_min_permille`; also backbone, domain sequence, public suite, compute budget | placeholder + gap | pin values and name an OLMoE or other backbone | name the backbone first (it fixes what a gate value means), then pin both |
| OD-9 | R003, E001, E002, E003, E004, E005 | pinned open backbone (ADR-0021, "separate later decision"); ADR-0015 coding backbone; `model.toml` fields; strong_rag revisions (`dense.revision`, `reranker.revision`, `answer.generator`, `answer.revision`) | gap | choose backbone and revisions / keep deferring | keep deferring until steps 4 to 13 have run |
| OD-10 | E005 | `token_budget`, `full_context_baseline`, `agent_memory_baselines`, `baseline_strong_rag_sha256` | placeholder | pin each | after OD-9 |
| OD-11 | R003 | declared `dual-rtx3090` profile (2 x 24 GiB) matches the two cards of this machine; the blocker is the missing backbone and multi-GPU trainer (original wording: vs a single 4090) | gap | change the profile in a successor / defer | defer with OD-9 |
| OD-12 | M016 | start only after R006 is completed (R006 itself is CPU and needs its own preregistration and 5 seeds) | gap | complete R006 first / change M016's design | complete R006 first |

Placeholders found in the GPU-phase manifests: only R004 (two keys) and E005 (four keys) hold
`must-be-pinned-...` strings. The manifests of M010 to M016 and R008 and E006 contain no
`must-be-pinned` or `must-be-signed` string; their blockers are missing code and the gaps
above. F003's four `must-be-signed-by-owner` keys (`calibration_set_rule`,
`adjudicators_see_intent`, `harm_breakdown_namespaces`, `adjudication_protocol`) are not GPU
work and are tracked elsewhere.

## 9. First-day checklist

1. Sections 1.1 items 1 to 7, then `make smoke-gpu` (step 1) and `make olmoe-sft-smoke`
   (step 2). Commit nothing from `training/runs/`.
2. Send back the step-2 `metrics.json` peak numbers and the two run manifests
   (`training/runs/run-<timestamp>.json`).
3. Answer OD-1 to OD-3; then step 3 and the M011 preparation can start.

## 10. VCW-MoE (Verifier-Closed Workspace MoE) GPU runs

Added 2026-10-09. The full plans, arms, margins, mutations and budgets are in
[`VCW_MOE_GPU_PLAN.md`](VCW_MOE_GPU_PLAN.md); this section is the operator's list. Everything in
it is **planned, not run**. GPU-hours quoted are **estimates** from the assumptions of that
document (section 3), not measurements, and the first pilot seed replaces them.

### 10.1 Prerequisites (all must hold; check each)

1. Steps 1 to 3 of section 4 done and OD-1 to OD-3 answered. The measured peak of the OLMoE SFT
   smoke is compared with the S0 estimate (16.4 GiB on the 4090, 19.3 GiB on the 5090); if the
   measurement is more than 20% higher, re-derive the VCW batch sizes before anything else.
2. The code that does not exist yet (VCW plan section 4.9): the stage entrypoints
   `ptr_training.vcw.*` (P1), the Python verifier fabric with finding codes and golden vectors
   against the Rust gate (P2), the loop binding or Rust harness (P3), the instruments and
   `aggregate.py` files (P5, P7). **Do not run a `vcw-*` config until its entrypoint exists:**
   `ptr_training.run --execute` refuses the `vcw-*` backends on purpose, and the synthetic-task
   `trl-moe` pipeline must never be run on a VCW config's behalf.
3. Hub downloads at the revisions of VCW plan section 2 (two new datasets beyond section 1.3:
   MBPP and Hendrycks MATH). Add to the snippet of section 1.3, with the training venv:
   ```bash
   training/.venv/bin/python - <<'PY'
   from huggingface_hub import snapshot_download
   print(snapshot_download("google-research-datasets/mbpp", repo_type="dataset",
                           revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c"))
   print(snapshot_download("EleutherAI/hendrycks_math", repo_type="dataset",
                           revision="21a5633873b6a120296cce3e2df9d5550074f4a3"))
   # C005 only:
   print(snapshot_download("allenai/OLMoE-1B-7B-0125-Instruct",
                           revision="b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"))
   PY
   ```
4. A clean tracked worktree, the environment outside the checkout for frozen runs (OD-3), and
   one hardware profile chosen per experiment (`hardware/rtx3090.toml`, or `hardware/dual-rtx3090.toml` for a two-card run; the original single-4090 text recommended `hardware/rtx4090.toml`).

### 10.2 Commands in order

CPU, any machine, before touching the GPU (these run today):

```bash
PYTHONPATH=training/src python3 -m unittest discover -s training/tests -p 'test_vcw*'
PYTHONPATH=training/src python3 -m ptr_training.vcw.config      # self-report audit: must print 0
PYTHONPATH=training/src python3 -m ptr_training.vcw.plan         # the GPU-hour ledger
```

On the GPU machine (each line starts only after the previous one is done and recorded):

| # | Run | Entry | Expected output | Estimate, 4090 mid |
|---|---|---|---|---|
| V1 | pilot seeds 1001 to 1003 of S0 | `vcw-perft-r-sft-rtx3090.toml` via the P1 entrypoint | `metrics.json` with `hardware.peak_cuda_memory_allocated_gib` (compare 16.4), loss curve, trainable parameters 4,456,448 | 6.4 h per seed |
| V2 | pilot seeds of S1, both arms | `vcw-rlvr-router-rtx3090.toml` | per-step reward, `expert_churn_per_layer`, `routing_entropy`, `load_cv`, `dead_experts`, `train_inference_kl`, `clip_fraction`, `crash_step`; peak memory (compare 21.1) | 0.84 h per seed |
| V3 | pilot seeds of S2 (per-class and generic) | `vcw-repair-expert-rtx3090.toml` | transitions per finding class (each trained class must occur often enough to pin), peak memory (compare 15.5) | 2.1 h per arm |
| V4 | pilot checks | harness | ceiling, floor, sensitivity, margin variance (VCW plan 4.5, 4.6), truncation rate at 160 completion tokens | minutes |
| V5 | freeze commits and `prepare` for F005, F004, C002, C003 | section 3.1 | `prepared` records at HEAD | none |
| V6 | T on final seeds 17, 29, 43, 71, 101 | the V1 to V3 entrypoints | `checkpoints.json` (adapter digests), committed | 11 h per seed |
| V7 | F005, then F004, then C002 and C003 | `python3 scripts/run_experiment.py run <ID> --seed <S>` one seed at a time | `results/<seed>/` records, then `aggregate.py` writes `results/run.json` and `metrics.json` | F005 1.9 h, F004 7.7 h, C002 1.1 h, C003 5.0 h per seed |
| V8 | F006 after M013, C006 after the C001 verdict, C005 last | same | same | F006 26 h, C006 8.8 h, C005 2.8 h per seed |

### 10.3 How results come back as experiment records

The route is section 3.1 and section 6, unchanged; what is specific to these experiments:

1. Each experiment is a successor-free normal commit sequence on its own branch: set `entrypoint`
   in `experiment.toml` while it is still `planned`, write `[preregistration]` (the pins and the
   margins of VCW plan section 5, signed by being committed), `criteria.toml`, `aggregate.py`
   with its mutation tests, `PREREGISTRATION.md`, list it in `experiments/preregistration.toml`,
   and set the two digests. The manifests F004 to F006 and C002 to C006 were being edited in a
   design branch when this was written; do this after that branch merges.
2. `run_experiment.py run` hands the entrypoint a restricted environment (section 1.1 item 8),
   so the entrypoint reads its config path from the repository and finds the model in the default
   Hugging Face cache. Each seed's record names the git sha, config digest, checkpoint digests
   (from `checkpoints.json`), hardware, driver, CUDA, versions, wall time, peak memory, the arm
   table with verifier calls and FLOPs per arm (VCW plan 4.2), and the hard counters.
3. `aggregate.py` reads `criteria.toml`, computes the seed-level one-sided bounds and writes one of
   supported, refuted, not shown or invalid with the reason, per the outcomes of VCW plan 4.4.
   It never reads a result to decide a margin.
4. Negative and invalid outcomes are committed like positive ones. A rerun is a successor
   experiment.
5. After every aggregate add a `welfare_check` line to the experiment's results README (families
   supported on OLMoE so far; level L0, L1 or L2 of MODEL_WELFARE section 4). At L1 inform the
   owner in the next report; at L2 stop and wait.
6. Send back as in section 6: committed records and derived configs by bundle or GitLab; nothing
   from `training/runs/`.

### 10.4 What can go wrong first (read before V1)

- **RL memory.** The RL phase has no gradient checkpointing and no accumulation; the 4090 S1
  configuration is sized to fit at about 21 GiB by estimate, which leaves 1.4 GiB. If it runs out
  of memory, lower `moe.rl.max_completion_length` is *not* allowed after the freeze (it changes the
  task); use the pilot to choose a different pre-freeze value and re-derive the estimate with
  `ptr_training.vcw.budget`.
- **Truncation.** At 160 completion tokens GSM8K reasoning is often cut off (`truncated` is a hard
  finding and scores 0). Read the rate on the pilot seeds before pinning the S0 corpus
  preprocessing.
- **Decode speed.** The plan assumes host-bound HF `generate` at 0.07 s per step. If the first
  pilot shows more than double the slow estimate, stop (abort rule A5).
- **M010 as pinned does not fit.** Do not start M010 before the backend has RL checkpointing and
  accumulation (VCW plan 3.3, P4).


## 11. Final state (2026-10-11): CPU today, GPU later

This is the short version for the operator; the details, estimates and margins are in
[`VCW_MOE_GPU_PLAN.md`](VCW_MOE_GPU_PLAN.md) section 10. Nothing below has a GPU result.

**Runs on CPU today.** C001 (`prepared`, verdict NOT_SHOWN). The Rust `VcwLoop`, router, workspace
and verifiers with simulated experts (`make vcw-demo`). `ptr_py` with the verifiers, the finding
router and its propensity log, the off-policy estimators, the bounded workspace and an in-process
loop simulation (`make py-bindings`); a Python RLVR trainer gets its propensities from
`FindingRouter.route` and its rewards from `settle(decision_id, candidate)`, which verifies the
candidate inside Rust; Python also chooses which decisions it settles, the routed report's task
and the candidate text (the log keeps digests of report and candidate; estimates are only as honest
as the harness), but cannot state or forge a reward, and cannot change a gate, threshold, verifier, eligibility or the required level. C007 and C008 are the CPU stage-A experiments of C002 and C003 as registered on their own branches; E008 is the split-pod
extension of E007. C007, C008 and E008 are not in the registry of this branch, so read their
status from the merged registry.

**Needs the GPU.** OLMoE-1B-7B L1 experts; PERFT-R (M011) and the S0 to S2 stages; RLVR router
training against verifier rewards; stage B of C002 and C003; C005 and C006; F004 to F006; E008
part C (split parity on `candle-cuda`, which no workflow compiles).

**Order.** Steps 1 to 3 of section 4, then M011, pilots on seeds 1001 to 1003, freeze, T on final
seeds, F005, F004, C002 and C003 stage B, M012, M013, F006, C006, C005, with E008 part C on any
CUDA machine independently and M014 and M015 last.

**Preregistration.** Among the GPU experiments only C001 has a preregistration entry; every other
one is `planned` with no entry, no freeze and no run. The head freezes.

**Hardware.** `rtx3090.toml` (24 GiB, the manifests' current profile and the right one for this machine, OD-1 resolved),
`dual-rtx3090.toml`, `rtx4090.toml` (24 GiB, the original single-card recommendation) and `rtx5090.toml` (32 GiB);
one profile per experiment (OD-1).

**Retraining the base model** changes the `BaseModelRef` digest and generation, so all experts
bound to the old base mismatch, the per-layer KV chain keys change from the first changed layer
on (old cached KV is never reused), and pinned checkpoint digests need successors.

**Framing.** Doc-38 indicators are functional properties; no run here claims or tests
consciousness, and `MODEL_WELFARE.md` applies.
