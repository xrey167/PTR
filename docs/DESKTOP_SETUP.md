# Desktop setup: Ubuntu 24.04 on two single-GPU PCs (one RTX 3090, one RTX 4090)

**Kurzfassung (Deutsch).** Dieses Runbook richtet einen frisch installierten Desktop (Ubuntu
24.04 LTS, native, kein WSL2; je PC eine Karte mit 24 GB, RTX 3090 oder RTX 4090) für
PTR-Experimente mit Claude Code ein. Reihenfolge: Ubuntu installieren (Secure Boot:
MOK-Schlüssel beim Neustart bestätigen; mindestens 200 GB frei), Treiber
`nvidia-driver-580-open` installieren und mit `nvidia-smi` prüfen, `git` und Repository klonen,
`make desktop-setup`, danach `make gpu-doctor`, Hugging-Face-Login und OLMoE-Download (ca. 14
GB, fester Revisions-Hash), dann `make smoke-gpu`, `make olmoe-sft-smoke` und zuletzt M011 (nur auf dem 3090-PC).
Claude Code startet man im Repo-Verzeichnis; der Skill `experiment-run` führt Läufe im
Hintergrund aus und überwacht sie. Die Smoke-Läufe sind **Pipeline-Belege, keine Evidenz**: ohne
Falsifikationskriterien, Seeds und Baselines des jeweiligen Experiments wird kein
Forschungsergebnis behauptet. Der Rest dieses Dokuments ist auf Englisch, wie die übrigen Docs.

Target: a fresh Ubuntu 24.04 LTS desktop (native, not WSL2) with one NVIDIA RTX 4090
(24 GiB, Ada, sm_89). The hardware profile is `hardware/rtx4090.toml`. Nothing here
claims a research result: the smoke runs are pipeline evidence only (see step 7).

**The owner has two PCs with one card each: an RTX 3090 (`hardware/rtx3090.toml`) and an RTX
4090 (`hardware/rtx4090.toml`).** Follow the setup and smoke steps (sections 1 to 6, 8 and 9, and items 1 and 2 of section 7) on
each PC with that PC's profile. M011 (section 7, item 3) declares `hardware/rtx3090.toml`
(`experiments/model/M011-routed-peft/experiment.toml`), so it runs **only on the 3090 PC**. The text
below says 4090 where it was written for the reference card. The Makefile selects the config
set with `GPU_PROFILE`, whose default is `rtx3090`: on the 3090 PC run the targets as written,
and on the 4090 PC pass `GPU_PROFILE=rtx4090` to every GPU target (for example
`make smoke-gpu GPU_PROFILE=rtx4090`), so the run uses the 4090 configs and records
`hardware/rtx4090.toml`. Section 10 also keeps the notes for the superseded two-3090 machine.

Everything below was written and unit-tested on a machine without a GPU. The commands that
need the GPU (driver install, `make smoke-gpu`, `make olmoe-sft-smoke`) have not been run
on the 4090 by the author of this document; the first real run is the first real test.

## 1. Ubuntu 24.04 install notes

- **Disk.** Keep at least **200 GB free** after the OS install. Rough budget: Rust
  `target/` several tens of GB, the locked CUDA torch wheels about 8 GB in `training/.venv`,
  OLMoE about 14 GB in the Hugging Face cache, plus checkpoints and run output. An NVMe SSD
  is strongly preferred. If `/home` is small, put the Hugging Face cache elsewhere (step 6).
- **Secure Boot.** The NVIDIA kernel module is built by DKMS and must be signed. With
  Secure Boot on, `apt` asks for a one-time password during the driver install. After the
  reboot a blue "MOK management" screen appears: choose *Enroll MOK*, *Continue*, enter that
  password, reboot. Without enrolment the module does not load and `nvidia-smi` fails with
  "couldn't communicate with the NVIDIA driver". The simplest alternative is to disable
  Secure Boot in the firmware. Check the state with `mokutil --sb-state`.
- Use the stock kernel (6.8 or the HWE kernel Ubuntu offers) and install all updates first:
  `sudo apt update && sudo apt full-upgrade`, then reboot.
- Base tools: `sudo apt install -y git curl build-essential pkg-config bubblewrap` (bubblewrap sandboxes the model-generated code M011 scores on HumanEval).
- Free disk: `make desktop-setup` stops in step 1, before any download or build, when the repository's disk has less than 20 GB free (`PTR_SETUP_MIN_FREE_GB` changes the floor).

## 2. NVIDIA driver 580

Driver **580, open kernel module**. No system CUDA toolkit is needed for PyTorch: the CUDA 13
wheels locked in `training/uv.lock` bring their own runtime. (`nvcc` is only needed for the
optional Candle CUDA path, which nothing here uses.)

```bash
sudo apt update
sudo apt install -y nvidia-driver-580-open
sudo reboot          # enrol the MOK key here if Secure Boot is on
```

Verify after the reboot:

```bash
nvidia-smi           # shows driver 580.x and "NVIDIA GeForce RTX 4090", 24564 MiB
```

`scripts/setup_desktop.sh` never installs the driver: it prints these commands and stops if
`nvidia-smi` is missing.

## 3. git, the clone, and the GitHub login

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
sudo apt install -y gh        # GitHub CLI, optional but easiest
gh auth login                 # GitHub.com, HTTPS or SSH, login with the browser
gh repo clone <owner>/PTR ~/PTR
cd ~/PTR
```

Claude Code itself logs in separately with your Claude account (step 8); it needs no GitHub
token to run experiments. `gh auth login` is only for pushing branches and opening PRs.

## 4. `make desktop-setup`

```bash
make desktop-setup                    # asks before installing rustup and uv
scripts/setup_desktop.sh --dry-run    # print the steps without changing anything
scripts/setup_desktop.sh --yes        # no confirmation prompts
```

It is idempotent and never runs `sudo`. The steps:

1. check the OS is Ubuntu and `git`, `curl`, `cc`, `pkg-config` exist;
2. check `nvidia-smi` (stops with the driver commands if it is missing);
3. install rustup and the toolchain pinned in `rust-toolchain.toml` (1.99.0);
4. install `uv` with the official installer;
5. `uv sync --project training --python 3.11 --extra moe --locked`, which creates
   `training/.venv` with the CUDA torch wheel from `uv.lock` (several GB, needs network);
6. install maturin into that venv and run `maturin develop --uv --locked -m bindings/ptr-py/Cargo.toml`,
   so `ptr_py` lands in **the training venv**. `make py-bindings` builds a different venv
   (`bindings/ptr-py/.venv`) that the training pipeline does not use;
7. run `scripts/gpu_doctor.py --require-gpu`.

Open a new shell afterwards (or `source ~/.cargo/env`) so `cargo` and `uv` are on `PATH`.

## 5. `make gpu-doctor`

```bash
make gpu-doctor                                   # table, exit 1 on any FAIL
python3 scripts/gpu_doctor.py --require-gpu       # a missing GPU is a FAIL, not a WARN
python3 scripts/gpu_doctor.py --offline --json /tmp/doctor.json
```

It checks the driver and GPU (parsed from `nvidia-smi`), Python >= 3.11, `uv`, the training
venv, and in that venv: torch, `torch.cuda.is_available()`, the CUDA build, bf16 and a small
CUDA matmul (run as a subprocess of the venv python). It compares `transformers`, `peft`
and `trl` with `training/uv.lock`, checks that `ptr_py` imports, warns below 60 GB free at
the repo and at the Hugging Face cache, checks Hugging Face Hub reachability (skip with
`--offline`) and the Rust toolchain. Every non-OK line has a `fix:` hint. Do not continue
until `--require-gpu` exits 0.

## 6. Hugging Face login, cache, and the first model download

```bash
# optional: move the cache to a big disk (add to ~/.bashrc)
export HF_HOME=/data/hf
training/.venv/bin/hf auth login      # token from https://huggingface.co/settings/tokens (read)
```

OLMoE is not gated, so the login is optional, but it avoids anonymous rate limits. Never put
a token in the repository or in `.claude/settings.json`.

Download the model once, at the **pinned revision** (about 14 GB), so the runs do not
download during a timed step:

```bash
training/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("allenai/OLMoE-1B-7B-0125",
                        revision="9b0c1aa87e34a20052389dce1f0cf01da783f654"))
PY
```

The revision is the one `training/configs/stage-moe-olmoe.toml` pins; never use `main`.

## 7. Step-by-step order and where results land

Runs refuse a dirty tracked worktree (`--execute`), so commit or stash tracked changes
first. Always `make gpu-doctor` before a run.

1. **`make smoke-gpu`**: the WP8 tiny-random OLMoE smoke on the GPU
   (`stage-moe-smoke-gpu-rtx3090.toml` by default, `stage-moe-smoke-gpu.toml` with
   `GPU_PROFILE=rtx4090` on the 4090 PC). Minutes. It checks the
   pipeline (SFT, GSPO and SAPO, routing metrics, `ptr_py` gates) runs on CUDA.
2. **`make olmoe-sft-smoke`**: real OLMoE-1B-7B, SFT only, LoRA, bf16, gradient
   checkpointing, one 24 GB GPU
   (`stage-moe-olmoe-sft-smoke-rtx3090.toml` by default, without the suffix under
   `GPU_PROFILE=rtx4090`). Override either
   variable on the command line: `make smoke-gpu SMOKE_GPU_CONFIG=path.toml`.
3. **M011** (3090 PC only: its manifest declares `hardware/rtx3090.toml`, and `GPU_PROFILE`
   cannot change that): experiment M011 is registered as *planned*. Its entrypoint is set only when
   its run is prepared, so do not look for a `make` target: ask Claude Code (step 8) to prepare
   the run per the experiment's manifest. Before it, `python3 scripts/run_experiment.py validate`
   and `python3 scripts/check_research_gates.py` must pass. Never edit a prepared experiment
   record; a change means a successor experiment (see `CLAUDE.md`).

Where results land (all gitignored, never commit them):

| Artifact | Path |
| --- | --- |
| Run manifest (git sha, dirty flag, config/lock hashes, platform) | `training/runs/run-<timestamp>.json` |
| Metrics | `training/runs/<moe.output.dir>/<run>/metrics.json` (for example `training/runs/moe-smoke/...`) |
| Experiment results | `experiments/<area>/<ID>/results/` as the experiment's manifest says |

**Interpretation.** Compare a smoke run with the WP8 CPU baseline table in
`training/README.md`. Smoke runs are **pipeline evidence, not research evidence**: one seed,
no baseline, no ablation, backend still unselected. A research claim needs the
experiment's falsification criteria, seeds and baselines, and negative results are kept.

## 8. Claude Code in the repo

```bash
curl -fsSL https://claude.ai/install.sh | bash     # official installer, then a new shell
cd ~/PTR
claude                                             # first start: /login with your Claude account
```

The repo carries project config: `.claude/settings.json` pre-approves only commands that run no
repository code (`nvidia-smi`, `git status/diff/log/show`, read-only `glab`). On this desktop, with
your own checkout, copy `.claude/settings.local.example.json` to `.claude/settings.local.json`
(git-ignored) to also pre-approve `make gpu-doctor`, `make smoke-gpu`, `make olmoe-sft-smoke`,
`cargo check` and test and the experiment validators; everything else still asks.
The repo also ships Claude Code plugins (details in `docs/CLAUDE_TOOLING.md`). Install them
once from the repository root:

```bash
claude plugin marketplace add ./
claude plugin install ptr-dev@ptr
claude plugin install ptr-mods@ptr
```

`ptr-dev` brings the `experiment-run` skill (how to run and monitor experiments), plus
`ptr-verify`, `ptr-results`, `ptr-offline-sync` and others, six agents (five read-only, one coding channel) and
guard hooks. `ptr-mods` adds `/runs`, a pane with the newest run log, its PID and the GPU line.
Example requests:

- "Run `make gpu-doctor`, then `make smoke-gpu` in the background with the experiment-run
  skill, watch the log and `nvidia-smi`, and summarize the metrics against the WP8 table."
- "Use the experiment-run skill to run `make olmoe-sft-smoke`; report peak VRAM and the
  loss curve, labelled as pipeline evidence."
- "Prepare the M011 run as its manifest describes; do not edit prepared experiment records."

The whole path from a powered-on machine to a running head session (repository from git bundles
or GitLab, toolchains, baseline checks with timings, plugins, worktree layout) is in
[`docs/LOCAL_START.md`](LOCAL_START.md); the copy-paste prompts for the head session, a coding
channel and the reviewers are in [`docs/agents/KICKOFF.md`](agents/KICKOFF.md). Use those two pages
instead of the ad-hoc requests above when a head session drives the work.

The ordered GPU run plan after the smoke runs, the per-experiment gaps and the open owner decisions are in
[`docs/GPU_HANDOVER.md`](GPU_HANDOVER.md).

GitHub is currently suspended: for the GitLab remote, the `ptr-gitlab` skill and registering this
desktop as a GitLab runner tagged `ptr-gpu`, see `docs/HOSTING.md`.

## 9. Troubleshooting

- **CUDA out of memory on 24 GB.** Close other GPU users (`nvidia-smi` lists them; the
  desktop session itself takes about 1 GB). Then lower `batch` or `max_length` in the
  run config's SFT section (see `training/configs/stage-moe-olmoe.toml` for the keys); keep
  gradient checkpointing and bf16 on. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
  fragmentation. Do not "fix" it by dropping the pinned LoRA targets silently: record any
  config change in the run notes.
- **Driver/torch mismatch** (`torch.cuda.is_available()` is False, "driver too old", or
  `no kernel image`). The locked wheels are CUDA 13 and need driver >= 580:
  `nvidia-smi` must show 580 or newer. If you installed another driver series, remove it and
  install `nvidia-driver-580-open`, reboot. If `nvidia-smi` works but torch reports a CPU
  build (`2.14.1+cpu`), the venv was built from a CPU wheel: rerun `make desktop-setup`
  (or `uv sync --project training --extra moe --locked --reinstall-package torch`).
- **`nvidia-smi` fails after the install.** Almost always Secure Boot without MOK
  enrolment: `mokutil --sb-state`, then redo the enrolment (`sudo mokutil --import
  /var/lib/shim-signed/mok/MOK.der`, reboot) or disable Secure Boot.
- **Hugging Face 401 / "Repository not found".** A stale or wrong token is sent even for a
  public model. Run `training/.venv/bin/hf auth logout` or `unset HF_TOKEN`, then log in
  again with a valid read token; check the repo id and that the revision is the pinned one.
- **Disk full.** `du -sh $HF_HOME training/runs target` shows the big users. Delete old runs
  under `training/runs/` (not tracked), `cargo clean` for `target/`, and
  remove unused model revisions from `$HF_HOME/hub`. The doctor warns below 60 GB.
- **`--execute` refuses to run.** The tracked worktree is dirty: commit or stash.

## 10. Two RTX 3090 instead of one RTX 4090

> **Superseded inventory (2026-10-13).** The owner now has two PCs with one card each, an RTX
> 3090 (`hardware/rtx3090.toml`) and an RTX 4090 (`hardware/rtx4090.toml`). Steps 1 to 9 apply
> to each PC as written, with that PC's profile. This section stays for a two-3090 machine and
> for R003, which declares `hardware/dual-rtx3090.toml`; no owner machine matches it today.

The two-3090 desktop had two RTX 3090 (Ampere, sm_86, 24 GiB each). Profiles:
`hardware/dual-rtx3090.toml` ("2x NVIDIA GeForce RTX 3090, 24 GiB each, 48 GiB combined, PCIe-only
(no NVLink bridge)") for the machine and `hardware/rtx3090.toml` (one card, 24 GiB; fp32, fp16,
bf16, int8, no fp8) for single-card runs. NVLink is not assumed. Steps 1 to 9 apply unchanged
except for the following.

- **Driver.** Same: `nvidia-driver-580-open`. `nvidia-smi -L` must list two GPUs (index 0 and 1).
  If only one shows, check the second card's power connectors, the slot and `lspci | grep -i nvidia`
  before touching software.
- **Memory.** Each run sees 24 GiB, the same as the 4090, so single-card fit estimates carry over
  (they were computed for 24 GiB). Timings do not: the 3090 has a different architecture and no
  fp8, so wall-time figures quoted for the 4090 do not transfer.
- **Which hardware profile a run declares.** The M010 to M015 and E006 manifests already name
  `hardware/rtx3090.toml`; that is correct for this machine, so no manifest edit is needed
  (`GPU_HANDOVER.md` section 5 is resolved in favour of the 3090 profiles). The smoke configs
  `stage-moe-smoke-gpu.toml` and `stage-moe-olmoe-sft-smoke.toml` and the VCW `*-rtx4090.toml` configs
  name `hardware/rtx4090.toml`; each has a `-rtx3090` variant naming `hardware/rtx3090.toml`
  (`make smoke-gpu` and `make olmoe-sft-smoke` pick them by default, `GPU_PROFILE=rtx3090`).
  Use the 3090 variant for every run that is meant to be cited, and always record
  the real `nvidia-smi` output.
- **One run per card.** Nothing in `training/` uses more than one GPU: the configs take
  `device = "cuda"`, the pipeline records `torch.cuda.get_device_name(0)` and peak memory of the
  default device, and no `DataParallel`, `DistributedDataParallel`, `device_map`, FSDP or ZeRO
  path exists under `training/src` (checked by search). So there is no multi-GPU training here; the
  48 GiB are not one pool. The supported use of the second card is a second, independent run:

  ```bash
  CUDA_VISIBLE_DEVICES=0 make smoke-gpu                       # run A on card 0
  CUDA_VISIBLE_DEVICES=1 make olmoe-sft-smoke                 # run B on card 1, a different run
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  ```

  Inside a run the pinned card is always `cuda:0`, so `metrics.json` shows the card name but not
  the physical index; write the index into the run log and your notes. Never start the same
  confirmatory seed twice, and never two runs that both default to card 0.
- **Frozen runs.** `scripts/run_experiment.py` passes a restricted environment and drops
  `CUDA_VISIBLE_DEVICES` (`GPU_HANDOVER.md` section 1.1 item 8), so a listed experiment's command
  cannot be pinned to card 1 through the environment; it uses card 0. The same holds for the generated
  freeze script: its `CUDA_VISIBLE_DEVICES` export reaches its own cargo, pilot and mutation steps but not
  the seed runs. GPU pinning of frozen seed runs (OD-3, decided 2026-10-13, no runner change): a GPU experiment's manifest declares `PTR_CUDA_DEVICE` in `command_environment` before its freeze (the runner passes declared `PTR_*` variables through, `scripts/run_experiment.py`), its entrypoint wrapper sets `CUDA_VISIBLE_DEVICES=$PTR_CUDA_DEVICE` and prints the index into its result JSON (the run record names the declared variable, never its value), and the freeze script's `gpu` field exports both variables. An experiment frozen without that declaration runs on card 0, one at a time. Exploratory and smoke runs are
  not affected.
- **Doctor.** `scripts/gpu_doctor.py` checks every visible card: a `gpu N` line per card, the summary
  "2x NVIDIA GeForce RTX 3090, N MiB each", a torch allocation and matmul probe on every device, and a
  `hardware profile` line that is OK and names `hardware/dual-rtx3090.toml`. A card that is not a 3090
  or 4090 is an untested profile and gets a WARN.
- **R003** declares `hardware/dual-rtx3090.toml`, the only profile in the plan that matches both
  cards, but it also needs a pinned backbone and a multi-GPU trainer that does not exist yet
  (`GPU_HANDOVER.md` step 14).
