---
name: experiment-run
description: Run and monitor a PTR GPU experiment or smoke run on the owner's local GPU PCs, one RTX 3090 and one RTX 4090 (hardware/rtx3090.toml, hardware/rtx4090.toml) (make smoke-gpu, make olmoe-sft-smoke, M011 and later), then report the results with honest evidence labels. Use when asked to start, watch, compare or write up a training or experiment run.
---

# Run and monitor an experiment

Use this for any run that trains or evaluates on the GPU. Setup is in `docs/DESKTOP_SETUP.md`;
repository rules are in `CLAUDE.md`.

## Hard rules

- **Never edit prepared experiment records.** Experiments listed in
  `experiments/preregistration.toml` are frozen by `scripts/check_research_gates.py`; a change
  needs a successor experiment, not a patch. Do not edit `experiments/` to make a run pass.
- **Never commit anything under `training/runs/`** (gitignored). Do not `git add -A` there.
- Smoke runs are **pipeline evidence, not research results**. Never write "improves",
  "outperforms", "specialises" or similar from them.
- `--execute` refuses a dirty tracked worktree. Do not work around it by committing for the
  user; tell them to commit or stash.
- No secrets in commands, logs, or files. Never print `HF_TOKEN`.

## Steps

1. **Doctor first.** `make gpu-doctor` (or `python3 scripts/gpu_doctor.py --require-gpu`).
   Stop on any FAIL and show the fix hint. A GPU WARN means the run would silently use CPU.
2. **Check the tree.** `git status --short` must show no tracked changes. Record
   `git rev-parse HEAD`.
3. **Start in the background with a log file**, never in the foreground:
   ```bash
   mkdir -p training/runs/logs
   LOG=training/runs/logs/<name>-$(date +%Y%m%dT%H%M%S).log
   nohup make <target> > "$LOG" 2>&1 &
   echo $! > "${LOG%.log}.pid"
   ```
   `<target>` is `smoke-gpu` or `olmoe-sft-smoke` (or the experiment's own entrypoint once its
   run is prepared). On the RTX 4090 PC append `GPU_PROFILE=rtx4090` (the Makefile default is
   `rtx3090`), so the run uses the 4090 configs and records `hardware/rtx4090.toml`. Use the Bash tool's `run_in_background` when available and say where the
   log is.
4. **Monitor.** Poll, do not busy-wait: `tail -n 30 "$LOG"` and
   `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv`
   every 30 to 60 seconds. Check the PID is alive (`kill -0 $(cat <pid file>)`). Watch for
   `CUDA out of memory` (see Troubleshooting in `docs/DESKTOP_SETUP.md`), NaN losses, or a
   stalled log. Report peak `memory.used`.
5. **Find the artifacts.** Run manifest: `training/runs/run-<timestamp>.json` (git sha,
   `git_dirty`, config/lock hashes, hardware). Metrics:
   `training/runs/<moe.output.dir>/<run>/metrics.json`; list with
   `ls -t training/runs/*/*/metrics.json | head`. Quote the manifest's `input_fingerprint_sha256`
   and `git_dirty` in the write-up.
6. **Compare with the WP8 baseline** table in `training/README.md` ("First training (WP8):
   CPU smoke run"): pass rate, routing entropy, load CV, dead experts, SFT loss. Say plainly
   whether the numbers are in the same range, and that the WP8 run was CPU, tiny-random,
   one seed.
7. **Write the results section** (in the chat, or a file the user names; never in a prepared
   experiment record). Required content:
   - what was run (target, config path, git sha, `git_dirty`, seed, hardware, versions, wall time);
   - the numbers, from `metrics.json`, unrounded where it matters;
   - an explicit label: **"Pipeline evidence only"** for smoke runs;
   - what it does not show: single seed, no baseline, no ablation, backend `unselected`;
   - for a real experiment (for example M011): only claim a result against that experiment's
     falsification criteria, seeds and baselines as written in its manifest, and report
     negative or inconclusive outcomes the same way.
8. Leave `training/runs/` uncommitted and tell the user where the files are.

## M011 (routed PEFT on OLMoE)

**Requires the `claude/m011-routed-peft` branch merged**: `training/src/ptr_training/moe/m011.py`
and its tests exist only there today, and the experiment manifest
`experiments/model/M011-routed-peft/` lives on `claude/experiments-kv-moe`. Without them
these commands do not exist. M011 is a preregistered-style experiment: follow its manifest for
seeds, baselines and falsification; the same hard rules above apply (do not edit prepared
records, no commits under `training/runs/`).

Entrypoint `python -m ptr_training.moe.m011`; arms (`--arm`, required): `perft_r_emergent`,
`perft_r_prescribed`, `plain_lora`, `mixlora`. Pinned seeds: 17 29 43 71 101; pinned 2000 steps.

```bash
# tiny smoke: random 2-layer OLMoE, in-memory fixture corpora, CPU, nothing downloaded
# (pipeline evidence only)
PYTHONPATH=training/src training/.venv/bin/python -m ptr_training.moe.m011 \
  --arm perft_r_prescribed --seed 1 --out /tmp/m011-smoke --model tiny --max-steps 2 --eval-limit 3
# tests (about 35 s on CPU)
PYTHONPATH=training/src training/.venv/bin/python -m unittest discover -s training/tests -p 'test_m011*'

# real run on one RTX 3090 (pin it with CUDA_VISIBLE_DEVICES) (downloads OLMoE and the three corpora at the pinned revisions),
# per arm: 1. tune once on the pilot seed 1001 (8 trials, 500 steps each), frozen tuning.json
python -m ptr_training.moe.m011 --arm mixlora --out runs/m011/mixlora/tune --model olmoe --tune
# 2. every pinned seed at the full 2000 steps with the frozen hyperparameters
python -m ptr_training.moe.m011 --arm mixlora --out runs/m011/mixlora --model olmoe --all-seeds \
  --hparams runs/m011/mixlora/tune/tuning.json     # writes runs/m011/mixlora/seed<N>/
# or both in one command:
python -m ptr_training.moe.m011 --arm mixlora --out runs/m011/mixlora --model olmoe --tune --all-seeds
# a single seed with frozen hyperparameters:
python -m ptr_training.moe.m011 --arm mixlora --seed 29 --out runs/m011/mixlora/seed29 \
  --model olmoe --hparams runs/m011/mixlora/tune/tuning.json
```

Other flags: `--max-steps`, `--device {auto,cpu,cuda}`, `--tune-budget`, `--tune-steps`, `--lr`,
`--batch-size`, `--micro-batch-size`, `--eval-limit`, `--max-new-tokens`,
`--corpora {fixture,hf}`, `--gradient-checkpointing / --no-gradient-checkpointing`.
`--all-seeds` rejects `--seed` and needs `--tune` or `--hparams`. Run it with the background
and monitoring steps above (one arm takes many hours); use the training venv's python and run
from the repo root, paths under `runs/` are relative to the working directory so prefer
`training/runs/m011/...` to stay inside the gitignored tree.

Outputs per run: `run.json`, `metrics.json`, and `tuning.json` for tuning. Any departure from
the pinned seed, steps or 8 trials is recorded in `deviations`/`tuning_deviations`; quote them.
Smoke (`--model tiny`) is pipeline evidence; only the full 4-arm, 5-seed run against the
manifest's falsification criterion can be reported as a research result (see `ptr-results`).
