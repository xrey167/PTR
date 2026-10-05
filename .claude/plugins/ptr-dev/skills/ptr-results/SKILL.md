---
name: ptr-results
description: Write up experiment, smoke-run or benchmark results honestly for PTR, with evidence classes (pipeline evidence, mechanism check, research result), required provenance (git sha, dirty flag, seeds, hardware, versions, baselines, confidence intervals), the statistics helpers in ptr-analytics, and comparison with research/novelty/CLAIMS.md. Use when asked to "write up results", "summarize the run", "update STATUS", "can we claim X", or "report the numbers".
---

# Write results honestly

## Hard rules

- Never edit prepared experiment records or anything under `experiments/` to fit a result. Results go in the chat, a file the user names, or the experiment's own `results/` output through the runner.
- Smoke runs are **pipeline evidence, not research results**. Forbidden wording from them: "improves", "outperforms", "specialises", "better than", "shows that", "validates the architecture".
- Negative and inconclusive results are first-class; report them with the same prominence and detail.
- Never commit `training/runs/`. Never print secrets such as `HF_TOKEN`.

## Evidence classes (state exactly one)

1. **Pipeline evidence**: the code path executes end to end and numbers are finite (CPU smoke, tiny-random model, one seed). Claims nothing about the hypothesis.
2. **Mechanism check**: a component behaves as designed on a controlled case (unit or property test, invariant holds, revocation is exact). Says the mechanism works, not that it helps.
3. **Research result**: only against an experiment's own manifest: its pinned seeds, baselines, metrics and falsification criterion, run from a clean tree after preparation. Everything else is class 1 or 2.

## Required content

- What was run: command, config path, experiment ID, entrypoint.
- `git rev-parse HEAD` and `git_dirty` (from the run manifest `training/runs/run-<timestamp>.json`, which also holds config/lock hashes, hardware and `input_fingerprint_sha256`).
- Seeds (all of them, not the best), hardware profile and actual device, versions (rustc/torch/transformers/trl/peft as relevant), wall time.
- Numbers from `metrics.json`, unrounded where it matters, with uncertainty: confidence intervals for rates, not bare point values.
- Baselines actually run at equal budget, with their numbers. "No baseline" must be stated.
- What it does not show: one seed, no ablation, unselected backend, untuned, tiny model.
- For a real experiment: verdict against the falsification criterion exactly as written.

## Statistics helpers

`crates/ptr-analytics` (exports in `src/lib.rs`): `wilson_interval(successes, trials, z)` returning `RateEstimate`; `clopper_pearson_upper(successes, trials, delta)` for one-sided exact upper bounds (use when failures are zero or rare); `binomial_cdf`; `brier_score`, `expected_calibration_error`; `krippendorff_alpha_nominal`; `weighted_rate`; `RunningMoments`. For Python-side numbers, state the formula used if no helper exists; do not invent a CI.

```bash
cargo test -p ptr-analytics --locked
```

## Compare with claims

Read `research/novelty/CLAIMS.md` ("Candidate Novelty Claims — NOT YET PROVEN") and `docs/DEFINITION_OF_DONE.md`. A single benchmark win is never sufficient. Say which numbered claim a result touches and that it stays provisional until benchmarked against prior art (`research/novelty/PRIOR_ART_MATRIX.md`). Update `STATUS.md` notes only with evidence-class labels; `docs/components/STATUS.md` is generated, never hand-edited.

## Template

```
Result: <experiment/run id>   Class: pipeline evidence | mechanism check | research result
Commit: <sha>  dirty: <bool>  Seeds: <list>  Hardware: <...>  Versions: <...>
Numbers: <metric = value [CI] per arm/baseline>
Verdict vs falsification criterion: <supported | rejected | inconclusive | n/a>
Does not show: <...>
Artifacts: <paths, uncommitted>
```
