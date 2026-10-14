---
name: ptr-new-experiment
description: Create and register a new PTR experiment with scripts/new_experiment.py (hypothesis, metrics, baseline, falsification, seeds), understand preregistration freezing in experiments/preregistration.toml, and validate it. Use when asked to "add an experiment", "preregister", "plan M0xx", "create a successor experiment", or when a prepared experiment would need changing.
---

# Create a new experiment

Every architectural claim must be falsifiable: baseline, ablation, failure criteria. Negative results are first-class.

## Hard rules

- **Never patch a prepared experiment record.** `scripts/check_research_gates.py` freezes records of experiments listed in `experiments/preregistration.toml` once their status is `prepared`, `running`, `completed` or `failed`. A change means a new successor experiment (for example `M002-v9-...` after `M002-v8-...`); the old one stays as history, status `superseded`.
- Unlisted prepared records are not covered by that check, but the same rule applies by convention.
- Do not edit `.cargo/config.toml` to add an `[env]` table.

## Create

Arguments (from `python3 scripts/new_experiment.py --help`): `path` is `<area>/<ID>-<name>`, ID is one capital letter plus three digits, name is lowercase words joined by `-`; area is an existing directory under `experiments/` (`feedback`, `lifecycle`, `model`, `retrieval`, `runtime`, `semdb`, `system`). Required: `--hypothesis`, `--metrics`, `--baseline`, `--falsification` (none may be empty). Optional: `--hardware-profile` (default `hardware/default.toml`, must exist), `--seeds` (comma-separated, default `17,29,43,71,101`).

```bash
python3 scripts/new_experiment.py model/M0NN-short-name \
  --hypothesis "Does X improve Y over Z at equal budget?" \
  --metrics "metric one; metric two; tokens; latency" \
  --baseline "named baseline(s) with equal compute or token budget" \
  --falsification "The concrete result that rejects the hypothesis." \
  --seeds 17,29,43,71,101
```

It refuses before writing anything, writes `experiment.toml` (status `planned`), `config.toml`, README and `tests/`, and appends the entry to `experiments/registry.toml`. Check `experiments/registry.toml` for the next free ID (the id must not already be registered).

Write a falsification criterion that names a measurable outcome, the baseline comparison and the seeds. A hypothesis that cannot fail is rejected in review.

## Preregistration

Experiments named in `experiments/preregistration.toml` must hold a `[preregistration]` table in their `config.toml` before leaving `planned`. Every key listed under `[experiment.<ID>.required]` there must be a pinned value of the declared type. Values still to be chosen are `must-be-pinned-...` strings and owner decisions are `must-be-signed-...` strings; either blocks `prepare`/`run`. Baselines are pinned by directory digest (`baseline_<name>_sha256`), files by `<key>_sha256`; digests come from `scripts/experiment_records.py`. Read the header comment of `experiments/preregistration.toml` and an existing entry (for example `[experiment.S003.required]`) before adding an entry for a new experiment. Once an experiment is listed it stays listed.

## Validate

```bash
python3 scripts/run_experiment.py validate
python3 scripts/check_research_gates.py
python3 -m unittest discover -s scripts/tests
make experiments-check      # same as the validate step
make docs                   # regenerate component docs if experiment status shows in them
```

Other runner subcommands: `list`, `show <id>`, `prepare <id>`, `run <id>`, `aggregate`. Preparing a listed experiment is refused until all preregistration pins hold.

## Examples

- Planned manifest whose preregistration is still to be pinned: `experiments/model/M011-routed-peft/` (`experiment.toml`, `config.toml`; empty `entrypoint`). A frozen and run one for comparison: `experiments/model/M008-fast-weight-memory/` (registry `prepared`, outcome in `results/OUTCOME.md`).
- M011 (routed PEFT on OLMoE): the manifest is `experiments/model/M011-routed-peft/` and its training entrypoint module is `training/src/ptr_training/moe/m011.py` (not yet named in the manifest).
- Status counts: `python3 -c "import tomllib,collections;print(collections.Counter(e['status'] for e in tomllib.load(open('experiments/registry.toml','rb'))['experiment']))"`.

Write up outcomes with the `ptr-results` skill; run GPU jobs with `experiment-run`.
