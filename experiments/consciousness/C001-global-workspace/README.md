# C001-global-workspace

A functional-mechanism test on a CPU toy task, the first experiment of the consciousness-indicator track
([doc 38](../../../docs/architecture/38-consciousness-indicators.md)). **Status: planned, redesigned (v2) and built to
freeze-ready; no declared seed has been run.** Indicator properties are not consciousness: this experiment tests whether a
mechanism (GWT-1..4 as a computation) helps, and says nothing about whether any system is conscious.

Version 1 (position-specific modules, online data, held-out position pairs, five seeds, lower-bound-only decision) was
withdrawn after review: its outcome was fixed by a ceiling. See `DRAFT_PREREGISTRATION.md`.

## What is tested, and what is not

A **learned-attention** workspace (a few slots, competitive top-k write, gated slot update, broadcast read) as an
inductive bias: with finite training data and more modules at test time than at training, does it generalise better than
pairwise attention at matched parameters and at matched bandwidth? It is **not** a test of the deterministic
`ptr-pods::workspace` contract (package A: bounded slots, verifier gate, bid competition, bounded-mailbox broadcast), which
learns nothing. A supportive result does not validate that contract and a refuting one does not touch it; the contract is
tested by its own tests and, on Pods, by C006. C001 answers one question for the design: is a *learned* write/read worth
building on top of the contract (GW-L1/GW-L2)?

## Task

Eight sensory modules **sharing one encoder and one update network** (no position information, so the module count is free at test
time) see one noisy observation each. One is the cue and carries a query type q; exactly two others have type q; the answer is the sum
of their two values (5 classes). Training: a finite set of 4096 eight-module instances. Evaluation: sixteen modules (primary), twenty-four,
and eight (more distractors than ever seen in training).

## Arms (parameter counts within 2%)

workspace; all_to_all; bw_all_to_all (value width reduced to the workspace's slot content); no_broadcast; competition_off (soft
write); capacity_off (8 slots); no_comm (floor); crippled_all_to_all (one round: the sensitivity control).

## Decision (three outcomes plus invalid)

| Outcome | Condition | Consequence for the Pod-MoE workspace design |
|---|---|---|
| COMPLETED (supported) | guards hold; every lower one-sided 95% bound of (workspace minus all_to_all, bw_all_to_all, no_broadcast) on 16 modules >= 50 per mille | build the learned L2 prototype and run C006 (a new test on Pods, not a consequence) |
| FALSIFIED (refuted) | guards hold, sensitivity control resolves, some upper bound < 50 per mille | negative result for this setting; downgrade GW-L1/GW-L2 (learned workspace) to exploratory |
| NOT_SHOWN | bounds straddle the margin | nothing changes, no downgrade; a larger successor may follow |
| INVALID | strongest comparator saturated (> 900) or floored (< 300) on any seed, sensitivity control unresolved, workspace untrained on > 2 of 10 seeds, evidence missing | nothing is concluded; repair and register a successor |

A structural defect on any seed (a hard counter or coverage rule of the harness) is FALSIFIED, kind structural. Exit status of
`aggregate.py`: 0 supported, 1 falsified, 3 not shown, 2 invalid or refusal.

Support, refutation and partial patterns are stated in advance in `criteria.toml` and the preregistration. Ten fresh seeds
3001-3010; pilot seeds 1001-1003 only for tuning and the dry run.

## Files

| File | Content |
|---|---|
| `DRAFT_PREREGISTRATION.md` | question, design, decision rule, pilot disclosure and tuning, limits, entry point and freeze steps |
| `config.toml` | `[pinned_thresholds]` read by the harness; `[preregistration]` with two digests still to be pinned at the freeze |
| `criteria.toml` | coverage, guards, falsifiers and the paired rule |
| `aggregate.py` | judges the seed records, writes `results/metrics.json` and `run.json`; refuses on stale mutation evidence |
| `tests/mutations.toml` | planted defects and the counter or control each must trip |
| `../../../research/consciousness/` | harness (`gw/harness.py`), task, models, mutation runner, tests, pinned `uv.lock` |
| `../../../scripts/tests/test_c001_aggregate.py` | tests of the aggregator and of the statistics |

## Running (after the freeze)

```bash
UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked --project research/consciousness   # once, network variables set; environment outside the repository
python3 scripts/run_experiment.py prepare C001
python3 scripts/run_experiment.py run C001 --seed 3001       # then 3002 ... 3010; each seed runs once
python3 research/consciousness/gw/run_mutations.py           # clean committed tree; writes results/mutations.json
python3 experiments/consciousness/C001-global-workspace/aggregate.py
```

## Gaps in the repository's tooling that this experiment reports instead of working around

* `scripts/mutation_check.py` builds a Cargo package; it cannot plant a defect in a Python entrypoint. `--check` (anchors only)
  works; mutations are run by `research/consciousness/gw/run_mutations.py` in a throwaway copy, writing the layout
  `experiment_records.mutation_evidence` reads.
* `experiment_records.CODE_PATHS` lists only Rust, SQL, proto and Cargo files, so the shared mutation-evidence staleness check does
  not see a change to the Python harness or `uv.lock`. Seed records are bound to the whole tree. `aggregate.py` therefore refuses locally
  when `research/consciousness` or this directory (outside `results/`) differs from the mutation commit, and when the mutation total is not the
  plan's size.
* The runner's command environment passes only `PATH`, `HOME`, `TMPDIR`, `TEMP`, `TMP`, locale and the cargo/rustup homes (no proxy,
  certificate or `UV_*` variable), and it **refuses to run while the repository holds git-ignored files** (`.venv`, `__pycache__`). The
  entrypoint is therefore `env UV_PROJECT_ENVIRONMENT=../../../c001-venv PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline
  --no-python-downloads ... python -B ...`: the environment is outside the repository and complete before the first run. It needs `uv`
  and CPython 3.11 on PATH. The harness refuses with exit 2, before training, when python, torch or the config are not the pinned ones.
  The locked environment has torch only, no numpy.
