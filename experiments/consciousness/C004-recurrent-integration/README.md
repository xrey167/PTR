# C004-recurrent-integration

A functional-mechanism test on a CPU toy task, in the consciousness-indicator track
([doc 38](../../../docs/architecture/38-consciousness-indicators.md)): indicator RPT-1 (algorithmic recurrence) as a computation. **Status: planned, built to
freeze-ready; no declared seed has been run.** Indicator properties are not consciousness: this experiment tests whether a carried state integrates evidence over
steps better than a matched fixed-depth stack, and says nothing about whether any system is conscious (RPT-2 is not targeted). See the preregistration (`PREREGISTRATION.md`).

## What is tested, and what is not

On a modular counter (the running sum modulo six, supervised after every step; training on 2 to 6 steps), does a gated recurrent net with a carried state answer at **twelve** steps
(twice the longest trained) better than (a) a causal attention stack of unshared layers with relative-position bias, matched to it in parameters (within 2%) and in FLOPs (within 10%,
at six steps), and (b) the same recurrent net with the state reset at every step? It is not a test of language models, of OLMoE or of the latent recurrence of `ptr-core`/`burn-a0`
themselves; it extends M003 (which asks only about tokens saved) by an integration test. Fixed-depth attention is known to struggle with counting beyond its training lengths: the expected
direction is the registered one, **and in the pilots the stack never extrapolated in any configuration tried** (12-step accuracy 0.18 to 0.30, chance 0.167), so a COMPLETED is close to a known answer
about a non-extrapolating architecture class, not a discovery. The content of the experiment is in the guards (a saturation ceiling of 980 per mille, so that a competent stack yields a refutation and not INVALID; the comparator must have
learned the training distribution), the controls (a label-permutation net, a corrupted-input deficit of the margin's size, and an **extrapolation-removed control**: the same stack trained with the 12-step length in its pool must recover
more than half of the recurrent arm's advantage, else the contrast is one of capacity and the run is INVALID) and the **null worlds** (permuted labels; a task that needs no integrated state) that stop a positional-extrapolation failure of
the stack from being read as a finding about recurrence. The claim is restricted to indicator RPT-1; RPT-L1 and RPT-L2 of doc 38 are not tested.

## Arms

`recurrent` (GRU, 63174 parameters); `feedforward` (causal stack, 5 unshared layers, ALiBi bias at slope scale 4; depth, scale and learning rate pilot-selected on pilot seed 1002 with the recorded rules, the stack's grid larger than the recurrent arm's); `reset_state`
(a floor by construction); `feedforward_inlength` (the same stack trained with the 12-step length in its pool: the extrapolation-removed control, never judged as an arm); sensitivity and null arms `recurrent_perm`, `feedforward_perm`, `recurrent_null`, `feedforward_null` and the eval-only `recurrent_corrupt`.

## Decision (three outcomes plus invalid)

| Outcome | Condition |
|---|---|
| COMPLETED (supported) | guards and controls hold; both lower one-sided 95% bounds over the 14 seeds of 12-step accuracy (recurrent minus feed-forward, minus reset-state) >= 100 per mille |
| FALSIFIED (refuted) | guards and controls hold and, for some comparator, the Bonferroni upper bound < 100 per mille (the advantage is below the margin; not "no advantage") |
| NOT_SHOWN | bounds straddle the margin |
| INVALID | a saturated (at or above 980) or untrained (below 800 in distribution) comparator, an untrained recurrent arm, a judged arm at the step cap, an unresolved control (including the extrapolation-removed control), a null world that shows an advantage, any integrity counter, evidence missing |

**INVALID precedes every FALSIFIED and COMPLETED branch.** The extrapolation-removed control is read only for a supported contrast (so a competent stack yields FALSIFIED, not INVALID); it was loosened after exploration on pilot seed 1002 because the stack is optimisation-limited, and its threshold of one half is a verbal a-priori argument, not from the planning model. A structural defect on any seed (the reset-state ablation carries information over an early step, the stack is not causal, nondeterminism, a coverage rule) is FALSIFIED, kind structural. Parameter or FLOP counts
beyond the tolerances, label or overlap defects and the other integrity counters exit 2 and make the experiment INVALID. Exit status of `aggregate.py`: 0 supported, 1 falsified, 3 not shown, 2 invalid or refusal.

## Expected outcome (pilots, not a result)

On pilot seeds (1003 to 1007, after the seeding fix) the recurrent arm reaches 978 to 995 per mille at 12 steps and the matched stack 178 to 295 (chance is 167); the stack trained on the 12-step length reaches 772 to 911, so it recovers
about 84 percent of the contrast; every guard and control held and the null worlds were resolved. Expectation: **COMPLETED about 0.85, INVALID about 0.12 (mostly the extrapolation-removed control and the step-cap allowance, the stack being
optimisation-limited), FALSIFIED (structural) about 0.01.** That confirms a known direction on a small scale and mainly validates a harness with a refutation path and a control that can fail; see the preregistration (power tables including a competent-baseline table; pilots and chronology).

## Files

| File | Content |
|---|---|
| `PREREGISTRATION.md` | question, design, decision rule, pilot disclosure and tuning, limits, entry point and freeze steps |
| `config.toml` | `[pinned_thresholds]` read by the harness; `[preregistration]` with two digests pinned at the freeze |
| `criteria.toml` | coverage, guards, falsifiers and the paired rule |
| `aggregate.py` | judges the seed records, writes `results/metrics.json` and `run.json`; refuses on stale mutation evidence |
| `tests/mutations.toml` | planted defects (44 harness defects; the aggregator's are in `tests/aggregator_mutations.toml`) and the counter, control or unit test each must trip |
| `../../../research/cseries/` | harness (`c004/harness.py`), task, models, planning model, pilot scripts and raw outputs (`c004/pilots/`), shared library, mutation runner, tests |
| `../../../scripts/tests/test_c004_aggregate.py` | tests of the aggregator and of the statistics |

## Running (after the freeze)

```bash
UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked --project research/consciousness   # once, network variables set; the environment is C001's, outside the repository
python3 scripts/run_experiment.py preflight C004   # the manifest's preflight command; the runner also runs it before every seed
python3 scripts/run_experiment.py prepare C004
python3 scripts/run_experiment.py run C004 --seed 5001       # then 5002 ... 5014; each seed runs once
OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_mutations.py C004   # clean committed tree; writes results/mutations.json
python3 experiments/consciousness/C004-recurrent-integration/aggregate.py
```

## Gaps in the repository's tooling that this experiment reports instead of working around

* `scripts/mutation_check.py` builds a Cargo package; it cannot plant a defect in a Python entrypoint. `--check` (anchors only) works; mutations are run by
  `research/cseries/lib/run_mutations.py` in a throwaway copy, writing the layout `experiment_records.mutation_evidence` reads.
* `experiment_records.CODE_PATHS` lists only Rust, SQL, proto and Cargo files, so the shared mutation-evidence staleness check does not see a change to the Python harness. `aggregate.py` therefore refuses locally
  when `research/cseries`, the lock files it reads, or this directory (outside `results/`) differs from the mutation commit, and when the mutation total is not the plan's size.
* The runner's command environment passes only `PATH`, `HOME`, `TMPDIR`, locale and the cargo/rustup homes and it refuses to run while the repository holds git-ignored files; the entrypoint uses `env UV_PROJECT_ENVIRONMENT=../../../c001-venv
  PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline --no-python-downloads ... python -B ...` with C001's locked environment (no lock file of its own was added: no dependency is new).
