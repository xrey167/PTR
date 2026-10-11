# C007-metacognitive-self-model-cpu

The CPU stage A of C002, registered as its own experiment (the scaffolder accepts only a capital letter and three digits). **Status: planned, built to freeze-ready (version 2, after the pre-freeze review); no declared seed has been run.** A calibration (known-answer) run: the likeliest outcome is the registered refutation. Indicators HOT-2/HOT-3 as a computation
([doc 38](../../../docs/architecture/38-consciousness-indicators.md)); indicator properties are not consciousness and nothing here claims that any system is conscious. **No verbal self-report is read, trained or scored.** The GPU
stage B (the Pod-MoE with OLMoE) is `C002-metacognitive-self-model`, which remains planned.

## What is tested, and what is not

A toy two-layer mixture-of-experts classifier, trained and frozen, exposes a trace (router probabilities and selections of both layers, hidden states, output probabilities and entropy). Does a **self-model** that
reads the trace predict the classifier's own errors on a distribution-shifted split (Brier score against the verifier's deterministic outcomes) better than (a) an **information-matched probe** (the same network, features, capacity, optimiser,
data and initial weights; it differs from the self-model only in the weight of an auxiliary self-modelling loss, 0 against 0.3: a masked prediction of the classifier's own layer-two expert choice and output entropy) and (b) the classifier's
own calibrated confidence, which both heads contain as a base term (so neither is weaker than the baseline because of tuning)? The self-model may only observe and propose: the classifier's parameters are digested before and after training, and consuming the advice must leave the verifier's outcomes, the gate's decisions and threshold and
the classifier's digest untouched (the write-path counters; in the toy the self-model has no consumer, so they verify the harness wiring: the real write-path test is stage B, C002). With information matching the likeliest finding is a refutation (a multi-task objective adds little to a probe that already reads the whole trace); it is reported as the
first-class result it is. The traces' value beyond input and output (a probe on those only) is descriptive.

## Controls

Removal controls (the self-model on shuffled traces; a self-model trained on permuted outcomes) must not improve on the constant predictor by the margin; a noise control (Gaussian noise on the predictions) must be resolved as a deficit; the traces must beat the
constant predictor by the margin on every seed (else nothing is left to compare); both heads must be within 4 per mille of the baseline in distribution (else they are too weak for the question); a treatment that never ran (`inert_treatment_arms`) or a calibration fit that saw the wrong split (`fit_split_errors`) is a defect; with the same shuffled traces the self-model's advantage over the probe must vanish too; a null world (labels independent of inputs) must show no difference between the heads in either direction. Any of these failing makes the experiment INVALID.

## Decision (three outcomes plus invalid)

| Outcome | Condition |
|---|---|
| COMPLETED (supported) | guards and controls hold; both lower one-sided 95% bounds over the 20 seeds of shifted-split Brier (probe minus self-model, calibrated own confidence minus self-model) >= 4 per mille |
| FALSIFIED (refuted) | guards and controls hold and, for some comparator, the Bonferroni upper bound < 4 per mille (the improvement is below the margin; not "no improvement") |
| NOT_SHOWN | bounds straddle the margin |
| INVALID | an integrity counter, seeds or mutation evidence missing or partial (these precede every other verdict, including a structural FALSIFIED), no signal in the traces, heads too weak, an uninformative error rate, a judged head at the step cap, an unresolved control, a removal control that does not remove, a null world that shows an effect |

A structural defect on any seed (training changed the classifier, advice reached the verifier or a gate, nondeterminism, a coverage rule) is FALSIFIED, kind structural. Exit status of `aggregate.py`: 0 supported, 1 falsified, 3 not shown, 2 invalid or refusal.

## Expected outcome (pilots, not a result)

On pilot seeds the self-model equals the information-matched probe (within a per mille of Brier) and is about 3 per mille of Brier worse than the classifier's own calibrated confidence it contains, so the registered outcome is expected to be
FALSIFIED (refuted), about 0.92; INVALID about 0.04. A refutation is a first-class result of this experiment; see the preregistration, sections 4 and 6.

## Files

| File | Content |
|---|---|
| `PREREGISTRATION.md` | question, design, decision rule, pilot disclosure and tuning, limits, entry point and the freeze procedure (commands in the repository: `research/cseries/lib/pin_freeze.py`) |
| `config.toml` | `[pinned_thresholds]` read by the harness; `[preregistration]` with two digests pinned at the freeze |
| `criteria.toml` | coverage, guards, falsifiers and the paired rule |
| `aggregate.py` | judges the seed records, writes `results/metrics.json` and `run.json`; refuses on stale mutation evidence |
| `tests/mutations.toml` | planted defects of the harness (50) and the counter, control or unit test each must trip |
| `tests/aggregator_mutations.toml` | planted defects of the aggregator and of `criteria.toml` (44), run with the shared library's 17 by `research/cseries/lib/run_agg_mutations.py C007` |
| `../../../research/cseries/` | harness (`c002a/harness.py`), world, models, planning model, pilot scripts and raw outputs (`c002a/pilots/`), shared library, mutation runner, tests |
| `../../../scripts/tests/test_c007_aggregate.py` | tests of the aggregator and of the statistics |

## Running (after the freeze)

```bash
UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked --project research/consciousness   # once; the environment is C001's, outside the repository
python3 scripts/run_experiment.py prepare C007
python3 scripts/run_experiment.py run C007 --seed 6001       # then 6002 ... 6020; each seed runs once
OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_mutations.py C007   # clean committed tree; writes results/mutations.json
python3 experiments/consciousness/C007-metacognitive-self-model-cpu/aggregate.py
```

Tooling gaps (mutation runner for Python entrypoints; the Python harness is not in `experiment_records.CODE_PATHS`; the stripped runner environment) are the same as C004's and are handled the same way (see its README).
