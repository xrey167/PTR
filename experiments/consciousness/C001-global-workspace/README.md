# C001-global-workspace

A functional-mechanism test on a CPU toy task, the first experiment of the consciousness-indicator track
([doc 38](../../../docs/architecture/38-consciousness-indicators.md)). **Status: planned, redesigned (v3) and built to
freeze-ready; no declared seed has been run.** Indicator properties are not consciousness: this experiment tests whether a
mechanism (GWT-1..4 as a computation) helps, and says nothing about whether any system is conscious.

Versions 1 and 2 were withdrawn after review (a ceiling; then an operating point and tuning that favoured one arm). See
the preregistration (`PREREGISTRATION.md`).

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
of their two values (5 classes). Training: a finite set of 2048 eight-module instances, every arm trained to a validation plateau. Evaluation: sixteen modules (primary), twenty-four,
and eight (more distractors than ever seen in training).

## Arms (parameter counts within 2%)

workspace; the three pairwise comparators all_to_all, bw_all_to_all (value width reduced to the workspace's slot content) and sparse_all_to_all
(top-k keys per query, k = slots); the attribution ablations no_broadcast (GWT-3), capacity_off (8 slots; GWT-2) and competition_off (soft write),
reported three-valued, never judged; no_comm (floor); crippled_all_to_all (one round) and weak_all_to_all (the all-to-all net with Adam weight decay 0.001): the gross and graded
sensitivity controls. Every arm runs at d = 48 with the update width matched to the workspace's parameter count.

## Decision (three outcomes plus invalid)

| Outcome | Condition | Consequence for the Pod-MoE workspace design |
|---|---|---|
| COMPLETED (supported) | guards hold; every lower one-sided 95% bound of (workspace minus all_to_all, bw_all_to_all, sparse_all_to_all) on 16 modules >= 30 per mille | build the learned L2 prototype and run C006 (a new test on Pods, not a consequence); the broadcast (no_broadcast), bottleneck (capacity_off) and competition (competition_off) contrasts are reported attributed / not attributed / unresolved by noise. **COMPLETED shows an advantage over pairwise attention at 2048 examples and 16 modules; it does not show that the broadcast read, the capacity limit or the competition is the reason (each may read "not attributed")** |
| FALSIFIED (refuted) | guards hold, both sensitivity controls resolve, the Bonferroni upper bound against some pairwise comparator < 30 per mille (the advantage is below the margin; not "no advantage") | negative result for this setting (2048 training examples, 16 modules); downgrade GW-L1/GW-L2 (learned workspace) to exploratory for that setting; a larger-D successor is not excluded (the pilot advantage grows with D) |
| NOT_SHOWN | bounds straddle the margin | nothing changes, no downgrade; a larger successor may follow |
| INVALID | strongest pairwise comparator saturated (> 900) or floored (< 300) on any seed, the workspace or a pairwise comparator at the step cap on more than 2 seeds, a sensitivity control unresolved, workspace untrained on > 4 of 20 seeds (they stay in the paired differences), a harness-integrity counter on any seed, evidence missing | nothing is concluded; repair and register a successor |

The 30 per mille margin is justified by cost (about three points of 16-module accuracy: the smallest gain worth a new architectural block); it was fixed
after the pilots had been seen, and the earlier draft's 50 (the pilot gap) is withdrawn; see the preregistration, section 4.

A structural defect on any seed (a coverage rule or a structural counter of the harness: isolation, bottleneck, dropped-module, determinism,
no-communication floor) is FALSIFIED, kind structural: a defective arm, not a result about the hypothesis. A defect of the task or the
harness (module counts, train/validation/evaluation overlap, parameter matching, non-finite loss, checkpoint or convergence bookkeeping)
exits 2 and makes the experiment INVALID. Exit status of `aggregate.py`: 0 supported, 1 falsified, 3 not shown, 2 invalid or refusal.

Support, refutation and partial patterns are stated in advance in `criteria.toml` and the preregistration. Twenty fresh seeds
4001-4020; pilot seeds 1001-1030 only for tuning, calibration and the dry runs (raw outputs and scripts in `research/consciousness/gw/pilots/`). There is **no single expected outcome**: reading the pilots three ways, P(COMPLETED) is between 0.02 and 0.69
(tuning-seed means: NOT_SHOWN 0.93-0.97; fresh out-of-sample seeds: COMPLETED 0.17-0.58; all trained pilot seeds: 0.20-0.69). The workspace stalled on 1 of 17 pilot seeds (not flagged by the guard's harness rule; it stays in the paired differences; the guard tolerates 4 flagged seeds of 20),
which dominates: at the best pooled estimate and that stall rate the outcome probabilities are NOT_SHOWN 0.78, COMPLETED 0.20, INVALID 0.02, FALSIFIED about 0 (preregistration section 4); at an advantage equal to the margin it is NOT_SHOWN 0.92-0.93.

## Files

| File | Content |
|---|---|
| `PREREGISTRATION.md` | question, design, decision rule, pilot disclosure and tuning, limits, entry point and freeze steps |
| `config.toml` | `[pinned_thresholds]` read by the harness; `[preregistration]` with two digests pinned at the freeze |
| `criteria.toml` | coverage, guards, falsifiers and the paired rule |
| `aggregate.py` | judges the seed records, writes `results/metrics.json` and `run.json`; refuses on stale mutation evidence |
| `tests/mutations.toml` | planted defects (31) and the counter, control or unit test each must trip |
| `../../../research/consciousness/` | harness (`gw/harness.py`), task, models, mutation runner, power estimate, pilot scripts and raw pilot outputs (`gw/pilots/`), tests, pinned `uv.lock` |
| `../../../scripts/tests/test_c001_aggregate.py` | tests of the aggregator and of the statistics |

## Running (after the freeze)

```bash
UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked --project research/consciousness   # once, network variables set; environment outside the repository
python -B research/consciousness/gw/harness.py --benchmark c001 --preflight   # via the same env/uv entrypoint, before every seed
python3 scripts/run_experiment.py prepare C001
python3 scripts/run_experiment.py run C001 --seed 4001       # then 4002 ... 4020; each seed runs once
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
