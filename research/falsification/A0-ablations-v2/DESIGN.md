# A0 mechanism ablation study v2: design

v2 is the successor of [A0-ablations-v1](../A0-ablations-v1/DESIGN.md). It tests the same hypotheses, with the same arms, contrasts, gates and thresholds, on the corrected A0 implementation, and it collects the correctness evidence (gate G6) that v1 lacks. It makes **no new claim**. Like v1 it is an A0-internal ablation on the synthetic benchmark operator-routing v1: it says nothing about language-model quality and is not the M001-M004 baseline comparison (those stay `planned`; owner decision O2).

## Why a successor

[EVIDENCE_REPAIR.md](../A0-ablations-v1/EVIDENCE_REPAIR.md) records that v1's gate G6 failed: its `logs/g6.json` held only pass labels, not bound to the evaluated commit `4b0ad160`, so every gated v1 verdict is INCONCLUSIVE. It also records two reasons v1 cannot be repaired in place:

1. Admission masking was corrected after v1's evaluation: admission-aware batches now replace the raw attribute tokens of non-live facts with PAD (commit `be0e17ab`), so the arms that read raw attribute tokens under an admission mask (the study owner counted 8 of v1's 9; the unmasked content-free control `no-semantic-slots` keeps its raw input) train differently now. Archived runs cannot be made to describe the corrected code, and the learning rates v1 selected were selected on the old code.
2. `scripts/run_experiment.py` refuses `a0_*` launches under M001-M004, because the exploratory runner cannot attest the build environment. "New training must use a newly registered study and the hardened runner's frozen entrypoint, isolated target and recorded executable/toolchain."

v1 stays as history, unchanged.

## What is the same as v1

- Hypotheses, arms, contrasts, composites, learnability bars, contingency rule, leak alarm and the decision rules: `criteria.toml` is v1's file with only the study name and tag replaced (a test compares the two).
- Data: operator-routing v1, generator version 1, `splits.lock.json` (data FNV-1a-64 `0ad71688f09b0d0d`), codebook V1.
- Model and training: one block, `d_model` as the calibration rule chooses (D14), Adam defaults, batch 128, warm-up 100 steps then cosine to 0.1x, seeds [17, 29, 43, 71, 101], calibration seed 0, sweep at runner seed 17 with the salted seeds.
- Learning-rate grid [0.002, 0.005, 0.0125], selection tolerance 0.005, swept **again** after the freeze, now on the corrected code.
- The budget rule (limit 20 min, hard cap 45, 3 workers, the same ladder) and the calibration ladder (base, R1, R2): `model/configs/a0_ablation_study_v2.toml` is v1's file with the study name changed (a test compares the arm tables and every threshold). The budget rule bounds training compute; it is applied to the training time projection exactly as in v1. Cold builds are not part of it (below).
- Gates G0-G6 as DESIGN.md of v1 defines them (data, competence, determinism, completeness, freeze, scorer agreement, correctness), except for the adaptations in the next section.

## Every difference from v1, with its reason

| # | v1 | v2 | Reason (repository rule) |
|---|---|---|---|
| D1 | A0 before the admission fix | A0 at the commit that holds the fix (`be0e17ab`); the unmasked control keeps its raw input by design | EVIDENCE_REPAIR.md: "a new frozen study must train the corrected implementation". |
| D2 | Runs through the exploratory runner, as `a0_*` entrypoints of the planned M001-M004 manifests | Every process is one seed of a **listed** experiment of its own (`experiments/preregistration.toml`), run through the hardened runner by its one frozen `entrypoint`; M001-M004 are untouched and the v1 refusal in `run_experiment.py` stays | `run_experiment.py`: "A0 v1 is archival ... new model runs require a newly registered frozen study using the hardened entrypoint runner". A listed experiment runs only through its manifest's `entrypoint`, takes every placeholder but `<seed>` from its frozen `[preregistration]` table, and runs each seed once (`check_research_gates.py`). |
| D3 | One sweep command per (family, lr), evaluation per family, as entrypoint keys | One experiment per (family, lr) for the sweep (12), one per family for the evaluation (4), one for the determinism rerun, one or two per contingency: ids `M00x-a0v2-sweep-lr<lr>`, `M00x-a0v2-eval`, `M001-a0v2-rerun`, `M00x-a0v2-cont-<arm>` | A seed runs once per listed experiment, so runs that share a seed need separate experiments. |
| D4 | One freeze (tag `a0-ablation-prereg-v1`), the lr selection committed afterwards | **Two freezes**. Freeze 1 (tag `a0-ablations-v2-prereg`): study files, criteria, budget, and the 12 sweep experiments. Freeze 2 (the commit G6 tests, tag `a0-ablations-v2-eval`): the committed `lr_selection.*` and the evaluation and rerun experiments. A contingency's experiments are registered in a later commit, only if the frozen learnability rule triggers it | The evaluation experiments must freeze the SHA-256 of `lr_file` before they leave `planned`, and that file exists only after the sweep. No evaluation outcome exists at freeze 2, so the preregistration still precedes every outcome it governs. |
| D5 | Gate G4: every record has one `git_sha`; the diff from the tag to it touches only `lr_selection.*` and sweep records | G4: every record is clean with an empty tracked diff, descends from the tag, and runs **the same code tree as the evaluation commit** (the commit G6 tested): sweep records at ancestors of it, later records at descendants, differing only by run records, G6 logs and the registration of later-stage experiments. The diff tag..evaluation commit may touch only `lr_selection.*`, run records and later-stage registration, and the two shared registries may only gain later-stage v2 entries | The hardened runner commits every record before the next launch, so records name different commits. The gate for each experiment (`earlier_run_problems`) already holds its seeds to one tree; G4 extends that across experiments. |
| D6 | G6 evidence at the lr-selection commit, with logs written during evaluation | `scripts/run_a0_ablation_study.py --study v2 g6` runs the checks at the clean freeze-2 commit, builds outside the repository (a listed run refuses any git-ignored file) and records the exact argv, exit codes, `git_sha` before and after, clean flags and log SHA-256. The aggregator requires `git_sha` to be the evaluation commit and every record to run the same code. The Python checks also cover the v2 mechanism's own tests | G6 must be commit-bound (GOAL step 2). Checks and evaluation both build from the same commit with the pinned toolchain `+1.95.0`; they use different target directories because the hardened runner builds into a scratch directory of its own. |
| D7 | `rustc` field in records | Records carry `toolchain` and `executable` (SHA-256 of rustc and of the program that started it) and `environment`; `rustc` is empty | Hardened record format. G4 compares host, toolchain and executable across all records. |
| D8 | A failed or NaN process gets one full retry (G3) | No retry: the hardened runner refuses a second run of a seed, which keeps any run from being chosen by its outcome. A failed process makes every dependent verdict INCONCLUSIVE and a further study needs a successor | `run_experiment.py`: "a listed experiment runs each seed once, so no run of it is chosen by its outcome". |
| D9 | Cross-check against `run_experiment.py aggregate` | The same comparison done in the aggregator on each record's printed `final` rows against the independent score.py table, to 1e-12 | The stock aggregate refuses records from more than one commit (`aggregate_experiment.py`), which hardened records always are. |
| D10 | Per-family `results/a0_internal_metrics.json` inside M001-M004 | `research/falsification/A0-ablations-v2/metrics/<family>.json` | The runner holds a listed experiment's directory to HEAD; the aggregate must not write into it. |
| D11 | Parallel runs (3 workers) on a warm shared target directory | Serial runs; each builds the example cold into a scratch directory outside the repository (`CARGO_TARGET_DIR`, removed afterwards) | The hardened runner holds the whole tree to HEAD, refuses git-ignored files and runs one launch at a time. Measured on this host in a rehearsal at 20 steps: a run took 135-200 s, cold build included. |
| D12 | Preregistration identified by a commit constant in the aggregator | Identified by the tag `a0-ablations-v2-prereg`, which must exist | A commit cannot name itself in its own tree. The tag is created by hand at freeze 1. |
| D13 | `scripts/run_a0_ablation_study.py`, `aggregate_a0_ablation.py`, `report_a0_ablation.py` | The same scripts with `--study v2`; identities and naming in `scripts/a0_study.py` | v1 behaviour and tests are unchanged. |

| D14 | Calibration missed its 0.88 target at `d_model` 32 (0.8615), so rung R1 chose `d_model` 48, and the budget ladder dropped `latent-4` and the blind-query pair | The same calibration rule on the corrected code reached 0.8820 at `d_model` 32 and 1500 steps (`calibration.json`), so the study runs at `d_model` 32 (S* = 1500), no ladder rung applies (`budget.json`: projected 10.5 minutes, limit 20), and the tier-2 arms `latent-4`, `blind-query-k0` and `blind-query-k0-no-typed-attention` run | The preregistered calibration and budget rules are applied unchanged; their outcome differs because the code does. It enlarges the arm set, which brings the M002 sufficiency contrast and the M003 dose response, not run in v1, into scope. |

Hardware profile: `hardware/a0-cpu-4core.toml`, the same container class as v1; every record names the measured host.

## Gates and what they decide

G0-G6 decide whether verdicts can be issued, as in v1; a verdict depending on a failed gate is INCONCLUSIVE. Learnability, the contingency, the manipulation check and the leak alarm are v1's rules in `criteria.toml`. Results are reported in `RESULTS.md` with every verdict, including nulls and INCONCLUSIVE ones.

## Which v1 verdicts v2 can supersede

If, and only if, v2's gates G0-G6 all pass, its verdicts supersede v1's current INCONCLUSIVE ones for the same contrast: M001-primary, M001-secondary, M002-necessity, M002-raw-blind-check, M003-depth, M003-nonlinearity and M004-negative-control. M002-sufficiency and the M003 dose response (v1: not run under the budget rule) have no v1 verdict to supersede; v2 can issue them first because its budget keeps the tier-2 arms (D14). `no-verifier-head` stays NOT TESTED. If any gate fails, v2 leaves v1's status unchanged and v2's own verdicts are INCONCLUSIVE. v1 and v2 data are never pooled, and v1's historical raw scores remain descriptive only.

## Budget

The budget is computed by `prefreeze` from the calibration (`budget.json`, as in v1) and frozen at freeze 1. Execution is serial, so the wall time is the training projection times the number of workers v1 assumed, plus one cold build (about 2 minutes on this host) per run: the sweep is 12 runs, the evaluation 20 plus the rerun (arms per run: 3, 4, 4 and 1 for M001-M004), and a contingency 5 to 10. Calibration measured 10.3 ms per step, so training is well under a minute per arm and the cold build dominates. A run that fails stops the study.

## Procedure (`...` is `python3 scripts/run_a0_ablation_study.py`; `--commit` makes the driver commit each record alone)

1. Commit the mechanism and this design. Run `python3 scripts/run_a0_ablation_study.py --study v2 prefreeze` from the clean commit (it builds into the ignored `model/burn-a0/target-a0-study`; delete it before step 2's runs) (writes `references.json`, `calibration.json`, `budget.json`); commit them.
2. `... --study v2 prereg` registers the 12 frozen sweep experiments (and appends to the two registries); commit. **Tag `a0-ablations-v2-prereg` on that commit.** Remove git-ignored files (`git clean -fdX`).
3. `... --study v2 sweep --commit` runs the 12 sweep experiments, one commit per record.
4. `... --study v2 select` writes `lr_selection.tsv/json`; `... --study v2 prereg-stage eval` registers the evaluation and rerun experiments; commit. **Tag `a0-ablations-v2-eval` on that commit** (the evaluation commit).
5. `... --study v2 g6` runs the correctness checks at that commit and writes `logs/`; commit the logs.
6. `... --study v2 eval --commit` runs 20 evaluation seeds and the rerun, one commit per record.
7. `... --study v2 aggregate` writes `results.json`, `metrics/`, `RESULTS.md`. If the frozen learnability rule requires a contingency, `prereg-stage contingency --arm <arm>`, commit, `contingency <arm> --commit`, aggregate again.
