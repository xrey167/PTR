# A0 baseline study v2: design

The complete successor of study v1 (M001-v5, M002-v10, M003-v2; replaced for a decider defect, see "Why v2 replaced v1"), and thereby of M001-v4 / M002-v4 (and of the superseded M002-v5 to M002-v9), against the pinned plain baseline. Three studies, one design: **M001-v6** (semantic slots), **M002-v11** (factorized typed pair attention), **M003-v3** (latent recurrence). M004-v2 is registered as planned and deferred. Every threshold is in [`criteria.toml`](criteria.toml), the frozen text is [`PREREGISTRATION.md`](PREREGISTRATION.md), and the decision procedure is `scripts/aggregate_a0_baseline_v2.py`.

**Scope.** A0 (one block, `d_model` 48, 1500 steps, CPU) on the synthetic operator-routing v2 benchmark. The baseline is the already pinned A0-matched plain model (`research/baselines/plain_model/config.toml`, `ptr-burn-a0@matched-cross-attention-v1`; ADR-0021). A result says at most something about A0 on this synthetic task; nothing about language-model quality, and nothing about a pretrained LM baseline (deferred to the GPU phase). A negative result is as valid a decision as a positive one and is recorded the same way.

## Why v2 replaced v1 (no v1 outcome was observed)

Study v1 (`research/falsification/A0-baseline-v1/`: M001-v5, M002-v10, M003-v2 with their sweeps) was replaced by this complete successor, by an owner decision of 2026-10-09, for a **procedural defect of its decider, not for any outcome**:

1. **Record identity.** The v1 decider (`scripts/aggregate_a0_baseline.py`, via the pinned core `scripts/aggregate_m002_v5.py`, `IDENTITY_FIELDS`) requires that the five run records of a study name the same `git_sha`. The hardened runner (`scripts/run_experiment.py`) commits each seed's record before the next run, and a record names the commit HEAD was at when its run started. Every correctly executed v1 study therefore has five records at five commits, and the v1 decider refuses ("run records disagree in git_sha") before it computes anything. The defect is in the rule that decides what counts as one build, not in a statistic, gate or threshold.
2. **Interruptions.** Two container restarts interrupted runs: the sweep `M002-v10-sweep` seed 17 (superseded by `M002-v10-sweep-r2`) and the confirmatory `M002-v10` seed 101 (record status `started`, no output, no exit code). v1 had a documented procedure for an interrupted sweep and none for an interrupted confirmatory seed; its decider would have refused the study. Account: `research/falsification/A0-baseline-v1/DEVIATIONS.md` (D1) and the `SUPERSEDED.md` files of the v1 confirmatory experiments.

**No v1 outcome was observed.** The v1 decider never ran on the confirmatory records; nobody opened their stdout; only record metadata (status, `git_sha`, seed) was read to diagnose the defect. The v1 confirmatory experiments (M001-v5, M002-v10, M003-v2) are marked `superseded`, their records stay in the repository as history and **must not be read or compared with v2 results before the v2 decision**. The v1 sweeps are validation-only, are not part of v2, and their learning-rate files (`lr_M001-v5.tsv`, `lr_M002-v10.tsv`, `lr_M003-v2.tsv`) are **not used** by v2: v2 runs fresh sweeps for all three studies and selects its own rates with the same fixed rule. Hence nothing observed in v1 can have informed a v2 choice, and the v2 protocol is v1's protocol unchanged.

### What is identical and what changed

Identical to v1, value for value (`criteria.toml` of v2 differs from v1's only in the names listed here; a test compares them): the protocol (`d_model` 48, 1500 steps, batch 128, Adam, warm-up 100 then cosine to 0.1x, calibrated router, rank 8, bias limit 1.0, metadata dropout 0.0), the data and fold tokens, the seeds 17, 29, 43, 71, 101, the lr grid {0.0025, 0.005, 0.0125, 0.025} and its selection rule, the arms and pins, the contrasts and compute modes, every gate and threshold, the decision rules, the runner (`model/burn-a0`, unchanged), the baseline pin. The statistics are not re-implemented: the v2 decider imports the v1 decider and the immutable core and re-runs every v1 test against itself.

| | v1 | v2 |
|---|---|---|
| (a) decider | `scripts/aggregate_a0_baseline.py` | `scripts/aggregate_a0_baseline_v2.py`, which imports the v1 decider (pinned unchanged in `criteria.toml`, `decision.base_sha256`) and adds (b) and (c) |
| (b) record identity | five identical `git_sha` | the rule below |
| (c) interruption | refusal (unhandled) | outcome `INCONCLUSIVE-INTERRUPTED` |
| (d) ids | M001-v5, M002-v10, M003-v2; sweeps `-sweep` (M002: `-sweep-r2`) | M001-v6, M002-v11, M003-v3; sweeps M001-v6-sweep, M002-v11-sweep, M003-v3-sweep |

M004 is not versioned: **M004-v3 is not created**; M004 stays deferred through M004-v2 (ADR-0021 item 2) and v2 does not touch it.

### Record identity rule (change b)

The five records are one build iff **all other identity fields are identical** (`manifest_sha256`, `parameters`, `rustc`, `toolchain`, `environment`, `executable`, `host`, `cargo_lock_sha256`, `uv_lock_sha256`, and the manifest's `preregistration_sha256` and `preregistration_rules_sha256`; unchanged from v1) **and**

1. the five `git_sha` are identical (v1 rule, no git needed), **or**
2. every `git_sha` is a commit of the history of the repository's HEAD, ordered by the record's `started_at` each commit is an ancestor of the next one (or equal to it), and `git diff --name-only` between the earliest and the latest record commit lists only files directly inside the `results/` directory of an experiment of this study (`experiments/model/<study id>-<slug>/results/`, sweeps included).

Anything else is a refusal, never a verdict: a record without a commit, without `started_at` while the commits differ, a commit the repository does not hold or that is not in HEAD's history, records on diverging branches or in an order that contradicts history, a code, configuration, criteria, script or other experiment's file changed between the earliest and the latest record. The rule is the code-side consequence of the run protocol (commit the record, run the next seed); it does not weaken the requirement that the five runs ran the same code, it states how that is verified. The aggregate takes the earliest record commit as its `git_sha` (as `experiment_records.source_revision` does).

### Interruption rule (change c)

A run record of a confirmatory seed with status `started`, no `stdout` and no exit code is a reservation that nothing finished (the process or its host was killed). If any record of a study is such a reservation, the study's outcome is **INCONCLUSIVE-INTERRUPTED**: named, reported in `DECISION.toml` and `paired-ci-95.json` (status `not-computed-interrupted`), exit code 0, `go = false`, no quality claim, every contrast carries the label. **No statistic is computed and the rows of the seeds that did complete are not read**, so the outcome cannot depend on any result. The seed is spent and **the study is never rerun**: a successor study (new ids, new directory, preregistration and freeze) is created by the owner if the question is still wanted. A failed run (exit code not 0) or any other malformed evidence stays a refusal; an interrupted sweep is handled by a successor sweep (`DEVIATIONS.md`).

## Why a successor: what failed in v4 and after

(Carried over from v1; these are the reasons the design differs from M001-v4 / M002-v4.)

1. **Accuracy up, calibration collapsed, no gate said so beforehand.** M001-v4 and M002-v4 raised OOD accuracy on `ood_compose_regime` (+0.087 / +0.094) while NLL rose by +7.1 / +6.8 and ECE15 by +0.26 / +0.25, both intervals excluding 0. The preregistration had no calibration gate, so the decision (INCONCLUSIVE/NO-GO) was recorded after the fact. Here the calibration gates and the label that follows from them (NO-GO-CALIBRATION) are written before any run.
2. **Raw evaluation path and legacy router.** v4 used the raw `evaluate` path with the legacy mean-linear router (logit scale 5). Every A0 arm now uses the calibrated cosine router (`calibrated-cosine-v2`) and is scored on raw logits (gates) and temperature-fitted logits (diagnostic).
3. **The plain baseline was tuned with the wrong learning rate.** `learning_rates(lr_file)[arm.name]` gave plain the rate selected for the A0 arm. Plain now has its own val-only sweep, and an own rate, in the same process.
4. **No committed producer of `paired-ci-95.json`.** The v4 CI artifact was built by hand and bound by hash in `check_research_gates.py`. `scripts/aggregate_a0_baseline_v2.py` now computes every paired interval, gate and verdict and writes `paired-ci-95.json` and `DECISION.toml`.
5. **M002-v4 did not isolate typed attention.** It contrasted `no-typed-attention` with plain, so it measured a bundle difference. The M002-v11 primary contrast is the identical FactorizedV2 computation with the pair bias multiplied by zero.
6. **v5 to v9 were superseded for harness defects**: fold-token format, arm-name binding per experiment (a pair named `-v9` could not be reused), a missing source-tree binding, a wrong checker, and a `model/burn-a0/Cargo.lock` change that invalidated `implementation_tree_git_digest`. Arm tables are now scoped to a `Study` value (not to a name suffix), and the digest is taken fresh after the lock stabilises (open point for registration).
7. **No pooled-hides-fold safeguard.** v4's failure was visible on one OOD split and diluted in pooled numbers. Every calibration gate is evaluated per fold.

## Arms and contrasts

All A0 arms: calibrated cosine router, `typed_attention_rank` 8, bias limit 1.0, metadata dropout 0.0, warm-up 100 steps then cosine to 0.1x, Adam, batch 128, 1500 steps, `d_model` 48, init seed = seed, order seed = seed XOR `0x0BA7C4` (identical data order across arms). The factorized pair keeps the M002-v9 regularisation (label smoothing 0.05, consistency 0.10); plain trains without label smoothing.

| Study | Arms (A0 + plain) | Primary (isolating) | Secondary | Bundle vs baseline (non-isolating) |
|---|---|---|---|---|
| M001-v6 | `full`, `no-semantic-slots-masked`, `no-semantic-slots`, `plain` | `full` vs `no-semantic-slots-masked` | none | `full` vs `plain` |
| M002-v11 | `factorized-v2`, `factorized-v2-off`, `full`, `plain` | `factorized-v2` vs `factorized-v2-off` (identical compute) | none | `full` vs `plain` |
| M003-v3 | `full` (k=2), `latent-0`, `latent-1`, `latent-linear`, `plain` | `full` vs `latent-0` (recurrence) | `full` vs `latent-linear` (nonlinearity, matched params and FLOPs) | `full` vs `plain` |

- `no-semantic-slots` (unmasked, M001-v6) is a diagnostic arm: the A0-v2 mask share is `masked - unmasked` accuracy and needs it (cost: about +20 ms/step). No gate uses it.
- `latent-1` is descriptive (dose response); it has no hypothesis and no verdict.
- A bundle row can never be read as mechanism evidence. `DECISION.toml` states `allow_bundle_claim_as_mechanism_evidence = false`.
- **Each contrast gets its own verdict. The headline verdict of a study is its primary contrast's.** No multiplicity correction is applied: contrasts answer different questions and are not combined.
- M002-v11 never reports `no-typed-attention` against plain as typed-attention evidence.

## Data

Operator-routing v2 (`datasets/generated/operator_routing_v2`, `benchmarks/operator-routing-v2/splits.lock.json`, SHA-256 `6a001b37...266a1`), the three confirmatory folds `evidence-temporal`, `evidence-tabular`, `claim-interventional`, passed as `name=FNV64` tokens copied from `experiments/model/M002-v9-*/config.toml`. A v2 fold has the splits `train`, `val`, `test_iid`, `test_ood`: **"per OOD split" means per fold on `test_ood`**; there is no `ood_compose_regime` split in v2. The decider verifies each fold's data FNV64 against the lock. v1 folds are not run (confirmatory data is v2 only).

Folds are not independent replicates. A pooled endpoint averages the three folds inside a seed; the five seed values enter the interval. A per-fold endpoint pairs over the five seeds. Seeds: 17, 29, 43, 71, 101.

## Learning-rate sweep and selection

One sweep **per study** at seed 17 over the fixed grid {0.0025, 0.005, 0.0125, 0.025} (runner phase `baseline-v1-sweep`; salted init seed `seed XOR 0x5EE9`; validation only, test splits never read), for every arm including plain. `baseline-v1-select` applies the rule fixed before the first sweep: highest mean (over folds) validation accuracy; tie lower mean validation NLL; tie smaller lr; a grid point that diverged on any fold is ineligible. It writes one lr file per study; there is no hand merge. `full` and `plain` are deterministic across studies, so their sweeps repeat; agreeing selections are a consistency check, not an assumption. The lr in force is recorded in every meta row; the decider refuses an lr off the grid or varying across seeds, and checks it against the frozen lr file when given (`--lr-file`).

## Metrics and where they enter

| Quantity | Column | Role |
|---|---|---|
| Accuracy (OOD, IID) | `accuracy` | OOD, mean over folds: the **sole primary endpoint** |
| NLL | `raw_nll` | per-fold OOD gate (raw logits, T=1) |
| ECE, 15 equal-width bins | `raw_ece15` | per-fold OOD gate |
| Coverage, selective error | `raw_coverage`, `raw_selective_error` | gates, threshold fit on validation at T=1 |
| Temperature-fitted NLL / ECE | `fitted_nll`, `fitted_ece15`, `fitted_temperature` | **labelled diagnostics, no gate** |
| Cost | `total_params`, `flops_per_example`, `ms_per_step`, `latency_p95_ms`, `latent_steps`, `explicit_token_count` | descriptive, plus the compute match |

The fitted-temperature grid runs 0.25 to 4.0 and **caps at 4.0**; for a badly overconfident arm the fit saturates and the fitted columns understate its miscalibration. They are diagnostics for exactly this reason, and the decider reports the saturated-cell count per arm and split.

## Gates

All comparisons are exact on float64; nothing is rounded before comparing. Intervals are paired Student-t over five seeds (4 df, t = 2.7764 from the core, printed 2.776; the unrounded value is slightly wider).

| Gate | Rule (treatment minus control) |
|---|---|
| `ood_accuracy` (primary) | OOD accuracy, mean over folds: mean delta >= 0.02 and CI lower > 0 |
| `positive_accuracy_each_fold` | mean OOD accuracy delta > 0 in every fold |
| `ood_nll_each_fold` | per fold on `test_ood`: upper CI of the raw NLL delta <= 0 |
| `ood_ece_each_fold` | per fold on `test_ood`: mean raw ECE15 delta <= 0 and upper CI <= 0.02 |
| `iid_accuracy` | IID accuracy, mean over folds: CI lower >= -0.01 |
| `coverage` | the treatment arm, every seed x fold: raw coverage IID >= 0.90 and OOD >= 0.80 |
| `selective_error` | mean OOD raw selective-error delta (mean over folds) <= 0 |
| `matched_compute` | by contrast mode, below |
| `complete_finite_bound_evidence` | five records, every seed x fold x arm x split present and finite, no NaN training, protocol pinned, dataset digests equal the lock, one build identity |

`matched_compute` modes: **vs plain** `total_params` within 1% (as `tests/plain_baseline.rs` pins); the FLOP columns are *not* gated against plain because the A0 and plain FLOP helpers are not comparable (1,533,888 vs 1,665,216 at rank 8, 7.9% apart; documented, descriptive). **A0-vs-A0 isolating pairs** (`factorized-v2` vs `-off`, `full` vs `latent-linear`): equal `total_params` and equal `flops_per_example`. **`full` vs `latent-0`**: equal `total_params`; FLOPs differ by the recurrence itself and are descriptive. p95 latency is reported, not gated (shared CPU, noisy).

## Decision rules (per contrast)

- **SUPPORTS**: every gate holds, the primary included.
- **FALSIFIES**: the pooled OOD accuracy CI upper bound < 0.02, both arms learnable (mean IID accuracy over seeds and folds >= 0.30, an a-priori floor of about 3.3x uniform chance over 11 operators) and `matched_compute` holds; **HARMFUL** when the upper bound is < 0.
- **NO-GO-CALIBRATION**: the accuracy endpoint holds but a calibration gate (`ood_nll_each_fold`, `ood_ece_each_fold`) fails. Recorded as INCONCLUSIVE with this label; it forbids any quality claim and is a documented outcome, not an execution failure.
- **INCONCLUSIVE** otherwise, naming every failing gate.
- Evidence that cannot support any decision (missing seed, fold, arm or split; non-finite value; arm or protocol mismatch; digest mismatch; mixed builds, where "one build" is the record-identity rule above) is a **refusal**, not a verdict. An interrupted seed is the named outcome **INCONCLUSIVE-INTERRUPTED** (interruption rule above), not a refusal. There are no reruns (below).

## M003 token definition X

"Explicit token" = a non-PAD input token of the split's raw dataset input, identical across arms by construction (`explicit_token_count`). **Token savings are not tested and no claim about them can follow**: the study varies latent computation (`latent_steps`), not the number of tokens the model reads. Reported: `latent_steps`, wall time per step, FLOPs per example. Typed arms additionally zero the attribute tokens of non-live facts inside the model (admission); those are not subtracted from the count.

## M004: deferred

M004-v2 is registered as planned, "not run at A0": ADR-0021 item 2 defers the router comparison with the pretrained LM baseline to the GPU phase, and A0 has no LM-only arm. `criteria.toml` lists it under `[[deferred]]`; the decider refuses to decide it.

## Prior for M002 (A0-v2)

A0-ablations-v2 (`d_model` 32, operator-routing v1) found the rank-1 typed bias adds nothing measurable beyond QK (M002-necessity FALSIFIES: composite-B mean -0.0003, CI upper 0.016) while a benefit appeared only where the bias was the sole metadata-dependent read (sufficiency +0.103). A null is therefore plausible for M002-v11 at `d_model` 48 on a different benchmark. The isolating on/off pair stays: a clean null is a finding and the design is not tuned toward a positive.

## Compute budget (WP1 timings)

Release build, one fold, seed 17, 1500 steps, lr 0.005, 4 shared vCPUs (noise about +-30%; load average about 5). ms/step: M001 `full` 24.2, `masked` 20.3, `no-semantic-slots` about 20 (84 measured under contention), `plain` 14.8; M002 `factorized-v2` 57.0, `-off` 52.8, `full` 15.1, `plain` 12.3; M003 `full` 21.8, `latent-0` 14.5, `latent-1` 16.3, `latent-linear` 14.7, `plain` 12.1.

| Stage | M001-v6 | M002-v11 | M003-v3 |
|---|---|---|---|
| Per fold and seed (training) | about 1.5 min | about 3.5 min | about 1.6 min |
| Evaluation (5 seeds x 3 folds) | about 25 min | about 53 min | about 24 min |
| Sweep (4 lr x 3 folds, seed 17) | about 24 min | about 41 min | about 24 min |

About 3.2 hours in total, serial, on this class of host; roughly 30% less unloaded. Use `--release`; debug builds take about 1.5 s/step. There are no reruns, so a hardware failure during a confirmatory seed makes that study INCONCLUSIVE-INTERRUPTED (interruption rule above); a successor study would be needed if the question is still wanted. A hardware failure during a sweep is handled by a successor sweep before the confirmatory freeze (`DEVIATIONS.md`).

## Threats to validity

- **Temperature cap.** Fitted values saturate at 4.0 for overconfident arms (diagnostic only; reported).
- **Plain overconfidence.** At some learning rates plain is badly overconfident on raw logits (for example, raw `test_ood` NLL 10.4 and ECE15 0.63 at lr 0.005, fitted T at the cap). Against such a plain, an A0 arm passes the raw NLL/ECE gates easily; the bundle contrast's calibration gates are lenient by construction, and the fitted diagnostics are the fairer calibration comparison. The own-lr sweep for plain reduces but cannot remove this; a bundle SUPPORTS must be read with the diagnostics beside it.
- **FLOP incomparability.** The A0 and plain FLOP helpers differ by 7.9%; parameters (within 1%) are the plain match, and FLOPs are descriptive across the A0/plain boundary.
- **Compute confound in M003 primary.** `latent-0` does less work than `full`; a gain from k=2 is recurrence plus its compute. The nonlinearity contrast (matched parameters and FLOPs) is the cleaner one.
- **Five seeds, folds not replicates.** Intervals are wide; CI lower > 0 is demanding. A true +0.02 effect would often be INCONCLUSIVE rather than SUPPORTS; that is the intended asymmetry (no claim without evidence), not a power-optimised design.
- **Conjunction of nine gates** makes SUPPORTS conservative; INCONCLUSIVE does not mean "no effect".
- **Learnability floor 0.30** is an a-priori, outcome-blind choice; if it is too high FALSIFIES can only become rarer (INCONCLUSIVE), never wrongly issued.
- **Sweep at one seed with salted init**, edge selections are flagged in the lr file, not re-swept.
- **Synthetic benchmark.** The routing task is generated; the result does not transfer to text.
- **Mask share** is descriptive; it decomposes the M001 effect into index tagging versus typed content and carries no verdict.

## Artifacts and procedure

1. Sweep each study fresh (`--phase baseline-v1-sweep`, seed 17; the v1 sweeps and lr files are not used) and select (`--phase baseline-v1-select`): one lr file per study, `lr_<id>.tsv` in this directory, frozen before evaluation.
2. Evaluate each study for the five seeds (`--phase baseline-v1`), one hardened run record per seed.
3. Decide: `python3 scripts/aggregate_a0_baseline_v2.py --study M002-v11 --lr-file <lr file> --criteria-sha256 <frozen digest> --paired-ci <results/paired-ci-95.json> --decision <DECISION.toml> <five run records>` (`--repo` defaults to this repository). Exit 0 with a decision (any verdict, `INCONCLUSIVE-INTERRUPTED` included), exit 2 with a refusal.
4. Report every verdict, nulls and INCONCLUSIVE ones included.
