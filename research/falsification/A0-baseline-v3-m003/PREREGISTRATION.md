# A0 baseline study v3-m003: preregistration

**Frozen text.** From the freeze commit on, nothing in this file, in `criteria.toml`, in `DESIGN.md`, in `scripts/aggregate_a0_baseline_v3.py`, in the v2 decider it imports (`scripts/aggregate_a0_baseline_v2.py`), in the v1 decider that one imports (`scripts/aggregate_a0_baseline.py`), in the immutable core `scripts/aggregate_m002_v5.py`, in the dataset lock, in the learning-rate file, or in the runner source tree may change. The experiment's `config.toml` freezes the SHA-256 of `criteria.toml`, of the learning-rate file and of the three deciders; the decider refuses a `criteria.toml` whose digest differs from `--criteria-sha256` and refuses a v2 decider, a v1 decider and a core whose digests differ from their pins. No evaluation process of this study has been run at the time of writing. Where this file and `DESIGN.md` differ, this file and `criteria.toml` govern.

## Relation to M003-v3 (interrupted, no outcome)

This study is the successor of **M003-v3** of the A0 baseline study v2 (`research/falsification/A0-baseline-v2/`), created by owner decision of 2026-10-10 because the question (does the tied latent recurrence help out-of-distribution routing at A0?) is still wanted.

M003-v3 has **no outcome**. Its seed 71 was killed by a container restart (run record status `started`, no `stdout`, no exit code) and seed 101 was never started; by the interruption rule of v2 its outcome is INCONCLUSIVE-INTERRUPTED with `go = false`, no statistic was computed, and the study is never rerun, patched or completed (`research/falsification/A0-baseline-v2/DEVIATIONS.md`, D1; successor registration: D3). The interruption is a property of the host, not of any result. Its seeds 17, 29 and 43 completed and are committed; they are **recorded as unobserved**: the decider does not read them for an interrupted study, D1 records that their rows were not read, and nothing in this registration uses them. They are **not pooled with this study**, are not compared with it, and their `stdout` must not be opened before the decision of M003-v4. M003-v3 is marked `superseded`; its records and decision files stay as history.

Nothing about the question was observed before this registration, so this study was not shaped by an outcome. The question, arms, contrasts, protocol, seeds, learning-rate grid and selection rule, gates, thresholds and decision rules are those of v2 **value for value**: `criteria.toml` of this study differs from the v2 file only in the study name, the study and contrast ids (M003-v3 to M003-v4), the absence of the M001/M002 studies (decided in v2 and not repeated), the pin of the v2 decider, and the `[sweep_reuse]` table below. A test compares the two files value by value. The only change of the decider is the study name it accepts (below).

## What this is and what it cannot show

One study against the pinned plain baseline (`research/baselines/plain_model`, `ptr-burn-a0@matched-cross-attention-v1`, ADR-0021), on operator-routing v2 at A0 scale (one block, `d_model` 48, 1500 steps, CPU). It is **not** evidence about language models, about a pretrained LM baseline, about token savings (not tested: the study varies latent computation, not the number of tokens read), or about the operator router (M004-v2 is deferred, ADR-0021 item 2). A bundle-vs-baseline result is never mechanism evidence.

## Frozen configuration

- **Data:** `datasets/generated/operator_routing_v2`, lock SHA-256 `6a001b37e4adb81bae66f8abc50c9a461f07eea918c905179300cc8d353266a1`, folds `evidence-temporal=8c3bc2324e0eb770,evidence-tabular=9ff1e487932c4638,claim-interventional=fd11dbf9270b2f39` (this order). Splits `test_iid` and `test_ood` per fold.
- **Seeds:** 17, 29, 43, 71, 101 (`init_seed = seed`, `order_seed = seed XOR 0x0BA7C4`). They are new experiment runs; nothing recorded by M003-v3 is read. The runs are deterministic on CPU, so seeds 17, 29 and 43 are expected to recompute bit for bit what M003-v3 computed for them; that is harmless for the decision because M003-v3's rows were never read (D1), and it is stated here so the equal numbers are not mistaken for independent replication of those three seeds.
- **Model and training:** `d_model` 48, steps 1500, batch 128, Adam, warm-up 100 then cosine to 0.1x, calibrated cosine router in every A0 arm, rank 8, bias limit 1.0, metadata dropout 0.0; plain without label smoothing; no other regularisation differences.
- **Arms:** `full` (latent steps 2), `latent-0` (no recurrence), `latent-1` (one step, descriptive), `latent-linear` (two steps, affine only), `plain`.
- **Learning rates (reuse of a completed val-only sweep).** The sweep is validation-only (test splits never read), at seed 17 over {0.0025, 0.005, 0.0125, 0.025} for every arm including plain, with the selection rule of v2: highest mean validation accuracy over folds, tie lower mean validation NLL, tie smaller lr, a diverged grid point ineligible. `M003-v3-sweep` ran it under the v2 criteria, completed with exit 0 and was committed together with its selection (`research/falsification/A0-baseline-v2/lr_M003-v3.tsv`) before any confirmatory seed of M003-v3 ran. Because the v2 and v3 criteria are identical in every value the sweep depends on (protocol, grid, arms, seed, rule, data), M003-v4 **registers no sweep of its own** and takes that selection: `lr_M003-v4.tsv` is a byte-for-byte copy of the v2 file (the gate compares them), and `config.toml` names `M003-v3-sweep` as its `sweep_experiment`. No selection depends on any confirmatory result of any study, so reuse can neither inform nor bias the confirmatory comparison; a fresh sweep would repeat the same computation on the same validation data and add a second chance for a host interruption before the confirmatory freeze. As in v2, the selection puts `full`, `latent-0`, `latent-1` and `latent-linear` at the upper grid edge (0.025) and `plain` at the lower edge (0.0025); neither is bracketed and no re-sweep follows (recorded as a limit, not changed).

## Hypotheses, one verdict each

For every contrast: delta = treatment minus control on OOD accuracy (mean over folds), paired on seed.

| Contrast | Hypothesis (H1) | Null reading | Kind |
|---|---|---|---|
| **M003-v4-primary** `full` vs `latent-0` | the tied latent recurrence (k=2) improves OOD routing by >= 0.02 over no recurrence at equal parameters | no benefit of 0.02 (compute differs by the recurrence itself) | isolating, **headline** |
| M003-v4-nonlinearity `full` vs `latent-linear` | the per-slot nonlinearity, not the extra affine parameters, carries the benefit | no benefit of 0.02 at matched parameters and FLOPs | isolating, secondary |
| M003-v4-bundle `full` vs `plain` | the A0 bundle improves OOD routing by >= 0.02 over the plain baseline | the bundle benefit is below 0.02 | bundle vs baseline, non-isolating |

`latent-1` has no hypothesis and no verdict.

## Endpoints

- **Primary (the only one):** OOD accuracy (`test_ood`), mean over the three folds inside each seed, paired over the five seeds; delta_min 0.02; 95% Student-t interval, 4 df (t = 2.7764 from the core).
- **Gated secondary, all per contrast (treatment minus control, raw logits, T = 1):** per-fold OOD `raw_nll` and `raw_ece15` (NLL upper CI <= 0; ECE mean <= 0 and upper CI <= 0.02); IID accuracy (mean over folds, CI lower >= -0.01); the treatment arm's raw coverage (IID >= 0.90, OOD >= 0.80, every seed x fold); mean OOD raw selective-error delta <= 0; positive mean OOD accuracy delta in every fold.
- **Compute match:** vs plain, `total_params` within 1%; `full` vs `latent-linear`, equal `total_params` and equal `flops_per_example`; `full` vs `latent-0`, equal `total_params` (FLOPs differ by the recurrence itself and are descriptive). FLOP columns are otherwise descriptive.
- **Diagnostics, no gate:** temperature-fitted NLL/ECE15 and the fitted temperature (grid 0.25 to 4.0, cap 4.0, may saturate), latency, wall time, `latent_steps`, `explicit_token_count`, per-arm learnability.

## Gates and decision rules

Gates, in order: `complete_finite_bound_evidence`, `ood_accuracy`, `positive_accuracy_each_fold`, `ood_nll_each_fold`, `ood_ece_each_fold`, `iid_accuracy`, `coverage`, `selective_error`, `matched_compute`. Exact comparisons on float64.

1. **SUPPORTS**: every gate holds.
2. **NO-GO-CALIBRATION** (INCONCLUSIVE with this label): `ood_accuracy` holds and `ood_nll_each_fold` or `ood_ece_each_fold` fails. No quality claim may follow.
3. **FALSIFIES**: `ood_accuracy` fails, the pooled OOD accuracy CI upper bound < 0.02, both arms have mean IID accuracy >= 0.30, and `matched_compute` holds; **HARMFUL** if the upper bound < 0.
4. **INCONCLUSIVE** otherwise, with every failing gate named.
5. The headline verdict is the primary contrast's. The nonlinearity and bundle contrasts are decided and reported by the same rules; the bundle is labelled non-isolating and cannot support a mechanism claim. No multiplicity correction.
6. **Record identity:** the five records are one build iff every identity field other than `git_sha` is identical and either the five `git_sha` are identical or each (ordered by `started_at`) is an ancestor of the next and `git diff --name-only` between the earliest and latest lists only files directly inside `results/` of this study's experiment (v2 rule, unchanged).
7. **Interrupted, not refused:** a confirmatory seed with a `started` record (no stdout, no exit code) makes the study INCONCLUSIVE-INTERRUPTED. It is reported, forbids any quality claim, and is never rerun. A second interruption would again be an interruption and not an outcome; whether a further successor is created is an owner decision.
8. **Refusal, not verdict:** a missing seed, fold, arm or split, a non-finite value, an arm/protocol/lr/digest mismatch, mixed build identities, a failed run (exit code not 0), a changed criteria, core, v1 or v2 decider digest.

## The decider

`scripts/aggregate_a0_baseline_v3.py` imports the v2 decider unchanged (pinned by `decision.parent_sha256` and by the experiment's `decision_base`), rebinds only the criteria name it accepts (`A0-baseline-v3-m003`), its default criteria path, its criteria loader (which adds the pin of the v2 file) and the script digest it reports, and delegates every statistic, gate, rule and output to v2 and through it to v1 and the core. `DECISION.toml` and `paired-ci-95.json` therefore carry this file's digest as `decision_script_sha256`, the study name `A0-baseline-v3-m003` and the experiment id `M003-v4`.

## No reruns

Each seed runs once, through the hardened runner's frozen entrypoint. A failed, refused or interrupted study is not rerun and not patched; a successor study with a new id would be created. Records, sweeps and decisions are immutable; `paired-ci-95.json` and `DECISION.toml` are created exclusively and `DECISION.toml` binds the SHA-256 of the paired-CI file. All verdicts are reported, nulls and INCONCLUSIVE ones included.

## Registered experiment

`M003-v4` (evaluation, five seeds). It has no sweep of its own: `M003-v3-sweep` (study v2, unchanged, not superseded) is its `sweep_experiment`. M003-v3 is superseded; M004 stays deferred through M004-v2 (M004-v3 is not created). The experiment freezes its command, `[preregistration]` values, the SHA-256 of `criteria.toml`, of the learning-rate file, of this file, of the three deciders and of the core, and the implementation-tree digest of `model/burn-a0`.
