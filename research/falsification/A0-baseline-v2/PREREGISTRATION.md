# A0 baseline study v2: preregistration

**Frozen text.** From the freeze commit on, nothing in this file, in `criteria.toml`, in `DESIGN.md`, in `scripts/aggregate_a0_baseline_v2.py`, in the v1 decider it imports (`scripts/aggregate_a0_baseline.py`), in the immutable core `scripts/aggregate_m002_v5.py`, in the dataset lock, or in the runner source tree may change. The experiments' `config.toml` files freeze the SHA-256 of `criteria.toml` and of the decision script; the decider refuses a `criteria.toml` whose digest differs from `--criteria-sha256` and refuses a core and a v1 decider whose digests differ from their pins. No evaluation process of this study has been run at the time of writing. Where this file and `DESIGN.md` differ, this file and `criteria.toml` govern.

## Relation to study v1 (replaced, no outcome observed)

This study is the complete successor of study v1 (M001-v5, M002-v10, M003-v2), which the owner replaced on 2026-10-09 for a **procedural defect of its decider**: the v1 decider demanded one `git_sha` in the five run records of a study, but the run protocol commits each record before the next run, so a correctly executed v1 study could only be refused. No v1 confirmatory outcome was observed by anyone (the v1 decider never ran on them; only record metadata was read); the v1 confirmatory records are unread history that must not be opened before the v2 decision. Sweeps and all confirmatory runs are repeated; the v1 sweeps and learning-rate files are not used. Protocol, arms, contrasts, gates, thresholds, seeds, learning-rate grid and selection rule are **identical** to v1. The only changes, all preregistered here:

- **(a)** the decider is `scripts/aggregate_a0_baseline_v2.py`, which reuses the pinned v1 decider and core by import;
- **(b) record identity:** the five records are one build iff every other identity field (`manifest_sha256`, `parameters`, `rustc`, `toolchain`, `environment`, `executable`, `host`, `cargo_lock_sha256`, `uv_lock_sha256`, the manifest's two preregistration digests) is identical across them and either the five `git_sha` are identical, or every `git_sha` is a commit of HEAD's history, each (ordered by `started_at`) an ancestor of the next or equal to it, and `git diff --name-only` between the earliest and the latest record commit lists only files directly inside the `results/` directories of this study's own experiments;
- **(c) interruption:** a confirmatory seed whose run record has status `started`, no stdout and no exit code makes that study **INCONCLUSIVE-INTERRUPTED** (named outcome, `go = false`, no quality claim, no statistic computed, completed seeds' rows not read, never rerun), instead of a refusal;
- **(d)** the ids are M001-v6, M002-v11, M003-v3 and the sweeps M001-v6-sweep, M002-v11-sweep, M003-v3-sweep. M004-v3 is not created; M004 stays deferred through M004-v2.

## What this is and what it cannot show

Three studies against the pinned plain baseline (`research/baselines/plain_model`, `ptr-burn-a0@matched-cross-attention-v1`, ADR-0021), on operator-routing v2 at A0 scale (one block, `d_model` 48, 1500 steps, CPU). It is **not** evidence about language models, about a pretrained LM baseline, about token savings, or about the operator router (M004-v2 is deferred, ADR-0021 item 2). A bundle-vs-baseline result is never mechanism evidence.

## Frozen configuration

- **Data:** `datasets/generated/operator_routing_v2`, lock SHA-256 `6a001b37e4adb81bae66f8abc50c9a461f07eea918c905179300cc8d353266a1`, folds `evidence-temporal=8c3bc2324e0eb770,evidence-tabular=9ff1e487932c4638,claim-interventional=fd11dbf9270b2f39` (this order). Splits `test_iid` and `test_ood` per fold.
- **Seeds:** 17, 29, 43, 71, 101 (`init_seed = seed`, `order_seed = seed XOR 0x0BA7C4`).
- **Model and training:** `d_model` 48, steps 1500, batch 128, Adam, warm-up 100 then cosine to 0.1x, calibrated cosine router in every A0 arm, rank 8, bias limit 1.0, metadata dropout 0.0. Label smoothing 0.05 and consistency weight 0.10 on the M002-v11 pair only; plain without label smoothing; no other regularisation differences.
- **Learning rates:** one val-only sweep per study at seed 17 over {0.0025, 0.005, 0.0125, 0.025} for every arm including plain; selection: highest mean validation accuracy over folds, tie lower mean validation NLL, tie smaller lr, a diverged grid point ineligible. One lr file per study; no hand merge.
- **Arms:** M001-v6: `full`, `no-semantic-slots-masked`, `no-semantic-slots` (diagnostic, mask share), `plain`. M002-v11: `factorized-v2`, `factorized-v2-off`, `full`, `plain`. M003-v3: `full`, `latent-0`, `latent-1` (descriptive), `latent-linear`, `plain`.

## Hypotheses, one verdict each

For every contrast: delta = treatment minus control on OOD accuracy (mean over folds), paired on seed.

| Contrast | Hypothesis (H1) | Null reading | Kind |
|---|---|---|---|
| **M001-v6-primary** `full` vs `no-semantic-slots-masked` | typed slot content improves OOD routing by >= 0.02 over index-tagged raw binding | the benefit of typed content is below 0.02 | isolating, **headline of M001-v6** |
| M001-v6-bundle `full` vs `plain` | the A0 bundle improves OOD routing by >= 0.02 over the plain baseline | the bundle benefit is below 0.02 | bundle vs baseline, non-isolating |
| **M002-v11-primary** `factorized-v2` vs `factorized-v2-off` | the factorized pair bias improves OOD routing by >= 0.02 over the identical computation with the bias zeroed | no benefit of 0.02 (a null is plausible: A0-v2 prior) | isolating, **headline of M002-v11** |
| M002-v11-bundle `full` vs `plain` | as M001-v6-bundle | as M001-v6-bundle | bundle, non-isolating |
| **M003-v3-primary** `full` vs `latent-0` | the tied latent recurrence (k=2) improves OOD routing by >= 0.02 over no recurrence at equal parameters | no benefit of 0.02 (compute differs by the recurrence) | isolating, **headline of M003-v3** |
| M003-v3-nonlinearity `full` vs `latent-linear` | the per-slot nonlinearity, not the extra affine parameters, carries the benefit | no benefit of 0.02 at matched parameters and FLOPs | isolating, secondary |
| M003-v3-bundle `full` vs `plain` | as M001-v6-bundle | as M001-v6-bundle | bundle, non-isolating |

## Endpoints

- **Primary (the only one):** OOD accuracy (`test_ood`), mean over the three folds inside each seed, paired over the five seeds; delta_min 0.02; 95% Student-t interval, 4 df (t = 2.7764 from the core).
- **Gated secondary, all per contrast (treatment minus control, raw logits, T = 1):** per-fold OOD `raw_nll` and `raw_ece15` (NLL upper CI <= 0; ECE mean <= 0 and upper CI <= 0.02); IID accuracy (mean over folds, CI lower >= -0.01); the treatment arm's raw coverage (IID >= 0.90, OOD >= 0.80, every seed x fold); mean OOD raw selective-error delta <= 0; positive mean OOD accuracy delta in every fold.
- **Compute match:** vs plain, `total_params` within 1%; A0-vs-A0 isolating pairs (`factorized-v2`/`-off`, `full`/`latent-linear`), equal `total_params` and equal `flops_per_example`; `full` vs `latent-0`, equal `total_params`. FLOP columns are otherwise descriptive.
- **Diagnostics, no gate:** temperature-fitted NLL/ECE15 and the fitted temperature (grid 0.25 to 4.0, cap 4.0, may saturate), latency, wall time, `latent_steps`, `explicit_token_count`, M001 mask share (`masked - unmasked` accuracy), per-arm learnability.

## Gates and decision rules

Gates, in order: `complete_finite_bound_evidence`, `ood_accuracy`, `positive_accuracy_each_fold`, `ood_nll_each_fold`, `ood_ece_each_fold`, `iid_accuracy`, `coverage`, `selective_error`, `matched_compute`. Exact comparisons on float64.

1. **SUPPORTS**: every gate holds.
2. **NO-GO-CALIBRATION** (INCONCLUSIVE with this label): `ood_accuracy` holds and `ood_nll_each_fold` or `ood_ece_each_fold` fails. No quality claim may follow.
3. **FALSIFIES**: `ood_accuracy` fails, the pooled OOD accuracy CI upper bound < 0.02, both arms have mean IID accuracy >= 0.30, and `matched_compute` holds; **HARMFUL** if the upper bound < 0.
4. **INCONCLUSIVE** otherwise, with every failing gate named.
5. A study's headline verdict is its primary contrast's. Bundle contrasts are decided and reported by the same rules, are labelled non-isolating, and cannot support a mechanism claim. No multiplicity correction.
6. **Interrupted, not refused:** a confirmatory seed with a `started` record (no stdout, no exit code) makes the study INCONCLUSIVE-INTERRUPTED. It is reported, forbids any quality claim, and is never rerun.
7. **Refusal, not verdict:** a missing seed, fold, arm or split, a non-finite value, an arm/protocol/lr/digest mismatch, mixed build identities (the record-identity rule above), a failed run (exit code not 0), a changed criteria, core or v1-decider digest.

## No reruns

Each seed of each study runs once, through the hardened runner's frozen entrypoint. A failed, refused or interrupted study is not rerun and not patched; a successor study with a new id is created. An interrupted sweep (record `started`, no output) is replaced by a successor sweep `<study>-sweep-r2` before the confirmatory freeze (`DEVIATIONS.md`). Records, sweeps and decisions are immutable; `paired-ci-95.json` and `DECISION.toml` are created exclusively and `DECISION.toml` binds the SHA-256 of the paired-CI file. All verdicts are reported, nulls and INCONCLUSIVE ones included.

## Registered experiments (at freeze, see open points of the registration work)

`M001-v6`, `M002-v11`, `M003-v3` (evaluation, five seeds each) and the three per-study sweep experiments `M001-v6-sweep`, `M002-v11-sweep`, `M003-v3-sweep`. M004-v3 is not created; `M004-v2` (registered by study v1) stays planned and not run at A0 (ADR-0021 item 2). Each experiment freezes its command, `[preregistration]` values, the SHA-256 of `criteria.toml`, of the decision script, of the v1 decider it imports and of the pinned core, and the implementation-tree digest of `model/burn-a0`.
