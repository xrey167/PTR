# A0 baseline study v2: results

Study `A0-baseline-v2` (`PREREGISTRATION.md`, `criteria.toml`, `DESIGN.md`, deviations in `DEVIATIONS.md`). Decider: `scripts/aggregate_a0_baseline_v2.py` (exit 0 for all three studies). Scope: A0 on the synthetic operator-routing v2 benchmark only (ADR-0021); no language-model claim. All outputs are in each experiment's `results/` (`paired-ci-95.json`, `DECISION.toml`; `run.json` and `metrics.json` for M001-v6 and M002-v11). Decider logs: not committed, reproducible with the command in `DESIGN.md`.

**Summary: no study produced a quality claim.** M001-v6 INCONCLUSIVE (coverage gate), M002-v11 INCONCLUSIVE (primary contrast is a null that narrowly missed FALSIFIES), M003-v3 INCONCLUSIVE-INTERRUPTED (no outcome). No NO-GO-CALIBRATION was issued: the calibration gates held wherever the accuracy gate held. `go = false` for all three.

| Study | Headline contrast (isolating) | Verdict / label | Failing gates |
|---|---|---|---|
| M001-v6 | `full` vs `no-semantic-slots-masked` | INCONCLUSIVE | `coverage` |
| M002-v11 | `factorized-v2` vs `factorized-v2-off` | INCONCLUSIVE | `ood_accuracy`, `positive_accuracy_each_fold`, `ood_nll_each_fold`, `ood_ece_each_fold`, `iid_accuracy`, `coverage` |
| M003-v3 | `full` vs `latent-0` | INCONCLUSIVE-INTERRUPTED | none computed |

All numbers below are paired differences, treatment minus control, mean over the five seeds (17, 29, 43, 71, 101) with the 95% t-interval (t = 2.776), pooled over the three folds, on `test_ood` unless stated. NLL and ECE are raw logits (lower is better); a negative delta is an improvement.

## M001-v6 (semantic slots): INCONCLUSIVE

Records: five, all exit 0, seeds 17 to 101.

| Contrast | Role | OOD accuracy delta [95% CI] | Raw OOD NLL delta | Raw OOD ECE15 delta | IID accuracy delta [CI] |
|---|---|---|---|---|---|
| `full` vs `no-semantic-slots-masked` | primary, isolating | +0.2424 [+0.1752, +0.3097] | -0.989 [-1.299, -0.679] | -0.138 [-0.178, -0.099] | +0.3130 [+0.2838, +0.3422] |
| `full` vs `plain` | bundle, non-isolating | +0.2513 [+0.2231, +0.2795] | -3.195 [-3.537, -2.854] | -0.314 [-0.336, -0.292] | +0.3044 [+0.2847, +0.3242] |

Arm means over seeds and folds (OOD / IID accuracy): `full` 0.499 / 0.688, `no-semantic-slots-masked` 0.256 / 0.375, `no-semantic-slots` 0.227 / 0.330, `plain` 0.247 / 0.384. Parameters: 33,324 for the A0 arms, 33,323 for plain.

Reading: every accuracy and calibration gate held for both contrasts (accuracy positive in all five seeds and every fold, NLL and ECE gates met per fold). Both contrasts failed the single `coverage` gate (the treatment arm's raw selective-prediction coverage must be at least 0.90 IID and 0.80 OOD in every seed and fold). The per-cell coverage values are not in the committed output; the paired coverage deltas for the primary are OOD -0.042 [-0.081, -0.002] pooled, with the two evidence folds worse than the control (-0.088, -0.085) and `claim-interventional` better (+0.049). Because one gate failed, the preregistered rule gives INCONCLUSIVE, not SUPPORTS, and forbids a quality claim, even though the accuracy effect is large and unambiguous in sign. The accuracy gain is a measured fact of these runs; it is not a SUPPORTS verdict.

Caveats stated by the design apply: the learning-rate selection put `full` at the upper grid edge (0.025) and `plain` at the lower edge (0.0025), so neither rate is bracketed; fitted-temperature diagnostics are less favourable than the raw calibration gates (primary diagnostic fitted ECE15 delta +0.052 [+0.016, +0.088], fitted NLL delta -0.469 [-0.718, -0.220]; diagnostics are descriptive, not gates); plain is much worse calibrated on raw logits, so the bundle's raw calibration gates are lenient by construction. The mask share (masked minus unmasked accuracy) is +0.030 OOD [+0.005, +0.054], descriptive only, no verdict.

## M002-v11 (factorized typed pair attention): INCONCLUSIVE

Records: five, all exit 0.

| Contrast | Role | OOD accuracy delta [95% CI] | Raw OOD NLL delta | Raw OOD ECE15 delta | IID accuracy delta [CI] |
|---|---|---|---|---|---|
| `factorized-v2` vs `factorized-v2-off` (pair bias times zero) | primary, isolating | +0.0027 [-0.0147, +0.0202] | -0.0003 [-0.042, +0.042] | -0.0069 [-0.010, -0.003] | -0.0010 [-0.0137, +0.0116] |
| `full` vs `plain` | bundle, non-isolating | +0.2513 [+0.2231, +0.2795] | -3.195 [-3.537, -2.854] | -0.314 [-0.336, -0.292] | +0.3044 [+0.2847, +0.3242] |

Arm means (OOD / IID accuracy): `factorized-v2` 0.506 / 0.697, `factorized-v2-off` 0.503 / 0.698, `full` 0.499 / 0.688, `plain` 0.247 / 0.384; equal parameters and FLOPs for the isolating pair.

Reading: the isolating primary is a null. The point estimate is +0.003 and the interval contains zero, so no accuracy gain from the typed pair bias is shown, and six gates fail (the accuracy and per-fold positivity gates, the per-fold NLL and ECE gates, the IID gate and `coverage`). The verdict is INCONCLUSIVE, not FALSIFIES: FALSIFIES needs the interval's upper bound below 0.02, and it is 0.02017, above the threshold by about 0.0002. The result is therefore consistent with no effect and with an effect up to about +0.02; it is not evidence of an effect. The bundle contrast is the same `full` versus `plain` comparison as in M001-v6 (same records of the `full` and `plain` arms by construction of the shared arm definitions), it is non-isolating (typed slots, typed attention, latent recurrence and the calibrated router all differ from plain), it fails only `coverage`, and it cannot be read as evidence for typed attention. The prior stated in `DESIGN.md` (a null is plausible) is not contradicted.

## M003-v3 (latent recurrence): INCONCLUSIVE-INTERRUPTED

Records: seeds 17, 29, 43 completed (exit 0); seed 71 was interrupted by a container restart (record status `started`, no exit code, no output); seed 101 was never started (D1 in `DEVIATIONS.md`). By the preregistered interruption rule the outcome is INCONCLUSIVE-INTERRUPTED for all three contrasts (primary `full` vs `latent-0`, nonlinearity `full` vs `latent-linear`, bundle `full` vs `plain`): no statistic was computed, the completed seeds were not read by the decider, and `paired-ci-95.json` carries status `not-computed-interrupted`. `DECISION.toml` records seeds completed [17, 29, 43], interrupted [71], not run [101], `go = false`, `rerun_allowed = false`. There is no result and no verdict on latent recurrence in this study. The study is not rerun and not completed; the experiment stays `prepared` until a successor study is registered if the question is still wanted. Its published results were not written (`--publish-results` is refused for an interrupted study), so there is no `run.json` or `metrics.json`.

## Limits

- Claims are A0, synthetic, five seeds with folds as non-replicates; intervals are wide and the conjunction of nine gates makes SUPPORTS conservative. INCONCLUSIVE does not mean no effect.
- Bundle contrasts (`full` vs `plain`) are non-isolating. They are reported for completeness and are not mechanism evidence.
- Learning-rate selections sit on grid edges for `full` and `plain` (see above); they were not re-swept.
- The runner label in the `lr_*.tsv` header comments carries the earlier study's name (`study=m001-v5` and so on) because the runner is unchanged from v1; the files are the v2 sweeps' selections.
- M004 stays deferred (M004-v2); nothing here touches it. No v1 confirmatory record was read or quoted.
