# A0 mechanism ablation study v1: preregistration

**Frozen at the git tag `a0-ablation-prereg-v1`.** Nothing in this file, in `criteria.toml`, in `DESIGN.md`, or in the code the study runs may change after that tag: gate G4 checks that the evaluation commit differs from the tag only in the learning-rate selection and the sweep records, and `scripts/aggregate_a0_ablation.py` refuses to run if `criteria.toml` differs from the tagged file. No declared seed (17, 29, 43, 71, 101) has been run.

## What this is

A preregistered study of A0's own mechanisms (typed slots, the typed attention bias, latent recurrence, the operator router) on the synthetic benchmark operator-routing v1, run through `scripts/run_experiment.py` under the M001-M004 manifests with separate `a0_*` entrypoints. It is **not** those manifests' baseline comparisons: they need the plain model baseline (owner decision O2), their `status` stays `planned`, and `metrics.json` stays reserved for them. `DESIGN.md` is the full design as the judge-panel workflow synthesized it; this file binds it to what was measured before the freeze and records every deviation from it. Where the two differ, this file and `criteria.toml` govern.

## What it cannot show

Anything about language-model quality or reasoning over real text (the task is synthetic and A0 has no LM head); M001's claim against a matched plain backbone; M002's hard-violation and constraint-retention claims; M003's claim about saving reasoning tokens (A0 generates none); M004's comparison with LLM-only reasoning; anything about a verifier head, an action head, `UncertaintyKind`, learned slot encodings, the raw->slot attention direction (it never reaches the output), runtime integration, scaling or latency. A clean positive result would show only that a mechanism helps this A0 on operator-routing v1. The complete list is in `DESIGN.md`.

## Frozen configuration

- **Model:** `PtrA0Config::new(216, 48).with_provenance_buckets(8)`, one block, codebook V1, `SlotEncoding` V1 at width 48, the full arm at `latent_steps = 2`. The width is 48, not the design's 32: calibration applied revision-ladder rung R1 (below).
- **Training:** Adam defaults, batch 128 without replacement, 1500 steps (S*), 100-step linear warm-up then cosine decay to 0.1x the peak, epoch order a Fisher-Yates shuffle from `splitmix64(order_seed ^ epoch)`, no early stopping; validation on all of val every S*/10 steps; the final parameters are scored on every test split.
- **Seeds:** declared [17, 29, 43, 71, 101]; `init_seed = seed`, `order_seed = seed ^ 0x0BA7C4`; `device.seed(init_seed)` immediately before each arm's init, so every arm of a seed starts from identical weights (tests T6) and sees the same batch order.
- **Learning rate:** swept after the freeze on grid [0.002, 0.005, 0.0125] at runner seed 17 with salted seeds (`init = 17 ^ 0x5EE9`), 1500 steps, val only; each arm takes the smallest rate within 0.005 of its best; a NaN loss makes a rate ineligible; an edge pick is flagged, not re-swept.
- **Arms that run** (the budget rule dropped tier 2):
  - M001: full, no-semantic-slots, no-semantic-slots-masked
  - M002: no-typed-attention, raw-blind
  - M003: latent-0, latent-1, latent-linear
  - M004: frozen-router
  - Not run (budget rule): `latent-4`, `blind-query-k0`, `blind-query-k0-no-typed-attention`. The M002 sufficiency contrast and the M003 dose response are therefore recorded as *not run (budget rule)*.

## Calibration (outcome-blind: full arm, seed 0, train and val only)

Target: final val accuracy >= 0.88. Candidates in order, each its own run at lr 0.005; the revision ladder written in the design was applied when the base width missed the target.

| Rung | d_model | Steps | Final val accuracy | ms/step | Val curve (10 checkpoints) |
|---|---:|---:|---:|---:|---|
| base | 32 | 1500 | 0.8615 | 13.4 | 0.701, 0.802, 0.832, 0.840, 0.849, 0.852, 0.855, 0.860, 0.869, 0.862 |
| base | 32 | 2000 | 0.8705 | 13.1 | 0.753, 0.816, 0.844, 0.842, 0.867, 0.865, 0.864, 0.867, 0.869, 0.871 |
| base | 32 | 3000 | 0.8725 | 13.4 | 0.802, 0.844, 0.852, 0.867, 0.864, 0.873, 0.873, 0.866, 0.874, 0.873 |
| R1 | 48 | 1500 | 0.8960 | 27.7 | 0.772, 0.852, 0.844, 0.880, 0.887, 0.874, 0.885, 0.893, 0.894, 0.896 |

Chosen: rung R1, d_model 48, S* = 1500.

## Budget rule

From the largest calibration ms/step (27.7): ladder steps applied ['drop latent-4', 'drop the blind-query pair']; projected wall time 18.5 minutes on 3 workers (limit 20, hard cap 45); sweep on at 1500 steps.

## Frozen data

Operator-routing v1, generator version 1, benchmark seed 20260925, codebook V1 (`2b6f8175a7a7bb648910e7acb17dcfcff5149834f2fe87355c5d2c4524c113a3`). Data FNV-1a-64 over the eight TSV files: `0ad71688f09b0d0d`. Full digests: `benchmarks/operator-routing/splits.lock.json`.

| Split | n | Label FNV-1a-64 | TSV SHA-256 |
|---|---:|---|---|
| train | 16000 | `89b416af24b05a74` | `bfbd8a956910302f3072b3117ee8cab2e2a69b09594a72733bc7a2b397e03bf1` |
| val | 2000 | `f5a682a6e264f087` | `ebfc79b10cc9387f1424f51da1c7ddf156030a274614dbbef79c493d5b731ef7` |
| test_iid | 3000 | `8aa96376d8db568e` | `e26a7ea299ec0232925c2a6044c7dc021f0ef89d3dffc3e37cf4e5f99eb3682b` |
| ood_compose_epi | 3000 | `6a1f66bea83db2b3` | `3b8c5f1edf4d4a3ee4c6131dfcc8cc8a3d9386479803e780aa4d0463863ab9bd` |
| ood_compose_regime | 3000 | `aa3b11bf19ad056c` | `ae32c8e9ef583f66e203f64ef4770c4607cf612d78abcb7fee61402e933be8e5` |
| ood_distractors | 3000 | `509c7f3bc62c98db` | `fa3ecb7cf13332f9635e166adf3a3f9761874d0b9f90d489efbc1f13caf785bd` |
| ood_validity | 3000 | `5ada2d0231cd1a35` | `2f118e72d9f11ba68a64caa20fe8f83abe75c3dc9e69d5c21518c436f3b13835` |
| ood_payload | 3000 | `cbcdf0c3584760ad` | `65c809128904140de9e8177931b473b053019f1cc0f057d4e519d299980d5a3f` |

## Reference lines (recomputed on the frozen splits; none needs a model)

| Reference | test_iid | ood_compose_epi | ood_compose_regime | ood_distractors | ood_validity | ood_payload |
|---|---:|---:|---:|---:|---:|---:|
| train-majority | 0.1940 | 0.2393 | 0.0730 | 0.1870 | 0.1940 | 0.1973 |
| count router (naive Bayes) | 0.3440 | 0.3033 | 0.4080 | 0.3443 | 0.3347 | 0.3427 |
| unbound bag-of-attributes linear | 0.5997 | 0.5937 | 0.3420 | 0.5953 | 0.4840 | 0.5980 |
| bound additive linear | 0.6510 | 0.6383 | 0.3623 | 0.6530 | 0.6410 | 0.6553 |
| hand-primary | 0.6443 | 0.6657 | 0.5917 | 0.6460 | 0.6943 | 0.6727 |
| hand-weighted | 0.8063 | 0.8063 | 0.7763 | 0.7990 | 0.8507 | 0.8067 |
| validity-blind rule | 0.8110 | 0.8247 | 0.8263 | 0.7877 | 0.5737 | 0.7903 |
| nuisance-only (entity ids, fillers) | 0.1713 | 0.1867 | 0.0787 | 0.1510 | 0.1583 | 0.0557 |
| raw-blind exact Bayes (training posterior) | 0.8213 | 0.8520 | 0.4390 | 0.8203 | 0.8523 | 0.8283 |

No-transfer references. ood_compose_epi: ignore held-out facts 0.6500, treat them as Inferred 0.8387; on the heldout subset (1050 examples) 0.0000 and 0.6924. ood_compose_regime: ignore Evidence 0.4597, Evidence as tabular 0.4843, temporal 0.4550, textual 0.4660; Causal share 0.5283; strict transfer subset 1352 examples, on which every no-transfer reference scores 0.0000.

### Data bands (gate G0), all measured before the freeze

| Band | Measured | Bound | Pass |
|---|---:|---|---|
| share of the most frequent label on test_iid <= 0.25 | 0.1940 | <= 0.25 | yes |
| every operator >= 0.03 of train labels | 0.0515 | >= 0.03 | yes |
| validity-decisive rate on test_iid in [0.12, 0.40] | 0.1890 | >= 0.12 and <= 0.4 | yes |
| regime-decisive rate on test_iid in [0.12, 0.40] | 0.2033 | >= 0.12 and <= 0.4 | yes |
| budget-decisive rate on test_iid in [0.12, 0.40] | 0.2097 | >= 0.12 and <= 0.4 | yes |
| confidence-decisive rate on test_iid in [0.12, 0.40] | 0.2623 | >= 0.12 and <= 0.4 | yes |
| epistemic-decisive rate on test_iid in [0.12, 0.40] | 0.2633 | >= 0.12 and <= 0.4 | yes |
| heldout-decisive rate on ood_compose_epi >= 0.25 | 0.3500 | >= 0.25 | yes |
| regime-decisive rate on ood_compose_regime >= 0.40 | 0.5870 | >= 0.4 | yes |
| exact-tie rate on test_iid <= 0.02 | 0.0153 | <= 0.02 | yes |
| hand-weighted accuracy on test_iid in [0.74, 0.88] | 0.8063 | >= 0.74 and <= 0.88 | yes |
| bound additive accuracy on test_iid in [0.58, 0.72] | 0.6510 | >= 0.58 and <= 0.72 | yes |
| raw-blind exact Bayes accuracy on test_iid in [0.76, 0.86] | 0.8213 | >= 0.76 and <= 0.86 | yes |
| nuisance-only predictor (entity ids, fillers) on test_iid <= train-majority + 0.02 | 0.1713 | <= 0.214 | yes |

## Decision criteria

`criteria.toml` is the rule. In short: for each contrast, delta_s = comparator - ablated on its endpoint, paired on seed; SUPPORTS needs the mean >= delta_min, the 95% t-interval (t = 2.776) above zero and all five seeds positive, with both arms learnable and every gate passing; FALSIFIES needs the interval's upper bound below delta_min (HARMFUL below zero); anything else is INCONCLUSIVE with its reason. Gate G1 (competence) requires the full arm's 5-seed mean test_iid >= max(0.85, raw-blind ceiling 0.8213 + 0.03); if it fails, no mechanism verdict is issued and the result is filed as a negative capacity finding.

| Contrast | Comparator vs ablated | Endpoint | delta_min | Prior on record |
|---|---|---|---:|---|
| M001-primary (mechanism) | full vs no-semantic-slots-masked | composite:A | 0.02 | supports 0.35, falsifies 0.3, inconclusive 0.35 |
| M001-secondary (mechanism) | full vs no-semantic-slots | composite:A | 0.02 | supports 0.45 |
| M002-raw-blind-check (manipulation-check) | full vs raw-blind | split:test_iid | 0.03 | — |
| M002-necessity (mechanism) | full vs no-typed-attention | composite:B | 0.02 | falsifies 0.55 |
| M002-sufficiency (mechanism) | blind-query-k0 vs blind-query-k0-no-typed-attention | subset:test_iid:regime | 0.03 | supports 0.4 |
| M003-nonlinearity (mechanism-pair) | full vs latent-0 and latent-linear | split:test_iid | 0.02 | supports 0.65 |
| M003-depth (mechanism) | full vs latent-1 | composite:C | 0.02 | supports 0.2 |
| M004-negative-control (negative-control) | full vs frozen-router | split:test_iid | 0.02 | equivalent 0.7 |
| no-verifier-head | — | — | — | not tested: no such module in PtrA0 |

## Deviations from DESIGN.md, and disclosures

- **Width.** Calibration at the design's d_model 32 reached 0.8725 at best, below the 0.88 target, so the pre-written ladder applied R1: d_model 48. The design's parameter counts (17,612 in total, 13,292 receiving gradient) are for 32; every run records its own total and effective (non-zero gradient in the first 10 batches) counts at 48.
- **Tier 2 not run.** The budget rule, applied from calibration timing alone, dropped `latent-4` and the blind-query pair.
- **An informal run before calibration.** While building the study binary, one 200-step run of the full arm at d_model 32, seed 0 (the calibration seed, not a declared one), lr 0.005, train and val only, measured 17.4 ms/step, 13,292 effective parameters and a final val accuracy of 0.742. It is disclosed here because calibration is the only model contact the design allows before the freeze; it saw nothing calibration does not.
- **The golden-logits test (T1) compares within 1e-5**, not bit for bit: the gemm kernel dispatches on CPU features and CI runners may lack this host's AVX-512. Bitwise reproduction on the recorded host is gate G2.
- **Order of implementation.** `with_typed_attention` and eager parameter initialization (`materialize`) predate the design (commit c9a13f7); the design reuses them, as it states.
- **Numbers.** The design's prose figures come from a prototype on Python's `random`; `references.json` holds the values on the frozen splits and governs (for example, Causal under the interventional regime in train is 0.090, not 0.096; the strict transfer subset is 45.1%, not 43%).
- **One factor per OOD split is approximate.** The compose splits' acceptance rules also shift the role and epistemic mix slightly (mean Live facts 4.85 against 4.68).
- **raw-blind on ood_compose_regime.** Its 'expected accuracy' under the training posterior is not a ceiling on that split, because the training posterior is miscalibrated there by design; only the test_iid ceiling enters a rule.
- **Implementation choices where the design was ambiguous** are listed in `benchmarks/operator-routing/README.md` (for example, a rejected example restarts on the same stream; the removal counterfactuals drop the removed fact's penalty too; the 1e-9 tolerance also applies to the 0.25 thresholds).
- **Calibration provenance.** The prefreeze phase ran from the clean commit c82efb8 (`scripts/run_a0_ablation_study.py prefreeze`): data regenerated and checked against the lock, references recomputed byte-identically, the study binary built, its self-test and the A0 test suite passed (`logs/prefreeze-cargo-test.log`).
- **The M001 pilot is superseded.** Rerun today at seed 17 it gives typed 1.0 and ablated 1.0 (delta 0), not the recorded 0.75: the recorded ablated arm could reach at most 0.25 by construction, and the model code has moved on. Its files are kept.

## Remaining order of commits

1. This commit, tagged `a0-ablation-prereg-v1`.
2. The sweep through the runner from this commit, then `select`, then the lr-selection commit (only `lr_selection.*` and the sweep records).
3. Evaluation, the rerun and any contingency runs, all from the lr-selection commit with a clean worktree (`scripts/run_a0_ablation_study.py eval`).
4. `scripts/run_a0_ablation_study.py aggregate`, then the results commit: run records, stock aggregates, `a0_internal_metrics.json` per experiment, `results.json`, and a `RESULTS.md` that reports every verdict, including nulls, HARMFUL and INCONCLUSIVE ones.
