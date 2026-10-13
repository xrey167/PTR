# F003 preregistration

Status: preregistration text of F003, frozen with the experiment's `[preregistration]` table (`config.toml` names this file and its SHA-256). Nothing here is evidence; the runs are. The harness is `ptr-bench calibrated-arbiter <seed>` (`bins/ptr-bench/src/experiments/f003/`), the decision rule is [criteria.toml](criteria.toml) applied by [aggregate.py](aggregate.py), the adjudicator is described in [PROTOCOL.md](PROTOCOL.md), and the mutation plan is [tests/mutations.toml](tests/mutations.toml).

Entrypoint (set in `experiment.toml` at the freeze): `cargo run --release --locked -p ptr-bench -- calibrated-arbiter <seed>`.

## 1. Claim and scope

**The adjudicator is a simulation.** No person adjudicates any branch in this experiment: a deterministic program with a known error rate stands in for people (PROTOCOL.md). The claim is therefore scoped to the mechanism:

1. the Learn-then-Test guarantee of `ptr_branch::certify_threshold` holds: over the draw of the calibration set, the true adjudicated harm rate of the chosen threshold exceeds alpha with probability at most delta (Monte-Carlo coverage, section 5);
2. a policy recorded with it, tested on the adjudications recorded after its calibration set, does not have a harm rate of its auto-proposals above alpha (the exact one-sided test, section 5);
3. no ineligible branch is ever auto-proposed, whatever its score (INVARIANT 11: a learned score cannot override a deterministic verifier contradiction), and only calibration-slice triages can be adjudicated;
4. every decision, threshold, held-out set and Clopper-Pearson bound the product produces equals an independent oracle's;
5. off-policy estimates (IPS, doubly robust) of a target policy the log supports are unbiased, and a target the log cannot support is refused.

**Regime.** The adjudicator errs on 2 per mille of branches. This is a low-noise regime, chosen because Learn-then-Test's fixed sequence starts at the first threshold with at least 59 admitted samples and stops at the first failure: at 20 per mille the policy certifies nothing in roughly three calibrations of four (a utility limit of the start rule, not a violation of the bound). That is measured, not assumed: every seed reports an ungated Monte-Carlo arm at 20 per mille (`descriptive.noisy_adjudicator_arm`: certified and never counts and violations, per policy size). Reviews saved are descriptive only (`ltt_reviews_saved_permille`, one gated permille, is a vacuity floor); F003 makes no claim that reviews are reduced in a deployment. The start rule is data-dependent (it reads the scores, not the labels); that the coverage guarantee holds with it is shown empirically for this scenario by the Monte-Carlo count, not proven here. The tests have a detection range: the coverage test (one-sided, level 1e-3 per seed, not Bonferroni-split over the seeds) rejects a violation rate above about 6.5 percent and the held-out test an observed harm rate above about 7 percent; smaller deviations are seen only by the exact oracle comparisons. `calibration_draw` is the product's and is also the draw the oracle's expected decision uses, so the slice itself is not derived independently: its uniformity is judged by the slice-rate z tests.

Not claimed: how well people judge harm; anything about the real scorer; that reviews are saved in a real deployment (the simulation reports the share of eligible branches auto-proposed, descriptively).

## 2. Hypothesis and baselines (experiment.toml)

Does the calibrated arbiter keep the harm rate of auto-proposed branches at or below alpha on held-out adjudications? (Reviews saved are reported descriptively, not gated.) Baselines, run through the same product triage on the same stream: **escalate every branch** (threshold `Never`) and a **fixed uncalibrated threshold** (0.5, `uncalibrated_threshold_permille = 500`).

## 3. Owner decisions (2026-10-09, signed as proposed; scratchpad/finish-decisions.md)

- `calibration_set_rule`: every adjudication recorded before the policy version is calibration; every one after it is held out.
- `adjudicators_see_intent = false`: the intent is free text the agent wrote and the label calibrates that agent's own threshold.
- `harm_breakdown_namespaces`: the first `:`-separated segment of each changed key; descriptive only.
- `adjudication_protocol`: `experiments/feedback/F003-calibrated-arbiter/PROTOCOL.md`; `adjudication_protocol_sha256` is its digest, pinned at the freeze (`must-be-pinned-before-prepared` until then).
- The falsification rule is the statistically correct one (section 5), not the manifest's literal sentence; `experiment.toml` was reworded to it before the freeze.
- The adjudicator is a simulated deterministic oracle with a known error rate, labelled as such.

## 4. The simulated world

Per seed, a time-ordered stream of **30,000 branches** (`stream_branches`). Each branch has a verification report, a scorer score, a true harm, an agent intent (never shown to the adjudicator) and one to three changed keys `<namespace>:<id>`.

**Verification kinds** (share in permille, `kind_weights_permille`): eligible 750; fail 60 (Fail at Deterministic, a hard finding); disputed 40 (Disputed at FullSemantic); unknown 40 (Unknown at Deterministic); latent_agreement 40 (Pass at LatentAgreement); sample_verified 30 (Pass at SampleVerified); hard_finding 40 (Pass at Deterministic with a hard finding: a forged-looking pass). An eligible branch is Pass at Deterministic or FullSemantic with no hard finding (some carry a soft finding). **Ineligible branches carry adversarially high scores** (0.90 to 1.00), so every threshold a policy holds would admit them were verification not consulted first.

**Namespaces** (`note`, `task`, `cfg`; branch share 500/300/200 permille): the scorer's score is the maximum of 4/3/3 uniform draws on 0..1,000,000 micro-units, so its mass in cell `c = m / 1000` is exactly `((c+1)/1000)^k - (c/1000)^k`; 30 percent of scores are then rounded down to a multiple of 0.001, so that a grid threshold sees many scores exactly equal to it (the cell, and so the model, is unchanged). The scorer is **overconfident**: the true harm probability of a branch in cell `c` is a table, linear in `c` between anchors `(cell, ppm)`:

| namespace | anchors (cell: harm ppm) |
|---|---|
| note | 0: 650000, 500: 300000, 700: 110000, 850: 12000, 1000: 300 |
| task | 0: 750000, 500: 350000, 700: 130000, 850: 18000, 1000: 500 |
| cfg | 0: 850000, 500: 400000, 700: 160000, 850: 28000, 1000: 1000 |

The simulated adjudicator labels a slice branch `true harm XOR flip`, the flip at `adjudicator_error_permille = 2` per mille keyed by a hash of the seed and branch id. The adjudicated harm rate above a threshold is therefore `eps + (1 - 2 eps) * H(t)` where `H(t)` is the model's true harm rate among eligible branches the threshold admits, summed exactly over the cells by the harness (`Truth`). With these tables the exact adjudicated harm rate at the uncalibrated threshold 0.5 is 7.4 percent (above alpha = 5 percent, so the scenario has teeth), and it falls below 5 percent from about 0.62 (4.5 percent at 0.64, 3.3 percent at 0.70), which is where the certified thresholds are expected to sit.

**Why 2 per mille and not 20.** The plan sketched 20 per mille. Learn-then-Test starts its sequence at the first grid threshold admitting at least 59 calibration samples (the fewest at which a clean set can pass at alpha = delta = 0.05) and stops at the first failure, so it certifies something only if the top ~59 adjudications are labelled harmless; at an adjudicator error rate of 20 per mille that has probability about 0.3 and a throw-away simulation of this model certified nothing in 72 to 76 percent of calibrations at every calibration size from 300 to 2,000, whatever the size. At 2 per mille the same simulation certified nothing in 17 to 21 percent. The experiment would otherwise have no subject; the mechanism is tested at the error rate it can use.

**Pilot exposure (disclosed).** Before this preregistration was frozen, the harness was run once at the pinned scale on each of the five declared seeds (17, 29, 43, 71, 101) in one batch (13:20 to 13:21 on 2026-10-09, before this draft was written at 13:27) and several more times on seed 17, all with clean outcomes (no coverage prerequisite unmet, no falsifier tripped; Learn-then-Test certified a threshold in all 20 recorded policies; coverage violations 56 to 71 of 2,000 against a critical value of 131). The outputs are kept, with these SHA-256 digests:

| file | sha256 |
|---|---|
| `scratchpad/f003-preexposure/f003-run-17.json` | `6a12ea763e12adbffe7ac95bbacbf22f86a9092bf65a1f05b98ff295c6d04202` |
| `scratchpad/f003-preexposure/f003-run-29.json` | `6d806738d96863cac4a7407d1e3d23a2861ba5d03d62bcd01052d480a44355e2` |
| `scratchpad/f003-preexposure/f003-run-43.json` | `bad2e019f3eba28ef2eea94c5b3df5a6b1a561f6ac76d62ef519d0a6dcbab3ce` |
| `scratchpad/f003-preexposure/f003-run-71.json` | `9b4b3b593651936a931d1eac5f5147b771f9306452bdfd3be9914c241c7b4577` |
| `scratchpad/f003-preexposure/f003-run-101.json` | `4ebe3331ccf44d1ac69106223faca86e4ceb13c9ea7e0742dd3370f469f8cec0` |

The harness is deterministic, so the post-freeze runs of these seeds reproduce those results; they are a reproducibility and provenance check, not blind confirmation, and the verdict reads "reproduces the pilot results plus the mutation evidence", not "five blind seeds". At the freeze the five post-freeze outputs are diffed against the files above, ignoring `timing_ms`. The world (namespace tables, 30 percent tie share, adjudicator error 2 per mille, policy sizes, shares) and the coverage threshold `policies_certified >= 1` were chosen with throw-away Python simulations of the model and seed-17 runs, to give the uncalibrated threshold teeth and the policy something to certify; criteria.toml, config.toml and the mutation plan were written before the first pinned-scale run, and no rule, threshold, table or parameter was changed after the five-seed batch. After that batch only formatting, clippy-driven edits, documentation, the containment of every product call (so that product panics and starvation falsify rather than abort), the ungated 20 per mille arm, the mutation plan's results and tests changed; the byte-identical re-run at the freeze is the evidence for that. The scenario was tuned so the mechanism has a subject; it was not tuned to make any falsifier pass, and every falsifier that compares the product with an independent oracle is exact and independent of the scenario's parameters.

**The 30 percent grid ties** are a test-sensitivity device: with real-valued scores `>=` against `>` would be an equivalent mutant of the threshold comparison; the atoms make it observable. They are not a claim about real scorers.

## 5. Pipeline, oracles, statistics

**Pipeline per seed** (all through the product's public API): each branch is triaged under three policies: escalate-all, the uncalibrated threshold, and the policy in force (a manual `Never` at first, replaced by `PolicyRecord::calibrate(.., LearnThenTest{alpha: 0.05, delta: 0.05})` after each of 600, 1,000, 1,500 and 2,200 adjudications). The calibration slice rate is 150 per mille of eligible branches (`calibration_draw(branch, seed) < 0.15`), independent of the threshold, so the adjudication sequence is the same whichever policy is in force. A slice triage is adjudicated by the simulated adjudicator through `TriageOutcome::adjudicate`; every non-slice triage is offered to `adjudicate` and must be refused. Held-out set of a policy: `PolicyRecord::held_out` over all adjudications.

**Independent oracles** (no code shared with the product; none reads a product value to compute an expectation): the eligibility rule written from the manifest; the decision, slice flag and propensity a policy must reach; the Learn-then-Test fixed-sequence rule over integer scores (a threshold passes iff the lower binomial tail at alpha is at most delta, so no bound is computed); the conformal risk-control threshold; the Clopper-Pearson bound by bisection on a log-space binomial CDF, checked against its closed forms (`k = 0` is `1 - delta^(1/n)`, `k = n` is 1) and against `ptr_analytics::clopper_pearson_upper` on a fixed grid of counts and on every held-out count; the held-out set by set difference over the harness's own record; the exact true adjudicated harm of a threshold by summing the model's tables; the exact value of an off-policy target. The harness also checks, before anything else, that the product's `f32` grid and the integer rule `m >= 1000 j` agree at every grid edge (exit 2 otherwise).

**Coverage of the guarantee (Monte Carlo).** `mc_replicates = 2000` replicates per seed, cycling over the policy sizes. Each draws a fresh calibration set from the model, adjudicates it through `triage` and `adjudicate`, records a policy with `PolicyRecord::calibrate` and compares its threshold with the oracle's. A replicate **violates** when it certifies a threshold whose exact adjudicated harm rate exceeds alpha; one that certifies nothing cannot. With `R = 2000` and delta = 0.05 the harness derives the **critical value** `c` = the largest integer with `P(Bin(R, delta) >= c) >= 1e-3` (131 at R = 2000) and the rule is `ltt_coverage_violations <= ltt_coverage_critical`.

**Held-out test per recorded policy.** Of the held-out adjudications (by the oracle's set difference) those whose score the policy's threshold admits are its auto-proposal candidates; with `n` of them and `k` labelled harmful, the exact one-sided p-value of "harm <= alpha" is `P(Bin(n, 0.05) >= k)`, rejected at `0.05 / 20` (familywise 0.05 over four policies and five seeds, `heldout_tests = 20`). The rejection is equivalent to the exact lower confidence bound of the harm rate exceeding alpha. A policy that certifies nothing has no candidates and rejects nothing.

**Why not the literal rule.** The manifest says an upper confidence bound above alpha on held-out adjudications is a hard failure. Read literally it fails a correct Learn-then-Test policy: the rule picks the lowest threshold whose calibration bound is at most alpha, so the true harm rate there approaches alpha from below as calibration grows, and a held-out upper bound at a few hundred candidates exceeds alpha unless the true rate is far below it. The harness prints that literal bound per policy (`literal_ucb_exceeds_alpha`) and counts the policies that exceed it (`heldout_violations_literal_ucb`); no rule reads either.

**Off-policy evaluation.** 200 logs of 4,000 branches per seed under a logging policy (threshold 0.5, slice rate 0.15, a fresh uniform draw per branch); an auto-proposal earns +1 if the branch is truly harmless and -9 if harmful, everything else 0. Targets: threshold 0.8 at slice rate 0.15, and threshold 0.65 at slice rate 0.30 (both supported by the log); estimators IPS, SNIPS (descriptive), and doubly robust with a deliberately wrong constant reward model and with a model near the truth. The exact value of each target is the model's; the mean error of IPS and of each doubly robust estimate over the 200 logs must be within 4 standard errors of zero. A target at threshold 0.3 (below the logging threshold) takes an action the log never took, and `evaluate_off_policy` and `doubly_robust` must both refuse it.

**Slice-rate uniformity.** z tests (overall, three namespaces, two score halves) of the share of eligible branches with `calibration_slice()` against 0.15, rejected beyond 4 standard deviations.

## 6. Coverage prerequisites (a run that misses one is vacuous)

- `stream_branches >= 30000`
- `eligible_branches >= 20000`
- `adjudications >= 3000`
- `policies_recorded >= 4`
- `policies_certified >= 1`
- `ineligible_kinds_seen >= 6`
- `nonslice_probes >= 10000`
- `mc_replicates >= 2000`
- `mc_certified_replicates >= 1000`
- `ope_logs >= 200`
- `ope_estimates >= 1600`
- `positivity_probes >= 400`
- `slice_rate_tests >= 6`
- `heldout_candidates_min_certified >= 100`
- `ltt_reviews_saved_permille >= 1`
- `uncalibrated_true_harm_ppm >= 50001`
- `uncalibrated_baseline_rejected >= 1`
- `negative_controls_tripped >= 4`

Counted from observed events except `stream_branches` (the generated length) and `uncalibrated_true_harm_ppm` (the model's exact value; its observed counterpart is `uncalibrated_baseline_rejected`: the held-out test rejects the uncalibrated threshold on the whole adjudication record). `policies_certified >= 1` is deliberately not "all four": Learn-then-Test certifies nothing with probability about one in five from the top 59 adjudications alone (section 4), a property of the mechanism, and a policy that certifies nothing escalates everything and has no subject for the held-out test.

## 7. Falsifiers (a tripped one falsifies F003 on that seed)

- `ineligible_auto_proposals <= 0`
- `decision_mismatches <= 0`
- `unexplained_triages <= 0`
- `nonslice_adjudicated <= 0`
- `adjudication_mismatches <= 0`
- `calibration_set_mismatches <= 0`
- `held_out_overlap <= 0`
- `held_out_missing <= 0`
- `threshold_mismatches <= 0`
- `crc_mismatches <= 0`
- `bound_oracle_mismatches <= 0`
- `ltt_coverage_violations <= ltt_coverage_critical`
- `ltt_coverage_excess <= 0`
- `heldout_policy_violations <= 0`
- `slice_rate_deviation_violations <= 0`
- `ips_bias_violations <= 0`
- `dr_bias_violations <= 0`
- `positivity_unrefused <= 0`
- `ope_errors <= 0`
- `product_errors <= 0`
- `product_panics <= 0`
- `data_starvation <= 0`

`ltt_coverage_excess` is the count of coverage violations beyond the critical value; it restates the cross-reference rule as a hard counter for the mutation checker (a correct policy has a non-zero violation count, so that count cannot be a hard counter). `hard_failures` is the sum of the hard counters. `data_starvation` is one when the product leaves the harness without the data the controls and counts need (fewer adjudications than the first policy size, or under half the Monte-Carlo replicates completed): a product defect that exits 1, never 2. Every call into the product is contained; a panic is `product_panics`, a refusal where success is required `product_errors`.

## 8. Negative controls (each must trip, exit 2 otherwise)

1. A held-out set drawn from a world whose harm is three times the model's, tested at the threshold of the last certified policy: the held-out test must reject it.
2. A rule that estimates without a bound (the same fixed sequence with `k/n <= alpha` for the upper bound) on the Monte-Carlo samples: its violation count must exceed the critical value.
3. The uncalibrated threshold under the held-out test on the whole adjudication record: it must be rejected.
4. The unweighted mean of the logged rewards as an estimate of a target: it must fall outside four standard errors.

## 9. Mutation plan

[tests/mutations.toml](tests/mutations.toml) plants defects in `ptr-branch` and `ptr-analytics` one at a time and runs the harness; COMPLETED requires every planned mutation killed (exit 1 with an expected hard counter fired) and the evidence to be this checkout's. Two mutants a plan would be expected to list are equivalent and are not in it: `<=` to `<` in the Learn-then-Test test (a bound exactly equal to alpha has probability zero) and swapping alpha and delta (both 0.05).

## 10. Pinned values and their derivations

| key | value | derivation |
|---|---|---|
| `alpha_permille`, `delta_permille` | 50, 50 | decided (risk 0.05, confidence 0.95) |
| `min_adjudications_per_policy` | 59 | `0.95^n <= 0.05` first holds at n = 59 |
| `uncalibrated_threshold_permille` | 500 | the conventional "propose when the score is at least one half", the value S003's review policy uses (`s003/params.rs`); scorers of the scenario are overconfident, so the baseline is expected to violate alpha, shown descriptively and as control 3 |
| `slice_rate_permille` | 150 | about 112 adjudications per 1,000 branches; 3,400 per seed, well above the 59 minimum for every policy |
| `adjudicator_error_permille` | 2 | section 4 |
| `policy_sizes` | 600, 1000, 1500, 2200 | several calibration sizes with at least 1,200 held-out adjudications for the last (862 to 984 candidates per seed in the pilot batch) |
| `mc_replicates`, `coverage_test_level_ppm` | 2000, 1000 | critical value 131 with the 6.5 percent tolerance above delta |
| `heldout_familywise_permille`, `heldout_tests` | 50, 20 | Bonferroni over four policies of each of five seeds; each seed is judged alone by its share, 0.0025 |
| `ope_logs`, `ope_log_branches`, `ope_bias_sigmas` | 200, 4000, 4 | a four-sigma bound on the mean error over 200 logs has a false-alarm probability of about 6e-5 per test |
| `stream_branches` | 30000 | about 22,500 eligible branches per seed |

## 11. Decision and limitations

Every seed must hold every coverage rule and every falsifier; COMPLETED iff so with complete mutation evidence, INVALID if a seed is missing, unresolvable or its record does not describe the frozen run, FALSIFIED otherwise. Limitations are listed in `aggregate.py` (`LIMITATIONS`) and repeated in the report: the adjudicator is a simulation; the claim is the mechanism's; the scorer and the world are synthetic; a violation-rate defect smaller than the coverage tolerance is seen by the oracle comparisons and not by the coverage rule; the harm breakdown by namespace is descriptive.
