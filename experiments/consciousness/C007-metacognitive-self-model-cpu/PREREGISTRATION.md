# C007 metacognitive self-model, CPU stage A: preregistration (v2, after the pre-freeze review)

Status: preregistration of version 2, built to freeze-ready; **nothing is frozen and no declared seed has been run**. C007 is the CPU stage A of C002
(`experiments/consciousness/C002-metacognitive-self-model`, whose text points here and which remains the GPU stage B on the Pod-MoE with OLMoE). The scaffolder
accepts only identifiers of a capital letter and three digits, so the stage-A experiment has its own number, C007. The pinned values are `[pinned_thresholds]` in
`config.toml`; `criteria.toml` holds the rules; this file states the reasons and the limits. Version 2 differs from version 1 by the changes the independent
pre-freeze review asked for (section 6, "Fix round"): equal and adequate tuning of both heads with a base term, a validity guard on the heads' strength, a detector of an
inert treatment, a calibration fit audit, a removal control against the probe, a signal floor, INVALID before every other verdict, an aggregator mutation runner and a
freeze procedure that lives in the repository. **What kind of run this is is stated in section 2: a calibration (known-answer) run, not a test with a real chance of support.**

## 1. Scope and non-claims

C007 is a functional-mechanism test, on a CPU toy, of the indicators HOT-2 and HOT-3 of Butlin et al. 2023 as a computation: does a **self-model** that sees a classifier's own
routing traces and internal state predict the classifier's own errors on a distribution-shifted split better than (a) an **information-matched probe** (the same network on the same trace,
with the same capacity, optimiser, data and initial weights) and (b) the classifier's own **calibrated confidence**? **It is not a test of consciousness and no result is evidence about
consciousness.** **No verbal self-report is read, trained or scored** (principle 2 of doc 38): the self-model reads routing traces and is scored against the verifier's deterministic
outcomes (the prediction is wrong or right against the true class).

**What the structural counters are.** `model_write_paths` digests the frozen classifier before and after everything else; `advice_leaks` digests the environment (the verifier's outcomes, the
gate's decisions and threshold, the classifier's digest) around a call to `propose`. In the CPU toy the self-model **has no consumer**: `propose` is a stub the harness itself wrote
(`advice > 0.5`) and the gate is built from the classifier's own confidence, not from the self-model. The counters therefore verify the harness wiring (and the mutations show the detectors
can fail); they are not evidence that a self-model cannot gain authority, and **a structural FALSIFIED is a statement about the implementation, never about a self-model**. The real write-path
test of the registered C002 (a self-model Pod whose advice a gate or a promotion could consume) belongs to stage B.

## 2. Hypothesis, the claim being tested and what the run is

H1. On the shifted split, the Brier score of the self-model's predicted failure probability is lower than that of the information-matched probe and lower than that of the classifier's
own Platt-calibrated confidence, in both cases by at least the pinned margin (4 per mille of Brier, i.e. 0.004), shown by the lower one-sided 95% t bound of the paired per-seed
difference over the declared seeds.

What "self-model" means here. Information-matched means the same features, capacity, optimiser, data and initial weights, so a self-model that differs from its probe can only differ in
what it is asked to learn: the self-model's training adds one **auxiliary self-modelling loss**, a masked prediction of the classifier's own layer-two expert choice and its own output entropy
from the trace with the layer-two part hidden (the targets the registered C002 names: the model's own expert choices and uncertainty); the probe's weight on that loss is 0 (a counter checks
it, and a second counter checks that the loss **ran**: see `inert_treatment_arms`). **This is the only difference.** Both heads are the classifier's own calibrated confidence plus a network
(section 3): the question is what a trace-reading network adds to what the classifier already says about itself, and whether the self-modelling objective adds to that.

**What the run is.** A multi-task objective on a head that already reads the whole trace, on top of the classifier's own calibrated confidence, has little to add; the pilots (section 6)
say it adds nothing measurable: the self-model equals its probe within a few tenths of a per mille of Brier, and both sit within a few per mille of the model's own confidence, which they contain.
So the likeliest outcome is the registered refutation ("the improvement is below the margin"), and the experiment is a **calibration (known-answer) run**: it shows that the pipeline resolves
a null honestly (its gross, graded, removal and null-world controls resolve and the detectors are shown able to fail), it records a negative result about this head family on this toy, and it
costs minutes. It is not a test of whether self-models help, and it does not show that the pipeline would detect a real gain (there is no positive-effect arm: the graded and gross controls are deficit controls, and the planning model's alternative is an assumption). A refutation against the **model's own confidence** is a statement about this head family trained on in-distribution outcomes
only, not about self-models in general; the aggregate report therefore names which comparator the verdict rests on (the one with the smaller lower bound) and which comparators it is refuted
against.

## 3. World, models, training

* **World** (`research/cseries/c002a/c007_task.py`). Eight classes in sixteen dimensions: Gaussian clouds (noise 1.0) around random class means of norm 3. The classifier is trained on 4000
  in-distribution inputs; the self-model and the probes learn from a separate calibration set of 2000 in-distribution inputs, with a validation set of 1000; the primary evaluation is a **shifted
  split** of 4000 inputs (displaced by 1.4 along a fixed direction and with noise scaled by 1.4), and an in-distribution test of 4000 is reported. The verifier's outcome is 1 where the
  classifier's prediction is wrong against the true class. The **null world** is the same pipeline on labels drawn independently of the inputs, so no trace can predict an error.
* **Classifier under study** (`c007_model.py`): a two-layer mixture of experts (four experts per layer, top two, hidden 32), trained for 3000 steps (Adam 0.003) and then **frozen**; its parameter digest
  is checked before and after everything else.
* **Trace** (89 features, plus the classifier's own failure score as a last column): router probabilities and selections of both layers, the hidden state after each layer, the output probabilities and their
  entropy. The "early" part (layer one) and the "late" part (layer two, the output and the failure score) are separable; the auxiliary view hides the late part.
* **Heads** (all the same network, trunk 32-16, three linear heads, AdamW with weight decay 0, batch 128, plateau protocol: the 1 - Brier score on the validation set evaluated every 100 steps, stop after 10
  evaluations without an improvement of 1 per mille, cap 4000 steps (1500 for the control heads), best checkpoint restored; the self-model and the probe start from the **same initial weights** and draw
  the same batches). **The error logit of every head is the classifier's own calibrated confidence plus the network's output** (`base` in `Predictor.forward`: a two-parameter logistic fit of the
  classifier's failure score on the calibration and validation labels of the head's world, a constant, not trained), and **every head's output logit is recalibrated after training** by the same two-parameter
  fit on the validation split (a held-out split for the head). The network can therefore only learn what the trace adds to the classifier's own confidence, and neither head is weaker than the baseline it is
  compared with because of tuning. `self_model` (auxiliary weight 0.3), `probe_matched` (auxiliary weight 0; judged), `probe_io` (input and output features only: descriptive), `self_model_perm`
  (the self-model trained on permuted outcomes, with the base fitted on the permuted outcomes), `self_model_null` and `probe_matched_null` (the null world).
* **Baselines without a trained head.** `own_confidence`: the same two-parameter fit, **on the calibration and validation labels** (the labels the heads were given), scored as it stands (judged);
  `own_confidence_raw` (descriptive), `constant` (the error rate of those labels), `self_model_shuffled` and `probe_matched_shuffled` (the trained heads on traces shuffled across inputs, the same permutation for
  both: removal controls), `self_model_noisy` (Gaussian noise of 0.25 added to the self-model's predictions: a graded control whose deficit is about the noise variance, 0.06, less after clipping).
* **Metrics.** Brier score (the judged metric, on the shifted split, as an integer sum in millionths audited against a plain loop); AUROC, ECE (15 bins), risk-coverage area, the layer-two expert
  accuracy of the auxiliary head: descriptive, in distribution and shifted.
* **Equal tuning.** Both heads got the same twelve configurations: learning rate {0.0003, 0.001, 0.003} x weight decay {0, 1.0} x width {64-32, 32-16}, on the pilot seeds 1001 to 1003, with the base term; the
  auxiliary weight is the treatment's strength and was not tuned in the fix round (0.3, from the first version's pilots). The configuration was chosen by the **sum of the two heads' in-distribution
  Brier** (never by their contrast, never on the shifted split); ties go to the smaller weight decay and then the lower learning rate. Section 6 has the grid, including the grid without the base term
  that showed both heads about ten per mille above the baseline in distribution.
* **CPU budget.** One seed trains two classifiers and six heads single-threaded: 35 CPU seconds measured on the pilot seeds (148 wall seconds on the loaded box), inside the
  budget of 15 minutes per seed on one core.

## 4. Decision rule (frozen in criteria.toml) and what each outcome means

Per seed: three **structural** counters (`model_write_paths`, `advice_leaks`, `nondeterminism`) and five coverage rules (heads trained, the three test-set sizes, the advice probe sees an influence). A
failing rule FALSIFIES (kind structural) **when no INVALID reason precedes it**. Fifteen **integrity** counters make the seed exit 2 and the experiment INVALID: the self-model and the probe matched in
capacity, features and auxiliary weight; **`inert_treatment_arms`** (the self-model's auxiliary term ran on every training step, the probe's never, the auxiliary loss has a gradient on the shared trunk, and the two
trunks diverged by any nonzero amount (on the pilot seeds the largest trunk-parameter difference was 0.05 to 0.06, far from zero, so the check is conservative in practice): the check establishes that the auxiliary loss **ran**, not that it **mattered**, because the best checkpoint is early and the trunks may differ by very little); **`fit_split_errors`** (the calibration fits saw exactly the named splits, checked against an independent recomputation from the splits' inputs); split overlap; the classifier's error rate
band; the verifier's outcomes and the Brier sum audited by plain loops; the permutation, shuffle and noise controls; loss and plateau bookkeeping. Ten negative controls run on doctored inputs and must trip or
the run is exit 2.

**Validity guards (every seed):** the best trace predictor must beat the constant predictor on the shifted split by at least **4 per mille of Brier** (the margin: the traces carry at least the information of an
effect the experiment could detect; the gross control carries the same requirement at the aggregate); **the larger of the two heads' in-distribution Brier must not exceed the model's own calibrated confidence's
by more than 4 per mille** (`head_gap_id_permille`: the heads are strong enough for the question; a head with a few tens of effective extra parameters is within it, an unconstrained 90-parameter head is not, see
`planning.py`); the classifier's error rate on the shifted split must lie between 100 and 700 per mille. Across seeds: no more than one seed with a judged head at the step cap; **gross control** (constant minus
self-model, lower bound at least the margin); **graded control** (noised self-model minus self-model, lower bound above 2 per mille); **removal controls** (shuffled traces; a self-model trained on permuted outcomes:
the improvement over the constant predictor must stay below the margin at its UPPER bound; and **the self-model's advantage over the probe given the same shuffled traces** must stay below the margin at its upper
bound, else the pipeline reports an effect the pairing of traces and outcomes cannot explain); **the null world** (neither self-model minus probe nor the reverse may reach the margin at its lower bound). A failing
guard or control is INVALID.

**Order of the verdicts** (`aggregate.py decide`, pure, tested): (1) an integrity counter above 0 on any seed; (2) a missing, duplicated or undeclared seed, missing or partial mutation evidence (killed not equal to
total, or total not the plan's), a seed that cannot be judged, an environment that differs between seeds: **INVALID**; (3) a structural counter or coverage rule, or failed static evidence: FALSIFIED (structural);
(4) a guard on some seed, more than the allowed seeds at the step cap: INVALID; (5) a control: INVALID; (6) only then the paired verdict: FALSIFIED (refuted), COMPLETED, NOT_SHOWN.

**Margin: 4 per mille of Brier (0.004).** From the planning model (`planning.py`, section 6): a per-seed paired difference has a standard deviation of about 3.3 per mille under the stated priors, the margin
must not sit inside it (1.2 sigma = 4.0), and a 2 per cent improvement of a Brier score of about 0.2 is the smallest that justifies a second network, a trace channel and a second objective over a probe that
reads the same trace. **Statistics.** Over 20 seeds the one-sided 95% t quantile for 19 degrees of freedom is 1.729133; the refutation uses the Bonferroni quantile over the two judged comparators (alpha 0.025,
2.093024): **it means that the improvement is below the margin, not that there is none.**

| Outcome | Condition | Consequence |
|---|---|---|
| COMPLETED (supported) | order of verdicts above holds; both lower bounds (t 1.729133) of probe minus self-model and own confidence minus self-model >= 4 per mille | **a self-model with an auxiliary self-modelling loss beats an information-matched probe and calibrated own confidence on this shifted split, nothing more** (a COMPLETED of this calibration run does not move HOT-L2 or any status in doc 38: that needs stage B; unlikely a priori, see section 2) |
| FALSIFIED (refuted) | guards and controls hold and some Bonferroni upper bound (t 2.093024) < 4 per mille | negative result for this toy and this head family: at the same information and capacity, a self-modelling objective adds less than the margin over a probe or over calibrated own confidence; no downgrade of the GPU stage, and no support for a distinct self-model Pod from this evidence |
| NOT_SHOWN | guards hold, a bound straddles the margin | nothing is concluded |
| INVALID | an integrity counter, seeds or mutation evidence missing or partial, no signal in the traces, heads too weak, an uninformative error rate, a judged head at the cap, an unresolved control, a removal control that does not remove, a null world that shows an effect | nothing is concluded; repair and register a successor |

Exit status of `aggregate.py`: 0, 1, 3, 2 respectively (2 also for any refusal to aggregate).

**Power and expected outcome.** Planning table (`research/cseries/c002a/planning.py`, margin and seed count written and pinned before the first pilot; the fix round added the head-gap and signal derivations, which are
from the sampling model of a logistic-type predictor and not from a pilot), then the table on the fix-round pilot seeds (`research/cseries/lib/pilot_power.py`; it changes neither the margin nor the seed count).

```
excess Brier of a 2-parameter Platt fit on 3000 examples (p 0.35): 0.15 per mille; of a 90-parameter head on 2000: 10.2; effective extra parameters the tolerance of 4 allows at n = 2000: 35
signal prior N(25, 10): P(seed below the floor of 4) = 0.018, P(some seed of 20 below) = 0.30
sigma_d = 3.3; margin rule 1.2 sigma_d = 4.0; margin = 4
  n  alt: COMPLETED  null: COMPLETED  null: FALSIFIED  margin: NOT_SHOWN  alt, 2x sd
  5           0.515            0.000            0.781              0.945       0.094
  6           0.663            0.000            0.884              0.947       0.127
  8           0.853            0.000            0.972              0.952       0.212
 10           0.941            0.000            0.995              0.950       0.303  <- smallest n meeting the first three conditions
 12           0.979            0.000            0.999              0.949       0.392
 14           0.992            0.000            1.000              0.952       0.475
 16           0.998            0.000            1.000              0.947       0.556
 20           1.000            0.000            1.000              0.950       0.690
smallest n meeting the first three conditions: 10; the robustness condition (power >= 0.8 at twice the sd) is met by no n <= 20, so the seed count is the cap, 20
```

```
C007: 10 pilot seeds [1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013]; margin 4 per mille; preregistered seeds 20
  contrast vs probe_matched      mean      0.4  sd    2.2  per-seed [-0.5, -0.1, -0.2, -0.2, 6.0, 0.9, 1.6, -1.9, -1.6, 0.2]
  contrast vs own_confidence     mean     -3.3  sd    5.4  per-seed [-3.0, 0.9, -1.4, 0.3, 7.1, -10.2, -6.3, -8.3, -9.3, -3.0]
  pilot covariance                     COMPLETED 0.000  NOT_SHOWN 0.000  FALSIFIED 1.000  (guards and controls assumed to hold)
  twice the pilot standard deviation   COMPLETED 0.000  NOT_SHOWN 0.045  FALSIFIED 0.955  (guards and controls assumed to hold)
  control gross (constant - self-model)              pilot mean    32.0 sd    7.2 threshold 4: resolves at n=20 with probability 1.000
  control graded (noised - self-model)               pilot mean    22.9 sd    3.8 threshold 2: resolves at n=20 with probability 1.000
  null world             forward pilot mean     0.0 sd    0.1: lower bound stays below 4 with probability 1.000
  null world             reverse pilot mean    -0.0 sd    0.1: lower bound stays below 4 with probability 1.000
  check removal (self_model_shuffled)                              pilot mean   -30.3 sd    6.4 (upper< 4): holds at n=20 with probability 1.000
  check removal (self_model_perm)                                  pilot mean    -0.7 sd    2.8 (upper< 4): holds at n=20 with probability 1.000
  check removal against the probe                                  pilot mean     0.4 sd    1.2 (upper< 4): holds at n=20 with probability 1.000
  guard 'signal_permille >= 4': scalar mean 32.0 sd 6.7, fails on 0 of 10 pilot seeds; normal-fit per-seed failure probability 0.0000, P(some of 20 seeds fails) = 0.000
  guard 'head_gap_id_permille <= 4': scalar mean -0.7 sd 0.5, fails on 0 of 10 pilot seeds; normal-fit per-seed failure probability 0.0000, P(some of 20 seeds fails) = 0.000
  guard 'shifted_error_permille >= 100': scalar mean 376.6 sd 25.4, fails on 0 of 10 pilot seeds; normal-fit per-seed failure probability 0.0000, P(some of 20 seeds fails) = 0.000
  guard 'shifted_error_permille <= 700': scalar mean 376.6 sd 25.4, fails on 0 of 10 pilot seeds; normal-fit per-seed failure probability 0.0000, P(some of 20 seeds fails) = 0.000
```

**Expected outcome (honest).** The planning model's seed count (20, the cap) gives a COMPLETED probability near 1 *if there were an advantage of 8 per mille*; the pilots say there is none against the probe
(mean +0.4 per mille in the self-model's favour, sd 2.2) and a disadvantage of
3.3 per mille against the classifier's own calibrated confidence (sd 5.4; both from the pilot-power block above). With the pilot covariance the decision rule gives
**FALSIFIED (refuted) with probability about 1** (the Bonferroni upper bound against either comparator is far below the margin), and with twice the pilot standard deviation about 0.96 (the rest NOT_SHOWN).
No guard fired on any pilot seed (the guard table above: the signal of the traces is about 32 per mille, 7 standard deviations above its floor, and the heads sit at the baseline). My expectation (the INVALID share is **pilot-based**: the signal of the traces has mean 32 and sd 7 over the pilot seeds; the registered planning prior for the signal, mean 25 and sd 10, would put about 0.30 on some of 20 seeds falling below the floor of 4, and the true share lies between the two):
**FALSIFIED (refuted) about 0.90, INVALID between 0.04 and 0.30 (the signal floor guard), NOT_SHOWN about 0.04, COMPLETED below 0.01.** The experiment is a calibration run (section 2): its refutation is the expected, first-class result, and it is registered so
that the pipeline is shown to return it.

## 5. Mutation evidence

`scripts/mutation_check.py C007 --check` verifies the anchors of `tests/mutations.toml` (50 planted defects; every `find` occurs exactly once in its file). The defects are run by
`research/cseries/lib/run_mutations.py C007` (planted one at a time in a throwaway copy, never in the working tree), because `scripts/mutation_check.py` builds a Cargo package and cannot plant a defect in a Python
entrypoint (a gap reported, as C001 reported it). A defect is killed in one of three ways, stated in the plan's header: the reduced-scale harness exits 1 with an expected structural counter or coverage rule; exits 2 naming an
expected integrity counter or negative control; or the named unit tests fail in the planted copy while they pass in the unmutated one. **Every structural and integrity counter, every coverage rule and each of the
negative controls is expected by at least one mutation** (a test pins that), and every guard derivation and falsifier is named by a mutation. New in the fix round: the inert treatment (`aux-branch-never-runs`,
`aux-term-is-counted-but-scaled-to-zero`), the calibration fits (`platt-baseline-fitted-on-the-shifted-split`, `head-recalibrated-on-the-shifted-split`), the two new negative controls' blinded detectors, the head-gap
derivation and the base term.

**The aggregator is mutated too** (`research/cseries/lib/run_agg_mutations.py C007`, plan `tests/aggregator_mutations.toml` and the shared library's `lib_aggregator_mutations.toml`): 61 planted defects in
`aggregate.py`, `criteria.toml` and the aggregation library (every guard, control, the Bonferroni and support quantiles, the allowance counts, each branch and the precedence of the verdicts, the comparator named in the
headline, the quantile and bound arithmetic, the exit-status agreement), each of which must fail `scripts/tests/test_c007_aggregate.py`, the library tests or the rule-agreement tests, which pass unmutated.

**Run on the uncommitted tree before the freeze (no record written): harness plan 50 of 50 killed; aggregator plan 61 of 61 killed.** `results/mutations.json` is written by the head after the freeze, from the clean tagged tree.

## 6. Pilot disclosure and tuning (pilot seeds 1001 and up only; none is declared)

**Chronology (in order; every number is on a pilot seed, 1001 or higher; nothing was run on a declared seed).**

*Version 1 (before the review).*

1. `planning.py` (margin 4 per mille of Brier, 20 seeds) was written after one tiny smoke run of the harness (300 classifier steps, seed 1001) whose numbers were looked at only to see that it runs.
2. First world (shift 2.0, shift noise 1.4, graded noise 0.09). Learning-rate and auxiliary-weight grids on seeds 1001 to 1003 (the probe got 3 configurations, the self-model 9, which favoured the hypothesis); selected by the mean
   shifted Brier: learning rate 0.001 for both heads, auxiliary weight 0.3.
3. Second power table on seeds 1004 to 1013 in the first world: **the graded control did not resolve** (noised minus self-model +0.8 per mille, sd 1.8, against a required lower bound of 2) and **the gross control resolved with
   probability 0.71** (constant minus self-model 12.5, sd 16.7, against 4). **The world and the graded noise were changed on pilot seeds** with the selection criterion fixed in advance as the resolution of the sensitivity controls
   (the paired contrasts, which I saw in the same table, were not the criterion): graded noise 0.25 and shift 1.4 (candidates (1.0, 1.2), (1.0, 1.4), (1.4, 1.4), (1.4, 1.6); the choice was made while the last was running and the
   candidate nearest to the first world was taken). The auxiliary weight and the learning rates were not re-tuned.
4. In the final world the per-seed signal guard (floor 10 per mille) fired on pilot seed 1008 (signal 2) and sat at exactly 10 on 1009 (mean signal about 21, sd about 14 over ten seeds); **the floor was lowered from 10 to 0**
   (the requirement that the traces help was left to the gross control).
5. Result of version 1 on ten pilot seeds (kept in `research/cseries/c002a/pilots/data/before-fix-round/`): the probe's Brier minus the self-model's averaged -1.5 per mille (sd 7), the model's own calibrated confidence's minus the self-model's
   -17.7, i.e. **the self-model was worse than the model's own confidence by 18 per mille**, and **both heads were 8 per mille worse than the 2-parameter fit of the model's confidence even in distribution**.

*The review and the fix round.* An independent pre-freeze review of version 1 found (among other things) that the heads looked under-tuned against the baseline (so that a refutation against the baseline was a statement about the heads), that nothing
detected an inert treatment, that the signal floor of 0 was vacuous, that the shuffled-traces control was compared with the constant predictor and not with the probe, that structural FALSIFIED preceded INVALID for missing evidence and that the advice
counter checks a stub. The head's decisions on it are what this version implements. Nothing below moves the margin (4), the seed count (20), the world (shift 1.4, noise 1.4, graded noise 0.25) or the auxiliary weight (0.3).

6. **Planning additions** (`planning.py`, "FIX ROUND"): the head-gap tolerance (4 per mille, from k p (1 - p) / n: a 2-parameter fit on 3000 examples costs 0.15 per mille, an unconstrained 90-parameter head on 2000 costs about 10) and the signal floor (the margin, 4),
   written after step 5 had shown the gaps of 5 to 18 per mille but derived from the sampling formula and not from a pilot number; under the planning prior for the signal (normal, mean 25, sd 10) a seed falls below 4 with probability 0.018 and one of 20 seeds
   with probability 0.30; the version-2 pilots (step 10) give a signal of mean 32, sd 7, minimum 23.
7. **Implementation of equal tuning:** weight decay (AdamW), two widths, the same post-hoc two-parameter recalibration on the validation split for every head, the baseline fitted on calibration and validation. A first run on seed 1001 (learning rate 0.001, no weight decay,
   widths 64-32) gave the probe 118.4 against 108.6 for the baseline in distribution (a gap of 9.8) and 238.4 against 238.1 shifted.
8. **Grid without the base term** (table 1; seeds 1001 to 1003; 4 of 12 planned cells, stopped when the pattern was clear): in every cell both heads were 7.5 to 9.5 per mille above the baseline in distribution, beyond the tolerance of 4. **Decision: add the base term**
   (the error logit is the model's own calibrated confidence plus the network's output), so that neither head can be weaker than the baseline it is compared with merely because of tuning on 2000 examples; this makes the question "what does a trace add to the
   model's own confidence".
9. **Grid with the base term** (table 2; 12 cells x 3 seeds, learning rate x weight decay x widths): every cell has both heads within about 2 per mille of the baseline in distribution; the cells are flat (the best checkpoint is the first evaluation, step 100, in
   all of them). The rule of section 3 (smallest sum of the two heads' in-distribution Brier, ties to the smaller weight decay and then the lower learning rate) picked **learning rate 0.0003, weight decay 0, widths 32-16**.
10. The pinned values were set and seeds 1004 to 1012 ran with them; the dry run (seed 1013) went through the pinned entry point in the runner's stripped environment (the last row of table 3): no counter, rule or guard fired and every control resolved.

Table 1. Grid without the base term (mean Brier per mille over seeds 1001 to 1003; lower is better).

| lr | weight decay | widths | seeds | self-model ID | probe ID | own conf. ID | self-model shifted | probe shifted | own conf. shifted | sum of the two heads' ID |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0003 | 0.0 | 32:16 | 3 | 118.6 | 117.8 | 109.1 | 250.8 | 249.0 | 236.8 | 236.4 |
| 0.0003 | 0.0 | 64:32 | 3 | 117.5 | 118.5 | 109.1 | 250.6 | 250.6 | 236.8 | 236.0 |
| 0.0003 | 0.1 | 32:16 | 3 | 116.6 | 116.8 | 109.1 | 246.5 | 247.0 | 236.8 | 233.3 |
| 0.0003 | 0.1 | 64:32 | 3 | 117.5 | 118.6 | 109.1 | 251.0 | 250.9 | 236.8 | 236.2 |

Table 2. Grid with the base term (same columns).

| lr | weight decay | widths | seeds | self-model ID | probe ID | own conf. ID | self-model shifted | probe shifted | own conf. shifted | sum of the two heads' ID |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0003 | 0.0 | 32:16 | 3 | 109.1 | 109.0 | 109.1 | 240.4 | 240.5 | 236.8 | 218.2 |
| 0.0003 | 0.0 | 64:32 | 3 | 109.5 | 109.3 | 109.1 | 239.3 | 239.9 | 236.8 | 218.8 |
| 0.0003 | 1.0 | 32:16 | 3 | 109.1 | 109.0 | 109.1 | 240.3 | 240.4 | 236.8 | 218.2 |
| 0.0003 | 1.0 | 64:32 | 3 | 109.5 | 109.2 | 109.1 | 239.2 | 239.9 | 236.8 | 218.7 |
| 0.001 | 0.0 | 32:16 | 3 | 109.5 | 109.5 | 109.1 | 241.5 | 241.2 | 236.8 | 219.0 |
| 0.001 | 0.0 | 64:32 | 3 | 110.0 | 109.6 | 109.1 | 240.3 | 241.6 | 236.8 | 219.5 |
| 0.001 | 1.0 | 32:16 | 3 | 109.5 | 109.4 | 109.1 | 241.3 | 240.9 | 236.8 | 218.8 |
| 0.001 | 1.0 | 64:32 | 3 | 109.8 | 109.4 | 109.1 | 240.2 | 241.7 | 236.8 | 219.3 |
| 0.003 | 0.0 | 32:16 | 3 | 111.2 | 110.7 | 109.1 | 241.3 | 241.2 | 236.8 | 221.9 |
| 0.003 | 0.0 | 64:32 | 3 | 110.8 | 111.1 | 109.1 | 238.5 | 240.4 | 236.8 | 221.9 |
| 0.003 | 1.0 | 32:16 | 3 | 110.1 | 110.7 | 109.1 | 240.3 | 241.3 | 236.8 | 220.9 |
| 0.003 | 1.0 | 64:32 | 3 | 110.1 | 110.3 | 109.1 | 238.8 | 240.2 | 236.8 | 220.4 |

Table 3. Pinned configuration on pilot seeds (Brier on the shifted split, per mille; the last row is the dry run).

| seed | self-model | probe | own confidence | constant | noised self-model | self-model shuffled | probe shuffled | probe io | signal | head gap (ID) | trunk divergence (micro) | shifted error | cpu s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1004 | 240 | 239 | 237 | 268 | 260 | 295 | 295 | 236 | 28 | -1 | 55967 | 360 | 35 | 181 |
| 1005 | 239 | 239 | 240 | 272 | 266 | 296 | 296 | 238 | 32 | 0 | 52790 | 368 | 35 | 182 |
| 1006 | 244 | 243 | 242 | 276 | 263 | 301 | 301 | 242 | 32 | -1 | 51149 | 380 | 35 | 167 |
| 1007 | 213 | 213 | 214 | 258 | 242 | 277 | 278 | 213 | 44 | -1 | 53558 | 331 | 34 | 173 |
| 1008 | 253 | 259 | 260 | 294 | 275 | 325 | 328 | 261 | 41 | 0 | 184682 | 423 | 36 | 167 |
| 1009 | 242 | 243 | 232 | 270 | 263 | 310 | 310 | 236 | 27 | -1 | 52820 | 379 | 35 | 154 |
| 1010 | 235 | 237 | 229 | 271 | 263 | 305 | 305 | 233 | 36 | 0 | 51911 | 359 | 35 | 126 |
| 1011 | 256 | 254 | 248 | 280 | 279 | 317 | 316 | 248 | 26 | -1 | 283288 | 406 | 36 | 131 |
| 1012 | 258 | 256 | 249 | 280 | 275 | 315 | 315 | 257 | 23 | -1 | 52028 | 383 | 35 | 128 |
| 1013 | 247 | 247 | 244 | 278 | 270 | 309 | 309 | 249 | 31 | -1 | 50363 | 377 | 33 | 66 |

What the pilots say, including what is unwelcome: (a) **the self-model is not better than the information-matched probe** (the paired difference averages +0.4 per mille in the self-model's favour, sd 2.2): the auxiliary
objective changes the heads' output predictions (the trunks themselves differ by about 0.05 in parameters) by less than a per mille; (b) **it is also not better than the classifier's own calibrated confidence it contains**: worse by 3.3 per mille on average and better on 3 of 10 seeds (the cause is not established; both heads are trained on in-distribution outcomes only); (c) the trace carries nothing for the head beyond input, output and the confidence it already has (the head on input and output only is +1.7 per mille better than the trace head, i.e. not worse), so the verdict has no room to be anything but a refutation;
(d) the sensitivity, removal and null-world controls resolve (section 4) and no guard fired on any seed; (e) the world is the first version's (changed once, step 3) and the treatment strength was not re-tuned.

## 7. Limits

* A result applies to this toy: a two-layer four-expert classifier on Gaussian class clouds, a displacement-and-noise shift, an 89-feature trace and a self-model that differs from its probe in one auxiliary loss (a masked prediction of the
  classifier's layer-two expert choice and output entropy). A different self-modelling objective, a self-model that sees more than the probe, one trained jointly with the classifier (which would be a write path from the self-model into
  the classifier and is excluded by the structural counter), or a real model's traces, are other experiments. It says nothing about OLMoE or the GPU stage B.
* **The structural counters verify the harness wiring** (section 1): in the toy the self-model has no consumer. The real write-path test is stage B (C002).
* "Information-matched" is read strictly (same features, capacity, optimiser, data, initial weights), and both heads contain the classifier's own calibrated confidence. With that reading the self-model has little to add; the likeliest
  outcome is a refutation. A refutation against the model's own confidence is a statement about this head family trained on in-distribution outcomes only.
* The classifier's own confidence is a two-parameter logistic fit; a monitor that could calibrate on the shifted split, or a better-calibrated confidence (temperature and bias per class, isotonic), would be a stronger comparator.
* The shifted split is one displacement and one noise scale; the heads are trained on in-distribution outcomes only, which is the point (a monitor must work where it was not trained), and also a limit.
* The noise control has a known deficit (about the noise variance, less after clipping) only approximately; the removal controls show that the measured improvement depends on the pairing of traces and outcomes, not that the main effect has any size.
* The auxiliary weight of the self-model (0.3) is the first version's pilot choice, made in the first world with heads without the base term and **not re-tuned**; a stronger or weaker treatment is another experiment.
* The per-seed signal floor (4 per mille) is a planning-model quantity; if the world's real signal is weak a seed can fall below it by chance and the run is then INVALID. The pilot rate is in section 4.
* The t bounds assume roughly normal paired differences, which 20 points cannot check; per-seed differences, medians and both bounds are reported.
* An indicator property here is a functional property: the experiment says nothing about whether any system is conscious.
* The world was changed once after pilots in the first version (shift 2.0 to 1.4, graded noise 0.09 to 0.25), because two of the experiment's own sensitivity controls did not resolve reliably; the criterion was the resolution of
  those controls, the paired contrasts were visible in the same table, so the choice is not independent of them (section 6).

## 8. Entry point, environment and freeze procedure

The entrypoint is `env UV_PROJECT_ENVIRONMENT=../../../c001-venv PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline --no-python-downloads
--project research/consciousness python -B research/cseries/c002a/harness.py --benchmark c007 --seed <seed>`, and the manifest declares the matching
`preflight` command (the same program and environment with `--preflight` instead of `--seed <seed>`; the runner runs it, in the entrypoint's exact environment,
before it reserves a seed, and `python3 scripts/run_experiment.py preflight C007` runs only it). Reasons, the same as C001's (found there by rehearsing the
path through `scripts/run_experiment.py` in a scratch clone):

* The runner passes only PATH, HOME, TMPDIR/TEMP/TMP, locale and the cargo/rustup homes (plus PYTHONNOUSERSITE=1 and PYTHONHASHSEED=0): no proxy, certificate
  or `UV_*` variable. So `--offline` (nothing is downloaded at run time) and `--no-python-downloads`.
* The runner **refuses to run while the repository holds git-ignored files** (`.venv/`, `__pycache__/`). The environment therefore lives outside the repository
  (`UV_PROJECT_ENVIRONMENT`, resolved against `research/consciousness`: the directory `c001-venv` next to the repository checkout), bytecode is not written (`-B`,
  `PYTHONDONTWRITEBYTECODE`), and the freeze must be run from a tree without `__pycache__` directories.
* The locked environment holds **torch only** (no numpy); the harness is written for it. It is **C001's** environment: `research/consciousness/pyproject.toml`
  and `uv.lock` are not changed by this experiment and are not in its provenance except as read-only digests in every result's environment record.

**Where the environment lives.** In a checkout whose parent has no `c001-venv`, create it first with the network variables set and as the user and HOME that
will run the seeds: `UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked --project research/consciousness`; then the smoke check `env -i PATH=... HOME=...
UV_PROJECT_ENVIRONMENT=../../../c001-venv uv run --locked --offline --no-python-downloads --project research/consciousness python -c "import torch"` must pass. The harness
checks its environment (python 3.11, torch 2.7.1, the config file) before training and exits 2 with a message otherwise, so an incomplete environment cannot be mistaken for
a falsification. `uv` and CPython 3.11 must be on PATH (absolute entries). **Never run two seeds concurrently** (each run is single-threaded and its environment record holds
the thread counts). **All 20 seeds run on one host in one environment:** the aggregator marks a seed invalid whose recorded environment differs from the first one. A seed takes
35 CPU seconds (section 3), so the 20 seeds take well under an hour of CPU time and more of wall time on a loaded shared machine.

**Freeze procedure.** Every step is a command in the repository; nothing refers to a script outside it. The two sha256 values of the `[preregistration]` table and the manifest digests read
`must-be-pinned-before-prepared` (absent) until step 4. Run every `python3` call with `export PYTHONDONTWRITEBYTECODE=1`. Commit messages carry the attribution trailer the session requires.

1. **Clean the tree and commit it** (the pre-freeze commit). The commit must include the untracked `research/cseries/` (the shared library, the three experiments' code, the pilots with their raw outputs, the planning scripts and
   the tests); `git status` shows new files as untracked until added. `find . -name __pycache__ -type d -prune -exec rm -rf {} +`; the repository must hold no git-ignored file when the runner starts.
2. **Check, before the digests are pinned:** `python3 scripts/run_experiment.py validate`; `python3 scripts/check_research_gates.py`; `python3 scripts/mutation_check.py C007 --check`;
   `python3 research/cseries/lib/run_agg_mutations.py C007 --check`; `python3 -m unittest scripts.tests.test_c007_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests scripts.tests.test_ci_local`;
   the research tests with the locked interpreter, `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B -m unittest discover -s research/cseries/tests`; `python3 scripts/update_component_docs.py --check`; `python3 scripts/check_repo.py`.
3. **Run the aggregator mutation plan** (seconds): `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_agg_mutations.py C007 --write`; every defect must be killed; commit the written
   `tests/aggregator-mutations.json`.
4. **Pin the digests:** `python3 research/cseries/lib/pin_freeze.py C007` (it sets `preregistration_text_sha256` and `criteria_sha256` in `config.toml`, the two manifest digests and `status = "prepared"` in
   `experiment.toml`, and `status = "prepared"` in `experiments/registry.toml`; it refuses a tree that is not a planned, unpinned one; `--dry-run` shows what it would write). `scripts/tests/test_research_gates.py` is not
   edited: its list names the experiments that preregister, as C001 stays in `LISTED_UNFROZEN` after its freeze. Then `python3 scripts/run_experiment.py validate`, `python3 scripts/check_research_gates.py`,
   `python3 -m unittest scripts.tests.test_c007_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests`; commit ("freeze preregistration") and tag it `c007-prereg`. The mutation evidence must run on this
   committed tree: `aggregate.py` refuses if `research/cseries` or this directory differs from the commit the mutation record names.
5. **Mutation evidence of the harness**, from the clean tagged tree: `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_mutations.py C007` (the harness at the plan's reduced scale, a minute or less
   of CPU per defect; it writes `results/mutations.json`); commit it.
6. **Prepare and run:** the pre-flight (`python3 scripts/run_experiment.py preflight C007`), `python3 scripts/run_experiment.py prepare C007`, and commit (`git add experiments && git commit`); then for each seed 6001 to 6020,
   once each: `python3 scripts/run_experiment.py run C007 --seed <s>` and **commit the new run record before the next seed** (the runner reads the committed earlier records when it checks a new run). A seed
   whose run exits 2 is not repeated by hand: stop and report; the experiment is INVALID and the repair is a successor registered with `scripts/new_experiment.py` (a prepared experiment is never patched).
7. **Aggregate:** `python3 experiments/consciousness/C007-metacognitive-self-model-cpu/aggregate.py --report report.md`; then `python3 scripts/check_research_gates.py` and `python3 scripts/run_experiment.py validate`; advance the
   registry status by hand only for COMPLETED or FALSIFIED, as the aggregate's `recommended_status` says.
