# C004 recurrent integration: preregistration (v2, after the pre-freeze review)

Status: preregistration of version 2, built to freeze-ready; **nothing is frozen and no declared seed has been run**. The experiment was registered
as a planned entry by the first extension of the consciousness track (indicator RPT-1 as a functional property, `docs/architecture/38`); this text
turns that entry into a design to the standard of C001: an information-matched and a floor comparator, a planned margin and seed count, pilots on
seeds from 1001 only, every falsifier and guard named by a planted mutation, and a freeze procedure. The pinned values are `[pinned_thresholds]` in
`config.toml`; `criteria.toml` holds the rules; this file states the reasons and the limits. Chronology of the design is in section 6: **the task
was changed once after pilots (from a word problem in S3 to a modular counter); the planned margin and seed count were written after one development run on seed 1001 and before the others; and version 2 follows an independent
review that found the comparator guard able to turn a competent baseline into INVALID and no control on which the main contrast could fail: both are repaired (section 4) and the stack was redesigned (section 3).**

**What this run is, plainly.** C004 tests recurrence against a **non-extrapolating architecture class**: a fixed-depth stack of attention layers does not count beyond the depth it was trained for, and nothing in the pilots shows a configuration of it that does. A COMPLETED is therefore
close to a known answer; the value of the experiment is a validated harness with a refutation path (a competent comparator yields FALSIFIED, not INVALID), a control on which the contrast can fail (the stack trained on the long length), and
null worlds, not a discovery.

## 1. Scope and non-claims

C004 is a functional-mechanism test, on a CPU toy task, of whether **algorithmic recurrence with a persistent state** (the indicator RPT-1 of Butlin
et al. 2023; the latent recurrence of `docs/architecture/01-ptr-core.md` and `latent_steps` of `model/burn-a0`) extrapolates in length where a parameter-matched
feed-forward stack of unshared layers does not. **It is not a test of consciousness and no result is evidence about consciousness.** **Only RPT-1 is targeted, and of
RPT-L1 of doc 38 only its extrapolation clause, at matched parameters; the stack is not depth-matched to anything, its FLOPs are matched at the longest trained length only
(it costs more at 12 steps), and RPT-L2 (persistence across calls, feeding the workspace) is not touched.** No self-report is produced, trained or measured. It says nothing about language models: the toy is a state-tracking problem chosen
because the answer needs a state carried over every step; whether a carried state helps the experts of an OLMoE-class model is a different, GPU, stage
that is not registered here. It extends M003 (which asks only about tokens saved by latent steps) by an integration test.

## 2. Hypothesis and the claim being tested

H1. Trained on sequences of 2 to 6 steps of a modular counter (the running sum modulo six, supervised after every step), a gated recurrent net with a
carried state answers correctly, at **twelve** steps (twice the longest trained sequence), by at least the pinned margin (100 per mille) more often
than (a) a causal feed-forward stack of unshared attention layers with relative-position bias, matched to it in parameters (within 2%) and, at the longest trained length, in FLOPs
(within 10%), and (b) the same recurrent net with the carried state reset at every step. Both lower one-sided 95% t bounds of the paired
per-seed differences over the declared seeds must reach the margin; **and more than half of that contrast must be due to length extrapolation**: the same stack trained with
the twelve-step length inside its training set must recover more than half of the recurrent arm's advantage over the stack (otherwise the contrast is one of capacity).

What a result means. A supportive result says: on this task, a carried state extrapolates in length where a fixed-depth attention stack of the same
size does not, that the stack can hold the twelve-step answer when it is trained on that length (so it is extrapolation that fails), and that the ablation shows it is the carried state that is used. It is the expected direction (fixed-depth attention is known to
struggle with counting beyond its training lengths), so the information in a COMPLETED outcome is small by design. A competent comparator (the stack extrapolating to 0.90, say) yields a refutation, reported as such, and the extrapolation-removed control cannot turn it into INVALID (it is read only for a supported contrast); the ceiling, saturated seeds and the other controls still can (section 4, `planning.py`). A refutation
would be the surprising outcome and is reported as a negative result.

## 3. Task, arms, training

* **Task** (`research/cseries/c004/c004_task.py`). A sequence of T symbols from {0..5}; the answer is the running sum modulo six after the last step. The
  state is one of six values; replacing any one step by a different symbol changes the answer (a test); the answer given the last step alone is uniform
  (a test), so the last step does not determine it. There is no position information, so T is free at test time. Training: finite pools of 1024
  sequences for each length 2 to 6, drawn once per seed; a batch has 64 sequences of one length drawn at random. **Every arm of a world sees the same batches
  in the same order.** Supervision is the counter after every step (the arms are causal read-outs; the answer to a sequence is the read-out at its last step).
* **Evaluation.** 2048 fresh sequences each at 6 steps (in distribution), **12 (primary)**, 18 and 24; 1024 each at 9 and 15 (for the accuracy-against-steps
  curve). No evaluation sequence equals a training or a validation sequence (rejection sampling, audited by hash; length is part of the hash). **Why 12:** twice the longest trained sequence, fixed in advance as the smallest
  length at which a stack with a fixed number of layers has to extrapolate and not interpolate; 18 and 24 are secondary and descriptive.
* **Arms.**
  * `recurrent`: embedding (32), one GRU layer (128), linear read-out (63174 parameters). The carried state is the GRU's hidden state.
  * `feedforward`: a causal stack of **unshared** pre-norm attention blocks (5 layers, 4 heads, width and feed-forward width chosen so that its parameter
    count is within 2% of the recurrent arm's and its FLOPs at the longest trained length (six steps) are the closest to the recurrent arm's among the candidates
    within the tolerance), with an ALiBi relative-position bias (slope scale 4, pilot-selected among {1, 4, 16}: no absolute
    positions, so that it accepts longer sequences) and the read-out at the last position. FLOPs count multiply-adds as 2 and leave elementwise operations out for
    both arms (a test pins the formulas to hand counts and, for the linear part, to torch's counter). At 12 steps and more the attention stack costs more FLOPs
    than the recurrent net; this is reported, not matched. **Depth:** **5 layers.** Chosen on pilot seed 1002 as the shallowest depth at which the stack trained on the long length reaches 0.80 at 12 steps within the step cap (table 1: 4 layers 0.72, 5 layers 0.83, 6 layers 0.86 for 12 percent more CPU); it is therefore a pilot-informed choice, made to give the comparator a fair chance to hold the answer, not an a-priori one (the first-version reasoning, "the shallowest depth of a parallel prefix over 12 steps is four", was a lower bound that training did not reach). The depths were compared at slope scale 1; the scale was chosen afterwards at depth 5 (below).
  * `feedforward_inlength` (new, the extrapolation-removed control): the same stack, trained with the primary length (12) in its training set (its pool of 1024 sequences of 12 steps is drawn after every other pool, and its plateau protocol validates on both the 6-step and the 12-step held-out sets). It is not a
    comparator: it shows whether the stack can hold the twelve-step answer at all.
  * `reset_state`: the recurrent net with the carried state reset at every step (the GRU is applied to each step alone), so the read-out sees only the last
    step. **On this task it is a floor by construction**; the comparison with it shows that the recurrent arm uses its carried state, not that it beats a strong
    alternative.
  * Sensitivity and null arms: `recurrent_perm` and `feedforward_perm` (the same nets trained on labels permuted across sequences), `recurrent_null` and
    `feedforward_null` (the two nets on a task whose answer is the sum of the LAST TWO steps only: no state integrated over the sequence is needed, so neither
    arm can have an advantage that carrying a state explains), and `recurrent_corrupt` (the trained recurrent net evaluated on sequences in which one step of 15% of
    them is replaced: a known deficit of the margin's order).
* **Plateau protocol (every trained arm, identical).** AdamW (weight decay 0), gradient clip 1, **a linear warmup of 300 steps for every arm**; the accuracy at the longest trained length on a held-out validation set of 1024
  sequences is evaluated every 250 steps; training stops when it has not improved by 5 per mille for 12 consecutive evaluations, or at the cap (20000 steps for
  the task arms, 8000 for the null-task arms, 3000 for the permuted-label arms and the reset arm, which cannot learn); the best-validation checkpoint is restored.
  The harness audits the loop (integrity counters): the restored weights must reproduce the best validation accuracy, and the convergence flag and the step count
  must agree with an independent recomputation from the recorded curve.
* **Tuning (what was actually run; section 6 has the numbers).** Pilot seed 1002 screened the recurrent arm's learning rate over {0.003, 0.01} (with the warmup), and the stack at 5 layers over {0.001, 0.003, 0.01}, with the in-length control at {0.001, 0.003, 0.01} (table 1). Selection rules, written before the screens were read: the recurrent arm's learning rate by the highest 12-step accuracy (0.01: 982 against 967); the stack's learning rate, **shared by the stack and the in-length control**, by the highest mean best-validation accuracy of the two (0.003: 891 and 923; 0.001: 858 and 901; 0.01: 894 and 769; these runs at slope scale 1). The slope scale was then screened at 5 layers and learning rate 0.003 over {1, 4, 16} (table 1), with the first version's eligibility clause (the stack must reach 900 at 12 steps on the null task) and the same rule: scale 1 (null task 900, best-validation accuracy 871 and 923 in the runs of that screen), **scale 4 (null task 999; 931 and 934)**, scale 16 (null task 1000; 938 and 829): **scale 4**. The first version's scale (16) was the steepest, most local one; with it the stack trained on the long length reaches only 0.72, and the standard scale (1) fails the null task at 0.90 on one seed and by 0.85 on another (pilot seed 1005, first run of this round: recurrent 1.00 against 0.85), which would have made the null-world check fire.
* **Optimisation effort and its limit.** Both families got the same warmup, optimiser and plateau rule; the learning-rate grids differ (two values for the recurrent arm, three for the stack and its in-length control), the stack's was the larger. The stack's in-distribution accuracy is 0.94 at best (table 1), not 1.0, and the stack trained on the long length reaches 0.83 at 5 layers and slope scale 1 and 0.86 at the pinned scale 4 (pilot seeds: 0.77 to 0.91): **the stack remains optimisation-limited on this task at this size and budget** (a longer patience and a 30000-step cap did not help: 0.71 against 0.72 at 4 layers; weight decay 0.1 and 1.0 hurt: 0.39 and 0.19). That is reported, not repaired; it is one reason the control asks for a share (one half) and not for equality.
* **CPU budget.** One seed trains nine arms single-threaded: 606 CPU seconds measured on the pilot seeds (wall time about 8.1 times longer on the shared, loaded box), inside the budget of 15 minutes per seed on one core.

## 4. Decision rule (frozen in criteria.toml) and what each outcome means

Per seed: three **structural** counters (`reset_carry_leaks`: replacing the first step moved the reset-state arm's read-out; `ff_causal_leaks`: replacing the last
step moved the feed-forward stack's read-out at an earlier position; `nondeterminism`) and five coverage rules (arms trained, the three evaluation set sizes, the
carry probe sees the recurrent arm). A failing rule FALSIFIES (kind structural) **when no INVALID reason precedes it**; a structural FALSIFIED is a statement about the
implementation, not about the hypothesis, and the repair is a successor. Thirteen **integrity** counters are defects of the task or the harness and make the seed
exit 2 and the experiment INVALID, never FALSIFIED: parameter match and FLOP match (the tolerance guard), a training batch with an evaluation length (the in-length control is exempt, and **`inlength_control_errors`** checks that it, and only it, saw the primary length), train/validation
overlap with evaluation, non-finite loss, checkpoint and convergence bookkeeping, labels that differ from an independent recomputation, a reset-state arm above chance
(the task leaks its answer to the last step), a label permutation that left too many labels in place, a corruption rate off its pinned value. Eight negative controls
run on doctored inputs, never on the arm under test, and must trip or the run is exit 2.

**Validity guards** (the aggregator; a failing experiment is uninformative, INVALID, never FALSIFIED): **the feed-forward comparator at 12 steps must stay below the saturation ceiling of 980 per mille on every seed** (a stack at or above it
leaves no room for the margin; a competent stack at 0.90 or 0.95 is **not** saturated: it yields a refutation, see the table below and the power section); the recurrent arm and the feed-forward comparator must each reach 800 per mille in distribution on all but one of the
seeds (an untrained comparator is a strawman); a judged arm at the step cap on more than one seed; **the gross sensitivity control** (recurrent minus the same net
trained on permuted labels, lower bound at least the margin: a collapse to chance is resolved), **the graded control** (recurrent minus the same trained net on
corrupted inputs, a deficit of about 100 per mille, lower bound above 50), **the extrapolation-removed control** (the stack trained on the long length recovers more than half of the recurrent arm's advantage over the stack: the lower one-sided bound of
(inlength minus stack) minus one half of (recurrent minus stack) is at least zero, else the SUPPORTED contrast is one of capacity and the experiment is INVALID. **This control is read only for a supported contrast** (a competent stack gains little from seeing the length, so reading it for a contrast below the margin would turn every refutation into INVALID and close the refutation path). The wording that applies: **loosened after exploration (seed 1002), because the stack is optimisation-limited (empirical); the 1/2 threshold is a verbal a-priori argument (the smallest share at which more of the contrast is due to length than to anything else), not from the planning model; depth 5 was chosen as the shallowest depth at which the in-length stack reaches 0.80, which is tuning toward the control** (at depth 4 the in-length stack would have recovered about 67 percent of the contrast; the saturation ceiling of 980 is likewise not planning-derived)),
and **two null worlds** (permuted labels for both nets; the last-two-steps task): the lower bound of the recurrent-minus-feed-forward contrast, in either direction, must stay below the margin, else the pipeline manufactures an advantage where none can exist
(for example because the feed-forward arm cannot extrapolate even a local task, which would confound the main comparison with a positional-encoding failure).

**Order of the verdicts** (`aggregate.py decide`, pure, tested): (1) an integrity counter above 0 on any seed; (2) a missing, duplicated or undeclared seed, missing or partial mutation evidence, a seed that cannot be judged, an environment that differs
between seeds: **INVALID**; (3) a structural counter, a coverage rule or failed static evidence: FALSIFIED (structural); (4) a guard on some seed, an untrained arm or an arm at the step cap beyond the allowances: INVALID; (5) a sensitivity control or a null world: INVALID; (6) the paired verdict: a refutation is FALSIFIED (refuted); a supported contrast is COMPLETED only if the extrapolation-removed control holds, else INVALID; otherwise NOT_SHOWN.

**Margin: 100 per mille of 12-step accuracy.** Derived from the planning model (`research/cseries/c004/planning.py`, written after one development run on seed 1001 and pinned before the other pilots): a per-seed
paired difference has a standard deviation of about 84 per mille under the stated priors (between-seed training sd 70 per mille per arm from the range of C001's
contrasts, correlation 0.3, binomial evaluation noise), the margin must not sit inside that noise (1.2 sigma = 101) and is a cost judgement (a sequential, non-parallel
computation in training is worth paying for a gain of ten points at twice the trained length): 100. **The margin is non-binding at the observed effect** (the pilots' contrast is nearly eight times it): the pilot power rows below are not evidence about
discrimination, which is why the table of competent baselines is the one that shows it.

**Statistics.** Over the 14 declared seeds the one-sided 95% t quantile for 13 degrees of freedom is 1.770933; the refutation uses the Bonferroni quantile over the two
judged comparators (alpha 0.025, 2.160369): **it means that the recurrent advantage is below the margin, not that there is none.**

| Outcome | Condition | Consequence |
|---|---|---|
| COMPLETED (supported) | order of verdicts above holds; both lower bounds (t 1.770933) of recurrent minus feed-forward and minus reset-state >= 100 per mille | **an extrapolation advantage of a carried state over a parameter-matched fixed-depth stack on a counting task, nothing more** (RPT-1 and the extrapolation clause of RPT-L1 get a CPU toy datum in their favour; RPT-L2 is not touched; a known direction) |
| FALSIFIED (refuted) | guards, sensitivity controls and null worlds hold and, for some comparator, the UPPER bound (t 2.160369) < 100 per mille | negative result for this setting; a carried state does not extrapolate better than the matched stack at this size and these lengths (a competent stack at, say, 0.90 lands here, not in INVALID); no downgrade of the GPU stage |
| NOT_SHOWN | guards hold, a bound straddles the margin | nothing is concluded |
| INVALID | a saturated (>= 980) or untrained comparator, an untrained recurrent arm, a judged arm at the cap, an unresolved sensitivity control, an extrapolation-removed control that fails for a supported contrast, a null world that shows an advantage, any integrity counter, seeds or mutation evidence missing or partial | nothing is concluded; repair and register a successor |

Exit status of `aggregate.py`: 0, 1, 3, 2 respectively (2 also for any refusal to aggregate).

**Power and expected outcome.** Planning table (`research/cseries/c004/planning.py`; priors: between-seed sd 70 per mille per arm, correlation 0.3, evaluation noise; alternative: a
true advantage of 250 per mille, null: none), with **rows near the margin and a competent-baseline table**; then the table on the fix-round pilot seeds (`research/cseries/lib/pilot_power.py`, which uses the aggregator's own bounds; it changes neither the margin nor the seed count).

```
sigma_d = 84.3; margin rule 1.2 sigma_d = 101.1; margin = 100
  n  alt: COMPLETED  null: COMPLETED  null: FALSIFIED  margin: NOT_SHOWN  alt, 2x sd
  5           0.892            0.000            0.770              0.945       0.255
  6           0.962            0.000            0.874              0.947       0.353
  8           0.996            0.000            0.966              0.952       0.540
 10           0.999            0.000            0.993              0.950       0.691
 12           1.000            0.000            0.998              0.949       0.800
 14           1.000            0.000            1.000              0.952       0.874  <- smallest n meeting every condition
 16           1.000            0.000            1.000              0.947       0.922
 20           1.000            0.000            1.000              0.950       0.974
smallest n meeting the rules: 14
a competent baseline (n = 14; recurrent 975, reset-state 175; the stack's 12-step accuracy in the first column): COMPLETED / NOT_SHOWN / FALSIFIED
  stack   980: contrast    -5: 0.000 / 0.010 / 0.990
  stack   940: contrast    35: 0.000 / 0.242 / 0.758
  stack   900: contrast    75: 0.004 / 0.823 / 0.173
  stack   875: contrast   100: 0.052 / 0.924 / 0.024
  stack   850: contrast   125: 0.281 / 0.717 / 0.001
  stack   800: contrast   175: 0.935 / 0.065 / 0.000
  stack   600: contrast   375: 1.000 / 0.000 / 0.000
  stack   250: contrast   725: 1.000 / 0.000 / 0.000
rows near the margin (n = 14, both comparators at the stated mean): COMPLETED / NOT_SHOWN / FALSIFIED
  mean     0: 0.000 / 0.000 / 1.000
  mean    50: 0.000 / 0.216 / 0.784
  mean    75: 0.000 / 0.683 / 0.317
  mean   100: 0.003 / 0.952 / 0.045
  mean   125: 0.080 / 0.917 / 0.003
  mean   150: 0.458 / 0.542 / 0.000
  mean   250: 1.000 / 0.000 / 0.000
```

```
C004: 5 pilot seeds [1003, 1004, 1005, 1006, 1007]; margin 100 per mille; preregistered seeds 14
  contrast vs feedforward        mean    764.8  sd   46.6  per-seed [781.2, 800.3, 802.2, 750.0, 690.4]
  contrast vs reset_state        mean    814.9  sd    7.9  per-seed [826.2, 805.2, 817.4, 815.9, 810.1]
  pilot covariance                     COMPLETED 1.000  NOT_SHOWN 0.000  FALSIFIED 0.000  (guards and controls assumed to hold)
  twice the pilot standard deviation   COMPLETED 1.000  NOT_SHOWN 0.000  FALSIFIED 0.000  (guards and controls assumed to hold)
  control gross (recurrent - permuted labels)        pilot mean   825.8 sd    6.6 threshold 100: resolves at n=14 with probability 1.000
  control graded (recurrent - corrupted input)       pilot mean   146.8 sd    3.4 threshold 50: resolves at n=14 with probability 1.000
  null (permuted labels) forward pilot mean    -6.9 sd   10.5: lower bound stays below 100 with probability 1.000
  null (permuted labels) reverse pilot mean     6.9 sd   10.5: lower bound stays below 100 with probability 1.000
  null (last two steps)  forward pilot mean     2.1 sd    0.8: lower bound stays below 100 with probability 1.000
  null (last two steps)  reverse pilot mean    -2.1 sd    0.8: lower bound stays below 100 with probability 1.000
  check extrapolation removed (recovered - 500/1000 advantage)  pilot mean   259.9 sd   51.7 (lower>= 0): holds at n=14 with probability 1.000
  guard 'comparator_primary_permille <= 980': scalar mean 220.4 sd 47.1, fails on 0 of 5 pilot seeds; normal-fit per-seed failure probability 0.0000, P(some of 14 seeds fails) = 0.000
```

**Expected outcome (honest).** The statistical risk is small: the pilots' contrasts are far above the margin (the table above). The validity risks are the new ones: the stack trained on the long length must recover more than half of the contrast on all seeds in aggregate (the pilot mean is
863 against a recurrent arm at 985 and the short-trained stack at 220: it recovers 84 percent of the contrast); the saturation ceiling cannot bind (the stack's 12-step accuracy is 178 to 295); a guard on a fresh seed. My expectation:
**COMPLETED about 0.85, INVALID about 0.12 (mostly the extrapolation-removed control and the step-cap allowance, the stack being optimisation-limited), FALSIFIED (structural) about 0.01, NOT_SHOWN and FALSIFIED (refuted) below 0.01.** A COMPLETED confirms a known direction on a small scale; it is a validated harness with a refutation path and a control that can fail, not a discovery. The competent-baseline table above says what the same harness would return for a stack that did extrapolate.

## 5. Mutation evidence

`scripts/mutation_check.py C004 --check` verifies the anchors of `tests/mutations.toml` (44 planted defects; every `find` occurs exactly once in its file). The defects are run by
`research/cseries/lib/run_mutations.py C004` (planted one at a time in a throwaway copy, never in the working tree), because `scripts/mutation_check.py` builds a Cargo package and cannot plant a defect in a Python
entrypoint (a gap reported, as C001 reported it). A defect is killed in one of three ways, stated in the plan's header: the reduced-scale harness exits 1 with an expected structural counter or coverage rule; exits 2 naming an
expected integrity counter or negative control; or the named unit tests fail in the planted copy while they pass in the unmutated one. **Every structural and integrity counter, every coverage rule and each of the
negative controls is expected by at least one mutation** (a test pins that). New in the fix round: the in-length control's lengths (never seeing the long length; every arm seeing it), its blinded detector, and the comparator scalar the saturation guard reads.

**The aggregator is mutated too** (`research/cseries/lib/run_agg_mutations.py C004`, plan `tests/aggregator_mutations.toml` and the shared library's `lib_aggregator_mutations.toml`): 68 planted defects in
`aggregate.py`, `criteria.toml` and the aggregation library (the saturation ceiling, every guard and control, the Bonferroni and support quantiles, the allowance counts, the extrapolation-removed share and its arms, each
branch and the precedence of the verdicts, the quantile and bound arithmetic, the exit-status agreement), each of which must fail `scripts/tests/test_c004_aggregate.py`, the library tests or the rule-agreement tests, which pass unmutated.

**Run on the uncommitted tree before the freeze (no record written): harness plan 44 of 44 killed; aggregator plan 68 of 68 killed.** `results/mutations.json` is written by the head after the freeze, from the clean tagged tree.

## 6. Pilot disclosure and tuning (pilot seeds 1001 and up only; none is declared)

**Chronology (in order; every number is on a pilot seed, 1001 or higher; nothing was run on a declared seed).**

*Version 1.*

1. `planning.py` (margin 100 per mille, 14 seeds) was written first, from stated priors, and pinned. **It was written after one development run of the harness on the first task design** (the word problem of the group S3, seed 1001), in which the feed-forward stack did not learn (about a third accuracy);
   those numbers were seen. The planning model is task-agnostic (an accuracy margin and a between-seed standard deviation taken from the range of C001's contrasts) and was not changed afterwards. (Version 1 of this text, the header of `config.toml` and `planning.py` said "before any pilot"; that was wrong and is corrected.)
2. Exploration on seed 1002 showed that matched transformers (even one of 135000 parameters in 4000 steps) do not learn S3 in the CPU budget, so the stack could not serve as a comparator; **the task was changed to the modular counter with dense supervision**. The margin and the seed count were not touched.
3. A development run of the full harness on seed 1001 (counter task, learning rate 0.003 for all arms, default positional slope): recurrent 975, feed-forward 236 at 12 steps, and on the null task the stack reached 862 at 12 steps against 1000 for the recurrent arm. A positional slope scale was added to the stack's grid and a null-world eligibility clause to the selection rule.
4. Screening on seed 1002 (stack: depth {3, 4, 6} x slope scale {1, 16} x learning rate {0.001, 0.003}; then the null task), selected **3 layers, slope scale 16, learning rate 0.003** among the configurations eligible by the rule (850 in distribution and 900 on the null task at 12 steps); every 12-step accuracy of the stack was 160 to 275 (chance 167), so the selection was close to arbitrary. The grid was incomplete (depth 6 and scale 64 were not run on the null task). Pilots on seeds 1003 to 1007: recurrent 969 to 995 at 12 steps, stack 186 to 214.
   The first version's data are in `research/cseries/c004/pilots/data/before-fix-round/`.

*The review and the fix round.* An independent pre-freeze review found two blocking defects: the comparator guard (`comparator_primary_permille <= 800`) turned any competent stack into INVALID, so a refutation was unreachable; and nothing in the design let the main contrast fail (no control in which the stack is given the long length). It also found that the claim of an
"equal tuning grid" was false, that structural FALSIFIED preceded INVALID for missing evidence, that the chronology said "before any pilot" for a planning model written after a development run, that the doc-38 claims reached RPT-L1/RPT-L2, and several wording errors. The head's decisions on it are implemented here.

5. **Guard.** The saturation ceiling is now 980 per mille (a stack at or above it leaves no room for the margin); tests: a stack at 900 yields FALSIFIED (refuted), not INVALID; the competent-baseline table in `planning.py` (section 4) shows the outcome probabilities for stacks from 980 down to 250.
6. **Control.** A new arm `feedforward_inlength` (the stack trained with the 12-step length in its pool) and a judged control. **The criterion was changed once, after seeing the first exploration:** it was first written as "the contrast must fall below the margin" (recurrent minus the in-length stack below 100 at its upper bound); the explorations (table 1) showed the best the stack reaches when trained on the length
   is 0.72 at 4 layers, 0.83 at 5 and 0.86 at 6 against a recurrent arm at about 0.98, a gap of 0.12 to 0.26 that no depth closed within the cap, so that criterion would have made the experiment INVALID by construction. It became **"the stack trained on the length recovers more than half of the recurrent arm's advantage over the stack"** (lower bound of
   (in-length minus stack) minus one half of (recurrent minus stack) at least zero); one half is the smallest share at which more of the contrast is due to length than to anything else. This was **loosened after exploration (seed 1002), because the stack is optimisation-limited (empirical); the 1/2 threshold is a verbal a-priori argument, not from the planning model; depth 5 was chosen as the shallowest depth at which the in-length stack reaches 0.80, which is tuning toward the control** (at depth 4 the share would have been about 67 percent). It is made with the numbers in view; the absolute accuracy of the in-length stack is reported next to the control.
7. **Validation set of the control.** A first exploration validated the in-length stack on 6-step sequences only and stopped early (0.56 to 0.70 at 12 steps); its plateau protocol now validates on both the 6-step and the 12-step held-out sets.
8. **Explorations on seed 1002** (table 1): depth 4, 5, 6 of the in-length stack; learning rates; weight decay 0.1 and 1.0 (hurt); patience 30 with a 30000-step cap (no gain); shorter training lengths 2 to 5 with the primary length 10 (the stack's in-distribution accuracy 0.80, no easier: not adopted).
   A first pilot run with **slope scale 1** (seed 1005, then stopped) showed the stack at 0.85 on the null task at 12 steps against 1.00 for the recurrent arm: a contrast above the null-world margin, i.e. the check that exists for this would have fired. The slope scale was screened (scales 1, 4, 16; step 8 below). **Decisions: 5 layers, slope scale 4, learning rate 0.003 for the stack and its in-length control, 0.01 for the recurrent arm, warmup 300 steps for every arm.**
8b. **A product defect found in this round and fixed** (disclosed because it concerns every number above): the harness built every arm's network before seeding its random state, so the initial weights depended on the process's global random state and a run could not be reproduced (three identical short runs gave best-validation accuracies of 707, 689 and 754 per mille). The structural counter `nondeterminism` only compared the first twenty steps of two seeded trainings and could not see it. Every arm's network is now built from a generator seeded by
   (experiment, seed, arm), the counter also builds the arms twice under two different global seeds and compares digests (mutation `initial-weights-are-not-seeded`), and a test runs the harness twice and compares the records. **All exploration and screening rows in table 1 and the grids of the other experiments were produced before the fix and are one draw each from a random initialisation; the pilot seeds of table 2 were run after it.** C008's harness had the same defect and was fixed the same way; C007 seeds its networks before building them (a test confirms).
9. The pinned values were set and seeds 1003, 1004, 1005, 1006 ran with them; the dry run (seed 1007) went through the pinned entry point in the runner's stripped environment (the last row of table 2): no counter, rule or guard fired and every control resolved.

Table 1. Explorations and screens on pilot seed 1002 (permille; one run each; the rows without an arm of the stack are the recurrent screen). The first rows (files `explore1`, `explore2`) validated the in-length stack on 6-step sequences only (step 7).

| file | arm | layers | lr | overrides | in distribution | 9 steps | primary length (12; 10 where overridden) | steps | converged | cpu s |
|---|---|---|---|---|---|---|---|---|---|---|
| explore1 | feedforward | 4 | 0.001 | slope_scale=1.0 | 824 | 391 | 235 | 13250 | 1 | 163 |
| explore1 | feedforward_inlength | 4 | 0.001 | slope_scale=1.0 | 956 | 821 | 564 | 16000 | 1 | 227 |
| explore1 | feedforward | 4 | 0.003 | slope_scale=1.0 | 878 | 417 | 213 | 6250 | 1 | 242 |
| explore1 | feedforward_inlength | 4 | 0.003 | slope_scale=1.0 | 938 | 832 | 699 | 7500 | 1 | 326 |
| explore2 | feedforward_inlength | 4 | 0.003 | slope_scale=1.0 | 959 | 852 | 719 | 7500 | 1 | 115 |
| explore2 | feedforward_inlength | 5 | 0.003 | slope_scale=1.0 | 978 | 923 | 831 | 20000 | 1 | 322 |
| explore3 | feedforward | 4 | 0.003 | curve_extra_lengths=[7, 12], id_length=5, primary_length=10, secondary_lengths=[15, 20], train_lengths=[2, 3, 4, 5], slope_scale=1.0 | 803 |  | 175 | 7250 | 1 | 86 |
| explore3 | feedforward_inlength | 4 | 0.003 | curve_extra_lengths=[7, 12], id_length=5, primary_length=10, secondary_lengths=[15, 20], train_lengths=[2, 3, 4, 5], slope_scale=1.0 | 956 |  | 584 | 11000 | 1 | 148 |
| explore4 | feedforward_inlength | 4 | 0.003 | weight_decay_milli=1000, slope_scale=1.0 | 452 | 273 | 191 | 3500 | 1 | 58 |
| explore4 | feedforward_inlength | 4 | 0.003 | weight_decay_milli=100, slope_scale=1.0 | 819 | 598 | 391 | 6750 | 1 | 106 |
| explore4 | feedforward_inlength | 4 | 0.003 | max_steps=30000, patience=30, weight_decay_milli=100, slope_scale=1.0 | 961 | 847 | 714 | 23750 | 1 | 351 |
| explore5 | feedforward_inlength | 5 | 0.001 | slope_scale=1.0 | 971 | 903 | 828 | 17500 | 1 | 289 |
| explore5 | feedforward_inlength | 6 | 0.003 | slope_scale=1.0 | 983 | 945 | 856 | 17750 | 1 | 323 |
| screen-ff5 | feedforward | 5 | 0.003 | slope_scale=1.0 | 886 | 415 | 190 | 11000 | 1 | 161 |
| screen-ff5 | feedforward | 5 | 0.01 | slope_scale=1.0 | 913 | 575 | 206 | 10500 | 1 | 164 |
| screen-ff5 | feedforward | 5 | 0.001 | slope_scale=1.0 | 855 | 457 | 224 | 15250 | 1 | 219 |
| screen-inlen5 | feedforward_inlength | 5 | 0.01 | slope_scale=1.0 | 962 | 788 | 555 | 10500 | 1 | 170 |
| screen-rec | recurrent |  | 0.003 | - | 1000 | 993 | 967 | 3250 | 1 | 15 |
| screen-rec | recurrent |  | 0.01 | - | 1000 | 996 | 982 | 3250 | 1 | 28 |
| scale-inlen | feedforward_inlength | 5 | 0.003 | slope_scale=16.0 | 944 | 856 | 721 | 7250 | 1 | 127 |
| scale-inlen | feedforward_inlength | 5 | 0.003 | slope_scale=4.0 | 982 | 934 | 860 | 12250 | 1 | 206 |
| scale-null | feedforward | 5 | 0.003 | slope_scale=4.0 | 941 | 620 | 294 | 9000 | 1 | 183 |
| scale-null | feedforward_null | 5 | 0.003 | slope_scale=4.0 | 1000 | 996 | 999 | 3500 | 1 | 183 |
| scale-null | feedforward | 5 | 0.003 | slope_scale=1.0 | 860 | 437 | 192 | 10250 | 1 | 208 |
| scale-null | feedforward_null | 5 | 0.003 | slope_scale=1.0 | 999 | 955 | 900 | 4250 | 1 | 208 |
| scale-null | feedforward | 5 | 0.003 | slope_scale=16.0 | 947 | 534 | 212 | 9750 | 1 | 193 |
| scale-null | feedforward_null | 5 | 0.003 | slope_scale=16.0 | 1000 | 999 | 1000 | 3500 | 1 | 193 |

Table 2. Pinned configuration on pilot seeds (permille; the last row is the dry run).

| seed | rec id | rec 12 | rec 24 | ff id | ff 12 | ff 24 | ff trained on 12: 12 | ff trained on 12: id | reset 12 | rec corrupted 12 | rec null 12 | ff null 12 | cpu s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1003 | 1000 | 995 | 974 | 920 | 213 | 153 | 837 | 972 | 168 | 844 | 1000 | 997 | 522 | 4520 |
| 1004 | 999 | 978 | 896 | 923 | 178 | 171 | 772 | 970 | 173 | 836 | 1000 | 997 | 540 | 4589 |
| 1005 | 1000 | 986 | 880 | 936 | 184 | 170 | 905 | 986 | 168 | 837 | 1000 | 999 | 675 | 5082 |
| 1006 | 1000 | 982 | 880 | 910 | 232 | 170 | 911 | 989 | 166 | 836 | 1000 | 997 | 689 | 5114 |
| 1007 | 1000 | 985 | 906 | 924 | 295 | 172 | 889 | 981 | 175 | 840 | 1000 | 998 | 605 | 5392 |

What the pilots say, including what is unwelcome: (a) **the stack never extrapolated, in any configuration tried** (12-step accuracy 178 to 295 here, 0.19 to 0.22 in the screens, chance 167), although it learned the training lengths to 910 to 936: the comparison is recurrence against a non-extrapolating class, as stated at the top;
(b) the stack trained on the long length reaches 772 to 911 at 12 steps (the recurrent arm 978 to 995), so the stack is optimisation-limited even there; (c) the reset-state ablation sits at chance (166 to 175), as built; (d) the pilot contrasts are far above the margin, so the pilot power table is not
evidence about discrimination.

## 7. Limits

* A result applies to this task family (a modular counter in the cyclic group of order six), to training on 2 to 6 steps and to evaluation at 12 steps (secondary 18 and 24), to a GRU of 63174 parameters
  and to the matched causal attention stack described in section 3. It says nothing about language, about the experts of an OLMoE-class model or about recurrent state on a real model (the GPU
  stage B is not registered).
* **The task was changed once after pilots.** The first design used the word problem of the group S3 (the product of a sequence of group elements). A feed-forward stack of the matched size did not learn it in the
  CPU budget even in distribution (about a third accuracy after 8000 steps; a stack of 135000 parameters did no better in 4000), so it could not serve as a comparator, and the task became the modular counter. The planned margin and seed count do not depend on the task. This is a choice made to give the comparator a fair chance; a reader who thinks the
  harder problem is the right test should read this result as one about the easier problem.
* **The comparator class does not extrapolate, and a COMPLETED is close to a known answer.** The stack was given standard relative positions, the depth and slope scale at which it can hold the twelve-step answer when trained on that length (to 0.86, not to 1.0), the recurrent arm's warmup and plateau rule and a learning-rate grid; it never extrapolated: its 12-step accuracy is 0.19 to 0.22 (chance 0.167) in every configuration of table 1 trained on the short lengths, at every depth. A stack with a longer schedule, a curriculum, a different positional scheme or far more parameters might. The experiment therefore tests recurrence against a non-extrapolating architecture class and a COMPLETED is close to a known answer.
* The stack's FLOPs are matched at six steps only: at 12 steps and more it costs more than the recurrent net (reported), which favours the comparator.
* The reset-state ablation is a floor by construction (it sees one step): beating it shows only that the recurrent arm uses its state.
* The null world with the sum of the last two steps is easy for the recurrent arm; the null check exists to expose a positional-extrapolation
  failure of the stack that would confound the main contrast. If it fires, the experiment is INVALID and the reading is that this stack cannot extrapolate even a local task, which is itself a finding about the comparator.
* The extrapolation-removed control asks for a share of one half, a verbal a-priori argument made after exploration and not a quantity of the planning model (loosened after exploration, see section 4), and is read only for a supported contrast; the absolute accuracy of the stack trained on the long length is reported next to it.
* The t bounds assume roughly normal paired differences, which 14 points cannot check; per-seed differences, medians and both bounds are reported.
* An indicator property here is a functional property: the experiment says nothing about whether any system is conscious.

## 8. Entry point, environment and freeze procedure

The entrypoint is `env UV_PROJECT_ENVIRONMENT=../../../c001-venv PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline --no-python-downloads
--project research/consciousness python -B research/cseries/c004/harness.py --benchmark c004 --seed <seed>`, and the manifest declares the matching
`preflight` command (the same program and environment with `--preflight` instead of `--seed <seed>`; the runner runs it, in the entrypoint's exact environment,
before it reserves a seed, and `python3 scripts/run_experiment.py preflight C004` runs only it). Reasons, the same as C001's (found there by rehearsing the
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
the thread counts). **All 14 seeds run on one host in one environment:** the aggregator marks a seed invalid whose recorded environment differs from the first one. A seed takes
606 CPU seconds (section 3), so the 14 seeds take about 2.4 hours of CPU time; on the loaded shared machine of the pilots a seed took 75 to 90 minutes of wall time (4520 to 5392 s), so the budget of 15 minutes per seed is met in CPU seconds only.



**Freeze procedure.** Every step is a command in the repository; nothing refers to a script outside it. The two sha256 values of the `[preregistration]` table and the manifest digests read
`must-be-pinned-before-prepared` (absent) until step 4. Run every `python3` call with `export PYTHONDONTWRITEBYTECODE=1`. Commit messages carry the attribution trailer the session requires.

1. **Clean the tree and commit it** (the pre-freeze commit). The commit must include the untracked `research/cseries/`; `git status` shows new files as untracked until added.
   `find . -name __pycache__ -type d -prune -exec rm -rf {} +`; the repository must hold no git-ignored file when the runner starts.
2. **Check, before the digests are pinned:** `python3 scripts/run_experiment.py validate`; `python3 scripts/check_research_gates.py`; `python3 scripts/mutation_check.py C004 --check`;
   `python3 research/cseries/lib/run_agg_mutations.py C004 --check`; `python3 -m unittest scripts.tests.test_c004_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests scripts.tests.test_ci_local`;
   the research tests with the locked interpreter, `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B -m unittest discover -s research/cseries/tests`; `python3 scripts/update_component_docs.py --check`; `python3 scripts/check_repo.py`.
3. **Run the aggregator mutation plan** (seconds): `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_agg_mutations.py C004 --write`; every defect must be killed; commit the written `tests/aggregator-mutations.json`.
4. **Pin the digests:** `python3 research/cseries/lib/pin_freeze.py C004` (it sets `preregistration_text_sha256` and `criteria_sha256` in `config.toml`, the two manifest digests and `status = "prepared"` in
   `experiment.toml`, and `status = "prepared"` in `experiments/registry.toml`; it refuses a tree that is not a planned, unpinned one; `--dry-run` shows what it would write). Then `python3 scripts/run_experiment.py validate`,
   `python3 scripts/check_research_gates.py`, `python3 -m unittest scripts.tests.test_c004_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests`; commit ("freeze preregistration") and tag it `c004-prereg`.
   The mutation evidence must run on this committed tree: `aggregate.py` refuses if `research/cseries` or this directory differs from the commit the mutation record names.
5. **Mutation evidence of the harness**, from the clean tagged tree: `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_mutations.py C004` (writes `results/mutations.json`); commit it.
6. **Prepare and run:** `python3 scripts/run_experiment.py preflight C004`, `python3 scripts/run_experiment.py prepare C004`, and commit; then for each seed 5001 to 5014, once each:
   `python3 scripts/run_experiment.py run C004 --seed <s>` and **commit the new run record before the next seed**. A seed whose run exits 2 is not repeated by hand: stop and report; the experiment is INVALID and the repair is
   a successor registered with `scripts/new_experiment.py` (a prepared experiment is never patched).
7. **Aggregate:** `python3 experiments/consciousness/C004-recurrent-integration/aggregate.py --report report.md`; then `python3 scripts/check_research_gates.py` and `python3 scripts/run_experiment.py validate`; advance the
   registry status by hand only for COMPLETED or FALSIFIED, as `recommended_status` says.
