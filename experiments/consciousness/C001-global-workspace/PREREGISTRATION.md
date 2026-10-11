# C001 global workspace: preregistration (v3)

Status: preregistration of version 3. Version 1 of this experiment (position-specific modules, held-out position pairs, online data, five declared seeds
17/29/43/71/101, a lower-bound-only decision) was withdrawn before any freeze after review: its outcome was fixed by a ceiling and the
position-pair split was not out of distribution. Version 2 (shared weights, finite data, 16/24-module evaluation, ten seeds 3001-3010) was
withdrawn before any freeze after a second review: its operating point (4096 examples) had been pinned against the stated selection rule
at the size where the workspace advantage appears, the tuning was asymmetric (the workspace structure was fixed from earlier pilots while
the comparators got a two-point learning-rate grid on two seeds selected by in-distribution accuracy), arms were not trained to a plateau,
the judged comparators lacked a sparse-attention baseline and the bottleneck ablations were not judged, and the sensitivity control could not
show that a difference of the margin's size is resolvable. Version 3 repaired those points on **pilot seeds 1004-1008** (never declared) and
re-declared fresh confirmatory seeds (twenty: 4001-4020, the largest number a CPU budget allows; see section 4). A third review of v3 found that
the convergence guard covered too many arms, that the pinned no_broadcast arm was not the tuned arm, and that the margin and the support
conjunction made support unreachable; **this text is v3 after those repairs and after the further reviews described in section 6**: the step cap is 20000, the guard covers the workspace
and the three pairwise comparators with a tolerance of two seeds, every arm runs at d = 48, no_broadcast leaves the support conjunction for the
reported attribution contrasts, and the margin is 30 per mille with a stated cost rationale (fixed before the freeze, after the pilots; disclosed). A fourth review found that the graded sensitivity control and the outcome probabilities rested on five tuning seeds that the two out-of-sample dry seeds contradicted; **round 6** recalibrated the graded control on fresh pilot seeds (the weak arm is now the all-to-all net with Adam weight decay 0.001), added the INVALID branch to the power estimate and replaced the single expected outcome by a range (sections 4 and 6). The dry runs of versions 1 and 2 (seeds 17 and
1003) and the v3 dry runs (seeds 1009 and 1010) are disclosed; the seeds 3001-3010 of version 2 were never run and are NOT reused (the declared
seeds of v3 are listed in `config.toml`). The pinned values are `[pinned_thresholds]` in `config.toml`; `criteria.toml` holds the rules; this file
states the reasons and the limits.

## 1. Scope and non-claims

C001 is a functional-mechanism test on a CPU toy task of whether a **learned-attention** global workspace (the
mechanism of Goyal et al., ICLR 2022: a few slots, competitive write, gated slot update, broadcast read) has an advantage
as an inductive bias. **It is not a test of consciousness and no result is evidence about consciousness.** No
self-report is produced, trained or measured.

**Relation to `ptr-pods::workspace`.** The workspace implemented in `crates/ptr-pods/src/workspace/` (package A) is a
deterministic contract: bounded slots, proposals that pass the verifier gate compete by bid and then by a SHA-256
tie-break, snapshots are broadcast to bounded mailboxes, findings are broadcast as refusals. It learns nothing. C001 tests a
learned analogue (attention write, GRU slot update, attention read). A supportive C001 does not validate that contract and a
refuting C001 does not touch it; the contract's evidence is its own tests and C006 on Pods. C001 informs one design
question only: whether a *learned* write and read are worth building on top of the contract (GW-L1/GW-L2, doc 38).

## 2. Hypothesis and the claim being tested

Goyal et al. claim an inductive-bias advantage (sample efficiency, generalisation, linear cost in modules), not an
expressivity advantage: an all-to-all attention can express everything the workspace can. So the test is at an
operating point where inductive bias can matter: a finite training set and more modules at test time than at training.

H1. With a finite training set of eight-module instances, trained to a plateau, the workspace arm is more accurate than each of the
all-to-all, the bandwidth-matched all-to-all and the sparse (top-k) all-to-all arms on instances of **sixteen** modules (more distractors than
ever seen in training), by at least the pinned margin (30 per mille), shown by the lower one-sided 95% t bound over twenty seeds. Three further
questions are **reported, not judged** (section 4): does the workspace beat no_broadcast (the broadcast read, GWT-3), capacity_off (8 slots: the
limited-capacity bottleneck, GWT-2) and competition_off (soft write: competition), each by the same margin. Each is reported as attributed, not
attributed or unresolved by noise.

## 3. Task, arms, training

* N_train = 8 sensory modules **sharing one encoder and one update network** (plus a reporter module with no input);
  there is no position information, so the number of modules is free at test time. Each module sees a noisy observation
  (type one-hot over 4 types, value one-hot over 3 values, cue flag, query one-hot, Gaussian noise 0.3). One module is the
  cue: it carries the query type q. Exactly two other modules have type q, the others have another type. The answer is the
  sum of the two values of type q (5 classes; the majority class has probability 1/3). A single module's value does not
  determine it, and which modules matter is known only after the cue's content has reached the others.
* Evaluation sets (2048 fresh instances each): in distribution (8 modules, a secondary check), **primary: 16 modules**, secondary:
  24 modules (same rule, more distractors; a real change of the input size, unlike a held-out pair of positions).
* Data: a finite training set of **2048** eight-module examples generated once per seed, cycled; every arm trains on the same
  set in the same order (paired). Adam, batch 64, gradient clip 1.
* **Plateau protocol (every arm, identical).** Training stops when the accuracy on a held-out in-distribution validation set of 1024
  instances (drawn from the seed's evaluation stream, never used for testing) has not improved by 5 per mille for 12 consecutive
  evaluations (every 150 steps), or at **20000 steps**; the best-validation checkpoint is restored. An arm is *converged* if it stopped by
  patience. The comparison is of plateau accuracy, not of optimisation speed: the workspace or a pairwise comparator at the step cap on
  more than 2 of 20 seeds makes the experiment INVALID (guard `unconverged_judged_arms`, section 4). The cap was raised from 9000 to 20000 in
  round 4: runs that stopped by patience are bit-identical under any larger cap, and the nine pilot runs that had hit 9000 were rerun
  (section 6). The patience was raised from 6 to 12 after a pilot showed the workspace stopping before a late jump ("ignition"); this was
  done on pilot seeds before the equal grid below. The harness audits the loop: the restored weights must reproduce the best validation
  accuracy and the convergence flag and step count must match a recomputation from the recorded curve (integrity counters, section 4);
  the validation set is disjoint from every evaluation set (integrity counter `val_eval_overlap`).
* **Equal tuning grid.** Every arm that is judged or used as an ablation was tuned on the same grid: learning rate {0.001, 0.003} x update
  width d {32, 48}, then, because every arm had selected the upper edge 0.003, extended for every arm by lr 0.006 at d 48 (the d = 32 cell at 0.006 was
  not run for any arm; CPU budget) and, because four arms then selected 0.006, the new edge, for exactly those four arms by lr 0.012 at d 48
  (section 6; the rule is "bracket every arm"), on the primary metric (accuracy on 16 modules) averaged over the five pilot seeds 1004-1008 (section 6). The
  workspace structure (3 slots, top-3, 3 rounds) is held fixed because it is the claim; the comparators' structure is their definition.
  **Parameter-matching rule: every arm runs at d = 48, the width the workspace selected, and its update-network width `ffn` is chosen so that
  its parameter count is the closest to the workspace's at d = 48 (the target is the workspace; ties to the smaller; tolerance 2%).** The
  pinned arm is therefore the tuned arm: in the grid each cell's `ffn` was matched to the workspace at that cell's width, and at d = 48
  no_broadcast has ffn 64, exactly the workspace architecture without the read path. Selected learning rates: 0.003 for the workspace, no_broadcast,
  capacity_off and no_comm (floor, untuned); 0.006 for all_to_all, bw_all_to_all, sparse_all_to_all, competition_off and the weak and crippled
  controls (which inherit all_to_all's). Every selected rate is interior to the tested grid at d = 48: 0.003 is bracketed by 0.001 and 0.006
  for the workspace, no_broadcast and capacity_off, and 0.006 by 0.003 and 0.012 for the four arms that selected it (section 6).
  **Width: d = 48 is the upper edge of the width grid {32, 48} for six of the seven tuned arms (all but no_broadcast, a tie), and d = 64 was
  not tried (CPU budget).** The gains from d = 32 to d = 48 at lr 0.003 were workspace +54, competition_off +36, bw_all_to_all +40,
  sparse_all_to_all +18, all_to_all +11, capacity_off +6 per mille, so the width optimum of some arms may lie above 48; the workspace gains
  most from width, which hints that a wider grid could favour it, but the direction of the bias is unknown (the comparators' optima could
  move too), and it is a limit of this experiment (section 7).
* Arms (parameter counts within 2% of the workspace arm by choosing the update-network width `ffn`; communication-cost proxies
  and message width are reported):
  * workspace: 3 slots, per-slot top-3 competitive write, GRU slot update, broadcast read by every module;
  * all_to_all: single-head pairwise attention among all modules;
  * **bw_all_to_all**: pairwise attention whose value width is reduced so that the values stored (N x d_value) equal the
    workspace's slot content (3 x 48 = 144 = 8 x 18 at d = 48), isolating "shared store" from "pairwise" at matched bandwidth;
  * **sparse_all_to_all**: pairwise attention in which every query keeps only its top-k keys (k = number of slots = 3): a sparse
    comparator with the workspace's per-reader fan-in but no shared store (it still scores every key, so its cost proxy is that of
    all_to_all);
  * no_broadcast: the same workspace (d = 48, ffn 64), but only the reporter reads it back (GWT-3 contrast);
  * no_comm: no communication (a floor);
  * competition_off: the workspace with soft write over all modules (no top-k); capacity_off: as many slots as training modules (8);
    these two and no_broadcast are **reported attribution contrasts** (section 4: attributed / not attributed / unresolved by noise), not
    conditions of the main claim;
  * crippled_all_to_all (one round instead of three) and **weak_all_to_all** (the all_to_all net, three rounds, trained with Adam weight decay
    0.001 instead of 0; `arm_weight_decay` in `config.toml`): the **sensitivity controls**, comparators handicapped in a known direction,
    by a gross and by a graded amount. The weak arm was two rounds until round 6 and was replaced because that deficit was unstable (section 6).
* Module dropout, ignition-style curves, learning curves (area under them = sample efficiency) and the communication
  cost per module count are descriptive.
## 4. Decision rule (frozen in criteria.toml) and what each outcome means

Per seed: **structural** coverage rules and six structural counters (no communication leak in the isolated arms, write attention with at most
k sources, dropped modules cannot influence an output, determinism, a no-communication arm that does not beat the majority class). A failing
rule is a defect of an arm and FALSIFIES (kind structural); **a structural FALSIFIED means a defective arm and is a statement about the
implementation, not about the hypothesis**, and the repair is a successor experiment. Seven **integrity** counters are defects of the task or
of the harness, not of the product: parameter match, no out-of-distribution module count in any training batch actually drawn, no evaluation
instance identical to a training instance or to a validation instance, finite losses, the restored weights reproduce the best validation
accuracy, and the recorded convergence flag and step count agree with an independent recomputation of the stopping rule from the curve. The
harness exits 2 on any of them and the experiment is **INVALID**, never FALSIFIED (exit 1 is for the product, exit 2 for the harness). The
negative controls run on doctored inputs or on the all-to-all net, never on the arm under test. Six controls must trip or the run is exit 2.

**Validity guards** (the aggregator, not the harness; a failing seed is uninformative, the experiment is **INVALID**, never FALSIFIED):
the best of the three pairwise comparators on the primary metric must lie in [300, 900] per mille on every seed (saturated or floored
otherwise); **the workspace and the three pairwise comparators must have stopped by patience, not at the step cap, on all but at most 2 of 20
seeds** (the same tolerance as for the workspace's training; an arm at the cap would make the comparison one of optimisation speed; the
attribution arms are not guarded, and an unconverged attribution arm is reported); the workspace arm must have trained (ID accuracy at least
100 per mille above the no-communication arm: the harness rule `rules.workspace_trained`, the only definition) on at least 16 of 20 seeds (at most 4 untrained; see below); and **both sensitivity controls must resolve**: the lower bound of
(all_to_all minus crippled_all_to_all) must reach 50 per mille (gross: the pipeline separates a floor from a working arm) and the lower bound of
(all_to_all minus weak_all_to_all, the same net with weight decay 0.001, a deficit of about 65 per mille on the pilot seeds) must exceed 15 per mille
(graded: the pipeline detects a deficit well below the crippled arm's 400 and about twice the margin). **The graded control is a hard gate**:
without it a NOT_SHOWN could not be told from an insensitive pipeline, so it has to be reliable; section 6 gives its calibration and the pilot-estimated
probability that it fails (0.005 on 20 calibration and confirmation seeds, 0.03 on all 17 seeds that have every arm). The graded deficit's per-seed standard deviation
(51 over the 20 calibration and confirmation seeds; gross control 22) is of the size of the contrasts' (43 / 36 / 72 for the three pairwise ones over the 16 pilot seeds on
which the workspace trained), so "the control resolves" shows that a deficit of about 65 per mille is resolvable at this noise; it is a necessary, not a sufficient, sign that a
difference of the margin's size is.

**Margin: 30 per mille of 16-module accuracy (about three points).** Rationale, independent of the pilot means: building the learned write
and read (GW-L1, GW-L2) adds a second attention path with its own slots, top-k, rounds and a training-stability burden, and a Pod prototype
(C006). A gain below about three points of accuracy at 16 modules is, in our judgement, not worth that block; at or above it, building is
justified. **Disclosure:** this value was chosen after the pilots had been seen (a post-hoc choice, though not fitted to them). An earlier version of this text
used 50, "the size of the pilot gap to all_to_all"; that was wrong, because it put both decision boundaries (support and refutation) on a
selection-inflated point estimate: at the pilot value the lower bound against all_to_all is about 47 - 11 = 36, so even a pairwise-only support
had a probability of about 0.002. A margin must be a number the pilot cannot move.

**Statistics.** Over the twenty seeds the one-sided 95% t quantile for 19 degrees of freedom is 1.729133. Judged comparators: all_to_all,
bw_all_to_all, sparse_all_to_all. A refutation uses the Bonferroni quantile over the three (alpha 0.05/3, 2.294392): **it means that the
workspace advantage is below the margin, not that there is no advantage.** The attribution contrasts (workspace minus no_broadcast, minus
capacity_off, minus competition_off) use the one-sided 95% quantile on each side and are **three-valued**: *attributed* (lower bound at or
above 30), *not attributed* (upper bound below 30) or *unresolved by noise* (neither); the half-width, the median and the number of seeds with
a positive difference are printed beside the bounds, because the per-seed differences are heavy-tailed (371 in one pilot seed; no_broadcast
swings from -58 to 73). They never decide the verdict. A *not attributed* broadcast contrast downgrades the broadcast claim (GWT-3) for this
setting, never the workspace claim; *broadcast attributed but the pairwise advantage not shown* is the reported partial pattern (it supports
"communication matters", not "the bottleneck helps").

| Outcome | Condition | Consequence |
|---|---|---|
| COMPLETED (supported) | guards and both controls hold; every lower bound (t 1.729133) against the three pairwise comparators >= 30 per mille | Build the learned L2 prototype; run C006 on Pods. The broadcast, bottleneck and competition contrasts are reported three-valued. **COMPLETED shows an advantage over pairwise attention at 2048 training examples and 16 modules, nothing more** (see below) |
| FALSIFIED (refuted) | guards and controls hold and, for some pairwise comparator, the UPPER bound (t 2.294392) < 30 per mille: the advantage is below the margin | Record a negative result for this setting (2048 training examples, 16 modules); downgrade the learned-workspace requirements GW-L1 and GW-L2 to exploratory **for that setting: a successor at a larger training set (the pilot advantage grows with D, section 6) is not excluded** |
| NOT_SHOWN | guards hold, the bounds straddle the margin | Nothing is concluded; status stays planned; no downgrade; a larger successor may follow |
| INVALID | out of band, an unconverged guarded arm on more than 2 seeds, insensitive control, workspace untrained on more than 4 seeds, any integrity counter, seeds or mutation evidence missing | Nothing is concluded; the status stays planned; repair and register a successor |

Exit status of `aggregate.py`: 0, 1, 3, 2 respectively (2 also for any refusal to aggregate). After NOT_SHOWN or INVALID the registry
status is not advanced and no claim in `research/novelty/CLAIMS.md` changes. (`scripts/run_experiment.py`, `scripts/check_research_gates.py`
and `experiment_records.publish_aggregate` read neither the verdict string nor the aggregator's exit status; a test publishes a NOT_SHOWN
aggregate with no recommended status and passes the gate's aggregate check. The registry status is advanced by hand.)

**Power and expected outcome (computed before the freeze, `research/consciousness/gw/power.py`; Monte Carlo of the final rule, 20000 trials per row,
seed 0, n = 20).** The model draws per seed the three contrasts (workspace minus all_to_all, bw_all_to_all, sparse_all_to_all), the graded deficit (all_to_all minus
weak_all_to_all) and the gross deficit (all_to_all minus crippled_all_to_all) from a multivariate normal fitted to the 16 pilot seeds on which the workspace
trained (so the contrasts, which share the workspace, are correlated, as they are in the data; an independent model was used up to round 5); a seed on which
the workspace does not train occurs with probability p_fail and carries the contrasts (-394, -424, -355) of the one failed pilot seed. The verdict follows
`aggregate.decide`: INVALID when the graded control's lower bound does not exceed 15, the gross control's does not reach 50, more than 4 seeds have an untrained
workspace, or more than 2 an unconverged guarded arm; otherwise FALSIFIED, COMPLETED or NOT_SHOWN. The comparator band guard is not modelled (no pilot comparator came near it).

**Finding of round 6, and what the guard counts (round 8).** The workspace *stalled* on one of the 17 pilot seeds that have every arm (seed 1011: 16-module accuracy 367, in-distribution 437, chance
333, converged by patience after 4350 steps): p_stall = 1/17 = 0.059 (exact 95% interval 0.002 to 0.29). On such a seed every contrast is about -400, which lowers each 20-seed mean by about 20 per mille and inflates the standard deviation.
**The guard's definition is the harness rule** `rules.workspace_trained`: the workspace's in-distribution accuracy is at least the no-communication arm's plus `trained_margin_permille` = 100. no_comm was measured on every pilot seed that lacked it
(`pilots/calib6/nocomm-*.jsonl`, `pilots/summary.txt`): **under that rule none of the 17 pilot seeds is untrained, and seed 1011 is not flagged** (workspace in-distribution 437, no_comm 333, difference 104, the rule needs 100; every other seed
has a difference of 470 or more). A stall like 1011 therefore does not count toward the guard; it simply stays in the paired differences, which is the conservative direction (it pushes toward NOT_SHOWN or FALSIFIED, never toward COMPLETED).
**The rule was not changed.** Its purpose is to detect a harness that did not train the workspace (in-distribution accuracy at the floor of the no-communication arm), and seed 1011 is not that: the workspace left the floor by 104 per mille and the three
pairwise comparators trained on the same data; requiring 16-module accuracy above chance plus a margin as well would be a change prompted by this one seed's outcome, and a stalled-but-trained seed is exactly what the paired comparison is meant to absorb. The guard
value stays at 4 of 20 (below). Earlier power tables (rounds 6 and 7) had counted stalled seeds toward the guard (variant G below); the harness would not, so variant S is the one that matches the harness.
Pilot paired differences (workspace minus all_to_all / bw_all_to_all / sparse_all_to_all, mean / sd):
tuning seeds 1004-1008 (selected the comparators' rates; 31 / 24, 47 / 42, 89 / 64); the two dry seeds 1009 and 1010 (129 / 10, 62 / 32, 91 / 23); the **fresh out-of-sample pilot
seeds 1011-1020**: all ten 4.8 / 144, 6.6 / 155, 65 / 168 (per seed with the failed seed first: all_to_all -394, -3, 10, 49, 116, 75, 84, 39, 35, 37; bw_all_to_all -424, 17, 43, 43, 88, 68, 119, 26, 79, 7; sparse_all_to_all
-355, 37, 119, 64, 169, 181, 9, 279, 58, 93), the nine on which the workspace trained **49 / 37, 54 / 37, 112 / 85**; the 16 trained seeds together 54 / 43, 53 / 36, 102 / 72.

| scenario (the contrast means of seeds on which the workspace does not stall) | p_stall | COMPLETED | NOT_SHOWN | FALSIFIED | INVALID |
|---|---|---|---|---|---|
| tuning-seed pilot means (5 seeds; 31 / 47 / 89) | 0 | 0.063 | 0.925 | 0.011 | 0.000 |
| | 0.059 | 0.017 | 0.968 | 0.015 | 0.000 |
| fresh-seed means 1011-1020, trained seeds (49 / 54 / 112) | 0 | 0.579 | 0.408 | 0.000 | 0.013 |
| | 0.059 | 0.169 | 0.818 | 0.001 | 0.012 |
| means pooled over all 16 trained pilot seeds incl. the dry seeds (54 / 53 / 102) | 0 | 0.687 | 0.294 | 0.000 | 0.019 |
| | 0.059 | 0.203 | 0.778 | 0.001 | 0.019 |
| zero effect | 0 | 0.000 | 0.027 | 0.954 | 0.019 |
| | 0.059 | 0.000 | 0.119 | 0.862 | 0.019 |
| effect = margin (30 everywhere) | 0 | 0.000 | 0.934 | 0.046 | 0.019 |
| | 0.059 | 0.000 | 0.924 | 0.057 | 0.019 |
| effect 45 everywhere | 0 | 0.078 | 0.901 | 0.002 | 0.019 |
| | 0.059 | 0.024 | 0.949 | 0.009 | 0.019 |
| effect 80 everywhere | 0 | 0.896 | 0.084 | 0.000 | 0.019 |
| | 0.059 | 0.265 | 0.717 | 0.000 | 0.019 |

The controls' means in every row are those of the 16 non-stalled pilot seeds (graded 69, gross 398). In variant S (the stalled seeds are drawn with probability p_stall and are not counted by the guard, as the harness rule does for 1011) P(INVALID) is 0.02 in every row, all of it from the controls
(the pilot-estimated probability that the graded control does not resolve is 0.018 bootstrap / 0.032 normal on all 17 seeds, and 0.001 / 0.005 on the 20 calibration and confirmation seeds). If such seeds were counted by the guard (variant G, the table of round 7) the rows change by at most 0.004 in P(INVALID) at p_stall 0.059 and
by 0.08 at 0.12 (INVALID 0.100). Pooled non-stalled means with p_stall 0 / 0.03 / 0.059 / 0.12 (variant S): COMPLETED 0.687 / 0.373 / 0.203 / 0.053, NOT_SHOWN 0.294 / 0.609 / 0.778 / 0.900, FALSIFIED 0 / 0 / 0.001 / 0.029. Each seed unconverged with probability 0.02 adds 0.007 to P(INVALID).

**Untrained-workspace guard: 4 of 20 (decision of round 7, 2026-10-10, made from pilot data only and without looking at any workspace contrast beyond those reported in round 6; kept in round 8 after the harness-rule recount).** The allowance was 2 in
rounds 1-6. Rationale (written before the freeze, from pilot data only): seeds that the harness rule flags as untrained **stay in the paired differences with their contrasts**, so they count against the workspace hypothesis (conservative: they push toward NOT_SHOWN
or FALSIFIED and never toward COMPLETED); the guard's only job is to detect a broken training harness, not to protect the claim. Tails, P(more than k of 20 seeds flagged), at flag rates 1/17 = 0.059 / 0.12 / 0.18 (the upper 95% end for the observed 0 of 17) / 0.29: for k = 4, 0.005 / 0.083 / 0.285 / 0.73;
for k = 2, 0.11 / 0.44 / 0.73 / 0.96. The observed harness-rule rate is 0 of 17, so neither value is exercised by the pilots; 4 is kept because a stalled seed sits close to the rule's threshold (seed 1011: 104 against 100), so a seed like it may fall either side, and at the stall rate 1/17 an
allowance of 2 would be exceeded with probability 0.11 by chance alone whereas 4 is exceeded with 0.005; a real harness defect would flag many more than 4 seeds.

**Expected outcome, honestly: no single outcome is expected.** The old statement ("at the pilot means the expected outcome is NOT_SHOWN, 0.95") is withdrawn: it
rested on the five tuning seeds, which also selected the comparators' rates, and ignored INVALID. Reading the pilots three ways gives P(COMPLETED) from **0.02 to 0.69**:
the tuning-seed means give NOT_SHOWN 0.93-0.97; the fresh out-of-sample seeds give COMPLETED 0.17 (at the observed stall rate) to 0.58 (if the workspace never stalled), NOT_SHOWN 0.41-0.82; all
trained pilot seeds give COMPLETED 0.20 to 0.69. At the best pooled estimate and the observed stall rate (variant S): **NOT_SHOWN 0.78, COMPLETED 0.20, INVALID 0.02, FALSIFIED about 0**; if the workspace never stalled: COMPLETED 0.69,
NOT_SHOWN 0.29. FALSIFIED is likely only if the true advantage is near zero (0.86 to 0.95). An advantage equal to the margin gives NOT_SHOWN 0.92-0.93 and FALSIFIED 0.05-0.06 (the
false-refutation rate at the boundary is about alpha); support needs a true advantage of roughly 2.5 times the margin (80 per mille: COMPLETED 0.90 without stalls, 0.27 at the observed rate). The earlier remark that the pilot
means are "selection-inflated" in the workspace's favour is replaced by: **direction unknown**. The comparators were selected on six cells and the workspace on five on the same five seeds, and the
out-of-sample seeds (1009-1020) show the comparators weaker (all_to_all fell from 754 on the tuning seeds to 675 on the dry seeds and 742 on the fresh ones) and the workspace sometimes stalling, so winner's curse acts on both sides. n = 20 is the most this CPU budget allows (about 7 minutes per seed); the power result does not allow fewer seeds, and fewer would be weaker.

**What COMPLETED does not mean.** The support conjunction contains the three pairwise comparators only. A COMPLETED can therefore occur
with the broadcast contrast (no_broadcast), the bottleneck contrast (capacity_off) and the competition contrast (competition_off) each
*not attributed* or *unresolved by noise*: the pilot gave no sign that the workspace beats competition_off (-4 +/- 24) or clearly beats
no_broadcast (19 +/- 50), and it may be the shared store, or the 3-slot write alone, that wins. The claim COMPLETED licenses is *a learned
workspace of this form beats parameter-matched pairwise attention at D = 2048, 16 modules, by the margin*; that the broadcast read, the
limited capacity or the competition is the reason is licensed only by the attribution table, contrast by contrast, and each of them can
read "not attributed". **Post-pilot disclosure:** an earlier draft had no_broadcast in the support conjunction; it was moved to the reported
contrasts after the third review, when the pilot gap against it was 17-19 +/- 50-71 per mille and against competition_off -4, i.e. a change
made after the pilot results were known that removes the arms the pilot says the workspace does not clearly beat. It makes COMPLETED easier
to reach and is therefore the reason for the explicit statement above.

**Patterns, stated in advance.** *Supports:* the guards hold, the workspace beats the three pairwise comparators by the margin on 16 modules;
the contrasts with no_broadcast, capacity_off and competition_off are reported with their state; (reported, not judged) the advantage on 24
modules, the slot behaviour. *Refutes (for this setting):* the guards hold, both controls resolve, and the workspace advantage over a pairwise
comparator is below the margin with power. *Partial (NOT_SHOWN, reported):* broadcast attributed but the pairwise advantage not shown.
*Uninformative (INVALID):* saturation, floor, too many seeds with an unconverged guarded arm, an insensitive control, an untrained workspace, a
harness-integrity counter.
## 5. Mutation evidence

`tests/mutations.toml` plants 38 defects, each naming the counter, negative control or unit test it must trip. A defect is killed in one of three
ways: the reduced-scale harness exits 1 with an expected structural counter (defects of the workspace arm must reach a counter, never exit 2);
it exits 2 with an expected integrity counter or negative control named on its stderr; or the named unit tests, which pass in the unmutated copy,
fail in the planted one (defects in the pure derivations of the guard scalars in `rules.py`, in the definitions of the weak and crippled
controls, in the weight decay the optimiser gets, and in the stopping rule of `train_arm`). Every structural and integrity counter, every coverage counter
(arms trained, the three evaluation-set sizes, the workspace, pairwise and no-broadcast influence probes) and every negative control is expected by at least
one mutation (`pairwise_influence_hits` is the same probe call as the negative control `isolation_probe`, so its mutation is killed through the control), and the plan's header maps each judged rule (comparator band, convergence guard, untrained-workspace guard, the two sensitivity
controls, the plateau protocol) to the mutations that kill it; the statistics (bounds, margin, sign, quantile, seed count, the four outcomes,
the three-valued attribution) are mutated in `scripts/tests/test_c001_aggregate.py`, which pins the bounds to hand-computed values and the quantile of the
refutation to a case where the unadjusted and the Bonferroni upper bound lie on different sides of the margin, and whose wiring tests cover the count of untrained-workspace seeds,
the invalidation of a seed recorded in another environment and the refusal of a seed with two finished records (hand-planted aggregator defects of these kinds fail the tests). The mutations
are planted in a throwaway copy by `research/consciousness/gw/run_mutations.py` (`scripts/mutation_check.py` cannot plant a defect in a Python
entrypoint; `python3 scripts/mutation_check.py C001 --check` verifies the anchors, `run_mutations.py` itself has no `--check`). The mutation run uses the locked environment's interpreter (`../c001-venv/bin/python`, section 8). **What is not killed by a planted mutation:** the values of the guards and allowances (the untrained-workspace allowance 4, the unconverged allowance 2, the margin, the graded threshold 15, the band, the t quantiles) are read from `criteria.toml` and `config.toml` by the tests of the aggregator, so a changed value is caught by the digest of the frozen files (`check_research_gates.py`), not by a test or mutation; the per-arm learning rates, widths and the weak arm's weight decay are comparator-fairness settings pinned by the equality tests of `test_rules.py` and by the digest (accuracy-only changes are not planted). The decision-relevant values `patience`, `trained_margin_permille`, `min_arms_trained`, the evaluation-set sizes and the probe settings are in the frozen `[preregistration]` table. `aggregate.py` refuses unless `results/mutations.json` counts as many mutations as the plan lists, all
killed, and `research/consciousness` and this directory are identical to the mutation commit.

## 6. Pilot disclosure and tuning (pilot seeds 1001-1030 only; none is declared)

The scripts and the raw outputs of everything in this section are committed under `research/consciousness/gw/pilots/` (`pilot.py` runs
a job, `summarise.py` recomputes every table below from the raw lines, `sweep/`, `grid/`, `calibration/`, `rerun-cap20000/`, `lr-edge/`, `lr-edge2/`,
`dry/`, `calib6/`). Pilot seeds are >= 1001; the declared seeds 4001-4020 were never run (the pilot runner refuses them). Round 6 added `calib6/` and the fresh pilot seeds 1011-1030.

**Operating-point rule.** The training-set size D* is the smallest D in {1024, 2048, 4096, 8192} at which the strongest pairwise comparator,
averaged over the sweep seeds, has a 16-module accuracy in [400, 800] per mille. **Order of events, as it happened:** the rule above was
written before the sweep; the sweep (two seeds only, 1004 and 1005) was then run; its D = 1024 row sits in the band (strongest comparator mean 458) but an arm
(all_to_all, seed 1005) had hit the 9000-step cap and the workspace had not trained on seed 1004 (ID 340, chance 333), so a convergence clause ("D must also be one at which every guarded arm
converged and the workspace trained", i.e. the experiment's own validity guards applied to the sweep) **was added after the 1024 result had
been seen**, and it gives D* = 2048. D = 4096 (mean 799) is also inside the band but is not the smallest. Only two sweep seeds were used, with
large noise (a2a at 1024: 329 and 587), so "smallest in band" has a random component. The sweep ran at the untuned width d = 32 while the final
comparators run at d = 48; the final per-seed strongest comparator (pilot seeds 753-771 at the pinned settings, dry runs 754 and 738) is consistent with
the band.

**Sweep** (pilot seeds 1004 and 1005, lr 0.003, d 32, plateau protocol; accuracy on 16 modules per mille, seed 1004 / seed 1005):

| D | all_to_all | bw_all_to_all | sparse_all_to_all | workspace | strongest comparator mean |
|---|---|---|---|---|---|
| 1024 | 329 / 587 | 320 / 577 | 323 / 336 | 293 / 448 | 458 |
| 2048 | 774 / 729 | 735 / 610 | 637 / 648 | 593 / 770 | 751.5 |
| 4096 | 781 / 817 | 771 / 792 | 781 / 813 | 863 / 860 | 799 |
| 8192 | 821 / 829 | 814 / 761 | 815 / 840 | 895 / 887 | 827.5 |

In round 4 the one capped sweep run (all_to_all, 1024, seed 1005) was rerun at cap 20000: it converged at 9450 steps with the same 587, so the
first half of the clause no longer applies, while the second half (the workspace, ID 340 against chance 333 on seed 1004; the sparse arm is
also near chance on both seeds) still excludes 1024 and D* = 2048 stands. Disclosure: D = 4096 (the v2 choice) is the size at which the workspace's advantage is visible in this sweep (workspace minus
all_to_all: +82 and +43 at 4096, +74 and +58 at 8192, against -181 and +41 at 2048); **D* = 2048 is NOT chosen to show it, and the advantage
grows with D, so NOT_SHOWN at 2048 must not be read as "no effect" at larger training sets.** At 2048 the workspace is not clearly better on the
sweep settings (593/770 against 774/729).

**Equal grid** (D = 2048, pilot seeds 1004-1008, plateau protocol; mean 16-module accuracy per mille over the five seeds; the number of runs of 5
that hit the step cap: none remain after the rerun). Highest mean per arm in bold:

| arm | lr .001 d 32 | lr .001 d 48 | lr .003 d 32 | lr .003 d 48 | lr .006 d 48 | lr .012 d 48 |
|---|---|---|---|---|---|---|
| workspace | 402 | 559 | 731 | **785** | 655 | - |
| all_to_all | 573 | 666 | 727 | 738 | **754** | 346 |
| bw_all_to_all | 566 | 677 | 692 | 732 | **738** | 486 |
| sparse_all_to_all | 418 | 595 | 656 | 674 | **696** | 343 |
| no_broadcast | 432 | 560 | **768** | 766 | 573 | - |
| competition_off | 627 | 639 | 751 | 786 | **790** | 340 |
| capacity_off | 415 | 536 | 721 | **727** | 608 | - |

(The lr .006 column is the round-4 extension and the lr .012 column the round-5 extension for the four arms that selected .006, both d 48 only; d 32 at lr .006 and .012 was not run for any arm; "-" = not run, the arm had already lost 120 to 190 per mille at .006.)

**Step cap.** The first version of this table (cap 9000) had eight grid runs and one sweep run at the cap. All nine were rerun at cap 20000
(`rerun-cap20000/`; deterministic, so every run that had stopped by patience is bit-identical and was kept). All nine converged (between 9150 and
13200 steps). No selected cell changed; the means of the affected non-selected cells moved as shown (for example capacity_off at lr 0.003, d 32:
715.8 to 721.4, still below its d = 48 cell). In the pinned cells the longest run is 8400 steps, so the cap of 20000 is inactive there.

**Learning-rate edge and extension (round 4).** All seven arms first selected lr 0.003, the upper edge of the grid. As decided in advance
(extend only if a quick pilot shows the selection changes), the same extension was run for every arm: lr 0.006 at d 48, seeds 1004-1008, cap
20000 (35 runs, none capped, longest 8400 steps; about 72 minutes of wall time on a loaded machine, so the 30-minute CPU allowance was
exceeded). Differences of the lr 0.006 mean from the lr 0.003 mean at d 48: workspace -131, all_to_all +16, bw_all_to_all +7,
sparse_all_to_all +22, no_broadcast -193, competition_off +3, capacity_off -118. **The selection changed for four arms** (all_to_all,
bw_all_to_all, sparse_all_to_all, competition_off; three of the four by less than the noise of a five-seed mean), by the mechanical rule it
now selects lr 0.006 for them, and this is adverse to the claim (it narrows the pairwise advantages). The workspace, no_broadcast and capacity_off
keep 0.003 and are unstable at 0.006. weak_all_to_all and crippled_all_to_all inherit all_to_all's 0.006 and were recalibrated (below). 0.006
was again the edge for those four arms, which would leave their comparators possibly under-tuned (an under-tuned comparator is a strawman and biases
toward a false COMPLETED). **Second extension (round 5), decided by the project lead after seeing the pilot comparisons: the rule is "bracket every arm",
so the grid was extended by one more step (lr 0.012, d 48, seeds 1004-1008, same plateau protocol, same metric) for exactly the four arms
that selected 0.006; if 0.012 had won for an arm it would have been adopted and 0.024 run for that arm, until every selected rate was
interior or a divergence bound was reached.** Result (20 runs, none capped, longest 4950 steps; raw lines in `lr-edge2/`): mean 16-module accuracy
at lr 0.012 against lr 0.006: all_to_all 346 against 754, bw_all_to_all 486 against 738, sparse_all_to_all 343 against 696, competition_off 340
against 790. 0.012 does not win for any arm; in 18 of the 20 runs the network did not learn at all (accuracy 329-356, chance 333; stopped by
patience after 1950-2400 steps), the other two (bw_all_to_all, seeds 1004 and 1006) reached 695-722. So 0.006 is interior for all four arms,
the rule terminates without a third extension (the stopping condition was interiority; 0.012 is itself a failure-to-train bound), and no
selection, pinned rate, weak/crippled calibration, pilot paired difference or power figure changes. The workspace, no_broadcast and capacity_off
are bracketed on both sides at d = 48 (0.001 / 0.003 / 0.006: workspace 559 / 785 / 655, no_broadcast 560 / 766 / 573, capacity_off 536 / 727 /
608). Disclosure: every extension (0.006 and 0.012) was run on the pilot seeds after the pilot comparisons had been seen, and the first
one changed four selections; the grid is geometric with ratio 2, so rates between the tested ones (for example 0.0045) were not tried, and
the competition_off selection of 0.006 over 0.003 is a tie within noise (790 against 786).

**Selection and pinned settings.** Width: the workspace, all_to_all, bw_all_to_all, sparse_all_to_all, competition_off and capacity_off
selected d = 48. **no_broadcast selected d = 32 by 2 per mille (768.2 against 765.8), a tie within noise; it is pinned at d = 48, which breaks
the tie toward architectural identity:** the arm that was tuned (ffn 64 matched to the workspace at the same width) and the arm that runs are then
the same object, exactly the workspace architecture without the read path, whereas the d = 32 cell would run with ffn 189 and 53724 parameters,
which was never tuned. (A round-3 draft pinned d = 32 and took the paired differences from the d = 32 cell, a different arm.) no_comm uses lr
0.003 and d 48 untuned (it is the floor). The same data and ordering are used for every arm of a seed. Pinned: `arm_lr` and `arm_d` in `config.toml`.
**Tuning budget.** It is not strictly equal: the workspace, no_broadcast and capacity_off were tuned on five cells (lr 0.001 and 0.003 at d 32
and 48, and lr 0.006 at d 48), all_to_all, bw_all_to_all, sparse_all_to_all and competition_off on six (those five and lr 0.012 at d 48);
the asymmetry favours the comparators, i.e. it is adverse to the claim. **no_broadcast tie, in full:** at lr 0.003 the d = 32 cell has the
higher mean by 2 per mille (768.2 against 765.8); the paired differences of the workspace from the two cells are 17 / sd 71 (d = 32) and
19 / sd 50 (d = 48, pinned). Neither is distinguishable from the margin on five seeds (the attribution state would be *unresolved by noise*
either way), the choice of d = 48 follows the matching rule (ffn 64, the workspace without its read path), and the contrast is
reported, never judged.

**Paired differences at the pinned cells** (workspace minus arm, 16 modules, mean / sd over the five pilot seeds; every one from the pinned
d = 48 cell at the arm's selected learning rate): all_to_all **31 / 24** (it was 47 / 29 at lr 0.003), bw_all_to_all **47 / 42** (53 / 38),
sparse_all_to_all **89 / 64** (111 / 27); reported contrasts: no_broadcast **19 / 50** (the round-3 draft's 17 / 71 came from the d = 32 cell and is
withdrawn), competition_off -4 / 24, capacity_off 59 / 179 (one seed at 371). **The workspace is better than the pairwise arms by about the
margin or more and not clearly better than no_broadcast or competition_off. These are selection-inflated pilot estimates (the workspace structure
was fixed from earlier pilots; the settings were selected per arm on the same seeds) and are not evidence.**

**Sensitivity calibration (round 6; replaces the two-round control of rounds 3 to 5).** Until round 5 the graded control was the all-to-all net with two rounds
instead of three; on the tuning seeds 1004-1008 its deficit was 61, 47, 72, 55, 13 (mean 50, sd 22; crippled, one round: 395, 421, 394, 403, 416, mean 406, sd 12; at lr 0.003 in round 3, crippled at the old width cap
384: weak 59 / sd 21, crippled 394 / sd 16). The two dry runs (seeds 1009 and 1010, out of sample) then gave -6 and -92, and ten fresh pilot seeds 1011-1020
(`calib6/graded-baseline-*.jsonl`, harness-identical evaluation) gave 34, 31, 24, 41, -5, 59, 10, 39, 54, 34 (mean 32, sd 19); over all 17 seeds the mean is 28
and the sd 38, and the pilot-estimated probability that the 20-seed lower bound does not exceed 15 is 0.51 (bootstrap) to 0.57 (normal), not about 0 as the earlier
text said (it quoted a deficit "about 1.7 times the margin" from the tuning seeds, which also selected the comparators' learning rate). Because the graded control is a hard INVALID gate (without it NOT_SHOWN cannot
be told from an insensitive pipeline) it has to be reliable, so a weakening was sought whose deficit is reliably positive and still graded.

*Rule (the project lead's decision before the calibration: reliably positive, estimated failure probability at most 0.05, still graded; the numerical form of "graded" and the bootstrap were fixed while the candidates were being tried, before the final arm was run):* a candidate must (i) be a single knob on the pinned three-round all-to-all at lr 0.006, d 48, (ii) have a mean deficit at most
half of the crippled arm's (about 410) so that it stays graded, and (iii) have a pilot-estimated probability that the 20-seed lower bound does not exceed 15 of at most 0.05, by a normal model and by a
bootstrap over seeds, on the ten calibration seeds 1011-1020; the smallest such weakening is then confirmed on ten further fresh seeds 1021-1030. Only deficits against all_to_all were looked at; no
contrast with the workspace was looked at before the choice (the workspace, bw_all_to_all and sparse_all_to_all runs on 1011-1020 were queued to start when the calibration run of the final arm
finished and were read afterwards: the contrasts of section 4). The candidates were tried in an ad hoc order, guided by the
previous results, and several were stopped before ten seeds when they were clearly gross, bimodal or no better than rounds (all rows are kept in `calib6/`):

| single knob on the three-round all_to_all (seeds 1011-1020) | seeds run | deficit all_to_all minus arm, mean / sd | note |
|---|---|---|---|
| two rounds (the round-4 control) | 10 | 32 / 19 | 17-seed pool 28 / 38; P(fail) 0.51-0.57 |
| value width 6 / 3 / 2 / 1 (instead of 48) | 5 / 5 / 7 / 8 | 25 / 51, 40 / 23, 63 / 89, 274 / 150 | width 2 and 1 bimodal (one seed 260 or 440 below, another unaffected) |
| top-2 / top-1 keys per query | 2 / 8 | 439, 413 / 24 | the net does not train (gross) |
| update width d 24 / d 16 | 4 / 5 | 33 / 23, 9 / 51 | |
| learning rate 0.001 | 10 | 27 / 52 | |
| Adam weight decay 1e-4 / 3e-3 (probes on all_to_all's initialisation) | 3 / 2 | 34 / 42, 234 / 8 | |
| Adam weight decay 1e-3 (same probe) | 10 | 80 / 55 | |
| **Adam weight decay 1e-3 as the arm weak_all_to_all (final)** | 10 | **65 / 60** | P(fail) 0.024 (normal), 0.009 (bootstrap) |
| Adam weight decay 2e-3 as the arm weak_all_to_all | 1 (stopped) | 193 | not needed once 1e-3 met the rule |

Weight decay 1e-3 is the first weakening, in the order tried, that met the rule on ten seeds (value width 3 and update width 24 looked similar, mean deficits 33-40, but were stopped at 4-5 seeds and not completed); the final arm `weak_all_to_all` (own initialisation, `arm_weight_decay` = 0.001) was run on the calibration seeds (65 / 60: 8, 170, 108, 87, -24, 34, 115, 55,
90, 6) and on the confirmation seeds 1021-1030 (`calib6/confirm-*.jsonl`): 110, 88, 26, 89, -5, 148, 35, 40, 67, 54 (mean 65, sd 45); over the 20 seeds mean 65.0, sd 51.4, median 61, with
P(the 20-seed lower bound does not exceed 15) 0.005 (normal) and 0.001 (bootstrap). On the tuning and dry seeds 1004-1010 the same arm gave 51, 160, 49, 150, 75, 9, -33. Over all 17 seeds
that have every arm: mean 65, sd 62, P(fail) 0.03 (normal), 0.02 (bootstrap). Individual deficits are negative on 3 of the 27 seeds (single seeds are noisy; the control is judged on the 20-seed mean). Crippled (one round) on the fresh seeds: deficit mean 410,
sd 22 (436, 432, 438, 394, 368, 398, 403, 400, 414, 418); the dry seeds gave 331 and 343. So the graded control has a deficit of about 65 per mille (about twice the margin, one sixth of the gross control's) and the gross one of about 400. This control demonstrates sensitivity to
a deficit of about 65 per mille; it does not by itself show that a difference of the margin's size is resolvable.

**Band.** The comparator band [300, 900] comes from the headroom: the strongest pairwise comparator is about 740 on the pilot seeds (719-762
over the five seeds) at 2048, which leaves 160 per mille to the 900 ceiling for a workspace advantage and sits above the 300 floor by more than a
seed-to-seed range. The margin is not derived from it (section 4).

History, so that the reader sees how the operating point was found:
1. Version 1 (position-specific modules, online data, six modules): nothing was learned in some variants; after two task
   reductions every comparator was at 95-97% and the workspace at 96%; the outcome was fixed by the ceiling. Withdrawn.
2. Version 2, first operating point (6 types, noise 0.5, shared weights, finite data): at 512 to 32768 training examples neither the
   workspace nor the all-to-all arm generalised beyond about 30-39% on the pilot seed (too hard for 3000 steps). Task reduced to 4 types
   and noise 0.3. Version 2 then pinned D = 4096 with fixed 3000-step training and a lr grid on two seeds selected by in-distribution
   accuracy (pilot seeds 1001-1003); withdrawn (see the header).
3. Version 3: the sweep, grid and calibration above on seeds 1004-1008; patience 6 was replaced by 12 after a pilot showed the workspace
   stopping before a late jump. Round 4: step cap 9000 to 20000 with the nine capped runs rerun; guard restricted to the workspace and the
   pairwise comparators; no_broadcast pinned at d 48 and moved to the attribution contrasts; margin 30 with its rationale; the harness audits
   of the training loop; the learning-rate extension (0.006) for every arm, adopted for four arms, with the controls recalibrated. Round 5: the second extension (0.012) for
   those four arms, which won for none (all failed to train), so 0.006 is interior and nothing else changed. Round 6: the graded control recalibrated on fresh pilot seeds (weight decay 1e-3 instead of two rounds), the INVALID branch added to the power estimate
   and the expected outcome restated as a range; the workspace's training failure on pilot seed 1011 found; `pilot.py` made to draw its evaluation sets like the harness; untested coverage rules and aggregator wiring given tests and mutations.
4. Dry runs (full pinned configuration, the locked environment through the pinned entrypoint; `weak_all_to_all` in these records is the two-round arm that round 6 replaced). **Seed 1009 (round 3 configuration: cap 9000,
   no_broadcast at d 32, comparators on the old guard):** exit 0; workspace 797, no_broadcast 799 (the d = 32 arm), bw_all_to_all 754,
   competition_off 751, capacity_off 744, all_to_all 708, weak_all_to_all 708, sparse_all_to_all 706, crippled_all_to_all 344, no_comm 347; it is
   superseded by the seed-1010 run because the configuration changed (the no_broadcast arm in particular). **Seed 1010 (final configuration):**
   exit 0 (pilot seed, so the aggregator's only complaint is "not a declared seed"), every arm converged (longest 7050 steps), no
   structural or integrity counter, no coverage rule failed, comparator 738, 895 s wall on a loaded machine. Accuracy on 16 modules, per mille:
   workspace 812, capacity_off 797, weak_all_to_all 768, competition_off 762, sparse_all_to_all 738, bw_all_to_all 728, all_to_all 676,
   no_broadcast 664, no_comm 344, crippled_all_to_all 333. **On this seed the weak arm is above all_to_all (768 against 676) and the
   workspace advantage over all_to_all is 136, far above the pilot mean: both are within the spread of single seeds (sd 22 and 24 over five
   pilot seeds is a small-sample estimate); the graded control is judged on the mean over twenty seeds.** The record is `pilots/dry/dry-1010.json`.
   **Seed 1009 again at the final configuration (round 5, after the second learning-rate extension, which changed no pinned value):** exit 0, every arm
   converged (longest 6600 steps), no structural, integrity or coverage counter, no failed guard, comparator 758, 7 min 41 s wall. Accuracy on 16
   modules, per mille: no_broadcast 808, workspace 797, bw_all_to_all 758, capacity_off 744, competition_off 741, sparse_all_to_all 690, weak_all_to_all
   681, all_to_all 675, no_comm 347, crippled_all_to_all 344. The workspace is above all_to_all by 122 and below no_broadcast by 11: one seed, not
   evidence. The record is `pilots/dry/dry-1009-final.json` (the file `dry-1009.json` is the superseded round-3 run).
   **Seed 1031, the final code (round 8)**, run through the pinned entrypoint exactly as section 8 runs seeds (`env -i` with PATH, HOME, TMPDIR, PYTHONNOUSERSITE, PYTHONHASHSEED; the locked `c001-venv`; `PYTHONDONTWRITEBYTECODE=1`; `--preflight` first): exit 0, no
   structural, integrity or coverage counter, no failed guard, every arm converged (longest 7950 steps), the weight-decayed weak arm recorded (`weight_decay_micro` 1000), the workspace trained. Accuracy on 16 modules, per mille: workspace 815, capacity_off 820, no_broadcast 796,
   competition_off 796, bw_all_to_all 739, sparse_all_to_all 738, all_to_all 656, weak_all_to_all 630, crippled_all_to_all 336, no_comm 320. Record and stderr: `pilots/dry/dry-1031-final.json`, `.err`, `.exit`.

**Evaluation sample of the pilot rows.** `pilot.py` trains with the harness's own `train_arm` (same data, same order, same initialisation), but until
round 6 it drew the in-distribution, 16-module and 24-module evaluation sets from one generator in sequence, whereas the harness uses one generator per set
(streams 500, 501, 502). Training, the validation curve (stream 503), the step counts and the selection of checkpoints are identical; the
16- and 24-module accuracies of the tuning-era rows (`sweep/`, `grid/`, `calibration*/`, `rerun-cap20000/`, `lr-edge*/`) therefore come from a
different evaluation sample of the same size and differ from what the harness gives for the same seed by evaluation sampling noise (standard error
about 10 per mille at 2048 instances and accuracy 0.75), not by training. From round 6 `pilot.py` draws like the harness, and this was checked: for
seed 1009 at the final configuration the pilot rows of `weak_all_to_all` (two rounds, as it then was) and `all_to_all` give 681 and 675 per mille on 16 modules, the numbers of
`dry/dry-1009-final.json` (`calib6/check-1009*.jsonl`). The tuning-era selections (learning rate, width) were made on the older sample.

## 7. Limits

See `aggregate.py` LIMITATIONS: one task family and size; twenty seeds and a t bound (normality unverifiable; the attribution contrasts are heavy-tailed
and also reported by median and sign count); low power for a support outcome unless the true advantage is about 2.5 times the margin (section 4); the
ablations are definitions; the learned analogue is not the deterministic contract; sample efficiency, dropout, 24-module accuracy and ignition are
unjudged; the mutation tooling gap; and the standing sentence that this experiment says nothing about whether any system is conscious. Specific to
v3:

* **Every arm first selected the upper edge of the learning-rate grid (0.003), and the grid was then extended twice on the pilot seeds after the
  pilot comparisons had been seen** (0.006 for every arm, which changed four selections; 0.012 for those four, which won for none, section 6). The
  rule was "bracket every arm"; every selected rate is now interior at d = 48 (workspace, no_broadcast, capacity_off 0.003; the others 0.006). The grid
  is geometric with ratio 2 and only d = 48 was extended, so a rate between two tested ones, or a better rate at d = 32, was not tried: a residual
  under-tuning of any arm is possible and its direction is unknown (under-tuned comparators favour support, an under-tuned workspace favours refutation).
* **The width grid stops at d = 48 for six of seven arms.** d = 64 was not tried; the workspace gained most from d = 32 to 48 (+54 at lr 0.003
  against +11 for all_to_all). A wider optimum may exist for any arm; the direction of the resulting bias is unknown.
* **The plateau protocol can stall.** A run that stops after 12 evaluations without improvement just before a late jump is flagged converged (pilot
  seed 1006, capacity_off, d 32, lr 0.003 at the 9000 cap before the rerun; the earlier patience 6 produced the same effect for the workspace). A
  stalled comparator inflates the advantage, a stalled workspace deflates it. The guards catch only extreme cases (the band, an untrained workspace).
* **The margin was chosen after the pilots** (section 4) and the operating point's convergence clause was added after the 1024 result (section 6).
* **The advantage grows with the training-set size** (section 6): NOT_SHOWN at 2048 examples must not be read as "no effect" at larger D.
* **The sensitivity controls** demonstrate resolution of deficits of about 65 per mille (graded) and 400 (gross), not of the margin's 30: the graded deficit's per-seed sd (51 over 20 seeds) is of the size of the contrasts' (36 to 72 on the pilot seeds on which the workspace trained, 150 or more when it did not), so
  they are necessary, not sufficient, signs that a difference of the margin's size is resolvable; the pilot-estimated probability that the graded control fails is 0.005 to 0.03 and it is a hard INVALID gate.
* **The workspace stalled on 1 of 17 pilot seeds** (seed 1011, 16-module accuracy 367, in-distribution 437, 104 per mille above the no-communication floor of 333). The harness rule that the guard enforces (in-distribution accuracy at least the no_comm arm's plus 100) does not flag it, so it does not count toward the guard
  (allowance 4 of 20) and stays in the paired differences (it lowers every contrast mean by about 20 per mille and inflates the variance: conservative, it never favours COMPLETED); a seed that did fall under the rule would be counted and also stay in. **Bounded cause check (round 7,
  `pilots/cause1011.py`, raw lines in `pilots/cause1011/`):** the failure is deterministic (an exact re-run gives the identical 437 / 367 / 4350 steps) and is a property of the initialisation, not of the data or the code: the same training
  and evaluation data with three other initialisations of the workspace (same lr 0.003, d 48) trained normally (in-distribution 845, 832, 856; 16 modules 796, 773, 802), and the three pairwise comparators trained to 722-791 on this very data.
  The failed run's validation curve stays at 313-367 (chance 333) through its first six evaluations, then creeps to 409-423 and stops by patience, whereas the other initialisations leave chance at the 4th evaluation (467, 435, 534) and rise to 800+: the gated slot
  update did not "ignite". No code defect was found (the structural counters, the influence probes and the other 16 seeds pass, and the pilot reproduces the harness bit for bit); per-module accuracies were not examined. The model and training were NOT changed in response (that
  would be post-hoc tuning of the claim's arm): an initialisation lottery of the workspace is part of what the experiment measures, and the failed seeds stay in the paired differences. One failure in four initialisations on one seed is a small sample; it does not give a failure rate
  beyond the 1/17 estimate. A successor could stabilise the optimisation; the learning rate 0.003 selected on five seeds may also be near an instability edge (0.006 was unstable for the workspace on the tuning seeds: 655 against 785).
* An equivalence reading ("same accuracy at lower communication cost at 8 modules and large data") was considered and is not part of this
  preregistration; it would need arms at a larger training set.

## 8. Entry point, environment and freeze procedure

The entrypoint is `env UV_PROJECT_ENVIRONMENT=../../../c001-venv PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline --no-python-downloads
--project research/consciousness python -B research/consciousness/gw/harness.py --benchmark c001 --seed <seed>`. Reasons, each found by
rehearsing the path through `scripts/run_experiment.py` in a scratch clone (the rehearsal log is not part of the repository; its three findings are these bullets):

* The runner passes only PATH, HOME, TMPDIR/TEMP/TMP, locale and the cargo/rustup homes (plus PYTHONNOUSERSITE=1 and PYTHONHASHSEED=0): no
  proxy, certificate or `UV_*` variable. So `--offline` (nothing is downloaded at run time) and `--no-python-downloads`.
* The runner **refuses to run while the repository holds git-ignored files** (`.venv/`, `__pycache__/`). A project-local `.venv` would
  therefore stop every run, so the environment lives outside the repository (`UV_PROJECT_ENVIRONMENT`, resolved against
  `research/consciousness`: the directory `c001-venv` next to the repository checkout), bytecode is not written (`-B`,
  `PYTHONDONTWRITEBYTECODE`), and the freeze must be run from a tree without `__pycache__` directories (the freeze scripts of F003 do
  `find . -name __pycache__ -prune -exec rm -rf {} +`).
* The locked environment holds **torch only** (no numpy); the harness is written for it (a numpy call hid in the development environment
  and was found by the rehearsal).

**Where the environment lives.** `UV_PROJECT_ENVIRONMENT=../../../c001-venv` is resolved against `research/consciousness`, so the environment is
the directory **`c001-venv` in the parent directory of the checkout**. The seeds are run from the worktree `scratchpad/wt-dev-gw`, whose
parent (`scratchpad/`) holds `c001-venv` (the locked environment: python 3.11, torch 2.7.1+cpu, the hash of `uv.lock` is in every result's environment record). In
any other checkout (for example `/home/user/PTR`, whose parent `/home/user` has none) the directory must be created first, with the
network variables set and as the same user and HOME that will run the seeds: `UV_PROJECT_ENVIRONMENT=../../../c001-venv uv sync --locked
--project research/consciousness`; then the smoke check `env -i PATH=... HOME=... UV_PROJECT_ENVIRONMENT=../../../c001-venv uv run --locked
--offline --no-python-downloads --project research/consciousness python -c "import torch"` must pass. A missing environment makes `uv run` exit 2
**inside a recorded run** and spends that seed, which is why the pre-flight below is run before a seed is reserved. The harness checks its
environment (python 3.11, torch 2.7.1, the config file) before training and exits 2 with a message otherwise, so an incomplete environment
cannot be mistaken for a falsification (a Python crash would exit 1). `uv` and CPython 3.11 must be on PATH (absolute entries). A listed seed runs once
(the runner refuses a second run of a seed, and `aggregate.py` refuses two finished records of one seed).

**Pre-flight before every seed reservation (run by hand: the repository has no hook for it).** `scripts/run_experiment.py`,
`scripts/experiment_records.py` and the manifest have no `preflight` key or command (checked on this branch), so the person who runs the
seeds runs it manually before `prepare` and before every `run`: `... python -B research/consciousness/gw/harness.py --benchmark c001 --preflight` (the same
`env UV_PROJECT_ENVIRONMENT=... uv run ...` entrypoint without `--seed`, with the runner's exact stripped environment, `env -i` plus PATH, HOME,
TMPDIR). It checks the environment and builds every arm once, reads no seed and trains nothing, so a failure costs nothing, whereas a
failure inside a recorded run spends the seed. It prints the environment record. Every result embeds an environment record (python, torch,
torch build-configuration hash, CPU model and flag hash, machine, threads, hashes of `uv.lock` and `pyproject.toml`).
**Never run two seeds concurrently** (each run is single-threaded and its environment record holds the thread counts). **All twenty seeds run on one host in one environment:** the aggregator marks a seed invalid whose recorded environment differs from
the first one (`environment_mismatches`), so a change of machine, CPU model, torch build or lock file in the middle of the run makes the experiment
INVALID (a different machine changes floating-point results and would not be a replication).

**Freeze procedure.** The two sha256 values of the `[preregistration]` table and the manifest digests read `must-be-pinned-before-prepared` until step 3, which pins them.
Every head-side `python3` call below is made with `export PYTHONDONTWRITEBYTECODE=1`: `__pycache__` directories are git-ignored and the
runner refuses to run while the repository holds git-ignored files.

1. Commit the tree (the pre-freeze commit). **The commit must include the untracked `research/consciousness/gw/pilots/` (scripts and raw
   pilot outputs, cited throughout section 6) and `research/consciousness/gw/power.py`** (cited in section 4); `git status` shows them as
   untracked until added. Remove `__pycache__` directories (`find . -name __pycache__ -prune -exec rm -rf {} +`) and any `.venv`; the repository must hold no
   git-ignored file when the runner starts.
2. (Done before the pre-freeze commit: the text already lives in `PREREGISTRATION.md`; there is nothing to rename.)
3. Pin the digests: in `config.toml` set `preregistration_text_sha256` to `experiment_records.preregistered_file_digest` of `PREREGISTRATION.md`
   and `criteria_sha256` to that of `criteria.toml`; in `experiment.toml` set `preregistration_sha256` to `experiment_records.preregistration_digest`
   of the `[preregistration]` table and `preregistration_rules_sha256` to `experiment_records.canonical_digest` of the C001 entry of
   `experiments/preregistration.toml` (as `scripts/check_research_gates.py` compares them); set `status = "prepared"` in `experiment.toml` and in
   `experiments/registry.toml`; take C001 out of `LISTED_UNFROZEN` in `scripts/tests/test_research_gates.py` as F003 was.
4. Commit again and tag it (`c001-prereg`; the mutation commit must contain the pinned files: `aggregate.py` refuses if `research/consciousness` or this directory
   differs from it). Then from the clean tree run the mutation plan with the locked interpreter and a single thread:
   `OMP_NUM_THREADS=1 ../c001-venv/bin/python research/consciousness/gw/run_mutations.py` (38 mutations; the harness at the plan's reduced
   scale, about 75 s each on a quiet core, then `results/mutations.json`); commit `results/mutations.json`.
5. Smoke check and pre-flight (above), then `python3 scripts/run_experiment.py prepare C001`, **and commit** (`git add experiments && git commit`); then for each seed 4001-4020, once each
   (about 7 minutes each): pre-flight, `python3 scripts/run_experiment.py run C001 --seed <s>`, and **commit the new run record before the next seed** (the runner reads the
   committed earlier records when it checks a new run, `earlier_run_problems`; the freeze script of F003 commits after `prepare` and after every seed). A seed whose run exits 2 is not
   repeated by hand: stop and report. Then `python3 experiments/consciousness/C001-global-workspace/aggregate.py --report report.md`.
6. Run `python3 scripts/check_research_gates.py` and `python3 scripts/run_experiment.py validate`; advance the registry status by hand only for
   COMPLETED or FALSIFIED (NOT_SHOWN and INVALID leave it `prepared`; the aggregate records them with `recommended_status` null).
