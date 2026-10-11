# C008 attention-schema routing, CPU stage A: preregistration (v2, after the pre-freeze review)

Status: preregistration of version 2, built to freeze-ready; **nothing is frozen and no declared seed has been run**. C008 is the CPU stage A of C003
(`experiments/consciousness/C003-attention-schema-routing`, whose text points here and which remains the GPU stage B on OLMoE routing). The scaffolder
accepts only identifiers of a capital letter and three digits, so the stage-A experiment has its own number, C008. The pinned values are `[pinned_thresholds]`
in `config.toml`; `criteria.toml` holds the rules; this file states the reasons and the limits. **What kind of run this is is stated in section 2: a construction-validity check of a
world that was built so that a schema can help, not a test of the hypothesis that a schema pays for itself.** The design went through several iterations on pilot seeds before the
final world and a further fix round after an independent review; **all of it is disclosed in section 6, with a mapping from each design to its pilot data and the numbers that motivated each change**.

## 1. Scope and non-claims

C008 is a functional-mechanism test, on a CPU toy router, of the indicator AST-1 of Butlin et al. 2023 (a predictive model representing and enabling control
over the current state of attention) as a computation: does a learned **attention schema**, a small network that reads the router's own recent routing history
and predicts the router's own next routing propensity, improve what the router achieves **when its prediction is used to control routing**, over (a) the same
router without a schema, matched in FLOPs, (b) matched in parameters, (c) a heuristic auxiliary-loss-free balancer, (d) the same schema trained identically but
**not used for control** (describe-only), and (e) a **router that reads the same history directly with no schema** (parameter-matched)? **It is not a test of consciousness and no
result is evidence about consciousness.** No self-report is produced, trained or measured. It says nothing about OLMoE routing or about any real router: the world is a synthetic demand
with a deterministic verifier. **AST-L2 of doc 38 is not changed by this experiment, whatever its outcome**: the requirement is tested at stage B (C003) or not at all.

The schema has no authority over expert promotion or revocation (the registered constraint of C003). Two structural counters check that it cannot reach what the verifier reads and that, when it
only describes, it cannot reach the router. **In the toy the authority counter is nominal**: the world has no promotion or revocation, the router outputs choices only, and `authority_leaks` checks that an
adversarial schema output does not change capacities or competence, which the architecture already precludes; the mutation shows the detector can fail. It is not evidence that a schema cannot gain authority.

## 2. Hypothesis, the claim being tested and what the run is

H1 (conditional on the constructed world). In a world where the demand on the experts follows a cycle that one token cannot reveal but the router's own load history does, a router whose logits include a
learned control law of the schema's prediction verifies a larger fraction of tokens on a fresh stream than each of the five comparators, by at least the pinned margin (25 per mille), shown by the lower
one-sided 95% t bound of the paired per-seed difference over the declared seeds, **and the gain is attributed to the router's own history** (the control arm reading the history of random other batches does
not keep it).

**What the run is.** The world was designed, after several failed designs, so that regime-aware control has a reachable gain (section 6), and two of its features were chosen while looking at schema-free and
hand-set-law results. The run is therefore a **construction-validity check**: a COMPLETED says that the toy contains what a schema is for and that a schema used for control can be learned and used in it, and that
this is the schema's doing and not the extra capacity's; it does not say that a schema pays for itself in any other setting, and it leaves AST-L2 where it is. The new comparator, the router that reads its own history
directly, is the natural strong baseline for "history helps": if the schema-controlled router does not beat it by the margin, the result is that **in this world access to the history, not a separate predictor with a control
law, accounts for the gain** (FALSIFIED, refuted against `history_router`). The pilots (section 6) point to that outcome.

## 3. World, arms, training

* **World** (`research/cseries/c003a/c008_task.py`). Eight experts, 256 tokens per batch. A token has a class c in 0..7 (a noisy one-hot feature, noise 0.25); the experts
  competent for it are c and c+1 (mod 8). Each expert serves the first 64 tokens routed to it in the batch and drops the rest. A token is **verified** iff its expert is
  competent and has room (a deterministic verifier). Demand is cyclic: eight regimes of three batches each; in regime r the classes r and r+1 carry 80% of the tokens, the
  others share the rest. Every class is the "first" hot class in one regime and the "second" in the next, and the best split of its tokens over its two competent experts is the opposite
  in the two, so a regime-blind router cannot be right in both; the router's own load history reveals the regime. The oracle (a maximum flow, which also coordinates tokens within a
  batch) is the ceiling that shows headroom; independent token sampling cannot reach it.
* **Arms** (`c008_model.py`; each trained online by policy gradient on the verifier's reward, REINFORCE with the batch-mean baseline and an entropy bonus, Adam):
  * `control`: the router (one hidden layer, 48) plus an attention schema (history of the router's propensity over the last 4 batches, one hidden layer of 32, softmax output;
    trained only by cross-entropy to the router's next propensity, detached from the policy) plus a **control law**: a zero-initialised learned map from the schema's expected relative
    load (8 times its predicted distribution) to additive logits (form full, diagonal or scalar, pilot-selected: diagonal). The propensity is the mean softmax of the router before any
    control term, so the schema's target does not depend on the law it feeds (no loop).
  * `schema_free`: the router alone, its hidden width chosen to match the control arm's FLOPs per batch (the schema's included) within 5%; `schema_free_wide`: its hidden width
    chosen to match the control arm's parameter count within 2% (it uses more FLOPs, reported).
  * `heuristic`: the schema-free router plus the auxiliary-loss-free balancer (one bias per expert, moved after each batch against the previous batch's realised load by a sign step,
    as in the bias-balancing design of doc 37; it keeps adapting during evaluation, as it would in use).
  * `describe_only`: the control arm's architecture and training, with zeros in place of the schema's output in the control law: the schema is computed and trained identically
    and has no path to the router. (Control against describe-only differs from control against schema-free only in the schema-free router's width and in the schema's training signal, so
    **the two contrasts are not independent**; the describe-only contrast is the registered "control role" test, reported as `control_role_not_shown`.)
  * `history_router` (new): a router with no schema and no law whose input is the token's features next to the flattened history of **its own** propensity over the last 4 batches (as relative
    loads), parameter-matched to the control arm within 2% (its FLOPs per batch are about twice the control arm's, which favours the comparator; reported). It receives exactly the information the
    schema reads, and nothing else. This is the arm that makes the control-versus-describe-only contrast **isolate the schema**: without it, a COMPLETED could mean only that a router given load history does better than one that is not.
  * Sensitivity, attribution and reference arms: eval-only `random_router` (uniform; the gross control), `schema_free_epsilon` (the schema-free router with 15% of its tokens re-routed at random: a known
    deficit of about the margin's order; the graded control), `control_shuffled` (the trained control arm reading the history of random other batches: the attribution, **judged**), `even_split` (a
    hand-set regime-blind reference: the class the noisy features point at, the two competent experts alternating at random; reported, never judged), and the stationary null
    world (`schema_free_null`, `control_null`: the regime never changes, so no schema can help).
* **Plateau protocol (every trained arm, identical).** The verified-token rate on a fresh validation stream of 48 batches is evaluated every 250 steps; training stops when it has not
  improved by 3 per mille for 12 consecutive evaluations, or at 12000 steps (6000 in the null world); the best-validation checkpoint is restored; the loop is audited by the harness
  (checkpoint and convergence integrity counters).
* **Evaluation.** A fresh stream of 240 batches (61440 tokens; ten cycles) with the policy frozen. Training, validation, evaluation and probe streams come from disjoint generator seeds
  and are checked for overlap by hash.
* **Equal tuning.** On the pilot seeds 1001 to 1003 **every arm got nine configurations**, selected by the mean evaluation reward over the three seeds among the configurations in which at most one of the three hit the step cap
  (a judged arm at the cap on more than one seed makes the run INVALID): learning rate {0.001, 0.003, 0.01} x entropy coefficient {0.005, 0.01, 0.02} for `schema_free`, `schema_free_wide`, `describe_only` and `history_router`;
  learning rate x the form of the control law {full, diagonal, scalar} for `control`; learning rate x balancer step {0.01, 0.03, 0.1} for `heuristic`. (In the first version the schema-free arms had three configurations and the
  control nine; the fix round equalised the breadth. The selection bias of the maximum of nine noisy means is of the order of 3 to 5 per mille for every arm, since the standard deviation of a pilot mean is about 4.5.)
  The numbers are in section 6.
* **Distance to the blind optimum.** The hand-set regime-blind reference (`even_split`) is evaluated on the same streams; the aggregate reports, for every arm, how far the reference is above it. In the first version the trained
  schema-free arms sat about 14 per mille below the even split with true classes (0.833 against 0.847): part of the control arm's contrast over them is comparator shortfall, not schema value. The margin (25) survives that
  (61 - 14 = 47 in the first version); the report says so per seed.
* **CPU budget.** One seed trains eight arms single-threaded: 198 CPU seconds measured on the pilot seeds (2014 wall seconds on the loaded shared box), inside the budget of 15 minutes per seed on one core.

## 4. Decision rule (frozen in criteria.toml) and what each outcome means

Per seed: three **structural** counters (`describe_only_influence`: the describe-only arm's routing moved when its schema's output was replaced; `authority_leaks`: routing with an adversarial
schema output changed what the verifier reads, the capacities or the competence; `nondeterminism`) and four coverage rules (arms trained, the two stream sizes, the wired control law moves the
routing). A failing rule FALSIFIES (kind structural) **when no INVALID reason precedes it**; a structural FALSIFIED is a statement about the implementation, not about the hypothesis. Twelve **integrity**
counters (FLOP and parameter matching, **the history router is not blind** (its routing moves when the history it is given changes), stream overlap, verifier audit, oracle bound, demand schedule, loss and plateau
bookkeeping, the epsilon and shuffle controls) make the seed exit 2 and the experiment INVALID. Eight negative controls run on doctored inputs and must trip or the run is exit 2.

**Validity guards:** the best comparator must leave 40 per mille below the oracle (the margin is reachable) and beat a uniform router by 100 per mille on every seed; the schema must beat
persistence ("the next batch is like the last") by 200 per mille of L1 distance on all but one seed (a schema that did not learn cannot control); no more than one seed with a judged arm at the
step cap; **gross control** (schema-free minus a uniform router, lower bound at least the margin); **graded control** (schema-free minus its 15%-epsilon-routed twin, lower bound above 12); **the stationary null
world** (control_null minus schema_free_null: the lower bound must stay below the margin **and**, as a tie is required, the upper bound of the cost schema_free_null minus control_null must stay below a **cost band of
50 per mille**, two margins: the schema may cost something where it cannot help, not more). A failing guard or control is INVALID.

**Order of the verdicts** (`aggregate.py decide`, pure, tested): (1) an integrity counter above 0 on any seed; (2) a missing, duplicated or undeclared seed, missing or partial mutation evidence, a seed that cannot be
judged, an environment that differs between seeds: **INVALID**; (3) a structural counter, a coverage rule or failed static evidence: FALSIFIED (structural); (4) a guard on some seed, more than the allowed seeds with an
untrained schema or at the step cap: INVALID; (5) a control: INVALID; (6) only then the paired verdict.

**Margin: 25 per mille of verified-token rate**, from the planning model (`planning.py`, section 6): a per-seed paired difference has a standard deviation of about 18 per mille under the stated
priors, the margin must not sit inside it (1.2 sigma = 22), and 2.5 points of verified tokens is the smallest gain that justifies adding a second network, a history buffer and a training signal to
the router. **Statistics.** Over 20 seeds the one-sided 95% t quantile for 19 degrees of freedom is 1.729133; the refutation uses the Bonferroni quantile over the five comparators (alpha 0.01,
2.539483): **it means that the advantage is below the margin, not that there is none.**

| Outcome | Condition | Consequence |
|---|---|---|
| COMPLETED (supported) | order of verdicts above holds; all five lower bounds (t 1.729133) >= 25 per mille **and the shuffled-history control is ATTRIBUTED** (control minus control with a shuffled history, lower bound >= 25) | **construction check passed**: on a toy built so that a schema can help, a schema used for control beats a describe-only schema, a heuristic balancer, the schema-free router (FLOP- and parameter-matched) and a router that reads the history directly, and the gain depends on the router's own history; **AST-L2 is not changed** |
| FALSIFIED (refuted) | guards and controls hold and some Bonferroni upper bound (t 2.539483) < 25 per mille | negative result for this world; against `history_router`: access to the load history, not a separate predictor with a control law, accounts for the gain here; against describe_only alone: the schema's control role is not shown; no downgrade of the GPU stage |
| NOT_SHOWN | guards hold, a bound straddles the margin, or the control beats every comparator by the margin but the shuffled-history control is not ATTRIBUTED | nothing is concluded (the reason is printed) |
| INVALID | no headroom, untrained comparators or schema, a judged arm at the cap, an unresolved control, a null world that shows an effect or a cost beyond the band, any integrity counter, seeds or mutation evidence missing or partial | nothing is concluded; repair and register a successor |

Exit status of `aggregate.py`: 0, 1, 3, 2 respectively (2 also for any refusal to aggregate).

**Power and expected outcome.** Planning table (`research/cseries/c003a/planning.py`, version 2 plus the fix-round additions: five comparators; between-seed sd 15 per mille per arm, correlation 0.3, evaluation noise 2.1, so a paired sd of 18; margin 25 from the reachable gain of 60; alternative: 45;
null: none) with rows near the margin, then the table on the fix-round pilot seeds (`research/cseries/lib/pilot_power.py`; it changes neither the margin nor the seed count).

```
sigma_d = 18.0; margin rule 1.2 sigma_d = 21.6; margin = 25
  n  alt: COMPLETED  null: COMPLETED  null: FALSIFIED  margin: NOT_SHOWN  alt, 2x sd
  5           0.121            0.000            0.929              0.951       0.002
  6           0.247            0.000            0.985              0.951       0.003
  8           0.533            0.000            1.000              0.954       0.012
 10           0.752            0.000            1.000              0.952       0.029
 12           0.881            0.000            1.000              0.952       0.058
 14           0.943            0.000            1.000              0.951       0.100  <- smallest n meeting the first three conditions
 16           0.975            0.000            1.000              0.951       0.153
 20           0.996            0.000            1.000              0.952       0.276
smallest n meeting the first three conditions: 14; the robustness condition (power >= 0.8 at twice the sd) is met by no n <= 20, so the seed count is the cap, 20
rows near the margin (n = 20, five comparators, all at the stated mean): COMPLETED / NOT_SHOWN / FALSIFIED
  mean   0.0: 0.000 / 0.000 / 1.000
  mean  10.0: 0.000 / 0.000 / 1.000
  mean  20.0: 0.000 / 0.532 / 0.468
  mean  25.0: 0.000 / 0.952 / 0.048
  mean  30.0: 0.005 / 0.994 / 0.001
  mean  35.0: 0.276 / 0.724 / 0.000
  mean  45.0: 0.996 / 0.004 / 0.000
  mean  60.0: 1.000 / 0.000 / 0.000
```

```
C008: 7 pilot seeds [1004, 1005, 1006, 1007, 1008, 1009, 1011]; margin 25 per mille; preregistered seeds 20
  contrast vs schema_free        mean     62.9  sd    2.9  per-seed [67.0, 61.8, 58.5, 61.6, 65.8, 61.4, 63.9]
  contrast vs schema_free_wide   mean     60.8  sd    4.6  per-seed [67.7, 57.4, 57.4, 57.8, 66.8, 60.8, 57.5]
  contrast vs heuristic          mean     61.9  sd    3.0  per-seed [67.4, 61.0, 58.2, 60.0, 63.1, 60.2, 63.2]
  contrast vs describe_only      mean     60.9  sd    3.9  per-seed [67.2, 57.1, 58.7, 59.8, 65.5, 60.1, 57.9]
  contrast vs history_router     mean      6.4  sd    6.2  per-seed [14.5, 0.6, 3.5, 9.1, 1.9, 14.4, 1.0]
  pilot covariance                     COMPLETED 0.000  NOT_SHOWN 0.000  FALSIFIED 1.000  (guards and controls assumed to hold)
  twice the pilot standard deviation   COMPLETED 0.000  NOT_SHOWN 0.000  FALSIFIED 1.000  (guards and controls assumed to hold)
  control gross (schema-free - random router)        pilot mean   582.6 sd    1.4 threshold 25: resolves at n=20 with probability 1.000
  control graded (schema-free - epsilon routing)     pilot mean    65.0 sd    2.5 threshold 12: resolves at n=20 with probability 1.000
  null world             forward pilot mean   -14.1 sd   25.6: lower bound stays below 25 with probability 1.000
  null world             reverse pilot mean    14.1 sd   25.6: lower bound stays below 25 with probability 1.000
  check null world cost (schema-free - control, stationary)        pilot mean    14.1 sd   25.6 (upper< 50): holds at n=20 with probability 1.000
  check attribution (control - shuffled history)                   pilot mean    61.2 sd    3.8 (lower>= 25): holds at n=20 with probability 1.000
  guard 'headroom_permille >= 40': scalar mean 60.4 sd 4.7, fails on 0 of 7 pilot seeds; normal-fit per-seed failure probability 0.0000, P(some of 20 seeds fails) = 0.000
  guard 'comparator_above_random_permille >= 100': scalar mean 639.0 sd 5.5, fails on 0 of 7 pilot seeds; normal-fit per-seed failure probability 0.0000, P(some of 20 seeds fails) = 0.000
```

**Expected outcome (honest).** With the pilot covariance the decision rule gives **FALSIFIED (refuted) with probability about 1** (the control arm is +6.3 per mille above the history router, adjusted upper bound near 13, against a margin of 25), and also with twice the pilot standard deviation (about 1.00; the control arm's sd of the contrast over the history router is 6.0 per mille); COMPLETED is below 0.01. The planning model at 20 seeds would give a COMPLETED probability 0.996 at an alternative of 45 against every comparator,
but the pilots say the alternative does not hold against the sixth, the history router. Validity risks: a judged arm at the step cap on more than one seed (none of the seven pilot seeds had a judged arm at the cap of 24000 steps; the control arm needed about 19000 steps in the grid, close to the cap, so this risk is real on a fresh seed), an untrained schema, a guard on a fresh seed. My expectation: **FALSIFIED (refuted) about 0.87, INVALID about 0.10, NOT_SHOWN about 0.02, COMPLETED below 0.01.**
The experiment is registered as a construction-validity check (section 2); its likeliest result is the finding that, in the world built so that a schema can help, a router with the same history and no schema helps as much.

## 5. Mutation evidence

`scripts/mutation_check.py C008 --check` verifies the anchors of `tests/mutations.toml` (42 planted defects; every `find` occurs exactly once in its file). The defects are run by
`research/cseries/lib/run_mutations.py C008` (planted one at a time in a throwaway copy, never in the working tree), because `scripts/mutation_check.py` builds a Cargo package and cannot plant a defect in a Python
entrypoint (a gap reported, as C001 reported it). A defect is killed in one of three ways, stated in the plan's header: the reduced-scale harness exits 1 with an expected structural counter or coverage rule; exits 2 naming an
expected integrity counter or negative control; or the named unit tests fail in the planted copy while they pass in the unmutated one. **Every structural and integrity counter, every coverage rule and each of the negative controls is expected by at least one mutation** (a test pins that). New in the fix round:
the history router that ignores the history, its blinded detector, its doubled width.

**The aggregator is mutated too** (`research/cseries/lib/run_agg_mutations.py C008`, plan `tests/aggregator_mutations.toml` and the shared library's `lib_aggregator_mutations.toml`): 65 planted defects in
`aggregate.py`, `criteria.toml` and the aggregation library (every guard, control and the null cost band, the Bonferroni and support quantiles, the allowance counts, the attribution gate and its three-valued reading, each
branch and the precedence of the verdicts, the quantile and bound arithmetic, the exit-status agreement), each of which must fail `scripts/tests/test_c008_aggregate.py`, the library tests or the rule-agreement tests, which pass unmutated.

**Run on the uncommitted tree before the freeze (no record written): harness plan 42 of 42 killed; aggregator plan 65 of 65 killed.** `results/mutations.json` is written by the head after the freeze, from the clean tagged tree.

## 6. Pilot disclosure and tuning (pilot seeds 1001 and up only; none is declared)

**Chronology (in order; every number is on a pilot seed, 1001 or higher; nothing was run on a declared seed; file times are the box's clock on 2026-10-10 for version 1 and are quoted from the files).**

*Version 1: the iterations of the design.* The planning model, version 1 (margin 30, alternative 50, 19:50) preceded the first design. **Design table** (seed 1001, the best reward of each arm in the pilot file; per mille; the "what" column is the change made before that file was produced):

| design | file time (2026-10-10) | what | schema-free | describe-only | heuristic | control (best law) | control minus schema-free |
|---|---|---|---|---|---|---|---|
| D1 | 23:28 | first design: 64 tokens per batch, capacity 14, 70% hot demand; the schema's prediction concatenated to the router's input | 717 | 713 |  | 692 | -25 |
| D2 | 23:28 | D1's world; the control arm with an additive control law | 710 | 714 |  | 692 | -18 |
| D3 | 23:28 | D2's world; history and target made of the router's propensity (the form kept), laws full / diagonal / scalar | 714 | 719 | 686 | 694 | -20 |
| D4 | 23:28 | revised world (256 tokens, capacity 64, 80% hot, eight regimes, hot pair (2r, 2r+1)), feature noise 0.5 | 754 | 757 | 732 | 752 | -2 |
| D5 | 23:28 | D4 with feature noise 0.25 | 893 | 892 | 858 | 881 | -12 |
| D6 | 23:28 | D5 again (the same world, laws full / diagonal) | 893 | 891 |  | 887 | -6 |
| final | 22:51-23:08 | hot pair (r, r+1) (after the hand-set-law sweep), the pinned configuration, seeds 1004-1009 (first version's pilots; heuristic re-selected before them, see step 6) | 832 |  |  | 893 |  |

1. D1 to D3 (19:58 to 20:18): in three architectures of the control arm (the schema's prediction concatenated to the router's input, an additive control law, and the form kept) the control arm was 20 to 30 per mille **below** the schema-free router; the world was small (64 tokens, capacity 14).
2. A calculation without learning (`research/cseries/c003a/pilots/regime_value.py`) showed that in that world a router that knew the regime could gain only about 30 per mille over the even split, so the alternative of version 1 was unattainable. **The world was changed** (256 tokens per batch, capacity 64, 80 per cent hot demand): even split 0.847, best regime-aware independent split 0.907, oracle 0.951.
   **Planning model, version 2** (20:18; margin 25, alternative 45) was derived from that reachable gain and the sd prior, not from a pilot result of the control arm.
3. D4 (20:23): in the revised world the schema-free router reached 754 against an even split of 847: it could not recognise the class from its noisy features (noise 0.5). **The feature noise was lowered to 0.25** (D5, 20:27; a variant of the schema loss was also looked at in that file and not kept). In D5 and D6 (20:31) **the control arm was 13 per mille below the schema-free router** (880 against 893; with the diagonal law 881, 887 in D6): this is the pilot that
   motivated the next change.
4. **A hand-set-law sweep on seed 1001** (a regime-blind router 0.837; a hand-set control law with the first hot pair (2r, 2r+1) added -1 per mille, with the pair (r, r+1) +62; the sweep's output was not kept as a file, these are the numbers recorded in version 1 of this text, and `pilots/regime_value.py` computes the no-learning reachable gains) **motivated the change of the hot pair to (r, r+1)** with eight regimes: with the first pair every class always had the same role and a regime-blind router already reached 0.895. These are defects of the world and not tuning of a result, but they are choices made with the
   control arm's results in view, and the world was built so that a schema can help: see section 7.
5. Grids on seeds 1001 to 1003 in the final world (20:54 free arms, 21:21 control, 21:39 balancer; tables 2 to 4 below): learning rate {0.001, 0.003, 0.01} for every trained arm; additionally the control arm's control-law form and the balancer's step. Selected by the mean evaluation reward among the configurations with at most one of three seeds at the step cap.
6. First set of second-power-table pilots (seeds 1004 to 1009, 21:56 to 22:15). **The heuristic arm was then re-selected** (learning rate 0.003 to 0.01, step 0.03 to 0.01: the configuration with the highest mean reward in table 4, 836.3 against 836.0 for the earlier one; the record of why the first set used the other is not preserved) and a second set (22:51 to 23:08) was run; the numbers of the other arms differ slightly between the two
   sets (for example the control arm's tokens on seed 1004: 54762 against 54788), so something other than the heuristic arm also changed between the sets. **That is the initial-weight seeding defect found in this fix round (step 6b)**: the networks were built before the random state was seeded, so no two runs were identical; the earlier text called it a difference that could not be reconstructed. The first set is kept in `research/cseries/c003a/pilots/data/before-fix-round/before-heuristic-reselection/`. In the second set the control arm verified about 60 per mille more tokens than every comparator.

6b. **A product defect found in this fix round and fixed** (disclosed because it concerns every grid number below): the harness built every arm's network before seeding its random state, so the initial weights depended on the process's global random state and a run could not be reproduced. The structural counter `nondeterminism` compared only the first steps of two seeded trainings and could not see it. Every arm's network is now built from a generator seeded by (experiment, seed, arm), the counter also builds the arms twice under two different global seeds and compares digests (mutation `initial-weights-are-not-seeded`), and a test runs the harness twice and compares the records. **Every grid, remedy and selection row of this text that predates the fix (tables 2 to 6, `grid-*.jsonl`, `rerun-cap24.jsonl`, `before-fix-round/`) is one draw from a random initialisation per cell; the configurations were selected from these unseeded single draws, and the noise of the selection includes the initialisation.** The grids were not re-run after the fix: the selected configurations, the cap and the entropy dimension stand as chosen, and only the pilots of table 7 (seeds 1004 to 1009 and the dry run) and the mutation runs were repeated after the fix. The 12000 to 24000 cap decision (step 9) rests on a pre-fix observation of two seeds at the cap, and was confirmed by the post-fix pilots (no judged arm at the cap on any of seven seeds). C004's harness had the same defect and was fixed the same way; C007 seeds before building (a test confirms).

*The review and the fix round.* An independent pre-freeze review found that the comparators lacked the natural strong baseline (a router that reads the history directly), that the shuffled-history attribution was reported and not judged, that the doc-38 consequence of a COMPLETED overstated it, that INVALID did not precede a structural FALSIFIED for missing evidence, that the tuning breadth was unequal (the schema-free arms had 3 configurations, the control 9), and that the chronology was incomplete. The head's decisions are implemented here;
the framing of the whole experiment changed (section 2).

7. **Planning additions** (`planning.py`, "FIX ROUND"): five comparators (Bonferroni 2.539483), a null-world cost band of 50 per mille (two margins; the old pilots had shown a mean cost of about 19 per mille with sd about 19, and the band was written with that in view), the attribution gate. The margin, the alternative and the seed count were not moved.
8. **The history router** was added (parameter-matched, reads the flattened history of its own propensity as relative loads) with nine configurations (learning rate x entropy coefficient), and the other schema-free-type arms got the entropy-coefficient dimension so that **every arm has nine configurations** (table 2). **The selections did not change** for the four old arms; the history router's best eligible configuration at the first cap was learning rate 0.003, entropy 0.005 (885.7).
   In the first run of the second set of pilots in this round (cap 12000 steps, seeds 1004 to 1009 and the dry run 1011) **a judged arm hit the step cap on 2 of 6 seeds** (1005: history_router, 1009: control); with the allowance of one seed the run would have been INVALID with probability near 1.
9. **Remedies tried on seeds 1005 and 1009 for the two arms that hit the cap** (table 5): a larger minimum improvement (6 per mille, about two standard errors of the validation estimate) converged earlier but under-trained the history router on seed 1009 (856 against 878), which would have favoured the control: rejected; **a larger cap (24000 steps)**: all four runs converged (control 886 and 901; history router 893 and 895). The cap was raised to 24000 steps for the task arms (12000 for the null-world arms); the 22 grid runs that had hit the old cap were re-run at the new one
   (the other grid cells were not re-run: they were single draws that did not hit the cap); **the selections changed for two arms**: the control arm to learning rate 0.001 with the diagonal law (901.3; it needs about 19000 steps) and the history router to learning rate 0.003 with entropy 0.02 (890.7); the null-world twin of the control arm got the control arm's learning rate. Table 6 lists the selection.
10. The pinned values were set and seeds 1004, 1005, 1006, 1007, 1008, 1009 ran with them; the dry run (seed 1011) went through the pinned entry point in the runner's stripped environment (the last row of table 7): no counter, rule or guard fired and every control resolved.

Table 2. Schema-free, wide schema-free, describe-only and history-router arms (seeds 1001 to 1003; mean evaluation reward in per mille). The rows with entropy coefficient 10 are the first version's grid (same world and code for those arms; `before-fix-round/grid-free.jsonl`); runs that hit the old cap were re-run at 24000 steps (`pilots/data/rerun-cap24.jsonl`) and the re-run rows are not in this table (the table shows the first runs).

| arm | lr | entropy coefficient (milli) | seeds | mean reward | at step cap |
|---|---|---|---|---|---|
| describe_only | 0.001 | 10 | 3 | 833.3 | 0 |
| describe_only | 0.001 | 20 | 3 | 834.0 | 0 |
| describe_only | 0.001 | 5 | 3 | 835.0 | 1 |
| describe_only | 0.003 | 10 | 3 | 837.0 | 0 |
| describe_only | 0.003 | 20 | 3 | 836.0 | 0 |
| describe_only | 0.003 | 5 | 3 | 835.7 | 0 |
| describe_only | 0.01 | 10 | 3 | 835.0 | 0 |
| describe_only | 0.01 | 20 | 3 | 836.3 | 0 |
| describe_only | 0.01 | 5 | 3 | 835.7 | 0 |
| history_router | 0.001 | 10 | 3 | 879.3 | 0 |
| history_router | 0.001 | 20 | 3 | 881.3 | 2 |
| history_router | 0.001 | 5 | 3 | 885.0 | 3 |
| history_router | 0.003 | 10 | 3 | 883.7 | 1 |
| history_router | 0.003 | 20 | 3 | 894.0 | 3 |
| history_router | 0.003 | 5 | 3 | 885.7 | 0 |
| history_router | 0.01 | 10 | 3 | 821.7 | 1 |
| history_router | 0.01 | 20 | 3 | 881.0 | 0 |
| history_router | 0.01 | 5 | 3 | 882.7 | 0 |
| schema_free | 0.001 | 10 | 3 | 833.0 | 1 |
| schema_free | 0.001 | 20 | 3 | 829.3 | 0 |
| schema_free | 0.001 | 5 | 3 | 831.7 | 0 |
| schema_free | 0.003 | 10 | 3 | 834.0 | 0 |
| schema_free | 0.003 | 20 | 3 | 835.7 | 0 |
| schema_free | 0.003 | 5 | 3 | 835.3 | 0 |
| schema_free | 0.01 | 10 | 3 | 837.0 | 0 |
| schema_free | 0.01 | 20 | 3 | 836.3 | 0 |
| schema_free | 0.01 | 5 | 3 | 834.7 | 0 |
| schema_free_wide | 0.001 | 10 | 3 | 834.3 | 0 |
| schema_free_wide | 0.001 | 20 | 3 | 833.7 | 0 |
| schema_free_wide | 0.001 | 5 | 3 | 835.0 | 0 |
| schema_free_wide | 0.003 | 10 | 3 | 836.3 | 0 |
| schema_free_wide | 0.003 | 20 | 3 | 834.7 | 0 |
| schema_free_wide | 0.003 | 5 | 3 | 835.3 | 0 |
| schema_free_wide | 0.01 | 10 | 3 | 835.3 | 0 |
| schema_free_wide | 0.01 | 20 | 3 | 835.3 | 0 |
| schema_free_wide | 0.01 | 5 | 3 | 834.0 | 0 |

Table 3. Control arm (first version's grid, `before-fix-round/grid-control.jsonl`; the control arm's capped runs are NOT in this table).

| arm | lr | law | seeds | mean reward | at step cap |
|---|---|---|---|---|---|
| control | 0.001 | diagonal | 3 | 891.7 | 3 |
| control | 0.001 | full | 3 | 889.3 | 1 |
| control | 0.001 | scalar | 3 | 898.3 | 3 |
| control | 0.003 | diagonal | 3 | 894.3 | 2 |
| control | 0.003 | full | 3 | 892.3 | 0 |
| control | 0.003 | scalar | 3 | 897.0 | 0 |
| control | 0.01 | diagonal | 3 | 894.7 | 0 |
| control | 0.01 | full | 3 | 889.3 | 0 |
| control | 0.01 | scalar | 3 | 896.7 | 0 |

Table 3a. The runs that hit the old cap, re-run at 24000 steps (`pilots/data/rerun-cap24.jsonl`; each replaces the original run of the same seed, arm and configuration).

| seed | arm | lr | law | entropy (milli) | balancer step | reward | steps | converged |
|---|---|---|---|---|---|---|---|---|
| 1001 | control | 0.001 | diagonal | 10 | 0.01 | 900 | 19250 | 1 |
| 1002 | control | 0.001 | diagonal | 10 | 0.01 | 902 | 19500 | 1 |
| 1003 | control | 0.001 | diagonal | 10 | 0.01 | 902 | 18500 | 1 |
| 1002 | control | 0.001 | full | 10 | 0.01 | 891 | 12750 | 1 |
| 1001 | control | 0.001 | scalar | 10 | 0.01 | 903 | 16250 | 1 |
| 1002 | control | 0.001 | scalar | 10 | 0.01 | 901 | 15500 | 1 |
| 1003 | control | 0.001 | scalar | 10 | 0.01 | 895 | 12500 | 1 |
| 1001 | control | 0.003 | diagonal | 10 | 0.01 | 904 | 14000 | 1 |
| 1003 | control | 0.003 | diagonal | 10 | 0.01 | 890 | 9000 | 1 |
| 1002 | describe_only | 0.001 | scalar | 5 | 0.01 | 838 | 12500 | 1 |
| 1002 | heuristic | 0.001 | scalar | 10 | 0.03 | 833 | 7500 | 1 |
| 1001 | history_router | 0.001 | scalar | 5 | 0.01 | 877 | 8500 | 1 |
| 1002 | history_router | 0.001 | scalar | 5 | 0.01 | 884 | 11000 | 1 |
| 1003 | history_router | 0.001 | scalar | 5 | 0.01 | 874 | 8250 | 1 |
| 1001 | history_router | 0.001 | scalar | 20 | 0.01 | 884 | 13250 | 1 |
| 1002 | history_router | 0.001 | scalar | 20 | 0.01 | 871 | 7250 | 1 |
| 1002 | history_router | 0.003 | scalar | 10 | 0.01 | 895 | 11250 | 1 |
| 1001 | history_router | 0.003 | scalar | 20 | 0.01 | 889 | 9500 | 1 |
| 1002 | history_router | 0.003 | scalar | 20 | 0.01 | 888 | 10500 | 1 |
| 1003 | history_router | 0.003 | scalar | 20 | 0.01 | 895 | 13500 | 1 |
| 1002 | history_router | 0.01 | scalar | 10 | 0.01 | 886 | 7250 | 1 |
| 1002 | schema_free | 0.001 | scalar | 10 | 0.01 | 835 | 9250 | 1 |

Table 3b. Control arm after the replacement (the table the selection was made on: 0.001 with the diagonal law, 901.3, is the best of the configurations with at most one seed at the cap).

| arm | lr | law | seeds | mean reward | at step cap |
|---|---|---|---|---|---|
| control | 0.001 | diagonal | 3 | 901.3 | 0 |
| control | 0.001 | full | 3 | 889.7 | 0 |
| control | 0.001 | scalar | 3 | 899.7 | 0 |
| control | 0.003 | diagonal | 3 | 894.0 | 0 |
| control | 0.003 | full | 3 | 892.3 | 0 |
| control | 0.003 | scalar | 3 | 897.0 | 0 |
| control | 0.01 | diagonal | 3 | 894.7 | 0 |
| control | 0.01 | full | 3 | 889.3 | 0 |
| control | 0.01 | scalar | 3 | 896.7 | 0 |

Table 2b. Schema-free, wide, describe-only and history-router arms after the replacement.

| arm | lr | entropy coefficient (milli) | seeds | mean reward | at step cap |
|---|---|---|---|---|---|
| describe_only | 0.001 | 10 | 3 | 833.3 | 0 |
| describe_only | 0.001 | 20 | 3 | 834.0 | 0 |
| describe_only | 0.001 | 5 | 3 | 834.7 | 0 |
| describe_only | 0.003 | 10 | 3 | 837.0 | 0 |
| describe_only | 0.003 | 20 | 3 | 836.0 | 0 |
| describe_only | 0.003 | 5 | 3 | 835.7 | 0 |
| describe_only | 0.01 | 10 | 3 | 835.0 | 0 |
| describe_only | 0.01 | 20 | 3 | 836.3 | 0 |
| describe_only | 0.01 | 5 | 3 | 835.7 | 0 |
| history_router | 0.001 | 10 | 3 | 879.3 | 0 |
| history_router | 0.001 | 20 | 3 | 876.3 | 0 |
| history_router | 0.001 | 5 | 3 | 878.3 | 0 |
| history_router | 0.003 | 10 | 3 | 884.0 | 0 |
| history_router | 0.003 | 20 | 3 | 890.7 | 0 |
| history_router | 0.003 | 5 | 3 | 885.7 | 0 |
| history_router | 0.01 | 10 | 3 | 876.7 | 0 |
| history_router | 0.01 | 20 | 3 | 881.0 | 0 |
| history_router | 0.01 | 5 | 3 | 882.7 | 0 |
| schema_free | 0.001 | 10 | 3 | 831.3 | 0 |
| schema_free | 0.001 | 20 | 3 | 829.3 | 0 |
| schema_free | 0.001 | 5 | 3 | 831.7 | 0 |
| schema_free | 0.003 | 10 | 3 | 834.0 | 0 |
| schema_free | 0.003 | 20 | 3 | 835.7 | 0 |
| schema_free | 0.003 | 5 | 3 | 835.3 | 0 |
| schema_free | 0.01 | 10 | 3 | 837.0 | 0 |
| schema_free | 0.01 | 20 | 3 | 836.3 | 0 |
| schema_free | 0.01 | 5 | 3 | 834.7 | 0 |
| schema_free_wide | 0.001 | 10 | 3 | 834.3 | 0 |
| schema_free_wide | 0.001 | 20 | 3 | 833.7 | 0 |
| schema_free_wide | 0.001 | 5 | 3 | 835.0 | 0 |
| schema_free_wide | 0.003 | 10 | 3 | 836.3 | 0 |
| schema_free_wide | 0.003 | 20 | 3 | 834.7 | 0 |
| schema_free_wide | 0.003 | 5 | 3 | 835.3 | 0 |
| schema_free_wide | 0.01 | 10 | 3 | 835.3 | 0 |
| schema_free_wide | 0.01 | 20 | 3 | 835.3 | 0 |
| schema_free_wide | 0.01 | 5 | 3 | 834.0 | 0 |

Table 4. Heuristic balancer (first version's grid). No run of this grid hit the old cap in a way that was re-run except the 0.001 / step 0.03 cell on seed 1002 (one row of `rerun-cap24.jsonl`), and the selected cell (learning rate 0.01, step 0.01, 836.3) is not affected, so there is no after-replacement table.

| arm | lr | balancer step | seeds | mean reward | at step cap |
|---|---|---|---|---|---|
| heuristic | 0.001 | 0.01 | 3 | 832.0 | 0 |
| heuristic | 0.001 | 0.03 | 3 | 832.7 | 1 |
| heuristic | 0.001 | 0.1 | 3 | 831.7 | 0 |
| heuristic | 0.003 | 0.01 | 3 | 835.3 | 0 |
| heuristic | 0.003 | 0.03 | 3 | 836.0 | 0 |
| heuristic | 0.003 | 0.1 | 3 | 835.3 | 0 |
| heuristic | 0.01 | 0.01 | 3 | 836.3 | 0 |
| heuristic | 0.01 | 0.03 | 3 | 835.7 | 0 |
| heuristic | 0.01 | 0.1 | 3 | 834.3 | 0 |

Table 5. Remedies for the step-cap hits (seeds 1005 and 1009, learning rate 0.003, one run each).

| seed | arm | setting | reward | steps | converged |
|---|---|---|---|---|---|
| 1005 | control | cap 24000 | 886 | 6250 | 1 |
| 1005 | history_router | cap 24000 | 893 | 14000 | 1 |
| 1009 | control | cap 24000 | 901 | 10500 | 1 |
| 1009 | history_router | cap 24000 | 895 | 12000 | 1 |
| 1005 | control | min_delta 6 | 885 | 6250 | 1 |
| 1005 | history_router | min_delta 6 | 887 | 11500 | 1 |
| 1009 | control | min_delta 6 | 900 | 10500 | 1 |
| 1009 | history_router | min_delta 6 | 856 | 4750 | 1 |

Table 6. Configuration selected for each arm after the re-runs at the cap of 24000 steps (mean over seeds 1001 to 1003 of the selected configuration, among those with at most one seed at the cap).

| arm | learning rate | other setting | mean reward over seeds 1001-1003 |
|---|---|---|---|
| schema_free | 0.01 | entropy 0.01 | 837.0 |
| schema_free_wide | 0.003 | entropy 0.01 | 836.3 |
| describe_only | 0.003 | entropy 0.01 | 837.0 |
| heuristic | 0.01 | balancer step 0.01 | 836.3 |
| history_router | 0.003 | entropy 0.02 | 890.7 |
| control | 0.001 | law diagonal | 901.3 |

Table 7. Final world and pinned configuration on pilot seeds (verified-token rate, per mille; the last row is the dry run).

| seed | control | schema-free | schema-free wide | heuristic | describe-only | history router | control shuffled history | regime-blind reference | epsilon routing | uniform | oracle | null: schema-free | null: control | cpu s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1004 | 901 | 834 | 833 | 834 | 834 | 887 | 833 | 839 | 766 | 250 | 947 | 927 | 897 | 216 | 2061 |
| 1005 | 893 | 831 | 836 | 832 | 836 | 892 | 832 | 840 | 767 | 250 | 950 | 926 | 925 | 204 | 1998 |
| 1006 | 891 | 832 | 834 | 833 | 832 | 887 | 834 | 838 | 770 | 250 | 948 | 895 | 920 | 194 | 1840 |
| 1007 | 893 | 832 | 836 | 833 | 833 | 884 | 833 | 843 | 768 | 250 | 950 | 924 | 923 | 190 | 1754 |
| 1008 | 898 | 833 | 832 | 835 | 833 | 897 | 835 | 842 | 769 | 250 | 951 | 922 | 869 | 190 | 1694 |
| 1009 | 896 | 835 | 836 | 836 | 836 | 882 | 837 | 842 | 766 | 250 | 949 | 926 | 917 | 196 | 1789 |
| 1011 | 894 | 830 | 836 | 831 | 836 | 893 | 835 | 838 | 767 | 249 | 950 | 899 | 869 | 196 | 2962 |

What the pilots say, including what is unwelcome: (a) **the control arm verifies about 63 per mille more tokens than the schema-free router and 61 more than the describe-only schema** (the margin is 25), and a shuffled history removes the gain (attribution 61); (b) **but a router that reads the same history directly does about as well: the control arm is only +6.3 per mille above the history router (sd 6.0; per seed +14, +1, +4, +9, +1, +14, +1)**, below the margin with an adjusted upper bound near 13: **the registered outcome is
FALSIFIED (refuted) against `history_router`**, i.e. in this world access to the history, not a separate predictor with a control law, accounts for the gain; (c) the trained schema-free arms sit 8 per mille below the hand-set regime-blind reference, so part of the contrast over them is comparator shortfall and not schema value (the history router is +49 per mille from it);
(d) the earlier designs failed and the final world was built so that it can show a gain (section 7); (e) in the stationary null world the schema-controlled arm is not better than the schema-free router (mean cost +14 per mille).


## 7. Limits

* A result applies to this toy world: eight experts, a cyclic demand of eight regimes of three batches, capacity 64 of 256 tokens, a history of four batches, a router of one hidden layer, policy-gradient training. The schema can
  only help where the demand is predictable from the router's own load history, which this world builds in on purpose (the stationary null world runs beside it). It says nothing about real routing traces, OLMoE or the GPU stage B.
* **A COMPLETED is a construction check and AST-L2 is unchanged.** The world and the control mechanism were changed several times on pilot seeds before the final design (section 6); the changes were made to remove defects of the world
  (a class that could not be recognised; a layout in which a regime-blind router was already near the regime-aware optimum) and to give the control arm a form of control law that can express the needed correction, and they were made while
  looking at the control arm's pilot results. This favours the hypothesis in the direction of making a gain possible; it does not guarantee that policy-gradient learning finds it, and a comparator that reads the history directly may find it as well.
* The control-versus-describe-only contrast is not independent of the control-versus-schema-free contrast (the two comparators share the control arm's training and differ in small ways), so the judged contrasts are correlated and the
  Bonferroni correction is conservative for them.
* The heuristic balancer is the schema-free router with an auxiliary-loss-free bias moved by the previous batch's load; its learning rate and step were tuned on pilot seeds with nine configurations like every other arm. A better-tuned or
  differently shaped balancer (a forecast, an integral term) might close the gap; the balancer is a lagged controller by design, and a world with a cycle punishes a lag.
* The history router is parameter-matched (about twice the control arm's FLOPs) and reads the history as relative loads; another way of feeding the history (a recurrent cell, a longer window) is another comparator.
* The trained schema-free comparators sit below the hand-set regime-blind reference with true classes (about 14 per mille in the first version): the distance of every arm to the reference is reported per seed.
* The schema is trained by its prediction loss alone, detached from the policy; a joint training is another experiment.
* Recovery after an expert is revoked and routing churn between policy versions are not judged at stage A (recovery needs an online adaptation protocol and belongs to stage B); load balance, entropy and the schema's gain over persistence are reported.
* The t bounds assume roughly normal paired differences, which 20 points cannot check; per-seed differences, medians and both bounds are reported.
* An indicator property here is a functional property: the experiment says nothing about whether any system is conscious.

## 8. Entry point, environment and freeze procedure

The entrypoint is `env UV_PROJECT_ENVIRONMENT=../../../c001-venv PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline --no-python-downloads
--project research/consciousness python -B research/cseries/c003a/harness.py --benchmark c008 --seed <seed>`, and the manifest declares the matching
`preflight` command (the same program and environment with `--preflight` instead of `--seed <seed>`; the runner runs it, in the entrypoint's exact environment,
before it reserves a seed, and `python3 scripts/run_experiment.py preflight C008` runs only it).

Reasons, the same as C001's (found there by rehearsing the
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
198 CPU seconds (section 3), so the 20 seeds take well under an hour of CPU time and more of wall time on a loaded shared machine.



**Freeze procedure.** Every step is a command in the repository; nothing refers to a script outside it. The two sha256 values of the `[preregistration]` table and the manifest digests read
`must-be-pinned-before-prepared` (absent) until step 4. Run every `python3` call with `export PYTHONDONTWRITEBYTECODE=1`. Commit messages carry the attribution trailer the session requires.

1. **Clean the tree and commit it** (the pre-freeze commit). The commit must include the untracked `research/cseries/`; `git status` shows new files as untracked until added.
   `find . -name __pycache__ -type d -prune -exec rm -rf {} +`; the repository must hold no git-ignored file when the runner starts.
2. **Check, before the digests are pinned:** `python3 scripts/run_experiment.py validate`; `python3 scripts/check_research_gates.py`; `python3 scripts/mutation_check.py C008 --check`;
   `python3 research/cseries/lib/run_agg_mutations.py C008 --check`; `python3 -m unittest scripts.tests.test_c008_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests scripts.tests.test_ci_local`;
   the research tests with the locked interpreter, `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B -m unittest discover -s research/cseries/tests`; `python3 scripts/update_component_docs.py --check`; `python3 scripts/check_repo.py`.
3. **Run the aggregator mutation plan** (seconds): `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_agg_mutations.py C008 --write`; every defect must be killed; commit the written `tests/aggregator-mutations.json`.
4. **Pin the digests:** `python3 research/cseries/lib/pin_freeze.py C008` (it sets `preregistration_text_sha256` and `criteria_sha256` in `config.toml`, the two manifest digests and `status = "prepared"` in
   `experiment.toml`, and `status = "prepared"` in `experiments/registry.toml`; it refuses a tree that is not a planned, unpinned one; `--dry-run` shows what it would write). Then `python3 scripts/run_experiment.py validate`,
   `python3 scripts/check_research_gates.py`, `python3 -m unittest scripts.tests.test_c008_aggregate scripts.tests.test_research_gates.EnrolledExperimentTests`; commit ("freeze preregistration") and tag it `c008-prereg`.
   The mutation evidence must run on this committed tree: `aggregate.py` refuses if `research/cseries` or this directory differs from the commit the mutation record names.
5. **Mutation evidence of the harness**, from the clean tagged tree: `OMP_NUM_THREADS=1 ../c001-venv/bin/python -B research/cseries/lib/run_mutations.py C008` (writes `results/mutations.json`); commit it.
6. **Prepare and run:** `python3 scripts/run_experiment.py preflight C008`, `python3 scripts/run_experiment.py prepare C008`, and commit; then for each seed 7001 to 7020, once each:
   `python3 scripts/run_experiment.py run C008 --seed <s>` and **commit the new run record before the next seed**. A seed whose run exits 2 is not repeated by hand: stop and report; the experiment is INVALID and the repair is
   a successor registered with `scripts/new_experiment.py` (a prepared experiment is never patched).
7. **Aggregate:** `python3 experiments/consciousness/C008-attention-schema-routing-cpu/aggregate.py --report report.md`; then `python3 scripts/check_research_gates.py` and `python3 scripts/run_experiment.py validate`; advance the
   registry status by hand only for COMPLETED or FALSIFIED, as `recommended_status` says.
