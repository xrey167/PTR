# F002 weak-supervision: preregistration

Pre-freeze text of the preregistration of F002; at the freeze it becomes the `preregistration_text` file of
`config.toml`'s `[preregistration]` table and its SHA-256 is frozen with the table. F002 is `planned`: nothing here
is a result and no run is evidence. The decisions are made by `criteria.toml` and applied by `aggregate.py`; this
file gives their reasons, the definitions, the modelling choices, what the harness cannot show and the limits of the
claim. It changes no success criterion of the original `experiment.toml` beyond what section 6 states. The harness
is `bins/ptr-bench/src/experiments/f002/` (subcommand `weak-supervision <seed>`); the products under test are
`crates/ptr-labeling` (`VoteMatrix`, `fit_label_model`, `resolve`, `rank_for_annotation`, `evaluate`,
`function_accuracy`, `EvaluationSet`) and the statistics kernel `crates/ptr-analytics` it scores with
(`brier_score`, `expected_calibration_error`, `krippendorff_alpha_nominal`, `wilson_interval`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- weak-supervision <seed>`

Claim and scope (narrowed after review). On synthetic labelling worlds with a known truth, the pipeline of
`ptr-labeling` (Dawid-Skene with verifier precedence) produces posteriors that are **better calibrated than the naive
vote-share posterior `(n_c + 1) / (n + C)`** of a majority vote at the same annotation budget, on a uniform gold
sample, and labels that are **not less accurate than the plain majority vote**, which ignores the verifiers; no label
and no resolved distribution ever contradicts a verifier veto; and the fitted model, the resolution and the metrics
are what their documentation says (an independent oracle). **This does not isolate Dawid-Skene or verifier
precedence**: the homogeneous control W3 shows the same calibration gain without either, a calibrated vote share
(`mvc`, descriptive) is as well calibrated as the label model in W3, and the accuracy gate is passed largely because
the plain vote ignores the verifiers (section 8). Gold is `GoldSource::Oracle` (`gold.rs`, a documented use): this
is a mechanism experiment, not a claim about real annotators, LLM labellers or real verifiers.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite
is written `>= N`, a "> 0" violation `<= 0`). COMPLETED iff every seed meets every coverage rule and trips no
falsifier **and the mutation evidence is complete**: `results/mutations.json` is present for this checkout's code
and plan and `killed == total ==` the number of mutations `tests/mutations.toml` lists. FALSIFIED if a coverage rule
(the run is vacuous) or a falsifier fails on any seed; a falsification is never downgraded by the mutation evidence.
INVALID if a record is missing, does not describe the frozen run, a counter is missing or mistyped, the exit status
disagrees with the rules, or the mutation evidence is missing or incomplete (INVALID says nothing about the
hypothesis).

The harness exits 0 when every rule holds, 1 when one fails, 2 when it cannot run: the oracle fails its own check
against values computed by hand, the generator is not deterministic or vetoes the truth in a sound world, or a
negative control does not trip. **A panic or an error of the product on a valid input is a falsification, not a
harness fault**: every call into the products, including `LabelSchema::new`, `VoteMatrix::new` and every
`EvaluationSet::push`, is wrapped in `catch_unwind`, a panic is counted in `product_panics`, an `Err` in
`unexpected_refusals`, the run goes on and exits 1. A panic in the harness's own code ends the process with status 101 and the seed is INVALID. No rule changes
after a run.

Coverage (the run is vacuous if unmet), counted from events the run observed unless marked by construction:

- `worlds_run >= 3` (by construction: the run's own work)
- `determined_seen >= 1`, `estimated_seen >= 1`, `unknown_seen >= 1`, `disputed_seen >= 1` (outcomes the product
  returned, summed over the three worlds and read from `Resolution::outcomes`)
- `vetoed_items >= 200` (items of W1 whose row in the vote matrix the product was given holds a veto)
- `gold_items >= 500` (the size of the smallest `EvaluationSet` the product holds over the worlds; the set is built
  by the harness, so this is by construction in the same sense as `worlds_run`)
- `few_function_warnings >= 1` (the `FewerThanThreeFunctions` warning the product returned in the deliberate
  two-function cell, and nowhere else)
- `budget_cells_seen >= 3` (budgets of 2, 3 and 5 class votes per item, each on at least 50 items, counted in the
  vote matrix of W1 the product was given)
- `correlated_pair_items >= 50` (items on which both functions of the correlated pair voted, counted in the matrix)
- `contaminated_vetoes >= 1` (vetoes of the true class in W2, counted in the matrix)
- `active_set_probes >= 1` (actively sampled gold sets the product evaluated)
- `gold_guard_probes >= 4` (inputs the gold guards had to refuse and the product answered with a refusal; a panic
  does not count as a probe)
- `acquisition_items >= 1` (items the product ranked)

Falsifiers (each tripped one falsifies F002 on that seed):

- `veto_contradictions <= 0`, `model_oracle_mismatches <= 0`, `resolution_oracle_mismatches <= 0`,
  `metric_mismatches <= 0`;
- `active_set_calibration_reported <= 0`, `gold_guard_violations <= 0`, `acquisition_violations <= 0`;
- `sound_world_determined_errors <= 0` (in a world whose verifiers are sound, elimination cannot determine a
  class other than the truth), `unexpected_refusals <= 0`, `determinism_mismatches <= 0`, `product_panics <= 0`;
- `gain_violations <= 0` (restates the three gain rules as a count, so that the mutation checker has a counter);
- `w1_brier_gain_lower_ppm >= 1` and `w1_ece_gain_lower_ppm >= 1`: the 2.5th percentile of the paired bootstrap of
  the gain (majority vote minus label model) over the 600 gold items of W1 is positive;
- `w1_census_accuracy_ds_permille >= w1_census_accuracy_mv_permille`: the accuracy of the labels each arm produces
  for all 2000 items of W1 (decided on the pilot seeds, section 7; the accuracy on the 600 gold items is reported and
  gates nothing).

The three gain rules are named in the mutation plan through `gain_violations`, which restates them as a count.

## 2. Worlds, functions, arms and oracle

Four classes with priors 0.4, 0.3, 0.2, 0.1. Five probabilistic functions `lf_a` to `lf_e` with a pinned accuracy
(0.85, 0.75, 0.65, 0.60, 0.55 in W1 and W2; 0.70 for all in W3); a wrong vote is drawn from the row of a pinned
confusability table (`wrong_class_weights`), so confusions are not uniform. One deliberately correlated pair
(`lf_d`, `lf_e`): when both vote on an item, `lf_e` repeats `lf_d`'s vote with probability 0.6 (W1, W2). Each item
receives class votes from a seeded subset of 2, 3 or 5 functions (weights 0.25, 0.5, 0.25), the same votes for
every arm: this is the common annotation budget. Four verifier functions `v_a` to `v_d` each vote on an item with a
pinned coverage and then rule out one class: a uniformly drawn wrong class (W1), or, in W2 only, the true class with
probability 0.08. A verifier is not an annotation and costs no budget.

- **W1, sound verifiers** (coverage 0.5, 0.4, 0.3, 0.2): a gain is expected by construction and stated as such.
  The only world whose gain rules gate.
- **W2, contaminated verifiers** (coverage 0.8, 0.7, 0.6, 0.5; 8% of the votes of a firing verifier rule out the
  true class): used for the hard veto check, for the Disputed rate (four verifiers can rule out all four classes
  only here) and as a descriptive measure of what verifier precedence costs when a verifier is wrong.
- **W3, homogeneous** (accuracy 0.70 for all, no verifier, no correlated pair): a descriptive control.

Deviation from the plan, stated: the plan had two verifiers and a contamination of 2% of items. With four classes a
verifier votes one veto, so two verifiers can never determine a class (three vetoes) or dispute an item (four), and
the coverage rules `determined_seen >= 1` and `disputed_seen >= 1` could not hold. Four verifiers, higher W2
coverage and a per-vote contamination of 8% are the smallest change that makes both outcomes occur in 27 of 27 pilot
runs (at least 11 disputed and 47 determined items per run).

Arms, all scored on the same 600 gold items drawn uniformly without replacement (per world, `GoldSampling::Uniform`,
`GoldSource::Oracle`):

- `ds`: `fit_label_model` (1000 iterations, tolerance 1e-6, smoothing 1.0) and `resolve` (`min_probability` 0.7), the
  product. The distribution scored is `Resolution::posterior(i)`, which is renormalised over the classes the
  verifiers leave. **The trap:** `evaluate` takes plain distributions, so feeding it `LabelModel::posteriors()`
  silently ignores every veto. The veto check is therefore run on the vectors actually handed to the scorer, and a
  harness mutation and a negative control cover it.
- `mv`: majority vote over the class votes, posterior `(n_c + 1) / (n + C)`, verifier vetoes ignored. Ties are
  broken to the lowest class by an epsilon of `1e-9` per class rank in the pseudo-count (the scorer's argmax is the
  last maximal class, so without it ties would go to the highest class); the lower classes are the more frequent
  ones, so this is generous to the baseline.
- `mvp` (descriptive): the same vote restricted to the classes the verifiers leave. It separates what verifier
  precedence contributes from what the label model contributes.
- `mvc` and `mvcp` (descriptive, never gated): a calibrated vote share, `p_c` proportional to `prior_c *
  exp(beta * n_c)` over the class votes (`mvcp`: with the classes the verifiers rule out removed). It uses the
  world's true class priors and one sharpening exponent `beta = 1.70` (`mvc_beta_milli = 1700`) fitted on the pilot
  seeds only: the grid search (0 to 4, step 0.05) for the exponent that minimises the mean Brier score of the vote
  share over all items of W1 and W3 of the 27 pilot seeds, the ignored test `fit_the_calibrated_vote_exponent` in
  `f002/runner.rs` (pinned in `config.toml`, never refitted on a confirmatory seed). It is generous to the baseline
  (true priors, exponent tuned on this generator), which is the conservative direction. **Declared in advance:** the
  gains of the label model over `mvc` and `mvcp` are reported per world (`wN_brier_gain_mvc_ppm`,
  `wN_ece_gain_mvc_ppm`, `wN_brier_gain_mvcp_ppm`, `wN_ece_gain_mvcp_ppm`) and gate nothing; a label model that is
  not better calibrated than `mvc` or `mvcp` is a **recorded negative result about the label model**, not a
  failure of the experiment, and COMPLETED does not depend on it.
- verifier-only (descriptive): only items the verifiers determine by elimination are labelled
  (`wN_vo_labelled_gold`, `wN_vo_correct_gold`); it labels about 2 to 3% of W1.

A disputed item (every class ruled out) has no distribution; every arm's scorer is given the uniform distribution
over the classes for it, and the veto check exempts exactly that substitute.

Independent oracle (`oracle.rs`; it imports neither `ptr-labeling` nor `ptr-analytics` and reads the harness's own
generated data, never a `VoteMatrix`):

- a second Dawid-Skene EM written from the documentation of `fit_label_model` (same initialisation: smoothed
  majority vote of the modelled functions; same iteration order), compared with the product's priors, confusion
  matrices, posteriors and warnings within 1e-6 (`model_oracle_mismatches`);
- the expected resolution of every item, computed from the harness's vetoes and the product's own model posterior,
  so that a defect of the fit shows in the first counter and a defect of `resolve` in
  `resolution_oracle_mismatches`: kind, class, probability and distribution within 1e-9;
- independent Brier score, equal-mass ECE (ten bins, stable sort of the confidences), accuracy, Krippendorff
  alpha and Wilson interval, compared within 1e-9 with `evaluate`, with `brier_score` and
  `expected_calibration_error` called directly on the same vectors, with `krippendorff_alpha_nominal` over the
  functions' class votes, and with `function_accuracy` (`metric_mismatches`). The oracle checks itself against
  values computed by hand at the start of every run (exit 2).

## 3. Metrics

Every metric is a JSON integer at the top level: counters (section 1), and per world `w1_`, `w2_`, `w3_`:
accuracy (permille) of `ds`, `mv`, `mvp`, `mvc` and `mvcp` on the gold sample and on every item
(`census_accuracy_*`; the labels are produced for all items, so the census has no sampling noise; the census accuracy
of `ds` and `mv` in W1 is gated), Brier score and ECE (ppm) of the five arms, the gains of `mv` over `ds` as point
values and as the lower end of the paired bootstrap (2000 resamples of the gold positions, seed 20261009, 95%
interval, oracle metrics only), the gains over `mvc` and `mvcp` (point values), outcome counts and rates (Determined,
Estimated, Unknown, Disputed; the Unknown and Disputed rates of the manifest), Krippendorff alpha of the functions'
class votes (ppm, and whether it is defined), and the verifier-only counts. `wN_mv_veto_violating_items` counts items
on which the majority vote's label is a vetoed class (the baseline ignores vetoes by design; it shows the veto rule
bites). A disputed item is scored with the uniform distribution, whose argmax in `evaluate` is the last, rarest class;
this touches `ds` only in W2 (about 15 items) and nothing that gates.

## 4. Cases and negative controls

Per seed: W1, W2, W3 (2000 items, 600 gold each), a two-function cell (W1 votes of `lf_a` and `lf_b` only, with
the verifiers: the deliberate cell where the model must warn that Dawid-Skene is not identifiable), a repeated fit
and resolution of every pipeline compared bitwise (`determinism_mismatches`), acquisition by entropy and by margin
(budget 100) checked against the oracle's reading (at most the budget, distinct, no determined item, disputed items
first in item order, exactly `min(budget, non-determined)`), an actively sampled gold set built from the ranking
and evaluated (calibration must be absent), and the gold guards (a duplicate item with the same and with another
class, a human label in an oracle set, a gold item without a posterior, a posterior that is no distribution: each
must be refused, and a refused label must leave the set unchanged).

Seven negative controls doctor one stage after the product produced it and show that the counter built for it
counts the defect; a control that does not trip is exit 2. They are: a resolver that ignores vetoes (argmax of the
raw posterior) and a scorer that reads the raw posterior (both `veto_contradictions`), posteriors nudged by 1e-3
(`model_oracle_mismatches`), a Brier score off by 1e-3 (`metric_mismatches`), a ranking that proposes a determined
item (`acquisition_violations`), two identical arms put through the paired bootstrap and the gain gates (`gain_violations`; this exercises the
bootstrap and the gate arithmetic, not the arms) and a resolution
with a wrong class (`resolution_oracle_mismatches`). They show the counters can fail; the mutation plan is what
shows the product's defects are caught.

## 5. Mutation plan and honesty

`tests/mutations.toml` plants 29 defects, one at a time: 20 in the products (`resolve_item`'s veto set, its
Determined and Disputed branches and its renormalisation and threshold; the M-step normalisation, the prior
smoothing and the E-step; a clock-dependent initialisation; `evaluate`'s accuracy and Brier score and its report of
calibration on an active set; `EvaluationSet::push` accepting duplicates; `function_accuracy`; the two orderings of
`rank_for_annotation`; the sign of Krippendorff's denominator; the absolute value in ECE; `VoteMatrix::new` refusing
a verifier's veto, which must show as `unexpected_refusals`) and 9 in the harness (the scorer that reads the raw
posterior, a label-model arm that is the majority vote, three defects of the oracle that its hand-value self-check
does not see, and three in the gain path: the sign of the Brier gain, an ECE lower bound collapsed to zero and a
baseline whose census accuracy is that of the veto-aware vote; these name the three gain rules through
`gain_violations`). A mutation is killed when the harness exits 1 and a counter the plan names for it fired. Every
hard counter is expected by at least one mutation; COMPLETED needs `killed == total == 29`. Anchors are checked to
occur exactly once (`python3 scripts/mutation_check.py F002 --check`).

Honesty rules applied: product defects falsify through counters (exit 1), exit 2 is for harness integrity only;
panics are contained and counted; coverage counters are observed (the by-construction ones, `worlds_run` and `gold_items`, are named above; the
per-world ones are counted in the world the rule names); every
falsifier can fail (a rule-mutation test on the emitted JSON) and is named by a planted mutation; every negative
control exercises what it claims; the independent oracle shares no code with the products; `criteria.toml` is pinned
to the harness lists by a test and every path resolves in a reduced run; the constants are pinned to
`config.toml` by a drift test.

## 6. Manifest text proposed for the freeze (to confirm)

`experiment.toml` keeps the hypothesis, metrics and baseline of the original. Its `falsification` is rewritten
faithfully to the rules above, and states the narrowed claim (calibration against the naive vote-share posterior) (the original said "no calibration gain on uniform gold, or any label that contradicts
a verifier veto, rejects the model"; the new text adds the oracle, guard and accuracy rules, the lower-bound form of
the gain, and states that W2 and W3 gate nothing but the veto and oracle checks). `required_artifacts` gains
`mutations.json`, `entrypoint` is set at the freeze, `seeds` stay 17, 29, 43, 71, 101. Proposed
`[preregistration]` table keys: `seeds`, `harness = "weak-supervision"`, `items`, `gold_items`, `classes`,
`bootstrap_resamples`, `bootstrap_seed`, `min_probability_permille`, `preregistration_text`,
`preregistration_text_sha256`, `criteria`, `criteria_sha256` (and an `[experiment.F002.required]` entry in
`experiments/preregistration.toml`, both added by the head).

## 7. Pilot, and the form of the gates

The plan left open whether the ECE gate needs a paired bootstrap, to be decided on pilot seeds, never confirmatory
ones. Pilot seeds: 1001, 1002, 1003 and, for a power analysis, 2001 to 2024 (`pilot/`); the numbers below are from the
final harness (27 runs, all exit 0, no coverage unmet, no falsifier tripped).

Calibration gates. Rule: gate a metric on the lower end of the paired bootstrap interval of its gain when that lower
end is at least half the point gain on every pilot seed (the gate then has power, and it does not pass a true gain of
zero half the time, as a point gain would); otherwise on the point gain. The "half" is a design choice, not derived.
Evidence: the lower end is between 0.79 and 0.91 of the point gain for the Brier score (minimum 132,352 ppm) and
between 0.76 and 0.86 for ECE (minimum 160,620 ppm). Decision: **both the Brier and the ECE gates read the lower
bound** (`brier_gate_on_lower_bound = true`, `ece_gate_on_lower_bound = true`). Stated plainly, the gates are
non-binding in practice: the pilot minima are five orders of magnitude above the 1 ppm threshold, so on a healthy
product these falsifiers cannot realistically fail (the gain is large against the naive baseline by construction,
section 8); they would catch a pipeline that stopped being better calibrated than that baseline.

Accuracy gate. The plan gated accuracy on the 600-item gold sample. Pilot evidence: the paired gain of the label
model over the majority vote in W1 has mean 46.6 permille, standard deviation 17.5 and minimum 13 on the gold sample
(a normal approximation gives a falsification by sampling noise alone of about 0.3% per seed, about 1.4% over five
seeds), and mean 49.1, standard deviation 16.3, minimum 16 on the census of all 2000 items (about 0.1% per seed,
about 0.5% over five). The claim is about the labels the pipeline produces, which the census measures exactly, and the
gold subsample only adds sampling noise; the gate was therefore **switched to the census**
(`w1_census_accuracy_ds_permille >= w1_census_accuracy_mv_permille`), decided on pilot seeds before any confirmatory
seed. The gold accuracy is still reported (`wN_accuracy_*_permille`). With the final harness: 0 failures of either form
in 27 runs.

Chronology, as it happened (nothing here has an external timestamp; the work is uncommitted until the pre-freeze
commit, whose hash then fixes it). The world parameters (four verifiers, coverages, 8% contamination, accuracies,
the confusability table) were written into `params.rs` before the first run and have not changed; the only change
after a first look at output was fixing a wrong fixture of the oracle's own ECE self-check. The first pilot (seeds
1001 to 1003) ran with point-gain gates and a gold accuracy gate; the rule for the form of the calibration gates was
formed before that run but written to a file only after it, and the 24-seed power analysis came after. The switch of
the accuracy gate to the census and the calibrated vote shares (`mvc`, `mvcp`, exponent fitted on the same 27 pilot
seeds) were added after an external review of the first draft, again on pilot seeds only; the pilot was rerun in full
with the final harness. Only seed 17 was run among the confirmatory seeds (one dry run, and the mutation checker's
seed); seeds 29, 43, 71 and 101 were never run.

## 8. What the pilot shows (descriptive, for the owner)

These are numbers of non-confirmatory runs and nothing here is a result; they bear on how the claim is read.

- Against the **naive** majority vote, W3 (homogeneous accuracy, no verifier) shows gains as large as W1's (mean Brier
  gain 203,000 ppm and ECE gain 342,000 against 223,000 and 282,000). The naive posterior `(n_c + 1) / (n + C)` is
  badly under-confident, so the calibration gates measure "better calibrated than that posterior".
- Against the **calibrated vote shares** (exponent 1.70, true priors; mean gains of the label model, positive is a
  win for the label model): in W3 the label model and `mvc` tie (Brier -133 ppm, ECE +509, ahead in 12 and 14 of 27
  runs); in W1 the label model beats `mvc` on Brier (+59,000 ppm) and not on ECE (-5,500, behind in 17 of 27), but
  `mvc` ignores the verifiers; against `mvcp`, the calibrated vote restricted to the classes the verifiers leave, the
  label model is **worse** in W1 (Brier -27,000 ppm, ECE -19,600). That is a recorded negative result about the label
  model, declared in advance (section 2): what Dawid-Skene adds beyond a calibrated veto-aware vote is not shown here.
- In W1 the label model is on average 16 permille less accurate on the census than the veto-aware vote (range -55
  to +12): the accuracy gate is passed because the plain majority vote ignores the verifiers.
- In W2 the label model is about 74 permille less accurate than the plain majority vote, and its Brier gain over it is
  **negative on average** (-19,500 ppm; negative in 22 of 27 runs, though the ECE gain is +146,700): verifier
  precedence costs accuracy and Brier score when a verifier is wrong 8% of the time, as it must.

## 9. Limits

Gold is the world's truth, the functions are seeded stand-ins and the worlds are small, regenerated per seed (five
seeds are five worlds, not a population statement). The oracle is a second implementation of the same documented
algorithm, so it shows the product implements the documentation, not that the documentation is the only sound
choice. The gated baseline's posterior is the plan's naive one, with the weakness described in section 8; a calibrated
baseline needs no labelled data of its own when its parameters are pinned from pilot seeds, which is what `mvc` and
`mvcp` do (generously: true priors, an exponent tuned on this generator), and they are descriptive only. Correlated functions double-count
evidence and make the posteriors overconfident; the calibration rule measures that and does not correct it. The
acquisition checks are against the stated ordering contract (disputed first, never determined), not against an
optimal annotation policy. Timings come from one shared cloud container. No product defect was found while building
the harness.
