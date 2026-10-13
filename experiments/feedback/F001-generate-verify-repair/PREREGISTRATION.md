# F001 generate-verify-repair: preregistration

Pre-freeze text of the preregistration of F001; at the freeze it becomes the `preregistration_text` file of `config.toml`'s
`[preregistration]` table (renamed `PREREGISTRATION.md` by the head, as S001's and Q002's were) and its SHA-256 is frozen with
the table. F001 is `planned`: nothing here is a result and no run is evidence. The decisions are made by `criteria.toml` and
applied by `aggregate.py`; this file gives their reasons, the definitions, the modelling choices, what the harness cannot
produce and the limits of the claim. Every criterion is the text of `experiment.toml` (hypothesis, falsification, metrics) and
the pins of `config.toml`'s `[pinned_thresholds]`; the harness is `bins/ptr-bench/src/experiments/f001/` (subcommand
`generate-verify-repair <seed>`).

Entrypoint (`experiment.toml`'s `entrypoint`, the runner's command, set at the freeze):
`cargo run --release --locked -p ptr-bench -- generate-verify-repair <seed>`

Scope (owner decision 2026-10-09): MECHANISM ONLY. The code under test is `ptr_feedback::RepairLoop` and
`Population::best_verified` (crates/ptr-feedback), driven through their public API over the real `ptr_verifier::VerifierFabric`
and `ActionContractVerifier`. The generator and the repairer are seeded stubs with a stated per-slot skill; the task is
synthetic. The claim is that the loop uses the verifier's verdicts and critiques correctly and never lets a rejected candidate
pass:

- it accepts a candidate only if the fabric passed it at `Deterministic` level with no hard finding (`Candidate::passes`);
- it verifies each candidate exactly once, and spends at most its verifier-call and round budget;
- it repairs only what the report's hard findings name, and rewrites whatever it cannot repair (`Unknown`, `Disputed`, a `Pass`
  that carries a hard finding, a `Fail` without a hard finding, a declined repair);
- it returns `Exhausted`, never a best-effort failure, when the budget is spent;
- its trajectory agrees with the verifier's log, and `Population::best_verified` agrees with its outcome.

NO CLAIM IS MADE ABOUT THE QUALITY OF ANY MODEL'S GENERATIONS, nor that repair is better than rewriting for any real generator.
The comparison of single-shot, rewrite-only and repair is made under the stated stub skill, with analytically known
expectations; its outcome is a consequence of that skill (section 8), and testing it shows the loop does what it is built to do.

## 1. Decision rule

Five seeds (17, 29, 43, 71, 101), one JSON object each, `criteria.toml` applied mechanically (a "< N" prerequisite is written
`>= N`, a "> N" violation `<= N`). COMPLETED iff every seed meets every coverage rule and no falsifier trips and
`results/mutations.json` shows every mutation of the plan killed; FALSIFIED if a coverage rule (the run is vacuous) or a
falsifier fails on any seed; INVALID if a record is missing, does not describe the frozen run, or a counter is missing or
mistyped, or the exit status disagrees with the rules (INVALID says nothing about the hypothesis). The harness exits 0 when every
rule holds, 1 when one fails, 2 when it cannot run (section 7). A defect of the code under test is never exit 2: it is a falsifier
(a panic of the loop is the counter `product_panics`). No rule changes after a run.

Coverage (the run is vacuous if unmet), twelve rules: `status_pass`, `status_fail`, `status_unknown`, `status_disputed` each >= 1
(the verifier statuses seen in the boundary's log); `profiles.uniform.tasks`, `profiles.heterogeneous.tasks`,
`profiles.hard.tasks` each >= 1000; `initial_fail_tasks` >= 500 (tasks whose first candidate was not accepted);
`repaired_successes` >= 100 (tasks accepted after a repair); `exhausted_runs` >= 50 (runs that returned Exhausted);
`lying_refused` >= 10 (reports of the lying stub, or forged Passes that keep their hard findings, that were refused);
`shallow_refused` >= 1 (Passes below `Deterministic` that were refused).

Falsifiers, twenty-six rules. The eighteen hard counters, each `<= 0` (section 5): `false_accepts`, `unverified_returns`,
`double_verifications`, `budget_violations`, `collateral_edits`, `exhausted_as_success`, `verdict_mismatches`,
`verifier_call_mismatches`, `expectation_violations`, `product_panics`, `calls_after_acceptance`, `premature_exhaustions`,
`missed_acceptances`, `trajectory_mismatches`, `selection_mismatches`, `repair_gain_shortfalls`, `draws_gain_shortfalls`,
`altered_candidates`. And eight strict comparative rules, which are CONSEQUENCES OF THE STUB SKILL (section 8):
`repair_gain.b<B> >= 1` (pooled successes of the repair arm minus those of the rewrite-only arm) and `draws_gain.b<B> >= 1` (the rewrite-only
arm's milli-draws per success minus the repair arm's) for the budgets B in {2, 4, 8, 16}, pooled over the three profiles. Gains of
thousands are guaranteed analytically (section 4), so the floor of one only separates "repair works" from "repair does nothing" (the two arms are then
the same run and every gain is 0). The non-strict comparisons of an earlier draft (repair >= rewrite, rewrite >= single, equal at budget 1) are no
falsifiers: they cannot see the most plausible defect (repair silently disabled gives equality), and the arms' successes stay in the JSON as descriptive
values. `repair_gain_shortfalls` and `draws_gain_shortfalls` count the gain rules that fail, so that a mutation can name them.

Rules that hold by construction or duplicate another (named so a reader does not mistake them for evidence): the three
`profiles.<name>.tasks >= 1000` (the plan runs 2000 tasks per profile; the counter counts the plan), `status_pass >= 1` and
`status_fail >= 1`. `repaired_successes`, `exhausted_runs`, `initial_fail_tasks`, `lying_refused`, `shallow_refused`, `status_unknown` and
`status_disputed` are observed from the boundary's log, the stubs' log and the loop's outcomes.

## 2. The task and the stubs

**Task.** Produce an `ActionIr` that meets an `ActionContract` (target `note`, operation `create`, capability `note.create`, input
type `note.v1`, effect `Mutation`, payload of at most 8 bytes and valid UTF-8). Seven independent defect slots: target,
operation, capability, input-type, effect, payload-size, payload-utf8. A wrong value of a string or effect slot is drawn from a
small list of wrong values; a payload is 1 to 8 ASCII bytes when right, 9 to 16 when the size slot is wrong, and contains an
invalid UTF-8 byte (0xff) when the encoding slot is wrong. The generation and the revision of the action are not part of the
contract; the harness uses them as the candidate's identity (task and serial).

**Generator.** A fresh draw makes slot `s` right with probability `p_s` (per mille), independently. Three profiles, chosen by
design and not tuned on results:

| profile | p per slot (per mille) | P(all seven right) | P(first call accepts) |
|---|---|---|---|
| uniform | 900 each | 478 | 416 |
| heterogeneous | 950, 900, 900, 800, 700, 500, 500 | 108 | 94 |
| hard | 650, 650, 600, 600, 550, 550, 500 | 23 | 20 |

**Repairer.** Redraws only the slots the critique names (a code of a hard finding that is one of the seven slot codes); each
redrawn slot is right with probability `q` = 800 per mille. Codes that name no slot (the lying stub's) are ignored; when none is
left the repairer declines and the loop draws a fresh candidate. An edit of a slot the critique did not name is a defect of the
loop (`collateral_edits`), not of the stub.

**Verifier fabric.** The real `ActionContractVerifier`, followed by four scripted stubs (`stub-unknown`, `stub-disputed`,
`stub-shallow`, `stub-lying`). On each verifier call a noise mode is drawn from the seed (independently of the candidate): clean
(850 per mille), Unknown (40: one stub answers `Unknown` at `LatentAgreement`), Disputed (30: one stub answers `Disputed` at
`Deterministic`), shallow (30: one stub answers `Pass` at `LatentAgreement`), lying (30: one stub answers `Pass` with a hard
finding), forged (20: the boundary rewrites a `Fail` of the fabric into a `Pass` at `Deterministic` that keeps its hard
findings, a verdict that contradicts itself). The fabric's own rules decide the rest: a hard finding forces `Fail`, the level is
the weakest level, the status is the worst status. A defect-free candidate is accepted on a clean or forged call (870 per
mille); a candidate with a defect is never accepted.

**Streams and common draws.** Every random choice comes from a SplitMix stream named by (seed, profile, task, kind, index), kind
being generation, repair, mode or self-test. The arms of a task share their draws wherever they make the same request (the first
fresh draw, the verifier modes by call index), so the comparison of arms is tight and at budget 1 the three arms are the same run.
The arms are therefore not independent samples.

## 3. Arms and budgets

Three arms on the same tasks: **single-shot** (one fresh draw, one verifier call: the loop with a budget of 1), **rewrite-only**
(after a rejected candidate always a fresh draw, the critique ignored: the loop with a repairer that always declines) and
**repair** (after a rejected candidate with a repairable critique the stub repairer, else a fresh draw). Rewrite-only and repair
are run at verifier-call budgets B in {1, 2, 4, 8, 16} (`max_verifier_calls = max_rounds = B`; the round bound is never the
binding one in this harness, the unit tests of the crate cover it). 2000 tasks per profile and seed, 6000 tasks per seed, 66000
loop runs per seed.

## 4. Oracle and measurement

Independent of the product: (a) a **harness-owned contract check** (`oracle.rs`, `std` only: no `ptr-verifier` or `ptr-feedback`
code) gives the defects of every action; the stubs' labelled defects must equal it, else exit 2. (b) A **call-counting boundary**
around the fabric draws the call's mode, forges the forged mode's `Pass`, and logs every call (candidate serial, defects, mode,
report): the log is the harness' count of verifier calls, and the required report (status, level, hard-finding codes in order)
for every call follows from the defects and the mode alone (`oracle::expected`). (c) The harness' own **acceptance rule**: a
`Pass` with no hard finding at `Deterministic` level. (d) An **exact analytic expectation** (`dp.rs`): a dynamic program over the
2^7 defect states gives, for each profile, arm and budget, the probability of success and the exact mean and variance of the
verifier calls and of the slot draws. It reads only the stub's parameters. A task counts as a success only when the loop returned
a candidate the oracle finds defect-free and whose report the harness accepts.

Measured per (profile, arm, budget) cell and compared with the expectation: `expectation_violations` counts the cells in which the
failure count (exact two-sided binomial tail) or the verifier calls or the slot draws (Bernstein bound for sums of bounded
per-task costs with exactly known mean and variance) has a tail probability below 573 per billion, the two-sided tail of five
standard deviations (chosen over four because a seed makes several hundred comparisons). A normal approximation is not used: it
would call a single failure in a nearly saturated cell a ten-sigma event (`stats.rs` tests this). The significance is a family-wise
choice: a seed makes about 99 comparisons (3 profiles x 11 arm/budget cells x 3 quantities), so 573 per billion each gives a
false-alarm rate of about 5.7e-5 per seed and 2.8e-4 over the five seeds for a correct product; two-sided four sigma would false-alarm
about 0.6% per seed.

**Expected gains (exact, from the dynamic program; seed-independent).** Pooled over the three profiles and 6000 tasks: repair minus
rewrite-only successes 1930 at B = 2, 2943 at B = 4, 2613 at B = 8, 1862 at B = 16; rewrite-only milli-draws per success 43643, 51552,
65163, 83466 against 14729, 11315, 10831, 10807 for repair, i.e. gains of 28914, 40238, 54332 and 72659. The measured values must exceed
the floor of one by three orders of magnitude, so the floor is not a tuned threshold.

## 5. Metrics (all JSON integers, or objects of integers)

`tasks`, `runs`; `repair_gain` and `draws_gain` (`{b2, b4, b8, b16}`, section 1); `success`, `verifier_calls`, `generator_draws` (slot draws) as `{single, rewrite: {b1..b16}, repair: {b1..b16}}`
pooled over the profiles; `draws_per_success` as `{rewrite: {...}, repair: {...}}` in milli-draws (`1000 * draws / successes`,
a very large number when an arm never succeeded); the coverage counters; and the eighteen hard counters:

- `false_accepts`: a returned candidate that the oracle finds defective, or whose report is not acceptable by the harness' rule.
- `unverified_returns`: a returned candidate no logged call verified, or whose report differs from the logged one.
- `double_verifications`: one candidate (one serial) verified more than once.
- `budget_violations`: a run with more verifier calls or rounds than its budget.
- `collateral_edits`: a repair made a slot defective that was right before it, or changed a payload whose two slots were right.
- `exhausted_as_success`: a candidate returned although no logged call was acceptable.
- `verdict_mismatches`: a logged report differing from the required status, level and hard-finding codes (this covers the
  real `ActionContractVerifier` and the fabric).
- `verifier_call_mismatches`: the loop's claimed verifier calls, generations, repairs, declined repairs and rounds against the
  boundary's and the stubs' counts.
- `expectation_violations`: section 4.
- `product_panics`: a panic of the loop (caught).
- `calls_after_acceptance`: verifier calls after a report the harness accepts.
- `premature_exhaustions`: Exhausted although the budget was not spent.
- `missed_acceptances`: Exhausted although a logged call was acceptable.
- `trajectory_mismatches`: the trajectory's steps, reports, serials, critiques (the hard-finding codes, repairability), decisions
  and origins against the log.
- `selection_mismatches`: `Population::best_verified` over the run's candidates disagreeing with the loop's outcome (some
  exactly when the loop returned, the same candidate).
- `repair_gain_shortfalls`, `draws_gain_shortfalls`: the number of budgets in {2, 4, 8, 16} at which `repair_gain` or `draws_gain` is below 1.
- `altered_candidates`: a candidate that reached the verifier different from the action the generator or repairer stub handed out (the
  harness records each action at the stub's output), or that no stub produced.

`hard_failures` is their sum. Elapsed time is recorded and never judged.

## 6. Negative controls and mutation plan

Two controls run 300 tasks per profile through the same accounting and the same falsifier rules, on the harness' own reference
loop (not the product): **score-accept** accepts on `score >= 0.5` and must trip `false_accepts <= 0`; **blind-repair** accepts
by the harness' rule but repairs with a repairer that ignores the critique and redraws every slot, and must trip
`collateral_edits <= 0`. If either does not trip, the harness cannot see the defect it plants and the run is exit 2. A unit test
runs both for all five seeds. They are harness-sensitivity controls: they show that the counters can fail, not that a defective
product is caught; that evidence is the mutation plan.

`tests/mutations.toml` (run with `python3 scripts/mutation_check.py F001`; `--check` verifies the anchors) plants 24 defects, one
at a time: in `ptr-feedback` (accept on score; hard-finding check, level check or status check removed from `Candidate::passes`;
the call budget one late or one early; a repaired candidate given a made-up Pass instead of being verified; exhaustion returning
the last failure as Verified; a critique naming only its first issue; a stale critique reused for every repair; a wrong claimed
call count; the repairer never called; an accepted candidate confirmed by a second call; the rewrite fallback panicking; acceptance demanding a strictly
deeper level; the accepted step missing from the trajectory; `best_verified` ignoring the acceptance rule; nothing repairable),
in `ptr-verifier` (the fabric ignoring hard findings; the fabric's level the strongest instead of the weakest; the contract
verifier dropping the payload-utf8 check) and in the harness (the analytic model ignoring the verifier noise; the boundary
forging nothing; the generator stub handing out an action other than the one it recorded, standing for a loop that alters a candidate:
the product is generic over `T` and cannot alter an `ActionIr`). Each names the hard counter that must fire; every one of the eighteen counters
is expected by at least one mutation, and the eight gain rules are named through the two shortfall counters by the mutations that stop the
repair arm from repairing (`nothing-is-repairable`, `repairer-is-never-called`). The checker runs `generate-verify-repair 400 17` (400 tasks per profile; the result says `"scale":"custom"`).

## 7. Exit 2 (the harness cannot run), and nothing else

A stub's labelled defects that differ from the oracle's reading of the action, checked where the stub produces the action (before the loop can
touch it; a candidate that reaches the verifier altered is the counter `altered_candidates`, not exit 2); the empirical frequency of a stub (the
generator's per-slot probability for each profile, the repairer's `q`, the verifier noise shares, 40000 draws each) outside five
standard deviations of its stated probability; a negative control that does not trip; an argument that does not parse. Anything
the code under test does wrong (a wrong acceptance, an unverified or doubly verified candidate, a budget overrun, a wrong
critique, a wrong count, a panic) is a counter and a falsifier.

## 8. Weak metrics and what is not measured

- **The comparative rules are consequences of the stub skill.** With a stub that fixes a named slot with probability 800 per
  mille and a fresh draw that is rarely defect-free, repairing the named slots must beat redrawing everything; repair must also spend
  fewer slot draws per success because a repair draws only the defective slots. They are
  tested so that a loop that mishandles verdicts (ignores the critique, accepts wrongly, wastes calls) fails them; they do not
  say that any model repairs better than it rewrites. The five-seed measurements are reported.
- **The expectation check is the main quantitative test:** the measured success, calls and draws of every cell against an exact
  expectation, with a stated significance. A defect that shifts a count by less than the test's resolution is caught only by the
  other counters. The Bernstein tests are conservative: at 573 per billion their effective resolution is above five standard
  deviations (about 5.5, and for the draws of the repair arm at the largest budget a shift of the order of a thousand draws), so a small
  per-cell shift (a few percent) is undetectable by `expectation_violations`; the claim is mechanism-structural and rests on the structural counters.
- **Draws are slot draws, not tokens.** A fresh candidate costs seven, a repair one per named slot; a verifier call costs one
  whatever it checks.
- The profiles are chosen by design: uniform (a good generator), heterogeneous (some hard slots) and hard (a poor generator).
  They were not tuned on a run.
- The negative controls do not test the product (section 6).
- `selection_mismatches` cannot see the score ordering among several passing candidates: the loop stops at the first accepted candidate, so a
  trajectory holds at most one. The ordering of `best_verified` is covered by the crate's `tests/selection.rs` only.
- `required_level` is the caller's policy: a loop built with `Unverified` accepts a clean shallow `Pass` by that policy. This harness requires
  `Deterministic`.

## 9. Stated limits (what this experiment does NOT show)

- Nothing about a real generator: no language or action model is run; real defects are correlated, partial and often not named
  by any verifier, and a real repair is not a redraw of a named slot with a fixed probability.
- Nothing about learned or statistical verifiers: the stubs answer from a script; a verifier that reports a clean `Pass` for a
  defective candidate (a lie with no hard finding) is not detectable by the loop and is not exercised. Only a self-contradicting
  `Pass` (status `Pass` with a hard finding) is refused.
- The task is synthetic with seven independent slots; the cost model is slot draws and verifier calls only.
- One machine profile, one process, no concurrency; the loop is single-threaded and synchronous.
- The round budget (`max_rounds`) is never binding in this harness; the crate's unit tests cover it.
- The trajectory is kept in memory; persisting it for training is not part of this experiment.

## 10. Pre-freeze history

**Pre-freeze runs and one change made after them.** The development of the harness ran the declared seeds before this text was frozen,
contrary to the rule that only seed 17 is a dry-run seed. (a) Seed 17 was run many times at the pinned and at a reduced scale. (b) A first
version of the expectation test used a normal-approximation 5-sigma z-test. A unit test on seed 17 at 120 tasks per profile (a debug
build) reported one expectation violation on a correct product, in a cell whose expected number of failures was about 0.1: one failure
there is a six-sigma event under a normal approximation and an ordinary event under the exact law. The test was replaced by the exact
two-sided binomial tail (failure counts) and the Bernstein bound (sums of bounded per-task costs of exactly known mean and variance), with
the same significance (573 per billion). This change was made because of seed 17 alone and its reasoning (saturated cells, bounded summands)
does not depend on any seed's outcome; it is tested by `stats.rs`. It made the test stricter than the plan's "binomial 4 sigma" in tail
and not lenient. (c) Seeds 29, 43, 71 and 101 were afterwards run once or more at the pinned scale, and 46 further non-declared seeds
(200 to 240) once, all with exit 0, no unmet coverage rule and no tripped falsifier. The stub self-tests and the negative controls had also been
run on all five declared seeds by unit tests. No threshold, profile, noise share, repair probability, coverage minimum or rule was changed in
response to any of these runs; after them the harness source changed only by formatting and lint fixes that do not alter behaviour, and
then, in response to the review of the pre-freeze draft and not to any run, by what sections 1 and 5 describe (the strict gain rules, the
`altered_candidates` counter and the move of the labelled-defect check to the stub's output). The profiles, the repair probability q = 800 per
mille, the noise shares, the coverage minima and the significance are design choices stated in sections 2 and 4: the plan fixed only the
uniform profile at 0.9, a heterogeneous one with some slots at 0.5 and a hard one; q and the noise shares were written into `params.rs` before
the first run and have not changed since; the coverage minima are one to two orders of magnitude below the analytic expectations
(exhausted_runs >= 50 against about 30000, lying_refused >= 10 against about 8000, repaired_successes >= 100 against about 4800,
initial_fail_tasks >= 500 against about 4900), so they cannot have been fitted. No log of the runs of seeds 29, 43, 71, 101 and 200 to 240 was
kept. The unit tests now iterate over pilot seeds (1, 2, 3, 5, 8) disjoint from the declared ones, and the only run of a declared seed made after
that review is the seed 17 dry run. These runs are not evidence; the evidence is the five-seed run from the frozen commit.
