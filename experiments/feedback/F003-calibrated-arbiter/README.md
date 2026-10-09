# F003-calibrated-arbiter

## Hypothesis
Does the calibrated arbiter keep the harm rate of auto-proposed branches at or below alpha on held-out adjudications? The claim is scoped to the mechanism and to a low-noise regime: the adjudicator is a **simulation** with a known error rate of 2 per mille ([PROTOCOL.md](PROTOCOL.md)), not a person, because at 20 per mille Learn-then-Test certifies nothing in roughly three calibrations of four (a utility limit of the start rule; an ungated descriptive arm reports it per seed). That the guarantee holds with the data-dependent start rule is shown empirically for this scenario, not proven. Reviews saved are descriptive only.

## Primary metrics
coverage of the Learn-then-Test guarantee over Monte-Carlo replicates; per-policy exact held-out test of the adjudicated harm rate of auto-proposals (the literal upper bound is reported); ineligible auto-proposals; escalation share and reviews saved; off-policy estimate error

## Baseline
escalate every branch; fixed uncalibrated threshold

## Falsification
F003 is falsified on a seed if any of the following holds; the statistics are those of the decided rule (2026-10-09), and the literal rule is reported, not gated:

1. Any ineligible branch is auto-proposed, or any triage differs from the independent oracle's decision.
2. Coverage of the guarantee: over the Monte-Carlo replicates, the number of certified thresholds whose true adjudicated harm rate exceeds alpha is above the critical value of the exact binomial test of "at most delta of the replicates" at level 1e-3.
3. Held-out: for any recorded policy, the exact one-sided binomial test of "the harm rate of the branches it would auto-propose is at most alpha" on the held-out adjudications rejects at the Bonferroni share (familywise 0.05 over the policies and seeds). The literal rule, "an upper confidence bound above alpha on held-out adjudications is a hard failure", is printed for each policy and judged by nothing: a correct Learn-then-Test policy fails it by construction.
4. A threshold, held-out set, conformal threshold or Clopper-Pearson bound that differs from the independent oracle's; a non-slice triage that can be adjudicated; an off-policy estimate whose mean error is outside four standard errors of zero; a target below the logging threshold that is not refused; a panic of the arbiter.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

Held-out adjudications are those outside the recorded calibration set of the policy under test (`PolicyRecord::held_out`), and the rule that picks each calibration set, for example every adjudication observed before the policy was recorded, is fixed before the run. Also fixed before the first adjudication: whether adjudicators see a branch's stated intent, and the key namespaces by which the adjudicated harm rate is reported as a descriptive breakdown next to the global rate.

### Preregistration
The `[preregistration]` table of `config.toml` is frozen before F003 leaves `planned`: `scripts/check_research_gates.py` fails CI until every key `experiments/preregistration.toml` requires of it is pinned and `experiment.toml` names its digest (`preregistration_sha256`).

- **Decided.** Risk level α = 0.05 and confidence 1 − δ with δ = 0.05 (`alpha_permille = delta_permille = 50`), under Learn-then-Test (`threshold_rule = "learn_then_test"`) on the fixed grid of `THRESHOLD_GRID_STEPS` = 1000 steps. At least 59 adjudications per policy (`min_adjudications_per_policy`): the fewest with which a calibration set without harm certifies any threshold at these levels, since 0.95ⁿ ≤ 0.05 first holds at n = 59.
- **Pinned.** The threshold of the fixed uncalibrated baseline, `uncalibrated_threshold_permille = 500`: the conventional "propose when the score is at least one half", the value S003's review policy uses; the scorers of the scenario are overconfident, so the baseline is expected to violate alpha. The stream, slice, policy, Monte-Carlo and off-policy sizes (`stream_branches`, `slice_rate_permille`, `adjudicator_error_permille`, `policy_sizes`, `mc_replicates`, `coverage_test_level_ppm`, `heldout_familywise_permille`, `heldout_tests`, `ope_logs`, `ope_log_branches`, `ope_bias_sigmas`) with their derivations in [DRAFT_PREREGISTRATION.md](DRAFT_PREREGISTRATION.md).
- **Owner decisions, signed 2026-10-09 as proposed.**
  - `calibration_set_rule`: every adjudication recorded before the policy version is calibration, and every one after it is held out.
  - `adjudicators_see_intent = false`. The intent is free text the agent wrote, and the harm label calibrates that agent's auto-proposal threshold, so a change is judged from its delta, the state before and after, and the verifier codes alone.
  - `harm_breakdown_namespaces`: the first `:`-separated segment of each changed key, reported only descriptively, next to the global rate.
  - `adjudication_protocol`: `experiments/feedback/F003-calibrated-arbiter/PROTOCOL.md`, which describes the simulated adjudicator and states plainly that it is a simulation. Its content is frozen with the table: `adjudication_protocol_sha256` must be its SHA-256, pinned at the freeze, so the protocol cannot change after an outcome is seen.

## Status and files (2026-10-09)
F003 is still `planned`: the harness, the criteria, the aggregator and the mutation plan are built, but nothing is frozen and no run is evidence. The preregistration text is [PREREGISTRATION.md](PREREGISTRATION.md) (frozen with the `[preregistration]` table at the freeze); the decision rules are [criteria.toml](criteria.toml), applied by [aggregate.py](aggregate.py); the defects the harness must detect are [tests/mutations.toml](tests/mutations.toml). The harness is `bins/ptr-bench/src/experiments/f003/` (`ptr-bench calibrated-arbiter <seed>`, one JSON object per seed, exit 0 / 1 / 2): a simulated stream of 30,000 branches per seed through the real `ptr-branch` pipeline (`calibration_draw`, `triage`, `adjudicate`, `PolicyRecord::calibrate` and `held_out`), 2,000 Monte-Carlo replicates of the certification, 200 off-policy logs, independent oracles for every comparison, and four negative controls. A seed on which no recorded policy certifies anything makes coverage unmet, which counts as FALSIFIED by the S001/R001 convention although it says nothing about the claim (the pilot batch certified in all 20 policies; per-policy chance of certifying nothing is about 18 percent, and the policies are nested). Stated limits: the adjudicator is a simulation, the world is synthetic, and the claim is the mechanism's.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
