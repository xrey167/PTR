# F003-calibrated-arbiter

## Hypothesis
Does the calibrated arbiter keep the harm rate of auto-proposed branches at or below alpha on held-out adjudications while reducing human reviews?

## Primary metrics
adjudicated harm rate of auto-proposals with its upper bound; escalation share; reviews saved; off-policy estimate error

## Baseline
escalate every branch; fixed uncalibrated threshold

## Falsification
An upper confidence bound above alpha on held-out adjudications, or any auto-proposal of an ineligible branch, is a hard failure.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

Held-out adjudications are those outside the recorded calibration set of the policy under test (`PolicyRecord::held_out`), and the rule that picks each calibration set, for example every adjudication observed before the policy was recorded, is fixed before the run. Also fixed before the first adjudication: whether adjudicators see a branch's stated intent, and the key namespaces by which the adjudicated harm rate is reported as a descriptive breakdown next to the global rate.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
