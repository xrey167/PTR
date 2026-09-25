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

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
