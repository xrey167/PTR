# L004-projection-equivalence

## Hypothesis
Is a PostgreSQL projection rebuilt by replay always equal to the reference projection, and does it refuse every foreign or rolled-back history?

## Primary metrics
state and lifecycle divergence count; foreign histories accepted; rebuild time per 100k commits

## Baseline
ptr-state MaterializedState reference and the Turso backend

## Falsification
Any divergence from the reference or any accepted foreign or rolled-back history is a hard failure.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
