# F002-weak-supervision

## Hypothesis
Does a verifier-precedence Dawid-Skene label model produce better calibrated and more accurate labels than majority vote at the same annotation budget?

## Primary metrics
accuracy on uniform gold; Brier score; ECE; Unknown and Disputed rates; Krippendorff alpha

## Baseline
majority vote; verifier-only labels

## Falsification
No calibration gain on uniform gold, or any label that contradicts a verifier veto, rejects the model.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
