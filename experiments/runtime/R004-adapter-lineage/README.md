# R004-adapter-lineage

## Hypothesis
Does gated adapter lineage with forgetting-driven replay and consolidation retain earlier domains better than a naive adapter chain at equal compute?

## Primary metrics
backward transfer; per-domain forgetting; public-suite regression; serving latency; subspace overlap against chance

## Baseline
naive sequential LoRA chain; single joint retrain; uniform replay

## Falsification
No backward-transfer improvement over the naive chain, or public regression beyond the gate, rejects the lineage policy.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
