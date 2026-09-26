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

Each arm is its own training chain with its own training clock, even on a shared base model. Until the work schema records adapter runs, each run's `run.json` names its chain, the run manifest's input fingerprint and the data fingerprint of each adapter it trains (two different digests), and the number of samples drawn from the replay pool and of new examples, since the arms are compared at equal compute.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
