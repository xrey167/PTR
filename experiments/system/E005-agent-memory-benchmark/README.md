# E005-agent-memory-benchmark

## Hypothesis
Does the combined stack (projection, fast memory, certified branches, hybrid retrieval) answer long-horizon memory questions better than current agent-memory systems at an equal budget?

## Primary metrics
LongMemEval and LoCoMo accuracy by category (knowledge update, temporal reasoning, abstention); tokens; latency

## Baseline
full-context model; RAG baseline; published agent-memory systems re-run where their licences allow

## Falsification
No category improvement over the strongest re-run baseline at an equal token budget rejects the claim.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
