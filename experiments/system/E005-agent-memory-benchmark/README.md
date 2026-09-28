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

The `[preregistration]` table of `config.toml` is frozen before E005 leaves `planned` (`experiments/preregistration.toml`, `scripts/check_research_gates.py`): the token budget, the full-context baseline and the agent-memory systems re-run, with the `strong_rag` baseline pinned. All are still placeholders and `strong_rag` is blocked, so the gate holds E005 in `planned`.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
