# M008-fast-weight-memory

## Hypothesis
Does a revocable fast-weight working memory improve recall of recent and updated facts within a session over a recency buffer and plain retrieval at an equal token budget?

## Primary metrics
update-aware recall@1; Unknown rate; false recall rate; tokens; latency

## Baseline
recency buffer of the last N facts; hybrid retrieval over the same facts (Q003 stack)

## Falsification
No gain in update-aware recall at equal tokens, or a false recall rate above the recency baseline, rejects the memory as a default.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
