# Q003-postgres-hybrid

## Hypothesis
Does generation-joined hybrid retrieval in PostgreSQL match the recall of the reference retrieval stack while never returning a non-live generation?

## Primary metrics
recall@10; nDCG@10; stale-generation hits; p50 and p99 latency; index build time

## Baseline
research/baselines/rag_reference and strong_rag; BM25 reference

## Falsification
Any stale-generation hit is a hard failure; recall below the reference by more than the preregistered margin rejects it as the default.

## Design
Mechanism and threat model: [35 — Agentic substrate](../../../docs/architecture/35-agentic-substrate.md).

The `[preregistration]` table of `config.toml` is frozen before Q003 leaves `planned` (`experiments/preregistration.toml`, `scripts/check_research_gates.py`): the recall margin, k and the embedding model and revision, with the `rag_reference` and `strong_rag` baselines pinned. The margin, the model and its revision are still placeholders, so the gate holds Q003 in `planned`.

## Rule
Record matched baselines, hardware, seeds and negative results. Do not change success criteria after observing results.
