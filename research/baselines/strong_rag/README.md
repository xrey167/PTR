# Strong RAG Baseline

This is the target comparison baseline for E002 and long-horizon memory claims.

The intended stack is:
1. BM25 lexical retrieval;
2. BGE-M3 dense retrieval;
3. reciprocal-rank fusion;
4. BGE reranking;
5. a pinned generator with a matched context/compute budget.

The baseline is deliberately marked **blocked** until exact model revisions and the generator revision are pinned. PTR must not claim superiority over strong RAG using only the lightweight deterministic `rag_reference` implementation.

Required reported metrics include answer/task quality, retrieval recall, update latency, delete correctness, stale-hit rejection, context tokens, wall-clock latency and hardware profile.
