# RAG Reference Baseline

This baseline provides deterministic BM25 plus dense-vector reciprocal-rank fusion with explicit update/delete semantics for E002 and retrieval tests.

The dense vectors are caller-supplied so the retrieval logic is reproducible without forcing a heavyweight model dependency into base CI. A **reported strong RAG run must pin the embedding model, reranker, chunking policy and corpus version** in its run manifest.

This is therefore a runnable hybrid retrieval reference, but it is not evidence that PTR beats strong RAG until E002 is executed against pinned high-quality embedding/reranking baselines.
