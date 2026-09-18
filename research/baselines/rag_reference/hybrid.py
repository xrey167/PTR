from __future__ import annotations

import math
from collections.abc import Iterable

from runner import BM25

def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

class HybridRAG:
    def __init__(self, rrf_k: float = 60.0):
        self.lexical = BM25()
        self.vectors: dict[str, list[float]] = {}
        self.rrf_k = rrf_k

    def upsert(self, doc_id: str, text: str, embedding: Iterable[float]) -> None:
        key = str(doc_id)
        self.lexical.upsert(key, text)
        self.vectors[key] = [float(x) for x in embedding]

    def delete(self, doc_id: str) -> None:
        key = str(doc_id)
        self.lexical.delete(key)
        self.vectors.pop(key, None)

    def search(self, query: str, query_embedding: Iterable[float], limit: int = 10):
        q = [float(x) for x in query_embedding]
        lexical = self.lexical.search(query, limit=max(limit * 4, 10))
        dense = sorted(
            ((doc_id, cosine(q, vector)) for doc_id, vector in self.vectors.items()),
            key=lambda item: (-item[1], item[0]),
        )[: max(limit * 4, 10)]

        scores: dict[str, float] = {}
        for ranking in (lexical, dense):
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
