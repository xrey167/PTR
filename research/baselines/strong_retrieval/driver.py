"""strong_retrieval: BM25 + bge-m3 dense + RRF(k=60) + bge-reranker-v2-m3 over the top 50.

A stdin/stdout JSON driver, protocol 1, the same operations and replies as
research/baselines/rag_reference/q003_driver.py (see its docstring), so a Rust harness
speaks one protocol to both baselines. Differences:

  * `search` answers from BM25 top-100 and dense top-100, fused by RRF, the fused top-50
    reranked by a cross-encoder; hits carry the reranker score. The response also holds
    `stages` (candidate counts), `reranked` and `rerank_seconds`.
  * Embeddings are caller-supplied (the pinned bge-m3 vectors); the driver never embeds.
  * The reranker needs torch and transformers (the research/embeddings environment) and
    the pinned model files:  --reranker bge --models-root <dir containing
    bge-reranker-v2-m3/>.  With `--reranker none` the fused list is returned unreranked
    and every reply says `"reranked": false`; that mode is for protocol tests and for
    measuring the first stages, not a baseline result.

Retrieval only: no answer generation exists here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAG_REFERENCE = HERE.parent / "rag_reference"
EMBEDDINGS = HERE.parent.parent / "embeddings"
sys.path.insert(0, str(RAG_REFERENCE))

from runner import BM25  # noqa: E402  (the digest-pinned lexical implementation)

PROTOCOL = 1
MAX_LIMIT = 1000


def _check_id(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("id must be a non-empty string")
    return value


def _check_embedding(value, dimension) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("embedding must be a non-empty list")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError("embedding must hold finite numbers")
        out.append(float(item))
    if dimension is not None and len(out) != dimension:
        raise ValueError(f"embedding has {len(out)} values, the index uses {dimension}")
    return out


def _normalised(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class CrossEncoder:
    """bge-reranker-v2-m3 on CPU: one logit per (query, passage) pair."""

    def __init__(self, spec: dict, config: dict, models_root: Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        import torch

        model_dir = models_root / "bge-reranker-v2-m3"
        sys.path.insert(0, str(EMBEDDINGS))
        from embed import load_models, verify_model_dir  # noqa: E402

        pinned = load_models()["bge-reranker-v2-m3"]
        if pinned["id"] != config["model"] or pinned["revision"] != config["revision"]:
            raise SystemExit("config.toml and research/embeddings/models.toml disagree on the reranker")
        verify_model_dir(pinned, model_dir)
        torch.set_num_threads(spec.get("threads", 4))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), torch_dtype=torch.float32
        )
        self.model.eval()
        self.max_length = config["max_length"]
        self.batch_size = config["batch_size"]

    def __call__(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                chunk = passages[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [query] * len(chunk),
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                logits = self.model(**encoded).logits.view(-1).to(self.torch.float32)
                scores.extend(float(x) for x in logits)
        return scores


class CachedScorer:
    """Memoise (query, passage) -> score, bounded. A score depends only on the pair, so a
    repeated pair (the same query at several checkpoints) is not recomputed."""

    def __init__(self, scorer, capacity: int = 200_000):
        self.scorer = scorer
        self.capacity = capacity
        self.cache: dict[tuple[str, str], float] = {}
        self.hits = 0

    def __call__(self, query: str, passages: list[str]) -> list[float]:
        known = {p: self.cache[(query, p)] for p in passages if (query, p) in self.cache}
        missing = [p for p in dict.fromkeys(passages) if p not in known]
        if missing:
            computed = self.scorer(query, missing)
            if len(computed) != len(missing):
                raise RuntimeError("reranker returned the wrong number of scores")
            for passage, score in zip(missing, computed):
                known[passage] = score
                if len(self.cache) >= self.capacity:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[(query, passage)] = score
        self.hits += len(passages) - len(missing)
        return [known[p] for p in passages]


class StrongRetrieval:
    """The pipeline, with the reranker injected as `scorer(query, passages) -> [float]`."""

    def __init__(self, config: dict, scorer=None):
        self.config = config
        self.scorer = scorer
        self.lexical_top = config["lexical"]["top_k"]
        self.dense_top = config["dense"]["top_k"]
        self.rrf_k = float(config["fusion"]["k"])
        self.rerank_top = config["reranker"]["top_k"]
        self.reset()

    def reset(self) -> None:
        self.lexical = BM25(k1=self.config["lexical"]["k1"], b=self.config["lexical"]["b"])
        self.texts: dict[str, str] = {}
        self.vectors: dict[str, list[float]] = {}
        self.dimension = None

    def upsert(self, doc) -> None:
        if not isinstance(doc, dict):
            raise ValueError("document must be an object")
        doc_id = _check_id(doc.get("id"))
        text = doc.get("text")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        embedding = _check_embedding(doc.get("embedding"), self.dimension)
        self.dimension = len(embedding)
        self.lexical.upsert(doc_id, text)
        self.texts[doc_id] = text
        self.vectors[doc_id] = _normalised(embedding)

    def delete(self, doc_id: str) -> None:
        self.lexical.delete(doc_id)
        self.texts.pop(doc_id, None)
        self.vectors.pop(doc_id, None)

    def _dense(self, query_vector: list[float], depth: int) -> list[tuple[str, float]]:
        q = _normalised(query_vector)
        scored = ((i, sum(a * b for a, b in zip(q, v))) for i, v in self.vectors.items())
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:depth]

    def search(self, query: str, query_vector: list[float], limit: int) -> dict:
        lexical = self.lexical.search(query, limit=self.lexical_top)
        dense = self._dense(query_vector, self.dense_top)
        fused: dict[str, float] = {}
        for ranking in (lexical, dense):
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        order = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        candidates = order[: self.rerank_top]
        reply = {
            "stages": {"lexical": len(lexical), "dense": len(dense), "fused": len(order),
                       "rerank_input": len(candidates)},
            "reranked": False,
            "rerank_seconds": 0.0,
        }
        if self.scorer is None or not candidates:
            reply["hits"] = [{"id": i, "score": s} for i, s in candidates[:limit]]
            return reply
        started = time.monotonic()
        scores = self.scorer(query, [self.texts[i] for i, _ in candidates])
        if len(scores) != len(candidates):
            raise RuntimeError("reranker returned the wrong number of scores")
        reranked = sorted(zip((i for i, _ in candidates), scores), key=lambda item: (-item[1], item[0]))
        reply["hits"] = [{"id": i, "score": s} for i, s in reranked[:limit]]
        reply["reranked"] = True
        reply["rerank_seconds"] = round(time.monotonic() - started, 4)
        return reply


class Session:
    def __init__(self, scorer=None, reranker_name="none"):
        self.config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
        self.pipeline = StrongRetrieval(self.config, scorer)
        self.reranker_name = reranker_name

    def handle(self, request) -> dict:
        if not isinstance(request, dict):
            return {"ok": False, "error": "request must be a JSON object"}
        op = request.get("op")
        try:
            if op == "hello":
                return {
                    "ok": True,
                    "driver": "strong_retrieval",
                    "protocol": PROTOCOL,
                    "dense": {"model": self.config["dense"]["model"],
                              "revision": self.config["dense"]["revision"]},
                    "reranker": {"model": self.config["reranker"]["model"],
                                 "revision": self.config["reranker"]["revision"],
                                 "active": self.reranker_name != "none"},
                    "rrf_k": self.pipeline.rrf_k,
                }
            if op == "quit":
                return {"ok": True}
            if op == "reset":
                self.pipeline.reset()
                return {"ok": True}
            if op == "upsert":
                self.pipeline.upsert(request)
                return {"ok": True}
            if op == "upsert_many":
                docs = request.get("docs")
                if not isinstance(docs, list):
                    raise ValueError("docs must be a list")
                for doc in docs:
                    self.pipeline.upsert(doc)
                return {"ok": True, "upserted": len(docs)}
            if op == "delete":
                self.pipeline.delete(_check_id(request.get("id")))
                return {"ok": True}
            if op == "size":
                return {"ok": True, "documents": len(self.pipeline.vectors)}
            if op == "search":
                query = request.get("query")
                if not isinstance(query, str):
                    raise ValueError("query must be a string")
                limit = request.get("limit", 10)
                if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
                    raise ValueError(f"limit must be an integer in 1..{MAX_LIMIT}")
                embedding = _check_embedding(request.get("embedding"), self.pipeline.dimension)
                reply = self.pipeline.search(query, embedding, limit)
                reply["ok"] = True
                return reply
            return {"ok": False, "error": f"unknown op {op!r}"}
        except ValueError as error:
            return {"ok": False, "error": str(error)}


def serve(session: Session, lines, out) -> None:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        quit_requested = False
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            reply = {"ok": False, "error": f"invalid JSON: {error}"}
        else:
            reply = session.handle(request)
            quit_requested = isinstance(request, dict) and request.get("op") == "quit"
        out.write(json.dumps(reply, separators=(",", ":")) + "\n")
        out.flush()
        if quit_requested:
            return


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker", choices=["bge", "none"], default="bge")
    parser.add_argument("--models-root", type=Path, help="directory holding bge-reranker-v2-m3/")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    scorer = None
    if args.reranker == "bge":
        if args.models_root is None:
            parser.error("--reranker bge needs --models-root (or pass --reranker none)")
        scorer = CachedScorer(
            CrossEncoder({"threads": args.threads}, config["reranker"], args.models_root)
        )
    serve(Session(scorer, args.reranker), sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
