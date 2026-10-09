"""Thin stdin/stdout JSON driver around HybridRAG, for harnesses written in Rust.

    python3 research/baselines/rag_reference/q003_driver.py

One JSON object per input line, one JSON object per output line, in order. A reply is
{"ok": true, ...} or {"ok": false, "error": "..."}; a malformed or invalid request
gets an error reply and the driver keeps running. End of input ends the driver.

Operations (protocol 1; strong_retrieval/driver.py speaks the same protocol):

    {"op": "hello"}
    {"op": "reset"}
    {"op": "upsert", "id": str, "text": str, "embedding": [number, ...]}
    {"op": "upsert_many", "docs": [{"id", "text", "embedding"}, ...]}
    {"op": "delete", "id": str}
    {"op": "size"}
    {"op": "search", "query": str, "embedding": [number, ...], "limit": int}
        -> {"ok": true, "hits": [{"id": str, "score": number}, ...]}
    {"op": "quit"}

It adds no retrieval logic: BM25 and HybridRAG are imported from this directory, so the
digest-pinned baseline is what runs. Embeddings are caller-supplied (the pinned vectors).
"""

from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hybrid import HybridRAG  # noqa: E402

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


class Session:
    def __init__(self):
        config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.rag = HybridRAG(rrf_k=float(self.config["hybrid"]["rrf_k"]))
        self.dimension = None

    def _upsert(self, doc) -> None:
        if not isinstance(doc, dict):
            raise ValueError("document must be an object")
        doc_id = _check_id(doc.get("id"))
        text = doc.get("text")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        embedding = _check_embedding(doc.get("embedding"), self.dimension)
        self.dimension = len(embedding)
        self.rag.upsert(doc_id, text, embedding)

    def handle(self, request) -> dict:
        if not isinstance(request, dict):
            return {"ok": False, "error": "request must be a JSON object"}
        op = request.get("op")
        try:
            if op == "hello":
                return {
                    "ok": True,
                    "driver": "rag_reference",
                    "protocol": PROTOCOL,
                    "rrf_k": self.rag.rrf_k,
                    "bm25": self.config["bm25"],
                    "embedding_model": self.config["hybrid"]["embedding_model"],
                }
            if op == "quit":
                return {"ok": True}
            if op == "reset":
                self.reset()
                return {"ok": True}
            if op == "upsert":
                self._upsert(request)
                return {"ok": True}
            if op == "upsert_many":
                docs = request.get("docs")
                if not isinstance(docs, list):
                    raise ValueError("docs must be a list")
                for doc in docs:
                    self._upsert(doc)
                return {"ok": True, "upserted": len(docs)}
            if op == "delete":
                self.rag.delete(_check_id(request.get("id")))
                return {"ok": True}
            if op == "size":
                return {"ok": True, "documents": len(self.rag.vectors)}
            if op == "search":
                query = request.get("query")
                if not isinstance(query, str):
                    raise ValueError("query must be a string")
                limit = request.get("limit", 10)
                if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
                    raise ValueError(f"limit must be an integer in 1..{MAX_LIMIT}")
                embedding = _check_embedding(request.get("embedding"), self.dimension)
                hits = self.rag.search(query, embedding, limit)
                return {"ok": True, "hits": [{"id": i, "score": s} for i, s in hits]}
            return {"ok": False, "error": f"unknown op {op!r}"}
        except ValueError as error:
            return {"ok": False, "error": str(error)}


def serve(lines, out) -> None:
    session = Session()
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


if __name__ == "__main__":
    serve(sys.stdin, sys.stdout)
