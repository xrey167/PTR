"""BM25-only list for Q003: the lexical half of rag_reference, on its own.

    python3 experiments/retrieval/Q003-v3-postgres-hybrid/bm25_driver.py

The same stdin/stdout JSON-lines protocol as research/baselines/rag_reference/q003_driver.py
(protocol 1; one object per line in, one per line out, {"ok": true, ...} or {"ok": false, "error": ...}),
reduced to the operations Q003's descriptive "BM25 only" arm needs: hello, reset, upsert_many (the
documents' `embedding` is accepted and ignored), size, search (the query's `embedding` is ignored) and quit.
The ranking is `BM25` of research/baselines/rag_reference/runner.py, imported, not copied, so the
digest-pinned lexical implementation is what runs. This helper is the harness's, not a baseline: it
adds no retrieval logic and its list is reported, never judged.

Framing and flushing (Q003-v3): a request is one line and a reply is one line, and every reply is flushed before the next
request is read. The harness bounds each exchange with a timeout, so a reply left in a buffer ends the run with a typed stop
instead of a wait (the mutation plan plants that defect). A failure this driver does not handle ends it with a traceback and a
non-zero exit status, which the harness reports as a stop (reason driver-exited), never as a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(REPOSITORY / "research/baselines/rag_reference"))

from runner import BM25  # noqa: E402

PROTOCOL = 1
MAX_LIMIT = 1000


class Session:
    def __init__(self) -> None:
        self.bm25 = BM25()

    def handle(self, request) -> dict:
        if not isinstance(request, dict):
            return {"ok": False, "error": "request must be a JSON object"}
        op = request.get("op")
        try:
            if op == "hello":
                return {"ok": True, "driver": "q003_bm25", "protocol": PROTOCOL}
            if op == "quit":
                return {"ok": True}
            if op == "reset":
                self.bm25 = BM25()
                return {"ok": True}
            if op == "upsert_many":
                docs = request.get("docs")
                if not isinstance(docs, list):
                    raise ValueError("docs must be a list")
                for doc in docs:
                    if not isinstance(doc, dict) or not isinstance(doc.get("id"), str) or not doc["id"]:
                        raise ValueError("a document needs a non-empty string id")
                    if not isinstance(doc.get("text"), str):
                        raise ValueError("text must be a string")
                    self.bm25.upsert(doc["id"], doc["text"])
                return {"ok": True, "upserted": len(docs)}
            if op == "size":
                return {"ok": True, "documents": len(self.bm25.docs)}
            if op == "search":
                query = request.get("query")
                if not isinstance(query, str):
                    raise ValueError("query must be a string")
                limit = request.get("limit", 10)
                if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
                    raise ValueError(f"limit must be an integer in 1..{MAX_LIMIT}")
                hits = self.bm25.search(query, limit=limit)
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
