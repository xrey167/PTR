"""Smoke-measure a baseline driver on the Q003 pool (not the Q003 experiment).

    python3 measure.py --driver strong --models-root <dir> --queries 40
    python3 measure.py --driver rag --queries 400

Indexes version 2 of every cue's fact (so versions 1 and 3 are stale, never indexed),
queries the first paraphrase of a seeded sample of cues, and reports recall@1, recall@10,
MRR and wall time per query, plus the reranker's share for `strong`. The driver runs as a
child process over the JSON protocol, exactly as a Rust harness would run it. The numbers
only show that the pipeline works and what it costs; they are not evidence about PTR.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "embeddings"))

from emb_common import SplitMix64  # noqa: E402
from pool_gen import load_pool  # noqa: E402

DRIVERS = {
    "strong": (ROOT / "research/baselines/strong_retrieval/driver.py", "q003-bge-m3", 1024),
    "rag": (ROOT / "research/baselines/rag_reference/q003_driver.py", "q003", 384),
}


def read_rows(path: Path, dim: int, wanted: set[int]) -> dict[int, list[float]]:
    out = {}
    data = path.read_bytes()
    for row in sorted(wanted):
        out[row] = list(struct.unpack_from(f"<{dim}e", data, row * dim * 2))
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", choices=sorted(DRIVERS), required=True)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    script, vectors_name, dim = DRIVERS[args.driver]
    pool = load_pool("q003")
    vectors = ROOT / "research/embeddings" / f"vectors-{vectors_name}.f16"
    facts = [(i, r) for i, r in enumerate(pool) if r["kind"] == "fact" and r["version"] == args.version]
    queries = [(i, r) for i, r in enumerate(pool) if r["kind"] == "query" and r["paraphrase"] == 0]
    sample = SplitMix64("ptr/measure/q003")
    chosen = sample.sample(queries, args.queries)
    wanted = {i for i, _ in facts} | {i for i, _ in chosen}
    embeddings = read_rows(vectors, dim, wanted)

    command = [sys.executable, str(script)]
    if args.driver == "strong":
        command += ["--reranker", "none"] if args.no_rerank else ["--reranker", "bge", "--models-root", str(args.models_root),
                                                                "--threads", str(args.threads)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def call(request):
        child.stdin.write(json.dumps(request) + "\n")
        child.stdin.flush()
        reply = json.loads(child.stdout.readline())
        if not reply["ok"]:
            raise SystemExit(f"driver error: {reply}")
        return reply

    print(json.dumps(call({"op": "hello"})))
    started = time.monotonic()
    docs = [{"id": r["id"], "text": r["text"], "embedding": embeddings[i]} for i, r in facts]
    for start in range(0, len(docs), 500):
        call({"op": "upsert_many", "docs": docs[start : start + 500]})
    index_seconds = time.monotonic() - started
    target = {r["cue"]: r["id"] for _, r in facts}
    hit1 = hit10 = 0
    mrr = 0.0
    rerank = 0.0
    started = time.monotonic()
    for i, r in chosen:
        reply = call({"op": "search", "query": r["text"], "embedding": embeddings[i], "limit": 10})
        ids = [h["id"] for h in reply["hits"]]
        rerank += reply.get("rerank_seconds", 0.0)
        want = target[r["cue"]]
        if ids[:1] == [want]:
            hit1 += 1
        if want in ids:
            hit10 += 1
            mrr += 1.0 / (ids.index(want) + 1)
    query_seconds = time.monotonic() - started
    call({"op": "quit"})
    child.wait(timeout=30)
    n = len(chosen)
    print(json.dumps({
        "driver": args.driver, "indexed": len(docs), "queries": n,
        "recall_at_1": round(hit1 / n, 4), "recall_at_10": round(hit10 / n, 4), "mrr_at_10": round(mrr / n, 4),
        "index_seconds": round(index_seconds, 1), "seconds_per_query": round(query_seconds / n, 3),
        "rerank_seconds_per_query": round(rerank / n, 3),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
