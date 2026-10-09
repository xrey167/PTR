import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DRIVER = HERE / "q003_driver.py"
sys.path.insert(0, str(HERE))

import q003_driver  # noqa: E402


def run(requests):
    """Feed lines to the real process and return its decoded replies."""
    payload = "\n".join(r if isinstance(r, str) else json.dumps(r) for r in requests) + "\n"
    done = subprocess.run(
        [sys.executable, str(DRIVER)], input=payload, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return [json.loads(line) for line in done.stdout.splitlines()]


class ProtocolTests(unittest.TestCase):
    def test_hello_reports_the_pinned_embedding_model(self):
        (reply,) = run([{"op": "hello"}])
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["protocol"], 1)
        self.assertEqual(reply["embedding_model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(reply["rrf_k"], 60.0)

    def test_upsert_search_delete_round_trip(self):
        replies = run(
            [
                {"op": "upsert", "id": "a", "text": "red apple", "embedding": [1, 0]},
                {"op": "upsert_many", "docs": [{"id": "b", "text": "blue ocean", "embedding": [0, 1]}]},
                {"op": "size"},
                {"op": "search", "query": "apple", "embedding": [1, 0], "limit": 2},
                {"op": "delete", "id": "a"},
                {"op": "search", "query": "apple", "embedding": [1, 0], "limit": 2},
                {"op": "reset"},
                {"op": "size"},
            ]
        )
        self.assertTrue(all(r["ok"] for r in replies))
        self.assertEqual(replies[2]["documents"], 2)
        self.assertEqual([h["id"] for h in replies[3]["hits"]], ["a", "b"])
        self.assertNotIn("a", [h["id"] for h in replies[5]["hits"]])
        self.assertEqual(replies[7]["documents"], 0)

    def test_hits_equal_hybridrag_directly(self):
        from hybrid import HybridRAG

        docs = [("d1", "green tea garden", [1, 0, 0]), ("d2", "black coffee beans", [0, 1, 0]),
                ("d3", "garden tea party", [0.8, 0.1, 0.1])]
        rag = HybridRAG()
        for doc_id, text, vec in docs:
            rag.upsert(doc_id, text, vec)
        expected = rag.search("tea garden", [1, 0, 0], 3)
        reqs = [{"op": "upsert", "id": i, "text": t, "embedding": v} for i, t, v in docs]
        reqs.append({"op": "search", "query": "tea garden", "embedding": [1, 0, 0], "limit": 3})
        reply = run(reqs)[-1]
        self.assertEqual([(h["id"], h["score"]) for h in reply["hits"]], expected)

    def test_bad_requests_get_error_replies_and_the_driver_keeps_running(self):
        replies = run(
            [
                "not json",
                "[1, 2]",
                {"op": "nope"},
                {"op": "upsert", "id": "", "text": "x", "embedding": [1]},
                {"op": "upsert", "id": "a", "text": "x", "embedding": [1, 0]},
                {"op": "upsert", "id": "b", "text": "x", "embedding": [1, 0, 0]},
                {"op": "upsert", "id": "c", "text": "x", "embedding": [1, "z"]},
                {"op": "search", "query": "x", "embedding": [1, 0], "limit": 0},
                {"op": "search", "query": 5, "embedding": [1, 0]},
                {"op": "size"},
            ]
        )
        self.assertEqual([r["ok"] for r in replies], [False] * 4 + [True] + [False] * 4 + [True])
        self.assertEqual(replies[-1]["documents"], 1)

    def test_quit_stops_reading(self):
        replies = run([{"op": "quit"}, {"op": "hello"}])
        self.assertEqual(len(replies), 1)

    def test_a_non_finite_embedding_is_refused(self):
        session = q003_driver.Session()
        reply = session.handle({"op": "upsert", "id": "a", "text": "x", "embedding": [float("nan")]})
        self.assertFalse(reply["ok"])


if __name__ == "__main__":
    unittest.main()
