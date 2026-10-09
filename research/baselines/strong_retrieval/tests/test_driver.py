import json
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DRIVER = HERE / "driver.py"
sys.path.insert(0, str(HERE))

import driver  # noqa: E402

CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))


def run(requests):
    payload = "\n".join(r if isinstance(r, str) else json.dumps(r) for r in requests) + "\n"
    done = subprocess.run(
        [sys.executable, str(DRIVER), "--reranker", "none"],
        input=payload, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return [json.loads(line) for line in done.stdout.splitlines()]


def docs():
    return [
        {"id": "tea", "text": "green tea garden", "embedding": [1, 0, 0]},
        {"id": "coffee", "text": "black coffee beans", "embedding": [0, 1, 0]},
        {"id": "party", "text": "garden tea party", "embedding": [0.8, 0.1, 0.1]},
        {"id": "ocean", "text": "deep blue ocean", "embedding": [0, 0, 1]},
    ]


class ConfigTests(unittest.TestCase):
    def test_the_pins_are_the_designed_ones(self):
        self.assertEqual(CONFIG["dense"]["model"], "BAAI/bge-m3")
        self.assertEqual(CONFIG["dense"]["revision"], "5617a9f61b028005a4858fdac845db406aefb181")
        self.assertEqual(CONFIG["dense"]["mode"], "dense-only")
        self.assertEqual(CONFIG["reranker"]["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(CONFIG["reranker"]["revision"], "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
        self.assertEqual(CONFIG["reranker"]["top_k"], 50)
        self.assertEqual(CONFIG["fusion"]["k"], 60.0)

    def test_it_is_retrieval_only(self):
        self.assertNotIn("answer", CONFIG)

    def test_the_pins_agree_with_the_embeddings_models_file(self):
        models = tomllib.loads(
            (HERE.parent.parent / "embeddings" / "models.toml").read_text(encoding="utf-8")
        )["models"]
        for section, key in (("dense", "bge-m3"), ("reranker", "bge-reranker-v2-m3")):
            self.assertEqual(models[key]["id"], CONFIG[section]["model"])
            self.assertEqual(models[key]["revision"], CONFIG[section]["revision"])


class PipelineTests(unittest.TestCase):
    def pipeline(self, scorer=None):
        pipe = driver.StrongRetrieval(CONFIG, scorer)
        for doc in docs():
            pipe.upsert(doc)
        return pipe

    def test_fusion_without_a_reranker(self):
        reply = self.pipeline().search("tea garden", [1, 0, 0], 3)
        self.assertFalse(reply["reranked"])
        self.assertEqual({h["id"] for h in reply["hits"][:2]}, {"tea", "party"})

    def test_the_injected_reranker_decides_the_order(self):
        def prefer_coffee(query, passages):
            return [10.0 if "coffee" in p else 0.0 for p in passages]

        reply = self.pipeline(prefer_coffee).search("tea garden", [1, 0, 0], 3)
        self.assertTrue(reply["reranked"])
        self.assertEqual(reply["hits"][0]["id"], "coffee")

    def test_only_the_fused_top_k_reach_the_reranker(self):
        seen = []

        def spy(query, passages):
            seen.append(list(passages))
            return [0.0] * len(passages)

        config = json.loads(json.dumps(CONFIG))
        config["reranker"]["top_k"] = 2
        pipe = driver.StrongRetrieval(config, spy)
        for doc in docs():
            pipe.upsert(doc)
        pipe.search("tea garden", [1, 0, 0], 4)
        self.assertEqual(len(seen[0]), 2)

    def test_a_deleted_document_is_never_returned_or_reranked(self):
        seen = []

        def spy(query, passages):
            seen.extend(passages)
            return [0.0] * len(passages)

        pipe = self.pipeline(spy)
        pipe.delete("tea")
        reply = pipe.search("tea garden", [1, 0, 0], 4)
        self.assertNotIn("tea", [h["id"] for h in reply["hits"]])
        self.assertNotIn("green tea garden", seen)

    def test_a_reranker_returning_the_wrong_count_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self.pipeline(lambda q, p: [0.0]).search("tea", [1, 0, 0], 3)

    def test_ties_break_by_id(self):
        pipe = self.pipeline(lambda q, p: [1.0] * len(p))
        ids = [h["id"] for h in pipe.search("tea garden", [1, 0, 0], 4)["hits"]]
        self.assertEqual(ids, sorted(ids))


class CachedScorerTests(unittest.TestCase):
    def test_repeated_pairs_are_not_recomputed_and_order_is_kept(self):
        calls = []

        def base(query, passages):
            calls.append(list(passages))
            return [float(len(p)) for p in passages]

        cached = driver.CachedScorer(base)
        self.assertEqual(cached("q", ["aa", "b", "aa"]), [2.0, 1.0, 2.0])
        self.assertEqual(calls, [["aa", "b"]])
        self.assertEqual(cached("q", ["b", "ccc"]), [1.0, 3.0])
        self.assertEqual(calls[1], ["ccc"])
        self.assertEqual(cached("other", ["b"]), [1.0])
        self.assertEqual(len(calls), 3)

    def test_the_cache_is_bounded(self):
        cached = driver.CachedScorer(lambda q, ps: [0.0] * len(ps), capacity=2)
        cached("q", ["a", "b", "c"])
        self.assertLessEqual(len(cached.cache), 2)

    def test_a_wrong_score_count_is_an_error(self):
        with self.assertRaises(RuntimeError):
            driver.CachedScorer(lambda q, ps: [0.0])("q", ["a", "b"])


class ProtocolTests(unittest.TestCase):
    def test_hello_names_the_pins_and_that_no_reranker_is_active(self):
        (reply,) = run([{"op": "hello"}])
        self.assertEqual(reply["driver"], "strong_retrieval")
        self.assertEqual(reply["dense"]["revision"], CONFIG["dense"]["revision"])
        self.assertFalse(reply["reranker"]["active"])

    def test_search_reports_stages_and_that_it_was_not_reranked(self):
        reqs = [{"op": "upsert_many", "docs": docs()},
                {"op": "search", "query": "tea garden", "embedding": [1, 0, 0], "limit": 2}]
        reply = run(reqs)[-1]
        self.assertTrue(reply["ok"])
        self.assertFalse(reply["reranked"])
        self.assertEqual(reply["stages"]["fused"], 4)
        self.assertEqual(len(reply["hits"]), 2)

    def test_malformed_input_does_not_stop_the_driver(self):
        replies = run(["{", {"op": "search", "query": "x", "embedding": [1, 0], "limit": 3},
                       {"op": "upsert", "id": "a", "text": "x", "embedding": [1, 0]},
                       {"op": "search", "query": "x", "embedding": [1], "limit": 3},
                       {"op": "size"}])
        self.assertFalse(replies[0]["ok"])
        self.assertFalse(replies[3]["ok"])
        self.assertEqual(replies[4]["documents"], 1)

    def test_bge_without_a_models_root_is_a_usage_error(self):
        done = subprocess.run([sys.executable, str(DRIVER)], input="", capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)


if __name__ == "__main__":
    unittest.main()
