import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import pool_gen  # noqa: E402
from emb_common import SplitMix64, count_tokens  # noqa: E402


class TokenCountTests(unittest.TestCase):
    def test_counts_follow_the_pinned_regex(self):
        self.assertEqual(count_tokens("Where does Ana Ng live?"), 6)
        self.assertEqual(count_tokens("a-b c1 2"), 5)  # a - b c1 2
        self.assertEqual(count_tokens("It's  here."), 5)  # It ' s here .

    def test_non_ascii_is_refused(self):
        with self.assertRaises(ValueError):
            count_tokens("café")
        with self.assertRaises(ValueError):
            count_tokens("tab\there")
        with self.assertRaises(ValueError):
            count_tokens("")


class PrngTests(unittest.TestCase):
    def test_splitmix_is_stable(self):
        # Pinned so a Rust reimplementation can be checked against these values.
        rng = SplitMix64("ptr/emb/test")
        first = [rng.next_u64() for _ in range(3)]
        again = SplitMix64("ptr/emb/test")
        self.assertEqual(first, [again.next_u64() for _ in range(3)])
        self.assertNotEqual(first, [SplitMix64("other").next_u64() for _ in range(3)])

    def test_below_stays_in_range_and_sample_is_distinct(self):
        rng = SplitMix64("x")
        self.assertTrue(all(0 <= rng.below(7) < 7 for _ in range(200)))
        picked = rng.sample(list(range(20)), 8)
        self.assertEqual(len(set(picked)), 8)
        with self.assertRaises(ValueError):
            rng.sample([1, 2], 3)


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {name: pool_gen.build_pool(spec) for name, spec in pool_gen.SPECS.items()}

    def test_generation_is_deterministic(self):
        for name, spec in pool_gen.SPECS.items():
            self.assertEqual(pool_gen.render(pool_gen.build_pool(spec)), pool_gen.render(self.built[name]))

    def test_committed_files_equal_the_generator_output(self):
        for name, rows in self.built.items():
            self.assertEqual(pool_gen.pool_path(name).read_bytes(), pool_gen.render(rows), name)

    def test_sizes_match_the_design(self):
        self.assertEqual(len(self.built["q003"]), 12000)
        self.assertEqual(len(self.built["m008"]), 14000)
        q = self.built["q003"]
        self.assertEqual(sum(r["kind"] == "fact" for r in q), 6000)
        self.assertEqual(len({r["cue"] for r in q}), 2000)
        m = self.built["m008"]
        self.assertEqual(sum(r["kind"] == "offtopic" for r in m), 700)
        self.assertEqual(len({r["cue"] for r in m if r["cue"] >= 0}), 1900)

    def test_token_counts_are_exact_and_independent(self):
        token = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
        for rows in self.built.values():
            for row in rows:
                self.assertEqual(row["tokens"], len(token.findall(row["text"])), row["id"])
                self.assertGreater(row["tokens"], 0)

    def test_ids_and_texts_are_unique_and_ascii(self):
        for rows in self.built.values():
            self.assertEqual(len({r["id"] for r in rows}), len(rows))
            self.assertEqual(len({r["text"] for r in rows}), len(rows))
            for row in rows:
                row["text"].encode("ascii")

    def test_every_cue_has_distinct_versions_and_queries_never_leak_values(self):
        for name, rows in self.built.items():
            by_cue = {}
            for row in rows:
                if row["cue"] >= 0:
                    by_cue.setdefault(row["cue"], []).append(row)
            for cue, group in list(by_cue.items())[:300]:
                facts = [r for r in group if r["kind"] == "fact"]
                queries = [r for r in group if r["kind"] == "query"]
                self.assertEqual([r["version"] for r in facts], [1, 2, 3], (name, cue))
                self.assertEqual(len({r["text"] for r in facts}), 3)
                attribute = next(a for a in pool_gen.ATTRIBUTES if a[0] == group[0]["attribute"])
                for query in queries:
                    for value in attribute[3]:
                        self.assertIsNone(re.search(rf"\b{re.escape(value)}\b", query["text"]), (name, cue))

    def test_off_topic_queries_name_no_universe_entity(self):
        for name, rows in self.built.items():
            spec = pool_gen.SPECS[name]
            if not spec.offtopic:
                continue
            universe = {r["entity"] for r in rows if r["cue"] >= 0}
            self.assertTrue(all(r["entity"] not in universe for r in rows if r["kind"] == "offtopic"))
            first_words = {r["text"] for r in rows if r["kind"] == "fact"}
            self.assertTrue(first_words)

    def test_the_two_pools_use_different_entities(self):
        q = {r["text"] for r in self.built["q003"] if r["kind"] == "query"}
        m = {r["text"] for r in self.built["m008"] if r["kind"] == "query"}
        self.assertLess(len(q & m), len(q) // 4)

    def test_fact_token_maximum_is_bounded(self):
        for rows in self.built.values():
            self.assertLessEqual(pool_gen.stats(rows)["facts"]["max"], 24)


if __name__ == "__main__":
    unittest.main()
