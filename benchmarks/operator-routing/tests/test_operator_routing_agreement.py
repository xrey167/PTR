"""generator.py and score.py agree on every record; the TSV and JSONL carry the same examples."""

from __future__ import annotations

import ast
import json
import unittest

from _support import DATA_DIR, SUITE, data_on_disk, gen, prefix, score, score_items_from_tsv

N_MEMORY = 300


class NoSharedCode(unittest.TestCase):
    def test_score_never_imports_the_generator(self):
        tree = ast.parse((SUITE / "score.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertFalse({name for name in imported if "generator" in name or "references" in name})


class LabelAgreement(unittest.TestCase):
    def test_in_memory_prefix_of_every_split(self):
        # Labels, tags, margins and utilities, on examples regenerated here.
        for split in gen.SPLITS:
            jsonl, tsv, _labels = prefix(split, N_MEMORY)
            items = score_items_from_tsv(split, tsv)
            records = [json.loads(line) for line in jsonl.decode("utf-8").splitlines()]
            for item, record in zip(items, records):
                with self.subTest(id=item.id):
                    self.assertEqual(item.gold, item.label)
                    self.assertEqual(record["label"]["code"], item.label)
                    self.assertEqual(record["tags"], item.tags)
                    self.assertAlmostEqual(record["margin"], item.margin, places=8)
                    self.assertEqual(score.round_trip_problems(item.id, item, record), [])

    @unittest.skipUnless(data_on_disk(), "generated splits are not on disk")
    def test_every_record_on_disk(self):
        # 100% of records: label, tags, margin, utility and the TSV/JSONL round trip.
        checked, problems = score.agree(DATA_DIR, verbose=False)
        self.assertEqual(problems, [])
        self.assertEqual(checked, sum(gen.SPLIT_SIZES.values()))


class RoundTrip(unittest.TestCase):
    def test_tsv_parses_back_to_the_generated_example(self):
        for split in gen.SPLITS:
            _jsonl, tsv, labels = prefix(split, 50)
            lines = tsv.decode("utf-8").splitlines()
            for index, line in enumerate(lines):
                example = gen.sample(split, index)
                ident, label, regime, budget, tokens, facts = line.split("\t")
                parsed = []
                for chunk in facts.split(";"):
                    r, e, c, v, ent = chunk.split(",")
                    parsed.append((int(r), int(e), float(c), int(v), int(ent)))
                with self.subTest(id=ident):
                    self.assertEqual(ident, example.id)
                    self.assertEqual((int(regime), int(budget)), (example.regime, example.budget))
                    self.assertEqual([int(t) for t in tokens.split(",")], example.tokens)
                    self.assertEqual(parsed, example.facts)  # c survives the 4-decimal text exactly
                    self.assertEqual(format(int(label), "x"), labels[index])

    def test_jsonl_names_are_codebook_names(self):
        jsonl, _tsv, _labels = prefix("ood_compose_regime", 50)
        for line in jsonl.decode("utf-8").splitlines():
            record = json.loads(line)
            self.assertEqual([r["operator"] for r in record["routes"]], gen.OP_NAMES)
            self.assertEqual(record["label"]["operator"], gen.OP_NAMES[record["label"]["code"]])
            for slot in record["slots"]:
                self.assertIn(slot["role"], gen.ROLE)
                self.assertIn(slot["epistemic"], gen.EPI)
                self.assertIn(slot["validity"], gen.VALIDITY)
            self.assertAlmostEqual(sum(r["target"] for r in record["routes"]), 1.0, places=12)
            # The label carries the largest target, up to a tie and the rounding residual.
            top = max(r["target"] for r in record["routes"])
            self.assertGreaterEqual(record["routes"][record["label"]["code"]]["target"], top - 1e-5)


if __name__ == "__main__":
    unittest.main()
