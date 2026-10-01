from __future__ import annotations

import hashlib
import unittest

from _support import gen, lock


class DigestTests(unittest.TestCase):
    def test_full_regeneration_matches_lock(self):
        self.assertEqual(gen.lock_document(gen.render_all(verbose=False)), lock())

    def test_lock_identity_and_fold_kinds(self):
        document = lock()
        self.assertEqual(document["generator_version"], 2)
        self.assertEqual(document["benchmark_seed"], 20261001)
        self.assertEqual(
            [name for name, fold in document["folds"].items() if fold["kind"] == "confirmatory"],
            list(gen.CONFIRMATORY_FOLDS),
        )
        self.assertEqual(
            [name for name, fold in document["folds"].items() if fold["kind"] == "diagnostic"],
            ["evidence-interventional"],
        )
        self.assertEqual(
            [name for name, fold in document["folds"].items() if fold["kind"] == "development"],
            ["claim-temporal-development"],
        )

    def test_regenerated_prefixes_match_lock(self):
        document = lock()
        for fold in gen.FOLDS:
            for split in gen.SPLITS:
                jsonl, tsv, _labels = gen.render_split(fold.name, split, document["prefix_n"])
                pinned = document["folds"][fold.name]["splits"][split]
                with self.subTest(fold=fold.name, split=split):
                    self.assertEqual(hashlib.sha256(jsonl).hexdigest(), pinned["prefix_jsonl_sha256"])
                    self.assertEqual(hashlib.sha256(tsv).hexdigest(), pinned["prefix_tsv_sha256"])

    def test_repeated_render_is_byte_identical(self):
        first = gen.render_split("claim-interventional", "test_ood", 40)
        self.assertEqual(first, gen.render_split("claim-interventional", "test_ood", 40))


if __name__ == "__main__":
    unittest.main()
