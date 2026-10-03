from __future__ import annotations

import unittest

from _support import gen


class LeakageTests(unittest.TestCase):
    def test_target_cell_absent_from_train_val_and_iid(self):
        for fold in gen.FOLDS:
            for split in ("train", "val", "test_iid"):
                for index in range(gen.SPLIT_SIZES[split]):
                    example = gen.sample(fold.name, split, index)
                    with self.subTest(fold=fold.name, split=split, index=index):
                        self.assertFalse(gen.contains_target(fold, example.regime, example.facts, usable_only=False))

    def test_every_ood_example_contains_live_nonzero_target_fact(self):
        for fold in gen.FOLDS:
            for index in range(gen.SPLIT_SIZES["test_ood"]):
                example = gen.sample(fold.name, "test_ood", index)
                with self.subTest(fold=fold.name, index=index):
                    self.assertEqual(example.regime, fold.regime)
                    self.assertTrue(gen.contains_target(fold, example.regime, example.facts, usable_only=True))

    def test_confirmatory_and_diagnostic_partition_is_explicit(self):
        self.assertEqual(
            gen.CONFIRMATORY_FOLDS,
            ("evidence-temporal", "evidence-tabular", "claim-interventional"),
        )
        self.assertEqual(
            tuple(fold.name for fold in gen.FOLDS if fold.kind == "diagnostic"),
            ("evidence-interventional",),
        )
        self.assertEqual(gen.DEVELOPMENT_FOLDS, ("claim-temporal-development",))


if __name__ == "__main__":
    unittest.main()
