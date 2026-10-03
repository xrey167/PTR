from __future__ import annotations

import ast
import json
import unittest

from _support import SUITE, gen, score


class LabelAgreementTests(unittest.TestCase):
    def test_scorer_does_not_import_generator(self):
        tree = ast.parse((SUITE / "score.py").read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any("generator" in name for name in imports))

    def test_independent_labels_and_utilities_agree(self):
        for fold in gen.FOLDS:
            for split in gen.SPLITS:
                jsonl, tsv, _labels = gen.render_split(fold.name, split)
                records = [json.loads(line) for line in jsonl.splitlines()]
                items = [score.parse_tsv_line(line) for line in tsv.decode().splitlines(True)]
                for record, item in zip(records, items):
                    label, values = score.derive(item)
                    with self.subTest(id=item.id):
                        self.assertEqual(item.gold, label)
                        self.assertEqual(record["label"]["code"], label)
                        self.assertEqual(record["label"]["operator"], score.OP_NAMES[label])
                        self.assertEqual(record["utility"], [round(value, 10) + 0.0 for value in values])


if __name__ == "__main__":
    unittest.main()
