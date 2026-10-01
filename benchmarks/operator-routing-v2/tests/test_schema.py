from __future__ import annotations

import json
import unittest

from _support import SCHEMA_PATH, gen


class SchemaTests(unittest.TestCase):
    def test_schema_matches_generator_contract(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        properties = schema["properties"]
        self.assertEqual(properties["generator_version"]["enum"], [gen.GENERATOR_VERSION])
        self.assertEqual(properties["fold"]["enum"], [fold.name for fold in gen.FOLDS])
        self.assertEqual(properties["split"]["enum"], list(gen.SPLITS))

    def test_generated_records_have_exact_required_shape(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        required = set(schema["required"])
        allowed = set(schema["properties"])
        for fold in gen.FOLDS:
            for split in gen.SPLITS:
                record = gen.build(gen.sample(fold.name, split, 0))[0]
                with self.subTest(fold=fold.name, split=split):
                    self.assertEqual(set(record), allowed)
                    self.assertTrue(required <= set(record))
                    self.assertEqual(len(record["routes"]), 11)
                    self.assertEqual(len(record["slots"]), 6)
                    self.assertEqual(len(record["raw_tokens"]), 36)
                    self.assertAlmostEqual(sum(route["target"] for route in record["routes"]), 1.0)


if __name__ == "__main__":
    unittest.main()
