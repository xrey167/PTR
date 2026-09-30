"""The committed samples validate against operator_route.schema.json and the dataset validator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from _support import ROOT, SCHEMA_PATH, gen, schema_errors

SAMPLES = [ROOT / "datasets/samples/operator_route.jsonl", ROOT / "datasets/samples/operator_route_v1.jsonl"]


def records(path):
    """Load the nonempty lines of a JSONL sample as records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Schema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the committed operator-routing schema for sample validation."""
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_codebook_version_is_an_integer(self):
        """Check codebook field types and the schema's backward-compatible required fields."""
        self.assertEqual(self.schema["properties"]["type_codebook_version"]["type"], "integer")
        self.assertIn("type_codebook_fingerprint", self.schema["properties"])
        self.assertEqual(self.schema["required"], ["task", "routes"])

    def test_samples_validate(self):
        """Validate every committed operator-routing sample against the schema."""
        for path in SAMPLES:
            for index, record in enumerate(records(path)):
                with self.subTest(path=path.name, record=index):
                    self.assertEqual(schema_errors(self.schema, record), [])

    def test_the_schema_rejects_what_it_should(self):
        """Reject string codebook versions, unknown fields, and duplicate tags."""
        record = records(SAMPLES[1])[0]
        self.assertTrue(schema_errors(self.schema, {**record, "type_codebook_version": "1"}))
        self.assertTrue(schema_errors(self.schema, {**record, "unexpected": 1}))
        self.assertTrue(schema_errors(self.schema, {**record, "tags": ["validity", "validity"]}))

    def test_v1_sample_is_example_zero_of_every_split(self):
        """Check that the v1 sample contains the first generated example from every split."""
        rows = records(SAMPLES[1])
        self.assertEqual([r["split"] for r in rows], gen.SPLITS)
        for row in rows:
            with self.subTest(split=row["split"]):
                self.assertEqual(row, gen.build(gen.sample(row["split"], 0))[0])


class DatasetValidator(unittest.TestCase):
    def test_cli_accepts_both_samples(self):
        """Run the training dataset validator CLI on both committed routing samples."""
        env = {**os.environ, "PYTHONPATH": str(ROOT / "training/src")}
        for path in SAMPLES:
            with self.subTest(path=path.name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "training/src/ptr_training/validate_dataset.py"), str(path)],
                    cwd=ROOT, env=env, capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("bad=0", result.stdout)

    def test_operator_route_rules_hold_for_v1_records(self):
        """Apply routing-specific validation directly to v1 records despite filename dispatch."""
        # validate_dataset.py dispatches on the file name, so the CLI gives
        # operator_route_v1.jsonl only the generic check. Apply the operator-route
        # rules (operator names, targets summing to 1, codebook identity) directly.
        sys.path.insert(0, str(ROOT / "training/src"))
        try:
            from ptr_training import validate_dataset
        finally:
            sys.path.remove(str(ROOT / "training/src"))
        for record in records(SAMPLES[1]):
            with self.subTest(split=record["split"]):
                validate_dataset._validate_operator_route(record)


if __name__ == "__main__":
    unittest.main()
