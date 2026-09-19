import tempfile
import unittest
from pathlib import Path

from ptr_training.validate_dataset import validate


class DatasetToolsTests(unittest.TestCase):
    def test_validate_jsonl_counts_bad_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
            ok, bad = validate(path)
            self.assertEqual((ok, bad), (1, 1))

    def test_operator_route_rejects_unknown_reasoning_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "operator_route.jsonl"
            path.write_text(
                '{"task":"x","routes":[{"operator":"free-form-string","target":1.0}]}\n',
                encoding="utf-8",
            )
            ok, bad = validate(path)
            self.assertEqual((ok, bad), (0, 1))

    def test_operator_route_accepts_shared_reasoning_operator_taxonomy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "operator_route.jsonl"
            path.write_text(
                '{"task":"x","routes":['
                '{"operator":"statistical","target":0.7},'
                '{"operator":"probabilistic","target":0.3}]}\n',
                encoding="utf-8",
            )
            ok, bad = validate(path)
            self.assertEqual((ok, bad), (1, 0))

    def test_epistemic_calibration_keeps_state_and_uncertainty_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epistemic_calibration.jsonl"
            path.write_text(
                '{"question":"q","epistemic_state":"hypothesis",'
                '"uncertainty_kind":"distribution",'
                '"distribution":{"yes":0.6,"no":0.4},"outcome":"yes"}\n',
                encoding="utf-8",
            )
            ok, bad = validate(path)
            self.assertEqual((ok, bad), (1, 0))

    def test_epistemic_calibration_rejects_distribution_as_epistemic_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epistemic_calibration.jsonl"
            path.write_text(
                '{"question":"q","epistemic_state":"distribution",'
                '"distribution":{"yes":1.0},"outcome":"yes"}\n',
                encoding="utf-8",
            )
            ok, bad = validate(path)
            self.assertEqual((ok, bad), (0, 1))


if __name__ == "__main__":
    unittest.main()
