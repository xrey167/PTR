import tempfile
import unittest
from pathlib import Path

from ptr_training import validate_dataset
from ptr_training.validate_dataset import validate

_VERSION = validate_dataset.CODEBOOK["version"]
_FINGERPRINT = validate_dataset.CODEBOOK["fingerprint_sha256"]


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
                '{"operator":"probabilistic","target":0.3}],'
                f'"type_codebook_version":{_VERSION},'
                f'"type_codebook_fingerprint":"{_FINGERPRINT}"}}\n',
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
                '"distribution":{"yes":0.6,"no":0.4},"outcome":"yes",'
                f'"type_codebook_version":{_VERSION},'
                f'"type_codebook_fingerprint":"{_FINGERPRINT}"}}\n',
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


class CodebookIdentityTests(unittest.TestCase):
    """A dataset must name the codebook it was produced under, and be checked."""

    def _record(self, **overrides):
        record = {
            "task": "forecast",
            "routes": [{"operator": "statistical", "target": 1.0}],
            "type_codebook_version": validate_dataset.CODEBOOK["version"],
            "type_codebook_fingerprint": validate_dataset.CODEBOOK["fingerprint_sha256"],
        }
        record.update(overrides)
        return record

    def test_a_record_naming_this_codebook_is_accepted(self):
        validate_dataset.validate_record(Path("operator_route.jsonl"), self._record())

    def test_an_absent_version_or_fingerprint_is_refused_with_no_default(self):
        for field in ("type_codebook_version", "type_codebook_fingerprint"):
            record = self._record()
            del record[field]
            with self.assertRaises(ValueError) as caught:
                validate_dataset.validate_record(Path("operator_route.jsonl"), record)
            self.assertIn(f"{field} is required", str(caught.exception))

    def test_an_unknown_version_and_a_moved_fingerprint_give_distinct_reasons(self):
        with self.assertRaises(ValueError) as unknown:
            validate_dataset.validate_record(
                Path("operator_route.jsonl"), self._record(type_codebook_version=99)
            )
        self.assertIn("is not this build's", str(unknown.exception))

        with self.assertRaises(ValueError) as moved:
            validate_dataset.validate_record(
                Path("operator_route.jsonl"),
                self._record(type_codebook_fingerprint="0" * 64),
            )
        self.assertIn("does not match this build's assignment", str(moved.exception))
        # Two faults, two reasons: the version can be right while the table
        # behind it has moved.
        self.assertNotEqual(str(unknown.exception), str(moved.exception))

    def test_the_member_sets_come_from_the_kernel_artifact(self):
        # Not a copy kept in Python: the sets must equal what the artifact lists.
        members = validate_dataset.CODEBOOK["members"]
        self.assertEqual(validate_dataset.REASONING_OPERATORS, members["reasoning_operator"])
        self.assertEqual(validate_dataset.EPISTEMIC_STATES, members["epistemic_state"])
        self.assertEqual(validate_dataset.UNCERTAINTY_KINDS, members["uncertainty_kind"])
        self.assertEqual(len(validate_dataset.REASONING_OPERATORS), 11)

    def test_the_epistemic_validator_also_requires_the_identity(self):
        record = {
            "question": "will it rain",
            "distribution": {"yes": 0.5, "no": 0.5},
            "outcome": "yes",
        }
        with self.assertRaises(ValueError) as caught:
            validate_dataset.validate_record(Path("epistemic_calibration.jsonl"), record)
        self.assertIn("type_codebook_version is required", str(caught.exception))
