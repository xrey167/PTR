"""The codebook checker must catch a hand-edited artifact, not merely run."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_codebook", ROOT / "scripts/check_codebook.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def artifact(members=("goal", "claim"), **overrides):
    canonical = b"PTRCODEBOOK\x00" + b"".join(name.encode() for name in members)
    document = {
        "schema": "1",
        "version": 1,
        "canonical_bytes_hex": canonical.hex(),
        "fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
        "families": [
            {
                "family": "semantic_role",
                "cardinality": len(members),
                "members": [
                    {"code": index, "name": name} for index, name in enumerate(members)
                ],
            }
        ],
    }
    document.update(overrides)
    return document


def run(document):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "codebook.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return mod.check(path)


class ConsistencyTests(unittest.TestCase):
    def test_a_generated_artifact_passes(self):
        self.assertEqual(run(artifact()), [])

    def test_a_hand_edited_fingerprint_is_refused(self):
        self.assertTrue(
            any("not the digest" in e for e in run(artifact(fingerprint_sha256="0" * 64)))
        )

    def test_a_cardinality_that_disagrees_with_the_member_list_is_refused(self):
        document = artifact()
        document["families"][0]["cardinality"] = 9
        self.assertTrue(any("does not match" in e for e in run(document)))

    def test_codes_must_be_dense_because_they_index_an_embedding_table(self):
        document = artifact()
        document["families"][0]["members"][1]["code"] = 5
        self.assertTrue(any("not dense" in e for e in run(document)))

    def test_a_duplicate_member_name_is_refused(self):
        document = artifact(members=("goal", "goal"))
        self.assertTrue(any("duplicate member names" in e for e in run(document)))

    def test_a_member_absent_from_the_canonical_bytes_is_refused(self):
        # The bytes commit to every name, so a name only in the table means the
        # document was edited rather than generated.
        document = artifact()
        document["families"][0]["members"][1]["name"] = "smuggled"
        self.assertTrue(
            any("not in the canonical bytes" in e for e in run(document)),
            run(document),
        )

    def test_a_missing_or_unhexed_body_is_refused(self):
        self.assertTrue(any("is missing" in e for e in mod.check(ROOT / "nope.json")))
        self.assertTrue(any("not hex" in e for e in run(artifact(canonical_bytes_hex="zz"))))
        self.assertTrue(any("unsupported schema" in e for e in run(artifact(schema="2"))))


class RealArtifactTests(unittest.TestCase):
    def test_the_committed_artifact_is_consistent(self):
        self.assertEqual(mod.check(ROOT / "datasets/generated/codebook.json"), [])


if __name__ == "__main__":
    unittest.main()


class TrackedTests(unittest.TestCase):
    """The artifact existing is not the same as the repository having it.

    Every other check here reads the working tree, so none of them can tell the
    difference — and the difference is the whole value of a committed contract.
    """

    def test_an_untracked_artifact_is_refused_when_that_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codebook.json"
            path.write_text(json.dumps(artifact()), encoding="utf-8")
            errors = mod.check(path, require_tracked=True)
            self.assertTrue(
                any("not tracked by git" in error for error in errors), errors
            )
            # And the same file passes every other check, so the refusal is about
            # tracking and nothing else.
            self.assertEqual(mod.check(path), [])

    def test_the_committed_artifact_is_tracked(self):
        # The regression itself: datasets/generated is ignored wholesale, so this
        # file needs its own exception and nothing else would notice its absence.
        self.assertTrue(mod.tracked(mod.ARTIFACT))
        self.assertEqual(mod.check(mod.ARTIFACT, require_tracked=True), [])
