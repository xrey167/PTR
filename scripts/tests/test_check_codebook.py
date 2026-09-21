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
        "exceptions": [
            {
                "name": "provenance_bucket_count",
                "width": 64,
                "reason": "Research-local bucketing with no members to assign codes to.",
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


class ExceptionTests(unittest.TestCase):
    """A width the taxonomy deliberately omits, recorded so it is a decision.

    The point of the section is that an unrecorded exception and an oversight look
    exactly alike. So the checker's job is to refuse a record that does not say
    enough to tell them apart, and to refuse one that contradicts the taxonomy it
    is an exception to.
    """

    def test_a_well_formed_exception_passes(self):
        # The control. Without it the refusals below could all be failing for a
        # reason that has nothing to do with what each one names.
        self.assertEqual(run(artifact()), [])

    def test_an_absent_section_is_refused_and_an_empty_one_is_not(self):
        document = artifact()
        del document["exceptions"]
        errors = run(document)
        self.assertTrue(any("exceptions is absent" in e for e in errors), errors)

        # A kernel with no exceptions says so with an empty list; silence is what
        # cannot be distinguished from an oversight.
        document = artifact()
        document["exceptions"] = []
        self.assertEqual(run(document), [])

    def test_an_exception_without_a_reason_is_refused(self):
        document = artifact()
        document["exceptions"][0]["reason"] = "   "
        errors = run(document)
        self.assertTrue(any("reads as an oversight" in e for e in errors), errors)

    def test_a_width_that_is_not_a_positive_integer_is_refused(self):
        for width in (0, -1, "64", 64.0, None, True):
            with self.subTest(width=width):
                document = artifact()
                document["exceptions"][0]["width"] = width
                errors = run(document)
                self.assertTrue(
                    any("is not a positive integer" in e for e in errors),
                    f"{width!r}: {errors}",
                )

    def test_a_name_that_is_also_a_family_is_refused(self):
        # It cannot be inside and outside the taxonomy at once.
        document = artifact()
        document["exceptions"][0]["name"] = "semantic_role"
        errors = run(document)
        self.assertTrue(any("as an exception and as a family" in e for e in errors), errors)

    def test_a_name_inside_the_canonical_bytes_is_refused(self):
        # The fingerprint commits to an assignment of codes. An exception assigns
        # none, so folding one in would move the fingerprint of a version whose
        # codes had not moved, invalidating every artifact bound to it.
        document = artifact(members=("goal", "provenance_bucket_count"))
        errors = run(document)
        self.assertTrue(any("appears in the canonical bytes" in e for e in errors), errors)

    def test_the_same_exception_twice_is_refused(self):
        document = artifact()
        document["exceptions"].append(dict(document["exceptions"][0]))
        errors = run(document)
        self.assertTrue(any("recorded twice" in e for e in errors), errors)

    def test_an_exception_with_no_name_is_refused(self):
        document = artifact()
        document["exceptions"][0]["name"] = ""
        errors = run(document)
        self.assertTrue(any("has no name" in e for e in errors), errors)


class ThisArtifactTests(unittest.TestCase):
    def test_the_committed_artifact_records_the_provenance_exception(self):
        document = json.loads(
            (ROOT / "datasets/generated/codebook.json").read_text(encoding="utf-8")
        )
        exceptions = {e["name"]: e for e in document["exceptions"]}
        self.assertIn("provenance_bucket_count", exceptions)
        self.assertEqual(exceptions["provenance_bucket_count"]["width"], 64)
        self.assertTrue(exceptions["provenance_bucket_count"]["reason"].strip())
