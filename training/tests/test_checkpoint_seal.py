"""A run that kept a checkpoint has to record the seal produced when it saved it.

These tests drive `require_seal` directly rather than through `ptrctl`, because
the python job builds no Rust. What the binary produces is proven in
`bins/ptrctl/tests/seal.rs`; what is proven here is that the manifest refuses a
run whose checkpoint is unbound, and refuses it one reason at a time.
"""

import hashlib
import tempfile
import tomllib
import unittest
from pathlib import Path

from ptr_training.checkpoint import SealError, require_seal

ANCHOR = """\
sealed = "model.sealed"
codebook = 1
journal_index = 2
journal_digest = "{digest}"
digest = "{digest}"
"""

DIGEST = "ab" * 32


class CheckpointSealTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        (self.root / "model.ckpt").write_bytes(b"weights")
        (self.root / "model.sealed").write_bytes(b"sealed weights")
        (self.root / "model.anchor.toml").write_text(
            ANCHOR.format(digest=DIGEST), encoding="utf-8"
        )

    def section(self, **overrides):
        base = {
            "artifact": "model.ckpt",
            "sealed": "model.sealed",
            "anchor": "model.anchor.toml",
        }
        base.update(overrides)
        return base

    def test_a_run_that_kept_no_checkpoint_declares_nothing_and_is_accepted(self):
        self.assertIsNone(require_seal({}, self.root, where="cfg [checkpoint]"))

    def test_a_complete_seal_is_recorded_with_the_anchors_own_fields(self):
        recorded = require_seal(self.section(), self.root, where="cfg [checkpoint]")
        self.assertEqual(recorded["codebook"], 1)
        self.assertEqual(recorded["journal_index"], 2)
        self.assertEqual(recorded["digest"], DIGEST)
        self.assertEqual(recorded["artifact"]["bytes"], len(b"weights"))
        self.assertEqual(recorded["sealed"]["bytes"], len(b"sealed weights"))

    def test_the_recorded_fingerprint_follows_the_bytes_on_disk(self):
        """The control.

        Every assertion above would hold just as well if the fingerprints were
        constants, so the observable has to be shown to move: change the artifact
        and the recorded digest changes with it.
        """
        before = require_seal(self.section(), self.root, where="cfg")["artifact"]["sha256"]
        (self.root / "model.ckpt").write_bytes(b"different weights")
        after = require_seal(self.section(), self.root, where="cfg")["artifact"]["sha256"]
        self.assertNotEqual(before, after)

    def test_a_checkpoint_with_no_sealed_artifact_is_refused(self):
        with self.assertRaises(SealError) as raised:
            require_seal(
                {"artifact": "model.ckpt"}, self.root, where="cfg [checkpoint]"
            )
        self.assertIn("no sealed artifact", str(raised.exception))

    def test_a_sealed_artifact_with_no_anchor_is_refused(self):
        with self.assertRaises(SealError) as raised:
            require_seal(
                {"artifact": "model.ckpt", "sealed": "model.sealed"},
                self.root,
                where="cfg [checkpoint]",
            )
        self.assertIn("no anchor", str(raised.exception))

    def test_a_section_naming_no_artifact_is_refused(self):
        with self.assertRaises(SealError):
            require_seal({"sealed": "model.sealed"}, self.root, where="cfg")

    def test_a_declared_file_that_is_not_there_is_refused_by_name(self):
        with self.assertRaises(SealError) as raised:
            require_seal(
                self.section(sealed="absent.sealed"), self.root, where="cfg"
            )
        self.assertIn("absent.sealed", str(raised.exception))

    def test_an_anchor_describing_a_different_artifact_is_refused(self):
        """Without this, a run could retain some other state's anchor intact.

        Every field would parse, every file would exist, and the manifest would
        record a seal that says nothing about the checkpoint beside it.
        """
        (self.root / "other.sealed").write_bytes(b"another sealed state")
        (self.root / "other.anchor.toml").write_text(
            ANCHOR.format(digest=DIGEST).replace(
                'sealed = "model.sealed"', 'sealed = "other.sealed"'
            ),
            encoding="utf-8",
        )
        with self.assertRaises(SealError) as raised:
            require_seal(
                self.section(anchor="other.anchor.toml"), self.root, where="cfg"
            )
        self.assertIn("describes other.sealed", str(raised.exception))

    def test_an_anchor_missing_a_field_is_refused_by_field(self):
        (self.root / "short.anchor.toml").write_text(
            'sealed = "model.sealed"\ncodebook = 1\n', encoding="utf-8"
        )
        with self.assertRaises(SealError) as raised:
            require_seal(
                self.section(anchor="short.anchor.toml"), self.root, where="cfg"
            )
        self.assertIn("journal_index", str(raised.exception))

    def test_an_unparsable_anchor_is_refused_rather_than_ignored(self):
        (self.root / "bad.anchor.toml").write_text("not = = toml", encoding="utf-8")
        with self.assertRaises(SealError):
            require_seal(self.section(anchor="bad.anchor.toml"), self.root, where="cfg")


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bound-a0"


class BoundSaveFixtureTests(unittest.TestCase):
    """A save the Rust side actually produced, not files this test wrote.

    `fixtures/bound-a0` is written by `model/burn-a0/bridge/examples/bound_checkpoint.rs`,
    a trainer that seals its checkpoint as it saves it through
    `ptr_a0_bridge::save_bound_checkpoint`. The tests above prove the refusals on
    files they compose; this proves the pipeline accepts what a sealed save really
    looks like, so the two sides agree about the format and not only about the idea.
    """

    def section(self):
        return {
            "artifact": "model.ckpt",
            "sealed": "model.sealed",
            "anchor": "model.anchor.toml",
        }

    def test_a_save_sealed_as_it_was_written_is_accepted(self):
        recorded = require_seal(self.section(), FIXTURE, where="fixture [checkpoint]")
        self.assertEqual(recorded["codebook"], 1)
        sealed = (FIXTURE / "model.sealed").read_bytes()
        self.assertEqual(recorded["sealed"]["sha256"], hashlib.sha256(sealed).hexdigest())

    def test_the_anchor_describes_these_sealed_bytes(self):
        """What `require_seal` does not itself check, checked once against a real save.

        A sealed state ends with the SHA-256 of everything before it, and the
        anchor retains that digest. Holding here is what makes the accepted
        fixture above a *matching* pair rather than two files that merely parse.
        """
        sealed = (FIXTURE / "model.sealed").read_bytes()
        anchor = tomllib.loads((FIXTURE / "model.anchor.toml").read_text(encoding="utf-8"))
        self.assertEqual(sealed[:8], b"PTRNEU01")
        self.assertEqual(hashlib.sha256(sealed[:-32]).hexdigest(), sealed[-32:].hex())
        self.assertEqual(anchor["digest"], sealed[-32:].hex())

    def test_the_checkpoint_alone_is_refused_without_its_seal(self):
        """The control: the same real checkpoint, declared with no seal."""
        with self.assertRaisesRegex(SealError, "never bound"):
            require_seal({"artifact": "model.ckpt"}, FIXTURE, where="fixture [checkpoint]")


if __name__ == "__main__":
    unittest.main()
