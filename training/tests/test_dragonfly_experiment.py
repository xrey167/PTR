import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from ptr_training.dragonfly import experiment
from ptr_training.dragonfly.experiment import EncoderUnavailable, main, pinned_encoder


def _config(directory: Path, model: str, revision: str) -> Path:
    path = directory / "config.toml"
    path.write_text(
        f'[preregistration]\nencoder_model = "{model}"\nencoder_revision = "{revision}"\n',
        encoding="utf-8",
    )
    return path


class PinnedEncoderTest(unittest.TestCase):
    def test_a_placeholder_pin_never_yields_an_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = _config(Path(raw), "must-be-pinned-before-prepared", "must-be-pinned-x")
            with self.assertRaises(EncoderUnavailable):
                pinned_encoder(config)

    def test_a_missing_pin_never_yields_an_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(EncoderUnavailable):
                pinned_encoder(_config(Path(raw), "", ""))

    @unittest.skipIf(
        importlib.util.find_spec("sentence_transformers") is not None,
        "the optional extra is installed; this test needs it absent",
    )
    def test_a_pin_whose_extra_is_not_installed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = _config(Path(raw), "sentence-transformers/all-MiniLM-L6-v2", "abc123")
            with self.assertRaises(EncoderUnavailable):
                pinned_encoder(config)

    def test_the_committed_config_is_still_unpinned(self) -> None:
        # R015 is `planned`: until the freeze pins the encoder, a confirmatory run must stop.
        with self.assertRaises(EncoderUnavailable):
            pinned_encoder()


class EntrypointTest(unittest.TestCase):
    def test_without_smoke_an_unpinned_encoder_stops_the_run_with_status_2(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "world"
            config = _config(Path(raw), "must-be-pinned-before-prepared", "must-be-pinned-x")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(["--seed", "1001", "--out", str(out), "--config", str(config)])
            self.assertEqual(status, 2)
            self.assertFalse(out.exists(), "a refused run writes no world")
            self.assertIn("not pinned", stderr.getvalue())

    def test_smoke_marks_the_world_as_no_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "world"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--seed", "1001", "--out", str(out), "--smoke", "--steps", "5"])
            self.assertEqual(status, 0)
            world = json.loads((out / "world.json").read_text(encoding="utf-8"))
            self.assertTrue(world["smoke"])
            self.assertEqual(world["evidence"], "none (stand-in encoder)")
            self.assertEqual(world["project"], experiment.PROJECT)
            self.assertEqual(world["encoder"]["id"], "hashed-bow-64")

    def test_build_marks_a_non_smoke_world_as_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            encoder = experiment.HashedBagOfWordsEncoder(experiment.ENCODER_DIMS)
            world = experiment.build(1001, Path(raw) / "w", encoder, steps=5, smoke=False)
            self.assertFalse(world["smoke"])
            self.assertEqual(world["evidence"], "candidate")


if __name__ == "__main__":
    unittest.main()
