import contextlib
import importlib.metadata
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptr_training.dragonfly import experiment
from ptr_training.dragonfly.experiment import EncoderUnavailable, main, pinned_encoder


def _running_python() -> str:
    return "{}.{}.{}".format(*sys.version_info[:3])


def _config(directory: Path, model: str, revision: str, python: str | None = None) -> Path:
    path = directory / "config.toml"
    path.write_text(
        f'[preregistration]\nencoder_model = "{model}"\nencoder_revision = "{revision}"\n'
        f'python_version = "{python or _running_python()}"\n',
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


def _lock(directory: Path, versions: dict[str, str]) -> Path:
    path = directory / "uv.lock"
    body = "version = 1\n" + "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n'
        for name, version in versions.items()
    )
    path.write_text(body, encoding="utf-8")
    return path


class PythonPinTest(unittest.TestCase):
    def test_only_the_exact_pinned_version_matches(self) -> None:
        version = (3, 13, 16)
        self.assertTrue(experiment.python_matches("3.13.16", version))
        for pin in ("3.13.15", "3.13", "3.13.16.1", "3.12.16", "3.13.*", ">=3.13", "", None,
                    "must-be-pinned-at-freeze"):
            self.assertFalse(experiment.python_matches(pin, version), pin)

    def test_a_placeholder_or_missing_pin_is_no_pin(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = _config(Path(raw), "m", "r", python="must-be-pinned-at-freeze")
            self.assertIsNone(experiment.pinned_python(config))
            self.assertEqual(experiment.pinned_python(_config(Path(raw), "m", "r")), _running_python())

    def test_a_confirmatory_run_on_another_interpreter_stops_with_status_2(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "world"
            config = _config(Path(raw), "m", "r", python="2.7.18")
            standin = experiment.HashedBagOfWordsEncoder(experiment.ENCODER_DIMS)
            stderr = io.StringIO()
            with (
                mock.patch.object(experiment, "pinned_encoder", return_value=standin),
                contextlib.redirect_stderr(stderr),
            ):
                status = main(["--seed", "1001", "--out", str(out), "--config", str(config)])
            self.assertEqual(status, 2)
            self.assertFalse(out.exists())
            self.assertIn("pins 2.7.18", stderr.getvalue())

    def test_a_smoke_run_only_records_the_interpreter_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "world"
            config = _config(Path(raw), "m", "r", python="2.7.18")
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    ["--seed", "1001", "--out", str(out), "--smoke", "--steps", "5",
                     "--config", str(config)]
                )
            self.assertEqual(status, 0)
            environment = json.loads((out / "world.json").read_text(encoding="utf-8"))["environment"]
            self.assertEqual(environment["python_pinned"], "2.7.18")
            self.assertFalse(environment["python_matches"])


class EnvironmentTest(unittest.TestCase):
    PACKAGES = ("alpha", "beta")

    def _report(self, lock: Path, installed: dict[str, str]) -> dict:
        def version(name: str) -> str:
            if name not in installed:
                raise importlib.metadata.PackageNotFoundError(name)
            return installed[name]

        with mock.patch.object(experiment.importlib.metadata, "version", version):
            return experiment.environment_report(lock, self.PACKAGES)

    def test_a_matching_environment_is_reported_with_the_lock_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lock = _lock(Path(raw), {"alpha": "1.0", "beta": "2.0"})
            report = self._report(lock, {"alpha": "1.0", "beta": "2.0"})
            self.assertTrue(report["matches_lock"])
            self.assertEqual(report["mismatches"], [])
            self.assertEqual(report["packages"]["alpha"], {"installed": "1.0", "locked": ["1.0"]})
            self.assertEqual(len(report["uv_lock_sha256"]), 64)
            self.assertIn("python", report)

    def test_a_different_or_missing_package_is_a_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lock = _lock(Path(raw), {"alpha": "1.0", "beta": "2.0"})
            report = self._report(lock, {"alpha": "1.1"})
            self.assertFalse(report["matches_lock"])
            self.assertEqual(report["mismatches"], ["alpha", "beta"])

    def test_a_package_the_lock_does_not_name_is_a_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lock = _lock(Path(raw), {"alpha": "1.0"})
            report = self._report(lock, {"alpha": "1.0", "beta": "2.0"})
            self.assertEqual(report["mismatches"], ["beta"])

    def test_the_repository_lock_names_the_packages_a_run_depends_on(self) -> None:
        report = experiment.environment_report()
        for name in experiment.LOCKED_PACKAGES:
            self.assertTrue(report["packages"][name]["locked"], name)

    def _pinned(self, directory: Path) -> Path:
        return _config(directory, "sentence-transformers/all-MiniLM-L6-v2", "abc123")

    def test_a_confirmatory_run_stops_with_status_2_when_the_environment_differs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "world"
            lock = _lock(Path(raw), {"sentence-transformers": "0.0.1"})
            stderr = io.StringIO()
            standin = experiment.HashedBagOfWordsEncoder(experiment.ENCODER_DIMS)
            with (
                mock.patch.object(experiment, "pinned_encoder", return_value=standin),
                contextlib.redirect_stderr(stderr),
            ):
                status = main(
                    [
                        "--seed", "1001", "--out", str(out),
                        "--config", str(self._pinned(Path(raw))), "--lock", str(lock),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(out.exists())
            self.assertIn("differ from uv.lock", stderr.getvalue())

    def test_a_smoke_run_only_records_a_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "world"
            lock = _lock(Path(raw), {"sentence-transformers": "0.0.1"})
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "--seed", "1001", "--out", str(out), "--smoke", "--steps", "5",
                        "--lock", str(lock),
                    ]
                )
            self.assertEqual(status, 0)
            world = json.loads((out / "world.json").read_text(encoding="utf-8"))
            self.assertFalse(world["environment"]["matches_lock"])
            self.assertIn("sentence-transformers", world["environment"]["mismatches"])
            self.assertEqual(len(world["environment"]["uv_lock_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
