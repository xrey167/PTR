import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from ptr_training import codebook as codebook_artifact
from ptr_training import preflight
from ptr_training import run as run_module
from ptr_training.moe.pipeline import stack_available
from ptr_training.run import (
    ROOT,
    build_manifest,
    verify_declared_file,
    write_manifest,
)

DEFAULT_CONFIG = ROOT / "training" / "configs" / "run-default.toml"


def _config_text(**codebook) -> str:
    """The default run config with its [codebook] section replaced."""
    original = DEFAULT_CONFIG.read_text(encoding="utf-8")
    body = original[original.index("[run]") :]
    lines = ["version = 1", ""]
    if codebook:
        lines.append("[codebook]")
        for key, value in codebook.items():
            rendered = f'"{value}"' if isinstance(value, str) else str(value)
            lines.append(f"{key} = {rendered}")
        lines.append("")
    return "\n".join(lines) + body


class TrainingRunManifestTests(unittest.TestCase):
    def test_default_run_manifest_verifies_registered_dataset_bytes(self):
        manifest = build_manifest(ROOT / "training" / "configs" / "run-default.toml")
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(manifest["dataset"]["name"], "typed_agent_behavior_v0_2")
        self.assertEqual(manifest["config"]["training"]["backend"], "dry-run")
        self.assertEqual(
            manifest["dataset_artifact"]["sha256"],
            manifest["dataset"]["sha256"],
        )
        self.assertEqual(
            manifest["dataset_artifact"]["bytes"],
            manifest["dataset"]["bytes"],
        )

    def test_manifest_hashes_model_hardware_config_and_locks(self):
        manifest = build_manifest(ROOT / "training" / "configs" / "run-default.toml")
        self.assertEqual(len(manifest["config_sha256"]), 64)
        self.assertEqual(len(manifest["model_config"]["sha256"]), 64)
        self.assertEqual(len(manifest["hardware_profile"]["sha256"]), 64)
        self.assertEqual(len(manifest["cargo_lock_sha256"]), 64)
        self.assertEqual(len(manifest["uv_lock_sha256"]), 64)
        self.assertEqual(manifest["provenance"]["a0_manifest"]["path"], "model/burn-a0/Cargo.toml")
        self.assertEqual(manifest["provenance"]["a0_cargo_lock"]["path"], "model/burn-a0/Cargo.lock")
        self.assertEqual(len(manifest["provenance"]["a0_cargo_lock"]["sha256"]), 64)
        self.assertEqual(len(manifest["input_fingerprint_sha256"]), 64)

    def test_input_fingerprint_is_stable_for_identical_inputs(self):
        config = ROOT / "training" / "configs" / "run-default.toml"
        first = build_manifest(config)
        second = build_manifest(config)
        self.assertEqual(
            first["input_fingerprint_sha256"],
            second["input_fingerprint_sha256"],
        )

    def test_manifest_records_the_codebook_the_run_is_interpretable_against(self):
        book = codebook_artifact.load()
        manifest = build_manifest(DEFAULT_CONFIG)
        recorded = manifest["codebook"]
        self.assertEqual(recorded["version"], book["version"])
        self.assertEqual(recorded["fingerprint_sha256"], book["fingerprint_sha256"])
        self.assertEqual(
            recorded["artifact"]["path"], "datasets/generated/codebook.json"
        )
        self.assertEqual(len(recorded["artifact"]["sha256"]), 64)
        # In provenance too, so the artifact's own bytes reach the input
        # fingerprint rather than only being reported.
        self.assertEqual(manifest["provenance"]["codebook"], recorded)

    def test_the_codebook_artifact_reaches_the_input_fingerprint(self):
        manifest = build_manifest(DEFAULT_CONFIG)
        from ptr_training.run import input_fingerprint

        provenance = dict(manifest["provenance"])
        moved = dict(provenance["codebook"])
        moved["fingerprint_sha256"] = "0" * 64
        provenance["codebook"] = moved
        self.assertNotEqual(
            input_fingerprint(provenance), manifest["input_fingerprint_sha256"]
        )

    def _refused(self, text: str, message: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "run.toml"
            config.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                build_manifest(config)

    def test_a_run_that_names_no_codebook_is_refused(self):
        self._refused(_config_text(), "version is required")

    def test_a_run_naming_only_a_version_is_refused(self):
        book = codebook_artifact.load()
        self._refused(
            _config_text(version=book["version"]), "fingerprint is required"
        )

    def test_an_unknown_codebook_version_is_refused_with_its_own_reason(self):
        book = codebook_artifact.load()
        self._refused(
            _config_text(version=99, fingerprint=book["fingerprint_sha256"]),
            "is not this build",
        )

    def test_a_moved_assignment_is_refused_with_a_different_reason(self):
        # The version is right and the assignment behind it has moved: the case a
        # version number cannot catch, and it must not read as "unknown version".
        book = codebook_artifact.load()
        self._refused(
            _config_text(version=book["version"], fingerprint="0" * 64),
            "does not match this build",
        )

    def test_the_default_config_is_what_these_negatives_vary(self):
        # Without this, a change to the config's shape could make every negative
        # above pass for the wrong reason.
        cfg = tomllib.loads(_config_text(**{
            "version": codebook_artifact.load()["version"],
            "fingerprint": codebook_artifact.load()["fingerprint_sha256"],
        }))
        declared = tomllib.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(cfg, declared)

    def test_declared_dataset_hash_mismatch_fails_closed(self):
        dataset = ROOT / "datasets" / "bundles" / "typed_agent_behavior_v0_2.zip"
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            verify_declared_file(dataset, expected_sha256="0" * 64)

    def test_manifest_write_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            write_manifest(out, {"status": "prepared"})
            with self.assertRaises(FileExistsError):
                write_manifest(out, {"status": "replacement"})


STAGE = ROOT / "training" / "configs" / "stage-moe-smoke.toml"
HAVE_STACK = stack_available(("torch", "trl", "transformers", "datasets", "peft"))


def _preflight_record(manifest: dict, **overrides) -> dict:
    fields = {
        "protocol_sha256": manifest["config_sha256"],
        "runner_sha256": preflight.runner_sha256(),
        "device": "cpu",
        "input_fingerprint_sha256": "i" * 64,
        "probe": {},
    }
    fields.update(overrides)
    return preflight.build_record(**fields)


class PreflightGateTests(unittest.TestCase):
    """A run without a matching preflight record is refused, fail closed."""

    manifest = {"config_sha256": "c" * 64}

    def gate(self, directory, **overrides):
        arguments = {"device": "cpu", "evidence_class": "pipeline-smoke", "allow_missing": False}
        arguments.update(overrides)
        return run_module.require_preflight_for(self.manifest, directory=directory, **arguments)

    def test_a_run_with_no_preflight_record_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as caught:
            self.gate(Path(tmp))
        self.assertIn("refusing training execution", str(caught.exception))
        self.assertIn("no preflight record matches", str(caught.exception))

    def test_a_matching_record_lets_the_run_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = _preflight_record(self.manifest)
            preflight.write_record(Path(tmp), record)
            self.assertEqual(self.gate(Path(tmp)), record)

    def test_a_record_for_another_config_is_a_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            preflight.write_record(Path(tmp), _preflight_record(self.manifest, protocol_sha256="d" * 64))
            with self.assertRaises(SystemExit) as caught:
                self.gate(Path(tmp))
        self.assertIn("protocol_sha256", str(caught.exception))

    def test_a_record_for_another_runner_version_is_a_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            preflight.write_record(Path(tmp), _preflight_record(self.manifest, runner_sha256="e" * 64))
            with self.assertRaises(SystemExit) as caught:
                self.gate(Path(tmp))
        self.assertIn("runner_sha256", str(caught.exception))

    def test_a_record_for_another_device_is_a_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            preflight.write_record(Path(tmp), _preflight_record(self.manifest))
            with self.assertRaises(SystemExit) as caught:
                self.gate(Path(tmp), device="cuda:NVIDIA RTX 3090")
        self.assertIn("device", str(caught.exception))

    def test_the_waiver_covers_pipeline_smoke_runs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self.gate(Path(tmp), allow_missing=True))
            with self.assertRaises(SystemExit) as caught:
                self.gate(Path(tmp), allow_missing=True, evidence_class="experiment")
        self.assertIn("only valid for evidence_class", str(caught.exception))


class PreflightCliTests(unittest.TestCase):
    def main_with(self, *flags):
        manifest = {
            "git_dirty": False,
            "config_sha256": "c" * 64,
            "config": {"training": {"backend": "trl-moe", "allow_unselected_backend": True}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["run", "--config", str(STAGE), "--out", str(Path(tmp) / "m.json"), "--execute", *flags]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run_module, "build_manifest", return_value=manifest
            ), mock.patch.object(run_module, "ROOT", Path(tmp)), mock.patch.object(
                run_module, "_execute_trl_moe", return_value=0
            ) as execute:
                with self.assertRaises(SystemExit):
                    run_module.main()
        return execute

    def test_the_waiver_flag_reaches_the_executor_only_when_given(self):
        self.assertFalse(self.main_with().call_args.kwargs["allow_missing_preflight"])
        self.assertTrue(
            self.main_with("--allow-missing-preflight").call_args.kwargs["allow_missing_preflight"]
        )


@unittest.skipUnless(HAVE_STACK, "torch/trl/transformers/datasets/peft not installed (moe extra)")
class PreflightExecutorTests(unittest.TestCase):
    def test_the_executor_refuses_before_any_training_without_a_record(self):
        from ptr_training.moe import pipeline

        manifest = {"config_sha256": preflight.file_sha256(STAGE)}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            preflight, "PREFLIGHT_DIR", Path(tmp)
        ), mock.patch.object(pipeline, "run", side_effect=AssertionError("ran")) as ran:
            with self.assertRaises(SystemExit) as caught:
                run_module._execute_trl_moe(manifest, STAGE)
        ran.assert_not_called()
        self.assertIn("no preflight record matches", str(caught.exception))

    def test_a_waived_smoke_run_proceeds_and_says_so(self):
        from ptr_training.moe import pipeline

        manifest = {"config_sha256": preflight.file_sha256(STAGE)}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            preflight, "PREFLIGHT_DIR", Path(tmp)
        ), mock.patch.object(pipeline, "run", return_value=Path(tmp) / "metrics.json"):
            self.assertEqual(
                run_module._execute_trl_moe(manifest, STAGE, allow_missing_preflight=True), 0
            )
        self.assertEqual(manifest["preflight"]["status"], "waived")


if __name__ == "__main__":
    unittest.main()
