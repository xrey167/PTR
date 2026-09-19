import tempfile
import unittest
from pathlib import Path

from ptr_training.run import (
    ROOT,
    build_manifest,
    verify_declared_file,
    write_manifest,
)


class TrainingRunManifestTests(unittest.TestCase):
    def test_default_run_manifest_verifies_registered_dataset_bytes(self):
        manifest = build_manifest(ROOT / "training" / "configs" / "run-default.toml")
        self.assertEqual(manifest["schema_version"], 2)
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
        self.assertEqual(len(manifest["input_fingerprint_sha256"]), 64)

    def test_input_fingerprint_is_stable_for_identical_inputs(self):
        config = ROOT / "training" / "configs" / "run-default.toml"
        first = build_manifest(config)
        second = build_manifest(config)
        self.assertEqual(
            first["input_fingerprint_sha256"],
            second["input_fingerprint_sha256"],
        )

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


if __name__ == "__main__":
    unittest.main()
