import unittest
from pathlib import Path

from ptr_training.run import ROOT, build_manifest

class TrainingRunManifestTests(unittest.TestCase):
    def test_default_run_manifest_resolves_registered_dataset(self):
        manifest = build_manifest(ROOT / "training" / "configs" / "run-default.toml")
        self.assertEqual(manifest["dataset"]["name"], "typed_agent_behavior_v0_2")
        self.assertEqual(manifest["config"]["training"]["backend"], "dry-run")
