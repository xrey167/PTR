import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("run_experiment",ROOT/"scripts/run_experiment.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class ExperimentRunnerTests(unittest.TestCase):
    def test_registry_resolves_known_experiment(self):
        item,root,data=mod.resolve("M001")
        self.assertEqual(data["id"],"M001")
        self.assertTrue(root.exists())
        self.assertEqual(item["status"],data["status"])
