import importlib.util
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("run_component_eval",ROOT/"scripts/run_component_eval.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class ComponentEvaluationTests(unittest.TestCase):
    def test_candidate_registry_validation(self):
        self.assertEqual(mod.validate(),0)
