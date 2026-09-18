import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "run_component_eval", ROOT / "scripts/run_component_eval.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ComponentEvaluationTests(unittest.TestCase):
    def test_candidate_registry_validation(self):
        self.assertEqual(mod.validate(), 0)

    def test_evaluating_candidate_has_executable_command(self):
        _, _, candidate = mod.candidate("ledger", "raft-engine")
        command = mod.build_command(candidate)
        self.assertEqual(command[0], "cargo")
        self.assertIn("raft_engine", command)

    def test_missing_candidate_command_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no declared command"):
            mod.build_command({"id": "missing"})

    def test_execute_command_captures_process_evidence(self):
        result = mod.execute_command(
            [sys.executable, "-c", "print('eval-ok')"]
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), "eval-ok")
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(result["launch_error"])
        self.assertGreaterEqual(result["duration_ns"], 0)


if __name__ == "__main__":
    unittest.main()
