import copy
import importlib.util
import unittest
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/model/M002-v8-factorized-typed-attention"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def v8_labels(stdout):
    return (
        stdout.replace("factorized-v2-off", "__CONTROL__")
        .replace("factorized-v2", "factorized-v2-v8")
        .replace("__CONTROL__", "factorized-v2-off-v8")
    )


class M002V8ContractTests(unittest.TestCase):
    def test_successor_is_superseded_without_execution(self):
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual((manifest["id"], manifest["status"]), ("M002-v8", "superseded"))
        self.assertIn("--experiment M002-v8", manifest["entrypoint"])
        self.assertIn("--arms factorized-v2-v8,factorized-v2-off-v8", manifest["entrypoint"])

    def test_superseded_study_has_no_runner_records(self):
        self.assertEqual(list((EXPERIMENT / "results").glob("run-*.json")), [])

    def test_adapter_restores_v8_identity_and_rejects_unknown_labels(self):
        v5_tests = load_module("m002_v5_fixture", ROOT / "scripts/tests/test_aggregate_m002_v5.py")
        adapter = load_module("aggregate_m002_v8", ROOT / "scripts/aggregate_m002_v8.py")
        core = adapter.load_core()
        records = []
        for name, record in v5_tests.records():
            clone = copy.deepcopy(record)
            clone["experiment_id"] = "M002-v8"
            clone["manifest"]["id"] = "M002-v8"
            clone["stdout"] = v8_labels(clone["stdout"])
            records.append((name, clone))
        decision = adapter.decide(records, protocol=core.fixture_protocol())
        self.assertEqual(decision["experiment_id"], "M002-v8")
        malformed = copy.deepcopy(records)
        malformed[0][1]["stdout"] = malformed[0][1]["stdout"].replace(
            "factorized-v2-v8", "factorized-v2-v8-unknown", 1
        )
        with self.assertRaisesRegex(adapter.EvidenceError, "unrecognized v8 arm label"):
            adapter.decide(malformed, protocol=core.fixture_protocol())

    def test_predecessors_are_superseded_and_do_not_unlock_m009(self):
        v6 = tomllib.loads(
            (ROOT / "experiments/model/M002-v6-factorized-typed-attention/experiment.toml").read_text(encoding="utf-8")
        )
        v7 = tomllib.loads(
            (ROOT / "experiments/model/M002-v7-factorized-typed-attention/experiment.toml").read_text(encoding="utf-8")
        )
        plan = tomllib.loads((EXPERIMENT / "PLAN.toml").read_text(encoding="utf-8"))
        self.assertEqual((v6["status"], v7["status"]), ("superseded", "superseded"))
        self.assertEqual(plan["supersedes"]["experiment_id"], "M002-v7")
        self.assertIn("separately preregistered M009", plan["downstream"]["unlock_rule"])


if __name__ == "__main__":
    unittest.main()
