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
    def test_successor_is_prepared_with_its_own_matched_pair(self):
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual((manifest["id"], manifest["status"]), ("M002-v8", "prepared"))
        self.assertIn("--experiment M002-v8", manifest["entrypoint"])
        self.assertIn("--arms factorized-v2-v8,factorized-v2-off-v8", manifest["entrypoint"])

    def test_command_runner_and_complete_implementation_tree_are_bound(self):
        gates = load_module("m002_v8_gate", ROOT / "scripts/check_research_gates.py")
        config = tomllib.loads((EXPERIMENT / "config.toml").read_text(encoding="utf-8"))["preregistration"]
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual(gates.m002_v8_fold_binding_errors(ROOT, config), [])
        self.assertEqual(
            gates.m002_versioned_runner_binding_errors(
                ROOT, config, manifest, "M002-v8", "factorized-v2-v8,factorized-v2-off-v8"
            ),
            [],
        )
        invalid = dict(config)
        invalid["implementation_tree_git_digest"] = "0" * 64
        self.assertTrue(
            gates.m002_versioned_runner_binding_errors(
                ROOT, invalid, manifest, "M002-v8", "factorized-v2-v8,factorized-v2-off-v8"
            )
        )

    def test_successor_freeze_accepts_only_the_immutable_pilot_archive(self):
        gates = load_module("m002_v8_successor_gate", ROOT / "scripts/check_research_gates.py")
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual(gates.m002_v8_successor_freeze_errors(ROOT, EXPERIMENT, manifest), [])

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
