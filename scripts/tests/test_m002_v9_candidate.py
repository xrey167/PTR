import copy
import importlib.util
import unittest
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "experiments/model/M002-v9-factorized-typed-attention"
MANIFEST = {
    "id": "M002-v9",
    "status": "prepared",
    "entrypoint": (
        "cargo run -- --phase paired-v5 --experiment M002-v9 "
        "--arms factorized-v2-v9,factorized-v2-off-v9"
    ),
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def v9_labels(stdout):
    return (
        stdout.replace("factorized-v2-off", "__CONTROL__")
        .replace("factorized-v2", "factorized-v2-v9")
        .replace("__CONTROL__", "factorized-v2-off-v9")
    )


class M002V9CandidateTests(unittest.TestCase):
    def test_committed_unregistered_candidate_passes_complete_freeze_gate(self):
        gates = load_module("m002_v9_gate", ROOT / "scripts/check_research_gates.py")
        self.assertEqual(gates.m002_v9_successor_freeze_errors(ROOT, CANDIDATE, MANIFEST), [])

    def test_runner_tree_digest_and_cross_study_pair_are_hard_requirements(self):
        gates = load_module("m002_v9_runner_gate", ROOT / "scripts/check_research_gates.py")
        config = tomllib.loads((CANDIDATE / "config.toml").read_text(encoding="utf-8"))["preregistration"]
        self.assertEqual(gates.m002_v9_fold_binding_errors(ROOT, config), [])
        self.assertEqual(
            gates.m002_versioned_runner_binding_errors(
                ROOT, config, MANIFEST, "M002-v9", "factorized-v2-v9,factorized-v2-off-v9"
            ),
            [],
        )
        invalid = dict(config)
        invalid["arm_pair"] = "factorized-v2-v8,factorized-v2-off-v8"
        self.assertTrue(
            gates.m002_versioned_runner_binding_errors(
                ROOT, invalid, MANIFEST, "M002-v9", "factorized-v2-v9,factorized-v2-off-v9"
            )
        )
        invalid = dict(config)
        invalid["implementation_tree_git_digest"] = "0" * 64
        self.assertTrue(
            gates.m002_versioned_runner_binding_errors(
                ROOT, invalid, MANIFEST, "M002-v9", "factorized-v2-v9,factorized-v2-off-v9"
            )
        )

    def test_adapter_restores_v9_identity_and_rejects_unknown_labels(self):
        v5_tests = load_module("m002_v5_fixture", ROOT / "scripts/tests/test_aggregate_m002_v5.py")
        adapter = load_module("aggregate_m002_v9", ROOT / "scripts/aggregate_m002_v9.py")
        core = adapter.load_core()
        records = []
        for name, record in v5_tests.records():
            clone = copy.deepcopy(record)
            clone["experiment_id"] = "M002-v9"
            clone["manifest"]["id"] = "M002-v9"
            clone["stdout"] = v9_labels(clone["stdout"])
            records.append((name, clone))
        decision = adapter.decide(records, protocol=core.fixture_protocol())
        self.assertEqual(decision["experiment_id"], "M002-v9")
        malformed = copy.deepcopy(records)
        malformed[0][1]["stdout"] = malformed[0][1]["stdout"].replace(
            "factorized-v2-v9", "factorized-v2-v9-unknown", 1
        )
        with self.assertRaisesRegex(adapter.EvidenceError, "unrecognized v9 arm label"):
            adapter.decide(malformed, protocol=core.fixture_protocol())


if __name__ == "__main__":
    unittest.main()
