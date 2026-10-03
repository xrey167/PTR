import copy
import importlib.util
import json
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/model/M002-v7-factorized-typed-attention"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def v7_labels(stdout):
    return (
        stdout.replace("factorized-v2-off", "__CONTROL__")
        .replace("factorized-v2", "factorized-v2-v7")
        .replace("__CONTROL__", "factorized-v2-off-v7")
    )


class M002V7ContractTests(unittest.TestCase):
    def test_successor_is_superseded_without_execution(self):
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual((manifest["id"], manifest["status"]), ("M002-v7", "superseded"))
        self.assertIn("--experiment M002-v7", manifest["entrypoint"])
        self.assertIn("--arms factorized-v2-v7,factorized-v2-off-v7", manifest["entrypoint"])

    def test_superseded_study_has_no_runner_records(self):
        self.assertEqual(list((EXPERIMENT / "results").glob("run-*.json")), [])

    def test_adapter_restores_v7_identity_and_maps_only_versioned_labels(self):
        v5_tests = load_module("m002_v5_fixture", ROOT / "scripts/tests/test_aggregate_m002_v5.py")
        adapter = load_module("aggregate_m002_v7", ROOT / "scripts/aggregate_m002_v7.py")
        core = adapter.load_core()
        records = []
        for name, record in v5_tests.records():
            clone = copy.deepcopy(record)
            clone["experiment_id"] = "M002-v7"
            clone["manifest"]["id"] = "M002-v7"
            clone["stdout"] = v7_labels(clone["stdout"])
            records.append((name, clone))
        decision = adapter.decide(records, protocol=core.fixture_protocol())
        self.assertEqual(decision["experiment_id"], "M002-v7")
        self.assertEqual(
            [item["canonical_sha256"] for item in decision["provenance"]["source_records"]],
            [
                adapter.hashlib.sha256(adapter.canonical(record).encode("utf-8")).hexdigest()
                for _, record in sorted(records)
            ],
        )
        malformed = copy.deepcopy(records)
        malformed[0][1]["stdout"] = malformed[0][1]["stdout"].replace(
            "factorized-v2-v7", "factorized-v2-v7-unknown", 1
        )
        with self.assertRaisesRegex(adapter.EvidenceError, "unrecognized v7 arm label"):
            adapter.decide(malformed, protocol=core.fixture_protocol())

    def test_v6_and_v7_remain_superseded_and_do_not_unlock_downstream(self):
        v6 = tomllib.loads(
            (ROOT / "experiments/model/M002-v6-factorized-typed-attention/experiment.toml").read_text(encoding="utf-8")
        )
        plan = tomllib.loads((EXPERIMENT / "PLAN.toml").read_text(encoding="utf-8"))
        self.assertEqual(v6["status"], "superseded")
        self.assertEqual(plan["supersedes"]["experiment_id"], "M002-v6")
        self.assertIn("separately preregistered M009", plan["downstream"]["unlock_rule"])


if __name__ == "__main__":
    unittest.main()
