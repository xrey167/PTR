import copy
import importlib.util
import json
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/model/M002-v6-factorized-typed-attention"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class M002V6ContractTests(unittest.TestCase):
    def test_successor_is_prepared_and_uses_only_the_matched_pair(self):
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual((manifest["id"], manifest["status"]), ("M002-v6", "prepared"))
        self.assertIn("--experiment M002-v6", manifest["entrypoint"])
        self.assertIn("--arms factorized-v2,factorized-v2-off", manifest["entrypoint"])

    def test_confirmatory_command_binds_the_three_exact_fold_digests(self):
        gates = load_module("m002_v6_gate", ROOT / "scripts/check_research_gates.py")
        config = tomllib.loads((EXPERIMENT / "config.toml").read_text(encoding="utf-8"))["preregistration"]
        self.assertEqual(gates.m002_v6_fold_binding_errors(ROOT, config), [])
        invalid = dict(config)
        invalid["folds"] = "evidence-temporal,evidence-tabular,claim-interventional"
        self.assertEqual(len(gates.m002_v6_fold_binding_errors(ROOT, invalid)), 1)

    def test_prepared_successor_freeze_accepts_only_the_immutable_pilot_source(self):
        gates = load_module("m002_v6_successor_gate", ROOT / "scripts/check_research_gates.py")
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual(gates.m002_v6_successor_freeze_errors(ROOT, EXPERIMENT, manifest), [])

    def test_adapter_preserves_v6_identity_and_rejects_other_studies(self):
        v5_tests = load_module("m002_v5_fixture", ROOT / "scripts/tests/test_aggregate_m002_v5.py")
        adapter = load_module("aggregate_m002_v6", ROOT / "scripts/aggregate_m002_v6.py")
        core = adapter.load_core()
        records = []
        for name, record in v5_tests.records():
            clone = copy.deepcopy(record)
            clone["experiment_id"] = "M002-v6"
            clone["manifest"]["id"] = "M002-v6"
            records.append((name, clone))
        decision = adapter.decide(records, protocol=core.fixture_protocol())
        self.assertEqual(decision["experiment_id"], "M002-v6")
        self.assertEqual(
            [item["canonical_sha256"] for item in decision["provenance"]["source_records"]],
            [
                adapter.hashlib.sha256(adapter.canonical(record).encode("utf-8")).hexdigest()
                for _, record in sorted(records)
            ],
        )
        with self.assertRaisesRegex(adapter.EvidenceError, "experiment_id"):
            adapter.decide(v5_tests.records(), protocol=core.fixture_protocol())

    def test_source_selection_is_explicitly_development_only(self):
        plan = tomllib.loads((EXPERIMENT / "PLAN.toml").read_text(encoding="utf-8"))
        self.assertEqual(plan["selection_source"]["experiment_id"], "M002-v5")
        self.assertFalse(plan["selection_source"]["claimable"])
        self.assertIn("separately preregistered M009", plan["downstream"]["unlock_rule"])


if __name__ == "__main__":
    unittest.main()
