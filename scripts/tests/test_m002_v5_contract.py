import importlib.util
import json
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/model/M002-v5-factorized-typed-attention"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M002V5ContractTests(unittest.TestCase):
    def test_manifest_is_planned_and_names_only_the_direct_matched_pair(self):
        manifest = tomllib.loads((EXPERIMENT / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(manifest["seeds"], [17, 29, 43, 71, 101])
        self.assertIn("--phase paired-v5", manifest["entrypoint"])
        self.assertIn("--arms factorized-v2,factorized-v2-off", manifest["entrypoint"])
        self.assertNotIn("plain-transformer", manifest["entrypoint"])

    def test_protocol_uses_seed_as_replicate_and_keeps_m009_locked(self):
        plan = tomllib.loads((EXPERIMENT / "PLAN.toml").read_text(encoding="utf-8"))
        self.assertEqual(plan["pilot"]["seeds"], [7, 13])
        self.assertFalse(plan["pilot"]["claimable"])
        self.assertEqual(
            plan["confirmatory"]["folds"],
            ["evidence-temporal", "evidence-tabular", "claim-interventional"],
        )
        self.assertIn("average folds within each seed", plan["confirmatory"]["replicate"])
        self.assertEqual(plan["downstream"]["M009"], "locked")
        self.assertEqual(len(plan["gate"]), 9)

    def test_preparation_is_blocked_until_pilot_and_freeze_digests_exist(self):
        config = tomllib.loads((EXPERIMENT / "config.toml").read_text(encoding="utf-8"))["preregistration"]
        for key in (
            "rank", "bias_limit", "metadata_dropout",
            "dataset_lock_sha256", "criteria_sha256", "decision_script_sha256",
            "pilot_selection_script_sha256",
        ):
            self.assertTrue(str(config[key]).startswith("must-be-pinned-"), key)
        self.assertEqual(config["learning_rate"], "0.005")

    def test_dataset_and_decider_share_the_exact_confirmatory_folds(self):
        aggregate = load_module("aggregate_m002_v5_contract", ROOT / "scripts/aggregate_m002_v5.py")
        lock = json.loads((ROOT / "benchmarks/operator-routing-v2/splits.lock.json").read_text(encoding="utf-8"))
        confirmatory = tuple(name for name, value in lock["folds"].items() if value["kind"] == "confirmatory")
        self.assertEqual(confirmatory, aggregate.FOLDS)
        self.assertEqual(
            tuple(name for name, value in lock["folds"].items() if value["kind"] == "development"),
            ("claim-temporal-development",),
        )


if __name__ == "__main__":
    unittest.main()
