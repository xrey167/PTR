import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("aggregate_m002_v5", ROOT / "scripts/aggregate_m002_v5.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def records():
    digests = mod.expected_fold_digests()
    result = []
    for seed in mod.SEEDS:
        rows = [{"row": "run-v5", "seed": seed, "rank": 16, "bias_limit": 2.0, "metadata_dropout": 0.1}]
        for fold in mod.FOLDS:
            rows.append({"row": "data-v2", "fold": fold, "data_fnv64": digests[fold], "hard_validity_violations": 0})
            for arm in mod.ARMS:
                treatment = arm == mod.TREATMENT
                rows.extend([
                    {
                        "row": "meta", "fold": fold, "arm": arm, "nan": 0,
                        "typed_attention_mode": "factorized-v2" if treatment else "off",
                        "typed_attention_rank": 16, "typed_attention_limit": 2.0,
                        "typed_query": "on", "latent_steps": "2", "latent_nonlinearity": "on",
                        "frozen_router": "off", "router_mode": "calibrated-cosine-v2",
                        "router_logit_scale": 5.0, "label_smoothing": 0.05,
                        "metadata_dropout": 0.1, "consistency_weight": 0.1,
                        "batch": "typed", "lr": 0.005, "steps": 1500,
                        "total_params": 1000, "estimated_flops_per_example": 2000,
                        "final_train_loss": 0.2,
                    },
                    {
                        "row": "calibration", "fold": fold, "arm": arm,
                        "temperature": 1.2, "confidence_threshold": 0.5,
                    },
                    {"row": "performance-v5", "fold": fold, "arm": arm, "latency_p95_ms": 1.0},
                    {
                        "row": "final-v5", "fold": fold, "arm": arm, "split": "test_iid",
                        "accuracy": 0.90, "nll": 0.40, "ece15": 0.04,
                        "n": 100, "correct": 90, "covered": 100,
                        "coverage": 1.0, "covered_correct": 90,
                        "selective_error": 0.10, "abstained": 0,
                    },
                    {
                        "row": "final-v5", "fold": fold, "arm": arm, "split": "test_ood",
                        "accuracy": 0.85 if treatment else 0.80,
                        "nll": 0.40 if treatment else 0.50,
                        "ece15": 0.04 if treatment else 0.06,
                        "n": 100, "correct": 85 if treatment else 80,
                        "covered": 100, "coverage": 1.0,
                        "covered_correct": 85 if treatment else 80,
                        "selective_error": 0.15 if treatment else 0.20,
                        "abstained": 0,
                    },
                ])
        identity = {
            "git_sha": "a" * 40,
            "manifest_sha256": "b" * 64,
            "manifest": {
                "id": "M002-v5",
                "entrypoint": "frozen",
                "preregistration_sha256": "c" * 64,
                "preregistration_rules_sha256": "d" * 64,
            },
            "parameters": {"steps": "1500"},
            "rustc": {"version": "test"},
            "toolchain": {"name": "test"},
            "environment": {"PATH": "bound"},
            "executable": {"path": "cargo", "sha256": "f" * 64},
            "host": {"cpu": "test"},
            "cargo_lock_sha256": "e" * 64,
            "uv_lock_sha256": "0" * 64,
        }
        result.append((f"run-{seed}.json", {
            "experiment_id": "M002-v5", "entrypoint": "entrypoint", "seed": seed,
            "status": "completed", "exit_code": 0, "git_dirty": False,
            "stdout": "\n".join(json.dumps(row, sort_keys=True) for row in rows),
            **identity,
        }))
    return result


def mutate_row(records_value, seed, *, row, fold, arm=None, split=None, field, value):
    changed = copy.deepcopy(records_value)
    index = mod.SEEDS.index(seed)
    lines = [json.loads(line) for line in changed[index][1]["stdout"].splitlines()]
    matches = [item for item in lines if item.get("row") == row and item.get("fold") == fold
               and (arm is None or item.get("arm") == arm)
               and (split is None or item.get("split") == split)]
    assert len(matches) == 1
    matches[0][field] = value
    changed[index][1]["stdout"] = "\n".join(json.dumps(item, sort_keys=True) for item in lines)
    return changed


class M002V5DecisionTests(unittest.TestCase):
    def test_complete_positive_fixture_passes_every_gate(self):
        decision = mod.decide(records())
        self.assertEqual(decision["decision"], "PASS")
        self.assertTrue(all(decision["gates"].values()))

    def test_missing_seed_and_nan_are_hard_evidence_errors(self):
        with self.assertRaisesRegex(mod.EvidenceError, "exactly 5"):
            mod.decide(records()[:-1])
        bad = mutate_row(records(), 17, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                         split="test_ood", field="accuracy", value=float("nan"))
        with self.assertRaisesRegex(mod.EvidenceError, "not finite"):
            mod.decide(bad)

    def test_fold_sign_change_is_no_go_even_when_pooled_accuracy_is_positive(self):
        bad = records()
        for seed in mod.SEEDS:
            bad = mutate_row(bad, seed, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                             split="test_ood", field="accuracy", value=0.79)
            bad = mutate_row(bad, seed, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                             split="test_ood", field="correct", value=79)
        decision = mod.decide(bad)
        self.assertFalse(decision["gates"]["positive_accuracy_each_fold"])
        self.assertEqual(decision["decision"], "INCONCLUSIVE/NO-GO")

    def test_nll_and_ece_regressions_are_independent_no_go_gates(self):
        for metric, value, gate in (("nll", 0.60, "ood_nll"), ("ece15", 0.09, "ood_ece")):
            bad = records()
            for seed in mod.SEEDS:
                for fold in mod.FOLDS:
                    bad = mutate_row(bad, seed, row="final-v5", fold=fold, arm=mod.TREATMENT,
                                     split="test_ood", field=metric, value=value)
            with self.subTest(metric=metric):
                decision = mod.decide(bad)
                self.assertFalse(decision["gates"][gate])
                self.assertEqual(decision["decision"], "INCONCLUSIVE/NO-GO")

    def test_low_coverage_and_worse_selective_error_are_no_go(self):
        low = records()
        low = mutate_row(low, 17, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                         split="test_ood", field="covered", value=79)
        low = mutate_row(low, 17, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                         split="test_ood", field="covered_correct", value=67)
        low = mutate_row(low, 17, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                         split="test_ood", field="coverage", value=0.79)
        low = mutate_row(low, 17, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                         split="test_ood", field="selective_error", value=12 / 79)
        low = mutate_row(low, 17, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                         split="test_ood", field="abstained", value=21)
        self.assertFalse(mod.decide(low)["gates"]["coverage"])
        worse = records()
        for seed in mod.SEEDS:
            for fold in mod.FOLDS:
                worse = mutate_row(worse, seed, row="final-v5", fold=fold, arm=mod.TREATMENT,
                                   split="test_ood", field="covered_correct", value=79)
                worse = mutate_row(worse, seed, row="final-v5", fold=fold, arm=mod.TREATMENT,
                                   split="test_ood", field="selective_error", value=0.21)
        self.assertFalse(mod.decide(worse)["gates"]["selective_error"])

    def test_parameter_flop_and_latency_mismatch_are_no_go(self):
        for field, value in (("total_params", 1001), ("estimated_flops_per_example", 2021)):
            bad = mutate_row(records(), 17, row="meta", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                             field=field, value=value)
            with self.subTest(field=field):
                self.assertFalse(mod.decide(bad)["gates"]["matched_compute"])
        slow = mutate_row(records(), 17, row="performance-v5", fold=mod.FOLDS[0],
                          arm=mod.TREATMENT, field="latency_p95_ms", value=1.11)
        self.assertFalse(mod.decide(slow)["gates"]["matched_compute"])

    def test_reported_abstention_metrics_must_reconcile_with_counts(self):
        bad = mutate_row(records(), 17, row="final-v5", fold=mod.FOLDS[0], arm=mod.TREATMENT,
                         split="test_ood", field="coverage", value=0.99)
        with self.assertRaisesRegex(mod.EvidenceError, "coverage disagrees with counts"):
            mod.decide(bad)

    def test_digest_and_duplicate_seed_cannot_be_reinterpreted_as_no_go(self):
        bad = mutate_row(records(), 17, row="data-v2", fold=mod.FOLDS[0], field="data_fnv64", value="0" * 16)
        with self.assertRaisesRegex(mod.EvidenceError, "digest mismatch"):
            mod.decide(bad)
        duplicate = records()
        duplicate[-1][1]["seed"] = 17
        with self.assertRaisesRegex(mod.EvidenceError, "duplicate"):
            mod.decide(duplicate)


if __name__ == "__main__":
    unittest.main()
