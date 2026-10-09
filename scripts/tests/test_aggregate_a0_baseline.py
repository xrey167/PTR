"""Tests of the A0 baseline-v1 decider (scripts/aggregate_a0_baseline.py).

Every rule of the decision procedure is made to fire with synthetic rows built
from the preregistered criteria, and every refusal of malformed evidence is
exercised. The rows imitate the runner's baseline-v1 schema (model/burn-a0
examples/a0_ablation/baseline.rs).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import statistics
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("aggregate_a0_baseline", ROOT / "scripts/aggregate_a0_baseline.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

CRITERIA = mod.load_criteria()
SEEDS = list(mod.CORE.SEEDS)
FOLDS = [fold for fold, _ in CRITERIA["folds"]]
T = mod.CORE.T_CRITICAL_95_DF4
N = 1000
PLAIN_PARAMS = 33323
A0_PARAMS = 33324
PLAIN_FLOPS = 1665216


def noise(seed_index: int, arm_index: int, fold_index: int) -> float:
    """Deterministic, mean-zero-ish jitter so paired deltas have a nonzero spread."""
    return ((seed_index * 3 + arm_index * 5 + fold_index) % 5 - 2) * 0.003


class Scenario:
    """Per-arm levels and per-cell tweaks for the synthetic rows of one study."""

    def __init__(self, study_id: str):
        self.study = CRITERIA["studies"][study_id]
        self.runner = self.study["runner_study"]
        self.level: dict[str, dict] = {}
        self.params: dict[str, int] = {}
        self.flops: dict[str, int] = {}
        self.lr: dict[str, float] = {}
        self.tweaks: list = []

    def base(self, arm: str) -> dict:
        values = {"acc_iid": 0.80, "acc_ood": 0.60, "nll": 1.0, "ece": 0.10, "cov_iid": 0.95, "cov_ood": 0.85, "sel": 0.10}
        values.update(self.level.get(arm, {}))
        return values

    def cell(self, arm: str, split: str, fold: str, seed: int) -> dict:
        si, ai, fi = (SEEDS.index(seed) if seed in SEEDS else 0), self.study["arms"].index(arm), FOLDS.index(fold)
        values = self.base(arm)
        ood = split == "test_ood"
        cell = {
            "acc": values["acc_ood" if ood else "acc_iid"] + noise(si, ai, fi),
            "nll": values["nll"] + noise(si, ai, fi) * 2,
            "ece": values["ece"] + noise(si, ai, fi) * 0.5,
            "cov": values["cov_ood" if ood else "cov_iid"],
            "sel": values["sel"],
        }
        for tweak in self.tweaks:
            tweak(arm, split, fold, seed, cell)
        return cell

    def meta_row(self, arm: str, fold: str, seed: int) -> dict:
        row = {
            "row": "meta-baseline-v1", "study": self.runner, "fold": fold, "arm": arm, "seed": seed,
            "lr": self.lr.get(arm, 0.005), "total_params": self.params.get(arm, PLAIN_PARAMS if arm == "plain" else A0_PARAMS),
            "flops_per_example": self.flops.get(arm, PLAIN_FLOPS if arm == "plain" else 1_500_000 + 1000 * self.pins(arm)),
            "flops_per_example_legacy": 1.0, "train_wall_ms": 1.0, "ms_per_step": 20.0,
            "final_train_loss": 0.5, "nan": 0,
        }
        row.update(mod.expected_meta(CRITERIA, self.study, arm))
        return row

    def pins(self, arm: str) -> int:
        return 1 if arm == "plain" else self.study["pins"][arm]["latent_steps"]

    def rows(self, seed: int) -> list[dict]:
        rows = [{
            "row": "run-baseline-v1", "study": self.runner, "seed": seed, "d_model": 48, "batch_size": 128,
            "steps": 1500, "arms": ",".join(self.study["arms"]), "order_salt": 765892,
        }]
        for fold, digest in CRITERIA["folds"]:
            rows.append({"row": "data-v2", "fold": fold, "data_fnv64": digest})
            for arm in self.study["arms"]:
                meta = self.meta_row(arm, fold, seed)
                rows.append(meta)
                for split in ("test_iid", "test_ood"):
                    cell = self.cell(arm, split, fold, seed)
                    correct = round(min(max(cell["acc"], 0.0), 1.0) * N)
                    rows.append({
                        "row": "final-baseline-v1", "study": self.runner, "fold": fold, "arm": arm, "split": split,
                        "seed": seed, "n": N, "correct": correct, "accuracy": correct / N,
                        "raw_nll": max(cell["nll"], 0.0), "raw_ece15": min(max(cell["ece"], 0.0), 1.0),
                        "raw_confidence_threshold": 0.5, "raw_coverage": cell["cov"], "raw_selective_error": cell["sel"],
                        "fitted_temperature": 1.5, "fitted_nll": 0.9, "fitted_ece15": 0.05,
                        "total_params": meta["total_params"], "flops_per_example": meta["flops_per_example"],
                        "flops_per_example_legacy": 1.0, "train_wall_ms": 1.0, "ms_per_step": 20.0,
                        "latency_p95_ms": 0.2, "latent_steps": meta["latent_steps"],
                        "explicit_token_count": 5000, "explicit_token_mean": 5.0,
                    })
            if self.study["require_mask_share"]:
                for split in ("test_iid", "test_ood"):
                    acc = {
                        arm: next(r for r in rows if r.get("row") == "final-baseline-v1" and r["fold"] == fold and r["arm"] == arm and r["split"] == split)["accuracy"]
                        for arm in ("full", "no-semantic-slots-masked", "no-semantic-slots")
                    }
                    gap = acc["full"] - acc["no-semantic-slots"]
                    rows.append({
                        "row": "mask-share-baseline-v1", "study": self.runner, "fold": fold, "split": split,
                        "masked_accuracy": acc["no-semantic-slots-masked"], "unmasked_accuracy": acc["no-semantic-slots"],
                        "mask_share": acc["no-semantic-slots-masked"] - acc["no-semantic-slots"],
                        "mask_share_of_full_gap": (acc["no-semantic-slots-masked"] - acc["no-semantic-slots"]) / gap if gap > 0 else None,
                        "nonlive_slot_share": 0.4,
                    })
        return rows

    def record(self, seed: int, **overrides) -> dict:
        rows = self.rows(seed)
        stdout = "\n".join(["noise line"] + [json.dumps(row) for row in rows] + ["PRED full test_iid 00 FOLD x"])
        record = {
            "schema_version": 2, "experiment_id": self.study["id"], "entrypoint": "entrypoint", "status": "completed",
            "exit_code": 0, "git_dirty": False, "git_sha": "a" * 40, "manifest_sha256": "b" * 64,
            "parameters": {"d_model": "48"}, "rustc": None, "toolchain": {"x": 1}, "environment": {"E": "1"},
            "executable": {"sha256": "c" * 64}, "host": {"system": "Linux"}, "cargo_lock_sha256": "d" * 64,
            "uv_lock_sha256": "e" * 64, "seed": seed, "stdout": stdout,
            "manifest": {
                "id": self.study["id"], "preregistration_sha256": "f" * 64, "preregistration_rules_sha256": "9" * 64,
            },
        }
        record.update(overrides)
        return record

    def sources(self) -> list[tuple[str, dict]]:
        return [(f"run-seed-{seed}.json", self.record(seed)) for seed in SEEDS]

    def decide(self, **kwargs) -> dict:
        return mod.decide(self.sources(), self.study["id"], CRITERIA, **kwargs)


def supports_m002() -> Scenario:
    scenario = Scenario("M002-v10")
    scenario.level["factorized-v2"] = {"acc_ood": 0.67, "acc_iid": 0.81, "nll": 0.9, "ece": 0.08}
    return scenario


def contrast(result: dict, contrast_id: str) -> dict:
    return next(item for item in result["contrasts"] if item["id"] == contrast_id)


def endpoint(item: dict, scope: str, split: str, key: str) -> dict:
    return next(e for e in item["endpoints"] if e["scope"] == scope and e["split"] == split and e["key"] == key)


class CriteriaPins(unittest.TestCase):
    def test_protocol_thresholds_and_contrasts_are_pinned(self):
        c = CRITERIA
        self.assertEqual(c["raw"]["seeds"], [17, 29, 43, 71, 101])
        protocol = c["protocol"]
        self.assertEqual(
            (protocol["d_model"], protocol["steps"], protocol["batch_size"], protocol["rank"]), (48, 1500, 128, 8)
        )
        self.assertEqual((protocol["bias_limit"], protocol["metadata_dropout"]), (1.0, 0.0))
        self.assertEqual(protocol["lr_grid"], [0.0025, 0.005, 0.0125, 0.025])
        self.assertEqual(protocol["sweep_seed"], 17)
        th = c["thresholds"]
        self.assertEqual(th["delta_min"], 0.02)
        self.assertEqual((th["ood_nll_ci_upper_le"], th["ood_ece_mean_le"], th["ood_ece_ci_upper_le"]), (0.0, 0.0, 0.02))
        self.assertEqual(th["iid_accuracy_ci_lower_ge"], -0.01)
        self.assertEqual((th["coverage_iid_ge"], th["coverage_ood_ge"]), (0.90, 0.80))
        self.assertEqual(th["plain_param_tolerance"], 0.01)
        self.assertEqual(c["columns"]["nll"], "raw_nll")
        self.assertEqual(c["columns"]["ece"], "raw_ece15")
        self.assertEqual(c["rules"]["calibration_gates"], ["ood_nll_each_fold", "ood_ece_each_fold"])
        self.assertAlmostEqual(c["raw"]["t_critical"], T, places=3)

    def test_fold_tokens_equal_the_m002_v9_tokens(self):
        config = next((ROOT / "experiments/model").glob("M002-v9-*/config.toml"))
        v9 = tomllib.loads(config.read_text(encoding="utf-8"))["preregistration"]["folds"]
        self.assertEqual(CRITERIA["raw"]["data"]["folds"], v9)
        self.assertEqual(mod.CORE.expected_fold_digests(ROOT), dict(CRITERIA["folds"]))

    def test_label_smoothing_and_consistency_only_on_the_m002_pair(self):
        study = CRITERIA["studies"]["M002-v10"]
        for arm in ("factorized-v2", "factorized-v2-off"):
            meta = mod.expected_meta(CRITERIA, study, arm)
            self.assertEqual((meta["label_smoothing"], meta["consistency_weight"]), (0.05, 0.10))
        self.assertEqual(mod.expected_meta(CRITERIA, study, "full")["label_smoothing"], 0.0)
        plain = mod.expected_meta(CRITERIA, study, "plain")
        self.assertEqual(plain["kind"], "plain")
        self.assertNotIn("label_smoothing", plain)
        for sid in ("M001-v5", "M003-v2"):
            for arm in CRITERIA["studies"][sid]["arms"]:
                if arm != "plain":
                    self.assertEqual(mod.expected_meta(CRITERIA, CRITERIA["studies"][sid], arm)["label_smoothing"], 0.0)

    def test_contrast_structure_per_study(self):
        shape = {
            sid: [(c["id"], c["role"], c["isolating"], c["treatment"], c["control"], c["compute"]) for c in s["contrast"]]
            for sid, s in CRITERIA["studies"].items()
        }
        self.assertEqual(shape["M001-v5"], [
            ("M001-v5-primary", "primary", True, "full", "no-semantic-slots-masked", "equal-params"),
            ("M001-v5-bundle", "bundle", False, "full", "plain", "plain-params"),
        ])
        self.assertEqual(shape["M002-v10"], [
            ("M002-v10-primary", "primary", True, "factorized-v2", "factorized-v2-off", "equal-params-flops"),
            ("M002-v10-bundle", "bundle", False, "full", "plain", "plain-params"),
        ])
        self.assertEqual(shape["M003-v2"], [
            ("M003-v2-primary", "primary", True, "full", "latent-0", "equal-params"),
            ("M003-v2-nonlinearity", "secondary", True, "full", "latent-linear", "equal-params-flops"),
            ("M003-v2-bundle", "bundle", False, "full", "plain", "plain-params"),
        ])
        self.assertIn("M004-v2", CRITERIA["deferred"])
        self.assertIn("no-semantic-slots", CRITERIA["studies"]["M001-v5"]["arms"])

    def test_the_immutable_core_is_pinned_and_unmodified(self):
        pin = CRITERIA["raw"]["decision"]["core_sha256"]
        self.assertEqual(hashlib.sha256((ROOT / "scripts/aggregate_m002_v5.py").read_bytes()).hexdigest(), pin)
        config = next((ROOT / "experiments/model").glob("M002-v9-*/config.toml"))
        v9 = tomllib.loads(config.read_text(encoding="utf-8"))["preregistration"]
        self.assertEqual(pin, v9["decision_core_sha256"])

    def test_a_tampered_criteria_file_is_refused(self):
        text = (ROOT / "research/falsification/A0-baseline-v1/criteria.toml").read_text(encoding="utf-8")
        cases = {
            "core hash": text.replace(CRITERIA["raw"]["decision"]["core_sha256"], "0" * 64),
            "fold token": text.replace("8c3bc2324e0eb770", "8c3bc2324e0eb771"),
            "seeds": text.replace("seeds = [17, 29, 43, 71, 101]", "seeds = [17, 29, 43, 71, 102]", 1),
            "lock hash": text.replace(CRITERIA["raw"]["data"]["lock_sha256"], "1" * 64),
            "t value": text.replace("t_critical = 2.776", "t_critical = 2.0"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, changed in cases.items():
                self.assertNotEqual(changed, text, name)
                path = Path(directory) / f"{name.replace(' ', '_')}.toml"
                path.write_text(changed, encoding="utf-8")
                with self.assertRaises(mod.EvidenceError, msg=name):
                    mod.load_criteria(path)


class Intervals(unittest.TestCase):
    def test_pooled_endpoint_averages_folds_inside_each_seed_then_pairs_over_seeds(self):
        scenario = supports_m002()
        scenario.tweaks.append(
            lambda arm, split, fold, seed, cell: cell.__setitem__("acc", cell["acc"] + (0.03 if fold == FOLDS[0] else 0.0))
            if arm == "factorized-v2" and split == "test_ood" else None
        )
        result = scenario.decide()
        item = contrast(result, "M002-v10-primary")
        pooled = endpoint(item, "pooled-over-folds", "test_ood", "accuracy")
        table = mod.Table(mod.build_table(scenario.sources(), CRITERIA, scenario.study)[0], CRITERIA)
        expected = [
            statistics.fmean(
                table.final(s, f, "factorized-v2", "test_ood", "accuracy") - table.final(s, f, "factorized-v2-off", "test_ood", "accuracy")
                for f in FOLDS
            )
            for s in SEEDS
        ]
        for got, want in zip(pooled["deltas"], expected, strict=True):
            self.assertAlmostEqual(got, want, places=12)
        mean, sd = statistics.fmean(expected), statistics.stdev(expected)
        half = T * sd / math.sqrt(5)
        self.assertAlmostEqual(pooled["mean"], mean, places=15)
        self.assertAlmostEqual(pooled["ci95"][0], mean - half, places=12)
        self.assertAlmostEqual(pooled["ci95"][1], mean + half, places=12)
        per_fold = endpoint(item, FOLDS[0], "test_ood", "accuracy")
        self.assertEqual(len(per_fold["deltas"]), 5)
        self.assertFalse(per_fold["diagnostic"])
        self.assertTrue(endpoint(item, "pooled-over-folds", "test_ood", "diagnostic_nll")["diagnostic"])


class Rules(unittest.TestCase):
    def test_supports_when_every_gate_and_the_primary_hold(self):
        result = supports_m002().decide()
        primary = contrast(result, "M002-v10-primary")
        self.assertEqual((primary["verdict"], primary["label"]), ("SUPPORTS", "SUPPORTS"))
        self.assertEqual(primary["failing_gates"], [])
        self.assertTrue(all(primary["gates"].values()))
        self.assertTrue(primary["isolating"])
        self.assertEqual(result["headline"], {"contrast": "M002-v10-primary", "verdict": "SUPPORTS", "label": "SUPPORTS"})
        self.assertTrue(primary["quality_claim_allowed"])

    def test_every_contrast_is_decided_separately(self):
        scenario = Scenario("M003-v2")
        scenario.level["full"] = {"acc_ood": 0.68, "nll": 0.9, "ece": 0.08}
        result = scenario.decide()
        self.assertEqual(
            {item["id"]: item["verdict"] for item in result["contrasts"]},
            {"M003-v2-primary": "SUPPORTS", "M003-v2-nonlinearity": "SUPPORTS", "M003-v2-bundle": "SUPPORTS"},
        )
        scenario.level["latent-linear"] = {"acc_ood": 0.68, "nll": 0.9, "ece": 0.08}
        result = scenario.decide()
        self.assertEqual(contrast(result, "M003-v2-primary")["verdict"], "SUPPORTS")
        self.assertEqual(contrast(result, "M003-v2-nonlinearity")["verdict"], "FALSIFIES")
        self.assertEqual(result["headline"]["verdict"], "SUPPORTS")

    def test_bundle_contrast_is_labelled_non_isolating(self):
        scenario = Scenario("M001-v5")
        scenario.level["full"] = {"acc_ood": 0.68, "nll": 0.9, "ece": 0.08}
        result = scenario.decide()
        bundle = contrast(result, "M001-v5-bundle")
        self.assertFalse(bundle["isolating"])
        self.assertIn("non-isolating", bundle["scope_note"])
        self.assertEqual(bundle["verdict"], "SUPPORTS")
        # The decision file refuses to let a bundle result stand in for mechanism evidence.
        self.assertIn("allow_bundle_claim_as_mechanism_evidence = false", mod.decision_toml(result, "0" * 64))

    def test_falsifies_and_harmful(self):
        scenario = supports_m002()
        scenario.level["factorized-v2"] = {"acc_ood": 0.60}
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertEqual((item["verdict"], item["label"]), ("FALSIFIES", "FALSIFIES"))
        self.assertIn("ood_accuracy", item["failing_gates"])
        self.assertFalse(item["quality_claim_allowed"])
        scenario.level["factorized-v2"] = {"acc_ood": 0.50}
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertEqual((item["verdict"], item["label"]), ("FALSIFIES", "HARMFUL"))

    def test_falsification_needs_both_arms_learnable(self):
        scenario = supports_m002()
        scenario.level["factorized-v2"] = {"acc_ood": 0.60, "acc_iid": 0.20}
        scenario.level["factorized-v2-off"] = {"acc_iid": 0.20}
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertEqual(item["verdict"], "INCONCLUSIVE")
        self.assertIn("not learnable", item["reason"])
        self.assertFalse(any(item["learnable"].values()))

    def test_falsification_needs_the_compute_match(self):
        scenario = supports_m002()
        scenario.level["factorized-v2"] = {"acc_ood": 0.60}
        scenario.flops["factorized-v2-off"] = 1_400_000
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertEqual(item["verdict"], "INCONCLUSIVE")
        self.assertIn("matched_compute", item["failing_gates"])

    def test_inconclusive_names_every_failing_gate(self):
        scenario = supports_m002()
        shifts = [0.08, -0.06, 0.07, -0.05, 0.06]
        scenario.level["factorized-v2"] = {"acc_ood": 0.60, "acc_iid": 0.70, "cov_ood": 0.5, "sel": 0.2}
        scenario.tweaks.append(
            lambda arm, split, fold, seed, cell: cell.__setitem__("acc", cell["acc"] + shifts[SEEDS.index(seed)])
            if arm == "factorized-v2" and split == "test_ood" else None
        )
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertEqual((item["verdict"], item["label"]), ("INCONCLUSIVE", "INCONCLUSIVE"))
        for gate in ("ood_accuracy", "iid_accuracy", "coverage", "selective_error"):
            self.assertIn(gate, item["failing_gates"])
        self.assertNotIn("matched_compute", item["failing_gates"])

    def test_accuracy_gain_with_a_failed_calibration_gate_is_no_go_calibration(self):
        scenario = supports_m002()
        scenario.level["factorized-v2"] = {"acc_ood": 0.67, "nll": 1.6, "ece": 0.08}
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertEqual((item["verdict"], item["label"]), ("INCONCLUSIVE", "NO-GO-CALIBRATION"))
        self.assertTrue(item["gates"]["ood_accuracy"])
        self.assertEqual(item["failing_gates"], ["ood_nll_each_fold"])
        self.assertFalse(item["quality_claim_allowed"])

    def test_per_fold_calibration_regression_blocks_supports_although_pooled_is_fine(self):
        """The M001-v4 failure: a pooled average hides a NLL regression on one fold."""
        scenario = supports_m002()
        scenario.level["factorized-v2"] = {"acc_ood": 0.67, "nll": 3.0, "ece": 0.5}
        scenario.level["factorized-v2-off"] = {"nll": 3.0, "ece": 0.5}

        def tweak(arm, split, fold, seed, cell):
            if arm == "factorized-v2" and split == "test_ood":
                cell["nll"] += 2.0 if fold == FOLDS[2] else -2.0
                cell["ece"] += 0.1 if fold == FOLDS[2] else -0.1

        scenario.tweaks.append(tweak)
        result = scenario.decide()
        item = contrast(result, "M002-v10-primary")
        pooled_nll = endpoint(item, "pooled-over-folds", "test_ood", "nll")
        pooled_ece = endpoint(item, "pooled-over-folds", "test_ood", "ece")
        self.assertLessEqual(pooled_nll["ci95"][1], 0.0)
        self.assertLessEqual(pooled_ece["mean"], 0.0)
        self.assertGreater(endpoint(item, FOLDS[2], "test_ood", "nll")["ci95"][0], 0.0)
        self.assertFalse(item["gates"]["ood_nll_each_fold"])
        self.assertFalse(item["gates"]["ood_ece_each_fold"])
        self.assertEqual((item["verdict"], item["label"]), ("INCONCLUSIVE", "NO-GO-CALIBRATION"))

    def test_ece_gate_needs_both_mean_and_upper_bound(self):
        for shift, expected in ((0.001, False), (-0.1, True)):
            scenario = supports_m002()
            scenario.level["factorized-v2"] = {"acc_ood": 0.67, "nll": 0.9, "ece": 0.10 + shift}
            item = contrast(scenario.decide(), "M002-v10-primary")
            self.assertEqual(item["gates"]["ood_ece_each_fold"], expected, shift)
        scenario = supports_m002()
        shifts = [0.04, -0.04, 0.03, -0.03, 0.0]
        scenario.level["factorized-v2"] = {"acc_ood": 0.67, "nll": 0.9, "ece": 0.09}
        scenario.tweaks.append(
            lambda arm, split, fold, seed, cell: cell.__setitem__("ece", cell["ece"] + shifts[SEEDS.index(seed)] * 3)
            if arm == "factorized-v2" and split == "test_ood" else None
        )
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertLessEqual(endpoint(item, FOLDS[0], "test_ood", "ece")["mean"], 0.0)
        self.assertGreater(endpoint(item, FOLDS[0], "test_ood", "ece")["ci95"][1], 0.02)
        self.assertFalse(item["gates"]["ood_ece_each_fold"])

    def test_negative_accuracy_in_one_fold_blocks_supports_without_a_calibration_label(self):
        scenario = supports_m002()
        scenario.level["factorized-v2"] = {"acc_ood": 0.75, "nll": 0.9, "ece": 0.08}
        scenario.tweaks.append(
            lambda arm, split, fold, seed, cell: cell.__setitem__("acc", cell["acc"] - 0.20)
            if arm == "factorized-v2" and split == "test_ood" and fold == FOLDS[1] else None
        )
        item = contrast(scenario.decide(), "M002-v10-primary")
        self.assertTrue(item["gates"]["ood_accuracy"])
        self.assertFalse(item["gates"]["positive_accuracy_each_fold"])
        self.assertEqual((item["verdict"], item["label"]), ("INCONCLUSIVE", "INCONCLUSIVE"))
        self.assertIn("positive_accuracy_each_fold", item["failing_gates"])

    def test_iid_harm_coverage_and_selective_error_gates(self):
        for level, gate in (
            ({"acc_iid": 0.70}, "iid_accuracy"),
            ({"cov_ood": 0.79}, "coverage"),
            ({"cov_iid": 0.89}, "coverage"),
            ({"sel": 0.11}, "selective_error"),
        ):
            scenario = supports_m002()
            scenario.level["factorized-v2"] = {**scenario.level["factorized-v2"], **level}
            item = contrast(scenario.decide(), "M002-v10-primary")
            self.assertEqual(item["failing_gates"], [gate], level)
            self.assertEqual(item["label"], "INCONCLUSIVE")

    def test_coverage_is_judged_for_the_treatment_arm_only(self):
        scenario = supports_m002()
        scenario.level["factorized-v2-off"] = {"cov_ood": 0.1, "cov_iid": 0.1}
        self.assertEqual(contrast(scenario.decide(), "M002-v10-primary")["verdict"], "SUPPORTS")

    def test_plain_parameter_match_uses_total_params_within_one_percent(self):
        scenario = supports_m002()
        scenario.level["full"] = {"acc_ood": 0.67, "nll": 0.9, "ece": 0.08}
        self.assertEqual(contrast(scenario.decide(), "M002-v10-bundle")["verdict"], "SUPPORTS")
        scenario.params["plain"] = 33323 + 400
        bundle = contrast(scenario.decide(), "M002-v10-bundle")
        self.assertEqual(bundle["failing_gates"], ["matched_compute"])
        self.assertEqual(bundle["label"], "INCONCLUSIVE")
        scenario.params["plain"] = 33323 + 300
        self.assertTrue(contrast(scenario.decide(), "M002-v10-bundle")["gates"]["matched_compute"])
        # FLOPs of plain and A0 are not comparable and are not gated against plain.
        scenario.flops["plain"] = 900_000
        self.assertTrue(contrast(scenario.decide(), "M002-v10-bundle")["gates"]["matched_compute"])

    def test_isolating_pairs_need_equal_parameters_and_flops(self):
        scenario = supports_m002()
        scenario.params["factorized-v2-off"] = A0_PARAMS + 1
        self.assertEqual(contrast(scenario.decide(), "M002-v10-primary")["failing_gates"], ["matched_compute"])
        scenario = supports_m002()
        scenario.flops["factorized-v2-off"] = 1_500_000 + 2001
        self.assertEqual(contrast(scenario.decide(), "M002-v10-primary")["failing_gates"], ["matched_compute"])
        scenario = Scenario("M003-v2")
        scenario.level["full"] = {"acc_ood": 0.68, "nll": 0.9, "ece": 0.08}
        scenario.flops["latent-linear"] = 1_500_000 + 2001
        result = scenario.decide()
        self.assertEqual(contrast(result, "M003-v2-nonlinearity")["failing_gates"], ["matched_compute"])
        # latent-0 differs in compute by construction: equal parameters suffice, FLOPs are descriptive.
        self.assertEqual(contrast(result, "M003-v2-primary")["verdict"], "SUPPORTS")
        scenario.params["latent-0"] = A0_PARAMS + 5
        self.assertEqual(contrast(scenario.decide(), "M003-v2-primary")["failing_gates"], ["matched_compute"])

    def test_mask_share_is_reported_and_cross_checked(self):
        scenario = Scenario("M001-v5")
        scenario.level["full"] = {"acc_ood": 0.68}
        scenario.level["no-semantic-slots"] = {"acc_ood": 0.50}
        result = scenario.decide()
        share = result["mask_share"]["test_ood"]
        self.assertAlmostEqual(share["mask_share"]["mean"], 0.10, places=2)
        self.assertEqual(share["mask_share_of_full_gap_defined_cells"], 15)
        self.assertIsNone(Scenario("M002-v10").decide()["mask_share"])

    def test_temperature_saturation_is_reported_as_a_diagnostic(self):
        scenario = supports_m002()
        scenario.tweaks.append(lambda *args: None)
        result = scenario.decide()
        self.assertEqual(result["arms"]["plain"]["test_ood"]["fitted_temperature_saturated_cells"], 0)
        sources = scenario.sources()
        name, record = sources[0]
        rows = [json.loads(line) for line in record["stdout"].splitlines() if line.startswith("{")]
        for row in rows:
            if row.get("row") == "final-baseline-v1" and row["arm"] == "plain" and row["split"] == "test_ood":
                row["fitted_temperature"] = 4.0
        record["stdout"] = "\n".join(json.dumps(row) for row in rows)
        result = mod.decide(sources, "M002-v10", CRITERIA)
        self.assertEqual(result["arms"]["plain"]["test_ood"]["fitted_temperature_saturated_cells"], 3)
        self.assertIn("grid ends at the cap", result["diagnostics"]["note"])

    def test_decision_is_deterministic(self):
        scenario = supports_m002()
        self.assertEqual(json.dumps(scenario.decide(), sort_keys=True), json.dumps(scenario.decide(), sort_keys=True))


def mutated(scenario: Scenario, edit) -> list[tuple[str, dict]]:
    """Sources with `edit(seed, rows) -> rows` applied to every record's rows."""
    sources = []
    for seed in SEEDS:
        record = scenario.record(seed)
        rows = [json.loads(line) for line in record["stdout"].splitlines() if line.startswith("{")]
        rows = edit(seed, rows)
        record["stdout"] = "\n".join(json.dumps(row) for row in rows)
        sources.append((f"run-seed-{seed}.json", record))
    return sources


class Refusals(unittest.TestCase):
    def refuse(self, sources, study="M002-v10", text=None, **kwargs):
        with self.assertRaises(mod.EvidenceError) as caught:
            mod.decide(sources, study, CRITERIA, **kwargs)
        if text:
            self.assertIn(text, str(caught.exception))

    def test_missing_duplicate_and_foreign_seeds(self):
        scenario = supports_m002()
        sources = scenario.sources()
        self.refuse(sources[:4], text="expected exactly 5")
        self.refuse([*sources[:4], sources[0]], text="duplicate")
        foreign = scenario.record(7)
        self.refuse([*sources[:4], ("x.json", foreign)], text="not confirmatory")
        mislabelled = scenario.record(101)
        mislabelled["seed"] = 17
        self.refuse([*sources[:4], ("x.json", mislabelled)], text="differs from the run row seed")

    def test_missing_evidence_cells_are_refused(self):
        scenario = supports_m002()

        def drop(predicate):
            return lambda seed, rows: [r for r in rows if not (seed == 43 and predicate(r))] if seed == 43 else rows

        self.refuse(mutated(scenario, drop(lambda r: r["row"] == "final-baseline-v1" and r["arm"] == "plain" and r["split"] == "test_ood" and r["fold"] == FOLDS[1])), text="missing final row")
        self.refuse(mutated(scenario, drop(lambda r: r["row"] == "meta-baseline-v1" and r["arm"] == "factorized-v2")), text="missing meta row")
        self.refuse(mutated(scenario, drop(lambda r: r["row"] == "data-v2" and r["fold"] == FOLDS[0])), text="missing data identity")
        self.refuse(mutated(scenario, drop(lambda r: r["row"] == "run-baseline-v1")), text="run-baseline-v1")

    def test_non_finite_values_are_refused(self):
        scenario = supports_m002()
        for value in (float("nan"), float("inf")):
            def edit(seed, rows, value=value):
                for row in rows:
                    if seed == 71 and row["row"] == "final-baseline-v1" and row["arm"] == "factorized-v2":
                        row["raw_nll"] = value
                return rows

            self.refuse(mutated(scenario, edit), text="not finite")

        def nan_training(seed, rows):
            for row in rows:
                if seed == 17 and row["row"] == "meta-baseline-v1" and row["arm"] == "full":
                    row["nan"] = 1
            return rows

        self.refuse(mutated(scenario, nan_training), text="NaN")

    def test_out_of_range_and_inconsistent_values_are_refused(self):
        scenario = supports_m002()
        for field, value, text in (
            ("raw_ece15", 1.5, "outside"),
            ("raw_nll", -0.1, "negative"),
            ("accuracy", 0.123, "disagrees with counts"),
            ("fitted_temperature", 0.0, "temperature"),
            ("total_params", 7, "final vs meta"),
        ):
            def edit(seed, rows, field=field, value=value):
                for row in rows:
                    if seed == 29 and row["row"] == "final-baseline-v1" and row["arm"] == "full" and row["split"] == "test_iid":
                        row[field] = value
                return rows

            self.refuse(mutated(scenario, edit), text=text)

    def test_protocol_and_arm_binding_mismatches_are_refused(self):
        scenario = supports_m002()
        for field, value, text in (
            ("steps", 1000, "steps"),
            ("d_model", 32, "d_model"),
            ("typed_attention_rank", 16, "typed_attention_rank"),
            ("typed_attention_limit", 2.0, "typed_attention_limit"),
            ("metadata_dropout", 0.1, "metadata_dropout"),
            ("router_mode", "legacy-mean-linear", "router_mode"),
            ("label_smoothing", 0.0, "label_smoothing"),
            ("consistency_weight", 0.0, "consistency_weight"),
            ("typed_attention_mode", "off", "typed_attention_mode"),
            ("kind", "plain", "kind"),
            ("lr", 0.007, "grid"),
        ):
            def edit(seed, rows, field=field, value=value):
                for row in rows:
                    if seed == 17 and row["row"] == "meta-baseline-v1" and row["arm"] == "factorized-v2" and row["fold"] == FOLDS[2]:
                        row[field] = value
                return rows

            self.refuse(mutated(scenario, edit), text=text)

        def plain_with_smoothing(seed, rows):
            for row in rows:
                if row["row"] == "meta-baseline-v1" and row["arm"] == "plain":
                    row["kind"] = "a0"
            return rows

        self.refuse(mutated(scenario, plain_with_smoothing), text="kind")

        def foreign_arm(seed, rows):
            for row in rows:
                if seed == 17 and row["row"] == "meta-baseline-v1" and row["arm"] == "full":
                    row["arm"] = "latent-0"
            return rows

        self.refuse(mutated(scenario, foreign_arm), text="not in the study table")

        def wrong_study(seed, rows):
            for row in rows:
                if seed == 17 and row["row"] == "final-baseline-v1":
                    row["study"] = "m003-v2"
            return rows

        self.refuse(mutated(scenario, wrong_study), text="inside 'm002-v10'")

    def test_dataset_digest_and_unknown_folds_are_refused(self):
        scenario = supports_m002()

        def digest(seed, rows):
            for row in rows:
                if seed == 17 and row["row"] == "data-v2" and row["fold"] == FOLDS[0]:
                    row["data_fnv64"] = "0" * 16
            return rows

        self.refuse(mutated(scenario, digest), text="digest mismatch")

        def fold(seed, rows):
            for row in rows:
                if seed == 17 and row["row"] == "final-baseline-v1" and row["fold"] == FOLDS[0]:
                    row["fold"] = "evidence-other"
            return rows

        self.refuse(mutated(scenario, fold), text="unknown fold")

    def test_cross_seed_and_cross_arm_inconsistency_is_refused(self):
        scenario = supports_m002()

        def lr(seed, rows):
            for row in rows:
                if seed == 101 and row["row"] == "meta-baseline-v1" and row["arm"] == "plain":
                    row["lr"] = 0.0125
            return rows

        self.refuse(mutated(scenario, lr), text="learning rate differs")

        def tokens(seed, rows):
            for row in rows:
                if seed == 43 and row["row"] == "final-baseline-v1" and row["arm"] == "plain" and row["fold"] == FOLDS[0]:
                    row["explicit_token_count"] = 4999
            return rows

        self.refuse(mutated(scenario, tokens), text="explicit_token_count differs")

        def params(seed, rows):
            for row in rows:
                if seed == 29 and row["row"] == "meta-baseline-v1" and row["arm"] == "full":
                    row["total_params"] = 99
                if seed == 29 and row["row"] == "final-baseline-v1" and row["arm"] == "full":
                    row["total_params"] = 99
            return rows

        self.refuse(mutated(scenario, params), text="total_params differs from the first seed")

    def test_record_admission(self):
        scenario = supports_m002()
        sources = scenario.sources()
        for override, text in (
            ({"experiment_id": "M002-v9"}, "experiment_id"),
            ({"entrypoint": "other"}, "frozen entrypoint"),
            ({"exit_code": 1}, "did not complete"),
            ({"status": "failed"}, "did not complete"),
            ({"git_dirty": True}, "clean commit"),
            ({"manifest": {"id": "M002-v9"}}, "manifest identity"),
            ({"manifest": {"id": "M002-v10", "preregistration_sha256": "1"}}, "preregistration_sha256"),
        ):
            self.refuse([*sources[:4], ("x.json", scenario.record(101, **override))], text=text)
        broken = scenario.record(101)
        del broken["cargo_lock_sha256"]
        self.refuse([*sources[:4], ("x.json", broken)], text="cargo_lock_sha256")
        self.refuse([*sources[:4], ("x.json", scenario.record(101, git_sha="1" * 40))], text="disagree in git_sha")

    def test_bare_row_files_need_an_explicit_allowance_and_cannot_mix(self):
        scenario = supports_m002()
        files = [(f"rows-{seed}.jsonl", {"rows": scenario.rows(seed)}) for seed in SEEDS]
        self.refuse(files, text="bare row file")
        result = mod.decide(files, "M002-v10", CRITERIA, allow_rows=True)
        self.assertEqual(result["evidence_kind"], "row-file")
        self.assertEqual(result["headline"]["verdict"], "SUPPORTS")
        self.refuse([*files[:4], scenario.sources()[4]], allow_rows=True, text="cannot be mixed")

    def test_unknown_and_deferred_studies_are_refused(self):
        scenario = supports_m002()
        self.refuse(scenario.sources(), study="M004-v2", text="deferred")
        self.refuse(scenario.sources(), study="M009-v1", text="unknown study")

    def test_mask_share_rows_are_required_and_must_agree(self):
        scenario = Scenario("M001-v5")
        self.refuse(
            mutated(scenario, lambda seed, rows: [r for r in rows if r["row"] != "mask-share-baseline-v1"]),
            study="M001-v5", text="row missing",
        )

        def wrong(seed, rows):
            for row in rows:
                if seed == 17 and row["row"] == "mask-share-baseline-v1" and row["split"] == "test_ood":
                    row["mask_share"] += 0.1
            return rows

        self.refuse(mutated(scenario, wrong), study="M001-v5", text="not masked - unmasked")

        def inconsistent(seed, rows):
            for row in rows:
                if seed == 17 and row["row"] == "mask-share-baseline-v1" and row["split"] == "test_iid":
                    row["masked_accuracy"] = 0.0
            return rows

        self.refuse(mutated(scenario, inconsistent), study="M001-v5", text="masked_accuracy")


class Outputs(unittest.TestCase):
    def test_outputs_are_bound_immutable_and_parseable(self):
        scenario = supports_m002()
        scenario.level["factorized-v2"] = {"acc_ood": 0.67, "nll": 1.6, "ece": 0.08}
        result = scenario.decide()
        with tempfile.TemporaryDirectory() as directory:
            ci, decision = Path(directory) / "results/paired-ci-95.json", Path(directory) / "DECISION.toml"
            digest = mod.write_outputs(result, ci, decision)
            self.assertEqual(hashlib.sha256(ci.read_bytes()).hexdigest(), digest)
            parsed = tomllib.loads(decision.read_text(encoding="utf-8"))
            self.assertEqual(parsed["paired_ci_sha256"], digest)
            self.assertEqual(parsed["verdict"], "INCONCLUSIVE")
            self.assertEqual(parsed["label"], "NO-GO-CALIBRATION")
            self.assertFalse(parsed["go"])
            self.assertEqual(parsed["reporting"]["required_label"], "NO-GO-CALIBRATION")
            self.assertFalse(parsed["reporting"]["allow_quality_claim"])
            self.assertEqual([c["id"] for c in parsed["contrast"]], ["M002-v10-primary", "M002-v10-bundle"])
            first = parsed["contrast"][0]
            self.assertFalse(first["gates"]["ood_nll_each_fold"])
            self.assertEqual(first["failing_gates"], ["ood_nll_each_fold"])
            self.assertEqual(len(first["ood_accuracy"]["ci95"]), 2)
            self.assertEqual(len(parsed["source_record"]), 5)
            document = json.loads(ci.read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "complete")
            self.assertEqual(len(document["contrasts"][0]["endpoints"]), 12 * 4)
            self.assertEqual(document["provenance"]["criteria_sha256"], CRITERIA["sha256"])
            with self.assertRaises(mod.EvidenceError):
                mod.write_outputs(result, ci, decision)

    def test_cli_exit_codes_and_files(self):
        scenario = supports_m002()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed in SEEDS:
                path = root / f"run-seed-{seed}.json"
                path.write_text(json.dumps(scenario.record(seed)), encoding="utf-8")
                paths.append(str(path))
            out, err = io.StringIO(), io.StringIO()
            args = ["--study", "M002-v10", "--paired-ci", str(root / "ci.json"), "--decision", str(root / "D.toml")]
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(mod.main([*args, *paths]), 0)
            self.assertEqual(json.loads(out.getvalue())["headline"]["verdict"], "SUPPORTS")
            self.assertTrue((root / "D.toml").exists())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                self.assertEqual(mod.main([*args, *paths]), 2)  # outputs are immutable
                self.assertEqual(mod.main(["--study", "M002-v10", *paths[:3]]), 2)  # missing seeds
                self.assertEqual(mod.main(["--study", "M004-v2", *paths]), 2)  # deferred
                self.assertEqual(mod.main(["--study", "M002-v10", "--criteria-sha256", "0" * 64, *paths]), 2)
                self.assertEqual(mod.main(["--study", "M002-v10", "--paired-ci", str(root / "only.json"), *paths]), 2)
            self.assertIn("immutable", err.getvalue())

    def test_cli_row_files_and_lr_file(self):
        scenario = supports_m002()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed in SEEDS:
                path = root / f"rows-{seed}.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in scenario.rows(seed)) + "\nPRED a b c\n", encoding="utf-8")
                paths.append(str(path))
            lr = root / "lr.tsv"
            lr.write_text(
                "# comment\narm\tlr\tval_accuracy\tval_nll\n"
                + "".join(f"{arm}\t0.005\t0.9\t0.5\n" for arm in scenario.study["arms"]),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(mod.main(["--study", "M002-v10", *paths]), 2)
                self.assertEqual(mod.main(["--study", "M002-v10", "--allow-row-files", *paths]), 0)
                self.assertEqual(mod.main(["--study", "M002-v10", "--allow-row-files", "--lr-file", str(lr), *paths]), 0)
                lr.write_text(lr.read_text(encoding="utf-8").replace("plain\t0.005", "plain\t0.0125"), encoding="utf-8")
                self.assertEqual(mod.main(["--study", "M002-v10", "--allow-row-files", "--lr-file", str(lr), *paths]), 2)
                lr.write_text("full\t0.005\n", encoding="utf-8")
                self.assertEqual(mod.main(["--study", "M002-v10", "--allow-row-files", "--lr-file", str(lr), *paths]), 2)

    def test_lr_file_parser_matches_the_runner_rules(self):
        self.assertEqual(mod.parse_lr_file("# c\narm\tlr\nfull\t0.005\textra\n"), {"full": 0.005})
        for bad in ("full\t0.005\nfull\t0.01\n", "full\t-1\n", "full\tnan\n", "full\n", "full\tabc\n"):
            with self.assertRaises(mod.EvidenceError, msg=bad):
                mod.parse_lr_file(bad)

    def test_publish_results_binds_metrics_and_run(self):
        scenario = supports_m002()
        result = scenario.decide()
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            mod.publish_results(result, results)
            run = json.loads((results / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["metrics_sha256"], hashlib.sha256((results / "metrics.json").read_bytes()).hexdigest())
            self.assertEqual(run["verdict"], "SUPPORTS")
            self.assertEqual(run["git_sha"], "a" * 40)
        rows_only = mod.decide(
            [(f"r{seed}", {"rows": scenario.rows(seed)}) for seed in SEEDS], "M002-v10", CRITERIA, allow_rows=True
        )
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(mod.EvidenceError):
            mod.publish_results(rows_only, Path(directory))

    def test_toml_writer_round_trips_awkward_values(self):
        lines: list[str] = []
        value = {"s": 'quote " back \\ line\nä', "f": 1e-05, "i": 3, "b": False, "l": [0.5, 1.5], "n": None}
        mod.toml_lines("", value, lines)
        parsed = tomllib.loads("\n".join(lines))
        self.assertEqual(parsed, {k: v for k, v in value.items() if v is not None})
        with self.assertRaises(mod.EvidenceError):
            mod.toml_scalar(float("nan"))


if __name__ == "__main__":
    unittest.main()
