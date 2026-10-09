"""Torch-free tests: the harness's rules agree with criteria.toml, and the plan covers every hard counter."""

import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/consciousness/C001-global-workspace"
sys.path.insert(0, str(ROOT / "research/consciousness/gw"))
import rules  # noqa: E402

CONFIG = tomllib.loads((EXP / "config.toml").read_text(encoding="utf-8"))
CRITERIA = tomllib.loads((EXP / "criteria.toml").read_text(encoding="utf-8"))
PLAN = tomllib.loads((EXP / "tests/mutations.toml").read_text(encoding="utf-8"))
PINNED = CONFIG["pinned_thresholds"]


class RulesAgreeWithCriteria(unittest.TestCase):
    def test_coverage_rules_equal_the_criteria_in_order(self):
        self.assertEqual(rules.coverage_rules(PINNED), CRITERIA["coverage"])

    def test_guard_rules_equal_the_criteria_in_order(self):
        self.assertEqual(rules.guard_rules(PINNED), CRITERIA["guards"])

    def test_falsifier_rules_equal_the_criteria_in_order(self):
        self.assertEqual(rules.falsifier_rules(), CRITERIA["falsifiers"])

    def test_every_hard_counter_is_expected_by_a_mutation(self):
        expected = {name for m in PLAN["mutation"] for name in m["expect"]}
        self.assertEqual(sorted(set(rules.HARD_COUNTERS) - expected), [])

    def test_every_negative_control_is_expected_by_a_mutation(self):
        expected = {name for m in PLAN["mutation"] for name in m["expect"]}
        for control in ("n_leak", "overlap", "param_mismatch", "isolation_probe", "bottleneck", "dropout_probe"):
            self.assertIn("control:" + control, expected)

    def test_a_defect_of_the_workspace_arm_is_expected_to_reach_a_counter_not_exit_two(self):
        by_name = {m["name"]: m for m in PLAN["mutation"]}
        self.assertEqual(by_name["workspace-arm-does-not-broadcast"]["expect"], ["workspace_influence_hits"])
        self.assertEqual(by_name["topk-write-not-applied"]["expect"], ["bottleneck_violations"])

    def test_plan_hard_counters_start_with_the_harness_ones(self):
        self.assertEqual(PLAN["hard_counters"][: len(rules.HARD_COUNTERS)], list(rules.HARD_COUNTERS))

    def test_the_preregistration_table_restates_the_pinned_values_when_present(self):
        if "preregistration" not in CONFIG:
            self.skipTest("the [preregistration] table is added at the freeze")
        table = CONFIG["preregistration"]
        for key, value in table.items():
            if key in PINNED:
                self.assertEqual(value, PINNED[key], key)
        self.assertEqual(table["harness"], "c001")
        self.assertEqual(table["noise_milli"], round(PINNED["noise"] * 1000))
        lr = {k: round(v * 1_000_000) for k, v in PINNED["arm_lr"].items()}
        self.assertEqual(table["arm_lr_micro"], ",".join(f"{k}={v}" for k, v in lr.items()))
        required = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"]["C001"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))

    def test_the_ood_module_counts_are_the_harness_constants(self):
        import ast
        source = (ROOT / "research/consciousness/gw/harness.py").read_text(encoding="utf-8")
        constant = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "OOD_MODULES")
        self.assertEqual(list(constant), PINNED["ood_modules"])
        self.assertGreater(min(PINNED["ood_modules"]), PINNED["n_train_modules"])

    def test_the_arm_learning_rates_name_every_arm(self):
        sys.path.insert(0, str(ROOT / "research/consciousness/gw"))
        arms = {"workspace", "all_to_all", "bw_all_to_all", "no_broadcast", "no_comm", "competition_off", "capacity_off", "crippled_all_to_all"}
        self.assertEqual(set(PINNED["arm_lr"]), arms)
        self.assertEqual(PINNED["min_arms_trained"], len(arms))

    def test_judge_reports_the_failing_rule(self):
        result = {rule.split(" ")[0]: int(rule.split(" ")[2]) for rule in rules.coverage_rules(PINNED)}
        result.update({counter: 0 for counter in rules.HARD_COUNTERS})
        self.assertEqual(rules.judge(result, PINNED), ([], []))
        result["n_leaks"] = 1
        result["workspace_influence_hits"] = 0
        coverage, falsifiers = rules.judge(result, PINNED)
        self.assertEqual(coverage, ["workspace_influence_hits >= 1"])
        self.assertEqual(falsifiers, ["n_leaks <= 0"])


if __name__ == "__main__":
    unittest.main()
