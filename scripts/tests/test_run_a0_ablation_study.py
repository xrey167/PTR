"""The A0 study driver's budget rule and learning-rate selection, which decide
what runs, checked against the design's own worked numbers."""

import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_a0_ablation_study", ROOT / "scripts/run_a0_ablation_study.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class BudgetRule(unittest.TestCase):
    def test_projection_includes_one_rerun_and_overhead_for_every_arm_run(self):
        """Verify runtime arithmetic independently with a small synthetic design."""
        rules = {"budget": {"workers": 2, "per_process_overhead_seconds": 3},
                 "seeds": {"declared": [17, 29]}, "learning_rate": {"grid": [0.01, 0.02]}}
        with patch.object(driver, "config", return_value=rules):
            # Two arms, two seeds and one rerun: 5 * (100 * .01 + 3) = 20s.
            # Four sweep arm-runs: 4 * (50 * .01 + 3) = 14s.
            self.assertAlmostEqual(driver.projected_minutes({"M001": ["full", "ablated"]},
                                                           100, 50, 10.0), 34 / 120)
            self.assertAlmostEqual(driver.projected_minutes({"M001": ["full", "ablated"]},
                                                           100, None, 10.0), 20 / 120)

    def test_projection_exactly_at_limit_keeps_all_arms(self):
        """The budget ceiling is inclusive; equality must not trigger a removal."""
        rules = driver.config()
        projected = driver.projected_minutes(driver.plan_arms([]), 2000, 2000, 15.0)
        rules["budget"]["limit_minutes"] = projected
        with patch.object(driver, "config", return_value=rules):
            plan = driver.budget(2000, 15.0)
        self.assertEqual(plan["ladder_applied"], [])
        self.assertTrue(plan["within_limit"])
        self.assertEqual(plan["projected_minutes"], projected)

    def test_fast_enough_runs_every_arm(self):
        """Keep every arm when projected runtime already fits the budget."""
        plan = driver.budget(2000, 15.0)
        self.assertEqual(plan["ladder_applied"], [])
        self.assertEqual(sum(len(v) for v in plan["arms"].values()), 12)

    def test_the_design_worked_example_drops_latent_4_then_the_pair(self):
        """Reproduce the design's ordered arm removals at its worked timing estimate."""
        # The design: at 20 ms/step, 97 arm-runs take about 23 minutes; dropping
        # latent-4 leaves 89 arm-runs, about 21 -- still over 20, so the pair goes too.
        plan = driver.budget(2000, 20.0)
        self.assertEqual(plan["ladder_applied"], ["drop latent-4", "drop the blind-query pair"])
        self.assertNotIn("latent-4", plan["arms"]["M003"])
        self.assertTrue(plan["within_limit"])

    def test_each_step_is_applied_only_when_still_needed(self):
        """Stop the budget ladder as soon as its first removal meets the limit."""
        plan = driver.budget(2000, 17.4)
        self.assertEqual(plan["ladder_applied"], ["drop latent-4"])
        self.assertIn("blind-query-k0", plan["arms"]["M002"])
        self.assertTrue(plan["within_limit"])

    def test_a_slow_host_skips_the_sweep_and_reports_the_cap(self):
        """Skip the sweep and report an exceeded limit when arm removals remain insufficient."""
        plan = driver.budget(2000, 60.0)
        self.assertFalse(plan["sweep"])
        self.assertFalse(plan["within_limit"])


GRID = [0.002, 0.005, 0.0125]


def sweep(**by_lr):
    """{lr: sweep row} from keyword pairs such as lr_0_005=(0.80, False)."""
    return {float(k[3:].replace("_", ".")): {"val_accuracy": acc, "nan": nan} for k, (acc, nan) in by_lr.items()}


class LearningRateSelection(unittest.TestCase):
    def test_tolerance_boundary_is_inclusive_and_order_independent(self):
        """Use binary-exact scores to distinguish equality from just outside tolerance."""
        by_lr = {0.0125: {"val_accuracy": 0.875, "nan": False},
                 0.005: {"val_accuracy": 0.75, "nan": False},
                 0.002: {"val_accuracy": 0.5, "nan": False}}
        lr, flag, _ = driver.choose_lr("full", by_lr, list(reversed(GRID)), 0.125)
        self.assertEqual((lr, flag), (0.005, ""))
        by_lr[0.005]["val_accuracy"] = math.nextafter(0.75, 0.0)
        lr, flag, _ = driver.choose_lr("full", by_lr, GRID, 0.125)
        self.assertEqual((lr, flag), (0.0125, "edge of grid"))

    def test_tied_rates_choose_the_smallest_and_infinities_are_ineligible(self):
        """Ties are deterministic and either infinity is excluded even without a NaN flag."""
        by_lr = {lr: {"val_accuracy": 0.75, "nan": False} for lr in reversed(GRID)}
        self.assertEqual(driver.choose_lr("full", by_lr, GRID, 0.0)[:2], (0.002, "edge of grid"))
        by_lr[0.002]["val_accuracy"] = float("inf")
        by_lr[0.0125]["val_accuracy"] = -float("inf")
        lr, flag, eligible = driver.choose_lr("full", by_lr, GRID, 0.0)
        self.assertEqual((lr, flag, eligible), (0.005, "", {0.005: 0.75}))

    def test_the_smallest_rate_within_the_tolerance_of_the_best_wins(self):
        """Choose the smallest eligible learning rate within tolerance of peak accuracy."""
        lr, flag, eligible = driver.choose_lr("a", sweep(lr_0_002=(0.70, False), lr_0_005=(0.795, False), lr_0_0125=(0.80, False)), GRID, 0.005)
        self.assertEqual((lr, flag), (0.005, ""))
        self.assertEqual(eligible[0.0125], 0.80)

    def test_outside_the_tolerance_the_best_wins_and_an_edge_is_flagged(self):
        """Select a uniquely best boundary rate and flag its position at the grid edge."""
        lr, flag, _ = driver.choose_lr("a", sweep(lr_0_002=(0.70, False), lr_0_005=(0.79, False), lr_0_0125=(0.80, False)), GRID, 0.005)
        self.assertEqual((lr, flag), (0.0125, "edge of grid"))

    def test_a_diverged_rate_is_never_chosen(self):
        """Exclude flagged divergence and non-finite validation accuracy from rate selection."""
        lr, _, eligible = driver.choose_lr("a", sweep(lr_0_002=(0.70, False), lr_0_005=(0.90, True), lr_0_0125=(float("nan"), False)), GRID, 0.005)
        self.assertEqual(lr, 0.002)
        self.assertEqual(sorted(eligible), [0.002])

    def test_a_missing_or_wholly_diverged_sweep_stops_the_study(self):
        """Terminate selection when grid results are missing or every rate diverged."""
        with self.assertRaises(SystemExit):
            driver.choose_lr("a", sweep(lr_0_002=(0.70, False)), GRID, 0.005)
        with self.assertRaises(SystemExit):
            driver.choose_lr("a", sweep(lr_0_002=(0.7, True), lr_0_005=(0.7, True), lr_0_0125=(0.7, True)), GRID, 0.005)


if __name__ == "__main__":
    unittest.main()
