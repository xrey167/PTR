"""The A0 study driver's budget rule and learning-rate selection, which decide
what runs, checked against the design's own worked numbers."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("run_a0_ablation_study", ROOT / "scripts/run_a0_ablation_study.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class BudgetRule(unittest.TestCase):
    def test_fast_enough_runs_every_arm(self):
        plan = driver.budget(2000, 15.0)
        self.assertEqual(plan["ladder_applied"], [])
        self.assertEqual(sum(len(v) for v in plan["arms"].values()), 12)

    def test_the_design_worked_example_drops_latent_4_then_the_pair(self):
        # The design: at 20 ms/step, 97 arm-runs take about 23 minutes; dropping
        # latent-4 leaves 89 arm-runs, about 21 -- still over 20, so the pair goes too.
        plan = driver.budget(2000, 20.0)
        self.assertEqual(plan["ladder_applied"], ["drop latent-4", "drop the blind-query pair"])
        self.assertNotIn("latent-4", plan["arms"]["M003"])
        self.assertTrue(plan["within_limit"])

    def test_each_step_is_applied_only_when_still_needed(self):
        plan = driver.budget(2000, 17.4)
        self.assertEqual(plan["ladder_applied"], ["drop latent-4"])
        self.assertIn("blind-query-k0", plan["arms"]["M002"])
        self.assertTrue(plan["within_limit"])

    def test_a_slow_host_skips_the_sweep_and_reports_the_cap(self):
        plan = driver.budget(2000, 60.0)
        self.assertFalse(plan["sweep"])
        self.assertFalse(plan["within_limit"])


if __name__ == "__main__":
    unittest.main()
