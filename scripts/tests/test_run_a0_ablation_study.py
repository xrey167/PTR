"""The A0 study driver's budget rule and learning-rate selection, which decide
what runs, checked against the design's own worked numbers."""

import importlib.util
import copy
import math
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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


class CorrectnessWriter(unittest.TestCase):
    def test_writer_binds_all_checks_before_publishing_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            def clean():
                self.assertEqual(list(study.glob("logs/g6-*.log")), [])
                return True
            with patch.multiple(driver, STUDY_DIR=study, worktree_clean=clean,
                                data_fnv=Mock(return_value="aa"),
                                run=Mock(return_value=subprocess.CompletedProcess([], 0, stdout="tests passed\n", stderr=""))), \
                    patch.object(driver.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="c" * 40, stderr="")):
                outcome = driver.correctness()
            evidence = json.loads((study / "logs/g6.json").read_text())
            self.assertEqual(set(outcome), driver.aggregator().G6_CHECKS)
            self.assertTrue(all(value == "pass" for value in outcome.values()))
            self.assertTrue(driver.aggregator().correctness_evidence(evidence, "c" * 40, study / "logs", "aa")["pass"])

    def test_eval_publishes_correctness_logs_only_after_model_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "budget.json").write_text(json.dumps({"arms": {e: ["full"] for e in driver.EXPERIMENTS}}))
            (study / "logs").mkdir()
            (study / "logs/g6.json").write_text("original")
            def checks(destination):
                (destination / "g6.json").write_text("new evidence")
                return {"check": "pass"}
            def launch(jobs, workers):
                self.assertEqual((study / "logs/g6.json").read_text(), "original")
                return [0] * len(jobs)
            with patch.multiple(driver, STUDY_DIR=study, require_eval_commit=Mock(), must=Mock(),
                                correctness=checks, parallel=launch,
                                config=Mock(return_value={"seeds": {"declared": [17]}, "budget": {"workers": 1}})):
                driver.eval_phase(None)
            self.assertEqual((study / "logs/g6.json").read_text(), "new evidence")

    def test_source_changes_during_checks_prevent_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(driver, STUDY_DIR=Path(directory), worktree_clean=Mock(side_effect=[True, False]),
                                data_fnv=Mock(return_value="aa"),
                                run=Mock(return_value=subprocess.CompletedProcess([], 0, stdout="passed", stderr=""))), \
                    patch.object(driver.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="c" * 40, stderr="")):
                self.assertEqual(driver.correctness()["source_provenance"], "fail")



class FrozenSweepEvidence(unittest.TestCase):
    def setUp(self):
        self.archived = {e: driver.records(e, "a0_sweep_entrypoint") for e in driver.EXPERIMENTS}

    def test_eval_rates_and_hosts_are_bound_to_frozen_selection(self):
        aggregate = driver.aggregator()
        records = {str(path): value for path in ROOT.glob("experiments/model/*/results/run-*.json")
                   if (value := json.loads(path.read_text())).get("entrypoint") == "a0_ablation_entrypoint"
                   and value.get("status") == "completed"}
        revision = next(iter(records.values()))["git_sha"]
        self.assertTrue(aggregate.sweep_evidence(revision, records)["pass"])
        name = next(iter(records))
        for field, value in [("host", None), ("host", {**records[name]["host"], "cpu_model": "other CPU"}),
                             ("hardware_profile", "hardware/default.toml")]:
            invalid = copy.deepcopy(records)
            invalid[name][field] = value
            self.assertFalse(aggregate.sweep_evidence(revision, invalid)["pass"])
        invalid = copy.deepcopy(records)
        rows = driver.rows(invalid[name]["stdout"])
        meta = next(row for row in rows if row.get("row") == "meta")
        meta["lr"] = 123.0
        invalid[name]["stdout"] = "\n".join(json.dumps(row) for row in rows)
        self.assertFalse(aggregate.sweep_evidence(revision, invalid)["pass"])
        original = aggregate.git
        def changed_tsv(*args):
            result = original(*args)
            return result.replace("0.0125", "0.1") if args[-1].endswith("lr_selection.tsv") else result
        with patch.object(aggregate, "git", side_effect=changed_tsv):
            self.assertFalse(aggregate.sweep_evidence(revision, records)["pass"])

    def test_archived_selection_reproduces_committed_rates(self):
        expected = json.loads((driver.STUDY_DIR / "lr_selection.json").read_text())
        self.assertEqual(driver.selection_table(), expected)

    def test_invalid_or_duplicate_sweeps_cannot_choose_rates(self):
        mutations = {
            "commit": lambda r: r.update(git_sha="f" * 40),
            "dirty": lambda r: r.update(git_dirty=True),
            "experiment": lambda r: r.update(experiment_id="M004"),
            "seed": lambda r: r.update(seed=29),
            "manifest": lambda r: r.update(manifest_sha256="f" * 64),
            "host": lambda r: r["host"].update(logical_cpus=999),
            "width": lambda r: r["command"].__setitem__(-1, "32"),
            "steps": lambda r: r["command"].__setitem__(r["command"].index("--steps") + 1, "100"),
            "grid": lambda r: r.update(parameters={"lr": "0.9"}),
            "data": lambda r: r.update(stdout=r["stdout"].replace('"data_fnv64":"0ad71688f09b0d0d"', '"data_fnv64":"bad"')),
            "profile": lambda r: r["hardware_profile_record"].update(sha256="f" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                inputs = copy.deepcopy(self.archived)
                mutate(inputs["M001"][0])
                with patch.object(driver, "records", side_effect=lambda e, _: inputs[e]):
                    with self.assertRaisesRegex(ValueError, "invalid frozen sweep"):
                        driver.selection_table()
        inputs = copy.deepcopy(self.archived)
        inputs["M001"].append(copy.deepcopy(inputs["M001"][0]))
        with patch.object(driver, "records", side_effect=lambda e, _: inputs[e]):
            with self.assertRaisesRegex(ValueError, "duplicate arm/rate"):
                driver.selection_table()

    def test_invalid_selection_writes_no_artifacts(self):
        with patch.object(driver, "selection_table", side_effect=ValueError("bad sweep")), patch.object(driver, "write_json") as write:
            with self.assertRaisesRegex(ValueError, "bad sweep"):
                driver.select(None)
            write.assert_not_called()

if __name__ == "__main__":
    unittest.main()
