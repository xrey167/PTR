"""The S003 aggregator's verdicts, on harness results built to show each one.

`aggregate.py` reads run records and mutation evidence and writes the
aggregate; what it concludes from them is `analyse`, which reads no file, and
the pilot's classification of cells, which is `pilot_classification`. These
tests give both results shaped to meet, or to miss, each preregistered
condition, so that a verdict that stopped depending on one fails here."""

import contextlib
import importlib.util
import io
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "semdb" / "S003-certified-branches"

spec = importlib.util.spec_from_file_location("s003_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

TABLE = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))["preregistration"]
AGENTS = TABLE["agents"]
LEVELS = len(TABLE["groups_ladder"])
ALL_CELLS = [aggregate.cell_name(level, agents) for level in range(LEVELS) for agents in AGENTS]
SERIAL_TICKS = 1000


def run(arm: str, agents: int, ticks: int, attempts: int = 128, conflicts: int = 0, lifecycle: int = 0) -> dict:
    return {
        "arm": arm,
        "agents": agents,
        "ticks": ticks,
        "attempts": attempts,
        "merged": attempts - conflicts - lifecycle,
        "no_change": 0,
        "conflicts": conflicts,
        "lifecycle_refusals": lifecycle,
        "verification_holds": 0,
        "escalations": 10 if arm == "certified-review" else 0,
        "review_voids": 8 if arm == "certified-review" else 0,
        "abandoned": 0,
        "wasted_ticks": 0,
        "unnecessary_refusals": 0,
        "complete": True,
    }


def make_case(index: int, ticks_of, conflicts_of) -> dict:
    level = index % LEVELS
    runs = [run("serial", 1, SERIAL_TICKS)]
    for agents in AGENTS:
        ticks = ticks_of(index, level, agents)
        runs.append(run("certified", agents, ticks, conflicts=conflicts_of(index, level, agents)))
        runs.append(run("certified-review", agents, ticks * 3))
        runs.append(run("lww", agents, ticks))
        runs.append(run("occ", agents, ticks))
    return {"case": index, "level": level, "groups": TABLE["groups_ladder"][level], "serial_ticks": SERIAL_TICKS, "runs": runs}


def seed_result(seed: int, ticks_of=None, conflicts_of=None, **overrides) -> dict:
    """A harness result for one seed: by default every certified run takes
    a little over a tenth longer than a perfect split of the serial run over
    its agents, and nothing conflicts."""
    ticks_of = ticks_of or (lambda index, level, agents: round(SERIAL_TICKS / agents * 1.1))
    conflicts_of = conflicts_of or (lambda index, level, agents: 0)
    result = {
        "benchmark": "certified-branches",
        "iterations": TABLE["cases_per_seed"],
        "seed": seed,
        "server": "in-memory",
        "preregistration": "text",
        "elapsed_ns": 1,
        "hard_failures": 0,
        "merge_wall_ns": 1_000_000_000,
        "merge_calls": 10_000,
        "merge_wall_le_10us": 0,
        "merge_wall_le_100us": 5_000,
        "merge_wall_le_1ms": 4_000,
        "merge_wall_le_10ms": 1_000,
        "merge_wall_gt_10ms": 0,
        "unnecessary_refusals": 5,
        "put_increment_conflicts": 3,
        "cases": [make_case(index, ticks_of, conflicts_of) for index in range(TABLE["cases_per_seed"])],
    }
    for name in aggregate.HARD:
        result[name] = 0
    for name in aggregate.COVERAGE:
        result[name] = 3
    for name in aggregate.HAZARDS:
        result[name] = 40
    # The fixed probes' repetitions and what the workload made of the two
    # rule classes: reported, and no gate.
    for name in aggregate.PROBE_HAZARDS:
        result[name] = 48
    result["hazard_negative"] = 0
    result["hazard_set_member"] = 0
    for name in ("occ_lost_updates", "occ_stale_scan_commits"):
        result[name] = 0
    result.update(overrides)
    return result


def records(seeds: list[int]) -> list[dict]:
    return [{"seed": seed, "record": f"run-seed-{seed}.json", "git_sha": "0" * 40, "exit_code": 0} for seed in seeds]


def analyse(results: list[dict], low_cells=None, mutations="all killed", exit_codes=None, **table_changes) -> dict:
    table = {**TABLE, "low_cells": ALL_CELLS if low_cells is None else low_cells, **table_changes}
    if mutations == "all killed":
        mutations = {"killed": 33, "total": 33, "git_sha": "0" * 40, "sha256": "0" * 64}
    held = records([result["seed"] for result in results])
    for record, code in zip(held, exit_codes or [0] * len(held)):
        record["exit_code"] = code
    return aggregate.analyse(results, held, table, mutations)


def five(**kwargs) -> list[dict]:
    return [seed_result(seed, **kwargs) for seed in TABLE["seeds"]]


class VerdictTests(unittest.TestCase):
    def test_a_run_that_meets_every_condition_and_gains_over_serial_completes(self):
        analysis = analyse(five())
        self.assertEqual(analysis["conditions"], {name: True for name in analysis["conditions"]})
        self.assertEqual(analysis["throughput"], "supported")
        self.assertEqual(analysis["efficiency"], "met")
        self.assertEqual(analysis["recommended"], "completed")
        for agents in AGENTS:
            entry = analysis["metrics"]["per_agent_count"][str(agents)]
            self.assertGreater(entry["bootstrap"]["lower"], 1.0)
            self.assertEqual(entry["bootstrap"]["cases"], 5 * TABLE["cases_per_seed"] // LEVELS * LEVELS)

    def test_no_gain_over_serial_execution_rejects_the_default(self):
        slower = lambda index, level, agents: round(SERIAL_TICKS * 1.2)
        analysis = analyse(five(ticks_of=slower))
        self.assertEqual(analysis["throughput"], "rejected")
        self.assertEqual(analysis["efficiency"], "missed")
        self.assertEqual(analysis["recommended"], "failed")
        self.assertTrue(analysis["conditions"]["hard_pass"], "a rejected default is not a safety failure")

    def test_an_interval_that_straddles_one_is_inconclusive_and_recorded_as_failed(self):
        # Half the cases twice as fast as serial, half as slow as half of it: a gain of one.
        mixed = lambda index, level, agents: 500 if index % 2 == 0 else 1500
        analysis = analyse(five(ticks_of=mixed))
        for agents in AGENTS:
            bootstrap = analysis["metrics"]["per_agent_count"][str(agents)]["bootstrap"]
            self.assertLess(bootstrap["lower"], 1.0)
            self.assertGreater(bootstrap["upper"], 1.0)
        self.assertEqual(analysis["throughput"], "inconclusive")
        self.assertEqual(analysis["recommended"], "failed")

    def test_the_gain_needs_every_agent_count_not_the_average(self):
        # N = 16 is no faster than serial; the others gain.
        ticks = lambda index, level, agents: round(SERIAL_TICKS * 1.2) if agents == 16 else round(SERIAL_TICKS / agents * 1.1)
        analysis = analyse(five(ticks_of=ticks))
        self.assertEqual(analysis["throughput"], "rejected")

    def test_a_hard_counter_or_a_failed_exit_is_no_hard_pass_whatever_the_gain(self):
        for change in ({"lost_updates": 1}, {"provenance_mismatches": 2}, {"harness_errors": 1}, {"hard_failures": 1}):
            with self.subTest(change=change):
                results = five()
                results[2].update(change)
                analysis = analyse(results)
                self.assertFalse(analysis["conditions"]["hard_pass"])
                self.assertEqual(analysis["throughput"], "inconclusive")
                self.assertEqual(analysis["recommended"], "failed")
        analysis = analyse(five(), exit_codes=[0, 0, 1, 0, 0])
        self.assertFalse(analysis["conditions"]["hard_pass"])
        self.assertEqual(analysis["recommended"], "failed")

    def test_a_coverage_counter_at_zero_in_one_seed_is_not_coverage(self):
        for name in ("reviewed_merges", "probe_p26_exercised", "durable_roundtrips", "occ_stale_input_commits"):
            with self.subTest(counter=name):
                results = five()
                results[3][name] = 0
                analysis = analyse(results)
                self.assertFalse(analysis["conditions"]["coverage_ok"])
                self.assertFalse(analysis["metrics"]["coverage"][f"seed {results[3]['seed']}: {name}"])
                self.assertEqual(analysis["recommended"], "failed")

    def test_a_hazard_class_with_too_few_trials_in_one_seed_is_not_coverage(self):
        minimum = TABLE["min_hazard_trials_per_class_per_seed"]
        results = five()
        results[0]["hazard_read"] = minimum - 1
        self.assertFalse(analyse(results)["conditions"]["coverage_ok"])
        results[0]["hazard_read"] = minimum
        self.assertTrue(analyse(results)["conditions"]["coverage_ok"])

    def test_the_fixed_probes_repetitions_are_no_trials_of_any_class(self):
        # Every seed's probes ran each class 48 times, yet a class the
        # workload's own merges never ran into has no trials.
        results = five()
        results[0]["hazard_write"] = 0
        analysis = analyse(results)
        self.assertFalse(analysis["conditions"]["coverage_ok"])
        self.assertEqual(analysis["metrics"]["descriptive"]["probe_hazard_repetitions"]["probe_hazard_write"], 240)

    def test_the_two_rule_classes_are_covered_by_their_probes_and_gate_nothing_else(self):
        results = five()
        for result in results:
            result["hazard_negative"] = 0
            result["hazard_set_member"] = 0
        analysis = analyse(results)
        self.assertTrue(analysis["conditions"]["coverage_ok"])
        reported = analysis["metrics"]["descriptive"]["rule_class_trials"]
        self.assertEqual(reported["hazard_negative"]["workload_trials"], 0)
        self.assertEqual(reported["hazard_negative"]["probe_repetitions"], 240)
        self.assertIn("P25", reported["hazard_negative"]["evidence"])
        self.assertIn("P26", reported["hazard_set_member"]["evidence"])
        for name in ("probe_p25_exercised", "probe_p26_exercised"):
            results[2][name] = 0
            self.assertFalse(analyse(results)["conditions"]["coverage_ok"], name)
            results[2][name] = 3

    def test_a_run_that_stopped_on_an_error_reads_no_throughput_and_the_aggregate_still_publishes(self):
        results = five()
        for index, case in enumerate(results[0]["cases"]):
            for run_ in case["runs"]:
                if run_["arm"] == "certified" and index % 2 == 0:
                    run_["complete"] = False
                    run_["ticks"] = 0
        analysis = analyse(results)
        self.assertFalse(analysis["conditions"]["runs_complete"])
        self.assertEqual(analysis["throughput"], "inconclusive")
        self.assertEqual(analysis["recommended"], "failed")
        self.assertTrue(analysis["metrics"]["incomplete_runs"])
        for entry in analysis["metrics"]["per_agent_count"].values():
            self.assertNotIn("bootstrap", entry)

    def test_every_run_of_a_cell_stopping_early_does_not_divide_by_zero(self):
        results = [
            seed_result(
                seed,
                ticks_of=lambda index, level, agents: 0,
            )
            for seed in TABLE["seeds"]
        ]
        for result in results:
            for case in result["cases"]:
                for run_ in case["runs"]:
                    run_["complete"] = False
        analysis = analyse(results)
        self.assertEqual(analysis["recommended"], "failed")
        self.assertIsNone(aggregate.cells(results)["L2N4"]["gain"])

    def test_an_unusable_run_with_no_error_is_no_gain_either(self):
        # A run that ended with a zero makespan measures nothing.
        results = five()
        results[1]["cases"][0]["runs"][1]["ticks"] = 0
        analysis = analyse(results)
        self.assertFalse(analysis["conditions"]["runs_complete"])
        self.assertEqual(analysis["recommended"], "failed")

    def test_the_time_model_needs_the_99th_percentile_merge_within_the_budget(self):
        slow = {"merge_wall_le_10ms": 800, "merge_wall_gt_10ms": 200}
        analysis = analyse([seed_result(seed, **slow) for seed in TABLE["seeds"]])
        self.assertFalse(analysis["conditions"]["time_model_ok"])
        self.assertEqual(analysis["metrics"]["time_model"]["p99_bucket"], "merge_wall_gt_10ms")
        self.assertEqual(analysis["recommended"], "failed")
        # One in a thousand over ten milliseconds leaves the 99th percentile inside the budget.
        rare = {"merge_wall_le_10ms": 999, "merge_wall_gt_10ms": 1}
        analysis = analyse([seed_result(seed, **rare) for seed in TABLE["seeds"]])
        self.assertTrue(analysis["conditions"]["time_model_ok"])
        self.assertEqual(analysis["metrics"]["time_model"]["p99_upper_bound_us"], 10_000)

    def test_the_merge_time_bucket_of_the_percentile_is_read_from_the_summed_histogram(self):
        results = [
            {"merge_calls": 200, "merge_wall_ns": 0, "merge_wall_le_10us": 0, "merge_wall_le_100us": 150,
             "merge_wall_le_1ms": 50, "merge_wall_le_10ms": 0, "merge_wall_gt_10ms": 0},
            {"merge_calls": 200, "merge_wall_ns": 0, "merge_wall_le_10us": 0, "merge_wall_le_100us": 100,
             "merge_wall_le_1ms": 95, "merge_wall_le_10ms": 5, "merge_wall_gt_10ms": 0},
        ]
        summary = aggregate.merge_time(results, 10_000)
        self.assertEqual(summary["calls"], 400)
        self.assertEqual(summary["p99_bucket"], "merge_wall_le_10ms")
        self.assertTrue(summary["ok"])
        self.assertFalse(aggregate.merge_time(results, 1_000)["ok"])
        self.assertFalse(aggregate.merge_time([{**results[0], "merge_calls": 0}], 10_000)["ok"])

    def test_every_agent_count_needs_a_low_cell(self):
        without_sixteen = [name for name in ALL_CELLS if not name.endswith("N16")]
        analysis = analyse(five(), low_cells=without_sixteen)
        self.assertFalse(analysis["conditions"]["low_cells_ok"])
        self.assertEqual(analysis["throughput"], "inconclusive")
        self.assertFalse(analyse(five(), low_cells=[])["conditions"]["low_cells_ok"])
        self.assertFalse(analyse(five(), low_cells=[*ALL_CELLS, "L9N3"])["conditions"]["low_cells_ok"])

    def test_the_manipulation_check_needs_a_conflict_rate_below_the_threshold_over_each_agent_counts_low_cells(self):
        # Every certified run of N = 8 in level 0 conflicts on a quarter of its attempts.
        conflicts = lambda index, level, agents: 32 if (agents, level) == (8, 0) else 0
        analysis = analyse(five(conflicts_of=conflicts))
        # Diluted over the six low cells of N = 8, the rate stays below the threshold.
        self.assertLess(analysis["metrics"]["per_agent_count"]["8"]["conflict_rate"], TABLE["conflict_threshold_permille"] / 1000)
        self.assertTrue(analysis["conditions"]["manipulation_ok"])
        # Named alone with the other agent counts' cells, its agent count is far above it.
        alone = ["L0N2", "L0N4", "L0N8", "L0N16"]
        analysis = analyse(five(conflicts_of=conflicts), low_cells=alone)
        self.assertAlmostEqual(analysis["metrics"]["per_agent_count"]["8"]["conflict_rate"], 32 / 128)
        self.assertFalse(analysis["conditions"]["manipulation_ok"])
        heavy = lambda index, level, agents: 32 if agents == 8 else 0
        analysis = analyse(five(conflicts_of=heavy))
        self.assertFalse(analysis["metrics"]["per_agent_count"]["8"]["manipulation_ok"])
        self.assertFalse(analysis["conditions"]["manipulation_ok"])
        self.assertEqual(analysis["throughput"], "inconclusive")
        self.assertEqual(analysis["recommended"], "failed")

    def test_mutation_evidence_must_be_there_and_every_mutation_killed(self):
        self.assertFalse(analyse(five(), mutations=None)["conditions"]["mutations_ok"])
        survivor = {"killed": 32, "total": 33, "git_sha": "0" * 40, "sha256": "0" * 64}
        analysis = analyse(five(), mutations=survivor)
        self.assertFalse(analysis["conditions"]["mutations_ok"])
        self.assertEqual(analysis["recommended"], "failed")
        self.assertEqual(analysis["metrics"]["mutation_checks"], survivor)

    def test_a_missed_efficiency_floor_does_not_decide_the_status(self):
        # Gains of 1.1 at every agent count: above one, but far below N times 0.5 for N = 16.
        analysis = analyse(five(ticks_of=lambda index, level, agents: round(SERIAL_TICKS / 1.1)))
        self.assertEqual(analysis["throughput"], "supported")
        self.assertEqual(analysis["efficiency"], "missed")
        self.assertEqual(analysis["recommended"], "completed")

    def test_the_descriptive_shares_and_the_rule_of_three_are_reported(self):
        analysis = analyse(five())
        descriptive = analysis["metrics"]["descriptive"]
        self.assertEqual(descriptive["hazard_trials"]["hazard_write"]["trials"], 200)
        self.assertEqual(descriptive["hazard_trials"]["hazard_write"]["rule_of_three_bound"], round(3 / 200, 6))
        self.assertEqual(descriptive["review_void_share"], 0.8)
        self.assertEqual(set(descriptive["hazard_trials"]), set(aggregate.HAZARDS))
        self.assertIn("certified", descriptive["arms"])
        self.assertEqual(analysis["totals"]["cases"], 5 * TABLE["cases_per_seed"])


class CellTests(unittest.TestCase):
    def test_a_cell_pools_its_cases_and_names_its_conflict_rate_gain_and_efficiency(self):
        result = seed_result(17, conflicts_of=lambda index, level, agents: 8 if (level, agents) == (2, 4) else 0)
        table = aggregate.cells([result])
        cell = table["L2N4"]
        self.assertEqual(cell["cases"], TABLE["cases_per_seed"] // LEVELS)
        self.assertEqual(cell["attempts"], cell["cases"] * 128)
        self.assertAlmostEqual(cell["conflict_rate"], 8 / 128)
        self.assertAlmostEqual(cell["merge_rate"], 120 / 128)
        self.assertAlmostEqual(cell["gain"], SERIAL_TICKS / round(SERIAL_TICKS / 4 * 1.1))
        self.assertAlmostEqual(cell["efficiency"], cell["gain"] / 4)
        self.assertEqual(len(table), LEVELS * len(AGENTS))

    def test_a_conflict_is_a_conflict_or_a_lifecycle_change_and_nothing_else(self):
        runs = [run("certified", 2, 100, attempts=100, conflicts=3, lifecycle=2)]
        runs[0]["verification_holds"] = 7
        self.assertAlmostEqual(aggregate.conflict_rate(runs), 0.05)
        self.assertIsNone(aggregate.conflict_rate([]))


class PilotTests(unittest.TestCase):
    def pilot(self, high) -> list[dict]:
        """Three pilot seeds where the cells `high` conflict on a fifth of their attempts."""
        conflicts = lambda index, level, agents: 25 if aggregate.cell_name(level, agents) in high else 0
        return [seed_result(seed, conflicts_of=conflicts) for seed in TABLE["pilot_seeds"]]

    def test_the_cells_below_the_threshold_are_low_and_the_minimums_are_checked(self):
        high = [aggregate.cell_name(level, agents) for level in range(3) for agents in (8, 16)]
        classification = aggregate.pilot_classification(self.pilot(high), TABLE)
        self.assertEqual(sorted(classification["high_cells"]), sorted(high))
        self.assertEqual(len(classification["low_cells"]), len(ALL_CELLS) - len(high))
        self.assertTrue(classification["ok"], classification["reasons"])
        self.assertTrue(classification["low_cells_toml"].startswith('low_cells = ["'))
        self.assertIn('"L5N16"', classification["low_cells_toml"])
        self.assertNotIn('"L0N16"', classification["low_cells_toml"])

    def test_a_pilot_with_every_cell_low_or_too_few_on_a_side_asks_for_the_fallback_ladder(self):
        classification = aggregate.pilot_classification(self.pilot([]), TABLE)
        self.assertFalse(classification["ok"])
        self.assertTrue(any("fewer than" in reason for reason in classification["reasons"]))
        few = aggregate.pilot_classification(self.pilot(ALL_CELLS[:3]), TABLE)
        self.assertFalse(few["ok"])
        self.assertEqual(len(few["high_cells"]), 3)

    def test_an_agent_count_with_no_low_cell_fails_the_pilot_whatever_the_counts_on_each_side(self):
        high = [aggregate.cell_name(level, 16) for level in range(LEVELS)] + [
            aggregate.cell_name(level, 8) for level in range(LEVELS)
        ]
        classification = aggregate.pilot_classification(self.pilot(high), TABLE)
        self.assertIn("an agent count has no low cell", classification["reasons"])
        self.assertFalse(classification["ok"])

    def test_a_cell_exactly_at_the_threshold_is_not_low(self):
        # 100 per mille: 100 conflicts in 1000 attempts.
        result = seed_result(1)
        for case in result["cases"]:
            for entry in case["runs"]:
                if entry["arm"] == "certified" and (case["level"], entry["agents"]) == (0, 2):
                    entry["attempts"], entry["conflicts"] = 1000, 100
        classification = aggregate.pilot_classification([result], TABLE)
        self.assertIn("L0N2", classification["high_cells"])


class BootstrapTests(unittest.TestCase):
    PAIRS = [(1000, 400 + 25 * index) for index in range(40)]

    def test_the_point_estimate_is_the_ratio_of_the_sums_and_the_interval_brackets_it(self):
        interval = aggregate.bootstrap_gain(self.PAIRS, 2000, 20260926, 950)
        self.assertAlmostEqual(interval["gain"], 40_000 / sum(certified for _, certified in self.PAIRS))
        self.assertLess(interval["lower"], interval["gain"])
        self.assertGreater(interval["upper"], interval["gain"])
        self.assertEqual(interval["cases"], 40)

    def test_a_seed_reproduces_its_interval_and_another_seed_moves_it(self):
        first = aggregate.bootstrap_gain(self.PAIRS, 2000, 20260926, 950)
        self.assertEqual(first, aggregate.bootstrap_gain(self.PAIRS, 2000, 20260926, 950))
        self.assertNotEqual(first, aggregate.bootstrap_gain(self.PAIRS, 2000, 1, 950))

    def test_the_interval_is_the_2_5th_and_97_5th_percentile_of_the_sorted_resamples(self):
        import random

        generator = random.Random(20260926)
        count = len(self.PAIRS)
        gains = []
        for _ in range(2000):
            chosen = [self.PAIRS[generator.randrange(count)] for _ in range(count)]
            gains.append(sum(serial for serial, _ in chosen) / sum(certified for _, certified in chosen))
        gains.sort()
        interval = aggregate.bootstrap_gain(self.PAIRS, 2000, 20260926, 950)
        # Fifty resamples lie below the lower bound and fifty above the upper.
        self.assertEqual(interval["lower"], gains[50])
        self.assertEqual(interval["upper"], gains[1949])

    def test_a_wider_interval_contains_a_narrower_one_and_identical_cases_give_a_point(self):
        narrow = aggregate.bootstrap_gain(self.PAIRS, 2000, 7, 500)
        wide = aggregate.bootstrap_gain(self.PAIRS, 2000, 7, 950)
        self.assertLessEqual(wide["lower"], narrow["lower"])
        self.assertGreaterEqual(wide["upper"], narrow["upper"])
        same = aggregate.bootstrap_gain([(1000, 500)] * 10, 200, 3, 950)
        self.assertEqual((same["lower"], same["gain"], same["upper"]), (2.0, 2.0, 2.0))


class NewestPerSeedTests(unittest.TestCase):
    def test_a_finished_run_beats_a_later_reservation_and_a_failed_launch(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def record(name, seed, status):
                path = root / name
                body = {"seed": seed, "status": status}
                if status in ("completed", "failed"):
                    body.update({"exit_code": 0 if status == "completed" else 1, "finished_at": "t", "stdout": "", "stderr": ""})
                path.write_text(json.dumps(body), encoding="utf-8")
                return path

            finished = record("run-20260101T000000Z-seed-1.json", 1, "completed")
            reserved = record("run-20260102T000000Z-seed-1.json", 1, "started")
            failed = record("run-20260103T000000Z-seed-1.json", 1, "failed-to-launch")
            self.assertEqual(aggregate.newest_per_seed([failed, reserved, finished]), {1: finished})


class OccAnomalyTests(unittest.TestCase):
    def test_the_occ_baseline_counts_lost_updates_and_stale_scans_with_its_other_anomalies(self):
        results = five()
        for result in results:
            result["occ_undetected_phantoms"] = 1
            result["occ_stale_input_commits"] = 1
            result["occ_stale_reliance_commits"] = 1
            result["occ_lost_updates"] = 2
            result["occ_stale_scan_commits"] = 3
        analysis = analyse(results)
        committed = analysis["metrics"]["descriptive"]["arms"]["occ"]["merged"]
        self.assertEqual(analysis["metrics"]["descriptive"]["occ_anomalies_per_commit"], round(5 * (1 + 1 + 1 + 2 + 3) / committed, 4))


class PilotInputTests(unittest.TestCase):
    def write(self, directory: Path, seed: int, table: dict, cases=None) -> Path:
        result = seed_result(seed)
        result["preregistration"] = aggregate.experiment_records.preregistration_canonical(table)
        if cases is not None:
            result["cases"] = result["cases"][:cases]
        path = directory / f"pilot-seed-{seed}.json"
        path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        return path

    def test_pilot_outputs_of_the_current_table_are_read_whatever_low_cells_held_then(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [self.write(Path(directory), seed, {**TABLE, "low_cells": []}) for seed in TABLE["pilot_seeds"]]
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                aggregate.pilot(paths)
        self.assertIn("low_cells = [", printed.getvalue())

    def test_pilot_outputs_of_another_preregistration_are_refused_before_any_cell_is_named(self):
        other = {**TABLE, "groups_ladder": [8, 16, 32, 64, 128, 256]}
        with tempfile.TemporaryDirectory() as directory:
            paths = [self.write(Path(directory), seed, other) for seed in TABLE["pilot_seeds"]]
            with self.assertRaises(SystemExit) as raised:
                aggregate.pilot(paths)
        self.assertIn("another preregistration", str(raised.exception))
        self.assertIn("groups_ladder", str(raised.exception))

    def test_a_pilot_output_with_fewer_cases_than_the_table_says_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [self.write(Path(directory), seed, TABLE, cases=10) for seed in TABLE["pilot_seeds"]]
            with self.assertRaises(SystemExit) as raised:
                aggregate.pilot(paths)
        self.assertIn("cases", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
