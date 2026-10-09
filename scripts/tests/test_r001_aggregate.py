"""The R001 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict that
stopped depending on it fails here. That every counter of the criteria resolves in the real harness
JSON, and that every rule is one the harness judges, is held on the Rust side (`r001` tests in
bins/ptr-bench). The mutation plan is held against the criteria here: every hard counter has a
rule and a mutation that expects it, and COMPLETED needs the whole plan killed."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "runtime" / "R001-mailbox-backpressure"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("r001_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
RULES = CRITERIA["coverage"] + CRITERIA["falsifiers"]


def put(result: dict, path: str, value) -> None:
    at = result
    keys = path.split(".")
    for key in keys[:-1]:
        at = at.setdefault(key, {})
    at[keys[-1]] = value


def lookup(result: dict, path: str):
    try:
        return aggregate.resolve(result, path)
    except aggregate.Unreadable:
        return None


def passing_result(seed: int) -> dict:
    """A harness result in which every rule of the criteria holds, at the boundary where it can."""
    result = {
        "benchmark": "mailbox-backpressure",
        "experiment": "R001",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "harness_gaps": [],
        "descriptive": {"threaded_latency_ns_p99": 48307, "peak_rss_kib": 29924},
        "timing_ms": {"elapsed": 383.1},
    }
    pending = []
    for rule in RULES:
        left, op, right = aggregate.parse_rule(rule)
        if aggregate.NUMBER.fullmatch(right):
            put(result, left, int(right))
        else:
            pending.append((left, right))
    for left, right in pending:
        # A bound against a bound: the measured value sits exactly at the bound.
        put(result, left, 5)
        put(result, right, 5)
    return result


def judged(results: dict[int, dict], exits: dict[int, int] | None = None) -> list[dict]:
    exits = exits or {}
    return [aggregate.judge_seed(results[seed], exits.get(seed, 0), CRITERIA, seed) for seed in results]


def clean_seeds() -> dict[int, dict]:
    return {seed: passing_result(seed) for seed in SEEDS}


class GrammarTests(unittest.TestCase):
    def test_a_rule_is_three_tokens_with_a_known_operator(self):
        for bad in ["a >= ", "a > 1", "a >=  1", "a >= 1 2", ">= 4", "", "a=1"]:
            with self.assertRaises(aggregate.Unreadable, msg=bad):
                aggregate.evaluate({"a": 1}, bad)

    def test_literals_references_and_types(self):
        root = {"a": 2, "f": 0.4921875, "o": {"n": 3}, "b": True, "z": None}
        self.assertTrue(aggregate.evaluate(root, "o.n >= a"))
        self.assertFalse(aggregate.evaluate(root, "a == 3"))
        self.assertTrue(aggregate.evaluate(root, "a <= 2"))
        for bad in ["f >= 1", "b >= 1", "z >= 1", "missing >= 1", "o >= 1", "a.b >= 1"]:
            with self.assertRaises(aggregate.Unreadable, msg=bad):
                aggregate.evaluate(root, bad)


class CriteriaFileTests(unittest.TestCase):
    def test_every_rule_reads_and_the_sets_are_not_empty(self):
        for key in ("coverage", "falsifiers"):
            rules = CRITERIA[key]
            self.assertTrue(rules)
            self.assertEqual(len(rules), len(set(rules)), key)
            for rule in rules:
                aggregate.parse_rule(rule)
        self.assertIn("capacity_violations <= 0", CRITERIA["falsifiers"])
        self.assertIn("bounded_p99_wait_ticks <= bounded_wait_bound_ticks", CRITERIA["falsifiers"])
        # The negative controls exit 2 when they miss, so a rule on them could never be the failing one.
        self.assertNotIn("negative_controls_tripped >= 2", CRITERIA["coverage"])
        self.assertIn("product_panics <= 0", CRITERIA["falsifiers"])
        self.assertIn("product_hangs <= 0", CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_reference_result_holds_every_rule(self):
        result = passing_result(17)
        for rule in RULES:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        """No rule is vacuous: some value of its left counter breaks it."""
        for rule in RULES:
            left, op, right = aggregate.parse_rule(rule)
            result = passing_result(17)
            value = aggregate.resolve(result, left)
            if op == ">=":
                put(result, left, (int(right) if right.isdigit() else aggregate.resolve(result, right)) - 1)
            else:
                put(result, left, value + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_no_rule_reads_a_wall_clock_or_descriptive_number(self):
        for rule in RULES:
            for word in ("descriptive", "timing", "latency", "rss", "_ns", "elapsed"):
                self.assertNotIn(word, rule)

    def test_the_unit_literals_follow_the_pinned_sweep(self):
        pinned = CONFIG["pinned_thresholds"]
        cells = len(pinned["capacities"]) * len(pinned["payload_class_bytes"])
        self.assertIn(f"unbounded_exceeding_cells >= {cells}", CRITERIA["coverage"])
        self.assertIn(f"recoveries_observed >= {cells}", CRITERIA["coverage"])
        self.assertIn(f"capacities_exercised >= {len(pinned['capacities'])}", CRITERIA["coverage"])
        self.assertIn(f"payload_classes >= {len(pinned['payload_class_bytes'])}", CRITERIA["coverage"])
        self.assertLessEqual(2, pinned["threaded"]["producers"])
        # Observed counts, not the configured producer count.
        self.assertNotIn("threaded_producers >= 2", CRITERIA["coverage"])
        self.assertIn("threaded_producers_with_admissions >= 2", CRITERIA["coverage"])
        self.assertIn(f"threaded_fill_rounds >= {pinned['threaded']['fill_rounds']}", CRITERIA["coverage"])
        for outcome in ("admitted", "full", "closed"):
            self.assertIn(f"threaded_round2_{outcome} >= 1", CRITERIA["coverage"])
        self.assertIn(f"closed_path_cases >= {cells + 1}", CRITERIA["coverage"])
        self.assertIn(f"overload_phases >= {3 * cells}", CRITERIA["coverage"])
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])

    @unittest.skipUnless(
        "R001" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "R001 is not in experiments/preregistration.toml until the freeze",
    )
    def test_the_config_table_and_the_registry_agree(self):
        """Skipped while R001 is absent from preregistration.toml (it is added at the freeze)."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["R001"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "mailbox-backpressure")
        for key in ("criteria", "preregistration_text"):
            self.assertTrue((ROOT / table[key]).is_file(), key)


class MutationPlanTests(unittest.TestCase):
    """The plan and the criteria describe one set of hard counters."""

    def test_every_hard_counter_has_a_rule_and_a_mutation_that_expects_it(self):
        hard = MUTATIONS["hard_counters"]
        ruled = {aggregate.parse_rule(rule)[0] for rule in RULES}
        expected = {counter for mutation in MUTATIONS["mutation"] for counter in mutation["expect"]}
        for counter in hard:
            self.assertIn(counter, ruled, counter)
            self.assertIn(counter, expected, counter)
        self.assertLessEqual(expected, set(hard))

    def test_every_falsifier_counter_that_must_be_zero_is_hard(self):
        hard = set(MUTATIONS["hard_counters"])
        for rule in CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            if op == "<=" and right == "0":
                self.assertIn(left, hard, rule)

    def test_the_plan_runs_the_harness_the_manifest_names(self):
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)
        self.assertEqual(MUTATIONS["package"], "ptr-bench")
        names = [mutation["name"] for mutation in MUTATIONS["mutation"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(any(m["file"].startswith("crates/ptr-exec/") for m in MUTATIONS["mutation"]))
        self.assertTrue(any(m["file"].startswith("bins/ptr-bench/") for m in MUTATIONS["mutation"]))

    def test_every_anchor_occurs_exactly_once(self):
        for mutation in MUTATIONS["mutation"]:
            edits = [(mutation["file"], mutation["find"])] + [
                (extra.get("file", mutation["file"]), extra["find"]) for extra in mutation.get("also", [])
            ]
            for file, find in edits:
                text = (ROOT / file).read_text(encoding="utf-8")
                self.assertEqual(text.count(find), 1, (mutation["name"], file))


class VerdictTests(unittest.TestCase):
    def verdict(self, results, exits=None, static=PASSED, mutations_ok=True):
        return aggregate.decide(judged(results, exits), static, SEEDS, mutations_ok)

    def test_every_seed_holding_every_rule_is_completed(self):
        verdict = self.verdict(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_each_falsifier_on_one_seed_falsifies(self):
        for rule in CRITERIA["falsifiers"]:
            left = aggregate.parse_rule(rule)[0]
            results = clean_seeds()
            put(results[43], left, aggregate.resolve(results[43], left) + 1)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["falsified_seeds"], [43], rule)
            self.assertEqual(verdict["recommended_status"], "failed")

    def test_either_side_of_a_bound_falsifies(self):
        for path, value in [("bounded_p99_wait_ticks", 6), ("bounded_wait_bound_ticks", 4),
                            ("bounded_peak_payload_bytes", 6), ("bounded_payload_bytes_bound", 4)]:
            results = clean_seeds()
            put(results[17], path, value)
            self.assertEqual(self.verdict(results, {17: 1})["verdict"], aggregate.FALSIFIED, path)

    def test_each_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
        for rule in CRITERIA["coverage"]:
            left, op, right = aggregate.parse_rule(rule)
            results = clean_seeds()
            put(results[101], left, int(right) - 1 if op == ">=" else int(right) + 1)
            verdict = self.verdict(results, {101: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["vacuous_seeds"], [101], rule)

    def test_a_missing_seed_is_invalid(self):
        results = clean_seeds()
        del results[71]
        verdict = self.verdict(results)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertIsNone(verdict["recommended_status"])
        self.assertTrue(any("71" in problem for problem in verdict["problems"]))

    def test_a_counter_that_is_missing_null_or_mistyped_is_invalid_not_failed(self):
        for damage in (
            lambda result: result.pop("capacity_violations"),
            lambda result: result.update(capacity_violations=None),
            lambda result: result.update(capacity_violations=0.0),
            lambda result: result.update(capacity_violations=False),
            lambda result: result.update(capacity_violations="0"),
            lambda result: result.pop("bounded_wait_bound_ticks"),
        ):
            results = clean_seeds()
            damage(results[29])
            judgement = judged(results)[SEEDS.index(29)]
            self.assertFalse(judgement["valid"])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_a_result_that_is_not_the_run_is_invalid(self):
        for damage in (
            lambda result: result.update(scale="reduced"),
            lambda result: result.update(seed=18),
            lambda result: result.update(benchmark="other"),
            lambda result: result.update(experiment="R007"),
            lambda result: result.update(harness_gaps=["x"]),
            lambda result: result["procedure"].update(seed_is_declared=False),
            lambda result: result["procedure"].update(shape_is_pinned=False),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "silent_drops", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "ownership_violations", 1)
        results[29] = None
        verdict = self.verdict(results, {17: 1, 29: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_judgements_do_not_alias_their_inputs(self):
        results = clean_seeds()
        before = copy.deepcopy(results)
        judged(results)
        self.assertEqual(results, before)


class R001Tests(unittest.TestCase):
    def verdict(self, results, exits=None):
        return aggregate.decide(judged(results, exits), PASSED, SEEDS, True)

    def test_wall_clock_and_descriptive_numbers_cannot_change_the_verdict(self):
        results = clean_seeds()
        for result in results.values():
            result["descriptive"] = {"threaded_latency_ns_p99": 10**12, "peak_rss_kib": 10**9}
            result["timing_ms"] = {"elapsed": 10**9}
        self.assertEqual(self.verdict(results)["verdict"], aggregate.COMPLETED)

    def test_a_panic_or_a_hang_of_the_product_falsifies_it_is_not_invalid(self):
        for path in ("product_panics", "product_hangs"):
            results = clean_seeds()
            put(results[43], path, 1)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
            self.assertEqual(verdict["falsified_seeds"], [43], path)
        # The same record read with the harness's old exit codes (a crash, 101) is INVALID.
        results = clean_seeds()
        put(results[43], "product_panics", 1)
        self.assertEqual(self.verdict(results, {43: 101})["verdict"], aggregate.INVALID)

    def test_the_threaded_arm_is_judged_by_its_ledger_and_capacity(self):
        for path in ("threaded_ledger_violations", "threaded_capacity_violations"):
            results = clean_seeds()
            put(results[71], path, 1)
            verdict = self.verdict(results, {71: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
            self.assertEqual(verdict["falsified_seeds"], [71])

    def test_a_missed_canary_makes_the_run_vacuous(self):
        for path, value in (("canary_misses", 1),):
            results = clean_seeds()
            put(results[29], path, value)
            verdict = self.verdict(results, {29: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
            self.assertEqual(verdict["vacuous_seeds"], [29])

    def test_the_metrics_of_a_result_are_its_top_level_integers_without_the_seed(self):
        result = passing_result(17)
        metrics = aggregate.metric_values(result)
        self.assertNotIn("seed", metrics)
        self.assertNotIn("descriptive", metrics)
        self.assertNotIn("timing_ms", metrics)
        self.assertEqual(metrics["capacity_violations"], 0)
        self.assertIsNone(aggregate.metric_values(None))
        self.assertTrue(all(isinstance(value, int) and not isinstance(value, bool) for value in metrics.values()))

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_the_scope_names_what_is_out_of_scope(self):
        self.assertIn("ptr-exec", aggregate.SCOPE)
        text = " ".join(aggregate.LIMITATIONS)
        for needle in ("RequestAdmission", "wall-clock", "descriptive", "threaded"):
            self.assertIn(needle, text)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "silent_drops", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, True), seeds, PASSED)
        self.assertIn("R001 mailbox-backpressure: FALSIFIED", text)
        self.assertIn("silent_drops <= 0", text)

    def test_the_report_carries_the_mutation_summary_when_there_is_one(self):
        seeds = judged(clean_seeds())
        mutations = {"killed": 27, "total": 27, "git_sha": "a" * 40}
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, True), seeds, PASSED, mutations)
        self.assertIn("27 of 27 killed", text)
        self.assertIn("mutations_ok: True", text)
        bare = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, False), seeds, PASSED)
        self.assertNotIn("mutation checks", bare)
        self.assertIn("mutations_ok: False", bare)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS, True)["verdict"], aggregate.COMPLETED)


class MutationEvidenceGateTests(unittest.TestCase):
    """COMPLETED needs mutations.json with killed == total == the planned count."""

    PLAN = len(MUTATIONS["mutation"])

    def decide(self, mutations, results=None, exits=None):
        results = results if results is not None else clean_seeds()
        ok = aggregate.mutations_ok(mutations, self.PLAN)
        return aggregate.decide(judged(results, exits), PASSED, SEEDS, ok)

    def test_the_plan_size_is_read_from_the_plan(self):
        self.assertEqual(aggregate.plan_size(), self.PLAN)
        self.assertGreater(self.PLAN, 0)

    def test_absent_evidence_is_invalid_never_completed(self):
        verdict = self.decide(None)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertFalse(verdict["mutations_ok"])
        self.assertIsNone(verdict["recommended_status"])
        self.assertIn("mutation evidence missing or incomplete", verdict["problems"])

    def test_one_survivor_is_invalid(self):
        verdict = self.decide({"killed": self.PLAN - 1, "total": self.PLAN})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertFalse(verdict["mutations_ok"])

    def test_every_planned_mutation_killed_is_completed(self):
        verdict = self.decide({"killed": self.PLAN, "total": self.PLAN})
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertTrue(verdict["mutations_ok"])

    def test_a_record_of_another_plan_size_does_not_count(self):
        # All killed, but of fewer mutations than the plan lists (a stale or partial run).
        for total in (self.PLAN - 1, self.PLAN + 1):
            verdict = self.decide({"killed": total, "total": total})
            self.assertEqual(verdict["verdict"], aggregate.INVALID, total)
        for bad in ({"killed": True, "total": True}, {"killed": "27", "total": "27"}, {}, "x"):
            self.assertFalse(aggregate.mutations_ok(bad, self.PLAN), bad)
        self.assertFalse(aggregate.mutations_ok({"killed": 0, "total": 0}, 0))

    def test_a_falsification_is_not_downgraded_and_reports_the_gate(self):
        results = clean_seeds()
        put(results[17], "ownership_violations", 1)
        for mutations in (None, {"killed": self.PLAN - 1, "total": self.PLAN}, {"killed": self.PLAN, "total": self.PLAN}):
            verdict = self.decide(mutations, results, {17: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
            self.assertEqual(verdict["mutations_ok"], mutations is not None and mutations["killed"] == self.PLAN)


if __name__ == "__main__":
    unittest.main()
