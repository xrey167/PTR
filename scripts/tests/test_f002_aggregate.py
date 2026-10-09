"""The F002 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict that
stopped depending on it fails here. That every counter of the criteria resolves in the real harness
JSON, and that every rule is one the harness judges, is held on the Rust side (`f002` tests in
bins/ptr-bench). The mutation plan is held against the criteria here: every hard counter has a
rule and a mutation that expects it, and COMPLETED needs the whole plan killed."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "feedback" / "F002-weak-supervision"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("f002_aggregate", HERE / "aggregate.py")
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


def violate(result: dict, rule: str) -> None:
    """Set the left counter of `rule` to a value that breaks it."""
    left, op, right = aggregate.parse_rule(rule)
    bound = int(right) if aggregate.NUMBER.fullmatch(right) else aggregate.resolve(result, right)
    put(result, left, bound - 1 if op == ">=" else bound + 1)


def passing_result(seed: int) -> dict:
    """A harness result in which every rule of the criteria holds, at the boundary where it can."""
    result = {
        "benchmark": "weak-supervision",
        "experiment": "F002",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "harness_gaps": [],
        "timing_ms": {"elapsed": 2300},
    }
    pending = []
    for rule in RULES:
        left, op, right = aggregate.parse_rule(rule)
        if aggregate.NUMBER.fullmatch(right):
            put(result, left, int(right))
        else:
            pending.append((left, right))
    for left, right in pending:
        # A measured number against a measured number: they sit exactly at the bound.
        put(result, left, 800)
        put(result, right, 800)
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
        self.assertIn("veto_contradictions <= 0", CRITERIA["falsifiers"])
        self.assertIn("product_panics <= 0", CRITERIA["falsifiers"])
        self.assertIn("w1_census_accuracy_ds_permille >= w1_census_accuracy_mv_permille", CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)
        # The negative controls exit 2 when they miss, so a rule on them could never be the failing one.
        self.assertFalse(any(rule.startswith("negative_controls") for rule in RULES))

    def test_the_reference_result_holds_every_rule(self):
        result = passing_result(17)
        for rule in RULES:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        """No rule is vacuous: some value of its left counter breaks it."""
        for rule in RULES:
            result = passing_result(17)
            violate(result, rule)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_no_rule_reads_a_wall_clock_number(self):
        for rule in RULES:
            for word in ("timing", "_ms", "elapsed"):
                self.assertNotIn(word, rule)

    def test_the_literals_follow_the_pinned_thresholds(self):
        pinned = CONFIG["pinned_thresholds"]
        self.assertIn(f"gold_items >= {pinned['min_gold_items']}", CRITERIA["coverage"])
        self.assertLessEqual(pinned["min_gold_items"], pinned["gold_items"])
        self.assertIn(f"vetoed_items >= {pinned['min_vetoed_items']}", CRITERIA["coverage"])
        self.assertIn(f"worlds_run >= {pinned['min_worlds_run']}", CRITERIA["coverage"])
        self.assertIn(f"budget_cells_seen >= {len(pinned['votes_per_item'])}", CRITERIA["coverage"])
        self.assertIn(f"gold_guard_probes >= {pinned['min_gold_guard_probes']}", CRITERIA["coverage"])
        for outcome in ("determined", "estimated", "unknown", "disputed"):
            self.assertIn(f"{outcome}_seen >= 1", CRITERIA["coverage"])
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        self.assertEqual(pinned["seeds"], SEEDS)
        # The pilot seeds are never confirmatory.
        self.assertFalse(set(pinned["pilot_seeds"]) & set(SEEDS))

    def test_the_gain_gates_read_the_lower_bound_exactly_when_the_config_says_so(self):
        pinned = CONFIG["pinned_thresholds"]
        for name, key in (("brier", "brier_gate_on_lower_bound"), ("ece", "ece_gate_on_lower_bound")):
            lower = f"w1_{name}_gain_lower_ppm >= 1" in CRITERIA["falsifiers"]
            point = f"w1_{name}_gain_ppm >= 1" in CRITERIA["falsifiers"]
            self.assertEqual(lower, pinned[key], name)
            self.assertNotEqual(lower, point, name)

    @unittest.skipUnless(
        "F002" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "F002 is not in experiments/preregistration.toml until the freeze",
    )
    def test_the_config_table_and_the_registry_agree(self):
        """Skipped while F002 is absent from preregistration.toml (it is added at the freeze)."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["F002"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "weak-supervision")
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
        files = [mutation["file"] for mutation in MUTATIONS["mutation"]]
        for prefix in ("crates/ptr-labeling/", "crates/ptr-analytics/", "bins/ptr-bench/"):
            self.assertTrue(any(file.startswith(prefix) for file in files), prefix)

    def test_every_anchor_occurs_exactly_once(self):
        for mutation in MUTATIONS["mutation"]:
            edits = [(mutation["file"], mutation["find"])] + [
                (extra.get("file", mutation["file"]), extra["find"]) for extra in mutation.get("also", [])
            ]
            for file, find in edits:
                text = (ROOT / file).read_text(encoding="utf-8")
                self.assertEqual(text.count(find), 1, (mutation["name"], file))

    def test_the_seed_pair_the_checker_passes_is_a_declared_seed(self):
        self.assertEqual(MUTATIONS["cases"], MUTATIONS["seed"])
        self.assertIn(MUTATIONS["seed"], SEEDS)


class VerdictTests(unittest.TestCase):
    def verdict(self, results, exits=None, static=PASSED, mutations_ok=True):
        return aggregate.decide(judged(results, exits), static, SEEDS, mutations_ok)

    def test_every_seed_holding_every_rule_is_completed(self):
        verdict = self.verdict(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_each_falsifier_on_one_seed_falsifies(self):
        for rule in CRITERIA["falsifiers"]:
            results = clean_seeds()
            violate(results[43], rule)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["falsified_seeds"], [43], rule)
            self.assertEqual(verdict["recommended_status"], "failed")

    def test_either_side_of_the_accuracy_comparison_falsifies(self):
        for path, value in [("w1_census_accuracy_ds_permille", 799), ("w1_census_accuracy_mv_permille", 801)]:
            results = clean_seeds()
            put(results[17], path, value)
            self.assertEqual(self.verdict(results, {17: 1})["verdict"], aggregate.FALSIFIED, path)

    def test_each_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
        for rule in CRITERIA["coverage"]:
            results = clean_seeds()
            violate(results[101], rule)
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
            lambda result: result.pop("veto_contradictions"),
            lambda result: result.update(veto_contradictions=None),
            lambda result: result.update(veto_contradictions=0.0),
            lambda result: result.update(veto_contradictions=False),
            lambda result: result.update(veto_contradictions="0"),
            lambda result: result.pop("w1_census_accuracy_mv_permille"),
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
        put(results[17], "veto_contradictions", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "model_oracle_mismatches", 1)
        results[29] = None
        verdict = self.verdict(results, {17: 1, 29: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_judgements_do_not_alias_their_inputs(self):
        results = clean_seeds()
        before = copy.deepcopy(results)
        judged(results)
        self.assertEqual(results, before)


class F002Tests(unittest.TestCase):
    def verdict(self, results, exits=None):
        return aggregate.decide(judged(results, exits), PASSED, SEEDS, True)

    def test_wall_clock_numbers_and_descriptive_arms_cannot_change_the_verdict(self):
        results = clean_seeds()
        for result in results.values():
            result["timing_ms"] = {"elapsed": 10**9}
            result["w2_census_accuracy_ds_permille"] = 0  # the contaminated world is descriptive
            result["w2_census_accuracy_mv_permille"] = 1000
            result["w3_brier_gain_ppm"] = -(10**6)
            result["w1_mvp_veto_violating_items"] = 10**6
        self.assertEqual(self.verdict(results)["verdict"], aggregate.COMPLETED)

    def test_a_veto_contradiction_a_panic_or_an_oracle_mismatch_falsifies_it_is_not_invalid(self):
        for path in ("veto_contradictions", "product_panics", "model_oracle_mismatches",
                     "resolution_oracle_mismatches", "metric_mismatches", "gold_guard_violations"):
            results = clean_seeds()
            put(results[43], path, 1)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
            self.assertEqual(verdict["falsified_seeds"], [43], path)
        # The same record read with a crash exit code (101) is INVALID.
        results = clean_seeds()
        put(results[43], "product_panics", 1)
        self.assertEqual(self.verdict(results, {43: 101})["verdict"], aggregate.INVALID)

    def test_a_gain_that_the_lower_bound_does_not_support_falsifies(self):
        for path in ("w1_brier_gain_lower_ppm", "w1_ece_gain_lower_ppm"):
            results = clean_seeds()
            put(results[71], path, 0)
            verdict = self.verdict(results, {71: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)

    def test_a_point_gain_alone_does_not_gate_when_the_lower_bound_does(self):
        results = clean_seeds()
        put(results[71], "w1_brier_gain_ppm", -5)
        self.assertEqual(self.verdict(results)["verdict"], aggregate.COMPLETED)

    def test_a_thin_coverage_run_is_vacuous(self):
        for path, value in (("disputed_seen", 0), ("gold_items", 499), ("few_function_warnings", 0)):
            results = clean_seeds()
            put(results[29], path, value)
            verdict = self.verdict(results, {29: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
            self.assertEqual(verdict["vacuous_seeds"], [29])

    def test_the_metrics_of_a_result_are_its_top_level_integers_without_the_seed(self):
        result = passing_result(17)
        metrics = aggregate.metric_values(result)
        self.assertNotIn("seed", metrics)
        self.assertNotIn("timing_ms", metrics)
        self.assertEqual(metrics["veto_contradictions"], 0)
        self.assertIsNone(aggregate.metric_values(None))
        self.assertTrue(all(isinstance(value, int) and not isinstance(value, bool) for value in metrics.values()))

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_the_scope_names_what_is_out_of_scope(self):
        self.assertIn("ptr-labeling", aggregate.SCOPE)
        text = " ".join(aggregate.LIMITATIONS)
        for needle in ("synthetic", "homogeneous", "contaminated", "Disputed", "mutation"):
            self.assertIn(needle, text)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "veto_contradictions", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, True), seeds, PASSED)
        self.assertIn("F002 weak-supervision: FALSIFIED", text)
        self.assertIn("veto_contradictions <= 0", text)

    def test_the_report_carries_the_mutation_summary_when_there_is_one(self):
        seeds = judged(clean_seeds())
        mutations = {"killed": 25, "total": 25, "git_sha": "a" * 40}
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, True), seeds, PASSED, mutations)
        self.assertIn("25 of 25 killed", text)
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
        for total in (self.PLAN - 1, self.PLAN + 1):
            verdict = self.decide({"killed": total, "total": total})
            self.assertEqual(verdict["verdict"], aggregate.INVALID, total)
        for bad in ({"killed": True, "total": True}, {"killed": "25", "total": "25"}, {}, "x"):
            self.assertFalse(aggregate.mutations_ok(bad, self.PLAN), bad)
        self.assertFalse(aggregate.mutations_ok({"killed": 0, "total": 0}, 0))

    def test_a_falsification_is_not_downgraded_and_reports_the_gate(self):
        results = clean_seeds()
        put(results[17], "acquisition_violations", 1)
        for mutations in (None, {"killed": self.PLAN - 1, "total": self.PLAN}, {"killed": self.PLAN, "total": self.PLAN}):
            verdict = self.decide(mutations, results, {17: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
            self.assertEqual(verdict["mutations_ok"], mutations is not None and mutations["killed"] == self.PLAN)


if __name__ == "__main__":
    unittest.main()
