"""The F003 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict that
stopped depending on it fails here. That every counter of the criteria resolves in the real harness
JSON, and that every rule is one the harness judges, is held on the Rust side (`f003` tests in
bins/ptr-bench). The mutation plan is held against the criteria here: every hard counter has a
rule and a mutation that expects it, and COMPLETED needs the whole plan killed."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "feedback" / "F003-calibrated-arbiter"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("f003_aggregate", HERE / "aggregate.py")
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
        "benchmark": "calibrated-arbiter",
        "experiment": "F003",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "harness_gaps": [],
        "descriptive": {"slice_rate_worst_abs_z_milli": 1993},
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
        self.assertIn("ineligible_auto_proposals <= 0", CRITERIA["falsifiers"])
        self.assertIn("ltt_coverage_violations <= ltt_coverage_critical", CRITERIA["falsifiers"])
        self.assertIn("heldout_policy_violations <= 0", CRITERIA["falsifiers"])
        # The negative controls exit 2 when they miss, so only the count of tripped controls is a rule.
        self.assertIn("negative_controls_tripped >= 4", CRITERIA["coverage"])
        self.assertIn("product_panics <= 0", CRITERIA["falsifiers"])
        # The literal upper-bound rule is reported and judged by nothing.
        for rule in RULES:
            self.assertNotIn("literal", rule)
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
            for word in ("descriptive", "timing", "latency", "rss", "_ns", "elapsed", "literal"):
                self.assertNotIn(word, rule)

    def test_the_unit_literals_follow_the_pinned_scale(self):
        pre = CONFIG["preregistration"]
        pinned = CONFIG["pinned_thresholds"]
        self.assertIn(f"stream_branches >= {pre['stream_branches']}", CRITERIA["coverage"])
        self.assertIn(f"policies_recorded >= {len(pre['policy_sizes'])}", CRITERIA["coverage"])
        self.assertIn(f"mc_replicates >= {pre['mc_replicates']}", CRITERIA["coverage"])
        self.assertIn(f"mc_certified_replicates >= {pre['mc_replicates'] // 2}", CRITERIA["coverage"])
        self.assertIn(f"ope_logs >= {pre['ope_logs']}", CRITERIA["coverage"])
        self.assertIn(f"ope_estimates >= {pre['ope_logs'] * 8}", CRITERIA["coverage"])
        self.assertIn(f"positivity_probes >= {pre['ope_logs'] * 2}", CRITERIA["coverage"])
        # The scenario has teeth: the uncalibrated threshold's exact harm rate is above alpha.
        self.assertIn(f"uncalibrated_true_harm_ppm >= {pre['alpha_permille'] * 1000 + 1}", CRITERIA["coverage"])
        # Bonferroni over the policies of every declared seed.
        self.assertEqual(pre["heldout_tests"], len(pre["policy_sizes"]) * len(pre["seeds"]))
        self.assertEqual(sum(pinned["kind_weights_permille"]), 1000)
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        # The decided signed values.
        self.assertIs(pre["adjudicators_see_intent"], False)
        self.assertEqual(pre["uncalibrated_threshold_permille"], 500)
        self.assertEqual(pre["adjudication_protocol"], "experiments/feedback/F003-calibrated-arbiter/PROTOCOL.md")
        self.assertTrue((ROOT / pre["adjudication_protocol"]).is_file())

    def test_the_protocol_says_plainly_that_the_adjudicator_is_a_simulation(self):
        text = (ROOT / CONFIG["preregistration"]["adjudication_protocol"]).read_text(encoding="utf-8")
        self.assertIn("It is a simulation", text)
        self.assertIn("No person adjudicates", text)
        self.assertIn("not shown the intent", text)

    def test_the_config_table_and_the_registry_agree(self):
        """Every key the list requires is in the table; the table adds only the digest of the protocol
        and, at the freeze, the preregistration text, the criteria and their digests."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["F003"]["required"]
        self.assertLessEqual(set(required), set(table))
        extra = set(table) - set(required)
        self.assertLessEqual(
            extra,
            {"adjudication_protocol_sha256", "preregistration_text", "preregistration_text_sha256", "criteria", "criteria_sha256"},
        )
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "calibrated-arbiter")
        # The placeholders left are the digests of the protocol, the text and the criteria, pinned at the freeze.
        placeholders = [k for k, v in table.items() if isinstance(v, str) and v.startswith("must-be-")]
        self.assertEqual(
            sorted(placeholders),
            ["adjudication_protocol_sha256", "criteria_sha256", "preregistration_text_sha256"],
        )


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
        # A correct policy has a non-zero violation count (at most delta of the replicates), so that
        # count is read against the critical value and the hard counter is the excess beyond it.
        self.assertNotIn("ltt_coverage_violations", hard)
        self.assertIn("ltt_coverage_excess", hard)

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
        self.assertTrue(any(m["file"] == "crates/ptr-branch/src/arbiter.rs" for m in MUTATIONS["mutation"]))
        self.assertTrue(any(m["file"] == "crates/ptr-analytics/src/binomial.rs" for m in MUTATIONS["mutation"]))
        # The plan lists no mutant of the two equivalents the preregistration documents.
        for mutation in MUTATIONS["mutation"]:
            self.assertNotEqual((mutation["find"], mutation["replace"]), ("bound <= alpha", "bound < alpha"))

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
        for path, value in [("ltt_coverage_violations", 6), ("ltt_coverage_critical", 4)]:
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
            lambda result: result.pop("ineligible_auto_proposals"),
            lambda result: result.update(ineligible_auto_proposals=None),
            lambda result: result.update(ineligible_auto_proposals=0.0),
            lambda result: result.update(ineligible_auto_proposals=False),
            lambda result: result.update(ineligible_auto_proposals="0"),
            lambda result: result.pop("ltt_coverage_critical"),
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
            lambda result: result.update(experiment="R001"),
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
        put(results[17], "held_out_overlap", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "threshold_mismatches", 1)
        results[29] = None
        verdict = self.verdict(results, {17: 1, 29: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_judgements_do_not_alias_their_inputs(self):
        results = clean_seeds()
        before = copy.deepcopy(results)
        judged(results)
        self.assertEqual(results, before)


class F003Tests(unittest.TestCase):
    def verdict(self, results, exits=None):
        return aggregate.decide(judged(results, exits), PASSED, SEEDS, True)

    def test_timings_and_descriptive_numbers_cannot_change_the_verdict(self):
        results = clean_seeds()
        for result in results.values():
            result["descriptive"] = {"slice_rate_worst_abs_z_milli": 10**12}
            result["timing_ms"] = {"elapsed": 10**9}
            # The literal upper-bound counter is reported, not judged.
            result["heldout_violations_literal_ucb"] = 4
        self.assertEqual(self.verdict(results)["verdict"], aggregate.COMPLETED)

    def test_a_panic_of_the_product_falsifies_it_is_not_invalid(self):
        results = clean_seeds()
        put(results[43], "product_panics", 1)
        verdict = self.verdict(results, {43: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [43])
        # The same record read with the harness's old exit codes (a crash, 101) is INVALID.
        self.assertEqual(self.verdict(results, {43: 101})["verdict"], aggregate.INVALID)

    def test_the_coverage_violations_are_judged_against_the_critical_value(self):
        results = clean_seeds()
        # A correct policy has violations, up to the critical value.
        put(results[17], "ltt_coverage_violations", 71)
        put(results[17], "ltt_coverage_critical", 131)
        self.assertEqual(self.verdict(results)["verdict"], aggregate.COMPLETED)
        put(results[17], "ltt_coverage_violations", 132)
        self.assertEqual(self.verdict(results, {17: 1})["verdict"], aggregate.FALSIFIED)

    def test_the_held_out_test_and_the_oracle_comparisons_falsify(self):
        for path in ("heldout_policy_violations", "threshold_mismatches", "ineligible_auto_proposals",
                     "ips_bias_violations", "positivity_unrefused", "bound_oracle_mismatches"):
            results = clean_seeds()
            put(results[71], path, 1)
            verdict = self.verdict(results, {71: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
            self.assertEqual(verdict["falsified_seeds"], [71])

    def test_a_policy_that_certifies_nothing_makes_the_run_vacuous(self):
        for path, value in (("policies_certified", 0), ("uncalibrated_baseline_rejected", 0),
                            ("heldout_candidates_min_certified", 99)):
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
        self.assertEqual(metrics["ineligible_auto_proposals"], 0)
        self.assertIsNone(aggregate.metric_values(None))
        self.assertTrue(all(isinstance(value, int) and not isinstance(value, bool) for value in metrics.values()))

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" in CONFIG and "criteria" not in CONFIG["preregistration"]:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_the_scope_says_the_adjudicator_is_a_simulation(self):
        self.assertIn("SIMULATED", aggregate.SCOPE)
        self.assertIn("ptr-branch", aggregate.SCOPE)
        text = " ".join(aggregate.LIMITATIONS)
        for needle in ("simulation", "mechanism", "literal rule", "equivalent", "descriptive"):
            self.assertIn(needle, text)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "held_out_overlap", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, True), seeds, PASSED)
        self.assertIn("F003 calibrated-arbiter: FALSIFIED", text)
        self.assertIn("held_out_overlap <= 0", text)

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
        put(results[17], "threshold_mismatches", 1)
        for mutations in (None, {"killed": self.PLAN - 1, "total": self.PLAN}, {"killed": self.PLAN, "total": self.PLAN}):
            verdict = self.decide(mutations, results, {17: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
            self.assertEqual(verdict["mutations_ok"], mutations is not None and mutations["killed"] == self.PLAN)


if __name__ == "__main__":
    unittest.main()
