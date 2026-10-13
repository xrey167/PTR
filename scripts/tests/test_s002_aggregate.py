"""The S002 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`s002` tests in bins/ptr-bench). The mutation plan is held against the
criteria here: every hard counter has a rule and a mutation that expects it, and COMPLETED needs
the whole plan killed."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "semdb" / "S002-snapshot-cancellation"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("s002_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
PLAN = len(MUTATIONS["mutation"])


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
        "benchmark": "snapshot-cancellation",
        "experiment": "S002",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "controls": {
            "bypass": {"required_rule_tripped": True},
            "ledger_scan": {"required_rule_tripped": True},
        },
        "harness_gaps": [],
    }
    pending = []
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
        left, op, right = aggregate.parse_rule(rule)
        if aggregate.NUMBER.fullmatch(right):
            put(result, left, int(right))
        else:
            pending.append((left, op, right))
    for left, op, right in pending:
        low, high = lookup(result, left), lookup(result, right)
        if op == "<=":
            put(result, left, 0 if low is None else low)
            put(result, right, max(3 if high is None else high, 0 if low is None else low))
        else:
            value = low if low is not None else high if high is not None else 3
            put(result, left, value)
            put(result, right, value)
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
        root = {"a": 2, "f": 0.4921875, "o": {"n": 3}, "s": "x", "t": "x", "b": True, "z": None}
        self.assertTrue(aggregate.evaluate(root, "o.n >= a"))
        self.assertFalse(aggregate.evaluate(root, "a == 3"))
        self.assertTrue(aggregate.evaluate(root, "a <= 2"))
        self.assertTrue(aggregate.evaluate(root, "s == t"))
        for bad in ["f >= 1", "f >= 0.4", "b >= 1", "z >= 1", "missing >= 1", "o >= 1", "s >= t", "s == 1", "a.b >= 1"]:
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
        self.assertIn("stale_actions_executed <= 0", CRITERIA["falsifiers"])
        self.assertIn("boundary_refusals_a0 >= 1", CRITERIA["coverage"])
        self.assertEqual(len(CRITERIA["coverage"]), 15)
        self.assertEqual(len(CRITERIA["falsifiers"]), 17)
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        """Every min_X / max_X of [pinned_thresholds] is the literal of the rule on X."""
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]}
        pinned = CONFIG["pinned_thresholds"]
        checked = 0
        for key, value in pinned.items():
            if key.startswith("min_"):
                self.assertIn((key[4:], ">=", str(value)), rules, key)
                checked += 1
            if key.startswith("max_"):
                self.assertIn((key[4:], "<=", str(value)), rules, key)
                checked += 1
        self.assertGreaterEqual(checked, 30)
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        self.assertEqual(pinned["seeds"], SEEDS)

    def test_the_reference_result_holds_every_rule(self):
        result = passing_result(17)
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        """No rule is vacuous: some value of its left counter breaks it."""
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            result = passing_result(17)
            value = aggregate.resolve(result, left)
            bound = int(right) if aggregate.NUMBER.fullmatch(right) else aggregate.resolve(result, right)
            if op == ">=":
                put(result, left, bound - 1)
            else:
                put(result, left, max(value, bound) + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_every_hard_counter_has_a_rule_and_a_mutation_that_expects_it(self):
        """The plan and the criteria describe one set of hard counters."""
        falsifier_counters = {aggregate.parse_rule(rule)[0] for rule in CRITERIA["falsifiers"]}
        expected = {counter for mutation in MUTATIONS["mutation"] for counter in mutation["expect"]}
        metrics = MANIFEST["metrics"].split(", ")
        for counter in MUTATIONS["hard_counters"]:
            self.assertIn(counter, metrics)
            self.assertIn(counter, expected, f"no mutation expects {counter}")
        # Every zero-bounded falsifier counter is a hard counter, bar the absolute bound on arm A1's waste.
        for rule in CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            if right == "0" and left != "wasted_units_a1":
                self.assertIn(left, MUTATIONS["hard_counters"], rule)
        self.assertLessEqual(falsifier_counters - set(MUTATIONS["hard_counters"]), {"cancel_latency_rounds_max", "wasted_units_a1"})

    def test_the_plan_runs_the_harness_the_manifest_names(self):
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)
        self.assertEqual(MUTATIONS["package"], "ptr-bench")
        self.assertEqual(len({m["name"] for m in MUTATIONS["mutation"]}), PLAN)
        self.assertEqual(PLAN, 21)

    @unittest.skipUnless(
        "S002" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "S002 is not in experiments/preregistration.toml until the freeze",
    )
    def test_the_config_table_and_the_registry_agree(self):
        """Skipped while S002 is absent from preregistration.toml (it is added at the freeze)."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["S002"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "snapshot-cancellation")


class VerdictTests(unittest.TestCase):
    def verdict(self, results, exits=None, static=PASSED, mutations_ok=True):
        return aggregate.decide(judged(results, exits), static, SEEDS, mutations_ok)

    def test_every_seed_holding_every_rule_with_the_mutation_evidence_is_completed(self):
        verdict = self.verdict(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_each_falsifier_on_one_seed_falsifies(self):
        for rule in CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            results = clean_seeds()
            put(results[43], left, int(right) + 1 if aggregate.NUMBER.fullmatch(right) else 99)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["falsified_seeds"], [43], rule)
            self.assertEqual(verdict["recommended_status"], "failed")

    def test_an_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
        for rule in CRITERIA["coverage"]:
            left, _, right = aggregate.parse_rule(rule)
            results = clean_seeds()
            put(results[101], left, int(right) - 1)
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
            lambda result: result.pop("stale_actions_executed"),
            lambda result: result.update(stale_actions_executed=None),
            lambda result: result.update(stale_actions_executed=0.0),
            lambda result: result.update(stale_actions_executed=False),
            lambda result: result.update(cancel_latency_rounds_max="1"),
            lambda result: result.pop("wasted_units_a0"),
        ):
            results = clean_seeds()
            damage(results[29])
            judgement = judged(results)[SEEDS.index(29)]
            self.assertFalse(judgement["valid"])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_a_result_that_is_not_the_run_is_invalid(self):
        for damage in (
            lambda result: result.update(scale="reduced"),
            lambda result: result.update(scale="custom"),
            lambda result: result.update(seed=18),
            lambda result: result.update(benchmark="other"),
            lambda result: result.update(experiment="S001"),
            lambda result: result.update(harness_gaps=["x"]),
            lambda result: result["procedure"].update(seed_is_declared=False),
            lambda result: result["procedure"].update(shape_is_pinned=False),
            lambda result: result["controls"]["bypass"].update(required_rule_tripped=False),
            lambda result: result["controls"]["ledger_scan"].update(required_rule_tripped=False),
            lambda result: result["controls"].pop("ledger_scan"),
            lambda result: result.pop("controls"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "mirror_drift", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "stale_actions_executed", 1)
        results[29] = None
        verdict = self.verdict(results, {17: 1, 29: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_judgements_do_not_alias_their_inputs(self):
        results = clean_seeds()
        before = copy.deepcopy(results)
        judged(results)
        self.assertEqual(results, before)


class MutationEvidenceGateTests(unittest.TestCase):
    """COMPLETED needs mutations.json with killed == total == the planned count."""

    def decide(self, mutations, results=None, exits=None):
        results = results if results is not None else clean_seeds()
        ok = aggregate.mutations_ok(mutations, PLAN)
        return aggregate.decide(judged(results, exits), PASSED, SEEDS, ok)

    def test_the_plan_size_is_read_from_the_plan(self):
        self.assertEqual(aggregate.plan_size(), PLAN)
        self.assertGreater(PLAN, 0)

    def test_absent_evidence_is_invalid_never_completed(self):
        verdict = self.decide(None)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertFalse(verdict["mutations_ok"])
        self.assertIsNone(verdict["recommended_status"])
        self.assertIn("mutation evidence missing or incomplete", verdict["problems"])

    def test_one_survivor_is_invalid(self):
        verdict = self.decide({"killed": PLAN - 1, "total": PLAN})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertFalse(verdict["mutations_ok"])

    def test_every_planned_mutation_killed_is_completed(self):
        verdict = self.decide({"killed": PLAN, "total": PLAN})
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertTrue(verdict["mutations_ok"])

    def test_a_record_of_another_plan_size_does_not_count(self):
        # All killed, but of fewer or more mutations than the plan lists (a stale or partial run).
        for total in (PLAN - 1, PLAN + 1):
            verdict = self.decide({"killed": total, "total": total})
            self.assertEqual(verdict["verdict"], aggregate.INVALID, total)
        for bad in ({"killed": True, "total": True}, {"killed": "21", "total": "21"}, {}, "x"):
            self.assertFalse(aggregate.mutations_ok(bad, PLAN), bad)
        self.assertFalse(aggregate.mutations_ok({"killed": 0, "total": 0}, 0))

    def test_a_falsification_is_not_downgraded_and_reports_the_gate(self):
        results = clean_seeds()
        put(results[17], "boundary_arm_divergences", 1)
        for mutations in (None, {"killed": PLAN - 1, "total": PLAN}, {"killed": PLAN, "total": PLAN}):
            verdict = self.decide(mutations, results, {17: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
            self.assertEqual(verdict["mutations_ok"], mutations == {"killed": PLAN, "total": PLAN})


class S002Tests(unittest.TestCase):
    def test_the_gated_claims_are_the_owner_decision(self):
        self.assertIn("wasted_units_a1 <= wasted_units_a0", CRITERIA["falsifiers"])
        self.assertIn("boundary_refusals_a0 >= 1", CRITERIA["coverage"])
        self.assertIn("stale_effect_attempts_committed <= 0", CRITERIA["falsifiers"])
        self.assertFalse(any("total_units" in rule or "_ns" in rule for rule in CRITERIA["falsifiers"]))
        self.assertIn("owner_decision_2026_10_09_cancellation", CONFIG["pinned_thresholds"])
        self.assertFalse(CONFIG["pinned_thresholds"]["total_units_gated"])

    def test_either_arm_of_a_comparison_rule_falsifies(self):
        for path, value in [("wasted_units_a1", 99), ("wasted_units_a1", 1)]:
            results = clean_seeds()
            put(results[17], path, value)
            verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, True)
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, (path, value))

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_scope_and_limitations_state_the_limits(self):
        text = " ".join(aggregate.LIMITATIONS)
        self.assertIn("total compute is not claimed to fall", text)
        self.assertIn("revision-granular", text)
        self.assertIn("cancel each other", text)
        self.assertIn("equivalent mutants", text)
        self.assertIn("scripted backend", aggregate.SCOPE + text)
        self.assertIn("semantic.cancel_stale_revisions = false", aggregate.SCOPE)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "stale_actions_executed", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, True), seeds, PASSED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("stale_actions_executed <= 0", text)
        self.assertIn("mutations_ok: True", text)

    def test_the_report_carries_the_mutation_summary_when_there_is_one(self):
        seeds = judged(clean_seeds())
        mutations = {"killed": PLAN, "total": PLAN, "git_sha": "a" * 40}
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, True), seeds, PASSED, mutations)
        self.assertIn(f"{PLAN} of {PLAN} killed", text)
        bare = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, False), seeds, PASSED)
        self.assertNotIn("mutation checks", bare)
        self.assertIn("mutations_ok: False", bare)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS, True)["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
