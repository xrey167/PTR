"""The F001 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`f001` tests in bins/ptr-bench)."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "feedback" / "F001-generate-verify-repair"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("f001_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
KILLED = {"killed": len(MUTATIONS["mutation"]), "total": len(MUTATIONS["mutation"])}


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
        "benchmark": "generate-verify-repair",
        "experiment": "F001",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "controls": {
            "score_accept": {"required_rule_tripped": True},
            "blind_repair": {"required_rule_tripped": True},
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
    for left, _, right in pending:
        known = [value for value in (lookup(result, left), lookup(result, right)) if value is not None]
        value = known[0] if known else 3
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
        self.assertIn("false_accepts <= 0", CRITERIA["falsifiers"])
        self.assertIn("repair_gain.b16 >= 1", CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        """Every min_X of [pinned_thresholds] is the literal of a coverage rule, and max_hard_counter
        the literal of every hard-counter rule."""
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]}
        pinned = CONFIG["pinned_thresholds"]
        literal = {
            "min_initial_fail_tasks": "initial_fail_tasks",
            "min_repaired_successes": "repaired_successes",
            "min_exhausted_runs": "exhausted_runs",
            "min_lying_refused": "lying_refused",
            "min_shallow_refused": "shallow_refused",
        }
        for key, counter in literal.items():
            self.assertIn((counter, ">=", str(pinned[key])), rules, key)
        for name in ("status_pass", "status_fail", "status_unknown", "status_disputed"):
            self.assertIn((name, ">=", str(pinned["min_status_seen"])), rules, name)
        for profile in ("uniform", "heterogeneous", "hard"):
            self.assertIn((f"profiles.{profile}.tasks", ">=", str(pinned["min_tasks_per_profile"])), rules)
        for counter in MUTATIONS["hard_counters"]:
            self.assertIn((counter, "<=", str(pinned["max_hard_counter"])), rules, counter)
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        self.assertEqual(pinned["seeds"], SEEDS)
        for rule_name in ("repair_gain", "draws_gain"):
            self.assertEqual(
                sorted(int(rule[0].rsplit(".b", 1)[1]) for rule in rules if rule[0].startswith(rule_name + ".")),
                [budget for budget in pinned["budgets"] if budget > 1],
            )
            for rule in rules:
                if rule[0].startswith(rule_name + "."):
                    self.assertEqual((rule[1], rule[2]), (">=", str(pinned["min_gain"])))

    def test_the_reference_result_holds_every_rule(self):
        result = passing_result(17)
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        """No rule is vacuous: some value of its left counter breaks it."""
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            result = passing_result(17)
            bound = int(right) if aggregate.NUMBER.fullmatch(right) else aggregate.resolve(result, right)
            put(result, left, bound - 1 if op == ">=" else bound + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_the_mutation_plan_names_only_counters_the_aggregator_can_read(self):
        for counter in MUTATIONS["hard_counters"]:
            self.assertIn(counter, MANIFEST["metrics"].split(", "))
        self.assertEqual(len(MUTATIONS["hard_counters"]), 18)

    @unittest.skipUnless(
        "preregistration" in CONFIG,
        "F001 is listed in experiments/preregistration.toml but unfrozen: config.toml gets its [preregistration] table at the freeze",
    )
    def test_the_config_table_and_the_registry_agree(self):
        """Skipped until the freeze adds the [preregistration] table."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["F001"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "generate-verify-repair")
        for key in ("criteria", "preregistration_text"):
            self.assertTrue((ROOT / table[key]).is_file(), key)


class VerdictTests(unittest.TestCase):
    def verdict(self, results, exits=None, static=PASSED, mutations=KILLED):
        return aggregate.decide(judged(results, exits), static, SEEDS, mutations)

    def test_every_seed_holding_every_rule_with_the_mutation_evidence_is_completed(self):
        verdict = self.verdict(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_each_falsifier_on_one_seed_falsifies(self):
        for rule in CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            results = clean_seeds()
            bound = int(right) if aggregate.NUMBER.fullmatch(right) else aggregate.resolve(results[43], right)
            put(results[43], left, bound + 1 if op == "<=" else bound - 1)
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
            lambda result: result.pop("false_accepts"),
            lambda result: result.update(false_accepts=None),
            lambda result: result.update(false_accepts=0.0),
            lambda result: result.update(false_accepts=False),
            lambda result: result.update(tasks="12", status_pass="x"),
            lambda result: result["repair_gain"].pop("b2"),
            lambda result: result["draws_gain"].pop("b4"),
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
            lambda result: result.update(experiment="S002"),
            lambda result: result.update(harness_gaps=["x"]),
            lambda result: result["procedure"].update(seed_is_declared=False),
            lambda result: result["procedure"].update(shape_is_pinned=False),
            lambda result: result["controls"]["score_accept"].update(required_rule_tripped=False),
            lambda result: result["controls"]["blind_repair"].update(required_rule_tripped=False),
            lambda result: result["controls"].pop("blind_repair"),
            lambda result: result.pop("controls"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "false_accepts", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "false_accepts", 1)
        results[29] = None
        verdict = self.verdict(results, {17: 1, 29: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_judgements_do_not_alias_their_inputs(self):
        results = clean_seeds()
        before = copy.deepcopy(results)
        judged(results)
        self.assertEqual(results, before)


class MutationEvidenceTests(unittest.TestCase):
    def test_without_mutation_evidence_or_with_a_survivor_the_verdict_is_invalid_not_completed(self):
        seeds = judged(clean_seeds())
        for mutations in (None, {"killed": 17, "total": 18}, {"killed": 0, "total": 18}):
            verdict = aggregate.decide(seeds, PASSED, SEEDS, mutations)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, mutations)
            self.assertIsNone(verdict["recommended_status"])
            self.assertTrue(any("mutation" in problem for problem in verdict["problems"]))

    def test_a_falsified_seed_is_falsified_whatever_the_mutation_evidence(self):
        results = clean_seeds()
        put(results[17], "collateral_edits", 4)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, None)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)

    def test_the_plan_lists_the_mutations_the_evidence_must_count(self):
        self.assertGreaterEqual(len(MUTATIONS["mutation"]), 15)
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)


class F001Tests(unittest.TestCase):
    def test_the_comparative_rules_are_labelled_as_consequences_of_the_stub_skill(self):
        header = (HERE / "criteria.toml").read_text(encoding="utf-8")
        self.assertIn("CONSEQUENCES OF THE STUB'S STATED SKILL", header)
        self.assertIn("not that any model repairs better than it rewrites", header)
        comparative = [rule for rule in CRITERIA["falsifiers"] if rule.split(" ")[0].startswith(("repair_gain.", "draws_gain."))]
        self.assertEqual(len(comparative), 8)
        self.assertEqual(len(CRITERIA["falsifiers"]), 26)

    def test_the_hard_counters_come_first_in_the_falsifiers(self):
        self.assertEqual(
            CRITERIA["falsifiers"][: len(MUTATIONS["hard_counters"])],
            [f"{counter} <= 0" for counter in MUTATIONS["hard_counters"]],
        )

    def test_a_repair_arm_that_gains_nothing_over_rewrite_falsifies_at_any_budget(self):
        """A loop that silently stops repairing makes the two arms the same run: every gain is 0."""
        for family in ("repair_gain", "draws_gain"):
            for budget in (2, 4, 8, 16):
                results = clean_seeds()
                put(results[17], f"{family}.b{budget}", 0)
                verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, KILLED)
                self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, (family, budget))

    def test_every_strict_rule_is_named_by_a_mutation_through_its_shortfall_counter(self):
        named = {counter for m in MUTATIONS["mutation"] for counter in m["expect"]}
        self.assertIn("repair_gain_shortfalls", named)
        self.assertIn("draws_gain_shortfalls", named)
        self.assertIn("altered_candidates", named)

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_scope_and_limitations_state_the_limits(self):
        text = " ".join(aggregate.LIMITATIONS)
        self.assertIn("consequences of that stated skill", text)
        self.assertIn("synthetic", text)
        self.assertIn("seeded stubs", text)
        self.assertIn("No claim about the quality of any model", aggregate.SCOPE)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "false_accepts", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, KILLED), seeds, PASSED, KILLED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("false_accepts <= 0", text)
        self.assertIn("mutation check", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS, KILLED)["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
