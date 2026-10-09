"""The Q002 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`q002` tests in bins/ptr-bench)."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "retrieval" / "Q002-evidence-promotion"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("q002_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
PLANNED = len(MUTATIONS["mutation"])
KILLED = {"killed": PLANNED, "total": PLANNED}


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
        "benchmark": "evidence-promotion",
        "experiment": "Q002",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "controls": {
            "baseline_direct": {"required_rule_tripped": True},
            "bool_trust": {"required_rule_tripped": True},
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
        self.assertIn("false_known_mechanism <= 0", CRITERIA["falsifiers"])
        self.assertIn("stage_skip_accepted <= 0", CRITERIA["falsifiers"])
        self.assertIn("stale_or_revoked_promoted <= 0", CRITERIA["falsifiers"])
        self.assertIn("recall_permille >= 950", CRITERIA["falsifiers"])
        self.assertIn("task_success_mechanism >= task_success_baseline", CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        """Every min_X / max_X of [pinned_thresholds] is the literal of a rule on X (per class and per kind
        thresholds are the literal of every rule on a member of the group)."""
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]}
        pinned = CONFIG["pinned_thresholds"]
        per_group = {
            "min_poison_attempts_per_class": "poison_attempts.",
            "min_stage_skip_attempts_per_kind": "stage_skip_by_kind.",
        }
        world_keys = {"max_generations"}  # the world's own parameters, not thresholds
        checked = 0
        for key, value in pinned.items():
            if key in world_keys:
                continue
            if key in per_group:
                members = [rule for rule in rules if rule[0].startswith(per_group[key])]
                self.assertTrue(members, key)
                for left, op, right in members:
                    self.assertEqual((op, right), (">=", str(value)), left)
                checked += 1
            elif key.startswith("min_"):
                self.assertIn((key[4:], ">=", str(value)), rules, key)
                checked += 1
            elif key.startswith("max_"):
                self.assertIn((key[4:], "<=", str(value)), rules, key)
                checked += 1
        self.assertGreaterEqual(checked, 15)
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
            put(result, left, bound - 1 if op == ">=" else value + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_the_mutation_plan_names_only_counters_the_aggregator_can_read(self):
        for counter in MUTATION_COUNTERS():
            self.assertIn(counter, MANIFEST["metrics"].split(", "))

    @unittest.skipUnless(
        "Q002" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "Q002 is not in experiments/preregistration.toml until the freeze",
    )
    def test_the_config_table_and_the_registry_agree(self):
        """Skipped while Q002 is absent from preregistration.toml (it is added at the freeze)."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["Q002"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "evidence-promotion")
        for key in ("criteria", "preregistration_text"):
            self.assertTrue((ROOT / table[key]).is_file(), key)


def MUTATION_COUNTERS() -> list[str]:
    return MUTATIONS["hard_counters"]


class VerdictTests(unittest.TestCase):
    def verdict(self, results, exits=None, static=PASSED, mutations=KILLED):
        return aggregate.decide(judged(results, exits), static, SEEDS, mutations, PLANNED)

    def test_every_seed_holding_every_rule_with_the_mutation_evidence_is_completed(self):
        verdict = self.verdict(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")
        self.assertTrue(verdict["mutations_ok"])

    def test_each_falsifier_on_one_seed_falsifies(self):
        for rule in CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            results = clean_seeds()
            if aggregate.NUMBER.fullmatch(right):
                put(results[43], left, int(right) - 1 if op == ">=" else int(right) + 1)
            else:
                put(results[43], left, 0)
                put(results[43], right, 9)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["falsified_seeds"], [43], rule)
            self.assertEqual(verdict["recommended_status"], "failed")

    def test_an_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
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
            lambda result: result.pop("false_known_mechanism"),
            lambda result: result.update(false_known_mechanism=None),
            lambda result: result.update(false_known_mechanism=0.0),
            lambda result: result.update(false_known_mechanism=False),
            lambda result: result.update(recall_permille="950"),
            lambda result: result["poison_attempts"].pop("P3"),
            lambda result: result["stage_skip_by_kind"].update(K4=None),
            lambda result: result.update(poison_attempts=20),
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
            lambda result: result["controls"]["baseline_direct"].update(required_rule_tripped=False),
            lambda result: result["controls"]["bool_trust"].pop("required_rule_tripped"),
            lambda result: result["controls"].pop("bool_trust"),
            lambda result: result.pop("controls"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_the_boolean_baseline_that_did_not_hallucinate_is_a_failed_coverage_rule_not_an_invalid_seed(self):
        """B2 runs code under test: its silence is judged by false_known_bool_trust >= 1, not by exit 2."""
        results = clean_seeds()
        put(results[17], "false_known_bool_trust", 0)
        results[17]["controls"]["bool_trust"]["required_rule_tripped"] = False
        verdict = self.verdict(results, {17: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["vacuous_seeds"], [17])

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "stage_skip_accepted", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "false_known_mechanism", 1)
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
    def test_without_complete_mutation_evidence_the_verdict_is_invalid_not_completed(self):
        seeds = judged(clean_seeds())
        for mutations in (
            None,
            {"killed": PLANNED - 1, "total": PLANNED},
            {"killed": 0, "total": PLANNED},
            {"killed": 0, "total": 0},
            {"killed": PLANNED - 2, "total": PLANNED - 2},
            {"killed": PLANNED + 1, "total": PLANNED + 1},
            {"killed": None, "total": None},
        ):
            verdict = aggregate.decide(seeds, PASSED, SEEDS, mutations, PLANNED)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, mutations)
            self.assertIsNone(verdict["recommended_status"])
            self.assertFalse(verdict["mutations_ok"])
            self.assertTrue(any("mutation" in problem for problem in verdict["problems"]))

    def test_all_mutations_killed_is_what_completed_needs(self):
        seeds = judged(clean_seeds())
        verdict = aggregate.decide(seeds, PASSED, SEEDS, KILLED, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertTrue(verdict["mutations_ok"])
        self.assertEqual(verdict["problems"], [])

    def test_a_falsified_seed_is_falsified_whatever_the_mutation_evidence(self):
        results = clean_seeds()
        put(results[17], "wrong_refusal_reason", 4)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, None, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertFalse(verdict["mutations_ok"])

    def test_the_plan_lists_the_mutations_the_evidence_must_count(self):
        self.assertGreaterEqual(PLANNED, 20)
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)
        names = [mutation["name"] for mutation in MUTATIONS["mutation"]]
        self.assertEqual(len(names), len(set(names)))
        for mutation in MUTATIONS["mutation"]:
            self.assertTrue(mutation["expect"], mutation["name"])
            for counter in mutation["expect"]:
                self.assertIn(counter, MUTATIONS["hard_counters"], mutation["name"])
        for counter in MUTATIONS["hard_counters"]:
            self.assertTrue(
                any(counter in mutation["expect"] for mutation in MUTATIONS["mutation"]),
                f"{counter} is expected by no mutation",
            )
        harness = [m for m in MUTATIONS["mutation"] if m["file"].startswith("bins/ptr-bench/")]
        self.assertGreaterEqual(len(harness), 2)


class Q002Tests(unittest.TestCase):
    def test_either_side_of_the_task_comparison_falsifies(self):
        for path, value in [("task_success_mechanism", 2), ("task_success_baseline", 9)]:
            results = clean_seeds()
            put(results[17], path, value)
            verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, KILLED, PLANNED)
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, (path, value))

    def test_the_recall_floor_is_the_owner_decision_and_the_answerable_only_reading_is_not_judged(self):
        self.assertIn("recall_permille >= 950", CRITERIA["falsifiers"])
        judged_text = " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for descriptive in ("answerable_success", "task_success_gen_filter_only", "abstentions_correct", "_ns"):
            self.assertNotIn(descriptive, judged_text)

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_scope_and_limitations_state_the_limits(self):
        text = " ".join(aggregate.LIMITATIONS)
        self.assertIn("synthetic structured world", text)
        self.assertIn("Disputed is not a state of PtrRuntime::generation_validity", text)
        self.assertIn("corrupt replicas", text)
        self.assertIn("does not tie the raw event to the capsule", text)
        self.assertIn("trusted-caller path", text)
        self.assertIn("byte-exact", aggregate.SCOPE)
        self.assertIn("lifecycle", aggregate.SCOPE)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "false_known_mechanism", 1)
        seeds = judged(results, {17: 1})
        verdict = aggregate.decide(seeds, PASSED, SEEDS, KILLED, PLANNED)
        text = aggregate.report(verdict, seeds, PASSED, KILLED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("false_known_mechanism <= 0", text)
        self.assertIn("mutation check", text)
        self.assertIn("mutations_ok", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        verdict = aggregate.decide(judged(clean_seeds()), [], SEEDS, KILLED, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
