"""The M008 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`m008` tests in bins/ptr-bench)."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "model" / "M008-fast-weight-memory"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("m008_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
PINNED = CONFIG["pinned_thresholds"]
CONTROLS = ("oracle_cheat", "random", "stale_first", "positive")
PASSED: list[dict] = []  # this experiment names no static evidence
INTEGRITY_COUNTERS = ("oracle_label_mismatches", "stale_in_live_index", "window_budget_violations")
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
        "benchmark": "fast-weight-memory",
        "experiment": "M008",
        "seed": seed,
        "scale": "pinned",
        "procedure": {
            "seed_is_declared": True,
            "seed_is_pilot": False,
            "shape_is_pinned": True,
            "pool_sha256": PINNED["pool_sha256"],
            "vectors_sha256": PINNED["vectors_sha256"],
            "embedding_model": PINNED["embedding_model"],
            "embedding_revision": PINNED["embedding_revision"],
        },
        "controls": {name: {"required_rule_tripped": True} for name in CONTROLS},
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


class GrammarTests(unittest.TestCase):
    def test_a_rule_is_three_tokens_with_a_known_operator(self):
        for bad in ["a >= ", "a > 1", "a >=  1", "a >= 1 2", ">= 4", "", "a=1"]:
            with self.assertRaises(aggregate.Unreadable, msg=bad):
                aggregate.evaluate({"a": 1}, bad)

    def test_literals_references_and_types(self):
        root = {"a": 2, "f": 0.5, "o": {"n": 3}, "s": "x", "t": "x", "b": True, "z": None}
        self.assertTrue(aggregate.evaluate(root, "o.n >= a"))
        self.assertFalse(aggregate.evaluate(root, "a == 3"))
        self.assertTrue(aggregate.evaluate(root, "s == t"))
        for bad in ["f >= 1", "b >= 1", "z >= 1", "missing >= 1", "o >= 1", "s >= t", "s == 1", "a.b >= 1"]:
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
        for rule in (
            "false_recall_fast_permille <= false_recall_recency_permille",
            "recall_gain_vs_recency_lo_permille >= 1",
            "recall_gain_vs_hybrid_live_lo_permille >= 1",
            "token_budget_violations <= 0",
            "product_panics <= 0",
        ):
            self.assertIn(rule, CRITERIA["falsifiers"])
        for rule in ("stale_in_live_index <= 0", "window_budget_violations <= 0"):
            self.assertIn(rule, CRITERIA["coverage"])
            self.assertNotIn(rule, CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)
        # hybrid_append is reported and never judged.
        self.assertNotIn("hybrid_append", " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"]))

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]}
        groups = {
            "max_unanswerable_answered_permille": ("unanswerable_answered_permille.", "<="),
            "min_recall_gain_lo_permille": ("recall_gain_vs_", ">="),
            "min_control_positive_gain_lo_permille": ("control_positive_gain_vs_", ">="),
        }
        checked = 0
        pool_parameters = {"max_fact_tokens"}  # the pool's own longest fact, not a threshold
        for key, value in PINNED.items():
            if key in pool_parameters:
                continue
            if key in groups:
                prefix, op = groups[key]
                members = [r for r in rules if r[0].startswith(prefix) and (not key.endswith("gain_lo_permille") or r[0].endswith("_lo_permille"))]
                self.assertTrue(members, key)
                for left, found_op, right in members:
                    self.assertEqual((found_op, right), (op, str(value)), left)
                checked += 1
            elif key.startswith("min_"):
                self.assertIn((key[4:], ">=", str(value)), rules, key)
                checked += 1
            elif key.startswith("max_"):
                self.assertIn((key[4:], "<=", str(value)), rules, key)
                checked += 1
        self.assertGreaterEqual(checked, 20)
        self.assertEqual(CRITERIA["seed_aggregation"], PINNED["seed_aggregation"])
        self.assertEqual(PINNED["seeds"], SEEDS)
        self.assertFalse(set(PINNED["pilot_seeds"]) & set(SEEDS))

    def test_the_reference_result_holds_every_rule(self):
        result = passing_result(17)
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
            left, op, right = aggregate.parse_rule(rule)
            result = passing_result(17)
            value = aggregate.resolve(result, left)
            bound = int(right) if aggregate.NUMBER.fullmatch(right) else aggregate.resolve(result, right)
            put(result, left, bound - 1 if op == ">=" else value + 1 if not aggregate.NUMBER.fullmatch(right) else bound + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_the_mutation_plan_names_only_counters_the_aggregator_can_read(self):
        for counter in MUTATIONS["hard_counters"]:
            self.assertIn(counter, MANIFEST["metrics"].split(", "))

    @unittest.skipUnless(
        "M008" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "the freeze adds the [preregistration] table and the keys",
    )
    def test_the_config_table_and_the_registry_agree(self):
        if "preregistration" not in CONFIG:
            self.skipTest("not frozen yet")
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["M008"]["required"]
        for key in required:
            self.assertIn(key, table)


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
                put(results[43], left, 9)
                put(results[43], right, 3)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["falsified_seeds"], [43], rule)
            self.assertEqual(verdict["recommended_status"], "failed")

    def test_an_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
        for rule in CRITERIA["coverage"]:
            left, op, right = aggregate.parse_rule(rule)
            if left in INTEGRITY_COUNTERS:
                continue
            results = clean_seeds()
            put(results[101], left, int(right) - 1 if op == ">=" else int(right) + 1)
            verdict = self.verdict(results, {101: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["vacuous_seeds"], [101], rule)

    def test_a_harness_integrity_failure_is_invalid_never_falsified(self):
        for counter in INTEGRITY_COUNTERS:
            self.assertIn(f"{counter} <= 0", CRITERIA["coverage"])
            for count in (1, 7):
                results = clean_seeds()
                put(results[71], counter, count)
                judgement = judged(results, {71: 1})[SEEDS.index(71)]
                self.assertFalse(judgement["valid"], counter)
                verdict = self.verdict(results, {71: 1})
                self.assertEqual(verdict["verdict"], aggregate.INVALID, counter)
                self.assertEqual(verdict["invalid_seeds"], [71])
                self.assertEqual(verdict["falsified_seeds"], [])
                self.assertIsNone(verdict["recommended_status"])

    def test_a_real_falsification_beats_an_integrity_failure_on_another_seed(self):
        results = clean_seeds()
        put(results[29], "oracle_label_mismatches", 1)
        put(results[43], "readback_failures", 1)
        verdict = self.verdict(results, {29: 1, 43: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [43])
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_a_missing_seed_is_invalid(self):
        results = clean_seeds()
        del results[71]
        verdict = self.verdict(results)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any("71" in problem for problem in verdict["problems"]))

    def test_a_counter_that_is_missing_null_or_mistyped_is_invalid_not_failed(self):
        for damage in (
            lambda result: result.pop("product_panics"),
            lambda result: result.update(product_panics=None),
            lambda result: result.update(product_panics=0.0),
            lambda result: result.update(product_panics=False),
            lambda result: result.update(recall_gain_vs_recency_lo_permille="1"),
            lambda result: result["unanswerable_answered_permille"].pop("fast"),
            lambda result: result.update(unanswerable_answered_permille=50),
        ):
            results = clean_seeds()
            damage(results[29])
            self.assertFalse(judged(results)[SEEDS.index(29)]["valid"])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_a_result_that_is_not_the_run_is_invalid(self):
        for damage in (
            lambda result: result.update(scale="reduced"),
            lambda result: result.update(scale="custom"),
            lambda result: result.update(seed=18),
            lambda result: result.update(benchmark="other"),
            lambda result: result.update(experiment="Q003"),
            lambda result: result.update(harness_gaps=["x"]),
            lambda result: result["procedure"].update(seed_is_declared=False),
            lambda result: result["procedure"].update(seed_is_pilot=True),
            lambda result: result["procedure"].pop("seed_is_pilot"),
            lambda result: result["procedure"].update(shape_is_pinned=False),
            lambda result: result["procedure"].update(pool_sha256="0" * 64),
            lambda result: result["procedure"].update(vectors_sha256="0" * 64),
            lambda result: result["procedure"].update(embedding_revision="x"),
            lambda result: result["controls"]["random"].update(required_rule_tripped=False),
            lambda result: result["controls"]["stale_first"].pop("required_rule_tripped"),
            lambda result: result["controls"].pop("oracle_cheat"),
            lambda result: result.pop("controls"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "stale_residual_violations", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid_and_a_falsified_seed_beats_it(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)
        put(results[17], "product_errors", 1)
        verdict = self.verdict(results, {17: 1, 43: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [43])

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
            {"killed": 0, "total": 0},
            {"killed": PLANNED - 2, "total": PLANNED - 2},
            {"killed": PLANNED + 1, "total": PLANNED + 1},
            {"killed": None, "total": None},
        ):
            verdict = aggregate.decide(seeds, PASSED, SEEDS, mutations, PLANNED)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, mutations)
            self.assertFalse(verdict["mutations_ok"])

    def test_a_falsified_seed_is_falsified_whatever_the_mutation_evidence(self):
        results = clean_seeds()
        put(results[17], "decode_contract_violations", 4)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, None, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertFalse(verdict["mutations_ok"])

    def test_the_plan_lists_the_mutations_the_evidence_must_count(self):
        self.assertGreaterEqual(PLANNED, 9)
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
        # Every falsifier's left counter is named by a mutation (the two outcome rules through their counts).
        named = {c for m in MUTATIONS["mutation"] for c in m["expect"]}
        for rule in CRITERIA["falsifiers"]:
            left = aggregate.parse_rule(rule)[0]
            left = {
                "false_recall_fast_permille": "false_recall_violation",
                "recall_gain_vs_recency_lo_permille": "gain_violation_recency",
                "recall_gain_vs_hybrid_live_lo_permille": "gain_violation_hybrid_live",
            }.get(left, left)
            self.assertIn(left, named, rule)
        harness = [m for m in MUTATIONS["mutation"] if m["file"].startswith("bins/ptr-bench/")]
        self.assertGreaterEqual(len(harness), 2)


class M008Tests(unittest.TestCase):
    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_hybrid_append_and_latency_are_never_judged(self):
        judged_text = " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for descriptive in ("hybrid_append", "latency", "_ns", "recall_at_k", "unknown_permille"):
            self.assertNotIn(descriptive, judged_text)

    def test_scope_and_limitations_state_the_limits(self):
        text = " ".join(aggregate.LIMITATIONS)
        for phrase in (
            "synthetic templated English",
            "cue extractor",
            "favours the memory",
            "untrained",
            "pilot seeds 1001..1005",
            "hybrid_live is told which document",
            "single seeded draws",
            "capacity table",
            "same rule in the harness",
            "never judged",
        ):
            self.assertIn(phrase, text)
        self.assertIn("update-aware recall@1", aggregate.SCOPE)
        self.assertIn("paired bootstrap", aggregate.SCOPE)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "readback_failures", 1)
        seeds = judged(results, {17: 1})
        verdict = aggregate.decide(seeds, PASSED, SEEDS, KILLED, PLANNED)
        text = aggregate.report(verdict, seeds, PASSED, KILLED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("readback_failures <= 0", text)
        self.assertIn("mutation check", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        verdict = aggregate.decide(judged(clean_seeds()), [], SEEDS, KILLED, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
