"""The Q003 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`q003` tests in bins/ptr-bench)."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "retrieval" / "Q003-v2-postgres-hybrid"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("q003_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
INTEGRITY_COUNTERS = ("workload_digest_mismatches",)
BASELINE_FLOOR = "recall10_reference_permille"
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
        "benchmark": "postgres-hybrid",
        "experiment": "Q003-v2",
        "seed": seed,
        "scale": "pinned",
        "procedure": {
            "seed_is_declared": True,
            "shape_is_pinned": True,
            "strong_arm": {"mode": "full", "ran": True, "reranked": True},
        },
        "controls": {"no_live_join": {"required_rule_tripped": True}},
        "harness_gaps": [],
        "unsupported": {},
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
        for rule in (
            "stale_hits <= 0",
            "planted_stale_returned <= 0",
            "late_write_accepted <= 0",
            "late_write_wrong_refusal <= 0",
            "product_errors <= 0",
            "lifecycle_projection_mismatches <= 0",
            "product_panics <= 0",
            "recall10_pg_deficit_vs_reference_permille <= 30",
        ):
            self.assertIn(rule, CRITERIA["falsifiers"])
        self.assertIn("workload_digest_mismatches <= 0", CRITERIA["coverage"])
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        """Every min_X / max_X of [pinned_thresholds] is the literal of a rule on X (per class and per kind
        thresholds are the literal of every rule on a member of the group)."""
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]}
        pinned = CONFIG["pinned_thresholds"]
        per_group = {"min_retrieval_hits_per_mode": "retrieval_hits."}
        checked = 0
        for key, value in pinned.items():
            if key in per_group:
                members = [rule for rule in rules if rule[0].startswith(per_group[key])]
                self.assertGreaterEqual(len(members), 2, key)
                for left, op, right in members:
                    self.assertEqual((op, right), (">=", str(value)), left)
                checked += 1
            elif key in ("recall_margin_permille", "recall_exact_margin_permille"):
                target = "reference" if key == "recall_margin_permille" else "exact"
                margin = [rule for rule in rules if rule[0] == f"recall10_pg_deficit_vs_{target}_permille"]
                self.assertEqual(len(margin), 1)
                for left, op, right in margin:
                    self.assertEqual((op, right), ("<=", str(value)), left)
                checked += 1
            elif key.startswith("min_"):
                self.assertIn((key[4:], ">=", str(value)), rules, key)
                checked += 1
            elif key.startswith("max_"):
                self.assertIn((key[4:], "<=", str(value)), rules, key)
                checked += 1
        self.assertGreaterEqual(checked, 20)
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

    def test_the_preregistration_table_repeats_the_pinned_thresholds_and_holds_only_two_placeholders(self):
        table = CONFIG["preregistration"]
        pinned = CONFIG["pinned_thresholds"]
        for key, value in table.items():
            if key in pinned:
                self.assertEqual(value, pinned[key], key)
        self.assertEqual(table["harness"], aggregate.BENCHMARK)
        # Two placeholders before the freeze; none after it, when both are SHA-256 digests.
        placeholders = sorted(key for key, value in table.items() if str(value).startswith("must-be-"))
        self.assertIn(placeholders, (["criteria_sha256", "preregistration_text_sha256"], []))
        if not placeholders:
            for key in ("criteria_sha256", "preregistration_text_sha256"):
                self.assertRegex(table[key], r"^[0-9a-f]{64}$")
        for key in ("preregistration_text", "criteria"):
            self.assertTrue((ROOT / table[key]).is_file(), key)

    def test_the_placeholder_preregistration_table_names_the_declared_seeds(self):
        """The [preregistration] table is still the placeholder one the gate holds Q003 in `planned` with; the freeze
        replaces it. Its seeds are already the manifest's."""
        self.assertEqual(CONFIG["preregistration"]["seeds"], SEEDS)


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
            if left in INTEGRITY_COUNTERS or left == BASELINE_FLOOR:
                continue  # a harness-integrity guard: INVALID, see the test below
            results = clean_seeds()
            put(results[101], left, int(right) - 1 if op == ">=" else int(right) + 1)
            verdict = self.verdict(results, {101: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["vacuous_seeds"], [101], rule)

    def test_a_harness_integrity_failure_is_invalid_never_falsified(self):
        """A label or corpus-digest mismatch is a harness defect: it says nothing about the hypothesis, although the
        harness also judges it as a coverage rule (the mutation checker wants a named counter and exit 1)."""
        for counter in INTEGRITY_COUNTERS:
            self.assertIn(f"{counter} <= 0", CRITERIA["coverage"])
            for count in (1, 7):
                results = clean_seeds()
                put(results[71], counter, count)
                judgement = judged(results, {71: 1})[SEEDS.index(71)]
                self.assertFalse(judgement["valid"], counter)
                self.assertTrue(any(counter in reason for reason in judgement["invalid"]), counter)
                verdict = self.verdict(results, {71: 1})
                self.assertEqual(verdict["verdict"], aggregate.INVALID, counter)
                self.assertEqual(verdict["invalid_seeds"], [71])
                self.assertEqual(verdict["vacuous_seeds"], [])
                self.assertEqual(verdict["falsified_seeds"], [])
                self.assertIsNone(verdict["recommended_status"])

    def test_a_reference_below_its_floor_voids_only_the_recall_rule_against_it(self):
        """The floor makes the reference-recall rule not evaluable: a clean seed is INVALID, never FALSIFIED by that alone,
        and any other tripped falsifier or unmet coverage rule of the seed still falsifies."""
        results = clean_seeds()
        put(results[71], BASELINE_FLOOR, 895)
        put(results[71], "recall10_pg_deficit_vs_reference_permille", 52)  # the deficit the weak reference is not a verdict on
        judgement = judged(results, {71: 1})[SEEDS.index(71)]
        self.assertTrue(judgement["valid"])
        self.assertTrue(any(BASELINE_FLOOR in reason for reason in judgement["unevaluable"]))
        self.assertEqual(judgement["falsifiers_failed"], [])
        self.assertEqual(judgement["coverage_failed"], [])
        verdict = self.verdict(results, {71: 1})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["unevaluable_seeds"], [71])
        self.assertEqual(verdict["falsified_seeds"], [])
        self.assertIsNone(verdict["recommended_status"])
        # A stale hit on the same seed is a falsification, whatever the reference did.
        put(results[71], "stale_hits", 3)
        verdict = self.verdict(results, {71: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [71])
        self.assertEqual(self.verdict(results, {71: 1})["recommended_status"], "failed")
        # So is a coverage rule that has nothing to do with the reference.
        results = clean_seeds()
        put(results[71], BASELINE_FLOOR, 895)
        put(results[71], "answerable_queries", 799)
        verdict = self.verdict(results, {71: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["vacuous_seeds"], [71])
        # The sensitivity control is not voided by a weak reference.
        results = clean_seeds()
        put(results[71], BASELINE_FLOOR, 895)
        put(results[71], "recall10_pg_deficit_vs_exact_permille", 31)
        self.assertEqual(self.verdict(results, {71: 1})["verdict"], aggregate.FALSIFIED)

    def test_the_exact_search_rule_and_the_lexical_probe_rule_gate(self):
        for rule in ("recall10_pg_deficit_vs_exact_permille <= 30", "nondeterministic_searches <= 0"):
            self.assertIn(rule, CRITERIA["falsifiers"])
        self.assertIn("live_probe_lexical_self_hits >= 45", CRITERIA["coverage"])
        results = clean_seeds()
        put(results[17], "live_probe_lexical_self_hits", 44)
        self.assertEqual(self.verdict(results, {17: 1})["verdict"], aggregate.FALSIFIED)

    def test_a_mismatch_counter_that_is_zero_leaves_the_seed_valid_and_a_real_falsification_still_wins(self):
        for counter in INTEGRITY_COUNTERS:
            results = clean_seeds()
            put(results[17], counter, 0)
            self.assertTrue(judged(results)[SEEDS.index(17)]["valid"])
            put(results[29], counter, 1)  # harness defect on one seed ...
            put(results[43], "stale_hits", 1)  # ... and a falsification on another
            verdict = self.verdict(results, {29: 1, 43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
            self.assertEqual(verdict["falsified_seeds"], [43])
            self.assertEqual(verdict["invalid_seeds"], [29])

    def test_a_missing_seed_is_invalid(self):
        results = clean_seeds()
        del results[71]
        verdict = self.verdict(results)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertIsNone(verdict["recommended_status"])
        self.assertTrue(any("71" in problem for problem in verdict["problems"]))

    def test_a_counter_that_is_missing_null_or_mistyped_is_invalid_not_failed(self):
        for damage in (
            lambda result: result.pop("stale_hits"),
            lambda result: result.update(stale_hits=None),
            lambda result: result.update(stale_hits=0.0),
            lambda result: result.update(stale_hits=False),
            lambda result: result.update(recall10_pg_deficit_vs_reference_permille="30"),
            lambda result: result["retrieval_hits"].pop("lexical"),
            lambda result: result["retrieval_hits"].update(vector=None),
            lambda result: result.update(retrieval_hits=20),
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
            lambda result: result["controls"]["no_live_join"].update(required_rule_tripped=False),
            lambda result: result["controls"]["no_live_join"].pop("required_rule_tripped"),
            lambda result: result["controls"].pop("no_live_join"),
            lambda result: result.pop("controls"),
            lambda result: result.update(unsupported={"recall10_strong_permille": "not run"}),
            lambda result: result.pop("unsupported"),
            lambda result: result["procedure"]["strong_arm"].update(mode="no-rerank"),
            lambda result: result["procedure"]["strong_arm"].update(mode="off"),
            lambda result: result["procedure"]["strong_arm"].update(ran=False),
            lambda result: result["procedure"]["strong_arm"].update(reranked=False),
            lambda result: result["procedure"].pop("strong_arm"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_a_control_that_returned_no_planted_row_is_the_harnesss_failure_not_a_finding(self):
        results = clean_seeds()
        put(results[17], "control_planted_returned", 0)
        results[17]["controls"]["no_live_join"]["required_rule_tripped"] = False
        verdict = self.verdict(results, {17: 1})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["invalid_seeds"], [17])

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "late_write_accepted", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "stale_hits", 1)
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
        put(results[17], "product_panics", 4)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, None, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertFalse(verdict["mutations_ok"])

    def test_the_plan_lists_the_mutations_the_evidence_must_count(self):
        self.assertEqual(PLANNED, 19)
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


class Q003Tests(unittest.TestCase):
    def test_the_reference_falling_short_falsifies_and_the_strong_baseline_never_gates(self):
        path = "recall10_pg_deficit_vs_reference_permille"
        results = clean_seeds()
        put(results[17], path, 31)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, KILLED, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
        put(results[17], path, 30)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, KILLED, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.INVALID, "exit 1 with every rule holding disagrees")
        judged_text = " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for descriptive in ("vs_strong", "recall10_strong", "on_strong_subset", "ndcg10_strong"):
            self.assertNotIn(descriptive, judged_text)
        self.assertIn("strong_queries >= 200", CRITERIA["coverage"])

    def test_the_descriptive_metrics_are_never_judged(self):
        judged_text = " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for descriptive in (
            "ndcg",
            "bm25_only",
            "pg_lexical_only",
            "pg_vector_only",
            "latency",
            "index_build",
            "_ns",
        ):
            self.assertNotIn(descriptive, judged_text)

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration_sha256" not in MANIFEST:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_scope_and_limitations_state_the_claim_the_criteria_gate_and_no_other(self):
        pinned = CONFIG["pinned_thresholds"]
        scope = aggregate.SCOPE
        # The numbers are the pinned ones; the strong baseline is descriptive; no rule the criteria lack is named.
        self.assertIn(f"{pinned['queries']} paraphrased queries per seed", scope)
        self.assertIn(f"at most {pinned['recall_margin_permille']} per mille below both that of the reference rag_reference", scope)
        self.assertIn("the harness's own exact dense search over the live set (the sensitivity control)", scope)
        self.assertIn("a sanity floor", scope)
        self.assertIn(f"a seeded subset of {pinned['strong_queries']} answerable queries and is reported, never gated", scope)
        self.assertIn(f"{pinned['capsules']} capsules", scope)
        self.assertIn("lexical list is empty", scope)
        self.assertIn("does not test the quality of a working lexical half", scope)
        for wrong in ("each of two baselines", "400", "recall_paired_se_vs_strong", "<= 10 on every seed"):
            self.assertNotIn(wrong, scope)
        text = " ".join(aggregate.LIMITATIONS)
        for wrong in ("each of two baselines", "at 400 queries", "coverage rules recall_paired_se"):
            self.assertNotIn(wrong, text)
        for needed in (
            "synthetic corpus",
            "plainto_tsquery",
            "not tested as a falsifier",
            "planted by raw SQL",
            "rebuilt from the live table",
            "approximate",
            "912",
            "projector_residual_rows",
            "discordant queries",
            "seed 17 was run end to end",
            "every mutation run",
            "final-harness dry run was the pilot seed 1001",
            "about 0.7 percent over the five seeds",
            "10 to 16 percent of seeds",
            "sensitivity control",
            "not independent corpora",
            "at most 10 per mille",
            f"{pinned['strong_queries']} answerable queries",
        ):
            self.assertIn(needed, text)
        # Only the criteria's rules gate: the descriptive strong and exact-cosine metrics are named by none.
        judged_text = " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for descriptive in ("exact_cosine", "main_query_lexical_hits", "discordan", "paired_se"):
            self.assertNotIn(descriptive, judged_text)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "stale_hits", 1)
        seeds = judged(results, {17: 1})
        verdict = aggregate.decide(seeds, PASSED, SEEDS, KILLED, PLANNED)
        text = aggregate.report(verdict, seeds, PASSED, KILLED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("stale_hits <= 0", text)
        self.assertIn("mutation check", text)
        self.assertTrue(text.startswith("# Q003 generation-joined retrieval: FALSIFIED"))
        self.assertIn("does not say that the PostgreSQL hybrid matches a hybrid", text.split("Limitations:")[0])
        self.assertIn("mutations_ok", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        verdict = aggregate.decide(judged(clean_seeds()), [], SEEDS, KILLED, PLANNED)
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)


class PinWorkloadTests(unittest.TestCase):
    """The committed re-pin script rewrites exactly the pins of params.rs and config.toml."""

    @classmethod
    def setUpClass(cls):
        pin_spec = importlib.util.spec_from_file_location("q003_pin_workload", HERE / "tests" / "pin_workload.py")
        cls.pin = importlib.util.module_from_spec(pin_spec)
        pin_spec.loader.exec_module(cls.pin)

    def test_the_declared_seeds_are_the_manifests(self):
        self.assertEqual(list(self.pin.SEEDS), SEEDS)

    def test_the_pins_of_the_committed_files_are_found_once_per_seed_and_replaced(self):
        new = {seed: f"{seed:02x}" * 32 for seed in SEEDS}
        params = self.pin.pin_params(self.pin.PARAMS.read_text(encoding="utf-8"), new)
        config = self.pin.pin_config(self.pin.CONFIG.read_text(encoding="utf-8"), new)
        for seed, digest in new.items():
            self.assertEqual(params.count(digest), 1)
            self.assertEqual(config.count(f'workload_digest_seed_{seed} = "{digest}"'), 2)

    def test_a_pin_that_is_missing_is_refused(self):
        with self.assertRaises(SystemExit):
            self.pin.pin_config("x = 1\n", {17: "0" * 64})
        with self.assertRaises(SystemExit):
            self.pin.pin_config('workload_digest_seed_17 = "' + "0" * 64 + '"\n', {17: "0" * 64})
        with self.assertRaises(SystemExit):
            self.pin.pin_params("pub const X: u8 = 1;\n", {17: "0" * 64})


if __name__ == "__main__":
    unittest.main()
