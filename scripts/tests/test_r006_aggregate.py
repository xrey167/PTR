"""The R006 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON is held on the Rust side (`r006` tests in bins/ptr-bench) once the harness is built;
that every rule is one the harness judges is held there today (`r006::metrics` tests)."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "runtime" / "R006-positional-reuse-oracle"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("r006_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
PINNED = CONFIG["pinned_thresholds"]
SEEDS = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
CELLS = PINNED["invalidation_coverage_matrix"]["cells"]
SHIFTS = PINNED["position_shift_tokens"]


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
        "benchmark": "positional-reuse-oracle",
        "experiment": "R006",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "model_pins": {"pinned": {"tiny_model_config_digest": aggregate.EXPECTED_MODEL_DIGEST}},
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
        value = known[0] if known else "sha256:0" if left.endswith("digest") else 0
        put(result, left, value)
        put(result, right, value)
    return result


def judged(results: dict, exits: dict | None = None) -> list[dict]:
    exits = exits or {}
    return [aggregate.judge_seed(results[seed], exits.get(seed, 0), CRITERIA, seed) for seed in results]


def clean_seeds() -> dict:
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
        self.assertNotIn("static_evidence", CRITERIA)
        self.assertEqual(CRITERIA["experiment"], "R006")

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
            if isinstance(value, str):
                put(result, left, "sha256:other")
            elif op == ">=":
                bound = int(right) if aggregate.NUMBER.fullmatch(right) else aggregate.resolve(result, right)
                put(result, left, bound - 1)
            else:
                put(result, left, value + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_the_rules_cover_every_pinned_shift_cell_and_inflight_member(self):
        rules = set(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for shift in SHIFTS:
            for metric in ("reuse_attempts_per_shift", "reused_tokens_per_shift", "secondary_index_reuse_cases_per_shift"):
                self.assertIn(f"{metric}.shift_{shift} >= 1", rules)
            self.assertIn(f"bitwise_mismatch_count.shift_{shift} <= 0", rules)
        for size in PINNED["nonaligned_insertion_tokens"]:
            self.assertIn(f"nonaligned_insertion_cases.tokens_{size} >= 1", rules)
        self.assertEqual(len(CELLS), 52)
        for cell in CELLS:
            self.assertIn(f"invalidation_matrix_cases.{cell} >= 1", rules)
            self.assertIn(f"invalidation_matrix_stale_hits.{cell} <= 0", rules)
        for kind in PINNED["inflight_race_required_kinds"]:
            for ty in PINNED["invalidation_types"]:
                self.assertIn(f"shifted_reuse_inflight_race_cases.{kind}_{ty} >= 1", rules)
        self.assertIn("fresh_prefills_completed_after_revocation >= upstream_revocation_cases", rules)
        self.assertIn("invalidated_case_id_set_digest == fresh_prefill_completed_case_id_set_digest", rules)
        self.assertEqual(len(CRITERIA["coverage"]), 97)
        self.assertEqual(len(CRITERIA["falsifiers"]), 69)

    def test_no_rule_reads_the_timing_metric(self):
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
            self.assertNotIn("latency", rule)

    def test_the_model_digest_is_the_one_layer_pin(self):
        self.assertEqual(
            aggregate.EXPECTED_MODEL_DIGEST,
            "sha256:a77e017c8ee3b56bb06a0b22512aae3a70a1b65d4ae283bf2d8bb02353029c2e",
        )
        self.assertEqual(PINNED["tiny_model_config"]["layers"], 1)

    @unittest.skipUnless(
        "R006" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "R006 is not in experiments/preregistration.toml until the freeze",
    )
    def test_the_config_table_and_the_registry_agree(self):
        """Skipped while R006 is absent from preregistration.toml (it is added at the freeze)."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["R006"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "positional-reuse-oracle")
        for key, kind in required.items():
            value = table[key]
            expected = {"int": int, "str": str, "file": str, "int-list": list, "str-list": list}[kind]
            self.assertIs(type(value), expected, key)
        for key in ("criteria", "preregistration_text"):
            self.assertTrue((ROOT / table[key]).is_file(), key)


class VerdictTests(unittest.TestCase):
    def verdict(self, results, exits=None, static=PASSED):
        return aggregate.decide(judged(results, exits), static, SEEDS)

    def test_every_seed_holding_every_rule_is_completed(self):
        verdict = self.verdict(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_one_tripped_falsifier_on_one_seed_falsifies(self):
        results = clean_seeds()
        put(results[43], "bitwise_mismatch_count.shift_513", 1)
        verdict = self.verdict(results, {43: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [43])
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_every_falsifier_family_falsifies(self):
        for path in [
            "revoked_upstream_page_reachable_count",
            "shift_hits_with_dead_original_upstream",
            "omitted_upstream_admissions",
            "superseded_upstream_page_reachable_count",
            "source_lineage_mismatch_admissions",
            "fresh_prefill_output_mismatches",
            f"invalidation_matrix_stale_hits.{CELLS[30]}",
            "shifted_reuse_inflight_outputs",
            "shifted_reuse_inflight_cache_inserts",
            "tainted_exact_results",
            "non_bit_exact_exact_results",
            "reuse_after_raw_ancestor_removed",
        ]:
            results = clean_seeds()
            put(results[17], path, 1)
            self.assertEqual(self.verdict(results, {17: 1})["verdict"], aggregate.FALSIFIED, path)

    def test_an_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
        for path in [
            "reuse_attempts_per_shift.shift_4096",
            "reused_tokens_per_shift.shift_1",
            "secondary_index_lookups",
            "nonaligned_insertion_cases.tokens_7",
            f"invalidation_matrix_cases.{CELLS[51]}",
            "shifted_reuse_inflight_race_cases.base_model_supersession",
            "raw_ancestor_omission_cases",
        ]:
            results = clean_seeds()
            put(results[101], path, 0)
            verdict = self.verdict(results, {101: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)
            self.assertEqual(verdict["vacuous_seeds"], [101], path)

    def test_fewer_completions_than_revocation_cases_is_vacuous(self):
        results = clean_seeds()
        put(results[29], "upstream_revocation_cases", 3)
        put(results[29], "fresh_prefills_completed_after_revocation", 2)
        verdict = self.verdict(results, {29: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["vacuous_seeds"], [29])
        put(results[29], "fresh_prefills_completed_after_revocation", 3)
        self.assertEqual(self.verdict(results)["verdict"], aggregate.COMPLETED)

    def test_the_two_case_id_set_digests_must_be_equal_as_sets_not_as_counts(self):
        results = clean_seeds()
        put(results[43], "fresh_prefill_completed_case_id_set_digest", "sha256:different")
        verdict = self.verdict(results, {43: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["vacuous_seeds"], [43])

    def test_a_missing_seed_is_invalid(self):
        results = clean_seeds()
        del results[71]
        verdict = self.verdict(results)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertIsNone(verdict["recommended_status"])
        self.assertTrue(any("71" in problem for problem in verdict["problems"]))

    def test_a_counter_or_member_that_is_missing_null_or_mistyped_is_invalid_not_failed(self):
        for damage in (
            lambda result: result["bitwise_mismatch_count"].pop("shift_64"),
            lambda result: result.pop("revoked_upstream_page_reachable_count"),
            lambda result: result.update(revoked_upstream_page_reachable_count=None),
            lambda result: result.update(revoked_upstream_page_reachable_count=0.0),
            lambda result: result.update(revoked_upstream_page_reachable_count=False),
            lambda result: result.pop("invalidated_case_id_set_digest"),
            lambda result: result["invalidation_matrix_stale_hits"].pop(CELLS[3]),
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
            lambda result: result["model_pins"]["pinned"].update(tiny_model_config_digest="sha256:91be"),
            lambda result: result.pop("model_pins"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "tainted_exact_results", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "omitted_upstream_admissions", 1)
        results[29] = None
        verdict = self.verdict(results, {17: 1, 29: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_every_seed_must_pass(self):
        for seed in SEEDS:
            results = clean_seeds()
            put(results[seed], "shift_hits_with_dead_original_upstream", 1)
            verdict = self.verdict(results, {seed: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, seed)
            self.assertEqual(verdict["falsified_seeds"], [seed])

    def test_judgements_do_not_alias_their_inputs(self):
        results = clean_seeds()
        before = copy.deepcopy(results)
        judged(results)
        self.assertEqual(results, before)


class R006Tests(unittest.TestCase):
    def test_an_unfrozen_experiment_is_not_aggregated(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_the_benchmark_and_experiment_are_r006s(self):
        self.assertEqual(aggregate.BENCHMARK, "positional-reuse-oracle")
        self.assertEqual(aggregate.EXPERIMENT, "R006")


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "bitwise_mismatch_count.shift_7", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS), seeds, PASSED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("bitwise_mismatch_count.shift_7 <= 0", text)
        self.assertIn("structural", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS)["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
