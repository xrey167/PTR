"""The R011 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the frozen criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, and that every rule is one the harness judges, is held on the Rust side
(`r011` tests in bins/ptr-bench)."""

import copy
import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "runtime" / "R011-self-modification-safety"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("r011_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
SEEDS = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence


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
        "benchmark": "self-modification-safety",
        "experiment": "R011",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True},
        "harness_gaps": [],
    }
    pending = []
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
        left, op, right = aggregate.parse_rule(rule)
        if aggregate.NUMBER.fullmatch(right):
            value = int(right)
            put(result, left, value)
        else:
            pending.append((left, op, right))
    for left, _, right in pending:
        known = [value for value in (lookup(result, left), lookup(result, right)) if value is not None]
        value = known[0] if known else "sha256:0" if left.endswith("digest") else 0
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
        self.assertIn("uncommitted_effective_changes <= 0", CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)

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
                put(result, left, value - 1)
            else:
                put(result, left, value + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    @unittest.skipUnless(
        "R011" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "R011 is not in experiments/preregistration.toml until the freeze",
    )
    def test_the_config_table_and_the_registry_agree(self):
        """Skipped while R011 is absent from preregistration.toml (it is added at the freeze)."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["R011"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "self-modification-safety")
        for key, kind in required.items():
            value = table[key]
            expected = {"int": int, "str": str, "file": str, "int-list": list, "str-list": list}[kind]
            self.assertIs(type(value), expected, key)
        for key in ("criteria", "preregistration_text"):
            self.assertTrue((ROOT / table[key]).is_file(), key)


class VerdictTests(unittest.TestCase):
    def verdict(self, results, exits=None, static=PASSED):
        return aggregate.decide(judged(results, exits), static, SEEDS)

    def test_every_seed_holding_every_rule_with_the_static_evidence_is_completed(self):
        verdict = self.verdict(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_one_tripped_falsifier_on_one_seed_falsifies(self):
        results = clean_seeds()
        put(results[43], "uncommitted_effective_changes", 1)
        verdict = self.verdict(results, {43: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [43])
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_a_second_and_third_falsifier_falsify(self):
        for path, value in [("stale_generation_commits_accepted_by_type.Merge", 2), ("routable_after_tightening_rotation", 1)]:
            results = clean_seeds()
            put(results[17], path, value)
            self.assertEqual(self.verdict(results, {17: 1})["verdict"], aggregate.FALSIFIED, path)

    def test_an_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
        results = clean_seeds()
        put(results[101], "runtime_fence_with_grant_succeeded", 0)
        verdict = self.verdict(results, {101: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["vacuous_seeds"], [101])

    def test_a_missing_seed_is_invalid(self):
        results = clean_seeds()
        del results[71]
        verdict = self.verdict(results)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertIsNone(verdict["recommended_status"])
        self.assertTrue(any("71" in problem for problem in verdict["problems"]))

    def test_a_counter_that_is_missing_null_or_mistyped_is_invalid_not_failed(self):
        for damage in (
            lambda result: result.pop("uncommitted_effective_changes"),
            lambda result: result.update(uncommitted_effective_changes=None),
            lambda result: result.update(uncommitted_effective_changes=0.0),
            lambda result: result.update(uncommitted_effective_changes=False),
            lambda result: result.pop("runtime_fence_with_grant_succeeded"),
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
            lambda result: result.update(harness_gaps=["x"]),
            lambda result: result["procedure"].update(seed_is_declared=False),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "uncommitted_effective_changes", 1)
        self.assertEqual(self.verdict(results, {17: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "uncommitted_effective_changes", 1)
        results[29] = None
        verdict = self.verdict(results, {17: 1, 29: 2})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [29])

    def test_judgements_do_not_alias_their_inputs(self):
        results = clean_seeds()
        before = copy.deepcopy(results)
        judged(results)
        self.assertEqual(results, before)


class ManifestTests(unittest.TestCase):
    MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))

    def test_the_required_artifacts_are_the_ones_the_aggregate_publishes(self):
        """`adversarial_attempts.json` was required and written by nothing: the per-case attempts are in each
        seed's harness record and reach metrics.json through the manifest's metric list."""
        self.assertEqual(self.MANIFEST["required_artifacts"], ["run.json", "metrics.json"])
        self.assertIn("adversarial_attempts_per_case", self.MANIFEST["metrics"].split(", "))
        self.assertNotIn("attempts_artifact", CONFIG["pinned_thresholds"])
        for name in ("experiment.toml", "config.toml", "criteria.toml", "aggregate.py", "PREREGISTRATION.md"):
            self.assertNotIn("adversarial_attempts.json", (HERE / name).read_text(encoding="utf-8"), name)

    def test_the_two_compound_events_are_in_the_direct_append_family(self):
        cases = CONFIG["pinned_thresholds"]["adversarial_cases"]
        types = CONFIG["pinned_thresholds"]["direct_append_event_types"]
        for name in ("InputInvalidated", "ExpertAuthorityChanged"):
            self.assertIn(name, types)
            self.assertIn(f"direct_ledger_append_{name}", cases)
            self.assertIn(f"adversarial_attempts_per_case.direct_ledger_append_{name} >= 1", CRITERIA["coverage"])
            self.assertIn(f"direct_ledger_append_accepted_by_type.{name} <= 0", CRITERIA["falsifiers"])
        self.assertEqual(len(cases), 91)
        self.assertEqual(len(types), 11)


class FenceFindingTests(unittest.TestCase):
    """Under-fencing and over-fencing are findings (exit 1): a falsification, never an invalid seed."""

    def test_a_fence_set_mismatch_falsifies_and_is_not_invalid(self):
        results = clean_seeds()
        put(results[29], "fence_set_digest_mismatches", 1)
        verdict = aggregate.decide(judged(results, {29: 1}), PASSED, SEEDS)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [29])
        self.assertEqual(verdict["invalid_seeds"], [])

    def test_a_stale_hit_in_a_cell_and_an_accepted_compound_append_falsify(self):
        for path in (
            "invalidation_matrix_stale_hits.training_input_normal_revocation",
            "direct_ledger_append_accepted_by_type.InputInvalidated",
            "direct_ledger_append_accepted_by_type.ExpertAuthorityChanged",
        ):
            results = clean_seeds()
            put(results[43], path, 1)
            verdict = aggregate.decide(judged(results, {43: 1}), PASSED, SEEDS)
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, path)

    def test_a_set_digest_mismatch_of_the_fenced_and_derived_sets_falsifies(self):
        results = clean_seeds()
        put(results[17], "fenced_and_probed_dependents_set_digest", "sha256:other")
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "uncommitted_effective_changes", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS), seeds, PASSED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("uncommitted_effective_changes <= 0", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS)["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
