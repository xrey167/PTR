"""The R013 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`r013` tests in bins/ptr-bench)."""

import copy
import importlib.util
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "runtime" / "R013-deferred-self-modification-authority"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("r013_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
KILLED = {"killed": len(MUTATIONS["mutation"]), "total": len(MUTATIONS["mutation"])}
SELECTION = CONFIG["pinned_thresholds"]["selection_digest"]


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
        "benchmark": "self-modification-authority",
        "experiment": "R013",
        "seed": seed,
        "scale": "pinned",
        "procedure": {
            "seed_is_declared": True,
            "shape_is_pinned": True,
            "selection_digest": SELECTION,
            "selection_digest_pinned": SELECTION,
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
        self.assertEqual(len(CRITERIA["coverage"]), 63)
        self.assertEqual(len(CRITERIA["falsifiers"]), 20)
        for name in (
            "structural_inflight_outputs_after_invalidation",
            "reactivations_from_compromised_seed_admitted",
            "mismatched_attestation_admissions",
            "untrusted_attestor_admissions",
            "self_trusted_attestor_admissions",
            "wrong_typed_refusals",
            "skipped_cells",
            "attestor_quarantine_readmissions_admitted",
            "product_panics",
        ):
            self.assertIn(f"{name} <= 0", CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        """Every min_X of [pinned_thresholds] is the literal of the rule(s) on X, and every falsifier
        is held at the one literal max_every_falsifier."""
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]}
        pinned = CONFIG["pinned_thresholds"]
        per_member = {
            "min_attempts_per_case": "adversarial_attempts_per_case.",
            "min_structural_inflight_race_cases": "structural_inflight_race_cases.",
            "min_transitive_authority_cases": "transitive_authority_revocation_cases.",
            "min_revoke_while_retired_cases": "revoke_while_retired_cases.",
            "min_reactivation_controls": "reactivation_controls_accepted.",
            "min_superseded_source_cases": "superseded_source_revocation_cases.",
        }
        checked = 0
        for key, value in pinned.items():
            if key in per_member:
                wanted = [rule for rule in rules if rule[0].startswith(per_member[key])]
                self.assertTrue(wanted, key)
                for rule in wanted:
                    self.assertEqual((rule[1], rule[2]), (">=", str(value)), key)
                checked += 1
            elif key.startswith("min_"):
                self.assertIn((key[4:], ">=", str(value)), rules, key)
                checked += 1
        for left, op, right in rules:
            if op == "<=":
                self.assertEqual(right, str(pinned["max_every_falsifier"]), left)
        self.assertEqual(checked, 19)
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        self.assertEqual(pinned["seeds"], SEEDS)

    def test_the_suite_of_config_toml_is_the_suite_of_the_attempt_rules(self):
        cases = CONFIG["pinned_thresholds"]["adversarial_cases"]
        rules = [aggregate.parse_rule(rule)[0] for rule in CRITERIA["coverage"]]
        self.assertEqual(
            [rule for rule in rules if rule.startswith("adversarial_attempts_per_case.")],
            [f"adversarial_attempts_per_case.{case}" for case in cases],
        )
        self.assertEqual(len(cases), len(set(cases)))

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
            if op == ">=":
                put(result, left, int(right) - 1)
            else:
                put(result, left, value + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_the_mutation_plan_names_only_counters_the_aggregator_can_read(self):
        for counter in MUTATIONS["hard_counters"]:
            self.assertIn(counter, MANIFEST["metrics"].split(", "))

    def test_the_config_table_and_the_registry_agree(self):
        """The [preregistration] table holds the required keys plus the two digests (placeholders until the freeze)
        and repeats [pinned_thresholds] value for value."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["R013"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "self-modification-authority")
        for key, value in CONFIG["pinned_thresholds"].items():
            if not key.startswith(("owner_decision_", "stated_limit_")):
                self.assertEqual(table[key], value, key)
        for key in ("criteria_sha256", "preregistration_text_sha256"):
            self.assertTrue(table[key] == "must-be-pinned-before-prepared" or re.fullmatch(r"[0-9a-f]{64}", table[key]), key)
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
            lambda result: result.pop("product_panics"),
            lambda result: result.update(product_panics=None),
            lambda result: result.update(product_panics=0.0),
            lambda result: result.update(product_panics=False),
            lambda result: result.update(bystanders_closed="12"),
            lambda result: result["revoke_while_retired_cases"].pop("attestor"),
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
            lambda result: result.update(experiment="R011"),
            lambda result: result.update(harness_gaps=["x"]),
            lambda result: result["procedure"].update(seed_is_declared=False),
            lambda result: result["procedure"].update(shape_is_pinned=False),
            lambda result: result["procedure"].update(selection_digest="sha256:" + "1" * 64),
            lambda result: result["procedure"].update(selection_digest_pinned="sha256:" + "1" * 64),
            lambda result: result.pop("procedure"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "product_panics", 1)
        # The record contradicts itself (a tripped falsifier, status 0): the seed is not valid, and
        # the tripped falsifier counts all the same.
        verdict = self.verdict(results, {17: 0})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [17])
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "product_panics", 1)
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
        total = len(MUTATIONS["mutation"])
        for mutations in (None, {"killed": total - 1, "total": total}, {"killed": 0, "total": total}):
            verdict = aggregate.decide(seeds, PASSED, SEEDS, mutations)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, mutations)
            self.assertIsNone(verdict["recommended_status"])
            self.assertTrue(any("mutation" in problem for problem in verdict["problems"]))

    def test_a_falsified_seed_is_falsified_whatever_the_mutation_evidence(self):
        results = clean_seeds()
        put(results[17], "reactivations_from_compromised_seed_admitted", 4)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, None)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)

    def test_the_plan_lists_the_mutations_the_evidence_must_count(self):
        self.assertGreaterEqual(len(MUTATIONS["mutation"]), 16)
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)


class R013Tests(unittest.TestCase):
    def test_the_controls_the_cases_and_the_timings_are_descriptive_or_coverage(self):
        rules = " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for name in ("reactivation_attempts", "attestation_controls_by_operation", "_ns", "timing"):
            self.assertNotIn(name, rules)
        self.assertFalse(CONFIG["pinned_thresholds"]["timing_gated"])

    def test_every_falsifier_is_a_hard_counter_of_the_mutation_plan(self):
        self.assertEqual([rule.split(" ")[0] for rule in CRITERIA["falsifiers"]], MUTATIONS["hard_counters"])
        for rule in CRITERIA["falsifiers"]:
            self.assertTrue(rule.endswith(" <= 0"), rule)

    def test_the_stated_limits_are_in_the_config_and_the_aggregate(self):
        pinned = CONFIG["pinned_thresholds"]
        self.assertIn("R014", pinned["owner_decision_2026_10_10_scope"])
        self.assertIn("Ed25519", pinned["stated_limit_attestation"])
        self.assertIn("monotone", pinned["stated_limit_transitive_authority"])
        self.assertIn("raw ledger writer", pinned["stated_limit_raw_ledger_writer"])
        text = " ".join(aggregate.LIMITATIONS)
        self.assertIn("not history", text)
        self.assertIn("epochs are monotone", text)
        self.assertIn("a raw ledger writer is outside the claim", text)
        self.assertIn("R014", text)

    def test_a_suite_that_is_not_the_pinned_one_invalidates_and_does_not_falsify(self):
        results = clean_seeds()
        results[29]["procedure"]["selection_digest"] = "sha256:" + "2" * 64
        verdict = aggregate.decide(judged(results), PASSED, SEEDS, KILLED)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["falsified_seeds"], [])

    def test_an_admitted_compromised_reactivation_falsifies_and_is_never_invalid(self):
        results = clean_seeds()
        put(results[29], "reactivations_from_compromised_seed_admitted", 2)
        put(results[29], "compromised_reactivations_routable", 2)
        verdict = aggregate.decide(judged(results, {29: 1}), PASSED, SEEDS, KILLED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [29])
        self.assertEqual(verdict["invalid_seeds"], [])
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_an_admitted_mismatched_attestation_falsifies_and_is_never_invalid(self):
        results = clean_seeds()
        put(results[71], "mismatched_attestation_admissions", 1)
        verdict = aggregate.decide(judged(results, {71: 1}), PASSED, SEEDS, KILLED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [71])
        self.assertEqual(verdict["invalid_seeds"], [])

    def test_a_falsifier_in_a_record_that_is_invalid_for_another_reason_still_falsifies(self):
        """A run whose suite is not the pinned one is invalid, and a call that escaped the cancellation falsifies all the same."""
        results = clean_seeds()
        put(results[43], "structural_inflight_outputs_after_invalidation", 3)
        results[43]["procedure"]["selection_digest"] = "sha256:" + "3" * 64
        verdict = aggregate.decide(judged(results, {43: 1}), PASSED, SEEDS, KILLED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [43])
        self.assertEqual(verdict["invalid_seeds"], [43])

    def test_an_unfrozen_experiment_is_not_aggregated(self):
        """Holds while a digest is still a placeholder; once both are pinned the frozen files are read."""
        table = CONFIG.get("preregistration", {})
        if any("must-be-pinned" in str(table.get(key, "must-be-pinned")) for key in ("criteria_sha256", "preregistration_text_sha256")):
            with self.assertRaises(SystemExit):
                aggregate.load_frozen()

    def test_the_status_is_planned_or_later_and_the_registry_agrees(self):
        later = ("planned", "prepared", "running", "completed", "failed")
        self.assertIn(MANIFEST["status"], later)
        self.assertEqual(MANIFEST["id"], "R013")
        registry = {e["id"]: e for e in tomllib.loads((ROOT / "experiments" / "registry.toml").read_text(encoding="utf-8"))["experiment"]}
        self.assertEqual(registry["R013"]["status"], MANIFEST["status"])
        self.assertEqual(registry["R013"]["path"], "runtime/R013-deferred-self-modification-authority")
        self.assertIn("preflight", MANIFEST)
        self.assertIn("R014", registry)
        self.assertEqual(registry["R014"]["status"], "planned")

    def test_the_manifest_states_the_scope_change_and_names_the_successor(self):
        text = MANIFEST["hypothesis"]
        self.assertIn("narrowed on 2026-10-10", text)
        self.assertIn("R014", text)
        successor = tomllib.loads(
            (ROOT / "experiments" / "runtime" / "R014-deferred-authority-mechanisms" / "experiment.toml").read_text(encoding="utf-8")
        )
        self.assertIn("R013", successor["hypothesis"])
        mine = set(MANIFEST["metrics"].split(", "))
        theirs = set(successor["metrics"].split(", "))
        self.assertFalse(mine & theirs, "a metric is carried by both experiments")


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "product_panics", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, KILLED), seeds, PASSED, KILLED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("product_panics <= 0", text)
        self.assertIn("mutation check", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS, KILLED)["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
