"""The L001-v2 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`l001` tests in bins/ptr-bench)."""

import copy
import importlib.util
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "lifecycle" / "L001-v2-revocation-crash"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("l001v2_aggregate", HERE / "aggregate.py")
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
    lie = {"required_to_trip": True, "lie_images": 20, "naive_accepted_stale": 12, "anchored_refused": 20}
    result = {
        "benchmark": "revocation-crash",
        "experiment": "L001-v2",
        "seed": seed,
        "scale": "pinned",
        "procedure": {
            "seed_is_declared": True,
            "shape_is_pinned": True,
            "children_ran": True,
            "durability_probe_available": True,
        },
        "controls": {"file_lie": dict(lie), "ack_lie": dict(lie)},
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
        self.assertEqual(len(CRITERIA["coverage"]), 43)
        self.assertEqual(len(CRITERIA["falsifiers"]), 14)
        for name in ("stale_accepts", "unsynced_acks", "raft_revocation_visible_before_commit", "product_hangs", "anchored_lie_accepted"):
            self.assertIn(f"{name} <= 0", CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        """Every min_X of [pinned_thresholds] is the literal of the rule on X (the grouped failpoint rules
        share min_failpoint_hits), and every falsifier is held at the one literal max_every_falsifier."""
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]}
        pinned = CONFIG["pinned_thresholds"]
        aliases = {"min_subjects": "subjects_min", "min_events": "events_min"}
        checked = 0
        for key, value in pinned.items():
            if key == "min_failpoint_hits":
                grouped = [rule for rule in rules if rule[0].startswith("failpoint_hits.")]
                self.assertEqual(len(grouped), 12)
                for left, op, right in grouped:
                    self.assertEqual((op, right), (">=", str(value)), left)
                checked += 1
            elif key.startswith("min_"):
                self.assertIn((aliases.get(key, key[4:]), ">=", str(value)), rules, key)
                checked += 1
        for left, op, right in rules:
            if op == "<=":
                self.assertEqual(right, str(pinned["max_every_falsifier"]), left)
        self.assertGreaterEqual(checked, 25)
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
        ]["L001-v2"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "revocation-crash")
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
            lambda result: result.pop("stale_accepts"),
            lambda result: result.update(stale_accepts=None),
            lambda result: result.update(stale_accepts=0.0),
            lambda result: result.update(stale_accepts=False),
            lambda result: result.update(unsynced_acks="12"),
            lambda result: result["failpoint_hits"].pop("engine_sync"),
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
            lambda result: result["procedure"].update(children_ran=False),
            lambda result: result["procedure"].update(durability_probe_available=False),
            lambda result: result["controls"]["file_lie"].update(naive_accepted_stale=0),
            lambda result: result["controls"]["ack_lie"].update(naive_accepted_stale=0),
            lambda result: result["controls"]["ack_lie"].update(required_to_trip=False),
            lambda result: result["controls"].pop("ack_lie"),
            lambda result: result.pop("controls"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "stale_accepts", 1)
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
        put(results[17], "stale_accepts", 1)
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
        for mutations in (None, {"killed": 22, "total": 23}, {"killed": 0, "total": 23}):
            verdict = aggregate.decide(seeds, PASSED, SEEDS, mutations)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, mutations)
            self.assertIsNone(verdict["recommended_status"])
            self.assertTrue(any("mutation" in problem for problem in verdict["problems"]))

    def test_a_falsified_seed_is_falsified_whatever_the_mutation_evidence(self):
        results = clean_seeds()
        put(results[17], "recovered_not_prefix", 4)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, None)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)

    def test_the_plan_lists_the_mutations_the_evidence_must_count(self):
        self.assertEqual(len(MUTATIONS["mutation"]), 24)
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)


class L001V2Tests(unittest.TestCase):
    def test_the_engine_lie_and_the_availability_counters_are_descriptive(self):
        rules = " ".join(CRITERIA["coverage"] + CRITERIA["falsifiers"])
        for name in ("undetected_rollback", "engine_lie", "refusals_corrupt_tail", "refusals_lost_ack", "b1_", "_ns"):
            self.assertNotIn(name, rules)
        self.assertFalse(CONFIG["pinned_thresholds"]["engine_lie_gated"])
        self.assertFalse(CONFIG["pinned_thresholds"]["timing_gated"])

    def test_every_falsifier_is_a_hard_counter_of_the_mutation_plan(self):
        self.assertEqual([rule.split(" ")[0] for rule in CRITERIA["falsifiers"]], MUTATIONS["hard_counters"])
        for rule in CRITERIA["falsifiers"]:
            self.assertTrue(rule.endswith(" <= 0"), rule)

    def test_the_two_owner_limits_are_stated_in_the_config_and_the_aggregate(self):
        pinned = CONFIG["pinned_thresholds"]
        self.assertIn("RaftEngineLedger", pinned["stated_limit_raft_engine_anchor"])
        self.assertIn("fsync-honest", pinned["stated_limit_power_loss"])
        self.assertIn("environment.filesystem", pinned["stated_limit_power_loss"])
        text = " ".join(aggregate.LIMITATIONS)
        self.assertIn("RaftEngineLedger has no hash chain and no protected anchor", text)
        self.assertIn("fsync-honest crash-image model", text)
        self.assertIn("not a real power cut", text)
        self.assertIn("no product work is attempted", text)

    def test_a_lie_that_does_not_fool_the_naive_recovery_invalidates_and_does_not_falsify(self):
        for update in ({"naive_accepted_stale": 0}, {"lie_images": 0}):
            results = clean_seeds()
            results[29]["controls"]["ack_lie"].update(update)
            verdict = aggregate.decide(judged(results), PASSED, SEEDS, KILLED)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, update)
            self.assertEqual(verdict["falsified_seeds"], [])

    def test_a_lie_the_anchored_product_accepted_falsifies_and_is_never_invalid(self):
        results = clean_seeds()
        put(results[29], "anchored_lie_accepted", 3)
        put(results[29], "recovered_not_prefix", 3)
        results[29]["controls"]["ack_lie"]["anchored_refused"] = 17
        verdict = aggregate.decide(judged(results, {29: 1}), PASSED, SEEDS, KILLED)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [29])
        self.assertEqual(verdict["invalid_seeds"], [])
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_a_falsifier_in_a_record_that_is_invalid_for_another_reason_still_falsifies(self):
        """A run cut short by the watchdog has fewer lie images and cannot judge its control, and
        it is falsified by the hang all the same."""
        results = clean_seeds()
        put(results[43], "product_hangs", 3)
        results[43]["controls"]["file_lie"].update(naive_accepted_stale=0, lie_images=0)
        put(results[43], "images_lost_bytes", 0)
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

    def test_the_status_is_planned_or_later_and_the_successor_pointers_are_in_place(self):
        later = ("planned", "prepared", "running", "completed", "failed")
        self.assertIn(MANIFEST["status"], later)
        self.assertEqual(MANIFEST["id"], "L001-v2")
        old = tomllib.loads((ROOT / "experiments" / "lifecycle" / "L001-revocation-crash" / "experiment.toml").read_text(encoding="utf-8"))
        self.assertEqual(old["status"], "superseded")
        registry = {e["id"]: e for e in tomllib.loads((ROOT / "experiments" / "registry.toml").read_text(encoding="utf-8"))["experiment"]}
        self.assertEqual(registry["L001"]["status"], "superseded")
        self.assertEqual(registry["L001-v2"]["status"], MANIFEST["status"])
        self.assertEqual(registry["L001-v2"]["path"], "lifecycle/L001-v2-revocation-crash")
        readme = (ROOT / "experiments" / "lifecycle" / "L001-revocation-crash" / "README.md").read_text(encoding="utf-8")
        self.assertIn("L001-v2-revocation-crash", readme)


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "stale_accepts", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, KILLED), seeds, PASSED, KILLED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("stale_accepts <= 0", text)
        self.assertIn("mutation check", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS, KILLED)["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
