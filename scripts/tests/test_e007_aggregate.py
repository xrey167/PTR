"""The E007 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed` and
`decide`, which read no file, under the rule grammar `evaluate` implements. These tests build a
result that holds every rule of the criteria and then break one rule at a time, so a verdict
that stopped depending on it fails here. That every counter of the criteria resolves in the real
harness JSON, that every rule is one the harness judges and that every rule can fail in that JSON
is held on the Rust side (`e007` tests in bins/ptr-bench)."""

import copy
import importlib.util
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "system" / "E007-split-pod-kv"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("e007_aggregate", HERE / "aggregate.py")
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
        "benchmark": "split-pod-kv",
        "experiment": "E007",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True},
        "harness_gaps": [],
    }
    pending = []
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["validity"]:
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


def violate(result: dict, rule: str) -> None:
    """Put the left counter of `rule` at a value that breaks it."""
    left, op, right = aggregate.parse_rule(rule)
    value = aggregate.resolve(result, left)
    if isinstance(value, str):
        put(result, left, "sha256:different")
        return
    bound = int(right) if aggregate.NUMBER.fullmatch(right) else aggregate.resolve(result, right)
    put(result, left, bound - 1 if op == ">=" else bound + 1)


class CriteriaFileTests(unittest.TestCase):
    def test_every_rule_reads_and_the_sets_are_not_empty(self):
        for key in ("coverage", "falsifiers", "validity"):
            rules = CRITERIA[key]
            self.assertTrue(rules)
            self.assertEqual(len(rules), len(set(rules)), key)
            for rule in rules:
                aggregate.parse_rule(rule)
        self.assertEqual(len(CRITERIA["coverage"]), 75)
        self.assertEqual(len(CRITERIA["falsifiers"]), 48)
        self.assertEqual(len(CRITERIA["validity"]), 4)
        for name in ("bitwise_mismatch_count", "stale_upstream_kv_served", "wire_kv_page_bytes", "replayed_frames_admitted"):
            self.assertIn(f"{name} <= 0", CRITERIA["falsifiers"])
        for rule in (
            "abandoned_chunk_id_set_digest == released_chunk_id_set_digest",
            "valid_sent_frame_id_set_digest == consistent_digest_frame_id_set_digest",
            "frames_with_consistent_whole_frame_digest == valid_frames_sent",
            "abandoned_chunks_released >= abandoned_chunks_sent",
        ):
            self.assertIn(rule, CRITERIA["falsifiers"])
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_thresholds_of_config_toml_are_the_rules_literals(self):
        rules = {aggregate.parse_rule(rule) for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["validity"]}
        pinned = CONFIG["pinned_thresholds"]
        self.assertIn(("activation_frames_crossed", ">=", str(pinned["min_activation_frames_crossed"])), rules)
        self.assertIn(("valid_frames_sent", ">=", str(pinned["min_valid_frames_sent"])), rules)
        self.assertEqual(pinned["min_activation_frames_crossed"], pinned["sequences"] * pinned["decode_steps"])
        self.assertIn(("tombstones_retained_at_saturation", "==", str(pinned["tombstones_retained_at_saturation"])), rules)
        self.assertEqual(pinned["tombstones_retained_at_saturation"], pinned["max_frame_tombstones_per_principal"])
        self.assertIn(("abandoned_frames_in_saturation_case", "==", str(pinned["abandoned_frames_in_saturation_case"])), rules)
        self.assertIn(("revoke_messages_dropped_or_delayed", ">=", str(pinned["min_revoke_messages_dropped_or_delayed"])), rules)
        self.assertIn(("stale_upstream_entries_before_revocation", ">=", str(pinned["min_stale_upstream_entries_before_revocation"])), rules)
        for left, op, right in rules:
            if op == "<=":
                self.assertEqual(right, str(pinned["max_violations"]), left)
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        self.assertEqual(pinned["seeds"], SEEDS)
        cells = pinned["invalidation_coverage_matrix"]["cells"]
        self.assertEqual(len(cells), 6)
        for cell in cells:
            self.assertIn((f"invalidation_matrix_cases.{cell}", ">=", "1"), rules)
            self.assertIn((f"invalidation_matrix_stale_hits.{cell}", "<=", "0"), rules)

    def test_the_reference_result_holds_every_rule(self):
        result = passing_result(17)
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["validity"]:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        """No rule is vacuous: some value of its left counter breaks it."""
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["validity"]:
            result = passing_result(17)
            violate(result, rule)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    def test_the_mutation_plan_names_only_counters_the_aggregator_can_read(self):
        mirrors = {"late_window_delays_out_of_range", "abandoned_chunks_unreleased", "frames_without_consistent_digest", "abandoned_id_set_mismatch", "valid_id_set_mismatch"}
        cells = {f"stale_hits_in_{cell}" for cell in CONFIG["pinned_thresholds"]["invalidation_coverage_matrix"]["cells"]}
        for counter in MUTATIONS["hard_counters"]:
            self.assertTrue(counter in MANIFEST["metrics"].split(", ") or counter in mirrors | cells, counter)

    def test_the_config_table_and_the_registry_agree(self):
        """The [preregistration] table holds the required keys plus the two digests (placeholders until the freeze)
        and repeats [pinned_thresholds] value for value."""
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))[
            "experiment"
        ]["E007"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        self.assertEqual(table["harness"], "split-pod-kv")
        for key, value in CONFIG["pinned_thresholds"].items():
            if key == "invalidation_coverage_matrix":
                # An inline table is no preregistered type; the table holds its two lists flat.
                self.assertEqual(table["invalidation_coverage_matrix_kinds"], value["kinds"])
                self.assertEqual(table["invalidation_coverage_matrix_cells"], value["cells"])
            elif not key.startswith(("owner_decision_", "stated_limit_")):
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
            violate(results[43], rule)
            verdict = self.verdict(results, {43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["falsified_seeds"], [43], rule)
            self.assertEqual(verdict["recommended_status"], "failed")

    def test_an_unmet_coverage_rule_makes_the_run_vacuous_and_falsified(self):
        for rule in CRITERIA["coverage"]:
            if rule.startswith("control_trips."):
                # A control that did not trip makes the seed INVALID (its probe proves nothing).
                continue
            results = clean_seeds()
            violate(results[101], rule)
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
            lambda result: result.pop("bitwise_mismatch_count"),
            lambda result: result.update(bitwise_mismatch_count=None),
            lambda result: result.update(bitwise_mismatch_count=0.0),
            lambda result: result.update(bitwise_mismatch_count=False),
            lambda result: result.update(stale_upstream_kv_served="12"),
            lambda result: result["control_trips"].pop("denylist_admits_mislabelled_payload"),
            lambda result: result["invalidation_matrix_stale_hits"].pop("upstream_stage_generation_lag_revocation"),
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
            lambda result: result["control_trips"].update(naive_downstream_cache_serves_stale=0),
            lambda result: result["control_trips"].update(unchunked_large_frame_refused_by_podwire=0),
            lambda result: result.pop("control_trips"),
        ):
            results = clean_seeds()
            damage(results[17])
            self.assertEqual(self.verdict(results)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(self.verdict(clean_seeds(), {17: 1})["verdict"], aggregate.INVALID)
        results = clean_seeds()
        put(results[17], "bitwise_mismatch_count", 1)
        # The record contradicts itself (a tripped falsifier, status 0): the seed is not valid, and
        # the tripped falsifier counts all the same.
        verdict = self.verdict(results, {17: 0})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["invalid_seeds"], [17])
        self.assertEqual(self.verdict(clean_seeds(), {17: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(self.verdict(clean_seeds(), {17: None})["verdict"], aggregate.INVALID)

    def test_a_failing_validity_rule_makes_the_seed_invalid_never_falsified_or_vacuous(self):
        for rule in CRITERIA["validity"]:
            for exit_code in (1, 2):
                results = clean_seeds()
                violate(results[43], rule)
                judgement = judged(results, {43: exit_code})[SEEDS.index(43)]
                self.assertFalse(judgement["valid"], rule)
                self.assertEqual(judgement["falsifiers_failed"], [], rule)
                verdict = self.verdict(results, {43: exit_code})
                self.assertEqual(verdict["verdict"], aggregate.INVALID, (rule, exit_code))
                self.assertEqual(verdict["vacuous_seeds"], [], rule)
                self.assertEqual(verdict["falsified_seeds"], [], rule)

    def test_an_unreadable_validity_counter_counts_as_fired(self):
        # Missing, null or mistyped: the procedure cannot be shown, so the seed is INVALID and a
        # falsifier of the same record is ignored (reported, not counted).
        for rule in CRITERIA["validity"]:
            counter = aggregate.parse_rule(rule)[0]
            for label, damage in (("missing", "missing"), ("null", None), ("string", "0")):
                results = clean_seeds()
                put(results[43], "replayed_frames_admitted", 2)
                if damage == "missing":
                    del results[43][counter]
                else:
                    put(results[43], counter, damage)
                for exit_code in (1, 2):
                    judgement = judged(results, {43: exit_code})[SEEDS.index(43)]
                    self.assertFalse(judgement["valid"], (rule, label))
                    self.assertEqual(judgement["falsifiers_failed"], [], (rule, label))
                    self.assertIn("replayed_frames_admitted <= 0", judgement["ignored_falsifiers"], (rule, label))
                    verdict = self.verdict(results, {43: exit_code})
                    self.assertEqual(verdict["verdict"], aggregate.INVALID, (rule, label, exit_code))
                    self.assertEqual(verdict["falsified_seeds"], [], (rule, label))

    def test_a_failing_validity_rule_beats_a_falsifier_and_a_dead_control(self):
        # Validity first: the run did not follow the preregistered procedure, so its product
        # observations are listed as ignored and do not falsify.
        results = clean_seeds()
        put(results[17], "model_pin_mismatches", 1)
        put(results[17], "replayed_frames_admitted", 2)
        put(results[17], "control_trips.tombstone_free_assembler_admits_replay", 0)
        for exit_code in (1, 2):
            verdict = self.verdict(results, {17: exit_code})
            self.assertEqual(verdict["verdict"], aggregate.INVALID, exit_code)
            self.assertEqual(verdict["falsified_seeds"], [], exit_code)
            self.assertEqual(verdict["invalid_seeds"], [17], exit_code)
        judgement = judged(results, {17: 2})[SEEDS.index(17)]
        self.assertEqual(judgement["falsifiers_failed"], [])
        self.assertIn("replayed_frames_admitted <= 0", judgement["ignored_falsifiers"])

    def test_a_falsifier_beside_a_dead_control_with_clean_validity_falsifies(self):
        results = clean_seeds()
        put(results[17], "replayed_frames_admitted", 2)
        put(results[17], "control_trips.tombstone_free_assembler_admits_replay", 0)
        verdict = self.verdict(results, {17: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["falsified_seeds"], [17])
        self.assertEqual(verdict["invalid_seeds"], [17])

    def test_a_dead_control_without_a_falsifier_is_invalid_never_completed(self):
        results = clean_seeds()
        put(results[29], "control_trips.denylist_admits_mislabelled_payload", 0)
        for exit_code in (0, 1, 2):
            self.assertEqual(self.verdict(results, {29: exit_code})["verdict"], aggregate.INVALID, exit_code)

    def test_a_seed_without_a_result_is_invalid(self):
        results = clean_seeds()
        results[43] = None
        self.assertEqual(self.verdict(results, {43: 2})["verdict"], aggregate.INVALID)

    def test_a_falsified_seed_beats_an_invalid_one(self):
        results = clean_seeds()
        put(results[17], "bitwise_mismatch_count", 1)
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
        put(results[17], "replayed_frames_admitted", 4)
        verdict = aggregate.decide(judged(results, {17: 1}), PASSED, SEEDS, None)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)

    def test_the_plan_lists_the_mutations_the_evidence_must_count(self):
        self.assertGreaterEqual(len(MUTATIONS["mutation"]), 40)
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)


class E007Tests(unittest.TestCase):
    def test_every_falsifier_counter_that_is_an_upper_bound_is_a_hard_counter_of_the_mutation_plan(self):
        hard = set(MUTATIONS["hard_counters"])
        for rule in CRITERIA["falsifiers"]:
            left, op, _ = aggregate.parse_rule(rule)
            if op == "<=" and "." not in left:
                self.assertIn(left, hard, rule)
        for cell in CONFIG["pinned_thresholds"]["invalidation_coverage_matrix"]["cells"]:
            self.assertIn(f"stale_hits_in_{cell}", hard)

    def test_the_scope_limits_are_stated_in_the_config_and_the_aggregate(self):
        pinned = CONFIG["pinned_thresholds"]
        self.assertIn("in process", pinned["stated_limit_transport"])
        self.assertIn("plumbing", pinned["stated_limit_model"])
        text = " ".join(aggregate.LIMITATIONS)
        self.assertIn("checks the plumbing", text)
        self.assertIn("no socket, no iroh transport", text)
        self.assertIn("E008", text)

    def test_a_control_that_did_not_trip_is_never_completed(self):
        for name in ("naive_downstream_cache_serves_stale", "chain_keyed_assembler_mixes_frames"):
            results = clean_seeds()
            results[29]["control_trips"][name] = 0
            verdict = aggregate.decide(judged(results, {29: 1}), PASSED, SEEDS, KILLED)
            self.assertNotEqual(verdict["verdict"], aggregate.COMPLETED, name)

    def test_a_foreign_record_is_invalid_and_never_falsifies(self):
        for tamper in (
            lambda result: result["procedure"].update(seed_is_declared=False),
            lambda result: result.update(scale="custom"),
            lambda result: result.update(seed=17),
        ):
            results = clean_seeds()
            put(results[43], "stale_upstream_kv_served", 3)
            tamper(results[43])
            verdict = aggregate.decide(judged(results, {43: 1}), PASSED, SEEDS, KILLED)
            self.assertEqual(verdict["verdict"], aggregate.INVALID)
            self.assertEqual(verdict["falsified_seeds"], [])
            self.assertEqual(verdict["invalid_seeds"], [43])

    def test_a_missing_seed_record_is_invalid_not_completed(self):
        results = clean_seeds()
        del results[43]
        verdict = aggregate.decide(judged(results, {}), PASSED, SEEDS, KILLED)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any("43" in problem for problem in verdict["problems"]))

    def test_a_falsifier_in_a_record_that_is_invalid_for_another_reason_still_falsifies(self):
        results = clean_seeds()
        put(results[43], "stale_upstream_kv_served", 3)
        results[43]["control_trips"]["naive_downstream_cache_serves_stale"] = 0
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
        self.assertEqual(MANIFEST["id"], "E007")
        registry = {e["id"]: e for e in tomllib.loads((ROOT / "experiments" / "registry.toml").read_text(encoding="utf-8"))["experiment"]}
        self.assertEqual(registry["E007"]["status"], MANIFEST["status"])
        self.assertEqual(registry["E007"]["path"], "system/E007-split-pod-kv")
        self.assertIn("E008", registry)
        self.assertEqual(registry["E008"]["status"], "planned")


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_and_every_failed_rule(self):
        results = clean_seeds()
        put(results[17], "bitwise_mismatch_count", 1)
        seeds = judged(results, {17: 1})
        text = aggregate.report(aggregate.decide(seeds, PASSED, SEEDS, KILLED), seeds, PASSED, KILLED)
        self.assertIn("FALSIFIED", text)
        self.assertIn("bitwise_mismatch_count <= 0", text)
        self.assertIn("mutation check", text)

    def test_no_static_evidence_means_the_verdict_rests_on_the_seeds_and_the_mutation_evidence(self):
        self.assertEqual(aggregate.static_evidence([]), [])
        self.assertEqual(aggregate.decide(judged(clean_seeds()), [], SEEDS, KILLED)["verdict"], aggregate.COMPLETED)


if __name__ == "__main__":
    unittest.main()
