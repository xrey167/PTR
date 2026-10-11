"""The C008 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge`, `paired_comparisons` and `verdict_of`, which read no
file. These tests build a result that holds every rule of the criteria and break one rule at a time, so a verdict that stopped depending on it
fails here. The paired bounds are pinned to hand-computed values. Outcomes: structural FALSIFIED, INVALID (integrity, no headroom, untrained
comparators or schema, unconverged, unresolved control, a null world that shows an effect, missing evidence), refuted FALSIFIED (the Bonferroni
upper bound below the margin), NOT_SHOWN (straddling) and COMPLETED (all four lower bounds at the margin). That the harness's rules equal
criteria.toml and that every hard counter is planted by a mutation is held by research/cseries/tests/test_c008.py."""

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "consciousness" / "C008-attention-schema-routing-cpu"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/cseries/lib"))

spec = importlib.util.spec_from_file_location("c008_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)
import agg  # noqa: E402

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
PAIRED = CRITERIA["paired"]
SEEDS = list(range(7001, 7001 + PAIRED["seeds_required"]))
PASSED: list[dict] = []
PLAN = len(MUTATIONS["mutation"])
KILLED = {"killed": PLAN, "total": PLAN}
MARGIN = PAIRED["margin_permille"]
TOTAL = 1000
ARMS = ("control", "schema_free", "schema_free_wide", "heuristic", "describe_only", "history_router", "control_shuffled", "schema_free_epsilon", "random_router",
        "even_split", "control_null", "schema_free_null", "oracle")
BASE = {"control": 800, "schema_free": 700, "schema_free_wide": 705, "heuristic": 710, "describe_only": 702, "history_router": 704, "control_shuffled": 690,
        "schema_free_epsilon": 640, "random_router": 250, "even_split": 720, "control_null": 800, "schema_free_null": 800, "oracle": 950}
ENVIRONMENT = {k: "x" for k in agg.ENVIRONMENT_KEYS}


def put(result: dict, path: str, value) -> None:
    at = result
    keys = path.split(".")
    for key in keys[:-1]:
        at = at.setdefault(key, {})
    at[keys[-1]] = value


def passing_result(seed: int, reward: dict | None = None) -> dict:
    """A harness result in which every rule and guard of the criteria holds."""
    given = reward or {}
    reward = {**BASE, **given}
    if "schema_free" in given:  # the graded control follows the arm it degrades unless a test sets it
        reward["schema_free_epsilon"] = given.get("schema_free_epsilon", reward["schema_free"] - 60)
    result = {
        "benchmark": "c008", "experiment": "C008", "seed": seed, "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "controls": {name: {"required_rule_tripped": True} for name in aggregate.CONTROLS},
        "harness_gaps": [], "environment": dict(ENVIRONMENT), "schema_trained": 1,
        "arms": {arm: {"reward_correct": reward[arm], "reward_total": TOTAL} for arm in ARMS},
    }
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["integrity"]:
        left, _, right = agg.parse_rule(rule)
        put(result, left, int(right))
    best = max(reward[a] for a in ("schema_free", "schema_free_wide", "heuristic", "describe_only", "history_router"))
    result["headroom_permille"] = reward["oracle"] - best
    result["comparator_above_random_permille"] = best - reward["random_router"]
    result["unconverged_judged_arms"] = 0
    return result


def judged(results: dict, exits: dict | None = None) -> list[dict]:
    exits = exits or {}
    return [aggregate.judge(results[seed], exits.get(seed, 0), CRITERIA, seed) for seed in results]


def clean_seeds(reward: dict | None = None) -> dict:
    return {seed: passing_result(seed, reward) for seed in SEEDS}


def paired(results: dict) -> dict:
    return aggregate.paired_comparisons(results, PAIRED)


def decide(results, exits=None, mutations=KILLED, use_paired=True, plan=PLAN):
    outcome = paired(results) if use_paired else None
    return aggregate.verdict_of(judged(results, exits), PASSED, SEEDS, mutations, outcome, PAIRED, plan)


def spread(centre: int, deviation: int, i: int) -> int:
    return centre + (deviation if i % 2 else -deviation)


class CriteriaFileTests(unittest.TestCase):
    def test_every_rule_reads_and_the_sets_are_not_empty(self):
        for key in ("coverage", "guards", "falsifiers", "integrity"):
            rules = CRITERIA[key]
            self.assertTrue(rules)
            self.assertEqual(len(rules), len(set(rules)), key)
            for rule in rules:
                agg.parse_rule(rule)
        self.assertEqual(len(CRITERIA["falsifiers"]), 3)
        self.assertEqual(len(CRITERIA["integrity"]), 12)
        self.assertEqual(len(CRITERIA["guards"]), 2)
        self.assertNotIn("static_evidence", CRITERIA)

    def test_a_synthetic_passing_result_holds_every_rule(self):
        entry = aggregate.judge(passing_result(SEEDS[0]), 0, CRITERIA, SEEDS[0])
        self.assertTrue(entry["valid"], entry["invalid"])
        self.assertEqual((entry["coverage_failed"], entry["falsifiers_failed"], entry["integrity_failed"], entry["guards_failed"]), ([], [], [], []))

    def test_the_manifest_declares_the_seeds_the_criteria_require(self):
        self.assertEqual(MANIFEST["seeds"], SEEDS)
        self.assertEqual(len(MANIFEST["seeds"]), PAIRED["seeds_required"])


class JudgeSeedTests(unittest.TestCase):
    def test_a_broken_structural_counter_is_a_falsifier_and_exits_one(self):
        for rule in CRITERIA["falsifiers"]:
            r = passing_result(SEEDS[0])
            put(r, rule.split(" ")[0], 1)
            entry = aggregate.judge(r, 1, CRITERIA, SEEDS[0])
            self.assertTrue(entry["valid"], rule)
            self.assertEqual(entry["falsifiers_failed"], [rule])

    def test_a_broken_coverage_rule_is_unmet(self):
        for rule in CRITERIA["coverage"]:
            r = passing_result(SEEDS[0])
            put(r, rule.split(" ")[0], 0)
            self.assertEqual(aggregate.judge(r, 1, CRITERIA, SEEDS[0])["coverage_failed"], [rule])

    def test_a_broken_integrity_counter_makes_the_seed_invalid(self):
        for rule in CRITERIA["integrity"]:
            r = passing_result(SEEDS[0])
            put(r, rule.split(" ")[0], 1)
            entry = aggregate.judge(r, 2, CRITERIA, SEEDS[0])
            self.assertFalse(entry["valid"], rule)
            self.assertEqual(entry["integrity_failed"], [rule])

    def test_each_guard_fails_without_invalidating_the_seed(self):
        r = passing_result(SEEDS[0])
        r["headroom_permille"] = 39
        entry = aggregate.judge(r, 0, CRITERIA, SEEDS[0])
        self.assertTrue(entry["valid"])
        self.assertEqual(entry["guards_failed"], ["headroom_permille >= 40"])
        r["headroom_permille"] = 40
        r["comparator_above_random_permille"] = 99
        self.assertEqual(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["guards_failed"], ["comparator_above_random_permille >= 100"])
        r["comparator_above_random_permille"] = 100
        self.assertEqual(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["guards_failed"], [])

    def test_an_untripped_control_or_a_malformed_count_makes_the_seed_invalid(self):
        for name in aggregate.CONTROLS:
            r = passing_result(SEEDS[0])
            r["controls"][name]["required_rule_tripped"] = False
            self.assertFalse(aggregate.judge(r, 2, CRITERIA, SEEDS[0])["valid"], name)
        for key in aggregate.COUNT_KEYS:
            r = passing_result(SEEDS[0])
            r[key] = -1
            self.assertFalse(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["valid"], key)

    def test_the_exit_status_must_agree_with_the_rules(self):
        self.assertFalse(aggregate.judge(passing_result(SEEDS[0]), 1, CRITERIA, SEEDS[0])["valid"])
        r = passing_result(SEEDS[0])
        put(r, "authority_leaks", 1)
        self.assertFalse(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["valid"])


class PairedBoundTests(unittest.TestCase):
    def test_the_lower_and_upper_bounds_are_pinned(self):
        diffs = [60, 45, 52, 38, 71, 49, 55, 43, 66, 50, 58, 41, 47, 62, 53, 46, 59, 44, 51, 57]
        results = {s: passing_result(s, {"schema_free": 700, "control": 700 + d}) for s, d in zip(SEEDS, diffs)}
        item = paired(results)["judged"][0]
        mean = sum(diffs) / 20
        sd = (sum((d - mean) ** 2 for d in diffs) / 19) ** 0.5
        self.assertAlmostEqual(item["mean_permille"], mean, places=6)
        self.assertAlmostEqual(item["sd_permille"], sd, places=6)
        self.assertAlmostEqual(item["lower_bound_permille"], mean - 1.729133 * sd / 20 ** 0.5, places=4)
        self.assertAlmostEqual(item["upper_bound_permille"], mean + 1.729133 * sd / 20 ** 0.5, places=4)
        self.assertAlmostEqual(item["upper_bound_adjusted_permille"], mean + 2.539483 * sd / 20 ** 0.5, places=4)
        self.assertEqual(item["n"], 20)

    def test_the_sign_is_control_minus_comparator(self):
        results = {s: passing_result(s, {"schema_free": 700, "control": 690 + (i % 2)}) for i, s in enumerate(SEEDS)}
        self.assertLess(paired(results)["judged"][0]["mean_permille"], 0)

    def test_the_wrong_number_of_seeds_is_refused(self):
        results = clean_seeds()
        del results[SEEDS[0]]
        with self.assertRaises(agg.PairedError):
            paired(results)

    def test_the_attribution_is_three_valued(self):
        for shuffled, expected in ((650, aggregate.ATTRIBUTED), (795, aggregate.NOT_ATTRIBUTED)):
            results = {s: passing_result(s, {"control_shuffled": spread(shuffled, 3, i)}) for i, s in enumerate(SEEDS)}
            self.assertEqual(paired(results)["attribution"]["state"], expected)
        results = {s: passing_result(s, {"control_shuffled": spread(770, 60, i)}) for i, s in enumerate(SEEDS)}
        self.assertEqual(paired(results)["attribution"]["state"], aggregate.UNRESOLVED)


class VerdictTests(unittest.TestCase):
    def test_completed_when_all_five_lower_bounds_reach_the_margin(self):
        verdict = decide(clean_seeds())
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED, verdict["problems"])
        self.assertEqual(verdict["recommended_status"], "completed")
        self.assertEqual(verdict["attribution"], aggregate.ATTRIBUTED)
        self.assertFalse(verdict["control_role_not_shown"])

    def test_not_shown_when_a_bound_straddles_the_margin(self):
        seeds = {s: passing_result(s, {"control": spread(700 + 30, 30, i)}) for i, s in enumerate(SEEDS)}
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN, verdict["problems"])
        self.assertIsNone(verdict["recommended_status"])

    def test_refuted_when_the_upper_bound_is_below_the_margin(self):
        seeds = {s: passing_result(s, {"control": spread(700 + 5, 3, i)}) for i, s in enumerate(SEEDS)}
        verdict = decide(seeds)
        self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "refuted"), verdict["problems"])
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_every_comparator_must_be_beaten_and_a_describe_only_tie_is_reported_as_the_control_role_not_shown(self):
        seeds = {s: passing_result(s, {"describe_only": 795}) for s in SEEDS}
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["refuted_against"], ["describe_only"])
        self.assertTrue(verdict["control_role_not_shown"])

    def test_the_refutation_uses_the_bonferroni_quantile(self):
        # mean 17, sd about 18: the plain upper bound (t 1.729) is below the margin 25, the adjusted one (t 2.539) is not
        seeds = {s: passing_result(s, {"control": spread(717, 18, i), "schema_free_wide": 700, "heuristic": 700, "describe_only": 700, "history_router": 700}) for i, s in enumerate(SEEDS)}
        item = paired(seeds)["judged"][0]
        self.assertLess(item["upper_bound_permille"], MARGIN)
        self.assertGreaterEqual(item["upper_bound_adjusted_permille"], MARGIN)
        self.assertEqual(decide(seeds)["verdict"], aggregate.NOT_SHOWN)

    def test_structural_falsification(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[3]], "describe_only_influence", 1)
        verdict = decide(seeds, exits={SEEDS[3]: 1})
        self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "structural"))
        self.assertEqual(verdict["falsified_seeds"], [SEEDS[3]])

    def test_an_integrity_counter_is_invalid_even_next_to_a_structural_failure(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[0]], "authority_leaks", 1)
        put(seeds[SEEDS[1]], "oracle_violations", 1)
        verdict = decide(seeds, exits={SEEDS[0]: 1, SEEDS[1]: 2}, use_paired=False)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["integrity_seeds"], [SEEDS[1]])

    def test_missing_or_incomplete_evidence_is_invalid(self):
        seeds = clean_seeds()
        self.assertEqual(decide(seeds, mutations=None)["verdict"], aggregate.INVALID)
        self.assertEqual(decide(seeds, mutations={"killed": PLAN - 1, "total": PLAN})["verdict"], aggregate.INVALID)
        self.assertEqual(decide(seeds, mutations={"killed": PLAN - 1, "total": PLAN - 1})["verdict"], aggregate.INVALID)
        short = {s: r for s, r in seeds.items() if s != SEEDS[0]}
        self.assertEqual(aggregate.verdict_of(judged(short), PASSED, SEEDS, KILLED, None, PAIRED, PLAN)["verdict"], aggregate.INVALID)
        failed_static = aggregate.verdict_of(judged(seeds), [{"command": "x", "exit_code": 1}], SEEDS, KILLED, paired(seeds), PAIRED, PLAN)
        self.assertEqual(failed_static["verdict"], aggregate.FALSIFIED)

    def test_invalid_precedes_every_other_verdict(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[3]], "describe_only_influence", 1)
        tripped = {SEEDS[3]: 1}
        for mutations in (None, {"killed": PLAN - 1, "total": PLAN}, {"killed": PLAN - 1, "total": PLAN - 1}, {"killed": PLAN + 1, "total": PLAN + 1}):
            self.assertEqual(decide(seeds, exits=tripped, mutations=mutations)["verdict"], aggregate.INVALID, mutations)
        short = {s: r for s, r in seeds.items() if s != SEEDS[0]}
        self.assertEqual(aggregate.verdict_of(judged(short, tripped), PASSED, SEEDS, KILLED, None, PAIRED, PLAN)["verdict"], aggregate.INVALID)
        seeds[SEEDS[5]]["scale"] = "custom"
        self.assertEqual(decide(seeds, exits=tripped)["verdict"], aggregate.INVALID)
        refuting = {s: passing_result(s, {"control": spread(700 + 5, 3, i)}) for i, s in enumerate(SEEDS)}
        self.assertEqual(decide(refuting)["verdict"], aggregate.FALSIFIED)
        self.assertEqual(decide(refuting, mutations={"killed": PLAN - 1, "total": PLAN})["verdict"], aggregate.INVALID)
        self.assertEqual(decide(clean_seeds(), mutations=None)["verdict"], aggregate.INVALID)
        put(seeds[SEEDS[5]], "scale", "pinned")
        self.assertEqual(decide(seeds, exits=tripped)["verdict"], aggregate.FALSIFIED)

    def test_a_counter_above_zero_is_named_and_an_unmet_coverage_rule_is_structural(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[1]], "oracle_violations", 1)
        verdict = decide(seeds, exits={SEEDS[1]: 2}, use_paired=False)
        self.assertTrue(any("harness integrity counters above 0 on seeds" in p for p in verdict["problems"]), verdict["problems"])
        seeds = clean_seeds()
        put(seeds[SEEDS[2]], "arms_trained", 3)
        verdict = decide(seeds, exits={SEEDS[2]: 1})
        self.assertEqual((verdict["verdict"], verdict["kind"], verdict["vacuous_seeds"]), (aggregate.FALSIFIED, "structural", [SEEDS[2]]))

    def test_a_missing_static_run_and_duplicated_judgements_are_invalid(self):
        seeds = clean_seeds()
        outcome = paired(seeds)
        self.assertEqual(aggregate.verdict_of(judged(seeds), None, SEEDS, KILLED, outcome, PAIRED, PLAN)["verdict"], aggregate.INVALID)
        doubled = judged(seeds) + [aggregate.judge(seeds[SEEDS[0]], 0, CRITERIA, SEEDS[0])]
        verdict = aggregate.verdict_of(doubled, PASSED, SEEDS, KILLED, outcome, PAIRED, PLAN)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any("duplicated" in p for p in verdict["problems"]))

    def test_a_gain_that_is_not_attributed_to_the_routers_own_history_is_not_completed(self):
        # the shuffled-history control is as good as the control (or not resolved): the gain over every comparator is not shown to come from the history
        for label, shuffled, state in (("same", lambda i: 795, aggregate.NOT_ATTRIBUTED), ("noisy", lambda i: spread(770, 60, i), aggregate.UNRESOLVED)):
            seeds = {s: passing_result(s, {"control_shuffled": shuffled(i)}) for i, s in enumerate(SEEDS)}
            verdict = decide(seeds)
            self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN, label)
            self.assertEqual(verdict["attribution"], state)
            self.assertTrue(any("not attributed" in p for p in verdict["problems"]))
        self.assertEqual(decide(clean_seeds())["verdict"], aggregate.COMPLETED)

    def test_one_straddling_comparator_is_not_shown_even_if_the_others_are_beaten(self):
        seeds = {s: passing_result(s, {"history_router": spread(790, 40, i)}) for i, s in enumerate(SEEDS)}
        item = [i for i in paired(seeds)["judged"] if i["against"] == "history_router"][0]
        self.assertLess(item["lower_bound_permille"], MARGIN)
        self.assertGreaterEqual(item["upper_bound_adjusted_permille"], MARGIN)
        self.assertEqual(decide(seeds)["verdict"], aggregate.NOT_SHOWN)

    def test_the_controls_are_judged_on_their_bounds_not_their_means(self):
        wide = lambda centre, i: spread(centre, 90, i)  # noqa: E731  mean `centre`, sd about 93 per mille
        # gross: schema-free minus the uniform router has a mean above the margin and a lower bound below it
        gross = {s: passing_result(s, {"random_router": 700 - wide(30, i)}) for i, s in enumerate(SEEDS)}
        for result in gross.values():
            put(result, "comparator_above_random_permille", 300)  # the per-seed guard speaks elsewhere; here only the control does
        self.assertGreaterEqual(paired(gross)["sensitivity"]["gross"]["mean_permille"], MARGIN)
        self.assertEqual(decide(gross)["verdict"], aggregate.INVALID)
        # graded: the degraded router is worse by a mean above the minimum and a lower bound below it
        graded = {s: passing_result(s, {"schema_free_epsilon": 700 - wide(25, i)}) for i, s in enumerate(SEEDS)}
        self.assertGreater(paired(graded)["sensitivity"]["graded"]["mean_permille"], PAIRED["sensitivity_graded"]["min_lower_permille"])
        self.assertEqual(decide(graded)["verdict"], aggregate.INVALID)
        # null world cost: a mean cost inside the band whose upper bound is outside it
        cost = {s: passing_result(s, {"control_null": wide(770, i), "schema_free_null": 800}) for i, s in enumerate(SEEDS)}
        out = paired(cost)["null_world"]
        self.assertLess(out["backward"]["mean_permille"], PAIRED["null_world"]["cost_band_permille"])
        self.assertGreaterEqual(out["backward"]["upper_bound_permille"], PAIRED["null_world"]["cost_band_permille"])
        self.assertEqual(decide(cost)["verdict"], aggregate.INVALID)

    def test_the_history_router_is_a_judged_comparator(self):
        seeds = {s: passing_result(s, {"history_router": 790}) for s in SEEDS}
        verdict = decide(seeds)
        self.assertEqual((verdict["verdict"], verdict["kind"], verdict["refuted_against"]), (aggregate.FALSIFIED, "refuted", ["history_router"]))

    def test_the_blind_reference_is_reported_never_judged(self):
        seeds = {s: passing_result(s, {"even_split": 990}) for s in SEEDS}
        outcome = paired(seeds)
        self.assertEqual([item["against"] for item in outcome["blind_reference"]], ["control", *PAIRED["against"]])
        self.assertAlmostEqual(outcome["blind_reference"][0]["mean_permille"], 190.0, places=6)
        self.assertEqual(decide(seeds)["verdict"], aggregate.COMPLETED)

    def test_no_headroom_is_invalid(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[2]], "headroom_permille", 10)
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["uninformative_seeds"], [SEEDS[2]])

    def test_an_untrained_schema_is_tolerated_up_to_the_allowance(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[0]], "schema_trained", 0)
        self.assertEqual(decide(seeds)["verdict"], aggregate.COMPLETED)
        put(seeds[SEEDS[1]], "schema_trained", 0)
        self.assertEqual(decide(seeds)["verdict"], aggregate.INVALID)

    def test_unconverged_judged_arms_are_tolerated_up_to_the_allowance(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[0]], "unconverged_judged_arms", 3)
        self.assertEqual(decide(seeds)["verdict"], aggregate.COMPLETED)
        put(seeds[SEEDS[1]], "unconverged_judged_arms", 1)
        self.assertEqual(decide(seeds)["verdict"], aggregate.INVALID)

    def test_each_control_must_resolve(self):
        # gross: the schema-free router is no better than the uniform one (the guard scalar is held at its passing value so that only the control speaks)
        weak = {s: passing_result(s, {"random_router": 690, "schema_free": 700}) for s in SEEDS}
        for result in weak.values():
            put(result, "comparator_above_random_permille", 300)
        self.assertEqual(decide(weak)["verdict"], aggregate.INVALID)
        # graded: the degraded router is as good as the intact one
        self.assertEqual(decide({s: passing_result(s, {"schema_free": 700, "schema_free_epsilon": 695}) for s in SEEDS})["verdict"], aggregate.INVALID)
        # null world: an advantage of the margin's size where none can exist, in either direction
        self.assertEqual(decide({s: passing_result(s, {"control_null": 850, "schema_free_null": 800}) for s in SEEDS})["verdict"], aggregate.INVALID)
        # a tie is required, with a cost band: the schema may cost less than the band where demand is stationary, not more
        self.assertEqual(decide({s: passing_result(s, {"control_null": 760, "schema_free_null": 800}) for s in SEEDS})["verdict"], aggregate.COMPLETED)
        self.assertEqual(decide({s: passing_result(s, {"control_null": 740, "schema_free_null": 800}) for s in SEEDS})["verdict"], aggregate.INVALID)

    def test_the_controls_precede_a_refutation(self):
        seeds = {s: passing_result(s, {"control": 702, "control_null": 900, "schema_free_null": 800}) for s in SEEDS}
        self.assertEqual(decide(seeds)["verdict"], aggregate.INVALID)


class WiringTests(unittest.TestCase):
    def test_a_seed_in_another_environment_is_invalidated(self):
        seeds = clean_seeds()
        entries = judged(seeds)
        environments = {s: dict(ENVIRONMENT) for s in SEEDS}
        environments[SEEDS[5]]["torch"] = "other"
        agg.invalidate_environment_mismatches(entries, environments)
        self.assertEqual([e["seed"] for e in entries if not e["valid"]], [SEEDS[5]])
        self.assertEqual(aggregate.verdict_of(entries, PASSED, SEEDS, KILLED, paired(seeds), PAIRED, PLAN)["verdict"], aggregate.INVALID)

    def test_the_report_names_the_verdict_and_every_comparison(self):
        seeds = clean_seeds()
        text = aggregate.report(decide(seeds), judged(seeds), PASSED, KILLED, paired(seeds))
        for needle in ("COMPLETED", "vs schema_free", "vs schema_free_wide", "vs heuristic", "vs describe_only", "vs history_router", "gross sensitivity", "graded sensitivity",
                       "null world", "attribution", "hand-set regime-blind reference", "mutation check"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
