"""The C004 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge`, `paired_comparisons` and `verdict_of`, which read
no file. These tests build a result that holds every rule of the criteria and break one rule at a time, so a verdict that stopped
depending on it fails here. The paired bounds are pinned to hand-computed values, so a flipped sign, a wrong quantile or a wrong divisor
fails. Outcomes: structural FALSIFIED, INVALID (integrity, saturated, untrained, unconverged, unresolved control, null world that shows an
effect, missing evidence), refuted FALSIFIED (the Bonferroni upper bound below the margin: the advantage is below the margin), NOT_SHOWN
(straddling) and COMPLETED (both lower bounds at the margin). That the harness's rules equal criteria.toml and that every hard counter is
planted by a mutation is held by research/cseries/tests/test_c004.py."""

import importlib.util
import sys
import tomllib
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "consciousness" / "C004-recurrent-integration"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/cseries/lib"))

spec = importlib.util.spec_from_file_location("c004_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)
import agg  # noqa: E402

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
PAIRED = CRITERIA["paired"]
SEEDS = list(range(5001, 5001 + PAIRED["seeds_required"]))
PASSED: list[dict] = []  # this experiment names no static evidence
PLAN = len(MUTATIONS["mutation"])
KILLED = {"killed": PLAN, "total": PLAN}
MARGIN = PAIRED["margin_permille"]
TOTAL = 1000  # evaluation instances per synthetic arm, so counts are per mille
ARMS = ("recurrent", "feedforward", "feedforward_inlength", "reset_state", "recurrent_perm", "feedforward_perm", "recurrent_null", "feedforward_null", "recurrent_corrupt")
BASE = {"recurrent": 900, "feedforward": 300, "feedforward_inlength": 880, "reset_state": 166, "recurrent_perm": 166, "feedforward_perm": 166, "recurrent_null": 990,
        "feedforward_null": 985, "recurrent_corrupt": 780}
ENVIRONMENT = {k: "x" for k in agg.ENVIRONMENT_KEYS}


def put(result: dict, path: str, value) -> None:
    at = result
    keys = path.split(".")
    for key in keys[:-1]:
        at = at.setdefault(key, {})
    at[keys[-1]] = value


def arm_record(ex12: int) -> dict:
    return {"ex12_correct": ex12, "ex12_total": TOTAL, "ex18_correct": ex12, "ex18_total": TOTAL, "ex24_correct": ex12, "ex24_total": TOTAL,
            "id_correct": 950, "id_total": TOTAL}


def passing_result(seed: int, ex12: dict | None = None) -> dict:
    """A harness result in which every rule and guard of the criteria holds."""
    given = ex12 or {}
    ex12 = {**BASE, **given}
    if "recurrent" in given:  # the graded control follows the arm it corrupts unless a test sets it
        ex12.setdefault("recurrent_corrupt", ex12["recurrent"] - 120)
        ex12["recurrent_corrupt"] = given.get("recurrent_corrupt", ex12["recurrent"] - 120)
    result = {
        "benchmark": "c004", "experiment": "C004", "seed": seed, "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "controls": {name: {"required_rule_tripped": True} for name in aggregate.CONTROLS},
        "harness_gaps": [], "environment": dict(ENVIRONMENT), "recurrent_trained": 1, "feedforward_trained": 1,
        "arms": {arm: arm_record(ex12[arm]) for arm in ARMS},
    }
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["integrity"]:
        left, _, right = agg.parse_rule(rule)
        put(result, left, int(right))
    result["comparator_primary_permille"] = ex12["feedforward"]
    result["unconverged_judged_arms"] = 0
    return result


def judged(results: dict, exits: dict | None = None) -> list[dict]:
    exits = exits or {}
    return [aggregate.judge(results[seed], exits.get(seed, 0), CRITERIA, seed) for seed in results]


def clean_seeds(ex12: dict | None = None) -> dict:
    return {seed: passing_result(seed, ex12) for seed in SEEDS}


def paired(results: dict) -> dict:
    return aggregate.paired_comparisons(results, PAIRED)


def decide(results, exits=None, mutations=KILLED, use_paired=True, plan=PLAN):
    outcome = paired(results) if use_paired else None
    return aggregate.verdict_of(judged(results, exits), PASSED, SEEDS, mutations, outcome, PAIRED, plan)


class CriteriaFileTests(unittest.TestCase):
    def test_every_rule_reads_and_the_sets_are_not_empty(self):
        for key in ("coverage", "guards", "falsifiers", "integrity"):
            rules = CRITERIA[key]
            self.assertTrue(rules)
            self.assertEqual(len(rules), len(set(rules)), key)
            for rule in rules:
                agg.parse_rule(rule)
        self.assertEqual(len(CRITERIA["falsifiers"]), 3)
        self.assertEqual(len(CRITERIA["integrity"]), 13)
        self.assertEqual(len(CRITERIA["guards"]), 1)
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
            entry = aggregate.judge(r, 1, CRITERIA, SEEDS[0])
            self.assertEqual(entry["coverage_failed"], [rule])

    def test_a_broken_integrity_counter_makes_the_seed_invalid(self):
        for rule in CRITERIA["integrity"]:
            r = passing_result(SEEDS[0])
            put(r, rule.split(" ")[0], 1)
            entry = aggregate.judge(r, 2, CRITERIA, SEEDS[0])
            self.assertFalse(entry["valid"], rule)
            self.assertEqual(entry["integrity_failed"], [rule])

    def test_a_saturated_comparator_fails_the_guard_without_invalidating_the_seed(self):
        r = passing_result(SEEDS[0], {"feedforward": 981})
        entry = aggregate.judge(r, 0, CRITERIA, SEEDS[0])
        self.assertTrue(entry["valid"])
        self.assertEqual(entry["guards_failed"], ["comparator_primary_permille <= 980"])
        self.assertEqual(aggregate.judge(passing_result(SEEDS[0], {"feedforward": 980}), 0, CRITERIA, SEEDS[0])["guards_failed"], [])
        self.assertEqual(aggregate.judge(passing_result(SEEDS[0], {"feedforward": 900}), 0, CRITERIA, SEEDS[0])["guards_failed"], [])  # a competent comparator is not saturated

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
        put(r, "reset_carry_leaks", 1)
        self.assertFalse(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["valid"])


class PairedBoundTests(unittest.TestCase):
    """The bounds are pinned to values computed by hand from the per-seed differences."""

    def results_with_differences(self, recurrent_minus_ff):
        # reset_state is 166 everywhere, feedforward is 300; recurrent = 300 + difference over the feed-forward arm
        results = {}
        for seed, d in zip(SEEDS, recurrent_minus_ff):
            results[seed] = passing_result(seed, {"recurrent": 300 + d})
        return results

    def test_the_lower_and_upper_bounds_are_pinned(self):
        diffs = [150, 250, 200, 220, 180, 210, 190, 230, 170, 240, 160, 205, 195, 215]
        out = paired(self.results_with_differences(diffs))
        item = out["judged"][0]
        mean = sum(diffs) / 14
        sd = (sum((d - mean) ** 2 for d in diffs) / 13) ** 0.5
        half_support = 1.770933 * sd / 14 ** 0.5
        half_refute = 2.160369 * sd / 14 ** 0.5
        self.assertAlmostEqual(item["mean_permille"], mean, places=6)
        self.assertAlmostEqual(item["sd_permille"], sd, places=6)
        self.assertAlmostEqual(item["lower_bound_permille"], mean - half_support, places=4)
        self.assertAlmostEqual(item["upper_bound_permille"], mean + half_support, places=4)
        self.assertAlmostEqual(item["upper_bound_adjusted_permille"], mean + half_refute, places=4)
        self.assertEqual(item["n"], 14)

    def test_the_sign_is_recurrent_minus_comparator(self):
        out = paired(self.results_with_differences([-50] * 7 + [-60] * 7))
        self.assertLess(out["judged"][0]["mean_permille"], 0)
        self.assertFalse(out["judged"][0]["supports"])

    def test_the_wrong_number_of_seeds_is_refused(self):
        results = clean_seeds()
        del results[SEEDS[0]]
        with self.assertRaises(agg.PairedError):
            paired(results)


class VerdictTests(unittest.TestCase):
    def test_completed_when_both_lower_bounds_reach_the_margin(self):
        verdict = decide(clean_seeds())
        self.assertEqual(verdict["verdict"], agg.COMPLETED, verdict["problems"])
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_not_shown_when_a_bound_straddles_the_margin(self):
        # the feed-forward comparator is 300; the recurrent arm is 100 above it on average with a spread, so the interval contains the margin
        seeds = {s: passing_result(s, {"recurrent": 300 + 100 + (60 if i % 2 else -60)}) for i, s in enumerate(SEEDS)}
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], agg.NOT_SHOWN, verdict["problems"])
        self.assertIsNone(verdict["recommended_status"])

    def test_refuted_when_the_upper_bound_is_below_the_margin(self):
        seeds = {s: passing_result(s, {"recurrent": 300 + 10 + (5 if i % 2 else -5)}) for i, s in enumerate(SEEDS)}
        verdict = decide(seeds)
        self.assertEqual((verdict["verdict"], verdict["kind"]), (agg.FALSIFIED, "refuted"), verdict["problems"])
        self.assertEqual(verdict["refuted_against"], ["feedforward"])
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_the_refutation_uses_the_bonferroni_quantile(self):
        # choose a spread for which mean + t_support * se < margin <= mean + t_refute * se: refuted at the plain quantile, not at the adjusted one
        diffs = [60 + (d if i % 2 else -d) for i, d in enumerate([75] * 14)]
        seeds = {s: passing_result(s, {"recurrent": 300 + diffs[i]}) for i, s in enumerate(SEEDS)}
        item = paired(seeds)["judged"][0]
        self.assertLess(item["upper_bound_permille"], MARGIN)
        self.assertGreaterEqual(item["upper_bound_adjusted_permille"], MARGIN)
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], agg.NOT_SHOWN)

    def test_both_comparators_must_be_beaten(self):
        # the reset-state ablation is strong on every seed: the recurrent arm is far above the feed-forward arm but not above the ablation
        seeds = {s: passing_result(s, {"reset_state": 880, "recurrent": 900}) for s in SEEDS}
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], agg.FALSIFIED)
        self.assertEqual(verdict["refuted_against"], ["reset_state"])

    def test_structural_falsification(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[3]], "reset_carry_leaks", 1)
        verdict = decide(seeds, exits={SEEDS[3]: 1})
        self.assertEqual((verdict["verdict"], verdict["kind"]), (agg.FALSIFIED, "structural"))
        self.assertEqual(verdict["falsified_seeds"], [SEEDS[3]])

    def test_an_integrity_counter_is_invalid_even_next_to_a_structural_failure(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[0]], "reset_carry_leaks", 1)
        put(seeds[SEEDS[1]], "label_mismatches", 1)
        verdict = decide(seeds, exits={SEEDS[0]: 1, SEEDS[1]: 2}, use_paired=False)
        self.assertEqual(verdict["verdict"], agg.INVALID)
        self.assertEqual(verdict["integrity_seeds"], [SEEDS[1]])

    def test_missing_or_incomplete_evidence_is_invalid(self):
        seeds = clean_seeds()
        self.assertEqual(decide(seeds, mutations=None)["verdict"], agg.INVALID)
        self.assertEqual(decide(seeds, mutations={"killed": PLAN - 1, "total": PLAN})["verdict"], agg.INVALID)
        self.assertEqual(decide(seeds, mutations={"killed": PLAN - 1, "total": PLAN - 1})["verdict"], agg.INVALID)  # not the plan's size
        short = {s: r for s, r in seeds.items() if s != SEEDS[0]}
        verdict = aggregate.verdict_of(judged(short), PASSED, SEEDS, KILLED, None, PAIRED, PLAN)
        self.assertEqual(verdict["verdict"], agg.INVALID)
        failed_static = aggregate.verdict_of(judged(seeds), [{"command": "x", "exit_code": 1}], SEEDS, KILLED, paired(seeds), PAIRED, PLAN)
        self.assertEqual(failed_static["verdict"], agg.FALSIFIED)

    def test_a_saturated_comparator_is_invalid(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[2]], "comparator_primary_permille", 985)
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], agg.INVALID)
        self.assertEqual(verdict["uninformative_seeds"], [SEEDS[2]])

    def test_a_competent_comparator_can_refute(self):
        # the stack reaches 900 at the primary length (a competent baseline): the recurrent arm's advantage (975 - 900 = 75) is below the margin of 100
        seeds = {s: passing_result(s, {"recurrent": 975 + (4 if i % 2 else -4), "feedforward": 900 + (6 if i % 2 else -6), "feedforward_inlength": 905}) for i, s in enumerate(SEEDS)}
        verdict = decide(seeds)
        self.assertEqual((verdict["verdict"], verdict["kind"]), (agg.FALSIFIED, "refuted"), verdict["problems"])
        self.assertEqual(verdict["refuted_against"], ["feedforward"])
        self.assertEqual(verdict["uninformative_seeds"], [])
        # at 940 the advantage is 35: still a refutation, never INVALID; at 985 the comparator is saturated: INVALID
        seeds = {s: passing_result(s, {"recurrent": 975, "feedforward": 940, "feedforward_inlength": 942}) for s in SEEDS}
        self.assertEqual(decide(seeds)["verdict"], agg.FALSIFIED)
        seeds = {s: passing_result(s, {"recurrent": 995, "feedforward": 985, "feedforward_inlength": 990}) for s in SEEDS}
        self.assertEqual(decide(seeds)["verdict"], agg.INVALID)

    def test_the_extrapolation_control_is_read_only_for_a_supported_contrast(self):
        # a competent stack (900) that gains nothing from seeing the length (905) fails the control's statistic, yet the verdict is the refutation
        competent = {s: passing_result(s, {"recurrent": 975, "feedforward": 900, "feedforward_inlength": 905}) for s in SEEDS}
        self.assertFalse(paired(competent)["extrapolation_removed"]["collapses"])
        verdict = decide(competent)
        self.assertEqual((verdict["verdict"], verdict["kind"]), (agg.FALSIFIED, "refuted"))
        self.assertEqual(verdict["problems"], [])
        # a contrast that straddles the margin is NOT_SHOWN whatever the control says
        straddle = {s: passing_result(s, {"recurrent": 940 + (50 if i % 2 else -50), "feedforward": 830, "feedforward_inlength": 832}) for i, s in enumerate(SEEDS)}
        self.assertFalse(paired(straddle)["supported"])
        self.assertEqual(decide(straddle)["verdict"], agg.NOT_SHOWN)
        # refuted with a passing control stays FALSIFIED (refuted)
        passing = {s: passing_result(s, {"recurrent": 975, "feedforward": 900, "feedforward_inlength": 940}) for s in SEEDS}
        self.assertTrue(paired(passing)["extrapolation_removed"]["collapses"])
        self.assertEqual(decide(passing)["verdict"], agg.FALSIFIED)
        # a moderately competent stack: contrast 200 supported, the in-length stack at 850 recovers 50 of 200 (statistic -50): INVALID
        moderate = {s: passing_result(s, {"recurrent": 1000, "feedforward": 800, "feedforward_inlength": 850}) for s in SEEDS}
        self.assertTrue(paired(moderate)["supported"])
        self.assertEqual(decide(moderate)["verdict"], agg.INVALID)
        # the report does not claim a capacity finding where the control was not read
        self.assertNotIn("capacity, not extrapolation", aggregate.report(decide(competent), judged(competent), PASSED, KILLED, paired(competent)))
        # a supported contrast with a failing control is INVALID
        supported = {s: passing_result(s, {"feedforward_inlength": 400}) for s in SEEDS}
        self.assertTrue(paired(supported)["supported"])
        self.assertEqual(decide(supported)["verdict"], agg.INVALID)

    def test_the_stack_trained_on_the_primary_length_must_recover_more_than_half_of_the_contrast(self):
        # the stack trained on the long length stays far below the recurrent arm (300 -> 500 of a 600 advantage): it cannot hold the answer, the main contrast would be one of capacity
        seeds = {s: passing_result(s, {"feedforward_inlength": 500}) for s in SEEDS}
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], agg.INVALID)
        self.assertTrue(any("does not recover" in p for p in verdict["problems"]))
        # the control reads the LOWER bound: a mean above the share whose lower bound is not does not resolve
        wide = {s: passing_result(s, {"feedforward_inlength": 630 + (90 if i % 2 else -90)}) for i, s in enumerate(SEEDS)}
        out = paired(wide)["extrapolation_removed"]
        self.assertGreater(out["mean_permille"], 0)
        self.assertLess(out["lower_bound_permille"], 0)
        self.assertEqual(decide(wide)["verdict"], agg.INVALID)
        # exactly the share is the boundary; the share itself is the criteria's, not a free parameter of the test
        self.assertEqual(PAIRED["extrapolation_removed"]["share_permille"], 500)
        half = {s: passing_result(s, {"feedforward_inlength": 600 + 1}) for s in SEEDS}
        self.assertEqual(decide(half)["verdict"], agg.COMPLETED)
        below = {s: passing_result(s, {"feedforward_inlength": 600 - 1}) for s in SEEDS}
        self.assertEqual(decide(below)["verdict"], agg.INVALID)
        self.assertAlmostEqual(paired(clean_seeds())["extrapolation_removed"]["recovered_mean_permille"], 580.0, places=6)
        self.assertAlmostEqual(paired(clean_seeds())["extrapolation_removed"]["advantage_mean_permille"], 600.0, places=6)
        self.assertEqual(decide(clean_seeds())["verdict"], agg.COMPLETED)

    def test_invalid_precedes_every_other_verdict(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[3]], "reset_carry_leaks", 1)
        tripped = {SEEDS[3]: 1}
        for mutations in (None, {"killed": PLAN - 1, "total": PLAN}, {"killed": PLAN - 1, "total": PLAN - 1}, {"killed": PLAN + 1, "total": PLAN + 1}):
            self.assertEqual(decide(seeds, exits=tripped, mutations=mutations)["verdict"], agg.INVALID, mutations)
        short = {s: r for s, r in seeds.items() if s != SEEDS[0]}
        self.assertEqual(aggregate.verdict_of(judged(short, tripped), PASSED, SEEDS, KILLED, None, PAIRED, PLAN)["verdict"], agg.INVALID)
        seeds[SEEDS[5]]["scale"] = "custom"
        self.assertEqual(decide(seeds, exits=tripped)["verdict"], agg.INVALID)
        refuting = {s: passing_result(s, {"recurrent": 975, "feedforward": 940, "feedforward_inlength": 942}) for s in SEEDS}
        self.assertEqual(decide(refuting)["verdict"], agg.FALSIFIED)
        self.assertEqual(decide(refuting, mutations={"killed": PLAN - 1, "total": PLAN})["verdict"], agg.INVALID)
        self.assertEqual(decide(clean_seeds(), mutations=None)["verdict"], agg.INVALID)
        put(seeds[SEEDS[5]], "scale", "pinned")
        self.assertEqual(decide(seeds, exits=tripped)["verdict"], agg.FALSIFIED)

    def test_a_counter_above_zero_is_named_and_an_unmet_coverage_rule_is_structural(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[1]], "label_mismatches", 1)
        verdict = decide(seeds, exits={SEEDS[1]: 2}, use_paired=False)
        self.assertTrue(any("harness integrity counters above 0 on seeds" in p for p in verdict["problems"]), verdict["problems"])
        seeds = clean_seeds()
        put(seeds[SEEDS[2]], "arms_trained", 3)
        verdict = decide(seeds, exits={SEEDS[2]: 1})
        self.assertEqual((verdict["verdict"], verdict["kind"], verdict["vacuous_seeds"]), (agg.FALSIFIED, "structural", [SEEDS[2]]))

    def test_a_missing_static_run_and_duplicated_judgements_are_invalid(self):
        seeds = clean_seeds()
        outcome = paired(seeds)
        self.assertEqual(aggregate.verdict_of(judged(seeds), None, SEEDS, KILLED, outcome, PAIRED, PLAN)["verdict"], agg.INVALID)
        doubled = judged(seeds) + [aggregate.judge(seeds[SEEDS[0]], 0, CRITERIA, SEEDS[0])]
        verdict = aggregate.verdict_of(doubled, PASSED, SEEDS, KILLED, outcome, PAIRED, PLAN)
        self.assertEqual(verdict["verdict"], agg.INVALID)
        self.assertTrue(any("duplicated" in p for p in verdict["problems"]))

    def test_the_controls_are_judged_on_their_bounds_not_their_means(self):
        base = {"recurrent": 900}
        wide = lambda centre, i: centre + (90 if i % 2 else -90)  # noqa: E731  mean `centre`, sd about 93 per mille
        gross = {s: passing_result(s, {**base, "recurrent_perm": 900 - wide(120, i), "feedforward_perm": 900 - wide(120, i)}) for i, s in enumerate(SEEDS)}
        self.assertGreaterEqual(paired(gross)["sensitivity"]["gross"]["mean_permille"], MARGIN)
        self.assertEqual(decide(gross)["verdict"], agg.INVALID)
        graded = {s: passing_result(s, {**base, "recurrent_corrupt": 900 - wide(60, i)}) for i, s in enumerate(SEEDS)}
        self.assertGreater(paired(graded)["sensitivity"]["graded"]["mean_permille"], PAIRED["sensitivity_graded"]["min_lower_permille"])
        self.assertEqual(decide(graded)["verdict"], agg.INVALID)

    def test_untrained_arms_are_tolerated_up_to_the_allowance(self):
        for key, field in (("untrained_recurrent", "recurrent_trained"), ("untrained_comparator", "feedforward_trained")):
            seeds = clean_seeds()
            put(seeds[SEEDS[0]], field, 0)
            self.assertEqual(decide(seeds)["verdict"], agg.COMPLETED, key)
            put(seeds[SEEDS[1]], field, 0)
            verdict = decide(seeds)
            self.assertEqual(verdict["verdict"], agg.INVALID, key)

    def test_unconverged_judged_arms_are_tolerated_up_to_the_allowance(self):
        seeds = clean_seeds()
        put(seeds[SEEDS[0]], "unconverged_judged_arms", 2)
        self.assertEqual(decide(seeds)["verdict"], agg.COMPLETED)  # one seed with any number of arms at the cap
        put(seeds[SEEDS[1]], "unconverged_judged_arms", 1)
        self.assertEqual(decide(seeds)["verdict"], agg.INVALID)

    def test_each_control_must_resolve(self):
        base = {"recurrent": 900}
        # gross: the permuted-label net is nearly as good as the recurrent one (the permutation did not remove the effect)
        self.assertEqual(decide({s: passing_result(s, {**base, "recurrent_perm": 850, "feedforward_perm": 850}) for s in SEEDS})["verdict"], agg.INVALID)
        # graded: the corrupted-input net is as good as the clean one (the pipeline does not see a deficit)
        self.assertEqual(decide({s: passing_result(s, {**base, "recurrent_corrupt": 890}) for s in SEEDS})["verdict"], agg.INVALID)
        # null worlds: an advantage of the margin's size where none can exist, in either direction
        self.assertEqual(decide({s: passing_result(s, {"recurrent_perm": 166, "feedforward_perm": 166 - 0, "recurrent_null": 700, "feedforward_null": 990}) for s in SEEDS})["verdict"], agg.INVALID)
        self.assertEqual(decide({s: passing_result(s, {"recurrent_null": 990, "feedforward_null": 700}) for s in SEEDS})["verdict"], agg.INVALID)
        self.assertEqual(decide({s: passing_result(s, {"recurrent_perm": 166 + 150}) for s in SEEDS})["verdict"], agg.INVALID)

    def test_the_controls_precede_a_refutation(self):
        seeds = {s: passing_result(s, {"recurrent": 310, "recurrent_perm": 300}) for s in SEEDS}
        self.assertEqual(decide(seeds)["verdict"], agg.INVALID)

    def test_exit_status_codes_of_the_verdicts(self):
        self.assertEqual({agg.COMPLETED: 0, agg.FALSIFIED: 1, agg.NOT_SHOWN: 3}.get(agg.INVALID, 2), 2)


class WiringTests(unittest.TestCase):
    def test_a_seed_in_another_environment_is_invalidated(self):
        seeds = clean_seeds()
        entries = judged(seeds)
        environments = {s: dict(ENVIRONMENT) for s in SEEDS}
        environments[SEEDS[5]]["torch"] = "other"
        agg.invalidate_environment_mismatches(entries, environments)
        self.assertEqual([e["seed"] for e in entries if not e["valid"]], [SEEDS[5]])
        outcome = aggregate.verdict_of(entries, PASSED, SEEDS, KILLED, paired(seeds), PAIRED, PLAN)
        self.assertEqual(outcome["verdict"], agg.INVALID)

    def test_two_finished_records_of_a_seed_are_refused(self):
        import json
        import tempfile

        import experiment_records
        from aggregate_run import newest_per_seed

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for stamp in ("a", "b"):
                path = Path(tmp) / f"run-{stamp}-seed-5001.json"
                path.write_text(json.dumps({"seed": 5001, "exit_code": 0, "command": ["x"], "started_at": "t"}), encoding="utf-8")
                paths.append(path)
            original = experiment_records.finished_run
            experiment_records.finished_run = lambda record: True
            try:
                with self.assertRaises(SystemExit) as caught:
                    newest_per_seed(paths, "C004", experiment_records)
            finally:
                experiment_records.finished_run = original
            self.assertEqual(caught.exception.code, 2)

    def test_the_report_names_the_verdict_and_every_comparison(self):
        seeds = clean_seeds()
        verdict = decide(seeds)
        text = aggregate.report(verdict, judged(seeds), PASSED, KILLED, paired(seeds))
        for needle in ("COMPLETED", "vs feedforward", "vs reset_state", "gross sensitivity", "graded sensitivity", "null world null_permuted", "null world null_window", "mutation check"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
