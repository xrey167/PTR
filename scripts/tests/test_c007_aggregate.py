"""The C007 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge`, `paired_comparisons` and `verdict_of`, which read no file.
These tests build a result that holds every rule of the criteria and break one rule at a time, so a verdict that stopped depending on it fails here.
The paired bounds are pinned to hand-computed values. The metric is the Brier score on the shifted split (lower is better; a paired difference is
comparator minus self-model). Outcomes: structural FALSIFIED, INVALID (integrity, no signal, an unconverged head, an unresolved control, a removal
control that does not remove, a null world that shows an effect, missing evidence), refuted FALSIFIED (the Bonferroni upper bound below the margin),
NOT_SHOWN (straddling) and COMPLETED (both lower bounds at the margin). That the harness's rules equal criteria.toml and that every hard counter is
planted by a mutation is held by research/cseries/tests/test_c007.py."""

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "consciousness" / "C007-metacognitive-self-model-cpu"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research/cseries/lib"))

spec = importlib.util.spec_from_file_location("c007_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)
import agg  # noqa: E402

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
PAIRED = CRITERIA["paired"]
SEEDS = list(range(6001, 6001 + PAIRED["seeds_required"]))
PASSED: list[dict] = []
PLAN = len(MUTATIONS["mutation"])
KILLED = {"killed": PLAN, "total": PLAN}
MARGIN = PAIRED["margin_permille"]
TOTAL = 1_000_000  # the harness reports Brier micro-sums over n examples with total n * 1e6; one example per synthetic arm, so a Brier of 0.2 is 200000
ARMS = ("self_model", "probe_matched", "own_confidence", "constant", "self_model_perm", "self_model_shuffled", "self_model_noisy", "self_model_null",
        "probe_matched_null", "probe_io", "own_confidence_raw", "probe_matched_shuffled")
# Brier scores in micro-units of one example: lower is better
BASE = {"self_model": 200_000, "probe_matched": 206_000, "own_confidence": 212_000, "constant": 240_000, "self_model_perm": 241_000,
        "self_model_shuffled": 250_000, "self_model_noisy": 209_000, "self_model_null": 105_000, "probe_matched_null": 105_000, "probe_io": 220_000,
        "own_confidence_raw": 225_000, "probe_matched_shuffled": 250_000}
ENVIRONMENT = {k: "x" for k in agg.ENVIRONMENT_KEYS}


def put(result: dict, path: str, value) -> None:
    at = result
    keys = path.split(".")
    for key in keys[:-1]:
        at = at.setdefault(key, {})
    at[keys[-1]] = value


def passing_result(seed: int, brier: dict | None = None) -> dict:
    """A harness result in which every rule and guard of the criteria holds."""
    given = brier or {}
    brier = {**BASE, **given}
    if "self_model" in given:  # the noised self-model follows the self-model it degrades unless a test sets it
        brier["self_model_noisy"] = given.get("self_model_noisy", brier["self_model"] + 9_000)
    result = {
        "benchmark": "c007", "experiment": "C007", "seed": seed, "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "controls": {name: {"required_rule_tripped": True} for name in aggregate.CONTROLS},
        "harness_gaps": [], "environment": dict(ENVIRONMENT),
        "arms": {arm: {"brier_correct": brier[arm], "brier_total": TOTAL} for arm in ARMS},
    }
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["integrity"]:
        left, _, right = agg.parse_rule(rule)
        put(result, left, int(right))
    result["signal_permille"] = 30
    result["head_gap_id_permille"] = 2
    result["shifted_error_permille"] = 330
    result["unconverged_judged_arms"] = 0
    return result


def judged(results: dict, exits: dict | None = None) -> list[dict]:
    exits = exits or {}
    return [aggregate.judge(results[seed], exits.get(seed, 0), CRITERIA, seed) for seed in results]


def clean_seeds(brier: dict | None = None) -> dict:
    return {seed: passing_result(seed, brier) for seed in SEEDS}


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
        self.assertEqual(len(CRITERIA["integrity"]), 15)
        self.assertEqual(len(CRITERIA["guards"]), 4)

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
        for scalar, value, rule in (("signal_permille", 3, "signal_permille >= 4"), ("shifted_error_permille", 99, "shifted_error_permille >= 100"),
                                    ("shifted_error_permille", 701, "shifted_error_permille <= 700"), ("head_gap_id_permille", 5, "head_gap_id_permille <= 4")):
            r = passing_result(SEEDS[0])
            r[scalar] = value
            entry = aggregate.judge(r, 0, CRITERIA, SEEDS[0])
            self.assertTrue(entry["valid"])
            self.assertEqual(entry["guards_failed"], [rule])
        r = passing_result(SEEDS[0])
        r["signal_permille"], r["shifted_error_permille"], r["head_gap_id_permille"] = 4, 700, 4
        self.assertEqual(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["guards_failed"], [])

    def test_an_untripped_control_or_a_malformed_count_makes_the_seed_invalid(self):
        for name in aggregate.CONTROLS:
            r = passing_result(SEEDS[0])
            r["controls"][name]["required_rule_tripped"] = False
            self.assertFalse(aggregate.judge(r, 2, CRITERIA, SEEDS[0])["valid"], name)
        r = passing_result(SEEDS[0])
        r["unconverged_judged_arms"] = -1
        self.assertFalse(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["valid"])

    def test_the_exit_status_must_agree_with_the_rules(self):
        self.assertFalse(aggregate.judge(passing_result(SEEDS[0]), 1, CRITERIA, SEEDS[0])["valid"])
        r = passing_result(SEEDS[0])
        put(r, "advice_leaks", 1)
        self.assertFalse(aggregate.judge(r, 0, CRITERIA, SEEDS[0])["valid"])


class PairedBoundTests(unittest.TestCase):
    def test_the_lower_and_upper_bounds_are_pinned(self):
        gains = [7000, 5500, 6200, 4800, 8100, 5900, 6600, 5300, 7400, 6000, 6800, 5100, 5700, 7200, 6300, 5600, 6900, 5200, 6100, 6500]  # probe minus self-model, micro-units
        results = {s: passing_result(s, {"self_model": 200_000, "probe_matched": 200_000 + g}) for s, g in zip(SEEDS, gains)}
        item = paired(results)["judged"][0]
        diffs = [g / 1000 for g in gains]  # per mille of Brier
        mean = sum(diffs) / 20
        sd = (sum((d - mean) ** 2 for d in diffs) / 19) ** 0.5
        self.assertAlmostEqual(item["mean_permille"], mean, places=6)
        self.assertAlmostEqual(item["sd_permille"], sd, places=6)
        self.assertAlmostEqual(item["lower_bound_permille"], mean - 1.729133 * sd / 20 ** 0.5, places=4)
        self.assertAlmostEqual(item["upper_bound_permille"], mean + 1.729133 * sd / 20 ** 0.5, places=4)
        self.assertAlmostEqual(item["upper_bound_adjusted_permille"], mean + 2.093024 * sd / 20 ** 0.5, places=4)

    def test_the_sign_is_comparator_minus_self_model_so_a_worse_self_model_is_negative(self):
        results = {s: passing_result(s, {"self_model": 210_000, "probe_matched": 206_000, "own_confidence": 212_000}) for s in SEEDS}
        by = {i["against"]: i for i in paired(results)["judged"]}
        self.assertLess(by["probe_matched"]["mean_permille"], 0)
        self.assertGreater(by["own_confidence"]["mean_permille"], 0)

    def test_the_wrong_number_of_seeds_is_refused(self):
        results = clean_seeds()
        del results[SEEDS[0]]
        with self.assertRaises(agg.PairedError):
            paired(results)


class VerdictTests(unittest.TestCase):
    STRONG = {"self_model": 200_000, "probe_matched": 215_000, "own_confidence": 214_000}

    def test_completed_when_both_lower_bounds_reach_the_margin(self):
        verdict = decide(clean_seeds(self.STRONG))
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED, verdict["problems"])
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_not_shown_when_a_bound_straddles_the_margin(self):
        seeds = {s: passing_result(s, {"self_model": 200_000, "probe_matched": spread(204_000, 3_000, i), "own_confidence": 214_000}) for i, s in enumerate(SEEDS)}
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN, verdict["problems"])
        self.assertIsNone(verdict["recommended_status"])

    def test_refuted_when_the_upper_bound_is_below_the_margin(self):
        seeds = {s: passing_result(s, {"self_model": 200_000, "probe_matched": spread(200_500, 400, i), "own_confidence": 214_000}) for i, s in enumerate(SEEDS)}
        verdict = decide(seeds)
        self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "refuted"), verdict["problems"])
        self.assertEqual(verdict["refuted_against"], ["probe_matched"])
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_the_refutation_uses_the_bonferroni_quantile(self):
        # mean 2.5, sd about 3.3: the plain upper bound (t 1.729) is below the margin 4, the adjusted one (t 2.093) is not
        seeds = {s: passing_result(s, {"self_model": 200_000, "probe_matched": spread(202_500, 3_300, i), "own_confidence": 214_000}) for i, s in enumerate(SEEDS)}
        item = paired(seeds)["judged"][0]
        self.assertLess(item["upper_bound_permille"], MARGIN)
        self.assertGreaterEqual(item["upper_bound_adjusted_permille"], MARGIN)
        self.assertEqual(decide(seeds)["verdict"], aggregate.NOT_SHOWN)

    def test_both_comparators_must_be_beaten(self):
        seeds = {s: passing_result(s, {"self_model": 200_000, "probe_matched": 215_000, "own_confidence": 200_500}) for s in SEEDS}
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["refuted_against"], ["own_confidence"])

    def test_structural_falsification(self):
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[3]], "model_write_paths", 1)
        verdict = decide(seeds, exits={SEEDS[3]: 1})
        self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "structural"))
        self.assertEqual(verdict["falsified_seeds"], [SEEDS[3]])

    def test_an_integrity_counter_is_invalid_even_next_to_a_structural_failure(self):
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[0]], "advice_leaks", 1)
        put(seeds[SEEDS[1]], "split_overlap", 1)
        verdict = decide(seeds, exits={SEEDS[0]: 1, SEEDS[1]: 2}, use_paired=False)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["integrity_seeds"], [SEEDS[1]])

    def test_missing_or_incomplete_evidence_is_invalid(self):
        seeds = clean_seeds(self.STRONG)
        self.assertEqual(decide(seeds, mutations=None)["verdict"], aggregate.INVALID)
        self.assertEqual(decide(seeds, mutations={"killed": PLAN - 1, "total": PLAN})["verdict"], aggregate.INVALID)
        self.assertEqual(decide(seeds, mutations={"killed": PLAN - 1, "total": PLAN - 1})["verdict"], aggregate.INVALID)
        short = {s: r for s, r in seeds.items() if s != SEEDS[0]}
        self.assertEqual(aggregate.verdict_of(judged(short), PASSED, SEEDS, KILLED, None, PAIRED, PLAN)["verdict"], aggregate.INVALID)
        failed_static = aggregate.verdict_of(judged(seeds), [{"command": "x", "exit_code": 1}], SEEDS, KILLED, paired(seeds), PAIRED, PLAN)
        self.assertEqual(failed_static["verdict"], aggregate.FALSIFIED)

    def test_invalid_precedes_every_other_verdict(self):
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[3]], "model_write_paths", 1)
        tripped = {SEEDS[3]: 1}
        # a structural failure with partial, absent or foreign mutation evidence, or a missing seed, is INVALID, not FALSIFIED
        for mutations in (None, {"killed": PLAN - 1, "total": PLAN}, {"killed": PLAN - 1, "total": PLAN - 1}, {"killed": PLAN + 1, "total": PLAN + 1}):
            verdict = decide(seeds, exits=tripped, mutations=mutations)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, mutations)
        short = {s: r for s, r in seeds.items() if s != SEEDS[0]}
        self.assertEqual(aggregate.verdict_of(judged(short, tripped), PASSED, SEEDS, KILLED, None, PAIRED, PLAN)["verdict"], aggregate.INVALID)
        # an unjudgeable seed next to a structural failure
        seeds[SEEDS[5]]["scale"] = "custom"
        self.assertEqual(decide(seeds, exits=tripped)["verdict"], aggregate.INVALID)
        # a refutation with partial evidence is INVALID too
        refuting = {s: passing_result(s, {"self_model": 200_000, "probe_matched": spread(200_500, 400, i), "own_confidence": 214_000}) for i, s in enumerate(SEEDS)}
        self.assertEqual(decide(refuting)["verdict"], aggregate.FALSIFIED)
        self.assertEqual(decide(refuting, mutations={"killed": PLAN - 1, "total": PLAN})["verdict"], aggregate.INVALID)
        # and a completed one
        self.assertEqual(decide(clean_seeds(self.STRONG), mutations=None)["verdict"], aggregate.INVALID)

    def test_a_counter_above_zero_is_named_and_an_unmet_coverage_rule_is_structural(self):
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[1]], "split_overlap", 1)
        verdict = decide(seeds, exits={SEEDS[1]: 2}, use_paired=False)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any("harness integrity counters above 0 on seeds" in p for p in verdict["problems"]), verdict["problems"])
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[2]], "arms_trained", 3)
        verdict = decide(seeds, exits={SEEDS[2]: 1})
        self.assertEqual((verdict["verdict"], verdict["kind"], verdict["vacuous_seeds"]), (aggregate.FALSIFIED, "structural", [SEEDS[2]]))

    def test_a_missing_static_run_and_duplicated_judgements_are_invalid(self):
        seeds = clean_seeds(self.STRONG)
        outcome = paired(seeds)
        self.assertEqual(aggregate.verdict_of(judged(seeds), None, SEEDS, KILLED, outcome, PAIRED, PLAN)["verdict"], aggregate.INVALID)
        doubled = judged(seeds) + [aggregate.judge(seeds[SEEDS[0]], 0, CRITERIA, SEEDS[0])]
        verdict = aggregate.verdict_of(doubled, PASSED, SEEDS, KILLED, outcome, PAIRED, PLAN)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any("duplicated" in p for p in verdict["problems"]))

    def test_the_controls_are_judged_on_their_bounds_not_their_means(self):
        s = self.STRONG
        wide = lambda centre, i: spread(centre, 8_000, i)  # noqa: E731  mean `centre`, sd about 8 per mille: the lower bound sits about 3 per mille below it
        gross = {q: passing_result(q, {**s, "constant": wide(206_000, i)}) for i, q in enumerate(SEEDS)}
        self.assertGreaterEqual(paired(gross)["sensitivity"]["gross"]["mean_permille"], MARGIN)
        self.assertEqual(decide(gross)["verdict"], aggregate.INVALID)
        graded = {q: passing_result(q, {**s, "self_model_noisy": 200_000 + spread(3_000, 5_000, i)}) for i, q in enumerate(SEEDS)}
        self.assertGreater(paired(graded)["sensitivity"]["graded"]["mean_permille"], PAIRED["sensitivity_graded"]["min_lower_permille"])
        self.assertEqual(decide(graded)["verdict"], aggregate.INVALID)
        removal = {q: passing_result(q, {**s, "self_model_shuffled": 240_000 - spread(2_000, 8_000, i), "probe_matched_shuffled": 240_000 - spread(2_000, 8_000, i)}) for i, q in enumerate(SEEDS)}
        self.assertLess(paired(removal)["removal"]["self_model_shuffled"]["lower_bound_permille"], MARGIN)
        self.assertEqual(decide(removal)["verdict"], aggregate.INVALID)
        against_probe = {q: passing_result(q, {**s, "probe_matched_shuffled": 250_000 + spread(2_000, 8_000, i), "self_model_shuffled": 250_000}) for i, q in enumerate(SEEDS)}
        self.assertLess(paired(against_probe)["removal_vs_probe"]["lower_bound_permille"], MARGIN)
        self.assertEqual(decide(against_probe)["verdict"], aggregate.INVALID)

    def test_the_headline_names_the_comparator_the_verdict_rests_on(self):
        seeds = {s: passing_result(s, {"self_model": 200_000, "probe_matched": 215_000, "own_confidence": 205_000}) for s in SEEDS}
        verdict = decide(seeds)
        self.assertEqual(paired(seeds)["binding_comparator"], "own_confidence")
        self.assertEqual(verdict["binding_comparator"], "own_confidence")
        self.assertIn("rests on the comparator own_confidence", aggregate.report(verdict, judged(seeds), PASSED, KILLED, paired(seeds)))
        seeds = {s: passing_result(s, {"self_model": 200_000, "probe_matched": 205_000, "own_confidence": 215_000}) for s in SEEDS}
        self.assertEqual(decide(seeds)["binding_comparator"], "probe_matched")

    def test_heads_too_weak_to_test_the_question_are_invalid(self):
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[4]], "head_gap_id_permille", 5)
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["uninformative_seeds"], [SEEDS[4]])

    def test_no_signal_is_invalid(self):
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[2]], "signal_permille", 3)
        verdict = decide(seeds)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["uninformative_seeds"], [SEEDS[2]])

    def test_unconverged_judged_heads_are_tolerated_up_to_the_allowance(self):
        seeds = clean_seeds(self.STRONG)
        put(seeds[SEEDS[0]], "unconverged_judged_arms", 2)
        self.assertEqual(decide(seeds)["verdict"], aggregate.COMPLETED)
        put(seeds[SEEDS[1]], "unconverged_judged_arms", 1)
        self.assertEqual(decide(seeds)["verdict"], aggregate.INVALID)

    def test_each_control_must_resolve(self):
        s = self.STRONG
        # gross: the constant predictor is as good as the self-model (the traces carry no usable information)
        self.assertEqual(decide({q: passing_result(q, {**s, "constant": 201_000}) for q in SEEDS})["verdict"], aggregate.INVALID)
        # graded: the noised self-model is as good as the intact one
        self.assertEqual(decide({q: passing_result(q, {**s, "self_model_noisy": 200_500}) for q in SEEDS})["verdict"], aggregate.INVALID)
        # removal: shuffled traces, or a self-model trained on permuted outcomes, still beat the constant predictor by the margin
        self.assertEqual(decide({q: passing_result(q, {**s, "self_model_shuffled": 232_000}) for q in SEEDS})["verdict"], aggregate.INVALID)
        self.assertEqual(decide({q: passing_result(q, {**s, "self_model_perm": 232_000}) for q in SEEDS})["verdict"], aggregate.INVALID)
        # removal against the probe: with the same shuffled traces the self-model still beats the probe by the margin
        self.assertEqual(decide({q: passing_result(q, {**s, "self_model_shuffled": 240_000, "probe_matched_shuffled": 245_000}) for q in SEEDS})["verdict"], aggregate.INVALID)
        self.assertEqual(decide({q: passing_result(q, {**s, "self_model_shuffled": 240_000, "probe_matched_shuffled": 243_900}) for q in SEEDS})["verdict"], aggregate.COMPLETED)
        # null world: an advantage of the margin's size where none can exist, in either direction
        self.assertEqual(decide({q: passing_result(q, {**s, "self_model_null": 100_000, "probe_matched_null": 108_000}) for q in SEEDS})["verdict"], aggregate.INVALID)
        self.assertEqual(decide({q: passing_result(q, {**s, "self_model_null": 108_000, "probe_matched_null": 100_000}) for q in SEEDS})["verdict"], aggregate.INVALID)

    def test_the_controls_precede_a_refutation(self):
        seeds = {q: passing_result(q, {"self_model": 200_000, "probe_matched": 200_500, "self_model_shuffled": 220_000}) for q in SEEDS}
        self.assertEqual(decide(seeds)["verdict"], aggregate.INVALID)


class WiringTests(unittest.TestCase):
    def test_a_seed_in_another_environment_is_invalidated(self):
        seeds = clean_seeds(VerdictTests.STRONG)
        entries = judged(seeds)
        environments = {s: dict(ENVIRONMENT) for s in SEEDS}
        environments[SEEDS[5]]["torch"] = "other"
        agg.invalidate_environment_mismatches(entries, environments)
        self.assertEqual([e["seed"] for e in entries if not e["valid"]], [SEEDS[5]])
        self.assertEqual(aggregate.verdict_of(entries, PASSED, SEEDS, KILLED, paired(seeds), PAIRED, PLAN)["verdict"], aggregate.INVALID)

    def test_the_report_names_the_verdict_and_every_comparison(self):
        seeds = clean_seeds(VerdictTests.STRONG)
        text = aggregate.report(decide(seeds), judged(seeds), PASSED, KILLED, paired(seeds))
        for needle in ("COMPLETED", "vs probe_matched", "vs own_confidence", "gross sensitivity", "graded sensitivity", "removal control self_model_shuffled",
                       "removal control self_model_perm", "removal control self_model_shuffled against probe_matched_shuffled", "headline", "null world", "(descriptive) probe_io", "mutation check"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
