"""The C001 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed`,
`paired_comparisons` and `decide`, which read no file. These tests build a result that holds every
rule of the criteria and break one rule at a time, so a verdict that stopped depending on it fails
here. The paired bounds are pinned to hand-computed values, so a flipped sign, a wrong quantile or a
wrong divisor fails. The outcomes are four: structural FALSIFIED, INVALID (saturated, floored,
unconverged, insensitive, harness integrity, missing evidence), refuted FALSIFIED (the UPPER bound below
the margin: the advantage is below the margin), NOT_SHOWN (straddling) and COMPLETED (every LOWER bound
at the margin). The attribution contrasts are three-valued and never change the verdict. That the harness's rules equal
criteria.toml and that every hard counter is planted by a mutation is held by
research/consciousness/tests/test_rules.py."""

import copy
import importlib.util
import subprocess
import sys
import tempfile
import tomllib
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "consciousness" / "C001-global-workspace"
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("c001_aggregate", HERE / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
PASSED: list[dict] = []  # this experiment names no static evidence
PLAN = len(MUTATIONS["mutation"])
KILLED = {"killed": PLAN, "total": PLAN}
ARMS = ("workspace", "all_to_all", "bw_all_to_all", "sparse_all_to_all", "no_broadcast", "no_comm", "competition_off", "capacity_off", "weak_all_to_all", "crippled_all_to_all")
TOTAL = 1000  # evaluation instances per synthetic arm, so counts are per mille
MARGIN = CRITERIA["paired"]["margin_permille"]
BASE = {"workspace": 700, "all_to_all": 600, "bw_all_to_all": 590, "sparse_all_to_all": 595, "no_broadcast": 400, "no_comm": 330,
        "competition_off": 650, "capacity_off": 640, "weak_all_to_all": 530, "crippled_all_to_all": 420}
ENVIRONMENT = {k: "x" for k in aggregate.ENVIRONMENT_KEYS}


def put(result: dict, path: str, value) -> None:
    at = result
    keys = path.split(".")
    for key in keys[:-1]:
        at = at.setdefault(key, {})
    at[keys[-1]] = value


def arm_record(ood: int, id_: int = 800) -> dict:
    return {
        "ood16_correct": ood, "ood16_total": TOTAL, "ood24_correct": ood, "ood24_total": TOTAL,
        "id_correct": id_, "id_total": TOTAL, "auc_correct": 500, "auc_total": TOTAL,
        "drop_irrelevant_correct": 700, "drop_irrelevant_total": TOTAL, "drop_relevant_correct": 300, "drop_relevant_total": TOTAL,
    }


def passing_result(seed: int, ood: dict | None = None) -> dict:
    """A harness result in which every rule and guard of the criteria holds."""
    ood = {**BASE, **(ood or {})}
    result = {
        "benchmark": "c001", "experiment": "C001", "seed": seed, "scale": "pinned",
        "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
        "controls": {name: {"required_rule_tripped": True} for name in aggregate.CONTROLS},
        "harness_gaps": [], "workspace_trained": 1, "environment": dict(ENVIRONMENT),
        "arms": {arm: arm_record(ood[arm]) for arm in ARMS},
    }
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["integrity"]:
        left, _, right = aggregate.parse_rule(rule)
        put(result, left, int(right))
    result["comparator_primary_permille"] = max(ood["all_to_all"], ood["bw_all_to_all"], ood["sparse_all_to_all"])
    result["unconverged_judged_arms"] = 0
    return result


def judged(results: dict, exits: dict | None = None) -> list[dict]:
    exits = exits or {}
    return [aggregate.judge_seed(results[seed], exits.get(seed, 0), CRITERIA, seed) for seed in results]


def clean_seeds(ood: dict | None = None) -> dict:
    return {seed: passing_result(seed, ood) for seed in SEEDS}


def paired(results: dict) -> dict:
    return aggregate.paired_comparisons(results, CRITERIA["paired"])


def decide(results, exits=None, mutations=KILLED, use_paired=True, plan=PLAN):
    outcome = paired(results) if use_paired else None
    return aggregate.verdict_of(judged(results, exits), PASSED, SEEDS, mutations, outcome, CRITERIA["paired"], plan)


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
        for key in ("coverage", "guards", "falsifiers", "integrity"):
            rules = CRITERIA[key]
            self.assertTrue(rules)
            self.assertEqual(len(rules), len(set(rules)), key)
            for rule in rules:
                aggregate.parse_rule(rule)
        self.assertEqual(len(CRITERIA["falsifiers"]), 6)
        self.assertEqual(len(CRITERIA["integrity"]), 7)
        self.assertEqual(len(CRITERIA["guards"]), 2)  # the convergence guard is a count across seeds, not a per-seed rule
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_paired_table_agrees_with_the_pinned_configuration(self):
        pinned, table = CONFIG["pinned_thresholds"], CRITERIA["paired"]
        self.assertEqual(table["margin_permille"], pinned["paired_margin_permille"])
        self.assertEqual(table["sensitivity_gross"]["margin_permille"], pinned["sensitivity_margin_permille"])
        self.assertEqual(table["t_support"], pinned["paired_t_support"])
        self.assertEqual(table["t_refute"], pinned["paired_t_refute"])
        self.assertEqual(table["attribution_margin_permille"], pinned["paired_margin_permille"])
        self.assertEqual(table["sensitivity_graded"]["min_lower_permille"], pinned["graded_min_lower_permille"])
        self.assertEqual(table["max_untrained_workspace_seeds"], pinned["max_untrained_workspace_seeds"])
        self.assertEqual(table["max_unconverged_seeds"], pinned["max_unconverged_seeds"])
        self.assertEqual(table["against"], table["pairwise"])
        self.assertEqual(table["attribution"], ["no_broadcast", "capacity_off", "competition_off"])
        self.assertEqual(table["seeds_required"], len(pinned["seeds"]))
        self.assertEqual(pinned["seeds"], SEEDS)
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        self.assertTrue(set(table["against"]) < set(ARMS) and table["arm"] == "workspace")
        self.assertEqual(table["metric"], "ood16")

    def test_confirmatory_seeds_are_fresh_and_disjoint_from_the_pilots(self):
        pinned = CONFIG["pinned_thresholds"]
        self.assertGreaterEqual(len(SEEDS), 10)
        self.assertLessEqual(len(SEEDS), 20)
        self.assertEqual(set(SEEDS) & {17, 29, 43, 71, 101}, set())
        self.assertEqual(set(SEEDS) & set(pinned["pilot_seeds"]), set())

    def test_the_t_quantiles_are_the_one_sided_values_for_the_declared_seed_count(self):
        # Student t quantiles: df 9: 1.833113 / 2.509587; df 14: 1.761310 / 2.359820; df 19: 1.729133 / 2.294392 (alpha 0.05 and 0.05/3).
        df = len(SEEDS) - 1
        support = {9: 1.833113, 14: 1.761310, 19: 1.729133}[df]
        refute = {9: 2.509587, 14: 2.359820, 19: 2.294392}[df]
        self.assertAlmostEqual(float(CRITERIA["paired"]["t_support"]), support, places=5)
        self.assertAlmostEqual(float(CRITERIA["paired"]["t_refute"]), refute, places=5)

    def test_the_reference_result_holds_every_rule_and_guard(self):
        result = passing_result(SEEDS[0])
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["integrity"] + CRITERIA["guards"]:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["integrity"] + CRITERIA["guards"]:
            left, op, right = aggregate.parse_rule(rule)
            result = passing_result(SEEDS[0])
            value = aggregate.resolve(result, left)
            put(result, left, int(right) - 1 if op == ">=" else int(right) + 1)
            self.assertFalse(aggregate.evaluate(result, rule), rule)

    @unittest.skipUnless(
        "C001" in tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"],
        "C001 is not in experiments/preregistration.toml",
    )
    def test_the_config_table_and_the_registry_agree(self):
        table = CONFIG["preregistration"]
        required = tomllib.loads((ROOT / "experiments" / "preregistration.toml").read_text(encoding="utf-8"))["experiment"]["C001"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))
        self.assertEqual(table["seeds"], SEEDS)
        for key in ("criteria", "preregistration_text"):
            path = ROOT / table[key]
            self.assertTrue(path.is_file(), key)


class PairedBoundTests(unittest.TestCase):
    T = aggregate.Decimal("1.833113")

    def test_the_bounds_equal_the_hand_computed_values(self):
        # ten differences 50, 70, 30, 60, 40, 50, 70, 30, 60, 40 per mille: mean 50, sample sd sqrt(2000/9)
        diffs = [Fraction(d, 1000) for d in (50, 70, 30, 60, 40, 50, 70, 30, 60, 40)]
        bound = aggregate.paired_bound(diffs, self.T)
        sd = (2000 / 9) ** 0.5
        self.assertAlmostEqual(bound["mean_permille"], 50.0, places=9)
        self.assertAlmostEqual(bound["sd_permille"], sd, places=9)
        self.assertAlmostEqual(bound["lower_bound_permille"], 50 - 1.833113 * sd / 10 ** 0.5, places=6)
        self.assertAlmostEqual(bound["upper_bound_permille"], 50 + 1.833113 * sd / 10 ** 0.5, places=6)

    def test_constant_differences_have_equal_bounds(self):
        bound = aggregate.paired_bound([Fraction(1, 20)] * 10, self.T)
        self.assertAlmostEqual(bound["lower_bound_permille"], 50.0, places=9)
        self.assertAlmostEqual(bound["upper_bound_permille"], 50.0, places=9)

    def test_negating_the_differences_swaps_the_bounds(self):
        diffs = [Fraction(d, 1000) for d in (50, 70, 30, 60, 40, 50, 70, 30, 60, 40)]
        a, b = aggregate.paired_bound(diffs, self.T), aggregate.paired_bound([-d for d in diffs], self.T)
        self.assertAlmostEqual(a["lower_bound_permille"], -b["upper_bound_permille"], places=9)

    def test_fewer_than_two_seeds_have_no_bound(self):
        with self.assertRaises(aggregate.PairedError):
            aggregate.paired_bound([Fraction(1, 10)], self.T)


class PairedRuleTests(unittest.TestCase):
    def results(self, ws, a2a=600, bw=590, nb=400, spread=None, sparse=595, **extra):
        spread = (spread or [0] * 10) * (len(SEEDS) // 10 + 1)
        out = {}
        for seed, jitter in zip(SEEDS, spread):
            out[seed] = passing_result(seed, {"workspace": ws + jitter, "all_to_all": a2a, "bw_all_to_all": bw, "sparse_all_to_all": sparse, "no_broadcast": nb, **extra})
        return out

    def test_support_needs_every_lower_bound_at_the_margin_inclusive(self):
        self.assertTrue(paired(self.results(600 + MARGIN, 600, 600, 400))["supported"])
        self.assertFalse(paired(self.results(600 + MARGIN - 1, 600, 600, 400))["supported"])

    def test_beating_only_some_comparators_is_not_support(self):
        outcome = paired(self.results(600 + MARGIN, 600, 600 + MARGIN, 400))
        self.assertFalse(outcome["supported"])
        self.assertEqual([item["supports"] for item in outcome["judged"]], [True, False, True])
        sparse_ties = paired(self.results(600 + MARGIN, 600, 600, 400, sparse=600 + MARGIN))
        self.assertFalse(sparse_ties["supported"])
        self.assertEqual([item["supports"] for item in sparse_ties["judged"]], [True, True, False])

    def test_refutation_needs_the_upper_bound_below_the_margin(self):
        pairwise = ["all_to_all", "bw_all_to_all", "sparse_all_to_all"]
        equal = paired(self.results(600, 600, 600, 400, sparse=600))
        self.assertEqual(equal["refuted_against"], pairwise)
        just_below = paired(self.results(600 + MARGIN - 1, 600, 600, 400, sparse=600))
        self.assertEqual(just_below["refuted_against"], pairwise)
        at_margin = paired(self.results(600 + MARGIN, 600, 600, 400, sparse=600))
        self.assertEqual(at_margin["refuted_against"], [])

    def test_a_tie_with_no_broadcast_is_not_in_the_support_conjunction_and_never_refutes_the_workspace(self):
        outcome = paired(self.results(700, 600, 590, 700))
        self.assertEqual(outcome["refuted_against"], [])
        self.assertTrue(outcome["supported"])
        self.assertEqual([item["against"] for item in outcome["judged"]], ["all_to_all", "bw_all_to_all", "sparse_all_to_all"])
        self.assertEqual(outcome["attribution_states"]["broadcast"], "not attributed")
        verdict = decide(self.results(700, 600, 590, 700))
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertTrue(verdict["broadcast_downgraded"])
        self.assertIn("broadcast: not attributed", verdict["attribution"])

    def test_the_margin_is_the_a_priori_thirty_per_mille(self):
        self.assertEqual(MARGIN, 30)
        self.assertEqual(CRITERIA["paired"]["attribution_margin_permille"], MARGIN)

    def test_refutation_uses_the_bonferroni_quantile(self):
        spread = [-80, 80, -60, 60, -40, 40, -20, 20, 0, 0]
        outcome = paired(self.results(600 + MARGIN, 600, 600, 400, sparse=600, spread=spread))
        item = outcome["judged"][0]
        self.assertGreater(item["upper_bound_adjusted_permille"], item["upper_bound_permille"])

    def test_the_unadjusted_quantile_would_refute_where_the_bonferroni_one_does_not(self):
        # Mean difference 5, sd about 50: upper bound 24.4 at t 1.729 (would refute) and 30.8 at t 2.294 (does not).
        spread = [-80, 80, -60, 60, -40, 40, -20, 20, 0, 0]
        outcome = paired(self.results(605, 600, 600, 400, sparse=600, spread=spread))
        for item in outcome["judged"]:
            self.assertLess(item["upper_bound_permille"], MARGIN)
            self.assertGreaterEqual(item["upper_bound_adjusted_permille"], MARGIN)
            self.assertFalse(item["refutes"], item["against"])
        self.assertEqual(outcome["refuted_against"], [])
        verdict = decide(self.results(605, 600, 600, 400, sparse=600, spread=spread))
        self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN)

    def test_a_noisy_difference_around_the_margin_is_neither_supported_nor_refuted(self):
        outcome = paired(self.results(600 + MARGIN, 600, 600, 400, spread=[-90, 90, -70, 70, -50, 50, -30, 30, -10, 10]))
        self.assertFalse(outcome["supported"])
        self.assertEqual(outcome["refuted_against"], [])

    def test_the_gross_sensitivity_control_must_resolve_its_margin(self):
        margin = CRITERIA["paired"]["sensitivity_gross"]["margin_permille"]
        self.assertTrue(paired(self.results(700, crippled_all_to_all=600 - margin))["sensitivity"]["gross"]["resolves"])
        self.assertFalse(paired(self.results(700, crippled_all_to_all=600 - margin + 1))["sensitivity"]["gross"]["resolves"])

    def test_the_graded_sensitivity_control_must_detect_its_deficit(self):
        floor = CRITERIA["paired"]["sensitivity_graded"]["min_lower_permille"]
        self.assertEqual(floor, MARGIN // 2)
        self.assertTrue(paired(self.results(700, weak_all_to_all=600 - floor - 1))["sensitivity"]["graded"]["resolves"])
        self.assertFalse(paired(self.results(700, weak_all_to_all=600 - floor))["sensitivity"]["graded"]["resolves"])

    def test_the_attribution_contrasts_are_three_valued_and_never_change_support(self):
        outcome = paired(self.results(700, capacity_off=700 - MARGIN, competition_off=700 - MARGIN + 1))
        self.assertTrue(outcome["supported"])
        self.assertTrue(outcome["bottleneck_attributed"])
        self.assertFalse(outcome["competition_attributed"])
        self.assertEqual(outcome["attribution_states"], {"broadcast": "attributed", "bottleneck": "attributed", "competition": "not attributed"})
        self.assertEqual(paired(self.results(700, capacity_off=700))["attribution_states"]["bottleneck"], "not attributed")
        # Boundaries (constant differences, so both bounds equal the difference): lower >= margin attributes, upper < margin does not.
        self.assertEqual(paired(self.results(700, capacity_off=700 - MARGIN))["attribution_states"]["bottleneck"], "attributed")
        self.assertEqual(paired(self.results(700, capacity_off=700 - MARGIN + 1))["attribution_states"]["bottleneck"], "not attributed")

    def test_a_noisy_contrast_is_unresolved_not_not_attributed(self):
        noisy = [-90, 90, -70, 70, -50, 50, -30, 30, -10, 10]
        outcome = paired(self.results(700, spread=noisy, capacity_off=700 - MARGIN))
        self.assertEqual(outcome["attribution_states"]["bottleneck"], "unresolved by noise")
        item = next(a for a in outcome["attribution"] if a["against"] == "capacity_off")
        self.assertAlmostEqual(item["mean_permille"], 30.0, places=9)
        self.assertLess(item["lower_bound_permille"], MARGIN)
        self.assertGreater(item["upper_bound_permille"], MARGIN)
        self.assertAlmostEqual(item["half_width_permille"], (item["upper_bound_permille"] - item["lower_bound_permille"]) / 2, places=9)

    def test_the_attribution_contrasts_report_the_median_and_the_sign_count(self):
        noisy = [-90, 90, -70, 70, -50, 50, -30, 30, -10, 10]
        outcome = paired(self.results(700, spread=noisy, capacity_off=700))
        item = next(a for a in outcome["attribution"] if a["against"] == "capacity_off")
        # differences are the twenty jitters: median (-10 + 10) / 2 = 0, ten of them positive
        self.assertAlmostEqual(item["median_permille"], 0.0, places=9)
        self.assertEqual((item["positive_seeds"], item["n"]), (10, 20))
        skewed = paired(self.results(700, spread=[300] + [0] * 9, capacity_off=700))
        item = next(a for a in skewed["attribution"] if a["against"] == "capacity_off")
        self.assertAlmostEqual(item["median_permille"], 0.0, places=9)
        self.assertGreater(item["mean_permille"], 25)
        self.assertEqual(item["positive_seeds"], 2)

    def test_the_wrong_number_of_seeds_cannot_be_judged(self):
        partial = self.results(700)
        del partial[SEEDS[0]]
        with self.assertRaises(aggregate.PairedError):
            paired(partial)

    def test_counts_that_are_not_counts_cannot_be_judged(self):
        for damage in (
            lambda r: r["arms"]["workspace"].update(ood16_correct=True),
            lambda r: r["arms"]["workspace"].update(ood16_correct=1001),
            lambda r: r["arms"]["workspace"].update(ood16_total=0),
            lambda r: r["arms"].pop("no_broadcast"),
            lambda r: r["arms"]["all_to_all"].update(ood16_correct=0.5),
        ):
            results = self.results(700)
            damage(results[SEEDS[0]])
            with self.assertRaises(aggregate.PairedError):
                paired(results)

    def test_descriptive_metrics_are_reported_and_never_judged(self):
        outcome = paired(self.results(700))
        metrics = {item["metric"] for item in outcome["descriptive"]}
        self.assertEqual(metrics, set(CRITERIA["paired"]["descriptive_metrics"]))
        against = {item["against"] for item in outcome["descriptive"]}
        self.assertIn("no_comm", against)
        results = self.results(700)
        for seed in SEEDS:
            results[seed]["arms"]["workspace"]["id_correct"] = 0
        self.assertTrue(paired(results)["supported"])


class VerdictTests(unittest.TestCase):
    def results(self, ws=700, **extra):
        return PairedRuleTests.results(self, ws, **extra)

    def test_support_with_the_evidence_is_completed(self):
        verdict = decide(self.results(700))
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        self.assertEqual(verdict["recommended_status"], "completed")

    def test_a_clear_null_is_refuted_and_falsified(self):
        verdict = decide(self.results(600))
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertEqual(verdict["kind"], "refuted")
        self.assertEqual(verdict["recommended_status"], "failed")

    def test_a_straddling_interval_is_not_shown_and_changes_nothing(self):
        verdict = decide(PairedRuleTests.results(self, 600 + MARGIN, spread=[-90, 90, -70, 70, -50, 50, -30, 30, -10, 10]))
        self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN)
        self.assertIsNone(verdict["recommended_status"])
        self.assertIsNone(verdict["kind"])

    def test_a_saturated_or_floored_comparator_is_invalid_never_falsified(self):
        for comparator in (CONFIG["pinned_thresholds"]["guard_ceiling_permille"] + 1, CONFIG["pinned_thresholds"]["guard_floor_permille"] - 1):
            results = self.results(600)
            put(results[SEEDS[3]], "comparator_primary_permille", comparator)
            verdict = decide(results)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, comparator)
            self.assertEqual(verdict["uninformative_seeds"], [SEEDS[3]])
            self.assertIsNone(verdict["recommended_status"])

    def test_an_insensitive_pipeline_is_invalid(self):
        for extra in ({"crippled_all_to_all": 600}, {"weak_all_to_all": 600}):
            verdict = decide(self.results(600, **extra))
            self.assertEqual(verdict["verdict"], aggregate.INVALID, extra)
            self.assertTrue(any("sensitivity" in p for p in verdict["problems"]))

    def test_a_supported_outcome_names_what_it_attributes_to(self):
        verdict = decide(self.results(700, capacity_off=700 - MARGIN))
        self.assertEqual((verdict["verdict"], verdict["attribution"]),
                         (aggregate.COMPLETED, "workspace effect shown; broadcast: attributed; bottleneck: attributed; competition: attributed"))
        verdict = decide(self.results(700, capacity_off=700))
        self.assertEqual((verdict["verdict"], verdict["attribution"]),
                         (aggregate.COMPLETED, "workspace effect shown; broadcast: attributed; bottleneck: not attributed; competition: attributed"))
        self.assertIsNone(decide(self.results(600))["attribution"])

    def test_broadcast_attributed_without_a_pairwise_advantage_is_the_reported_partial_pattern(self):
        verdict = decide(PairedRuleTests.results(self, 600 + MARGIN, 600, 600 + MARGIN // 2, 400, spread=[-90, 90, -70, 70, -50, 50, -30, 30, -10, 10]))
        self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN)
        self.assertIn("GWT-3", verdict["partial_pattern"])
        self.assertFalse(decide(self.results(700))["partial_pattern"])

    def test_the_refuted_outcome_reads_below_the_margin_not_no_advantage(self):
        verdict = decide(PairedRuleTests.results(self, 600 + MARGIN - 1, 600, 600, 400, sparse=600))
        self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "refuted"))
        self.assertEqual(verdict["refuted_against"], ["all_to_all", "bw_all_to_all", "sparse_all_to_all"])
        text = " ".join(aggregate.LIMITATIONS) + aggregate.SCOPE
        self.assertNotIn("no advantage", text)

    def test_the_convergence_guard_tolerates_the_workspace_allowance_of_seeds_and_no_more(self):
        limit = CRITERIA["paired"]["max_unconverged_seeds"]
        results = self.results(600)
        for seed in SEEDS[:limit]:
            put(results[seed], "unconverged_judged_arms", 1)
        verdict = decide(results)
        self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "refuted"))
        self.assertEqual(verdict["unconverged_seeds"], SEEDS[:limit])
        put(results[SEEDS[limit]], "unconverged_judged_arms", 2)
        verdict = decide(results)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["unconverged_seeds"], SEEDS[: limit + 1])
        self.assertTrue(any("step cap" in p for p in verdict["problems"]))

    def test_the_convergence_count_must_be_a_count(self):
        for bad in (None, True, -1, 0.0, "0"):
            results = self.results(700)
            put(results[SEEDS[1]], "unconverged_judged_arms", bad)
            self.assertFalse(judged(results)[1]["valid"], bad)
        results = self.results(700)
        del results[SEEDS[1]]["unconverged_judged_arms"]
        self.assertFalse(judged(results)[1]["valid"])

    def test_a_mixed_environment_is_reported_per_seed(self):
        environments = {seed: dict(ENVIRONMENT) for seed in SEEDS}
        self.assertEqual(aggregate.environment_mismatches(environments), {})
        environments[SEEDS[2]]["torch"] = "other"
        environments[SEEDS[3]]["cpu_model"] = "other"
        self.assertEqual(aggregate.environment_mismatches(environments), {SEEDS[2]: ["torch"], SEEDS[3]: ["cpu_model"]})
        environments[SEEDS[4]] = None
        self.assertEqual(sorted(aggregate.environment_mismatches(environments)), [SEEDS[2], SEEDS[3]])

    def test_a_result_without_a_complete_environment_is_invalid(self):
        for damage in (lambda r: r.pop("environment"), lambda r: r["environment"].pop("torch"), lambda r: r["environment"].update(cpu_model="")):
            results = self.results(700)
            damage(results[SEEDS[0]])
            self.assertFalse(judged(results)[0]["valid"])

    def test_a_workspace_that_does_not_train_is_invalid_beyond_the_allowance(self):
        limit = CRITERIA["paired"]["max_untrained_workspace_seeds"]
        results = self.results(600)
        for seed in SEEDS[:limit]:
            results[seed]["workspace_trained"] = 0
        self.assertEqual(decide(results)["verdict"], aggregate.FALSIFIED)
        results[SEEDS[limit]]["workspace_trained"] = 0
        self.assertEqual(decide(results)["verdict"], aggregate.INVALID)

    def test_each_integrity_counter_on_one_seed_is_invalid_never_falsified(self):
        for rule in CRITERIA["integrity"]:
            left, _, right = aggregate.parse_rule(rule)
            for exit_code in (2, 1, 0):  # whatever the harness exits, an integrity counter makes the seed unjudgeable
                results = self.results(700)
                put(results[SEEDS[2]], left, int(right) + 1)
                verdict = decide(results, {SEEDS[2]: exit_code})
                self.assertEqual(verdict["verdict"], aggregate.INVALID, (rule, exit_code))
                self.assertIsNone(verdict["kind"], rule)
                self.assertEqual(verdict["integrity_seeds"], [SEEDS[2]], rule)
                self.assertIsNone(verdict["recommended_status"], rule)

    def test_the_four_formerly_structural_counters_are_integrity_counters(self):
        for counter in ("n_leaks", "train_eval_overlap", "param_mismatch_arms", "nonfinite_arms"):
            self.assertIn(f"{counter} <= 0", CRITERIA["integrity"])
            self.assertNotIn(f"{counter} <= 0", CRITERIA["falsifiers"])

    def test_an_integrity_counter_outranks_a_structural_finding_on_another_seed(self):
        results = self.results(700)
        put(results[SEEDS[0]], "no_comm_leaks", 1)
        put(results[SEEDS[1]], "n_leaks", 1)
        self.assertEqual(decide(results, {SEEDS[0]: 1, SEEDS[1]: 2})["verdict"], aggregate.INVALID)

    def test_each_falsifier_on_one_seed_is_a_structural_falsification(self):
        for rule in CRITERIA["falsifiers"]:
            left, _, right = aggregate.parse_rule(rule)
            results = self.results(700)
            put(results[SEEDS[2]], left, int(right) + 1)
            verdict = decide(results, {SEEDS[2]: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["kind"], "structural", rule)
            self.assertEqual(verdict["falsified_seeds"], [SEEDS[2]], rule)

    def test_an_unmet_coverage_rule_is_a_structural_falsification(self):
        for rule in CRITERIA["coverage"]:
            left, _, right = aggregate.parse_rule(rule)
            results = self.results(700)
            put(results[SEEDS[9]], left, int(right) - 1)
            verdict = decide(results, {SEEDS[9]: 1})
            self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "structural"), rule)
            self.assertEqual(verdict["vacuous_seeds"], [SEEDS[9]], rule)

    def test_a_structural_defect_beats_an_uninformative_seed(self):
        results = self.results(700)
        put(results[SEEDS[0]], "no_comm_leaks", 1)
        put(results[SEEDS[1]], "comparator_primary_permille", 999)
        self.assertEqual(decide(results, {SEEDS[0]: 1})["verdict"], aggregate.FALSIFIED)

    def test_a_missing_seed_is_invalid(self):
        results = self.results(700)
        del results[SEEDS[6]]
        verdict = aggregate.decide(judged(results), PASSED, SEEDS, KILLED, None, 0, 2, PLAN, 2)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_a_counter_that_is_missing_null_or_mistyped_is_invalid_not_failed(self):
        for damage in (
            lambda r: r.pop("n_leaks"), lambda r: r.update(n_leaks=None), lambda r: r.update(n_leaks=0.0),
            lambda r: r.update(n_leaks=False), lambda r: r.update(n_leaks="0"), lambda r: r.pop("workspace_influence_hits"),
            lambda r: r.pop("comparator_primary_permille"), lambda r: r.update(workspace_trained=2), lambda r: r.pop("workspace_trained"),
        ):
            results = self.results(700)
            damage(results[SEEDS[4]])
            self.assertFalse(judged(results)[SEEDS.index(SEEDS[4])]["valid"])
            self.assertEqual(aggregate.decide(judged(results), PASSED, SEEDS, KILLED, None, 0, 2, PLAN, 2)["verdict"], aggregate.INVALID)

    def test_a_result_that_is_not_the_run_is_invalid(self):
        damages = [
            lambda r: r.update(scale="custom"), lambda r: r.update(seed=18), lambda r: r.update(benchmark="other"),
            lambda r: r.update(experiment="S002"), lambda r: r.update(harness_gaps=["x"]),
            lambda r: r["procedure"].update(seed_is_declared=False), lambda r: r["procedure"].update(shape_is_pinned=False),
            lambda r: r["controls"].pop("bottleneck"), lambda r: r.pop("controls"),
        ] + [(lambda name: lambda r: r["controls"][name].update(required_rule_tripped=False))(name) for name in aggregate.CONTROLS]
        for damage in damages:
            results = self.results(700)
            damage(results[SEEDS[0]])
            self.assertEqual(decide(results, use_paired=False)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(decide(self.results(700), {SEEDS[0]: 1})["verdict"], aggregate.INVALID)
        results = self.results(700)
        put(results[SEEDS[0]], "n_leaks", 1)
        self.assertEqual(decide(results, {SEEDS[0]: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(decide(self.results(700), {SEEDS[0]: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(decide(self.results(700), {SEEDS[0]: None})["verdict"], aggregate.INVALID)

    def test_a_guard_failure_does_not_change_the_harness_exit_status(self):
        results = self.results(700)
        put(results[SEEDS[0]], "comparator_primary_permille", 999)
        self.assertTrue(judged(results)[0]["valid"])
        self.assertEqual(judged(results)[0]["guards_failed"], ["comparator_primary_permille <= 900"])

    def test_a_seed_without_a_result_is_invalid(self):
        results = self.results(700)
        results[SEEDS[2]] = None
        self.assertEqual(aggregate.decide(judged(results, {SEEDS[2]: 2}), PASSED, SEEDS, KILLED, None, 0, 2, PLAN, 2)["verdict"], aggregate.INVALID)

    def test_judgements_do_not_alias_their_inputs(self):
        results = self.results(700)
        before = copy.deepcopy(results)
        judged(results)
        paired(results)
        self.assertEqual(results, before)


class WiringTests(unittest.TestCase):
    """The glue `aggregate()` applies around the pure rules: the count of seeds whose workspace did not train, the invalidation of
    seeds recorded in another environment, and the refusal of a seed that ran twice. Each was a mutation of the aggregator that the
    tests of `decide` and `environment_mismatches` alone did not catch."""

    def results(self):
        return PairedRuleTests.results(self, 700)

    def test_the_untrained_workspace_seeds_are_counted_from_the_judgements(self):
        results = self.results()
        self.assertEqual(aggregate.untrained_workspace_seeds(judged(results)), 0)
        count = CRITERIA["paired"]["max_untrained_workspace_seeds"] + 1
        for seed in SEEDS[:count]:
            results[seed]["workspace_trained"] = 0
        self.assertEqual(aggregate.untrained_workspace_seeds(judged(results)), count)
        verdict = aggregate.verdict_of(judged(results), PASSED, SEEDS, KILLED, paired(results), CRITERIA["paired"], PLAN)
        self.assertEqual(verdict["untrained_workspace_seeds"], count)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any(f"did not train on {count} seeds" in problem for problem in verdict["problems"]))

    def test_the_allowances_come_from_the_criteria(self):
        results = self.results()
        limit = CRITERIA["paired"]["max_untrained_workspace_seeds"]
        for seed in SEEDS[:limit]:
            results[seed]["workspace_trained"] = 0
        verdict = aggregate.verdict_of(judged(results), PASSED, SEEDS, KILLED, paired(results), CRITERIA["paired"], PLAN)
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)
        stricter = {**CRITERIA["paired"], "max_untrained_workspace_seeds": limit - 1}
        self.assertEqual(aggregate.verdict_of(judged(results), PASSED, SEEDS, KILLED, paired(results), stricter, PLAN)["verdict"], aggregate.INVALID)

    def test_a_seed_recorded_in_another_environment_is_invalidated_and_the_run_is_invalid(self):
        results = self.results()
        entries = judged(results)
        environments = {seed: dict(ENVIRONMENT) for seed in SEEDS}
        self.assertEqual(aggregate.invalidate_environment_mismatches(entries, environments), {})
        self.assertTrue(all(entry["valid"] for entry in entries))
        environments[SEEDS[5]]["cpu_model"] = "another"
        mismatches = aggregate.invalidate_environment_mismatches(entries, environments)
        self.assertEqual(mismatches, {SEEDS[5]: ["cpu_model"]})
        flagged = [entry for entry in entries if not entry["valid"]]
        self.assertEqual([entry["seed"] for entry in flagged], [SEEDS[5]])
        self.assertIn("cpu_model", " ".join(flagged[0]["invalid"]))
        verdict = aggregate.verdict_of(entries, PASSED, SEEDS, KILLED, paired(results), CRITERIA["paired"], PLAN)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertEqual(verdict["invalid_seeds"], [SEEDS[5]])

    def test_two_finished_records_of_one_seed_are_refused_and_an_unfinished_one_is_not(self):
        import json
        def write(directory, name, seed, status):
            body = {"seed": seed, "status": status}
            if status in ("completed", "failed"):
                body.update({"exit_code": 0 if status == "completed" else 1, "finished_at": "2026-01-01T00:00:00+00:00", "stdout": "", "stderr": ""})
            path = Path(directory) / name
            path.write_text(json.dumps(body), encoding="utf-8")
            return path
        with tempfile.TemporaryDirectory() as directory:
            first = write(directory, "run-20260101T000000Z-seed-1.json", 1, "completed")
            reserved = write(directory, "run-20260102T000000Z-seed-1.json", 1, "started")
            self.assertEqual(aggregate.newest_per_seed([first, reserved]), {1: first})
            second = write(directory, "run-20260103T000000Z-seed-1.json", 1, "failed")
            with self.assertRaises(SystemExit) as raised:
                aggregate.newest_per_seed([first, reserved, second])
            self.assertEqual(raised.exception.code, 2)


class MutationEvidenceTests(unittest.TestCase):
    def test_without_complete_mutation_evidence_the_verdict_is_invalid_not_completed(self):
        results = VerdictTests.results(self)
        for mutations in (None, {"killed": PLAN - 1, "total": PLAN}, {"killed": 0, "total": PLAN}):
            verdict = decide(results, mutations=mutations)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, mutations)
            self.assertTrue(any("mutation" in problem for problem in verdict["problems"]))

    def test_the_total_must_be_the_size_of_the_plan(self):
        results = VerdictTests.results(self)
        verdict = decide(results, mutations={"killed": 1, "total": 1})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any("plan lists" in problem for problem in verdict["problems"]))

    def test_a_structural_falsification_stands_whatever_the_mutation_evidence(self):
        results = VerdictTests.results(self)
        put(results[SEEDS[0]], "no_broadcast_leaks", 4)
        self.assertEqual(decide(results, {SEEDS[0]: 1}, mutations=None)["verdict"], aggregate.FALSIFIED)

    def test_the_plan_names_a_killing_mutation_for_every_falsifier(self):
        self.assertGreaterEqual(PLAN, 14)
        self.assertEqual(MUTATIONS["subcommand"], aggregate.BENCHMARK)


class CodeChangedSinceTests(unittest.TestCase):
    """aggregate.code_changed_since closes the window experiment_records leaves for Python files."""

    def repo(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        def run(*args):
            subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(root)})
        run("init", "-q")
        for rel in ("research/consciousness/gw/model.py", "experiments/consciousness/C001-global-workspace/aggregate.py", "experiments/consciousness/C001-global-workspace/results/run.json", "other.txt"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("one\n")
        run("add", "-A"); run("commit", "-qm", "base")
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
        return root, sha, run

    def test_unchanged_code_and_changes_elsewhere_are_accepted(self):
        root, sha, run = self.repo()
        (root / "other.txt").write_text("two\n")
        (root / "experiments/consciousness/C001-global-workspace/results/run.json").write_text("two\n")
        run("add", "-A"); run("commit", "-qm", "elsewhere")
        self.assertEqual(aggregate.code_changed_since(sha, root), "")

    def test_a_committed_change_to_the_harness_is_refused(self):
        root, sha, run = self.repo()
        (root / "research/consciousness/gw/model.py").write_text("two\n")
        run("add", "-A"); run("commit", "-qm", "harness")
        self.assertIn("research/consciousness/gw/model.py", aggregate.code_changed_since(sha, root))

    def test_a_committed_change_to_the_experiment_files_is_refused(self):
        root, sha, run = self.repo()
        (root / "experiments/consciousness/C001-global-workspace/aggregate.py").write_text("two\n")
        run("add", "-A"); run("commit", "-qm", "aggregate")
        self.assertNotEqual(aggregate.code_changed_since(sha, root), "")

    def test_an_uncommitted_change_is_refused(self):
        root, sha, _ = self.repo()
        (root / "research/consciousness/gw/model.py").write_text("two\n")
        self.assertIn("uncommitted", aggregate.code_changed_since(sha, root))

    def test_an_unknown_commit_is_refused(self):
        root, _, _ = self.repo()
        self.assertNotEqual(aggregate.code_changed_since("0" * 40, root), "")


class ExitStatusTests(unittest.TestCase):
    def test_a_refusal_to_aggregate_is_exit_two_and_never_a_verdict(self):
        with self.assertRaises(SystemExit) as raised:
            aggregate.refuse("nothing to aggregate")
        self.assertEqual(raised.exception.code, 2)

    def test_an_unfrozen_experiment_is_refused(self):
        if "preregistration" not in CONFIG:
            with self.assertRaises(SystemExit) as raised:
                aggregate.load_frozen()
            self.assertEqual(raised.exception.code, 2)


class NotShownIsAcceptedByTheRepositoryToolsTests(unittest.TestCase):
    """NOT_SHOWN (exit 3, no recommended status) is a verdict only this experiment's aggregate.py knows. The shared tools read neither the
    verdict string nor the aggregator's exit status: scripts/run_experiment.py records seed runs only, scripts/check_research_gates.py
    and experiment_records.publish_aggregate bind run.json to metrics.json and mutations.json, and the registry status is moved by hand."""

    def not_shown(self):
        results = VerdictTests.results(self, 600 + MARGIN, spread=[-90, 90, -70, 70, -50, 50, -30, 30, -10, 10])
        verdict = decide(results)
        self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN)
        return verdict, results

    def test_publish_aggregate_records_not_shown_and_the_gate_check_passes(self):
        import json
        import experiment_records
        verdict, results = self.not_shown()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_dir = root / "experiments" / "x" / "results"
            results_dir.mkdir(parents=True)
            run = {"experiment_id": "C001", "verdict": verdict["verdict"], "recommended_status": verdict["recommended_status"],
                   "paired": paired(results), "mutation_checks": None}
            experiment_records.publish_aggregate(results_dir, {"experiment_id": "C001", "verdict": verdict["verdict"], "decision": verdict}, run)
            written = json.loads((results_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((written["verdict"], written["recommended_status"]), ("NOT_SHOWN", None))
            self.assertEqual(experiment_records.aggregate_problems(written, results_dir), [])
            self.assertEqual(experiment_records.aggregate_errors("C001", results_dir.parent, results_dir, root), [])

    def test_the_exit_status_of_each_verdict(self):
        import sys as system
        original_aggregate, original_argv = aggregate.aggregate, system.argv
        try:
            system.argv = ["aggregate.py"]
            for verdict, code in ((aggregate.COMPLETED, 0), (aggregate.FALSIFIED, 1), (aggregate.INVALID, 2), (aggregate.NOT_SHOWN, 3)):
                aggregate.aggregate = lambda report_path=None, verdict=verdict: verdict
                with self.assertRaises(SystemExit) as raised:
                    aggregate.main()
                self.assertEqual(raised.exception.code, code, verdict)
        finally:
            aggregate.aggregate, system.argv = original_aggregate, original_argv


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_the_bounds_and_every_failed_rule(self):
        results = VerdictTests.results(self, 600)
        put(results[SEEDS[0]], "no_comm_leaks", 1)
        seeds = judged(results, {SEEDS[0]: 1})
        outcome = paired(results)
        verdict = aggregate.decide(seeds, PASSED, SEEDS, KILLED, outcome, 0, 2, PLAN, 2)
        text = aggregate.report(verdict, seeds, PASSED, KILLED, outcome)
        self.assertIn("FALSIFIED", text)
        self.assertIn("no_comm_leaks <= 0", text)
        self.assertIn("attribution contrast vs no_broadcast", text)
        self.assertIn("seeds positive", text)
        self.assertIn("upper", text)
        self.assertIn("sensitivity control", text)

    def test_scope_and_limitations_state_the_limits(self):
        text = " ".join(aggregate.LIMITATIONS)
        self.assertIn("ptr-pods::workspace", text)
        self.assertIn("scripts/mutation_check.py", text)
        self.assertIn("functional property", text)
        self.assertIn("no claim about consciousness", aggregate.SCOPE)
        self.assertIn("does not test the deterministic", aggregate.SCOPE)


if __name__ == "__main__":
    unittest.main()
