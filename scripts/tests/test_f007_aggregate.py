"""The F007 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed`,
`analyse` and `decide`, which read no file. These tests build a result in which every rule of the
criteria holds and every control passes, then break one thing at a time, so a verdict that stopped
depending on it fails here. That every counter of the criteria resolves in the real harness JSON, that
the lists equal the harness's own and that every rule can fail is held on the Rust side (`f007` tests
in bins/ptr-bench)."""

import copy
import importlib.util
import math
import os
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments" / "feedback" / "F007-vcw-loop"
sys.path.insert(0, str(ROOT / "scripts"))

# The aggregator under test: the real one, or (for the aggregator's own mutation check,
# tests/mutate_aggregator.py) a copy with one defect planted.
AGGREGATE_PATH = Path(os.environ.get("F007_AGGREGATE_PATH", HERE / "aggregate.py"))
spec = importlib.util.spec_from_file_location("f007_aggregate", AGGREGATE_PATH)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)

CRITERIA = tomllib.loads((HERE / "criteria.toml").read_text(encoding="utf-8"))
DECISION = CRITERIA["decision"]
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
MANIFEST = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
MUTATIONS = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
SEEDS = MANIFEST["seeds"]
KILLED = {"killed": len(MUTATIONS["mutation"]), "total": len(MUTATIONS["mutation"])}
N = 4000

# Success rate of each arm per world: the primary world is the one the decision reads.
RATES = {
    "hetero": {"full": 0.955, "gate": 0.88, "bon": 0.89, "no_workspace": 0.95, "no_finding": 0.70,
               "no_selfmodel": 0.95, "no_l3": 0.94, "shuffled": 0.86, "oracle": 0.965},
    "homog": {"full": 0.80, "gate": 0.80, "bon": 0.70, "no_workspace": 0.80, "no_finding": 0.80,
              "no_selfmodel": 0.80, "no_l3": 0.80, "shuffled": 0.80, "oracle": 0.80},
    "mild": {"full": 0.92, "gate": 0.86, "bon": 0.88, "no_workspace": 0.92, "no_finding": 0.66,
             "no_selfmodel": 0.92, "no_l3": 0.91, "shuffled": 0.87, "oracle": 0.93},
    "ctx": {"full": 0.96, "gate": 0.87, "bon": 0.88, "no_workspace": 0.93, "no_finding": 0.70,
            "no_selfmodel": 0.96, "no_l3": 0.95, "shuffled": 0.88, "oracle": 0.97},
}


def arm_cell(rate: float) -> dict:
    return {"requests": N, "successes": round(N * rate), "first_pass": round(N * rate * 0.55),
            "verifier_calls": 4500, "pod_calls": 5200, "elapsed_ns": 10**9}


def pair_cell(ra: float, rb: float, identical: bool = False) -> dict:
    a, b = round(N * ra), round(N * rb)
    if identical:
        return {"both": a, "only_a": 0, "only_b": 0, "neither": N - a}
    only_a = round(N * ra * (1 - rb))
    only_b = round(N * rb * (1 - ra))
    # make the margins exact: both = a - only_a must equal b - only_b
    both = a - only_a
    only_b = b - both
    return {"both": both, "only_a": only_a, "only_b": only_b, "neither": N - both - only_a - only_b}


def make_world(rates: dict, seed_shift: float = 0.0, ctx: bool = False) -> dict:
    rates = {arm: min(rate + (seed_shift if arm == "full" else 0.0), 0.999) for arm, rate in rates.items()}
    arms = {arm: arm_cell(rate) for arm, rate in rates.items()}
    names = list(rates)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = {a, b} == {"full", "no_workspace"} and not ctx
            pairs[f"{a}__{b}"] = pair_cell(rates[a], rates[b], identical=same)
    # 150 requests carried the unrepairable code first; the full loop clears 8 more of them than the
    # loop without L3 does (a gain of 5.3 points on that population).
    population = {"both": 90, "only_a": 8 if rates["full"] > rates["no_l3"] else 0, "only_b": 0,
                  "neither": 52 if rates["full"] > rates["no_l3"] else 60}
    return {"requests": N, "arms": arms, "pairs": pairs, "l3_population": population,
            "first_candidates_differ": 0}


def passing_result(seed: int, shift: float = 0.0) -> dict:
    result = {
        "benchmark": "vcw-loop",
        "experiment": "F007",
        "seed": seed,
        "scale": "pinned",
        "procedure": {"seed_is_declared": True, "seed_is_pilot": False, "shape_is_pinned": True,
                      "source_digest": "ab" * 32},
        "coverage": {"min_kind_share_permille": 250, "min_hard_events_per_code": 99, "repairs_applied": 3000,
                     "exhausted_runs": 900, "births": 2},
        "worlds": {world: make_world(rates, shift, world == "ctx") for world, rates in RATES.items()},
    }
    for rule in CRITERIA["falsifiers"] + CRITERIA["invalid_if"]:
        result[aggregate.parse_rule(rule)[0]] = 0
    return result


def all_valid(shift_by_seed=None) -> dict[int, dict]:
    shift_by_seed = shift_by_seed or {}
    return {seed: passing_result(seed, shift_by_seed.get(seed, 0.0)) for seed in SEEDS}


def judged(results: dict[int, dict], exits: dict[int, int] | None = None) -> list[dict]:
    exits = exits or {}
    return [aggregate.judge_seed(results[s], exits.get(s, 0), CRITERIA, s) for s in results]


def verdict_of(results, mutations=KILLED, exits=None, declared=SEEDS):
    seeds = judged(results, exits)
    analysis = None
    if len(results) == len(declared) and all(entry["valid"] for entry in seeds):
        analysis = aggregate.analyse(results, DECISION)
    return aggregate.decide(seeds, analysis, declared, DECISION, mutations), analysis


def violate(result: dict, rule: str) -> None:
    left, op, right = aggregate.parse_rule(rule)
    at = result
    keys = left.split(".")
    for key in keys[:-1]:
        at = at[key]
    value = int(right)
    at[keys[-1]] = value + 1 if op == "<=" else value - 1


class RuleTests(unittest.TestCase):
    def test_a_result_in_which_every_rule_holds_is_valid_and_clean(self):
        for entry in judged(all_valid()):
            self.assertTrue(entry["valid"], entry)
            self.assertEqual(
                (entry["coverage_failed"], entry["falsifiers_failed"], entry["controls_failed"]), ([], [], [])
            )

    def test_each_rule_alone_moves_the_seed_into_its_own_bucket(self):
        for key, bucket in (("coverage", "coverage_failed"), ("falsifiers", "falsifiers_failed"),
                            ("invalid_if", "controls_failed")):
            for rule in CRITERIA[key]:
                result = passing_result(17)
                violate(result, rule)
                entry = aggregate.judge_seed(result, 1, CRITERIA, 17)
                self.assertTrue(entry["valid"], (rule, entry))
                self.assertEqual(entry[bucket], [rule])

    def test_the_exit_status_must_agree_with_the_rules(self):
        self.assertIn("exited 1 but the rules all hold", " ".join(
            aggregate.judge_seed(passing_result(17), 1, CRITERIA, 17)["invalid"]))
        result = passing_result(17)
        violate(result, CRITERIA["falsifiers"][0])
        self.assertIn("exited 0 but the rules fail", " ".join(
            aggregate.judge_seed(result, 0, CRITERIA, 17)["invalid"]))
        for code in (2, -9, None, True):
            self.assertFalse(aggregate.judge_seed(passing_result(17), code, CRITERIA, 17)["valid"])

    def test_a_result_that_is_not_the_frozen_run_is_invalid(self):
        for edit in (
            lambda r: r.update(benchmark="other"),
            lambda r: r.update(experiment="F001"),
            lambda r: r.update(seed=29),
            lambda r: r.update(scale="custom"),
            lambda r: r["procedure"].update(seed_is_declared=False),
            lambda r: r["procedure"].update(seed_is_pilot=True),
            lambda r: r["procedure"].update(shape_is_pinned=False),
        ):
            result = passing_result(17)
            edit(result)
            self.assertFalse(aggregate.judge_seed(result, 0, CRITERIA, 17)["valid"])
        self.assertFalse(aggregate.judge_seed(None, 0, CRITERIA, 17)["valid"])

    def test_a_rule_that_cannot_be_read_is_invalid_not_failed(self):
        for edit in (lambda r: r.pop("false_accepts"), lambda r: r.update(false_accepts=None),
                     lambda r: r.update(false_accepts=True), lambda r: r.update(false_accepts=0.5)):
            result = passing_result(17)
            edit(result)
            entry = aggregate.judge_seed(result, 0, CRITERIA, 17)
            self.assertFalse(entry["valid"], entry)
            self.assertEqual(entry["falsifiers_failed"], [])


class IntervalTests(unittest.TestCase):
    def test_the_interval_is_the_t_interval_over_seeds(self):
        entry = aggregate.interval([0.10, 0.12, 0.08, 0.11, 0.09])
        self.assertAlmostEqual(entry["mean"], 0.10)
        sd = math.sqrt(sum((d - 0.10) ** 2 for d in entry["per_seed"]) / 4)
        half = 2.131847 * sd / math.sqrt(5)
        self.assertAlmostEqual(entry["lcb"], 0.10 - half)
        self.assertAlmostEqual(entry["ucb"], 0.10 + half)

    def test_a_pair_difference_has_the_sign_of_the_order_it_is_asked_in(self):
        result = passing_result(17)
        forward = aggregate.pair_difference(result, "hetero", "full", "gate")
        backward = aggregate.pair_difference(result, "hetero", "gate", "full")
        self.assertAlmostEqual(forward[0], -backward[0])
        self.assertAlmostEqual(forward[0], 0.075, places=3)
        self.assertAlmostEqual(forward[1], backward[1])
        self.assertGreater(forward[1], 0)

    def test_seeds_without_a_quantile_are_refused(self):
        with self.assertRaises(ValueError):
            aggregate.t_quantile(99)


class DecisionTests(unittest.TestCase):
    def test_a_clear_gain_over_both_baselines_with_every_control_passing_is_completed(self):
        verdict, analysis = verdict_of(all_valid({17: 0.004, 29: -0.004, 43: 0.002, 71: -0.002, 101: 0.0}))
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED, verdict)
        self.assertEqual(verdict["recommended_status"], "completed")
        self.assertTrue(all(entry["passes"] for entry in analysis["controls"].values()))
        self.assertEqual(analysis["ablations"]["no_workspace"]["label"], "non_discriminating")
        self.assertEqual(analysis["ablations"]["no_finding"]["label"], "helps")

    def test_a_gain_below_the_margin_against_one_baseline_is_falsified(self):
        results = all_valid()
        for seed in SEEDS:
            # full only 1 point above best-of-N
            for world in ("hetero",):
                pairs = results[seed]["worlds"][world]["pairs"]
                pairs["full__bon"] = pair_cell(0.87, 0.86)
        verdict, analysis = verdict_of(results)
        self.assertEqual(analysis["baselines"]["bon"]["verdict"], "below")
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)
        self.assertIn("bon", " ".join(verdict["reasons"]))

    def test_an_interval_that_straddles_the_margin_is_not_shown(self):
        results = all_valid()
        noisy = [0.035, 0.010, 0.060, 0.000, 0.045]
        for seed, gain in zip(SEEDS, noisy):
            results[seed]["worlds"]["hetero"]["pairs"]["full__gate"] = pair_cell(0.85 + gain, 0.85)
        verdict, analysis = verdict_of(results)
        self.assertEqual(analysis["baselines"]["gate"]["verdict"], "straddles")
        self.assertEqual(verdict["verdict"], aggregate.NOT_SHOWN)
        self.assertIsNone(verdict["recommended_status"])

    def test_a_tripped_falsifier_is_falsified_whatever_the_statistics_say(self):
        results = all_valid()
        for rule in CRITERIA["falsifiers"]:
            broken = copy.deepcopy(results)
            violate(broken[43], rule)
            verdict, _ = verdict_of(broken, exits={43: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["falsified_seeds"], [43])

    def test_a_failed_control_makes_the_verdict_invalid_even_with_a_big_gain(self):
        for rule in CRITERIA["invalid_if"]:
            broken = all_valid()
            violate(broken[71], rule)
            verdict, _ = verdict_of(broken, exits={71: 1})
            self.assertEqual(verdict["verdict"], aggregate.INVALID, rule)

    def test_unmet_coverage_makes_the_verdict_invalid(self):
        broken = all_valid()
        violate(broken[29], CRITERIA["coverage"][-1])
        verdict, _ = verdict_of(broken, exits={29: 1})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_the_sensitivity_control_across_seeds(self):
        results = all_valid()
        for seed in SEEDS:  # the wrong map loses by only 2 points: the experiment cannot see a bad map
            results[seed]["worlds"]["hetero"]["pairs"]["full__shuffled"] = pair_cell(0.95, 0.93)
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["shuffled_loses"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertIn("shuffled_loses", verdict["controls_failed"])

    def test_the_null_world_must_tie(self):
        results = all_valid()
        for seed in SEEDS:  # a built-in 5 point advantage in the world where all Pods are equal
            results[seed]["worlds"]["homog"]["pairs"]["gate__no_l3"] = pair_cell(0.75, 0.80)
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["null_tie_no_l3_vs_gate"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_an_untrained_gate_is_a_straw_man_and_invalidates(self):
        results = all_valid()
        for seed in SEEDS:
            results[seed]["worlds"]["hetero"]["pairs"]["gate__no_finding"] = pair_cell(0.71, 0.70)
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["gate_is_trained"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_a_workspace_ablation_that_changes_outcomes_where_no_pod_reads_it_is_invalid(self):
        results = all_valid()
        results[17]["worlds"]["hetero"]["pairs"]["full__no_workspace"] = pair_cell(0.95, 0.949)
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["workspace_withheld_changes_nothing"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_the_workspace_secondary_is_judged_in_the_declared_mechanism_world_and_never_gates(self):
        verdict, analysis = verdict_of(all_valid())
        self.assertEqual(analysis["secondary"]["workspace"]["verdict"], "SUPPORTED")
        results = all_valid()
        for seed in SEEDS:  # the broadcast changes nothing even where a Pod reads it
            results[seed]["worlds"]["ctx"]["pairs"]["full__no_workspace"] = pair_cell(0.96, 0.96, identical=True)
        verdict, analysis = verdict_of(results)
        self.assertEqual(analysis["secondary"]["workspace"]["verdict"], "REFUTED")
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED, "a secondary never gates the primary verdict")

    def test_the_l3_secondary_is_read_on_the_population_it_acts_on(self):
        results = all_valid()
        for seed in SEEDS:  # a 20 point gain on the 150 requests that met the unrepairable code
            results[seed]["worlds"]["hetero"]["l3_population"] = {"both": 90, "only_a": 30, "only_b": 0, "neither": 30}
        verdict, analysis = verdict_of(results)
        self.assertEqual(analysis["secondary"]["l3"]["verdict"], "SUPPORTED")
        self.assertEqual(analysis["secondary"]["l3"]["requests_per_seed"], [150] * 5)
        for seed in SEEDS:  # L3 removed: no difference at all
            results[seed]["worlds"]["hetero"]["l3_population"] = {"both": 90, "only_a": 0, "only_b": 0, "neither": 60}
        verdict, analysis = verdict_of(results)
        self.assertEqual(analysis["secondary"]["l3"]["verdict"], "REFUTED")
        self.assertEqual(analysis["secondary"]["self_model"]["verdict"], "UNTESTED")
        self.assertEqual(verdict["verdict"], aggregate.COMPLETED)

    def test_the_three_contrasts_the_primary_world_cannot_discriminate_are_never_called_no_effect(self):
        verdict, analysis = verdict_of(all_valid())
        for arm in ("no_workspace", "no_selfmodel", "no_l3"):
            self.assertEqual(analysis["ablations"][arm]["label"], aggregate.NON_DISCRIMINATING)
        text = aggregate.report(verdict, judged(all_valid()), analysis, KILLED)
        self.assertNotIn("no effect shown | \n", text)
        self.assertIn("non-discriminating in the primary world", text)
        self.assertEqual(aggregate.label("no_finding", {"lcb": 0.2, "ucb": 0.3}, 0.01), "helps")
        self.assertEqual(aggregate.label("no_finding", {"lcb": -0.3, "ucb": -0.2}, 0.01), "hurts")
        self.assertEqual(aggregate.label("no_finding", {"lcb": -0.1, "ucb": 0.2}, 0.01), "no effect shown")

    def test_a_crippled_best_of_n_is_invalid(self):
        results = all_valid()
        for seed in SEEDS:  # best-of-N barely beats its own first call
            results[seed]["worlds"]["hetero"]["arms"]["bon"]["first_pass"] = round(N * 0.89 * 0.97)
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["bon_beats_single_shot"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_an_oracle_below_the_full_loop_is_invalid(self):
        results = all_valid()
        for seed in SEEDS:
            results[seed]["worlds"]["hetero"]["pairs"]["full__oracle"] = pair_cell(0.955, 0.90)
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["oracle_is_an_upper_reference"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_a_harness_far_from_the_planning_model_or_at_a_ceiling_is_invalid(self):
        results = all_valid()
        for seed in SEEDS:  # the gate sits 3 points above the model's prediction
            results[seed]["worlds"]["hetero"]["arms"]["gate"]["successes"] = round(N * 0.91)
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["harness_agrees_with_planning_model"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        results = all_valid()
        for seed in SEEDS:  # best-of-N at the ceiling
            results[seed]["worlds"]["hetero"]["arms"]["bon"]["successes"] = N - 4
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["no_arm_at_ceiling_or_floor"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_results_of_different_harness_sources_are_invalid(self):
        results = all_valid()
        results[43]["procedure"]["source_digest"] = "cd" * 32
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["one_harness_for_every_seed"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        for bad in (None, "xyz", "AB" * 32):
            broken = passing_result(17)
            broken["procedure"]["source_digest"] = bad
            self.assertFalse(aggregate.judge_seed(broken, 0, CRITERIA, 17)["valid"])

    def test_a_first_draft_that_differs_between_arms_is_invalid(self):
        results = all_valid()
        verdict, analysis = verdict_of(results)
        self.assertTrue(analysis["controls"]["first_draft_is_shared"]["passes"])
        results[29]["worlds"]["homog"]["first_candidates_differ"] = 1
        verdict, analysis = verdict_of(results)
        self.assertFalse(analysis["controls"]["first_draft_is_shared"]["passes"])
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_the_printed_secondary_lines_carry_their_caveats(self):
        verdict, analysis = verdict_of(all_valid())
        text = aggregate.report(verdict, judged(all_valid()), analysis, KILLED)
        self.assertIn("low power", text)
        self.assertIn("harness-defined reader", text)

    def test_a_missing_pair_table_is_invalid_not_a_crash(self):
        results = all_valid()
        del results[43]["worlds"]["hetero"]["pairs"]["full__gate"]
        with self.assertRaises(aggregate.Unreadable):
            aggregate.analyse(results, DECISION)
        seeds = judged(results)
        verdict = aggregate.decide(seeds, None, SEEDS, DECISION, KILLED, "the harness results lack what the analysis reads")
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_invalid_takes_precedence_over_falsified(self):
        results = all_valid()
        violate(results[43], CRITERIA["falsifiers"][0])
        verdict, _ = verdict_of(results, mutations=None, exits={43: 1})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertIn("also tripped", " ".join(verdict["reasons"]))
        verdict, _ = verdict_of(results, exits={43: 1})
        self.assertEqual(verdict["verdict"], aggregate.FALSIFIED)

    def test_the_wording_of_a_completed_verdict_names_the_loop_not_the_routing(self):
        verdict, _ = verdict_of(all_valid())
        text = " ".join(verdict["reasons"])
        self.assertIn("beats best-of-N at equal verifier budget", text)
        self.assertIn("shuffled", text)
        self.assertNotIn("finding routing beats best-of-N", text)

    def test_the_frozen_table_and_criteria_agree_on_margins_and_confidence(self):
        table = {"primary_margin_permille": 30, "equivalence_band_permille": 15,
                 "shuffled_control_margin_permille": 50, "confidence_one_sided_permille": 950}
        self.assertEqual(aggregate.preregistration_disagreements(table, DECISION), [])
        for key in table:
            wrong = dict(table, **{key: table[key] + 1})
            self.assertEqual(len(aggregate.preregistration_disagreements(wrong, DECISION)), 1, key)
        placeholders = {key: "must-be-signed-x" for key in table}
        self.assertEqual(aggregate.preregistration_disagreements(placeholders, DECISION), [])
        committed = CONFIG["preregistration"]
        self.assertEqual(aggregate.preregistration_disagreements(committed, DECISION), [])

    def test_the_planning_model_numbers_in_the_criteria_are_the_ones_power_py_derives(self):
        spec = importlib.util.spec_from_file_location("f007_power", HERE / "power.py")
        power = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(power)
        derived = power.derive(n=20000, seed=11)
        model = DECISION["model"]["predicted_permille"]
        for arm, value in model.items():
            self.assertAlmostEqual(derived["predicted_rate_permille"][arm], value, delta=3, msg=arm)
        self.assertAlmostEqual(derived["baseline_min_gain_permille"], DECISION["baseline_min_gain_permille"], delta=4)
        self.assertAlmostEqual(derived["bon_min_gain_permille"], DECISION["bon_min_gain_permille"], delta=4)
        self.assertAlmostEqual(derived["workspace_margin_permille"], DECISION["secondary"]["workspace"]["margin_permille"], delta=3)
        self.assertAlmostEqual(derived["l3_margin_permille"], DECISION["secondary"]["l3"]["margin_permille"], delta=5)

    def test_missing_seeds_and_incomplete_mutation_evidence_are_invalid(self):
        results = all_valid()
        partial = {seed: results[seed] for seed in SEEDS[:-1]}
        verdict, _ = verdict_of(partial)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        for evidence in (None, {"killed": KILLED["total"] - 1, "total": KILLED["total"]}):
            verdict, _ = verdict_of(all_valid(), mutations=evidence)
            self.assertEqual(verdict["verdict"], aggregate.INVALID)
        verdict, _ = verdict_of(all_valid(), exits={17: 7})
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_the_gain_without_l3_and_the_descriptive_worlds_are_reported_not_decided(self):
        verdict, analysis = verdict_of(all_valid())
        self.assertIn("gate", analysis["robust_without_l3"])
        self.assertEqual(set(analysis["descriptive_worlds"]), {"mild", "ctx"})
        text = aggregate.report(verdict, judged(all_valid()), analysis, KILLED)
        self.assertIn("Descriptive world mild", text)
        self.assertIn("Limitations:", text)

    def test_the_decision_table_is_the_one_the_criteria_name(self):
        self.assertEqual(DECISION["baselines"], ["gate", "bon"])
        self.assertEqual(DECISION["primary_world"], "hetero")
        self.assertGreater(DECISION["shuffled_min_loss_permille"], DECISION["margin_permille"])
        self.assertLess(DECISION["equivalence_band_permille"], DECISION["margin_permille"])


if __name__ == "__main__":
    unittest.main()
