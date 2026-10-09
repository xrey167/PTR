"""The C001 aggregator's verdicts, on harness results built from criteria.toml to show each one.

`aggregate.py` reads run records and writes the aggregate; what it concludes is `judge_seed`,
`paired_comparisons` and `decide`, which read no file. These tests build a result that holds every
rule of the criteria and break one rule at a time, so a verdict that stopped depending on it fails
here. The paired bounds are pinned to hand-computed values, so a flipped sign, a wrong quantile or a
wrong divisor fails. The outcomes are four: structural FALSIFIED, INVALID (saturated, floored,
insensitive, missing evidence), refuted FALSIFIED (the UPPER bound below the margin), NOT_SHOWN
(straddling) and COMPLETED (every LOWER bound at the margin). That the harness's rules equal
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
ARMS = ("workspace", "all_to_all", "bw_all_to_all", "no_broadcast", "no_comm", "competition_off", "capacity_off", "crippled_all_to_all")
TOTAL = 1000  # evaluation instances per synthetic arm, so counts are per mille
MARGIN = CRITERIA["paired"]["margin_permille"]
BASE = {"workspace": 700, "all_to_all": 600, "bw_all_to_all": 590, "no_broadcast": 400, "no_comm": 330,
        "competition_off": 650, "capacity_off": 640, "crippled_all_to_all": 420}


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
        "harness_gaps": [], "workspace_trained": 1,
        "arms": {arm: arm_record(ood[arm]) for arm in ARMS},
    }
    for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"]:
        left, _, right = aggregate.parse_rule(rule)
        put(result, left, int(right))
    result["comparator_primary_permille"] = max(ood["all_to_all"], ood["bw_all_to_all"])
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
    untrained = sum(1 for r in results.values() if r and r.get("workspace_trained") == 0)
    return aggregate.decide(judged(results, exits), PASSED, SEEDS, mutations, outcome, untrained,
                            CRITERIA["paired"]["max_untrained_workspace_seeds"], plan)


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
        for key in ("coverage", "guards", "falsifiers"):
            rules = CRITERIA[key]
            self.assertTrue(rules)
            self.assertEqual(len(rules), len(set(rules)), key)
            for rule in rules:
                aggregate.parse_rule(rule)
        self.assertEqual(len(CRITERIA["falsifiers"]), 10)
        self.assertNotIn("static_evidence", CRITERIA)

    def test_the_paired_table_agrees_with_the_pinned_configuration(self):
        pinned, table = CONFIG["pinned_thresholds"], CRITERIA["paired"]
        self.assertEqual(table["margin_permille"], pinned["paired_margin_permille"])
        self.assertEqual(table["sensitivity"]["margin_permille"], pinned["sensitivity_margin_permille"])
        self.assertEqual(table["t_one_sided_95_df9"], pinned["paired_t_one_sided_95_df9"])
        self.assertEqual(table["max_untrained_workspace_seeds"], pinned["max_untrained_workspace_seeds"])
        self.assertEqual(table["seeds_required"], len(pinned["seeds"]))
        self.assertEqual(pinned["seeds"], SEEDS)
        self.assertEqual(CRITERIA["seed_aggregation"], pinned["seed_aggregation"])
        self.assertTrue(set(table["against"]) < set(ARMS) and table["arm"] == "workspace")
        self.assertEqual(table["metric"], "ood16")

    def test_confirmatory_seeds_are_fresh_and_disjoint_from_the_pilots(self):
        pinned = CONFIG["pinned_thresholds"]
        self.assertEqual(len(SEEDS), 10)
        self.assertEqual(set(SEEDS) & {17, 29, 43, 71, 101}, set())
        self.assertEqual(set(SEEDS) & set(pinned["pilot_seeds"]), set())

    def test_the_t_quantile_is_the_one_sided_95_percent_value_for_nine_degrees_of_freedom(self):
        # Student t, df = 9: P(T <= t) = 0.95 at t = 1.833113 (tables: 1.833).
        self.assertAlmostEqual(float(CRITERIA["paired"]["t_one_sided_95_df9"]), 1.8331129, places=5)

    def test_the_reference_result_holds_every_rule_and_guard(self):
        result = passing_result(3001)
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["guards"]:
            self.assertTrue(aggregate.evaluate(result, rule), rule)

    def test_every_rule_can_fail(self):
        for rule in CRITERIA["coverage"] + CRITERIA["falsifiers"] + CRITERIA["guards"]:
            left, op, right = aggregate.parse_rule(rule)
            result = passing_result(3001)
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
            # Until the freeze renames it, the preregistration text is DRAFT_PREREGISTRATION.md.
            self.assertTrue(path.is_file() or path.with_name("DRAFT_" + path.name).is_file(), key)


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
    def results(self, ws, a2a=600, bw=590, nb=400, spread=None, **extra):
        spread = spread or [0] * len(SEEDS)
        out = {}
        for seed, jitter in zip(SEEDS, spread):
            out[seed] = passing_result(seed, {"workspace": ws + jitter, "all_to_all": a2a, "bw_all_to_all": bw, "no_broadcast": nb, **extra})
        return out

    def test_support_needs_every_lower_bound_at_the_margin_inclusive(self):
        self.assertTrue(paired(self.results(600 + MARGIN, 600, 600, 400))["supported"])
        self.assertFalse(paired(self.results(600 + MARGIN - 1, 600, 600, 400))["supported"])

    def test_beating_only_some_comparators_is_not_support(self):
        outcome = paired(self.results(600 + MARGIN, 600, 600 + MARGIN, 400))
        self.assertFalse(outcome["supported"])
        self.assertEqual([item["supports"] for item in outcome["judged"]], [True, False, True])

    def test_refutation_needs_the_upper_bound_below_the_margin(self):
        equal = paired(self.results(600, 600, 600, 400))
        self.assertEqual(equal["refuted_against"], ["all_to_all", "bw_all_to_all"])
        just_below = paired(self.results(600 + MARGIN - 1, 600, 600, 400))
        self.assertEqual(just_below["refuted_against"], ["all_to_all", "bw_all_to_all"])
        at_margin = paired(self.results(600 + MARGIN, 600, 600, 400))
        self.assertEqual(at_margin["refuted_against"], [])

    def test_a_noisy_difference_around_the_margin_is_neither_supported_nor_refuted(self):
        outcome = paired(self.results(600 + MARGIN, 600, 600, 400, spread=[-90, 90, -70, 70, -50, 50, -30, 30, -10, 10]))
        self.assertFalse(outcome["supported"])
        self.assertEqual(outcome["refuted_against"], [])

    def test_the_sensitivity_control_must_resolve_its_margin(self):
        self.assertTrue(paired(self.results(700, crippled_all_to_all=600 - CRITERIA["paired"]["sensitivity"]["margin_permille"]))["sensitivity"]["resolves"])
        self.assertFalse(paired(self.results(700, crippled_all_to_all=600 - CRITERIA["paired"]["sensitivity"]["margin_permille"] + 1))["sensitivity"]["resolves"])

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
        self.assertTrue({"competition_off", "capacity_off"} <= against)
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
            put(results[3004], "comparator_primary_permille", comparator)
            verdict = decide(results)
            self.assertEqual(verdict["verdict"], aggregate.INVALID, comparator)
            self.assertEqual(verdict["uninformative_seeds"], [3004])
            self.assertIsNone(verdict["recommended_status"])

    def test_an_insensitive_pipeline_is_invalid(self):
        verdict = decide(self.results(600, crippled_all_to_all=600))
        self.assertEqual(verdict["verdict"], aggregate.INVALID)
        self.assertTrue(any("sensitivity" in p for p in verdict["problems"]))

    def test_a_workspace_that_does_not_train_is_invalid_beyond_the_allowance(self):
        limit = CRITERIA["paired"]["max_untrained_workspace_seeds"]
        results = self.results(600)
        for seed in SEEDS[:limit]:
            results[seed]["workspace_trained"] = 0
        self.assertEqual(decide(results)["verdict"], aggregate.FALSIFIED)
        results[SEEDS[limit]]["workspace_trained"] = 0
        self.assertEqual(decide(results)["verdict"], aggregate.INVALID)

    def test_each_falsifier_on_one_seed_is_a_structural_falsification(self):
        for rule in CRITERIA["falsifiers"]:
            left, _, right = aggregate.parse_rule(rule)
            results = self.results(700)
            put(results[3003], left, int(right) + 1)
            verdict = decide(results, {3003: 1})
            self.assertEqual(verdict["verdict"], aggregate.FALSIFIED, rule)
            self.assertEqual(verdict["kind"], "structural", rule)
            self.assertEqual(verdict["falsified_seeds"], [3003], rule)

    def test_an_unmet_coverage_rule_is_a_structural_falsification(self):
        for rule in CRITERIA["coverage"]:
            left, _, right = aggregate.parse_rule(rule)
            results = self.results(700)
            put(results[3010], left, int(right) - 1)
            verdict = decide(results, {3010: 1})
            self.assertEqual((verdict["verdict"], verdict["kind"]), (aggregate.FALSIFIED, "structural"), rule)
            self.assertEqual(verdict["vacuous_seeds"], [3010], rule)

    def test_a_structural_defect_beats_an_uninformative_seed(self):
        results = self.results(700)
        put(results[3001], "n_leaks", 1)
        put(results[3002], "comparator_primary_permille", 999)
        self.assertEqual(decide(results, {3001: 1})["verdict"], aggregate.FALSIFIED)

    def test_a_missing_seed_is_invalid(self):
        results = self.results(700)
        del results[3007]
        verdict = aggregate.decide(judged(results), PASSED, SEEDS, KILLED, None, 0, 2, PLAN)
        self.assertEqual(verdict["verdict"], aggregate.INVALID)

    def test_a_counter_that_is_missing_null_or_mistyped_is_invalid_not_failed(self):
        for damage in (
            lambda r: r.pop("n_leaks"), lambda r: r.update(n_leaks=None), lambda r: r.update(n_leaks=0.0),
            lambda r: r.update(n_leaks=False), lambda r: r.update(n_leaks="0"), lambda r: r.pop("workspace_influence_hits"),
            lambda r: r.pop("comparator_primary_permille"), lambda r: r.update(workspace_trained=2), lambda r: r.pop("workspace_trained"),
        ):
            results = self.results(700)
            damage(results[3005])
            self.assertFalse(judged(results)[SEEDS.index(3005)]["valid"])
            self.assertEqual(aggregate.decide(judged(results), PASSED, SEEDS, KILLED, None, 0, 2, PLAN)["verdict"], aggregate.INVALID)

    def test_a_result_that_is_not_the_run_is_invalid(self):
        damages = [
            lambda r: r.update(scale="custom"), lambda r: r.update(seed=18), lambda r: r.update(benchmark="other"),
            lambda r: r.update(experiment="S002"), lambda r: r.update(harness_gaps=["x"]),
            lambda r: r["procedure"].update(seed_is_declared=False), lambda r: r["procedure"].update(shape_is_pinned=False),
            lambda r: r["controls"].pop("bottleneck"), lambda r: r.pop("controls"),
        ] + [(lambda name: lambda r: r["controls"][name].update(required_rule_tripped=False))(name) for name in aggregate.CONTROLS]
        for damage in damages:
            results = self.results(700)
            damage(results[3001])
            self.assertEqual(decide(results, use_paired=False)["verdict"], aggregate.INVALID)

    def test_an_exit_status_that_disagrees_with_the_rules_is_invalid(self):
        self.assertEqual(decide(self.results(700), {3001: 1})["verdict"], aggregate.INVALID)
        results = self.results(700)
        put(results[3001], "n_leaks", 1)
        self.assertEqual(decide(results, {3001: 0})["verdict"], aggregate.INVALID)
        self.assertEqual(decide(self.results(700), {3001: 2})["verdict"], aggregate.INVALID)
        self.assertEqual(decide(self.results(700), {3001: None})["verdict"], aggregate.INVALID)

    def test_a_guard_failure_does_not_change_the_harness_exit_status(self):
        results = self.results(700)
        put(results[3001], "comparator_primary_permille", 999)
        self.assertTrue(judged(results)[0]["valid"])
        self.assertEqual(judged(results)[0]["guards_failed"], ["comparator_primary_permille <= 900"])

    def test_a_seed_without_a_result_is_invalid(self):
        results = self.results(700)
        results[3003] = None
        self.assertEqual(aggregate.decide(judged(results, {3003: 2}), PASSED, SEEDS, KILLED, None, 0, 2, PLAN)["verdict"], aggregate.INVALID)

    def test_judgements_do_not_alias_their_inputs(self):
        results = self.results(700)
        before = copy.deepcopy(results)
        judged(results)
        paired(results)
        self.assertEqual(results, before)


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
        put(results[3001], "no_broadcast_leaks", 4)
        self.assertEqual(decide(results, {3001: 1}, mutations=None)["verdict"], aggregate.FALSIFIED)

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


class ReportTests(unittest.TestCase):
    def test_the_report_names_the_verdict_the_bounds_and_every_failed_rule(self):
        results = VerdictTests.results(self, 600)
        put(results[3001], "n_leaks", 1)
        seeds = judged(results, {3001: 1})
        outcome = paired(results)
        verdict = aggregate.decide(seeds, PASSED, SEEDS, KILLED, outcome, 0, 2, PLAN)
        text = aggregate.report(verdict, seeds, PASSED, KILLED, outcome)
        self.assertIn("FALSIFIED", text)
        self.assertIn("n_leaks <= 0", text)
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
