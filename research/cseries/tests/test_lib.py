"""Torch-free tests of the shared library of the stage-A experiments: statistics pinned to table values and hand-computed bounds, the
rule grammar, one seed's judgement, the plateau audit and the matching counters."""

import sys
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/cseries/lib"))
import agg  # noqa: E402
import common  # noqa: E402
import stats  # noqa: E402


class StatisticsTests(unittest.TestCase):
    def test_t_quantiles_match_published_tables(self):
        # one-sided 95% and 97.5% critical values of Student's t (standard tables, six decimals)
        for p, df, expected in [(0.95, 9, 1.833113), (0.95, 13, 1.770933), (0.95, 19, 1.729133), (0.975, 13, 2.160369), (0.975, 9, 2.262157),
                                (0.975, 19, 2.093024), (1 - 0.05 / 3, 19, 2.294392)]:
            self.assertAlmostEqual(stats.t_quantile(p, df), expected, places=5, msg=f"p={p} df={df}")

    def test_quantile_string_has_six_decimals(self):
        self.assertEqual(stats.quantile_string(0.95, 13), "1.770933")

    def test_paired_bound_is_pinned_to_hand_computed_values(self):
        # differences 0.10, 0.20, 0.30 (n = 3): mean 0.2, sd 0.1, standard error 0.1 / sqrt(3); with t = 2 the half-width is 0.2 / sqrt(3)
        diffs = [Fraction(1, 10), Fraction(2, 10), Fraction(3, 10)]
        bound = stats.paired_bound(diffs, Decimal("2"))
        half = 1000 * 0.2 / 3 ** 0.5
        self.assertAlmostEqual(bound["mean_permille"], 200.0, places=6)
        self.assertAlmostEqual(bound["sd_permille"], 100.0, places=6)
        self.assertAlmostEqual(bound["lower_bound_permille"], 200.0 - half, places=6)
        self.assertAlmostEqual(bound["upper_bound_permille"], 200.0 + half, places=6)
        self.assertEqual(bound["n"], 3)

    def test_a_sign_flip_swaps_the_bounds(self):
        diffs = [Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(5, 20)]
        a = stats.paired_bound(diffs, Decimal("1.5"))
        b = stats.paired_bound([-d for d in diffs], Decimal("1.5"))
        self.assertAlmostEqual(a["lower_bound_permille"], -b["upper_bound_permille"], places=9)

    def test_one_seed_has_no_bound(self):
        with self.assertRaises(stats.PairedError):
            stats.paired_bound([Fraction(1, 10)], Decimal("2"))

    def test_median_of_even_and_odd(self):
        self.assertEqual(stats.median([Fraction(3), Fraction(1), Fraction(2)]), Fraction(2))
        self.assertEqual(stats.median([Fraction(4), Fraction(1), Fraction(2), Fraction(3)]), Fraction(5, 2))

    def test_cholesky_reproduces_the_matrix(self):
        m = [[4.0, 2.0], [2.0, 3.0]]
        low = stats.cholesky(m, jitter=0.0)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(sum(low[i][k] * low[j][k] for k in range(2)), m[i][j], places=6)


class GrammarTests(unittest.TestCase):
    def test_a_rule_is_three_tokens_with_a_known_operator(self):
        for bad in ["a >= ", "a > 1", "a >=  1", "a >= 1 2", ">= 4", "", "a=1"]:
            with self.assertRaises(agg.Unreadable, msg=bad):
                agg.evaluate({"a": 1}, bad)

    def test_literals_references_and_types(self):
        root = {"a": 2, "f": 0.5, "o": {"n": 3}, "s": "x", "t": "x", "b": True, "z": None}
        self.assertTrue(agg.evaluate(root, "o.n >= a"))
        self.assertFalse(agg.evaluate(root, "a == 3"))
        self.assertTrue(agg.evaluate(root, "a <= 2"))
        self.assertTrue(agg.evaluate(root, "s == t"))
        for bad in ["f >= 1", "f >= 0.4", "b >= 1", "z >= 1", "missing >= 1", "o >= 1", "s >= t", "s == 1", "a.b >= 1"]:
            with self.assertRaises(agg.Unreadable, msg=bad):
                agg.evaluate(root, bad)


CRITERIA = {"coverage": ["arms_trained >= 7"], "falsifiers": ["leaks <= 0"], "integrity": ["mismatch <= 0"], "guards": ["band <= 800"]}
ENV = {k: "x" for k in agg.ENVIRONMENT_KEYS}


def clean(seed=1):
    return {"benchmark": "b", "experiment": "E", "seed": seed, "scale": "pinned", "procedure": {"seed_is_declared": True, "shape_is_pinned": True},
            "controls": {"c": {"required_rule_tripped": True}}, "harness_gaps": [], "environment": dict(ENV), "arms_trained": 7, "leaks": 0,
            "mismatch": 0, "band": 500, "n": 1}


def judge(result, code=0, seed=1):
    return agg.judge_seed(result, code, CRITERIA, seed, "b", "E", ("c",), ("n",))


class JudgeSeedTests(unittest.TestCase):
    def test_a_clean_result_is_valid(self):
        entry = judge(clean())
        self.assertTrue(entry["valid"], entry["invalid"])

    def test_each_failure_is_reported_where_it_belongs(self):
        r = clean(); r["leaks"] = 1
        entry = judge(r, code=1)
        self.assertTrue(entry["valid"])
        self.assertEqual(entry["falsifiers_failed"], ["leaks <= 0"])
        r = clean(); r["arms_trained"] = 6
        self.assertEqual(judge(r, code=1)["coverage_failed"], ["arms_trained >= 7"])
        r = clean(); r["band"] = 900
        self.assertEqual(judge(r)["guards_failed"], ["band <= 800"])
        r = clean(); r["mismatch"] = 1
        entry = judge(r, code=2)
        self.assertFalse(entry["valid"])
        self.assertEqual(entry["integrity_failed"], ["mismatch <= 0"])

    def test_invalid_reasons(self):
        cases = {
            "wrong seed": lambda r: r.update(seed=2),
            "custom scale": lambda r: r.update(scale="custom"),
            "undeclared": lambda r: r["procedure"].update(seed_is_declared=False),
            "control": lambda r: r["controls"]["c"].update(required_rule_tripped=False),
            "gaps": lambda r: r.update(harness_gaps=["x"]),
            "environment": lambda r: r["environment"].pop("torch"),
            "count": lambda r: r.update(n=-1),
            "benchmark": lambda r: r.update(benchmark="z"),
        }
        for name, edit in cases.items():
            r = clean()
            edit(r)
            self.assertFalse(judge(r)["valid"], name)
        self.assertFalse(judge(clean(), code=2)["valid"])
        self.assertFalse(judge(clean(), code=1)["valid"])  # exit 1 with every rule holding
        self.assertFalse(judge(None, code=0)["valid"])

    def test_exit_status_must_agree_with_the_rules(self):
        r = clean(); r["leaks"] = 1
        self.assertFalse(judge(r, code=0)["valid"])


class ProvenanceTests(unittest.TestCase):
    def test_environment_mismatch_invalidates_the_odd_seed(self):
        entries = [{"seed": s, "valid": True, "invalid": []} for s in (1, 2, 3)]
        envs = {1: {"a": "x"}, 2: {"a": "x"}, 3: {"a": "y"}}
        mismatches = agg.invalidate_environment_mismatches(entries, envs)
        self.assertEqual(list(mismatches), [3])
        self.assertEqual([e["valid"] for e in entries], [True, True, False])

    def test_arm_fraction_refuses_malformed_counts(self):
        good = {"arms": {"a": {"m_correct": 3, "m_total": 4}}}
        self.assertEqual(agg.arm_fraction(good, "a", "m"), Fraction(3, 4))
        for bad in ({"arms": {"a": {"m_correct": 5, "m_total": 4}}}, {"arms": {"a": {"m_correct": True, "m_total": 4}}},
                    {"arms": {"a": {"m_correct": 1, "m_total": 0}}}, {"arms": {}}):
            with self.assertRaises(agg.PairedError):
                agg.arm_fraction(bad, "a", "m")


class PinFreezeTests(unittest.TestCase):
    """Step 3 of the freeze procedure writes exactly the digests, the two statuses and nothing else. The tests rebuild a pre-freeze tree from the
    committed files (placeholders back in, status planned), so they hold before and after the head freezes. `pin_freeze` needs scripts/experiment_records,
    so it is imported here and not at the top of the module (the mutation runner's throwaway copies carry no scripts directory and run other classes)."""

    @classmethod
    def setUpClass(cls):
        global pin_freeze
        import pin_freeze  # noqa: F401

    @staticmethod
    def pre_freeze_tree(scratch: str, eid: str) -> Path:
        import re
        import shutil

        root = Path(scratch)
        d = root / pin_freeze.DIRS[eid]
        d.mkdir(parents=True)
        (root / "experiments").mkdir(exist_ok=True)
        for name in ("config.toml", "criteria.toml", "experiment.toml", "PREREGISTRATION.md"):
            shutil.copy(ROOT / pin_freeze.DIRS[eid] / name, d / name)
        shutil.copy(ROOT / "experiments/preregistration.toml", root / "experiments/preregistration.toml")
        config = re.sub(r'(preregistration_text_sha256|criteria_sha256) = "[0-9a-f]{64}"', rf'\1 = "{pin_freeze.PLACEHOLDER}"', (d / "config.toml").read_text())
        (d / "config.toml").write_text(config)
        manifest = re.sub(r'\n(preregistration_sha256|preregistration_rules_sha256) = "[0-9a-f]{64}"', "", (d / "experiment.toml").read_text())
        (d / "experiment.toml").write_text(re.sub(r'status = "[a-z]+"', 'status = "planned"', manifest, count=1))
        registry = (ROOT / "experiments/registry.toml").read_text()
        registry = re.sub(rf'(id = "{eid}"\npath = "[^"]+"\nstatus = )"[a-z]+"', r'\1"planned"', registry)
        (root / "experiments/registry.toml").write_text(registry)
        return root

    def test_each_experiment_is_pinned_without_a_placeholder_and_prepared(self):
        import tempfile
        import tomllib

        for eid in pin_freeze.DIRS:
            with tempfile.TemporaryDirectory() as scratch:
                root = self.pre_freeze_tree(scratch, eid)
                texts = pin_freeze.pinned_texts(eid, root)
                d = pin_freeze.DIRS[eid] + "/"
                config = tomllib.loads(texts[d + "config.toml"])
                self.assertNotIn(pin_freeze.PLACEHOLDER, texts[d + "config.toml"])
                for key in ("preregistration_text_sha256", "criteria_sha256"):
                    self.assertRegex(config["preregistration"][key], "^[0-9a-f]{64}$", eid)
                manifest = tomllib.loads(texts[d + "experiment.toml"])
                self.assertEqual(manifest["status"], "prepared")
                for key in ("preregistration_sha256", "preregistration_rules_sha256"):
                    self.assertRegex(manifest[key], "^[0-9a-f]{64}$", eid)
                before = tomllib.loads((root / "experiments/registry.toml").read_text())["experiment"]
                after = tomllib.loads(texts["experiments/registry.toml"])["experiment"]
                self.assertEqual([(b["id"], b["status"], a["status"]) for b, a in zip(before, after) if b != a], [(eid, "planned", "prepared")])

    def test_a_tree_that_is_already_pinned_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            root = self.pre_freeze_tree(scratch, "C007")
            d = root / pin_freeze.DIRS["C007"]
            (d / "config.toml").write_text((d / "config.toml").read_text().replace(pin_freeze.PLACEHOLDER, "0" * 64, 1))
            with self.assertRaises(ValueError):
                pin_freeze.pinned_texts("C007", root)
        with tempfile.TemporaryDirectory() as scratch:
            root = self.pre_freeze_tree(scratch, "C007")
            d = root / pin_freeze.DIRS["C007"]
            (d / "experiment.toml").write_text((d / "experiment.toml").read_text().replace('status = "planned"', 'status = "prepared"'))
            with self.assertRaises(ValueError):
                pin_freeze.pinned_texts("C007", root)


class CommonTests(unittest.TestCase):
    def test_plateau_reached(self):
        # improvements of more than 5 reset the count; 3 evaluations without one stop it
        self.assertTrue(common.plateau_reached([100, 200, 201, 202, 203], 3, 5))
        self.assertFalse(common.plateau_reached([100, 200, 201, 202, 300], 3, 5))
        self.assertFalse(common.plateau_reached([100, 200, 201], 3, 5))
        self.assertFalse(common.plateau_reached([100, 106, 112, 118, 124], 3, 5))  # each step beats the best by 6 > 5
        self.assertTrue(common.plateau_reached([100, 105, 105, 105], 3, 5))  # 105 is not > 100 + 5

    def test_matching_counters(self):
        self.assertEqual(common.param_mismatch_arms({"r": 1000, "a": 1020, "b": 1021}, "r", 20), 1)
        self.assertEqual(common.param_mismatch_arms({"r": 1000, "a": 980, "b": 979}, "r", 20), 1)
        self.assertEqual(common.flop_mismatch_arms({"r": 1000, "a": 1100, "b": 1101}, "r", 100), 1)

    def test_rule_helpers(self):
        self.assertTrue(common.holds({"a": 2}, "a >= 2"))
        self.assertFalse(common.holds({"a": 2}, "a <= 1"))
        self.assertEqual(common.counter_rules(("x", "y")), ["x <= 0", "y <= 0"])
        self.assertEqual(common.permille(1, 3), 333)


if __name__ == "__main__":
    unittest.main()
