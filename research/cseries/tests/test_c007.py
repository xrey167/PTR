"""Tests of C007's rules, world, models, metrics and harness. The rule tests are torch-free and pin the harness's lists to criteria.toml and to the
mutation plan; the others need torch and skip without it (the repository's CI python has none; the locked environment runs them)."""

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/consciousness/C007-metacognitive-self-model-cpu"
sys.path.insert(0, str(ROOT / "research/cseries/lib"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rules = load("c007_rules", ROOT / "research/cseries/c002a/c007_rules.py")
CONFIG = tomllib.loads((EXP / "config.toml").read_text(encoding="utf-8"))
CRITERIA = tomllib.loads((EXP / "criteria.toml").read_text(encoding="utf-8"))
PLAN = tomllib.loads((EXP / "tests/mutations.toml").read_text(encoding="utf-8"))
PINNED = CONFIG["pinned_thresholds"]

try:
    import torch  # noqa: F401

    sys.path.insert(0, str(ROOT / "research/cseries/c002a"))
    task = load("c007_task", ROOT / "research/cseries/c002a/c007_task.py")
    model = load("c007_model", ROOT / "research/cseries/c002a/c007_model.py")
    harness = load("c007_harness", ROOT / "research/cseries/c002a/harness.py")
    harness._load_torch()
except ModuleNotFoundError:  # no torch here
    torch = None


class RulesAgreeWithCriteria(unittest.TestCase):
    def test_coverage_rules_equal_the_criteria_in_order(self):
        self.assertEqual(rules.coverage_rules(PINNED), CRITERIA["coverage"])

    def test_guard_rules_equal_the_criteria_in_order(self):
        self.assertEqual(rules.guard_rules(PINNED), CRITERIA["guards"])

    def test_falsifier_rules_equal_the_criteria_in_order(self):
        self.assertEqual(rules.falsifier_rules(), CRITERIA["falsifiers"])

    def test_integrity_rules_equal_the_criteria_in_order(self):
        self.assertEqual(rules.integrity_rules(), CRITERIA["integrity"])
        self.assertEqual(set(rules.STRUCTURAL_COUNTERS) & set(rules.INTEGRITY_COUNTERS), set())

    def test_the_paired_table_agrees_with_the_pinned_configuration(self):
        table = CRITERIA["paired"]
        self.assertEqual(table["margin_permille"], PINNED["paired_margin_permille"])
        self.assertEqual(table["sensitivity_gross"]["margin_permille"], PINNED["sensitivity_margin_permille"])
        self.assertEqual(table["sensitivity_graded"]["min_lower_permille"], PINNED["graded_min_lower_permille"])
        self.assertEqual(table["null_world"]["margin_permille"], PINNED["null_world_margin_permille"])
        self.assertEqual(table["removal_margin_permille"], PINNED["paired_margin_permille"])
        self.assertEqual(table["t_support"], PINNED["paired_t_support"])
        self.assertEqual(table["t_refute"], PINNED["paired_t_refute"])
        self.assertEqual(table["seeds_required"], len(PINNED["seeds"]))
        self.assertEqual(table["max_unconverged_seeds"], PINNED["max_unconverged_seeds"])
        self.assertEqual(table["against"], list(rules.COMPARATORS))
        self.assertEqual(table["arm"], "self_model")

    def test_the_quantiles_are_those_of_the_seed_count(self):
        import stats

        n = len(PINNED["seeds"])
        self.assertEqual(PINNED["paired_t_support"], stats.quantile_string(0.95, n - 1))
        self.assertEqual(PINNED["paired_t_refute"], stats.quantile_string(1 - 0.05 / len(rules.COMPARATORS), n - 1))

    def test_pilot_and_declared_seeds_are_disjoint_and_pilots_start_at_1001(self):
        self.assertFalse(set(PINNED["seeds"]) & set(PINNED["pilot_seeds"]))
        self.assertGreaterEqual(min(PINNED["pilot_seeds"]), 1001)
        self.assertLessEqual(len(PINNED["seeds"]), 20)
        self.assertEqual(len(set(PINNED["seeds"])), len(PINNED["seeds"]))

    def test_the_preregistration_table_restates_the_pinned_values(self):
        table = CONFIG["preregistration"]
        for key, value in table.items():
            if key.endswith("_sha256") or key in ("preregistration_text", "criteria", "harness"):
                continue
            scaled = key[: -len("_milli")] if key.endswith("_milli") and key[: -len("_milli")] in PINNED else key[: -len("_micro")] if key.endswith("_micro") else key
            self.assertIn(scaled, PINNED, key)
            expected = PINNED[scaled]
            if scaled != key:
                expected = round(expected * (1000 if key.endswith("_milli") else 1_000_000))
            self.assertEqual(value, expected, key)

    def test_the_required_keys_of_the_list_are_the_preregistered_table_with_the_types_of_its_values(self):
        required = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"]["C007"]["required"]
        table = CONFIG["preregistration"]
        self.assertEqual(sorted(required), sorted(k for k in table if not k.endswith("_sha256")))
        kinds = {"int": int, "str": str, "bool": bool, "int-list": list, "str-list": list, "file": str}
        for key, kind in required.items():
            self.assertIsInstance(table[key], kinds[kind], key)
            if kind.endswith("-list"):
                self.assertTrue(table[key], key)
        for key in ("preregistration_text", "criteria"):
            self.assertEqual(required[key], "file")
            self.assertTrue((ROOT / table[key]).is_file(), key)

    def test_the_judged_heads_and_controls_are_what_the_aggregator_reads(self):
        self.assertEqual(rules.JUDGED_ARMS, ("self_model", "probe_matched"))
        self.assertEqual(len(rules.CONTROLS), 10)


class PlanCoversTheRules(unittest.TestCase):
    def expected(self):
        return {name for m in PLAN["mutation"] for name in m["expect"]}

    def test_every_hard_counter_is_expected_by_a_mutation(self):
        self.assertEqual(sorted(set(rules.HARD_COUNTERS) - self.expected()), [])

    def test_every_coverage_counter_is_expected_by_a_mutation(self):
        counters = {rule.split(" ")[0] for rule in CRITERIA["coverage"]}
        self.assertEqual(sorted(counters - self.expected()), [])

    def test_every_negative_control_is_expected_by_a_mutation(self):
        for control in rules.CONTROLS:
            self.assertIn("control:" + control, self.expected())

    def test_every_expected_name_is_declared_in_the_plan(self):
        self.assertEqual(sorted(self.expected() - set(PLAN["hard_counters"])), [])

    def test_every_mutation_is_killed_by_the_harness_or_by_named_unit_tests(self):
        sources = {p.stem: p.read_text(encoding="utf-8") for p in (ROOT / "research/cseries/tests").glob("test_*.py")}
        for mutation in PLAN["mutation"]:
            if not mutation.get("unit_tests"):
                self.assertTrue(all(not e.startswith("unit:") for e in mutation["expect"]), mutation["name"])
                continue
            self.assertTrue(any(e.startswith("unit:") for e in mutation["expect"]), mutation["name"])
            for test in mutation["unit_tests"]:
                module, *path = test.split(".")
                self.assertIn(f"class {path[0]}", sources[module], test)
                if len(path) > 1:
                    self.assertIn(f"def {path[1]}", sources[module], test)

    def test_the_plan_runs_at_a_reduced_scale_on_a_pilot_seed(self):
        self.assertGreaterEqual(PLAN["seed"], 1001)
        self.assertNotIn(PLAN["seed"], PINNED["seeds"])
        self.assertEqual(PLAN["subcommand"], "c007")


class RuleDerivationTests(unittest.TestCase):
    def test_signal_is_the_brier_gain_over_the_constant_predictor(self):
        self.assertEqual(rules.signal(2_600_000, 2_000_000, 10_000_000), 60)
        self.assertEqual(rules.signal(2_000_000, 2_600_000, 10_000_000), -60)
        self.assertEqual(rules.signal(2_000_000, 2_000_000, 10_000_000), 0)

    def test_the_head_gap_is_how_far_the_head_is_above_the_reference(self):
        self.assertEqual(rules.gap_permille(2_300_000, 2_000_000, 10_000_000), 30)
        self.assertEqual(rules.gap_permille(2_000_000, 2_300_000, 10_000_000), -30)
        self.assertEqual(rules.gap_permille(2_000_000, 2_000_000, 10_000_000), 0)

    def test_unconverged_guard_counts_the_judged_heads(self):
        self.assertEqual(rules.unconverged_guard_arms({"self_model": 0, "probe_matched": 1}), 1)
        self.assertEqual(rules.unconverged_guard_arms({"self_model": 0, "probe_matched": 0}), 2)
        self.assertEqual(rules.unconverged_guard_arms({"self_model": 1, "probe_matched": 1, "probe_io": 0}), 0)

    def test_band_error_is_inclusive_of_its_ends(self):
        self.assertEqual(rules.band_error(100, 100, 700), 0)
        self.assertEqual(rules.band_error(700, 100, 700), 0)
        self.assertEqual(rules.band_error(99, 100, 700), 1)
        self.assertEqual(rules.band_error(701, 100, 700), 1)

    def test_agreement_error_needs_more_than_chance_plus_slack(self):
        self.assertEqual(rules.agreement_error(226, 1000, 166, 60), 0)
        self.assertEqual(rules.agreement_error(227, 1000, 166, 60), 1)
        self.assertEqual(rules.agreement_error(1000, 1000, 166, 60), 1)

    def test_unchanged_error_needs_more_than_the_ceiling(self):
        self.assertEqual(rules.unchanged_error(100, 1000, 100), 0)
        self.assertEqual(rules.unchanged_error(101, 1000, 100), 1)
        self.assertEqual(rules.unchanged_error(1000, 1000, 100), 1)


@unittest.skipIf(torch is None, "torch is not installed")
class MetricTests(unittest.TestCase):
    def test_brier_matches_hand_counts(self):
        p, y = torch.tensor([0.5, 0.25]), torch.tensor([1.0, 0.0])
        self.assertEqual(model.brier_sum_micro(p, y), 312500)
        self.assertEqual(model.brier_reference_micro([0.5, 0.25], [1.0, 0.0]), 312500)

    def test_auroc_matches_hand_counts(self):
        self.assertEqual(model.auroc_permille(torch.tensor([0.9, 0.8, 0.3, 0.3]), torch.tensor([1.0, 1.0, 0.0, 0.0])), 1000)
        self.assertEqual(model.auroc_permille(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0])), 500)  # a tie counts half
        self.assertEqual(model.auroc_permille(torch.tensor([0.4, 0.8, 0.5]), torch.tensor([1.0, 1.0, 0.0])), 500)
        self.assertEqual(model.auroc_permille(torch.tensor([0.1, 0.9]), torch.tensor([1.0, 0.0])), 0)
        self.assertEqual(model.auroc_permille(torch.tensor([0.1, 0.9]), torch.tensor([0.0, 0.0])), 500)  # no positive: undefined, neutral

    def test_ece_matches_hand_counts(self):
        # two bins, each off by 0.1 in the mean: ECE 0.1
        self.assertEqual(model.ece_permille(torch.tensor([0.1, 0.1, 0.9, 0.9]), torch.tensor([0.0, 0.0, 1.0, 1.0])), 100)
        self.assertEqual(model.ece_permille(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])), 0)

    def test_the_risk_coverage_area_keeps_the_most_confident_first(self):
        # kept in order of lowest predicted failure: outcomes 0, 0, 1: selective risks 0, 0, 1/3: area 1/9
        self.assertEqual(model.risk_coverage_area_permille(torch.tensor([0.1, 0.9, 0.5]), torch.tensor([0.0, 1.0, 0.0])), 111)

    def test_the_confidence_score_rises_as_the_model_gets_less_sure(self):
        z = model.confidence_score(torch.tensor([[0.9, 0.1], [0.6, 0.4]]))
        self.assertLess(float(z[0]), float(z[1]))
        self.assertAlmostEqual(float(z[0]), -2.197225, places=4)
        self.assertAlmostEqual(float(z[1]), -0.405465, places=4)

    def test_platt_recovers_a_known_calibration(self):
        z = torch.linspace(-3, 3, 2000)
        y = torch.sigmoid(2.0 * z - 1.0)  # soft targets of a known calibration
        a, b = model.platt_fit(z, y, steps=800)
        self.assertAlmostEqual(a, 2.0, delta=0.15)
        self.assertAlmostEqual(b, -1.0, delta=0.15)


@unittest.skipIf(torch is None, "torch is not installed")
class TaskTests(unittest.TestCase):
    CFG = {"classes": 4, "dim": 8, "separation_milli": 3000, "noise_milli": 1000, "shift_milli": 2000, "shift_noise_milli": 1400}

    def test_class_means_have_the_pinned_separation(self):
        means = task.class_means(torch.Generator().manual_seed(1), self.CFG)
        self.assertEqual(tuple(means.shape), (4, 8))
        self.assertTrue(torch.allclose(means.norm(dim=1), torch.full((4,), 3.0), atol=1e-5))

    def test_the_shifted_split_is_displaced_and_noisier_and_the_null_labels_are_independent(self):
        gen = torch.Generator().manual_seed(2)
        means = task.class_means(gen, self.CFG)
        direction = task.shift_direction(gen, self.CFG)
        self.assertAlmostEqual(float(direction.norm()), 1.0, places=5)
        x0, y0 = task.sample(torch.Generator().manual_seed(3), means, 4000, self.CFG)
        x1, y1 = task.sample(torch.Generator().manual_seed(3), means, 4000, self.CFG, shifted=True, direction=direction)
        self.assertTrue(torch.equal(y0, y1))
        displacement = ((x1 - means[y1]) @ direction).mean()
        self.assertAlmostEqual(float(displacement), 2.0, delta=0.1)
        self.assertGreater(float((x1 - means[y1]).std()), float((x0 - means[y0]).std()) * 1.3)
        xn, yn = task.sample(torch.Generator().manual_seed(3), means, 4000, self.CFG, null=True)
        self.assertTrue(torch.equal(xn, x0))  # same inputs, labels redrawn
        self.assertLess(float((yn == y0).float().mean()), 0.35)

    def test_outcomes_equal_the_independent_computation_and_mean_wrong(self):
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 1.0, 5.0], [1.0, 1.0, 0.0]])
        labels = torch.tensor([0, 2, 2, 1])
        self.assertEqual(task.outcomes(logits, labels).tolist(), [0.0, 1.0, 0.0, 1.0])  # row 3: a tie goes to the lowest class, which is not 1
        self.assertEqual(task.outcomes_independent(logits, labels), [0, 1, 0, 1])

    def test_permuted_outcomes_keep_the_marginal(self):
        y = (torch.arange(200) % 5 == 0).float()
        p = task.permute_outcomes(y, torch.Generator().manual_seed(4))
        self.assertEqual(float(p.sum()), float(y.sum()))
        self.assertFalse(torch.equal(p, y))


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTests(unittest.TestCase):
    CFG = {"dim": 6, "hidden": 8, "experts": 4, "top_k": 2, "classes": 5, "probe_hidden": 7, "probe_hidden2": 5}

    def test_the_moe_layer_selects_top_k_and_weights_sum_to_one(self):
        layer = model.MoELayer(6, 8, 4, 2)
        x = torch.randn(10, 6)
        out, probs, mask = layer(x)
        self.assertTrue(torch.equal(mask.sum(dim=-1), torch.full((10,), 2.0)))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5))
        self.assertEqual(tuple(out.shape), (10, 8))

    def test_feature_widths_equal_the_trace(self):
        net = model.Classifier(self.CFG)
        trace = net.trace(torch.randn(3, 6))
        for parts in (model.EARLY, model.LATE, model.EARLY + model.LATE):
            self.assertEqual(model.features(trace, parts).shape[1], model.feature_width(self.CFG, parts), parts)

    def test_the_auxiliary_view_masks_the_late_part(self):
        cfg = self.CFG
        width, late = model.feature_width(cfg, model.EARLY + model.LATE), model.feature_width(cfg, model.LATE)
        net = model.Predictor(width, late, cfg)
        v = torch.randn(4, width)
        w = v.clone()
        w[:, width - late :] += 5.0  # only the late part changes
        with torch.no_grad():
            for a, b in zip(net.auxiliary(v), net.auxiliary(w)):
                self.assertTrue(torch.equal(a, b))
            self.assertFalse(torch.equal(net(v), net(w)))  # the error head reads the full view

    def test_the_self_model_and_the_probe_share_an_architecture_and_start_from_the_same_weights(self):
        cfg = self.CFG
        width, late = model.feature_width(cfg, model.EARLY + model.LATE), model.feature_width(cfg, model.LATE)
        torch.manual_seed(5)
        a = model.Predictor(width, late, cfg)
        torch.manual_seed(5)
        b = model.Predictor(width, late, cfg)
        self.assertEqual(model.count_params(a), model.count_params(b))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())))

    def test_the_base_adds_the_calibrated_own_confidence_to_the_network_output(self):
        cfg = self.CFG
        width, late = model.feature_width(cfg, model.EARLY + model.LATE) + 1, model.feature_width(cfg, model.LATE) + 1
        torch.manual_seed(6)
        plain = model.Predictor(width, late, cfg)
        torch.manual_seed(6)
        based = model.Predictor(width, late, cfg, base=(2.0, -1.0))
        v = torch.randn(5, width)
        with torch.no_grad():
            self.assertTrue(torch.allclose(based(v), plain(v) + 2.0 * v[:, -1] - 1.0, atol=1e-6))
            self.assertTrue(torch.equal(plain(v), model.Predictor.forward(plain, v)))
            w = v.clone()
            w[:, :-1] = 0.0
            self.assertTrue(torch.allclose(based(w) - plain(w), 2.0 * w[:, -1] - 1.0, atol=1e-6))
        self.assertEqual(model.count_params(plain), model.count_params(based))  # the base has no trainable parameter

    def test_the_parameter_digest_sees_one_weight_change(self):
        net = model.Classifier(self.CFG)
        before = model.params_digest(net)
        self.assertEqual(before, model.params_digest(net))
        with torch.no_grad():
            next(net.parameters()).add_(1e-3)
        self.assertNotEqual(before, model.params_digest(net))


@unittest.skipIf(torch is None, "torch is not installed")
class HarnessTests(unittest.TestCase):
    def test_overlap_counts_inputs_in_two_splits(self):
        a = torch.randn(10, 8)
        self.assertEqual(harness.overlap_count({"a": a, "b": torch.randn(10, 8)}), 0)
        self.assertEqual(harness.overlap_count({"a": a, "b": torch.cat([a[:3], torch.randn(5, 8)])}), 3)

    def test_advice_that_writes_the_environment_is_seen(self):
        env = harness.Environment(torch.tensor([0.0, 1.0]), torch.tensor([True, False]), 0.3, "d")
        self.assertEqual(harness.advice_leak(env, torch.tensor([0.9, 0.2])), 0)
        self.assertEqual(harness.advice_probe(harness.Environment(torch.tensor([0.0]), torch.tensor([True]), 0.3, "d")), True)

    def test_the_write_probe_sees_a_change_in_a_copy_and_leaves_the_original(self):
        net = model.Classifier({"dim": 6, "hidden": 8, "experts": 4, "top_k": 2, "classes": 5})
        digest = model.params_digest(net)
        self.assertTrue(harness.write_probe(net, digest))
        self.assertEqual(digest, model.params_digest(net))

    def test_helpers(self):
        self.assertEqual(harness.mismatches([1, 2, 3], [1, 0, 3]), 1)
        self.assertEqual(harness.differ(1, 1), 0)
        self.assertEqual(harness.differ(1, 2), 1)
        x = torch.arange(6.0).reshape(2, 3)
        self.assertEqual(harness.feature_digest(x), harness.feature_digest(x.clone()))
        self.assertNotEqual(harness.feature_digest(x), harness.feature_digest(x.flip(0)))
        self.assertEqual(harness.score_counts(torch.tensor([0.5, 0.25]), torch.tensor([1.0, 0.0])), (2_000_000 - 312500, 2_000_000))

    def test_a_tiny_run_passes_with_every_control_tripped(self):
        overrides = {"model_steps": 100, "max_steps": 100, "max_steps_control": 60, "eval_every": 20, "patience": 2}
        result = harness.run(1002, overrides)
        self.assertEqual(result["scale"], "custom")
        self.assertEqual(result["integrity_failed"], [])
        self.assertEqual(result["falsifiers_failed"], [])
        self.assertEqual(result["coverage_failed"], [])
        self.assertTrue(all(c["required_rule_tripped"] for c in result["controls"].values()))
        self.assertEqual(sorted(result["controls"]), sorted(rules.CONTROLS))
        for arm in ("self_model", "probe_matched", "probe_io", "constant", "own_confidence", "self_model_shuffled", "self_model_noisy", "self_model_perm"):
            self.assertIn(arm, result["arms"])
        self.assertEqual(result["arms"]["self_model"]["params"], result["arms"]["probe_matched"]["params"])
        self.assertEqual(result["arms"]["self_model"]["input_digest"], result["arms"]["probe_matched"]["input_digest"])
        self.assertEqual(result["arms"]["probe_matched"]["aux_weight_milli"], 0)

    def test_a_seed_gives_the_same_record_every_time(self):
        """Run twice in one process (the second run starts from a different global random state): every arm's record, and so every number the aggregator reads, is identical."""
        overrides = {"model_steps": 100, "max_steps": 100, "max_steps_control": 60, "eval_every": 20, "patience": 2}
        first, second = harness.run(1002, overrides), harness.run(1002, overrides)
        self.assertEqual(first["arms"], second["arms"])
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first["nondeterminism"], 0)


if __name__ == "__main__":
    unittest.main()
