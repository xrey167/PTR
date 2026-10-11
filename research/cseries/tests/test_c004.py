"""Tests of C004's rules, task, models and harness. The rule tests are torch-free and pin the harness's lists to criteria.toml and to
the mutation plan; the others need torch and skip without it (the repository's CI python has none; the locked environment runs them)."""

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/consciousness/C004-recurrent-integration"
sys.path.insert(0, str(ROOT / "research/cseries/lib"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rules = load("c004_rules", ROOT / "research/cseries/c004/c004_rules.py")
CONFIG = tomllib.loads((EXP / "config.toml").read_text(encoding="utf-8"))
CRITERIA = tomllib.loads((EXP / "criteria.toml").read_text(encoding="utf-8"))
PLAN = tomllib.loads((EXP / "tests/mutations.toml").read_text(encoding="utf-8"))
PINNED = CONFIG["pinned_thresholds"]

try:
    import torch  # noqa: F401

    sys.path.insert(0, str(ROOT / "research/cseries/c004"))
    task = load("c004_task", ROOT / "research/cseries/c004/c004_task.py")
    model = load("c004_model", ROOT / "research/cseries/c004/c004_model.py")
    harness = load("c004_harness", ROOT / "research/cseries/c004/harness.py")
    import train as plateau_train  # noqa: E402

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
        for name in ("null_permuted", "null_window"):
            self.assertEqual(table[name]["margin_permille"], PINNED["null_world_margin_permille"])
        self.assertEqual(table["t_support"], PINNED["paired_t_support"])
        self.assertEqual(table["t_refute"], PINNED["paired_t_refute"])
        self.assertEqual(table["seeds_required"], len(PINNED["seeds"]))
        self.assertEqual(table["max_untrained_recurrent_seeds"], PINNED["max_untrained_recurrent_seeds"])
        self.assertEqual(table["max_untrained_comparator_seeds"], PINNED["max_untrained_comparator_seeds"])
        self.assertEqual(table["max_unconverged_seeds"], PINNED["max_unconverged_seeds"])
        self.assertEqual(table["metric"], f"ex{PINNED['primary_length']}")
        self.assertEqual(table["against"], ["feedforward", "reset_state"])
        for key in ("sensitivity_gross", "sensitivity_graded", "null_permuted", "null_window"):
            self.assertIn(key, table)

    def test_the_quantiles_are_those_of_the_seed_count(self):
        sys.path.insert(0, str(ROOT / "research/cseries/lib"))
        import stats

        n = len(PINNED["seeds"])
        self.assertEqual(PINNED["paired_t_support"], stats.quantile_string(0.95, n - 1))
        self.assertEqual(PINNED["paired_t_refute"], stats.quantile_string(1 - 0.05 / 2, n - 1))

    def test_pilot_and_declared_seeds_are_disjoint_and_pilots_start_at_1001(self):
        self.assertFalse(set(PINNED["seeds"]) & set(PINNED["pilot_seeds"]))
        self.assertGreaterEqual(min(PINNED["pilot_seeds"]), 1001)
        self.assertEqual(len(set(PINNED["seeds"])), len(PINNED["seeds"]))
        self.assertLessEqual(len(PINNED["seeds"]), 20)

    def test_the_preregistration_table_restates_the_pinned_values(self):
        table = CONFIG["preregistration"]
        for key, value in table.items():
            if key.endswith("_sha256") or key in ("preregistration_text", "criteria", "arm_lr_micro", "harness"):
                continue
            scaled = key[: -len("_milli")] if key.endswith("_milli") and key[: -len("_milli")] in PINNED else key[: -len("_micro")] if key.endswith("_micro") else key
            self.assertIn(scaled, PINNED, key)
            expected = PINNED[scaled]
            if scaled != key:
                expected = round(expected * (1000 if key.endswith("_milli") else 1_000_000))
            self.assertEqual(value, expected, key)
        lr = ",".join(f"{name}={round(rate * 1_000_000)}" for name, rate in PINNED["arm_lr"].items())
        self.assertEqual(table["arm_lr_micro"], lr)

    def test_the_required_keys_of_the_list_are_the_preregistered_table_with_the_types_of_its_values(self):
        required = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"]["C004"]["required"]
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

    def test_the_judged_arms_and_controls_are_what_the_aggregator_reads(self):
        self.assertEqual(set(rules.JUDGED_ARMS), {"recurrent", "feedforward"})
        self.assertEqual(len(rules.CONTROLS), 8)


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
        declared = set(PLAN["hard_counters"])
        self.assertEqual(sorted(self.expected() - declared), [])

    def test_every_mutation_is_killed_by_the_harness_or_by_named_unit_tests(self):
        sources = {p.stem: p.read_text(encoding="utf-8") for p in (ROOT / "research/cseries/tests").glob("test_*.py")}
        for mutation in PLAN["mutation"]:
            if not mutation.get("unit_tests"):
                self.assertTrue(all(not e.startswith("unit:") for e in mutation["expect"]), mutation["name"])
                continue
            self.assertTrue(all(e.startswith("unit:") for e in mutation["expect"]), mutation["name"])
            for test in mutation["unit_tests"]:
                module, *path = test.split(".")
                self.assertIn(f"class {path[0]}", sources[module], test)
                if len(path) > 1:
                    self.assertIn(f"def {path[1]}", sources[module], test)

    def test_the_plan_names_a_killing_mutation_for_every_guard_and_the_protocol(self):
        names = {m["name"] for m in PLAN["mutation"]}
        for name in ("trained-ignores-the-floor", "unconverged-guard-ignores-the-comparator", "train-patience-off-by-one", "train-min-delta-ignored",
                     "best-checkpoint-not-restored", "converged-flag-always-true", "plateau-audit-min-delta-ignored", "plateau-audit-patience-off-by-one"):
            self.assertIn(name, names)

    def test_the_plan_runs_at_a_reduced_scale_on_a_pilot_seed(self):
        self.assertGreaterEqual(PLAN["seed"], 1001)
        self.assertNotIn(PLAN["seed"], PINNED["seeds"])
        self.assertEqual(PLAN["subcommand"], "c004")


class RuleDerivationTests(unittest.TestCase):
    def test_trained_uses_the_floor_inclusively(self):
        self.assertEqual(rules.trained(800, 800), 1)
        self.assertEqual(rules.trained(799, 800), 0)

    def test_unconverged_guard_counts_only_the_judged_arms(self):
        converged = {"recurrent": 1, "feedforward": 0, "reset_state": 0}
        self.assertEqual(rules.unconverged_guard_arms(converged), 1)
        self.assertEqual(rules.unconverged_guard_arms({"recurrent": 0, "feedforward": 0}), 2)
        self.assertEqual(rules.unconverged_guard_arms({"recurrent": 1, "feedforward": 1, "reset_state": 0}), 0)

    def test_above_chance_needs_more_than_the_slack(self):
        self.assertEqual(rules.above_chance(216, 166, 50), 0)
        self.assertEqual(rules.above_chance(217, 166, 50), 1)

    def test_rate_error_compares_the_measured_fraction_with_the_pinned_rate(self):
        self.assertEqual(rules.rate_error(300, 2000, 150, 50), 0)  # 150 per mille
        self.assertEqual(rules.rate_error(400, 2000, 150, 50), 0)  # 200: inside the slack
        self.assertEqual(rules.rate_error(401, 2000, 150, 50), 1)
        self.assertEqual(rules.rate_error(199, 2000, 150, 50), 1)  # 99.5: 50.5 below
        self.assertEqual(rules.rate_error(0, 2000, 150, 50), 1)

    def test_label_leak_needs_more_than_chance_plus_slack(self):
        self.assertEqual(rules.label_leak(226, 1000, 166, 60), 0)
        self.assertEqual(rules.label_leak(227, 1000, 166, 60), 1)
        self.assertEqual(rules.label_leak(1000, 1000, 166, 60), 1)


@unittest.skipIf(torch is None, "torch is not installed")
class TaskTests(unittest.TestCase):
    def batch(self, length, n=512, seed=1):
        return task.sample(torch.Generator().manual_seed(seed), n, length)

    def test_labels_equal_the_independent_computation(self):
        x = self.batch(9)
        self.assertEqual(task.labels(x).tolist(), task.labels_independent(x))
        self.assertEqual(task.window_labels(x).tolist(), task.window_labels_independent(x))

    def test_the_answer_is_the_last_column_of_the_running_counter(self):
        x = self.batch(7)
        self.assertTrue(torch.equal(task.running_labels(x)[:, -1], task.labels(x)))
        self.assertTrue(torch.equal(task.running_labels(x)[:, 0], x[:, 0]))
        self.assertEqual(task.running_labels(torch.tensor([[1, 2, 4]])).tolist(), [[1, 3, 1]])

    def test_window_running_labels_end_in_the_window_answer(self):
        x = self.batch(5)
        self.assertTrue(torch.equal(task.window_running_labels(x)[:, -1], task.window_labels(x)))
        self.assertTrue(torch.equal(task.window_running_labels(x)[:, 0], x[:, 0]))

    def test_changing_any_one_step_changes_the_answer(self):
        x = self.batch(10, n=64)
        for position in range(10):
            bumped = x.clone()
            bumped[:, position] = (bumped[:, position] + 1) % 6
            self.assertTrue((task.labels(bumped) != task.labels(x)).all(), position)

    def test_the_last_step_alone_determines_nothing(self):
        x = self.batch(8, n=6000)
        y, last = task.labels(x), x[:, -1]
        for symbol in range(6):
            counts = torch.bincount(y[last == symbol], minlength=6).float()
            self.assertGreater(float(counts.min() / counts.sum()), 0.10)  # roughly uniform (1/6 each)

    def test_the_window_answer_does_not_depend_on_earlier_steps(self):
        x = self.batch(9, n=64)
        bumped = x.clone()
        bumped[:, 0] = (bumped[:, 0] + 1) % 6
        self.assertTrue(torch.equal(task.window_labels(bumped), task.window_labels(x)))

    def test_the_task_is_a_pure_function_of_the_generator(self):
        a, b = self.batch(6, seed=3), self.batch(6, seed=3)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, self.batch(6, seed=4)))

    def test_corrupt_changes_about_the_rate_and_each_change_alters_the_answer(self):
        x = self.batch(12, n=4000)
        c = task.corrupt(x, torch.Generator().manual_seed(9), 150)
        changed = (c != x).any(dim=1)
        self.assertEqual(int((c != x).sum(dim=1).max()), 1)  # one step at most
        self.assertAlmostEqual(float(changed.float().mean()), 0.15, delta=0.03)
        self.assertTrue((task.labels(c)[changed] != task.labels(x)[changed]).all())
        self.assertTrue(torch.equal(task.corrupt(x, torch.Generator().manual_seed(9), 0), x))

    def test_permute_labels_keeps_the_marginal_and_moves_the_pairing(self):
        y = task.running_labels(self.batch(6, n=600))
        p = task.permute_labels(y, torch.Generator().manual_seed(2))
        self.assertEqual(sorted(map(tuple, p.tolist())), sorted(map(tuple, y.tolist())))
        self.assertLess(float((p[:, -1] == y[:, -1]).float().mean()), 0.3)


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTests(unittest.TestCase):
    def test_analytic_parameter_counts_equal_the_modules(self):
        for de, dh in [(4, 8), (32, 128), (7, 5)]:
            self.assertEqual(model.recurrent_params(de, dh), model.count_params(model.Recurrent(de, dh)))
        for d, h, f, layers in [(8, 2, 16, 1), (48, 4, 96, 4), (24, 2, 50, 3)]:
            self.assertEqual(model.feedforward_params(d, f, layers), model.count_params(model.Feedforward(d, h, f, layers)))

    def test_flop_formulas_match_hand_counts(self):
        # GRU, T = 2, embedding 3, hidden 4: 2 steps * 6 * 4 * (3 + 4) = 336, read-out 2 * 4 * 6 = 48
        self.assertEqual(model.recurrent_flops(2, 3, 4), 384)
        # feed-forward, T = 3, width 4, ffn 8, one layer: 3 * (8 * 16 + 4 * 4 * 8) = 768, attention 2 * 4 * 3 * 4 = 96, read-out 48
        self.assertEqual(model.feedforward_flops(3, 4, 8, 1), 912)

    def test_the_flop_formula_agrees_with_torch_on_the_linear_part(self):
        from torch.utils.flop_counter import FlopCounterMode

        d, f, layers, t = 8, 16, 2, 5
        net = model.Feedforward(d, 2, f, layers)
        x = torch.randint(0, 6, (1, t))
        with FlopCounterMode(display=False) as counter:
            net(x)
        # torch counts the matrix products only: the linear layers, the two attention products over the FULL t x t matrix, and the read-out at every position
        linear = layers * t * (8 * d * d + 4 * d * f)
        self.assertEqual(counter.get_total_flops(), linear + 4 * layers * t * t * d + 2 * d * 6 * t)
        # the formula counts the causal half of the attention and the read-out at the last position only
        self.assertEqual(model.feedforward_flops(t, d, f, layers), linear + 2 * d * t * (t + 1) * layers + 2 * d * 6)

    def test_the_reset_arm_ignores_every_step_but_the_last(self):
        net = model.Recurrent(6, 10, reset=True)
        x = torch.randint(0, 6, (16, 9))
        y = x.clone()
        y[:, :-1] = (y[:, :-1] + 1) % 6
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(x), net(y)))

    def test_the_recurrent_arm_depends_on_the_first_step(self):
        net = model.Recurrent(6, 10)
        x = torch.randint(0, 6, (16, 9))
        y = x.clone()
        y[:, 0] = (y[:, 0] + 1) % 6
        with torch.no_grad():
            self.assertFalse(torch.allclose(net(x), net(y)))

    def test_the_feedforward_stack_is_causal_and_the_unmasked_one_is_not(self):
        x = torch.randint(0, 6, (8, 7))
        for causal, leaks in ((True, 0), (False, 8)):
            net = model.Feedforward(8, 2, 16, 2, causal=causal)
            self.assertEqual(harness.causal_leaks(net, x), leaks, causal)

    def test_the_relative_bias_is_alibi_and_masked_above_the_diagonal(self):
        net = model.Feedforward(8, 2, 16, 1)
        bias = net.bias(4)
        self.assertEqual(tuple(bias.shape), (1, 2, 4, 4))
        slope = float(net.slopes[0])
        self.assertAlmostEqual(float(bias[0, 0, 3, 1]), -2 * slope, places=6)
        self.assertEqual(float(bias[0, 0, 2, 2]), 0.0)
        self.assertEqual(float(bias[0, 0, 1, 3]), float("-inf"))
        self.assertTrue(float(net.slopes[0]) > float(net.slopes[1]) > 0)
        self.assertTrue(bool((model.Feedforward(8, 2, 16, 1, positional="none").bias(4)[0, 0, 3, :3] == 0).all()))

    def test_the_matcher_meets_the_tolerances_for_the_pinned_configuration(self):
        de, dh, ref = PINNED["d_embed"], PINNED["d_hidden"], PINNED["max_train_length"] if "max_train_length" in PINNED else max(PINNED["train_lengths"])
        target = model.recurrent_params(de, dh)
        width, ffn = model.match_feedforward(target, model.recurrent_flops(ref, de, dh), ref, PINNED["ff_layers"], PINNED["ff_heads"],
                                             PINNED["param_tolerance_permille"], PINNED["flop_tolerance_permille"])
        params = model.feedforward_params(width, ffn, PINNED["ff_layers"])
        flops = model.feedforward_flops(ref, width, ffn, PINNED["ff_layers"])
        self.assertLessEqual(1000 * abs(params - target), PINNED["param_tolerance_permille"] * target)
        self.assertLessEqual(1000 * abs(flops - model.recurrent_flops(ref, de, dh)), PINNED["flop_tolerance_permille"] * model.recurrent_flops(ref, de, dh))
        self.assertEqual(width % PINNED["ff_heads"], 0)


@unittest.skipIf(torch is None, "torch is not installed")
class TrainingProtocolTests(unittest.TestCase):
    """`train_to_plateau` on a model whose validation accuracy is scripted, so that the stopping rule is read off exactly."""

    def run_protocol(self, curve, patience=3, min_delta=5, cap=1000):
        net = torch.nn.Linear(1, 1)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
        cfg = {"max_steps": cap, "grad_clip": 1.0, "eval_every": 1, "patience": patience, "min_delta_permille": min_delta, "loss_window_evals": 1}
        values = iter(curve)
        calls = []

        def validate(n):
            if len(calls) < len(curve):
                calls.append(1)
                value = curve[len(calls) - 1]
            else:
                value = curve[-1]
            return value, 1000

        def step_loss(step):
            return net(torch.ones(1, 1)).sum() * 0.0 + 1.0 + 0.0 * step

        return plateau_train.train_to_plateau(net, opt, step_loss, validate, cfg, max_steps=cap)

    def test_it_stops_after_exactly_patience_evaluations_without_a_gain(self):
        _, info, _, _ = self.run_protocol([100, 200, 201, 202, 203, 999], patience=3)
        self.assertTrue(info["converged"])
        self.assertEqual(info["steps"], 5)  # the best (200) was seen at evaluation 2; evaluations 3, 4, 5 brought no gain over 5

    def test_a_gain_not_above_the_minimum_does_not_reset_the_count(self):
        _, info, _, _ = self.run_protocol([100, 105, 105, 105], patience=3, min_delta=5)
        self.assertTrue(info["converged"])
        self.assertEqual(info["steps"], 4)  # 105 is not more than 100 + 5, so evaluations 2, 3, 4 count

    def test_it_stops_at_the_cap_unconverged(self):
        _, info, _, _ = self.run_protocol([100, 200, 300, 400, 500, 600], patience=3, cap=6)
        self.assertFalse(info["converged"])
        self.assertEqual(info["steps"], 6)

    def test_the_best_weights_are_restored(self):
        net = torch.nn.Linear(1, 1)
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        cfg = {"max_steps": 6, "grad_clip": 10.0, "eval_every": 1, "patience": 100, "min_delta_permille": 0, "loss_window_evals": 1}
        accuracy = [900, 100, 100, 100, 100, 100]
        seen = []

        def validate(n):
            seen.append(n.weight.detach().clone())
            return accuracy[len(seen) - 1], 1000

        def step_loss(step):
            return (net(torch.ones(1, 1)) - 5.0).pow(2).sum()

        restored, info, _, _ = plateau_train.train_to_plateau(net, opt, step_loss, validate, cfg)
        self.assertEqual(info["best_val"], 900)
        self.assertTrue(torch.equal(restored.weight, seen[0]))
        checkpoint, convergence = plateau_train.audit(restored, info, lambda n: (900 if torch.equal(n.weight, seen[0]) else 0, 1000), {**cfg, "patience": 100})
        self.assertEqual((checkpoint, convergence), (0, 0))


@unittest.skipIf(torch is None, "torch is not installed")
class HarnessTests(unittest.TestCase):
    def test_length_leak_overlap_and_label_detectors(self):
        self.assertEqual(harness.length_leak_count({2, 3, 6}, [12, 18, 24]), 0)
        self.assertEqual(harness.length_leak_count({2, 12}, [12, 18, 24]), 1)
        x = task.sample(torch.Generator().manual_seed(1), 8, 6)
        forbidden = set(harness.row_hashes(x[:3]))
        self.assertEqual(harness.overlap_count(forbidden, x), 3)
        # a sequence of another length with the same symbols is not the same sequence
        self.assertEqual(harness.overlap_count(forbidden, x[:, :5]), 0)
        y = task.labels(x)
        self.assertEqual(harness.label_mismatch_count(x, y), 0)
        self.assertEqual(harness.label_mismatch_count(x, (y + 1) % 6), 8)

    def test_fresh_batch_avoids_the_forbidden_set(self):
        gen = torch.Generator().manual_seed(4)
        pool = task.sample(gen, 200, 4)
        forbidden = set(harness.row_hashes(pool))
        batch = harness.fresh_batch(gen, 300, 4, forbidden)
        self.assertEqual(batch.shape, (300, 4))
        self.assertEqual(harness.overlap_count(forbidden, batch), 0)

    def test_carry_probe_sees_the_recurrent_arm_and_not_the_reset_arm(self):
        x = task.sample(torch.Generator().manual_seed(2), 32, 10)
        moved, _ = harness.carry_hits(model.Recurrent(6, 12), x)
        self.assertEqual(moved, 32)
        moved, _ = harness.carry_hits(model.Recurrent(6, 12, reset=True), x)
        self.assertEqual(moved, 0)

    def test_a_tiny_run_passes_with_every_control_tripped(self):
        overrides = {"max_steps": 60, "max_steps_control": 40, "eval_every": 20, "patience": 2, "pool_per_length": 64}
        result = harness.run(1002, overrides)
        self.assertEqual(result["scale"], "custom")
        self.assertEqual(result["integrity_failed"], [])
        self.assertEqual(result["falsifiers_failed"], [])
        self.assertEqual(result["coverage_failed"], [])
        self.assertTrue(all(c["required_rule_tripped"] for c in result["controls"].values()))
        self.assertEqual(sorted(result["controls"]), sorted(rules.CONTROLS))
        self.assertEqual(result["arms_trained"], 8)
        arm = result["arms"]["feedforward"]
        self.assertEqual(result["comparator_primary_permille"], harness.permille(arm["ex12_correct"], arm["ex12_total"]))  # the saturation guard reads the stack, not another arm
        self.assertIn("ex12_correct", result["arms"]["feedforward_inlength"])
        self.assertEqual(result["inlength_control_errors"], 0)

    def test_a_seed_gives_the_same_record_every_time(self):
        """Run twice in one process (the second run starts from a different global random state): every arm's record, and so every number the aggregator reads, is identical."""
        overrides = {"max_steps": 60, "max_steps_control": 40, "eval_every": 20, "patience": 2, "pool_per_length": 64}
        first, second = harness.run(1002, overrides), harness.run(1002, overrides)
        self.assertEqual(first["arms"], second["arms"])
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first["nondeterminism"], 0)


if __name__ == "__main__":
    unittest.main()
