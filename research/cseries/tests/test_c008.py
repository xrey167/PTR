"""Tests of C008's rules, world, models and harness. The rule tests are torch-free and pin the harness's lists to criteria.toml and to the
mutation plan; the others need torch and skip without it (the repository's CI python has none; the locked environment runs them)."""

import importlib.util
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/consciousness/C008-attention-schema-routing-cpu"
sys.path.insert(0, str(ROOT / "research/cseries/lib"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rules = load("c008_rules", ROOT / "research/cseries/c003a/c008_rules.py")
CONFIG = tomllib.loads((EXP / "config.toml").read_text(encoding="utf-8"))
CRITERIA = tomllib.loads((EXP / "criteria.toml").read_text(encoding="utf-8"))
PLAN = tomllib.loads((EXP / "tests/mutations.toml").read_text(encoding="utf-8"))
PINNED = CONFIG["pinned_thresholds"]

try:
    import torch  # noqa: F401

    sys.path.insert(0, str(ROOT / "research/cseries/c003a"))
    task = load("c008_task", ROOT / "research/cseries/c003a/c008_task.py")
    model = load("c008_model", ROOT / "research/cseries/c003a/c008_model.py")
    harness = load("c008_harness", ROOT / "research/cseries/c003a/harness.py")
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
        self.assertEqual(table["t_support"], PINNED["paired_t_support"])
        self.assertEqual(table["t_refute"], PINNED["paired_t_refute"])
        self.assertEqual(table["seeds_required"], len(PINNED["seeds"]))
        self.assertEqual(table["max_untrained_schema_seeds"], PINNED["max_untrained_schema_seeds"])
        self.assertEqual(table["max_unconverged_seeds"], PINNED["max_unconverged_seeds"])
        self.assertEqual(table["against"], list(rules.COMPARATORS))
        self.assertEqual(table["arm"], "control")

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
            if key.endswith("_sha256") or key in ("preregistration_text", "criteria", "arm_lr_micro", "arm_entropy_milli", "harness"):
                continue
            scaled = key[: -len("_milli")] if key.endswith("_milli") and key[: -len("_milli")] in PINNED else key[: -len("_micro")] if key.endswith("_micro") else key
            self.assertIn(scaled, PINNED, key)
            expected = PINNED[scaled]
            if scaled != key:
                expected = round(expected * (1000 if key.endswith("_milli") else 1_000_000))
            self.assertEqual(value, expected, key)
        lr = ",".join(f"{name}={round(rate * 1_000_000)}" for name, rate in PINNED["arm_lr"].items())
        self.assertEqual(table["arm_lr_micro"], lr)
        self.assertEqual(table["arm_entropy_milli"], ",".join(f"{name}={rate}" for name, rate in PINNED["arm_entropy_milli"].items()))

    def test_the_required_keys_of_the_list_are_the_preregistered_table_with_the_types_of_its_values(self):
        required = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"]["C008"]["required"]
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
        self.assertEqual(rules.JUDGED_ARMS[0], "control")
        self.assertEqual(set(rules.COMPARATORS) - set(rules.JUDGED_ARMS), set())
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
        self.assertEqual(sorted(self.expected() - set(PLAN["hard_counters"])), [])

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

    def test_the_plan_runs_at_a_reduced_scale_on_a_pilot_seed(self):
        self.assertGreaterEqual(PLAN["seed"], 1001)
        self.assertNotIn(PLAN["seed"], PINNED["seeds"])
        self.assertEqual(PLAN["subcommand"], "c008")


class RuleDerivationTests(unittest.TestCase):
    def test_headroom_is_oracle_minus_comparator(self):
        self.assertEqual(rules.headroom(946, 860), 86)
        self.assertEqual(rules.headroom(900, 905), -5)

    def test_schema_trained_needs_the_floor_over_persistence(self):
        self.assertEqual(rules.schema_trained(150, 150), 1)
        self.assertEqual(rules.schema_trained(149, 150), 0)
        self.assertEqual(rules.schema_trained(-300, 150), 0)

    def test_the_gain_over_persistence_is_the_relative_l1_reduction(self):
        self.assertEqual(rules.gain_over_persistence(0.6, 1.0), 400)
        self.assertEqual(rules.gain_over_persistence(1.0, 1.0), 0)
        self.assertEqual(rules.gain_over_persistence(1.5, 1.0), -500)
        self.assertEqual(rules.gain_over_persistence(0.5, 0.0), 0)

    def test_unconverged_guard_counts_the_judged_arms(self):
        converged = {"control": 1, "schema_free": 0, "schema_free_wide": 1, "heuristic": 0, "describe_only": 1, "history_router": 1}
        self.assertEqual(rules.unconverged_guard_arms(converged), 2)
        self.assertEqual(rules.unconverged_guard_arms({**converged, "control": 0}), 3)
        self.assertEqual(rules.unconverged_guard_arms({**converged, "history_router": 0}), 3)

    def test_rate_error_compares_the_measured_fraction_with_the_pinned_rate(self):
        self.assertEqual(rules.rate_error(4608, 30720, 150, 50), 0)  # 150 per mille
        self.assertEqual(rules.rate_error(6144, 30720, 150, 50), 0)  # 200: inside the slack
        self.assertEqual(rules.rate_error(6145, 30720, 150, 50), 1)
        self.assertEqual(rules.rate_error(3071, 30720, 150, 50), 1)  # just below 100
        self.assertEqual(rules.rate_error(0, 30720, 150, 50), 1)

    def test_shuffle_error_needs_more_than_the_ceiling(self):
        self.assertEqual(rules.shuffle_error(0, 476, 100), 0)
        self.assertEqual(rules.shuffle_error(47, 476, 100), 0)
        self.assertEqual(rules.shuffle_error(48, 476, 100), 1)
        self.assertEqual(rules.shuffle_error(476, 476, 100), 1)


@unittest.skipIf(torch is None, "torch is not installed")
class TaskTests(unittest.TestCase):
    def cfg(self):
        return {"experts": 8, "tokens": 64, "regimes": 8, "dwell": 3, "hot_permille": 700, "noise_milli": 500}

    def test_class_probabilities_sum_to_one_and_put_the_hot_mass_on_two_classes(self):
        for regime in range(8):
            p = task.class_probabilities(regime, 8, 700)
            self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
            self.assertAlmostEqual(float(p[regime] + p[(regime + 1) % 8]), 0.7, places=6)
            self.assertEqual(sorted(p.argsort(descending=True)[:2].tolist()), sorted([regime, (regime + 1) % 8]))

    def test_regimes_cycle_with_the_dwell_and_stay_put_when_stationary(self):
        self.assertEqual([task.regime_of(t, 4, 3, False) for t in range(14)], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0])
        self.assertEqual({task.regime_of(t, 4, 3, True) for t in range(50)}, {0})

    def test_the_vectorised_verifier_equals_the_plain_loop(self):
        gen = torch.Generator().manual_seed(3)
        for _ in range(20):
            classes = torch.randint(0, 8, (64,), generator=gen)
            choices = torch.randint(0, 8, (64,), generator=gen)
            self.assertEqual(task.serve(classes, choices, 8, 9).tolist(), task.serve_reference(classes.tolist(), choices.tolist(), 8, 9))

    def test_capacity_counts_arrivals_in_batch_order(self):
        classes = torch.tensor([0, 0, 0, 0])
        choices = torch.tensor([0, 0, 0, 1])
        self.assertEqual(task.serve(classes, choices, 8, 2).tolist(), [True, True, False, True])

    def test_competence_is_the_class_and_the_next_expert(self):
        classes = torch.tensor([7, 7, 7, 3])
        choices = torch.tensor([7, 0, 1, 5])
        self.assertEqual(task.competent(classes, choices, 8).tolist(), [True, True, False, False])

    def test_the_oracle_equals_hand_computed_flows(self):
        # eight tokens of class 0, capacity 3: experts 0 and 1 can take six
        self.assertEqual(task.oracle_reward(torch.zeros(8, dtype=torch.long), 8, 3), 6)
        # classes 0 and 1 with four tokens each, capacity 3: experts 0, 1, 2 take nine, only eight tokens exist
        self.assertEqual(task.oracle_reward(torch.tensor([0] * 4 + [1] * 4), 8, 3), 8)
        # twelve tokens of class 0 and twelve of class 1, capacity 8: experts 0, 1, 2 take 24
        self.assertEqual(task.oracle_reward(torch.tensor([0] * 12 + [1] * 12), 8, 8), 24)
        # twelve tokens of class 0 and twelve of class 1, capacity 6: experts 0, 1, 2 take 18
        self.assertEqual(task.oracle_reward(torch.tensor([0] * 12 + [1] * 12), 8, 6), 18)

    def test_the_oracle_is_never_below_any_assignment(self):
        gen = torch.Generator().manual_seed(5)
        cfg = self.cfg()
        for t in range(12):
            classes, _ = task.sample_batch(gen, t, cfg)
            best = task.oracle_reward(classes, 8, 14)
            for _ in range(5):
                choices = (classes + torch.randint(0, 2, (64,), generator=gen)) % 8
                self.assertLessEqual(int(task.serve(classes, choices, 8, 14).sum()), best)

    def test_the_stream_is_a_pure_function_of_the_generator_and_the_batch_index(self):
        a = task.sample_batch(torch.Generator().manual_seed(1), 4, self.cfg())
        b = task.sample_batch(torch.Generator().manual_seed(1), 4, self.cfg())
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        self.assertEqual(tuple(a[1].shape), (64, 8))


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTests(unittest.TestCase):
    SMALL = {"experts": 2, "history": 2, "tokens": 5, "schema_hidden": 3, "law_form": "full"}

    def test_flop_and_parameter_formulas_match_hand_counts(self):
        self.assertEqual(model.router_params(4, 3, 2), 23)
        self.assertEqual(model.schema_params(2, 2, 3), 23)
        self.assertEqual(model.router_flops(4, 3, 2), 36)
        self.assertEqual(model.schema_flops(2, 2, 3), 36)
        # experts 2, history 2, tokens 5, schema hidden 3, router hidden 3: the router takes the 2 token features
        self.assertEqual(model.router_params(2, 3, 2), 17)
        per_batch_router = 5 * 2 * (2 * 3 + 3 * 2)  # 120
        self.assertEqual(model.batch_flops("schema_free", self.SMALL, 3), per_batch_router)
        self.assertEqual(model.batch_flops("heuristic", self.SMALL, 3), per_batch_router)
        self.assertEqual(model.batch_flops("control", self.SMALL, 3), per_batch_router + 36 + 2 * 2 * 2)  # the schema and the E x E control law, once per batch
        self.assertEqual(model.batch_flops("describe_only", self.SMALL, 3), per_batch_router + 36 + 8)
        self.assertEqual(model.total_params("control", self.SMALL, 3), 17 + 23 + 4)
        self.assertEqual(model.total_params("heuristic", self.SMALL, 3), 17)

    def test_analytic_parameter_counts_equal_the_modules(self):
        cfg = {"experts": 8, "history": 4, "tokens": 64, "schema_hidden": 32, "law_form": "full"}
        for kind, hidden in (("schema_free", 73), ("schema_free_wide", 148), ("heuristic", 73), ("describe_only", 48), ("control", 48), ("history_router", 44)):
            arm = model.Arm(kind, cfg, hidden)
            self.assertEqual(model.total_params(kind, cfg, hidden), sum(p.numel() for p in arm.parameters()), kind)

    def test_the_history_router_reads_the_history_next_to_every_token_and_has_no_schema(self):
        cfg = self.SMALL
        # input width E (1 + history) = 6: parameters 6 h + h + h E + E, FLOPs per token 2 (6 h + h E)
        self.assertEqual(model.router_input_width("history_router", cfg), 6)
        self.assertEqual(model.router_input_width("control", cfg), 2)
        self.assertEqual(model.total_params("history_router", cfg, 3), 6 * 3 + 3 + 3 * 2 + 2)
        self.assertEqual(model.batch_flops("history_router", cfg, 3), 5 * 2 * (6 * 3 + 3 * 2))
        arm = model.Arm("history_router", cfg, 3)
        self.assertFalse(arm.schema_on)
        x = torch.randn(5, 2)
        same = arm(x, torch.zeros(4))[0]
        moved = arm(x, torch.tensor([0.9, 0.1, 0.2, 0.8]))[0]
        self.assertFalse(torch.allclose(same, moved))
        free = model.Arm("schema_free", cfg, 3)
        self.assertTrue(torch.equal(free(x, torch.zeros(4))[0], free(x, torch.tensor([0.9, 0.1, 0.2, 0.8]))[0]))  # a schema-free router has no history input

    def test_the_matcher_finds_the_closest_width_and_ties_go_to_the_smaller(self):
        self.assertEqual(model.match_hidden(100, lambda h: 10 * h), 10)
        self.assertEqual(model.match_hidden(105, lambda h: 10 * h), 10)  # 100 and 110 are equally close
        self.assertEqual(model.match_hidden(106, lambda h: 10 * h), 11)

    def test_the_balancer_moves_its_bias_against_the_load(self):
        arm = model.Arm("heuristic", {"experts": 4, "history": 2, "tokens": 8, "schema_hidden": 3, "law_form": "full"}, 5)
        arm.update_bias(torch.tensor([3.0, 1.0, 1.0, 3.0]), 0.5)
        self.assertEqual(arm.bias.tolist(), [-0.5, 0.5, 0.5, -0.5])
        arm.update_bias(torch.tensor([2.0, 2.0, 2.0, 2.0]), 0.5)
        self.assertEqual(arm.bias.tolist(), [-0.5, 0.5, 0.5, -0.5])  # no error, no move

    def test_only_the_heuristic_arm_has_a_balancer(self):
        for kind in model.KINDS:
            arm = model.Arm(kind, {"experts": 4, "history": 2, "tokens": 8, "schema_hidden": 3, "law_form": "full"}, 5)
            arm.update_bias(torch.tensor([3.0, 1.0, 1.0, 3.0]), 0.5)
            self.assertEqual(bool(arm.bias.abs().sum() > 0), kind == "heuristic", kind)

    def test_the_policy_loss_does_not_reach_the_schema(self):
        cfg = {"experts": 4, "history": 2, "tokens": 8, "schema_hidden": 3, "law_form": "full"}
        arm = model.Arm("control", cfg, 5)
        logits = arm(torch.randn(8, 4), torch.rand(8))[0]
        logits.sum().backward()
        self.assertTrue(all(p.grad is None for p in arm.schema_parameters()))
        self.assertTrue(all(p.grad is not None for p in arm.policy_parameters()))

    def test_the_describe_only_arm_ignores_the_schema_output_and_the_control_arm_reads_it(self):
        cfg = {"experts": 4, "history": 2, "tokens": 8, "schema_hidden": 3, "law_form": "full"}
        x, history = torch.randn(8, 4), torch.rand(8)
        flat, spike = torch.full((4,), 0.25), torch.tensor([1.0, 0.0, 0.0, 0.0])
        describe, control = model.Arm("describe_only", cfg, 5), model.Arm("control", cfg, 5)
        for arm in (describe, control):  # the control law starts at zero; give both the same non-zero law
            torch.nn.init.eye_(arm.law.weight)
        with torch.no_grad():
            self.assertTrue(torch.equal(describe(x, history, flat)[0], describe(x, history, spike)[0]))
            self.assertFalse(torch.equal(control(x, history, flat)[0], control(x, history, spike)[0]))
        self.assertEqual(harness.schema_influence(describe, x, history, 4), 0)
        self.assertEqual(harness.schema_influence(control, x, history, 4), 1)

    def test_the_control_law_starts_at_zero_so_the_untrained_control_arm_is_the_schema_free_router(self):
        cfg = {"experts": 4, "history": 2, "tokens": 8, "schema_hidden": 3, "law_form": "full"}
        torch.manual_seed(0)
        control = model.Arm("control", cfg, 5)
        x, history = torch.randn(8, 4), torch.rand(8)
        with torch.no_grad():
            self.assertTrue(torch.equal(control(x, history)[0], control.router(x)))

    def test_the_wired_copy_shows_the_path_from_the_schema_to_the_logits(self):
        cfg = {"experts": 4, "history": 2, "tokens": 8, "schema_hidden": 3, "law_form": "full"}
        control = model.Arm("control", cfg, 5)
        x, history = torch.randn(8, 4), torch.rand(8)
        self.assertEqual(harness.schema_influence(control, x, history, 4), 0)  # untrained law: no influence yet
        self.assertEqual(harness.schema_influence(harness.wire_control_law(control, 4), x, history, 4), 1)
        self.assertEqual(float(control.law.weight.abs().sum()), 0.0)  # the original is untouched

    def test_the_schema_predicts_a_distribution(self):
        cfg = {"experts": 4, "history": 2, "tokens": 8, "schema_hidden": 3, "law_form": "full"}
        p = model.Arm("control", cfg, 5).predict(torch.rand(8))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


@unittest.skipIf(torch is None, "torch is not installed")
class HarnessTests(unittest.TestCase):
    def test_the_schedule_check_reads_the_hot_classes(self):
        on_hot = torch.tensor([2, 3, 2, 3, 2, 3, 0, 1, 4, 5])
        self.assertEqual(harness.schedule_error(on_hot, (2, 3), 8, 450), 0)
        self.assertEqual(harness.schedule_error(on_hot, (6, 7), 8, 450), 1)
        self.assertEqual(harness.schedule_error(torch.tensor([2, 3, 0, 1]), (2, 3), 8, 500), 0)  # exactly half is enough
        self.assertEqual(harness.schedule_error(torch.tensor([2, 0, 0, 1]), (2, 3), 8, 500), 1)

    def test_the_environment_digest_sees_a_capacity_change(self):
        env = harness.make_environment({"experts": 8, "capacity": 14})
        same = harness.Environment(env.capacity.clone(), env.competent.clone())
        self.assertEqual(env.digest(), same.digest())
        same.capacity[3] -= 1
        self.assertNotEqual(env.digest(), same.digest())
        same = harness.Environment(env.capacity.clone(), env.competent.clone())
        same.competent[0, 5] = True
        self.assertNotEqual(env.digest(), same.digest())

    def test_the_environment_verdicts_equal_the_world_functions(self):
        env = harness.make_environment({"experts": 8, "capacity": 9})
        gen = torch.Generator().manual_seed(8)
        classes, choices = torch.randint(0, 8, (64,), generator=gen), torch.randint(0, 8, (64,), generator=gen)
        self.assertEqual(harness.serve(env, classes, choices).tolist(), task.serve(classes, choices, 8, 9).tolist())
        self.assertEqual(harness.reward_audit(env, classes, choices), 0)

    def test_overlap_and_audit_detectors(self):
        a, b = torch.tensor([1, 2, 3]), torch.tensor([3, 2, 1])
        self.assertEqual(harness.overlap_count({harness.batch_hash(a)}, [harness.batch_hash(a), harness.batch_hash(b)]), 1)
        self.assertEqual(harness.audit_mismatches([True, False], [True, False]), 0)
        self.assertEqual(harness.audit_mismatches([True, False], [False, False]), 1)

    def test_an_authority_leak_is_seen_and_a_clean_route_is_not(self):
        cfg = {"experts": 8, "history": 4, "tokens": 16, "schema_hidden": 6, "capacity": 14, "law_form": "full"}
        env = harness.make_environment(cfg)
        arm = model.Arm("control", cfg, 8)
        x, history, gen = torch.randn(16, 8), torch.zeros(32), torch.Generator().manual_seed(2)
        self.assertEqual(harness.authority_leak(env, arm, x, history, gen, None), 0)

    def test_a_tiny_run_passes_with_every_control_tripped(self):
        overrides = {"max_steps": 60, "max_steps_null": 40, "eval_every": 20, "patience": 2, "eval_batches": 480, "val_batches": 12}
        result = harness.run(1002, overrides)
        self.assertEqual(result["scale"], "custom")
        self.assertEqual(result["integrity_failed"], [])
        self.assertEqual(result["falsifiers_failed"], [])
        self.assertEqual(result["coverage_failed"], [])
        self.assertTrue(all(c["required_rule_tripped"] for c in result["controls"].values()))
        self.assertEqual(sorted(result["controls"]), sorted(rules.CONTROLS))
        for arm in ("control", "schema_free", "history_router", "even_split", "random_router", "oracle", "control_shuffled", "schema_free_epsilon"):
            self.assertIn(arm, result["arms"])
        self.assertEqual(result["arms_trained"], 8)
        self.assertEqual(result["history_router_blind"], 0)
        self.assertEqual(result["blind_reference_permille"], harness.permille(result["arms"]["even_split"]["reward_correct"], result["arms"]["even_split"]["reward_total"]))
        self.assertGreaterEqual(result["arms"]["oracle"]["reward_correct"], result["arms"]["control"]["reward_correct"])

    def test_a_seed_gives_the_same_record_every_time(self):
        """Run twice in one process (the second run starts from a different global random state): every arm's record, and so every number the aggregator reads, is identical."""
        overrides = {"max_steps": 60, "max_steps_null": 40, "eval_every": 20, "patience": 2, "eval_batches": 480, "val_batches": 12}
        first, second = harness.run(1002, overrides), harness.run(1002, overrides)
        self.assertEqual(first["arms"], second["arms"])
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first["nondeterminism"], 0)


if __name__ == "__main__":
    unittest.main()
