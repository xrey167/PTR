"""Torch-free tests: the harness's rules agree with criteria.toml, and the plan covers every hard counter."""

import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/consciousness/C001-global-workspace"
sys.path.insert(0, str(ROOT / "research/consciousness/gw"))
import power  # noqa: E402
import rules  # noqa: E402

CONFIG = tomllib.loads((EXP / "config.toml").read_text(encoding="utf-8"))
CRITERIA = tomllib.loads((EXP / "criteria.toml").read_text(encoding="utf-8"))
PLAN = tomllib.loads((EXP / "tests/mutations.toml").read_text(encoding="utf-8"))
PINNED = CONFIG["pinned_thresholds"]


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
        for counter in ("n_leaks", "train_eval_overlap", "param_mismatch_arms", "nonfinite_arms"):
            self.assertIn(counter, rules.INTEGRITY_COUNTERS)

    def test_the_guarded_arms_are_the_workspace_and_the_pairwise_comparators(self):
        self.assertEqual(list(rules.PAIRWISE_COMPARATORS), CRITERIA["paired"]["pairwise"])
        self.assertEqual(list(rules.PAIRWISE_COMPARATORS), CRITERIA["paired"]["against"])
        self.assertEqual(set(rules.CONVERGENCE_GUARD_ARMS), {"workspace", *CRITERIA["paired"]["against"]})
        self.assertEqual(CRITERIA["paired"]["max_unconverged_seeds"], PINNED["max_unconverged_seeds"])

    def test_every_hard_counter_is_expected_by_a_mutation(self):
        expected = {name for m in PLAN["mutation"] for name in m["expect"]}
        self.assertEqual(sorted(set(rules.HARD_COUNTERS) - expected), [])

    def test_every_coverage_counter_is_expected_by_a_mutation(self):
        expected = {name for m in PLAN["mutation"] for name in m["expect"]}
        counters = {rule.split(" ")[0] for rule in CRITERIA["coverage"]}
        self.assertEqual(sorted(counters - expected), [])
        self.assertEqual(counters, {"arms_trained", "id_eval_instances", "ood16_eval_instances", "ood24_eval_instances",
                                    "workspace_influence_hits", "pairwise_influence_hits", "no_broadcast_reporter_hits"})

    def test_every_negative_control_is_expected_by_a_mutation(self):
        expected = {name for m in PLAN["mutation"] for name in m["expect"]}
        for control in ("n_leak", "overlap", "param_mismatch", "isolation_probe", "bottleneck", "dropout_probe"):
            self.assertIn("control:" + control, expected)

    def test_every_mutation_is_killed_by_the_harness_or_by_named_unit_tests(self):
        sources = {name: (ROOT / "research/consciousness/tests" / f"{name}.py").read_text(encoding="utf-8")
                   for name in ("test_rules", "test_task_model")}
        for mutation in PLAN["mutation"]:
            if not mutation.get("unit_tests"):
                self.assertTrue(all(not e.startswith("unit:") for e in mutation["expect"]), mutation["name"])
                continue
            self.assertTrue(all(e.startswith("unit:") for e in mutation["expect"]), mutation["name"])
            for test in mutation["unit_tests"]:
                module, *path = test.split(".")
                self.assertTrue(f"class {path[0]}" in sources[module], test)
                if len(path) > 1:
                    self.assertTrue(f"def {path[1]}" in sources[module], test)

    def test_the_plan_names_a_killing_mutation_for_every_judged_guard_and_control(self):
        names = {m["name"] for m in PLAN["mutation"]}
        for name in (
            "comparator-set-includes-no-broadcast", "workspace-trained-ignores-the-margin", "convergence-guard-ignores-the-comparators",
            "weak-arm-has-no-weight-decay", "training-ignores-the-arm-weight-decay", "crippled-arm-has-all-rounds", "best-checkpoint-not-restored", "converged-flag-always-true",
            "patience-never-reached", "validation-set-is-the-evaluation-set", "min-delta-ignored", "patience-off-by-one",
            "plateau-audit-ignores-min-delta", "plateau-audit-patience-off-by-one",
        ):
            self.assertIn(name, names)
        expected = {e for m in PLAN["mutation"] for e in m["expect"]}
        self.assertEqual({e for e in PLAN["hard_counters"]} - expected, set())  # every listed name is expected by a mutation

    def test_a_defect_of_the_workspace_arm_is_expected_to_reach_a_counter_not_exit_two(self):
        by_name = {m["name"]: m for m in PLAN["mutation"]}
        self.assertEqual(by_name["workspace-arm-does-not-broadcast"]["expect"], ["workspace_influence_hits"])
        self.assertEqual(by_name["topk-write-not-applied"]["expect"], ["bottleneck_violations"])

    def test_plan_hard_counters_start_with_the_harness_ones(self):
        self.assertEqual(PLAN["hard_counters"][: len(rules.HARD_COUNTERS)], list(rules.HARD_COUNTERS))

    def test_the_preregistration_table_restates_the_pinned_values_when_present(self):
        if "preregistration" not in CONFIG:
            self.skipTest("the [preregistration] table is added at the freeze")
        table = CONFIG["preregistration"]
        for key, value in table.items():
            if key in PINNED:
                self.assertEqual(value, PINNED[key], key)
        self.assertEqual(table["harness"], "c001")
        self.assertEqual(table["noise_milli"], round(PINNED["noise"] * 1000))
        lr = {k: round(v * 1_000_000) for k, v in PINNED["arm_lr"].items()}
        self.assertEqual(table["arm_lr_micro"], ",".join(f"{k}={v}" for k, v in lr.items()))
        self.assertEqual(table["weight_decay_micro"], round(PINNED["weight_decay"] * 1_000_000))
        self.assertEqual(table["grad_clip_milli"], round(PINNED["grad_clip"] * 1000))
        self.assertEqual(table["probe_scale_milli"], round(PINNED["probe_scale"] * 1000))
        wd = {k: round(v * 1_000_000) for k, v in PINNED["arm_weight_decay"].items()}
        self.assertEqual(table["arm_weight_decay_micro"], ",".join(f"{k}={v}" for k, v in wd.items()))
        self.assertEqual(table["arm_d_list"], ",".join(f"{k}={v}" for k, v in PINNED["arm_d"].items()))
        required = tomllib.loads((ROOT / "experiments/preregistration.toml").read_text(encoding="utf-8"))["experiment"]["C001"]["required"]
        self.assertEqual(sorted(table), sorted([*required, "criteria_sha256", "preregistration_text_sha256"]))

    def test_the_ood_module_counts_are_the_harness_constants(self):
        import ast
        source = (ROOT / "research/consciousness/gw/harness.py").read_text(encoding="utf-8")
        constant = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(source))
                        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "OOD_MODULES")
        self.assertEqual(list(constant), PINNED["ood_modules"])
        self.assertGreater(min(PINNED["ood_modules"]), PINNED["n_train_modules"])

    def test_the_pilot_seeds_are_disjoint_from_the_declared_seeds_and_are_all_at_least_1001(self):
        self.assertEqual(set(PINNED["pilot_seeds"]) & set(PINNED["seeds"]), set())
        self.assertGreaterEqual(min(PINNED["pilot_seeds"]), 1001)

    def test_every_arm_runs_at_the_workspace_width(self):
        # Parameter-matching rule: target = the workspace at d = 48; no arm is pinned at a width it was not matched at.
        self.assertEqual(set(PINNED["arm_d"].values()), {PINNED["arm_d"]["workspace"]})

    def test_the_margin_is_the_a_priori_one(self):
        self.assertEqual(PINNED["paired_margin_permille"], 30)
        self.assertEqual(PINNED["max_steps"], 20000)
        self.assertEqual(PINNED["graded_min_lower_permille"], PINNED["paired_margin_permille"] // 2)

    def test_the_arm_learning_rates_name_every_arm(self):
        sys.path.insert(0, str(ROOT / "research/consciousness/gw"))
        arms = {"workspace", "all_to_all", "bw_all_to_all", "sparse_all_to_all", "no_broadcast", "no_comm", "competition_off", "capacity_off", "weak_all_to_all", "crippled_all_to_all"}
        self.assertEqual(set(PINNED["arm_lr"]), arms)
        self.assertEqual(set(PINNED["arm_d"]), arms)
        self.assertEqual(PINNED["min_arms_trained"], len(arms))

    def test_the_pinned_t_quantiles_match_the_student_t_distribution_for_the_declared_seed_count(self):
        df = len(PINNED["seeds"]) - 1
        self.assertAlmostEqual(float(PINNED["paired_t_support"]), power.t_quantile(0.95, df), places=5)
        self.assertAlmostEqual(float(PINNED["paired_t_refute"]), power.t_quantile(1 - 0.05 / 3, df), places=5)

    def test_published_t_table_values(self):
        for df, one_sided_95 in ((4, 2.131847), (9, 1.833113), (14, 1.761310), (19, 1.729133)):
            self.assertAlmostEqual(power.t_quantile(0.95, df), one_sided_95, places=5)

    def test_the_power_estimate_is_monotone_in_the_effect_and_in_the_seed_count(self):
        t = float(PINNED["paired_t_support"])
        args = dict(sds=[60.0], margin=50, t_support=t, t_refute=t, pairwise=1, trials=3000, seed=1)
        weak = power.decision_probabilities([40.0], n=10, **args)[0]
        strong = power.decision_probabilities([120.0], n=10, **args)[0]
        self.assertLess(weak, strong)
        self.assertGreater(strong, 0.9)

    def test_the_invalid_branch_of_the_power_estimate_follows_the_decision_order(self):
        t_s, t_r = float(PINNED["paired_t_support"]), float(PINNED["paired_t_refute"])
        cov = [[100.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
        args = dict(n=20, margin=30, t_support=t_s, t_refute=t_r, graded_min=15, gross_min=50, trials=1500, seed=3, cov=cov)
        strong = power.decision_outcomes([80, 80, 80, 90, 400], **args)
        self.assertAlmostEqual(sum(strong), 1.0)
        self.assertGreater(strong[0], 0.99)  # COMPLETED
        null = power.decision_outcomes([0, 0, 0, 90, 400], **args)
        self.assertGreater(null[2], 0.99)  # FALSIFIED
        blind = power.decision_outcomes([0, 0, 0, 0, 400], **args)  # an unresolved graded control outranks FALSIFIED
        self.assertGreater(blind[3], 0.99)
        floor = power.decision_outcomes([80, 80, 80, 90, 0], **args)  # so does an unresolved gross control
        self.assertGreater(floor[3], 0.99)
        stalled = power.decision_outcomes([80, 80, 80, 90, 400], p_unconverged=1.0, **args)  # and more than 2 unconverged seeds
        self.assertEqual(stalled[3], 1.0)

    def test_only_seeds_flagged_by_the_harness_rule_count_toward_the_untrained_guard_in_the_power_estimate(self):
        self.assertIs(power.rules, rules)  # the guard's definition is rules.workspace_trained, not a 16-module cutoff
        t_s, t_r = float(PINNED["paired_t_support"]), float(PINNED["paired_t_refute"])
        cov = [[100.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
        args = dict(n=20, margin=30, t_support=t_s, t_refute=t_r, graded_min=15, gross_min=50, trials=800, seed=3, cov=cov,
                    p_workspace_fail=1.0, fail_contrasts=(-400.0, -400.0, -400.0))
        counted = power.decision_outcomes([80, 80, 80, 90, 400], fail_counted=True, **args)
        stalls_only = power.decision_outcomes([80, 80, 80, 90, 400], fail_counted=False, **args)
        self.assertEqual(counted[3], 1.0)  # twenty flagged seeds: INVALID
        self.assertEqual(stalls_only[3], 0.0)  # twenty stalled seeds the rule does not flag: they stay in the differences, no INVALID
        self.assertEqual(rules.workspace_trained(437, 333, PINNED["trained_margin_permille"]), 1)  # pilot seed 1011 is not flagged
        self.assertEqual(rules.workspace_trained(432, 333, PINNED["trained_margin_permille"]), 0)

    def test_the_covariance_and_its_factor_reproduce_the_sample(self):
        rows = [[1.0, 2.0], [2.0, 1.0], [4.0, 5.0], [3.0, 6.0]]
        mean, cov = power.covariance(rows)
        self.assertEqual(mean, [2.5, 3.5])
        self.assertAlmostEqual(cov[0][0], 5 / 3)
        low = power.cholesky(cov, jitter=0.0)
        for a in range(2):
            for b in range(2):
                self.assertAlmostEqual(sum(low[a][m] * low[b][m] for m in range(2)), cov[a][b], places=5)

    def test_judge_reports_the_failing_rule(self):
        result = {rule.split(" ")[0]: int(rule.split(" ")[2]) for rule in rules.coverage_rules(PINNED)}
        result.update({counter: 0 for counter in rules.HARD_COUNTERS})
        self.assertEqual(rules.judge(result, PINNED), ([], []))
        self.assertEqual(rules.integrity_failed(result), [])
        result["no_comm_leaks"] = 1
        result["n_leaks"] = 1
        result["workspace_influence_hits"] = 0
        coverage, falsifiers = rules.judge(result, PINNED)
        self.assertEqual(coverage, ["workspace_influence_hits >= 1"])
        self.assertEqual(falsifiers, ["no_comm_leaks <= 0"])
        self.assertEqual(rules.integrity_failed(result), ["n_leaks <= 0"])


class GuardDerivationTests(unittest.TestCase):
    """The pure derivations the harness uses for its guard scalars; each is pinned to hand-computed values so that a defect in
    one (planted by the mutation plan) fails here."""

    def test_the_comparator_is_the_best_pairwise_arm_and_nothing_else(self):
        ood = {"all_to_all": 700, "bw_all_to_all": 710, "sparse_all_to_all": 690, "no_broadcast": 900, "workspace": 950, "capacity_off": 999}
        self.assertEqual(rules.comparator_primary_permille(ood), 710)
        self.assertEqual(rules.comparator_primary_permille({**ood, "sparse_all_to_all": 720}), 720)

    def test_the_workspace_has_trained_iff_it_is_the_margin_above_the_floor(self):
        self.assertEqual(rules.workspace_trained(550, 450, 100), 1)
        self.assertEqual(rules.workspace_trained(549, 450, 100), 0)
        self.assertEqual(rules.workspace_trained(460, 450, 100), 0)

    def test_the_convergence_guard_counts_the_workspace_and_the_three_pairwise_comparators_only(self):
        every = {"workspace": True, "all_to_all": True, "bw_all_to_all": True, "sparse_all_to_all": True,
                 "no_broadcast": True, "competition_off": True, "capacity_off": True}
        self.assertEqual(rules.unconverged_guard_arms(every), 0)
        self.assertEqual(rules.unconverged_guard_arms({**every, "no_broadcast": False, "competition_off": False, "capacity_off": False}), 0)
        self.assertEqual(rules.unconverged_guard_arms({**every, "workspace": False}), 1)
        self.assertEqual(rules.unconverged_guard_arms({**every, "bw_all_to_all": 0, "sparse_all_to_all": 0}), 2)
        self.assertEqual(rules.unconverged_guard_arms({**every, "workspace": 0, "all_to_all": 0, "bw_all_to_all": 0, "sparse_all_to_all": 0}), 4)

    def test_the_plateau_rule_needs_patience_evaluations_without_an_improvement_of_more_than_min_delta(self):
        self.assertTrue(rules.plateau_reached([400, 400, 400, 400], 3, 5))
        self.assertFalse(rules.plateau_reached([400, 400, 400], 3, 5))  # two evaluations without improvement, not three
        self.assertTrue(rules.plateau_reached([400, 405, 405, 405], 3, 5))  # +5 is not more than 5
        self.assertFalse(rules.plateau_reached([400, 405, 405, 405], 3, 4))  # +5 is more than 4: the count restarts
        self.assertFalse(rules.plateau_reached([400, 400, 400, 420, 420, 420], 3, 5))
        self.assertTrue(rules.plateau_reached([400, 400, 400, 420, 420, 420, 420], 3, 5))
        self.assertFalse(rules.plateau_reached([], 3, 5))


if __name__ == "__main__":
    unittest.main()
