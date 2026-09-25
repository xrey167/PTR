"""The A0 ablation study's decision core, driven through every branch with
synthetic scores: each verdict, each gate failure, the leak alarm, learnability
failures and the contingency path, and an arm the budget rule dropped. The
thresholds come from the real criteria.toml, so a test here fails if the
preregistered rules and the code that applies them drift apart.
"""

import copy
import importlib.util
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("aggregate_a0_ablation", ROOT / "scripts/aggregate_a0_ablation.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)

CRITERIA = tomllib.loads((ROOT / "research/falsification/A0-ablations-v1/criteria.toml").read_text(encoding="utf-8"))
SEEDS = CRITERIA["seeds"]
SPLITS = agg.TEST_SPLITS
TAGS = ("validity", "regime", "budget", "confidence", "epistemic", "clear_margin")

REFERENCES = {
    "references": {
        "bound_additive": {"test_iid": 0.651},
        "train_majority": {"test_iid": 0.194},
        "raw_blind_exact_bayes": {"test_iid": 0.821},
        "hand_weighted": {"test_iid": 0.806},
    },
    "no_transfer": {"ood_compose_epi": {"heldout_as_inferred_on_heldout_subset": 0.69}},
}
PASSING_GATES = {name: {"pass": True, "detail": None} for name in ("G0", "G2", "G3", "G4", "G5", "G6")}
ARMS = [
    "full", "no-semantic-slots", "no-semantic-slots-masked", "no-typed-attention", "raw-blind",
    "blind-query-k0", "blind-query-k0-no-typed-attention", "latent-0", "latent-1",
    "latent-linear", "latent-4", "frozen-router",
]


def scores(level: float, jitter=(0.0,) * 5):
    """One arm: every split and subset at `level` + the seed's jitter."""
    arm = {}
    for i, seed in enumerate(SEEDS):
        value = level + jitter[i]
        splits = {}
        for split in SPLITS:
            subsets = {tag: {"accuracy": value} for tag in TAGS}
            subsets["heldout"] = {"accuracy": value}
            subsets["transfer"] = {"accuracy": value}
            splits[split] = {"accuracy": value, "subsets": subsets}
        arm[seed] = splits
    return arm


def table(**levels):
    """Every arm at 0.90 unless given, raw-blind at 0.80 (a used raw path, no leak)."""
    base = {arm: 0.90 for arm in ARMS}
    base["raw-blind"] = 0.80
    base.update({k.replace("_", "-"): v for k, v in levels.items()})
    return {arm: scores(*v) if isinstance(v, tuple) else scores(v) for arm, v in base.items()}


def run(t, gates=None, contingency=None):
    return agg.decide(t, REFERENCES, CRITERIA, gates or PASSING_GATES, contingency)


def verdict(result, cid):
    return result["verdicts"][cid]["verdict"]


SMALL = (0.001, -0.001, 0.002, -0.002, 0.0)


class CommonRule(unittest.TestCase):
    def test_supports(self):
        result = run(table(no_semantic_slots_masked=(0.85, SMALL)))
        self.assertEqual(verdict(result, "M001-primary"), "SUPPORTS")

    def test_falsifies(self):
        result = run(table(no_semantic_slots_masked=(0.899, SMALL)))
        self.assertEqual(verdict(result, "M001-primary"), "FALSIFIES")

    def test_harmful(self):
        result = run(table(no_semantic_slots_masked=(0.93, SMALL)))
        self.assertEqual(verdict(result, "M001-primary"), "HARMFUL")

    def test_inconclusive_when_noisy(self):
        result = run(table(no_semantic_slots_masked=(0.87, (0.06, -0.05, 0.04, -0.06, 0.05))))
        self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")

    def test_one_negative_seed_blocks_supports(self):
        result = run(table(no_semantic_slots_masked=(0.85, (0.0, 0.0, 0.0, 0.0, 0.051))))
        self.assertNotEqual(verdict(result, "M001-primary"), "SUPPORTS")


class ChecksAndControls(unittest.TestCase):
    def test_a_failed_raw_blind_check_makes_necessity_not_exercised(self):
        # full - raw-blind = 0.02 < 0.03 fails the check, and 0.84 stays under the
        # leak alarm (ceiling 0.821 + 0.02), so only the check decides.
        result = run(table(full=0.86, raw_blind=0.84, no_typed_attention=0.76))
        self.assertFalse(result["leak_alarm"]["active"])
        self.assertEqual(verdict(result, "M002-raw-blind-check"), "FAIL")
        self.assertEqual(verdict(result, "M002-necessity"), "NOT EXERCISED")

    def test_the_leak_alarm_outranks_not_exercised(self):
        result = run(table(raw_blind=0.895, no_typed_attention=0.80))
        self.assertTrue(result["leak_alarm"]["active"])
        self.assertIn("HOLD", result["verdicts"]["M002-necessity"]["reason"])

    def test_the_frozen_router_equivalent(self):
        result = run(table(frozen_router=(0.90, SMALL)))
        self.assertEqual(verdict(result, "M004-negative-control"), "EQUIVALENT")

    def test_the_frozen_router_harmful(self):
        result = run(table(frozen_router=(0.95, SMALL)))
        self.assertEqual(verdict(result, "M004-negative-control"), "HARMFUL")

    def test_harmful_is_recorded_even_inside_the_equivalence_band(self):
        # CI about (-0.007, -0.003): inside (-0.02, 0.02) and entirely below 0.
        # DESIGN.md records HARMFUL whenever the upper bound is < 0.
        entry = run(table(frozen_router=(0.905, SMALL)))["verdicts"]["M004-negative-control"]
        self.assertLess(entry["stats"]["ci"][1], 0)
        self.assertGreater(entry["stats"]["ci"][0], -0.02)
        self.assertEqual(entry["verdict"], "HARMFUL")
        self.assertIn("inside the equivalence band", entry["reason"])

    def test_nonlinearity_supported_and_not_attributable(self):
        both = run(table(latent_0=(0.80, SMALL), latent_linear=(0.80, SMALL)))
        self.assertEqual(verdict(both, "M003-nonlinearity"), "SUPPORTS-NONLINEARITY")
        only_zero = run(table(latent_0=(0.80, SMALL), latent_linear=(0.90, SMALL)))
        self.assertEqual(verdict(only_zero, "M003-nonlinearity"), "INCONCLUSIVE")
        self.assertIn("not attributable", only_zero["verdicts"]["M003-nonlinearity"]["reason"])

    def test_blindness_failure_is_inconclusive(self):
        result = run(table(blind_query_k0=0.95, blind_query_k0_no_typed_attention=0.90))
        self.assertEqual(verdict(result, "M002-sufficiency"), "INCONCLUSIVE")
        self.assertIn("blindness failed", result["verdicts"]["M002-sufficiency"]["reason"])

    def test_the_verifier_head_is_never_a_null(self):
        self.assertEqual(verdict(run(table()), "no-verifier-head"), "NOT TESTED")


class GatesAndPreconditions(unittest.TestCase):
    def test_a_failed_gate_blocks_every_mechanism_verdict(self):
        gates = copy.deepcopy(PASSING_GATES)
        gates["G5"]["pass"] = False
        result = run(table(no_semantic_slots_masked=(0.85, SMALL)), gates)
        self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")
        self.assertIn("G5", result["verdicts"]["M001-primary"]["reason"])

    def test_the_leak_alarm_holds_every_verdict(self):
        result = run(table(raw_blind=0.86, no_semantic_slots_masked=(0.85, SMALL)))
        self.assertTrue(result["leak_alarm"]["active"])
        self.assertIn("HOLD", result["verdicts"]["M001-primary"]["reason"])

    def test_competence_failure_issues_no_verdict(self):
        result = run(table(full=0.80))
        self.assertFalse(result["gates"]["G1"]["pass"])
        issued = {cid: v["verdict"] for cid, v in result["verdicts"].items() if cid != "no-verifier-head"}
        self.assertTrue(all(v == "NOT ISSUED" for v in issued.values()))
        # A contrast that was never testable stays NOT TESTED whatever the gates say.
        self.assertEqual(verdict(result, "no-verifier-head"), "NOT TESTED")

    def test_a_restricted_arm_that_failed_to_train(self):
        result = run(table(latent_0=0.40))
        self.assertIn("failed to train", result["verdicts"]["M003-nonlinearity"]["reason"])

    def test_the_contingency_decides_a_complete_path_arm(self):
        weak = table(no_semantic_slots_masked=(0.60, SMALL))
        waiting = run(weak)
        self.assertIn("contingency decides", waiting["verdicts"]["M001-primary"]["reason"])
        rescue = {"full": scores(0.92), "no-semantic-slots-masked": scores(0.86, SMALL)}
        decided = run(weak, contingency=rescue)
        self.assertEqual(verdict(decided, "M001-primary"), "SUPPORTS")
        self.assertEqual(decided["verdicts"]["M001-primary"]["note"], "decided at 4000 steps")

    def test_an_arm_still_below_its_bar_at_4000_steps_is_an_optimisation_failure(self):
        weak = table(no_semantic_slots_masked=(0.60, SMALL))
        still_weak = {"full": scores(0.92), "no-semantic-slots-masked": scores(0.62, SMALL)}
        result = run(weak, contingency=still_weak)
        self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")
        self.assertIn("still below its learnability bar at 4000 steps", result["verdicts"]["M001-primary"]["reason"])

    def test_the_contingency_is_reported_beside_its_bar(self):
        weak = table(no_semantic_slots=(0.60, SMALL))
        rescue = {"full": scores(0.92), "no-semantic-slots": scores(0.86, SMALL)}
        result = run(weak, contingency=rescue)
        rescued = result["reported"]["contingency"]
        self.assertEqual(rescued["steps"], CRITERIA["learnability"]["contingency_steps"])
        self.assertTrue(rescued["arms"]["no-semantic-slots"]["passes"])
        self.assertEqual(rescued["arms"]["no-semantic-slots"]["bar"], REFERENCES["references"]["bound_additive"]["test_iid"])
        self.assertEqual(len(rescued["arms"]["full"]["composite_A"]), len(SEEDS))

    def test_the_mask_share_is_labelled_with_its_budget(self):
        # The contingency decides M001-secondary but did not re-run the masked arm,
        # so the share is the one at S*, and says so; the verdict is unaffected.
        weak = table(no_semantic_slots=(0.60, SMALL), no_semantic_slots_masked=0.80)
        rescue = {"full": scores(0.92), "no-semantic-slots": scores(0.86, SMALL)}
        entry = run(weak, contingency=rescue)["verdicts"]["M001-secondary"]
        self.assertEqual(entry["note"], "decided at 4000 steps")
        self.assertAlmostEqual(entry["mask_share"], 0.20)
        self.assertEqual(entry["mask_share_budget"], "S*")
        both = dict(rescue, **{"no-semantic-slots-masked": scores(0.88)})
        entry = run(weak, contingency=both)["verdicts"]["M001-secondary"]
        self.assertAlmostEqual(entry["mask_share"], 0.02)
        self.assertEqual(entry["mask_share_budget"], "4000 steps")

    def test_an_arm_the_budget_rule_dropped(self):
        t = table()
        del t["blind-query-k0"], t["blind-query-k0-no-typed-attention"], t["latent-4"]
        result = run(t)
        self.assertEqual(verdict(result, "M002-sufficiency"), "NOT RUN")
        self.assertEqual(result["verdicts"]["M003-depth"]["dose_response"], "not run (budget rule)")


class Reported(unittest.TestCase):
    def test_the_mask_effect_names_an_arm_below_its_bar(self):
        clean = run(table(no_semantic_slots=0.80, no_semantic_slots_masked=0.85))
        self.assertEqual(clean["reported"]["mask_effect"]["below_learnability_bar"], [])
        undertrained = run(table(no_semantic_slots=0.60, no_semantic_slots_masked=0.85))
        self.assertEqual(undertrained["reported"]["mask_effect"]["below_learnability_bar"], ["no-semantic-slots"])

    def test_transfer_statements_and_the_mask_effect(self):
        result = run(table(no_semantic_slots=0.80, no_semantic_slots_masked=0.85, latent_0=0.20))
        self.assertAlmostEqual(result["reported"]["mask_effect"]["ood_validity"], 0.05)
        self.assertEqual(result["reported"]["transfer"]["full"]["new_role_regime_cell"], "transfers")
        self.assertEqual(result["reported"]["transfer"]["latent-0"]["new_role_regime_cell"], "does not transfer")


LOCK = {"data_fnv1a64": "aa", "splits": {"test_iid": {"label_fnv1a64": "bb"}}}
DATA_ROW = '{"row":"data","data_fnv64":"aa","label_fnv64_test_iid":"bb"}'


def record(seed, status="completed", stdout=DATA_ROW, sha="c" * 40, dirty=False):
    return {"seed": seed, "status": status, "stdout": stdout, "git_sha": sha, "git_dirty": dirty}


class RecordGates(unittest.TestCase):
    """The gate conditions on run records, which cover the evaluation, the G2
    rerun and the contingency alike."""

    def test_a_retry_replaces_a_failed_process_and_a_failure_never_enters_a_table(self):
        chosen = agg.chosen_per_seed([record(17, status="failed"), record(17), record(29, status="failed")])
        self.assertEqual(sorted(chosen), [17])

    def test_completeness_over_every_kind_of_process(self):
        full = {s: record(s) for s in SEEDS}
        scores_for = {arm: {s: {} for s in SEEDS} for arm in ("full", "latent-0")}
        planned = {"M001": ["full"], "M003": ["latent-0"]}
        chosen = {"eval": {"M001": full, "M003": full}, "rerun": {"M001": {17: record(17)}},
                  "contingency": {"M001": full}}
        self.assertEqual(agg.completeness(chosen, SEEDS, planned, scores_for), [])

        short = dict(full)
        del short[43]
        nan = {**full, 71: record(71, stdout=DATA_ROW + '\n{"row":"final","nan":1}')}
        problems = agg.completeness(
            {"eval": {"M001": full, "M003": nan}, "rerun": {"M001": {}}, "contingency": {"M001": short}},
            SEEDS, dict(planned, M004=["frozen-router"]), scores_for)
        self.assertIn("contingency M001/43: no completed process", problems)
        self.assertIn("eval M003/71: a non-finite loss", problems)
        self.assertIn("rerun M001/17: no completed process", problems)
        self.assertIn("eval M004: no records", problems)
        self.assertTrue(any("the budget kept frozen-router" in p for p in problems))

    def test_only_the_lr_selection_and_sweep_records_may_follow_the_tag(self):
        sweep = "experiments/model/M001-semantic-slots/results/run-1-seed-17.json"
        evaluation = "experiments/model/M002-typed-attention/results/run-2-seed-17.json"
        entrypoints = {sweep: "a0_sweep_entrypoint", evaluation: "a0_ablation_entrypoint"}
        changed = ["research/falsification/A0-ablations-v1/lr_selection.tsv", sweep, evaluation,
                   "scripts/aggregate_a0_ablation.py"]
        self.assertEqual(agg.freeze_violations(changed, entrypoints.get),
                         [evaluation, "scripts/aggregate_a0_ablation.py"])

    def test_provenance_needs_one_commit_and_a_known_clean_worktree(self):
        self.assertTrue(agg.provenance({"a": record(17), "b": record(29)})["pass"])
        self.assertFalse(agg.provenance({"a": record(17), "b": record(29, sha="d" * 40)})["pass"])
        self.assertEqual(agg.provenance({"a": record(17), "b": record(29, dirty=True)})["dirty"], ["b"])
        self.assertEqual(agg.provenance({"a": record(17, dirty=None)})["dirty"], ["a"])

    def test_data_identity_does_not_pass_vacuously(self):
        self.assertTrue(agg.data_identity({"a": record(17)}, LOCK)["pass"])
        self.assertFalse(agg.data_identity({}, LOCK)["pass"])
        self.assertEqual(agg.data_identity({"a": record(17, stdout="")}, LOCK)["records_without_data_row"], ["a"])
        other = record(17, stdout=DATA_ROW.replace('"aa"', '"ff"'))
        self.assertEqual(agg.data_identity({"a": other}, LOCK)["records_with_other_data"], ["a"])

    def test_the_rerun_must_reproduce_real_lines(self):
        lines = '{"row":"final","arm":"full","correct":3}\nPRED full test_iid 0123'
        self.assertTrue(agg.rerun_reproduces(record(17, stdout=lines), record(17, stdout=lines))["pass"])
        self.assertFalse(agg.rerun_reproduces(record(17, stdout=""), record(17, stdout=""))["pass"])
        self.assertFalse(agg.rerun_reproduces(record(17, stdout=lines), None)["pass"])
        changed = lines.replace("0123", "0124")
        self.assertFalse(agg.rerun_reproduces(record(17, stdout=lines), record(17, stdout=changed))["pass"])


if __name__ == "__main__":
    unittest.main()
