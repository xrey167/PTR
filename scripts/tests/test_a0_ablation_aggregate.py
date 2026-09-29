"""The A0 ablation study's decision core, driven through every branch with
synthetic scores: each verdict, each gate failure, the leak alarm, learnability
failures and the contingency path, and an arm the budget rule dropped. The
thresholds come from the real criteria.toml, so a test here fails if the
preregistered rules and the code that applies them drift apart.
"""

import copy
import contextlib
import importlib.util
import io
import json
import hashlib
import math
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
    """Apply the real decision rules to synthetic scores and optional gate or contingency overrides."""
    return agg.decide(t, REFERENCES, CRITERIA, gates or PASSING_GATES, contingency)


def verdict(result, cid):
    """Return one contrast's verdict label from an aggregate result."""
    return result["verdicts"][cid]["verdict"]


SMALL = (0.001, -0.001, 0.002, -0.002, 0.0)


class CommonRule(unittest.TestCase):
    def test_exact_decision_boundaries(self):
        """Preserve inclusive effect size and strict interval/positive-seed rules."""
        cases = [
            ("mean equals minimum", 0.02, [0.01, 0.03], 5, "SUPPORTS"),
            ("lower equals zero", 0.02, [0.0, 0.04], 5, "INCONCLUSIVE"),
            ("upper equals minimum", 0.01, [0.0, 0.02], 5, "INCONCLUSIVE"),
            ("upper below minimum", 0.01, [0.0, math.nextafter(0.02, 0.0)], 5, "FALSIFIES"),
            ("upper equals zero", -0.01, [-0.02, 0.0], 0, "FALSIFIES"),
            ("upper below zero", -0.01, [-0.02, math.nextafter(0.0, -1.0)], 0, "HARMFUL"),
            ("one zero delta", 0.02, [0.01, 0.03], 4, "INCONCLUSIVE"),
        ]
        for name, mean, ci, positives, expected in cases:
            with self.subTest(case=name):
                stats = {"mean": mean, "ci": ci, "positive_seeds": positives,
                         "deltas": [0.025] * positives + [0.0] * (5 - positives)}
                self.assertEqual(agg.common_rule(stats, 0.02, None)[0], expected)
                self.assertEqual(agg.common_rule(stats, 0.02, "G4 failed"),
                                 ("INCONCLUSIVE", "G4 failed"))

    def test_supports(self):
        """Issue SUPPORTS when the paired effect clears the preregistered minimum."""
        result = run(table(no_semantic_slots_masked=(0.85, SMALL)))
        self.assertEqual(verdict(result, "M001-primary"), "SUPPORTS")

    def test_falsifies(self):
        """Issue FALSIFIES when the paired effect remains below the required minimum."""
        result = run(table(no_semantic_slots_masked=(0.899, SMALL)))
        self.assertEqual(verdict(result, "M001-primary"), "FALSIFIES")

    def test_harmful(self):
        """Issue HARMFUL when the ablated arm reliably outperforms its comparator."""
        result = run(table(no_semantic_slots_masked=(0.93, SMALL)))
        self.assertEqual(verdict(result, "M001-primary"), "HARMFUL")

    def test_inconclusive_when_noisy(self):
        """Keep a noisy paired effect INCONCLUSIVE when its interval crosses decision bounds."""
        result = run(table(no_semantic_slots_masked=(0.87, (0.06, -0.05, 0.04, -0.06, 0.05))))
        self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")

    def test_one_negative_seed_blocks_supports(self):
        """Prevent SUPPORTS when any paired seed favors the ablation."""
        result = run(table(no_semantic_slots_masked=(0.85, (0.0, 0.0, 0.0, 0.0, 0.051))))
        self.assertNotEqual(verdict(result, "M001-primary"), "SUPPORTS")


class StatisticsAndEndpoints(unittest.TestCase):
    def test_paired_statistics_use_seedwise_differences_and_sample_deviation(self):
        """Verify interval arithmetic with exactly representable paired differences."""
        stats = agg.paired([0.75, 0.5, 0.25], [0.25, 0.25, 0.25], 2.0)
        self.assertEqual(stats["deltas"], [0.5, 0.25, 0.0])
        self.assertEqual((stats["mean"], stats["sd"]), (0.25, 0.25))
        self.assertEqual((stats["min"], stats["max"], stats["positive_seeds"]), (0.0, 0.5, 2))
        self.assertAlmostEqual(stats["ci"][0], 0.25 - 0.5 / math.sqrt(3))
        self.assertAlmostEqual(stats["ci"][1], 0.25 + 0.5 / math.sqrt(3))

    def test_one_pair_has_a_zero_width_interval(self):
        """A single observation has no measured variation and keeps its signed delta."""
        stats = agg.paired([0.25], [0.5], 2.0)
        self.assertEqual(stats, {"deltas": [-0.25], "mean": -0.25, "sd": 0.0,
                                "min": -0.25, "max": -0.25, "positive_seeds": 0,
                                "ci": [-0.25, -0.25]})

    def test_composite_weights_splits_equally_and_subset_zero_is_valid(self):
        """Split sizes cannot turn the preregistered composite into a pooled rate."""
        fixture = {"full": {17: {
            "small": {"accuracy": 1.0, "n": 1, "subsets": {"validity": {"accuracy": 0.0}}},
            "large": {"accuracy": 0.0, "n": 99},
        }}}
        criteria = {"composites": {"A": ["small", "large"]}}
        self.assertEqual(agg.endpoint(fixture, "full", 17, "composite:A", criteria), 0.5)
        self.assertEqual(agg.endpoint(fixture, "full", 17, "split:small", criteria), 1.0)
        self.assertEqual(agg.endpoint(fixture, "full", 17, "subset:small:validity", criteria), 0.0)
        fixture["full"][17]["small"]["subsets"]["validity"]["accuracy"] = None
        with self.assertRaisesRegex(ValueError, "full seed 17: subset small:validity is empty"):
            agg.endpoint(fixture, "full", 17, "subset:small:validity", criteria)
        with self.assertRaisesRegex(ValueError, "unknown endpoint"):
            agg.endpoint(fixture, "full", 17, "pooled:A", criteria)


class ChecksAndControls(unittest.TestCase):
    def test_a_failed_raw_blind_check_makes_necessity_not_exercised(self):
        """Mark attention necessity NOT EXERCISED when raw-input use is insufficient without a leak."""
        # full - raw-blind = 0.02 < 0.03 fails the check, and 0.84 stays under the
        # leak alarm (ceiling 0.821 + 0.02), so only the check decides.
        result = run(table(full=0.86, raw_blind=0.84, no_typed_attention=0.76))
        self.assertFalse(result["leak_alarm"]["active"])
        self.assertEqual(verdict(result, "M002-raw-blind-check"), "FAIL")
        self.assertEqual(verdict(result, "M002-necessity"), "NOT EXERCISED")

    def test_the_leak_alarm_outranks_not_exercised(self):
        """Give the leak alarm priority over a failed raw-blind manipulation check."""
        result = run(table(raw_blind=0.895, no_typed_attention=0.80))
        self.assertTrue(result["leak_alarm"]["active"])
        self.assertIn("HOLD", result["verdicts"]["M002-necessity"]["reason"])

    def test_the_frozen_router_equivalent(self):
        """Recognize a frozen router whose paired interval lies within the equivalence band."""
        result = run(table(frozen_router=(0.90, SMALL)))
        self.assertEqual(verdict(result, "M004-negative-control"), "EQUIVALENT")

    def test_the_frozen_router_harmful(self):
        """Mark the learned router HARMFUL when the frozen router reliably performs better."""
        result = run(table(frozen_router=(0.95, SMALL)))
        self.assertEqual(verdict(result, "M004-negative-control"), "HARMFUL")

    def test_harmful_is_recorded_even_inside_the_equivalence_band(self):
        """Preserve HARMFUL when an entirely negative interval also fits the equivalence band."""
        # CI about (-0.007, -0.003): inside (-0.02, 0.02) and entirely below 0.
        # DESIGN.md records HARMFUL whenever the upper bound is < 0.
        entry = run(table(frozen_router=(0.905, SMALL)))["verdicts"]["M004-negative-control"]
        self.assertLess(entry["stats"]["ci"][1], 0)
        self.assertGreater(entry["stats"]["ci"][0], -0.02)
        self.assertEqual(entry["verdict"], "HARMFUL")
        self.assertIn("inside the equivalence band", entry["reason"])

    def test_nonlinearity_supported_and_not_attributable(self):
        """Require both latent-zero and latent-linear comparisons to support nonlinearity."""
        both = run(table(latent_0=(0.80, SMALL), latent_linear=(0.80, SMALL)))
        self.assertEqual(verdict(both, "M003-nonlinearity"), "SUPPORTS-NONLINEARITY")
        only_zero = run(table(latent_0=(0.80, SMALL), latent_linear=(0.90, SMALL)))
        self.assertEqual(verdict(only_zero, "M003-nonlinearity"), "INCONCLUSIVE")
        self.assertIn("not attributable", only_zero["verdicts"]["M003-nonlinearity"]["reason"])

    def test_blindness_failure_is_inconclusive(self):
        """Block sufficiency claims when the blind-query manipulation fails."""
        result = run(table(blind_query_k0=0.95, blind_query_k0_no_typed_attention=0.90))
        self.assertEqual(verdict(result, "M002-sufficiency"), "INCONCLUSIVE")
        self.assertIn("blindness failed", result["verdicts"]["M002-sufficiency"]["reason"])

    def test_the_verifier_head_is_never_a_null(self):
        """Keep the absent verifier-head mechanism NOT TESTED."""
        self.assertEqual(verdict(run(table()), "no-verifier-head"), "NOT TESTED")


class GatesAndPreconditions(unittest.TestCase):
    def test_a_failed_gate_blocks_every_mechanism_verdict(self):
        """Require a failed correctness gate to block an otherwise supported mechanism claim."""
        gates = copy.deepcopy(PASSING_GATES)
        gates["G5"]["pass"] = False
        result = run(table(no_semantic_slots_masked=(0.85, SMALL)), gates)
        self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")
        self.assertIn("G5", result["verdicts"]["M001-primary"]["reason"])

    def test_the_leak_alarm_holds_every_verdict(self):
        """Hold a mechanism verdict when raw-blind accuracy activates the leak alarm."""
        result = run(table(raw_blind=0.86, no_semantic_slots_masked=(0.85, SMALL)))
        self.assertTrue(result["leak_alarm"]["active"])
        self.assertIn("HOLD", result["verdicts"]["M001-primary"]["reason"])

    def test_competence_failure_issues_no_verdict(self):
        """Withhold testable verdicts when the full model fails the competence gate."""
        result = run(table(full=0.80))
        self.assertFalse(result["gates"]["G1"]["pass"])
        issued = {cid: v["verdict"] for cid, v in result["verdicts"].items() if cid != "no-verifier-head"}
        self.assertTrue(all(v == "NOT ISSUED" for v in issued.values()))
        # A contrast that was never testable stays NOT TESTED whatever the gates say.
        self.assertEqual(verdict(result, "no-verifier-head"), "NOT TESTED")

    def test_a_restricted_arm_that_failed_to_train(self):
        """Report a restricted arm below its learnability bar as a training failure."""
        result = run(table(latent_0=0.40))
        self.assertIn("failed to train", result["verdicts"]["M003-nonlinearity"]["reason"])

    def test_the_contingency_decides_a_complete_path_arm(self):
        """Use a successful 4000-step contingency to resolve a weak complete-path arm."""
        weak = table(no_semantic_slots_masked=(0.60, SMALL))
        waiting = run(weak)
        self.assertIn("contingency decides", waiting["verdicts"]["M001-primary"]["reason"])
        rescue = {"full": scores(0.92), "no-semantic-slots-masked": scores(0.86, SMALL)}
        decided = run(weak, contingency=rescue)
        self.assertEqual(verdict(decided, "M001-primary"), "SUPPORTS")
        self.assertEqual(decided["verdicts"]["M001-primary"]["note"], "decided at 4000 steps")

    def test_an_arm_still_below_its_bar_at_4000_steps_is_an_optimisation_failure(self):
        """Keep the contrast INCONCLUSIVE if the contingency arm still fails its learnability bar."""
        weak = table(no_semantic_slots_masked=(0.60, SMALL))
        still_weak = {"full": scores(0.92), "no-semantic-slots-masked": scores(0.62, SMALL)}
        result = run(weak, contingency=still_weak)
        self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")
        self.assertIn("still below its learnability bar at 4000 steps", result["verdicts"]["M001-primary"]["reason"])

    def test_contingency_rechecks_previously_passing_comparator(self):
        weak = table(no_semantic_slots_masked=(0.60, SMALL))
        regressed = {"full": scores(0.40), "no-semantic-slots-masked": scores(0.86, SMALL)}
        result = run(weak, contingency=regressed)
        self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")
        self.assertIn("full is still below", result["verdicts"]["M001-primary"]["reason"])

    def test_incomplete_scores_produce_a_gated_report_without_statistics(self):
        for missing in ("all", "seed", "split"):
            incomplete = table()
            if missing == "all":
                incomplete = {}
            elif missing == "seed":
                incomplete["full"].pop(CRITERIA["seeds"][0])
            else:
                incomplete["full"][CRITERIA["seeds"][0]].pop("test_iid")
            result = run(incomplete)
            self.assertFalse(result["gates"]["G3"]["pass"])
            self.assertEqual(verdict(result, "M001-primary"), "INCONCLUSIVE")

    def test_undertrained_raw_blind_arm_cannot_pass_manipulation(self):
        result = run(table(raw_blind=0.10))
        check = next(c["id"] for c in CRITERIA["contrast"] if c["kind"] == "manipulation-check")
        self.assertEqual(verdict(result, check), "INCONCLUSIVE")
        self.assertIn("failed to train", result["verdicts"][check]["reason"])

    def test_the_contingency_is_reported_beside_its_bar(self):
        """Report contingency steps, learnability thresholds, and per-seed composite scores together."""
        weak = table(no_semantic_slots=(0.60, SMALL))
        rescue = {"full": scores(0.92), "no-semantic-slots": scores(0.86, SMALL)}
        result = run(weak, contingency=rescue)
        rescued = result["reported"]["contingency"]
        self.assertEqual(rescued["steps"], CRITERIA["learnability"]["contingency_steps"])
        self.assertTrue(rescued["arms"]["no-semantic-slots"]["passes"])
        self.assertEqual(rescued["arms"]["no-semantic-slots"]["bar"], REFERENCES["references"]["bound_additive"]["test_iid"])
        self.assertEqual(len(rescued["arms"]["full"]["composite_A"]), len(SEEDS))

    def test_the_mask_share_is_labelled_with_its_budget(self):
        """Label mask-share estimates with the budget used by the available arm comparisons."""
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
        """Report dropped arms as NOT RUN instead of assigning a mechanism verdict."""
        t = table()
        del t["blind-query-k0"], t["blind-query-k0-no-typed-attention"], t["latent-4"]
        result = run(t)
        self.assertEqual(verdict(result, "M002-sufficiency"), "NOT RUN")
        self.assertEqual(result["verdicts"]["M003-depth"]["dose_response"], "not run (budget rule)")


class Reported(unittest.TestCase):
    def test_the_mask_effect_names_an_arm_below_its_bar(self):
        """Identify undertrained arms that limit interpretation of the mask comparison."""
        clean = run(table(no_semantic_slots=0.80, no_semantic_slots_masked=0.85))
        self.assertEqual(clean["reported"]["mask_effect"]["below_learnability_bar"], [])
        undertrained = run(table(no_semantic_slots=0.60, no_semantic_slots_masked=0.85))
        self.assertEqual(undertrained["reported"]["mask_effect"]["below_learnability_bar"], ["no-semantic-slots"])

    def test_transfer_statements_and_the_mask_effect(self):
        """Derive mask deltas and transfer statements from the synthetic arm scores."""
        result = run(table(no_semantic_slots=0.80, no_semantic_slots_masked=0.85, latent_0=0.20))
        self.assertAlmostEqual(result["reported"]["mask_effect"]["ood_validity"], 0.05)
        self.assertEqual(result["reported"]["transfer"]["full"]["new_role_regime_cell"], "transfers")
        self.assertEqual(result["reported"]["transfer"]["latent-0"]["new_role_regime_cell"], "does not transfer")


LOCK = {"data_fnv1a64": "aa", "splits": {"test_iid": {"label_fnv1a64": "bb"}}}
DATA_ROW = '{"row":"data","data_fnv64":"aa","label_fnv64_test_iid":"bb"}'


def record(seed, status="completed", stdout=DATA_ROW, sha="c" * 40, dirty=False):
    """Build a synthetic run record with configurable outcome, stdout, and Git provenance."""
    return {"schema_version": 2, "git_tracked_diff_sha256": __import__("hashlib").sha256(b"").hexdigest(), "seed": seed, "status": status, "stdout": stdout, "git_sha": sha, "git_dirty": dirty}


class RecordGates(unittest.TestCase):
    """The gate conditions on run records, which cover the evaluation, the G2
    rerun and the contingency alike."""

    def test_clean_flag_requires_versioned_empty_diff_evidence(self):
        for field, value in [("schema_version", None), ("schema_version", True),
                             ("git_tracked_diff_sha256", None), ("git_tracked_diff_sha256", "f" * 64)]:
            invalid = record(17)
            invalid[field] = value
            self.assertFalse(agg.provenance({"invalid": invalid})["pass"])

    def test_other_experiment_contingency_requires_full_comparator_at_every_seed(self):
        chosen = {"contingency": {"M002": {(17, "no-typed-attention"): {
            "seed": 17, "parameters": {"arms": "no-typed-attention"}, "stdout": ""}}}}
        problems = agg.completeness(chosen, [17, 29], {}, {})
        self.assertIn("contingency M001/full/17: missing paired comparator process", problems)
        self.assertIn("contingency M001/full/29: missing paired comparator process", problems)

    def test_a_retry_replaces_a_failed_process_and_a_failure_never_enters_a_table(self):
        """Select successful retries while excluding seeds with only failed processes."""
        chosen = agg.chosen_per_seed([record(17, status="failed"), record(17), record(29, status="failed")])
        self.assertEqual(sorted(chosen), [17])

    def test_later_failures_do_not_displace_the_last_completed_retry(self):
        """Select the last success for each seed independently of later failures."""
        first, last, other = record(17, stdout="first"), record(17, stdout="retry"), record(29)
        chosen = agg.chosen_per_seed([first, other, last, record(17, status="failed")])
        self.assertEqual(chosen, {17: last, 29: other})

    def test_rerun_ignores_other_arms_but_requires_full_arm_line_order(self):
        """Compare only full-arm evidence, without sorting or dropping duplicate lines."""
        lines = ['{"arm":"full","accuracy":0.75}', 'PRED full test_iid 012']
        original = record(17, stdout="\n".join(lines))
        noisy = record(17, stdout="progress\n" + "\n".join(lines)
                       + '\nPRED raw-blind test_iid 999\n{"arm":"latent-0","accuracy":0.1}')
        self.assertEqual(agg.rerun_reproduces(original, noisy), {"full_lines": 2, "pass": True})
        for changed in (list(reversed(lines)), lines + [lines[1]], [lines[0]],
                        [lines[0], 'PRED full test_iid 013']):
            with self.subTest(lines=changed):
                self.assertFalse(agg.rerun_reproduces(original, record(17, stdout="\n".join(changed)))["pass"])

    def test_completeness_over_every_kind_of_process(self):
        """Check missing seeds, missing arms, and divergence across evaluation, rerun, and contingency."""
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
        self.assertIn("contingency M001//43: missing or non-finite process", problems)
        self.assertIn("eval M003/71: a non-finite loss", problems)
        self.assertIn("rerun M001/17: no completed process", problems)
        self.assertIn("eval M004: no records", problems)
        self.assertTrue(any("the budget kept frozen-router" in p for p in problems))

    def test_only_the_lr_selection_and_sweep_records_may_follow_the_tag(self):
        """Reject post-freeze changes outside learning-rate selection and sweep records."""
        sweep = "experiments/model/M001-semantic-slots/results/run-1-seed-17.json"
        evaluation = "experiments/model/M002-typed-attention/results/run-2-seed-17.json"
        entrypoints = {sweep: "a0_sweep_entrypoint", evaluation: "a0_ablation_entrypoint"}
        changed = ["research/falsification/A0-ablations-v1/lr_selection.tsv", sweep, evaluation,
                   "scripts/aggregate_a0_ablation.py"]
        self.assertEqual(agg.freeze_violations(changed, entrypoints.get),
                         [evaluation, "scripts/aggregate_a0_ablation.py"])

    def test_provenance_needs_one_commit_and_a_known_clean_worktree(self):
        """Reject mixed commits and dirty or unknown tracked worktree states."""
        self.assertTrue(agg.provenance({"a": record(17), "b": record(29)})["pass"])
        self.assertFalse(agg.provenance({"a": record(17), "b": record(29, sha="d" * 40)})["pass"])
        self.assertEqual(agg.provenance({"a": record(17), "b": record(29, dirty=True)})["dirty"], ["b"])
        self.assertEqual(agg.provenance({"a": record(17, dirty=None)})["dirty"], ["a"])

    def test_data_identity_does_not_pass_vacuously(self):
        """Fail data identity for absent records, missing data rows, or mismatched digests."""
        self.assertTrue(agg.data_identity({"a": record(17)}, LOCK)["pass"])
        self.assertFalse(agg.data_identity({}, LOCK)["pass"])
        self.assertEqual(agg.data_identity({"a": record(17, stdout="")}, LOCK)["records_without_data_row"], ["a"])
        other = record(17, stdout=DATA_ROW.replace('"aa"', '"ff"'))
        self.assertEqual(agg.data_identity({"a": other}, LOCK)["records_with_other_data"], ["a"])

    def test_the_rerun_must_reproduce_real_lines(self):
        """Require nonempty, byte-identical full-arm output for a successful rerun gate."""
        lines = '{"row":"final","arm":"full","correct":3}\nPRED full test_iid 0123'
        self.assertTrue(agg.rerun_reproduces(record(17, stdout=lines), record(17, stdout=lines))["pass"])
        self.assertFalse(agg.rerun_reproduces(record(17, stdout=""), record(17, stdout=""))["pass"])
        self.assertFalse(agg.rerun_reproduces(record(17, stdout=lines), None)["pass"])
        changed = lines.replace("0123", "0124")
        self.assertFalse(agg.rerun_reproduces(record(17, stdout=lines), record(17, stdout=changed))["pass"])

    def test_an_identical_rerun_of_another_seed_does_not_satisfy_preregistration(self):
        lines = '{"row":"final","arm":"full","correct":3}\nPRED full test_iid 0123'
        self.assertFalse(agg.rerun_reproduces(record(29, stdout=lines), record(29, stdout=lines))["pass"])
        for reruns in ({}, {"M001": {29: record(29)}}, {"M001": {17: record(17), 29: record(29)}}):
            with self.subTest(reruns=reruns):
                problems = agg.completeness({"rerun": reruns}, SEEDS, {}, {})
                self.assertTrue(any("rerun M001" in problem for problem in problems))

    def test_scoring_data_must_match_the_locked_data_for_every_phase(self):
        self.assertTrue(agg.scored_data_identity({"eval": "aa", "rerun": "aa", "contingency": "aa"}, LOCK))
        for fingerprints in ({}, {"eval": None}, {"eval": "ff"}, {"eval": "aa", "rerun": "ff"},
                             {"eval": "aa", "contingency": "ff"}):
            with self.subTest(fingerprints=fingerprints):
                self.assertFalse(agg.scored_data_identity(fingerprints, LOCK))
        self.assertFalse(agg.scored_data_identity({"eval": None}, {}))

    def test_build_table_preserves_the_fingerprint_of_the_data_actually_scored(self):
        scorer = Mock(score_records=Mock(return_value=({"data_fnv1a64": "changed", "results": []}, [])))
        table_, problems, entries, fingerprint = agg.build_table([], scorer)
        self.assertEqual((table_, problems, entries, fingerprint), ({}, [], [], "changed"))


class FrozenInputs(unittest.TestCase):
    def test_each_decision_input_is_compared_with_its_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "research/falsification/A0-ablations-v1"
            paths = [study / name for name in ("criteria.toml", "references.json", "budget.json", "PREREGISTRATION.md", "DESIGN.md")]
            lock = root / "benchmarks/operator-routing/splits.lock.json"
            paths += [lock, root / "datasets/generated/codebook.json"]
            frozen = {}
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen")
                frozen[path.relative_to(root).as_posix()] = b"frozen"
            with patch.multiple(agg, ROOT=root, STUDY_DIR=study, CRITERIA=paths[0], LOCK=lock,
                                at_tag=Mock(side_effect=frozen.get)):
                self.assertEqual(agg.frozen_input_violations(), [])
                for path in paths:
                    with self.subTest(path=path.name):
                        path.write_bytes(b"edited after evaluation")
                        self.assertEqual(agg.frozen_input_violations(), [str(path.relative_to(root))])
                        path.unlink()
                        self.assertEqual(agg.frozen_input_violations(), [str(path.relative_to(root))])
                        path.write_bytes(b"frozen")

    def test_a_clone_needs_no_tag_but_a_conflicting_tag_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
            git("init")
            (root / "frozen.txt").write_text("frozen", encoding="utf-8")
            git("add", "frozen.txt")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "freeze")
            commit = git("rev-parse", "HEAD")
            with patch.multiple(agg, ROOT=root, PREREG_COMMIT=commit):
                self.assertEqual(agg.preregistration_ref(), commit)
                self.assertEqual(agg.at_tag("frozen.txt"), b"frozen")
                git("tag", agg.PREREG_TAG)
                self.assertEqual(agg.preregistration_ref(), commit)
                git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "later")
                git("tag", "-f", agg.PREREG_TAG)
                with self.assertRaisesRegex(ValueError, "not frozen commit"):
                    agg.preregistration_ref()


class CorrectnessEvidence(unittest.TestCase):
    def test_legacy_stale_partial_or_changed_test_evidence_cannot_pass_g6(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            sha = "c" * 40
            commands = {
                "cargo_test": ["cargo", "+1.95.0", "test", "--locked", "--manifest-path", "model/burn-a0/Cargo.toml"],
                "self_test": ["/repo/model/burn-a0/target-a0-study/release/examples/a0_ablation", "--phase", "self-test",
                              "--data", "datasets/generated/operator_routing_v1", "--data-fnv64", "aa"],
                "benchmark_tests": ["python3", "-m", "unittest", "discover", "-s", "benchmarks/operator-routing/tests"],
                "aggregator_tests": ["python3", "-m", "unittest", "scripts/tests/test_a0_ablation_aggregate.py"],
                "config_tests": ["python3", "-m", "unittest", "scripts/tests/test_a0_ablation_config.py"],
            }
            evidence = {"schema_version": 1, "git_sha": sha, "git_sha_after": sha,
                        "git_dirty": False, "git_dirty_after": False, "checks": {}}
            for name, command in commands.items():
                (logs / f"g6-{name}.log").write_bytes(b"tests passed\n")
                evidence["checks"][name] = {"argv": command, "exit_code": 0,
                                            "log_sha256": hashlib.sha256(b"tests passed\n").hexdigest()}
            self.assertTrue(agg.correctness_evidence(evidence, sha, logs, "aa")["pass"])
            for change in (
                lambda e: e.update(git_sha="d" * 40),
                lambda e: e.update(git_sha_after="d" * 40),
                lambda e: e.update(git_dirty=True),
                lambda e: e.update(git_dirty_after=True),
                lambda e: e["checks"].pop("self_test"),
                lambda e: e["checks"].update(extra={}),
                lambda e: e["checks"]["cargo_test"].update(exit_code=1),
                lambda e: e["checks"]["self_test"].update(argv=["true"]),
                lambda e: e["checks"]["benchmark_tests"].update(log_sha256="0" * 64),
            ):
                invalid = copy.deepcopy(evidence)
                change(invalid)
                self.assertFalse(agg.correctness_evidence(invalid, sha, logs, "aa")["pass"])
            self.assertFalse(agg.correctness_evidence({name: "pass" for name in commands}, sha, logs, "aa")["pass"])
            self.assertFalse(agg.correctness_evidence(evidence, None, logs, "aa")["pass"])
            (logs / "g6-cargo_test.log").write_bytes(b"edited")
            self.assertFalse(agg.correctness_evidence(evidence, sha, logs, "aa")["pass"])
            (logs / "g6-cargo_test.log").unlink()
            self.assertFalse(agg.correctness_evidence(evidence, sha, logs, "aa")["pass"])


class StockCrossCheck(unittest.TestCase):
    def test_load_records_keeps_all_study_phases_and_failures_in_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "experiments/model/fixture/results"
            results.mkdir(parents=True)
            entries = {
                "run-3.json": {"entrypoint": "a0_contingency_entrypoint", "status": "completed"},
                "run-2.json": {"entrypoint": "a0_rerun_entrypoint", "status": "failed"},
                "run-1.json": {"entrypoint": "a0_ablation_entrypoint", "status": "completed"},
                "run-0.json": {"entrypoint": "a0_sweep_entrypoint", "status": "completed"},
                "aggregate-1.json": {"entrypoint": "a0_ablation_entrypoint"},
            }
            for name, entry in entries.items():
                (results / name).write_text(json.dumps(entry), encoding="utf-8")
            with patch.multiple(agg, ROOT=root, EXPERIMENTS={"M001": "model/fixture"}):
                loaded = agg.load_records("M001")
            self.assertEqual([p.name for p in loaded], ["run-1.json", "run-2.json", "run-3.json"])
            self.assertEqual(list(loaded.values()), [entries[f"run-{i}.json"] for i in (1, 2, 3)])

    def test_missing_scores_are_reported_and_complete_scores_are_compared(self):
        complete = {"full": scores(0.9)}
        missing_seed = copy.deepcopy(complete)
        del missing_seed["full"][SEEDS[0]]
        missing_split = copy.deepcopy(complete)
        del missing_split["full"][SEEDS[0]][SPLITS[0]]
        cases = [("no table", None, 0.9, ": missing scores"),
                 ("missing arm", {}, 0.9, ": missing scores"),
                 ("missing seed", missing_seed, 0.9, ": missing scores"),
                 ("missing split", missing_split, 0.9, ": missing scores"),
                 ("matching", complete, 0.9, None),
                 ("mismatch", complete, 0.8, "")]
        for kind in ("eval", "contingency"):
            for name, score_table, stock_mean, suffix in cases:
                with self.subTest(kind=kind, case=name), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    study = root / "study"
                    study.mkdir()
                    criteria = agg.CRITERIA.read_bytes()
                    (study / "criteria.toml").write_bytes(criteria)
                    (study / "references.json").write_text('{"all_bands_pass": true}')
                    (study / "budget.json").write_text('{"arms": {}}')
                    (study / "lock.json").write_text('{"data_fnv1a64": "aa"}')
                    (study / "PREREGISTRATION.md").write_bytes(b"preregistered")
                    results = root / "experiments/model/fixture/results"
                    results.mkdir(parents=True)
                    run_path = results / "run-1.json"
                    run_path.write_text(json.dumps({
                        **record(SEEDS[0], stdout=""),
                        "entrypoint": agg.STUDY_KINDS[kind],
                        "parameters": {"arms": "full"},
                    }))
                    stock_path = results / "stock.json"
                    stock_path.write_text(json.dumps({"groups": [
                        {"key": {"row": "meta"}},
                        {"key": {"row": "final", "arm": "full", "split": SPLITS[0]},
                         "metrics": {"accuracy": {"mean": stock_mean}}},
                        {"key": {"row": "final", "arm": "absent", "split": SPLITS[0]},
                         "metrics": {"accuracy": {"mean": 0.9}}},
                    ]}))
                    with patch.multiple(
                        agg, ROOT=root, STUDY_DIR=study, CRITERIA=study / "criteria.toml",
                        LOCK=study / "lock.json", EXPERIMENTS={"M001": "model/fixture"},
                        at_tag=Mock(side_effect=lambda p: criteria if p.endswith("criteria.toml") else b"preregistered"),
                        load_score_module=Mock(return_value=Mock(agree=Mock(return_value=(True, [])))),
                        build_table=Mock(return_value=(score_table, [], [], "aa")),
                        preregistration_ref=Mock(return_value="c" * 40),
                        frozen_input_violations=Mock(return_value=[]),
                        sweep_evidence=Mock(return_value={"pass": True}),
                        decide=Mock(return_value={"verdicts": {}}),
                    ), patch.object(agg.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 0, stdout=str(stock_path), stderr=""
                    )), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(agg.main([]), 0)
                    check = json.loads((study / "results.json").read_text())["stock_aggregate_cross_check"]
                    prefix = f"M001/{agg.STUDY_KINDS[kind]}"
                    expected = [] if suffix is None else [f"{prefix}/full/{SPLITS[0]}{suffix}"]
                    expected.append(f"{prefix}/absent/{SPLITS[0]}: missing scores")
                    self.assertEqual(check, {"pass": False, "mismatches": expected})



class ContingencyRetention(unittest.TestCase):
    def test_separate_arms_and_retries_are_retained_by_seed_and_arm_set(self):
        first = {"seed": 17, "status": "completed", "parameters": {"arms": "full,a"}}
        second = {"seed": 17, "status": "completed", "parameters": {"arms": "full,b"}}
        failed = {**first, "status": "failed"}
        retry = {**first, "stdout": "retry"}
        chosen = agg.chosen_contingencies([first, second, failed, retry])
        self.assertEqual(len(chosen), 2)
        self.assertIn(second, chosen.values())
        self.assertIn(retry, chosen.values())

    def test_each_contingency_arm_set_requires_every_seed(self):
        runs = [{"seed": s, "status": "completed", "parameters": {"arms": arms}}
                for s, arms in [(17, "full,a"), (29, "full,a"), (17, "full,b")]]
        chosen = {"contingency": {"M001": agg.chosen_contingencies(runs)}, "rerun": {"M001": {17: {"seed": 17}}}}
        problems = agg.completeness(chosen, [17, 29], {}, {})
        self.assertEqual(problems, ["contingency M001/full,b/29: missing or non-finite process"])

    def test_repeated_comparators_must_have_identical_scored_evidence(self):
        entry = {"arm": "full", "seed": 17, "split": "test_iid", "record": "one", "route_accuracy": 0.9}
        for accuracy, allowed in [(0.9, True), (0.8, False)]:
            with self.subTest(accuracy=accuracy):
                scorer = Mock(score_records=Mock(return_value=({"results": [entry, {**entry, "record": "two", "route_accuracy": accuracy}], "data_fnv1a64": "aa"}, [])))
                _, problems, _, _ = agg.build_table([], scorer, repeated_comparators=True)
                self.assertEqual(not problems, allowed)

if __name__ == "__main__":
    unittest.main()
