"""C004 rules that do not need torch: the hard counters, the coverage rules, the guards and the derivations of the guard scalars (so
that a unit test can pin each of them to criteria.toml and a mutation can break each).

Two kinds of counter, as in C001:

* STRUCTURAL counters: a defect of the product arm or of an isolated ablation. A seed that trips one exits 1 (FALSIFIED, kind
  structural).
* INTEGRITY counters: a defect of the task or the harness. The seed exits 2 and the experiment is INVALID, never FALSIFIED.

The guards are not judged by the harness: they say whether the seed was informative, and the aggregator turns a failing one into
INVALID."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from common import counter_rules, holds, judge, plateau_reached  # noqa: E402,F401

STRUCTURAL_COUNTERS = (
    "reset_carry_leaks",
    "ff_causal_leaks",
    "nondeterminism",
)

INTEGRITY_COUNTERS = (
    "param_mismatch_arms",
    "flop_mismatch_arms",
    "length_leaks",
    "inlength_control_errors",
    "train_eval_overlap",
    "val_eval_overlap",
    "nonfinite_arms",
    "checkpoint_mismatch_arms",
    "convergence_mismatch_arms",
    "label_mismatches",
    "reset_above_chance",
    "perm_label_leaks",
    "corruption_rate_errors",
)

HARD_COUNTERS = STRUCTURAL_COUNTERS + INTEGRITY_COUNTERS

# The arms whose convergence the verdict depends on: the two arms compared on the primary metric.
JUDGED_ARMS = ("recurrent", "feedforward")
CONTROLS = ("length_leak", "inlength_probe", "overlap", "param_mismatch", "flop_mismatch", "label_check", "isolation_probe", "causal_probe")


def coverage_rules(cfg: dict) -> list[str]:
    """Coverage rules in the grammar of criteria.toml. A rule that fails makes the seed vacuous (counts as falsified): an arm that was
    not trained, an evaluation set smaller than pinned, a probe that could not see the influence it exists to rule out."""
    return [
        f"arms_trained >= {cfg['min_arms_trained']}",
        f"id_eval_instances >= {cfg['min_eval_instances']}",
        f"primary_eval_instances >= {cfg['min_eval_instances']}",
        f"secondary_eval_instances >= {cfg['min_eval_instances']}",
        "recurrent_carry_hits >= 1",
    ]


def guard_rules(cfg: dict) -> list[str]:
    """Per-seed validity guards (judged by the aggregator): the feed-forward comparator on the primary metric must leave the
    margin room above it (not saturated). Its being trained and the arms' convergence are counts across seeds."""
    return [f"comparator_primary_permille <= {cfg['guard_ceiling_permille']}"]


def falsifier_rules() -> list[str]:
    return counter_rules(STRUCTURAL_COUNTERS)


def integrity_rules() -> list[str]:
    return counter_rules(INTEGRITY_COUNTERS)


def integrity_failed(result: dict) -> list[str]:
    return [r for r in integrity_rules() if not holds(result, r)]


# -- derivations of the guard scalars (pure; the harness uses them, the unit tests pin each) ----------

def trained(id_permille: int, floor_permille: int) -> int:
    """1 iff an arm learned the training distribution: its in-distribution accuracy reaches the floor."""
    return int(id_permille >= floor_permille)


def unconverged_guard_arms(converged: dict) -> int:
    """How many of the judged arms stopped at the step cap instead of by patience. `converged` maps an arm name to a truthy flag."""
    return sum(1 for arm in JUDGED_ARMS if not converged[arm])


def above_chance(accuracy_permille: int, chance_permille: int, slack_permille: int) -> int:
    """1 iff an arm that sees only the last step beats the chance rate by more than the slack (the task leaks its answer)."""
    return int(accuracy_permille > chance_permille + slack_permille)


def rate_error(changed: int, total: int, rate_permille: int, slack_permille: int) -> int:
    """1 iff the measured fraction of corrupted sequences differs from the pinned rate by more than the slack (per mille)."""
    return int(abs(1000 * changed - rate_permille * total) > slack_permille * total)


def label_leak(same: int, total: int, chance_permille: int, slack_permille: int) -> int:
    """1 iff a label permutation left more than chance + slack of the labels where they were (it did not destroy the association)."""
    return int(1000 * same > (chance_permille + slack_permille) * total)
