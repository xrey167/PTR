"""C007 rules that do not need torch: the hard counters, the coverage rules, the guards and the derivations of the guard scalars (so that a
unit test can pin each of them to criteria.toml and a mutation can break each). Structural counters exit 1 (FALSIFIED, kind structural),
integrity counters exit 2 (INVALID); the guards are the aggregator's."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from common import counter_rules, holds, judge, plateau_reached  # noqa: E402,F401

STRUCTURAL_COUNTERS = (
    "model_write_paths",
    "advice_leaks",
    "nondeterminism",
)

INTEGRITY_COUNTERS = (
    "capacity_mismatch_arms",
    "input_mismatches",
    "aux_weight_errors",
    "inert_treatment_arms",
    "fit_split_errors",
    "split_overlap",
    "error_rate_errors",
    "outcome_audit_mismatches",
    "metric_audit_mismatches",
    "permutation_leaks",
    "shuffle_errors",
    "noise_errors",
    "nonfinite_arms",
    "checkpoint_mismatch_arms",
    "convergence_mismatch_arms",
)

HARD_COUNTERS = STRUCTURAL_COUNTERS + INTEGRITY_COUNTERS

# Arms whose convergence the verdict depends on (counted across seeds by the aggregator).
JUDGED_ARMS = ("self_model", "probe_matched")
COMPARATORS = ("probe_matched", "own_confidence")
CONTROLS = ("split_overlap", "capacity", "input_hash", "outcome_audit", "metric_audit", "write_probe", "advice_probe", "shuffle_probe", "treatment_probe", "fit_probe")


def coverage_rules(cfg: dict) -> list[str]:
    return [
        f"arms_trained >= {cfg['min_arms_trained']}",
        f"id_eval_instances >= {cfg['min_eval_instances']}",
        f"shifted_eval_instances >= {cfg['min_eval_instances']}",
        f"null_eval_instances >= {cfg['min_eval_instances']}",
        "advice_probe_hits >= 1",
    ]


def guard_rules(cfg: dict) -> list[str]:
    """Per-seed validity guards: the traces must carry information about the model's errors on the shifted split (the best trace predictor beats the
    constant predictor by the floor), and the model must err often enough and not always on the shifted split."""
    return [
        f"signal_permille >= {cfg['signal_floor_permille']}",
        f"head_gap_id_permille <= {cfg['head_gap_tolerance_permille']}",
        f"shifted_error_permille >= {cfg['shifted_error_low_permille']}",
        f"shifted_error_permille <= {cfg['shifted_error_high_permille']}",
    ]


def falsifier_rules() -> list[str]:
    return counter_rules(STRUCTURAL_COUNTERS)


def integrity_rules() -> list[str]:
    return counter_rules(INTEGRITY_COUNTERS)


def integrity_failed(result: dict) -> list[str]:
    return [r for r in integrity_rules() if not holds(result, r)]


# -- derivations of the guard scalars ---------------------------------------------------------------

def signal(constant_brier_micro: int, best_brier_micro: int, total: int) -> int:
    """How far the best trace predictor is below the constant predictor on the shifted split, in Brier per mille (1000 x the Brier difference)."""
    return (1000 * (constant_brier_micro - best_brier_micro)) // total


def gap_permille(head_sum_micro: int, reference_sum_micro: int, total: int) -> int:
    """How far above the reference predictor's Brier the head sits (positive: the head is worse), in Brier per mille (1000 x the difference), on the
    in-distribution test split."""
    return (1000 * (head_sum_micro - reference_sum_micro)) // total


def unconverged_guard_arms(converged: dict) -> int:
    return sum(1 for arm in JUDGED_ARMS if not converged[arm])


def band_error(rate_permille: int, low_permille: int, high_permille: int) -> int:
    """1 iff the model's error rate is outside the band in which the experiment is informative."""
    return int(not low_permille <= rate_permille <= high_permille)


def agreement_error(same: int, total: int, chance_permille: int, slack_permille: int) -> int:
    """1 iff a permutation left more labels in place than chance + slack (it did not destroy the association)."""
    return int(1000 * same > (chance_permille + slack_permille) * total)


def unchanged_error(unchanged: int, total: int, ceiling_permille: int) -> int:
    """1 iff a shuffled or noised quantity equals the original in more than `ceiling_permille` of the cases (it was not changed)."""
    return int(1000 * unchanged > ceiling_permille * total)
