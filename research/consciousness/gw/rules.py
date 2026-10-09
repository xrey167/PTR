"""C001 rules that do not need torch: the hard counters, the coverage rules, the validity guards.

The harness judges coverage and the hard counters itself (exit 1 when one fails: a defect of the
product arm, the task or the harness). The guards are NOT judged by the harness: a seed whose
strongest comparator is saturated or floored says nothing about the claim, and the aggregator turns
that into INVALID, never FALSIFIED. criteria.toml states the same rules; a unit test pins them."""

from __future__ import annotations

HARD_COUNTERS = (
    "param_mismatch_arms",
    "no_comm_leaks",
    "no_broadcast_leaks",
    "bottleneck_violations",
    "dropped_module_leaks",
    "n_leaks",
    "train_eval_overlap",
    "nondeterminism",
    "nonfinite_arms",
    "no_comm_above_chance",
)


def coverage_rules(cfg: dict) -> list[str]:
    """Coverage rules in the grammar of criteria.toml ("<counter> >= <integer>"). A rule that fails makes the
    seed vacuous (counts as falsified): an arm that was not trained, an evaluation set that was too small, or a
    probe that could not see the influence it exists to rule out."""
    return [
        f"arms_trained >= {cfg['min_arms_trained']}",
        f"id_eval_instances >= {cfg['min_eval_instances']}",
        f"ood16_eval_instances >= {cfg['min_eval_instances']}",
        f"ood24_eval_instances >= {cfg['min_eval_instances']}",
        "workspace_influence_hits >= 1",
        "pairwise_influence_hits >= 1",
        "no_broadcast_reporter_hits >= 1",
    ]


def guard_rules(cfg: dict) -> list[str]:
    """Validity guards, judged by the aggregator: a seed that fails one is uninformative (INVALID)."""
    return [
        f"comparator_primary_permille >= {cfg['guard_floor_permille']}",
        f"comparator_primary_permille <= {cfg['guard_ceiling_permille']}",
    ]


def falsifier_rules() -> list[str]:
    return [f"{counter} <= 0" for counter in HARD_COUNTERS]


def holds(result: dict, rule: str) -> bool:
    left, op, right = rule.split(" ")
    a, b = result[left], int(right)
    return a >= b if op == ">=" else a <= b


def judge(result: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """(coverage rules that fail, falsifier rules that fail)."""
    return (
        [r for r in coverage_rules(cfg) if not holds(result, r)],
        [r for r in falsifier_rules() if not holds(result, r)],
    )
