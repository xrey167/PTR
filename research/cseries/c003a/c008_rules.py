"""C008 rules that do not need torch: the hard counters, the coverage rules, the guards and the derivations of the guard scalars (so that a
unit test can pin each of them to criteria.toml and a mutation can break each). Structural counters exit 1 (FALSIFIED, kind structural),
integrity counters exit 2 (INVALID); the guards are the aggregator's."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from common import counter_rules, holds, judge, plateau_reached  # noqa: E402,F401

STRUCTURAL_COUNTERS = (
    "describe_only_influence",
    "authority_leaks",
    "nondeterminism",
)

INTEGRITY_COUNTERS = (
    "param_mismatch_arms",
    "flop_mismatch_arms",
    "history_router_blind",
    "stream_overlap",
    "reward_audit_mismatches",
    "oracle_violations",
    "schedule_errors",
    "nonfinite_arms",
    "checkpoint_mismatch_arms",
    "convergence_mismatch_arms",
    "epsilon_rate_errors",
    "shuffle_errors",
)

HARD_COUNTERS = STRUCTURAL_COUNTERS + INTEGRITY_COUNTERS

# Arms the verdict depends on: the control and the comparators it is judged against. Their convergence is a count across seeds.
JUDGED_ARMS = ("control", "schema_free", "schema_free_wide", "heuristic", "describe_only", "history_router")
COMPARATORS = ("schema_free", "schema_free_wide", "heuristic", "describe_only", "history_router")
CONTROLS = ("stream_overlap", "param_mismatch", "flop_mismatch", "reward_audit", "authority_probe", "influence_probe", "describe_probe", "history_probe")


def coverage_rules(cfg: dict) -> list[str]:
    return [
        f"arms_trained >= {cfg['min_arms_trained']}",
        f"eval_instances >= {cfg['min_eval_instances']}",
        f"null_eval_instances >= {cfg['min_eval_instances']}",
        "control_influence_hits >= 1",
    ]


def guard_rules(cfg: dict) -> list[str]:
    """Per-seed validity guards: the best comparator must leave the margin room below the oracle (the margin is reachable), and the
    best comparator must beat a uniform router by the floor (the comparators are trained)."""
    return [
        f"headroom_permille >= {cfg['headroom_floor_permille']}",
        f"comparator_above_random_permille >= {cfg['comparator_floor_permille']}",
    ]


def falsifier_rules() -> list[str]:
    return counter_rules(STRUCTURAL_COUNTERS)


def integrity_rules() -> list[str]:
    return counter_rules(INTEGRITY_COUNTERS)


def integrity_failed(result: dict) -> list[str]:
    return [r for r in integrity_rules() if not holds(result, r)]


# -- derivations of the guard scalars -----------------------------------------------------------------

def headroom(oracle_permille: int, comparator_permille: int) -> int:
    """How far the best comparator is below the oracle, per mille."""
    return oracle_permille - comparator_permille


def gain_over_persistence(l1_schema: float, l1_persistence: float) -> int:
    """How much closer (in L1 distance, per mille) the schema's prediction of the router's next propensity is than "the next batch is like the last"."""
    if not math.isfinite(l1_schema):
        return -1000
    return int(round(1000 * (l1_persistence - l1_schema) / l1_persistence)) if l1_persistence > 0 else 0


def schema_trained(gain_permille: int, floor_permille: int) -> int:
    """1 iff the schema beat persistence by the floor: it learned more of the router's own load dynamics than "no change"."""
    return int(gain_permille >= floor_permille)


def unconverged_guard_arms(converged: dict) -> int:
    return sum(1 for arm in JUDGED_ARMS if not converged[arm])


def rate_error(changed: int, total: int, rate_permille: int, slack_permille: int) -> int:
    """1 iff the measured fraction of re-routed tokens differs from the pinned rate by more than the slack (per mille)."""
    return int(abs(1000 * changed - rate_permille * total) > slack_permille * total)


def shuffle_error(same: int, total: int, ceiling_permille: int) -> int:
    """1 iff a shuffled history equals the real one in more than `ceiling_permille` of the cases (it was not shuffled)."""
    return int(1000 * same > ceiling_permille * total)
