"""C001 rules that do not need torch: the hard counters, the coverage rules, the validity guards and the derivations
of the guard scalars (so that a unit test can pin each of them).

The harness judges coverage and the counters itself. Two kinds of counter:

* STRUCTURAL counters: a defect of the product arm or of an isolated ablation. A seed that trips one exits 1 and the
  experiment is FALSIFIED, kind structural.
* INTEGRITY counters: a defect of the task or of the harness (a wrong module count in a training batch, an evaluation
  instance seen in training or in validation, arms whose parameter counts were not matched, a non-finite loss, a restored
  checkpoint that is not the best one, a convergence flag that disagrees with the learning curve). The seed exits 2 (harness
  integrity) and the experiment is INVALID, never FALSIFIED: the repair is a successor, and it says nothing about the product.

The guards are NOT judged by the harness: a seed whose strongest comparator is saturated or floored says nothing about the
claim, and the aggregator turns that into INVALID. The convergence guard is judged across seeds by the aggregator
(`max_unconverged_seeds`). criteria.toml states the same rules; a unit test pins them."""

from __future__ import annotations

STRUCTURAL_COUNTERS = (
    "no_comm_leaks",
    "no_broadcast_leaks",
    "bottleneck_violations",
    "dropped_module_leaks",
    "nondeterminism",
    "no_comm_above_chance",
)

INTEGRITY_COUNTERS = (
    "param_mismatch_arms",
    "n_leaks",
    "train_eval_overlap",
    "nonfinite_arms",
    "val_eval_overlap",
    "checkpoint_mismatch_arms",
    "convergence_mismatch_arms",
)

HARD_COUNTERS = STRUCTURAL_COUNTERS + INTEGRITY_COUNTERS

# The arms the verdict depends on: the three pairwise comparators the support and refutation rules look at, and the workspace.
PAIRWISE_COMPARATORS = ("all_to_all", "bw_all_to_all", "sparse_all_to_all")
CONVERGENCE_GUARD_ARMS = ("workspace",) + PAIRWISE_COMPARATORS


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
    """Per-seed validity guards, judged by the aggregator: a seed that fails one is uninformative (INVALID): the strongest
    pairwise comparator saturated or floored. (The convergence guard is a count across seeds, `unconverged_judged_arms`.)"""
    return [
        f"comparator_primary_permille >= {cfg['guard_floor_permille']}",
        f"comparator_primary_permille <= {cfg['guard_ceiling_permille']}",
    ]


def falsifier_rules() -> list[str]:
    return [f"{counter} <= 0" for counter in STRUCTURAL_COUNTERS]


def integrity_rules() -> list[str]:
    return [f"{counter} <= 0" for counter in INTEGRITY_COUNTERS]


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


def integrity_failed(result: dict) -> list[str]:
    """The integrity rules that fail."""
    return [r for r in integrity_rules() if not holds(result, r)]


# -- derivations of the guard scalars (pure; the harness uses them, test_rules.py pins each) --------

def comparator_primary_permille(ood16_permille: dict) -> int:
    """The best of the pairwise comparators on the primary metric (accuracy on sixteen modules, per mille)."""
    return max(ood16_permille[arm] for arm in PAIRWISE_COMPARATORS)


def workspace_trained(workspace_id: int, no_comm_id: int, margin_permille: int) -> int:
    """1 iff the workspace arm learned: its in-distribution accuracy is `margin_permille` above the no-communication floor."""
    return int(workspace_id >= no_comm_id + margin_permille)


def unconverged_guard_arms(converged: dict) -> int:
    """How many of the guarded arms (workspace and the three pairwise comparators) stopped at the step cap instead of by
    patience. `converged` maps an arm name to a truthy flag."""
    return sum(1 for arm in CONVERGENCE_GUARD_ARMS if not converged[arm])


def plateau_reached(curve: list, patience: int, min_delta_permille: int) -> bool:
    """Whether a validation curve (accuracy per mille at each evaluation) contains `patience` consecutive evaluations
    without an improvement of more than `min_delta_permille` over the best so far. An independent recomputation of the
    stopping rule of `harness.train_arm`, used to audit the convergence flag the harness records."""
    best, since = -1, 0
    for value in curve:
        if value > best + min_delta_permille:
            best, since = value, 0
        else:
            since += 1
        if since >= patience:
            return True
    return False
