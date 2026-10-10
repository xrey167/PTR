"""Aggregate the C001 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run C001 --seed <s>` writes
(results/run-<timestamp>-seed-<s>.json): the newest record of every declared seed
whose command ran to an exit code. Nothing of the experiment is run again. Every
declared seed must have a record, the records must be of one configuration and have run
the checkout's code (`experiment_records.source_revision`), and the preregistration
digests of the manifest, the table, the preregistration text and `criteria.toml` must be
the frozen ones.

The decision is mechanical, and it is `criteria.toml`'s (three outcomes plus invalid):

- a seed is judged by applying every coverage rule and every STRUCTURAL falsifier rule (integer
  counters; the grammar is S001's) to the one JSON object its harness printed; a failure
  is a STRUCTURAL defect of the product arm or an isolated ablation: FALSIFIED (kind
  "structural"), judged first;
- INTEGRITY counters (a defect of the task or the harness: module counts, overlap of
  training, validation and evaluation data, parameter matching, non-finite loss, checkpoint
  and convergence bookkeeping): the harness exits 2 and the seed, hence the run, is INVALID;
- validity guards: a seed whose strongest comparator is saturated or floored on the
  primary metric, more than the allowed number of seeds with an unconverged guarded arm
  (workspace and the three pairwise comparators), a workspace arm that did not train on more
  than the allowed number of seeds, or a sensitivity control (a crippled or weak all-to-all)
  that the paired bound cannot tell from the intact one: INVALID. The run says nothing about
  the claim; it is never FALSIFIED;
- the paired comparisons (workspace minus each pairwise comparator, accuracy on the primary
  metric over the declared seeds, one-sided 95% t bounds): COMPLETED (supported) iff every
  lower bound reaches the margin; FALSIFIED (kind "refuted") iff the guards hold and some
  UPPER bound is below the margin (the advantage is below the margin, which is not "no
  advantage"); NOT_SHOWN otherwise (the bounds straddle the margin: nothing is concluded, the
  status stays planned, no requirement is downgraded). The no_broadcast, capacity_off and
  competition_off contrasts are reported three-valued (attributed / not attributed /
  unresolved by noise) and never decide the verdict;
- INVALID also when a seed cannot be judged, a declared seed is missing, or the mutation
  evidence is missing, incomplete, not the plan's, or older than the harness code.

Exit status: 0 COMPLETED, 1 FALSIFIED, 2 INVALID or a refusal to aggregate, 3 NOT_SHOWN.

The harness is shown able to fail by `results/mutations.json`
(`research/consciousness/gw/run_mutations.py`, because `scripts/mutation_check.py` builds a
Cargo package and cannot plant defects in a Python entrypoint; the file has the layout
`experiment_records.mutation_evidence` reads). `--report PATH` also writes the Markdown
reading of the verdict to PATH.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tomllib
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
BENCHMARK = "c001"
EXPERIMENT = "C001"
TAIL = 2000
ENVIRONMENT_KEYS = ("python", "torch", "torch_config_sha256", "cpu_model", "cpu_flags_sha256", "uv_lock_sha256", "pyproject_sha256")
METRIC_NAMES = (
    "arms",
    "majority_permille",
    "no_comm_id_permille",
    "comparator_primary_permille",
    "workspace_trained",
    "hard_failures",
    "guards_failed",
    "unconverged_judged_arms",
    "environment",
    "param_mismatch_arms",
    "no_comm_leaks",
    "no_broadcast_leaks",
    "bottleneck_violations",
    "dropped_module_leaks",
    "n_leaks",
    "train_eval_overlap",
    "val_eval_overlap",
    "checkpoint_mismatch_arms",
    "convergence_mismatch_arms",
    "nondeterminism",
    "nonfinite_arms",
    "no_comm_above_chance",
)

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, INVALID, NOT_SHOWN = "COMPLETED", "FALSIFIED", "INVALID", "NOT_SHOWN"
OPERATORS = (">=", "<=", "==")
CONTROLS = ("n_leak", "overlap", "param_mismatch", "isolation_probe", "bottleneck", "dropout_probe")
NUMBER = re.compile(r"-?[0-9]+")
SCOPE = (
    "a functional-mechanism test, on a CPU toy task, of a learned-attention global workspace (competitive top-k write, gated "
    "slot update, broadcast read-back by every module) against parameter-matched pairwise comparators (all-to-all, "
    "bandwidth-matched and sparse top-k pairwise attention; the judged comparison) and ablations (no-broadcast, competition off, capacity off: reported attribution contrasts; no communication: floor) and crippled and weak "
    "all-to-all as gross and graded sensitivity controls: shared-weight modules, a finite training set of eight modules, evaluation on sixteen "
    "modules (primary) and twenty-four (secondary) with more distractors; the primary endpoint is accuracy on sixteen modules, "
    "judged by one-sided 95% t bounds over twenty seeds of the paired difference. It does not test the deterministic "
    "ptr-pods::workspace contract, and it makes no claim about consciousness"
)
LIMITATIONS = [
    "a result applies to this task family, its size (eight training modules, d=48, three rounds), the training-set size chosen on pilot seeds and these hyperparameters (the same learning-rate x width grid for every arm, selected on the primary metric over five pilot seeds disjoint from the declared ones; the grid was extended by lr 0.006 at d 48 for every arm after every arm had selected the edge 0.003, and by lr 0.012 for the four arms that then selected 0.006 (the pairwise comparators and competition_off), both extensions run on pilot seeds after the pilot comparisons had been seen; 0.012 won for none of them (the networks did not train), so every arm's selected rate is interior to the tested grid at d 48, but the grid is geometric and a rate between two tested ones or at d 32 was not tried, so an arm may still be under-tuned; d = 48 is the upper edge of the width grid {32, 48} for six of seven arms and d = 64 was not tried, the workspace gaining most from width, direction of the bias unknown); the workspace had extra structural choices (slots, top-k, rounds) made from pilot observations; arms are trained to a validation plateau and a converged arm is assumed to be at its optimum, which a bounded patience cannot prove",
    "the plateau protocol can stop an arm prematurely (a run that stalls for patience evaluations before a late jump is flagged converged): a stalled comparator inflates the workspace advantage, a stalled workspace deflates it; the guards catch only the extreme cases (band, untrained workspace), and the per-seed differences are heavy-tailed, so the median and the number of seeds with a positive difference are reported beside the t bounds",
    "the t bounds assume roughly normal paired differences, which twenty points cannot check; per-seed differences and both bounds are reported; the decision uses the upper bound for refutation and the lower bound for support; the 30 per mille margin is justified by the cost of building the learned workspace but was fixed before the freeze and after the pilots had been seen (not fitted to them; an earlier version used the pilot gap, 50); disclosed",
    "the ablations differ from the workspace arm in more than one way at matched parameter counts (update-network width); the no-broadcast arm keeps one reader (the reporter) so that the readout path is shared; the bandwidth-matched arm is one definition of matched bandwidth",
    "a learned-attention network is not the deterministic bid-ordered workspace of ptr-pods::workspace: a supportive result does not validate that contract and a refuting one does not touch it",
    "in distribution (eight modules) the pairwise arms can express everything the workspace can; the claim tested is an inductive-bias one (finite data, more modules), not an expressivity one",
    "sample efficiency (area under the learning curve), module dropout, ID accuracy, the 24-module accuracy, ignition and communication-cost proxies are reported with bounds where they are differences and are never judged",
    "mutation evidence comes from run_mutations.py (scripts/mutation_check.py plants Cargo packages only) and was run at a reduced scale with the locked uv environment's interpreter (the c001-venv python of the preregistration, section 8); experiment_records tracks only Rust and Cargo files as code, so this aggregator separately refuses when research/consciousness or this experiment's files differ from the mutation commit, and when the mutation total is not the plan's size",
    "COMPLETED shows an advantage over pairwise attention at 2048 training examples and 16 modules; it does not show that the broadcast read, the capacity limit or the competition is responsible (the attribution contrasts, which are not in the support conjunction, may read not attributed or unresolved), and FALSIFIED is scoped to the same setting (the pilot advantage grows with the training-set size)",
    "the workspace stalled (16-module accuracy near chance) on 1 of 17 pilot seeds without being flagged by the guard's harness rule (in-distribution accuracy at least the no-communication arm's plus 100 per mille); such a seed stays in the paired differences and counts against the workspace, and the guard tolerates up to 4 flagged seeds of 20 (more make the run INVALID), so the outcome probabilities are wide (P(COMPLETED) 0.02 to 0.69 depending on how the pilots are read; preregistration section 4)",
    "an indicator property here is a functional property: the experiment says nothing about whether any system is conscious",
]


CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))


def refuse(message: str):
    """Refuse to aggregate: exit status 2, which is never a verdict about the hypothesis."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


class Unreadable(Exception):
    """A rule that cannot be read or resolved: the seed is INVALID."""


def parse_rule(rule: str) -> tuple[str, str, str]:
    """`(left, op, right)` of a rule, which is exactly three tokens separated
    by single spaces with a known operator. Raises Unreadable."""
    tokens = rule.split(" ") if isinstance(rule, str) else []
    if len(tokens) != 3 or tokens[1] not in OPERATORS or not all(tokens):
        raise Unreadable(f"not '<left> <op> <right>' with an operator of {OPERATORS}: {rule!r}")
    return tokens[0], tokens[1], tokens[2]


def resolve(result: dict, path: str):
    """The value at the dotted path of object keys from the root of `result`.
    Raises Unreadable when a key is missing or a step is not an object."""
    at = result
    for key in path.split("."):
        if not isinstance(at, dict) or key not in at:
            raise Unreadable(f"{path} does not resolve in the harness result")
        at = at[key]
    return at


def operand(token: str, result: dict, decimal: bool, equality: bool):
    """`("int" | "num" | "text", value)` of a literal or reference token."""
    if NUMBER.fullmatch(token):
        return ("num", Decimal(token)) if "." in token else ("int", int(token))
    value = resolve(result, token)
    if isinstance(value, bool) or value is None:
        raise Unreadable(f"{token} is {value!r}, not a number")
    if isinstance(value, int):
        return "int", value
    if isinstance(value, float) and decimal:
        return "num", Decimal(repr(value))
    if isinstance(value, str) and equality:
        return "text", value
    raise Unreadable(f"{token} is {value!r}, which this rule does not compare")


def evaluate(result: dict, rule: str) -> bool:
    """Whether `rule` holds for the harness result. Raises Unreadable."""
    left, op, right = parse_rule(rule)
    decimal = any(NUMBER.fullmatch(token) and "." in token for token in (left, right))
    a = operand(left, result, decimal, op == "==")
    b = operand(right, result, decimal, op == "==")
    if (a[0] == "text") != (b[0] == "text"):
        raise Unreadable(f"{rule!r} compares a string with a number")
    x, y = a[1], b[1]
    if a[0] == "int" and b[0] == "num":
        x = Decimal(x)
    if a[0] == "num" and b[0] == "int":
        y = Decimal(y)
    return x >= y if op == ">=" else x <= y if op == "<=" else x == y


def judge_seed(result: dict | None, exit_code, criteria: dict, seed: int) -> dict:
    """One seed's judgement: the coverage and falsifier rules that fail,
    the reasons it cannot be judged, and its validity. Pure."""
    invalid: list[str] = []
    coverage_failed: list[str] = []
    falsifiers_failed: list[str] = []
    guards_failed: list[str] = []
    integrity_failed: list[str] = []
    if not isinstance(result, dict):
        return {"seed": seed, "valid": False, "invalid": ["no harness result"], "coverage_failed": [], "falsifiers_failed": [], "integrity_failed": [], "guards_failed": [], "workspace_trained": None, "unconverged_judged_arms": None}
    if result.get("benchmark") != BENCHMARK or result.get("experiment") != EXPERIMENT:
        invalid.append(f"reports benchmark {result.get('benchmark')!r}, experiment {result.get('experiment')!r}")
    if result.get("seed") != seed or isinstance(result.get("seed"), bool):
        invalid.append(f"reports seed {result.get('seed')!r}, not {seed}")
    if result.get("scale") != "pinned":
        invalid.append(f"ran at scale {result.get('scale')!r}, not the pinned one")
    procedure = result.get("procedure")
    if not isinstance(procedure, dict) or procedure.get("seed_is_declared") is not True:
        invalid.append("does not report a declared seed")
    elif procedure.get("shape_is_pinned") is not True:
        invalid.append("did not run the pinned scale")
    controls = result.get("controls")
    for name in CONTROLS:
        control = controls.get(name) if isinstance(controls, dict) else None
        if not isinstance(control, dict) or control.get("required_rule_tripped") is not True:
            invalid.append(f"the negative control {name} did not trip the rule it exists for")
    if result.get("harness_gaps") != []:
        invalid.append(f"harness gaps: {result.get('harness_gaps')!r}")
    for key, failed in (("coverage", coverage_failed), ("falsifiers", falsifiers_failed)):
        for rule in criteria[key]:
            try:
                if not evaluate(result, rule):
                    failed.append(rule)
            except Unreadable as error:
                invalid.append(f"{key}: {error}")
    # Integrity counters (a defect of the task or the harness): the seed cannot be judged, so the experiment is INVALID,
    # never FALSIFIED. The harness exits 2 for them.
    for rule in criteria.get("integrity", []):
        try:
            if not evaluate(result, rule):
                integrity_failed.append(rule)
        except Unreadable as error:
            invalid.append(f"integrity: {error}")
    if integrity_failed:
        invalid.append("harness integrity counters above 0: " + ", ".join(integrity_failed))
    unconverged = result.get("unconverged_judged_arms")
    if isinstance(unconverged, bool) or not isinstance(unconverged, int) or unconverged < 0:
        invalid.append(f"unconverged_judged_arms is {unconverged!r}, not a count")
        unconverged = None
    # Guards do not enter the harness's exit status: a saturated or floored comparator is not a defect.
    for rule in criteria.get("guards", []):
        try:
            if not evaluate(result, rule):
                guards_failed.append(rule)
        except Unreadable as error:
            invalid.append(f"guards: {error}")
    environment = result.get("environment")
    if not isinstance(environment, dict) or not all(isinstance(environment.get(k), str) and environment.get(k) for k in ENVIRONMENT_KEYS):
        invalid.append("the result records no complete environment (python, torch, CPU, lock digests)")
    trained = result.get("workspace_trained")
    if trained not in (0, 1) or isinstance(trained, bool):
        invalid.append(f"workspace_trained is {trained!r}, not 0 or 1")
    if exit_code not in (0, 1) or isinstance(exit_code, bool):
        invalid.append(f"the harness exited with {exit_code!r}")
    elif not invalid:
        failing = bool(coverage_failed or falsifiers_failed)
        if failing != (exit_code == 1):
            invalid.append(f"the harness exited {exit_code} but the rules {'fail' if failing else 'all hold'}")
    return {
        "seed": seed,
        "valid": not invalid,
        "invalid": invalid,
        "coverage_failed": coverage_failed,
        "falsifiers_failed": falsifiers_failed,
        "integrity_failed": integrity_failed,
        "guards_failed": guards_failed,
        "workspace_trained": trained if trained in (0, 1) else None,
        "unconverged_judged_arms": unconverged,
    }


getcontext().prec = 40


class PairedError(Exception):
    """A paired comparison that cannot be computed from the seeds' results."""


def arm_fraction(result: dict, arm: str, metric: str) -> Fraction:
    """Accuracy of `arm` on `metric` ("ood16", "id", ...) as an exact fraction of the counts the harness reports."""
    try:
        record = result["arms"][arm]
        correct, total = record[f"{metric}_correct"], record[f"{metric}_total"]
    except (KeyError, TypeError) as error:
        raise PairedError(f"arm {arm!r} has no {metric} counts: {error}") from error
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (correct, total)) or total <= 0 or not 0 <= correct <= total:
        raise PairedError(f"arm {arm!r} reports {correct!r} correct of {total!r}")
    return Fraction(correct, total)


def paired_bound(differences: list[Fraction], t_quantile: Decimal) -> dict:
    """Mean, sample standard deviation and the one-sided 95% t bounds (lower and upper) of paired differences
    (exact fractions in, Decimal arithmetic, floats out). The quantile is the pinned one for n-1 degrees of freedom."""
    n = len(differences)
    if n < 2:
        raise PairedError("a paired bound needs at least two seeds")
    mean = sum(differences, Fraction(0)) / n
    variance = sum(((d - mean) ** 2 for d in differences), Fraction(0)) / (n - 1)
    sd = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
    mean_d = Decimal(mean.numerator) / Decimal(mean.denominator)
    half = t_quantile * sd / Decimal(n).sqrt()
    return {
        "mean_permille": float(mean_d * 1000),
        "sd_permille": float(sd * 1000),
        "lower_bound_permille": float((mean_d - half) * 1000),
        "upper_bound_permille": float((mean_d + half) * 1000),
        "n": n,
    }


def differences(results: dict[int, dict], seeds: list[int], metric: str, left: str, right: str) -> list[Fraction]:
    return [arm_fraction(results[s], left, metric) - arm_fraction(results[s], right, metric) for s in seeds]


ATTRIBUTED, NOT_ATTRIBUTED, UNRESOLVED = "attributed", "not attributed", "unresolved by noise"
ATTRIBUTION_ARMS = {"broadcast": "no_broadcast", "bottleneck": "capacity_off", "competition": "competition_off"}


def attribution_state(bound: dict, margin: int) -> str:
    """Three-valued reading of one attribution contrast from its one-sided 95% bounds (permille): the lower bound reaches the
    margin (attributed), the upper bound is below it (not attributed), or neither (unresolved by noise)."""
    if bound["lower_bound_permille"] >= margin:
        return ATTRIBUTED
    if bound["upper_bound_permille"] < margin:
        return NOT_ATTRIBUTED
    return UNRESOLVED


def median(values: list[Fraction]) -> Fraction:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def paired_comparisons(results: dict[int, dict], paired: dict) -> dict:
    """The pre-specified comparisons on `metric` accuracy over the declared seeds (workspace minus comparator):

    - judged comparators (`against`, the pairwise arms): SUPPORTED needs every lower one-sided 95% bound >= `margin_permille`;
    - REFUTED needs, for one of the pairwise comparators (`pairwise`), the upper bound at the Bonferroni-adjusted
      quantile (`t_refute`, alpha/3) below the margin (the advantage is below the margin, not absent);
    - attribution contrasts (`attribution`: no_broadcast = broadcast, capacity_off = bottleneck, competition_off = competition),
      three-valued from the one-sided 95% bounds: ATTRIBUTED (lower >= `attribution_margin_permille`), NOT ATTRIBUTED (upper
      below it) or UNRESOLVED by noise; the half-width, the median and the count of positive per-seed differences are reported;
      they never change the verdict;
    - sensitivity controls: a gross one (a crippled all-to-all must be told apart by `margin_permille`) and a graded one
      (the all-to-all net trained with Adam weight decay 0.001, a known deficit of about 65 per mille on the pilot seeds, must be detected: lower bound above its
      `min_lower_permille`); either failing makes the run insensitive (INVALID);
    - descriptive comparisons carry the same bounds and are never judged. Raises PairedError."""
    t_support, t_refute = Decimal(paired["t_support"]), Decimal(paired["t_refute"])
    seeds = sorted(results)
    if len(seeds) != int(paired["seeds_required"]):
        raise PairedError(f"{len(seeds)} seeds, the rule is preregistered for {paired['seeds_required']}")
    margin = paired["margin_permille"]
    judged = []
    for other in paired["against"]:
        diffs = differences(results, seeds, paired["metric"], paired["arm"], other)
        bound = paired_bound(diffs, t_support)
        adjusted = paired_bound(diffs, t_refute)
        pairwise = other in paired["pairwise"]
        judged.append(
            {
                "against": other,
                "metric": paired["metric"],
                "per_seed_permille": [float(d * 1000) for d in diffs],
                "supports": bound["lower_bound_permille"] >= margin,
                "refutes": pairwise and adjusted["upper_bound_permille"] < margin,
                "upper_bound_adjusted_permille": adjusted["upper_bound_permille"],
                "margin_permille": margin,
                **bound,
            }
        )
    attribution = []
    for other in paired["attribution"]:
        diffs = differences(results, seeds, paired["metric"], paired["arm"], other)
        bound = paired_bound(diffs, t_support)
        attribution.append(
            {
                "against": other,
                "state": attribution_state(bound, paired["attribution_margin_permille"]),
                "attribution_margin_permille": paired["attribution_margin_permille"],
                "half_width_permille": (bound["upper_bound_permille"] - bound["lower_bound_permille"]) / 2,
                "median_permille": float(median(diffs) * 1000),
                "positive_seeds": sum(1 for d in diffs if d > 0),
                "per_seed_permille": [float(d * 1000) for d in diffs],
                **bound,
            }
        )
    states = {name: next((a["state"] for a in attribution if a["against"] == arm), None) for name, arm in ATTRIBUTION_ARMS.items()}
    gross, graded = paired["sensitivity_gross"], paired["sensitivity_graded"]
    gb = paired_bound(differences(results, seeds, paired["metric"], gross["arm"], gross["crippled"]), t_support)
    rb = paired_bound(differences(results, seeds, paired["metric"], graded["arm"], graded["weak"]), t_support)
    descriptive = []
    for metric in paired.get("descriptive_metrics", []):
        for other in paired["against"] + paired.get("descriptive_against", []):
            descriptive.append({"against": other, "metric": metric, **paired_bound(differences(results, seeds, metric, paired["arm"], other), t_support)})
    return {
        "supported": all(item["supports"] for item in judged),
        "refuted_against": [item["against"] for item in judged if item["refutes"]],
        "judged": judged,
        "attribution": attribution,
        "attribution_states": states,
        "broadcast_attributed": states["broadcast"] == ATTRIBUTED,
        "bottleneck_attributed": states["bottleneck"] == ATTRIBUTED,
        "competition_attributed": states["competition"] == ATTRIBUTED,
        "sensitivity": {
            "gross": {**gross, "resolves": gb["lower_bound_permille"] >= gross["margin_permille"], **gb},
            "graded": {**graded, "resolves": rb["lower_bound_permille"] > graded["min_lower_permille"], **rb},
        },
        "descriptive": descriptive,
    }


def decide(
    seeds: list[dict],
    static: list[dict] | None,
    declared: list[int],
    mutations: dict | None,
    paired: dict | None,
    untrained_workspace_seeds: int,
    max_untrained: int,
    plan_size: int | None,
    max_unconverged: int,
) -> dict:
    """The verdict from the seeds' judgements (`judge_seed`, in any order), the static-evidence results, the mutation
    evidence (`killed`, `total`), the paired comparisons and the count of seeds whose workspace arm did not train.
    Four outcomes:

    - INVALID first of all when any seed trips a harness-integrity counter (a defect of the task or the harness);
    - FALSIFIED, kind "structural": a valid seed fails a coverage rule or a structural counter (a defect of the product
      arm or an isolated ablation);
    - INVALID: no judgement is possible or the run is uninformative (seeds missing or unjudgeable, mutation evidence
      missing, incomplete or not the plan's, a harness-integrity counter above 0 on any seed, a saturated or floored
      comparator on any seed, an unconverged guarded arm (workspace or a pairwise comparator) on more than `max_unconverged`
      seeds, a workspace arm that failed to train on more than `max_untrained` seeds, a sensitivity control that cannot
      resolve its margin);
    - FALSIFIED, kind "refuted": the guards hold and some upper bound is below the margin;
    - NOT_SHOWN: the guards hold and the bounds straddle the margin (nothing changes, no downgrade);
    - COMPLETED: every lower bound reaches the margin. Pure."""
    present = sorted(entry["seed"] for entry in seeds)
    missing = sorted(set(declared) - set(present))
    duplicate = sorted({seed for seed in present if present.count(seed) > 1})
    extra = sorted(set(present) - set(declared))
    problems = []
    if missing:
        problems.append(f"no judgement of seeds {missing}")
    if duplicate or extra:
        problems.append(f"judgements of seeds {duplicate + extra} that are duplicated or not declared")
    if static is None:
        problems.append("the static evidence was not run")
    if mutations is None:
        problems.append("no mutation evidence (results/mutations.json)")
    elif mutations.get("killed") != mutations.get("total"):
        problems.append(f"the mutation check killed {mutations.get('killed')} of {mutations.get('total')} mutations")
    elif plan_size is not None and mutations.get("total") != plan_size:
        problems.append(f"the mutation evidence counts {mutations.get('total')} mutations, the plan lists {plan_size}")
    unmet = [entry["seed"] for entry in seeds if entry["valid"] and entry["coverage_failed"]]
    tripped = [entry["seed"] for entry in seeds if entry["valid"] and entry["falsifiers_failed"]]
    unjudged = [entry["seed"] for entry in seeds if not entry["valid"]]
    uninformative = [entry["seed"] for entry in seeds if entry["valid"] and entry.get("guards_failed")]
    unconverged_seeds = [entry["seed"] for entry in seeds if entry["valid"] and entry.get("unconverged_judged_arms")]
    static_failed = [item["command"] for item in static or [] if item.get("exit_code") != 0]
    broken = [entry["seed"] for entry in seeds if entry.get("integrity_failed")]
    kind = None
    if broken:
        # A defect of the task or the harness (an integrity counter) is never a result about the product, and it makes any
        # structural finding on the other seeds suspect: INVALID before everything else.
        verdict = INVALID
        problems.append(f"harness integrity counters above 0 on seeds {broken}: a defect of the task or the harness, repaired in a successor")
    elif unmet or tripped or static_failed:
        verdict, kind = FALSIFIED, "structural"
    elif problems or unjudged:
        verdict = INVALID
    elif uninformative:
        verdict = INVALID
        problems.append(f"the strongest comparator is saturated or floored on seeds {uninformative}: the run says nothing about the claim")
    elif len(unconverged_seeds) > max_unconverged:
        verdict = INVALID
        problems.append(
            f"a guarded arm (workspace or a pairwise comparator) hit the step cap on seeds {unconverged_seeds} "
            f"(more than {max_unconverged}): the comparison would be one of optimisation speed"
        )
    elif untrained_workspace_seeds > max_untrained:
        verdict = INVALID
        problems.append(f"the workspace arm did not train on {untrained_workspace_seeds} seeds (more than {max_untrained}): an optimisation finding, reported, not a verdict")
    elif paired is None:
        verdict = INVALID
        problems.append("the paired comparison could not be computed")
    elif not (paired["sensitivity"]["gross"]["resolves"] and paired["sensitivity"]["graded"]["resolves"]):
        verdict = INVALID
        problems.append("a sensitivity control is not resolved (gross: a crippled all-to-all; graded: the all-to-all net with Adam weight decay 0.001, a known deficit): the pipeline is insensitive at this operating point")
    elif paired["refuted_against"]:
        verdict, kind = FALSIFIED, "refuted"
    elif paired["supported"]:
        verdict = COMPLETED
    else:
        verdict = NOT_SHOWN
    return {
        "verdict": verdict,
        "kind": kind,
        "vacuous_seeds": unmet,
        "falsified_seeds": tripped,
        "invalid_seeds": unjudged,
        "integrity_seeds": broken,
        "uninformative_seeds": uninformative,
        "unconverged_seeds": unconverged_seeds,
        "untrained_workspace_seeds": untrained_workspace_seeds,
        "static_evidence_failed": static_failed,
        "refuted_against": (paired or {}).get("refuted_against", []),
        "attribution_states": (paired or {}).get("attribution_states"),
        "attribution": (
            None if verdict != COMPLETED
            else "workspace effect shown; " + "; ".join(f"{name}: {state}" for name, state in paired["attribution_states"].items())
        ),
        "partial_pattern": (
            "broadcast attributed (GWT-3) but the pairwise advantage (GWT-2) is not shown"
            if verdict == NOT_SHOWN and paired["broadcast_attributed"] else None
        ),
        "broadcast_downgraded": bool(paired and paired["attribution_states"]["broadcast"] == NOT_ATTRIBUTED),
        "problems": problems,
        "recommended_status": {COMPLETED: "completed", FALSIFIED: "failed"}.get(verdict),
    }


def run_command(command: str) -> tuple[int, str]:
    """Run one static-evidence command in the repository root. A command that
    cannot start is exit code -1."""
    try:
        process = subprocess.run(shlex.split(command), cwd=ROOT, text=True, capture_output=True)
    except OSError as error:
        return -1, str(error)
    return process.returncode, process.stdout + process.stderr


def static_evidence(commands: list[str], runner: Callable[[str], tuple[int, str]] = run_command) -> list[dict]:
    """Run every command and record its exit code and output digest."""
    results = []
    for command in commands:
        code, output = runner(command)
        results.append(
            {
                "command": command,
                "exit_code": code,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "output_tail": output[-TAIL:],
            }
        )
    return results


def report(
    verdict: dict, seeds: list[dict], static: list[dict] | None, mutations: dict | None = None, paired: dict | None = None
) -> str:
    """The Markdown reading of the verdict."""
    lines = [f"# C001 global workspace: {verdict['verdict']}", ""]
    lines.append(f"Seeds: {', '.join(str(entry['seed']) for entry in sorted(seeds, key=lambda e: e['seed']))}. "
                 "Every seed must hold every coverage and falsifier rule; the paired rule must hold across the seeds.")
    lines.append("")
    for entry in sorted(seeds, key=lambda e: e["seed"]):
        state = "valid" if entry["valid"] else "INVALID"
        lines.append(f"- seed {entry['seed']}: {state}; {len(entry['coverage_failed'])} coverage rules unmet, "
                     f"{len(entry['falsifiers_failed'])} falsifier rules tripped, {len(entry.get('integrity_failed', []))} integrity counters above 0")
        for key in ("invalid", "coverage_failed", "falsifiers_failed", "integrity_failed"):
            lines.extend(f"    - {key}: {item}" for item in entry.get(key, []))
    lines.append("")
    for item in static or []:
        lines.append(f"- static evidence `{item['command']}`: exit {item['exit_code']}")
    if paired is not None:
        lines.append("")
        lines.append("Paired comparisons (workspace minus comparator, per mille; one-sided 95% t bounds against the margin):")
        for item in paired["judged"]:
            state = "supports" if item["supports"] else ("REFUTES" if item["refutes"] else "straddles")
            lines.append(
                f"- {item['metric']} vs {item['against']}: mean {item['mean_permille']:.1f}, sd {item['sd_permille']:.1f}, "
                f"lower {item['lower_bound_permille']:.1f}, upper {item['upper_bound_permille']:.1f}, margin {item['margin_permille']}: {state}"
            )
        for kind in ("gross", "graded"):
            sens = paired["sensitivity"][kind]
            other = sens.get("crippled", sens.get("weak"))
            lines.append(f"- {kind} sensitivity control ({sens['arm']} minus {other}): mean {sens['mean_permille']:.1f}, lower {sens['lower_bound_permille']:.1f}: {'resolves' if sens['resolves'] else 'DOES NOT RESOLVE'}")
        for item in paired["attribution"]:
            lines.append(
                f"- attribution contrast vs {item['against']} (margin {item['attribution_margin_permille']}): mean {item['mean_permille']:.1f}, "
                f"median {item['median_permille']:.1f}, {item['positive_seeds']} of {item['n']} seeds positive, half-width {item['half_width_permille']:.1f}, "
                f"lower {item['lower_bound_permille']:.1f}, upper {item['upper_bound_permille']:.1f}: {item['state'].upper()}"
            )
        for item in paired["descriptive"]:
            lines.append(f"- (descriptive) {item['metric']} vs {item['against']}: mean {item['mean_permille']:.1f}, lower {item['lower_bound_permille']:.1f}, upper {item['upper_bound_permille']:.1f}")
    if mutations is not None:
        lines.append(f"- mutation check: {mutations.get('killed')} of {mutations.get('total')} killed")
    for problem in verdict["problems"]:
        lines.append(f"- problem: {problem}")
    lines.append("")
    lines.append("Limitations: " + "; ".join(LIMITATIONS) + ".")
    return "\n".join(lines) + "\n"


def newest_per_seed(paths: list[Path]) -> dict[int, Path]:
    """The record to aggregate for each seed: the one whose run finished, the newest of the others only where none finished.
    A seed is declared to run once (the runner refuses a second run of it): two finished records of one seed are refused here too
    rather than resolved by name, so a repeated seed cannot be chosen after the fact (SystemExit 2)."""
    chosen: dict[int, tuple[bool, Path]] = {}
    finished: dict[int, list[Path]] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = int(record["seed"])
        launched = experiment_records.finished_run(record)
        if launched:
            finished.setdefault(seed, []).append(path)
        if launched or not chosen.get(seed, (False, path))[0]:
            chosen[seed] = (launched, path)
    repeated = {seed: [p.name for p in found] for seed, found in sorted(finished.items()) if len(found) > 1}
    if repeated:
        refuse(f"{EXPERIMENT}: refusing to aggregate: more than one finished run record for a seed ({repeated}); each declared seed runs once")
    return {seed: path for seed, (_, path) in chosen.items()}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


GUARDED_PATHS = ("research/consciousness", "experiments/consciousness/C001-global-workspace")
RESULTS_EXCLUDE = ":(exclude)experiments/consciousness/C001-global-workspace/results"


def environment_mismatches(environments: dict) -> dict:
    """seed -> the environment keys in which its recorded environment differs from the first recorded one (seeds without a
    record are judged elsewhere)."""
    reference = next((e for e in environments.values() if isinstance(e, dict)), None)
    return {
        seed: [k for k in sorted(set(environment) | set(reference)) if environment.get(k) != reference.get(k)]
        for seed, environment in environments.items()
        if isinstance(environment, dict) and environment != reference
    }


def invalidate_environment_mismatches(judgements: list[dict], environments: dict) -> dict:
    """Mark invalid every seed whose recorded environment differs from the first one (the experiment must run in one environment);
    returns the mismatches per seed."""
    mismatches = environment_mismatches(environments)
    for entry in judgements:
        if entry["seed"] in mismatches:
            entry["valid"] = False
            entry["invalid"].append("its environment differs from that of the other seeds: " + ", ".join(mismatches[entry["seed"]]))
    return mismatches


def untrained_workspace_seeds(judgements: list[dict]) -> int:
    """The seeds on which the workspace arm did not train (`workspace_trained` is 0)."""
    return sum(1 for entry in judgements if entry.get("workspace_trained") == 0)


def verdict_of(judgements: list[dict], static: list[dict], declared: list[int], mutations: dict | None, paired: dict | None,
               criteria_paired: dict, plan_size: int) -> dict:
    """`decide` with the allowances of the criteria and the count of seeds whose workspace did not train (the wiring `aggregate` uses)."""
    return decide(
        judgements, static, declared, mutations, paired, untrained_workspace_seeds(judgements),
        int(criteria_paired["max_untrained_workspace_seeds"]), plan_size, int(criteria_paired["max_unconverged_seeds"]),
    )


def code_changed_since(sha: str, root: Path = ROOT) -> str:
    """Why the harness code is not the code the mutation evidence ran at ("" when it is). scripts/experiment_records
    tracks only Rust and Cargo files as code, so a change to the Python harness, the task, the models, the lock file
    or this experiment's own files between the mutation commit and now would go unseen; this check closes that window
    locally (it does not touch the shared tracking). Any difference to `sha`, committed or not, is a reason."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    committed = git("diff", "--name-only", sha, "HEAD", "--", *GUARDED_PATHS, RESULTS_EXCLUDE)
    if committed.returncode != 0:
        return f"git cannot compare {sha} with HEAD: {committed.stderr.strip()}"
    if committed.stdout.strip():
        return "these files changed since the mutation commit " + sha[:12] + ": " + ", ".join(committed.stdout.split())
    dirty = git("status", "--porcelain", "--", *GUARDED_PATHS, RESULTS_EXCLUDE)
    if dirty.stdout.strip():
        return "uncommitted changes in " + ", ".join(line[3:] for line in dirty.stdout.splitlines())
    return ""


def load_frozen() -> tuple[dict, dict, dict, str]:
    """The manifest, the preregistered table, the criteria and the table's digest, once the
    manifest, the table, the preregistration text and the criteria agree. Raises SystemExit."""
    manifest = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
    if "preregistration" not in CONFIG:
        refuse(
            f"{EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table (C001 is not frozen yet)"
        )
    table = CONFIG["preregistration"]
    try:
        digest = experiment_records.preregistration_digest(table)
    except ValueError as error:
        refuse(f"{EXPERIMENT}: the preregistration cannot be read: {error}")
    if manifest.get("preregistration_sha256") != digest:
        refuse(
            f"{EXPERIMENT}: refusing to aggregate: experiment.toml names preregistration_sha256 "
            f"{manifest.get('preregistration_sha256')!r}, not {digest}, the digest of config.toml's table"
        )
    for key in ("criteria", "preregistration_text"):
        path = ROOT / table[key]
        actual = experiment_records.preregistered_file_digest(path)
        if table.get(f"{key}_sha256") != actual:
            refuse(f"{EXPERIMENT}: refusing to aggregate: {table[key]} is not the file frozen ({actual})")
    criteria = tomllib.loads((ROOT / table["criteria"]).read_text(encoding="utf-8"))
    return manifest, table, criteria, digest


def aggregate(report_path: Path | None = None) -> str:
    """Validate the declared seed records, judge them and publish metrics.json and run.json.
    Missing records or failed preregistration or provenance checks raise SystemExit; a
    verdict of any kind is published."""
    manifest, table, criteria, digest = load_frozen()
    per_seed = newest_per_seed(sorted(RESULTS.glob("run-*-seed-*.json")))
    missing = sorted(set(manifest["seeds"]) - set(per_seed))
    if missing:
        refuse(f"no run record for seeds {missing}")
    chosen = {seed: json.loads(per_seed[seed].read_text(encoding="utf-8")) for seed in manifest["seeds"]}
    named = {per_seed[seed].name: chosen[seed] for seed in manifest["seeds"]}
    results_path = experiment_records.relative_to_root(RESULTS, ROOT)
    experiment_path = experiment_records.relative_to_root(HERE, ROOT)
    try:
        revision = experiment_records.source_revision(
            EXPERIMENT,
            manifest,
            named,
            ROOT,
            HERE,
            paths=experiment_records.listed_record_paths(results_path),
            checkout_paths=experiment_records.listed_staleness_paths(results_path, experiment_path),
            listed_experiment=True,
        )
        seed_records = experiment_records.seed_record_digests(RESULTS)
        mutations = experiment_records.mutation_evidence(EXPERIMENT, RESULTS / "mutations.json", BENCHMARK, ROOT, HERE)
    except experiment_records.ProvenanceError as error:
        refuse(f"{EXPERIMENT}: refusing to aggregate: {error}")
    if mutations is not None:
        changed = code_changed_since(mutations["git_sha"])
        if changed:
            refuse(f"{EXPERIMENT}: refusing to aggregate: {changed}; rerun research/consciousness/gw/run_mutations.py on a clean committed tree")

    judgements, records = [], []
    result_of: dict[int, dict | None] = {}
    metric_names = METRIC_NAMES
    for seed in manifest["seeds"]:
        record = chosen[seed]
        result = None
        try:
            result = experiment_records.harness_results({per_seed[seed].name: record}, BENCHMARK)[per_seed[seed].name]
            error = None
        except experiment_records.ProvenanceError as problem:
            error = str(problem)
        judgement = judge_seed(result, record.get("exit_code"), criteria, seed)
        result_of[seed] = result
        if error:
            judgement["valid"] = False
            judgement["invalid"].append(error)
        judgements.append(judgement)
        records.append(
            {"seed": seed, "record": per_seed[seed].name, "git_sha": record["git_sha"], "exit_code": record.get("exit_code")}
        )
    # One environment for all seeds: a mixed record (another torch build, CPU or lock file) cannot be compared across seeds.
    invalidate_environment_mismatches(judgements, {seed: (result_of[seed] or {}).get("environment") for seed in manifest["seeds"]})
    static = static_evidence(criteria.get("static_evidence", []))
    paired = None
    all_valid = all(entry["valid"] for entry in judgements)
    if all_valid:
        try:
            paired = paired_comparisons({seed: result_of[seed] for seed in manifest["seeds"]}, criteria["paired"])
        except PairedError as error:
            judgements[0]["invalid"].append(f"paired comparison: {error}")
            judgements[0]["valid"] = False
    # After every invalidation above, so that a seed made invalid by its environment or the paired comparison is recorded as invalid.
    metrics_seeds = [
        {
            **judgement,
            "metrics": {name: result_of[judgement["seed"]].get(name) for name in metric_names} if result_of[judgement["seed"]] else None,
            "summary": result_of[judgement["seed"]].get("summary") if result_of[judgement["seed"]] else None,
        }
        for judgement in judgements
    ]
    plan = tomllib.loads((HERE / "tests/mutations.toml").read_text(encoding="utf-8"))
    verdict = verdict_of(judgements, static, manifest["seeds"], mutations, paired, criteria["paired"], len(plan["mutation"]))
    text = report(verdict, judgements, static, mutations, paired)
    metrics = {
        "experiment_id": EXPERIMENT,
        "scope": SCOPE,
        "verdict": verdict["verdict"],
        "decision": verdict,
        "paired": paired,
        "seeds": metrics_seeds,
        "static_evidence": static,
        "mutation_checks": mutations,
        "report": text,
    }
    run = {
        "experiment_id": EXPERIMENT,
        "status": manifest["status"],
        "scope": SCOPE,
        "git_sha": revision,
        "aggregated_at_git_sha": git_sha(),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": digest,
        "preregistration_rules_sha256": manifest.get("preregistration_rules_sha256"),
        "criteria_sha256": table["criteria_sha256"],
        "entrypoint": manifest["entrypoint"],
        "seeds": records,
        "seed_records": seed_records,
        "verdict": verdict["verdict"],
        "paired": paired,
        "recommended_status": verdict["recommended_status"],
        "static_evidence": [{key: item[key] for key in ("command", "exit_code", "output_sha256")} for item in static],
        "mutation_checks": mutations,
        "limitations": LIMITATIONS,
    }
    try:
        experiment_records.publish_aggregate(RESULTS, metrics, run)
    except experiment_records.ProvenanceError as error:
        refuse(f"{EXPERIMENT}: {error}")
    if report_path is not None:
        report_path.write_text(text, encoding="utf-8")
    print(f"{EXPERIMENT} verdict={verdict['verdict']} kind={verdict['kind']} recommended_status={verdict['recommended_status']} "
          f"vacuous={verdict['vacuous_seeds']} falsified={verdict['falsified_seeds']} invalid={verdict['invalid_seeds']} "
          f"uninformative={verdict['uninformative_seeds']} unconverged={verdict['unconverged_seeds']} refuted_against={verdict['refuted_against']} problems={len(verdict['problems'])}")
    return verdict["verdict"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="also write the Markdown reading of the verdict here")
    args = parser.parse_args()
    verdict = aggregate(report_path=args.report)
    sys.exit({COMPLETED: 0, FALSIFIED: 1, NOT_SHOWN: 3}.get(verdict, 2))


if __name__ == "__main__":
    main()
