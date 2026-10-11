"""Aggregate the F007 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run F007 --seed <s>` writes
(results/run-<timestamp>-seed-<s>.json). Nothing of the experiment is run again. Every declared seed
must have a record, the records must be of one configuration and have run the checkout's code, this
script included, and the preregistration digests must be the frozen ones.

The decision is mechanical and is `criteria.toml`'s:

1. A seed is judged by its integrity: the coverage rules, the falsifier rules and the control rules
   of criteria.toml are applied to the one JSON object its harness printed (`judge_seed`, the rule
   grammar of `evaluate`). A seed whose rules cannot be read, whose benchmark, seed or scale is not
   the frozen one, or whose exit status disagrees with its rules, is INVALID.
2. A falsifier tripped on a valid seed (an accepted candidate that fails independent
   re-verification, a budget overrun, a routing outside the admissible set, ...) is a product defect:
   FALSIFIED. Controls and coverage that fail make the experiment unable to discriminate: INVALID,
   which says nothing about the hypothesis. So does a missing seed, or mutation evidence that does not
   kill every mutation.
3. Only then the statistics (`analyse`): the unit of replication is the seed. For each baseline the
   difference of the verified-success rate of the full loop and the baseline, paired over requests
   within the seed, is taken per seed, and the mean over the declared seeds gets a one-sided t
   interval (df = seeds - 1). COMPLETED iff the lower bound exceeds the margin against every baseline;
   FALSIFIED iff the upper bound is below the margin against any; NOT_SHOWN otherwise.
   The controls that need more than one seed (the sensitivity control, the null-world equivalence, the
   trained-baseline check, the wiring of the ablations) are judged on the same intervals and make the
   verdict INVALID when they fail.

The ablations (no workspace, no finding routing, no self-model, no L3) and the descriptive worlds are
reported with their intervals and a label; they do not change the verdict. metrics.json and run.json
are published as one aggregate (`experiment_records.publish_aggregate`). `--report PATH` also writes
the Markdown reading of the verdict to PATH.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
BENCHMARK = "vcw-loop"
EXPERIMENT = "F007"
TAIL = 2000

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, NOT_SHOWN, INVALID = "COMPLETED", "FALSIFIED", "NOT_SHOWN", "INVALID"
OPERATORS = (">=", "<=", "==")
NUMBER = re.compile(r"-?[0-9]+")
ARMS = ["full", "gate", "bon", "no_workspace", "no_finding", "no_selfmodel", "no_l3", "shuffled"]
SCOPE = (
    "the whole verifier-closed workspace loop (ptr-runtime VcwLoop with its finding router, learned router "
    "promoted through the off-policy gate, workspace, self-model and attention schema, and L3 through the "
    "ledger) on seeded SIMULATED expert Pods with declared per-(Pod, finding code) skill tables, over arithmetic, "
    "JSON-schema and small code tasks judged by the real ptr-verifier verifiers, against a trained softmax "
    "gate and best-of-N with the same verifier at a matched verifier-call budget. The skill tables are the "
    "assumption under test: a gain is a consequence of them, bounded by them. No claim about any model."
)
LIMITATIONS = [
    "the expert Pods are seeded simulations with declared skill tables; whether finding routing helps is decided by those tables (the assumption under test), and the null world shows only that the harness adds no advantage of its own where the tables are equal",
    "a trained gate over task features can capture at most the kind-level skill; in the declared worlds the codes of one kind are served best by different Pods, so a task-feature gate is handicapped by construction, which is the structure finding routing exploits",
    "no Pod reads the workspace broadcast outside the descriptive world that declares it, so the workspace ablation is an exact tie by construction (a wiring check), not a measurement of the workspace's value",
    "the self-model acts only through bounded hint boosts; the ablation measures that path alone",
    "L3 is Birth only, on one code that no initial Pod can repair; the baselines have no L3, and the effect of L3 is shown by the no-L3 arm",
    "the learned arms are trained once per run seed on pilot-domain traffic with no online learning during evaluation; the reported intervals are over seeds and so include the variance of training",
    "five seeds give a one-sided t interval with four degrees of freedom, which is wide; a small true gain is NOT_SHOWN, not absent",
    "the workspace, the self-model and L3 are not discriminated by the primary world (the tie of the workspace is by construction, the self-model has no discriminating condition in this design and is untested, L3 acts on one rare code); the workspace and L3 have preregistered secondary hypotheses with margins derived from the planning model, the workspace secondary tests a harness-defined reader of the broadcast and not the workspace's value, and the L3 secondary has low power",
    "best-of-N lacks any repair Pod, so a gain over it bundles having a repair Pod with routing it; the routing claim rests on the contrasts to the gate and to the shuffled map",
    "success is verified success under the real verifiers; a task whose verifier is blind to a defect would count as success (the independent oracle counts these as false accepts, and none was expected)",
]

CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))

# One-sided 95% quantiles of Student's t by degrees of freedom (exact to the digits shown).
T95 = {
    1: 6.313752, 2: 2.919986, 3: 2.353363, 4: 2.131847, 5: 2.015048, 6: 1.943180, 7: 1.894579,
    8: 1.859548, 9: 1.833113, 10: 1.812461, 11: 1.795885, 12: 1.782288, 13: 1.770933, 14: 1.761310,
    15: 1.753050, 16: 1.745884, 17: 1.739607, 18: 1.734064, 19: 1.729133, 20: 1.724718,
}


class Unreadable(Exception):
    """A rule that cannot be read or resolved: the seed is INVALID."""


# ---- the rule grammar (the repository's) ----------------------------------------------------


def parse_rule(rule: str) -> tuple[str, str, str]:
    tokens = rule.split(" ") if isinstance(rule, str) else []
    if len(tokens) != 3 or tokens[1] not in OPERATORS or not all(tokens):
        raise Unreadable(f"not '<left> <op> <right>' with an operator of {OPERATORS}: {rule!r}")
    return tokens[0], tokens[1], tokens[2]


def resolve(result: dict, path: str):
    at = result
    for key in path.split("."):
        if not isinstance(at, dict) or key not in at:
            raise Unreadable(f"{path} does not resolve in the harness result")
        at = at[key]
    return at


def operand(token: str, result: dict, equality: bool):
    if NUMBER.fullmatch(token):
        return "int", int(token)
    value = resolve(result, token)
    if isinstance(value, bool) or value is None:
        raise Unreadable(f"{token} is {value!r}, not a number")
    if isinstance(value, int):
        return "int", value
    if isinstance(value, str) and equality:
        return "text", value
    raise Unreadable(f"{token} is {value!r}, which this rule does not compare")


def evaluate(result: dict, rule: str) -> bool:
    left, op, right = parse_rule(rule)
    a, b = operand(left, result, op == "=="), operand(right, result, op == "==")
    if (a[0] == "text") != (b[0] == "text"):
        raise Unreadable(f"{rule!r} compares a string with a number")
    x, y = a[1], b[1]
    return x >= y if op == ">=" else x <= y if op == "<=" else x == y


# ---- one seed's integrity ---------------------------------------------------------------------


def judge_seed(result: dict | None, exit_code, criteria: dict, seed: int) -> dict:
    """The seed's judgement: the rules of each list that fail, the reasons it cannot be judged and
    its validity. Pure."""
    empty = {"seed": seed, "valid": False, "coverage_failed": [], "falsifiers_failed": [], "controls_failed": []}
    if not isinstance(result, dict):
        return {**empty, "invalid": ["no harness result"]}
    invalid: list[str] = []
    if result.get("benchmark") != BENCHMARK or result.get("experiment") != EXPERIMENT:
        invalid.append(f"reports benchmark {result.get('benchmark')!r}, experiment {result.get('experiment')!r}")
    if result.get("seed") != seed or isinstance(result.get("seed"), bool):
        invalid.append(f"reports seed {result.get('seed')!r}, not {seed}")
    if result.get("scale") != "pinned":
        invalid.append(f"ran at scale {result.get('scale')!r}, not the pinned one")
    procedure = result.get("procedure")
    if not isinstance(procedure, dict) or procedure.get("seed_is_declared") is not True:
        invalid.append("does not report a declared seed")
    elif procedure.get("seed_is_pilot") is not False or procedure.get("shape_is_pinned") is not True:
        invalid.append("is a pilot seed or did not run the pinned scale")
    elif not re.fullmatch(r"[0-9a-f]{64}", str(procedure.get("source_digest"))):
        invalid.append("reports no source digest of the harness")
    failed = {"coverage": [], "falsifiers": [], "invalid_if": []}
    for key, bucket in failed.items():
        for rule in criteria[key]:
            try:
                if not evaluate(result, rule):
                    bucket.append(rule)
            except Unreadable as error:
                invalid.append(f"{key}: {error}")
    if exit_code not in (0, 1) or isinstance(exit_code, bool):
        invalid.append(f"the harness exited with {exit_code!r}")
    elif not invalid:
        failing = any(failed.values())
        if failing != (exit_code == 1):
            invalid.append(f"the harness exited {exit_code} but the rules {'fail' if failing else 'all hold'}")
    return {
        "seed": seed,
        "valid": not invalid,
        "invalid": invalid,
        "coverage_failed": failed["coverage"],
        "falsifiers_failed": failed["falsifiers"],
        "controls_failed": failed["invalid_if"],
    }


# ---- the statistics -----------------------------------------------------------------------------


def t_quantile(df: int) -> float:
    if df not in T95:
        raise ValueError(f"no t quantile for {df} degrees of freedom: declare between 2 and 21 seeds")
    return T95[df]


def interval(diffs: list[float]) -> dict:
    """Mean of per-seed differences and its one-sided 95% bounds (t over seeds)."""
    k = len(diffs)
    mean = sum(diffs) / k
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (k - 1))
    half = t_quantile(k - 1) * sd / math.sqrt(k)
    return {"mean": mean, "lcb": mean - half, "ucb": mean + half, "sd": sd, "seeds": k, "per_seed": diffs}


def pair_difference(result: dict, world: str, a: str, b: str) -> tuple[float, float, int]:
    """`(rate(a) - rate(b), McNemar variance of that difference, requests)` of the pair, over the
    requests of one seed, from the 2x2 table the harness printed."""
    pairs = result["worlds"][world]["pairs"]
    if f"{a}__{b}" in pairs:
        table, sign = pairs[f"{a}__{b}"], 1
    else:
        table, sign = pairs[f"{b}__{a}"], -1
    only_a, only_b = table["only_a"], table["only_b"]
    n = only_a + only_b + table["both"] + table["neither"]
    diff = sign * (only_a - only_b) / n
    variance = ((only_a + only_b) - (only_a - only_b) ** 2 / n) / n**2
    return diff, variance, n


def seed_differences(results: dict[int, dict], world: str, a: str, b: str) -> list[float]:
    return [pair_difference(results[seed], world, a, b)[0] for seed in sorted(results)]


def rate(result: dict, world: str, arm: str) -> float:
    cell = result["worlds"][world]["arms"][arm]
    return cell["successes"] / cell["requests"]


def per_request(result: dict, world: str, arm: str, field: str) -> float:
    cell = result["worlds"][world]["arms"][arm]
    return cell[field] / cell["requests"]


# The first draft of a request is the same in every arm (a function of the seed and the request id, not of
# any arm's state), so the L3 population (the requests whose first candidate carried the unrepairable code,
# read from the loop without L3) is fixed before any arm acts. The count of (arm, request) pairs whose first
# candidate differs from the full arm's must be exactly 0 in every world of every seed; otherwise arm state
# reached round 0, a harness defect that touches every paired contrast and the L3 population alike: INVALID.
NON_DISCRIMINATING = "non_discriminating"
# The primary world cannot discriminate these three contrasts by construction (see the preregistration,
# section 3 and 10); their hypotheses are the two secondary hypotheses and, for the self-model, untested.
UNTESTED_IN_PRIMARY = ("no_workspace", "no_selfmodel", "no_l3")


def label(arm: str, entry: dict, margin: float) -> str:
    if arm in UNTESTED_IN_PRIMARY:
        return NON_DISCRIMINATING
    if entry["lcb"] > margin:
        return "helps"
    if entry["ucb"] < -margin:
        return "hurts"
    return "no effect shown"


def secondary_verdict(entry: dict, margin: float) -> str:
    if entry["lcb"] > margin:
        return "SUPPORTED"
    if entry["ucb"] < margin:
        return "REFUTED"
    return NOT_SHOWN


def population_difference(result: dict, world: str, field: str) -> tuple[float, int]:
    table = result["worlds"][world][field]
    n = table["both"] + table["only_a"] + table["only_b"] + table["neither"]
    if n == 0:
        raise Unreadable(f"{world}.{field} is empty")
    return (table["only_a"] - table["only_b"]) / n, n


def mean_rate(results: dict[int, dict], world: str, arm: str) -> float:
    return sum(rate(results[s], world, arm) for s in results) / len(results)


def analyse(results: dict[int, dict], decision: dict) -> dict:
    """Every interval the decision and the report need, from the seeds' harness results. Pure.
    Raises Unreadable when a result lacks something the analysis reads (the verdict is then
    INVALID, never a crash)."""
    try:
        return _analyse(results, decision)
    except (KeyError, TypeError, ZeroDivisionError, IndexError) as error:
        raise Unreadable(f"the harness results lack what the analysis reads: {error!r}") from error


def _analyse(results: dict[int, dict], decision: dict) -> dict:
    primary, null = decision["primary_world"], decision["null_world"]
    margin = decision["margin_permille"] / 1000
    band = decision["equivalence_band_permille"] / 1000
    component = decision["component_margin_permille"] / 1000
    out: dict = {"seeds": sorted(results), "margin": margin}

    def banded(entry: dict) -> dict:
        entry["verdict"] = "above" if entry["lcb"] > margin else "below" if entry["ucb"] < margin else "straddles"
        return entry

    out["baselines"] = {b: banded(interval(seed_differences(results, primary, "full", b))) for b in decision["baselines"]}
    out["robust_without_l3"] = {
        b: banded(interval(seed_differences(results, primary, "no_l3", b))) for b in decision["baselines"]
    }
    out["ablations"] = {}
    for arm in ("no_workspace", "no_finding", "no_selfmodel", "no_l3"):
        entry = interval(seed_differences(results, primary, "full", arm))
        entry["label"] = label(arm, entry, component)
        out["ablations"][arm] = entry
    entry = interval(seed_differences(results, primary, "no_l3", "no_finding"))
    entry["label"] = label("no_finding", entry, component)
    out["finding_routing_without_l3"] = entry
    out["oracle_minus_full"] = interval(seed_differences(results, primary, "oracle", "full"))

    secondary = {}
    spec = decision["secondary"]["workspace"]
    entry = interval(seed_differences(results, spec["world"], "full", "no_workspace"))
    entry["margin"] = spec["margin_permille"] / 1000
    entry["verdict"] = secondary_verdict(entry, entry["margin"])
    secondary["workspace"] = entry
    spec = decision["secondary"]["l3"]
    pops = [population_difference(results[s], spec["world"], "l3_population") for s in sorted(results)]
    entry = interval([d for d, _ in pops])
    entry["margin"] = spec["margin_permille"] / 1000
    entry["requests_per_seed"] = [n for _, n in pops]
    entry["verdict"] = secondary_verdict(entry, entry["margin"])
    secondary["l3"] = entry
    secondary["self_model"] = {"verdict": "UNTESTED", "reason": "no discriminating condition in this design"}
    out["secondary"] = secondary

    controls = {}
    shuffled = interval(seed_differences(results, primary, "full", "shuffled"))
    shuffled["passes"] = shuffled["lcb"] >= decision["shuffled_min_loss_permille"] / 1000
    controls["shuffled_loses"] = shuffled
    trained = interval(seed_differences(results, primary, "gate", "no_finding"))
    trained["passes"] = trained["lcb"] >= decision["baseline_min_gain_permille"] / 1000
    controls["gate_is_trained"] = trained
    singles = [
        (c["successes"] - c["first_pass"]) / c["requests"]
        for c in (results[s]["worlds"][primary]["arms"]["bon"] for s in sorted(results))
    ]
    single = interval(singles)
    single["passes"] = single["lcb"] >= decision["bon_min_gain_permille"] / 1000
    controls["bon_beats_single_shot"] = single
    oracle = out["oracle_minus_full"]
    controls["oracle_is_an_upper_reference"] = {**oracle, "passes": oracle["lcb"] >= -decision["oracle_tolerance_permille"] / 1000}
    for left in ("no_l3", "full"):
        for other in ("gate", "no_finding", "shuffled"):
            entry = interval(seed_differences(results, null, left, other))
            entry["passes"] = entry["lcb"] >= -band and entry["ucb"] <= band
            controls[f"null_tie_{left}_vs_{other}"] = entry
    discordant = 0
    for world in (primary, null):
        for seed in results:
            table = results[seed]["worlds"][world]["pairs"]["full__no_workspace"]
            discordant += table["only_a"] + table["only_b"]
    controls["workspace_withheld_changes_nothing"] = {"discordant": discordant, "passes": discordant == 0}
    digests = sorted({str(results[s]["procedure"]["source_digest"]) for s in results})
    controls["one_harness_for_every_seed"] = {"digests": digests, "passes": len(digests) == 1}
    differing = sum(world["first_candidates_differ"] for s in results for world in results[s]["worlds"].values())
    population = sum(population_difference(results[s], primary, "l3_population")[1] for s in results)
    controls["first_draft_is_shared"] = {
        "first_candidates_differ": differing,
        "population": population,
        "passes": differing == 0,
    }
    model = decision["model"]
    agreement = {}
    for arm, predicted in model["predicted_permille"].items():
        measured = mean_rate(results, primary, arm)
        agreement[arm] = {"measured": measured, "predicted": predicted / 1000,
                          "passes": abs(measured - predicted / 1000) <= model["agreement_permille"] / 1000}
    controls["harness_agrees_with_planning_model"] = {"arms": agreement, "passes": all(v["passes"] for v in agreement.values())}
    saturated = {}
    for arm in ("full", "gate", "bon"):
        measured = mean_rate(results, primary, arm)
        saturated[arm] = measured
    controls["no_arm_at_ceiling_or_floor"] = {
        "rates": saturated,
        "passes": all(model["floor_permille"] / 1000 <= v <= model["ceiling_permille"] / 1000 for v in saturated.values()),
    }
    out["controls"] = controls

    cost = {}
    for arm in ARMS:
        cost[arm] = {
            "success_rate": mean_rate(results, primary, arm),
            "verifier_calls_per_request": sum(per_request(results[s], primary, arm, "verifier_calls") for s in results) / len(results),
            "pod_calls_per_request": sum(per_request(results[s], primary, arm, "pod_calls") for s in results) / len(results),
            "microseconds_per_request": sum(per_request(results[s], primary, arm, "elapsed_ns") for s in results) / len(results) / 1000,
        }
    out["cost_primary_world"] = cost

    out["descriptive_worlds"] = {}
    for world in decision["descriptive_worlds"]:
        if any(world not in results[s]["worlds"] for s in results):
            continue
        rows = {}
        for baseline in decision["baselines"]:
            rows[f"full_vs_{baseline}"] = interval(seed_differences(results, world, "full", baseline))
        for arm in ("no_workspace", "no_finding", "no_selfmodel", "no_l3", "shuffled", "oracle"):
            rows[f"full_vs_{arm}"] = interval(seed_differences(results, world, "full", arm))
        out["descriptive_worlds"][world] = rows
    # Best-of-N and the gate where nothing is specialised: descriptive, to show what "having a
    # repair Pod" is worth by itself.
    out["null_world_descriptive"] = {
        "full_vs_bon": interval(seed_differences(results, null, "full", "bon")),
        "gate_vs_bon": interval(seed_differences(results, null, "gate", "bon")),
    }
    return out


COMPLETED_WORDING = (
    "a verifier-closed loop with finding-routed repair beats best-of-N at equal verifier budget and beats the "
    "task-feature gate and the shuffled finding map, in a world whose Pod skills are declared (a consequence of "
    "those tables, not a result about a model)"
)


def decide(
    seeds: list[dict],
    analysis: dict | None,
    declared: list[int],
    decision: dict,
    mutations: dict | None = None,
    analysis_problem: str | None = None,
) -> dict:
    """The verdict from the seeds' judgements, the analysis (None when none could be made) and the
    mutation evidence. INVALID takes precedence over FALSIFIED: a product defect seen in a run that
    cannot be trusted is reported as a note, not as a verdict. Pure."""
    present = sorted(entry["seed"] for entry in seeds)
    problems = []
    missing = sorted(set(declared) - set(present))
    duplicate = sorted({s for s in present if present.count(s) > 1})
    extra = sorted(set(present) - set(declared))
    if missing:
        problems.append(f"no judgement of seeds {missing}")
    if duplicate or extra:
        problems.append(f"judgements of seeds {duplicate + extra} that are duplicated or not declared")
    if mutations is None:
        problems.append("no mutation evidence (results/mutations.json)")
    elif mutations.get("killed") != mutations.get("total"):
        problems.append(f"the mutation check killed {mutations.get('killed')} of {mutations.get('total')} mutations")
    unjudged = [entry["seed"] for entry in seeds if not entry["valid"]]
    if unjudged:
        problems.append(f"seeds {unjudged} cannot be judged")
    tripped = [entry["seed"] for entry in seeds if entry["valid"] and entry["falsifiers_failed"]]
    unmet = [entry["seed"] for entry in seeds if entry["valid"] and entry["coverage_failed"]]
    seed_controls = [entry["seed"] for entry in seeds if entry["valid"] and entry["controls_failed"]]
    if unmet:
        problems.append(f"coverage unmet on seeds {unmet}: the run is vacuous")
    if seed_controls:
        problems.append(f"controls failed on seeds {seed_controls}")
    if analysis_problem:
        problems.append(analysis_problem)
    controls_failed = []
    baselines = {}
    if analysis is not None:
        controls_failed = [name for name, entry in analysis["controls"].items() if not entry["passes"]]
        baselines = analysis["baselines"]
    if controls_failed:
        problems.append(f"controls failed over the seeds: {controls_failed}")
    reasons = []
    if problems or analysis is None:
        verdict = INVALID
        if tripped:
            reasons.append(f"integrity falsifiers also tripped on seeds {tripped} (noted; the run is not trusted)")
    elif tripped:
        verdict = FALSIFIED
        reasons.append(f"integrity falsifiers tripped on seeds {tripped}")
    elif all(entry["verdict"] == "above" for entry in baselines.values()):
        verdict = COMPLETED
        reasons.append("COMPLETED means: " + COMPLETED_WORDING)
        reasons.append("the routing claim rests on the contrasts to the gate and to the shuffled map, not on best-of-N, "
                       "which also lacks any repair Pod; see the mild-world and null-world rows")
    elif any(entry["verdict"] == "below" for entry in baselines.values()):
        verdict = FALSIFIED
        reasons.append("the full loop is below the margin against " + ", ".join(
            name for name, entry in baselines.items() if entry["verdict"] == "below"))
    else:
        verdict = NOT_SHOWN
    return {
        "verdict": verdict,
        "reasons": reasons,
        "problems": problems,
        "falsified_seeds": tripped,
        "invalid_seeds": unjudged,
        "controls_failed": controls_failed,
        "recommended_status": {COMPLETED: "completed", FALSIFIED: "failed"}.get(verdict),
    }


# ---- reporting ---------------------------------------------------------------------------------


def points(value: float) -> str:
    return f"{100 * value:+.2f}"


def report(verdict: dict, seeds: list[dict], analysis: dict | None, mutations: dict | None) -> str:
    lines = [f"# F007 vcw-loop: {verdict['verdict']}", ""]
    lines.append(f"Seeds: {', '.join(str(e['seed']) for e in sorted(seeds, key=lambda e: e['seed']))}.")
    lines.append("")
    for entry in sorted(seeds, key=lambda e: e["seed"]):
        state = "valid" if entry["valid"] else "INVALID"
        lines.append(
            f"- seed {entry['seed']}: {state}; {len(entry['coverage_failed'])} coverage rules unmet, "
            f"{len(entry['falsifiers_failed'])} falsifiers tripped, {len(entry['controls_failed'])} controls failed"
        )
        for key in ("invalid", "coverage_failed", "falsifiers_failed", "controls_failed"):
            lines.extend(f"    - {key}: {item}" for item in entry[key])
    if mutations is not None:
        lines.append(f"- mutation check: {mutations.get('killed')} of {mutations.get('total')} killed")
    for problem in verdict["problems"]:
        lines.append(f"- problem: {problem}")
    for reason in verdict["reasons"]:
        lines.append(f"- reason: {reason}")
    if analysis is not None:
        lines += ["", f"Margin: {analysis['margin'] * 100:.1f} points. Differences are in points of verified-success "
                  "rate per request, paired within seed, mean over seeds, one-sided 95% t bounds.", ""]
        lines.append("| comparison | mean | lower | upper | reading |")
        lines.append("|---|---|---|---|---|")
        for name, entry in analysis["baselines"].items():
            lines.append(f"| full - {name} | {points(entry['mean'])} | {points(entry['lcb'])} | {points(entry['ucb'])} | {entry['verdict']} the margin |")
        for name, entry in analysis["robust_without_l3"].items():
            lines.append(f"| no_l3 - {name} (robustness) | {points(entry['mean'])} | {points(entry['lcb'])} | {points(entry['ucb'])} | {entry['verdict']} the margin |")
        for name, entry in analysis["ablations"].items():
            reading = entry["label"].replace("_", "-") + (" in the primary world (see the secondary hypotheses)" if entry["label"] == NON_DISCRIMINATING else "")
            lines.append(f"| full - {name} | {points(entry['mean'])} | {points(entry['lcb'])} | {points(entry['ucb'])} | {reading} |")
        entry = analysis["finding_routing_without_l3"]
        lines.append(f"| no_l3 - no_finding | {points(entry['mean'])} | {points(entry['lcb'])} | {points(entry['ucb'])} | {entry['label']} |")
        entry = analysis["oracle_minus_full"]
        lines.append(f"| oracle - full (upper reference) | {points(entry['mean'])} | {points(entry['lcb'])} | {points(entry['ucb'])} | descriptive |")
        lines += ["", "Secondary hypotheses (preregistered, margins derived from the planning model, never gate the verdict):", ""]
        for name, entry in analysis["secondary"].items():
            if "mean" in entry:
                note = (" (low power, about 0.2 at the modelled gain: a NOT_SHOWN or REFUTED here is not evidence against L3)"
                        if name == "l3" else (" (a harness-defined reader of the broadcast: an implementation check, not the value of the workspace)" if name == "workspace" else ""))
                lines.append(f"- {name}: {points(entry['mean'])} [{points(entry['lcb'])}, {points(entry['ucb'])}] against margin {points(entry['margin'])}: {entry['verdict']}{note}")
            else:
                lines.append(f"- {name}: {entry['verdict']} ({entry['reason']})")
        lines += ["", "Controls:", ""]
        for name, entry in analysis["controls"].items():
            if "mean" in entry:
                lines.append(f"- {name}: {points(entry['mean'])} [{points(entry['lcb'])}, {points(entry['ucb'])}] {'passes' if entry['passes'] else 'FAILS'}")
            elif "digests" in entry:
                lines.append(f"- {name}: {'passes' if entry['passes'] else 'FAILS'} ({len(entry['digests'])} distinct source digests)")
            elif "first_candidates_differ" in entry:
                lines.append(f"- {name}: {entry['first_candidates_differ']} (arm, request) pairs whose first draft differed from the full arm's, L3 population {entry['population']} {'passes' if entry['passes'] else 'FAILS'}")
            elif "discordant" in entry:
                lines.append(f"- {name}: discordant requests {entry['discordant']} {'passes' if entry['passes'] else 'FAILS'}")
            else:
                lines.append(f"- {name}: {'passes' if entry['passes'] else 'FAILS'} ({json.dumps(entry.get('arms', entry.get('rates')))})")
        lines += ["", "Cost per request in the primary world (mean over seeds):", "",
                  "| arm | success | verifier calls | Pod calls | microseconds |", "|---|---|---|---|---|"]
        for arm, row in analysis["cost_primary_world"].items():
            lines.append(f"| {arm} | {row['success_rate']:.3f} | {row['verifier_calls_per_request']:.3f} | {row['pod_calls_per_request']:.3f} | {row['microseconds_per_request']:.0f} |")
        for world, rows in analysis["descriptive_worlds"].items():
            lines += ["", f"Descriptive world {world} (never gates the verdict):", ""]
            for name, entry in rows.items():
                lines.append(f"- {name}: {points(entry['mean'])} [{points(entry['lcb'])}, {points(entry['ucb'])}]")
        lines += ["", "Null world, descriptive (what a repair Pod is worth where nothing is specialised):", ""]
        for name, entry in analysis["null_world_descriptive"].items():
            lines.append(f"- {name}: {points(entry['mean'])} [{points(entry['lcb'])}, {points(entry['ucb'])}]")
    lines += ["", "Limitations: " + "; ".join(LIMITATIONS) + "."]
    return "\n".join(lines) + "\n"


def run_command(command: str) -> tuple[int, str]:
    try:
        process = subprocess.run(shlex.split(command), cwd=ROOT, text=True, capture_output=True)
    except OSError as error:
        return -1, str(error)
    return process.returncode, process.stdout + process.stderr


def static_evidence(commands: list[str], runner: Callable[[str], tuple[int, str]] = run_command) -> list[dict]:
    results = []
    for command in commands:
        code, output = runner(command)
        results.append(
            {"command": command, "exit_code": code,
             "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(), "output_tail": output[-TAIL:]}
        )
    return results


def newest_per_seed(paths: list[Path]) -> dict[int, Path]:
    chosen: dict[int, tuple[bool, Path]] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = int(record["seed"])
        launched = experiment_records.finished_run(record)
        if launched or not chosen.get(seed, (False, path))[0]:
            chosen[seed] = (launched, path)
    return {seed: path for seed, (_, path) in chosen.items()}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_frozen() -> tuple[dict, dict, dict, str]:
    manifest = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
    if "preregistration" not in CONFIG:
        raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table (F007 is not frozen yet)")
    table = CONFIG["preregistration"]
    try:
        digest = experiment_records.preregistration_digest(table)
    except ValueError as error:
        raise SystemExit(f"{EXPERIMENT}: the preregistration cannot be read: {error}")
    if manifest.get("preregistration_sha256") != digest:
        raise SystemExit(
            f"{EXPERIMENT}: refusing to aggregate: experiment.toml names preregistration_sha256 "
            f"{manifest.get('preregistration_sha256')!r}, not {digest}, the digest of config.toml's table"
        )
    for key in ("criteria", "preregistration_text"):
        path = ROOT / table[key]
        actual = experiment_records.preregistered_file_digest(path)
        if table.get(f"{key}_sha256") != actual:
            raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: {table[key]} is not the file frozen ({actual})")
    criteria = tomllib.loads((ROOT / table["criteria"]).read_text(encoding="utf-8"))
    disagreement = preregistration_disagreements(table, criteria["decision"])
    if disagreement:
        raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: " + "; ".join(disagreement))
    return manifest, table, criteria, digest


def preregistration_disagreements(table: dict, decision: dict) -> list[str]:
    """The frozen table and criteria.toml must state the same margins and the same confidence. A key
    still holding a `must-be-...` placeholder is left to the freeze gate."""
    pairs = {
        "primary_margin_permille": decision["margin_permille"],
        "equivalence_band_permille": decision["equivalence_band_permille"],
        "shuffled_control_margin_permille": decision["shuffled_min_loss_permille"],
        "confidence_one_sided_permille": 950,
    }
    out = []
    for key, expected in pairs.items():
        value = table.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value != expected:
            out.append(f"config.toml {key} = {value} but criteria.toml states {expected}")
    return out


def counters(criteria: dict) -> tuple[str, ...]:
    names = []
    for rule in criteria["falsifiers"] + criteria["invalid_if"]:
        names.append(parse_rule(rule)[0])
    return tuple(names)


def aggregate(report_path: Path | None = None) -> None:
    manifest, table, criteria, digest = load_frozen()
    per_seed = newest_per_seed(sorted(RESULTS.glob("run-*-seed-*.json")))
    missing = sorted(set(manifest["seeds"]) - set(per_seed))
    if missing:
        raise SystemExit(f"no run record for seeds {missing}")
    chosen = {seed: json.loads(per_seed[seed].read_text(encoding="utf-8")) for seed in manifest["seeds"]}
    named = {per_seed[seed].name: chosen[seed] for seed in manifest["seeds"]}
    results_path = experiment_records.relative_to_root(RESULTS, ROOT)
    experiment_path = experiment_records.relative_to_root(HERE, ROOT)
    try:
        revision = experiment_records.source_revision(
            EXPERIMENT, manifest, named, ROOT, HERE,
            paths=experiment_records.listed_record_paths(results_path),
            checkout_paths=experiment_records.listed_staleness_paths(results_path, experiment_path),
            listed_experiment=True,
        )
        seed_records = experiment_records.seed_record_digests(RESULTS)
        mutations = experiment_records.mutation_evidence(EXPERIMENT, RESULTS / "mutations.json", BENCHMARK, ROOT, HERE)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: {error}")

    judgements, records, usable = [], [], {}
    for seed in manifest["seeds"]:
        record = chosen[seed]
        result = None
        error = None
        try:
            result = experiment_records.harness_results(
                {per_seed[seed].name: record}, BENCHMARK, counts=counters(criteria)
            )[per_seed[seed].name]
        except experiment_records.ProvenanceError as problem:
            error = str(problem)
        judgement = judge_seed(result, record.get("exit_code"), criteria, seed)
        if error:
            judgement["valid"] = False
            judgement["invalid"].append(error)
        judgements.append(judgement)
        if judgement["valid"] or not judgement["invalid"]:
            usable[seed] = result
        records.append({"seed": seed, "record": per_seed[seed].name, "git_sha": record["git_sha"], "exit_code": record.get("exit_code")})
    analysis, analysis_problem = None, None
    if len(usable) == len(manifest["seeds"]) and all(j["valid"] for j in judgements):
        try:
            analysis = analyse(usable, criteria["decision"])
        except Unreadable as error:
            analysis_problem = str(error)
    verdict = decide(judgements, analysis, manifest["seeds"], criteria["decision"], mutations, analysis_problem)
    text = report(verdict, judgements, analysis, mutations)
    metrics = {
        "experiment_id": EXPERIMENT, "scope": SCOPE, "verdict": verdict["verdict"], "decision": verdict,
        "analysis": analysis, "seeds": judgements, "mutation_checks": mutations, "report": text,
    }
    run = {
        "experiment_id": EXPERIMENT, "status": manifest["status"], "scope": SCOPE, "git_sha": revision,
        "aggregated_at_git_sha": git_sha(), "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": digest, "preregistration_rules_sha256": manifest.get("preregistration_rules_sha256"),
        "criteria_sha256": table["criteria_sha256"], "entrypoint": manifest["entrypoint"], "seeds": records,
        "seed_records": seed_records, "verdict": verdict["verdict"], "recommended_status": verdict["recommended_status"],
        "mutation_checks": mutations, "limitations": LIMITATIONS,
    }
    try:
        experiment_records.publish_aggregate(RESULTS, metrics, run)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"{EXPERIMENT}: {error}")
    if report_path is not None:
        report_path.write_text(text, encoding="utf-8")
    print(f"{EXPERIMENT} verdict={verdict['verdict']} recommended_status={verdict['recommended_status']} "
          f"falsified={verdict['falsified_seeds']} invalid={verdict['invalid_seeds']} controls_failed={verdict['controls_failed']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="also write the Markdown reading of the verdict here")
    args = parser.parse_args()
    aggregate(report_path=args.report)


if __name__ == "__main__":
    main()
