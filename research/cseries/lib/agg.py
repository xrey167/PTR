"""The mechanics every stage-A aggregator shares: the rule grammar, one seed's judgement, exact paired bounds, the run-record
provenance checks and the publication of metrics.json and run.json.

What an experiment concludes (its guards, comparisons and verdict) is in its own aggregate.py; this module reads no experiment's
configuration. The grammar and the provenance handling follow C001's aggregator, which this does not import (C001 is frozen)."""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from stats import PairedError, median, paired_bound  # noqa: F401  (re-exported)

COMPLETED, FALSIFIED, INVALID, NOT_SHOWN = "COMPLETED", "FALSIFIED", "INVALID", "NOT_SHOWN"
OPERATORS = (">=", "<=", "==")
NUMBER = re.compile(r"-?[0-9]+")
ENVIRONMENT_KEYS = ("python", "torch", "torch_config_sha256", "cpu_model", "cpu_flags_sha256", "uv_lock_sha256", "pyproject_sha256")
TAIL = 2000


class Unreadable(Exception):
    """A rule that cannot be read or resolved: the seed is INVALID."""


def parse_rule(rule: str) -> tuple[str, str, str]:
    """`(left, op, right)` of a rule: exactly three tokens separated by single spaces with a known operator."""
    tokens = rule.split(" ") if isinstance(rule, str) else []
    if len(tokens) != 3 or tokens[1] not in OPERATORS or not all(tokens):
        raise Unreadable(f"not '<left> <op> <right>' with an operator of {OPERATORS}: {rule!r}")
    return tokens[0], tokens[1], tokens[2]


def resolve(result: dict, path: str):
    """The value at the dotted path of object keys from the root of `result`."""
    at = result
    for key in path.split("."):
        if not isinstance(at, dict) or key not in at:
            raise Unreadable(f"{path} does not resolve in the harness result")
        at = at[key]
    return at


def operand(token: str, result: dict, decimal: bool, equality: bool):
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


def judge_seed(result: dict | None, exit_code, criteria: dict, seed: int, benchmark: str, experiment: str, controls_required: tuple,
               count_keys: tuple = ()) -> dict:
    """One seed's judgement: the coverage and falsifier rules that fail, the integrity counters above 0, the guards that fail, the
    reasons it cannot be judged. `count_keys` names result fields that must be non-negative integers (the guard scalars the
    experiment's decision reads). Pure."""
    invalid: list[str] = []
    coverage_failed: list[str] = []
    falsifiers_failed: list[str] = []
    guards_failed: list[str] = []
    integrity_failed: list[str] = []
    if not isinstance(result, dict):
        return {"seed": seed, "valid": False, "invalid": ["no harness result"], "coverage_failed": [], "falsifiers_failed": [],
                "integrity_failed": [], "guards_failed": []}
    if result.get("benchmark") != benchmark or result.get("experiment") != experiment:
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
    for name in controls_required:
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
    for rule in criteria.get("integrity", []):
        try:
            if not evaluate(result, rule):
                integrity_failed.append(rule)
        except Unreadable as error:
            invalid.append(f"integrity: {error}")
    if integrity_failed:
        invalid.append("harness integrity counters above 0: " + ", ".join(integrity_failed))
    for rule in criteria.get("guards", []):
        try:
            if not evaluate(result, rule):
                guards_failed.append(rule)
        except Unreadable as error:
            invalid.append(f"guards: {error}")
    for key in count_keys:
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            invalid.append(f"{key} is {value!r}, not a count")
    environment = result.get("environment")
    if not isinstance(environment, dict) or not all(isinstance(environment.get(k), str) and environment.get(k) for k in ENVIRONMENT_KEYS):
        invalid.append("the result records no complete environment (python, torch, CPU, lock digests)")
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
    }


def arm_fraction(result: dict, arm: str, metric: str) -> Fraction:
    """Accuracy of `arm` on `metric` as an exact fraction of the counts the harness reports (`<metric>_correct`, `<metric>_total`)."""
    try:
        record = result["arms"][arm]
        correct, total = record[f"{metric}_correct"], record[f"{metric}_total"]
    except (KeyError, TypeError) as error:
        raise PairedError(f"arm {arm!r} has no {metric} counts: {error}") from error
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (correct, total)) or total <= 0 or not 0 <= correct <= total:
        raise PairedError(f"arm {arm!r} reports {correct!r} correct of {total!r}")
    return Fraction(correct, total)


def differences(results: dict, seeds: list, metric: str, left: str, right: str) -> list[Fraction]:
    return [arm_fraction(results[s], left, metric) - arm_fraction(results[s], right, metric) for s in seeds]


def bound_of(results: dict, seeds: list, metric: str, left: str, right: str, t_value: str) -> dict:
    """The paired bound of `left` minus `right` on `metric`, with the per-seed differences and their median."""
    diffs = differences(results, seeds, metric, left, right)
    return {
        "arms": f"{left} - {right}",
        "metric": metric,
        "per_seed_permille": [float(d * 1000) for d in diffs],
        "median_permille": float(median(diffs) * 1000),
        "positive_seeds": sum(1 for d in diffs if d > 0),
        **paired_bound(diffs, Decimal(t_value)),
    }


def environment_mismatches(environments: dict) -> dict:
    """seed -> the environment keys in which its recorded environment differs from the first recorded one."""
    reference = next((e for e in environments.values() if isinstance(e, dict)), None)
    return {
        seed: [k for k in sorted(set(environment) | set(reference)) if environment.get(k) != reference.get(k)]
        for seed, environment in environments.items()
        if isinstance(environment, dict) and environment != reference
    }


def invalidate_environment_mismatches(judgements: list, environments: dict) -> dict:
    mismatches = environment_mismatches(environments)
    for entry in judgements:
        if entry["seed"] in mismatches:
            entry["valid"] = False
            entry["invalid"].append("its environment differs from that of the other seeds: " + ", ".join(mismatches[entry["seed"]]))
    return mismatches


def git(root: Path, *args: str):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def code_changed_since(sha: str, root: Path, guarded: tuple, exclude: str) -> str:
    """Why the harness code is not the code the mutation evidence ran at ("" when it is). scripts/experiment_records tracks only Rust
    and Cargo files as code, so a change to the Python harness, the task, the models or this experiment's own files between the
    mutation commit and now would go unseen; this check closes that window locally. Any difference to `sha`, committed or not."""
    committed = git(root, "diff", "--name-only", sha, "HEAD", "--", *guarded, exclude)
    if committed.returncode != 0:
        return f"git cannot compare {sha} with HEAD: {committed.stderr.strip()}"
    if committed.stdout.strip():
        return "these files changed since the mutation commit " + sha[:12] + ": " + ", ".join(committed.stdout.split())
    dirty = git(root, "status", "--porcelain", "--", *guarded, exclude)
    if dirty.stdout.strip():
        return "uncommitted changes in " + ", ".join(line[3:] for line in dirty.stdout.splitlines())
    return ""


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))
