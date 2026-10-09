"""Aggregate the C001 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run C001 --seed <s>` writes
(results/run-<timestamp>-seed-<s>.json): the newest record of every declared seed
whose command ran to an exit code. Nothing of the experiment is run again. Every
declared seed must have a record, the records must be of one configuration and have run
the checkout's code (`experiment_records.source_revision`), and the preregistration
digests of the manifest, the table, the preregistration text and `criteria.toml` must be
the frozen ones.

The decision is mechanical, and it is `criteria.toml`'s (three outcomes plus invalid):

- a seed is judged by applying every coverage rule and every falsifier rule (integer
  counters; the grammar is S001's) to the one JSON object its harness printed; a failure
  is a STRUCTURAL defect of the product arm, the task or the harness: FALSIFIED (kind
  "structural"), judged first;
- validity guards: a seed whose strongest comparator is saturated or floored on the
  primary metric, a workspace arm that did not train on more than the allowed number of
  seeds, or a sensitivity control (a crippled all-to-all) that the paired bound cannot tell
  from the intact one: INVALID. The run says nothing about the claim; it is never FALSIFIED;
- the paired comparisons (workspace minus each comparator, accuracy on the primary metric
  over the declared seeds, one-sided 95% t bounds): COMPLETED (supported) iff every lower
  bound reaches the margin; FALSIFIED (kind "refuted") iff the guards hold and some UPPER
  bound is below the margin; NOT_SHOWN otherwise (the bounds straddle the margin: nothing
  is concluded, the status stays planned, no requirement is downgraded);
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
METRIC_NAMES = (
    "arms",
    "majority_permille",
    "no_comm_id_permille",
    "comparator_primary_permille",
    "workspace_trained",
    "hard_failures",
    "guards_failed",
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

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, INVALID, NOT_SHOWN = "COMPLETED", "FALSIFIED", "INVALID", "NOT_SHOWN"
OPERATORS = (">=", "<=", "==")
CONTROLS = ("n_leak", "overlap", "param_mismatch", "isolation_probe", "bottleneck", "dropout_probe")
NUMBER = re.compile(r"-?[0-9]+")
SCOPE = (
    "a functional-mechanism test, on a CPU toy task, of a learned-attention global workspace (competitive top-k write, gated "
    "slot update, broadcast read-back by every module) against parameter-matched ablations (all-to-all pairwise attention, "
    "bandwidth-matched pairwise attention, no-broadcast, competition off, capacity off, no communication) and a crippled "
    "all-to-all as a sensitivity control: shared-weight modules, a finite training set of eight modules, evaluation on sixteen "
    "modules (primary) and twenty-four (secondary) with more distractors; the primary endpoint is accuracy on sixteen modules, "
    "judged by one-sided 95% t bounds over ten seeds of the paired difference. It does not test the deterministic "
    "ptr-pods::workspace contract, and it makes no claim about consciousness"
)
LIMITATIONS = [
    "a result applies to this task family, its size (eight training modules, d=32, three rounds), the training-set size chosen on pilot seeds and these hyperparameters, tuned on pilot seeds disjoint from the declared ones; the same pilot budget was given to every arm but the workspace had extra structural choices (slots, top-k) made from pilot observations",
    "the t bounds assume roughly normal paired differences, which ten points cannot check; per-seed differences and both bounds are reported; the decision uses the upper bound for refutation and the lower bound for support",
    "the ablations differ from the workspace arm in more than one way at matched parameter counts (update-network width); the no-broadcast arm keeps one reader (the reporter) so that the readout path is shared; the bandwidth-matched arm is one definition of matched bandwidth",
    "a learned-attention network is not the deterministic bid-ordered workspace of ptr-pods::workspace: a supportive result does not validate that contract and a refuting one does not touch it",
    "in distribution (eight modules) the pairwise arms can express everything the workspace can; the claim tested is an inductive-bias one (finite data, more modules), not an expressivity one",
    "sample efficiency (area under the learning curve), module dropout, ID accuracy, the 24-module accuracy, ignition and communication-cost proxies are reported with bounds where they are differences and are never judged",
    "mutation evidence comes from run_mutations.py (scripts/mutation_check.py plants Cargo packages only) and was run at a reduced scale with the system interpreter; experiment_records tracks only Rust and Cargo files as code, so this aggregator separately refuses when research/consciousness or this experiment's files differ from the mutation commit, and when the mutation total is not the plan's size",
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
    if not isinstance(result, dict):
        return {"seed": seed, "valid": False, "invalid": ["no harness result"], "coverage_failed": [], "falsifiers_failed": [], "guards_failed": [], "workspace_trained": None}
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
    # Guards do not enter the harness's exit status: a saturated or floored comparator is not a defect.
    for rule in criteria.get("guards", []):
        try:
            if not evaluate(result, rule):
                guards_failed.append(rule)
        except Unreadable as error:
            invalid.append(f"guards: {error}")
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
        "guards_failed": guards_failed,
        "workspace_trained": trained if trained in (0, 1) else None,
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


def paired_comparisons(results: dict[int, dict], paired: dict) -> dict:
    """The pre-specified comparisons. For every ablation in `against`, the bounds of the mean paired difference
    (`arm` minus the ablation) in `metric` accuracy: SUPPORTED needs every lower bound >= `margin_permille`; REFUTED
    needs one upper bound < `margin_permille`. The sensitivity control (`sensitivity`) must resolve a difference
    of its own margin. Descriptive comparisons carry the same bounds and are never judged. Raises PairedError."""
    t_quantile = Decimal(paired["t_one_sided_95_df9"])
    seeds = sorted(results)
    if len(seeds) != int(paired["seeds_required"]):
        raise PairedError(f"{len(seeds)} seeds, the rule is preregistered for {paired['seeds_required']}")
    margin = paired["margin_permille"]
    judged = []
    for other in paired["against"]:
        diffs = differences(results, seeds, paired["metric"], paired["arm"], other)
        bound = paired_bound(diffs, t_quantile)
        judged.append(
            {
                "against": other,
                "metric": paired["metric"],
                "per_seed_permille": [float(d * 1000) for d in diffs],
                "supports": bound["lower_bound_permille"] >= margin,
                "refutes": bound["upper_bound_permille"] < margin,
                "margin_permille": margin,
                **bound,
            }
        )
    sens = paired["sensitivity"]
    sens_bound = paired_bound(differences(results, seeds, paired["metric"], sens["arm"], sens["crippled"]), t_quantile)
    sensitivity = {"arm": sens["arm"], "crippled": sens["crippled"], "margin_permille": sens["margin_permille"], "resolves": sens_bound["lower_bound_permille"] >= sens["margin_permille"], **sens_bound}
    descriptive = []
    for metric in paired.get("descriptive_metrics", []):
        for other in paired["against"] + paired.get("descriptive_against", []):
            descriptive.append({"against": other, "metric": metric, **paired_bound(differences(results, seeds, metric, paired["arm"], other), t_quantile)})
    return {
        "supported": all(item["supports"] for item in judged),
        "refuted_against": [item["against"] for item in judged if item["refutes"]],
        "judged": judged,
        "sensitivity": sensitivity,
        "descriptive": descriptive,
    }


def decide(
    seeds: list[dict],
    static: list[dict] | None,
    declared: list[int],
    mutations: dict | None = None,
    paired: dict | None = None,
    untrained_workspace_seeds: int = 0,
    max_untrained: int = 2,
    plan_size: int | None = None,
) -> dict:
    """The verdict from the seeds' judgements (`judge_seed`, in any order), the static-evidence results, the mutation
    evidence (`killed`, `total`), the paired comparisons and the count of seeds whose workspace arm did not train.
    Four outcomes:

    - FALSIFIED, kind "structural": a valid seed fails a coverage rule or a hard counter (a defect of the product arm,
      the task or the harness; always judged first);
    - INVALID: no judgement is possible or the run is uninformative (seeds missing or unjudgeable, mutation evidence
      missing, incomplete or not the plan's, a saturated or floored comparator on any seed, a workspace arm that failed
      to train on more than `max_untrained` seeds, a sensitivity control that cannot resolve its margin);
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
    static_failed = [item["command"] for item in static or [] if item.get("exit_code") != 0]
    kind = None
    if unmet or tripped or static_failed:
        verdict, kind = FALSIFIED, "structural"
    elif problems or unjudged:
        verdict = INVALID
    elif uninformative:
        verdict = INVALID
        problems.append(f"the strongest comparator is saturated or floored on seeds {uninformative}: the run says nothing about the claim")
    elif untrained_workspace_seeds > max_untrained:
        verdict = INVALID
        problems.append(f"the workspace arm did not train on {untrained_workspace_seeds} seeds (more than {max_untrained}): an optimisation finding, reported, not a verdict")
    elif paired is None:
        verdict = INVALID
        problems.append("the paired comparison could not be computed")
    elif not paired["sensitivity"]["resolves"]:
        verdict = INVALID
        problems.append("the sensitivity control does not resolve a difference of its margin: the pipeline is insensitive at this operating point")
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
        "uninformative_seeds": uninformative,
        "untrained_workspace_seeds": untrained_workspace_seeds,
        "static_evidence_failed": static_failed,
        "refuted_against": (paired or {}).get("refuted_against", []),
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
                     f"{len(entry['falsifiers_failed'])} falsifier rules tripped")
        for key in ("invalid", "coverage_failed", "falsifiers_failed"):
            lines.extend(f"    - {key}: {item}" for item in entry[key])
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
        sens = paired["sensitivity"]
        lines.append(f"- sensitivity control ({sens['arm']} minus {sens['crippled']}): lower {sens['lower_bound_permille']:.1f} vs margin {sens['margin_permille']}: {'resolves' if sens['resolves'] else 'DOES NOT RESOLVE'}")
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
    """The record to aggregate for each seed: the newest by name of those whose run
    finished, the newest of the others only where none finished."""
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


GUARDED_PATHS = ("research/consciousness", "experiments/consciousness/C001-global-workspace")
RESULTS_EXCLUDE = ":(exclude)experiments/consciousness/C001-global-workspace/results"


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

    judgements, records, metrics_seeds = [], [], []
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
        metrics_seeds.append(
            {
                **judgement,
                "metrics": {name: result.get(name) for name in metric_names} if result else None,
                "summary": result.get("summary") if result else None,
            }
        )
    static = static_evidence(criteria.get("static_evidence", []))
    paired = None
    all_valid = all(entry["valid"] for entry in judgements)
    if all_valid:
        try:
            paired = paired_comparisons({seed: result_of[seed] for seed in manifest["seeds"]}, criteria["paired"])
        except PairedError as error:
            judgements[0]["invalid"].append(f"paired comparison: {error}")
            judgements[0]["valid"] = False
    untrained = sum(1 for entry in judgements if entry.get("workspace_trained") == 0)
    plan = tomllib.loads((HERE / "tests/mutations.toml").read_text(encoding="utf-8"))
    verdict = decide(
        judgements, static, manifest["seeds"], mutations, paired, untrained,
        int(criteria["paired"]["max_untrained_workspace_seeds"]), len(plan["mutation"]),
    )
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
          f"uninformative={verdict['uninformative_seeds']} refuted_against={verdict['refuted_against']} problems={len(verdict['problems'])}")
    return verdict["verdict"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="also write the Markdown reading of the verdict here")
    args = parser.parse_args()
    verdict = aggregate(report_path=args.report)
    sys.exit({COMPLETED: 0, FALSIFIED: 1, NOT_SHOWN: 3}.get(verdict, 2))


if __name__ == "__main__":
    main()
