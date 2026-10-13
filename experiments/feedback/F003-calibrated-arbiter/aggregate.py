"""Aggregate the F003 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run F003 --seed <s>` writes
(results/run-<timestamp>-seed-<s>.json): the newest record of every declared
seed whose command ran to an exit code. Nothing of the experiment is run again.
Every declared seed must have a record, the records must be of one
configuration and have run the checkout's code, this script included
(`experiment_records.source_revision`), and the preregistration digests of the
manifest, the table, `PREREGISTRATION.md`, `PROTOCOL.md` and `criteria.toml` must be the
frozen ones.

The decision is mechanical, and it is `criteria.toml`'s (its header defines the
rule grammar, which `evaluate` below implements):

- a seed is judged by applying every coverage rule and every falsifier rule to
  the one JSON object its harness printed;
- COMPLETED iff every seed holds every coverage rule and every falsifier rule;
- FALSIFIED if some valid seed fails a rule (a coverage rule that fails makes
  the run vacuous, which counts as falsified);
- INVALID otherwise when a seed cannot be judged: no result line, a harness
  that exited 2 or by a signal (a panic of the product is not one of these: the harness
  contains it, counts it in `product_panics` and exits 1, which falsifies; only a panic in
  the harness's own code, exit 101, is INVALID), another benchmark, seed, scale or experiment
  than the record's, harness gaps, a rule whose counter is missing, null or of
  the wrong type, or an exit status that disagrees with the rules (0 with a
  failed rule, 1 with none). INVALID says nothing about the hypothesis.

This experiment has no static evidence (criteria.toml names none). COMPLETED additionally requires
the mutation evidence, `results/mutations.json` (`scripts/mutation_check.py F003`): it must be this
checkout's code, checker and plan (`experiment_records.mutation_evidence`) and every mutation of the
plan must be killed (`killed == total == len(plan)`, `mutations_ok`). The experiment's only claim to
discriminate a sound arbiter from a broken one is that plan, so a pass over a harness not shown to
fail would be vacuous. Without it (absent, or fewer killed than planned) the verdict is INVALID, never
COMPLETED; a FALSIFIED verdict is not downgraded (a falsification the harness produced stays a
finding), and `mutations_ok` is reported next to it. run.json carries the summary, and the report
names it.

metrics.json and run.json are published as one aggregate
(`experiment_records.publish_aggregate`). `--report PATH` also writes the
Markdown reading of the verdict to PATH (not into the results directory: a file
added there after the seeds ran would make the archived results stale).
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
from decimal import Decimal
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
BENCHMARK = "calibrated-arbiter"
EXPERIMENT = "F003"
TAIL = 2000

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, INVALID = "COMPLETED", "FALSIFIED", "INVALID"
OPERATORS = (">=", "<=", "==")
NUMBER = re.compile(r"-?[0-9]+")
SCOPE = (
    "the calibrated arbiter of ptr-branch (TriagePolicy::triage, calibration_draw, TriageOutcome::adjudicate, "
    "PolicyRecord::calibrate and held_out, certify_threshold, calibrate_threshold, evaluate_off_policy, doubly_robust) "
    "and the Clopper-Pearson bound of ptr-analytics, on a simulated branch stream with a deliberately overconfident "
    "scorer and a SIMULATED adjudicator with a known error rate (PROTOCOL.md; no person adjudicates anything): the "
    "Learn-then-Test guarantee holds over Monte-Carlo replicates (violations of the true adjudicated harm bound at most "
    "delta of the certified thresholds, within an exact-binomial tolerance), each recorded policy passes an exact "
    "one-sided held-out test of harm at most alpha at its Bonferroni share, no ineligible branch is auto-proposed, "
    "every decision, threshold, held-out set and bound equals an independent oracle's, and off-policy estimates of "
    "supported targets are unbiased while an unsupported target is refused; the claim is the mechanism's"
)
LIMITATIONS = [
    "the adjudicator is a simulation (PROTOCOL.md): label = true harm XOR a hash-keyed flip at 2 per mille; nothing here says how well people judge harm, how consistent they are or what they cost; a real-adjudication successor needs its own protocol",
    "the claim is scoped to the mechanism; the scorer, the namespaces' score and harm tables and the verification mix are synthetic, one process, one machine profile",
    "the adjudicator's error rate is 2 per mille, not 20: Learn-then-Test starts its sequence at the first threshold with at least 59 admitted samples and stops at the first failure, so at 20 per mille the policy would certify nothing in about three calibrations of four and the experiment would have no subject (arithmetic in PREREGISTRATION.md); the guarantee is tested at the error rate the mechanism can use, and a noisier adjudicator is expected to escalate more, not to violate the bound",
    "the held-out decision rule is the exact one-sided test of harm at most alpha, Bonferroni over 4 policies x 5 seeds; the literal rule of the manifest (an upper confidence bound above alpha on held-out adjudications is a hard failure) is reported and judged by nothing, because a correct policy fails it by construction (its true harm rate at the certified threshold approaches alpha from below as calibration grows)",
    "coverage over replicates is tested against delta with an exact tolerance, so a defect that raises the violation rate by less than the tolerance (about 1.6 percentage points at 2000 replicates) is not seen by that rule alone; the oracle comparisons (threshold_mismatches) see such defects",
    "mutants that the plan cannot kill are documented equivalents: `<=` to `<` in the Learn-then-Test test (a bound equal to alpha has probability zero) and swapping alpha and delta (both are 0.05)",
    "off-policy evaluation tests unbiasedness of IPS and doubly robust estimates of two supported targets over 200 logs per seed at four standard errors; SNIPS is biased by design and descriptive; a bias smaller than the standard error of the mean over the logs is not seen",
    "the slice-rate z tests (overall, per namespace, per score half) detect a non-uniform calibration_draw only when it moves the slice share by more than four standard deviations of the group",
    "the harm breakdown by key namespace is descriptive and gates nothing",
    "the 30 percent of scores placed exactly on a grid point are a test-sensitivity device that makes >= against > observable; with real-valued scores that mutant would be equivalent",
    "the start rule of the fixed sequence is data-dependent; that the coverage guarantee holds with it is shown empirically for this scenario by the Monte-Carlo count, not proven; the ungated 20 per mille arm (descriptive.noisy_adjudicator_arm) reports how rarely it certifies under a noisier adjudicator",
    "the held-out test detects an observed harm rate above about 7 percent and the coverage test (level 1e-3 per seed, not Bonferroni-split over seeds) a violation rate above about 6.5 percent; smaller deviations are seen only by the oracle comparisons",
    "calibration_draw is the product's and is shared with the oracle's expected decision; its uniformity is judged by the slice-rate z tests only",
    "the five seeds were run once at the pinned scale before the freeze (digests in PREREGISTRATION.md); the harness is deterministic, so the confirmatory runs reproduce them and are no blind confirmation",
    "a seed on which no recorded policy certifies makes coverage unmet, which counts as FALSIFIED by the S001/R001 convention though it says nothing about the claim",
    "the mutation checker's evidence (results/mutations.json) is about the harness, not the arbiter; COMPLETED requires it (every planned mutation killed) as a precondition of trusting a pass, so without it the verdict is INVALID, and it says nothing about the arbiter itself",
    "a panic of the product is a falsification (product_panics), contained by the harness; a panic in harness code itself is a harness fault (INVALID); timings come from one shared cloud container; no capacity claim",
]


CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))


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
    if not isinstance(result, dict):
        return {"seed": seed, "valid": False, "invalid": ["no harness result"], "coverage_failed": [], "falsifiers_failed": []}
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
        invalid.append("did not run the pinned stream shape")
    if result.get("harness_gaps") != []:
        invalid.append(f"harness gaps: {result.get('harness_gaps')!r}")
    for key, failed in (("coverage", coverage_failed), ("falsifiers", falsifiers_failed)):
        for rule in criteria[key]:
            try:
                if not evaluate(result, rule):
                    failed.append(rule)
            except Unreadable as error:
                invalid.append(f"{key}: {error}")
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
    }


def plan_size() -> int:
    """How many mutations tests/mutations.toml plans."""
    plan = tomllib.loads((HERE / "tests" / "mutations.toml").read_text(encoding="utf-8"))
    return len(plan.get("mutation", []))


def mutations_ok(mutations: dict | None, planned: int) -> bool:
    """Whether the mutation evidence is there and complete: `killed == total == planned`, with a
    non-empty plan. Pure."""
    if not isinstance(mutations, dict):
        return False
    killed, total = mutations.get("killed"), mutations.get("total")
    counts = all(isinstance(value, int) and not isinstance(value, bool) for value in (killed, total))
    return counts and planned > 0 and killed == total == planned


def decide(seeds: list[dict], static: list[dict] | None, declared: list[int], mutations_ok: bool) -> dict:
    """The verdict from the seeds' judgements (`judge_seed`, in any order), the
    static-evidence results (None when they were not run) and whether the mutation evidence
    is complete (`mutations_ok`). COMPLETED requires it; a FALSIFIED verdict does not depend
    on it. Pure."""
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
    if not mutations_ok:
        problems.append("mutation evidence missing or incomplete")
    unmet = [entry["seed"] for entry in seeds if entry["valid"] and entry["coverage_failed"]]
    tripped = [entry["seed"] for entry in seeds if entry["valid"] and entry["falsifiers_failed"]]
    unjudged = [entry["seed"] for entry in seeds if not entry["valid"]]
    static_failed = [item["command"] for item in static or [] if item.get("exit_code") != 0]
    if unmet or tripped or static_failed:
        verdict = FALSIFIED
    elif problems or unjudged or not mutations_ok:
        verdict = INVALID
    else:
        verdict = COMPLETED
    return {
        "verdict": verdict,
        "mutations_ok": mutations_ok,
        "vacuous_seeds": unmet,
        "falsified_seeds": tripped,
        "invalid_seeds": unjudged,
        "static_evidence_failed": static_failed,
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


def report(verdict: dict, seeds: list[dict], static: list[dict] | None, mutations: dict | None = None) -> str:
    """The Markdown reading of the verdict."""
    lines = [f"# F003 calibrated-arbiter: {verdict['verdict']}", ""]
    lines.append(f"Seeds: {', '.join(str(entry['seed']) for entry in sorted(seeds, key=lambda e: e['seed']))}. "
                 "Every seed must hold every coverage and falsifier rule.")
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
    for problem in verdict["problems"]:
        lines.append(f"- problem: {problem}")
    if mutations:
        lines.append(f"- mutation checks of the harness (required for COMPLETED): "
                     f"{mutations['killed']} of {mutations['total']} killed at {mutations['git_sha']}")
    lines.append(f"- mutations_ok: {verdict['mutations_ok']}")
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


def load_frozen() -> tuple[dict, dict, dict, str]:
    """The manifest, the preregistered table, the criteria and the table's digest, once the
    manifest, the table, the preregistration text and the criteria agree. Raises SystemExit."""
    manifest = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
    if "preregistration" not in CONFIG:
        raise SystemExit(
            f"{EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table (F003 is not frozen yet)"
        )
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
    for key in ("criteria", "preregistration_text", "adjudication_protocol"):
        path = ROOT / table[key]
        actual = experiment_records.preregistered_file_digest(path)
        if table.get(f"{key}_sha256") != actual:
            raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: {table[key]} is not the file frozen ({actual})")
    criteria = tomllib.loads((ROOT / table["criteria"]).read_text(encoding="utf-8"))
    return manifest, table, criteria, digest


def metric_values(result: dict | None) -> dict | None:
    """The metrics of a harness result: its top-level integers (every metric is one; the seed is
    not a metric). Nothing under `descriptive` or `timing_ms` is among them."""
    if not isinstance(result, dict):
        return None
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, int) and not isinstance(value, bool) and key != "seed"
    }


def aggregate(report_path: Path | None = None) -> None:
    """Validate the declared seed records, judge them and publish metrics.json and run.json.
    Missing records or failed preregistration or provenance checks raise SystemExit; a
    verdict of any kind is published."""
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
        raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: {error}")

    judgements, records, metrics_seeds = [], [], []
    for seed in manifest["seeds"]:
        record = chosen[seed]
        result = None
        try:
            result = experiment_records.harness_results({per_seed[seed].name: record}, BENCHMARK)[per_seed[seed].name]
            error = None
        except experiment_records.ProvenanceError as problem:
            error = str(problem)
        judgement = judge_seed(result, record.get("exit_code"), criteria, seed)
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
                "metrics": metric_values(result),
                "summary": result.get("summary") if result else None,
            }
        )
    static = static_evidence(criteria.get("static_evidence", []))
    verdict = decide(judgements, static, manifest["seeds"], mutations_ok(mutations, plan_size()))
    text = report(verdict, judgements, static, mutations)
    metrics = {
        "experiment_id": EXPERIMENT,
        "scope": SCOPE,
        "verdict": verdict["verdict"],
        "decision": verdict,
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
        "recommended_status": verdict["recommended_status"],
        "static_evidence": [{key: item[key] for key in ("command", "exit_code", "output_sha256")} for item in static],
        "mutation_checks": mutations,
        "limitations": LIMITATIONS,
    }
    try:
        experiment_records.publish_aggregate(RESULTS, metrics, run)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"{EXPERIMENT}: {error}")
    if report_path is not None:
        report_path.write_text(text, encoding="utf-8")
    print(f"{EXPERIMENT} verdict={verdict['verdict']} recommended_status={verdict['recommended_status']} "
          f"vacuous={verdict['vacuous_seeds']} falsified={verdict['falsified_seeds']} invalid={verdict['invalid_seeds']} "
          f"static_failed={len(verdict['static_evidence_failed'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="also write the Markdown reading of the verdict here")
    args = parser.parse_args()
    aggregate(report_path=args.report)


if __name__ == "__main__":
    main()
