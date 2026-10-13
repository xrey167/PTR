"""Aggregate the L002 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run L002 --seed <s>` writes
(results/run-<timestamp>-seed-<s>.json): the newest record of every declared
seed whose command ran to an exit code. Nothing of the experiment is run again.
Every declared seed must have a record, the records must be of one
configuration and have run the checkout's code, this script included
(`experiment_records.source_revision`), and the preregistration digests of the
manifest, the table, the preregistration text and `criteria.toml` must be the
frozen ones.

The decision is mechanical, and it is `criteria.toml`'s (its header defines the
rule grammar, which `evaluate` below implements):

- a seed is judged by applying every coverage rule and every falsifier rule to
  the one JSON object its harness printed;
- the order is validity, then the falsifiers, then the controls;
- INVALID if anything about the run is not sound, even when a falsifier fired in
  a record that can be read: a seed that cannot be judged (no result line, a
  harness that exited 2 or by a signal, another benchmark, seed, scale or
  experiment than the record's, harness gaps, a control that did not trip, a rule
  whose counter is missing, null or of the wrong type, or an exit status that
  disagrees with the rules: 0 with a failed rule, 1 with none), a missing or
  duplicated seed, a mutation record that is absent or short, or a failing static
  command. INVALID says nothing about the hypothesis;
- FALSIFIED if every seed can be judged and some seed fails a rule (a coverage
  rule that fails makes the run vacuous, which counts as falsified);
- COMPLETED iff every seed holds every coverage rule and every falsifier rule.

The harness is shown able to fail by `results/mutations.json` (scripts/mutation_check.py L002), which must be this
experiment's evidence from the same code, checker and mutation plan with every mutation killed; a missing or partly
surviving record makes the verdict INVALID (the harness was not shown to see the defects it claims to see), never
COMPLETED. criteria.toml names no static evidence, so otherwise the verdict rests on the seeds alone.

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
BENCHMARK = "raft-recovery"
EXPERIMENT = "L002"
TAIL = 2000

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

COMPLETED, FALSIFIED, INVALID = "COMPLETED", "FALSIFIED", "INVALID"
OPERATORS = (">=", "<=", "==")
NUMBER = re.compile(r"-?[0-9]+")
SCOPE = (
    "recovery of a clustered ledger in-process on one host, no network: the compaction cutover of AcknowledgedLedger with its "
    "ProtectedAnchor, crashed at each of its nine failpoints (in-process panic, fsync-honest power-loss images of the new log "
    "and of the anchor's publication) and appends above the new floor crashed at the eight log and anchor failpoints, with the "
    "runtime restored from the compacted snapshot and the retained records compared with an independent oracle (arm K); three "
    "in-process Raft members over FileRaftStorage on a deterministic router crashed at twelve storage failpoints and failed at two "
    "writes during isolation, snapshot recording and installation, a deposed leader and message faults, held to persist-before-send, "
    "no forgetting and convergence to a single-node reference within a pinned number of heartbeat rounds (arms S and G); "
    "power-loss images of a member's log; a forged snapshot that must be refused by its anchor"
)
LIMITATIONS = [
    "power loss is a fsync-honest crash-image model: bytes synced before the crash are intact and nothing else is promised; it is not a real power cut, and images exist for the new log and the anchor of the cutover and for a member's log, not for a member's state or snapshot file",
    "the process-abort class is an in-process panic at a failpoint (the unwinding drops the handles; no destructor of the code under test writes); child processes that call abort() are not repeated from L001-v2",
    "the durability probe sees file data pages only; a missing directory fsync is outside what it detects, and the model takes the directory fsync on trust",
    "the Raft members return all their messages from one call, so the order of persisting and sending inside RaftNode::drain is not observable; the checked contract is that nothing a call returns promises a vote or an append that is not in the member's files and that a failed write returns no message",
    "three in-process members on a deterministic router: no sockets, no election timer, no membership change, no real wire (ptr-cluster, ptr-net); recovery time is a count of heartbeat rounds, not a duration",
    "RaftEngineLedger is not touched (L001-v2's stated limit and owner decision stand)",
    "synthetic seeded histories of 10 to 16 events; one process for the in-process arms, one machine profile",
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
        invalid.append("did not run the pinned plan")
    controls = result.get("controls")
    for name, field in (("cutover_lie", "lie_images"), ("forged_snapshot", "refused_by_anchor")):
        control = controls.get(name) if isinstance(controls, dict) else None
        # Only the sensitivity of the control is a condition of validity: a lie that does not fool
        # the naive recovery (or a forged snapshot whose unanchored take is harmless) shows the run
        # could not tell a recovery that notices lost records from one that does not. That the
        # anchored product refused every lie is a falsifier (anchored_lie_accepted,
        # tampered_snapshot_accepted), counted whatever else is wrong with the record.
        if (
            not isinstance(control, dict)
            or control.get("required_to_trip") is not True
            or not isinstance(control.get("naive_accepted_stale"), int)
            or isinstance(control.get("naive_accepted_stale"), bool)
            or control["naive_accepted_stale"] < 1
            or not isinstance(control.get(field), int)
            or isinstance(control.get(field), bool)
            or control[field] < 1
        ):
            invalid.append(f"the control {name} did not trip")
    if procedure is not None and isinstance(procedure, dict):
        if procedure.get("durability_probe_available") is not True:
            invalid.append("the durability probe was not available on the scratch directory")
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


def decide(seeds: list[dict], static: list[dict] | None, declared: list[int], mutations: dict | None = None) -> dict:
    """The verdict from the seeds' judgements (`judge_seed`, in any order), the
    static-evidence results (None when they were not run) and the mutation evidence
    (`killed` and `total`; None when there is none). Pure."""
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
    unmet = [entry["seed"] for entry in seeds if entry["valid"] and entry["coverage_failed"]]
    # Reported whatever the verdict: the seeds whose own record trips a falsifier.
    tripped = [entry["seed"] for entry in seeds if entry["falsifiers_failed"]]
    unjudged = [entry["seed"] for entry in seeds if not entry["valid"]]
    static_failed = [item["command"] for item in static or [] if item.get("exit_code") != 0]
    # The order is validity, then the falsifiers, then the controls (the repository rule): a
    # verdict about the product is only given on a run that is itself sound. A seed that cannot
    # be judged, a missing seed, a mutation record that is absent or short, a static-evidence
    # command that fails or cannot start: any of these is INVALID even when a falsifier fired in
    # a record that can still be read (the fired falsifier stays listed in `falsified_seeds`).
    # The controls of a seed (a lie that must fool the naive recovery) are validity conditions
    # of that seed and are judged inside `judge_seed`, before its falsifiers.
    if problems or unjudged or static_failed:
        verdict = INVALID
    elif unmet or tripped:
        verdict = FALSIFIED
    else:
        verdict = COMPLETED
    return {
        "verdict": verdict,
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
    lines = [f"# L002 recovery of a clustered ledger: {verdict['verdict']}", ""]
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


def load_frozen() -> tuple[dict, dict, dict, str]:
    """The manifest, the preregistered table, the criteria and the table's digest, once the
    manifest, the table, the preregistration text and the criteria agree. Raises SystemExit."""
    manifest = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
    if "preregistration" not in CONFIG:
        raise SystemExit(
            f"{EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table (L002 is not frozen yet)"
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
    for key in ("criteria", "preregistration_text"):
        path = ROOT / table[key]
        actual = experiment_records.preregistered_file_digest(path)
        if table.get(f"{key}_sha256") != actual:
            raise SystemExit(f"{EXPERIMENT}: refusing to aggregate: {table[key]} is not the file frozen ({actual})")
    criteria = tomllib.loads((ROOT / table["criteria"]).read_text(encoding="utf-8"))
    return manifest, table, criteria, digest


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
    metric_names = manifest["metrics"].split(", ")
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
                "metrics": {name: result.get(name) for name in metric_names} if result else None,
                "summary": result.get("summary") if result else None,
            }
        )
    static = static_evidence(criteria.get("static_evidence", []))
    verdict = decide(judgements, static, manifest["seeds"], mutations)
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
