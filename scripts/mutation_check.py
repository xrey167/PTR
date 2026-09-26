"""Mutation checks for experiment harnesses: can a broken implementation pass?

An experiment that has only ever passed says little until it has been shown to
fail. Each experiment that opts in lists defects in `tests/mutations.toml`: a
source file, an exact text that must occur there once, and its replacement, plus
optional further edits (`[[mutation.also]]`, each with its own `find`,
`replace` and optionally `file`) when one defect spans several places.
Each mutation also names the hard counters that show its defect (`expect`).
This script plants one defect at a time, rebuilds the harness, runs it, and
restores the file whatever happens. A mutation is *killed* when the harness
exits with status 1 and one of its expected counters fired. A run that fails
only on other counters is recorded as `failed-elsewhere`: the defect may have
broken something unrelated first (a query that no longer binds, say), which
shows nothing about whether the harness sees the defect itself. Exiting any
other way (a panic, a build failure, a timeout) or passing is recorded as it
is. The record goes to the experiment's `results/mutations.json`.

    python scripts/mutation_check.py L004
    python scripts/mutation_check.py L003 --only append-without-row-lock
    python scripts/mutation_check.py L004 --check   # anchors only, no build

The harness reads its server from the environment the experiment names (for
the PostgreSQL experiments `PTR_PG_EXPERIMENT_DSN`); nothing here stores it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.toml"


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def experiment_root(exp_id: str) -> Path:
    for item in load_toml(REGISTRY).get("experiment", []):
        if item["id"] == exp_id:
            return ROOT / "experiments" / item["path"]
    raise SystemExit(f"unknown experiment: {exp_id}")


def load_plan(exp_root: Path) -> dict:
    path = exp_root / "tests/mutations.toml"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} does not exist")
    plan = load_toml(path)
    for key in ("package", "features", "subcommand", "cases", "seed"):
        if key not in plan:
            raise SystemExit(f"{path.relative_to(ROOT)}: missing {key!r}")
    names = [mutation["name"] for mutation in plan.get("mutation", [])]
    if len(names) != len(set(names)):
        raise SystemExit(f"{path.relative_to(ROOT)}: duplicate mutation names")
    errors = expectation_errors(plan)
    if errors:
        raise SystemExit(f"{path.relative_to(ROOT)}: " + "; ".join(errors))
    return plan


def expectation_errors(plan: dict) -> list[str]:
    """Every mutation must name at least one expected counter, and each must
    be one of the plan's hard counters."""
    hard = set(plan.get("hard_counters", []))
    errors = []
    for mutation in plan.get("mutation", []):
        expected = mutation.get("expect", [])
        if not expected:
            errors.append(f"{mutation['name']}: no expected counter")
        for counter in expected:
            if counter not in hard:
                errors.append(f"{mutation['name']}: {counter!r} is not a hard counter")
    return errors


def edits(mutation: dict) -> list[tuple[str, str, str]]:
    """The (file, find, replace) edits of one mutation, in order."""
    first = [(mutation["file"], mutation["find"], mutation["replace"])]
    return first + [
        (extra.get("file", mutation["file"]), extra["find"], extra["replace"])
        for extra in mutation.get("also", [])
    ]


def anchor_errors(plan: dict, root: Path = ROOT) -> list[str]:
    """Every edit's text must occur exactly once in its file, and its
    replacement must differ; otherwise the list has drifted from the code."""
    errors = []
    for mutation in plan.get("mutation", []):
        name = mutation["name"]
        for file, find, replace in edits(mutation):
            path = root / file
            if not path.exists():
                errors.append(f"{name}: {file} does not exist")
                continue
            count = path.read_text(encoding="utf-8").count(find)
            if count != 1:
                errors.append(f"{name}: the anchor occurs {count} times in {file}")
            if find == replace:
                errors.append(f"{name}: the replacement equals the anchor")
    return errors


def binary_command(plan: dict, cases: int, seed: int) -> tuple[list[str], list[str]]:
    toolchain = [f"+{plan['toolchain']}"] if plan.get("toolchain") else []
    build = [
        "cargo",
        *toolchain,
        "build",
        "--release",
        "--locked",
        "-p",
        plan["package"],
        "--features",
        plan["features"],
    ]
    run = [
        str(ROOT / "target/release" / plan["package"]),
        plan["subcommand"],
        str(cases),
        str(seed),
    ]
    return build, run


def last_json_line(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {}


def classify(returncode: int, metrics: dict, plan: dict, mutation: dict) -> tuple[str, dict]:
    """The result of one run of a mutated harness and the hard counters that
    fired."""
    fired = {
        key: value
        for key, value in metrics.items()
        if key in plan.get("hard_counters", []) and isinstance(value, int) and value > 0
    }
    hard = int(metrics.get("hard_failures", 0))
    if returncode == 0:
        return "survived", fired
    if returncode == 1 and hard > 0:
        if any(counter in fired for counter in mutation.get("expect", [])):
            return "killed", fired
        return "failed-elsewhere", fired
    return "crashed", fired


def run_mutation(plan: dict, mutation: dict, timeout: int) -> dict:
    cases = int(mutation.get("cases", plan["cases"]))
    seed = int(mutation.get("seed", plan["seed"]))
    build, run = binary_command(plan, cases, seed)
    planned = edits(mutation)
    originals = {file: (ROOT / file).read_text(encoding="utf-8") for file, _, _ in planned}
    outcome: dict = {"name": mutation["name"], "file": mutation["file"], "cases": cases, "seed": seed}
    try:
        mutated = dict(originals)
        for file, find, replace in planned:
            mutated[file] = mutated[file].replace(find, replace, 1)
        for file, text in mutated.items():
            (ROOT / file).write_text(text, encoding="utf-8")
        built = subprocess.run(build, cwd=ROOT, capture_output=True, text=True, check=False)
        if built.returncode != 0:
            outcome.update(result="build-failed", detail=built.stderr[-2000:])
            return outcome
        try:
            ran = subprocess.run(
                run, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            outcome.update(result="timeout")
            return outcome
    finally:
        for file, text in originals.items():
            (ROOT / file).write_text(text, encoding="utf-8")
    metrics = last_json_line(ran.stdout)
    result, fired = classify(ran.returncode, metrics, plan, mutation)
    outcome.update(
        result=result,
        exit_code=ran.returncode,
        hard_failures=int(metrics.get("hard_failures", 0)),
        expected=list(mutation.get("expect", [])),
        counters=fired,
    )
    if result != "killed":
        outcome["detail"] = ran.stderr[-2000:]
    return outcome


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--check", action="store_true", help="verify anchors only")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    exp_root = experiment_root(args.id)
    plan = load_plan(exp_root)
    errors = anchor_errors(plan)
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    if args.check:
        print(f"OK: {len(plan.get('mutation', []))} mutation anchors for {args.id}")
        return 0

    selected = [m for m in plan["mutation"] if not args.only or m["name"] in args.only]
    outcomes = []
    for mutation in selected:
        outcome = run_mutation(plan, mutation, args.timeout)
        outcomes.append(outcome)
        print(f"{outcome['name']}: {outcome['result']} {outcome.get('counters', {})}", flush=True)

    # Rebuild the unmutated harness so no mutated binary is left behind.
    build, _ = binary_command(plan, 1, int(plan["seed"]))
    subprocess.run(build, cwd=ROOT, capture_output=True, text=True, check=False)

    if not args.only:
        record = {
            "experiment_id": args.id,
            "git_sha": git_sha(),
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "subcommand": plan["subcommand"],
            "killed": sum(outcome["result"] == "killed" for outcome in outcomes),
            "total": len(outcomes),
            "mutations": outcomes,
        }
        out = exp_root / "results/mutations.json"
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out.relative_to(ROOT))
    return 0 if all(outcome["result"] == "killed" for outcome in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
