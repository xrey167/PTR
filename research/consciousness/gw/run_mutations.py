"""Plant the defects of experiments/consciousness/C001-global-workspace/tests/mutations.toml, one at a time,
in a throwaway copy of the harness.

    python research/consciousness/gw/run_mutations.py            # all, writes results/mutations.json
    python research/consciousness/gw/run_mutations.py --only topk-write-not-applied

scripts/mutation_check.py is the repository's checker, but it builds a Cargo package and runs
target/release/<package>; it cannot plant a defect in a Python entrypoint. This runner follows the
same contract (one defect at a time, restore whatever happens, killed = exit 1 with an expected
counter) and writes the file layout experiment_records.mutation_evidence reads. `--check` of
scripts/mutation_check.py still verifies the anchors.

The harness runs at the reduced scale of the plan's `runner_args`. The unmutated harness must pass at
that scale first; otherwise nothing is planted and no record is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENT = ROOT / "experiments/consciousness/C001-global-workspace"
PLAN = EXPERIMENT / "tests/mutations.toml"
HARNESS = HERE / "harness.py"
RESULT = EXPERIMENT / "results/mutations.json"


def edits(mutation: dict) -> list[tuple[str, str, str]]:
    first = [(mutation["file"], mutation["find"], mutation["replace"])]
    return first + [(x.get("file", mutation["file"]), x["find"], x["replace"]) for x in mutation.get("also", [])]


def run_harness(plan: dict, timeout: int, tree: Path) -> tuple[int | str, dict, str]:
    harness = tree / "research/consciousness/gw/harness.py"
    command = [sys.executable, str(harness), "--benchmark", "c001", "--seed", str(plan["seed"]), *plan["runner_args"]]
    try:
        done = subprocess.run(command, cwd=tree, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return "timeout", {}, ""
    result: dict = {}
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                result = {}
            break
    return done.returncode, result, done.stderr[-1500:]


def classify(code, result: dict, stderr: str, expect: list[str]) -> tuple[str, list[str]]:
    fired = [r.split(" ")[0] for r in result.get("falsifiers_failed", [])]
    unmet = [r.split(" ")[0] for r in result.get("coverage_failed", [])]
    seen = fired + unmet
    if code == 0:
        return "survived", seen
    if code == 1:
        return ("killed" if any(name in seen for name in expect) else "failed-elsewhere"), seen
    if code == 2:
        controls = [name for name in expect if name.startswith("control:") and name.split(":", 1)[1] in stderr]
        return ("killed" if controls else "crashed"), seen
    return ("timeout" if code == "timeout" else "crashed"), seen


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def prepare_copy(tmp: Path) -> None:
    """A throwaway copy of exactly what the harness reads (gw/*.py and the experiment's config.toml), laid out so
    that the copy's own ROOT is `tmp`: defects are planted there, never in the working tree, so an interrupted run
    cannot leave a mutated harness behind."""
    shutil.copytree(HERE, tmp / "research/consciousness/gw", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    target = tmp / "experiments/consciousness/C001-global-workspace"
    target.mkdir(parents=True)
    shutil.copy(EXPERIMENT / "config.toml", target / "config.toml")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    plan = tomllib.loads(PLAN.read_text(encoding="utf-8"))
    names = {m["name"] for m in plan["mutation"]}
    unknown = [n for n in args.only if n not in names]
    if unknown:
        print(f"unknown mutations: {unknown}", file=sys.stderr)
        return 2
    selected = [m for m in plan["mutation"] if not args.only or m["name"] in args.only]
    if not args.only:
        dirty = git("status", "--porcelain", "--", "research/consciousness", "experiments/consciousness/C001-global-workspace").strip()
        if dirty:
            print("refusing to write evidence from a tree with uncommitted provenance files:\n" + dirty, file=sys.stderr)
            return 2
    head = git("rev-parse", "HEAD").strip()
    outcomes = []
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch) / "baseline"
        prepare_copy(base)
        code, result, err = run_harness(plan, args.timeout, base)
        if code != 0:
            print(f"the unmutated harness does not pass at the mutation scale (exit {code}): {err}", file=sys.stderr)
            return 2
        for mutation in selected:
            tree = Path(scratch) / mutation["name"]
            prepare_copy(tree)
            for f, find, replace in edits(mutation):
                path = tree / f
                text = path.read_text(encoding="utf-8")
                if text.count(find) != 1:
                    print(f"{mutation['name']}: anchor not found exactly once in {f}", file=sys.stderr)
                    return 2
                path.write_text(text.replace(find, replace, 1), encoding="utf-8")
            code, result, err = run_harness(plan, args.timeout, tree)
            verdict, seen = classify(code, result, err, mutation["expect"])
            outcomes.append({"name": mutation["name"], "file": mutation["file"], "result": verdict, "exit": code, "counters": seen, "expected": mutation["expect"]})
            print(f"{mutation['name']}: {verdict} (exit {code}; {seen})", flush=True)
            shutil.rmtree(tree, ignore_errors=True)
    killed = sum(o["result"] == "killed" for o in outcomes)
    print(f"{killed} of {len(outcomes)} killed")
    if not args.only:
        import platform
        torch_version = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"], capture_output=True, text=True).stdout.strip()
        record = {
            "experiment_id": "C001", "subcommand": "c001", "git_sha": head, "killed": killed, "total": len(outcomes),
            "runner_args": plan["runner_args"], "interpreter": sys.version.split()[0], "torch": torch_version, "machine": platform.machine(),
            "note": "planted in a throwaway copy at a reduced scale with the interpreter that ran this script (not the uv-locked environment)",
            "mutations": outcomes,
        }
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        tmp = RESULT.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, RESULT)
    return 0 if killed == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
