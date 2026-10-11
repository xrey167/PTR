"""Plant defects in an experiment's AGGREGATOR (and the shared aggregation library), one at a time, in a throwaway copy, and require the
aggregator's unit tests to fail on each.

    python research/cseries/lib/run_agg_mutations.py C007 --check          # anchors only: every `find` occurs exactly once
    python research/cseries/lib/run_agg_mutations.py C007                  # all, prints the verdicts
    python research/cseries/lib/run_agg_mutations.py C007 --only a-name    # one (repeatable)
    python research/cseries/lib/run_agg_mutations.py C007 --write          # all, and writes tests/aggregator-mutations.json

The plan is `<experiment>/tests/aggregator_mutations.toml` plus the shared library's `lib_aggregator_mutations.toml` ([[mutation]] with name, file, find, replace). The harness's counters are planted by
`run_mutations.py`; the rules that live only in the aggregator (guards, Bonferroni quantile, sensitivity and removal controls, the verdict order,
the allowance counts) are planted here. A mutation is killed iff `scripts.tests.test_<id>_aggregate` (run inside the planted copy, whose own ROOT it is) or the library tests fail while both pass unmutated. The interpreter that runs this script runs the tests."""

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
EXPERIMENTS = {
    "C004": "experiments/consciousness/C004-recurrent-integration",
    "C007": "experiments/consciousness/C007-metacognitive-self-model-cpu",
    "C008": "experiments/consciousness/C008-attention-schema-routing-cpu",
}


def prepare(tmp: Path, rel_dir: str) -> None:
    shutil.copytree(ROOT / "research/cseries", tmp / "research/cseries", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "pilots"))
    for other in EXPERIMENTS.values():  # the library tests read every stage-A experiment's files (the freeze pinning)
        target = tmp / other
        target.mkdir(parents=True, exist_ok=True)
        for name in ("aggregate.py", "config.toml", "criteria.toml", "experiment.toml", "PREREGISTRATION.md"):
            if (ROOT / other / name).is_file():
                shutil.copy(ROOT / other / name, target / name)
    (tmp / rel_dir / "tests").mkdir(exist_ok=True)
    for plan in ("mutations.toml", "aggregator_mutations.toml"):
        shutil.copy(ROOT / rel_dir / "tests" / plan, tmp / rel_dir / "tests" / plan)
    shutil.copytree(ROOT / "scripts", tmp / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("preregistration.toml", "registry.toml"):
        shutil.copy(ROOT / "experiments" / name, tmp / "experiments" / name)


def run_tests(experiment: str, tree: Path) -> tuple[int, str]:
    """Run, in the planted tree, the aggregator's tests, then the library tests and the experiment's rule-agreement tests (which pin criteria.toml and
    config.toml to each other and to the harness); the first failing suite ends the run."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-m", "unittest", f"scripts.tests.test_{experiment.lower()}_aggregate"], cwd=tree, capture_output=True, text=True, env=env, check=False)
    tail = (out.stdout + out.stderr)[-400:]
    if out.returncode != 0:
        return out.returncode, tail
    lib = subprocess.run([sys.executable, "-m", "unittest", "test_lib", f"test_{experiment.lower()}.RulesAgreeWithCriteria"], cwd=tree / "research/cseries/tests",
                         capture_output=True, text=True, env=env, check=False)
    return lib.returncode, tail + (lib.stdout + lib.stderr)[-300:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-lib", action="store_true", help="skip the shared library's plan (lib_aggregator_mutations.toml)")
    args = parser.parse_args()
    rel_dir = EXPERIMENTS[args.experiment]
    plan = tomllib.loads((ROOT / rel_dir / "tests/aggregator_mutations.toml").read_text(encoding="utf-8"))["mutation"]
    if not args.no_lib:
        plan = plan + tomllib.loads((HERE / "lib_aggregator_mutations.toml").read_text(encoding="utf-8"))["mutation"]
    problems = []
    names = [m["name"] for m in plan]
    if len(set(names)) != len(names):
        problems.append("duplicate mutation names")
    for m in plan:
        text = (ROOT / m["file"]).read_text(encoding="utf-8")
        if text.count(m["find"]) != 1:
            problems.append(f"{m['name']}: the anchor occurs {text.count(m['find'])} times in {m['file']}")
        if m["find"] == m["replace"]:
            problems.append(f"{m['name']}: replacement equals the anchor")
    if problems:
        print("\n".join("ERROR: " + p for p in problems))
        return 1
    if args.check:
        print(f"OK: {len(plan)} aggregator mutation anchors for {args.experiment}")
        return 0
    selected = [m for m in plan if not args.only or m["name"] in args.only]
    outcomes = []
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch) / "base"
        prepare(base, rel_dir)
        code, tail = run_tests(args.experiment, base)
        if code != 0:
            print(f"the aggregator tests do not pass unmutated: {tail}", file=sys.stderr)
            return 2
        for m in selected:
            tree = Path(scratch) / m["name"]
            prepare(tree, rel_dir)
            path = tree / m["file"]
            path.write_text(path.read_text(encoding="utf-8").replace(m["find"], m["replace"], 1), encoding="utf-8")
            code, _ = run_tests(args.experiment, tree)
            outcomes.append({"name": m["name"], "result": "killed" if code != 0 else "survived"})
            print(f"{m['name']}: {outcomes[-1]['result']}", flush=True)
            shutil.rmtree(tree, ignore_errors=True)
    killed = sum(o["result"] == "killed" for o in outcomes)
    print(f"{killed} of {len(outcomes)} killed")
    if args.write and not args.only:
        record = {"experiment_id": args.experiment, "killed": killed, "total": len(outcomes), "mutations": outcomes}
        (ROOT / rel_dir / "tests/aggregator-mutations.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0 if killed == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
