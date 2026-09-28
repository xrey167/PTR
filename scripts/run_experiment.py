from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.toml"
PLACEHOLDER = re.compile(r"<([A-Za-z][A-Za-z0-9_-]*)>")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_research_gates  # noqa: E402
import experiment_records  # noqa: E402


def load(path: Path):
    return tomllib.loads(path.read_text(encoding="utf-8"))


def registry():
    return {e["id"]: e for e in load(REGISTRY).get("experiment", [])}


def resolve(exp_id: str):
    item = registry().get(exp_id)
    if not item:
        raise SystemExit(f"unknown experiment: {exp_id}")
    root = ROOT / "experiments" / item["path"]
    return item, root, load(root / "experiment.toml")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def write_json_exclusive(path: Path, record: dict) -> None:
    """Write `record` to the new file `path`, whole or not at all: a write
    that fails leaves no partial record for an aggregator to select. Raises
    `FileExistsError` when `path` exists; a record never replaces another."""
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment_records.write_exclusively(path, json.dumps(record, indent=2) + "\n")


def validate():
    schema = load(ROOT / "experiments/schema.toml")
    required = set(schema["required"])
    allowed = set(schema["allowed_status"])
    errors = []
    for exp_id, item in registry().items():
        root = ROOT / "experiments" / item["path"]
        path = root / "experiment.toml"
        if not path.exists():
            errors.append(f"{exp_id}: missing experiment.toml")
            continue
        data = load(path)
        missing = required - set(data)
        if missing:
            errors.append(f"{exp_id}: missing {sorted(missing)}")
        if data.get("id") != exp_id:
            errors.append(f"{exp_id}: id mismatch")
        if data.get("status") not in allowed:
            errors.append(f"{exp_id}: invalid status")
        if item.get("status") != data.get("status"):
            errors.append(f"{exp_id}: registry/manifest status mismatch")
        if data.get("status") in {"running", "completed", "failed"} and not str(
            data.get("entrypoint", "")
        ).strip():
            errors.append(f"{exp_id}: active experiment needs executable entrypoint")
        if not (root / "config.toml").exists():
            errors.append(f"{exp_id}: missing config.toml")
        if not (root / "tests").exists():
            errors.append(f"{exp_id}: missing tests/")
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    print(f"OK: validated {len(registry())} experiments")
    return 0


def base_record(exp_id: str, data: dict, root: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": exp_id,
        "git_sha": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "manifest": data,
        "manifest_sha256": sha(root / "experiment.toml"),
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
        "uv_lock_sha256": sha(ROOT / "training/uv.lock"),
        "hardware_profile": data.get("hardware_profile"),
    }


def launch_refused(exp_id: str) -> bool:
    """Print why `exp_id` may not be run or prepared now and return True, or
    return False when it may: an experiment `experiments/preregistration.toml`
    lists runs only once its preregistration is frozen
    (`check_research_gates.launch_errors`)."""
    problems = check_research_gates.launch_errors(ROOT, exp_id)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    return bool(problems)


def prepare(exp_id: str):
    _, root, data = resolve(exp_id)
    if launch_refused(exp_id):
        return 2
    results = root / data.get("results_dir", "results")
    timestamp = utc_stamp()
    record = base_record(exp_id, data, root)
    record.update({"status": "prepared", "prepared_at": timestamp})
    out = results / f"run-{timestamp}.json"
    write_json_exclusive(out, record)
    print(out.relative_to(ROOT))
    return 0


def parse_params(items: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"parameter must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if not PLACEHOLDER.fullmatch(f"<{key}>"):
            raise ValueError(f"invalid parameter name: {key}")
        if key in params:
            raise ValueError(f"duplicate parameter: {key}")
        params[key] = value
    return params


def build_command(
    data: dict,
    *,
    entrypoint: str,
    seed: int,
    params: dict[str, str] | None = None,
) -> list[str]:
    seeds = data.get("seeds", [])
    if seed not in seeds:
        raise ValueError(f"seed {seed} is not declared in experiment seeds {seeds}")

    template = str(data.get(entrypoint, "")).strip()
    if not template:
        raise ValueError(f"experiment has no executable {entrypoint!r}")

    values = {"seed": str(seed)}
    values.update(params or {})
    command = []
    for token in shlex.split(template):
        unresolved = PLACEHOLDER.findall(token)
        for name in unresolved:
            if name not in values:
                raise ValueError(f"missing value for <{name}>")
            token = token.replace(f"<{name}>", values[name])
        command.append(token)

    leftovers = [name for token in command for name in PLACEHOLDER.findall(token)]
    if leftovers:
        raise ValueError(f"unresolved placeholders: {sorted(set(leftovers))}")
    return command


def execute_command(command: list[str]) -> dict:
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        result = {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "launch_error": None,
        }
    except OSError as error:
        result = {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "launch_error": f"{type(error).__name__}: {error}",
        }
    result["duration_ns"] = time.perf_counter_ns() - started
    return result


def run_experiment(
    exp_id: str,
    *,
    entrypoint: str,
    seed: int,
    params: dict[str, str] | None = None,
) -> int:
    """Run one seed of `exp_id` through `entrypoint` and write its record to
    the experiment's results, stamped with the commit HEAD was at when the
    run started. Refuses with status 2, before anything runs or is written,
    an undeclared seed, an unresolved command, and a working tree with
    uncommitted or untracked provenance files or experiment files
    (`experiment_records.uncommitted_files`, which also covers the files that
    decide whether it may launch, `check_research_gates.launch_inputs`), and
    a listed experiment whose preregistration is not frozen
    (`launch_refused`). After the command ends it looks
    at the same tree again (`experiment_records.ProvenanceWatch`) and writes
    no record, returning 2, when HEAD moved or a provenance or experiment
    file was written, created or removed while the command ran, even if its
    content was put back, so the record's `git_sha` is the code that ran as
    far as that watch can see (its `changes` names what it cannot). The
    record is written whole or not at all (`write_json_exclusive`)."""
    _, root, data = resolve(exp_id)
    if launch_refused(exp_id):
        return 2
    try:
        command = build_command(
            data, entrypoint=entrypoint, seed=seed, params=params or {}
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    results = root / data.get("results_dir", "results")
    # The record names HEAD as the code it ran, so HEAD must hold every file
    # that decides the run: refuse before anything runs or is written.
    try:
        watch = experiment_records.ProvenanceWatch(
            ROOT,
            [
                *experiment_records.tree_pathspecs(
                    root, results, ROOT, experiment_records.seed_record_paths(root, ROOT)
                ),
                # What decided that this experiment may launch: an edit to it
                # would be undone after the outcome is seen.
                *check_research_gates.launch_inputs(ROOT, exp_id),
            ],
        )
    except experiment_records.ProvenanceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if watch.uncommitted:
        print(
            "ERROR: refusing to run from a working tree whose sources HEAD does not hold; "
            f"commit or remove {experiment_records.listed(watch.uncommitted)}",
            file=sys.stderr,
        )
        return 2

    timestamp = utc_stamp()
    record = base_record(exp_id, data, root)
    record.update(
        {
            "git_sha": watch.head,
            "status": "running",
            "started_at": timestamp,
            "entrypoint": entrypoint,
            "seed": seed,
            "parameters": params or {},
            "command": command,
        }
    )

    execution = execute_command(command)
    exit_code = execution["exit_code"]
    # The command read the tree while it ran (a `cargo run` entrypoint
    # compiles it first): the record may name HEAD only if the tree stayed so.
    try:
        changes = watch.changes()
    except experiment_records.ProvenanceError as error:
        changes = [str(error)]
    if changes:
        print(
            f"ERROR: not recording the run (exit status {exit_code}): its sources changed while it ran, "
            f"so {watch.head} may not be the code it ran; {'; '.join(changes)}; "
            "rerun from a working tree that stays at HEAD",
            file=sys.stderr,
        )
        return 2
    record.update(
        {
            "status": (
                "completed"
                if exit_code == 0
                else "failed" if exit_code is not None else "failed-to-launch"
            ),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **execution,
        }
    )

    out = results / f"run-{timestamp}-seed-{seed}.json"
    write_json_exclusive(out, record)
    print(out.relative_to(ROOT))
    return exit_code if exit_code is not None else 127


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("validate")

    show = sub.add_parser("show")
    show.add_argument("id")

    prep = sub.add_parser("prepare")
    prep.add_argument("id")

    run = sub.add_parser("run")
    run.add_argument("id")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--entrypoint", default="entrypoint")
    run.add_argument("--set", dest="params", action="append", default=[])

    args = parser.parse_args()
    if args.cmd == "list":
        for key, value in registry().items():
            print(key, value["status"], value["path"])
        return
    if args.cmd == "validate":
        raise SystemExit(validate())
    if args.cmd == "show":
        _, _, data = resolve(args.id)
        print(json.dumps(data, indent=2))
        return
    if args.cmd == "prepare":
        raise SystemExit(prepare(args.id))
    if args.cmd == "run":
        try:
            params = parse_params(args.params)
        except ValueError as error:
            parser.error(str(error))
        raise SystemExit(
            run_experiment(
                args.id,
                entrypoint=args.entrypoint,
                seed=args.seed,
                params=params,
            )
        )


if __name__ == "__main__":
    main()
