from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def components():
    return {
        path.parent.name: path
        for path in sorted((ROOT / "evaluations/components").glob("*/candidates.toml"))
    }


def candidate(component: str, candidate_id: str):
    path = components().get(component)
    if path is None:
        raise ValueError(f"unknown component {component}")
    data = load(path)
    match = next(
        (item for item in data.get("candidate", []) if item.get("id") == candidate_id),
        None,
    )
    if match is None:
        raise ValueError(f"unknown candidate {candidate_id} for {component}")
    return path, data, match


def validate():
    schema = load(ROOT / "evaluations/schema.toml")
    required = set(schema["required_candidate"])
    allowed = set(schema["allowed_status"])
    errors = []
    candidates = 0
    for name, path in components().items():
        data = load(path)
        if data.get("component") != name:
            errors.append(f"{path}: component mismatch")
        seen = set()
        weights = sum(float(value) for value in data.get("criteria", {}).values())
        if abs(weights - 1.0) > 1e-6:
            errors.append(f"{path}: criteria weights sum to {weights}")
        for item in data.get("candidate", []):
            candidates += 1
            missing = required - set(item)
            if missing:
                errors.append(f"{path}:{item.get('id', '?')}: missing {sorted(missing)}")
            if item.get("id") in seen:
                errors.append(f"{path}: duplicate candidate {item.get('id')}")
            seen.add(item.get("id"))
            if item.get("status") not in allowed:
                errors.append(f"{path}:{item.get('id')}: invalid status")
            if item.get("status") == "evaluating" and not str(item.get("command", "")).strip():
                errors.append(
                    f"{path}:{item.get('id')}: evaluating candidate needs command"
                )
        if not (path.parent / "config.toml").exists():
            errors.append(f"{path.parent}: missing config.toml")
        if not (path.parent / "tests").exists():
            errors.append(f"{path.parent}: missing tests/")
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    print(f"OK: validated {candidates} component candidates")
    return 0


def base_record(component: str, match: dict, data: dict, path: Path) -> dict:
    return {
        "schema_version": 1,
        "component": component,
        "candidate": match,
        "criteria": data.get("criteria", {}),
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "python": sys.version,
        "candidate_manifest_sha256": sha(path),
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
    }


def prepare(component: str, candidate_id: str) -> int:
    try:
        path, data, match = candidate(component, candidate_id)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    stamp = utc_stamp()
    record = base_record(component, match, data, path)
    record.update(
        {
            "prepared_at": stamp,
            "status": "prepared-no-benchmark-evidence",
        }
    )
    out = path.parent / "evidence" / f"{stamp}-{candidate_id}.json"
    write_json_exclusive(out, record)
    print(out.relative_to(ROOT))
    return 0


def build_command(match: dict) -> list[str]:
    command = str(match.get("command", "")).strip()
    if not command:
        raise ValueError(f"candidate {match.get('id', '?')} has no declared command")
    return shlex.split(command)


def run_candidate(component: str, candidate_id: str) -> int:
    try:
        path, data, match = candidate(component, candidate_id)
        command = build_command(match)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    stamp = utc_stamp()
    record = base_record(component, match, data, path)
    record.update(
        {
            "status": "running",
            "started_at": stamp,
            "command": command,
        }
    )

    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        launch_error = None
    except OSError as error:
        exit_code = None
        stdout = ""
        stderr = ""
        launch_error = f"{type(error).__name__}: {error}"
    finished = time.perf_counter_ns()

    record.update(
        {
            "status": (
                "completed"
                if exit_code == 0
                else "failed" if exit_code is not None else "failed-to-launch"
            ),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "duration_ns": finished - started,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "launch_error": launch_error,
        }
    )
    out = path.parent / "evidence" / f"{stamp}-{candidate_id}-run.json"
    write_json_exclusive(out, record)
    print(out.relative_to(ROOT))
    return exit_code if exit_code is not None else 127


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("list")

    prep = sub.add_parser("prepare")
    prep.add_argument("component")
    prep.add_argument("candidate")

    run = sub.add_parser("run")
    run.add_argument("component")
    run.add_argument("candidate")

    args = parser.parse_args()
    if args.command == "validate":
        raise SystemExit(validate())
    if args.command == "list":
        for name, path in components().items():
            data = load(path)
            for item in data.get("candidate", []):
                print(name, item["id"], item["status"])
        return
    if args.command == "prepare":
        raise SystemExit(prepare(args.component, args.candidate))
    if args.command == "run":
        raise SystemExit(run_candidate(args.component, args.candidate))


if __name__ == "__main__":
    main()
