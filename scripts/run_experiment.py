from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import statistics
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.toml"
PLACEHOLDER = re.compile(r"<([A-Za-z][A-Za-z0-9_-]*)>")


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


def git_worktree_state() -> dict:
    """The same rule as training/src/ptr_training/run.py: only tracked files count,
    so the untracked run records this runner itself writes never make a run dirty."""
    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        )
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        return {
            "git_dirty": bool(porcelain.strip()),
            "git_tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {"git_dirty": None, "git_tracked_diff_sha256": None}


def host_facts() -> dict:
    """The machine a run executed on, measured rather than declared.

    A manifest names a hardware profile, but `hardware/default.toml`, which most
    experiments name, is `unspecified` in every field, so the profile alone says
    nothing about where a number came from."""
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    if not cpu_model:
        cpu_model = platform.processor() or None
    memory_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    if memory_bytes is None:
        try:
            memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            memory_bytes = None
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory_bytes,
    }


def hardware_profile_record(relative: str | None) -> dict | None:
    """The declared profile's bytes and contents, not only its path, and which of
    its fields still say `unspecified`."""
    if not relative:
        return None
    path = ROOT / relative
    if not path.is_file():
        return {"path": relative, "sha256": None, "contents": None, "unspecified_fields": None}
    contents = load(path)
    return {
        "path": relative,
        "sha256": sha(path),
        "contents": contents,
        "unspecified_fields": sorted(
            key for key, value in contents.items() if value == "unspecified"
        ),
    }


def toolchain(command: list[str]) -> str | None:
    """`rustc --version` for the toolchain a cargo command runs on: the `+name` it
    names, or the one rust-toolchain.toml pins for the repository root."""
    if not command or Path(command[0]).name != "cargo":
        return None
    query = ["rustc"]
    if len(command) > 1 and command[1].startswith("+"):
        query.append(command[1])
    query.append("--version")
    try:
        return subprocess.check_output(query, cwd=ROOT, text=True).strip()
    except Exception:
        return None


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def write_json_exclusive(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


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
    # The loop above sees only what the registry names. A manifest the registry
    # omits cannot be run (resolve() refuses it) and was checked by nothing,
    # which is where every experiment the old scaffolder produced ended up.
    registered = {item["path"] for item in registry().values()}
    for path in sorted((ROOT / "experiments").glob("**/experiment.toml")):
        relative = path.parent.relative_to(ROOT / "experiments").as_posix()
        if relative not in registered:
            errors.append(f"experiments/{relative}: not listed in experiments/registry.toml")
    if errors:
        print("\n".join("ERROR: " + error for error in errors))
        return 1
    print(f"OK: validated {len(registry())} experiments")
    return 0


def base_record(exp_id: str, data: dict, root: Path) -> dict:
    # Version 2 adds the worktree state, the measured host and the declared
    # hardware profile's contents; every version-1 field keeps its meaning.
    return {
        "schema_version": 2,
        "experiment_id": exp_id,
        "git_sha": git_sha(),
        **git_worktree_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "host": host_facts(),
        "manifest": data,
        "manifest_sha256": sha(root / "experiment.toml"),
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
        "uv_lock_sha256": sha(ROOT / "training/uv.lock"),
        "hardware_profile": data.get("hardware_profile"),
        "hardware_profile_record": hardware_profile_record(data.get("hardware_profile")),
    }


def prepare(exp_id: str):
    _, root, data = resolve(exp_id)
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
    _, root, data = resolve(exp_id)
    try:
        command = build_command(
            data, entrypoint=entrypoint, seed=seed, params=params or {}
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    results = root / data.get("results_dir", "results")
    timestamp = utc_stamp()
    record = base_record(exp_id, data, root)
    record.update(
        {
            "status": "running",
            "started_at": timestamp,
            "entrypoint": entrypoint,
            "seed": seed,
            "parameters": params or {},
            "command": command,
            "rustc": toolchain(command),
        }
    )

    execution = execute_command(command)
    exit_code = execution["exit_code"]
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


def metric_rows(stdout: str) -> list[dict]:
    """The JSON objects a run printed, one per line. A line that starts like an
    object but does not parse is an error rather than a row silently lost."""
    rows = []
    for number, line in enumerate(stdout.splitlines(), 1):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"stdout line {number} is not valid JSON: {error}") from None
        if isinstance(value, dict):
            rows.append(value)
    return rows


def metric_value(name: str, value) -> float | None:
    """A row field as a metric: a number, as a float. Strings are labels, and
    booleans, nulls, lists and objects are not metrics. A number too large for a
    float is refused rather than rounded to infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"metric {name} = {value} does not fit a float") from None


def summarize(by_seed: dict[int, float]) -> dict:
    if not by_seed:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "by_seed": {}}
    ordered = [by_seed[seed] for seed in sorted(by_seed)]
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        # Sample standard deviation: the seeds are a sample of the runs one could
        # have made, not the whole population of them.
        "std": statistics.stdev(ordered) if len(ordered) > 1 else None,
        "min": min(ordered),
        "max": max(ordered),
        "by_seed": {str(seed): by_seed[seed] for seed in sorted(by_seed)},
    }


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True)


def aggregate(
    exp_id: str,
    *,
    entrypoint: str,
    git_sha_filter: str | None = None,
    allow_dirty: bool = False,
) -> int:
    """Summarize one entrypoint's run records across seeds into a new, immutable
    aggregate record.

    A row is a JSON object a run printed on its own line. Its string fields name
    the row (for example `{"arm": "typed", "split": "ood"}`) and its numeric
    fields are the metrics summarized across seeds; a numeric `seed` field is
    taken as a label, not a metric.

    Refused, so nothing is written: records from more than one commit; records
    from a dirty or unrecorded worktree (unless `allow_dirty`, which the aggregate
    then records); records that differ in manifest, parameters, toolchain or host,
    since their numbers do not measure the same thing; a seed the manifest does
    not declare; two completed records for one seed; one row name printed twice by
    one run; and a record that is not a well-formed run record.

    Reported, never dropped, and each one makes the aggregate `incomplete`: failed
    runs; declared seeds with no completed run; a row some completed seeds did not
    print; a metric some seeds did not report; and a non-finite value (NaN or
    infinity), which is listed per seed instead of entering the mean."""
    _, root, data = resolve(exp_id)
    results = root / data.get("results_dir", "results")
    declared = list(data.get("seeds", []))

    records = []
    for path in sorted(results.glob("run-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path.name}: not a readable run record: {error}") from None
        if not isinstance(record, dict):
            raise ValueError(f"{path.name}: not a run record (not a JSON object)")
        if record.get("entrypoint") != entrypoint:
            continue
        if git_sha_filter and record.get("git_sha") != git_sha_filter:
            continue
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path.name}: seed must be an integer, found {seed!r}")
        if seed not in declared:
            raise ValueError(f"{path.name}: seed {seed} is not one of the manifest's seeds {declared}")
        records.append((path, record))
    if not records:
        raise ValueError(f"no run records for entrypoint {entrypoint!r} in {results.relative_to(ROOT)}")

    shas = sorted({str(record.get("git_sha")) for _, record in records})
    if len(shas) > 1:
        raise ValueError(
            f"records span {len(shas)} commits ({', '.join(shas)}); pass --git-sha to choose one"
        )
    unclean = [path.name for path, record in records if record.get("git_dirty") is not False]
    if unclean and not allow_dirty:
        raise ValueError(
            f"{len(unclean)} records come from a dirty or unrecorded worktree "
            f"({', '.join(unclean)}); pass --allow-dirty to aggregate them anyway"
        )
    for key in ("manifest_sha256", "parameters", "rustc", "host"):
        seen = sorted({canonical(record.get(key)) for _, record in records})
        if len(seen) > 1:
            raise ValueError(f"records differ in {key} ({' vs '.join(seen)}); they do not measure the same thing")

    completed: dict[int, tuple[Path, dict]] = {}
    failed = []
    for path, record in records:
        seed = record["seed"]
        if record.get("status") == "completed":
            if seed in completed:
                raise ValueError(
                    f"seed {seed} has two completed records "
                    f"({completed[seed][0].name}, {path.name}); pass --git-sha or remove one"
                )
            completed[seed] = (path, record)
        else:
            failed.append(
                {
                    "seed": seed,
                    "status": record.get("status"),
                    "exit_code": record.get("exit_code"),
                    "record": path.name,
                }
            )

    # group label -> metric -> seed -> value (finite) / non-finite spelling
    finite: dict[tuple, dict[str, dict[int, float]]] = {}
    non_finite: dict[tuple, dict[str, dict[int, str]]] = {}
    printed_by: dict[tuple, set[int]] = {}
    for seed, (path, record) in sorted(completed.items()):
        stdout = record.get("stdout")
        if not isinstance(stdout, str):
            raise ValueError(f"{path.name}: stdout must be text, found {type(stdout).__name__}")
        try:
            rows = metric_rows(stdout)
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from None
        seen = set()
        for row in rows:
            key = tuple(sorted((k, v) for k, v in row.items() if isinstance(v, str)))
            if key in seen:
                raise ValueError(f"{path.name}: row {dict(key)} is printed more than once")
            seen.add(key)
            printed_by.setdefault(key, set()).add(seed)
            finite.setdefault(key, {})
            non_finite.setdefault(key, {})
            for name, raw in row.items():
                if name == "seed":
                    continue
                try:
                    value = metric_value(name, raw)
                except ValueError as error:
                    raise ValueError(f"{path.name}: {error}") from None
                if value is None:
                    continue
                if math.isfinite(value):
                    finite[key].setdefault(name, {})[seed] = value
                else:
                    non_finite[key].setdefault(name, {})[seed] = repr(value)
    if completed and not printed_by:
        raise ValueError("the completed runs printed no JSON metric rows")

    completed_seeds = sorted(completed)
    groups = []
    gaps = False
    for key in sorted(printed_by):
        missing_row = sorted(set(completed_seeds) - printed_by[key])
        metrics = {}
        for name in sorted(set(finite[key]) | set(non_finite[key])):
            by_seed = finite[key].get(name, {})
            bad = non_finite[key].get(name, {})
            summary = summarize(by_seed)
            summary["missing_seeds"] = sorted(set(completed_seeds) - set(by_seed) - set(bad))
            summary["non_finite"] = {str(seed): bad[seed] for seed in sorted(bad)}
            gaps = gaps or bool(summary["missing_seeds"] or summary["non_finite"])
            metrics[name] = summary
        gaps = gaps or bool(missing_row)
        groups.append({"key": dict(key), "missing_seeds": missing_row, "metrics": metrics})

    missing = sorted(set(declared) - set(completed))
    status = "complete" if not missing and not failed and not gaps else "incomplete"
    timestamp = utc_stamp()
    first = records[0][1]
    summary = {
        "schema_version": 1,
        "kind": "aggregate",
        "experiment_id": exp_id,
        "entrypoint": entrypoint,
        "git_sha": shas[0],
        "allow_dirty": allow_dirty,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "parameters": first.get("parameters"),
        "rustc": first.get("rustc"),
        "host": first.get("host"),
        "declared_seeds": declared,
        "completed_seeds": completed_seeds,
        "missing_seeds": missing,
        "failed_runs": failed,
        "source_records": [
            {"path": path.name, "sha256": sha(path)} for path, _ in records
        ],
        "groups": groups,
    }
    out = results / f"aggregate-{timestamp}-{entrypoint}.json"
    write_json_exclusive(out, summary)
    print(out.relative_to(ROOT))
    return 0 if status == "complete" else 1


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

    agg = sub.add_parser("aggregate")
    agg.add_argument("id")
    agg.add_argument("--entrypoint", default="entrypoint")
    agg.add_argument("--git-sha")
    agg.add_argument("--allow-dirty", action="store_true")

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
    if args.cmd == "aggregate":
        try:
            code = aggregate(
                args.id,
                entrypoint=args.entrypoint,
                git_sha_filter=args.git_sha,
                allow_dirty=args.allow_dirty,
            )
        except ValueError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(code)
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
