"""Aggregate exploratory run records; this does not publish lifecycle certification."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/registry.toml"

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


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def write_json_exclusive(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


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


def metric_rows(stdout: str) -> list[dict]:
    """The JSON objects a run printed, one per line. A line that starts like an
    object but does not parse raises ValueError rather than being silently lost."""
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
    float raises ValueError. Non-numeric values return None; existing NaN and
    infinity values pass through for the caller to classify.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"metric {name} = {value} does not fit a float") from None


def summarize(by_seed: dict[int, float]) -> dict:
    """Summarize per-seed values with sample deviation and None for unavailable statistics."""
    if not by_seed:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "by_seed": {}}
    seeds = sorted(by_seed)
    ordered = [by_seed[seed] for seed in seeds]
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        # Sample standard deviation: the seeds are a sample of the runs one could
        # have made, not the whole population of them.
        "std": statistics.stdev(ordered) if len(ordered) > 1 else None,
        "min": min(ordered),
        "max": max(ordered),
        "by_seed": {str(seed): by_seed[seed] for seed in seeds},
    }


def canonical(value) -> str:
    """Serialize a value with sorted JSON keys for stable comparisons."""
    return json.dumps(value, sort_keys=True)


def aggregate(
    exp_id: str,
    *,
    entrypoint: str,
    git_sha_filter: str | None = None,
    allow_dirty: bool = False,
    parameters_filter: dict[str, str] | None = None,
) -> int:
    """Summarize one entrypoint's run records across seeds into a new, immutable
    aggregate record.

    A row is a JSON object a run printed on its own line. Its string fields name
    the row (for example `{"arm": "typed", "split": "ood"}`) and its numeric
    fields are the metrics summarized across seeds; a numeric `seed` field is
    ignored.

    Refused, so nothing is written: records from more than one commit; records
    from a dirty or unrecorded worktree (unless `allow_dirty`, which the aggregate
    then records; an explicitly dirty record must still carry a valid diff hash);
    records that differ in manifest, parameters, dependency locks, toolchain or host,
    since their numbers do not measure the same thing; a seed the manifest does
    not declare; two completed records for one seed; one row name printed twice by
    one run; and a record that is not a well-formed run record.

    Reported, never dropped, and each one makes the aggregate `incomplete`: failed
    runs; declared seeds with no completed run; a row some completed seeds did not
    print; a metric some seeds did not report; and a non-finite value (NaN or
    infinity), which is listed per seed instead of entering the mean.

    `git_sha_filter` selects an exact recorded commit before comparisons. Input
    refusals raise ValueError, including no matching records or completed runs
    with no JSON rows. Return 0 after writing a complete aggregate, or 1 after
    writing an incomplete one. Unknown experiments raise SystemExit; manifest
    reads, source hashing, and output-write errors propagate.
    """
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
        if parameters_filter and any(record.get("parameters", {}).get(k) != v for k, v in parameters_filter.items()):
            continue
        if git_sha_filter and record.get("git_sha") != git_sha_filter:
            continue
        if record.get("experiment_id") != exp_id:
            raise ValueError(f"{path.name}: experiment_id must be {exp_id!r}, found {record.get('experiment_id')!r}")
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
    for path, record in records:
        if record.get("git_dirty") is True:
            digest = record.get("git_tracked_diff_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"{path.name}: dirty record requires a valid git_tracked_diff_sha256")
    for key in ("manifest_sha256", "parameters", "rustc", "python", "executable", "toolchain", "environment",
                "host", "git_tracked_diff_sha256",
                "cargo_lock_path", "cargo_lock_sha256", "command_cargo_lock_path", "command_cargo_lock_sha256"):
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
        for row in rows:
            key = tuple(sorted((k, v) for k, v in row.items() if isinstance(v, str)))
            row_seeds = printed_by.setdefault(key, set())
            if seed in row_seeds:
                raise ValueError(f"{path.name}: row {dict(key)} is printed more than once")
            row_seeds.add(seed)
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

    completed_seed_set = set(completed)
    completed_seeds = sorted(completed_seed_set)
    groups = []
    gaps = False
    for key in sorted(printed_by):
        missing_row = sorted(completed_seed_set - printed_by[key])
        metrics = {}
        for name in sorted(set(finite[key]) | set(non_finite[key])):
            by_seed = finite[key].get(name, {})
            bad = non_finite[key].get(name, {})
            summary = summarize(by_seed)
            summary["missing_seeds"] = sorted(completed_seed_set - by_seed.keys() - bad.keys())
            summary["non_finite"] = {str(seed): bad[seed] for seed in sorted(bad)}
            gaps = gaps or bool(summary["missing_seeds"] or summary["non_finite"])
            metrics[name] = summary
        gaps = gaps or bool(missing_row)
        groups.append({"key": dict(key), "missing_seeds": missing_row, "metrics": metrics})

    missing = sorted(set(declared) - completed_seed_set)
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
    parser.add_argument("command", choices=["aggregate"])
    parser.add_argument("id")
    parser.add_argument("--entrypoint", default="entrypoint")
    parser.add_argument("--git-sha")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--set", dest="params", action="append", default=[])
    args = parser.parse_args()
    try:
        code = aggregate(args.id, entrypoint=args.entrypoint, git_sha_filter=args.git_sha,
                         allow_dirty=args.allow_dirty, parameters_filter=parse_params(args.params))
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(code)

if __name__ == "__main__":
    main()
