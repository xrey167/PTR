"""Aggregate the L004 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run L004 --seed <s>
--set iterations=<n>` writes (results/run-<timestamp>-seed-<s>.json); by
default the newest record of every declared seed. Every declared seed must be
present, and the run passes only if every seed exited 0 with no hard failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"

COVERAGE = [
    "crashes_committed",
    "crashes_rolled_back",
    "commit_faults",
    "client_aborts",
    "server_kills",
    "redeliveries",
    "gap_probes",
    "duel_probes",
    "foreign_probes",
    "backup_catchups",
    "nul_probes",
    "long_replay_commits",
]
HARD = [
    "invalid_logs",
    "state_divergences",
    "extra_or_missing_keys",
    "lifecycle_divergences",
    "revision_divergences",
    "event_log_divergences",
    "watermark_divergences",
    "crash_recovery_failures",
    "commit_fault_failures",
    "redelivery_failures",
    "gap_failures",
    "foreign_histories_accepted",
    "false_refusals",
    "catchup_divergences",
    "rebuild_divergences",
    "duel_failures",
    "turso_divergences",
    "read_failures",
    "nul_failures",
]
SCOPE = (
    "generated ledgers of every event kind replayed into a PostgreSQL projection on one "
    "local server, with client and server crashes, commits the server fails, redeliveries, "
    "gaps, dueling projectors, forked histories, a backup catch-up, rebuilds onto unrelated "
    "and true histories, "
    "compared with the runtime replay, the MaterializedState reference and Turso"
)
LIMITATIONS = [
    "one PostgreSQL 18 server with pgvector 0.8 on loopback; 16 and 17 are supported but not run here",
    "timings come from one shared cloud container with the seeds run one after another; the "
    "server's durability settings are part of each seed's server string; not a capacity claim",
    "logs of 40-160 records per case plus one long replay per seed; no multi-million-commit ledger",
    "crashes are session terminations and dropped clients; no power loss, torn page or storage fault",
    "readers are not run concurrently with the projector; the fence semantics are covered by the ptr-pg tests",
    "the seed fixes every log and probe; only whether a crashed transaction had committed depends on timing",
    "Turso exposes point reads only, so it is compared for every key the reference holds but extra Turso keys would not show",
]


def last_json_line(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit("a run record has no result line")


def newest_per_seed(paths: list[Path]) -> dict[int, Path]:
    chosen: dict[int, Path] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        chosen[int(record["seed"])] = path
    return chosen


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    args = parser.parse_args()

    manifest = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
    paths = args.inputs or sorted(RESULTS.glob("run-*-seed-*.json"))
    per_seed = newest_per_seed(paths)
    missing = sorted(set(manifest["seeds"]) - set(per_seed))
    if missing:
        raise SystemExit(f"no run record for seeds {missing}")

    seeds = []
    records = []
    for seed in manifest["seeds"]:
        record = json.loads(per_seed[seed].read_text(encoding="utf-8"))
        result = last_json_line(record["stdout"])
        seeds.append(result)
        records.append(
            {"seed": seed, "record": per_seed[seed].name, "exit_code": record["exit_code"]}
        )

    counters = sorted(
        key
        for key, value in seeds[0].items()
        if isinstance(value, int) and not isinstance(value, bool) and key not in ("seed", "iterations")
    )
    totals = {key: sum(result.get(key, 0) for result in seeds) for key in counters}
    totals["cases"] = sum(result["iterations"] for result in seeds)
    long_ns, long_commits = totals.get("long_replay_ns", 0), totals.get("long_replay_commits", 0)
    derived = {
        "rebuild_seconds_per_100k_commits": (
            round(long_ns / long_commits * 100_000 / 1e9, 1) if long_commits else None
        ),
        "apply_ms_per_commit_with_crashes": (
            round(totals["apply_ns"] / totals["apply_commits"] / 1e6, 3)
            if totals.get("apply_commits")
            else None
        ),
        "crashes_committed_share": (
            round(totals["crashes_committed"] / totals["crash_injections"], 3)
            if totals.get("crash_injections")
            else None
        ),
    }
    mutations_path = RESULTS / "mutations.json"
    mutations = None
    if mutations_path.exists():
        recorded = json.loads(mutations_path.read_text(encoding="utf-8"))
        mutations = {"killed": recorded["killed"], "total": recorded["total"], "git_sha": recorded["git_sha"]}

    hard_failures = sum(totals.get(key, 0) for key in HARD)
    hard_pass = hard_failures == 0 and all(record["exit_code"] == 0 for record in records)
    # A run that never reached a probe proves nothing about it: every seed must
    # have crashed on both sides of the commit and exercised every probe.
    coverage = {
        f"seed {result['seed']}: {name}": result.get(name, 0) > 0
        for result in seeds
        for name in COVERAGE
    }
    coverage_ok = all(coverage.values())
    servers = sorted({result["server"] for result in seeds})

    metrics = {
        "experiment_id": "L004",
        "scope": SCOPE,
        "server": servers,
        "turso_oracle": all(result.get("turso_oracle") for result in seeds),
        "iterations_per_seed": sorted({result["iterations"] for result in seeds}),
        "seeds": seeds,
        "totals": totals,
        "derived": derived,
        "hard_failures": hard_failures,
        "probe_coverage": coverage,
        "mutation_checks": mutations,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run = {
        "experiment_id": "L004",
        "status": manifest["status"],
        "scope": SCOPE,
        "git_sha": git_sha(),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "entrypoint": manifest["entrypoint"],
        "seeds": records,
        "hard_pass": hard_pass,
        "probe_coverage_ok": coverage_ok,
        "mutation_checks": mutations,
        "limitations": LIMITATIONS,
    }
    (RESULTS / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"L004 hard_pass={hard_pass} coverage_ok={coverage_ok} cases={totals['cases']} "
        f"hard_failures={hard_failures}"
    )


if __name__ == "__main__":
    main()
