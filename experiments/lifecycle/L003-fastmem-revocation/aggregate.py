"""Aggregate the L003 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run L003 --seed <s>
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
    "revocations_with_removal",
    "window_denials",
    "stale_checkpoints_refused",
    "planted_stale_checkpoints",
    "checkpoints_verified",
    "inadmissible_probes",
    "process_crashes",
    "append_crashes_committed",
    "writes_lost_in_crashes",
    "revocation_crashes_committed",
    "revocation_crashes_rolled_back",
    "append_races_append_first",
    "append_races_revocation_first",
    "checkpoint_races_stored",
    "checkpoint_races_refused",
]
HARD = [
    "invalid_logs",
    "bit_identity_failures",
    "readout_failures",
    "resurrected_reads",
    "false_denials",
    "admission_divergences",
    "journal_mismatches",
    "removed_count_mismatches",
    "inadmissible_appends_accepted",
    "append_false_refusals",
    "checkpoint_false_refusals",
    "stale_checkpoints_accepted",
    "checkpoint_violations",
    "stale_checkpoint_rows",
    "checkpoints_lost",
    "atomicity_failures",
    "crash_recovery_failures",
    "race_violations",
    "read_failures",
]
SCOPE = (
    "fast memories with PostgreSQL journals and checkpoints, fed from lifecycle targets a "
    "ledger commits, supersedes and revokes through the projector, with crashes mid-append, "
    "mid-revocation and between operations, appends and checkpoints racing revocations and "
    "stale checkpoints planted in storage, compared bit for bit with the never-saw-it fold "
    "the runtime replay admits"
)
LIMITATIONS = [
    "one PostgreSQL 18 server on loopback; one process per memory, as the design has it",
    "memory shapes up to 4 heads of 12x12 and journals up to a few hundred writes",
    "an input's digest is fixed per generation: an input edit at an unchanged generation is the consumer's to exclude and is not exercised",
    "exact revocation only: copies in dead tuples, WAL and backups are storage erasure, audited separately",
    "the never-saw-it fold uses ptr-fastmem's own delta-rule arithmetic; that arithmetic is checked by the crate's unit tests, not here",
    "a restore always folds the whole journal; restoring from a checkpoint plus the journal suffix has no API and is not exercised",
    "the process is handed each committed record; reading the projection event log to drive the refold is not exercised",
    "the seed fixes every choice; crash and race outcomes depend on timing",
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


def ratio(numerator: int, denominator: int, digits: int = 3):
    return round(numerator / denominator, digits) if denominator else None


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
    totals["max_replayed"] = max(result["max_replayed"] for result in seeds)
    totals["max_journal_len"] = max(result["max_journal_len"] for result in seeds)
    derived = {
        "mean_writes_replayed_per_revocation": ratio(
            totals["writes_replayed"], totals["revocations_with_removal"], 2
        ),
        "replayed_share_of_full_refold": ratio(
            totals["writes_replayed"], totals["full_refold_writes"]
        ),
        "window_denial_share": ratio(totals["window_denials"], totals["window_reads"]),
        "append_race_append_first_share": ratio(
            totals["append_races_append_first"], totals["append_races"]
        ),
        "checkpoint_race_stored_share": ratio(
            totals["checkpoint_races_stored"], totals["checkpoint_races"]
        ),
        "revocation_crash_committed_share": ratio(
            totals["revocation_crashes_committed"], totals["revocation_crashes"]
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
    # have seen both outcomes of every crash and race and exercised every probe.
    coverage = {
        f"seed {result['seed']}: {name}": result.get(name, 0) > 0
        for result in seeds
        for name in COVERAGE
    }
    coverage_ok = all(coverage.values())
    servers = sorted({result["server"] for result in seeds})

    metrics = {
        "experiment_id": "L003",
        "scope": SCOPE,
        "server": servers,
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
        "experiment_id": "L003",
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
        f"L003 hard_pass={hard_pass} coverage_ok={coverage_ok} cases={totals['cases']} "
        f"hard_failures={hard_failures}"
    )


if __name__ == "__main__":
    main()
