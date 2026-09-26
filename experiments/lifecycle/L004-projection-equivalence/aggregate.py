"""Aggregate the L004 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run L004 --seed <s>
--set iterations=<n>` writes (results/run-<timestamp>-seed-<s>.json); by
default the newest record of every declared seed. Every declared seed must be
present. The records must be of one configuration and have run the checkout's
code, this script included, and each record's harness result must report the
benchmark, seed and iteration count its wrapper ran (`scripts/experiment_records.py`);
`git_sha` in run.json is the commit they ran at, and `aggregated_at_git_sha` the
commit this script ran at. results/mutations.json, when present, must be this
experiment's evidence from the same code, mutation checker and mutation plan;
otherwise aggregation is refused rather than carrying counts of other code.

`verdict` in run.json is, in this order of precedence:

- `hard-fail`: a seed exited nonzero or a hard counter fired;
- `coverage-incomplete`: no failure, but a seed never reached a probe, so the
  run proves nothing about it;
- `no-turso-oracle`: it would pass, but a seed ran without Turso (the
  manifest's `postgres_entrypoint`), the third implementation the baseline
  names; a PostgreSQL-only run is never a hard pass;
- `hard-pass`: none of these; `hard_pass` is true for this verdict only.

Writing fresh results removes results/STALE.toml, which marks archived results
of older code (`scripts/check_research_gates.py`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
BENCHMARK = "projection-equivalence"

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

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

    chosen = {seed: json.loads(per_seed[seed].read_text(encoding="utf-8")) for seed in manifest["seeds"]}
    named = {per_seed[seed].name: chosen[seed] for seed in manifest["seeds"]}
    try:
        revision = experiment_records.source_revision("L004", manifest, named, ROOT, HERE)
        results = experiment_records.harness_results(named, BENCHMARK, (*HARD, "hard_failures"))
        mutations = experiment_records.mutation_evidence(
            "L004", RESULTS / "mutations.json", BENCHMARK, ROOT, HERE
        )
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"L004: refusing to aggregate: {error}")

    seeds = []
    records = []
    for seed in manifest["seeds"]:
        record = chosen[seed]
        seeds.append(results[per_seed[seed].name])
        records.append(
            {
                "seed": seed,
                "record": per_seed[seed].name,
                "git_sha": record["git_sha"],
                "exit_code": record["exit_code"],
            }
        )

    counters = sorted(
        {
            key
            for result in seeds
            for key, value in result.items()
            if isinstance(value, int) and not isinstance(value, bool) and key not in ("seed", "iterations")
        }
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
    hard_failures = sum(totals.get(key, 0) for key in HARD)
    # A run that never reached a probe proves nothing about it: every seed must
    # have crashed on both sides of the commit and exercised every probe.
    coverage = {
        f"seed {result['seed']}: {name}": result.get(name, 0) > 0
        for result in seeds
        for name in COVERAGE
    }
    coverage_ok = all(coverage.values())
    # Turso is the third implementation the baseline names; a seed run through
    # postgres_entrypoint compared only PostgreSQL with the references.
    turso_oracle = all(result.get("turso_oracle") is True for result in seeds)
    # The harness's own total also counts hard counters HARD may not list.
    if hard_failures or totals["hard_failures"] or any(record["exit_code"] != 0 for record in records):
        verdict = "hard-fail"
    elif not coverage_ok:
        verdict = "coverage-incomplete"
    elif not turso_oracle:
        verdict = "no-turso-oracle"
    else:
        verdict = "hard-pass"
    hard_pass = verdict == "hard-pass"
    servers = sorted({result["server"] for result in seeds})

    metrics = {
        "experiment_id": "L004",
        "scope": SCOPE,
        "server": servers,
        "turso_oracle": turso_oracle,
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
        "git_sha": revision,
        "aggregated_at_git_sha": git_sha(),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        # The command the records ran, which they agree on.
        "entrypoint": manifest[chosen[manifest["seeds"][0]]["entrypoint"]],
        "seeds": records,
        "verdict": verdict,
        "hard_pass": hard_pass,
        "probe_coverage_ok": coverage_ok,
        "turso_oracle": turso_oracle,
        "mutation_checks": mutations,
        "limitations": LIMITATIONS,
    }
    (RESULTS / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    experiment_records.clear_stale_marker(RESULTS)
    print(
        f"L004 verdict={verdict} hard_pass={hard_pass} coverage_ok={coverage_ok} cases={totals['cases']} "
        f"hard_failures={hard_failures}"
    )


if __name__ == "__main__":
    main()
