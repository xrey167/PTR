"""Aggregate the S003 seed runs into results/metrics.json and results/run.json.

Inputs are the run records `scripts/run_experiment.py run S003 --seed <s>`
writes (results/run-<timestamp>-seed-<s>.json); by default the newest record
of every declared seed whose command started, one whose launch failed only
where none did (`newest_per_seed`). Every declared seed must be present. The
records must be of one configuration and have run the checkout's code, this
script included, and each record's harness result must report the benchmark,
seed and case count its wrapper ran and echo the canonical text of the
preregistered table the manifest's digest names
(`scripts/experiment_records.py`). results/mutations.json, when present, must
be this experiment's evidence from the same code, mutation checker and
mutation plan; otherwise aggregation is refused rather than carrying counts
of other code.

The verdicts, as `README.md` preregisters them:

- `hard_pass`: every seed exited 0 with no hard counter above zero;
- `coverage_ok`: every coverage counter is above zero in every seed, and
  every hazard class had at least `min_hazard_trials_per_class_per_seed`
  trials in every seed;
- `time_model_ok`: the merge-time histogram's 99th percentile bucket is at
  most `merge_wall_budget_us_p99`, so think time dominates merge time as the
  time model assumes;
- `low_cells_ok`: every agent count has a low cell, and `manipulation_ok`: for
  every agent count the pooled conflict rate over its low cells in these
  seeds is below `conflict_threshold_permille`;
- `mutations_ok`: mutations.json is there and every mutation is killed;
- `throughput_verdict`: over each agent count's low cells, the paired case
  bootstrap of gain_N (serial makespan over certified makespan) is `supported`
  if the lower bound of its interval is above 1.0 for every N, `rejected` if
  the upper bound is at most 1.0 for some N, and `inconclusive` otherwise or
  whenever a condition above fails;
- `efficiency_verdict`: `met` if gain_N / N reaches `efficiency_floor_permille`
  for every N at its point estimate, else `missed`; it does not decide the
  status.

`recommended_status` is `completed` only for a hard pass with every condition
met and a `supported` throughput verdict; anything else, an inconclusive
result included, is `failed`, as preregistered.

metrics.json and run.json are published as one aggregate
(`experiment_records.publish_aggregate`): run.json names the SHA-256 of the
metrics.json written with it. Only then does writing fresh results remove
results/STALE.toml.

`--pilot` reads the pilot outputs (pilot/pilot-seed-<s>.json, each the JSON
line the harness printed for a pilot seed run directly), classifies each
(contention level, agent count) cell as low when its pooled conflict rate is
below `conflict_threshold_permille`, checks the preregistered minimums, and
prints the `low_cells` value the freeze pins; it writes nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
PILOT = HERE / "pilot"
BENCHMARK = "certified-branches"

sys.path.insert(0, str(ROOT / "scripts"))
import experiment_records  # noqa: E402

HARD = [
    "lost_updates",
    "stale_read_merges",
    "undetected_phantoms",
    "stale_scan_merges",
    "stale_input_merges",
    "stale_reliance_merges",
    "lost_increments",
    "serialization_divergences",
    "invariant_violations",
    "spurious_refusals",
    "refusal_kind_mismatches",
    "conflict_key_errors",
    "rebase_classification_errors",
    "ungated_commits",
    "approval_bypasses",
    "double_merges",
    "bypass_commits_accepted",
    "reserved_writes_accepted",
    "seal_invariant_failures",
    "undeclared_input_merges",
    "forged_histories_accepted",
    "provenance_mismatches",
    "model_divergences",
    "program_mirror_divergences",
    "replay_divergences",
    "canary_misses",
    "nondeterminism",
    "harness_errors",
]
# Each must be above zero in every seed, or the run proves nothing about it.
COVERAGE = [
    "merges_clean",
    "merges_rebased",
    "conflicts",
    "lifecycle_refusals",
    "verification_holds",
    "escalations",
    "reviewed_merges",
    "no_change_merges",
    "lww_lost_updates",
    "lww_lost_increments",
    "occ_undetected_phantoms",
    "occ_stale_input_commits",
    "occ_stale_reliance_commits",
    *(f"probe_p{number}_exercised" for number in range(1, 27)),
    "canaries_run",
    "replays",
    "durable_roundtrips",
    "compaction_roundtrips",
]
# The trials of each hazard class of certification, each at least the
# preregistered minimum in every seed.
HAZARDS = [
    "hazard_write",
    "hazard_read",
    "hazard_scan_keys",
    "hazard_scan_values",
    "hazard_inputs",
    "hazard_lifecycle",
    "hazard_rebase",
    "hazard_negative",
    "hazard_set_member",
]
# The merge-time histogram's buckets, with their upper bounds in microseconds.
BUCKETS = [
    ("merge_wall_le_10us", 10),
    ("merge_wall_le_100us", 100),
    ("merge_wall_le_1ms", 1_000),
    ("merge_wall_le_10ms", 10_000),
    ("merge_wall_gt_10ms", None),
]
SCOPE = (
    "certified agent branches merged through PtrRuntime::merge_branch in memory, against a reference "
    "model and oracle that share no code with certification, in simulated ticks: five arms (serial, "
    "certified, certified-review, last-writer-wins, key-level OCC) at 2, 4, 8 and 16 agents over six "
    "contention levels, with twenty-six adversarial probes and nine oracle canaries in every case"
)
LIMITATIONS = [
    "simulated agents are pure programs; the certification and verification paths are exercised, not agent reasoning",
    "the ledger is in memory and the runtime one single writer; stored branches, seal tags and projections are "
    "covered by ptr-pg tests, not here",
    "the agent-time model makes think time dominate merge time by design, so the throughput half is expected to "
    "hold whenever waste is modest; its informative content is the efficiency and the waste",
    "the review arm has one reviewer and a plan digest that covers the revision, so most approvals are void; "
    "it is reported, not tuned",
    "timings come from one shared cloud container with the seeds run one after another; the hardware profile "
    "is unspecified until measured; not a capacity claim",
    "the generalisation of the safety claim is the rule-of-three bound 3/n per hazard class, not a proof",
]


def newest_per_seed(paths: list[Path]) -> dict[int, Path]:
    """The record to aggregate for each seed: the newest by name of those
    whose run finished (`experiment_records.finished_run`), and the newest
    of the others only where none finished. A failed launch ran nothing, so
    one named later than its retry, as a clock set back between them names
    it, is not the seed's run; nor is a reservation nothing finished, left
    by a runner that died, which holds no outcome."""
    chosen: dict[int, tuple[bool, Path]] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = int(record["seed"])
        launched = experiment_records.finished_run(record)
        if launched or not chosen.get(seed, (False, path))[0]:
            chosen[seed] = (launched, path)
    return {seed: path for seed, (_, path) in chosen.items()}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def cell_name(level: int, agents: int) -> str:
    return f"L{level}N{agents}"


def certified_runs(results: list[dict]):
    """(level, agents, serial ticks, certified run) for every case of every
    result and every agent count, in order."""
    for result in results:
        for case in result["cases"]:
            for run in case["runs"]:
                if run["arm"] == "certified":
                    yield case["level"], run["agents"], case["serial_ticks"], run


def conflict_rate(runs: list[dict]) -> float | None:
    """Certification refusals (a conflict or a lifecycle change) over merge
    attempts, pooled over `runs`; verification holds are not conflicts."""
    attempts = sum(run["attempts"] for run in runs)
    if not attempts:
        return None
    return sum(run["conflicts"] + run["lifecycle_refusals"] for run in runs) / attempts


def cells(results: list[dict]) -> dict[str, dict]:
    """Per (level, agents) cell of the certified arm: cases, attempts, the
    pooled conflict rate and certified merge rate, and the gain and its
    efficiency over serial execution at the point estimate."""
    grouped: dict[tuple[int, int], list[tuple[int, dict]]] = {}
    for level, agents, serial, run in certified_runs(results):
        grouped.setdefault((level, agents), []).append((serial, run))
    table = {}
    for (level, agents), members in sorted(grouped.items()):
        runs = [run for _, run in members]
        attempts = sum(run["attempts"] for run in runs)
        gain = sum(serial for serial, _ in members) / sum(run["ticks"] for run in runs)
        table[cell_name(level, agents)] = {
            "level": level,
            "agents": agents,
            "cases": len(members),
            "attempts": attempts,
            "conflict_rate": conflict_rate(runs),
            "merge_rate": sum(run["merged"] for run in runs) / attempts if attempts else None,
            "gain": gain,
            "efficiency": gain / agents,
        }
    return table


def pilot_classification(results: list[dict], table: dict) -> dict:
    """The pilot's cells: which are low (pooled conflict rate below
    `conflict_threshold_permille`), whether at least
    `min_cells_per_side` are on each side and every agent count has a low
    cell, and the `low_cells` value to pin."""
    threshold = table["conflict_threshold_permille"] / 1000
    per_cell = cells(results)
    low = [name for name, cell in per_cell.items() if cell["conflict_rate"] is not None and cell["conflict_rate"] < threshold]
    high = [name for name in per_cell if name not in low]
    every_agent_low = all(any(per_cell[name]["agents"] == agents for name in low) for agents in table["agents"])
    minimum = table["min_cells_per_side"]
    reasons = []
    if len(low) < minimum:
        reasons.append(f"{len(low)} low cells, fewer than {minimum}")
    if len(high) < minimum:
        reasons.append(f"{len(high)} cells at or above the threshold, fewer than {minimum}")
    if not every_agent_low:
        reasons.append("an agent count has no low cell")
    return {
        "cells": per_cell,
        "low_cells": low,
        "high_cells": high,
        "ok": not reasons,
        "reasons": reasons,
        "low_cells_toml": "low_cells = [" + ", ".join(json.dumps(name) for name in low) + "]",
    }


def bootstrap_gain(pairs: list[tuple[int, int]], resamples: int, seed: int, interval_permille: int) -> dict:
    """The point estimate of gain (the sum of serial makespans over the sum
    of certified ones) over `pairs` of (serial, certified) makespans of one
    case each, and the percentile interval of `resamples` resamples of the
    cases with replacement, drawn from `random.Random(seed)`."""
    generator = random.Random(seed)
    count = len(pairs)
    gains = []
    for _ in range(resamples):
        chosen = [pairs[generator.randrange(count)] for _ in range(count)]
        gains.append(sum(serial for serial, _ in chosen) / sum(certified for _, certified in chosen))
    gains.sort()
    tail = (1000 - interval_permille) / 2000
    return {
        "cases": count,
        "gain": sum(serial for serial, _ in pairs) / sum(certified for _, certified in pairs),
        "lower": gains[int(tail * resamples)],
        "upper": gains[min(resamples - 1, int((1 - tail) * resamples))],
        "resamples": resamples,
    }


def merge_time(results: list[dict], budget_us: int) -> dict:
    """The merge-time histogram summed over the seeds and the bucket its
    99th percentile falls in; the time model holds when that bucket's upper
    bound is at most `budget_us`."""
    calls = sum(result["merge_calls"] for result in results)
    counts = {name: sum(result[name] for result in results) for name, _ in BUCKETS}
    running = 0
    p99_bound: int | None = None
    p99_bucket = BUCKETS[-1][0]
    for name, bound in BUCKETS:
        running += counts[name]
        if running >= 0.99 * calls:
            p99_bucket, p99_bound = name, bound
            break
    return {
        "calls": calls,
        "buckets": counts,
        "p99_bucket": p99_bucket,
        "p99_upper_bound_us": p99_bound,
        "budget_us": budget_us,
        "ok": calls > 0 and p99_bound is not None and p99_bound <= budget_us,
        "mean_us": round(sum(result["merge_wall_ns"] for result in results) / calls / 1000, 1) if calls else None,
    }


def share(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def descriptive(results: list[dict], totals: dict) -> dict:
    """What every result reports beside the verdict: the baselines'
    anomalies, the shares that trigger predicate digests and typed merge
    operators, the review voids, and the rule-of-three bound per hazard
    class."""
    runs = [run for result in results for case in result["cases"] for run in case["runs"]]
    by_arm = {}
    for run in runs:
        totals_of = by_arm.setdefault(run["arm"], {"runs": 0, "attempts": 0, "merged": 0, "conflicts": 0})
        totals_of["runs"] += 1
        for key in ("attempts", "merged", "conflicts"):
            totals_of[key] += run[key]
    certified = [run for run in runs if run["arm"] in ("certified", "serial", "certified-review")]
    refusals = sum(run["conflicts"] for run in certified)
    reviews = [run for run in runs if run["arm"] == "certified-review"]
    return {
        "arms": by_arm,
        "lww_anomalies_per_commit": share(
            totals["lww_lost_updates"] + totals["lww_lost_increments"], by_arm.get("lww", {}).get("merged", 0)
        ),
        "occ_anomalies_per_commit": share(
            totals["occ_undetected_phantoms"] + totals["occ_stale_input_commits"] + totals["occ_stale_reliance_commits"],
            by_arm.get("occ", {}).get("merged", 0),
        ),
        "unnecessary_refusal_share": share(totals["unnecessary_refusals"], refusals),
        "put_increment_share": share(totals["put_increment_conflicts"], refusals),
        "review_void_share": share(sum(run["review_voids"] for run in reviews), sum(run["escalations"] for run in reviews)),
        "hazard_trials": {
            name: {"trials": totals[name], "rule_of_three_bound": round(3 / totals[name], 6) if totals[name] else None}
            for name in HAZARDS
        },
    }


def analyse(seeds: list[dict], records: list[dict], table: dict, mutations: dict | None) -> dict:
    """What the seeds' harness results (`seeds`, in manifest order) and the
    run records they came from (`records`) show, under the preregistered
    `table` and the mutation evidence `mutations` (None when there is none):
    the metrics, each condition, the throughput and efficiency verdicts and
    the status they recommend. Pure: it reads no file."""
    counters = sorted(
        {
            key
            for result in seeds
            for key, value in result.items()
            if isinstance(value, int) and not isinstance(value, bool) and key not in ("seed", "iterations", "elapsed_ns")
        }
    )
    totals = {key: sum(result.get(key, 0) for result in seeds) for key in counters}
    totals["cases"] = sum(result["iterations"] for result in seeds)
    hard_failures = sum(totals.get(key, 0) for key in HARD)
    hard_pass = not (
        hard_failures or totals["hard_failures"] or any(record["exit_code"] != 0 for record in records)
    )

    minimum = table["min_hazard_trials_per_class_per_seed"]
    coverage = {f"seed {result['seed']}: {name}": result.get(name, 0) > 0 for result in seeds for name in COVERAGE}
    coverage.update(
        {
            f"seed {result['seed']}: {name} at least {minimum}": result.get(name, 0) >= minimum
            for result in seeds
            for name in HAZARDS
        }
    )
    coverage_ok = all(coverage.values())
    time_model = merge_time(seeds, table["merge_wall_budget_us_p99"])

    low_cells = list(table["low_cells"])
    per_cell = cells(seeds)
    low_by_agents = {
        agents: [name for name in low_cells if per_cell.get(name, {}).get("agents") == agents]
        for agents in table["agents"]
    }
    unknown_cells = [name for name in low_cells if name not in per_cell]
    threshold = table["conflict_threshold_permille"] / 1000
    per_agents = {}
    for agents in table["agents"]:
        names = low_by_agents[agents]
        members = [
            (serial, run)
            for level, count, serial, run in certified_runs(seeds)
            if count == agents and cell_name(level, agents) in names
        ]
        rate = conflict_rate([run for _, run in members])
        entry = {
            "low_cells": names,
            "conflict_rate": rate,
            "manipulation_ok": rate is not None and rate < threshold,
        }
        if members:
            entry["bootstrap"] = bootstrap_gain(
                [(serial, run["ticks"]) for serial, run in members],
                table["bootstrap_resamples"],
                table["bootstrap_seed"],
                table["bootstrap_interval_permille"],
            )
            entry["efficiency"] = entry["bootstrap"]["gain"] / agents
            entry["efficiency_met"] = entry["efficiency"] >= table["efficiency_floor_permille"] / 1000
        per_agents[str(agents)] = entry
    low_cells_ok = bool(low_cells) and not unknown_cells and all(low_by_agents[agents] for agents in table["agents"])
    manipulation_ok = low_cells_ok and all(entry["manipulation_ok"] for entry in per_agents.values())
    mutations_ok = bool(mutations) and mutations["killed"] == mutations["total"]

    conditions = {
        "hard_pass": hard_pass,
        "coverage_ok": coverage_ok,
        "time_model_ok": time_model["ok"],
        "low_cells_ok": low_cells_ok,
        "manipulation_ok": manipulation_ok,
        "mutations_ok": mutations_ok,
    }
    if not low_cells_ok or not all(conditions.values()):
        throughput = "inconclusive"
    elif all(entry["bootstrap"]["lower"] > 1.0 for entry in per_agents.values()):
        throughput = "supported"
    elif any(entry["bootstrap"]["upper"] <= 1.0 for entry in per_agents.values()):
        throughput = "rejected"
    else:
        throughput = "inconclusive"
    efficiency = (
        "met"
        if low_cells_ok and all(entry.get("efficiency_met") for entry in per_agents.values())
        else "missed"
    )
    recommended = "completed" if all(conditions.values()) and throughput == "supported" else "failed"

    metrics = {
        "experiment_id": "S003",
        "scope": SCOPE,
        "cases_per_seed": sorted({result["iterations"] for result in seeds}),
        "seeds": [{key: value for key, value in result.items() if key not in ("cases", "preregistration")} for result in seeds],
        "totals": totals,
        "hard_failures": hard_failures,
        "coverage": coverage,
        "time_model": time_model,
        "cells": per_cell,
        "per_agent_count": per_agents,
        "throughput_verdict": throughput,
        "efficiency_verdict": efficiency,
        "descriptive": descriptive(seeds, totals),
        "mutation_checks": mutations,
    }
    return {
        "metrics": metrics,
        "conditions": conditions,
        "throughput": throughput,
        "efficiency": efficiency,
        "recommended": recommended,
        "hard_failures": hard_failures,
        "totals": totals,
    }


def aggregate() -> None:
    manifest = tomllib.loads((HERE / "experiment.toml").read_text(encoding="utf-8"))
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    table = config["preregistration"]
    try:
        canonical = experiment_records.preregistration_canonical(table)
        digest = experiment_records.preregistration_digest(table)
    except ValueError as error:
        raise SystemExit(f"S003: the preregistration cannot be read: {error}")
    if manifest.get("preregistration_sha256") != digest:
        raise SystemExit(
            f"S003: refusing to aggregate: experiment.toml names preregistration_sha256 "
            f"{manifest.get('preregistration_sha256')!r}, not {digest}, the digest of config.toml's table"
        )
    paths = sorted(RESULTS.glob("run-*-seed-*.json"))
    per_seed = newest_per_seed(paths)
    missing = sorted(set(manifest["seeds"]) - set(per_seed))
    if missing:
        raise SystemExit(f"no run record for seeds {missing}")

    chosen = {seed: json.loads(per_seed[seed].read_text(encoding="utf-8")) for seed in manifest["seeds"]}
    named = {per_seed[seed].name: chosen[seed] for seed in manifest["seeds"]}
    results_path = experiment_records.relative_to_root(RESULTS, ROOT)
    experiment_path = experiment_records.relative_to_root(HERE, ROOT)
    try:
        revision = experiment_records.source_revision(
            "S003",
            manifest,
            named,
            ROOT,
            HERE,
            paths=experiment_records.listed_record_paths(results_path),
            checkout_paths=experiment_records.listed_staleness_paths(results_path, experiment_path),
            listed_experiment=True,
        )
        harness = experiment_records.harness_results(named, BENCHMARK, (*HARD, "hard_failures", "merge_calls"))
        mutations = experiment_records.mutation_evidence("S003", RESULTS / "mutations.json", BENCHMARK, ROOT, HERE)
        seed_records = experiment_records.seed_record_digests(RESULTS)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"S003: refusing to aggregate: {error}")

    seeds, records = [], []
    for seed in manifest["seeds"]:
        record = chosen[seed]
        result = harness[per_seed[seed].name]
        if result.get("preregistration") != canonical:
            raise SystemExit(f"S003: seed {seed} ran under another preregistration than config.toml's table")
        if result.get("iterations") != table["cases_per_seed"] or len(result.get("cases", [])) != table["cases_per_seed"]:
            raise SystemExit(f"S003: seed {seed} ran {result.get('iterations')!r} cases, not {table['cases_per_seed']}")
        seeds.append(result)
        records.append(
            {
                "seed": seed,
                "record": per_seed[seed].name,
                "git_sha": record["git_sha"],
                "exit_code": record["exit_code"],
            }
        )

    analysis = analyse(seeds, records, table, mutations)
    metrics, conditions = analysis["metrics"], analysis["conditions"]
    throughput, efficiency, recommended = analysis["throughput"], analysis["efficiency"], analysis["recommended"]
    totals, hard_failures = analysis["totals"], analysis["hard_failures"]
    run = {
        "experiment_id": "S003",
        "status": manifest["status"],
        "scope": SCOPE,
        "git_sha": revision,
        "aggregated_at_git_sha": git_sha(),
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": digest,
        "preregistration_rules_sha256": manifest.get("preregistration_rules_sha256"),
        # The command the records ran, which they agree on.
        "entrypoint": manifest[chosen[manifest["seeds"][0]]["entrypoint"]],
        "seeds": records,
        "seed_records": seed_records,
        **conditions,
        "throughput_verdict": throughput,
        "efficiency_verdict": efficiency,
        "recommended_status": recommended,
        "mutation_checks": mutations,
        "limitations": LIMITATIONS,
    }
    try:
        experiment_records.publish_aggregate(RESULTS, metrics, run)
    except experiment_records.ProvenanceError as error:
        raise SystemExit(f"S003: {error}")
    print(
        f"S003 recommended_status={recommended} throughput={throughput} efficiency={efficiency} "
        + " ".join(f"{name}={value}" for name, value in conditions.items())
        + f" cases={totals['cases']} hard_failures={hard_failures}"
    )


def pilot(paths: list[Path]) -> int:
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    table = config["preregistration"]
    files = paths or [PILOT / f"pilot-seed-{seed}.json" for seed in table["pilot_seeds"]]
    results = []
    for path in files:
        if not path.exists():
            raise SystemExit(f"S003: no pilot output {path}")
        result = experiment_records.last_result_line(path.read_text(encoding="utf-8"))
        if result.get("benchmark") != BENCHMARK:
            raise SystemExit(f"S003: {path.name} is not a {BENCHMARK} result")
        results.append(result)
    seeds = sorted(result["seed"] for result in results)
    if seeds != sorted(table["pilot_seeds"]):
        raise SystemExit(f"S003: the pilot outputs are of seeds {seeds}, not {sorted(table['pilot_seeds'])}")
    classification = pilot_classification(results, table)
    summary = {key: classification[key] for key in ("low_cells", "high_cells", "ok", "reasons")}
    summary["conflict_rates"] = {
        name: None if cell["conflict_rate"] is None else round(cell["conflict_rate"], 4)
        for name, cell in classification["cells"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(classification["low_cells_toml"])
    return 0 if classification["ok"] else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="classify the cells from the pilot outputs")
    parser.add_argument("inputs", nargs="*", type=Path, help="pilot outputs, for --pilot")
    args = parser.parse_args()
    if args.pilot:
        raise SystemExit(pilot(args.inputs))
    if args.inputs:
        raise SystemExit("S003: aggregate reads the records of results/; name inputs only with --pilot")
    aggregate()


if __name__ == "__main__":
    main()
