"""The shared driver of the stage-A aggregators: read the run records, check their provenance and the frozen preregistration, judge
the seeds with the experiment's own functions, publish metrics.json and run.json.

`mod` is the experiment's aggregate module. It provides: HERE, ROOT, EXPERIMENT, BENCHMARK, CONFIG (its config.toml), SCOPE,
LIMITATIONS, METRIC_NAMES, GUARDED_PATHS, RESULTS_EXCLUDE, judge(result, exit_code, criteria, seed), paired_comparisons(results, paired),
verdict_of(judgements, static, declared, mutations, paired, criteria_paired, plan_size) and report(...). Nothing of the experiment is
run again; a verdict of any kind is published, a refusal (exit 2) is not a verdict."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable

import agg
from agg import COMPLETED, FALSIFIED, NOT_SHOWN, PairedError

TAIL = 2000


def refuse(message: str):
    """Refuse to aggregate: exit status 2, which is never a verdict about the hypothesis."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def run_command(command: str, root: Path) -> tuple[int, str]:
    try:
        process = subprocess.run(shlex.split(command), cwd=root, text=True, capture_output=True)
    except OSError as error:
        return -1, str(error)
    return process.returncode, process.stdout + process.stderr


def static_evidence(commands: list[str], runner: Callable[[str], tuple[int, str]]) -> list[dict]:
    results = []
    for command in commands:
        code, output = runner(command)
        results.append({"command": command, "exit_code": code, "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                        "output_tail": output[-TAIL:]})
    return results


def newest_per_seed(paths: list[Path], experiment: str, experiment_records) -> dict[int, Path]:
    """The record to aggregate for each seed: the one whose run finished, the newest of the others only where none finished. Two
    finished records of one seed are refused rather than resolved by name, so a repeated seed cannot be chosen after the fact."""
    chosen: dict[int, tuple[bool, Path]] = {}
    finished: dict[int, list[Path]] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = int(record["seed"])
        launched = experiment_records.finished_run(record)
        if launched:
            finished.setdefault(seed, []).append(path)
        if launched or not chosen.get(seed, (False, path))[0]:
            chosen[seed] = (launched, path)
    repeated = {seed: [p.name for p in found] for seed, found in sorted(finished.items()) if len(found) > 1}
    if repeated:
        refuse(f"{experiment}: refusing to aggregate: more than one finished run record for a seed ({repeated}); each declared seed runs once")
    return {seed: path for seed, (_, path) in chosen.items()}


def load_frozen(mod, experiment_records) -> tuple[dict, dict, dict, str]:
    """The manifest, the preregistered table, the criteria and the table's digest, once the manifest, the table, the preregistration
    text and the criteria agree. Raises SystemExit."""
    manifest = tomllib.loads((mod.HERE / "experiment.toml").read_text(encoding="utf-8"))
    if "preregistration" not in mod.CONFIG:
        refuse(f"{mod.EXPERIMENT}: refusing to aggregate: config.toml has no [preregistration] table ({mod.EXPERIMENT} is not frozen yet)")
    table = mod.CONFIG["preregistration"]
    try:
        digest = experiment_records.preregistration_digest(table)
    except ValueError as error:
        refuse(f"{mod.EXPERIMENT}: the preregistration cannot be read: {error}")
    if manifest.get("preregistration_sha256") != digest:
        refuse(f"{mod.EXPERIMENT}: refusing to aggregate: experiment.toml names preregistration_sha256 {manifest.get('preregistration_sha256')!r}, "
               f"not {digest}, the digest of config.toml's table")
    for key in ("criteria", "preregistration_text"):
        path = mod.ROOT / table[key]
        actual = experiment_records.preregistered_file_digest(path)
        if table.get(f"{key}_sha256") != actual:
            refuse(f"{mod.EXPERIMENT}: refusing to aggregate: {table[key]} is not the file frozen ({actual})")
    criteria = tomllib.loads((mod.ROOT / table["criteria"]).read_text(encoding="utf-8"))
    return manifest, table, criteria, digest


def aggregate(mod, report_path: Path | None = None) -> str:
    sys.path.insert(0, str(mod.ROOT / "scripts"))
    import experiment_records

    manifest, table, criteria, digest = load_frozen(mod, experiment_records)
    results_dir = mod.HERE / "results"
    per_seed = newest_per_seed(sorted(results_dir.glob("run-*-seed-*.json")), mod.EXPERIMENT, experiment_records)
    missing = sorted(set(manifest["seeds"]) - set(per_seed))
    if missing:
        refuse(f"no run record for seeds {missing}")
    chosen = {seed: json.loads(per_seed[seed].read_text(encoding="utf-8")) for seed in manifest["seeds"]}
    named = {per_seed[seed].name: chosen[seed] for seed in manifest["seeds"]}
    results_path = experiment_records.relative_to_root(results_dir, mod.ROOT)
    experiment_path = experiment_records.relative_to_root(mod.HERE, mod.ROOT)
    try:
        revision = experiment_records.source_revision(
            mod.EXPERIMENT, manifest, named, mod.ROOT, mod.HERE,
            paths=experiment_records.listed_record_paths(results_path),
            checkout_paths=experiment_records.listed_staleness_paths(results_path, experiment_path),
            listed_experiment=True,
        )
        seed_records = experiment_records.seed_record_digests(results_dir)
        mutations = experiment_records.mutation_evidence(mod.EXPERIMENT, results_dir / "mutations.json", mod.BENCHMARK, mod.ROOT, mod.HERE)
    except experiment_records.ProvenanceError as error:
        refuse(f"{mod.EXPERIMENT}: refusing to aggregate: {error}")
    if mutations is not None:
        changed = agg.code_changed_since(mutations["git_sha"], mod.ROOT, mod.GUARDED_PATHS, mod.RESULTS_EXCLUDE)
        if changed:
            refuse(f"{mod.EXPERIMENT}: refusing to aggregate: {changed}; rerun research/cseries/lib/run_mutations.py {mod.EXPERIMENT} on a clean committed tree")

    judgements, records, result_of = [], [], {}
    for seed in manifest["seeds"]:
        record = chosen[seed]
        result, error = None, None
        try:
            result = experiment_records.harness_results({per_seed[seed].name: record}, mod.BENCHMARK)[per_seed[seed].name]
        except experiment_records.ProvenanceError as problem:
            error = str(problem)
        judgement = mod.judge(result, record.get("exit_code"), criteria, seed)
        result_of[seed] = result
        if error:
            judgement["valid"] = False
            judgement["invalid"].append(error)
        judgements.append(judgement)
        records.append({"seed": seed, "record": per_seed[seed].name, "git_sha": record["git_sha"], "exit_code": record.get("exit_code")})
    agg.invalidate_environment_mismatches(judgements, {seed: (result_of[seed] or {}).get("environment") for seed in manifest["seeds"]})
    static = static_evidence(criteria.get("static_evidence", []), lambda command: run_command(command, mod.ROOT))
    paired = None
    if all(entry["valid"] for entry in judgements):
        try:
            paired = mod.paired_comparisons({seed: result_of[seed] for seed in manifest["seeds"]}, criteria["paired"])
        except PairedError as error:
            judgements[0]["invalid"].append(f"paired comparison: {error}")
            judgements[0]["valid"] = False
    metrics_seeds = [
        {**j, "metrics": {name: result_of[j["seed"]].get(name) for name in mod.METRIC_NAMES} if result_of[j["seed"]] else None,
         "summary": result_of[j["seed"]].get("summary") if result_of[j["seed"]] else None}
        for j in judgements
    ]
    plan = tomllib.loads((mod.HERE / "tests/mutations.toml").read_text(encoding="utf-8"))
    verdict = mod.verdict_of(judgements, static, manifest["seeds"], mutations, paired, criteria["paired"], len(plan["mutation"]))
    text = mod.report(verdict, judgements, static, mutations, paired)
    metrics = {"experiment_id": mod.EXPERIMENT, "scope": mod.SCOPE, "verdict": verdict["verdict"], "decision": verdict, "paired": paired,
               "seeds": metrics_seeds, "static_evidence": static, "mutation_checks": mutations, "report": text}
    run = {
        "experiment_id": mod.EXPERIMENT, "status": manifest["status"], "scope": mod.SCOPE, "git_sha": revision,
        "aggregated_at_git_sha": agg.git_sha(mod.ROOT), "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": digest, "preregistration_rules_sha256": manifest.get("preregistration_rules_sha256"),
        "criteria_sha256": table["criteria_sha256"], "entrypoint": manifest["entrypoint"], "seeds": records, "seed_records": seed_records,
        "verdict": verdict["verdict"], "paired": paired, "recommended_status": verdict["recommended_status"],
        "static_evidence": [{key: item[key] for key in ("command", "exit_code", "output_sha256")} for item in static],
        "mutation_checks": mutations, "limitations": mod.LIMITATIONS,
    }
    try:
        mod.publish(results_dir, metrics, run)
    except experiment_records.ProvenanceError as error:
        refuse(f"{mod.EXPERIMENT}: {error}")
    if report_path is not None:
        report_path.write_text(text, encoding="utf-8")
    print(f"{mod.EXPERIMENT} verdict={verdict['verdict']} kind={verdict['kind']} recommended_status={verdict['recommended_status']} "
          f"problems={len(verdict['problems'])}")
    return verdict["verdict"]


def main(mod) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="also write the Markdown reading of the verdict here")
    args = parser.parse_args()
    verdict = aggregate(mod, report_path=args.report)
    sys.exit({COMPLETED: 0, FALSIFIED: 1, NOT_SHOWN: 3}.get(verdict, 2))
