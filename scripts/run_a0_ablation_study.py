"""Drive the A0 mechanism ablation study, one explicit phase at a time.

The study's design, criteria and order of commits are in
research/falsification/A0-ablations-v1/PREREGISTRATION.md. This driver only
sequences what that document fixes:

  prefreeze    generate the splits and check them against splits.lock.json,
               compute the references and the G0 data bands, build the study
               binary, run its self-test and the A0 test suite, calibrate the full
               arm on seed 0 (train and val only), and apply the budget rule.
               Writes calibration.json and budget.json. Run from a clean commit.
  prereg       write the manifests' a0_* entrypoint keys (with S*, the arms and
               the data FNV as literals) and point them at the measured
               hardware profile. The preregistration commit follows by hand.
  sweep        one runner invocation per (experiment, learning rate), seed 17.
  select       pick each arm's learning rate from the sweep records.
  eval         one runner invocation per (experiment, seed), plus the rerun.
  contingency  the 4000-step reruns the learnability criterion calls for.
  aggregate    scripts/aggregate_a0_ablation.py, then scripts/report_a0_ablation.py
               (RESULTS.md and a FALSIFIED-<contrast>.md note per null or HARMFUL).

It never edits criteria.toml, and `eval` refuses to start unless the worktree
is clean and the only files changed since the preregistration tag are the
learning-rate selection and the sweep records (gate G4).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "research/falsification/A0-ablations-v1"
CONFIG = ROOT / "model/configs/a0_ablation_study.toml"
BENCHMARK = ROOT / "benchmarks/operator-routing"
LOCK = BENCHMARK / "splits.lock.json"
DATA = "datasets/generated/operator_routing_v1"
TARGET_DIR = "model/burn-a0/target-a0-study"
BINARY = ROOT / TARGET_DIR / "release/examples/a0_ablation"
PREREG_TAG = "a0-ablation-prereg-v1"
EXPERIMENTS = {
    "M001": "model/M001-semantic-slots",
    "M002": "model/M002-typed-attention",
    "M003": "model/M003-latent-recurrence",
    "M004": "model/M004-operator-router",
}
TOOLCHAIN = "+1.95.0"
BUILD = [
    "cargo", TOOLCHAIN, "build", "--release", "--locked", "--offline", "--quiet",
    "--target-dir", TARGET_DIR, "--manifest-path", "model/burn-a0/Cargo.toml",
    "--example", "a0_ablation",
]
RUN_PREFIX = (
    f"cargo {TOOLCHAIN} run --release --locked --offline --quiet --target-dir {TARGET_DIR} "
    "--manifest-path model/burn-a0/Cargo.toml --example a0_ablation --"
)


def config() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def data_fnv() -> str:
    return json.loads(LOCK.read_text(encoding="utf-8"))["data_fnv1a64"]


def say(message: str) -> None:
    print(f"[a0-study] {message}", flush=True)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    say("$ " + " ".join(command))
    return subprocess.run(command, cwd=ROOT, text=True, **kwargs)


def must(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = run(command, **kwargs)
    if result.returncode != 0:
        if kwargs.get("capture_output"):
            sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise SystemExit(f"failed ({result.returncode}): {' '.join(command)}")
    return result


def worktree_clean() -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout
    return not status.strip()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def arms_by_experiment(tiers: set[int]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {name: [] for name in EXPERIMENTS}
    for arm in config()["arm"]:
        if arm["tier"] in tiers:
            grouped[arm["experiment"]].append(arm["name"])
    return grouped


def rows(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


# --------------------------------------------------------------------------- prefreeze


def calibrate(d_model: int, steps: int, lr: float) -> dict:
    result = must(
        [str(BINARY), "--phase", "calibrate", "--seed", str(config()["seeds"]["calibration"]),
         "--steps", str(steps), "--lr", str(lr), "--d-model", str(d_model),
         "--data", DATA, "--data-fnv64", data_fnv()],
        capture_output=True,
    )
    parsed = rows(result.stdout)
    val = [row for row in parsed if row.get("row") == "val"]
    meta = next(row for row in parsed if row.get("row") == "meta")
    timing = re.search(r"ms_per_step=([0-9.]+)", result.stderr)
    return {
        "d_model": d_model,
        "steps": steps,
        "lr": lr,
        "final_val_accuracy": val[-1]["val_accuracy"],
        "val_curve": [row["val_accuracy"] for row in val],
        "final_train_loss": meta["final_train_loss"],
        "nan": meta["nan"],
        "ms_per_step": float(timing.group(1)) if timing else None,
    }


def plan_arms(ladder_applied: list[str]) -> dict[str, list[str]]:
    plan = arms_by_experiment({1, 2})
    if "drop latent-4" in ladder_applied:
        plan["M003"] = [a for a in plan["M003"] if a != "latent-4"]
    if "drop the blind-query pair" in ladder_applied:
        plan["M002"] = [a for a in plan["M002"] if not a.startswith("blind-query")]
    return plan


def projected_minutes(plan: dict[str, list[str]], steps: int, sweep_steps: int | None, ms_per_step: float) -> float:
    """W = [N_eval (S* t + overhead) + N_sweep (S_sweep t + overhead)] / workers,
    where N counts arm-runs: every arm at every declared seed plus the rerun, and
    every arm at every grid rate when the sweep runs."""
    rules = config()
    t = ms_per_step / 1000.0
    overhead = rules["budget"]["per_process_overhead_seconds"]
    arms = sum(len(v) for v in plan.values())
    n_eval = arms * len(rules["seeds"]["declared"]) + 1
    seconds = n_eval * (steps * t + overhead)
    if sweep_steps is not None:
        n_sweep = arms * len(rules["learning_rate"]["grid"])
        seconds += n_sweep * (sweep_steps * t + overhead)
    return seconds / rules["budget"]["workers"] / 60.0


def budget(steps: int, ms_per_step: float) -> dict:
    """The budget rule: starting from every arm and a full-length sweep, apply the
    ladder in order while the projection exceeds the limit, recomputing after
    each step."""
    rules = config()["budget"]
    applied: list[str] = []
    sweep_steps: int | None = steps
    for step in [None, *rules["ladder"]]:
        if step is not None:
            applied.append(step)
            if step == "halve the sweep's steps":
                sweep_steps = steps // 2
            elif step.startswith("skip the sweep"):
                sweep_steps = None
        if projected_minutes(plan_arms(applied), steps, sweep_steps, ms_per_step) <= rules["limit_minutes"]:
            break
    plan = plan_arms(applied)
    minutes = projected_minutes(plan, steps, sweep_steps, ms_per_step)
    return {
        "steps": steps,
        "ms_per_step_used": ms_per_step,
        "ladder_applied": applied,
        "sweep": sweep_steps is not None,
        "sweep_steps": sweep_steps,
        "arms": plan,
        "projected_minutes": minutes,
        "within_limit": minutes <= rules["limit_minutes"],
        "within_hard_cap": minutes <= rules["hard_cap_minutes"],
    }


def prefreeze(_args) -> None:
    if not worktree_clean():
        raise SystemExit("prefreeze runs from a clean commit")
    must([sys.executable, str(BENCHMARK / "generator.py")])
    must([sys.executable, str(BENCHMARK / "generator.py"), "--check"])
    must([sys.executable, str(BENCHMARK / "references.py"), "--out", str(STUDY_DIR / "references.json")])
    references = json.loads((STUDY_DIR / "references.json").read_text(encoding="utf-8"))
    if not references["all_bands_pass"]:
        failed = [band["id"] for band in references["bands"] if not band["pass"]]
        raise SystemExit(f"G0 data bands failed: {failed}; fix the generator before the freeze")
    must(BUILD)
    must([str(BINARY), "--phase", "self-test", "--data", DATA, "--data-fnv64", data_fnv()])
    tests = must(["cargo", TOOLCHAIN, "test", "--locked", "--manifest-path", "model/burn-a0/Cargo.toml"],
                 capture_output=True)
    (STUDY_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "logs/prefreeze-cargo-test.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")

    rules = config()
    target = rules["calibration"]["target_val_accuracy"]
    attempts = []
    chosen = None
    for rung, d_model in [("base", rules["model"]["d_model"]), ("R1", 48)]:
        for steps in rules["training"]["steps_candidates"]:
            attempt = calibrate(d_model, steps, rules["learning_rate"]["calibration"])
            attempt["rung"] = rung
            attempts.append(attempt)
            say(f"calibration {rung} d_model={d_model} steps={steps}: val {attempt['final_val_accuracy']:.4f}")
            if attempt["final_val_accuracy"] >= target:
                chosen = attempt
                break
        if chosen:
            break
    calibration = {
        "target_val_accuracy": target,
        "attempts": attempts,
        "chosen": chosen,
        "outcome": "calibrated" if chosen else "no rung reached the target; R2 needs a regenerated rule and is decided by hand",
    }
    write_json(STUDY_DIR / "calibration.json", calibration)
    if not chosen:
        raise SystemExit("calibration failed at base and R1; see calibration.json")
    worst = max(a["ms_per_step"] for a in attempts if a["ms_per_step"] is not None)
    plan = budget(chosen["steps"], worst)
    plan["d_model"] = chosen["d_model"]
    write_json(STUDY_DIR / "budget.json", plan)
    say(f"S* = {chosen['steps']}, d_model {chosen['d_model']}, projected {plan['projected_minutes']:.1f} min")
    if not plan["within_hard_cap"]:
        raise SystemExit("the budget exceeds the hard cap; redesign before the freeze")


# --------------------------------------------------------------------------- prereg


def entrypoints(experiment: str, arms: list[str], steps: int, d_model: int) -> dict[str, str]:
    common = f"--data {DATA} --data-fnv64 {data_fnv()} --d-model {d_model}"
    lr_file = "research/falsification/A0-ablations-v1/lr_selection.tsv"
    keys = {
        "a0_ablation_entrypoint": (
            f"{RUN_PREFIX} --phase eval --experiment {experiment} --arms {','.join(arms)} "
            f"--seed <seed> --steps {steps} --lr-file {lr_file} {common}"
        ),
        "a0_sweep_entrypoint": (
            f"{RUN_PREFIX} --phase sweep --experiment {experiment} --arms {','.join(arms)} "
            f"--seed <seed> --steps {{sweep_steps}} --lr <lr> {common}"
        ),
        "a0_contingency_entrypoint": (
            f"{RUN_PREFIX} --phase eval --experiment {experiment} --arms <arms> "
            f"--seed <seed> --steps 4000 --lr-file {lr_file} {common}"
        ),
    }
    if experiment == "M001":
        keys["a0_rerun_entrypoint"] = (
            f"{RUN_PREFIX} --phase eval --experiment M001 --arms full "
            f"--seed <seed> --steps {steps} --lr-file {lr_file} {common}"
        )
    return keys


def toml_string(value: str) -> str:
    return json.dumps(value)


def prereg(_args) -> None:
    plan = json.loads((STUDY_DIR / "budget.json").read_text(encoding="utf-8"))
    for experiment, path in EXPERIMENTS.items():
        manifest = ROOT / "experiments" / path / "experiment.toml"
        text = manifest.read_text(encoding="utf-8")
        if "a0_ablation_entrypoint" in text:
            raise SystemExit(f"{manifest} already carries a0 entrypoints")
        keys = entrypoints(experiment, plan["arms"][experiment], plan["steps"], plan["d_model"])
        sweep_steps = plan["sweep_steps"] or plan["steps"]
        keys = {k: v.replace("{sweep_steps}", str(sweep_steps)) for k, v in keys.items()}
        text = text.replace('hardware_profile = "hardware/default.toml"',
                            'hardware_profile = "hardware/a0-cpu-4core.toml"')
        lines = [text.rstrip("\n"), "",
                 "# The A0 mechanism ablation study (research/falsification/A0-ablations-v1).",
                 "# A0-internal; not this manifest's baseline comparison, whose status stays planned."]
        lines += [f"{key} = {toml_string(value)}" for key, value in keys.items()]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tomllib.loads(manifest.read_text(encoding="utf-8"))
        say(f"wrote a0 entrypoints into {manifest.relative_to(ROOT)}")


# --------------------------------------------------------------------------- sweep / select


def runner(experiment: str, entrypoint: str, seed: int, params: dict[str, str]) -> int:
    command = [sys.executable, "scripts/run_experiment.py", "run", experiment,
               "--entrypoint", entrypoint, "--seed", str(seed)]
    for key, value in params.items():
        command += ["--set", f"{key}={value}"]
    return run(command, capture_output=True).returncode


def parallel(jobs: list[tuple[str, str, int, dict]], workers: int) -> list[int]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: runner(*job), jobs))


def require_frozen() -> None:
    tag = subprocess.run(["git", "rev-parse", "--verify", "--quiet", PREREG_TAG],
                         cwd=ROOT, text=True, capture_output=True)
    if tag.returncode != 0:
        raise SystemExit(f"no {PREREG_TAG} tag: commit and tag the preregistration first")
    if not worktree_clean():
        raise SystemExit("the worktree must be clean")


def sweep(_args) -> None:
    require_frozen()
    plan = json.loads((STUDY_DIR / "budget.json").read_text(encoding="utf-8"))
    if not plan["sweep"]:
        say("the budget rule skipped the sweep")
        return
    jobs = [(experiment, "a0_sweep_entrypoint", 17, {"lr": str(lr)})
            for experiment in EXPERIMENTS
            for lr in config()["learning_rate"]["grid"]]
    codes = parallel(jobs, config()["budget"]["workers"])
    if any(codes):
        raise SystemExit(f"sweep processes failed: {codes}")


def records(experiment: str, entrypoint: str) -> list[dict]:
    results = ROOT / "experiments" / EXPERIMENTS[experiment] / "results"
    out = []
    for path in sorted(results.glob("run-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("entrypoint") == entrypoint:
            record["_path"] = str(path.relative_to(ROOT))
            out.append(record)
    return out


def choose_lr(arm: str, by_lr: dict[float, dict], grid: list[float], tolerance: float) -> tuple[float, str, dict[float, float]]:
    """The preregistered selection rule for one arm: among the learning rates that
    did not diverge, the smallest whose validation accuracy is within `tolerance`
    of the best; flagged when it sits on the edge of the grid. Returns the rate,
    the flag and every eligible rate's validation accuracy."""
    missing = [lr for lr in grid if lr not in by_lr]
    if missing:
        raise SystemExit(f"{arm}: no sweep result for lr {missing}")
    eligible = {lr: r["val_accuracy"] for lr, r in by_lr.items()
                if not r["nan"] and math.isfinite(r["val_accuracy"])}
    if not eligible:
        raise SystemExit(f"{arm}: every learning rate diverged")
    best = max(eligible.values())
    lr = min(lr for lr, acc in eligible.items() if acc >= best - tolerance)
    flag = "edge of grid" if lr in (min(grid), max(grid)) else ""
    return lr, flag, eligible


def select(_args) -> None:
    grid = config()["learning_rate"]["grid"]
    tolerance = config()["learning_rate"]["selection_tolerance"]
    plan = json.loads((STUDY_DIR / "budget.json").read_text(encoding="utf-8"))
    table = []
    for experiment in EXPERIMENTS:
        results: dict[str, dict[float, dict]] = {}
        for record in records(experiment, "a0_sweep_entrypoint"):
            if record.get("status") != "completed":
                continue
            for row in rows(record["stdout"]):
                if row.get("row") == "sweep":
                    results.setdefault(row["arm"], {})[row["lr"]] = row
        for arm in plan["arms"][experiment]:
            if not plan["sweep"]:
                table.append({"arm": arm, "lr": config()["learning_rate"]["calibration"],
                              "flag": "no per-arm lr selection", "val_accuracy": None})
                continue
            lr, flag, eligible = choose_lr(arm, results.get(arm, {}), grid, tolerance)
            table.append({"arm": arm, "lr": lr, "flag": flag, "val_accuracy": eligible[lr],
                          "by_lr": {str(k): v for k, v in sorted(eligible.items())}})
    lines = ["arm\tlr\tval_accuracy\tflag"]
    lines += [f"{row['arm']}\t{row['lr']}\t{row['val_accuracy']}\t{row['flag']}" for row in table]
    (STUDY_DIR / "lr_selection.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(STUDY_DIR / "lr_selection.json", table)
    say(f"selected learning rates for {len(table)} arms")


# --------------------------------------------------------------------------- eval / contingency


def aggregator():
    """The aggregator module, whose G4 diff rule the eval phase applies before it
    starts, so that the check here and the gate in the aggregate are one rule."""
    spec = importlib.util.spec_from_file_location("aggregate_a0_ablation", ROOT / "scripts/aggregate_a0_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_eval_commit() -> None:
    require_frozen()
    changed = subprocess.run(["git", "diff", "--name-only", f"{PREREG_TAG}..HEAD"],
                             cwd=ROOT, text=True, capture_output=True, check=True).stdout.split()

    def entrypoint_of(path: str) -> str | None:
        try:
            return json.loads((ROOT / path).read_text(encoding="utf-8")).get("entrypoint")
        except (OSError, json.JSONDecodeError):
            return None
    stray = aggregator().freeze_violations(changed, entrypoint_of)
    if stray:
        raise SystemExit(f"G4: files changed since {PREREG_TAG} beyond the lr selection: {stray}")
    if not (STUDY_DIR / "lr_selection.tsv").exists():
        raise SystemExit("no lr_selection.tsv: run select and commit it first")


def correctness() -> dict:
    """Gate G6 at the evaluation commit: the A0 tests (T1-T6), the binary's
    self-test (T7, T8), and the benchmark, aggregator and config test suites.
    Each log is kept under logs/ for the results commit."""
    logs = STUDY_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    checks = {
        "cargo_test": ["cargo", TOOLCHAIN, "test", "--locked", "--manifest-path", "model/burn-a0/Cargo.toml"],
        "self_test": [str(BINARY), "--phase", "self-test", "--data", DATA, "--data-fnv64", data_fnv()],
        "benchmark_tests": [sys.executable, "-m", "unittest", "discover", "-s", "benchmarks/operator-routing/tests"],
        "aggregator_tests": [sys.executable, "-m", "unittest", "scripts/tests/test_a0_ablation_aggregate.py"],
        "config_tests": [sys.executable, "-m", "unittest", "scripts/tests/test_a0_ablation_config.py"],
    }
    outcome = {}
    for name, command in checks.items():
        result = run(command, capture_output=True)
        (logs / f"g6-{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        outcome[name] = "pass" if result.returncode == 0 else "fail"
    write_json(logs / "g6.json", outcome)
    return outcome


def eval_phase(_args) -> None:
    require_eval_commit()
    must(BUILD)
    outcome = correctness()
    if any(value != "pass" for value in outcome.values()):
        raise SystemExit(f"G6 failed before evaluation: {outcome}")
    plan = json.loads((STUDY_DIR / "budget.json").read_text(encoding="utf-8"))
    seeds = config()["seeds"]["declared"]
    # Longest first: the experiments with the most arms.
    order = sorted(EXPERIMENTS, key=lambda e: -len(plan["arms"][e]))
    jobs = [(experiment, "a0_ablation_entrypoint", seed, {}) for experiment in order for seed in seeds]
    jobs.append(("M001", "a0_rerun_entrypoint", 17, {}))
    codes = parallel(jobs, config()["budget"]["workers"])
    failed = [job for job, code in zip(jobs, codes) if code != 0]
    if failed:
        say(f"retrying once (G3): {failed}")
        retry = parallel(failed, config()["budget"]["workers"])
        if any(retry):
            raise SystemExit(f"evaluation failed twice: {failed}")


def contingency(args) -> None:
    require_eval_commit()
    seeds = config()["seeds"]["declared"]
    arm = args.arm
    experiment = next(a["experiment"] for a in config()["arm"] if a["name"] == arm)
    arms = f"full,{arm}" if experiment == "M001" else arm
    jobs = [(experiment, "a0_contingency_entrypoint", seed, {"arms": arms}) for seed in seeds]
    if experiment != "M001":
        jobs += [("M001", "a0_contingency_entrypoint", seed, {"arms": "full"}) for seed in seeds]
    codes = parallel(jobs, config()["budget"]["workers"])
    if any(codes):
        raise SystemExit(f"contingency runs failed: {codes}")


def aggregate(_args) -> None:
    must([sys.executable, "scripts/aggregate_a0_ablation.py"])
    must([sys.executable, "scripts/report_a0_ablation.py"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ["prefreeze", "prereg", "sweep", "select", "eval", "aggregate"]:
        sub.add_parser(name)
    cont = sub.add_parser("contingency")
    cont.add_argument("arm")
    args = parser.parse_args(argv)
    {
        "prefreeze": prefreeze,
        "prereg": prereg,
        "sweep": sweep,
        "select": select,
        "eval": eval_phase,
        "contingency": contingency,
        "aggregate": aggregate,
    }[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
