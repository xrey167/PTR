"""Recompute the pilot tables of the C001 preregistration (section 6) from the raw outputs next to this file.

    python summarise.py

A rerun at `max_steps=20000` (rerun-cap20000/) replaces the run of the same (seed, D, arm, lr, d) that hit the 9000-step cap.
Standard library only."""

import glob
import json
import statistics
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("workspace", "all_to_all", "bw_all_to_all", "sparse_all_to_all", "no_broadcast", "competition_off", "capacity_off")
PAIRWISE = ("all_to_all", "bw_all_to_all", "sparse_all_to_all")


def load(pattern: str) -> list[dict]:
    return [json.loads(line) for path in sorted(glob.glob(str(HERE / pattern))) for line in open(path, encoding="utf-8") if line.strip()]


def key(row: dict) -> tuple:
    return (row["seed"], row["D"], row["arm"], row["lr"], row["d"])


def merged(pattern: str) -> dict:
    rows = {key(r): r for r in load(pattern)}
    rows.update({key(r): {**r, "rerun": True} for r in load("rerun-cap20000/*.jsonl") if key(r) in rows})
    return rows


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def sweep() -> None:
    rows = merged("sweep/*.jsonl")
    print("operating-point sweep (lr 0.003, d 32; 16-module accuracy per mille, seeds 1004 / 1005):")
    for size in (1024, 2048, 4096, 8192):
        line = {arm: [rows[(s, size, arm, 0.003, 32)] for s in (1004, 1005)] for arm in ("all_to_all", "bw_all_to_all", "sparse_all_to_all", "workspace")}
        text = "  ".join(f"{arm} {'/'.join(str(r['ood16']) + ('*' if not r['converged'] else '') for r in rs)}" for arm, rs in line.items())
        strongest = max(mean(r["ood16"] for r in line[arm]) for arm in PAIRWISE)
        print(f"  D {size}: {text}  strongest pairwise mean {strongest:.1f}")
    print("  (* = hit the step cap)")


def grid() -> dict:
    rows = merged("grid/*.jsonl")
    cells: dict[tuple, list] = {}
    for (seed, _, arm, lr, d), r in rows.items():
        cells.setdefault((arm, lr, d), []).append(r)
    for r in load("lr-edge*/*.jsonl"):  # the extensions: lr 0.006 and above, at d 48 only
        cells.setdefault((r["arm"], r["lr"], r["d"]), []).append(r)
    print("equal grid (D 2048, seeds 1004-1008; mean 16-module accuracy per mille; [n] = runs that hit the cap; '-' = not run):")
    for arm in ARMS:
        parts = []
        measured = [(mean(r["ood16"] for r in rs), lr, d) for (a, lr, d), rs in cells.items() if a == arm]
        best_any = max(measured)
        for lr, d in ((0.001, 32), (0.001, 48), (0.003, 32), (0.003, 48), (0.006, 48), (0.012, 48), (0.024, 48)):
            rs = cells.get((arm, lr, d))
            if rs is None:
                parts.append(f"lr {lr} d {d}: -")
                continue
            capped = sum(1 for r in rs if not r["converged"])
            parts.append(f"lr {lr} d {d}: {mean(r['ood16'] for r in rs):.1f}{'[' + str(capped) + ']' if capped else ''}{'*' if (lr, d) == best_any[1:] else ''}")
        print(f"  {arm:18s} " + "  ".join(parts))
    print("pinned cell per arm (every arm at d 48; the lr with the highest mean at d 48):")
    for arm in ARMS:
        lr = max((mean(r["ood16"] for r in cells[(arm, lr, 48)]), lr) for lr in (0.001, 0.003, 0.006, 0.012, 0.024) if (arm, lr, 48) in cells)[1]
        PINNED[arm] = lr
        print(f"  {arm:18s} lr {lr}")
    return cells


PINNED: dict[str, float] = {}


def paired(cells: dict) -> None:
    print("paired differences at the pinned cells (d 48, pinned lr): workspace minus arm, 16 modules, per mille:")
    ws = {r["seed"]: r["ood16"] for r in cells[("workspace", PINNED["workspace"], 48)]}
    for arm in ARMS[1:]:
        arm_cell = {r["seed"]: r["ood16"] for r in cells[(arm, PINNED[arm], 48)]}
        diffs = [ws[s] - arm_cell[s] for s in sorted(ws)]
        print(f"  {arm:18s} {diffs}  mean {mean(diffs):.1f}  sd {statistics.stdev(diffs):.1f}  median {statistics.median(diffs)}")


def calibration(cells: dict) -> None:
    previous = {key(r): r for r in load("calibration/*.jsonl")}
    current = {key(r): r for r in load("calibration-lr006/*.jsonl")}
    seeds = sorted({r["seed"] for r in previous.values()})
    for label, rows, lr in (("lr 0.003 (round 3; crippled at width cap 384)", previous, 0.003), ("lr 0.006 (round 4; crippled at 413)", current, 0.006)):
        if not rows:
            continue
        base = {r["seed"]: r["ood16"] for r in cells[("all_to_all", lr, 48)]}
        for arm in ("weak_all_to_all", "crippled_all_to_all"):
            diffs = [base[s] - rows[(s, 2048, arm, lr, 48)]["ood16"] for s in seeds if (s, 2048, arm, lr, 48) in rows]
            if diffs:
                print(f"sensitivity at {label}: all_to_all minus {arm}: {diffs} mean {mean(diffs):.1f} sd {statistics.stdev(diffs):.1f}")


def lr_edge(cells: dict) -> None:
    if not load("lr-edge*/*.jsonl"):
        return
    print("learning-rate ladder at d 48 (mean 16-module accuracy per mille over seeds 1004-1008; [n] = runs capped; 'nc' = runs not converged):")
    for arm in ARMS:
        parts = []
        for lr in (0.001, 0.003, 0.006, 0.012, 0.024):
            rs = cells.get((arm, lr, 48))
            if rs:
                parts.append(f"{lr}: {mean(r['ood16'] for r in rs):.1f} (steps max {max(r['steps'] for r in rs)}, nc {sum(1 for r in rs if not r['converged'])})")
        print(f"  {arm:18s} " + "   ".join(parts))


# The graded control before round 6 was a two-round all-to-all under the arm name weak_all_to_all; the final weak_all_to_all is the
# weight-decayed all-to-all (config arm_weight_decay). Rows of the old definition are keyed "weak_rounds2" here.
OLD = {"weak_all_to_all": "weak_rounds2"}
FINAL_WEAK_PATTERNS = ("calib6/final-weak-*.jsonl", "calib6/confirm-weak*.jsonl")
CALIB6_PATTERNS = ("calib6/final-weak-*.jsonl", "calib6/confirm-*.jsonl", "calib6/graded-*.jsonl", "calib6/cand-*.jsonl", "calib6/gross-*.jsonl", "calib6/contrast-*.jsonl")
TUNING_SEEDS = (1004, 1005, 1006, 1007, 1008)
DRY_SEEDS = (1009, 1010)


def seed_table() -> dict[int, dict[str, int]]:
    """16-module accuracy per mille, per pilot seed and arm, at the pinned cells (d 48, the arm's selected learning rate): the tuning
    seeds 1004-1008 (workspace from the grid at lr 0.003; all_to_all, bw_all_to_all and sparse_all_to_all from the lr 0.006 extension;
    weak and crippled all-to_all from the lr 0.006 calibration), the dry runs of the harness 1009 (final configuration) and 1010, and the
    fresh pilot seeds of round 6 (calib6/: every row written by pilot.py, keyed by arm name, the last row of an arm and seed wins)."""
    table: dict[int, dict[str, int]] = {}
    rows = merged("grid/*.jsonl")
    for seed in TUNING_SEEDS:
        table[seed] = {"workspace": rows[(seed, 2048, "workspace", 0.003, 48)]["ood16"]}
    for r in load("lr-edge/*.jsonl"):
        if r["arm"] in PAIRWISE and r["lr"] == 0.006 and r["d"] == 48:
            table[r["seed"]][r["arm"]] = r["ood16"]
    for r in load("calibration-lr006/*.jsonl"):
        table[r["seed"]][OLD.get(r["arm"], r["arm"])] = r["ood16"]
    for name, seed in (("dry-1009-final.json", 1009), ("dry-1010.json", 1010)):
        arms = json.loads((HERE / "dry" / name).read_text(encoding="utf-8"))["arms"]
        table[seed] = {OLD.get(arm, arm): 1000 * a["ood16_correct"] // a["ood16_total"] for arm, a in arms.items()}
    pinned_wd = round(tomllib.loads((HERE.parents[2] / "../experiments/consciousness/C001-global-workspace/config.toml").resolve().read_text(encoding="utf-8"))["pinned_thresholds"]["arm_weight_decay"]["weak_all_to_all"] * 1_000_000)
    for pattern in CALIB6_PATTERNS:  # the weight-decay probes (wd-*.jsonl) reuse the arm name all_to_all and are read separately
        for path in sorted(glob.glob(str(HERE / pattern))):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["seed"] < 1011 and pattern not in FINAL_WEAK_PATTERNS:
                    continue
                arm = r["arm"]
                if pattern == "calib6/graded-*.jsonl":  # the round-6 baseline of the two-round weak arm
                    arm = OLD.get(arm, arm)
                elif arm == "weak_all_to_all":
                    wd = r.get("wd_micro", 1000 if "wd001" in path else pinned_wd)  # the first candidate file predates the field
                    arm = "weak_all_to_all" if wd == pinned_wd else f"weak_wd{wd}"
                table.setdefault(r["seed"], {})[arm] = r["ood16"]
    return table


def id_table() -> dict[int, dict[str, int]]:
    """In-distribution accuracy per mille of the workspace (pinned cell, lr 0.003, d 48) and of no_comm, per pilot seed: the two numbers of the
    harness rule `rules.workspace_trained` (workspace ID >= no_comm ID + trained_margin). Sources: grid/ (workspace 1004-1008), dry/ (both, 1009-1010),
    calib6/contrast-workspace (workspace 1011-1020), calib6/nocomm-* (no_comm on the other seeds)."""
    out: dict[int, dict[str, int]] = {}
    rows = merged("grid/*.jsonl")
    for seed in TUNING_SEEDS:
        out.setdefault(seed, {})["workspace"] = rows[(seed, 2048, "workspace", 0.003, 48)]["id"]
    for name, seed in (("dry-1009-final.json", 1009), ("dry-1010.json", 1010)):
        arms = json.loads((HERE / "dry" / name).read_text(encoding="utf-8"))["arms"]
        out[seed] = {arm: 1000 * arms[arm]["id_correct"] // arms[arm]["id_total"] for arm in ("workspace", "no_comm")}
    for r in load("calib6/contrast-workspace.jsonl"):
        out.setdefault(r["seed"], {})["workspace"] = r["id"]
    for r in load("calib6/nocomm-*.jsonl"):
        out.setdefault(r["seed"], {})["no_comm"] = r["id"]
    return out


def weight_decay_probes() -> None:
    """The weight-decay probes of round 6 (calib6/wd-*.jsonl): the arm all_to_all (same initialisation as the baseline) trained with Adam
    weight decay; the deficit against the baseline all_to_all of the same seed."""
    table = seed_table()
    for path in sorted(glob.glob(str(HERE / "calib6/wd-*.jsonl"))):
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        diffs = [table[r["seed"]]["all_to_all"] - r["ood16"] for r in rows]
        sd = f" sd {statistics.stdev(diffs):5.1f}" if len(diffs) > 1 else ""
        print(f"  weight-decay probe {Path(path).stem[3:]:8s} n {len(diffs):2d} deficit mean {mean(diffs):6.1f}{sd}  {diffs}")


def calib6_report() -> None:
    table = seed_table()
    ids = id_table()
    print("harness rule for 'workspace trained' (in-distribution accuracy >= no_comm + 100 per mille), per pilot seed: workspace ID / no_comm ID / trained")
    for seed in sorted(ids):
        w, n = ids[seed].get("workspace"), ids[seed].get("no_comm")
        print(f"  {seed}: {w} / {n} / {int(w >= n + 100) if w is not None and n is not None else '?'}")
    fresh = sorted(s for s in table if s >= 1011)
    print("round 6: fresh pilot seeds 1011-1020 and later (16-module accuracy per mille at d 48, lr 0.006 unless the arm is the workspace):")
    arms = sorted({a for s in fresh for a in table[s]})
    for arm in arms:
        seeds = [s for s in fresh if arm in table[s]]
        print(f"  {arm:22s} n {len(seeds):2d}  " + " ".join(f"{table[s][arm]:4d}" for s in seeds))
    weight_decay_probes()
    base = [s for s in fresh if "all_to_all" in table[s]]
    for arm in arms:
        if arm == "all_to_all" or not arm.startswith(("weak", "crippled")):
            continue
        both = [s for s in base if arm in table[s]]
        if len(both) > 1:
            diffs = [table[s]["all_to_all"] - table[s][arm] for s in both]
            print(f"  all_to_all minus {arm:22s} n {len(both):2d} mean {mean(diffs):6.1f} sd {statistics.stdev(diffs):5.1f} median {statistics.median(diffs):5.1f}  " + str(diffs))
    calibration_seeds = [s for s in fresh if s <= 1020]
    if all("workspace" in table[s] for s in calibration_seeds):
        for label, seeds in (("all ten seeds", calibration_seeds), ("the nine seeds on which the workspace trained", [s for s in calibration_seeds if table[s]["workspace"] >= 450])):
            for arm in PAIRWISE:
                both = [s for s in seeds if arm in table[s]]
                if len(both) > 1:
                    diffs = [table[s]["workspace"] - table[s][arm] for s in both]
                    print(f"  workspace minus {arm:18s} {label}: n {len(both):2d} mean {mean(diffs):6.1f} sd {statistics.stdev(diffs):5.1f} median {statistics.median(diffs):5.1f}  " + str(diffs))


if __name__ == "__main__":
    sweep()
    cells = grid()
    paired(cells)
    calibration(cells)
    lr_edge(cells)
    calib6_report()
