"""Pilot tuning of the arms' learning rates and of the balancer's step, on PILOT seeds only (1001 and up; never a declared seed). Appends one
JSON line per (seed, arm, learning rate, balancer step) to the output file, through the harness's own `run(..., only_arms=...)`, so the
streams, the plateau protocol and the matching are the pinned ones.

    python research/cseries/c003a/pilots/tune.py --seeds 1001,1002,1003 --out grid.jsonl --arms schema_free,control --lr 0.001,0.003,0.01
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--lr", default="0.003")
    parser.add_argument("--rate", default=None, help="balancer steps to try (heuristic arm only)")
    parser.add_argument("--entropy", default=None, help="entropy coefficients (milli) to try (every arm except the control and the heuristic balancer: the fix round gives each arm nine configurations)")
    parser.add_argument("--law", default=None, help="control-law forms to try (full, diagonal, scalar; the control arm only)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override any other pinned value for every configuration (recorded in the row)")
    args = parser.parse_args()
    harness._load_torch()
    cfg0 = harness.load_config()
    seeds = [int(s) for s in args.seeds.split(",")]
    if min(seeds) < 1001 or set(seeds) & set(cfg0["seeds"]):
        print("pilots use seeds from 1001 up and never a declared seed", file=sys.stderr)
        return 2
    for seed in seeds:
        for arm in args.arms.split(","):
            for lr in [float(v) for v in args.lr.split(",")]:
                entropies = [int(v) for v in args.entropy.split(",")] if (args.entropy and arm not in ("control", "heuristic")) else [cfg0["arm_entropy_milli"][arm]]
                for rate, law, entropy in itertools.product([float(v) for v in args.rate.split(",")] if (args.rate and arm == "heuristic") else [cfg0["heuristic_rate"]],
                                                            args.law.split(",") if (args.law and arm == "control") else [cfg0["law_form"]], entropies):
                    overrides = {"arm_lr": {**cfg0["arm_lr"], arm: lr}, "heuristic_rate": rate, "law_form": law, "arm_entropy_milli": {**cfg0["arm_entropy_milli"], arm: entropy}}
                    if args.max_steps:
                        overrides["max_steps"] = args.max_steps
                    extra = {k: json.loads(v) for k, _, v in (item.partition("=") for item in args.set)}
                    overrides.update(extra)
                    t0 = time.time()
                    out = harness.run(seed, overrides, only_arms=(arm,))
                    rec = out["arms"][arm]
                    row = {"seed": seed, "arm": arm, "lr": lr, "rate": rate, "law": law, "entropy": entropy, "set": extra, "reward": round(1000 * rec["reward_correct"] / rec["reward_total"]), "steps": rec["steps"],
                           "converged": rec["converged"], "best_val": rec["best_val_permille"], "top1": rec["schema_top1_permille"], "gain": rec.get("schema_gain_permille"), "load_cv_milli": rec["load_cv_milli"],
                           "wall_s": int(time.time() - t0)}
                    with open(args.out, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
