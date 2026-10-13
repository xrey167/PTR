"""Pilot tuning of the arms' learning rates and of the feed-forward comparator's depth and positional slope, on PILOT seeds only
(1001 and up; never a declared seed). Appends one JSON line per configuration to the output file.

    python research/cseries/c004/pilots/tune.py --seed 1002 --out screen.jsonl --arm feedforward --layers 3,4,6 --scale 1,16 --lr 0.001,0.003
    python research/cseries/c004/pilots/tune.py --seed 1002 --out screen.jsonl --arm recurrent --lr 0.001,0.003,0.01

Each configuration trains just the named arm (and, with --null, its null-world twin) through the harness's own `run(..., only_arms=...)`, so
the data, the schedule, the plateau protocol and the matching are the pinned ones; only the named values are overridden."""

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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=("feedforward", "feedforward_inlength", "recurrent"), required=True)
    parser.add_argument("--layers", default="4")
    parser.add_argument("--scale", default="1")
    parser.add_argument("--lr", default="0.003")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override any other pinned value for every configuration (recorded in the row)")
    parser.add_argument("--null", action="store_true", help="also train the null-world twin of the arm")
    args = parser.parse_args()
    if args.seed < 1001:
        print("pilots use seeds from 1001 up", file=sys.stderr)
        return 2
    harness._load_torch()
    cfg0 = harness.load_config()
    if args.seed in cfg0["seeds"]:
        print("a declared seed is never a pilot seed", file=sys.stderr)
        return 2
    arms = (args.arm, args.arm + "_null") if args.null else (args.arm,)
    for layers, scale, lr in itertools.product([int(v) for v in args.layers.split(",")], [float(v) for v in args.scale.split(",")], [float(v) for v in args.lr.split(",")]):
        overrides = {"ff_layers": layers, "ff_slope_scale": scale, "arm_lr": {**cfg0["arm_lr"], **{a: lr for a in arms}}}
        if args.arm == "recurrent":
            overrides.pop("ff_layers")
            overrides.pop("ff_slope_scale")
        if args.max_steps:
            overrides["max_steps"] = args.max_steps
        extra = {k: json.loads(v) for k, _, v in (item.partition("=") for item in args.set)}
        overrides.update(extra)
        t0 = time.time()
        out = harness.run(args.seed, overrides, only_arms=arms)
        row = {"seed": args.seed, "arm": args.arm, "layers": layers, "slope_scale": scale, "lr": lr, "max_steps": overrides.get("max_steps", cfg0["max_steps"]), "set": extra,
               "width": out["feedforward_shape"], "wall_s": int(time.time() - t0), "cpu_s": int(time.process_time())}
        for name in arms:
            rec = out["arms"][name]
            row[name] = {"steps": rec["steps"], "converged": rec["converged"], "best_val": rec["best_val_permille"],
                         **{k[: -len("_correct")]: round(1000 * rec[k] / rec[k[: -len("_correct")] + "_total"]) for k in rec if k.endswith("_correct")}}
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
