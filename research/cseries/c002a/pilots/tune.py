"""Pilot tuning of the heads' learning rate and of the self-model's auxiliary weight, on PILOT seeds only (1001 and up; never a declared seed).
Appends one JSON line per (seed, head, learning rate, auxiliary weight) to the output file, through the harness's own `run(..., only_arms=...)`, so the
data, the classifier, the plateau protocol and the matching are the pinned ones.

    python research/cseries/c002a/pilots/tune.py --seeds 1001,1002,1003 --out grid.jsonl --arms probe_matched --lr 0.0003,0.001,0.003
    python research/cseries/c002a/pilots/tune.py --seeds 1001,1002,1003 --out grid.jsonl --arms self_model --lr 0.0003,0.001,0.003 --aux 300,1000,3000
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
    parser.add_argument("--lr", default="0.001")
    parser.add_argument("--aux", default=None, help="auxiliary weights (per mille) to try (the self-model only)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args()
    harness._load_torch()
    cfg0 = harness.load_config()
    seeds = [int(s) for s in args.seeds.split(",")]
    if min(seeds) < 1001 or set(seeds) & set(cfg0["seeds"]):
        print("pilots use seeds from 1001 up and never a declared seed", file=sys.stderr)
        return 2
    extra = {k: json.loads(v) for k, _, v in (item.partition("=") for item in args.set)}
    for seed in seeds:
        for arm in args.arms.split(","):
            auxes = [int(v) for v in args.aux.split(",")] if (args.aux and arm.startswith("self_model")) else [cfg0["aux_weight_milli"]]
            for lr, aux in itertools.product([float(v) for v in args.lr.split(",")], auxes):
                overrides = {"probe_lr": lr, "aux_weight_milli": aux, **extra}
                t0 = time.time()
                out = harness.run(seed, overrides, only_arms=(arm,))
                rec = out["arms"][arm]
                row = {"seed": seed, "arm": arm, "lr": lr, "aux": aux, "set": extra, "steps": rec["steps"], "converged": rec["converged"], "best_val": rec["best_val_permille"],
                       "brier_shift_milli": round(1000 * rec["brier_correct"] / rec["brier_total"], 3), "brier_id_milli": round(1000 * rec["id_brier_correct"] / rec["id_brier_total"], 3),
                       "auroc": rec["auroc_permille"], "ece": rec["ece_permille"], "wall_s": int(time.time() - t0)}
                with open(args.out, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
