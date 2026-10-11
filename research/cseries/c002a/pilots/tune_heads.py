"""Pilot tuning of BOTH heads together (fix round): the same learning rate, weight decay and widths for the self-model and the information-matched
probe, so neither gets a larger grid. Pilot seeds only (1001 and up). One JSON line per (seed, configuration): both heads' in-distribution and shifted
Brier and the model's own calibrated confidence. The configuration is chosen by the sum of the two heads' in-distribution Brier (never by the contrast,
never on the shifted split).

    python research/cseries/c002a/pilots/tune_heads.py --seeds 1001,1002,1003 --out grid.jsonl --lr 0.0003,0.001 --wd 0,100,1000 --widths 64:32,32:16
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
    parser.add_argument("--lr", default="0.001")
    parser.add_argument("--wd", default="0")
    parser.add_argument("--widths", default="64:32")
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
        for lr, wd, width in itertools.product([float(v) for v in args.lr.split(",")], [int(v) for v in args.wd.split(",")], args.widths.split(",")):
            h1, h2 = (int(v) for v in width.split(":"))
            overrides = {"probe_lr": lr, "probe_weight_decay_milli": wd, "probe_hidden": h1, "probe_hidden2": h2, **extra}
            t0 = time.time()
            out = harness.run(seed, overrides, only_arms=("self_model", "probe_matched"))
            row = {"seed": seed, "lr": lr, "wd": wd, "width": width, "set": extra, "wall_s": int(time.time() - t0)}
            for arm in ("self_model", "probe_matched"):
                rec = out["arms"][arm]
                row[arm] = {"steps": rec["steps"], "converged": rec["converged"], "id": round(1000 * rec["id_brier_correct"] / rec["id_brier_total"], 3),
                            "shift": round(1000 * rec["brier_correct"] / rec["brier_total"], 3), "aux_steps": rec["aux_steps"], "aux_grad": rec["aux_grad_norm_micro"]}
            own = out["own_confidence"]
            row["own"] = {"id": round(1000 * own["id_brier_correct"] / own["id_brier_total"], 3), "shift": round(1000 * own["brier_correct"] / own["brier_total"], 3)}
            with open(args.out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
