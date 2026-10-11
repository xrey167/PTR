"""C001 pilot runner (development only; never evidence, never a declared seed).

    python pilot.py OUT.jsonl [key=json ...] seed:D:arm:lr:d [seed:D:arm:lr:d ...]

Trains each job with the harness's own `train_arm` on pilot seeds only (>= 1001; the declared seeds 4001-4020 are refused), at update
width `d` for every arm (the parameter-matched `ffn` is searched against the workspace at that width, as in the pinned run), and appends
one JSON line per job: the accuracies on 8, 16 and 24 modules (per mille), whether the run stopped by patience, its step count and its
best validation accuracy. `key=json` overrides a pinned value (e.g. `max_steps=9000`); `spec:NAME=json` registers an ad-hoc calibration arm NAME
(an `ARM_SPECS` entry, used by the round-6 calibration of the graded control) and `armd:NAME=int` gives it its own update width.

The raw outputs next to this file are the ones the preregistration (section 6) cites:

    sweep/            operating-point sweep, seeds 1004/1005, D in {1024, 2048, 4096, 8192}, lr 0.003, d 32, max_steps=9000
    grid/             equal grid, seeds 1004-1008, D 2048, lr {0.001, 0.003} x d {32, 48}, all arms, max_steps=9000
    calibration/      weak and crippled all-to-all, seeds 1004-1008, d 48, lr {0.003, 0.001} (crippled at the old width cap 384)
    rerun-cap20000/   the eight grid runs and the one sweep run that hit the 9000-step cap, rerun with max_steps=20000
    lr-edge/          lr 0.006 at d 48 for the seven grid arms, seeds 1004-1008, max_steps=20000 (the upper edge of the grid)
    lr-edge2/         lr 0.012 at d 48 for the four arms that selected 0.006 (all_to_all, bw, sparse, competition_off), seeds 1004-1008, max_steps=20000
    calib6/           round 6, fresh pilot seeds 1011-1030, harness-identical evaluation sets (check-1009*: the check against dry-1009-final):
                      graded-baseline-*  all_to_all and the old two-round weak arm, 1011-1020 (the two-round arm is keyed weak_rounds2 in summarise.py)
                      cand-weak_*        candidate single-knob weakenings (value width dv, top-k keys k, update width d, lr001, weight decay wd<micro>)
                      wd-*               weight-decay probes on all_to_all's initialisation (arm name all_to_all; read separately)
                      confirm-a2a, confirm-weak   all_to_all and the final weak_all_to_all (weight decay 0.001) on 1021-1030
                      final-weak-tuning  the final weak_all_to_all on 1004-1010; gross-crippled  crippled_all_to_all on 1011-1020
                      nocomm-*           no_comm on 1004-1008 and 1011-1020 (round 8: the harness rule for 'workspace trained' needs the no_comm floor)
                      contrast-*         workspace, bw_all_to_all, sparse_all_to_all on 1011-1020 (read after the weak arm was chosen)
                      run_candidates.sh, run_lr.sh, lane.sh   the launch scripts
    cause1011/        round 7: bounded cause check of the workspace's training failure on seed 1011 (cause1011.py: exact re-run and three other initialisations)
    dry/              full pinned dry runs of the harness (dry-1031-final.*: the final code through the pinned entrypoint, round 8; dry-1009.json: seed 1009 at the round-3 configuration; dry-1010.json and dry-1009-final.json: the final configuration)

Runs that stopped by patience are bit-identical under any larger `max_steps` (the cap only matters for runs that reach it).
"""

import json
import sys
import time
from pathlib import Path

GW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GW))
import harness as H  # noqa: E402

H._load_torch()
torch = H.torch
torch.set_num_threads(1)


def main(argv: list[str]) -> None:
    out, extra, jobs = argv[0], {}, []
    for arg in argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            extra[key] = json.loads(value)
        else:
            jobs.append(arg)
    for job in jobs:
        seed, size, arm, lr, d = job.split(":")
        seed, size, lr, d = int(seed), int(size), float(lr), int(d)
        cfg = H.load_config()
        if seed < 1001 or seed in cfg["seeds"]:
            raise SystemExit(f"seed {seed} is not a pilot seed")
        specs = {k[5:]: v for k, v in extra.items() if k.startswith("spec:")}
        widths = {k[5:]: v for k, v in extra.items() if k.startswith("armd:")}
        cfg.update({k: v for k, v in extra.items() if not k.startswith(("spec:", "armd:"))})
        for name, spec in specs.items():  # ad-hoc calibration arms: spec:NAME={"scheme": "all_to_all", "rounds": 2}
            H.gwmodel.ARM_SPECS[name] = spec
        cfg["train_size"] = size
        cfg["arm_d"] = {k: d for k in list(cfg["arm_d"]) + list(specs)}
        cfg["arm_d"].update(widths)  # armd:NAME=24 runs the ad-hoc arm NAME at update width 24 (the workspace target stays at d)
        n = cfg["n_train_modules"]
        train = H.make_batch(torch.Generator().manual_seed(seed * 1000 + 17), size, n, cfg)
        # Evaluation sets are drawn like the harness (one generator per set, stream offsets 500 + 0..3). Pilot rows written before
        # round 6 (sweep/ grid/ calibration/ calibration-lr006/ rerun-cap20000/ lr-edge/ lr-edge2/) drew id, ood16 and ood24 from ONE
        # generator, so their ood16/ood24 counts use a different (equally sized) evaluation sample from the harness; training and the
        # validation curve are identical either way.
        gens = [torch.Generator().manual_seed(seed * 1000 + cfg["eval_stream_offset"] + i) for i in range(4)]
        id_batch, ood16, ood24 = (H.make_batch(g, cfg["eval_size"], m, cfg) for g, m in zip(gens, (n, 16, 24)))
        val = H.make_batch(gens[3], cfg["curve_eval_size"], n, cfg)
        target = H.gwmodel.count_params(H.gwmodel.build("workspace", cfg))
        ffn = cfg["ffn_workspace"] if arm == "workspace" else H.gwmodel.match_width(arm, cfg, target)
        started = time.time()
        net, info, _, _ = H.train_arm(arm, cfg, seed, train, val, ffn, lr, set())
        row = {
            "seed": seed, "D": size, "arm": arm, "lr": lr, "d": d, "ffn": ffn, "wd_micro": round(H.gwmodel.arm_weight_decay(arm, cfg) * 1_000_000),
            "id": H.permille(*H.evaluate(net, id_batch)), "ood16": H.permille(*H.evaluate(net, ood16)),
            "ood24": H.permille(*H.evaluate(net, ood24)), "converged": int(info["converged"]), "steps": info["steps"],
            "best_val": info["best_val"], "loss_w": info["loss_window_micro"], "sec": int(time.time() - started),
        }
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
