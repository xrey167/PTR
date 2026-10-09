"""C001 harness (v2): global workspace versus parameter-matched ablations, one seed.

    python research/consciousness/gw/harness.py --benchmark c001 --seed 3001

Prints one JSON object on its last line (the result the aggregator judges).
Exit 0: every coverage rule and hard counter holds. Exit 1: one of them failed (a defect of the product
arm, the task or the harness: structural, never a statistical comparison). Exit 2: harness integrity (the
environment is incomplete, an exception, a negative control that did not trip, bad arguments).
The comparison of the arms across seeds, and the validity guards, are the aggregator's.

Data regime: a FINITE training set of D examples of N_train modules (cycled), and evaluation on N_train
(in distribution) and on more modules (16 and 24: out of distribution, with more distractors). All arms see
the same data stream. Observed events, not configuration, feed the counters (the module counts and the
example hashes of what training actually used).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
import tomllib
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG_PATH = ROOT / "experiments/consciousness/C001-global-workspace/config.toml"
BENCHMARK = "c001"
EXPERIMENT = "C001"
PINNED_TORCH = "2.7.1"
PINNED_PYTHON = (3, 11)
IGNITION_ALPHAS = tuple(i / 10 for i in range(11))
OOD_MODULES = (16, 24)


def environment_problems() -> list[str]:
    """What is missing from the environment, before anything trains. The harness refuses (exit 2) rather than
    crash with Python's exit 1, which the runner would read as a falsification."""
    problems = []
    if sys.version_info[:2] != PINNED_PYTHON:
        problems.append(f"python {sys.version_info[0]}.{sys.version_info[1]}, pinned {PINNED_PYTHON[0]}.{PINNED_PYTHON[1]}")
    try:
        import torch  # noqa: F401
    except ImportError as error:
        problems.append(f"torch cannot be imported: {error}")
    else:
        if not torch.__version__.startswith(PINNED_TORCH):
            problems.append(f"torch {torch.__version__}, pinned {PINNED_TORCH}")
    if not CONFIG_PATH.is_file():
        problems.append(f"{CONFIG_PATH} is missing")
    return problems


def _load_torch():
    global torch, gwmodel, gwtask
    import torch as _torch

    torch = _torch
    sys.path.insert(0, str(HERE))
    import model as _model
    import task as _task

    gwmodel, gwtask = _model, _task


sys.path.insert(0, str(HERE))
from rules import HARD_COUNTERS, coverage_rules, falsifier_rules, guard_rules, holds, judge  # noqa: E402,F401


def load_config(path: Path = CONFIG_PATH) -> dict:
    table = tomllib.loads(path.read_text(encoding="utf-8"))["pinned_thresholds"]
    cfg = dict(table)
    cfg["d_in"] = 2 * cfg["n_types"] + cfg["n_values"] + 1
    return cfg


def arm_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"C001:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def permille(num: int, den: int) -> int:
    return (1000 * num) // den


# -- structural checks (each is also run on a doctored input as a negative control) ------------------

def n_leak_count(observed_train_n: set, ood_n: tuple) -> int:
    """Out-of-distribution module counts that training batches actually used."""
    return len(set(observed_train_n) & set(ood_n))


def overlap_count(train_hashes: set, eval_x) -> int:
    """Evaluation instances identical to a training instance (by hash of the first values)."""
    return sum(1 for h in row_hashes(eval_x) if h in train_hashes)


def row_hashes(x) -> list:
    # No numpy: the locked environment (uv.lock) holds torch only.
    flat = x.reshape(x.shape[0], -1)[:, :24].contiguous()
    return [hashlib.sha256(struct.pack(f"{len(row)}f", *row.tolist())).digest()[:8] for row in flat.unbind(0)]


def param_mismatch_arms(counts: dict, reference: str, tolerance_permille: int) -> int:
    ref = counts[reference]
    return sum(1 for arm, n in counts.items() if arm != reference and 1000 * abs(n - ref) > tolerance_permille * ref)


def bottleneck_violations(trace: dict, limit: int) -> int:
    """(instance, round, slot) triples whose write attention is nonzero on more than `limit` source modules."""
    return sum(int(((attn > 0).sum(dim=-1) > limit).sum()) for name, attn in trace.items() if name.startswith("write"))


def _states(net, x, alive, source=None, scale=0.0):
    with torch.no_grad():
        _, base = net(x, alive, states=True)
        if source is None:
            return base, None
        bumped = x.clone()
        bumped[:, source] = bumped[:, source] + scale
        _, moved = net(bumped, alive, states=True)
    return base, moved


def leak_count(net, x, scale: float, targets: str) -> int:
    """Number of (source, destination) module pairs, destinations "sensory" (not the source) or "reporter", whose
    final state moved by more than 1e-6 when the source's input was perturbed."""
    n = x.shape[1]
    alive = torch.ones(x.shape[0], n + 1, dtype=torch.bool)
    moved_total = 0
    for j in range(n):
        base, moved = _states(net, x, alive, j, scale)
        delta = (moved - base).abs().amax(dim=(0, 2))
        ids = [i for i in range(n) if i != j] if targets == "sensory" else [n]
        moved_total += sum(1 for i in ids if float(delta[i]) > 1e-6)
    return moved_total


def dropped_leaks(net, x, scale: float) -> int:
    """Perturbing a dropped sensory module's input must change nothing."""
    n = x.shape[1]
    leaks = 0
    for j in range(n):
        alive = torch.ones(x.shape[0], n + 1, dtype=torch.bool)
        alive[:, j] = False
        with torch.no_grad():
            a = net(x, alive)
            bumped = x.clone()
            bumped[:, j] = bumped[:, j] + scale
            b = net(bumped, alive)
        if float((a - b).abs().max()) > 1e-6:
            leaks += 1
    return leaks


def dropout_probe_is_sensitive(net, x, scale: float) -> bool:
    """The dropped-module probe must see a change when the perturbed module is NOT masked."""
    n = x.shape[1]
    alive = torch.ones(x.shape[0], n + 1, dtype=torch.bool)
    with torch.no_grad():
        a = net(x, alive)
        bumped = x.clone()
        bumped[:, 0] = bumped[:, 0] + scale
        b = net(bumped, alive)
    return float((a - b).abs().max()) > 1e-6


# -- training and evaluation ------------------------------------------------------------------

def make_batch(gen, n, n_modules, cfg):
    return gwtask.sample(gen, n, n_modules, cfg["n_types"], cfg["n_values"], cfg["noise"])


def evaluate(net, batch, alive=None):
    with torch.no_grad():
        pred = net(batch["x"], alive).argmax(dim=-1)
    return int((pred == batch["y"]).sum()), int(batch["y"].numel())


def train_arm(name, cfg, seed, train_data, curve_batch, steps, ffn, lr, observed_n):
    torch.manual_seed(arm_seed(seed, name))
    net = gwmodel.build(name, cfg, ffn=ffn)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    perm_gen = torch.Generator().manual_seed(seed * 1000 + cfg["train_stream_offset"] + 1)
    size = train_data["y"].shape[0]
    order, cursor = torch.randperm(size, generator=perm_gen), 0
    curve, counts, last_loss, finite = [], [], 0.0, True
    for step in range(1, steps + 1):
        if cursor + cfg["batch_size"] > size:
            order, cursor = torch.randperm(size, generator=perm_gen), 0
        idx = order[cursor : cursor + cfg["batch_size"]]
        cursor += cfg["batch_size"]
        bx, by = train_data["x"][idx], train_data["y"][idx]
        observed_n.add(int(bx.shape[1]))
        loss = torch.nn.functional.cross_entropy(net(bx), by)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step()
        last_loss = float(loss)
        if not math.isfinite(last_loss):
            finite = False
        if step % cfg["eval_every"] == 0:
            c, t = evaluate(net, curve_batch)
            curve.append(permille(c, t))
            counts.append((c, t))
    return net, (curve, counts), last_loss, finite


def first_steps_loss(name, cfg, seed, train_data, ffn, steps=20) -> float:
    torch.manual_seed(arm_seed(seed, "determinism"))
    net = gwmodel.build(name, cfg, ffn=ffn)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["arm_lr"][name])
    loss = None
    for k in range(steps):
        idx = torch.arange(k * cfg["batch_size"], (k + 1) * cfg["batch_size"]) % train_data["y"].shape[0]
        loss = torch.nn.functional.cross_entropy(net(train_data["x"][idx]), train_data["y"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def dropout_eval(net, batch, n_modules, gen) -> dict:
    """Accuracy with one module dropped per instance: a module that is neither type-q nor the cue (it should not
    matter), and a type-q module."""
    n = batch["x"].shape[0]
    rows = torch.arange(n)
    irrelevant = batch["match"].clone()
    irrelevant[rows, batch["cue"]] = True
    drop_irrelevant = torch.rand(n, n_modules, generator=gen).masked_fill(irrelevant, -1.0).argmax(dim=1)
    alive = torch.ones(n, n_modules + 1, dtype=torch.bool)
    alive[rows, drop_irrelevant] = False
    out = {}
    out["drop_irrelevant_correct"], out["drop_irrelevant_total"] = evaluate(net, batch, alive)
    alive = torch.ones(n, n_modules + 1, dtype=torch.bool)
    alive[rows, batch["pi"]] = False
    out["drop_relevant_correct"], out["drop_relevant_total"] = evaluate(net, batch, alive)
    return out


def ignition_curve(net, batch, cfg, n_modules) -> list:
    """Mass of the final-round attention on the two type-q modules, per mille, as the cue's query evidence rises
    from 0 to 1. Descriptive only."""
    curve = []
    n = batch["x"].shape[0]
    rows = torch.arange(n)
    for alpha in IGNITION_ALPHAS:
        x = gwtask.set_cue_strength(batch, alpha, cfg["n_types"], cfg["n_values"], torch.Generator().manual_seed(5))
        with torch.no_grad():
            net(x)
        if net.scheme in ("workspace", "no_broadcast"):
            per = net.trace[f"write{net.rounds - 1}"].mean(dim=1)
        elif net.scheme == "all_to_all":
            per = net.trace["pairwise"][:, -1, :n_modules]
        else:
            curve.append(0)
            continue
        mass = per[rows, batch["pi"]] + per[rows, batch["pj"]]
        curve.append(int(1000 * float(mass.mean())))
    return curve


def sharpness(curve: list) -> int:
    rise = curve[-1] - curve[0]
    if rise <= 0:
        return 0
    steps = [b - a for a, b in zip(curve, curve[1:])]
    return permille(max(steps), rise) if max(steps) > 0 else 0


# -- the run ---------------------------------------------------------------------------------

def run(seed: int, overrides: dict) -> dict:
    t0 = time.time()
    cfg = load_config()
    cfg.update(overrides)
    scale = "pinned" if not overrides else "custom"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    n_train = cfg["n_train_modules"]
    train_seed = seed * 1000 + cfg["train_stream_offset"]
    gens = {k: torch.Generator().manual_seed(seed * 1000 + cfg["eval_stream_offset"] + i) for i, k in enumerate(("id", "ood16", "ood24", "curve", "probe", "drop", "ign"))}
    train_data = make_batch(torch.Generator().manual_seed(train_seed), cfg["train_size"], n_train, cfg)
    train_hashes = set(row_hashes(train_data["x"]))
    id_batch = make_batch(gens["id"], cfg["eval_size"], n_train, cfg)
    ood16 = make_batch(gens["ood16"], cfg["eval_size"], 16, cfg)
    ood24 = make_batch(gens["ood24"], cfg["eval_size"], 24, cfg)
    curve_batch = make_batch(gens["curve"], cfg["curve_eval_size"], n_train, cfg)
    probe_x = make_batch(gens["probe"], cfg["probe_size"], n_train, cfg)["x"]
    ign_batch = make_batch(gens["ign"], cfg["probe_size"], n_train, cfg)
    drop_batch = make_batch(gens["drop"], cfg["eval_size"], n_train, cfg)

    # Parameter matching: every arm's update width is chosen to match the workspace arm.
    target = gwmodel.count_params(gwmodel.build("workspace", cfg))
    widths = {"workspace": cfg["ffn_workspace"]}
    for name in gwmodel.ARM_SPECS:
        if name != "workspace":
            widths[name] = gwmodel.match_width(name, cfg, target)
    arms, nets, observed_n, nonfinite = {}, {}, set(), 0
    for name in gwmodel.ARM_SPECS:
        net, (curve, counts), loss, finite = train_arm(
            name, cfg, seed, train_data, curve_batch, cfg["steps"], widths[name], cfg["arm_lr"][name], observed_n
        )
        nonfinite += 0 if finite else 1
        rec = {
            "scheme": net.scheme,
            "slots": net.slots,
            "ffn": widths[name],
            "lr_micro": round(cfg["arm_lr"][name] * 1_000_000),
            "params": gwmodel.count_params(net),
            "comm_ops_n8": gwmodel.comm_ops(name, cfg, n_train),
            "comm_ops_n16": gwmodel.comm_ops(name, cfg, 16),
            "comm_ops_n24": gwmodel.comm_ops(name, cfg, 24),
            "curve_permille": curve,
            "auc_correct": sum(c for c, _ in counts),
            "auc_total": sum(t for _, t in counts),
            "final_loss_micro": int(1e6 * loss) if math.isfinite(loss) else -1,
        }
        for key, batch in (("id", id_batch), ("ood16", ood16), ("ood24", ood24)):
            rec[f"{key}_correct"], rec[f"{key}_total"] = evaluate(net, batch)
        rec.update(dropout_eval(net, drop_batch, n_train, gens["drop"]))
        if name in ("workspace", "all_to_all"):
            rec["ignition_curve_permille"] = ignition_curve(net, ign_batch, cfg, n_train)
            rec["ignition_sharpness_permille"] = sharpness(rec["ignition_curve_permille"])
        arms[name], nets[name] = rec, net

    counts_p = {n: arms[n]["params"] for n in gwmodel.ARM_SPECS}
    mismatches = param_mismatch_arms(counts_p, "workspace", cfg["param_tolerance_permille"])

    scale_p = float(cfg["probe_scale"])
    nb_leaks = leak_count(nets["no_broadcast"], probe_x, scale_p, "sensory")
    nc_leaks = leak_count(nets["no_comm"], probe_x, scale_p, "reporter")
    nc_sens = leak_count(nets["no_comm"], probe_x, scale_p, "sensory")
    ws_reach = leak_count(nets["workspace"], probe_x, scale_p, "sensory")
    a2a_reach = leak_count(nets["all_to_all"], probe_x, scale_p, "sensory")
    nb_reporter = leak_count(nets["no_broadcast"], probe_x, scale_p, "reporter")
    bott = 0
    for name in ("workspace", "no_broadcast", "competition_off", "capacity_off"):
        with torch.no_grad():
            nets[name](probe_x)
        bott += bottleneck_violations(dict(nets[name].trace), nets[name].top_k) if name != "competition_off" else 0
    drop_leaks = sum(dropped_leaks(nets[n], probe_x[: cfg["drop_probe_size"]], scale_p) for n in ("workspace", "all_to_all", "no_broadcast"))
    nc_id = permille(arms["no_comm"]["id_correct"], arms["no_comm"]["id_total"])
    majority = permille(int(torch.bincount(id_batch["y"]).max()), int(id_batch["y"].numel()))
    above_chance = int(nc_id > majority + cfg["chance_slack_permille"])
    overlap = overlap_count(train_hashes, id_batch["x"]) + overlap_count(train_hashes, ood16["x"][:, :n_train]) + overlap_count(train_hashes, ood24["x"][:, :n_train])
    n_leaks = n_leak_count(observed_n, OOD_MODULES)
    nondet = int(first_steps_loss("workspace", cfg, seed, train_data, widths["workspace"]) != first_steps_loss("workspace", cfg, seed, train_data, widths["workspace"]))

    # Negative controls: each doctored input (or a net known to communicate, never the product arm under test)
    # must trip the check it exists for. A control that fails is a harness defect (exit 2).
    fake_trace = {"write0": torch.tensor([[[0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]]])}
    controls = {
        "n_leak": {"required_rule_tripped": n_leak_count(observed_n | {16}, OOD_MODULES) > 0},
        "overlap": {"required_rule_tripped": overlap_count(train_hashes, train_data["x"][:8]) > 0},
        "param_mismatch": {"required_rule_tripped": param_mismatch_arms({"workspace": target, "x": target * 2}, "workspace", cfg["param_tolerance_permille"]) > 0},
        "isolation_probe": {"required_rule_tripped": leak_count(nets["all_to_all"], probe_x, scale_p, "sensory") > 0},
        "bottleneck": {"required_rule_tripped": bottleneck_violations(fake_trace, cfg["top_k"]) > 0},
        "dropout_probe": {"required_rule_tripped": dropout_probe_is_sensitive(nets["all_to_all"], probe_x, scale_p)},
    }

    comparator = max(permille(arms[n]["ood16_correct"], arms[n]["ood16_total"]) for n in ("all_to_all", "bw_all_to_all"))
    ws_id = permille(arms["workspace"]["id_correct"], arms["workspace"]["id_total"])
    result = {
        "benchmark": BENCHMARK,
        "experiment": EXPERIMENT,
        "seed": seed,
        "scale": scale,
        "procedure": {
            "seed_is_declared": seed in cfg["seeds"],
            "shape_is_pinned": scale == "pinned",
            "train_size": cfg["train_size"],
            "n_train_modules": n_train,
            "ood_modules": list(OOD_MODULES),
        },
        "arms": arms,
        "arms_trained": len(arms),
        "id_eval_instances": int(id_batch["y"].numel()),
        "ood16_eval_instances": int(ood16["y"].numel()),
        "ood24_eval_instances": int(ood24["y"].numel()),
        "majority_permille": majority,
        "no_comm_id_permille": nc_id,
        "comparator_primary_permille": comparator,
        "workspace_trained": int(ws_id >= nc_id + cfg["trained_margin_permille"]),
        "workspace_influence_hits": ws_reach,
        "pairwise_influence_hits": a2a_reach,
        "no_broadcast_reporter_hits": nb_reporter,
        "no_comm_sensory_hits": nc_sens,
        "param_mismatch_arms": mismatches,
        "no_comm_leaks": nc_leaks,
        "no_broadcast_leaks": nb_leaks,
        "bottleneck_violations": bott,
        "dropped_module_leaks": drop_leaks,
        "n_leaks": n_leaks,
        "train_eval_overlap": overlap,
        "nondeterminism": nondet,
        "nonfinite_arms": nonfinite,
        "no_comm_above_chance": above_chance,
        "controls": controls,
        "harness_gaps": [],
        "timing_s": int(time.time() - t0),
        "summary": {n: {"id": permille(r["id_correct"], r["id_total"]), "ood16": permille(r["ood16_correct"], r["ood16_total"]), "ood24": permille(r["ood24_correct"], r["ood24_total"])} for n, r in arms.items()},
    }
    coverage_failed, falsifiers_failed = judge(result, cfg)
    result["coverage_failed"] = coverage_failed
    result["falsifiers_failed"] = falsifiers_failed
    result["guards_failed"] = [r for r in guard_rules(cfg) if not holds(result, r)]
    result["hard_failures"] = len(falsifiers_failed)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="must be c001; the runner checks this token in the command")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a pinned value (scale 'custom', never evidence)")
    args = parser.parse_args()
    if args.benchmark != BENCHMARK:
        print(f"unknown benchmark {args.benchmark!r}", file=sys.stderr)
        return 2
    problems = environment_problems()
    if problems:
        print("the environment is incomplete, refusing to run: " + "; ".join(problems), file=sys.stderr)
        return 2
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    _load_torch()
    try:
        result = run(args.seed, overrides)
    except Exception:  # harness integrity
        traceback.print_exc()
        return 2
    failed_controls = [k for k, v in result["controls"].items() if not v["required_rule_tripped"]]
    print(json.dumps(result, sort_keys=True))
    if failed_controls:
        print(f"harness integrity: negative controls did not trip: {failed_controls}", file=sys.stderr)
        return 2
    return 1 if (result["coverage_failed"] or result["falsifiers_failed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
