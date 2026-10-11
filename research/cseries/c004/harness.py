"""C004 harness: algorithmic recurrence with a carried state against a parameter- and FLOP-matched feed-forward stack, one seed.

    python research/cseries/c004/harness.py --benchmark c004 --seed 5001

Prints one JSON object on its last line (the result the aggregator judges).
Exit 0: every coverage rule and counter holds. Exit 1: a coverage rule or a STRUCTURAL counter failed (a defect of the recurrent
arm, the feed-forward arm or an isolated ablation; never a statistical comparison). Exit 2: harness integrity: the environment is
incomplete, an exception, a negative control that did not trip, bad arguments, or an INTEGRITY counter above 0 (a defect of the
task or the harness: parameter or FLOP matching, a sequence length leak, train/validation/evaluation overlap, labels, checkpoint
or convergence bookkeeping), which makes the seed, and so the experiment, INVALID rather than FALSIFIED.
The comparison of the arms across seeds, the validity guards and the sensitivity controls are the aggregator's.

Data regime: finite pools of sequences of 2..8 steps (cycled with replacement); evaluation on 8 steps (in distribution) and on
12..32 steps (longer than any trained sequence). All arms of a world see the same batches in the same order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIB = HERE.parent / "lib"
EXPERIMENT_DIR = ROOT / "experiments/consciousness/C004-recurrent-integration"
CONFIG_PATH = EXPERIMENT_DIR / "config.toml"
BENCHMARK = "c004"
EXPERIMENT = "C004"

sys.path.insert(0, str(LIB))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import c004_rules as rules  # noqa: E402
from common import arm_seed as _arm_seed  # noqa: E402
from common import flop_mismatch_arms, param_mismatch_arms, permille  # noqa: E402

torch = task = model = train = None


def arm_seed(seed: int, name: str) -> int:
    return _arm_seed(EXPERIMENT, seed, name)


def _load_torch():
    global torch, task, model, train
    import torch as _torch

    torch = _torch
    import c004_model as _model
    import c004_task as _task
    import train as _train

    task, model, train = _task, _model, _train


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = dict(tomllib.loads(path.read_text(encoding="utf-8"))["pinned_thresholds"])
    cfg["max_train_length"] = max(cfg["train_lengths"])
    cfg["extrapolation_lengths"] = [cfg["primary_length"], *cfg["secondary_lengths"]]
    cfg["eval_lengths"] = sorted({cfg["id_length"], cfg["primary_length"], *cfg["secondary_lengths"], *cfg["curve_extra_lengths"]})
    return cfg


def preflight() -> int:
    """Check the environment and build every arm and run one forward pass, without reading a seed or training. Run it through the
    runner's exact environment before the runner reserves a seed."""
    problems = common.environment_problems(CONFIG_PATH)
    if problems:
        print("preflight failed: " + "; ".join(problems), file=sys.stderr)
        return 2
    _load_torch()
    try:
        cfg = load_config()
        torch.manual_seed(0)
        x = torch.randint(0, 6, (2, cfg["max_train_length"]))
        for net in build_arms(cfg)[0].values():
            net(x)
        print(json.dumps({"preflight": "ok", "environment": common.environment_record(ROOT, torch)}, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 2


# -- structural checks (each is also run on a doctored input as a negative control) --------------------

def length_leak_count(observed_lengths: set, extrapolation_lengths: list) -> int:
    """Evaluation-only sequence lengths that a training batch actually had."""
    return len(set(observed_lengths) & set(extrapolation_lengths))


def row_hashes(x) -> list:
    return [hashlib.sha256(bytes([x.shape[1]] + row)).digest()[:8] for row in x.tolist()]


def overlap_count(forbidden: set, x) -> int:
    """Sequences of `x` identical to one in `forbidden` (by hash of length and content)."""
    return sum(1 for h in row_hashes(x) if h in forbidden)


def label_mismatch_count(tokens, y) -> int:
    """Sequences whose label differs from an independent recomputation (permutations as tuples, no table)."""
    return sum(1 for a, b in zip(task.labels_independent(tokens), y.tolist()) if a != b)


def fresh_batch(gen, n: int, length: int, forbidden: set):
    """n sequences of `length` steps none of which is in `forbidden` (rejection sampling)."""
    rows = []
    while len(rows) < n:
        batch = task.sample(gen, max(n, 64), length)
        for row, h in zip(batch, row_hashes(batch)):
            if h not in forbidden:
                rows.append(row)
                if len(rows) == n:
                    break
    return torch.stack(rows)


def predictions(net, x, chunk: int = 1024):
    with torch.no_grad():
        return torch.cat([net(x[i : i + chunk]).argmax(dim=-1) for i in range(0, x.shape[0], chunk)])


def accuracy(net, x, y) -> tuple[int, int]:
    return int((predictions(net, x) == y).sum()), int(y.numel())


def carry_hits(net, x, position: int = 0) -> tuple[int, int]:
    """(instances whose logits moved by more than 1e-6, instances whose argmax changed) when the step at `position` is replaced by a
    different symbol."""
    bumped = x.clone()
    bumped[:, position] = (bumped[:, position] + 1) % 6
    with torch.no_grad():
        a, b = net(x), net(bumped)
    moved = int(((a - b).abs().amax(dim=-1) > 1e-6).sum())
    flipped = int((a.argmax(-1) != b.argmax(-1)).sum())
    return moved, flipped


def causal_leaks(net, x) -> int:
    """Instances in which replacing the LAST step changed the read-out at an EARLIER position (a causal stack must not move)."""
    bumped = x.clone()
    bumped[:, -1] = (bumped[:, -1] + 1) % 6
    with torch.no_grad():
        a, b = net.forward_all(x)[:, :-1], net.forward_all(bumped)[:, :-1]
    return int(((a - b).abs().amax(dim=(1, 2)) > 1e-6).sum())


# -- arms --------------------------------------------------------------------------------------------

# name -> (family, world); the world names the labels the arm trains on
ARM_SPECS = {
    "recurrent": ("recurrent", "task"),
    "feedforward": ("feedforward", "task"),
    "feedforward_inlength": ("feedforward", "task"),  # the same stack, trained with the primary length in its training set (extrapolation removed)
    "reset_state": ("reset", "task"),
    "recurrent_perm": ("recurrent", "perm"),
    "feedforward_perm": ("feedforward", "perm"),
    "recurrent_null": ("recurrent", "null"),
    "feedforward_null": ("feedforward", "null"),
}


def build_arms(cfg: dict, seed: int | None = None):
    """The nets, their parameter counts and analytic FLOPs at the reference length. The feed-forward arm is matched to the recurrent
    arm in parameters (tolerance) and, among the candidates, in FLOPs at the longest trained length. With a `seed`, every net's initial weights
    are drawn from a generator seeded by (experiment, seed, arm) just before it is built, so they do not depend on the process's global random state."""
    de, dh, ref = cfg["d_embed"], cfg["d_hidden"], cfg["max_train_length"]
    target_params = model.recurrent_params(de, dh)
    target_flops = model.recurrent_flops(ref, de, dh)
    width, ffn = model.match_feedforward(
        target_params, target_flops, ref, cfg["ff_layers"], cfg["ff_heads"], cfg["param_tolerance_permille"], cfg["flop_tolerance_permille"]
    )
    nets = {}
    for name, (family, _) in ARM_SPECS.items():
        if seed is not None:
            torch.manual_seed(arm_seed(seed, "init:" + name))
        if family == "feedforward":
            nets[name] = model.Feedforward(width, cfg["ff_heads"], ffn, cfg["ff_layers"], positional=cfg["ff_positional"], slope_scale=cfg["ff_slope_scale"])
        else:
            nets[name] = model.Recurrent(de, dh, reset=(family == "reset"))
    flops = {
        name: (model.feedforward_flops(ref, width, ffn, cfg["ff_layers"]) if family == "feedforward" else model.recurrent_flops(ref, de, dh))
        for name, (family, _) in ARM_SPECS.items()
    }
    return nets, flops, {"width": width, "ffn": ffn}


def weights_digest(net) -> str:
    """A digest of every parameter and buffer of `net` (it changes iff some weight does)."""
    digest = hashlib.sha256()
    for key, tensor in sorted(net.state_dict().items()):
        digest.update(key.encode())
        digest.update(repr(tensor.detach().flatten().tolist()).encode())
    return digest.hexdigest()


def init_state_dependence(cfg: dict, seed: int) -> int:
    """How many arms' initial weights depend on the process's global random state: the arms are built twice for the same seed after two different global
    seeds and their weight digests compared. Zero means the initial weights are a function of the seed alone."""
    digests = []
    for global_seed in (11, 22):
        torch.manual_seed(global_seed)
        digests.append({name: weights_digest(net) for name, net in build_arms(cfg, seed)[0].items()})
    return sum(int(digests[0][name] != digests[1][name]) for name in digests[0])


def sequence_loss(net, x, y):
    """Cross-entropy of the read-out at every step against the counter after that step."""
    return torch.nn.functional.cross_entropy(net.forward_all(x).reshape(-1, 6), y.reshape(-1))


def first_steps_loss(name, cfg, seed, pools, ys, steps: int = 20) -> float:
    torch.manual_seed(arm_seed(seed, "determinism:" + name))
    net = build_arms(cfg)[0][name]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["arm_lr"][name], weight_decay=cfg["weight_decay_milli"] / 1000)
    length = cfg["max_train_length"]
    loss = None
    for k in range(steps):
        idx = torch.arange(k * cfg["batch_size"], (k + 1) * cfg["batch_size"]) % pools[length].shape[0]
        loss = sequence_loss(net, pools[length][idx], ys[length][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


# -- the run -----------------------------------------------------------------------------------------

def run(seed: int, overrides: dict, only_arms: tuple | None = None) -> dict:
    """One seed. `only_arms` (the pilot tuning scripts only) trains just those arms and returns their records."""
    t0 = time.time()
    cfg = load_config()
    cfg.update(overrides)
    if overrides:
        cfg["eval_lengths"] = sorted({cfg["id_length"], cfg["primary_length"], *cfg["secondary_lengths"], *cfg["curve_extra_lengths"]})
        cfg["extrapolation_lengths"] = [cfg["primary_length"], *cfg["secondary_lengths"]]
        cfg["max_train_length"] = max(cfg["train_lengths"])
    scale = "pinned" if not overrides else "custom"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = seed * 1000
    names = ("pool", "val", "eval", "probe", "corrupt", "perm", "schedule")
    gens = {k: torch.Generator().manual_seed(base + cfg["stream_offset"] + i) for i, k in enumerate(names)}

    # -- data: finite pools, a validation set and evaluation sets that overlap neither the pools nor each other
    pools = {length: task.sample(gens["pool"], cfg["pool_per_length"], length) for length in cfg["train_lengths"]}
    ys = {length: task.running_labels(x) for length, x in pools.items()}  # the counter after every step: the supervision of every arm
    null_ys = {length: task.window_running_labels(x) for length, x in pools.items()}
    perm_ys = {length: task.permute_labels(y, gens["perm"]) for length, y in ys.items()}
    perm_same = sum(int((perm_ys[length][:, -1] == ys[length][:, -1]).sum()) for length in pools)
    perm_total = sum(int(ys[length].shape[0]) for length in pools)
    primary = cfg["primary_length"]
    long_pool = task.sample(gens["pool"], cfg["pool_per_length"], primary)  # drawn after every other pool: only the in-length control trains on it
    long_ys = task.running_labels(long_pool)
    train_hashes = {h for x in [*pools.values(), long_pool] for h in row_hashes(x)}
    id_len = cfg["id_length"]
    val_x = fresh_batch(gens["val"], cfg["val_size"], id_len, train_hashes)
    val_long = fresh_batch(gens["val"], cfg["val_size"], primary, train_hashes)  # the in-length control validates on its own training distribution too (the primary length)
    val_hashes = set(row_hashes(val_x)) | set(row_hashes(val_long))
    val_y = task.labels(val_x)
    val_long_y = task.labels(val_long)
    val_worlds = {"task": val_y, "null": task.window_labels(val_x), "perm": task.permute_labels(val_y, gens["perm"])}
    evals = {}
    for length in cfg["eval_lengths"]:
        size = cfg["eval_size"] if length in (id_len, cfg["primary_length"], *cfg["secondary_lengths"]) else cfg["curve_eval_size"]
        evals[length] = fresh_batch(gens["eval"], size, length, train_hashes | val_hashes)
    eval_y = {length: task.labels(x) for length, x in evals.items()}
    eval_null = {length: task.window_labels(x) for length, x in evals.items()}

    # -- arms
    nets, flops, ff_shape = build_arms(cfg, seed)
    target = model.count_params(nets["recurrent"])
    observed_lengths: set = set()
    seen_lengths: dict = {}
    arms, trained, nonfinite, checkpoint_mismatch, convergence_mismatch = {}, {}, 0, 0, 0
    labels_of = {"task": ys, "null": null_ys, "perm": perm_ys}
    for name, (family, world) in ARM_SPECS.items():
        if only_arms is not None and name not in only_arms:
            continue
        torch.manual_seed(arm_seed(seed, name))
        net = nets[name]
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["arm_lr"][name], weight_decay=cfg["weight_decay_milli"] / 1000)
        sched = torch.Generator().manual_seed(base + cfg["stream_offset"] + 17)
        lengths = [*cfg["train_lengths"], primary] if name == "feedforward_inlength" else cfg["train_lengths"]
        pool_labels = {**labels_of[world], primary: long_ys}
        pool_x = {**pools, primary: long_pool}
        seen_lengths[name] = set()

        def step_loss(step, net=net, sched=sched, lengths=lengths, pool_labels=pool_labels, pool_x=pool_x, opt=opt, name=name):
            for group in opt.param_groups:  # the same linear warmup for every arm
                group["lr"] = cfg["arm_lr"][name] * min(1.0, step / cfg["warmup_steps"])
            length = lengths[int(torch.randint(0, len(lengths), (1,), generator=sched))]
            idx = torch.randint(0, cfg["pool_per_length"], (cfg["batch_size"],), generator=sched)
            seen_lengths[name].add(int(pool_x[length].shape[1]))
            if name != "feedforward_inlength":
                observed_lengths.add(int(pool_x[length].shape[1]))
            return sequence_loss(net, pool_x[length][idx], pool_labels[length][idx])

        if name == "feedforward_inlength":
            def validate(n):
                (c1, t1), (c2, t2) = accuracy(n, val_x, val_y), accuracy(n, val_long, val_long_y)
                return c1 + c2, t1 + t2
        else:
            validate = lambda n, w=world: accuracy(n, val_x, val_worlds[w])  # noqa: E731
        cap = {"task": cfg["max_steps"], "null": cfg["max_steps_null"], "perm": cfg["max_steps_control"]}[world] if family != "reset" else cfg["max_steps_control"]
        net, info, loss, finite = train.train_to_plateau(net, opt, step_loss, validate, cfg, max_steps=cap)
        nonfinite += 0 if finite else 1
        cm, vm = train.audit(net, info, validate, cfg)
        checkpoint_mismatch += cm
        convergence_mismatch += vm
        rec = {
            "family": family,
            "world": world,
            "lr_micro": round(cfg["arm_lr"][name] * 1_000_000),
            "converged": int(info["converged"]),
            "steps": info["steps"],
            "step_cap": info["cap"],
            "best_val_permille": info["best_val"],
            "train_loss_window_micro": info["loss_window_micro"],
            "params": model.count_params(net),
            "flops_ref": flops[name],
            "curve_permille": info["curve"],
            "final_loss_micro": int(1e6 * loss) if math.isfinite(loss) else -1,
        }
        # perm arms are scored against the TRUE labels (the association they were denied), null arms against the null labels.
        scoring = eval_null if world == "null" else eval_y
        for length in cfg["eval_lengths"]:
            c, t = accuracy(net, evals[length], scoring[length])
            key = "id" if length == id_len else f"ex{length}"
            rec[f"{key}_correct"], rec[f"{key}_total"] = c, t
        arms[name], nets[name] = rec, net

    if only_arms is not None:
        return {"arms": arms, "feedforward_shape": ff_shape}

    # -- the graded sensitivity control: the trained recurrent net on corrupted evaluation inputs (labels of the clean sequences)
    corrupted = task.corrupt(evals[primary], gens["corrupt"], cfg["corruption_rate_permille"])
    changed = int((corrupted != evals[primary]).any(dim=1).sum())
    c, t = accuracy(nets["recurrent"], corrupted, eval_y[primary])
    arms["recurrent_corrupt"] = {"family": "recurrent", "world": "task", "params": arms["recurrent"]["params"], f"ex{primary}_correct": c, f"ex{primary}_total": t,
                                 "corrupted_sequences": changed}

    # -- counters
    counts_p = {n: arms[n]["params"] for n in ARM_SPECS}
    flops_ref = {n: arms[n]["flops_ref"] for n in ARM_SPECS}
    mismatches = param_mismatch_arms(counts_p, "recurrent", cfg["param_tolerance_permille"])
    flop_mismatches = flop_mismatch_arms(flops_ref, "recurrent", cfg["flop_tolerance_permille"])
    probe_x = task.sample(gens["probe"], cfg["probe_size"], primary)
    carry_moved, carry_flipped = carry_hits(nets["recurrent"], probe_x)
    _, ff_flipped = carry_hits(nets["feedforward"], probe_x)
    reset_moved, _ = carry_hits(nets["reset_state"], probe_x)
    ff_leaks = causal_leaks(nets["feedforward"], probe_x)
    label_flip = int((task.labels(probe_x) != task.labels(torch.cat([(probe_x[:, :1] + 1) % 6, probe_x[:, 1:]], dim=1))).sum())
    id_chance = cfg["chance_permille"]
    reset_id = permille(arms["reset_state"]["id_correct"], arms["reset_state"]["id_total"])
    overlap = sum(overlap_count(train_hashes, evals[length]) for length in evals)
    val_overlap = sum(overlap_count(val_hashes, evals[length]) for length in evals)
    n_leaks = length_leak_count(observed_lengths, cfg["extrapolation_lengths"])
    inlength_errors = int(seen_lengths["feedforward_inlength"] != set(cfg["train_lengths"]) | {primary}) + int(any(seen_lengths[n] != set(cfg["train_lengths"]) for n in seen_lengths if n != "feedforward_inlength"))
    label_checks = sum(label_mismatch_count(evals[length][:128], eval_y[length][:128]) for length in evals)
    label_checks += sum(sum(int(a != b) for a, b in zip(task.window_labels_independent(evals[length][:128]), eval_null[length][:128].tolist())) for length in evals)
    nondet = sum(
        int(first_steps_loss(n, cfg, seed, pools, ys) != first_steps_loss(n, cfg, seed, pools, ys)) for n in ("recurrent", "feedforward")
    ) + init_state_dependence(cfg, seed)

    # -- negative controls: each doctored input (or a net known to carry state or to be non-causal, never the arm under test)
    noncausal = model.Feedforward(nets["feedforward"].embed.embedding_dim, cfg["ff_heads"], ff_shape["ffn"], cfg["ff_layers"], causal=False, positional=cfg["ff_positional"], slope_scale=cfg["ff_slope_scale"])
    wrong = (eval_y[id_len] + 1) % 6
    controls = {
        "length_leak": {"required_rule_tripped": length_leak_count(observed_lengths | {primary}, cfg["extrapolation_lengths"]) > 0},
        "inlength_probe": {"required_rule_tripped": int(seen_lengths["feedforward"] != set(cfg["train_lengths"]) | {primary}) > 0},
        "overlap": {"required_rule_tripped": overlap_count(train_hashes, pools[id_len][:8]) > 0},
        "param_mismatch": {"required_rule_tripped": param_mismatch_arms({"recurrent": target, "x": target * 2}, "recurrent", cfg["param_tolerance_permille"]) > 0},
        "flop_mismatch": {"required_rule_tripped": flop_mismatch_arms({"recurrent": 1000, "x": 2000}, "recurrent", cfg["flop_tolerance_permille"]) > 0},
        "label_check": {"required_rule_tripped": label_mismatch_count(evals[id_len][:64], wrong[:64]) > 0},
        "isolation_probe": {"required_rule_tripped": carry_moved > 0},
        "causal_probe": {"required_rule_tripped": causal_leaks(noncausal, probe_x) > 0},
    }

    comparator = permille(arms["feedforward"][f"ex{primary}_correct"], arms["feedforward"][f"ex{primary}_total"])
    ids = {n: permille(arms[n]["id_correct"], arms[n]["id_total"]) for n in ("recurrent", "feedforward")}
    result = {
        "benchmark": BENCHMARK,
        "experiment": EXPERIMENT,
        "seed": seed,
        "scale": scale,
        "procedure": {
            "seed_is_declared": seed in cfg["seeds"],
            "shape_is_pinned": scale == "pinned",
            "train_lengths": list(cfg["train_lengths"]),
            "primary_length": primary,
            "pool_per_length": cfg["pool_per_length"],
        },
        "arms": arms,
        "arms_trained": len(ARM_SPECS),
        "id_eval_instances": arms["recurrent"]["id_total"],
        "primary_eval_instances": arms["recurrent"][f"ex{primary}_total"],
        "secondary_eval_instances": min(arms["recurrent"][f"ex{length}_total"] for length in cfg["secondary_lengths"]),
        "comparator_primary_permille": comparator,
        "recurrent_trained": rules.trained(ids["recurrent"], cfg["train_floor_permille"]),
        "feedforward_trained": rules.trained(ids["feedforward"], cfg["train_floor_permille"]),
        "unconverged_judged_arms": rules.unconverged_guard_arms({n: arms[n]["converged"] for n in rules.JUDGED_ARMS}),
        "unconverged_arms": [n for n in arms if "converged" in arms[n] and not arms[n]["converged"]],
        "environment": common.environment_record(ROOT, torch),
        "feedforward_shape": {**ff_shape, "layers": cfg["ff_layers"], "heads": cfg["ff_heads"], "positional": cfg["ff_positional"], "slope_scale": cfg["ff_slope_scale"]},
        "recurrent_carry_hits": carry_moved,
        "recurrent_carry_flip_permille": permille(carry_flipped, probe_x.shape[0]),
        "feedforward_carry_flip_permille": permille(ff_flipped, probe_x.shape[0]),
        "label_flip_permille": permille(label_flip, probe_x.shape[0]),
        "reset_carry_leaks": reset_moved,
        "ff_causal_leaks": ff_leaks,
        "nondeterminism": nondet,
        "param_mismatch_arms": mismatches,
        "flop_mismatch_arms": flop_mismatches,
        "length_leaks": n_leaks,
        "inlength_control_errors": inlength_errors,
        "train_eval_overlap": overlap,
        "val_eval_overlap": val_overlap,
        "nonfinite_arms": nonfinite,
        "checkpoint_mismatch_arms": checkpoint_mismatch,
        "convergence_mismatch_arms": convergence_mismatch,
        "label_mismatches": label_checks,
        "reset_above_chance": rules.above_chance(reset_id, id_chance, cfg["chance_slack_permille"]),
        "perm_label_leaks": rules.label_leak(perm_same, perm_total, id_chance, cfg["perm_label_slack_permille"]),
        "corruption_rate_errors": rules.rate_error(changed, int(evals[primary].shape[0]), cfg["corruption_rate_permille"], cfg["corruption_slack_permille"]),
        "controls": controls,
        "harness_gaps": [],
        "timing_s": int(time.time() - t0),
        "cpu_s": int(time.process_time()),
        "summary": {n: {k: permille(r[f"{k}_correct"], r[f"{k}_total"]) for k in ("id", *(f"ex{ln}" for ln in cfg["eval_lengths"] if ln != id_len)) if f"{k}_correct" in r} for n, r in arms.items()},
    }
    coverage_failed, falsifiers_failed = common.judge(result, rules.coverage_rules(cfg), rules.falsifier_rules())
    result["coverage_failed"] = coverage_failed
    result["falsifiers_failed"] = falsifiers_failed
    result["integrity_failed"] = rules.integrity_failed(result)
    result["guards_failed"] = [r for r in rules.guard_rules(cfg) if not common.holds(result, r)]
    result["hard_failures"] = len(falsifiers_failed) + len(result["integrity_failed"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="must be c004; the runner checks this token in the command")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight", action="store_true", help="check the environment and exit; reads no seed, trains nothing")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override a pinned value (scale 'custom', never evidence)")
    args = parser.parse_args()
    if args.benchmark != BENCHMARK:
        print(f"unknown benchmark {args.benchmark!r}", file=sys.stderr)
        return 2
    if args.preflight:
        return preflight()
    if args.seed is None:
        print("--seed is required", file=sys.stderr)
        return 2
    problems = common.environment_problems(CONFIG_PATH)
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
    broken = [rule.split(" ")[0] for rule in result["integrity_failed"]]
    print(json.dumps(result, sort_keys=True))
    if failed_controls or broken:
        print(f"harness integrity: negative controls did not trip: {failed_controls}", file=sys.stderr)
        print(f"harness integrity: integrity counters above 0: {broken}", file=sys.stderr)
        return 2
    return 1 if (result["coverage_failed"] or result["falsifiers_failed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
