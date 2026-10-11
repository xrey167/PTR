"""C008 harness: an attention schema used to control routing against a schema-free router, a heuristic balancer and a describe-only schema, one seed.

    python research/cseries/c003a/harness.py --benchmark c008 --seed 7001

Prints one JSON object on its last line (the result the aggregator judges).
Exit 0: every coverage rule and counter holds. Exit 1: a coverage rule or a STRUCTURAL counter failed (the describe-only schema reaches the
router, the schema reaches the environment, nondeterminism; never a statistical comparison). Exit 2: harness integrity: the environment is
incomplete, an exception, a negative control that did not trip, bad arguments, or an INTEGRITY counter above 0 (matching, streams, the
verifier, the demand schedule, bookkeeping), which makes the seed, and so the experiment, INVALID rather than FALSIFIED.
The comparison of the arms across seeds, the guards and the sensitivity controls are the aggregator's.

World: bursty cyclic demand over eight experts with limited capacity (task.py). Every arm is trained online by policy gradient on the verifier's
reward and sees the same stream of batches; evaluation is on a fresh stream of the same cycle, with the policy frozen (the balancer's bias keeps
adapting, as it would in use)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIB = HERE.parent / "lib"
EXPERIMENT_DIR = ROOT / "experiments/consciousness/C008-attention-schema-routing-cpu"
CONFIG_PATH = EXPERIMENT_DIR / "config.toml"
BENCHMARK = "c008"
EXPERIMENT = "C008"

sys.path.insert(0, str(LIB))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import c008_rules as rules  # noqa: E402
from common import arm_seed as _arm_seed  # noqa: E402
from common import flop_mismatch_arms, param_mismatch_arms, permille  # noqa: E402

torch = task = model = train = None

# name -> (kind, world); the eval-only arms (shuffled history, epsilon routing, random routing) are built from trained ones
ARM_SPECS = {
    "schema_free": ("schema_free", "task"),
    "schema_free_wide": ("schema_free_wide", "task"),
    "heuristic": ("heuristic", "task"),
    "describe_only": ("describe_only", "task"),
    "control": ("control", "task"),
    "history_router": ("history_router", "task"),
    "schema_free_null": ("schema_free", "null"),
    "control_null": ("control", "null"),
}


def arm_seed(seed: int, name: str) -> int:
    return _arm_seed(EXPERIMENT, seed, name)


def _load_torch():
    global torch, task, model, train
    import torch as _torch

    torch = _torch
    import c008_model as _model
    import c008_task as _task
    import train as _train

    task, model, train = _task, _model, _train


def load_config(path: Path = CONFIG_PATH) -> dict:
    return dict(tomllib.loads(path.read_text(encoding="utf-8"))["pinned_thresholds"])


@dataclass
class Environment:
    """What the verifier reads: the capacity of every expert and which experts are competent for which class. The schema must never write it."""

    capacity: "torch.Tensor"
    competent: "torch.Tensor"  # (classes, experts) bool

    def digest(self) -> str:
        return hashlib.sha256(bytes(self.capacity.tolist()) + bytes(self.competent.flatten().tolist())).hexdigest()


def make_environment(cfg: dict) -> Environment:
    e = cfg["experts"]
    competent = torch.zeros(e, e, dtype=torch.bool)
    for c in range(e):
        competent[c, c] = True
        competent[c, (c + 1) % e] = True
    return Environment(torch.full((e,), cfg["capacity"], dtype=torch.long), competent)


def serve(env: Environment, classes, choices):
    """Verified tokens: competent expert AND within the expert's capacity, counting arrivals in batch order."""
    rank = torch.zeros_like(choices)
    for e in range(env.capacity.shape[0]):
        mask = choices == e
        rank[mask] = torch.arange(int(mask.sum()))
    return env.competent[classes, choices] & (rank < env.capacity[choices])


def batch_hash(classes) -> bytes:
    return hashlib.sha256(bytes(classes.tolist())).digest()[:8]


def preflight() -> int:
    problems = common.environment_problems(CONFIG_PATH)
    if problems:
        print("preflight failed: " + "; ".join(problems), file=sys.stderr)
        return 2
    _load_torch()
    try:
        cfg = load_config()
        torch.manual_seed(0)
        arms, _ = build_arms(cfg)
        x = torch.randn(cfg["tokens"], cfg["experts"])
        history = torch.zeros(cfg["history"] * cfg["experts"])
        for arm in arms.values():
            arm(x, history)
        print(json.dumps({"preflight": "ok", "environment": common.environment_record(ROOT, torch)}, sort_keys=True))
        return 0
    except Exception:
        traceback.print_exc()
        return 2


# -- checks (each is also run on a doctored input as a negative control) -------------------------------

def overlap_count(forbidden: set, hashes: list) -> int:
    """Batches whose class sequence is in `forbidden` (the training stream)."""
    return sum(1 for h in hashes if h in forbidden)


def audit_mismatches(fast: list, reference: list) -> int:
    """Tokens on which two verdict lists differ."""
    return sum(1 for a, b in zip(fast, reference) if a != b)


def reward_audit(env: Environment, classes, choices) -> int:
    """Tokens on which the vectorised verifier differs from the plain-loop reference."""
    fast = serve(env, classes, choices).tolist()
    ref = task.serve_reference(classes.tolist(), choices.tolist(), env.capacity.shape[0], int(env.capacity[0]))
    return audit_mismatches(fast, ref)


def schedule_error(classes, hot_classes: tuple, experts: int, floor_permille: int) -> int:
    """1 iff a batch puts less than `floor_permille` of its tokens on the regime's two hot classes."""
    mass = sum(int((classes == c).sum()) for c in hot_classes)
    return int(1000 * mass < floor_permille * classes.numel())


def schema_influence(arm, x, history, experts: int) -> int:
    """1 iff the router's distribution moves when the schema's output is replaced by a different distribution: the schema reaches the router."""
    with torch.no_grad():
        a = arm(x, history, torch.softmax(torch.zeros(experts), -1))[0]
        spike = torch.zeros(experts)
        spike[0] = 1.0
        b = arm(x, history, spike)[0]
    return int(float((torch.softmax(a, -1) - torch.softmax(b, -1)).abs().max()) > 1e-6)


def authority_leak(env: Environment, arm, x, history, gen, probe_classes) -> int:
    """1 iff routing with an adversarial schema output changed what the verifier reads (capacities, competence)."""
    before = env.digest()
    for k in range(3):
        override = torch.softmax(torch.randn(env.capacity.shape[0], generator=gen) * 4, -1)
        route(arm, env, x, history, gen, override)
    return int(env.digest() != before)


def history_influence(arm, x, experts: int, history_len: int) -> int:
    """1 iff the router's distribution moves when the history it is given changes: the history reaches the router (the history comparator is not blind)."""
    with torch.no_grad():
        a = arm(x, torch.zeros(history_len * experts))[0]
        b = arm(x, torch.rand(history_len * experts, generator=torch.Generator().manual_seed(3)))[0]
    moved = float((torch.softmax(a, -1) - torch.softmax(b, -1)).abs().max())
    return int(moved > 1e-7)


def wire_control_law(arm, experts: int):
    """A copy of the (trained) control arm whose control law is the identity (all gains one): the path from the schema's output to the router's logits, which a
    trained law may have learned to leave near zero, made visible to the influence probe."""
    import copy

    wired = copy.deepcopy(arm)
    with torch.no_grad():
        wired.law.weight.copy_(torch.eye(experts) if wired.law.form == "full" else torch.ones_like(wired.law.weight))
    return wired


def route(arm, env: Environment, x, history, gen, override=None):
    """One routing pass: (choices, probs, logits, prediction, base probabilities). The environment is passed so that a schema with a write path could be caught."""
    logits, prediction, base = arm(x, history, override)
    probs = torch.softmax(logits, dim=-1)
    choices = torch.multinomial(torch.nan_to_num(probs, nan=1.0 / probs.shape[-1]), 1, generator=gen).squeeze(1)  # non-finite weights give uniform routing, not a crash
    return choices, probs, logits, prediction, base


# -- arms --------------------------------------------------------------------------------------------

def build_arms(cfg: dict, seed: int | None = None):
    """The six arms with their hidden widths. With a `seed`, every arm's initial weights are drawn from a generator seeded by (experiment, seed, arm) just before
    it is built, so they do not depend on the process's global random state. The schema-free arm is matched to the control in FLOPs per batch (the schema's FLOPs
    included), the wide schema-free arm and the history router (the router that reads the history directly, without a schema) in parameters; the heuristic arm
    shares the schema-free arm's router."""
    control_hidden = cfg["router_hidden"]
    target_flops = model.batch_flops("control", cfg, control_hidden)
    target_params = model.total_params("control", cfg, control_hidden)
    free_hidden = model.match_hidden(target_flops, lambda h: model.batch_flops("schema_free", cfg, h))
    wide_hidden = model.match_hidden(target_params, lambda h: model.total_params("schema_free_wide", cfg, h))
    history_hidden = model.match_hidden(target_params, lambda h: model.total_params("history_router", cfg, h))
    hidden = {"schema_free": free_hidden, "schema_free_wide": wide_hidden, "heuristic": free_hidden, "describe_only": control_hidden, "control": control_hidden,
              "history_router": history_hidden}
    arms = {}
    for name, (kind, _) in ARM_SPECS.items():
        if seed is not None:
            torch.manual_seed(arm_seed(seed, "init:" + name))
        arms[name] = model.Arm(kind, cfg, hidden[kind])
    return arms, hidden


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
        digests.append({name: weights_digest(arm) for name, arm in build_arms(cfg, seed)[0].items()})
    return sum(int(digests[0][name] != digests[1][name]) for name in digests[0])


def zero_history(cfg: dict):
    return torch.zeros(cfg["history"], cfg["experts"])


def push(history, load):
    return torch.cat([history[1:], load[None]], dim=0)


def evaluate(arm, env, cfg, stream, gen_seed: int, mode: str = "normal", eps_permille: int = 0):
    """Run the frozen policy over `stream` (a list of (classes, x)). Returns per-batch rewarded counts and descriptive totals."""
    gen = torch.Generator().manual_seed(gen_seed)
    shuffle_gen = torch.Generator().manual_seed(gen_seed + 1)
    bias0 = arm.bias.clone()
    history, pool = zero_history(cfg), []
    out = {"rewarded": [], "tokens": 0, "load_cv": 0.0, "entropy": 0.0, "top1_hits": 0, "batches": 0, "eps_changed": 0, "shuffle_same": 0, "shuffle_total": 0,
           "requests": [], "probs": [], "l1_schema": 0.0, "l1_persist": 0.0}
    with torch.no_grad():
        for classes, x in stream:
            use = history
            if mode == "shuffled" and len(pool) > cfg["history"]:
                idx = torch.randint(0, len(pool), (cfg["history"],), generator=shuffle_gen)
                use = torch.stack([pool[int(i)] for i in idx])
                out["shuffle_same"] += int(torch.equal(use, history))
                out["shuffle_total"] += 1
            if mode == "even_split":  # the hand-set regime-blind reference: the class the noisy features point at, the two competent experts alternating at random
                guess = x.argmax(-1)
                choices = (guess + torch.randint(0, 2, (cfg["tokens"],), generator=gen)) % cfg["experts"]
                probs = torch.full((cfg["tokens"], cfg["experts"]), 1.0 / cfg["experts"])
                prediction, base = None, probs
            elif mode == "random":
                choices = torch.randint(0, cfg["experts"], (cfg["tokens"],), generator=gen)
                probs = torch.full((cfg["tokens"], cfg["experts"]), 1.0 / cfg["experts"])
                prediction, base = None, probs
            else:
                choices, probs, _, prediction, base = route(arm, env, x, use.flatten(), gen)
            if mode == "epsilon":
                hit = torch.rand(cfg["tokens"], generator=gen) < eps_permille / 1000
                choices = torch.where(hit, torch.randint(0, cfg["experts"], (cfg["tokens"],), generator=gen), choices)
                out["eps_changed"] += int(hit.sum())
            reward = serve(env, classes, choices)
            counts = torch.bincount(choices, minlength=cfg["experts"]).float()
            load = base.mean(0)  # the router's own propensity this batch (before any control term)
            out["rewarded"].append(int(reward.sum()))
            out["tokens"] += cfg["tokens"]
            out["load_cv"] += float(counts.std() / counts.mean())
            out["entropy"] += float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1).mean())
            if prediction is not None:
                out["top1_hits"] += int(prediction.argmax() == load.argmax())
                if out["batches"] > 0:  # the first batch has no previous one
                    out["l1_schema"] += float((prediction - load).abs().sum())
                    out["l1_persist"] += float((history[-1] - load).abs().sum())
            out["requests"].append(load)
            out["batches"] += 1
            pool.append(load)
            arm.update_bias(counts, cfg["heuristic_rate"])
            history = push(history, load)
    arm.bias.copy_(bias0)
    return out


def make_stream(cfg: dict, gen_seed: int, batches: int, stationary: bool):
    gen = torch.Generator().manual_seed(gen_seed)
    return [task.sample_batch(gen, t, cfg, stationary) for t in range(batches)]


def first_steps_loss(name, cfg, seed, env, steps: int = 30) -> float:
    torch.manual_seed(arm_seed(seed, "determinism:" + name))
    arm = build_arms(cfg)[0][name]
    trainer = Trainer(arm, env, cfg, seed, "task", cfg["arm_lr"][name], name)
    loss = None
    for step in range(steps):
        loss = trainer.step_loss(step + 1)
        trainer.optimiser.zero_grad()
        loss.backward()
        trainer.optimiser.step()
    return float(loss)


class Trainer:
    """Online policy-gradient training of one arm on the shared training stream; the schema (if any) is trained by its own loss."""

    def __init__(self, arm, env, cfg, seed, world, lr, name):
        self.arm, self.env, self.cfg, self.name = arm, env, cfg, name
        self.stationary = world == "null"
        self.stream_gen = torch.Generator().manual_seed(seed * 1000 + cfg["stream_offset"])
        self.sample_gen = torch.Generator().manual_seed(arm_seed(seed, "sample:" + name))
        self.history, self.t = zero_history(cfg), 0
        self.optimiser = torch.optim.Adam(arm.policy_parameters(), lr=lr)
        self.schema_optimiser = torch.optim.Adam(arm.schema_parameters(), lr=cfg["schema_lr"]) if arm.schema_on else None
        self.train_hashes: set = set()

    def step_loss(self, step):
        cfg = self.cfg
        classes, x = task.sample_batch(self.stream_gen, self.t, cfg, self.stationary)
        if self.t < cfg["overlap_window"]:
            self.train_hashes.add(batch_hash(classes))
        self.t += 1
        logits, prediction, base = self.arm(x, self.history.flatten())
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        choices = torch.multinomial(torch.nan_to_num(probs.detach(), nan=1.0 / cfg["experts"]), 1, generator=self.sample_gen).squeeze(1)
        reward = serve(self.env, classes, choices).float()
        advantage = reward - reward.mean()
        taken = log_probs.gather(1, choices[:, None]).squeeze(1)
        entropy = -(probs * log_probs).sum(-1).mean()
        loss = -(advantage * taken).mean() - cfg["arm_entropy_milli"][self.name] / 1000 * entropy
        counts = torch.bincount(choices, minlength=cfg["experts"]).float()
        load = base.mean(0)  # the router's own propensity this batch (before any control term): the history and the schema's target
        if self.schema_optimiser is not None:
            schema_loss = -(load * torch.log(prediction.clamp_min(1e-8))).sum()  # cross-entropy to the router's own propensity
            self.schema_optimiser.zero_grad()
            schema_loss.backward()
            self.schema_optimiser.step()
        self.arm.update_bias(counts, cfg["heuristic_rate"])
        self.history = push(self.history, load.detach())
        return loss


# -- the run -----------------------------------------------------------------------------------------

def run(seed: int, overrides: dict, only_arms: tuple | None = None) -> dict:
    """One seed. `only_arms` (the pilot tuning scripts only) trains just those arms and returns their evaluation."""
    t0 = time.time()
    cfg = load_config()
    cfg.update(overrides)
    scale = "pinned" if not overrides else "custom"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = seed * 1000
    env = make_environment(cfg)
    e, tokens = cfg["experts"], cfg["tokens"]

    # -- streams: the training stream is regenerated identically by every arm; validation and evaluation streams are fresh
    val_streams = {w: make_stream(cfg, base + cfg["val_offset"], cfg["val_batches"], w == "null") for w in ("task", "null")}
    eval_streams = {w: make_stream(cfg, base + cfg["eval_offset"], cfg["eval_batches"], w == "null") for w in ("task", "null")}
    oracle = {w: [task.oracle_reward(c, e, int(env.capacity[0])) for c, _ in eval_streams[w]] for w in eval_streams}
    probe_stream = make_stream(cfg, base + cfg["probe_offset"], cfg["probe_batches"], False)

    arms, hidden = build_arms(cfg, seed)
    results, nonfinite, checkpoint_mismatch, convergence_mismatch = {}, 0, 0, 0
    train_hashes: set = set()
    records, evals = {}, {}
    for name, (kind, world) in ARM_SPECS.items():
        if only_arms is not None and name not in only_arms:
            continue
        torch.manual_seed(arm_seed(seed, name))
        arm = arms[name]
        trainer = Trainer(arm, env, cfg, seed, world, cfg["arm_lr"][name], name)

        def validate(net, w=world):
            out = evaluate(net, env, cfg, val_streams[w], base + cfg["val_offset"] + 7)
            return sum(out["rewarded"]), out["tokens"]

        cap = cfg["max_steps"] if world == "task" else cfg["max_steps_null"]
        arm, info, loss, finite = train.train_to_plateau(arm, trainer.optimiser, trainer.step_loss, validate, cfg, max_steps=cap)
        train_hashes |= trainer.train_hashes
        nonfinite += 0 if finite else 1
        cm, vm = train.audit(arm, info, validate, cfg)
        checkpoint_mismatch += cm
        convergence_mismatch += vm
        evals[name] = evaluate(arm, env, cfg, eval_streams[world], base + cfg["eval_offset"] + 11)
        records[name] = {
            "kind": kind, "world": world, "hidden": hidden[kind], "lr_micro": round(cfg["arm_lr"][name] * 1_000_000), "converged": int(info["converged"]),
            "steps": info["steps"], "step_cap": info["cap"], "best_val_permille": info["best_val"],
            "params": model.total_params(kind, cfg, hidden[kind]), "flops_batch": model.batch_flops(kind, cfg, hidden[kind]),
            "curve_permille": info["curve"], "train_loss_window_micro": info["loss_window_micro"],
        }
    if only_arms is not None:
        return {"arms": {n: {**records[n], **score(evals[n], cfg)} for n in records}, "hidden": hidden}

    # -- eval-only arms: built from trained ones, no training of their own
    evals["control_shuffled"] = evaluate(arms["control"], env, cfg, eval_streams["task"], base + cfg["eval_offset"] + 11, mode="shuffled")
    evals["schema_free_epsilon"] = evaluate(arms["schema_free"], env, cfg, eval_streams["task"], base + cfg["eval_offset"] + 11, mode="epsilon", eps_permille=cfg["epsilon_permille"])
    evals["random_router"] = evaluate(arms["schema_free"], env, cfg, eval_streams["task"], base + cfg["eval_offset"] + 11, mode="random")
    evals["even_split"] = evaluate(arms["schema_free"], env, cfg, eval_streams["task"], base + cfg["eval_offset"] + 11, mode="even_split")
    for name in ("control_shuffled", "schema_free_epsilon", "random_router", "even_split"):
        records[name] = {"kind": "eval_only", "world": "task"}

    arms_out = {}
    for name, ev in evals.items():
        world = records[name]["world"]
        arms_out[name] = {**records[name], **score(ev, cfg)}
    arms_out["oracle"] = {"kind": "oracle", "world": "task", "reward_correct": sum(oracle["task"]), "reward_total": tokens * len(oracle["task"])}
    arms_out["oracle_null"] = {"kind": "oracle", "world": "null", "reward_correct": sum(oracle["null"]), "reward_total": tokens * len(oracle["null"])}

    # -- counters
    params = {n: records[n]["params"] for n in ("control", "describe_only", "schema_free_wide", "history_router")}
    flops = {n: records[n]["flops_batch"] for n in ("control", "describe_only", "schema_free", "heuristic")}
    mismatches = param_mismatch_arms(params, "control", cfg["param_tolerance_permille"])
    flop_mismatches = flop_mismatch_arms(flops, "control", cfg["flop_tolerance_permille"])
    eval_hashes = [batch_hash(c) for w in eval_streams for c, _ in eval_streams[w]] + [batch_hash(c) for w in val_streams for c, _ in val_streams[w]] + [batch_hash(c) for c, _ in probe_stream]
    overlap = overlap_count(train_hashes, eval_hashes)
    audit_gen = torch.Generator().manual_seed(base + cfg["probe_offset"] + 5)
    audit_mismatch = 0
    for classes, _ in eval_streams["task"][:20]:
        # three kinds of choices: uniform (no expert is full), each token to a competent expert (the hot experts fill up and overflow), all to one expert
        for choices in (torch.randint(0, e, (tokens,), generator=audit_gen), classes.clone(), torch.zeros_like(classes)):
            audit_mismatch += reward_audit(env, classes, choices)
    violations = 0
    for name, ev in evals.items():
        world = records[name]["world"]
        violations += sum(1 for got, best in zip(ev["rewarded"], oracle[world]) if got > best)
    schedule = 0
    for w, stream in eval_streams.items():
        for t, (classes, _) in enumerate(stream):
            regime = 0 if w == "null" else (t // cfg["dwell"]) % cfg["regimes"]  # recomputed here, not by the generator under audit
            schedule += schedule_error(classes, (regime % e, (regime + 1) % e), e, cfg["schedule_floor_permille"])
    eps_error = rules.rate_error(evals["schema_free_epsilon"]["eps_changed"], evals["schema_free_epsilon"]["tokens"], cfg["epsilon_permille"], cfg["epsilon_slack_permille"])
    shuffle_err = rules.shuffle_error(evals["control_shuffled"]["shuffle_same"], max(1, evals["control_shuffled"]["shuffle_total"]), cfg["shuffle_same_ceiling_permille"])
    probe_x = probe_stream[0][1]
    zero_h = torch.zeros(cfg["history"] * e)
    describe_hits = sum(schema_influence(arms["describe_only"], x, zero_h, e) for _, x in probe_stream)
    history_hits = history_influence(arms["history_router"], probe_stream[0][1], e, cfg["history"])
    wired = wire_control_law(arms["control"], e)
    control_hits = sum(schema_influence(wired, x, zero_h, e) for _, x in probe_stream)
    leak_gen = torch.Generator().manual_seed(base + cfg["probe_offset"] + 9)
    leaks = authority_leak(env, arms["control"], probe_x, zero_h, leak_gen, probe_stream[0][0])
    nondet = sum(int(first_steps_loss(n, cfg, seed, env) != first_steps_loss(n, cfg, seed, env)) for n in ("schema_free", "control")) + init_state_dependence(cfg, seed)

    # -- negative controls: each doctored input (or an arm known to reach the router), never the arm under test
    fed = wire_control_law(model.Arm("describe_only", cfg, cfg["router_hidden"]), e)
    fed.feeds_router = True  # a describe-only arm doctored to route through its schema
    doctored = Environment(env.capacity.clone(), env.competent.clone())
    doctored.capacity[0] -= 1
    controls = {
        "stream_overlap": {"required_rule_tripped": overlap_count(train_hashes | {batch_hash(eval_streams["task"][0][0])}, eval_hashes) > 0},
        "param_mismatch": {"required_rule_tripped": param_mismatch_arms({"control": 1000, "x": 2000}, "control", cfg["param_tolerance_permille"]) > 0},
        "flop_mismatch": {"required_rule_tripped": flop_mismatch_arms({"control": 1000, "x": 2000}, "control", cfg["flop_tolerance_permille"]) > 0},
        "reward_audit": {"required_rule_tripped": audit_mismatches([True, False, True], [False, False, True]) > 0},
        "authority_probe": {"required_rule_tripped": doctored.digest() != env.digest()},
        "influence_probe": {"required_rule_tripped": control_hits > 0},
        "history_probe": {"required_rule_tripped": history_influence(arms["schema_free"], probe_stream[0][1], e, cfg["history"]) == 0},
        "describe_probe": {"required_rule_tripped": sum(schema_influence(fed, x, zero_h, e) for _, x in probe_stream) > 0},
    }

    permille_of = {n: permille(a["reward_correct"], a["reward_total"]) for n, a in arms_out.items()}
    best_comparator = max(permille_of[n] for n in rules.COMPARATORS)
    top1 = permille(evals["control"]["top1_hits"], evals["control"]["batches"])
    gain = rules.gain_over_persistence(evals["control"]["l1_schema"], evals["control"]["l1_persist"])
    result = {
        "benchmark": BENCHMARK, "experiment": EXPERIMENT, "seed": seed, "scale": scale,
        "procedure": {"seed_is_declared": seed in cfg["seeds"], "shape_is_pinned": scale == "pinned", "eval_batches": cfg["eval_batches"], "tokens": tokens},
        "arms": arms_out,
        "arms_trained": len(ARM_SPECS),
        "eval_instances": arms_out["control"]["reward_total"],
        "null_eval_instances": arms_out["control_null"]["reward_total"],
        "oracle_permille": permille_of["oracle"],
        "comparator_best_permille": best_comparator,
        "headroom_permille": rules.headroom(permille_of["oracle"], best_comparator),
        "comparator_above_random_permille": best_comparator - permille_of["random_router"],
        "schema_top1_permille": top1,
        "schema_gain_permille": gain,
        "schema_trained": rules.schema_trained(gain, cfg["schema_floor_permille"]),
        "unconverged_judged_arms": rules.unconverged_guard_arms({n: records[n]["converged"] for n in rules.JUDGED_ARMS}),
        "unconverged_arms": [n for n in ARM_SPECS if not records[n]["converged"]],
        "environment": common.environment_record(ROOT, torch),
        "control_influence_hits": control_hits,
        "describe_only_influence": describe_hits,
        "history_router_blind": int(history_hits == 0),
        "blind_reference_permille": permille_of["even_split"],
        "authority_leaks": leaks,
        "nondeterminism": nondet,
        "param_mismatch_arms": mismatches,
        "flop_mismatch_arms": flop_mismatches,
        "stream_overlap": overlap,
        "reward_audit_mismatches": audit_mismatch,
        "oracle_violations": violations,
        "schedule_errors": schedule,
        "nonfinite_arms": nonfinite,
        "checkpoint_mismatch_arms": checkpoint_mismatch,
        "convergence_mismatch_arms": convergence_mismatch,
        "epsilon_rate_errors": eps_error,
        "shuffle_errors": shuffle_err,
        "controls": controls,
        "harness_gaps": [],
        "timing_s": int(time.time() - t0),
        "cpu_s": int(time.process_time()),
        "summary": {n: permille_of[n] for n in permille_of},
    }
    coverage_failed, falsifiers_failed = common.judge(result, rules.coverage_rules(cfg), rules.falsifier_rules())
    result["coverage_failed"] = coverage_failed
    result["falsifiers_failed"] = falsifiers_failed
    result["integrity_failed"] = rules.integrity_failed(result)
    result["guards_failed"] = [r for r in rules.guard_rules(cfg) if not common.holds(result, r)]
    result["hard_failures"] = len(falsifiers_failed) + len(result["integrity_failed"])
    return result


def finite_round(value: float) -> int:
    """round() that gives -1 for a non-finite value (an arm with non-finite weights is counted by nonfinite_arms, not allowed to crash the harness)."""
    return round(value) if math.isfinite(value) else -1


def score(ev: dict, cfg: dict) -> dict:
    """The record of one evaluation: verified tokens and the descriptive totals."""
    n = max(1, ev["batches"])
    return {
        "reward_correct": sum(ev["rewarded"]), "reward_total": ev["tokens"], "load_cv_milli": finite_round(1000 * ev["load_cv"] / n),
        "entropy_milli": finite_round(1000 * ev["entropy"] / n), "schema_top1_permille": permille(ev["top1_hits"], n),
        "schema_gain_permille": rules.gain_over_persistence(ev["l1_schema"], ev["l1_persist"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="must be c008; the runner checks this token in the command")
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
