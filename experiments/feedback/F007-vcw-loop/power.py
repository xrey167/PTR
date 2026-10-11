"""Planning model and power analysis of F007 (not evidence, not part of the verdict).

`simulate` is an independent re-implementation, in plain Python, of what one request does in
each arm of the harness, with the learned routers replaced by a stated imperfection (they pick the
best admissible Pod with probability `1 - exploration - slip` and a uniformly random one
otherwise). It reads the world from config.toml, so the skill tables, the defect rates and the
budget cannot drift from the harness. It exists for two things written down in
PREREGISTRATION.md sections 2, 6 and 9: calibrating the difficulty of the tasks so that no arm is at
its ceiling or floor, and the power of the decision rule at the primary margin. Its numbers are
plans; the pilots and the confirmatory runs are measurements.

    python3 experiments/feedback/F007-vcw-loop/power.py model --world hetero
    python3 experiments/feedback/F007-vcw-loop/power.py power --sd-seed 0.006
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))["pinned_thresholds"]

KINDS = {
    "arithmetic": ["arith.format", "arith.value", "arith.bounds"],
    "json": ["json.parse", "json.type", "json.required", "json.extra"],
    "code": ["code.syntax", "code.test_failed", "code.timeout"],
}
KIND_ORDER = list(KINDS)
ROLE = {
    "arith.format": "format", "arith.value": "value", "arith.bounds": "value",
    "json.parse": "format", "json.type": "schema", "json.required": "schema",
    "json.extra": "schema", "code.syntax": "code", "code.test_failed": "code",
    "code.timeout": "resource",
}
REPAIRERS = [("fmt_1", "format"), ("fmt_2", "format"), ("val_1", "value"), ("val_2", "value"),
             ("sch_1", "schema"), ("sch_2", "schema"), ("cod_1", "code"), ("cod_2", "code"),
             ("gen_r", "generalist")]
PROPOSERS = ["gen_1", "gen_2", "gen_3"]
SHIFT = {"format": "value", "value": "schema", "schema": "code", "code": "format"}
# What is visible at once: the order in which the rendering reveals defects.
MASK = {
    "arithmetic": [["arith.format"], ["arith.bounds"], ["arith.value"]],
    "code": [["code.syntax"], ["code.timeout"], ["code.test_failed"]],
}


def table(text: str) -> dict[str, list[int]]:
    out = {}
    for group in text.split(";"):
        name, rest = group.split("=")
        out[name] = [int(v) for v in rest.split(",")]
    return out


def error_rates() -> dict[str, int]:
    return {k: int(v) for k, v in (p.split("=") for p in CONFIG["error_permille"].split(","))}


def role_of(world: str, code: str) -> str | None:
    role = ROLE[code]
    if role == "resource":
        return None if world != "homog" else "code"
    return role


def skill(world: str, pod: tuple[str, str], code: str) -> float:
    role = role_of(world, code)
    if role is None:
        return CONFIG["gap_skill_permille"] / 1000
    on, span, gen, off = table(CONFIG["skill_permille"])[world]
    if pod[1] == "generalist":
        return gen / 1000
    if pod[1] == role:
        if span == 0:
            return on / 1000
        steps = span // 10
        digest = hashlib.sha256(f"f007-skill|{world}|{pod[0]}|{code}".encode()).digest()
        return (on + (digest[0] % (2 * steps + 1)) * 10 - steps * 10) / 1000
    return off / 1000


def proposer_rate(world: str, proposer: str, kind: str, code: str) -> float:
    base = error_rates()[code]
    if world == "homog":
        return base / 1000
    good = PROPOSERS[KIND_ORDER.index(kind)] == proposer
    mult = CONFIG["proposer_good_multiplier_permille" if good else "proposer_other_multiplier_permille"]
    return min(base * mult // 1000, 950) / 1000


def visible(kind: str, defects: set[str]) -> list[str]:
    """The codes the verifier reports for a candidate carrying `defects`."""
    if not defects:
        return []
    if kind == "json":
        if "json.parse" in defects:
            return ["json.parse"]
        shown = []
        if "json.required" in defects:
            shown.append("json.required")
        elif "json.type" in defects:
            shown.append("json.type")
        if "json.extra" in defects:
            shown.append("json.extra")
        return shown
    for tier in MASK[kind]:
        shown = [c for c in tier if c in defects]
        if shown:
            return shown
    return []


def registry(world: str, code: str, mode: str) -> list[tuple[str, str]]:
    if mode == "none":
        return []
    if mode == "generic":
        return list(REPAIRERS)
    role = role_of(world, code)
    if role is None:
        return []
    if mode == "oracle":
        return [max(REPAIRERS, key=lambda p: skill(world, p, code))]
    if mode == "shuffled":
        role = SHIFT[role]
    return [p for p in REPAIRERS if p[1] in (role, "generalist")]


# The learned routers are modelled by an imperfection: the best admissible Pod with probability
# 1 - exploration - slip, a uniformly random one otherwise. The Thompson-sampling router of the full
# loop keeps a slip; the softmax gate and the gate-weighted proposers are tuned over a grid that
# includes the greedy limit, so their slip is zero (the router's own exploration floor remains).
SLIP = {"full": 0.05, "no_l3": 0.05, "no_workspace": 0.05, "shuffled": 0.05, "gate": 0.0, "bon": 0.0,
        "no_finding": 0.0, "oracle": 0.0}
EXPLORATION = 0.05


def simulate_request(world, arm, kind, rng, budget, born):
    """One request. Returns (success, verifier calls, first call passed, first candidate carried the
    unrepairable code)."""
    codes = KINDS[kind]
    mode = {"full": "true", "no_l3": "true", "no_workspace": "true", "shuffled": "shuffled",
            "gate": "generic", "no_finding": "generic", "bon": "none", "oracle": "oracle"}[arm]
    good = PROPOSERS[KIND_ORDER.index(kind)]
    slip = SLIP[arm]
    broadcast = arm != "no_workspace"
    second_chance = {"ctx": CONFIG["broadcast_second_chance_permille"] / 1000}.get(world, 0.0)
    l3 = arm == "full"

    def propose(first=False):
        if first:
            # The first draft is shared by every arm: the kind's specialised proposer.
            who = good
        elif arm == "no_finding" or world == "homog":
            who = rng.choice(PROPOSERS)
        elif arm == "oracle" or rng.random() > EXPLORATION + slip:
            who = good
        else:
            who = rng.choice(PROPOSERS)
        return {c for c in codes if rng.random() < proposer_rate(world, who, kind, c)}

    defects = propose(first=True)
    carried = "code.timeout" in visible(kind, defects)
    for call in range(1, budget + 1):
        shown = visible(kind, defects)
        if not shown:
            return True, call, call == 1, carried
        if call == budget:
            break
        routed = False
        for c in shown:
            pods = registry(world, c, mode)
            is_born = c == "code.timeout" and born and l3
            if is_born:
                pods = pods + [("born", "born")]
            if not pods:
                continue
            routed = True
            if is_born:
                p = pods[-1] if rng.random() > EXPLORATION else rng.choice(pods)
                s = CONFIG["born_skill_permille"] / 1000
            else:
                if arm == "no_finding":
                    p = rng.choice(pods)
                elif arm in ("gate", "bon"):
                    def mean_skill(q):
                        return sum(skill(world, q, k) for k in codes) / len(codes)
                    best = max(pods, key=mean_skill)
                    p = best if rng.random() > EXPLORATION + slip else rng.choice(pods)
                else:
                    best = max(pods, key=lambda q: skill(world, q, c))
                    p = best if rng.random() > EXPLORATION + slip else rng.choice(pods)
                s = skill(world, p, c)
            if rng.random() < s:
                defects = defects - {c}
            elif broadcast and second_chance and rng.random() < second_chance:
                defects = defects - {c}
        if not routed:
            defects = propose()
    return False, budget, False, carried


ALL_ARMS = ("full", "no_l3", "no_workspace", "gate", "bon", "no_finding", "shuffled", "oracle")


def simulate(world: str, n: int, seed: int, arms=ALL_ARMS):
    rng_tasks = random.Random(seed)
    weights = CONFIG["kind_weights_permille"]
    out = {a: {"ok": [], "first": [], "carried": [], "kind": []} for a in arms}
    for i in range(n):
        r = rng_tasks.randrange(1000)
        kind = KIND_ORDER[0] if r < weights[0] else KIND_ORDER[1] if r < weights[0] + weights[1] else KIND_ORDER[2]
        born = i > 400  # L3 has acted after the first windows
        for arm in arms:
            rng = random.Random(f"{seed}|{i}|{arm}")
            ok, _, first, carried = simulate_request(world, arm, kind, rng, CONFIG["verifier_call_budget"], born)
            out[arm]["ok"].append(ok)
            out[arm]["first"].append(first)
            out[arm]["carried"].append(carried)
            out[arm]["kind"].append(kind)
    return out


def rate(res, arm):
    return sum(res[arm]["ok"]) / len(res[arm]["ok"])


def conditional_rate(res, arm):
    pairs = [ok for ok, c in zip(res[arm]["ok"], res[arm]["carried"]) if c]
    return sum(pairs) / max(len(pairs), 1)


def derive(n: int = 20000, seed: int = 11) -> dict:
    """Every threshold the preregistration derives from the planning model, and nothing from a pilot.
    Each is half of the modelled gain (or the modelled value itself where stated)."""
    het = simulate("hetero", n, seed)
    ctx = simulate("ctx", n, seed)
    values = {
        "predicted_rate_permille": {a: round(1000 * rate(het, a)) for a in ("full", "gate", "bon")},
        "baseline_min_gain_permille": round(500 * (rate(het, "gate") - rate(het, "no_finding"))),
        "bon_min_gain_permille": round(500 * (rate(het, "bon") - sum(het["bon"]["first"]) / n)),
        "workspace_margin_permille": round(500 * (rate(ctx, "full") - rate(ctx, "no_workspace"))),
        "l3_margin_permille": round(500 * (conditional_rate(het, "full") - conditional_rate(het, "no_l3"))),
        "modelled": {
            "gate_minus_untrained_pool": round(1000 * (rate(het, "gate") - rate(het, "no_finding"))),
            "bon_minus_single_shot": round(1000 * (rate(het, "bon") - sum(het["bon"]["first"]) / n)),
            "ctx_full_minus_no_workspace": round(1000 * (rate(ctx, "full") - rate(ctx, "no_workspace"))),
            "l3_gain_on_requests_carrying_the_gap_code": round(1000 * (conditional_rate(het, "full") - conditional_rate(het, "no_l3"))),
            "oracle_minus_full": round(1000 * (rate(het, "oracle") - rate(het, "full"))),
            "full_minus_gate": round(1000 * (rate(het, "full") - rate(het, "gate"))),
            "full_minus_bon": round(1000 * (rate(het, "full") - rate(het, "bon"))),
        },
    }
    return values


def cmd_derive(args):
    import json
    print(json.dumps(derive(args.n, args.seed), indent=1))


def cmd_model(args):
    for world in args.world:
        res = simulate(world, args.n, args.seed)
        print(f"{world}: " + " ".join(f"{a}={rate(res, a):.3f}" for a in res))


def t_quantile_one_sided_95(df: int) -> float:
    table = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 9: 1.833}
    return table.get(df, 1.645)


def cmd_power(args):
    """Power of 'lower one-sided 95% bound of the mean per-seed difference exceeds the margin'
    (t-interval over seeds, df = seeds - 1) as the true mean difference and the between-seed sd
    vary; the within-seed sampling error of a seed's difference is added from n and the
    discordance rate."""
    rng = random.Random(1)
    seeds = args.seeds
    t = t_quantile_one_sided_95(seeds - 1)
    margin = args.margin
    within = math.sqrt(args.discordant / args.n)
    print(f"seeds={seeds} n={args.n} margin={margin} between-seed sd={args.sd_seed} within-seed sd={within:.4f}")
    print("true difference  P(lcb > margin)  P(ucb < margin)")
    for true in [margin - 0.02, margin - 0.01, margin, margin + 0.01, margin + 0.02, margin + 0.03, margin + 0.05, margin + 0.08]:
        reject_low = reject_high = 0
        reps = 20000
        for _ in range(reps):
            ds = [rng.gauss(true, math.hypot(args.sd_seed, within)) for _ in range(seeds)]
            mean = sum(ds) / seeds
            sd = math.sqrt(sum((d - mean) ** 2 for d in ds) / (seeds - 1))
            half = t * sd / math.sqrt(seeds)
            reject_low += mean - half > margin
            reject_high += mean + half < margin
        print(f"{true:+.3f}          {reject_low / reps:.3f}            {reject_high / reps:.3f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    m = sub.add_parser("model")
    m.add_argument("--world", nargs="+", default=["hetero", "homog", "mild"])
    m.add_argument("--n", type=int, default=4000)
    m.add_argument("--seed", type=int, default=7)
    m.set_defaults(fn=cmd_model)
    d = sub.add_parser("derive")
    d.add_argument("--n", type=int, default=20000)
    d.add_argument("--seed", type=int, default=11)
    d.set_defaults(fn=cmd_derive)
    p = sub.add_parser("power")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--sd-seed", type=float, default=0.006)
    p.add_argument("--discordant", type=float, default=0.2)
    p.set_defaults(fn=cmd_power)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
