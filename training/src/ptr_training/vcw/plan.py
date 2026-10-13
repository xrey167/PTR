"""GPU-hour ledger of the VCW-MoE plan (ESTIMATES; see `budget` for the assumptions).

`PROTOCOL` holds the per-profile batch, length and checkpointing choices that the
`training/configs/vcw-*.toml` files carry; a unit test keeps the two in sync. `ledger()` returns
one row per costed item; `print_plan()` prints the markdown used in docs/VCW_MOE_GPU_PLAN.md
(`PYTHONPATH=training/src python3 -m ptr_training.vcw.budget`).
"""

from __future__ import annotations

from ptr_training.vcw import budget as b

PERFT_PARAMS = 4_456_448  # M011 common target (8 experts, rank 8, top-2, router, 16 layers)
REPAIR_PARAMS = 4_194_304  # generic rank 16 on q,k,v,o == 8 classes x rank 2
SEEDS = 5
PILOT_SEEDS = 3
EVAL_TASKS_ID = 659 + 54 + 133  # GSM8K test mod2==1, HumanEval mod3==2, BFCL simple mod3==2
EVAL_TASKS_OOD = 500 + 500  # MBPP test, hendrycks MATH (levels 1-3, numeric answers) first 500
EVAL_TASKS = EVAL_TASKS_ID + EVAL_TASKS_OOD
EVAL_PROMPT_TOKENS = 256
REPAIR_PROMPT_TOKENS = 400  # task + failed candidate + finding codes

PROTOCOL = {
    "rtx4090-reference": {
        "sft": {"micro_batch": 4, "accumulation": 8, "seq_tokens": 768, "steps": 2000},
        "rl": {"prompts": 1, "generations": 8, "mini_steps": 4, "prompt_tokens": 128, "completion_tokens": 160, "steps": 400},
        "repair": {"micro_batch": 2, "accumulation": 8, "seq_tokens": 1024, "steps": 1000},
        "probe": {"micro_batch": 16, "seq_tokens": 320},
        "gen_batch": 32,
    },
    "rtx3090-reference": {
        "sft": {"micro_batch": 4, "accumulation": 8, "seq_tokens": 768, "steps": 2000},
        "rl": {"prompts": 1, "generations": 8, "mini_steps": 4, "prompt_tokens": 128, "completion_tokens": 160, "steps": 400},
        "repair": {"micro_batch": 2, "accumulation": 8, "seq_tokens": 1024, "steps": 1000},
        "probe": {"micro_batch": 16, "seq_tokens": 320},
        "gen_batch": 32,
    },
    "rtx5090-reference": {
        "sft": {"micro_batch": 8, "accumulation": 4, "seq_tokens": 768, "steps": 2000},
        "rl": {"prompts": 1, "generations": 8, "mini_steps": 4, "prompt_tokens": 192, "completion_tokens": 256, "steps": 400},
        "repair": {"micro_batch": 4, "accumulation": 4, "seq_tokens": 1024, "steps": 1000},
        "probe": {"micro_batch": 32, "seq_tokens": 448},
        "gen_batch": 64,
    },
}


def _sft(p, key, params, steps=None):
    c = PROTOCOL[p][key]
    return b.SftWorkload(steps or c["steps"], c["micro_batch"], c["accumulation"], c["seq_tokens"], True, params)


def _rl(p, params, steps=None, **over):
    c = dict(PROTOCOL[p]["rl"], **over)
    return b.RlWorkload(steps or c["steps"], c["prompts"], c["generations"], c["mini_steps"],
                        c["prompt_tokens"], c["completion_tokens"], params, over.get("checkpointing", False))


def h(secs):
    return b.hours(secs)


def _scale(t, k):
    return tuple(x * k for x in t)


def _plus(*ts):
    return tuple(sum(t[i] for t in ts) for i in range(3))


def _gen(p, tasks, samples, prompt):
    c = PROTOCOL[p]
    return h(b.eval_seconds(tasks, samples, prompt, c["rl"]["completion_tokens"], c["gen_batch"], p))


def costs(p):
    """Per-seed hours (slowest, mid, fastest) of each building block on profile `p`."""
    s0 = h(b.sft_seconds(_sft(p, "sft", PERFT_PARAMS), p))
    s1 = h(b.rl_seconds(_rl(p, PERFT_PARAMS), p))
    s2 = h(b.sft_seconds(_sft(p, "repair", REPAIR_PARAMS), p))
    return {"S0": s0, "S1": s1, "S2": s2, "T": _plus(s0, s1, _scale(s2, 2))}


def ledger(p):
    """Rows (experiment, item, hours-per-seed (slow, mid, fast), seeds, note)."""
    c = costs(p)
    comp = PROTOCOL[p]["rl"]["completion_tokens"]
    rows = []
    rows.append(("T", "checkpoint production per seed: S0 PERFT-R SFT, S1 RLVR (1 arm), S2 repair experts (per-class + generic)", c["T"], SEEDS, "shared by F004-F006, C002, C003, C006"))
    rows.append(("T", "pilot allowance: 3 pilot seeds of T plus lr tuning (2 x S0)", _plus(_scale(c["T"], PILOT_SEEDS), _scale(c["S0"], 2)), 1, "seeds >= 1001 only"))
    f5 = _gen(p, int(0.5 * EVAL_TASKS * 1.25), 12, REPAIR_PROMPT_TOKENS)
    rows.append(("F005", "repair eval: 4 arms x <=3 rounds on ~50% failing candidates (+25% held-out classes)", _plus(f5, _gen(p, EVAL_TASKS, 1, EVAL_PROMPT_TOKENS)), SEEDS, "no training beyond T"))
    f4 = _gen(p, EVAL_TASKS, 35, 300)
    rows.append(("F004", "8 arms, 35 sampled candidates/task in total (B=4 primary; B=1 floor, B=8 sensitivity, BoN-rerank M=6)", f4, SEEDS, "no training beyond T"))
    f6 = _plus(c["S0"], _gen(p, 900, 1, EVAL_PROMPT_TOKENS))
    rows.append(("F006", "per arm: 4 domain stages (~1 S0 of training) + stage evals; 4 arms standalone", _scale(f6, 4), SEEDS, "2 arms (static, M013 heuristic) reusable from M013 if digest-pinned"))
    c2 = _plus(h(b.eval_seconds(6000, 1, EVAL_PROMPT_TOKENS, comp, PROTOCOL[p]["gen_batch"], p)), _gen(p, EVAL_TASKS, 1, EVAL_PROMPT_TOKENS), (0.3, 0.2, 0.1))
    rows.append(("C002", "trace extraction (6000 train + eval completions) + 5 probe arms (0.2 h)", c2, SEEDS, "probe training is small; extraction dominates"))
    c3 = _scale(_plus(h(b.rl_seconds(_rl(p, PERFT_PARAMS, steps=1000), p)), _gen(p, EVAL_TASKS, 1, EVAL_PROMPT_TOKENS)), 4)
    rows.append(("C003", "4 arms x (1000-step RLVR + eval)", c3, SEEDS, "starts from S0"))
    c5 = _scale(h(b.eval_seconds(12000, 1, 200, 100, PROTOCOL[p]["gen_batch"], p)), 3)
    rows.append(("C005", "no training: 12000 trials x 3 checkpoints, 100 new tokens", c5, SEEDS, "seed = trial assignment only"))
    c6 = _plus(c["S0"], _scale(_gen(p, 800, 6, 400), 4))
    rows.append(("C006", "3 domain Pod adapters (~1 S0) + 4 arms x 800 composite tasks x ~6 Pod samples", c6, SEEDS, "needs the composite-task generator"))
    m10 = h(b.rl_seconds(_rl(p, PERFT_PARAMS, steps=2000, prompts=4, prompt_tokens=256, completion_tokens=512, checkpointing=True), p))
    rows.append(("M010", "10 arms x 2000 optimizer steps, pinned 512-token protocol (time only: does NOT fit as is)", m10, 10 * SEEDS, "plus 8 tuning trials x 500 steps x 10 arms"))
    s0u = c["S0"]
    rows.append(("M011", "4 arms x (8 tuning trials x 500 + 2000 steps) per seed-equivalent of 14000 steps", _scale(s0u, 7), 4, "per arm: 7 S0-equivalents incl. 5 seeds + tuning"))
    rows.append(("M012", "3 arms x 14000 steps", _scale(s0u, 7), 3, "per arm"))
    rows.append(("M013", "5 arms x (1 S0-equivalent per seed) + tuning; step counts are not pinned by the manifest", _plus(_scale(s0u, SEEDS), _scale(s0u, 2)), 5, "per arm; assumption"))
    return rows


def _fmt(t):
    return f"{t[2]:.1f} / {t[1]:.1f} / {t[0]:.1f}"


def print_plan() -> None:
    for p in PROTOCOL:
        print(f"\n### {p}  (hours: fast / mid / slow)\n")
        print("| Exp | Item | Hours per seed (or unit) | Units | Total hours | Note |")
        print("|---|---|---|---|---|---|")
        for exp, item, hrs, n, note in ledger(p):
            print(f"| {exp} | {item} | {_fmt(hrs)} | {n} | {_fmt(_scale(hrs, n))} | {note} |")


if __name__ == "__main__":
    print_plan()
