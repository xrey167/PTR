"""Round 7: bounded diagnosis of the workspace's training failure on pilot seed 1011 (diagnostic only; nothing here is adopted).

    python cause1011.py OUT.jsonl            # deterministic re-run, then the same data with other initialisations

Prints/writes per run: validation curve, accuracy on 8/16/24 modules, accuracy by query type on 16 modules, final step. Pilot seed only."""
import json
import sys
from pathlib import Path

GW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GW))
import harness as H  # noqa: E402

H._load_torch()
torch = H.torch
torch.set_num_threads(1)
SEED = 1011


def run(tag, init_salt=None, lr=0.003):
    cfg = H.load_config()
    cfg["arm_d"] = {k: 48 for k in cfg["arm_d"]}
    n = cfg["n_train_modules"]
    train = H.make_batch(torch.Generator().manual_seed(SEED * 1000 + 17), cfg["train_size"], n, cfg)
    gens = [torch.Generator().manual_seed(SEED * 1000 + 500 + i) for i in range(4)]
    id_b, o16, o24 = (H.make_batch(g, cfg["eval_size"], m, cfg) for g, m in zip(gens, (n, 16, 24)))
    val = H.make_batch(gens[3], cfg["curve_eval_size"], n, cfg)
    original = H.arm_seed
    if init_salt is not None:
        H.arm_seed = lambda seed, name: original(seed, f"{name}#{init_salt}")
    try:
        net, info, _, _ = H.train_arm("workspace", cfg, SEED, train, val, cfg["ffn_workspace"], lr, set())
    finally:
        H.arm_seed = original
    row = {"tag": tag, "lr": lr, "init_salt": init_salt, "steps": info["steps"], "converged": info["converged"], "best_val": info["best_val"],
           "id": H.permille(*H.evaluate(net, id_b)), "ood16": H.permille(*H.evaluate(net, o16)), "ood24": H.permille(*H.evaluate(net, o24)),
           "curve": info["curve"]}
    return row


if __name__ == "__main__":
    jobs = [("rerun-same-init", None, 0.003), ("init-1", 1, 0.003), ("init-2", 2, 0.003), ("init-3", 3, 0.003)]
    for tag, salt, lr in jobs:
        row = run(tag, salt, lr)
        with open(sys.argv[1], "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(tag, row["id"], row["ood16"], row["steps"], row["curve"][:6], "...", row["curve"][-3:], flush=True)
