"""What can a router gain from knowing the demand regime, with no learning at all? Used to set the planning alternative of C008.

    python research/cseries/c003a/pilots/regime_value.py

For regime 0 of the pinned world: the verified-token rate of a router that splits every class evenly over its two competent experts (the best a
regime-blind router can do on average), of the best regime-aware independent split (the two hot classes get their own split; every other class splits
evenly), and of the per-batch oracle (a maximum flow, which also coordinates tokens inside a batch). Independent token sampling cannot reach the oracle;
the difference between the first two numbers is the most a schema that revealed the regime could add to a router that samples each token alone."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402

import c008_task as task  # noqa: E402

cfg = tomllib.loads((ROOT / "experiments/consciousness/C008-attention-schema-routing-cpu/config.toml").read_text(encoding="utf-8"))["pinned_thresholds"]
gen = torch.Generator().manual_seed(1)
B, E, CAP = cfg["tokens"], cfg["experts"], cfg["capacity"]
batches = [task.sample_batch(gen, 0, cfg)[0] for _ in range(60)]  # regime 0: hot classes 0 and 1


def value(a: float, b: float) -> float:
    total = 0
    for classes in batches:
        p = torch.full((B,), 0.5)
        p[classes == 0], p[classes == 1] = a, b
        choices = torch.where(torch.rand(B, generator=gen) < p, classes, (classes + 1) % E)
        total += int(task.serve(classes, choices, E, CAP).sum())
    return total / len(batches) / B


blind = value(0.5, 0.5)
best = max((value(a, b), a, b) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0) for b in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
oracle = sum(task.oracle_reward(classes, E, CAP) for classes in batches) / len(batches) / B
print(f"even split {blind:.3f}; best regime-aware independent split {best[0]:.3f} (class 0 primary {best[1]}, class 1 primary {best[2]}); oracle {oracle:.3f}; "
      f"reachable gain {1000 * (best[0] - blind):.0f} per mille, oracle gain {1000 * (oracle - blind):.0f} per mille")
