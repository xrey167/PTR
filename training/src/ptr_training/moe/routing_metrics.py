"""Routing metrics over plain Python lists (no torch, no numpy).

A routing snapshot is `snapshot[layer][token] -> list[float]` of router logits over
experts. All metrics are deterministic functions of those lists.
"""

from __future__ import annotations

import math

Logits = list[float]
LayerLogits = list[Logits]  # [token][expert]


def top_k_set(logits: Logits, k: int) -> frozenset[int]:
    """Indices of the k largest logits; ties break toward the lower expert index."""
    if not 1 <= k <= len(logits):
        raise ValueError(f"k={k} must be in 1..{len(logits)}")
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return frozenset(order[:k])


def top_k_sets(layer: LayerLogits, k: int) -> list[frozenset[int]]:
    return [top_k_set(row, k) for row in layer]


def expert_churn(before: LayerLogits, after: LayerLogits, k: int) -> float:
    """Fraction of tokens whose top-k expert set differs between two snapshots."""
    if len(before) != len(after):
        raise ValueError("snapshots must cover the same tokens")
    if not before:
        raise ValueError("empty snapshot")
    changed = sum(1 for a, b in zip(top_k_sets(before, k), top_k_sets(after, k)) if a != b)
    return changed / len(before)


def softmax(logits: Logits) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def routing_entropy(layer: LayerLogits) -> float:
    """Mean per-token entropy (nats) of the softmax router distribution."""
    if not layer:
        raise ValueError("empty snapshot")
    total = 0.0
    for row in layer:
        total -= sum(p * math.log(p) for p in softmax(row) if p > 0.0)
    return total / len(layer)


def expert_load(layer: LayerLogits, k: int) -> list[int]:
    """Number of (token, slot) assignments each expert receives under top-k routing."""
    if not layer:
        raise ValueError("empty snapshot")
    counts = [0] * len(layer[0])
    for chosen in top_k_sets(layer, k):
        for e in chosen:
            counts[e] += 1
    return counts


def load_cv(layer: LayerLogits, k: int) -> float:
    """Coefficient of variation (population std / mean) of expert load; 0 is perfectly balanced."""
    load = expert_load(layer, k)
    mean = sum(load) / len(load)
    if mean == 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in load) / len(load)
    return math.sqrt(var) / mean


def dead_experts(layer: LayerLogits, k: int) -> int:
    """Experts that receive zero tokens under top-k routing."""
    return sum(1 for x in expert_load(layer, k) if x == 0)


def layer_metrics(layer: LayerLogits, k: int) -> dict:
    return {
        "entropy": routing_entropy(layer),
        "load_cv": load_cv(layer, k),
        "dead_experts": dead_experts(layer, k),
        "load": expert_load(layer, k),
    }


def snapshot_metrics(snapshot: list[LayerLogits], k: int) -> list[dict]:
    return [layer_metrics(layer, k) for layer in snapshot]


def snapshot_churn(before: list[LayerLogits], after: list[LayerLogits], k: int) -> list[float]:
    if len(before) != len(after):
        raise ValueError("snapshots must have the same number of layers")
    return [expert_churn(b, a, k) for b, a in zip(before, after)]


def router_logits_to_lists(router_logits, attention_mask=None) -> list[LayerLogits]:
    """Torch adapter: tuple of per-layer `[batch*seq, experts]` tensors -> nested lists.

    `attention_mask` (`[batch, seq]`) drops padding tokens. torch is only touched through
    the tensors passed in, so this module stays importable without torch.
    """
    keep = None
    if attention_mask is not None:
        keep = [bool(x) for x in attention_mask.reshape(-1).tolist()]
    out: list[LayerLogits] = []
    for layer in router_logits:
        rows = layer.detach().float().cpu().tolist()
        if keep is not None:
            if len(keep) != len(rows):
                raise ValueError("attention_mask does not match router logits token count")
            rows = [r for r, kp in zip(rows, keep) if kp]
        out.append(rows)
    return out
