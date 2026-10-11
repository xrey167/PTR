"""C007 models: a two-layer mixture-of-experts classifier (the model whose reliability is predicted), its trace, and the self-model, probes and
confidence baselines that read the trace.

The classifier is trained once and then FROZEN; nothing below may change its parameters. The trace of an input is what the model exposes
about its own computation: the router probabilities and selections of both layers, the hidden state after each layer, the output
probabilities and their entropy. A `Predictor` maps a trace view to the probability that the model's answer is wrong. The SELF-MODEL and the
information-matched PROBE are the same network on the same trace, with the same optimiser and data; they differ in one pinned number, the
weight of the auxiliary self-modelling loss (the model predicts its own layer-two expert choice and its own output entropy from the
trace with the layer-two part masked): the self-model's is positive, the probe's is zero. Nothing here reads or trains on text."""

from __future__ import annotations

import math

import torch
from torch import nn

NATIVE_EPS = 1e-6


class MoELayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, experts: int, top_k: int):
        super().__init__()
        self.router = nn.Linear(d_in, experts)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d_in, d_out), nn.GELU()) for _ in range(experts))
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        probs = torch.softmax(self.router(x), dim=-1)
        top = probs.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(probs).scatter(-1, top.indices, 1.0)
        weights = probs * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(NATIVE_EPS)
        out = sum(weights[:, e : e + 1] * expert(x) for e, expert in enumerate(self.experts))
        return out, probs, mask


class Classifier(nn.Module):
    """x -> MoE layer 1 -> MoE layer 2 -> linear head. `forward` returns the logits; `trace` returns them with everything the model exposes."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, h, e, k = cfg["dim"], cfg["hidden"], cfg["experts"], cfg["classes"]
        self.layer1 = MoELayer(d, h, e, cfg["top_k"])
        self.layer2 = MoELayer(h, h, e, cfg["top_k"])
        self.head = nn.Linear(h, k)

    def trace(self, x: torch.Tensor) -> dict:
        h1, p1, m1 = self.layer1(x)
        h2, p2, m2 = self.layer2(h1)
        logits = self.head(h2)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(NATIVE_EPS).log()).sum(-1, keepdim=True)
        return {"logits": logits, "p1": p1, "m1": m1, "h1": h1, "p2": p2, "m2": m2, "h2": h2, "probs": probs, "entropy": entropy}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trace(x)["logits"]


# the parts of a trace, in the order of the feature vector; "late" is what layer two and the output add to what layer one has already shown
EARLY = ("p1", "m1", "h1")
LATE = ("p2", "m2", "h2", "probs", "entropy")


def features(trace: dict, parts: tuple) -> torch.Tensor:
    return torch.cat([trace[p] for p in parts], dim=-1)


def feature_width(cfg: dict, parts: tuple) -> int:
    e, h, k = cfg["experts"], cfg["hidden"], cfg["classes"]
    widths = {"p1": e, "m1": e, "h1": h, "p2": e, "m2": e, "h2": h, "probs": k, "entropy": 1}
    return sum(widths[p] for p in parts)


class Predictor(nn.Module):
    """A trunk over a feature vector with three heads: the probability of an error (the judged output), and two auxiliary heads that read the trunk
    of the trace with its late part masked out: the layer-two top-1 expert and the output entropy (self-modelling targets)."""

    def __init__(self, width: int, late_width: int, cfg: dict, base: tuple | None = None):
        super().__init__()
        self.base = None if base is None else (float(base[0]), float(base[1]))
        self.trunk = nn.Sequential(nn.Linear(width, cfg["probe_hidden"]), nn.GELU(), nn.Linear(cfg["probe_hidden"], cfg["probe_hidden2"]), nn.GELU())
        self.error = nn.Linear(cfg["probe_hidden2"], 1)
        self.expert = nn.Linear(cfg["probe_hidden2"], cfg["experts"])
        self.entropy = nn.Linear(cfg["probe_hidden2"], 1)
        self.late_width = late_width

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """The logit of the probability of an error. With a `base` (a, b) it is a * z + b + the network's output, z being the LAST column of the view (the
        classifier's own failure score): the network then learns what it can add to the classifier's own calibrated confidence."""
        out = self.error(self.trunk(v)).squeeze(-1)
        return out if self.base is None else out + self.base[0] * v[:, -1] + self.base[1]

    def auxiliary(self, v: torch.Tensor):
        """Heads on the masked view (the last `late_width` features zeroed)."""
        masked = torch.cat([v[:, : v.shape[1] - self.late_width], torch.zeros_like(v[:, v.shape[1] - self.late_width :])], dim=-1)
        z = self.trunk(masked)
        return self.expert(z), self.entropy(z).squeeze(-1)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def params_digest(net: nn.Module) -> str:
    """A digest of every parameter and buffer value of `net`: it changes if and only if some weight does."""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode())
        digest.update(repr(tensor.detach().flatten().tolist()).encode())
    return digest.hexdigest()


def platt_fit(z: torch.Tensor, y: torch.Tensor, steps: int = 400) -> tuple[float, float]:
    """Logistic calibration p = sigmoid(a z + b) of a one-dimensional score, fitted by full-batch gradient descent on the log loss."""
    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.05)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(a), float(b)


def confidence_score(probs: torch.Tensor) -> torch.Tensor:
    """logit of the model's own failure probability, 1 - max softmax (clipped)."""
    p = (1.0 - probs.max(dim=-1).values).clamp(NATIVE_EPS, 1 - NATIVE_EPS)
    return torch.log(p / (1 - p))


# -- metrics (also recomputed by a plain loop for the harness's audit) ---------------------------------

def brier_sum_micro(p: torch.Tensor, y: torch.Tensor) -> int:
    """The sum over the examples of the squared error, in millionths. A non-finite prediction counts as the worst possible error (one per example), so a
    head with non-finite weights is scored, not a crash; the harness counts it (nonfinite_arms)."""
    total = float(((p.double() - y.double()) ** 2).sum())
    return int(round(1e6 * total)) if math.isfinite(total) else int(y.numel()) * 1_000_000


def brier_reference_micro(p: list, y: list) -> int:
    return int(round(1e6 * sum((a - b) ** 2 for a, b in zip(p, y))))


def auroc_permille(p: torch.Tensor, y: torch.Tensor) -> int:
    """Probability that a wrong answer gets a higher failure probability than a right one (ties count half), per mille."""
    pos, neg = p[y == 1], p[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 500
    greater = (pos[:, None] > neg[None, :]).sum().item()
    ties = (pos[:, None] == neg[None, :]).sum().item()
    return int(round(1000 * (greater + 0.5 * ties) / (pos.numel() * neg.numel())))


def ece_permille(p: torch.Tensor, y: torch.Tensor, bins: int = 15) -> int:
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if float(lo) > 0 else (p >= lo) & (p <= hi)
        if mask.any():
            total += float(mask.sum()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return int(round(1000 * total / p.numel()))


def risk_coverage_area_permille(p: torch.Tensor, y: torch.Tensor) -> int:
    """Mean selective risk over coverage levels when the answers with the lowest predicted failure are kept first, per mille (lower is better)."""
    order = p.argsort()
    cumulative = y[order].cumsum(0) / torch.arange(1, y.numel() + 1)
    return int(round(1000 * float(cumulative.mean())))
