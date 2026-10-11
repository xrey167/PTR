"""C008 arms: a token router with and without an attention schema, and the heuristic balancer.

The router maps a token's noisy features (and, in the schema arms, the schema's output) to a distribution over experts. The attention
schema is a small network that reads the router's OWN recent history (the router's propensity in each of the last `history` batches: its mean
softmax over experts before any control term) and predicts the router's own next propensity: which experts it will send tokens to and how many.
Both the history and the target are the uncontrolled propensity, so the schema's target does not depend on the control law it feeds (no loop). It is trained only by
that prediction loss (its output is detached before the router sees it). In the control arm the router adds a learned control law of the prediction to its
logits (a zero-initialised E x E matrix applied to the expected relative load); in the describe-only arm the schema is computed and trained
identically but the control law is given zeros in its place. The heuristic balancer is the
schema-free router plus an auxiliary-loss-free bias per expert, moved against the realised load of the previous batch.

FLOPs count multiply-adds as 2 and leave elementwise operations out; the schema's FLOPs are counted once per batch."""

from __future__ import annotations

import torch
from torch import nn

KINDS = ("schema_free", "schema_free_wide", "heuristic", "describe_only", "control", "history_router")


def law_params(form: str, experts: int) -> int:
    """Parameters of the control law: a full E x E matrix, one gain per expert, or a single gain."""
    return {"full": experts * experts, "diagonal": experts, "scalar": 1}[form]


def law_flops(form: str, experts: int) -> int:
    """Per batch: a matrix-vector product, a product with a vector, or a scaling."""
    return {"full": 2 * experts * experts, "diagonal": experts, "scalar": experts}[form]


class ControlLaw(nn.Module):
    """The map from the schema's expected relative load (E times its predicted distribution) to additive logits, zero at initialisation."""

    def __init__(self, form: str, experts: int):
        super().__init__()
        self.form = form
        self.weight = nn.Parameter(torch.zeros(experts, experts) if form == "full" else torch.zeros(experts) if form == "diagonal" else torch.zeros(1))

    def forward(self, relative_load: torch.Tensor) -> torch.Tensor:
        return relative_load @ self.weight.T if self.form == "full" else relative_load * self.weight


def router_params(d_in: int, hidden: int, experts: int) -> int:
    return d_in * hidden + hidden + hidden * experts + experts


def schema_params(experts: int, history: int, hidden: int) -> int:
    return experts * history * hidden + hidden + hidden * experts + experts


def router_flops(d_in: int, hidden: int, experts: int) -> int:
    """Per token."""
    return 2 * (d_in * hidden + hidden * experts)


def schema_flops(experts: int, history: int, hidden: int) -> int:
    """Per batch."""
    return 2 * (experts * history * hidden + hidden * experts)


def router_input_width(kind: str, cfg: dict) -> int:
    """The router's input: the token's features, and for the history router also the flattened history of the router's own propensity."""
    return cfg["experts"] * (1 + cfg["history"]) if kind == "history_router" else cfg["experts"]


def batch_flops(kind: str, cfg: dict, hidden: int) -> int:
    """Forward FLOPs of one batch of `tokens` tokens: the router per token, and per batch the schema and the control law."""
    e, h, b = cfg["experts"], cfg["history"], cfg["tokens"]
    uses_schema = kind in ("describe_only", "control")
    total = b * router_flops(router_input_width(kind, cfg), hidden, e)
    if uses_schema:
        total += schema_flops(e, h, cfg["schema_hidden"]) + law_flops(cfg["law_form"], e)
    return total


def total_params(kind: str, cfg: dict, hidden: int) -> int:
    e, h = cfg["experts"], cfg["history"]
    uses_schema = kind in ("describe_only", "control")
    total = router_params(router_input_width(kind, cfg), hidden, e)
    if uses_schema:
        total += schema_params(e, h, cfg["schema_hidden"]) + law_params(cfg["law_form"], e)
    return total


class Arm(nn.Module):
    """One routing arm. `forward(x, history, override)` returns (logits, prediction, base probabilities); `prediction` is None without a schema. `override`
    replaces the schema's output (used by the probes and by the shuffled-history control)."""

    def __init__(self, kind: str, cfg: dict, hidden: int):
        super().__init__()
        e = cfg["experts"]
        self.kind = kind
        self.experts = e
        self.schema_on = kind in ("describe_only", "control")
        self.feeds_router = kind == "control"
        self.reads_history = kind == "history_router"
        self.router = nn.Sequential(nn.Linear(router_input_width(kind, cfg), hidden), nn.GELU(), nn.Linear(hidden, e))
        if self.schema_on:
            self.schema = nn.Sequential(nn.Linear(e * cfg["history"], cfg["schema_hidden"]), nn.GELU(), nn.Linear(cfg["schema_hidden"], e))
            self.law = ControlLaw(cfg["law_form"], e)  # zero-initialised: a router that has not learned to use the schema is the schema-free one
        self.register_buffer("bias", torch.zeros(e))

    def predict(self, history: torch.Tensor) -> torch.Tensor:
        """The schema's prediction of the next batch's request fractions (a distribution over experts), from the flattened history."""
        return torch.softmax(self.schema(history), dim=-1)

    def forward(self, x: torch.Tensor, history: torch.Tensor, override: torch.Tensor | None = None):
        """(logits, prediction, base probabilities): the base probabilities are the router's softmax before any control term, detached; their batch mean is
        the router's propensity, which the history and the schema's target are made of."""
        prediction = None
        if self.reads_history:  # the router's own recent propensity, as relative loads (mean one), next to every token's features: history without a schema
            x = torch.cat([x, (self.experts * history).unsqueeze(0).expand(x.shape[0], -1)], dim=-1)
        base_logits = self.router(x)
        logits = base_logits
        if self.schema_on:
            prediction = self.predict(history) if override is None else override
            given = prediction.detach() if self.feeds_router else torch.zeros_like(prediction)
            logits = logits + self.law(self.experts * given)
        if self.kind == "heuristic":
            logits = logits + self.bias
        return logits, prediction, torch.softmax(base_logits, dim=-1).detach()

    def policy_parameters(self):
        return list(self.router.parameters()) + (list(self.law.parameters()) if self.schema_on else [])

    def schema_parameters(self):
        return list(self.schema.parameters()) if self.schema_on else []

    def update_bias(self, load: torch.Tensor, rate: float) -> None:
        """The auxiliary-loss-free balancer: move each expert's bias against its load error in the batch just routed (sign update)."""
        if self.kind == "heuristic":
            self.bias += rate * torch.sign(load.mean() - load)


def match_hidden(target: int, count, lo: int = 4, hi: int = 4096) -> int:
    """The hidden width whose `count(hidden)` is closest to `target`; ties to the smaller."""
    best = min(range(lo, hi + 1), key=lambda h: (abs(count(h) - target), h))
    return best
