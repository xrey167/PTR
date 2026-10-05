"""Routed PEFT arm topologies for experiment M011 (torch only; no PTR imports beyond metrics).

Four arms, each pinned in the M011 config (`arm_lora`, `arm_lora_computation`):

* `perft_r_emergent` / `perft_r_prescribed`: PERFT-R. N parallel bottleneck PEFT experts
  (`W_down` [H, r], `W_up` [r, H]) and a router [H, N] with top-k sit beside every OLMoE sparse
  MoE block; the output is added to the MoE block output, scaled by alpha / r. The base expert
  projections are frozen and not adapted. The prescribed arm masks the adapter router to the
  experts of the sequence's domain label before top-k (training and evaluation).
* `mixlora`: N LoRA experts as deltas on the fused expert projections (`gate_up_proj` delta
  A [H, r] / B [r, 2I], `down_proj` delta A [I, r] / B [r, H]). One delta set is shared by all
  base experts; a top-k router [H, N] over the block input picks the LoRA experts per token.
* `plain_lora`: PEFT `LoraConfig` on attention q/k/v/o (PEFT is used as is).

Everything except the adapters (and their routers) is frozen.

transformers 5.18 layout: `OlmoeSparseMoeBlock` holds `gate` (`OlmoeTopKRouter`) and `experts`
(`OlmoeExperts` with fused `gate_up_proj` [E, 2I, H] and `down_proj` [E, H, I]); its forward
takes `hidden_states [batch, seq, H]` and returns the block output `[batch, seq, H]`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------------------------
# Pinned arm specifications (mirror the M011 `arm_lora` table).
# ----------------------------------------------------------------------------------------------

PRESCRIBED_ASSIGNMENT: dict[str, list[int]] = {
    "code": [0, 1, 2],
    "math": [3, 4, 5],
    "tool": [6, 7],
}
DOMAINS: tuple[str, ...] = ("code", "math", "tool")
# Pinned in the config as `expert_domain_assignment_expected_digest`.
PRESCRIBED_ASSIGNMENT_DIGEST = (
    "sha256:b9028e88fa67731865af758a396194920d13520504ef746198837cbab31035b4"
)

# Real OLMoE-1B-7B-0125 shapes used by the pinned arithmetic.
OLMOE_SHAPES = {"hidden": 2048, "intermediate": 1024, "layers": 16, "experts": 64}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    method: str  # perft_r | mixlora | lora
    specialisation: str  # emergent | prescribed | none
    num_experts: int
    router_top_k: int
    rank: int
    alpha: int
    pinned_trainable_params: int  # for OLMOE_SHAPES

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def masked(self) -> bool:
        return self.specialisation == "prescribed"


ARMS: dict[str, ArmSpec] = {
    "perft_r_emergent": ArmSpec("perft_r_emergent", "perft_r", "emergent", 8, 2, 8, 16, 4456448),
    "perft_r_prescribed": ArmSpec(
        "perft_r_prescribed", "perft_r", "prescribed", 8, 2, 8, 16, 4456448
    ),
    "plain_lora": ArmSpec("plain_lora", "lora", "none", 1, 1, 17, 34, 4456448),
    "mixlora": ArmSpec("mixlora", "mixlora", "none", 9, 2, 4, 8, 4423680),
}
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
COMMON_TARGET = 4456448
RELATIVE_TOLERANCE = 0.01

# Float32 min is finite: masked logits stay finite for `routing_metrics.check_finite` while the
# softmax weight of a masked expert is exactly 0.
_MASK_FILL = float(torch.finfo(torch.float32).min)


def assignment_digest(assignment: dict[str, Sequence[int]]) -> str:
    """sha256 of canonical JSON (sorted keys, no spaces) of the expert-to-domain assignment."""
    canon = json.dumps(
        {k: [int(x) for x in v] for k, v in assignment.items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def expected_pinned_params(arm: ArmSpec, hidden: int, intermediate: int, layers: int) -> int:
    """Closed-form trainable-parameter count (the config arithmetic) for given shapes."""
    n, r = arm.num_experts, arm.rank
    if arm.method == "perft_r":
        per_layer = n * (hidden * r + r * hidden) + hidden * n
    elif arm.method == "mixlora":
        per_expert = r * (hidden + 2 * intermediate) + r * (intermediate + hidden)
        per_layer = n * per_expert + hidden * n
    elif arm.method == "lora":
        per_layer = 4 * r * (hidden + hidden)
    else:  # pragma: no cover - exhaustive
        raise ValueError(arm.method)
    return per_layer * layers


# ----------------------------------------------------------------------------------------------
# Routing: top-k with optional domain mask.
# ----------------------------------------------------------------------------------------------


def route_topk(logits: torch.Tensor, k: int, allowed: torch.Tensor | None = None):
    """Top-k routing over `logits` [T, N]; returns (gates [T, N], index [T, k], masked_logits).

    `allowed` [T, N] (bool) restricts the choice: logits outside it are replaced by the
    float32 minimum *before* top-k, so an out-of-domain expert is never selected as long as
    every row has at least k allowed experts (checked). Gate weights are the softmax of the
    selected logits (renormalised over the k chosen experts); `gates` is that weight scattered
    into a dense [T, N] matrix (zero for unselected experts).
    """
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError(f"k={k} must be in 1..{logits.shape[-1]}")
    logits = logits.float()
    if allowed is not None:
        if allowed.shape != logits.shape:
            raise ValueError(f"allowed {tuple(allowed.shape)} != logits {tuple(logits.shape)}")
        if int(allowed.sum(dim=-1).min()) < k:
            raise ValueError("every token must have at least top_k allowed experts")
        logits = logits.masked_fill(~allowed, _MASK_FILL)
    top_val, top_idx = torch.topk(logits, k, dim=-1)
    weights = torch.softmax(top_val, dim=-1)
    gates = torch.zeros_like(logits).scatter(-1, top_idx, weights)
    return gates, top_idx, logits


class RoutingContext:
    """Shared state between the training/eval loop and the installed adapter modules.

    `set_domains(labels)` gives one domain label per sequence of the next forward pass; the
    prescribed arm builds its expert mask from it. A masked context with no labels set raises
    instead of silently routing unmasked.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        assignment: dict[str, Sequence[int]] | None,
        domains: Sequence[str] = DOMAINS,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.domains = tuple(domains)
        self.assignment = None if assignment is None else {k: list(v) for k, v in assignment.items()}
        self._domain_index: torch.Tensor | None = None
        self.recording = False
        self.records: dict[int, list[torch.Tensor]] = {}
        self._table: torch.Tensor | None = None
        if self.assignment is not None:
            table = torch.zeros(len(self.domains), num_experts, dtype=torch.bool)
            for d, name in enumerate(self.domains):
                ids = self.assignment[name]
                if len(set(ids)) != len(ids) or any(not 0 <= e < num_experts for e in ids):
                    raise ValueError(f"bad expert ids for domain {name}: {ids}")
                if len(ids) < top_k:
                    raise ValueError(f"domain {name} has fewer than top_k={top_k} experts")
                table[d, ids] = True
            self._table = table

    @property
    def masked(self) -> bool:
        return self._table is not None

    def set_domains(self, labels: Sequence[str] | None) -> None:
        if labels is None:
            self._domain_index = None
            return
        try:
            idx = [self.domains.index(x) for x in labels]
        except ValueError as error:
            raise ValueError(f"unknown domain label in {list(labels)}") from error
        self._domain_index = torch.tensor(idx, dtype=torch.long)

    def allowed_mask(self, batch: int, seq: int, device) -> torch.Tensor | None:
        """[batch*seq, N] bool mask (row-major over batch, then seq) or None if unmasked."""
        if self._table is None:
            return None
        if self._domain_index is None:
            raise RuntimeError("masked routing requires RoutingContext.set_domains() per forward")
        if self._domain_index.shape[0] != batch:
            raise ValueError(
                f"{self._domain_index.shape[0]} domain labels for a batch of {batch} sequences"
            )
        table = self._table.to(device)
        rows = table[self._domain_index.to(device)]  # [B, N]
        return rows.repeat_interleave(seq, dim=0)

    def record(self, layer: int, logits: torch.Tensor) -> None:
        if self.recording:
            self.records.setdefault(layer, []).append(logits.detach().float().cpu())

    def take_records(self) -> list[list[list[float]]]:
        """Recorded adapter-router logits as `[layer][token][expert]` nested lists; clears them."""
        out = [torch.cat(self.records[i], 0).tolist() for i in sorted(self.records)]
        self.records = {}
        return out


# ----------------------------------------------------------------------------------------------
# PERFT-R
# ----------------------------------------------------------------------------------------------


def _activation(name: str):
    return {"silu": F.silu, "relu": F.relu, "gelu": F.gelu, "identity": lambda x: x}[name]


class PerftRBlock(nn.Module):
    """`base MoE block output + alpha/r * sum_{i in topk} g_i * W_up_i act(W_down_i x)`."""

    def __init__(
        self,
        base: nn.Module,
        spec: ArmSpec,
        ctx: RoutingContext,
        layer: int,
        activation: str = "silu",
    ):
        super().__init__()
        hidden = base.experts.hidden_dim
        device = base.experts.gate_up_proj.device
        n, r = spec.num_experts, spec.rank
        self.base = base
        self.spec, self.layer, self.activation = spec, layer, activation
        # The context is shared by every layer; keep it out of the module tree.
        object.__setattr__(self, "ctx", ctx)
        self.peft_expert_w_down = nn.Parameter(torch.empty(n, hidden, r, device=device))
        self.peft_expert_w_up = nn.Parameter(torch.empty(n, r, hidden, device=device))
        self.peft_router = nn.Parameter(torch.empty(hidden, n, device=device))
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self) -> None:
        with torch.no_grad():
            for i in range(self.spec.num_experts):
                nn.init.kaiming_uniform_(self.peft_expert_w_down[i].T, a=math.sqrt(5))
            self.peft_expert_w_up.zero_()  # adapter starts as an exact no-op
            nn.init.normal_(self.peft_router, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.base(hidden_states)
        batch, seq, hidden = hidden_states.shape
        x = hidden_states.reshape(-1, hidden).float()
        logits = x @ self.peft_router
        allowed = self.ctx.allowed_mask(batch, seq, hidden_states.device) if self.spec.masked else None
        gates, _, masked_logits = route_topk(logits, self.spec.router_top_k, allowed)
        self.ctx.record(self.layer, masked_logits)
        down = torch.einsum("th,nhr->tnr", x, self.peft_expert_w_down)
        down = _activation(self.activation)(down) * gates.unsqueeze(-1)
        delta = torch.einsum("tnr,nrh->th", down, self.peft_expert_w_up) * self.spec.scaling
        return out + delta.reshape(batch, seq, hidden).to(out.dtype)


# ----------------------------------------------------------------------------------------------
# MixLoRA style
# ----------------------------------------------------------------------------------------------


class MixLoraBlock(nn.Module):
    """OLMoE sparse block whose expert projections carry top-k routed LoRA deltas.

    Replicates `OlmoeSparseMoeBlock.forward` + `OlmoeExperts.forward` (explicit per-expert loop)
    and adds `alpha/r * sum_n g_n (x A_n) B_n` to both fused projections of every base expert
    a token visits. One delta set (N LoRA experts) is shared across all base experts.
    """

    def __init__(self, base: nn.Module, spec: ArmSpec, ctx: RoutingContext, layer: int):
        super().__init__()
        e = base.experts
        h, i = e.hidden_dim, e.intermediate_dim
        device = e.gate_up_proj.device
        n, r = spec.num_experts, spec.rank
        self.base = base
        self.spec, self.layer = spec, layer
        object.__setattr__(self, "ctx", ctx)
        self.expert_gate_up_delta_a = nn.Parameter(torch.empty(n, h, r, device=device))
        self.expert_gate_up_delta_b = nn.Parameter(torch.empty(n, r, 2 * i, device=device))
        self.expert_down_delta_a = nn.Parameter(torch.empty(n, i, r, device=device))
        self.expert_down_delta_b = nn.Parameter(torch.empty(n, r, h, device=device))
        self.lora_router = nn.Parameter(torch.empty(h, n, device=device))
        self.reset_adapter_parameters()

    def reset_adapter_parameters(self) -> None:
        with torch.no_grad():
            for a in (self.expert_gate_up_delta_a, self.expert_down_delta_a):
                for j in range(a.shape[0]):
                    nn.init.kaiming_uniform_(a[j].T, a=math.sqrt(5))
            self.expert_gate_up_delta_b.zero_()
            self.expert_down_delta_b.zero_()
            nn.init.normal_(self.lora_router, std=0.02)

    @staticmethod
    def _delta(x, gates, a, b, scaling):
        xa = torch.einsum("th,nhr->tnr", x.float(), a) * gates.unsqueeze(-1)
        return torch.einsum("tnr,nrk->tk", xa, b) * scaling

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        _, top_k_weights, top_k_index = self.base.gate(flat)
        # LoRA-expert routing on the block input (adapter router, unmasked: it is not a domain arm).
        lora_logits = flat.float() @ self.lora_router
        lora_gates, _, lora_logits = route_topk(lora_logits, self.spec.router_top_k, None)
        self.ctx.record(self.layer, lora_logits)

        experts = self.base.experts
        scaling = self.spec.scaling
        out = torch.zeros_like(flat)
        for expert in torch.unique(top_k_index).tolist():
            if expert >= experts.num_experts:
                continue
            tok, pos = torch.where(top_k_index == expert)
            x = flat[tok]
            g = lora_gates[tok]
            gu = F.linear(x, experts.gate_up_proj[expert])
            gu = gu + self._delta(
                x, g, self.expert_gate_up_delta_a, self.expert_gate_up_delta_b, scaling
            ).to(gu.dtype)
            gate, up = gu.chunk(2, dim=-1)
            h = experts.act_fn(gate) * up
            y = F.linear(h, experts.down_proj[expert])
            y = y + self._delta(
                h, g, self.expert_down_delta_a, self.expert_down_delta_b, scaling
            ).to(y.dtype)
            y = y * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out.reshape(batch, seq, hidden)


# ----------------------------------------------------------------------------------------------
# Installation, freezing, counting
# ----------------------------------------------------------------------------------------------


def moe_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """(qualified name, module) of every OLMoE sparse MoE block, in layer order."""
    out = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "OlmoeSparseMoeBlock"]
    if not out:
        raise ValueError("no OlmoeSparseMoeBlock found (not an OLMoE model, or already wrapped)")
    return out


@dataclass
class InstalledArm:
    model: nn.Module
    spec: ArmSpec
    ctx: RoutingContext | None  # None for plain LoRA (no adapter router)

    @property
    def has_adapter_router(self) -> bool:
        return self.ctx is not None


def install_arm(model: nn.Module, arm: str | ArmSpec, *, activation: str = "silu") -> InstalledArm:
    """Install an arm's adapters on an OLMoE model and freeze everything else.

    Returns the (possibly PEFT-wrapped) model with the routing context. For the PERFT-R and
    MixLoRA arms the MoE blocks are replaced by wrappers holding the original block as `.base`.
    """
    spec = ARMS[arm] if isinstance(arm, str) else arm
    for p in model.parameters():
        p.requires_grad_(False)
    if spec.method == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=spec.rank,
            lora_alpha=spec.alpha,
            lora_dropout=0.0,
            target_modules=list(LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        )
        return InstalledArm(get_peft_model(model, cfg), spec, None)

    assignment = PRESCRIBED_ASSIGNMENT if spec.masked else None
    ctx = RoutingContext(
        num_experts=spec.num_experts, top_k=spec.router_top_k, assignment=assignment
    )
    for layer, (name, block) in enumerate(moe_blocks(model)):
        wrapper = (
            PerftRBlock(block, spec, ctx, layer, activation)
            if spec.method == "perft_r"
            else MixLoraBlock(block, spec, ctx, layer)
        )
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, wrapper)
    # Wrappers created their own parameters trainable; the base block stays frozen.
    for _, module in model.named_modules():
        if isinstance(module, (PerftRBlock, MixLoraBlock)):
            for pname, p in module.named_parameters(recurse=False):
                p.requires_grad_(True)
    return InstalledArm(model, spec, ctx)


def count_trainable_parameters(model: nn.Module) -> int:
    """Number of parameters with `requires_grad` in the instantiated modules."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_inventory(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """name -> shape for every trainable tensor."""
    return {n: tuple(p.shape) for n, p in model.named_parameters() if p.requires_grad}


def non_adapter_trainable(model: nn.Module) -> list[str]:
    """Names of trainable parameters that are not adapter/router parameters (must be empty)."""
    markers = (
        "peft_expert_w_down",
        "peft_expert_w_up",
        "peft_router",
        "expert_gate_up_delta_",
        "expert_down_delta_",
        "lora_router",
        "lora_A",
        "lora_B",
    )
    return [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and not any(m in n for m in markers)
    ]


def within_tolerance(count: int, target: int = COMMON_TARGET, tol: float = RELATIVE_TOLERANCE):
    return abs(count - target) / target <= tol


__all__ = [
    "ARMS",
    "ArmSpec",
    "DOMAINS",
    "InstalledArm",
    "MixLoraBlock",
    "PRESCRIBED_ASSIGNMENT",
    "PerftRBlock",
    "RoutingContext",
    "assignment_digest",
    "count_trainable_parameters",
    "expected_pinned_params",
    "install_arm",
    "moe_blocks",
    "non_adapter_trainable",
    "route_topk",
    "trainable_inventory",
    "within_tolerance",
]
