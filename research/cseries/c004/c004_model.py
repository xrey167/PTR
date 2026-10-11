"""C004 arms: a gated recurrent cell with a carried state, a causal feed-forward stack with unshared layers, and the same
recurrent net with the carried state reset at every step.

All three read a sequence of symbols and answer from the last position. Analytic parameter and FLOP counts are provided so that
the harness can match the arms and audit the match (multiply-add = 2 FLOPs; elementwise operations, layer norms and softmax are
left out of both counts and said so in the preregistration)."""

from __future__ import annotations

import math

import torch
from torch import nn

N_SYMBOLS = 6


class Recurrent(nn.Module):
    """Embedding, one GRU layer over the sequence, linear read-out of the last hidden state. With `reset=True` the carried state is
    reset to zero at every step (the GRU is applied to each step alone), so the read-out sees only the last step."""

    def __init__(self, d_embed: int, d_hidden: int, reset: bool = False):
        super().__init__()
        self.embed = nn.Embedding(N_SYMBOLS, d_embed)
        self.cell = nn.GRU(d_embed, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, N_SYMBOLS)
        self.reset = reset

    def forward_all(self, tokens: torch.Tensor) -> torch.Tensor:
        """The read-out after every step, (batch, steps, symbols)."""
        x = self.embed(tokens)
        b, t, e = x.shape
        if self.reset:
            out, _ = self.cell(x.reshape(b * t, 1, e))
            out = out.reshape(b, t, -1)
        else:
            out, _ = self.cell(x)
        return self.head(out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_all(tokens)[:, -1]


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.up, self.down = nn.Linear(d, ffn), nn.Linear(ffn, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).reshape(b, t, 3, h, d // h).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // h) + bias[..., :t, :t]
        attn = scores.softmax(dim=-1)
        x = x + self.out((attn @ v).transpose(1, 2).reshape(b, t, d))
        return x + self.down(torch.nn.functional.gelu(self.up(self.ln2(x))))


class Feedforward(nn.Module):
    """A stack of `layers` unshared causal self-attention blocks (pre-norm, GELU feed-forward, ALiBi relative-position bias, no
    absolute positions, so that it accepts sequences longer than any it was trained on; `slope_scale` multiplies the slopes (a steeper bias makes a head more local); `positional="none"` drops the bias and leaves only
    the causal mask), read out at the last position. `causal=False`
    removes the mask (only the negative control uses it)."""

    def __init__(self, d: int, heads: int, ffn: int, layers: int, causal: bool = True, positional: str = "alibi", slope_scale: float = 1.0):
        super().__init__()
        self.positional = positional
        self.embed = nn.Embedding(N_SYMBOLS, d)
        self.blocks = nn.ModuleList(Block(d, heads, ffn) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_SYMBOLS)
        self.causal = causal
        slopes = torch.tensor([slope_scale * 2.0 ** (-8.0 * (i + 1) / heads) for i in range(heads)])
        self.register_buffer("slopes", slopes, persistent=False)

    def bias(self, t: int) -> torch.Tensor:
        pos = torch.arange(t)
        distance = (pos[:, None] - pos[None, :]).clamp(min=0).float()
        bias = -self.slopes[:, None, None] * distance
        if self.positional == "none":
            bias = torch.zeros_like(bias)
        if self.causal:
            bias = bias.masked_fill(pos[None, :] > pos[:, None], float("-inf"))
        return bias[None]

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        bias = self.bias(tokens.shape[1])
        for block in self.blocks:
            x = block(x, bias)
        return self.norm(x)

    def forward_all(self, tokens: torch.Tensor) -> torch.Tensor:
        """The read-out at every position, (batch, steps, symbols); also what the causality probe compares."""
        return self.head(self.hidden(tokens))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_all(tokens)[:, -1]


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def feedforward_params(d: int, ffn: int, layers: int) -> int:
    """Analytic parameter count of `Feedforward` (a test pins it to the module)."""
    block = 2 * 2 * d + (3 * d * d + 3 * d) + (d * d + d) + (d * ffn + ffn) + (ffn * d + d)
    return N_SYMBOLS * d + layers * block + 2 * d + (d * N_SYMBOLS + N_SYMBOLS)


def recurrent_params(d_embed: int, d_hidden: int) -> int:
    """Analytic parameter count of `Recurrent` (a test pins it to the module)."""
    return N_SYMBOLS * d_embed + 3 * (d_hidden * d_embed + d_hidden * d_hidden + 2 * d_hidden) + d_hidden * N_SYMBOLS + N_SYMBOLS


def recurrent_flops(length: int, d_embed: int, d_hidden: int) -> int:
    """GRU: three gates, each an input and a hidden matrix product, per step; plus the read-out once."""
    return length * 6 * d_hidden * (d_embed + d_hidden) + 2 * d_hidden * N_SYMBOLS


def feedforward_flops(length: int, d: int, ffn: int, layers: int) -> int:
    """Per layer and token: qkv (6 d^2), output (2 d^2), feed-forward (4 d ffn); attention: scores and weighted sum over a causal
    context, 4 d per (query, key) pair, T (T + 1) / 2 pairs; plus the read-out once."""
    per_layer = length * (8 * d * d + 4 * d * ffn) + 2 * d * length * (length + 1)
    return layers * per_layer + 2 * d * N_SYMBOLS


def match_feedforward(target_params: int, target_flops: int, ref_length: int, layers: int, heads: int, tolerance_permille: int,
                      flop_tolerance_permille: int, widths=range(8, 129, 2)) -> tuple[int, int]:
    """(width, ffn) of the feed-forward arm: among the candidates whose parameter count is within the tolerance of the target (width
    a multiple of the head count), the one whose FLOPs at the reference length are closest to the target's; ties to the smaller.
    If no candidate meets the parameter tolerance, the closest in parameters (the harness's integrity counters then report it)."""
    best = None
    for d in widths:
        if d % heads:
            continue
        for f in range(d, 4 * d + 1, 2):
            params = feedforward_params(d, f, layers)
            flops = feedforward_flops(ref_length, d, f, layers)
            ok = 1000 * abs(params - target_params) <= tolerance_permille * target_params
            key = (0 if ok else 1, abs(flops - target_flops) if ok else abs(params - target_params), d, f)
            if best is None or key < best[0]:
                best = (key, (d, f))
    return best[1]
