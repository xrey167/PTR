"""C001 models (v2): shared-weight module bank, four communication schemes, variable module count.

Sensory modules share one encoder and one update network (the module count is free at test time);
module N is the reporter, which has no sensory input and from which the answer is read. A round is
communication followed by a per-module update. The schemes differ only in the communication step:

  workspace     S slots, competitive top-k write, gated slot update, broadcast read by every module
  all_to_all    full pairwise self-attention among all modules (no bottleneck)
  no_broadcast  the same write path into the same workspace, but only the reporter reads it back
  no_comm       no communication (a floor: the reporter never sees an input)

Arms are `ARM_SPECS` entries: the scheme plus the knobs that make the ablations (competition off =
soft write over all modules; capacity off = as many slots as training modules; bandwidth-matched
pairwise = value width reduced so that the values stored equal the slots' content; a crippled
all-to-all with one round, the sensitivity control). The update-network width is chosen per arm so
that parameter counts agree (`match_width`).
"""

from __future__ import annotations

import math

import torch
from torch import nn

SCHEMES = ("workspace", "all_to_all", "no_broadcast", "no_comm")

# name -> scheme and overrides of the configuration (slots, top_k, rounds, bw)
ARM_SPECS: dict[str, dict] = {
    "workspace": {"scheme": "workspace"},
    "all_to_all": {"scheme": "all_to_all"},
    "bw_all_to_all": {"scheme": "all_to_all", "bw": True},
    "no_broadcast": {"scheme": "no_broadcast"},
    "no_comm": {"scheme": "no_comm"},
    "competition_off": {"scheme": "workspace", "top_k": 10**6},
    "capacity_off": {"scheme": "workspace", "slots": "n_train"},
    "crippled_all_to_all": {"scheme": "all_to_all", "rounds": 1},
}


class Net(nn.Module):
    def __init__(
        self,
        scheme: str,
        d_in: int,
        d: int,
        ffn: int,
        rounds: int,
        n_out: int,
        slots: int,
        top_k: int,
        out_hidden: int,
        d_value: int | None = None,
    ) -> None:
        super().__init__()
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}")
        self.scheme, self.rounds, self.d = scheme, rounds, d
        self.slots, self.top_k = slots, top_k
        self.d_value = d_value or d
        self.encoder = nn.Linear(d_in, d)
        self.reporter_init = nn.Parameter(torch.randn(d) * 0.1)
        self.norm = nn.LayerNorm(d)
        self.up = nn.ModuleList([nn.Linear(d, ffn) for _ in range(rounds)])
        self.down = nn.ModuleList([nn.Linear(ffn, d) for _ in range(rounds)])
        if scheme in ("workspace", "no_broadcast"):
            self.slot_init = nn.Parameter(torch.randn(slots, d) * 0.1)
            self.wq, self.wk, self.wv = (nn.Linear(d, d) for _ in range(3))
            self.gru = nn.GRUCell(d, d)
            self.rq, self.rk, self.rv, self.ro = (nn.Linear(d, d) for _ in range(4))
        if scheme == "all_to_all":
            self.aq, self.ak = nn.Linear(d, d), nn.Linear(d, d)
            self.av, self.ao = nn.Linear(d, self.d_value), nn.Linear(self.d_value, d)
        self.head = nn.Sequential(nn.Linear(d, out_hidden), nn.GELU(), nn.Linear(out_hidden, n_out))
        self.trace: dict[str, torch.Tensor] = {}  # last forward's attention maps (detached), for the probes

    @property
    def n_sensory(self) -> int:
        return self._n

    # -- communication ------------------------------------------------------
    def _write(self, h, ws, alive):
        n = self._n
        q, k, v = self.wq(ws), self.wk(h[:, :n]), self.wv(h[:, :n])
        scores = torch.einsum("bsd,bmd->bsm", q, k) / math.sqrt(self.d)
        scores = scores.masked_fill(~alive[:, None, :n], float("-inf"))
        top = scores.topk(min(self.top_k, n), dim=-1)
        masked = torch.full_like(scores, float("-inf")).scatter(-1, top.indices, top.values)
        attn = torch.nan_to_num(torch.softmax(masked, dim=-1), nan=0.0)
        write = torch.einsum("bsm,bmd->bsd", attn, v)
        b, s, d = write.shape
        new = self.gru(write.reshape(b * s, d), ws.reshape(b * s, d)).reshape(b, s, d)
        return new, attn

    def _read(self, h, ws, only_reporter):
        q, k, v = self.rq(h), self.rk(ws), self.rv(ws)
        attn = torch.softmax(torch.einsum("bmd,bsd->bms", q, k) / math.sqrt(self.d), dim=-1)
        msg = self.ro(torch.einsum("bms,bsd->bmd", attn, v))
        if only_reporter:
            keep = torch.zeros(1, h.shape[1], 1, device=h.device)
            keep[0, -1, 0] = 1.0
            msg = msg * keep
        self.trace["read"] = attn.detach()
        return msg

    def _pairwise(self, h, alive):
        q, k, v = self.aq(h), self.ak(h), self.av(h)
        scores = torch.einsum("bmd,bnd->bmn", q, k) / math.sqrt(self.d)
        scores = scores.masked_fill(~alive[:, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        self.trace["pairwise"] = attn.detach()
        return self.ao(torch.einsum("bmn,bnd->bmd", attn, v))

    # -- forward ------------------------------------------------------------
    def forward(self, x, alive=None, states: bool = False):
        b, n = x.shape[0], x.shape[1]
        self._n = n
        if alive is None:
            alive = torch.ones(b, n + 1, dtype=torch.bool)
        h_s = torch.nn.functional.gelu(self.encoder(x * alive[:, :n, None]))
        h = torch.cat([h_s, self.reporter_init.expand(b, 1, self.d)], dim=1)
        ws = self.slot_init.expand(b, -1, -1) if hasattr(self, "slot_init") else None
        self.trace = {}
        for r in range(self.rounds):
            if self.scheme == "workspace":
                ws, attn = self._write(h, ws, alive)
                self.trace[f"write{r}"] = attn.detach()
                h = h + self._read(h, ws, only_reporter=False)
            elif self.scheme == "no_broadcast":
                ws, attn = self._write(h, ws, alive)
                self.trace[f"write{r}"] = attn.detach()
                h = h + self._read(h, ws, only_reporter=True)
            elif self.scheme == "all_to_all":
                h = h + self._pairwise(h, alive)
            h = h + self.down[r](torch.nn.functional.gelu(self.up[r](self.norm(h))))
        out = self.head(h[:, -1])
        return (out, h) if states else out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def arm_settings(name: str, cfg: dict) -> dict:
    """The scheme, slots, top_k, rounds and value width of arm `name` under `cfg`."""
    spec = ARM_SPECS[name]
    slots = cfg["n_train_modules"] if spec.get("slots") == "n_train" else spec.get("slots", cfg["slots"])
    d_value = None
    if spec.get("bw"):
        d_value = max(1, round(cfg["slots"] * cfg["d_model"] / cfg["n_train_modules"]))
    return {
        "scheme": spec["scheme"],
        "slots": slots,
        "top_k": spec.get("top_k", cfg["top_k"]),
        "rounds": spec.get("rounds", cfg["rounds"]),
        "d_value": d_value,
    }


def build(name: str, cfg: dict, ffn: int | None = None) -> Net:
    s = arm_settings(name, cfg)
    return Net(
        scheme=s["scheme"],
        d_in=cfg["d_in"],
        d=cfg["d_model"],
        ffn=cfg["ffn_workspace"] if ffn is None else ffn,
        rounds=s["rounds"],
        n_out=2 * cfg["n_values"] - 1,
        slots=s["slots"],
        top_k=s["top_k"],
        out_hidden=cfg["out_hidden"],
        d_value=s["d_value"],
    )


def match_width(name: str, cfg: dict, target: int) -> int:
    """The update-network width for arm `name` whose parameter count is closest to `target` (ties to the smaller)."""
    best = None
    for ffn in range(4, 6 * cfg["ffn_workspace"] + 1):
        diff = abs(count_params(build(name, cfg, ffn=ffn)) - target)
        if best is None or diff < best[0]:
            best = (diff, ffn)
    assert best is not None
    return best[1]


def comm_ops(name: str, cfg: dict, n_modules: int) -> int:
    """Attention score computations per round-forward and instance, an integer proxy of communication cost:
    workspace write + read = 2 * S * N; pairwise = (N + 1)^2; none = 0."""
    s = arm_settings(name, cfg)
    rounds = s["rounds"]
    if s["scheme"] == "workspace":
        per = 2 * s["slots"] * n_modules
    elif s["scheme"] == "no_broadcast":
        per = s["slots"] * n_modules + s["slots"]
    elif s["scheme"] == "all_to_all":
        per = (n_modules + 1) ** 2
    else:
        per = 0
    return per * rounds
