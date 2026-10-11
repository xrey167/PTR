"""C008 world: routing under a bursty, cyclic demand with limited expert capacity, and its verifier.

E experts, B tokens per batch. A token has a class c in 0..E-1; the experts competent for it are c and (c + 1) mod E. A token is
rewarded (verified) iff the expert it was routed to is competent for it AND the expert has room: each expert serves the first `cap` tokens
routed to it in the batch's arrival order and drops the rest. Demand is bursty and cyclic: the stream passes through R regimes of
`dwell` batches each, in a fixed cycle; in regime r the classes r and r + 1 (modulo E) carry `hot_permille`/1000 of the tokens and the
others share the rest. Every class is the "first" hot class in one regime and the "second" in the next, and the best split of its tokens over its two
competent experts is the opposite in the two: a router that cannot tell the regimes apart cannot be right in both. So the load on the experts depends on the position in a cycle that a router seeing one token at a time cannot
observe, but that its own past loads reveal. In the stationary (null) world the regime never changes.

`oracle_reward` is the largest number of verified tokens any assignment can reach (a maximum flow), the ceiling that shows how much
headroom the arms have. `serve_reference` recomputes the verifier by a plain loop, for the harness's audit of the vectorised one.
Everything is a pure function of a torch Generator and the batch index."""

from __future__ import annotations

import torch


def regime_of(batch_index: int, regimes: int, dwell: int, stationary: bool) -> int:
    return 0 if stationary else (batch_index // dwell) % regimes


def class_probabilities(regime: int, experts: int, hot_permille: int) -> torch.Tensor:
    """Mass of each class in a regime: classes r and r + 1 (modulo E) share `hot_permille`/1000, the others share the rest."""
    probs = torch.full((experts,), (1000 - hot_permille) / 1000 / (experts - 2))
    probs[regime % experts] = hot_permille / 2000
    probs[(regime + 1) % experts] = hot_permille / 2000
    return probs


def sample_batch(gen: torch.Generator, batch_index: int, cfg: dict, stationary: bool = False):
    """(classes (B,), features (B, E)) of batch `batch_index`: classes drawn from the regime's mass, features a one-hot plus Gaussian noise."""
    regime = regime_of(batch_index, cfg["regimes"], cfg["dwell"], stationary)
    probs = class_probabilities(regime, cfg["experts"], cfg["hot_permille"])
    classes = torch.multinomial(probs, cfg["tokens"], replacement=True, generator=gen)
    noise = torch.randn(cfg["tokens"], cfg["experts"], generator=gen) * (cfg["noise_milli"] / 1000)
    return classes, torch.nn.functional.one_hot(classes, cfg["experts"]).float() + noise


def competent(classes: torch.Tensor, choices: torch.Tensor, experts: int) -> torch.Tensor:
    """True where the chosen expert is competent for the token's class."""
    return (choices == classes) | (choices == (classes + 1) % experts)


def serve(classes: torch.Tensor, choices: torch.Tensor, experts: int, capacity: int) -> torch.Tensor:
    """Verified tokens: competent expert AND within its capacity, counting arrivals in batch order."""
    rank = torch.zeros_like(choices)
    for e in range(experts):
        mask = choices == e
        rank[mask] = torch.arange(int(mask.sum()))
    return competent(classes, choices, experts) & (rank < capacity)


def serve_reference(classes: list, choices: list, experts: int, capacity: int) -> list[bool]:
    """`serve` by a plain loop (the harness's audit)."""
    used = [0] * experts
    out = []
    for c, e in zip(classes, choices):
        ok = e in (c, (c + 1) % experts) and used[e] < capacity
        used[e] += 1
        out.append(ok)
    return out


def oracle_reward(classes: torch.Tensor, experts: int, capacity: int) -> int:
    """The most tokens any assignment can get verified: a maximum flow (Edmonds-Karp on a graph of 2E + 2 nodes) from the source
    through the classes (capacity = their token count) to the experts competent for them to the sink (capacity = the expert's)."""
    demand = torch.bincount(classes, minlength=experts).tolist()
    n = 2 * experts + 2
    source, sink = 2 * experts, 2 * experts + 1
    cap = [[0] * n for _ in range(n)]
    for c in range(experts):
        cap[source][c] = demand[c]
        for k in (0, 1):
            cap[c][experts + (c + k) % experts] = max(demand[c], 1)
        cap[experts + c][sink] = capacity
    total = 0
    while True:
        parent = [-1] * n
        parent[source] = source
        queue = [source]
        for u in queue:
            for v in range(n):
                if parent[v] < 0 and cap[u][v] > 0:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] < 0:
            return total
        bottleneck, v = float("inf"), sink
        while v != source:
            bottleneck = min(bottleneck, cap[parent[v]][v])
            v = parent[v]
        v = sink
        while v != source:
            cap[parent[v]][v] -= bottleneck
            cap[v][parent[v]] += bottleneck
            v = parent[v]
        total += bottleneck
