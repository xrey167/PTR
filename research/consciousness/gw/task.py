"""C001 task (v2): query-conditioned integration across a variable number of specialist modules.

N sensory modules each see one noisy observation (type one-hot, value one-hot, cue flag, query
one-hot). One module (the cue) carries a query type q instead of a (type, value) pair; exactly
two other modules have type q; the answer is the sum of their two values (a class in 0..2B-2).
A single module's value does not determine the answer, and which modules matter is known only
after the cue's content has reached the others.

Nothing in an observation depends on the module's position, and the networks share their weights
across modules, so the number of modules is free at test time: training uses N_train modules and
the out-of-distribution evaluations use more modules (more distractors), which is a real change
of the computation's input size, unlike a held-out pair of positions.

Everything is a pure function of a torch Generator.
"""

from __future__ import annotations

import torch


def feature_dim(n_types: int, n_values: int) -> int:
    # type one-hot, value one-hot, cue flag, query one-hot
    return n_types + n_values + 1 + n_types


def sample(
    gen: torch.Generator,
    n: int,
    n_modules: int,
    n_types: int,
    n_values: int,
    noise: float,
) -> dict[str, torch.Tensor]:
    """A batch: x [n, n_modules, d_in], y [n], plus the latent structure used by the probes."""
    rows = torch.arange(n)
    # Three distinct positions per row: the two type-q modules and the cue.
    order = torch.rand(n, n_modules, generator=gen).argsort(dim=1)
    pi, pj, cue = order[:, 0], order[:, 1], order[:, 2]
    match = torch.zeros(n, n_modules, dtype=torch.bool)
    match[rows, pi] = True
    match[rows, pj] = True
    q = torch.randint(n_types, (n,), generator=gen)
    # Types of the other modules: uniform over the n_types-1 types that are not q.
    other = torch.randint(n_types - 1, (n, n_modules), generator=gen)
    other = other + (other >= q[:, None]).long()
    types = torch.where(match, q[:, None].expand(n, n_modules), other)
    values = torch.randint(n_values, (n, n_modules), generator=gen)
    y = values[rows, pi] + values[rows, pj]
    d_in = feature_dim(n_types, n_values)
    x = torch.zeros(n, n_modules, d_in)
    is_cue = torch.zeros(n, n_modules, dtype=torch.bool)
    is_cue[rows, cue] = True
    plain = ~is_cue
    x[..., :n_types] = torch.nn.functional.one_hot(types, n_types).float() * plain[..., None]
    x[..., n_types : n_types + n_values] = torch.nn.functional.one_hot(values, n_values).float() * plain[..., None]
    x[..., n_types + n_values] = is_cue.float()
    x[..., n_types + n_values + 1 :] = torch.nn.functional.one_hot(q, n_types).float()[:, None, :] * is_cue[..., None]
    x = x + noise * torch.randn(x.shape, generator=gen)
    return {"x": x, "y": y, "match": match, "cue": cue, "q": q, "pi": pi, "pj": pj}


def set_cue_strength(batch: dict, alpha: float, n_types: int, n_values: int, gen: torch.Generator) -> torch.Tensor:
    """x with the cue module's query one-hot replaced by alpha * onehot(q) + (1 - alpha) * onehot(q')
    for a random q' != q (the ignition probe)."""
    x = batch["x"].clone()
    n = x.shape[0]
    rows = torch.arange(n)
    q = batch["q"]
    other = torch.randint(n_types - 1, (n,), generator=gen)
    other = other + (other >= q).long()
    mix = alpha * torch.nn.functional.one_hot(q, n_types).float() + (1 - alpha) * torch.nn.functional.one_hot(other, n_types).float()
    base = n_types + n_values + 1
    x[rows, batch["cue"], base:] = mix
    return x
