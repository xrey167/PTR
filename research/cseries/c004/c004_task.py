"""C004 task: a modular counter (state tracking in the cyclic group Z_6), length-extrapolation split.

A sequence of T steps; step t is a symbol x_t in {0..5} (no position information); the answer is the running sum modulo 6,
s_T, with s_0 = 0 and s_t = (s_{t-1} + x_t) mod 6. The state is one of six values, it must be carried over every step, and the
answer depends on every step (replacing any one step by a different symbol changes the sum modulo 6) and not on the last step
alone (the sum is uniform given the last step). The number of steps is free at test time: training uses 2..6 steps (config),
evaluation the longest trained length (in distribution) and longer ones.

The task is also posed online: `running_labels` gives the counter after every step, the supervision of every arm (each position of
a causal read-out is trained to report the current state; the answer to a sequence is the read-out at its last step).
`window_labels` is the null world: the answer is the sum of the last two steps, so no state integrated over the sequence is
needed. `labels_independent` and `window_labels_independent` are second computations (plain Python sums) used by the harness to
audit the labels. Everything is a pure function of a torch Generator."""

from __future__ import annotations

import torch

N_SYMBOLS = 6


def sample(gen: torch.Generator, n: int, length: int) -> torch.Tensor:
    """n sequences of `length` steps, uniform over the six symbols."""
    return torch.randint(0, N_SYMBOLS, (n, length), generator=gen)


def running_labels(tokens: torch.Tensor) -> torch.Tensor:
    """The counter after every step, (n, T)."""
    return torch.cumsum(tokens, dim=1) % N_SYMBOLS


def labels(tokens: torch.Tensor) -> torch.Tensor:
    """The answer to each sequence: the counter after its last step."""
    return running_labels(tokens)[:, -1]


def window_labels(tokens: torch.Tensor) -> torch.Tensor:
    """The null world: the answer is the sum of the last two steps modulo 6. It needs a state of one step, not a state integrated
    over the sequence, so it does not depend on the sequence length (the reset-state arm cannot solve it: it sees one step)."""
    return (tokens[:, -1] + tokens[:, -2]) % N_SYMBOLS


def window_running_labels(tokens: torch.Tensor) -> torch.Tensor:
    """The null world after every step: the sum of this step and the one before it (the first step alone), modulo 6."""
    previous = torch.cat([torch.zeros_like(tokens[:, :1]), tokens[:, :-1]], dim=1)
    return (tokens + previous) % N_SYMBOLS


def labels_independent(tokens: torch.Tensor) -> list[int]:
    """`labels` by a plain Python loop."""
    out = []
    for row in tokens.tolist():
        state = 0
        for symbol in row:
            state = (state + symbol) % N_SYMBOLS
        out.append(state)
    return out


def window_labels_independent(tokens: torch.Tensor) -> list[int]:
    """`window_labels` by plain Python."""
    return [(row[-1] + row[-2]) % N_SYMBOLS for row in tokens.tolist()]


def corrupt(tokens: torch.Tensor, gen: torch.Generator, rate_permille: int) -> torch.Tensor:
    """Replace, in a fraction `rate_permille`/1000 of the sequences, one uniformly chosen step by a different symbol. The answer of a
    corrupted sequence differs from the original's, so a model that computes the answer of what it reads is wrong on exactly these
    sequences."""
    n, length = tokens.shape
    hit = torch.randint(0, 1000, (n,), generator=gen) < rate_permille
    where = torch.randint(0, length, (n,), generator=gen)
    shift = torch.randint(1, N_SYMBOLS, (n,), generator=gen)
    out = tokens.clone()
    rows = torch.arange(n)[hit]
    out[rows, where[hit]] = (tokens[rows, where[hit]] + shift[hit]) % N_SYMBOLS
    return out


def permute_labels(y: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """The labels shuffled across the sequences (the label-permutation control): the marginal is kept, the association with the input
    is destroyed. Works on (n,) answers and on (n, T) trajectories (whole rows move)."""
    return y[torch.randperm(y.shape[0], generator=gen)]
