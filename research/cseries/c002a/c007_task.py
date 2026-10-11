"""C007 world: a classification task with a distribution shift, and the verifier.

K classes are Gaussian clouds in d dimensions around random class means. The mixture-of-experts classifier is trained on in-distribution
inputs; the self-model and the probes are trained on a separate calibration set of in-distribution inputs with the VERIFIER's outcome (the
prediction is wrong or right against the true class); the primary evaluation is on a SHIFTED split (inputs displaced along a fixed direction and
noisier), where the model errs more and its traces look different. The null world draws the class labels independently of the inputs, so no
trace can carry information about an error. Everything is a pure function of a torch Generator."""

from __future__ import annotations

import torch


def class_means(gen: torch.Generator, cfg: dict) -> torch.Tensor:
    means = torch.randn(cfg["classes"], cfg["dim"], generator=gen)
    return cfg["separation_milli"] / 1000 * means / means.norm(dim=1, keepdim=True)


def shift_direction(gen: torch.Generator, cfg: dict) -> torch.Tensor:
    direction = torch.randn(cfg["dim"], generator=gen)
    return direction / direction.norm()


def sample(gen: torch.Generator, means: torch.Tensor, n: int, cfg: dict, shifted: bool = False, direction: torch.Tensor | None = None, null: bool = False):
    """(inputs, true classes). Shifted: displaced by `shift_milli`/1000 along `direction` and with the noise scaled by `shift_noise_milli`/1000. Null: the
    class labels are drawn independently of the inputs."""
    labels = torch.randint(0, cfg["classes"], (n,), generator=gen)
    scale = cfg["noise_milli"] / 1000 * (cfg["shift_noise_milli"] / 1000 if shifted else 1.0)
    x = means[labels] + scale * torch.randn(n, cfg["dim"], generator=gen)
    if shifted:
        x = x + cfg["shift_milli"] / 1000 * direction
    if null:
        labels = torch.randint(0, cfg["classes"], (n,), generator=gen)
    return x, labels


def outcomes(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The verifier: 1 where the model's prediction is wrong."""
    return (logits.argmax(dim=-1) != labels).float()


def outcomes_independent(logits: torch.Tensor, labels: torch.Tensor) -> list[int]:
    """`outcomes` by a plain loop over the rows."""
    out = []
    for row, label in zip(logits.tolist(), labels.tolist()):
        best = max(range(len(row)), key=lambda k: (row[k], -k))
        out.append(int(best != label))
    return out


def permute_outcomes(y: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """The verifier outcomes shuffled across the examples (the label-permutation control)."""
    return y[torch.randperm(y.shape[0], generator=gen)]
