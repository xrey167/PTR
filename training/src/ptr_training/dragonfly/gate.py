"""Scoring and the activation gate of a Dragonfly address.

`score` is the formula `ptr-router`'s `LearnedPodRouter` evaluates, in the same order and in
float64, on parameters already rounded to float32 (see `round_f32`). The gate decides whether a
trained address may be loaded at all: every trusted alias must score at least 0.8 and at least
90 % of the negative examples must be rejected; otherwise the status is `not-activated`.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence

ALIAS_THRESHOLD_PERMILLE = 800
NEGATIVE_REJECT_THRESHOLD_PERMILLE = 900
COSINE_WEIGHT = 12.0
METADATA_WEIGHT = 3.0

STATUS_ACTIVATED = "activated"
STATUS_NOT_ACTIVATED = "not-activated"


def round_f32(value: float) -> float:
    """The float32 nearest to `value`, as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def cosine(query: Sequence[float], z: Sequence[float]) -> float:
    dot = sum(q * v for q, v in zip(query, z))
    q_norm = math.sqrt(sum(q * q for q in query))
    z_norm = math.sqrt(sum(v * v for v in z))
    if q_norm == 0.0 or z_norm == 0.0:
        raise ValueError("cosine of a zero vector")
    return max(-1.0, min(1.0, dot / (q_norm * z_norm)))


def logit(
    query: Sequence[float],
    z: Sequence[float],
    bias: float,
    *,
    cosine_weight: float = COSINE_WEIGHT,
    metadata_weight: float = METADATA_WEIGHT,
) -> float:
    """`cosine_weight * cos(q, z) + metadata_weight * 1 - bias`: metadata always matches."""
    return cosine_weight * cosine(query, z) + metadata_weight - bias


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def score(
    query: Sequence[float],
    z: Sequence[float],
    bias: float,
    *,
    cosine_weight: float = COSINE_WEIGHT,
    metadata_weight: float = METADATA_WEIGHT,
) -> float:
    return sigmoid(
        logit(query, z, bias, cosine_weight=cosine_weight, metadata_weight=metadata_weight)
    )


@dataclass(frozen=True)
class GateResult:
    alias_threshold_permille: int
    alias_min_permille: int
    negative_threshold_permille: int
    negative_reject_permille: int
    alias_count: int
    negative_count: int

    @property
    def activated(self) -> bool:
        return (
            self.alias_count > 0
            and self.negative_count > 0
            and self.alias_min_permille >= self.alias_threshold_permille
            and self.negative_reject_permille >= self.negative_threshold_permille
        )

    @property
    def status(self) -> str:
        return STATUS_ACTIVATED if self.activated else STATUS_NOT_ACTIVATED


def evaluate_gate(
    *,
    alias_vectors: Sequence[Sequence[float]],
    negative_vectors: Sequence[Sequence[float]],
    z: Sequence[float],
    bias: float,
    cosine_weight: float = COSINE_WEIGHT,
    metadata_weight: float = METADATA_WEIGHT,
) -> GateResult:
    """Measure the gate on already rounded parameters.

    Permille values are floored, so a measured value never exceeds the real one. A negative is
    rejected when its logit is below zero (a score below 0.5, the router's abstention line).
    """

    def kwargs():
        return {"cosine_weight": cosine_weight, "metadata_weight": metadata_weight}

    alias_scores = [score(v, z, bias, **kwargs()) for v in alias_vectors]
    rejected = sum(1 for v in negative_vectors if logit(v, z, bias, **kwargs()) < 0.0)
    return GateResult(
        alias_threshold_permille=ALIAS_THRESHOLD_PERMILLE,
        alias_min_permille=math.floor(min(alias_scores) * 1000) if alias_scores else 0,
        negative_threshold_permille=NEGATIVE_REJECT_THRESHOLD_PERMILLE,
        negative_reject_permille=(
            math.floor(rejected * 1000 / len(negative_vectors)) if negative_vectors else 0
        ),
        alias_count=len(alias_scores),
        negative_count=len(negative_vectors),
    )
