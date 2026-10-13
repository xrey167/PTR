"""Train one independent Dragonfly address (`z`, bias) per pod over a frozen encoder.

Per pod: a vector `z` and a bias, fitted with AdamW on a class-balanced binary cross-entropy over
explicit positive and negative address questions. Nothing is shared between pods, so revoking one
pod's address leaves every other address valid. The metadata counterfactual: every training
question is scored as if its hard metadata (project, capability, input type) matched, so the
metadata term is the constant `metadata_weight` and cannot solve the task; only the question
vector is learned. At run time the registry has already enforced that metadata, so the same
constant applies (see `ptr-router` `learned.rs`).

Pure Python on purpose: no framework, no download, bit-for-bit repeatable on any machine.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .encoder import Encoder, HashedBagOfWordsEncoder, l2_normalize, normalized
from .gate import (
    COSINE_WEIGHT,
    METADATA_WEIGHT,
    GateResult,
    cosine,
    evaluate_gate,
    logit,
    round_f32,
    sigmoid,
)
from .identity import PodIdentityDescriptor


@dataclass(frozen=True)
class PodTrainingSpec:
    """Trusted, independently authored training inputs of one pod generation."""

    pod_id: str
    descriptor: PodIdentityDescriptor
    generation: int
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    #: Similarly named other entities the pod must be separated from.
    negative_aliases: tuple[str, ...] = ()
    #: Text the vector starts from; the mean of the positives when absent.
    description: str | None = None


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 300
    learning_rate: float = 0.04
    weight_decay: float = 0.01
    cosine_weight: float = COSINE_WEIGHT
    metadata_weight: float = METADATA_WEIGHT


@dataclass(frozen=True)
class TrainedAddress:
    spec: PodTrainingSpec
    encoder_id: str
    encoder_revision: str
    dims: int
    #: Parameters already rounded to float32: what is exported and what the gate measured.
    z: tuple[float, ...]
    bias: float
    cosine_weight: float
    metadata_weight: float
    gate: GateResult
    steps: int
    loss_before: float
    loss_after: float

    @property
    def activated(self) -> bool:
        return self.gate.activated


def expand_alias_questions(
    aliases: Sequence[str], positives: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Compile trusted aliases into training questions.

    Returns the sorted aliases and the positive questions: the originals, each original that names
    an alias repeated with every other alias, and the bare aliases with their normalised forms.
    Only trusted aliases are expanded; a soft, unconfirmed alias never reaches training.
    """
    trusted = sorted(set(aliases))
    if not trusted or not all(a.strip() for a in trusted):
        raise ValueError("nonempty trusted aliases are required")
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(trusted, key=len, reverse=True)) + r")(?!\w)",
        re.IGNORECASE,
    )
    expanded = set(positives)
    for question in positives:
        if pattern.search(question):
            for alias in trusted:
                expanded.add(pattern.sub(lambda _match, alias=alias: alias, question))
    for alias in trusted:
        expanded.add(alias)
        expanded.add(normalized(alias))
    return trusted, sorted(expanded)


def _balanced_loss(logits: Sequence[float], labels: Sequence[float]) -> float:
    def bce(value: float, label: float) -> float:
        return max(value, 0.0) - value * label + math.log1p(math.exp(-abs(value)))

    positive = [bce(v, y) for v, y in zip(logits, labels) if y > 0.5]
    negative = [bce(v, y) for v, y in zip(logits, labels) if y <= 0.5]
    return (sum(positive) / len(positive) + sum(negative) / len(negative)) / 2


def _check_inputs(spec: PodTrainingSpec, positives: Sequence[str], negatives: Sequence[str]) -> None:
    if spec.generation < 1:
        raise ValueError("generation must be at least 1")
    if not spec.pod_id:
        raise ValueError("pod_id is required")
    spec.descriptor.validate()
    if not spec.positives or not negatives:
        raise ValueError("positive and negative address examples are required")
    for question in [*spec.positives, *spec.negatives, *spec.negative_aliases]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("address questions must be nonempty text")
    if set(spec.positives) & set(spec.negatives):
        raise ValueError("conflicting address labels")
    if {normalized(q) for q in positives} & {normalized(q) for q in negatives}:
        raise ValueError("a trusted alias conflicts with a negative training example")


def train_pod_address(
    spec: PodTrainingSpec, encoder: Encoder, config: TrainConfig = TrainConfig()
) -> TrainedAddress:
    """Fit and gate the address of one pod generation. Deterministic for a fixed encoder."""
    if config.steps < 1:
        raise ValueError("steps must be positive")
    aliases, positives = expand_alias_questions(spec.descriptor.aliases, spec.positives)
    negatives = sorted(set(spec.negatives) | set(spec.negative_aliases))
    _check_inputs(spec, positives, negatives)

    questions = [*positives, *negatives]
    vectors = [l2_normalize(encoder.encode(q)) for q in questions]
    labels = [1.0] * len(positives) + [0.0] * len(negatives)
    weights = [
        1.0 / (2 * len(positives)) if label > 0.5 else 1.0 / (2 * len(negatives)) for label in labels
    ]
    dims = len(vectors[0])
    if dims != encoder.dims:
        raise ValueError("encoder dims disagree with its embeddings")

    if spec.description is not None:
        z = l2_normalize(encoder.encode(spec.description))
    else:
        z = l2_normalize([sum(v[d] for v, y in zip(vectors, labels) if y > 0.5) for d in range(dims)])
    bias = 0.0

    def logits(current_z: Sequence[float], current_bias: float) -> list[float]:
        return [
            logit(
                q,
                current_z,
                current_bias,
                cosine_weight=config.cosine_weight,
                metadata_weight=config.metadata_weight,
            )
            for q in vectors
        ]

    loss_before = _balanced_loss(logits(z, bias), labels)
    first = [0.0] * (dims + 1)
    second = [0.0] * (dims + 1)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, config.steps + 1):
        z_norm = math.sqrt(sum(v * v for v in z))
        z_hat = [v / z_norm for v in z]
        grad = [0.0] * (dims + 1)
        for q, label, weight in zip(vectors, labels, weights):
            cos = cosine(q, z)
            value = config.cosine_weight * cos + config.metadata_weight - bias
            delta = (sigmoid(value) - label) * weight
            scale = delta * config.cosine_weight / z_norm
            for d in range(dims):
                grad[d] += scale * (q[d] - cos * z_hat[d])
            grad[dims] -= delta
        for d in range(dims + 1):
            first[d] = beta1 * first[d] + (1 - beta1) * grad[d]
            second[d] = beta2 * second[d] + (1 - beta2) * grad[d] * grad[d]
            m_hat = first[d] / (1 - beta1**step)
            v_hat = second[d] / (1 - beta2**step)
            update = config.learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
            if d < dims:
                z[d] = z[d] * (1 - config.learning_rate * config.weight_decay) - update
            else:
                bias -= update

    z_f32 = tuple(round_f32(v) for v in z)
    bias_f32 = round_f32(bias)
    cosine_weight = round_f32(config.cosine_weight)
    metadata_weight = round_f32(config.metadata_weight)
    loss_after = _balanced_loss(logits(z_f32, bias_f32), labels)
    gate = evaluate_gate(
        alias_vectors=[l2_normalize(encoder.encode(alias)) for alias in aliases],
        negative_vectors=[vectors[len(positives) + i] for i in range(len(negatives))],
        z=z_f32,
        bias=bias_f32,
        cosine_weight=cosine_weight,
        metadata_weight=metadata_weight,
    )
    return TrainedAddress(
        spec=spec,
        encoder_id=encoder.encoder_id,
        encoder_revision=encoder.encoder_revision,
        dims=dims,
        z=z_f32,
        bias=bias_f32,
        cosine_weight=cosine_weight,
        metadata_weight=metadata_weight,
        gate=gate,
        steps=config.steps,
        loss_before=loss_before,
        loss_after=loss_after,
    )


def _spec_from_json(data: dict) -> PodTrainingSpec:
    descriptor = PodIdentityDescriptor.from_json(data["descriptor"])
    return PodTrainingSpec(
        pod_id=data["pod_id"],
        descriptor=descriptor,
        generation=data["generation"],
        positives=tuple(data["positives"]),
        negatives=tuple(data["negatives"]),
        negative_aliases=tuple(data.get("negative_aliases", ())),
        description=data.get("description"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """`ptr-dragonfly-train SPEC.json OUT_DIR`: train and export every pod of a spec file."""
    from .export import export_address

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("spec", type=Path, help="JSON file with a `pods` list of training specs")
    parser.add_argument("out", type=Path, help="directory the artifacts are written to")
    parser.add_argument("--encoder", default="hashed-bow", help="hashed-bow or sentence-transformers")
    parser.add_argument("--dims", type=int, default=64, help="dims of the hashed-bow encoder")
    parser.add_argument("--model", help="sentence-transformers model name")
    parser.add_argument("--revision", help="pinned sentence-transformers revision")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    arguments = parser.parse_args(argv)

    if arguments.encoder == "hashed-bow":
        encoder: Encoder = HashedBagOfWordsEncoder(arguments.dims)
    elif arguments.encoder == "sentence-transformers":
        from .encoder import SentenceTransformerEncoder

        if not arguments.model or not arguments.revision:
            parser.error("--model and --revision are required for sentence-transformers")
        encoder = SentenceTransformerEncoder(arguments.model, arguments.revision)
    else:
        parser.error(f"unknown encoder {arguments.encoder}")

    document = json.loads(arguments.spec.read_text(encoding="utf-8"))
    exit_code = 0
    for pod in document["pods"]:
        trained = train_pod_address(
            _spec_from_json(pod), encoder, TrainConfig(steps=arguments.steps)
        )
        json_path, _ = export_address(trained, arguments.out)
        print(f"{trained.spec.pod_id}: {trained.gate.status} -> {json_path.name}")
        if not trained.activated:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
