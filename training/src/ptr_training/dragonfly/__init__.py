"""Dragonfly: independently trained, generation-bound pod addresses (advisory routing).

Pure Python (no torch). A pod's address is a vector `z` and a bias over a frozen text encoder; the
Rust side (`ptr-router` `learned.rs`) loads only activated ones. See
`docs/architecture/43-pod-linking.md` and ADR-0023.
"""

from .encoder import Encoder, HashedBagOfWordsEncoder, SentenceTransformerEncoder, normalized
from .export import artifact_digest, artifact_stem, export_address, verify_artifact
from .gate import (
    ALIAS_THRESHOLD_PERMILLE,
    NEGATIVE_REJECT_THRESHOLD_PERMILLE,
    GateResult,
    evaluate_gate,
    score,
)
from .identity import PodIdentityDescriptor
from .train import PodTrainingSpec, TrainConfig, TrainedAddress, train_pod_address

__all__ = [
    "ALIAS_THRESHOLD_PERMILLE",
    "NEGATIVE_REJECT_THRESHOLD_PERMILLE",
    "Encoder",
    "GateResult",
    "HashedBagOfWordsEncoder",
    "PodIdentityDescriptor",
    "PodTrainingSpec",
    "SentenceTransformerEncoder",
    "TrainConfig",
    "TrainedAddress",
    "artifact_digest",
    "artifact_stem",
    "evaluate_gate",
    "export_address",
    "normalized",
    "score",
    "train_pod_address",
    "verify_artifact",
]
